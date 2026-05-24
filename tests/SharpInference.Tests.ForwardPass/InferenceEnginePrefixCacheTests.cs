using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression coverage for <see cref="InferenceEngine"/>'s prefix-cache reuse path
/// (issue #20). Two scenarios:
///
/// <list type="number">
///   <item>A forward pass that does NOT support partial rewind (Qwen3.6-style
///         GDN hybrid) must not be asked to <see cref="IForwardPass.TruncateTo"/>
///         to an intermediate length; the engine must fall back to a full reset.</item>
///   <item>A forward pass that DOES support partial rewind keeps the existing
///         prefix-cache fast path — guards against accidentally disabling it for
///         every backend.</item>
/// </list>
/// </summary>
public sealed class InferenceEnginePrefixCacheTests
{
    private const int Eos = 99;

    /// <summary>
    /// Drives two sequential <c>GenerateAsync</c> calls with prompts that share a
    /// 32-token page-aligned prefix (2 × <see cref="PagedKvCache.PageSize"/>). On a
    /// rewind-incompatible pass the engine must not propagate the
    /// <see cref="NotSupportedException"/> from <see cref="IForwardPass.TruncateTo"/>.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnRewindIncompatiblePass_DoesNotPropagateTruncateThrow()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindIncompatibleForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        // Turn 1: 32 prompt tokens. After this _prevTokens has 32 entries.
        await Drain(engine.GenerateAsync("turn1", sp));

        // Turn 2: 48 prompt tokens, first 32 shared with turn 1. FindCacheablePrefix
        // returns 32; without the fix the engine calls TruncateTo(32) which throws.
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.False(
            fwd.PartialTruncateAttempted,
            "InferenceEngine called TruncateTo with a partial length on a rewind-incompatible pass.");
        // ResetCache is the cold-start branch the engine takes whenever prefixLen == 0. On a
        // rewind-incompatible pass the gate forces prefixLen to 0 even when prefixes match,
        // so observing ResetCache here confirms the gate fired (no partial TruncateTo was tried).
        Assert.True(fwd.ResetCacheCalled, "Engine should reach the ResetCache branch when the prefix gate forces prefixLen to 0.");
    }

    /// <summary>
    /// Same two-turn pattern on a rewind-capable pass — the engine should call
    /// <see cref="IForwardPass.TruncateTo"/> with the matched prefix length and
    /// only re-prefill the new suffix.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnRewindCapablePass_ReusesPrefixOnSecondCall()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("turn1", sp));
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Equal(32, fwd.LastTruncateLength);
        // Second prefill should cover only the 16-token suffix, not all 48 tokens.
        Assert.Equal(16, fwd.LastPrefillLength);
        Assert.Equal(32, fwd.LastPrefillStartPos);
    }

    private static async Task Drain(IAsyncEnumerable<string> stream)
    {
        await foreach (var _ in stream) { }
    }

    // ── Mocks ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Hand-rolled tokenizer that returns two distinct token sequences sharing a
    /// 32-token (two-page) prefix:
    ///   "turn1" → [0..32)
    ///   "turn2" → [0..32) followed by [100..116)
    /// </summary>
    private sealed class MultiTurnTokenizer : ITokenizer
    {
        public int VocabSize => 200;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text)
        {
            // 32-token shared prefix for any prompt; "turn2" appends 16 fresh tokens.
            var prefix = Enumerable.Range(0, 32).ToArray();
            if (text == "turn2")
                return prefix.Concat(Enumerable.Range(100, 16)).ToArray();
            return prefix;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Models the Qwen3.6 GDN hybrid contract: <see cref="TruncateTo"/> only accepts
    /// length 0 or the current length. Any other call records the attempt and throws.
    /// </summary>
    private sealed class RewindIncompatibleForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];
        private int _length;

        public bool PartialTruncateAttempted { get; private set; }
        public bool ResetCacheCalled { get; private set; }

        public bool SupportsPartialRewind => false;

        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _length = position + 1;
            return EosLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            _length = startPos + tokens.Count;
            return EosLogits();
        }

        public void TruncateTo(int length)
        {
            if (length == _length || length == 0)
            {
                if (length == 0) _length = 0;
                return;
            }
            PartialTruncateAttempted = true;
            throw new NotSupportedException(
                $"RewindIncompatibleForwardPass.TruncateTo({length}): only length == 0 or current ({_length}) is supported.");
        }

        public void ResetCache()
        {
            ResetCacheCalled = true;
            _length = 0;
        }

        public void Dispose() { }

        private ReadOnlySpan<float> EosLogits()
        {
            Array.Clear(_logits);
            _logits[Eos] = 1.0f;
            return _logits;
        }
    }

    /// <summary>
    /// Rewind-capable forward pass that records the arguments to the last
    /// <see cref="TruncateTo"/> and <see cref="Prefill"/> calls so the test can
    /// confirm the prefix path was taken.
    /// </summary>
    private sealed class RewindCapableForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];

        // Sentinel -1 means "never called" — distinguishes a missing call from a TruncateTo(0).
        public int LastTruncateLength { get; private set; } = -1;
        public int LastPrefillLength { get; private set; } = -1;
        public int LastPrefillStartPos { get; private set; } = -1;

        public bool SupportsPartialRewind => true;

        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public ReadOnlySpan<float> Forward(int token, int position) => EosLogits();

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPrefillLength = tokens.Count;
            LastPrefillStartPos = startPos;
            return EosLogits();
        }

        public void TruncateTo(int length) => LastTruncateLength = length;
        public void ResetCache() { }
        public void Dispose() { }

        private ReadOnlySpan<float> EosLogits()
        {
            Array.Clear(_logits);
            _logits[Eos] = 1.0f;
            return _logits;
        }
    }
}
