using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression coverage for <see cref="InferenceEngine"/>'s prefix-cache reuse path
/// (issues #20 and #21). Three scenarios:
///
/// <list type="number">
///   <item>A forward pass that does NOT support partial rewind AND has no snapshot
///         facility (e.g. unknown future backend) must not be asked to
///         <see cref="IForwardPass.TruncateTo"/> to an intermediate length; the
///         engine must fall back to a full reset (issue #20).</item>
///   <item>A forward pass that DOES support partial rewind keeps the existing
///         page-aligned prefix-cache fast path — guards against accidentally
///         disabling it for every backend.</item>
///   <item>A forward pass that does NOT support partial rewind but DOES expose
///         <see cref="IForwardPass.CaptureSnapshot"/> /
///         <see cref="IForwardPass.SnapshotLength"/> (GDN hybrid; issue #21)
///         can reuse state across chat turns when the new prompt extends the
///         previous full sequence — the engine calls <c>TruncateTo(snapLen)</c>
///         instead of resetting.</item>
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

    /// <summary>
    /// Issue #22: rewind-incompatible passes must surface <see cref="IInferenceEngine.PrefixCacheEnabled"/>
    /// as <c>false</c>, and <see cref="IInferenceEngine.PrefillTokensReused"/> must remain at zero across
    /// turns because the engine is forced down the full-reset branch.
    /// </summary>
    [Fact]
    public async Task PrefixCacheState_OnRewindIncompatiblePass_ReportsDisabledAndZeroReused()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindIncompatibleForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.False(engine.PrefixCacheEnabled);
        Assert.Equal(0, engine.PrefillTokensReused);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Drain(engine.GenerateAsync("turn1", sp));
        await Drain(engine.GenerateAsync("turn2", sp));

        Assert.Equal(0, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #22: rewind-capable passes report <see cref="IInferenceEngine.PrefixCacheEnabled"/> true
    /// and accumulate the matched-prefix length into <see cref="IInferenceEngine.PrefillTokensReused"/>
    /// after a cache hit. The fixture's two-turn prompt shares a 32-token prefix.
    /// </summary>
    [Fact]
    public async Task PrefixCacheState_OnRewindCapablePass_ReportsEnabledAndAccumulatesReused()
    {
        var tokenizer = new MultiTurnTokenizer();
        var fwd = new RewindCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        Assert.True(engine.PrefixCacheEnabled);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };
        await Drain(engine.GenerateAsync("turn1", sp));
        // First turn is a cold start — no prefix to reuse.
        Assert.Equal(0, engine.PrefillTokensReused);

        await Drain(engine.GenerateAsync("turn2", sp));
        // Second turn shares a 32-token prefix with turn 1.
        Assert.Equal(32, engine.PrefillTokensReused);
    }

    /// <summary>
    /// Issue #21: a non-rewindable pass with a snapshot facility (GDN hybrid)
    /// should still reuse cached state across chat-continuation turns by way
    /// of <see cref="IForwardPass.CaptureSnapshot"/> at end-of-decode and
    /// <see cref="IForwardPass.TruncateTo"/> on the snapshot length next turn.
    ///
    /// Turn 1: 32 prompt tokens; after the loop (1 generated token + EOS),
    /// the stub captures a snapshot at the full-sequence length.
    /// Turn 2: 33 prompt tokens whose first 33 tokens are exactly the turn-1
    /// full sequence (32 prompt + 1 generated). The engine must call
    /// TruncateTo(snapLen) — not propagate any throw, not call ResetCache.
    /// </summary>
    [Fact]
    public async Task GenerateAsync_OnSnapshotCapablePass_RestoresSnapshotOnContinuation()
    {
        var tokenizer = new SnapshotMultiTurnTokenizer();
        var fwd = new SnapshotCapableForwardPass();
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 1 };

        await Drain(engine.GenerateAsync("turn1", sp));

        // After turn 1, _prevTokens is the 32-token prompt + the 1 generated token.
        // The stub generated token (id 50) is not in stopIds, so it's appended; the
        // loop then iterates once more and hits MaxNewTokens, exiting cleanly.
        // CaptureSnapshot records the GdnStateCache.Length-equivalent (mock).
        Assert.Equal(33, fwd.LastCapturedSnapshotLength);

        await Drain(engine.GenerateAsync("turn2", sp));

        // Engine should have asked the snapshot-capable pass to restore at length 33.
        Assert.Equal(33, fwd.LastTruncateLength);
        Assert.False(
            fwd.ResetCalledAfter,
            "ResetCache must not run on turn 2 — the snapshot branch already prepared the cache.");
        // Suffix prefill covers only the tokens after the snapshot length.
        Assert.True(fwd.LastPrefillLength > 0);
        Assert.Equal(33, fwd.LastPrefillStartPos);
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
    /// Tokenizer paired with <see cref="SnapshotCapableForwardPass"/>. Generates
    /// a 32-token prefix; for "turn2" extends by 1 sampled-token-slot + 1 fresh
    /// token so the snapshot length (33) is a strict prefix of the new prompt
    /// and at least one suffix token remains.
    /// </summary>
    private sealed class SnapshotMultiTurnTokenizer : ITokenizer
    {
        public const int SampledTokenId = 50;

        public int VocabSize => 200;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;

        public IReadOnlyList<int> Encode(string text)
        {
            // turn1: 32-token prefix (matches the GDN-hybrid 'apply chat template' result).
            var prefix = Enumerable.Range(0, 32).ToArray();
            if (text == "turn2")
            {
                // turn2 = [prefix(32) , SampledTokenId , fresh-token(110)]
                // The engine's full-sequence for turn1 is [prefix(32) , SampledTokenId],
                // length 33. turn2's first 33 tokens must equal that; the 34th token is
                // the fresh user-message extension that drives a non-empty prefill.
                return prefix.Concat([SampledTokenId, 110]).ToArray();
            }
            return prefix;
        }

        public string Decode(IEnumerable<int> tokens) => string.Empty;
        public byte[] DecodeBytes(int token) => [];
    }

    /// <summary>
    /// Models a GDN-hybrid contract: <see cref="SupportsPartialRewind"/> is false
    /// (recurrent state is destructively updated), but the pass exposes
    /// <see cref="CaptureSnapshot"/> / <see cref="SnapshotLength"/> so the engine
    /// can reuse state across chat turns whose prompt extends the prior transcript.
    /// </summary>
    private sealed class SnapshotCapableForwardPass : IForwardPass
    {
        private readonly float[] _logits = new float[200];
        private int _length;

        public int LastCapturedSnapshotLength { get; private set; } = -1;
        public int LastTruncateLength { get; private set; } = -1;
        public int LastPrefillLength { get; private set; } = -1;
        public int LastPrefillStartPos { get; private set; } = -1;
        public bool ResetCalledAfter { get; private set; }

        public bool SupportsPartialRewind => false;
        public int VocabSize => 200;
        public int MaxSeqLen => 4096;

        public int SnapshotLength { get; private set; } = -1;

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            _length = position + 1;
            return SampledLogits();
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            LastPrefillLength = tokens.Count;
            LastPrefillStartPos = startPos;
            _length = startPos + tokens.Count;
            return SampledLogits();
        }

        public void TruncateTo(int length)
        {
            // Mirror GDN semantics: accept 0, current, and SnapshotLength.
            // Record every call so the test can verify the engine took the
            // snapshot branch even when current == SnapshotLength (the trivial
            // no-op overlap that the real GDN pass would also accept).
            LastTruncateLength = length;
            if (length == 0) { _length = 0; SnapshotLength = -1; return; }
            if (length == _length) return;
            if (length == SnapshotLength && SnapshotLength >= 0)
            {
                _length = length;
                return;
            }
            throw new NotSupportedException(
                $"SnapshotCapableForwardPass.TruncateTo({length}): only 0, current ({_length}), or SnapshotLength ({SnapshotLength}) supported.");
        }

        public void ResetCache()
        {
            // After a successful TruncateTo(snapLen), the engine must NOT call ResetCache.
            // Record only when reset happens after the snapshot was captured — that's
            // the actual regression we're guarding against.
            if (SnapshotLength >= 0)
                ResetCalledAfter = true;
            _length = 0;
            SnapshotLength = -1;
        }

        public void CaptureSnapshot()
        {
            LastCapturedSnapshotLength = _length;
            SnapshotLength = _length;
        }

        public void Dispose() { }

        private ReadOnlySpan<float> SampledLogits()
        {
            // Force a non-stop token so the decode loop runs once and the snapshot
            // is captured at "prompt + 1 generated token" length, mirroring real
            // generation. EOS is the only default stop id.
            Array.Clear(_logits);
            _logits[SnapshotMultiTurnTokenizer.SampledTokenId] = 1.0f;
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
