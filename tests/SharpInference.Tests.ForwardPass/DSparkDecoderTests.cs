using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Pure-logic tests for <see cref="DSparkDecoder"/> (docs/dspark-plan.md, PR #413)
/// over a scripted chain target + scripted draft, mirroring the
/// <c>ScriptedMtpPass</c> pattern: the fakes hard-assert the decoder's
/// cache-length, tap, and context-append contracts on every call, and the
/// tests pin greedy parity, stop handling, confidence trimming, the verify
/// caps, and the env-var resolvers.
/// </summary>
public sealed class DSparkDecoderTests
{
    private const int Vocab = 16;
    private const int EmbDim = 4;
    private const int Block = 3;

    /// <summary>
    /// Deterministic chain target: greedy next token after t is (t+1) % Vocab.
    /// One-hot logits; taps are marker rows (every element == position + 0.5f).
    /// </summary>
    private sealed class TapChainForwardPass : IForwardPass
    {
        public int CacheLen;
        public int TapHighWater;
        public int ForwardCalls;
        public int BatchVerifyCalls;
        public readonly List<int> VerifyLengths = [];
        public bool BatchVerifySupported = true;
        public bool TapsPopulated = true;
        private int _tapLayers;

        public int VocabSize => Vocab;
        public int MaxSeqLen => 512;
        public bool SupportsPartialRewind => true;
        public bool SupportsBatchVerify => BatchVerifySupported;
        public bool SupportsHiddenTaps => true;
        public int HiddenTapDim => _tapLayers * EmbDim;

        public void EnableHiddenTaps(ReadOnlySpan<int> layerIds) => _tapLayers = layerIds.Length;

        public ReadOnlySpan<float> HiddenTapsAt(int position)
        {
            if (!TapsPopulated || position < 0 || position >= TapHighWater) return default;
            var row = new float[HiddenTapDim];
            Array.Fill(row, position + 0.5f);
            return row;
        }

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            Assert.Equal(CacheLen, position);
            CacheLen++;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            ForwardCalls++;
            return Logits((token + 1) % Vocab);
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
        {
            Assert.Equal(CacheLen, startPos);
            CacheLen = startPos + tokens.Count;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            return Logits((tokens[^1] + 1) % Vocab);
        }

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            Assert.True(BatchVerifySupported);
            Assert.Equal(CacheLen, startPos);
            BatchVerifyCalls++;
            VerifyLengths.Add(tokens.Length);
            CacheLen = startPos + tokens.Length;
            if (CacheLen > TapHighWater) TapHighWater = CacheLen;
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
                result[i] = Logits((tokens[i] + 1) % Vocab);
            return result;
        }

        public void TruncateTo(int length)
        {
            Assert.True(length <= CacheLen, $"TruncateTo({length}) would extend cache of {CacheLen}.");
            Assert.True(length >= 0);
            CacheLen = length;
        }

        public void ResetCache() => CacheLen = 0;
        public void Dispose() { }

        public static float[] Logits(int next)
        {
            var l = new float[Vocab];
            l[next] = 1f;
            return l;
        }
    }

    /// <summary>
    /// Scripted draft: proposals come from a queue; the default (empty queue)
    /// proposes the correct chain continuation with confidence 1. Hard-asserts
    /// the AppendContext contiguity + tap-content contract and the
    /// ProposeBlock anchor contract.
    /// </summary>
    private sealed class ScriptedDraft : IDSparkDraft
    {
        public readonly Queue<(int[] Tokens, float[] Conf)> Script = new();
        public int ContextLength { get; private set; }
        public int BlockSize => Block;
        public int VocabSize => Vocab;
        public int MaxContext { get; init; } = int.MaxValue;
        public int TapDim { get; }

        public ScriptedDraft(int tapDim) => TapDim = tapDim;

        public void AppendContext(ReadOnlySpan<float> taps, int startPos, int count)
        {
            Assert.Equal(ContextLength, startPos);
            Assert.Equal(count * TapDim, taps.Length);
            for (int r = 0; r < count; r++)
                for (int d = 0; d < TapDim; d++)
                    Assert.Equal(startPos + r + 0.5f, taps[r * TapDim + d]);
            ContextLength = startPos + count;
        }

        public DSparkProposal ProposeBlock(int anchorToken, int anchorPos)
        {
            Assert.Equal(ContextLength, anchorPos);
            if (Script.Count > 0)
            {
                var (tokens, conf) = Script.Dequeue();
                return new DSparkProposal(tokens, conf);
            }
            var chain = new int[Block];
            for (int j = 0; j < Block; j++) chain[j] = (anchorToken + 1 + j) % Vocab;
            var ones = new float[Block];
            Array.Fill(ones, 1f);
            return new DSparkProposal(chain, ones);
        }

        public void TruncateContext(int length) => ContextLength = Math.Min(ContextLength, length);
        public void ResetContext() => ContextLength = 0;
        public void Dispose() { }
    }

    private static (DSparkDecoder Decoder, TapChainForwardPass Target, ScriptedDraft Draft) Setup(
        int promptLen = 4, int firstToken = 5)
    {
        var target = new TapChainForwardPass();
        ((IForwardPass)target).EnableHiddenTaps([0, 2]);
        var draft = new ScriptedDraft(target.HiddenTapDim);
        var tokens = new int[promptLen];
        var logits = target.Prefill(tokens);
        _ = logits;
        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(promptLen, TapChainForwardPass.Logits(firstToken));
        return (decoder, target, draft);
    }

    [Fact]
    public void Decode_AllCorrectDrafts_EmitsExactChain()
    {
        var (decoder, target, draft) = Setup();

        var emitted = new List<int>();
        decoder.Decode(9, [], emitted.Add);

        Assert.Equal(new[] { 5, 6, 7, 8, 9, 10, 11, 12, 13 }, emitted);
        // Step 1: t1=5 + 3 drafts; step 2: t1=9 + 3 drafts; step 3: t1=13,
        // budget exhausted → kDraft trimmed to 0 (a 1-token verify).
        Assert.Equal(3, target.BatchVerifyCalls);
        Assert.Equal(new[] { 4, 4, 1 }, target.VerifyLengths);
        Assert.Equal(6, decoder.TotalDraftsEmitted);
        Assert.Equal(6, decoder.TotalDraftsAccepted);
        // Draft context tracks the committed positions exactly.
        Assert.Equal(target.CacheLen, draft.ContextLength);
    }

    [Fact]
    public void Decode_WrongDraft_RejectsAndCorrects_GreedyParity()
    {
        var (decoder, target, draft) = Setup();
        // First proposal: middle draft is wrong (3 breaks the chain at 7).
        draft.Script.Enqueue(([6, 3, 8], [1f, 1f, 1f]));

        var emitted = new List<int>();
        decoder.Decode(6, [], emitted.Add);

        // Byte-identical to the plain chain despite the rejection.
        Assert.Equal(new[] { 5, 6, 7, 8, 9, 10 }, emitted);
        Assert.Equal(target.CacheLen, draft.ContextLength);
        Assert.True(decoder.TotalDraftsAccepted < decoder.TotalDraftsEmitted);
    }

    [Fact]
    public void StopToken_InAcceptedDrafts_NeverEmitted()
    {
        var (decoder, _, _) = Setup();

        var emitted = new List<int>();
        decoder.Decode(20, [7], emitted.Add);

        // Chain 5,6 then 7 = stop: accepted-stop ends the chain, excluded from
        // the commit, never emitted.
        Assert.Equal(new[] { 5, 6 }, emitted);
    }

    [Fact]
    public void StopToken_AsCertainToken_EmitsNothing()
    {
        var (decoder, target, _) = Setup(firstToken: 9);

        var emitted = new List<int>();
        decoder.Decode(20, [9], emitted.Add);

        Assert.Empty(emitted);
        Assert.Equal(0, target.BatchVerifyCalls);
    }

    [Fact]
    public void ConfidenceTrim_ShrinksVerifyBatch()
    {
        var (decoder, target, draft) = Setup();
        draft.Script.Enqueue(([6, 7, 8], [1f, 0.3f, 1f]));

        var emitted = new List<int>();
        decoder.Decode(2, [], emitted.Add, minConfidence: 0.5f);

        Assert.Equal(new[] { 5, 6 }, emitted);
        // Leading confident prefix is 1 draft → verify batch = t1 + 1.
        Assert.Equal(new[] { 2 }, target.VerifyLengths);
    }

    [Fact]
    public void VerifyLenCap_LimitsEveryBatch()
    {
        var (decoder, target, _) = Setup();

        var emitted = new List<int>();
        decoder.Decode(6, [], emitted.Add, verifyLenCap: 1);

        Assert.Equal(new[] { 5, 6, 7, 8, 9, 10 }, emitted);
        Assert.All(target.VerifyLengths, len => Assert.True(len <= 2));
    }

    [Fact]
    public void MaxTokens_NeverVerifiesBeyondBudget()
    {
        var (decoder, target, _) = Setup();

        var emitted = new List<int>();
        decoder.Decode(2, [], emitted.Add);

        Assert.Equal(new[] { 5, 6 }, emitted);
        // t1 emitted leaves budget 1 → exactly one draft verified.
        Assert.Equal(new[] { 2 }, target.VerifyLengths);
    }

    [Fact]
    public void SequentialFallback_WhenBatchVerifyUnsupported()
    {
        var (decoder, target, _) = Setup();
        target.BatchVerifySupported = false;

        var emitted = new List<int>();
        decoder.Decode(5, [], emitted.Add);

        Assert.Equal(new[] { 5, 6, 7, 8, 9 }, emitted);
        Assert.Equal(0, target.BatchVerifyCalls);
        Assert.True(target.ForwardCalls > 0);
        Assert.Equal(target.CacheLen, target.TapHighWater);
    }

    [Fact]
    public void Initialize_WithoutCapturedTaps_Throws()
    {
        var target = new TapChainForwardPass();
        ((IForwardPass)target).EnableHiddenTaps([0, 2]);
        var draft = new ScriptedDraft(target.HiddenTapDim);
        target.Prefill(new int[4]);
        target.TapsPopulated = false;

        var decoder = new DSparkDecoder(target, draft);
        Assert.Throws<InvalidOperationException>(
            () => decoder.Initialize(4, TapChainForwardPass.Logits(5)));
    }

    [Fact]
    public void Initialize_TapDimMismatch_Throws()
    {
        var target = new TapChainForwardPass();
        ((IForwardPass)target).EnableHiddenTaps([0]);   // dim 4
        var draft = new ScriptedDraft(tapDim: 8);        // expects 8
        target.Prefill(new int[4]);

        var decoder = new DSparkDecoder(target, draft);
        Assert.Throws<InvalidOperationException>(
            () => decoder.Initialize(4, TapChainForwardPass.Logits(5)));
    }

    [Fact]
    public void Ctor_RejectsTargetWithoutTapsOrRewind()
    {
        var draft = new ScriptedDraft(tapDim: 8);
        Assert.Throws<ArgumentException>(() => new DSparkDecoder(new NoTapPass(), draft));
        Assert.Throws<ArgumentException>(() => new DSparkDecoder(new NoRewindPass(), draft));
    }

    private sealed class NoTapPass : IForwardPass
    {
        public int VocabSize => Vocab;
        public int MaxSeqLen => 512;
        public bool SupportsPartialRewind => true;
        public ReadOnlySpan<float> Forward(int token, int position) => default;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => default;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class NoRewindPass : IForwardPass
    {
        public int VocabSize => Vocab;
        public int MaxSeqLen => 512;
        public bool SupportsHiddenTaps => true;
        public void EnableHiddenTaps(ReadOnlySpan<int> layerIds) { }
        public ReadOnlySpan<float> Forward(int token, int position) => default;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => default;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    [Fact]
    public void ResolveVerifyLen_Precedence()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_DSPARK_VERIFY_LEN");
        try
        {
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_VERIFY_LEN", "5");
            Assert.Equal(3, DSparkDecoder.ResolveVerifyLen(3));   // flag wins
            Assert.Equal(5, DSparkDecoder.ResolveVerifyLen(0));   // env fallback
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_VERIFY_LEN", null);
            Assert.Equal(0, DSparkDecoder.ResolveVerifyLen(0));   // default: uncapped
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_VERIFY_LEN", prev);
        }
    }

    [Fact]
    public void ResolveMinConfidence_Precedence()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_DSPARK_MIN_CONFIDENCE");
        try
        {
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_MIN_CONFIDENCE", "0.25");
            Assert.Equal(0.5f, DSparkDecoder.ResolveMinConfidence(0.5f));  // flag wins
            Assert.Equal(0f, DSparkDecoder.ResolveMinConfidence(0f));      // explicit 0 wins too
            Assert.Equal(0.25f, DSparkDecoder.ResolveMinConfidence(-1f));  // env fallback
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_MIN_CONFIDENCE", null);
            Assert.Equal(0f, DSparkDecoder.ResolveMinConfidence(-1f));     // default
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_DSPARK_MIN_CONFIDENCE", prev);
        }
    }
}
