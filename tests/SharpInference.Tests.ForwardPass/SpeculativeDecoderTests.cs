using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Construction-time contract for <see cref="SpeculativeDecoder"/>. Speculative decoding
/// rewinds rejected draft tokens via <see cref="IForwardPass.TruncateTo"/>; any forward
/// pass whose state is destructively updated per token (Gated DeltaNet hybrid models,
/// <see cref="IForwardPass.SupportsPartialRewind"/> == false) is rejected up front so the
/// failure surfaces at construction rather than mid-decode. See issue #20.
/// </summary>
public sealed class SpeculativeDecoderTests
{
    [Fact]
    public void Ctor_RewindIncompatibleTarget_ThrowsWithParamNameTarget()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Equal("target", ex.ParamName);
    }

    [Fact]
    public void Ctor_RewindIncompatibleDraft_ThrowsWithParamNameDraft()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Equal("draft", ex.ParamName);
    }

    [Fact]
    public void Ctor_BothRewindCapable_ConstructsSuccessfully()
    {
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);
        var draft = new MockForwardPass(vocabSize: 16, supportsPartialRewind: true);

        // No throw expected. Lookahead defaults to 4 (clamped below in the ctor for ≤ 0).
        var decoder = new SpeculativeDecoder(target, draft);
        Assert.Equal(4, decoder.Lookahead);
    }

    [Fact]
    public void Ctor_VocabSizeMismatch_ThrowsArgumentExceptionRegardlessOfRewindSupport()
    {
        // Regression guard: the vocab-size check (pre-existing) and the partial-rewind check
        // (new in #20) coexist. The vocab-size check runs first.
        var target = new MockForwardPass(vocabSize: 16, supportsPartialRewind: false);
        var draft = new MockForwardPass(vocabSize: 32, supportsPartialRewind: false);

        var ex = Assert.Throws<ArgumentException>(() => new SpeculativeDecoder(target, draft));
        Assert.Contains("vocab", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    // ── BatchVerify routing (issue #207) ────────────────────────────────────────────
    //
    // The decoder dispatches verification through the IForwardPass.SupportsBatchVerify
    // capability (replacing the old `is ForwardPass` CPU type-check), with
    // SHARPI_SPEC_BATCH_VERIFY=0 as the kill-switch back to sequential Forward calls.
    // Scripted chain models (next = token+1 mod vocab) make the greedy accept/reject
    // logic fully deterministic so the call counts can be asserted exactly.

    [Fact]
    public void Decode_BatchVerifyCapableTarget_RoutesThroughBatchVerify()
    {
        var target = new ChainForwardPass(vocab: 16, supportsBatchVerify: true);
        var draft = new ChainForwardPass(vocab: 16, supportsBatchVerify: false);

        var spec = new SpeculativeDecoder(target, draft, lookahead: 3);
        spec.Initialize(prefillLength: 1, ChainForwardPass.Logits(16, next: 2), ChainForwardPass.Logits(16, next: 2));

        var emitted = new List<int>();
        spec.Decode(6, [], emitted.Add);

        // Draft and target agree everywhere (same chain), so both steps accept fully:
        // step 1 (k=3) emits 3+1 tokens, step 2 (k=min(3, remaining=2)=2) emits 2+1.
        Assert.Equal(new[] { 2, 3, 4, 5, 6, 7 }, emitted);
        Assert.Equal(2, target.BatchVerifyCalls);
        // Target Forward runs only for the per-step correction commit, never for verify.
        Assert.Equal(2, target.ForwardCalls);
        // Full acceptance: 5 accepted out of 7 emitted (corrections always count against
        // the rate — k/(k+1) per step is the ceiling).
        Assert.Equal(5f / 7f, spec.AcceptanceRate, 3);
    }

    [Fact]
    public void Decode_KillSwitch_FallsBackToSequentialForward()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SPEC_BATCH_VERIFY");
        Environment.SetEnvironmentVariable("SHARPI_SPEC_BATCH_VERIFY", "0");
        SpeculativeDecoder spec;
        var target = new ChainForwardPass(vocab: 16, supportsBatchVerify: true);
        var draft = new ChainForwardPass(vocab: 16, supportsBatchVerify: false);
        try
        {
            // The kill-switch is read once at construction.
            spec = new SpeculativeDecoder(target, draft, lookahead: 3);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SPEC_BATCH_VERIFY", prev);
        }

        spec.Initialize(prefillLength: 1, ChainForwardPass.Logits(16, next: 2), ChainForwardPass.Logits(16, next: 2));
        var emitted = new List<int>();
        spec.Decode(6, [], emitted.Add);

        // Identical emitted sequence, but verification ran as k sequential Forwards plus
        // the per-step correction commit: step 1 (k=3) 3+1, step 2 (k=2) 2+1 → 7 total.
        Assert.Equal(new[] { 2, 3, 4, 5, 6, 7 }, emitted);
        Assert.Equal(0, target.BatchVerifyCalls);
        Assert.Equal(7, target.ForwardCalls);
    }

    [Fact]
    public void Decode_DraftDiverges_RejectionEmitsCorrectionFromBatchLogits()
    {
        var target = new ChainForwardPass(vocab: 16, supportsBatchVerify: true);
        // Draft diverges from the chain at its second proposal of each step.
        var draft = new ChainForwardPass(vocab: 16, supportsBatchVerify: false, divergeEvery: 2);

        var spec = new SpeculativeDecoder(target, draft, lookahead: 3);
        spec.Initialize(prefillLength: 1, ChainForwardPass.Logits(16, next: 2), ChainForwardPass.Logits(16, next: 2));

        var emitted = new List<int>();
        spec.Decode(4, [], emitted.Add);

        // The emitted sequence must still be the target's greedy chain regardless of
        // where the draft diverged — corrections come from the verify logits.
        Assert.Equal(new[] { 2, 3, 4, 5 }, emitted);
        Assert.True(spec.AcceptanceRate < 1f);
        Assert.True(target.BatchVerifyCalls > 0);
    }

    /// <summary>
    /// Deterministic "chain" model: greedy next token is always (token+1) mod vocab.
    /// Tracks Forward/BatchVerify call counts; <c>divergeEvery</c> &gt; 0 makes every
    /// Nth Forward propose (token+2) instead, simulating a draft that goes off-chain.
    /// </summary>
    private sealed class ChainForwardPass : IForwardPass
    {
        private readonly bool _supportsBatchVerify;
        private readonly int _divergeEvery;

        public int ForwardCalls;
        public int BatchVerifyCalls;

        public ChainForwardPass(int vocab, bool supportsBatchVerify, int divergeEvery = 0)
        {
            VocabSize = vocab;
            _supportsBatchVerify = supportsBatchVerify;
            _divergeEvery = divergeEvery;
        }

        public int VocabSize { get; }
        public int MaxSeqLen => 4096;
        public bool SupportsPartialRewind => true;
        public bool SupportsBatchVerify => _supportsBatchVerify;

        public static float[] Logits(int vocab, int next)
        {
            var l = new float[vocab];
            l[next] = 1f;
            return l;
        }

        public ReadOnlySpan<float> Forward(int token, int position)
        {
            ForwardCalls++;
            int next = (token + 1) % VocabSize;
            if (_divergeEvery > 0 && ForwardCalls % _divergeEvery == 0)
                next = (token + 2) % VocabSize;
            return Logits(VocabSize, next);
        }

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            if (!_supportsBatchVerify)
                throw new NotSupportedException("BatchVerify called on a non-capable mock.");
            BatchVerifyCalls++;
            var result = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++)
                result[i] = Logits(VocabSize, (tokens[i] + 1) % VocabSize);
            return result;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[VocabSize];
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }

    private sealed class MockForwardPass : IForwardPass
    {
        public MockForwardPass(int vocabSize, bool supportsPartialRewind)
        {
            VocabSize = vocabSize;
            SupportsPartialRewind = supportsPartialRewind;
        }

        public int VocabSize { get; }
        public int MaxSeqLen => 4096;
        public bool SupportsPartialRewind { get; }

        public ReadOnlySpan<float> Forward(int token, int position) => new float[VocabSize];
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => new float[VocabSize];
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
