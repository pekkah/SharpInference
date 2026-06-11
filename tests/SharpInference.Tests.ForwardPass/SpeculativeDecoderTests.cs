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
        // each k=3 step packs [certain, d1, d2] into one verify and emits all 3.
        Assert.Equal(new[] { 2, 3, 4, 5, 6, 7 }, emitted);
        Assert.Equal(2, target.BatchVerifyCalls);
        // The certain token rides in the verify batch, so the target NEVER runs a
        // single-token Forward — one batched pass per step is the whole target cost.
        Assert.Equal(0, target.ForwardCalls);
        // 2 accepted proposals out of 3 emitted per step (the certain token never
        // counts as accepted): 4/6.
        Assert.Equal(4f / 6f, spec.AcceptanceRate, 3);
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

        // Identical emitted sequence, but verification ran as k sequential Forwards:
        // 2 steps × 3 = 6 total (no batched pass, no separate commit forward either).
        Assert.Equal(new[] { 2, 3, 4, 5, 6, 7 }, emitted);
        Assert.Equal(0, target.BatchVerifyCalls);
        Assert.Equal(6, target.ForwardCalls);
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

    [Fact]
    public void Decode_PromptLookupMode_EmitsTargetChainAndUsesLookupProposals()
    {
        var target = new ChainForwardPass(vocab: 16, supportsBatchVerify: true);
        var spec = new SpeculativeDecoder(target, new PromptLookupDraft(ngramMax: 3, ngramMin: 2), lookahead: 4);

        // Prompt [10,11,12,10]; the target's saved logits continue the chain with 11.
        // Step 1: certain 11 joins the history → tail [10,11] matches index 0 → proposals
        // [12,10,11]. The chain target accepts 12 (the chain's true next) and rejects 10,
        // so the step emits [11,12]. Later steps find no matching tail and degrade to
        // plain single-token decode steps — the floor behavior.
        spec.Initialize(new[] { 10, 11, 12, 10 }, ChainForwardPass.Logits(16, next: 11));

        var emitted = new List<int>();
        spec.Decode(4, [], emitted.Add);

        // The emitted sequence is the target's greedy chain regardless of proposal quality.
        Assert.Equal(new[] { 11, 12, 13, 14 }, emitted);
        // Step 1 verified [11,12,10,11] (one batch), steps 2 and 3 verified the lone
        // certain token; the certain token rides in the verify, so no target Forward runs.
        Assert.Equal(3, target.BatchVerifyCalls);
        Assert.Equal(0, target.ForwardCalls);
        // Exactly one proposal (the 12) was accepted across the run.
        Assert.True(spec.AcceptanceRate > 0f);
    }

    [Fact]
    public void Initialize_PromptOverloadWithoutLookup_Throws()
    {
        var target = new ChainForwardPass(vocab: 16, supportsBatchVerify: true);
        var draft = new ChainForwardPass(vocab: 16, supportsBatchVerify: false);
        var spec = new SpeculativeDecoder(target, draft);

        Assert.Throws<InvalidOperationException>(
            () => spec.Initialize(new[] { 1, 2, 3 }, ChainForwardPass.Logits(16, next: 4)));
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
