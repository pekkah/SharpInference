using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Sampled (temp &gt; 0) speculative decoding — issue #178. The core invariant is
/// distribution preservation: with the distribution-preserving accept rule (draft sampled with
/// proposal prob q, accepted with min(1, p/q), rejection resampled from the residual max(0, p−q),
/// full accept drawn from the last verify position), the emitted-token distribution is identical
/// to sampling directly from the target — REGARDLESS of the draft's distribution.
///
/// Tested with a position/token-independent fixed-distribution mock: the target always returns the
/// same distribution p and the draft a deliberately different q, so the aggregate histogram of all
/// emitted tokens must converge to softmax(p). Also covers RNG determinism and that temp ≤ 0 still
/// routes to the byte-stable greedy path.
/// </summary>
public sealed class SpeculativeSamplingTests
{
    // Two deliberately different fixed distributions over a small vocab.
    private static readonly float[] PTarget = [0.2f, 2.5f, 0.1f, 1.0f, 0.3f, 1.8f, 0.0f, 0.5f];
    private static readonly float[] QDraft  = [2.0f, 0.1f, 1.5f, 0.2f, 1.0f, 0.0f, 2.2f, 0.3f];

    [Fact]
    public void SampledSpec_PreservesTargetDistribution()
    {
        var target = new FixedDistForwardPass(PTarget, supportsBatchVerify: true);
        var draft  = new FixedDistForwardPass(QDraft,  supportsBatchVerify: false);
        var spec = new SpeculativeDecoder(target, draft, new SamplingParams { Temperature = 1f }, new Random(12345), lookahead: 4);
        spec.Initialize(1, PTarget);

        const int n = 40000;
        var counts = new int[PTarget.Length];
        spec.Decode(n, [], t => counts[t]++);

        var expected = Softmax(PTarget);
        double maxDev = 0;
        for (int i = 0; i < PTarget.Length; i++)
            maxDev = Math.Max(maxDev, Math.Abs((double)counts[i] / n - expected[i]));

        // ~8σ headroom at N=40000 (max prob ≈ 0.45 → σ ≈ 0.0025) yet far below the ≥0.1
        // deviation a broken accept rule (emitting ≈q, or a biased p/q mix) would produce.
        Assert.True(maxDev < 0.02, $"emitted histogram deviates from softmax(target) by {maxDev:F4} (> 0.02)");
        // The distributions differ, so some drafts are rejected and some accepted.
        Assert.InRange(spec.AcceptanceRate, 0.01f, 0.99f);
    }

    [Fact]
    public void SampledSpec_PMinLooserAccept_StillProducesValidTokens()
    {
        // --spec-draft-p-min in (0,1) opts into the looser accept; not distribution-preserving,
        // but must still emit valid in-vocab tokens and accept more than the strict rule would.
        var target = new FixedDistForwardPass(PTarget, supportsBatchVerify: true);
        var draft  = new FixedDistForwardPass(QDraft,  supportsBatchVerify: false);
        var spec = new SpeculativeDecoder(target, draft,
            new SamplingParams { Temperature = 1f, SpecDraftPMin = 0.05f }, new Random(7), lookahead: 4);
        spec.Initialize(1, PTarget);

        var emitted = new List<int>();
        spec.Decode(500, [], emitted.Add);
        Assert.Equal(500, emitted.Count);
        Assert.All(emitted, t => Assert.InRange(t, 0, PTarget.Length - 1));
    }

    [Fact]
    public void SampledSpec_SameSeed_Deterministic()
        => Assert.Equal(RunSampled(seed: 999, n: 200), RunSampled(seed: 999, n: 200));

    [Fact]
    public void SampledSpec_DifferentSeed_Differs()
        => Assert.NotEqual(RunSampled(seed: 1, n: 200), RunSampled(seed: 2, n: 200));

    [Fact]
    public void SamplingCtor_TempZero_RoutesToGreedy()
    {
        // Temp ≤ 0 leaves _sampling null → the greedy path runs (byte-stable). With a fixed
        // target distribution, greedy emits argmax(p) at every position; the draft never matches.
        var target = new FixedDistForwardPass(PTarget, supportsBatchVerify: true);
        var draft  = new FixedDistForwardPass(QDraft,  supportsBatchVerify: false);
        var spec = new SpeculativeDecoder(target, draft, new SamplingParams { Temperature = 0f }, new Random(1), lookahead: 4);
        spec.Initialize(1, PTarget);

        var emitted = new List<int>();
        spec.Decode(20, [], emitted.Add);
        int argmax = Sampler.Greedy(PTarget);
        Assert.All(emitted, t => Assert.Equal(argmax, t));
    }

    private static List<int> RunSampled(int seed, int n)
    {
        var target = new FixedDistForwardPass(PTarget, supportsBatchVerify: true);
        var draft  = new FixedDistForwardPass(QDraft,  supportsBatchVerify: false);
        var spec = new SpeculativeDecoder(target, draft, new SamplingParams { Temperature = 1f }, new Random(seed), lookahead: 4);
        spec.Initialize(1, PTarget);
        var emitted = new List<int>();
        spec.Decode(n, [], emitted.Add);
        return emitted;
    }

    private static double[] Softmax(float[] logits)
    {
        double max = logits.Max();
        double sum = 0;
        var e = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) { e[i] = Math.Exp(logits[i] - max); sum += e[i]; }
        for (int i = 0; i < logits.Length; i++) e[i] /= sum;
        return e;
    }

    /// <summary>
    /// Position/token-independent mock: <see cref="Forward"/> and <see cref="BatchVerify"/> always
    /// return the same fixed logits, so every verified position has the same distribution and the
    /// aggregate emitted histogram is directly comparable to that distribution.
    /// </summary>
    private sealed class FixedDistForwardPass : IForwardPass
    {
        private readonly float[] _logits;
        private readonly bool _bv;

        public FixedDistForwardPass(float[] logits, bool supportsBatchVerify)
        {
            _logits = logits;
            _bv = supportsBatchVerify;
        }

        public int VocabSize => _logits.Length;
        public int MaxSeqLen => 1 << 20;          // mock ignores positions; allow long runs
        public bool SupportsPartialRewind => true;
        public bool SupportsBatchVerify => _bv;

        public ReadOnlySpan<float> Forward(int token, int position) => _logits;

        public float[][] BatchVerify(int[] tokens, int startPos)
        {
            var r = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++) r[i] = (float[])_logits.Clone();
            return r;
        }

        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => _logits;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public void Dispose() { }
    }
}
