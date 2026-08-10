using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Pins two properties of the window penalties that are cheap to assert without a model, using
/// <see cref="Sampler.BuildFilteredDistribution"/> (which runs the slow path's exact pipeline).
/// <list type="number">
///   <item>
///     <b>Path equivalence.</b> <c>TopK = 0</c> (full-vocabulary slow path) and <c>TopK = k</c>
///     (the <c>SampleTopK</c> fast path) must produce the same post-filter distribution whenever
///     the surviving support fits inside k, and the slow path's support must never be
///     <em>narrower</em> than the fast path's when it does not. Reported against 0.17.0 as a
///     suspected source of penalty-only degeneration on the slow path; it is not one — the paths
///     agree exactly.
///   </item>
///   <item>
///     <b>Shift dependence.</b> Softmax is shift-invariant, so adding a constant to every logit
///     must not change the sampled distribution — and does not, at
///     <see cref="SamplingParams.RepetitionPenalty"/> 1. Under a penalty it does, because the
///     penalty scales the logit (divide if positive, multiply if negative) around a hard zero.
///     That divergence was measured and deliberately retained: the sign-split form is llama.cpp's,
///     HuggingFace's and vLLM's, and the shift-invariant alternative already ships as
///     <see cref="SamplingParams.PresencePenalty"/> (issue #459). These tests therefore <b>pin</b>
///     the parity form rather than merely recording it.
///   </item>
/// </list>
/// </summary>
public sealed class PenaltyPathEquivalenceTests
{
    private const int Vocab = 32768;
    private const int DominantToken = 7;
    private const int BandStart = 100;
    private const int BandSize = 400;

    /// <summary>
    /// A peaked logit vector: one dominant token, a descending band of plausible alternatives, and
    /// a long low tail. <paramref name="shift"/> is added to every entry — a softmax no-op, so any
    /// behavioural difference it produces comes from a stage that is not shift-invariant.
    /// </summary>
    private static float[] MakeLogits(float peak, float bandTop, float shift)
    {
        uint s = 1234;
        float Next() { s = s * 1664525u + 1013904223u; return (s >> 8) * (1f / 16777216f); }

        var l = new float[Vocab];
        for (int i = 0; i < Vocab; i++) l[i] = -6f + 8f * (Next() - 0.5f);
        for (int i = 0; i < BandSize; i++) l[BandStart + i] = bandTop - 0.06f * i;
        l[DominantToken] = peak;
        for (int i = 0; i < Vocab; i++) l[i] += shift;
        return l;
    }

    /// <summary>The "previous reply": the dominant token, most of the band, and a tail slice.</summary>
    private static List<int> MakeWindow()
    {
        var w = new List<int> { DominantToken };
        for (int i = 0; i < 120; i++) w.Add(BandStart + i);
        for (int i = 0; i < 200; i++) w.Add(9000 + i * 7);
        return w;
    }

    private static SamplingParams Params(int topK, float rep, float minP) => new()
    {
        Temperature = 0.85f,
        TopK = topK,
        TopP = 1.0f,
        MinP = minP,
        RepetitionPenalty = rep,
        PreviousTokens = rep == 1.0f ? null : MakeWindow(),
    };

    private static float[] Distribution(float[] logits, SamplingParams p)
    {
        var probs = new float[Vocab];
        Sampler.BuildFilteredDistribution(logits, p, probs);
        return probs;
    }

    private static int Support(float[] probs)
    {
        int n = 0;
        foreach (float v in probs) if (v > 0f) n++;
        return n;
    }

    private static double Entropy(float[] probs)
    {
        double h = 0;
        foreach (float v in probs) if (v > 0f) h -= v * Math.Log(v);
        return h;
    }

    /// <summary>
    /// With min-p keeping fewer than k tokens — the ordinary case at minP 0.05 — the slow path and
    /// the fast path build the SAME distribution, penalty or no penalty. The fast path only ever
    /// penalises candidates inside the top-(k + window) raw logits, so this also confirms that the
    /// windowed tokens the slow path additionally penalises (deep tail, already below the min-p
    /// cutoff) have no effect on the sampled distribution.
    /// </summary>
    [Theory]
    [InlineData(1.0f, 0f)]
    [InlineData(1.15f, 0f)]
    [InlineData(1.3f, 0f)]
    [InlineData(1.0f, -12f)]
    [InlineData(1.15f, -12f)]
    [InlineData(1.3f, -12f)]
    public void SlowAndFastPathsAgreeWhenSupportFitsInK(float rep, float shift)
    {
        var logits = MakeLogits(peak: 6.0f, bandTop: 5.5f, shift);
        var slow = Distribution(logits, Params(topK: 0, rep, minP: 0.05f));
        var fast = Distribution(logits, Params(topK: 64, rep, minP: 0.05f));

        Assert.InRange(Support(slow), 1, 64);   // guard: the premise of this test
        Assert.Equal(Support(slow), Support(fast));
        for (int i = 0; i < Vocab; i++)
            Assert.Equal(slow[i], fast[i], 6);
    }

    /// <summary>
    /// When min-p keeps MORE than k tokens the paths necessarily differ — but only by the fast
    /// path truncating the tail at k. The slow path's support is strictly wider and its entropy
    /// higher, so the slow path can never be the more deterministic of the two.
    /// </summary>
    [Theory]
    [InlineData(1.0f, 0f)]
    [InlineData(1.15f, 0f)]
    [InlineData(1.15f, -12f)]
    public void SlowPathIsNeverNarrowerThanFastPath(float rep, float shift)
    {
        var logits = MakeLogits(peak: 6.0f, bandTop: 5.9f, shift);
        var slow = Distribution(logits, Params(topK: 0, rep, minP: 0.002f));
        var fast = Distribution(logits, Params(topK: 64, rep, minP: 0.002f));

        Assert.True(Support(slow) > 64, "premise: min-p keeps more than k tokens here");
        Assert.Equal(64, Support(fast));
        Assert.True(Support(slow) > Support(fast));
        Assert.True(Entropy(slow) > Entropy(fast));
    }

    /// <summary>
    /// Control: with no penalty active, shifting every logit by a constant changes nothing. Every
    /// stage of the pipeline (temperature, softmax, top-k, min-p, top-p) is shift-invariant.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    public void WithoutPenaltyTheDistributionIsShiftInvariant(int topK)
    {
        var p = Params(topK, rep: 1.0f, minP: 0.05f);
        var atZero = Distribution(MakeLogits(6.0f, 5.5f, 0f), p);
        var shifted = Distribution(MakeLogits(6.0f, 5.5f, -12f), p);

        for (int i = 0; i < Vocab; i++)
            Assert.Equal(atZero[i], shifted[i], 6);
    }

    /// <summary>
    /// The repetition penalty breaks that invariance, and inverts direction with it.
    /// <para>
    /// The penalty demotes by an amount proportional to <c>|logit|</c> — a positive logit is
    /// divided (its gap to the rest COMPRESSES, widening the distribution), a negative one is
    /// multiplied (its gap EXPANDS, sharpening it). So the same model state, offset by a constant
    /// that softmax cannot see, gets opposite treatment: raising the penalty widens the surviving
    /// support when the head of the distribution sits above zero and narrows it when it sits below.
    /// </para>
    /// <para>
    /// Both sampling paths do this identically — it is the penalty formula, not the path.
    /// </para>
    /// <para>
    /// This is the <b>parity pin</b>. The behaviour is deliberate (see
    /// <see cref="SamplingParams.RepetitionPenalty"/>): the sign split is what stops the CTRL
    /// formula from promoting negative-logit tokens, and it is byte-compatible with llama.cpp.
    /// If this test fails, either the formula was changed on purpose — in which case update this
    /// class, the <see cref="SamplingParams.RepetitionPenalty"/> docs, and both penalty sites in
    /// <c>Sampler</c>, and expect llama.cpp cross-check divergence on penalised configs — or a
    /// regression was introduced.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(64)]
    public void RepetitionPenaltyDirectionDependsOnLogitSign(int topK)
    {
        var positive = MakeLogits(6.0f, 5.5f, 0f);
        var negative = MakeLogits(6.0f, 5.5f, -12f);

        var basePositive = Distribution(positive, Params(topK, 1.0f, 0.05f));
        var baseNegative = Distribution(negative, Params(topK, 1.0f, 0.05f));

        // Same starting point either way (the control above, restated as this test's premise).
        Assert.Equal(Support(basePositive), Support(baseNegative));

        // Positive logits: rising penalty widens the support and raises entropy, monotonically.
        int prevSupport = Support(basePositive);
        double prevEntropy = Entropy(basePositive);
        foreach (float rep in new[] { 1.05f, 1.15f, 1.3f })
        {
            var d = Distribution(positive, Params(topK, rep, 0.05f));
            Assert.True(Support(d) > prevSupport, $"positive logits, rep {rep}: support should widen");
            Assert.True(Entropy(d) > prevEntropy, $"positive logits, rep {rep}: entropy should rise");
            prevSupport = Support(d);
            prevEntropy = Entropy(d);
        }

        // Negative logits: the identical softmax input, penalised the same way, narrows instead.
        prevSupport = Support(baseNegative);
        prevEntropy = Entropy(baseNegative);
        foreach (float rep in new[] { 1.05f, 1.15f, 1.3f })
        {
            var d = Distribution(negative, Params(topK, rep, 0.05f));
            Assert.True(Support(d) < prevSupport, $"negative logits, rep {rep}: support narrows");
            Assert.True(Entropy(d) < prevEntropy, $"negative logits, rep {rep}: entropy falls");
            prevSupport = Support(d);
            prevEntropy = Entropy(d);
        }
    }

    /// <summary>
    /// The invariant that actually matters, and the one any future replacement formula must still
    /// satisfy: raising <see cref="SamplingParams.RepetitionPenalty"/> must strictly reduce a
    /// windowed token's probability <em>relative to an unwindowed competitor</em>. Unlike
    /// <see cref="RepetitionPenaltyDirectionDependsOnLogitSign"/> — which pins the current form and
    /// would legitimately need rewriting if the formula ever changed — this pin outlives a
    /// deliberate formula change, and holds in both sign regimes.
    /// <para>
    /// A dedicated two-token head is used rather than the shared <see cref="MakeLogits"/> fixture,
    /// whose window covers the entire band: an unpenalised reference token has to survive
    /// filtering for the ratio to be observable. Both <c>TopK</c> settings run through
    /// <see cref="Sampler.BuildFilteredDistribution"/>, whose top-k stage is equivalent to the
    /// <c>SampleTopK</c> fast path per
    /// <see cref="SlowAndFastPathsAgreeWhenSupportFitsInK"/>.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(0, 0f)]      // slow path, positive-logit head
    [InlineData(8, 0f)]      // top-k path,  positive-logit head
    [InlineData(0, -12f)]    // slow path, all-negative head
    [InlineData(8, -12f)]    // top-k path,  all-negative head
    public void RepetitionPenaltyNeverPromotesASeenTokenRelativeToAnUnseenOne(int topK, float shift)
    {
        const int Seen = 0;      // in the penalty window
        const int Unseen = 1;    // not in the window, and slightly behind Seen to begin with

        var logits = new float[Vocab];
        Array.Fill(logits, -20f + shift);
        logits[Seen] = 5f + shift;
        logits[Unseen] = 4.8f + shift;
        var window = new List<int> { Seen };

        double previousRatio = double.PositiveInfinity;
        foreach (float rep in new[] { 1.0f, 1.05f, 1.15f, 1.3f })
        {
            var p = new SamplingParams
            {
                Temperature = 0.85f,
                TopK = topK,
                TopP = 1.0f,
                MinP = 0f,
                RepetitionPenalty = rep,
                PreviousTokens = rep == 1.0f ? null : window,
            };
            var probs = new float[Vocab];
            Sampler.BuildFilteredDistribution(logits, p, probs);

            Assert.True(probs[Unseen] > 0f, $"rep {rep}: the unpenalised reference must survive filtering");
            double ratio = probs[Seen] / probs[Unseen];
            Assert.True(ratio < previousRatio,
                $"rep {rep}: seen/unseen ratio {ratio:F4} should fall below {previousRatio:F4}");
            previousRatio = ratio;
        }
    }
}
