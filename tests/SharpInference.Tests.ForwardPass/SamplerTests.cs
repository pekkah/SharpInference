using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

public sealed class SamplerTests
{
    [Fact]
    public void Greedy_ReturnsArgmax()
    {
        float[] logits = [1f, 5f, 3f, 2f];
        Assert.Equal(1, Sampler.Greedy(logits));
    }

    [Fact]
    public void Greedy_FirstElementIsMax()
    {
        float[] logits = [10f, 1f, 2f, 3f];
        Assert.Equal(0, Sampler.Greedy(logits));
    }

    [Fact]
    public void Greedy_LastElementIsMax()
    {
        float[] logits = [1f, 2f, 3f, 100f];
        Assert.Equal(3, Sampler.Greedy(logits));
    }

    [Fact]
    public void Sample_TemperatureZero_IsGreedy()
    {
        float[] logits = [1f, 5f, 3f, 2f];
        var p = new SamplingParams { Temperature = 0f };
        Assert.Equal(1, Sampler.Sample(logits, p));
    }

    [Fact]
    public void Sample_VeryLowTemperature_PeaksAtMax()
    {
        float[] logits = [1f, 10f, 2f, 3f];
        var p = new SamplingParams { Temperature = 0.001f };
        var rng = new Random(42);

        // With near-zero temperature, should almost always pick the max
        int maxPicks = 0;
        for (int i = 0; i < 100; i++)
        {
            if (Sampler.Sample(logits, p, rng) == 1)
                maxPicks++;
        }
        Assert.True(maxPicks >= 99, $"Expected >=99 picks of max, got {maxPicks}");
    }

    [Fact]
    public void Sample_HighTemperature_MoreUniform()
    {
        float[] logits = [0f, 0f, 0f, 10f];
        var p = new SamplingParams { Temperature = 100f };
        var rng = new Random(42);

        // With very high temperature, distribution approaches uniform
        int[] counts = new int[4];
        for (int i = 0; i < 10000; i++)
            counts[Sampler.Sample(logits, p, rng)]++;

        // Each bucket should get roughly 25% ± generous margin
        for (int i = 0; i < 4; i++)
            Assert.InRange(counts[i], 1500, 3500);
    }

    [Fact]
    public void Sample_TopK_RestrictsToKTokens()
    {
        // 10 tokens, only top 2 should be sampled
        float[] logits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        var p = new SamplingParams { Temperature = 1f, TopK = 2 };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 1000; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        // Should only sample tokens 8 and 9 (indices of top-2 logits)
        Assert.True(sampled.IsSubsetOf(new[] { 8, 9 }),
            $"Sampled tokens outside top-2: {string.Join(",", sampled)}");
    }

    [Fact]
    public void Sample_TopP_RestrictsToNucleus()
    {
        // Create a distribution where one token dominates
        // After softmax, the top token will have most of the probability
        float[] logits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 20];
        var p = new SamplingParams { Temperature = 1f, TopP = 0.5f };
        var rng = new Random(42);

        int[] counts = new int[10];
        for (int i = 0; i < 100; i++)
            counts[Sampler.Sample(logits, p, rng)]++;

        // Token 9 should get almost all samples
        Assert.True(counts[9] >= 95, $"Expected token 9 to dominate, got {counts[9]}/100");
    }

    [Fact]
    public void Sample_MinP_FiltersLowProbTokens()
    {
        // One dominant token, rest very low
        float[] logits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 50];
        var p = new SamplingParams { Temperature = 1f, MinP = 0.1f };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 100; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        // With such a dominant token and minP=0.1, only token 9 should survive
        Assert.Contains(9, sampled);
        Assert.True(sampled.Count <= 2, $"Expected at most 2 tokens, got {sampled.Count}");
    }

    [Fact]
    public void Sample_ReturnedToken_InValidRange()
    {
        float[] logits = [1f, 2f, 3f, 4f, 5f];
        var p = new SamplingParams { Temperature = 1f };
        var rng = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            int token = Sampler.Sample(logits, p, rng);
            Assert.InRange(token, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Sample_UniformLogits_AllTokensSampled()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        var p = new SamplingParams { Temperature = 1f };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 10000; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        Assert.Equal(4, sampled.Count);
    }

    // ── LogitBias ─────────────────────────────────────────────────────────────

    [Fact]
    public void Sample_LogitBias_NegativeHundred_BlocksToken()
    {
        // Token 0 has the highest base logit, but bias -100 should prevent it.
        float[] logits = [10f, 1f, 1f, 1f];
        var bias = new Dictionary<int, float> { { 0, -100f } };
        var p = new SamplingParams { Temperature = 1f, LogitBias = bias };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 200; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        Assert.DoesNotContain(0, sampled);
    }

    [Fact]
    public void Sample_LogitBias_PositiveHundred_ForcesToken()
    {
        // All tokens equal, but token 2 gets +100 bias — should always win.
        float[] logits = [0f, 0f, 0f, 0f];
        var bias = new Dictionary<int, float> { { 2, 100f } };
        var p = new SamplingParams { Temperature = 1f, LogitBias = bias };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 200; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        Assert.True(sampled.IsSubsetOf(new[] { 2 }),
            $"Expected only token 2, got: {string.Join(",", sampled)}");
    }

    [Fact]
    public void Sample_LogitBias_OutOfRangeId_IsIgnored()
    {
        float[] logits = [1f, 2f, 3f];
        var bias = new Dictionary<int, float> { { 999, 100f }, { -1, 100f } };
        var p = new SamplingParams { Temperature = 1f, LogitBias = bias };
        var rng = new Random(42);

        // Should not throw; out-of-range IDs are silently skipped.
        int token = Sampler.Sample(logits, p, rng);
        Assert.InRange(token, 0, logits.Length - 1);
    }

    [Fact]
    public void Sample_NullLogitBias_BehavesNormally()
    {
        float[] logits = [0f, 0f, 0f, 0f];
        var p = new SamplingParams { Temperature = 1f, LogitBias = null };
        var rng = new Random(42);

        var sampled = new HashSet<int>();
        for (int i = 0; i < 1000; i++)
            sampled.Add(Sampler.Sample(logits, p, rng));

        Assert.Equal(4, sampled.Count);
    }

    // ── Top-k-first fast path (SampleTopK) ────────────────────────────────────
    // The fast path runs when TopK>0 and there is no logit bias. Adding an out-of-range
    // LogitBias entry (never applied — the id is bounds-checked away) forces the otherwise
    // identical slow path, giving a differential oracle for the fast path.
    private static readonly IReadOnlyDictionary<int, float> ForceSlowPath =
        new Dictionary<int, float> { { -1, 0f } };

    private static float[] RandomLogits(int vocab, int seed)
    {
        var rng = new Random(seed);
        var l = new float[vocab];
        for (int i = 0; i < vocab; i++) l[i] = (float)(rng.NextDouble() * 12 - 6);
        return l;
    }

    private static HashSet<int> ReachableSet(float[] logits, SamplingParams p, int seed, int trials)
    {
        var rng = new Random(seed);
        var set = new HashSet<int>();
        for (int i = 0; i < trials; i++) set.Add(Sampler.Sample(logits, p, rng));
        return set;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(99)]
    public void SampleTopK_FastPath_MatchesSlowPath_ReachableSet(int seed)
    {
        // Same config, same logits: the fast path (no bias) and the slow path (forced via
        // out-of-range bias) must reach exactly the same set of tokens — both apply
        // top-k → renormalize → min-p → top-p over the same nucleus.
        float[] logits = RandomLogits(96, seed);
        var prev = new List<int> { 3, 3, 10, 25 };
        var baseP = new SamplingParams
        {
            Temperature = 0.8f, TopK = 16, TopP = 0.9f, MinP = 0.02f,
            RepetitionPenalty = 1.3f, PreviousTokens = prev,
        };

        var fast = ReachableSet(logits, baseP, seed: 1234, trials: 4000);
        var slow = ReachableSet(logits, baseP with { LogitBias = ForceSlowPath }, seed: 1234, trials: 4000);

        Assert.Equal(slow.OrderBy(x => x), fast.OrderBy(x => x));
    }

    [Fact]
    public void SampleTopK_Penalty_DemotesPenalizedTokenOutOfTopK()
    {
        // Token 1 is the 2nd-highest raw logit; penalising it pushes it below token 2, so
        // with TopK=2 the surviving pair becomes {0, 2} rather than {0, 1}.
        float[] logits = [10f, 8f, 7f, 1f, 0f];
        var p = new SamplingParams
        {
            Temperature = 1f, TopK = 2, RepetitionPenalty = 4f,
            PreviousTokens = new List<int> { 1 },
        };
        var reach = ReachableSet(logits, p, seed: 5, trials: 500);
        Assert.DoesNotContain(1, reach);          // demoted out of the top-2
        Assert.True(reach.IsSubsetOf(new[] { 0, 2 }), $"got {string.Join(",", reach)}");
        Assert.Contains(2, reach);                // token 2 took the freed slot
    }

    [Fact]
    public void SampleTopK_Penalty_DuplicateOccurrences_DoNotCompound()
    {
        // Issue #457: the penalty lands once per DISTINCT token, so repeating a token in the
        // window must not deepen its demotion. Token 1 (logit 8) sits 2nd; token 2 has logit 6.5.
        // 8/1.2 = 6.67 > 6.5, so token 1 keeps the second top-2 slot however often it recurs.
        // (Before the fix, three occurrences compounded to 8/1.2^3 = 4.63 and demoted it — which
        // is what made a nominal penalty land far harder than the same number in llama.cpp.)
        float[] logits = [9f, 8f, 6.5f, 1f];
        var once = new SamplingParams
        {
            Temperature = 1f, TopK = 2, RepetitionPenalty = 1.2f,
            PreviousTokens = new List<int> { 1 },
        };
        var thrice = once with { PreviousTokens = new List<int> { 1, 1, 1 } };

        var reachOnce = ReachableSet(logits, once, seed: 9, trials: 500);
        var reachThrice = ReachableSet(logits, thrice, seed: 9, trials: 500);

        Assert.Contains(1, reachOnce);
        Assert.Equal(reachOnce, reachThrice);     // occurrence count changes nothing
        Assert.DoesNotContain(2, reachThrice);    // token 2 never takes the slot
    }

    [Fact]
    public void SampleTopK_NegInfLogits_NeverReturnsInvalidToken()
    {
        // Masked tokens (-inf) must never be selected, and the sentinel index must never
        // leak out: every returned token is a valid, finite-logit index.
        float[] logits = [2f, float.NegativeInfinity, 5f, float.NegativeInfinity, 1f, 3f];
        var p = new SamplingParams { Temperature = 1f, TopK = 5, TopP = 0.95f };
        var rng = new Random(3);
        for (int i = 0; i < 1000; i++)
        {
            int tok = Sampler.Sample(logits, p, rng);
            Assert.InRange(tok, 0, logits.Length - 1);
            Assert.False(float.IsNegativeInfinity(logits[tok]), $"sampled masked token {tok}");
        }
    }

    [Fact]
    public void SampleTopK_AllNegInf_FallsBackToValidToken()
    {
        float[] logits = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity];
        var p = new SamplingParams { Temperature = 1f, TopK = 2, TopP = 0.9f };
        int tok = Sampler.Sample(logits, p, new Random(1));
        Assert.InRange(tok, 0, logits.Length - 1);   // valid index, not the -1 sentinel
    }

    [Fact]
    public void SampleTopK_TopKThenTopP_NucleusTakenAfterTopK()
    {
        // top-k keeps the 3 highest; after renormalising over them, top-p=0.5 keeps the
        // smallest leading prefix reaching 0.5 — here just token 0 (dominant).
        float[] logits = [6f, 3f, 2.5f, 0f, 0f, 0f, 0f, 0f];
        var p = new SamplingParams { Temperature = 1f, TopK = 3, TopP = 0.5f };
        var reach = ReachableSet(logits, p, seed: 2, trials: 500);
        Assert.True(reach.IsSubsetOf(new[] { 0 }), $"got {string.Join(",", reach)}");
    }

    // ── Distribution helpers for speculative sampling (issue #178) ───────────────────

    [Fact]
    public void BuildFilteredDistribution_NoFilters_MatchesSoftmax()
    {
        float[] logits = [1f, 2f, 3f, 0f];
        var p = new SamplingParams { Temperature = 1f }; // no top-k/top-p/min-p
        var probs = new float[logits.Length];
        Sampler.BuildFilteredDistribution(logits, p, probs);

        // Reference softmax.
        double max = 3.0, sum = 0;
        var expected = new double[logits.Length];
        for (int i = 0; i < logits.Length; i++) { expected[i] = Math.Exp(logits[i] - max); sum += expected[i]; }
        for (int i = 0; i < logits.Length; i++)
            Assert.Equal(expected[i] / sum, probs[i], 5);
        Assert.Equal(1f, probs.Sum(), 5);
    }

    [Fact]
    public void BuildFilteredDistribution_TempZero_OneHotAtArgmax()
    {
        float[] logits = [1f, 9f, 3f, 2f];
        var probs = new float[logits.Length];
        Sampler.BuildFilteredDistribution(logits, new SamplingParams { Temperature = 0f }, probs);
        Assert.Equal(1f, probs[1]);
        Assert.Equal(0f, probs[0] + probs[2] + probs[3]);
    }

    [Fact]
    public void BuildFilteredDistribution_TopK_ZerosOutsideTopKAndSumsToOne()
    {
        float[] logits = [5f, 4f, 3f, 2f, 1f];
        var probs = new float[logits.Length];
        Sampler.BuildFilteredDistribution(logits, new SamplingParams { Temperature = 1f, TopK = 2 }, probs);
        Assert.True(probs[0] > 0f && probs[1] > 0f);
        Assert.Equal(0f, probs[2] + probs[3] + probs[4]);   // outside top-2 zeroed
        Assert.Equal(1f, probs.Sum(), 5);
    }

    [Fact]
    public void SampleWithDistribution_FillsProbsAndReturnsDrawnToken()
    {
        // Dominant token 1 → with low temperature the draw is token 1 and probs[1] ≈ 1.
        float[] logits = [0f, 12f, 0f, 0f];
        var probs = new float[logits.Length];
        int tok = Sampler.SampleWithDistribution(logits, new SamplingParams { Temperature = 0.1f }, probs, new Random(7));
        Assert.Equal(1, tok);
        Assert.True(probs[1] > 0.99f);
        Assert.Equal(1f, probs.Sum(), 5);
    }

    [Fact]
    public void ResampleResidual_ConcentratedResidual_AlwaysReturnsResidualToken()
    {
        // p mass on {1,2}; q takes all of token 1 → residual = {2}. Every draw must be token 2.
        float[] p = [0f, 0.5f, 0.5f, 0f];
        float[] q = [0f, 1.0f, 0.0f, 0f];
        var rng = new Random(3);
        for (int i = 0; i < 200; i++)
            Assert.Equal(2, Sampler.ResampleResidual(p, q, rng));
    }

    [Fact]
    public void ResampleResidual_EmptyResidual_FallsBackToP()
    {
        // q dominates p everywhere on the support → residual is empty → fall back to sampling p.
        float[] p = [0.25f, 0.25f, 0.25f, 0.25f];
        float[] q = [0.25f, 0.25f, 0.25f, 0.25f];
        var rng = new Random(5);
        int tok = Sampler.ResampleResidual(p, q, rng);
        Assert.InRange(tok, 0, p.Length - 1);   // a valid token from p, not a sentinel
    }

    [Fact]
    public void ResampleResidual_MismatchedLengths_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            Sampler.ResampleResidual(new float[4], new float[3], new Random(1)));
    }
}
