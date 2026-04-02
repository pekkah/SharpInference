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
}
