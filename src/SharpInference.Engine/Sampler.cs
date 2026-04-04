namespace SharpInference.Engine;

/// <summary>
/// Token sampler: greedy, temperature, top-k, top-p (nucleus), and min-p.
/// Operates on a logits span and returns the next token ID.
/// </summary>
public static class Sampler
{
    /// <summary>
    /// Sample a token from logits using the given sampling parameters.
    /// </summary>
    public static int Sample(ReadOnlySpan<float> logits, SamplingParams p, Random? rng = null)
    {
        if (p.Temperature <= 0f)
            return Greedy(logits);

        rng ??= Random.Shared;
        int vocabSize = logits.Length;

        // Copy logits so we can modify them
        Span<float> probs = vocabSize <= 4096
            ? stackalloc float[vocabSize]
            : new float[vocabSize];
        logits.CopyTo(probs);

        // Apply logit bias (additive in logit space, before temperature scaling)
        if (p.LogitBias is { Count: > 0 })
        {
            foreach (var (id, bias) in p.LogitBias)
                if ((uint)id < (uint)vocabSize)
                    probs[id] += bias;
        }

        // Repetition penalty (applied in logit space before temperature)
        if (p.RepetitionPenalty != 1.0f && p.PreviousTokens is { Count: > 0 })
        {
            foreach (int id in p.PreviousTokens)
            {
                if ((uint)id < (uint)vocabSize)
                {
                    // Positive logits are divided; negative logits are multiplied
                    if (probs[id] > 0f)
                        probs[id] /= p.RepetitionPenalty;
                    else
                        probs[id] *= p.RepetitionPenalty;
                }
            }
        }

        // Temperature scaling
        if (p.Temperature != 1.0f)
        {
            float invTemp = 1.0f / p.Temperature;
            for (int i = 0; i < vocabSize; i++)
                probs[i] *= invTemp;
        }

        // Softmax
        Softmax(probs);

        // Top-k filtering
        if (p.TopK > 0 && p.TopK < vocabSize)
            ApplyTopK(probs, p.TopK);

        // Min-p filtering
        if (p.MinP > 0f)
            ApplyMinP(probs, p.MinP);

        // Top-p (nucleus) filtering
        if (p.TopP < 1.0f && p.TopP > 0f)
            ApplyTopP(probs, p.TopP);

        // Renormalize after filtering
        Normalize(probs);

        // Sample from the distribution
        return SampleFromDistribution(probs, rng);
    }

    /// <summary>
    /// Greedy decoding: return the token with the highest logit.
    /// </summary>
    public static int Greedy(ReadOnlySpan<float> logits)
    {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > maxVal)
            {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static void Softmax(Span<float> x)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        float sum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < x.Length; i++)
            x[i] *= invSum;
    }

    /// <summary>
    /// Zero out all but the top-k highest probability tokens.
    /// </summary>
    private static void ApplyTopK(Span<float> probs, int k)
    {
        // Find the k-th largest value
        // For correctness (not speed), use a simple partial sort approach
        float kthValue = FindKthLargest(probs, k);

        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < kthValue)
                probs[i] = 0f;
        }
    }

    /// <summary>
    /// Zero out tokens whose probability is less than minP * max_prob.
    /// </summary>
    private static void ApplyMinP(Span<float> probs, float minP)
    {
        float maxProb = 0;
        for (int i = 0; i < probs.Length; i++)
            if (probs[i] > maxProb) maxProb = probs[i];

        float threshold = minP * maxProb;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] < threshold)
                probs[i] = 0f;
        }
    }

    /// <summary>
    /// Keep only the smallest set of tokens whose cumulative probability >= topP.
    /// </summary>
    private static void ApplyTopP(Span<float> probs, float topP)
    {
        // Build index-probability pairs and sort descending
        var indexed = new (int idx, float prob)[probs.Length];
        for (int i = 0; i < probs.Length; i++)
            indexed[i] = (i, probs[i]);

        Array.Sort(indexed, (a, b) => b.prob.CompareTo(a.prob));

        // Find cutoff
        float cumSum = 0;
        int cutoff = indexed.Length;
        for (int i = 0; i < indexed.Length; i++)
        {
            cumSum += indexed[i].prob;
            if (cumSum >= topP)
            {
                cutoff = i + 1;
                break;
            }
        }

        // Zero out tokens beyond the cutoff
        for (int i = cutoff; i < indexed.Length; i++)
            probs[indexed[i].idx] = 0f;
    }

    private static void Normalize(Span<float> probs)
    {
        float sum = 0;
        for (int i = 0; i < probs.Length; i++)
            sum += probs[i];

        if (sum > 0f && sum != 1.0f)
        {
            float invSum = 1.0f / sum;
            for (int i = 0; i < probs.Length; i++)
                probs[i] *= invSum;
        }
    }

    private static int SampleFromDistribution(ReadOnlySpan<float> probs, Random rng)
    {
        float r = (float)rng.NextDouble();
        float cumSum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (r <= cumSum)
                return i;
        }
        // Fallback: return last non-zero token (rounding errors)
        return probs.Length - 1;
    }

    /// <summary>
    /// Find the k-th largest value in the span (1-indexed: k=1 is the largest).
    /// Simple O(n*k) selection — fine for Phase 1 correctness.
    /// </summary>
    private static float FindKthLargest(ReadOnlySpan<float> data, int k)
    {
        // Collect top-k values
        Span<float> topK = k <= 256
            ? stackalloc float[k]
            : new float[k];
        topK.Fill(float.NegativeInfinity);

        for (int i = 0; i < data.Length; i++)
        {
            float val = data[i];
            if (val > topK[k - 1])
            {
                topK[k - 1] = val;
                // Insertion sort to maintain descending order
                for (int j = k - 1; j > 0 && topK[j] > topK[j - 1]; j--)
                    (topK[j], topK[j - 1]) = (topK[j - 1], topK[j]);
            }
        }

        return topK[k - 1];
    }
}

public sealed record SamplingParams
{
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 0;
    public float TopP { get; init; } = 1.0f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int MaxNewTokens { get; init; } = 512;
    public int[]? StopTokenIds { get; init; }

    /// <summary>
    /// Additive logit bias applied before temperature scaling.
    /// Maps token IDs to bias values in the range [-100, 100].
    /// Use -100 to effectively prevent a token; +100 to strongly favour it.
    /// </summary>
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }

    /// <summary>
    /// Recently generated token IDs for repetition penalty.
    /// Typically a sliding window of the last N generated tokens.
    /// </summary>
    public IReadOnlyList<int>? PreviousTokens { get; init; }
}
