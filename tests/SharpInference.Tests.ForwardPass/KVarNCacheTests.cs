using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Cache-level accuracy tests for <see cref="KVarNKvCache"/> (issue #180, P0).
/// Validates that 4-bit-key / 2-bit-value compression preserves enough fidelity
/// for (a) needle retrieval through the key-score path and (b) full attention
/// output through the value-aggregate path, both against an exact FP32 baseline.
/// </summary>
public sealed class KVarNCacheTests
{
    private const int HeadDim = 128;

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static float[] Normalize(float[] v)
    {
        float mag = MathF.Sqrt(Dot(v, v));
        var o = new float[v.Length];
        for (int i = 0; i < v.Length; i++) o[i] = v[i] / mag;
        return o;
    }

    /// <summary>Raw (pre-softmax) attention scores across the whole cache.</summary>
    private static unsafe float[] RawScores(KVarNKvCache cache, float[] query)
    {
        int seqLen = cache.Length;
        int tqLen = cache.GetTqLength(0);
        var rotated = new float[HeadDim];
        cache.RotateQueryKey(0, 0, query, rotated);

        var scores = new float[seqLen];
        fixed (float* qPtr = rotated)
            cache.ComputeKScores(0, 0, new ReadOnlySpan<float>(qPtr, HeadDim), 1.0f,
                scores.AsSpan(0, tqLen));

        for (int pos = tqLen; pos < seqLen; pos++)
        {
            float* k = cache.Fp32KeyAt(0, pos);
            scores[pos] = Dot(new ReadOnlySpan<float>(k, HeadDim), query);
        }
        return scores;
    }

    private static void RunNeedle(int contextLen, int needlePos, int fp32Window = 256)
    {
        int maxSeqLen = contextLen + KVarN_TileSlack;
        using var cache = new KVarNKvCache(
            numLayers: 1, maxSeqLen: maxSeqLen, numKvHeads: 1, headDim: HeadDim,
            fp32WindowSize: fp32Window);

        var rng = new Random(2024 + contextLen);
        var needleKey = Normalize(RandomVec(rng));
        var needleValue = new float[HeadDim];
        needleValue[0] = 1f;

        for (int pos = 0; pos < contextLen; pos++)
        {
            if (pos == needlePos)
                cache.Append(0, needleKey, needleValue);
            else
                cache.Append(0, Normalize(RandomVec(rng)), Normalize(RandomVec(rng)));
            cache.IncrementPosition();
        }

        Assert.True(cache.GetTqLength(0) > 0, "Needle should be in the compressed region");
        Assert.True(needlePos < cache.GetTqLength(0),
            $"needlePos={needlePos} must be compressed (tqLen={cache.GetTqLength(0)})");

        var scores = RawScores(cache, (float[])needleKey.Clone());

        var sorted = (float[])scores.Clone();
        Array.Sort(sorted);
        float median = sorted[sorted.Length / 2];
        float needleScore = scores[needlePos];

        Assert.True(needleScore > median + 0.3f,
            $"Needle score ({needleScore:F4}) not above median ({median:F4})");

        int topK = Math.Max(1, contextLen / 100);
        float threshold = sorted[sorted.Length - topK];
        Assert.True(needleScore >= threshold,
            $"Needle score ({needleScore:F4}) not in top-1% (threshold={threshold:F4})");
    }

    private const int KVarN_TileSlack = 128;

    private static float[] RandomVec(Random rng)
    {
        var v = new float[HeadDim];
        for (int d = 0; d < HeadDim; d++) v[d] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    [Fact]
    public void Needle_At_1K() => RunNeedle(1024, 0);

    [Fact]
    public void Needle_At_2K() => RunNeedle(2048, 0);

    [Fact]
    public void Needle_At_4K() => RunNeedle(4096, 0);

    [Fact]
    public void Needle_AtMiddle_4K() => RunNeedle(4096, 2048 - 256 - 1);

    /// <summary>
    /// Full attention parity: run scores → softmax → value-aggregate through the
    /// KVarN cache and compare the output vector to exact FP32 attention over the
    /// same original K/V. The 2-bit value error largely cancels in the weighted
    /// sum, so the directions should stay highly aligned.
    /// </summary>
    [Fact]
    public unsafe void Attention_Output_MatchesFp32()
    {
        const int N = 600;
        const int fp32Window = 64;
        using var cache = new KVarNKvCache(
            numLayers: 1, maxSeqLen: N + 128, numKvHeads: 1, headDim: HeadDim,
            fp32WindowSize: fp32Window);

        var rng = new Random(7);
        var keys = new float[N][];
        var values = new float[N][];
        for (int i = 0; i < N; i++)
        {
            keys[i] = Normalize(RandomVec(rng));
            values[i] = Normalize(RandomVec(rng));
            cache.Append(0, keys[i], values[i]);
            cache.IncrementPosition();
        }
        Assert.True(cache.GetTqLength(0) >= 256, "Most positions should be compressed");

        // Query correlated with a handful of keys so the softmax is peaked.
        var query = new float[HeadDim];
        for (int i = 100; i < 110; i++)
            for (int d = 0; d < HeadDim; d++) query[d] += keys[i][d];
        query = Normalize(query);

        float scale = 1.0f / MathF.Sqrt(HeadDim);

        // --- Exact FP32 reference ---
        var refScores = new float[N];
        for (int t = 0; t < N; t++) refScores[t] = Dot(query, keys[t]) * scale;
        Softmax(refScores);
        var refOut = new float[HeadDim];
        for (int t = 0; t < N; t++)
            for (int d = 0; d < HeadDim; d++) refOut[d] += refScores[t] * values[t][d];

        // --- KVarN cache path ---
        int tqLen = cache.GetTqLength(0);
        var rotated = new float[HeadDim];
        cache.RotateQueryKey(0, 0, query, rotated);
        var scores = new float[N];
        fixed (float* qPtr = rotated)
            cache.ComputeKScores(0, 0, new ReadOnlySpan<float>(qPtr, HeadDim), scale,
                scores.AsSpan(0, tqLen));
        for (int t = tqLen; t < N; t++)
        {
            float* k = cache.Fp32KeyAt(0, t);
            scores[t] = Dot(new ReadOnlySpan<float>(k, HeadDim), query) * scale;
        }
        Softmax(scores);
        var outv = new float[HeadDim];
        cache.ComputeVAggregation(0, 0, scores.AsSpan(0, tqLen), outv);
        for (int t = tqLen; t < N; t++)
        {
            float* v = cache.Fp32ValueAt(0, t);
            for (int d = 0; d < HeadDim; d++) outv[d] += scores[t] * v[d];
        }

        float cos = Dot(refOut, outv) / (MathF.Sqrt(Dot(refOut, refOut)) * MathF.Sqrt(Dot(outv, outv)));
        Assert.True(cos > 0.9f, $"KVarN attention output cosine vs FP32 too low: {cos:F4}");
    }

    private static void Softmax(float[] s)
    {
        float max = float.NegativeInfinity;
        foreach (var x in s) if (x > max) max = x;
        float sum = 0f;
        for (int i = 0; i < s.Length; i++) { s[i] = MathF.Exp(s[i] - max); sum += s[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < s.Length; i++) s[i] *= inv;
    }
}
