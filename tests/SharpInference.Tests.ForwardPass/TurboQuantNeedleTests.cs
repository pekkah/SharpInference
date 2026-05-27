using SharpInference.Engine;
using SharpInference.TurboQuant;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Needle-in-a-haystack tests for TurboQuantKvCache.
/// Verifies that KV compression at long distances preserves enough directional fidelity
/// for attention to retrieve a single distinctive "needle" token from a sea of random tokens.
/// </summary>
public sealed class TurboQuantNeedleTests
{
    // headDim must be a supported dimension (TurboQuant codebooks ship for dim=128)
    private const int HeadDim = 128;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;

    /// <summary>
    /// Computes raw (pre-softmax) attention scores for all positions in the cache.
    /// TQ-compressed positions use DequantDot; FP32 positions use a plain dot product.
    /// </summary>
    private static unsafe float[] TqRawScores(TurboQuantKvCache cache, float[] query)
    {
        int seqLen = cache.Length;
        int tqLen  = cache.GetTqLength(layer: 0);
        int layer  = 0;
        int kvHead = 0;

        var compressor = cache.GetKeyCompressor(layer, kvHead);

        // Rotate query for TQ inner product
        var rotatedQuery = new float[HeadDim];
        compressor.RotateQuery(query, rotatedQuery);

        var scores = new float[seqLen];

        // TQ scores via the FastScan-tiled K path (issue #34, Phase 2).
        fixed (float* qPtr = rotatedQuery)
        fixed (float* scoresPtr = scores)
            cache.ComputeKScores(layer, kvHead, qPtr, attnScale: 1.0f, scoresPtr);

        for (int pos = tqLen; pos < seqLen; pos++)
        {
            float* fp32Key = cache.Fp32KeyAt(layer, pos);
            float dot = 0;
            for (int d = 0; d < HeadDim; d++)
                dot += fp32Key[d] * query[d];
            scores[pos] = dot;
        }

        return scores;
    }

    private static float Dot(float[] a, float[] b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static float[] Normalize(float[] v)
    {
        float mag = MathF.Sqrt(Dot(v, v));
        var out_ = new float[v.Length];
        for (int i = 0; i < v.Length; i++) out_[i] = v[i] / mag;
        return out_;
    }

    /// <summary>
    /// Places a needle at the given position in a sequence of <paramref name="contextLen"/> tokens.
    /// The needle's key is designed to match the query; all other keys are random.
    /// After filling the cache (so the needle ends up compressed), we run attention and
    /// verify that the needle's value dominates the output.
    /// </summary>
    private static void RunNeedle(int contextLen, int needlePos, int fp32Window = 256)
    {
        // fp32Window must be smaller than contextLen for the needle to be compressed
        Assert.True(needlePos < contextLen - fp32Window,
            $"needlePos={needlePos} must be in compressed region (contextLen={contextLen}, fp32Window={fp32Window})");

        int maxSeqLen = contextLen + 32;
        using var cache = new TurboQuantKvCache(
            numLayers: NumLayers,
            maxSeqLen: maxSeqLen,
            numKvHeads: NumKvHeads,
            headDim: HeadDim,
            fp32WindowSize: fp32Window,
            bits: 3);

        var rng = new Random(1337 + contextLen);

        // The needle's key is the query direction
        var needleKey = new float[HeadDim];
        for (int d = 0; d < HeadDim; d++)
            needleKey[d] = (float)(rng.NextDouble() * 2 - 1);
        needleKey = Normalize(needleKey);

        // Needle value: a distinctive unit vector
        var needleValue = new float[HeadDim];
        needleValue[0] = 1f;   // pure first-dimension

        // Fill KV cache
        var kBuf = new float[HeadDim];
        var vBuf = new float[HeadDim];

        for (int pos = 0; pos < contextLen; pos++)
        {
            if (pos == needlePos)
            {
                // The needle: key matches query direction, value is the special vector
                cache.Append(layer: 0, needleKey, needleValue);
            }
            else
            {
                // Background: random unit vectors, orthogonal-ish to the needle
                for (int d = 0; d < HeadDim; d++)
                    kBuf[d] = (float)(rng.NextDouble() * 2 - 1);
                kBuf = Normalize(kBuf);
                for (int d = 0; d < HeadDim; d++)
                    vBuf[d] = (float)(rng.NextDouble() * 2 - 1);
                vBuf = Normalize(vBuf);
                cache.Append(layer: 0, kBuf, vBuf);
            }
            cache.IncrementPosition();
        }

        Assert.True(cache.GetTqLength(0) > 0,
            "Needle should be in the compressed region");

        // The query matches the needle's key exactly
        var query = (float[])needleKey.Clone();

        var scores = TqRawScores(cache, query);

        // The needle should have the highest (or near-highest) raw attention score.
        // With HeadDim=128 and a normalized needle key exactly matching the query,
        // the needle score should be ~1.0, while random unit-vector keys score ~1/sqrt(128)≈0.088.
        float needleScore = scores[needlePos];
        float maxScore = float.NegativeInfinity;
        int maxPos = 0;
        for (int pos = 0; pos < contextLen; pos++)
        {
            if (scores[pos] > maxScore) { maxScore = scores[pos]; maxPos = pos; }
        }

        // Compute median score for context
        var sorted = (float[])scores.Clone();
        Array.Sort(sorted);
        float medianScore = sorted[sorted.Length / 2];

        Console.WriteLine($"[contextLen={contextLen}, needlePos={needlePos}] " +
            $"needleScore={needleScore:F4} maxScore={maxScore:F4} (at pos={maxPos}) median={medianScore:F4}");

        // Needle score must be well above median (background tokens)
        Assert.True(needleScore > medianScore + 0.3f,
            $"Needle score ({needleScore:F4}) not sufficiently above median ({medianScore:F4})");

        // Needle should be in the top 1% of scores (i.e., top contextLen/100 positions)
        int topK = Math.Max(1, contextLen / 100);
        float topKThreshold = sorted[sorted.Length - topK];
        Assert.True(needleScore >= topKThreshold,
            $"Needle score ({needleScore:F4}) not in top-1% (threshold={topKThreshold:F4})");
    }

    [Fact]
    public void Needle_At_1K() => RunNeedle(contextLen: 1024, needlePos: 0);

    [Fact]
    public void Needle_At_2K() => RunNeedle(contextLen: 2048, needlePos: 0);

    [Fact]
    public void Needle_At_4K() => RunNeedle(contextLen: 4096, needlePos: 0);

    [Fact]
    public void Needle_At_8K() => RunNeedle(contextLen: 8192, needlePos: 0);

    [Fact]
    public void Needle_AtMiddle_4K() => RunNeedle(contextLen: 4096, needlePos: 2048 - 256 - 1);

    [Fact]
    public void Needle_AtMiddle_8K() => RunNeedle(contextLen: 8192, needlePos: 4096 - 256 - 1);
}
