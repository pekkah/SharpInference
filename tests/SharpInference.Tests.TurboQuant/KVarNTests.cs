using SharpInference.TurboQuant;

namespace SharpInference.Tests.TurboQuant;

/// <summary>
/// Unit tests for the KVarN quantizer core (issue #180, P0 CPU reference):
/// Sinkhorn variance normalization, asymmetric RTN round-trip, and the fused
/// dequant-dot / value-aggregate fold math.
/// </summary>
public sealed class KVarNTests
{
    private const int HeadDim = 128;
    private const int T = 128;

    private static float[] RandomMatrix(int t, int d, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var m = new float[t * d];
        for (int i = 0; i < m.Length; i++)
            m[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return m;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static float RelL2(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        float num = 0f, den = 0f;
        for (int i = 0; i < reference.Length; i++)
        {
            float diff = reference[i] - actual[i];
            num += diff * diff;
            den += reference[i] * reference[i];
        }
        return MathF.Sqrt(num / MathF.Max(den, 1e-12f));
    }

    [Fact]
    public void Sinkhorn_ReconstructsExactly()
    {
        var y = RandomMatrix(T, HeadDim, 1, scale: 3f);
        var original = (float[])y.Clone();
        var cscale = new float[HeadDim];
        var rscale = new float[T];

        KVarN.Sinkhorn(y, T, HeadDim, cscale, rscale, KVarN.DefaultSinkhornIters);

        // The normalization is a pure dual-axis rescaling, so it inverts up to
        // float rounding accumulated over the alternation iterations.
        for (int t = 0; t < T; t++)
            for (int d = 0; d < HeadDim; d++)
                Assert.Equal(original[t * HeadDim + d],
                    y[t * HeadDim + d] * cscale[d] * rscale[t], 2);
    }

    [Fact]
    public void Sinkhorn_RowsAreUnitRms()
    {
        // The row (per-token) pass runs last, so every row ends at unit RMS.
        var y = RandomMatrix(T, HeadDim, 7, scale: 5f);
        var cscale = new float[HeadDim];
        var rscale = new float[T];

        KVarN.Sinkhorn(y, T, HeadDim, cscale, rscale, KVarN.DefaultSinkhornIters);

        for (int t = 0; t < T; t++)
        {
            float sumSq = 0f;
            for (int d = 0; d < HeadDim; d++)
            {
                float v = y[t * HeadDim + d];
                sumSq += v * v;
            }
            Assert.Equal(1f, MathF.Sqrt(sumSq / HeadDim), 2);
        }
    }

    [Fact]
    public void Sinkhorn_EqualizesChannelVariance()
    {
        // Inject a 50x outlier channel; normalization should collapse the
        // per-channel RMS spread dramatically.
        var y = RandomMatrix(T, HeadDim, 11);
        for (int t = 0; t < T; t++) y[t * HeadDim + 5] *= 50f;

        float SpreadOf(float[] m)
        {
            float min = float.PositiveInfinity, max = 0f;
            for (int d = 0; d < HeadDim; d++)
            {
                float sumSq = 0f;
                for (int t = 0; t < T; t++) { float v = m[t * HeadDim + d]; sumSq += v * v; }
                float rms = MathF.Sqrt(sumSq / T);
                if (rms < min) min = rms;
                if (rms > max) max = rms;
            }
            return max / MathF.Max(min, 1e-9f);
        }

        float before = SpreadOf((float[])y.Clone());
        var cscale = new float[HeadDim];
        var rscale = new float[T];
        KVarN.Sinkhorn(y, T, HeadDim, cscale, rscale, KVarN.DefaultSinkhornIters);
        float after = SpreadOf(y);

        Assert.True(after < before / 5f,
            $"Channel RMS spread not reduced enough: before={before:F1} after={after:F1}");
    }

    [Fact]
    public void KeyTile_RoundTrip_Is4BitAccurate()
    {
        var src = RandomMatrix(T, HeadDim, 21);
        var sign = KVarN.GenerateSignPattern(HeadDim, 3);
        var tile = KVarN.CompressKeyTile(src, T, HeadDim, sign);

        float totalErr = 0f;
        var recon = new float[HeadDim];
        for (int t = 0; t < T; t++)
        {
            KVarN.ReconstructKey(tile, t, sign, recon);
            totalErr += RelL2(src.AsSpan(t * HeadDim, HeadDim), recon);
        }
        float avg = totalErr / T;
        Assert.True(avg < 0.12f, $"4-bit key round-trip rel-L2 too high: {avg:F4}");
    }

    [Fact]
    public void ValueTile_RoundTrip_Is2BitReasonable()
    {
        var src = RandomMatrix(T, HeadDim, 22);
        var sign = KVarN.GenerateSignPattern(HeadDim, 4);
        var tile = KVarN.CompressValueTile(src, T, HeadDim, sign);

        float totalErr = 0f;
        var recon = new float[HeadDim];
        for (int t = 0; t < T; t++)
        {
            KVarN.ReconstructValue(tile, t, sign, recon);
            totalErr += RelL2(src.AsSpan(t * HeadDim, HeadDim), recon);
        }
        float avg = totalErr / T;
        // 2-bit (4 levels) on a near-Gaussian rotated vector has an expected
        // per-vector rel-L2 of ~0.5 (step/√12 ≈ 0.58σ). KVarN's accuracy comes
        // from error cancellation during the weighted V-aggregate, not from
        // per-vector reconstruction — this only guards against a gross regression.
        Assert.True(avg < 0.6f, $"2-bit value round-trip rel-L2 too high: {avg:F4}");
    }

    [Fact]
    public void KScore_MatchesReconstructedDot()
    {
        // The fused fold must reproduce q·k computed against the quantized
        // reconstruction, exactly (up to fp rounding).
        var src = RandomMatrix(T, HeadDim, 31);
        var sign = KVarN.GenerateSignPattern(HeadDim, 5);
        var tile = KVarN.CompressKeyTile(src, T, HeadDim, sign);

        var query = RandomMatrix(1, HeadDim, 32);
        var rotated = new float[HeadDim];
        KVarN.Rotate(query, rotated, sign, HeadDim);

        var scores = new float[T];
        KVarN.KScore(tile, rotated, attnScale: 1f, scores);

        var recon = new float[HeadDim];
        for (int t = 0; t < T; t++)
        {
            KVarN.ReconstructKey(tile, t, sign, recon);
            float expected = Dot(query, recon);
            Assert.Equal(expected, scores[t], 2);
        }
    }

    [Fact]
    public void VAggregate_MatchesReconstructedWeightedSum()
    {
        var src = RandomMatrix(T, HeadDim, 41);
        var sign = KVarN.GenerateSignPattern(HeadDim, 6);
        var tile = KVarN.CompressValueTile(src, T, HeadDim, sign);

        var rng = new Random(42);
        var weights = new float[T];
        for (int t = 0; t < T; t++) weights[t] = (float)rng.NextDouble();

        var fused = new float[HeadDim];
        KVarN.VAggregate(tile, weights, sign, fused);

        // Brute-force reference: Σ_t w[t]·reconstruct(t).
        var expected = new float[HeadDim];
        var recon = new float[HeadDim];
        for (int t = 0; t < T; t++)
        {
            KVarN.ReconstructValue(tile, t, sign, recon);
            for (int d = 0; d < HeadDim; d++)
                expected[d] += weights[t] * recon[d];
        }

        Assert.True(RelL2(expected, fused) < 1e-3f,
            $"VAggregate fold drift: relL2={RelL2(expected, fused):E2}");
    }

    [Fact]
    public void TileCodes_PackUnpack_Boundaries()
    {
        var key = new KVarNTile(2, HeadDim, perChannel: true);
        var val = new KVarNTile(2, HeadDim, perChannel: false);
        for (int d = 0; d < HeadDim; d++)
        {
            key.SetKeyCode(0, d, d % 16);
            key.SetKeyCode(1, d, 15 - (d % 16));
            val.SetValueCode(0, d, d % 4);
            val.SetValueCode(1, d, 3 - (d % 4));
        }
        for (int d = 0; d < HeadDim; d++)
        {
            Assert.Equal(d % 16, key.GetKeyCode(0, d));
            Assert.Equal(15 - (d % 16), key.GetKeyCode(1, d));
            Assert.Equal(d % 4, val.GetValueCode(0, d));
            Assert.Equal(3 - (d % 4), val.GetValueCode(1, d));
        }
    }
}
