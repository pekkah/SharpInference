using SharpInference.TurboQuant;
using Xunit.Abstractions;

namespace SharpInference.Tests.TurboQuant;

/// <summary>
/// AVX2-vs-scalar parity for the fused KVarN read kernels (issue #180 P1).
/// The AVX2 kernels compute the same algebra as the scalar reference but
/// reassociate the floating-point accumulation (chunked Vector256 lanes vs a
/// sequential channel/token walk), so parity is asserted relative to the
/// tile's max |result| rather than element-exact. The scalar path is forced
/// via <see cref="KVarNCompressor.ForceScalar"/>; on hardware without AVX2
/// both invocations take the scalar path and the comparison is vacuous (but
/// still runs).
/// </summary>
public sealed class KVarNSimdParityTests(ITestOutputHelper output)
{
    private const int T = KVarNCompressor.TileTokens;
    private const float RelTolerance = 1e-5f;

    // ─────────────────────────────────────────────────────────────────────────
    // KeyScores
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(8, false)]
    [InlineData(64, false)]
    [InlineData(128, false)]
    [InlineData(128, true)] // token/channel outliers — the KVarN target regime
    [InlineData(256, false)]
    [InlineData(512, false)] // upper vectorized dims stay covered
    public void KeyScores_Avx2MatchesScalar(int dim, bool plantOutliers)
    {
        var c = new KVarNCompressor(dim);
        var rng = new Random(dim * 1000 + (plantOutliers ? 1 : 0));

        float[] keys = GaussianMatrix(T, dim, rng);
        if (plantOutliers)
        {
            for (int ch = 0; ch < dim; ch++) keys[41 * dim + ch] *= 100f;
            for (int t = 0; t < T; t++) keys[t * dim + 9] *= 50f;
        }

        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);

        float[] qRot = new float[dim];
        float[] scoresSimd = new float[T];
        float[] scoresScalar = new float[T];

        float worstRel = 0f;
        for (int qi = 0; qi < 8; qi++)
        {
            float[] q = GaussianVector(dim, rng);
            // Planted exact zeros exercise the qw == 0 skip path in both kernels.
            q[3] = 0f;
            if (dim > 17) q[17] = 0f;
            c.RotateQuery(q, qRot);

            c.KeyScores(tile, qRot, scoresSimd); // dispatch (AVX2 where supported)
            RunForcedScalar(() => c.KeyScores(tile, qRot, scoresScalar));

            float maxAbs = 0f, maxDiff = 0f;
            for (int t = 0; t < T; t++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(scoresScalar[t]));
                maxDiff = MathF.Max(maxDiff, MathF.Abs(scoresSimd[t] - scoresScalar[t]));
            }
            float rel = maxDiff / MathF.Max(1f, maxAbs);
            worstRel = MathF.Max(worstRel, rel);
        }

        output.WriteLine($"KeyScores D={dim} outliers={plantOutliers}: worst relative deviation {worstRel:E3}");
        Assert.True(worstRel <= RelTolerance,
            $"AVX2 KeyScores deviates from scalar by {worstRel:E3} (> {RelTolerance:E1}) at D={dim}");
    }

    [Fact]
    public void KeyScores_AllChannelsConstant_BothPathsSkipEverything()
    {
        // Identical token rows → every rotated channel is constant across the
        // tile → all chanStep == 0 → both kernels reduce to rowScale·bias.
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        var rng = new Random(97);

        float[] row = GaussianVector(dim, rng);
        float[] keys = new float[T * dim];
        for (int t = 0; t < T; t++)
            row.CopyTo(keys, t * dim);

        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);

        float[] qRot = new float[dim];
        c.RotateQuery(GaussianVector(dim, rng), qRot);

        float[] scoresSimd = new float[T];
        float[] scoresScalar = new float[T];
        c.KeyScores(tile, qRot, scoresSimd);
        RunForcedScalar(() => c.KeyScores(tile, qRot, scoresScalar));

        float maxAbs = 0f, maxDiff = 0f;
        for (int t = 0; t < T; t++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(scoresScalar[t]));
            maxDiff = MathF.Max(maxDiff, MathF.Abs(scoresSimd[t] - scoresScalar[t]));
        }
        output.WriteLine($"all-constant-channel tile: max |scalar| {maxAbs:E3}, max diff {maxDiff:E3}");
        Assert.True(maxDiff <= RelTolerance * MathF.Max(1f, maxAbs));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AggregateValues
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(64, false)]
    [InlineData(128, false)]
    [InlineData(128, true)]
    [InlineData(256, false)] // two-group (G=2) path
    [InlineData(512, false)] // four-group path, upper vectorized dims
    public void AggregateValues_Avx2MatchesScalar(int dim, bool plantOutliers)
    {
        var c = new KVarNCompressor(dim);
        var rng = new Random(dim * 2000 + (plantOutliers ? 1 : 0));

        float[] values = GaussianMatrix(T, dim, rng);
        if (plantOutliers)
        {
            for (int ch = 0; ch < dim; ch++) values[13 * dim + ch] *= 100f;
            for (int t = 0; t < T; t++) values[t * dim + 27] *= 50f;
        }
        // All-zero token row → tokStep == tokMin == 0 → the ws == 0 skip path.
        Array.Clear(values, 77 * dim, dim);

        byte[] tile = new byte[c.ValueTileBytes];
        c.CompressValueTile(values, tile);

        float worstRel = 0f;
        for (int trial = 0; trial < 8; trial++)
        {
            float[] weights = SoftmaxWeights(T, rng);
            // Exact-zero weights exercise the w == 0 skip path in both kernels.
            weights[0] = 0f;
            weights[63] = 0f;
            weights[127] = 0f;

            // Pre-seeded accumulators (same seed) so the += contract is compared too.
            float[] seed = GaussianVector(dim, rng);
            float[] accSimd = (float[])seed.Clone();
            float[] accScalar = (float[])seed.Clone();

            c.AggregateValues(tile, weights, accSimd);
            RunForcedScalar(() => c.AggregateValues(tile, weights, accScalar));

            float maxAbs = 0f, maxDiff = 0f;
            for (int ch = 0; ch < dim; ch++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(accScalar[ch]));
                maxDiff = MathF.Max(maxDiff, MathF.Abs(accSimd[ch] - accScalar[ch]));
            }
            worstRel = MathF.Max(worstRel, maxDiff / MathF.Max(1f, maxAbs));
        }

        output.WriteLine($"AggregateValues D={dim} outliers={plantOutliers}: worst relative deviation {worstRel:E3}");
        Assert.True(worstRel <= RelTolerance,
            $"AVX2 AggregateValues deviates from scalar by {worstRel:E3} (> {RelTolerance:E1}) at D={dim}");
    }

    [Theory]
    [InlineData(8)]
    [InlineData(32)]
    public void AggregateValues_SmallDims_DispatchFallsBackAndMatches(int dim)
    {
        // d < 64 always takes the scalar kernel (the AVX2 unpack needs 64-channel
        // strides); this pins the dispatch producing identical results either way.
        var c = new KVarNCompressor(dim);
        var rng = new Random(dim);

        float[] values = GaussianMatrix(T, dim, rng);
        byte[] tile = new byte[c.ValueTileBytes];
        c.CompressValueTile(values, tile);

        float[] weights = SoftmaxWeights(T, rng);
        float[] accDefault = new float[dim];
        float[] accScalar = new float[dim];

        c.AggregateValues(tile, weights, accDefault);
        RunForcedScalar(() => c.AggregateValues(tile, weights, accScalar));

        for (int ch = 0; ch < dim; ch++)
            Assert.Equal(accScalar[ch], accDefault[ch]); // bit-identical: same kernel
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunForcedScalar(Action action)
    {
        KVarNCompressor.ForceScalar = true;
        try
        {
            action();
        }
        finally
        {
            KVarNCompressor.ForceScalar = false;
        }
    }

    private static float[] GaussianMatrix(int rows, int cols, Random rng)
    {
        float[] m = new float[rows * cols];
        for (int i = 0; i < m.Length; i++)
            m[i] = NextGaussian(rng);
        return m;
    }

    private static float[] GaussianVector(int dim, Random rng)
    {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++)
            v[i] = NextGaussian(rng);
        return v;
    }

    private static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private static float[] SoftmaxWeights(int count, Random rng)
    {
        float[] w = new float[count];
        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            w[i] = MathF.Exp(NextGaussian(rng));
            sum += w[i];
        }
        for (int i = 0; i < count; i++)
            w[i] = (float)(w[i] / sum);
        return w;
    }
}
