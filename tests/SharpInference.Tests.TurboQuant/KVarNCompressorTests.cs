using SharpInference.TurboQuant;
using Xunit.Abstractions;

namespace SharpInference.Tests.TurboQuant;

public sealed class KVarNCompressorTests(ITestOutputHelper output)
{
    private const int T = KVarNCompressor.TileTokens;

    // ─────────────────────────────────────────────────────────────────────────
    // Layout / construction
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void TileSizes_MatchLayoutFormula()
    {
        // K: 4·T + 8·D + D·T/2   V: 4·D + 8·T·G + T·D/4
        Assert.Equal(4 * 128 + 8 * 128 + 128 * 64, KVarNCompressor.KeyTileSize(128));   // 9728
        Assert.Equal(4 * 128 + 8 * 128 + 128 * 32, KVarNCompressor.ValueTileSize(128)); // 5632
        Assert.Equal(4 * 128 + 8 * 64 + 64 * 64, KVarNCompressor.KeyTileSize(64));      // 5120
        Assert.Equal(4 * 64 + 8 * 128 + 128 * 16, KVarNCompressor.ValueTileSize(64));   // 3328
        Assert.Equal(4 * 256 + 8 * 128 * 2 + 128 * 64, KVarNCompressor.ValueTileSize(256)); // 11264

        var c = new KVarNCompressor(128);
        Assert.Equal(KVarNCompressor.KeyTileSize(128), c.KeyTileBytes);
        Assert.Equal(KVarNCompressor.ValueTileSize(128), c.ValueTileBytes);
    }

    [Theory]
    [InlineData(100)] // not a power of 2
    [InlineData(4)]   // too small
    [InlineData(2048)] // too large
    public void InvalidHeadDim_Throws(int headDim)
    {
        Assert.Throws<ArgumentException>(() => new KVarNCompressor(headDim));
        Assert.Throws<ArgumentException>(() => KVarNCompressor.KeyTileSize(headDim));
    }

    [Fact]
    public void UndersizedBuffers_Throw()
    {
        var c = new KVarNCompressor(128);
        var keys = new float[T * 128];
        Assert.Throws<ArgumentException>(() => c.CompressKeyTile(keys, new byte[c.KeyTileBytes - 1]));
        Assert.Throws<ArgumentException>(() => c.CompressValueTile(keys, new byte[c.ValueTileBytes - 1]));
        Assert.Throws<ArgumentException>(() => c.CompressKeyTile(new float[T * 128 - 1], new byte[c.KeyTileBytes]));
    }

    [Fact]
    public void BitPacking2_RoundTrip_AllValuesAndMixed()
    {
        byte[] buffer = new byte[BitPacking.PackedBytes2Bit];
        for (int value = 0; value < 4; value++)
        {
            for (int pos = 0; pos < 128; pos++)
                BitPacking.PackBits2(buffer, 0, pos, value);
            for (int pos = 0; pos < 128; pos++)
                Assert.Equal(value, BitPacking.UnpackBits2(buffer, 0, pos));
        }

        var rng = new Random(5);
        int[] expected = new int[128];
        for (int pos = 0; pos < 128; pos++)
        {
            expected[pos] = rng.Next(4);
            BitPacking.PackBits2(buffer, 0, pos, expected[pos]);
        }
        for (int pos = 0; pos < 128; pos++)
            Assert.Equal(expected[pos], BitPacking.UnpackBits2(buffer, 0, pos));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Sinkhorn convergence
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Sinkhorn_GaussianTile_RowAndColumnRmsNearUnit()
    {
        const int dim = 128;
        float[] tile = GaussianMatrix(T, dim, seed: 42);
        float[] original = (float[])tile.Clone();
        float[] rowScale = new float[T];
        float[] colScale = new float[dim];

        KVarNCompressor.SinkhornNormalize(tile, T, dim, rowScale, colScale, iterations: 4);

        (float rowLo, float rowHi, float colLo, float colHi) = AxisRmsRange(tile, T, dim);
        output.WriteLine($"Gaussian tile after 4 iters: row RMS [{rowLo:F4}, {rowHi:F4}], col RMS [{colLo:F4}, {colHi:F4}]");

        Assert.InRange(rowLo, 0.9f, 1.1f);
        Assert.InRange(rowHi, 0.9f, 1.1f);
        Assert.InRange(colLo, 0.9f, 1.1f);
        Assert.InRange(colHi, 0.9f, 1.1f);

        AssertScalesReconstruct(original, tile, rowScale, colScale, T, dim);
    }

    [Fact]
    public void Sinkhorn_OutlierTile_RowAndColumnRmsNearUnit()
    {
        const int dim = 128;
        float[] tile = GaussianMatrix(T, dim, seed: 7);

        // Plant an outlier channel, an outlier token, and one huge lone element.
        for (int t = 0; t < T; t++) tile[t * dim + 9] *= 100f;
        for (int c = 0; c < dim; c++) tile[41 * dim + c] *= 100f;
        tile[3 * dim + 5] = 1000f;

        float[] original = (float[])tile.Clone();
        float[] rowScale = new float[T];
        float[] colScale = new float[dim];

        KVarNCompressor.SinkhornNormalize(tile, T, dim, rowScale, colScale, iterations: 5);

        (float rowLo, float rowHi, float colLo, float colHi) = AxisRmsRange(tile, T, dim);
        output.WriteLine($"Outlier tile after 5 iters: row RMS [{rowLo:F4}, {rowHi:F4}], col RMS [{colLo:F4}, {colHi:F4}]");

        Assert.InRange(rowLo, 0.9f, 1.1f);
        Assert.InRange(rowHi, 0.9f, 1.1f);
        Assert.InRange(colLo, 0.9f, 1.1f);
        Assert.InRange(colHi, 0.9f, 1.1f);

        AssertScalesReconstruct(original, tile, rowScale, colScale, T, dim);
    }

    [Fact]
    public void Sinkhorn_AllZerosTile_LeavesScalesAtOne()
    {
        const int dim = 64;
        float[] tile = new float[T * dim];
        float[] rowScale = new float[T];
        float[] colScale = new float[dim];

        KVarNCompressor.SinkhornNormalize(tile, T, dim, rowScale, colScale, iterations: 4);

        Assert.All(rowScale, s => Assert.Equal(1f, s));
        Assert.All(colScale, s => Assert.Equal(1f, s));
        Assert.All(tile, v => Assert.Equal(0f, v));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Rotation preserves scores
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void HadamardRotation_PreservesDotProducts()
    {
        const int dim = 128;
        var compressor = new KVarNCompressor(dim);
        var rng = new Random(11);

        float maxErr = 0f;
        for (int trial = 0; trial < 32; trial++)
        {
            float[] q = GaussianVector(dim, rng);
            float[] k = GaussianVector(dim, rng);

            float direct = Dot(q, k);

            float[] qRot = new float[dim];
            float[] kRot = new float[dim];
            compressor.RotateQuery(q, qRot);
            WalshHadamard.Transform(k, kRot, dim);

            float rotated = Dot(qRot, kRot);
            maxErr = MathF.Max(maxErr, MathF.Abs(rotated - direct));
        }

        output.WriteLine($"(HQ)·(HK) vs Q·K max abs error over 32 trials: {maxErr:E3}");
        Assert.True(maxErr < 1e-3f, $"Rotation broke scores: max err {maxErr}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Round-trip reconstruction error
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void KeyTile_RoundTrip_RelativeFrobeniusBounded()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        float[] keys = GaussianMatrix(T, dim, seed: 21);

        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);

        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(tile, decoded);

        // Measured ~0.099 — consistent with the ~0.106σ theoretical RMS error of
        // 4-bit min/max RTN over 128 Gaussian samples (step ≈ 5.6σ/15, err ≈ step/√12).
        float relErr = RelativeFrobenius(keys, decoded);
        output.WriteLine($"K 4-bit round-trip relative Frobenius error (Gaussian, D=128): {relErr:F4}");
        Assert.True(relErr < 0.12f, $"K round-trip error too high: {relErr}");
    }

    [Fact]
    public void ValueTile_RoundTrip_RelativeFrobeniusBounded()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        float[] values = GaussianMatrix(T, dim, seed: 22);

        byte[] tile = new byte[c.ValueTileBytes];
        c.CompressValueTile(values, tile);

        float[] decoded = new float[T * dim];
        c.DecompressValueTile(tile, decoded);

        // Measured ~0.50 — the expected floor for 2-bit min/max RTN on Gaussian data
        // (step ≈ 5.6σ/3, err ≈ step/√12 ≈ 0.54σ; even optimal 2-bit Lloyd-Max is 0.34σ).
        // The softmax-aggregate relative error is about the same (~0.46-0.51, see the
        // aggregate test): heavy-tailed softmax weights give a small effective sample
        // count, and the reference aggregate's own norm shrinks by the same averaging.
        // Whether ~0.5 element-level V error is acceptable end-to-end is exactly what
        // the P0 model-level accuracy gate must decide — these unit tests do not.
        float relErr = RelativeFrobenius(values, decoded);
        output.WriteLine($"V 2-bit round-trip relative Frobenius error (Gaussian, D=128): {relErr:F4}");
        Assert.True(relErr < 0.55f, $"V round-trip error too high: {relErr}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Attention-score fidelity vs the existing TurboQuant compressor
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void KeyScores_ErrorComparableToTurboQuant4Bit(bool plantOutliers)
    {
        const int dim = 128;
        const int numQueries = 8;
        var rng = new Random(31);

        float[] keys = GaussianMatrix(T, dim, seed: 33);
        if (plantOutliers)
        {
            // Token-scale outliers (KVarN's target failure mode) + a hot channel.
            var orng = new Random(34);
            for (int i = 0; i < 8; i++)
            {
                int t = orng.Next(T);
                for (int c = 0; c < dim; c++) keys[t * dim + c] *= 20f;
            }
            for (int t = 0; t < T; t++) keys[t * dim + 3] *= 10f;
        }

        // KVarN tile.
        var kvarn = new KVarNCompressor(dim);
        byte[] kTile = new byte[kvarn.KeyTileBytes];
        kvarn.CompressKeyTile(keys, kTile);

        // Existing TurboQuant 4-bit, per-token blocks on the same data.
        var tq = new KvCacheCompressor(4, dim, layerIndex: 0);
        byte[] tqBlocks = new byte[T * tq.BlockSize];
        for (int t = 0; t < T; t++)
            tq.Compress(keys.AsSpan(t * dim, dim), tqBlocks.AsSpan(t * tq.BlockSize, tq.BlockSize));

        double kvarnAbsErr = 0, tqAbsErr = 0;
        int samples = 0;

        float[] scores = new float[T];
        float[] qRotKvarn = new float[dim];
        float[] qRotTq = new float[dim];

        for (int qi = 0; qi < numQueries; qi++)
        {
            float[] q = GaussianVector(dim, rng);
            kvarn.RotateQuery(q, qRotKvarn);
            tq.RotateQuery(q, qRotTq);
            kvarn.KeyScores(kTile, qRotKvarn, scores);

            for (int t = 0; t < T; t++)
            {
                double truth = DotDouble(q, keys.AsSpan(t * dim, dim));
                kvarnAbsErr += Math.Abs(scores[t] - truth);
                tqAbsErr += Math.Abs(tq.DequantDot(tqBlocks.AsSpan(t * tq.BlockSize, tq.BlockSize), qRotTq) - truth);
                samples++;
            }
        }

        double kvarnMean = kvarnAbsErr / samples;
        double tqMean = tqAbsErr / samples;
        output.WriteLine($"Mean |q·k̂ − q·k| ({(plantOutliers ? "outlier" : "Gaussian")} K tile, D=128, {samples} samples):");
        output.WriteLine($"  KVarN K4 (RTN+Sinkhorn):    {kvarnMean:F4}");
        output.WriteLine($"  TurboQuant 4-bit Lloyd-Max: {tqMean:F4}");
        output.WriteLine($"  ratio KVarN/TQ:             {kvarnMean / tqMean:F3}");

        // Measured ratios: 0.98 (Gaussian), 1.05 (outliers) — same ballpark by construction.
        Assert.True(kvarnMean <= tqMean * 1.25,
            $"KVarN score error {kvarnMean:F4} not in the same ballpark as TurboQuant {tqMean:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. DequantDot == dot(query, Decompress(tile))
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void DequantDot_MatchesDecompressedDot()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        var rng = new Random(51);

        float[] keys = GaussianMatrix(T, dim, seed: 52);
        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);

        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(tile, decoded);

        float[] scores = new float[T];
        float[] qRot = new float[dim];
        float maxErr = 0f, maxScoresErr = 0f;

        for (int qi = 0; qi < 4; qi++)
        {
            float[] q = GaussianVector(dim, rng);
            c.RotateQuery(q, qRot);
            c.KeyScores(tile, qRot, scores);

            for (int t = 0; t < T; t++)
            {
                float reference = Dot(q, decoded.AsSpan(t * dim, dim));
                float fused = c.DequantDot(tile, qRot, t);
                maxErr = MathF.Max(maxErr, MathF.Abs(fused - reference));
                maxScoresErr = MathF.Max(maxScoresErr, MathF.Abs(scores[t] - reference));
            }
        }

        output.WriteLine($"max |DequantDot − dot(q, decompressed)|: {maxErr:E3}");
        output.WriteLine($"max |KeyScores  − dot(q, decompressed)|: {maxScoresErr:E3}");
        Assert.True(maxErr < 1e-4f, $"DequantDot deviates from decompressed dot: {maxErr}");
        Assert.True(maxScoresErr < 1e-4f, $"KeyScores deviates from decompressed dot: {maxScoresErr}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 6. V-aggregate parity
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)] // exercises the two-group (G=2) path
    public void ValueAggregate_MatchesDecompressedAggregate(int dim)
    {
        var c = new KVarNCompressor(dim);
        var rng = new Random(61);

        float[] values = GaussianMatrix(T, dim, seed: 62);
        byte[] tile = new byte[c.ValueTileBytes];
        c.CompressValueTile(values, tile);

        float[] decoded = new float[T * dim];
        c.DecompressValueTile(tile, decoded);

        // Softmax weights from Gaussian logits.
        float[] weights = SoftmaxWeights(T, rng);

        // Compressed path: aggregate in rotated domain, un-rotate once.
        float[] compressedAgg = new float[dim];
        c.AggregateValues(tile, weights, compressedAgg);
        c.UnrotateOutput(compressedAgg);

        // Reference: weighted sum over the decompressed rows.
        float[] referenceAgg = new float[dim];
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < dim; ch++)
                referenceAgg[ch] += weights[t] * decoded[t * dim + ch];

        // Informational: error of the compressed aggregate vs the fp32 truth.
        float[] trueAgg = new float[dim];
        for (int t = 0; t < T; t++)
            for (int ch = 0; ch < dim; ch++)
                trueAgg[ch] += weights[t] * values[t * dim + ch];

        float maxParityErr = 0f;
        for (int ch = 0; ch < dim; ch++)
            maxParityErr = MathF.Max(maxParityErr, MathF.Abs(compressedAgg[ch] - referenceAgg[ch]));

        output.WriteLine($"D={dim}: max |compressed-agg − decompressed-agg| = {maxParityErr:E3}");
        output.WriteLine($"D={dim}: aggregate rel. error vs fp32 truth = {RelativeFrobenius(trueAgg, compressedAgg):F4}");
        Assert.True(maxParityErr < 1e-4f, $"V-aggregate parity broken: {maxParityErr}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 7. Edge cases
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void AllZerosTile_ReconstructsExactZero_AndScoresZero()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        float[] zeros = new float[T * dim];

        byte[] kTile = new byte[c.KeyTileBytes];
        byte[] vTile = new byte[c.ValueTileBytes];
        c.CompressKeyTile(zeros, kTile);
        c.CompressValueTile(zeros, vTile);

        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(kTile, decoded);
        Assert.All(decoded, v => Assert.Equal(0f, v));
        c.DecompressValueTile(vTile, decoded);
        Assert.All(decoded, v => Assert.Equal(0f, v));

        var rng = new Random(71);
        float[] qRot = new float[dim];
        c.RotateQuery(GaussianVector(dim, rng), qRot);

        float[] scores = new float[T];
        c.KeyScores(kTile, qRot, scores);
        Assert.All(scores, s => Assert.Equal(0f, s));
        Assert.Equal(0f, c.DequantDot(kTile, qRot, 17));

        float[] agg = new float[dim];
        c.AggregateValues(vTile, SoftmaxWeights(T, rng), agg);
        c.UnrotateOutput(agg);
        Assert.All(agg, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void ConstantToken_RoundTripsWithinTolerance()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        float[] keys = GaussianMatrix(T, dim, seed: 72);
        for (int ch = 0; ch < dim; ch++)
            keys[19 * dim + ch] = 5f; // constant row → rotated row is a lone DC spike

        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);
        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(tile, decoded);

        // Measured: constant row ~0.018 (its rotated DC spike quantizes almost exactly);
        // whole tile ~0.14 — slightly above plain Gaussian (~0.10) because the DC spike
        // widens channel 0's per-channel RTN range for the other tokens.
        float constErr = RelativeFrobenius(keys.AsSpan(19 * dim, dim), decoded.AsSpan(19 * dim, dim));
        float wholeErr = RelativeFrobenius(keys, decoded);
        output.WriteLine($"constant-token row rel. error: {constErr:F4}; whole tile: {wholeErr:F4}");
        Assert.True(constErr < 0.05f, $"Constant token reconstructed poorly: {constErr}");
        Assert.True(wholeErr < 0.18f, $"Tile with constant token reconstructed poorly: {wholeErr}");
    }

    [Fact]
    public void SingleHugeOutlier_DoesNotDestroyOtherTokens()
    {
        const int dim = 128;
        var c = new KVarNCompressor(dim);
        float[] keys = GaussianMatrix(T, dim, seed: 73);
        keys[17 * dim + 3] = 1e4f;

        byte[] tile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, tile);
        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(tile, decoded);

        float maxOtherRowErr = 0f;
        for (int t = 0; t < T; t++)
        {
            if (t == 17) continue;
            maxOtherRowErr = MathF.Max(maxOtherRowErr,
                RelativeFrobenius(keys.AsSpan(t * dim, dim), decoded.AsSpan(t * dim, dim)));
        }
        float outlierRowErr = RelativeFrobenius(keys.AsSpan(17 * dim, dim), decoded.AsSpan(17 * dim, dim));
        output.WriteLine($"1e4 outlier at [17,3]: outlier row rel. err {outlierRowErr:F4}, worst other row {maxOtherRowErr:F4}");

        // Measured: outlier row ~0.096, worst other row ~0.111 — the Sinkhorn row/col
        // factors absorb the spike, so neither the outlier token nor its neighbors degrade.
        Assert.True(maxOtherRowErr < 0.15f, $"Outlier bled into other tokens: {maxOtherRowErr}");
        Assert.True(outlierRowErr < 0.15f, $"Outlier row itself reconstructed poorly: {outlierRowErr}");
    }

    [Theory]
    [InlineData(64)]
    [InlineData(256)]
    public void OtherHeadDims_RoundTripAndDotParity(int dim)
    {
        var c = new KVarNCompressor(dim);
        var rng = new Random(74);

        float[] keys = GaussianMatrix(T, dim, seed: 75);
        byte[] kTile = new byte[c.KeyTileBytes];
        c.CompressKeyTile(keys, kTile);

        float[] decoded = new float[T * dim];
        c.DecompressKeyTile(kTile, decoded);
        float kErr = RelativeFrobenius(keys, decoded);
        output.WriteLine($"D={dim}: K round-trip rel. Frobenius {kErr:F4}");
        Assert.True(kErr < 0.12f, $"K round-trip error too high at D={dim}: {kErr}");

        float[] q = GaussianVector(dim, rng);
        float[] qRot = new float[dim];
        c.RotateQuery(q, qRot);
        float[] scores = new float[T];
        c.KeyScores(kTile, qRot, scores);

        float maxErr = 0f;
        for (int t = 0; t < T; t++)
        {
            float reference = Dot(q, decoded.AsSpan(t * dim, dim));
            maxErr = MathF.Max(maxErr, MathF.Abs(scores[t] - reference));
            maxErr = MathF.Max(maxErr, MathF.Abs(c.DequantDot(kTile, qRot, t) - reference));
        }
        output.WriteLine($"D={dim}: max score parity error {maxErr:E3}");
        Assert.True(maxErr < 2e-4f, $"Score parity broken at D={dim}: {maxErr}");

        float[] values = GaussianMatrix(T, dim, seed: 76);
        byte[] vTile = new byte[c.ValueTileBytes];
        c.CompressValueTile(values, vTile);
        c.DecompressValueTile(vTile, decoded);
        float vErr = RelativeFrobenius(values, decoded);
        output.WriteLine($"D={dim}: V round-trip rel. Frobenius {vErr:F4}");
        Assert.True(vErr < 0.55f, $"V round-trip error too high at D={dim}: {vErr}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static float[] GaussianMatrix(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
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
        // Box-Muller; 1 - NextDouble() avoids log(0).
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

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static double DotDouble(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += (double)a[i] * b[i];
        return sum;
    }

    private static float RelativeFrobenius(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        double errSq = 0, refSq = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double e = actual[i] - reference[i];
            errSq += e * e;
            refSq += (double)reference[i] * reference[i];
        }
        return refSq > 0 ? (float)Math.Sqrt(errSq / refSq) : (float)Math.Sqrt(errSq);
    }

    private static (float rowLo, float rowHi, float colLo, float colHi) AxisRmsRange(
        ReadOnlySpan<float> tile, int rows, int cols)
    {
        float rowLo = float.MaxValue, rowHi = float.MinValue;
        for (int t = 0; t < rows; t++)
        {
            double sumSq = 0;
            for (int c = 0; c < cols; c++)
            {
                float v = tile[t * cols + c];
                sumSq += (double)v * v;
            }
            float rms = (float)Math.Sqrt(sumSq / cols);
            rowLo = MathF.Min(rowLo, rms);
            rowHi = MathF.Max(rowHi, rms);
        }

        float colLo = float.MaxValue, colHi = float.MinValue;
        for (int c = 0; c < cols; c++)
        {
            double sumSq = 0;
            for (int t = 0; t < rows; t++)
            {
                float v = tile[t * cols + c];
                sumSq += (double)v * v;
            }
            float rms = (float)Math.Sqrt(sumSq / rows);
            colLo = MathF.Min(colLo, rms);
            colHi = MathF.Max(colHi, rms);
        }

        return (rowLo, rowHi, colLo, colHi);
    }

    private static void AssertScalesReconstruct(
        ReadOnlySpan<float> original, ReadOnlySpan<float> normalized,
        ReadOnlySpan<float> rowScale, ReadOnlySpan<float> colScale,
        int rows, int cols)
    {
        for (int t = 0; t < rows; t++)
        {
            for (int c = 0; c < cols; c++)
            {
                float reconstructed = normalized[t * cols + c] * rowScale[t] * colScale[c];
                float expected = original[t * cols + c];
                float tolerance = 1e-3f * MathF.Max(1f, MathF.Abs(expected));
                Assert.True(MathF.Abs(reconstructed - expected) <= tolerance,
                    $"Scale reconstruction failed at [{t},{c}]: {reconstructed} vs {expected}");
            }
        }
    }
}
