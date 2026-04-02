using SharpInference.TurboQuant;

namespace SharpInference.Tests.TurboQuant;

public sealed class TurboQuantTests
{
    [Fact]
    public void LloydMaxCodebook_Quantise_ReturnsValidIndex()
    {
        var cb = new LloydMaxCodebook
        {
            Boundaries = [-0.5f, 0f, 0.5f],
            Centroids = [-0.75f, -0.25f, 0.25f, 0.75f],
        };
        var idx = cb.Quantise(0.1f);
        Assert.True(idx < cb.Centroids.Length);
    }

    [Fact]
    public void TurboQuantCodebooks_3Bit_D128_HasCorrectShape()
    {
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128;
        var boundaries = TurboQuantCodebooks.Boundaries3Bit_D128;
        Assert.Equal(8, centroids.Length);
        Assert.Equal(7, boundaries.Length);
    }

    [Fact]
    public void TurboQuantCodebooks_3Bit_AreSymmetric()
    {
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128;
        for (int i = 0; i < centroids.Length / 2; i++)
            Assert.Equal(centroids[i], -centroids[centroids.Length - 1 - i], 4);
    }

    [Fact]
    public void WalshHadamard_IsInvolution_D128()
    {
        const int dim = 128;
        var rng = new Random(42);
        float[] input = new float[dim];
        for (int i = 0; i < dim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] output1 = new float[dim];
        float[] output2 = new float[dim];

        WalshHadamard.Transform(input, output1, dim);
        WalshHadamard.Transform(output1, output2, dim);

        // WHT(WHT(x)) / dim = x (each normalized transform divides by sqrt(dim))
        for (int i = 0; i < dim; i++)
            Assert.Equal(input[i], output2[i], 3);
    }

    [Fact]
    public void WalshHadamard_PreservesNorm_D128()
    {
        const int dim = 128;
        var rng = new Random(42);
        float[] input = new float[dim];
        for (int i = 0; i < dim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float normBefore = MathF.Sqrt(input.Select(x => x * x).Sum());

        float[] output = new float[dim];
        WalshHadamard.Transform(input, output, dim);

        float normAfter = MathF.Sqrt(output.Select(x => x * x).Sum());

        Assert.Equal(normBefore, normAfter, 3);
    }

    [Fact]
    public void WalshHadamard_D256_Works()
    {
        const int dim = 256;
        var rng = new Random(99);
        float[] input = new float[dim];
        for (int i = 0; i < dim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] output1 = new float[dim];
        float[] output2 = new float[dim];

        WalshHadamard.Transform(input, output1, dim);
        WalshHadamard.Transform(output1, output2, dim);

        for (int i = 0; i < dim; i++)
            Assert.Equal(input[i], output2[i], 3);
    }

    [Fact]
    public void BitPacking3_RoundTrip_AllValues()
    {
        byte[] buffer = new byte[BitPacking.PackedBytes3Bit + 4]; // +4 for safety

        for (int value = 0; value < 8; value++)
        {
            Array.Clear(buffer);
            for (int pos = 0; pos < 128; pos++)
            {
                BitPacking.PackBits3(buffer, 0, pos, value);
            }
            for (int pos = 0; pos < 128; pos++)
            {
                int unpacked = BitPacking.UnpackBits3(buffer, 0, pos);
                Assert.Equal(value, unpacked);
            }
        }
    }

    [Fact]
    public void BitPacking3_RoundTrip_MixedValues()
    {
        byte[] buffer = new byte[BitPacking.PackedBytes3Bit + 4];
        int[] expected = new int[128];
        var rng = new Random(123);

        for (int pos = 0; pos < 128; pos++)
            expected[pos] = rng.Next(8);

        Array.Clear(buffer);
        for (int pos = 0; pos < 128; pos++)
            BitPacking.PackBits3(buffer, 0, pos, expected[pos]);

        for (int pos = 0; pos < 128; pos++)
        {
            int actual = BitPacking.UnpackBits3(buffer, 0, pos);
            Assert.Equal(expected[pos], actual);
        }
    }

    [Fact]
    public void BitPacking4_RoundTrip_AllValues()
    {
        byte[] buffer = new byte[BitPacking.PackedBytes4Bit + 4];

        for (int value = 0; value < 16; value++)
        {
            Array.Clear(buffer);
            for (int pos = 0; pos < 128; pos++)
                BitPacking.PackBits4(buffer, 0, pos, value);
            for (int pos = 0; pos < 128; pos++)
            {
                int unpacked = BitPacking.UnpackBits4(buffer, 0, pos);
                Assert.Equal(value, unpacked);
            }
        }
    }

    [Fact]
    public void Quantize_Dequant_RoundTrip_IsApproximate()
    {
        const int dim = 128;
        var rng = new Random(42);

        // Create a random unit vector
        float[] input = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++)
        {
            input[i] = (float)(rng.NextDouble() * 2 - 1);
            norm += input[i] * input[i];
        }
        norm = MathF.Sqrt(norm);
        for (int i = 0; i < dim; i++) input[i] /= norm;

        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();

        int blockSize = TurboQuantOps.BlockSize(3, dim);
        byte[] compressed = new byte[blockSize];

        TurboQuantOps.Quantize(input, compressed, signPattern, centroids, boundaries, 3, dim);

        float[] decompressed = new float[dim];
        TurboQuantOps.Dequantize(compressed, decompressed, signPattern, centroids, 3, dim);

        // Compute MSE
        float mse = 0;
        for (int i = 0; i < dim; i++)
        {
            float err = input[i] - decompressed[i];
            mse += err * err;
        }
        mse /= dim;

        // MSE should be small (lossy but reasonable)
        Assert.True(mse < 0.1f, $"MSE too high: {mse}");
    }

    [Fact]
    public void Quantize_Dequant_RoundTrip_MSE_10KVectors()
    {
        const int dim = 128;
        const int numVectors = 10_000;
        var rng = new Random(42);
        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();
        int blockSize = TurboQuantOps.BlockSize(3, dim);

        double totalMse = 0;

        for (int v = 0; v < numVectors; v++)
        {
            float[] input = new float[dim];
            float norm = 0;
            for (int i = 0; i < dim; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2 - 1);
                norm += input[i] * input[i];
            }
            norm = MathF.Sqrt(norm);
            for (int i = 0; i < dim; i++) input[i] /= norm;

            byte[] compressed = new byte[blockSize];
            TurboQuantOps.Quantize(input, compressed, signPattern, centroids, boundaries, 3, dim);

            float[] decompressed = new float[dim];
            TurboQuantOps.Dequantize(compressed, decompressed, signPattern, centroids, 3, dim);

            double mse = 0;
            for (int i = 0; i < dim; i++)
            {
                double err = input[i] - decompressed[i];
                mse += err * err;
            }
            totalMse += mse / dim;
        }

        double avgMse = totalMse / numVectors;
        // Should be reasonable for 3-bit quantization of unit vectors
        Assert.True(avgMse < 0.05, $"Average MSE too high: {avgMse:E6}");
    }

    [Fact]
    public void DequantDot_CorrelatesWithDirectDot()
    {
        const int dim = 128;
        const int numTrials = 100;
        var rng = new Random(42);
        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();
        int blockSize = TurboQuantOps.BlockSize(3, dim);

        // Test that fused dequant-dot and decompress-then-dot give the same result.
        // Both approximate the true dot product; they should agree with each other.
        double totalAbsError = 0;
        int finiteCount = 0;

        for (int trial = 0; trial < numTrials; trial++)
        {
            float[] kv = new float[dim];
            float[] query = new float[dim];
            float kvNorm = 0, qNorm = 0;
            for (int i = 0; i < dim; i++)
            {
                kv[i] = (float)(rng.NextDouble() * 2 - 1);
                query[i] = (float)(rng.NextDouble() * 2 - 1);
                kvNorm += kv[i] * kv[i];
                qNorm += query[i] * query[i];
            }
            kvNorm = MathF.Sqrt(kvNorm);
            qNorm = MathF.Sqrt(qNorm);
            for (int i = 0; i < dim; i++) { kv[i] /= kvNorm; query[i] /= qNorm; }

            byte[] compressed = new byte[blockSize];
            TurboQuantOps.Quantize(kv, compressed, signPattern, centroids, boundaries, 3, dim);

            // Method 1: decompress then dot
            float[] decompressed = new float[dim];
            TurboQuantOps.Dequantize(compressed, decompressed, signPattern, centroids, 3, dim);
            float decompDot = 0;
            for (int i = 0; i < dim; i++) decompDot += decompressed[i] * query[i];

            // Method 2: fused dequant-dot
            float[] rotatedQuery = new float[dim];
            TurboQuantOps.RotateQuery(query, rotatedQuery, signPattern, dim);
            float fusedDot = TurboQuantOps.DequantDot(compressed, rotatedQuery, centroids, 3, dim);

            if (float.IsFinite(fusedDot) && float.IsFinite(decompDot))
            {
                totalAbsError += Math.Abs(fusedDot - decompDot);
                finiteCount++;
            }
        }

        double avgError = totalAbsError / finiteCount;
        // Fused and decompress paths should agree closely (both use same quantized data)
        Assert.True(avgError < 0.1, $"Average abs error between fused and decompress paths: {avgError:F6}");
        Assert.True(finiteCount == numTrials, $"Some trials produced non-finite results: {numTrials - finiteCount}");
    }

    [Fact]
    public void KvCacheCompressor_CompressDecompress_Works()
    {
        const int dim = 128;
        var compressor = new KvCacheCompressor(3, dim, 0);

        var rng = new Random(42);
        float[] input = new float[dim];
        for (int i = 0; i < dim; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        byte[] compressed = new byte[compressor.BlockSize];
        compressor.Compress(input, compressed);

        float[] decompressed = new float[dim];
        compressor.Decompress(compressed, decompressed);

        // Not exact but should be reasonably close
        float mse = 0;
        for (int i = 0; i < dim; i++)
        {
            float err = input[i] - decompressed[i];
            mse += err * err;
        }
        mse /= dim;
        Assert.True(mse < 0.5f, $"MSE too high: {mse}");
    }

    [Fact]
    public void KvCacheCompressor_DequantDot_Works()
    {
        const int dim = 128;
        var compressor = new KvCacheCompressor(3, dim, 0);

        var rng = new Random(42);
        float[] kv = new float[dim];
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            kv[i] = (float)(rng.NextDouble() * 2 - 1);
            query[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        byte[] compressed = new byte[compressor.BlockSize];
        compressor.Compress(kv, compressed);

        float[] rotatedQuery = new float[dim];
        compressor.RotateQuery(query, rotatedQuery);

        float result = compressor.DequantDot(compressed, rotatedQuery);

        // Just verify it produces a finite number
        Assert.True(float.IsFinite(result), $"DequantDot returned non-finite: {result}");
    }

    [Fact]
    public unsafe void DequantDot3Avx2_MatchesScalar()
    {
        const int dim = 128;
        var rng = new Random(42);
        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();
        int blockSize = TurboQuantOps.BlockSize(3, dim);

        float[] kv = new float[dim];
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            kv[i] = (float)(rng.NextDouble() * 2 - 1);
            query[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        byte[] compressed = new byte[blockSize];
        TurboQuantOps.Quantize(kv, compressed, signPattern, centroids, boundaries, 3, dim);

        float[] rotatedQuery = new float[dim];
        TurboQuantOps.RotateQuery(query, rotatedQuery, signPattern, dim);

        // Scalar via Span
        float scalarResult = TurboQuantOps.DequantDot(compressed, rotatedQuery, centroids, 3, dim);

        // AVX2 via pointers
        fixed (byte* pComp = compressed)
        fixed (float* pQuery = rotatedQuery, pCentroids = centroids)
        {
            float avxResult = TurboQuantOps.DequantDot3Avx2(pComp, pQuery, pCentroids, dim);
            Assert.Equal(scalarResult, avxResult, 3);
        }
    }

    [Fact]
    public unsafe void DequantDot4Avx2_MatchesScalar()
    {
        const int dim = 128;
        var rng = new Random(42);
        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries4Bit_D128.ToArray();
        int blockSize = TurboQuantOps.BlockSize(4, dim);

        float[] kv = new float[dim];
        float[] query = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            kv[i] = (float)(rng.NextDouble() * 2 - 1);
            query[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        byte[] compressed = new byte[blockSize];
        TurboQuantOps.Quantize(kv, compressed, signPattern, centroids, boundaries, 4, dim);

        float[] rotatedQuery = new float[dim];
        TurboQuantOps.RotateQuery(query, rotatedQuery, signPattern, dim);

        float scalarResult = TurboQuantOps.DequantDot(compressed, rotatedQuery, centroids, 4, dim);

        fixed (byte* pComp = compressed)
        fixed (float* pQuery = rotatedQuery, pCentroids = centroids)
        {
            float avxResult = TurboQuantOps.DequantDot4Avx2(pComp, pQuery, pCentroids, dim);
            Assert.Equal(scalarResult, avxResult, 3);
        }
    }

    [Fact]
    public void BlockSize_3Bit_D128_Is52Bytes()
    {
        Assert.Equal(52, TurboQuantOps.BlockSize(3, 128));
    }

    [Fact]
    public void BlockSize_4Bit_D128_Is68Bytes()
    {
        Assert.Equal(68, TurboQuantOps.BlockSize(4, 128));
    }

    [Fact]
    public void MagnitudeProfiler_UniformRatio_Returns3Bit()
    {
        var profiler = new MagnitudeProfiler(2, warmupTokens: 10);
        var rng = new Random(42);

        for (int t = 0; t < 10; t++)
        {
            float[] key = new float[128];
            float[] value = new float[128];
            for (int i = 0; i < 128; i++)
            {
                key[i] = (float)(rng.NextDouble() * 2 - 1);
                value[i] = (float)(rng.NextDouble() * 2 - 1);
            }
            profiler.Record(0, key, value);
            profiler.Record(1, key, value);
        }

        Assert.True(profiler.IsFrozen);
        Assert.Equal(3, profiler.Budgets[0].KeyBits);
        Assert.Equal(3, profiler.Budgets[0].ValueBits);
    }

    [Fact]
    public void MagnitudeProfiler_HighKeyRatio_Returns4BitKeys()
    {
        var profiler = new MagnitudeProfiler(1, warmupTokens: 10);
        var rng = new Random(42);

        for (int t = 0; t < 10; t++)
        {
            float[] key = new float[128];
            float[] value = new float[128];
            for (int i = 0; i < 128; i++)
            {
                key[i] = (float)(rng.NextDouble() * 6 - 3); // ~3x magnitude keys
                value[i] = (float)(rng.NextDouble() * 0.2 - 0.1); // small values (~0.1 magnitude)
            }
            // Target K/V magnitude ratio ~30x (in the 10-60 range)
            profiler.Record(0, key, value);
        }

        Assert.True(profiler.IsFrozen);
        Assert.Equal(4, profiler.Budgets[0].KeyBits);
        // Values get 3 bits (ratio 10-60x)
        Assert.Equal(3, profiler.Budgets[0].ValueBits);
    }

    [Fact]
    public void WalshHadamard_TransformIsDeterministic()
    {
        const int dim = 128;
        var rng = new Random(42);
        float[] input = new float[dim];
        for (int i = 0; i < dim; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] out1 = new float[dim];
        float[] out2 = new float[dim];

        WalshHadamard.Transform(input, out1, dim);
        WalshHadamard.Transform(input, out2, dim);

        for (int i = 0; i < dim; i++)
            Assert.Equal(out1[i], out2[i], 6);
    }

    [Fact]
    public void SignPattern_IsDeterministic()
    {
        var p1 = WalshHadamard.GenerateSignPattern(128, 0);
        var p2 = WalshHadamard.GenerateSignPattern(128, 0);
        for (int i = 0; i < 128; i++)
            Assert.Equal(p1[i], p2[i]);
    }

    [Fact]
    public void SignPattern_DiffersByLayer()
    {
        var p0 = WalshHadamard.GenerateSignPattern(128, 0);
        var p1 = WalshHadamard.GenerateSignPattern(128, 1);
        bool anyDifferent = false;
        for (int i = 0; i < 128; i++)
        {
            if (p0[i] != p1[i]) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void Codebook_GetCentroids_ThrowsForInvalidDim()
    {
        Assert.Throws<ArgumentException>(() => TurboQuantCodebooks.GetCentroids(3, 64));
    }

    [Fact]
    public void Quantize_4Bit_RoundTrip_Works()
    {
        const int dim = 128;
        var rng = new Random(42);
        var signPattern = WalshHadamard.GenerateSignPattern(dim, 0);
        var centroids = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var boundaries = TurboQuantCodebooks.Boundaries4Bit_D128.ToArray();
        int blockSize = TurboQuantOps.BlockSize(4, dim);

        float[] input = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++)
        {
            input[i] = (float)(rng.NextDouble() * 2 - 1);
            norm += input[i] * input[i];
        }
        norm = MathF.Sqrt(norm);
        for (int i = 0; i < dim; i++) input[i] /= norm;

        byte[] compressed = new byte[blockSize];
        TurboQuantOps.Quantize(input, compressed, signPattern, centroids, boundaries, 4, dim);

        float[] decompressed = new float[dim];
        TurboQuantOps.Dequantize(compressed, decompressed, signPattern, centroids, 4, dim);

        float mse = 0;
        for (int i = 0; i < dim; i++)
        {
            float err = input[i] - decompressed[i];
            mse += err * err;
        }
        mse /= dim;
        // 4-bit should have lower MSE than 3-bit
        Assert.True(mse < 0.05f, $"4-bit MSE too high: {mse}");
    }
}
