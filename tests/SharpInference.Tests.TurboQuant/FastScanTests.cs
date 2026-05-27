using System.Runtime.InteropServices;
using SharpInference.TurboQuant;

namespace SharpInference.Tests.TurboQuant;

public sealed class FastScanTests
{
    [Fact]
    public void TileBytes_Computes_NormsPlusCodes()
    {
        // 32 fp16 norms + dim × 16 nibble-packed code bytes.
        Assert.Equal(64 + 128 * 16, FastScan.TileBytes(128));
        Assert.Equal(64 + 256 * 16, FastScan.TileBytes(256));
    }

    [Fact]
    public void PackTile4Bit_Roundtrip_PreservesNormsAndCodes()
    {
        const int dim = 128;
        var (blocks, signPattern) = QuantizeRandomBlocks(dim, seed: 1);
        var tile = new byte[FastScan.TileBytes(dim)];
        FastScan.PackTile4Bit(blocks, tile, dim);

        int blockSize = TurboQuantOps.BlockSize(4, dim);
        var codes = tile.AsSpan(FastScan.NormBytesPerTile);

        for (int t = 0; t < FastScan.TileSize; t++)
        {
            // Norms — bytewise equal to the source block header.
            int srcNorm = t * blockSize + TurboQuantOps.NormOffset;
            Assert.Equal(blocks[srcNorm],     tile[t * 2]);
            Assert.Equal(blocks[srcNorm + 1], tile[t * 2 + 1]);

            var blockPacked = new ReadOnlySpan<byte>(blocks, t * blockSize + TurboQuantOps.IndicesOffset, blockSize - TurboQuantOps.IndicesOffset);
            int b = t & 15;
            for (int d = 0; d < dim; d++)
            {
                int expected = BitPacking.UnpackBits4(blockPacked, 0, d) & 0x0F;
                byte pair = codes[d * FastScan.CodeBytesPerDim + b];
                int got = t >= 16 ? (pair >> 4) & 0x0F : pair & 0x0F;
                Assert.Equal(expected, got);
            }
        }
    }

    [Fact]
    public void KScoreTile4BitScalar_Matches_PerBlockDequantDot_WithinLutTolerance()
    {
        const int dim = 128;
        VerifyScalarMatchesPerBlock(dim, seed: 7);
    }

    [Fact]
    public void KScoreTile4BitScalar_Matches_PerBlockDequantDot_Dim256()
    {
        const int dim = 256;
        VerifyScalarMatchesPerBlock(dim, seed: 11);
    }

    [Fact]
    public unsafe void KScoreTile4BitAvx2_Matches_Scalar()
    {
        const int dim = 128;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 23);
        var tile = new byte[FastScan.TileBytes(dim)];
        FastScan.PackTile4Bit(blocks, tile, dim);

        var rng = new Random(23);
        var rotatedQuery = new float[dim];
        for (int i = 0; i < dim; i++)
            rotatedQuery[i] = (float)(rng.NextDouble() * 2 - 1);

        var centroids = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var lut = new sbyte[dim * 16];
        float scale = FastScan.BuildLut4Bit(rotatedQuery, centroids, lut, dim);

        const float attnScale = 0.0884f; // 1 / sqrt(128); the actual value doesn't matter as long as both paths see the same one.
        var scoresScalar = new float[FastScan.TileSize];
        var scoresAvx2   = new float[FastScan.TileSize];

        FastScan.KScoreTile4BitScalar(tile, lut, scale, attnScale, scoresScalar, dim);

        fixed (byte* tilePtr = tile)
        fixed (sbyte* lutPtr = lut)
        fixed (float* outPtr = scoresAvx2)
            FastScan.KScoreTile4BitAvx2(tilePtr, lutPtr, scale, attnScale, outPtr, dim);

        // The two paths are required to produce identical i16 accumulators, so
        // the final scores can only differ by the fp32 rounding of the last
        // multiply chain. A few ULP of relative error is the upper bound here.
        for (int t = 0; t < FastScan.TileSize; t++)
            Assert.Equal(scoresScalar[t], scoresAvx2[t], 4);
    }

    [Fact]
    public void VTileBytes_Computes_NormsPlusTransposedCodes()
    {
        // 32 fp16 norms + 32 positions × dim/2 bytes of position-major nibble pairs.
        Assert.Equal(64 + 32 * 64,  FastScan.VTileBytes(128));
        Assert.Equal(64 + 32 * 128, FastScan.VTileBytes(256));
    }

    [Fact]
    public void PackVTile4Bit_Roundtrip_PreservesCodesAtTransposedLocations()
    {
        const int dim = 128;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 17);
        var vTile = new byte[FastScan.VTileBytes(dim)];
        FastScan.PackVTile4Bit(blocks, vTile, dim);

        int blockSize = TurboQuantOps.BlockSize(4, dim);
        var codes = vTile.AsSpan(FastScan.NormBytesPerTile);
        int packedPerBlock = dim / 2;

        for (int t = 0; t < FastScan.TileSize; t++)
        {
            var blockPacked = new ReadOnlySpan<byte>(blocks, t * blockSize + TurboQuantOps.IndicesOffset, packedPerBlock);
            for (int d = 0; d < dim; d += 2)
            {
                int expectedLow  = BitPacking.UnpackBits4(blockPacked, 0, d);
                int expectedHigh = BitPacking.UnpackBits4(blockPacked, 0, d + 1);
                byte got = codes[t * packedPerBlock + d / 2];
                Assert.Equal(expectedLow,  got & 0x0F);
                Assert.Equal(expectedHigh, (got >> 4) & 0x0F);
            }
        }
    }

    [Fact]
    public void VAggregateTile4BitScalar_Matches_PerBlockRotatedDomainAcc()
    {
        const int dim = 128;
        VerifyVScalarMatchesPerBlock(dim, seed: 31);
    }

    [Fact]
    public void VAggregateTile4BitScalar_Matches_PerBlockRotatedDomainAcc_Dim256()
    {
        const int dim = 256;
        VerifyVScalarMatchesPerBlock(dim, seed: 37);
    }

    [Fact]
    public unsafe void VAggregateTile4BitAvx2_Matches_Scalar()
    {
        const int dim = 128;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 41);
        var vTile = new byte[FastScan.VTileBytes(dim)];
        FastScan.PackVTile4Bit(blocks, vTile, dim);

        var rng = new Random(41);
        var weights = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
            weights[t] = (float)(rng.NextDouble() * 0.1);

        var effectiveW = new float[FastScan.TileSize];
        int blockSize = TurboQuantOps.BlockSize(4, dim);
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            float norm = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(blocks.AsSpan(t * blockSize, 2));
            effectiveW[t] = weights[t] * norm;
        }

        var centroids = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var vLut = new sbyte[FastScan.TileSize * 16];
        float vScale = FastScan.BuildVLut4Bit(effectiveW, centroids, vLut);

        var accScalar = new float[dim];
        var accAvx2   = new float[dim];
        FastScan.VAggregateTile4BitScalar(vTile, vLut, vScale, accScalar, dim);

        fixed (byte* vTilePtr = vTile)
        fixed (sbyte* vLutPtr = vLut)
        fixed (float* accPtr  = accAvx2)
            FastScan.VAggregateTile4BitAvx2(vTilePtr, vLutPtr, vScale, accPtr, dim);

        for (int d = 0; d < dim; d++)
            Assert.Equal(accScalar[d], accAvx2[d], 4);
    }

    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    public void KScoreTile3BitScalar_Matches_PerBlockDequantDot(int dim)
    {
        const int bits = 3;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 51, bits);
        var tile = new byte[FastScan.TileBytes(dim)];
        FastScan.PackTile3Bit(blocks, tile, dim);

        var rng = new Random(53);
        var rotatedQuery = new float[dim];
        for (int i = 0; i < dim; i++)
            rotatedQuery[i] = (float)(rng.NextDouble() * 2 - 1);

        var (centroids, _) = GetCodebook(bits, dim);
        var lut = new sbyte[dim * 16];
        float scale = FastScan.BuildLut3Bit(rotatedQuery, centroids, lut, dim);

        const float attnScale = 1.0f;
        var fastScores = new float[FastScan.TileSize];
        FastScan.KScoreTile4BitScalar(tile, lut, scale, attnScale, fastScores, dim);

        int blockSize = TurboQuantOps.BlockSize(bits, dim);
        var referenceScores = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            referenceScores[t] = TurboQuantOps.DequantDot(
                blocks.AsSpan(t * blockSize, blockSize),
                rotatedQuery, centroids, bits, dim) * attnScale;
        }

        // Same tolerance model as the 4-bit test, with the 3-bit codebook's
        // smaller scale (max centroid |x| is roughly half that of the 4-bit
        // codebook, so the absolute tolerance shrinks accordingly).
        float maxNorm = 0f;
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            float n = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(blocks.AsSpan(t * blockSize, 2));
            if (n > maxNorm) maxNorm = n;
        }
        float tol = dim * 0.5f * scale * maxNorm * attnScale;
        for (int t = 0; t < FastScan.TileSize; t++)
            Assert.InRange(fastScores[t] - referenceScores[t], -tol, tol);
    }

    [Theory]
    [InlineData(128)]
    [InlineData(256)]
    public void VAggregateTile3BitScalar_Matches_PerBlockRotatedDomainAcc(int dim)
    {
        const int bits = 3;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 61, bits);
        var vTile = new byte[FastScan.VTileBytes(dim)];
        FastScan.PackVTile3Bit(blocks, vTile, dim);

        var rng = new Random(67);
        var weights = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
            weights[t] = (float)(rng.NextDouble() * 0.1);

        int blockSize = TurboQuantOps.BlockSize(bits, dim);
        var effectiveW = new float[FastScan.TileSize];
        var norms = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            norms[t] = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(blocks.AsSpan(t * blockSize, 2));
            effectiveW[t] = weights[t] * norms[t];
        }

        var (centroids, _) = GetCodebook(bits, dim);
        var vLut = new sbyte[FastScan.TileSize * 16];
        float vScale = FastScan.BuildVLut3Bit(effectiveW, centroids, vLut);

        var fastAcc = new float[dim];
        FastScan.VAggregateTile4BitScalar(vTile, vLut, vScale, fastAcc, dim);

        var referenceAcc = new float[dim];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            ReadOnlySpan<byte> packed = blocks.AsSpan(t * blockSize + TurboQuantOps.IndicesOffset, blockSize - TurboQuantOps.IndicesOffset);
            for (int d = 0; d < dim; d++)
            {
                int code = BitPacking.UnpackBits3(packed, 0, d);
                referenceAcc[d] += weights[t] * norms[t] * centroids[code];
            }
        }

        float tol = FastScan.TileSize * 0.5f * vScale * 2f;
        for (int d = 0; d < dim; d++)
            Assert.InRange(fastAcc[d] - referenceAcc[d], -tol, tol);
    }

    [Fact]
    public unsafe void VAggregateTile3BitAvx2_Matches_Scalar()
    {
        const int dim = 128;
        const int bits = 3;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 71, bits);
        var vTile = new byte[FastScan.VTileBytes(dim)];
        FastScan.PackVTile3Bit(blocks, vTile, dim);

        var rng = new Random(73);
        var effectiveW = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
            effectiveW[t] = (float)(rng.NextDouble() * 0.5);

        var (centroids, _) = GetCodebook(bits, dim);
        var vLut = new sbyte[FastScan.TileSize * 16];
        float vScale = FastScan.BuildVLut3Bit(effectiveW, centroids, vLut);

        var accScalar = new float[dim];
        var accAvx2   = new float[dim];
        FastScan.VAggregateTile4BitScalar(vTile, vLut, vScale, accScalar, dim);
        fixed (byte* vTilePtr = vTile)
        fixed (sbyte* vLutPtr = vLut)
        fixed (float* accPtr  = accAvx2)
            FastScan.VAggregateTile4BitAvx2(vTilePtr, vLutPtr, vScale, accPtr, dim);

        for (int d = 0; d < dim; d++)
            Assert.Equal(accScalar[d], accAvx2[d], 4);
    }

    [Fact]
    public void KScoreTile4BitScalar_Survives_AllZeroQuery()
    {
        const int dim = 128;
        var (blocks, _) = QuantizeRandomBlocks(dim, seed: 99);
        var tile = new byte[FastScan.TileBytes(dim)];
        FastScan.PackTile4Bit(blocks, tile, dim);

        var rotatedQuery = new float[dim]; // all zeros
        var centroids = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var lut = new sbyte[dim * 16];
        float scale = FastScan.BuildLut4Bit(rotatedQuery, centroids, lut, dim);

        var scores = new float[FastScan.TileSize];
        FastScan.KScoreTile4BitScalar(tile, lut, scale, 1.0f, scores, dim);

        // A zero query should produce zero scores. BuildLut4Bit's invScale=0
        // sentinel guarantees the LUT is all zero, so the i16 accumulator is
        // also zero and the final multiply yields exactly 0f per lane.
        for (int t = 0; t < FastScan.TileSize; t++)
            Assert.Equal(0f, scores[t]);
    }

    /// <summary>
    /// Quantize 32 random vectors using the existing TurboQuantOps.Quantize so
    /// the test ingests blocks in the exact byte layout the engine produces today.
    /// </summary>
    private static (byte[] blocks, float[] signPattern) QuantizeRandomBlocks(int dim, int seed, int bits = 4)
    {
        int blockSize = TurboQuantOps.BlockSize(bits, dim);
        var blocks = new byte[FastScan.TileSize * blockSize];
        var signPattern = WalshHadamard.GenerateSignPattern(dim, seed).ToArray();
        var (centroids, boundaries) = GetCodebook(bits, dim);

        var rng = new Random(seed);
        var input = new float[dim];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            for (int i = 0; i < dim; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            TurboQuantOps.Quantize(
                input,
                blocks.AsSpan(t * blockSize, blockSize),
                signPattern, centroids, boundaries,
                bits, dim);
        }

        return (blocks, signPattern);
    }

    private static (float[] centroids, float[] boundaries) GetCodebook(int bits, int dim) =>
        (bits, dim) switch
        {
            (4, 128) => (TurboQuantCodebooks.Centroids4Bit_D128.ToArray(), TurboQuantCodebooks.Boundaries4Bit_D128.ToArray()),
            (4, 256) => (TurboQuantCodebooks.Centroids4Bit_D256.ToArray(), TurboQuantCodebooks.Boundaries4Bit_D256.ToArray()),
            (3, 128) => (TurboQuantCodebooks.Centroids3Bit_D128.ToArray(), TurboQuantCodebooks.Boundaries3Bit_D128.ToArray()),
            (3, 256) => (TurboQuantCodebooks.Centroids3Bit_D256.ToArray(), TurboQuantCodebooks.Boundaries3Bit_D256.ToArray()),
            _ => throw new ArgumentException($"Unsupported (bits, dim) = ({bits}, {dim})")
        };

    /// <summary>
    /// Verify that scoring an entire tile via the FastScan i8 path matches 32
    /// independent <see cref="TurboQuantOps.DequantDot"/> calls within the
    /// LUT-quantization tolerance.
    /// </summary>
    private static unsafe void VerifyScalarMatchesPerBlock(int dim, int seed)
    {
        var (blocks, _) = QuantizeRandomBlocks(dim, seed);
        var tile = new byte[FastScan.TileBytes(dim)];
        FastScan.PackTile4Bit(blocks, tile, dim);

        var rng = new Random(seed + 1);
        var rotatedQuery = new float[dim];
        for (int i = 0; i < dim; i++)
            rotatedQuery[i] = (float)(rng.NextDouble() * 2 - 1);

        var centroids  = dim == 128 ? TurboQuantCodebooks.Centroids4Bit_D128.ToArray() : TurboQuantCodebooks.Centroids4Bit_D256.ToArray();
        var lut = new sbyte[dim * 16];
        float scale = FastScan.BuildLut4Bit(rotatedQuery, centroids, lut, dim);

        const float attnScale = 1.0f;
        var fastScores = new float[FastScan.TileSize];
        FastScan.KScoreTile4BitScalar(tile, lut, scale, attnScale, fastScores, dim);

        int blockSize = TurboQuantOps.BlockSize(4, dim);
        var referenceScores = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            referenceScores[t] = TurboQuantOps.DequantDot(
                blocks.AsSpan(t * blockSize, blockSize),
                rotatedQuery,
                centroids,
                bits: 4, dim) * attnScale;
        }

        // Tolerance budget: each LUT entry rounds to nearest, contributing
        // ≤ 0.5 · scale of error per dim term. The worst case is fully-aligned
        // rounding (all sign-same) over `dim` terms multiplied by the block's
        // FP16 norm: |Δ score| ≤ dim · 0.5 · scale · norm · attnScale. In
        // practice the rounding errors random-walk so observed error is closer
        // to √(dim/12) · scale · norm, but we use the worst-case bound here so
        // the test is robust to seed choice.
        float maxNorm = 0f;
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            float n = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(blocks.AsSpan(t * blockSize, 2));
            if (n > maxNorm) maxNorm = n;
        }
        float tol = dim * 0.5f * scale * maxNorm * attnScale;
        for (int t = 0; t < FastScan.TileSize; t++)
            Assert.InRange(fastScores[t] - referenceScores[t], -tol, tol);
    }

    /// <summary>
    /// V-aggregation parity vs the per-block hot loop, in the <em>rotated</em>
    /// domain. The reference inlines the rotated-domain part of
    /// <see cref="TurboQuantOps.Dequantize"/> (centroids × norm only — no sign
    /// flip, no inverse WHT) so the comparison isolates the kernel's
    /// dim-folding work from the deferred-IWHT optimisation that the engine
    /// will pick up in Phase 2.
    /// </summary>
    private static unsafe void VerifyVScalarMatchesPerBlock(int dim, int seed)
    {
        var (blocks, _) = QuantizeRandomBlocks(dim, seed);
        var vTile = new byte[FastScan.VTileBytes(dim)];
        FastScan.PackVTile4Bit(blocks, vTile, dim);

        var rng = new Random(seed + 1);
        var weights = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
            weights[t] = (float)(rng.NextDouble() * 0.1);

        int blockSize = TurboQuantOps.BlockSize(4, dim);
        var effectiveW = new float[FastScan.TileSize];
        var norms = new float[FastScan.TileSize];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            norms[t] = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(blocks.AsSpan(t * blockSize, 2));
            effectiveW[t] = weights[t] * norms[t];
        }

        var centroids = dim == 128 ? TurboQuantCodebooks.Centroids4Bit_D128.ToArray() : TurboQuantCodebooks.Centroids4Bit_D256.ToArray();
        var vLut = new sbyte[FastScan.TileSize * 16];
        float vScale = FastScan.BuildVLut4Bit(effectiveW, centroids, vLut);

        var fastAcc = new float[dim];
        FastScan.VAggregateTile4BitScalar(vTile, vLut, vScale, fastAcc, dim);

        // Reference: for each block t, decompress in rotated domain only
        // (centroids[code[d]] · norm) and weighted-accumulate.
        var referenceAcc = new float[dim];
        for (int t = 0; t < FastScan.TileSize; t++)
        {
            ReadOnlySpan<byte> packed = blocks.AsSpan(t * blockSize + TurboQuantOps.IndicesOffset, blockSize - TurboQuantOps.IndicesOffset);
            for (int d = 0; d < dim; d++)
            {
                int code = BitPacking.UnpackBits4(packed, 0, d);
                referenceAcc[d] += weights[t] * norms[t] * centroids[code];
            }
        }

        // Per-d error bound: TileSize · 0.5 · scale (each LUT entry contributes
        // ≤ ½ LSB), folded over the running fp32 multiply. Pad 2× for rounding
        // direction stacking.
        float tol = FastScan.TileSize * 0.5f * vScale * 2f;
        for (int d = 0; d < dim; d++)
            Assert.InRange(fastAcc[d] - referenceAcc[d], -tol, tol);
    }
}
