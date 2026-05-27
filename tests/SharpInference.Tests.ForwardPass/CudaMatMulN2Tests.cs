using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #43 parity tests for <see cref="CudaBackend.MatMulN2"/>. For each
/// supported weight dtype (F32, Q4_K, Q5_K, Q6_K) the two-input batched
/// matvec must match the result of two sequential <see cref="CudaBackend.MatMul"/>
/// calls with the same weight matrix and inputs.
///
/// The acceptance criterion in the issue is "within F32/Bf16/Fp16 roundoff".
/// For matvec with fp32 accumulators the only divergence between N1 and N2
/// is intra-warp reduction ordering, so a tight absolute+relative tolerance
/// (1e-3) suffices.
///
/// Silently skips on hosts without CUDA, mirroring the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMatMulN2Tests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// Synthesize <paramref name="rows"/> × <paramref name="cols"/> Q5_K bytes
    /// (176 B per 256-element super-block). Same layout as the production GGUF
    /// path; matches the kernel's expected byte gathers exactly.
    private static byte[] BuildQ5KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 176;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 176;
                float d    = (float)(rng.NextDouble() * 0.09 + 0.01);
                float dmin = (float)(rng.NextDouble() * 0.04 + 0.005);
                ushort dh = HalfToUshort((Half)d), dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF); bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF); bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12;  i++) bytes[off +  4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 32;  i++) bytes[off + 16 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off + 48 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    /// Q4_K layout: 144 bytes per 256-element super-block.
    ///   [0:2]    fp16 d
    ///   [2:4]    fp16 dmin
    ///   [4:16]   12 bytes of packed 6-bit (scale, min) pairs (same packing as Q5_K)
    ///   [16:144] 128 bytes of 4-bit nibbles (low and high nibble per byte)
    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                // Keep scales modest so the dot product stays in a comfortable
                // fp32 range — large dmin values don't reveal additional bugs
                // and inflate the absolute tolerance budget.
                float d    = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.03 + 0.005);
                ushort dh = HalfToUshort((Half)d), dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF); bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF); bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12;  i++) bytes[off +   4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off +  16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    /// Q6_K layout: 210 bytes per 256-element super-block.
    ///   [0:128]   ql — lower 4-bit nibbles (two 64-byte halves)
    ///   [128:192] qh — upper 2-bit pairs
    ///   [192:208] 16 int8 scales
    ///   [208:210] fp16 d
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 210;
                for (int i = 0; i < 128; i++) bytes[off + i] = (byte)rng.Next(256);
                for (int i = 0; i < 64;  i++) bytes[off + 128 + i] = (byte)rng.Next(256);
                // int8 scales: keep them small so the dot product stays bounded.
                for (int i = 0; i < 16;  i++) bytes[off + 192 + i] = (byte)(rng.Next(33) - 16);
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                ushort dh = HalfToUshort((Half)d);
                bytes[off + 208] = (byte)(dh & 0xFF);
                bytes[off + 209] = (byte)(dh >> 8);
            }
        return bytes;
    }

    private static void AssertParity(string label, float[] n2a, float[] n2b,
                                     float[] refA, float[] refB,
                                     float absTol, float relTol)
    {
        int rows = refA.Length;
        int mismatches = 0;
        float maxAbsA = 0, maxAbsB = 0, maxRel = 0;
        for (int r = 0; r < rows; r++)
        {
            float dA = MathF.Abs(n2a[r] - refA[r]);
            float dB = MathF.Abs(n2b[r] - refB[r]);
            float relA = dA / (MathF.Abs(refA[r]) + 1e-6f);
            float relB = dB / (MathF.Abs(refB[r]) + 1e-6f);
            maxAbsA = MathF.Max(maxAbsA, dA);
            maxAbsB = MathF.Max(maxAbsB, dB);
            maxRel  = MathF.Max(maxRel, MathF.Max(relA, relB));
            if ((dA > absTol && relA > relTol) || (dB > absTol && relB > relTol))
            {
                if (mismatches < 3)
                    Console.WriteLine(
                        $"  {label}[{r}]: A n2={n2a[r]:F5} ref={refA[r]:F5} dA={dA:E2} | B n2={n2b[r]:F5} ref={refB[r]:F5} dB={dB:E2}");
                mismatches++;
            }
        }
        Console.WriteLine(
            $"{label}: maxAbsA={maxAbsA:E2} maxAbsB={maxAbsB:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
        Assert.True(mismatches == 0,
            $"{label} MatMulN2 parity mismatches ({mismatches}/{rows}), maxAbsA={maxAbsA:E3}, maxAbsB={maxAbsB:E3}, maxRel={maxRel:E3}");
    }

    /// Run one (rows, cols) configuration: upload weights, allocate two inputs
    /// and four outputs (refA, refB, n2a, n2b), dispatch both paths, and check.
    private static void RunQuantParityCase(CudaBackend gpu, string label,
        DType dtype, byte[] weightBytes, int rows, int cols, Random rng,
        float absTol, float relTol)
    {
        var inA = new float[cols];
        var inB = new float[cols];
        for (int i = 0; i < cols; i++) { inA[i] = (float)(rng.NextDouble() * 2 - 1); inB[i] = (float)(rng.NextDouble() * 2 - 1); }

        var gpuW   = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), dtype);
        var gpuInA = gpu.Upload(inA, TensorShape.D1(cols));
        var gpuInB = gpu.Upload(inB, TensorShape.D1(cols));
        var gpuRefA = gpu.Allocate(TensorShape.D1(rows));
        var gpuRefB = gpu.Allocate(TensorShape.D1(rows));
        var gpuN2A  = gpu.Allocate(TensorShape.D1(rows));
        var gpuN2B  = gpu.Allocate(TensorShape.D1(rows));

        gpu.MatMul(gpuRefA, gpuW, gpuInA, dtype);
        gpu.MatMul(gpuRefB, gpuW, gpuInB, dtype);
        gpu.MatMulN2(gpuN2A, gpuN2B, gpuW, gpuInA, gpuInB, dtype);
        gpu.Synchronize();

        var refA = new float[rows]; gpu.Download(gpuRefA, refA);
        var refB = new float[rows]; gpu.Download(gpuRefB, refB);
        var n2A  = new float[rows]; gpu.Download(gpuN2A,  n2A);
        var n2B  = new float[rows]; gpu.Download(gpuN2B,  n2B);

        gpu.Free(gpuW); gpu.Free(gpuInA); gpu.Free(gpuInB);
        gpu.Free(gpuRefA); gpu.Free(gpuRefB); gpu.Free(gpuN2A); gpu.Free(gpuN2B);

        AssertParity(label, n2A, n2B, refA, refB, absTol, relTol);
    }

    [Fact]
    public void MatMulN2_F32_MatchesSequentialMatMul()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (8, 256), (33, 512), (64, 1024), (128, 2048) })
        {
            var rng = new Random(20260527 + rows * 17 + cols);

            var weights = new float[rows * cols];
            for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
            var inA = new float[cols];
            var inB = new float[cols];
            for (int i = 0; i < cols; i++) { inA[i] = (float)(rng.NextDouble() * 2 - 1); inB[i] = (float)(rng.NextDouble() * 2 - 1); }

            var gpuW   = gpu.Upload(weights, TensorShape.D1(weights.Length));
            var gpuInA = gpu.Upload(inA, TensorShape.D1(cols));
            var gpuInB = gpu.Upload(inB, TensorShape.D1(cols));
            var gpuRefA = gpu.Allocate(TensorShape.D1(rows));
            var gpuRefB = gpu.Allocate(TensorShape.D1(rows));
            var gpuN2A  = gpu.Allocate(TensorShape.D1(rows));
            var gpuN2B  = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gpuRefA, gpuW, gpuInA, DType.Float32);
            gpu.MatMul(gpuRefB, gpuW, gpuInB, DType.Float32);
            gpu.MatMulN2(gpuN2A, gpuN2B, gpuW, gpuInA, gpuInB, DType.Float32);
            gpu.Synchronize();

            var refA = new float[rows]; gpu.Download(gpuRefA, refA);
            var refB = new float[rows]; gpu.Download(gpuRefB, refB);
            var n2A  = new float[rows]; gpu.Download(gpuN2A,  n2A);
            var n2B  = new float[rows]; gpu.Download(gpuN2B,  n2B);

            gpu.Free(gpuW); gpu.Free(gpuInA); gpu.Free(gpuInB);
            gpu.Free(gpuRefA); gpu.Free(gpuRefB); gpu.Free(gpuN2A); gpu.Free(gpuN2B);

            AssertParity($"F32 rows={rows} cols={cols}", n2A, n2B, refA, refB, absTol: 1e-3f, relTol: 1e-3f);
        }
    }

    [Fact]
    public void MatMulN2_Q4K_MatchesSequentialMatMul()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (8, 256), (33, 512), (64, 1024) })
        {
            var rng = new Random(20260527 + rows * 19 + cols);
            byte[] weights = BuildQ4KMatrix(rows, cols, rng);
            // Q4_K MatMul re-quantizes inputs to Q8_1 (per-32-elem amax/127);
            // tiny per-element quantization noise compounds across cols, so
            // give a slightly looser absolute tolerance than F32.
            RunQuantParityCase(gpu, $"Q4_K rows={rows} cols={cols}", DType.Q4_K, weights, rows, cols, rng,
                absTol: 5e-3f, relTol: 5e-3f);
        }
    }

    [Fact]
    public void MatMulN2_Q5K_MatchesSequentialMatMul()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (8, 256), (33, 512), (64, 1024) })
        {
            var rng = new Random(20260527 + rows * 23 + cols);
            byte[] weights = BuildQ5KMatrix(rows, cols, rng);
            RunQuantParityCase(gpu, $"Q5_K rows={rows} cols={cols}", DType.Q5_K, weights, rows, cols, rng,
                absTol: 1e-3f, relTol: 1e-3f);
        }
    }

    [Fact]
    public void MatMulN2_Q6K_MatchesSequentialMatMul()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (8, 256), (33, 512), (64, 1024) })
        {
            var rng = new Random(20260527 + rows * 29 + cols);
            byte[] weights = BuildQ6KMatrix(rows, cols, rng);
            RunQuantParityCase(gpu, $"Q6_K rows={rows} cols={cols}", DType.Q6_K, weights, rows, cols, rng,
                absTol: 1e-3f, relTol: 1e-3f);
        }
    }
}
