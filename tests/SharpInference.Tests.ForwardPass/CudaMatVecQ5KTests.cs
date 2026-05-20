using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity test for the Q5_K CUDA matvec kernel (<c>llm_matvec_q5k</c>).
///
/// Synthesizes a small Q5_K-encoded weight matrix by hand-building block bytes
/// (FP16 d / dmin, packed 6-bit scales/mins, qh high bits, ql low nibbles),
/// then compares <see cref="CudaBackend.MatMul"/> against
/// <see cref="SimdKernels.MatVecQ5K"/>. The byte layout is identical between
/// the two paths so we only test the GPU dequant-dot semantics, not the
/// quantizer's choice of d/dmin.
///
/// Silently skips on hosts without CUDA, mirroring the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMatVecQ5KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/> Q5_K-encoded
    /// values. Layout per 256-element block (176 bytes): [d:fp16][dmin:fp16]
    /// [scales:12 packed 6-bit pairs][qh:32 high bits][ql:128 low nibbles].
    /// Scales and qs are filled from <paramref name="rng"/>.
    /// </summary>
    private static byte[] BuildQ5KMatrix(int rows, int cols, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 176;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 176;
                // d ∈ (0, 0.1], dmin ∈ (0, 0.05]. Plausible Q5_K-style scales.
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                float dmin = (float)(rng.NextDouble() * 0.04 + 0.005);
                ushort dHalf = HalfToUshort((Half)d);
                ushort dminHalf = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                bytes[off + 2] = (byte)(dminHalf & 0xFF);
                bytes[off + 3] = (byte)(dminHalf >> 8);

                // 12 bytes scales: random 6-bit fields. For simplicity fill all
                // 12 bytes with random data; the get_scale_min_k4 unpacking we
                // wrote in the kernel matches the CPU one regardless of values.
                for (int i = 0; i < 12; i++)
                    bytes[off + 4 + i] = (byte)rng.Next(256);

                // 32 bytes qh: random high bits.
                for (int i = 0; i < 32; i++)
                    bytes[off + 16 + i] = (byte)rng.Next(256);

                // 128 bytes ql: random 4-bit nibbles.
                for (int i = 0; i < 128; i++)
                    bytes[off + 48 + i] = (byte)rng.Next(256);
            }
        }
        return bytes;
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    [Fact]
    public void MatVecQ5K_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Use one Q5_K super-block per row (256 cols) over many rows so we
        // exercise per-row dispatch with cheap synthetic data. Add a second
        // configuration with cols=512 (2 blocks) to validate the inner loop.
        foreach ((int rows, int cols) in new[] { (8, 256), (33, 512), (64, 1024) })
        {
            var rng = new Random(20260520 + rows * 31 + cols);
            byte[] weightBytes = BuildQ5KMatrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference via SimdKernels.MatVecQ5K on the same raw bytes.
            var cpuOutput = new float[rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = cpuOutput)
            {
                SimdKernels.MatVecQ5K(oPtr, wPtr, iPtr, rows, cols);
            }

            // GPU: upload raw Q5_K bytes and dispatch matvec.
            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q5_K);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q5_K);
            gpu.Synchronize();

            var gpuResult = new float[rows];
            gpu.Download(gpuOutput, gpuResult);

            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            // Q5_K dequant is exact for matching bytes (no quantization step
            // here), so the only error sources are fp16 rounding of d/dmin
            // (already identical between CPU and GPU) and float reduction
            // ordering. Tolerance 1e-3 absolute or 1e-3 relative.
            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(gpuResult[r] - cpuOutput[r]);
                float rel = diff / (MathF.Abs(cpuOutput[r]) + 1e-6f);
                maxAbs = MathF.Max(maxAbs, diff);
                maxRel = MathF.Max(maxRel, rel);
                if (diff > 1e-3f && rel > 1e-3f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: gpu={gpuResult[r]:F5} cpu={cpuOutput[r]:F5} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"MatVecQ5K rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"Q5_K matvec mismatches ({mismatches}/{rows}) for rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }
}
