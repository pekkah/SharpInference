using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #162: parity for the Q5_K dequant→fp16→cuBLAS GEMM batched prefill path
/// (<see cref="CudaBackend.MatMulBatchedGemm"/> with <see cref="DType.Q5_K"/>, kernel
/// <c>llm_dequant_q5k_to_f16</c>). Q5_K_M mixes keep q/k/o/gate/up in Q5_K; before this
/// path those trunk matmuls fell back to the per-token GEMM-N matvec.
///
/// Mirrors <see cref="CudaGemmQ6KTests"/>: a Q5_K weight matrix and an fp32 activation
/// batch are multiplied on the GPU (dequant→fp16 GEMM) and on the CPU
/// (<see cref="SimdKernels.DotQ5K"/>, fp32 reference). fp16 rounding → loose per-RMS
/// tolerance. Validates the Q5_K element decode (qh high-bit + shared Q4_K scale packing)
/// and the row-major output layout, isolated from the model.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaGemmQ5KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // Canonical fp16 bit pattern as ushort (matches the GPU/CPU fp16 decode).
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    /// <summary>Build a <paramref name="rows"/>×<paramref name="cols"/> Q5_K matrix
    /// (176 B / 256-element super-block: d, dmin, 12 scale bytes, 32 qh, 128 ql).</summary>
    private static byte[] BuildQ5KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 176;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 176;
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.04 + 0.005));
                ushort dmHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.02 + 0.002));
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                bytes[off + 2] = (byte)(dmHalf & 0xFF);
                bytes[off + 3] = (byte)(dmHalf >> 8);
                // 12 scale bytes + 32 qh + 128 ql: any pattern is a valid Q5_K block.
                for (int i = 4; i < 176; i++) bytes[off + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedGemm_Q5K_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (128, 2560, 64), (300, 256, 5) })
        {
            var rng = new Random(20260607 + rows * 17 + cols * 11 + nTok);
            byte[] weightBytes = BuildQ5KMatrix(rows, cols, rng);

            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 256) * 176;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ5K(wPtr + (long)r * bytesPerRow, aPtr + (long)t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q5_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q5_K);
            gpu.Synchronize();

            var gpuOut = new float[nTok * rows];
            gpu.Download(gpuY, gpuOut);
            gpu.Free(gpuW);
            gpu.Free(gpuX);
            gpu.Free(gpuY);

            double sumSq = 0;
            for (int i = 0; i < cpuOut.Length; i++) sumSq += (double)cpuOut[i] * cpuOut[i];
            float refRms = (float)Math.Sqrt(sumSq / cpuOut.Length);

            int mismatches = 0;
            float maxAbs = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 0.04f * refRms) mismatches++;
            }
            Console.WriteLine(
                $"GEMM-Q5K rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"Q5_K GEMM drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 4% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
