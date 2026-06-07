using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #162: parity for the Q6_K dequant→fp16→cuBLAS GEMM batched prefill path
/// (<see cref="CudaBackend.MatMulBatchedGemm"/> with <see cref="DType.Q6_K"/>, kernel
/// <c>llm_dequant_q6k_to_f16</c>). Qwen3-8B-Q4_K_M keeps ~half of ffn_down + attn_v in
/// Q6_K; before this path those trunk matmuls fell back to the per-token GEMM-N matvec
/// (weight re-streamed once per token) — the dominant large-N prefill cost.
///
/// A small Q6_K weight matrix [rows×cols] and an fp32 activation batch [nTok×cols] are
/// multiplied on the GPU (dequant→fp16 GEMM) and on the CPU (<see cref="SimdKernels.DotQ6K"/>,
/// fp32 reference). The weight + activation are rounded to fp16 before the GEMM, so the
/// result tracks the fp32 reference to a loose per-RMS tolerance rather than bit-exactly
/// — this validates the Q6_K element decode (ql/qh split + 16 int8 scales) and the row-
/// major output layout, isolated from the model.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaGemmQ6KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>Build a <paramref name="rows"/>×<paramref name="cols"/> Q6_K matrix
    /// (210 B / 256-element super-block: 128 ql, 64 qh, 16 int8 scales, fp16 d).</summary>
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 210;
                // 192 ql/qh bytes: any pattern is a valid 6-bit packing.
                for (int i = 0; i < 192; i++) bytes[off + i] = (byte)rng.Next(256);
                // 16 signed int8 scales (small magnitude → realistic).
                for (int i = 0; i < 16; i++) bytes[off + 192 + i] = (byte)(sbyte)(rng.Next(33) - 16);
                // fp16 super-block scale d.
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.04 + 0.005));
                bytes[off + 208] = (byte)(dHalf & 0xFF);
                bytes[off + 209] = (byte)(dHalf >> 8);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedGemm_Q6K_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Square, a wide multi-superblock single-token-tile batch, and a partial-tile
        // (non-power-of-two rows) case to exercise the GEMM tail.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (128, 2560, 64), (300, 256, 5) })
        {
            var rng = new Random(20260607 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ6KMatrix(rows, cols, rng);

            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, exact-byte Q6_K).
            int bytesPerRow = (cols / 256) * 210;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ6K(wPtr + (long)r * bytesPerRow, aPtr + (long)t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q6_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q6_K);
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
                $"GEMM-Q6K rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"Q6_K GEMM drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 4% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
