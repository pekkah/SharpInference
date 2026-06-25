using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #388: parity for the Q3_K dequant→fp16→cuBLAS GEMM
/// (<see cref="CudaBackend.MatMulBatchedGemm"/> with <see cref="DType.Q3_K"/>, kernel
/// <c>llm_dequant_q3k_to_f16</c>). Carnice's MoE routed experts are ~76% Q3_K, and Q3_K
/// was the only quant excluded from every fast GEMM path — it fell back to the per-token
/// re-reading <c>llm_matvec_q3k_gemm_n</c>. This path dequants each Q3_K weight to an fp16
/// temp ONCE and cuBLAS-GEMMs it (the same fast path Q5_K/Q6_K experts already use),
/// collapsing the routed-MoE prefill cost.
///
/// A small Q3_K weight matrix [rows×cols] and an fp32 activation batch [nTok×cols] are
/// multiplied on the GPU (dequant kernel → fp16 → GemmEx) and on the CPU
/// (<see cref="SimdKernels.DotQ3K"/>, the fp32 dequant-and-dot reference). Both GPU operands
/// are fp16-rounded, so it tracks the fp32 reference to a loose per-RMS tolerance
/// (argmax-stable — same accuracy class as the Q5_K/Q6_K dequant-GEMM, NOT bit-exact). This
/// validates the new <c>llm_dequant_q3k_to_f16</c> element decode (hmask high-bit, the qs
/// 2-bit unpack, the kmask1/kmask2 6-bit scale unpack) and the row-major fp16 output layout
/// the GemmEx consumes, isolated from a model.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaDequantGemmQ3KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    /// <summary>Build a <paramref name="rows"/>×<paramref name="cols"/> Q3_K matrix
    /// (110 B / 256-element super-block: 32 hmask, 64 qs, 12 scale bytes, fp16 d). Any byte
    /// pattern is a valid 3-bit packing, so the quant/hmask/scale regions are random; only d
    /// is kept in a realistic small-magnitude range. Mirrors <c>CudaGemmQ3KTests</c>.</summary>
    private static byte[] BuildQ3KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 110;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 110;
                for (int i = 0; i < 108; i++) bytes[off + i] = (byte)rng.Next(256);
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.04 + 0.005));
                bytes[off + 108] = (byte)(dHalf & 0xFF);
                bytes[off + 109] = (byte)(dHalf >> 8);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedGemm_Q3K_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Square, a wide multi-superblock batch, a tall single-superblock batch, and a
        // partial-tile (non-multiple-of-8 rows) case. cols are 256-aligned (Q3_K super-block).
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (128, 2560, 64), (300, 256, 5) })
        {
            var rng = new Random(20260625 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, exact-byte Q3_K).
            int bytesPerRow = (cols / 256) * 110;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ3K(wPtr + (long)r * bytesPerRow, aPtr + (long)t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q3_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q3_K);
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
                $"GEMM-Q3K(dequant) rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"Q3_K dequant-GEMM drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 4% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
