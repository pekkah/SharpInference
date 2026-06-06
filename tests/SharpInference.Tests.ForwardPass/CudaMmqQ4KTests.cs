using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #156 Item C2 (MMQ): parity for the int8 tensor-core Q4_K×Q8_1 batched matmul
/// (<see cref="CudaBackend.MatMulBatchedMmq"/>, kernel <c>llm_mmq_q4k</c>). A small
/// Q4_K weight matrix [rows×cols] and an fp32 activation batch [nTok×cols] are
/// multiplied both on the GPU (MMQ) and on the CPU (<see cref="SimdKernels.DotQ4K"/>,
/// fp32 reference). MMQ quantizes the activation to int8 (Q8_1) before the int8 mma,
/// so — like the dp4a decode matvec — it tracks the fp32 reference to a loose
/// per-row-RMS tolerance rather than bit-exactly. This isolates the Q4_K-specific
/// decode (nibble expansion + get_scale_min + asymmetric min-bias) and the mma
/// fragment map (shared byte-for-byte with the validated Q8_0 MMQ) from the model.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMmqQ4KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>Build a <paramref name="rows"/>×<paramref name="cols"/> Q4_K matrix
    /// (144 B / 256-element super-block: d, dmin, 12 scale bytes, 128 qs bytes).</summary>
    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                float d = (float)(rng.NextDouble() * 0.04 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.02 + 0.002);
                ushort dHalf = HalfToUshort((Half)d);
                ushort dmHalf = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                bytes[off + 2] = (byte)(dmHalf & 0xFF);
                bytes[off + 3] = (byte)(dmHalf >> 8);
                // 12 scale bytes + 128 packed-nibble qs bytes: any pattern is a valid
                // Q4_K block (both kernels decode it identically via get_scale_min_k4).
                for (int i = 4; i < 144; i++)
                    bytes[off + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedMmq_Q4K_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Cover: square, non-multiple-of-16 rows + non-multiple-of-8 tokens (partial
        // tile guards), and a wide multi-superblock single-token-tile batch.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (33, 256, 5), (128, 2560, 64) })
        {
            var rng = new Random(20260606 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ4KMatrix(rows, cols, rng);

            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, exact-byte Q4_K).
            int bytesPerRow = (cols / 256) * 144;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ4K(wPtr + r * bytesPerRow, aPtr + t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();

            var gpuOut = new float[nTok * rows];
            gpu.Download(gpuY, gpuOut);
            gpu.Free(gpuW);
            gpu.Free(gpuX);
            gpu.Free(gpuY);

            // Per-row magnitude scale for the relative bound.
            double sumSq = 0;
            for (int i = 0; i < cpuOut.Length; i++) sumSq += (double)cpuOut[i] * cpuOut[i];
            float refRms = (float)Math.Sqrt(sumSq / cpuOut.Length);

            int mismatches = 0;
            float maxAbs = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 0.03f * refRms) mismatches++;
            }
            Console.WriteLine(
                $"MMQ-Q4K rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"MMQ Q4_K drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 3% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
