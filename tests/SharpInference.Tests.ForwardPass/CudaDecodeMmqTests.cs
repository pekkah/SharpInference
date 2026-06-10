using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #201: parity for the int8 tensor-core batched-decode matmul
/// (<see cref="CudaBackend.MatMulBatchedDecodeMmq"/>, kernel
/// <c>llm_mmq_q4k_soa_acts_n16</c> — the BN=16 decode tile of the prefill MMQ).
/// Same contract as the prefill MMQ (argmax-stable, both operands int8-quantized):
/// the output tracks the fp32 CPU reference (<see cref="SimdKernels.DotQ4K"/>) to a
/// per-batch-RMS tolerance. Batch sizes cover the partial token tile (n_tok &lt; 16),
/// the second grid.y tile (n_tok &gt; 16), and the row guards; the ineligible-shape
/// case pins the weight-stationary fallback.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaDecodeMmqTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

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
                for (int i = 4; i < 144; i++)
                    bytes[off + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    private static void RunCase(CudaBackend gpu, Tensor gpuW, byte[] weightBytes,
                                int rows, int cols, int nTok, Random rng)
    {
        var acts = new float[nTok * cols];
        for (int i = 0; i < acts.Length; i++)
            acts[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = (cols / 256) * 144;
        var cpuOut = new float[nTok * rows];
        fixed (byte* wPtr = weightBytes)
        fixed (float* aPtr = acts)
        {
            for (int t = 0; t < nTok; t++)
                for (int r = 0; r < rows; r++)
                    cpuOut[t * rows + r] = SimdKernels.DotQ4K(wPtr + r * bytesPerRow, aPtr + t * cols, cols);
        }

        var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
        var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

        gpu.MatMulBatchedDecodeMmq(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
        gpu.Synchronize();

        var gpuOut = new float[nTok * rows];
        gpu.Download(gpuY, gpuOut);
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
            if (diff > 0.03f * refRms) mismatches++;
        }
        Assert.True(mismatches <= cpuOut.Length / 100 + 1,
            $"Decode MMQ rows={rows} cols={cols} nTok={nTok} drifted from fp32 reference: " +
            $"{mismatches}/{cpuOut.Length} beyond 3% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
    }

    [Fact]
    public void DecodeMmq_Q4K_Soa_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // rows ≥ 2048 (eligible) with non-multiple-of-64 coverage via the second case;
        // batch sizes cover partial token tiles (2, 5, 11), the full tile (16), and the
        // grid.y = 2 round-up (17).
        foreach ((int rows, int cols) in new[] { (2048, 512), (2112, 256) })
        {
            var rng = new Random(20260610 + rows * 13 + cols * 5);
            byte[] weightBytes = BuildQ4KMatrix(rows, cols, rng);
            var gpuWAos = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_K);
            var gpuW = gpu.RepackQ4KSoa(gpuWAos, rows, cols);
            foreach (int nTok in new[] { 2, 5, 8, 11, 16, 17 })
                RunCase(gpu, gpuW, weightBytes, rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }

    /// <summary>Ineligible shapes (rows &lt; 2048 here) must take the weight-stationary
    /// fallback — same fp32-tracking contract holds trivially (the WS path is bit-exact
    /// to the per-token matvec).</summary>
    [Fact]
    public void DecodeMmq_SmallRows_FallsBackToWeightStationary()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int rows = 1024, cols = 512;
        var rng = new Random(20260610 + 7);
        byte[] weightBytes = BuildQ4KMatrix(rows, cols, rng);
        var gpuWAos = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_K);
        var gpuW = gpu.RepackQ4KSoa(gpuWAos, rows, cols);
        RunCase(gpu, gpuW, weightBytes, rows, cols, 8, rng);
        gpu.Free(gpuW);
    }
}
