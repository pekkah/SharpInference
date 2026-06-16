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
        BitConverter.HalfToUInt16Bits(h);

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

        // rows ≥ 2048 (eligible). #205: the dispatcher routes low-row shapes
        // (ceil(rows/64) < 2·SM) to the BM=32 tile and high-row shapes to BM=64, so the
        // cases span both kernels on a ≤64-SM card: 2048 (BM=32, no tail), 2096 (BM=32 with
        // a non-multiple-of-32 row tail + non-multiple-of-64), 8192 (BM=64). Both tiles are
        // bit-identical and tracked against the same fp32 CPU reference. Batch sizes cover
        // partial token tiles (2, 5, 11), the full tile (16), and the grid.y = 2 round-up (17).
        foreach ((int rows, int cols) in new[] { (2048, 512), (2096, 256), (8192, 512) })
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

    // ── #204 Q6_K decode MMQ ────────────────────────────────────────────────
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 210;
                // ql[0:128], qh[128:192], scales[192:208] (int8), d (208:210 fp16).
                for (int i = 0; i < 192; i++)
                    bytes[off + i] = (byte)rng.Next(256);
                for (int i = 0; i < 16; i++)
                    bytes[off + 192 + i] = (byte)(sbyte)(rng.Next(127) - 63);   // signed int8 scale
                float d = (float)(rng.NextDouble() * 0.04 + 0.005);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 208] = (byte)(dHalf & 0xFF);
                bytes[off + 209] = (byte)(dHalf >> 8);
            }
        return bytes;
    }

    private static void RunCaseQ6K(CudaBackend gpu, Tensor gpuW, byte[] weightBytes,
                                   int rows, int cols, int nTok, Random rng)
    {
        var acts = new float[nTok * cols];
        for (int i = 0; i < acts.Length; i++)
            acts[i] = (float)(rng.NextDouble() * 2 - 1);

        int bytesPerRow = (cols / 256) * 210;
        var cpuOut = new float[nTok * rows];
        fixed (byte* wPtr = weightBytes)
        fixed (float* aPtr = acts)
        {
            for (int t = 0; t < nTok; t++)
                for (int r = 0; r < rows; r++)
                    cpuOut[t * rows + r] = SimdKernels.DotQ6K(wPtr + r * bytesPerRow, aPtr + t * cols, cols);
        }

        var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
        var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

        gpu.MatMulBatchedDecodeMmq(gpuY, gpuW, gpuX, nTok, DType.Q6_K);
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
            $"Q6_K decode MMQ rows={rows} cols={cols} nTok={nTok} drifted from fp32 reference: " +
            $"{mismatches}/{cpuOut.Length} beyond 3% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
    }

    [Fact]
    public void DecodeMmq_Q6K_Soa_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // rows ≥ 2048 (eligible). Same #205 dual-tile coverage as the Q4_K case: 2048 (BM=32,
        // no tail), 2096 (BM=32 with a non-multiple-of-32 row tail), 8192 (BM=64). Batch sizes
        // cover partial token tiles (2, 5), the full tile (16), and the grid.y = 2 round-up (17).
        foreach ((int rows, int cols) in new[] { (2048, 512), (2096, 256), (8192, 512) })
        {
            var rng = new Random(20260616 + rows * 13 + cols * 5);
            byte[] weightBytes = BuildQ6KMatrix(rows, cols, rng);
            var gpuWAos = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q6_K);
            // #204: RepackQ6KSoa now FREES the AoS weight and returns the SoA buffer (the only
            // copy); MatMulBatchedDecodeMmq reads it directly via llm_mmq_q6k_soa_acts_n16.
            var gpuW = gpu.RepackQ6KSoa(gpuWAos, rows, cols);
            foreach (int nTok in new[] { 2, 5, 8, 16, 17 })
                RunCaseQ6K(gpu, gpuW, weightBytes, rows, cols, nTok, rng);
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
