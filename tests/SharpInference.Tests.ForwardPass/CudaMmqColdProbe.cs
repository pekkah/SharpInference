using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #162 cold-regime prefill probe (not a correctness test). The warm
/// <see cref="CudaQ4KPrefillMatmulProbe"/> loops the SAME weight 30× → the weight
/// stays resident in L2, so it measures the kernel's compute/IPC ceiling (~0.5 s for
/// the Qwen3-8B trunk). Real prefill walks 36 layers of DISTINCT weights (~3.9 GB
/// working set ≫ 48 MB L2), so each weight is cold on its first/only visit; that real
/// trunk costs ~4 s — an ~8× "cold" penalty the warm probe can't see.
///
/// This probe reproduces that regime cheaply (no model load): it allocates a ring of
/// N distinct weight matrices per shape (total bytes ≫ L2), round-robins matmuls
/// through them at the real prefill nTok, and reports per-matmul cold time. It is the
/// fast A/B harness for the MMQ tiling rewrite — the number to beat is the cold time,
/// not the warm roofline.
///
/// Run explicitly: --filter FullyQualifiedName~CudaMmqColdProbe. Silent no-op without
/// CUDA. Always asserts true — it only prints the measurement.
/// </summary>
public sealed unsafe class CudaMmqColdProbe
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    // block_q4_K = 144 B / 256 elems: d(fp16) dmin(fp16) scales[12] qs[128].
    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256, bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 144;
                ushort d = HalfToUshort((Half)(float)(rng.NextDouble() * 0.04 + 0.01));
                ushort dmin = HalfToUshort((Half)(float)(rng.NextDouble() * 0.02 + 0.005));
                bytes[off] = (byte)(d & 0xFF); bytes[off + 1] = (byte)(d >> 8);
                bytes[off + 2] = (byte)(dmin & 0xFF); bytes[off + 3] = (byte)(dmin >> 8);
                for (int i = 0; i < 12; i++) bytes[off + 4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off + 16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void Q4K_PrefillMatmul_ColdRegime_AtRealShapes()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Qwen3-8B trunk matmuls (rows=out, cols=in) at the real prefill batch N=1844.
        const int nTok = 1844;
        (int rows, int cols, string what)[] shapes =
        {
            (6144, 4096, "qkv"),
            (4096, 4096, "o-proj"),
            (12288, 4096, "ffn-gate"),
            (12288, 4096, "ffn-up"),
            (4096, 12288, "ffn-down"),
        };

        // A ring of distinct weights big enough to evict L2 (48 MB on a 4070 Ti) between
        // revisits. 16 copies of the largest shape (~28 MB each) ≈ 448 MB ≫ L2, so every
        // visit is a cold L2 miss — the real-prefill regime, without a 36-layer model.
        const int ringLen = 16;

        var rng = new Random(20260607);
        double totColdMs = 0, totGflopPerLayer = 0;

        foreach (var (rows, cols, what) in shapes)
        {
            var gpuW = new Tensor[ringLen];
            for (int k = 0; k < ringLen; k++)
            {
                byte[] wb = BuildQ4KMatrix(rows, cols, rng);
                gpuW[k] = gpu.UploadRaw(wb, TensorShape.D1(wb.Length), DType.Q4_K);
            }
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            double macs = (double)rows * cols * nTok;
            double gflop = 2.0 * macs / 1e9;
            totGflopPerLayer += gflop;

            // Warm up the kernel (JIT/quantize buffers) on one weight, then time the ring:
            // each iteration hits a different weight → its bytes were evicted by the 15
            // others since the last visit, so the load streams cold from HBM.
            for (int i = 0; i < 3; i++) gpu.MatMulBatchedMmq(gpuY, gpuW[0], gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();

            const int iters = ringLen * 4;   // 64 matmuls, 4 full passes over the ring
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                gpu.MatMulBatchedMmq(gpuY, gpuW[i % ringLen], gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();
            sw.Stop();

            double coldMs = sw.Elapsed.TotalMilliseconds / iters;
            totColdMs += coldMs;
            double coldTops = 2.0 * macs / (coldMs * 1e-3) / 1e12;
            Console.WriteLine(
                $"{what,-9} [{rows}×{cols}]·[{cols}×{nTok}]  COLD {coldMs,7:F2} ms/call  ({coldTops,5:F1} TOP/s)");

            foreach (var w in gpuW) gpu.Free(w);
            gpu.Free(gpuX); gpu.Free(gpuY);
        }

        Console.WriteLine(
            $"per-layer trunk: {totGflopPerLayer:F1} GFLOP  |  COLD {totColdMs:F2} ms  " +
            $"(×36 layers @ N={nTok}: {totColdMs * 36 / 1000:F2} s)");
        Assert.True(true);
    }
}
