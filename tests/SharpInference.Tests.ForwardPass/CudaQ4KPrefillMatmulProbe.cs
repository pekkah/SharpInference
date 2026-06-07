using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #159/#162 prefill probe (not a correctness test): times the two compute-bound
/// Q4_K trunk-matmul paths — int8 MMQ (<see cref="CudaBackend.MatMulBatchedMmq"/>) and
/// dequant→fp16→cuBLAS GEMM (<see cref="CudaBackend.MatMulBatchedGemm"/>) — at the REAL
/// Qwen3-8B prefill shapes (nTok=1844), and reports achieved TFLOP/s vs the RTX 4070 Ti's
/// ~165 TFLOP/s fp16-TC / ~330 TOP/s int8-TC peak.
///
/// Motivation: the full batched prefill achieves only ~6 TFLOP/s-equivalent (25.6 TFLOP /
/// ~4.0 s at N=1844), yet the older <see cref="CudaMmqRooflineProbe"/> reported 36–54 int8
/// TOPS for the isolated MMQ kernel at nTok=1024. This probe times the kernels at the exact
/// production shapes so we can tell whether the matmul kernels themselves are the bottleneck
/// (→ #149/#152 tiling rewrite) or whether per-call overhead / redundant work outside the
/// kernel dominates (→ a cheaper engine-side fix).
///
/// Run explicitly: --filter FullyQualifiedName~CudaQ4KPrefillMatmulProbe. Silent no-op
/// without CUDA. Always asserts true — it only prints the measurement.
/// </summary>
public sealed unsafe class CudaQ4KPrefillMatmulProbe
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

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
    public void Q4K_PrefillMatmul_AchievedThroughput_AtRealShapes()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Qwen3-8B trunk matmuls (rows=out, cols=in) at the real prefill batch N=1844.
        // hidden=4096, intermediate=12288, qkv=6144 (32+8+8 heads × 128).
        const int nTok = 1844;
        (int rows, int cols, string what)[] shapes =
        {
            (6144, 4096, "qkv"),
            (4096, 4096, "o-proj"),
            (12288, 4096, "ffn-gate"),
            (12288, 4096, "ffn-up"),
            (4096, 12288, "ffn-down"),
        };
        const double fp16Peak = 165.0;   // ~RTX 4070 Ti fp16-TC TFLOP/s
        const double int8Peak = 330.0;   // ~RTX 4070 Ti int8-TC TOP/s

        var rng = new Random(20260607);
        double totGflopPerLayer = 0, totMmqMs = 0, totGemmMs = 0;

        foreach (var (rows, cols, what) in shapes)
        {
            byte[] weightBytes = BuildQ4KMatrix(rows, cols, rng);
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            double macs = (double)rows * cols * nTok;
            double gflop = 2.0 * macs / 1e9;
            totGflopPerLayer += gflop;

            // --- MMQ (int8 tensor core, default prefill path) ---
            for (int i = 0; i < 5; i++) gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();
            const int iters = 30;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();
            sw.Stop();
            double mmqMs = sw.Elapsed.TotalMilliseconds / iters;
            totMmqMs += mmqMs;
            double mmqTops = 2.0 * macs / (mmqMs * 1e-3) / 1e12;

            // --- C1: dequant→fp16→cuBLAS GEMM ---
            for (int i = 0; i < 5; i++) gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();
            sw.Restart();
            for (int i = 0; i < iters; i++) gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q4_K);
            gpu.Synchronize();
            sw.Stop();
            double gemmMs = sw.Elapsed.TotalMilliseconds / iters;
            totGemmMs += gemmMs;
            double gemmTflop = gflop / (gemmMs * 1e-3) / 1e3;

            Console.WriteLine(
                $"{what,-9} [{rows}×{cols}]·[{cols}×{nTok}]  " +
                $"MMQ {mmqMs,6:F2} ms ({mmqTops,5:F1} TOP/s, {100 * mmqTops / int8Peak,2:F0}% int8) | " +
                $"GEMM {gemmMs,6:F2} ms ({gemmTflop,5:F1} TFLOP/s, {100 * gemmTflop / fp16Peak,2:F0}% fp16)");

            gpu.Free(gpuW); gpu.Free(gpuX); gpu.Free(gpuY);
        }

        // Per-token full-trunk projection (×36 layers) to compare with the e2e prefill.
        Console.WriteLine(
            $"per-layer trunk: {totGflopPerLayer:F1} GFLOP  |  MMQ {totMmqMs:F2} ms  GEMM {totGemmMs:F2} ms  " +
            $"(×36 layers @ N={nTok}: MMQ {totMmqMs * 36 / 1000:F2} s, GEMM {totGemmMs * 36 / 1000:F2} s)");
        Assert.True(true);
    }
}
