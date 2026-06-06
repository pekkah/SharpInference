using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #141/#145 roofline probe (not a correctness test): times the int8 MMQ
/// GEMM (<see cref="CudaBackend.MatMulBatchedMmq"/>) at an FFN-representative prefill
/// shape and reports the achieved int8 TOPS, so the gap to the RTX 4070 Ti's ~160
/// TOPS dense int8 tensor-core peak tells us how much headroom a pipelined MMQ
/// rewrite would have. Profiling (#146/#147) showed the matmul/FFN GEMMs are now the
/// dominant prefill cost; this quantifies whether MMQ is compute-saturated or not.
///
/// Run explicitly: --filter FullyQualifiedName~CudaMmqRooflineProbe. Silent no-op
/// without CUDA. Always asserts true — it only prints the measurement.
/// </summary>
public sealed unsafe class CudaMmqRooflineProbe
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    private static byte[] BuildQ8_0Matrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 32, bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 34;
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.09 + 0.01));
                bytes[off] = (byte)(dHalf & 0xFF); bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        return bytes;
    }

    [Fact]
    public void Mmq_Q8_0_AchievedTops_AtFfnShape()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // FFN-ish prefill GEMMs for a ~Gemma-4-E4B layer (rows=out, cols=in, N=tokens).
        (int rows, int cols, int nTok, string what)[] shapes =
        {
            (8192, 2048, 1024, "ffn-gate/up"),
            (2048, 8192, 1024, "ffn-down"),
            (6144, 2048, 1024, "qkv"),
        };

        var rng = new Random(20260606);
        foreach (var (rows, cols, nTok, what) in shapes)
        {
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            // Warm up (incl. first-call NVRTC/quantize buffer alloc), then time.
            for (int i = 0; i < 5; i++) gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q8_0);
            gpu.Synchronize();

            const int iters = 50;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q8_0);
            gpu.Synchronize();
            sw.Stop();

            gpu.Free(gpuW); gpu.Free(gpuX); gpu.Free(gpuY);

            double msPer = sw.Elapsed.TotalMilliseconds / iters;
            double macs = (double)rows * cols * nTok;          // multiply-accumulates
            double tops = (2.0 * macs) / (msPer * 1e-3) / 1e12; // ×2 (MAC=mul+add)
            double peak = 160.0;                                // ~RTX 4070 Ti dense int8 TC TOPS
            Console.WriteLine(
                $"MMQ {what,-12} [{rows}×{cols}]·[{cols}×{nTok}]: {msPer:F3} ms/call  {tops:F1} int8 TOPS  ({100*tops/peak:F0}% of ~{peak:F0} peak)");
        }
        Assert.True(true);
    }
}
