using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #142 DECODE roofline probe (not a correctness test): times the Q8_0 dp4a
/// decode matvec (<see cref="CudaBackend.MatMul(Tensor,Tensor,Tensor,DType)"/> on the
/// <c>llm_matvec_q8_0_dp4a</c> / <c>_soa</c> path) at the real decode shape (nTok=1)
/// for Gemma 4 E4B's FFN / qkv / o-proj weights, and reports achieved HBM bandwidth
/// vs the RTX 4070 Ti's ~504 GB/s peak. Decode is bandwidth-bound; this quantifies
/// whether the matvec kernel is near the memory ceiling (gap is elsewhere — launch /
/// clock / attention) or has read-efficiency headroom (coalescing / load width /
/// per-block scale overhead). Probes both the AoS and SoA (#149) weight layouts.
///
/// Run explicitly: --filter FullyQualifiedName~CudaDecodeMatvecRooflineProbe. Silent
/// no-op without CUDA. Always asserts true — it only prints the measurement.
/// </summary>
public sealed unsafe class CudaDecodeMatvecRooflineProbe
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
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + b * 34;
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.09 + 0.01));
                bytes[off] = (byte)(dHalf & 0xFF); bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        return bytes;
    }

    [Fact]
    public void DecodeMatvec_Q8_0_AchievedBandwidth_AtDecodeShape()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Gemma-4-E4B decode matvecs (rows=out, cols=in), nTok=1.
        (int rows, int cols, string what)[] shapes =
        {
            (16384, 2048, "ffn-gate"),   // intermediate=16384 hidden=2048
            (16384, 2048, "ffn-up"),
            (2048, 16384, "ffn-down"),
            (5120, 2048, "qkv"),         // q+k+v fused-ish width
            (2048, 4096, "o-proj"),
        };
        const double peakGBs = 504.0;     // RTX 4070 Ti HBM peak

        var rng = new Random(20260606);
        foreach (var (rows, cols, what) in shapes)
        {
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuX = gpu.Upload(vec, TensorShape.D1(cols));
            var gpuY = gpu.Allocate(TensorShape.D1(rows));

            // Real decode touches each weight once/token against a working set = the
            // whole model (>> the 4070 Ti's 48 MB L2), so every read is a cold HBM read.
            // A single looped buffer (<48 MB) stays L2-resident and overstates BW. Rotate
            // over enough distinct buffers (~256 MB) to exceed L2 and force cold reads.
            double oneMB = (double)rows * cols / 32.0 * 34.0 / 1e6;
            int pool = Math.Max(2, (int)Math.Ceiling(256.0 / oneMB));

            foreach (var layout in new[] { "AoS", "SoA" })
            {
                var ws = new Tensor[pool];
                for (int p = 0; p < pool; p++)
                {
                    ws[p] = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
                    if (layout == "SoA") ws[p] = gpu.RepackQ8_0Soa(ws[p], rows, cols);
                }

                for (int i = 0; i < pool * 2; i++) gpu.MatMul(gpuY, ws[i % pool], gpuX, DType.Q8_0);
                gpu.Synchronize();

                const int iters = 400;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) gpu.MatMul(gpuY, ws[i % pool], gpuX, DType.Q8_0);
                gpu.Synchronize();
                sw.Stop();

                for (int p = 0; p < pool; p++) gpu.Free(ws[p]);

                double msPer = sw.Elapsed.TotalMilliseconds / iters;
                // Bytes moved: Q8_0 weight (34 B/32) + Q8_1 act read + fp32 out (negligible).
                double wBytes = (double)rows * cols / 32.0 * 34.0;
                double gbs = wBytes / (msPer * 1e-3) / 1e9;
                Console.WriteLine(
                    $"matvec {what,-9} [{rows}x{cols}] {layout}: {msPer:F4} ms  {gbs:F0} GB/s  ({100 * gbs / peakGBs:F0}% of ~{peakGBs:F0} peak)");
            }

            gpu.Free(gpuX); gpu.Free(gpuY);
        }
        Assert.True(true);
    }
}
