using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #156 DECODE roofline probe (not a correctness test): times the Q4_K dp4a/Q8_1
/// decode matvec (<see cref="CudaBackend.MatMul(Tensor,Tensor,Tensor,DType)"/> on the
/// <c>llm_matvec_q4k</c> path) at the real decode shape (nTok=1) for Qwen3-8B's FFN /
/// qkv / o-proj weights, and reports achieved HBM bandwidth vs the RTX 4070 Ti's
/// ~504 GB/s peak. Decode is bandwidth-bound; this quantifies whether the Q4_K matvec
/// is near the memory ceiling (gap is elsewhere — sampler / attention / launch) or has
/// read-efficiency headroom that a layout repack (cf. #149 Q8_0 SoA) could recover.
///
/// Unlike Q8_0 (34-byte block → 2-byte misalignment → funnelshift), the Q4_K 144-byte
/// super-block is 16-byte aligned and every uint32 weight load in llm_matvec_q4k is
/// already aligned — so this probe is the empirical gate on whether a Q4_K SoA repack
/// is worth building at all.
///
/// Run explicitly: --filter FullyQualifiedName~CudaDecodeMatvecQ4KRooflineProbe. Silent
/// no-op without CUDA. Always asserts true — it only prints the measurement.
/// </summary>
public sealed unsafe class CudaDecodeMatvecQ4KRooflineProbe
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
                for (int i = 0; i < 12; i++) bytes[off + 4 + i] = (byte)rng.Next(256);   // 6-bit packed scales/mins
                for (int i = 0; i < 128; i++) bytes[off + 16 + i] = (byte)rng.Next(256);  // 4-bit quants
            }
        return bytes;
    }

    [Fact]
    public void DecodeMatvec_Q4K_AchievedBandwidth_AtDecodeShape()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Qwen3-8B decode matvecs (rows=out, cols=in), nTok=1.
        // hidden=4096, intermediate=12288, n_head=32 d_head=128, n_kv=8.
        (int rows, int cols, string what)[] shapes =
        {
            (12288, 4096, "ffn-gate"),
            (12288, 4096, "ffn-up"),
            (4096, 12288, "ffn-down"),
            (6144, 4096, "qkv"),      // (32+8+8)*128 = 6144
            (4096, 4096, "o-proj"),
        };
        const double peakGBs = 504.0;     // RTX 4070 Ti HBM peak

        var rng = new Random(20260606);
        foreach (var (rows, cols, what) in shapes)
        {
            byte[] weightBytes = BuildQ4KMatrix(rows, cols, rng);
            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuX = gpu.Upload(vec, TensorShape.D1(cols));
            var gpuY = gpu.Allocate(TensorShape.D1(rows));

            // Rotate over enough distinct weight buffers (~256 MB) to exceed the 4070 Ti's
            // ~48 MB L2 and force cold HBM reads — a single looped buffer stays L2-resident
            // and overstates BW (real decode's working set is the whole model).
            double oneMB = (double)rows * cols / 256.0 * 144.0 / 1e6;
            int pool = Math.Max(2, (int)Math.Ceiling(256.0 / oneMB));

            var ws = new Tensor[pool];
            for (int p = 0; p < pool; p++)
                ws[p] = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_K);

            for (int i = 0; i < pool * 2; i++) gpu.MatMul(gpuY, ws[i % pool], gpuX, DType.Q4_K);
            gpu.Synchronize();

            const int iters = 400;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) gpu.MatMul(gpuY, ws[i % pool], gpuX, DType.Q4_K);
            gpu.Synchronize();
            sw.Stop();

            for (int p = 0; p < pool; p++) gpu.Free(ws[p]);

            double msPer = sw.Elapsed.TotalMilliseconds / iters;
            double wBytes = (double)rows * cols / 256.0 * 144.0;   // Q4_K weight bytes
            double gbs = wBytes / (msPer * 1e-3) / 1e9;
            Console.WriteLine(
                $"matvec {what,-9} [{rows}x{cols}]: {msPer:F4} ms  {gbs:F0} GB/s  ({100 * gbs / peakGBs:F0}% of ~{peakGBs:F0} peak)");

            gpu.Free(gpuX); gpu.Free(gpuY);
        }
        Assert.True(true);
    }
}
