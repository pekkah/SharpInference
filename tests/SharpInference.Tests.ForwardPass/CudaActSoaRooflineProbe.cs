using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Track A/B ncu probe (not a correctness test): runs the Q8_0 prefill MMQ at an
/// FFN-representative shape (nTok=2048) with the SoA <i>activation</i> path OFF
/// (<c>llm_mmq_q8_0_soa</c>, interleaved 36-B AoS Q8_1) and ON
/// (<c>llm_mmq_q8_0_soa_acts</c>, contiguous SoA Q8_1) so Nsight Compute can capture
/// both kernels in one run and the uncoalesced-global / L1TEX / occupancy counters can
/// be compared. The question it answers: did splitting the activation d/s out (so a
/// token's quants are contiguous) actually reduce the uncoalesced activation sectors
/// ncu flagged as the L1TEX ceiling on the AoS-acts kernel?
///
/// Run explicitly: --filter FullyQualifiedName~CudaActSoaRooflineProbe. Silent no-op
/// without CUDA. Always asserts true.
/// </summary>
public sealed unsafe class CudaActSoaRooflineProbe
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
        int nb = cols / 32, bytesPerRow = nb * 34;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 34;
                ushort d = HalfToUshort((Half)(float)(rng.NextDouble() * 0.09 + 0.01));
                bytes[off] = (byte)(d & 0xFF); bytes[off + 1] = (byte)(d >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        return bytes;
    }

    private static byte[] BuildQ4_0Matrix(int rows, int cols, Random rng)
    {
        int nb = cols / 32, bytesPerRow = nb * 18;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 18;
                ushort d = HalfToUshort((Half)(float)(rng.NextDouble() * 0.09 + 0.01));
                bytes[off] = (byte)(d & 0xFF); bytes[off + 1] = (byte)(d >> 8);
                for (int i = 0; i < 16; i++) bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    /// <summary>
    /// Track B: cp.async vs AoS at the Gemma 4 12B's REAL Q4_0 matmul dims (emb=3840,
    /// ffn=15360) across the prefill nTok range — to find why the +9-15% probe-shape win is
    /// e2e-neutral. ffn-down (cols=15360, large-K) was the neutral regime at the generic
    /// [8192x2048] probe; this pins it at the actual shapes.
    /// </summary>
    [Fact]
    public void ProbeActSoaMmq_Q40_Gemma12BDims()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // (rows=out, cols=in) for the 12B's dominant prefill GEMMs.
        (int rows, int cols, string what)[] dims =
        {
            (15360, 3840, "ffn-gate/up"),
            (3840, 15360, "ffn-down"),
            (4096, 3840, "qkv~"),
        };
        int[] nToks = { 256, 512, 1024, 2048 };
        var rng = new Random(20260608);

        foreach (var (rows, cols, what) in dims)
        {
            byte[] w = BuildQ4_0Matrix(rows, cols, rng);
            var gW = gpu.UploadRaw(w, TensorShape.D1(w.Length), DType.Q4_0);
            var gSoa = gpu.RepackQ4_0Soa(gW, rows, cols);
            foreach (int nTok in nToks)
            {
                var acts = new float[(long)nTok * cols];
                for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
                var gX = gpu.Upload(acts, TensorShape.D1(acts.Length));
                var gY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

                double Time(Action run)
                {
                    for (int i = 0; i < 8; i++) run();
                    gpu.Synchronize();
                    const int iters = 50;
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < iters; i++) run();
                    gpu.Synchronize(); sw.Stop();
                    return sw.Elapsed.TotalMilliseconds / iters;
                }
                gpu.ActSoaEnabled = false; gpu.ActSoaCpaEnabled = false;
                double msAos = Time(() => gpu.MatMulBatchedMmq(gY, gSoa, gX, nTok, DType.Q4_0));
                gpu.ActSoaEnabled = true; gpu.ActSoaCpaEnabled = true;
                double msCpa = Time(() => gpu.MatMulBatchedMmq(gY, gSoa, gX, nTok, DType.Q4_0));
                gpu.ActSoaEnabled = false; gpu.ActSoaCpaEnabled = false;

                double macs = (double)rows * cols * nTok;
                double tAos = 2.0 * macs / (msAos * 1e-3) / 1e12, tCpa = 2.0 * macs / (msCpa * 1e-3) / 1e12;
                Console.WriteLine($"q40-12b {what,-12} [{rows}x{cols}] nTok={nTok,4}: AoS {msAos:F3}ms ({tAos:F0}T) cpa {msCpa:F3}ms ({tCpa:F0}T)  {100*(msAos-msCpa)/msAos:+0.0;-0.0}%");
                gpu.Free(gX); gpu.Free(gY);
            }
            gpu.Free(gSoa);
        }
        Assert.True(true);
    }

    [Fact]
    public void ProbeActSoaMmq_FfnShape()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        (int rows, int cols, int nTok, string what)[] shapes =
        {
            (8192, 2048, 2048, "ffn-gate/up"),
            (2048, 8192, 2048, "ffn-down"),
            (6144, 2048, 2048, "qkv"),
            (8192, 2048, 512, "gate@512"),
        };
        var rng = new Random(20260608);
        foreach (var (rows, cols, nTok, what) in shapes)
        {
        byte[] w = BuildQ8_0Matrix(rows, cols, rng);
        var gW = gpu.UploadRaw(w, TensorShape.D1(w.Length), DType.Q8_0);
        var gSoa = gpu.RepackQ8_0Soa(gW, rows, cols);
        var acts = new float[(long)nTok * cols];
        for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
        var gX = gpu.Upload(acts, TensorShape.D1(acts.Length));
        var gY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

        double Time(Action run)
        {
            for (int i = 0; i < 8; i++) run();
            gpu.Synchronize();
            const int iters = 60;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) run();
            gpu.Synchronize(); sw.Stop();
            return sw.Elapsed.TotalMilliseconds / iters;
        }
        double macs = (double)rows * cols * nTok;
        double Tops(double ms) => 2.0 * macs / (ms * 1e-3) / 1e12;

        gpu.ActSoaEnabled = false; gpu.ActSoaCpaEnabled = false;
        double msAos = Time(() => gpu.MatMulBatchedMmq(gY, gSoa, gX, nTok, DType.Q8_0));   // llm_mmq_q8_0_soa
        gpu.ActSoaEnabled = true; gpu.ActSoaCpaEnabled = false;
        double msSoa = Time(() => gpu.MatMulBatchedMmq(gY, gSoa, gX, nTok, DType.Q8_0));   // llm_mmq_q8_0_soa_acts
        gpu.ActSoaCpaEnabled = true;
        double msCpa = Time(() => gpu.MatMulBatchedMmq(gY, gSoa, gX, nTok, DType.Q8_0));   // ..._soa_acts_cpa
        gpu.ActSoaEnabled = false; gpu.ActSoaCpaEnabled = false;

        gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gY);
        Console.WriteLine(
            $"probe {what,-12} [{rows}x{cols}]x{nTok}: AoS {msAos:F3}ms ({Tops(msAos):F1}T) | " +
            $"SoA {msSoa:F3}ms ({Tops(msSoa):F1}T) | cp.async {msCpa:F3}ms ({Tops(msCpa):F1}T) | " +
            $"cpa vs AoS {100*(msAos-msCpa)/msAos:+0.0;-0.0}%");
        }
        Assert.True(true);
    }
}
