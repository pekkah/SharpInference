using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #156: the scale-pre-unpacked Q4_K SoA decode matvec
/// (<see cref="CudaBackend.RepackQ4KSoa"/> + kernel <c>llm_matvec_q4k_soa</c>) repacks
/// each 144-byte Q4_K super-block into [Q quants][S unpacked scale/min bytes][D d/dmin]
/// so the matvec reads plain bytes instead of running the per-super-block 6-bit
/// (scale,min) unpack switch — recovering the memory-level parallelism the dependent
/// unpack chain starved. The stored scale/min integers are identical to the AoS switch
/// output, so this (a) asserts <b>bit-identical</b> output vs the interleaved
/// <c>llm_matvec_q4k</c> path and (b) A/Bs the two decode matvecs at Qwen3-8B shapes.
///
/// Silent no-op without CUDA.
/// </summary>
public sealed unsafe class CudaQ4KSoaTests
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
        int nb = cols / 256, bytesPerRow = nb * 144;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
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
    public void Q4KSoaDecodeMatvec_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[]
                 { (256, 256), (1024, 512), (4096, 4096), (6144, 4096), (4096, 12288), (12288, 4096) })
        {
            var rng = new Random(20260606 + rows * 31 + cols * 7);
            byte[] interleaved = BuildQ4KMatrix(rows, cols, rng);
            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);
            var gVec = gpu.Upload(vec, TensorShape.D1(cols));

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gSoa = gpu.RepackQ4KSoa(gWr, rows, cols);   // frees gWr, marks gSoa SoA
            var gAos = gpu.Allocate(TensorShape.D1(rows));
            var gSo  = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gAos, gW, gVec, DType.Q4_K);    // interleaved llm_matvec_q4k
            gpu.MatMul(gSo, gSoa, gVec, DType.Q4_K);   // SoA (auto-routed) llm_matvec_q4k_soa
            gpu.Synchronize();

            var a = new float[rows]; var b = new float[rows];
            gpu.Download(gAos, a); gpu.Download(gSo, b);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gAos); gpu.Free(gSo); gpu.Free(gVec);

            int diffs = 0; float maxAbs = 0;
            for (int i = 0; i < rows; i++)
            {
                float d = MathF.Abs(a[i] - b[i]);
                maxAbs = MathF.Max(maxAbs, d);
                if (d != 0f) diffs++;
            }
            Console.WriteLine($"Q4K-SoA rows={rows} cols={cols}: maxAbs={maxAbs:E2} diffs={diffs}/{rows}");
            Assert.True(maxAbs == 0f,
                $"Q4_K SoA decode matvec not bit-identical to interleaved: {diffs}/{rows} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols}).");
        }
    }

    [Fact]
    public void Q4KSoaMmqPrefill_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Exercises the prefill reader: a Q4_K weight repacked to SoA must give MMQ output
        // bit-identical to the interleaved llm_mmq_q4k (same int8 mma, only the weight read
        // layout differs) — the other half of the dual-use weight (decode is covered above).
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 40), (4096, 4096, 200), (6144, 4096, 333) })
        {
            var rng = new Random(20260606 + rows * 13 + cols * 5 + nTok);
            byte[] interleaved = BuildQ4KMatrix(rows, cols, rng);
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gSoa = gpu.RepackQ4KSoa(gWr, rows, cols);
            var gX   = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gYi  = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var gYs  = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gYi, gW, gX, nTok, DType.Q4_K);    // interleaved llm_mmq_q4k
            gpu.MatMulBatchedMmq(gYs, gSoa, gX, nTok, DType.Q4_K);  // SoA (auto-routed) llm_mmq_q4k_soa
            gpu.Synchronize();

            var yi = new float[(long)nTok * rows]; var ys = new float[(long)nTok * rows];
            gpu.Download(gYi, yi); gpu.Download(gYs, ys);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gYi); gpu.Free(gYs);

            int diffs = 0; float maxAbs = 0;
            for (int i = 0; i < yi.Length; i++)
            {
                float d = MathF.Abs(yi[i] - ys[i]);
                maxAbs = MathF.Max(maxAbs, d);
                if (d != 0f) diffs++;
            }
            Console.WriteLine($"Q4K-MMQ-SoA rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} diffs={diffs}/{yi.Length}");
            Assert.True(maxAbs == 0f,
                $"Q4_K SoA MMQ not bit-identical to interleaved: {diffs}/{yi.Length} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }

    [Fact]
    public void Q4KSoaN2_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Exercises the MTP batched-verify reader (issue #160): the dense Qwen3.6-MTP
        // trunk weights are repacked SoA when SHARPI_Q4K_SOA is default-on, so MatMulN2
        // must route to llm_matvec_q4k_n2_soa and stay bit-identical to the interleaved
        // llm_matvec_q4k_n2 (same two-input dp4a + accumulation order). Bit-parity here
        // is the gate that protects MtpDecoder_GreedyParity_LlamaCpp's cumulative-drift
        // oracle (see memory feedback_qkv_matvecdual_breaks_mtp_parity).
        foreach ((int rows, int cols) in new[]
                 { (256, 256), (1024, 512), (4096, 4096), (6144, 4096), (4096, 12288), (12288, 4096) })
        {
            var rng = new Random(20260606 + rows * 17 + cols * 3);
            byte[] interleaved = BuildQ4KMatrix(rows, cols, rng);
            var vecA = new float[cols];
            var vecB = new float[cols];
            for (int i = 0; i < cols; i++) { vecA[i] = (float)(rng.NextDouble() * 2 - 1); vecB[i] = (float)(rng.NextDouble() * 2 - 1); }
            var gA = gpu.Upload(vecA, TensorShape.D1(cols));
            var gB = gpu.Upload(vecB, TensorShape.D1(cols));

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gSoa = gpu.RepackQ4KSoa(gWr, rows, cols);
            var aA = gpu.Allocate(TensorShape.D1(rows)); var aB = gpu.Allocate(TensorShape.D1(rows));
            var sA = gpu.Allocate(TensorShape.D1(rows)); var sB = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMulN2(aA, aB, gW,   gA, gB, DType.Q4_K);  // interleaved llm_matvec_q4k_n2
            gpu.MatMulN2(sA, sB, gSoa, gA, gB, DType.Q4_K);  // SoA (auto-routed) llm_matvec_q4k_n2_soa
            gpu.Synchronize();

            var haA = new float[rows]; var haB = new float[rows];
            var hsA = new float[rows]; var hsB = new float[rows];
            gpu.Download(aA, haA); gpu.Download(aB, haB); gpu.Download(sA, hsA); gpu.Download(sB, hsB);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gA); gpu.Free(gB);
            gpu.Free(aA); gpu.Free(aB); gpu.Free(sA); gpu.Free(sB);

            int diffs = 0; float maxAbs = 0;
            for (int i = 0; i < rows; i++)
            {
                float da = MathF.Abs(haA[i] - hsA[i]), db = MathF.Abs(haB[i] - hsB[i]);
                maxAbs = MathF.Max(maxAbs, MathF.Max(da, db));
                if (da != 0f || db != 0f) diffs++;
            }
            Console.WriteLine($"Q4K-N2-SoA rows={rows} cols={cols}: maxAbs={maxAbs:E2} diffs={diffs}/{rows}");
            Assert.True(maxAbs == 0f,
                $"Q4_K SoA N=2 matvec not bit-identical to interleaved: {diffs}/{rows} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols}).");
        }
    }

    [Fact]
    public void Q4KSoaFallbackPrefill_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // The two non-MMQ prefill fallback readers (SHARPI_PREFILL_MMQ=0): the GEMM-N
        // dp4a path (MatMulBatched → llm_matvec_q4k_gemm_n_soa) and the dequant→fp16→GEMM
        // path (MatMulBatchedGemm → llm_dequant_q4k_to_f16_soa). Both must be bit-identical
        // to their interleaved counterparts so default-on SoA is safe with MMQ disabled.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 40), (4096, 4096, 64) })
        {
            var rng = new Random(20260606 + rows * 11 + cols * 9 + nTok);
            byte[] interleaved = BuildQ4KMatrix(rows, cols, rng);
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
            var gSoa = gpu.RepackQ4KSoa(gWr, rows, cols);
            var gX   = gpu.Upload(acts, TensorShape.D1(acts.Length));

            // GEMM-N dp4a fallback.
            var gnI = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var gnS = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            gpu.MatMulBatched(gnI, gW,   gX, nTok, DType.Q4_K);
            gpu.MatMulBatched(gnS, gSoa, gX, nTok, DType.Q4_K);

            // dequant→fp16→GEMM fallback.
            var dqI = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var dqS = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            gpu.MatMulBatchedGemm(dqI, gW,   gX, nTok, DType.Q4_K);
            gpu.MatMulBatchedGemm(dqS, gSoa, gX, nTok, DType.Q4_K);
            gpu.Synchronize();

            var hgnI = new float[(long)nTok * rows]; var hgnS = new float[(long)nTok * rows];
            var hdqI = new float[(long)nTok * rows]; var hdqS = new float[(long)nTok * rows];
            gpu.Download(gnI, hgnI); gpu.Download(gnS, hgnS);
            gpu.Download(dqI, hdqI); gpu.Download(dqS, hdqS);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gX);
            gpu.Free(gnI); gpu.Free(gnS); gpu.Free(dqI); gpu.Free(dqS);

            float gnMax = 0, dqMax = 0; int gnDiffs = 0, dqDiffs = 0;
            for (int i = 0; i < hgnI.Length; i++)
            {
                float g = MathF.Abs(hgnI[i] - hgnS[i]); gnMax = MathF.Max(gnMax, g); if (g != 0f) gnDiffs++;
                float d = MathF.Abs(hdqI[i] - hdqS[i]); dqMax = MathF.Max(dqMax, d); if (d != 0f) dqDiffs++;
            }
            Console.WriteLine($"Q4K-fallback rows={rows} cols={cols} nTok={nTok}: gemm-n maxAbs={gnMax:E2} ({gnDiffs}), dequant maxAbs={dqMax:E2} ({dqDiffs})");
            Assert.True(gnMax == 0f, $"Q4_K SoA GEMM-N not bit-identical: {gnDiffs} differ, maxAbs={gnMax:E3} (rows={rows} cols={cols} nTok={nTok}).");
            Assert.True(dqMax == 0f, $"Q4_K SoA dequant→GEMM not bit-identical: {dqDiffs} differ, maxAbs={dqMax:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }

    [Fact]
    public void Q4KSoa_Vs_Interleaved_DecodeSpeed()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        (int rows, int cols, string what)[] shapes =
        {
            (12288, 4096, "ffn-gate/up"),
            (4096, 12288, "ffn-down"),
            (6144, 4096, "qkv"),
            (4096, 4096, "o-proj"),
        };
        const double peakGBs = 504.0;
        var rng = new Random(20260606);
        foreach (var (rows, cols, what) in shapes)
        {
            byte[] interleaved = BuildQ4KMatrix(rows, cols, rng);
            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);
            var gVec = gpu.Upload(vec, TensorShape.D1(cols));
            var gY = gpu.Allocate(TensorShape.D1(rows));

            // Rotate ~256 MB of distinct buffers to exceed L2 → cold HBM reads (real decode).
            double oneMB = (double)rows * cols / 256.0 * 144.0 / 1e6;
            int pool = Math.Max(2, (int)Math.Ceiling(256.0 / oneMB));
            double wBytesAos = (double)rows * cols / 256.0 * 144.0;
            double wBytesSoa = (double)rows * cols / 256.0 * 148.0;

            double Time(bool soa)
            {
                var ws = new Tensor[pool];
                for (int p = 0; p < pool; p++)
                {
                    ws[p] = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_K);
                    if (soa) ws[p] = gpu.RepackQ4KSoa(ws[p], rows, cols);
                }
                for (int i = 0; i < pool * 2; i++) gpu.MatMul(gY, ws[i % pool], gVec, DType.Q4_K);
                gpu.Synchronize();
                const int iters = 400;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) gpu.MatMul(gY, ws[i % pool], gVec, DType.Q4_K);
                gpu.Synchronize(); sw.Stop();
                for (int p = 0; p < pool; p++) gpu.Free(ws[p]);
                return sw.Elapsed.TotalMilliseconds / iters;
            }

            double msAos = Time(false);
            double msSoa = Time(true);
            gpu.Free(gVec); gpu.Free(gY);

            double gbsAos = wBytesAos / (msAos * 1e-3) / 1e9;
            double gbsSoa = wBytesSoa / (msSoa * 1e-3) / 1e9;
            Console.WriteLine(
                $"Q4K-decode {what,-12} [{rows}x{cols}]: AoS {msAos:F4}ms ({gbsAos:F0} GB/s {100*gbsAos/peakGBs:F0}%) → SoA {msSoa:F4}ms ({gbsSoa:F0} GB/s {100*gbsSoa/peakGBs:F0}%)  {100*(msAos-msSoa)/msAos:+0.0;-0.0}%");
        }
        Assert.True(true);
    }
}
