using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #149: the SoA-layout int8 MMQ (<see cref="CudaBackend.MatMulBatchedMmqSoa"/>,
/// kernel <c>llm_mmq_q8_0_soa</c>) repacks Q8_0 weights so the 32 quants/block are
/// contiguous &amp; 16-byte aligned and the fp16 scales are separate, eliminating the
/// 2-byte-misalignment <c>__funnelshift</c> the interleaved 34-byte block forces on
/// every weight word. This (a) checks it is <b>bit-identical</b> to the interleaved
/// <see cref="CudaBackend.MatMulBatchedMmq"/> (same int8 mma + scale math, only the
/// weight read differs) and (b) times both at an FFN-shaped prefill GEMM at the
/// <i>real</i> prefill nTok to measure the funnelshift-elimination speedup.
///
/// Silent no-op without CUDA.
/// </summary>
public sealed unsafe class CudaMmqSoaTests
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
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                int off = r * bytesPerRow + b * 34;
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.09 + 0.01));
                bytes[off] = (byte)(dHalf & 0xFF); bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        return bytes;
    }

    /// <summary>Repack interleaved Q8_0 [34 B/block] → a single SoA buffer
    /// [quants rows*cols B][scales rows*nb fp16] (matches CudaBackend's split at rows*cols).</summary>
    private static byte[] RepackToSoA(byte[] interleaved, int rows, int cols)
    {
        int nb = cols / 32, bytesPerRow = nb * 34;
        var soa = new byte[rows * cols + rows * nb * 2];
        int scaleBase = rows * cols;
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < nb; b++)
            {
                int off = r * bytesPerRow + b * 34;
                int blk = r * nb + b;
                soa[scaleBase + blk * 2] = interleaved[off]; soa[scaleBase + blk * 2 + 1] = interleaved[off + 1];
                Array.Copy(interleaved, off + 2, soa, blk * 32, 32);
            }
        return soa;
    }

    [Fact]
    public void MatMulBatchedMmqSoa_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (128, 2560, 64), (2048, 8192, 256), (6144, 2048, 200) })
        {
            var rng = new Random(20260606 + rows * 31 + cols * 7 + nTok);
            byte[] interleaved = BuildQ8_0Matrix(rows, cols, rng);
            byte[] soa = RepackToSoA(interleaved, rows, cols);

            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q8_0);
            var gSoa = gpu.UploadRaw(soa, TensorShape.D1(soa.Length), DType.Q8_0);
            var gX   = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gYi  = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var gYs  = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gYi, gW, gX, nTok, DType.Q8_0);
            gpu.MatMulBatchedMmqSoa(gYs, gSoa, gX, nTok);
            gpu.Synchronize();

            var yi = new float[nTok * rows];
            var ys = new float[nTok * rows];
            gpu.Download(gYi, yi);
            gpu.Download(gYs, ys);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gYi); gpu.Free(gYs);

            int diffs = 0; float maxAbs = 0;
            for (int i = 0; i < yi.Length; i++)
            {
                float d = MathF.Abs(yi[i] - ys[i]);
                maxAbs = MathF.Max(maxAbs, d);
                if (d != 0f) diffs++;
            }
            Console.WriteLine($"MMQ-SoA rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} diffs={diffs}/{yi.Length}");
            Assert.True(maxAbs == 0f,
                $"SoA MMQ not bit-identical to interleaved: {diffs}/{yi.Length} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }

    [Fact]
    public void MmqSoa_Vs_Interleaved_Speed_AtRealPrefillNtok()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // FFN/qkv prefill GEMMs at a REALISTIC prefill nTok (~2K prompt batched at once),
        // not the 1024 the earlier roofline probe used (which mismeasured occupancy).
        (int rows, int cols, int nTok, string what)[] shapes =
        {
            (8192, 2048, 2048, "ffn-gate/up"),
            (2048, 8192, 2048, "ffn-down"),
            (6144, 2048, 2048, "qkv"),
        };
        var rng = new Random(20260606);
        foreach (var (rows, cols, nTok, what) in shapes)
        {
            byte[] interleaved = BuildQ8_0Matrix(rows, cols, rng);
            byte[] soa = RepackToSoA(interleaved, rows, cols);
            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q8_0);
            var gSoa = gpu.UploadRaw(soa, TensorShape.D1(soa.Length), DType.Q8_0);
            var gX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            for (int i = 0; i < 5; i++) gpu.MatMulBatchedMmq(gY, gW, gX, nTok, DType.Q8_0);
            gpu.Synchronize();
            const int iters = 50;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) gpu.MatMulBatchedMmq(gY, gW, gX, nTok, DType.Q8_0);
            gpu.Synchronize(); sw.Stop();
            double msAos = sw.Elapsed.TotalMilliseconds / iters;

            for (int i = 0; i < 5; i++) gpu.MatMulBatchedMmqSoa(gY, gSoa, gX, nTok);
            gpu.Synchronize();
            sw.Restart();
            for (int i = 0; i < iters; i++) gpu.MatMulBatchedMmqSoa(gY, gSoa, gX, nTok);
            gpu.Synchronize(); sw.Stop();
            double msSoa = sw.Elapsed.TotalMilliseconds / iters;

            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gY);

            double macs = (double)rows * cols * nTok;
            double topsAos = 2.0 * macs / (msAos * 1e-3) / 1e12;
            double topsSoa = 2.0 * macs / (msSoa * 1e-3) / 1e12;
            Console.WriteLine(
                $"MMQ-SoA {what,-12} nTok={nTok}: AoS {msAos:F3}ms ({topsAos:F1} TOPS) → SoA {msSoa:F3}ms ({topsSoa:F1} TOPS)  {100*(msAos-msSoa)/msAos:+0.0;-0.0}%");
        }
        Assert.True(true);
    }

    /// <summary>
    /// Exercises the production <see cref="CudaBackend.RepackQ8_0Soa"/> (GPU repack) and
    /// the auto-routing through ALL five Q8_0 readers — dp4a + fp32 decode matvec, GEMM-N,
    /// dequant→GEMM, and MMQ — by repacking on the GPU and asserting each reader's output
    /// is bit-identical to the interleaved kernel. GGUF-free, runs on any CUDA box, so it
    /// is the durable regression net the model-level oracles (bench-machine only) lack.
    /// </summary>
    [Fact]
    public void GpuRepack_AllSoaReaders_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (256, 256), (1024, 512), (512, 2560) })
        {
            var rng = new Random(20260607 + rows * 7 + cols);
            byte[] interleaved = BuildQ8_0Matrix(rows, cols, rng);
            const int nTok = 20;

            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);
            var input = new float[(long)nTok * cols];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            var gVec = gpu.Upload(vec, TensorShape.D1(cols));
            var gIn = gpu.Upload(input, TensorShape.D1(input.Length));

            // Run `reader` on an interleaved weight and on a GPU-repacked SoA weight (the
            // backend auto-routes the SoA handle to the aligned-load kernel) → bit-identical.
            void Check(string label, int outElems, Action<Tensor, Tensor> reader)
            {
                var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q8_0);
                var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q8_0);
                var gSoa = gpu.RepackQ8_0Soa(gWr, rows, cols);   // frees gWr, marks gSoa SoA
                var gA = gpu.Allocate(TensorShape.D1(outElems));
                var gB = gpu.Allocate(TensorShape.D1(outElems));
                reader(gW, gA);     // interleaved
                reader(gSoa, gB);   // SoA (auto-routed)
                gpu.Synchronize();
                var a = new float[outElems]; var b = new float[outElems];
                gpu.Download(gA, a); gpu.Download(gB, b);
                gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gA); gpu.Free(gB);
                float maxAbs = 0; int diffs = 0;
                for (int i = 0; i < outElems; i++) { float d = MathF.Abs(a[i] - b[i]); maxAbs = MathF.Max(maxAbs, d); if (d != 0f) diffs++; }
                Console.WriteLine($"SoA-reader {label,-16} rows={rows} cols={cols}: maxAbs={maxAbs:E2} diffs={diffs}/{outElems}");
                Assert.True(maxAbs == 0f,
                    $"GPU-repacked SoA reader '{label}' not bit-identical to interleaved: {diffs}/{outElems} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols}).");
            }

            gpu.Q80Dp4aEnabled = true;
            Check("decode-dp4a", rows, (w, o) => gpu.MatMul(o, w, gVec, DType.Q8_0));
            gpu.Q80Dp4aEnabled = false;
            Check("decode-fp32", rows, (w, o) => gpu.MatMul(o, w, gVec, DType.Q8_0));
            gpu.Q80Dp4aEnabled = true;
            Check("gemm-n", nTok * rows, (w, o) => gpu.MatMulBatched(o, w, gIn, nTok, DType.Q8_0));
            Check("dequant-gemm", nTok * rows, (w, o) => gpu.MatMulBatchedGemm(o, w, gIn, nTok, DType.Q8_0));
            Check("mmq", nTok * rows, (w, o) => gpu.MatMulBatchedMmq(o, w, gIn, nTok, DType.Q8_0));

            gpu.Free(gVec); gpu.Free(gIn);
        }
    }
}
