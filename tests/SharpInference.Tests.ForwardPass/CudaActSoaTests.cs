using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Track A (#124/#173): the SoA Q8_1 <i>activation</i> layout (<c>llm_quantize_q8_1_soa</c>
/// → the <c>llm_mmq_*_soa_acts</c> kernels). The interleaved 36-B AoS Q8_1 block is split
/// into a contiguous int8-quants array + a separate {d,s} array, so a token's quants are
/// aligned/contiguous — the substrate Phase B's coalesced per-token load reads. Phase A
/// keeps the SAME load mapping / fragment map / accumulation order, so the SoA-activation
/// MMQ must be <b>bit-identical</b> to the AoS-activation MMQ over the same SoA-repacked
/// weight. This A/Bs <see cref="CudaBackend.ActSoaEnabled"/> off vs on across all three
/// production prefill quant types (Q8_0, Q4_0, Q4_K) and asserts maxAbs==0.
///
/// This is the durable, GGUF-free regression net for the activation-SoA producer + the
/// three SoA-activation MMQ kernels (the model-level oracles run bench-machine only).
/// Silent no-op without CUDA.
/// </summary>
public sealed unsafe class CudaActSoaTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    // block_q8_0 = 34 B / 32 elems: [d:fp16][qs:32 × int8].
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

    // block_q4_0 = 18 B / 32 elems: [d:fp16][qs:16 × uint8].
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

    /// <summary>
    /// For each quant type: build a weight, GPU-repack it SoA (the production prefill
    /// weight layout), then run the prefill MMQ with ActSoaEnabled off (interleaved Q8_1
    /// activations) and on (SoA Q8_1 activations) over that same weight → bit-identical.
    /// </summary>
    [Fact]
    public void ActSoaMmq_BitIdenticalToAosActs_AllQuants()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // (rows, cols, nTok): widths spanning Gemma-12B FFN/qkv shapes + small tiles, with
        // nTok both < and > MMQ_BN (128) to exercise multiple token-blocks. Q4_K needs
        // cols % 256; Q8_0/Q4_0 need cols % 32 — all listed cols satisfy 256.
        (int rows, int cols, int nTok)[] shapes =
        {
            (256, 256, 8),
            (1024, 512, 40),
            (2048, 2048, 200),
            (8192, 2048, 130),
            (3840, 3840, 96),
        };

        (DType dt, Func<int, int, Random, byte[]> build, Func<Tensor, int, int, Tensor> repack)[] quants =
        {
            (DType.Q8_0, BuildQ8_0Matrix, (w, r, c) => gpu.RepackQ8_0Soa(w, r, c)),
            (DType.Q4_0, BuildQ4_0Matrix, (w, r, c) => gpu.RepackQ4_0Soa(w, r, c)),
            (DType.Q4_K, BuildQ4KMatrix, (w, r, c) => gpu.RepackQ4KSoa(w, r, c)),
        };

        foreach (var (dt, build, repack) in quants)
            foreach (var (rows, cols, nTok) in shapes)
            {
                var rng = new Random(20260608 + rows * 31 + cols * 7 + nTok + (int)dt * 101);
                byte[] w = build(rows, cols, rng);
                var gW = gpu.UploadRaw(w, TensorShape.D1(w.Length), dt);
                var gSoa = repack(gW, rows, cols);   // frees gW, marks gSoa SoA

                var acts = new float[(long)nTok * cols];
                for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
                var gX = gpu.Upload(acts, TensorShape.D1(acts.Length));
                var gYa = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                var gYs = gpu.Allocate(TensorShape.D1((long)nTok * rows));

                gpu.ActSoaEnabled = false;
                gpu.MatMulBatchedMmq(gYa, gSoa, gX, nTok, dt);   // AoS Q8_1 activations
                gpu.ActSoaEnabled = true;
                gpu.MatMulBatchedMmq(gYs, gSoa, gX, nTok, dt);   // SoA Q8_1 activations
                gpu.Synchronize();
                gpu.ActSoaEnabled = false;

                var ya = new float[(long)nTok * rows];
                var ys = new float[(long)nTok * rows];
                gpu.Download(gYa, ya);
                gpu.Download(gYs, ys);
                gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gYa); gpu.Free(gYs);

                int diffs = 0; float maxAbs = 0;
                for (int i = 0; i < ya.Length; i++)
                {
                    float d = MathF.Abs(ya[i] - ys[i]);
                    maxAbs = MathF.Max(maxAbs, d);
                    if (d != 0f) diffs++;
                }
                Console.WriteLine($"act-SoA {dt} rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} diffs={diffs}/{ya.Length}");
                Assert.True(maxAbs == 0f,
                    $"SoA-activation MMQ not bit-identical to AoS-activation MMQ ({dt}): " +
                    $"{diffs}/{ya.Length} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols} nTok={nTok}).");
            }
    }

    /// <summary>
    /// Track B: the cp.async double-buffered Q8_0 MMQ (<c>llm_mmq_q8_0_soa_acts_cpa</c>,
    /// <see cref="CudaBackend.ActSoaCpaEnabled"/>) streams global→shared off the L1TEX LSU
    /// pipe, but stages the SAME int8 quants into the SAME shared tiles with the SAME
    /// K-order accumulation, so it must be bit-identical to the scalar-load SoA-acts MMQ.
    /// cols straddle K-block parity and nTok straddle the 128-token tile; rows straddle the
    /// 64-row tile (tail rows hit the cp.async zero-fill path).
    /// </summary>
    [Theory]
    [InlineData(DType.Q8_0)]
    [InlineData(DType.Q4_0)]
    public void ActSoaCpaMmq_BitIdenticalToSoaActs(DType dt)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        (int rows, int cols, int nTok)[] shapes =
        {
            (256, 256, 8),
            (1024, 512, 40),
            (2048, 2048, 200),
            (8192, 2048, 130),
            (130, 1184, 96),   // tail rows (130 % 64) + tail tokens
        };

        foreach (var (rows, cols, nTok) in shapes)
        {
            var rng = new Random(20260608 + rows * 17 + cols * 3 + nTok + (int)dt * 911);
            byte[] w = dt == DType.Q8_0 ? BuildQ8_0Matrix(rows, cols, rng) : BuildQ4_0Matrix(rows, cols, rng);
            var gW = gpu.UploadRaw(w, TensorShape.D1(w.Length), dt);
            var gSoa = dt == DType.Q8_0 ? gpu.RepackQ8_0Soa(gW, rows, cols) : gpu.RepackQ4_0Soa(gW, rows, cols);

            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
            var gX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gYs = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var gYc = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.ActSoaEnabled = true;
            gpu.ActSoaCpaEnabled = false;
            gpu.MatMulBatchedMmq(gYs, gSoa, gX, nTok, dt);   // scalar-load SoA acts
            gpu.ActSoaCpaEnabled = true;
            gpu.MatMulBatchedMmq(gYc, gSoa, gX, nTok, dt);   // cp.async pipelined
            gpu.Synchronize();
            gpu.ActSoaEnabled = false;
            gpu.ActSoaCpaEnabled = false;

            var ys = new float[(long)nTok * rows];
            var yc = new float[(long)nTok * rows];
            gpu.Download(gYs, ys);
            gpu.Download(gYc, yc);
            gpu.Free(gSoa); gpu.Free(gX); gpu.Free(gYs); gpu.Free(gYc);

            int diffs = 0; float maxAbs = 0;
            for (int i = 0; i < ys.Length; i++)
            {
                float d = MathF.Abs(ys[i] - yc[i]);
                maxAbs = MathF.Max(maxAbs, d);
                if (d != 0f) diffs++;
            }
            Console.WriteLine($"act-SoA-cpa {dt} rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} diffs={diffs}/{ys.Length}");
            Assert.True(maxAbs == 0f,
                $"cp.async MMQ ({dt}) not bit-identical to scalar-load SoA-activation MMQ: " +
                $"{diffs}/{ys.Length} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }
}
