using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #124/#173 (mirrors #149): the Q4_0 SoA repack
/// (<see cref="CudaBackend.RepackQ4_0Soa"/>) splits each 18-byte Q4_0 block into a
/// [quants rows*cols/2 B (16 B/block, 16-byte aligned)][scales rows*nb fp16] layout so
/// every reader uses plain aligned uint loads instead of the qs-misalignment funnelshift
/// (18 isn't a multiple of 4 → half the blocks' qs start 2-byte misaligned). The repacked
/// quant bytes + fp16 scales are bit-identical to the interleaved block, so each SoA
/// reader is <b>bit-identical</b> to its AoS twin. This asserts maxAbs==0 across all four
/// production readers (decode dp4a, decode fp32 matvec, prefill MMQ, prefill dequant→GEMM
/// fallback) and A/Bs each. The SoA MMQ is bit-identical to <c>llm_mmq_q4_0</c> (itself
/// only argmax-stable vs fp — both operands int8-quantized).
///
/// Silent no-op without CUDA.
/// </summary>
public sealed unsafe class CudaQ40SoaTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) => BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

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

    /// <summary>Decode matvec, both dp4a (default) and fp32 (Q40Dp4aEnabled=false).</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Q40SoaDecodeMatvec_BitIdenticalToInterleaved(bool dp4a)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        gpu.Q40Dp4aEnabled = dp4a;

        // Gemma 4 12B widths + an odd row count for the 8-rows/block tail.
        foreach ((int rows, int cols) in new[]
                 { (256, 256), (33, 128), (3840, 3840), (15360, 3840), (3840, 15360), (4096, 3840) })
        {
            var rng = new Random(20260608 + rows * 31 + cols * 7 + (dp4a ? 1 : 0));
            byte[] interleaved = BuildQ4_0Matrix(rows, cols, rng);
            var vec = new float[cols];
            for (int i = 0; i < vec.Length; i++) vec[i] = (float)(rng.NextDouble() * 2 - 1);
            var gVec = gpu.Upload(vec, TensorShape.D1(cols));

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gSoa = gpu.RepackQ4_0Soa(gWr, rows, cols);   // frees gWr, marks gSoa SoA
            var gAos = gpu.Allocate(TensorShape.D1(rows));
            var gSo  = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gAos, gW, gVec, DType.Q4_0);    // interleaved (dp4a or fp32)
            gpu.MatMul(gSo, gSoa, gVec, DType.Q4_0);   // SoA (auto-routed)
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
            Console.WriteLine($"Q40-SoA-decode(dp4a={dp4a}) rows={rows} cols={cols}: maxAbs={maxAbs:E2} diffs={diffs}/{rows}");
            Assert.True(maxAbs == 0f,
                $"Q4_0 SoA decode matvec (dp4a={dp4a}) not bit-identical to interleaved: {diffs}/{rows} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols}).");
        }
    }

    [Fact]
    public void Q40SoaMmqPrefill_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (33, 256, 5), (3840, 3840, 64), (15360, 3840, 128), (3840, 15360, 200) })
        {
            var rng = new Random(20260608 + rows * 13 + cols * 5 + nTok);
            byte[] interleaved = BuildQ4_0Matrix(rows, cols, rng);
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gSoa = gpu.RepackQ4_0Soa(gWr, rows, cols);
            var gX   = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gYi  = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var gYs  = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gYi, gW, gX, nTok, DType.Q4_0);    // interleaved llm_mmq_q4_0
            gpu.MatMulBatchedMmq(gYs, gSoa, gX, nTok, DType.Q4_0);  // SoA (auto-routed) llm_mmq_q4_0_soa
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
            Console.WriteLine($"Q40-MMQ-SoA rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} diffs={diffs}/{yi.Length}");
            Assert.True(maxAbs == 0f,
                $"Q4_0 SoA MMQ not bit-identical to interleaved: {diffs}/{yi.Length} differ, maxAbs={maxAbs:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }

    [Fact]
    public void Q40SoaDequantGemm_BitIdenticalToInterleaved()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // The dequant→fp16→cuBLAS GEMM fallback prefill reader (SHARPI_PREFILL_MMQ=0):
        // a repacked SoA weight must give MatMulBatchedGemm output bit-identical to the
        // interleaved llm_dequant_q4_0_to_f16 path, so default-on SoA is safe with MMQ off.
        // (MatMulBatched / GEMM-N is intentionally untested: it throws for Q4_0 — that
        // dtype was never wired into the per-token GEMM-N batched fallback.)
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (3840, 3840, 64), (3840, 15360, 64) })
        {
            var rng = new Random(20260608 + rows * 11 + cols * 9 + nTok);
            byte[] interleaved = BuildQ4_0Matrix(rows, cols, rng);
            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);

            var gW   = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gWr  = gpu.UploadRaw(interleaved, TensorShape.D1(interleaved.Length), DType.Q4_0);
            var gSoa = gpu.RepackQ4_0Soa(gWr, rows, cols);
            var gX   = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var dqI  = gpu.Allocate(TensorShape.D1((long)nTok * rows));
            var dqS  = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedGemm(dqI, gW,   gX, nTok, DType.Q4_0);  // interleaved dequant
            gpu.MatMulBatchedGemm(dqS, gSoa, gX, nTok, DType.Q4_0);  // SoA (auto-routed) dequant
            gpu.Synchronize();

            var hdqI = new float[(long)nTok * rows]; var hdqS = new float[(long)nTok * rows];
            gpu.Download(dqI, hdqI); gpu.Download(dqS, hdqS);
            gpu.Free(gW); gpu.Free(gSoa); gpu.Free(gX); gpu.Free(dqI); gpu.Free(dqS);

            float dqMax = 0; int dqDiffs = 0;
            for (int i = 0; i < hdqI.Length; i++)
            {
                float d = MathF.Abs(hdqI[i] - hdqS[i]); dqMax = MathF.Max(dqMax, d); if (d != 0f) dqDiffs++;
            }
            Console.WriteLine($"Q40-dequant-SoA rows={rows} cols={cols} nTok={nTok}: maxAbs={dqMax:E2} diffs={dqDiffs}/{hdqI.Length}");
            Assert.True(dqMax == 0f,
                $"Q4_0 SoA dequant→GEMM not bit-identical: {dqDiffs} differ, maxAbs={dqMax:E3} (rows={rows} cols={cols} nTok={nTok}).");
        }
    }
}
