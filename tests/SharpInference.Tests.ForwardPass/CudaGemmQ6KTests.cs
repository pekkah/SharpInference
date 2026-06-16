using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #162: parity for the Q6_K dequant→fp16→cuBLAS GEMM batched prefill path
/// (<see cref="CudaBackend.MatMulBatchedGemm"/> with <see cref="DType.Q6_K"/>, kernel
/// <c>llm_dequant_q6k_to_f16</c>). Qwen3-8B-Q4_K_M keeps ~half of ffn_down + attn_v in
/// Q6_K; before this path those trunk matmuls fell back to the per-token GEMM-N matvec
/// (weight re-streamed once per token) — the dominant large-N prefill cost.
///
/// A small Q6_K weight matrix [rows×cols] and an fp32 activation batch [nTok×cols] are
/// multiplied on the GPU (dequant→fp16 GEMM) and on the CPU (<see cref="SimdKernels.DotQ6K"/>,
/// fp32 reference). The weight + activation are rounded to fp16 before the GEMM, so the
/// result tracks the fp32 reference to a loose per-RMS tolerance rather than bit-exactly
/// — this validates the Q6_K element decode (ql/qh split + 16 int8 scales) and the row-
/// major output layout, isolated from the model.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaGemmQ6KTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // Canonical fp16 bit pattern as ushort (matches the GPU/CPU fp16 decode).
    private static ushort HalfToUshort(Half h) => BitConverter.HalfToUInt16Bits(h);

    /// <summary>Build a <paramref name="rows"/>×<paramref name="cols"/> Q6_K matrix
    /// (210 B / 256-element super-block: 128 ql, 64 qh, 16 int8 scales, fp16 d).</summary>
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[(long)rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = (long)r * bytesPerRow + (long)b * 210;
                // 192 ql/qh bytes: any pattern is a valid 6-bit packing.
                for (int i = 0; i < 192; i++) bytes[off + i] = (byte)rng.Next(256);
                // 16 signed int8 scales (small magnitude → realistic).
                for (int i = 0; i < 16; i++) bytes[off + 192 + i] = (byte)(sbyte)(rng.Next(33) - 16);
                // fp16 super-block scale d.
                ushort dHalf = HalfToUshort((Half)(float)(rng.NextDouble() * 0.04 + 0.005));
                bytes[off + 208] = (byte)(dHalf & 0xFF);
                bytes[off + 209] = (byte)(dHalf >> 8);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedGemm_Q6K_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Square, a wide multi-superblock single-token-tile batch, and a partial-tile
        // (non-power-of-two rows) case to exercise the GEMM tail.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (128, 2560, 64), (300, 256, 5) })
        {
            var rng = new Random(20260607 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ6KMatrix(rows, cols, rng);

            var acts = new float[(long)nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, exact-byte Q6_K).
            int bytesPerRow = (cols / 256) * 210;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ6K(wPtr + (long)r * bytesPerRow, aPtr + (long)t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q6_K);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedGemm(gpuY, gpuW, gpuX, nTok, DType.Q6_K);
            gpu.Synchronize();

            var gpuOut = new float[nTok * rows];
            gpu.Download(gpuY, gpuOut);
            gpu.Free(gpuW);
            gpu.Free(gpuX);
            gpu.Free(gpuY);

            double sumSq = 0;
            for (int i = 0; i < cpuOut.Length; i++) sumSq += (double)cpuOut[i] * cpuOut[i];
            float refRms = (float)Math.Sqrt(sumSq / cpuOut.Length);

            int mismatches = 0;
            float maxAbs = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 0.04f * refRms) mismatches++;
            }
            Console.WriteLine(
                $"GEMM-Q6K rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"Q6_K GEMM drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 4% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }

    /// <summary>#204: every Q6_K reader ported to the SoA layout
    /// (<see cref="CudaBackend.RepackQ6KSoa"/>) must be BIT-IDENTICAL to its AoS counterpart —
    /// same value, same reduction order. This test runs each reader (single-token
    /// <c>MatMul</c>, two-input <c>MatMulN2</c>, batched GEMM-N <c>MatMulBatched</c>, and the
    /// weight-stationary <c>MatMulBatchedWeightStationary</c>) against the SAME weight bytes,
    /// once via the AoS-uploaded weight and once via the SoA-repacked weight, and asserts the
    /// two GPU outputs are byte-for-byte equal. (The decode-MMQ tile is only argmax-stable, so
    /// it is covered separately by <see cref="CudaDecodeMmqTests"/> against the fp32 reference.)
    /// </summary>
    [Fact]
    public void Q6KSoaReaders_AreBitIdenticalToAos()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Shapes: attn_v-like (1024 rows → WS/decode-ineligible), ffn_down-half-like (2048),
        // a partial-tile (300), and a wider multi-superblock (512 rows × 1024 cols).
        foreach ((int rows, int cols) in new[] { (1024, 512), (2048, 256), (300, 512), (512, 1024) })
        {
            var rng = new Random(20260616 + rows * 17 + cols * 3);
            byte[] weightBytes = BuildQ6KMatrix(rows, cols, rng);

            // Two GPU copies of the SAME bytes: AoS (interleaved) and SoA (repack frees its src).
            var aosW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q6_K);
            var soaSrc = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q6_K);
            var soaW = gpu.RepackQ6KSoa(soaSrc, rows, cols);   // frees soaSrc, returns the SoA buffer

            // ── Single-token MatMul ──
            {
                var x = new float[cols];
                for (int i = 0; i < cols; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
                var gx = gpu.Upload(x, TensorShape.D1(cols));
                var aosY = gpu.Allocate(TensorShape.D1(rows));
                var soaY = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(aosY, aosW, gx, DType.Q6_K);
                gpu.MatMul(soaY, soaW, gx, DType.Q6_K);
                gpu.Synchronize();
                AssertBitIdentical(gpu, aosY, soaY, rows, $"MatMul rows={rows} cols={cols}");
                gpu.Free(gx); gpu.Free(aosY); gpu.Free(soaY);
            }

            // ── Two-input MatMulN2 ──
            {
                var xa = new float[cols];
                var xb = new float[cols];
                for (int i = 0; i < cols; i++) { xa[i] = (float)(rng.NextDouble() * 2 - 1); xb[i] = (float)(rng.NextDouble() * 2 - 1); }
                var gxa = gpu.Upload(xa, TensorShape.D1(cols));
                var gxb = gpu.Upload(xb, TensorShape.D1(cols));
                var aosYa = gpu.Allocate(TensorShape.D1(rows));
                var aosYb = gpu.Allocate(TensorShape.D1(rows));
                var soaYa = gpu.Allocate(TensorShape.D1(rows));
                var soaYb = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMulN2(aosYa, aosYb, aosW, gxa, gxb, DType.Q6_K);
                gpu.MatMulN2(soaYa, soaYb, soaW, gxa, gxb, DType.Q6_K);
                gpu.Synchronize();
                AssertBitIdentical(gpu, aosYa, soaYa, rows, $"MatMulN2.A rows={rows} cols={cols}");
                AssertBitIdentical(gpu, aosYb, soaYb, rows, $"MatMulN2.B rows={rows} cols={cols}");
                gpu.Free(gxa); gpu.Free(gxb);
                gpu.Free(aosYa); gpu.Free(aosYb); gpu.Free(soaYa); gpu.Free(soaYb);
            }

            // ── Batched GEMM-N (MatMulBatched) and weight-stationary, across capacities ──
            foreach (int nTok in new[] { 1, 2, 4, 8, 16 })
            {
                var acts = new float[(long)nTok * cols];
                for (int i = 0; i < acts.Length; i++) acts[i] = (float)(rng.NextDouble() * 2 - 1);
                var gx = gpu.Upload(acts, TensorShape.D1(acts.Length));

                var aosY = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                var soaY = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                gpu.MatMulBatched(aosY, aosW, gx, nTok, DType.Q6_K);
                gpu.MatMulBatched(soaY, soaW, gx, nTok, DType.Q6_K);
                gpu.Synchronize();
                AssertBitIdentical(gpu, aosY, soaY, nTok * rows, $"MatMulBatched n={nTok} rows={rows} cols={cols}");

                // WS path (nTok ≥ 2 uses the WS kernels; nTok == 1 delegates to GEMM-N).
                var aosWs = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                var soaWs = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                gpu.MatMulBatchedWeightStationary(aosWs, aosW, gx, nTok, DType.Q6_K);
                gpu.MatMulBatchedWeightStationary(soaWs, soaW, gx, nTok, DType.Q6_K);
                gpu.Synchronize();
                AssertBitIdentical(gpu, aosWs, soaWs, nTok * rows, $"MatMulBatchedWS n={nTok} rows={rows} cols={cols}");

                // Prefill dequant→fp16→cuBLAS GEMM path (llm_dequant_q6k_to_f16{,_soa}): the SoA
                // dequant emits the SAME fp16 weight bytes as the AoS dequant, so the GEMM output
                // is bit-identical too (the hot prefill path — covers the SoA dequant kernel).
                var aosG = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                var soaG = gpu.Allocate(TensorShape.D1((long)nTok * rows));
                gpu.MatMulBatchedGemm(aosG, aosW, gx, nTok, DType.Q6_K);
                gpu.MatMulBatchedGemm(soaG, soaW, gx, nTok, DType.Q6_K);
                gpu.Synchronize();
                AssertBitIdentical(gpu, aosG, soaG, nTok * rows, $"MatMulBatchedGemm n={nTok} rows={rows} cols={cols}");

                gpu.Free(gx); gpu.Free(aosY); gpu.Free(soaY); gpu.Free(aosWs); gpu.Free(soaWs);
                gpu.Free(aosG); gpu.Free(soaG);
            }

            gpu.Free(aosW);
            gpu.Free(soaW);
        }
    }

    private static void AssertBitIdentical(CudaBackend gpu, Tensor a, Tensor b, int n, string ctx)
    {
        var ha = new float[n];
        var hb = new float[n];
        gpu.Download(a, ha);
        gpu.Download(b, hb);
        for (int i = 0; i < n; i++)
            Assert.True(BitConverter.SingleToInt32Bits(ha[i]) == BitConverter.SingleToInt32Bits(hb[i]),
                $"{ctx}: SoA Q6_K reader not bit-identical to AoS at [{i}]: AoS={ha[i]:R} SoA={hb[i]:R}.");
    }
}
