using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #111 bit-exactness tests for <see cref="CudaBackend.MatMulBatched"/>
/// (GEMM-N). The batched path must produce results <b>bit-identical</b> to N
/// sequential <see cref="CudaBackend.MatMul"/> calls with the same weight matrix
/// — not just within tolerance. The GEMM-N kernel runs the identical per-row
/// reduction as the GEMV, only shifting the input/output base pointer per token,
/// so any divergence means the reduction order was reordered (the failure mode
/// the GDN/MTP byte-parity oracles trip on — see the K/V MatVecDual note).
///
/// Silently skips on hosts without CUDA, mirroring the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMatMulBatchedTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// Q4_K layout: 144 bytes per 256-element super-block (matches the GGUF path).
    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                float d    = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.03 + 0.005);
                ushort dh = HalfToUshort((Half)d), dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF); bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF); bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12;  i++) bytes[off +   4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off +  16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    private static void AssertBitIdentical(string label, int rows, int nTok,
                                           float[] batched, float[][] reference)
    {
        for (int t = 0; t < nTok; t++)
            for (int r = 0; r < rows; r++)
            {
                float bat = batched[(long)t * rows + r];
                float refv = reference[t][r];
                if (BitConverter.SingleToInt32Bits(bat) != BitConverter.SingleToInt32Bits(refv))
                    Assert.Fail(
                        $"{label}: token {t} row {r} GEMM-N={bat} (0x{BitConverter.SingleToInt32Bits(bat):X8}) " +
                        $"!= sequential GEMV={refv} (0x{BitConverter.SingleToInt32Bits(refv):X8}). " +
                        "MatMulBatched must be bit-identical to N sequential MatMul calls.");
            }
    }

    [Fact]
    public void MatMulBatched_Q4K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in
                 new[] { (8, 256, 2), (33, 512, 5), (64, 1024, 17), (256, 2048, 64) })
        {
            var rng = new Random(20260603 + rows * 31 + cols * 7 + nTok);
            byte[] weights = BuildQ4KMatrix(rows, cols, rng);

            var inAll = new float[(long)nTok * cols];
            for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q4_K);
            var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatched(gpuOutAll, gpuW, gpuInAll, nTok, DType.Q4_K);
            gpu.Synchronize();
            var batched = new float[(long)nTok * rows];
            gpu.Download(gpuOutAll, batched);

            // Sequential reference: one GEMV per token over the same weight.
            var reference = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var inT = new float[cols];
                Array.Copy(inAll, (long)t * cols, inT, 0, cols);
                var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
                var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(gpuRefT, gpuW, gpuInT, DType.Q4_K);
                gpu.Synchronize();
                reference[t] = new float[rows];
                gpu.Download(gpuRefT, reference[t]);
                gpu.Free(gpuInT); gpu.Free(gpuRefT);
            }

            gpu.Free(gpuW); gpu.Free(gpuInAll); gpu.Free(gpuOutAll);

            AssertBitIdentical($"Q4_K rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
        }
    }

    /// Q6_K layout: 210 bytes per 256-element super-block.
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 210;
                for (int i = 0; i < 128; i++) bytes[off + i] = (byte)rng.Next(256);
                for (int i = 0; i < 64;  i++) bytes[off + 128 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 16;  i++) bytes[off + 192 + i] = (byte)(rng.Next(33) - 16);
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                ushort dh = HalfToUshort((Half)d);
                bytes[off + 208] = (byte)(dh & 0xFF);
                bytes[off + 209] = (byte)(dh >> 8);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatched_Q6K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in
                 new[] { (8, 256, 2), (33, 512, 5), (64, 1024, 17), (256, 512, 64) })
        {
            var rng = new Random(20260603 + rows * 37 + cols * 11 + nTok);
            byte[] weights = BuildQ6KMatrix(rows, cols, rng);

            var inAll = new float[(long)nTok * cols];
            for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q6_K);
            var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatched(gpuOutAll, gpuW, gpuInAll, nTok, DType.Q6_K);
            gpu.Synchronize();
            var batched = new float[(long)nTok * rows];
            gpu.Download(gpuOutAll, batched);

            var reference = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var inT = new float[cols];
                Array.Copy(inAll, (long)t * cols, inT, 0, cols);
                var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
                var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(gpuRefT, gpuW, gpuInT, DType.Q6_K);
                gpu.Synchronize();
                reference[t] = new float[rows];
                gpu.Download(gpuRefT, reference[t]);
                gpu.Free(gpuInT); gpu.Free(gpuRefT);
            }
            gpu.Free(gpuW); gpu.Free(gpuInAll); gpu.Free(gpuOutAll);
            AssertBitIdentical($"Q6_K rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
        }
    }

    /// Q5_K layout: 176 bytes per 256-element super-block
    /// ([d:fp16][dmin:fp16][scales:12][qh:32][ql:128]).
    private static byte[] BuildQ5KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 176;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 176;
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.02);
                ushort dh = HalfToUshort((Half)d);
                ushort dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF);
                bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF);
                bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 4; i < 176; i++) bytes[off + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatched_Q5K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in
                 new[] { (8, 256, 2), (33, 512, 5), (64, 1024, 17), (256, 512, 64) })
        {
            var rng = new Random(20260603 + rows * 41 + cols * 13 + nTok);
            byte[] weights = BuildQ5KMatrix(rows, cols, rng);

            var inAll = new float[(long)nTok * cols];
            for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q5_K);
            var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatched(gpuOutAll, gpuW, gpuInAll, nTok, DType.Q5_K);
            gpu.Synchronize();
            var batched = new float[(long)nTok * rows];
            gpu.Download(gpuOutAll, batched);

            var reference = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var inT = new float[cols];
                Array.Copy(inAll, (long)t * cols, inT, 0, cols);
                var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
                var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(gpuRefT, gpuW, gpuInT, DType.Q5_K);
                gpu.Synchronize();
                reference[t] = new float[rows];
                gpu.Download(gpuRefT, reference[t]);
                gpu.Free(gpuInT); gpu.Free(gpuRefT);
            }
            gpu.Free(gpuW); gpu.Free(gpuInAll); gpu.Free(gpuOutAll);
            AssertBitIdentical($"Q5_K rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
        }
    }

    /// Q8_0 layout: 34 bytes per 32-element block ([d:fp16][32×int8]).
    private static byte[] BuildQ80Matrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 34;
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                ushort dh = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dh & 0xFF);
                bytes[off + 1] = (byte)(dh >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatched_Q8_0_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in
                 new[] { (8, 256, 2), (33, 512, 5), (64, 1024, 17), (130, 2048, 40) })
        {
            var rng = new Random(20260604 + rows * 17 + cols * 5 + nTok);
            byte[] weights = BuildQ80Matrix(rows, cols, rng);

            var inAll = new float[(long)nTok * cols];
            for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q8_0);
            var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatched(gpuOutAll, gpuW, gpuInAll, nTok, DType.Q8_0);
            gpu.Synchronize();
            var batched = new float[(long)nTok * rows];
            gpu.Download(gpuOutAll, batched);

            var reference = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var inT = new float[cols];
                Array.Copy(inAll, (long)t * cols, inT, 0, cols);
                var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
                var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(gpuRefT, gpuW, gpuInT, DType.Q8_0);
                gpu.Synchronize();
                reference[t] = new float[rows];
                gpu.Download(gpuRefT, reference[t]);
                gpu.Free(gpuInT); gpu.Free(gpuRefT);
            }
            gpu.Free(gpuW); gpu.Free(gpuInAll); gpu.Free(gpuOutAll);
            AssertBitIdentical($"Q8_0 rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
        }
    }

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static void AssertRowsBitIdentical(string label, int rows, int nTok,
                                               float[] batched, float[][] reference)
    {
        for (int t = 0; t < nTok; t++)
            for (int r = 0; r < rows; r++)
            {
                float bat = batched[(long)t * rows + r];
                float refv = reference[t][r];
                if (BitConverter.SingleToInt32Bits(bat) != BitConverter.SingleToInt32Bits(refv))
                    Assert.Fail($"{label}: token {t} idx {r} batched={bat} != single={refv}.");
            }
    }

    [Fact]
    public void RmsNormBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        foreach ((int dim, int nTok) in new[] { (256, 3), (2048, 17), (2048, 64) })
        {
            var rng = new Random(7 + dim + nTok);
            var w = Rand(dim, rng);
            var xAll = new float[(long)nTok * dim];
            for (int i = 0; i < xAll.Length; i++) xAll[i] = (float)(rng.NextDouble() * 4 - 2);

            var gpuW = gpu.Upload(w, TensorShape.D1(dim));
            var gpuX = gpu.Upload(xAll, TensorShape.D1((long)nTok * dim));
            var gpuOut = gpu.Allocate(TensorShape.D1((long)nTok * dim));
            gpu.RmsNormBatched(gpuOut, gpuX, gpuW, nTok, dim, 1e-6f);
            gpu.Synchronize();
            var bat = new float[(long)nTok * dim]; gpu.Download(gpuOut, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var xt = new float[dim]; Array.Copy(xAll, (long)t * dim, xt, 0, dim);
                var gx = gpu.Upload(xt, TensorShape.D1(dim));
                var go = gpu.Allocate(TensorShape.D1(dim));
                gpu.RmsNorm(go, gx, gpuW, 1e-6f);
                gpu.Synchronize();
                refs[t] = new float[dim]; gpu.Download(go, refs[t]);
                gpu.Free(gx); gpu.Free(go);
            }
            gpu.Free(gpuW); gpu.Free(gpuX); gpu.Free(gpuOut);
            AssertRowsBitIdentical($"RmsNorm dim={dim} nTok={nTok}", dim, nTok, bat, refs);
        }
    }

    [Fact]
    public void HeadNormBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 16, headDim = 256;
        foreach (int nTok in new[] { 3, 17, 64 })
        {
            var rng = new Random(11 + nTok);
            var w = Rand(headDim, rng);
            int qDim = numHeads * headDim;
            var dataAll = new float[(long)nTok * qDim];
            for (int i = 0; i < dataAll.Length; i++) dataAll[i] = (float)(rng.NextDouble() * 4 - 2);

            var gpuW = gpu.Upload(w, TensorShape.D1(headDim));
            var gpuData = gpu.Upload(dataAll, TensorShape.D1((long)nTok * qDim));
            gpu.HeadNormBatched(gpuData, gpuW, numHeads, headDim, nTok, 1e-6f, false);
            gpu.Synchronize();
            var bat = new float[(long)nTok * qDim]; gpu.Download(gpuData, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var dt = new float[qDim]; Array.Copy(dataAll, (long)t * qDim, dt, 0, qDim);
                var gd = gpu.Upload(dt, TensorShape.D1(qDim));
                gpu.HeadNorm(gd, gpuW, numHeads, headDim, 1e-6f, false);
                gpu.Synchronize();
                refs[t] = new float[qDim]; gpu.Download(gd, refs[t]);
                gpu.Free(gd);
            }
            gpu.Free(gpuW); gpu.Free(gpuData);
            AssertRowsBitIdentical($"HeadNorm nTok={nTok}", qDim, nTok, bat, refs);
        }
    }

    [Fact]
    public void HeadNormBatched_PerChannelWeight_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 16, headDim = 256;
        int qDim = numHeads * headDim;
        foreach (int nTok in new[] { 3, 17, 64 })
        {
            var rng = new Random(101 + nTok);
            // Per-channel weight is [numHeads * headDim] (OLMoE-style QK norm).
            var w = Rand(qDim, rng);
            var dataAll = new float[(long)nTok * qDim];
            for (int i = 0; i < dataAll.Length; i++) dataAll[i] = (float)(rng.NextDouble() * 4 - 2);

            var gpuW = gpu.Upload(w, TensorShape.D1(qDim));
            var gpuData = gpu.Upload(dataAll, TensorShape.D1((long)nTok * qDim));
            gpu.HeadNormBatched(gpuData, gpuW, numHeads, headDim, nTok, 1e-6f, perChannelWeight: true);
            gpu.Synchronize();
            var bat = new float[(long)nTok * qDim]; gpu.Download(gpuData, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var dt = new float[qDim]; Array.Copy(dataAll, (long)t * qDim, dt, 0, qDim);
                var gd = gpu.Upload(dt, TensorShape.D1(qDim));
                gpu.HeadNorm(gd, gpuW, numHeads, headDim, 1e-6f, perChannelWeight: true);
                gpu.Synchronize();
                refs[t] = new float[qDim]; gpu.Download(gd, refs[t]);
                gpu.Free(gd);
            }
            gpu.Free(gpuW); gpu.Free(gpuData);
            AssertRowsBitIdentical($"HeadNorm(perChannel) nTok={nTok}", qDim, nTok, bat, refs);
        }
    }

    [Fact]
    public void View_RoundTrips_FreeDoesNotFreeParent_AndBoundsChecked()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int n = 4096, off = 512, len = 1024;
        var data = new float[n];
        var rng = new Random(7);
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        var parent = gpu.Upload(data, TensorShape.D1(n));

        // (a) View reads back the matching parent sub-range.
        var view = gpu.View(parent, off, len);
        var got = new float[len];
        gpu.Download(view, got);
        for (int i = 0; i < len; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(data[off + i]), BitConverter.SingleToInt32Bits(got[i]));

        // (b) Freeing the view must NOT free the parent — parent still reads back.
        gpu.Free(view);
        var parentAfter = new float[n];
        gpu.Download(parent, parentAfter);
        for (int i = 0; i < n; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(data[i]), BitConverter.SingleToInt32Bits(parentAfter[i]));

        // (c) Bounds + negative-arg checks throw.
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.View(parent, n - 10, 100));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.View(parent, -1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.View(parent, 0, -10));

        gpu.Free(parent);
    }

    [Fact]
    public void SplitQGBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 16, headDim = 256;
        int qDim = numHeads * headDim;
        foreach (int nTok in new[] { 2, 17, 64 })
        {
            var rng = new Random(13 + nTok);
            var qgAll = Rand(nTok * qDim * 2, rng);
            var gpuQg = gpu.Upload(qgAll, TensorShape.D1((long)nTok * qDim * 2));
            var gpuQ = gpu.Allocate(TensorShape.D1((long)nTok * qDim));
            var gpuG = gpu.Allocate(TensorShape.D1((long)nTok * qDim));
            gpu.SplitQGBatched(gpuQ, gpuG, gpuQg, numHeads, headDim, nTok);
            gpu.Synchronize();
            var batQ = new float[(long)nTok * qDim]; gpu.Download(gpuQ, batQ);
            var batG = new float[(long)nTok * qDim]; gpu.Download(gpuG, batG);

            var refsQ = new float[nTok][];
            var refsG = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var qgt = new float[qDim * 2]; Array.Copy(qgAll, (long)t * qDim * 2, qgt, 0, qDim * 2);
                var gqg = gpu.Upload(qgt, TensorShape.D1(qDim * 2));
                var gq = gpu.Allocate(TensorShape.D1(qDim));
                var gg = gpu.Allocate(TensorShape.D1(qDim));
                gpu.SplitQG(gq, gg, gqg, numHeads, headDim);
                gpu.Synchronize();
                refsQ[t] = new float[qDim]; gpu.Download(gq, refsQ[t]);
                refsG[t] = new float[qDim]; gpu.Download(gg, refsG[t]);
                gpu.Free(gqg); gpu.Free(gq); gpu.Free(gg);
            }
            gpu.Free(gpuQg); gpu.Free(gpuQ); gpu.Free(gpuG);
            AssertRowsBitIdentical($"SplitQG-Q nTok={nTok}", qDim, nTok, batQ, refsQ);
            AssertRowsBitIdentical($"SplitQG-G nTok={nTok}", qDim, nTok, batG, refsG);
        }
    }

    [Fact]
    public void RoPEPartialBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 16, headDim = 256, ropeDim = 64;
        int qDim = numHeads * headDim;
        float theta = 1000000f;
        foreach ((int basePos, int nTok) in new[] { (0, 3), (37, 17), (512, 64) })
        {
            var rng = new Random(17 + basePos + nTok);
            var xAll = Rand(nTok * qDim, rng);
            var gpuX = gpu.Upload(xAll, TensorShape.D1((long)nTok * qDim));
            gpu.RoPEPartialBatched(gpuX, basePos, headDim, ropeDim, theta, numHeads, nTok, neox: true);
            gpu.Synchronize();
            var bat = new float[(long)nTok * qDim]; gpu.Download(gpuX, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var xt = new float[qDim]; Array.Copy(xAll, (long)t * qDim, xt, 0, qDim);
                var gx = gpu.Upload(xt, TensorShape.D1(qDim));
                gpu.RoPEPartial(gx, basePos + t, headDim, ropeDim, theta, neox: true);
                gpu.Synchronize();
                refs[t] = new float[qDim]; gpu.Download(gx, refs[t]);
                gpu.Free(gx);
            }
            gpu.Free(gpuX);
            AssertRowsBitIdentical($"RoPE basePos={basePos} nTok={nTok}", qDim, nTok, bat, refs);
        }
    }

    [Fact]
    public void HeadNormQk_BitwiseMatchesSeparate()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 16, numKvHeads = 4, headDim = 256;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        var rng = new Random(909);
        var qw = Rand(headDim, rng);
        var kw = Rand(headDim, rng);
        var qd = Rand(qDim, rng);
        var kd = Rand(kvDim, rng);
        var gqw = gpu.Upload(qw, TensorShape.D1(headDim));
        var gkw = gpu.Upload(kw, TensorShape.D1(headDim));

        // Dual single-token.
        var gqd = gpu.Upload(qd, TensorShape.D1(qDim));
        var gkd = gpu.Upload(kd, TensorShape.D1(kvDim));
        gpu.HeadNormQk(gqd, gqw, gkd, gkw, numHeads, numKvHeads, headDim, 1e-6f, false);
        gpu.Synchronize();
        var dualQ = new float[qDim]; gpu.Download(gqd, dualQ);
        var dualK = new float[kvDim]; gpu.Download(gkd, dualK);

        // Reference: two separate HeadNorm calls.
        var sqd = gpu.Upload(qd, TensorShape.D1(qDim));
        var skd = gpu.Upload(kd, TensorShape.D1(kvDim));
        gpu.HeadNorm(sqd, gqw, numHeads, headDim, 1e-6f, false);
        gpu.HeadNorm(skd, gkw, numKvHeads, headDim, 1e-6f, false);
        gpu.Synchronize();
        var refQ = new float[qDim]; gpu.Download(sqd, refQ);
        var refK = new float[kvDim]; gpu.Download(skd, refK);
        for (int i = 0; i < qDim; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(refQ[i]), BitConverter.SingleToInt32Bits(dualQ[i]));
        for (int i = 0; i < kvDim; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(refK[i]), BitConverter.SingleToInt32Bits(dualK[i]));
        gpu.Free(gqd); gpu.Free(gkd); gpu.Free(sqd); gpu.Free(skd);

        // Batched dual vs separate batched.
        foreach (int nTok in new[] { 3, 17, 40 })
        {
            var rng2 = new Random(31 + nTok);
            var qAll = Rand(nTok * qDim, rng2);
            var kAllArr = Rand(nTok * kvDim, rng2);
            var gq = gpu.Upload(qAll, TensorShape.D1((long)nTok * qDim));
            var gk = gpu.Upload(kAllArr, TensorShape.D1((long)nTok * kvDim));
            gpu.HeadNormQkBatched(gq, gqw, gk, gkw, numHeads, numKvHeads, headDim, nTok, 1e-6f, false);
            gpu.Synchronize();
            var bq = new float[(long)nTok * qDim]; gpu.Download(gq, bq);
            var bk = new float[(long)nTok * kvDim]; gpu.Download(gk, bk);

            var rq = gpu.Upload(qAll, TensorShape.D1((long)nTok * qDim));
            var rk = gpu.Upload(kAllArr, TensorShape.D1((long)nTok * kvDim));
            gpu.HeadNormBatched(rq, gqw, numHeads, headDim, nTok, 1e-6f, false);
            gpu.HeadNormBatched(rk, gkw, numKvHeads, headDim, nTok, 1e-6f, false);
            gpu.Synchronize();
            var refbq = new float[(long)nTok * qDim]; gpu.Download(rq, refbq);
            var refbk = new float[(long)nTok * kvDim]; gpu.Download(rk, refbk);
            for (int i = 0; i < bq.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(refbq[i]), BitConverter.SingleToInt32Bits(bq[i]));
            for (int i = 0; i < bk.Length; i++)
                Assert.Equal(BitConverter.SingleToInt32Bits(refbk[i]), BitConverter.SingleToInt32Bits(bk[i]));
            gpu.Free(gq); gpu.Free(gk); gpu.Free(rq); gpu.Free(rk);
        }
        gpu.Free(gqw); gpu.Free(gkw);
    }

    [Fact]
    public void RoPEWithFactorsBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 8, headDim = 256;       // Gemma 4 global-layer shape
        int qDim = numHeads * headDim;
        float theta = 1000000f;
        // Per-half-dim frequency factors (strictly positive — they divide the freq).
        var rngF = new Random(4242);
        var factors = new float[headDim / 2];
        for (int i = 0; i < factors.Length; i++) factors[i] = (float)(rngF.NextDouble() * 3 + 0.25);
        var gpuFactors = gpu.Upload(factors, TensorShape.D1(headDim / 2));

        foreach ((int basePos, int nTok) in new[] { (0, 3), (37, 17), (512, 64) })
        {
            var rng = new Random(91 + basePos + nTok);
            var xAll = Rand(nTok * qDim, rng);
            var gpuX = gpu.Upload(xAll, TensorShape.D1((long)nTok * qDim));
            gpu.RoPEWithFactorsBatched(gpuX, basePos, headDim, theta, gpuFactors, numHeads, nTok);
            gpu.Synchronize();
            var bat = new float[(long)nTok * qDim]; gpu.Download(gpuX, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var xt = new float[qDim]; Array.Copy(xAll, (long)t * qDim, xt, 0, qDim);
                var gx = gpu.Upload(xt, TensorShape.D1(qDim));
                gpu.RoPEWithFactors(gx, basePos + t, headDim, theta, gpuFactors);
                gpu.Synchronize();
                refs[t] = new float[qDim]; gpu.Download(gx, refs[t]);
                gpu.Free(gx);
            }
            gpu.Free(gpuX);
            AssertRowsBitIdentical($"RoPEFactors basePos={basePos} nTok={nTok}", qDim, nTok, bat, refs);
        }
        gpu.Free(gpuFactors);
    }

    [Fact]
    public void AttentionSwaBatched_BitwiseMatchesSingle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        int numHeads = 8, numKvHeads = 2, headDim = 256, maxSeqLen = 1024;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;

        foreach ((int startPos, int nTok, int window) in
                 new[] { (0, 3, 8), (5, 17, 8), (37, 64, 512), (300, 40, 512) })
        {
            var rng = new Random(53 + startPos * 7 + nTok + window);
            // KV cache: fill the positions that will be read; rest is irrelevant.
            var kc = new float[(long)maxSeqLen * kvDim];
            var vc = new float[(long)maxSeqLen * kvDim];
            int filled = (startPos + nTok) * kvDim;
            for (int i = 0; i < filled; i++) { kc[i] = (float)(rng.NextDouble() * 2 - 1); vc[i] = (float)(rng.NextDouble() * 2 - 1); }
            var xAll = Rand(nTok * qDim, rng);

            var gpuK = gpu.Upload(kc, TensorShape.D1((long)maxSeqLen * kvDim));
            var gpuV = gpu.Upload(vc, TensorShape.D1((long)maxSeqLen * kvDim));
            var gpuQ = gpu.Upload(xAll, TensorShape.D1((long)nTok * qDim));
            var gpuOut = gpu.Allocate(TensorShape.D1((long)nTok * qDim));
            gpu.AttentionSwaBatched(gpuQ, gpuK, gpuV, gpuOut, numHeads, numKvHeads, headDim,
                startPos, window, maxSeqLen, nTok);
            gpu.Synchronize();
            var bat = new float[(long)nTok * qDim]; gpu.Download(gpuOut, bat);

            var refs = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var xt = new float[qDim]; Array.Copy(xAll, (long)t * qDim, xt, 0, qDim);
                var gx = gpu.Upload(xt, TensorShape.D1(qDim));
                var go = gpu.Allocate(TensorShape.D1(qDim));
                gpu.AttentionSwa(gx, gpuK, gpuV, go, null, startPos + t, window, headDim,
                    numHeads, numKvHeads, maxSeqLen);
                gpu.Synchronize();
                refs[t] = new float[qDim]; gpu.Download(go, refs[t]);
                gpu.Free(gx); gpu.Free(go);
            }
            gpu.Free(gpuK); gpu.Free(gpuV); gpu.Free(gpuQ); gpu.Free(gpuOut);
            AssertRowsBitIdentical($"SWA startPos={startPos} nTok={nTok} window={window}", qDim, nTok, bat, refs);
        }
    }

    [Fact]
    public void MatMulBatched_F32_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols, int nTok) in
                 new[] { (8, 256, 2), (33, 500, 5), (64, 1024, 17), (130, 2048, 40) })
        {
            var rng = new Random(20260603 + rows * 13 + cols * 3 + nTok);
            var weights = new float[(long)rows * cols];
            for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);

            var inAll = new float[(long)nTok * cols];
            for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuW = gpu.Upload(weights, TensorShape.D1((long)rows * cols));
            var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatched(gpuOutAll, gpuW, gpuInAll, nTok, DType.Float32);
            gpu.Synchronize();
            var batched = new float[(long)nTok * rows];
            gpu.Download(gpuOutAll, batched);

            var reference = new float[nTok][];
            for (int t = 0; t < nTok; t++)
            {
                var inT = new float[cols];
                Array.Copy(inAll, (long)t * cols, inT, 0, cols);
                var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
                var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
                gpu.MatMul(gpuRefT, gpuW, gpuInT, DType.Float32);
                gpu.Synchronize();
                reference[t] = new float[rows];
                gpu.Download(gpuRefT, reference[t]);
                gpu.Free(gpuInT); gpu.Free(gpuRefT);
            }

            gpu.Free(gpuW); gpu.Free(gpuInAll); gpu.Free(gpuOutAll);

            AssertBitIdentical($"F32 rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
        }
    }
}
