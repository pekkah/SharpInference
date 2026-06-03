using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #114-B per-kernel bit-exactness tests for the batched GDN trunk +
/// batched-query SDPA kernels. Each batched kernel must produce results
/// <b>bit-identical</b> to the N sequential single-token kernels it replaces —
/// not just within tolerance. A divergence means a reduction was reordered (the
/// failure mode the GDN/MTP byte-parity oracles trip on — see the K/V MatVecDual
/// note). Mirrors <see cref="CudaMatMulBatchedTests"/>; silently skips without CUDA.
/// </summary>
public sealed unsafe class CudaGdnBatchedTrunkTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static float[] Rand(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    private static void AssertBitId(string label, float[] batched, float[] reference)
    {
        Assert.Equal(reference.Length, batched.Length);
        for (int i = 0; i < reference.Length; i++)
            if (Bits(batched[i]) != Bits(reference[i]))
                Assert.Fail($"{label}: index {i} batched={batched[i]} (0x{Bits(batched[i]):X8}) " +
                            $"!= sequential={reference[i]} (0x{Bits(reference[i]):X8}).");
    }

    // ── GDN conv1d (decode + state advance) ────────────────────────────────
    [Fact]
    public void GdnConv1dDecodeBatched_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int channels = 8192, kernelSize = 4, retained = kernelSize - 1;
        // N=1 hits the maximal state-aliasing edge the state-update kernel guards against
        // (all retained taps come from carried state); N<K-1 mixes carried state + chunk.
        foreach (int N in new[] { 1, 2, 5, 33 })
        {
            var rng = new Random(114 + N);
            var xAll = Rand(N * channels, rng);
            var state0 = Rand(retained * channels, rng);
            var weight = Rand(kernelSize * channels, rng);

            var gpuW = gpu.Upload(weight, TensorShape.D1(kernelSize * channels));

            // Sequential reference: shared state mutated per token.
            var gpuStateRef = gpu.Upload(state0, TensorShape.D1(retained * channels));
            var refOut = new float[(long)N * channels];
            for (int i = 0; i < N; i++)
            {
                var xt = new float[channels]; Array.Copy(xAll, (long)i * channels, xt, 0, channels);
                var gx = gpu.Upload(xt, TensorShape.D1(channels));
                var go = gpu.Allocate(TensorShape.D1(channels));
                gpu.GdnConv1dDecode(gx, gpuStateRef, gpuW, go, channels, kernelSize);
                gpu.Synchronize();
                var ot = new float[channels]; gpu.Download(go, ot);
                Array.Copy(ot, 0, refOut, (long)i * channels, channels);
                gpu.Free(gx); gpu.Free(go);
            }
            var refState = new float[retained * channels]; gpu.Download(gpuStateRef, refState);

            // Batched: read-only old state, then advance.
            var gpuXAll = gpu.Upload(xAll, TensorShape.D1((long)N * channels));
            var gpuStateBat = gpu.Upload(state0, TensorShape.D1(retained * channels));
            var gpuOutAll = gpu.Allocate(TensorShape.D1((long)N * channels));
            gpu.GdnConv1dDecodeBatched(gpuXAll, gpuStateBat, gpuW, gpuOutAll, channels, kernelSize, N);
            gpu.GdnConv1dStateUpdateBatched(gpuXAll, gpuStateBat, channels, kernelSize, N);
            gpu.Synchronize();
            var batOut = new float[(long)N * channels]; gpu.Download(gpuOutAll, batOut);
            var batState = new float[retained * channels]; gpu.Download(gpuStateBat, batState);

            gpu.Free(gpuW); gpu.Free(gpuStateRef); gpu.Free(gpuXAll); gpu.Free(gpuStateBat); gpu.Free(gpuOutAll);

            AssertBitId($"conv1d out N={N}", batOut, refOut);
            AssertBitId($"conv1d state N={N}", batState, refState);
        }
    }

    // ── GDN L2-norm per head ────────────────────────────────────────────────
    [Fact]
    public void GdnL2NormPerHeadBatched_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int headDim = 128, numHeads = 16;
        int rowDim = numHeads * headDim;
        foreach (int N in new[] { 3, 17, 64 })
        {
            var rng = new Random(200 + N);
            var dataAll = Rand(N * rowDim, rng);

            var refOut = new float[(long)N * rowDim];
            for (int i = 0; i < N; i++)
            {
                var dt = new float[rowDim]; Array.Copy(dataAll, (long)i * rowDim, dt, 0, rowDim);
                var gd = gpu.Upload(dt, TensorShape.D1(rowDim));
                gpu.GdnL2NormPerHead(gd, 0, numHeads, headDim, eps: 1e-6f);
                gpu.Synchronize();
                var ot = new float[rowDim]; gpu.Download(gd, ot);
                Array.Copy(ot, 0, refOut, (long)i * rowDim, rowDim);
                gpu.Free(gd);
            }

            var gpuData = gpu.Upload(dataAll, TensorShape.D1((long)N * rowDim));
            gpu.GdnL2NormPerHeadBatched(gpuData, 0, numHeads, headDim, rowDim, N, eps: 1e-6f);
            gpu.Synchronize();
            var batOut = new float[(long)N * rowDim]; gpu.Download(gpuData, batOut);
            gpu.Free(gpuData);

            AssertBitId($"l2norm N={N}", batOut, refOut);
        }
    }

    // ── GDN tile heads (GQA broadcast) ──────────────────────────────────────
    [Fact]
    public void GdnTileHeadsBatched_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int srcHeads = 16, repeat = 2, headDim = 128;
        int srcDim = srcHeads * headDim;
        int dstDim = srcHeads * repeat * headDim;
        foreach (int N in new[] { 2, 17, 40 })
        {
            var rng = new Random(300 + N);
            var srcAll = Rand(N * srcDim, rng);

            var refOut = new float[(long)N * dstDim];
            for (int i = 0; i < N; i++)
            {
                var st = new float[srcDim]; Array.Copy(srcAll, (long)i * srcDim, st, 0, srcDim);
                var gs = gpu.Upload(st, TensorShape.D1(srcDim));
                var gd = gpu.Allocate(TensorShape.D1(dstDim));
                gpu.GdnTileHeads(gs, 0, gd, 0, srcHeads, repeat, headDim);
                gpu.Synchronize();
                var ot = new float[dstDim]; gpu.Download(gd, ot);
                Array.Copy(ot, 0, refOut, (long)i * dstDim, dstDim);
                gpu.Free(gs); gpu.Free(gd);
            }

            var gpuSrc = gpu.Upload(srcAll, TensorShape.D1((long)N * srcDim));
            var gpuDst = gpu.Allocate(TensorShape.D1((long)N * dstDim));
            gpu.GdnTileHeadsBatched(gpuSrc, 0, gpuDst, 0, srcHeads, repeat, headDim, srcDim, dstDim, N);
            gpu.Synchronize();
            var batOut = new float[(long)N * dstDim]; gpu.Download(gpuDst, batOut);
            gpu.Free(gpuSrc); gpu.Free(gpuDst);

            AssertBitId($"tile N={N}", batOut, refOut);
        }
    }

    // ── GDN fused recurrence scan ───────────────────────────────────────────
    [Fact]
    public void GdnRecurrenceScan_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int hv = 4, d = 128;
        int hd = hv * d;
        foreach (int N in new[] { 1, 2, 7, 33 })
        {
            var rng = new Random(400 + N);
            var state0 = Rand(hv * d * d, rng);
            var qAll = Rand(N * hd, rng);
            var kAll = Rand(N * hd, rng);
            var vAll = Rand(N * hd, rng);
            var zAll = Rand(N * hd, rng);
            var alphaAll = Rand(N * hv, rng);
            var betaAll = Rand(N * hv, rng);
            var ssmA = Rand(hv, rng);
            var dtBias = Rand(hv, rng);
            var normW = Rand(d, rng);

            var gpuSsmA = gpu.Upload(ssmA, TensorShape.D1(hv));
            var gpuDt = gpu.Upload(dtBias, TensorShape.D1(hv));
            var gpuNw = gpu.Upload(normW, TensorShape.D1(d));

            // Sequential reference: state mutated per token.
            var gpuStateRef = gpu.Upload(state0, TensorShape.D1(hv * d * d));
            var refOut = new float[(long)N * hd];
            for (int i = 0; i < N; i++)
            {
                float[] Slice(float[] a, int stride) { var s = new float[stride]; Array.Copy(a, (long)i * stride, s, 0, stride); return s; }
                var gq = gpu.Upload(Slice(qAll, hd), TensorShape.D1(hd));
                var gk = gpu.Upload(Slice(kAll, hd), TensorShape.D1(hd));
                var gv = gpu.Upload(Slice(vAll, hd), TensorShape.D1(hd));
                var gz = gpu.Upload(Slice(zAll, hd), TensorShape.D1(hd));
                var ga = gpu.Upload(Slice(alphaAll, hv), TensorShape.D1(hv));
                var gb = gpu.Upload(Slice(betaAll, hv), TensorShape.D1(hv));
                var go = gpu.Allocate(TensorShape.D1(hd));
                gpu.GdnRecurrenceDecode(gpuStateRef, gq, gk, gv, ga, gb, gpuSsmA, gpuDt, gpuNw, gz, go, hv, d, 1e-6f);
                gpu.Synchronize();
                var ot = new float[hd]; gpu.Download(go, ot);
                Array.Copy(ot, 0, refOut, (long)i * hd, hd);
                gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(gz); gpu.Free(ga); gpu.Free(gb); gpu.Free(go);
            }
            var refState = new float[hv * d * d]; gpu.Download(gpuStateRef, refState);

            // Batched fused scan: q/k/v/z laid out [N × hd] (stride hd, vHeadOff 0).
            var gpuStateBat = gpu.Upload(state0, TensorShape.D1(hv * d * d));
            var gQ = gpu.Upload(qAll, TensorShape.D1((long)N * hd));
            var gK = gpu.Upload(kAll, TensorShape.D1((long)N * hd));
            var gV = gpu.Upload(vAll, TensorShape.D1((long)N * hd));
            var gZ = gpu.Upload(zAll, TensorShape.D1((long)N * hd));
            var gA = gpu.Upload(alphaAll, TensorShape.D1((long)N * hv));
            var gB = gpu.Upload(betaAll, TensorShape.D1((long)N * hv));
            var gOut = gpu.Allocate(TensorShape.D1((long)N * hd));
            gpu.GdnRecurrenceScan(gpuStateBat, gQ, gK, gV, gA, gB, gpuSsmA, gpuDt, gpuNw, gZ, gOut,
                hv, d, 1e-6f, qStride: hd, kStride: hd, vStride: hd, vHeadOff: 0, zStride: hd, oStride: hd, nTok: N);
            gpu.Synchronize();
            var batOut = new float[(long)N * hd]; gpu.Download(gOut, batOut);
            var batState = new float[hv * d * d]; gpu.Download(gpuStateBat, batState);

            gpu.Free(gpuSsmA); gpu.Free(gpuDt); gpu.Free(gpuNw);
            gpu.Free(gpuStateRef); gpu.Free(gpuStateBat);
            gpu.Free(gQ); gpu.Free(gK); gpu.Free(gV); gpu.Free(gZ); gpu.Free(gA); gpu.Free(gB); gpu.Free(gOut);

            AssertBitId($"scan out N={N}", batOut, refOut);
            AssertBitId($"scan state N={N}", batState, refState);
        }
    }

    // ── Batched KV-append + SDPA (fp32) ─────────────────────────────────────
    [Fact]
    public void AttentionBatched_F32_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int numHeads = 8, numKvHeads = 2, headDim = 128;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        const int maxSeq = 4096;
        // (4090,6): last query seq_len == 4096, the inclusive edge of the shared-scores
        // buffer and the AttentionBatched startPos+N<=4096 guard boundary.
        foreach ((int startPos, int N) in new[] { (0, 3), (0, 64), (100, 17), (3000, 40), (4090, 6) })
        {
            var rng = new Random(500 + startPos + N);
            // Prefix K/V for positions [0, startPos), plus chunk K/V for [startPos, startPos+N).
            var prefixKV = Rand((startPos) * kvDim * 2, rng);
            var kAll = Rand(N * kvDim, rng);
            var vAll = Rand(N * kvDim, rng);
            var qAll = Rand(N * qDim, rng);

            float[] RunRef()
            {
                var kc = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim));
                var vc = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim));
                var scratch = gpu.Allocate(TensorShape.D1((long)numHeads * maxSeq));
                // Fill prefix.
                for (int p = 0; p < startPos; p++)
                {
                    var kt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2, kt, 0, kvDim);
                    var vt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2 + kvDim, vt, 0, kvDim);
                    var gk = gpu.Upload(kt, TensorShape.D1(kvDim)); var gv = gpu.Upload(vt, TensorShape.D1(kvDim));
                    gpu.KvAppend(gk, gv, kc, vc, kvDim, p, maxSeq);
                    gpu.Free(gk); gpu.Free(gv);
                }
                var outAll = new float[(long)N * qDim];
                for (int i = 0; i < N; i++)
                {
                    int pos = startPos + i;
                    float[] S(float[] a, int st) { var s = new float[st]; Array.Copy(a, (long)i * st, s, 0, st); return s; }
                    var gk = gpu.Upload(S(kAll, kvDim), TensorShape.D1(kvDim));
                    var gv = gpu.Upload(S(vAll, kvDim), TensorShape.D1(kvDim));
                    var gq = gpu.Upload(S(qAll, qDim), TensorShape.D1(qDim));
                    var go = gpu.Allocate(TensorShape.D1(qDim));
                    gpu.KvAppend(gk, gv, kc, vc, kvDim, pos, maxSeq);
                    gpu.Attention(gq, kc, vc, go, scratch, numHeads, numKvHeads, headDim, pos + 1, maxSeq);
                    gpu.Synchronize();
                    var ot = new float[qDim]; gpu.Download(go, ot);
                    Array.Copy(ot, 0, outAll, (long)i * qDim, qDim);
                    gpu.Free(gk); gpu.Free(gv); gpu.Free(gq); gpu.Free(go);
                }
                gpu.Free(kc); gpu.Free(vc); gpu.Free(scratch);
                return outAll;
            }

            float[] RunBatched()
            {
                var kc = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim));
                var vc = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim));
                for (int p = 0; p < startPos; p++)
                {
                    var kt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2, kt, 0, kvDim);
                    var vt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2 + kvDim, vt, 0, kvDim);
                    var gk = gpu.Upload(kt, TensorShape.D1(kvDim)); var gv = gpu.Upload(vt, TensorShape.D1(kvDim));
                    gpu.KvAppend(gk, gv, kc, vc, kvDim, p, maxSeq);
                    gpu.Free(gk); gpu.Free(gv);
                }
                var gKAll = gpu.Upload(kAll, TensorShape.D1((long)N * kvDim));
                var gVAll = gpu.Upload(vAll, TensorShape.D1((long)N * kvDim));
                var gQAll = gpu.Upload(qAll, TensorShape.D1((long)N * qDim));
                var gOut = gpu.Allocate(TensorShape.D1((long)N * qDim));
                gpu.KvAppendBatched(gKAll, gVAll, kc, vc, kvDim, startPos, maxSeq, N);
                gpu.AttentionBatched(gQAll, kc, vc, gOut, numHeads, numKvHeads, headDim, startPos, maxSeq, N);
                gpu.Synchronize();
                var outAll = new float[(long)N * qDim]; gpu.Download(gOut, outAll);
                gpu.Free(kc); gpu.Free(vc); gpu.Free(gKAll); gpu.Free(gVAll); gpu.Free(gQAll); gpu.Free(gOut);
                return outAll;
            }

            AssertBitId($"attn f32 startPos={startPos} N={N}", RunBatched(), RunRef());
        }
    }

    // ── Batched KV-append + SDPA (bf16 cache) ───────────────────────────────
    [Fact]
    public void AttentionBatched_Bf16_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int numHeads = 8, numKvHeads = 2, headDim = 128;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        const int maxSeq = 4096;
        foreach ((int startPos, int N) in new[] { (0, 5), (0, 64), (256, 17), (3500, 40) })
        {
            var rng = new Random(600 + startPos + N);
            var prefixKV = Rand((startPos) * kvDim * 2, rng);
            var kAll = Rand(N * kvDim, rng);
            var vAll = Rand(N * kvDim, rng);
            var qAll = Rand(N * qDim, rng);

            void FillPrefix(Tensor kc, Tensor vc)
            {
                for (int p = 0; p < startPos; p++)
                {
                    var kt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2, kt, 0, kvDim);
                    var vt = new float[kvDim]; Array.Copy(prefixKV, (long)p * kvDim * 2 + kvDim, vt, 0, kvDim);
                    var gk = gpu.Upload(kt, TensorShape.D1(kvDim)); var gv = gpu.Upload(vt, TensorShape.D1(kvDim));
                    gpu.KvAppendBf16(gk, gv, kc, vc, kvDim, p, maxSeq);
                    gpu.Free(gk); gpu.Free(gv);
                }
            }

            // Reference.
            var kcR = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim), DType.BFloat16);
            var vcR = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim), DType.BFloat16);
            var scratch = gpu.Allocate(TensorShape.D1((long)numHeads * maxSeq));
            FillPrefix(kcR, vcR);
            var refOut = new float[(long)N * qDim];
            for (int i = 0; i < N; i++)
            {
                int pos = startPos + i;
                float[] S(float[] a, int st) { var s = new float[st]; Array.Copy(a, (long)i * st, s, 0, st); return s; }
                var gk = gpu.Upload(S(kAll, kvDim), TensorShape.D1(kvDim));
                var gv = gpu.Upload(S(vAll, kvDim), TensorShape.D1(kvDim));
                var gq = gpu.Upload(S(qAll, qDim), TensorShape.D1(qDim));
                var go = gpu.Allocate(TensorShape.D1(qDim));
                gpu.KvAppendBf16(gk, gv, kcR, vcR, kvDim, pos, maxSeq);
                gpu.AttentionBf16(gq, kcR, vcR, go, scratch, numHeads, numKvHeads, headDim, pos + 1, maxSeq);
                gpu.Synchronize();
                var ot = new float[qDim]; gpu.Download(go, ot);
                Array.Copy(ot, 0, refOut, (long)i * qDim, qDim);
                gpu.Free(gk); gpu.Free(gv); gpu.Free(gq); gpu.Free(go);
            }
            gpu.Free(kcR); gpu.Free(vcR); gpu.Free(scratch);

            // Batched.
            var kcB = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim), DType.BFloat16);
            var vcB = gpu.Allocate(TensorShape.D1((long)maxSeq * kvDim), DType.BFloat16);
            FillPrefix(kcB, vcB);
            var gKAll = gpu.Upload(kAll, TensorShape.D1((long)N * kvDim));
            var gVAll = gpu.Upload(vAll, TensorShape.D1((long)N * kvDim));
            var gQAll = gpu.Upload(qAll, TensorShape.D1((long)N * qDim));
            var gOut = gpu.Allocate(TensorShape.D1((long)N * qDim));
            gpu.KvAppendBatchedBf16(gKAll, gVAll, kcB, vcB, kvDim, startPos, maxSeq, N);
            gpu.AttentionBatchedBf16(gQAll, kcB, vcB, gOut, numHeads, numKvHeads, headDim, startPos, maxSeq, N);
            gpu.Synchronize();
            var batOut = new float[(long)N * qDim]; gpu.Download(gOut, batOut);
            gpu.Free(kcB); gpu.Free(vcB); gpu.Free(gKAll); gpu.Free(gVAll); gpu.Free(gQAll); gpu.Free(gOut);

            AssertBitId($"attn bf16 startPos={startPos} N={N}", batOut, refOut);
        }
    }
}
