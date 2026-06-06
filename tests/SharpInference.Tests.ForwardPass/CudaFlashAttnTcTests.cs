using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #146: parity for the tensor-core flash-attention prefill
/// (<see cref="CudaBackend.FlashAttentionPrefillTc"/>, kernel
/// <c>llm_flash_attn_prefill_tc</c>) against the validated scalar batched kernels
/// (<see cref="CudaBackend.AttentionBatched"/> / <see cref="CudaBackend.AttentionSwaBatched"/>).
/// The TC kernel rounds Q, K, V *and* the softmax probabilities P to fp16 for the
/// mma multiplicands (the half2 kernel kept V fp32), so it tracks the reference to a
/// looser fp16 tolerance. Same config matrix as the half2 flash test: GQA, both
/// Gemma 4 head_dims (256 SWA / 512 global), causal, windowing, partial last tile.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaFlashAttnTcTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void FlashAttentionPrefillTc_MatchesScalarBatched()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // (numHeads, numKvHeads, headDim, window, nTok). window=0 → global (full causal).
        (int nh, int nkv, int hd, int win, int nTok)[] cfgs =
        {
            (8, 2, 256, 0, 200),    // global, SWA head_dim
            (8, 2, 512, 0, 173),    // global, global head_dim, partial last tile
            (8, 2, 256, 64, 200),   // sliding window 64 < nTok
            (8, 2, 512, 96, 130),   // sliding window 96, global head_dim
            (4, 4, 128, 0, 64),     // MHA (no GQA), small head_dim
        };

        foreach (var (nh, nkv, hd, win, nTok) in cfgs)
        {
            var rng = new Random(20260606 + nh * 7 + hd * 13 + win * 17 + nTok);
            int qDim = nh * hd, kvDim = nkv * hd;

            var q = new float[(long)nTok * qDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            var k = new float[(long)nTok * kvDim];
            var v = new float[(long)nTok * kvDim];
            for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }

            var gq = gpu.Upload(q, TensorShape.D1(q.Length));
            var gk = gpu.Upload(k, TensorShape.D1(k.Length));
            var gv = gpu.Upload(v, TensorShape.D1(v.Length));
            var gRef = gpu.Allocate(TensorShape.D1(q.Length));
            var gTc = gpu.Allocate(TensorShape.D1(q.Length));

            if (win == 0)
                gpu.AttentionBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos: 0, maxSeqLen: nTok, nTok: nTok);
            else
                gpu.AttentionSwaBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos: 0, windowSize: win, maxSeqLen: nTok, nTok: nTok);
            gpu.FlashAttentionPrefillTc(gq, gk, gv, gTc, nh, nkv, hd, startPos: 0, windowSize: win, maxSeqLen: nTok, nTok: nTok);
            gpu.Synchronize();

            var outRef = new float[q.Length];
            var outTc = new float[q.Length];
            gpu.Download(gRef, outRef);
            gpu.Download(gTc, outTc);
            gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(gRef); gpu.Free(gTc);

            double sumSq = 0;
            for (int i = 0; i < outRef.Length; i++) sumSq += (double)outRef[i] * outRef[i];
            float rms = (float)Math.Sqrt(sumSq / outRef.Length) + 1e-9f;

            float maxAbs = 0;
            int mismatches = 0;
            for (int i = 0; i < outRef.Length; i++)
            {
                float diff = MathF.Abs(outTc[i] - outRef[i]);
                // NaN/Inf must fail (a masked-tile online-softmax bug produces NaN, and
                // NaN > threshold is false — count it explicitly so it can't slip through).
                if (!float.IsFinite(outTc[i])) { mismatches++; continue; }
                maxAbs = MathF.Max(maxAbs, diff);
                // Q,K,V,P all fp16-rounded for the mma → ~fp16-relative attention error.
                if (diff > 2e-2f * rms) mismatches++;
            }
            Console.WriteLine(
                $"FlashTC nh={nh} nkv={nkv} hd={hd} win={win} nTok={nTok}: maxAbs={maxAbs:E2} rms={rms:E2} mismatches={mismatches}/{outRef.Length}");
            // Allow a tiny tail of outliers from fp16 P·V accumulation; the bulk must match.
            Assert.True(mismatches <= outRef.Length / 200 + 1,
                $"TC flash attention diverged from scalar reference: {mismatches}/{outRef.Length} beyond 2e-2·rms ({rms:E3}), maxAbs={maxAbs:E3} (nh={nh} hd={hd} win={win} nTok={nTok}).");
        }
    }
}
