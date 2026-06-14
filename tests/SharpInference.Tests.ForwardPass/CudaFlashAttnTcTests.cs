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
/// Issue #151 hardens three coverage gaps from the #148 review:
///   1. <c>startPos &gt; 0</c> — the continued-prefill / chat-continuation re-prefill path
///      (Q is only the new tokens at absolute positions <c>[startPos, startPos+nTok)</c>,
///      while K/V hold the full <c>[0, startPos+nTok)</c> history).
///   2. single partial tile (<c>nTok &lt; 16</c>, <c>gy == 1</c>) — the degenerate
///      online-softmax / masking case.
///   3. a genuinely TC1-only head_dim (<c>%16==0 &amp;&amp; %64!=0</c>) — the #146 single-warp
///      shared-O sizing path that the #147 multi-warp kernel (needs <c>%64</c>) never reaches.
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

    // (numHeads, numKvHeads, headDim, window, nTok, startPos). window=0 → global (full causal).
    // All head_dims are multiples of 64 so the same matrix exercises both the #146 single-warp
    // (tc) and the #147 multi-warp (tc2, W·16=64) kernels.
    private static (int nh, int nkv, int hd, int win, int nTok, int startPos)[] Configs() => new[]
    {
        (8, 2, 256, 0, 200, 0),    // global, SWA head_dim
        (8, 2, 512, 0, 173, 0),    // global, global head_dim, partial last tile
        (8, 2, 256, 64, 200, 0),   // sliding window 64 < nTok
        (8, 2, 512, 96, 130, 0),   // sliding window 96, global head_dim
        (4, 4, 128, 0, 64, 0),     // MHA (no GQA), small head_dim

        // #151 gap 1 — startPos > 0 (continued prefill): K/V carry prior context.
        // These keep maxSeqLen = startPos+nTok (a flat cache), so they validate the
        // causal/window mask + key-tile-span interaction with startPos — NOT the SWA
        // ring wrap (abs_k % maxSeqLen is the identity here). Ring-wrap is covered at
        // the model level by CudaForwardPassKvDtypeTests' long-prompt chunked prefill.
        (8, 2, 256, 0, 64, 137),   // global, prior context before the new tokens
        (8, 2, 512, 96, 80, 211),  // SWA, window (96) bounded well inside the prior context (startPos 211)

        // #151 gap 2 — single partial tile (nTok < 16, gy == 1).
        (8, 2, 256, 0, 1, 0),      // single query, single key
        (8, 2, 256, 0, 7, 0),      // sub-tile (7 < 16), global
        (8, 2, 256, 32, 7, 0),     // sub-tile with a sliding window
        (8, 2, 256, 0, 7, 40),     // sub-tile with prior context (nTok<16 AND startPos>0)
    };

    // #151 gap 3 — head_dim % 16 == 0 but % 64 != 0. Only the #146 single-warp TC1 kernel
    // (shared-O sizing) accepts these; FlashAttentionPrefillTc2 requires % 64 and throws,
    // so these run in the TC1 path only.
    private static (int nh, int nkv, int hd, int win, int nTok, int startPos)[] Tc1OnlyConfigs() => new[]
    {
        (8, 2, 80, 0, 130, 0),     // hd=80 (5×16), global — exercises the TC1-only shared-O path
        (8, 2, 48, 64, 96, 0),     // hd=48 (3×16), sliding window
        (4, 4, 112, 0, 33, 17),    // hd=112 (7×16), MHA, startPos>0, partial last tile
    };

    [Fact]
    public void FlashAttentionPrefillTc_MatchesScalarBatched()   // #146 single-warp
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        RunParity(gpu, Configs(), tc2: false, label: "TC1");
        RunParity(gpu, Tc1OnlyConfigs(), tc2: false, label: "TC1-only-hd");
    }

    [Fact]
    public void FlashAttentionPrefillTc2_MatchesScalarBatched()  // #147 multi-warp/d-split
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        RunParity(gpu, Configs(), tc2: true, label: "TC2");
    }

    private static void RunParity(
        CudaBackend gpu,
        (int nh, int nkv, int hd, int win, int nTok, int startPos)[] configs,
        bool tc2, string label)
    {
        foreach (var (nh, nkv, hd, win, nTok, startPos) in configs)
        {
            var rng = new Random(20260606 + nh * 7 + hd * 13 + win * 17 + nTok + startPos * 101);
            int qDim = nh * hd, kvDim = nkv * hd;
            int kvLen = startPos + nTok;   // K/V cache holds the full [0, startPos+nTok) history

            var q = new float[(long)nTok * qDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            var k = new float[(long)kvLen * kvDim];
            var v = new float[(long)kvLen * kvDim];
            for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }

            var gq = gpu.Upload(q, TensorShape.D1(q.Length));
            var gk = gpu.Upload(k, TensorShape.D1(k.Length));
            var gv = gpu.Upload(v, TensorShape.D1(v.Length));
            var gRef = gpu.Allocate(TensorShape.D1(q.Length));
            var gTc = gpu.Allocate(TensorShape.D1(q.Length));

            if (win == 0)
                gpu.AttentionBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos, maxSeqLen: kvLen, nTok: nTok);
            else
                gpu.AttentionSwaBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos, windowSize: win, maxSeqLen: kvLen, nTok: nTok);
            if (tc2)
                gpu.FlashAttentionPrefillTc2(gq, gk, gv, gTc, nh, nkv, hd, startPos, windowSize: win, maxSeqLen: kvLen, nTok: nTok);
            else
                gpu.FlashAttentionPrefillTc(gq, gk, gv, gTc, nh, nkv, hd, startPos, windowSize: win, maxSeqLen: kvLen, nTok: nTok);
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
                $"Flash{label} nh={nh} nkv={nkv} hd={hd} win={win} nTok={nTok} startPos={startPos}: maxAbs={maxAbs:E2} rms={rms:E2} mismatches={mismatches}/{outRef.Length}");
            // Allow a tiny tail of outliers from fp16 P·V accumulation; the bulk must match.
            Assert.True(mismatches <= outRef.Length / 200 + 1,
                $"{label} flash attention diverged from scalar reference: {mismatches}/{outRef.Length} beyond 2e-2·rms ({rms:E3}), maxAbs={maxAbs:E3} (nh={nh} hd={hd} win={win} nTok={nTok} startPos={startPos}).");
        }
    }
}
