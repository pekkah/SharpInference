using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #141 (attention): parity for the memory-efficient flash-attention prefill
/// (<see cref="CudaBackend.FlashAttentionPrefill"/>, kernel
/// <c>llm_flash_attn_prefill_f32</c>) against the validated scalar batched kernels
/// it replaces — <see cref="CudaBackend.AttentionBatched"/> (global) and
/// <see cref="CudaBackend.AttentionSwaBatched"/> (sliding window). Random Q/K/V are
/// run through both; the flash kernel's online softmax reassociates the same sum, so
/// it tracks the reference to fp tolerance, not bit-exactly. Exercises GQA, both
/// head_dims Gemma 4 uses (256 SWA / 512 global), causal masking, windowing, and a
/// partial final query tile (nTok % 8 ≠ 0).
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaFlashAttnTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void FlashAttentionPrefill_MatchesScalarBatched()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // (numHeads, numKvHeads, headDim, window, nTok). window=0 → global (full causal).
        (int nh, int nkv, int hd, int win, int nTok)[] cfgs =
        {
            (8, 2, 256, 0, 200),    // global, SWA head_dim, partial last tile (200%8=0 → exact; ok)
            (8, 2, 512, 0, 173),    // global, global head_dim, partial last tile (173%8≠0)
            (8, 2, 256, 64, 200),   // sliding window 64 < nTok (windowing exercised)
            (8, 2, 512, 96, 130),   // sliding window 96, global head_dim
            (4, 4, 128, 0, 64),     // MHA (no GQA), small head_dim
        };

        foreach (var (nh, nkv, hd, win, nTok) in cfgs)
        {
            var rng = new Random(20260605 + nh * 7 + hd * 13 + win * 17 + nTok);
            int qDim = nh * hd, kvDim = nkv * hd;

            var q = new float[(long)nTok * qDim];
            for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
            // K/V cache filled for positions [0, nTok) (prefill startPos=0).
            var k = new float[(long)nTok * kvDim];
            var v = new float[(long)nTok * kvDim];
            for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }

            var gq = gpu.Upload(q, TensorShape.D1(q.Length));
            var gk = gpu.Upload(k, TensorShape.D1(k.Length));
            var gv = gpu.Upload(v, TensorShape.D1(v.Length));
            var gRef = gpu.Allocate(TensorShape.D1(q.Length));
            var gFla = gpu.Allocate(TensorShape.D1(q.Length));

            if (win == 0)
                gpu.AttentionBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos: 0, maxSeqLen: nTok, nTok: nTok);
            else
                gpu.AttentionSwaBatched(gq, gk, gv, gRef, nh, nkv, hd, startPos: 0, windowSize: win, maxSeqLen: nTok, nTok: nTok);
            gpu.FlashAttentionPrefill(gq, gk, gv, gFla, nh, nkv, hd, startPos: 0, windowSize: win, maxSeqLen: nTok, nTok: nTok);
            gpu.Synchronize();

            var outRef = new float[q.Length];
            var outFla = new float[q.Length];
            gpu.Download(gRef, outRef);
            gpu.Download(gFla, outFla);
            gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(gRef); gpu.Free(gFla);

            double sumSq = 0;
            for (int i = 0; i < outRef.Length; i++) sumSq += (double)outRef[i] * outRef[i];
            float rms = (float)Math.Sqrt(sumSq / outRef.Length) + 1e-9f;

            float maxAbs = 0;
            int mismatches = 0;
            for (int i = 0; i < outRef.Length; i++)
            {
                float diff = MathF.Abs(outFla[i] - outRef[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 1e-3f * rms) mismatches++;
            }
            Console.WriteLine(
                $"Flash nh={nh} nkv={nkv} hd={hd} win={win} nTok={nTok}: maxAbs={maxAbs:E2} rms={rms:E2} mismatches={mismatches}/{outRef.Length}");
            Assert.True(mismatches == 0,
                $"Flash attention diverged from scalar reference: {mismatches}/{outRef.Length} beyond 1e-3·rms ({rms:E3}), maxAbs={maxAbs:E3} (nh={nh} hd={hd} win={win} nTok={nTok}).");
        }
    }
}
