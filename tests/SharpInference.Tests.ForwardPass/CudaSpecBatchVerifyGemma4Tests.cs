using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #178: single-user speculative-decode <see cref="CudaForwardPass.BatchVerify"/> on the
/// <b>Gemma-4</b> path. <see cref="CudaForwardPass.SupportsBatchVerify"/> now admits Gemma-4
/// (it routed only the dense gate before), so a GPU draft can batch-verify on the 12B/E4B target.
/// The packed pass dispatches through <c>RunBatchedTrunkGemma4</c>, whose per-sequence attention
/// loop appends each of the k rows' K/V into the SHARED owned cache then attends in ascending row
/// order — the same append-then-attend causality the dense ragged path documents, but exercising
/// per-layer head_dim, SWA rings, the shared-KV tail, k_eq_v, PLE, sandwich norms, and the final
/// softcap.
///
/// Correctness contract (argmax-stable class, like <see cref="Gemma4CudaBatchForwardMultiTests"/>):
/// the Gemma-4 batched decode routes matmuls through cuBLAS GEMM (fp16), so BatchVerify is
/// argmax-stable — NOT bit-exact — vs the per-token fp32 <c>ForwardGemma4</c> loop. Asserted with
/// the maxAbs/top-5 tolerances of the dense <see cref="CudaSpecBatchVerifyTests"/>.
///
/// The ring-boundary oracle is the case the dense test (non-SWA Qwen3) never reaches and the one
/// the plan flagged as highest-risk: it prefills PAST the SWA ring (window + headroom) so the
/// verify operates on a wrapped ring where physical slot != logical position — the common case at
/// the ≥32K context this feature targets. Silent-skips when CUDA or the GGUF is absent; mirrors
/// <see cref="Gemma4CudaBatchForwardMultiTests"/>.
/// </summary>
public sealed class CudaSpecBatchVerifyGemma4Tests
{
    // E4B Q8_0 exercises the richest Gemma-4 geometry (per-layer head_dim 256, SWA rings, the
    // 18-layer shared-KV tail, PLE) and its Q8_0 weights are GEMM-N-batchable, so it reports
    // SupportsBatchVerify. Falls back to a 12B Q4_K_M (the issue's headline model) if present.
    private static readonly string[] TargetCandidates =
    {
        "gemma-4-E4B-it-Q8_0.gguf",
        "gemma-4-12B-it-qat-Q4_K_M.gguf",
        "gemma4-12b-q4km.gguf",
    };
    // Optional small same-vocab draft for the e2e oracle (skipped if absent).
    private static readonly string[] DraftCandidates =
    {
        "gemma-3-1b-it-Q8_0.gguf",
        "gemma-4-E2B-it-Q8_0.gguf",
    };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // SnapKV pinned off (it is structurally off for Gemma-4 anyway): keeps the oracle
    // machine-independent, matching the other Gemma-4 CUDA test fixtures.
    private static CudaForwardPass NewFwd(GgufModel model, CudaBackend gpu, ModelHyperparams hp,
        int ctx, string? kvDtype = null)
    {
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        var prevKv = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        if (kvDtype is not null) Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", kvDtype);
        try { return new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevKv);
        }
    }

    private static string? FindFirst(string[] candidates)
    {
        foreach (var file in candidates)
        {
            string[] absolute = { $@"E:\models\{file}", $@"C:\p\sharpi\models\{file}" };
            foreach (var p in absolute)
                if (File.Exists(p)) return p;
            var dir = Directory.GetCurrentDirectory();
            for (int i = 0; i < 8; i++)
            {
                var p = Path.Combine(dir, "models", file);
                if (File.Exists(p)) return p;
                var parent = Directory.GetParent(dir);
                if (parent is null) break;
                dir = parent.FullName;
            }
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static HashSet<int> TopKSet(ReadOnlySpan<float> logits, int k)
    {
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        var arr = logits.ToArray();
        Array.Sort(idx, (a, b) => arr[b].CompareTo(arr[a]));
        var set = new HashSet<int>();
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    private static (float maxAbs, int overlap) Compare(float[] reference, float[] candidate)
    {
        Assert.Equal(reference.Length, candidate.Length);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - candidate[i]));
        var refTop = TopKSet(reference, 5);
        var candTop = TopKSet(candidate, 5);
        int overlap = 0;
        foreach (var t in candTop) if (refTop.Contains(t)) overlap++;
        return (maxAbs, overlap);
    }

    // Argmax parity tolerant of an fp16-GEMM near-tie flip — accepted ONLY when the reference's
    // top-2 are within tieEps (mirrors Gemma4CudaBatchForwardMultiTests.AssertArgmaxOrNearTie).
    private static void AssertArgmaxOrNearTie(float[] reference, float[] candidate, float tieEps, string label)
    {
        int rArg = Argmax(reference), cArg = Argmax(candidate);
        if (rArg == cArg) return;
        float gap = MathF.Abs(reference[rArg] - reference[cArg]);
        Assert.True(gap < tieEps,
            $"{label}: batched argmax {cArg} != sequential {rArg}, NOT a near-tie (reference gap {gap:F3} ≥ {tieEps:F1}) " +
            "— a real wiring divergence (per-layer geometry / SWA ring / shared-KV / PLE), not fp16 rounding.");
    }

    // Real Gemma-4 token ids (BOS=2 + natural mid-vocab subwords), matching the activation regime
    // the established Gemma-4 batched oracles assert under. Natural tokens for the long ring prefill.
    private static readonly int[] GemmaPrompt = { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222 };
    private static int[] NaturalTokens(int count)
    {
        var t = new int[count];
        for (int i = 0; i < count; i++) t[i] = i == 0 ? 2 : 200 + (i * 37) % 8000;
        return t;
    }

    // The k-row packed verify batches more rows through the fp16 cuBLAS GEMM than the N=2 decode
    // the sibling oracles' maxAbs<1.0 bound was calibrated for, so absolute logit divergence is
    // modestly larger (~1.1 at k=4/6 on the ±softcap range). Argmax-or-near-tie + top-5 overlap are
    // the real correctness contract; maxAbs is a coarse divergence guard at the fp16-GEMM scale.
    private static void AssertParity(CudaForwardPass fwd, int[] prompt, int k, string label, float maxAbsTol = 1.5f)
    {
        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(prompt);
        int P = prompt.Length;

        // Greedy-chain k tokens so the verified positions carry realistic activations.
        var tokens = new int[k];
        tokens[0] = Argmax(prefillLogits);
        var reference = new float[k][];
        for (int i = 0; i < k; i++)
        {
            var logits = fwd.Forward(tokens[i], P + i);
            reference[i] = logits.ToArray();
            if (i + 1 < k) tokens[i + 1] = Argmax(logits);
        }

        // Soft rewind (stale K/V must be overwritten) and batch-verify.
        fwd.TruncateTo(P);
        float[][] batch = fwd.BatchVerify(tokens, P);

        Assert.Equal(k, batch.Length);
        for (int i = 0; i < k; i++)
        {
            var (maxAbs, overlap) = Compare(reference[i], batch[i]);
            AssertArgmaxOrNearTie(reference[i], batch[i], tieEps: 0.5f, $"{label} pos {i}");
            Assert.True(overlap >= 4,
                $"{label} pos {i}: batched top-5 overlaps sequential in {overlap}/5 (maxAbs={maxAbs}).");
            Assert.True(maxAbs < maxAbsTol,
                $"{label} pos {i}: batched vs sequential diverged: maxAbs={maxAbs} (tol {maxAbsTol}).");
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(6)]   // not a capacity-stamped WS size — exercises pad-to-capacity dispatch
    public void Gemma4_BatchVerify_MatchesSequentialForward(int k)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindFirst(TargetCandidates);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);   // Gemma-4 marker

        using var fwd = NewFwd(model, gpu, hp, ctx: 512);
        Assert.True(fwd.SupportsBatchVerify,
            "A GEMM-N-batchable Gemma-4 model must report SupportsBatchVerify on the CUDA path (#178).");

        AssertParity(fwd, GemmaPrompt, k, "Gemma4 verify");
    }

    /// <summary>
    /// q8_0 KV variant — the issue's headline config (q8 KV frees the VRAM the draft needs). The
    /// quantized ring is lossy, so the maxAbs tolerance (not exact equality) carries it; argmax
    /// must stay stable, the same contract the q8 KV decode path holds elsewhere.
    /// </summary>
    [Fact]
    public void Gemma4_BatchVerify_Q8Kv_MatchesSequentialForward()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindFirst(TargetCandidates);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp, ctx: 512, kvDtype: "q8_0");
        if (!fwd.SupportsBatchVerify) return;   // q8 KV geometry unsupported on this build → skip

        // q8 KV is lossy → a slightly looser maxAbs than the fp32-KV path; argmax stays stable.
        AssertParity(fwd, GemmaPrompt, k: 4, "Gemma4 q8-KV verify", maxAbsTol: 2.0f);
    }

    /// <summary>
    /// Ring-wrap oracle (the highest-risk Gemma-4 case): prefill PAST the SWA ring
    /// (<c>window + headroom</c>, ≈5K) so the verify writes ring slots whose physical index !=
    /// logical position — the common case at ≥32K context. The batched packed verify must still
    /// match k sequential forwards at the wrapped offset. Heavy (multi-thousand-token prefill);
    /// model-gated so it only runs locally on a GPU with the GGUF.
    /// </summary>
    [Fact]
    public void Gemma4_BatchVerify_AcrossSwaRingBoundary()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindFirst(TargetCandidates);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (hp.SlidingWindowSize <= 0) return;   // no SWA layers → nothing to wrap

        // SwaRingSize = min(ctx, window + SwaRingHeadroom>=4096). Pick a ctx comfortably above
        // window+4096 and a prefill length past the ring so the SWA cache has wrapped.
        int ring = hp.SlidingWindowSize + 4096;
        int ctx = ring + 3072;
        int prefillLen = ring + 512;

        using var fwd = NewFwd(model, gpu, hp, ctx: ctx);
        if (!fwd.SupportsBatchVerify) return;
        if (prefillLen + 8 >= fwd.MaxSeqLen) return;   // model can't seat the wrap → skip

        // Long synthetic context → looser maxAbs (deeper fp16 accumulation over thousands of
        // positions); the point is argmax-stable correctness on the WRAPPED ring.
        AssertParity(fwd, NaturalTokens(prefillLen), k: 4, "Gemma4 ring-wrap verify", maxAbsTol: 3.0f);
    }

    /// <summary>
    /// Rollback oracle: verify [t0, junk, junk, junk], accept only t0, TruncateTo(P+1), commit the
    /// correction t1. Post-rollback logits must match the sequential trajectory that never saw the
    /// rejected tokens — catches stale SWA ring-slot leaks past the truncation point.
    /// </summary>
    [Fact]
    public void Gemma4_BatchVerify_TruncateAndCommit_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindFirst(TargetCandidates);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp, ctx: 512);
        Assert.True(fwd.SupportsBatchVerify);

        var prompt = GemmaPrompt;
        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(prompt);
        int P = prompt.Length;
        int t0 = Argmax(prefillLogits);

        // Sequential reference trajectory: accept t0 → t1, then commit t1.
        float[] afterT0 = fwd.Forward(t0, P).ToArray();
        int t1 = Argmax(afterT0);
        float[] reference = fwd.Forward(t1, P + 1).ToArray();

        // Spec-step shape: rewind, verify [t0, junk, junk, junk], accept only t0, commit t1.
        fwd.TruncateTo(P);
        int junk = (t0 + 7919) % hp.VocabSize;
        float[][] batch = fwd.BatchVerify([t0, junk, junk, junk], P);
        AssertArgmaxOrNearTie(afterT0, batch[0], tieEps: 0.5f, "after-t0");   // verify[0] still picks t1

        fwd.TruncateTo(P + 1);
        float[] committed = fwd.Forward(t1, P + 1).ToArray();

        var (maxAbs, overlap) = Compare(reference, committed);
        AssertArgmaxOrNearTie(reference, committed, tieEps: 0.5f, "post-rollback commit");
        Assert.True(overlap >= 4, $"Post-rollback top-5 overlap {overlap}/5 (maxAbs={maxAbs}).");
        Assert.True(maxAbs < 1.0f, $"Post-rollback diverged: maxAbs={maxAbs}.");
    }

    /// <summary>
    /// E2E greedy parity: SpeculativeDecoder with a CUDA Gemma-4 target + a small same-vocab CPU
    /// draft must emit the target's own non-spec greedy continuation — the spec invariant (the
    /// draft only proposes; every emitted token is the target's argmax). Gemma-4 BatchVerify is
    /// argmax-stable (not bit-exact), so a divergence means a real verify/rollback bug or an
    /// FP-borderline argmax flip — investigate before weakening. Skips without the draft GGUF.
    /// </summary>
    [Fact]
    public void Gemma4_SpecDecode_GreedyParity_E2E()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var targetPath = FindFirst(TargetCandidates);
        var draftPath = FindFirst(DraftCandidates);
        if (targetPath is null || draftPath is null) return;

        const int DecodeTokens = 32;

        using var targetModel = GgufModel.Open(targetPath);
        var targetHp = ModelHyperparams.FromGgufMetadata(targetModel.Metadata, targetModel);
        using var target = NewFwd(targetModel, gpu, targetHp, ctx: 512);
        Assert.True(target.SupportsBatchVerify);

        using var draftModel = GgufModel.Open(draftPath);
        var draftHp = ModelHyperparams.FromGgufMetadata(draftModel.Metadata, draftModel);
        if (targetHp.VocabSize != draftHp.VocabSize) return;   // different tokenizer → not a draft

        var prompt = GemmaPrompt;

        // Non-spec greedy baseline on the CUDA target.
        target.ResetCache();
        var logits = target.Prefill(prompt);
        int P = prompt.Length;
        var baseline = new List<int>();
        int tok = Argmax(logits);
        for (int i = 0; i < DecodeTokens; i++)
        {
            baseline.Add(tok);
            logits = target.Forward(tok, P + i);
            tok = Argmax(logits);
        }

        using var cpu = new CpuBackend();
        using var draft = new SharpInference.Engine.ForwardPass(draftModel, cpu, draftHp);

        target.ResetCache();
        var targetLogits = target.Prefill(prompt).ToArray();
        var draftLogits = draft.Prefill(prompt).ToArray();

        var spec = new SpeculativeDecoder(target, draft, lookahead: 4);
        spec.Initialize(P, targetLogits, draftLogits);

        var emitted = new List<int>();
        spec.Decode(DecodeTokens, [], emitted.Add);

        Assert.Equal(baseline, emitted);
    }
}
