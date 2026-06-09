using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// bf16 KV-cache parity for the dense CudaForwardPass (issue #179). With
/// <c>SHARPI_KV_DTYPE=bf16</c> the K/V cache is stored half-width; kernel
/// arithmetic stays fp32, so decode must be argmax-stable vs the fp32 cache at
/// short context (only the stored value's mantissa is narrowed). Both runs are
/// pinned to the per-token attention path (the fp32 batched flash/TC kernels are
/// not bit-exact vs scalar, and bf16 batched prefill is a follow-up), and the bf16
/// decode is teacher-forced onto the fp32 trajectory so the KV dtype is the only
/// variable at each position. Asserts, per teacher-forced position:
/// <list type="bullet">
///   <item>all bf16 logits are finite,</item>
///   <item>fp32's top-1 stays within bf16's top-5 — the reorder-tolerant
///         "argmax-stable" criterion (a genuine near-tie can flip top-1 with no
///         kernel bug, so top-1 equality is not asserted),</item>
///   <item>the logit max-abs gap is within a per-model rounding budget.</item>
/// </list>
///
/// Covers Qwen3-8B (non-SWA dense → global <c>AttentionBf16</c>/<c>KvAppendBf16</c>)
/// and Gemma 4 (SWA + global → also <c>AttentionSwaBf16</c>; the 12B Q4_0 adds
/// <c>attention_k_eq_v</c> global layers and a wider rounding budget). SnapKV is
/// forced off so the KV dtype is the only variable. Each case is skipped silently
/// when CUDA is unavailable or the GGUF isn't on disk.
/// </summary>
public sealed class CudaForwardPassKvDtypeTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindModelPath(string filename)
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{filename}",
            $@"E:\models\{filename}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", filename);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Prefill <paramref name="prompt"/>, then decode <paramref name="steps"/> tokens
    /// on a fresh CudaForwardPass with the given KV dtype. When <paramref name="forced"/>
    /// is non-null the decode is TEACHER-FORCED on those tokens (instead of the model's
    /// own greedy pick) so two runs follow an identical trajectory — that keeps the KV
    /// dtype the only variable at each decode position. Returns the per-position logits
    /// (index 0 = prefill, 1.. = each decode step) and the greedy argmax at each. The
    /// env var is read in the constructor, so it must be set before construction.
    /// </summary>
    private static (float[][] logits, int[] argmax) RunPrefillDecode(
        CudaBackend gpu, string path, string? kvDtype, string prompt, int steps, int ctx, int[]? forced,
        bool batchedPrefill = false)
    {
        var prevKv = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", kvDtype);
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0"); // isolate the KV dtype
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int useCtx = Math.Min(hp.ContextLength, ctx);
            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: useCtx);
            // Pin the prefill path. The argmax-stability test compares fp32 vs bf16 with
            // both on the per-token path (the fp32 batched flash/TC online softmax isn't
            // bit-exact vs scalar, which would conflate with the KV-dtype effect). The
            // batched-prefill test (#179 Increment 1.5) instead toggles this true to
            // exercise the scalar bf16 batched kernels.
            fwd.BatchedPrefillEnabled = batchedPrefill;

            var tokens = tokenizer.Encode(prompt).ToArray();
            var perPos = new float[steps + 1][];
            var argmax = new int[steps + 1];

            var logits = fwd.Prefill(tokens).ToArray();
            perPos[0] = logits;
            argmax[0] = Sampler.Greedy(logits);

            for (int i = 0; i < steps; i++)
            {
                int fed = forced is not null ? forced[i] : argmax[i];
                logits = fwd.Forward(fed, tokens.Length + i).ToArray();
                perPos[i + 1] = logits;
                argmax[i + 1] = Sampler.Greedy(logits);
            }
            return (perPos, argmax);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevKv);
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
        }
    }

    private static void AssertKvParity(string filename, string kvDtype, string prompt, int? eosToken, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        const int steps = 6;
        const int ctx = 2048;

        // fp32 reference first; teacher-force the narrowed dtype onto the SAME trajectory so
        // each decode position sees identical inputs and the KV dtype is the only variable.
        // Greedy-token equality alone is fragile on near-tie tokens (a borderline pair
        // flips on store rounding without any kernel bug), so parity is asserted at the
        // logit level — top-1 stable + small max-abs — per feedback_cross_backend_parity_test.
        var (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null);
        var (kv, kvArgmax) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: f32Argmax);

        // Coherence (feedback_forward_pass_tests): the fp32 reference must be a real
        // decode — IsFinite alone passes on a degenerate all-EOS run.
        Assert.True(eosToken is null || f32Argmax[0] != eosToken,
            $"{filename}: fp32 reference decoded EOS first — prompt put the model OOD, not a KV test.");

        // Per-position parity. Index 0 = prefill (full-prompt trunk over the whole KV
        // cache); 1.. = teacher-forced decode steps. Both runs saw identical inputs at
        // every position, so a faithful narrowed path differs only by accumulated store
        // rounding. We assert (a) logit max-abs is within the rounding budget — the
        // primary faithfulness measure — and (b) fp32's top-1 stays in the narrowed top-5,
        // a reorder-tolerant "argmax-stable" check. We do NOT assert top-1 equality: a
        // genuine near-tie (e.g. the 12B's degenerate-repeat positions, where adjacent
        // token IDs sit within store noise) can flip top-1 with no kernel bug.
        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(f32[p].Length, kv[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32[p].Length; i++)
            {
                Assert.True(float.IsFinite(kv[p][i]), $"{filename}: non-finite {kvDtype} logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - kv[p][i]));
            }
            Assert.True(maxAbs < maxAbsTol,
                $"{filename}: pos {p} {kvDtype} vs fp32 logit max-abs diff {maxAbs:F3} exceeds the " +
                $"rounding budget ({maxAbsTol:F1}) — likely an arithmetic divergence (SWA-ring / k_eq_v / attn_scale).");
            Assert.True(TopK(kv[p], 5).Contains(f32Argmax[p]),
                $"{filename}: pos {p} fp32 top-1 ({f32Argmax[p]}) fell out of {kvDtype}'s top-5 " +
                $"(max-abs {maxAbs:F3}) — the {kvDtype} path reordered the head of the distribution.");
        }
    }

    /// <summary>
    /// Increment 1.5: bf16 batched prefill must match fp32 batched prefill. Both runs use
    /// the SAME batched path — for head_dim%64 models that's the Tc2 tensor-core flash
    /// kernel (the bf16 thunk vs the fp32 one), with per-token decode after — so the only
    /// variable is the KV-cache dtype. This isolates the bf16 flash/scalar batched kernels
    /// and the dispatch + chunking gate. fp32 is the reference; bf16 is teacher-forced onto
    /// its trajectory. Tolerances match the per-token argmax-stable test (store-rounding
    /// budget), since the attention algorithm is identical between the two runs.
    /// </summary>
    private static void AssertKvBatchedPrefillParity(string filename, string kvDtype, string prompt, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        const int steps = 6;
        const int ctx = 2048;

        var (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null, batchedPrefill: true);
        var (kv, _) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: f32Argmax, batchedPrefill: true);

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(f32[p].Length, kv[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32[p].Length; i++)
            {
                Assert.True(float.IsFinite(kv[p][i]), $"{filename}: non-finite batched-{kvDtype} logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - kv[p][i]));
            }
            Assert.True(maxAbs < maxAbsTol,
                $"{filename}: pos {p} batched {kvDtype}-vs-fp32 logit max-abs {maxAbs:F3} exceeds " +
                $"{maxAbsTol:F1} — the {kvDtype} batched/flash kernels diverge from the fp32 batched path.");
            Assert.True(TopK(kv[p], 5).Contains(f32Argmax[p]),
                $"{filename}: pos {p} fp32 top-1 ({f32Argmax[p]}) fell out of batched-{kvDtype}'s top-5.");
        }
    }

    /// <summary>
    /// Increment 1.5b: bf16 chunked prefill PAST the 4096 cap must stay argmax-stable vs
    /// fp32 chunked prefill. A prompt longer than PrefillBatchChunk (4096) forces the chunk
    /// loop, which for bf16 requires the Tc2 flash thunk on every layer
    /// (Bf16FlashTc2CoversAllLayers) and exercises the SWA KV ring wrapping across chunk
    /// boundaries under a bf16 store — the path most likely to hide a ring/index bug.
    ///
    /// Unlike the short-prompt tests this asserts the issue's LONG-context bar (coherent +
    /// argmax-stable), not tight logit parity: bf16-store rounding compounds with sequence
    /// length and autoregressive depth (observed max-abs grows 2→5 over a 5000-token
    /// context), so the hard check is top-5 stability at every position plus a loose
    /// finite/blow-up ceiling. A structural cross-chunk bug would drop top-5 or NaN, not
    /// nudge logits by a few units.
    /// </summary>
    private static void AssertKvChunkedPrefillParity(string filename, string kvDtype, float maxAbsCeiling)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        const int steps = 4;
        const int ctx = 6144;           // > 4096 so the prefill chunk loop runs
        const int promptLen = 5000;     // spans two PrefillBatchChunk windows

        // A long, varied prompt so attention sees a non-degenerate pattern across chunks.
        using (var model = GgufModel.Open(path))
        {
            var tok = GgufTokenizer.FromGgufModel(model);
            var sb = new System.Text.StringBuilder();
            const string seed = "The quick brown fox jumps over the lazy dog. " +
                                "Sphinx of black quartz, judge my vow. " +
                                "Pack my box with five dozen liquor jugs. ";
            while (tok.Encode(sb.ToString()).Count < promptLen) sb.Append(seed);
            string prompt = sb.ToString();

            var (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null, batchedPrefill: true);
            var (kv, _) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: f32Argmax, batchedPrefill: true);

            for (int p = 0; p <= steps; p++)
            {
                Assert.Equal(f32[p].Length, kv[p].Length);
                float maxAbs = 0f;
                for (int i = 0; i < f32[p].Length; i++)
                {
                    Assert.True(float.IsFinite(kv[p][i]), $"{filename}: non-finite chunked-{kvDtype} logit at pos {p}, idx {i}.");
                    maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - kv[p][i]));
                }
                // Hard check: argmax-stable (top-5 overlap). Soft ceiling: catch a blown-up
                // kernel / NaN, not the expected long-context rounding accumulation.
                Assert.True(TopK(kv[p], 5).Contains(f32Argmax[p]),
                    $"{filename}: pos {p} fp32 top-1 ({f32Argmax[p]}) fell out of chunked-{kvDtype}'s top-5 " +
                    $"(max-abs {maxAbs:F3}) — a cross-chunk SWA-ring or Tc2-{kvDtype} divergence.");
                Assert.True(maxAbs < maxAbsCeiling,
                    $"{filename}: pos {p} chunked {kvDtype}-vs-fp32 logit max-abs {maxAbs:F3} exceeds the " +
                    $"blow-up ceiling {maxAbsCeiling:F1} — likely a kernel bug, not rounding.");
            }
        }
    }

    /// <summary>Indices of the <paramref name="k"/> largest entries of <paramref name="v"/>.</summary>
    private static HashSet<int> TopK(float[] v, int k)
    {
        var idx = new int[v.Length];
        for (int i = 0; i < v.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        var set = new HashSet<int>(k);
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    private const string LowEntropyPrompt = "The quick brown fox jumps over the lazy";

    /// <summary>Qwen3-8B Q4_K: non-SWA dense, exercises the global bf16 attention + append.</summary>
    [Fact]
    public void Qwen3_8B_Bf16Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("Qwen3-8B-Q4_K_M.gguf", "bf16", LowEntropyPrompt, eosToken: null, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 E4B Q8_0: SWA + global layers, exercises AttentionSwaBf16.</summary>
    [Fact]
    public void Gemma4_E4B_Bf16Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("gemma-4-E4B-it-Q8_0.gguf", "bf16", LowEntropyPrompt, eosToken: null, maxAbsTol: 1.5f);

    /// <summary>
    /// Gemma 4 12B QAT: the driving model — adds attention_k_eq_v global layers. Q4_0
    /// 4-bit weights over 48 layers accumulate more bf16-store rounding, so the budget
    /// is wider than the Q8_0/Q4_K cases (observed peak ~4.0); top-1/top-5 stay stable.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Bf16Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("gemma-4-12b-it-qat-q4_0.gguf", "bf16", LowEntropyPrompt, eosToken: null, maxAbsTol: 8.0f);

    // ── Increment 1.5: bf16 batched prefill agrees with bf16 per-token ──────

    /// <summary>Qwen3-8B Q4_K: bf16 global batched prefill (AttentionBatchedBf16).</summary>
    [Fact]
    public void Qwen3_8B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("Qwen3-8B-Q4_K_M.gguf", "bf16", LowEntropyPrompt, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 E4B Q8_0: bf16 SWA + global batched prefill (AttentionSwaBatchedBf16).</summary>
    [Fact]
    public void Gemma4_E4B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("gemma-4-E4B-it-Q8_0.gguf", "bf16", LowEntropyPrompt, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 12B QAT Q4_0: bf16 batched prefill with k_eq_v globals.</summary>
    [Fact]
    public void Gemma4_12B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("gemma-4-12b-it-qat-q4_0.gguf", "bf16", LowEntropyPrompt, maxAbsTol: 8.0f);

    // ── Increment 1.5b: bf16 chunked prefill past 4096 (Tc2 flash + SWA ring) ──

    /// <summary>
    /// Gemma 4 E4B Q8_0: bf16 Tc2-flash chunked prefill across the SWA ring boundary.
    /// The budget is wider than the short-prompt tests: bf16-store rounding accumulates
    /// with sequence length, so a 5000-token prefill diverges more than a ~10-token one
    /// (observed ~2.1); a structural cross-chunk bug would produce garbage, not ~2. Top-5
    /// stability is the real argmax-stable check.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Bf16ChunkedPrefill_MatchesFp32()
        => AssertKvChunkedPrefillParity("gemma-4-E4B-it-Q8_0.gguf", "bf16", maxAbsCeiling: 25.0f);

    // ── Increment 2: q8_0 KV (block-quantized, ~quarter-fp32) ────────────────
    // Same parity oracles as bf16, with the q8_0 cache. q8_0's per-32-block 8-bit
    // quantization is a coarser store than bf16's per-element 8-bit mantissa, so the
    // max-abs budgets are a touch wider; the top-5 argmax-stable check is unchanged
    // and is the real correctness gate. Tolerances were set from observed peaks.

    /// <summary>Qwen3-8B Q4_K: non-SWA dense, exercises AttentionQ8_0 / KvAppendQ8_0.</summary>
    [Fact]
    public void Qwen3_8B_Q8Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("Qwen3-8B-Q4_K_M.gguf", "q8_0", LowEntropyPrompt, eosToken: null, maxAbsTol: 2.5f);

    /// <summary>Gemma 4 E4B Q8_0: SWA + global layers, exercises AttentionSwaQ8_0.</summary>
    [Fact]
    public void Gemma4_E4B_Q8Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("gemma-4-E4B-it-Q8_0.gguf", "q8_0", LowEntropyPrompt, eosToken: null, maxAbsTol: 2.5f);

    /// <summary>Gemma 4 12B QAT Q4_0: the driving model — attention_k_eq_v globals + q8_0 KV.</summary>
    [Fact]
    public void Gemma4_12B_Q8Kv_ArgmaxStable_VsFp32()
        => AssertKvParity("gemma-4-12b-it-qat-q4_0.gguf", "q8_0", LowEntropyPrompt, eosToken: null, maxAbsTol: 10.0f);

    /// <summary>Qwen3-8B Q4_K: q8_0 global batched prefill (AttentionBatchedQ8_0).</summary>
    [Fact]
    public void Qwen3_8B_Q8BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("Qwen3-8B-Q4_K_M.gguf", "q8_0", LowEntropyPrompt, maxAbsTol: 2.5f);

    /// <summary>Gemma 4 E4B Q8_0: q8_0 SWA + global batched prefill (AttentionSwaBatchedQ8_0).</summary>
    [Fact]
    public void Gemma4_E4B_Q8BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("gemma-4-E4B-it-Q8_0.gguf", "q8_0", LowEntropyPrompt, maxAbsTol: 2.5f);

    /// <summary>Gemma 4 12B QAT Q4_0: q8_0 batched prefill with k_eq_v globals.</summary>
    [Fact]
    public void Gemma4_12B_Q8BatchedPrefill_MatchesPerToken()
        => AssertKvBatchedPrefillParity("gemma-4-12b-it-qat-q4_0.gguf", "q8_0", LowEntropyPrompt, maxAbsTol: 10.0f);

    /// <summary>Gemma 4 E4B Q8_0: q8_0 Tc2-flash chunked prefill across the SWA ring boundary.</summary>
    [Fact]
    public void Gemma4_E4B_Q8ChunkedPrefill_MatchesFp32()
        => AssertKvChunkedPrefillParity("gemma-4-E4B-it-Q8_0.gguf", "q8_0", maxAbsCeiling: 30.0f);

    /// <summary>
    /// Gemma 4 12B QAT Q4_0: the 128K driving model — q8_0 Tc2-flash chunked prefill past
    /// 4096 over its attention_k_eq_v global layers (V reuses K storage). The other tests
    /// cross the chunk/SWA-ring boundary only on E4B, which has no k_eq_v; this is the one
    /// long-context test that exercises the headline config (12B + q8_0 + >4096) end to end.
    /// Wider blow-up ceiling: Q4_0 weights over 48 layers + q8_0 store accumulate more over
    /// a 5000-token context; top-5 stability is the real argmax-stable gate.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q8ChunkedPrefill_MatchesFp32()
        => AssertKvChunkedPrefillParity("gemma-4-12b-it-qat-q4_0.gguf", "q8_0", maxAbsCeiling: 45.0f);

    // ── Issue #185 item 1: auto-narrow KV dtype decision ─────────────────────
    // The decision is factored into the pure CudaForwardPass.ResolveKvDType /
    // EstimateKvCacheBytes / Q8KvGeometrySupported helpers so it's unit-testable
    // without a GPU or a model on disk. These pin the precedence rule: an oversized
    // context narrows (bf16 preferred, q8_0 if bf16 still won't fit and the geometry
    // allows), but an explicit operator choice (or a TQ run) is NEVER overridden.

    /// <summary>fp32 fits → keep fp32, no narrowing.</summary>
    [Fact]
    public void AutoNarrow_KeepsFp32_WhenItFits()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: false, tqEnabled: false,
            availableKvBytes: 1000, fp32KvBytes: 800, bf16KvBytes: 400, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.Float32, dt);
        Assert.False(narrowed);
    }

    /// <summary>fp32 too big, bf16 fits → bf16 (preferred over the coarser q8_0).</summary>
    [Fact]
    public void AutoNarrow_PicksBf16_WhenFp32TooBigButBf16Fits()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: false, tqEnabled: false,
            availableKvBytes: 600, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.BFloat16, dt);
        Assert.True(narrowed);
    }

    /// <summary>Both fp32 and bf16 too big, geometry supports q8_0 → q8_0 (narrowest).</summary>
    [Fact]
    public void AutoNarrow_PicksQ8_WhenBf16TooBig_AndGeometrySupported()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: false, tqEnabled: false,
            availableKvBytes: 300, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.Q8_0, dt);
        Assert.True(narrowed);
    }

    /// <summary>
    /// bf16 too big and q8_0 geometry unsupported (some layer kvDim not %32) → bf16
    /// best-effort: the only narrowed store valid for any geometry. Still flagged as
    /// narrowed even though it may not fit (the alloc then fails loudly, halved vs fp32).
    /// </summary>
    [Fact]
    public void AutoNarrow_FallsToBf16_WhenBf16TooBig_ButQ8Unsupported()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: false, tqEnabled: false,
            availableKvBytes: 300, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: false,
            out bool narrowed);
        Assert.Equal(DType.BFloat16, dt);
        Assert.True(narrowed);
    }

    /// <summary>Explicit fp32 that does NOT fit is kept — errors loudly later, never silently narrowed.</summary>
    [Fact]
    public void AutoNarrow_NeverOverrides_ExplicitFp32()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: true, tqEnabled: false,
            availableKvBytes: 100, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.Float32, dt);
        Assert.False(narrowed);
    }

    /// <summary>An explicit narrowed request is returned unchanged regardless of fit.</summary>
    [Fact]
    public void AutoNarrow_NeverOverrides_ExplicitBf16()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.BFloat16, explicitChoice: true, tqEnabled: false,
            availableKvBytes: 100, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.BFloat16, dt);
        Assert.False(narrowed);
    }

    /// <summary>TQ owns its own quantized KV ring — auto-narrow stands down even at fp32.</summary>
    [Fact]
    public void AutoNarrow_SkipsWhenTqEnabled()
    {
        var dt = CudaForwardPass.ResolveKvDType(
            DType.Float32, explicitChoice: false, tqEnabled: true,
            availableKvBytes: 100, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed);
        Assert.Equal(DType.Float32, dt);
        Assert.False(narrowed);
    }

    /// <summary>A flat (non-gemma) ModelHyperparams for the byte/geometry estimators.</summary>
    private static ModelHyperparams FlatHp(int numLayers, int numKvHeads, int headDim, int ctx = 4096)
        => new()
        {
            NumLayers = numLayers,
            NumHeads = numKvHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            ContextLength = ctx,
            VocabSize = 1000,
            EmbeddingDim = numKvHeads * headDim,
            IntermediateDim = numKvHeads * headDim * 2,
        };

    /// <summary>
    /// EstimateKvCacheBytes orders by element width (q8_0 &lt; bf16 &lt; fp32) AND accounts
    /// for the per-buffer power-of-two pool rounding the ctor's gpu.Allocate applies. Dims
    /// are chosen so each dtype's per-buffer raw size lands in a distinct power-of-two
    /// bucket: kvDim 1536 × ctx 2048 = 3,145,728 elements/buffer.
    ///   fp32 = 12,582,912 B → rounds up to 16,777,216 (2^24)
    ///   bf16 =  6,291,456 B → rounds up to  8,388,608 (2^23)
    ///   q8_0 =  3,342,336 B → rounds up to  4,194,304 (2^22)
    /// 2 layers × 2 (K+V) = 4 buffers each.
    /// </summary>
    [Fact]
    public void EstimateKvCacheBytes_RoundsEachBufferToPoolBucket()
    {
        var hp = FlatHp(numLayers: 2, numKvHeads: 12, headDim: 128); // kvDim = 1536 (%32 == 0)
        const int ctx = 2048;
        long fp32 = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32);
        long bf16 = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.BFloat16);
        long q8   = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Q8_0);

        Assert.Equal(4L * 16_777_216, fp32);
        Assert.Equal(4L * 8_388_608, bf16);
        Assert.Equal(4L * 4_194_304, q8);
        Assert.True(q8 < bf16 && bf16 < fp32);

        // The rounding is real, not a no-op: q8_0's raw footprint (34 B / 32-elem block)
        // never lands on a power of two, so the pooled allocation is strictly larger than
        // the raw byte sum — the undercount the estimate must avoid.
        long q8Raw = 4L * (3_145_728 / 32 * 34);
        Assert.True(q8 > q8Raw, "q8_0 estimate must include the per-buffer pool rounding.");
    }

    /// <summary>Q8KvGeometrySupported is true only when every layer's kvDim is a multiple of 32.</summary>
    [Fact]
    public void Q8KvGeometry_RequiresKvDimMultipleOf32()
    {
        // kvDim = 8 × 128 = 1024 → %32 == 0 → supported.
        Assert.True(CudaForwardPass.Q8KvGeometrySupported(FlatHp(4, 8, 128)));
        // kvDim = 1 × 48 = 48 → 48 % 32 == 16 → unsupported.
        Assert.False(CudaForwardPass.Q8KvGeometrySupported(FlatHp(4, 1, 48)));
    }
}
