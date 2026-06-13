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

    // ── Issue #191: narrowed-KV GREEDY decode coherence (template-correct) ───
    // The parity tests above teacher-force the narrowed dtype onto fp32's trajectory, so
    // they never let bf16/q8_0 pick their OWN greedy tokens — exactly the path a real 12 GB
    // user hits (#185 auto-narrows to bf16 by default) and the one #188 found degenerate on
    // a synthetic OOD prompt. These decode the narrowed path GREEDILY on a TEMPLATE-CORRECT
    // prompt (turn-structured, healthy top-logit margin per the 'prompt must match chat
    // template' lesson) and assert coherence (≥2 distinct, non-EOS, finite). The fp32
    // synthetic-prompt tests in Gemma4Cuda12BForwardPassTests stay as the trunk-math guards.

    // Gemma 4 turn format: <bos> is added by the tokenizer (add_bos=true); the control tokens
    // encode as singletons. An open-ended instruction elicits a multi-token answer (a factual
    // prompt makes the 12B-IT emit a 1-token <end_of_turn>, which is why the fp32 guard uses a
    // synthetic prompt instead — see Gemma4Cuda12BForwardPassTests).
    private const string Gemma4TemplatePrompt =
        "<start_of_turn>user\nWrite a short sentence about the ocean.<end_of_turn>\n<start_of_turn>model\n";

    // Qwen3 ChatML. Thinking is left on (the template auto-opens <think>), which only makes the
    // continuation longer/more varied — fine for a coherence check.
    private const string Qwen3TemplatePrompt =
        "<|im_start|>user\nWrite a short sentence about the ocean.<|im_end|>\n<|im_start|>assistant\n";

    private static int ReadEosId(string path)
    {
        using var model = GgufModel.Open(path);
        return GgufTokenizer.FromGgufModel(model).EosTokenId;
    }

    /// <summary>
    /// Prefill a template-correct prompt and GREEDILY decode (the narrowed path picks its own
    /// tokens — not teacher-forced) on the given KV dtype, asserting the decode is coherent:
    /// all logits finite, the first generated token is not EOS, and ≥2 distinct tokens over the
    /// run (an all-one-token repetition or all-EOS collapse — the #188 failure mode — fails).
    /// </summary>
    private static void AssertGreedyCoherence(
        string filename, string kvDtype, string prompt, int ctx = 2048, bool batchedPrefill = false)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        int eosId = ReadEosId(path);
        const int steps = 6;
        var (logits, argmax) = RunPrefillDecode(
            gpu, path, kvDtype, prompt, steps, ctx, forced: null, batchedPrefill: batchedPrefill);

        for (int p = 0; p <= steps; p++)
            for (int i = 0; i < logits[p].Length; i++)
                Assert.True(float.IsFinite(logits[p][i]),
                    $"{filename} {kvDtype}: non-finite logit at pos {p}, idx {i} — a narrowed-KV (SWA-ring / k_eq_v) bug.");

        Assert.True(argmax[0] != eosId,
            $"{filename} {kvDtype}: first greedy token was EOS — the template-correct prompt should have a real " +
            "continuation, so this means the narrowed greedy path collapsed.");

        var seen = new HashSet<int>(argmax);
        Assert.True(seen.Count >= 2,
            $"{filename} {kvDtype}: greedy decode produced only {seen.Count} distinct token(s) " +
            $"([{string.Join(",", argmax)}]) — narrowed-KV greedy decode is degenerate.");
    }

    /// <summary>Qwen3-8B Q4_K (non-SWA dense) bf16 KV: greedy decode stays coherent.</summary>
    [Fact]
    public void Qwen3_8B_Bf16Kv_GreedyDecode_Coherent()
        => AssertGreedyCoherence("Qwen3-8B-Q4_K_M.gguf", "bf16", Qwen3TemplatePrompt);

    /// <summary>Qwen3-8B Q4_K q8_0 KV: greedy decode stays coherent.</summary>
    [Fact]
    public void Qwen3_8B_Q8Kv_GreedyDecode_Coherent()
        => AssertGreedyCoherence("Qwen3-8B-Q4_K_M.gguf", "q8_0", Qwen3TemplatePrompt);

    /// <summary>
    /// Gemma 4 12B QAT bf16 KV: the headline 12 GB-card config (auto-narrows to bf16 by
    /// default). The fp32 synthetic-prompt guard can't run bf16 greedily — this is the only
    /// coverage that the default narrowed dtype decodes coherently on a real prompt.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Bf16Kv_GreedyDecode_Coherent()
        => AssertGreedyCoherence("gemma-4-12b-it-qat-q4_0.gguf", "bf16", Gemma4TemplatePrompt);

    /// <summary>Gemma 4 12B QAT q8_0 KV: greedy decode stays coherent.</summary>
    [Fact]
    public void Gemma4_12B_Q8Kv_GreedyDecode_Coherent()
        => AssertGreedyCoherence("gemma-4-12b-it-qat-q4_0.gguf", "q8_0", Gemma4TemplatePrompt);

    /// <summary>
    /// Issue #166 (latent half): bf16 KV with the SWA ring WRAPPED, then a GREEDY decode. E4B
    /// has sliding_window=512, so the ring is min(ctx, 512+4096=4608); a >4608-token prefill
    /// wraps it (the batched bf16 append overwrites earlier slots), and the subsequent greedy
    /// decode keeps writing wrapped slots via the single-token bf16 append — the decode path
    /// the existing chunked-prefill parity test only exercises teacher-forced. A wrapped-ring
    /// OOB read/write would surface as NaN or a degenerate single-token collapse here.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Bf16Kv_GreedyDecodePastSwaRingWrap_Coherent()
    {
        // AssertGreedyCoherence creates its own backend; just gate the prompt build here.
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath("gemma-4-E4B-it-Q8_0.gguf");
        if (path is null) return;

        // Build a >4608-token prompt so the 512-window SWA ring (4608 slots) actually wraps.
        string prompt;
        using (var model = GgufModel.Open(path))
        {
            var tok = GgufTokenizer.FromGgufModel(model);
            var sb = new System.Text.StringBuilder();
            const string seed = "The quick brown fox jumps over the lazy dog. " +
                                "Sphinx of black quartz, judge my vow. " +
                                "Pack my box with five dozen liquor jugs. ";
            while (tok.Encode(sb.ToString()).Count < 5000) sb.Append(seed);
            prompt = sb.ToString();
        }

        // ctx 6144 > ring (4608) so the wrap is real; batched prefill drives the chunked
        // append, then RunPrefillDecode greedily decodes past the wrap.
        AssertGreedyCoherence("gemma-4-E4B-it-Q8_0.gguf", "bf16", prompt, ctx: 6144, batchedPrefill: true);
    }

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

    /// <summary>
    /// A gemma4-shaped ModelHyperparams: per-layer head_dim / kv-head counts, an SWA layer,
    /// and a KV-share (aliased) tail layer. Exercises the per-layer branches of
    /// EstimateKvCacheBytes / Q8KvGeometrySupported that the flat model never reaches — the
    /// branches that matter for the #185 driving models (12B/E4B).
    /// </summary>
    private static ModelHyperparams Gemma4ShapedHp(
        int[] layerHeadDim, int[] layerKvHeads, bool[] isSwa, int[] kvSource, int slidingWindow)
        => new()
        {
            NumLayers = layerHeadDim.Length,
            NumHeads = 8,
            NumKvHeads = 8,
            HeadDim = layerHeadDim[0],
            ContextLength = 131072,
            VocabSize = 1000,
            EmbeddingDim = 2048,
            IntermediateDim = 4096,
            SlidingWindowSize = slidingWindow,
            LayerHeadDim = layerHeadDim,
            LayerKvHeads = layerKvHeads,
            IsSwaLayer = isSwa,
            KvSourceLayer = kvSource,
        };

    /// <summary>
    /// EstimateKvCacheBytes mirrors the gemma4 per-layer allocation: a KV-share (aliased)
    /// layer allocates nothing, an SWA layer is capped at the window-ring size (not maxCtx),
    /// and a global layer uses full maxCtx — each with its own head_dim / kv-head count.
    /// </summary>
    [Fact]
    public void EstimateKvCacheBytes_Gemma4_AliasedSkipped_SwaRingCapped()
    {
        // 3 layers: [0]=global (256 hd, 8 kv), [1]=SWA (256 hd, 8 kv), [2]=KV-share aliasing 0.
        const int ctx = 65536;
        const int window = 1024;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],   // layer 2 aliases layer 0 → no own pages
            slidingWindow: window);

        long bytes = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.BFloat16);

        // Hand-compute against the same per-buffer power-of-two rounding the helper applies.
        // SwaRingHeadroom is large (>= 4096); the helper caps the ring at min(maxCtx, window+headroom).
        int swaRing = SwaRingSizeForTest(ctx, window);
        long kvDim = 8L * 256;                                  // 2048
        long globalBuf = RoundUpPow2(kvDim * ctx * 2);          // bf16 = 2 B/elem
        long swaBuf    = RoundUpPow2(kvDim * swaRing * 2);
        long expected  = 2 * globalBuf + 2 * swaBuf;            // layer0 (K+V) + layer1 (K+V); layer2 aliased
        Assert.Equal(expected, bytes);
    }

    // ── Issues #220 / #215: gpuLayers scoping + per-layer-KV ctx solver ──────
    // EstimateKvCacheBytes gained a final gpuLayers param: when < 0 it sums ALL
    // layers (old behavior); when >= 0 it sums only the first gpuLayers (clamped to
    // NumLayers). TierPlanner.SolveGpuCtxForPerLayerKv binary-searches the largest
    // context whose scoped KV allocation fits the VRAM budget — the #220 contract is
    // that the chosen context never UNDER-reserves (the true allocation must fit).

    /// <summary>
    /// gpuLayers scopes the sum to the first N layers. On a flat/uniform model every
    /// layer is identical, so the first 2 of 4 is exactly half of all 4; -1 / NumLayers /
    /// an over-cap value / the no-arg default all mean "all layers"; 0 means nothing.
    /// </summary>
    [Fact]
    public void EstimateKvCacheBytes_GpuLayersScoping_SumsOnlyFirstN()
    {
        var hp = FlatHp(numLayers: 4, numKvHeads: 8, headDim: 128);
        const int ctx = 2048;

        long all = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32, gpuLayers: 4);
        long firstTwo = CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32, gpuLayers: 2);
        Assert.Equal(all / 2, firstTwo);            // uniform model → first 2 == 2/4 of all 4

        // -1 (default), == NumLayers, and an over-cap value (clamped to NumLayers) all mean "all".
        Assert.Equal(all, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32, gpuLayers: -1));
        Assert.Equal(all, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32, gpuLayers: 100));
        Assert.Equal(all, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32)); // no-arg == -1

        // gpuLayers 0 → no layers summed.
        Assert.Equal(0L, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.Float32, gpuLayers: 0));
    }

    /// <summary>
    /// gpuLayers scoping over the gemma4 per-layer shape: the scoped prefix still honors
    /// SWA-ring capping and KV-share aliasing. layer0 = global, layer1 = SWA, layer2 aliases
    /// layer0 (allocates nothing). So gpuLayers 1 = layer0 only; gpuLayers 2 = layer0+layer1;
    /// gpuLayers 3 (== all) is identical to 2 because the aliased tail contributes 0.
    /// </summary>
    [Fact]
    public void EstimateKvCacheBytes_GpuLayersScoping_Gemma4_SkipsAliasedAndScopes()
    {
        const int ctx = 65536;
        const int window = 1024;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],   // layer 2 aliases layer 0 → no own pages
            slidingWindow: window);

        int swaRing = SwaRingSizeForTest(ctx, window);
        long kvDim = 8L * 256;                          // 2048
        long globalBuf = RoundUpPow2(kvDim * ctx * 2);  // bf16 = 2 B/elem
        long swaBuf = RoundUpPow2(kvDim * swaRing * 2);

        long expected1 = 2 * globalBuf;                 // layer0 only (K+V)
        long expected2 = 2 * globalBuf + 2 * swaBuf;    // layer0 + layer1 (SWA)
        // layer2 is aliased → 0, so all (3) == first 2.
        Assert.Equal(expected1, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.BFloat16, gpuLayers: 1));
        Assert.Equal(expected2, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.BFloat16, gpuLayers: 2));
        Assert.Equal(expected2, CudaForwardPass.EstimateKvCacheBytes(hp, ctx, DType.BFloat16, gpuLayers: 3));
    }

    /// <summary>
    /// SolveGpuCtxForPerLayerKv returns the LARGEST context whose scoped KV allocation fits
    /// the budget (#220). Sizing the budget to exactly a reference context's footprint, the
    /// solver must (a) fit — never under-reserve, (b) return at least that reference context
    /// (it fits), and (c) be maximal — the next context up overflows the budget (unless the
    /// auto cap is hit first).
    /// </summary>
    [Fact]
    public void SolveGpuCtxForPerLayerKv_ReturnsLargestFittingContext()
    {
        const int gpuLayers = 3;
        const int autoCtxCap = 32768;
        const int refCtx = 8192;
        var dtype = DType.BFloat16;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],
            slidingWindow: 1024);

        long budget = CudaForwardPass.EstimateKvCacheBytes(hp, refCtx, dtype, gpuLayers);
        int got = TierPlanner.SolveGpuCtxForPerLayerKv(hp, autoCtxCap, budget, dtype, gpuLayers);

        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, got, dtype, gpuLayers) <= budget,
            $"solved ctx {got} over-reserves vs budget {budget} — would OOM at runtime (#220).");
        Assert.True(got >= refCtx,
            $"solved ctx {got} below the reference {refCtx}, which fits exactly — not maximal.");
        Assert.True(got == autoCtxCap ||
            CudaForwardPass.EstimateKvCacheBytes(hp, got + 1, dtype, gpuLayers) > budget,
            $"ctx {got + 1} also fits budget {budget} — solver did not return the LARGEST fitting context.");
    }

    /// <summary>
    /// SolveGpuCtxForPerLayerKv floors at 512: a budget too small for even 512-ctx returns
    /// the floor (the alloc then fails loudly rather than silently picking a smaller, unusable
    /// context), and an autoCtxCap below the floor clamps to the cap (Math.Min(512, cap)).
    /// </summary>
    [Fact]
    public void SolveGpuCtxForPerLayerKv_BudgetBelowFloor_ReturnsFloor()
    {
        var dtype = DType.BFloat16;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],
            slidingWindow: 1024);

        // 1 byte can't hold even a 512-ctx cache → floor (512).
        Assert.Equal(512, TierPlanner.SolveGpuCtxForPerLayerKv(hp, autoCtxCap: 32768, vramBudget: 1, dtype, gpuLayers: 3));

        // autoCtxCap below the floor → clamp to the cap (Math.Min(512, cap)), even with a huge budget.
        Assert.Equal(256, TierPlanner.SolveGpuCtxForPerLayerKv(
            hp, autoCtxCap: 256, vramBudget: long.MaxValue, dtype, gpuLayers: 3));
    }

    // ── Issue #220: dense auto-context is dtype-aware (CudaForwardPass.SolveMaxCtxForKv) ──
    // The full-GPU Gemma path's auto-context comes from EstimateMaxContext → SolveMaxCtxForKv,
    // which previously priced fp32 unconditionally, so --kv-type bf16/q8_0 bought NO extra
    // context (observed 1770 flat across all three dtypes on a 12 GB card). The fix binary-
    // searches the largest ctx whose EstimateKvCacheBytes(.., kvDType) fits — the same
    // allocator-exact arithmetic the ctor reserves — so narrowed KV expands the window. These
    // pin: (1) the dtype response in a linear all-global regime (clean ratios), (2) the
    // allocator-maximal contract on the SWA shape, (3) uniform-attention models are UNCHANGED.

    /// <summary>
    /// (1) Dtype response, linear regime. An all-global per-layer model (LayerHeadDim set,
    /// IsSwaLayer all false → the per-layer binary-search branch, but no SWA-ring capping) is
    /// linear in ctx, so a budget sized to exactly fit 4096-ctx fp32 yields a clean dtype
    /// progression: bf16 doubles the ctx (half the per-element width), q8_0 ~3.76× it (its
    /// 34-byte/32-elem blocks fall short of a clean 4×). Dims chosen so fp32/bf16 buffers land
    /// exactly on power-of-two pool buckets (kvDim 1024 × power-of-two ctx), isolating the ratio
    /// from round-up noise.
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_RespondsToDtype_LinearRegime()
    {
        var hp = Gemma4ShapedHp(
            layerHeadDim: [128, 128, 128, 128],
            layerKvHeads: [8, 8, 8, 8],          // kvDim = 1024
            isSwa: [false, false, false, false], // all global → linear in ctx, no SWA cap
            kvSource: [-1, -1, -1, -1],
            slidingWindow: 4096);                // irrelevant (no SWA layer)

        // Budget = exactly the fp32 footprint at 4096 ctx (each [4096×1024] fp32 buffer = 2^24).
        long budget = CudaForwardPass.EstimateKvCacheBytes(hp, 4096, DType.Float32);

        int fp32 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Float32);
        int bf16 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.BFloat16);
        int q8   = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Q8_0);

        Assert.Equal(4096, fp32);
        Assert.Equal(8192, bf16);                 // exactly 2× — half the element width
        Assert.Equal(2 * fp32, bf16);
        Assert.True(q8 > bf16 && q8 >= 15000,     // ~3.76× (q8_0's 34/32 block overhead < clean 4×)
            $"q8_0 ctx {q8} should be well past 2× fp32 ({fp32}) — narrowed KV must expand the window.");
        // Allocator-exact: the chosen q8_0 ctx fits the budget and the next step up does not.
        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, q8, DType.Q8_0) <= budget);
        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, q8 + 1, DType.Q8_0) > budget);
    }

    /// <summary>
    /// (2) Allocator-maximal on the SWA shape (the #220 contract). Sizing the budget to a
    /// reference context's bf16 footprint, the solver must fit (never under-reserve → no
    /// runtime OOM), return at least the reference, and be maximal (the next ctx overflows).
    /// Uses the gemma4 shape (per-layer head_dim + SWA ring + KV-share aliasing) so the search
    /// runs against the real per-layer allocator arithmetic.
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_SwaShape_IsAllocatorMaximal()
    {
        const int refCtx = 8192;
        var dtype = DType.BFloat16;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],   // layer 2 aliases layer 0
            slidingWindow: 1024);

        long budget = CudaForwardPass.EstimateKvCacheBytes(hp, refCtx, dtype);
        int got = CudaForwardPass.SolveMaxCtxForKv(hp, budget, dtype);

        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, got, dtype) <= budget,
            $"solved ctx {got} over-reserves vs budget {budget} — would OOM at runtime (#220).");
        Assert.True(got >= refCtx, $"solved ctx {got} below the reference {refCtx}, which fits exactly.");
        Assert.True(got == hp.ContextLength ||
            CudaForwardPass.EstimateKvCacheBytes(hp, got + 1, dtype) > budget,
            $"ctx {got + 1} also fits budget {budget} — solver did not return the LARGEST fitting context.");
    }

    /// <summary>
    /// (3) Uniform-attention models are UNCHANGED: a flat model (no LayerHeadDim / IsSwaLayer)
    /// keeps the fp32 formula regardless of the requested KV dtype, so bf16/q8_0 do NOT alter
    /// its auto-context. This is the #220 acceptance "no change for uniform-attention models"
    /// guard — the dtype-aware sizing is scoped to the SWA/per-layer Gemma path only.
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_UniformModel_IgnoresDtype()
    {
        var hp = FlatHp(numLayers: 8, numKvHeads: 8, headDim: 128, ctx: 131072);
        long budget = 256L * 1024 * 1024;
        int fp32 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Float32);
        int bf16 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.BFloat16);
        int q8   = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Q8_0);
        Assert.True(fp32 > 512, $"sanity: expected a mid-range ctx, got {fp32}.");
        Assert.Equal(fp32, bf16);   // dtype ignored for uniform models — no change vs pre-#220
        Assert.Equal(fp32, q8);
    }

    /// <summary>
    /// (1b) The headline #220 claim: once the context clears the SWA ring cap, the SWA layers
    /// stop growing, so a narrower KV dtype's freed budget flows entirely into the (few) global
    /// layers — gaining MORE than the bare width ratio (2×/4×). Shape: 1 global + 5 SWA layers
    /// with a 512 window (ring = 4608); the budget is sized to an fp32 context (8192) already
    /// PAST that ring, so all SWA layers are capped for every dtype. Relational asserts (not
    /// magic numbers) so the test is robust to the exact pow2-bucket arithmetic.
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_SwaSaturation_DtypeGainExceedsWidthRatio()
    {
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256, 256, 256, 256],
            layerKvHeads: [8, 8, 8, 8, 8, 8],                 // kvDim = 2048
            isSwa: [false, true, true, true, true, true],      // 1 global, 5 SWA → SWA dominates
            kvSource: [-1, -1, -1, -1, -1, -1],
            slidingWindow: 512);                               // ring = min(ctx, 512+4096) = 4608

        // Budget = fp32 footprint at ctx 8192 (> 4608 ring → SWA layers capped for all dtypes).
        long budget = CudaForwardPass.EstimateKvCacheBytes(hp, 8192, DType.Float32);

        int fp32 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Float32);
        int bf16 = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.BFloat16);
        int q8   = CudaForwardPass.SolveMaxCtxForKv(hp, budget, DType.Q8_0);

        Assert.True(q8 > bf16 && bf16 > fp32, $"monotonic dtype response expected; got fp32={fp32} bf16={bf16} q8={q8}.");
        // Super-linear: past the SWA cap only the global layers grow, so narrowing beats the
        // width ratio. (Asserting strict > the ratio, with margin from the 5:1 SWA:global mix.)
        Assert.True(bf16 > 2 * fp32, $"bf16 ctx {bf16} should exceed 2× fp32 ({fp32}) once SWA layers are capped.");
        Assert.True(q8 > 4 * fp32, $"q8_0 ctx {q8} should exceed 4× fp32 ({fp32}) once SWA layers are capped.");
        // Allocator-maximal for the narrowest dtype (fits + next step overflows or hits the cap).
        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, q8, DType.Q8_0) <= budget);
        Assert.True(q8 == hp.ContextLength ||
            CudaForwardPass.EstimateKvCacheBytes(hp, q8 + 1, DType.Q8_0) > budget);
    }

    /// <summary>
    /// (1c) Floor and cap clamps on the per-layer branch (distinct from the sibling
    /// SolveGpuCtxForPerLayerKv floor test — different signature: the cap here is hp.ContextLength).
    /// A budget too small for even 512-ctx returns the floor (the alloc then fails loudly, not a
    /// silently-smaller context); a model whose ContextLength is below 512 clamps to that; and a
    /// huge budget clamps UP to the model max (not unbounded).
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_ClampsToFloorAndModelMax()
    {
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],
            slidingWindow: 1024);

        // Budget = 1 byte → can't hold even a 512-ctx cache → floor (512).
        Assert.Equal(512, CudaForwardPass.SolveMaxCtxForKv(hp, 1, DType.BFloat16));

        // ContextLength below the floor → clamp to the cap even with a huge budget (must NOT throw
        // — the SWA branch and the uniform branch both route through the cap<=floor guard, else the
        // uniform Math.Clamp(_, 512, cap) would throw ArgumentException for cap < 512).
        var tinyCapSwa = hp with { ContextLength = 256 };
        Assert.Equal(256, CudaForwardPass.SolveMaxCtxForKv(tinyCapSwa, long.MaxValue, DType.BFloat16));
        var tinyCapUniform = FlatHp(numLayers: 8, numKvHeads: 8, headDim: 128, ctx: 256);
        Assert.Equal(256, CudaForwardPass.SolveMaxCtxForKv(tinyCapUniform, long.MaxValue, DType.Float32));

        // Huge budget → clamp UP to the model max, not beyond.
        Assert.Equal(hp.ContextLength, CudaForwardPass.SolveMaxCtxForKv(hp, long.MaxValue, DType.Q8_0));
    }

    /// <summary>
    /// (1d) Mixed per-layer head_dim: the solver must price each layer at its own head_dim (via
    /// EstimateKvCacheBytes), not collapse to hp.HeadDim or layer 0. A shape with distinct
    /// per-layer dims, asserted allocator-maximal (the contract that depends on the per-layer
    /// arithmetic being exact).
    /// </summary>
    [Fact]
    public void SolveMaxCtxForKv_MixedPerLayerHeadDim_IsAllocatorMaximal()
    {
        const int refCtx = 8192;
        var dtype = DType.Q8_0;
        var hp = Gemma4ShapedHp(
            layerHeadDim: [256, 128, 256, 128],   // mixed per-layer head_dim
            layerKvHeads: [8, 8, 8, 8],
            isSwa: [false, true, false, true],
            kvSource: [-1, -1, -1, -1],
            slidingWindow: 1024);

        long budget = CudaForwardPass.EstimateKvCacheBytes(hp, refCtx, dtype);
        int got = CudaForwardPass.SolveMaxCtxForKv(hp, budget, dtype);

        Assert.True(CudaForwardPass.EstimateKvCacheBytes(hp, got, dtype) <= budget,
            $"mixed-headdim solved ctx {got} over-reserves vs budget {budget}.");
        Assert.True(got >= refCtx, $"solved ctx {got} below the reference {refCtx}, which fits exactly.");
        Assert.True(got == hp.ContextLength ||
            CudaForwardPass.EstimateKvCacheBytes(hp, got + 1, dtype) > budget,
            $"ctx {got + 1} also fits — not the largest fitting context for the mixed-headdim shape.");
    }

    // ── Issue #228: KvVramReserveBytes — bounded reserve for SWA, unchanged for dense ──
    // The reserve held back from the KV budget. Dense models keep the proven
    // max(VRAM/3, 2GB) (guards the #185 spill cliff — their KV fills any budget). SWA/Gemma
    // models use a bounded system allowance + prefill working set, safe because SWA KV
    // saturates past the ring (a bigger budget can't grow KV to eat the headroom).

    /// <summary>Dense (no IsSwaLayer) keeps the exact max(VRAM/3, 2 GiB) reserve — unchanged.</summary>
    [Fact]
    public void KvVramReserveBytes_DenseModel_KeepsMaxVramThirdOr2GiB()
    {
        var hp = FlatHp(numLayers: 32, numKvHeads: 8, headDim: 128);   // IsSwaLayer == null
        const long GiB = 1024L * 1024 * 1024;
        // 12 GiB → VRAM/3 (4 GiB) wins the max.
        Assert.Equal(4 * GiB, CudaForwardPass.KvVramReserveBytes(hp, 12 * GiB));
        // 3 GiB → the 2 GiB floor wins.
        Assert.Equal(2 * GiB, CudaForwardPass.KvVramReserveBytes(hp, 3 * GiB));
        // 48 GiB → VRAM/3 (16 GiB) — the dense reserve scales with total VRAM (the #228 complaint,
        // intentionally preserved for dense where KV grows linearly to fill the budget).
        Assert.Equal(16 * GiB, CudaForwardPass.KvVramReserveBytes(hp, 48 * GiB));
    }

    /// <summary>
    /// SWA/Gemma reserve is bounded BELOW the dense reserve at the same VRAM (the #228 win):
    /// a fixed system allowance (≥ 2 GiB floor) plus a positive prefill working set, never the
    /// VRAM/3 fraction. On a 12 GiB card it's well under the dense 4 GiB.
    /// </summary>
    [Fact]
    public void KvVramReserveBytes_SwaModel_BoundedBelowDense()
    {
        const long GiB = 1024L * 1024 * 1024;
        var swa = Gemma4ShapedHp(
            layerHeadDim: [256, 256, 256],
            layerKvHeads: [8, 8, 8],
            isSwa: [false, true, false],
            kvSource: [-1, -1, 0],
            slidingWindow: 1024);
        var dense = FlatHp(numLayers: 3, numKvHeads: 8, headDim: 256);

        long swaReserve = CudaForwardPass.KvVramReserveBytes(swa, 12 * GiB);
        long denseReserve = CudaForwardPass.KvVramReserveBytes(dense, 12 * GiB);

        Assert.True(swaReserve > 2 * GiB, "SWA reserve must add a prefill working set on top of the 2 GiB floor.");
        Assert.True(swaReserve < denseReserve,
            $"SWA reserve ({swaReserve}) must be below the dense VRAM/3 reserve ({denseReserve}) — the #228 win.");
        Assert.True(swaReserve < 4 * GiB, $"SWA reserve ({swaReserve}) should be well under the dense 4 GiB at 12 GiB VRAM.");
    }

    /// <summary>
    /// The SWA reserve's working-set term scales with MODEL width (not just VRAM): a wider model
    /// (larger intermediate dim) reserves more. Distinguishes it from a pure VRAM fraction.
    /// </summary>
    [Fact]
    public void KvVramReserveBytes_SwaModel_ScalesWithModelWidth()
    {
        const long GiB = 1024L * 1024 * 1024;
        var narrow = Gemma4ShapedHp([256, 256], [8, 8], [false, true], [-1, -1], 1024);
        var wide = narrow with { IntermediateDim = narrow.IntermediateDim * 4 };

        long narrowReserve = CudaForwardPass.KvVramReserveBytes(narrow, 12 * GiB);
        long wideReserve = CudaForwardPass.KvVramReserveBytes(wide, 12 * GiB);
        Assert.True(wideReserve > narrowReserve,
            $"wider model (4× intermediate dim) should reserve more for its prefill working set " +
            $"(narrow={narrowReserve} wide={wideReserve}).");
    }

    /// <summary>
    /// The system-allowance term is max(2 GiB, VRAM/6): the 2 GiB floor wins at ≤ 12 GiB, but
    /// VRAM/6 takes over on larger cards. The working set is VRAM-independent, so the SWA reserve
    /// delta between a 48 GiB and a 12 GiB card is exactly the VRAM/6 delta (8 − 2 = 6 GiB) —
    /// pinning the only VRAM-scaling term (untested when every case uses 12 GiB → /6 == floor).
    /// </summary>
    [Fact]
    public void KvVramReserveBytes_SwaModel_SystemReserveScalesOnLargeCard()
    {
        const long GiB = 1024L * 1024 * 1024;
        var swa = Gemma4ShapedHp([256, 256, 256], [8, 8, 8], [false, true, false], [-1, -1, 0], 1024);
        long at12 = CudaForwardPass.KvVramReserveBytes(swa, 12 * GiB);
        long at48 = CudaForwardPass.KvVramReserveBytes(swa, 48 * GiB);
        // Both below their respective dense caps (workset is tiny here), so neither clamps.
        Assert.Equal(6 * GiB, at48 - at12);   // (48/6 − 12/6) GiB — the VRAM/6 term, isolated
    }

    /// <summary>
    /// The prefill working set must count the buffers EnsureBatchedTrunkScratch actually
    /// allocates: at the WIDEST per-layer head_dim (Gemma 4 12B's global 512, not the per-layer
    /// min), and — for a per-layer-token-embedding (PLE) model — the stacked PLE proj/row buffers
    /// (NumLayers × pleWidth). Dropping either (the original formula did both) under-reserves and
    /// risks a prefill-time OOM (#231 review). Exact-value pin so a silently-dropped term fails.
    /// </summary>
    [Fact]
    public void KvVramReserveBytes_SwaModel_CountsPleAndWidestHeadDim()
    {
        const long GiB = 1024L * 1024 * 1024;
        // Mixed head_dim (256 SWA / 512 global) + PLE. NumHeads/NumKvHeads = 8 (Gemma4ShapedHp).
        var hp = Gemma4ShapedHp([256, 512], [8, 8], [false, true], [-1, -1], 1024)
            with { HasPerLayerTokenEmbd = true, PerLayerEmbeddingWidth = 256 };

        const int chunk = 4096;          // PrefillBatchChunk
        const int maxHeadDim = 512;      // widest per-layer head_dim
        long perToken =
            hp.EmbeddingDim * 4L                          // hidden+residual+norm+PleY
            + 2L * hp.NumHeads * maxHeadDim               // Q + AttnOut (at maxHeadDim, not 256)
            + 2L * hp.NumKvHeads * maxHeadDim             // K + V
            + hp.IntermediateDim * 2L                     // FFN gate + up
            + 2L * hp.NumLayers * hp.PerLayerEmbeddingWidth + hp.PerLayerEmbeddingWidth; // PLE stack
        long expected = 2 * GiB + chunk * perToken * sizeof(float);   // systemReserve(2 GiB floor) + workset

        Assert.Equal(expected, CudaForwardPass.KvVramReserveBytes(hp, 12 * GiB));

        // And the PLE term is genuinely load-bearing: the same shape without PLE reserves strictly less.
        var noPle = hp with { HasPerLayerTokenEmbd = false };
        Assert.True(CudaForwardPass.KvVramReserveBytes(noPle, 12 * GiB) < expected,
            "dropping the PLE table must lower the reserve — the PLE stack is part of the prefill working set.");
    }

    /// <summary>
    /// Q8KvGeometrySupported returns false when ANY single (non-aliased) layer violates the
    /// %32 rule — not just when all do. A mixed set with one bad layer must fail, matching
    /// the ctor's per-layer throw (else auto-narrow would pick q8_0 and then crash).
    /// </summary>
    [Fact]
    public void Q8KvGeometry_FailsWhenAnySingleLayerViolates()
    {
        // layer 0 kvDim = 8×128 = 1024 (ok); layer 1 kvDim = 8×52 = 416 (416 % 32 == 0 → ok too).
        // Make layer 1 the violator: 1 kv-head × 52 hd = 52 → %32 == 20.
        var hp = Gemma4ShapedHp(
            layerHeadDim: [128, 52],
            layerKvHeads: [8, 1],
            isSwa: [false, false],
            kvSource: [-1, -1],
            slidingWindow: 4096);
        Assert.False(CudaForwardPass.Q8KvGeometrySupported(hp));

        // An aliased violator must be skipped (it owns no pages): layer 1 bad but aliases layer 0.
        var aliased = Gemma4ShapedHp(
            layerHeadDim: [128, 52],
            layerKvHeads: [8, 1],
            isSwa: [false, false],
            kvSource: [-1, 0],
            slidingWindow: 4096);
        Assert.True(CudaForwardPass.Q8KvGeometrySupported(aliased));
    }

    /// <summary>ResolveKvDType fit comparisons are inclusive (&lt;=): an exact-fit fp32/bf16 is kept, not narrowed past.</summary>
    [Fact]
    public void ResolveKvDType_FitBoundaryIsInclusive()
    {
        // fp32 exactly equals budget → keep fp32 (no narrow).
        var dt1 = CudaForwardPass.ResolveKvDType(
            DType.Float32, false, false,
            availableKvBytes: 1000, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed1);
        Assert.Equal(DType.Float32, dt1);
        Assert.False(narrowed1);

        // fp32 just over, bf16 exactly equals budget → pick bf16 (not fall to q8).
        var dt2 = CudaForwardPass.ResolveKvDType(
            DType.Float32, false, false,
            availableKvBytes: 500, fp32KvBytes: 1000, bf16KvBytes: 500, q8Supported: true,
            out bool narrowed2);
        Assert.Equal(DType.BFloat16, dt2);
        Assert.True(narrowed2);
    }

    // Local mirrors of the production rounding/SWA-ring math, so the gemma4 estimate test
    // computes its expectation independently rather than echoing the implementation.
    private const int SwaRingHeadroomForTest = 4096; // PrefillBatchChunk is 4096 by default
    private static int SwaRingSizeForTest(int maxSeqLen, int window)
        => (int)Math.Min(maxSeqLen, (long)window + SwaRingHeadroomForTest);
    private static long RoundUpPow2(long v)
    {
        if (v <= 64) return 64;
        ulong u = (ulong)(v - 1);
        u |= u >> 1; u |= u >> 2; u |= u >> 4; u |= u >> 8; u |= u >> 16; u |= u >> 32;
        return (long)(u + 1);
    }
}
