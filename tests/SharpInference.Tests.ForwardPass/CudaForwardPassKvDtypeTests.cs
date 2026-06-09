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

    private static void AssertBf16Parity(string filename, string prompt, int? eosToken, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        const int steps = 6;
        const int ctx = 2048;

        // fp32 reference first; teacher-force bf16 onto the SAME trajectory so each
        // decode position sees identical inputs and the KV dtype is the only variable.
        // Greedy-token equality alone is fragile on near-tie tokens (a borderline pair
        // flips on store rounding without any kernel bug), so parity is asserted at the
        // logit level — top-1 stable + small max-abs — per feedback_cross_backend_parity_test.
        var (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null);
        var (bf16, bf16Argmax) = RunPrefillDecode(gpu, path, "bf16", prompt, steps, ctx, forced: f32Argmax);

        // Coherence (feedback_forward_pass_tests): the fp32 reference must be a real
        // decode — IsFinite alone passes on a degenerate all-EOS run.
        Assert.True(eosToken is null || f32Argmax[0] != eosToken,
            $"{filename}: fp32 reference decoded EOS first — prompt put the model OOD, not a KV test.");

        // Per-position parity. Index 0 = prefill (full-prompt trunk over the whole KV
        // cache); 1.. = teacher-forced decode steps. Both runs saw identical inputs at
        // every position, so a faithful bf16 path differs only by accumulated store
        // rounding. We assert (a) logit max-abs is within the rounding budget — the
        // primary faithfulness measure — and (b) fp32's top-1 stays in bf16's top-5, a
        // reorder-tolerant "argmax-stable" check. We do NOT assert top-1 equality: a
        // genuine near-tie (e.g. the 12B's degenerate-repeat positions, where adjacent
        // token IDs sit within bf16 noise) can flip top-1 with no kernel bug.
        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(f32[p].Length, bf16[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32[p].Length; i++)
            {
                Assert.True(float.IsFinite(bf16[p][i]), $"{filename}: non-finite bf16 logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - bf16[p][i]));
            }
            Assert.True(maxAbs < maxAbsTol,
                $"{filename}: pos {p} bf16 vs fp32 logit max-abs diff {maxAbs:F3} exceeds the " +
                $"rounding budget ({maxAbsTol:F1}) — likely an arithmetic divergence (SWA-ring / k_eq_v / attn_scale).");
            Assert.True(TopK(bf16[p], 5).Contains(f32Argmax[p]),
                $"{filename}: pos {p} fp32 top-1 ({f32Argmax[p]}) fell out of bf16's top-5 " +
                $"(max-abs {maxAbs:F3}) — the bf16 path reordered the head of the distribution.");
        }
    }

    /// <summary>
    /// Increment 1.5: with bf16 KV, batched prefill must agree with the bf16 per-token
    /// prefill. This isolates the scalar batched bf16 kernels (KvAppendBatchedBf16,
    /// AttentionBatchedBf16, AttentionSwaBatchedBf16) + the batched dispatch: both runs
    /// store bf16 KV, so the only differences are the batched attention algorithm
    /// (documented bit-identical per (head, token) to the per-token kernel) and the GEMM
    /// vs matvec trunk matmul — the same gap the fp32 batched path already tolerates.
    /// The prompt is kept ≤4096 tokens so it stays on the batched path (bf16 can't chunk
    /// past 4096 until the flash port). Reference is bf16 per-token; candidate is bf16
    /// batched, teacher-forced onto the reference trajectory.
    /// </summary>
    private static void AssertBf16BatchedPrefillParity(string filename, string prompt, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(filename);
        if (path is null) return;

        const int steps = 6;
        const int ctx = 2048;

        var (pt, ptArgmax) = RunPrefillDecode(gpu, path, "bf16", prompt, steps, ctx, forced: null, batchedPrefill: false);
        var (bt, _) = RunPrefillDecode(gpu, path, "bf16", prompt, steps, ctx, forced: ptArgmax, batchedPrefill: true);

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(pt[p].Length, bt[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < pt[p].Length; i++)
            {
                Assert.True(float.IsFinite(bt[p][i]), $"{filename}: non-finite batched-bf16 logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(pt[p][i] - bt[p][i]));
            }
            Assert.True(maxAbs < maxAbsTol,
                $"{filename}: pos {p} batched-bf16 vs per-token-bf16 logit max-abs {maxAbs:F3} exceeds " +
                $"{maxAbsTol:F1} — the bf16 batched kernels/dispatch diverge from the per-token path.");
            Assert.True(TopK(bt[p], 5).Contains(ptArgmax[p]),
                $"{filename}: pos {p} per-token top-1 ({ptArgmax[p]}) fell out of batched-bf16's top-5.");
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
        => AssertBf16Parity("Qwen3-8B-Q4_K_M.gguf", LowEntropyPrompt, eosToken: null, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 E4B Q8_0: SWA + global layers, exercises AttentionSwaBf16.</summary>
    [Fact]
    public void Gemma4_E4B_Bf16Kv_ArgmaxStable_VsFp32()
        => AssertBf16Parity("gemma-4-E4B-it-Q8_0.gguf", LowEntropyPrompt, eosToken: null, maxAbsTol: 1.5f);

    /// <summary>
    /// Gemma 4 12B QAT: the driving model — adds attention_k_eq_v global layers. Q4_0
    /// 4-bit weights over 48 layers accumulate more bf16-store rounding, so the budget
    /// is wider than the Q8_0/Q4_K cases (observed peak ~4.0); top-1/top-5 stay stable.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Bf16Kv_ArgmaxStable_VsFp32()
        => AssertBf16Parity("gemma-4-12b-it-qat-q4_0.gguf", LowEntropyPrompt, eosToken: null, maxAbsTol: 8.0f);

    // ── Increment 1.5: bf16 batched prefill agrees with bf16 per-token ──────

    /// <summary>Qwen3-8B Q4_K: bf16 global batched prefill (AttentionBatchedBf16).</summary>
    [Fact]
    public void Qwen3_8B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertBf16BatchedPrefillParity("Qwen3-8B-Q4_K_M.gguf", LowEntropyPrompt, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 E4B Q8_0: bf16 SWA + global batched prefill (AttentionSwaBatchedBf16).</summary>
    [Fact]
    public void Gemma4_E4B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertBf16BatchedPrefillParity("gemma-4-E4B-it-Q8_0.gguf", LowEntropyPrompt, maxAbsTol: 1.5f);

    /// <summary>Gemma 4 12B QAT Q4_0: bf16 batched prefill with k_eq_v globals.</summary>
    [Fact]
    public void Gemma4_12B_Bf16BatchedPrefill_MatchesPerToken()
        => AssertBf16BatchedPrefillParity("gemma-4-12b-it-qat-q4_0.gguf", LowEntropyPrompt, maxAbsTol: 8.0f);
}
