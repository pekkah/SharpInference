using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Pipeline;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #230: narrowed KV (--kv-type bf16/q8_0, #179) on the layer-split pure-attention MoE
/// hybrid (<see cref="CudaHybridForwardPass"/> — Qwen3-Coder-30B, OLMoE). It was previously
/// silently ignored (KV always fp32) even though TierPlanner priced the budget at the narrowed
/// dtype. These oracles teacher-force a narrowed-KV prefill+decode onto the fp32 trajectory (so
/// the KV dtype is the only variable per position) and assert it stays <b>argmax-stable</b>:
/// finite logits, fp32's top-1 inside the narrowed top-5, and a bounded logit max-abs gap. The
/// store rounding is lossy, so token-exact equality is not asserted (mirrors the dense
/// <see cref="CudaForwardPassKvDtypeTests"/> contract).
///
/// Skipped silently when CUDA is unavailable, the model isn't on disk, or construction OOMs.
/// q8_0 is skipped for a model whose per-layer kvDim isn't a multiple of 32 (the geometry the
/// ctor rejects). Compute-routing is pinned OFF so the batched prefill is deterministic per arm.
/// </summary>
public sealed class CudaHybridKvDtypeTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly bool _prevCompute = CudaHybridForwardPass.HybridPrefillComputeEnabled;

    public CudaHybridKvDtypeTests(ITestOutputHelper o)
    {
        _out = o;
        CudaHybridForwardPass.HybridPrefillComputeEnabled = false; // deterministic batched prefill
    }
    public void Dispose() => CudaHybridForwardPass.HybridPrefillComputeEnabled = _prevCompute;

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FirstExisting(params string[] c)
    {
        foreach (var p in c) if (File.Exists(p)) return p;
        return null;
    }

    private static (float[][] logits, int[] argmax) RunPrefillDecode(
        CudaBackend gpu, string path, string? kvDtype, string prompt, int steps, int ctx, int[]? forced,
        string? splitDecode = null)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        var prevSplit = Environment.GetEnvironmentVariable("SHARPI_SPLIT_DECODE");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", kvDtype);
        // null → flash-decoding split default (on, #238); "0" forces the single-block path.
        if (splitDecode is not null) Environment.SetEnvironmentVariable("SHARPI_SPLIT_DECODE", splitDecode);
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int useCtx = Math.Min(hp.ContextLength, ctx);
            var hw = HardwareProfile.Detect(gpu);
            var placement = TierPlanner.Plan(model, hp, hw, turboQuant: false, requestedCtxSize: useCtx,
                kvDtype: CudaForwardPass.ResolveConfiguredKvDType());
            using var fwd = new CudaHybridForwardPass(model, gpu, hp, placement);

            // Guard against a vacuous pass: confirm the requested dtype actually applied (the GPU
            // KV is genuinely narrowed). If env plumbing regressed and KV silently stayed fp32,
            // fp32-vs-fp32 would be trivially argmax-stable and hide the very bug #230 fixes.
            DType expected = kvDtype switch
            {
                "bf16" => DType.BFloat16,
                "q8_0" => DType.Q8_0,
                _ => DType.Float32,
            };
            Assert.Equal(expected, fwd.KvCacheDType);

            var tokens = tokenizer.Encode(prompt).ToArray();
            var perPos = new float[steps + 1][];
            var argmax = new int[steps + 1];
            var logits = fwd.Prefill(tokens).ToArray();
            perPos[0] = logits; argmax[0] = Sampler.Greedy(logits);
            for (int i = 0; i < steps; i++)
            {
                int fed = forced is not null ? forced[i] : argmax[i];
                logits = fwd.Forward(fed, tokens.Length + i).ToArray();
                perPos[i + 1] = logits; argmax[i + 1] = Sampler.Greedy(logits);
            }
            return (perPos, argmax);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prev);
            Environment.SetEnvironmentVariable("SHARPI_SPLIT_DECODE", prevSplit);
        }
    }

    /// <summary>
    /// Flash-decoding split-KV parity on the hybrid pass (#238). At long ctx the hybrid decode
    /// attention switches from the single-block kernel to the split-KV + combine path (the same
    /// kernels the dense #235/#237 tests validate bit-faithfully). This confirms the hybrid WIRING
    /// (dispatch gate, partials buffer, graph interaction) is correct: a &gt;4096-token prompt puts
    /// every decode step past the hybrid split threshold, and the single-block run (split off) is the
    /// trusted reference. Argmax-stable, not bit-identical (the combine reorders the reduction).
    /// </summary>
    private void AssertHybridSplitKvParity(string? path, string kvDtype, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) { _out.WriteLine("SKIP: no CUDA"); return; }
        if (path is null) { _out.WriteLine("SKIP: model not on disk"); return; }

        const int steps = 5;
        const int ctx = 5120;          // > the 4096 hybrid split threshold
        const int promptLen = 4200;    // every decode step lands at seqLen > 4096 → split path

        string prompt;
        using (var model = GgufModel.Open(path))
        {
            var tok = GgufTokenizer.FromGgufModel(model);
            var sb = new System.Text.StringBuilder();
            const string seed = "The quick brown fox jumps over the lazy dog. " +
                                "Sphinx of black quartz, judge my vow. Pack my box with five dozen liquor jugs. ";
            while (tok.Encode(sb.ToString()).Count < promptLen) sb.Append(seed);
            prompt = sb.ToString();
        }

        float[][] single, split; int[] singleArgmax;
        try
        {
            (single, singleArgmax) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: null, splitDecode: "0");
            (split, _) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: singleArgmax, splitDecode: "1");
        }
        catch (InvalidOperationException) { _out.WriteLine("SKIP: construction OOM'd"); return; }

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(single[p].Length, split[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < single[p].Length; i++)
            {
                Assert.True(float.IsFinite(split[p][i]), $"{kvDtype}: non-finite split logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(single[p][i] - split[p][i]));
            }
            Assert.True(TopK(split[p], 5).Contains(singleArgmax[p]),
                $"{kvDtype}: pos {p} single-block top-1 ({singleArgmax[p]}) fell out of the hybrid split top-5 " +
                $"(maxAbs {maxAbs:F4}) — the hybrid split-KV wiring reordered the head of the distribution.");
            Assert.True(maxAbs < maxAbsTol,
                $"{kvDtype}: pos {p} hybrid split-vs-single logit max-abs {maxAbs:F4} exceeds {maxAbsTol:F2} — a wiring bug.");
        }
    }

    private static HashSet<int> TopK(float[] v, int k)
    {
        var idx = new int[v.Length];
        for (int i = 0; i < v.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        var set = new HashSet<int>(k);
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    private void AssertKvParity(string? path, string kvDtype, string prompt, float maxAbsTol, int ctx = 2048)
    {
        using var gpu = TryCreate();
        if (gpu is null) { _out.WriteLine("SKIP: no CUDA"); return; }
        if (path is null) { _out.WriteLine("SKIP: model not on disk"); return; }

        const int steps = 6;
        float[][] f32, kv; int[] f32Argmax;
        try
        {
            (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null);
            (kv, _) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: f32Argmax);
        }
        catch (InvalidOperationException) { _out.WriteLine("SKIP: construction OOM'd"); return; }
        // NOTE: q8_0 is supported for both test models (kvDim % 32 == 0: OLMoE 16×128, Coder 4×128),
        // so a NotSupportedException here is a real regression — let it FAIL, don't skip.

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(f32[p].Length, kv[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32[p].Length; i++)
            {
                Assert.True(float.IsFinite(kv[p][i]), $"{kvDtype}: non-finite logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - kv[p][i]));
            }
            Assert.True(TopK(kv[p], 5).Contains(f32Argmax[p]),
                $"{kvDtype}: fp32 top-1 ({f32Argmax[p]}) fell out of the {kvDtype} top-5 at pos {p} " +
                $"(maxAbs {maxAbs:F3}) — the narrowed-KV hybrid path reordered the head of the distribution.");
            Assert.True(maxAbs < maxAbsTol,
                $"{kvDtype}: pos {p} vs fp32 logit max-abs {maxAbs:F3} exceeds the rounding budget ({maxAbsTol:F1}).");
        }
        _out.WriteLine($"OK {kvDtype} argmax-stable vs fp32 ({Path.GetFileName(path)})");
    }

    /// <summary>
    /// Greedy (NOT teacher-forced) coherence on the narrowed-KV hybrid path — the narrowed run
    /// picks its OWN tokens, so an #188-style degenerate collapse (all-EOS / single-token repeat)
    /// that teacher-forcing would mask is caught here. Asserts finite logits, first token ≠ EOS,
    /// and ≥2 distinct tokens over the run. (Separate from AssertKvParity, which teacher-forces.)
    /// </summary>
    private void AssertGreedyCoherence(string? path, string kvDtype, string userMessage, int ctx = 2048)
    {
        using var gpu = TryCreate();
        if (gpu is null) { _out.WriteLine("SKIP: no CUDA"); return; }
        if (path is null) { _out.WriteLine("SKIP: model not on disk"); return; }
        // Render the model's OWN chat template (#230 review): a raw continuation prompt makes an
        // instruct model collapse to a single token regardless of KV dtype (the 'prompt must match
        // the chat template' trap), which would falsely flag the narrowed path. Falls back to the
        // raw message when the GGUF carries no template.
        int eosId; string prompt;
        using (var m = GgufModel.Open(path))
        {
            var tok = GgufTokenizer.FromGgufModel(m);
            eosId = tok.EosTokenId;
            prompt = ApplyChatTemplate(tok, userMessage);
        }

        const int steps = 6;
        int[] argmax; float[][] logits;
        try { (logits, argmax) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: null); }
        catch (InvalidOperationException) { _out.WriteLine("SKIP: construction OOM'd"); return; }

        for (int p = 0; p <= steps; p++)
            for (int i = 0; i < logits[p].Length; i++)
                Assert.True(float.IsFinite(logits[p][i]), $"{kvDtype}: non-finite greedy logit at pos {p}, idx {i}.");
        Assert.True(argmax[0] != eosId, $"{kvDtype}: first greedy token was EOS — narrowed-KV greedy decode collapsed.");
        Assert.True(new HashSet<int>(argmax).Count >= 2,
            $"{kvDtype}: greedy decode produced only 1 distinct token ([{string.Join(",", argmax)}]) — degenerate narrowed-KV decode.");
        _out.WriteLine($"OK {kvDtype} greedy-coherent ({Path.GetFileName(path)})");
    }

    /// <summary>
    /// Render the model's GGUF chat template (tokenizer.chat_template) around a single user turn
    /// with add_generation_prompt, mirroring the CLI/server path. Returns the raw message when the
    /// model carries no template. Used by the greedy-coherence tests so an instruct model sees a
    /// template-correct prompt rather than a raw continuation it would degenerate on.
    /// </summary>
    private static string ApplyChatTemplate(GgufTokenizer tok, string userMessage)
    {
        if (tok.ChatTemplate is null) return userMessage;
        var messages = JinjaChatTemplate.BuildMessages(userMessage, systemContent: null);
        return tok.ChatTemplate.Render(new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["add_generation_prompt"] = true,
            ["tools"] = null,
            ["enable_thinking"] = false,
        });
    }

    private const string Prompt = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";

    /// <summary>Repeat a varied seed until the tokenizer emits ≥ <paramref name="approx"/> tokens (for the >4096 wave path).</summary>
    private static string LongPrompt(string path, int approx)
    {
        using var model = GgufModel.Open(path);
        var tok = GgufTokenizer.FromGgufModel(model);
        const string seed = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow. " +
                            "Pack my box with five dozen liquor jugs. How razorback-jumping frogs can level six piqued gymnasts. ";
        var sb = new System.Text.StringBuilder();
        while (tok.Encode(sb.ToString()).Count < approx)
        {
            sb.Append(seed);
            if (sb.Length > 400_000) break;
        }
        return sb.ToString();
    }

    private static string? OlmoePath() => FirstExisting(
        @"C:\p\sharpi\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        @"E:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");

    private static string? CoderPath() => FirstExisting(
        @"C:\p\sharpi\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        @"E:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");

    [Fact] public void OLMoE_Bf16Kv_ArgmaxStable() => AssertKvParity(OlmoePath(), "bf16", Prompt, maxAbsTol: 2.5f);
    [Fact] public void OLMoE_Q8Kv_ArgmaxStable()  => AssertKvParity(OlmoePath(), "q8_0", Prompt, maxAbsTol: 3.5f);
    [Fact] public void Coder30B_Bf16Kv_ArgmaxStable() => AssertKvParity(CoderPath(), "bf16", Prompt, maxAbsTol: 5.0f);
    [Fact] public void Coder30B_Q8Kv_ArgmaxStable()  => AssertKvParity(CoderPath(), "q8_0", Prompt, maxAbsTol: 6.0f);

    /// <summary>Hybrid split-KV decode (#238) vs single-block on Coder-30B q8 at &gt;4096 ctx —
    /// the layer-split MoE hybrid is the model that reaches the hybrid split threshold (OLMoE caps
    /// at 4096 and never splits). Decode A/B confirmed the win (1.49× @6K → 2.08× @16K); this fences
    /// correctness of the wiring.</summary>
    [Fact] public void Coder30B_Q8SplitKv_MatchesSingleBlock() => AssertHybridSplitKvParity(CoderPath(), "q8_0", maxAbsTol: 6.0f);

    /// <summary>
    /// >4096 wave path (AttentionBatchedWaveQ8_0): a single Prefill of &gt;4096 tokens makes
    /// PrefillBatchedTrunk take the wave SDPA branch under a narrowed cache. ctx 6144.
    /// </summary>
    [Fact]
    public void Coder30B_Q8Kv_Wave_ArgmaxStable()
    {
        var path = CoderPath();
        if (path is null || !CudaBackend.IsAvailable()) { _out.WriteLine("SKIP"); return; }
        AssertKvParity(path, "q8_0", LongPrompt(path, 4600), maxAbsTol: 8.0f, ctx: 6144);
    }

    // Narrowed-KV greedy self-decode stays coherent (no #188-style collapse) on a template-correct
    // prompt — both the headline Coder-30B and OLMoE (its raw-prompt collapse was the chat-template
    // trap, fixed by ApplyChatTemplate above).
    private const string CoherenceMessage = "Write a short sentence about the ocean.";
    [Fact] public void Coder30B_Q8Kv_GreedyCoherent() => AssertGreedyCoherence(CoderPath(), "q8_0", CoherenceMessage);
    [Fact] public void OLMoE_Q8Kv_GreedyCoherent()    => AssertGreedyCoherence(OlmoePath(), "q8_0", CoherenceMessage);
}
