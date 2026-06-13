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
        CudaBackend gpu, string path, string? kvDtype, string prompt, int steps, int ctx, int[]? forced)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", kvDtype);
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
        finally { Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prev); }
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

    private void AssertKvParity(string? path, string kvDtype, string prompt, float maxAbsTol)
    {
        using var gpu = TryCreate();
        if (gpu is null) { _out.WriteLine("SKIP: no CUDA"); return; }
        if (path is null) { _out.WriteLine("SKIP: model not on disk"); return; }

        const int steps = 6, ctx = 2048;
        float[][] f32, kv; int[] f32Argmax;
        try
        {
            (f32, f32Argmax) = RunPrefillDecode(gpu, path, "fp32", prompt, steps, ctx, forced: null);
            (kv, _) = RunPrefillDecode(gpu, path, kvDtype, prompt, steps, ctx, forced: f32Argmax);
        }
        catch (InvalidOperationException) { _out.WriteLine("SKIP: construction OOM'd"); return; }
        catch (NotSupportedException e)   { _out.WriteLine($"SKIP: {e.Message}"); return; } // e.g. q8 geometry

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

    private const string Prompt = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";

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
}
