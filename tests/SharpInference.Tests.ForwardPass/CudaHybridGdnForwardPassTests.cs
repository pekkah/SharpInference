using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Smoke tests for <see cref="CudaHybridGdnForwardPass"/> — the CUDA + CPU hybrid
/// for qwen35moe (Qwen3.6-35B-A3B). Skipped silently when CUDA is unavailable or
/// the 22 GB qwen35moe GGUF isn't on disk.
///
/// Like <see cref="HybridGdnForwardPassTests"/>, this is well-formedness only —
/// finite logits, non-degenerate range, and at least two distinct argmax tokens
/// across a short decode window. Greedy parity vs the CPU path is the goal but
/// not asserted here (Q4_K matvec quantizes activations via Q8_1; expert top-K
/// is sensitive to that drift).
/// </summary>
public sealed class CudaHybridGdnForwardPassTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindHybridModelPath()
    {
        // Big model: typically on E:\models\.
        string[] absoluteCandidates =
        {
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        string[] relativeCandidates =
        {
            @"models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in relativeCandidates)
            {
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Loads qwen35moe via the CUDA hybrid path, runs prefill + 4 greedy decode
    /// tokens, and asserts the output is well-formed.
    ///
    /// LayerPlacement is constructed with <c>GpuLayers = NumLayers</c> — that's
    /// a hint to the class that the user wants GPU usage; the class's internal
    /// dispatch always routes GDN blocks to CPU and attention blocks to GPU
    /// regardless. Phase 6c will introduce a hybrid-aware planner.
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_ProducesWellFormedLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;   // silent skip — same pattern as CudaMoeTests

        var path = FindHybridModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: this test should only fire on a hybrid GDN model with MoE.
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for qwen35moe model");
        Assert.NotNull(hp.Gdn);
        Assert.NotNull(hp.LayerTypes);
        Assert.True(hp.IsMoE, "Expected hp.IsMoE for qwen35moe model");

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // GpuLayers = NumLayers is the "user wants GPU" sentinel; the class ignores
        // the cut point and dispatches per LayerTypes.
        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers,
            CpuLayers: 0,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

        using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);

        // Prefill — sequential T-step recurrence under the hood.
        var logits = fwd.Prefill(tokens);

        // Range + finiteness.
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            Assert.True(float.IsFinite(v), $"Non-finite logit at vocab idx {i}: {v}");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"Logit range too tight ({min:F3}..{max:F3}); CUDA hybrid GDN forward pass " +
            "is producing degenerate output.");

        // Greedy decode 4 tokens; assert at least 2 distinct argmaxes.
        var decoded = new List<int>(4);
        for (int i = 0; i < 4; i++)
        {
            int next = Sampler.Greedy(logits);
            decoded.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);

            for (int k = 0; k < logits.Length; k++)
                Assert.True(float.IsFinite(logits[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {logits[k]}");
        }

        int distinct = decoded.Distinct().Count();
        Assert.True(distinct >= 2,
            $"Greedy decode produced only {distinct} distinct token(s) across 4 steps " +
            $"({string.Join(",", decoded)}); the CUDA hybrid forward pass may be stuck in a " +
            "degenerate loop (logits collapsed onto a single output).");
    }
}
