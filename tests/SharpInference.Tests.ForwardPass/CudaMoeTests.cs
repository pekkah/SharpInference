using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end smoke tests for the CUDA MoE path. Skipped silently when CUDA is
/// unavailable or no MoE GGUF is on disk — same pattern as <see cref="CudaTurboQuantTests"/>.
///
/// We don't do CPU↔CUDA greedy parity here because:
///   • Q4_K matvec quantizes activations to Q8_1 (small numerical drift);
///   • MoE router top-K is sensitive to that drift (logits at the K/K+1 boundary
///     can flip experts), so identical greedy outputs are not expected even when
///     both paths are correct;
///   • CPU MoE has a separate stability issue on OLMoE that's out of scope here.
/// Instead these tests verify the pipeline produces well-formed logits and that
/// decode yields more than one distinct token (catches the "all-EOS" failure mode
/// that <see cref="VulkanShaderTests.HybridForwardPass_DenseSmallVocab_ProducesCoherentDecode"/>
/// caught for the dense path).
/// </summary>
public sealed class CudaMoeTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindMoEModelPath()
    {
        // Smallest first — OLMoE is the only MoE that fits a 12 GB card with CUDA full-offload.
        string[] candidates =
        {
            "models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
            "models\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        };
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in candidates)
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
    /// Loads a small MoE model on CUDA, runs prefill + 5 greedy decode tokens, and
    /// asserts the output is well-formed: finite logits, a non-degenerate range,
    /// and at least two distinct argmax tokens across the decode window.
    ///
    /// The "two distinct tokens" check is the load-bearing assertion — when the
    /// MoE forward pass produces garbage logits, every position picks the same
    /// fallback token (typically 0 / EOS / newline), which is the exact failure
    /// mode that has bitten prior MoE bring-ups. Two distinct argmaxes means the
    /// model is actually conditioning on the prompt + accumulated context.
    /// </summary>
    [Fact]
    public void CudaMoeForwardPass_ProducesWellFormedLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMoEModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        if (!hp.IsMoE) return; // Defensive — finder should only return MoE models.

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp);

        var prompt = "Once upon a time";
        var tokens = tokenizer.Encode(prompt);

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = fwd.Forward(tokens[i], i);

        // Range + finiteness on the final prefill position's logits.
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            Assert.True(float.IsFinite(v), $"Non-finite logit at vocab idx {i}: {v}");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"Logit range too tight ({min:F3}..{max:F3}); the model is producing degenerate output. " +
            "MoE FFN likely returning near-zero hidden state.");

        // Greedy decode 5 tokens; assert at least 2 distinct argmaxes.
        // (A trained model on a non-pathological prompt should pick varied tokens;
        // an all-same output is the canonical "MoE broken" signature.)
        var decoded = new List<int>(5);
        for (int i = 0; i < 5; i++)
        {
            int next = SharpInference.Engine.Sampler.Greedy(logits);
            decoded.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);
        }

        int distinct = decoded.Distinct().Count();
        Assert.True(distinct >= 2,
            $"Greedy decode produced only {distinct} distinct token(s) across 5 steps " +
            $"({string.Join(",", decoded)}); MoE FFN may be stuck in a degenerate loop.");
    }
}
