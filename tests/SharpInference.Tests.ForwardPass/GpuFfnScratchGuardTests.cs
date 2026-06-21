using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #315: hardening the Vulkan MoE FFN scratch-buffer sizing
/// ("the MoE-on-Vulkan garble chased since #2").
///
/// Part 1 (GPU-free): the centralized sizing helper <see cref="GpuForwardPass.ComputeFfnScratchDim"/>
/// and its invariant guard <see cref="GpuForwardPass.ValidateFfnScratchDim"/> are exercised directly
/// (no model, no GPU) so the MoE-vs-dense distinction can't silently drift. The regression direction
/// — scratch sized to max(intermDim, expertDim) instead of expertDim — is covered deterministically
/// here by <see cref="ValidateFfnScratchDim_MoE_OversizedToIntermDim_Throws"/>.
///
/// Part 2 (GPU): a real MoE model (OLMoE) is run end-to-end on the Vulkan
/// <see cref="GpuForwardPass"/>. This is a no-false-positive + well-formedness smoke test: it confirms
/// (a) the scratch guard does NOT false-positive on a genuine MoE layout and (b) MoE inference produces
/// well-formed logits on Vulkan. It is NOT itself a regression detector: OLMoE's GGUF carries only
/// feed_forward_length (1024) and no expert_feed_forward_length, so ExpertIntermediateDim falls back to
/// feed_forward_length and expertDim == intermDim == 1024. With those dims equal, max(intermDim, expertDim)
/// equals the correct expertDim, so reverting to the bug would neither garble OLMoE nor trip the guard —
/// the unit test above is what pins the regression direction.
/// </summary>
public sealed class GpuFfnScratchGuardTests
{
    // ---- Part 1: GPU-free guard logic ---------------------------------------

    [Fact]
    public void ComputeFfnScratchDim_MoE_UsesExpertDim()
    {
        // illustrative MoE dims (Qwen3-Coder-style: expertDim 2816 < intermDim 8192).
        // NOT OLMoE, whose real dims are expertDim == intermDim == 1024.
        Assert.Equal(2816, GpuForwardPass.ComputeFfnScratchDim(isMoE: true, intermDim: 8192, expertDim: 2816));
    }

    [Fact]
    public void ComputeFfnScratchDim_Dense_UsesIntermDim()
    {
        Assert.Equal(8192, GpuForwardPass.ComputeFfnScratchDim(isMoE: false, intermDim: 8192, expertDim: 0));
    }

    [Fact]
    public void ValidateFfnScratchDim_MoE_CorrectSizing_DoesNotThrow()
    {
        // scratch == expertDim → the only correct MoE sizing.
        GpuForwardPass.ValidateFfnScratchDim(isMoE: true, scratchDim: 2816, intermDim: 8192, expertDim: 2816);
    }

    [Fact]
    public void ValidateFfnScratchDim_MoE_OversizedToIntermDim_Throws()
    {
        // The exact regression: scratch sized to max(intermDim, expertDim).
        Assert.Throws<InvalidOperationException>(() =>
            GpuForwardPass.ValidateFfnScratchDim(isMoE: true, scratchDim: 8192, intermDim: 8192, expertDim: 2816));
    }

    [Fact]
    public void ValidateFfnScratchDim_MoE_ZeroExpertDim_Throws()
    {
        // MoE flagged but ExpertIntermediateDim is 0 (bad GGUF metadata).
        Assert.Throws<InvalidOperationException>(() =>
            GpuForwardPass.ValidateFfnScratchDim(isMoE: true, scratchDim: 0, intermDim: 8192, expertDim: 0));
    }

    [Fact]
    public void ValidateFfnScratchDim_Dense_CorrectSizing_DoesNotThrow()
    {
        GpuForwardPass.ValidateFfnScratchDim(isMoE: false, scratchDim: 8192, intermDim: 8192, expertDim: 0);
    }

    [Fact]
    public void ValidateFfnScratchDim_Dense_WrongSizing_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            GpuForwardPass.ValidateFfnScratchDim(isMoE: false, scratchDim: 2816, intermDim: 8192, expertDim: 0));
    }

    // ---- Part 2: GPU MoE forward-pass well-formedness ------------------------

    /// <summary>
    /// Runs a real MoE model (OLMoE: no shared expert) through the Vulkan
    /// <see cref="GpuForwardPass"/> and asserts the output logits are well-formed.
    /// This is a no-false-positive + well-formedness smoke test: it confirms the FFN scratch
    /// guard (issue #315) does NOT false-positive on a genuine MoE layout and that MoE inference
    /// produces well-formed logits on Vulkan.
    ///
    /// It is NOT a regression detector: OLMoE has expertDim == intermDim == 1024 (its GGUF has
    /// feed_forward_length but no expert_feed_forward_length), so max(intermDim, expertDim) equals
    /// the correct expertDim — reverting the sizing bug would neither garble OLMoE nor trip the
    /// guard. The regression direction is pinned by the GPU-free unit test
    /// <see cref="ValidateFfnScratchDim_MoE_OversizedToIntermDim_Throws"/>.
    ///
    /// Silently no-ops if the Vulkan backend can't be created or no MoE GGUF is present —
    /// matches the model-dependent skip idiom used throughout this test project.
    /// </summary>
    [Fact]
    public void GpuForwardPassMoE_ProducesWellFormedLogits()
    {
        var path = FindMoEModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        // Pass `model` so HasQkNorm / IsPerChannelQkNorm probe the tensor index (OLMoE needs it).
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return; // dense model found instead — nothing MoE-specific to assert.

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Vulkan.VulkanBackend gpu;
        try
        {
            gpu = new Vulkan.VulkanBackend();
        }
        catch
        {
            return; // No usable Vulkan device — skip.
        }

        using (gpu)
        using (var gpuFwd = new GpuForwardPass(model, gpu, hp))
        {
            var tokens = tokenizer.Encode("The capital of France is");

            // Prefill the prompt one token at a time (mirrors GpuForwardPassMatchesCpuOutput).
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < tokens.Count; i++)
                logits = gpuFwd.Forward(tokens[i], i);

            // Greedy-decode ~4 steps, collecting the argmax token at each step.
            const int steps = 4;
            var decoded = new int[steps];
            int pos = tokens.Count;
            for (int s = 0; s < steps; s++)
            {
                Assert.Equal(hp.VocabSize, logits.Length);

                // (1) Every logit must be finite — a corrupted expert MatMul cascades to NaN/Inf.
                float min = float.MaxValue, max = float.MinValue;
                for (int i = 0; i < logits.Length; i++)
                {
                    float v = logits[i];
                    Assert.True(float.IsFinite(v), $"Non-finite logit at step {s} index {i}: {v}");
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                // (2) Logits must have spread — all-zero / all-equal output is degenerate.
                Assert.True(max - min > 0.1f,
                    $"Degenerate logit range at step {s}: [{min:F4}, {max:F4}] (expected spread > 0.1).");

                int next = Argmax(logits);
                decoded[s] = next;
                logits = gpuFwd.Forward(next, pos++);
            }

            // (3) Coherent decode produces variety — corruption tends to lock onto one token.
            int distinct = decoded.Distinct().Count();
            Assert.True(distinct >= 2,
                $"Greedy decode produced only {distinct} distinct token(s): " +
                $"[{string.Join(", ", decoded)}] — expert output looks garbled (issue #315).");
        }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Finds a MoE GGUF. OLMoE is a small MoE model (1B active / 7B total) that fully fits the
    /// pure Vulkan <see cref="GpuForwardPass"/> path — which uploads all layers — on a typical
    /// 12 GB GPU. Larger MoE models (Qwen3-Coder-30B, Llama-4-Scout) are deliberately excluded:
    /// they would OOM/crash rather than skip on that path.
    /// </summary>
    private static string? FindMoEModelPath()
    {
        return FindModelPath(
            "models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
    }

    private static string? FindModelPath(params string[] candidates)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var candidate in candidates)
            {
                var fullPath = Path.Combine(dir, candidate);
                if (File.Exists(fullPath))
                    return fullPath;
            }
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
