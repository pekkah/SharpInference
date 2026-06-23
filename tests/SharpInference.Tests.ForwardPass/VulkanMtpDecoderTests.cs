using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Pure-Vulkan coverage for the NEXTN/MTP head forward pass (#357 PR3) on the dense qwen36
/// 27B-MTP GDN model (Qwen3.6-27B-MTP). PR3 loads the head weights + wires
/// <see cref="VulkanHybridGdnForwardPass.MtpForward"/> / <c>GpuMtpAttnBlock</c> /
/// <see cref="VulkanHybridGdnForwardPass.PrefillMtp"/> + the absolute-position hidden-history
/// surface, flipping <see cref="VulkanHybridGdnForwardPass.HasMtpHead"/> /
/// <see cref="VulkanHybridGdnForwardPass.SupportsBatchVerify"/> true so the
/// <see cref="MtpDecoder"/> selects this pass and does self-speculative decoding on Vulkan.
/// <list type="bullet">
///   <item>focused: a single <c>MtpForward</c> produces all-finite draft logits with the argmax
///         in vocab range (the head wiring + concat order + KV append are sound);</item>
///   <item>e2e: <see cref="MtpDecoder"/> batched greedy decode is coherent and the chained drafts
///         actually get accepted (a broken MtpLastHidden chain / MTP KV refresh would tank the
///         acceptance rate even if the output stays correct).</item>
/// </list>
/// Silent-skips when Vulkan is unavailable, the device is out of memory (ring/head didn't
/// allocate), or the 27B-MTP GGUF isn't on disk. NOT run by the implementation pass (the
/// orchestrator verifies on a real GPU).
/// </summary>
public sealed class VulkanMtpDecoderTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindMtpModelPath()
    {
        const string fileName = "Qwen3.6-27B-MTP-Q4_K_M.gguf";
        string[] absoluteRoots = { @"E:\models", @"C:\p\sharpi\models" };
        foreach (var root in absoluteRoots)
        {
            var p = Path.Combine(root, fileName);
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static LayerPlacement GdnPlacement(ModelHyperparams hp) => new(
        GpuLayers: hp.NumLayers,
        CpuLayers: 0,
        GpuWeightBytes: 0,
        GpuKvBytes: 0,
        RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

    /// <summary>
    /// Constructs the pass with a 4-token snapshot ring. SHARPI_MTP_BATCH_MAX is instance-resolved
    /// at construction, so the env scope only needs to cover the ctor. Returns null (caller skips)
    /// when the device can't fit the dense GDN trunk + ring + MTP head.
    /// </summary>
    private static VulkanHybridGdnForwardPass? CreatePass(GgufModel model, VulkanBackend gpu,
                                                          ModelHyperparams hp)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_MTP_BATCH_MAX");
        Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", "4");
        try
        {
            return new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp));
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            return null; // device too small → caller skips
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", prev);
        }
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    [Fact]
    public void MtpForward_ProducesFiniteDraftLogits()
    {
        using var gpu = TryCreateVulkan();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");
        Assert.True(hp.NumMtpLayers > 0, "Expected a NEXTN/MTP head on the 27B-MTP model");
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var fwd = CreatePass(model, gpu, hp);
        if (fwd is null) return;                                    // device OOM → skip
        Assert.True(fwd.HasMtpHead, "PR3 must load the MTP head → HasMtpHead == true.");

        var prompt = tokenizer.Encode("The capital of France is").ToArray();
        int P = prompt.Length;

        // Prefill the trunk (populates the absolute-position hidden history) + the MTP KV cache.
        var prefillLogits = fwd.Prefill(prompt).ToArray();
        fwd.PrefillMtp(prompt);

        // One MTP draft step off the last main token + its pre-output-norm hidden.
        int lastTok = ArgMax(prefillLogits);
        var draftLogits = fwd.MtpForward(lastTok, P, fwd.LastHidden);

        Assert.Equal(hp.VocabSize, draftLogits.Length);
        for (int i = 0; i < draftLogits.Length; i++)
            Assert.True(float.IsFinite(draftLogits[i]),
                $"MtpForward draft logit {i} is non-finite ({draftLogits[i]}).");
        int argmax = ArgMax(draftLogits);
        Assert.InRange(argmax, 0, hp.VocabSize - 1);

        // MtpLastHidden (the chained-draft self-hidden) must be populated + finite.
        var selfHidden = fwd.MtpLastHidden;
        Assert.Equal(hp.EmbeddingDim, selfHidden.Length);
        for (int i = 0; i < selfHidden.Length; i++)
            Assert.True(float.IsFinite(selfHidden[i]),
                $"MtpLastHidden[{i}] is non-finite ({selfHidden[i]}).");
    }

    [Fact]
    public void MtpDecoder_BatchedGreedy_CoherentWithAcceptedDrafts()
    {
        using var gpu = TryCreateVulkan();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var fwd = CreatePass(model, gpu, hp);
        if (fwd is null) return;                                    // device OOM → skip
        Assert.True(fwd.HasMtpHead);
        Assert.True(fwd.SupportsBatchVerify,
            "27B-MTP without SnapKV must support batched verify (GDN ring + MTP head must have loaded).");
        if (fwd.MaxBatchVerifyTokens < 4) return;                  // ring didn't allocate ≥4 → skip

        var prompt = tokenizer.Encode(
            "Write a Python function that sorts a list using the quicksort algorithm:").ToArray();
        var logits = fwd.Prefill(prompt);

        var decoder = new MtpDecoder(fwd);
        decoder.Initialize(prompt.Length, logits);
        fwd.PrefillMtp(prompt);

        var produced = new List<int>(24);
        int[] stops = tokenizer.EogTokenIds.ToArray();
        decoder.Decode(24, stops, produced.Add, pMin: 1f, draftN: 3);

        Assert.True(produced.Count >= 8,
            $"Batched MTP decode stopped after {produced.Count} tokens — unexpectedly early EOS.");
        Assert.True(produced.Distinct().Count() >= 2,
            $"Degenerate decode: [{string.Join(",", produced)}]");
        // Chained drafting must actually land accepts. The 27B head accepts 95-100% at depth 1;
        // depth-3 chains compound but anything below ~30% means the chain / self-hidden wiring is
        // broken even if the output stays correct.
        Assert.True(decoder.TotalDraftsEmitted > 0);
        Assert.True(decoder.AcceptanceRate >= 0.3f,
            $"Chained-draft acceptance {decoder.AcceptanceRate:P0} " +
            $"({decoder.TotalDraftsAccepted}/{decoder.TotalDraftsEmitted}) is far below the " +
            "depth-1 reference (95-100%); MtpLastHidden chaining or the MTP KV refresh is off.");
    }
}
