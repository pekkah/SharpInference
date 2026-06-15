using SharpInference.Core;
using SharpInference.Engine;
using Xunit;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #224: <see cref="TierPlanner.Plan"/> with <c>pinGpuLayers: N</c> must price the routed-
/// expert cache budget (and KV / trunk-weight bytes) for the pinned split, not the greedy auto
/// split. The CLI and server previously pinned an explicit <c>-g N</c> via
/// <c>Plan(...) with { GpuLayers = N }</c>, which kept the auto-split
/// <see cref="LayerPlacement.ExpertCacheBudgetBytes"/> — a stale value the MoE CPU-vs-SLRU
/// auto-decision in <see cref="CudaHybridForwardPass"/> then read.
///
/// Loads OLMoE (a small MoE) and reads only GGUF metadata — no GPU, no inference — with a fixed
/// <see cref="HardwareProfile"/> so the budget math is deterministic. Skipped when the model is
/// not on disk (the planner needs real tensor sizes).
/// </summary>
public sealed class TierPlannerPinTests
{
    private static string? OlmoePath() => FirstExisting(
        @"C:\p\sharpi\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
        @"E:\models\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");

    private static string? FirstExisting(params string[] paths)
    {
        foreach (var p in paths)
            if (File.Exists(p)) return p;
        return null;
    }

    // 16 GB VRAM, fixed so the result does not depend on the test host's GPU.
    private static HardwareProfile FixedHw() =>
        new(VramBytes: 16L * 1024 * 1024 * 1024, RamBytes: 64L * 1024 * 1024 * 1024,
            CpuCores: 8, EstPcieBandwidthGBps: 16.0, HasAvx512: true);

    [Fact]
    public void Plan_PinGpuLayers_RepricesExpertBudgetForThatSplit()
    {
        string? path = OlmoePath();
        if (path is null) return; // model not on disk — skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.IsMoE, "OLMoE must be detected as MoE");
        Assert.True(hp.NumLayers >= 4, "need enough layers for a meaningful split");

        var hw = FixedHw();
        int few = Math.Max(1, hp.NumLayers / 4);
        int many = Math.Max(few + 1, hp.NumLayers * 3 / 4);

        // Fixed context so KV (and therefore the leftover expert budget) is deterministic and the
        // auto-context solver does not absorb all free VRAM into KV.
        var pFew = TierPlanner.Plan(model, hp, hw, requestedCtxSize: 4096, pinGpuLayers: few);
        var pMany = TierPlanner.Plan(model, hp, hw, requestedCtxSize: 4096, pinGpuLayers: many);

        // The pin is honored exactly.
        Assert.Equal(few, pFew.GpuLayers);
        Assert.Equal(many, pMany.GpuLayers);
        Assert.Equal(hp.NumLayers - few, pFew.CpuLayers);

        // More GPU trunk layers => more trunk weights + more KV => LESS room for the routed-expert
        // cache. The old `with { GpuLayers = }` override would have given both the SAME (auto)
        // budget; this asserts the budget now tracks the pinned split (the #224 fix).
        Assert.True(pMany.GpuKvBytes > pFew.GpuKvBytes,
            $"more layers must cost more KV: few={pFew.GpuKvBytes} many={pMany.GpuKvBytes}");
        Assert.True(pMany.ExpertCacheBudgetBytes < pFew.ExpertCacheBudgetBytes,
            $"few-layer budget ({pFew.ExpertCacheBudgetBytes}) must exceed many-layer budget ({pMany.ExpertCacheBudgetBytes})");
        Assert.True(pFew.ExpertCacheBudgetBytes > 0, "few-layer split should leave a positive expert budget");
        Assert.True(pMany.MoeRoutedExpertBytes > 0, "MoE must report a non-zero routed-expert cost");
    }

    [Fact]
    public void Plan_PinGpuLayers_ClampsToValidRange()
    {
        string? path = OlmoePath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var hw = FixedHw();

        Assert.Equal(hp.NumLayers, TierPlanner.Plan(model, hp, hw, pinGpuLayers: hp.NumLayers + 100).GpuLayers);
        Assert.Equal(0, TierPlanner.Plan(model, hp, hw, pinGpuLayers: -5).GpuLayers);
    }
}
