using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Static-estimate placement decisions for the DSpark draft head
/// (<see cref="DSparkPlacementPlanner"/>, docs/dspark-plan.md §4). Pure budget math over
/// hand-built <see cref="HardwareProfile"/> / <see cref="LayerPlacement"/> records:
/// auto mode prefers GPU colocation, falls back to CPU when VRAM is tight but RAM and
/// cores allow, else Off; explicit user overrides pin the mode outright (TierPlanner
/// pin philosophy — negative headroom is reported, not rejected).
/// </summary>
public sealed class DSparkPlacementPlannerTests
{
    private const long GiB = 1024L * 1024 * 1024;
    private const long MiB = 1024L * 1024;

    private static HardwareProfile Hw(long vram, long ram, int cores) =>
        new(VramBytes: vram, RamBytes: ram, CpuCores: cores,
            EstPcieBandwidthGBps: vram > 0 ? 25.0 : 0.0, HasAvx512: false);

    // ── Auto mode ────────────────────────────────────────────────────────────────

    [Fact]
    public void Auto_PicksGpu_WhenVramFits()
    {
        // 24 GB card, target uses 4 GB weights + 1 GB KV; reserved = 2.4 GB (10%).
        // Free ≈ 16.6 GB ≥ 2 GB × 1.15 scratch margin → GPU colocation.
        var hw = Hw(24 * GiB, 64 * GiB, 16);
        var target = new LayerPlacement(36, 0, 4 * GiB, 1 * GiB, 8192);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 3 * GiB);

        Assert.Equal(DSparkPlacement.Gpu, d.Placement);
        Assert.Equal(2 * GiB, d.DraftHeadBytes);
        Assert.True(d.HeadroomBytes > 0);
        Assert.Contains("gpu", d.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Auto_PicksCpu_WhenVramTight_RamFits()
    {
        // 8 GB card mostly consumed by the target (6 GB weights + 1 GB KV, 0.8 GB
        // reserved → ~0.2 GB free < 2.3 GB needed), but 64 GB RAM / 16 cores → CPU.
        var hw = Hw(8 * GiB, 64 * GiB, 16);
        var target = new LayerPlacement(28, 0, 6 * GiB, 1 * GiB, 4096);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 3 * GiB);

        Assert.Equal(DSparkPlacement.Cpu, d.Placement);
        Assert.Equal(3 * GiB, d.DraftHeadBytes);
        Assert.True(d.HeadroomBytes > 0);
        Assert.Contains("cpu", d.Reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("VRAM too tight", d.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void Auto_Off_WhenNeitherFits()
    {
        // 1 GB VRAM (512 MB free after the 512 MB floor) and 2 GB RAM (1.5 GB free):
        // both below the 2.3 GB scratch-margined head cost → Off, naming both budgets.
        var hw = Hw(1 * GiB, 2 * GiB, 16);
        var target = new LayerPlacement(0, 0, 0, 0, 2048);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 2 * GiB);

        Assert.Equal(DSparkPlacement.Off, d.Placement);
        Assert.Equal(0, d.DraftHeadBytes);
        Assert.Contains("VRAM", d.Reason, StringComparison.Ordinal);
        Assert.Contains("RAM", d.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void Auto_Off_WhenFewCores()
    {
        // RAM comfortably fits the head, but 2 cores < the MinCpuCores=4 floor:
        // a CPU draft chain would bottleneck the target, so the planner opts Off.
        var hw = Hw(0, 64 * GiB, 2);
        var target = new LayerPlacement(0, 36, 0, 0, 4096, CpuWeightBytes: 8 * GiB);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 2 * GiB);

        Assert.Equal(DSparkPlacement.Off, d.Placement);
        Assert.Contains("cores", d.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Auto_NoGpu_PicksCpu()
    {
        // VramBytes == 0 skips the GPU branch entirely; 32 GB RAM minus 8 GB target
        // trunk minus the 3.2 GB reserve leaves ~20.8 GB ≥ 2.3 GB needed → CPU.
        var hw = Hw(0, 32 * GiB, 8);
        var target = new LayerPlacement(0, 36, 0, 0, 4096, CpuWeightBytes: 8 * GiB);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 2 * GiB);

        Assert.Equal(DSparkPlacement.Cpu, d.Placement);
        Assert.Equal(2 * GiB, d.DraftHeadBytes);
        Assert.True(d.HeadroomBytes > 0);
        Assert.Contains("no GPU", d.Reason, StringComparison.Ordinal);
    }

    // ── User overrides ───────────────────────────────────────────────────────────

    [Fact]
    public void Override_Gpu_Honored_EvenWhenTooSmall()
    {
        // 2 GB card with 1.5 GB already claimed by the target and the 512 MB floor:
        // 0 free VRAM, yet the explicit pin is honored (same contract as -g N) —
        // the decision reports negative headroom instead of downgrading.
        var hw = Hw(2 * GiB, 16 * GiB, 8);
        var target = new LayerPlacement(10, 0, 1536 * MiB, 0, 2048);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 1 * GiB, draftHeadBytesCpuQuant: 1 * GiB,
            userOverride: DSparkPlacement.Gpu);

        Assert.Equal(DSparkPlacement.Gpu, d.Placement);
        Assert.Equal(1 * GiB, d.DraftHeadBytes);
        Assert.True(d.HeadroomBytes < 0);
        Assert.Contains("user override", d.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void Override_Off_DisablesRegardlessOfHeadroom()
    {
        var hw = Hw(24 * GiB, 64 * GiB, 16);
        var target = new LayerPlacement(36, 0, 4 * GiB, 1 * GiB, 8192);

        var d = DSparkPlacementPlanner.Plan(hw, target,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 2 * GiB,
            userOverride: DSparkPlacement.Off);

        Assert.Equal(DSparkPlacement.Off, d.Placement);
        Assert.Equal(0, d.DraftHeadBytes);
        Assert.Equal(0, d.HeadroomBytes);
        Assert.Contains("user override", d.Reason, StringComparison.Ordinal);
    }

    // ── Shared reserve helper (TierPlanner.ReservedVramBytes, PR #413 spec §4) ────

    [Fact]
    public void ReservedVramBytes_FloorsAt512Mb_ThenTenPercent()
    {
        // 4 GB card: 10% = ~410 MB < the 512 MB floor → 512 MB exactly.
        Assert.Equal(512L * 1024 * 1024, TierPlanner.ReservedVramBytes(4 * GiB));
        // 20 GB card: 10% = 2 GB > 512 MB → 2 GB exactly.
        Assert.Equal(2 * GiB, TierPlanner.ReservedVramBytes(20 * GiB));
        // Exact formula parity: Math.Max(v/10, 512 MB).
        Assert.Equal(Math.Max(20 * GiB / 10, 512L * 1024 * 1024),
            TierPlanner.ReservedVramBytes(20 * GiB));
    }

    // ── ParsePlacement / ResolvePlacement ────────────────────────────────────────

    [Theory]
    [InlineData("auto", DSparkPlacement.Auto)]
    [InlineData("gpu", DSparkPlacement.Gpu)]
    [InlineData("cpu", DSparkPlacement.Cpu)]
    [InlineData("off", DSparkPlacement.Off)]
    [InlineData("none", DSparkPlacement.Off)]
    [InlineData("AUTO", DSparkPlacement.Auto)]
    [InlineData(null, DSparkPlacement.Auto)]
    [InlineData("  ", DSparkPlacement.Auto)]
    public void ParsePlacement_KnownValues(string? value, DSparkPlacement expected)
    {
        Assert.Equal(expected, DSparkPlacementPlanner.ParsePlacement(value));
    }

    [Fact]
    public void ParsePlacement_Garbage_Throws()
    {
        Assert.Throws<ArgumentException>(() => DSparkPlacementPlanner.ParsePlacement("banana"));
    }

    [Fact]
    public void ResolvePlacement_FlagBeatsEnv_EnvUsedWhenFlagAbsent()
    {
        const string envName = "SHARPI_DSPARK_PLACE";
        string? original = Environment.GetEnvironmentVariable(envName);
        try
        {
            Environment.SetEnvironmentVariable(envName, "cpu");
            // Explicit flag wins over the env var.
            Assert.Equal(DSparkPlacement.Gpu, DSparkPlacementPlanner.ResolvePlacement("gpu"));
            // Null / whitespace flag falls through to the env var.
            Assert.Equal(DSparkPlacement.Cpu, DSparkPlacementPlanner.ResolvePlacement(null));
            Assert.Equal(DSparkPlacement.Cpu, DSparkPlacementPlanner.ResolvePlacement("   "));

            // Neither flag nor env → Auto.
            Environment.SetEnvironmentVariable(envName, null);
            Assert.Equal(DSparkPlacement.Auto, DSparkPlacementPlanner.ResolvePlacement(null));
        }
        finally
        {
            Environment.SetEnvironmentVariable(envName, original);
        }
    }

    // ── MoE interplay ────────────────────────────────────────────────────────────

    [Fact]
    public void ExpertCacheBudget_CountsAgainstVram()
    {
        // Same 24 GB card / head sizes as the happy GPU case, but the MoE target's
        // SLRU expert-cache budget claims 17 GB of the leftover VRAM: the head no
        // longer fits on GPU and the decision flips to CPU (64 GB RAM, 16 cores).
        var hw = Hw(24 * GiB, 64 * GiB, 16);
        var dense = new LayerPlacement(48, 0, 4 * GiB, 1 * GiB, 8192);
        var moe = dense with { ExpertCacheBudgetBytes = 17 * GiB };

        var withoutBudget = DSparkPlacementPlanner.Plan(hw, dense,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 3 * GiB);
        var withBudget = DSparkPlacementPlanner.Plan(hw, moe,
            draftHeadBytesGpuQuant: 2 * GiB, draftHeadBytesCpuQuant: 3 * GiB);

        Assert.Equal(DSparkPlacement.Gpu, withoutBudget.Placement);
        Assert.Equal(DSparkPlacement.Cpu, withBudget.Placement);
        Assert.Equal(3 * GiB, withBudget.DraftHeadBytes);
    }
}
