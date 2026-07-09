namespace SharpInference.Engine;

/// <summary>Where the DSpark draft head runs (docs/dspark-plan.md §3).</summary>
public enum DSparkPlacement
{
    /// <summary>Planner decides between Gpu/Cpu/Off from measured headroom.</summary>
    Auto,
    /// <summary>Same GPU as the target, own backend instance.</summary>
    Gpu,
    /// <summary>CPU backend, system RAM.</summary>
    Cpu,
    /// <summary>DSpark disabled.</summary>
    Off,
}

/// <summary>
/// The planner's verdict. <paramref name="Reason"/> is a human-readable
/// rationale printed like <see cref="LayerPlacement.Summary"/>;
/// <paramref name="DraftHeadBytes"/> is the head's resident cost in the chosen
/// location; <paramref name="HeadroomBytes"/> the free budget left there after
/// the decision (negative when a user override exceeds it).
/// </summary>
public sealed record DSparkPlacementDecision(
    DSparkPlacement Placement,
    string Reason,
    long DraftHeadBytes,
    long HeadroomBytes);

/// <summary>
/// Static-estimate placement decision for a DSpark draft head (docs/dspark-plan.md §4),
/// mirroring <see cref="TierPlanner"/>'s shape: auto-decide from
/// <see cref="HardwareProfile"/> + the target's <see cref="LayerPlacement"/>, but an
/// explicit user value pins the mode outright (same philosophy as
/// <c>pinGpuLayers</c> — a pin that exceeds the budget is the user's call; the
/// runtime allocation enforces real fit and must re-check free memory right
/// before allocating, downgrading Gpu → Cpu → Off if the static estimate no
/// longer holds).
/// </summary>
public static class DSparkPlacementPlanner
{
    /// <summary>15% margin over the head's weight bytes for its own KV/scratch.</summary>
    private const double ScratchMargin = 1.15;

    /// <summary>Below this core count a CPU draft chain becomes the bottleneck.</summary>
    private const int MinCpuCores = 4;

    /// <param name="hostTapBytes">Host-RAM cost of the TARGET's hidden-tap buffer
    /// (ctx × TapDim × 4 bytes). Resident regardless of where the DRAFT runs, so it
    /// is charged against RAM before both the Gpu and Cpu branches — a Gpu placement
    /// without host headroom for the taps would still OOM the host.</param>
    public static DSparkPlacementDecision Plan(
        HardwareProfile hardware,
        LayerPlacement targetPlacement,
        long draftHeadBytesGpuQuant,
        long draftHeadBytesCpuQuant,
        DSparkPlacement userOverride = DSparkPlacement.Auto,
        long hostTapBytes = 0)
    {
        long vramFree = VramHeadroom(hardware, targetPlacement);
        long ramFreeRaw = RamHeadroom(hardware, targetPlacement);
        long ramFree = Math.Max(0, ramFreeRaw - hostTapBytes);

        // Explicit override skips the budget math entirely (still reports the
        // headroom so the caller can print what it's doing instead of silently
        // OOMing — same contract as the TierPlanner pin).
        switch (userOverride)
        {
            case DSparkPlacement.Off:
                return new DSparkPlacementDecision(DSparkPlacement.Off,
                    "user override: off", 0, 0);
            case DSparkPlacement.Gpu:
                return new DSparkPlacementDecision(DSparkPlacement.Gpu,
                    $"user override: gpu (free VRAM after target ≈ {Mb(vramFree)} MB, " +
                    $"head needs ≈ {Mb((long)(draftHeadBytesGpuQuant * ScratchMargin))} MB)",
                    draftHeadBytesGpuQuant, vramFree - draftHeadBytesGpuQuant);
            case DSparkPlacement.Cpu:
                return new DSparkPlacementDecision(DSparkPlacement.Cpu,
                    $"user override: cpu (free RAM after target ≈ {Mb(ramFree)} MB, " +
                    $"head needs ≈ {Mb((long)(draftHeadBytesCpuQuant * ScratchMargin))} MB)",
                    draftHeadBytesCpuQuant, ramFree - draftHeadBytesCpuQuant);
        }

        // Auto: prefer GPU colocation (no PCIe round-trip per draft step),
        // fall back to CPU when VRAM is tight but RAM and cores allow, else Off.
        // Even a GPU draft needs host RAM for the target's tap buffer.
        long gpuNeed = (long)(draftHeadBytesGpuQuant * ScratchMargin);
        if (hardware.VramBytes > 0 && vramFree >= gpuNeed && ramFreeRaw >= hostTapBytes)
        {
            return new DSparkPlacementDecision(DSparkPlacement.Gpu,
                $"auto: gpu — {Mb(vramFree)} MB free VRAM ≥ {Mb(gpuNeed)} MB needed",
                draftHeadBytesGpuQuant, vramFree - draftHeadBytesGpuQuant);
        }

        long cpuNeed = (long)(draftHeadBytesCpuQuant * ScratchMargin);
        if (ramFree >= cpuNeed && hardware.CpuCores >= MinCpuCores)
        {
            string vramNote = hardware.VramBytes > 0
                ? $"VRAM too tight ({Mb(vramFree)} MB free < {Mb(gpuNeed)} MB), "
                : "no GPU, ";
            return new DSparkPlacementDecision(DSparkPlacement.Cpu,
                $"auto: cpu — {vramNote}{Mb(ramFree)} MB free RAM ≥ {Mb(cpuNeed)} MB needed " +
                $"({hardware.CpuCores} cores)",
                draftHeadBytesCpuQuant, ramFree - draftHeadBytesCpuQuant);
        }

        string why;
        if (ramFree >= cpuNeed && hardware.CpuCores < MinCpuCores)
            why = $"only {hardware.CpuCores} CPU cores (< {MinCpuCores})";
        else if (hardware.VramBytes > 0)
            why = $"VRAM {Mb(vramFree)} MB < {Mb(gpuNeed)} MB and RAM {Mb(ramFree)} MB < {Mb(cpuNeed)} MB";
        else
            why = $"RAM {Mb(ramFree)} MB < {Mb(cpuNeed)} MB";
        // Off has no chosen location, so it reports no headroom.
        return new DSparkPlacementDecision(DSparkPlacement.Off, $"auto: off — {why}", 0, 0);
    }

    /// <summary>Parse a placement string (flag or SHARPI_DSPARK_PLACE): auto|gpu|cpu|off.</summary>
    public static DSparkPlacement ParsePlacement(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return DSparkPlacement.Auto;
        return value.Trim().ToLowerInvariant() switch
        {
            "auto" => DSparkPlacement.Auto,
            "gpu" => DSparkPlacement.Gpu,
            "cpu" => DSparkPlacement.Cpu,
            "off" or "none" or "disabled" => DSparkPlacement.Off,
            _ => throw new ArgumentException(
                $"Unknown DSpark placement '{value}' (expected auto|gpu|cpu|off)."),
        };
    }

    /// <summary>
    /// Resolve the placement with the standard precedence: explicit flag &gt;
    /// SHARPI_DSPARK_PLACE env var &gt; Auto.
    /// </summary>
    public static DSparkPlacement ResolvePlacement(string? flagValue)
    {
        if (!string.IsNullOrWhiteSpace(flagValue)) return ParsePlacement(flagValue);
        return ParsePlacement(Environment.GetEnvironmentVariable("SHARPI_DSPARK_PLACE"));
    }

    private static long VramHeadroom(HardwareProfile hardware, LayerPlacement target)
    {
        if (hardware.VramBytes <= 0) return 0;
        long free = hardware.VramBytes
            - target.GpuWeightBytes
            - target.GpuKvBytes
            - target.ExpertCacheBudgetBytes
            - TierPlanner.ReservedVramBytes(hardware.VramBytes);
        return Math.Max(0, free);
    }

    /// <summary>
    /// System-RAM reserve: 10% of RAM or 2 GB, whichever is larger. Deliberately
    /// NOT <see cref="TierPlanner.ReservedVramBytes"/> — that floor (512 MB) is
    /// tuned for driver/display overhead on a GPU; the OS, mmap page cache, and
    /// engine scratch need more headroom on the RAM side.
    /// </summary>
    private static long ReservedRamBytes(long ramTotal) =>
        Math.Max(ramTotal / 10, 2L * 1024 * 1024 * 1024);

    private static long RamHeadroom(HardwareProfile hardware, LayerPlacement target)
    {
        long free = hardware.RamBytes
            - target.CpuWeightBytes
            - ReservedRamBytes(hardware.RamBytes);
        return Math.Max(0, free);
    }

    private static long Mb(long bytes) => bytes / (1024 * 1024);
}
