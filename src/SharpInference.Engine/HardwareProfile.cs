namespace SharpInference.Engine;

/// <summary>
/// Auto-detected hardware capabilities for tier placement decisions.
/// </summary>
public sealed record HardwareProfile(
    long VramBytes,
    long RamBytes,
    int CpuCores,
    double EstPcieBandwidthGBps,
    bool HasAvx512)
{
    /// <summary>
    /// Detect hardware capabilities given a CUDA device's VRAM. Used by the CUDA hybrid
    /// path; the underlying placement math only needs total VRAM bytes regardless of
    /// which GPU backend reported them.
    /// </summary>
    public static HardwareProfile Detect(Cuda.CudaBackend gpu) => DetectFromVram((long)gpu.VramBytes);

    /// <summary>
    /// Detect hardware capabilities from the current system and Vulkan device.
    /// </summary>
    public static HardwareProfile Detect(Vulkan.VulkanBackend? gpu = null)
        => DetectFromVram(gpu != null ? (long)gpu.VramBytes : 0);

    private static HardwareProfile DetectFromVram(long vram)
    {

        // System RAM: use GC info as a portable approximation
        var gcInfo = GC.GetGCMemoryInfo();
        long ram = gcInfo.TotalAvailableMemoryBytes;

        int cores = Environment.ProcessorCount;

        // PCIe bandwidth estimate based on VRAM size heuristic:
        // 8-12 GB GPUs: likely PCIe 3.0 or 4.0 (~15-25 GB/s)
        // 16-24 GB GPUs: likely PCIe 4.0 (~25 GB/s)
        // This is a rough estimate; actual measurement would be better.
        double pcieBw = vram switch
        {
            >= 20L * 1024 * 1024 * 1024 => 25.0,  // 20+ GB → likely PCIe 4.0
            >= 10L * 1024 * 1024 * 1024 => 20.0,  // 10-20 GB → PCIe 3.0/4.0 mix
            > 0 => 15.0,                            // <10 GB → likely PCIe 3.0
            _ => 0.0,                                // no GPU
        };

        bool avx512 = System.Runtime.Intrinsics.X86.Avx512F.IsSupported;

        return new HardwareProfile(vram, ram, cores, pcieBw, avx512);
    }

    /// <summary>
    /// Estimate how many KV-cache token positions can be held concurrently in host RAM
    /// before risking out-of-memory, given the per-token cost reported by the forward pass
    /// (<see cref="ForwardPass.KvBytesPerToken"/>). This is the "Phase-0 autotune" budget
    /// used by <see cref="ContinuousBatchingEngine"/> to bound admission: a burst of
    /// long-prompt requests is deferred rather than allowed to allocate unbounded
    /// <see cref="PagedKvCache"/> pages.
    /// </summary>
    /// <param name="kvBytesPerToken">Bytes one KV position costs across all layers (keys+values).</param>
    /// <param name="memoryFraction">
    /// Fraction of currently-available RAM the KV caches may occupy. Defaults to 0.5 to
    /// leave headroom for weights already resident, activations, and the rest of the
    /// process. Clamped to [0, 0.9].
    /// </param>
    /// <returns>
    /// A positive token budget, or <c>0</c> when it cannot be estimated (no RAM figure or
    /// non-positive per-token cost) — callers treat 0 as "unlimited / disabled".
    /// </returns>
    public long EstimateKvTokenBudget(long kvBytesPerToken, double memoryFraction = 0.5)
    {
        if (kvBytesPerToken <= 0 || RamBytes <= 0) return 0;
        double frac = Math.Clamp(memoryFraction, 0.0, 0.9);
        if (frac <= 0.0) return 0;
        long budgetBytes = (long)(RamBytes * frac);
        long tokens = budgetBytes / kvBytesPerToken;
        return tokens > 0 ? tokens : 0;
    }

    public string Summary()
    {
        string vramStr = VramBytes > 0 ? $"{VramBytes / (1024.0 * 1024 * 1024):F1} GB" : "none";
        string ramStr = $"{RamBytes / (1024.0 * 1024 * 1024):F1} GB";
        string isa = HasAvx512 ? "AVX-512" : "AVX2";
        return $"VRAM: {vramStr}, RAM: {ramStr}, {CpuCores} cores ({isa}), PCIe ~{EstPcieBandwidthGBps:F0} GB/s";
    }
}
