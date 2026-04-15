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
    /// Detect hardware capabilities from the current system and Vulkan device.
    /// </summary>
    public static HardwareProfile Detect(Vulkan.VulkanBackend? gpu = null)
    {
        long vram = gpu != null ? (long)gpu.VramBytes : 0;

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

    public string Summary()
    {
        string vramStr = VramBytes > 0 ? $"{VramBytes / (1024.0 * 1024 * 1024):F1} GB" : "none";
        string ramStr = $"{RamBytes / (1024.0 * 1024 * 1024):F1} GB";
        string isa = HasAvx512 ? "AVX-512" : "AVX2";
        return $"VRAM: {vramStr}, RAM: {ramStr}, {CpuCores} cores ({isa}), PCIe ~{EstPcieBandwidthGBps:F0} GB/s";
    }
}
