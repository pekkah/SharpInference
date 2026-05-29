namespace SharpInference.Engine;

/// <summary>
/// Reads the opt-in warm-pinning configuration from the environment, once.
/// <list type="bullet">
///   <item><c>SHARPI_MOE_WARMPIN</c> — number of hottest experts to pin <i>per layer</i>
///     into the GPU expert cache's protected segment. <c>0</c> (default) disables
///     warm-pinning entirely, so behaviour is unchanged unless opted in.</item>
///   <item><c>SHARPI_MOE_WARMPIN_AFTER</c> — number of expert accesses to observe
///     before the warm set is chosen (default 512), so pinning reflects real routing
///     rather than the first few cold tokens.</item>
/// </list>
/// Shared by <see cref="ExpertSlotManager"/> (Vulkan) and <see cref="CudaExpertSlotManager"/>.
/// </summary>
internal static class WarmPinConfig
{
    public static readonly int PerLayer = ParseInt("SHARPI_MOE_WARMPIN", 0);
    public static readonly long AfterAccesses = ParseLong("SHARPI_MOE_WARMPIN_AFTER", 512);

    private static int ParseInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= 0 ? v : fallback;

    private static long ParseLong(string name, long fallback) =>
        long.TryParse(Environment.GetEnvironmentVariable(name), out long v) && v > 0 ? v : fallback;
}
