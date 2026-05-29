namespace SharpInference.Engine;

/// <summary>
/// Reads the opt-in warm-pinning configuration from the environment, once.
/// <list type="bullet">
///   <item><c>SHARPI_MOE_WARMPIN</c> — number of hottest experts to pin <i>per layer</i>
///     into the GPU expert cache's protected segment. <c>0</c> (default) disables
///     warm-pinning entirely, so behaviour is unchanged unless opted in.</item>
///   <item><c>SHARPI_MOE_WARMPIN_AFTER</c> — number of expert accesses to observe
///     before the warm set is chosen (default 512), so pinning reflects real routing
///     rather than the first few cold tokens. Must be &gt; 0 (any positive value);
///     0 / negative / malformed are rejected with a stderr message and the default
///     applies.</item>
/// </list>
/// Shared by <see cref="ExpertSlotManager"/> (Vulkan) and <see cref="CudaExpertSlotManager"/>.
/// Malformed values fall back to the default <i>and</i> log a warning to stderr so
/// typos don't silently turn the feature off.
/// </summary>
internal static class WarmPinConfig
{
    public static readonly int PerLayer = ParseInt("SHARPI_MOE_WARMPIN", 0, allowZero: true);
    public static readonly long AfterAccesses = ParseLong("SHARPI_MOE_WARMPIN_AFTER", 512, allowZero: false);

    private static int ParseInt(string name, int fallback, bool allowZero)
    {
        var s = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(s)) return fallback;
        if (!int.TryParse(s, out int v) || v < 0 || (!allowZero && v == 0))
        {
            Console.Error.WriteLine(
                $"[WarmPinConfig] {name}='{s}' is not a {(allowZero ? "non-negative" : "positive")} integer; using default {fallback}.");
            return fallback;
        }
        return v;
    }

    private static long ParseLong(string name, long fallback, bool allowZero)
    {
        var s = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(s)) return fallback;
        if (!long.TryParse(s, out long v) || v < 0 || (!allowZero && v == 0))
        {
            Console.Error.WriteLine(
                $"[WarmPinConfig] {name}='{s}' is not a {(allowZero ? "non-negative" : "positive")} integer; using default {fallback}.");
            return fallback;
        }
        return v;
    }
}
