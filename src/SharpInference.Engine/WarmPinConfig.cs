namespace SharpInference.Engine;

/// <summary>
/// Reads the opt-in warm-pinning configuration from the environment, once.
/// <list type="bullet">
///   <item><c>SHARPI_MOE_WARMPIN</c> — number of hottest experts to pin <i>per layer</i>
///     into the GPU expert cache's protected segment. <c>0</c> (default) leaves warm-pinning
///     to the auto-enable rule below; a positive value forces the override.</item>
///   <item><c>SHARPI_MOE_WARMPIN_AFTER</c> — number of expert accesses to observe
///     before the warm set is chosen (default 512), so pinning reflects real routing
///     rather than the first few cold tokens. Must be &gt; 0 (any positive value);
///     0 / negative / malformed are rejected with a stderr message and the default
///     applies.</item>
/// </list>
/// When the env var is unset, <see cref="ResolvePerLayer"/> auto-enables warm-pinning
/// at <c>NumActiveExperts</c> per layer whenever the SLRU slot capacity is smaller than
/// the total expert count — that is the regime where eviction churn happens and the
/// hot set is worth protecting. If the cache holds the full expert set, warm-pinning
/// is a no-op so we leave it off to avoid wasted work.
/// Shared by <see cref="ExpertSlotManager"/> (Vulkan) and <see cref="CudaExpertSlotManager"/>.
/// Malformed values fall back to the default <i>and</i> log a warning to stderr so
/// typos don't silently turn the feature off.
/// </summary>
internal static class WarmPinConfig
{
    public static readonly int PerLayer = ParseInt("SHARPI_MOE_WARMPIN", 0, allowZero: true);
    public static readonly long AfterAccesses = ParseLong("SHARPI_MOE_WARMPIN_AFTER", 512, allowZero: false);

    /// <summary>
    /// Resolve the effective per-layer warm-pin count for a slot manager. Explicit
    /// <see cref="PerLayer"/> from the environment wins; otherwise auto-enable at
    /// <paramref name="numActiveExperts"/> per layer when the cache cannot hold the
    /// full expert set (and therefore benefits from protecting the hot set).
    /// </summary>
    public static int ResolvePerLayer(int numLayers, int numExperts, int numActiveExperts, int slotCapacity)
    {
        if (PerLayer > 0) return PerLayer;
        long total = (long)numLayers * numExperts;
        if (slotCapacity >= total) return 0;
        // numActiveExperts can be 0 on dense models routed through this path — fall
        // back to 1 so the auto-enable still does something useful, but never more
        // than numExperts (a layer cannot have more pinned experts than it has).
        int perLayer = numActiveExperts > 0 ? numActiveExperts : 1;
        return Math.Min(perLayer, numExperts);
    }

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
