namespace SharpInference.Pipeline;

/// <summary>
/// Sizes the GPU expert SLRU cache for a MoE model. Pure, deterministic, and unit-tested
/// so the policy is verifiable without a GPU.
///
/// <para>
/// Capacity is bounded below by 1 and above by the total GPU-layer expert count, and is
/// otherwise the most the VRAM budget allows. It never exceeds the budget, so it cannot
/// cause an OOM the budget wouldn't already imply. Separately we compute a
/// <see cref="MoeCachePlan.RecommendedSlots"/> from the routing-locality finding of
/// "Not All Models Suit Expert Offloading" (arXiv:2505.16056): a cache of roughly
/// <c>2 × active-experts</c> per layer covers a token segment well. When the budget forces
/// capacity below that, callers should warn that hit rate may suffer (fewer GPU layers or
/// more VRAM would help) rather than silently underperform.
/// </para>
/// </summary>
public static class MoeCacheSizing
{
    public static MoeCachePlan Plan(
        int gpuLayers, int numExperts, int numActiveExperts,
        long freeVramBytes, long perExpertBytes, long reserveBytes)
    {
        long total = (long)gpuLayers * numExperts;
        if (total <= 0) return new MoeCachePlan(0, 0, 0);

        long byBudget = perExpertBytes > 0
            ? Math.Max(0, (freeVramBytes - reserveBytes) / perExpertBytes)
            : total;

        // Never exceed the budget or the total; keep at least one slot so the cache works.
        int capacity = (int)Math.Clamp(byBudget, 1, total);

        // Locality sweet spot: ~2× active experts per GPU layer (capped at the full set).
        long recommended = Math.Min(total, (long)gpuLayers * Math.Min(numExperts, 2 * numActiveExperts));

        return new MoeCachePlan(capacity, (int)recommended, (int)Math.Min(byBudget, int.MaxValue));
    }
}

/// <summary>
/// Result of <see cref="MoeCacheSizing.Plan"/>. <paramref name="Slots"/> is the capacity to
/// use; <paramref name="RecommendedSlots"/> is the locality-based target (warn if Slots is
/// materially below it); <paramref name="BudgetSlots"/> is how many slots the VRAM budget
/// alone would allow (for diagnostics).
/// </summary>
public readonly record struct MoeCachePlan(int Slots, int RecommendedSlots, int BudgetSlots);
