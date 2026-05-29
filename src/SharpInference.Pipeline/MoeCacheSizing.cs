namespace SharpInference.Pipeline;

/// <summary>
/// Sizes the GPU expert SLRU cache for a MoE model. Pure, deterministic, and unit-tested
/// so the policy is verifiable without a GPU.
///
/// <para>
/// Capacity is bounded below by 1 (the cache must function even if the budget cannot fit
/// a single expert — callers detect this case via <see cref="MoeCachePlan.Status"/>) and
/// above by the total GPU-layer expert count; otherwise it is the most the VRAM budget
/// allows. Separately we compute a <see cref="MoeCachePlan.RecommendedSlots"/> from the
/// routing-locality finding of "Not All Models Suit Expert Offloading" (arXiv:2505.16056):
/// a cache of roughly <c>2 × active-experts</c> per layer covers a token segment well.
/// When the budget forces capacity below that, callers should warn that hit rate may
/// suffer (fewer GPU layers or more VRAM would help) rather than silently underperform.
/// </para>
/// </summary>
public static class MoeCacheSizing
{
    public static MoeCachePlan Plan(
        int gpuLayers, int numExperts, int numActiveExperts,
        long freeVramBytes, long perExpertBytes, long reserveBytes)
    {
        long total = (long)gpuLayers * numExperts;
        if (total <= 0) return new MoeCachePlan(0, 0, 0, MoeCacheSizingStatus.Empty);

        MoeCacheSizingStatus status;
        long byBudget;
        if (perExpertBytes <= 0)
        {
            // Caller couldn't size an expert (missing tensor, dtype unknown). Falling
            // back to `total` silently would defeat the planner's purpose; surface the
            // condition so the caller can decide whether to abort or proceed unbounded.
            byBudget = total;
            status = MoeCacheSizingStatus.UnknownExpertSize;
        }
        else
        {
            byBudget = Math.Max(0, (freeVramBytes - reserveBytes) / perExpertBytes);
            if (byBudget == 0)
                status = MoeCacheSizingStatus.BudgetExhausted;
            else if (byBudget < (long)gpuLayers * Math.Min(numExperts, 2 * numActiveExperts))
                status = MoeCacheSizingStatus.BelowRecommended;
            else
                status = MoeCacheSizingStatus.Ok;
        }

        // Never exceed the total; keep at least one slot so the cache works. Note:
        // when byBudget==0 (BudgetExhausted) the clamp raises capacity to 1; the
        // Status enum is the caller's only signal that this happened.
        int capacity = (int)Math.Clamp(byBudget, 1, total);

        // Locality sweet spot: ~2× active experts per GPU layer (capped at the full set).
        long recommended = Math.Min(total, (long)gpuLayers * Math.Min(numExperts, 2 * numActiveExperts));

        return new MoeCachePlan(capacity, (int)recommended, (int)Math.Min(byBudget, int.MaxValue), status);
    }
}

/// <summary>
/// Outcome categories for <see cref="MoeCacheSizing.Plan"/>.
/// </summary>
public enum MoeCacheSizingStatus
{
    /// <summary>No GPU layers or no experts — nothing to size.</summary>
    Empty,
    /// <summary>Budget exceeds the locality recommendation; cache fits the working set.</summary>
    Ok,
    /// <summary>Budget fits the cache but below the routing-locality recommendation.</summary>
    BelowRecommended,
    /// <summary>VRAM budget cannot fit even one expert; capacity was clamped to 1.</summary>
    BudgetExhausted,
    /// <summary>Per-expert size unknown (missing tensor); capacity fell back to total.</summary>
    UnknownExpertSize,
}

/// <summary>
/// Result of <see cref="MoeCacheSizing.Plan"/>. <paramref name="Slots"/> is the capacity to
/// use; <paramref name="RecommendedSlots"/> is the locality-based target (warn if Slots is
/// materially below it); <paramref name="BudgetSlots"/> is how many slots the VRAM budget
/// alone would allow (for diagnostics); <paramref name="Status"/> distinguishes
/// "budget fits cache" from the clamped-to-1 and unknown-expert-size edge cases.
/// </summary>
public readonly record struct MoeCachePlan(int Slots, int RecommendedSlots, int BudgetSlots, MoeCacheSizingStatus Status);
