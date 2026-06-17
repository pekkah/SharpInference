using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for <see cref="CudaExpertSlotManager"/> — the CUDA port of the
/// Vulkan SLRU expert cache. Tests skip silently when CUDA is unavailable or
/// when the required MoE GGUF isn't on disk, same pattern as
/// <see cref="CudaMoeTests"/>.
/// </summary>
public sealed class CudaExpertSlotManagerTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    /// <summary>
    /// Probe upwards from the current directory for a candidate MoE GGUF path.
    /// Returns the first existing match (relative paths) or the first existing
    /// absolute path. Mirrors <see cref="CudaMoeTests.FindMoEModelPath"/>.
    /// </summary>
    private static string? FindFirstExisting(params string[] candidates)
    {
        // Absolute paths: probe directly.
        foreach (var c in candidates)
        {
            if (Path.IsPathRooted(c) && File.Exists(c)) return c;
        }

        // Relative paths: walk up from CWD looking for a match.
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in candidates)
            {
                if (Path.IsPathRooted(c)) continue;
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Cache miss/hit accounting and eviction behaviour against a small MoE
    /// model (OLMoE) — fits a 12 GB card easily, runs on any CUDA setup with
    /// the GGUF on disk. Verifies:
    ///   • The same (layer, expertId) the second time hits cache (no miss recorded).
    ///   • 5 distinct entries with capacity 4 forces 1 eviction; cache size stays ≤ 4.
    ///   • TryGetCached returns true exactly for currently-resident entries.
    /// </summary>
    [Fact]
    public void GetOrLoad_HitMissEvictionAccounting_OnSmallMoE()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindFirstExisting("models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return;

        var dtypes = new Dictionary<nint, DType>();
        using var slots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: 4, dtypes);

        // First GetOrLoad for (0, 0): miss.
        var slot00a = slots.GetOrLoad(layer: 0, expertId: 0);
        Assert.Equal(1, slots.Profiler.TotalMisses);
        Assert.Equal(0, slots.Profiler.TotalHits);

        // Second call for (0, 0): hit.
        var slot00b = slots.GetOrLoad(layer: 0, expertId: 0);
        Assert.Equal(1, slots.Profiler.TotalMisses);
        Assert.Equal(1, slots.Profiler.TotalHits);
        Assert.Equal(slot00a.Gate.Handle, slot00b.Gate.Handle);
        Assert.Equal(slot00a.Up.Handle, slot00b.Up.Handle);
        Assert.Equal(slot00a.Down.Handle, slot00b.Down.Handle);

        // dtypes map should now have 3 entries (gate, up, down) for the one slot.
        Assert.Equal(3, dtypes.Count);

        // Load 4 more distinct entries — capacity is 4, so this forces exactly 1
        // eviction (the first (0, 0) entry is the LRU victim, assuming SLRU
        // probationary admission). Even if the SLRU choice differs, the
        // invariant "cache size ≤ capacity" must hold.
        slots.GetOrLoad(layer: 0, expertId: 1);
        slots.GetOrLoad(layer: 0, expertId: 2);
        slots.GetOrLoad(layer: 0, expertId: 3);
        slots.GetOrLoad(layer: 0, expertId: 4); // total seen: 5 distinct, capacity 4

        // Five misses total (one per distinct key); one hit (the repeat of (0,0)).
        Assert.Equal(5, slots.Profiler.TotalMisses);
        Assert.Equal(1, slots.Profiler.TotalHits);

        // Count residency via TryGetCached across all 5 keys we touched.
        int resident = 0;
        for (int e = 0; e <= 4; e++)
        {
            if (slots.TryGetCached(0, e, out _)) resident++;
        }
        Assert.True(resident <= 4,
            $"Cache held {resident} entries with slotCapacity=4; eviction did not fire.");
        Assert.True(resident >= 1, "Cache should retain at least one entry.");

        // dtypes map size mirrors residency × 3 tensors per slot.
        Assert.Equal(resident * 3, dtypes.Count);
    }

    /// <summary>
    /// Issue #216 — exact-size slab: the expert cache VRAM footprint equals
    /// <c>(slotCapacity + 1) × per-expert bytes</c> (the +1 is the eviction-staging slot)
    /// and does NOT grow as more distinct experts than capacity churn through, proving the
    /// slab is preallocated once and slots are reused (no per-eviction cudaMalloc/cudaFree).
    /// </summary>
    [Fact]
    public void ExpertCacheVram_IsExactSizeSlab_AndStableUnderChurn()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindFirstExisting("models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return;

        const int cap = 4;
        var dtypes = new Dictionary<nint, DType>();
        using var slots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: cap, dtypes);

        // Before any upload the slabs are unallocated.
        Assert.Equal(0, slots.ExpertCacheVramBytes);

        // A role's slab stride is the MAX per-expert footprint across ALL layers (raw bytes
        // for Q4_K/Q5_K/Q6_K, else F32). OLMoE Q4_K_M (like Coder/35B) stores ffn_down as
        // Q6_K on a subset of layers and Q4_K/Q5_K on the rest, so sizing from blk.0 alone
        // would under-size the slab and overflow when a larger-quant expert is routed —
        // issue #216's mixed-quant regression. The expectation must mirror the max sizing.
        long MaxRoleBytes(string roleSuffix, int rows, int cols)
        {
            long max = 0;
            for (int l = 0; l < hp.NumLayers; l++)
            {
                if (model.FindTensor($"blk.{l}.{roleSuffix}.weight") is not { } info) continue;
                long b = info.DType is DType.Q4_K or DType.Q5_K or DType.Q6_K
                    ? (long)rows * (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType)
                    : (long)rows * cols * sizeof(float);
                max = Math.Max(max, b);
            }
            return max;
        }
        long perExpert =
            MaxRoleBytes("ffn_gate_exps", hp.ExpertIntermediateDim, hp.EmbeddingDim) +
            MaxRoleBytes("ffn_up_exps",   hp.ExpertIntermediateDim, hp.EmbeddingDim) +
            MaxRoleBytes("ffn_down_exps", hp.EmbeddingDim, hp.ExpertIntermediateDim);
        long expectedSlab = perExpert * (cap + 1); // slotCapacity >= 2 → slab slots == cap + 1

        // First load allocates all three slabs at full (cap+1) stride.
        slots.GetOrLoad(0, 0);
        Assert.Equal(expectedSlab, slots.ExpertCacheVramBytes);

        // Churn distinct experts ACROSS LAYERS (not just layer 0) so the larger-quant ffn_down
        // layers actually exercise the slab. Before the mixed-quant fix this threw
        // "source exceeds destination capacity" the first time a Q6_K expert hit a Q4_K-seeded
        // slot. Footprint must stay constant: slab preallocated once, slots recycled, no growth.
        for (int l = 0; l < hp.NumLayers; l++)
            for (int e = 0; e < 6; e++)
                slots.GetOrLoad(l, e);

        Assert.Equal(expectedSlab, slots.ExpertCacheVramBytes);
    }

    /// <summary>
    /// Issue #216 regression: <c>slotCapacity == 1</c> must not over-pop the slab's fixed
    /// free-slot pool. <see cref="ExpertCache{T}"/> floors both SLRU segments at 1, so a
    /// capacity-1 cache can hold 2 resident entries and the insert-before-evict transient
    /// momentarily needs a 3rd slot. Provisioning only (slotCapacity + 1) = 2 slices crashed
    /// decode with <c>InvalidOperationException</c> ("Pop on empty stack"). The fix sizes the
    /// pool from the SLRU's true residency + 1. A capacity-1 slot count is reachable in
    /// production via <c>MoeCacheSizing.Plan</c>'s BudgetExhausted clamp under VRAM pressure.
    /// </summary>
    [Fact]
    public void ExpertCacheCapacityOne_ChurnDoesNotOverflowFreeSlots()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindFirstExisting("models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return;

        var dtypes = new Dictionary<nint, DType>();
        using var slots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: 1, dtypes);

        // Promote expert 0 into the protected segment (a hit), so it survives every later Put;
        // then churn many distinct experts. The protected entry forces the transient peak onto
        // the 3rd slot that the under-provisioned pool lacked → the exact crashing sequence.
        slots.GetOrLoad(0, 0);
        slots.GetOrLoad(0, 0); // hit → promote to protected
        for (int e = 1; e <= 12; e++)
            slots.GetOrLoad(0, e); // must not throw (empty-stack Pop)

        // Residency stays within the SLRU's true capacity (2 for slotCapacity 1).
        int resident = 0;
        for (int e = 0; e <= 12; e++)
            if (slots.TryGetCached(0, e, out _)) resident++;
        Assert.True(resident is >= 1 and <= 2,
            $"cap=1 SLRU held {resident} entries (expected 1..2).");
    }

    /// <summary>
    /// Same accounting check against the 22 GB qwen35moe model, only runs when
    /// the file is present at the expected path on this machine. This is the
    /// model the CUDA SLRU is actually intended for — it cannot fit eagerly on
    /// a 12 GB card and SLRU eviction is the path to making it work.
    /// </summary>
    [Fact]
    public void GetOrLoad_HitMissEvictionAccounting_OnQwen35Moe()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindFirstExisting("E:\\models\\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE) return;

        var dtypes = new Dictionary<nint, DType>();
        using var slots = new CudaExpertSlotManager(gpu, model, hp, slotCapacity: 4, dtypes);

        // (0, 0) miss then hit.
        var first = slots.GetOrLoad(layer: 0, expertId: 0);
        var second = slots.GetOrLoad(layer: 0, expertId: 0);
        Assert.Equal(first.Gate.Handle, second.Gate.Handle);
        Assert.Equal(1, slots.Profiler.TotalMisses);
        Assert.Equal(1, slots.Profiler.TotalHits);

        // Five distinct (layer, expertId) entries with capacity 4 → ≥1 eviction.
        slots.GetOrLoad(0, 1);
        slots.GetOrLoad(0, 2);
        slots.GetOrLoad(0, 3);
        slots.GetOrLoad(0, 4);

        Assert.Equal(5, slots.Profiler.TotalMisses);
        Assert.Equal(1, slots.Profiler.TotalHits);

        int resident = 0;
        for (int e = 0; e <= 4; e++)
        {
            if (slots.TryGetCached(0, e, out _)) resident++;
        }
        Assert.True(resident <= 4,
            $"Cache held {resident} entries with slotCapacity=4; eviction did not fire.");
        Assert.Equal(resident * 3, dtypes.Count);
    }
}
