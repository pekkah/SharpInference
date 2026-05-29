
namespace SharpInference.Tests.Pipeline;

public sealed class PipelineTests
{
    [Fact]
    public void ExpertCache_Miss_ReturnsFalse()
    {
        var cache = new SharpInference.Pipeline.ExpertCache<int>(capacity: 4);
        Assert.False(cache.TryGet(0, 0, out _));
        cache.Dispose();
    }

    [Fact]
    public void ExpertCache_PutAndGet_ReturnsValue()
    {
        var cache = new SharpInference.Pipeline.ExpertCache<string>(capacity: 4);
        cache.Put(layer: 0, expertId: 3, "hello");
        Assert.True(cache.TryGet(0, 3, out var value));
        Assert.Equal("hello", value);
    }

    [Fact]
    public void ExpertCache_EvictsWhenFull()
    {
        string? evicted = null;
        var cache = new SharpInference.Pipeline.ExpertCache<string>(capacity: 2, onEvict: v => evicted = v);
        cache.Put(0, 0, "a");
        cache.Put(0, 1, "b");
        cache.Put(0, 2, "c"); // should evict "a" (LRU)
        Assert.NotNull(evicted);
    }

    [Fact]
    public void ExpertCache_DifferentLayersSameExpertId_TrackedSeparately()
    {
        var cache = new SharpInference.Pipeline.ExpertCache<int>(capacity: 8);
        cache.Put(0, 0, 100);
        cache.Put(1, 0, 200);
        Assert.True(cache.TryGet(0, 0, out var v0)); Assert.Equal(100, v0);
        Assert.True(cache.TryGet(1, 0, out var v1)); Assert.Equal(200, v1);
    }

    [Fact]
    public void SlruCache_ProbationaryHitPromotesToProtected()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 2);
        cache.Put(1, "a", out _, out _);
        Assert.Equal(1, cache.ProbationaryCount);
        Assert.Equal(0, cache.ProtectedCount);
        cache.TryGet(1, out _); // promote
        Assert.Equal(0, cache.ProbationaryCount);
        Assert.Equal(1, cache.ProtectedCount);
    }

    [Fact]
    public void SlruCache_EvictsFromProbationaryWhenFull()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 2);
        cache.Put(1, "a", out _, out _);
        cache.Put(2, "b", out _, out _);
        bool evicted = cache.Put(3, "c", out int evKey, out string evVal);
        Assert.True(evicted);
        Assert.Equal("a", evVal); // oldest item evicted
    }

    [Fact]
    public void SlruCache_ProtectedSurvivesEviction()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 2);
        cache.Put(1, "a", out _, out _);
        cache.TryGet(1, out _); // promote 1 to protected
        cache.Put(2, "b", out _, out _);
        cache.Put(3, "c", out _, out _);
        bool evicted = cache.Put(4, "d", out _, out var evVal);
        Assert.True(evicted);
        // "a" is in protected and should NOT be evicted; "b" or "c" is evicted
        Assert.NotEqual("a", evVal);
    }

    [Fact]
    public void ExpertAccessProfiler_RecordsHitsAndMisses()
    {
        var profiler = new SharpInference.Pipeline.ExpertAccessProfiler(numLayers: 2, numExperts: 4);
        profiler.RecordHit(0, 1);
        profiler.RecordHit(0, 1);
        profiler.RecordMiss(0, 2);
        Assert.Equal(2, profiler.TotalHits);
        Assert.Equal(1, profiler.TotalMisses);
        Assert.InRange(profiler.GetLayerHitRate(0), 0.66, 0.68);
    }

    [Fact]
    public void ExpertAccessProfiler_TopExperts_OrderedByAccess()
    {
        var profiler = new SharpInference.Pipeline.ExpertAccessProfiler(numLayers: 1, numExperts: 4);
        profiler.RecordHit(0, 3); profiler.RecordHit(0, 3); profiler.RecordHit(0, 3);
        profiler.RecordMiss(0, 1);
        profiler.RecordHit(0, 2); profiler.RecordHit(0, 2);
        int[] top = profiler.GetTopExperts(layer: 0, n: 2);
        Assert.Equal(3, top[0]); // most accesses (3)
        Assert.Equal(2, top[1]); // second most accesses (2)
    }

    [Fact]
    public void ExpertAccessProfiler_GetAccessCount_SumsHitsAndMisses()
    {
        var profiler = new SharpInference.Pipeline.ExpertAccessProfiler(numLayers: 2, numExperts: 4);
        profiler.RecordHit(1, 2);
        profiler.RecordHit(1, 2);
        profiler.RecordMiss(1, 2);
        Assert.Equal(3, profiler.GetAccessCount(1, 2));
        Assert.Equal(0, profiler.GetAccessCount(0, 2)); // different layer untouched
    }

    // ── Frequency-aware eviction ───────────────────────────────────────────

    [Fact]
    public void SlruCache_FrequencyAware_EvictsLeastAccessed_NotLruTail()
    {
        // freq accessor: key 1 is hot, keys 2 and 3 are cold.
        var freq = new System.Collections.Generic.Dictionary<int, long> { [1] = 100, [2] = 1, [3] = 1 };
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(
            probationaryCapacity: 3, protectedCapacity: 1, frequencyOf: k => freq.GetValueOrDefault(k));
        cache.Put(1, "hot", out _, out _);   // tail-most (oldest) but highest freq
        cache.Put(2, "cold-a", out _, out _);
        cache.Put(3, "cold-b", out _, out _);
        // Insert a 4th → probationary overflows. Plain LRU would evict key 1 (oldest);
        // frequency-aware keeps the hot key 1 and evicts the least-accessed older entry (key 2).
        bool evicted = cache.Put(4, "new", out int evKey, out _);
        Assert.True(evicted);
        Assert.Equal(2, evKey);
        Assert.True(cache.Contains(1)); // hot survived despite being oldest
    }

    [Fact]
    public void SlruCache_FrequencyAware_NeverEvictsJustInsertedEntry()
    {
        // The just-inserted entry has frequency 0 (coldest) but must not be evicted.
        var freq = new System.Collections.Generic.Dictionary<int, long> { [1] = 5, [2] = 5 };
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(
            probationaryCapacity: 2, protectedCapacity: 1, frequencyOf: k => freq.GetValueOrDefault(k));
        cache.Put(1, "a", out _, out _);
        cache.Put(2, "b", out _, out _);
        bool evicted = cache.Put(99, "fresh", out int evKey, out _); // freq(99)=0
        Assert.True(evicted);
        Assert.NotEqual(99, evKey);       // the fresh insert is protected from immediate eviction
        Assert.True(cache.Contains(99));
    }

    // ── Pinning ────────────────────────────────────────────────────────────

    [Fact]
    public void SlruCache_Pin_MovesToProtectedAndSurvivesEviction()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 2);
        cache.Put(1, "pinme", out _, out _);
        cache.Pin(1);
        Assert.True(cache.IsPinned(1));
        Assert.Equal(1, cache.ProtectedCount);     // pinning moved it to protected
        Assert.Equal(1, cache.PinnedCount);

        // Churn probationary hard; pinned key 1 must never be evicted.
        for (int k = 10; k < 30; k++)
            cache.Put(k, $"v{k}", out _, out _);
        Assert.True(cache.Contains(1));
    }

    [Fact]
    public void SlruCache_Pin_NotEvictedAndNotChosenAsVictim()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 1);
        cache.Put(1, "a", out _, out _);
        cache.Pin(1);                                  // → protected, pinned
        cache.Put(2, "b", out _, out _);
        cache.Put(3, "c", out _, out _);
        bool evicted = cache.Put(4, "d", out int evKey, out _); // prob overflow (2,3,4) → evict one
        Assert.True(evicted);
        Assert.NotEqual(1, evKey);                     // pinned never the victim
        Assert.True(cache.Contains(1));
    }

    [Fact]
    public void SlruCache_Unpin_AllowsEvictionAgain()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 1, protectedCapacity: 1);
        cache.Put(1, "a", out _, out _);
        cache.Pin(1);
        Assert.True(cache.IsPinned(1));
        cache.Unpin(1);
        Assert.False(cache.IsPinned(1));
        Assert.Equal(0, cache.PinnedCount);
    }

    [Fact]
    public void SlruCache_Pin_NonResidentKey_IsNoOp()
    {
        var cache = new SharpInference.Pipeline.SlruCache<int, string>(probationaryCapacity: 2, protectedCapacity: 2);
        cache.Pin(42); // not resident
        Assert.False(cache.IsPinned(42));
        Assert.Equal(0, cache.PinnedCount);
    }

    [Fact]
    public void ExpertCache_Pin_KeepsHotExpertResidentUnderChurn()
    {
        var evicted = new System.Collections.Generic.List<string>();
        var cache = new SharpInference.Pipeline.ExpertCache<string>(capacity: 4, onEvict: evicted.Add);
        cache.Put(0, 7, "hot");
        cache.Pin(0, 7);
        Assert.True(cache.IsPinned(0, 7));
        for (int e = 100; e < 130; e++)
            cache.Put(0, e, $"e{e}");
        Assert.True(cache.TryGet(0, 7, out var v));
        Assert.Equal("hot", v);
        Assert.DoesNotContain("hot", evicted);
        cache.Dispose();
    }

    [Fact]
    public void ExpertCache_FrequencyAware_EvictsLeastAccessedExpert()
    {
        var accesses = new System.Collections.Generic.Dictionary<(int, int), long>
        {
            [(0, 1)] = 50, [(0, 2)] = 1, [(0, 3)] = 1,
        };
        string? evicted = null;
        var cache = new SharpInference.Pipeline.ExpertCache<string>(
            capacity: 4, onEvict: v => evicted = v,
            frequencyOf: (l, e) => accesses.GetValueOrDefault((l, e)));
        // capacity 4 → probCap=1, protCap=3. Force probationary overflow.
        cache.Put(0, 1, "hot");
        cache.Put(0, 2, "cold-a");
        cache.Put(0, 3, "cold-b");
        cache.Put(0, 4, "new");   // overflow; least-accessed older non-head evicted
        Assert.NotNull(evicted);
        Assert.NotEqual("hot", evicted); // the hot expert is retained
    }

    // ── Predictive prefetch: ExpertRoutePredictor ───────────────────────────

    [Fact]
    public void ExpertRoutePredictor_UnseenLayer_PredictsNothing()
    {
        var p = new SharpInference.Pipeline.ExpertRoutePredictor(numLayers: 4, maxActiveExperts: 8);
        Assert.False(p.TryPredict(0, out _));
    }

    [Fact]
    public void ExpertRoutePredictor_RecallsLastSelection()
    {
        var p = new SharpInference.Pipeline.ExpertRoutePredictor(numLayers: 4, maxActiveExperts: 8);
        p.Record(2, stackalloc int[] { 5, 9, 13 });
        Assert.True(p.TryPredict(2, out var pred));
        Assert.Equal(new[] { 5, 9, 13 }, pred.ToArray());
        Assert.False(p.TryPredict(3, out _)); // independent per layer
    }

    [Fact]
    public void ExpertRoutePredictor_LatestRecordWins()
    {
        var p = new SharpInference.Pipeline.ExpertRoutePredictor(numLayers: 2, maxActiveExperts: 4);
        p.Record(0, stackalloc int[] { 1, 2 });
        p.Record(0, stackalloc int[] { 7, 8, 9 }); // next token's selection replaces
        Assert.True(p.TryPredict(0, out var pred));
        Assert.Equal(new[] { 7, 8, 9 }, pred.ToArray());
    }

    [Fact]
    public void ExpertRoutePredictor_Reset_ClearsHistory()
    {
        var p = new SharpInference.Pipeline.ExpertRoutePredictor(numLayers: 2, maxActiveExperts: 4);
        p.Record(1, stackalloc int[] { 3 });
        p.Reset();
        Assert.False(p.TryPredict(1, out _));
    }

    [Fact]
    public void ExpertRoutePredictor_ClampsToMaxActive()
    {
        var p = new SharpInference.Pipeline.ExpertRoutePredictor(numLayers: 1, maxActiveExperts: 2);
        p.Record(0, stackalloc int[] { 4, 5, 6, 7 }); // more than maxActive
        Assert.True(p.TryPredict(0, out var pred));
        Assert.Equal(2, pred.Length);
        Assert.Equal(new[] { 4, 5 }, pred.ToArray());
    }
}

