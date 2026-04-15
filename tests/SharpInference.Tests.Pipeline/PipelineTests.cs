
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
}

