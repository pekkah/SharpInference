using SharpInference.Core;

namespace SharpInference.Pipeline;

/// <summary>
/// LRU cache for Mixture-of-Experts expert weights.
/// Tracks access frequency across recent tokens to keep hot experts GPU-resident.
/// </summary>
public sealed class ExpertCache : IDisposable
{
    private readonly int _capacity;
    private readonly Dictionary<int, CachedExpert> _cache = [];
    private readonly LinkedList<int> _lruList = [];

    public ExpertCache(int capacity) => _capacity = capacity;

    public bool TryGet(int expertId, out Tensor? weights)
    {
        if (_cache.TryGetValue(expertId, out var entry))
        {
            _lruList.Remove(entry.LruNode);
            _lruList.AddFirst(entry.LruNode);
            weights = entry.Weights;
            return true;
        }
        weights = null;
        return false;
    }

    public void Put(int expertId, Tensor weights)
    {
        // TODO: evict LRU expert if at capacity
        throw new NotImplementedException();
    }

    public void Dispose() { }

    private sealed class CachedExpert
    {
        public required Tensor Weights { get; init; }
        public required LinkedListNode<int> LruNode { get; init; }
    }
}
