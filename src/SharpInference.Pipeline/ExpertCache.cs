namespace SharpInference.Pipeline;

/// <summary>
/// SLRU cache for Mixture-of-Experts expert weights.
/// Maps (layer, expertId) keys to cached values of type <typeparamref name="T"/>.
/// Exploits MoE routing skew (~20% of experts handle ~80% of tokens):
/// frequently accessed experts migrate to the protected segment and resist eviction.
/// </summary>
/// <typeparam name="T">Cached value type (e.g. a GPU tensor handle struct).</typeparam>
public sealed class ExpertCache<T> : IDisposable
{
    private readonly SlruCache<(int Layer, int ExpertId), T> _slru;
    private Action<T>? _onEvict;

    /// <param name="capacity">Total number of expert slots the cache can hold.</param>
    /// <param name="onEvict">
    /// Optional callback invoked with the evicted value so the caller can release
    /// any associated GPU resources.
    /// </param>
    /// <param name="frequencyOf">
    /// Optional per-(layer, expertId) access-count accessor. When supplied, eviction
    /// becomes frequency-aware (least-accessed probationary expert is evicted first),
    /// exploiting MoE routing skew. When null, eviction is plain LRU.
    /// </param>
    public ExpertCache(int capacity, Action<T>? onEvict = null,
        Func<int, int, long>? frequencyOf = null)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        // Split: 25% probationary, 75% protected — biased toward retention since routing is skewed.
        int probCap = Math.Max(1, capacity / 4);
        int protCap = Math.Max(1, capacity - probCap);
        // Both segments floor at 1, so the true max residency is probCap + protCap, which
        // exceeds the requested capacity at capacity == 1 (1 + 1 = 2). Callers that
        // pre-provision per-entry resources (e.g. the CUDA slab's fixed slot pool) must
        // size from this, not the requested capacity, or they under-provision and overflow.
        Capacity = probCap + protCap;
        Func<(int Layer, int ExpertId), long>? freq =
            frequencyOf is null ? null : k => frequencyOf(k.Layer, k.ExpertId);
        _slru = new SlruCache<(int, int), T>(probCap, protCap, freq);
        _onEvict = onEvict;
    }

    /// <summary>
    /// The true maximum number of entries that can be simultaneously resident
    /// (<c>probationary + protected</c> segment capacities). Equals the requested
    /// capacity for capacity ≥ 2, but is 2 for capacity == 1 (both segments floor at 1).
    /// </summary>
    public int Capacity { get; }

    public int Count => _slru.Count;
    public int PinnedCount => _slru.PinnedCount;

    /// <summary>Look up the cached value for (<paramref name="layer"/>, <paramref name="expertId"/>).</summary>
    public bool TryGet(int layer, int expertId, out T value) =>
        _slru.TryGet((layer, expertId), out value);

    /// <summary>
    /// Insert a value for (<paramref name="layer"/>, <paramref name="expertId"/>).
    /// If capacity is exceeded the LRU probationary entry is evicted; the <see cref="onEvict"/>
    /// callback is invoked so callers can release GPU resources.
    /// </summary>
    public void Put(int layer, int expertId, T value)
    {
        if (_slru.Put((layer, expertId), value, out _, out var evicted))
            _onEvict?.Invoke(evicted);
    }

    public bool Contains(int layer, int expertId) =>
        _slru.Contains((layer, expertId));

    /// <summary>
    /// Pin (<paramref name="layer"/>, <paramref name="expertId"/>) so it is never
    /// evicted while resident. The expert must already be cached (call <see cref="Put"/>
    /// first). Use for warm-pinning the hottest experts identified by profiling.
    /// </summary>
    public void Pin(int layer, int expertId) => _slru.Pin((layer, expertId));

    /// <summary>Remove the pin on (<paramref name="layer"/>, <paramref name="expertId"/>).</summary>
    public void Unpin(int layer, int expertId) => _slru.Unpin((layer, expertId));

    public bool IsPinned(int layer, int expertId) => _slru.IsPinned((layer, expertId));

    /// <summary>
    /// Invoke <paramref name="action"/> for every currently-cached value, then clear the cache.
    /// Use this in dispose paths to release GPU resources without skipping the evict callback.
    /// </summary>
    public void Drain(Action<T> action)
    {
        foreach (var value in _slru.Values.ToArray())
            action(value);
        _slru.Clear();
    }

    public void Clear() => _slru.Clear();

    public void Dispose() { }
}
