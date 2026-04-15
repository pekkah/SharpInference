namespace SharpInference.Pipeline;

/// <summary>
/// Segmented LRU (SLRU) cache with probationary and protected segments.
/// New items enter the probationary segment. Items accessed in probationary
/// are promoted to the protected segment, exploiting temporal locality.
/// Eviction always targets the tail of the probationary segment.
/// </summary>
public sealed class SlruCache<TKey, TValue> where TKey : notnull
{
    private readonly record struct Entry(TKey Key, TValue Value);

    private readonly int _probCapacity;
    private readonly int _protCapacity;

    private readonly LinkedList<Entry> _prob = new();
    private readonly LinkedList<Entry> _prot = new();
    private readonly Dictionary<TKey, LinkedListNode<Entry>> _probIndex = new();
    private readonly Dictionary<TKey, LinkedListNode<Entry>> _protIndex = new();

    public int Count => _prob.Count + _prot.Count;
    public int ProbationaryCount => _prob.Count;
    public int ProtectedCount => _prot.Count;

    /// <param name="probationaryCapacity">Slots reserved for newly-inserted (cold) items.</param>
    /// <param name="protectedCapacity">Slots reserved for promoted (hot) items.</param>
    public SlruCache(int probationaryCapacity, int protectedCapacity)
    {
        if (probationaryCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(probationaryCapacity));
        if (protectedCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(protectedCapacity));
        _probCapacity = probationaryCapacity;
        _protCapacity = protectedCapacity;
    }

    /// <summary>Look up <paramref name="key"/>. Promotes probationary hits to protected.</summary>
    /// <returns><c>true</c> if the key was found; <c>false</c> on miss.</returns>
    public bool TryGet(TKey key, out TValue value)
    {
        if (_protIndex.TryGetValue(key, out var protNode))
        {
            _prot.Remove(protNode);
            _prot.AddFirst(protNode);
            value = protNode.Value.Value;
            return true;
        }

        if (_probIndex.TryGetValue(key, out var probNode))
        {
            _prob.Remove(probNode);
            _probIndex.Remove(key);

            // If protected is full, demote its tail to the head of probationary.
            // The probationary count went down by one (we removed probNode), so
            // adding the demoted entry keeps probationary ≤ _probCapacity.
            if (_prot.Count >= _protCapacity)
            {
                var demoted = _prot.Last!.Value;
                _prot.RemoveLast();
                _protIndex.Remove(demoted.Key);
                _prob.AddFirst(demoted);
                _probIndex[demoted.Key] = _prob.First!;
            }

            _prot.AddFirst(probNode.Value);
            _protIndex[key] = _prot.First!;
            value = probNode.Value.Value;
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>
    /// Insert <paramref name="key"/> → <paramref name="value"/> into the probationary segment.
    /// If insertion causes the probationary segment to exceed capacity, the LRU tail is evicted
    /// and returned via <paramref name="evictedKey"/> / <paramref name="evictedValue"/>.
    /// </summary>
    /// <returns><c>true</c> if an entry was evicted; <c>false</c> otherwise.</returns>
    public bool Put(TKey key, TValue value, out TKey evictedKey, out TValue evictedValue)
    {
        _prob.AddFirst(new Entry(key, value));
        _probIndex[key] = _prob.First!;

        if (_prob.Count > _probCapacity)
        {
            var victim = _prob.Last!.Value;
            _probIndex.Remove(victim.Key);
            _prob.RemoveLast();
            evictedKey = victim.Key;
            evictedValue = victim.Value;
            return true;
        }

        evictedKey = default!;
        evictedValue = default!;
        return false;
    }

    public bool Contains(TKey key) =>
        _protIndex.ContainsKey(key) || _probIndex.ContainsKey(key);

    /// <summary>All values currently in the cache (probationary + protected), unordered.</summary>
    public IEnumerable<TValue> Values =>
        _prob.Select(static e => e.Value).Concat(_prot.Select(static e => e.Value));

    public void Clear()
    {
        _prob.Clear();
        _probIndex.Clear();
        _prot.Clear();
        _protIndex.Clear();
    }
}
