namespace SharpInference.Pipeline;

/// <summary>
/// Segmented LRU (SLRU) cache with probationary and protected segments.
/// New items enter the probationary segment. Items accessed in probationary
/// are promoted to the protected segment, exploiting temporal locality.
/// Eviction targets the probationary segment.
///
/// Two optional refinements (both default-off, so behaviour is plain SLRU
/// unless configured):
/// <list type="bullet">
///   <item><b>Frequency-aware eviction</b> — when a <c>frequencyOf</c> accessor is
///     supplied, the probationary victim is the <i>least-frequently-accessed</i>
///     entry (recency breaks ties), rather than the strict LRU tail. This biases
///     the cache toward keeping hot experts resident under MoE routing skew. The
///     most-recently-inserted entry is never chosen, avoiding the LFU cold-start
///     trap of evicting the item we just loaded.</item>
///   <item><b>Pinning</b> — pinned keys live in the protected segment and are
///     never evicted or demoted while resident, so a warm set of hot experts can
///     be guaranteed in fast memory.</item>
/// </list>
/// </summary>
public sealed class SlruCache<TKey, TValue> where TKey : notnull
{
    private readonly record struct Entry(TKey Key, TValue Value);

    private readonly int _probCapacity;
    private readonly int _protCapacity;
    private readonly Func<TKey, long>? _frequencyOf;

    private readonly LinkedList<Entry> _prob = new();
    private readonly LinkedList<Entry> _prot = new();
    private readonly Dictionary<TKey, LinkedListNode<Entry>> _probIndex = new();
    private readonly Dictionary<TKey, LinkedListNode<Entry>> _protIndex = new();
    private readonly HashSet<TKey> _pinned = new();

    public int Count => _prob.Count + _prot.Count;
    public int ProbationaryCount => _prob.Count;
    public int ProtectedCount => _prot.Count;
    public int PinnedCount => _pinned.Count;

    /// <param name="probationaryCapacity">Slots reserved for newly-inserted (cold) items.</param>
    /// <param name="protectedCapacity">Slots reserved for promoted (hot) items.</param>
    /// <param name="frequencyOf">
    /// Optional access-count accessor enabling frequency-aware eviction. When null,
    /// eviction is plain LRU (probationary tail).
    /// </param>
    public SlruCache(int probationaryCapacity, int protectedCapacity,
        Func<TKey, long>? frequencyOf = null)
    {
        if (probationaryCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(probationaryCapacity));
        if (protectedCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(protectedCapacity));
        _probCapacity = probationaryCapacity;
        _protCapacity = protectedCapacity;
        _frequencyOf = frequencyOf;
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
            value = probNode.Value.Value;

            if (_prot.Count < _protCapacity)
            {
                // Room in protected: promote directly.
                _prob.Remove(probNode);
                _probIndex.Remove(key);
                _prot.AddFirst(probNode.Value);
                _protIndex[key] = _prot.First!;
                return true;
            }

            // Protected is full: demote the tail-most UNPINNED protected entry to
            // make room. The probationary count goes -1 (remove probNode) +1 (demoted
            // in) = unchanged, so it stays ≤ _probCapacity.
            var demoteNode = LastUnpinnedProtected();
            if (demoteNode is not null)
            {
                _prob.Remove(probNode);
                _probIndex.Remove(key);

                var demoted = demoteNode.Value;
                _prot.Remove(demoteNode);
                _protIndex.Remove(demoted.Key);
                _prob.AddFirst(demoted);
                _probIndex[demoted.Key] = _prob.First!;

                _prot.AddFirst(probNode.Value);
                _protIndex[key] = _prot.First!;
            }
            else
            {
                // Every protected slot is pinned — cannot promote. Refresh recency
                // within probationary instead.
                _prob.Remove(probNode);
                _prob.AddFirst(probNode.Value);
                _probIndex[key] = _prob.First!;
            }
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>Tail-most protected node that is not pinned, or null if all are pinned.</summary>
    private LinkedListNode<Entry>? LastUnpinnedProtected()
    {
        for (var node = _prot.Last; node is not null; node = node.Previous)
            if (!_pinned.Contains(node.Value.Key)) return node;
        return null;
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
            var victim = SelectProbationaryVictim();
            _probIndex.Remove(victim.Value.Key);
            _prob.Remove(victim);
            _pinned.Remove(victim.Value.Key); // defensive; pinned entries live in protected
            evictedKey = victim.Value.Key;
            evictedValue = victim.Value.Value;
            return true;
        }

        evictedKey = default!;
        evictedValue = default!;
        return false;
    }

    /// <summary>
    /// Choose which probationary entry to evict. Never the most-recently-inserted
    /// entry (<see cref="LinkedList{T}.First"/>) and never a pinned entry. With a
    /// frequency accessor, picks the least-frequently-accessed candidate (older
    /// entry breaks ties); otherwise the LRU tail.
    /// </summary>
    private LinkedListNode<Entry> SelectProbationaryVictim()
    {
        // Walk from tail (oldest) toward head, skipping the just-inserted head and
        // any pinned entries.
        LinkedListNode<Entry>? best = null;
        long bestFreq = long.MaxValue;
        for (var node = _prob.Last; node is not null && node != _prob.First; node = node.Previous)
        {
            if (_pinned.Contains(node.Value.Key)) continue;
            if (_frequencyOf is null)
                return node; // first unpinned from the tail == LRU victim

            long freq = _frequencyOf(node.Value.Key);
            if (freq < bestFreq)
            {
                bestFreq = freq;
                best = node;
            }
        }
        // Fallback when every other entry is pinned: evict the tail regardless.
        return best ?? _prob.Last!;
    }

    /// <summary>
    /// Pin <paramref name="key"/> so it is never evicted or demoted while resident.
    /// Pinned entries are moved into the protected segment. No-op if the key is not
    /// currently resident (load it via <see cref="Put"/> first) or already pinned.
    /// </summary>
    public void Pin(TKey key)
    {
        if (_pinned.Contains(key)) return;

        if (_protIndex.ContainsKey(key))
        {
            _pinned.Add(key);
            return;
        }

        if (_probIndex.TryGetValue(key, out var node))
        {
            _prob.Remove(node);
            _probIndex.Remove(key);
            // Demote an unpinned protected tail entry if protected is full so the
            // pinned entry has a home; if all are pinned it simply grows protected
            // (bounded by the caller pinning ≤ protected capacity).
            if (_prot.Count >= _protCapacity)
            {
                var demote = LastUnpinnedProtected();
                if (demote is not null)
                {
                    var demoted = demote.Value;
                    _prot.Remove(demote);
                    _protIndex.Remove(demoted.Key);
                    _prob.AddFirst(demoted);
                    _probIndex[demoted.Key] = _prob.First!;
                }
            }
            _prot.AddFirst(node.Value);
            _protIndex[key] = _prot.First!;
            _pinned.Add(key);
        }
    }

    /// <summary>Remove the pin on <paramref name="key"/> (entry stays resident in protected).</summary>
    public void Unpin(TKey key) => _pinned.Remove(key);

    public bool IsPinned(TKey key) => _pinned.Contains(key);

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
        _pinned.Clear();
    }
}
