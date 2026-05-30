using System.Text;

namespace SharpInference.Pipeline;

/// <summary>
/// Tracks per-(layer, expert) hit and miss counts for the expert slot cache.
/// Used to identify hot experts and measure cache effectiveness.
/// All counter updates are thread-safe via Interlocked.
/// </summary>
public sealed class ExpertAccessProfiler
{
    private readonly int _numLayers;
    private readonly int _numExperts;
    private readonly long[] _hits;   // [layer * numExperts + expertId]
    private readonly long[] _misses; // same layout
    private long _totalHits;
    private long _totalMisses;

    // Warm-pin snapshot. Populated once when the slot manager pins the hot set;
    // PrintStats uses these to compute before/after-warm hit rates so callers can
    // see whether warm-pinning actually improved cache effectiveness.
    private long _preWarmHits;
    private long _preWarmMisses;
    private (int Layer, int ExpertId)[]? _warmPinned;

    public ExpertAccessProfiler(int numLayers, int numExperts)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        _numLayers = numLayers;
        _numExperts = numExperts;
        _hits = new long[numLayers * numExperts];
        _misses = new long[numLayers * numExperts];
    }

    public void RecordHit(int layer, int expertId)
    {
        Interlocked.Increment(ref _hits[layer * _numExperts + expertId]);
        Interlocked.Increment(ref _totalHits);
    }

    public void RecordMiss(int layer, int expertId)
    {
        Interlocked.Increment(ref _misses[layer * _numExperts + expertId]);
        Interlocked.Increment(ref _totalMisses);
    }

    public long TotalHits => Interlocked.Read(ref _totalHits);
    public long TotalMisses => Interlocked.Read(ref _totalMisses);

    public double OverallHitRate
    {
        get
        {
            long total = TotalHits + TotalMisses;
            return total == 0 ? 0.0 : (double)TotalHits / total;
        }
    }

    /// <summary>Hit rate for a specific layer (all experts combined).</summary>
    public double GetLayerHitRate(int layer)
    {
        long hits = 0, misses = 0;
        int offset = layer * _numExperts;
        for (int e = 0; e < _numExperts; e++)
        {
            hits += Interlocked.Read(ref _hits[offset + e]);
            misses += Interlocked.Read(ref _misses[offset + e]);
        }
        return (hits + misses) == 0 ? 0.0 : (double)hits / (hits + misses);
    }

    /// <summary>
    /// Total access count (hits + misses) for one expert. Used as the popularity
    /// signal for frequency-aware cache eviction.
    /// </summary>
    public long GetAccessCount(int layer, int expertId)
    {
        int i = layer * _numExperts + expertId;
        return Interlocked.Read(ref _hits[i]) + Interlocked.Read(ref _misses[i]);
    }

    /// <summary>
    /// Total access count for a whole layer (sum across experts). Used to rank
    /// layers by hotness so warm-pinning budgets the hottest layers first instead
    /// of iterating in layer-index order (which biases pins to low-index layers
    /// for hybrid GDN+MoE models that cluster MoE FFN at high indices).
    /// </summary>
    public long GetLayerAccessCount(int layer)
    {
        long total = 0;
        int offset = layer * _numExperts;
        for (int e = 0; e < _numExperts; e++)
        {
            total += Interlocked.Read(ref _hits[offset + e]);
            total += Interlocked.Read(ref _misses[offset + e]);
        }
        return total;
    }

    /// <summary>
    /// Returns the <paramref name="n"/> most-accessed expert IDs for <paramref name="layer"/>,
    /// sorted descending by total access count (hits + misses).
    /// </summary>
    public int[] GetTopExperts(int layer, int n)
    {
        int offset = layer * _numExperts;
        var counts = new (int expertId, long count)[_numExperts];
        for (int e = 0; e < _numExperts; e++)
        {
            long total = Interlocked.Read(ref _hits[offset + e])
                       + Interlocked.Read(ref _misses[offset + e]);
            counts[e] = (e, total);
        }
        Array.Sort(counts, static (a, b) => b.count.CompareTo(a.count));
        int take = Math.Min(n, _numExperts);
        var result = new int[take];
        for (int i = 0; i < take; i++) result[i] = counts[i].expertId;
        return result;
    }

    /// <summary>
    /// Snapshot the current hit/miss totals and remember the warm-pin set for later
    /// reporting. Called once by the slot manager immediately after it pins hot
    /// experts so <see cref="PrintStats"/> can split before-warm vs post-warm hit
    /// rates and show which experts were pinned. Safe to call from any thread, but
    /// expected to fire exactly once over the lifetime of the profiler.
    /// </summary>
    public void RecordWarmPin(ReadOnlySpan<(int Layer, int ExpertId)> warmPinned)
    {
        _preWarmHits = Interlocked.Read(ref _totalHits);
        _preWarmMisses = Interlocked.Read(ref _totalMisses);
        _warmPinned = warmPinned.ToArray();
    }

    /// <summary>
    /// The (layer, expertId) pairs the slot manager warm-pinned, or an empty span
    /// if warm-pinning never fired. Exposed mainly for tests.
    /// </summary>
    public ReadOnlySpan<(int Layer, int ExpertId)> WarmPinnedSet =>
        _warmPinned is null ? ReadOnlySpan<(int, int)>.Empty : _warmPinned;

    /// <summary>True once <see cref="RecordWarmPin"/> has been invoked.</summary>
    public bool WarmPinRecorded => _warmPinned is not null;

    /// <summary>
    /// Hit rate over accesses that occurred AFTER <see cref="RecordWarmPin"/> was
    /// called. Returns 0 if warm-pin never fired or no accesses since.
    /// </summary>
    public double PostWarmHitRate
    {
        get
        {
            if (_warmPinned is null) return 0.0;
            long postHits = Interlocked.Read(ref _totalHits) - _preWarmHits;
            long postMisses = Interlocked.Read(ref _totalMisses) - _preWarmMisses;
            long post = postHits + postMisses;
            return post == 0 ? 0.0 : (double)postHits / post;
        }
    }

    /// <summary>
    /// Hit rate over accesses that occurred BEFORE <see cref="RecordWarmPin"/>
    /// was called. Returns the overall rate if warm-pin never fired.
    /// </summary>
    public double PreWarmHitRate
    {
        get
        {
            long pre = _preWarmHits + _preWarmMisses;
            if (_warmPinned is null) return OverallHitRate;
            return pre == 0 ? 0.0 : (double)_preWarmHits / pre;
        }
    }

    /// <summary>Write a human-readable summary to <paramref name="output"/>.</summary>
    public void PrintStats(TextWriter output)
    {
        var sb = new StringBuilder();
        long th = TotalHits, tm = TotalMisses;
        double rate = (th + tm) == 0 ? 0.0 : (double)th / (th + tm);
        sb.AppendLine($"[ExpertAccessProfiler] overall hit rate: {rate:P1} ({th} hits / {th + tm} accesses)");

        if (_warmPinned is not null)
        {
            long preTotal = _preWarmHits + _preWarmMisses;
            double preRate = preTotal == 0 ? 0.0 : (double)_preWarmHits / preTotal;
            long postHits = th - _preWarmHits;
            long postMisses = tm - _preWarmMisses;
            long postTotal = postHits + postMisses;
            double postRate = postTotal == 0 ? 0.0 : (double)postHits / postTotal;
            sb.AppendLine($"[ExpertAccessProfiler] warm-pin: {_warmPinned.Length} expert(s) pinned after {preTotal} accesses");
            sb.AppendLine($"[ExpertAccessProfiler]   pre-warm  hit rate: {preRate:P1} ({_preWarmHits}/{preTotal})");
            sb.AppendLine($"[ExpertAccessProfiler]   post-warm hit rate: {postRate:P1} ({postHits}/{postTotal})");
            var perLayer = new Dictionary<int, List<int>>();
            foreach (var (layer, expertId) in _warmPinned)
            {
                if (!perLayer.TryGetValue(layer, out var list))
                    perLayer[layer] = list = [];
                list.Add(expertId);
            }
            foreach (var kv in perLayer.OrderBy(static p => p.Key))
            {
                kv.Value.Sort();
                sb.AppendLine($"[ExpertAccessProfiler]   pinned layer {kv.Key,3}: [{string.Join(", ", kv.Value)}]");
            }
        }

        for (int layer = 0; layer < _numLayers; layer++)
        {
            long lh = 0, lm = 0;
            int offset = layer * _numExperts;
            for (int e = 0; e < _numExperts; e++)
            {
                lh += Interlocked.Read(ref _hits[offset + e]);
                lm += Interlocked.Read(ref _misses[offset + e]);
            }
            // Layers that never report into the profiler (CPU-resident under hybrid
            // offload, or non-MoE layers) have zero counts. Skip the bogus 0.0 % +
            // arbitrary "top expert" output that would come from sorting equal zeros.
            if (lh + lm == 0)
            {
                sb.AppendLine($"  layer {layer,3}: (no GPU SLRU accesses recorded)");
                continue;
            }
            double lr = (double)lh / (lh + lm);
            var top = GetTopExperts(layer, 3);
            string topStr = string.Join(", ", top);
            sb.AppendLine($"  layer {layer,3}: hit {lr:P1}  top experts: [{topStr}]");
        }

        output.Write(sb.ToString());
    }
}
