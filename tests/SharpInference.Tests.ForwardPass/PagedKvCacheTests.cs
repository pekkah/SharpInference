using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for PagedKvCache: paged memory layout, cross-page access, soft truncate, prefix reuse.
/// </summary>
public sealed unsafe class PagedKvCacheTests : IDisposable
{
    // 2 layers, 2 KV heads, head dim 4 → kvDim = 8, PageSize = 16
    private readonly PagedKvCache _cache = new(numLayers: 2, numKvHeads: 2, headDim: 4);

    public void Dispose() => _cache.Dispose();

    private static void Append(PagedKvCache cache, int layer, float k, float v)
    {
        float[] ks = [k, k, k, k, k, k, k, k];
        float[] vs = [v, v, v, v, v, v, v, v];
        cache.Append(layer, ks, vs);
    }

    private static void AppendToken(PagedKvCache cache, float k, float v)
    {
        Append(cache, 0, k, v);
        Append(cache, 1, k, v);
        cache.IncrementPosition();
    }

    [Fact]
    public void NewCache_LengthIsZero()
    {
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void Append_IncreasesLength()
    {
        AppendToken(_cache, 1f, 2f);
        Assert.Equal(1, _cache.Length);
    }

    [Fact]
    public void KeyAt_ReturnsCorrectValue()
    {
        AppendToken(_cache, 42f, 0f);

        float* k = _cache.KeyAt(0, 0);
        Assert.Equal(42f, k[0]);
        Assert.Equal(42f, k[7]);
    }

    [Fact]
    public void ValueAt_ReturnsCorrectValue()
    {
        AppendToken(_cache, 0f, 99f);

        float* v = _cache.ValueAt(0, 0);
        Assert.Equal(99f, v[0]);
        Assert.Equal(99f, v[7]);
    }

    [Fact]
    public void LayersAreIndependent()
    {
        Append(_cache, 0, 11f, 12f);
        Append(_cache, 1, 21f, 22f);
        _cache.IncrementPosition();

        Assert.Equal(11f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(12f, _cache.ValueAt(0, 0)[0]);
        Assert.Equal(21f, _cache.KeyAt(1, 0)[0]);
        Assert.Equal(22f, _cache.ValueAt(1, 0)[0]);
    }

    [Fact]
    public void MultiplePositions_CorrectLayout()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, -i);

        for (int i = 0; i < 5; i++)
        {
            Assert.Equal((float)i, _cache.KeyAt(0, i)[0]);
            Assert.Equal((float)-i, _cache.ValueAt(0, i)[0]);
        }
    }

    [Fact]
    public void CrossPageBoundary_CorrectAccess()
    {
        // Fill exactly two pages (PageSize = 16, so positions 0..31)
        for (int i = 0; i < PagedKvCache.PageSize * 2; i++)
            AppendToken(_cache, i, i * 10f);

        Assert.Equal(PagedKvCache.PageSize * 2, _cache.Length);

        // Check first position of second page
        int p = PagedKvCache.PageSize;
        Assert.Equal((float)p, _cache.KeyAt(0, p)[0]);
        Assert.Equal((float)(p * 10), _cache.ValueAt(0, p)[0]);

        // Check last position of second page
        int last = PagedKvCache.PageSize * 2 - 1;
        Assert.Equal((float)last, _cache.KeyAt(0, last)[0]);
    }

    [Fact]
    public void TruncateTo_SoftTruncate_PreservesExistingPages()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, 0f);

        _cache.TruncateTo(2);
        Assert.Equal(2, _cache.Length);

        // Pages are still valid — can read position 0 and 1
        Assert.Equal(0f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(1f, _cache.KeyAt(0, 1)[0]);
    }

    [Fact]
    public void TruncateTo_ThenAppend_OverwritesCorrectly()
    {
        for (int i = 0; i < 5; i++)
            AppendToken(_cache, i, 0f);

        _cache.TruncateTo(3);

        // Overwrite position 3 with a new value
        AppendToken(_cache, 99f, 0f);

        Assert.Equal(4, _cache.Length);
        Assert.Equal(99f, _cache.KeyAt(0, 3)[0]);
        // Positions before 3 are unchanged
        Assert.Equal(2f, _cache.KeyAt(0, 2)[0]);
    }

    [Fact]
    public void Reset_ReturnsLengthToZero()
    {
        for (int i = 0; i < 8; i++)
            AppendToken(_cache, i, 0f);

        _cache.Reset();
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void Reset_AllowsReuseOfPages()
    {
        // Fill, reset, and fill again — pages should be reused from warm pool
        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, (float)i, 0f);

        _cache.Reset();

        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, (float)(i + 100), 0f);

        Assert.Equal(PagedKvCache.PageSize, _cache.Length);
        Assert.Equal(100f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(115f, _cache.KeyAt(0, 15)[0]);
    }

    [Fact]
    public void PrefixReuse_TruncateAndContinue()
    {
        // Simulate prefix caching: fill 32 positions (2 pages), truncate to 16, continue from 16
        for (int i = 0; i < 32; i++)
            AppendToken(_cache, (float)i, 0f);

        // New request: same prefix (0..15 preserved), truncate to prefix length
        _cache.TruncateTo(16);

        // Fill positions 16..19 with new values
        for (int i = 0; i < 4; i++)
            AppendToken(_cache, (float)(200 + i), 0f);

        Assert.Equal(20, _cache.Length);

        // Prefix positions still valid
        Assert.Equal(0f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(15f, _cache.KeyAt(0, 15)[0]);

        // New suffix positions have new values
        Assert.Equal(200f, _cache.KeyAt(0, 16)[0]);
        Assert.Equal(203f, _cache.KeyAt(0, 19)[0]);
    }

    [Fact]
    public void MaxSeqLen_ReflectsMaxBlocks()
    {
        using var small = new PagedKvCache(numLayers: 1, numKvHeads: 1, headDim: 4, maxBlocks: 4);
        Assert.Equal(4 * PagedKvCache.PageSize, small.MaxSeqLen);
    }

    [Fact]
    public void ReserveBlock_AllowsLayerOneToAppendFirst()
    {
        // ReserveBlock makes the "layer 0 must call Append first" invariant optional —
        // hybrid models can call any layer's Append after a ReserveBlock at the page boundary.
        _cache.ReserveBlock();
        Append(_cache, 1, 7f, 11f);            // layer 1 first, no layer-0 write
        // Layer 1 should now read back what it wrote.
        Assert.Equal(7f,  _cache.KeyAt(1, 0)[0]);
        Assert.Equal(11f, _cache.ValueAt(1, 0)[0]);
        _cache.IncrementPosition();
        Assert.Equal(1, _cache.Length);
    }

    [Fact]
    public void ReserveBlock_IdempotentWithinSameBlock()
    {
        // Multiple ReserveBlock calls inside the same PageSize window should be no-ops.
        _cache.ReserveBlock();
        _cache.ReserveBlock();
        _cache.ReserveBlock();
        Append(_cache, 0, 1f, 2f);
        _cache.IncrementPosition();
        Append(_cache, 0, 3f, 4f);
        _cache.IncrementPosition();
        Assert.Equal(1f, _cache.KeyAt(0, 0)[0]);
        Assert.Equal(3f, _cache.KeyAt(0, 1)[0]);
        Assert.Equal(2, _cache.Length);
    }

    [Fact]
    public void ReserveBlock_AcrossPageBoundary_AllocatesNewBlock()
    {
        // Fill page 0 with 16 tokens, then ReserveBlock at the boundary and verify page 1
        // is usable from layer 1 only (layer 0's page-1 slot stays unallocated).
        for (int i = 0; i < PagedKvCache.PageSize; i++)
            AppendToken(_cache, i, i + 100f);
        Assert.Equal(PagedKvCache.PageSize, _cache.Length);

        _cache.ReserveBlock();                 // crosses into page 1
        Append(_cache, 1, 42f, 99f);           // layer 1 only writes page 1
        _cache.IncrementPosition();

        Assert.Equal(PagedKvCache.PageSize + 1, _cache.Length);
        Assert.Equal(42f, _cache.KeyAt(1, PagedKvCache.PageSize)[0]);
        Assert.Equal(99f, _cache.ValueAt(1, PagedKvCache.PageSize)[0]);
    }

    // ── SnapKV (issue #51) compaction ──────────────────────────────────────

    [Fact]
    public void Compact_KeepEverything_NoOp()
    {
        for (int i = 0; i < 20; i++) AppendToken(_cache, i, -i);
        var keep = new int[20];
        for (int i = 0; i < 20; i++) keep[i] = i;

        _cache.Compact(keep);

        Assert.Equal(20, _cache.Length);
        Assert.Equal(20, _cache.LogicalLength);
        for (int i = 0; i < 20; i++)
        {
            Assert.Equal((float)i, _cache.KeyAt(0, i)[0]);
            Assert.Equal((float)-i, _cache.ValueAt(0, i)[0]);
        }
    }

    [Fact]
    public void Compact_DropsEvictedPositions_KeepsOrder()
    {
        // 20 tokens at positions 0..19; keep {0, 5, 11, 17, 19}.
        for (int i = 0; i < 20; i++) AppendToken(_cache, i + 1f, -(i + 1f));
        int[] keep = { 0, 5, 11, 17, 19 };

        _cache.Compact(keep);

        Assert.Equal(5, _cache.Length);              // slot count drops
        Assert.Equal(20, _cache.LogicalLength);      // RoPE frame preserved
        // Slot i now holds what was at position keep[i].
        for (int i = 0; i < keep.Length; i++)
        {
            float expectedK = keep[i] + 1f;
            float expectedV = -(keep[i] + 1f);
            Assert.Equal(expectedK, _cache.KeyAt(0, i)[0]);
            Assert.Equal(expectedV, _cache.ValueAt(0, i)[0]);
            Assert.Equal(expectedK, _cache.KeyAt(1, i)[0]);
            Assert.Equal(expectedV, _cache.ValueAt(1, i)[0]);
        }
    }

    [Fact]
    public void Compact_ThenAppend_NewTokenLandsAtCompactedTail()
    {
        for (int i = 0; i < 20; i++) AppendToken(_cache, i + 1f, 0f);
        int[] keep = { 0, 5, 11, 17, 19 };
        _cache.Compact(keep);

        // A decode-side append should write at slot 5 (the new tail), while
        // LogicalLength advances from 20 to 21 — the decode caller will RoPE
        // the new K at position 20 (the original sequence frame).
        AppendToken(_cache, 999f, 0f);

        Assert.Equal(6, _cache.Length);
        Assert.Equal(21, _cache.LogicalLength);
        Assert.Equal(999f, _cache.KeyAt(0, 5)[0]);
        // Pre-compaction survivors untouched.
        Assert.Equal(6f, _cache.KeyAt(0, 1)[0]);  // was position 5, value 6f
    }

    [Fact]
    public void Compact_RejectsOutOfRange()
    {
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        // 8 stored — position 8 is out of range.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _cache.Compact(new[] { 0, 8 }));
    }

    [Fact]
    public void Compact_RejectsUnsorted()
    {
        for (int i = 0; i < 8; i++) AppendToken(_cache, i, 0f);
        Assert.Throws<ArgumentException>(() =>
            _cache.Compact(new[] { 3, 1 }));
    }
}
