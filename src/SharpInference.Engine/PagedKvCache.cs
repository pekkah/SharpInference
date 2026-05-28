using System.Runtime.InteropServices;

namespace SharpInference.Engine;

/// <summary>
/// Paged KV cache: dynamically allocates fixed-size page blocks as tokens are appended,
/// eliminating pre-allocation of the full context window.
///
/// Memory model:
/// - Pages are lazily allocated (NativeMemory) on first write to each block.
/// - <see cref="TruncateTo"/> is a "soft" truncate — it moves the length pointer without freeing pages.
///   Pages beyond the new length are reused on the next write (used during batched prefill
///   per-layer resets and speculative decoding rewinds).
/// - <see cref="Reset"/> returns all allocated page slots to the warm pool for reuse across requests.
///
/// All layers share the same physical slot index per block, so a single free list manages
/// all layer pages together. Total memory at any point:
///   allocatedBlocks × numLayers × PageSize × kvDim × 2 × sizeof(float)
/// </summary>
public sealed unsafe class PagedKvCache : IDisposable
{
    /// <summary>Number of KV positions stored per page block.</summary>
    public const int PageSize = 16;

    private readonly int _numLayers;
    private readonly int _kvDim;
    private readonly int _maxBlocks;
    private readonly nuint _pageBytes;

    // Per-layer pool: _pool[layer][slot] → pointer to PageSize * kvDim * 2 floats (keys then values).
    // Pointers are null until the slot is first allocated (lazy allocation).
    private readonly float*[][] _pool;

    // Block table: _blockTable[layer][blockIdx] = physical slot index in _pool[layer].
    private int[][] _blockTable;

    // High-water mark: number of blocks currently in the block table.
    // Only grows via Append; never shrinks (TruncateTo is a soft operation).
    // Reset sets this to 0 and pushes all slots back to _warmPool.
    private int _allocatedBlocks;

    // Slot allocator: warm pool returns recently-freed slots; _nextFreshSlot allocates never-used ones.
    private readonly Stack<int> _warmPool;
    private int _nextFreshSlot;

    // Current logical position count (can be < _allocatedBlocks * PageSize after TruncateTo).
    private int _length;

    // SnapKV (issue #51): when prefill eviction is active, _length becomes the
    // *slot* count after compaction while _logicalLength stays at the original
    // (pre-eviction) prompt length. The two diverge only after Compact() is
    // called — outside of SnapKV they track each other exactly.
    //
    //   _length         : how many K/V vectors are physically stored, == slot
    //                     index of the next append.
    //   _logicalLength  : the absolute position the next decode token sits at,
    //                     used for RoPE so cached K's (RoPE'd at their
    //                     original positions) and the incoming query share the
    //                     same reference frame.
    private int _logicalLength;

    private bool _disposed;

    public PagedKvCache(int numLayers, int numKvHeads, int headDim, int maxBlocks = 8192)
    {
        _numLayers = numLayers;
        _kvDim = numKvHeads * headDim;
        _maxBlocks = maxBlocks;

        // Page layout: [PageSize keys at offset 0, PageSize values at offset PageSize] × kvDim floats each.
        _pageBytes = (nuint)(PageSize * _kvDim * 2 * sizeof(float));

        _pool = new float*[numLayers][];
        for (int l = 0; l < numLayers; l++)
            _pool[l] = new float*[maxBlocks]; // all null until lazily allocated

        _blockTable = new int[numLayers][];
        for (int l = 0; l < numLayers; l++)
            _blockTable[l] = Array.Empty<int>();

        _warmPool = new Stack<int>(64);
        _nextFreshSlot = 0;
    }

    public int Length => _length;
    public int KvDim => _kvDim;

    /// <summary>
    /// Absolute position the next decode token will sit at, == <see cref="Length"/>
    /// unless SnapKV (issue #51) eviction has been applied. After
    /// <see cref="Compact"/>, this stays at the original prompt length while
    /// <see cref="Length"/> drops to the surviving slot count; downstream
    /// callers should use <see cref="LogicalLength"/> for RoPE on new tokens
    /// so the cached RoPE'd K's stay in the right reference frame.
    /// </summary>
    public int LogicalLength => _logicalLength;

    /// <summary>Maximum sequence length this cache can hold (slot pool limit).</summary>
    public int MaxSeqLen => _maxBlocks * PageSize;

    // ── Slot management ──────────────────────────────────────────────────

    private int AllocSlot()
    {
        if (_warmPool.Count > 0) return _warmPool.Pop();
        if (_nextFreshSlot >= _maxBlocks)
            throw new InvalidOperationException(
                $"PagedKvCache exhausted: max {_maxBlocks} blocks × {PageSize} positions = {MaxSeqLen} tokens");
        return _nextFreshSlot++;
    }

    private void EnsureBlockTableCapacity(int blockIdx)
    {
        for (int l = 0; l < _numLayers; l++)
        {
            if (_blockTable[l].Length <= blockIdx)
                Array.Resize(ref _blockTable[l], Math.Max(blockIdx + 8, _blockTable[l].Length * 2 + 8));
        }
    }

    private float* GetPage(int layer, int slot)
    {
        if (_pool[layer][slot] == null)
            _pool[layer][slot] = (float*)NativeMemory.AllocZeroed(_pageBytes);
        return _pool[layer][slot];
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Write key and value vectors for <paramref name="layer"/> at position <see cref="Length"/>.
    /// Layer 0 must be appended first for each token to trigger block allocation.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        int blockIdx = _length / PageSize;
        int offset = _length % PageSize;

        if (blockIdx >= _allocatedBlocks)
        {
            if (layer != 0)
                throw new InvalidOperationException(
                    "PagedKvCache.Append: layer 0 must be appended first to allocate a new block");
            EnsureBlockTableCapacity(blockIdx);
            int slot = AllocSlot();
            for (int l = 0; l < _numLayers; l++)
                _blockTable[l][blockIdx] = slot;
            _allocatedBlocks++;
        }

        int physSlot = _blockTable[layer][blockIdx];
        float* page = GetPage(layer, physSlot);
        float* keyDst = page + (long)offset * _kvDim;
        float* valDst = page + (long)(PageSize + offset) * _kvDim;

        key[.._kvDim].CopyTo(new Span<float>(keyDst, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(valDst, _kvDim));
    }

    /// <summary>
    /// Reserves a new block slot in the per-layer block table without allocating any page.
    /// Use this for hybrid architectures where layer 0 does not call <see cref="Append"/>
    /// (e.g. qwen35moe, whose layer 0 is a recurrent Gated DeltaNet block that stores no KV).
    /// Call once per token when crossing a <see cref="PageSize"/> boundary, BEFORE any layer
    /// in the trunk calls Append for the current token. Per-layer pages remain null until
    /// each layer's first KV write triggers lazy allocation.
    /// </summary>
    public void ReserveBlock()
    {
        int blockIdx = _length / PageSize;
        if (blockIdx >= _allocatedBlocks)
        {
            EnsureBlockTableCapacity(blockIdx);
            int slot = AllocSlot();
            for (int l = 0; l < _numLayers; l++)
                _blockTable[l][blockIdx] = slot;
            _allocatedBlocks++;
        }
    }

    /// <summary>
    /// Write key and value vectors at an explicit <paramref name="position"/> regardless
    /// of <see cref="Length"/>. Used by the MTP batched verify path (issue #30) where two
    /// tokens at adjacent positions share the same cache: token 1's per-layer Append fires
    /// at <c>startPos</c>, token 2's at <c>startPos + 1</c>, both before either token
    /// bumps <see cref="Length"/>. The block table must already cover the page containing
    /// <paramref name="position"/>; call <see cref="ReserveBlockAt"/> on token 1's first
    /// layer for each new page.
    /// </summary>
    public void AppendAt(int layer, int position, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        int blockIdx = position / PageSize;
        int offset = position % PageSize;

        if (blockIdx >= _allocatedBlocks)
            throw new InvalidOperationException(
                $"PagedKvCache.AppendAt({position}): block {blockIdx} not reserved " +
                $"(allocated={_allocatedBlocks}). Call ReserveBlockAt before the first " +
                "AppendAt that crosses a page boundary.");

        int physSlot = _blockTable[layer][blockIdx];
        float* page = GetPage(layer, physSlot);
        float* keyDst = page + (long)offset * _kvDim;
        float* valDst = page + (long)(PageSize + offset) * _kvDim;

        key[.._kvDim].CopyTo(new Span<float>(keyDst, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(valDst, _kvDim));
    }

    /// <summary>
    /// Reserve the block containing <paramref name="position"/> if not already allocated.
    /// Mirrors <see cref="ReserveBlock"/> but for an explicit position rather than the
    /// current <see cref="Length"/>. Used by the batched verify path to make room for
    /// two tokens that may straddle a page boundary.
    /// </summary>
    public void ReserveBlockAt(int position)
    {
        int blockIdx = position / PageSize;
        if (blockIdx >= _allocatedBlocks)
        {
            EnsureBlockTableCapacity(blockIdx);
            // Stretch the high-water mark to include any skipped blocks too — the cache
            // expects contiguous block ids 0.._allocatedBlocks-1 with valid slots.
            for (int b = _allocatedBlocks; b <= blockIdx; b++)
            {
                int slot = AllocSlot();
                for (int l = 0; l < _numLayers; l++)
                    _blockTable[l][b] = slot;
            }
            _allocatedBlocks = blockIdx + 1;
        }
    }

    /// <summary>Advances the logical length. Call once per token after all layers are appended.</summary>
    public void IncrementPosition()
    {
        _length++;
        _logicalLength++;
    }

    /// <summary>Returns a pointer to the key vector at <paramref name="position"/> for <paramref name="layer"/>.</summary>
    public float* KeyAt(int layer, int position)
    {
        int slot = _blockTable[layer][position / PageSize];
        return _pool[layer][slot] + (long)(position % PageSize) * _kvDim;
    }

    /// <summary>Returns a pointer to the value vector at <paramref name="position"/> for <paramref name="layer"/>.</summary>
    public float* ValueAt(int layer, int position)
    {
        int slot = _blockTable[layer][position / PageSize];
        return _pool[layer][slot] + (long)(PageSize + position % PageSize) * _kvDim;
    }

    /// <summary>
    /// Soft truncate: sets the logical length to <paramref name="length"/> without freeing pages.
    /// Positions ≥ <paramref name="length"/> will be overwritten by subsequent appends.
    /// Used during per-layer prefill resets and speculative decoding rewinds.
    /// </summary>
    public void TruncateTo(int length)
    {
        _length = length;
        _logicalLength = length;
    }

    /// <summary>
    /// Full reset: returns all allocated page slots to the warm pool for reuse.
    /// Use at the start of a new request when the KV prefix cannot be reused.
    /// </summary>
    public void Reset()
    {
        // All layers share the same slot index per block — push each slot once.
        for (int b = 0; b < _allocatedBlocks; b++)
            _warmPool.Push(_blockTable[0][b]);
        _allocatedBlocks = 0;
        _length = 0;
        _logicalLength = 0;
    }

    /// <summary>
    /// SnapKV (issue #51) compaction: keep only the K/V vectors at the positions
    /// listed in <paramref name="keepPositions"/> (sorted ascending, all within
    /// <c>[0, Length)</c>); discard the rest. After compaction the cache holds
    /// <c>keepPositions.Length</c> entries in slots <c>[0, keepPositions.Length)</c>;
    /// <see cref="LogicalLength"/> is left at the pre-compaction value so RoPE on
    /// subsequent decode tokens stays in the original position frame.
    /// </summary>
    /// <remarks>
    /// Uniform-across-layers eviction only: the same keep set applies to every
    /// layer because the block table is shared. Per-layer eviction (true SnapKV)
    /// is a follow-up; for the common case where attention sparsity patterns
    /// correlate across layers this is the right tradeoff for a first cut.
    ///
    /// The implementation copies survivors into a staging buffer one page at a
    /// time, then writes them back contiguously into the existing pages. This
    /// in-place style avoids allocating a parallel cache and works regardless
    /// of how non-contiguous the keep set is.
    /// </remarks>
    public void Compact(ReadOnlySpan<int> keepPositions)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PagedKvCache));
        int K = keepPositions.Length;
        if (K > _length)
            throw new ArgumentException(
                $"Compact: keep count {K} exceeds current Length {_length}.",
                nameof(keepPositions));
        // Validate sorted + bounds. The selector contract guarantees this; the
        // assert here catches caller bugs (it's cheap relative to the layer loop).
        int prev = -1;
        for (int i = 0; i < K; i++)
        {
            int p = keepPositions[i];
            if (p < 0 || p >= _length)
                throw new ArgumentOutOfRangeException(nameof(keepPositions),
                    $"keepPositions[{i}]={p} is outside [0,{_length}).");
            if (p <= prev)
                throw new ArgumentException(
                    $"keepPositions must be strictly increasing; got {prev} then {p} at index {i}.",
                    nameof(keepPositions));
            prev = p;
        }

        if (K == _length)
        {
            // No-op compaction (every position kept). Skip the staging dance.
            return;
        }

        int preCompactLength = _logicalLength;

        // Stage one layer at a time: read survivors into a contiguous host
        // buffer, then write them back into slot 0..K-1 of the same layer.
        // Buffer size: K × kvDim × 2 floats. For K=2048, kvDim=1024 (Qwen3-8B-
        // class), that's 16 MiB — fine for a one-shot post-prefill op.
        nuint stageBytes = (nuint)((long)K * _kvDim * 2 * sizeof(float));
        float* stage = (float*)NativeMemory.Alloc(stageBytes);
        try
        {
            for (int l = 0; l < _numLayers; l++)
            {
                // Pull survivors into the stage buffer (K rows × kvDim K-then-V layout).
                for (int i = 0; i < K; i++)
                {
                    int srcPos = keepPositions[i];
                    int srcSlot = _blockTable[l][srcPos / PageSize];
                    float* srcPage = _pool[l][srcSlot];
                    if (srcPage == null)
                    {
                        // Layer's page was never allocated for that block — write
                        // zeros to the stage row. This happens for hybrid GDN
                        // models on non-attention layers; the slot exists but no
                        // K/V was ever appended. Compaction should leave them
                        // empty rather than crash.
                        for (int j = 0; j < _kvDim * 2; j++) stage[(long)i * _kvDim * 2 + j] = 0f;
                        continue;
                    }
                    int srcOff = srcPos % PageSize;
                    float* srcKey = srcPage + (long)srcOff * _kvDim;
                    float* srcVal = srcPage + (long)(PageSize + srcOff) * _kvDim;
                    float* dstKey = stage + (long)i * _kvDim * 2;
                    float* dstVal = dstKey + _kvDim;
                    new ReadOnlySpan<float>(srcKey, _kvDim)
                        .CopyTo(new Span<float>(dstKey, _kvDim));
                    new ReadOnlySpan<float>(srcVal, _kvDim)
                        .CopyTo(new Span<float>(dstVal, _kvDim));
                }

                // Write survivors back into slot 0..K-1 of this layer. We touch
                // each destination page through GetPage so any never-allocated
                // pages get materialised on demand (matters when the source
                // survivors came from a higher slot than was previously
                // populated for this layer).
                int writeBlocks = (K + PageSize - 1) / PageSize;
                for (int i = 0; i < K; i++)
                {
                    int dstBlk = i / PageSize;
                    int dstOff = i % PageSize;
                    int dstSlot = _blockTable[l][dstBlk];
                    float* dstPage = GetPage(l, dstSlot);
                    float* dstKey = dstPage + (long)dstOff * _kvDim;
                    float* dstVal = dstPage + (long)(PageSize + dstOff) * _kvDim;
                    float* srcKey = stage + (long)i * _kvDim * 2;
                    float* srcVal = srcKey + _kvDim;
                    new ReadOnlySpan<float>(srcKey, _kvDim)
                        .CopyTo(new Span<float>(dstKey, _kvDim));
                    new ReadOnlySpan<float>(srcVal, _kvDim)
                        .CopyTo(new Span<float>(dstVal, _kvDim));
                }

                // Free pages beyond the compacted prefix. These slots return to
                // the warm pool for the *next* request's prefill — the current
                // request can still grow into them via Append for decode tokens.
                // We DON'T free the native pages here because the slot may be
                // reused immediately; freeing would force a re-alloc.
            }

            // Free trailing block-table entries whose slots are no longer used
            // for any kept position. Push their slots back to the warm pool so
            // decode appends reuse them (or future Reset can.)
            int compactedBlocks = (K + PageSize - 1) / PageSize;
            for (int b = compactedBlocks; b < _allocatedBlocks; b++)
            {
                // All layers share the slot index per block — push once.
                _warmPool.Push(_blockTable[0][b]);
            }
            _allocatedBlocks = compactedBlocks;
            _length = K;
            _logicalLength = preCompactLength;
        }
        finally
        {
            NativeMemory.Free(stage);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int l = 0; l < _numLayers; l++)
        {
            for (int s = 0; s < _nextFreshSlot; s++)
            {
                if (_pool[l][s] != null)
                {
                    NativeMemory.Free(_pool[l][s]);
                    _pool[l][s] = null;
                }
            }
        }
    }
}
