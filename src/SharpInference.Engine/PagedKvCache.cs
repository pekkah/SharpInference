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

    /// <summary>Advances the logical length. Call once per token after all layers are appended.</summary>
    public void IncrementPosition() => _length++;

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
    public void TruncateTo(int length) => _length = length;

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
