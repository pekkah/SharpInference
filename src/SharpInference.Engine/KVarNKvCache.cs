using System.Numerics;
using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid FP32 / KVarN KV cache (issue #180, P0 CPU reference). Recent tokens
/// live in a full-precision window; once <see cref="KVarN.TileSize"/> tokens have
/// aged out of the window they are quantized together into a KVarN tile
/// (4-bit keys / 2-bit values, per the <c>kvarn_k4v2_g128</c> preset). KVarN's
/// dual-axis variance normalization needs the whole 128-token tile assembled
/// before it can quantize, which is exactly what this aging buffer provides.
///
/// The public surface mirrors <see cref="TurboQuantKvCache"/> so the attention
/// dispatch in <c>ForwardPass</c> can host a sibling branch. Per the issue's
/// scope note this is positioned to eventually fold into the TurboQuant cache
/// machinery as a selectable quantizer; for the P0 accuracy gate it is kept as
/// a separate reference cache to avoid disturbing the shipping TQ path.
/// </summary>
public sealed unsafe class KVarNKvCache : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvDim;
    private readonly int _fp32WindowSize;
    private readonly int _sinkhornIters;
    private readonly int _fp32Capacity;   // window + TileSize, in tokens

    // FP32 region per layer: positions [tqLen, total), slot = pos - tqLen.
    private readonly float*[] _fp32Keys;
    private readonly float*[] _fp32Values;

    // Compressed tiles per layer: each entry covers TileSize tokens × all kv-heads.
    private readonly List<KVarNTile[]>[] _keyTiles;
    private readonly List<KVarNTile[]>[] _valueTiles;

    // Per (layer, head) sign patterns for the randomized Hadamard rotation.
    private readonly float[][] _keySign;     // [layer*numKvHeads + head]
    private readonly float[][] _valueSign;

    private int _totalLength;
    private readonly int[] _layerTqLengths;
    private bool _disposed;

    public int Length => _totalLength;
    public int TqLength => _numLayers > 0 ? _layerTqLengths[0] : 0;
    public int Fp32Length => _totalLength - TqLength;
    public int Fp32WindowSize => _fp32WindowSize;
    public int KvDim => _kvDim;
    public int MaxSeqLen => _maxSeqLen;
    public int HeadDim => _headDim;
    public int NumKvHeads => _numKvHeads;

    public KVarNKvCache(int numLayers, int maxSeqLen, int numKvHeads, int headDim,
        int fp32WindowSize = 256, int sinkhornIters = KVarN.DefaultSinkhornIters,
        int layerIndexBase = 0, int totalLayerCountForSeeds = 0)
    {
        if (!BitOperations.IsPow2(headDim))
            throw new ArgumentException(
                $"KVarN requires a power-of-two head dimension (got {headDim}).", nameof(headDim));

        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _kvDim = numKvHeads * headDim;
        _fp32WindowSize = Math.Min(fp32WindowSize, maxSeqLen);
        _sinkhornIters = sinkhornIters;
        _fp32Capacity = _fp32WindowSize + KVarN.TileSize;
        _layerTqLengths = new int[numLayers];
        if (totalLayerCountForSeeds == 0)
            totalLayerCountForSeeds = numLayers;

        _fp32Keys = new float*[numLayers];
        _fp32Values = new float*[numLayers];
        var fp32Bytes = (nuint)((long)_fp32Capacity * _kvDim * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _fp32Keys[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
            _fp32Values[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
        }

        _keyTiles = new List<KVarNTile[]>[numLayers];
        _valueTiles = new List<KVarNTile[]>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _keyTiles[i] = new List<KVarNTile[]>();
            _valueTiles[i] = new List<KVarNTile[]>();
        }

        _keySign = new float[numLayers * numKvHeads][];
        _valueSign = new float[numLayers * numKvHeads][];
        for (int layer = 0; layer < numLayers; layer++)
        {
            int globalLayer = layer + layerIndexBase;
            for (int head = 0; head < numKvHeads; head++)
            {
                int idx = layer * numKvHeads + head;
                _keySign[idx] = KVarN.GenerateSignPattern(headDim, globalLayer * numKvHeads + head);
                _valueSign[idx] = KVarN.GenerateSignPattern(headDim,
                    (globalLayer + totalLayerCountForSeeds) * numKvHeads + head);
            }
        }
    }

    public KVarNKvCache(ModelHyperparams hp, int fp32WindowSize = 256,
        int sinkhornIters = KVarN.DefaultSinkhornIters,
        int layerIndexBase = 0, int totalLayerCountForSeeds = 0)
        : this(hp.NumLayers, hp.ContextLength, hp.NumKvHeads, hp.HeadDim,
               fp32WindowSize, sinkhornIters, layerIndexBase, totalLayerCountForSeeds)
    {
    }

    /// <summary>Number of compressed positions for the given layer.</summary>
    public int GetTqLength(int layer) => _layerTqLengths[layer];

    /// <summary>True if the absolute position is in the compressed (tiled) region.</summary>
    public bool IsCompressed(int position) => position < TqLength;

    /// <summary>Pointer to a FP32 key for a position in the full-precision region.</summary>
    public float* Fp32KeyAt(int layer, int position)
    {
        int slot = position - _layerTqLengths[layer];
        return _fp32Keys[layer] + (long)slot * _kvDim;
    }

    /// <summary>Pointer to a FP32 value for a position in the full-precision region.</summary>
    public float* Fp32ValueAt(int layer, int position)
    {
        int slot = position - _layerTqLengths[layer];
        return _fp32Values[layer] + (long)slot * _kvDim;
    }

    /// <summary>Pre-rotate a query for fused key scoring (key sign pattern for this head).</summary>
    public void RotateQueryKey(int layer, int kvHead, ReadOnlySpan<float> query, Span<float> rotated)
        => KVarN.Rotate(query, rotated, _keySign[layer * _numKvHeads + kvHead], _headDim);

    /// <summary>
    /// Append one token's K/V for a layer. If the FP32 region is full, the
    /// oldest <see cref="KVarN.TileSize"/> tokens are quantized into a tile first.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_totalLength >= _maxSeqLen)
            throw new InvalidOperationException(
                $"KVarN KV cache full: {_totalLength} >= {_maxSeqLen}");

        int fp32Count = _totalLength - _layerTqLengths[layer];
        if (fp32Count >= _fp32Capacity)
        {
            CompressOldestTile(layer);
            fp32Count = _totalLength - _layerTqLengths[layer];
        }

        long offset = (long)fp32Count * _kvDim;
        key[.._kvDim].CopyTo(new Span<float>(_fp32Keys[layer] + offset, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(_fp32Values[layer] + offset, _kvDim));
    }

    /// <summary>Advance the global position counter (once per token, after appending to all layers).</summary>
    public void IncrementPosition() => _totalLength++;

    /// <summary>
    /// Compute raw key scores for every compressed (tiled) position of one
    /// (layer, kv-head) into <paramref name="scoresOut"/> (length ≥ TqLength).
    /// </summary>
    public void ComputeKScores(int layer, int kvHead, ReadOnlySpan<float> rotatedQuery,
        float attnScale, Span<float> scoresOut)
    {
        var tiles = _keyTiles[layer];
        int pos = 0;
        for (int ti = 0; ti < tiles.Count; ti++)
        {
            KVarNTile tile = tiles[ti][kvHead];
            KVarN.KScore(tile, rotatedQuery, attnScale, scoresOut.Slice(pos, tile.T));
            pos += tile.T;
        }
    }

    /// <summary>
    /// Aggregate <c>Σ_t weights[t]·value[t]</c> over the compressed (tiled)
    /// positions of one (layer, kv-head) into <paramref name="outAcc"/>
    /// (length ≥ headDim).
    /// </summary>
    public void ComputeVAggregation(int layer, int kvHead, ReadOnlySpan<float> weights,
        Span<float> outAcc)
    {
        var tiles = _valueTiles[layer];
        var sign = _valueSign[layer * _numKvHeads + kvHead];
        int pos = 0;
        for (int ti = 0; ti < tiles.Count; ti++)
        {
            KVarNTile tile = tiles[ti][kvHead];
            KVarN.VAggregate(tile, weights.Slice(pos, tile.T), sign, outAcc);
            pos += tile.T;
        }
    }

    private void CompressOldestTile(int layer)
    {
        int t = KVarN.TileSize;

        // Gather each head's TileSize×headDim sub-matrix from the oldest tokens
        // (slots 0..t-1 of the FP32 region) and quantize.
        var keyTile = new KVarNTile[_numKvHeads];
        var valueTile = new KVarNTile[_numKvHeads];
        float[] kScratch = new float[t * _headDim];
        float[] vScratch = new float[t * _headDim];

        for (int head = 0; head < _numKvHeads; head++)
        {
            int headOffset = head * _headDim;
            for (int i = 0; i < t; i++)
            {
                long src = (long)i * _kvDim + headOffset;
                new ReadOnlySpan<float>(_fp32Keys[layer] + src, _headDim)
                    .CopyTo(kScratch.AsSpan(i * _headDim, _headDim));
                new ReadOnlySpan<float>(_fp32Values[layer] + src, _headDim)
                    .CopyTo(vScratch.AsSpan(i * _headDim, _headDim));
            }
            keyTile[head] = KVarN.CompressKeyTile(kScratch, t, _headDim,
                _keySign[layer * _numKvHeads + head], _sinkhornIters);
            valueTile[head] = KVarN.CompressValueTile(vScratch, t, _headDim,
                _valueSign[layer * _numKvHeads + head], _sinkhornIters);
        }

        _keyTiles[layer].Add(keyTile);
        _valueTiles[layer].Add(valueTile);
        _layerTqLengths[layer] += t;

        // Shift the FP32 region down by one tile.
        int remaining = (_totalLength - _layerTqLengths[layer]);
        if (remaining > 0)
        {
            long copyBytes = (long)remaining * _kvDim * sizeof(float);
            Buffer.MemoryCopy(_fp32Keys[layer] + (long)t * _kvDim, _fp32Keys[layer], copyBytes, copyBytes);
            Buffer.MemoryCopy(_fp32Values[layer] + (long)t * _kvDim, _fp32Values[layer], copyBytes, copyBytes);
        }
    }

    /// <summary>Estimated memory usage in bytes (FP32 region + compressed tiles).</summary>
    public long EstimatedMemoryBytes()
    {
        long fp32Bytes = (long)_numLayers * 2 * _fp32Capacity * _kvDim * sizeof(float);
        long tileBytes = 0;
        for (int layer = 0; layer < _numLayers; layer++)
        {
            foreach (var heads in _keyTiles[layer])
                foreach (var tile in heads) tileBytes += tile.EstimatedBytes;
            foreach (var heads in _valueTiles[layer])
                foreach (var tile in heads) tileBytes += tile.EstimatedBytes;
        }
        return fp32Bytes + tileBytes;
    }

    /// <summary>Set the global position counter without touching per-layer state (batched prefill).</summary>
    public void ResetTotalLengthForBatchedPrefill(int length) => _totalLength = length;

    /// <summary>Reset the cache for a new generation.</summary>
    public void Reset()
    {
        _totalLength = 0;
        Array.Clear(_layerTqLengths);
        for (int i = 0; i < _numLayers; i++)
        {
            _keyTiles[i].Clear();
            _valueTiles[i].Clear();
        }
    }

    /// <summary>
    /// Truncate to <paramref name="length"/>, which must fall within the
    /// full-precision region (i.e. length ≥ TqLength). Compressed positions
    /// cannot be undone.
    /// </summary>
    public void TruncateTo(int length)
    {
        int tqLen = TqLength;
        if (length < tqLen)
            throw new NotSupportedException(
                $"TruncateTo({length}) cannot truncate into the KVarN-compressed region (tqLength={tqLen}).");
        _totalLength = length;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            if (_fp32Keys[i] != null) { NativeMemory.Free(_fp32Keys[i]); _fp32Keys[i] = null; }
            if (_fp32Values[i] != null) { NativeMemory.Free(_fp32Values[i]); _fp32Values[i] = null; }
        }
    }
}
