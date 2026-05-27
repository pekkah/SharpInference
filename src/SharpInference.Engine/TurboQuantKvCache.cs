using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid FP32/TurboQuant KV cache: recent tokens in FP32, older tokens compressed to 3-4 bits.
/// The FP32 window provides full-precision attention for recently generated tokens.
/// When tokens age out of the window, they are compressed to TQ format.
/// </summary>
public sealed unsafe class TurboQuantKvCache : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvDim;         // numKvHeads * headDim
    private readonly int _fp32WindowSize; // recent tokens in FP32

    // Per-layer FP32 window (ring buffer)
    private readonly float*[] _fp32Keys;
    private readonly float*[] _fp32Values;

    // Per-layer TQ compressed storage. Keys live in FastScan tile layout
    // (one tile = 32 positions, contiguous per kv-head) with the in-flight
    // 0..31 positions staged in per-block format. Values stay per-block —
    // V-tile migration is Phase 3 of issue #34.
    private readonly byte*[] _tqKeyTiles;     // [layer][numTiles * numKvHeads * _tileBytes]
    private readonly byte*[] _tqKeyStaging;   // [layer][numKvHeads * TileSize * _tqBlockSize]
    private readonly byte*[] _tqValues;
    private readonly KvCacheCompressor[][] _keyCompressors;   // [layer][kvHead]
    private readonly KvCacheCompressor[][] _valueCompressors;

    private readonly int _tqBlockSize;       // bytes per compressed block per KV head
    private readonly int _tqBytesPerPosition; // tqBlockSize * numKvHeads (V-side only now)
    private readonly int _tileBytes;          // FastScan.TileBytes(_headDim) per (tile, head)
    private readonly int _stagingBytesPerHead; // TileSize * _tqBlockSize
    private readonly int _bits;

    private int _totalLength;    // total positions stored (TQ + FP32)
    private readonly int[] _layerTqLengths; // positions in TQ storage per layer
    private bool _disposed;

    /// <summary>Total positions stored in the cache (compressed + FP32).</summary>
    public int Length => _totalLength;

    /// <summary>Number of compressed positions.</summary>
    public int TqLength => _numLayers > 0 ? _layerTqLengths[0] : 0;

    /// <summary>Number of FP32 positions.</summary>
    public int Fp32Length => _totalLength - TqLength;

    /// <summary>FP32 window size.</summary>
    public int Fp32WindowSize => _fp32WindowSize;

    public int KvDim => _kvDim;
    public int MaxSeqLen => _maxSeqLen;
    public int HeadDim => _headDim;
    public int NumKvHeads => _numKvHeads;
    public int TqBlockSize => _tqBlockSize;
    public int Bits => _bits;
    public int TileBytes => _tileBytes;
    public int FastScanTileSize => SharpInference.TurboQuant.FastScan.TileSize;

    public TurboQuantKvCache(int numLayers, int maxSeqLen, int numKvHeads, int headDim,
        int fp32WindowSize = 256, int bits = 3, int layerIndexBase = 0, int totalLayerCountForSeeds = 0)
    {
        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _kvDim = numKvHeads * headDim;
        _fp32WindowSize = fp32WindowSize;
        _bits = bits;
        _tqBlockSize = TurboQuantOps.BlockSize(bits, headDim);
        _tqBytesPerPosition = _tqBlockSize * numKvHeads;
        _tileBytes = SharpInference.TurboQuant.FastScan.TileBytes(headDim);
        _stagingBytesPerHead = SharpInference.TurboQuant.FastScan.TileSize * _tqBlockSize;
        if (totalLayerCountForSeeds == 0)
            totalLayerCountForSeeds = numLayers;
        _layerTqLengths = new int[numLayers];

        // FP32 window storage per layer
        _fp32Keys = new float*[numLayers];
        _fp32Values = new float*[numLayers];
        var fp32Bytes = (nuint)((long)fp32WindowSize * _kvDim * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _fp32Keys[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
            _fp32Values[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
        }

        // TQ compressed storage per layer.
        // Keys: tile storage (numTiles × numKvHeads × _tileBytes per layer) +
        //       staging in per-block layout for the in-flight <32 positions.
        // Values: per-block layout (Phase 3 will migrate this to V-tile).
        int maxTqPositions = maxSeqLen - fp32WindowSize;
        if (maxTqPositions < 0) maxTqPositions = 0;
        int maxTiles = (maxTqPositions + SharpInference.TurboQuant.FastScan.TileSize - 1) / SharpInference.TurboQuant.FastScan.TileSize;

        _tqKeyTiles   = new byte*[numLayers];
        _tqKeyStaging = new byte*[numLayers];
        _tqValues     = new byte*[numLayers];
        if (maxTqPositions > 0)
        {
            var valBytes      = (nuint)((long)maxTqPositions * _tqBytesPerPosition);
            var keyTileBytes  = (nuint)((long)maxTiles * numKvHeads * _tileBytes);
            var keyStageBytes = (nuint)((long)numKvHeads * _stagingBytesPerHead);
            for (int i = 0; i < numLayers; i++)
            {
                _tqKeyTiles[i]   = (byte*)NativeMemory.AllocZeroed(keyTileBytes);
                _tqKeyStaging[i] = (byte*)NativeMemory.AllocZeroed(keyStageBytes);
                _tqValues[i]     = (byte*)NativeMemory.AllocZeroed(valBytes);
            }
        }

        // Create compressors per layer per KV head
        _keyCompressors = new KvCacheCompressor[numLayers][];
        _valueCompressors = new KvCacheCompressor[numLayers][];
        for (int layer = 0; layer < numLayers; layer++)
        {
            _keyCompressors[layer] = new KvCacheCompressor[numKvHeads];
            _valueCompressors[layer] = new KvCacheCompressor[numKvHeads];
            for (int head = 0; head < numKvHeads; head++)
            {
                int globalLayer = layer + layerIndexBase;
                _keyCompressors[layer][head] = new KvCacheCompressor(bits, headDim, globalLayer * numKvHeads + head);
                _valueCompressors[layer][head] = new KvCacheCompressor(bits, headDim, (globalLayer + totalLayerCountForSeeds) * numKvHeads + head);
            }
        }
    }

    public TurboQuantKvCache(ModelHyperparams hp, int fp32WindowSize = 256, int bits = 3,
        int layerIndexBase = 0, int totalLayerCountForSeeds = 0)
        : this(hp.NumLayers, hp.ContextLength, hp.NumKvHeads, hp.HeadDim,
               fp32WindowSize, bits, layerIndexBase, totalLayerCountForSeeds)
    {
    }

    /// <summary>
    /// Append K/V vectors for a given layer. If the FP32 window is full,
    /// the oldest FP32 position is compressed to TQ first.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_totalLength >= _maxSeqLen)
            throw new InvalidOperationException(
                $"TQ KV cache full: {_totalLength} >= {_maxSeqLen}");

        int fp32Count = _totalLength - _layerTqLengths[layer];

        // If FP32 window is full, compress the oldest FP32 entry
        if (fp32Count >= _fp32WindowSize)
        {
            CompressOldestFp32(layer);
            fp32Count = _totalLength - _layerTqLengths[layer];
        }

        // Write to FP32 window at the current FP32 slot
        int fp32Slot = fp32Count;
        long offset = (long)fp32Slot * _kvDim;
        key[.._kvDim].CopyTo(new Span<float>(_fp32Keys[layer] + offset, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(_fp32Values[layer] + offset, _kvDim));
    }

    /// <summary>
    /// Advances the position counter. Call once per token after appending to all layers.
    /// </summary>
    public void IncrementPosition() => _totalLength++;

    /// <summary>
    /// Returns true if the given position is in the TQ-compressed region.
    /// </summary>
    public bool IsCompressed(int position) => position < TqLength;

    /// <summary>Returns the number of compressed positions for the given layer.</summary>
    public int GetTqLength(int layer) => _layerTqLengths[layer];

    /// <summary>
    /// Returns a pointer to a FP32 key at the given position for a given layer.
    /// Only valid for positions in the FP32 window (position >= TqLength).
    /// </summary>
    public float* Fp32KeyAt(int layer, int position)
    {
        int fp32Idx = position - _layerTqLengths[layer];
        return _fp32Keys[layer] + (long)fp32Idx * _kvDim;
    }

    /// <summary>
    /// Returns a pointer to a FP32 value at the given position for a given layer.
    /// Only valid for positions in the FP32 window.
    /// </summary>
    public float* Fp32ValueAt(int layer, int position)
    {
        int fp32Idx = position - _layerTqLengths[layer];
        return _fp32Values[layer] + (long)fp32Idx * _kvDim;
    }

    /// <summary>
    /// Pointer to a FastScan K-tile (32 positions × one kv-head). Tile index
    /// is 0..NumKeyTiles(layer)-1; positions beyond <c>NumKeyTiles × 32</c> are
    /// in the staging buffer and accessed via <see cref="KeyStagingBlockAt"/>.
    /// </summary>
    public byte* KeyTileAt(int layer, int tileIdx, int kvHead)
    {
        long byteOffset = ((long)tileIdx * _numKvHeads + kvHead) * _tileBytes;
        return _tqKeyTiles[layer] + byteOffset;
    }

    /// <summary>
    /// Pointer to one per-block compressed key in the in-flight staging buffer.
    /// Valid for <c>stagingIdx in 0..KeyStagingCount(layer)-1</c>.
    /// </summary>
    public byte* KeyStagingBlockAt(int layer, int stagingIdx, int kvHead)
    {
        long byteOffset = (long)kvHead * _stagingBytesPerHead + stagingIdx * _tqBlockSize;
        return _tqKeyStaging[layer] + byteOffset;
    }

    /// <summary>Number of complete FastScan tiles for this layer's K cache.</summary>
    public int NumKeyTiles(int layer) =>
        _layerTqLengths[layer] / SharpInference.TurboQuant.FastScan.TileSize;

    /// <summary>Number of positions currently in the staging buffer (0..31).</summary>
    public int KeyStagingCount(int layer) =>
        _layerTqLengths[layer] % SharpInference.TurboQuant.FastScan.TileSize;

    /// <summary>
    /// Returns a pointer to the TQ-compressed value block at the given position and KV head.
    /// </summary>
    public byte* TqValueAt(int layer, int position, int kvHead)
    {
        long byteOffset = (long)position * _tqBytesPerPosition + kvHead * _tqBlockSize;
        return _tqValues[layer] + byteOffset;
    }

    /// <summary>
    /// Returns the key compressor for the given layer and KV head.
    /// Used to rotate queries and compute fused dequant-dot.
    /// </summary>
    public KvCacheCompressor GetKeyCompressor(int layer, int kvHead) =>
        _keyCompressors[layer][kvHead];

    /// <summary>
    /// Returns the value compressor for the given layer and KV head.
    /// </summary>
    public KvCacheCompressor GetValueCompressor(int layer, int kvHead) =>
        _valueCompressors[layer][kvHead];

    /// <summary>Resets the cache for a new generation.</summary>
    public void Reset()
    {
        _totalLength = 0;
        Array.Clear(_layerTqLengths);
    }

    /// <summary>
    /// Truncates the cache to the given length, which must fall within the FP32 window
    /// (i.e., length >= TqLength). Positions in the compressed TQ region cannot be undone.
    /// Throws <see cref="NotSupportedException"/> if length would truncate into TQ-compressed data.
    /// </summary>
    public void TruncateTo(int length)
    {
        int tqLen = TqLength;
        if (length < tqLen)
            throw new NotSupportedException(
                $"TruncateTo({length}) cannot truncate into TQ-compressed region (tqLength={tqLen}). " +
                "Speculative decoding is not supported when truncation would enter the compressed range.");
        _totalLength = length;
    }

    /// <summary>
    /// Reports estimated memory usage in bytes.
    /// </summary>
    public long EstimatedMemoryBytes()
    {
        long fp32Bytes = (long)_numLayers * 2 * _fp32WindowSize * _kvDim * sizeof(float);
        int maxTqPositions = _maxSeqLen - _fp32WindowSize;
        if (maxTqPositions <= 0) return fp32Bytes;

        int maxTiles = (maxTqPositions + SharpInference.TurboQuant.FastScan.TileSize - 1)
                       / SharpInference.TurboQuant.FastScan.TileSize;
        long keyTileBytes  = (long)_numLayers * maxTiles * _numKvHeads * _tileBytes;
        long keyStageBytes = (long)_numLayers * _numKvHeads * _stagingBytesPerHead;
        long valBytes      = (long)_numLayers * maxTqPositions * _tqBytesPerPosition;
        return fp32Bytes + keyTileBytes + keyStageBytes + valBytes;
    }

    /// <summary>
    /// Compute K-scores (q·k for all TQ-compressed positions) for one
    /// (layer, kv-head). Walks complete FastScan tiles via the i8 LUT kernel
    /// and the staging tail via per-block <c>DequantDot</c>. Final score per
    /// position is multiplied by <paramref name="attnScale"/>.
    /// </summary>
    /// <param name="rotatedQuery">Pre-rotated query (caller invokes
    /// <see cref="KvCacheCompressor.RotateQuery"/> once per head).</param>
    /// <param name="attnScale">Attention scale, typically 1/√headDim.</param>
    /// <param name="scoresOut">Output buffer, length ≥ <c>TqLength</c>.</param>
    public void ComputeKScores(
        int layer,
        int kvHead,
        float* rotatedQuery,
        float attnScale,
        float* scoresOut)
    {
        int hd = _headDim;
        int totalTq = _layerTqLengths[layer];
        int numFullTiles = totalTq / SharpInference.TurboQuant.FastScan.TileSize;
        int stagingCount = totalTq % SharpInference.TurboQuant.FastScan.TileSize;

        var centroids = SharpInference.TurboQuant.TurboQuantCodebooks.GetCentroids(_bits, hd);

        // Per-query LUT, stack-allocated for the head dims we ship today
        // (128 and 256 — capped at 4096 bytes here).
        Span<sbyte> lut = stackalloc sbyte[256 * 16];
        lut = lut.Slice(0, hd * 16);

        float lutScale = _bits == 4
            ? SharpInference.TurboQuant.FastScan.BuildLut4Bit(
                new ReadOnlySpan<float>(rotatedQuery, hd), centroids, lut, hd)
            : SharpInference.TurboQuant.FastScan.BuildLut3Bit(
                new ReadOnlySpan<float>(rotatedQuery, hd), centroids, lut, hd);

        fixed (sbyte* lutPtr = lut)
        {
            for (int tileIdx = 0; tileIdx < numFullTiles; tileIdx++)
            {
                byte* tile = KeyTileAt(layer, tileIdx, kvHead);
                SharpInference.TurboQuant.FastScan.KScoreTile4BitAvx2(
                    tile, lutPtr, lutScale, attnScale,
                    scoresOut + (long)tileIdx * SharpInference.TurboQuant.FastScan.TileSize,
                    hd);
            }
        }

        if (stagingCount > 0)
        {
            var keyCompressor = _keyCompressors[layer][kvHead];
            var rotatedQuerySpan = new ReadOnlySpan<float>(rotatedQuery, hd);
            int tailStart = numFullTiles * SharpInference.TurboQuant.FastScan.TileSize;
            for (int s = 0; s < stagingCount; s++)
            {
                byte* block = KeyStagingBlockAt(layer, s, kvHead);
                float dot = keyCompressor.DequantDot(
                    new ReadOnlySpan<byte>(block, _tqBlockSize),
                    rotatedQuerySpan);
                scoresOut[tailStart + s] = dot * attnScale;
            }
        }
    }

    private void CompressOldestFp32(int layer)
    {
        // The oldest FP32 position is at index 0 in the FP32 window
        // (we use simple linear addressing, not a ring buffer, for now)
        int fp32Slot = 0;
        long fp32Offset = (long)fp32Slot * _kvDim;

        int stagingSlot = _layerTqLengths[layer] % SharpInference.TurboQuant.FastScan.TileSize;

        // Compress each KV head independently. Keys go to the staging slot;
        // values continue using the existing per-block layout.
        for (int head = 0; head < _numKvHeads; head++)
        {
            int headOffset = head * _headDim;
            var keySpan = new ReadOnlySpan<float>(_fp32Keys[layer] + fp32Offset + headOffset, _headDim);
            var valSpan = new ReadOnlySpan<float>(_fp32Values[layer] + fp32Offset + headOffset, _headDim);

            long stagingOffset = (long)head * _stagingBytesPerHead + (long)stagingSlot * _tqBlockSize;
            var keyDest = new Span<byte>(_tqKeyStaging[layer] + stagingOffset, _tqBlockSize);

            long valOffset = (long)_layerTqLengths[layer] * _tqBytesPerPosition + head * _tqBlockSize;
            var valDest = new Span<byte>(_tqValues[layer] + valOffset, _tqBlockSize);

            _keyCompressors[layer][head].Compress(keySpan, keyDest);
            _valueCompressors[layer][head].Compress(valSpan, valDest);
        }

        _layerTqLengths[layer]++;

        // If the staging buffer just filled, pack it into a new K-tile.
        // Per-head staging is contiguous, so PackTile reads it directly.
        if (_layerTqLengths[layer] % SharpInference.TurboQuant.FastScan.TileSize == 0)
        {
            int tileIdx = _layerTqLengths[layer] / SharpInference.TurboQuant.FastScan.TileSize - 1;
            for (int head = 0; head < _numKvHeads; head++)
            {
                var stageSpan = new ReadOnlySpan<byte>(
                    _tqKeyStaging[layer] + (long)head * _stagingBytesPerHead,
                    _stagingBytesPerHead);
                var tileSpan = new Span<byte>(KeyTileAt(layer, tileIdx, head), _tileBytes);
                if (_bits == 4)
                    SharpInference.TurboQuant.FastScan.PackTile4Bit(stageSpan, tileSpan, _headDim);
                else
                    SharpInference.TurboQuant.FastScan.PackTile3Bit(stageSpan, tileSpan, _headDim);
            }
        }

        // Shift FP32 window: move remaining entries down by 1
        int fp32Count = _totalLength - (_layerTqLengths[layer] - 1);
        if (fp32Count > 1)
        {
            long copyBytes = (long)(fp32Count - 1) * _kvDim * sizeof(float);
            Buffer.MemoryCopy(_fp32Keys[layer] + _kvDim, _fp32Keys[layer], copyBytes, copyBytes);
            Buffer.MemoryCopy(_fp32Values[layer] + _kvDim, _fp32Values[layer], copyBytes, copyBytes);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            NativeMemory.Free(_fp32Keys[i]);
            NativeMemory.Free(_fp32Values[i]);
            _fp32Keys[i] = null;
            _fp32Values[i] = null;
            if (_tqKeyTiles[i]   != null) { NativeMemory.Free(_tqKeyTiles[i]);   _tqKeyTiles[i]   = null; }
            if (_tqKeyStaging[i] != null) { NativeMemory.Free(_tqKeyStaging[i]); _tqKeyStaging[i] = null; }
            if (_tqValues[i]     != null) { NativeMemory.Free(_tqValues[i]);     _tqValues[i]     = null; }
        }
    }
}
