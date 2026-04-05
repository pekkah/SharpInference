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

    // Per-layer TQ compressed storage
    private readonly byte*[] _tqKeys;
    private readonly byte*[] _tqValues;
    private readonly KvCacheCompressor[][] _keyCompressors;   // [layer][kvHead]
    private readonly KvCacheCompressor[][] _valueCompressors;

    private readonly int _tqBlockSize;   // bytes per compressed block per KV head
    private readonly int _tqBytesPerPosition; // tqBlockSize * numKvHeads
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

        // TQ compressed storage per layer
        int maxTqPositions = maxSeqLen - fp32WindowSize;
        if (maxTqPositions < 0) maxTqPositions = 0;

        _tqKeys = new byte*[numLayers];
        _tqValues = new byte*[numLayers];
        if (maxTqPositions > 0)
        {
            var tqBytes = (nuint)((long)maxTqPositions * _tqBytesPerPosition);
            for (int i = 0; i < numLayers; i++)
            {
                _tqKeys[i] = (byte*)NativeMemory.AllocZeroed(tqBytes);
                _tqValues[i] = (byte*)NativeMemory.AllocZeroed(tqBytes);
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
    /// Returns a pointer to the TQ-compressed key block at the given position and KV head.
    /// Only valid for positions in the TQ region (position &lt; TqLength).
    /// </summary>
    public byte* TqKeyAt(int layer, int position, int kvHead)
    {
        long byteOffset = (long)position * _tqBytesPerPosition + kvHead * _tqBlockSize;
        return _tqKeys[layer] + byteOffset;
    }

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
        long tqBytes = maxTqPositions > 0 ? (long)_numLayers * 2 * maxTqPositions * _tqBytesPerPosition : 0;
        return fp32Bytes + tqBytes;
    }

    private void CompressOldestFp32(int layer)
    {
        // The oldest FP32 position is at index 0 in the FP32 window
        // (we use simple linear addressing, not a ring buffer, for now)
        int fp32Slot = 0;
        long fp32Offset = (long)fp32Slot * _kvDim;

        // Compress each KV head independently
        for (int head = 0; head < _numKvHeads; head++)
        {
            int headOffset = head * _headDim;
            var keySpan = new ReadOnlySpan<float>(_fp32Keys[layer] + fp32Offset + headOffset, _headDim);
            var valSpan = new ReadOnlySpan<float>(_fp32Values[layer] + fp32Offset + headOffset, _headDim);

            long tqOffset = (long)_layerTqLengths[layer] * _tqBytesPerPosition + head * _tqBlockSize;
            var keyDest = new Span<byte>(_tqKeys[layer] + tqOffset, _tqBlockSize);
            var valDest = new Span<byte>(_tqValues[layer] + tqOffset, _tqBlockSize);

            _keyCompressors[layer][head].Compress(keySpan, keyDest);
            _valueCompressors[layer][head].Compress(valSpan, valDest);
        }

        // Shift FP32 window: move remaining entries down by 1
        int fp32Count = _totalLength - _layerTqLengths[layer];
        if (fp32Count > 1)
        {
            long copyBytes = (long)(fp32Count - 1) * _kvDim * sizeof(float);
            Buffer.MemoryCopy(_fp32Keys[layer] + _kvDim, _fp32Keys[layer], copyBytes, copyBytes);
            Buffer.MemoryCopy(_fp32Values[layer] + _kvDim, _fp32Values[layer], copyBytes, copyBytes);
        }

        _layerTqLengths[layer]++;
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
            if (_tqKeys[i] != null) { NativeMemory.Free(_tqKeys[i]); _tqKeys[i] = null; }
            if (_tqValues[i] != null) { NativeMemory.Free(_tqValues[i]); _tqValues[i] = null; }
        }
    }
}
