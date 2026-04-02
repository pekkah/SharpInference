using System.Runtime.InteropServices;
using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Per-layer FP32 key-value cache for autoregressive generation.
/// Backed by NativeMemory. Each layer stores keys and values as contiguous float buffers
/// with shape [max_seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim.
/// </summary>
public sealed unsafe class KvCache : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _kvDim; // num_kv_heads * head_dim
    private readonly float*[] _keys;
    private readonly float*[] _values;
    private int _length;
    private bool _disposed;

    public KvCache(int numLayers, int maxSeqLen, int numKvHeads, int headDim)
    {
        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _kvDim = numKvHeads * headDim;
        _keys = new float*[numLayers];
        _values = new float*[numLayers];

        var bytesPerLayer = (nuint)((long)maxSeqLen * _kvDim * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _keys[i] = (float*)NativeMemory.AllocZeroed(bytesPerLayer);
            _values[i] = (float*)NativeMemory.AllocZeroed(bytesPerLayer);
        }
    }

    public KvCache(ModelHyperparams hp)
        : this(hp.NumLayers, hp.ContextLength, hp.NumKvHeads, hp.EmbeddingDim / hp.NumHeads)
    {
    }

    /// <summary>Number of positions currently stored in the cache.</summary>
    public int Length => _length;

    public int KvDim => _kvDim;
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>
    /// Append a key and value vector for the given layer at the current position.
    /// key and value should each have _kvDim elements.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_length >= _maxSeqLen)
            throw new InvalidOperationException(
                $"KV cache full: {_length} >= {_maxSeqLen}");

        var offset = (long)_length * _kvDim;
        key[.._kvDim].CopyTo(new Span<float>(_keys[layer] + offset, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(_values[layer] + offset, _kvDim));
    }

    /// <summary>
    /// Advances the position counter. Call once per token after appending to all layers.
    /// </summary>
    public void IncrementPosition() => _length++;

    /// <summary>
    /// Returns a read-only span over cached keys for the given layer,
    /// covering positions [0, Length). Shape: [Length * kvDim].
    /// </summary>
    public ReadOnlySpan<float> GetKeys(int layer) =>
        new(_keys[layer], _length * _kvDim);

    /// <summary>
    /// Returns a read-only span over cached values for the given layer,
    /// covering positions [0, Length). Shape: [Length * kvDim].
    /// </summary>
    public ReadOnlySpan<float> GetValues(int layer) =>
        new(_values[layer], _length * _kvDim);

    /// <summary>
    /// Returns a pointer to the key at a specific position for a given layer.
    /// </summary>
    public float* KeyAt(int layer, int position) =>
        _keys[layer] + (long)position * _kvDim;

    /// <summary>
    /// Returns a pointer to the value at a specific position for a given layer.
    /// </summary>
    public float* ValueAt(int layer, int position) =>
        _values[layer] + (long)position * _kvDim;

    /// <summary>Resets the cache for a new generation.</summary>
    public void Reset() => _length = 0;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            NativeMemory.Free(_keys[i]);
            NativeMemory.Free(_values[i]);
            _keys[i] = null;
            _values[i] = null;
        }
    }
}
