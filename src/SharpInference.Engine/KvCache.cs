using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Per-layer key-value cache for the current generation sequence.
/// Backed by preallocated tensors; grows up to the model context length.
/// </summary>
public sealed class KvCache : IDisposable
{
    private readonly Tensor[] _keys;
    private readonly Tensor[] _values;
    private int _length;

    public KvCache(ModelHyperparams hp)
    {
        _keys = new Tensor[hp.NumLayers];
        _values = new Tensor[hp.NumLayers];
        // TODO: preallocate key/value tensors for the full context window
    }

    public int Length => _length;

    public void Append(int layer, Tensor key, Tensor value)
    {
        // TODO: copy key/value into the cache at position _length
        throw new NotImplementedException();
    }

    public (Tensor Keys, Tensor Values) GetSlice(int layer, int upTo) =>
        (_keys[layer], _values[layer]);

    public void Reset() => _length = 0;

    public void Dispose()
    {
        foreach (var t in _keys) t?.Dispose();
        foreach (var t in _values) t?.Dispose();
    }
}
