using SharpInference.Core;

namespace SharpInference.TurboQuant;

/// <summary>
/// KV-cache compression using Lloyd-Max scalar quantisation.
/// Reduces KV-cache memory footprint by 4–8x with minimal quality loss.
/// Codebooks are loaded from JSON files in the <c>codebooks/</c> directory.
/// </summary>
public sealed class KvCacheCompressor
{
    private readonly LloydMaxCodebook _keyCodebook;
    private readonly LloydMaxCodebook _valueCodebook;

    public KvCacheCompressor(LloydMaxCodebook keyCodebook, LloydMaxCodebook valueCodebook)
    {
        _keyCodebook = keyCodebook;
        _valueCodebook = valueCodebook;
    }

    /// <summary>Quantise a key tensor into compressed form.</summary>
    public CompressedKvSlice CompressKey(Tensor key)
    {
        // TODO: scalar quantisation using Lloyd-Max codebook
        throw new NotImplementedException();
    }

    /// <summary>Quantise a value tensor into compressed form.</summary>
    public CompressedKvSlice CompressValue(Tensor value)
    {
        // TODO: scalar quantisation using Lloyd-Max codebook
        throw new NotImplementedException();
    }

    /// <summary>Reconstruct a tensor from its compressed representation.</summary>
    public Tensor Decompress(CompressedKvSlice slice, TensorShape originalShape)
    {
        // TODO: dequantise using centroid lookup
        throw new NotImplementedException();
    }
}

/// <summary>A quantised KV-cache slice with centroid indices.</summary>
public sealed class CompressedKvSlice
{
    public required byte[] Indices { get; init; }
    public required TensorShape OriginalShape { get; init; }
    public required int BitsPerEntry { get; init; }
}
