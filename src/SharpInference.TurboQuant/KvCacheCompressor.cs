namespace SharpInference.TurboQuant;

/// <summary>
/// KV-cache compression using TurboQuant: Walsh-Hadamard rotation + Lloyd-Max
/// scalar quantization. Reduces KV-cache memory by 4-6x with minimal quality loss.
/// </summary>
public sealed class KvCacheCompressor
{
    private readonly int _bits;
    private readonly int _headDim;
    private readonly int _blockSize;
    private readonly float[] _signPattern;
    private readonly float[] _centroids;
    private readonly float[] _boundaries;

    /// <summary>Compressed block size in bytes for one head_dim vector.</summary>
    public int BlockSize => _blockSize;

    /// <summary>Quantization bit width.</summary>
    public int Bits => _bits;

    public KvCacheCompressor(int bits, int headDim, int layerIndex)
    {
        _bits = bits;
        _headDim = headDim;
        _blockSize = TurboQuantOps.BlockSize(bits, headDim);
        _signPattern = WalshHadamard.GenerateSignPattern(headDim, layerIndex);

        // Copy codebook to arrays for reuse
        ReadOnlySpan<float> c = TurboQuantCodebooks.GetCentroids(bits, headDim);
        ReadOnlySpan<float> b = TurboQuantCodebooks.GetBoundaries(bits, headDim);
        _centroids = c.ToArray();
        _boundaries = b.ToArray();
    }

    /// <summary>
    /// Compress a single head_dim-sized KV vector into a TQ block.
    /// </summary>
    /// <param name="input">Float vector of length head_dim.</param>
    /// <param name="output">Output buffer of at least <see cref="BlockSize"/> bytes.</param>
    public void Compress(ReadOnlySpan<float> input, Span<byte> output)
    {
        TurboQuantOps.Quantize(input, output, _signPattern, _centroids, _boundaries, _bits, _headDim);
    }

    /// <summary>
    /// Decompress a TQ block back to a float vector.
    /// </summary>
    public void Decompress(ReadOnlySpan<byte> compressed, Span<float> output)
    {
        TurboQuantOps.Dequantize(compressed, output, _signPattern, _centroids, _bits, _headDim);
    }

    /// <summary>
    /// Fused dequant-dot: computes dot(compressed, query) without materializing
    /// the decompressed vector. The query must be pre-rotated via <see cref="RotateQuery"/>.
    /// </summary>
    public float DequantDot(ReadOnlySpan<byte> compressed, ReadOnlySpan<float> rotatedQuery)
    {
        return TurboQuantOps.DequantDot(compressed, rotatedQuery, _centroids, _bits, _headDim);
    }

    /// <summary>
    /// Pre-rotate a query vector for use with <see cref="DequantDot"/>.
    /// Call once per head, reuse across all cached positions.
    /// </summary>
    public void RotateQuery(ReadOnlySpan<float> query, Span<float> rotatedQuery)
    {
        TurboQuantOps.RotateQuery(query, rotatedQuery, _signPattern, _headDim);
    }

    /// <summary>
    /// Returns the sign pattern for this compressor (used by GPU shaders).
    /// </summary>
    public ReadOnlySpan<float> SignPattern => _signPattern;

    /// <summary>
    /// Returns the centroids for this compressor (used by GPU shaders).
    /// </summary>
    public ReadOnlySpan<float> Centroids => _centroids;

    /// <summary>
    /// Returns the boundaries for this compressor (used by GPU shaders).
    /// </summary>
    public ReadOnlySpan<float> Boundaries => _boundaries;
}
