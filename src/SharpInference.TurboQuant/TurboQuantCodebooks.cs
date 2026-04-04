namespace SharpInference.TurboQuant;

/// <summary>
/// Precomputed Lloyd-Max codebooks for TurboQuant KV cache compression.
/// Generated offline by tools/CodebookGen for the Beta(d/2, d/2) distribution
/// induced by Walsh-Hadamard rotation of unit vectors.
///
/// Reference: Zandieh et al., "TurboQuant: Online Vector Quantization with
/// Near-optimal Distortion Rate" (arXiv:2504.19874), Section 3.1.
/// See also: docs/research/TurboQuant-arXiv-2504.19874.pdf
/// </summary>
public static class TurboQuantCodebooks
{
    // ── 3-bit codebooks (8 centroids, 7 boundaries) ──

    public static ReadOnlySpan<float> Centroids3Bit_D128 =>
    [
        -0.1903f, -0.1189f, -0.0669f, -0.0217f, 0.0217f, 0.0669f, 0.1189f, 0.1903f
    ];

    public static ReadOnlySpan<float> Boundaries3Bit_D128 =>
    [
        -0.1546f, -0.0929f, -0.0443f, 0.0000f, 0.0443f, 0.0929f, 0.1546f
    ];

    public static ReadOnlySpan<float> Centroids3Bit_D256 =>
    [
        -0.1346f, -0.0841f, -0.0474f, -0.0154f, 0.0154f, 0.0474f, 0.0841f, 0.1346f
    ];

    public static ReadOnlySpan<float> Boundaries3Bit_D256 =>
    [
        -0.1094f, -0.0657f, -0.0314f, 0.0000f, 0.0314f, 0.0657f, 0.1094f
    ];

    // ── 4-bit codebooks (16 centroids, 15 boundaries) ──

    public static ReadOnlySpan<float> Centroids4Bit_D128 =>
    [
        -0.2422f, -0.1836f, -0.1438f, -0.1118f, -0.0840f, -0.0586f, -0.0347f, -0.0115f,
         0.0115f,  0.0347f,  0.0586f,  0.0840f,  0.1118f,  0.1438f,  0.1836f,  0.2422f
    ];

    public static ReadOnlySpan<float> Boundaries4Bit_D128 =>
    [
        -0.2129f, -0.1637f, -0.1278f, -0.0979f, -0.0713f, -0.0467f, -0.0231f, 0.0000f,
         0.0231f,  0.0467f,  0.0713f,  0.0979f,  0.1278f,  0.1637f,  0.2129f
    ];

    public static ReadOnlySpan<float> Centroids4Bit_D256 =>
    [
        -0.1714f, -0.1300f, -0.1018f, -0.0791f, -0.0595f, -0.0415f, -0.0246f, -0.0081f,
         0.0081f,  0.0246f,  0.0415f,  0.0595f,  0.0791f,  0.1018f,  0.1300f,  0.1714f
    ];

    public static ReadOnlySpan<float> Boundaries4Bit_D256 =>
    [
        -0.1507f, -0.1159f, -0.0905f, -0.0693f, -0.0505f, -0.0330f, -0.0164f, 0.0000f,
         0.0164f,  0.0330f,  0.0505f,  0.0693f,  0.0905f,  0.1159f,  0.1507f
    ];

    /// <summary>
    /// Gets the centroids for a given bit width and dimension.
    /// </summary>
    public static ReadOnlySpan<float> GetCentroids(int bits, int dim) => (bits, dim) switch
    {
        (3, 128) => Centroids3Bit_D128,
        (3, 256) => Centroids3Bit_D256,
        (4, 128) => Centroids4Bit_D128,
        (4, 256) => Centroids4Bit_D256,
        _ => throw new ArgumentException($"No codebook for {bits}-bit, d={dim}")
    };

    /// <summary>
    /// Gets the boundaries for a given bit width and dimension.
    /// </summary>
    public static ReadOnlySpan<float> GetBoundaries(int bits, int dim) => (bits, dim) switch
    {
        (3, 128) => Boundaries3Bit_D128,
        (3, 256) => Boundaries3Bit_D256,
        (4, 128) => Boundaries4Bit_D128,
        (4, 256) => Boundaries4Bit_D256,
        _ => throw new ArgumentException($"No codebook for {bits}-bit, d={dim}")
    };
}
