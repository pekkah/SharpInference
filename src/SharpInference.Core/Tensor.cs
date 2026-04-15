namespace SharpInference.Core;

/// <summary>
/// A multi-dimensional array of elements residing on a compute backend.
/// Shape and strides follow row-major (C) order.
/// </summary>
public sealed class Tensor : IDisposable
{
    public TensorShape Shape { get; }
    public DType DType { get; }

    /// <summary>Opaque handle owned by the backend that allocated this tensor.</summary>
    public nint Handle { get; }

    public Tensor(TensorShape shape, DType dtype, nint handle)
    {
        Shape = shape;
        DType = dtype;
        Handle = handle;
    }

    public long ElementCount => Shape.ElementCount;

    public void Dispose() { /* backend-owned; disposed via backend */ }
}

/// <summary>N-dimensional shape descriptor.</summary>
public readonly record struct TensorShape(long[] Dims)
{
    public int Rank => Dims.Length;
    public long ElementCount => Dims.Aggregate(1L, (a, d) => a * d);

    public static TensorShape D1(long d0) => new([d0]);
    public static TensorShape D2(long d0, long d1) => new([d0, d1]);
    public static TensorShape D3(long d0, long d1, long d2) => new([d0, d1, d2]);
    public static TensorShape D4(long d0, long d1, long d2, long d3) => new([d0, d1, d2, d3]);
}

/// <summary>
/// Supported element data types, matching the GGML type enum values.
/// Numeric values match the GGUF spec for direct binary parsing.
/// </summary>
public enum DType : uint
{
    Float32   = 0,
    Float16   = 1,
    Q4_0      = 2,
    Q4_1      = 3,
    // 4, 5 are deprecated/removed
    Q5_0      = 6,
    Q5_1      = 7,
    Q8_0      = 8,
    Q8_1      = 9,
    Q2_K      = 10,
    Q3_K      = 11,
    Q4_K      = 12,
    Q5_K      = 13,
    Q6_K      = 14,
    Q8_K      = 15,
    IQ2_XXS   = 16,
    IQ2_XS    = 17,
    IQ3_XXS   = 18,
    IQ1_S     = 19,
    IQ4_NL    = 20,
    IQ3_S     = 21,
    IQ2_S     = 22,
    IQ4_XS    = 23,
    Int8      = 24,
    Int16     = 25,
    Int32     = 26,
    Int64     = 27,
    Float64   = 28,
    IQ1_M     = 29,
    BFloat16  = 30,
    Q4_0_4_4  = 31,
    Q4_0_4_8  = 32,
    Q4_0_8_8  = 33,
    TQ1_0     = 34,
    TQ2_0     = 35,
    Float8E4M3 = 36,
}

/// <summary>
/// Provides block-size and byte-size information for GGML quantized types.
/// </summary>
public static class DTypeInfo
{
    /// <summary>
    /// Returns the block size for a quantized type, or 1 for scalar types.
    /// </summary>
    public static int BlockSize(DType dtype) => dtype switch
    {
        DType.Float32   => 1,
        DType.Float16   => 1,
        DType.BFloat16  => 1,
        DType.Float64   => 1,
        DType.Int8      => 1,
        DType.Int16     => 1,
        DType.Int32     => 1,
        DType.Int64     => 1,
        DType.Q4_0      => 32,
        DType.Q4_1      => 32,
        DType.Q5_0      => 32,
        DType.Q5_1      => 32,
        DType.Q8_0      => 32,
        DType.Q8_1      => 32,
        DType.Q2_K      => 256,
        DType.Q3_K      => 256,
        DType.Q4_K      => 256,
        DType.Q5_K      => 256,
        DType.Q6_K      => 256,
        DType.Q8_K      => 256,
        DType.IQ2_XXS   => 256,
        DType.IQ2_XS    => 256,
        DType.IQ3_XXS   => 256,
        DType.IQ1_S     => 256,
        DType.IQ4_NL    => 32,
        DType.IQ3_S     => 256,
        DType.IQ2_S     => 256,
        DType.IQ4_XS    => 256,
        DType.IQ1_M     => 256,
        DType.Q4_0_4_4  => 32,
        DType.Q4_0_4_8  => 32,
        DType.Q4_0_8_8  => 32,
        DType.TQ1_0     => 256,
        DType.TQ2_0     => 256,
        DType.Float8E4M3 => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown dtype")
    };

    /// <summary>
    /// Returns the number of bytes per block for a given type.
    /// For scalar types this is just the element size.
    /// </summary>
    public static int BytesPerBlock(DType dtype) => dtype switch
    {
        DType.Float32   => 4,
        DType.Float16   => 2,
        DType.BFloat16  => 2,
        DType.Float64   => 8,
        DType.Int8      => 1,
        DType.Int16     => 2,
        DType.Int32     => 4,
        DType.Int64     => 8,
        DType.Q4_0      => 18,   // 2 (scale) + 16 (32 * 4 bits / 8)
        DType.Q4_1      => 20,   // 2 (scale) + 2 (min) + 16
        DType.Q5_0      => 22,   // 2 (scale) + 4 (high bits) + 16
        DType.Q5_1      => 24,   // 2 (scale) + 2 (min) + 4 + 16
        DType.Q8_0      => 34,   // 2 (scale) + 32
        DType.Q8_1      => 36,   // 4 (scale) + 32 (note: Q8_1 uses f32 scale)
        DType.Q2_K      => 84,
        DType.Q3_K      => 110,
        DType.Q4_K      => 144,
        DType.Q5_K      => 176,
        DType.Q6_K      => 210,
        DType.Q8_K      => 292,
        DType.IQ2_XXS   => 66,
        DType.IQ2_XS    => 74,
        DType.IQ3_XXS   => 98,
        DType.IQ1_S     => 50,
        DType.IQ4_NL    => 18,
        DType.IQ3_S     => 110,
        DType.IQ2_S     => 82,
        DType.IQ4_XS    => 136,
        DType.IQ1_M     => 56,
        DType.Q4_0_4_4  => 18,
        DType.Q4_0_4_8  => 18,
        DType.Q4_0_8_8  => 18,
        DType.TQ1_0     => 54,
        DType.TQ2_0     => 66,
        DType.Float8E4M3 => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Unknown dtype")
    };

    /// <summary>
    /// Returns the byte size of one element for non-quantized types.
    /// For quantized types, use ByteSize() instead.
    /// </summary>
    public static int BytesPerElement(DType dtype) => dtype switch
    {
        DType.Float32  => 4,
        DType.Float16  => 2,
        DType.BFloat16 => 2,
        DType.Float64  => 8,
        DType.Int8     => 1,
        DType.Int16    => 2,
        DType.Int32    => 4,
        DType.Int64    => 8,
        DType.Float8E4M3 => 1,
        _ => throw new ArgumentException($"BytesPerElement not supported for quantized type {dtype}")
    };

    /// <summary>
    /// Computes the total byte size for a tensor with the given element count and dtype.
    /// </summary>
    public static long ByteSize(long elementCount, DType dtype)
    {
        var blockSize = BlockSize(dtype);
        var bytesPerBlock = BytesPerBlock(dtype);
        return (elementCount / blockSize) * bytesPerBlock;
    }
}
