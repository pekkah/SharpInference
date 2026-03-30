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

/// <summary>Supported element data types.</summary>
public enum DType
{
    Float32,
    Float16,
    BFloat16,
    Int8,
    UInt8,
    Int32,
    Q4_0,    // 4-bit quantized, block size 32
    Q8_0,    // 8-bit quantized, block size 32
}
