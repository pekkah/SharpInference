namespace SharpInference.Core;

/// <summary>
/// GGUF metadata value type tags, matching the GGUF spec.
/// </summary>
public enum GgufValueType : uint
{
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
}

/// <summary>
/// Parsed GGUF file header.
/// </summary>
public readonly record struct GgufHeader(
    uint Magic,
    uint Version,
    ulong TensorCount,
    ulong MetadataKvCount);

/// <summary>
/// Descriptor for a single tensor in a GGUF file.
/// </summary>
public readonly record struct GgufTensorInfo(
    string Name,
    int NDimensions,
    long[] Dimensions,
    DType DType,
    ulong DataOffset)
{
    /// <summary>Total number of elements in the tensor.</summary>
    public long ElementCount
    {
        get
        {
            if (Dimensions.Length == 0) return 0;
            long count = 1;
            for (int i = 0; i < NDimensions; i++)
                count *= Dimensions[i];
            return count;
        }
    }

    /// <summary>Total byte size of the tensor data.</summary>
    public long ByteSize => DTypeInfo.ByteSize(ElementCount, DType);
}
