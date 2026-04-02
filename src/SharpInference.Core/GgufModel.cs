using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace SharpInference.Core;

/// <summary>
/// Parses and provides zero-copy access to a GGUF model file.
/// Uses memory-mapped I/O so tensor data is paged from disk on demand.
/// </summary>
public sealed unsafe class GgufModel : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF" as little-endian uint32
    private const int DefaultAlignment = 32;

    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _accessor;
    private readonly byte* _basePtr;
    private readonly long _fileSize;
    private readonly long _dataStartOffset;

    public GgufHeader Header { get; }
    public IReadOnlyDictionary<string, object> Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    private GgufModel(
        MemoryMappedFile mmf,
        MemoryMappedViewAccessor accessor,
        byte* basePtr,
        long fileSize,
        long dataStartOffset,
        GgufHeader header,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<GgufTensorInfo> tensors)
    {
        _mmf = mmf;
        _accessor = accessor;
        _basePtr = basePtr;
        _fileSize = fileSize;
        _dataStartOffset = dataStartOffset;
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
    }

    /// <summary>
    /// Opens and parses a GGUF file. Metadata is parsed eagerly; tensor data is accessed lazily via memory mapping.
    /// </summary>
    public static GgufModel Open(string path)
    {
        var fileInfo = new FileInfo(path);
        if (!fileInfo.Exists)
            throw new FileNotFoundException("GGUF file not found.", path);

        var fileSize = fileInfo.Length;
        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mmf.CreateViewAccessor(0, fileSize, MemoryMappedFileAccess.Read);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

        try
        {
            var reader = new GgufBinaryReader(basePtr, fileSize);

            // Parse header
            var magic = reader.ReadUInt32();
            if (magic != GgufMagic)
                throw new InvalidDataException($"Invalid GGUF magic: 0x{magic:X8} (expected 0x{GgufMagic:X8})");

            var version = reader.ReadUInt32();
            if (version is < 2 or > 3)
                throw new InvalidDataException($"Unsupported GGUF version: {version} (supported: 2, 3)");

            var tensorCount = reader.ReadUInt64();
            var metadataKvCount = reader.ReadUInt64();
            var header = new GgufHeader(magic, version, tensorCount, metadataKvCount);

            // Parse metadata
            var metadata = new Dictionary<string, object>((int)metadataKvCount);
            for (ulong i = 0; i < metadataKvCount; i++)
            {
                var key = reader.ReadGgufString();
                var valueType = (GgufValueType)reader.ReadUInt32();
                var value = reader.ReadGgufValue(valueType);
                metadata[key] = value;
            }

            // Determine alignment
            var alignment = DefaultAlignment;
            if (metadata.TryGetValue("general.alignment", out var alignObj))
                alignment = Convert.ToInt32(alignObj);

            // Parse tensor infos
            var tensors = new GgufTensorInfo[(int)tensorCount];
            for (ulong i = 0; i < tensorCount; i++)
            {
                var name = reader.ReadGgufString();
                var nDims = reader.ReadUInt32();
                var dims = new long[nDims];
                for (uint d = 0; d < nDims; d++)
                    dims[d] = (long)reader.ReadUInt64();
                var dtype = (DType)reader.ReadUInt32();
                var offset = reader.ReadUInt64();
                tensors[i] = new GgufTensorInfo(name, (int)nDims, dims, dtype, offset);
            }

            // Data section starts at alignment boundary after all header/metadata/tensor-info
            var dataStartOffset = AlignUp(reader.Position, alignment);

            return new GgufModel(mmf, accessor, basePtr, fileSize, dataStartOffset, header, metadata, tensors);
        }
        catch
        {
            accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            accessor.Dispose();
            mmf.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Returns a raw pointer to the tensor data in the memory-mapped file. Zero-copy, no span overhead.
    /// The pointer is valid for the lifetime of this GgufModel instance.
    /// </summary>
    public unsafe byte* GetTensorDataPtr(GgufTensorInfo tensor) =>
        _basePtr + _dataStartOffset + (long)tensor.DataOffset;

    /// <summary>
    /// Returns a read-only span directly into the memory-mapped file for the given tensor. Zero-copy.
    /// </summary>
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        var absoluteOffset = _dataStartOffset + (long)tensor.DataOffset;
        var byteSize = tensor.ByteSize;

        if (absoluteOffset + byteSize > _fileSize)
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' data (offset={absoluteOffset}, size={byteSize}) exceeds file size ({_fileSize}).");

        return new ReadOnlySpan<byte>(_basePtr + absoluteOffset, (int)byteSize);
    }

    /// <summary>
    /// Copies tensor data from the memory-mapped file into the destination buffer.
    /// </summary>
    public void LoadTensor(GgufTensorInfo tensor, Span<byte> destination)
    {
        var data = GetTensorData(tensor);
        if (destination.Length < data.Length)
            throw new ArgumentException(
                $"Destination buffer ({destination.Length} bytes) is smaller than tensor data ({data.Length} bytes).");
        data.CopyTo(destination);
    }

    /// <summary>
    /// Finds a tensor by name, or returns null if not found.
    /// </summary>
    public GgufTensorInfo? FindTensor(string name)
    {
        for (int i = 0; i < Tensors.Count; i++)
        {
            if (Tensors[i].Name == name)
                return Tensors[i];
        }
        return null;
    }

    /// <summary>
    /// Gets a metadata value by key, or returns the default if not found.
    /// </summary>
    public T GetMetadata<T>(string key, T defaultValue = default!) =>
        Metadata.TryGetValue(key, out var value) ? (T)Convert.ChangeType(value, typeof(T)) : defaultValue;

    public void Dispose()
    {
        _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        _accessor.Dispose();
        _mmf.Dispose();
    }

    private static long AlignUp(long value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    /// <summary>
    /// Pointer-based reader for parsing GGUF binary data from a memory-mapped region.
    /// </summary>
    private ref struct GgufBinaryReader
    {
        private readonly byte* _base;
        private readonly long _length;
        private long _pos;

        public GgufBinaryReader(byte* basePtr, long length)
        {
            _base = basePtr;
            _length = length;
            _pos = 0;
        }

        public long Position => _pos;

        public byte ReadByte()
        {
            EnsureAvailable(1);
            return _base[_pos++];
        }

        public uint ReadUInt32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public int ReadInt32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadInt32LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public ulong ReadUInt64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadUInt64LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public long ReadInt64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadInt64LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public ushort ReadUInt16()
        {
            EnsureAvailable(2);
            var value = BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 2));
            _pos += 2;
            return value;
        }

        public short ReadInt16()
        {
            EnsureAvailable(2);
            var value = BinaryPrimitives.ReadInt16LittleEndian(new ReadOnlySpan<byte>(_base + _pos, 2));
            _pos += 2;
            return value;
        }

        public float ReadFloat32()
        {
            EnsureAvailable(4);
            var value = BinaryPrimitives.ReadSingleLittleEndian(new ReadOnlySpan<byte>(_base + _pos, 4));
            _pos += 4;
            return value;
        }

        public double ReadFloat64()
        {
            EnsureAvailable(8);
            var value = BinaryPrimitives.ReadDoubleLittleEndian(new ReadOnlySpan<byte>(_base + _pos, 8));
            _pos += 8;
            return value;
        }

        public bool ReadBool() => ReadByte() != 0;

        /// <summary>
        /// Reads a GGUF string: uint64 length + UTF-8 bytes (no null terminator).
        /// </summary>
        public string ReadGgufString()
        {
            var length = ReadUInt64();
            if (length > int.MaxValue)
                throw new InvalidDataException($"GGUF string length {length} exceeds maximum.");
            var len = (int)length;
            EnsureAvailable(len);
            var value = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(_base + _pos, len));
            _pos += len;
            return value;
        }

        /// <summary>
        /// Reads a typed GGUF metadata value.
        /// </summary>
        public object ReadGgufValue(GgufValueType type) => type switch
        {
            GgufValueType.UInt8   => ReadByte(),
            GgufValueType.Int8    => (sbyte)ReadByte(),
            GgufValueType.UInt16  => ReadUInt16(),
            GgufValueType.Int16   => ReadInt16(),
            GgufValueType.UInt32  => ReadUInt32(),
            GgufValueType.Int32   => ReadInt32(),
            GgufValueType.Float32 => ReadFloat32(),
            GgufValueType.Bool    => ReadBool(),
            GgufValueType.String  => ReadGgufString(),
            GgufValueType.UInt64  => ReadUInt64(),
            GgufValueType.Int64   => ReadInt64(),
            GgufValueType.Float64 => ReadFloat64(),
            GgufValueType.Array   => ReadGgufArray(),
            _ => throw new InvalidDataException($"Unknown GGUF value type: {type}")
        };

        private object[] ReadGgufArray()
        {
            var elementType = (GgufValueType)ReadUInt32();
            var count = ReadUInt64();
            if (count > int.MaxValue)
                throw new InvalidDataException($"GGUF array length {count} exceeds maximum.");
            var array = new object[(int)count];
            for (ulong i = 0; i < count; i++)
                array[i] = ReadGgufValue(elementType);
            return array;
        }

        private void EnsureAvailable(long bytes)
        {
            if (_pos + bytes > _length)
                throw new InvalidDataException(
                    $"Unexpected end of GGUF data at offset {_pos} (need {bytes} bytes, {_length - _pos} available).");
        }
    }
}
