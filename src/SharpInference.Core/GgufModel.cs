using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text;

namespace SharpInference.Core;

/// <summary>
/// Parses and provides zero-copy access to a GGUF model file (or split multi-shard model).
/// Uses memory-mapped I/O so tensor data is paged from disk on demand.
/// Split models are detected automatically by the -00001-of-NNNNN.gguf filename pattern
/// or by the general.split.count metadata key.
/// </summary>
public sealed unsafe class GgufModel : IDisposable
{
    private const uint GgufMagic = 0x46554747; // "GGUF" as little-endian uint32
    private const int DefaultAlignment = 32;

    // Per-shard memory mapping state (index 0 = first/only shard)
    private readonly MemoryMappedFile[] _shardMmfs;
    private readonly MemoryMappedViewAccessor[] _shardAccessors;
    private readonly byte*[] _shardBasePtrs;
    private readonly long[] _shardFileSizes;
    private readonly long[] _shardDataStartOffsets;

    public GgufHeader Header { get; }
    public IReadOnlyDictionary<string, object> Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    private GgufModel(
        MemoryMappedFile[] shardMmfs,
        MemoryMappedViewAccessor[] shardAccessors,
        byte*[] shardBasePtrs,
        long[] shardFileSizes,
        long[] shardDataStartOffsets,
        GgufHeader header,
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<GgufTensorInfo> tensors)
    {
        _shardMmfs = shardMmfs;
        _shardAccessors = shardAccessors;
        _shardBasePtrs = shardBasePtrs;
        _shardFileSizes = shardFileSizes;
        _shardDataStartOffsets = shardDataStartOffsets;
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
    }

    /// <summary>
    /// Opens and parses a GGUF model file (or the first shard of a split model).
    /// If the filename matches the pattern *-00001-of-NNNNN.gguf, all sibling shards
    /// are opened automatically. Metadata is parsed eagerly; tensor data is lazy (mmap).
    /// </summary>
    public static GgufModel Open(string path)
    {
        var shardPaths = ResolveShardPaths(path);
        int shardCount = shardPaths.Length;

        var mmfs       = new MemoryMappedFile[shardCount];
        var accessors  = new MemoryMappedViewAccessor[shardCount];
        var basePtrs   = new byte*[shardCount];
        var fileSizes  = new long[shardCount];
        var dataStarts = new long[shardCount];

        try
        {
            // Open all shard files and acquire memory-mapped pointers
            for (int s = 0; s < shardCount; s++)
            {
                var fi = new FileInfo(shardPaths[s]);
                if (!fi.Exists)
                    throw new FileNotFoundException("GGUF shard file not found.", shardPaths[s]);
                fileSizes[s] = fi.Length;
                mmfs[s]      = MemoryMappedFile.CreateFromFile(shardPaths[s], FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                accessors[s] = mmfs[s].CreateViewAccessor(0, fileSizes[s], MemoryMappedFileAccess.Read);
                byte* ptr = null;
                accessors[s].SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                basePtrs[s] = ptr;
            }

            // Parse shard 0: full metadata + first batch of tensors
            var reader0 = new GgufBinaryReader(basePtrs[0], fileSizes[0]);
            ValidateMagicAndVersion(ref reader0);
            var tensorCount0 = reader0.ReadUInt64();
            var kvCount0     = reader0.ReadUInt64();
            var header       = new GgufHeader(GgufMagic, 3, tensorCount0, kvCount0);

            var metadata = new Dictionary<string, object>((int)kvCount0);
            for (ulong i = 0; i < kvCount0; i++)
            {
                var key       = reader0.ReadGgufString();
                var valueType = (GgufValueType)reader0.ReadUInt32();
                var value     = reader0.ReadGgufValue(valueType);
                metadata[key] = value;
            }

            int alignment = DefaultAlignment;
            if (metadata.TryGetValue("general.alignment", out var alignObj))
                alignment = Convert.ToInt32(alignObj);

            var allTensors = new List<GgufTensorInfo>((int)tensorCount0 * shardCount);
            ParseTensorInfos(ref reader0, tensorCount0, alignment, shardIndex: 0, fileSizes[0], out dataStarts[0], allTensors);

            // Parse remaining shards: skip their metadata, collect tensor infos
            for (int s = 1; s < shardCount; s++)
            {
                var reader = new GgufBinaryReader(basePtrs[s], fileSizes[s]);
                ValidateMagicAndVersion(ref reader);
                var tensorCountS = reader.ReadUInt64();
                var kvCountS     = reader.ReadUInt64();

                // Skip metadata entries (present but redundant in shards > 0)
                for (ulong i = 0; i < kvCountS; i++)
                {
                    reader.ReadGgufString();                         // key
                    var vt = (GgufValueType)reader.ReadUInt32();
                    reader.SkipGgufValue(vt);                        // value
                }

                ParseTensorInfos(ref reader, tensorCountS, alignment, shardIndex: s, fileSizes[s], out dataStarts[s], allTensors);
            }

            // Inject synthetic metadata from tensor inspection
            var arch = metadata.TryGetValue("general.architecture", out var archVal) ? (string)archVal : "llama";
            foreach (var t in allTensors)
            {
                if (t.Name == "blk.0.attn_q.bias")     metadata["_sharpi.has_attn_bias"] = true;
                else if (t.Name == "blk.0.attn_q_norm.weight") metadata["_sharpi.has_qk_norm"] = true;
            }

            if (!metadata.ContainsKey($"{arch}.vocab_size") &&
                metadata.TryGetValue("tokenizer.ggml.tokens", out var tokArr) && tokArr is object[] toks)
                metadata[$"{arch}.vocab_size"] = (ulong)toks.Length;

            // Report actual version from header
            var finalHeader = new GgufHeader(header.Magic, header.Version, (ulong)allTensors.Count, header.MetadataKvCount);

            return new GgufModel(mmfs, accessors, basePtrs, fileSizes, dataStarts, finalHeader, metadata, allTensors);
        }
        catch
        {
            // Cleanup on failure
            for (int s = 0; s < shardCount; s++)
            {
                if (basePtrs[s] != null) accessors[s]?.SafeMemoryMappedViewHandle.ReleasePointer();
                accessors[s]?.Dispose();
                mmfs[s]?.Dispose();
            }
            throw;
        }
    }

    /// <summary>
    /// Returns a raw pointer to the tensor data in the memory-mapped file. Zero-copy.
    /// The pointer is valid for the lifetime of this GgufModel instance.
    /// </summary>
    public unsafe byte* GetTensorDataPtr(GgufTensorInfo tensor) =>
        _shardBasePtrs[tensor.ShardIndex] + _shardDataStartOffsets[tensor.ShardIndex] + (long)tensor.DataOffset;

    /// <summary>
    /// Returns a read-only span directly into the memory-mapped file for the given tensor. Zero-copy.
    /// </summary>
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor)
    {
        int s              = tensor.ShardIndex;
        var absoluteOffset = _shardDataStartOffsets[s] + (long)tensor.DataOffset;
        var byteSize       = tensor.ByteSize;

        if (absoluteOffset + byteSize > _shardFileSizes[s])
            throw new InvalidDataException(
                $"Tensor '{tensor.Name}' data (offset={absoluteOffset}, size={byteSize}) exceeds shard {s} file size ({_shardFileSizes[s]}).");

        return new ReadOnlySpan<byte>(_shardBasePtrs[s] + absoluteOffset, (int)byteSize);
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

    /// <summary>Finds a tensor by name, or returns null if not found.</summary>
    public GgufTensorInfo? FindTensor(string name)
    {
        for (int i = 0; i < Tensors.Count; i++)
            if (Tensors[i].Name == name) return Tensors[i];
        return null;
    }

    /// <summary>Gets a metadata value by key, or returns the default if not found.</summary>
    public T GetMetadata<T>(string key, T defaultValue = default!) =>
        Metadata.TryGetValue(key, out var value) ? (T)Convert.ChangeType(value, typeof(T)) : defaultValue;

    public void Dispose()
    {
        for (int s = 0; s < _shardAccessors.Length; s++)
        {
            _shardAccessors[s].SafeMemoryMappedViewHandle.ReleasePointer();
            _shardAccessors[s].Dispose();
            _shardMmfs[s].Dispose();
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// Given a model path, returns the ordered list of shard file paths.
    /// Detects the *-00001-of-NNNNN.gguf naming convention used by llama.cpp split models.
    /// Falls back to [path] for single-file models.
    /// </summary>
    private static string[] ResolveShardPaths(string path)
    {
        var name = Path.GetFileName(path);
        // Match pattern like "Model-00001-of-00002.gguf"
        var m = System.Text.RegularExpressions.Regex.Match(
            name, @"^(.+)-(\d+)-of-(\d+)(\.gguf)$",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

        if (!m.Success) return [path];

        string prefix   = m.Groups[1].Value;
        int    total    = int.Parse(m.Groups[3].Value);
        string ext      = m.Groups[4].Value;
        string dir      = Path.GetDirectoryName(path) ?? ".";

        var paths = new string[total];
        for (int i = 0; i < total; i++)
            paths[i] = Path.Combine(dir, $"{prefix}-{i + 1:D5}-of-{total:D5}{ext}");

        return paths;
    }

    private static void ValidateMagicAndVersion(ref GgufBinaryReader reader)
    {
        var magic = reader.ReadUInt32();
        if (magic != GgufMagic)
            throw new InvalidDataException($"Invalid GGUF magic: 0x{magic:X8}");
        var version = reader.ReadUInt32();
        if (version is < 2 or > 3)
            throw new InvalidDataException($"Unsupported GGUF version: {version}");
    }

    private static void ParseTensorInfos(
        ref GgufBinaryReader reader,
        ulong tensorCount,
        int alignment,
        int shardIndex,
        long fileSize,
        out long dataStartOffset,
        List<GgufTensorInfo> result)
    {
        var tensors = new GgufTensorInfo[(int)tensorCount];
        for (ulong i = 0; i < tensorCount; i++)
        {
            var tName  = reader.ReadGgufString();
            var nDims  = reader.ReadUInt32();
            var dims   = new long[nDims];
            for (uint d = 0; d < nDims; d++)
                dims[d] = (long)reader.ReadUInt64();
            var dtype  = (DType)reader.ReadUInt32();
            var offset = reader.ReadUInt64();
            tensors[i] = new GgufTensorInfo(tName, (int)nDims, dims, dtype, offset, shardIndex);
        }

        dataStartOffset = AlignUp(reader.Position, alignment);
        result.AddRange(tensors);
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

        /// <summary>
        /// Skips over a GGUF metadata value without allocating. Used when parsing redundant
        /// metadata in shard files 1+.
        /// </summary>
        public void SkipGgufValue(GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.UInt8:
                case GgufValueType.Int8:
                case GgufValueType.Bool:  _pos += 1; break;
                case GgufValueType.UInt16:
                case GgufValueType.Int16: _pos += 2; break;
                case GgufValueType.UInt32:
                case GgufValueType.Int32:
                case GgufValueType.Float32: _pos += 4; break;
                case GgufValueType.UInt64:
                case GgufValueType.Int64:
                case GgufValueType.Float64: _pos += 8; break;
                case GgufValueType.String:
                {
                    var len = ReadUInt64();
                    _pos += (long)len;
                    break;
                }
                case GgufValueType.Array:
                {
                    var elemType = (GgufValueType)ReadUInt32();
                    var count    = ReadUInt64();
                    for (ulong i = 0; i < count; i++)
                        SkipGgufValue(elemType);
                    break;
                }
                default:
                    throw new InvalidDataException($"Unknown GGUF value type to skip: {type}");
            }
        }

        private void EnsureAvailable(long bytes)
        {
            if (_pos + bytes > _length)
                throw new InvalidDataException(
                    $"Unexpected end of GGUF data at offset {_pos} (need {bytes} bytes, {_length - _pos} available).");
        }
    }
}
