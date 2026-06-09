using System.Buffers.Binary;
using System.Text;
using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class GgufModelTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var f in _tempFiles)
        {
            try { File.Delete(f); } catch { }
        }
    }

    [Fact]
    public void Open_ValidMinimalGguf_ParsesHeader()
    {
        var path = CreateMinimalGguf(metadataKvCount: 0, tensors: []);

        using var model = GgufModel.Open(path);

        Assert.Equal(3u, model.Header.Version);
        Assert.Equal(0UL, model.Header.TensorCount);
        Assert.Equal(0UL, model.Header.MetadataKvCount);
    }

    [Fact]
    public void Open_WithStringMetadata_ParsesMetadata()
    {
        var path = CreateGgufWithMetadata(new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["general.name"] = (GgufValueType.String, "TestModel"),
        });

        using var model = GgufModel.Open(path);

        Assert.Equal(2UL, model.Header.MetadataKvCount);
        Assert.Equal("llama", model.Metadata["general.architecture"]);
        Assert.Equal("TestModel", model.Metadata["general.name"]);
    }

    [Fact]
    public void Open_WithNumericMetadata_ParsesAllTypes()
    {
        var path = CreateGgufWithMetadata(new Dictionary<string, (GgufValueType, object)>
        {
            ["test.uint32"] = (GgufValueType.UInt32, 42u),
            ["test.int32"] = (GgufValueType.Int32, -7),
            ["test.float32"] = (GgufValueType.Float32, 3.14f),
            ["test.bool"] = (GgufValueType.Bool, true),
            ["test.uint64"] = (GgufValueType.UInt64, 999UL),
        });

        using var model = GgufModel.Open(path);

        Assert.Equal(42u, model.Metadata["test.uint32"]);
        Assert.Equal(-7, model.Metadata["test.int32"]);
        Assert.Equal(3.14f, (float)model.Metadata["test.float32"], 0.001f);
        Assert.Equal(true, model.Metadata["test.bool"]);
        Assert.Equal(999UL, model.Metadata["test.uint64"]);
    }

    [Fact]
    public void Open_WithTensor_ParsesTensorInfo()
    {
        var tensorData = new byte[128]; // 32 float32 elements = 128 bytes
        Random.Shared.NextBytes(tensorData);

        var path = CreateGgufWithTensors(
        [
            ("weight.0", [4, 8], DType.Float32, tensorData)
        ]);

        using var model = GgufModel.Open(path);

        Assert.Single(model.Tensors);
        var t = model.Tensors[0];
        Assert.Equal("weight.0", t.Name);
        Assert.Equal(2, t.NDimensions);
        Assert.Equal(4L, t.Dimensions[0]);
        Assert.Equal(8L, t.Dimensions[1]);
        Assert.Equal(DType.Float32, t.DType);
        Assert.Equal(32L, t.ElementCount);
        Assert.Equal(128L, t.ByteSize);
    }

    [Fact]
    public void GetTensorData_ReturnsCorrectBytes()
    {
        var tensorData = new byte[64];
        for (int i = 0; i < tensorData.Length; i++)
            tensorData[i] = (byte)(i + 1);

        var path = CreateGgufWithTensors(
        [
            ("test_tensor", [16], DType.Float32, tensorData)
        ]);

        using var model = GgufModel.Open(path);
        var data = model.GetTensorData(model.Tensors[0]);

        Assert.Equal(64, data.Length);
        for (int i = 0; i < 64; i++)
            Assert.Equal((byte)(i + 1), data[i]);
    }

    [Fact]
    public void LoadTensor_CopiesToDestination()
    {
        var tensorData = new byte[32];
        Random.Shared.NextBytes(tensorData);

        var path = CreateGgufWithTensors(
        [
            ("copy_test", [8], DType.Float32, tensorData)
        ]);

        using var model = GgufModel.Open(path);
        var dest = new byte[32];
        model.LoadTensor(model.Tensors[0], dest);

        Assert.Equal(tensorData, dest);
    }

    [Fact]
    public void FindTensor_ReturnsCorrectTensor()
    {
        var data1 = new byte[16];
        var data2 = new byte[32];

        var path = CreateGgufWithTensors(
        [
            ("layer.0.weight", [4], DType.Float32, data1),
            ("layer.1.weight", [8], DType.Float32, data2),
        ]);

        using var model = GgufModel.Open(path);

        var found = model.FindTensor("layer.1.weight");
        Assert.NotNull(found);
        Assert.Equal("layer.1.weight", found.Value.Name);
        Assert.Equal(8L, found.Value.Dimensions[0]);
    }

    [Fact]
    public void FindTensor_ReturnsNull_WhenNotFound()
    {
        var path = CreateMinimalGguf(0, []);

        using var model = GgufModel.Open(path);

        Assert.Null(model.FindTensor("nonexistent"));
    }

    [Fact]
    public void Open_InvalidMagic_Throws()
    {
        var path = CreateTempFile();
        using (var fs = File.Create(path))
        {
            Span<byte> buf = stackalloc byte[24];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, 0xDEADBEEF);
            BinaryPrimitives.WriteUInt32LittleEndian(buf[4..], 3);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[8..], 0);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[16..], 0);
            fs.Write(buf);
        }

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Open_UnsupportedVersion_Throws()
    {
        var path = CreateTempFile();
        using (var fs = File.Create(path))
        {
            Span<byte> buf = stackalloc byte[24];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, 0x46554747); // "GGUF"
            BinaryPrimitives.WriteUInt32LittleEndian(buf[4..], 1); // unsupported v1
            BinaryPrimitives.WriteUInt64LittleEndian(buf[8..], 0);
            BinaryPrimitives.WriteUInt64LittleEndian(buf[16..], 0);
            fs.Write(buf);
        }

        Assert.Throws<InvalidDataException>(() => GgufModel.Open(path));
    }

    [Fact]
    public void Open_MultipleTensors_CorrectOffsets()
    {
        // Two tensors: first 64 bytes, second 128 bytes
        var data1 = new byte[64];
        var data2 = new byte[128];
        for (int i = 0; i < 64; i++) data1[i] = 0xAA;
        for (int i = 0; i < 128; i++) data2[i] = 0xBB;

        var path = CreateGgufWithTensors(
        [
            ("tensor_a", [16], DType.Float32, data1),
            ("tensor_b", [32], DType.Float32, data2),
        ]);

        using var model = GgufModel.Open(path);

        Assert.Equal(2, model.Tensors.Count);

        var spanA = model.GetTensorData(model.Tensors[0]);
        Assert.Equal(64, spanA.Length);
        Assert.True(spanA.ToArray().All(b => b == 0xAA));

        var spanB = model.GetTensorData(model.Tensors[1]);
        Assert.Equal(128, spanB.Length);
        Assert.True(spanB.ToArray().All(b => b == 0xBB));
    }

    [Fact]
    public void DTypeInfo_ByteSize_CorrectForFloat32()
    {
        Assert.Equal(128L, DTypeInfo.ByteSize(32, DType.Float32));
    }

    [Fact]
    public void DTypeInfo_ByteSize_CorrectForQ4K()
    {
        // Q4_K: block_size=256, bytes_per_block=144
        // 256 elements => 1 block => 144 bytes
        Assert.Equal(144L, DTypeInfo.ByteSize(256, DType.Q4_K));
        Assert.Equal(288L, DTypeInfo.ByteSize(512, DType.Q4_K));
    }

    [Fact]
    public void DTypeInfo_ByteSize_CorrectForQ8_0()
    {
        // Q8_0: block_size=32, bytes_per_block=34 (32 int8 + 1 fp16 scale). This is the
        // sizing CudaBackend.Allocate uses for the q8_0 KV cache (#179) — a regression
        // here would silently under-allocate the cache.
        Assert.Equal(34L, DTypeInfo.ByteSize(32, DType.Q8_0));
        Assert.Equal(1088L, DTypeInfo.ByteSize(1024, DType.Q8_0)); // 32 blocks * 34
    }

    [Fact]
    public void GgufTensorInfo_ElementCount_IsProduct()
    {
        var info = new GgufTensorInfo("test", 3, [2, 3, 4], DType.Float32, 0);
        Assert.Equal(24L, info.ElementCount);
    }

    #region Helper methods

    private string CreateTempFile()
    {
        var path = Path.GetTempFileName();
        _tempFiles.Add(path);
        return path;
    }

    private string CreateMinimalGguf(int metadataKvCount, (string name, long[] dims, DType dtype, byte[] data)[] tensors)
    {
        var path = CreateTempFile();
        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(3, (ulong)tensors.Length, (ulong)metadataKvCount);

        // Write tensor infos
        ulong offset = 0;
        foreach (var (name, dims, dtype, data) in tensors)
        {
            writer.WriteTensorInfo(name, dims, dtype, offset);
            offset += (ulong)data.Length;
        }

        // Pad to alignment
        writer.PadToAlignment(32);

        // Write tensor data
        foreach (var (_, _, _, data) in tensors)
            fs.Write(data);

        return path;
    }

    private string CreateGgufWithMetadata(Dictionary<string, (GgufValueType type, object value)> metadata)
    {
        var path = CreateTempFile();
        using var fs = File.Create(path);
        using var writer = new GgufWriter(fs);

        writer.WriteHeader(3, 0, (ulong)metadata.Count);

        foreach (var (key, (type, value)) in metadata)
            writer.WriteMetadataKv(key, type, value);

        writer.PadToAlignment(32);
        return path;
    }

    private string CreateGgufWithTensors((string name, long[] dims, DType dtype, byte[] data)[] tensors)
    {
        return CreateMinimalGguf(0, tensors);
    }

    /// <summary>
    /// Writes GGUF binary data for test fixtures.
    /// </summary>
    private sealed class GgufWriter(Stream stream) : IDisposable
    {
        private long _bytesWritten;

        public void WriteHeader(uint version, ulong tensorCount, ulong metadataKvCount)
        {
            WriteUInt32(0x46554747); // "GGUF" magic
            WriteUInt32(version);
            WriteUInt64(tensorCount);
            WriteUInt64(metadataKvCount);
        }

        public void WriteTensorInfo(string name, long[] dims, DType dtype, ulong offset)
        {
            WriteGgufString(name);
            WriteUInt32((uint)dims.Length);
            foreach (var dim in dims)
                WriteUInt64((ulong)dim);
            WriteUInt32((uint)dtype);
            WriteUInt64(offset);
        }

        public void WriteMetadataKv(string key, GgufValueType type, object value)
        {
            WriteGgufString(key);
            WriteUInt32((uint)type);
            WriteGgufValue(type, value);
        }

        public void PadToAlignment(int alignment)
        {
            var remainder = _bytesWritten % alignment;
            if (remainder != 0)
            {
                var padding = alignment - (int)remainder;
                Span<byte> zeros = stackalloc byte[padding];
                zeros.Clear();
                stream.Write(zeros);
                _bytesWritten += padding;
            }
        }

        private void WriteGgufString(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            WriteUInt64((ulong)bytes.Length);
            stream.Write(bytes);
            _bytesWritten += bytes.Length;
        }

        private void WriteGgufValue(GgufValueType type, object value)
        {
            switch (type)
            {
                case GgufValueType.UInt8:   WriteByte((byte)value); break;
                case GgufValueType.Int8:    WriteByte((byte)(sbyte)value); break;
                case GgufValueType.UInt16:  WriteUInt16((ushort)value); break;
                case GgufValueType.Int16:   WriteInt16((short)value); break;
                case GgufValueType.UInt32:  WriteUInt32((uint)value); break;
                case GgufValueType.Int32:   WriteInt32((int)value); break;
                case GgufValueType.Float32: WriteFloat32((float)value); break;
                case GgufValueType.Bool:    WriteByte((bool)value ? (byte)1 : (byte)0); break;
                case GgufValueType.String:  WriteGgufString((string)value); break;
                case GgufValueType.UInt64:  WriteUInt64((ulong)value); break;
                case GgufValueType.Int64:   WriteInt64((long)value); break;
                case GgufValueType.Float64: WriteFloat64((double)value); break;
                default: throw new NotSupportedException($"Unsupported GGUF value type: {type}");
            }
        }

        private void WriteByte(byte v) { stream.WriteByte(v); _bytesWritten += 1; }

        private void WriteUInt16(ushort v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteUInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteInt16(short v)
        {
            Span<byte> buf = stackalloc byte[2];
            BinaryPrimitives.WriteInt16LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 2;
        }

        private void WriteUInt32(uint v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteUInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteInt32(int v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteFloat32(float v)
        {
            Span<byte> buf = stackalloc byte[4];
            BinaryPrimitives.WriteSingleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 4;
        }

        private void WriteUInt64(ulong v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteInt64(long v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        private void WriteFloat64(double v)
        {
            Span<byte> buf = stackalloc byte[8];
            BinaryPrimitives.WriteDoubleLittleEndian(buf, v);
            stream.Write(buf);
            _bytesWritten += 8;
        }

        public void Dispose() { }
    }

    #endregion
}

/// <summary>
/// Integration test that reads the real SmolLM2 GGUF file.
/// Skipped if the model file is not present.
/// </summary>
public sealed class GgufModelIntegrationTests
{
    private const string ModelPath = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";

    private static string? FindModelPath()
    {
        // Walk up from test execution directory to find the repo root
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, ModelPath);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Open_SmolLM2_ParsesCorrectly()
    {
        var path = FindModelPath();
        if (path is null) return; // Model file not available — skip

        using var model = GgufModel.Open(path);

        // Basic header checks
        Assert.True(model.Header.TensorCount > 0, "Expected at least one tensor");
        Assert.True(model.Metadata.Count > 0, "Expected metadata entries");

        // Architecture should be present
        Assert.True(model.Metadata.ContainsKey("general.architecture"),
            "Expected 'general.architecture' metadata key");

        // Should have many tensors for a 1.7B model
        Assert.True(model.Tensors.Count > 100, $"Expected >100 tensors, got {model.Tensors.Count}");

        // Each tensor should have valid dimensions and data accessible
        foreach (var tensor in model.Tensors)
        {
            Assert.False(string.IsNullOrEmpty(tensor.Name), "Tensor name should not be empty");
            Assert.True(tensor.NDimensions > 0, $"Tensor '{tensor.Name}' has 0 dimensions");
            Assert.True(tensor.ElementCount > 0, $"Tensor '{tensor.Name}' has 0 elements");
            Assert.True(tensor.ByteSize > 0, $"Tensor '{tensor.Name}' has 0 byte size");

            // Verify we can read the first byte without throwing
            var data = model.GetTensorData(tensor);
            Assert.True(data.Length > 0, $"Tensor '{tensor.Name}' returned empty data span");
        }
    }

    [Fact]
    public void Open_SmolLM2_MetadataContainsModelInfo()
    {
        var path = FindModelPath();
        if (path is null) return; // Model file not available — skip

        using var model = GgufModel.Open(path);

        var arch = model.GetMetadata<string>("general.architecture");
        Assert.False(string.IsNullOrEmpty(arch), "Architecture should not be empty");

        // SmolLM2 is a LLaMA-family model
        Assert.Equal("llama", arch);
    }
}
