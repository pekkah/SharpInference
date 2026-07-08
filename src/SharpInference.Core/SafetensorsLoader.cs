using System.Runtime.InteropServices;
using System.Text.Json;

namespace SharpInference.Core;

/// <summary>
/// Safetensors file reader supporting both single-file and multi-shard directory layouts.
///
/// Single file:  SafetensorsLoader.Open("model.safetensors")
/// Sharded dir:  SafetensorsLoader.OpenDirectory("path/to/model/")
///   — reads model.safetensors.index.json to map tensor names to shard files,
///     OR falls back to merging all model*.safetensors files in the directory.
///
/// Format per file: [u64-LE header_size] [header_size JSON bytes] [raw tensor data]
/// JSON maps tensor_name → {dtype, shape, data_offsets:[start, end]}.
/// </summary>
public sealed class SafetensorsLoader : IWeightLoader
{
    // One entry per tensor, points to its shard
    private sealed record TensorInfo(string Dtype, int[] Shape, long Start, long End, int ShardIndex)
    {
        public int ElementCount
        {
            get { int n = 1; foreach (var d in Shape) n *= d; return n; }
        }
    }

    private readonly List<(FileStream file, long dataOffset)> _shards;
    private readonly Dictionary<string, TensorInfo> _tensors;

    private SafetensorsLoader(List<(FileStream, long)> shards, Dictionary<string, TensorInfo> tensors)
    {
        _shards  = shards;
        _tensors = tensors;
    }

    // ── Factory methods ───────────────────────────────────────────────────

    public static SafetensorsLoader Open(string path)
    {
        var (shard, tensors) = ParseFile(path, 0);
        return new SafetensorsLoader([shard], tensors);
    }

    /// <summary>
    /// Load a multi-shard model directory.
    /// Reads model.safetensors.index.json if present; otherwise merges all model*.safetensors files.
    /// </summary>
    public static SafetensorsLoader OpenDirectory(string dir)
    {
        string indexPath = Path.Combine(dir, "model.safetensors.index.json");
        if (File.Exists(indexPath))
            return OpenFromIndex(dir, indexPath);

        // Fallback: merge all model*.safetensors (or diffusion_pytorch_model*.safetensors) in directory
        string[] candidates = [
            ..Directory.GetFiles(dir, "model*.safetensors"),
            ..Directory.GetFiles(dir, "diffusion_pytorch_model*.safetensors"),
        ];
        Array.Sort(candidates, StringComparer.Ordinal);

        if (candidates.Length == 0)
            throw new FileNotFoundException($"No safetensors files found in directory: {dir}");

        var shards  = new List<(FileStream, long)>(candidates.Length);
        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);

        for (int i = 0; i < candidates.Length; i++)
        {
            var (shard, shardTensors) = ParseFile(candidates[i], i);
            shards.Add(shard);
            foreach (var kv in shardTensors) tensors[kv.Key] = kv.Value;
        }

        return new SafetensorsLoader(shards, tensors);
    }

    private static SafetensorsLoader OpenFromIndex(string dir, string indexPath)
    {
        var indexJson = File.ReadAllBytes(indexPath);
        using var doc = JsonDocument.Parse(indexJson);
        var weightMap = doc.RootElement.GetProperty("weight_map");

        // Collect unique shard filenames in sorted order
        var shardFileNames = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var prop in weightMap.EnumerateObject())
            shardFileNames.Add(prop.Value.GetString()!);

        var shards  = new List<(FileStream, long)>(shardFileNames.Count);
        var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);

        foreach (var shardFile in shardFileNames)
        {
            int idx = shards.Count;
            var (shard, shardTensors) = ParseFile(Path.Combine(dir, shardFile), idx);
            shards.Add(shard);
            foreach (var kv in shardTensors) tensors[kv.Key] = kv.Value;
        }

        return new SafetensorsLoader(shards, tensors);
    }

    // ── Tensor access ─────────────────────────────────────────────────────

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public IEnumerable<string> TensorNames => _tensors.Keys;

    /// <summary>Read a tensor as float32. Handles F32, F16, BF16, F8_E4M3, F8_E5M2.</summary>
    public float[] ReadF32(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");

        long byteLen = info.End - info.Start;
        var raw = new byte[byteLen];
        var (file, dataOffset) = _shards[info.ShardIndex];

        lock (file)
        {
            file.Seek(dataOffset + info.Start, SeekOrigin.Begin);
            file.ReadExactly(raw);
        }

        int count  = info.ElementCount;
        var result = new float[count];

        switch (info.Dtype)
        {
            case "F32":
                MemoryMarshal.Cast<byte, float>(raw).CopyTo(result);
                break;
            case "F16":
                var f16 = MemoryMarshal.Cast<byte, Half>(raw);
                for (int i = 0; i < count; i++) result[i] = (float)f16[i];
                break;
            case "BF16":
                var bf16 = MemoryMarshal.Cast<byte, ushort>(raw);
                for (int i = 0; i < count; i++)
                    result[i] = BitConverter.Int32BitsToSingle((int)((uint)bf16[i] << 16));
                break;
            case "F8_E4M3":
                for (int i = 0; i < count; i++) result[i] = F8E4M3ToFloat(raw[i]);
                break;
            case "F8_E5M2":
                for (int i = 0; i < count; i++) result[i] = F8E5M2ToFloat(raw[i]);
                break;
            default:
                throw new NotSupportedException($"Safetensors dtype '{info.Dtype}' not supported.");
        }

        return result;
    }

    /// <summary>
    /// Read a tensor's raw (unconverted) bytes plus its safetensors dtype string
    /// (e.g. "BF16", "F32"). For consumers that keep large tensors in their storage
    /// dtype and convert rows on demand (e.g. the DSpark draft head's BF16
    /// embedding/markov tables) instead of materializing a full F32 copy.
    /// </summary>
    public byte[] ReadRaw(string name, out string dtype)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");

        long byteLen = info.End - info.Start;
        var raw = new byte[byteLen];
        var (file, dataOffset) = _shards[info.ShardIndex];

        lock (file)
        {
            file.Seek(dataOffset + info.Start, SeekOrigin.Begin);
            file.ReadExactly(raw);
        }

        dtype = info.Dtype;
        return raw;
    }

    /// <summary>Read tensor shape without loading data.</summary>
    public int[] GetShape(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Safetensors tensor not found: '{name}'");
        return (int[])info.Shape.Clone();
    }

    // ── Internal parsing ──────────────────────────────────────────────────

    private static ((FileStream file, long dataOffset), Dictionary<string, TensorInfo>) ParseFile(string path, int shardIdx)
    {
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                  bufferSize: 65536, useAsync: false);
        try
        {
            Span<byte> hdrLenBuf = stackalloc byte[8];
            file.ReadExactly(hdrLenBuf);
            ulong hdrLen = MemoryMarshal.Read<ulong>(hdrLenBuf);

            var hdrBytes   = new byte[(int)hdrLen];
            file.ReadExactly(hdrBytes);
            long dataOffset = 8L + (long)hdrLen;

            var tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);
            using var doc = JsonDocument.Parse(hdrBytes);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "__metadata__") continue;
                var obj      = prop.Value;
                var dtype    = obj.GetProperty("dtype").GetString()!;
                var shapeArr = obj.GetProperty("shape");
                var offsets  = obj.GetProperty("data_offsets");

                int[] shape = new int[shapeArr.GetArrayLength()];
                int si = 0;
                foreach (var el in shapeArr.EnumerateArray()) shape[si++] = el.GetInt32();

                long start = offsets[0].GetInt64();
                long end   = offsets[1].GetInt64();

                tensors[prop.Name] = new TensorInfo(dtype, shape, start, end, shardIdx);
            }

            return ((file, dataOffset), tensors);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    // ── F8 helpers ────────────────────────────────────────────────────────

    private static float F8E4M3ToFloat(byte b)
    {
        int sign = (b >> 7) & 1, exp = (b >> 3) & 0xF, mant = b & 0x7;
        if (exp == 0 && mant == 0) return 0f;
        float v = exp == 0
            ? MathF.Pow(2f, -6f) * (mant / 8f)
            : MathF.Pow(2f, exp - 7f) * (1f + mant / 8f);
        return sign == 0 ? v : -v;
    }

    private static float F8E5M2ToFloat(byte b)
    {
        int sign = (b >> 7) & 1, exp = (b >> 2) & 0x1F, mant = b & 0x3;
        if (exp == 0 && mant == 0) return 0f;
        float v = exp == 0
            ? MathF.Pow(2f, -14f) * (mant / 4f)
            : MathF.Pow(2f, exp - 15f) * (1f + mant / 4f);
        return sign == 0 ? v : -v;
    }

    public void Dispose()
    {
        foreach (var (file, _) in _shards) file.Dispose();
    }

    /// <inheritdoc/>
    /// Safetensors data is plain float32 (no block quantization) and is accessed via
    /// <see cref="ReadF32"/>. Raw pointer access is not supported for this backend.
    public unsafe bool TryGetRaw(string name,
        out nint dataPtr, out long byteLen,
        out DType dtype, out int rows, out int cols)
    {
        dataPtr = 0; byteLen = 0; dtype = default; rows = 0; cols = 0;
        return false;
    }
}
