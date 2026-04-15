using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Diffusion;

/// <summary>
/// IWeightLoader backed by a GGUF file (via GgufModel + Dequantize).
///
/// GGUF files converted from diffusers safetensors typically preserve the
/// original tensor names (e.g. "all_x_embedder.2-1.weight").
/// If a name is not found verbatim, a few common prefix patterns are tried
/// ("model.diffusion_model.", "transformer.", "model.") so that both
/// naming conventions are handled transparently.
/// </summary>
public sealed class GgufWeightLoader : IWeightLoader
{
    private readonly GgufModel _model;
    private readonly Dictionary<string, float[]> _cache = new(StringComparer.Ordinal);
    private readonly HashSet<string> _knownNames;
    private bool _disposed;

    private static readonly string[] _prefixTries =
    [
        "",
        "model.diffusion_model.",
        "transformer.",
        "model.",
    ];

    public GgufWeightLoader(GgufModel model)
    {
        _model = model;
        _knownNames = new HashSet<string>(
            model.Tensors.Select(t => t.Name), StringComparer.Ordinal);
    }

    public static GgufWeightLoader Open(string path)
        => new(GgufModel.Open(path));

    // ── IWeightLoader ─────────────────────────────────────────────────────

    public bool Contains(string name) => Resolve(name) is not null;

    /// <summary>
    /// Dequantize tensor to float32. Results are cached for small tensors (&lt;= 1 M elements);
    /// large weight matrices should be accessed via <see cref="TryGetRaw"/> instead.
    /// </summary>
    public float[] ReadF32(string name)
    {
        if (_cache.TryGetValue(name, out var cached)) return cached;

        var resolved = Resolve(name)
            ?? throw new KeyNotFoundException($"GGUF tensor not found: '{name}'");

        var info = _model.FindTensor(resolved)!.Value;
        long count = info.ElementCount;
        var raw    = _model.GetTensorData(info);
        var result = new float[count];

        Dequantize.ToFloat32(raw, result, info.DType, count);

        // Only cache small tensors (norms, biases, small embeddings) to avoid GB-scale heap usage.
        // Large weight matrices should be accessed via TryGetRaw → SimdKernels.MatMulBatched.
        if (count <= 1_000_000)
            _cache[name] = result;

        return result;
    }

    /// <inheritdoc/>
    public unsafe bool TryGetRaw(string name,
        out nint dataPtr, out long byteLen,
        out DType dtype, out int rows, out int cols)
    {
        string? resolved = Resolve(name);
        if (resolved is null || _model.FindTensor(resolved) is not { } info || info.NDimensions < 2)
        {
            dataPtr = 0; byteLen = 0; dtype = default; rows = 0; cols = 0;
            return false;
        }

        // Zero-copy: return a direct pointer into the memory-mapped file data.
        // GGUF convention: Dimensions[0] = ne0 = innermost = input features (cols),
        //                  Dimensions[1] = ne1 = outer     = output features (rows).
        dataPtr = (nint)_model.GetTensorDataPtr(info);
        byteLen = info.ByteSize;
        dtype   = info.DType;
        cols    = (int)info.Dimensions[0];
        rows    = (int)info.Dimensions[1];
        return true;
    }

    // ── Name resolution ───────────────────────────────────────────────────

    private string? Resolve(string name)
    {
        foreach (var prefix in _prefixTries)
        {
            var candidate = prefix.Length == 0 ? name : prefix + name;
            if (_knownNames.Contains(candidate)) return candidate;
        }
        return null;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────

    /// <summary>List all tensor names in the GGUF file (for debugging weight name mismatches).</summary>
    public IEnumerable<string> TensorNames => _knownNames;

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _cache.Clear();
            _model.Dispose();
        }
    }
}
