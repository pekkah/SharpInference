using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Pipeline;
using SharpInference.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// Manages a GPU-resident SLRU cache of MoE expert weight tensors.
/// Instead of uploading all expert weights at model-load time, this class
/// lazily loads experts on first access and evicts cold experts when VRAM
/// pressure requires it, enabling models whose total expert weights exceed
/// available VRAM.
/// </summary>
public sealed class ExpertSlotManager : IDisposable
{
    private readonly VulkanBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly ExpertCache<ExpertGpuSlot> _cache;
    private readonly ExpertAccessProfiler _profiler;
    private readonly Dictionary<nint, DType> _dtypes;
    private readonly object _lock = new();
    private bool _disposed;

    public ExpertAccessProfiler Profiler => _profiler;

    /// <param name="gpu">Vulkan backend to allocate/free GPU tensors on.</param>
    /// <param name="model">GGUF model for mmap weight access.</param>
    /// <param name="hp">Model hyperparameters.</param>
    /// <param name="slotCapacity">
    /// Number of expert slots to keep resident in VRAM.
    /// Size each slot as 3 GPU tensors (gate, up, down) × expert weight bytes.
    /// </param>
    /// <param name="dtypes">
    /// Shared DType map used by <c>GpuMatMul</c> to select the right shader.
    /// </param>
    public ExpertSlotManager(VulkanBackend gpu, GgufModel model, ModelHyperparams hp,
        int slotCapacity, Dictionary<nint, DType> dtypes)
    {
        _gpu = gpu;
        _model = model;
        _hp = hp;
        _dtypes = dtypes;
        _profiler = new ExpertAccessProfiler(hp.NumLayers, hp.NumExperts);
        _cache = new ExpertCache<ExpertGpuSlot>(slotCapacity, EvictSlot);
    }

    /// <summary>
    /// Return the GPU tensors for the given expert only if they are already cached.
    /// Does NOT load from disk on miss — use <see cref="GetOrLoad"/> for that.
    /// Thread-safe.
    /// </summary>
    public bool TryGetCached(int layer, int expertId, out ExpertGpuSlot slot)
    {
        lock (_lock)
            return _cache.TryGet(layer, expertId, out slot);
    }

    /// <summary>
    /// Return the GPU tensors for the given expert, loading from the GGUF mmap if not cached.
    /// Thread-safe: concurrent calls are serialized by an internal lock.
    /// </summary>
    public ExpertGpuSlot GetOrLoad(int layer, int expertId)
    {
        lock (_lock)
        {
            if (_cache.TryGet(layer, expertId, out var slot))
            {
                _profiler.RecordHit(layer, expertId);
                return slot;
            }

            _profiler.RecordMiss(layer, expertId);
            slot = UploadExpert(layer, expertId);
            _cache.Put(layer, expertId, slot);
            return slot;
        }
    }

    /// <summary>
    /// Pre-load the given expert into the cache if not already present.
    /// Uses <see cref="VulkanBackend.UploadBackground"/> so it is safe to call
    /// from a background thread concurrently with the main recording session.
    /// </summary>
    public void Preload(int layer, int expertId)
    {
        lock (_lock)
        {
            if (!_cache.Contains(layer, expertId))
            {
                var slot = UploadExpert(layer, expertId, background: true);
                _cache.Put(layer, expertId, slot);
            }
        }
    }

    private void EvictSlot(ExpertGpuSlot slot)
    {
        _dtypes.Remove(slot.Gate.Handle);
        _dtypes.Remove(slot.Up.Handle);
        _dtypes.Remove(slot.Down.Handle);
        _gpu.Free(slot.Gate);
        _gpu.Free(slot.Up);
        _gpu.Free(slot.Down);
    }

    private ExpertGpuSlot UploadExpert(int layer, int expertId, bool background = false)
    {
        return new ExpertGpuSlot(
            Gate: UploadExpertWeight($"blk.{layer}.ffn_gate_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, background),
            Up: UploadExpertWeight($"blk.{layer}.ffn_up_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, background),
            Down: UploadExpertWeight($"blk.{layer}.ffn_down_exps.weight",
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId, background));
    }

    private Tensor UploadExpertWeight(string tensorName, int rows, int cols, int expertIdx,
        bool background = false)
    {
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        var data = _model.GetTensorData(info);

        Func<ReadOnlySpan<float>, TensorShape, Tensor> upload = background
            ? _gpu.UploadBackground
            : (d, s) => _gpu.Upload(d, s);

        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var result = upload(floats, TensorShape.D1(floats.Length));
            _dtypes[result.Handle] = DType.Float32;
            return result;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            int floatCount = expertData.Length / 4;
            var rawFloats = new float[floatCount];
            expertData.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            var result = upload(rawFloats, TensorShape.D1(floatCount));
            _dtypes[result.Handle] = info.DType;
            return result;
        }

        int count = rows * cols;
        var f32 = new float[count];
        Dequantize.ToFloat32(expertData, f32, info.DType, count);
        var tensor = upload(f32, TensorShape.D1(count));
        _dtypes[tensor.Handle] = DType.Float32;
        return tensor;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Drain cache, invoking EvictSlot for every resident entry to free GPU tensors.
        _cache.Drain(EvictSlot);
    }
}

/// <summary>GPU tensors for one MoE expert: gate, up, and down projection weights.</summary>
public readonly record struct ExpertGpuSlot(Tensor Gate, Tensor Up, Tensor Down);
