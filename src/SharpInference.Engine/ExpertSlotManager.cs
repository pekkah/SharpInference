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
public sealed class ExpertSlotManager : IDisposable, IExpertPrefetchTarget
{
    private readonly VulkanBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly ExpertCache<ExpertGpuSlot> _cache;
    private readonly ExpertAccessProfiler _profiler;
    private readonly Dictionary<nint, DType> _dtypes;
    private readonly object _lock = new();
    private bool _disposed;

    // Opt-in warm-pinning of hot experts (SHARPI_MOE_WARMPIN=N). Disabled by default.
    private readonly int _warmPinPerLayer;
    private readonly long _warmPinAfter;
    private readonly int _pinBudget;
    private bool _warmed;

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
        // Frequency-aware eviction: under MoE routing skew, the least-accessed
        // probationary expert is a better victim than the strict LRU tail.
        _cache = new ExpertCache<ExpertGpuSlot>(slotCapacity, EvictSlot,
            frequencyOf: _profiler.GetAccessCount);
        _warmPinPerLayer = WarmPinConfig.ResolvePerLayer(hp.NumLayers, hp.NumExperts, hp.NumActiveExperts, slotCapacity);
        _warmPinAfter = WarmPinConfig.AfterAccesses;
        _pinBudget = Math.Max(1, slotCapacity / 2); // never pin more than half the cache
    }

    /// <summary>
    /// Return the GPU tensors for the given expert only if they are already cached.
    /// Does NOT load from disk on miss — use <see cref="GetOrLoad"/> for that.
    /// Records the lookup outcome on the profiler so frequency-aware eviction +
    /// warm-pinning have real data on the Vulkan path (where the forward pass
    /// never calls <see cref="GetOrLoad"/>).
    /// Thread-safe.
    /// </summary>
    public bool TryGetCached(int layer, int expertId, out ExpertGpuSlot slot)
    {
        lock (_lock)
        {
            bool hit = _cache.TryGet(layer, expertId, out slot);
            if (hit) _profiler.RecordHit(layer, expertId);
            else _profiler.RecordMiss(layer, expertId);
            return hit;
        }
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
            // Time only the blocking upload — the stall issue #217's overlap hides.
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            slot = UploadExpert(layer, expertId);
            _profiler.RecordMissStall(System.Diagnostics.Stopwatch.GetTimestamp() - t0);
            _cache.Put(layer, expertId, slot);
            MaybeWarmPin();
            return slot;
        }
    }

    /// <summary>
    /// Once enough routing history has accumulated, pin the hottest currently-resident
    /// experts (top <c>SHARPI_MOE_WARMPIN</c> per layer) into the protected segment so
    /// they are never evicted. No-op unless warm-pinning is enabled. Runs once, under
    /// the caller's lock. Layers are visited in descending hotness so a tight pin
    /// budget protects the layers that route most often, not whatever happens to sit
    /// at low indices (matters for hybrid GDN+MoE models where MoE FFN sits at high
    /// layer indices).
    /// </summary>
    private void MaybeWarmPin()
    {
        if (_warmed || _warmPinPerLayer <= 0) return;
        if (_profiler.TotalHits + _profiler.TotalMisses < _warmPinAfter) return;
        _warmed = true;
        var layerOrder = new int[_hp.NumLayers];
        for (int l = 0; l < _hp.NumLayers; l++) layerOrder[l] = l;
        Array.Sort(layerOrder, (a, b) => _profiler.GetLayerAccessCount(b).CompareTo(_profiler.GetLayerAccessCount(a)));
        var pinnedList = new List<(int, int)>();
        foreach (int layer in layerOrder)
        {
            if (pinnedList.Count >= _pinBudget) break;
            // Cold layers contribute nothing; once the sort hits zero we can stop.
            if (_profiler.GetLayerAccessCount(layer) == 0) break;
            foreach (int e in _profiler.GetTopExperts(layer, _warmPinPerLayer))
            {
                if (pinnedList.Count >= _pinBudget) break;
                if (_cache.Contains(layer, e)) { _cache.Pin(layer, e); pinnedList.Add((layer, e)); }
            }
        }
        _profiler.RecordWarmPin(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(pinnedList));
    }

    /// <summary>
    /// Pre-load the given expert into the cache if not already present.
    /// Uses <see cref="VulkanBackend.UploadBackground"/> so it is safe to call
    /// from a background thread concurrently with the main recording session.
    /// Also runs <see cref="MaybeWarmPin"/> so warm-pinning fires on the Vulkan
    /// path (where <see cref="GetOrLoad"/> is never called by the forward pass).
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
            MaybeWarmPin();
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
