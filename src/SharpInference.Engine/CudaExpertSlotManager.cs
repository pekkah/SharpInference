using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Pipeline;

namespace SharpInference.Engine;

/// <summary>
/// Manages a CUDA-resident SLRU cache of MoE expert weight tensors.
/// Parallel implementation of <see cref="ExpertSlotManager"/> for the
/// <see cref="CudaBackend"/>: lazily loads expert weights on first access and
/// evicts cold experts when VRAM pressure requires it, enabling models whose
/// total expert weights exceed available VRAM (e.g. Qwen3.6-35B-A3B's 256
/// experts × 40 layers cannot fit a 12 GB card eagerly).
///
/// <para>
/// Kept as a near-clone of the Vulkan-backed class rather than generified over
/// <see cref="IComputeBackend"/> on purpose: the Vulkan SLRU is the hot path
/// for OLMoE / Qwen3-Coder today and we don't want this work to destabilize
/// it. The two classes should look obviously parallel; if you change one,
/// consider changing the other.
/// </para>
/// </summary>
public sealed class CudaExpertSlotManager : IDisposable, IExpertPrefetchTarget
{
    private readonly CudaBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly ExpertCache<ExpertCudaSlot> _cache;
    private readonly ExpertAccessProfiler _profiler;
    private readonly Dictionary<nint, DType> _dtypes;
    private readonly object _lock = new();
    private bool _disposed;

    // Pending background uploads keyed by tensor handle. A slot inserted via
    // Preload is in _cache immediately, but its three weight tensors' DMAs may
    // still be in flight on the upload stream; consumers consult this map and
    // either WaitForUpload (cross-stream fence on the compute stream) or skip
    // it when the event has already signaled.
    private readonly Dictionary<nint, CudaUploadHandle> _pendingUploads = new();

    // Opt-in warm-pinning of hot experts (SHARPI_MOE_WARMPIN=N). Disabled by default.
    private readonly int _warmPinPerLayer;
    private readonly long _warmPinAfter;
    private readonly int _pinBudget;
    private bool _warmed;

    public ExpertAccessProfiler Profiler => _profiler;

    /// <param name="gpu">CUDA backend to allocate/free GPU tensors on.</param>
    /// <param name="model">GGUF model for mmap weight access.</param>
    /// <param name="hp">Model hyperparameters.</param>
    /// <param name="slotCapacity">
    /// Number of expert slots to keep resident in VRAM.
    /// Size each slot as 3 GPU tensors (gate, up, down) × expert weight bytes.
    /// </param>
    /// <param name="dtypes">
    /// Shared DType map used by the CUDA MatMul dispatcher to select the right
    /// matvec kernel (Q4_K / Q5_K / Q6_K / F32). Keyed by CUDA Tensor handle,
    /// same role as the Vulkan dtype map.
    /// </param>
    public CudaExpertSlotManager(CudaBackend gpu, GgufModel model, ModelHyperparams hp,
        int slotCapacity, Dictionary<nint, DType> dtypes)
    {
        _gpu = gpu;
        _model = model;
        _hp = hp;
        _dtypes = dtypes;
        _profiler = new ExpertAccessProfiler(hp.NumLayers, hp.NumExperts);
        // Frequency-aware eviction: under MoE routing skew, the least-accessed
        // probationary expert is a better victim than the strict LRU tail.
        _cache = new ExpertCache<ExpertCudaSlot>(slotCapacity, EvictSlot,
            frequencyOf: _profiler.GetAccessCount);
        _warmPinPerLayer = WarmPinConfig.ResolvePerLayer(hp.NumLayers, hp.NumExperts, hp.NumActiveExperts, slotCapacity);
        _warmPinAfter = WarmPinConfig.AfterAccesses;
        _pinBudget = Math.Max(1, slotCapacity / 2); // never pin more than half the cache
    }

    /// <summary>
    /// Return the GPU tensors for the given expert only if they are already cached.
    /// Does NOT load from disk on miss — use <see cref="GetOrLoad"/> for that.
    /// If the slot was admitted via <see cref="Preload"/> and its DMA is still in
    /// flight, this call also inserts the cross-stream wait so the caller's
    /// compute kernels block on the upload event.
    /// Thread-safe.
    /// </summary>
    public bool TryGetCached(int layer, int expertId, out ExpertCudaSlot slot)
    {
        lock (_lock)
        {
            if (!_cache.TryGet(layer, expertId, out slot)) return false;
            FenceSlotReadyLocked(slot);
            return true;
        }
    }

    /// <summary>
    /// Return the GPU tensors for the given expert, loading from the GGUF mmap if not cached.
    /// Thread-safe: concurrent calls are serialized by an internal lock.
    /// If <see cref="Preload"/> already started a background DMA for this expert,
    /// the slot is in cache and we only need to fence the compute stream behind
    /// the upload event — no extra synchronous staging copy.
    /// </summary>
    public ExpertCudaSlot GetOrLoad(int layer, int expertId)
    {
        lock (_lock)
        {
            if (_cache.TryGet(layer, expertId, out var slot))
            {
                _profiler.RecordHit(layer, expertId);
                FenceSlotReadyLocked(slot);
                return slot;
            }

            _profiler.RecordMiss(layer, expertId);
            // Time the on-miss UploadExpert call — the synchronous expert-weight streaming
            // (host stage + H2D, drained on this or the next call) that #217's overlap aimed
            // to hide. Wraps only the upload, not the cache/lock bookkeeping.
            long t0 = Stopwatch.GetTimestamp();
            slot = UploadExpert(layer, expertId);
            _profiler.RecordMissStall(Stopwatch.GetTimestamp() - t0);
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
    ///
    /// <para>
    /// Issues three <see cref="CudaBackend.UploadBackgroundRaw"/> /
    /// <see cref="CudaBackend.UploadBackground"/> calls on the backend's dedicated
    /// upload stream so the host→device DMA can overlap with whatever is
    /// currently running on the compute stream. The slot is admitted to the
    /// cache immediately, with the per-tensor upload event tracked in
    /// <c>_pendingUploads</c>; the next <see cref="GetOrLoad"/> or
    /// <see cref="TryGetCached"/> hit will fence the compute stream behind those
    /// events via <see cref="CudaBackend.WaitForUpload"/> before returning.
    /// </para>
    /// </summary>
    public void Preload(int layer, int expertId)
    {
        lock (_lock)
        {
            if (_cache.Contains(layer, expertId)) return;
            var slot = UploadExpertAsync(layer, expertId);
            _cache.Put(layer, expertId, slot);
            MaybeWarmPin();
        }
    }

    /// <summary>
    /// If any of <paramref name="slot"/>'s tensors is still tied to a pending
    /// background upload event, insert a cross-stream wait so the compute stream
    /// blocks until the DMA completes, then release the event and forget the
    /// pending entry. Called under <c>_lock</c>.
    /// </summary>
    private void FenceSlotReadyLocked(ExpertCudaSlot slot)
    {
        FenceTensorReadyLocked(slot.Gate.Handle);
        FenceTensorReadyLocked(slot.Up.Handle);
        FenceTensorReadyLocked(slot.Down.Handle);
    }

    private void FenceTensorReadyLocked(nint handle)
    {
        if (!_pendingUploads.Remove(handle, out var pending)) return;
        _gpu.WaitForUpload(pending);
        _gpu.ReleaseUploadHandle(pending);
    }

    private void EvictSlot(ExpertCudaSlot slot)
    {
        // If a still-pending background upload is being evicted (rare — the
        // cache would only evict an unconsumed prefetch under tight capacity),
        // drain the event so the DMA isn't writing to a freed pointer.
        FenceTensorReadyLocked(slot.Gate.Handle);
        FenceTensorReadyLocked(slot.Up.Handle);
        FenceTensorReadyLocked(slot.Down.Handle);

        _dtypes.Remove(slot.Gate.Handle);
        _dtypes.Remove(slot.Up.Handle);
        _dtypes.Remove(slot.Down.Handle);
        _gpu.Free(slot.Gate);
        _gpu.Free(slot.Up);
        _gpu.Free(slot.Down);
    }

    private ExpertCudaSlot UploadExpert(int layer, int expertId)
    {
        return new ExpertCudaSlot(
            Gate: UploadExpertWeight($"blk.{layer}.ffn_gate_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId),
            Up: UploadExpertWeight($"blk.{layer}.ffn_up_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId),
            Down: UploadExpertWeight($"blk.{layer}.ffn_down_exps.weight",
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId));
    }

    private ExpertCudaSlot UploadExpertAsync(int layer, int expertId)
    {
        return new ExpertCudaSlot(
            Gate: UploadExpertWeightAsync($"blk.{layer}.ffn_gate_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId),
            Up: UploadExpertWeightAsync($"blk.{layer}.ffn_up_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId),
            Down: UploadExpertWeightAsync($"blk.{layer}.ffn_down_exps.weight",
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId));
    }

    private Tensor UploadExpertWeight(string tensorName, int rows, int cols, int expertIdx)
    {
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        var data = _model.GetTensorData(info);

        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _dtypes[result.Handle] = DType.Float32;
            return result;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (info.DType == DType.Q4_K || info.DType == DType.Q5_K || info.DType == DType.Q6_K)
        {
            // CudaBackend.UploadRaw accepts ReadOnlySpan<byte> directly — no need
            // for the float-cast trick the Vulkan port uses to fit the
            // single-overload Upload(ReadOnlySpan<float>) signature. Q5_K matters
            // here: qwen35moe stores ffn_down_exps as Q5_K, so keeping the raw
            // bytes instead of expanding to F32 halves the per-expert footprint
            // and ~doubles SLRU capacity.
            var result = _gpu.UploadRaw(expertData, TensorShape.D1(expertData.Length), info.DType);
            _dtypes[result.Handle] = info.DType;
            return result;
        }

        // Less-common dtypes (Q8_0, Q3_K, …): the CUDA matvec only dispatches on
        // Q4_K / Q5_K / Q6_K / F32, so dequantize on CPU and upload as F32. Same
        // fallback strategy the Vulkan port uses.
        int count = rows * cols;
        var f32 = new float[count];
        Dequantize.ToFloat32(expertData, f32, info.DType, count);
        var tensor = _gpu.Upload(f32, TensorShape.D1(count));
        _dtypes[tensor.Handle] = DType.Float32;
        return tensor;
    }

    /// <summary>
    /// Async sibling of <see cref="UploadExpertWeight"/>. Issues the H2D copy on
    /// the backend's upload stream and registers the returned event in
    /// <c>_pendingUploads</c>; the tensor is otherwise indistinguishable from a
    /// sync-uploaded one once <see cref="FenceTensorReadyLocked"/> has run.
    /// </summary>
    private Tensor UploadExpertWeightAsync(string tensorName, int rows, int cols, int expertIdx)
    {
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        var data = _model.GetTensorData(info);

        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var pending = _gpu.UploadBackground(floats, TensorShape.D1(floats.Length));
            _dtypes[pending.Tensor.Handle] = DType.Float32;
            _pendingUploads[pending.Tensor.Handle] = pending;
            return pending.Tensor;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (info.DType == DType.Q4_K || info.DType == DType.Q5_K || info.DType == DType.Q6_K)
        {
            var pending = _gpu.UploadBackgroundRaw(expertData, TensorShape.D1(expertData.Length), info.DType);
            _dtypes[pending.Tensor.Handle] = info.DType;
            _pendingUploads[pending.Tensor.Handle] = pending;
            return pending.Tensor;
        }

        int count = rows * cols;
        var f32 = new float[count];
        Dequantize.ToFloat32(expertData, f32, info.DType, count);
        var asyncTensor = _gpu.UploadBackground(f32, TensorShape.D1(count));
        _dtypes[asyncTensor.Tensor.Handle] = DType.Float32;
        _pendingUploads[asyncTensor.Tensor.Handle] = asyncTensor;
        return asyncTensor.Tensor;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Drain cache, invoking EvictSlot for every resident entry to free GPU tensors.
        // EvictSlot fences any still-pending uploads before the Free, so a teardown
        // mid-prefetch can't tear down memory the upload stream is still writing to.
        _cache.Drain(EvictSlot);
    }
}

/// <summary>GPU tensors for one MoE expert on CUDA: gate, up, and down projection weights.</summary>
public readonly record struct ExpertCudaSlot(Tensor Gate, Tensor Up, Tensor Down);
