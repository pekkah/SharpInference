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
///
/// <para>
/// <b>Divergence (issue #216):</b> this CUDA class allocates a preallocated, exact-size
/// expert <i>slab</i> and carves fixed-stride slot views out of it (no per-expert pool
/// allocation, no power-of-two bucket rounding, no <c>cudaFree</c> on eviction). The Vulkan
/// twin still uses pooled per-tensor uploads and is a candidate for the same treatment once
/// the CUDA slab is validated on hardware — it was left untouched deliberately to avoid
/// destabilizing the Vulkan hot path.
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

    // ── Exact-size expert slab (issue #216) ──────────────────────────────────
    // One preallocated slab per expert tensor role (gate/up/down), carved into
    // fixed-stride slots. All experts of a given tensor name share identical byte
    // sizes (rows × bytesPerRow, fixed per model), so a slab has zero fragmentation
    // risk: eviction reuses the slot's offsets and never calls cudaFree. Replaces the
    // old per-tensor pooled UploadRaw allocations whose power-of-two bucket rounding
    // wasted up to ~2× VRAM per expert (e.g. 1.05 MiB Q5_K → 2 MiB), holding 12 GB
    // cards below the auto-router's 50% capacity threshold.
    //
    // The slab is sized for (_slotCapacity + 1) slices: the SLRU's Put inserts the new
    // entry *before* evicting the victim, so capacity+1 slots are transiently live —
    // the +1 is the eviction-staging slice, recycled on the very next miss (mirrors the
    // prior pooled path's transient capacity+1 peak). _freeSlots hands out the offsets.
    private readonly int _slotCapacity;
    private readonly Stack<int> _freeSlots;
    private RoleSlab _gateSlab;
    private RoleSlab _upSlab;
    private RoleSlab _downSlab;

    public ExpertAccessProfiler Profiler => _profiler;

    /// <summary>
    /// Total VRAM held by the expert slab(s): <c>(slotCapacity + 1) × per-expert bytes</c>
    /// once all three roles have been allocated (slabs are allocated lazily on first upload).
    /// Used to verify the cache footprint matches the exact-size accounting (issue #216).
    /// </summary>
    public long ExpertCacheVramBytes
    {
        get
        {
            lock (_lock)
            {
                long b = 0;
                if (_gateSlab.Allocated) b += _gateSlab.Stride * (_slotCapacity + 1);
                if (_upSlab.Allocated)   b += _upSlab.Stride   * (_slotCapacity + 1);
                if (_downSlab.Allocated) b += _downSlab.Stride  * (_slotCapacity + 1);
                return b;
            }
        }
    }

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

        _slotCapacity = slotCapacity;
        // Seed (slotCapacity + 1) free slot indices; see _gateSlab note for the +1.
        _freeSlots = new Stack<int>(slotCapacity + 1);
        for (int i = slotCapacity; i >= 0; i--) _freeSlots.Push(i);
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
            int slotIdx = _freeSlots.Pop();
            long t0 = Stopwatch.GetTimestamp();
            try { slot = UploadExpert(layer, expertId, slotIdx); }
            catch { _freeSlots.Push(slotIdx); throw; }
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
            int slotIdx = _freeSlots.Pop();
            ExpertCudaSlot slot;
            try { slot = UploadExpertAsync(layer, expertId, slotIdx); }
            catch { _freeSlots.Push(slotIdx); throw; }
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
        // drain the event so the DMA isn't writing to a slot about to be reused.
        FenceTensorReadyLocked(slot.Gate.Handle);
        FenceTensorReadyLocked(slot.Up.Handle);
        FenceTensorReadyLocked(slot.Down.Handle);

        _dtypes.Remove(slot.Gate.Handle);
        _dtypes.Remove(slot.Up.Handle);
        _dtypes.Remove(slot.Down.Handle);
        // The three tensors are non-owning slab views: Free drops their handle
        // registration only — NO cudaFree. The slab memory is recycled by handing
        // the slot index back to the free list for the next miss to overwrite.
        _gpu.Free(slot.Gate);
        _gpu.Free(slot.Up);
        _gpu.Free(slot.Down);
        _freeSlots.Push(slot.SlotIndex);
    }

    private ExpertCudaSlot UploadExpert(int layer, int expertId, int slotIdx)
    {
        return new ExpertCudaSlot(
            Gate: UploadExpertWeight(ref _gateSlab, $"blk.{layer}.ffn_gate_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Up: UploadExpertWeight(ref _upSlab, $"blk.{layer}.ffn_up_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Down: UploadExpertWeight(ref _downSlab, $"blk.{layer}.ffn_down_exps.weight",
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId, slotIdx),
            SlotIndex: slotIdx);
    }

    private ExpertCudaSlot UploadExpertAsync(int layer, int expertId, int slotIdx)
    {
        return new ExpertCudaSlot(
            Gate: UploadExpertWeightAsync(ref _gateSlab, $"blk.{layer}.ffn_gate_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Up: UploadExpertWeightAsync(ref _upSlab, $"blk.{layer}.ffn_up_exps.weight",
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Down: UploadExpertWeightAsync(ref _downSlab, $"blk.{layer}.ffn_down_exps.weight",
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId, slotIdx),
            SlotIndex: slotIdx);
    }

    /// <summary>
    /// Lazily allocate the slab for one expert tensor role from the model's actual tensor
    /// (dtype + dimensions). All layers/experts of a role share identical byte sizes, so the
    /// first upload defines the slab — this also handles hybrid models where the MoE layers
    /// don't start at blk.0. Q4_K/Q5_K/Q6_K stay raw; every other dtype expands to F32 (the
    /// CUDA matvec only dispatches Q4_K/Q5_K/Q6_K/F32). The slab holds (slotCapacity + 1)
    /// fixed-stride slices, allocated once with <c>exact: true</c> (no pool rounding).
    /// </summary>
    private void EnsureRoleSlab(ref RoleSlab role, in GgufTensorInfo info, int rows, int cols)
    {
        if (role.Allocated) return;
        bool raw = info.DType is DType.Q4_K or DType.Q5_K or DType.Q6_K;
        long stride;
        DType viewDType;
        if (raw)
        {
            int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
            stride = (long)rows * bytesPerRow;
            viewDType = info.DType;
        }
        else
        {
            // F32 native or F32-dequant fallback: one slot is rows×cols floats.
            stride = (long)rows * cols * sizeof(float);
            viewDType = DType.Float32;
        }
        long slabBytes = stride * (_slotCapacity + 1);
        role.Slab = _gpu.AllocateRawBytes(slabBytes, viewDType, exact: true);
        role.Stride = stride;
        role.ViewDType = viewDType;
        role.Raw = raw;
        role.Allocated = true;
    }

    /// <summary>
    /// Upload one expert's weight bytes into <paramref name="role"/>'s slab at
    /// <paramref name="slotIdx"/> and return a non-owning view tensor over that slice. The
    /// view's dtype is registered in the shared dispatch map so MatMul picks the right kernel.
    /// </summary>
    private Tensor UploadExpertWeight(ref RoleSlab role, string tensorName, int rows, int cols, int expertIdx, int slotIdx)
    {
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        EnsureRoleSlab(ref role, info, rows, cols);
        var data = _model.GetTensorData(info);
        long byteOffset = (long)slotIdx * role.Stride;

        if (role.Raw)
        {
            int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
            int expertBytes = rows * bytesPerRow;
            var expertData = data.Slice(expertIdx * expertBytes, expertBytes);
            var view = _gpu.ViewRawBytes(role.Slab, byteOffset, role.Stride,
                TensorShape.D1(expertData.Length), info.DType);
            _gpu.UploadRawInto(view, expertData);
            _dtypes[view.Handle] = info.DType;
            return view;
        }

        // F32 slab slot: copy native floats or dequantize the source dtype into it.
        int count = rows * cols;
        var fView = _gpu.ViewRawBytes(role.Slab, byteOffset, role.Stride,
            TensorShape.D1(count), DType.Float32);
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(expertIdx * count, count);
            _gpu.UploadInto(fView, floats);
        }
        else
        {
            int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
            int expertBytes = rows * bytesPerRow;
            var expertData = data.Slice(expertIdx * expertBytes, expertBytes);
            var f32 = new float[count];
            Dequantize.ToFloat32(expertData, f32, info.DType, count);
            _gpu.UploadInto(fView, f32);
        }
        _dtypes[fView.Handle] = DType.Float32;
        return fView;
    }

    /// <summary>
    /// Async sibling of <see cref="UploadExpertWeight"/>. Issues the H2D copy into the slab
    /// slot on the backend's upload stream and registers the returned event in
    /// <c>_pendingUploads</c>; the view is otherwise indistinguishable from a sync-uploaded
    /// one once <see cref="FenceTensorReadyLocked"/> has run.
    /// </summary>
    private Tensor UploadExpertWeightAsync(ref RoleSlab role, string tensorName, int rows, int cols, int expertIdx, int slotIdx)
    {
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        EnsureRoleSlab(ref role, info, rows, cols);
        var data = _model.GetTensorData(info);
        long byteOffset = (long)slotIdx * role.Stride;

        if (role.Raw)
        {
            int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
            int expertBytes = rows * bytesPerRow;
            var expertData = data.Slice(expertIdx * expertBytes, expertBytes);
            var view = _gpu.ViewRawBytes(role.Slab, byteOffset, role.Stride,
                TensorShape.D1(expertData.Length), info.DType);
            var pending = _gpu.UploadBackgroundRawInto(view, expertData);
            _dtypes[view.Handle] = info.DType;
            _pendingUploads[view.Handle] = pending;
            return view;
        }

        int count = rows * cols;
        var fView = _gpu.ViewRawBytes(role.Slab, byteOffset, role.Stride,
            TensorShape.D1(count), DType.Float32);
        CudaUploadHandle fPending;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(expertIdx * count, count);
            fPending = _gpu.UploadBackgroundInto(fView, floats);
        }
        else
        {
            int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
            int expertBytes = rows * bytesPerRow;
            var expertData = data.Slice(expertIdx * expertBytes, expertBytes);
            var f32 = new float[count];
            Dequantize.ToFloat32(expertData, f32, info.DType, count);
            fPending = _gpu.UploadBackgroundInto(fView, f32);
        }
        _dtypes[fView.Handle] = DType.Float32;
        _pendingUploads[fView.Handle] = fPending;
        return fView;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        // Drain cache, invoking EvictSlot for every resident entry. EvictSlot fences any
        // still-pending uploads and drops the slab-view handle registrations (no cudaFree),
        // so a teardown mid-prefetch can't tear down memory the upload stream is still
        // writing to. Then free the slabs themselves — the only owning allocations.
        _cache.Drain(EvictSlot);
        if (_gateSlab.Allocated) _gpu.Free(_gateSlab.Slab);
        if (_upSlab.Allocated)   _gpu.Free(_upSlab.Slab);
        if (_downSlab.Allocated) _gpu.Free(_downSlab.Slab);
    }

    /// <summary>
    /// One preallocated expert-tensor slab (gate, up, or down). Carved into
    /// (slotCapacity + 1) fixed-stride slots; <see cref="Allocated"/> guards lazy init.
    /// </summary>
    private struct RoleSlab
    {
        public Tensor Slab;       // owning exact-size allocation (freed only on Dispose)
        public long Stride;       // bytes per expert slot
        public DType ViewDType;   // dtype each carved view is tagged with
        public bool Raw;          // true → raw quant bytes; false → F32 (native or dequant)
        public bool Allocated;
    }
}

/// <summary>
/// GPU tensors for one MoE expert on CUDA: gate, up, and down projection weights, plus the
/// slab <see cref="SlotIndex"/> they occupy (recycled on eviction — see <c>CudaExpertSlotManager</c>).
/// </summary>
public readonly record struct ExpertCudaSlot(Tensor Gate, Tensor Up, Tensor Down, int SlotIndex);
