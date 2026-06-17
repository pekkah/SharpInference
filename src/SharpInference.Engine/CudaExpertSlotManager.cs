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
    // fixed-stride slots. Eviction reuses a slot's offsets and never calls cudaFree —
    // zero fragmentation. Replaces the old per-tensor pooled UploadRaw allocations whose
    // power-of-two bucket rounding wasted up to ~2× VRAM per expert (e.g. 1.05 MiB Q5_K →
    // 2 MiB), holding 12 GB cards below the auto-router's 50% capacity threshold.
    //
    // Stride sizing: a role's experts do NOT all share one byte size. llama.cpp's K_M
    // mixes (and Unsloth "UD" dynamic quants) store ffn_down_exps as a larger dtype on a
    // subset of layers (e.g. Q6_K on some, Q4_K/Q5_K on the rest). The slab is shared
    // across every MoE layer of a role, so its stride is the MAXIMUM per-expert footprint
    // over all layers — smaller-quant layers simply under-fill their slot. Sizing from a
    // single (first-uploaded) layer would overflow when a later, larger-quant expert is
    // routed (UploadRawInto throws "source exceeds destination capacity"). Each upload
    // carves a view of its own actual size and tags it with its own dtype, so a slot can
    // hold a Q4_K expert from one layer and a Q6_K expert from another across evictions.
    //
    // The slab is sized for (_slabSlots) slices = the SLRU's true max residency + 1: Put
    // inserts the new entry *before* evicting the victim, so residency+1 slots are
    // transiently live — the +1 is the eviction-staging slice, recycled on the very next
    // miss. The SLRU's residency exceeds the requested slotCapacity at capacity 1 (both
    // segments floor at 1 → 2), so this is read from ExpertCache.Capacity, not slotCapacity,
    // to avoid under-provisioning _freeSlots. _freeSlots hands out the offsets.
    private readonly int _slotCapacity;
    private readonly int _slabSlots;
    private readonly Stack<int> _freeSlots;
    private RoleSlab _gateSlab;
    private RoleSlab _upSlab;
    private RoleSlab _downSlab;

    public ExpertAccessProfiler Profiler => _profiler;

    /// <summary>
    /// Total VRAM held by the expert slab(s): <c>_slabSlots × per-role max per-expert bytes</c>
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
                if (_gateSlab.Allocated) b += _gateSlab.Stride * _slabSlots;
                if (_upSlab.Allocated)   b += _upSlab.Stride   * _slabSlots;
                if (_downSlab.Allocated) b += _downSlab.Stride  * _slabSlots;
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
        // Provision physical slots from the SLRU's TRUE max residency (+1 staging), not the
        // requested slotCapacity: ExpertCache floors both segments at 1, so capacity 1 can hold
        // 2 entries and the insert-before-evict transient needs a 3rd slot. Under-seeding here
        // crashes decode with "Pop on empty stack". For slotCapacity ≥ 2 this equals slotCapacity + 1.
        _slabSlots = _cache.Capacity + 1;
        _freeSlots = new Stack<int>(_slabSlots);
        for (int i = _slabSlots - 1; i >= 0; i--) _freeSlots.Push(i);
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
            Gate: UploadExpertWeight(ref _gateSlab, "ffn_gate_exps", layer,
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Up: UploadExpertWeight(ref _upSlab, "ffn_up_exps", layer,
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Down: UploadExpertWeight(ref _downSlab, "ffn_down_exps", layer,
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId, slotIdx),
            SlotIndex: slotIdx);
    }

    private ExpertCudaSlot UploadExpertAsync(int layer, int expertId, int slotIdx)
    {
        return new ExpertCudaSlot(
            Gate: UploadExpertWeightAsync(ref _gateSlab, "ffn_gate_exps", layer,
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Up: UploadExpertWeightAsync(ref _upSlab, "ffn_up_exps", layer,
                _hp.ExpertIntermediateDim, _hp.EmbeddingDim, expertId, slotIdx),
            Down: UploadExpertWeightAsync(ref _downSlab, "ffn_down_exps", layer,
                _hp.EmbeddingDim, _hp.ExpertIntermediateDim, expertId, slotIdx),
            SlotIndex: slotIdx);
    }

    /// <summary>
    /// Lazily allocate the slab for one expert tensor role. The slab is shared by EVERY MoE
    /// layer of this role, but a role's experts do not all share one byte size (K_M mixes and
    /// "UD" dynamic quants store ffn_down at a larger dtype on a subset of layers), so the
    /// stride is the MAXIMUM per-expert footprint over all layers that carry the role — a
    /// smaller-quant expert simply under-fills its slot. Sizing from one layer would overflow
    /// when a later, larger-quant expert is routed. Per-upload, Q4_K/Q5_K/Q6_K stay raw and
    /// any other dtype expands to F32; the stride accounts for whichever is largest. The slab
    /// holds <c>_slabSlots</c> fixed-stride slices, allocated once with <c>exact: true</c>
    /// (no pool rounding). Robust to hybrid models whose MoE layers don't start at blk.0.
    /// </summary>
    private void EnsureRoleSlab(ref RoleSlab role, string roleSuffix, int rows, int cols)
    {
        if (role.Allocated) return;
        long stride = MaxRoleExpertBytes(_model, _hp.NumLayers, roleSuffix, rows, cols);
        if (stride <= 0)
            throw new InvalidOperationException(
                $"No '{roleSuffix}' expert tensor found in any layer to size the slab.");
        long slabBytes = stride * _slabSlots;
        // dtype tag is cosmetic — every carved view re-tags itself with its own per-layer dtype.
        role.Slab = _gpu.AllocateRawBytes(slabBytes, DType.Float32, exact: true);
        role.Stride = stride;
        role.Allocated = true;
    }

    /// <summary>
    /// On-VRAM bytes one expert of the given dtype occupies: raw quant bytes for
    /// Q4_K/Q5_K/Q6_K (kept quantized by the CUDA matvec), else rows×cols F32 (the
    /// dequant fallback for dtypes the matvec can't dispatch).
    /// </summary>
    private static long ExpertFootprintBytes(DType dt, int rows, int cols) =>
        dt is DType.Q4_K or DType.Q5_K or DType.Q6_K
            ? (long)rows * (cols / DTypeInfo.BlockSize(dt)) * DTypeInfo.BytesPerBlock(dt)
            : (long)rows * cols * sizeof(float);

    /// <summary>
    /// The fixed slab stride for one expert role = the MAX per-expert footprint over all MoE
    /// layers (a role's experts do NOT all share one byte size — K_M mixes and Unsloth "UD"
    /// quants store a role at a larger dtype, sometimes one expanding to F32, on a subset of
    /// layers). Public so the SLRU capacity planners
    /// (<see cref="CudaHybridForwardPass"/>.PerExpertBytes /
    /// <see cref="CudaHybridGdnForwardPass"/>.EstimatePerExpertBytes) price EXACTLY what the slab
    /// allocates: sizing from blk.0 alone under-counts, so the planner would derive more slots
    /// than the slab fits and over-commit VRAM (a hard cudaMalloc OOM when a later layer's role
    /// expands to F32). Returns 0 if no layer carries the role.
    /// </summary>
    public static long MaxRoleExpertBytes(GgufModel model, int numLayers, string roleSuffix, int rows, int cols)
    {
        long max = 0;
        for (int l = 0; l < numLayers; l++)
            if (model.FindTensor($"blk.{l}.{roleSuffix}.weight") is { } info)
                max = Math.Max(max, ExpertFootprintBytes(info.DType, rows, cols));
        return max;
    }

    /// <summary>
    /// Upload one expert's weight bytes into <paramref name="role"/>'s slab at
    /// <paramref name="slotIdx"/> and return a non-owning view tensor over that slice. The
    /// view's dtype is registered in the shared dispatch map so MatMul picks the right kernel.
    /// </summary>
    private Tensor UploadExpertWeight(ref RoleSlab role, string roleSuffix, int layer, int rows, int cols, int expertIdx, int slotIdx)
    {
        string tensorName = $"blk.{layer}.{roleSuffix}.weight";
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        EnsureRoleSlab(ref role, roleSuffix, rows, cols);
        var data = _model.GetTensorData(info);
        long byteOffset = (long)slotIdx * role.Stride;

        // Raw-vs-F32 is decided per upload from THIS layer's dtype (not a per-slab flag):
        // mixed-quant roles can land a Q4_K expert and a Q6_K expert in the same recycled slot.
        if (info.DType is DType.Q4_K or DType.Q5_K or DType.Q6_K)
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
    private Tensor UploadExpertWeightAsync(ref RoleSlab role, string roleSuffix, int layer, int rows, int cols, int expertIdx, int slotIdx)
    {
        string tensorName = $"blk.{layer}.{roleSuffix}.weight";
        var info = _model.FindTensor(tensorName)
            ?? throw new InvalidOperationException($"Missing tensor: {tensorName}");
        EnsureRoleSlab(ref role, roleSuffix, rows, cols);
        var data = _model.GetTensorData(info);
        long byteOffset = (long)slotIdx * role.Stride;

        // Raw-vs-F32 decided per upload from THIS layer's dtype (see UploadExpertWeight).
        if (info.DType is DType.Q4_K or DType.Q5_K or DType.Q6_K)
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
    /// <c>_slabSlots</c> fixed-stride slots; <see cref="Allocated"/> guards lazy init.
    /// The stride is the role's MAX per-expert footprint over all layers, so views of
    /// differing per-layer dtypes (each re-tagged on upload) coexist across recycles.
    /// </summary>
    private struct RoleSlab
    {
        public Tensor Slab;       // owning exact-size allocation (freed only on Dispose)
        public long Stride;       // bytes per expert slot (max footprint across the role's layers)
        public bool Allocated;
    }
}

/// <summary>
/// GPU tensors for one MoE expert on CUDA: gate, up, and down projection weights, plus the
/// slab <see cref="SlotIndex"/> they occupy (recycled on eviction — see <c>CudaExpertSlotManager</c>).
/// </summary>
public readonly record struct ExpertCudaSlot(Tensor Gate, Tensor Up, Tensor Down, int SlotIndex);
