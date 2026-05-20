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
public sealed class CudaExpertSlotManager : IDisposable
{
    private readonly CudaBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly ExpertCache<ExpertCudaSlot> _cache;
    private readonly ExpertAccessProfiler _profiler;
    private readonly Dictionary<nint, DType> _dtypes;
    private readonly object _lock = new();
    private bool _disposed;

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
    /// matvec kernel (Q4_K / Q6_K / F32). Keyed by CUDA Tensor handle, same role
    /// as the Vulkan dtype map.
    /// </param>
    public CudaExpertSlotManager(CudaBackend gpu, GgufModel model, ModelHyperparams hp,
        int slotCapacity, Dictionary<nint, DType> dtypes)
    {
        _gpu = gpu;
        _model = model;
        _hp = hp;
        _dtypes = dtypes;
        _profiler = new ExpertAccessProfiler(hp.NumLayers, hp.NumExperts);
        _cache = new ExpertCache<ExpertCudaSlot>(slotCapacity, EvictSlot);
    }

    /// <summary>
    /// Return the GPU tensors for the given expert only if they are already cached.
    /// Does NOT load from disk on miss — use <see cref="GetOrLoad"/> for that.
    /// Thread-safe.
    /// </summary>
    public bool TryGetCached(int layer, int expertId, out ExpertCudaSlot slot)
    {
        lock (_lock)
            return _cache.TryGet(layer, expertId, out slot);
    }

    /// <summary>
    /// Return the GPU tensors for the given expert, loading from the GGUF mmap if not cached.
    /// Thread-safe: concurrent calls are serialized by an internal lock.
    /// </summary>
    public ExpertCudaSlot GetOrLoad(int layer, int expertId)
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
    ///
    /// <para>
    /// Unlike the Vulkan port (which has a dedicated <c>UploadBackground</c> path
    /// for off-recording-session uploads), CUDA does not expose a separate
    /// background-upload entry point. We use the synchronous <see cref="CudaBackend.Upload"/>
    /// / <see cref="CudaBackend.UploadRaw"/> path here. This is acceptable for v1
    /// because each call still issues a true async PCIe DMA under the hood
    /// (cudaMemcpyAsync via the pinned staging buffer on the backend's stream),
    /// so the bulk transfer overlaps with whatever else is on the stream. The
    /// practical hit vs Vulkan's UploadBackground is small. If prefetcher
    /// pipelining is needed later we can plumb a dedicated upload stream.
    /// </para>
    /// </summary>
    public void Preload(int layer, int expertId)
    {
        lock (_lock)
        {
            if (!_cache.Contains(layer, expertId))
            {
                var slot = UploadExpert(layer, expertId);
                _cache.Put(layer, expertId, slot);
            }
        }
    }

    private void EvictSlot(ExpertCudaSlot slot)
    {
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

        if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            // CudaBackend.UploadRaw accepts ReadOnlySpan<byte> directly — no need
            // for the float-cast trick the Vulkan port uses to fit the
            // single-overload Upload(ReadOnlySpan<float>) signature.
            var result = _gpu.UploadRaw(expertData, TensorShape.D1(expertData.Length), info.DType);
            _dtypes[result.Handle] = info.DType;
            return result;
        }

        // Less-common dtypes (Q5_K, Q8_0, …): the CUDA matvec only dispatches on
        // Q4_K / Q6_K / F32, so dequantize on CPU and upload as F32. Same
        // fallback strategy the Vulkan port uses.
        int count = rows * cols;
        var f32 = new float[count];
        Dequantize.ToFloat32(expertData, f32, info.DType, count);
        var tensor = _gpu.Upload(f32, TensorShape.D1(count));
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

/// <summary>GPU tensors for one MoE expert on CUDA: gate, up, and down projection weights.</summary>
public readonly record struct ExpertCudaSlot(Tensor Gate, Tensor Up, Tensor Down);
