using System.Collections.Concurrent;
using System.Threading;
using SharpInference.Core;

namespace SharpInference.Cuda;

/// <summary>
/// CUDA/cuBLAS compute backend for DiT SGEMM acceleration.
/// Manages CUDA device memory and dispatches cuBLAS GemmEx kernels.
/// Precision is auto-detected at creation time:
///   sm_90+ (Hopper/H100) + CUDA 12 → fp8 E4M3 (cublasGemmEx fp8 requires sm_90)
///   sm_80+ (Ampere/RTX 30xx) → bf16 inputs, fp32 accumulation (no overflow, 2× smaller than fp32)
///   sm_53+ (Pascal/any CUDA GPU) → fp16 inputs, fp32 accumulation (avoids fp16 accum overflow)
///   fallback → fp32
/// All LLM transformer operations throw NotSupportedException; this backend is DiT-only.
/// </summary>
public sealed unsafe class CudaBackend : IComputeBackend, IImageOpsBackend, IDisposable
{
    /// <summary>
    /// Round <paramref name="byteSize"/> up to the bucket size the buffer pool will
    /// actually allocate (next power-of-two, min 64 bytes). Use this when sizing
    /// budgets that share VRAM with pooled allocations — the pool's round-up can
    /// inflate per-allocation footprint up to ~2× and a budget computed from raw
    /// byte sizes will overshoot real capacity. Bypasses the pool entirely when
    /// callers pass <c>exact: true</c> to <see cref="Upload"/> / <see cref="UploadRaw"/>.
    /// </summary>
    public static nuint RoundUpAllocBytes(nuint byteSize) => GpuBufferPool.RoundUp(byteSize);

    private readonly nint _handle;
    private readonly SgemmPrecision _precision;
    private readonly int _smVersion;
    private readonly nint _stream;
    private readonly ConcurrentDictionary<nint, (nint devPtr, nuint byteSize)> _devPtrs = new();
    // Issue #111: handles registered by View() — non-owning slices into another
    // tensor's allocation. Free() drops the registration without freeing memory.
    private readonly ConcurrentDictionary<nint, byte> _viewHandles = new();
    private long _nextHandle = 1;

    // Pinned host staging buffer for DMA-capable async H2D/D2H transfers.
    // Grows on demand; never shrinks (amortised over the pipeline lifetime).
    private nint   _pinnedBuf;
    private nuint  _pinnedBufSize;
    private const nuint InitialPinnedSize = 32 * 1024 * 1024; // 32 MB

    // Dedicated upload stream for background H2D transfers (UploadBackground*).
    // Separate from _stream so that prefetch DMA can overlap with the compute
    // stream without flushing it. Lazily created on first background upload —
    // costs nothing for backends that never prefetch experts.
    private nint   _uploadStream;
    // Per-upload pinned staging used by UploadBackground*. A second cudaMallocHost'd
    // buffer that the upload stream reads from. The async-upload lock serializes
    // concurrent calls so the staging contents stay valid until the in-flight DMA
    // completes (we wait on the previous event before re-filling the buffer).
    private nint   _asyncPinnedBuf;
    private nuint  _asyncPinnedBufSize;
    private readonly object _asyncUploadLock = new();
    // Event recorded at the end of the most recent UploadBackground; the next
    // call waits on it before re-using _asyncPinnedBuf to avoid overwriting an
    // in-flight DMA source. Created lazily, reused for the buffer's lifetime.
    private nint   _asyncPinnedBufEvent;

    // Maximum im2col tile buffer size. All row-aligned tile sizes fit within this bound.
    private const long MaxTileBytes = 2560L * 1024 * 1024; // 2.5 GiB — fits all layers in a single tile

    // GPU buffer pool: reuse device allocations by rounded size to avoid cudaMalloc overhead.
    // Each MatQ call (GEMM) does 2 alloc+free cycles; pooling eliminates driver round-trips.
    private readonly GpuBufferPool _pool = new();

    // Handles allocated with exact=true (no pool, no power-of-2 rounding). Tracked so
    // Free() can release them with cudaFree directly instead of returning to the pool
    // (where their unique sizes would create per-tensor buckets that never get reused).
    private readonly ConcurrentDictionary<nint, byte> _exactHandles = new();

    private bool _disposed;

    // ── NVRTC / image-ops state ────────────────────────────────────────────
    private readonly object _kernelInitLock = new();
    private bool   _imageKernelsInitialized;
    private bool   _imageKernelsAvailable;
    private nint   _nvModule;           // CUmodule loaded from compiled PTX
    private nint   _im2colKernel;

    // Persistent GPU buffer for im2col tiles — allocated once to MaxTileBytes (2.5 GiB).
    private nint   _im2colBuf;
    private nuint  _im2colBufSize;

    private nint   _biasAddKernel;
    private nint   _leakyReluKernel;
    private nint   _scaleKernel;
    private nint   _addKernel;
    private nint   _addScaledKernel;
    private nint   _clampKernel;
    private nint   _pshuffleKernel;
    private nint   _punshuffleKernel;
    private nint   _upsample2xKernel;

    // LLM transformer kernels (loaded by the same NVRTC compilation as the image kernels).
    private nint   _rmsNormKernel;
    private nint   _headNormKernel;
    private nint   _headNormPureKernel;
    private nint   _siluMulKernel;
    private nint   _sigmoidKernel;
    private nint   _softmaxKernel;
    private nint   _ropeInterleavedKernel;
    private nint   _ropeNeoxKernel;
    private nint   _ropeNeoxPartialKernel;
    private nint   _ropeNeoxWithFactorsKernel;
    private nint   _ropeNeoxWithFactorsBatchedKernel;
    private nint   _mulKernel;
    private nint   _sigmoidMulInPlaceKernel;
    private nint   _splitQgKernel;
    private nint   _kvAppendKernel;
    // Issue #27: bf16-store KV cache for hybrid GDN models. Halves cache VRAM
    // vs the fp32 ring; arithmetic still happens in fp32 (the bf16 → fp32
    // promotion is done at the kernel read sites).
    private nint   _kvAppendBf16Kernel;
    private nint   _attentionBf16Kernel;
    // SnapKV (issue #58): per-(query, head) attention scoring + position-gather
    // compaction. Used by CudaHybridGdnForwardPass.Prefill when SHARPI_SNAPKV_BUDGET
    // is set. Bf16 variants compose with the bf16-store KV path.
    private nint   _snapKvScoreKernel;
    private nint   _snapKvScoreBf16Kernel;
    private nint   _kvCompactKernel;
    private nint   _kvCompactBf16Kernel;
    private nint   _embedLookupF32Kernel;
    private nint   _embedLookupQ4KKernel;
    private nint   _embedLookupQ5KKernel;
    private nint   _embedLookupQ80Kernel;
    private nint   _matvecF32Kernel;
    private nint   _matvecQ4KKernel;
    private nint   _matvecQ5KKernel;
    private nint   _matvecQ6KKernel;
    // Q8_0 matvec (Phase 0 of the Gemma-4 plan): keeps Q8_0 weights packed on
    // the GPU. Without this, Q8_0 weights would dequant to F32 on upload and
    // blow out VRAM ~2.1×. Geometry mirrors Q5_K/Q6_K (8 rows/block × 32 thr/row).
    private nint   _matvecQ80Kernel;
    // Issue #43: N=2 (two-input, two-output) variants — read each weight row
    // once and accumulate into two outputs. Used by MTP BatchForward2's
    // on-GPU dense FFN to halve weight-bandwidth cost per output.
    private nint   _matvecF32N2Kernel;
    private nint   _matvecQ4KN2Kernel;
    private nint   _matvecQ5KN2Kernel;
    private nint   _matvecQ6KN2Kernel;
    // Issue #111: batched GEMM-N variants — one weight matrix, N input vectors,
    // N output rows in a single launch. Each (row, token) runs the identical
    // per-row reduction as the GEMV so results are bit-identical to N sequential
    // matvecs. Collapses the per-token trunk launches that dominate GDN-hybrid
    // prefill into one launch per projection.
    private nint   _matvecF32GemmNKernel;
    private nint   _matvecQ4KGemmNKernel;
    private nint   _matvecQ5KGemmNKernel;
    private nint   _matvecQ6KGemmNKernel;
    private nint   _matvecQ80GemmNKernel;
    // Issue #111: batched trunk elementwise/norm kernels (one launch over N tokens).
    private nint   _rmsNormBatchedKernel;
    private nint   _headNormBatchedKernel;
    private nint   _headNormQkKernel;
    private nint   _headNormQkBatchedKernel;
    private nint   _splitQgBatchedKernel;
    private nint   _ropeNeoxPartialBatchedKernel;
    private nint   _attentionKernel;
    // Gemma 4 (Phase 7): sliding-window attention, tanh-GELU FFN, final-logit
    // softcap. Kernel work only — forward-pass wiring lands in Phase 8.
    private nint   _attentionSwaKernel;
    private nint   _attentionSwaBatchedKernel;
    private nint   _geluTanhMulKernel;
    private nint   _geluTanhMulStridedKernel;
    private nint   _softcapKernel;
    private nint   _clearF32Kernel;
    private nint   _quantizeQ81Kernel;
    private nint   _bwBaselineKernel;
    // Issue #129: batched GPU-SLRU MoE host-loop replacements. scale_rows applies a
    // per-row scalar (shared-expert sigmoid gate) over [rows × cols] in one launch;
    // moe_weighted_reduce does the whole Phase-3 top-k weighted scatter-reduce +
    // shared add over all N tokens in one launch (replacing ~N·(na+2) tiny ops).
    private nint   _scaleRowsKernel;
    private nint   _moeWeightedReduceKernel;

    // TurboQuant KV-cache compression kernels.
    private nint   _tqRotateQueryKernel;
    private nint   _tqKvAppendKernel;
    private nint   _tqAttentionKernel;

    // qwen35moe Gated-DeltaNet (GDN) kernels.
    private nint   _siluInplaceKernel;
    private nint   _gdnConv1dDecodeKernel;
    private nint   _gdnL2NormPerHeadKernel;
    private nint   _gdnTileHeadsKernel;
    private nint   _gdnRecurrenceDecodeKernel;

    // Issue #114-B: batched GDN trunk + batched-query SDPA kernels.
    private nint   _gdnConv1dDecodeBatchedKernel;
    private nint   _gdnConv1dStateUpdateBatchedKernel;
    private nint   _gdnL2NormPerHeadBatchedKernel;
    private nint   _gdnTileHeadsBatchedKernel;
    private nint   _gdnRecurrenceScanKernel;
    private nint   _kvAppendBatchedKernel;
    private nint   _kvAppendBatchedBf16Kernel;
    private nint   _fullSeqAttentionKernel;
    private nint   _fullSeqAttentionBf16Kernel;
    private nint   _fullSeqAttentionGlobalKernel;
    private nint   _fullSeqAttentionGlobalBf16Kernel;

    // Grow-only global score scratch for the wave-based >4096 batched-query SDPA
    // (issue #118). Sized W × num_heads × score_stride floats; W is chosen so this
    // stays under a bounded budget. Freed in Dispose.
    private nint   _waveScratchBuf;
    private nuint  _waveScratchBufSize;

    // Persistent Q8_1 scratch for the Q4_K matvec input. Grows on demand and
    // never shrinks. Sized in 36-byte sub-blocks (one block_q8_1 per 32 elements).
    private nint   _q81Buf;
    private nuint  _q81BufSize;
    // Issue #43: second Q8_1 scratch for MatMulN2's second input vector.
    // Same grow-only sizing policy as _q81Buf. Only allocated when MatMulN2
    // dispatches the Q4_K path (sequential MatMul callers never touch it).
    private nint   _q81BufB;
    private nuint  _q81BufBSize;
    // Issue #111: Q8_1 scratch for MatMulBatched's N input vectors (laid out as
    // N contiguous q81 rows). Grow-only, same policy as _q81Buf.
    private nint   _q81BatchBuf;
    private nuint  _q81BatchBufSize;

    // Tracks dtype per tensor handle so MatMul can dispatch to the right matvec variant
    // (Q4_K / Q5_K / Q6_K / F32). Norm/bias weights upload as F32; quantized weight bytes
    // get tagged via UploadRaw.
    private readonly ConcurrentDictionary<nint, DType> _tensorDTypes = new();

    public string Name => $"CUDA GPU (cuBLAS, {_precision})";

    public SgemmPrecision BestSgemmPrecision => _precision;

    public bool SupportsGpuDequant => false;

    /// <summary>Total VRAM on the active CUDA device, in bytes. Queried once at backend creation.</summary>
    public ulong VramBytes
    {
        get
        {
            if (CuBlasInterop.MemGetInfo(out _, out nuint total) == 0)
                return total;
            return 0;
        }
    }

    /// <summary>Currently-free VRAM on the active CUDA device, in bytes. Queries the
    /// driver each call so callers can size allocations against live headroom.</summary>
    public ulong FreeVramBytes
    {
        get
        {
            if (CuBlasInterop.MemGetInfo(out nuint free, out _) == 0)
                return free;
            return 0;
        }
    }

    /// <summary>SM compute capability ×10 (e.g. 86 for sm_86 / RTX 30xx Ampere).</summary>
    public int SmVersion => _smVersion;

    /// <summary>
    /// CUDA stream that all kernels and memcpys are enqueued on. Callers that need to
    /// schedule their own async work against the backend pipeline can use this handle —
    /// it is owned by the backend and synchronized by <see cref="Synchronize"/>.
    /// </summary>
    public nint Stream => _stream;

    /// <summary>
    /// Returns the device pointer backing the given tensor. Intended for forward-pass
    /// implementations that need to perform raw cudaMemcpy operations between tensors.
    /// </summary>
    public nint GetTensorDevicePtr(Tensor tensor) => GetDevPtr(tensor);

    /// <summary>
    /// Async device-to-device copy of an entire tensor. Element-count of <paramref name="dst"/>
    /// and <paramref name="src"/> must match; both must be FP32 (the only element type the
    /// LLM forward path uses for scratch buffers).
    /// </summary>
    public void CopyDevice(Tensor dst, Tensor src)
    {
        if (dst.ElementCount != src.ElementCount)
            throw new ArgumentException($"CopyDevice: element-count mismatch ({src.ElementCount} → {dst.ElementCount}).");
        nuint bytes = (nuint)(src.ElementCount * sizeof(float));
        nint srcPtr = GetDevPtr(src);
        nint dstPtr = GetDevPtr(dst);
        int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, srcPtr, bytes, CuBlasInterop.DeviceToDevice, _stream);
        if (r != 0)
            throw new InvalidOperationException($"cudaMemcpyAsync (D2D) failed: {r}");
    }

    /// <summary>
    /// Async device-to-device copy of a sub-region. Offsets and size are measured in bytes.
    /// Used by the TurboQuant path to extract a single FP32 KV row out of the layer ring
    /// buffer for compression.
    /// </summary>
    public void CopyDeviceRegion(Tensor dst, long dstByteOffset,
                                 Tensor src, long srcByteOffset, long sizeBytes)
    {
        nint dstPtr = GetDevPtr(dst) + (nint)dstByteOffset;
        nint srcPtr = GetDevPtr(src) + (nint)srcByteOffset;
        int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, srcPtr, (nuint)sizeBytes,
            CuBlasInterop.DeviceToDevice, _stream);
        if (r != 0)
            throw new InvalidOperationException($"cudaMemcpyAsync (D2D region) failed: {r}");
    }

    private CudaBackend(nint handle, SgemmPrecision precision, int smVersion, nint stream,
                        nint pinnedBuf, nuint pinnedBufSize)
    {
        _handle        = handle;
        _precision     = precision;
        _smVersion     = smVersion;
        _stream        = stream;
        _pinnedBuf     = pinnedBuf;
        _pinnedBufSize = pinnedBufSize;
    }

    public static bool IsAvailable()
    {
        try
        {
            int status = CuBlasInterop.Create(out nint h);
            if (status == 0) { CuBlasInterop.Destroy(h); return true; }
            return false;
        }
        catch { return false; }
    }

    /// <summary>Create a CudaBackend, auto-detecting the best supported precision.</summary>
    public static CudaBackend Create() => Create(precision: null);

    /// <summary>
    /// Create a CudaBackend with an explicit precision override.
    /// Useful for benchmarking or testing a specific SGEMM path regardless of device capability.
    /// </summary>
    public static CudaBackend Create(SgemmPrecision forcedPrecision) => Create(precision: forcedPrecision);

    private static CudaBackend Create(SgemmPrecision? precision)
    {
        int status = CuBlasInterop.Create(out nint handle);
        if (status != 0)
            throw new InvalidOperationException($"cublasCreate failed: {status}");

        int smVersion = 0;
        if (CuBlasInterop.DeviceGetAttribute(out int major, CuBlasInterop.CudaDevAttrComputeCapabilityMajor, 0) == 0 &&
            CuBlasInterop.DeviceGetAttribute(out int minor, CuBlasInterop.CudaDevAttrComputeCapabilityMinor, 0) == 0)
            smVersion = major * 10 + minor;

        // Dedicated CUDA stream — all memcpy and GEMM are enqueued on this stream,
        // so cudaStreamSynchronize(stream) waits only for our work (not the whole device).
        if (CuBlasInterop.StreamCreate(out nint stream) != 0)
            stream = nint.Zero; // fall back to default stream

        if (stream != nint.Zero)
            CuBlasInterop.SetStream(handle, stream);

        // Enable TF32 tensor cores for Sgemm on Ampere/Ada (sm_80+).
        // TF32: on sm_80+ (Ampere+), enable TF32 tensor cores for cublasSgemm.
        // TF32 rounds mantissa to 10 bits but uses tensor cores — ~2× faster while
        // numerically close to FP32. No algorithm benchmarking overhead with SetMathMode.
        if (smVersion >= 80)
        {
            int mmr = CuBlasInterop.SetMathMode(handle, CuBlasInterop.CUBLAS_TF32_TENSOR_OP_MATH);
            if (mmr != 0)
                Console.Error.WriteLine($"[CudaBackend] cublasSetMathMode(TF32) returned {mmr} — using default math");
        }

        // Pinned (page-locked) staging buffer for DMA-capable async H2D/D2H transfers.
        CuBlasInterop.MallocHost(out nint pinnedBuf, InitialPinnedSize);

        var resolvedPrecision = precision ?? DetectBestPrecision(smVersion);
        var backend = new CudaBackend(handle, resolvedPrecision, smVersion, stream, pinnedBuf, InitialPinnedSize);

        // The 2.5 GiB im2col tile buffer is allocated lazily on the first Conv2d call.
        // Pre-allocating it here used to push LLM contexts that estimated max-context based
        // on free VRAM into driver-managed system-memory spillover (kernels then ran at
        // ~30 GB/s PCIe instead of ~500 GB/s HBM, cratering Q4_K matvec to ~20 GB/s).
        return backend;
    }

    private static SgemmPrecision DetectBestPrecision(int sm)
    {
        // SHARPI_CUDA_PRECISION = fp32 | fp16 | bf16 | fp8 — debug override for the
        // cuBLAS GEMM compute type. Bypasses the SM-based auto-detection below.
        // Unrecognised values fall through to auto-detect (no error).
        //
        // When to use: isolating whether an output regression is driven by
        // mantissa precision (use fp32 as the high-precision floor) vs algorithm
        // / kernel divergence. The default bf16 path on Ampere+ matches fp32 for
        // most workloads, but greedy decode is sensitive to single-ulp argmax
        // flips at low-margin steps. If output stays identical at fp32, the
        // regression is not precision-related.
        //
        // Override only affects the cuBLAS path; custom NVRTC kernels (Q4_K /
        // Q5_K matvec, RmsNorm, attention) keep their fp32 accumulators.
        var env = Environment.GetEnvironmentVariable("SHARPI_CUDA_PRECISION");
        if (env is not null)
        {
            switch (env.Trim().ToLowerInvariant())
            {
                case "fp32": return SgemmPrecision.Fp32;
                case "fp16": return SgemmPrecision.Fp16;
                case "bf16": return SgemmPrecision.Bf16;
                case "fp8":
                case "fp8e4m3": return SgemmPrecision.Fp8E4M3;
            }
        }
        // fp8 via cublasGemmEx requires sm_90+ (Hopper). Ada Lovelace (sm_89) only supports
        // fp8 through cublasLt (light), not the standard cublasGemmEx API.
        if (sm >= 90 && IsCuda12OrNewer())
            return SgemmPrecision.Fp8E4M3;
        if (sm >= 80) return SgemmPrecision.Bf16;    // Ampere+ has native bf16
        if (sm >= 53) return SgemmPrecision.Fp16;    // Pascal+ supports fp16 GemmEx
        return SgemmPrecision.Fp32;
    }

    /// <summary>
    /// Returns true when the loaded CUDA runtime is version 12 or newer.
    /// fp8 via cublasGemmEx requires CUDA 12+ (CUDA 11 returns NOT_SUPPORTED or hangs).
    /// Runtime version is encoded as major*1000 + minor*10 (e.g. 12010 = CUDA 12.1).
    /// </summary>
    private static bool IsCuda12OrNewer()
    {
        if (CuBlasInterop.RuntimeGetVersion(out int ver) != 0) return false;
        return ver >= 12000;
    }

    // ── Memory management ─────────────────────────────────────────────────

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32, bool exact = false)
    {
        nuint byteSize  = (nuint)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        // exact=true bypasses the power-of-2 pool rounding. Use for one-shot weight
        // uploads that won't be freed/realloc'd during decode — pooling is pure waste
        // for those, and the power-of-2 rounding can inflate per-allocation footprint
        // by up to 2× (a 17 MiB attn_gate rounds to 32 MiB; ~50 % average waste).
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        return new Tensor(shape, dtype, handle);
    }

    /// <summary>
    /// Registers a non-owning view into <paramref name="parent"/> starting at
    /// <paramref name="elemOffset"/> elements, spanning <paramref name="elemCount"/>
    /// elements (issue #111). The returned <see cref="Tensor"/> shares the parent's
    /// device memory; <see cref="Free"/> on it only drops the handle registration and
    /// never frees the underlying allocation. Used by batched prefill to pass a
    /// per-token slice of a <c>[N×dim]</c> batched buffer to the per-position
    /// recurrence / KV-append / attention kernels without an extra device copy.
    /// Element size is taken from the parent dtype (Float32 for activation buffers).
    /// </summary>
    public Tensor View(Tensor parent, long elemOffset, long elemCount, DType dtype = DType.Float32)
    {
        if (!_devPtrs.TryGetValue(parent.Handle, out var pe))
            throw new InvalidOperationException($"View: parent handle {parent.Handle} not registered.");
        if (elemOffset < 0 || elemCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elemOffset),
                $"View: elemOffset ({elemOffset}) and elemCount ({elemCount}) must be non-negative.");
        int elemBytes = DTypeInfo.BytesPerElement(dtype);
        long byteOffset = elemOffset * elemBytes;
        long byteCount = elemCount * elemBytes;
        if (byteOffset < 0 || byteOffset + byteCount > (long)pe.byteSize)
            throw new ArgumentOutOfRangeException(nameof(elemOffset),
                $"View [{elemOffset}, {elemOffset + elemCount}) (×{elemBytes}B) out of parent bounds ({pe.byteSize}B).");
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (pe.devPtr + (nint)byteOffset, (nuint)byteCount);
        _viewHandles[handle] = 0;
        return new Tensor(TensorShape.D1(elemCount), dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_viewHandles.TryRemove(tensor.Handle, out _))
        {
            // Non-owning view: drop the handle registration only; the parent owns
            // the device memory and frees it on its own Free().
            _tensorDTypes.TryRemove(tensor.Handle, out _);
            _devPtrs.TryRemove(tensor.Handle, out _);
            return;
        }
        _tensorDTypes.TryRemove(tensor.Handle, out _);
        if (_pinnedAllocs.Remove(tensor.Handle, out var pinned))
        {
            // Pinned allocations bypass the device pool — cudaMallocHost'd memory
            // must be released via cudaFreeHost rather than returned to the pool.
            _devPtrs.TryRemove(tensor.Handle, out _);
            CuBlasInterop.FreeHost(pinned.Ptr);
            return;
        }
        if (_exactHandles.TryRemove(tensor.Handle, out _))
        {
            // exact=true allocations bypass the pool; free directly so the memory
            // returns to the system / driver pool rather than getting stranded in
            // a per-tensor bucket the pool can't reuse.
            if (_devPtrs.TryRemove(tensor.Handle, out var ex))
                CuBlasInterop.CudaFree(ex.devPtr);
            return;
        }
        if (_devPtrs.TryRemove(tensor.Handle, out var entry))
            _pool.Return(entry.byteSize, entry.devPtr);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape, bool exact = false)
    {
        nuint byteSize  = (nuint)(data.Length * sizeof(float));
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (float* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        return new Tensor(shape, DType.Float32, handle);
    }

    public void Download(Tensor src, Span<float> dst)
    {
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)(dst.Length * sizeof(float));
        fixed (float* d = dst)
            DownloadViaStaging(d, devPtr, byteSize);
    }

    /// <summary>
    /// Copy host floats into an existing device tensor (no allocation).
    /// Mirrors <see cref="Upload"/> but reuses the destination's storage — used by
    /// the hybrid forward pass to push CPU-produced hidden state back into the
    /// GPU pipeline without churning tensor handles per token.
    /// </summary>
    public void UploadInto(Tensor dst, ReadOnlySpan<float> src)
    {
        if ((long)src.Length != dst.ElementCount)
            throw new ArgumentException($"UploadInto: element-count mismatch ({src.Length} → {dst.ElementCount}).");
        nint dstPtr = GetDevPtr(dst);
        nuint byteSize = (nuint)(src.Length * sizeof(float));
        fixed (float* s = src)
            UploadViaStaging(dstPtr, s, byteSize);
    }

    // ── Direct-pinned Download/Upload overloads (issues #48/#49) ──────────
    //
    // The Span<float> Download/UploadInto overloads above route every transfer
    // through the internal pinned staging buffer (_pinnedBuf) with an extra
    // host-side Buffer.MemoryCopy hop. When the caller already owns a pinned
    // host allocation (cudaMallocHost), that staging hop is wasted work.
    // These overloads cudaMemcpyAsync directly between the caller's pinned
    // buffer and the device, avoiding the round-trip via _pinnedBuf.
    //
    // The `pinnedDst`/`pinnedSrc` argument MUST point at memory allocated via
    // cudaMallocHost (see AllocatePinnedHost) — pageable memory will fall back
    // to a slow sync copy under cudaMemcpyAsync, defeating the optimisation.

    /// <summary>
    /// Download a float-typed device tensor into a caller-owned pinned host buffer
    /// via a single <c>cudaMemcpyAsync</c>, skipping the internal <c>_pinnedBuf</c>
    /// staging hop. Synchronous: blocks the host until the transfer completes.
    /// Caller is responsible for ensuring <paramref name="pinnedDst"/> points at
    /// memory allocated via <see cref="AllocatePinnedHost"/> (cudaMallocHost).
    /// (Issue #48.)
    /// </summary>
    public unsafe void Download(Tensor src, nint pinnedDst, int floatCount)
    {
        if ((long)floatCount > src.ElementCount)
            throw new ArgumentException($"Download: floatCount={floatCount} exceeds src.ElementCount={src.ElementCount}.");
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            int r = CuBlasInterop.CudaMemcpyAsync(pinnedDst, devPtr, byteSize,
                                                  CuBlasInterop.DeviceToHost, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (D2H, pinned) failed: {r}");
            CuBlasInterop.StreamSynchronize(_stream);
            // The sync above drains any prior in-flight async H2D on _pinnedBuf.
            _uploadInFlight = false;
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(pinnedDst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (D2H, pinned) failed: {r}");
        }
    }

    /// <summary>
    /// Like <see cref="Download(Tensor, nint, int)"/> but does NOT sync the stream.
    /// The transfer is queued on <c>_stream</c>; the buffer's contents are valid
    /// only after a subsequent stream sync (e.g. the next Download/Synchronize
    /// call). Used to overlap a hidden-state snapshot D2H with subsequent GPU work
    /// whose D2H sync will drain both. (Issue #49.)
    /// </summary>
    public unsafe void DownloadAsync(Tensor src, nint pinnedDst, int floatCount)
    {
        if ((long)floatCount > src.ElementCount)
            throw new ArgumentException($"DownloadAsync: floatCount={floatCount} exceeds src.ElementCount={src.ElementCount}.");
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            int r = CuBlasInterop.CudaMemcpyAsync(pinnedDst, devPtr, byteSize,
                                                  CuBlasInterop.DeviceToHost, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (D2H, pinned async) failed: {r}");
            // No sync — caller is responsible for draining via a subsequent
            // Download/Synchronize. _uploadInFlight is NOT cleared here.
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(pinnedDst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (D2H, pinned async fallback) failed: {r}");
        }
    }

    /// <summary>
    /// Upload from a caller-owned pinned host buffer to the device via a single
    /// <c>cudaMemcpyAsync</c> on <c>_stream</c>, skipping the internal
    /// <c>_pinnedBuf</c> staging hop. Subsequent enqueued GPU operations on
    /// <c>_stream</c> automatically sequence behind this transfer; no host-side
    /// sync is needed unless the caller mutates the pinned buffer before the next
    /// stream sync. Caller is responsible for ensuring <paramref name="pinnedSrc"/>
    /// points at memory allocated via <see cref="AllocatePinnedHost"/>
    /// (cudaMallocHost). (Issue #48.)
    /// </summary>
    public unsafe void UploadInto(Tensor dst, nint pinnedSrc, int floatCount)
    {
        if ((long)floatCount != dst.ElementCount)
            throw new ArgumentException($"UploadInto: element-count mismatch ({floatCount} → {dst.ElementCount}).");
        nint dstPtr = GetDevPtr(dst);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            // No need to drain _uploadInFlight: this transfer reads from the
            // caller's pinned buffer, not _pinnedBuf, so it can't race with a
            // prior async H2D that was using _pinnedBuf as its source.
            int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, pinnedSrc, byteSize,
                                                  CuBlasInterop.HostToDevice, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (H2D, pinned) failed: {r}");
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(dstPtr, pinnedSrc, byteSize, CuBlasInterop.HostToDevice);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (H2D, pinned) failed: {r}");
        }
    }

    /// <summary>
    /// Allocate a pinned host buffer via <c>cudaMallocHost</c>. Returns
    /// <see cref="IntPtr.Zero"/> on failure. Used by Engine code to allocate
    /// per-token scratch suitable for the direct-pinned
    /// <see cref="Download(Tensor, nint, int)"/> /
    /// <see cref="UploadInto(Tensor, nint, int)"/> overloads.
    /// </summary>
    public static nint AllocatePinnedHost(nuint byteSize)
    {
        if (CuBlasInterop.MallocHost(out nint ptr, byteSize) != 0)
            return nint.Zero;
        return ptr;
    }

    /// <summary>Free a pinned host buffer allocated via <see cref="AllocatePinnedHost"/>.</summary>
    public static void FreePinnedHost(nint ptr)
    {
        if (ptr != nint.Zero)
            CuBlasInterop.FreeHost(ptr);
    }

    // ── Vulkan-API shims (used by CudaHybridForwardPass) ──────────────────
    //
    // CUDA executes ops immediately on the stream and orders dependent work
    // implicitly, so Vulkan's command-buffer recording vocabulary degenerates
    // to either no-ops or a Synchronize. These shims let HybridForwardPass-shape
    // code call the same names on either backend.
    public void BeginRecord() { }
    public void EndRecordAndSubmit() => Synchronize();
    public void RecordBarrier() { }
    public void RecordComputeToHostBarrier() { }
    public void RecordComputeToTransferBarrier() { }
    public void RecordComputeCopy(Tensor dst, Tensor src) => CopyDevice(dst, src);
    public void RecordComputeCopyRegion(Tensor dst, long dstByteOffset,
                                        Tensor src, long srcByteOffset, long sizeBytes)
        => CopyDeviceRegion(dst, dstByteOffset, src, srcByteOffset, sizeBytes);

    // The Vulkan staging-buffer-on-fence pattern collapses to a stream-ordered
    // Download in CUDA. We defer the actual copy until ReadFromStaging so the
    // call sequence still reads Record→Submit→Read at the call site, even
    // though no separate staging buffer is needed under CUDA.
    private Tensor? _stagingPendingSrc;
    private int _stagingPendingCount;
    public void RecordDownloadToStaging(Tensor src, int floatCount)
    {
        _stagingPendingSrc = src;
        _stagingPendingCount = floatCount;
    }
    public void ReadFromStaging(Span<float> dst)
    {
        if (_stagingPendingSrc is not { } src)
            throw new InvalidOperationException(
                "ReadFromStaging called without a prior RecordDownloadToStaging.");
        if (dst.Length < _stagingPendingCount)
            throw new ArgumentException(
                $"ReadFromStaging dst too small: {dst.Length} < {_stagingPendingCount}.");
        Download(src, dst[.._stagingPendingCount]);
        _stagingPendingSrc = null;
    }

    // ── Pinned host memory exposed as device-accessible tensors ──
    // Vulkan exposes host-visible-device-local buffers as a single Tensor that
    // both the GPU shaders and CPU mapped pointer can use. Under UVA, CUDA's
    // cudaMallocHost gives the equivalent: one pointer addressable from both
    // host and device. We track the (host = device) pointer per handle so
    // MapPinned/Free can find it without a separate dictionary.
    private readonly Dictionary<nint, (nint Ptr, nuint Bytes)> _pinnedAllocs = new();

    public Tensor AllocatePinned(TensorShape shape, DType dtype = DType.Float32)
    {
        long elemBytes = DTypeInfo.BytesPerElement(dtype);
        nuint byteSize = (nuint)(shape.ElementCount * elemBytes);
        if (CuBlasInterop.MallocHost(out nint hostPtr, byteSize) != 0)
            throw new InvalidOperationException($"cudaMallocHost({byteSize}) failed.");
        // Zero the buffer so initial reads see deterministic values.
        new Span<byte>((void*)hostPtr, (int)byteSize).Clear();

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        // UVA: the host pointer is also a valid device pointer for kernel launches.
        _devPtrs[handle] = (hostPtr, byteSize);
        _pinnedAllocs[handle] = (hostPtr, byteSize);
        _tensorDTypes[handle] = dtype;
        return new Tensor(shape, dtype, handle);
    }

    public float* MapPinned(Tensor tensor)
    {
        if (!_pinnedAllocs.TryGetValue(tensor.Handle, out var alloc))
            throw new InvalidOperationException($"Tensor {tensor.Handle} was not allocated via AllocatePinned.");
        return (float*)alloc.Ptr;
    }

    public void UnmapPinned(Tensor tensor)
    {
        // No-op under CUDA: cudaMallocHost memory stays mapped for its lifetime.
        _ = tensor;
    }

    public Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)(data.Length * 2);
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (Half* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.Float16, handle);
    }

    public void DownloadHalf(Tensor src, Span<Half> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (Half* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)(dst.Length * 2));
    }

    public Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)(data.Length * 2);
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (ushort* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.BFloat16, handle);
    }

    public void DownloadBf16(Tensor src, Span<ushort> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (ushort* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)(dst.Length * 2));
    }

    public Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)data.Length;
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (byte* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.Float8E4M3, handle);
    }

    public void DownloadFp8(Tensor src, Span<byte> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (byte* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)dst.Length);
    }

    public Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype, bool exact = false)
    {
        nuint byteSize  = (nuint)data.Length;
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (byte* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        var tensor = new Tensor(shape, dtype, handle);
        _tensorDTypes[handle] = dtype;
        return tensor;
    }

    // ── Async background upload path (issue #78) ──────────────────────────
    //
    // Mirrors the Vulkan UploadBackground entry point: dispatches the host→device
    // copy on a dedicated upload stream so the predictive MoE prefetcher can
    // start moving the next layer's experts over PCIe while the compute stream
    // is still finishing the current layer's matmuls.
    //
    // The returned CudaUploadHandle owns a CUDA event recorded at the end of
    // the DMA. Consumers MUST call WaitForUpload on the compute stream (or any
    // other CUDA stream that reads the tensor) before launching dependent work,
    // otherwise the read can race ahead of the DMA. After WaitForUpload, the
    // tensor behaves identically to one returned from the synchronous Upload* —
    // it lives in the same _devPtrs table and is freed via Free() like any
    // other tensor.
    //
    // Concurrent UploadBackground* calls are serialized by _asyncUploadLock
    // because they share the _asyncPinnedBuf staging — the lock also ensures
    // the previous transfer's DMA drains before we overwrite the staging bytes.

    /// <summary>
    /// Async H2D upload on the dedicated upload stream. The returned tensor's
    /// readiness is tracked by <see cref="CudaUploadHandle.UploadEvent"/>;
    /// callers MUST invoke <see cref="WaitForUpload"/> on the compute stream
    /// (or whichever stream consumes the tensor) before launching dependent
    /// kernels. The event is owned by the handle and released by
    /// <see cref="ReleaseUploadHandle"/>.
    /// </summary>
    public CudaUploadHandle UploadBackground(ReadOnlySpan<float> data, TensorShape shape, bool exact = false)
    {
        nuint byteSize = (nuint)(data.Length * sizeof(float));
        fixed (float* src = data)
        {
            var (tensor, ev) = UploadBackgroundCore(src, byteSize, shape, DType.Float32, exact);
            return new CudaUploadHandle(tensor, ev);
        }
    }

    /// <summary>
    /// Async H2D upload of raw bytes with explicit dtype tagging on the
    /// dedicated upload stream. Same readiness model as
    /// <see cref="UploadBackground(ReadOnlySpan{float}, TensorShape, bool)"/>.
    /// </summary>
    public CudaUploadHandle UploadBackgroundRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype, bool exact = false)
    {
        nuint byteSize = (nuint)data.Length;
        fixed (byte* src = data)
        {
            var (tensor, ev) = UploadBackgroundCore(src, byteSize, shape, dtype, exact);
            _tensorDTypes[tensor.Handle] = dtype;
            return new CudaUploadHandle(tensor, ev);
        }
    }

    /// <summary>
    /// Make the compute stream wait for the background upload referenced by
    /// <paramref name="handle"/> to complete before launching any further work.
    /// Cheap if the DMA has already finished. Safe to call from any thread
    /// (cudaStreamWaitEvent is async — it inserts a fence into the stream and
    /// returns immediately, the actual wait happens on the device).
    /// </summary>
    public void WaitForUpload(CudaUploadHandle handle)
    {
        if (handle.UploadEvent == nint.Zero) return;
        EnsureUploadStream();
        if (_stream == nint.Zero)
        {
            // No compute stream → use device-wide sync (extremely rare path).
            int sr = CuBlasInterop.EventSynchronize(handle.UploadEvent);
            if (sr != 0)
                throw new InvalidOperationException($"cudaEventSynchronize failed: {sr}");
            return;
        }
        int r = CuBlasInterop.StreamWaitEvent(_stream, handle.UploadEvent, 0);
        if (r != 0)
            throw new InvalidOperationException($"cudaStreamWaitEvent failed: {r}");
    }

    /// <summary>
    /// Non-blocking poll: returns true if the background upload has completed
    /// (cudaEventQuery == cudaSuccess). Allows callers (e.g. SLRU GetOrLoad)
    /// to skip the WaitForUpload fence when the DMA has already drained.
    /// </summary>
    public bool IsUploadComplete(CudaUploadHandle handle)
    {
        if (handle.UploadEvent == nint.Zero) return true;
        int r = CuBlasInterop.EventQuery(handle.UploadEvent);
        return r == CuBlasInterop.CudaSuccess;
    }

    /// <summary>
    /// Destroy the CUDA event owned by <paramref name="handle"/>. Call once the
    /// tensor's readiness no longer needs to be tracked (typically after
    /// <see cref="WaitForUpload"/> for short-lived prefetches, or never if the
    /// tensor is long-lived — destroying the event is a courtesy that returns a
    /// few tens of bytes of driver state, not a correctness requirement).
    /// Idempotent.
    /// </summary>
    public void ReleaseUploadHandle(CudaUploadHandle handle)
    {
        if (handle.UploadEvent != nint.Zero)
            CuBlasInterop.EventDestroy(handle.UploadEvent);
    }

    private (Tensor tensor, nint ev) UploadBackgroundCore(void* src, nuint byteSize, TensorShape shape, DType dtype, bool exact)
    {
        EnsureUploadStream();

        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }

        nint ev;
        lock (_asyncUploadLock)
        {
            EnsureAsyncPinnedBuf(byteSize);

            // Drain the previous async upload before overwriting the shared staging
            // buffer — the in-flight DMA still reads from _asyncPinnedBuf and a
            // host-side memcpy here would corrupt it. The wait is on the host because
            // it gates a host memcpy, not another GPU launch.
            if (_asyncPinnedBufEvent != nint.Zero)
            {
                int sr = CuBlasInterop.EventSynchronize(_asyncPinnedBufEvent);
                if (sr != 0)
                    throw new InvalidOperationException($"cudaEventSynchronize (drain prev async upload) failed: {sr}");
            }

            Buffer.MemoryCopy(src, (void*)_asyncPinnedBuf, _asyncPinnedBufSize, byteSize);

            int rc = CuBlasInterop.CudaMemcpyAsync(devPtr, _asyncPinnedBuf, byteSize,
                CuBlasInterop.HostToDevice, _uploadStream);
            if (rc != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (UploadBackground) failed: {rc}");

            // Per-call event: timing disabled (we never measure these — the readiness
            // event is consumed by stream-wait or an EventQuery poll only).
            int er = CuBlasInterop.EventCreateWithFlags(out ev, CuBlasInterop.EventDisableTiming);
            if (er != 0)
                throw new InvalidOperationException($"cudaEventCreateWithFlags failed: {er}");

            int rr = CuBlasInterop.EventRecord(ev, _uploadStream);
            if (rr != 0)
            {
                CuBlasInterop.EventDestroy(ev);
                throw new InvalidOperationException($"cudaEventRecord failed: {rr}");
            }

            // Replace the buffer-reuse fence with the freshly recorded event. We
            // never destroy this reference — ev is owned by the caller's handle;
            // the *next* UploadBackgroundCore call only reads it via EventSynchronize.
            _asyncPinnedBufEvent = ev;
        }

        var handleId = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handleId] = (devPtr, allocSize);
        if (exact) _exactHandles[handleId] = 0;
        var tensor = new Tensor(shape, dtype, handleId);
        return (tensor, ev);
    }

    private void EnsureUploadStream()
    {
        if (_uploadStream != nint.Zero) return;
        lock (_asyncUploadLock)
        {
            if (_uploadStream != nint.Zero) return;
            EnsurePrimaryContextCurrent();
            int r = CuBlasInterop.StreamCreate(out nint s);
            if (r != 0)
                throw new InvalidOperationException($"cudaStreamCreate (upload stream) failed: {r}");
            _uploadStream = s;
        }
    }

    private void EnsureAsyncPinnedBuf(nuint required)
    {
        if (_asyncPinnedBuf != nint.Zero && required <= _asyncPinnedBufSize) return;
        // Drain any in-flight DMA reading the old buffer before freeing it.
        if (_asyncPinnedBufEvent != nint.Zero && _asyncPinnedBuf != nint.Zero)
            CuBlasInterop.EventSynchronize(_asyncPinnedBufEvent);
        if (_asyncPinnedBuf != nint.Zero)
            CuBlasInterop.FreeHost(_asyncPinnedBuf);

        nuint newSize = Math.Max(required, _asyncPinnedBufSize * 2);
        if (newSize < 1024 * 1024) newSize = 1024 * 1024;
        int r = CuBlasInterop.MallocHost(out _asyncPinnedBuf, newSize);
        if (r != 0)
            throw new InvalidOperationException($"cudaMallocHost (async upload staging, {newSize} B) failed: {r}");
        _asyncPinnedBufSize = newSize;
    }

    /// <summary>
    /// CUDA upload stream — exposed for tests and for callers that need to
    /// schedule additional copies on the same async transfer queue.
    /// <see cref="IntPtr.Zero"/> until the first <see cref="UploadBackground"/>
    /// (or <see cref="UploadBackgroundRaw"/>) call creates it.
    /// </summary>
    public nint UploadStream => _uploadStream;

    public void DequantQ5KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    public void DequantQ4KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    // ── SGEMM ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GEMM: C[M,N] = A[M,K] × B[N,K]^T using cublasGemmEx with fp32 accumulation.
    /// fp16 and bf16 inputs both accumulate in fp32 — prevents the overflow that plagued
    /// pure cublasHgemm (fp16 accum overflows for deep DiT layers with large residuals).
    /// Row-major layout is handled via the column-major transpose identity:
    ///   row-major C=A*B^T  ≡  col-major C^T = B*A^T
    /// </summary>
    public void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N)
    {
        nint aPtr = GetDevPtr(A);
        nint bPtr = GetDevPtr(B);
        nint cPtr = GetDevPtr(C);
        float alpha = 1.0f, beta = 0.0f;

        int cudaTypeA = ToCudaDataType(A.DType);
        int cudaTypeB = ToCudaDataType(B.DType);
        int cudaTypeC = ToCudaDataType(C.DType);

        if (cudaTypeA != CuBlasInterop.CUDA_R_32F || cudaTypeB != CuBlasInterop.CUDA_R_32F)
        {
            // fp16, bf16, or fp8: use GemmEx with fp32 accumulation.
            // fp8 E4M3 requires: both A and B fp8, and C must be bf16 or fp16 (not fp32).
            // fp16/bf16 use fp32 accumulation to avoid overflow on large DiT residuals.
            if (cudaTypeA == CuBlasInterop.CUDA_R_8F_E4M3 && cudaTypeC == CuBlasInterop.CUDA_R_32F)
                throw new InvalidOperationException(
                    "fp8 GemmEx: cuBLAS requires bf16/fp16 output (not fp32) when inputs are fp8. " +
                    "Allocate C as DType.BFloat16 and use DownloadBf16.");

            int status = CuBlasInterop.GemmEx(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                bPtr, cudaTypeB, K,
                aPtr, cudaTypeA, K,
                ref beta,
                cPtr, cudaTypeC, N,
                CuBlasInterop.CUBLAS_COMPUTE_32F,
                CuBlasInterop.CUBLAS_GEMM_DEFAULT);
            if (status != 0)
                throw new InvalidOperationException($"cublasGemmEx failed: {status}");
        }
        else if (_smVersion >= 80)
        {
            // Ampere+ with fp32 inputs: use TF32 tensor cores (~8× vs cublasSgemm).
            // TF32 has 10-bit mantissa (same as bf16) with fp32 range — no accuracy loss for DiT inference.
            int status = CuBlasInterop.GemmEx(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                bPtr, CuBlasInterop.CUDA_R_32F, K,
                aPtr, CuBlasInterop.CUDA_R_32F, K,
                ref beta,
                cPtr, CuBlasInterop.CUDA_R_32F, N,
                CuBlasInterop.CUBLAS_COMPUTE_32F_FAST_TF32,
                CuBlasInterop.CUBLAS_GEMM_DEFAULT);
            if (status != 0)
                throw new InvalidOperationException($"cublasGemmEx (TF32) failed: {status}");
        }
        else
        {
            int status = CuBlasInterop.Sgemm(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha, bPtr, K, aPtr, K,
                ref beta, cPtr, N);
            if (status != 0)
                throw new InvalidOperationException($"cublasSgemm failed: {status}");
        }
    }

    private static int ToCudaDataType(DType dtype) => dtype switch
    {
        DType.Float32    => CuBlasInterop.CUDA_R_32F,
        DType.Float16    => CuBlasInterop.CUDA_R_16F,
        DType.BFloat16   => CuBlasInterop.CUDA_R_16BF,
        DType.Float8E4M3 => CuBlasInterop.CUDA_R_8F_E4M3,
        _ => CuBlasInterop.CUDA_R_32F,
    };

    public void Synchronize()
    {
        int status = _stream != nint.Zero
            ? CuBlasInterop.StreamSynchronize(_stream)
            : CuBlasInterop.DeviceSync();
        if (status != 0)
            throw new InvalidOperationException($"CUDA synchronize failed: {status}");
        _uploadInFlight = false;
    }

    // ── Pinned staging buffer ─────────────────────────────────────────────

    /// <summary>
    /// Ensure the pinned staging buffer is at least <paramref name="required"/> bytes.
    /// The buffer grows but never shrinks (amortised cost over pipeline lifetime).
    /// </summary>
    private unsafe void EnsurePinnedBuf(nuint required)
    {
        if (required <= _pinnedBufSize) return;
        if (_pinnedBuf != nint.Zero) CuBlasInterop.FreeHost(_pinnedBuf);
        nuint newSize = Math.Max(required, _pinnedBufSize * 2);
        if (CuBlasInterop.MallocHost(out _pinnedBuf, newSize) != 0)
        {
            _pinnedBuf = nint.Zero; // allocation failed — fall back to sync copies
            _pinnedBufSize = 0;
            return;
        }
        _pinnedBufSize = newSize;
    }

    // Tracks whether the most recent UploadViaStaging call left an async H2D
    // memcpy in flight reading _pinnedBuf. Cleared on the next stream sync —
    // either explicit (Synchronize / DownloadViaStaging) or implicit (next
    // upload's drain). Any operation that overwrites _pinnedBuf must drain
    // first to avoid corrupting an in-flight DMA. See issue #47.
    private bool _uploadInFlight;

    /// <summary>
    /// Copy <paramref name="src"/> to the device via the pinned staging buffer.
    /// Issues a non-blocking <c>cudaMemcpyAsync</c> on <c>_stream</c> after the
    /// host-side memcpy into the pinned buffer (issue #47). The host returns
    /// immediately; the actual PCIe transfer overlaps with subsequent enqueued
    /// GPU work on the same stream. The next operation that reuses
    /// <c>_pinnedBuf</c> drains the prior async copy first, so the buffer is
    /// never read by the DMA while a new write is in progress.
    /// </summary>
    private unsafe void UploadViaStaging(nint devPtr, void* src, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            // Drain any prior in-flight upload before reusing _pinnedBuf.
            // Downloads call StreamSynchronize themselves; consecutive uploads
            // without an interleaved download are the only path that needs this.
            if (_uploadInFlight)
            {
                CuBlasInterop.StreamSynchronize(_stream);
                _uploadInFlight = false;
            }
            Buffer.MemoryCopy(src, (void*)_pinnedBuf, _pinnedBufSize, byteSize);
            int r = CuBlasInterop.CudaMemcpyAsync(devPtr, _pinnedBuf, byteSize,
                                                  CuBlasInterop.HostToDevice, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (H2D) failed: {r}");
            _uploadInFlight = true;
        }
        else
        {
            CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
        }
    }

    /// <summary>
    /// Copy from device to <paramref name="dst"/> via the pinned staging buffer (async DMA).
    /// The implicit <c>StreamSynchronize</c> here also drains any in-flight async upload,
    /// since both ops are queued on <c>_stream</c>.
    /// </summary>
    private unsafe void DownloadViaStaging(void* dst, nint devPtr, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            CuBlasInterop.CudaMemcpyAsync(_pinnedBuf, devPtr, byteSize,
                                          CuBlasInterop.DeviceToHost, _stream);
            CuBlasInterop.StreamSynchronize(_stream);
            _uploadInFlight = false;
            Buffer.MemoryCopy((void*)_pinnedBuf, dst, byteSize, byteSize);
        }
        else
        {
            CuBlasInterop.CudaMemcpy((nint)dst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
        }
    }

    // ── LLM transformer ops ───────────────────────────────────────────────

    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMul(output, matrix, vector, dtype);
    }

    /// <summary>
    /// Matrix-vector multiply with explicit weight dtype dispatch.
    ///   • Q4_K → llama.cpp-style int8 / __dp4a path: quantize the input vector
    ///     to Q8_1 once, then dispatch a 1-row-per-block × 4-warp cooperative
    ///     matvec that uses __dp4a for the inner dot products.
    ///   • Q5_K / Q6_K / Q8_0 / F32 → custom fp32 matvec kernels (8 rows / block).
    /// Output is FP32 in every case.
    /// </summary>
    public void MatMul(Tensor output, Tensor matrix, Tensor vector, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");

        int rows = (int)output.ElementCount;
        int cols = (int)vector.ElementCount;
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(vector);
        nint yPtr = GetDevPtr(output);

        if (weightDType == DType.Q4_K)
        {
            DispatchMatVecQ4K(wPtr, xPtr, yPtr, rows, cols);
            return;
        }

        int  pRows = rows, pCols = cols;
        nint* args = stackalloc nint[5]
        {
            (nint)(&wPtr), (nint)(&xPtr), (nint)(&yPtr),
            (nint)(&pRows), (nint)(&pCols)
        };

        nint kernel = weightDType switch
        {
            DType.Q5_K    => _matvecQ5KKernel,
            DType.Q6_K    => _matvecQ6KKernel,
            DType.Q8_0    => _matvecQ80Kernel,
            DType.Float32 => _matvecF32Kernel,
            _ => throw new NotSupportedException($"CUDA MatMul: weight dtype {weightDType} not supported (expected Q4_K, Q5_K, Q6_K, Q8_0, or Float32)."),
        };

        uint grid = (uint)((rows + 7) / 8);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec) failed: {r}");
    }

    /// <summary>
    /// Two-input matrix-vector multiply: produces <paramref name="outputA"/> and
    /// <paramref name="outputB"/> from a single <paramref name="matrix"/> read,
    /// dispatched to a per-dtype custom kernel (<c>llm_matvec_*_n2</c>). The
    /// weight matrix is read once per row, then folded into two independent
    /// dot products — halves the per-output weight bandwidth versus two
    /// sequential <see cref="MatMul"/> calls, mirroring the CPU
    /// <c>SimdKernels.MatVec2In</c> design. Used by MTP batched-verify
    /// (issue #43, follow-up to #30) to amortise on-GPU dense FFN weight
    /// reads across the two draft tokens.
    /// </summary>
    public void MatMulN2(Tensor outputA, Tensor outputB,
                        Tensor matrix,
                        Tensor inputA, Tensor inputB,
                        DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");

        if (outputA.ElementCount != outputB.ElementCount)
            throw new ArgumentException(
                $"MatMulN2: outputA.ElementCount ({outputA.ElementCount}) != outputB.ElementCount ({outputB.ElementCount}).");
        if (inputA.ElementCount != inputB.ElementCount)
            throw new ArgumentException(
                $"MatMulN2: inputA.ElementCount ({inputA.ElementCount}) != inputB.ElementCount ({inputB.ElementCount}).");

        int rows = (int)outputA.ElementCount;
        int cols = (int)inputA.ElementCount;
        nint wPtr  = GetDevPtr(matrix);
        nint xAPtr = GetDevPtr(inputA);
        nint xBPtr = GetDevPtr(inputB);
        nint yAPtr = GetDevPtr(outputA);
        nint yBPtr = GetDevPtr(outputB);

        if (weightDType == DType.Q4_K)
        {
            DispatchMatVecQ4KN2(wPtr, xAPtr, xBPtr, yAPtr, yBPtr, rows, cols);
            return;
        }

        int  pRows = rows, pCols = cols;
        nint* args = stackalloc nint[7]
        {
            (nint)(&wPtr),
            (nint)(&xAPtr), (nint)(&xBPtr),
            (nint)(&yAPtr), (nint)(&yBPtr),
            (nint)(&pRows), (nint)(&pCols)
        };

        nint kernel = weightDType switch
        {
            DType.Q5_K    => _matvecQ5KN2Kernel,
            DType.Q6_K    => _matvecQ6KN2Kernel,
            DType.Float32 => _matvecF32N2Kernel,
            _ => throw new NotSupportedException(
                $"CUDA MatMulN2: weight dtype {weightDType} not supported (expected Q4_K, Q5_K, Q6_K, or Float32)."),
        };

        uint grid = (uint)((rows + 7) / 8);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_n2) failed: {r}");
    }

    /// <summary>
    /// Convenience overload: looks up the weight dtype from the
    /// <see cref="UploadRaw"/> tag dictionary, matching the single-input
    /// <see cref="MatMul(Tensor, Tensor, Tensor)"/> behaviour.
    /// </summary>
    public void MatMulN2(Tensor outputA, Tensor outputB,
                        Tensor matrix,
                        Tensor inputA, Tensor inputB)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMulN2(outputA, outputB, matrix, inputA, inputB, dtype);
    }

    /// <summary>
    /// Batched matrix-vector multiply (GEMM-N, issue #111): one <paramref name="matrix"/>
    /// applied to <paramref name="nTok"/> input vectors, producing <paramref name="nTok"/>
    /// output rows — in a single kernel launch. Layout is token-major:
    /// <paramref name="inputAll"/> is <c>[nTok × cols]</c> and <paramref name="outputAll"/>
    /// is <c>[nTok × rows]</c> (token <c>t</c>'s slice starts at <c>t × rows</c>).
    ///
    /// <para><b>Bit-exact:</b> each (row, token) pair runs the identical per-row reduction
    /// as the single-token <see cref="MatMul(Tensor,Tensor,Tensor,DType)"/> — same weight
    /// decode, same dp4a/FMA chain, same warp + shared reduce — so the result is
    /// bit-identical to <paramref name="nTok"/> sequential <see cref="MatMul"/> calls. This
    /// is the property the GDN/MTP byte-parity oracles depend on; do not reorder the
    /// reduction. Only the launch count collapses (N → 1), killing the host launch
    /// overhead that dominates GDN-hybrid prefill.</para>
    ///
    /// Supports Q4_K (via per-token Q8_1 quantize + the GEMM-N dp4a kernel) and Float32.
    /// Other dtypes throw — the GDN-hybrid trunk projections are all Q4_K.
    /// </summary>
    public void MatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll,
                              int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatched: outputAll ({outputAll.ElementCount}) and inputAll ({inputAll.ElementCount}) " +
                $"element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        if (weightDType == DType.Q4_K)
        {
            DispatchMatVecQ4KBatched(wPtr, xPtr, yPtr, rows, cols, nTok);
            return;
        }
        if (weightDType is DType.Float32 or DType.Q6_K or DType.Q5_K or DType.Q8_0)
        {
            // All take F32 input; the Q5_K/Q6_K/Q8_0 kernels decode the weight per
            // element. Same (rows+7)/8 × nTok geometry across all four.
            nint kernel = weightDType switch
            {
                DType.Q6_K => _matvecQ6KGemmNKernel,
                DType.Q5_K => _matvecQ5KGemmNKernel,
                DType.Q8_0 => _matvecQ80GemmNKernel,
                _          => _matvecF32GemmNKernel,
            };
            int pRows = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&xPtr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            uint gridX = (uint)((rows + 7) / 8);
            int r = NvrtcInterop.LaunchKernel(kernel, gridX, (uint)nTok, 1,
                                              256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_gemm_n) failed: {r}");
            return;
        }
        throw new NotSupportedException(
            $"CUDA MatMulBatched: weight dtype {weightDType} not supported (expected Q4_K, Q5_K, Q6_K, Q8_0, or Float32).");
    }

    /// <summary>
    /// Convenience overload: looks up the weight dtype from the
    /// <see cref="UploadRaw"/> tag dictionary.
    /// </summary>
    public void MatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll, int nTok)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMulBatched(outputAll, matrix, inputAll, nTok, dtype);
    }

    /// <summary>
    /// Q4_K batched GEMM-N: quantizes all <paramref name="nTok"/> input vectors into a
    /// single contiguous Q8_1 scratch (one launch over <c>nTok × subBlocks</c> blocks —
    /// the per-token quantize is independent so this is bit-identical to per-token
    /// quantization), then dispatches <c>llm_matvec_q4k_gemm_n</c> over a (rows, nTok)
    /// grid.
    /// </summary>
    private void DispatchMatVecQ4KBatched(nint wPtr, nint xPtr, nint yPtr,
                                          int rows, int cols, int nTok)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_gemm_n requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;                          // per token
        long totalSub = (long)subBlocks * nTok;
        nuint q81Bytes = (nuint)(totalSub * 36L);
        EnsureQ81BatchBuf(q81Bytes);

        // Quantize all nTok inputs → contiguous Q8_1. The single-token kernel's
        // index math (elem_idx = block_id*32 + lane, out += block_id*36) already
        // covers a contiguous [nTok × cols] batch: block_id runs [0, nTok*subBlocks).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81BatchBuf;
            int  qN      = (int)((long)cols * nTok);
            if ((long)cols * nTok > int.MaxValue)
                throw new InvalidOperationException(
                    $"MatMulBatched: cols*nTok ({(long)cols * nTok}) exceeds int range.");
            nint* args = stackalloc nint[3]
            {
                (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)totalSub, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 batched) failed: {rq}");
        }

        {
            nint q81Ptr = _q81BatchBuf;
            int  pRows  = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            // grid = (rows, nTok); block = 32 × MATVEC_Q4K_NWARPS(8) = 256 threads.
            int rm = NvrtcInterop.LaunchKernel(
                _matvecQ4KGemmNKernel, (uint)rows, (uint)nTok, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_gemm_n) failed: {rm}");
        }
    }

    private void EnsureQ81BatchBuf(nuint required)
    {
        if (_q81BatchBuf != nint.Zero && _q81BatchBufSize >= required) return;
        if (_q81BatchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchBuf);
            _q81BatchBuf = nint.Zero;
            _q81BatchBufSize = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81BatchBuf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 batch scratch, {newSize} B) failed: {r}");
        _q81BatchBufSize = newSize;
    }

    /// <summary>
    /// Q4_K N=2 matvec: quantizes both input vectors into independent Q8_1
    /// scratches (<c>_q81Buf</c> for A, <c>_q81BufB</c> for B), then dispatches
    /// the cooperative <c>llm_matvec_q4k_n2</c> kernel that reads each weight
    /// super-block once and accumulates into two outputs per row.
    /// </summary>
    private void DispatchMatVecQ4KN2(nint wPtr, nint xAPtr, nint xBPtr,
                                     nint yAPtr, nint yBPtr,
                                     int rows, int cols)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_n2 requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);
        EnsureQ81BufB(q81Bytes);

        // Quantize both inputs into their own Q8_1 scratch (32 threads per sub-block).
        // Two unrolled launches — keeps stackalloc out of any loop (CA2014).
        nint qInA  = xAPtr;
        nint qOutA = _q81Buf;
        nint qInB  = xBPtr;
        nint qOutB = _q81BufB;
        int  qN    = cols;
        nint* qArgsA = stackalloc nint[3] { (nint)(&qInA), (nint)(&qOutA), (nint)(&qN) };
        int rqA = NvrtcInterop.LaunchKernel(
            _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
            32, 1, 1, 0, _stream, qArgsA, null);
        if (rqA != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 A) failed: {rqA}");
        nint* qArgsB = stackalloc nint[3] { (nint)(&qInB), (nint)(&qOutB), (nint)(&qN) };
        int rqB = NvrtcInterop.LaunchKernel(
            _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
            32, 1, 1, 0, _stream, qArgsB, null);
        if (rqB != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 B) failed: {rqB}");

        {
            nint q81PtrA = _q81Buf;
            nint q81PtrB = _q81BufB;
            int  pRows   = rows, pCols = cols;
            nint* args = stackalloc nint[7]
            {
                (nint)(&wPtr),
                (nint)(&q81PtrA), (nint)(&q81PtrB),
                (nint)(&yAPtr),   (nint)(&yBPtr),
                (nint)(&pRows),   (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                _matvecQ4KN2Kernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_n2) failed: {rm}");
        }
    }

    /// <summary>
    /// Q4_K matvec via the llama.cpp Q8_1 + __dp4a path. Quantizes <paramref name="xPtr"/>
    /// (cols floats) into the persistent Q8_1 scratch in 32-element sub-blocks, then
    /// dispatches the cooperative matvec (1 row per block, 4 warps cooperating).
    /// </summary>
    private void DispatchMatVecQ4K(nint wPtr, nint xPtr, nint yPtr, int rows, int cols)
    {
        // cols must be a multiple of 256 (one Q4_K super-block = 8 Q8_1 sub-blocks).
        // Every transformer hidden dim in practice satisfies this; assert defensively.
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);

        // ── Quantize input → Q8_1 (one CUDA block of 32 threads per sub-block).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81Buf;
            int  qN      = cols;
            nint* args = stackalloc nint[3]
            {
                (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1) failed: {rq}");
        }

        // ── Cooperative matvec: 1 row/block, MATVEC_Q4K_NWARPS warps × 32 threads.
        // 8 warps (256 threads) per block delivers enough in-flight instructions
        // to hide global-memory latency on Ada at the modest weight footprint of
        // a single LLM row (a few KB of Q4_K).
        {
            nint q81Ptr = _q81Buf;
            int  pRows  = rows, pCols = cols;
            nint* args = stackalloc nint[5]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                _matvecQ4KKernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k) failed: {rm}");
        }
    }

    private void EnsureQ81Buf(nuint required)
    {
        if (_q81Buf != nint.Zero && _q81BufSize >= required) return;
        if (_q81Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81Buf);
            _q81Buf = nint.Zero;
            _q81BufSize = 0;
        }
        // Grow generously — the largest activation in a typical LLM is intermediate_dim
        // (Qwen3 = 12288 → ~14 KB Q8_1). Round up to the next 64 KB to amortise growth.
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81Buf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 scratch, {newSize} B) failed: {r}");
        _q81BufSize = newSize;
    }

    private void EnsureQ81BufB(nuint required)
    {
        if (_q81BufB != nint.Zero && _q81BufBSize >= required) return;
        if (_q81BufB != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BufB);
            _q81BufB = nint.Zero;
            _q81BufBSize = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81BufB, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 scratch B, {newSize} B) failed: {r}");
        _q81BufBSize = newSize;
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n  = (int)dst.ElementCount;
        nint p0 = GetDevPtr(dst);
        nint p1 = GetDevPtr(src);
        int  p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_addKernel, n, args);
    }

    /// <summary>Element-wise multiply: output[i] = a[i] * b[i]. Tensors must be 1-D with matching element counts.</summary>
    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long n = output.ElementCount;
        if (a.ElementCount != n || b.ElementCount != n)
            throw new ArgumentException(
                $"ElementwiseMul element-count mismatch: output={n} a={a.ElementCount} b={b.ElementCount}.");

        nint oPtr = GetDevPtr(output);
        nint aPtr = GetDevPtr(a);
        nint bPtr = GetDevPtr(b);
        int  pN = (int)n;
        nint* args = stackalloc nint[4]
        {
            (nint)(&oPtr), (nint)(&aPtr), (nint)(&bPtr), (nint)(&pN)
        };
        uint grid = (uint)(((int)n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_mulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(mul) failed: {r}");
    }

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        nint wPtr = GetDevPtr(weight);
        nint yPtr = GetDevPtr(output);
        int   pN = n;
        float pE = eps;
        nint* args = stackalloc nint[5]
        {
            (nint)(&xPtr), (nint)(&wPtr), (nint)(&yPtr),
            (nint)(&pN),   (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_rmsNormKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rmsnorm) failed: {r}");
    }

    /// <summary>
    /// Batched RmsNorm over <paramref name="nTok"/> rows (issue #111). <paramref name="x"/>
    /// and <paramref name="output"/> are <c>[nTok × dim]</c> contiguous; <paramref name="weight"/>
    /// is shared across rows. One block per token runs the identical reduction as
    /// <see cref="RmsNorm"/>, so output is bit-identical to nTok sequential calls.
    /// </summary>
    public void RmsNormBatched(Tensor output, Tensor x, Tensor weight, int nTok, int dim, float eps = 1e-5f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xPtr = GetDevPtr(x);
        nint wPtr = GetDevPtr(weight);
        nint yPtr = GetDevPtr(output);
        int   pN = dim, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr), (nint)(&wPtr), (nint)(&yPtr),
            (nint)(&pN), (nint)(&pE), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_rmsNormBatchedKernel, (uint)nTok, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rmsnorm_batched) failed: {r}");
    }

    /// <summary>Per-head RMS norm with learned weights (Qwen3 / OLMoE QK norm).
    /// <paramref name="perChannelWeight"/> false → weight is shared <c>[headDim]</c> vector
    /// applied identically to every head (Qwen3); true → weight is
    /// <c>[numHeads * headDim]</c> with one slice per head (OLMoE).</summary>
    public void HeadNorm(Tensor data, Tensor weight, int numHeads, int headDim,
        float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        nint wPtr = GetDevPtr(weight);
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        int  pWS = perChannelWeight ? headDim : 0;
        nint* args = stackalloc nint[6]
        {
            (nint)(&dPtr), (nint)(&wPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pWS)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm) failed: {r}");
    }

    /// <summary>
    /// Batched per-head RmsNorm over <paramref name="nTok"/> rows (issue #111).
    /// <paramref name="data"/> is <c>[nTok × numHeads × headDim]</c>; grid = (numHeads, nTok).
    /// Bit-identical to nTok sequential <see cref="HeadNorm"/> calls.
    /// </summary>
    public void HeadNormBatched(Tensor data, Tensor weight, int numHeads, int headDim,
        int nTok, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        nint wPtr = GetDevPtr(weight);
        int  pHD = headDim, pNH = numHeads, pWS = perChannelWeight ? headDim : 0, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[7]
        {
            (nint)(&dPtr), (nint)(&wPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pWS), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_batched) failed: {r}");
    }

    /// <summary>
    /// Dual per-head RmsNorm of Q and K in one launch (Gemma 4 QK-norm). Grid covers
    /// <c>numHeads + numKvHeads</c> blocks; the first <c>numHeads</c> normalize Q with
    /// <paramref name="qWeight"/>, the rest K with <paramref name="kWeight"/>. Per block
    /// bit-identical to <see cref="HeadNorm"/>.
    /// </summary>
    public void HeadNormQk(Tensor qData, Tensor qWeight, Tensor kData, Tensor kWeight,
        int numHeads, int numKvHeads, int headDim, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(qData), qwP = GetDevPtr(qWeight), kP = GetDevPtr(kData), kwP = GetDevPtr(kWeight);
        int pHD = headDim, pNH = numHeads, pNKV = numKvHeads, pWS = perChannelWeight ? headDim : 0;
        float pE = eps;
        nint* args = stackalloc nint[9]
        {
            (nint)(&qP), (nint)(&qwP), (nint)(&kP), (nint)(&kwP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pNKV), (nint)(&pE), (nint)(&pWS)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormQkKernel, (uint)(numHeads + numKvHeads), 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_qk) failed: {r}");
    }

    /// <summary>Batched <see cref="HeadNormQk"/> over <paramref name="nTok"/> tokens.</summary>
    public void HeadNormQkBatched(Tensor qData, Tensor qWeight, Tensor kData, Tensor kWeight,
        int numHeads, int numKvHeads, int headDim, int nTok, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(qData), qwP = GetDevPtr(qWeight), kP = GetDevPtr(kData), kwP = GetDevPtr(kWeight);
        int pHD = headDim, pNH = numHeads, pNKV = numKvHeads, pWS = perChannelWeight ? headDim : 0, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&qwP), (nint)(&kP), (nint)(&kwP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pNKV), (nint)(&pE), (nint)(&pWS), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormQkBatchedKernel, (uint)(numHeads + numKvHeads), (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_qk_batched) failed: {r}");
    }

    /// <summary>Per-head L2 normalize (no learned weights). Llama-4 style.</summary>
    public void HeadNormPure(Tensor data, int numHeads, int headDim, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        nint* args = stackalloc nint[4]
        {
            (nint)(&dPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormPureKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_pure) failed: {r}");
    }

    public void Softmax(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xPtr), (nint)(&pN) };
        int r = NvrtcInterop.LaunchKernel(_softmaxKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(softmax) failed: {r}");
    }

    public void Sigmoid(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_sigmoidKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(sigmoid) failed: {r}");
    }

    /// <summary>Fused SiLU(gate) * up in-place into gate.</summary>
    public void SiLuMul(Tensor gate, Tensor up)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)gate.ElementCount;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pN = n;
        nint* args = stackalloc nint[3] { (nint)(&gPtr), (nint)(&uPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_siluMulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(silu_mul) failed: {r}");
    }

    /// <summary>Fused tanh-approximate GELU(gate) * up in-place into gate.
    /// Gemma-style FFN activation. Element-count of <paramref name="gate"/>
    /// must equal element-count of <paramref name="up"/>.</summary>
    public void GeluTanhMul(Tensor gate, Tensor up)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)gate.ElementCount;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pN = n;
        nint* args = stackalloc nint[3] { (nint)(&gPtr), (nint)(&uPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_geluTanhMulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gelu_tanh_mul) failed: {r}");
    }

    /// <summary>
    /// Strided-up <see cref="GeluTanhMul"/> over <paramref name="nTok"/> tokens:
    /// <paramref name="gate"/> is <c>[nTok × width]</c> contiguous; the up operand for
    /// token t is at <c>up + t*upStride + upOffset</c>. Lets batched PLE inject the
    /// per-layer slice of a <c>[nTok × (L*pleWidth)]</c> projection buffer without a
    /// gather. Per element bit-identical to <see cref="GeluTanhMul"/>.
    /// </summary>
    public void GeluTanhMulStrided(Tensor gate, Tensor up, int width, long upStride, long upOffset, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long total = (long)nTok * width;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pW = width, pNT = nTok;
        long pStride = upStride, pOff = upOffset;
        nint* args = stackalloc nint[6]
        {
            (nint)(&gPtr), (nint)(&uPtr), (nint)(&pW), (nint)(&pStride), (nint)(&pOff), (nint)(&pNT)
        };
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_geluTanhMulStridedKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gelu_tanh_mul_strided) failed: {r}");
    }

    /// <summary>
    /// In-place final-logit softcap: <c>x[i] = tanh(x[i] / cap) * cap</c>.
    /// Used by Gemma 4 to clip extreme logits before sampling.
    /// </summary>
    public void SoftcapInPlace(Tensor x, float cap)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        float pCap = cap;
        nint* args = stackalloc nint[3] { (nint)(&xPtr), (nint)(&pN), (nint)(&pCap) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_softcapKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(softcap_inplace) failed: {r}");
    }

    public void SiLU(Tensor x) =>
        throw new NotSupportedException("Use SiLuMul(gate, up) for fused SwiGLU on CUDA.");

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f, bool neox = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (headDim / 2);

        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[5]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        nint kernel = neox ? _ropeNeoxKernel : _ropeInterleavedKernel;
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope) failed: {r}");
    }

    /// <summary>
    /// Partial NEOX RoPE: rotate the first <paramref name="ropeDim"/> dimensions of every
    /// head; leave dims <c>[ropeDim, headDim)</c> untouched. Used by qwen35moe attention
    /// (ropeDim=64, headDim=256). Mirrors the CPU reference
    /// <see cref="Cpu.SimdKernels.ApplyRoPECachedNeoxPartial"/>.
    /// </summary>
    public void RoPEPartial(Tensor x, int position, int headDim, int ropeDim, float ropeTheta, bool neox)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (!neox)
            throw new ArgumentException("RoPEPartial currently supports only neox=true.", nameof(neox));
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim.", nameof(ropeDim));

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (ropeDim / 2);

        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pRD = ropeDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pRD), (nint)(&pPos), (nint)(&pT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxPartialKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_partial) failed: {r}");
    }

    /// <summary>
    /// Batched partial NEOX RoPE over <paramref name="nTok"/> rows (issue #111). Token t
    /// rotates at position <paramref name="basePosition"/> + t (prefill assigns contiguous
    /// positions). <paramref name="x"/> is <c>[nTok × numHeads × headDim]</c>. Bit-identical
    /// to nTok sequential <see cref="RoPEPartial"/> calls at the matching positions.
    /// </summary>
    public void RoPEPartialBatched(Tensor x, int basePosition, int headDim, int ropeDim,
        float ropeTheta, int numHeads, int nTok, bool neox)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (!neox)
            throw new ArgumentException("RoPEPartialBatched currently supports only neox=true.", nameof(neox));
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim.", nameof(ropeDim));

        int totalPairs = numHeads * (ropeDim / 2);
        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pRD = ropeDim, pPos = basePosition, pNT = nTok;
        float pT = ropeTheta;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pRD), (nint)(&pPos), (nint)(&pT), (nint)(&pNT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxPartialBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_partial_batched) failed: {r}");
    }

    /// <summary>
    /// NEOX RoPE with per-half-dim frequency factors (Gemma 4 global layers /
    /// Gemma-3n). <paramref name="freqFactors"/> is a <c>head_dim/2</c> F32 device
    /// vector that divides each pair's frequency; mirrors llama.cpp
    /// <c>gemma4.cpp:191</c> passing <c>rope_freqs.weight</c> only for non-SWA
    /// layers and the CPU <see cref="Cpu.SimdKernels.BuildRopeTable"/>
    /// globalFreqFactors path.
    /// </summary>
    public void RoPEWithFactors(Tensor x, int position, int headDim, float ropeTheta, Tensor freqFactors)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentException("headDim must be a positive even number.", nameof(headDim));
        if (freqFactors.ElementCount != headDim / 2)
            throw new ArgumentException(
                $"RoPEWithFactors: freqFactors length {freqFactors.ElementCount} != headDim/2 ({headDim / 2}).",
                nameof(freqFactors));

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (headDim / 2);

        nint xPtr = GetDevPtr(x);
        nint fPtr = GetDevPtr(freqFactors);
        int  pNH = numHeads, pHD = headDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT),
            (nint)(&fPtr)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxWithFactorsKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_with_factors) failed: {r}");
    }

    /// <summary>
    /// Batched <see cref="RoPEWithFactors"/> over <paramref name="nTok"/> tokens (Gemma 4
    /// global layers in batched-trunk prefill). Token <c>t</c> uses position
    /// <c>basePosition + t</c>; <paramref name="x"/> is <c>[nTok × numHeads*headDim]</c>.
    /// Bit-identical per row to the per-token kernel.
    /// </summary>
    public void RoPEWithFactorsBatched(Tensor x, int basePosition, int headDim,
        float ropeTheta, Tensor freqFactors, int numHeads, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentException("headDim must be a positive even number.", nameof(headDim));
        if (freqFactors.ElementCount != headDim / 2)
            throw new ArgumentException(
                $"RoPEWithFactorsBatched: freqFactors length {freqFactors.ElementCount} != headDim/2 ({headDim / 2}).",
                nameof(freqFactors));

        int totalPairs = numHeads * (headDim / 2);
        nint xPtr = GetDevPtr(x);
        nint fPtr = GetDevPtr(freqFactors);
        int  pNH = numHeads, pHD = headDim, pPos = basePosition, pNT = nTok;
        float pT = ropeTheta;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT),
            (nint)(&fPtr), (nint)(&pNT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxWithFactorsBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_with_factors_batched) failed: {r}");
    }

    /// <summary>
    /// Fused <c>x[i] *= sigmoid(gate[i])</c> in-place. Replaces a Sigmoid + ElementwiseMul
    /// pair for the qwen35moe GLU attention gate.
    /// </summary>
    public void SigmoidMulInPlace(Tensor x, Tensor gate)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (x.ElementCount != gate.ElementCount)
            throw new ArgumentException(
                $"SigmoidMulInPlace element-count mismatch: x={x.ElementCount} gate={gate.ElementCount}.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        nint gPtr = GetDevPtr(gate);
        int  pN = n;
        nint* args = stackalloc nint[3]
        {
            (nint)(&xPtr), (nint)(&gPtr), (nint)(&pN)
        };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_sigmoidMulInPlaceKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(sigmoid_mul_inplace) failed: {r}");
    }

    /// <summary>
    /// Strided de-interleave of qwen35moe's GLU-gated Q output. Input
    /// <paramref name="qg"/> is laid out per head as <c>[Q[headDim] ‖ G[headDim]]</c>
    /// (output stride = <c>2*headDim</c>); this splits into contiguous
    /// <c>q[numHeads*headDim]</c> and <c>g[numHeads*headDim]</c>.
    /// </summary>
    public void SplitQG(Tensor q, Tensor g, Tensor qg, int numHeads, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long expected = (long)numHeads * headDim * 2;
        if (qg.ElementCount != expected)
            throw new ArgumentException(
                $"SplitQG: qg.ElementCount {qg.ElementCount} != numHeads*headDim*2 ({expected}).");
        long perOut = (long)numHeads * headDim;
        if (q.ElementCount != perOut || g.ElementCount != perOut)
            throw new ArgumentException(
                $"SplitQG: q/g element counts must each equal numHeads*headDim ({perOut}); got q={q.ElementCount}, g={g.ElementCount}.");

        nint qgPtr = GetDevPtr(qg);
        nint qPtr  = GetDevPtr(q);
        nint gPtr  = GetDevPtr(g);
        int  pNH = numHeads, pHD = headDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&qgPtr), (nint)(&qPtr), (nint)(&gPtr),
            (nint)(&pNH), (nint)(&pHD)
        };
        int total = numHeads * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_splitQgKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(split_qg) failed: {r}");
    }

    /// <summary>
    /// Batched <see cref="SplitQG"/> over <paramref name="nTok"/> rows (issue #111).
    /// <paramref name="qg"/> is <c>[nTok × numHeads × headDim × 2]</c>; <paramref name="q"/>
    /// and <paramref name="g"/> are <c>[nTok × numHeads × headDim]</c>. Bit-identical to
    /// nTok sequential <see cref="SplitQG"/> calls.
    /// </summary>
    public void SplitQGBatched(Tensor q, Tensor g, Tensor qg, int numHeads, int headDim, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qgPtr = GetDevPtr(qg);
        nint qPtr  = GetDevPtr(q);
        nint gPtr  = GetDevPtr(g);
        int  pNH = numHeads, pHD = headDim, pNT = nTok;
        nint* args = stackalloc nint[6]
        {
            (nint)(&qgPtr), (nint)(&qPtr), (nint)(&gPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pNT)
        };
        int total = numHeads * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_splitQgBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(split_qg_batched) failed: {r}");
    }

    /// <summary>Write fresh K and V vectors for one token into the layer KV cache at <paramref name="position"/>.</summary>
    public void KvAppend(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
                         int kvDim, int position, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kInput);
        nint vPtr = GetDevPtr(vInput);
        nint kcP  = GetDevPtr(kCache);
        nint vcP  = GetDevPtr(vCache);
        int  pKD = kvDim, pPos = position, pMSL = maxSeqLen;
        nint* args = stackalloc nint[7]
        {
            (nint)(&kPtr), (nint)(&vPtr),
            (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pPos), (nint)(&pMSL)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append) failed: {r}");
    }

    /// <summary>
    /// Scaled dot-product attention with GQA support. Output: [numHeads * headDim].
    ///
    /// When <c>seqLen ≤ 4096</c> the kernel keeps per-position scores in shared memory and
    /// <paramref name="scoresScratch"/> is ignored. Above that threshold the kernel spills
    /// scores to <paramref name="scoresScratch"/>, which must have room for
    /// <c>numHeads × maxSeqLen</c> floats. Passing a non-null scratch always works.
    /// </summary>
    public void Attention(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                          Tensor? scoresScratch,
                          int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                          float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSL), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention) failed: {r}");
    }

    /// <summary>
    /// Sliding-window attention (Gemma 4 SWA layers). Iterates positions over
    /// <c>[max(0, position+1-windowSize), position+1)</c> instead of the full
    /// prefix. Per-layer <paramref name="headDim"/> is passed explicitly so a
    /// model with varying head_dim across layers (Gemma 4: 256 SWA / 512 global)
    /// dispatches correctly.
    ///
    /// When the effective windowed range ≤ 4096 the kernel keeps per-position
    /// scores in shared memory and <paramref name="scoresScratch"/> is ignored.
    /// Above that threshold the kernel spills to <paramref name="scoresScratch"/>
    /// which must have room for <c>numHeads × maxSeqLen</c> floats.
    /// </summary>
    public void AttentionSwa(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                             Tensor? scoresScratch,
                             int position, int windowSize, int headDim,
                             int numHeads, int numKvHeads, int maxSeqLen,
                             float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int windowEnd   = position + 1;
        int windowStart = windowEnd - windowSize;
        if (windowStart < 0) windowStart = 0;

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int  pWS = windowStart, pWE = windowEnd, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pWS), (nint)(&pWE), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa) failed: {r}");
    }

    /// <summary>
    /// Batched <see cref="AttentionSwa"/> over <paramref name="nTok"/> query tokens
    /// (Gemma 4 SWA layers in batched-trunk prefill). Query token <c>i</c> sits at
    /// absolute position <c>startPos+i</c> and attends its sliding window. The window
    /// bounds eff_seq ≤ <paramref name="windowSize"/>, so the shared-scores path always
    /// suffices (windowSize ≤ 4096 required). Bit-identical per (head, token) to the
    /// per-token kernel — no global scores scratch needed.
    /// </summary>
    public void AttentionSwaBatched(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (windowSize <= 0 || windowSize > 4096)
            throw new ArgumentException(
                $"AttentionSwaBatched requires 0 < windowSize ≤ 4096 (shared-scores path); got {windowSize}.",
                nameof(windowSize));

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_batched) failed: {r}");
    }

    /// <summary>
    /// Bf16-store variant of <see cref="KvAppend"/>. Inputs stay fp32; the K/V
    /// cache tensors must be <see cref="DType.BFloat16"/>-allocated (half the
    /// element count of an fp32 cache). See issue #27.
    /// </summary>
    public void KvAppendBf16(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
                             int kvDim, int position, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kInput);
        nint vPtr = GetDevPtr(vInput);
        nint kcP  = GetDevPtr(kCache);
        nint vcP  = GetDevPtr(vCache);
        int  pKD = kvDim, pPos = position, pMSL = maxSeqLen;
        nint* args = stackalloc nint[7]
        {
            (nint)(&kPtr), (nint)(&vPtr),
            (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pPos), (nint)(&pMSL)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBf16Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_bf16) failed: {r}");
    }

    /// <summary>
    /// Bf16-read variant of <see cref="Attention"/>. K/V cache tensors must be
    /// <see cref="DType.BFloat16"/>; query, output, and the score scratch stay
    /// fp32. Arithmetic precision matches the fp32 kernel — only the cache
    /// footprint changes.
    /// </summary>
    public void AttentionBf16(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                              Tensor? scoresScratch,
                              int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pMSL = maxSeqLen;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSL), (nint)(&pMSL)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionBf16Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_bf16) failed: {r}");
    }

    // ── Issue #114-B: batched prompt-prefill SDPA ──────────────────────────

    /// <summary>
    /// Batched <see cref="KvAppend"/>: writes the K/V vectors for <paramref name="nTok"/>
    /// tokens into the cache at consecutive positions <c>startPos .. startPos+nTok-1</c>
    /// in a single launch. <paramref name="kAll"/>/<paramref name="vAll"/> are
    /// <c>[nTok × kvDim]</c> token-major. Bit-identical to nTok sequential
    /// <see cref="KvAppend"/> calls.
    /// </summary>
    public void KvAppendBatched(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache,
                                int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kAll), vPtr = GetDevPtr(vAll);
        nint kcP = GetDevPtr(kCache), vcP = GetDevPtr(vCache);
        int pKD = kvDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kPtr), (nint)(&vPtr), (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_batched) failed: {r}");
    }

    /// <summary>Bf16-store variant of <see cref="KvAppendBatched"/> (default KV dtype).</summary>
    public void KvAppendBatchedBf16(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache,
                                    int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kAll), vPtr = GetDevPtr(vAll);
        nint kcP = GetDevPtr(kCache), vcP = GetDevPtr(vCache);
        int pKD = kvDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kPtr), (nint)(&vPtr), (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBatchedBf16Kernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_batched_bf16) failed: {r}");
    }

    /// <summary>
    /// Batched-query scaled dot-product attention (issue #114-B). All
    /// <paramref name="nTok"/> prompt queries attend over their causal prefix in a
    /// single launch (grid = numHeads × nTok), instead of nTok sequential
    /// <see cref="Attention"/> launches. Query i (row of <paramref name="qAll"/>,
    /// <c>[nTok × numHeads·headDim]</c>) attends over cache positions
    /// <c>[0, startPos+i+1)</c>; output written to <paramref name="outAll"/> in the
    /// same layout. Bit-identical to the per-token <see cref="Attention"/> path.
    ///
    /// <para><b>Constraint:</b> uses the shared-scores fast path only, so the caller
    /// MUST guarantee <c>startPos + nTok ≤ 4096</c> (every block's seqLen stays
    /// ≤ MAX_STORED_SCORES). Beyond that, fall back to the per-token loop.</para>
    /// </summary>
    public void AttentionBatched(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                 int numHeads, int numKvHeads, int headDim,
                                 int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (startPos + nTok > 4096)
            throw new ArgumentException(
                $"AttentionBatched requires startPos+nTok ≤ 4096 (shared-scores path); got {startPos}+{nTok}.");

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention) failed: {r}");
    }

    /// <summary>Bf16-read variant of <see cref="AttentionBatched"/> (default KV dtype).</summary>
    public void AttentionBatchedBf16(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                     int numHeads, int numKvHeads, int headDim,
                                     int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (startPos + nTok > 4096)
            throw new ArgumentException(
                $"AttentionBatchedBf16 requires startPos+nTok ≤ 4096 (shared-scores path); got {startPos}+{nTok}.");

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionBf16Kernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_bf16) failed: {r}");
    }

    // ── Issue #118: wave-based >4096 batched-query SDPA ─────────────────────

    /// <summary>
    /// Bounded scratch budget (in floats) for the wave-based SDPA. The wave width is
    /// chosen so <c>W × numHeads × scoreStride</c> floats fit this budget. Override
    /// via <c>SHARPI_ATTN_WAVE_BUDGET_MB</c> (default 256 MiB).
    /// </summary>
    private static long WaveScratchBudgetFloats()
    {
        long mb = 256;
        var ov = Environment.GetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB");
        if (ov is not null && long.TryParse(ov, out long m) && m > 0) mb = m;
        return mb * 1024 * 1024 / sizeof(float);
    }

    private void EnsureWaveScratch(long floats)
    {
        nuint required = (nuint)floats * sizeof(float);
        if (_waveScratchBuf != nint.Zero && _waveScratchBufSize >= required) return;
        if (_waveScratchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_waveScratchBuf);
            _waveScratchBuf = nint.Zero;
            _waveScratchBufSize = 0;
        }
        int r = CuBlasInterop.CudaMalloc(out _waveScratchBuf, required);
        if (r != 0)
            throw new InvalidOperationException($"cudaMalloc(wave attn scratch, {required} bytes) failed: {r}");
        _waveScratchBufSize = required;
    }

    /// <summary>
    /// Wave-based batched-query SDPA for prompt prefill past the 4096-position
    /// shared-scores window (issue #118). Splits the <paramref name="nTok"/> queries
    /// into waves of <c>W</c> (chosen so <c>W × numHeads × (startPos+nTok)</c> floats
    /// of global score scratch fit <see cref="WaveScratchBudgetFloats"/>); each wave is
    /// one launch over <c>(numHeads, W)</c> blocks, each block cloning
    /// <c>llm_attention</c>'s global-scratch path. Bit-identical to the per-token
    /// <see cref="Attention"/> path (same per-(head,query) dot / tree softmax /
    /// V-weighted sum). Query/output layout matches <see cref="AttentionBatched"/>.
    /// </summary>
    public void AttentionBatchedWave(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                     int numHeads, int numKvHeads, int headDim,
                                     int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int scoreStride = startPos + nTok;            // max seq_len any query reaches
        long perQuery = (long)numHeads * scoreStride; // score floats per query row
        long budget = WaveScratchBudgetFloats();
        int W = (int)Math.Max(1, Math.Min(nTok, budget / Math.Max(1, perQuery)));
        EnsureWaveScratch((long)W * perQuery);

        int qDim = numHeads * headDim;
        nint qBase = GetDevPtr(qAll), oBase = GetDevPtr(outAll);
        nint kP = GetDevPtr(kCache), vP = GetDevPtr(vCache);
        nint scP = _waveScratchBuf;
        // Per-launch mutables (qP/oP/spEff/pN change each wave); the rest are constant.
        nint qP = qBase, oP = oBase;
        int spEff = startPos, pN = 0;
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pStride = scoreStride;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP), (nint)(&scP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&spEff), (nint)(&pMSL), (nint)(&pN), (nint)(&pStride)
        };
        for (int waveStart = 0; waveStart < nTok; waveStart += W)
        {
            int wThis = Math.Min(W, nTok - waveStart);
            qP = qBase + (nint)((long)waveStart * qDim * sizeof(float));
            oP = oBase + (nint)((long)waveStart * qDim * sizeof(float));
            spEff = startPos + waveStart;
            pN = wThis;
            int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionGlobalKernel,
                (uint)numHeads, (uint)wThis, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_global) failed: {r}");
        }
    }

    /// <summary>Bf16-read variant of <see cref="AttentionBatchedWave"/> (default KV dtype).</summary>
    public void AttentionBatchedWaveBf16(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                         int numHeads, int numKvHeads, int headDim,
                                         int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int scoreStride = startPos + nTok;
        long perQuery = (long)numHeads * scoreStride;
        long budget = WaveScratchBudgetFloats();
        int W = (int)Math.Max(1, Math.Min(nTok, budget / Math.Max(1, perQuery)));
        EnsureWaveScratch((long)W * perQuery);

        int qDim = numHeads * headDim;
        nint qBase = GetDevPtr(qAll), oBase = GetDevPtr(outAll);
        nint kP = GetDevPtr(kCache), vP = GetDevPtr(vCache);
        nint scP = _waveScratchBuf;
        nint qP = qBase, oP = oBase;
        int spEff = startPos, pN = 0;
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pStride = scoreStride;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP), (nint)(&scP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&spEff), (nint)(&pMSL), (nint)(&pN), (nint)(&pStride)
        };
        for (int waveStart = 0; waveStart < nTok; waveStart += W)
        {
            int wThis = Math.Min(W, nTok - waveStart);
            qP = qBase + (nint)((long)waveStart * qDim * sizeof(float));
            oP = oBase + (nint)((long)waveStart * qDim * sizeof(float));
            spEff = startPos + waveStart;
            pN = wThis;
            int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionGlobalBf16Kernel,
                (uint)numHeads, (uint)wThis, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_global_bf16) failed: {r}");
        }
    }

    // ================================================================
    //  SnapKV (issue #58): prefill KV eviction support
    // ================================================================

    /// <summary>
    /// Score one captured query <paramref name="q"/> (size <c>numHeads * headDim</c>)
    /// against the layer's K cache (positions <c>[0, promptLen)</c>), softmax across
    /// the causally-valid prefix, and atomicAdd the per-position weights into
    /// <paramref name="scoreAccum"/>. Caller is responsible for zeroing the accumulator
    /// before the first call and looping over the W captured queries × attention
    /// layers; the kernel pools naturally because every call's softmaxed weights are
    /// summed into the same buffer.
    /// </summary>
    /// <param name="qAbsPos">Absolute prompt position of the query (causal mask cutoff).</param>
    public void SnapKvScore(Tensor q, Tensor kCache, Tensor scoreAccum, Tensor scoresScratch,
                            int numHeads, int numKvHeads, int headDim,
                            int promptLen, int qAbsPos, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(q);
        nint kP  = GetDevPtr(kCache);
        nint sP  = GetDevPtr(scoreAccum);
        nint ssP = GetDevPtr(scoresScratch);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pPL = promptLen, pQAP = qAbsPos, pMSL = maxSeqLen;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&sP), (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pPL), (nint)(&pQAP), (nint)(&pMSL)
        };
        int r = NvrtcInterop.LaunchKernel(_snapKvScoreKernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(snapkv_score) failed: {r}");
    }

    /// <summary>
    /// Bf16-K-cache variant of <see cref="SnapKvScore"/>. <paramref name="kCache"/>
    /// must be a <see cref="DType.BFloat16"/> tensor (raw unsigned short storage);
    /// the scoring math stays in fp32.
    /// </summary>
    public void SnapKvScoreBf16(Tensor q, Tensor kCache, Tensor scoreAccum, Tensor scoresScratch,
                                int numHeads, int numKvHeads, int headDim,
                                int promptLen, int qAbsPos, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(q);
        nint kP  = GetDevPtr(kCache);
        nint sP  = GetDevPtr(scoreAccum);
        nint ssP = GetDevPtr(scoresScratch);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pPL = promptLen, pQAP = qAbsPos, pMSL = maxSeqLen;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&sP), (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pPL), (nint)(&pQAP), (nint)(&pMSL)
        };
        int r = NvrtcInterop.LaunchKernel(_snapKvScoreBf16Kernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(snapkv_score_bf16) failed: {r}");
    }

    /// <summary>
    /// Gather kept positions of one KV ring (K or V) into a dense
    /// <c>[K * kvDim]</c> prefix of <paramref name="dst"/>. <paramref name="src"/>
    /// and <paramref name="dst"/> MUST be different tensors (the destination is
    /// later copied back over the original ring's <c>[0, K * kvDim)</c> region by
    /// the caller). <paramref name="keepPositions"/> must be sorted ascending and
    /// hold indices in <c>[0, originalLength)</c>.
    /// </summary>
    public void KvCompact(Tensor src, Tensor dst, Tensor keepPositions,
                          int K, int kvDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP  = GetDevPtr(src);
        nint dP  = GetDevPtr(dst);
        nint kpP = GetDevPtr(keepPositions);
        int  pK = K, pKD = kvDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&kpP),
            (nint)(&pK), (nint)(&pKD)
        };
        uint gridX = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvCompactKernel,
            gridX, (uint)K, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_compact) failed: {r}");
    }

    /// <summary>
    /// Bf16-store variant of <see cref="KvCompact"/>. Source and destination must
    /// both be <see cref="DType.BFloat16"/> tensors; the gather copies raw unsigned
    /// short elements with no fp32 round-trip.
    /// </summary>
    public void KvCompactBf16(Tensor src, Tensor dst, Tensor keepPositions,
                              int K, int kvDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP  = GetDevPtr(src);
        nint dP  = GetDevPtr(dst);
        nint kpP = GetDevPtr(keepPositions);
        int  pK = K, pKD = kvDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&kpP),
            (nint)(&pK), (nint)(&pKD)
        };
        uint gridX = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvCompactBf16Kernel,
            gridX, (uint)K, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_compact_bf16) failed: {r}");
    }

    // ================================================================
    //  TurboQuant KV-cache compression
    // ================================================================

    /// <summary>
    /// Rotate query vectors for TurboQuant attention: applies the Walsh-Hadamard
    /// transform and the per-(layer × kv_head) sign flip. Call once per layer
    /// (before <see cref="TqAttention"/>), reuse the result across all cached positions.
    /// </summary>
    public void TqRotateQuery(Tensor qInput, Tensor rotatedQ, Tensor signPatterns,
                              int numHeads, int numKvHeads, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(qInput);
        nint rqP = GetDevPtr(rotatedQ);
        nint sP  = GetDevPtr(signPatterns);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        nint* args = stackalloc nint[6]
        {
            (nint)(&qP), (nint)(&rqP), (nint)(&sP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD)
        };
        int r = NvrtcInterop.LaunchKernel(_tqRotateQueryKernel,
            (uint)numHeads, 1, 1, (uint)headDim, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_rotate_query) failed: {r}");
    }

    /// <summary>
    /// Compress one (K, V) token-position into the layer's TurboQuant cache.
    /// Writes a packed 3-bit block per KV head at byte offset
    /// <c>position * numKvHeads * blockBytes + kvHead * blockBytes</c>.
    /// </summary>
    public void TqKvAppend(Tensor kInput, Tensor vInput, Tensor kCacheTq, Tensor vCacheTq,
                           Tensor signPatterns, Tensor codebook, Tensor boundaries,
                           int kvDim, int headDim, int position, int maxSeqLen,
                           int numKvHeads, int blockBytes)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kP = GetDevPtr(kInput);
        nint vP = GetDevPtr(vInput);
        nint kc = GetDevPtr(kCacheTq);
        nint vc = GetDevPtr(vCacheTq);
        nint sP = GetDevPtr(signPatterns);
        nint cP = GetDevPtr(codebook);
        nint bP = GetDevPtr(boundaries);
        int  pKD = kvDim, pHD = headDim, pPos = position, pMSL = maxSeqLen,
             pNKV = numKvHeads, pBB = blockBytes;
        nint* args = stackalloc nint[13]
        {
            (nint)(&kP), (nint)(&vP), (nint)(&kc), (nint)(&vc),
            (nint)(&sP), (nint)(&cP), (nint)(&bP),
            (nint)(&pKD), (nint)(&pHD), (nint)(&pPos),
            (nint)(&pMSL), (nint)(&pNKV), (nint)(&pBB)
        };
        int r = NvrtcInterop.LaunchKernel(_tqKvAppendKernel,
            (uint)numKvHeads, 1, 1, (uint)headDim, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_kv_append) failed: {r}");
    }

    /// <summary>
    /// Hybrid TurboQuant attention: scaled dot-product attention over the
    /// TQ-compressed history plus the FP32 recent-window cache.
    ///
    /// When <c>tqSeqLen + fp32SeqLen ≤ 4096</c> the kernel keeps per-position scores
    /// in shared memory and <paramref name="scoresScratch"/> is ignored. Above that
    /// threshold the kernel spills the scores to <paramref name="scoresScratch"/>,
    /// which must have room for <c>numHeads × maxSeqLen</c> floats (one slot per
    /// (head, position) pair). Passing a non-null scratch always works regardless
    /// of context size — callers can allocate once for the worst case.
    /// </summary>
    public void TqAttention(Tensor q, Tensor rotatedQ, Tensor kCacheTq, Tensor vCacheTq,
                            Tensor kCacheFp32, Tensor vCacheFp32, Tensor output, Tensor codebook,
                            Tensor? scoresScratch,
                            int numHeads, int numKvHeads, int headDim,
                            int tqSeqLen, int fp32SeqLen, int maxSeqLen, int blockBytes)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP   = GetDevPtr(q);
        nint rqP  = GetDevPtr(rotatedQ);
        nint kctP = GetDevPtr(kCacheTq);
        nint vctP = GetDevPtr(vCacheTq);
        nint kfP  = GetDevPtr(kCacheFp32);
        nint vfP  = GetDevPtr(vCacheFp32);
        nint oP   = GetDevPtr(output);
        nint cbP  = GetDevPtr(codebook);
        nint ssP  = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pTQ = tqSeqLen, pFP = fp32SeqLen, pMSL = maxSeqLen, pBB = blockBytes;
        nint* args = stackalloc nint[16]
        {
            (nint)(&qP), (nint)(&rqP),
            (nint)(&kctP), (nint)(&vctP),
            (nint)(&kfP),  (nint)(&vfP),
            (nint)(&oP),   (nint)(&cbP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pTQ), (nint)(&pFP),  (nint)(&pMSL), (nint)(&pBB)
        };
        int r = NvrtcInterop.LaunchKernel(_tqAttentionKernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_attention) failed: {r}");
    }

    /// <summary>Look up one row from an F32 embedding table into <paramref name="output"/>.</summary>
    public void EmbedLookup(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        uint grid = (uint)((embDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_embedLookupF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_f32) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q4_K-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256.
    /// </summary>
    public void EmbedLookupQ4K(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ4K requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ4KKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q4k) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q5_K-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256 (Q5_K block size).
    /// </summary>
    public void EmbedLookupQ5K(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ5K requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ5KKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q5k) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q8_0-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256 (8 Q8_0 blocks per outer iteration).
    /// Phase 0 of the Gemma-4 plan: keeps the (10752, 262144) Q8_0 token-embd table off
    /// the dequant-to-F32 upload path that would otherwise blow out VRAM.
    /// </summary>
    public void EmbedLookupQ8_0(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ8_0 requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ80Kernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q8_0) failed: {r}");
    }

    /// <summary>Set every element of <paramref name="dst"/> to zero.</summary>
    /// <remarks>
    /// The kernel writes fp32-sized lanes; for sub-fp32 dtypes (e.g. BFloat16)
    /// the byte count is converted to a 4-byte lane count so we don't overrun
    /// the underlying buffer. For non-fp32 element sizes that aren't a multiple
    /// of 4 bytes, callers should instead use a memset path (none exposed yet).
    /// </remarks>
    public void Clear(Tensor dst)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long byteCount = dst.ElementCount * DTypeInfo.BytesPerElement(dst.DType);
        if ((byteCount & 3) != 0)
            throw new InvalidOperationException(
                $"Clear: dst byte count ({byteCount}) is not a multiple of 4; " +
                "dtype-aware memset path is not implemented yet.");
        int n = (int)(byteCount >> 2);
        nint dP = GetDevPtr(dst);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&dP), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_clearF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(clear) failed: {r}");
    }

    /// <summary>Zero a sub-region of a tensor starting at <paramref name="elementOffset"/>
    /// for <paramref name="elementCount"/> FP32 elements.</summary>
    public void ClearRegion(Tensor dst, long elementOffset, int elementCount)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        nint dP = GetDevPtr(dst) + (nint)(elementOffset * sizeof(float));
        int  pN = elementCount;
        nint* args = stackalloc nint[2] { (nint)(&dP), (nint)(&pN) };
        uint grid = (uint)((elementCount + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_clearF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(clear region) failed: {r}");
    }

    /// <summary>In-place SiLU activation: x[i] = x[i] / (1 + exp(-x[i])). One thread per element.</summary>
    public void SiLUInPlace(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xP = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xP), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_siluInplaceKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(silu_inplace) failed: {r}");
    }

    /// <summary>
    /// GDN depthwise causal conv1d for a single decode token. State layout
    /// <c>[(kernel-1), channels]</c> row-major, oldest first; updated in place.
    /// Weight layout <c>[kernel, channels]</c>.
    /// </summary>
    public void GdnConv1dDecode(Tensor x, Tensor state, Tensor weight, Tensor output,
                                int channels, int kernelSize)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x);
        nint sP = GetDevPtr(state);
        nint wP = GetDevPtr(weight);
        nint oP = GetDevPtr(output);
        int  pC = channels, pK = kernelSize;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xP), (nint)(&sP), (nint)(&wP), (nint)(&oP),
            (nint)(&pC), (nint)(&pK)
        };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dDecodeKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_decode) failed: {r}");
    }

    /// <summary>
    /// L2 normalize each <paramref name="headDim"/>-sized slice independently
    /// (no learned weights). Matches <c>GdnKernels.L2NormPerHead</c>:
    /// <c>scale = 1 / max(sqrt(Σ x²), eps)</c>. Operates on the sub-region of
    /// <paramref name="data"/> starting at <paramref name="elementOffset"/>
    /// for <paramref name="numHeads"/> × <paramref name="headDim"/> floats.
    /// </summary>
    public void GdnL2NormPerHead(Tensor data, long elementOffset, int numHeads, int headDim, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dP = GetDevPtr(data) + (nint)(elementOffset * sizeof(float));
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        nint* args = stackalloc nint[4]
        {
            (nint)(&dP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_gdnL2NormPerHeadKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_l2_norm) failed: {r}");
    }

    /// <summary>
    /// Tile pattern: dst[h_dst, j] = src[h_dst % srcHeads, j] for h_dst ∈ [0, srcHeads·repeat).
    /// Matches <c>GdnKernels.TileHeads</c> (GQA-style broadcast, NOT torch repeat_interleave).
    /// <paramref name="srcOffset"/> and <paramref name="dstOffset"/> are FP32 element offsets.
    /// </summary>
    public void GdnTileHeads(Tensor src, long srcOffset, Tensor dst, long dstOffset,
                             int srcHeads, int repeat, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(src) + (nint)(srcOffset * sizeof(float));
        nint dP = GetDevPtr(dst) + (nint)(dstOffset * sizeof(float));
        int  pSH = srcHeads, pR = repeat, pHD = headDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP),
            (nint)(&pSH), (nint)(&pR), (nint)(&pHD)
        };
        int total = srcHeads * repeat * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnTileHeadsKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_tile_heads) failed: {r}");
    }

    /// <summary>
    /// Single-token Gated DeltaNet recurrence. Updates per-head state matrices in
    /// place and writes per-head post-norm, post-gate output. Mirrors
    /// <c>GdnKernels.GdnRecurrenceDecode</c>.
    /// State layout: <c>[numVHeads, headDim, headDim]</c> row-major (i = key axis,
    /// j = value/output axis).
    /// </summary>
    public void GdnRecurrenceDecode(
        Tensor state, Tensor q, Tensor k, Tensor v,
        Tensor alphaIn, Tensor beta, Tensor ssmA, Tensor dtBias,
        Tensor normWeight, Tensor z, Tensor output,
        int numVHeads, int headDim, float normEps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(state);
        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(k);
        nint vP = GetDevPtr(v);
        nint aP = GetDevPtr(alphaIn);
        nint bP = GetDevPtr(beta);
        nint aaP = GetDevPtr(ssmA);
        nint dbP = GetDevPtr(dtBias);
        nint nwP = GetDevPtr(normWeight);
        nint zP = GetDevPtr(z);
        nint oP = GetDevPtr(output);
        int  pHV = numVHeads, pD = headDim;
        float pE = normEps;
        nint* args = stackalloc nint[14]
        {
            (nint)(&sP), (nint)(&qP), (nint)(&kP), (nint)(&vP),
            (nint)(&aP), (nint)(&bP), (nint)(&aaP), (nint)(&dbP),
            (nint)(&nwP), (nint)(&zP), (nint)(&oP),
            (nint)(&pHV), (nint)(&pD), (nint)(&pE)
        };
        // Shared memory: 8 × headDim × 4 bytes (sK, sQ, sV, sZ, sNormW, sP, sD, sRed).
        uint sharedBytes = (uint)(8 * headDim * sizeof(float));
        int r = NvrtcInterop.LaunchKernel(_gdnRecurrenceDecodeKernel,
            (uint)numVHeads, 1, 1, (uint)headDim, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_recurrence_decode) failed: {r}");
    }

    // ── Issue #114-B: batched GDN trunk over a chunk of N prompt tokens ────────

    /// <summary>
    /// Batched GDN depthwise conv1d over <paramref name="nTok"/> tokens (read-only
    /// state). <paramref name="x"/>/<paramref name="output"/> are <c>[nTok × channels]</c>;
    /// <paramref name="state"/> is the carried <c>[(K-1) × channels]</c>. Bit-identical to
    /// nTok sequential <see cref="GdnConv1dDecode"/> calls. State is advanced separately by
    /// <see cref="GdnConv1dStateUpdateBatched"/> (so concurrent token blocks read one snapshot).
    /// </summary>
    public void GdnConv1dDecodeBatched(Tensor x, Tensor state, Tensor weight, Tensor output,
                                       int channels, int kernelSize, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x), sP = GetDevPtr(state), wP = GetDevPtr(weight), oP = GetDevPtr(output);
        int pC = channels, pK = kernelSize, pN = nTok;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xP), (nint)(&sP), (nint)(&wP), (nint)(&oP),
            (nint)(&pC), (nint)(&pK), (nint)(&pN)
        };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dDecodeBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_decode_batched) failed: {r}");
    }

    /// <summary>Advance the conv1d state past a chunk of <paramref name="nTok"/> tokens
    /// (matches the sequential state evolution). See <see cref="GdnConv1dDecodeBatched"/>.</summary>
    public void GdnConv1dStateUpdateBatched(Tensor x, Tensor state, int channels, int kernelSize, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x), sP = GetDevPtr(state);
        int pC = channels, pK = kernelSize, pN = nTok;
        nint* args = stackalloc nint[5] { (nint)(&xP), (nint)(&sP), (nint)(&pC), (nint)(&pK), (nint)(&pN) };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dStateUpdateBatchedKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_state_update_batched) failed: {r}");
    }

    /// <summary>
    /// Batched per-head L2-norm over <paramref name="nTok"/> tokens. <paramref name="data"/>
    /// is offset to the region base; <paramref name="rowStride"/> is the per-token element
    /// stride. grid = (numHeads, nTok). Bit-identical to nTok sequential
    /// <see cref="GdnL2NormPerHead"/> calls.
    /// </summary>
    public void GdnL2NormPerHeadBatched(Tensor data, long elementOffset, int numHeads, int headDim,
                                        int rowStride, int nTok, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dP = GetDevPtr(data) + (nint)(elementOffset * sizeof(float));
        int pHD = headDim, pNH = numHeads, pRS = rowStride, pN = nTok;
        float pE = eps;
        nint* args = stackalloc nint[6]
        {
            (nint)(&dP), (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pRS), (nint)(&pN)
        };
        int r = NvrtcInterop.LaunchKernel(_gdnL2NormPerHeadBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_l2_norm_per_head_batched) failed: {r}");
    }

    /// <summary>
    /// Batched GQA-broadcast tile over <paramref name="nTok"/> tokens. <paramref name="src"/>
    /// is offset to the region base; <paramref name="srcStride"/>/<paramref name="dstStride"/>
    /// are per-token strides. Bit-identical to nTok sequential <see cref="GdnTileHeads"/> calls.
    /// </summary>
    public void GdnTileHeadsBatched(Tensor src, long srcOffset, Tensor dst, long dstOffset,
                                    int srcHeads, int repeat, int headDim,
                                    int srcStride, int dstStride, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(src) + (nint)(srcOffset * sizeof(float));
        nint dP = GetDevPtr(dst) + (nint)(dstOffset * sizeof(float));
        int pSH = srcHeads, pR = repeat, pHD = headDim, pSS = srcStride, pDS = dstStride, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&pSH), (nint)(&pR), (nint)(&pHD),
            (nint)(&pSS), (nint)(&pDS), (nint)(&pN)
        };
        int total = srcHeads * repeat * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnTileHeadsBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_tile_heads_batched) failed: {r}");
    }

    /// <summary>
    /// Fused sequential GDN recurrence scan over <paramref name="nTok"/> tokens: one launch
    /// (one block per v-head) loops the positions internally, carrying the per-head state in
    /// place. Bit-identical to nTok sequential <see cref="GdnRecurrenceDecode"/> calls — the
    /// fused form of the per-token decode, NOT the parallel chunked-scan. Per-head input
    /// strides let q/k come from the tiled <c>[nTok × valueDim]</c> buffers, v from the
    /// silu'd conv output (<paramref name="vHeadOff"/> into a <c>[nTok × convChannels]</c>
    /// buffer), z from a <c>[nTok × valueDim]</c> gate, alpha/beta from <c>[nTok × numVHeads]</c>.
    /// </summary>
    public void GdnRecurrenceScan(
        Tensor state, Tensor qAll, Tensor kAll, Tensor vAll,
        Tensor alphaAll, Tensor betaAll, Tensor ssmA, Tensor dtBias,
        Tensor normWeight, Tensor zAll, Tensor outputAll,
        int numVHeads, int headDim, float normEps,
        int qStride, int kStride, int vStride, int vHeadOff, int zStride, int oStride, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(state);
        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kAll), vP = GetDevPtr(vAll);
        nint aP = GetDevPtr(alphaAll), bP = GetDevPtr(betaAll);
        nint aaP = GetDevPtr(ssmA), dbP = GetDevPtr(dtBias), nwP = GetDevPtr(normWeight);
        nint zP = GetDevPtr(zAll), oP = GetDevPtr(outputAll);
        int pHV = numVHeads, pD = headDim;
        float pE = normEps;
        int pQS = qStride, pKS = kStride, pVS = vStride, pVO = vHeadOff, pZS = zStride, pOS = oStride, pN = nTok;
        nint* args = stackalloc nint[21]
        {
            (nint)(&sP), (nint)(&qP), (nint)(&kP), (nint)(&vP),
            (nint)(&aP), (nint)(&bP), (nint)(&aaP), (nint)(&dbP),
            (nint)(&nwP), (nint)(&zP), (nint)(&oP),
            (nint)(&pHV), (nint)(&pD), (nint)(&pE),
            (nint)(&pQS), (nint)(&pKS), (nint)(&pVS), (nint)(&pVO), (nint)(&pZS), (nint)(&pOS), (nint)(&pN)
        };
        uint sharedBytes = (uint)(8 * headDim * sizeof(float));
        int r = NvrtcInterop.LaunchKernel(_gdnRecurrenceScanKernel,
            (uint)numVHeads, 1, 1, (uint)headDim, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_recurrence_scan) failed: {r}");
    }

    public void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                                 int nTok, int nHeads, int headDim, float scale) =>
        throw new NotSupportedException("CudaBackend.FullSeqAttention is not implemented (LLM path uses single-token Attention).");

    // ── Helpers ───────────────────────────────────────────────────────────

    private nint GetDevPtr(Tensor tensor) =>
        _devPtrs.TryGetValue(tensor.Handle, out var entry)
            ? entry.devPtr
            : throw new InvalidOperationException($"Tensor handle {tensor.Handle} not registered in CudaBackend");

    // ── IImageOpsBackend ──────────────────────────────────────────────────

    /// <summary>
    /// Returns true when NVRTC image kernels are available and compiled.
    /// Triggers lazy compilation on first call.
    /// </summary>
    public bool ImageKernelsAvailable
    {
        get
        {
            EnsureImageKernels();
            return _imageKernelsAvailable;
        }
    }

    /// <summary>
    /// Lazily compile all image-ops CUDA kernels via NVRTC and load the resulting PTX.
    /// Idempotent: subsequent calls are a no-op once initialised (success or failure).
    /// On failure sets <c>_imageKernelsAvailable = false</c> so callers can fall back gracefully.
    /// </summary>
    private void EnsureImageKernels()
    {
        // Always bind the primary CUDA context on the calling thread — even after the
        // module is loaded. cuModuleLoadData (init) and cuLaunchKernel (every launch
        // below) both require a current context, but cuBLAS handles bind to it
        // internally so the rest of the backend works cross-thread without this.
        //
        // Hosted scenarios (e.g. ASP.NET request threads) construct CudaBackend on
        // one thread and decode on another — without this guard cuModuleLoadData
        // returns CUDA_ERROR_INVALID_CONTEXT (201). See issue #94.
        EnsurePrimaryContextCurrent();

        if (_imageKernelsInitialized) return;
        lock (_kernelInitLock)
        {
            if (_imageKernelsInitialized) return;
            try
            {
                CompileAndLoadKernels();
                _imageKernelsAvailable = true;
            }
            catch (Exception ex)
            {
                _imageKernelsAvailable = false;
                // Log to stderr so the user can see NVRTC failure reason when debugging.
                Console.Error.WriteLine($"[CudaBackend] NVRTC kernel init failed: {ex.Message}");
            }
            finally
            {
                _imageKernelsInitialized = true;
            }
        }
    }

    // ── CUDA primary-context binding ──────────────────────────────────────
    //
    // Process-wide retained primary context (device 0), made current on each thread
    // that calls into the NVRTC kernel path. ThreadStatic flag skips redundant
    // cuCtxSetCurrent on a thread that's already bound.

    private static readonly object   s_primaryCtxLock = new();
    private static          nint     s_primaryCtx;
    private static          int      s_primaryCtxDevice = -1;
    [ThreadStatic]
    private static          bool     t_primaryCtxBoundOnThisThread;

    /// <summary>
    /// Ensure the device's primary context is current on the calling thread. Cheap after
    /// the first call per thread (single ThreadStatic check). The retained primary context
    /// is the same one cuBLAS attaches to, so once it's current on a thread the entire
    /// CUDA Driver API surface works there.
    /// </summary>
    internal static void EnsurePrimaryContextCurrent()
    {
        if (t_primaryCtxBoundOnThisThread) return;

        if (s_primaryCtx == nint.Zero)
        {
            lock (s_primaryCtxLock)
            {
                if (s_primaryCtx == nint.Zero)
                {
                    int ir = NvrtcInterop.CuInit(0);
                    if (ir != 0) throw new InvalidOperationException($"cuInit failed: {ir}");

                    int dr = NvrtcInterop.DeviceGet(out int dev, 0);
                    if (dr != 0) throw new InvalidOperationException($"cuDeviceGet failed: {dr}");

                    int rr = NvrtcInterop.DevicePrimaryCtxRetain(out nint ctx, dev);
                    if (rr != 0) throw new InvalidOperationException($"cuDevicePrimaryCtxRetain failed: {rr}");

                    s_primaryCtxDevice = dev;
                    s_primaryCtx       = ctx;
                }
            }
        }

        int sr = NvrtcInterop.CtxSetCurrent(s_primaryCtx);
        if (sr != 0) throw new InvalidOperationException($"cuCtxSetCurrent failed: {sr}");
        t_primaryCtxBoundOnThisThread = true;
    }

    /// <summary>Combined CUDA source for image + text kernels — one NVRTC compilation.</summary>
    private static string CombinedKernelSource => CudaKernels.Source + "\n" + CudaTextKernels.Source;

    private void CompileAndLoadKernels()
    {
        // Ensure the CUDA Driver API context exists (shares the primary context with the runtime).
        NvrtcInterop.CuInit(0);

        // Try to load from cubin cache first (avoids both NVRTC compilation and PTX JIT overhead).
        string cacheFile = GetCubinCachePath();
        if (TryLoadCubinFromCache(cacheFile)) return;

        byte[] srcBytes  = NvrtcInterop.ToUtf8(CombinedKernelSource);
        byte[] nameBytes = NvrtcInterop.ToUtf8("sharpi_kernels.cu");

        nint prog = nint.Zero;
        fixed (byte* pSrc = srcBytes)
        fixed (byte* pName = nameBytes)
        {
            int r = NvrtcInterop.CreateProgram(out prog, pSrc, pName, 0, nint.Zero, nint.Zero);
            if (r != 0) throw new InvalidOperationException($"nvrtcCreateProgram failed: {r}");
        }

        byte[]? binary = null;
        try
        {
            // Compile targeting the actual GPU's SM version to get a cubin (no JIT at launch).
            // Falls back to PTX (with JIT overhead) if the SM version is unknown or cubin fails.
            string archFlag = _smVersion > 0 ? $"--gpu-architecture=sm_{_smVersion}" : "--gpu-architecture=compute_52";
            byte[] archBytes = NvrtcInterop.ToUtf8(archFlag);
            int rc;
            fixed (byte* pArch = archBytes)
            {
                nint opts = (nint)(&pArch);
                rc = NvrtcInterop.CompileProgramWithOptions(prog, 1, opts);
            }
            if (rc != 0)
            {
                NvrtcInterop.GetProgramLogSize(prog, out nuint logSize);
                byte[] logBuf = new byte[(int)logSize];
                string log;
                fixed (byte* pLog = logBuf)
                {
                    NvrtcInterop.GetProgramLog(prog, pLog);
                    log = System.Text.Encoding.UTF8.GetString(logBuf);
                }
                throw new InvalidOperationException($"nvrtcCompileProgram failed ({rc}):\n{log}");
            }

            // Prefer cubin (no JIT) over PTX (lazy JIT on first kernel launch = slow).
            // nvrtcGetCubin requires NVRTC 11.1+; fall through to PTX on older versions.
            bool isCubin = false;
            int cubinRc = -1;
            nuint cubinSize = 0;
            try
            {
                cubinRc = NvrtcInterop.GetCubinSize(prog, out cubinSize);
                if (cubinRc == 0 && cubinSize > 0)
                {
                    binary = new byte[(int)cubinSize];
                    fixed (byte* pBin = binary)
                    {
                        int r2 = NvrtcInterop.GetCubin(prog, pBin);
                        if (r2 != 0) { binary = null; cubinRc = r2; }
                        else isCubin = true;
                    }
                }
            }
            catch (EntryPointNotFoundException)
            {
                // nvrtcGetCubin / nvrtcGetCubinSize unavailable (NVRTC < 11.1).
                binary = null;
            }

            if (binary is null)
            {
                // Fall back to PTX (triggers JIT at first kernel launch, slower).
                Console.Error.WriteLine(
                    $"[CudaBackend] NVRTC produced no cubin for sm_{_smVersion} " +
                    $"(nvrtcGetCubinSize rc={cubinRc}, size={cubinSize}); falling back to PTX. " +
                    "Expect ~6× slower first-token latency from per-kernel PTX→SASS JIT.");
                NvrtcInterop.GetPTXSize(prog, out nuint ptxSize);
                binary = new byte[(int)ptxSize];
                fixed (byte* pPtx = binary)
                {
                    NvrtcInterop.GetPTX(prog, pPtx);
                }
            }

            fixed (byte* pBin = binary)
            {
                int mr = NvrtcInterop.ModuleLoadData(out _nvModule, pBin);
                if (mr != 0) throw new InvalidOperationException($"cuModuleLoadData failed: {mr}");
            }

            // Persist *only real cubin* to disk; caching PTX would make every future run
            // re-pay the JIT cost (and obscure the diagnostic above).
            if (isCubin)
            {
                try { File.WriteAllBytes(cacheFile, binary); }
                catch { /* ignore cache write failures */ }
            }
            else
            {
                // Stale PTX or partial cubin from a previous run would mask the problem.
                try { if (File.Exists(cacheFile)) File.Delete(cacheFile); }
                catch { /* ignore */ }
            }
        }
        finally
        {
            NvrtcInterop.DestroyProgram(ref prog);
        }

        LoadKernelFunctions();
        ForceEagerJit();
    }

    private string GetCubinCachePath()
    {
        // Cache key: SHA-256 of combined kernel source + SM version.
        // Any source change or GPU change invalidates the cache.
        byte[] hash = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(CombinedKernelSource + _smVersion));
        string hex = Convert.ToHexString(hash)[..16];
        return Path.Combine(Path.GetTempPath(), $"sharpi_cubin_sm{_smVersion}_{hex}.bin");
    }

    private bool TryLoadCubinFromCache(string cacheFile)
    {
        if (!File.Exists(cacheFile)) return false;
        try
        {
            byte[] cubinBuf = File.ReadAllBytes(cacheFile);

            // Defend against an older sharpi build that wrote PTX into the cubin path:
            // PTX would still load via cuModuleLoadData, but every kernel would then pay
            // PTX→SASS JIT on first launch. Real cubin is ELF — magic "\x7FELF".
            if (cubinBuf.Length < 4 ||
                cubinBuf[0] != 0x7F || cubinBuf[1] != (byte)'E' ||
                cubinBuf[2] != (byte)'L' || cubinBuf[3] != (byte)'F')
            {
                Console.Error.WriteLine(
                    $"[CudaBackend] Cubin cache at {cacheFile} is not ELF (likely stale PTX " +
                    "from an earlier build); deleting and recompiling.");
                try { File.Delete(cacheFile); } catch { }
                return false;
            }

            fixed (byte* pBin = cubinBuf)
            {
                int mr = NvrtcInterop.ModuleLoadData(out _nvModule, pBin);
                if (mr != 0) return false;
            }
            LoadKernelFunctions();
            ForceEagerJit();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Walk every loaded kernel handle and query <c>CU_FUNC_ATTRIBUTE_NUM_REGS</c>.
    /// Querying a function attribute forces the driver to finalize SASS for that kernel
    /// right now, even under lazy module loading or when the module was loaded from PTX.
    /// Without this, the first decode pays per-kernel JIT during the user's first prefill.
    /// </summary>
    private void ForceEagerJit()
    {
        ReadOnlySpan<nint> kernels = [
            _im2colKernel, _biasAddKernel, _leakyReluKernel, _scaleKernel, _addKernel,
            _addScaledKernel, _clampKernel, _pshuffleKernel, _punshuffleKernel, _upsample2xKernel,
            _rmsNormKernel, _headNormKernel, _headNormPureKernel, _siluMulKernel, _sigmoidKernel,
            _softmaxKernel, _ropeInterleavedKernel, _ropeNeoxKernel, _ropeNeoxPartialKernel,
            _ropeNeoxWithFactorsKernel, _ropeNeoxWithFactorsBatchedKernel,
            _mulKernel, _sigmoidMulInPlaceKernel, _splitQgKernel, _kvAppendKernel,
            _kvAppendBf16Kernel,
            _snapKvScoreKernel, _snapKvScoreBf16Kernel,
            _kvCompactKernel, _kvCompactBf16Kernel,
            _embedLookupF32Kernel, _embedLookupQ4KKernel, _embedLookupQ5KKernel,
            _embedLookupQ80Kernel,
            _matvecF32Kernel, _matvecQ4KKernel, _matvecQ5KKernel, _matvecQ6KKernel,
            _matvecQ80Kernel,
            _matvecF32N2Kernel, _matvecQ4KN2Kernel, _matvecQ5KN2Kernel, _matvecQ6KN2Kernel,
            _matvecF32GemmNKernel, _matvecQ4KGemmNKernel, _matvecQ5KGemmNKernel, _matvecQ6KGemmNKernel,
            _matvecQ80GemmNKernel,
            _rmsNormBatchedKernel, _headNormBatchedKernel, _headNormQkKernel, _headNormQkBatchedKernel,
            _splitQgBatchedKernel, _ropeNeoxPartialBatchedKernel,
            _attentionKernel, _attentionBf16Kernel, _attentionSwaKernel, _attentionSwaBatchedKernel,
            _geluTanhMulKernel, _geluTanhMulStridedKernel, _softcapKernel,
            _clearF32Kernel, _quantizeQ81Kernel,
            _scaleRowsKernel, _moeWeightedReduceKernel,
            _tqRotateQueryKernel, _tqKvAppendKernel, _tqAttentionKernel,
            _siluInplaceKernel, _gdnConv1dDecodeKernel, _gdnL2NormPerHeadKernel,
            _gdnTileHeadsKernel, _gdnRecurrenceDecodeKernel,
            _gdnConv1dDecodeBatchedKernel, _gdnConv1dStateUpdateBatchedKernel,
            _gdnL2NormPerHeadBatchedKernel, _gdnTileHeadsBatchedKernel, _gdnRecurrenceScanKernel,
            _kvAppendBatchedKernel, _kvAppendBatchedBf16Kernel,
            _fullSeqAttentionKernel, _fullSeqAttentionBf16Kernel,
            _fullSeqAttentionGlobalKernel, _fullSeqAttentionGlobalBf16Kernel,
        ];
        foreach (nint k in kernels)
        {
            if (k != nint.Zero)
                NvrtcInterop.FuncGetAttribute(out _, NvrtcInterop.CU_FUNC_ATTRIBUTE_NUM_REGS, k);
        }
    }

    private void LoadKernelFunctions()
    {
        _im2colKernel      = GetKernelFunc("im2col");
        _biasAddKernel     = GetKernelFunc("bias_add");
        _leakyReluKernel   = GetKernelFunc("leaky_relu_inplace");
        _scaleKernel       = GetKernelFunc("scale_inplace");
        _addKernel         = GetKernelFunc("add_inplace");
        _addScaledKernel   = GetKernelFunc("add_scaled_inplace");
        _clampKernel       = GetKernelFunc("clamp_inplace");
        _pshuffleKernel    = GetKernelFunc("pixel_shuffle");
        _punshuffleKernel  = GetKernelFunc("pixel_unshuffle");
        _upsample2xKernel  = GetKernelFunc("upsample2x");

        // LLM kernels (same NVRTC module).
        _rmsNormKernel         = GetKernelFunc("llm_rmsnorm");
        _headNormKernel        = GetKernelFunc("llm_head_norm");
        _headNormPureKernel    = GetKernelFunc("llm_head_norm_pure");
        _siluMulKernel         = GetKernelFunc("llm_silu_mul");
        _sigmoidKernel         = GetKernelFunc("llm_sigmoid_inplace");
        _softmaxKernel         = GetKernelFunc("llm_softmax");
        _ropeInterleavedKernel = GetKernelFunc("llm_rope_interleaved");
        _ropeNeoxKernel        = GetKernelFunc("llm_rope_neox");
        _ropeNeoxPartialKernel = GetKernelFunc("llm_rope_neox_partial");
        _ropeNeoxWithFactorsKernel = GetKernelFunc("llm_rope_neox_with_factors");
        _mulKernel             = GetKernelFunc("llm_mul");
        _sigmoidMulInPlaceKernel = GetKernelFunc("llm_sigmoid_mul_inplace");
        _splitQgKernel         = GetKernelFunc("llm_split_qg");
        _kvAppendKernel        = GetKernelFunc("llm_kv_append");
        _kvAppendBf16Kernel    = GetKernelFunc("llm_kv_append_bf16");
        _snapKvScoreKernel     = GetKernelFunc("llm_snapkv_score");
        _snapKvScoreBf16Kernel = GetKernelFunc("llm_snapkv_score_bf16");
        _kvCompactKernel       = GetKernelFunc("llm_kv_compact");
        _kvCompactBf16Kernel   = GetKernelFunc("llm_kv_compact_bf16");
        _embedLookupF32Kernel  = GetKernelFunc("llm_embed_lookup_f32");
        _embedLookupQ4KKernel  = GetKernelFunc("llm_embed_lookup_q4k");
        _embedLookupQ5KKernel  = GetKernelFunc("llm_embed_lookup_q5k");
        _embedLookupQ80Kernel  = GetKernelFunc("llm_embed_lookup_q8_0");
        _matvecF32Kernel       = GetKernelFunc("llm_matvec_f32");
        _matvecQ4KKernel       = GetKernelFunc("llm_matvec_q4k");
        _matvecQ5KKernel       = GetKernelFunc("llm_matvec_q5k");
        _matvecQ6KKernel       = GetKernelFunc("llm_matvec_q6k");
        _matvecQ80Kernel       = GetKernelFunc("llm_matvec_q8_0");
        _matvecF32N2Kernel     = GetKernelFunc("llm_matvec_f32_n2");
        _matvecQ4KN2Kernel     = GetKernelFunc("llm_matvec_q4k_n2");
        _matvecQ5KN2Kernel     = GetKernelFunc("llm_matvec_q5k_n2");
        _matvecQ6KN2Kernel     = GetKernelFunc("llm_matvec_q6k_n2");
        _matvecF32GemmNKernel  = GetKernelFunc("llm_matvec_f32_gemm_n");
        _matvecQ4KGemmNKernel  = GetKernelFunc("llm_matvec_q4k_gemm_n");
        _matvecQ5KGemmNKernel  = GetKernelFunc("llm_matvec_q5k_gemm_n");
        _matvecQ6KGemmNKernel  = GetKernelFunc("llm_matvec_q6k_gemm_n");
        _matvecQ80GemmNKernel  = GetKernelFunc("llm_matvec_q8_0_gemm_n");
        _rmsNormBatchedKernel  = GetKernelFunc("llm_rmsnorm_batched");
        _headNormBatchedKernel = GetKernelFunc("llm_head_norm_batched");
        _headNormQkKernel        = GetKernelFunc("llm_head_norm_qk");
        _headNormQkBatchedKernel = GetKernelFunc("llm_head_norm_qk_batched");
        _splitQgBatchedKernel  = GetKernelFunc("llm_split_qg_batched");
        _ropeNeoxPartialBatchedKernel = GetKernelFunc("llm_rope_neox_partial_batched");
        _ropeNeoxWithFactorsBatchedKernel = GetKernelFunc("llm_rope_neox_with_factors_batched");
        _attentionKernel       = GetKernelFunc("llm_attention");
        _attentionBf16Kernel   = GetKernelFunc("llm_attention_bf16");
        _attentionSwaKernel    = GetKernelFunc("llm_attention_swa");
        _attentionSwaBatchedKernel = GetKernelFunc("llm_attention_swa_batched");
        _geluTanhMulKernel     = GetKernelFunc("llm_gelu_tanh_mul");
        _geluTanhMulStridedKernel = GetKernelFunc("llm_gelu_tanh_mul_strided");
        _softcapKernel         = GetKernelFunc("llm_softcap_inplace");
        _clearF32Kernel        = GetKernelFunc("llm_clear_f32");
        _quantizeQ81Kernel     = GetKernelFunc("llm_quantize_q8_1");
        _bwBaselineKernel      = GetKernelFunc("llm_bw_baseline");
        _scaleRowsKernel       = GetKernelFunc("llm_scale_rows_inplace");
        _moeWeightedReduceKernel = GetKernelFunc("llm_moe_weighted_reduce");

        // TurboQuant kernels (loaded from the same NVRTC module).
        _tqRotateQueryKernel = GetKernelFunc("llm_tq_rotate_query");
        _tqKvAppendKernel    = GetKernelFunc("llm_tq_kv_append");
        _tqAttentionKernel   = GetKernelFunc("llm_tq_attention");

        // qwen35moe GDN kernels.
        _siluInplaceKernel        = GetKernelFunc("llm_silu_inplace");
        _gdnConv1dDecodeKernel    = GetKernelFunc("llm_gdn_conv1d_decode");
        _gdnL2NormPerHeadKernel   = GetKernelFunc("llm_gdn_l2_norm_per_head");
        _gdnTileHeadsKernel       = GetKernelFunc("llm_gdn_tile_heads");
        _gdnRecurrenceDecodeKernel = GetKernelFunc("llm_gdn_recurrence_decode");

        // qwen35moe GDN batched trunk + batched-query SDPA kernels (issue #114-B).
        _gdnConv1dDecodeBatchedKernel      = GetKernelFunc("llm_gdn_conv1d_decode_batched");
        _gdnConv1dStateUpdateBatchedKernel = GetKernelFunc("llm_gdn_conv1d_state_update_batched");
        _gdnL2NormPerHeadBatchedKernel     = GetKernelFunc("llm_gdn_l2_norm_per_head_batched");
        _gdnTileHeadsBatchedKernel         = GetKernelFunc("llm_gdn_tile_heads_batched");
        _gdnRecurrenceScanKernel           = GetKernelFunc("llm_gdn_recurrence_scan");
        _kvAppendBatchedKernel             = GetKernelFunc("llm_kv_append_batched");
        _kvAppendBatchedBf16Kernel         = GetKernelFunc("llm_kv_append_batched_bf16");
        _fullSeqAttentionKernel            = GetKernelFunc("llm_full_seq_attention");
        _fullSeqAttentionBf16Kernel        = GetKernelFunc("llm_full_seq_attention_bf16");
        _fullSeqAttentionGlobalKernel      = GetKernelFunc("llm_full_seq_attention_global");
        _fullSeqAttentionGlobalBf16Kernel  = GetKernelFunc("llm_full_seq_attention_global_bf16");
    }

    /// <summary>
    /// Diagnostic: launches a pure-streaming copy kernel over <paramref name="bytes"/>
    /// bytes (must be multiple of 4) so the caller can measure achievable HBM bandwidth.
    /// </summary>
    public void RunBandwidthBaseline(nint srcDev, nint dstDev, int bytes)
    {
        EnsureImageKernels();
        int n_uint4 = bytes / 16;
        nint p0 = srcDev, p1 = dstDev;
        int  p2 = n_uint4;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        uint grid = (uint)((n_uint4 + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_bwBaselineKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(bw_baseline) failed: {r}");
    }

    private nint GetKernelFunc(string name)
    {
        byte[] nameBytes = NvrtcInterop.ToUtf8(name);
        fixed (byte* pName = nameBytes)
        {
            int r = NvrtcInterop.ModuleGetFunction(out nint func, _nvModule, pName);
            if (r != 0) throw new InvalidOperationException($"cuModuleGetFunction({name}) failed: {r}");
            return func;
        }
    }

    /// <summary>Launch a 1-D kernel with 1024 threads per block over <paramref name="total"/> elements.</summary>
    private void Launch1D(nint kernel, int total, nint* args)
    {
        uint grid = (uint)((total + 1023) / 1024);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 1024, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel failed: {r}");
    }

    /// <summary>
    /// Ensure the GPU im2col tile buffer is at least <paramref name="minBytes"/> bytes.
    /// On first call, allocates exactly <see cref="MaxTileBytes"/> (2.5 GiB) so that all
    /// possible tile sizes for any RRDB or upsample layer fit in a single tile without
    /// reallocation. Single-tile mode keeps lda=ldc=N so all cuBLAS reads/writes are
    /// contiguous — multi-tile with strided ldc is never needed.
    /// </summary>
    private void EnsureIm2ColBuf(long minBytes)
    {
        if (_im2colBuf != nint.Zero && _im2colBufSize >= (nuint)minBytes) return;
        if (_im2colBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_im2colBuf);
            _im2colBuf     = nint.Zero;
            _im2colBufSize = 0;
        }
        // Allocate MaxTileBytes so subsequent calls never need to grow.
        // All valid tilePixels (aligned to full rows) produce minBytes ≤ MaxTileBytes.
        nuint newSize = (nuint)MaxTileBytes;
        int r = CuBlasInterop.CudaMalloc(out _im2colBuf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc({newSize / 1024 / 1024} MiB im2col buf) failed: {r}");
        _im2colBufSize = newSize;
    }

    /// <inheritdoc/>
    public Tensor Conv2d(Tensor input, Tensor weight, Tensor bias,
                         int inCh, int outCh, int h, int w, int ksize, int padding = -1)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if (ksize != 3)
            throw new NotSupportedException($"Conv2d CUDA only supports ksize=3 (got ksize={ksize}).");

        int N = h * w;
        int K = inCh * 9;   // im2col columns

        nint inputPtr  = GetDevPtr(input);
        nint weightPtr = GetDevPtr(weight);
        nint biasPtr   = GetDevPtr(bias);

        // ── Output allocation (CHW: [outCh, N] row-major) ──────────────────
        var  output    = Allocate(TensorShape.D1((long)outCh * N));
        nint outputPtr = GetDevPtr(output);

        // ── Tile size ───────────────────────────────────────────────────────
        // With MaxTileBytes=2.5 GiB, every real layer fits in a single tile:
        //   RRDB max (K=1728, N=262144): 1.81 GiB  < 2.5 GiB ✓
        //   Upsample  (K=576,  N=4M):   2.41 GiB  < 2.5 GiB ✓
        // Single-tile: lda=tileN=N, ldc=N — all cuBLAS accesses are contiguous.
        int tilePixels = (int)Math.Min((long)N, MaxTileBytes / ((long)K * sizeof(float)));
        tilePixels = Math.Max(tilePixels, w); // at least one full row per tile

        // Align tile to complete rows so ph_start = pixel_start / w is integer
        tilePixels = (tilePixels / w) * w;

        EnsureIm2ColBuf((long)tilePixels * K * sizeof(float));

        float alpha = 1.0f, beta = 0.0f;

        // Hoist kernel-arg pointers outside the loop (CA2014: no stackalloc in loops).
        // Only cp5 (ph_start) and cp6 (tileN) vary per tile; we update them before each launch.
        nint cp0 = inputPtr, cp1 = _im2colBuf;
        int  cp2 = h, cp3 = w, cp4 = N, cp5 = 0, cp6 = 0, cp7 = inCh, cp8 = K;
        nint* args = stackalloc nint[9]
        {
            (nint)(&cp0), (nint)(&cp1),
            (nint)(&cp2), (nint)(&cp3), (nint)(&cp4),
            (nint)(&cp5), (nint)(&cp6),
            (nint)(&cp7), (nint)(&cp8)
        };

        for (int pixelStart = 0; pixelStart < N; pixelStart += tilePixels)
        {
            int tileN    = Math.Min(tilePixels, N - pixelStart);
            cp5 = pixelStart / w;  // ph_start
            cp6 = tileN;

            // ── im2col kernel: fills _im2colBuf[K, tileN] ──────────────────
            // Block (32=pixel, 8=k) — consecutive tx (pixel) → coalesced writes.
            // Grid (ceil(tileN/32), ceil(K/8)).
            {
                uint grX = ((uint)tileN + 31) / 32;
                uint grY = ((uint)K     +  7) / 8;
                int er = NvrtcInterop.LaunchKernel(_im2colKernel, grX, grY, 1, 32, 8, 1, 0, _stream, args, null);
                if (er != 0) throw new InvalidOperationException($"im2col launch failed: {er}");
            }

            // ── GEMM: C = A*B where A=col[K,tileN], B=weight[K,outCh], C=out[tileN,outCh] ─
            // col[K, tileN]: column k starts at k*tileN → lda=tileN (contiguous columns).
            // weight[outCh, K] row-major = [K, outCh] col-major → ldb=K.
            // Output at outputPtr + pixelStart, ldc=N → C[pixel, oc] = out[pixelStart+pixel+oc*N].
            nint gemmDst = outputPtr + (nint)(pixelStart * sizeof(float));
            int gr = CuBlasInterop.Sgemm(
                _handle,
                CuBlasInterop.OpN, CuBlasInterop.OpN,
                tileN, outCh, K,
                ref alpha,
                _im2colBuf, tileN,   // A=[K,tileN] col-major, lda=tileN
                weightPtr,  K,       // B=[K,outCh] col-major, ldb=K
                ref beta,
                gemmDst, N);         // C=[tileN,outCh] col-major, ldc=N
            if (gr != 0) throw new InvalidOperationException($"cublasSgemm (tile {pixelStart}/{N}) failed: {gr}");
        }

        // ── Bias: output[oc, pixel] += bias[oc]  (full output, one kernel) ─
        nint bp0 = outputPtr, bp1 = biasPtr;
        int  bp2 = N, bp3 = outCh;
        nint* bargs = stackalloc nint[4] { (nint)(&bp0), (nint)(&bp1), (nint)(&bp2), (nint)(&bp3) };
        uint grBias = ((uint)N + 255) / 256;
        int br = NvrtcInterop.LaunchKernel(_biasAddKernel, grBias, 1, 1, 256, 1, 1, 0, _stream, bargs, null);
        if (br != 0) throw new InvalidOperationException($"bias_add launch failed: {br}");

        return output;
    }

    /// <inheritdoc/>
    public void LeakyReluInPlace(Tensor x, float negSlope)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint p0 = GetDevPtr(x);
        float p1 = negSlope;
        int   p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_leakyReluKernel, n, args);
    }

    /// <inheritdoc/>
    public void ScaleInPlace(Tensor x, float scale)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint  p0 = GetDevPtr(x);
        float p1 = scale;
        int   p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_scaleKernel, n, args);
    }

    /// <inheritdoc/>
    public void AddScaledInPlace(Tensor dst, Tensor src, float scale)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)dst.ElementCount;
        nint  p0 = GetDevPtr(dst);
        nint  p1 = GetDevPtr(src);
        float p2 = scale;
        int   p3 = n;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        Launch1D(_addScaledKernel, n, args);
    }

    /// <summary>
    /// Issue #129: per-row scalar multiply over a [rows × cols] buffer:
    /// <c>buf[i*cols + e] *= scales[i]</c>. The device <paramref name="scales"/> buffer
    /// holds one scalar per row. Bit-identical to calling
    /// <see cref="ScaleInPlace"/>(row_i, scales[i]) once per row — a single float
    /// multiply per element, rounded to float. Used to apply the per-token shared-expert
    /// sigmoid gate to the batched shared-expert down output in one launch.
    /// </summary>
    public void ScaleRowsInPlace(Tensor buf, Tensor scales, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if ((uint)rows > 65535)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, "ScaleRowsInPlace: rows must fit the CUDA gridDim.y limit (65535).");

        // 2D grid: x walks columns (256-wide blocks), y is the row — the kernel
        // recovers (i, e) from block/thread indices with no integer divide.
        nint p0 = GetDevPtr(buf);
        nint p1 = GetDevPtr(scales);
        int  p2 = rows, p3 = cols;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        uint gridX = (uint)((cols + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_scaleRowsKernel, gridX, (uint)rows, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(scale_rows) failed: {r}");
    }

    /// <summary>
    /// Issue #129: batched MoE top-k weighted scatter-reduce + shared-expert add, in one
    /// launch over all N tokens. For each (token i, element e):
    /// <c>acc = Σ_k downPartial[(i*na+k)*embDim+e] * weights[i*na+k]; acc += shared[i*embDim+e]; shared[i*embDim+e] = acc;</c>
    /// The per-k <c>acc += partial*weight</c> contracts to <c>fmaf</c> (NVRTC fmad=true),
    /// one rounding per term, exactly matching the sequential <c>AddScaledInPlace</c> loop;
    /// the final <c>acc += shared</c> is a plain add (one rounding), matching the sequential
    /// <c>AddInPlace</c>. Routed slots are summed in k=0..na-1 order, shared added last —
    /// byte-identical to the per-token <c>Clear + AddScaledInPlace×na + AddInPlace</c>.
    /// <paramref name="shared"/> is in/out (must already hold the scaled+rounded shared
    /// output per token); each thread owns its (i,e) element so the read-modify-write is
    /// race-free.
    /// </summary>
    public void MoeWeightedReduce(Tensor downPartial, Tensor weights, Tensor shared,
                                  int N, int na, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if ((uint)N > 65535)
            throw new ArgumentOutOfRangeException(nameof(N), N, "MoeWeightedReduce: N must fit the CUDA gridDim.y limit (65535).");

        // 2D grid: x walks embDim (256-wide blocks), y is the token — the kernel
        // recovers (i, e) from block/thread indices with no integer divide/modulo.
        nint p0 = GetDevPtr(downPartial);
        nint p1 = GetDevPtr(weights);
        nint p2 = GetDevPtr(shared);
        int  p3 = N, p4 = na, p5 = embDim;
        nint* args = stackalloc nint[6]
            { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5) };
        uint gridX = (uint)((embDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_moeWeightedReduceKernel, gridX, (uint)N, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(moe_weighted_reduce) failed: {r}");
    }

    /// <inheritdoc/>
    public void ClampInPlace(Tensor x, float min, float max)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint  p0 = GetDevPtr(x);
        float p1 = min, p2 = max;
        int   p3 = n;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        Launch1D(_clampKernel, n, args);
    }

    /// <inheritdoc/>
    public Tensor CatChannels(Tensor a, int aCh, Tensor b, int bCh, int hw)
    {
        var output = Allocate(TensorShape.D1((long)(aCh + bCh) * hw));
        nint outPtr = GetDevPtr(output);
        nint aPtr   = GetDevPtr(a);
        nint bPtr   = GetDevPtr(b);
        nuint aBytes = (nuint)(aCh * hw * sizeof(float));
        nuint bBytes = (nuint)(bCh * hw * sizeof(float));
        // Two async DMA copies on the same stream — no kernel dispatch overhead.
        CuBlasInterop.CudaMemcpyAsync(outPtr,           aPtr, aBytes, CuBlasInterop.DeviceToDevice, _stream);
        CuBlasInterop.CudaMemcpyAsync(outPtr + (nint)aBytes, bPtr, bBytes, CuBlasInterop.DeviceToDevice, _stream);
        return output;
    }

    /// <inheritdoc/>
    public Tensor PixelShuffleGpu(Tensor input, int inCh, int h, int w, int upscaleFactor)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int outCh = inCh / (upscaleFactor * upscaleFactor);
        var output = Allocate(TensorShape.D1((long)outCh * h * upscaleFactor * w * upscaleFactor));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        int  p2 = outCh, p3 = h, p4 = w, p5 = upscaleFactor;
        nint* args = stackalloc nint[6]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5)
        };
        Launch1D(_pshuffleKernel, outCh * h * upscaleFactor * w * upscaleFactor, args);
        return output;
    }

    /// <inheritdoc/>
    public Tensor PixelUnshuffleGpu(Tensor input, int inCh, int h, int w, int downscaleFactor)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int outCh = inCh * downscaleFactor * downscaleFactor;
        int outH  = h / downscaleFactor;
        int outW  = w / downscaleFactor;
        var output = Allocate(TensorShape.D1((long)outCh * outH * outW));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        // kernel signature: (input, output, inCh, outH, outW, r)
        int  p2 = inCh, p3 = outH, p4 = outW, p5 = downscaleFactor;
        nint* args = stackalloc nint[6]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5)
        };
        Launch1D(_punshuffleKernel, outCh * outH * outW, args);
        return output;
    }

    /// <inheritdoc/>
    public Tensor Upsample2xGpu(Tensor input, int ch, int h, int w)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        var output = Allocate(TensorShape.D1((long)ch * h * 2 * w * 2));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        int  p2 = ch, p3 = h, p4 = w;
        nint* args = stackalloc nint[5]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4)
        };
        Launch1D(_upsample2xKernel, ch * h * 2 * w * 2, args);
        return output;
    }

    /// <inheritdoc/>
    /// <remarks>No-op: CUDA streams are already asynchronous.</remarks>
    public void BeginBatch() { }

    /// <inheritdoc/>
    /// <remarks>No-op: CUDA kernels on the same stream execute in order.</remarks>
    public void BatchBarrier() { }

    /// <inheritdoc/>
    /// <remarks>
    /// No-op for CUDA: all kernels are queued on <c>_stream</c> and execute in order,
    /// so no explicit submission or synchronisation is needed between RDB blocks.
    /// The stream is synchronised exactly once at <see cref="Download"/> time.
    /// </remarks>
    public void EndBatch() { }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _devPtrs.Values)
            CuBlasInterop.CudaFree(entry.devPtr);
        _devPtrs.Clear();

        _pool.Dispose();

        if (_nvModule != nint.Zero)
        {
            NvrtcInterop.ModuleUnload(_nvModule);
            _nvModule = nint.Zero;
        }

        if (_im2colBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_im2colBuf);
            _im2colBuf = nint.Zero;
        }
        if (_q81Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81Buf);
            _q81Buf = nint.Zero;
            _q81BufSize = 0;
        }
        if (_q81BufB != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BufB);
            _q81BufB = nint.Zero;
            _q81BufBSize = 0;
        }
        if (_q81BatchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchBuf);
            _q81BatchBuf = nint.Zero;
            _q81BatchBufSize = 0;
        }
        if (_waveScratchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_waveScratchBuf);
            _waveScratchBuf = nint.Zero;
            _waveScratchBufSize = 0;
        }

        CuBlasInterop.Destroy(_handle);

        if (_stream != nint.Zero)
            CuBlasInterop.StreamDestroy(_stream);

        if (_uploadStream != nint.Zero)
        {
            // Drain any straggling background DMA so we don't tear down the stream
            // out from under a still-in-flight transfer.
            CuBlasInterop.StreamSynchronize(_uploadStream);
            CuBlasInterop.StreamDestroy(_uploadStream);
            _uploadStream = nint.Zero;
        }

        if (_asyncPinnedBuf != nint.Zero)
        {
            CuBlasInterop.FreeHost(_asyncPinnedBuf);
            _asyncPinnedBuf = nint.Zero;
            _asyncPinnedBufSize = 0;
        }

        if (_pinnedBuf != nint.Zero)
            CuBlasInterop.FreeHost(_pinnedBuf);
    }
}

/// <summary>
/// Pool of reusable CUDA device buffers keyed by rounded allocation size.
/// Eliminates the cudaMalloc/cudaFree overhead on the hot path (one pair per GEMM call).
/// Sizes are rounded up to the next power-of-two so all Allocate/Upload callers must use
/// RoundUp() when deciding how many bytes to cudaMalloc — this guarantees a pooled pointer
/// is always large enough for any request that maps to the same bucket.
/// Thread-safe via per-bucket ConcurrentStack.
/// </summary>
internal sealed class GpuBufferPool : IDisposable
{
    // One stack of available device pointers per power-of-two bucket.
    private readonly ConcurrentDictionary<nuint, ConcurrentStack<nint>> _buckets = new();
    private bool _disposed;

    /// <summary>Round <paramref name="v"/> up to the next power-of-two (minimum 64 bytes).</summary>
    public static nuint RoundUp(nuint v)
    {
        if (v <= 64) return 64;
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }

    /// <summary>
    /// Return a cached device pointer for a bucket of exactly <paramref name="bucketSize"/> bytes
    /// (must be a power-of-two, i.e. the result of <see cref="RoundUp"/>), or Zero if none available.
    /// </summary>
    public nint Rent(nuint bucketSize)
    {
        if (_buckets.TryGetValue(bucketSize, out var stack) && stack.TryPop(out nint ptr))
            return ptr;
        return nint.Zero;
    }

    /// <summary>
    /// Return a device pointer to the pool. <paramref name="bucketSize"/> must be the
    /// power-of-two size originally passed to <see cref="Rent"/> (or stored in _devPtrs).
    /// </summary>
    public void Return(nuint bucketSize, nint devPtr)
    {
        if (devPtr == nint.Zero || _disposed) { CuBlasInterop.CudaFree(devPtr); return; }
        _buckets.GetOrAdd(bucketSize, _ => new ConcurrentStack<nint>()).Push(devPtr);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var stack in _buckets.Values)
            while (stack.TryPop(out nint ptr))
                CuBlasInterop.CudaFree(ptr);
        _buckets.Clear();
    }
}
