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
    private readonly nint _handle;
    private readonly SgemmPrecision _precision;
    private readonly int _smVersion;
    private readonly nint _stream;
    private readonly ConcurrentDictionary<nint, (nint devPtr, nuint byteSize)> _devPtrs = new();
    private long _nextHandle = 1;

    // Pinned host staging buffer for DMA-capable async H2D/D2H transfers.
    // Grows on demand; never shrinks (amortised over the pipeline lifetime).
    private nint   _pinnedBuf;
    private nuint  _pinnedBufSize;
    private const nuint InitialPinnedSize = 32 * 1024 * 1024; // 32 MB

    // Maximum im2col tile buffer size. All row-aligned tile sizes fit within this bound.
    private const long MaxTileBytes = 2560L * 1024 * 1024; // 2.5 GiB — fits all layers in a single tile

    // GPU buffer pool: reuse device allocations by rounded size to avoid cudaMalloc overhead.
    // Each MatQ call (GEMM) does 2 alloc+free cycles; pooling eliminates driver round-trips.
    private readonly GpuBufferPool _pool = new();

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

    public string Name => $"CUDA GPU (cuBLAS, {_precision})";

    public SgemmPrecision BestSgemmPrecision => _precision;

    public bool SupportsGpuDequant => false;

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

        // Pre-allocate im2col buffer now (2.5 GiB) so the first Conv2d call doesn't trigger
        // a blocking cudaMalloc that would stall GPU work enqueued before it.
        backend.EnsureIm2ColBuf(MaxTileBytes);

        return backend;
    }

    private static SgemmPrecision DetectBestPrecision(int sm)
    {
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

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        nuint byteSize  = (nuint)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_devPtrs.TryRemove(tensor.Handle, out var entry))
            _pool.Return(entry.byteSize, entry.devPtr);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)(data.Length * sizeof(float));
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
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
        return new Tensor(shape, DType.Float32, handle);
    }

    public void Download(Tensor src, Span<float> dst)
    {
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)(dst.Length * sizeof(float));
        fixed (float* d = dst)
            DownloadViaStaging(d, devPtr, byteSize);
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

    public Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("CudaBackend does not support raw quantized upload (GPU dequant not implemented)");

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

    /// <summary>
    /// Copy <paramref name="src"/> to the device pointer via the pinned staging buffer
    /// using a synchronous cudaMemcpy.  Pinned memory avoids the runtime's internal
    /// pageable→pinned staging copy, so DMA starts immediately.
    /// The shared staging buffer is safe here because the copy is synchronous — the
    /// buffer is fully consumed before returning, so the next upload can reuse it.
    /// </summary>
    private unsafe void UploadViaStaging(nint devPtr, void* src, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero)
        {
            Buffer.MemoryCopy(src, (void*)_pinnedBuf, _pinnedBufSize, byteSize);
            CuBlasInterop.CudaMemcpy(devPtr, _pinnedBuf, byteSize, CuBlasInterop.HostToDevice);
        }
        else
        {
            CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
        }
    }

    /// <summary>
    /// Copy from device to <paramref name="dst"/> via the pinned staging buffer (async DMA).
    /// Caller must call <see cref="Synchronize"/> before reading <paramref name="dst"/>.
    /// </summary>
    private unsafe void DownloadViaStaging(void* dst, nint devPtr, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            CuBlasInterop.CudaMemcpyAsync(_pinnedBuf, devPtr, byteSize,
                                          CuBlasInterop.DeviceToHost, _stream);
            CuBlasInterop.StreamSynchronize(_stream);
            Buffer.MemoryCopy((void*)_pinnedBuf, dst, byteSize, byteSize);
        }
        else
        {
            CuBlasInterop.CudaMemcpy((nint)dst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
        }
    }

    // ── Unsupported LLM ops ───────────────────────────────────────────────

    public void MatMul(Tensor output, Tensor matrix, Tensor vector) =>
        throw new NotSupportedException("CudaBackend is DiT-only; use VulkanBackend for full LLM inference");

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

    public void ElementwiseMul(Tensor output, Tensor a, Tensor b) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

    public void Softmax(Tensor x) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

    public void SiLU(Tensor x) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

    public void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                                 int nTok, int nHeads, int headDim, float scale) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

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

    private void CompileAndLoadKernels()
    {
        // Ensure the CUDA Driver API context exists (shares the primary context with the runtime).
        NvrtcInterop.CuInit(0);

        // Try to load from cubin cache first (avoids both NVRTC compilation and PTX JIT overhead).
        string cacheFile = GetCubinCachePath();
        if (TryLoadCubinFromCache(cacheFile)) return;

        byte[] srcBytes  = NvrtcInterop.ToUtf8(CudaKernels.Source);
        byte[] nameBytes = NvrtcInterop.ToUtf8("sharpi_image_ops.cu");

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
            try
            {
                if (NvrtcInterop.GetCubinSize(prog, out nuint cubinSize) == 0 && cubinSize > 0)
                {
                    binary = new byte[(int)cubinSize];
                    fixed (byte* pBin = binary)
                    {
                        int r2 = NvrtcInterop.GetCubin(prog, pBin);
                        if (r2 != 0) binary = null;
                    }
                }
            }
            catch { binary = null; }  // nvrtcGetCubin not available on this NVRTC version

            if (binary is null)
            {
                // Fall back to PTX (triggers JIT at first kernel launch, slower).
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
        }
        finally
        {
            NvrtcInterop.DestroyProgram(ref prog);
        }

        // Persist cubin to disk so future runs skip both NVRTC compilation and JIT.
        if (binary is not null)
        {
            try { File.WriteAllBytes(cacheFile, binary); }
            catch { /* ignore cache write failures */ }
        }

        LoadKernelFunctions();
    }

    private string GetCubinCachePath()
    {
        // Cache key: SHA-256 of kernel source + SM version.
        // Any source change or GPU change invalidates the cache.
        byte[] hash = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(CudaKernels.Source + _smVersion));
        string hex = Convert.ToHexString(hash)[..16];
        return Path.Combine(Path.GetTempPath(), $"sharpi_cubin_sm{_smVersion}_{hex}.bin");
    }

    private bool TryLoadCubinFromCache(string cacheFile)
    {
        if (!File.Exists(cacheFile)) return false;
        try
        {
            byte[] cubinBuf = File.ReadAllBytes(cacheFile);
            fixed (byte* pBin = cubinBuf)
            {
                int mr = NvrtcInterop.ModuleLoadData(out _nvModule, pBin);
                if (mr != 0) return false;
            }
            LoadKernelFunctions();
            return true;
        }
        catch
        {
            return false;
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

        CuBlasInterop.Destroy(_handle);

        if (_stream != nint.Zero)
            CuBlasInterop.StreamDestroy(_stream);

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
