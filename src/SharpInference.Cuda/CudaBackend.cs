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
public sealed unsafe class CudaBackend : IComputeBackend, IDisposable
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

    // GPU buffer pool: reuse device allocations by rounded size to avoid cudaMalloc overhead.
    // Each MatQ call (GEMM) does 2 alloc+free cycles; pooling eliminates driver round-trips.
    private readonly GpuBufferPool _pool = new();

    private bool _disposed;

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

        // Pinned (page-locked) staging buffer for DMA-capable async H2D/D2H transfers.
        CuBlasInterop.MallocHost(out nint pinnedBuf, InitialPinnedSize);

        var resolvedPrecision = precision ?? DetectBestPrecision(smVersion);
        return new CudaBackend(handle, resolvedPrecision, smVersion, stream, pinnedBuf, InitialPinnedSize);
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
        nuint byteSize = (nuint)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        nint devPtr = _pool.Rent(byteSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, byteSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, byteSize);
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_devPtrs.TryRemove(tensor.Handle, out var entry))
            _pool.Return(entry.byteSize, entry.devPtr);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        nuint byteSize = (nuint)(data.Length * sizeof(float));
        nint devPtr = _pool.Rent(byteSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, byteSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (float* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, byteSize);
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
        nuint byteSize = (nuint)(data.Length * 2);
        nint devPtr = _pool.Rent(byteSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, byteSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (Half* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, byteSize);
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
        nuint byteSize = (nuint)(data.Length * 2);
        nint devPtr = _pool.Rent(byteSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, byteSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (ushort* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, byteSize);
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
        nuint byteSize = (nuint)data.Length;
        nint devPtr = _pool.Rent(byteSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, byteSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (byte* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, byteSize);
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
    /// Copy <paramref name="src"/> to the device pointer via the pinned staging buffer,
    /// using async DMA when possible. Falls back to synchronous copy if pinned alloc failed.
    /// </summary>
    private unsafe void UploadViaStaging(nint devPtr, void* src, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            Buffer.MemoryCopy(src, (void*)_pinnedBuf, _pinnedBufSize, byteSize);
            CuBlasInterop.CudaMemcpyAsync(devPtr, _pinnedBuf, byteSize,
                                          CuBlasInterop.HostToDevice, _stream);
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

    public void AddInPlace(Tensor dst, Tensor src) =>
        throw new NotSupportedException("CudaBackend is DiT-only");

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

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _devPtrs.Values)
            CuBlasInterop.CudaFree(entry.devPtr);
        _devPtrs.Clear();

        _pool.Dispose();

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
/// Sizes are rounded up to the next power-of-two to maximise reuse across different shapes.
/// Thread-safe via per-bucket ConcurrentStack.
/// </summary>
internal sealed class GpuBufferPool : IDisposable
{
    // One stack of available device pointers per power-of-two bucket.
    private readonly ConcurrentDictionary<nuint, ConcurrentStack<nint>> _buckets = new();
    private bool _disposed;

    /// <summary>Return a device pointer of at least <paramref name="byteSize"/> bytes, or Zero if none available.</summary>
    public nint Rent(nuint byteSize)
    {
        nuint bucket = NextPow2(byteSize);
        if (_buckets.TryGetValue(bucket, out var stack) && stack.TryPop(out nint ptr))
            return ptr;
        return nint.Zero;
    }

    /// <summary>Return a device pointer to the pool. <paramref name="exactSize"/> must be the original allocation size.</summary>
    public void Return(nuint exactSize, nint devPtr)
    {
        if (devPtr == nint.Zero || _disposed) { CuBlasInterop.CudaFree(devPtr); return; }
        nuint bucket = NextPow2(exactSize);
        _buckets.GetOrAdd(bucket, _ => new ConcurrentStack<nint>()).Push(devPtr);
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

    private static nuint NextPow2(nuint v)
    {
        if (v == 0) return 1;
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }
}
