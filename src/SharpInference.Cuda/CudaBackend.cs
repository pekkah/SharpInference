using System.Collections.Concurrent;
using System.Threading;
using SharpInference.Core;

namespace SharpInference.Cuda;

/// <summary>
/// CUDA/cuBLAS compute backend for DiT SGEMM acceleration.
/// Manages CUDA device memory and dispatches cuBLAS GemmEx kernels.
/// Precision is auto-detected at creation time:
///   sm_80+ (Ampere/RTX 30xx) → bf16 inputs, fp32 accumulation (no overflow, 2× smaller than fp32)
///   sm_53+ (Pascal/any CUDA GPU) → fp16 inputs, fp32 accumulation (avoids fp16 accum overflow)
///   fallback → fp32
/// All LLM transformer operations throw NotSupportedException; this backend is DiT-only.
/// </summary>
public sealed unsafe class CudaBackend : IComputeBackend, IDisposable
{
    private readonly nint _handle;
    private readonly SgemmPrecision _precision;
    private readonly ConcurrentDictionary<nint, nint> _devPtrs = new();
    private long _nextHandle = 1;
    private bool _disposed;

    public string Name => $"CUDA GPU (cuBLAS, {_precision})";

    public SgemmPrecision BestSgemmPrecision => _precision;

    public bool SupportsGpuDequant => false;

    private CudaBackend(nint handle, SgemmPrecision precision)
    {
        _handle    = handle;
        _precision = precision;
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
    public static CudaBackend Create()
    {
        int status = CuBlasInterop.Create(out nint handle);
        if (status != 0)
            throw new InvalidOperationException($"cublasCreate failed: {status}");

        var precision = DetectBestPrecision();
        return new CudaBackend(handle, precision);
    }

    private static SgemmPrecision DetectBestPrecision()
    {
        // Query compute capability of device 0
        if (CuBlasInterop.DeviceGetAttribute(out int major, CuBlasInterop.CudaDevAttrComputeCapabilityMajor, 0) == 0 &&
            CuBlasInterop.DeviceGetAttribute(out int minor, CuBlasInterop.CudaDevAttrComputeCapabilityMinor, 0) == 0)
        {
            int sm = major * 10 + minor;
            if (sm >= 89) return SgemmPrecision.Fp8E4M3; // Ada Lovelace+ (RTX 40xx) has fp8 tensor cores
            if (sm >= 80) return SgemmPrecision.Bf16;    // Ampere+ has native bf16
            if (sm >= 53) return SgemmPrecision.Fp16;    // Pascal+ supports fp16 GemmEx
        }
        return SgemmPrecision.Fp32;
    }

    // ── Memory management ─────────────────────────────────────────────────

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        nuint byteSize = (nuint)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed: {status}");
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_devPtrs.TryRemove(tensor.Handle, out nint devPtr))
            CuBlasInterop.CudaFree(devPtr);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        nuint byteSize = (nuint)(data.Length * sizeof(float));
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed: {status}");
        fixed (float* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, DType.Float32, handle);
    }

    public void Download(Tensor src, Span<float> dst)
    {
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)(dst.Length * sizeof(float));
        fixed (float* d = dst)
        {
            int status = CuBlasInterop.CudaMemcpy((nint)d, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy D2H failed: {status}");
        }
    }

    public Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape)
    {
        nuint byteSize = (nuint)(data.Length * 2);
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed: {status}");
        fixed (Half* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D (fp16) failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, DType.Float16, handle);
    }

    public void DownloadHalf(Tensor src, Span<Half> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (Half* d = dst)
        {
            int status = CuBlasInterop.CudaMemcpy((nint)d, devPtr, (nuint)(dst.Length * 2), CuBlasInterop.DeviceToHost);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy D2H (fp16) failed: {status}");
        }
    }

    public Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape)
    {
        nuint byteSize = (nuint)(data.Length * 2);
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed: {status}");
        fixed (ushort* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D (bf16) failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, DType.BFloat16, handle);
    }

    public void DownloadBf16(Tensor src, Span<ushort> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (ushort* d = dst)
        {
            int status = CuBlasInterop.CudaMemcpy((nint)d, devPtr, (nuint)(dst.Length * 2), CuBlasInterop.DeviceToHost);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy D2H (bf16) failed: {status}");
        }
    }

    public Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape)
    {
        nuint byteSize = (nuint)data.Length;
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed: {status}");
        fixed (byte* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D (fp8) failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, DType.Float8E4M3, handle);
    }

    public void DownloadFp8(Tensor src, Span<byte> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (byte* d = dst)
        {
            int status = CuBlasInterop.CudaMemcpy((nint)d, devPtr, (nuint)dst.Length, CuBlasInterop.DeviceToHost);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy D2H (fp8) failed: {status}");
        }
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
            // fp8 E4M3 requires both A and B to be fp8 (mixed fp8+bf16 needs cublasLt).
            // fp16/bf16 use fp32 accumulation to avoid overflow on large DiT residuals.
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
        int status = CuBlasInterop.DeviceSync();
        if (status != 0)
            throw new InvalidOperationException($"cudaDeviceSynchronize failed: {status}");
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
        _devPtrs.TryGetValue(tensor.Handle, out nint ptr)
            ? ptr
            : throw new InvalidOperationException($"Tensor handle {tensor.Handle} not registered in CudaBackend");

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var devPtr in _devPtrs.Values)
            CuBlasInterop.CudaFree(devPtr);
        _devPtrs.Clear();

        CuBlasInterop.Destroy(_handle);
    }
}
