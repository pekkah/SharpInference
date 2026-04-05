using System.Buffers;
using System.Collections.Concurrent;
using System.Threading;
using SharpInference.Core;

namespace SharpInference.Cuda;

/// <summary>
/// CUDA/cuBLAS compute backend for DiT SGEMM acceleration.
/// Manages CUDA device memory and dispatches cuBLAS GEMM kernels.
/// All LLM transformer operations (RmsNorm, RoPE, Attention etc.) are
/// not implemented — this backend is exclusively for the DiT SGEMM path.
/// </summary>
public sealed unsafe class CudaBackend : IComputeBackend, IDisposable
{
    private readonly nint _handle;
    private readonly ConcurrentDictionary<nint, nint> _devPtrs = new();
    private long _nextHandle = 1;
    private bool _disposed;

    public string Name => "CUDA GPU (cuBLAS)";

    public SgemmPrecision BestSgemmPrecision => SgemmPrecision.Fp16;

    public bool SupportsGpuDequant => false;

    private CudaBackend(nint handle) => _handle = handle;

    /// <summary>
    /// Returns true if a CUDA device and cuBLAS are available on this system.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            int status = CuBlasInterop.Create(out nint h);
            if (status == 0)
            {
                CuBlasInterop.Destroy(h);
                return true;
            }
            return false;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>Create a new CudaBackend. Throws if cuBLAS is unavailable.</summary>
    public static CudaBackend Create()
    {
        int status = CuBlasInterop.Create(out nint handle);
        if (status != 0)
            throw new InvalidOperationException($"cublasCreate failed with status {status}");
        return new CudaBackend(handle);
    }

    // ── Memory management ─────────────────────────────────────────────────

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        nuint byteSize = (nuint)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed with status {status}");

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
            throw new InvalidOperationException($"cudaMalloc failed with status {status}");

        fixed (float* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D failed with status {status}");
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
                throw new InvalidOperationException($"cudaMemcpy D2H failed with status {status}");
        }
    }

    public Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape)
    {
        nuint byteSize = (nuint)(data.Length * sizeof(ushort));
        int status = CuBlasInterop.CudaMalloc(out nint devPtr, byteSize);
        if (status != 0)
            throw new InvalidOperationException($"cudaMalloc failed with status {status}");

        fixed (Half* src = data)
        {
            status = CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy H2D (fp16) failed with status {status}");
        }

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = devPtr;
        return new Tensor(shape, DType.Float16, handle);
    }

    public void DownloadHalf(Tensor src, Span<Half> dst)
    {
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)(dst.Length * sizeof(ushort));
        fixed (Half* d = dst)
        {
            int status = CuBlasInterop.CudaMemcpy((nint)d, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (status != 0)
                throw new InvalidOperationException($"cudaMemcpy D2H (fp16) failed with status {status}");
        }
    }

    public Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape) =>
        throw new NotSupportedException("CudaBackend does not support bf16");

    public void DownloadBf16(Tensor src, Span<ushort> dst) =>
        throw new NotSupportedException("CudaBackend does not support bf16");

    public Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("CudaBackend does not support raw quantized upload");

    public void DequantQ5KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    public void DequantQ4KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    // ── SGEMM ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GEMM: C[M,N] = A[M,K] × B[N,K]^T using cuBLAS.
    /// Uses Hgemm for fp16 tensors, Sgemm for fp32.
    /// Row-major inputs are handled via the column-major transpose trick.
    /// </summary>
    public void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N)
    {
        nint aPtr = GetDevPtr(A);
        nint bPtr = GetDevPtr(B);
        nint cPtr = GetDevPtr(C);

        if (A.DType == DType.Float16 && B.DType == DType.Float16)
        {
            // fp16: cublasHgemm
            // Row-major C[M,N] = A[M,K] * B[N,K]^T
            // → cuBLAS col-major: swap A/B, use Op_T for B, Op_N for A
            ushort alpha = CuBlasInterop.FP16One;
            ushort beta  = CuBlasInterop.FP16Zero;
            int status = CuBlasInterop.Hgemm(
                _handle,
                CuBlasInterop.OpT,  // transa: transpose B (B is N×K row-major → Op_T)
                CuBlasInterop.OpN,  // transb: no transpose A (A is M×K row-major → Op_N)
                N, M, K,            // m=N, n=M (swapped for col-major)
                ref alpha,
                bPtr, K,            // A_cublas = B_ptr, lda = K
                aPtr, K,            // B_cublas = A_ptr, ldb = K
                ref beta,
                cPtr, N);           // C_cublas = C_ptr, ldc = N
            if (status != 0)
                throw new InvalidOperationException($"cublasHgemm failed with status {status}");
        }
        else
        {
            // fp32: cublasSgemm
            float alpha = 1.0f;
            float beta  = 0.0f;
            int status = CuBlasInterop.Sgemm(
                _handle,
                CuBlasInterop.OpT,
                CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                bPtr, K,
                aPtr, K,
                ref beta,
                cPtr, N);
            if (status != 0)
                throw new InvalidOperationException($"cublasSgemm failed with status {status}");
        }
    }

    public void Synchronize()
    {
        int status = CuBlasInterop.DeviceSync();
        if (status != 0)
            throw new InvalidOperationException($"cudaDeviceSynchronize failed with status {status}");
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
