using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>
/// P/Invoke bindings for cuBLAS and CUDA runtime.
/// Uses source-generated LibraryImport for NativeAOT compatibility.
/// </summary>
internal static partial class CuBlasInterop
{
    static CuBlasInterop()
    {
        // CUDA 11.7+ defaults to lazy module loading: SASS for each kernel is JIT'd from
        // PTX on the first cuLaunchKernel, not at cuModuleLoadData time. For an LLM that
        // launches ~700 kernels per token, the cold-cache JIT cost lands in the middle
        // of the first prefill and craters throughput. Forcing EAGER moves the JIT cost
        // back to module-load time, where it shows up as a one-shot startup delay.
        // Must be set before any CUDA call: the driver reads it during cuInit.
        if (Environment.GetEnvironmentVariable("CUDA_MODULE_LOADING") is null)
            Environment.SetEnvironmentVariable("CUDA_MODULE_LOADING", "EAGER");
    }

    // ── cuBLAS handle lifecycle ──────────────────────────────────────────

    [LibraryImport("cublas64_11", EntryPoint = "cublasCreate_v2")]
    internal static partial int Create(out nint handle);

    [LibraryImport("cublas64_11", EntryPoint = "cublasDestroy_v2")]
    internal static partial int Destroy(nint handle);

    [LibraryImport("cublas64_11", EntryPoint = "cublasSetStream_v2")]
    internal static partial int SetStream(nint handle, nint stream);

    // CUBLAS_MATH_ALLOW_REDUCED_PRECISION_REDUCTION = 0 means standard FP32.
    // CUBLAS_TF32_TENSOR_OP_MATH = 3 enables TF32 tensor cores in Sgemm transparently.
    public const int CUBLAS_DEFAULT_MATH = 0;
    public const int CUBLAS_TF32_TENSOR_OP_MATH = 3;

    [LibraryImport("cublas64_11", EntryPoint = "cublasSetMathMode")]
    internal static partial int SetMathMode(nint handle, int mode);

    // ── cublasSgemm: C = alpha*op(A)*op(B) + beta*C (fp32) ──────────────
    [LibraryImport("cublas64_11", EntryPoint = "cublasSgemm_v2")]
    internal static partial int Sgemm(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        ref float alpha,
        nint A, int lda,
        nint B, int ldb,
        ref float beta,
        nint C, int ldc);

    // ── cublasGemmEx: mixed-precision GEMM (fp16/bf16 in, fp32 accum) ────
    // alpha and beta are float* when computeType == CUBLAS_COMPUTE_32F.
    [LibraryImport("cublas64_11", EntryPoint = "cublasGemmEx")]
    internal static partial int GemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        ref float alpha,
        nint A, int Atype, int lda,
        nint B, int Btype, int ldb,
        ref float beta,
        nint C, int Ctype, int ldc,
        int computeType,
        int algo);

    // ── CUDA memory management ────────────────────────────────────────────

    [LibraryImport("cudart64_110", EntryPoint = "cudaMalloc")]
    internal static partial int CudaMalloc(out nint devPtr, nuint size);

    [LibraryImport("cudart64_110", EntryPoint = "cudaFree")]
    internal static partial int CudaFree(nint devPtr);

    [LibraryImport("cudart64_110", EntryPoint = "cudaMemcpy")]
    internal static partial int CudaMemcpy(nint dst, nint src, nuint count, int kind);

    [LibraryImport("cudart64_110", EntryPoint = "cudaMemcpyAsync")]
    internal static partial int CudaMemcpyAsync(nint dst, nint src, nuint count, int kind, nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaMemcpy2DAsync")]
    internal static partial int CudaMemcpy2DAsync(
        nint dst, nuint dpitch,
        nint src, nuint spitch,
        nuint width, nuint height,
        int kind, nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaDeviceSynchronize")]
    internal static partial int DeviceSync();

    [LibraryImport("cudart64_110", EntryPoint = "cudaStreamCreate")]
    internal static partial int StreamCreate(out nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaStreamDestroy")]
    internal static partial int StreamDestroy(nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaStreamSynchronize")]
    internal static partial int StreamSynchronize(nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaRuntimeGetVersion")]
    internal static partial int RuntimeGetVersion(out int version);

    [LibraryImport("cudart64_110", EntryPoint = "cudaDeviceGetAttribute")]
    internal static partial int DeviceGetAttribute(out int value, int attr, int device);

    /// <summary>cudaMemGetInfo: free and total VRAM on the current device, in bytes.</summary>
    [LibraryImport("cudart64_110", EntryPoint = "cudaMemGetInfo")]
    internal static partial int MemGetInfo(out nuint free, out nuint total);

    [LibraryImport("cudart64_110", EntryPoint = "cudaEventCreate")]
    internal static partial int EventCreate(out nint ev);

    [LibraryImport("cudart64_110", EntryPoint = "cudaEventRecord")]
    internal static partial int EventRecord(nint ev, nint stream);

    [LibraryImport("cudart64_110", EntryPoint = "cudaEventSynchronize")]
    internal static partial int EventSynchronize(nint ev);

    [LibraryImport("cudart64_110", EntryPoint = "cudaEventElapsedTime")]
    internal static partial int EventElapsedTime(out float ms, nint start, nint stop);

    [LibraryImport("cudart64_110", EntryPoint = "cudaEventDestroy")]
    internal static partial int EventDestroy(nint ev);

    // ── Pinned host memory (enables DMA-based async transfers) ────────────

    [LibraryImport("cudart64_110", EntryPoint = "cudaMallocHost")]
    internal static partial int MallocHost(out nint ptr, nuint size);

    [LibraryImport("cudart64_110", EntryPoint = "cudaFreeHost")]
    internal static partial int FreeHost(nint ptr);

    // ── Constants ─────────────────────────────────────────────────────────

    internal const int HostToDevice   = 1;
    internal const int DeviceToHost   = 2;
    internal const int DeviceToDevice = 3;

    internal const int OpN = 0;
    internal const int OpT = 1;

    // cudaDataType values
    internal const int CUDA_R_32F      = 0;   // float
    internal const int CUDA_R_16F      = 2;   // half (fp16)
    internal const int CUDA_R_16BF     = 14;  // bfloat16
    internal const int CUDA_R_8F_E4M3  = 28;  // fp8 E4M3FN (sm_89+, CUDA 11.8+)

    // cublasComputeType_t — values from cublas_api.h (CUDA 11.8+)
    internal const int CUBLAS_COMPUTE_32F          = 68;  // fp32 accumulator (no tensor cores)
    internal const int CUBLAS_COMPUTE_32F_FAST_16F = 74;  // fast fp32 via fp16 down-convert
    internal const int CUBLAS_COMPUTE_32F_FAST_TF32 = 77; // fast fp32 via TF32 (sm_80+)

    // cublasGemmAlgo_t
    internal const int CUBLAS_GEMM_DEFAULT = -1;

    // cudaDeviceAttr: compute capability
    internal const int CudaDevAttrComputeCapabilityMajor = 75;
    internal const int CudaDevAttrComputeCapabilityMinor = 76;
}
