using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>
/// P/Invoke bindings for cuBLAS and CUDA runtime.
/// Uses source-generated LibraryImport for NativeAOT compatibility.
/// </summary>
internal static partial class CuBlasInterop
{
    // ── cuBLAS handle lifecycle ──────────────────────────────────────────

    [LibraryImport("cublas64_11", EntryPoint = "cublasCreate_v2")]
    internal static partial int Create(out nint handle);

    [LibraryImport("cublas64_11", EntryPoint = "cublasDestroy_v2")]
    internal static partial int Destroy(nint handle);

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

    [LibraryImport("cudart64_110", EntryPoint = "cudaDeviceSynchronize")]
    internal static partial int DeviceSync();

    [LibraryImport("cudart64_110", EntryPoint = "cudaRuntimeGetVersion")]
    internal static partial int RuntimeGetVersion(out int version);

    [LibraryImport("cudart64_110", EntryPoint = "cudaDeviceGetAttribute")]
    internal static partial int DeviceGetAttribute(out int value, int attr, int device);

    // ── Constants ─────────────────────────────────────────────────────────

    internal const int HostToDevice = 1;
    internal const int DeviceToHost = 2;

    internal const int OpN = 0;
    internal const int OpT = 1;

    // cudaDataType values
    internal const int CUDA_R_32F      = 0;   // float
    internal const int CUDA_R_16F      = 2;   // half (fp16)
    internal const int CUDA_R_16BF     = 14;  // bfloat16
    internal const int CUDA_R_8F_E4M3  = 28;  // fp8 E4M3FN (sm_89+, CUDA 11.8+)

    // cublasComputeType_t
    internal const int CUBLAS_COMPUTE_32F = 68;  // fp32 accumulator (no tensor cores)
    internal const int CUBLAS_COMPUTE_32F_FAST_TF32 = 74; // tf32 tensor cores

    // cublasGemmAlgo_t
    internal const int CUBLAS_GEMM_DEFAULT = -1;

    // cudaDeviceAttr: compute capability
    internal const int CudaDevAttrComputeCapabilityMajor = 75;
    internal const int CudaDevAttrComputeCapabilityMinor = 76;
}
