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
    // cuBLAS uses column-major convention; callers must transpose appropriately.
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

    // ── cublasHgemm: fp16 GEMM ────────────────────────────────────────────
    // alpha and beta are __half* (16-bit float) passed as ref ushort.
    [LibraryImport("cublas64_11", EntryPoint = "cublasHgemm")]
    internal static partial int Hgemm(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        ref ushort alpha,
        nint A, int lda,
        nint B, int ldb,
        ref ushort beta,
        nint C, int ldc);

    // ── CUDA memory management ────────────────────────────────────────────

    [LibraryImport("cudart64_110", EntryPoint = "cudaMalloc")]
    internal static partial int CudaMalloc(out nint devPtr, nuint size);

    [LibraryImport("cudart64_110", EntryPoint = "cudaFree")]
    internal static partial int CudaFree(nint devPtr);

    [LibraryImport("cudart64_110", EntryPoint = "cudaMemcpy")]
    internal static partial int CudaMemcpy(nint dst, nint src, nuint count, int kind);

    [LibraryImport("cudart64_110", EntryPoint = "cudaDeviceSynchronize")]
    internal static partial int DeviceSync();

    // ── Constants ─────────────────────────────────────────────────────────

    /// <summary>cudaMemcpyHostToDevice</summary>
    internal const int HostToDevice = 1;
    /// <summary>cudaMemcpyDeviceToHost</summary>
    internal const int DeviceToHost = 2;

    /// <summary>CUBLAS_OP_N — no transpose</summary>
    internal const int OpN = 0;
    /// <summary>CUBLAS_OP_T — transpose</summary>
    internal const int OpT = 1;

    /// <summary>IEEE 754 fp16 bit pattern for 1.0f</summary>
    internal const ushort FP16One  = 0x3C00;
    /// <summary>IEEE 754 fp16 bit pattern for 0.0f</summary>
    internal const ushort FP16Zero = 0x0000;
}
