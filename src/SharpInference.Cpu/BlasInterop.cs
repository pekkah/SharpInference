using System.Runtime.InteropServices;

namespace SharpInference.Cpu;

/// <summary>
/// P/Invoke bindings for OpenBLAS cblas_sgemm via .NET LibraryImport source generator.
/// Requires libopenblas.dll in the runtime directory or system PATH.
/// Run scripts/setup-openblas.ps1 to download it.
/// </summary>
internal static unsafe partial class BlasInterop
{
    // CBLAS constants
    internal const int RowMajor = 101;
    internal const int NoTrans = 111;
    internal const int Trans = 112;

    /// <summary>
    /// Single-precision General Matrix Multiply.
    /// C = alpha * op(A) * op(B) + beta * C
    /// where op(X) is X or X^T depending on transA/transB.
    /// </summary>
    [LibraryImport("libopenblas", EntryPoint = "cblas_sgemm")]
    internal static partial void Sgemm(
        int order,   // RowMajor (101) or ColMajor (102)
        int transA,  // NoTrans (111) or Trans (112)
        int transB,  // NoTrans (111) or Trans (112)
        int m,       // rows of op(A) and C
        int n,       // columns of op(B) and C
        int k,       // columns of op(A) / rows of op(B)
        float alpha, // scalar multiplier for A*B
        float* a,    // pointer to matrix A
        int lda,     // leading dimension of A
        float* b,    // pointer to matrix B
        int ldb,     // leading dimension of B
        float beta,  // scalar multiplier for C
        float* c,    // pointer to output matrix C
        int ldc);    // leading dimension of C

    private static readonly Lazy<bool> s_available = new(ProbeLibrary);

    /// <summary>
    /// Check if OpenBLAS is available at runtime (cached after first check).
    /// </summary>
    internal static bool IsAvailable => s_available.Value;

    private static bool ProbeLibrary()
    {
        // Try standard search paths first
        if (NativeLibrary.TryLoad("libopenblas", out _))
            return true;

        // Try tools/openblas relative to the assembly
        var asmDir = Path.GetDirectoryName(typeof(BlasInterop).Assembly.Location);
        if (asmDir is not null)
        {
            // Check sibling tools/openblas directory
            for (var dir = asmDir; dir is not null; dir = Path.GetDirectoryName(dir))
            {
                var candidate = Path.Combine(dir, "tools", "openblas", "libopenblas.dll");
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out _))
                    return true;
            }
        }

        return false;
    }
}
