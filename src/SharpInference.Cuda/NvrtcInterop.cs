using System.Runtime.InteropServices;
using System.Text;

namespace SharpInference.Cuda;

/// <summary>
/// P/Invoke bindings for NVRTC (CUDA runtime compilation) and the CUDA Driver API.
/// Uses source-generated LibraryImport for NativeAOT compatibility.
///
/// NVRTC DLL resolution: tries nvrtc64_120_0 (CUDA 12.x), then nvrtc64_112_0 (CUDA 11.2+),
/// then nvrtc64_11 (CUDA 11.0) via a DllImportResolver registered in the static constructor.
/// </summary>
internal static unsafe partial class NvrtcInterop
{
    static NvrtcInterop()
    {
        NativeLibrary.SetDllImportResolver(typeof(NvrtcInterop).Assembly, static (name, _, _) =>
        {
            if (name == "nvrtc")
                foreach (var candidate in new[] { "nvrtc64_120_0", "nvrtc64_112_0", "nvrtc64_11" })
                    if (NativeLibrary.TryLoad(candidate, out nint h)) return h;
            return nint.Zero;
        });
    }

    // ── NVRTC ─────────────────────────────────────────────────────────────

    /// <summary>Create an NVRTC program from CUDA C source.</summary>
    [LibraryImport("nvrtc", EntryPoint = "nvrtcCreateProgram")]
    internal static partial int CreateProgram(out nint prog, byte* src, byte* name,
        int numHeaders, nint headers, nint includeNames);

    /// <summary>Compile the NVRTC program. Pass numOptions=0 and options=null for defaults.</summary>
    [LibraryImport("nvrtc", EntryPoint = "nvrtcCompileProgram")]
    internal static partial int CompileProgram(nint prog, int numOptions, nint options);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetPTXSize")]
    internal static partial int GetPTXSize(nint prog, out nuint ptxSize);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetPTX")]
    internal static partial int GetPTX(nint prog, byte* ptx);

    // Note: NVRTC exports these with UPPERCASE "CUBIN" (nvrtcGetCUBIN / nvrtcGetCUBINSize).
    // Spelling them with lowercase "Cubin" makes the entry-point lookup throw
    // EntryPointNotFoundException at first call, which silently routes the kernel
    // module through the PTX fallback and pays per-kernel JIT every prefill.
    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetCUBINSize")]
    internal static partial int GetCubinSize(nint prog, out nuint cubinSize);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetCUBIN")]
    internal static partial int GetCubin(nint prog, byte* cubin);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcCompileProgram")]
    internal static partial int CompileProgramWithOptions(nint prog, int numOptions, nint options);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcDestroyProgram")]
    internal static partial int DestroyProgram(ref nint prog);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetProgramLogSize")]
    internal static partial int GetProgramLogSize(nint prog, out nuint logSize);

    [LibraryImport("nvrtc", EntryPoint = "nvrtcGetProgramLog")]
    internal static partial int GetProgramLog(nint prog, byte* log);

    // ── CUDA Driver API ───────────────────────────────────────────────────

    [LibraryImport("nvcuda", EntryPoint = "cuInit")]
    internal static partial int CuInit(uint flags);

    [LibraryImport("nvcuda", EntryPoint = "cuDeviceGet")]
    internal static partial int DeviceGet(out int device, int ordinal);

    /// <summary>
    /// Retain the device's primary context (the one the Runtime API auto-attaches and that
    /// cuBLAS handles bind to internally). Bumps an internal refcount; pair with
    /// <see cref="DevicePrimaryCtxRelease"/> on shutdown.
    /// </summary>
    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRetain")]
    internal static partial int DevicePrimaryCtxRetain(out nint pctx, int dev);

    [LibraryImport("nvcuda", EntryPoint = "cuDevicePrimaryCtxRelease")]
    internal static partial int DevicePrimaryCtxRelease(int dev);

    /// <summary>
    /// Make <paramref name="ctx"/> the current CUDA context on the calling thread.
    /// Required before raw Driver-API calls (cuModuleLoadData, cuLaunchKernel) when the
    /// thread isn't the one that originally established the context. cuBLAS handles its
    /// own context binding so its APIs work cross-thread without this; cuModule does not.
    /// </summary>
    [LibraryImport("nvcuda", EntryPoint = "cuCtxSetCurrent")]
    internal static partial int CtxSetCurrent(nint ctx);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxGetCurrent")]
    internal static partial int CtxGetCurrent(out nint ctx);

    /// <summary>Load a PTX string (null-terminated) into a module.</summary>
    [LibraryImport("nvcuda", EntryPoint = "cuModuleLoadData")]
    internal static partial int ModuleLoadData(out nint module, byte* image);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleGetFunction")]
    internal static partial int ModuleGetFunction(out nint hfunc, nint hmod, byte* name);

    /// <summary>
    /// Launch a CUDA kernel. kernelParams is an array of void* where each element
    /// points to the actual argument value (device pointer, int, float, etc.).
    /// Pass extra = null.
    /// </summary>
    [LibraryImport("nvcuda", EntryPoint = "cuLaunchKernel")]
    internal static partial int LaunchKernel(
        nint f,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint hStream,
        nint* kernelParams, nint* extra);

    [LibraryImport("nvcuda", EntryPoint = "cuCtxSynchronize")]
    internal static partial int CtxSynchronize();

    [LibraryImport("nvcuda", EntryPoint = "cuStreamSynchronize")]
    internal static partial int CuStreamSync(nint hStream);

    [LibraryImport("nvcuda", EntryPoint = "cuModuleUnload")]
    internal static partial int ModuleUnload(nint hmod);

    /// <summary>
    /// Query an attribute of a kernel function. Calling this on a freshly loaded module
    /// has the side-effect of forcing the driver to JIT-compile that kernel up-front,
    /// even when CUDA's default lazy module loading would otherwise defer JIT until the
    /// first <c>cuLaunchKernel</c> for the function.
    /// CU_FUNC_ATTRIBUTE_NUM_REGS = 4 is queried because it is the cheapest attribute
    /// that still requires SASS to be generated.
    /// </summary>
    [LibraryImport("nvcuda", EntryPoint = "cuFuncGetAttribute")]
    internal static partial int FuncGetAttribute(out int pi, int attrib, nint hfunc);

    internal const int CU_FUNC_ATTRIBUTE_NUM_REGS = 4;

    // ── Helpers ───────────────────────────────────────────────────────────

    /// <summary>Return a null-terminated UTF-8 byte array for use as a C string.</summary>
    internal static byte[] ToUtf8(string s)
    {
        int len = Encoding.UTF8.GetByteCount(s);
        var buf = new byte[len + 1];
        Encoding.UTF8.GetBytes(s, buf);
        return buf; // buf[len] == 0 by default
    }
}
