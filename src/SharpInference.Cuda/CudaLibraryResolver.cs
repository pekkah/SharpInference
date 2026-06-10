using System.Runtime.InteropServices;

namespace SharpInference.Cuda;

/// <summary>
/// The assembly's single <see cref="DllImportResolver"/> (only one may be registered
/// per assembly): handles NVRTC version probing and CUDA runtime/cuBLAS major-version
/// selection. Registered from the static constructors of both <see cref="NvrtcInterop"/>
/// and <see cref="CuBlasInterop"/> so it is in place before the first P/Invoke binds,
/// whichever interop class is touched first.
///
/// The runtime imports stay pinned to the CUDA 12 SONAMEs (<c>cudart64_12</c> /
/// <c>cublas64_12</c>). Resolution policy:
///   - <c>SHARPI_CUDA13=1</c> prefers the CUDA 13 pair when both DLLs load (A/B knob —
///     every entry point we import exists unchanged in 13.x);
///   - otherwise CUDA 12 is used when present (default, matches the NVRTC the kernels
///     are JIT'd against; PTX is forward-compatible via the driver);
///   - when 12 is absent (CUDA-13-only toolkit installs), 13 loads as a fallback
///     instead of hard-failing model load.
/// cudart and cublas are decided as a PAIR so a mixed 12/13 process never occurs.
/// </summary>
internal static class CudaLibraryResolver
{
    private static int s_registered;
    private static int s_runtimeMajor; // 0 = undecided; then 12 or 13

    internal static void Register()
    {
        if (Interlocked.Exchange(ref s_registered, 1) != 0) return;
        NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, Resolve);
    }

    private static nint Resolve(string name, System.Reflection.Assembly asm, DllImportSearchPath? path)
    {
        if (name == "nvrtc")
        {
            foreach (var candidate in new[] { "nvrtc64_120_0", "nvrtc64_112_0", "nvrtc64_11" })
                if (NativeLibrary.TryLoad(candidate, out nint h)) return h;
            return nint.Zero;
        }
        if (name is "cudart64_12" or "cublas64_12")
        {
            string target = DecideRuntimeMajor() == 13 ? name.Replace("_12", "_13") : name;
            return NativeLibrary.TryLoad(target, out nint h) ? h : nint.Zero;
        }
        return nint.Zero;
    }

    private static int DecideRuntimeMajor()
    {
        int major = Volatile.Read(ref s_runtimeMajor);
        if (major != 0) return major;

        bool prefer13 = Environment.GetEnvironmentVariable("SHARPI_CUDA13") == "1";
        bool has13 = CanLoadPair("cudart64_13", "cublas64_13");
        bool has12 = CanLoadPair("cudart64_12", "cublas64_12");
        major = prefer13 && has13 ? 13
              : has12 ? 12
              : has13 ? 13
              : 12; // neither loads — keep the pinned name so the standard load error surfaces
        if (major == 13)
        {
            Console.Error.WriteLine("[SharpInference] CUDA runtime: using cudart64_13/cublas64_13"
                + (prefer13 ? " (SHARPI_CUDA13=1)" : " (CUDA 12 runtime not found)"));
        }
        Volatile.Write(ref s_runtimeMajor, major);
        return major;

        static bool CanLoadPair(string cudart, string cublas) =>
            NativeLibrary.TryLoad(cudart, out _) && NativeLibrary.TryLoad(cublas, out _);
    }
}
