using System.Runtime.InteropServices;

namespace SharpInference.Cpu;

/// <summary>
/// P/Invoke bindings for the native AVX-512-VNNI CPU kernels (sharpi_cpu_vnni.dll)
/// via the .NET LibraryImport source generator (NativeAOT/trim-clean).
///
/// First vertical slice (perf/carnice-vnni-moe): a single-input Q3_K · Q8_KS dot
/// product using vpdpbusd, to match llama.cpp's ggml-cpu-zen4 speed on Carnice
/// (Qwen3.6-35B-A3B), whose routed experts are ~75% Q3_K. The native path is
/// optional: when the DLL is absent or the CPU lacks AVX512_VNNI, or the
/// SHARPI_CPU_VNNI env var is "0", <see cref="IsAvailable"/> is false and the
/// managed AVX2 kernel in <see cref="SimdKernels"/> runs unchanged.
///
/// Run scripts/build-vnni.ps1 to produce the DLL (clang-cl, optional).
/// </summary>
internal static unsafe partial class Q8VnniInterop
{
    private const string LibName = "sharpi_cpu_vnni";

    /// <summary>
    /// CPUID-based probe (no AVX-512 instruction executed): 1 if the CPU
    /// supports AVX512_VNNI, else 0. Safe to call on any x86-64 host.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "sharpi_has_avx512vnni")]
    private static partial int HasAvx512Vnni();

    /// <summary>
    /// Single-input Q3_K weight row · Q8_KS prequantized activation dot product.
    /// Bit-identical (integer domain) to <see cref="SimdKernels.DotQ3K_Q8KS_Scalar"/>.
    /// </summary>
    /// <param name="row">Q3_K weights, 110 bytes per super-block of 256 elements.</param>
    /// <param name="scratch">Q8_KS activation scratch (see SimdKernels layout).</param>
    /// <param name="numBlocks">Number of 256-element super-blocks (cols / 256).</param>
    [LibraryImport(LibName, EntryPoint = "sharpi_dot_q3k_q8ks")]
    [SuppressGCTransition]
    private static partial float DotQ3KQ8KsNative(byte* row, byte* scratch, int numBlocks);

    /// <summary>
    /// Two-input dequant-once Q3_K · Q8_KS dot: decodes the weight row once and
    /// dots it against two Q8_KS activations, each accumulation bit-identical
    /// (integer domain) to two <see cref="DotQ3KQ8KsNative"/> calls.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "sharpi_dot_q3k_q8ks_2in")]
    [SuppressGCTransition]
    private static partial void DotQ3KQ8Ks2InNative(byte* row, byte* s0, byte* s1,
        int numBlocks, float* out0, float* out1);

    /// <summary>
    /// Four-input dequant-once Q3_K · Q8_KS dot: decodes the weight row once and
    /// dots it against four Q8_KS activations, each accumulation bit-identical
    /// (integer domain) to four <see cref="DotQ3KQ8KsNative"/> calls.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "sharpi_dot_q3k_q8ks_4in")]
    [SuppressGCTransition]
    private static partial void DotQ3KQ8Ks4InNative(byte* row, byte* s0, byte* s1,
        byte* s2, byte* s3, int numBlocks,
        float* out0, float* out1, float* out2, float* out3);

    private static readonly Lazy<bool> s_available = new(Probe);
    private static readonly Lazy<bool> s_hasVnni = new(ProbeVnniSupport);

    /// <summary>
    /// True iff the native VNNI path should be used: the SHARPI_CPU_VNNI kill
    /// switch is not "0", the DLL loads, and the CPU supports AVX512_VNNI.
    /// Cached after first check. Never throws.
    /// </summary>
    internal static bool IsAvailable => s_available.Value;

    /// <summary>
    /// True iff the native DLL loads and its CPUID probe reports AVX512_VNNI —
    /// independent of the SHARPI_CPU_VNNI kill switch. Used by tests to tell a
    /// genuinely-VNNI host (where the native path must engage) apart from one
    /// where it is legitimately unavailable. Never throws.
    /// </summary>
    internal static bool HasVnniSupport => s_hasVnni.Value;

    private static bool ProbeVnniSupport()
    {
        try
        {
            return TryLoad() && HasAvx512Vnni() != 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool Probe()
    {
        try
        {
            if (Environment.GetEnvironmentVariable("SHARPI_CPU_VNNI") == "0")
                return false;

            if (!TryLoad())
                return false;

            return HasAvx512Vnni() != 0;
        }
        catch
        {
            // Any failure (missing DLL, missing entry point, load fault) means
            // the native path is unavailable; the AVX2 fallback runs unchanged.
            return false;
        }
    }

    private static bool TryLoad()
    {
        // Plain name first (runtime dir / PATH / NATIVE_DLL_SEARCH_DIRECTORIES).
        if (NativeLibrary.TryLoad(LibName, out _))
            return true;

        var asmDir = Path.GetDirectoryName(typeof(Q8VnniInterop).Assembly.Location);
        if (asmDir is null)
            return false;

        // Walk up from the assembly dir looking for the DLL in known locations.
        for (var dir = asmDir; dir is not null; dir = Path.GetDirectoryName(dir))
        {
            var inNative = Path.Combine(dir, "native", "cpu_vnni", LibName + ".dll");
            if (File.Exists(inNative) && NativeLibrary.TryLoad(inNative, out _))
                return true;

            var inTools = Path.Combine(dir, "tools", "vnni", LibName + ".dll");
            if (File.Exists(inTools) && NativeLibrary.TryLoad(inTools, out _))
                return true;
        }

        // The assembly dir itself (CopyToOutputDirectory drops it next to the binary).
        var sibling = Path.Combine(asmDir, LibName + ".dll");
        return File.Exists(sibling) && NativeLibrary.TryLoad(sibling, out _);
    }

    /// <summary>
    /// Native single-input Q3_K · Q8_KS dot. Caller must ensure
    /// <see cref="IsAvailable"/> is true.
    /// </summary>
    internal static float DotQ3K_Q8KS(byte* row, byte* scratch, int numBlocks) =>
        DotQ3KQ8KsNative(row, scratch, numBlocks);

    /// <summary>
    /// Native two-input dequant-once Q3_K · Q8_KS dot. Caller must ensure
    /// <see cref="IsAvailable"/> is true. No managed allocations.
    /// </summary>
    internal static void DotQ3K_Q8KS_2In(byte* row, byte* s0, byte* s1, int numBlocks,
        out float o0, out float o1)
    {
        float r0, r1;
        DotQ3KQ8Ks2InNative(row, s0, s1, numBlocks, &r0, &r1);
        o0 = r0;
        o1 = r1;
    }

    /// <summary>
    /// Native four-input dequant-once Q3_K · Q8_KS dot. Caller must ensure
    /// <see cref="IsAvailable"/> is true. No managed allocations.
    /// </summary>
    internal static void DotQ3K_Q8KS_4In(byte* row, byte* s0, byte* s1, byte* s2, byte* s3,
        int numBlocks, out float o0, out float o1, out float o2, out float o3)
    {
        float r0, r1, r2, r3;
        DotQ3KQ8Ks4InNative(row, s0, s1, s2, s3, numBlocks, &r0, &r1, &r2, &r3);
        o0 = r0;
        o1 = r1;
        o2 = r2;
        o3 = r3;
    }
}
