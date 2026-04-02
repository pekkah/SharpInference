using System.Diagnostics;

namespace SharpInference.Vulkan;

/// <summary>
/// Compiles GLSL compute shaders to SPIR-V using the Vulkan SDK's glslc compiler.
/// Caches compiled SPIR-V to avoid recompilation.
/// </summary>
public static class ShaderCompiler
{
    private static readonly Dictionary<int, byte[]> s_cache = new();

    /// <summary>
    /// Compile a GLSL compute shader source string to SPIR-V bytes.
    /// Results are cached by source hash.
    /// </summary>
    public static byte[] Compile(string glslSource, string entryPoint = "main")
    {
        int hash = glslSource.GetHashCode();
        if (s_cache.TryGetValue(hash, out var cached))
            return cached;

        var glslcPath = FindGlslc()
            ?? throw new FileNotFoundException("glslc not found. Install the Vulkan SDK.");

        // Write source to temp file, compile to SPIR-V
        var tempGlsl = Path.GetTempFileName() + ".comp";
        var tempSpv = Path.GetTempFileName() + ".spv";
        try
        {
            File.WriteAllText(tempGlsl, glslSource);

            var psi = new ProcessStartInfo
            {
                FileName = glslcPath,
                Arguments = $"\"{tempGlsl}\" -o \"{tempSpv}\" --target-env=vulkan1.3 -O",
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            using var proc = Process.Start(psi)!;
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(30_000);

            if (proc.ExitCode != 0)
                throw new InvalidOperationException($"glslc failed:\n{stderr}");

            var spirv = File.ReadAllBytes(tempSpv);
            s_cache[hash] = spirv;
            return spirv;
        }
        finally
        {
            File.Delete(tempGlsl);
            File.Delete(tempSpv);
        }
    }

    private static string? FindGlslc()
    {
        // Check PATH
        var pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator) ?? [];
        foreach (var dir in pathDirs)
        {
            var candidate = Path.Combine(dir, "glslc.exe");
            if (File.Exists(candidate)) return candidate;
            candidate = Path.Combine(dir, "glslc");
            if (File.Exists(candidate)) return candidate;
        }

        // Check common Vulkan SDK locations on Windows
        var vulkanSdk = Environment.GetEnvironmentVariable("VULKAN_SDK");
        if (vulkanSdk is not null)
        {
            var candidate = Path.Combine(vulkanSdk, "Bin", "glslc.exe");
            if (File.Exists(candidate)) return candidate;
        }

        return null;
    }
}
