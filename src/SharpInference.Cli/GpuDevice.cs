namespace SharpInference.Cli;

/// <summary>
/// Parses the llama.cpp-style <c>-dev/--device</c> option into a single GPU selection and
/// applies it. Unlike llama.cpp's comma-separated multi-GPU list, SharpInference targets a
/// single device (it has no tensor/row split), so only one device may be named.
///
/// Accepted values:
///   <list type="bullet">
///     <item><c>null</c> / empty / <c>auto</c> — auto-select (returns index -1, no change)</item>
///     <item><c>none</c> / <c>cpu</c> — don't offload (sets <paramref name="none"/> = true)</item>
///     <item>a bare index — <c>0</c>, <c>1</c>, …</item>
///     <item>a named device with a trailing index — <c>CUDA0</c>, <c>Vulkan1</c>, <c>GPU2</c></item>
///   </list>
///
/// When a concrete index is selected it pins the CUDA runtime to that physical GPU via
/// <c>CUDA_VISIBLE_DEVICES</c> (honored process-wide by every CUDA host thread, unlike a
/// per-thread <c>cudaSetDevice</c>) and returns the same index for the Vulkan physical-device
/// selector. The CUDA env var is only set when the caller has not already constrained the
/// visible set themselves.
/// </summary>
internal static class GpuDevice
{
    public static int Resolve(string? device, out bool none)
    {
        none = false;
        if (string.IsNullOrWhiteSpace(device))
            return -1;

        var value = device.Trim();
        if (value.Equals("auto", StringComparison.OrdinalIgnoreCase))
            return -1;
        if (value.Equals("none", StringComparison.OrdinalIgnoreCase) ||
            value.Equals("cpu", StringComparison.OrdinalIgnoreCase))
        {
            none = true;
            return -1;
        }

        if (value.Contains(','))
            throw new InvalidOperationException(
                $"--device '{device}': multi-device split is not supported; specify a single device " +
                "(e.g. 0, CUDA0, Vulkan1, or 'none' for CPU).");

        // Strip an optional leading backend name (CUDA/Vulkan/GPU/…) and read the trailing index.
        int i = value.Length;
        while (i > 0 && char.IsDigit(value[i - 1])) i--;
        var digits = value[i..];
        // The part before the index must be empty or a plain backend name (letters only) —
        // this rejects things like "-1" (which would otherwise parse as device 1).
        bool prefixOk = true;
        for (int j = 0; j < i; j++)
            if (!char.IsLetter(value[j])) { prefixOk = false; break; }
        if (!prefixOk || digits.Length == 0 || !int.TryParse(digits, out int index) || index < 0)
            throw new InvalidOperationException(
                $"--device '{device}': expected a device index (0, 1, …), a named device " +
                "(CUDA0, Vulkan1), 'auto', or 'none'.");

        // Pin CUDA to this physical device process-wide. Harmless on the Vulkan path (Vulkan
        // enumerates all devices regardless and uses the returned index). Don't override an
        // explicit CUDA_VISIBLE_DEVICES the user already set in the environment.
        if (Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") is null)
            Environment.SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", index.ToString());

        return index;
    }
}
