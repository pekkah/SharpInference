using System.Numerics;

namespace SharpInference.Engine;

/// <summary>
/// Single source of truth for the TurboQuant KVarN-vs-Lloyd-Max support matrix
/// (issue #437). The CLI run/perplexity commands and the server all resolve the
/// auto <c>--tq-mode</c> and validate explicit modes through here, so the matrix,
/// its reason strings, and the #432 quality warning can't drift across the three
/// frontends. Before #437 the ternary chain was hand-copied into
/// <c>RunCommand.Execute</c>, <c>PerplexityCommand.ResolveAutoQuantizer</c>, and
/// <c>InferenceEngineLoader.ResolveTq</c>, and a missed edit silently resolved
/// auto → Lloyd-Max on one surface only, reintroducing the #432 quality collapse.
///
/// Ground-truth bounds mirror the engine guards:
/// <list type="bullet">
///   <item>KVarN CPU head dim — power of two in [8, 1024] (<c>KVarNCompressor.ValidateHeadDim</c>)</item>
///   <item>KVarN CUDA head dim — power of two in [8, 256] (<c>CudaForwardPass</c> ctor: shared-mem WHT cap)</item>
///   <item>Lloyd-Max head dim — {128, 256} (<c>TurboQuantCodebooks</c>)</item>
///   <item>KVarN tile — 128 tokens (the <c>--tq-window</c> floor)</item>
/// </list>
/// </summary>
public static class TqSupport
{
    /// <summary>KVarN compression tile size; the FP32 window must hold at least one full tile.</summary>
    public const int KVarNTile = 128;

    /// <summary>Largest head dim the CUDA KVarN kernels handle (shared-memory WHT staging cap).</summary>
    public const int KVarNCudaMaxHeadDim = 256;

    /// <summary>
    /// The shared #432 collapse warning clause. Callers prefix a per-frontend lead-in
    /// (and, for the CLI, markup) and append a per-frontend "silence this warning" hint.
    /// </summary>
    public const string QualityWarningReason =
        "Lloyd-Max severely degrades quality on QK-norm models such as Qwen3 (issue #432)";

    // ── Head-dim envelopes (mirror the engine guards) ──

    /// <summary>KVarN CPU envelope: power of two in [8, 1024].</summary>
    public static bool IsKVarNHeadDim(int headDim) =>
        BitOperations.IsPow2(headDim) && headDim is >= 8 and <= 1024;

    /// <summary>KVarN CUDA envelope: power of two in [8, 256] (shared-memory WHT cap).</summary>
    public static bool IsKVarNCudaHeadDim(int headDim) =>
        IsKVarNHeadDim(headDim) && headDim <= KVarNCudaMaxHeadDim;

    /// <summary>Lloyd-Max envelope: only 128 and 256 ship hardcoded codebooks.</summary>
    public static bool IsLloydMaxHeadDim(int headDim) => headDim is 128 or 256;

    // ── Reason strings (why KVarN is unavailable on a given path) ──

    public const string SnapKvReason =
        "SnapKV eviction (SHARPI_SNAPKV_BUDGET) does not compose with KVarN yet";
    public const string VulkanReason = "KVarN is not supported on the Vulkan backend";
    public const string CudaDeviceReason = "GPU KVarN requires a CUDA device";
    public const string CudaMoeReason = "KVarN on CUDA supports dense models only";

    public static string HeadDimReason(int headDim) =>
        $"KVarN needs a power-of-2 head dim in [8, 1024]; this model has {headDim}";

    public static string CudaHeadDimReason(int headDim) =>
        $"KVarN on CUDA requires head dim ≤ 256; this model has {headDim}";

    public static string WindowReason(int window) =>
        $"KVarN needs --tq-window >= 128 (one full tile); got {window}";

    /// <summary>
    /// The auto <c>--tq-mode</c> matrix: returns <c>null</c> when KVarN is usable on the
    /// resolved path, otherwise the reason it is blocked. On a <c>null</c> return the
    /// caller uses KVarN; otherwise it falls back to Lloyd-Max with the #432 warning
    /// (auto mode) or errors (explicit <c>--tq-mode kvarn</c>).
    ///
    /// <paramref name="onGpu"/> is true when any layers are offloaded; the GPU sub-checks
    /// (<paramref name="isVulkan"/>, <paramref name="cudaAvailable"/>, <paramref name="isMoE"/>,
    /// head dim &gt; 256) apply only then. Partial-vs-full offload is not knowable here
    /// (TierPlanner runs later), so the GPU branch is evaluated as the full-offload target;
    /// callers downgrade an auto-resolved KVarN on a partial split separately.
    ///
    /// <paramref name="window"/> is the FP32 window size when the frontend exposes it
    /// (perplexity's <c>--tq-window</c>); pass <c>null</c> when the window is fixed ≥ one tile.
    /// </summary>
    public static string? KVarNBlockedReason(
        int headDim, bool snapKvEnabled, bool onGpu,
        bool isVulkan, bool cudaAvailable, bool isMoE, int? window = null)
    {
        if (snapKvEnabled) return SnapKvReason;
        if (window is int w && w < KVarNTile) return WindowReason(w);
        if (!IsKVarNHeadDim(headDim)) return HeadDimReason(headDim);
        if (!onGpu) return null;                     // CPU KVarN: any pow-2 head dim, dense or MoE
        if (isVulkan) return VulkanReason;
        if (!cudaAvailable) return CudaDeviceReason;
        if (isMoE) return CudaMoeReason;
        if (headDim > KVarNCudaMaxHeadDim) return CudaHeadDimReason(headDim);
        return null;
    }
}
