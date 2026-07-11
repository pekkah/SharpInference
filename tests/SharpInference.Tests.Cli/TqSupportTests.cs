using SharpInference.Engine;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for the centralized TurboQuant support matrix (<see cref="TqSupport"/>,
/// issue #437). This is the single source of truth all three frontends (RunCommand,
/// PerplexityCommand, InferenceEngineLoader) resolve auto --tq-mode through, so the
/// matrix — and especially the GPU sub-cases the perplexity wrapper can't express
/// (Vulkan backend, no CUDA device, CUDA head dim &gt; 256) — is verified once here.
/// </summary>
public sealed class TqSupportTests
{
    // (headDim, snapKv, onGpu, isVulkan, cudaAvailable, isMoE, window, kvarnUsable)
    [Theory]
    // CPU path: KVarN for any pow-2 head dim, dense or MoE, ignoring GPU sub-flags.
    [InlineData(128, false, false, false, true, false, null, true)]
    [InlineData(64, false, false, false, false, true, null, true)]    // CPU MoE, no CUDA — still KVarN
    [InlineData(1024, false, false, false, true, false, null, true)]  // largest CPU KVarN head dim
    // CUDA full offload: dense pow-2 [8,256] → KVarN.
    [InlineData(128, false, true, false, true, false, null, true)]
    [InlineData(256, false, true, false, true, false, null, true)]
    // CUDA blockers.
    [InlineData(128, false, true, false, true, true, null, false)]    // MoE on CUDA → Lloyd-Max
    [InlineData(512, false, true, false, true, false, null, false)]   // head dim > 256 on CUDA
    // GPU sub-cases perplexity can't express:
    [InlineData(128, false, true, true, true, false, null, false)]    // Vulkan backend
    [InlineData(128, false, true, false, false, false, null, false)]  // GPU requested, no CUDA device
    // Codec-independent blockers (apply on any path, before the GPU checks).
    [InlineData(128, true, false, false, true, false, null, false)]   // SnapKV enabled
    [InlineData(96, false, false, false, true, false, null, false)]   // non-pow-2 head dim
    [InlineData(128, false, false, false, true, false, 64, false)]    // window below one KVarN tile
    [InlineData(128, false, false, false, true, false, 128, true)]    // window exactly one tile
    public void KVarNBlockedReason_Matrix(int headDim, bool snapKv, bool onGpu, bool isVulkan,
        bool cudaAvailable, bool isMoE, int? window, bool kvarnUsable)
    {
        string? reason = TqSupport.KVarNBlockedReason(
            headDim, snapKv, onGpu, isVulkan, cudaAvailable, isMoE, window);

        if (kvarnUsable)
            Assert.Null(reason);
        else
            Assert.False(string.IsNullOrEmpty(reason));
    }

    [Theory]
    [InlineData(8, true)]
    [InlineData(64, true)]
    [InlineData(128, true)]
    [InlineData(1024, true)]
    [InlineData(96, false)]    // not a power of two
    [InlineData(4, false)]     // below the floor
    [InlineData(2048, false)]  // above the ceiling
    public void IsKVarNHeadDim_Envelope(int headDim, bool expected) =>
        Assert.Equal(expected, TqSupport.IsKVarNHeadDim(headDim));

    [Theory]
    [InlineData(128, true)]
    [InlineData(256, true)]
    [InlineData(512, false)]   // valid CPU KVarN, but over the CUDA shared-mem WHT cap
    [InlineData(96, false)]
    public void IsKVarNCudaHeadDim_Envelope(int headDim, bool expected) =>
        Assert.Equal(expected, TqSupport.IsKVarNCudaHeadDim(headDim));

    [Theory]
    [InlineData(128, true)]
    [InlineData(256, true)]
    [InlineData(64, false)]    // no Lloyd-Max codebook
    [InlineData(512, false)]
    public void IsLloydMaxHeadDim_Envelope(int headDim, bool expected) =>
        Assert.Equal(expected, TqSupport.IsLloydMaxHeadDim(headDim));
}
