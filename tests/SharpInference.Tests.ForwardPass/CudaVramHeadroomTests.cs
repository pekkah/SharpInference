using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// The VRAM headroom predicate behind <c>CudaForwardPass</c>'s low-headroom warning. A pinned
/// context (or an explicit <c>--kv-type</c>) bypasses the issue-#185 auto-narrow heuristic, so
/// the KV cache can consume enough VRAM that the weights no longer fit — which on Windows/WDDM
/// does not fail, it silently spills to shared host memory and drops throughput by ~20x. The
/// numbers below are the measured RTX 4070 Ti (12 GB) load of Rocinante-X-12B-Q4_K_M at q8_0 KV,
/// where context 24576 ran at full speed and 32768 spilled.
/// </summary>
public sealed class CudaVramHeadroomTests
{
    private const long Mib = 1024 * 1024;
    private const long RocinanteWeightsMib = 7127;   // ~6.96 GiB of Q4_K_M weights

    [Fact]
    public void WeightsWontFit_TrueWhenKvCacheLeavesTooLittleForWeights()
    {
        // ctx 32768: 5730 MiB free after the KV allocation, weights need 7127 MiB.
        Assert.True(CudaForwardPass.WeightsWontFit(5730 * Mib, RocinanteWeightsMib * Mib));
    }

    [Fact]
    public void WeightsWontFit_FalseWhenHeadroomRemains()
    {
        // ctx 24576: 8290 MiB free, weights 7127 MiB — over a GiB spare, ran at full speed.
        Assert.False(CudaForwardPass.WeightsWontFit(8290 * Mib, RocinanteWeightsMib * Mib));
    }

    [Fact]
    public void WeightsWontFit_TrueWhenOnlyTheScratchReserveIsMissing()
    {
        // Weights fit by 64 MiB, but not with the activation/cuBLAS/graph reserve on top —
        // "just barely fits" is exactly the case that spills once decode allocates.
        long free = RocinanteWeightsMib * Mib + 64 * Mib;
        Assert.True(CudaForwardPass.WeightsWontFit(free, RocinanteWeightsMib * Mib));
        Assert.False(CudaForwardPass.WeightsWontFit(
            free + CudaForwardPass.VramUploadReserveBytes, RocinanteWeightsMib * Mib));
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(-1L)]
    public void WeightsWontFit_FalseWhenTheDriverQueryFailed(long freeVramBytes)
    {
        // cuMemGetInfo failure reports 0 — warning on every load would be worse than silence.
        Assert.False(CudaForwardPass.WeightsWontFit(freeVramBytes, RocinanteWeightsMib * Mib));
    }
}
