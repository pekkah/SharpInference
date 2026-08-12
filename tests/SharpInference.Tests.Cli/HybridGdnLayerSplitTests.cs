using SharpInference.Cli;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="RunCommand.ResolveHybridGdnLayerSplit"/> — maps <c>-g N</c> onto
/// the CUDA/Vulkan hybrid-GDN forward passes' dense-FFN-on-GPU cap. No GPU/model needed (pure
/// arithmetic); no env-var side effects.
/// </summary>
public sealed class HybridGdnLayerSplitTests
{
    [Theory]
    [InlineData(64, -1, 64, 0)]  // auto → no cap (all layers)
    [InlineData(64, 99, 64, 0)]  // over numLayers → clamp to no cap
    [InlineData(64, 24, 24, 40)] // explicit split
    [InlineData(64, 12, 12, 52)] // explicit split
    [InlineData(64, 0, 0, 64)]   // -g 0: passed through as GpuLayers=0 (CudaHybridGdnForwardPass /
                                 // VulkanHybridGdnForwardPass interpret GpuLayers==0 as "zero
                                 // dense-FFN layers on GPU", not "no cap" — negative-only means
                                 // uncapped there). Unreachable via the CLI today: RunCommand
                                 // short-circuits -g 0 to the CPU-only pass before this helper runs.
    [InlineData(0, 5, 0, 0)]     // degenerate numLayers=0
    public void ResolveHybridGdnLayerSplit_MapsMinusOneAndOverflowToNoCap(
        int numLayers, int nGpuLayers, int expectedGpu, int expectedCpu)
    {
        var (gpu, cpu) = RunCommand.ResolveHybridGdnLayerSplit(numLayers, nGpuLayers);

        Assert.Equal(expectedGpu, gpu);
        Assert.Equal(expectedCpu, cpu);
    }
}
