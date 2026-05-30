using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for <see cref="WarmPinConfig"/>'s auto-enable rule. The env-var
/// override path is exercised at module-load time (static readonly) so it can't
/// be unit-tested in-process without process recycling; the resolution helper
/// itself is pure and covered here.
/// </summary>
public sealed class WarmPinConfigTests
{
    [Fact]
    public void ResolvePerLayer_CacheHoldsFullSet_DisablesWarmPin()
    {
        // 8 layers × 4 experts = 32 total; cache holds 32 → warm-pin is a no-op.
        int n = WarmPinConfig.ResolvePerLayer(numLayers: 8, numExperts: 4, numActiveExperts: 2, slotCapacity: 32);
        // Env-var override may have been set externally for the suite; only assert
        // the auto-enable rule when the explicit override is off.
        if (WarmPinConfig.PerLayer == 0) Assert.Equal(0, n);
    }

    [Fact]
    public void ResolvePerLayer_TightCache_AutoEnablesAtActiveExperts()
    {
        int n = WarmPinConfig.ResolvePerLayer(numLayers: 8, numExperts: 64, numActiveExperts: 8, slotCapacity: 16);
        if (WarmPinConfig.PerLayer == 0) Assert.Equal(8, n);
    }

    [Fact]
    public void ResolvePerLayer_DenseRoute_FallsBackToOne()
    {
        int n = WarmPinConfig.ResolvePerLayer(numLayers: 4, numExperts: 8, numActiveExperts: 0, slotCapacity: 4);
        if (WarmPinConfig.PerLayer == 0) Assert.Equal(1, n);
    }

    [Fact]
    public void ResolvePerLayer_CapsAtNumExperts()
    {
        // numActiveExperts (16) exceeds numExperts (8) — cap at numExperts.
        int n = WarmPinConfig.ResolvePerLayer(numLayers: 4, numExperts: 8, numActiveExperts: 16, slotCapacity: 4);
        if (WarmPinConfig.PerLayer == 0) Assert.Equal(8, n);
    }
}
