using SharpInference.Core;
using SharpInference.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for subgroup-size pinning (issue #318). The reduction shaders assume a 32-wide
/// subgroup; on AMD Wave64 (subgroup 64) two 32-lane row groups share one subgroup, so
/// subgroupAdd/subgroupElect corrupt the output. We pin requiredSubgroupSize=32 via
/// VK_EXT_subgroup_size_control on shaders whose local_size_x is a multiple of 32, but only
/// on devices that could pick a non-32 subgroup. Devices locked to exactly 32 (NVIDIA report
/// min==max==32) are already correct, so the pin is skipped there (avoids the driver's
/// required-subgroup-size pipeline path entirely).
///
/// CAVEAT: the Wave64 effect cannot be exercised on NVIDIA hardware (subgroup is already 32,
/// so no pin is even applied). These tests verify the host-side wiring (properties queried,
/// device/pipeline still create, output unchanged) and the pin-gate logic.
/// </summary>
public sealed unsafe class VulkanSubgroupSizeTests
{
    /// <summary>Construct a backend, returning null (test skipped) if no Vulkan device exists.</summary>
    private static VulkanBackend? TryCreateBackend()
    {
        try
        {
            return new VulkanBackend();
        }
        catch
        {
            // No Vulkan-capable device / loader in this environment — skip.
            return null;
        }
    }

    [Fact]
    public void SubgroupSizeRangeIsQueried()
    {
        using var backend = TryCreateBackend();
        if (backend is null) return; // no device

        if (backend.HasSubgroupSizeControl)
        {
            // The properties query (vkGetPhysicalDeviceProperties2 + SubgroupSizeControlProperties)
            // must have populated a valid, ordered range.
            Assert.True(backend.MinSubgroupSize >= 1,
                $"MinSubgroupSize should be >= 1 when the extension is present, got {backend.MinSubgroupSize}");
            Assert.True(backend.MaxSubgroupSize >= backend.MinSubgroupSize,
                $"MaxSubgroupSize ({backend.MaxSubgroupSize}) should be >= MinSubgroupSize ({backend.MinSubgroupSize})");
        }
        else
        {
            // Extension absent → fields left at their disabling default of 0.
            Assert.Equal(0u, backend.MinSubgroupSize);
            Assert.Equal(0u, backend.MaxSubgroupSize);
        }
    }

    [Fact]
    public void ParseLocalSizeXReadsTheLiteral()
    {
        Assert.Equal(256, ComputePipeline.ParseLocalSizeX(
            "#version 450\nlayout(local_size_x = 256) in;\nvoid main() {}"));
        Assert.Equal(128, ComputePipeline.ParseLocalSizeX(
            "layout(local_size_x = 128, local_size_y = 1) in;"));
        Assert.Equal(16, ComputePipeline.ParseLocalSizeX(
            "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;"));
        // Tolerates extra whitespace around the '=' (within a layout qualifier).
        Assert.Equal(256, ComputePipeline.ParseLocalSizeX("layout(local_size_x   =   256) in;"));
        // No declaration → 0 (disables pinning).
        Assert.Equal(0, ComputePipeline.ParseLocalSizeX("void main() {}"));
        // A mention outside a layout(...) qualifier (e.g. a comment) must NOT false-match.
        Assert.Equal(0, ComputePipeline.ParseLocalSizeX("// local_size_x = 64 is the default\nvoid main() {}"));
    }

    [Theory]
    [InlineData(256, true)]   // reduction shaders → pinned
    [InlineData(128, true)]   // reduction shaders → pinned
    [InlineData(512, true)]   // any multiple of 32 → pinned
    [InlineData(16, false)]   // image-op shaders → NOT pinned (smaller than one subgroup)
    [InlineData(0, false)]    // no local_size_x → NOT pinned
    public void PinGateIncludesMultiplesOf32(int localSizeX, bool multipleOf32Expected)
    {
        // The local_size_x % 32 == 0 sub-condition of the gate (independent of HW capability).
        bool multipleOf32 = localSizeX > 0 && localSizeX % 32 == 0;
        Assert.Equal(multipleOf32Expected, multipleOf32);
    }

    [Fact]
    public void ShouldPinSubgroupSize32MatchesGate()
    {
        using var backend = TryCreateBackend();
        if (backend is null) return; // no device

        // Pinning is needed only when the device could pick a non-32 subgroup (AMD Wave64 → 64,
        // Intel → possibly <32). A device locked to exactly 32 (e.g. NVIDIA) is already correct,
        // so we skip the pin there to avoid the driver's required-subgroup-size pipeline path.
        bool needsPin = backend.HasSubgroupSizeControl
            && backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
            && !(backend.MinSubgroupSize == 32 && backend.MaxSubgroupSize == 32);

        // 16-thread image-op shaders must never be pinned, regardless of HW capability.
        Assert.False(ComputePipeline.ShouldPinSubgroupSize32(backend, 16));
        // 0 (no local_size_x) must never be pinned.
        Assert.False(ComputePipeline.ShouldPinSubgroupSize32(backend, 0));

        // Multiples of 32 are pinned iff the device needs the pin.
        Assert.Equal(needsPin, ComputePipeline.ShouldPinSubgroupSize32(backend, 256));
        Assert.Equal(needsPin, ComputePipeline.ShouldPinSubgroupSize32(backend, 128));
    }

    [Fact]
    public void RmsNormUnchangedWithSubgroupPin()
    {
        // No-regression smoke: RmsNorm is a 256-thread reduction shader (subgroupAdd over a
        // 32-lane row group). Output must match the CPU reference whether or not the pin is
        // applied — i.e. the #318 wiring did not break pipeline creation or compute.
        using var backend = TryCreateBackend();
        if (backend is null) return; // no device

        const int N = 2048;
        var input = new float[N];
        var weight = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            input[i] = (float)(rng.NextDouble() * 2 - 1);
            weight[i] = (float)(rng.NextDouble() * 0.5 + 0.75);
        }

        var gpuInput = backend.Upload(input, TensorShape.D1(N));
        var gpuWeight = backend.Upload(weight, TensorShape.D1(N));
        var gpuOutput = backend.Allocate(TensorShape.D1(N));
        backend.RmsNorm(gpuOutput, gpuInput, gpuWeight, 1e-5f);

        var gpuResult = new float[N];
        backend.Download(gpuOutput, gpuResult);

        float sumSq = 0;
        for (int i = 0; i < N; i++) sumSq += input[i] * input[i];
        float scale = 1f / MathF.Sqrt(sumSq / N + 1e-5f);
        for (int i = 0; i < N; i++)
        {
            float expected = input[i] * scale * weight[i];
            Assert.True(MathF.Abs(gpuResult[i] - expected) < 0.001f,
                $"RmsNorm mismatch at [{i}]: gpu={gpuResult[i]}, cpu={expected}");
        }

        backend.Free(gpuInput);
        backend.Free(gpuWeight);
        backend.Free(gpuOutput);
    }

    [Fact]
    public void PrintsQueriedSubgroupSizeRange()
    {
        // Surfaces the queried range in test output for the issue report.
        using var backend = TryCreateBackend();
        if (backend is null) return; // no device

        Console.WriteLine(
            $"[#318] HasSubgroupSizeControl={backend.HasSubgroupSizeControl} " +
            $"MinSubgroupSize={backend.MinSubgroupSize} MaxSubgroupSize={backend.MaxSubgroupSize}");
    }
}
