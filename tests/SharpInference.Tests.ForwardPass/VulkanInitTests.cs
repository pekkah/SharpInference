namespace SharpInference.Tests.ForwardPass;

public sealed class VulkanInitTests
{
    [Fact]
    public void CreateBackendAndPrintDeviceInfo()
    {
        using var backend = new Vulkan.VulkanBackend();
        backend.PrintDeviceInfo();

        Assert.Contains("GPU", backend.Name);
        Assert.NotEqual(0u, backend.ComputeQueueFamily);
    }

    [Fact]
    public void FindsDeviceLocalMemoryType()
    {
        using var backend = new Vulkan.VulkanBackend();

        // Should find a device-local memory type (for VRAM buffers)
        uint memType = backend.FindMemoryType(
            uint.MaxValue,
            Vortice.Vulkan.VkMemoryPropertyFlags.DeviceLocal);

        // Memory type index should be reasonable
        Assert.True(memType < 32);
    }
}
