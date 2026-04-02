using SharpInference.Core;

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
        uint memType = backend.FindMemoryType(
            uint.MaxValue,
            Vortice.Vulkan.VkMemoryPropertyFlags.DeviceLocal);
        Assert.True(memType < 32);
    }

    [Fact]
    public void AllocateAndFreeBuffer()
    {
        using var backend = new Vulkan.VulkanBackend();
        var tensor = backend.Allocate(TensorShape.D1(1024));
        Assert.NotEqual(0, tensor.Handle);
        Assert.Equal(1024, tensor.ElementCount);
        backend.Free(tensor);
    }

    [Fact]
    public void UploadDownloadRoundTrip()
    {
        using var backend = new Vulkan.VulkanBackend();

        // Create test data
        var src = new float[256];
        for (int i = 0; i < src.Length; i++) src[i] = i * 0.1f;

        // Upload to VRAM
        var tensor = backend.Upload(src, TensorShape.D1(256));
        Assert.NotEqual(0, tensor.Handle);

        // Download back
        var dst = new float[256];
        backend.Download(tensor, dst);

        // Verify
        for (int i = 0; i < 256; i++)
            Assert.Equal(src[i], dst[i], 5);

        backend.Free(tensor);
    }

    [Fact]
    public void UploadDownloadLargeBuffer()
    {
        using var backend = new Vulkan.VulkanBackend();

        // 4MB buffer (simulate a weight matrix)
        int size = 1024 * 1024;
        var src = new float[size];
        var rng = new Random(42);
        for (int i = 0; i < size; i++) src[i] = (float)(rng.NextDouble() * 2 - 1);

        var tensor = backend.Upload(src, TensorShape.D2(1024, 1024));
        var dst = new float[size];
        backend.Download(tensor, dst);

        for (int i = 0; i < size; i++)
            Assert.Equal(src[i], dst[i], 5);

        backend.Free(tensor);
    }
}
