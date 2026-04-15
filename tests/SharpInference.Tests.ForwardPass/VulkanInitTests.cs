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
    public unsafe void ComputeShaderVectorAdd()
    {
        using var backend = new Vulkan.VulkanBackend();

        const string shaderSource = """
            #version 450
            layout(local_size_x = 256) in;
            layout(binding = 0) buffer BufA { float a[]; };
            layout(binding = 1) buffer BufB { float b[]; };
            layout(binding = 2) buffer BufC { float c[]; };
            layout(push_constant) uniform Params { uint n; };
            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i < n) c[i] = a[i] + b[i];
            }
            """;

        const int N = 1024;
        var srcA = new float[N];
        var srcB = new float[N];
        for (int i = 0; i < N; i++) { srcA[i] = i; srcB[i] = i * 2; }

        var gpuA = backend.Upload(srcA, TensorShape.D1(N));
        var gpuB = backend.Upload(srcB, TensorShape.D1(N));
        var gpuC = backend.Allocate(TensorShape.D1(N));

        using var pipeline = new Vulkan.ComputePipeline(backend, shaderSource,
            bindingCount: 3, pushConstantSize: sizeof(uint));

        var ds = pipeline.AllocateDescriptorSet();
        pipeline.UpdateDescriptorSet(ds,
            backend.GetBuffer(gpuA),
            backend.GetBuffer(gpuB),
            backend.GetBuffer(gpuC));

        uint n = N;
        uint groups = (N + 255) / 256;
        pipeline.Dispatch(backend.TransferCmd, ds, groups, pushConstants: &n);

        var result = new float[N];
        backend.Download(gpuC, result);

        for (int i = 0; i < N; i++)
            Assert.Equal(srcA[i] + srcB[i], result[i], 3);

        backend.Free(gpuA);
        backend.Free(gpuB);
        backend.Free(gpuC);
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
