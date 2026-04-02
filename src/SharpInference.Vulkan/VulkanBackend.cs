using Vortice.Vulkan;
using SharpInference.Core;
using static Vortice.Vulkan.Vulkan;

namespace SharpInference.Vulkan;

/// <summary>
/// Vulkan compute backend. Dispatches GLSL compute shaders via Vortice.Vulkan.
/// Shaders are compiled from <c>shaders/</c> at build time (glslc) and loaded as SPIR-V.
/// </summary>
public sealed unsafe class VulkanBackend : IComputeBackend
{
#pragma warning disable CS0169 // Never used — scaffold fields for Phase 2
    private VkInstance _instance;
    private VkPhysicalDevice _physicalDevice;
    private VkDevice _device;
    private VkQueue _computeQueue;
    private uint _computeQueueFamily;
#pragma warning restore CS0169

    public string Name => "Vulkan GPU";

    public VulkanBackend()
    {
        // TODO Phase 2: create Vulkan instance, pick physical device, create logical device
    }

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        // TODO: allocate VkBuffer on device-local memory
        throw new NotImplementedException();
    }

    public void Free(Tensor tensor)
    {
        // TODO: free VkBuffer and device memory
        throw new NotImplementedException();
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        // TODO: allocate VkBuffer on device-local memory, stage-copy via transfer queue
        throw new NotImplementedException();
    }

    public void Download(Tensor src, Span<float> dst)
    {
        // TODO: copy from device-local buffer to host via staging buffer
        throw new NotImplementedException();
    }

    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        // TODO: dispatch matmul.comp shader
        throw new NotImplementedException();
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        // TODO: dispatch add_inplace.comp shader
        throw new NotImplementedException();
    }

    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        // TODO: dispatch elementwise_mul.comp shader
        throw new NotImplementedException();
    }

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        // TODO: dispatch rmsnorm.comp shader
        throw new NotImplementedException();
    }

    public void Softmax(Tensor x)
    {
        // TODO: dispatch softmax.comp shader
        throw new NotImplementedException();
    }

    public void SiLU(Tensor x)
    {
        // TODO: dispatch silu.comp shader
        throw new NotImplementedException();
    }

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f)
    {
        // TODO: dispatch rope.comp shader
        throw new NotImplementedException();
    }

    public void Synchronize()
    {
        // TODO: vkQueueWaitIdle or fence
        throw new NotImplementedException();
    }

    public void Dispose()
    {
        // TODO Phase 2: destroy device, instance
    }
}
