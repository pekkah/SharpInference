using Vortice.Vulkan;

namespace SharpInference.Vulkan;

/// <summary>
/// A Vulkan buffer with its backing device memory.
/// Tracks size, usage flags, and whether the memory is host-visible (mappable).
/// </summary>
public sealed unsafe class GpuBuffer : IDisposable
{
    private readonly VkDeviceApi _vkd;

    public VkBuffer Buffer { get; }
    public VkDeviceMemory Memory { get; }
    public ulong Size { get; }
    public bool IsHostVisible { get; }
    private bool _disposed;

    private GpuBuffer(VkDeviceApi vkd, VkBuffer buffer, VkDeviceMemory memory, ulong size, bool hostVisible)
    {
        _vkd = vkd;
        Buffer = buffer;
        Memory = memory;
        Size = size;
        IsHostVisible = hostVisible;
    }

    /// <summary>
    /// Create a device-local buffer (VRAM, not host-visible).
    /// Used for weight storage and compute scratch buffers.
    /// </summary>
    public static GpuBuffer CreateDeviceLocal(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        return Create(backend, size, usage | VkBufferUsageFlags.TransferDst,
            VkMemoryPropertyFlags.DeviceLocal);
    }

    /// <summary>
    /// Create a host-visible, host-coherent staging buffer.
    /// Used for CPU→GPU and GPU→CPU data transfers.
    /// </summary>
    public static GpuBuffer CreateStaging(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        return Create(backend, size, usage,
            VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
    }

    /// <summary>
    /// Create a pinned host-visible buffer accessible from both CPU and GPU.
    /// Tries BAR memory (HostVisible + DeviceLocal) first for zero-copy GPU access,
    /// falls back to plain host-visible if BAR is not available.
    /// Used for small, frequently-transferred data like hidden states at GPU↔CPU boundaries.
    /// </summary>
    public static GpuBuffer CreatePinned(VulkanBackend backend, ulong size, VkBufferUsageFlags usage)
    {
        // Try resizable BAR: host-visible + device-local (best perf, zero-copy from GPU)
        try
        {
            return Create(backend, size,
                usage | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent | VkMemoryPropertyFlags.DeviceLocal);
        }
        catch
        {
            // Fallback: plain host-visible (requires staging for GPU access)
            return Create(backend, size,
                usage | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent);
        }
    }

    /// <summary>
    /// Map the buffer memory for CPU access. Only valid for host-visible buffers.
    /// Returns a pointer to the mapped region.
    /// </summary>
    public void* Map()
    {
        if (!IsHostVisible)
            throw new InvalidOperationException("Cannot map device-local buffer");
        void* ptr;
        _vkd.vkMapMemory(Memory, 0, Size, 0, &ptr).CheckResult();
        return ptr;
    }

    /// <summary>Unmap previously mapped memory.</summary>
    public void Unmap() => _vkd.vkUnmapMemory(Memory);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _vkd.vkDestroyBuffer(Buffer, null);
        _vkd.vkFreeMemory(Memory, null);
    }

    private static GpuBuffer Create(VulkanBackend backend, ulong size,
        VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryFlags)
    {
        var vkd = backend.Vkd;

        // Create buffer
        VkBufferCreateInfo bufferCI = new()
        {
            size = size,
            usage = usage,
            sharingMode = VkSharingMode.Exclusive,
        };
        VkBuffer buffer;
        vkd.vkCreateBuffer(&bufferCI, null, &buffer).CheckResult();

        // Query memory requirements
        VkMemoryRequirements memReqs;
        vkd.vkGetBufferMemoryRequirements(buffer, &memReqs);

        // Allocate device memory
        VkMemoryAllocateInfo allocInfo = new()
        {
            allocationSize = memReqs.size,
            memoryTypeIndex = backend.FindMemoryType(memReqs.memoryTypeBits, memoryFlags),
        };
        VkDeviceMemory memory;
        vkd.vkAllocateMemory(&allocInfo, null, &memory).CheckResult();

        // Bind buffer to memory
        vkd.vkBindBufferMemory(buffer, memory, 0).CheckResult();

        bool hostVisible = (memoryFlags & VkMemoryPropertyFlags.HostVisible) != 0;
        return new GpuBuffer(vkd, buffer, memory, size, hostVisible);
    }
}
