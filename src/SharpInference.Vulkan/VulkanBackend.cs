using Vortice.Vulkan;
using SharpInference.Core;
using static Vortice.Vulkan.Vulkan;

namespace SharpInference.Vulkan;

/// <summary>
/// Vulkan compute backend using Vortice.Vulkan.
/// Selects a discrete GPU, creates a compute-only queue, and manages
/// VRAM buffers for inference tensor operations.
/// </summary>
public sealed unsafe class VulkanBackend : IComputeBackend, IDisposable
{
    private readonly VkInstance _instance;
    private readonly VkInstanceApi _vki;
    private readonly VkPhysicalDevice _physicalDevice;
    private readonly VkDevice _device;
    private readonly VkDeviceApi _vkd;
    private readonly VkQueue _computeQueue;
    private readonly uint _computeQueueFamily;
    private readonly VkPhysicalDeviceProperties _deviceProperties;
    private readonly VkPhysicalDeviceMemoryProperties _memoryProperties;
    private readonly VkCommandPool _commandPool;
    private readonly VkCommandBuffer _transferCmd; // single-use for staging transfers
    private bool _disposed;

    public string Name { get; }
    public uint ComputeQueueFamily => _computeQueueFamily;
    public VkInstanceApi Vki => _vki;
    public VkDeviceApi Vkd => _vkd;
    public VkDevice Device => _device;
    public VkQueue ComputeQueue => _computeQueue;
    public VkCommandBuffer TransferCmd => _transferCmd;

    public VulkanBackend()
    {
        // 1. Initialize Vulkan loader
        vkInitialize().CheckResult();

        // 2. Create instance (Vulkan 1.3, no extensions for compute-only)
        VkApplicationInfo appInfo = new()
        {
            apiVersion = VkVersion.Version_1_3,
        };
        VkInstanceCreateInfo instanceCI = new()
        {
            pApplicationInfo = &appInfo,
        };
        vkCreateInstance(in instanceCI, out _instance).CheckResult();
        _vki = new VkInstanceApi(in _instance);

        // 3. Select physical device (prefer discrete GPU)
        _physicalDevice = SelectPhysicalDevice();
        _vki.vkGetPhysicalDeviceProperties(_physicalDevice, out _deviceProperties);
        VkPhysicalDeviceMemoryProperties memProps;
        _vki.vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &memProps);
        _memoryProperties = memProps;

        fixed (byte* namePtr = _deviceProperties.deviceName)
            Name = $"Vulkan GPU ({new string((sbyte*)namePtr)})";

        // 4. Find best compute queue family (prefer dedicated compute over shared graphics+compute)
        _computeQueueFamily = FindComputeQueueFamily();

        // 5. Create logical device with one compute queue
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCI = new()
        {
            queueFamilyIndex = _computeQueueFamily,
            queueCount = 1,
            pQueuePriorities = &queuePriority,
        };
        VkDeviceCreateInfo deviceCI = new()
        {
            queueCreateInfoCount = 1,
            pQueueCreateInfos = &queueCI,
        };
        _vki.vkCreateDevice(_physicalDevice, in deviceCI, out _device).CheckResult();
        _vkd = new VkDeviceApi(_vki, in _device);

        // 6. Get the compute queue handle
        VkQueue queue;
        _vkd.vkGetDeviceQueue(_computeQueueFamily, 0, &queue);
        _computeQueue = queue;

        // 7. Create command pool + one reusable command buffer for transfers
        VkCommandPoolCreateInfo poolCI = new()
        {
            flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
            queueFamilyIndex = _computeQueueFamily,
        };
        VkCommandPool pool;
        _vkd.vkCreateCommandPool(&poolCI, null, &pool).CheckResult();
        _commandPool = pool;

        VkCommandBufferAllocateInfo cmdAllocInfo = new()
        {
            commandPool = _commandPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VkCommandBuffer cmd;
        _vkd.vkAllocateCommandBuffers(&cmdAllocInfo, &cmd).CheckResult();
        _transferCmd = cmd;
    }

    /// <summary>Print device info to console.</summary>
    public void PrintDeviceInfo()
    {
        Console.WriteLine($"Device: {Name}");
        Console.WriteLine($"  API: {_deviceProperties.apiVersion}");
        Console.WriteLine($"  Compute queue family: {_computeQueueFamily}");
        for (int i = 0; i < (int)_memoryProperties.memoryHeapCount; i++)
        {
            var heap = _memoryProperties.memoryHeaps[i];
            var flags = (heap.flags & VkMemoryHeapFlags.DeviceLocal) != 0 ? "VRAM" : "RAM";
            Console.WriteLine($"  Heap {i}: {heap.size / 1024 / 1024}MB ({flags})");
        }
    }

    /// <summary>Find the memory type index matching the requested flags.</summary>
    public uint FindMemoryType(uint typeFilter, VkMemoryPropertyFlags properties)
    {
        for (int i = 0; i < (int)_memoryProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1u << i)) != 0 &&
                (_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return (uint)i;
        }
        throw new InvalidOperationException($"No memory type found for filter={typeFilter}, properties={properties}");
    }

    // ================================================================
    //  Physical device selection
    // ================================================================

    private VkPhysicalDevice SelectPhysicalDevice()
    {
        uint count = 0;
        _vki.vkEnumeratePhysicalDevices(&count, null);
        if (count == 0) throw new InvalidOperationException("No Vulkan-capable GPU found");

        var devices = new VkPhysicalDevice[count];
        fixed (VkPhysicalDevice* p = devices)
            _vki.vkEnumeratePhysicalDevices(&count, p);

        // Prefer discrete GPU, fall back to any compute-capable device
        VkPhysicalDevice fallback = default;
        foreach (var gpu in devices)
        {
            var props = _vki.vkGetPhysicalDeviceProperties(gpu);
            if (props.deviceType == VkPhysicalDeviceType.DiscreteGpu)
                return gpu;
            if (fallback.IsNull && HasComputeQueue(gpu))
                fallback = gpu;
        }
        return fallback.IsNull
            ? throw new InvalidOperationException("No compute-capable GPU found")
            : fallback;
    }

    private bool HasComputeQueue(VkPhysicalDevice gpu)
    {
        uint count = 0;
        _vki.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, null);
        var families = new VkQueueFamilyProperties[count];
        fixed (VkQueueFamilyProperties* p = families)
            _vki.vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, p);
        return families.Any(f => (f.queueFlags & VkQueueFlags.Compute) != 0);
    }

    private uint FindComputeQueueFamily()
    {
        uint count = 0;
        _vki.vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &count, null);
        var families = new VkQueueFamilyProperties[count];
        fixed (VkQueueFamilyProperties* p = families)
            _vki.vkGetPhysicalDeviceQueueFamilyProperties(_physicalDevice, &count, p);

        // Prefer dedicated compute queue (no graphics bit) for async compute
        uint dedicated = uint.MaxValue;
        uint shared = uint.MaxValue;
        for (uint i = 0; i < count; i++)
        {
            if ((families[i].queueFlags & VkQueueFlags.Compute) == 0) continue;
            if ((families[i].queueFlags & VkQueueFlags.Graphics) == 0)
                dedicated = i;  // Pure compute queue — best for async dispatch
            else if (shared == uint.MaxValue)
                shared = i;
        }
        return dedicated != uint.MaxValue ? dedicated : shared != uint.MaxValue ? shared
            : throw new InvalidOperationException("No compute queue family found");
    }

    // ================================================================
    //  Buffer tracking: Tensor.Handle → GpuBuffer
    // ================================================================

    private readonly Dictionary<nint, GpuBuffer> _buffers = new();
    private nint _nextHandle = 1;

    public GpuBuffer GetBuffer(Tensor tensor) =>
        _buffers.TryGetValue(tensor.Handle, out var buf)
            ? buf
            : throw new InvalidOperationException($"Tensor handle {tensor.Handle} not found");

    public GpuBuffer GetBuffer(nint handle) =>
        _buffers.TryGetValue(handle, out var buf)
            ? buf
            : throw new InvalidOperationException($"Handle {handle} not found");

    // ================================================================
    //  IComputeBackend — Memory management
    // ================================================================

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        ulong byteSize = (ulong)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst);

        var handle = _nextHandle++;
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_buffers.Remove(tensor.Handle, out var buf))
            buf.Dispose();
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(float));

        // Create device-local destination buffer
        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        // Create staging buffer, copy data, transfer to VRAM
        using var staging = GpuBuffer.CreateStaging(this, byteSize,
            VkBufferUsageFlags.TransferSrc);

        // Map staging, copy data
        float* mapped = (float*)staging.Map();
        data.CopyTo(new Span<float>(mapped, data.Length));
        staging.Unmap();

        // Record and submit copy command
        CopyBuffer(staging, gpuBuf, byteSize);

        var handle = _nextHandle++;
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.Float32, handle);
    }

    public void Download(Tensor src, Span<float> dst)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)(dst.Length * sizeof(float));

        using var staging = GpuBuffer.CreateStaging(this, byteSize,
            VkBufferUsageFlags.TransferDst);

        CopyBuffer(gpuBuf, staging, byteSize);

        float* mapped = (float*)staging.Map();
        new Span<float>(mapped, dst.Length).CopyTo(dst);
        staging.Unmap();
    }

    public void Synchronize()
    {
        _vkd.vkQueueWaitIdle(_computeQueue);
    }

    // ================================================================
    //  Buffer copy via command buffer
    // ================================================================

    private void CopyBuffer(GpuBuffer src, GpuBuffer dst, ulong size)
    {
        VkCommandBufferBeginInfo beginInfo = new()
        {
            flags = VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        _vkd.vkBeginCommandBuffer(_transferCmd, &beginInfo).CheckResult();

        VkBufferCopy copyRegion = new() { size = size };
        _vkd.vkCmdCopyBuffer(_transferCmd, src.Buffer, dst.Buffer, 1, &copyRegion);

        _vkd.vkEndCommandBuffer(_transferCmd).CheckResult();

        VkCommandBuffer cmd = _transferCmd;
        VkSubmitInfo submitInfo = new()
        {
            commandBufferCount = 1,
            pCommandBuffers = &cmd,
        };
        _vkd.vkQueueSubmit(_computeQueue, 1, &submitInfo, VkFence.Null).CheckResult();
        _vkd.vkQueueWaitIdle(_computeQueue); // synchronous for now
    }

    // ================================================================
    //  Compute shader pipelines (created lazily on first use)
    // ================================================================

    private ComputePipeline? _rmsNormPipeline;
    private ComputePipeline? _siluMulPipeline;
    private ComputePipeline? _addInPlacePipeline;
    private ComputePipeline? _elementwiseMulPipeline;
    private ComputePipeline? _ropePipeline;
    private ComputePipeline? _softmaxPipeline;
    private ComputePipeline? _matVecQ4KPipeline;
    private ComputePipeline? _matVecF32Pipeline;

    private struct RmsNormParams { public uint n; public float eps; }
    private struct CountParams { public uint n; }
    private struct RoPEParams { public uint numHeads; public uint headDim; public int position; public float theta; }
    private struct MatVecParams { public uint rows; public uint cols; }

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        _rmsNormPipeline ??= new ComputePipeline(this, Shaders.RmsNorm, 3, pushConstantSize: sizeof(RmsNormParams));
        var p = new RmsNormParams { n = (uint)x.ElementCount, eps = eps };
        _rmsNormPipeline.DispatchWith(_transferCmd,
            [GetBuffer(x), GetBuffer(weight), GetBuffer(output)], 1, pushConstants: &p);
    }

    public void SiLU(Tensor x) => throw new NotImplementedException("Use SiLuMul for fused SiLU*gate");

    public void SiLuMul(Tensor gate, Tensor up)
    {
        _siluMulPipeline ??= new ComputePipeline(this, Shaders.SiLuMul, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)gate.ElementCount };
        _siluMulPipeline.DispatchWith(_transferCmd,
            [GetBuffer(gate), GetBuffer(up)], ((uint)gate.ElementCount + 255) / 256, pushConstants: &p);
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        _addInPlacePipeline ??= new ComputePipeline(this, Shaders.AddInPlace, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)dst.ElementCount };
        _addInPlacePipeline.DispatchWith(_transferCmd,
            [GetBuffer(dst), GetBuffer(src)], ((uint)dst.ElementCount + 255) / 256, pushConstants: &p);
    }

    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        _elementwiseMulPipeline ??= new ComputePipeline(this, Shaders.ElementwiseMul, 3, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)a.ElementCount };
        _elementwiseMulPipeline.DispatchWith(_transferCmd,
            [GetBuffer(a), GetBuffer(b), GetBuffer(output)], ((uint)a.ElementCount + 255) / 256, pushConstants: &p);
    }

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f)
    {
        _ropePipeline ??= new ComputePipeline(this, Shaders.RoPE, 1, pushConstantSize: sizeof(RoPEParams));
        uint numHeads = (uint)(x.ElementCount / headDim);
        uint totalPairs = numHeads * (uint)(headDim / 2);
        var p = new RoPEParams { numHeads = numHeads, headDim = (uint)headDim, position = position, theta = ropeTheta };
        _ropePipeline.DispatchWith(_transferCmd,
            [GetBuffer(x)], (totalPairs + 255) / 256, pushConstants: &p);
    }

    public void Softmax(Tensor x)
    {
        _softmaxPipeline ??= new ComputePipeline(this, Shaders.Softmax, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)x.ElementCount };
        _softmaxPipeline.DispatchWith(_transferCmd, [GetBuffer(x)], 1, pushConstants: &p);
    }

    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        // Default: assume Q4_K weights
        MatMul(output, matrix, vector, DType.Q4_K);
    }

    public void MatMul(Tensor output, Tensor matrix, Tensor vector, DType weightDType)
    {
        var p = new MatVecParams { rows = (uint)output.ElementCount, cols = (uint)vector.ElementCount };
        if (weightDType == DType.Float32)
        {
            _matVecF32Pipeline ??= new ComputePipeline(this, Shaders.MatVecF32, 3, pushConstantSize: sizeof(MatVecParams));
            _matVecF32Pipeline.DispatchWith(_transferCmd,
                [GetBuffer(matrix), GetBuffer(vector), GetBuffer(output)], (uint)output.ElementCount, pushConstants: &p);
        }
        else
        {
            _matVecQ4KPipeline ??= new ComputePipeline(this, Shaders.MatVecQ4K, 3, pushConstantSize: sizeof(MatVecParams));
            _matVecQ4KPipeline.DispatchWith(_transferCmd,
                [GetBuffer(matrix), GetBuffer(vector), GetBuffer(output)], (uint)output.ElementCount, pushConstants: &p);
        }
    }

    // ================================================================
    //  Disposal
    // ================================================================

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _vkd.vkDeviceWaitIdle();

        // Dispose compute pipelines
        _rmsNormPipeline?.Dispose();
        _siluMulPipeline?.Dispose();
        _addInPlacePipeline?.Dispose();
        _elementwiseMulPipeline?.Dispose();
        _ropePipeline?.Dispose();
        _softmaxPipeline?.Dispose();
        _matVecQ4KPipeline?.Dispose();
        _matVecF32Pipeline?.Dispose();

        // Free all tracked GPU buffers
        foreach (var buf in _buffers.Values)
            buf.Dispose();
        _buffers.Clear();

        _vkd.vkDestroyCommandPool(_commandPool, null);
        _vkd.vkDestroyDevice(null);
        _vki.vkDestroyInstance(null);
    }
}
