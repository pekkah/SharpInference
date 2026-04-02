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
    private bool _disposed;

    public string Name { get; }
    public uint ComputeQueueFamily => _computeQueueFamily;
    public VkInstanceApi Vki => _vki;
    public VkDeviceApi Vkd => _vkd;
    public VkDevice Device => _device;

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
    //  IComputeBackend stubs (Phase 2 implementation)
    // ================================================================

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32) => throw new NotImplementedException();
    public void Free(Tensor tensor) => throw new NotImplementedException();
    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape) => throw new NotImplementedException();
    public void Download(Tensor src, Span<float> dst) => throw new NotImplementedException();
    public void MatMul(Tensor output, Tensor matrix, Tensor vector) => throw new NotImplementedException();
    public void AddInPlace(Tensor dst, Tensor src) => throw new NotImplementedException();
    public void ElementwiseMul(Tensor output, Tensor a, Tensor b) => throw new NotImplementedException();
    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f) => throw new NotImplementedException();
    public void Softmax(Tensor x) => throw new NotImplementedException();
    public void SiLU(Tensor x) => throw new NotImplementedException();
    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f) => throw new NotImplementedException();
    public void Synchronize() => throw new NotImplementedException();

    // ================================================================
    //  Disposal
    // ================================================================

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _vkd.vkDeviceWaitIdle();
        _vkd.vkDestroyDevice(null);
        _vki.vkDestroyInstance(null);
    }
}
