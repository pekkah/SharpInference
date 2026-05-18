using System.Collections.Concurrent;
using System.Threading;
using Vortice.Vulkan;
using SharpInference.Core;
using static Vortice.Vulkan.Vulkan;

namespace SharpInference.Vulkan;

/// <summary>
/// Vulkan compute backend using Vortice.Vulkan.
/// Selects a discrete GPU, creates a compute-only queue, and manages
/// VRAM buffers for inference tensor operations.
/// </summary>
public sealed unsafe class VulkanBackend : IComputeBackend, IImageOpsBackend, IDisposable
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
    private readonly VkCommandBuffer _transferCmd; // main-thread staging transfers
    private readonly VkFence _fence; // main-thread fence

    // Separate command pool/buffer/fence for background (prefetcher) uploads.
    // Vulkan requires all cmd buffer operations from a single thread per pool;
    // giving the prefetcher its own pool makes concurrent uploads safe.
    private readonly VkCommandPool _asyncPool;
    private readonly VkCommandBuffer _asyncCmd;
    private readonly VkFence _asyncFence;
    private GpuBuffer? _asyncStaging;
    private ulong _asyncStagingSize;
    private readonly object _asyncCmdLock = new(); // serializes concurrent UploadBackground calls

    // Serializes vkQueueSubmit from both main and background threads.
    private readonly object _queueLock = new();

    private bool _disposed;

    public string Name { get; }
    public uint ComputeQueueFamily => _computeQueueFamily;
    public VkInstanceApi Vki => _vki;
    public VkDeviceApi Vkd => _vkd;
    public VkDevice Device => _device;
    public VkQueue ComputeQueue => _computeQueue;
    public VkCommandBuffer TransferCmd => _transferCmd;

    // Batched recording mode: record multiple dispatches, submit once
    private bool _recording;

    // Deferred-free support for image-ops batch recording:
    // Frees issued while _deferringFrees=true are held until EndBatch() completes the submit.
    private bool _deferringFrees;
    private readonly List<Tensor> _deferredFrees = [];

    private long _recordingEpoch;

    /// <summary>
    /// Monotonically-increasing counter incremented on every <see cref="BeginRecord"/>.
    /// <see cref="ComputePipeline.RecordWith"/> reads this to recycle its per-recording
    /// descriptor sets when a new session starts — see the comment on RecordWith for why
    /// per-dispatch descriptor sets are required for correctness.
    /// </summary>
    public long RecordingEpoch => _recordingEpoch;

    /// <summary>Begin recording a batch of compute dispatches.</summary>
    public void BeginRecord()
    {
        VkCommandBufferBeginInfo begin = new() { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
        _vkd.vkBeginCommandBuffer(_transferCmd, &begin).CheckResult();
        _recording = true;
        _recordingEpoch++;
    }

    /// <summary>Insert a compute→compute memory barrier (all writes visible to reads).</summary>
    public void RecordBarrier()
    {
        VkMemoryBarrier barrier = new()
        {
            srcAccessMask = VkAccessFlags.ShaderWrite,
            dstAccessMask = VkAccessFlags.ShaderRead,
        };
        _vkd.vkCmdPipelineBarrier(_transferCmd,
            VkPipelineStageFlags.ComputeShader, VkPipelineStageFlags.ComputeShader,
            0, 1, &barrier, 0, null, 0, null);
    }

    /// <summary>Insert a compute→transfer barrier (shader writes visible to transfer reads).</summary>
    public void RecordComputeToTransferBarrier()
    {
        VkMemoryBarrier barrier = new()
        {
            srcAccessMask = VkAccessFlags.ShaderWrite,
            dstAccessMask = VkAccessFlags.TransferRead,
        };
        _vkd.vkCmdPipelineBarrier(_transferCmd,
            VkPipelineStageFlags.ComputeShader, VkPipelineStageFlags.Transfer,
            0, 1, &barrier, 0, null, 0, null);
    }

    /// <summary>
    /// Insert a compute→host barrier so subsequent host reads of host-visible (pinned/BAR)
    /// memory observe the latest compute-shader writes after the next submit completes.
    /// Fence wait alone is insufficient on some drivers; an explicit Host-stage barrier is required.
    /// </summary>
    public void RecordComputeToHostBarrier()
    {
        VkMemoryBarrier barrier = new()
        {
            srcAccessMask = VkAccessFlags.ShaderWrite,
            dstAccessMask = VkAccessFlags.HostRead,
        };
        _vkd.vkCmdPipelineBarrier(_transferCmd,
            VkPipelineStageFlags.ComputeShader, VkPipelineStageFlags.Host,
            0, 1, &barrier, 0, null, 0, null);
    }

    /// <summary>
    /// Record a GPU→staging copy into the current command buffer.
    /// Call before <see cref="EndRecordAndSubmit"/>, then <see cref="ReadFromStaging"/>
    /// after the fence fires — eliminates a second command-buffer submission for logits download.
    /// </summary>
    public void RecordDownloadToStaging(Tensor src, int floatCount)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)((long)floatCount * sizeof(float));

        if (_downloadStaging == null || _downloadStagingSize < byteSize)
        {
            _downloadStaging?.Dispose();
            _downloadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferDst);
            _downloadStagingSize = byteSize;
        }

        VkBufferCopy copyRegion = new() { size = byteSize };
        _vkd.vkCmdCopyBuffer(_transferCmd, gpuBuf.Buffer, _downloadStaging.Buffer, 1, &copyRegion);
    }

    /// <summary>CPU map-and-copy from the staging buffer populated by <see cref="RecordDownloadToStaging"/>.</summary>
    public void ReadFromStaging(Span<float> dst)
    {
        float* mapped = (float*)_downloadStaging!.Map();
        new Span<float>(mapped, dst.Length).CopyTo(dst);
        _downloadStaging.Unmap();
    }

    /// <summary>Insert a transfer→compute barrier (copy finished before shader reads).</summary>
    public void RecordTransferBarrier()
    {
        VkMemoryBarrier barrier = new()
        {
            srcAccessMask = VkAccessFlags.TransferWrite,
            dstAccessMask = VkAccessFlags.ShaderRead,
        };
        _vkd.vkCmdPipelineBarrier(_transferCmd,
            VkPipelineStageFlags.Transfer, VkPipelineStageFlags.ComputeShader,
            0, 1, &barrier, 0, null, 0, null);
    }

    /// <summary>End recording and submit all dispatches. Synchronous wait via fence.</summary>
    public void EndRecordAndSubmit()
    {
        _recording = false;
        _vkd.vkEndCommandBuffer(_transferCmd).CheckResult();
        SubmitAndWait();
    }

    /// <summary>End recording and submit without waiting. Call <see cref="WaitForGpu"/> before reading results.</summary>
    public void EndRecordAndSubmitAsync()
    {
        _recording = false;
        _vkd.vkEndCommandBuffer(_transferCmd).CheckResult();
        VkCommandBuffer cmd = _transferCmd;
        VkSubmitInfo submit = new() { commandBufferCount = 1, pCommandBuffers = &cmd };
        var fence = _fence;
        _vkd.vkResetFences(1, &fence).CheckResult();
        _vkd.vkQueueSubmit(_computeQueue, 1, &submit, _fence).CheckResult();
    }

    /// <summary>Wait for a previously submitted async batch to complete.</summary>
    public void WaitForGpu()
    {
        var fence = _fence;
        _vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    // ── Image-ops batch recording (IImageOpsBackend) ──────────────────────

    /// <summary>
    /// Begin recording an image-ops batch. All dispatches until <see cref="EndBatch"/> are
    /// recorded into one command buffer and submitted together.
    /// Free() calls during the batch are deferred until EndBatch() completes the submit.
    /// </summary>
    public void BeginBatch()
    {
        _deferringFrees = true;
        BeginRecord();
    }

    /// <summary>
    /// Insert a compute→compute memory barrier.
    /// Required between dependent dispatches within a batch recording.
    /// No-op outside batch recording.
    /// </summary>
    public void BatchBarrier()
    {
        if (_recording) RecordBarrier();
    }

    /// <summary>
    /// End the image-ops batch: submit all recorded dispatches, wait for GPU completion,
    /// then process all deferred frees.
    /// </summary>
    public void EndBatch()
    {
        EndRecordAndSubmit();
        _deferringFrees = false;
        foreach (var t in _deferredFrees) Free(t);
        _deferredFrees.Clear();
    }

    /// <summary>Submit the transfer command buffer and wait for completion via fence.</summary>
    private void SubmitAndWait()
    {
        VkCommandBuffer cmd = _transferCmd;
        VkSubmitInfo submit = new() { commandBufferCount = 1, pCommandBuffers = &cmd };
        var fence = _fence;
        lock (_queueLock)
        {
            _vkd.vkResetFences(1, &fence).CheckResult();
            _vkd.vkQueueSubmit(_computeQueue, 1, &submit, _fence).CheckResult();
        }
        _vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    /// <summary>Submit the async command buffer (background thread) and wait via its fence.</summary>
    private void SubmitAndWaitAsync()
    {
        VkCommandBuffer cmd = _asyncCmd;
        VkSubmitInfo submit = new() { commandBufferCount = 1, pCommandBuffers = &cmd };
        var fence = _asyncFence;
        lock (_queueLock)
        {
            _vkd.vkResetFences(1, &fence).CheckResult();
            _vkd.vkQueueSubmit(_computeQueue, 1, &submit, _asyncFence).CheckResult();
        }
        _vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    public VulkanBackend()
    {
        // 1. Initialize Vulkan loader
        vkInitialize().CheckResult();

        // 2. Create instance (Vulkan 1.3+)
        VkApplicationInfo appInfo = new()
        {
            apiVersion = VkVersion.Version_1_3, // Vortice may not have 1.4 constant yet
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

        // 5. Detect supported device extensions (must happen before device creation to enable them)
        uint extCount = 0;
        _vki.vkEnumerateDeviceExtensionProperties(_physicalDevice, null, &extCount, null);
        var exts = new VkExtensionProperties[extCount];
        fixed (VkExtensionProperties* p = exts)
            _vki.vkEnumerateDeviceExtensionProperties(_physicalDevice, null, &extCount, p);
        var extNames = new HashSet<string>();
        for (int i = 0; i < (int)extCount; i++)
        {
            fixed (byte* namePtr = exts[i].extensionName)
                extNames.Add(new string((sbyte*)namePtr));
        }

        bool hasFloat16Int8   = extNames.Contains("VK_KHR_shader_float16_int8");
        bool has16BitStorage  = extNames.Contains("VK_KHR_16bit_storage");
        bool has8BitStorage   = extNames.Contains("VK_KHR_8bit_storage");
        bool hasIntDot        = extNames.Contains("VK_KHR_shader_integer_dot_product");
        bool hasBfloat16      = extNames.Contains("VK_KHR_shader_bfloat16");
        bool hasFloat8        = extNames.Contains("VK_EXT_shader_float8");
        HasShaderFloat16Int8    = hasFloat16Int8;
        Has16BitStorage         = has16BitStorage;
        Has8BitStorage          = has8BitStorage;
        HasShaderIntegerDotProduct = hasIntDot;
        HasCooperativeMatrix    = extNames.Contains("VK_KHR_cooperative_matrix");
        HasSubgroupSizeControl  = extNames.Contains("VK_EXT_subgroup_size_control");
        HasShaderBfloat16       = hasBfloat16;
        HasShaderFloat8         = hasFloat8;

        // 6. Create logical device with one compute queue, enabling detected extensions
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCI = new()
        {
            queueFamilyIndex = _computeQueueFamily,
            queueCount = 1,
            pQueuePriorities = &queuePriority,
        };

        // Build null-terminated UTF-8 extension name arrays
        byte[] f16Int8NameBytes  = System.Text.Encoding.UTF8.GetBytes("VK_KHR_shader_float16_int8\0");
        byte[] storage16NameBytes = System.Text.Encoding.UTF8.GetBytes("VK_KHR_16bit_storage\0");
        byte[] storage8NameBytes  = System.Text.Encoding.UTF8.GetBytes("VK_KHR_8bit_storage\0");
        byte[] intDotNameBytes    = System.Text.Encoding.UTF8.GetBytes("VK_KHR_shader_integer_dot_product\0");
        byte[] bf16NameBytes      = System.Text.Encoding.UTF8.GetBytes("VK_KHR_shader_bfloat16\0");
        byte[] fp8NameBytes       = System.Text.Encoding.UTF8.GetBytes("VK_EXT_shader_float8\0");

        int enabledExtCount = (hasFloat16Int8 ? 1 : 0) + (has16BitStorage ? 1 : 0)
                            + (has8BitStorage ? 1 : 0) + (hasIntDot ? 1 : 0)
                            + (hasBfloat16 ? 1 : 0) + (hasFloat8 ? 1 : 0);
        int extIdx = 0;

        fixed (byte* pF16Int8   = f16Int8NameBytes,
                     pStorage16 = storage16NameBytes,
                     pStorage8  = storage8NameBytes,
                     pIntDot    = intDotNameBytes,
                     pBf16      = bf16NameBytes,
                     pFp8       = fp8NameBytes)
        {
            byte** extPtrs = stackalloc byte*[enabledExtCount > 0 ? enabledExtCount : 1];
            if (hasFloat16Int8)  extPtrs[extIdx++] = pF16Int8;
            if (has16BitStorage) extPtrs[extIdx++] = pStorage16;
            if (has8BitStorage)  extPtrs[extIdx++] = pStorage8;
            if (hasIntDot)       extPtrs[extIdx++] = pIntDot;
            if (hasBfloat16)     extPtrs[extIdx++] = pBf16;
            if (hasFloat8)       extPtrs[extIdx++] = pFp8;

            // Build pNext feature chain (back to front so earlier structs point to later ones)
            VkPhysicalDevice8BitStorageFeatures storage8Features = new()
            {
                storageBuffer8BitAccess = VkBool32.True,
            };
            VkPhysicalDevice16BitStorageFeatures storage16Features = new()
            {
                storageBuffer16BitAccess = VkBool32.True,
                pNext = has8BitStorage ? &storage8Features : null,
            };
            VkPhysicalDeviceShaderFloat16Int8Features f16Features = new()
            {
                shaderFloat16 = VkBool32.True,
                shaderInt8    = VkBool32.True,
                pNext = has16BitStorage ? &storage16Features : (has8BitStorage ? &storage8Features : null),
            };

            void* baseChain =
                hasFloat16Int8  ? (void*)&f16Features :
                has16BitStorage ? (void*)&storage16Features :
                has8BitStorage  ? (void*)&storage8Features :
                null;

            // Prepend fp8 and bf16 feature structs to the chain
            VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features = new()
            {
                shaderFloat8 = VkBool32.True,
                pNext = baseChain,
            };
            VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features = new()
            {
                shaderBFloat16Type = VkBool32.True,
                pNext = hasFloat8 ? (void*)&fp8Features : baseChain,
            };

            void* pNextChain =
                hasBfloat16 ? (void*)&bf16Features :
                hasFloat8   ? (void*)&fp8Features :
                baseChain;

            VkDeviceCreateInfo deviceCI = new()
            {
                pNext = pNextChain,
                queueCreateInfoCount = 1,
                pQueueCreateInfos = &queueCI,
                enabledExtensionCount = (uint)enabledExtCount,
                ppEnabledExtensionNames = enabledExtCount > 0 ? extPtrs : null,
            };
            _vki.vkCreateDevice(_physicalDevice, in deviceCI, out _device).CheckResult();
        }
        _vkd = new VkDeviceApi(_vki, in _device);

        // 7. Get the compute queue handle
        VkQueue queue;
        _vkd.vkGetDeviceQueue(_computeQueueFamily, 0, &queue);
        _computeQueue = queue;

        // 8. Create command pool + one reusable command buffer for transfers
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

        // 9. Create reusable fence for submission synchronization
        VkFenceCreateInfo fenceCI = new();
        VkFence fence;
        _vkd.vkCreateFence(&fenceCI, null, &fence).CheckResult();
        _fence = fence;

        // 10. Create a separate command pool + buffer + fence for background (prefetcher) uploads.
        VkCommandPoolCreateInfo asyncPoolCI = new()
        {
            flags = VkCommandPoolCreateFlags.ResetCommandBuffer,
            queueFamilyIndex = _computeQueueFamily,
        };
        VkCommandPool asyncPool;
        _vkd.vkCreateCommandPool(&asyncPoolCI, null, &asyncPool).CheckResult();
        _asyncPool = asyncPool;

        VkCommandBufferAllocateInfo asyncCmdAllocInfo = new()
        {
            commandPool = _asyncPool,
            level = VkCommandBufferLevel.Primary,
            commandBufferCount = 1,
        };
        VkCommandBuffer asyncCmd;
        _vkd.vkAllocateCommandBuffers(&asyncCmdAllocInfo, &asyncCmd).CheckResult();
        _asyncCmd = asyncCmd;

        VkFenceCreateInfo asyncFenceCI = new();
        VkFence asyncFence;
        _vkd.vkCreateFence(&asyncFenceCI, null, &asyncFence).CheckResult();
        _asyncFence = asyncFence;
    }

    // Capability flags detected at init (before device creation)
    public bool Has8BitStorage { get; private set; }
    public bool Has16BitStorage { get; private set; }
    public bool HasShaderFloat16Int8 { get; private set; }
    public bool HasCooperativeMatrix { get; private set; }
    public bool HasSubgroupSizeControl { get; private set; }
    public bool HasShaderIntegerDotProduct { get; private set; }
    public bool HasShaderBfloat16 { get; private set; }
    public bool HasShaderFloat8 { get; private set; }

    public SgemmPrecision BestSgemmPrecision =>
        HasShaderFloat16Int8 && Has16BitStorage ? SgemmPrecision.Fp16 :
        SgemmPrecision.Fp32;

    public bool SupportsGpuDequant => HasShaderFloat16Int8 && Has16BitStorage;

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

        var found = new List<string>();
        if (Has8BitStorage) found.Add("8bit_storage");
        if (Has16BitStorage) found.Add("16bit_storage");
        if (HasShaderFloat16Int8) found.Add("float16_int8");
        if (HasCooperativeMatrix) found.Add("cooperative_matrix");
        if (HasSubgroupSizeControl) found.Add("subgroup_size_control");
        if (HasShaderIntegerDotProduct) found.Add("integer_dot_product");
        if (HasShaderBfloat16) found.Add("bfloat16");
        if (HasShaderFloat8) found.Add("float8_e4m3");
        if (found.Count > 0)
            Console.WriteLine($"  Compute extensions: {string.Join(", ", found)}");
        Console.WriteLine($"  Best SGEMM precision: {BestSgemmPrecision}");
    }

    /// <summary>Total device-local (VRAM) heap size in bytes.</summary>
    public ulong VramBytes
    {
        get
        {
            ulong total = 0;
            for (int i = 0; i < (int)_memoryProperties.memoryHeapCount; i++)
            {
                if ((_memoryProperties.memoryHeaps[i].flags & VkMemoryHeapFlags.DeviceLocal) != 0)
                    total += _memoryProperties.memoryHeaps[i].size;
            }
            return total;
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

    private readonly ConcurrentDictionary<nint, GpuBuffer> _buffers = new();
    private long _nextHandle = 1;

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

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        if (_deferringFrees) { _deferredFrees.Add(tensor); return; }
        if (_buffers.TryRemove(tensor.Handle, out var buf))
            buf.Dispose();
    }

    /// <summary>
    /// Allocate a pinned host-visible buffer accessible from both CPU and GPU.
    /// The buffer can be mapped for CPU read/write and used as a GPU storage buffer.
    /// Ideal for small, frequently-transferred data like hidden states.
    /// </summary>
    public Tensor AllocatePinned(TensorShape shape, DType dtype = DType.Float32)
    {
        ulong byteSize = (ulong)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        var gpuBuf = GpuBuffer.CreatePinned(this, byteSize, VkBufferUsageFlags.StorageBuffer);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, dtype, handle);
    }

    /// <summary>Unmap a previously mapped pinned tensor.</summary>
    public unsafe float* MapPinned(Tensor tensor)
    {
        var buf = GetBuffer(tensor);
        return (float*)buf.Map();
    }

    /// <summary>Unmap a previously mapped pinned tensor.</summary>
    public void UnmapPinned(Tensor tensor)
    {
        var buf = GetBuffer(tensor);
        buf.Unmap();
    }

    // Cached staging buffer for uploads (avoids per-call alloc/free)
    private GpuBuffer? _uploadStaging;
    private ulong _uploadStagingSize;

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(float));

        // Create device-local destination buffer
        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        // Reuse or grow staging buffer
        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize,
                VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        // Map staging, copy data
        float* mapped = (float*)_uploadStaging.Map();
        data.CopyTo(new Span<float>(mapped, data.Length));
        _uploadStaging.Unmap();

        // Record and submit copy command
        CopyBuffer(_uploadStaging, gpuBuf, byteSize);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.Float32, handle);
    }

    /// <summary>
    /// Upload data to a new device-local GPU buffer using the background (async) command buffer.
    /// Safe to call from a background thread concurrently with the main thread's recording session.
    /// Uses a dedicated command pool isolated from the main-thread <c>_transferCmd</c>.
    /// </summary>
    public unsafe Tensor UploadBackground(ReadOnlySpan<float> data, TensorShape shape)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(float));

        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        lock (_asyncCmdLock)
        {
            if (_asyncStaging == null || _asyncStagingSize < byteSize)
            {
                _asyncStaging?.Dispose();
                _asyncStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
                _asyncStagingSize = byteSize;
            }

            float* mapped = (float*)_asyncStaging.Map();
            data.CopyTo(new Span<float>(mapped, data.Length));
            _asyncStaging.Unmap();

            VkCommandBufferBeginInfo beginInfo = new() { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
            _vkd.vkBeginCommandBuffer(_asyncCmd, &beginInfo).CheckResult();
            VkBufferCopy region = new() { size = byteSize };
            _vkd.vkCmdCopyBuffer(_asyncCmd, _asyncStaging.Buffer, gpuBuf.Buffer, 1, &region);
            _vkd.vkEndCommandBuffer(_asyncCmd).CheckResult();
            SubmitAndWaitAsync();
        }

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.Float32, handle);
    }

    // Cached staging buffer for downloads (avoids per-call alloc/free)
    private GpuBuffer? _downloadStaging;
    private ulong _downloadStagingSize;

    public void Download(Tensor src, Span<float> dst)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)(dst.Length * sizeof(float));

        // Reuse or grow staging buffer
        if (_downloadStaging == null || _downloadStagingSize < byteSize)
        {
            _downloadStaging?.Dispose();
            _downloadStaging = GpuBuffer.CreateStaging(this, byteSize,
                VkBufferUsageFlags.TransferDst);
            _downloadStagingSize = byteSize;
        }

        CopyBuffer(gpuBuf, _downloadStaging, byteSize);

        float* mapped = (float*)_downloadStaging.Map();
        new Span<float>(mapped, dst.Length).CopyTo(dst);
        _downloadStaging.Unmap();
    }

    public unsafe Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(ushort));

        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        Half* mapped = (Half*)_uploadStaging.Map();
        data.CopyTo(new Span<Half>(mapped, data.Length));
        _uploadStaging.Unmap();

        CopyBuffer(_uploadStaging, gpuBuf, byteSize);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.Float16, handle);
    }

    public unsafe void DownloadHalf(Tensor src, Span<Half> dst)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)(dst.Length * sizeof(ushort));

        if (_downloadStaging == null || _downloadStagingSize < byteSize)
        {
            _downloadStaging?.Dispose();
            _downloadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferDst);
            _downloadStagingSize = byteSize;
        }

        CopyBuffer(gpuBuf, _downloadStaging, byteSize);

        Half* mapped = (Half*)_downloadStaging.Map();
        new ReadOnlySpan<Half>(mapped, dst.Length).CopyTo(dst);
        _downloadStaging.Unmap();
    }

    public unsafe Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(ushort));

        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        ushort* mapped = (ushort*)_uploadStaging.Map();
        data.CopyTo(new Span<ushort>(mapped, data.Length));
        _uploadStaging.Unmap();

        CopyBuffer(_uploadStaging, gpuBuf, byteSize);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.BFloat16, handle);
    }

    public unsafe void DownloadBf16(Tensor src, Span<ushort> dst)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)(dst.Length * sizeof(ushort));

        if (_downloadStaging == null || _downloadStagingSize < byteSize)
        {
            _downloadStaging?.Dispose();
            _downloadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferDst);
            _downloadStagingSize = byteSize;
        }

        CopyBuffer(gpuBuf, _downloadStaging, byteSize);

        ushort* mapped = (ushort*)_downloadStaging.Map();
        new ReadOnlySpan<ushort>(mapped, dst.Length).CopyTo(dst);
        _downloadStaging.Unmap();
    }

    public unsafe Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape)
    {
        ulong byteSize = (ulong)data.Length;

        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        byte* mapped = (byte*)_uploadStaging.Map();
        data.CopyTo(new Span<byte>(mapped, data.Length));
        _uploadStaging.Unmap();

        CopyBuffer(_uploadStaging, gpuBuf, byteSize);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, DType.Float8E4M3, handle);
    }

    public unsafe void DownloadFp8(Tensor src, Span<byte> dst)
    {
        var gpuBuf = GetBuffer(src);
        ulong byteSize = (ulong)dst.Length;

        if (_downloadStaging == null || _downloadStagingSize < byteSize)
        {
            _downloadStaging?.Dispose();
            _downloadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferDst);
            _downloadStagingSize = byteSize;
        }

        CopyBuffer(gpuBuf, _downloadStaging, byteSize);

        byte* mapped = (byte*)_downloadStaging.Map();
        new ReadOnlySpan<byte>(mapped, dst.Length).CopyTo(dst);
        _downloadStaging.Unmap();
    }

    /// <summary>
    /// Upload raw quantized bytes to a device-local GPU buffer.
    /// The returned tensor's shape is D1(byteLen) and its DType reflects the quantized format.
    /// </summary>
    public unsafe Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype)
    {
        ulong byteSize = (ulong)data.Length;

        var gpuBuf = GpuBuffer.CreateDeviceLocal(this, byteSize,
            VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferDst);

        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        byte* mapped = (byte*)_uploadStaging.Map();
        data.CopyTo(new Span<byte>(mapped, data.Length));
        _uploadStaging.Unmap();

        CopyBuffer(_uploadStaging, gpuBuf, byteSize);

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, dtype, handle);
    }

    public VkFence Fence => _fence;

    public void Synchronize()
    {
        _vkd.vkDeviceWaitIdle();
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
        SubmitAndWait();
    }

    // ================================================================
    //  Compute shader pipelines (created lazily on first use)
    // ================================================================

    private ComputePipeline? _rmsNormPipeline;
    private ComputePipeline? _headNormPipeline;
    private ComputePipeline? _headNormPurePipeline;
    private ComputePipeline? _siluMulPipeline;
    private ComputePipeline? _addInPlacePipeline;
    private ComputePipeline? _addScaledInPlacePipeline;
    private ComputePipeline? _scaleInPlacePipeline;
    private ComputePipeline? _clearPipeline;
    private ComputePipeline? _elementwiseMulPipeline;
    private ComputePipeline? _ropePipeline;
    private ComputePipeline? _ropeNeoxPipeline;
    private ComputePipeline? _softmaxPipeline;
    private ComputePipeline? _sigmoidPipeline;
    private ComputePipeline? _matVecQ4KPipeline;
    private ComputePipeline? _matVecQ6KPipeline;
    private ComputePipeline? _matVecF32Pipeline;
    private ComputePipeline? _kvAppendPipeline;
    private ComputePipeline? _attentionPipeline;
    private ComputePipeline? _embedLookupPipeline;
    private ComputePipeline? _embedLookupQ4KPipeline;
    private ComputePipeline? _tqRotateQueryPipeline;
    private ComputePipeline? _tqKvAppendPipeline;
    private ComputePipeline? _tqAttentionPipeline;
    private ComputePipeline? _bufCopyPipeline;
    private ComputePipeline? _sgemmF32Pipeline;
    private ComputePipeline? _sgemmF16Pipeline;
    private ComputePipeline? _sgemmBf16Pipeline;
    private ComputePipeline? _sgemmFp8Pipeline;
    private ComputePipeline? _dequantQ5KMPipeline;
    private ComputePipeline? _dequantQ4KMPipeline;

    // Image ops pipelines (IImageOpsBackend)
    private ComputePipeline? _conv2dPipeline;
    private ComputePipeline? _leakyReluPipeline;
    private ComputePipeline? _clampPipeline;
    private ComputePipeline? _catChannelsPipeline;
    private ComputePipeline? _pixelShufflePipeline;
    private ComputePipeline? _pixelUnshufflePipeline;
    private ComputePipeline? _upsample2xPipeline;

    private struct RmsNormParams{ public uint n; public float eps; }
    private struct HeadNormParams { public uint headDim; public uint numHeads; public float eps; }
    private struct CountParams { public uint n; }
    private struct ScaleParams { public uint n; public float scale; }
    private struct RoPEParams { public uint numHeads; public uint headDim; public int position; public float theta; }
    private struct MatVecParams { public uint rows; public uint cols; }
    private struct EmbedParams { public uint tokenId; public uint embDim; }
    private struct KvAppendParams { public uint kvDim; public uint position; public uint maxSeqLen; }
    private struct AttentionParams { public uint numHeads; public uint numKvHeads; public uint headDim; public uint seqLen; public uint maxSeqLen; }
    private struct TqRotateQueryParams { public uint numHeads; public uint numKvHeads; public uint headDim; }
    private struct TqKvAppendParams { public uint kvDim; public uint headDim; public uint position; public uint maxSeqLen; public uint numKvHeads; public uint blockBytes; }
    private struct TqAttentionParams { public uint numHeads; public uint numKvHeads; public uint headDim; public uint tqSeqLen; public uint fp16SeqLen; public uint maxSeqLen; public uint blockBytes; }
    private struct BufCopyParams { public uint count; public uint srcOffset; public uint dstOffset; }
    private struct SgemmParams { public uint M; public uint N; public uint K; }
    private struct DequantParams { public uint numBlocks; }

    // Image ops push constant structs
    private struct Conv2dParams   { public uint inCh; public uint outCh; public uint height; public uint width; public uint ksize; public uint padding; }
    private struct LeakyReluParams { public uint n; public float negSlope; }
    private struct ClampParams    { public uint n; public float minVal; public float maxVal; }
    private struct CatChannelsParams { public uint aCh; public uint bCh; public uint hw; }
    private struct PixelShuffleParams { public uint inCh; public uint h; public uint w; public uint factor; }
    private struct SpatialParams  { public uint ch; public uint h; public uint w; }

    private void DispatchOrRecord(ComputePipeline pipe, ReadOnlySpan<GpuBuffer> buffers,
        uint groupX, void* push, uint groupY = 1, uint groupZ = 1)
    {
        if (_recording)
        {
            pipe.RecordWith(_transferCmd, buffers, groupX, groupY, groupZ, push);
            // In image-batch mode (_deferringFrees), automatically insert a compute barrier
            // after every dispatch so dependent ops see the writes without explicit caller code.
            if (_deferringFrees) RecordBarrier();
        }
        else
            pipe.DispatchWith(_transferCmd, buffers, groupX, groupY, groupZ, push);
    }

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        _rmsNormPipeline ??= new ComputePipeline(this, Shaders.RmsNorm, 3, pushConstantSize: sizeof(RmsNormParams));
        var p = new RmsNormParams { n = (uint)x.ElementCount, eps = eps };
        DispatchOrRecord(_rmsNormPipeline, [GetBuffer(x), GetBuffer(weight), GetBuffer(output)], 1, &p);
    }

    public void HeadNorm(Tensor data, Tensor weight, uint numHeads, uint headDim, float eps = 1e-6f)
    {
        _headNormPipeline ??= new ComputePipeline(this, Shaders.HeadNorm, 2, pushConstantSize: sizeof(HeadNormParams));
        var p = new HeadNormParams { headDim = headDim, numHeads = numHeads, eps = eps };
        DispatchOrRecord(_headNormPipeline, [GetBuffer(data), GetBuffer(weight)], numHeads, &p);
    }

    /// <summary>
    /// Per-head RMS normalization without learned weights (L2 normalize).
    /// Used for Llama4TextL2Norm in QK-norm.
    /// </summary>
    public void HeadNormPure(Tensor data, uint numHeads, uint headDim, float eps = 1e-6f)
    {
        _headNormPurePipeline ??= new ComputePipeline(this, Shaders.HeadNormPure, 1, pushConstantSize: sizeof(HeadNormParams));
        var p = new HeadNormParams { headDim = headDim, numHeads = numHeads, eps = eps };
        DispatchOrRecord(_headNormPurePipeline, [GetBuffer(data)], numHeads, &p);
    }

    public void SiLU(Tensor x) => throw new NotImplementedException("Use SiLuMul for fused SiLU*gate");

    public void SiLuMul(Tensor gate, Tensor up)
    {
        _siluMulPipeline ??= new ComputePipeline(this, Shaders.SiLuMul, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)gate.ElementCount };
        DispatchOrRecord(_siluMulPipeline, [GetBuffer(gate), GetBuffer(up)], ((uint)gate.ElementCount + 255) / 256, &p);
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        _addInPlacePipeline ??= new ComputePipeline(this, Shaders.AddInPlace, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)dst.ElementCount };
        DispatchOrRecord(_addInPlacePipeline, [GetBuffer(dst), GetBuffer(src)], ((uint)dst.ElementCount + 255) / 256, &p);
    }

    public void AddScaledInPlace(Tensor dst, Tensor src, float scale)
    {
        _addScaledInPlacePipeline ??= new ComputePipeline(this, Shaders.AddScaledInPlace, 2, pushConstantSize: sizeof(ScaleParams));
        var p = new ScaleParams { n = (uint)dst.ElementCount, scale = scale };
        DispatchOrRecord(_addScaledInPlacePipeline, [GetBuffer(dst), GetBuffer(src)], ((uint)dst.ElementCount + 255) / 256, &p);
    }

    public void ScaleInPlace(Tensor x, float scale)
    {
        _scaleInPlacePipeline ??= new ComputePipeline(this, Shaders.ScaleInPlace, 1, pushConstantSize: sizeof(ScaleParams));
        var p = new ScaleParams { n = (uint)x.ElementCount, scale = scale };
        DispatchOrRecord(_scaleInPlacePipeline, [GetBuffer(x)], ((uint)x.ElementCount + 255) / 256, &p);
    }

    /// <summary>Copy an entire device-local tensor using a compute shader (stays in compute pipeline stage).</summary>
    public void RecordComputeCopy(Tensor dst, Tensor src)
    {
        var srcBuf = GetBuffer(src);
        RecordComputeCopyWords(GetBuffer(dst), 0, srcBuf, 0, (uint)(srcBuf.Size / 4));
    }

    /// <summary>Copy a sub-region between device-local tensors using a compute shader.</summary>
    public void RecordComputeCopyRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
    {
        RecordComputeCopyWords(GetBuffer(dst), (uint)(dstOffsetBytes / 4),
                               GetBuffer(src), (uint)(srcOffsetBytes / 4),
                               (uint)(sizeBytes / 4));
    }

    private void RecordComputeCopyWords(GpuBuffer dst, uint dstOffset, GpuBuffer src, uint srcOffset, uint wordCount)
    {
        _bufCopyPipeline ??= new ComputePipeline(this, Shaders.BufferCopy, 2, pushConstantSize: sizeof(BufCopyParams));
        var p = new BufCopyParams { count = wordCount, srcOffset = srcOffset, dstOffset = dstOffset };
        DispatchOrRecord(_bufCopyPipeline, [src, dst], (wordCount + 255) / 256, &p);
    }

    public void Clear(Tensor dst)
    {
        _clearPipeline ??= new ComputePipeline(this, Shaders.Clear, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)dst.ElementCount };
        DispatchOrRecord(_clearPipeline, [GetBuffer(dst)], ((uint)dst.ElementCount + 255) / 256, &p);
    }

    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        _elementwiseMulPipeline ??= new ComputePipeline(this, Shaders.ElementwiseMul, 3, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)a.ElementCount };
        DispatchOrRecord(_elementwiseMulPipeline, [GetBuffer(a), GetBuffer(b), GetBuffer(output)], ((uint)a.ElementCount + 255) / 256, &p);
    }

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f, bool neox = false)
    {
        ComputePipeline pipeline;
        if (neox)
        {
            _ropeNeoxPipeline ??= new ComputePipeline(this, Shaders.RoPENeox, 1, pushConstantSize: sizeof(RoPEParams));
            pipeline = _ropeNeoxPipeline;
        }
        else
        {
            _ropePipeline ??= new ComputePipeline(this, Shaders.RoPE, 1, pushConstantSize: sizeof(RoPEParams));
            pipeline = _ropePipeline;
        }
        uint numHeads = (uint)(x.ElementCount / headDim);
        uint totalPairs = numHeads * (uint)(headDim / 2);
        var p = new RoPEParams { numHeads = numHeads, headDim = (uint)headDim, position = position, theta = ropeTheta };
        DispatchOrRecord(pipeline, [GetBuffer(x)], (totalPairs + 255) / 256, &p);
    }

    public void Softmax(Tensor x)
    {
        _softmaxPipeline ??= new ComputePipeline(this, Shaders.Softmax, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)x.ElementCount };
        DispatchOrRecord(_softmaxPipeline, [GetBuffer(x)], 1, &p);
    }

    public void Sigmoid(Tensor x)
    {
        _sigmoidPipeline ??= new ComputePipeline(this, Shaders.Sigmoid, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)x.ElementCount };
        DispatchOrRecord(_sigmoidPipeline, [GetBuffer(x)], ((uint)x.ElementCount + 255) / 256, &p);
    }

    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        // Default: assume Q4_K weights
        MatMul(output, matrix, vector, DType.Q4_K);
    }

    public void MatMul(Tensor output, Tensor matrix, Tensor vector, DType weightDType)
    {
        var p = new MatVecParams { rows = (uint)output.ElementCount, cols = (uint)vector.ElementCount };
        var bufs = (ReadOnlySpan<GpuBuffer>)[GetBuffer(matrix), GetBuffer(vector), GetBuffer(output)];
        uint totalRows = (uint)output.ElementCount;

        switch (weightDType)
        {
            case DType.Float32:
                _matVecF32Pipeline ??= new ComputePipeline(this, Shaders.MatVecF32, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecF32Pipeline, bufs, (totalRows + 7) / 8, &p);
                break;
            case DType.Q6_K:
                _matVecQ6KPipeline ??= new ComputePipeline(this, Shaders.MatVecQ6K, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ6KPipeline, bufs, (totalRows + 7) / 8, &p);
                break;
            default: // Q4_K — 256 threads, 8 rows per workgroup, subgroupAdd reduction
                _matVecQ4KPipeline ??= new ComputePipeline(this, Shaders.MatVecQ4K, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ4KPipeline, bufs, (totalRows + 7) / 8, &p);
                break;
        }
    }

    // ================================================================
    //  KV Cache + Attention (GPU-resident)
    // ================================================================

    public void EmbedLookup(Tensor embTable, Tensor output, uint tokenId, uint embDim)
    {
        _embedLookupPipeline ??= new ComputePipeline(this, Shaders.EmbedLookup, 2, pushConstantSize: sizeof(EmbedParams));
        var p = new EmbedParams { tokenId = tokenId, embDim = embDim };
        DispatchOrRecord(_embedLookupPipeline, [GetBuffer(embTable), GetBuffer(output)], (embDim + 255) / 256, &p);
    }

    public void EmbedLookupQ4K(Tensor embTable, Tensor output, uint tokenId, uint embDim)
    {
        _embedLookupQ4KPipeline ??= new ComputePipeline(this, Shaders.EmbedLookupQ4K, 2, pushConstantSize: sizeof(EmbedParams));
        var p = new EmbedParams { tokenId = tokenId, embDim = embDim };
        DispatchOrRecord(_embedLookupQ4KPipeline, [GetBuffer(embTable), GetBuffer(output)], 1, &p);
    }

    public void KvAppend(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
        uint kvDim, uint position, uint maxSeqLen)
    {
        _kvAppendPipeline ??= new ComputePipeline(this, Shaders.KvAppend, 4, pushConstantSize: sizeof(KvAppendParams));
        var p = new KvAppendParams { kvDim = kvDim, position = position, maxSeqLen = maxSeqLen };
        DispatchOrRecord(_kvAppendPipeline,
            [GetBuffer(kInput), GetBuffer(vInput), GetBuffer(kCache), GetBuffer(vCache)],
            (kvDim + 255) / 256, &p);
    }

    /// <summary>
    /// Scaled dot-product attention with GQA support. <paramref name="scoresScratch"/> is a
    /// VRAM buffer the kernel spills per-position softmax scores into when <c>seqLen &gt; 4096</c>;
    /// the fast path uses shared memory instead. Vulkan descriptors require a bound buffer
    /// regardless, so callers pass a 1-float placeholder when the whole context is guaranteed
    /// to fit in shared memory.
    /// </summary>
    public void Attention(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor scoresScratch,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen)
    {
        _attentionPipeline ??= new ComputePipeline(this, Shaders.Attention, 5, pushConstantSize: sizeof(AttentionParams));
        var p = new AttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, maxSeqLen = maxSeqLen
        };
        DispatchOrRecord(_attentionPipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(output),
             GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    // ================================================================
    //  TurboQuant KV Cache Operations
    // ================================================================

    public void TqRotateQuery(Tensor qInput, Tensor rotatedQ, Tensor signPatterns,
        uint numHeads, uint numKvHeads, uint headDim)
    {
        _tqRotateQueryPipeline ??= new ComputePipeline(this, Shaders.TqRotateQuery, 3, pushConstantSize: sizeof(TqRotateQueryParams));
        var p = new TqRotateQueryParams { numHeads = numHeads, numKvHeads = numKvHeads, headDim = headDim };
        DispatchOrRecord(_tqRotateQueryPipeline,
            [GetBuffer(qInput), GetBuffer(rotatedQ), GetBuffer(signPatterns)],
            numHeads, &p);
    }

    public void TqKvAppend(Tensor kInput, Tensor vInput, Tensor kCacheTq, Tensor vCacheTq,
        Tensor signPatterns, Tensor codebook, Tensor boundaries,
        uint kvDim, uint headDim, uint position, uint maxSeqLen, uint numKvHeads, uint blockBytes)
    {
        _tqKvAppendPipeline ??= new ComputePipeline(this, Shaders.TqKvAppend, 7, pushConstantSize: sizeof(TqKvAppendParams));
        var p = new TqKvAppendParams
        {
            kvDim = kvDim, headDim = headDim, position = position,
            maxSeqLen = maxSeqLen, numKvHeads = numKvHeads, blockBytes = blockBytes
        };
        DispatchOrRecord(_tqKvAppendPipeline,
            [GetBuffer(kInput), GetBuffer(vInput), GetBuffer(kCacheTq), GetBuffer(vCacheTq),
             GetBuffer(signPatterns), GetBuffer(codebook), GetBuffer(boundaries)],
            numKvHeads, &p);
    }

    /// <summary>
    /// Hybrid TQ + FP16 attention. <paramref name="scoresScratch"/> is a VRAM
    /// buffer the kernel spills per-position softmax scores into when
    /// <c>tqSeqLen + fp16SeqLen &gt; 4096</c>; the fast path uses shared memory
    /// instead. Vulkan descriptors require a bound buffer regardless of which
    /// path runs, so callers pass a 1-float placeholder when the whole context
    /// is guaranteed to fit in shared memory.
    /// </summary>
    public void TqAttention(Tensor q, Tensor rotatedQ, Tensor kCacheTq, Tensor vCacheTq,
        Tensor kCacheFp16, Tensor vCacheFp16, Tensor output, Tensor codebook,
        Tensor scoresScratch,
        uint numHeads, uint numKvHeads, uint headDim,
        uint tqSeqLen, uint fp16SeqLen, uint maxSeqLen, uint blockBytes)
    {
        _tqAttentionPipeline ??= new ComputePipeline(this, Shaders.TqAttention, 9, pushConstantSize: sizeof(TqAttentionParams));
        var p = new TqAttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads, headDim = headDim,
            tqSeqLen = tqSeqLen, fp16SeqLen = fp16SeqLen, maxSeqLen = maxSeqLen, blockBytes = blockBytes
        };
        DispatchOrRecord(_tqAttentionPipeline,
            [GetBuffer(q), GetBuffer(rotatedQ), GetBuffer(kCacheTq), GetBuffer(vCacheTq),
             GetBuffer(kCacheFp16), GetBuffer(vCacheFp16), GetBuffer(output), GetBuffer(codebook),
             GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    // ================================================================
    //  DiT / Diffusion — batched GEMM and full-sequence attention
    // ================================================================

    /// <summary>
    /// Tiled GEMM: C[M,N] = A[M,K] × B[N,K]^T.
    /// Dispatches the best available precision shader (fp8 > bf16 > fp16 > fp32).
    /// A, B, C must already be GPU-resident tensors.
    /// </summary>
    public unsafe void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N)
    {
        var p = new SgemmParams { M = (uint)M, N = (uint)N, K = (uint)K };
        uint gx = ((uint)M + 15u) / 16u;
        uint gy = ((uint)N + 15u) / 16u;

        if (A.DType == DType.Float8E4M3 && B.DType == DType.Float8E4M3 && HasShaderFloat8)
        {
            try
            {
                _sgemmFp8Pipeline ??= new ComputePipeline(this, Shaders.SgemmFp8, 3,
                    pushConstantSize: sizeof(SgemmParams));
                DispatchOrRecord(_sgemmFp8Pipeline, [GetBuffer(A), GetBuffer(B), GetBuffer(C)], gx, &p, gy);
                return;
            }
            catch (Exception)
            {
                HasShaderFloat8 = false;
                _sgemmFp8Pipeline?.Dispose();
                _sgemmFp8Pipeline = null;
            }
        }

        if (A.DType == DType.BFloat16 && HasShaderBfloat16)
        {
            try
            {
                _sgemmBf16Pipeline ??= new ComputePipeline(this, Shaders.SgemmBf16, 3,
                    pushConstantSize: sizeof(SgemmParams));
                DispatchOrRecord(_sgemmBf16Pipeline, [GetBuffer(A), GetBuffer(B), GetBuffer(C)], gx, &p, gy);
                return;
            }
            catch (Exception)
            {
                HasShaderBfloat16 = false;
                _sgemmBf16Pipeline?.Dispose();
                _sgemmBf16Pipeline = null;
            }
        }

        if (A.DType == DType.Float32 && B.DType == DType.Float16 && HasShaderFloat16Int8 && Has16BitStorage)
        {
            _sgemmF16Pipeline ??= new ComputePipeline(this, Shaders.SgemmF16, 3,
                pushConstantSize: sizeof(SgemmParams));
            DispatchOrRecord(_sgemmF16Pipeline, [GetBuffer(A), GetBuffer(B), GetBuffer(C)], gx, &p, gy);
            return;
        }

        _sgemmF32Pipeline ??= new ComputePipeline(this, Shaders.SgemmF32, 3,
            pushConstantSize: sizeof(SgemmParams));
        DispatchOrRecord(_sgemmF32Pipeline, [GetBuffer(A), GetBuffer(B), GetBuffer(C)], gx, &p, gy);
    }

    /// <summary>GPU-side dequantize Q5_K raw bytes → fp16 output.</summary>
    public unsafe void DequantQ5KM(Tensor src, Tensor dst, int numBlocks)
    {
        try
        {
            _dequantQ5KMPipeline ??= new ComputePipeline(this, Shaders.DequantQ5KM, 2,
                pushConstantSize: sizeof(DequantParams));
            var p = new DequantParams { numBlocks = (uint)numBlocks };
            DispatchOrRecord(_dequantQ5KMPipeline, [GetBuffer(src), GetBuffer(dst)], (uint)numBlocks, &p);
        }
        catch (Exception)
        {
            _dequantQ5KMPipeline?.Dispose();
            _dequantQ5KMPipeline = null;
            throw;
        }
    }

    /// <summary>GPU-side dequantize Q4_K raw bytes → fp16 output.</summary>
    public unsafe void DequantQ4KM(Tensor src, Tensor dst, int numBlocks)
    {
        try
        {
            _dequantQ4KMPipeline ??= new ComputePipeline(this, Shaders.DequantQ4KM, 2,
                pushConstantSize: sizeof(DequantParams));
            var p = new DequantParams { numBlocks = (uint)numBlocks };
            DispatchOrRecord(_dequantQ4KMPipeline, [GetBuffer(src), GetBuffer(dst)], (uint)numBlocks, &p);
        }
        catch (Exception)
        {
            _dequantQ4KMPipeline?.Dispose();
            _dequantQ4KMPipeline = null;
            throw;
        }
    }

    /// <summary>
    /// Full-sequence self-attention computed on CPU (attention is ~1% of DiT FLOPs).
    /// Downloads Q/K/V from GPU, runs the attention, uploads the result.
    /// Layout: element (tok, head, d) at index tok*nHeads*headDim + head*headDim + d.
    /// </summary>
    public unsafe void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                                        int nTok, int nHeads, int headDim, float scale)
    {
        int dim = nHeads * headDim;
        int count = nTok * dim;
        float[] qHost = new float[count];
        float[] kHost = new float[count];
        float[] vHost = new float[count];
        float[] oHost = new float[count];
        Download(q, qHost);
        Download(k, kHost);
        Download(v, vHost);

        float[] scoresBuf = new float[nHeads * nTok * nTok];
        float[] vhBuf     = new float[nTok * headDim];

        fixed (float* qPtr = qHost, kPtr = kHost, vPtr = vHost, oPtr = oHost,
                      sBuf = scoresBuf, vhPtr = vhBuf)
        {
            for (int h = 0; h < nHeads; h++)
            {
                int sBase = h * nTok * nTok;
                for (int i = 0; i < nTok; i++)
                {
                    float* qi = qPtr + ((long)i * nHeads + h) * headDim;
                    int sRow = sBase + i * nTok;
                    for (int j = 0; j < nTok; j++)
                    {
                        float* kj = kPtr + ((long)j * nHeads + h) * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += qi[d] * kj[d];
                        sBuf[sRow + j] = dot * scale;
                    }
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < nTok; j++)
                        if (sBuf[sRow + j] > max) max = sBuf[sRow + j];
                    float sum = 0f;
                    for (int j = 0; j < nTok; j++)
                    {
                        sBuf[sRow + j] = MathF.Exp(sBuf[sRow + j] - max);
                        sum += sBuf[sRow + j];
                    }
                    float invSum = 1f / sum;
                    for (int j = 0; j < nTok; j++)
                        sBuf[sRow + j] *= invSum;
                }

                // Gather V for this head
                for (int j = 0; j < nTok; j++)
                {
                    float* src = vPtr + ((long)j * nHeads + h) * headDim;
                    float* dst = vhPtr + j * headDim;
                    for (int d = 0; d < headDim; d++)
                        dst[d] = src[d];
                }
                // Weighted sum into output
                for (int i = 0; i < nTok; i++)
                {
                    int sRow = sBase + i * nTok;
                    float* outRow = oPtr + ((long)i * nHeads + h) * headDim;
                    for (int d = 0; d < headDim; d++)
                        outRow[d] = 0f;
                    for (int j = 0; j < nTok; j++)
                    {
                        float w = sBuf[sRow + j];
                        float* vj = vhPtr + j * headDim;
                        for (int d = 0; d < headDim; d++)
                            outRow[d] += w * vj[d];
                    }
                }
            }
        }

        // Upload result into the pre-allocated output tensor via staging copy
        ulong byteSize = (ulong)((long)count * sizeof(float));
        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }
        fixed (float* src = oHost)
        {
            float* mapped = (float*)_uploadStaging.Map();
            new System.Span<float>(src, count).CopyTo(new System.Span<float>(mapped, count));
            _uploadStaging.Unmap();
        }
        CopyBuffer(_uploadStaging, GetBuffer(output), byteSize);
    }

    // ================================================================
    //  IImageOpsBackend — GPU-native image operations for RRDBNet
    // ================================================================

    public Tensor Conv2d(Tensor input, Tensor weight, Tensor bias,
                         int inCh, int outCh, int h, int w, int ksize, int padding = -1)
    {
        if (padding < 0) padding = ksize / 2;
        var output = Allocate(TensorShape.D1(outCh * h * w));
        _conv2dPipeline ??= new ComputePipeline(this, Shaders.Conv2d, 4, pushConstantSize: sizeof(Conv2dParams));
        var p = new Conv2dParams { inCh = (uint)inCh, outCh = (uint)outCh, height = (uint)h, width = (uint)w, ksize = (uint)ksize, padding = (uint)padding };
        // 2D dispatch: X=outCh (one workgroup per channel), Y=ceil(H*W/256) (spatial tiles).
        // All 256 threads per workgroup share the same output channel → cooperative weight
        // loading into shared memory eliminates 256× redundant global weight reads.
        uint groupY = ((uint)(h * w) + 255u) / 256u;
        DispatchOrRecord(_conv2dPipeline, [GetBuffer(input), GetBuffer(weight), GetBuffer(bias), GetBuffer(output)], (uint)outCh, &p, groupY);
        return output;
    }

    public void LeakyReluInPlace(Tensor x, float negSlope)
    {
        _leakyReluPipeline ??= new ComputePipeline(this, Shaders.LeakyRelu, 1, pushConstantSize: sizeof(LeakyReluParams));
        var p = new LeakyReluParams { n = (uint)x.ElementCount, negSlope = negSlope };
        uint groups = ((uint)x.ElementCount + 255u) / 256u;
        DispatchOrRecord(_leakyReluPipeline, [GetBuffer(x)], groups, &p);
    }

    public void ClampInPlace(Tensor x, float min, float max)
    {
        _clampPipeline ??= new ComputePipeline(this, Shaders.ClampInPlace, 1, pushConstantSize: sizeof(ClampParams));
        var p = new ClampParams { n = (uint)x.ElementCount, minVal = min, maxVal = max };
        uint groups = ((uint)x.ElementCount + 255u) / 256u;
        DispatchOrRecord(_clampPipeline, [GetBuffer(x)], groups, &p);
    }

    public Tensor CatChannels(Tensor a, int aCh, Tensor b, int bCh, int hw)
    {
        var output = Allocate(TensorShape.D1((aCh + bCh) * hw));
        _catChannelsPipeline ??= new ComputePipeline(this, Shaders.CatChannels, 3, pushConstantSize: sizeof(CatChannelsParams));
        var p = new CatChannelsParams { aCh = (uint)aCh, bCh = (uint)bCh, hw = (uint)hw };
        uint groups = ((uint)((aCh + bCh) * hw) + 255u) / 256u;
        DispatchOrRecord(_catChannelsPipeline, [GetBuffer(a), GetBuffer(b), GetBuffer(output)], groups, &p);
        return output;
    }

    public Tensor PixelShuffleGpu(Tensor input, int inCh, int h, int w, int upscaleFactor)
    {
        int outCh = inCh / (upscaleFactor * upscaleFactor);
        var output = Allocate(TensorShape.D1(outCh * h * upscaleFactor * w * upscaleFactor));
        _pixelShufflePipeline ??= new ComputePipeline(this, Shaders.PixelShuffle, 2, pushConstantSize: sizeof(PixelShuffleParams));
        var p = new PixelShuffleParams { inCh = (uint)inCh, h = (uint)h, w = (uint)w, factor = (uint)upscaleFactor };
        uint total = (uint)(outCh * h * upscaleFactor * w * upscaleFactor);
        uint groups = (total + 255u) / 256u;
        DispatchOrRecord(_pixelShufflePipeline, [GetBuffer(input), GetBuffer(output)], groups, &p);
        return output;
    }

    public Tensor PixelUnshuffleGpu(Tensor input, int inCh, int h, int w, int downscaleFactor)
    {
        int outCh = inCh * downscaleFactor * downscaleFactor;
        int oh = h / downscaleFactor;
        int ow = w / downscaleFactor;
        var output = Allocate(TensorShape.D1(outCh * oh * ow));
        _pixelUnshufflePipeline ??= new ComputePipeline(this, Shaders.PixelUnshuffle, 2, pushConstantSize: sizeof(PixelShuffleParams));
        var p = new PixelShuffleParams { inCh = (uint)inCh, h = (uint)oh, w = (uint)ow, factor = (uint)downscaleFactor };
        uint total = (uint)(outCh * oh * ow);
        uint groups = (total + 255u) / 256u;
        DispatchOrRecord(_pixelUnshufflePipeline, [GetBuffer(input), GetBuffer(output)], groups, &p);
        return output;
    }

    public Tensor Upsample2xGpu(Tensor input, int ch, int h, int w)
    {
        var output = Allocate(TensorShape.D1(ch * h * 2 * w * 2));
        _upsample2xPipeline ??= new ComputePipeline(this, Shaders.Upsample2xNearest, 2, pushConstantSize: sizeof(SpatialParams));
        var p = new SpatialParams { ch = (uint)ch, h = (uint)h, w = (uint)w };
        uint groups = ((uint)(ch * h * 2 * w * 2) + 255u) / 256u;
        DispatchOrRecord(_upsample2xPipeline, [GetBuffer(input), GetBuffer(output)], groups, &p);
        return output;
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
        _headNormPipeline?.Dispose();
        _headNormPurePipeline?.Dispose();
        _siluMulPipeline?.Dispose();
        _addInPlacePipeline?.Dispose();
        _addScaledInPlacePipeline?.Dispose();
        _scaleInPlacePipeline?.Dispose();
        _clearPipeline?.Dispose();
        _elementwiseMulPipeline?.Dispose();
        _ropePipeline?.Dispose();
        _ropeNeoxPipeline?.Dispose();
        _softmaxPipeline?.Dispose();
        _sigmoidPipeline?.Dispose();
        _matVecQ4KPipeline?.Dispose();
        _matVecQ6KPipeline?.Dispose();
        _matVecF32Pipeline?.Dispose();
        _kvAppendPipeline?.Dispose();
        _attentionPipeline?.Dispose();
        _embedLookupPipeline?.Dispose();
        _embedLookupQ4KPipeline?.Dispose();
        _bufCopyPipeline?.Dispose();
        _sgemmF32Pipeline?.Dispose();
        _sgemmF16Pipeline?.Dispose();
        _sgemmBf16Pipeline?.Dispose();
        _sgemmFp8Pipeline?.Dispose();
        _dequantQ5KMPipeline?.Dispose();
        _dequantQ4KMPipeline?.Dispose();
        _conv2dPipeline?.Dispose();
        _leakyReluPipeline?.Dispose();
        _clampPipeline?.Dispose();
        _catChannelsPipeline?.Dispose();
        _pixelShufflePipeline?.Dispose();
        _pixelUnshufflePipeline?.Dispose();
        _upsample2xPipeline?.Dispose();

        _downloadStaging?.Dispose();
        _uploadStaging?.Dispose();
        _asyncStaging?.Dispose();

        // Free all tracked GPU buffers
        foreach (var buf in _buffers.Values)
            buf.Dispose();
        _buffers.Clear();

        _vkd.vkDestroyFence(_fence, null);
        _vkd.vkDestroyFence(_asyncFence, null);
        _vkd.vkDestroyCommandPool(_commandPool, null);
        _vkd.vkDestroyCommandPool(_asyncPool, null);
        _vkd.vkDestroyDevice(null);
        _vki.vkDestroyInstance(null);
    }
}
