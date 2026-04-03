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
    private readonly VkFence _fence; // reusable fence for submission synchronization
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

    /// <summary>Begin recording a batch of compute dispatches.</summary>
    public void BeginRecord()
    {
        VkCommandBufferBeginInfo begin = new() { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
        _vkd.vkBeginCommandBuffer(_transferCmd, &begin).CheckResult();
        _recording = true;
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

    /// <summary>Submit the transfer command buffer and wait for completion via fence.</summary>
    private void SubmitAndWait()
    {
        VkCommandBuffer cmd = _transferCmd;
        VkSubmitInfo submit = new() { commandBufferCount = 1, pCommandBuffers = &cmd };
        var fence = _fence;
        _vkd.vkResetFences(1, &fence).CheckResult();
        _vkd.vkQueueSubmit(_computeQueue, 1, &submit, _fence).CheckResult();
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

        // 8. Create reusable fence for submission synchronization
        VkFenceCreateInfo fenceCI = new();
        VkFence fence;
        _vkd.vkCreateFence(&fenceCI, null, &fence).CheckResult();
        _fence = fence;
    }

    /// <summary>Print device info to console.</summary>
    // Capability flags detected at init
    public bool Has8BitStorage { get; private set; }
    public bool Has16BitStorage { get; private set; }
    public bool HasShaderFloat16Int8 { get; private set; }
    public bool HasCooperativeMatrix { get; private set; }
    public bool HasSubgroupSizeControl { get; private set; }

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

        // Query compute-relevant extensions
        uint extCount = 0;
        _vki.vkEnumerateDeviceExtensionProperties(_physicalDevice, null, &extCount, null);
        var exts = new VkExtensionProperties[extCount];
        fixed (VkExtensionProperties* p = exts)
            _vki.vkEnumerateDeviceExtensionProperties(_physicalDevice, null, &extCount, p);

        var extNames = new HashSet<string>();
        for (int i = 0; i < extCount; i++)
        {
            fixed (byte* namePtr = exts[i].extensionName)
                extNames.Add(new string((sbyte*)namePtr));
        }

        Has8BitStorage = extNames.Contains("VK_KHR_8bit_storage");
        Has16BitStorage = extNames.Contains("VK_KHR_16bit_storage");
        HasShaderFloat16Int8 = extNames.Contains("VK_KHR_shader_float16_int8");
        HasCooperativeMatrix = extNames.Contains("VK_KHR_cooperative_matrix");
        HasSubgroupSizeControl = extNames.Contains("VK_EXT_subgroup_size_control");

        var found = new List<string>();
        if (Has8BitStorage) found.Add("8bit_storage");
        if (Has16BitStorage) found.Add("16bit_storage");
        if (HasShaderFloat16Int8) found.Add("float16_int8");
        if (HasCooperativeMatrix) found.Add("cooperative_matrix");
        if (HasSubgroupSizeControl) found.Add("subgroup_size_control");
        if (found.Count > 0)
            Console.WriteLine($"  Compute extensions: {string.Join(", ", found)}");
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

    /// <summary>
    /// Allocate a pinned host-visible buffer accessible from both CPU and GPU.
    /// The buffer can be mapped for CPU read/write and used as a GPU storage buffer.
    /// Ideal for small, frequently-transferred data like hidden states.
    /// </summary>
    public Tensor AllocatePinned(TensorShape shape, DType dtype = DType.Float32)
    {
        ulong byteSize = (ulong)(shape.ElementCount * DTypeInfo.BytesPerElement(dtype));
        var gpuBuf = GpuBuffer.CreatePinned(this, byteSize, VkBufferUsageFlags.StorageBuffer);

        var handle = _nextHandle++;
        _buffers[handle] = gpuBuf;
        return new Tensor(shape, dtype, handle);
    }

    /// <summary>
    /// Map a pinned tensor for CPU access. Returns a float pointer.
    /// Only valid for tensors created with AllocatePinned.
    /// </summary>
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

        var handle = _nextHandle++;
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
    private ComputePipeline? _siluMulPipeline;
    private ComputePipeline? _addInPlacePipeline;
    private ComputePipeline? _addScaledInPlacePipeline;
    private ComputePipeline? _clearPipeline;
    private ComputePipeline? _elementwiseMulPipeline;
    private ComputePipeline? _ropePipeline;
    private ComputePipeline? _softmaxPipeline;
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

    private struct RmsNormParams { public uint n; public float eps; }
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

    private void DispatchOrRecord(ComputePipeline pipe, ReadOnlySpan<GpuBuffer> buffers,
        uint groupX, void* push, uint groupY = 1, uint groupZ = 1)
    {
        if (_recording)
            pipe.RecordWith(_transferCmd, buffers, groupX, groupY, groupZ, push);
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

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f)
    {
        _ropePipeline ??= new ComputePipeline(this, Shaders.RoPE, 1, pushConstantSize: sizeof(RoPEParams));
        uint numHeads = (uint)(x.ElementCount / headDim);
        uint totalPairs = numHeads * (uint)(headDim / 2);
        var p = new RoPEParams { numHeads = numHeads, headDim = (uint)headDim, position = position, theta = ropeTheta };
        DispatchOrRecord(_ropePipeline, [GetBuffer(x)], (totalPairs + 255) / 256, &p);
    }

    public void Softmax(Tensor x)
    {
        _softmaxPipeline ??= new ComputePipeline(this, Shaders.Softmax, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)x.ElementCount };
        DispatchOrRecord(_softmaxPipeline, [GetBuffer(x)], 1, &p);
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
                DispatchOrRecord(_matVecQ6KPipeline, bufs, totalRows, &p);
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

    public void Attention(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen)
    {
        _attentionPipeline ??= new ComputePipeline(this, Shaders.Attention, 4, pushConstantSize: sizeof(AttentionParams));
        var p = new AttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, maxSeqLen = maxSeqLen
        };
        DispatchOrRecord(_attentionPipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(output)],
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

    public void TqAttention(Tensor q, Tensor rotatedQ, Tensor kCacheTq, Tensor vCacheTq,
        Tensor kCacheFp16, Tensor vCacheFp16, Tensor output, Tensor codebook,
        uint numHeads, uint numKvHeads, uint headDim,
        uint tqSeqLen, uint fp16SeqLen, uint maxSeqLen, uint blockBytes)
    {
        _tqAttentionPipeline ??= new ComputePipeline(this, Shaders.TqAttention, 8, pushConstantSize: sizeof(TqAttentionParams));
        var p = new TqAttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads, headDim = headDim,
            tqSeqLen = tqSeqLen, fp16SeqLen = fp16SeqLen, maxSeqLen = maxSeqLen, blockBytes = blockBytes
        };
        DispatchOrRecord(_tqAttentionPipeline,
            [GetBuffer(q), GetBuffer(rotatedQ), GetBuffer(kCacheTq), GetBuffer(vCacheTq),
             GetBuffer(kCacheFp16), GetBuffer(vCacheFp16), GetBuffer(output), GetBuffer(codebook)],
            numHeads, &p);
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
        _siluMulPipeline?.Dispose();
        _addInPlacePipeline?.Dispose();
        _addScaledInPlacePipeline?.Dispose();
        _clearPipeline?.Dispose();
        _elementwiseMulPipeline?.Dispose();
        _ropePipeline?.Dispose();
        _softmaxPipeline?.Dispose();
        _matVecQ4KPipeline?.Dispose();
        _matVecQ6KPipeline?.Dispose();
        _matVecF32Pipeline?.Dispose();
        _kvAppendPipeline?.Dispose();
        _attentionPipeline?.Dispose();
        _embedLookupPipeline?.Dispose();
        _embedLookupQ4KPipeline?.Dispose();

        _downloadStaging?.Dispose();
        _uploadStaging?.Dispose();

        // Free all tracked GPU buffers
        foreach (var buf in _buffers.Values)
            buf.Dispose();
        _buffers.Clear();

        _vkd.vkDestroyFence(_fence, null);
        _vkd.vkDestroyCommandPool(_commandPool, null);
        _vkd.vkDestroyDevice(null);
        _vki.vkDestroyInstance(null);
    }
}
