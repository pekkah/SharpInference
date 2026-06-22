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

    // Opt-in Vulkan validation layers (SHARPI_VULKAN_VALIDATION=1). Used to turn
    // flaky native access-violations into deterministic validation diagnostics
    // (issue #153). Null when validation is off (the default).
    private VkDebugUtilsMessengerEXT _debugMessenger;
    private readonly bool _validationEnabled;

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

    /// <param name="deviceIndex">
    /// Physical-device index to select (from <c>--device</c>), or -1 to auto-select
    /// (prefer a discrete GPU, fall back to any compute-capable device).
    /// </param>
    public VulkanBackend(int deviceIndex = -1)
    {
        // 1. Initialize Vulkan loader
        vkInitialize().CheckResult();

        // Opt-in validation layers (issue #153): when SHARPI_VULKAN_VALIDATION=1 we
        // enable the Khronos validation layer + VK_EXT_debug_utils so invalid buffer/
        // descriptor/pipeline usage is reported deterministically (even on runs that
        // don't crash). Default (env unset/0) keeps the instance exactly as before —
        // no extra layers, no debug-utils, zero validation overhead.
        var validationEnv = Environment.GetEnvironmentVariable("SHARPI_VULKAN_VALIDATION");
        _validationEnabled =
            validationEnv is "1" || string.Equals(validationEnv, "true", StringComparison.OrdinalIgnoreCase);

        // 2. Create instance (Vulkan 1.3+)
        VkApplicationInfo appInfo = new()
        {
            apiVersion = VkVersion.Version_1_3, // Vortice may not have 1.4 constant yet
        };

        // A messenger create-info chained into pNext also catches validation errors
        // raised *during* vkCreateInstance/vkDestroyInstance themselves.
        VkDebugUtilsMessengerCreateInfoEXT dbgCI = MakeDebugMessengerCreateInfo();

        // Validation-layer name + debug-utils extension name (null-terminated UTF-8 literals
        // → no heap allocation; only consumed when validation is enabled).
        fixed (byte* pValidationLayer = "VK_LAYER_KHRONOS_validation\0"u8)
        fixed (byte* pDebugUtilsExt   = "VK_EXT_debug_utils\0"u8)
        {
            byte** layerPtrs = stackalloc byte*[1];
            byte** instExtPtrs = stackalloc byte*[1];
            layerPtrs[0] = pValidationLayer;
            instExtPtrs[0] = pDebugUtilsExt;

            VkInstanceCreateInfo instanceCI = new()
            {
                pApplicationInfo = &appInfo,
                pNext = _validationEnabled ? &dbgCI : null,
                enabledLayerCount = _validationEnabled ? 1u : 0u,
                ppEnabledLayerNames = _validationEnabled ? layerPtrs : null,
                enabledExtensionCount = _validationEnabled ? 1u : 0u,
                ppEnabledExtensionNames = _validationEnabled ? instExtPtrs : null,
            };
            vkCreateInstance(in instanceCI, out _instance).CheckResult();
        }
        _vki = new VkInstanceApi(in _instance);

        // Register the standalone debug messenger now that we have an instance + loaded
        // instance-extension entry points. Errors/warnings go to Console.Error.
        if (_validationEnabled)
        {
            VkDebugUtilsMessengerEXT messenger;
            var res = _vki.vkCreateDebugUtilsMessengerEXT(&dbgCI, &messenger);
            if (res == VkResult.Success)
            {
                _debugMessenger = messenger;
                Console.Error.WriteLine("[VK-VALIDATION] Vulkan validation layers ENABLED (SHARPI_VULKAN_VALIDATION=1)");
            }
            else
            {
                Console.Error.WriteLine($"[VK-VALIDATION] WARNING: vkCreateDebugUtilsMessengerEXT failed ({res}); " +
                    "validation messages from pNext-chained create/destroy still fire, standalone messenger disabled.");
            }
        }

        // 3. Select physical device (prefer discrete GPU)
        _physicalDevice = SelectPhysicalDevice(deviceIndex);
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
        bool hasSubgroupSizeControl = HasSubgroupSizeControl;

        // 5b. Query the supported subgroup-size range (issue #318). The reduction shaders pack
        // "8 rows × 32 lanes" per workgroup and assume the subgroup is exactly 32 wide; on AMD
        // Wave64 a subgroup would span two row groups and corrupt subgroupAdd/subgroupElect.
        // We pin requiredSubgroupSize=32 at pipeline creation (ComputePipeline) when the device
        // could pick a non-32 subgroup (see ComputePipeline.ShouldPinSubgroupSize32). If the
        // extension is absent these stay 0 → pinning disabled.
        if (hasSubgroupSizeControl)
        {
            VkPhysicalDeviceSubgroupSizeControlProperties sgProps = new();
            VkPhysicalDeviceProperties2 props2 = new()
            {
                pNext = &sgProps,
            };
            _vki.vkGetPhysicalDeviceProperties2(_physicalDevice, &props2);
            MinSubgroupSize = sgProps.minSubgroupSize;
            MaxSubgroupSize = sgProps.maxSubgroupSize;
        }

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
        byte[] sgSizeNameBytes    = System.Text.Encoding.UTF8.GetBytes("VK_EXT_subgroup_size_control\0");

        int enabledExtCount = (hasFloat16Int8 ? 1 : 0) + (has16BitStorage ? 1 : 0)
                            + (has8BitStorage ? 1 : 0) + (hasIntDot ? 1 : 0)
                            + (hasBfloat16 ? 1 : 0) + (hasFloat8 ? 1 : 0)
                            + (hasSubgroupSizeControl ? 1 : 0);
        int extIdx = 0;

        fixed (byte* pF16Int8   = f16Int8NameBytes,
                     pStorage16 = storage16NameBytes,
                     pStorage8  = storage8NameBytes,
                     pIntDot    = intDotNameBytes,
                     pBf16      = bf16NameBytes,
                     pFp8       = fp8NameBytes,
                     pSgSize    = sgSizeNameBytes)
        {
            byte** extPtrs = stackalloc byte*[enabledExtCount > 0 ? enabledExtCount : 1];
            if (hasFloat16Int8)  extPtrs[extIdx++] = pF16Int8;
            if (has16BitStorage) extPtrs[extIdx++] = pStorage16;
            if (has8BitStorage)  extPtrs[extIdx++] = pStorage8;
            if (hasIntDot)       extPtrs[extIdx++] = pIntDot;
            if (hasBfloat16)     extPtrs[extIdx++] = pBf16;
            if (hasFloat8)       extPtrs[extIdx++] = pFp8;
            if (hasSubgroupSizeControl) extPtrs[extIdx++] = pSgSize;

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

            void* featureChain =
                hasBfloat16 ? (void*)&bf16Features :
                hasFloat8   ? (void*)&fp8Features :
                baseChain;

            // Prepend the subgroup-size-control feature (issue #318). We only enable
            // subgroupSizeControl; computeFullSubgroups is unnecessary because every pinned
            // shader has local_size_x a multiple of 32 (so the workgroup already fills whole
            // subgroups). The actual requiredSubgroupSize=32 is set per pipeline stage.
            VkPhysicalDeviceSubgroupSizeControlFeatures sgSizeFeatures = new()
            {
                subgroupSizeControl = VkBool32.True,
                pNext = featureChain,
            };

            void* pNextChain =
                hasSubgroupSizeControl ? (void*)&sgSizeFeatures :
                featureChain;

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

    // Subgroup-size range reported by VK_EXT_subgroup_size_control (issue #318).
    // 0 when the extension is absent (subgroup-size pinning then stays disabled).
    public uint MinSubgroupSize { get; private set; }
    public uint MaxSubgroupSize { get; private set; }

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

    private VkPhysicalDevice SelectPhysicalDevice(int deviceIndex)
    {
        uint count = 0;
        _vki.vkEnumeratePhysicalDevices(&count, null);
        if (count == 0) throw new InvalidOperationException("No Vulkan-capable GPU found");

        var devices = new VkPhysicalDevice[count];
        fixed (VkPhysicalDevice* p = devices)
            _vki.vkEnumeratePhysicalDevices(&count, p);

        // Explicit device requested via --device: honor it exactly (no discrete-GPU fallback).
        if (deviceIndex >= 0)
        {
            if (deviceIndex >= (int)count)
                throw new InvalidOperationException(
                    $"--device {deviceIndex}: only {count} Vulkan device(s) present (valid indices 0..{count - 1}).");
            var chosen = devices[deviceIndex];
            if (!HasComputeQueue(chosen))
                throw new InvalidOperationException(
                    $"--device {deviceIndex}: the selected Vulkan device has no compute queue.");
            return chosen;
        }

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

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32, bool exact = false)
    {
        // Vulkan path doesn't pool/round; the exact hint is a no-op here.
        _ = exact;
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

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape, bool exact = false)
    {
        _ = exact;
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
    public unsafe Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype, bool exact = false)
    {
        _ = exact;
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
    private ComputePipeline? _geluTanhMulPipeline;
    private ComputePipeline? _softcapPipeline;
    private ComputePipeline? _siluPipeline;
    private ComputePipeline? _addInPlacePipeline;
    private ComputePipeline? _addScaledInPlacePipeline;
    private ComputePipeline? _scaleInPlacePipeline;
    private ComputePipeline? _clearPipeline;
    private ComputePipeline? _elementwiseMulPipeline;
    private ComputePipeline? _ropePipeline;
    private ComputePipeline? _ropeNeoxPipeline;
    private ComputePipeline? _ropeNeoxWithFactorsPipeline;
    private ComputePipeline? _softmaxPipeline;
    private ComputePipeline? _sigmoidPipeline;
    private ComputePipeline? _matVecQ4KPipeline;
    private ComputePipeline? _matVecBatchedQ4KPipeline;
    private ComputePipeline? _matVecQ6KPipeline;
    private ComputePipeline? _matVecQ5KPipeline;
    private ComputePipeline? _matVecQ8_0Pipeline;
    private ComputePipeline? _matVecQ4_0Pipeline;
    private ComputePipeline? _matVecF32Pipeline;
    private ComputePipeline? _kvAppendPipeline;
    private ComputePipeline? _attentionPipeline;
    private ComputePipeline? _kvAppendBf16Pipeline;
    private ComputePipeline? _attentionBf16Pipeline;
    private ComputePipeline? _kvAppendQ8Pipeline;
    private ComputePipeline? _attentionQ8Pipeline;
    private ComputePipeline? _splitKvPartialPipeline;
    private ComputePipeline? _splitKvPartialBf16Pipeline;
    private ComputePipeline? _splitKvPartialQ8Pipeline;
    private ComputePipeline? _splitKvCombinePipeline;
    private ComputePipeline? _snapKvScorePipeline;
    private ComputePipeline? _kvCompactPipeline;
    private ComputePipeline? _embedLookupPipeline;
    private ComputePipeline? _embedLookupQ4KPipeline;
    private ComputePipeline? _embedLookupQ6KPipeline;
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
    private struct WeightedHeadNormParams { public uint headDim; public uint numHeads; public float eps; public uint weightStride; }
    private struct CountParams { public uint n; }
    private struct ScaleParams { public uint n; public float scale; }
    private struct RoPEParams { public uint numHeads; public uint headDim; public int position; public float theta; }
    private struct MatVecParams { public uint rows; public uint cols; }
    private struct MatVecBatchedParams { public uint rows; public uint cols; public uint nTok; }
    private struct EmbedParams { public uint tokenId; public uint embDim; }
    private struct KvAppendParams { public uint kvDim; public uint position; public uint maxSeqLen; }
    private struct AttentionParams { public uint numHeads; public uint numKvHeads; public uint headDim; public uint seqLen; public uint maxSeqLen; public uint window; }
    private struct SplitKvPartialParams { public uint numHeads; public uint numKvHeads; public uint headDim; public uint seqLen; public uint nSplits; public uint window; }
    private struct SplitKvCombineParams { public uint numHeads; public uint headDim; public uint nSplits; }
    private struct SnapKvScoreParams { public uint numHeads; public uint numKvHeads; public uint headDim; public uint promptLen; public uint qAbsPos; public uint maxSeqLen; }
    private struct KvCompactParams { public uint K; public uint kvDim; }
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

    /// <summary>Per-head RMS norm with learned weights. <paramref name="perChannelWeight"/>
    /// false → weight is shared <c>[headDim]</c> vector applied identically per head (Qwen3);
    /// true → weight is <c>[numHeads * headDim]</c> with one slice per head (OLMoE).</summary>
    public void HeadNorm(Tensor data, Tensor weight, uint numHeads, uint headDim,
        float eps = 1e-6f, bool perChannelWeight = false)
    {
        _headNormPipeline ??= new ComputePipeline(this, Shaders.HeadNorm, 2, pushConstantSize: sizeof(WeightedHeadNormParams));
        var p = new WeightedHeadNormParams
        {
            headDim = headDim,
            numHeads = numHeads,
            eps = eps,
            weightStride = perChannelWeight ? headDim : 0u,
        };
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

    public void SiLU(Tensor x)
    {
        _siluPipeline ??= new ComputePipeline(this, Shaders.SiLU, 1, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)x.ElementCount };
        // 64-bit arithmetic before the cast (activation buffers are small, but avoids
        // any theoretical uint wrap that would dispatch 0 workgroups).
        DispatchOrRecord(_siluPipeline, [GetBuffer(x)], (uint)((x.ElementCount + 255) / 256), &p);
    }

    public void SiLuMul(Tensor gate, Tensor up)
    {
        _siluMulPipeline ??= new ComputePipeline(this, Shaders.SiLuMul, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)gate.ElementCount };
        DispatchOrRecord(_siluMulPipeline, [GetBuffer(gate), GetBuffer(up)], ((uint)gate.ElementCount + 255) / 256, &p);
    }

    public void GeluTanhMul(Tensor gate, Tensor up)
    {
        _geluTanhMulPipeline ??= new ComputePipeline(this, Shaders.GeluTanhMul, 2, pushConstantSize: sizeof(CountParams));
        var p = new CountParams { n = (uint)gate.ElementCount };
        DispatchOrRecord(_geluTanhMulPipeline, [GetBuffer(gate), GetBuffer(up)], ((uint)gate.ElementCount + 255) / 256, &p);
    }

    public void SoftcapInPlace(Tensor x, float cap)
    {
        // Reuse the { uint n, float scale } push-constant layout (scale carries the cap).
        _softcapPipeline ??= new ComputePipeline(this, Shaders.Softcap, 1, pushConstantSize: sizeof(ScaleParams));
        var p = new ScaleParams { n = (uint)x.ElementCount, scale = cap };
        DispatchOrRecord(_softcapPipeline, [GetBuffer(x)], ((uint)x.ElementCount + 255) / 256, &p);
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

    /// <summary>
    /// NEOX RoPE with a per-half-dim <paramref name="freqFactors"/> table (size head_dim/2) that
    /// divides each pair's frequency. The Vulkan mirror of CUDA's <c>RoPEWithFactors</c>: Gemma 4
    /// global (non-SWA) layers apply <c>rope_freqs.weight</c> here to mask the high-frequency tail,
    /// while SWA layers use the plain <see cref="RoPE"/>. Computes cos/sin in-shader (no tables).
    /// </summary>
    public void RoPEWithFactors(Tensor x, int position, int headDim, float ropeTheta, Tensor freqFactors)
    {
        _ropeNeoxWithFactorsPipeline ??= new ComputePipeline(this, Shaders.RoPENeoxWithFactors, 2, pushConstantSize: sizeof(RoPEParams));
        uint numHeads = (uint)(x.ElementCount / headDim);
        uint totalPairs = numHeads * (uint)(headDim / 2);
        var p = new RoPEParams { numHeads = numHeads, headDim = (uint)headDim, position = position, theta = ropeTheta };
        DispatchOrRecord(_ropeNeoxWithFactorsPipeline, [GetBuffer(x), GetBuffer(freqFactors)], (totalPairs + 255) / 256, &p);
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
            case DType.Q5_K:
                _matVecQ5KPipeline ??= new ComputePipeline(this, Shaders.MatVecQ5K, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ5KPipeline, bufs, (totalRows + 7) / 8, &p);
                break;
            case DType.Q8_0:
                _matVecQ8_0Pipeline ??= new ComputePipeline(this, Shaders.MatVecQ8_0, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ8_0Pipeline, bufs, (totalRows + 7) / 8, &p);
                break;
            case DType.Q4_0:
                _matVecQ4_0Pipeline ??= new ComputePipeline(this, Shaders.MatVecQ4_0, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ4_0Pipeline, bufs, (totalRows + 7) / 8, &p);
                break;
            default: // Q4_K — 256 threads, 8 rows per workgroup, subgroupAdd reduction
                _matVecQ4KPipeline ??= new ComputePipeline(this, Shaders.MatVecQ4K, 3, pushConstantSize: sizeof(MatVecParams));
                DispatchOrRecord(_matVecQ4KPipeline, bufs, (totalRows + 7) / 8, &p);
                break;
        }
    }

    /// <summary>
    /// Batched (weight-stationary) matrix-vector multiply: computes <paramref name="nTok"/>
    /// independent matvecs against the SAME weight matrix. The Q4_K weight is read from VRAM
    /// once and multiplied into <paramref name="nTok"/> accumulators — the weight-amortization
    /// behind Vulkan speculative decoding (issue #308).
    ///
    /// Layouts: <paramref name="inputAll"/> is row-major [nTok][cols], <paramref name="outputAll"/>
    /// is row-major [nTok][rows]. <c>rows = outputAll.ElementCount / nTok</c>,
    /// <c>cols = inputAll.ElementCount / nTok</c>.
    ///
    /// For Q4_K this dispatches the batched shader (the win). For every other dtype it falls
    /// back to a correctness-only loop of the single-row <see cref="MatMul"/> over the K
    /// input/output slices (no amortization), so the method is total across all weight dtypes.
    /// The Q4_K batched result is bit-identical to nTok separate single-row MatMul calls.
    /// </summary>
    public void MatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll, int nTok, DType weightDType)
    {
        if (nTok is < 1 or > 8)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be in [1, 8].");
        if (outputAll.ElementCount % nTok != 0)
            throw new ArgumentException($"outputAll.ElementCount ({outputAll.ElementCount}) must be divisible by nTok ({nTok}).", nameof(outputAll));
        if (inputAll.ElementCount % nTok != 0)
            throw new ArgumentException($"inputAll.ElementCount ({inputAll.ElementCount}) must be divisible by nTok ({nTok}).", nameof(inputAll));

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);

        if (weightDType != DType.Q4_K)
        {
            // Fallback: K independent single-row matvecs over the [nTok][·] slices. Correct for
            // all dtypes but with NO weight amortization (later PRs add batched shaders). Tensor
            // has no offset sub-view, so each slice is staged through a per-token temp F32 tensor.
            const int f32Bytes = 4;
            var tmpIn = Allocate(TensorShape.D1(cols));
            var tmpOut = Allocate(TensorShape.D1(rows));
            try
            {
                for (int k = 0; k < nTok; k++)
                {
                    RecordComputeCopyRegion(tmpIn, 0, inputAll, (long)k * cols * f32Bytes, (long)cols * f32Bytes);
                    MatMul(tmpOut, matrix, tmpIn, weightDType);
                    RecordComputeCopyRegion(outputAll, (long)k * rows * f32Bytes, tmpOut, 0, (long)rows * f32Bytes);
                }
            }
            finally
            {
                Free(tmpIn);
                Free(tmpOut);
            }
            return;
        }

        _matVecBatchedQ4KPipeline ??= new ComputePipeline(this, Shaders.MatVecBatchedQ4K, 3, pushConstantSize: sizeof(MatVecBatchedParams));
        var p = new MatVecBatchedParams { rows = (uint)rows, cols = (uint)cols, nTok = (uint)nTok };
        var bufs = (ReadOnlySpan<GpuBuffer>)[GetBuffer(matrix), GetBuffer(inputAll), GetBuffer(outputAll)];
        DispatchOrRecord(_matVecBatchedQ4KPipeline, bufs, ((uint)rows + 7) / 8, &p);
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

    /// <summary>
    /// Dequantize one row from a Q6_K-packed embedding table directly into
    /// <paramref name="output"/> (issue #124, Gemma 4 12B tied token_embd). Keeps the large
    /// Q6_K table packed (~787 MiB for [3840, 262144]) off the F32 dequant path that would
    /// burn ~4 GB of VRAM. <paramref name="embDim"/> must be a multiple of 256 (Q6_K block size).
    /// </summary>
    public void EmbedLookupQ6K(Tensor embTable, Tensor output, uint tokenId, uint embDim)
    {
        _embedLookupQ6KPipeline ??= new ComputePipeline(this, Shaders.EmbedLookupQ6K, 2, pushConstantSize: sizeof(EmbedParams));
        var p = new EmbedParams { tokenId = tokenId, embDim = embDim };
        DispatchOrRecord(_embedLookupQ6KPipeline, [GetBuffer(embTable), GetBuffer(output)], 1, &p);
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
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        _attentionPipeline ??= new ComputePipeline(this, Shaders.Attention, 5, pushConstantSize: sizeof(AttentionParams));
        var p = new AttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, maxSeqLen = maxSeqLen, window = window
        };
        DispatchOrRecord(_attentionPipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(output),
             GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    /// <summary>
    /// Flash-decoding split-KV attention (issue #312) — the Vulkan mirror of CUDA's
    /// <c>SHARPI_SPLIT_DECODE</c> path. Splits each head's causal KV range into fixed 512-position
    /// slices dispatched across a 2D grid of <c>numHeads × nSplits</c> workgroups (parallelizing
    /// the long-context KV read instead of serially scanning it in one workgroup like
    /// <see cref="Attention"/>), then LSE-merges the per-slice partials in a second combine pass.
    /// fp32 K/V only. Opt-in (DEFAULT-OFF) via the caller's <c>SHARPI_VULKAN_SPLIT_DECODE</c> gate;
    /// when the gate is off this method is never reached, so the spill path is byte-identical.
    ///
    /// <paramref name="partialO"/> is <c>[numHeads * maxSplits * headDim]</c> (un-normalized
    /// weighted-V numerators) and <paramref name="partialMeta"/> is <c>[numHeads * maxSplits * 2]</c>
    /// ((m_i, l_i) per (head, split)); the caller allocates both sized to maxSplits =
    /// ceil(maxSeqLen/512). nSplits = ceil(seqLen/512) ≤ maxSplits selects the live grid.
    /// </summary>
    public void AttentionSplitKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor partialO, Tensor partialMeta,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        // The combine shader bounds its per-head rescale array at 256 splits (MAX_SPLITS) ⇔
        // seqLen <= 256*512 = 131072. Check seqLen directly so the +511 can't overflow.
        if (seqLen > 131072)
            throw new ArgumentOutOfRangeException(nameof(seqLen),
                $"split-KV supports up to 256 splits (seqLen <= 131072); got seqLen={seqLen}.");
        uint nSplits = (seqLen + 511) / 512;
        _splitKvPartialPipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvPartial, 5, pushConstantSize: sizeof(SplitKvPartialParams));
        _splitKvCombinePipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvCombine, 3, pushConstantSize: sizeof(SplitKvCombineParams));

        // Partial pass: numHeads × nSplits workgroups (2D dispatch; workgroup (x=head, y=split)).
        var pp = new SplitKvPartialParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, nSplits = nSplits, window = window
        };
        DispatchOrRecord(_splitKvPartialPipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(partialO),
             GetBuffer(partialMeta)],
            numHeads, &pp, groupY: nSplits);

        // Combine reads the partials the partial pass just wrote, so the two dispatches must be
        // ordered. When recording (the engine path) insert a compute→compute barrier; in the
        // immediate path (DispatchWith) each dispatch is its own submit + fence-wait, so the
        // partial pass has fully completed before the combine is submitted — no barrier needed,
        // and RecordBarrier on a non-recording command buffer would be invalid.
        if (_recording) RecordBarrier();

        var cp = new SplitKvCombineParams { numHeads = numHeads, headDim = headDim, nSplits = nSplits };
        DispatchOrRecord(_splitKvCombinePipeline,
            [GetBuffer(partialO), GetBuffer(partialMeta), GetBuffer(output)],
            numHeads, &cp);
    }

    /// <summary>
    /// bf16 (issue #332) variant of <see cref="AttentionSplitKv"/>: identical 2-pass split-KV
    /// flow, but the partial pass reads the K/V cache as IEEE fp16 packed two-per-uint (via
    /// <c>AttentionSplitKvPartialBf16</c>). The combine pass is the SAME dtype-agnostic
    /// <c>AttentionSplitKvCombine</c> as fp32 (it reads only the fp32 partial buffers). Same
    /// nSplits guard, dispatch, and barrier as <see cref="AttentionSplitKv"/>.
    /// </summary>
    public void AttentionSplitKvBf16(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor partialO, Tensor partialMeta,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        // Mirrors AttentionSplitKv's guard (combine bounds the per-head rescale at 256 splits).
        if (seqLen > 131072)
            throw new ArgumentOutOfRangeException(nameof(seqLen),
                $"split-KV supports up to 256 splits (seqLen <= 131072); got seqLen={seqLen}.");
        uint nSplits = (seqLen + 511) / 512;
        _splitKvPartialBf16Pipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvPartialBf16, 5, pushConstantSize: sizeof(SplitKvPartialParams));
        _splitKvCombinePipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvCombine, 3, pushConstantSize: sizeof(SplitKvCombineParams));

        var pp = new SplitKvPartialParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, nSplits = nSplits, window = window
        };
        DispatchOrRecord(_splitKvPartialBf16Pipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(partialO),
             GetBuffer(partialMeta)],
            numHeads, &pp, groupY: nSplits);

        if (_recording) RecordBarrier();

        var cp = new SplitKvCombineParams { numHeads = numHeads, headDim = headDim, nSplits = nSplits };
        DispatchOrRecord(_splitKvCombinePipeline,
            [GetBuffer(partialO), GetBuffer(partialMeta), GetBuffer(output)],
            numHeads, &cp);
    }

    /// <summary>
    /// q8_0 (issue #332) variant of <see cref="AttentionSplitKv"/>: identical 2-pass split-KV
    /// flow, but the partial pass reads the K/V cache as ggml <c>block_q8_0</c> (34 bytes/block,
    /// dequant <c>fp16(d) * int8</c> per element) via <c>AttentionSplitKvPartialQ8</c>. The
    /// combine pass is the SAME dtype-agnostic <c>AttentionSplitKvCombine</c> as fp32 (it reads
    /// only the fp32 partial buffers). Same nSplits guard, dispatch, and barrier as
    /// <see cref="AttentionSplitKv"/>.
    /// </summary>
    public void AttentionSplitKvQ8(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor partialO, Tensor partialMeta,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        if (seqLen > 131072)
            throw new ArgumentOutOfRangeException(nameof(seqLen),
                $"split-KV supports up to 256 splits (seqLen <= 131072); got seqLen={seqLen}.");
        uint nSplits = (seqLen + 511) / 512;
        _splitKvPartialQ8Pipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvPartialQ8, 5, pushConstantSize: sizeof(SplitKvPartialParams));
        _splitKvCombinePipeline ??= new ComputePipeline(this, Shaders.AttentionSplitKvCombine, 3, pushConstantSize: sizeof(SplitKvCombineParams));

        var pp = new SplitKvPartialParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, nSplits = nSplits, window = window
        };
        DispatchOrRecord(_splitKvPartialQ8Pipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(partialO),
             GetBuffer(partialMeta)],
            numHeads, &pp, groupY: nSplits);

        if (_recording) RecordBarrier();

        var cp = new SplitKvCombineParams { numHeads = numHeads, headDim = headDim, nSplits = nSplits };
        DispatchOrRecord(_splitKvCombinePipeline,
            [GetBuffer(partialO), GetBuffer(partialMeta), GetBuffer(output)],
            numHeads, &cp);
    }

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="KvAppend"/>: writes the K/V vectors into the
    /// cache as IEEE fp16 packed two-per-uint (core-GLSL <c>packHalf2x16</c>, no extension).
    /// The cache buffers (<paramref name="kCache"/>/<paramref name="vCache"/>) are bound as
    /// <c>uint[]</c> in the shader regardless of the tensor's declared dtype. Indexes the
    /// cache identically to the fp32 path (<c>position * kv_dim + i</c>, just word-granular).
    /// kv_dim is even, so one thread covers 2 elements.
    /// </summary>
    public void KvAppendBf16(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
        uint kvDim, uint position, uint maxSeqLen)
    {
        _kvAppendBf16Pipeline ??= new ComputePipeline(this, Shaders.KvAppendBf16, 4, pushConstantSize: sizeof(KvAppendParams));
        var p = new KvAppendParams { kvDim = kvDim, position = position, maxSeqLen = maxSeqLen };
        DispatchOrRecord(_kvAppendBf16Pipeline,
            [GetBuffer(kInput), GetBuffer(vInput), GetBuffer(kCache), GetBuffer(vCache)],
            ((kvDim >> 1) + 255) / 256, &p);
    }

    /// <summary>
    /// bf16 (issue #311) variant of <see cref="Attention"/>: identical control flow to the
    /// fp32 path, but the K/V cache buffers (<paramref name="kCache"/>/<paramref name="vCache"/>)
    /// hold IEEE fp16 packed two-per-uint and are read via <c>unpackHalf2x16</c>. All
    /// arithmetic (scores / softmax / value accumulation) stays fp32 — only the stored K/V
    /// mantissa is narrowed. <paramref name="scoresScratch"/> stays fp32 (see
    /// <see cref="Attention"/> for the spill-buffer convention).
    /// </summary>
    public void AttentionBf16(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor scoresScratch,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        _attentionBf16Pipeline ??= new ComputePipeline(this, Shaders.AttentionBf16, 5, pushConstantSize: sizeof(AttentionParams));
        var p = new AttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, maxSeqLen = maxSeqLen, window = window
        };
        DispatchOrRecord(_attentionBf16Pipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(output),
             GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="KvAppend"/>: block-quantizes the K/V vectors
    /// into the cache as ggml <c>block_q8_0</c> (34 bytes/block = fp16 scale + 32 int8, per 32
    /// elements; ~4× smaller than fp32). The cache buffers are bound as <c>uint[]</c> and
    /// indexed identically to the fp32 path (<c>position * kv_dim + i</c>, expressed in blocks).
    /// Dispatched ONE THREAD PER 32-ELEMENT BLOCK; kv_dim must be a multiple of 32 (enforced in
    /// GpuForwardPass). The shader owns a whole 34-byte block per thread and uses masked atomics
    /// for the (non-4-aligned) seam words shared between adjacent blocks.
    /// </summary>
    public void KvAppendQ8_0(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
        uint kvDim, uint position, uint maxSeqLen)
    {
        _kvAppendQ8Pipeline ??= new ComputePipeline(this, Shaders.KvAppendQ8_0, 4, pushConstantSize: sizeof(KvAppendParams));
        var p = new KvAppendParams { kvDim = kvDim, position = position, maxSeqLen = maxSeqLen };
        DispatchOrRecord(_kvAppendQ8Pipeline,
            [GetBuffer(kInput), GetBuffer(vInput), GetBuffer(kCache), GetBuffer(vCache)],
            ((kvDim >> 5) + 255) / 256, &p);
    }

    /// <summary>
    /// q8_0 (issue #325) variant of <see cref="Attention"/>: identical control flow to the fp32
    /// path, but the K/V cache buffers hold ggml <c>block_q8_0</c> (34 bytes/block) and are read
    /// via a per-element byte-gather + dequant (<c>fp16(d) * int8</c>). All arithmetic (scores /
    /// softmax / value accumulation) stays fp32 — only the stored K/V is narrowed.
    /// <paramref name="scoresScratch"/> stays fp32 (see <see cref="Attention"/> for the spill
    /// convention).
    /// </summary>
    public void AttentionQ8_0(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
        Tensor scoresScratch,
        uint numHeads, uint numKvHeads, uint headDim, uint seqLen, uint maxSeqLen,
        uint window = 0u)
    {
        _attentionQ8Pipeline ??= new ComputePipeline(this, Shaders.AttentionQ8_0, 5, pushConstantSize: sizeof(AttentionParams));
        var p = new AttentionParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads,
            headDim = headDim, seqLen = seqLen, maxSeqLen = maxSeqLen, window = window
        };
        DispatchOrRecord(_attentionQ8Pipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(vCache), GetBuffer(output),
             GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    /// <summary>
    /// SnapKV (issue #59) — score one (layer, query) pair against the layer's K
    /// cache and atomicAdd-pool the post-softmax weights into
    /// <paramref name="scoreAccum"/>. Mirrors <c>CudaBackend.SnapKvScore</c>.
    ///
    /// <paramref name="scoresScratch"/> is only read/written when
    /// <c>promptLen &gt; 4096</c> (the shared-memory fast-path cap). Callers always
    /// have to bind a buffer regardless — pass a 1-float placeholder for shorter
    /// prompts, mirroring the <see cref="Attention"/> convention.
    /// </summary>
    public void SnapKvScore(Tensor q, Tensor kCache, Tensor scoreAccum, Tensor scoresScratch,
                            uint numHeads, uint numKvHeads, uint headDim,
                            uint promptLen, uint qAbsPos, uint maxSeqLen)
    {
        _snapKvScorePipeline ??= new ComputePipeline(this, Shaders.SnapKvScore, 4, pushConstantSize: sizeof(SnapKvScoreParams));
        var p = new SnapKvScoreParams
        {
            numHeads = numHeads, numKvHeads = numKvHeads, headDim = headDim,
            promptLen = promptLen, qAbsPos = qAbsPos, maxSeqLen = maxSeqLen,
        };
        DispatchOrRecord(_snapKvScorePipeline,
            [GetBuffer(q), GetBuffer(kCache), GetBuffer(scoreAccum), GetBuffer(scoresScratch)],
            numHeads, &p);
    }

    /// <summary>
    /// SnapKV (issue #59) — gather the kept positions of one KV ring (K or V)
    /// into a dense <c>[K × kvDim]</c> prefix of <paramref name="dst"/>.
    /// <paramref name="src"/> and <paramref name="dst"/> MUST be different
    /// tensors; the destination is later copied back over the original ring's
    /// <c>[0, K × kvDim)</c> region by the caller. <paramref name="keepPositions"/>
    /// must hold int32 indices in <c>[0, originalLength)</c>.
    ///
    /// Dispatched as <c>(ceil(kvDim/256), K, 1)</c> workgroups of 256 threads —
    /// matches the CUDA reference grid.
    /// </summary>
    public void KvCompact(Tensor src, Tensor dst, Tensor keepPositions,
                          uint K, uint kvDim)
    {
        _kvCompactPipeline ??= new ComputePipeline(this, Shaders.KvCompact, 3, pushConstantSize: sizeof(KvCompactParams));
        var p = new KvCompactParams { K = K, kvDim = kvDim };
        uint groupsX = (kvDim + 255) / 256;
        DispatchOrRecord(_kvCompactPipeline,
            [GetBuffer(src), GetBuffer(dst), GetBuffer(keepPositions)],
            groupsX, &p, groupY: K);
    }

    /// <summary>
    /// SnapKV (issue #59) — zero a sub-region of an f32 tensor starting at
    /// <paramref name="elementOffset"/> for <paramref name="elementCount"/>
    /// elements. Mirrors <c>CudaBackend.ClearRegion</c>. SnapKV calls this once
    /// per prefill over <c>promptLen</c> floats (≤ a few KB), so we go via a
    /// CPU-side zero buffer staged through the upload pipeline rather than
    /// adding an offset-aware compute shader. Must NOT be called while a
    /// command-buffer recording session is active.
    /// </summary>
    public unsafe void ClearRegion(Tensor dst, long elementOffset, int elementCount)
    {
        if (elementCount <= 0) return;
        ulong byteSize = (ulong)((long)elementCount * sizeof(float));
        ulong dstByteOffset = (ulong)(elementOffset * sizeof(float));

        if (_uploadStaging == null || _uploadStagingSize < byteSize)
        {
            _uploadStaging?.Dispose();
            _uploadStaging = GpuBuffer.CreateStaging(this, byteSize, VkBufferUsageFlags.TransferSrc);
            _uploadStagingSize = byteSize;
        }

        // Zero the staging window then issue a one-region copy.
        byte* mapped = (byte*)_uploadStaging.Map();
        new Span<byte>(mapped, (int)byteSize).Clear();
        _uploadStaging.Unmap();

        var gpuBuf = GetBuffer(dst);
        VkCommandBufferBeginInfo beginInfo = new() { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
        _vkd.vkBeginCommandBuffer(_transferCmd, &beginInfo).CheckResult();
        VkBufferCopy copyRegion = new() { srcOffset = 0, dstOffset = dstByteOffset, size = byteSize };
        _vkd.vkCmdCopyBuffer(_transferCmd, _uploadStaging!.Buffer, gpuBuf.Buffer, 1, &copyRegion);
        _vkd.vkEndCommandBuffer(_transferCmd).CheckResult();
        SubmitAndWait();
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

    // ── Vulkan validation layer support (issue #153, opt-in) ──────────────

    /// <summary>
    /// Build the debug-messenger create-info used both for the pNext chain on
    /// instance creation and for the standalone messenger. Severity is limited to
    /// WARNING + ERROR; all message types are reported.
    /// </summary>
    private static VkDebugUtilsMessengerCreateInfoEXT MakeDebugMessengerCreateInfo() => new()
    {
        messageSeverity = VkDebugUtilsMessageSeverityFlagsEXT.Warning
                        | VkDebugUtilsMessageSeverityFlagsEXT.Error,
        messageType = VkDebugUtilsMessageTypeFlagsEXT.General
                    | VkDebugUtilsMessageTypeFlagsEXT.Validation
                    | VkDebugUtilsMessageTypeFlagsEXT.Performance,
        pfnUserCallback = &DebugCallback,
    };

    /// <summary>
    /// Validation-layer callback. AOT-safe: a <c>[UnmanagedCallersOnly]</c> static method
    /// (no managed delegate to keep alive / GC). Writes WARNING/ERROR lines — including any
    /// named objects involved — to <see cref="Console.Error"/> with a <c>[VK-VALIDATION]</c> prefix.
    /// </summary>
    [System.Runtime.InteropServices.UnmanagedCallersOnly]
    private static uint DebugCallback(
        VkDebugUtilsMessageSeverityFlagsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT messageTypes,
        VkDebugUtilsMessengerCallbackDataEXT* data,
        void* userData)
    {
        // An exception escaping an [UnmanagedCallersOnly] callback fail-fasts the process,
        // so swallow everything. PtrToStringUTF8 decodes the native strings as UTF-8
        // (new string((sbyte*)..) would use the ANSI code page on Windows).
        try
        {
            // Return VK_FALSE (0): the callback does not abort the triggering API call.
            if (data == null) return 0u;

            string sev = severity.HasFlag(VkDebugUtilsMessageSeverityFlagsEXT.Error) ? "ERROR" : "WARNING";
            string msg = data->pMessage != null
                ? System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)data->pMessage) ?? "(no message)"
                : "(no message)";
            string idName = data->pMessageIdName != null
                ? System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)data->pMessageIdName) ?? ""
                : "";

            Console.Error.WriteLine($"[VK-VALIDATION] {sev} ({messageTypes}) [{idName}] {msg}");

            // Dump any named objects involved (buffer / descriptor set / pipeline / etc.)
            for (uint i = 0; i < data->objectCount; i++)
            {
                VkDebugUtilsObjectNameInfoEXT obj = data->pObjects[i];
                string objName = obj.pObjectName != null
                    ? System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)obj.pObjectName) ?? "(unnamed)"
                    : "(unnamed)";
                Console.Error.WriteLine(
                    $"[VK-VALIDATION]   object[{i}] type={obj.objectType} handle=0x{obj.objectHandle:X} name={objName}");
            }
            Console.Error.Flush();
        }
        catch
        {
            // Never let a managed exception cross back into the Vulkan driver.
        }

        return 0u; // VK_FALSE
    }

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
        _geluTanhMulPipeline?.Dispose();
        _softcapPipeline?.Dispose();
        _siluPipeline?.Dispose();
        _addInPlacePipeline?.Dispose();
        _addScaledInPlacePipeline?.Dispose();
        _scaleInPlacePipeline?.Dispose();
        _clearPipeline?.Dispose();
        _elementwiseMulPipeline?.Dispose();
        _ropePipeline?.Dispose();
        _ropeNeoxPipeline?.Dispose();
        _ropeNeoxWithFactorsPipeline?.Dispose();
        _softmaxPipeline?.Dispose();
        _sigmoidPipeline?.Dispose();
        _matVecQ4KPipeline?.Dispose();
        _matVecBatchedQ4KPipeline?.Dispose();
        _matVecQ6KPipeline?.Dispose();
        _matVecQ5KPipeline?.Dispose();
        _matVecQ8_0Pipeline?.Dispose();
        _matVecQ4_0Pipeline?.Dispose();
        _matVecF32Pipeline?.Dispose();
        _kvAppendPipeline?.Dispose();
        _attentionPipeline?.Dispose();
        _kvAppendBf16Pipeline?.Dispose();
        _attentionBf16Pipeline?.Dispose();
        _kvAppendQ8Pipeline?.Dispose();
        _attentionQ8Pipeline?.Dispose();
        _splitKvPartialPipeline?.Dispose();
        _splitKvPartialBf16Pipeline?.Dispose();
        _splitKvPartialQ8Pipeline?.Dispose();
        _splitKvCombinePipeline?.Dispose();
        _snapKvScorePipeline?.Dispose();
        _kvCompactPipeline?.Dispose();
        _tqRotateQueryPipeline?.Dispose();
        _tqKvAppendPipeline?.Dispose();
        _tqAttentionPipeline?.Dispose();
        _embedLookupPipeline?.Dispose();
        _embedLookupQ4KPipeline?.Dispose();
        _embedLookupQ6KPipeline?.Dispose();
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

        // Tear down the validation messenger (if registered) before the instance.
        if (_validationEnabled && _debugMessenger.IsNotNull)
            _vki.vkDestroyDebugUtilsMessengerEXT(_debugMessenger);

        _vki.vkDestroyInstance(null);
    }
}
