using System.Text.RegularExpressions;
using Vortice.Vulkan;

namespace SharpInference.Vulkan;

/// <summary>
/// Wraps a Vulkan compute pipeline with its shader module, descriptor set layout,
/// pipeline layout, descriptor pool, and descriptor sets.
///
/// Usage pattern:
///   1. Create with GLSL source and binding count
///   2. AllocateDescriptorSet() for each unique buffer combination
///   3. UpdateDescriptorSet() to bind GPU buffers
///   4. Dispatch() to record and submit compute work
/// </summary>
public sealed unsafe class ComputePipeline : IDisposable
{
    private readonly VulkanBackend _backend;
    private readonly VkShaderModule _shaderModule;
    private readonly VkDescriptorSetLayout _descriptorSetLayout;
    private readonly VkPipelineLayout _pipelineLayout;
    private readonly VkPipeline _pipeline;
    private readonly int _bindingCount;
    private readonly int _setsPerPool;
    private readonly List<VkDescriptorPool> _pools = [];
    private readonly VkDescriptorPool _reusablePool;        // separate pool for DispatchWith's reusable set
    private readonly VkDescriptorSet _reusableDs;           // for synchronous DispatchWith only
    private readonly List<VkDescriptorSet> _recordingSets = [];
    private long _lastSeenRecordingEpoch = -1;
    private int _recordingIdx;
    private readonly int _pushConstantSize;
    private bool _disposed;

    // Public for tests/diagnostics that need to assert per-recording DS isolation.
    public int CurrentRecordingDispatchCount => _recordingIdx;

    // Matches "local_size_x = <N>" inside a GLSL layout(...) qualifier (scoped to the
    // qualifier so a stray mention in a comment can't false-match). Every compute shader in
    // this backend declares it as a literal; used to decide subgroup-size pinning (issue #318).
    private static readonly Regex LocalSizeXRegex =
        new(@"layout\s*\([^)]*local_size_x\s*=\s*(\d+)", RegexOptions.Compiled);

    /// <summary>
    /// Parse the <c>local_size_x</c> workgroup dimension from GLSL source.
    /// Returns 0 if not found (which disables subgroup-size pinning — every real compute
    /// shader declares one, so a 0 result signals a malformed/non-compute source).
    /// </summary>
    internal static int ParseLocalSizeX(string glsl)
    {
        var m = LocalSizeXRegex.Match(glsl);
        return m.Success ? int.Parse(m.Groups[1].Value) : 0;
    }

    /// <summary>
    /// Whether requiredSubgroupSize=32 should be pinned for a shader with the given
    /// <paramref name="localSizeX"/> on the given <paramref name="backend"/> (issue #318).
    /// True only when:
    ///   • the extension is present, AND
    ///   • 32 lies within the supported subgroup-size range, AND
    ///   • the device is NOT already locked to exactly 32 (min==max==32) — on such devices
    ///     (e.g. NVIDIA) every subgroup is already 32 wide, so pinning is a pure no-op and we
    ///     skip it to avoid forcing the driver's required-subgroup-size pipeline path, AND
    ///   • the workgroup x-dimension is a whole multiple of 32 (so a 32-wide subgroup is valid —
    ///     pinning 32 on the 16-thread image-op workgroups would make pipeline creation fail).
    /// The bug only manifests when the device could pick a non-32 subgroup (AMD Wave64 → 64;
    /// Intel → possibly &lt;32), which is exactly the !(min==max==32) case.
    /// </summary>
    internal static bool ShouldPinSubgroupSize32(VulkanBackend backend, int localSizeX) =>
        backend.HasSubgroupSizeControl
        && backend.MinSubgroupSize <= 32 && 32 <= backend.MaxSubgroupSize
        && !(backend.MinSubgroupSize == 32 && backend.MaxSubgroupSize == 32)
        && localSizeX > 0 && localSizeX % 32 == 0;

    /// <summary>
    /// Create a compute pipeline from GLSL source.
    /// </summary>
    /// <param name="backend">The Vulkan backend (device, queue).</param>
    /// <param name="glslSource">GLSL compute shader source code.</param>
    /// <param name="bindingCount">Number of storage buffer bindings (binding 0, 1, 2, ...).</param>
    /// <param name="maxDescriptorSets">Max number of descriptor sets to allocate from the pool.</param>
    /// <param name="pushConstantSize">Size of push constant block in bytes (0 for none).</param>
    public ComputePipeline(VulkanBackend backend, string glslSource, int bindingCount,
        int maxDescriptorSets = 16, int pushConstantSize = 0)
    {
        _backend = backend;
        _pushConstantSize = pushConstantSize;
        var vkd = backend.Vkd;

        // 1. Compile GLSL → SPIR-V → VkShaderModule
        var spirv = ShaderCompiler.Compile(glslSource);
        fixed (byte* spirvPtr = spirv)
        {
            VkShaderModuleCreateInfo moduleCI = new()
            {
                codeSize = (nuint)spirv.Length,
                pCode = (uint*)spirvPtr,
            };
            VkShaderModule module;
            vkd.vkCreateShaderModule(&moduleCI, null, &module).CheckResult();
            _shaderModule = module;
        }

        // 2. Descriptor set layout: N storage buffer bindings
        var bindings = stackalloc VkDescriptorSetLayoutBinding[bindingCount];
        for (int i = 0; i < bindingCount; i++)
        {
            bindings[i] = new VkDescriptorSetLayoutBinding
            {
                binding = (uint)i,
                descriptorType = VkDescriptorType.StorageBuffer,
                descriptorCount = 1,
                stageFlags = VkShaderStageFlags.Compute,
            };
        }
        VkDescriptorSetLayoutCreateInfo layoutCI = new()
        {
            bindingCount = (uint)bindingCount,
            pBindings = bindings,
        };
        VkDescriptorSetLayout dsLayout;
        vkd.vkCreateDescriptorSetLayout(&layoutCI, null, &dsLayout).CheckResult();
        _descriptorSetLayout = dsLayout;

        // 3. Pipeline layout (descriptor set layout + optional push constants)
        VkPushConstantRange pushRange = new()
        {
            stageFlags = VkShaderStageFlags.Compute,
            offset = 0,
            size = (uint)pushConstantSize,
        };
        VkPipelineLayoutCreateInfo pipelineLayoutCI = new()
        {
            setLayoutCount = 1,
            pSetLayouts = &dsLayout,
            pushConstantRangeCount = pushConstantSize > 0 ? 1u : 0u,
            pPushConstantRanges = pushConstantSize > 0 ? &pushRange : null,
        };
        VkPipelineLayout pipeLayout;
        vkd.vkCreatePipelineLayout(&pipelineLayoutCI, null, &pipeLayout).CheckResult();
        _pipelineLayout = pipeLayout;

        // 4. Compute pipeline.
        // Pin requiredSubgroupSize=32 for reduction shaders (issue #318): they pack
        // "8 rows × 32 lanes" and assume a 32-wide subgroup, which breaks on AMD Wave64
        // (subgroup 64 → subgroupAdd sums across two rows). Pin only when the workgroup
        // x-dim is a multiple of 32 (the 256/128 reduction shaders) — pinning 32 on the
        // 16-thread image-op shaders would make the workgroup smaller than one subgroup and
        // pipeline creation would fail. On NVIDIA (subgroup already 32) this is a no-op.
        int localSizeX = ParseLocalSizeX(glslSource);
        bool pinSubgroup = ShouldPinSubgroupSize32(backend, localSizeX);
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo requiredSgSize = new()
        {
            requiredSubgroupSize = 32,
        };

        var entryName = "main"u8;
        // requiredSgSize is a stack local; &requiredSgSize stays valid for the whole synchronous
        // vkCreateComputePipelines call below. It is chained into the shader-stage pNext only when
        // pinning is enabled (issue #318).
        fixed (byte* entryPtr = entryName)
        {
            VkComputePipelineCreateInfo pipelineCI = new()
            {
                stage = new VkPipelineShaderStageCreateInfo
                {
                    pNext = pinSubgroup ? &requiredSgSize : null,
                    stage = VkShaderStageFlags.Compute,
                    module = _shaderModule,
                    pName = entryPtr,
                },
                layout = _pipelineLayout,
            };
            VkPipeline pipe;
            vkd.vkCreateComputePipelines(VkPipelineCache.Null, 1, &pipelineCI, null, &pipe).CheckResult();
            _pipeline = pipe;
        }

        // 5. Descriptor pools.
        // We keep two parallel pool universes:
        //   • _reusablePool — one set, reused by the synchronous `DispatchWith` path
        //     (no reuse hazard because it submits + waits before returning).
        //   • _pools[] — growable set of pools used by the asynchronous
        //     `RecordWith` path. A single Vulkan descriptor set must NOT be
        //     updated between record-time and submit/execute if a recorded
        //     dispatch is going to read from it (Vulkan validity rule), so the
        //     `RecordWith` path allocates a fresh DS per dispatch and recycles
        //     them all when the next recording session starts.
        _bindingCount = bindingCount;
        _setsPerPool = Math.Max(maxDescriptorSets, 64);

        VkDescriptorPoolSize reusablePoolSize = new()
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = (uint)bindingCount,
        };
        VkDescriptorPoolCreateInfo reusablePoolCI = new()
        {
            flags = VkDescriptorPoolCreateFlags.FreeDescriptorSet,
            maxSets = 1,
            poolSizeCount = 1,
            pPoolSizes = &reusablePoolSize,
        };
        VkDescriptorPool reusablePool;
        vkd.vkCreateDescriptorPool(&reusablePoolCI, null, &reusablePool).CheckResult();
        _reusablePool = reusablePool;
        _reusableDs = AllocateDescriptorSetFromPool(_reusablePool);

        _pools.Add(CreateRecordingPool());
    }

    private VkDescriptorPool CreateRecordingPool()
    {
        VkDescriptorPoolSize poolSize = new()
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = (uint)(_bindingCount * _setsPerPool),
        };
        VkDescriptorPoolCreateInfo poolCI = new()
        {
            // No FreeDescriptorSet flag — we recycle the pool wholesale via
            // vkResetDescriptorPool when a new recording session begins.
            maxSets = (uint)_setsPerPool,
            poolSizeCount = 1,
            pPoolSizes = &poolSize,
        };
        VkDescriptorPool pool;
        _backend.Vkd.vkCreateDescriptorPool(&poolCI, null, &pool).CheckResult();
        return pool;
    }

    private VkDescriptorSet AllocateDescriptorSetFromPool(VkDescriptorPool pool)
    {
        var layout = _descriptorSetLayout;
        VkDescriptorSetAllocateInfo allocInfo = new()
        {
            descriptorPool = pool,
            descriptorSetCount = 1,
            pSetLayouts = &layout,
        };
        VkDescriptorSet ds;
        _backend.Vkd.vkAllocateDescriptorSets(&allocInfo, &ds).CheckResult();
        return ds;
    }

    /// <summary>
    /// Allocate a descriptor set from the recording pool, growing the pool list on demand.
    /// </summary>
    public VkDescriptorSet AllocateDescriptorSet()
    {
        int poolIdx = _recordingSets.Count / _setsPerPool;
        if (poolIdx >= _pools.Count)
            _pools.Add(CreateRecordingPool());
        return AllocateDescriptorSetFromPool(_pools[poolIdx]);
    }

    /// <summary>
    /// Update a descriptor set to bind GPU buffers at sequential bindings (0, 1, 2, ...).
    /// </summary>
    public void UpdateDescriptorSet(VkDescriptorSet ds, params ReadOnlySpan<GpuBuffer> buffers)
    {
        var writes = stackalloc VkWriteDescriptorSet[buffers.Length];
        var bufferInfos = stackalloc VkDescriptorBufferInfo[buffers.Length];

        for (int i = 0; i < buffers.Length; i++)
        {
            bufferInfos[i] = new VkDescriptorBufferInfo
            {
                buffer = buffers[i].Buffer,
                offset = 0,
                range = buffers[i].Size,
            };
            writes[i] = new VkWriteDescriptorSet
            {
                dstSet = ds,
                dstBinding = (uint)i,
                dstArrayElement = 0,
                descriptorCount = 1,
                descriptorType = VkDescriptorType.StorageBuffer,
                pBufferInfo = &bufferInfos[i],
            };
        }
        _backend.Vkd.vkUpdateDescriptorSets((uint)buffers.Length, writes, 0, null);
    }

    /// <summary>
    /// Record and submit a compute dispatch. Synchronous (waits for completion).
    /// </summary>
    public void Dispatch(VkCommandBuffer cmd, VkDescriptorSet ds,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        var vkd = _backend.Vkd;

        VkCommandBufferBeginInfo beginInfo = new()
        {
            flags = VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        vkd.vkBeginCommandBuffer(cmd, &beginInfo).CheckResult();

        RecordDispatch(cmd, ds, groupCountX, groupCountY, groupCountZ, pushConstants);

        vkd.vkEndCommandBuffer(cmd).CheckResult();

        var fence = _backend.Fence;
        vkd.vkResetFences(1, &fence).CheckResult();
        VkSubmitInfo submitInfo = new()
        {
            commandBufferCount = 1,
            pCommandBuffers = &cmd,
        };
        vkd.vkQueueSubmit(_backend.ComputeQueue, 1, &submitInfo, fence).CheckResult();
        vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    /// <summary>
    /// Record a dispatch into an already-recording command buffer (no submit/wait).
    /// Used for batching multiple dispatches into one submission.
    /// </summary>
    public void RecordDispatch(VkCommandBuffer cmd, VkDescriptorSet ds,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        var vkd = _backend.Vkd;
        vkd.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _pipeline);
        vkd.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _pipelineLayout,
            0, 1, &ds, 0, null);
        if (pushConstants != null && _pushConstantSize > 0)
            vkd.vkCmdPushConstants(cmd, _pipelineLayout, VkShaderStageFlags.Compute,
                0, (uint)_pushConstantSize, pushConstants);
        vkd.vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
    }

    /// <summary>
    /// Record a dispatch into the current command buffer using a fresh descriptor set.
    ///
    /// Vulkan §14.2.4 makes it undefined behaviour to call <c>vkUpdateDescriptorSets</c>
    /// on a descriptor set that was bound by a recorded but not-yet-completed dispatch.
    /// A single <c>_reusableDs</c> updated and bound N times in one command buffer means
    /// every recorded dispatch reads the descriptor state at execution time — i.e. the
    /// last update wins, and earlier dispatches see the wrong buffers. NVIDIA drivers
    /// usually snapshot at bind, but Mesa / Intel do not. This per-dispatch DS allocation
    /// fixes the reuse hazard regardless of driver behaviour.
    /// </summary>
    public void RecordWith(VkCommandBuffer cmd, ReadOnlySpan<GpuBuffer> buffers,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        // If the backend started a new recording session, recycle every set allocated
        // during the previous session by resetting all of this pipeline's recording pools.
        long epoch = _backend.RecordingEpoch;
        if (epoch != _lastSeenRecordingEpoch)
        {
            ResetRecordingPools();
            _lastSeenRecordingEpoch = epoch;
        }

        if (_recordingIdx >= _recordingSets.Count)
            _recordingSets.Add(AllocateDescriptorSet());

        var ds = _recordingSets[_recordingIdx++];
        UpdateDescriptorSet(ds, buffers);
        RecordDispatch(cmd, ds, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    private void ResetRecordingPools()
    {
        var vkd = _backend.Vkd;
        foreach (var pool in _pools)
            vkd.vkResetDescriptorPool(pool, 0).CheckResult();
        _recordingSets.Clear();
        _recordingIdx = 0;
    }

    /// <summary>
    /// Update the reusable descriptor set, record, and submit a dispatch. Synchronous.
    /// Convenience method for the common pattern of bind-then-dispatch.
    /// </summary>
    public void DispatchWith(VkCommandBuffer cmd, ReadOnlySpan<GpuBuffer> buffers,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        // Always update descriptor set; VkBuffer handles can be reused after Free()
        // causing a hash collision that would silently bind stale/freed descriptors.
        UpdateDescriptorSet(_reusableDs, buffers);
        Dispatch(cmd, _reusableDs, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var vkd = _backend.Vkd;
        foreach (var pool in _pools)
            vkd.vkDestroyDescriptorPool(pool, null);
        vkd.vkDestroyDescriptorPool(_reusablePool, null);
        vkd.vkDestroyPipeline(_pipeline, null);
        vkd.vkDestroyPipelineLayout(_pipelineLayout, null);
        vkd.vkDestroyDescriptorSetLayout(_descriptorSetLayout, null);
        vkd.vkDestroyShaderModule(_shaderModule, null);
    }
}
