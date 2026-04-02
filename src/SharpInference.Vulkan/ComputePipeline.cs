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
    private readonly VkDescriptorPool _descriptorPool;
    private readonly VkDescriptorSet _reusableDs; // single pre-allocated descriptor set
    private readonly int _pushConstantSize;
    private bool _disposed;

    // Descriptor set caching: skip vkUpdateDescriptorSets when bindings unchanged
    private ulong _cachedBindingHash;

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

        // 4. Compute pipeline
        var entryName = "main"u8;
        fixed (byte* entryPtr = entryName)
        {
            VkComputePipelineCreateInfo pipelineCI = new()
            {
                stage = new VkPipelineShaderStageCreateInfo
                {
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

        // 5. Descriptor pool
        VkDescriptorPoolSize poolSize = new()
        {
            type = VkDescriptorType.StorageBuffer,
            descriptorCount = (uint)(bindingCount * maxDescriptorSets),
        };
        VkDescriptorPoolCreateInfo poolCI = new()
        {
            flags = VkDescriptorPoolCreateFlags.FreeDescriptorSet,
            maxSets = (uint)maxDescriptorSets,
            poolSizeCount = 1,
            pPoolSizes = &poolSize,
        };
        VkDescriptorPool pool;
        vkd.vkCreateDescriptorPool(&poolCI, null, &pool).CheckResult();
        _descriptorPool = pool;

        // Pre-allocate one reusable descriptor set
        _reusableDs = AllocateDescriptorSet();
    }

    /// <summary>Allocate a descriptor set from the pool.</summary>
    public VkDescriptorSet AllocateDescriptorSet()
    {
        var layout = _descriptorSetLayout;
        VkDescriptorSetAllocateInfo allocInfo = new()
        {
            descriptorPool = _descriptorPool,
            descriptorSetCount = 1,
            pSetLayouts = &layout,
        };
        VkDescriptorSet ds;
        _backend.Vkd.vkAllocateDescriptorSets(&allocInfo, &ds).CheckResult();
        return ds;
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

    /// <summary>Record a dispatch using the reusable descriptor set (no submit).</summary>
    public void RecordWith(VkCommandBuffer cmd, ReadOnlySpan<GpuBuffer> buffers,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        // Hash buffer handles to detect binding changes
        ulong hash = (ulong)buffers.Length * 2654435761ul;
        for (int i = 0; i < buffers.Length; i++)
            hash ^= ((ulong)buffers[i].Buffer.Handle * 2654435761ul) << (i & 3);

        if (hash != _cachedBindingHash)
        {
            UpdateDescriptorSet(_reusableDs, buffers);
            _cachedBindingHash = hash;
        }
        RecordDispatch(cmd, _reusableDs, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    /// <summary>
    /// Update the reusable descriptor set, record, and submit a dispatch. Synchronous.
    /// Convenience method for the common pattern of bind-then-dispatch.
    /// </summary>
    public void DispatchWith(VkCommandBuffer cmd, ReadOnlySpan<GpuBuffer> buffers,
        uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1,
        void* pushConstants = null)
    {
        ulong hash = (ulong)buffers.Length * 2654435761ul;
        for (int i = 0; i < buffers.Length; i++)
            hash ^= ((ulong)buffers[i].Buffer.Handle * 2654435761ul) << (i & 3);

        if (hash != _cachedBindingHash)
        {
            UpdateDescriptorSet(_reusableDs, buffers);
            _cachedBindingHash = hash;
        }
        Dispatch(cmd, _reusableDs, groupCountX, groupCountY, groupCountZ, pushConstants);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        var vkd = _backend.Vkd;
        vkd.vkDestroyDescriptorPool(_descriptorPool, null);
        vkd.vkDestroyPipeline(_pipeline, null);
        vkd.vkDestroyPipelineLayout(_pipelineLayout, null);
        vkd.vkDestroyDescriptorSetLayout(_descriptorSetLayout, null);
        vkd.vkDestroyShaderModule(_shaderModule, null);
    }
}
