using SharpInference.Core;

namespace SharpInference.Pipeline;

/// <summary>
/// Three-tier memory hierarchy: GPU VRAM ? CPU RAM ? NVMe.
/// Manages tensor placement, eviction, and asynchronous prefetching
/// to keep hot weights resident on the fastest available tier.
/// </summary>
public sealed class MemoryHierarchy : IAsyncDisposable
{
    private readonly TierConfig _gpu;
    private readonly TierConfig _cpu;
    private readonly TierConfig _nvme;

    public MemoryHierarchy(TierConfig gpu, TierConfig cpu, TierConfig nvme)
    {
        _gpu = gpu;
        _cpu = cpu;
        _nvme = nvme;
    }

    /// <summary>Ensure <paramref name="tensorName"/> is resident on the GPU tier.</summary>
    public ValueTask<Tensor> PromoteToGpuAsync(string tensorName, CancellationToken ct = default)
    {
        // TODO: move tensor up the hierarchy, evicting LRU tensors as needed
        throw new NotImplementedException();
    }

    /// <summary>Evict the least-recently-used tensor from GPU to CPU tier.</summary>
    public ValueTask EvictFromGpuAsync(CancellationToken ct = default)
    {
        // TODO: LRU eviction with dirty-tracking
        throw new NotImplementedException();
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

public sealed record TierConfig(string Name, long CapacityBytes, string? MmapPath = null);
