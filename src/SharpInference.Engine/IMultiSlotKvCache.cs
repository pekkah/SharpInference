namespace SharpInference.Engine;

/// <summary>
/// Optional capability: a forward pass that can hold more than one resident KV region
/// (the native "owned" cache plus extra "scratch" regions) so a short interleaved request
/// can be served without evicting a long resident prefix in the owned cache. Issue #212.
/// Implemented by the dense CudaForwardPass; null/absent for backends that don't support it.
/// </summary>
internal interface IMultiSlotKvCache
{
    /// <summary>True when extra KV slots are allocatable (dense, non-TQ/SnapKV/MoE/Gemma-shared, etc.).</summary>
    bool SupportsMultiSlotPrefix { get; }

    /// <summary>The native owned KV region as a slot handle (slot 0). Never disposed by the engine.</summary>
    ISequenceKvCache OwnedSlot { get; }

    /// <summary>Allocate one extra resident KV region capped at <paramref name="capacityTokens"/> positions.
    /// Engine owns/disposes it.</summary>
    ISequenceKvCache AllocateScratchSlot(int capacityTokens);

    /// <summary>Bind <paramref name="slot"/> as the active KV region for the whole request
    /// (prefill + decode + truncate). Subsequent Prefill/Forward/TruncateTo/ResetCache operate on it.</summary>
    void ActivateSlot(ISequenceKvCache slot);

    /// <summary>Write the active slot's advanced length back and restore the owned region. Idempotent.</summary>
    void DeactivateSlot();
}
