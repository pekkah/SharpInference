using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// An opaque per-sequence KV-cache handle owned by <see cref="ContinuousBatchingEngine"/>. The
/// engine allocates one per admitted sequence via <see cref="IBatchedForwardPass.CreateCache"/>,
/// threads it back through the batched forward-pass calls, and disposes it on retirement — it
/// never inspects the contents, so the concrete type (the CPU <see cref="PagedKvCache"/> today,
/// a GPU-resident cache tomorrow) stays private to the forward pass that produced it (issue #190).
/// </summary>
public interface ISequenceKvCache : IDisposable
{
}

/// <summary>
/// The forward-pass surface <see cref="ContinuousBatchingEngine"/> needs, decoupled from any
/// concrete backend (issue #190). The engine drives prefill + ragged batched decode and reads a
/// few scheduling properties; it holds per-sequence caches only as opaque
/// <see cref="ISequenceKvCache"/> handles. Implemented by the CPU <see cref="ForwardPass"/>
/// today; a CUDA implementation (per-sequence GPU KV caches + batched decode) can slot in behind
/// the same interface without the engine knowing which backend it drives.
/// </summary>
public interface IBatchedForwardPass : IThreadAffineBackend
{
    /// <summary>Allocate a fresh, empty per-sequence KV cache for one admitted sequence.</summary>
    ISequenceKvCache CreateCache();

    /// <summary>
    /// Prefill <paramref name="tokens"/> into <paramref name="cache"/> starting at
    /// <paramref name="startPos"/> (chunked admission calls this repeatedly with advancing
    /// <paramref name="startPos"/>); returns the logits at the final token.
    /// </summary>
    ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0);

    /// <summary>
    /// Prefill several pending sequences' chunks in ONE packed forward pass (weights amortized
    /// across the prompts, like <see cref="BatchForwardMulti"/> does for decode). Returns the
    /// per-sequence logits where <paramref name="wantLogits"/> is set (the chunk completes that
    /// prompt), and null otherwise.
    /// </summary>
    float[]?[] PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits);

    /// <summary>
    /// Ragged batched decode: one token per sequence, each at its own <paramref name="positions"/>
    /// against its own <paramref name="caches"/> entry, with the weight reads amortized N× across
    /// the batch. Returns the per-sequence next-token logits.
    /// </summary>
    float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches);

    /// <summary>
    /// Whether <see cref="BatchForwardMultiArgmax"/> avoids the full per-row logits download by
    /// running the greedy argmax on-device (mirrors <see cref="IForwardPass.SupportsGpuArgmax"/>,
    /// issue #219). Default false — backends without a GPU tail to elide use the full path.
    /// </summary>
    bool SupportsBatchedGpuArgmax => false;

    /// <summary>
    /// Greedy counterpart of <see cref="BatchForwardMulti"/>: returns only the per-sequence
    /// (argmax token, its logit) instead of the full N×vocab logits — a rows*8-byte device→host
    /// copy in place of the N×vocab download + host split (issue #205/#206 follow-up). Valid only
    /// when <see cref="SupportsBatchedGpuArgmax"/> and every sequence in the batch is greedy
    /// (temp ≤ 0, no forced-token override); the caller gates on those before invoking. Default
    /// throws — callers must check <see cref="SupportsBatchedGpuArgmax"/> first.
    /// </summary>
    (int Token, float Logit)[] BatchForwardMultiArgmax(int[] tokens, int[] positions, ISequenceKvCache[] caches) =>
        throw new NotSupportedException(
            $"{GetType().Name} does not implement BatchForwardMultiArgmax. Check SupportsBatchedGpuArgmax first.");

    /// <summary>
    /// Whether SnapKV prefill eviction is configured. The engine disables chunked/packed prefill
    /// when true (SnapKV scoring only runs on a fresh full-prompt prefill).
    /// </summary>
    bool SnapKvEnabled { get; }

    /// <summary>
    /// Bytes of KV one token occupies across all layers — the engine converts a memory budget
    /// into a token budget for admission backpressure (issue #183).
    /// </summary>
    long KvBytesPerToken { get; }

    /// <summary>Maximum supported sequence length; caps each request's projected KV reservation.</summary>
    int MaxSeqLen { get; }

    /// <summary>
    /// Whether the dequant-once weight cache is budgeted to cover the model (issue #189); the
    /// engine picks a smaller default prefill chunk when true.
    /// </summary>
    bool PrefillDequantCacheActive { get; }
}
