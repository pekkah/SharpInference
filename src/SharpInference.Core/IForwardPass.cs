namespace SharpInference.Core;

/// <summary>
/// Abstraction over a transformer forward pass. Supports both autoregressive decode (Forward)
/// and position truncation needed by speculative decoding.
/// </summary>
public interface IForwardPass : IDisposable
{
    /// <summary>Run one token through the model and return logits[vocabSize].</summary>
    ReadOnlySpan<float> Forward(int token, int position);

    /// <summary>
    /// Batch-process prompt tokens and return logits for the last token.
    /// Faster than sequential Forward() calls for long prompts due to batched GEMM.
    /// </summary>
    /// <param name="tokens">Prompt token IDs to process.</param>
    /// <param name="startPos">
    /// Position at which to begin writing into the KV cache (default 0).
    /// Set to the prefix length when reusing a cached prefix (cache must already have
    /// K/V for positions 0..startPos-1 from a previous call).
    /// </param>
    ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0);

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions &gt;= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// <para>
    /// Implementations that return <c>false</c> from <see cref="SupportsPartialRewind"/>
    /// accept only <c>length == 0</c> (full reset) or <c>length == currentCacheLength</c>
    /// (no-op); any other value must throw <see cref="NotSupportedException"/>. Callers that
    /// need to rewind to an intermediate position must check <see cref="SupportsPartialRewind"/>
    /// first.
    /// </para>
    /// </summary>
    void TruncateTo(int length);

    /// <summary>Vocabulary size of the model.</summary>
    int VocabSize { get; }

    /// <summary>Maximum supported sequence length.</summary>
    int MaxSeqLen { get; }

    /// <summary>Reset the KV cache to empty (start of a new conversation).</summary>
    void ResetCache();

    /// <summary>
    /// Whether <see cref="TruncateTo"/> accepts arbitrary intermediate lengths (i.e. values
    /// other than 0 or the current cache length). Defaults to <c>false</c> so new
    /// implementations are safe by default; rewindable transformer passes must opt in by
    /// overriding this to <c>true</c>. Models whose state is destructively updated per
    /// token (e.g. Gated DeltaNet hybrid) leave this at the default. Consumed by the
    /// inference engine and speculative decoder to skip code paths that would otherwise
    /// throw on partial rewind.
    /// </summary>
    bool SupportsPartialRewind => false;

    /// <summary>
    /// Length (in tokens) of the most recently captured end-of-decode snapshot, or
    /// <c>-1</c> when no snapshot is held. Used by <see cref="InferenceEngine"/> to
    /// reuse cached state across chat-continuation turns on forward passes that don't
    /// support arbitrary partial rewind (specifically GDN hybrids — issue #21).
    /// The default <c>-1</c> means "no snapshot facility"; rewindable transformer
    /// passes ignore this and the engine instead consults <see cref="SupportsPartialRewind"/>.
    /// </summary>
    int SnapshotLength => -1;

    /// <summary>
    /// Capture a snapshot of the current cache state so a subsequent
    /// <see cref="TruncateTo"/> at <see cref="SnapshotLength"/> can restore it.
    /// Called by the inference engine at end-of-decode for non-canonical paths,
    /// or mid-prefill at the canonical-history boundary for chat-continuation
    /// reuse (issue #102); no-op by default. Implementations that support
    /// snapshots must update <see cref="SnapshotLength"/> to the cache's current
    /// token length when this returns AND override
    /// <see cref="SupportsSnapshot"/> to <c>true</c>.
    /// </summary>
    void CaptureSnapshot() { }

    /// <summary>
    /// Static capability flag: <c>true</c> iff this pass implements
    /// <see cref="CaptureSnapshot"/> / <see cref="SnapshotLength"/>. Distinct from
    /// <see cref="SnapshotLength"/> because that value is <c>-1</c> both when no
    /// snapshot has been captured yet AND when the pass doesn't support snapshots
    /// at all — callers need to know the capability up-front (e.g. constructor-time
    /// log lines, <c>/metrics</c> exposition) before any request has run.
    /// Implementations that override <see cref="CaptureSnapshot"/> must override this to <c>true</c>.
    /// </summary>
    bool SupportsSnapshot => false;

    // ── Multi-Token Prediction (MTP / NEXTN) self-speculative decoding ──
    //
    // Forward passes that load an MTP head report HasMtpHead = true; callers can
    // then route through MtpDecoder. The MTP head is a single transformer block
    // appended at GGUF block index NumLayers (== blk.{NumLayers}), with its own
    // attention KV cache slot that is independent of the main trunk's KV cache
    // but advances in lockstep (one position per forward).

    /// <summary>
    /// True when this pass has loaded an MTP / NEXTN head and can serve
    /// <see cref="MtpForward"/> calls.
    /// </summary>
    bool HasMtpHead => false;

    /// <summary>
    /// Last token's post-trunk pre-final-norm hidden state (length =
    /// EmbeddingDim). Refreshed at the end of every <see cref="Forward"/> /
    /// <see cref="Prefill"/> call. Consumed by <see cref="MtpForward"/> as the
    /// "previous hidden" input. Returns an empty span when <see cref="HasMtpHead"/>
    /// is false.
    /// </summary>
    ReadOnlySpan<float> LastHidden => default;

    /// <summary>
    /// Drive the MTP head for one draft step.
    /// </summary>
    /// <param name="token">Token at <paramref name="position"/> (= argmax of the
    /// most recent main logits — already known correct by greedy construction).</param>
    /// <param name="position">Absolute position where <paramref name="token"/> sits.
    /// Same position the main pass would use if it were processing the token.</param>
    /// <param name="prevHidden">Previous main forward's <see cref="LastHidden"/>;
    /// length must equal EmbeddingDim.</param>
    /// <returns>Logits predicting the token at <paramref name="position"/> + 1.</returns>
    ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden) =>
        throw new NotSupportedException(
            $"{GetType().Name} does not implement an MTP head. Check HasMtpHead before calling.");

    /// <summary>
    /// Populate the MTP attention KV cache for a sequence of prompt tokens, calling
    /// <see cref="MtpForward"/> at each position with the previous main forward's
    /// hidden state. Must be called after a matching <see cref="Prefill"/> so that
    /// the MTP head has access to per-position hiddens (the implementation buffers
    /// them during <see cref="Prefill"/> when <see cref="HasMtpHead"/> is true).
    /// <para>
    /// Without this, the MTP attention KV cache is empty at the first decode step
    /// and the MTP head's attention scores only see its own freshly-written K/V at
    /// the decode position — issue #33. Self-parity tests don't detect this because
    /// the emitted sequence is always argmax(main_logits); MTP KV quality only
    /// affects acceptance rate (and thus speculative speedup).
    /// </para>
    /// </summary>
    /// <param name="tokens">Prompt token IDs to populate MTP KV for. Should match the
    /// list passed to the preceding <see cref="Prefill"/>.</param>
    /// <param name="startPos">Position at which the first token sits (== same
    /// <c>startPos</c> as the preceding <see cref="Prefill"/>).</param>
    /// <remarks>Default is a no-op so non-MTP forward passes need no override.</remarks>
    void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0) { }

    /// <summary>
    /// True when this pass implements a batched two-token verify path (issue #30).
    /// Callers (<see cref="MtpDecoder"/>) dispatch to <see cref="BatchForward2"/> on
    /// the hybrid GDN passes where it pays off; everything else stays on the
    /// sequential N=1 algorithm.
    /// </summary>
    bool SupportsBatchVerify => false;

    /// <summary>
    /// Last completed <see cref="BatchForward2"/>'s token-1 pre-output-norm hidden.
    /// Used by the MTP commit step on the batched verify path. Empty when no batched
    /// forward has been run.
    /// </summary>
    ReadOnlySpan<float> LastHiddenT1 => default;

    /// <summary>
    /// Two-token batched forward (issue #30). On entry both caches must be at length
    /// <paramref name="startPos"/>. On return both caches are at length
    /// <c>startPos + 2</c>, <see cref="LastHidden"/> holds h@startPos+1, and
    /// <see cref="LastHiddenT1"/> holds h@startPos. A per-layer GDN snapshot is
    /// captured at the "between t1 and t2" point so a rejected draft can be rolled
    /// back via <see cref="RestoreBatchSnapshot"/>.
    /// </summary>
    void BatchForward2(int t1, int t2, int startPos,
        out ReadOnlySpan<float> logits1, out ReadOnlySpan<float> logits2) =>
        throw new NotSupportedException(
            $"{GetType().Name} does not implement BatchForward2. " +
            "Check SupportsBatchVerify before calling.");

    /// <summary>
    /// Roll caches back to <paramref name="lengthAfter"/> using the snapshot taken
    /// in the most recent <see cref="BatchForward2"/>. Called by the MTP decoder
    /// when the t2 draft is rejected; the follow-up <see cref="Forward"/> with the
    /// corrected token then advances state back to <c>startPos + 2</c>.
    /// </summary>
    void RestoreBatchSnapshot(int lengthAfter) =>
        throw new NotSupportedException(
            $"{GetType().Name} does not implement RestoreBatchSnapshot. " +
            "Check SupportsBatchVerify before calling.");

    /// <summary>Reset the MTP attention KV cache. No-op when <see cref="HasMtpHead"/> is false.</summary>
    void MtpResetCache() { }

    /// <summary>
    /// Truncate the MTP attention KV cache to the given length. Used by the MTP
    /// verify-and-accept loop to roll back rejected draft positions. The MTP
    /// attention KV cache is a standard paged cache (supports arbitrary lengths
    /// up to its current length); destructive GDN state on the main pass is a
    /// separate concern handled via <see cref="TruncateTo"/> / snapshots.
    /// </summary>
    void MtpTruncateTo(int length) { }
}
