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
}
