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
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// </summary>
    void TruncateTo(int length);

    /// <summary>Vocabulary size of the model.</summary>
    int VocabSize { get; }

    /// <summary>Maximum supported sequence length.</summary>
    int MaxSeqLen { get; }

    /// <summary>Reset the KV cache to empty (start of a new conversation).</summary>
    void ResetCache();
}
