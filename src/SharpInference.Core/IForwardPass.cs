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
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// </summary>
    void TruncateTo(int length);

    /// <summary>Vocabulary size of the model.</summary>
    int VocabSize { get; }

    /// <summary>Maximum supported sequence length.</summary>
    int MaxSeqLen { get; }
}
