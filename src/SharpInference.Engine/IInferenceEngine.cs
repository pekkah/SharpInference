using System.Runtime.CompilerServices;

namespace SharpInference.Engine;

/// <summary>
/// Abstraction over an inference engine: tokenizes input, runs the model, and yields decoded chunks.
/// The caller is responsible for applying chat templates before passing the prompt.
/// </summary>
public interface IInferenceEngine
{
    /// <summary>Identifier of the loaded model (used in API responses).</summary>
    string ModelId { get; }

    /// <summary>Number of requests waiting to be admitted for generation.</summary>
    int QueueDepth { get; }

    /// <summary>Number of requests currently being generated.</summary>
    int ActiveRequests { get; }

    /// <summary>
    /// Generate typed chunks from a pre-formatted prompt string. Each chunk is tagged as
    /// either user-facing <see cref="GenerateChunkKind.Text"/> or internal
    /// <see cref="GenerateChunkKind.Thinking"/> reasoning. The boundary tokens
    /// (<c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>) are consumed by the engine and never
    /// surface in chunk text. Requests are serialized — concurrent calls block until
    /// the current request finishes.
    /// </summary>
    IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        CancellationToken ct = default);

    /// <summary>
    /// Generate text from a pre-formatted prompt string. Yields decoded text chunks
    /// (one or more characters) as they are produced. Reasoning content emitted inside
    /// <c>&lt;think&gt;...&lt;/think&gt;</c> is suppressed — only user-facing answer text
    /// is yielded. Requests are serialized — concurrent calls block until the current
    /// request finishes.
    /// </summary>
    /// <remarks>
    /// Default implementation adapts <see cref="GenerateChunksAsync"/>, yielding only
    /// <see cref="GenerateChunkKind.Text"/> chunks. Implementations may override for efficiency,
    /// but the typed stream is the canonical source of truth.
    /// </remarks>
    async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (var c in GenerateChunksAsync(prompt, sp, ct).WithCancellation(ct).ConfigureAwait(false))
        {
            if (c.Kind == GenerateChunkKind.Text)
                yield return c.Text;
        }
    }
}
