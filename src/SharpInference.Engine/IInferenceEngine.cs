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
    /// Whether this engine is *capable* of prefix-cache reuse — not whether a reuse just
    /// happened. Returns <c>false</c> when the underlying forward pass cannot partially
    /// rewind its KV cache (e.g. Gated DeltaNet hybrid models like Qwen3.6); in that case
    /// every request re-prefills the full prompt. To observe whether prefix reuse is
    /// actually saving work, scrape <see cref="PrefillTokensReused"/>.
    /// </summary>
    bool PrefixCacheEnabled { get; }

    /// <summary>
    /// Running count of prompt tokens skipped via the prefix-cache fast path across the
    /// lifetime of this engine instance. <c>long</c> because the value can grow unbounded
    /// on a long-running server.
    /// </summary>
    long PrefillTokensReused { get; }

    /// <summary>
    /// Generate typed chunks from a pre-formatted prompt string. Each chunk is tagged as
    /// either user-facing <see cref="GenerateChunkKind.Text"/> or internal
    /// <see cref="GenerateChunkKind.Thinking"/> reasoning. The boundary tokens
    /// (<c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>) are consumed by the engine and never
    /// surface in chunk text. Requests are serialized — concurrent calls block until
    /// the current request finishes.
    /// </summary>
    /// <param name="canonicalHistoryPrefix">
    /// Optional chat-template render of the request's message history rendered with
    /// <c>add_generation_prompt=false</c> — i.e. how the same messages would appear as
    /// pure history in a future turn, with no assistant-generation prefix appended
    /// (no <c>&lt;|im_start|&gt;assistant\n&lt;think&gt;</c> tail for Qwen3-style
    /// templates). When provided, must be a string-prefix of <paramref name="prompt"/>;
    /// the engine validates it tokenizes to a strict token-level prefix and uses that
    /// boundary as the position to snapshot KV state for next-turn prefix reuse
    /// (issue #102). If the validation fails the engine falls back to the legacy
    /// generating-prompt-only path. Endpoints that don't render a separate canonical
    /// (CLI completions, raw /v1/completions) pass <c>null</c>.
    /// </param>
    IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        CancellationToken ct = default,
        string? canonicalHistoryPrefix = null);

    /// <summary>
    /// Whether this engine can accept image input (issue #253). True only when an mmproj
    /// vision projector is loaded <i>and</i> the underlying forward pass supports
    /// precomputed-embedding input (CPU or full-CUDA Gemma 4 today). Endpoints check this
    /// before routing an image request and return a clear error otherwise.
    /// </summary>
    bool SupportsImageInput => false;

    /// <summary>
    /// Generate from a pre-formatted prompt that contains the model's <c>&lt;|image|&gt;</c>
    /// placeholder token(s), splicing each image's projected soft-token embeddings in place of
    /// the i-th placeholder (left to right). <paramref name="imageBytes"/> holds the raw encoded
    /// image bytes (e.g. base64-decoded PNG), one per placeholder, in the same order. Same typed
    /// chunk semantics as <see cref="GenerateChunksAsync"/>. The number of placeholders in the
    /// rendered prompt must equal <c>imageBytes.Count</c>.
    /// </summary>
    IAsyncEnumerable<GenerateChunk> GenerateImageChunksAsync(
        string prompt,
        IReadOnlyList<byte[]> imageBytes,
        SamplingParams sp,
        CancellationToken ct = default) =>
        throw new NotSupportedException(
            $"{GetType().Name} does not support image input. Check SupportsImageInput first.");

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
