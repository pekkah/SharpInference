namespace SharpInference.Engine;

/// <summary>
/// Abstraction over an inference engine: tokenizes input, runs the model, and yields decoded text chunks.
/// The caller is responsible for applying chat templates before passing the prompt.
/// </summary>
public interface IInferenceEngine
{
    /// <summary>Identifier of the loaded model (used in API responses).</summary>
    string ModelId { get; }

    /// <summary>
    /// Generate text from a pre-formatted prompt string.
    /// Yields decoded text chunks (one or more characters) as they are produced.
    /// Requests are serialized — concurrent calls block until the current request finishes.
    /// </summary>
    IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        CancellationToken ct = default);
}
