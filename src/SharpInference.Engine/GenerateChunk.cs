namespace SharpInference.Engine;

/// <summary>
/// Classifies a <see cref="GenerateChunk"/> as user-facing answer text or as
/// internal reasoning content emitted inside a <c>&lt;think&gt;...&lt;/think&gt;</c> block.
/// Endpoints route each kind into the protocol-appropriate field
/// (Anthropic <c>thinking</c> blocks, OpenAI <c>reasoning_content</c>).
/// </summary>
public enum GenerateChunkKind
{
    /// <summary>Normal answer text — the content the end user sees.</summary>
    Text,

    /// <summary>Reasoning content emitted between <c>&lt;think&gt;</c> and <c>&lt;/think&gt;</c>.</summary>
    Thinking,

    /// <summary>
    /// Out-of-band usage metadata emitted once per request (not user-facing text). Its
    /// <see cref="GenerateChunk.Text"/> is empty; <see cref="GenerateChunk.PromptTokens"/>
    /// carries the encoded prompt-token count so endpoints can populate
    /// <c>usage.prompt_tokens</c> / <c>usage.input_tokens</c> (issue #150) without
    /// re-tokenizing. Consumers that only surface text (e.g. the default
    /// <see cref="IInferenceEngine.GenerateAsync"/> adapter) ignore it.
    /// </summary>
    Usage,

    /// <summary>
    /// Out-of-band chunk emitted once, at the very end of a successful (non-cancelled,
    /// non-errored) generation — after all Text/Thinking chunks. Its
    /// <see cref="GenerateChunk.Text"/> is empty; <see cref="GenerateChunk.TruncatedByMaxTokens"/>
    /// tells the caller whether decoding stopped because the token budget
    /// (<c>SamplingParams.MaxNewTokens</c>) was exhausted rather than a natural stop
    /// token/EOS. Endpoints use this to report <c>finish_reason: "length"</c> /
    /// <c>stop_reason: "max_tokens"</c> instead of <c>"stop"</c>/<c>"end_turn"</c> when the
    /// response was cut off mid-thought with no tool call in progress. Engines that don't
    /// emit this chunk (e.g. test doubles) leave callers defaulting to "not truncated".
    /// </summary>
    Stop,
}

/// <summary>
/// A single typed chunk in an engine output stream. The wrapper itself is
/// allocation-free (<c>readonly record struct</c>); only the inner string allocates.
/// The boundary tokens themselves (<c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>) are
/// never emitted as content — they're protocol markers consumed by the engine.
/// <para>
/// <see cref="PromptTokens"/> is meaningful only on a <see cref="GenerateChunkKind.Usage"/>
/// chunk (0 on Text/Thinking chunks). <see cref="TruncatedByMaxTokens"/> is meaningful only
/// on a <see cref="GenerateChunkKind.Stop"/> chunk (false on Text/Thinking/Usage chunks).
/// </para>
/// </summary>
public readonly record struct GenerateChunk(
    GenerateChunkKind Kind, string Text, int PromptTokens = 0, bool TruncatedByMaxTokens = false);
