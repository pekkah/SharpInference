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
}

/// <summary>
/// A single typed chunk in an engine output stream. The wrapper itself is
/// allocation-free (<c>readonly record struct</c>); only the inner string allocates.
/// The boundary tokens themselves (<c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>) are
/// never emitted as content — they're protocol markers consumed by the engine.
/// <para>
/// <see cref="PromptTokens"/> is meaningful only on a <see cref="GenerateChunkKind.Usage"/>
/// chunk (0 on Text/Thinking chunks).
/// </para>
/// </summary>
public readonly record struct GenerateChunk(GenerateChunkKind Kind, string Text, int PromptTokens = 0);
