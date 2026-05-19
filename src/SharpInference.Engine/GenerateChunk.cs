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
}

/// <summary>
/// A single typed chunk in an engine output stream. The wrapper itself is
/// allocation-free (<c>readonly record struct</c>); only the inner string allocates.
/// The boundary tokens themselves (<c>&lt;think&gt;</c> / <c>&lt;/think&gt;</c>) are
/// never emitted as content — they're protocol markers consumed by the engine.
/// </summary>
public readonly record struct GenerateChunk(GenerateChunkKind Kind, string Text);
