using System.Text.Json;
using SharpInference.Core.Grammar;

namespace SharpInference.Server.Endpoints;

/// <summary>
/// Shared plumbing for schema/grammar-constrained tool-call decoding (issue #374): the opt-in gate
/// and the conversion of an endpoint's wire-format tool definitions into the engine's
/// <see cref="ToolSchema"/> model. Kept endpoint-agnostic so the OpenAI and Anthropic paths build
/// the constraint identically.
/// </summary>
internal static class ToolGrammarHelper
{
    /// <summary>
    /// Whether grammar-constrained tool decoding is enabled — driven solely by the
    /// <see cref="SharpInferenceServerOptions.ToolGrammar"/> option, the single source of truth.
    /// The runnable host still honours <c>SHARPI_TOOL_GRAMMAR=1</c> by mapping that environment
    /// variable onto the option at startup (mirrors the SHARPI_* env pattern), so the library layer
    /// doesn't re-read the environment here.
    /// </summary>
    public static bool Enabled(SharpInferenceServerOptions opts) => opts.ToolGrammar;

    /// <summary>Parses (name, JSON-Schema) pairs into <see cref="ToolSchema"/>s, skipping nameless entries.</summary>
    public static List<ToolSchema> ToSchemas(IEnumerable<(string? Name, JsonElement? Parameters)> tools)
    {
        var schemas = new List<ToolSchema>();
        foreach (var (name, parameters) in tools)
            if (name is { Length: > 0 })
                schemas.Add(ToolSchema.FromOpenAiFunction(name, parameters));
        return schemas;
    }
}
