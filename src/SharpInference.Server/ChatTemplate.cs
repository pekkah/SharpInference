using System.Text;
using System.Text.RegularExpressions;
using SharpInference.Core;
using SharpInference.Core.Grammar;

namespace SharpInference.Server;

/// <summary>
/// Static helpers for chat-template handling that don't need DI: regex-based scrubbing of
/// historical <c>&lt;think&gt;...&lt;/think&gt;</c> blocks. Rendering of new prompts lives on
/// the DI-resolved <see cref="ChatTemplateRenderer"/>.
/// </summary>
public static partial class ChatTemplate
{
    // NonBacktracking keeps the regex AOT-safe and immune to pathological input.
    // Greedy match (first `<think>` to last `</think>`) is correct for reasoning
    // models, which produce exactly one block per turn at the start of the answer;
    // it also collapses pathological nested/duplicated tags without orphan leakage.
    [GeneratedRegex(@"<think>[\s\S]*</think>", RegexOptions.NonBacktracking)]
    private static partial Regex ThinkBlockRegex();

    [GeneratedRegex(@"<think>[\s\S]*$", RegexOptions.NonBacktracking)]
    private static partial Regex UnclosedThinkRegex();

    [GeneratedRegex(@"</think>", RegexOptions.NonBacktracking)]
    private static partial Regex OrphanCloseRegex();

    /// <summary>
    /// Removes <c>&lt;think&gt;...&lt;/think&gt;</c> blocks from a prior assistant turn.
    /// Reasoning-model chat templates (Qwen3, DeepSeek-R1, ...) are trained assuming
    /// historical assistant turns contain no reasoning, so leaving them in bloats the
    /// context window and degrades quality.
    /// </summary>
    public static string ScrubAssistantThinking(string content)
    {
        if (string.IsNullOrEmpty(content)
            || (content.IndexOf("<think>", StringComparison.Ordinal) < 0
                && content.IndexOf("</think>", StringComparison.Ordinal) < 0))
            return content;

        // Greedy match handles nested/duplicated blocks in one pass.
        string result = ThinkBlockRegex().Replace(content, string.Empty);
        // Unclosed <think> with no matching </think>: drop the tag and everything after.
        result = UnclosedThinkRegex().Replace(result, string.Empty);
        // Orphan </think> with no preceding <think>: drop the stray tag.
        result = OrphanCloseRegex().Replace(result, string.Empty);
        return result;
    }

    /// <summary>
    /// Re-wraps a prior assistant turn's reasoning as a leading <c>&lt;think&gt;...&lt;/think&gt;</c>
    /// block — the mirror image of <see cref="ScrubAssistantThinking"/> — used when the request opts
    /// into <c>preserve_thinking</c>. Endpoints carry reasoning out-of-band from the model's answer
    /// text (OpenAI's <c>reasoning_content</c> field, Anthropic's <c>thinking</c> content block), so
    /// it has to be re-inlined here before the chat template sees a single content string. No-op when
    /// there's no reasoning to inject, or <paramref name="content"/> already has its own
    /// <c>&lt;think&gt;</c> tag (a client echoing the model's literal output rather than the separate
    /// reasoning channel — nothing to add).
    /// </summary>
    public static string InjectThinking(string content, string? reasoning)
    {
        if (string.IsNullOrEmpty(reasoning) || content.Contains("<think>", StringComparison.Ordinal))
            return content;
        return $"<think>\n{reasoning}\n</think>\n\n{content}";
    }
}

/// <summary>
/// Renders chat messages into a single prompt string for the loaded model. Registered as a
/// DI singleton; the <see cref="ServiceCollectionExtensions.AddSharpInference"/> wiring
/// reconfigures the renderer with the model's GGUF Jinja template once the engine has
/// finished loading. Tests that fake out <see cref="Engine.IInferenceEngine"/> keep the
/// constructor defaults (no Jinja template) and fall through to the hardcoded
/// architecture-based fallbacks below — sufficient for every supported chat format.
/// </summary>
public sealed class ChatTemplateRenderer
{
    private JinjaChatTemplate? _template;
    private string _architecture;
    private IToolCallAdapter _toolCallAdapter;

    /// <summary>Architecture string used to pick a fallback template when no Jinja is loaded.</summary>
    public string Architecture => _architecture;

    /// <summary>
    /// True when the loaded model's family should run with reasoning <em>off</em> unless a
    /// request explicitly opts in. Gemma 4 is not trained for the engine's <c>&lt;|channel&gt;</c>
    /// thought split the way Qwen3 is — enabling it makes the model ramble in the thought channel
    /// (and go badly out-of-distribution on multimodal input, producing garbage). Mirrors the
    /// CLI's <c>modelDefaultsThinkingOff = s_arch == "gemma4"</c> gate so the server and CLI agree.
    /// </summary>
    public bool ModelDefaultsThinkingOff => _architecture == "gemma4";

    /// <summary>Compiled Jinja template, if the loaded model shipped one.</summary>
    public JinjaChatTemplate? JinjaTemplate => _template;

    /// <summary>
    /// Tool-call translation layer matched to the loaded model's architecture. Resolved
    /// once at <see cref="Configure"/> time from <see cref="ToolCallAdapterRegistry"/>;
    /// endpoint code reads this without having to look up by string each request.
    /// </summary>
    public IToolCallAdapter ToolCallAdapter => _toolCallAdapter;

    /// <summary>
    /// Token IDs that mark the end of the loaded model's tool-call section (Gemma 4:
    /// <c>&lt;|tool_response&gt;</c>), resolved at <see cref="Configure"/> time from the adapter's
    /// <see cref="IToolCallAdapter.ToolBoundaryStopMarkers"/> against the model vocab. The chat
    /// endpoints union these into the stop set on tool-active requests so generation halts when the
    /// tool calls finish instead of running on (issue #304). Empty for families that need none.
    /// </summary>
    public IReadOnlyList<int> ToolBoundaryStopTokenIds { get; private set; } = [];

    /// <summary>
    /// Vocabulary view for grammar-constrained tool-call decoding (issue #374), set at
    /// <see cref="Configure"/> time. Null until the model is loaded, or when a custom engine factory
    /// supplied no vocabulary — <see cref="BuildToolArgumentConstraint"/> then returns null.
    /// </summary>
    private GrammarVocabulary? _grammarVocabulary;

    /// <summary>
    /// Public view of <see cref="_grammarVocabulary"/> (issue #423), so a host's
    /// <see cref="SharpInferenceServerOptions.OutputConstraintFactory"/> can build a
    /// vocabulary-aware whole-turn constraint the same way <see cref="BuildToolArgumentConstraint"/>
    /// builds tool-argument constraints. Null until the model is loaded, or when a custom engine
    /// factory supplied no vocabulary.
    /// </summary>
    public GrammarVocabulary? Grammar => _grammarVocabulary;

    /// <param name="architecture">Default architecture (used both for fallback and exposed via <see cref="Architecture"/>).</param>
    /// <param name="template">Optional compiled Jinja template; null means "use the hardcoded fallback".</param>
    public ChatTemplateRenderer(string architecture = "qwen2", JinjaChatTemplate? template = null)
    {
        _architecture = architecture;
        _template = template;
        _toolCallAdapter = ToolCallAdapterRegistry.Get(architecture);
    }

    /// <summary>
    /// Builds an argument-grammar constraint (issue #374) for the active tools using the loaded
    /// model's adapter + vocabulary, or null when the family has no constraint support, no
    /// vocabulary is available, or no supplied tool is constrainable. Endpoints attach the result to
    /// <see cref="Engine.SamplingParams.Constraint"/> on a tool-active request when tool-grammar is
    /// enabled.
    /// </summary>
    public ITokenConstraint? BuildToolArgumentConstraint(IReadOnlyList<ToolSchema> tools)
    {
        if (_grammarVocabulary is null || tools.Count == 0) return null;
        return _toolCallAdapter.BuildArgumentConstraint(tools, _grammarVocabulary);
    }

    /// <summary>
    /// Reconfigures the renderer with model-specific metadata. Called by the built-in
    /// engine loader once the GGUF file has been opened. Safe to call once; subsequent
    /// calls overwrite previous values (used by hot-reload scenarios).
    /// <paramref name="toolBoundaryStopTokenIds"/> are the vocab-resolved tool-boundary stops
    /// (see <see cref="ToolBoundaryStopTokenIds"/>); null leaves the set empty.
    /// </summary>
    public void Configure(string architecture, JinjaChatTemplate? template,
        IReadOnlyList<int>? toolBoundaryStopTokenIds = null,
        GrammarVocabulary? grammarVocabulary = null)
    {
        _architecture = architecture;
        _template = template;
        _toolCallAdapter = ToolCallAdapterRegistry.Get(architecture);
        ToolBoundaryStopTokenIds = toolBoundaryStopTokenIds ?? [];
        _grammarVocabulary = grammarVocabulary;
    }

    /// <param name="messages">Messages in order (system, user, assistant, ...).</param>
    /// <param name="enableThinking">
    /// When false, sets <c>enable_thinking=false</c> in the Jinja context so reasoning-capable
    /// templates (Qwen3, SmolLM3, ...) skip emitting a <c>&lt;think&gt;</c> block. Ignored by the
    /// hardcoded fallback paths since those archs have no thinking mode.
    /// </param>
    /// <param name="addGenerationPrompt">
    /// When false, omits the trailing assistant-generation prefix (e.g. Qwen3.x
    /// <c>&lt;|im_start|&gt;assistant\n&lt;think&gt;\n\n&lt;/think&gt;\n\n</c>). Used by the
    /// chat-completion endpoints to render a history-only view of the same messages as
    /// they would appear in a future turn's chat history — issue #102: the canonical
    /// rendering is what the engine snapshots so the next turn can reuse KV state.
    /// </param>
    public string Format(
        IReadOnlyList<(string role, string content)> messages,
        bool enableThinking = true,
        bool addGenerationPrompt = true)
    {
        if (_template != null)
        {
            var msgList = messages
                .Select(m => (Dictionary<string, object?>)new Dictionary<string, object?>
                {
                    ["role"]    = (object?)m.role,
                    ["content"] = (object?)m.content,
                })
                .Cast<object?>()
                .ToList();

            return _template.Render(new Dictionary<string, object?>
            {
                ["messages"]              = msgList,
                ["add_generation_prompt"] = (object?)addGenerationPrompt,
                ["tools"]                 = null,
                ["enable_thinking"]       = (object?)enableThinking,
            });
        }

        return RenderFallback(messages, _architecture, addGenerationPrompt);
    }

    /// <summary>
    /// Formats a list of pre-built message dictionaries into a prompt string.
    /// Used for tool-calling paths where messages may include <c>tool_calls</c> entries
    /// or role="tool" messages. Passes <paramref name="tools"/> to the Jinja context so
    /// the model's chat template can inject the tool schema into the system prompt.
    /// Falls back to extracting plain text when no Jinja template is available.
    /// </summary>
    public string Format(
        IReadOnlyList<Dictionary<string, object?>> messages,
        bool enableThinking = true,
        object? tools = null,
        bool addGenerationPrompt = true)
    {
        if (_template != null)
        {
            var msgList = messages.Cast<object?>().ToList();
            return _template.Render(new Dictionary<string, object?>
            {
                ["messages"]              = msgList,
                ["add_generation_prompt"] = (object?)addGenerationPrompt,
                ["tools"]                 = tools,
                ["enable_thinking"]       = (object?)enableThinking,
            });
        }

        var simple = messages
            .Select(m => (
                role:    m.TryGetValue("role",    out var r) ? (r as string ?? "") : "",
                content: m.TryGetValue("content", out var c) ? (c as string ?? "") : ""
            ))
            .ToList();
        return RenderFallback(simple, _architecture, addGenerationPrompt);
    }

    private static string RenderFallback(
        IReadOnlyList<(string role, string content)> messages,
        string arch,
        bool addGenerationPrompt = true)
    {
        var sb = new StringBuilder();

        if (arch is "llama4")
        {
            sb.Append("<|begin_of_text|>");
            foreach (var (role, content) in messages)
                sb.Append($"<|header_start|>{role}<|header_end|>\n\n{content}<|eot_id|>");
            if (addGenerationPrompt)
                sb.Append("<|header_start|>assistant<|header_end|>\n\n");
        }
        else if (arch is "llama")
        {
            sb.Append("<|begin_of_text|>");
            foreach (var (role, content) in messages)
                sb.Append($"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>");
            if (addGenerationPrompt)
                sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        else
        {
            // ChatML: Qwen, SmolLM2, default
            foreach (var (role, content) in messages)
                sb.Append($"<|im_start|>{role}\n{content}<|im_end|>\n");
            if (addGenerationPrompt)
                sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
