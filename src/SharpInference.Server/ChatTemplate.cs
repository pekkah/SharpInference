using System.Text;
using System.Text.RegularExpressions;
using SharpInference.Core;

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

    /// <summary>Architecture string used to pick a fallback template when no Jinja is loaded.</summary>
    public string Architecture => _architecture;

    /// <summary>Compiled Jinja template, if the loaded model shipped one.</summary>
    public JinjaChatTemplate? JinjaTemplate => _template;

    /// <param name="architecture">Default architecture (used both for fallback and exposed via <see cref="Architecture"/>).</param>
    /// <param name="template">Optional compiled Jinja template; null means "use the hardcoded fallback".</param>
    public ChatTemplateRenderer(string architecture = "qwen2", JinjaChatTemplate? template = null)
    {
        _architecture = architecture;
        _template = template;
    }

    /// <summary>
    /// Reconfigures the renderer with model-specific metadata. Called by the built-in
    /// engine loader once the GGUF file has been opened. Safe to call once; subsequent
    /// calls overwrite previous values (used by hot-reload scenarios).
    /// </summary>
    public void Configure(string architecture, JinjaChatTemplate? template)
    {
        _architecture = architecture;
        _template = template;
    }

    /// <param name="messages">Messages in order (system, user, assistant, ...).</param>
    /// <param name="enableThinking">
    /// When false, sets <c>enable_thinking=false</c> in the Jinja context so reasoning-capable
    /// templates (Qwen3, SmolLM3, ...) skip emitting a <c>&lt;think&gt;</c> block. Ignored by the
    /// hardcoded fallback paths since those archs have no thinking mode.
    /// </param>
    public string Format(
        IReadOnlyList<(string role, string content)> messages,
        bool enableThinking = true)
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
                ["add_generation_prompt"] = true,
                ["tools"]                 = null,
                ["enable_thinking"]       = (object?)enableThinking,
            });
        }

        return RenderFallback(messages, _architecture);
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
        object? tools = null)
    {
        if (_template != null)
        {
            var msgList = messages.Cast<object?>().ToList();
            return _template.Render(new Dictionary<string, object?>
            {
                ["messages"]              = msgList,
                ["add_generation_prompt"] = true,
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
        return RenderFallback(simple, _architecture);
    }

    private static string RenderFallback(IReadOnlyList<(string role, string content)> messages, string arch)
    {
        var sb = new StringBuilder();

        if (arch is "llama4")
        {
            sb.Append("<|begin_of_text|>");
            foreach (var (role, content) in messages)
                sb.Append($"<|header_start|>{role}<|header_end|>\n\n{content}<|eot_id|>");
            sb.Append("<|header_start|>assistant<|header_end|>\n\n");
        }
        else if (arch is "llama")
        {
            sb.Append("<|begin_of_text|>");
            foreach (var (role, content) in messages)
                sb.Append($"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>");
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        else
        {
            // ChatML: Qwen, SmolLM2, default
            foreach (var (role, content) in messages)
                sb.Append($"<|im_start|>{role}\n{content}<|im_end|>\n");
            sb.Append("<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
