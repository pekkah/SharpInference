using System.Text.RegularExpressions;
using SharpInference.Core;

namespace SharpInference.Server;

/// <summary>
/// Formats a list of chat messages into a single prompt string using the model's chat template.
/// When <see cref="Template"/> is set, Jinja2 rendering is used.
/// Falls back to hardcoded arch-based formats for models without tokenizer.chat_template.
/// </summary>
public static partial class ChatTemplate
{
    /// <summary>
    /// Set once at startup from <c>GgufTokenizer.ChatTemplate</c> if the model
    /// has a <c>tokenizer.chat_template</c> key in its GGUF metadata.
    /// </summary>
    public static JinjaChatTemplate? Template { get; set; }

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

    /// <param name="messages">Messages in order (system, user, assistant, ...).</param>
    /// <param name="arch">Architecture string from GGUF metadata (e.g. "llama4", "llama", "qwen2").</param>
    /// <param name="enableThinking">
    /// When false, sets <c>enable_thinking=false</c> in the Jinja context so reasoning-capable
    /// templates (Qwen3, SmolLM3, ...) skip emitting a <c>&lt;think&gt;</c> block. Ignored by the
    /// hardcoded fallback paths since those archs have no thinking mode.
    /// </param>
    public static string Format(
        IReadOnlyList<(string role, string content)> messages,
        string arch,
        bool enableThinking = true)
    {
        if (Template != null)
        {
            var msgList = messages
                .Select(m => (Dictionary<string, object?>)new Dictionary<string, object?>
                {
                    ["role"]    = (object?)m.role,
                    ["content"] = (object?)m.content,
                })
                .Cast<object?>()
                .ToList();

            return Template.Render(new Dictionary<string, object?>
            {
                ["messages"]              = msgList,
                ["add_generation_prompt"] = true,
                ["tools"]                 = null,
                ["enable_thinking"]       = (object?)enableThinking,
            });
        }

        // Fallback: hardcoded templates for known architectures.
        var sb = new System.Text.StringBuilder();

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

    /// <summary>
    /// Formats a list of pre-built message dictionaries into a prompt string.
    /// Used for tool-calling paths where messages may include <c>tool_calls</c> entries
    /// or role="tool" messages. Passes <paramref name="tools"/> to the Jinja context so
    /// the model's chat template can inject the tool schema into the system prompt.
    /// Falls back to extracting plain text when no Jinja template is available.
    /// </summary>
    public static string Format(
        IReadOnlyList<Dictionary<string, object?>> messages,
        string arch,
        bool enableThinking = true,
        object? tools = null)
    {
        if (Template != null)
        {
            var msgList = messages.Cast<object?>().ToList();
            return Template.Render(new Dictionary<string, object?>
            {
                ["messages"]              = msgList,
                ["add_generation_prompt"] = true,
                ["tools"]                 = tools,
                ["enable_thinking"]       = (object?)enableThinking,
            });
        }

        // Fallback: extract basic role/content pairs and use hardcoded arch templates.
        var simple = messages
            .Select(m => (
                role:    m.TryGetValue("role",    out var r) ? (r as string ?? "") : "",
                content: m.TryGetValue("content", out var c) ? (c as string ?? "") : ""
            ))
            .ToList();
        return Format(simple, arch, enableThinking);
    }

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
