namespace SharpInference.Server;

/// <summary>
/// Formats a list of chat messages into a single prompt string using the model's chat template.
/// Supported architectures: llama4, llama (Llama 3.x), qwen2/smollm (ChatML default).
/// </summary>
public static class ChatTemplate
{
    /// <param name="messages">Messages in order (system, user, assistant, ...).</param>
    /// <param name="arch">Architecture string from GGUF metadata (e.g. "llama4", "llama", "qwen2").</param>
    public static string Format(IReadOnlyList<(string role, string content)> messages, string arch)
    {
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
}
