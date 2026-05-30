using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Options;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class OpenAiEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/chat/completions", HandleChatCompletion);
        app.MapGet("/v1/models", HandleListModels);
        return app;
    }

    private static async Task HandleChatCompletion(
        HttpContext ctx,
        IInferenceEngine engine,
        ChatTemplateRenderer chatTemplate,
        ServerMetrics metrics,
        IOptions<SharpInferenceServerOptions> options)
    {
        var opts = options.Value;
        ChatCompletionRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(SharpInferenceJsonContext.Default.ChatCompletionRequest, ctx.RequestAborted);
        }
        catch
        {
            ctx.Response.StatusCode = 400;
            return;
        }

        if (req is null || req.Messages is null || req.Messages.Length == 0)
        {
            ctx.Response.StatusCode = 400;
            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(new ErrorResponse("invalid_request_error", "messages array is required"),
                    SharpInferenceJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        metrics.RecordRequest();

        bool enableThinking = req.EnableThinking ?? true;
        var adapter = chatTemplate.ToolCallAdapter;

        // Tool-aware rendering: if either tool definitions or a history-side
        // tool_call / tool message is present, route through the rich-message path
        // so the chat template can inject the tool schema and replay prior calls.
        bool toolsActive = req.Tools is { Length: > 0 } || HasToolMessages(req.Messages);
        string prompt;
        if (toolsActive)
        {
            var (richMessages, tools) = BuildRichMessageList(req, adapter);
            prompt = chatTemplate.Format(richMessages, enableThinking, tools);
        }
        else
        {
            var messages = BuildMessageList(req.Messages, req.ResponseFormat?.Type);
            prompt = chatTemplate.Format(messages, enableThinking);
        }

        // Parse logit_bias: OpenAI sends {"tokenId": biasValue} with string keys
        IReadOnlyDictionary<int, float>? logitBias = null;
        if (req.LogitBias is { Count: > 0 })
        {
            var d = new Dictionary<int, float>(req.LogitBias.Count);
            foreach (var (k, v) in req.LogitBias)
                if (int.TryParse(k, out int id)) d[id] = v;
            if (d.Count > 0) logitBias = d;
        }

        var sp = SamplingParamsBuilder.Build(opts,
            temperature: req.Temperature,
            topP:        req.TopP,
            maxTokens:   req.MaxTokens,
            maxThinking: req.MaxThinkingTokens,
            logitBias:   logitBias);

        var requestId = $"chatcmpl-{Guid.NewGuid():N}";
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (req.Stream == true)
        {
            await HandleStreaming(ctx, engine, metrics, adapter, toolsActive, prompt, sp, requestId, created);
        }
        else
        {
            await HandleNonStreaming(ctx, engine, metrics, adapter, toolsActive, prompt, sp, requestId, created);
        }
    }

    private static async Task HandleNonStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        bool toolsActive, string prompt, SamplingParams sp, string requestId, long created)
    {
        var textSb = new StringBuilder();
        var reasoningSb = new StringBuilder();
        int textTokens = 0;
        int reasoningTokens = 0;

        await foreach (var c in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
        {
            if (c.Kind == GenerateChunkKind.Thinking)
            {
                reasoningSb.Append(c.Text);
                reasoningTokens++;
            }
            else
            {
                textSb.Append(c.Text);
                textTokens++;
            }
        }

        int completionTokens = textTokens + reasoningTokens;
        metrics.RecordTokens(completionTokens);

        var rawText = textSb.ToString();
        IReadOnlyList<ParsedToolCall> parsedCalls;
        string plainText;
        if (toolsActive)
        {
            (plainText, parsedCalls) = adapter.Parse(rawText);
        }
        else
        {
            // No tools declared on this request → don't try to interpret the output as
            // structured calls. Surfaces raw text identically to the pre-tools behaviour.
            plainText = rawText;
            parsedCalls = [];
        }

        OaiToolCall[]? toolCalls = null;
        string finishReason = "stop";
        string? content = plainText;
        if (parsedCalls.Count > 0)
        {
            toolCalls = parsedCalls
                .Select(c => new OaiToolCall(
                    Id: $"call_{Guid.NewGuid():N}",
                    Type: "function",
                    Function: new OaiToolCallFunction(c.Name, JinjaChatTemplate.SerializeToJson(c.Arguments))))
                .ToArray();
            finishReason = "tool_calls";
            // OpenAI returns content: null when a tool_calls array is present and there
            // was no accompanying text; an empty string would be a wire-shape mismatch.
            if (plainText.Length == 0) content = null;
        }

        var message = new OaiAssistantMessage(
            "assistant",
            content,
            reasoningSb.Length > 0 ? reasoningSb.ToString() : null,
            toolCalls);
        var usage = new ChatUsage(
            0, completionTokens, completionTokens,
            reasoningTokens > 0 ? new CompletionTokensDetails(reasoningTokens) : null);

        var response = new ChatCompletionResponse(
            requestId, "chat.completion", created, engine.ModelId,
            [new CompletionChoice(0, message, finishReason)],
            usage);

        ctx.Response.ContentType = "application/json";
        await ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.ChatCompletionResponse),
            ctx.RequestAborted);
    }

    private static async Task HandleStreaming(
        HttpContext ctx, IInferenceEngine engine, ServerMetrics metrics, IToolCallAdapter adapter,
        bool toolsActive, string prompt, SamplingParams sp, string requestId, long created)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.Headers.Connection = "keep-alive";

        // First chunk: role delta
        var firstChunk = new ChatCompletionChunk(
            requestId, "chat.completion.chunk", created, engine.ModelId,
            [new ChunkChoice(0, new ChunkDelta("assistant", null), null)]);
        await WriteEvent(ctx.Response, JsonSerializer.Serialize(firstChunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));

        int maxOpenLen = adapter.MaxOpenTagLength;
        bool inToolCall = false;
        int toolCallContentStart = -1;
        var toolCallBuf = new StringBuilder();
        string pendingText = "";
        int toolCallIndex = 0;       // monotonic index into delta.tool_calls
        bool hasToolCalls = false;
        long tokenCount = 0;

        async Task WriteContentDelta(string text)
        {
            if (text.Length == 0) return;
            var chunk = new ChatCompletionChunk(
                requestId, "chat.completion.chunk", created, engine.ModelId,
                [new ChunkChoice(0, new ChunkDelta(null, text), null)]);
            await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
        }

        async Task EmitToolCallsFromBlock(string blockContent)
        {
            var calls = adapter.ParseBlock(blockContent);
            foreach (var tc in calls)
            {
                hasToolCalls = true;
                var delta = new OaiToolCallDelta(
                    Index: toolCallIndex,
                    Id: $"call_{Guid.NewGuid():N}",
                    Type: "function",
                    Function: new OaiToolCallFunction(tc.Name, JinjaChatTemplate.SerializeToJson(tc.Arguments)));
                var chunk = new ChatCompletionChunk(
                    requestId, "chat.completion.chunk", created, engine.ModelId,
                    [new ChunkChoice(0, new ChunkDelta(null, null, null, [delta]), null)]);
                await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
                toolCallIndex++;
            }
        }

        async Task ProcessTextChunk(string chunk)
        {
            if (inToolCall)
            {
                toolCallBuf.Append(chunk);
                string buf = toolCallBuf.ToString();
                int closeIdx = adapter.FindCloseMarker(buf, toolCallContentStart, out int afterClose);
                if (closeIdx >= 0)
                {
                    string block = buf[toolCallContentStart..closeIdx];
                    string remaining = buf[afterClose..];
                    toolCallBuf.Clear();
                    toolCallContentStart = -1;
                    inToolCall = false;
                    await EmitToolCallsFromBlock(block);
                    if (remaining.Length > 0)
                        await ProcessTextChunk(remaining);
                }
                return;
            }

            pendingText += chunk;
            int openIdx = adapter.FindOpenMarker(pendingText, 0, out int contentStart);
            if (openIdx >= 0)
            {
                if (openIdx > 0)
                    await WriteContentDelta(pendingText[..openIdx]);

                inToolCall = true;
                toolCallBuf.Clear();
                toolCallBuf.Append(pendingText, contentStart, pendingText.Length - contentStart);
                toolCallContentStart = 0;
                pendingText = "";

                if (toolCallBuf.Length > 0)
                    await ProcessTextChunk("");
                return;
            }

            int safeLen = Math.Max(0, pendingText.Length - (maxOpenLen - 1));
            if (safeLen > 0)
            {
                await WriteContentDelta(pendingText[..safeLen]);
                pendingText = pendingText[safeLen..];
            }
        }

        try
        {
            await foreach (var c in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
            {
                tokenCount++;
                if (c.Kind == GenerateChunkKind.Thinking)
                {
                    var delta = new ChunkDelta(null, null, c.Text);
                    var chunk = new ChatCompletionChunk(
                        requestId, "chat.completion.chunk", created, engine.ModelId,
                        [new ChunkChoice(0, delta, null)]);
                    await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
                }
                else if (toolsActive)
                {
                    await ProcessTextChunk(c.Text);
                }
                else
                {
                    // No tools declared → skip the buffering state machine and forward each
                    // chunk as a separate content delta (clients rely on the streaming cadence).
                    await WriteContentDelta(c.Text);
                }
            }
        }
        finally
        {
            // Flush any remaining buffered text (cannot contain a partial open marker now).
            if (!inToolCall && pendingText.Length > 0)
            {
                try { await WriteContentDelta(pendingText); } catch { /* response aborted */ }
                pendingText = "";
            }
        }

        // Final chunk with finish_reason
        var finishReason = hasToolCalls ? "tool_calls" : "stop";
        var finalChunk = new ChatCompletionChunk(
            requestId, "chat.completion.chunk", created, engine.ModelId,
            [new ChunkChoice(0, new ChunkDelta(null, null), finishReason)]);
        await WriteEvent(ctx.Response, JsonSerializer.Serialize(finalChunk, SharpInferenceJsonContext.Default.ChatCompletionChunk));
        await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

        metrics.RecordTokens(tokenCount);
    }

    private static Task HandleListModels(HttpContext ctx, IInferenceEngine engine)
    {
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var response = new ModelsResponse("list", [new ModelInfo(engine.ModelId, "model", created, "sharpi")]);
        ctx.Response.ContentType = "application/json";
        return ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.ModelsResponse),
            ctx.RequestAborted);
    }

    private static List<(string role, string content)> BuildMessageList(
        OaiMessage[] messages, string? responseFormatType = null)
    {
        var list = new List<(string, string)>(messages.Length + 1);
        if (responseFormatType == "json_object")
            list.Add(("system", "Respond with valid JSON only. Do not include any text outside the JSON object."));
        foreach (var m in messages)
        {
            var role = m.Role ?? "user";
            var content = m.Content ?? "";
            if (role == "assistant")
                content = ChatTemplate.ScrubAssistantThinking(content);
            list.Add((role, content));
        }
        return list;
    }

    private static bool HasToolMessages(OaiMessage[]? messages)
    {
        if (messages is null) return false;
        foreach (var m in messages)
            if (m.Role == "tool" || m.ToolCalls is { Length: > 0 })
                return true;
        return false;
    }

    /// <summary>
    /// Builds rich message dictionaries and converts OpenAI tool definitions to the
    /// <c>{type:"function", function:{...}}</c> shape the Jinja chat template expects.
    /// Mirrors <see cref="AnthropicEndpoints"/>' BuildRichMessageList but for the
    /// OpenAI wire shape: tool definitions live at top level, assistant tool calls
    /// arrive as a <c>tool_calls</c> array, and tool results arrive as a separate
    /// <c>role:"tool"</c> message with <c>tool_call_id</c>.
    /// </summary>
    private static (List<Dictionary<string, object?>> messages, List<object?>? tools)
        BuildRichMessageList(ChatCompletionRequest req, IToolCallAdapter adapter)
    {
        var messages = new List<Dictionary<string, object?>>();

        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";
            var content = m.Content ?? "";

            if (role == "tool")
            {
                messages.Add(adapter.RenderToolResult(m.ToolCallId ?? "", content));
                continue;
            }

            if (role == "assistant")
            {
                string textStr = ChatTemplate.ScrubAssistantThinking(content);
                var msg = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["role"]    = "assistant",
                    ["content"] = textStr,
                };
                if (m.ToolCalls is { Length: > 0 })
                {
                    var toolCalls = new List<object?>();
                    foreach (var tc in m.ToolCalls)
                    {
                        toolCalls.Add(new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["id"]   = tc.Id,
                            ["type"] = tc.Type,
                            ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                            {
                                ["name"]      = tc.Function.Name,
                                // OpenAI stringifies arguments; pass through as-is.
                                ["arguments"] = tc.Function.Arguments,
                            },
                        });
                    }
                    msg["tool_calls"] = (object?)toolCalls;
                }
                messages.Add(msg);
                continue;
            }

            // system / user / other roles
            messages.Add(new(StringComparer.Ordinal) { ["role"] = role, ["content"] = content });
        }

        List<object?>? toolsList = null;
        if (req.Tools is { Length: > 0 })
        {
            toolsList = req.Tools
                .Select(t => (object?)new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["type"] = t.Type,
                    ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["name"]        = t.Function.Name,
                        ["description"] = t.Function.Description,
                        ["parameters"]  = t.Function.Parameters,
                    },
                })
                .ToList();
        }

        return (messages, toolsList);
    }

    private static async Task WriteEvent(HttpResponse response, string data)
    {
        await response.WriteAsync($"data: {data}\n\n", response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record ChatCompletionRequest(
    string? Model,
    OaiMessage[]? Messages,
    int? MaxTokens,
    float? Temperature,
    float? TopP,
    bool? Stream,
    Dictionary<string, float>? LogitBias,
    ResponseFormat? ResponseFormat,
    [property: JsonPropertyName("enable_thinking")] bool? EnableThinking = null,
    [property: JsonPropertyName("max_thinking_tokens")] int? MaxThinkingTokens = null,
    OaiTool[]? Tools = null,
    [property: JsonPropertyName("tool_choice")] JsonElement? ToolChoice = null);

/// <summary>
/// Message in an OpenAI <c>/v1/chat/completions</c> request. Both single-string
/// <c>content</c> and structured fields are supported; an assistant message echoing
/// a prior tool call uses <see cref="ToolCalls"/> in place of (or alongside) text,
/// and a <c>role: "tool"</c> message carries <see cref="ToolCallId"/> + the result
/// text in <see cref="Content"/>.
/// </summary>
public sealed record OaiMessage(
    string? Role,
    string? Content,
    [property: JsonPropertyName("tool_call_id")] string? ToolCallId = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCall[]? ToolCalls = null,
    string? Name = null);

/// <summary>
/// OpenAI tool definition. <c>function.parameters</c> is the JSON Schema; we keep it
/// as a raw <see cref="JsonElement"/> so it round-trips through the Jinja template unchanged.
/// </summary>
public sealed record OaiTool(string Type, OaiToolFunction Function);

public sealed record OaiToolFunction(
    string Name,
    string? Description,
    JsonElement? Parameters);

/// <summary>
/// Single tool-call entry. Emitted on assistant messages (in responses) and accepted
/// on history-side assistant messages (in requests). OpenAI's spec uses a stringly
/// JSON-encoded <c>arguments</c> field.
/// </summary>
public sealed record OaiToolCall(
    string Id,
    string Type,
    OaiToolCallFunction Function);

public sealed record OaiToolCallFunction(string Name, string Arguments);

public sealed record ChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    CompletionChoice[] Choices,
    ChatUsage Usage);

public sealed record CompletionChoice(int Index, OaiAssistantMessage Message, string? FinishReason);
public sealed record OaiAssistantMessage(
    string Role,
    string? Content,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCall[]? ToolCalls = null);
public sealed record ChatUsage(
    int PromptTokens,
    int CompletionTokens,
    int TotalTokens,
    [property: JsonPropertyName("completion_tokens_details")] CompletionTokensDetails? CompletionTokensDetails = null);
public sealed record CompletionTokensDetails(
    [property: JsonPropertyName("reasoning_tokens")] int ReasoningTokens);

public sealed record ChatCompletionChunk(
    string Id,
    string Object,
    long Created,
    string Model,
    ChunkChoice[] Choices);

public sealed record ChunkChoice(int Index, ChunkDelta Delta, string? FinishReason);
public sealed record ChunkDelta(
    string? Role,
    string? Content,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null,
    [property: JsonPropertyName("tool_calls")] OaiToolCallDelta[]? ToolCalls = null);

/// <summary>
/// Per-chunk tool-call delta. OpenAI streams partial tool calls in array index order
/// — clients reconstruct each call by concatenating the <c>function.arguments</c> JSON
/// fragments across deltas sharing the same <see cref="Index"/>. We emit the full call
/// in a single delta (matching how the engine surfaces a complete <c>tool_use</c> block).
/// </summary>
public sealed record OaiToolCallDelta(
    int Index,
    string? Id,
    string? Type,
    OaiToolCallFunction? Function);

public sealed record ModelsResponse(string Object, ModelInfo[] Data);
public sealed record ModelInfo(string Id, string Object, long Created, string OwnedBy);

public sealed record ResponseFormat(string? Type);

public sealed record ErrorResponse(string Type, string Message);
