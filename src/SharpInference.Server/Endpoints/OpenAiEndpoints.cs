using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http;
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
        string? arch = null)
    {
        ChatCompletionRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(AppJsonContext.Default.ChatCompletionRequest, ctx.RequestAborted);
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
                    AppJsonContext.Default.ErrorResponse), ctx.RequestAborted);
            return;
        }

        HealthEndpoints.RecordRequest();

        var modelArch = arch ?? Environment.GetEnvironmentVariable("SHARPI_ARCH") ?? "qwen2";
        bool enableThinking = req.EnableThinking ?? true;
        var messages = BuildMessageList(req.Messages, modelArch, req.ResponseFormat?.Type);
        var prompt = ChatTemplate.Format(messages, modelArch, enableThinking);

        // Parse logit_bias: OpenAI sends {"tokenId": biasValue} with string keys
        IReadOnlyDictionary<int, float>? logitBias = null;
        if (req.LogitBias is { Count: > 0 })
        {
            var d = new Dictionary<int, float>(req.LogitBias.Count);
            foreach (var (k, v) in req.LogitBias)
                if (int.TryParse(k, out int id)) d[id] = v;
            if (d.Count > 0) logitBias = d;
        }

        var sp = new SamplingParams
        {
            Temperature = req.Temperature ?? 1.0f,
            TopP = req.TopP ?? 1.0f,
            MaxNewTokens = req.MaxTokens ?? 512,
            LogitBias = logitBias,
        };

        var requestId = $"chatcmpl-{Guid.NewGuid():N}";
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (req.Stream == true)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers.Connection = "keep-alive";

            // First chunk: role delta
            var firstChunk = new ChatCompletionChunk(
                requestId, "chat.completion.chunk", created, engine.ModelId,
                [new ChunkChoice(0, new ChunkDelta("assistant", null), null)]);
            await WriteEvent(ctx.Response, JsonSerializer.Serialize(firstChunk, AppJsonContext.Default.ChatCompletionChunk));

            long tokenCount = 0;
            await foreach (var c in engine.GenerateChunksAsync(prompt, sp, ctx.RequestAborted))
            {
                tokenCount++;
                ChunkDelta delta = c.Kind == GenerateChunkKind.Thinking
                    ? new ChunkDelta(null, null, c.Text)
                    : new ChunkDelta(null, c.Text);
                var chunk = new ChatCompletionChunk(
                    requestId, "chat.completion.chunk", created, engine.ModelId,
                    [new ChunkChoice(0, delta, null)]);
                await WriteEvent(ctx.Response, JsonSerializer.Serialize(chunk, AppJsonContext.Default.ChatCompletionChunk));
            }

            // Final chunk with finish_reason
            var finalChunk = new ChatCompletionChunk(
                requestId, "chat.completion.chunk", created, engine.ModelId,
                [new ChunkChoice(0, new ChunkDelta(null, null), "stop")]);
            await WriteEvent(ctx.Response, JsonSerializer.Serialize(finalChunk, AppJsonContext.Default.ChatCompletionChunk));
            await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

            HealthEndpoints.RecordTokens(tokenCount);
        }
        else
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
            HealthEndpoints.RecordTokens(completionTokens);

            var message = new OaiAssistantMessage(
                "assistant",
                textSb.ToString(),
                reasoningSb.Length > 0 ? reasoningSb.ToString() : null);
            var usage = new ChatUsage(
                0, completionTokens, completionTokens,
                reasoningTokens > 0 ? new CompletionTokensDetails(reasoningTokens) : null);

            var response = new ChatCompletionResponse(
                requestId, "chat.completion", created, engine.ModelId,
                [new CompletionChoice(0, message, "stop")],
                usage);

            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(response, AppJsonContext.Default.ChatCompletionResponse),
                ctx.RequestAborted);
        }
    }

    private static Task HandleListModels(HttpContext ctx, IInferenceEngine engine)
    {
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var response = new ModelsResponse("list", [new ModelInfo(engine.ModelId, "model", created, "sharpi")]);
        ctx.Response.ContentType = "application/json";
        return ctx.Response.WriteAsync(
            JsonSerializer.Serialize(response, AppJsonContext.Default.ModelsResponse),
            ctx.RequestAborted);
    }

    private static List<(string role, string content)> BuildMessageList(
        OaiMessage[] messages, string arch, string? responseFormatType = null)
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
    [property: JsonPropertyName("enable_thinking")] bool? EnableThinking = null);

public sealed record OaiMessage(string? Role, string? Content);

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
    string Content,
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null);
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
    [property: JsonPropertyName("reasoning_content")] string? ReasoningContent = null);

public sealed record ModelsResponse(string Object, ModelInfo[] Data);
public sealed record ModelInfo(string Id, string Object, long Created, string OwnedBy);

public sealed record ResponseFormat(string? Type);

public sealed record ErrorResponse(string Type, string Message);
