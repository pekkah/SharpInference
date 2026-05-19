using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

public static class AnthropicEndpoints
{
    public static IEndpointRouteBuilder MapAnthropicEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/messages", HandleMessages);
        return app;
    }

    private static async Task HandleMessages(
        HttpContext ctx,
        IInferenceEngine engine)
    {
        AnthropicMessageRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(AppJsonContext.Default.AnthropicMessageRequest, ctx.RequestAborted);
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
                JsonSerializer.Serialize(new AErrorResponse("invalid_request_error", "messages array is required"),
                    AppJsonContext.Default.AErrorResponse), ctx.RequestAborted);
            return;
        }

        HealthEndpoints.RecordRequest();

        var modelArch = Environment.GetEnvironmentVariable("SHARPI_ARCH") ?? "qwen2";
        // Anthropic-style thinking control: {"type":"disabled"} turns it off; absence or any
        // other value (including {"type":"enabled"}) leaves it on. BudgetTokens is ignored
        // in this batch — it will map to max_thinking_tokens later.
        bool enableThinking = req.Thinking?.Type != "disabled";
        var messages = BuildMessageList(req);
        var prompt = ChatTemplate.Format(messages, modelArch, enableThinking);

        var sp = new SamplingParams
        {
            Temperature = req.Temperature ?? 1.0f,
            TopP = req.TopP ?? 1.0f,
            MaxNewTokens = req.MaxTokens,
        };

        var msgId = $"msg_{Guid.NewGuid():N}";
        var modelId = engine.ModelId;

        if (req.Stream == true)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers.Connection = "keep-alive";

            // message_start
            var startMsg = new AMessageStartEvent("message_start",
                new AMessageStartInner(msgId, "message", "assistant", modelId, "max_tokens", new AUsage(0, 0)));
            await WriteAnthropicEvent(ctx.Response, "message_start",
                JsonSerializer.Serialize(startMsg, AppJsonContext.Default.AMessageStartEvent));

            // content_block_start
            var blockStart = new AContentBlockStartEvent("content_block_start", 0,
                new AContentBlock("text", ""));
            await WriteAnthropicEvent(ctx.Response, "content_block_start",
                JsonSerializer.Serialize(blockStart, AppJsonContext.Default.AContentBlockStartEvent));

            // content_block_delta events
            int outputTokens = 0;
            await foreach (var token in engine.GenerateAsync(prompt, sp, ctx.RequestAborted))
            {
                outputTokens++;
                var delta = new AContentBlockDeltaEvent("content_block_delta", 0,
                    new ATextDelta("text_delta", token));
                await WriteAnthropicEvent(ctx.Response, "content_block_delta",
                    JsonSerializer.Serialize(delta, AppJsonContext.Default.AContentBlockDeltaEvent));
            }

            // content_block_stop
            await WriteAnthropicEvent(ctx.Response, "content_block_stop",
                JsonSerializer.Serialize(new ATypeOnly("content_block_stop"), AppJsonContext.Default.ATypeOnly));

            // message_delta
            var msgDelta = new AMessageDeltaEvent("message_delta",
                new AMessageDelta("end_turn", null), new AUsage(0, outputTokens));
            await WriteAnthropicEvent(ctx.Response, "message_delta",
                JsonSerializer.Serialize(msgDelta, AppJsonContext.Default.AMessageDeltaEvent));

            // message_stop
            await WriteAnthropicEvent(ctx.Response, "message_stop",
                JsonSerializer.Serialize(new ATypeOnly("message_stop"), AppJsonContext.Default.ATypeOnly));

            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            HealthEndpoints.RecordTokens(outputTokens);
        }
        else
        {
            var sb = new StringBuilder();
            int nonStreamTokens = 0;
            await foreach (var token in engine.GenerateAsync(prompt, sp, ctx.RequestAborted))
            {
                nonStreamTokens++;
                sb.Append(token);
            }

            HealthEndpoints.RecordTokens(nonStreamTokens);

            var response = new AnthropicMessageResponse(
                msgId, "message", "assistant",
                [new AContent("text", sb.ToString())],
                modelId, "end_turn",
                new AUsage(0, nonStreamTokens));

            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(response, AppJsonContext.Default.AnthropicMessageResponse),
                ctx.RequestAborted);
        }
    }

    private static List<(string role, string content)> BuildMessageList(AnthropicMessageRequest req)
    {
        var list = new List<(string, string)>();
        if (req.System is { Length: > 0 })
            list.Add(("system", req.System));
        foreach (var m in req.Messages!)
        {
            var role = m.Role ?? "user";
            var content = m.Content ?? "";
            if (role == "assistant")
                content = ChatTemplate.ScrubAssistantThinking(content);
            list.Add((role, content));
        }
        return list;
    }

    private static async Task WriteAnthropicEvent(HttpResponse response, string eventType, string data)
    {
        await response.WriteAsync($"event: {eventType}\ndata: {data}\n\n", response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record AnthropicMessageRequest(
    string? Model,
    AnthropicMessage[]? Messages,
    int MaxTokens,
    string? System,
    bool? Stream,
    float? Temperature,
    float? TopP,
    int? TopK,
    AnthropicThinking? Thinking = null);

public sealed record AnthropicThinking(string? Type, int? BudgetTokens);

public sealed record AnthropicMessage(string? Role, string? Content);

public sealed record AnthropicMessageResponse(
    string Id,
    string Type,
    string Role,
    AContent[] Content,
    string Model,
    string StopReason,
    AUsage Usage);

public sealed record AContent(string Type, string Text);
public sealed record AUsage(int InputTokens, int OutputTokens);

// Streaming event types
public sealed record AMessageStartEvent(string Type, AMessageStartInner Message);
public sealed record AMessageStartInner(string Id, string Type, string Role, string Model, string StopReason, AUsage Usage);
public sealed record AContentBlockStartEvent(string Type, int Index, AContentBlock ContentBlock);
public sealed record AContentBlock(string Type, string Text);
public sealed record AContentBlockDeltaEvent(string Type, int Index, ATextDelta Delta);
public sealed record ATextDelta(string Type, string Text);
public sealed record AMessageDeltaEvent(string Type, AMessageDelta Delta, AUsage Usage);
public sealed record AMessageDelta(string StopReason, string? StopSequence);
public sealed record ATypeOnly(string Type);
public sealed record AErrorResponse(string Type, string Message);
