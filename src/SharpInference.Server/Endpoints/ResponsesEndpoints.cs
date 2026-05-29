using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.Options;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

/// <summary>
/// OpenAI Responses API: POST /v1/responses
/// Supports both streaming (SSE) and non-streaming responses.
/// Input can be a plain string or an array of message objects.
/// </summary>
public static class ResponsesEndpoints
{
    public static IEndpointRouteBuilder MapResponsesEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/responses", HandleCreateResponse);
        return app;
    }

    private static async Task HandleCreateResponse(
        HttpContext ctx,
        IInferenceEngine engine,
        ChatTemplateRenderer chatTemplate,
        ServerMetrics metrics,
        IOptions<SharpInferenceServerOptions> options)
    {
        var opts = options.Value;
        ResponsesRequest? req;
        try
        {
            req = await ctx.Request.ReadFromJsonAsync(SharpInferenceJsonContext.Default.ResponsesRequest, ctx.RequestAborted);
        }
        catch
        {
            ctx.Response.StatusCode = 400;
            return;
        }

        if (req is null)
        {
            ctx.Response.StatusCode = 400;
            return;
        }

        metrics.RecordRequest();

        var messages = BuildMessageList(req);
        var prompt = chatTemplate.Format(messages);

        var sp = SamplingParamsBuilder.Build(opts,
            temperature: req.Temperature,
            topP:        req.TopP,
            maxTokens:   req.MaxOutputTokens);

        var responseId = $"resp_{Guid.NewGuid():N}";
        var itemId = $"msg_{Guid.NewGuid():N}";
        long createdAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        string modelId = engine.ModelId;

        if (req.Stream == true)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers.Connection = "keep-alive";

            // response.created
            var createdEvent = new RespCreatedEvent("response.created",
                new RespObject(responseId, "response", createdAt, modelId, "in_progress",
                    [], new RespUsage(0, 0, 0)));
            await WriteEvent(ctx.Response, "response.created",
                JsonSerializer.Serialize(createdEvent, SharpInferenceJsonContext.Default.RespCreatedEvent));

            // response.output_item.added
            var itemAddedEvent = new RespOutputItemAddedEvent("response.output_item.added", 0,
                new RespOutputItem(itemId, "message", "assistant", []));
            await WriteEvent(ctx.Response, "response.output_item.added",
                JsonSerializer.Serialize(itemAddedEvent, SharpInferenceJsonContext.Default.RespOutputItemAddedEvent));

            // response.content_part.added
            var partAddedEvent = new RespContentPartAddedEvent("response.content_part.added",
                itemId, 0, 0, new RespContentPart("output_text", ""));
            await WriteEvent(ctx.Response, "response.content_part.added",
                JsonSerializer.Serialize(partAddedEvent, SharpInferenceJsonContext.Default.RespContentPartAddedEvent));

            var sb = new StringBuilder();
            long tokenCount = 0;
            await foreach (var token in engine.GenerateAsync(prompt, sp, ctx.RequestAborted))
            {
                tokenCount++;
                sb.Append(token);
                var deltaEvent = new RespOutputTextDeltaEvent("response.output_text.delta",
                    itemId, 0, 0, token);
                await WriteEvent(ctx.Response, "response.output_text.delta",
                    JsonSerializer.Serialize(deltaEvent, SharpInferenceJsonContext.Default.RespOutputTextDeltaEvent));
            }

            string fullText = sb.ToString();

            // response.output_text.done
            var textDoneEvent = new RespOutputTextDoneEvent("response.output_text.done",
                itemId, 0, 0, fullText);
            await WriteEvent(ctx.Response, "response.output_text.done",
                JsonSerializer.Serialize(textDoneEvent, SharpInferenceJsonContext.Default.RespOutputTextDoneEvent));

            // response.output_item.done
            var finalItem = new RespOutputItem(itemId, "message", "assistant",
                [new RespContentPart("output_text", fullText)]);
            var itemDoneEvent = new RespOutputItemDoneEvent("response.output_item.done", 0, finalItem);
            await WriteEvent(ctx.Response, "response.output_item.done",
                JsonSerializer.Serialize(itemDoneEvent, SharpInferenceJsonContext.Default.RespOutputItemDoneEvent));

            // response.completed
            var usage = new RespUsage(0, (int)tokenCount, (int)tokenCount);
            var completedResp = new RespObject(responseId, "response", createdAt, modelId, "completed",
                [finalItem], usage);
            var completedEvent = new RespCompletedEvent("response.completed", completedResp);
            await WriteEvent(ctx.Response, "response.completed",
                JsonSerializer.Serialize(completedEvent, SharpInferenceJsonContext.Default.RespCompletedEvent));

            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            metrics.RecordTokens(tokenCount);
        }
        else
        {
            var sb = new StringBuilder();
            long tokenCount = 0;
            await foreach (var token in engine.GenerateAsync(prompt, sp, ctx.RequestAborted))
            {
                tokenCount++;
                sb.Append(token);
            }

            string text = sb.ToString();
            var usage = new RespUsage(0, (int)tokenCount, (int)tokenCount);
            var item = new RespOutputItem(itemId, "message", "assistant",
                [new RespContentPart("output_text", text)]);
            var response = new RespObject(responseId, "response", createdAt, modelId, "completed",
                [item], usage);

            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(
                JsonSerializer.Serialize(response, SharpInferenceJsonContext.Default.RespObject),
                ctx.RequestAborted);

            metrics.RecordTokens(tokenCount);
        }
    }

    private static List<(string role, string content)> BuildMessageList(ResponsesRequest req)
    {
        var list = new List<(string, string)>();

        // instructions → system message
        if (!string.IsNullOrEmpty(req.Instructions))
            list.Add(("system", req.Instructions));

        if (req.Input.HasValue)
        {
            var input = req.Input.Value;
            if (input.ValueKind == JsonValueKind.String)
            {
                list.Add(("user", input.GetString() ?? ""));
            }
            else if (input.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in input.EnumerateArray())
                {
                    string role = "user";
                    string content = "";
                    if (item.TryGetProperty("role", out var roleProp))
                        role = roleProp.GetString() ?? "user";
                    if (item.TryGetProperty("content", out var contentProp))
                    {
                        if (contentProp.ValueKind == JsonValueKind.String)
                            content = contentProp.GetString() ?? "";
                        else if (contentProp.ValueKind == JsonValueKind.Array)
                        {
                            // content is array of {type, text} parts — concatenate text parts
                            var sb = new StringBuilder();
                            foreach (var part in contentProp.EnumerateArray())
                                if (part.TryGetProperty("text", out var textProp))
                                    sb.Append(textProp.GetString());
                            content = sb.ToString();
                        }
                    }
                    if (role == "assistant")
                        content = ChatTemplate.ScrubAssistantThinking(content);
                    list.Add((role, content));
                }
            }
        }

        if (list.Count == 0)
            list.Add(("user", ""));

        return list;
    }

    private static async Task WriteEvent(HttpResponse response, string eventType, string data)
    {
        await response.WriteAsync($"event: {eventType}\ndata: {data}\n\n",
            response.HttpContext.RequestAborted);
        await response.Body.FlushAsync(response.HttpContext.RequestAborted);
    }
}

// ── Request / Response types ──────────────────────────────────────────────────

public sealed record ResponsesRequest(
    string? Model,
    JsonElement? Input,
    string? Instructions,
    int? MaxOutputTokens,
    float? Temperature,
    float? TopP,
    bool? Stream);

public sealed record RespObject(
    string Id,
    string Object,
    long CreatedAt,
    string Model,
    string Status,
    RespOutputItem[] Output,
    RespUsage Usage);

public sealed record RespOutputItem(
    string Id,
    string Type,
    string Role,
    RespContentPart[] Content);

public sealed record RespContentPart(string Type, string Text);

public sealed record RespUsage(int InputTokens, int OutputTokens, int TotalTokens);

// Streaming event types

public sealed record RespCreatedEvent(string Type, RespObject Response);

public sealed record RespOutputItemAddedEvent(string Type, int OutputIndex, RespOutputItem Item);

public sealed record RespContentPartAddedEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, RespContentPart Part);

public sealed record RespOutputTextDeltaEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, string Delta);

public sealed record RespOutputTextDoneEvent(
    string Type, string ItemId, int OutputIndex, int ContentIndex, string Text);

public sealed record RespOutputItemDoneEvent(string Type, int OutputIndex, RespOutputItem Item);

public sealed record RespCompletedEvent(string Type, RespObject Response);
