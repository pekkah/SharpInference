using Microsoft.AspNetCore.Mvc;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

/// <summary>Anthropic Messages API-compatible /v1/messages endpoint.</summary>
public static class AnthropicEndpoints
{
    public static IEndpointRouteBuilder MapAnthropicEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/messages", HandleMessages);
        return app;
    }

    private static IResult HandleMessages(
        [FromBody] MessageRequest req,
        InferenceEngine engine,
        CancellationToken ct)
    {
        // TODO: apply prompt formatting, run engine, return MessageResponse or SSE stream
        throw new NotImplementedException();
    }
}

public sealed record MessageRequest(string Model, Message[] Messages, int MaxTokens = 1024);
public sealed record Message(string Role, string Content);
public sealed record MessageResponse(string Id, string Type, string Role, ContentBlock[] Content, string Model, string StopReason);
public sealed record ContentBlock(string Type, string Text);
