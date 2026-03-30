using Microsoft.AspNetCore.Mvc;
using SharpInference.Engine;

namespace SharpInference.Server.Endpoints;

/// <summary>OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.</summary>
public static class OpenAiEndpoints
{
    public static IEndpointRouteBuilder MapOpenAiEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/completions", HandleCompletion);
        app.MapPost("/v1/chat/completions", HandleChatCompletion);
        return app;
    }

    private static IResult HandleCompletion(
        [FromBody] CompletionRequest req,
        InferenceEngine engine,
        CancellationToken ct)
    {
        // TODO: tokenise prompt, run engine, stream back tokens
        throw new NotImplementedException();
    }

    private static IResult HandleChatCompletion(
        [FromBody] CompletionRequest req,
        InferenceEngine engine,
        CancellationToken ct)
    {
        // TODO: apply chat template, run engine, stream SSE
        throw new NotImplementedException();
    }
}

public sealed record CompletionRequest(string Model, string Prompt, int MaxTokens = 256, float Temperature = 1.0f, bool Stream = false);
public sealed record CompletionResponse(string Id, string Object, long Created, string Model, Choice[] Choices);
public sealed record Choice(int Index, string Text, string FinishReason);
