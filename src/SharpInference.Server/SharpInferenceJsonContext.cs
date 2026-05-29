using System.Text.Json.Serialization;
using SharpInference.Server.Endpoints;

namespace SharpInference.Server;

/// <summary>
/// Source-generated <see cref="System.Text.Json.Serialization.JsonSerializerContext"/> covering
/// every request/response shape served by the SharpInference HTTP endpoints. Registered with
/// ASP.NET Core's JSON pipeline by <see cref="ServiceCollectionExtensions.AddSharpInference"/>,
/// so consumers don't usually need to reference it directly. Made <c>public</c> so AOT-published
/// hosts can chain it onto their own <c>JsonOptions</c> if they add SharpInference's endpoints
/// alongside their own routes.
/// </summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(OaiMessage[]))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(CompletionChoice[]))]
[JsonSerializable(typeof(OaiAssistantMessage))]
[JsonSerializable(typeof(ChatUsage))]
[JsonSerializable(typeof(CompletionTokensDetails))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(ChunkChoice[]))]
[JsonSerializable(typeof(ChunkDelta))]
[JsonSerializable(typeof(ModelsResponse))]
[JsonSerializable(typeof(ModelInfo[]))]
[JsonSerializable(typeof(ErrorResponse))]
[JsonSerializable(typeof(ResponseFormat))]
[JsonSerializable(typeof(AnthropicMessageRequest))]
[JsonSerializable(typeof(AnthropicThinking))]
[JsonSerializable(typeof(AnthropicTool))]
[JsonSerializable(typeof(AnthropicTool[]))]
[JsonSerializable(typeof(AnthropicMessage[]))]
[JsonSerializable(typeof(AnthropicMessageResponse))]
[JsonSerializable(typeof(AContent[]))]
[JsonSerializable(typeof(AUsage))]
[JsonSerializable(typeof(AMessageStartEvent))]
[JsonSerializable(typeof(AContentBlockStartEvent))]
[JsonSerializable(typeof(AContentBlockDeltaEvent))]
[JsonSerializable(typeof(AMessageDeltaEvent))]
[JsonSerializable(typeof(AContentBlockStopEvent))]
[JsonSerializable(typeof(ATypeOnly))]
[JsonSerializable(typeof(AErrorResponse))]
[JsonSerializable(typeof(HealthStatus))]
[JsonSerializable(typeof(ResponsesRequest))]
[JsonSerializable(typeof(RespObject))]
[JsonSerializable(typeof(RespOutputItem))]
[JsonSerializable(typeof(RespOutputItem[]))]
[JsonSerializable(typeof(RespContentPart))]
[JsonSerializable(typeof(RespContentPart[]))]
[JsonSerializable(typeof(RespUsage))]
[JsonSerializable(typeof(RespCreatedEvent))]
[JsonSerializable(typeof(RespOutputItemAddedEvent))]
[JsonSerializable(typeof(RespContentPartAddedEvent))]
[JsonSerializable(typeof(RespOutputTextDeltaEvent))]
[JsonSerializable(typeof(RespOutputTextDoneEvent))]
[JsonSerializable(typeof(RespOutputItemDoneEvent))]
[JsonSerializable(typeof(RespCompletedEvent))]
[JsonSerializable(typeof(Dictionary<string, float>))]
public partial class SharpInferenceJsonContext : JsonSerializerContext { }
