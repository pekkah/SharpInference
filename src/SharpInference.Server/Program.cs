using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Server.Endpoints;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(o =>
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default));

// Load model from environment or appsettings, then register engine as singleton.
// For testing, tests may replace IInferenceEngine with a fake via WebApplicationFactory overrides.
builder.Services.AddSingleton<IInferenceEngine>(sp =>
{
    var modelPath =
        Environment.GetEnvironmentVariable("SHARPI_MODEL")
        ?? builder.Configuration["SharpInference:ModelPath"]
        ?? "model.gguf";

    if (!File.Exists(modelPath))
        throw new InvalidOperationException(
            $"Model file not found: '{modelPath}'. Set SHARPI_MODEL env var or SharpInference:ModelPath in appsettings.json.");

    var model = GgufModel.Open(modelPath);
    var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
    var tokenizer = GgufTokenizer.FromGgufModel(model);
    var arch = model.Metadata.TryGetValue("general.architecture", out var a) ? (string)a : "qwen2";

    // Publish arch so endpoints can apply the right chat template
    Environment.SetEnvironmentVariable("SHARPI_ARCH", arch);

    var cpuBackend = new CpuBackend();
    var fwd = new ForwardPass(model, cpuBackend, hp);
    var modelId = Path.GetFileNameWithoutExtension(modelPath);

    int maxBatch = 1;
    if (int.TryParse(Environment.GetEnvironmentVariable("SHARPI_MAX_BATCH"), out int mb) && mb > 1)
        maxBatch = mb;

    if (maxBatch > 1)
        return new ContinuousBatchingEngine(fwd, tokenizer, modelId, maxBatch);
    return new InferenceEngine(fwd, tokenizer, modelId, cpuBackend, model);
});

var app = builder.Build();

app.MapOpenAiEndpoints();
app.MapAnthropicEndpoints();
app.MapHealthEndpoints();

app.Run();

// Required for WebApplicationFactory<Program> in integration tests
public partial class Program { }

// ── Source-generated JSON context ─────────────────────────────────────────────
// NativeAOT requires all serialized types to be registered here.
[System.Text.Json.Serialization.JsonSourceGenerationOptions(
    PropertyNamingPolicy = System.Text.Json.Serialization.JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull)]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChatCompletionRequest))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.OaiMessage[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChatCompletionResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.CompletionChoice[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.OaiAssistantMessage))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChatUsage))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChatCompletionChunk))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChunkChoice[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ChunkDelta))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ModelsResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ModelInfo[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ErrorResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AnthropicMessageRequest))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AnthropicMessage[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AnthropicMessageResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AContent[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AUsage))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AMessageStartEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AContentBlockStartEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AContentBlockDeltaEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AMessageDeltaEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ATypeOnly))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.AErrorResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.HealthStatus))]
internal partial class AppJsonContext : System.Text.Json.Serialization.JsonSerializerContext { }
