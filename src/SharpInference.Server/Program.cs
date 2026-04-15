using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Server;
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

    // Resolve relative paths against common roots so `SHARPI_MODEL=models/foo.gguf`
    // works regardless of whether the process CWD is the repo root, the project
    // directory (as `dotnet run --project` sets it), or the published binary's dir.
    if (!Path.IsPathRooted(modelPath) && !File.Exists(modelPath))
    {
        var candidates = new List<string>
        {
            Path.Combine(Directory.GetCurrentDirectory(), modelPath),
            Path.Combine(AppContext.BaseDirectory, modelPath),
        };
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 5 && dir is not null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, modelPath));
        var resolved = candidates.FirstOrDefault(File.Exists);
        if (resolved is not null) modelPath = resolved;
    }

    if (!File.Exists(modelPath))
        throw new InvalidOperationException(
            $"Model file not found: '{modelPath}'. Set SHARPI_MODEL env var or SharpInference:ModelPath in appsettings.json.");

    var model = GgufModel.Open(modelPath);
    var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
    var tokenizer = GgufTokenizer.FromGgufModel(model);
    var arch = model.Metadata.TryGetValue("general.architecture", out var a) ? (string)a : "qwen2";

    // Publish arch so endpoints can apply the right chat template
    Environment.SetEnvironmentVariable("SHARPI_ARCH", arch);

    // Use the model's own Jinja2 chat template when available
    ChatTemplate.Template = tokenizer.ChatTemplate;

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
app.MapResponsesEndpoints();
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
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ResponseFormat))]
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
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.ResponsesRequest))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespObject))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputItem))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputItem[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespContentPart))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespContentPart[]))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespUsage))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespCreatedEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputItemAddedEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespContentPartAddedEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputTextDeltaEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputTextDoneEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespOutputItemDoneEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.RespCompletedEvent))]
[System.Text.Json.Serialization.JsonSerializable(typeof(System.Collections.Generic.Dictionary<string, float>))]
internal partial class AppJsonContext : System.Text.Json.Serialization.JsonSerializerContext { }
