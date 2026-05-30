# SharpInference.Server

ASP.NET Core endpoints, options, and DI extensions that expose [SharpInference](https://www.nuget.org/packages/SharpInference) as a drop-in **OpenAI- and Anthropic-compatible HTTP API**. Bring your own host (Kestrel, IIS, YARP, …); this package only ships the routes, request/response shapes, and DI wiring.

For the bare inference library, use [`SharpInference`](https://www.nuget.org/packages/SharpInference). For the standalone CLI, use [`SharpInference.Cli`](https://www.nuget.org/packages/SharpInference.Cli).

## Install

```
dotnet add package SharpInference.Server
```

This transitively pulls in `SharpInference` (the bundled inference engine + CPU/Vulkan/CUDA backends). You must be on the `Microsoft.NET.Sdk.Web` SDK — the package's `Microsoft.AspNetCore.App` framework reference is propagated.

## Quick start

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSharpInference(opt =>
{
    opt.ModelPath = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    opt.GpuLayers = -1; // -1 = all layers on GPU; 0 = pure CPU
});

var app = builder.Build();
app.MapSharpInference();
app.Run();
```

Bind from configuration instead:

```csharp
// appsettings.json: { "SharpInference": { "ModelPath": "...", "GpuLayers": -1 } }
builder.Services.AddSharpInference(builder.Configuration);
```

## What you get

| Endpoint                       | Wire-compatible with     |
|--------------------------------|--------------------------|
| `POST /v1/chat/completions`    | OpenAI Chat Completions  |
| `POST /v1/completions`         | OpenAI Completions       |
| `POST /v1/messages`            | Anthropic Messages       |
| `POST /v1/responses`           | OpenAI Responses         |
| `GET  /v1/models`              | OpenAI Models            |
| `GET  /health`, `/metrics`     | Liveness + Prometheus    |

Streaming (SSE) is enabled for every chat/completion endpoint, and the JSON pipeline is wired through a source-generated `JsonSerializerContext` so the package is AOT-friendly even though the project itself is not AOT-published.

## Configuration

`SharpInferenceServerOptions` is the single options record (`Options` pattern, validated on first request):

```csharp
public sealed class SharpInferenceServerOptions
{
    public string  ModelPath      { get; set; } = "";
    public int     GpuLayers      { get; set; }       // -1 = all, 0 = CPU-only
    public int     MaxContext     { get; set; } = 4096;
    public string? Architecture   { get; set; }       // override GGUF detection
    public Func<IServiceProvider, LoadedEngine>? EngineFactory { get; set; } // tests
    // …
}
```

Override `EngineFactory` in tests to inject a fake `IInferenceEngine`; the rest of the DI graph (chat-template renderer, metrics, JSON context) stays intact.

## Links

- [Repository & docs](https://github.com/pekkah/SharpInference)
- [Design document](https://github.com/pekkah/SharpInference/blob/master/docs/SharpInference-Design.md)
- [Issues](https://github.com/pekkah/SharpInference/issues)

## License

MIT. Copyright (c) 2026 Pekka Heikura.
