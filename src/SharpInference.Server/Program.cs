using SharpInference.Server.Endpoints;

var builder = WebApplication.CreateSlimBuilder(args);

builder.Services.Configure<Microsoft.AspNetCore.Http.Json.JsonOptions>(o =>
{
    o.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

builder.Services.AddSingleton<SharpInference.Engine.InferenceEngine>(sp =>
{
    // TODO: load model from config and wire up backend
    throw new NotImplementedException("Configure InferenceEngine in Program.cs");
});

var app = builder.Build();

app.MapOpenAiEndpoints();
app.MapAnthropicEndpoints();
app.MapHealthEndpoints();

app.Run();

[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.CompletionRequest))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.CompletionResponse))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.MessageRequest))]
[System.Text.Json.Serialization.JsonSerializable(typeof(SharpInference.Server.Endpoints.MessageResponse))]
internal partial class AppJsonContext : System.Text.Json.Serialization.JsonSerializerContext { }
