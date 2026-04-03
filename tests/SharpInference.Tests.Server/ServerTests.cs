using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Engine;
using SharpInference.Server.Endpoints;

namespace SharpInference.Tests.Server;

/// <summary>
/// Integration tests for the API server endpoints.
/// Uses a fake IInferenceEngine that does not require a real model file.
/// </summary>
public sealed class ServerTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ServerTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
            {
                // Replace the real InferenceEngine with a controllable fake
                services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
            }))
            .CreateClient();
    }

    // ── Health ────────────────────────────────────────────────────────────────

    [Fact]
    public async Task Health_ReturnsOk()
    {
        var response = await _client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("ok", json);
        Assert.Contains("test-model", json);
    }

    [Fact]
    public async Task Metrics_ReturnsPrometheusText()
    {
        var response = await _client.GetAsync("/metrics");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("sharpi_requests_total", body);
        Assert.Contains("sharpi_tokens_generated_total", body);
        Assert.Contains("sharpi_uptime_seconds", body);
    }

    // ── OpenAI models ─────────────────────────────────────────────────────────

    [Fact]
    public async Task ListModels_ReturnsList()
    {
        var response = await _client.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("test-model", json);
        Assert.Contains("list", json);
    }

    // ── OpenAI chat completions ───────────────────────────────────────────────

    [Fact]
    public async Task ChatCompletion_NonStreaming_ReturnsCompletion()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 10,
            stream = false
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("chat.completion", json);
        Assert.Contains("stop", json);
        // FakeEngine emits "Hello world" -> should appear in content
        Assert.Contains("Hello", json);
    }

    [Fact]
    public async Task ChatCompletion_MissingMessages_Returns400()
    {
        var req = new { model = "test-model" };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task ChatCompletion_Streaming_ReturnsSseEvents()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hello" } },
            max_tokens = 10,
            stream = true
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("data:", body);
        Assert.Contains("[DONE]", body);
        Assert.Contains("chat.completion.chunk", body);
    }

    // ── Anthropic messages ────────────────────────────────────────────────────

    [Fact]
    public async Task AnthropicMessages_NonStreaming_ReturnsMessage()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false
        };
        var response = await _client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("message", json);
        Assert.Contains("end_turn", json);
        Assert.Contains("Hello", json);
    }

    [Fact]
    public async Task AnthropicMessages_Streaming_ReturnsSseEvents()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = true
        };
        var response = await _client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("event: message_start", body);
        Assert.Contains("event: content_block_start", body);
        Assert.Contains("event: content_block_delta", body);
        Assert.Contains("event: message_stop", body);
    }
}

/// <summary>
/// Fake inference engine for integration tests. Emits "Hello world" as individual word tokens.
/// </summary>
internal sealed class FakeInferenceEngine : IInferenceEngine
{
    public string ModelId { get; }

    public FakeInferenceEngine(string modelId) => ModelId = modelId;

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var tokens = new[] { "Hello", " world", "!" };
        foreach (var token in tokens)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return token;
        }
    }
}
