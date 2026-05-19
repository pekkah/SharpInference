using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Engine;
using SharpInference.Server;
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
        Assert.Contains("sharpi_tokens_per_second", body);
        Assert.Contains("sharpi_queue_depth", body);
        Assert.Contains("sharpi_active_requests", body);
    }

    [Fact]
    public async Task Metrics_RequestCountIncrements_AfterChatRequest()
    {
        // Make one request then verify the counter went up.
        // Note: this test uses its own factory to get an isolated counter state.
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("m"))));
        var client = factory.CreateClient();

        var before = await GetMetricValue(client, "sharpi_requests_total");

        var req = new
        {
            model = "m",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 5,
            stream = false
        };
        await client.PostAsJsonAsync("/v1/chat/completions", req);

        var after = await GetMetricValue(client, "sharpi_requests_total");
        Assert.True(after > before, $"Expected counter to increment, got before={before} after={after}");
    }

    [Fact]
    public async Task Metrics_TokenCountIncrements_AfterChatRequest()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("m"))));
        var client = factory.CreateClient();

        var before = await GetMetricValue(client, "sharpi_tokens_generated_total");

        var req = new
        {
            model = "m",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false
        };
        await client.PostAsJsonAsync("/v1/chat/completions", req);

        var after = await GetMetricValue(client, "sharpi_tokens_generated_total");
        Assert.True(after > before, $"Expected token counter to increment, got before={before} after={after}");
    }

    private static async Task<double> GetMetricValue(HttpClient client, string metricName)
    {
        var body = await (await client.GetAsync("/metrics")).Content.ReadAsStringAsync();
        foreach (var line in body.Split('\n'))
        {
            if (line.StartsWith(metricName + ' ') || line.StartsWith(metricName + '{'))
            {
                var parts = line.Split(' ');
                if (parts.Length >= 2 && double.TryParse(parts[^1], out double val))
                    return val;
            }
        }
        return double.NaN;
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

    // ── logit_bias ────────────────────────────────────────────────────────────

    [Fact]
    public async Task ChatCompletion_WithLogitBias_AcceptsRequest()
    {
        // Verifies that logit_bias is accepted and does not cause a 400/500.
        // The FakeEngine ignores it, so output is the same — we only check wire format.
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 5,
            stream = false,
            logit_bias = new Dictionary<string, float> { { "5", -100f }, { "42", 10f } }
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("chat.completion", json);
    }

    // ── response_format (structured outputs) ─────────────────────────────────

    [Fact]
    public async Task ChatCompletion_WithResponseFormatJsonObject_AcceptsRequest()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Give me JSON" } },
            max_tokens = 5,
            stream = false,
            response_format = new { type = "json_object" }
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("chat.completion", json);
    }

    [Fact]
    public async Task ChatCompletion_WithResponseFormatText_AcceptsRequest()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 5,
            stream = false,
            response_format = new { type = "text" }
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    // ── OpenAI Responses API (/v1/responses) ─────────────────────────────────

    [Fact]
    public async Task Responses_NonStreaming_StringInput_ReturnsResponseObject()
    {
        var req = new { model = "test-model", input = "Hello!", max_output_tokens = 10, stream = false };
        var response = await _client.PostAsJsonAsync("/v1/responses", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("\"object\":\"response\"", json.Replace(" ", ""));
        Assert.Contains("completed", json);
        Assert.Contains("output_text", json);
        Assert.Contains("Hello", json); // from FakeEngine output
    }

    [Fact]
    public async Task Responses_NonStreaming_ArrayInput_ReturnsResponseObject()
    {
        var req = new
        {
            model = "test-model",
            input = new[] { new { role = "user", content = "Hi" } },
            max_output_tokens = 10,
            stream = false
        };
        var response = await _client.PostAsJsonAsync("/v1/responses", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("completed", json);
        Assert.Contains("output_text", json);
    }

    [Fact]
    public async Task Responses_NonStreaming_WithInstructions_ReturnsOk()
    {
        var req = new
        {
            model = "test-model",
            input = "Say hello",
            instructions = "Be concise.",
            max_output_tokens = 10,
            stream = false
        };
        var response = await _client.PostAsJsonAsync("/v1/responses", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("completed", json);
    }

    [Fact]
    public async Task Responses_Streaming_ReturnsSseEvents()
    {
        var req = new
        {
            model = "test-model",
            input = "Hello",
            max_output_tokens = 10,
            stream = true
        };
        var response = await _client.PostAsJsonAsync("/v1/responses", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("event: response.created", body);
        Assert.Contains("event: response.output_text.delta", body);
        Assert.Contains("event: response.output_text.done", body);
        Assert.Contains("event: response.completed", body);
        Assert.Contains("in_progress", body);
        Assert.Contains("completed", body);
    }

    [Fact]
    public async Task Responses_MissingInput_ReturnsEmptyCompletion()
    {
        // Null input → treated as empty user message, should still return 200
        var req = new { model = "test-model", max_output_tokens = 5, stream = false };
        var response = await _client.PostAsJsonAsync("/v1/responses", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    // ── Thinking-mode request fields ─────────────────────────────────────────

    [Fact]
    public async Task ChatCompletion_WithEnableThinkingFalse_AcceptsRequest()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 5,
            stream = false,
            enable_thinking = false,
            reasoning_effort = "low",
        };
        var response = await _client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task AnthropicMessages_WithThinkingDisabled_AcceptsRequest()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
            thinking = new { type = "disabled" },
        };
        var response = await _client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task AnthropicMessages_WithThinkingEnabledAndBudget_AcceptsRequest()
    {
        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
            thinking = new { type = "enabled", budget_tokens = 1024 },
        };
        var response = await _client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}

public sealed class ChatTemplateScrubTests
{
    [Fact]
    public void Scrub_RemovesClosedBlock()
    {
        var input = "<think>plan stuff</think>The answer is 42.";
        Assert.Equal("The answer is 42.", ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_RemovesMultipleBlocksGreedily()
    {
        // Greedy match drops everything between the first <think> and the last
        // </think>, including text between successive blocks. Real reasoning
        // models emit exactly one block per turn, so this only affects malformed
        // history; greedy avoids orphan-tag leakage on nested input.
        var input = "<think>a</think>foo<think>b</think>bar";
        Assert.Equal("bar", ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_HandlesNestedBlocks()
    {
        var input = "<think><think>nested</think></think>after";
        Assert.Equal("after", ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_RemovesOrphanCloseTag()
    {
        var input = "</think>stray close";
        Assert.Equal("stray close", ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_DropsUnclosedThinkAndEverythingAfter()
    {
        var input = "intro <think>started but never finished answering";
        Assert.Equal("intro ", ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_LeavesContentWithoutThinkUntouched()
    {
        var input = "Just a normal answer with no reasoning.";
        Assert.Equal(input, ChatTemplate.ScrubAssistantThinking(input));
    }

    [Fact]
    public void Scrub_HandlesEmptyAndNullSafely()
    {
        Assert.Equal("", ChatTemplate.ScrubAssistantThinking(""));
    }

    [Fact]
    public void Scrub_HandlesMultilineThinkBlock()
    {
        var input = "<think>line 1\nline 2\nline 3</think>final answer";
        Assert.Equal("final answer", ChatTemplate.ScrubAssistantThinking(input));
    }
}

/// <summary>
/// Fake inference engine for integration tests. Emits "Hello world" as individual word tokens.
/// </summary>
internal sealed class FakeInferenceEngine : IInferenceEngine
{
    public string ModelId { get; }
    public int QueueDepth => 0;
    public int ActiveRequests => 0;

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
