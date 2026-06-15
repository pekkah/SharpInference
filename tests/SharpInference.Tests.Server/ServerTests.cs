using System.Net;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Core;
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

    // ── Anthropic thinking-block routing ──────────────────────────────────────

    [Fact]
    public async Task AnthropicMessages_NonStreaming_TextOnly_EmitsSingleTextBlock()
    {
        // Regression guard: a response with no Thinking chunks must keep the pre-thinking
        // wire shape — exactly one element in content[], of type "text".
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [(GenerateChunkKind.Text, "just the answer")]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var content = doc.RootElement.GetProperty("content");
        Assert.Equal(1, content.GetArrayLength());
        Assert.Equal("text", content[0].GetProperty("type").GetString());
        Assert.Equal("just the answer", content[0].GetProperty("text").GetString());
        Assert.False(content[0].TryGetProperty("thinking", out _));
        Assert.False(content[0].TryGetProperty("signature", out _));
    }

    [Fact]
    public async Task AnthropicMessages_NonStreaming_WithThinking_EmitsThinkingThenTextBlock()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [
                    (GenerateChunkKind.Thinking, "Let me "),
                    (GenerateChunkKind.Thinking, "reason."),
                    (GenerateChunkKind.Text, "Answer is 42."),
                ]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "What is 6*7?" } },
            max_tokens = 20,
            stream = false,
            thinking = new { type = "enabled", budget_tokens = 1024 },
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var content = doc.RootElement.GetProperty("content");
        Assert.Equal(2, content.GetArrayLength());

        // Index 0 must be the thinking block with concatenated reasoning + non-empty signature.
        var t = content[0];
        Assert.Equal("thinking", t.GetProperty("type").GetString());
        Assert.Equal("Let me reason.", t.GetProperty("thinking").GetString());
        var sig = t.GetProperty("signature").GetString();
        Assert.False(string.IsNullOrEmpty(sig), "signature must be present and non-empty");
        // The thinking block must not leak a stray "text" field.
        Assert.False(t.TryGetProperty("text", out _));

        // Index 1 is the text block.
        var x = content[1];
        Assert.Equal("text", x.GetProperty("type").GetString());
        Assert.Equal("Answer is 42.", x.GetProperty("text").GetString());
        Assert.False(x.TryGetProperty("thinking", out _));
    }

    [Fact]
    public async Task AnthropicMessages_Streaming_TextOnly_NoThinkingEvents()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [(GenerateChunkKind.Text, "Hi"), (GenerateChunkKind.Text, "!")]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = true,
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        // No thinking-related events should appear at all.
        Assert.DoesNotContain("thinking_delta", body);
        Assert.DoesNotContain("signature_delta", body);
        Assert.DoesNotContain("\"type\":\"thinking\"", body);

        // Text block opens at index 0 with text_delta payloads.
        Assert.Contains("event: content_block_start", body);
        Assert.Contains("\"index\":0", body);
        Assert.Contains("text_delta", body);
        Assert.Contains("event: content_block_stop", body);
        Assert.Contains("event: message_stop", body);
    }

    [Fact]
    public async Task AnthropicMessages_Streaming_WithThinking_EmitsTypedEventSequence()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [
                    (GenerateChunkKind.Thinking, "step1"),
                    (GenerateChunkKind.Thinking, "step2"),
                    (GenerateChunkKind.Text, "final"),
                ]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Reason." } },
            max_tokens = 30,
            stream = true,
            thinking = new { type = "enabled", budget_tokens = 512 },
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        // Thinking block must be index 0 with a "thinking" content_block kind.
        int thinkingStart = body.IndexOf("\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"", StringComparison.Ordinal);
        Assert.True(thinkingStart >= 0, "thinking content_block_start at index 0 missing\n" + body);

        // At least one thinking_delta arrives at index 0.
        int firstThinkingDelta = body.IndexOf("thinking_delta", thinkingStart, StringComparison.Ordinal);
        Assert.True(firstThinkingDelta > thinkingStart);

        // signature_delta arrives before the thinking block's content_block_stop.
        int sigDelta = body.IndexOf("signature_delta", StringComparison.Ordinal);
        int firstStop = body.IndexOf("\"type\":\"content_block_stop\",\"index\":0", StringComparison.Ordinal);
        Assert.True(sigDelta > 0 && firstStop > sigDelta, $"signature_delta must precede content_block_stop@0; sig={sigDelta}, stop={firstStop}");

        // Text block then opens at index 1 with text_delta.
        int textStart = body.IndexOf("\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"", StringComparison.Ordinal);
        Assert.True(textStart > firstStop, "text content_block_start at index 1 missing or out of order");
        int textDelta = body.IndexOf("text_delta", textStart, StringComparison.Ordinal);
        Assert.True(textDelta > textStart);

        // And the text block closes with its own stop@1, followed by message_delta/message_stop.
        int textStop = body.IndexOf("\"type\":\"content_block_stop\",\"index\":1", textStart, StringComparison.Ordinal);
        Assert.True(textStop > textDelta);
        Assert.Contains("event: message_delta", body);
        Assert.Contains("event: message_stop", body);
    }

    // ── OpenAI reasoning_content routing ──────────────────────────────────────

    [Fact]
    public async Task ChatCompletion_NonStreaming_TextOnly_OmitsReasoningContent()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [(GenerateChunkKind.Text, "just the answer")]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("just the answer", message.GetProperty("content").GetString());
        Assert.False(message.TryGetProperty("reasoning_content", out _));
        Assert.False(doc.RootElement.GetProperty("usage").TryGetProperty("completion_tokens_details", out _));
    }

    [Fact]
    public async Task ChatCompletion_NonStreaming_WithReasoning_EmitsReasoningContentAndUsageDetails()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [
                    (GenerateChunkKind.Thinking, "Let me "),
                    (GenerateChunkKind.Thinking, "think."),
                    (GenerateChunkKind.Text, "The answer is 42."),
                ]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "What is 6*7?" } },
            max_tokens = 20,
            stream = false,
            enable_thinking = true,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("The answer is 42.", message.GetProperty("content").GetString());
        Assert.Equal("Let me think.", message.GetProperty("reasoning_content").GetString());

        var details = doc.RootElement.GetProperty("usage").GetProperty("completion_tokens_details");
        Assert.Equal(2, details.GetProperty("reasoning_tokens").GetInt32());
        Assert.Equal(3, doc.RootElement.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
    }

    [Fact]
    public async Task ChatCompletion_Streaming_TextOnly_NoReasoningContentDeltas()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [(GenerateChunkKind.Text, "Hi"), (GenerateChunkKind.Text, "!")]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = true,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        Assert.DoesNotContain("reasoning_content", body);
        Assert.DoesNotContain("completion_tokens_details", body);
        Assert.Contains("\"content\":\"Hi\"", body);
        Assert.Contains("\"content\":\"!\"", body);
        Assert.Contains("\"finish_reason\":\"stop\"", body);
        Assert.Contains("[DONE]", body);
    }

    [Fact]
    public async Task ChatCompletion_Streaming_WithReasoning_SplitsReasoningThenContentDeltas()
    {
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine(
                "test-model",
                [
                    (GenerateChunkKind.Thinking, "step1"),
                    (GenerateChunkKind.Thinking, "step2"),
                    (GenerateChunkKind.Text, "final"),
                ]))));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Reason." } },
            max_tokens = 30,
            stream = true,
            enable_thinking = true,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        int firstReasoning = body.IndexOf("\"reasoning_content\":\"step1\"", StringComparison.Ordinal);
        int secondReasoning = body.IndexOf("\"reasoning_content\":\"step2\"", StringComparison.Ordinal);
        int textDelta = body.IndexOf("\"content\":\"final\"", StringComparison.Ordinal);
        Assert.True(firstReasoning > 0, "first reasoning_content delta missing\n" + body);
        Assert.True(secondReasoning > firstReasoning, "second reasoning_content delta out of order");
        Assert.True(textDelta > secondReasoning, "content delta must follow all reasoning_content deltas");

        // A single delta must never carry both fields.
        Assert.DoesNotContain("\"content\":\"step", body);
        Assert.DoesNotContain("\"reasoning_content\":\"final", body);

        Assert.Contains("\"finish_reason\":\"stop\"", body);
        Assert.Contains("[DONE]", body);
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

    // ── Thinking-token budget plumbing (SamplingParams.MaxThinkingTokens) ────

    [Fact]
    public async Task AnthropicMessages_ThinkingBudgetTokens_ReachesSamplingParams()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
            thinking = new { type = "enabled", budget_tokens = 2 },
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(2, fake.LastSamplingParams!.MaxThinkingTokens);
    }

    [Fact]
    public async Task AnthropicMessages_NoThinkingBudget_DefaultsToZero()
    {
        // Absence of thinking.budget_tokens must leave MaxThinkingTokens at its 0 (unlimited) default.
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(0, fake.LastSamplingParams!.MaxThinkingTokens);
    }

    [Fact]
    public async Task ChatCompletion_MaxThinkingTokens_ReachesSamplingParams()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
            max_thinking_tokens = 2,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(2, fake.LastSamplingParams!.MaxThinkingTokens);
    }

    [Fact]
    public async Task ChatCompletion_NoMaxThinkingTokens_DefaultsToZero()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 10,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(0, fake.LastSamplingParams!.MaxThinkingTokens);
    }

    // ── Canonical history-prefix plumbing (issue #102) ───────────────────────

    [Fact]
    public async Task ChatCompletion_PassesCanonicalHistoryPrefixToEngine()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[]
            {
                new { role = "system", content = "You are concise." },
                new { role = "user", content = "Hi" },
            },
            max_tokens = 4,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        // The endpoint must render history-only canonical AND pass it through. Generic
        // assertions only — we don't pin a specific template here because the test fake
        // uses the hardcoded ChatML fallback in ChatTemplateRenderer.
        Assert.NotNull(fake.LastCanonicalHistoryPrefix);
        Assert.NotEqual(string.Empty, fake.LastCanonicalHistoryPrefix);
        // History-only render must NOT include the trailing assistant-prep marker
        // (ChatML: `<|im_start|>assistant\n`). That's the whole point of the param.
        Assert.DoesNotContain("<|im_start|>assistant\n", fake.LastCanonicalHistoryPrefix!);
    }

    [Fact]
    public async Task AnthropicMessages_PassesCanonicalHistoryPrefixToEngine()
    {
        var fake = new FakeInferenceEngine("test-model");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(fake)));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            system = "You are concise.",
            messages = new[] { new { role = "user", content = "Hi" } },
            max_tokens = 4,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastCanonicalHistoryPrefix);
        Assert.DoesNotContain("<|im_start|>assistant\n", fake.LastCanonicalHistoryPrefix!);
    }

    // ── Tool calling ─────────────────────────────────────────────────────────

    // Builds a test host with the qwen <tool_call> tool-call adapter pinned. Without this the
    // default ChatTemplateRenderer is constructed from the bound Architecture option, which a
    // developer's (gitignored) appsettings.Local.json can set to a non-qwen value — that file is
    // loaded by WebApplicationFactory<Program>, so the tool-call tests would otherwise pass in CI
    // but fail locally with whatever adapter the local config selects.
    private static WebApplicationFactory<Program> ToolHostFactory(FakeInferenceEngine engine) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton(new ChatTemplateRenderer("qwen2"));
                s.AddSingleton<IInferenceEngine>(engine);
            }));

    [Fact]
    public async Task AnthropicMessages_WithTools_NonStreaming_ReturnsToolUseBlock()
    {
        // Model output contains a <tool_call> block — endpoint must parse it and return
        // a tool_use content block with stop_reason = "tool_use".
        var factory = ToolHostFactory(new FakeInferenceEngine(
            "test-model",
            [
                (GenerateChunkKind.Text, "<tool_call>\n"),
                (GenerateChunkKind.Text, "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}"),
                (GenerateChunkKind.Text, "\n</tool_call>"),
            ]));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "What's the weather?" } },
            max_tokens = 50,
            stream = false,
            tools = new[]
            {
                new
                {
                    name = "get_weather",
                    description = "Get weather for a city",
                    input_schema = new { type = "object", properties = new { city = new { type = "string" } } }
                }
            }
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);

        Assert.Equal("tool_use", doc.RootElement.GetProperty("stop_reason").GetString());

        var content = doc.RootElement.GetProperty("content");
        Assert.Equal(1, content.GetArrayLength());
        var block = content[0];
        Assert.Equal("tool_use", block.GetProperty("type").GetString());
        Assert.Equal("get_weather", block.GetProperty("name").GetString());
        Assert.True(block.TryGetProperty("id", out _), "tool_use block must have id");
        var input = block.GetProperty("input");
        Assert.Equal("Paris", input.GetProperty("city").GetString());
    }

    [Fact]
    public async Task AnthropicMessages_WithTools_NonStreaming_TextBeforeToolCall_ReturnsBothBlocks()
    {
        var factory = ToolHostFactory(new FakeInferenceEngine(
            "test-model",
            [
                (GenerateChunkKind.Text, "Let me check that for you."),
                (GenerateChunkKind.Text, "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"/foo\"}}\n</tool_call>"),
            ]));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Read /foo" } },
            max_tokens = 50,
            stream = false,
            tools = new[] { new { name = "read_file", description = "Read a file", input_schema = new { type = "object" } } }
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);

        Assert.Equal("tool_use", doc.RootElement.GetProperty("stop_reason").GetString());
        var content = doc.RootElement.GetProperty("content");
        Assert.Equal(2, content.GetArrayLength());
        Assert.Equal("text",     content[0].GetProperty("type").GetString());
        Assert.Equal("tool_use", content[1].GetProperty("type").GetString());
        Assert.Contains("check", content[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task AnthropicMessages_WithTools_Streaming_EmitsToolUseEvents()
    {
        var factory = ToolHostFactory(new FakeInferenceEngine(
            "test-model",
            [
                (GenerateChunkKind.Text, "<tool_call>\n"),
                (GenerateChunkKind.Text, "{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}"),
                (GenerateChunkKind.Text, "\n</tool_call>"),
            ]));
        var client = factory.CreateClient();

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Run ls" } },
            max_tokens = 50,
            stream = true,
            tools = new[] { new { name = "bash", description = "Run shell command", input_schema = new { type = "object" } } }
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        // Must have a tool_use content_block_start event
        Assert.Contains("\"type\":\"tool_use\"", body);
        Assert.Contains("\"name\":\"bash\"", body);
        Assert.Contains("input_json_delta", body);
        Assert.Contains("tool_use", body); // stop_reason in message_delta
        Assert.Contains("event: message_stop", body);
    }

    [Fact]
    public async Task AnthropicMessages_ToolResultInMessages_IsAccepted()
    {
        // Verifies that a multi-turn conversation with tool_result content blocks is
        // accepted without error (the FakeEngine just echoes its script, so we only
        // check status code and basic response shape here).
        var req = new
        {
            model = "test-model",
            messages = new object[]
            {
                new { role = "user", content = "What's the weather?" },
                new
                {
                    role = "assistant",
                    content = new object[]
                    {
                        new { type = "tool_use", id = "toolu_01", name = "get_weather", input = new { city = "Paris" } }
                    }
                },
                new
                {
                    role = "user",
                    content = new object[]
                    {
                        new { type = "tool_result", tool_use_id = "toolu_01", content = "Sunny, 22°C" }
                    }
                }
            },
            max_tokens = 20,
            stream = false,
            tools = new[] { new { name = "get_weather", description = "Get weather", input_schema = new { type = "object" } } }
        };
        var response = await _client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        Assert.Contains("message", json);
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

    // ── Server-level DisableThinking (SHARPI_NO_THINKING) ──────────────────────

    // Jinja template that records whether enable_thinking reached it, so the test can assert
    // the rendered prompt the endpoint produced.
    private const string ThinkProbeTemplate =
        "{% if enable_thinking %}<<THINK>>{% else %}<<NOTHINK>>{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}";

    private static (HttpClient client, FakeInferenceEngine fake) ProbeClient(bool disableThinking)
    {
        var fake = new FakeInferenceEngine("m");
        var renderer = new ChatTemplateRenderer("test", new JinjaChatTemplate(ThinkProbeTemplate));
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton(renderer);                       // overrides the TryAddSingleton default
                s.AddSingleton<IInferenceEngine>(fake);
                s.Configure<SharpInferenceServerOptions>(o => o.DisableThinking = disableThinking);
            }));
        return (factory.CreateClient(), fake);
    }

    [Fact]
    public async Task Anthropic_DisableThinking_ForcesNoThinking_EvenWhenRequestDoesNotOptOut()
    {
        var (client, fake) = ProbeClient(disableThinking: true);
        // Request carries no thinking field → would normally enable thinking; the server flag wins.
        var req = new { model = "m", max_tokens = 16, messages = new[] { new { role = "user", content = "hi" } } };
        var resp = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("<<NOTHINK>>", fake.LastPrompt);
        Assert.DoesNotContain("<<THINK>>", fake.LastPrompt);
    }

    [Fact]
    public async Task Anthropic_ThinkingStaysOn_WhenServerFlagDisabledAndRequestDoesNotOptOut()
    {
        var (client, fake) = ProbeClient(disableThinking: false);
        var req = new { model = "m", max_tokens = 16, messages = new[] { new { role = "user", content = "hi" } } };
        var resp = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("<<THINK>>", fake.LastPrompt);
    }

    [Fact]
    public async Task OpenAi_DisableThinking_ForcesNoThinking()
    {
        var (client, fake) = ProbeClient(disableThinking: true);
        var req = new { model = "m", max_tokens = 16, messages = new[] { new { role = "user", content = "hi" } } };
        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.NotNull(fake.LastPrompt);
        Assert.Contains("<<NOTHINK>>", fake.LastPrompt);
    }
}

/// <summary>
/// Fake inference engine for integration tests. By default emits "Hello world!" as
/// text-only chunks (the back-compat behavior expected by older tests). A test can
/// pass an explicit script of <c>(kind, text)</c> pairs to drive the thinking-routing
/// logic in the endpoints — e.g. <c>[(Thinking, "let me think"), (Text, "the answer")]</c>
/// to exercise the Anthropic <c>thinking</c> block path.
/// </summary>
internal sealed class FakeInferenceEngine : IInferenceEngine
{
    private readonly (GenerateChunkKind Kind, string Text)[] _script;

    public string ModelId { get; }
    public int QueueDepth => 0;
    public int ActiveRequests => 0;
    public bool PrefixCacheEnabled => true;
    public long PrefillTokensReused => 0;

    /// <summary>
    /// Captures the <see cref="SamplingParams"/> handed to the most recent
    /// <see cref="GenerateChunksAsync"/> call. Lets wire-level tests confirm that request
    /// fields (e.g. <c>thinking.budget_tokens</c>, <c>max_thinking_tokens</c>) reach the
    /// engine without needing the full engine plumbing.
    /// </summary>
    public SamplingParams? LastSamplingParams { get; private set; }

    public FakeInferenceEngine(string modelId)
        : this(modelId, [(GenerateChunkKind.Text, "Hello"), (GenerateChunkKind.Text, " world"), (GenerateChunkKind.Text, "!")])
    {
    }

    public FakeInferenceEngine(string modelId, (GenerateChunkKind Kind, string Text)[] script)
    {
        ModelId = modelId;
        _script = script;
    }

    /// <summary>
    /// Captures the canonical-history hint handed to the most recent
    /// <see cref="GenerateChunksAsync"/> call so endpoint tests can verify the
    /// chat-template render reached the engine (issue #102).
    /// </summary>
    public string? LastCanonicalHistoryPrefix { get; private set; }

    /// <summary>The rendered prompt handed to the most recent generation call.</summary>
    public string? LastPrompt { get; private set; }

    /// <summary>Image-input support flag (issue #253) and capture of the most recent image
    /// dispatch, so wire-level tests can confirm image content reached the engine.</summary>
    public bool SupportsImages { get; init; }
    public bool SupportsImageInput => SupportsImages;
    public int LastImageCount { get; private set; }

    public async IAsyncEnumerable<GenerateChunk> GenerateChunksAsync(
        string prompt,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default,
        string? canonicalHistoryPrefix = null)
    {
        LastSamplingParams = sp;
        LastCanonicalHistoryPrefix = canonicalHistoryPrefix;
        LastPrompt = prompt;
        foreach (var (kind, text) in _script)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return new GenerateChunk(kind, text);
        }
    }

    public async IAsyncEnumerable<GenerateChunk> GenerateImageChunksAsync(
        string prompt,
        IReadOnlyList<byte[]> imageBytes,
        SamplingParams sp,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        LastSamplingParams = sp;
        LastCanonicalHistoryPrefix = null;
        LastPrompt = prompt;
        LastImageCount = imageBytes.Count;
        foreach (var (kind, text) in _script)
        {
            ct.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return new GenerateChunk(kind, text);
        }
    }
}
