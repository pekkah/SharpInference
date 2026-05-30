using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// End-to-end coverage for the tool-call wire formats wired up by issues #95–#97:
/// Qwen3-Coder's bare <c>&lt;function=&gt;</c> shape on /v1/messages, and the
/// OpenAI /v1/chat/completions tool-call request + response parity.
///
/// The fake engine emits canned script output regardless of architecture, so we
/// exercise the parser by swapping the configured architecture via
/// <see cref="SharpInferenceServerOptions.Architecture"/>.
/// </summary>
public sealed class ToolCallEndpointTests
{
    private static HttpClient CreateClient(
        FakeInferenceEngine fake,
        string architecture = "qwen2") =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<SharpInferenceServerOptions>(o => o.Architecture = architecture);
                s.AddSingleton<IInferenceEngine>(fake);
            }))
            .CreateClient();

    // ── /v1/messages with Qwen3-Coder bare-function shape (#95) ────────────────

    [Fact]
    public async Task Anthropic_QwenCoder_NonStreaming_ParsesBareFunctionAsToolUse()
    {
        var fake = new FakeInferenceEngine("qwen3-coder", [
            (GenerateChunkKind.Text, "<function=get_weather>"),
            (GenerateChunkKind.Text, "<parameter=city>Paris</parameter>"),
            (GenerateChunkKind.Text, "</function>"),
        ]);
        var client = CreateClient(fake, "qwen3coder");

        var req = new
        {
            model = "qwen3-coder",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 50,
            stream = false,
            tools = new[] { new
            {
                name = "get_weather",
                description = "Get weather",
                input_schema = new { type = "object", properties = new { city = new { type = "string" } } }
            } }
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
        Assert.Equal("Paris", block.GetProperty("input").GetProperty("city").GetString());
    }

    [Fact]
    public async Task Anthropic_QwenCoder_Streaming_EmitsToolUseEvents()
    {
        var fake = new FakeInferenceEngine("qwen3-coder", [
            (GenerateChunkKind.Text, "<function=bash>"),
            (GenerateChunkKind.Text, "<parameter=command>ls</parameter>"),
            (GenerateChunkKind.Text, "</function>"),
        ]);
        var client = CreateClient(fake, "qwen3coder");

        var req = new
        {
            model = "qwen3-coder",
            messages = new[] { new { role = "user", content = "ls" } },
            max_tokens = 50,
            stream = true,
            tools = new[] { new { name = "bash", description = "shell", input_schema = new { type = "object" } } }
        };
        var response = await client.PostAsJsonAsync("/v1/messages", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        Assert.Contains("\"type\":\"tool_use\"", body);
        Assert.Contains("\"name\":\"bash\"", body);
        Assert.Contains("input_json_delta", body);
        Assert.Contains("event: message_stop", body);
        // The stop_reason on the terminating message_delta must be tool_use.
        Assert.Contains("tool_use", body);
    }

    // ── /v1/chat/completions tool-call non-streaming (#97) ────────────────────

    [Fact]
    public async Task OpenAi_WithTools_NonStreaming_EmitsToolCallsArray()
    {
        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "<tool_call>"),
            (GenerateChunkKind.Text, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"),
            (GenerateChunkKind.Text, "</tool_call>"),
        ]);
        var client = CreateClient(fake);

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 50,
            stream = false,
            tools = new[] { new
            {
                type = "function",
                function = new
                {
                    name = "get_weather",
                    description = "Get weather",
                    parameters = new { type = "object", properties = new { city = new { type = "string" } } }
                }
            } }
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);

        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());

        var message = choice.GetProperty("message");
        // Content must be null (or omitted) when only tool_calls were produced.
        if (message.TryGetProperty("content", out var c))
            Assert.True(c.ValueKind == JsonValueKind.Null, "content must be null when only tool_calls produced");

        var toolCalls = message.GetProperty("tool_calls");
        Assert.Equal(1, toolCalls.GetArrayLength());
        var call = toolCalls[0];
        Assert.Equal("function", call.GetProperty("type").GetString());
        Assert.True(call.TryGetProperty("id", out _), "tool_call must have id");
        Assert.Equal("get_weather", call.GetProperty("function").GetProperty("name").GetString());
        var argsStr = call.GetProperty("function").GetProperty("arguments").GetString();
        Assert.NotNull(argsStr);
        using var argsDoc = JsonDocument.Parse(argsStr!);
        Assert.Equal("Paris", argsDoc.RootElement.GetProperty("city").GetString());
    }

    [Fact]
    public async Task OpenAi_WithTools_NonStreaming_TextBeforeCall_SurfacesBoth()
    {
        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "Looking it up. "),
            (GenerateChunkKind.Text, "<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>"),
        ]);
        var client = CreateClient(fake);

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "go" } },
            max_tokens = 50,
            stream = false,
            tools = new[] { new { type = "function", function = new { name = "x", description = "", parameters = new { type = "object" } } } }
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());

        var message = choice.GetProperty("message");
        Assert.Equal("Looking it up. ", message.GetProperty("content").GetString());
        Assert.Equal(1, message.GetProperty("tool_calls").GetArrayLength());
    }

    // ── /v1/chat/completions tool-call streaming (#97) ────────────────────────

    [Fact]
    public async Task OpenAi_WithTools_Streaming_EmitsToolCallDelta()
    {
        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "<tool_call>"),
            (GenerateChunkKind.Text, "{\"name\":\"bash\",\"arguments\":{\"cmd\":\"ls\"}}"),
            (GenerateChunkKind.Text, "</tool_call>"),
        ]);
        var client = CreateClient(fake);

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "ls" } },
            max_tokens = 50,
            stream = true,
            tools = new[] { new { type = "function", function = new { name = "bash", description = "shell", parameters = new { type = "object" } } } }
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var body = await response.Content.ReadAsStringAsync();

        // tool_calls delta carrying name + arguments, finish_reason flips to tool_calls.
        Assert.Contains("\"tool_calls\":[", body);
        Assert.Contains("\"name\":\"bash\"", body);
        Assert.Contains("\"finish_reason\":\"tool_calls\"", body);
        Assert.Contains("[DONE]", body);
    }

    // ── /v1/chat/completions tool history echo (#97) ──────────────────────────

    [Fact]
    public async Task OpenAi_ToolMessageInHistory_IsAccepted()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake);

        var req = new
        {
            model = "test-model",
            messages = new object[]
            {
                new { role = "user", content = "Weather?" },
                new
                {
                    role = "assistant",
                    content = (string?)null,
                    tool_calls = new[]
                    {
                        new
                        {
                            id = "call_1",
                            type = "function",
                            function = new { name = "get_weather", arguments = "{\"city\":\"Paris\"}" }
                        }
                    }
                },
                new { role = "tool", tool_call_id = "call_1", content = "Sunny, 22C" },
            },
            max_tokens = 20,
            stream = false,
            tools = new[] { new { type = "function", function = new { name = "get_weather", description = "weather", parameters = new { type = "object" } } } }
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var json = await response.Content.ReadAsStringAsync();
        // FakeEngine has no clue about tools — it just echoes; we only want to confirm
        // the rich-message + role:"tool" path doesn't bomb out on parse.
        Assert.Contains("chat.completion", json);
    }

    // ── /v1/chat/completions no-tools path stays unchanged ────────────────────

    [Fact]
    public async Task OpenAi_NoTools_StreamingPreservesPerChunkContentDeltas()
    {
        // Sanity: streaming without tools must NOT activate the buffering state machine,
        // so per-chunk content_deltas continue to arrive separately.
        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "Hi"),
            (GenerateChunkKind.Text, "!"),
        ]);
        var client = CreateClient(fake);

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
        Assert.Contains("\"content\":\"Hi\"", body);
        Assert.Contains("\"content\":\"!\"", body);
        Assert.Contains("\"finish_reason\":\"stop\"", body);
    }
}
