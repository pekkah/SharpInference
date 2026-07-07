using System.Collections.Immutable;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// End-to-end wiring tests for <c>response_format.json_schema</c> / the llama.cpp-style flat
/// <c>response_format.schema</c> extension (issue #423 follow-up), mirroring
/// <see cref="OutputConstraintEndpointTests"/>'s <c>WebApplicationFactory</c> harness.
/// <see cref="FakeInferenceEngine"/> serves a fixed canned script regardless of <c>sp.Constraint</c>,
/// so these assert WIRING (the right constraint/error reaches
/// <see cref="FakeInferenceEngine.LastSamplingParams"/> or the HTTP response) -- masking behavior
/// itself is covered by <c>SharpInference.Tests.Core.JsonSchemaOutputConstraintTests</c>.
/// </summary>
public sealed class JsonSchemaResponseFormatEndpointTests
{
    /// <summary>Bare-minimum tokenizer. Whole-body mode needs no special/envelope token, only a
    /// working <see cref="GrammarVocabulary"/> to construct against -- but the "AND-composed with
    /// tool-grammar" test also needs Qwen's JSON tool-arg constraint to actually engage, which
    /// requires <c>&lt;tool_call&gt;</c> to be registered as a special token (see
    /// <c>JsonToolArgumentConstraint</c>'s constructor).</summary>
    private sealed class MinimalTokenizer : ITokenizer
    {
        public int VocabSize => 8;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            new Dictionary<string, int>(StringComparer.Ordinal) { ["<tool_call>"] = 1 };
        public byte[] DecodeBytes(int token) => [];
        public IReadOnlyList<int> Encode(string text) => [];
        public string Decode(IEnumerable<int> tokens) => "";
    }

    private static ChatTemplateRenderer RendererWithVocab()
    {
        var renderer = new ChatTemplateRenderer("qwen2");
        renderer.Configure("qwen2", null, null, new GrammarVocabulary(new MinimalTokenizer()));
        return renderer;
    }

    private static HttpClient CreateClient(
        FakeInferenceEngine fake,
        Action<SharpInferenceServerOptions>? configure = null,
        ChatTemplateRenderer? renderer = null) =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                if (configure is not null) s.Configure(configure);
                if (renderer is not null) s.AddSingleton(renderer);
                s.AddSingleton<IInferenceEngine>(fake);
            }))
            .CreateClient();

    private static readonly object ValidSchema = new
    {
        type = "object",
        properties = new { answer = new { type = "string" } },
        required = new[] { "answer" },
    };

    [Fact]
    public async Task NoResponseFormat_NoConstraint()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Null(fake.LastSamplingParams!.Constraint);
    }

    [Fact]
    public async Task JsonSchemaType_ValidSchema_ConstraintIsJsonSchemaOutputConstraint()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_schema", json_schema = new { name = "answer", schema = ValidSchema } },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsType<JsonSchemaOutputConstraint>(fake.LastSamplingParams!.Constraint);
    }

    [Fact]
    public async Task JsonObjectType_WithFlatSchemaExtension_ConstraintIsJsonSchemaOutputConstraint()
    {
        // llama.cpp's flat response_format.schema extension (no OpenAI json_schema envelope needed).
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_object", schema = ValidSchema },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsType<JsonSchemaOutputConstraint>(fake.LastSamplingParams!.Constraint);
    }

    [Fact]
    public async Task JsonSchemaType_UncompilableSchema_Returns400()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_schema", json_schema = new { schema = new { type = "string" } } },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        Assert.Contains("could not be compiled", doc.RootElement.GetProperty("message").GetString());
    }

    [Fact]
    public async Task JsonSchemaType_MissingSchema_Returns400()
    {
        // A missing json_schema.schema must be a client error, not a silent no-op (the client
        // explicitly asked for type:"json_schema" -- proceeding unconstrained would violate the
        // "explicit schema request is a hard requirement" contract).
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_schema" },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        Assert.Contains("no json_schema.schema", doc.RootElement.GetProperty("message").GetString());
    }

    [Fact]
    public async Task JsonObjectType_NoSchema_IsNotAnError()
    {
        // Unlike "json_schema", "json_object" without a schema is the pre-existing lenient behavior
        // (valid JSON, no shape enforced) -- must NOT become an error.
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, renderer: RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_object" },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Null(fake.LastSamplingParams!.Constraint);
    }

    [Fact]
    public async Task JsonSchemaType_NoVocabularyAvailable_Returns400()
    {
        // No pinned renderer -- the default DI-constructed ChatTemplateRenderer never gets a
        // GrammarVocabulary in these tests (no GGUF is loaded), so the request must fail loudly
        // rather than silently generate unconstrained output.
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake);

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "hi" } },
            max_tokens = 10,
            stream = false,
            response_format = new { type = "json_schema", json_schema = new { schema = ValidSchema } },
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        var json = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(json);
        Assert.Contains("vocabulary", doc.RootElement.GetProperty("message").GetString());
    }

    [Fact]
    public async Task JsonSchemaType_AndToolGrammarActive_ConstraintIsAndComposed()
    {
        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "<tool_call>"),
            (GenerateChunkKind.Text, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"),
            (GenerateChunkKind.Text, "</tool_call>"),
        ]);
        var client = CreateClient(fake, o => o.ToolGrammar = true, RendererWithVocab());

        var req = new
        {
            model = "test-model",
            messages = new[] { new { role = "user", content = "Weather?" } },
            max_tokens = 50,
            stream = false,
            response_format = new { type = "json_schema", json_schema = new { schema = ValidSchema } },
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

        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsType<AndTokenConstraint>(fake.LastSamplingParams!.Constraint);
    }
}
