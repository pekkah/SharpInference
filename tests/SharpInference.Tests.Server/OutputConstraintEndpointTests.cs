using System.Collections.Immutable;
using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// End-to-end wiring tests for the caller-supplied whole-turn output constraint
/// (<see cref="SharpInferenceServerOptions.OutputConstraintFactory"/>, issue #423): confirms the
/// factory's constraint reaches <c>SamplingParams.Constraint</c> through the real HTTP endpoints, and
/// composes (AND) with the tool-argument constraint when both are active. <see cref="FakeInferenceEngine"/>
/// serves a fixed canned script regardless of <c>sp.Constraint</c>, so these tests assert WIRING (the
/// right constraint instance/type reaches <see cref="FakeInferenceEngine.LastSamplingParams"/>), not
/// masked output -- masking behavior itself is covered by
/// <c>SharpInference.Tests.ForwardPass.SayShowTagConstraintTests</c> and
/// <c>SharpInference.Tests.Core.AndTokenConstraintTests</c>.
/// </summary>
public sealed class OutputConstraintEndpointTests
{
    /// <summary>Trivial never-constraining stub -- its TYPE is all these tests need to assert on.</summary>
    private sealed class StubConstraint : ITokenConstraint
    {
        public bool IsConstraining => false;
        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits) => logits;
        public void Accept(int token) { }
        public void Reset() { }
    }

    /// <summary>
    /// Bare-minimum tokenizer that only needs to make the Qwen JSON tool-argument constraint
    /// constructible (registers <c>&lt;tool_call&gt;</c> as a special token, per
    /// <c>JsonToolArgumentConstraint</c>'s constructor). These tests only assert wiring/composition
    /// and never exercise <c>Filter</c>/<c>Accept</c>, so byte-level decode is unused.
    /// </summary>
    private sealed class MinimalToolTokenizer : ITokenizer
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

    [Fact]
    public async Task NoFactory_NoTools_ConstraintStaysNull()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake);

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
    public async Task FactorySet_NoTools_ConstraintIsFactorysInstance()
    {
        var fake = new FakeInferenceEngine("test-model");
        var client = CreateClient(fake, o => o.OutputConstraintFactory = _ => new StubConstraint());

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
        Assert.IsType<StubConstraint>(fake.LastSamplingParams!.Constraint);
    }

    [Fact]
    public async Task FactorySet_AndToolGrammarActive_ConstraintIsAndComposed()
    {
        // Pin a ChatTemplateRenderer with a real (non-null) GrammarVocabulary so
        // BuildToolArgumentConstraint can actually engage -- the default DI-constructed renderer
        // never gets a vocabulary in these tests (no GGUF is loaded).
        var renderer = new ChatTemplateRenderer("qwen2");
        renderer.Configure("qwen2", null, null, new GrammarVocabulary(new MinimalToolTokenizer()));

        var fake = new FakeInferenceEngine("test-model", [
            (GenerateChunkKind.Text, "<tool_call>"),
            (GenerateChunkKind.Text, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"),
            (GenerateChunkKind.Text, "</tool_call>"),
        ]);
        var client = CreateClient(fake,
            o => { o.OutputConstraintFactory = _ => new StubConstraint(); o.ToolGrammar = true; },
            renderer);

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

        Assert.NotNull(fake.LastSamplingParams);
        Assert.IsType<AndTokenConstraint>(fake.LastSamplingParams!.Constraint);
    }
}
