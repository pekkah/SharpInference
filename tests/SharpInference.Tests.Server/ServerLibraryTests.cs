using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.Routing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Server;
using SharpInference.Server.Endpoints;

namespace SharpInference.Tests.Server;

/// <summary>Defaults baked into <see cref="SharpInferenceServerOptions"/>.</summary>
public sealed class SharpInferenceServerOptionsTests
{
    [Fact]
    public void Defaults_AreSafeForOutOfTheBoxBoot()
    {
        var opts = new SharpInferenceServerOptions();

        Assert.Null(opts.ModelPath);
        Assert.Equal(1, opts.MaxBatchSize);
        Assert.Equal("qwen2", opts.Architecture);
        Assert.Null(opts.EngineFactory);

        // Backend / hardware defaults: CPU-only out of the box.
        Assert.Equal(ServerBackend.Auto, opts.Backend);
        Assert.Equal(0, opts.NGpuLayers);
        Assert.Equal(0, opts.ContextSize);
        Assert.False(opts.TurboQuant);
        Assert.Equal(0, opts.MinBatchBlas);

        // MoE knob defaults: predictive prefetch on, nothing pinned.
        Assert.Null(opts.MoeWarmPin);
        Assert.Equal(0L, opts.MoeWarmPinAfter);
        Assert.True(opts.MoePredictPrefetch);
        Assert.Null(opts.ExpertStatsPath);

        // Spec-decode defaults: auto-engage MTP when supported, strict argmax-match.
        Assert.Equal(ServerSpecType.Auto, opts.SpecType);
        Assert.Equal(0, opts.SpecDraftNMax);
        Assert.Equal(0, opts.SpecDraftNMin);
        Assert.Equal(1f, opts.SpecDraftPMin);

        // Sampling defaults: same as the CLI's defaults.
        Assert.NotNull(opts.Sampling);
        Assert.Equal(1f,   opts.Sampling.Temperature);
        Assert.Equal(0,    opts.Sampling.TopK);
        Assert.Equal(1f,   opts.Sampling.TopP);
        Assert.Equal(0f,   opts.Sampling.MinP);
        Assert.Equal(1f,   opts.Sampling.RepetitionPenalty);
        Assert.Equal(512,  opts.Sampling.MaxNewTokens);
        Assert.Equal(0,    opts.Sampling.MaxThinkingTokens);
    }
}

/// <summary>
/// Verifies that the full CLI surface — backend / GPU / spec-decode / sampling — binds
/// from a flat <see cref="IConfiguration"/> tree, the way operators would author it in
/// <c>appsettings.Local.json</c>.
/// </summary>
public sealed class OptionsConfigurationBindingTests
{
    [Fact]
    public void BindsBackendAndHardwareFields()
    {
        var cfg = new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["SharpInference:Backend"]      = "Cuda",
            ["SharpInference:NGpuLayers"]   = "-1",
            ["SharpInference:ContextSize"]  = "4096",
            ["SharpInference:TurboQuant"]   = "true",
            ["SharpInference:MinBatchBlas"] = "32",
        }).Build();

        var s = new ServiceCollection();
        s.AddSharpInference(cfg);

        var opts = s.BuildServiceProvider().GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;
        Assert.Equal(ServerBackend.Cuda, opts.Backend);
        Assert.Equal(-1,   opts.NGpuLayers);
        Assert.Equal(4096, opts.ContextSize);
        Assert.True(opts.TurboQuant);
        Assert.Equal(32, opts.MinBatchBlas);
    }

    [Fact]
    public void BindsMoeAndSpecAndSamplingNestedTrees()
    {
        var cfg = new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            // MoE knobs
            ["SharpInference:MoeWarmPin"]          = "4",
            ["SharpInference:MoeWarmPinAfter"]     = "1024",
            ["SharpInference:MoePredictPrefetch"]  = "false",
            ["SharpInference:ExpertStatsPath"]     = "/tmp/stats.json",

            // Spec decode
            ["SharpInference:SpecType"]            = "Mtp",
            ["SharpInference:SpecDraftNMax"]       = "2",
            ["SharpInference:SpecDraftPMin"]       = "0.75",

            // Nested sampling defaults
            ["SharpInference:Sampling:Temperature"]   = "0.7",
            ["SharpInference:Sampling:TopK"]          = "40",
            ["SharpInference:Sampling:TopP"]          = "0.95",
            ["SharpInference:Sampling:MinP"]          = "0.05",
            ["SharpInference:Sampling:MaxNewTokens"]  = "2048",
        }).Build();

        var s = new ServiceCollection();
        s.AddSharpInference(cfg);
        var opts = s.BuildServiceProvider().GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;

        Assert.Equal(4,                opts.MoeWarmPin);
        Assert.Equal(1024L,            opts.MoeWarmPinAfter);
        Assert.False(opts.MoePredictPrefetch);
        Assert.Equal("/tmp/stats.json", opts.ExpertStatsPath);

        Assert.Equal(ServerSpecType.Mtp, opts.SpecType);
        Assert.Equal(2,    opts.SpecDraftNMax);
        Assert.Equal(0.75f, opts.SpecDraftPMin);

        Assert.Equal(0.7f, opts.Sampling.Temperature);
        Assert.Equal(40,   opts.Sampling.TopK);
        Assert.Equal(0.95f, opts.Sampling.TopP);
        Assert.Equal(0.05f, opts.Sampling.MinP);
        Assert.Equal(2048, opts.Sampling.MaxNewTokens);
    }
}

/// <summary>
/// Per-request sampling overrides should win over the host's
/// <see cref="SamplingDefaults"/> when the HTTP request supplies them. Verified by
/// inspecting the <see cref="SamplingParams"/> the engine receives via
/// <see cref="FakeInferenceEngine.LastSamplingParams"/>.
/// </summary>
public sealed class SamplingDefaultsTests
{
    [Fact]
    public async Task RequestOverrides_BeatHostDefaults()
    {
        var fake = new FakeInferenceEngine("m");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.Configure<SharpInferenceServerOptions>(o =>
                {
                    o.Sampling.Temperature = 0.7f;
                    o.Sampling.TopP        = 0.5f;
                    o.Sampling.TopK        = 40;
                    o.Sampling.MaxNewTokens = 999;
                });
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        var client = factory.CreateClient();

        var req = new
        {
            model = "m",
            messages = new[] { new { role = "user", content = "Hi" } },
            temperature = 0.1f,   // overrides 0.7
            max_tokens = 5,       // overrides 999
            // top_p, top_k omitted → host defaults apply
            stream = false,
        };
        var response = await client.PostAsJsonAsync("/v1/chat/completions", req);
        Assert.Equal(System.Net.HttpStatusCode.OK, response.StatusCode);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(0.1f, fake.LastSamplingParams!.Temperature);  // request wins
        Assert.Equal(5,    fake.LastSamplingParams.MaxNewTokens);  // request wins
        Assert.Equal(0.5f, fake.LastSamplingParams.TopP);          // host default
        Assert.Equal(40,   fake.LastSamplingParams.TopK);          // host default
    }

    [Fact]
    public async Task SpecDecodeOptions_ReachSamplingParams()
    {
        var fake = new FakeInferenceEngine("m");
        var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.Configure<SharpInferenceServerOptions>(o =>
                {
                    o.SpecType        = ServerSpecType.Mtp;
                    o.SpecDraftNMax   = 2;
                    o.SpecDraftPMin   = 0.75f;
                });
                s.AddSingleton<IInferenceEngine>(fake);
            }));
        var client = factory.CreateClient();

        var req = new { model = "m", messages = new[] { new { role = "user", content = "Hi" } }, max_tokens = 5, stream = false };
        await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.NotNull(fake.LastSamplingParams);
        Assert.Equal(SpecType.Mtp, fake.LastSamplingParams!.SpecType);
        Assert.Equal(2,     fake.LastSamplingParams.SpecDraftNMax);
        Assert.Equal(0.75f, fake.LastSamplingParams.SpecDraftPMin);
    }
}

/// <summary>Behaviour of the chat-template renderer in isolation from DI.</summary>
public sealed class ChatTemplateRendererTests
{
    [Fact]
    public void Default_ArchitectureIsQwen2_NoTemplate()
    {
        var r = new ChatTemplateRenderer();
        Assert.Equal("qwen2", r.Architecture);
        Assert.Null(r.JinjaTemplate);
    }

    [Fact]
    public void Fallback_Qwen2_EmitsChatMLFraming()
    {
        var r = new ChatTemplateRenderer("qwen2");
        var prompt = r.Format([("user", "hi")]);

        // ChatML: <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n
        Assert.Contains("<|im_start|>user\nhi<|im_end|>", prompt);
        Assert.EndsWith("<|im_start|>assistant\n", prompt);
    }

    [Fact]
    public void Fallback_Llama_EmitsLlama3Framing()
    {
        var r = new ChatTemplateRenderer("llama");
        var prompt = r.Format([("system", "be brief"), ("user", "hi")]);

        Assert.StartsWith("<|begin_of_text|>", prompt);
        Assert.Contains("<|start_header_id|>system<|end_header_id|>\n\nbe brief<|eot_id|>", prompt);
        Assert.Contains("<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>", prompt);
        Assert.EndsWith("<|start_header_id|>assistant<|end_header_id|>\n\n", prompt);
    }

    [Fact]
    public void Fallback_Llama4_EmitsHeaderStartFraming()
    {
        var r = new ChatTemplateRenderer("llama4");
        var prompt = r.Format([("user", "hi")]);

        Assert.StartsWith("<|begin_of_text|>", prompt);
        Assert.Contains("<|header_start|>user<|header_end|>\n\nhi<|eot_id|>", prompt);
        Assert.EndsWith("<|header_start|>assistant<|header_end|>\n\n", prompt);
    }

    [Fact]
    public void Fallback_UnknownArch_FallsThroughToChatML()
    {
        // Any non-llama/llama4 arch falls into the ChatML branch (default for
        // Qwen, SmolLM2, and most other GGUF models).
        var r = new ChatTemplateRenderer("phi3");
        var prompt = r.Format([("user", "hi")]);

        Assert.Contains("<|im_start|>user\nhi<|im_end|>", prompt);
        Assert.EndsWith("<|im_start|>assistant\n", prompt);
    }

    [Fact]
    public void Configure_ReplacesArchAndTemplate()
    {
        var r = new ChatTemplateRenderer("qwen2");
        Assert.Null(r.JinjaTemplate);

        var jinja = new JinjaChatTemplate("HELLO {{ messages[0].content }}");
        r.Configure("llama4", jinja);

        Assert.Equal("llama4", r.Architecture);
        Assert.Same(jinja, r.JinjaTemplate);
    }

    [Fact]
    public void Configure_CanClearTemplate()
    {
        var r = new ChatTemplateRenderer("qwen2", new JinjaChatTemplate("X"));
        r.Configure("qwen2", null);
        Assert.Null(r.JinjaTemplate);
    }

    [Fact]
    public void JinjaTemplate_TakesPrecedenceOverArchFallback()
    {
        // Once configured with a Jinja template, the architecture fallback is bypassed.
        var jinja = new JinjaChatTemplate("FROM-TEMPLATE: {{ messages[0].content }}");
        var r = new ChatTemplateRenderer("llama", jinja);

        var prompt = r.Format([("user", "abc")]);

        Assert.Equal("FROM-TEMPLATE: abc", prompt);
        // The hardcoded llama framing should NOT appear:
        Assert.DoesNotContain("<|begin_of_text|>", prompt);
        Assert.DoesNotContain("<|start_header_id|>", prompt);
    }

    [Fact]
    public void JinjaTemplate_ReceivesEnableThinkingFlag()
    {
        // Renderers expose enable_thinking to the Jinja context so reasoning-capable
        // templates can branch on it.
        var jinja = new JinjaChatTemplate(
            "{% if enable_thinking %}THINK{% else %}NO-THINK{% endif %}");
        var r = new ChatTemplateRenderer("qwen2", jinja);

        Assert.Equal("THINK",    r.Format([("user", "x")], enableThinking: true));
        Assert.Equal("NO-THINK", r.Format([("user", "x")], enableThinking: false));
    }

    [Fact]
    public void RichFormat_WithoutJinja_FallsBackToSimplePairs()
    {
        // The dict overload (tool-call path) without a Jinja template should still
        // produce sensible output by extracting role/content fields.
        var r = new ChatTemplateRenderer("qwen2");
        var msgs = new List<Dictionary<string, object?>>
        {
            new() { ["role"] = "user", ["content"] = "hi" },
        };

        var prompt = r.Format(msgs);

        Assert.Contains("<|im_start|>user\nhi<|im_end|>", prompt);
    }

    [Fact]
    public void RichFormat_WithJinjaAndTools_PassesToolsToContext()
    {
        // tools is passed to the Jinja context, but the fallback path ignores it.
        // Verify the Jinja path receives it.
        var jinja = new JinjaChatTemplate(
            "{% if tools %}HAS-TOOLS{% else %}NO-TOOLS{% endif %}");
        var r = new ChatTemplateRenderer("qwen2", jinja);

        var msgs = new List<Dictionary<string, object?>>
        {
            new() { ["role"] = "user", ["content"] = "x" },
        };

        Assert.Equal("HAS-TOOLS", r.Format(msgs, enableThinking: true, tools: new List<object?> { "tool1" }));
        Assert.Equal("NO-TOOLS",  r.Format(msgs, enableThinking: true, tools: null));
    }
}

/// <summary>Counter behaviour of the metrics service.</summary>
public sealed class ServerMetricsTests
{
    [Fact]
    public void Counters_StartAtZero()
    {
        var m = new ServerMetrics();
        Assert.Equal(0, m.TotalRequests);
        Assert.Equal(0, m.TotalTokens);
    }

    [Fact]
    public void Uptime_StartsAtZeroAndAdvances()
    {
        var m = new ServerMetrics();
        var t1 = m.Uptime;
        Thread.Sleep(10);
        var t2 = m.Uptime;
        Assert.True(t1 >= TimeSpan.Zero);
        Assert.True(t2 > t1, $"uptime should advance: t1={t1} t2={t2}");
    }

    [Fact]
    public void RecordRequest_IncrementsByOne()
    {
        var m = new ServerMetrics();
        m.RecordRequest();
        m.RecordRequest();
        m.RecordRequest();
        Assert.Equal(3, m.TotalRequests);
    }

    [Fact]
    public void RecordTokens_Accumulates()
    {
        var m = new ServerMetrics();
        m.RecordTokens(10);
        m.RecordTokens(5);
        m.RecordTokens(100);
        Assert.Equal(115, m.TotalTokens);
    }

    [Fact]
    public void Counters_AreThreadSafe()
    {
        // Confirms the Interlocked.* writes don't lose updates under contention.
        var m = new ServerMetrics();
        Parallel.For(0, 1000, _ =>
        {
            m.RecordRequest();
            m.RecordTokens(2);
        });
        Assert.Equal(1000, m.TotalRequests);
        Assert.Equal(2000, m.TotalTokens);
    }
}

/// <summary>Failure modes of <see cref="InferenceEngineLoader.Load"/>.</summary>
public sealed class InferenceEngineLoaderTests
{
    [Fact]
    public void Load_NullModelPath_Throws()
    {
        var opts = new SharpInferenceServerOptions { ModelPath = null };
        var ex = Assert.Throws<InvalidOperationException>(() => InferenceEngineLoader.Load(opts));
        Assert.Contains("ModelPath", ex.Message);
    }

    [Fact]
    public void Load_EmptyModelPath_Throws()
    {
        var opts = new SharpInferenceServerOptions { ModelPath = "" };
        Assert.Throws<InvalidOperationException>(() => InferenceEngineLoader.Load(opts));
    }

    [Fact]
    public void Load_WhitespaceModelPath_Throws()
    {
        var opts = new SharpInferenceServerOptions { ModelPath = "   " };
        Assert.Throws<InvalidOperationException>(() => InferenceEngineLoader.Load(opts));
    }

    [Fact]
    public void Load_MissingFile_ThrowsWithHelpfulMessage()
    {
        var bogus = Path.Combine(Path.GetTempPath(), $"sharpi-nonexistent-{Guid.NewGuid():N}.gguf");
        var opts = new SharpInferenceServerOptions { ModelPath = bogus };
        var ex = Assert.Throws<InvalidOperationException>(() => InferenceEngineLoader.Load(opts));
        Assert.Contains("Model file not found", ex.Message);
        Assert.Contains("SHARPI_MODEL", ex.Message);
    }
}

/// <summary>DI registrations produced by <see cref="ServiceCollectionExtensions.AddSharpInference"/>.</summary>
public sealed class ServiceCollectionExtensionsTests
{
    [Fact]
    public void AddSharpInference_RegistersExpectedServices()
    {
        var s = new ServiceCollection();
        s.AddSharpInference();

        Assert.Contains(s, d => d.ServiceType == typeof(ServerMetrics));
        Assert.Contains(s, d => d.ServiceType == typeof(ChatTemplateRenderer));
        Assert.Contains(s, d => d.ServiceType == typeof(IInferenceEngine));
        // IOptions<T> is registered as an open generic by AddOptions<T>(), so check by
        // resolving — a closed-generic descriptor never appears in the collection itself.
        using var sp = s.BuildServiceProvider();
        Assert.NotNull(sp.GetService<IOptions<SharpInferenceServerOptions>>());
    }

    [Fact]
    public void AddSharpInference_RegistersAllServicesAsSingletons()
    {
        var s = new ServiceCollection();
        s.AddSharpInference();

        Assert.Equal(ServiceLifetime.Singleton, s.First(d => d.ServiceType == typeof(ServerMetrics)).Lifetime);
        Assert.Equal(ServiceLifetime.Singleton, s.First(d => d.ServiceType == typeof(ChatTemplateRenderer)).Lifetime);
        Assert.Equal(ServiceLifetime.Singleton, s.First(d => d.ServiceType == typeof(IInferenceEngine)).Lifetime);
    }

    [Fact]
    public void AddSharpInference_InlineConfigureCallbackRuns()
    {
        var s = new ServiceCollection();
        s.AddSharpInference(opts =>
        {
            opts.Architecture = "llama4";
            opts.MaxBatchSize = 8;
        });

        var sp = s.BuildServiceProvider();
        var bound = sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;

        Assert.Equal("llama4", bound.Architecture);
        Assert.Equal(8, bound.MaxBatchSize);
    }

    [Fact]
    public void AddSharpInference_RendererUsesArchitectureFromOptions()
    {
        var s = new ServiceCollection();
        s.AddSharpInference(opts => opts.Architecture = "llama");

        var sp = s.BuildServiceProvider();
        var renderer = sp.GetRequiredService<ChatTemplateRenderer>();

        Assert.Equal("llama", renderer.Architecture);
    }

    [Fact]
    public void AddSharpInference_PreRegisteredFakes_AreNotOverridden()
    {
        // TryAddSingleton semantics: tests that wire a fake IInferenceEngine BEFORE
        // calling AddSharpInference should keep the fake.
        var fakeEngine = new FakeInferenceEngine("fake-model");
        var fakeMetrics = new ServerMetrics();
        var fakeRenderer = new ChatTemplateRenderer("custom-arch");

        var s = new ServiceCollection();
        s.AddSingleton<IInferenceEngine>(fakeEngine);
        s.AddSingleton(fakeMetrics);
        s.AddSingleton(fakeRenderer);
        s.AddSharpInference();

        var sp = s.BuildServiceProvider();
        Assert.Same(fakeEngine,   sp.GetRequiredService<IInferenceEngine>());
        Assert.Same(fakeMetrics,  sp.GetRequiredService<ServerMetrics>());
        Assert.Same(fakeRenderer, sp.GetRequiredService<ChatTemplateRenderer>());
    }

    [Fact]
    public void AddSharpInference_EngineFactoryOption_BypassesModelPath()
    {
        // Caller-supplied factory is the canonical escape hatch when there's no GGUF on disk
        // (tests, alt loaders). It runs lazily on first IInferenceEngine resolve.
        var fakeEngine = new FakeInferenceEngine("from-factory");
        var jinja = new JinjaChatTemplate("J: {{ messages[0].content }}");

        var s = new ServiceCollection();
        s.AddSharpInference(opts =>
        {
            opts.EngineFactory = _ => new LoadedEngine(fakeEngine, "phi3", jinja);
        });

        var sp = s.BuildServiceProvider();
        var engine = sp.GetRequiredService<IInferenceEngine>();
        Assert.Same(fakeEngine, engine);

        // Resolving the engine should also have reconfigured the renderer with the
        // loaded model's metadata.
        var renderer = sp.GetRequiredService<ChatTemplateRenderer>();
        Assert.Equal("phi3", renderer.Architecture);
        Assert.Same(jinja, renderer.JinjaTemplate);
    }

    [Fact]
    public void AddSharpInference_EngineFactory_NotInvokedUntilEngineResolved()
    {
        // Lazy registration: if no one resolves IInferenceEngine, the factory never runs.
        int invocationCount = 0;
        var s = new ServiceCollection();
        s.AddSharpInference(opts =>
        {
            opts.EngineFactory = _ =>
            {
                invocationCount++;
                return new LoadedEngine(new FakeInferenceEngine("x"), "qwen2", null);
            };
        });

        using var sp = s.BuildServiceProvider();
        Assert.Equal(0, invocationCount);

        _ = sp.GetRequiredService<ChatTemplateRenderer>();
        Assert.Equal(0, invocationCount);  // renderer must not transitively trigger engine load

        _ = sp.GetRequiredService<IInferenceEngine>();
        Assert.Equal(1, invocationCount);

        _ = sp.GetRequiredService<IInferenceEngine>();
        Assert.Equal(1, invocationCount);  // singleton — second resolve hits the cache
    }

    [Fact]
    public void AddSharpInference_RegistersJsonSourceGenContextAtIndexZero()
    {
        // Endpoints serialize via SharpInferenceJsonContext.Default; this guards that the
        // context is the FIRST type resolver in the chain (otherwise an upstream resolver
        // could intercept a request/response type and break AOT publishing).
        var s = new ServiceCollection();
        s.AddOptions(); // needed for IOptions<JsonOptions> below
        s.AddSharpInference();
        var sp = s.BuildServiceProvider();

        var jsonOpts = sp.GetRequiredService<IOptions<JsonOptions>>().Value;
        Assert.NotEmpty(jsonOpts.SerializerOptions.TypeInfoResolverChain);
        Assert.Same(SharpInferenceJsonContext.Default, jsonOpts.SerializerOptions.TypeInfoResolverChain[0]);
    }

    [Fact]
    public void AddSharpInference_WithConfiguration_BindsSharpInferenceSection()
    {
        // The IConfiguration overload binds the "SharpInference" section onto the options.
        var cfg = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["SharpInference:ModelPath"]     = "/tmp/foo.gguf",
                ["SharpInference:MaxBatchSize"]  = "16",
                ["SharpInference:Architecture"]  = "llama",
            })
            .Build();

        var s = new ServiceCollection();
        s.AddSharpInference(cfg);

        var sp = s.BuildServiceProvider();
        var opts = sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;
        Assert.Equal("/tmp/foo.gguf", opts.ModelPath);
        Assert.Equal(16, opts.MaxBatchSize);
        Assert.Equal("llama", opts.Architecture);
    }

    [Fact]
    public void AddSharpInference_InlineConfigure_OverridesConfigurationValues()
    {
        // The Action<Options> callback runs AFTER the IConfiguration bind, so inline
        // tweaks win over appsettings/env-var values — important for tests that need
        // to override one knob without rewriting the whole config tree.
        var cfg = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["SharpInference:MaxBatchSize"] = "2",
                ["SharpInference:Architecture"] = "llama",
            })
            .Build();

        var s = new ServiceCollection();
        s.AddSharpInference(cfg, opts => opts.MaxBatchSize = 32);

        var sp = s.BuildServiceProvider();
        var opts = sp.GetRequiredService<IOptions<SharpInferenceServerOptions>>().Value;
        Assert.Equal(32, opts.MaxBatchSize);
        Assert.Equal("llama", opts.Architecture);  // not overridden — config value preserved
    }
}

/// <summary>
/// End-to-end smoke test for <see cref="EndpointRouteBuilderExtensions.MapSharpInference"/>:
/// boots a minimal in-memory host and verifies every route the composite extension is supposed
/// to map actually responds. Uses TestServer so we don't bind a real port.
/// </summary>
public sealed class EndpointRouteBuilderExtensionsTests
{
    [Fact]
    public async Task MapSharpInference_RegistersAllEndpointGroups()
    {
        using var host = await new HostBuilder()
            .ConfigureWebHost(b => b
                .UseTestServer()
                .ConfigureServices(s =>
                {
                    s.AddRouting();
                    s.AddSharpInference();
                    // Override IInferenceEngine so MapSharpInference() doesn't trigger a model load.
                    s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("ut-model"));
                })
                .Configure(app =>
                {
                    app.UseRouting();
                    app.UseEndpoints(e => e.MapSharpInference());
                }))
            .StartAsync();

        var client = host.GetTestClient();

        // OpenAI surface
        Assert.Equal(System.Net.HttpStatusCode.OK,         (await client.GetAsync("/v1/models")).StatusCode);
        // Anthropic surface — POST with empty body returns 400 (missing messages), which is
        // proof the route is mapped (a missing route would return 404).
        var anthropic = await client.PostAsync("/v1/messages", new StringContent("{}", System.Text.Encoding.UTF8, "application/json"));
        Assert.Equal(System.Net.HttpStatusCode.BadRequest, anthropic.StatusCode);
        // Responses surface
        var responses = await client.PostAsync("/v1/responses", new StringContent("{\"input\":\"hi\"}", System.Text.Encoding.UTF8, "application/json"));
        Assert.Equal(System.Net.HttpStatusCode.OK,         responses.StatusCode);
        // Observability surface
        Assert.Equal(System.Net.HttpStatusCode.OK,         (await client.GetAsync("/health")).StatusCode);
        Assert.Equal(System.Net.HttpStatusCode.OK,         (await client.GetAsync("/metrics")).StatusCode);
    }

    [Fact]
    public void MapSharpInference_ReturnsTheSameBuilder_ForChaining()
    {
        // Fluent return — common ASP.NET Core idiom: app.MapSharpInference().MapHub<...>().
        var builder = WebApplication.CreateSlimBuilder();
        builder.Services.AddSharpInference();
        builder.Services.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("x"));
        var app = builder.Build();

        IEndpointRouteBuilder ret = app.MapSharpInference();
        Assert.Same(app, ret);
    }
}

/// <summary>Public-surface sanity check on <see cref="SharpInferenceJsonContext"/>.</summary>
public sealed class SharpInferenceJsonContextTests
{
    [Fact]
    public void Default_CoversCoreRequestAndResponseTypes()
    {
        // Source-generated context must include every type the endpoints (de)serialize.
        // The Default property gives access to the generated TypeInfo instances; null
        // means the type wasn't registered and AOT-publishing would fail at runtime.
        var ctx = SharpInferenceJsonContext.Default;
        Assert.NotNull(ctx.ChatCompletionRequest);
        Assert.NotNull(ctx.ChatCompletionResponse);
        Assert.NotNull(ctx.ChatCompletionChunk);
        Assert.NotNull(ctx.AnthropicMessageRequest);
        Assert.NotNull(ctx.AnthropicMessageResponse);
        Assert.NotNull(ctx.ResponsesRequest);
        Assert.NotNull(ctx.RespObject);
        Assert.NotNull(ctx.HealthStatus);
    }

    [Fact]
    public void Default_UsesSnakeCaseLowerNaming()
    {
        // The wire format MUST be snake_case (OpenAI/Anthropic conventions) — guard against
        // anyone flipping JsonKnownNamingPolicy by accident.
        var req = new ChatCompletionRequest(
            Model: "m", Messages: null, MaxTokens: 5, Temperature: 0.7f, TopP: null,
            Stream: null, LogitBias: null, ResponseFormat: null);
        var json = JsonSerializer.Serialize(req, SharpInferenceJsonContext.Default.ChatCompletionRequest);
        Assert.Contains("\"max_tokens\":5", json);
        Assert.Contains("\"temperature\":0.7", json);
    }

    [Fact]
    public void Default_OmitsNullFieldsWhenWriting()
    {
        // WhenWritingNull is what makes the Anthropic AContent block (one record, optional
        // text/thinking/tool_use fields) emit a clean per-type shape. Lose this setting and
        // the wire format breaks.
        var content = new AContent("text", Text: "hi");
        var json = JsonSerializer.Serialize(content, SharpInferenceJsonContext.Default.AContent);
        Assert.Contains("\"text\":\"hi\"", json);
        Assert.DoesNotContain("\"thinking\"", json);
        Assert.DoesNotContain("\"signature\"", json);
        Assert.DoesNotContain("\"input\"", json);
    }
}
