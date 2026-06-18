using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// Issue #109: when <see cref="SharpInferenceServerOptions.MaxConcurrentRequests"/> is set, the
/// server fast-rejects overlapping generation requests with HTTP 429 instead of silently
/// serializing them on the single-user engine (which an agentic client reads as a hang). When
/// unset, the legacy passthrough behaviour is preserved.
/// </summary>
public sealed class ConcurrencyLimitTests
{
    private static HttpClient CreateClient(FakeInferenceEngine fake, int? maxConcurrent) =>
        new WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.ConfigureServices(s =>
            {
                s.Configure<SharpInferenceServerOptions>(o => o.MaxConcurrentRequests = maxConcurrent);
                s.AddSingleton<IInferenceEngine>(fake);
            }))
            .CreateClient();

    private static object ChatRequest() => new
    {
        model = "test-model",
        messages = new[] { new { role = "user", content = "hi" } },
        max_tokens = 5,
        stream = false,
    };

    [Fact]
    public async Task MaxConcurrent1_OverlappingRequest_Rejected429()
    {
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: 1);

        // Request A enters the engine and blocks (holding the single admission slot).
        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "Request A never reached the engine.");

        // Request B overlaps A → must be fast-rejected with 429 (it never reaches the engine).
        var bResp = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.Equal(HttpStatusCode.TooManyRequests, bResp.StatusCode);
        Assert.True(bResp.Headers.Contains("Retry-After"));
        var body = await bResp.Content.ReadAsStringAsync();
        Assert.Contains("SHARPI_MAX_BATCH", body);

        // Release A; it completes normally, and the slot frees for a subsequent request.
        fake.Hold!.SetResult();
        var aResp = await aTask;
        Assert.Equal(HttpStatusCode.OK, aResp.StatusCode);

        var cResp = await client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.Equal(HttpStatusCode.OK, cResp.StatusCode);
    }

    [Fact]
    public async Task Unset_OverlappingRequests_BothSucceed_NoRejection()
    {
        // Default (no limit): the gate is a passthrough — overlapping requests are not rejected.
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: null);

        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        var bTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "No request reached the engine.");

        // Both requests are in flight against the (un-gated) engine; release and confirm neither
        // was rejected.
        fake.Hold!.SetResult();
        var responses = await Task.WhenAll(aTask, bTask);
        Assert.All(responses, r => Assert.Equal(HttpStatusCode.OK, r.StatusCode));
    }

    [Fact]
    public async Task MaxConcurrent1_NonGenerationEndpoints_NotGated()
    {
        // /v1/models and /health must stay reachable even while a generation request holds the slot.
        var fake = new FakeInferenceEngine("test-model") { Hold = new TaskCompletionSource() };
        var client = CreateClient(fake, maxConcurrent: 1);

        var aTask = client.PostAsJsonAsync("/v1/chat/completions", ChatRequest());
        Assert.True(fake.Entered.Wait(TimeSpan.FromSeconds(5)), "Request A never reached the engine.");

        var models = await client.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.OK, models.StatusCode);
        var health = await client.GetAsync("/health");
        Assert.Equal(HttpStatusCode.OK, health.StatusCode);

        fake.Hold!.SetResult();
        Assert.Equal(HttpStatusCode.OK, (await aTask).StatusCode);
    }
}
