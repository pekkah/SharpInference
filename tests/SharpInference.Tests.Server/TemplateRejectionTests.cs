using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// A chat template that rejects its own message list — via Jinja <c>raise_exception</c> — must
/// surface as HTTP 400, not 500. Several families guard their conversation shape this way; Mistral
/// v3 refuses a history whose roles don't alternate. That is the caller's input being wrong, so
/// the client needs a diagnosable error rather than a bodyless server fault.
/// </summary>
public sealed class TemplateRejectionTests(WebApplicationFactory<Program> factory)
    : IClassFixture<WebApplicationFactory<Program>>
{
    // The alternation guard from Mistral's v3 template, reduced to just the check.
    private const string AlternationGuardTemplate = """
        {%- set ns = namespace() %}
        {%- set ns.index = 0 %}
        {%- for message in messages %}
            {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
                {{- raise_exception("conversation roles must alternate user/assistant/user/assistant/...") }}
            {%- endif %}
            {%- set ns.index = ns.index + 1 %}
        {%- endfor %}
        {%- for message in messages %}{{- "[INST]" + message["content"] + "[/INST]" }}{%- endfor %}
        """;

    private HttpClient ClientWithGuardTemplate()
    {
        var engine = new FakeInferenceEngine("mistral");
        return factory.WithWebHostBuilder(b => b.ConfigureServices(s =>
        {
            s.AddSingleton<IInferenceEngine>(engine);
            s.AddSingleton(_ =>
            {
                var r = new ChatTemplateRenderer("llama", new JinjaChatTemplate(AlternationGuardTemplate));
                return r;
            });
        })).CreateClient();
    }

    [Fact]
    public async Task OpenAi_NonAlternatingRoles_Returns400WithTemplateMessage()
    {
        var resp = await ClientWithGuardTemplate().PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "mistral",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = "one" },
                new { role = "user", content = "two" },   // two users in a row
            },
        });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("invalid_request_error", body);
        // The template's own wording must reach the client — that is the whole point of 400
        // over an opaque 500.
        Assert.Contains("must alternate", body);
    }

    [Fact]
    public async Task Anthropic_NonAlternatingRoles_Returns400WithTemplateMessage()
    {
        var resp = await ClientWithGuardTemplate().PostAsJsonAsync("/v1/messages", new
        {
            model = "mistral",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = "one" },
                new { role = "user", content = "two" },
            },
        });

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        var body = await resp.Content.ReadAsStringAsync();
        Assert.Contains("invalid_request_error", body);
        Assert.Contains("must alternate", body);
    }

    [Fact]
    public async Task AlternatingRoles_StillSucceed()
    {
        // The guard must only fire on genuinely bad input — proof the handler didn't just
        // turn every request into a 400.
        var resp = await ClientWithGuardTemplate().PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "mistral",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = "one" },
                new { role = "assistant", content = "two" },
                new { role = "user", content = "three" },
            },
        });

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
    }
}
