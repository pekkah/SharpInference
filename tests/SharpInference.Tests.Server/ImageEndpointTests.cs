using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Engine;
using SharpInference.Server;

namespace SharpInference.Tests.Server;

/// <summary>
/// Wire-format tests for multimodal image input (issue #253). They exercise the Anthropic
/// <c>image</c> content-block and OpenAI <c>image_url</c> part parsing, the dispatch to the
/// engine's image path, and the rejection cases — all against a <see cref="FakeInferenceEngine"/>
/// so no model or mmproj is required. (The fake never decodes the bytes; the endpoint's
/// PNG-signature validation is what the bad-format test exercises.)
/// </summary>
public sealed class ImageEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public ImageEndpointTests(WebApplicationFactory<Program> factory) => _factory = factory;

    // A real, fully decodable 2x2 RGB PNG. The endpoint now decode-validates images at parse
    // time (so format errors become a clean 400 before generation), so the routing tests need a
    // genuinely decodable payload rather than a bare signature.
    private const string RealPng2x2Base64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR42mM4IScHRAwQCgAfJgQRSo6NIAAAAABJRU5ErkJggg==";

    private static string FakePngBase64() => RealPng2x2Base64;

    // Valid 8-byte PNG signature but no IHDR — passes a signature-only check, fails a real decode.
    private static string SignatureOnlyPngBase64()
        => Convert.ToBase64String([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 1, 2, 3, 4]);

    private HttpClient ClientWith(FakeInferenceEngine engine)
        => _factory.WithWebHostBuilder(b => b.ConfigureServices(s =>
            s.AddSingleton<IInferenceEngine>(engine))).CreateClient();

    [Fact]
    public async Task Anthropic_ImageBlock_RoutesToImagePathWithPlaceholder()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "text", text = "What is in this image?" },
                    new { type = "image", source = new { type = "base64", media_type = "image/png", data = FakePngBase64() } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/messages", req);

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Equal(1, fake.LastImageCount);
        Assert.Contains("<|image|>", fake.LastPrompt!);
        Assert.Contains("What is in this image?", fake.LastPrompt!);
    }

    [Fact]
    public async Task OpenAi_ImageUrlDataUrl_RoutesToImagePathWithPlaceholder()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var dataUrl = "data:image/png;base64," + FakePngBase64();
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "text", text = "Describe:" },
                    new { type = "image_url", image_url = new { url = dataUrl } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Equal(1, fake.LastImageCount);
        Assert.Contains("<|image|>", fake.LastPrompt!);
    }

    [Fact]
    public async Task OpenAi_MultipleImages_AllCounted()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var dataUrl = "data:image/png;base64," + FakePngBase64();
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "text", text = "Compare" },
                    new { type = "image_url", image_url = new { url = dataUrl } },
                    new { type = "text", text = "and" },
                    new { type = "image_url", image_url = new { url = dataUrl } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Equal(2, fake.LastImageCount);
    }

    [Fact]
    public async Task OpenAi_PlainStringContent_StillWorks()
    {
        // The OaiMessage.Content type changed from string? to JsonElement?; a plain-string
        // content must still deserialize and route to the text path (back-compat).
        var fake = new FakeInferenceEngine("m");
        var client = ClientWith(fake);
        var req = new { model = "m", max_tokens = 8, messages = new object[] { new { role = "user", content = "hello" } } };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
        Assert.Equal(0, fake.LastImageCount);
        Assert.Contains("hello", fake.LastPrompt!);
    }

    [Fact]
    public async Task Anthropic_ImageOnNonImageModel_Returns400()
    {
        var fake = new FakeInferenceEngine("text-only"); // SupportsImages = false
        var client = ClientWith(fake);
        var req = new
        {
            model = "text-only",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "image", source = new { type = "base64", media_type = "image/png", data = FakePngBase64() } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/messages", req);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("image input", await resp.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Anthropic_NonPngImage_Returns400()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var jpegBase64 = Convert.ToBase64String([0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4]); // JPEG magic, not PNG
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "image", source = new { type = "base64", media_type = "image/jpeg", data = jpegBase64 } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/messages", req);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("PNG", await resp.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task OpenAi_RemoteImageUrl_Returns400()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "image_url", image_url = new { url = "https://example.com/cat.png" } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("data URL", await resp.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Anthropic_SignatureValidButUndecodablePng_Returns400()
    {
        // A payload with a valid 8-byte PNG signature but no decodable body must be rejected with
        // a clean 400 at parse time, not surface as a 500 deep in the engine's prefill (#259 review).
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[]
            {
                new { role = "user", content = new object[]
                {
                    new { type = "image", source = new { type = "base64", media_type = "image/png", data = SignatureOnlyPngBase64() } },
                } },
            },
        };

        var resp = await client.PostAsJsonAsync("/v1/messages", req);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Equal(0, fake.LastImageCount); // never reached the engine
    }

    [Fact]
    public async Task OpenAi_TooManyImages_Returns400()
    {
        var fake = new FakeInferenceEngine("gemma-4-12b") { SupportsImages = true };
        var client = ClientWith(fake);
        var dataUrl = "data:image/png;base64," + RealPng2x2Base64;
        // 17 images (one over the cap of 16).
        var parts = new List<object> { new { type = "text", text = "Compare" } };
        for (int i = 0; i < 17; i++)
            parts.Add(new { type = "image_url", image_url = new { url = dataUrl } });
        var req = new
        {
            model = "gemma-4-12b",
            max_tokens = 8,
            messages = new object[] { new { role = "user", content = parts.ToArray() } },
        };

        var resp = await client.PostAsJsonAsync("/v1/chat/completions", req);

        Assert.Equal(HttpStatusCode.BadRequest, resp.StatusCode);
        Assert.Contains("too many images", await resp.Content.ReadAsStringAsync());
        Assert.Equal(0, fake.LastImageCount);
    }
}
