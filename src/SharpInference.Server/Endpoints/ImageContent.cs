using SharpInference.Engine;
using SharpInference.Vision;

namespace SharpInference.Server.Endpoints;

/// <summary>
/// Shared parsing for multimodal image content blocks (issue #253): decodes base64 / data-URL
/// image payloads to raw bytes and validates the format. The Gemma 4 vision decoder (ImageIO)
/// supports PNG only, so non-PNG payloads are rejected here with a clear error rather than
/// failing deep inside the engine's prefill.
/// </summary>
internal static class ImageContent
{
    /// <summary>
    /// The model's image-placeholder token text, inserted into the message stream at each image
    /// position. The chat template passes it through verbatim; the engine tokenizes it to the
    /// <c>&lt;|image|&gt;</c> id and expands it to the projected soft tokens during prefill.
    /// </summary>
    public const string Placeholder = "<|image|>";

    /// <summary>
    /// Upper bound on image content blocks per request. A decoded image is bounded by ImageIO's
    /// 64M-pixel guard (~hundreds of MB), so without a per-request cap a single request could
    /// fan out into many large allocations. The chat endpoints reject requests that exceed this.
    /// </summary>
    public const int MaxImagesPerRequest = 16;

    /// <summary>
    /// Throws if the running image count has reached the per-request cap. Called BEFORE decoding
    /// the next image so an over-cap request is rejected without paying to decode every block in it.
    /// </summary>
    public static void CheckCap(int currentImageCount)
    {
        if (currentImageCount >= MaxImagesPerRequest)
            throw new ImageContentException(
                $"too many images in one request (max {MaxImagesPerRequest}).");
    }

    /// <summary>
    /// Routes to the image-aware generate when images are present, else the text path (#253).
    /// Shared by the OpenAI and Anthropic endpoints — pure engine-level dispatch with no
    /// wire-format dependency, so it lives here rather than being copied into each endpoint.
    /// </summary>
    public static IAsyncEnumerable<GenerateChunk> Generate(
        IInferenceEngine engine, string prompt, string? canonical, IReadOnlyList<byte[]> images,
        SamplingParams sp, CancellationToken ct)
        => images.Count > 0
            ? engine.GenerateImageChunksAsync(prompt, images, sp, ct)
            : engine.GenerateChunksAsync(prompt, sp, ct, canonical);

    /// <summary>Decode a base64 image payload to bytes and validate it is a PNG.</summary>
    public static byte[] FromBase64(string base64)
    {
        byte[] bytes;
        try { bytes = Convert.FromBase64String(base64.Trim()); }
        catch (FormatException) { throw new ImageContentException("image data is not valid base64."); }
        ValidatePng(bytes);
        return bytes;
    }

    /// <summary>Decode an OpenAI <c>image_url</c> that is a base64 data URL
    /// (<c>data:image/png;base64,...</c>). Remote URL fetching is not supported.</summary>
    public static byte[] FromDataUrl(string url)
    {
        const string prefix = "data:";
        if (!url.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new ImageContentException(
                "image_url must be a base64 data URL (data:image/png;base64,...); remote URL fetching is not supported.");
        int comma = url.IndexOf(',');
        if (comma < 0)
            throw new ImageContentException("malformed data URL (missing comma).");
        string meta = url[prefix.Length..comma];
        if (!meta.Contains("base64", StringComparison.OrdinalIgnoreCase))
            throw new ImageContentException("only base64-encoded data URLs are supported.");
        return FromBase64(url[(comma + 1)..]);
    }

    private static void ValidatePng(byte[] bytes)
    {
        // Fully decode the payload here, at request-parse time, rather than only checking the
        // 8-byte signature. ImageIO supports 8-bit non-interlaced PNG (grayscale/RGB/RGBA) only;
        // a signature-valid but interlaced / 16-bit / palette / truncated / decompression-bomb
        // payload would otherwise slip past a signature-only check and fail deep inside the
        // engine's prefill — on the streaming path that lands AFTER the 200 response has begun,
        // so it can only abort the connection. Decoding up front turns every such case into a
        // clean HTTP 400. The decode is bounded (ImageIO caps inflation to the declared pixel
        // size) and the result is discarded; the engine re-decodes the same bytes during prefill.
        try
        {
            _ = ImageIO.LoadRgb(new MemoryStream(bytes, writable: false), out _, out _);
        }
        // ImageIO throws InvalidDataException/NotSupportedException by contract, but a crafted
        // payload can still trip an IndexOutOfRange/Argument/InvalidOperation deep in the decoder.
        // Map every decode failure to a clean 400 — only truly fatal conditions (OOM, etc.) escape.
        catch (Exception ex) when (ex is not OutOfMemoryException and not StackOverflowException)
        {
            throw new ImageContentException($"unsupported or corrupt image: {ex.Message}");
        }
    }
}

/// <summary>Raised on malformed / unsupported image content; endpoints map it to HTTP 400.</summary>
internal sealed class ImageContentException(string message) : Exception(message);
