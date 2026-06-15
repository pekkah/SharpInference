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
        // PNG 8-byte signature. Validating the magic (rather than the declared media_type) catches
        // mislabelled or unsupported (JPEG/WebP) payloads regardless of the client's content type.
        ReadOnlySpan<byte> sig = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        if (bytes.Length < 8 || !bytes.AsSpan(0, 8).SequenceEqual(sig))
            throw new ImageContentException(
                "only PNG images are supported (the decoded image data is not a PNG).");
    }
}

/// <summary>Raised on malformed / unsupported image content; endpoints map it to HTTP 400.</summary>
internal sealed class ImageContentException(string message) : Exception(message);
