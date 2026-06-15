using System.IO.Compression;

namespace SharpInference.Vision;

/// <summary>
/// Minimal PNG decoder — the read counterpart to SharpInference.Diffusion.PngWriter.
/// Supports 8-bit non-interlaced PNGs in grayscale (0), RGB (2), and RGBA (6) color
/// types, with all five scanline filters (None/Sub/Up/Average/Paeth). Depends only on
/// System.IO.Compression (NativeAOT-compatible); no imaging library.
///
/// Returned buffer is interleaved RGB, 8-bit, HWC order: <c>rgb[(y*width + x)*3 + c]</c>.
/// </summary>
public static class ImageIO
{
    private static readonly byte[] Signature = [137, 80, 78, 71, 13, 10, 26, 10];

    /// <summary>Decode a PNG file to interleaved 8-bit RGB (HWC). Throws on unsupported variants.</summary>
    public static byte[] LoadRgb(string path, out int width, out int height)
    {
        using var fs = File.OpenRead(path);
        return LoadRgb(fs, out width, out height);
    }

    /// <summary>Decode a PNG stream to interleaved 8-bit RGB (HWC).</summary>
    public static byte[] LoadRgb(Stream input, out int width, out int height)
    {
        using var ms = new MemoryStream();
        input.CopyTo(ms);
        byte[] png = ms.GetBuffer();
        int len = (int)ms.Length;

        for (int i = 0; i < Signature.Length; i++)
            if (i >= len || png[i] != Signature[i])
                throw new InvalidDataException("Not a PNG file (bad signature).");

        int pos = 8;
        int w = 0, h = 0, bitDepth = 0, colorType = 0, interlace = 0;
        bool seenIhdr = false;
        using var idat = new MemoryStream();

        while (pos + 8 <= len)
        {
            int clen = ReadUInt32BE(png, pos);
            string type = System.Text.Encoding.ASCII.GetString(png, pos + 4, 4);
            int dataOff = pos + 8;
            // ReadUInt32BE returns a signed int; a high-bit-set length is negative and would
            // wrap past the bounds check, so reject it explicitly (malformed/hostile PNG).
            if (clen < 0 || (long)dataOff + clen + 4 > len)
                throw new InvalidDataException($"Truncated PNG chunk '{type}'.");

            switch (type)
            {
                case "IHDR":
                    w = ReadUInt32BE(png, dataOff);
                    h = ReadUInt32BE(png, dataOff + 4);
                    bitDepth = png[dataOff + 8];
                    colorType = png[dataOff + 9];
                    interlace = png[dataOff + 12];
                    seenIhdr = true;
                    break;
                case "IDAT":
                    idat.Write(png, dataOff, clen);
                    break;
                case "IEND":
                    pos = len; // stop
                    break;
            }
            if (type == "IEND") break;
            pos = dataOff + clen + 4; // skip data + CRC
        }

        if (!seenIhdr) throw new InvalidDataException("PNG missing IHDR.");
        // IHDR width/height come through ReadUInt32BE (signed) too; reject non-positive or
        // absurd dimensions before they reach the pixel-buffer allocation below.
        if (w <= 0 || h <= 0 || (long)w * h > 64_000_000)
            throw new InvalidDataException($"PNG dimensions out of range: {w}x{h}.");
        if (bitDepth != 8) throw new NotSupportedException($"Only 8-bit PNGs supported (got {bitDepth}-bit).");
        if (interlace != 0) throw new NotSupportedException("Interlaced PNGs are not supported.");
        int channels = colorType switch
        {
            0 => 1, // grayscale
            2 => 3, // RGB
            6 => 4, // RGBA
            _ => throw new NotSupportedException($"PNG color type {colorType} not supported (use grayscale/RGB/RGBA).")
        };

        // Inflate the zlib stream (handles header + adler).
        idat.Position = 0;
        byte[] raw;
        using (var zlib = new ZLibStream(idat, CompressionMode.Decompress))
        using (var outMs = new MemoryStream(h * (1 + w * channels)))
        {
            zlib.CopyTo(outMs);
            raw = outMs.ToArray();
        }

        int stride = w * channels;
        int expected = h * (1 + stride);
        if (raw.Length < expected)
            throw new InvalidDataException($"PNG data too short: {raw.Length} < {expected}.");

        // Unfilter scanlines in place into a contiguous pixel buffer.
        var pixels = new byte[h * stride];
        for (int y = 0; y < h; y++)
        {
            int filter = raw[y * (1 + stride)];
            int srcRow = y * (1 + stride) + 1;
            int dstRow = y * stride;
            int dstPrev = dstRow - stride;
            for (int x = 0; x < stride; x++)
            {
                int rawVal = raw[srcRow + x];
                int a = x >= channels ? pixels[dstRow + x - channels] : 0;          // left
                int b = y > 0 ? pixels[dstPrev + x] : 0;                            // up
                int c = (x >= channels && y > 0) ? pixels[dstPrev + x - channels] : 0; // up-left
                int val = filter switch
                {
                    0 => rawVal,
                    1 => rawVal + a,
                    2 => rawVal + b,
                    3 => rawVal + (a + b) / 2,
                    4 => rawVal + Paeth(a, b, c),
                    _ => throw new InvalidDataException($"Unknown PNG filter {filter}.")
                };
                pixels[dstRow + x] = (byte)val;
            }
        }

        // Expand to interleaved RGB.
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++)
        {
            int s = i * channels;
            if (channels == 1) { rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = pixels[s]; }
            else { rgb[i * 3] = pixels[s]; rgb[i * 3 + 1] = pixels[s + 1]; rgb[i * 3 + 2] = pixels[s + 2]; }
        }

        width = w; height = h;
        return rgb;
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        return pb <= pc ? b : c;
    }

    private static int ReadUInt32BE(byte[] buf, int off) =>
        (buf[off] << 24) | (buf[off + 1] << 16) | (buf[off + 2] << 8) | buf[off + 3];
}
