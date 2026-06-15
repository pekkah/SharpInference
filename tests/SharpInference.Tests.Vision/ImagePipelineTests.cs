using SharpInference.Diffusion;
using SharpInference.Vision;

namespace SharpInference.Tests.Vision;

public class ImagePipelineTests
{
    [Fact]
    public void PngRoundTrip_RecoversPixels()
    {
        // Build a CHW float image whose per-channel values map to exact bytes
        // (PngWriter does clamp(int(v*255+0.5)); v=b/255 -> b), so decode is lossless.
        const int w = 7, h = 5;
        var chw = new float[3 * h * w];
        int plane = h * w;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = y * w + x;
                chw[i] = ((x * 37 + y * 11) % 256) / 255f;          // R
                chw[plane + i] = ((x * 5 + y * 53) % 256) / 255f;   // G
                chw[2 * plane + i] = ((x * 97 + y * 3) % 256) / 255f; // B
            }

        string tmp = Path.Combine(Path.GetTempPath(), $"sharpi_png_{Guid.NewGuid():N}.png");
        try
        {
            PngWriter.Write(tmp, chw, w, h);
            var rgb = ImageIO.LoadRgb(tmp, out int dw, out int dh);
            Assert.Equal(w, dw);
            Assert.Equal(h, dh);
            Assert.Equal(w * h * 3, rgb.Length);
            for (int i = 0; i < plane; i++)
            {
                Assert.Equal((byte)Math.Round(chw[i] * 255f), rgb[i * 3]);
                Assert.Equal((byte)Math.Round(chw[plane + i] * 255f), rgb[i * 3 + 1]);
                Assert.Equal((byte)Math.Round(chw[2 * plane + i] * 255f), rgb[i * 3 + 2]);
            }
        }
        finally { if (File.Exists(tmp)) File.Delete(tmp); }
    }

    [Theory]
    [InlineData(224, 224)]   // small square -> upscaled to fit min tokens
    [InlineData(1000, 200)]  // wide
    [InlineData(4000, 3000)] // large -> downscaled under max tokens
    [InlineData(33, 800)]    // extreme aspect
    public void CalcSizePreservedRatio_StaysAlignedAndInBudget(int w, int h)
    {
        const int align = 48;
        long min = 40L * align * align;   // gemma4uv set_limit_image_tokens(40, 280)
        long max = 280L * align * align;

        var (ow, oh) = ImagePreprocessor.CalcSizePreservedRatio(w, h, align, min, max);

        Assert.Equal(0, ow % align);
        Assert.Equal(0, oh % align);
        Assert.True(ow >= align && oh >= align);
        int tokens = (ow / align) * (oh / align);
        // Rounding to multiples of 48 can nudge slightly past the nominal bounds; allow a small margin.
        Assert.InRange(tokens, 36, 320);
    }

    [Fact]
    public void Preprocess_Then_Embed_ProducesFiniteSoftTokens()
    {
        var mmproj = VisionTestPaths.FindMmproj();
        if (mmproj is null) return;

        // Synthesize a 200x150 RGB gradient, encode to PNG, then run the full pipeline.
        const int sw = 200, sh = 150;
        var rgb = new byte[sw * sh * 3];
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < sw; x++)
            {
                int i = (y * sw + x) * 3;
                rgb[i] = (byte)(x * 255 / sw);
                rgb[i + 1] = (byte)(y * 255 / sh);
                rgb[i + 2] = (byte)((x + y) % 256);
            }

        using var m = VisionModel.Open(mmproj);
        var pre = ImagePreprocessor.Preprocess(rgb, sw, sh, m);
        Assert.Equal(0, pre.Width % m.PatchSize);
        Assert.Equal(0, pre.Height % m.PatchSize);
        Assert.Equal(3 * pre.Width * pre.Height, pre.Chw.Length);

        var embedder = new GemmaUvVisionEmbedder(m);
        var soft = embedder.Forward(pre.Chw, pre.Height, pre.Width, out int nTok);

        Assert.Equal((pre.Width / m.PatchSize) * (pre.Height / m.PatchSize), nTok);
        Assert.InRange(nTok, m.MinImageTokens, m.MaxImageTokens);
        Assert.Equal(nTok * m.EmbeddingLength, soft.Length);
        foreach (var v in soft) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Forward_RejectsOversizedUnpreprocessedImage()
    {
        var mmproj = VisionTestPaths.FindMmproj();
        if (mmproj is null) return;
        using var m = VisionModel.Open(mmproj);
        var embedder = new GemmaUvVisionEmbedder(m);

        // A grid larger than the position table (1120 per axis) must throw a clear error,
        // not an opaque out-of-range from the position lookup.
        int big = (1121) * m.PatchSize; // 1121 patches on one axis
        var chw = new float[3 * m.PatchSize * big];
        var ex = Assert.Throws<ArgumentException>(() => embedder.Forward(chw, m.PatchSize, big, out _));
        Assert.Contains("position-table", ex.Message);
    }

    [Fact]
    public void LoadRgb_BadSignature_Throws()
    {
        using var ms = new MemoryStream([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        Assert.Throws<InvalidDataException>(() => ImageIO.LoadRgb(ms, out _, out _));
    }

    [Fact]
    public void LoadRgb_NonPositiveDimensions_Throws()
    {
        // Valid signature + an IHDR declaring width=0 (a hostile/corrupt PNG). The decoder
        // doesn't verify the CRC, so this exercises the dimension guard, not the CRC. It must
        // throw InvalidDataException (inside the CLI's catch filter), not a raw alloc failure.
        using var ms = new MemoryStream();
        ms.Write([137, 80, 78, 71, 13, 10, 26, 10]);   // PNG signature
        WriteBE(ms, 13);                               // IHDR length
        ms.Write("IHDR"u8.ToArray());
        WriteBE(ms, 0);                                // width = 0 (invalid)
        WriteBE(ms, 1);                                // height
        ms.WriteByte(8); ms.WriteByte(2);              // bit depth, color type (RGB)
        ms.WriteByte(0); ms.WriteByte(0); ms.WriteByte(0); // compression, filter, interlace
        WriteBE(ms, 0);                                // CRC (unchecked)
        ms.Position = 0;

        Assert.Throws<InvalidDataException>(() => ImageIO.LoadRgb(ms, out _, out _));

        static void WriteBE(Stream s, int v)
        {
            s.WriteByte((byte)(v >> 24)); s.WriteByte((byte)(v >> 16));
            s.WriteByte((byte)(v >> 8)); s.WriteByte((byte)v);
        }
    }

    [Fact]
    public void LoadRgb_HugeInflatingIdat_DecodesToDeclaredSizeOnly()
    {
        // Decompression-bomb guard (#259 review): a 1x1 PNG whose IDAT inflates to ~64 MB of
        // zeros must decode to exactly the declared 1x1 pixel, reading only the `expected` bytes
        // (4 = one filter byte + one RGB triple). Without the bound the decoder would materialize
        // the full 64 MB. The decode must succeed (not throw) and return the declared size.
        byte[] idat = ZlibCompress(new byte[64 * 1024 * 1024]); // 64 MB of zeros -> tiny compressed
        Assert.True(idat.Length < 100_000, "zlib bomb payload should be small");
        using var ms = new MemoryStream(BuildPng(1, 1, colorType: 2, idat));

        byte[] rgb = ImageIO.LoadRgb(ms, out int w, out int h);

        Assert.Equal(1, w);
        Assert.Equal(1, h);
        Assert.Equal(3, rgb.Length);            // 1x1 RGB
        Assert.Equal(new byte[] { 0, 0, 0 }, rgb);
    }

    [Fact]
    public void LoadRgb_IdatShorterThanDeclared_Throws()
    {
        // IHDR declares 4x4 RGB (expected raw = 4*(1+4*3) = 52 bytes) but the IDAT only inflates
        // to 10 bytes — the bounded read returns fewer than `expected`, which must surface as the
        // "too short" InvalidDataException rather than reading past the buffer.
        byte[] idat = ZlibCompress(new byte[10]);
        using var ms = new MemoryStream(BuildPng(4, 4, colorType: 2, idat));

        Assert.Throws<InvalidDataException>(() => ImageIO.LoadRgb(ms, out _, out _));
    }

    private static byte[] ZlibCompress(byte[] data)
    {
        using var outMs = new MemoryStream();
        using (var z = new System.IO.Compression.ZLibStream(outMs, System.IO.Compression.CompressionLevel.SmallestSize, leaveOpen: true))
            z.Write(data, 0, data.Length);
        return outMs.ToArray();
    }

    // Builds a minimal PNG (signature + IHDR + IDAT + IEND); CRCs are zero since the decoder
    // does not verify them. `idat` is the already-zlib-compressed image data.
    private static byte[] BuildPng(int w, int h, int colorType, byte[] idat)
    {
        using var ms = new MemoryStream();
        ms.Write([137, 80, 78, 71, 13, 10, 26, 10]);
        WriteBE(ms, 13); ms.Write("IHDR"u8.ToArray());
        WriteBE(ms, w); WriteBE(ms, h);
        ms.WriteByte(8); ms.WriteByte((byte)colorType);
        ms.WriteByte(0); ms.WriteByte(0); ms.WriteByte(0);
        WriteBE(ms, 0); // IHDR CRC (unchecked)
        WriteBE(ms, idat.Length); ms.Write("IDAT"u8.ToArray()); ms.Write(idat); WriteBE(ms, 0);
        WriteBE(ms, 0); ms.Write("IEND"u8.ToArray()); WriteBE(ms, 0);
        return ms.ToArray();

        static void WriteBE(Stream s, int v)
        {
            s.WriteByte((byte)(v >> 24)); s.WriteByte((byte)(v >> 16));
            s.WriteByte((byte)(v >> 8)); s.WriteByte((byte)v);
        }
    }
}
