using System.IO.Compression;
using System.Runtime.InteropServices;

namespace SharpInference.Diffusion;

/// <summary>
/// Minimal PNG encoder. Writes a 24-bit RGB PNG from a float32 image.
///
/// Input: float[] with values in [0, 1], layout [3, H, W] (CHW, channel-first).
/// Output: valid PNG file with a single IDAT chunk (zlib/deflate compressed).
///
/// Does not depend on any imaging library — uses only System.IO.Compression.
/// NativeAOT-compatible.
/// </summary>
public static class PngWriter
{
    /// <summary>Write a CHW float32 image [3, H, W] to a PNG file.</summary>
    public static void Write(string path, float[] image, int width, int height)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        Write(fs, image, width, height);
    }

    /// <summary>Write a CHW float32 image [3, H, W] to a stream.</summary>
    public static void Write(Stream output, float[] image, int width, int height)
    {
        // PNG signature
        output.Write(PngSignature);

        // IHDR chunk
        var ihdr = new byte[13];
        WriteUInt32BE(ihdr, 0, (uint)width);
        WriteUInt32BE(ihdr, 4, (uint)height);
        ihdr[8] = 8;   // bit depth
        ihdr[9] = 2;   // color type = RGB
        // compression=0, filter=0, interlace=0 already zero
        WriteChunk(output, "IHDR"u8, ihdr);

        // Build raw scanlines: filter byte (0x00 = None) + RGB pixels
        // Layout: for each row, one filter byte + width * 3 bytes
        var raw = new byte[height * (1 + width * 3)];
        int plane = height * width;
        for (int y = 0; y < height; y++)
        {
            int rowOff = y * (1 + width * 3);
            raw[rowOff] = 0x00; // filter type None
            for (int x = 0; x < width; x++)
            {
                int pixIdx = y * width + x;
                raw[rowOff + 1 + x * 3]     = FloatToByte(image[pixIdx]);             // R
                raw[rowOff + 1 + x * 3 + 1] = FloatToByte(image[plane + pixIdx]);     // G
                raw[rowOff + 1 + x * 3 + 2] = FloatToByte(image[2 * plane + pixIdx]); // B
            }
        }

        // Compress with zlib (level 6 default)
        byte[] compressed;
        using (var ms = new MemoryStream())
        {
            // zlib header: CMF=0x78, FLG=0x9C (deflate, default compression)
            ms.WriteByte(0x78);
            ms.WriteByte(0x9C);
            using (var deflate = new DeflateStream(ms, CompressionLevel.Optimal, leaveOpen: true))
                deflate.Write(raw);
            // Adler-32 checksum
            uint adler = Adler32(raw);
            ms.WriteByte((byte)(adler >> 24));
            ms.WriteByte((byte)(adler >> 16));
            ms.WriteByte((byte)(adler >> 8));
            ms.WriteByte((byte)adler);
            compressed = ms.ToArray();
        }

        WriteChunk(output, "IDAT"u8, compressed);

        // IEND chunk
        WriteChunk(output, "IEND"u8, []);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static readonly byte[] PngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    private static byte FloatToByte(float v) => (byte)Math.Clamp((int)(v * 255f + 0.5f), 0, 255);

    private static void WriteChunk(Stream s, ReadOnlySpan<byte> typeBytes, byte[] data)
    {
        Span<byte> lenBuf = stackalloc byte[4];
        WriteUInt32BE(lenBuf, 0, (uint)data.Length);
        s.Write(lenBuf);
        s.Write(typeBytes);
        s.Write(data);

        // CRC32 over type + data
        uint crc = Crc32Init;
        foreach (byte b in typeBytes) crc = Crc32Update(crc, b);
        foreach (byte b in data)      crc = Crc32Update(crc, b);
        crc ^= 0xFFFFFFFF;
        WriteUInt32BE(lenBuf, 0, crc);
        s.Write(lenBuf);
    }

    private static void WriteUInt32BE(Span<byte> buf, int off, uint v)
    {
        buf[off]     = (byte)(v >> 24);
        buf[off + 1] = (byte)(v >> 16);
        buf[off + 2] = (byte)(v >> 8);
        buf[off + 3] = (byte)v;
    }

    // CRC32 (PNG uses standard CRC-32)
    private const uint Crc32Init = 0xFFFFFFFF;
    private static readonly uint[] Crc32Table = BuildCrc32Table();

    private static uint[] BuildCrc32Table()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        return table;
    }

    private static uint Crc32Update(uint crc, byte b) =>
        Crc32Table[(crc ^ b) & 0xFF] ^ (crc >> 8);

    // Adler-32 (zlib trailer)
    private static uint Adler32(ReadOnlySpan<byte> data)
    {
        uint s1 = 1, s2 = 0;
        foreach (byte b in data) { s1 = (s1 + b) % 65521; s2 = (s2 + s1) % 65521; }
        return (s2 << 16) | s1;
    }
}
