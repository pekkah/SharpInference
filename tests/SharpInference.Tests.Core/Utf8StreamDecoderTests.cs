using System.Text;
using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class Utf8StreamDecoderTests
{
    [Fact]
    public void Append_EmptyInput_ReturnsEmpty()
    {
        var dec = new Utf8StreamDecoder();
        Assert.Equal("", dec.Append(ReadOnlySpan<byte>.Empty));
    }

    [Fact]
    public void Append_AsciiBytes_ReturnsSameText()
    {
        var dec = new Utf8StreamDecoder();
        Assert.Equal("Hello", dec.Append("Hello"u8));
    }

    [Fact]
    public void Append_CompleteMultiByteChar_ReturnsChar()
    {
        // 中 is 0xE4 0xB8 0xAD in UTF-8.
        var dec = new Utf8StreamDecoder();
        Assert.Equal("中", dec.Append([0xE4, 0xB8, 0xAD]));
    }

    [Fact]
    public void Append_SplitMultiByteChar_BuffersUntilComplete()
    {
        // 中 = 0xE4 0xB8 0xAD split across three calls.
        var dec = new Utf8StreamDecoder();
        Assert.Equal("", dec.Append([0xE4]));
        Assert.Equal("", dec.Append([0xB8]));
        Assert.Equal("中", dec.Append([0xAD]));
    }

    [Fact]
    public void Append_SplitFourByteEmoji_BuffersUntilComplete()
    {
        // 😀 = U+1F600 = F0 9F 98 80 in UTF-8.
        var dec = new Utf8StreamDecoder();
        Assert.Equal("", dec.Append([0xF0, 0x9F]));
        Assert.Equal("", dec.Append([0x98]));
        Assert.Equal("😀", dec.Append([0x80]));
    }

    [Fact]
    public void Append_AsciiAfterIncompleteCharBoundary_ReassemblesCorrectly()
    {
        // "a中b" = 0x61, 0xE4 0xB8 0xAD, 0x62. Split between bytes 2 and 3 of 中.
        var dec = new Utf8StreamDecoder();
        Assert.Equal("a", dec.Append([0x61, 0xE4, 0xB8]));
        Assert.Equal("中b", dec.Append([0xAD, 0x62]));
    }

    [Fact]
    public void Flush_AfterCompleteInput_ReturnsEmpty()
    {
        var dec = new Utf8StreamDecoder();
        dec.Append("Hello"u8);
        Assert.Equal("", dec.Flush());
    }

    [Fact]
    public void Flush_AfterTrulyIncompleteBytes_EmitsReplacementChar()
    {
        // 0xE4 0xB8 starts a 3-byte sequence; if the third byte never arrives,
        // flush should surface a single U+FFFD.
        var dec = new Utf8StreamDecoder();
        Assert.Equal("", dec.Append([0xE4, 0xB8]));
        Assert.Equal("�", dec.Flush());
    }

    [Fact]
    public void Streaming_ConcatenatedAppends_MatchesGetString()
    {
        // Stream a Chinese sentence one byte at a time and verify the output
        // matches Encoding.UTF8.GetString of all bytes.
        var text = "你好，世界！😀";
        var bytes = Encoding.UTF8.GetBytes(text);

        var dec = new Utf8StreamDecoder();
        var sb = new StringBuilder();
        foreach (var b in bytes)
            sb.Append(dec.Append([b]));
        sb.Append(dec.Flush());

        Assert.Equal(text, sb.ToString());
    }
}
