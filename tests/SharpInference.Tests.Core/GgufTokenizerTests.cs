using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class GgufTokenizerTests
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static GgufTokenizer? CreateTokenizer()
    {
        var path = FindModelPath();
        if (path is null) return null;
        using var model = GgufModel.Open(path);
        return GgufTokenizer.FromGgufModel(model);
    }

    [Fact]
    public void FromGgufModel_LoadsSuccessfully()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        Assert.Equal(49152, tokenizer.VocabSize);
        Assert.Equal(1, tokenizer.BosTokenId);
        Assert.Equal(2, tokenizer.EosTokenId);
        Assert.Equal(0, tokenizer.UnknownTokenId);
        Assert.False(tokenizer.AddBosToken);
    }

    [Fact]
    public void Encode_SimpleText_ReturnsTokenIds()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var ids = tokenizer.Encode("Hello");
        Assert.NotEmpty(ids);
        Assert.True(ids.All(id => id >= 0 && id < tokenizer.VocabSize));
    }

    [Fact]
    public void Decode_RoundTrips_SimpleText()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var text = "Hello, world!";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Decode_RoundTrips_LongerText()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var text = "The quick brown fox jumps over the lazy dog.";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var ids = tokenizer.Encode("");
        Assert.Empty(ids);
    }

    [Fact]
    public void Encode_MultipleWords_ProducesMultipleTokens()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var ids = tokenizer.Encode("This is a test of the tokenizer");
        // A sentence with common words should produce several tokens
        Assert.True(ids.Count >= 3, $"Expected at least 3 tokens, got {ids.Count}");
    }

    [Fact]
    public void Decode_RoundTrips_SpecialCharacters()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var text = "x = 42; // comment\nprint(x)";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Decode_RoundTrips_Unicode()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var text = "café résumé naïve";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void DecodeBytes_PerTokenStream_ReassemblesMultiByteUnicode()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        // Multi-byte UTF-8 (3-byte CJK and curly quotes, em-dash) is the regression
        // case for issue #13: a single character is split across token boundaries.
        var text = "你好，世界 — “hello”";
        var ids = tokenizer.Encode(text);

        // Concat all per-token DecodeBytes output and verify it equals UTF-8 of original.
        var bytes = new System.Collections.Generic.List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        Assert.Equal(System.Text.Encoding.UTF8.GetBytes(text), bytes.ToArray());
    }

    [Fact]
    public void DecodeBytes_StreamedThroughUtf8Decoder_ProducesNoReplacementChars()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var text = "你好世界";
        var ids = tokenizer.Encode(text);

        var dec = new Utf8StreamDecoder();
        var sb = new System.Text.StringBuilder();
        foreach (var id in ids)
            sb.Append(dec.Append(tokenizer.DecodeBytes(id)));
        sb.Append(dec.Flush());

        var output = sb.ToString();
        Assert.Equal(text, output);
        Assert.DoesNotContain('�', output);
    }

    [Fact]
    public void DecodeBytes_AsciiToken_RoundTripsThroughUtf8()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        var ids = tokenizer.Encode("Hello, world!");
        var bytes = new System.Collections.Generic.List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        Assert.Equal("Hello, world!", System.Text.Encoding.UTF8.GetString(bytes.ToArray()));
    }
}
