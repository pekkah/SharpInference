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
    public void Encode_IndentedCode_StaysWithinVocab()
    {
        // Issue #267: CodeGenTokenizer injects model-independent consecutive-whitespace tokens
        // at ids beyond this GGUF's 49152-row embedding (e.g. an 8-space run → id 50280). Feeding
        // one to the GPU embedding gather reads out of bounds and aborts the CUDA context (error
        // 700). The tokenizer must decompose such tokens into in-vocab byte tokens so every id is
        // addressable in the embedding table.
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        foreach (var text in new[]
                 {
                     "        private const int PageSize = 16;", // the original repro (8-space indent)
                     "    if (x) {\n        return;\n    }",      // 4- and 8-space runs + newlines/tabs
                     new string(' ', 8) + "x",
                 })
        {
            var ids = tokenizer.Encode(text);
            Assert.NotEmpty(ids);
            Assert.All(ids, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
        }
    }

    [Fact]
    public void Encode_MultiSpaceRun_DecomposesToInVocabSpaceTokens()
    {
        // The 2–8-space CodeGenTokenizer tokens (ids 50280–50286) decompose into repeated
        // single-space in-vocab tokens (issue #267), preserving the whitespace rather than
        // dropping it or emitting an unembeddable id.
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        // Encode N spaces followed by a sentinel and confirm the count of leading whitespace
        // tokens scales with N and every id is in range.
        var four = tokenizer.Encode(new string(' ', 4) + "X");
        var eight = tokenizer.Encode(new string(' ', 8) + "X");
        Assert.All(four, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
        Assert.All(eight, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
        // More spaces → at least as many tokens (decomposition is per-space).
        Assert.True(eight.Count > four.Count,
            $"expected more tokens for 8 spaces ({eight.Count}) than 4 ({four.Count})");
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
