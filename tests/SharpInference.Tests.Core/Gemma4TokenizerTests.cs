using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class Gemma4TokenizerTests
{
    private const string Gemma4ModelPath = @"E:\models\gemma-4-E4B-it-Q8_0.gguf";

    private static GgufTokenizer? CreateTokenizer()
    {
        if (!File.Exists(Gemma4ModelPath)) return null;
        using var model = GgufModel.Open(Gemma4ModelPath);
        return GgufTokenizer.FromGgufModel(model);
    }

    [Fact]
    public void Gemma4_Tokenizer_LoadsFromGguf()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        Assert.Equal(262_144, tokenizer.VocabSize);
        Assert.Equal(2, tokenizer.BosTokenId);
        Assert.Equal(106, tokenizer.EosTokenId);
        Assert.Equal(3, tokenizer.UnknownTokenId);
    }

    [Fact]
    public void Gemma4_Tokenizer_RoundTripsHello()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        const string text = "Hello world";
        var ids = tokenizer.Encode(text);
        Assert.NotEmpty(ids);

        var decoded = tokenizer.Decode(ids);
        Assert.Equal(text, decoded.TrimStart());
    }

    [Fact]
    public void Gemma4_Tokenizer_HandlesSpecialTokens()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        // In Gemma 4, only token-type-3 control tokens are special; the metadata EOS
        // for this model is <turn|> at id 106 (literal <eos> at id 1 is a normal token).
        // Verify control tokens like <bos> and <turn|> survive as single IDs through encode.
        Assert.True(tokenizer.SpecialTokens.ContainsKey("<bos>"));
        Assert.True(tokenizer.SpecialTokens.ContainsKey("<turn|>"));

        var ids = tokenizer.Encode("<bos>Hello<turn|>");
        Assert.Contains(tokenizer.BosTokenId, ids);
        Assert.Contains(tokenizer.EosTokenId, ids);
    }

    [Fact]
    public void Gemma4_DecodeBytes_RoundTripsAscii()
    {
        var tokenizer = CreateTokenizer();
        if (tokenizer is null) return;

        const string text = "Hello, world!";
        var ids = tokenizer.Encode(text);
        var bytes = new List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        var roundTripped = System.Text.Encoding.UTF8.GetString(bytes.ToArray());
        Assert.Equal(text, roundTripped.TrimStart());
    }
}
