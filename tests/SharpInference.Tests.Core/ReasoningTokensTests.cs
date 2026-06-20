using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-free unit tests for <see cref="GgufTokenizer.ResolveReasoningTokens"/> — the open/close
/// boundary-token resolution an engine uses to split the reasoning channel out of the text stream.
/// Centralizing it on the tokenizer (exposed as <see cref="ITokenizer.ReasoningTokens"/>) is the
/// fix for issue #304: an in-process consumer using the convenience engine constructor now gets the
/// same Gemma 4 <c>&lt;|channel&gt;</c>/<c>&lt;channel|&gt;</c> split the server and CLI configure.
/// </summary>
public sealed class ReasoningTokensTests
{
    private static Dictionary<string, int> Vocab(params (string tok, int id)[] entries)
    {
        var d = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (tok, id) in entries) d[tok] = id;
        return d;
    }

    [Fact]
    public void NoMarkers_ReturnsDisabled()
    {
        Assert.Equal((-1, -1), GgufTokenizer.ResolveReasoningTokens(Vocab()));
    }

    [Fact]
    public void ChatMlThinkTokens_Resolved()
    {
        var vocab = Vocab(("<think>", 50), ("</think>", 51));
        Assert.Equal((50, 51), GgufTokenizer.ResolveReasoningTokens(vocab));
    }

    [Fact]
    public void Gemma4ChannelTokens_Resolved()
    {
        // Real Gemma 4 12B QAT ids: <|channel> = 100, <channel|> = 101 (both USER_DEFINED).
        var vocab = Vocab(("<|channel>", 100), ("<channel|>", 101));
        Assert.Equal((100, 101), GgufTokenizer.ResolveReasoningTokens(vocab));
    }

    [Fact]
    public void ThinkTokens_TakePrecedenceOverChannel()
    {
        var vocab = Vocab(("<think>", 50), ("</think>", 51), ("<|channel>", 100), ("<channel|>", 101));
        Assert.Equal((50, 51), GgufTokenizer.ResolveReasoningTokens(vocab));
    }

    [Fact]
    public void ZeroIds_Rejected()
    {
        // id 0 is usually <pad>/<unk> and would mis-trigger the split — both ids must be positive.
        Assert.Equal((-1, -1), GgufTokenizer.ResolveReasoningTokens(Vocab(("<think>", 0), ("</think>", 51))));
        Assert.Equal((-1, -1), GgufTokenizer.ResolveReasoningTokens(Vocab(("<|channel>", 100), ("<channel|>", 0))));
    }

    [Fact]
    public void PartialPair_Rejected()
    {
        // Only one half present → no split (an unmatched boundary would corrupt the stream).
        Assert.Equal((-1, -1), GgufTokenizer.ResolveReasoningTokens(Vocab(("<think>", 50))));
        Assert.Equal((-1, -1), GgufTokenizer.ResolveReasoningTokens(Vocab(("<|channel>", 100))));
    }
}
