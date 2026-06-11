using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #207: prompt-lookup (n-gram) drafting — the model-free draft source for
/// speculative decoding. Pure token-matching logic, no models involved.
/// </summary>
public sealed class PromptLookupDraftTests
{
    [Fact]
    public void Propose_TailBigramRecurs_ReturnsContinuationOfMostRecentMatch()
    {
        var d = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        // Tail [10, 11] matches index 0; continuation is [12, 10, 11].
        d.Reset(new[] { 10, 11, 12, 10, 11 });

        Assert.Equal(new[] { 12, 10, 11 }, d.Propose(3));
        // maxTokens caps the proposal length.
        Assert.Equal(new[] { 12 }, d.Propose(1));
    }

    [Fact]
    public void Propose_LongerNgramPreferred()
    {
        var d = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        // Tail trigram [7, 8, 9] matches at index 0 (continuation 1); the bigram [8, 9]
        // ALSO matches at index 4 (continuation 2). The trigram match must win.
        d.Reset(new[] { 7, 8, 9, 1, 8, 9, 2, 7, 8, 9 });

        Assert.Equal(new[] { 1 }, d.Propose(1));
    }

    [Fact]
    public void Propose_MostRecentOccurrenceWins()
    {
        var d = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        // [5, 6] occurs twice; the later (index 3) continuation 9 must win over 7.
        d.Reset(new[] { 5, 6, 7, 5, 6, 9, 5, 6 });

        Assert.Equal(new[] { 9, 5, 6 }, d.Propose(5));
    }

    [Fact]
    public void Propose_NoMatch_ReturnsEmpty()
    {
        var d = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        d.Reset(new[] { 1, 2, 3, 4, 5, 6 });

        Assert.Empty(d.Propose(4));
        Assert.Empty(d.Propose(0));
    }

    [Fact]
    public void Append_ExtendsMatchableHistory()
    {
        var d = new PromptLookupDraft(ngramMax: 3, ngramMin: 2);
        d.Reset(new[] { 1, 2, 3 });
        Assert.Empty(d.Propose(2));

        // Generated tokens repeat the prompt opening: tail [1, 2] now matches index 0.
        d.Append(1);
        d.Append(2);
        Assert.Equal(new[] { 3, 1, 2 }, d.Propose(3));
    }

    [Fact]
    public void Reset_ClearsPreviousSession()
    {
        var d = new PromptLookupDraft(ngramMax: 2, ngramMin: 2);
        d.Reset(new[] { 1, 2, 1, 2 });
        Assert.NotEmpty(d.Propose(1));

        d.Reset(new[] { 9, 8, 7 });
        Assert.Empty(d.Propose(1));
        Assert.Equal(3, d.Count);
    }
}
