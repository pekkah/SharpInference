using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-free unit tests for <see cref="GgufTokenizer.BuildEogTokenIds"/> — the end-of-generation
/// resolution that lets generation stop on alternate end tokens (the fix for Gemma 4's run-on,
/// where <c>&lt;eos&gt;</c> id 1 is distinct from the configured EOS <c>&lt;turn|&gt;</c> id 106).
/// </summary>
public sealed class EogTokenIdsTests
{
    private static Dictionary<string, int> Vocab(params (string tok, int id)[] entries)
    {
        var d = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var (tok, id) in entries) d[tok] = id;
        return d;
    }

    [Fact]
    public void AlwaysContainsConfiguredEos()
    {
        var eog = GgufTokenizer.BuildEogTokenIds(Vocab(), new HashSet<int>(), eosTokenId: 42);
        Assert.Equal(new[] { 42 }, eog);
    }

    [Fact]
    public void CanonicalEos_AcceptedEvenWhenNotControl()
    {
        // Gemma 4 case: <eos> (id 1) is NORMAL-typed (not in specialIds) and the configured EOS
        // is <turn|> (id 106). <eos> must still be picked up.
        var vocab = Vocab(("<eos>", 1), ("<turn|>", 106));
        var eog = GgufTokenizer.BuildEogTokenIds(vocab, specialIds: new HashSet<int>(), eosTokenId: 106);
        Assert.Contains(106, eog);
        Assert.Contains(1, eog);
    }

    [Fact]
    public void BracketMarker_AcceptedOnlyWhenControlTyped()
    {
        // <|endoftext|> present in vocab but NOT typed control → must NOT become a stop, to avoid
        // silently truncating a model that uses the string as ordinary text.
        var vocab = Vocab(("<|endoftext|>", 50), ("<eos-real>", 2));
        var notControl = GgufTokenizer.BuildEogTokenIds(vocab, specialIds: new HashSet<int>(), eosTokenId: 2);
        Assert.DoesNotContain(50, notControl);

        // Same token, now typed control → accepted.
        var control = GgufTokenizer.BuildEogTokenIds(vocab, specialIds: new HashSet<int> { 50 }, eosTokenId: 2);
        Assert.Contains(50, control);
    }

    [Fact]
    public void NoDuplicates_WhenEosNameAlsoMatches()
    {
        // Configured EOS id is also reachable by name → appears once.
        var vocab = Vocab(("<|im_end|>", 7));
        var eog = GgufTokenizer.BuildEogTokenIds(vocab, specialIds: new HashSet<int> { 7 }, eosTokenId: 7);
        Assert.Equal(new[] { 7 }, eog);
    }

    [Fact]
    public void IgnoresIdZeroAndAbsentNames()
    {
        // id 0 is pad/unk territory — never a stop; absent names are skipped.
        var vocab = Vocab(("<eos>", 0));
        var eog = GgufTokenizer.BuildEogTokenIds(vocab, specialIds: new HashSet<int> { 0 }, eosTokenId: 5);
        Assert.Equal(new[] { 5 }, eog);
    }

    [Fact]
    public void CollectsMultipleControlMarkers()
    {
        var vocab = Vocab(("<|im_end|>", 10), ("<|eot_id|>", 11), ("<end_of_turn>", 12));
        var eog = GgufTokenizer.BuildEogTokenIds(
            vocab, specialIds: new HashSet<int> { 10, 11 }, eosTokenId: 9);
        Assert.Contains(9, eog);    // configured EOS
        Assert.Contains(10, eog);   // control bracket
        Assert.Contains(11, eog);   // control bracket
        Assert.Contains(12, eog);   // canonical, accepted regardless of type
    }
}
