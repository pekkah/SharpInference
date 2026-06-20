using SharpInference.Cli;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="RunCommand.FormatTopLogits"/> — the partial top-k selection that
/// replaced the per-token full-vocab <c>OrderByDescending</c> + <c>logits.ToArray()</c> in the
/// <c>--verbose-prompt</c> debug line (issue #155). The new O(V·k) pass must produce the exact
/// same string the old LINQ path did: top-k by descending value, ties broken by lower index.
/// </summary>
public sealed class VerbosePromptTopKTests
{
    // The original LINQ implementation, kept here as the oracle the fast path must match.
    private static string ReferenceTopK(float[] logits, int k) =>
        string.Join(" ", Enumerable.Range(0, logits.Length)
            .OrderByDescending(j => logits[j])
            .Take(Math.Min(k, logits.Length))
            .Select(j => $"{j}({logits[j]:F2})"));

    [Fact]
    public void Basic_DescendingByValue()
    {
        float[] logits = [1f, 5f, 3f, 2f, 4f];
        Assert.Equal("1(5.00) 4(4.00) 2(3.00)", RunCommand.FormatTopLogits(logits, 3));
    }

    [Fact]
    public void Ties_BreakByLowerIndex()
    {
        // Two equal top values: the lower index must come first (stable OrderByDescending).
        float[] logits = [5f, 1f, 5f, 5f];
        Assert.Equal("0(5.00) 2(5.00)", RunCommand.FormatTopLogits(logits, 2));
    }

    [Fact]
    public void KLargerThanLength_ReturnsAllSorted()
    {
        float[] logits = [2f, 1f];
        Assert.Equal("0(2.00) 1(1.00)", RunCommand.FormatTopLogits(logits, 5));
    }

    [Fact]
    public void EmptyOrNonPositiveK_ReturnsEmpty()
    {
        Assert.Equal("", RunCommand.FormatTopLogits([], 5));
        Assert.Equal("", RunCommand.FormatTopLogits([1f, 2f], 0));
    }

    [Fact]
    public void NegativeAndEqualValues_MatchReference()
    {
        float[] logits = [-3.5f, -3.5f, -1.0f, -3.5f, 0.0f];
        Assert.Equal(ReferenceTopK(logits, 5), RunCommand.FormatTopLogits(logits, 5));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(8)]
    public void MatchesReferenceOnPseudoRandomVocab(int k)
    {
        // Deterministic seed so the oracle comparison is reproducible. Includes deliberate
        // duplicate values (mod) to exercise tie-breaking against the stable reference.
        var rng = new Random(12345);
        var logits = new float[4096];
        for (int i = 0; i < logits.Length; i++)
            logits[i] = (float)Math.Round(rng.NextDouble() * 20.0 - 10.0, 1);

        Assert.Equal(ReferenceTopK(logits, k), RunCommand.FormatTopLogits(logits, k));
    }
}
