using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Parity tests for <see cref="GgufTokenizer.SpmMergePieces"/> — the O(n log n) priority-queue
/// SentencePiece merge that replaced an O(n²) greedy loop. These run with no model file: they
/// feed synthetic merge tables and assert the new algorithm is byte-identical to a reference
/// implementation of the old behaviour (the exact greedy loop that shipped before), which is the
/// contract the rewrite must preserve.
/// </summary>
public sealed class SpmMergeTests
{
    /// <summary>
    /// Reference (oracle): the original O(n²) algorithm verbatim — repeatedly find the
    /// lowest-rank adjacent merge (leftmost on a tie) and apply it. The fast path must match
    /// this exactly for every input.
    /// </summary>
    private static List<string> NaiveMerge(List<string> input, IReadOnlyDictionary<(string, string), int> merges)
    {
        var pieces = new List<string>(input);
        while (true)
        {
            int bestIdx = -1;
            int bestPriority = int.MaxValue;
            for (int i = 0; i < pieces.Count - 1; i++)
            {
                if (merges.TryGetValue((pieces[i], pieces[i + 1]), out int pri) && pri < bestPriority)
                {
                    bestPriority = pri;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            pieces[bestIdx] = pieces[bestIdx] + pieces[bestIdx + 1];
            pieces.RemoveAt(bestIdx + 1);
        }
        return pieces;
    }

    [Fact]
    public void EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(GgufTokenizer.SpmMergePieces([], new Dictionary<(string, string), int>()));
    }

    [Fact]
    public void SingleSymbol_Unchanged()
    {
        Assert.Equal(["a"], GgufTokenizer.SpmMergePieces(["a"], new Dictionary<(string, string), int>()));
    }

    [Fact]
    public void NoApplicableMerges_Unchanged()
    {
        var merges = new Dictionary<(string, string), int> { [("x", "y")] = 0 };
        Assert.Equal(["a", "b", "c"], GgufTokenizer.SpmMergePieces(["a", "b", "c"], merges));
    }

    [Fact]
    public void HighestPriorityMergeAppliedFirst_ThenCascades()
    {
        // Ranks: (b,c)=0 fires before (a,b)=1. After "bc" forms, (a,"bc") isn't a merge,
        // so the result is ["a","bc"] — not ["ab","c"]. Exercises priority ordering.
        var merges = new Dictionary<(string, string), int> { [("a", "b")] = 1, [("b", "c")] = 0 };
        Assert.Equal(["a", "bc"], GgufTokenizer.SpmMergePieces(["a", "b", "c"], merges));
    }

    [Fact]
    public void RepeatedBigram_MergesLeftmostFirst()
    {
        // Two (a,a) candidates share rank 0; leftmost must win — matching the old scan.
        var merges = new Dictionary<(string, string), int> { [("a", "a")] = 0, [("aa", "a")] = 5 };
        // leftmost (a,a)->"aa" gives ["aa","a"], then ("aa","a")=5 -> "aaa".
        Assert.Equal(["aaa"], GgufTokenizer.SpmMergePieces(["a", "a", "a"], merges));
    }

    [Fact]
    public void MultiCharOperandMerges_Cascade()
    {
        // a+b -> ab (rank 0), then ab+c -> abc (rank 1).
        var merges = new Dictionary<(string, string), int> { [("a", "b")] = 0, [("ab", "c")] = 1 };
        Assert.Equal(["abc"], GgufTokenizer.SpmMergePieces(["a", "b", "c"], merges));
    }

    /// <summary>
    /// Fuzz: thousands of random (input, merge-table) pairs over a tiny alphabet — including
    /// multi-char operands, cascading merges, and duplicate bigrams with shared ranks — must
    /// produce identical output to the reference greedy algorithm.
    /// </summary>
    [Fact]
    public void FastPath_MatchesNaive_AcrossRandomInputs()
    {
        var rng = new Random(20260611);
        string[] alphabet = ["a", "b", "c"];

        // Candidate merge operands: all 1- to 3-char strings over the alphabet, so the table
        // can express multi-level merges (e.g. "ab"+"c").
        var operands = new List<string>(alphabet);
        foreach (var x in alphabet)
            foreach (var y in alphabet)
            {
                operands.Add(x + y);
                foreach (var z in alphabet) operands.Add(x + y + z);
            }

        for (int iter = 0; iter < 5000; iter++)
        {
            // Random merge table: shuffle a random subset of (left,right) operand pairs and
            // assign each a unique rank (its position) — mirrors a real merges file's ordering.
            int pairCount = rng.Next(1, 40);
            var pairs = new List<(string, string)>(pairCount);
            for (int k = 0; k < pairCount; k++)
                pairs.Add((operands[rng.Next(operands.Count)], operands[rng.Next(operands.Count)]));
            var merges = new Dictionary<(string, string), int>();
            int rank = 0;
            foreach (var p in pairs) merges[p] = rank++; // later dup keys overwrite — fine, still a valid table

            // Random input of single symbols (the real entry shape: one code point per piece).
            int len = rng.Next(0, 30);
            var input = new List<string>(len);
            for (int k = 0; k < len; k++) input.Add(alphabet[rng.Next(alphabet.Length)]);

            var expected = NaiveMerge(input, merges);
            var actual = GgufTokenizer.SpmMergePieces(new List<string>(input), merges);

            Assert.True(expected.SequenceEqual(actual),
                $"mismatch on input=[{string.Join(",", input)}] " +
                $"expected=[{string.Join(",", expected)}] actual=[{string.Join(",", actual)}]");
        }
    }
}
