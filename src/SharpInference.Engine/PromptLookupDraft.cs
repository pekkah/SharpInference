namespace SharpInference.Engine;

/// <summary>
/// Model-free draft source for speculative decoding (issue #207, llama.cpp lookup-decoding
/// analog): proposes continuation tokens by matching the tail n-gram of the generated
/// context against the prompt + everything generated so far. Proposals cost no forward
/// passes, so on copy-heavy workloads (RAG quotation, summarization, code edits,
/// self-repetitive output) the speculative step gets its draft for free; when nothing
/// matches it proposes nothing and the step degrades to a plain single-token decode.
///
/// Matching: longest tail n-gram first (<see cref="NgramMax"/> down to
/// <see cref="NgramMin"/>), most recent occurrence wins (generated text repeats locally).
/// <see cref="NgramMin"/> defaults to 2 — 1-gram matches fire constantly and mostly
/// propose junk, which makes every step pay a wider verify batch for nothing.
/// </summary>
public sealed class PromptLookupDraft
{
    private readonly List<int> _history = new();

    /// <summary>Largest tail n-gram length tried for a match.</summary>
    public int NgramMax { get; }

    /// <summary>Smallest tail n-gram length tried before giving up (no proposal).</summary>
    public int NgramMin { get; }

    public PromptLookupDraft(int ngramMax = 3, int ngramMin = 2)
    {
        if (ngramMin < 1) throw new ArgumentOutOfRangeException(nameof(ngramMin));
        if (ngramMax < ngramMin) throw new ArgumentOutOfRangeException(nameof(ngramMax));
        NgramMax = ngramMax;
        NgramMin = ngramMin;
    }

    /// <summary>Tokens observed so far (prompt + emitted).</summary>
    public int Count => _history.Count;

    /// <summary>Reset the history to a new prompt (start of a generation).</summary>
    public void Reset(IReadOnlyList<int> promptTokens)
    {
        _history.Clear();
        if (promptTokens is not null) _history.AddRange(promptTokens);
    }

    /// <summary>Record an emitted token so future proposals can match it.</summary>
    public void Append(int token) => _history.Add(token);

    /// <summary>
    /// Propose up to <paramref name="maxTokens"/> continuation tokens for the current
    /// history. Returns an empty array when no tail n-gram of length
    /// [<see cref="NgramMin"/>, <see cref="NgramMax"/>] recurs earlier in the history.
    /// </summary>
    public int[] Propose(int maxTokens)
    {
        var h = _history;
        int len = h.Count;
        if (maxTokens <= 0) return Array.Empty<int>();

        for (int n = NgramMax; n >= NgramMin; n--)
        {
            if (len < n + 1) continue;

            // Most recent earlier occurrence of the tail h[len-n .. len): candidate start
            // positions i walk backward; the matched occurrence must end before the tail
            // ends (i + n < len) so there is at least one continuation token to copy.
            for (int i = len - n - 1; i >= 0; i--)
            {
                bool match = true;
                for (int j = 0; j < n; j++)
                {
                    if (h[i + j] != h[len - n + j]) { match = false; break; }
                }
                if (!match) continue;

                int start = i + n;
                int count = Math.Min(maxTokens, len - start);
                var proposal = new int[count];
                for (int t = 0; t < count; t++) proposal[t] = h[start + t];
                return proposal;
            }
        }
        return Array.Empty<int>();
    }
}
