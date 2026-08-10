using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Regression tests for issue #454: <see cref="SamplingParams.RepetitionPenalty"/> was a silent
/// no-op for every library caller. The penalty is gated on <see cref="SamplingParams.PreviousTokens"/>
/// being populated, and no engine ever populated it — only the CLI's own decode loop did — so a
/// consumer that set the penalty and called <c>GenerateAsync</c> got unpenalised sampling.
/// <para>
/// The forward pass here returns a constant two-candidate distribution and the tests sample at a
/// near-zero temperature, so sampling is deterministic without needing to control the engine's RNG
/// (<c>InferenceEngine</c> seeds its own <c>Random</c>): whichever candidate has the higher
/// post-penalty logit is drawn with overwhelming probability. That makes "did the penalty apply?"
/// observable as an exact output string.
/// </para>
/// </summary>
public sealed class RepetitionPenaltyTests
{
    private const int Vocab = 128;
    private const int TokA = 'a';    // logit 2.0 — the model's preferred token
    private const int TokB = 'b';    // logit 1.5 — the runner-up the penalty promotes

    // 1/Temperature = 200, so a post-penalty logit gap of 0.167 (the smallest any test relies on)
    // becomes a softmax ratio of e^33 — deterministic in practice, with no dependence on the seed.
    private const float SharpTemp = 0.005f;

    /// <summary>Single-byte tokenizer: token id == ASCII byte. EOG is an id no forward emits.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 0;
        public int EosTokenId => 0;          // NUL — never produced below
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            new Dictionary<string, int>(StringComparer.Ordinal);

        public byte[] DecodeBytes(int token) => token is > 0 and < Vocab ? [(byte)token] : [];

        public IReadOnlyList<int> Encode(string text)
        {
            var ids = new int[text.Length];
            for (int i = 0; i < text.Length; i++) ids[i] = text[i];
            return ids;
        }

        public string Decode(IEnumerable<int> tokens)
        {
            var sb = new StringBuilder();
            foreach (int t in tokens) sb.Append(Encoding.UTF8.GetString(DecodeBytes(t)));
            return sb.ToString();
        }
    }

    private static float[] Row()
    {
        var r = new float[Vocab];
        r[TokA] = 2.0f;
        r[TokB] = 1.5f;
        return r;
    }

    /// <summary>Context-free forward pass: the same two-candidate logits at every position, so any
    /// variation in the output can only come from the sampler's penalty state.</summary>
    private sealed class ConstantForwardPass : IForwardPass
    {
        private readonly float[] _logits = Row();
        public int VocabSize => Vocab;
        public int MaxSeqLen => 4096;
        public ReadOnlySpan<float> Forward(int token, int position) => _logits;
        public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0) => _logits;
        public void TruncateTo(int length) { }
        public void ResetCache() { }
        public bool SupportsPartialRewind => true;
        public void Dispose() { }
    }

    private sealed class FakeCache : ISequenceKvCache { public void Dispose() { } }

    /// <summary>Batched twin of <see cref="ConstantForwardPass"/> for the continuous-batching path.</summary>
    private sealed class ConstantBatchedForwardPass : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 8192;
        public bool PrefillDequantCacheActive => false;
        public bool SupportsBatchedGpuArgmax => true;

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
            => Row();

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        {
            var outp = new float[]?[chunks.Length];
            for (int i = 0; i < chunks.Length; i++) outp[i] = wantLogits[i] ? Row() : null;
            return outp;
        }

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var outp = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++) outp[i] = Row();
            return outp;
        }

        public (int Token, float Logit)[] BatchForwardMultiArgmax(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var outp = new (int, float)[tokens.Length];
            for (int i = 0; i < tokens.Length; i++) outp[i] = (TokA, 2.0f);
            return outp;
        }
    }

    private static async Task<string> Run(IInferenceEngine engine, string prompt, SamplingParams sp)
    {
        var sb = new StringBuilder();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await foreach (var s in engine.GenerateAsync(prompt, sp, cts.Token))
            sb.Append(s);
        return sb.ToString();
    }

    /// <summary>Adjacent-repeat count — the "repetition spiral" signal from the bug report.</summary>
    private static int AdjacentRepeats(string s)
    {
        int n = 0;
        for (int i = 1; i < s.Length; i++)
            if (s[i] == s[i - 1]) n++;
        return n;
    }

    // ── The reported defect ──────────────────────────────────────────────

    [Fact]
    public async Task GenerateAsync_RepetitionPenalty_ReducesRepeatedTokens()
    {
        // The assertion from the bug report: the same generation at penalty 1.5 must repeat
        // measurably less than at 1.0. Before #454 both runs produced "aaaaaaaaaaaa" — the
        // engine never built the PreviousTokens window, so the penalty was dropped on the floor.
        const int n = 12;
        using var baseline = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string unpenalised = await Run(baseline, "prompt",
            new SamplingParams { Temperature = SharpTemp, MaxNewTokens = n, RepetitionPenalty = 1.0f });

        using var penalisedEngine = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string penalised = await Run(penalisedEngine, "prompt",
            new SamplingParams { Temperature = SharpTemp, MaxNewTokens = n, RepetitionPenalty = 1.5f });

        // Unpenalised: the top logit wins every step, so the model repeats itself forever.
        Assert.Equal(new string('a', n), unpenalised);
        Assert.Equal(n - 1, AdjacentRepeats(unpenalised));

        // Penalised: 'a' is demoted below 'b' once emitted, so 'b' breaks the run. It then settles
        // back onto 'a' — with the penalty applied once per distinct token (#457), both candidates
        // are in the window from step 3 on, which scales them both and restores the original
        // ordering (a: 2.0/1.5 = 1.33 > b: 1.5/1.5 = 1.0). That settle-back is the llama.cpp
        // contract, and an artifact of a two-candidate mock; a real vocabulary has thousands of
        // alternatives for the freed mass. What matters here is that the penalty demonstrably
        // changes the output and cuts the repeat rate, which it could not do at all before #454.
        Assert.Equal("abaaaaaaaaaa", penalised);
        Assert.True(AdjacentRepeats(penalised) < AdjacentRepeats(unpenalised),
            $"penalised={AdjacentRepeats(penalised)} unpenalised={AdjacentRepeats(unpenalised)}");
    }

    [Fact]
    public async Task GenerateAsync_RepetitionPenalty_AppliesOnTopKFastPath()
    {
        // TopK > 0 routes sampling through Sampler.SampleTopK, which implements the penalty
        // separately from the slow path (it over-selects top-(k+W) raw candidates, penalises that
        // small set, and re-sorts). Every other test here leaves TopK at 0 and so exercises only
        // the slow path — but top-k is set in most real configs, and the over-select is sized off
        // the window this change introduced, so the engine-built window has to be covered here too.
        const int n = 10;
        using var penalisedEngine = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string penalised = await Run(penalisedEngine, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = n,
            TopK = 2,
            RepetitionPenalty = 1.5f,
        });

        using var baseline = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string unpenalised = await Run(baseline, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = n,
            TopK = 2,
            RepetitionPenalty = 1.0f,
        });

        Assert.Equal("abaaaaaaaa", penalised);   // see ReducesRepeatedTokens for the settle-back
        Assert.Equal(new string('a', n), unpenalised);
        Assert.True(AdjacentRepeats(penalised) < AdjacentRepeats(unpenalised));
    }

    [Fact]
    public async Task GenerateAsync_PenaltyWindow_IsBoundedByPenaltyLastN()
    {
        // PenaltyLastN = 1 keeps only the previous token, so the penalty can never compound:
        // 'a' emitted → 'a' demoted to 1.33, 'b' (1.5) wins → window holds only 'b', so 'b' is
        // demoted to 1.0 and 'a' (back at its full 2.0) wins. Alternation, driven by a window
        // that is provably evicting.
        using var engine = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string text = await Run(engine, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 8,
            RepetitionPenalty = 1.5f,
            PenaltyLastN = 1,
        });

        Assert.Equal("abababab", text);
    }

    [Fact]
    public async Task GenerateAsync_SeedsPenaltyWindowFromPrompt()
    {
        // Cross-turn repetition (bug report item 3): a generation-only window is empty at token 0,
        // which is exactly where an identical reply opening gets chosen. Seeded from the prompt
        // tail (llama.cpp's behaviour), the very FIRST generated token already avoids what the
        // prompt just said.
        using var seeded = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string withSeed = await Run(seeded, "aaaa", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 4,
            RepetitionPenalty = 1.5f,
        });

        using var unseeded = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        string withoutSeed = await Run(unseeded, "aaaa", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 4,
            RepetitionPenalty = 1.5f,
            PenaltySeedFromPrompt = false,
        });

        Assert.Equal('b', withSeed[0]);       // prompt's 'a's already demoted at token 0
        Assert.Equal('a', withoutSeed[0]);    // opt-out: penalty covers only what this call generates
    }

    [Fact]
    public async Task GenerateAsync_CallerSuppliedPreviousTokens_IsNotOverwritten()
    {
        // Back-compat: a caller that maintains its own window keeps owning it. The engine must not
        // replace it, and must not append to it either — the penalty then covers exactly the tokens
        // supplied. Here that is a single 'a', so 'b' wins the first step; because the window never
        // grows, 'b' is never penalised and wins every subsequent step too.
        using var engine = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
        var caller = new List<int> { TokA };
        string text = await Run(engine, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 6,
            RepetitionPenalty = 1.5f,
            PreviousTokens = caller,
        });

        Assert.Equal("bbbbbb", text);
        Assert.Equal([TokA], caller);   // untouched by the engine
    }

    [Fact]
    public async Task ContinuousBatching_AppliesRepetitionPenalty_PerSequence()
    {
        // The batcher had the same gap (seq.Sp / req.Sp passed straight through). Two co-resident
        // sequences with different penalties prove the window is per-sequence, not shared: one
        // alternates, its co-tenant repeats, in the same batched decode steps.
        using var engine = new ContinuousBatchingEngine(
            new ConstantBatchedForwardPass(), new CharTokenizer(), "test", maxBatchSize: 2);

        var penalised = Run(engine, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 10,
            RepetitionPenalty = 1.5f,
        });
        var plain = Run(engine, "prompt", new SamplingParams
        {
            Temperature = SharpTemp,
            MaxNewTokens = 10,
            RepetitionPenalty = 1.0f,
        });

        string[] results = await Task.WhenAll(penalised, plain);

        Assert.Equal("abaaaaaaaa", results[0]);   // penalty applied to this sequence
        Assert.Equal("aaaaaaaaaa", results[1]);   // co-tenant, penalty off — window never shared
    }

    // ── Once per distinct token, not per occurrence (#457) ───────────────

    /// <summary>
    /// Materialises the post-penalty, post-filter distribution the sampler would draw from.
    /// </summary>
    private static float[] Distribution(float[] logits, SamplingParams sp)
    {
        var probs = new float[logits.Length];
        Sampler.BuildFilteredDistribution(logits, sp, probs);
        return probs;
    }

    [Theory]
    [InlineData(0)]      // slow path
    [InlineData(8)]      // top-k fast path (separate penalty implementation)
    public void Penalty_AppliesOncePerDistinctToken_NotPerOccurrence(int topK)
    {
        // The reported defect: the effective penalty was penalty^occurrences, so a token that
        // recurs ~10 times in a 256-token window took ~3.5x at a nominal 1.15. llama.cpp applies
        // the repeat penalty exactly once however high the occurrence count (the count feeds only
        // its separate frequency/presence terms), and that is the contract these paths now hold to.
        float[] logits = [3.0f, 2.5f, 2.0f, -1.0f, -2.0f, 1.0f, 0.5f, 0.25f];

        var once = new SamplingParams
        {
            Temperature = 1f, TopK = topK, MinP = 0f, TopP = 1f,
            RepetitionPenalty = 1.15f,
            PreviousTokens = new List<int> { 0, 3 },
        };
        // Same two distinct tokens, each repeated many times — the shape a real 256-token chat
        // window has, and the input that used to compound.
        var many = once with
        {
            PreviousTokens = new List<int> { 0, 3, 0, 0, 3, 0, 3, 3, 0, 0, 3, 0, 0, 3, 0 },
        };

        Assert.Equal(Distribution(logits, once), Distribution(logits, many));
    }

    [Fact]
    public void Penalty_WindowSize_ChangesCoverage_NotStrength()
    {
        // The property that makes PenaltyLastN = 256 a safe default: widening the window brings
        // more tokens into scope but must never deepen the penalty on a token already in it.
        // Token 0 is in scope in both cases; token 5 only in the wider one.
        float[] logits = [3.0f, 2.5f, 2.0f, -1.0f, -2.0f, 1.0f];
        var baseSp = new SamplingParams { Temperature = 1f, MinP = 0f, TopP = 1f, RepetitionPenalty = 1.2f };

        var narrow = baseSp with { PreviousTokens = new List<int> { 0 } };
        var wide = baseSp with { PreviousTokens = new List<int> { 0, 0, 0, 0, 5 } };

        var dNarrow = Distribution(logits, narrow);
        var dWide = Distribution(logits, wide);

        // Token 5 entered scope, so the distributions are not identical...
        Assert.NotEqual(dNarrow, dWide);
        // ...but token 0's demotion relative to the untouched token 1 is unchanged: the extra
        // occurrences bought nothing. (Compare ratios — adding token 5 renormalises the whole
        // distribution, so the absolute probabilities shift even for untouched tokens.)
        Assert.Equal(dNarrow[0] / dNarrow[1], dWide[0] / dWide[1], 5);
    }

    [Fact]
    public async Task GenerateAsync_RepeatedTokenRate_IsMonotonicInPenalty()
    {
        // Suggestion from the report: the repeated-token rate should never INCREASE as the penalty
        // rises. Per-occurrence compounding could reverse that curve, because the exponent grew
        // with the window as generation proceeded.
        int[] repeats = new int[5];
        float[] penalties = [1.0f, 1.05f, 1.1f, 1.2f, 1.5f];

        for (int i = 0; i < penalties.Length; i++)
        {
            using var engine = new InferenceEngine(new ConstantForwardPass(), new CharTokenizer(), "mock");
            string text = await Run(engine, "prompt", new SamplingParams
            {
                Temperature = SharpTemp,
                MaxNewTokens = 16,
                RepetitionPenalty = penalties[i],
            });
            repeats[i] = AdjacentRepeats(text);
        }

        for (int i = 1; i < repeats.Length; i++)
            Assert.True(repeats[i] <= repeats[i - 1],
                $"penalty {penalties[i]} repeated more than {penalties[i - 1]}: " +
                $"[{string.Join(", ", repeats)}]");
    }

    // ── PenaltyWindow unit behaviour ─────────────────────────────────────

    [Fact]
    public void PenaltyWindow_EvictsOldestPastCapacity()
    {
        var w = new PenaltyWindow(3);
        for (int i = 1; i <= 5; i++) w.Add(i);

        Assert.Equal(3, w.Count);
        Assert.Equal([3, 4, 5], w.ToArray());          // oldest-first, 1 and 2 evicted
        Assert.Equal(3, w[0]);
        Assert.Equal(5, w[2]);
        Assert.Throws<ArgumentOutOfRangeException>(() => w[3]);
    }

    [Fact]
    public void PenaltyWindow_Seed_KeepsOnlyTheTail()
    {
        var w = new PenaltyWindow(3);
        w.Seed([1, 2, 3, 4, 5, 6]);
        Assert.Equal([4, 5, 6], w.ToArray());

        // The IReadOnlyList overload (what ITokenizer.Encode returns) trims identically.
        var fromList = new PenaltyWindow(3);
        fromList.Seed(new List<int> { 1, 2, 3, 4, 5, 6 });
        Assert.Equal([4, 5, 6], fromList.ToArray());

        // Seeding then appending keeps evicting from the front.
        w.Add(7);
        Assert.Equal([5, 6, 7], w.ToArray());
    }

    [Fact]
    public void PenaltyWindow_ZeroCapacity_IsUnbounded()
    {
        var w = new PenaltyWindow(0);
        for (int i = 0; i < 200; i++) w.Add(i);

        Assert.Equal(200, w.Count);
        Assert.Equal(0, w[0]);
        Assert.Equal(199, w[199]);

        w.Clear();
        Assert.Empty(w);
    }

    [Fact]
    public void PenaltyWindow_ForRequest_SkipsWhenNotNeeded()
    {
        var promptTokens = new[] { 1, 2, 3 };

        // Penalty disabled → no window, so the caller samples with the params unchanged.
        var off = new SamplingParams { RepetitionPenalty = 1.0f };
        Assert.Null(PenaltyWindow.ForRequest(off, promptTokens));
        Assert.Same(off, PenaltyWindow.Bind(off, null));

        // Caller supplied its own window → the engine leaves it alone.
        var owned = new SamplingParams { RepetitionPenalty = 1.5f, PreviousTokens = new List<int> { 7 } };
        Assert.Null(PenaltyWindow.ForRequest(owned, promptTokens));

        // Penalty on → a seeded window, bound into the params for sampling.
        var on = new SamplingParams { RepetitionPenalty = 1.5f, PenaltyLastN = 2 };
        var window = PenaltyWindow.ForRequest(on, promptTokens);
        Assert.NotNull(window);
        Assert.Equal([2, 3], window.ToArray());
        Assert.Same(window, PenaltyWindow.Bind(on, window).PreviousTokens);
    }
}
