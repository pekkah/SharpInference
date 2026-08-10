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

        // Penalised: 'a' is demoted below 'b' as soon as it is emitted, and the two alternate.
        Assert.Equal("ababababab", penalised[..10]);
        Assert.Equal(0, AdjacentRepeats(penalised));
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

        Assert.Equal("ababababab", results[0]);
        Assert.Equal("aaaaaaaaaa", results[1]);
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
