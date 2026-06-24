using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Model-independent wiring tests for grammar-constrained tool-argument decoding under continuous
/// batching (issue #377). A scripted <see cref="ITokenConstraint"/> and a canned
/// <see cref="IBatchedForwardPass"/> stand in for the real Gemma grammar + GPU forward pass, so the
/// per-sequence mask/Accept plumbing is exercised in CI without a GGUF. The grammar correctness
/// itself is covered by <c>ToolGrammarMockTests</c>/<c>ToolGrammarConstraintTests</c>; here we only
/// assert that <see cref="ContinuousBatchingEngine"/> (a) masks a constraining sequence's logits row,
/// (b) advances the constraint on every token, and (c) leaves co-tenant unconstrained sequences
/// byte-identical to the no-constraint path.
/// </summary>
public sealed class ContinuousBatchingConstraintTests
{
    // The forward pass always makes 'X' the unconstrained argmax; the constraint, while engaged,
    // forces 'Y' instead. So a constrained span reads 'Y' and an unconstrained span reads 'X'.
    private const int Vocab = 128;
    private static int Tok(char c) => c;
    private const int PreferX = 'X';
    private const int ForceY = 'Y';

    /// <summary>Single-byte tokenizer: token id == ASCII byte. EOG is an id no forward ever emits.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 0;
        public int EosTokenId => 0;          // NUL — the canned forward never produces it
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } =
            new Dictionary<string, int>(StringComparer.Ordinal);

        public byte[] DecodeBytes(int token) =>
            token is > 0 and < Vocab ? [(byte)token] : [];

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

    /// <summary>
    /// Begins constraining once <paramref name="EngageAfter"/> tokens have been accepted and stays
    /// constraining for <paramref name="ForcedLen"/> tokens, masking every logit except
    /// <see cref="ForceY"/>. Records Accept/Filter counts so the wiring can be asserted.
    /// </summary>
    private sealed class ScriptedConstraint(int engageAfter, int forcedLen) : ITokenConstraint
    {
        private float[]? _masked;
        public int AcceptCount { get; private set; }
        public int FilterCount { get; private set; }
        public int ResetCount { get; private set; }

        public bool IsConstraining => AcceptCount >= engageAfter && AcceptCount < engageAfter + forcedLen;

        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
        {
            FilterCount++;
            var m = _masked ??= new float[logits.Length];
            if (m.Length != logits.Length) return logits;
            logits.CopyTo(m);
            for (int i = 0; i < m.Length; i++)
                if (i != ForceY) m[i] = float.NegativeInfinity;
            return m;
        }

        public void Accept(int token) => AcceptCount++;
        public void Reset() { AcceptCount = 0; ResetCount++; }
    }

    private sealed class FakeCache : ISequenceKvCache { public void Dispose() { } }

    /// <summary>Canned forward pass: every sequence's next-token logits put 'X' on top. Records the
    /// widest batch it was ever asked to decode so a test can confirm two sequences actually co-resided
    /// in one batched step (the per-sequence-vs-batch-wide masking guarantee).</summary>
    private sealed class FakeBatchedForwardPass : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 8192;
        public bool PrefillDequantCacheActive => false;
        public bool SupportsBatchedGpuArgmax => true;

        private int _maxBatchWidth;
        public int MaxBatchWidth => Volatile.Read(ref _maxBatchWidth);
        private void RecordWidth(int w) { if (w > _maxBatchWidth) _maxBatchWidth = w; }

        private static float[] Row()
        {
            var r = new float[Vocab];   // all zeros…
            r[PreferX] = 1f;            // …except 'X', the unconstrained argmax
            return r;
        }

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
            RecordWidth(tokens.Length);
            var outp = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++) outp[i] = Row();
            return outp;
        }

        public (int Token, float Logit)[] BatchForwardMultiArgmax(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            RecordWidth(tokens.Length);
            var outp = new (int, float)[tokens.Length];
            for (int i = 0; i < tokens.Length; i++) outp[i] = (PreferX, 1f);
            return outp;
        }
    }

    private static async Task<string> RunOne(ContinuousBatchingEngine engine, string prompt, SamplingParams sp)
    {
        var sb = new StringBuilder();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        await foreach (var s in engine.GenerateAsync(prompt, sp, cts.Token))
            sb.Append(s);
        return sb.ToString();
    }

    [Fact]
    public async Task ConstrainedSequence_IsMasked_WhileConstraining()
    {
        using var engine = new ContinuousBatchingEngine(
            new FakeBatchedForwardPass(), new CharTokenizer(), "test", maxBatchSize: 1);

        // Engage after the first token, force 'Y' for 3 tokens, then release. With 6 emitted tokens
        // (first + 5 decode; the 6th decode sample is dropped at the MaxNewTokens cutoff) the stream
        // is: X (unconstrained first) · YYY (masked) · XX (released → model's preferred 'X' again).
        var scripted = new ScriptedConstraint(engageAfter: 1, forcedLen: 3);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6, Constraint = scripted };

        string text = await RunOne(engine, "prompt", sp);

        Assert.Equal("XYYYXX", text);
        // Reset at admission + Accept on every emitted/sampled token (first + 6 decode samples).
        Assert.Equal(1, scripted.ResetCount);
        Assert.True(scripted.AcceptCount >= 6, $"expected ≥6 accepts, got {scripted.AcceptCount}");
        Assert.Equal(3, scripted.FilterCount);  // masked exactly the 3 constrained steps
    }

    [Fact]
    public async Task NoConstraint_IsUnaffected()
    {
        using var engine = new ContinuousBatchingEngine(
            new FakeBatchedForwardPass(), new CharTokenizer(), "test", maxBatchSize: 1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };   // Constraint == null
        string text = await RunOne(engine, "prompt", sp);

        Assert.Equal("XXXXXX", text);   // pure unconstrained greedy = the forward's argmax every step
    }

    [Fact]
    public async Task ConstrainedAndUnconstrained_Coexist_PerSequenceMasking()
    {
        var fake = new FakeBatchedForwardPass();
        using var engine = new ContinuousBatchingEngine(fake, new CharTokenizer(), "test", maxBatchSize: 2);

        // The constraint engages only after 5 accepted tokens, by which point BOTH sequences are
        // co-resident in the batched decode step, so masking the constrained row while the co-tenant
        // is in the SAME BatchForwardMulti call directly exercises per-sequence (not batch-wide)
        // masking. Stream A: XXXXX (pre-engage) · YYY (masked) · XXXX (released). Stream B: all X.
        var spConstrained = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 12,
            Constraint = new ScriptedConstraint(engageAfter: 5, forcedLen: 3),
        };
        var spPlain = new SamplingParams { Temperature = 0f, MaxNewTokens = 12 };

        var (a, b) = await RunBoth(engine, ("alpha", spConstrained), ("beta", spPlain));

        // The constrained request is masked only in its engaged span; the co-tenant sharing the batch
        // is byte-identical to the no-constraint path — proof the mask is applied per logits row.
        Assert.Equal("XXXXXYYYXXXX", a);
        Assert.Equal("XXXXXXXXXXXX", b);
        // …and the two genuinely shared a batched decode step (else the above would be vacuous).
        Assert.Equal(2, fake.MaxBatchWidth);
    }

    /// <summary>
    /// Drives two requests so they are enqueued back-to-back BEFORE either is drained — the batcher
    /// then admits both in one pass and decodes them in a single batched step, making the co-residency
    /// the masking test depends on reliable rather than scheduling-dependent.
    /// </summary>
    private static async Task<(string a, string b)> RunBoth(
        ContinuousBatchingEngine engine, (string prompt, SamplingParams sp) a, (string prompt, SamplingParams sp) b)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var ea = engine.GenerateAsync(a.prompt, a.sp, cts.Token).GetAsyncEnumerator(cts.Token);
        var eb = engine.GenerateAsync(b.prompt, b.sp, cts.Token).GetAsyncEnumerator(cts.Token);

        // Kick both iterators before awaiting: GenerateChunksAsync enqueues its request synchronously
        // at the top of the method, so starting both MoveNextAsync calls queues both requests before
        // the first result is consumed.
        var ma = ea.MoveNextAsync();
        var mb = eb.MoveNextAsync();
        bool ha = await ma, hb = await mb;

        var sa = new StringBuilder();
        var sb = new StringBuilder();
        try
        {
            while (ha || hb)
            {
                if (ha) { sa.Append(ea.Current); ha = await ea.MoveNextAsync(); }
                if (hb) { sb.Append(eb.Current); hb = await eb.MoveNextAsync(); }
            }
        }
        finally
        {
            await ea.DisposeAsync();
            await eb.DisposeAsync();
        }
        return (sa.ToString(), sb.ToString());
    }
}
