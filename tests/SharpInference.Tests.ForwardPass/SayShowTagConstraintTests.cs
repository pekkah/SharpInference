using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Demo whole-turn output constraint (issue #423) implementing the grammar from the issue's own
/// example: <c>output := ( text | "&lt;say&gt;" text "&lt;/say&gt;" | "&lt;show&gt;" text "&lt;/show&gt;" )*</c>.
/// Free text passes through unconstrained; the instant a '&lt;' begins a marker it must complete to
/// a valid open tag, and once open, the tag's content is free until the matching close tag is
/// required. Operates on a byte-per-token vocabulary (token id == ASCII byte value) to keep the demo
/// self-contained -- a real consumer's grammar would walk
/// <see cref="GrammarVocabulary.TokenBytes"/> the way <c>GemmaToolArgumentConstraint</c> does for a
/// multi-byte-token vocabulary, but that machinery isn't the point of this sample (issue #423
/// explicitly leaves the output grammar itself to the consumer).
/// </summary>
public sealed class SayShowTagConstraint : ITokenConstraint
{
    private static readonly byte[] SayOpenTail = "say>"u8.ToArray();     // after the leading '<'
    private static readonly byte[] ShowOpenTail = "show>"u8.ToArray();
    private static readonly byte[] SayCloseTail = "/say>"u8.ToArray();   // after the leading '<'
    private static readonly byte[] ShowCloseTail = "/show>"u8.ToArray();

    private enum State { Outside, OpenMatch, Inside, CloseMatch }

    private readonly int _vocab;
    private float[]? _masked;

    private State _state = State.Outside;
    private int _matchLen;
    private bool _sayCandidate;
    private bool _showCandidate;
    private bool _insideSay;

    public SayShowTagConstraint(int vocabSize) => _vocab = vocabSize;

    public bool IsConstraining => _state is State.OpenMatch or State.CloseMatch;

    public void Reset()
    {
        _state = State.Outside;
        _matchLen = 0;
        _sayCandidate = false;
        _showCandidate = false;
        _insideSay = false;
    }

    public void Accept(int token)
    {
        var b = (byte)token;
        switch (_state)
        {
            case State.Outside:
                if (b == (byte)'<') { _state = State.OpenMatch; _matchLen = 0; _sayCandidate = true; _showCandidate = true; }
                return;
            case State.Inside:
                if (b == (byte)'<') { _state = State.CloseMatch; _matchLen = 0; }
                return;
            case State.OpenMatch:
                StepOpen(b);
                return;
            case State.CloseMatch:
                StepClose(b);
                return;
        }
    }

    private void StepOpen(byte b)
    {
        if (_sayCandidate && (_matchLen >= SayOpenTail.Length || SayOpenTail[_matchLen] != b)) _sayCandidate = false;
        if (_showCandidate && (_matchLen >= ShowOpenTail.Length || ShowOpenTail[_matchLen] != b)) _showCandidate = false;

        if (!_sayCandidate && !_showCandidate) { Reset(); return; }   // a sampled token the mask should have forbidden

        _matchLen++;
        if (_sayCandidate && _matchLen == SayOpenTail.Length) { _state = State.Inside; _insideSay = true; return; }
        if (_showCandidate && _matchLen == ShowOpenTail.Length) { _state = State.Inside; _insideSay = false; }
    }

    private void StepClose(byte b)
    {
        var tail = _insideSay ? SayCloseTail : ShowCloseTail;
        if (_matchLen >= tail.Length || tail[_matchLen] != b) { Reset(); return; }   // ditto
        _matchLen++;
        if (_matchLen == tail.Length) _state = State.Outside;
    }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        if (logits.Length != _vocab) return logits;   // vocab mismatch -- never wedge

        byte? b1 = null, b2 = null;
        switch (_state)
        {
            case State.OpenMatch:
                if (_sayCandidate && _matchLen < SayOpenTail.Length) b1 = SayOpenTail[_matchLen];
                if (_showCandidate && _matchLen < ShowOpenTail.Length) b2 = ShowOpenTail[_matchLen];
                break;
            case State.CloseMatch:
            {
                var tail = _insideSay ? SayCloseTail : ShowCloseTail;
                if (_matchLen < tail.Length) b1 = tail[_matchLen];
                break;
            }
            default:
                return logits;   // not constraining -- shouldn't be called, but never wedge regardless
        }

        if (b1 is null && b2 is null) return logits;   // dead state -- never wedge

        var masked = _masked ??= new float[_vocab];
        logits.CopyTo(masked);
        bool any = false;
        for (int i = 0; i < masked.Length; i++)
        {
            if ((byte)i == b1 || (byte)i == b2) { any = true; continue; }
            masked[i] = float.NegativeInfinity;
        }
        return any ? masked : logits;
    }
}

/// <summary>
/// Tests for <see cref="SayShowTagConstraint"/> (issue #423 acceptance criterion 5): standalone
/// grammar correctness, direct composition with another constraint via
/// <see cref="TokenConstraints.Combine"/>, and a full run through a real
/// <see cref="ContinuousBatchingEngine"/> -- mirroring <see cref="ContinuousBatchingConstraintTests"/>.
/// </summary>
public sealed class SayShowTagConstraintTests
{
    private const int Vocab = 128;
    private static int Tok(char c) => c;

    private static void Feed(ITokenConstraint c, string text)
    {
        foreach (char ch in text) c.Accept(Tok(ch));
    }

    private static bool Allowed(ITokenConstraint c, int tokenId)
    {
        Span<float> logits = new float[Vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    // ── Standalone grammar correctness ─────────────────────────────────────────

    [Fact]
    public void FreeText_IsNeverConstraining()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "just some ordinary text, no tags here");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void AfterAngleBracket_OnlyLegalTagStartByte_IsAllowed()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "hi<");
        Assert.True(c.IsConstraining);
        Assert.True(Allowed(c, Tok('s')));    // both "say" and "show" start with 's'
        Assert.False(Allowed(c, Tok('x')));   // no valid tag starts with 'x'
    }

    [Fact]
    public void AfterAngleS_BothSayAndShow_AreLegalContinuations()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<s");
        Assert.True(Allowed(c, Tok('a')));    // -> "say"
        Assert.True(Allowed(c, Tok('h')));    // -> "show"
        Assert.False(Allowed(c, Tok('x')));
    }

    [Fact]
    public void OnceShowIsChosen_SayContinuation_IsNoLongerLegal()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<sh");   // committed to "show" (say's 2nd byte would have been 'a', not 'h')
        Assert.True(Allowed(c, Tok('o')));    // "show" continues
        Assert.False(Allowed(c, Tok('a')));   // "say" is no longer reachable
    }

    [Fact]
    public void InsideOpenTag_ContentIsFree()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<say>anything at all");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void ClosingTag_MustMatchTheOneThatOpened()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<show>hi<");
        Assert.True(c.IsConstraining);
        Assert.True(Allowed(c, Tok('/')));
        Feed(c, "/");
        Assert.True(Allowed(c, Tok('s')));
        Feed(c, "s");
        Assert.True(Allowed(c, Tok('h')));    // "/show>" -- not "/say>"
        Assert.False(Allowed(c, Tok('a')));
    }

    [Fact]
    public void FullyClosedTag_ReturnsToFreeText()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<say>hi</say>and more free text");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void Reset_ReturnsToOutsideState()
    {
        var c = new SayShowTagConstraint(Vocab);
        Feed(c, "<sa");
        Assert.True(c.IsConstraining);
        c.Reset();
        Assert.False(c.IsConstraining);
        Assert.True(Allowed(c, Tok('x')));    // back to fully free text
    }

    // ── Direct composition (no engine) ──────────────────────────────────────────

    /// <summary>Always-constraining, single-allowed-byte mock -- stands in for an unrelated
    /// constraint (e.g. a tool-argument constraint) simultaneously active with the tag grammar.</summary>
    private sealed class SingleByteConstraint(int allowed) : ITokenConstraint
    {
        public bool IsConstraining => true;

        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
        {
            var m = new float[logits.Length];
            logits.CopyTo(m);
            for (int i = 0; i < m.Length; i++)
                if (i != allowed) m[i] = float.NegativeInfinity;
            return m;
        }

        public void Accept(int token) { }
        public void Reset() { }
    }

    [Fact]
    public void ComposesWithAnotherConstraint_CanSteerWhichBranchTheGrammarTakes()
    {
        var tag = new SayShowTagConstraint(Vocab);
        Feed(tag, "<s");   // matchLen=1: both 'a' (say) and 'h' (show) are legal per the tag grammar alone
        Assert.True(Allowed(tag, Tok('a')));
        Assert.True(Allowed(tag, Tok('h')));

        // A second, independent constraint that only allows 'h' -- narrower than the tag grammar
        // alone, so the combined mask must reflect BOTH restrictions, not just the tag grammar's.
        var combined = TokenConstraints.Combine(tag, new SingleByteConstraint(Tok('h')))!;
        var masked = combined.Filter(new float[Vocab]);
        Assert.False(float.IsNegativeInfinity(masked[Tok('h')]));
        Assert.True(float.IsNegativeInfinity(masked[Tok('a')]));

        // Accepting the forced 'h' (through the combined constraint, which forwards to `tag` too)
        // must update the tag grammar's own state: "show", not "say", is now the only reachable branch.
        combined.Accept(Tok('h'));
        Assert.True(Allowed(tag, Tok('o')));    // "show" continues
        Assert.False(Allowed(tag, Tok('a')));   // "say" is no longer reachable
    }

    // ── Full run through a real ContinuousBatchingEngine ────────────────────────

    /// <summary>Single-byte tokenizer: token id == ASCII byte (mirrors ContinuousBatchingConstraintTests;
    /// duplicated locally since Tests.ForwardPass doesn't reference Tests.Core).</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
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

    private sealed class FakeCache : ISequenceKvCache { public void Dispose() { } }

    /// <summary>
    /// Forward pass whose per-position preference is scripted by an explicit byte array (indexed by
    /// an internal call counter), rather than always preferring one fixed token -- needed because
    /// this constraint's grammar (unlike a schema constraint) reacts to WHERE in the stream a '&lt;'
    /// appears, so the test has to say "the model wants to type THIS byte at THIS position" to prove
    /// a deviation gets corrected. Single-sequence only (maxBatchSize:1): a per-sequence-aware script
    /// would need to key the step counter by cache identity, which
    /// <see cref="ContinuousBatchingConstraintTests"/> already covers generically with a uniform
    /// (position-independent) preference, so a two-sequence variant isn't duplicated here.
    /// </summary>
    private sealed class ScriptedBytePreferenceForwardPass(byte[] preferred) : IBatchedForwardPass
    {
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 1;
        public int MaxSeqLen => 8192;
        public bool PrefillDequantCacheActive => false;
        public bool SupportsBatchedGpuArgmax => true;

        private int _step;

        private float[] NextRow()
        {
            byte b = preferred[Math.Min(_step, preferred.Length - 1)];
            _step++;
            var r = new float[Vocab];
            r[b] = 1f;
            return r;
        }

        public ISequenceKvCache CreateCache() => new FakeCache();

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
            => NextRow();

        public float[]?[] PrefillPackedMulti(
            ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        {
            var outp = new float[]?[chunks.Length];
            for (int i = 0; i < chunks.Length; i++) outp[i] = wantLogits[i] ? NextRow() : null;
            return outp;
        }

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var outp = new float[tokens.Length][];
            for (int i = 0; i < tokens.Length; i++) outp[i] = NextRow();
            return outp;
        }

        public (int Token, float Logit)[] BatchForwardMultiArgmax(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            var outp = new (int, float)[tokens.Length];
            for (int i = 0; i < tokens.Length; i++)
            {
                var row = NextRow();
                outp[i] = (Array.IndexOf(row, 1f), 1f);
            }
            return outp;
        }
    }

    /// <summary>Constraining only while AcceptCount is in [engageAfter, engageAfter+forcedLen); while
    /// constraining, forbids every token except forcedByte. Stands in for an unrelated constraint
    /// (e.g. a tool-argument constraint) simultaneously active with the tag grammar.</summary>
    private sealed class FixedByteWindowConstraint(int engageAfter, int forcedLen, int forcedByte) : ITokenConstraint
    {
        private int _acceptCount;
        public bool IsConstraining => _acceptCount >= engageAfter && _acceptCount < engageAfter + forcedLen;

        public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
        {
            var m = new float[logits.Length];
            logits.CopyTo(m);
            for (int i = 0; i < m.Length; i++)
                if (i != forcedByte) m[i] = float.NegativeInfinity;
            return m;
        }

        public void Accept(int token) => _acceptCount++;
        public void Reset() => _acceptCount = 0;
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
    public async Task ComposedWithAnotherConstraint_ThroughRealEngine_ProducesValidBalancedTags()
    {
        // Scripted "model preference" per position: mostly an inert filler byte ('Z'), except where
        // the trace needs a deliberate choice -- index 2 opens a tag, index 4 is the say/show
        // divergence point (deliberately preferring 'a' / "say" to prove the second constraint's
        // override below), index 10 opens the close tag. Every other forced-grammar position is
        // overridden regardless of what's scripted here (only one legal byte survives at each), so
        // 'Z' elsewhere is inert padding.
        var preferred = new byte[24];
        Array.Fill(preferred, (byte)'Z');
        preferred[2] = (byte)'<';
        preferred[4] = (byte)'a';
        preferred[10] = (byte)'<';

        var tag = new SayShowTagConstraint(Vocab);
        // Stands in for an unrelated (e.g. tool-argument) constraint active only at the say/show
        // divergence step, forcing 'h' -- narrower than the tag grammar's own {'a','h'} there.
        var forceShow = new FixedByteWindowConstraint(engageAfter: 4, forcedLen: 1, forcedByte: Tok('h'));
        var sp = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 17,
            Constraint = TokenConstraints.Combine(tag, forceShow),
        };

        using var engine = new ContinuousBatchingEngine(
            new ScriptedBytePreferenceForwardPass(preferred), new CharTokenizer(), "test", maxBatchSize: 1);

        string text = await RunOne(engine, "prompt", sp);

        // The model's own preference at the divergence point wanted "say" ('a'); the second
        // constraint forced 'h' instead -- so the grammar committed to "show", and correctly
        // required "/show>" (not "/say>") to close, despite every unforced byte being filler 'Z'.
        Assert.Equal("ZZ<show>ZZ</show>", text);
        Assert.False(tag.IsConstraining);   // ended back at the Outside/free state -- balanced
    }
}
