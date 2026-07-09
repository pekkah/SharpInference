using System.Text;

namespace SharpInference.Core.Grammar;

/// <summary>
/// The tool-call envelope a <see cref="JsonToolArgumentConstraint"/> targets — i.e. how the model
/// wraps the call and where the argument object lives relative to the tool name.
/// </summary>
internal enum JsonToolEnvelope
{
    /// <summary>
    /// The call is one JSON object whose <c>name</c> field selects the tool and whose
    /// <c>arguments</c>/<c>parameters</c> field is the argument object — Qwen
    /// (<c>&lt;tool_call&gt;{"name":"X","arguments":{…}}&lt;/tool_call&gt;</c>) and Llama-3
    /// (<c>&lt;|python_tag|&gt;{"name":"X","parameters":{…}}</c>).
    /// </summary>
    NameValueObject,

    /// <summary>
    /// The tool name is bare text terminated by a separator token, after which the argument object
    /// is emitted directly — DeepSeek
    /// (<c>&lt;|tool_call_begin|&gt;NAME&lt;|tool_sep|&gt;{…}&lt;|tool_call_end|&gt;</c>).
    /// </summary>
    NameThenSeparator,
}

/// <summary>
/// Constrained-decoding state machine (issue #376) for the families that emit <b>standard JSON</b>
/// tool-call arguments — Qwen (<c>&lt;tool_call&gt;{json}&lt;/tool_call&gt;</c>), Llama-3
/// (<c>&lt;|python_tag|&gt;{json}</c>), and DeepSeek
/// (<c>NAME&lt;|tool_sep|&gt;{json}</c>) — the JSON sibling of
/// <see cref="GemmaToolArgumentConstraint"/>. It watches the emitted stream until a call opens and a
/// known tool's argument object begins, then constrains that object so only declared keys appear,
/// every required key appears exactly once, each value matches its declared shape (JSON strings in
/// double quotes with free escaped content, bare numbers/booleans/null, <c>[…]</c> arrays and nested
/// <c>{…}</c> objects), and enum-typed values stay in the declared set. The closing <c>}</c> ends
/// constraint and the machine returns to watching, so the trailing envelope and any later text are
/// unconstrained.
///
/// <para>
/// Unlike Gemma's bespoke wire format — where the string delimiter is a single <c>&lt;|"|&gt;</c>
/// special token — JSON's structural bytes (<c>{ } [ ] " : ,</c>) are ordinary bytes the model's BPE
/// freely merges with content and whitespace (e.g. one token carries <c>{"</c>, another <c>":</c>,
/// another <c>"}}</c>, and the empty object is a single <c>{}</c> token). So matching is entirely at
/// the byte level: a token is permitted iff replaying its bytes from the current state keeps the
/// machine alive. The constraint engages the instant the argument key's colon is seen — BEFORE the
/// <c>{</c> — so a merged <c>{}</c> token can't drop the required arguments.
/// </para>
///
/// <para>Default-off byte-identical: if no supplied tool is fully constrainable, or the structural
/// open-marker token isn't in the vocabulary, the constraint is inert and generation is unconstrained.</para>
/// </summary>
public sealed class JsonToolArgumentConstraint : ITokenConstraint
{
    private const int MaxDepth = 32;        // nesting cap; deeper schemas aren't constrained
    private const int MaxPreambleScan = 512; // give up watching a call whose preamble runs this long

    private readonly GrammarVocabulary _vocab;
    private readonly Dictionary<string, CompiledObject> _tools;
    private readonly HashSet<int> _forbidden;       // EOG ids — never legal inside the argument object
    private readonly int _openMarkerId;             // arms watching (e.g. <tool_call> / <|python_tag|>)
    private readonly int _separatorId;              // NameThenSeparator: ends the bare name (DeepSeek)
    private readonly JsonToolEnvelope _envelope;
    private readonly byte[][] _argsKeys;            // NameValueObject: "arguments"/"parameters" bytes
    private static readonly byte[] s_nameKey = Encoding.UTF8.GetBytes("name");

    // Whole-body mode (issue #423 follow-up, JsonSchemaOutputConstraint): when set, there is no
    // envelope/preamble at all -- the object is constrained from the very first token. Reset() must
    // re-engage immediately rather than return to the watching-idle state (see Reset() below).
    private readonly CompiledObject? _wholeBodyRoot;

    // Whole-body mode only: set once the root object closes successfully. From here on Filter()
    // forces EOG-only (no further content is legal) instead of returning to the tool-argument path's
    // "watching" idle state -- "the ENTIRE response" means nothing may follow the object.
    private bool _wholeBodyDone;

    // Watching / preamble state.
    private bool _armed;                            // saw the open marker, scanning the preamble
    private int _watchState;                        // preamble FSM state (W* constants)
    private readonly StringBuilder _keyBuf = new();  // current preamble key
    private readonly StringBuilder _nameBuf = new(); // captured tool name
    private bool _wEscaped;                          // preamble string escape flag
    private int _skipDepth;                          // WSkipValue brace/bracket balance
    private int _skipKind;                           // WSkipValue: 0=undecided 1=string 2=balanced 3=bare
    private int _preambleLen;                        // bytes scanned since arming (runaway guard)

    // Constraining state: an explicit frame stack (pushdown automaton).
    private Frame[] _stack = new Frame[MaxDepth];
    private Frame[] _scratch = new Frame[MaxDepth];
    private int _depth;

    // Masking scratch — a reusable full-vocab logits buffer (allocated on first constrained step).
    private float[]? _masked;
    private readonly bool[] _firstByteOk = new bool[256];

    internal JsonToolArgumentConstraint(
        GrammarVocabulary vocab,
        IReadOnlyList<ToolSchema> tools,
        string openMarker,
        JsonToolEnvelope envelope,
        IReadOnlyList<string>? argsKeys = null,
        string? separatorMarker = null)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentException.ThrowIfNullOrEmpty(openMarker);
        _vocab = vocab;
        _envelope = envelope;
        _argsKeys = (argsKeys ?? ["arguments", "parameters"]).Select(Encoding.UTF8.GetBytes).ToArray();

        _forbidden = new HashSet<int>(vocab.EogTokenIds);
        _ = vocab.TryGetSpecialToken(openMarker, out _openMarkerId);
        _separatorId = -1;
        if (separatorMarker is { Length: > 0 })
            _ = vocab.TryGetSpecialToken(separatorMarker, out _separatorId);

        _tools = new Dictionary<string, CompiledObject>(StringComparer.Ordinal);
        bool haveMarker = _openMarkerId > 0
            && (_envelope != JsonToolEnvelope.NameThenSeparator || _separatorId > 0);
        if (haveMarker)
        {
            foreach (var t in tools)
            {
                if (t.Arguments.Open) continue;                 // unconstrained body — skip
                var compiled = ToolSchemaCompiler.TryCompileObject(t.Arguments);
                if (compiled is not null) _tools[t.Name] = compiled;
            }
        }
        else if (tools.Count > 0)
        {
            // The caller asked for a constraint but this vocabulary doesn't define the family's
            // structural open marker — the constraint is inert. Surface it once so an operator who
            // enabled SHARPI_TOOL_GRAMMAR on a mismatched model isn't left wondering why arguments
            // are still unconstrained.
            WarnOnce($"no-open-marker:{openMarker}",
                $"open marker '{openMarker}' not found in this vocabulary — tool-grammar is inert for "
                + "this model (arguments generate unconstrained).");
        }
    }

    /// <summary>
    /// Whole-body mode (issue #423 follow-up): constrains the ENTIRE output to
    /// <paramref name="wholeBodyRoot"/> from the first token — no tool-call envelope, no preamble.
    /// Used via <see cref="ForWholeBody"/> by <see cref="JsonSchemaOutputConstraint"/>.
    /// </summary>
    private JsonToolArgumentConstraint(GrammarVocabulary vocab, CompiledObject wholeBodyRoot)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(wholeBodyRoot);
        _vocab = vocab;
        _envelope = JsonToolEnvelope.NameValueObject;   // unused in this mode
        _argsKeys = [];
        _forbidden = new HashSet<int>(vocab.EogTokenIds);
        _openMarkerId = 0;
        _separatorId = -1;
        _tools = new Dictionary<string, CompiledObject>(StringComparer.Ordinal);
        _wholeBodyRoot = wholeBodyRoot;
        Reset();
    }

    /// <summary>Builds a whole-body constraint (see the private ctor above) for <see cref="JsonSchemaOutputConstraint"/>.</summary>
    internal static JsonToolArgumentConstraint ForWholeBody(GrammarVocabulary vocab, CompiledObject root) =>
        new(vocab, root);

    /// <summary>Whether this constraint can ever restrict anything (any tool was constrainable, or
    /// whole-body mode is active).</summary>
    public bool HasConstrainableTools => _tools.Count > 0 || _wholeBodyRoot is not null;

    public bool IsConstraining => _depth > 0 || _wholeBodyDone;

    public void Reset()
    {
        _armed = false;
        _depth = 0;
        _wholeBodyDone = false;
        ResetPreamble();
        // Whole-body mode has no envelope to watch for -- re-engage immediately so the very first
        // token of the (new) generation is already constrained, rather than going permanently inert.
        if (_wholeBodyRoot is not null) PushObject(_wholeBodyRoot, OExpectOpenBrace);
    }

    private void ResetPreamble()
    {
        _watchState = WStart;
        _keyBuf.Clear();
        _nameBuf.Clear();
        _wEscaped = false;
        _skipDepth = 0;
        _skipKind = 0;
        _preambleLen = 0;
    }

    // ── Frame stack ───────────────────────────────────────────────────────────

    private enum FK : byte { Object, Array, Str, StrEnum, Num, Lit, Free }

    private struct Frame
    {
        public FK Kind;
        public int State;
        public CompiledObject? Obj;     // Object frame
        public CompiledNode? Node;      // Array (item), StrEnum / Lit (literals), Num
        public ulong Emitted;           // Object: keys emitted
        public ulong Cand;              // Object key-match / StrEnum / Lit candidates
        public int MatchLen;            // chars into current key / literal
        public int PendingKey;          // Object: key index whose value to push at ':'
        public int FreeDepth;           // Free: nesting balance of {}/[] in a free value
        public bool SeenDigit;          // Num
        public bool SeenDot;            // Num
        public bool SeenSign;           // Num
        public bool Escaped;            // Str / Free content: previous byte was '\'
    }

    // Object sub-states.
    private const int OExpectOpenBrace = 0;   // engaged before '{' (early-engage path)
    private const int OExpectKeyOrClose = 1;  // '"' to open a key, or '}' if all required emitted
    private const int OExpectKey = 2;         // after ',', a key is required ('}' not allowed)
    private const int OKeyContent = 3;        // matching a key name (byte-level); '"' closes
    private const int OExpectColon = 4;       // ':'
    private const int OExpectCommaOrClose = 5;

    // Array sub-states.
    private const int AExpectOpen = 0;        // '['
    private const int AExpectItemOrClose = 1;
    private const int AExpectCommaOrClose = 2;
    private const int AExpectItem = 3;        // after ',', an item is required

    // Str sub-states.
    private const int SExpectOpen = 0;        // open '"'
    private const int SContent = 1;           // free content until unescaped '"'

    // StrEnum sub-states.
    private const int SeExpectOpen = 0;       // open '"'
    private const int SeMatch = 1;            // match enum bytes; '"' closes

    // Num sub-states.
    private const int NStart = 0;
    private const int NIntDigits = 1;
    private const int NFracStart = 2;
    private const int NFracDigits = 3;

    // Lit sub-state.
    private const int LMatch = 0;

    // Free-value sub-states (issue #378): a permissive value of unknown type, balanced to completion.
    private const int FrStart = 0;       // value not yet started
    private const int FrStr = 1;         // a top-level string value
    private const int FrBare = 2;        // a bare scalar — ends at a top-level delimiter
    private const int FrBalanced = 3;    // inside {…}/[…], FreeDepth ≥ 1
    private const int FrBalancedStr = 4; // a string inside a balanced free value

    // Preamble (watching) sub-states.
    private const int WStart = 0;        // before envelope '{' (NameValueObject) / scanning name (NameThenSeparator)
    private const int WKeyExpect = 1;    // '"' opens a key, '}' gives up
    private const int WKeyContent = 2;   // accumulate key bytes until '"'
    private const int WColon = 3;        // ':'
    private const int WValueStart = 4;   // dispatch on the captured key
    private const int WNameString = 5;   // read the "name" string value
    private const int WSkipValue = 6;    // skip an unrelated key's value (balanced)
    private const int WAfterValue = 7;   // ',' continues, '}' gives up

    private void PushObject(CompiledObject obj, int state)
    {
        ref var f = ref _stack[_depth++];
        f = default;
        f.Kind = FK.Object; f.State = state; f.Obj = obj; f.Emitted = 0;
    }

    private bool PushValue(CompiledNode node)
    {
        if (_depth >= MaxDepth) return false;
        if (node.Kind == JsonSchemaKind.Object)
        {
            if (node.Object is null) return false;
            PushObject(node.Object, OExpectOpenBrace);
            return true;
        }
        ref var f = ref _stack[_depth++];
        f = default;
        f.Node = node;
        switch (node.Kind)
        {
            case JsonSchemaKind.String:
                if (node.Literals is not null) { f.Kind = FK.StrEnum; f.State = SeExpectOpen; f.Cand = AllBits(node.Literals.Length); }
                else { f.Kind = FK.Str; f.State = SExpectOpen; }
                break;
            case JsonSchemaKind.Array:
                f.Kind = FK.Array; f.State = AExpectOpen;
                break;
            case JsonSchemaKind.Any:                        // free value (issue #378)
                f.Kind = FK.Free; f.State = FrStart;
                break;
            default: // Number / Integer / Boolean / Null
                if (node.Literals is not null) { f.Kind = FK.Lit; f.State = LMatch; f.Cand = AllBits(node.Literals.Length); }
                else { f.Kind = FK.Num; f.State = NStart; }
                break;
        }
        return true;
    }

    private static ulong AllBits(int n) => n >= 64 ? ulong.MaxValue : (1UL << n) - 1;

    // ── Public lifecycle ──────────────────────────────────────────────────────

    public void Accept(int token)
    {
        if (_wholeBodyDone)
        {
            // Terminal state: Filter() only ever offered EOG ids here, so there's nothing further
            // to track regardless of what was actually sampled.
            return;
        }
        if (IsConstraining)
        {
            bool ok = !_forbidden.Contains(token) && RunToken(token);
            // The engine draws from the masked logits, so a permitted token must replay cleanly. A
            // rejection here is an invariant violation (a Filter/Accept divergence); flag it once and
            // stop constraining either way (a normal end-of-object is ok && _depth == 0).
            if (!ok)
                WarnOnce("accept-divergence",
                    "a sampled token permitted by the mask was rejected by the grammar — constraint "
                    + "disabled for the rest of this call (possible Filter/Accept divergence).");
            if (ok && _depth == 0 && _wholeBodyRoot is not null)
            {
                // Whole-body mode's root object closed successfully: force EOG-only for the rest of
                // the turn instead of returning to the tool-argument path's "watching" idle state,
                // which has no envelope to (re-)arm on in this mode anyway.
                _wholeBodyDone = true;
            }
            else if (!ok || _depth == 0) { _depth = 0; _armed = false; ResetPreamble(); }
            return;
        }

        // Watching.
        if (!_armed)
        {
            if (token == _openMarkerId) { _armed = true; ResetPreamble(); }
            return;
        }

        // NameThenSeparator: the name is bare text up to the separator token, then the argument
        // object is emitted directly — engage on the separator so the NEXT token (the object open)
        // is already masked.
        if (_envelope == JsonToolEnvelope.NameThenSeparator)
        {
            if (token == _separatorId)
            {
                if (EngageOnTool()) _armed = false; else Disarm();
                return;
            }
            AppendName(_vocab.TokenBytes(token));
            if (_nameBuf.Length > MaxPreambleScan) Disarm();
            return;
        }

        // Walk the token's bytes through the preamble FSM; engage at the argument key's colon so the
        // following token (which opens the object) is masked — a merged "{}" can't slip the required
        // arguments past. The colon is consumed by the watcher; the rest of this token replays into
        // the constrained machine (covers a tokenizer that merges ':' with the opening '{').
        var bytes = _vocab.TokenBytes(token);
        for (int i = 0; i < bytes.Length; i++)
        {
            if (++_preambleLen > MaxPreambleScan) { Disarm(); return; }
            var r = WatchByte(bytes[i]);
            if (r == WatchResult.Disarm) { Disarm(); return; }
            if (r == WatchResult.Engage)
            {
                _armed = false;
                if (i + 1 < bytes.Length && !FeedRawBytes(bytes[(i + 1)..]))
                {
                    _depth = 0;
                    WarnOnce("trailing-replay",
                        "could not replay argument bytes after the key colon — arguments generate "
                        + "unconstrained for this call.");
                }
                return;
            }
        }
    }

    private void AppendName(ReadOnlySpan<byte> bytes)
    {
        for (int i = 0; i < bytes.Length; i++) _nameBuf.Append((char)bytes[i]);
    }

    private void Disarm() { _armed = false; ResetPreamble(); }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        if (_wholeBodyDone) return FilterEogOnly(logits);
        if (_depth == 0) return logits;

        var masked = _masked ??= new float[_vocab.VocabSize];
        if (masked.Length != logits.Length) return logits;     // vocab mismatch — never wedge
        logits.CopyTo(masked);

        int allowedCount = ComputeMask(masked);
        if (allowedCount == 0)
        {
            ref var top = ref _stack[_depth - 1];
            WarnOnce($"dead-state:{top.Kind}:{top.State}",
                $"grammar reached a dead state (no legal token) at kind={top.Kind} state={top.State} "
                + "— tool arguments continue unconstrained from here.");
            return logits;
        }
        return masked;
    }

    /// <summary>Whole-body mode's terminal state (after the root object closed): only an
    /// end-of-generation token is legal, so nothing can follow the object. Reuses
    /// <see cref="_forbidden"/> (the EOG id set) as the allow-list here, inverted from its normal
    /// "never legal inside the object" meaning.</summary>
    private ReadOnlySpan<float> FilterEogOnly(ReadOnlySpan<float> logits)
    {
        var masked = _masked ??= new float[_vocab.VocabSize];
        if (masked.Length != logits.Length) return logits;     // vocab mismatch — never wedge

        // _forbidden (the EOG id set) is tiny (typically 1-5 ids) against a vocab that can be
        // 150k+ tokens -- fill once (vectorized) and sparsely restore just the EOG logits, instead
        // of a HashSet.Contains check per vocab entry on this per-token hot path.
        Array.Fill(masked, float.NegativeInfinity);
        bool anyEog = false;
        foreach (int id in _forbidden)
        {
            if ((uint)id >= (uint)masked.Length) continue;
            masked[id] = logits[id];
            anyEog = true;
        }
        // No EOG id in this vocabulary (shouldn't happen for a real tokenizer): never wedge.
        return anyEog ? masked : logits;
    }

    /// <summary>Sets every forbidden token to -inf in <paramref name="buf"/>; returns the count kept.</summary>
    private int ComputeMask(float[] buf)
    {
        Array.Clear(_firstByteOk);
        CollectFirstBytes(_depth - 1, _firstByteOk);

        // Fast-path free string content. Inside a JSON string value (SContent) the close delimiter and
        // escape are ordinary bytes, so CollectStr marks all 256 first-bytes — without this every step
        // would SimulateToken (frame copy + byte walk) the WHOLE vocabulary (150k+ tokens on Qwen/Llama),
        // costing ~seconds/token. But a token that contains neither '"' nor '\' is pure content that
        // can only keep the string open, so it's legal without simulation; only tokens that could close
        // or escape need the full replay. Disabled mid-escape (the next byte is consumed literally), so
        // those tokens fall back to SimulateToken. This is the byte-level analogue of the Gemma sibling's
        // token-level free-content shortcut.
        bool fastStrContent, fastFree;
        {
            ref var top = ref _stack[_depth - 1];
            fastStrContent = top.Kind == FK.Str && top.State == SContent && !top.Escaped;
            // A free value (issue #378) marks all 256 first-bytes too; a token carrying none of the
            // structural bytes that can balance / delimit / quote it is pure content valid in any free
            // state, so it skips SimulateToken — same shape as the string-content fast-path.
            fastFree = top.Kind == FK.Free && !top.Escaped;
        }

        int kept = 0;
        for (int id = 0; id < buf.Length; id++)
        {
            var bytes = _vocab.TokenBytes(id);
            // Empty-byte tokens (EOG / control) never advance the structure — forbidding them keeps
            // an end-of-generation token from truncating the call mid-object.
            if (bytes.Length == 0 || !_firstByteOk[bytes[0]]) { buf[id] = float.NegativeInfinity; continue; }
            bool ok = (fastStrContent && !ContainsQuoteOrBackslash(bytes))
                   || (fastFree && !ContainsFreeStructural(bytes))
                   || SimulateToken(id);
            if (ok) kept++;
            else buf[id] = float.NegativeInfinity;
        }

        // Belt-and-suspenders: forbid every EOG id regardless of its bytes (a tokenizer whose EOS
        // decodes to ordinary text could otherwise pass first-byte pruning inside a free string).
        foreach (int id in _forbidden)
            if ((uint)id < (uint)buf.Length && !float.IsNegativeInfinity(buf[id]))
            { buf[id] = float.NegativeInfinity; kept--; }

        return kept;
    }

    private static bool ContainsQuoteOrBackslash(ReadOnlySpan<byte> bytes)
    {
        for (int i = 0; i < bytes.Length; i++)
            if (bytes[i] is (byte)'"' or (byte)'\\') return true;
        return false;
    }

    /// <summary>Whether a token carries any byte that could balance, delimit, quote, or escape a free
    /// value (and so must be fully simulated rather than fast-pathed as pure content).</summary>
    private static bool ContainsFreeStructural(ReadOnlySpan<byte> bytes)
    {
        for (int i = 0; i < bytes.Length; i++)
            if (bytes[i] is (byte)'{' or (byte)'}' or (byte)'[' or (byte)']'
                         or (byte)'"' or (byte)'\\' or (byte)',') return true;
        return false;
    }

    private bool SimulateToken(int token)
    {
        int savedDepth = _depth;
        for (int i = 0; i < savedDepth; i++) _scratch[i] = _stack[i];
        bool ok = FeedRawBytes(_vocab.TokenBytes(token));
        for (int i = 0; i < savedDepth; i++) _stack[i] = _scratch[i];
        _depth = savedDepth;
        return ok;
    }

    // ── Token execution ───────────────────────────────────────────────────────

    private bool RunToken(int tokenId) => _depth != 0 && FeedRawBytes(_vocab.TokenBytes(tokenId));

    /// <summary>Byte-walks a raw byte span through the structural automaton, mutating the stack.</summary>
    private bool FeedRawBytes(ReadOnlySpan<byte> bytes)
    {
        int i = 0;
        while (i < bytes.Length)
        {
            if (_depth == 0) return false;                 // closed the object but bytes remain
            ref var top = ref _stack[_depth - 1];
            var r = StepByte(ref top, bytes[i]);
            switch (r)
            {
                case Step.Consume: i++; break;
                case Step.Retry: break;                    // frame pushed/popped; re-evaluate top
                default: return false;                     // Reject
            }
        }
        return true;
    }

    private enum Step { Consume, Retry, Reject }

    private Step StepByte(ref Frame top, byte b)
    {
        switch (top.Kind)
        {
            case FK.Object: return StepObject(ref top, b);
            case FK.Array: return StepArray(ref top, b);
            case FK.Str: return StepStr(ref top, b);
            case FK.StrEnum: return StepStrEnum(ref top, b);
            case FK.Num: return StepNum(ref top, b);
            case FK.Lit: return StepLit(ref top, b);
            case FK.Free: return StepFree(ref top, b);
            default: return Step.Reject;
        }
    }

    /// <summary>Free-value (issue #378): accept any single well-formed JSON value — a string, a bare
    /// scalar, or a balanced {…}/[…] — and pop when it completes, so the enclosing object resumes and
    /// keeps enforcing its declared/required keys. The value's contents are unconstrained.</summary>
    private Step StepFree(ref Frame f, byte b)
    {
        switch (f.State)
        {
            case FrStart:
                if (IsWs(b)) return Step.Consume;
                if (b is (byte)'{' or (byte)'[') { f.FreeDepth = 1; f.State = FrBalanced; return Step.Consume; }
                if (b == (byte)'"') { f.State = FrStr; return Step.Consume; }
                if (b is (byte)',' or (byte)'}' or (byte)']') return Step.Reject;   // a value can't be empty
                f.State = FrBare; return Step.Consume;                              // bare scalar start

            case FrStr:
                if (f.Escaped) { f.Escaped = false; return Step.Consume; }
                if (b == (byte)'\\') { f.Escaped = true; return Step.Consume; }
                if (b == (byte)'"') { _depth--; return PostValueOrDone(); }         // string closes the value
                return Step.Consume;

            case FrBare:
                if (b is (byte)',' or (byte)'}' or (byte)']') { _depth--; return PostValueRetry(); }
                return Step.Consume;

            case FrBalanced:
                if (b == (byte)'"') { f.State = FrBalancedStr; return Step.Consume; }
                if (b is (byte)'{' or (byte)'[') { f.FreeDepth++; return Step.Consume; }
                if (b is (byte)'}' or (byte)']')
                {
                    if (--f.FreeDepth == 0) { _depth--; return PostValueOrDone(); }
                    return Step.Consume;
                }
                return Step.Consume;

            case FrBalancedStr:
                if (f.Escaped) { f.Escaped = false; return Step.Consume; }
                if (b == (byte)'\\') { f.Escaped = true; return Step.Consume; }
                if (b == (byte)'"') { f.State = FrBalanced; return Step.Consume; }
                return Step.Consume;

            default:
                return Step.Reject;
        }
    }

    private static bool IsWs(byte b) => b is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\r';

    // After a value frame pops, resume the parent (object/array) at its post-value state.
    private bool PostValue()
    {
        if (_depth == 0) return true;
        ref var parent = ref _stack[_depth - 1];
        if (parent.Kind == FK.Object) { parent.State = OExpectCommaOrClose; return true; }
        if (parent.Kind == FK.Array) { parent.State = AExpectCommaOrClose; return true; }
        return false;
    }

    private Step PostValueOrDone()
    {
        if (_depth == 0) return Step.Consume;              // top-level object done
        return PostValue() ? Step.Consume : Step.Reject;
    }

    private Step PostValueRetry() => PostValue() ? Step.Retry : Step.Reject;

    private Step StepObject(ref Frame f, byte b)
    {
        var obj = f.Obj!;
        switch (f.State)
        {
            case OExpectOpenBrace:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'{') { f.State = OExpectKeyOrClose; return Step.Consume; }
                return Step.Reject;

            case OExpectKeyOrClose:
            case OExpectKey:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'}' && f.State == OExpectKeyOrClose)
                {
                    if ((f.Emitted & obj.RequiredMask) != obj.RequiredMask) return Step.Reject;
                    _depth--; return PostValueOrDone();
                }
                if (b == (byte)'"')
                {
                    // Begin a key: candidates are every not-yet-emitted key (unordered), or the
                    // declaration-order window (ordered, issue #425).
                    ulong cand = obj.NextKeyCandidates(f.Emitted);
                    if (cand == 0) return Step.Reject;     // no key may appear here; only '}' is legal
                    f.Cand = cand; f.MatchLen = 0; f.State = OKeyContent;
                    return Step.Consume;
                }
                return Step.Reject;

            case OKeyContent:
                if (b == (byte)'"')
                {
                    int complete = CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen);
                    if (complete < 0) return Step.Reject;  // closing quote at a non-key-boundary
                    f.Emitted |= 1UL << complete;
                    f.PendingKey = complete;
                    f.State = OExpectColon;
                    return Step.Consume;
                }
                else
                {
                    ulong narrowed = 0;
                    for (int i = 0; i < obj.Count; i++)
                        if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen && obj.KeyBytes[i][f.MatchLen] == b)
                            narrowed |= 1UL << i;
                    if (narrowed == 0) return Step.Reject;
                    f.Cand = narrowed; f.MatchLen++;
                    return Step.Consume;
                }

            case OExpectColon:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)':')
                {
                    f.State = OExpectCommaOrClose;          // resume here after the value
                    return PushValue(obj.Values[f.PendingKey]) ? Step.Consume : Step.Reject;
                }
                return Step.Reject;

            case OExpectCommaOrClose:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)',')
                {
                    // A ',' commits to another key — reject it when none can follow (all keys
                    // emitted, or the ordered window is exhausted) so the machine can't be steered
                    // into a whitespace-only OExpectKey livelock where only '}' was ever viable.
                    if (!obj.HasNextKey(f.Emitted)) return Step.Reject;
                    f.State = OExpectKey; return Step.Consume;
                }
                if (b == (byte)'}')
                {
                    if ((f.Emitted & obj.RequiredMask) != obj.RequiredMask) return Step.Reject;
                    _depth--; return PostValueOrDone();
                }
                return Step.Reject;

            default:
                return Step.Reject;
        }
    }

    private Step StepArray(ref Frame f, byte b)
    {
        var item = f.Node!.Items!;
        switch (f.State)
        {
            case AExpectOpen:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'[') { f.State = AExpectItemOrClose; return Step.Consume; }
                return Step.Reject;

            case AExpectItemOrClose:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)']') { _depth--; return PostValueOrDone(); }
                f.State = AExpectCommaOrClose;              // resume here after the item
                return PushValue(item) ? Step.Retry : Step.Reject;

            case AExpectCommaOrClose:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)',') { f.State = AExpectItem; return Step.Consume; }
                if (b == (byte)']') { _depth--; return PostValueOrDone(); }
                return Step.Reject;

            case AExpectItem:
                if (IsWs(b)) return Step.Consume;
                f.State = AExpectCommaOrClose;
                return PushValue(item) ? Step.Retry : Step.Reject;

            default:
                return Step.Reject;
        }
    }

    private Step StepStr(ref Frame f, byte b)
    {
        switch (f.State)
        {
            case SExpectOpen:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'"') { f.State = SContent; return Step.Consume; }
                return Step.Reject;

            case SContent:
                if (f.Escaped) { f.Escaped = false; return Step.Consume; }
                if (b == (byte)'\\') { f.Escaped = true; return Step.Consume; }
                if (b == (byte)'"') { _depth--; return PostValueOrDone(); }  // close string
                return Step.Consume;                        // free content byte

            default:
                return Step.Reject;
        }
    }

    private Step StepStrEnum(ref Frame f, byte b)
    {
        var lits = f.Node!.Literals!;
        switch (f.State)
        {
            case SeExpectOpen:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'"') { f.State = SeMatch; f.MatchLen = 0; return Step.Consume; }
                return Step.Reject;

            case SeMatch:
                if (b == (byte)'"')
                {
                    if (CompleteIndex(lits, f.Cand, f.MatchLen) < 0) return Step.Reject;
                    _depth--; return PostValueOrDone();
                }
                ulong narrowed = 0;
                for (int i = 0; i < lits.Length; i++)
                    if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen && lits[i][f.MatchLen] == b)
                        narrowed |= 1UL << i;
                if (narrowed == 0) return Step.Reject;
                f.Cand = narrowed; f.MatchLen++;
                return Step.Consume;

            default:
                return Step.Reject;
        }
    }

    private Step StepNum(ref Frame f, byte b)
    {
        bool digit = b is >= (byte)'0' and <= (byte)'9';
        switch (f.State)
        {
            case NStart:
                if (b == (byte)'-' && !f.SeenSign) { f.SeenSign = true; return Step.Consume; }
                if (digit) { f.SeenDigit = true; f.State = NIntDigits; return Step.Consume; }
                return Step.Reject;
            case NIntDigits:
                if (digit) return Step.Consume;
                if (b == (byte)'.' && !f.Node!.IntegerOnly && !f.SeenDot) { f.SeenDot = true; f.State = NFracStart; return Step.Consume; }
                if (f.SeenDigit) { _depth--; return PostValueRetry(); }
                return Step.Reject;
            case NFracStart:
                if (digit) { f.State = NFracDigits; return Step.Consume; }
                return Step.Reject;
            case NFracDigits:
                if (digit) return Step.Consume;
                _depth--; return PostValueRetry();
            default:
                return Step.Reject;
        }
    }

    private Step StepLit(ref Frame f, byte b)
    {
        var lits = f.Node!.Literals!;
        ulong narrowed = 0;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen && lits[i][f.MatchLen] == b)
                narrowed |= 1UL << i;
        if (narrowed != 0) { f.Cand = narrowed; f.MatchLen++; return Step.Consume; }
        if (CompleteIndex(lits, f.Cand, f.MatchLen) >= 0) { _depth--; return PostValueRetry(); }
        return Step.Reject;
    }

    // ── Preamble (watching) byte FSM ──────────────────────────────────────────

    private enum WatchResult { Continue, Engage, Disarm }

    /// <summary>Walks one preamble byte for the NameValueObject envelope, returning whether the
    /// argument object now opens (engage), the call should be abandoned (disarm), or scanning
    /// continues.</summary>
    private WatchResult WatchByte(byte b)
    {
        switch (_watchState)
        {
            case WStart:
                if (IsWs(b)) return WatchResult.Continue;
                if (b == (byte)'{') { _watchState = WKeyExpect; return WatchResult.Continue; }
                return WatchResult.Disarm;

            case WKeyExpect:
                if (IsWs(b)) return WatchResult.Continue;
                if (b == (byte)'"') { _keyBuf.Clear(); _wEscaped = false; _watchState = WKeyContent; return WatchResult.Continue; }
                return WatchResult.Disarm;                  // '}' or junk before any key → no args

            case WKeyContent:
                if (_wEscaped) { _keyBuf.Append((char)b); _wEscaped = false; return WatchResult.Continue; }
                if (b == (byte)'\\') { _wEscaped = true; return WatchResult.Continue; }
                if (b == (byte)'"') { _watchState = WColon; return WatchResult.Continue; }
                _keyBuf.Append((char)b);
                return WatchResult.Continue;

            case WColon:
                if (IsWs(b)) return WatchResult.Continue;
                if (b == (byte)':')
                {
                    // The args key's colon is the engage point: push the object frame now so the next
                    // token is masked. The colon is consumed; any remaining token bytes replay into
                    // the machine. A non-args key falls through to its value.
                    if (CurrentKeyIsArgsKey())
                        return EngageOnTool() ? WatchResult.Engage : WatchResult.Disarm;
                    _watchState = WValueStart; return WatchResult.Continue;
                }
                return WatchResult.Disarm;

            case WValueStart:                               // key is NOT the args key (engaged above)
                if (IsWs(b)) return WatchResult.Continue;
                if (KeyBufEquals(s_nameKey))
                {
                    if (b != (byte)'"') return WatchResult.Disarm;   // name must be a string
                    _nameBuf.Clear(); _wEscaped = false; _watchState = WNameString;
                    return WatchResult.Continue;
                }
                // Some other key — skip its value and resume at the next key.
                _skipDepth = 0; _skipKind = 0; _wEscaped = false; _watchState = WSkipValue;
                return SkipValueByte(b);

            case WNameString:
                if (_wEscaped) { _nameBuf.Append((char)b); _wEscaped = false; return WatchResult.Continue; }
                if (b == (byte)'\\') { _wEscaped = true; return WatchResult.Continue; }
                if (b == (byte)'"') { _watchState = WAfterValue; return WatchResult.Continue; }
                _nameBuf.Append((char)b);
                return WatchResult.Continue;

            case WSkipValue:
                return SkipValueByte(b);

            case WAfterValue:
                if (IsWs(b)) return WatchResult.Continue;
                if (b == (byte)',') { _keyBuf.Clear(); _watchState = WKeyExpect; return WatchResult.Continue; }
                return WatchResult.Disarm;                  // '}' (envelope closed, no args) or junk

            default:
                return WatchResult.Disarm;
        }
    }

    private bool CurrentKeyIsArgsKey()
    {
        foreach (var k in _argsKeys) if (KeyBufEquals(k)) return true;
        return false;
    }

    /// <summary>Engages the constrained machine on the identified tool's argument object, starting in
    /// <see cref="OExpectOpenBrace"/> so an explicit '{' is forced and a merged "{}" can't drop the
    /// required arguments. Returns false (caller disarms) for an unknown / non-constrainable tool.
    /// Assumes the canonical name-before-arguments key order; a model that emitted the arguments first
    /// with two-plus tools active leaves the name uncaptured and degrades to unconstrained.</summary>
    private bool EngageOnTool()
    {
        string name = _nameBuf.ToString().Trim();
        if (name.Length == 0 && _tools.Count == 1)
            name = _tools.Keys.First();                     // single tool, no name needed (DeepSeek/bare)
        if (!_tools.TryGetValue(name, out var obj)) return false;

        _depth = 0;
        PushObject(obj, OExpectOpenBrace);
        return true;
    }

    /// <summary>Skips one JSON value during preamble scanning (an unrelated envelope key's value),
    /// balancing strings/objects/arrays so the next key is found correctly. On the value's end
    /// transitions to <see cref="WAfterValue"/>; a bare scalar's terminating ',' or '}' is left for
    /// <see cref="WAfterValue"/> to consume.</summary>
    private WatchResult SkipValueByte(byte b)
    {
        switch (_skipKind)
        {
            case 0:                                          // undecided — classify the value
                if (IsWs(b)) return WatchResult.Continue;
                if (b == (byte)'"') { _skipKind = 1; _wEscaped = false; return WatchResult.Continue; }
                if (b is (byte)'{' or (byte)'[') { _skipKind = 2; _skipDepth = 1; return WatchResult.Continue; }
                _skipKind = 3;                               // bare scalar
                return WatchResult.Continue;

            case 1:                                          // string
                if (_wEscaped) { _wEscaped = false; return WatchResult.Continue; }
                if (b == (byte)'\\') { _wEscaped = true; return WatchResult.Continue; }
                if (b == (byte)'"') { _watchState = WAfterValue; return WatchResult.Continue; }
                return WatchResult.Continue;

            case 2:                                          // balanced object/array (string-aware)
                if (b == (byte)'"') { _skipKind = 4; _wEscaped = false; return WatchResult.Continue; }
                if (b is (byte)'{' or (byte)'[') _skipDepth++;
                else if (b is (byte)'}' or (byte)']')
                {
                    if (--_skipDepth == 0) { _watchState = WAfterValue; return WatchResult.Continue; }
                }
                return WatchResult.Continue;

            case 4:                                          // string inside a balanced value
                if (_wEscaped) { _wEscaped = false; return WatchResult.Continue; }
                if (b == (byte)'\\') { _wEscaped = true; return WatchResult.Continue; }
                if (b == (byte)'"') _skipKind = 2;
                return WatchResult.Continue;

            default:                                         // 3: bare scalar — ends at a delimiter
                if (b is (byte)',' or (byte)'}' or (byte)']')
                {
                    _watchState = WAfterValue;
                    return WatchByte(b);                     // re-process the delimiter in WAfterValue
                }
                return WatchResult.Continue;
        }
    }

    private bool KeyBufEquals(byte[] ascii)
    {
        if (_keyBuf.Length != ascii.Length) return false;
        for (int i = 0; i < ascii.Length; i++) if (_keyBuf[i] != (char)ascii[i]) return false;
        return true;
    }

    // ── First-byte collection (mask pruning) ──────────────────────────────────

    private void CollectFirstBytes(int d, bool[] set)
    {
        while (d >= 0)
        {
            ref var f = ref _stack[d];
            bool popThrough = false;
            switch (f.Kind)
            {
                case FK.Object: CollectObject(ref f, set); break;
                case FK.Array: CollectArray(ref f, set); break;
                case FK.Str: CollectStr(ref f, set); break;
                case FK.StrEnum: CollectStrEnum(ref f, set); break;
                case FK.Num: popThrough = CollectNum(ref f, set); break;
                case FK.Lit: popThrough = CollectLit(ref f, set); break;
                case FK.Free: CollectFree(ref f, set); break;
            }
            if (!popThrough) break;
            d--;                                            // bare value can end here — parent bytes also start a token
        }
    }

    private static void CollectObject(ref Frame f, bool[] set)
    {
        var obj = f.Obj!;
        switch (f.State)
        {
            case OExpectOpenBrace: MarkWs(set); set['{'] = true; break;
            case OExpectKeyOrClose:
            case OExpectKey:
                MarkWs(set);
                if (f.State == OExpectKeyOrClose && (f.Emitted & obj.RequiredMask) == obj.RequiredMask) set['}'] = true;
                // A key opens with '"' whenever a candidate remains (any unemitted key, or the
                // ordered declaration-order window — issue #425).
                if (obj.HasNextKey(f.Emitted)) set['"'] = true;
                break;
            case OKeyContent:
                // Either a content byte continuing a candidate key, or '"' if a candidate is complete.
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen)
                        set[obj.KeyBytes[i][f.MatchLen]] = true;
                if (CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen) >= 0) set['"'] = true;
                break;
            case OExpectColon: MarkWs(set); set[':'] = true; break;
            case OExpectCommaOrClose:
                MarkWs(set);
                if (obj.HasNextKey(f.Emitted)) set[','] = true;   // ',' commits to a key
                if ((f.Emitted & obj.RequiredMask) == obj.RequiredMask) set['}'] = true;
                break;
        }
    }

    private static void CollectArray(ref Frame f, bool[] set)
    {
        switch (f.State)
        {
            case AExpectOpen: MarkWs(set); set['['] = true; break;
            case AExpectItemOrClose:
                MarkWs(set); set[']'] = true;
                MarkValueFirstBytes(f.Node!.Items!, set);
                break;
            case AExpectCommaOrClose: MarkWs(set); set[','] = true; set[']'] = true; break;
            case AExpectItem: MarkWs(set); MarkValueFirstBytes(f.Node!.Items!, set); break;
        }
    }

    private static void CollectStr(ref Frame f, bool[] set)
    {
        if (f.State == SExpectOpen) { MarkWs(set); set['"'] = true; return; }
        // SContent: any byte is legal content (and '"' closes, '\' escapes) — mark all so the
        // simulate pass (authoritative) decides which tokens stay alive.
        for (int i = 0; i < 256; i++) set[i] = true;
    }

    private static void CollectFree(ref Frame f, bool[] set)
    {
        // A free value admits almost anything — mark broadly and let the (fast-pathed) simulate pass
        // decide. The only structural restriction is that the value can't START with a parent
        // delimiter (that would be an empty value).
        for (int i = 0; i < 256; i++) set[i] = true;
        if (f.State == FrStart) { set[','] = false; set['}'] = false; set[']'] = false; }
    }

    private static void CollectStrEnum(ref Frame f, bool[] set)
    {
        if (f.State == SeExpectOpen) { MarkWs(set); set['"'] = true; return; }
        var lits = f.Node!.Literals!;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen)
                set[lits[i][f.MatchLen]] = true;
        if (CompleteIndex(lits, f.Cand, f.MatchLen) >= 0) set['"'] = true;
    }

    private static bool CollectNum(ref Frame f, bool[] set)
    {
        switch (f.State)
        {
            case NStart:
                if (!f.SeenSign) set['-'] = true;
                MarkDigits(set);
                return false;
            case NIntDigits:
                MarkDigits(set);
                if (!f.SeenDot && !f.Node!.IntegerOnly) set['.'] = true;
                return f.SeenDigit;
            case NFracStart:
                MarkDigits(set);
                return false;
            case NFracDigits:
                MarkDigits(set);
                return true;
            default:
                return false;
        }
    }

    private static bool CollectLit(ref Frame f, bool[] set)
    {
        var lits = f.Node!.Literals!;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen)
                set[lits[i][f.MatchLen]] = true;
        return CompleteIndex(lits, f.Cand, f.MatchLen) >= 0;
    }

    private static void MarkValueFirstBytes(CompiledNode node, bool[] set)
    {
        switch (node.Kind)
        {
            case JsonSchemaKind.Array: set['['] = true; break;
            case JsonSchemaKind.Object: set['{'] = true; break;
            case JsonSchemaKind.String: set['"'] = true; break;   // JSON strings open with a '"' byte
            case JsonSchemaKind.Any:
                // A free array item (issue #378) can be any JSON value — mark every value-START byte.
                // (Don't touch ']'; the array frame marks it for the empty-array close, and unsetting
                // it here would wrongly forbid closing.)
                set['"'] = true; set['{'] = true; set['['] = true; set['-'] = true;
                MarkDigits(set);
                set['t'] = true; set['f'] = true; set['n'] = true;   // true / false / null
                break;
            default:
                if (node.Literals is { } lits)
                    foreach (var l in lits) { if (l.Length > 0) set[l[0]] = true; }
                else { set['-'] = true; MarkDigits(set); }
                break;
        }
    }

    private static void MarkDigits(bool[] set) { for (byte b = (byte)'0'; b <= (byte)'9'; b++) set[b] = true; }
    private static void MarkWs(bool[] set) { set[' '] = true; set['\t'] = true; set['\n'] = true; set['\r'] = true; }

    // ── Small helpers ─────────────────────────────────────────────────────────

    private static int CompleteIndex(byte[][] lits, ulong cand, int matchLen)
    {
        for (int i = 0; i < lits.Length; i++)
            if ((cand & (1UL << i)) != 0 && lits[i].Length == matchLen) return i;
        return -1;
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> s_warned = new();
    private static void WarnOnce(string key, string message)
    {
        if (s_warned.TryAdd(key, 0))
            Console.Error.WriteLine($"[SharpInference.ToolGrammar] {message}");
    }
}
