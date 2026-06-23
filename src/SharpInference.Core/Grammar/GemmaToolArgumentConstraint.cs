using System.Text;

namespace SharpInference.Core.Grammar;

// ── Compiled schema ───────────────────────────────────────────────────────────
//
// The parsed ToolSchema is "compiled" once per request into match tables (UTF-8 key/enum bytes,
// required-key bitmask) so the per-token hot loop never re-encodes strings. Only tools whose schema
// is FULLY constrainable (every value is a concrete type / typed array / nested typed object — no
// Any-typed value, no open object) are compiled; a tool that isn't is simply left out of the
// constraint and generates its arguments unconstrained, so the constraint never blocks generation.

/// <summary>Compiled value-type descriptor (see <see cref="ToolSchemaNode"/>).</summary>
internal sealed class CompiledNode
{
    public required JsonSchemaKind Kind { get; init; }
    /// <summary>For an enum / boolean / null: the candidate literal byte sequences (≤64).</summary>
    public byte[][]? Literals { get; init; }
    /// <summary>True when <see cref="Literals"/> are matched inside the <c>&lt;|"|&gt;</c> quotes (string enum).</summary>
    public bool QuotedLiterals { get; init; }
    /// <summary>Element type for an array.</summary>
    public CompiledNode? Items { get; init; }
    /// <summary>Nested object shape.</summary>
    public CompiledObject? Object { get; init; }
    /// <summary>Integer (no fractional part) vs general number.</summary>
    public bool IntegerOnly { get; init; }
}

/// <summary>Compiled object shape: per-property name bytes + value node, plus the required-key mask.</summary>
internal sealed class CompiledObject
{
    public required byte[][] KeyBytes { get; init; }
    public required CompiledNode[] Values { get; init; }
    public required ulong RequiredMask { get; init; }
    public int Count => KeyBytes.Length;
}

/// <summary>
/// Constrained-decoding state machine (issue #374) for Gemma 4's native tool-call wire format:
/// <c>&lt;|tool_call&gt;call:NAME{key:&lt;|"|&gt;val&lt;|"|&gt;,n:3}&lt;tool_call|&gt;</c>.
///
/// <para>
/// It watches the emitted token stream passively until it sees a call open
/// (<c>&lt;|tool_call&gt;</c>) followed by a known tool name and the argument-object brace
/// <c>{</c>; from there it constrains the argument object so that only keys declared in that tool's
/// schema appear, every required key appears exactly once, each value matches its declared shape
/// (strings in Gemma's <c>&lt;|"|&gt;</c> quotes with free content; numbers/booleans/null bare;
/// arrays/objects recursively), and enum-typed values are limited to the declared set. The object's
/// closing <c>}</c> ends constraint and the machine returns to watching, so subsequent text
/// (<c>&lt;tool_call|&gt;</c>, further calls, plain answer) is unconstrained.
/// </para>
///
/// <para>
/// Matching is at the byte level for the structural skeleton (so multi-token keys, merged delimiters
/// like <c>:[</c>/<c>]}</c>, and multi-token numbers/enums all work) and at the token level for the
/// quote special token and free string content. A token is permitted iff replaying its bytes from
/// the current state keeps the machine alive.
/// </para>
/// </summary>
public sealed class GemmaToolArgumentConstraint : ITokenConstraint
{
    private const int MaxNameScan = 256;    // give up watching a call whose name region runs this long
    private const int MaxDepth = 32;        // nesting cap; deeper schemas aren't constrained

    private readonly GrammarVocabulary _vocab;
    private readonly Dictionary<string, CompiledObject> _tools;
    private readonly int _openMarkerId;     // <|tool_call>
    private readonly int _quoteId;          // <|"|>
    private readonly HashSet<int> _forbidden; // EOG ids — never legal inside the argument object

    // Watching state.
    private bool _armed;                     // saw <|tool_call>, accumulating the name region
    private readonly StringBuilder _nameBuf = new();

    // Constraining state: an explicit frame stack (pushdown automaton).
    private Frame[] _stack = new Frame[MaxDepth];
    private Frame[] _scratch = new Frame[MaxDepth];
    private int _depth;

    // Masking scratch — a reusable full-vocab logits buffer (allocated on first constrained step).
    private float[]? _masked;
    private readonly bool[] _firstByteOk = new bool[256];

    public GemmaToolArgumentConstraint(GrammarVocabulary vocab, IReadOnlyList<ToolSchema> tools)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(tools);
        _vocab = vocab;

        // Resolve Gemma's structural special tokens. Without the quote token we can't constrain
        // string values, so leave _tools empty (passive) if it's missing.
        _ = vocab.TryGetSpecialToken(Gemma4ToolCallAdapter.OpenMarker, out _openMarkerId);
        _ = vocab.TryGetSpecialToken(Gemma4ToolCallAdapter.Quote, out _quoteId);

        _forbidden = new HashSet<int>(vocab.EogTokenIds);

        _tools = new Dictionary<string, CompiledObject>(StringComparer.Ordinal);
        if (_openMarkerId > 0 && _quoteId > 0)
        {
            foreach (var t in tools)
            {
                if (t.Arguments.Open) continue;             // unconstrained body — skip
                var compiled = TryCompileObject(t.Arguments);
                if (compiled is not null) _tools[t.Name] = compiled;
            }
        }
        else if (tools.Count > 0)
        {
            // The caller asked for a constraint (tools present) but this vocabulary doesn't define
            // Gemma's structural tokens — the constraint is inert. Surface it once so an operator who
            // enabled SHARPI_TOOL_GRAMMAR on a non-Gemma / mistokenized model isn't left wondering
            // why arguments are still unconstrained.
            WarnOnce("no-structural-tokens",
                "structural tokens <|tool_call> / <|\"|> not found in this vocabulary — tool-grammar is "
                + "inert for this model (arguments generate unconstrained).");
        }

        // Names we can safely engage on the instant the model finishes typing them (before the '{'),
        // so a merged "{}" token can't slip an empty argument object past the constraint. A name that
        // is a strict prefix of another tool name is ambiguous mid-stream (the model might still be
        // typing the longer one), so it's excluded — those fall back to the '{'-triggered path.
        _engageNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var name in _tools.Keys)
        {
            bool isPrefixOfAnother = false;
            foreach (var other in _tools.Keys)
                if (!ReferenceEquals(name, other) && other.Length > name.Length
                    && other.StartsWith(name, StringComparison.Ordinal))
                { isPrefixOfAnother = true; break; }
            if (!isPrefixOfAnother) _engageNames.Add(name);
        }
    }

    private readonly HashSet<string> _engageNames;

    /// <summary>Whether this constraint can ever restrict anything (any tool was constrainable).</summary>
    public bool HasConstrainableTools => _tools.Count > 0;

    public bool IsConstraining => _depth > 0;

    public void Reset()
    {
        _armed = false;
        _nameBuf.Clear();
        _depth = 0;
    }

    // ── Compilation ───────────────────────────────────────────────────────────

    private static CompiledObject? TryCompileObject(ToolSchemaObject obj)
    {
        if (obj.Open || obj.Properties.Count is 0 or > 64) return null;
        var keys = new byte[obj.Properties.Count][];
        var values = new CompiledNode[obj.Properties.Count];
        ulong reqMask = 0;
        for (int i = 0; i < obj.Properties.Count; i++)
        {
            var p = obj.Properties[i];
            if (p.Name.Length == 0) return null;
            keys[i] = ToolSchema.Utf8(p.Name);
            var node = TryCompileNode(p.Value);
            if (node is null) return null;                  // a non-constrainable value → skip the tool
            values[i] = node;
            if (p.Required) reqMask |= 1UL << i;
        }
        return new CompiledObject { KeyBytes = keys, Values = values, RequiredMask = reqMask };
    }

    private static readonly byte[][] s_boolLiterals = [ToolSchema.Utf8("true"), ToolSchema.Utf8("false")];
    private static readonly byte[][] s_nullLiterals = [ToolSchema.Utf8("null")];

    private static CompiledNode? TryCompileNode(ToolSchemaNode node)
    {
        switch (node.Kind)
        {
            case JsonSchemaKind.String:
                byte[][]? strEnum = node.EnumValues is { } se ? Encode(se) : null;
                return new CompiledNode { Kind = JsonSchemaKind.String, Literals = strEnum, QuotedLiterals = strEnum is not null };

            case JsonSchemaKind.Number:
            case JsonSchemaKind.Integer:
                byte[][]? numEnum = node.EnumValues is { } ne ? Encode(ne) : null;
                return new CompiledNode
                {
                    Kind = node.Kind,
                    Literals = numEnum,
                    QuotedLiterals = false,
                    IntegerOnly = node.Kind == JsonSchemaKind.Integer,
                };

            case JsonSchemaKind.Boolean:
                return new CompiledNode { Kind = JsonSchemaKind.Boolean, Literals = s_boolLiterals };

            case JsonSchemaKind.Null:
                return new CompiledNode { Kind = JsonSchemaKind.Null, Literals = s_nullLiterals };

            case JsonSchemaKind.Array:
                // An untyped array (no items) isn't fully constrainable — skip the tool.
                if (node.Items is null) return null;
                var item = TryCompileNode(node.Items);
                return item is null ? null : new CompiledNode { Kind = JsonSchemaKind.Array, Items = item };

            case JsonSchemaKind.Object:
                if (node.Object is null) return null;
                var obj = TryCompileObject(node.Object);
                return obj is null ? null : new CompiledNode { Kind = JsonSchemaKind.Object, Object = obj };

            default:
                return null;   // Any / unknown — not constrainable
        }
    }

    private static byte[][] Encode(IReadOnlyList<string> values)
    {
        var r = new byte[values.Count][];
        for (int i = 0; i < values.Count; i++) r[i] = ToolSchema.Utf8(values[i]);
        return r;
    }

    // ── Frame stack ───────────────────────────────────────────────────────────

    private enum FK : byte { Object, Array, Str, StrEnum, Num, Lit }

    private struct Frame
    {
        public FK Kind;
        public int State;
        public CompiledObject? Obj;     // Object frame
        public CompiledNode? Node;      // Array (item), StrEnum / Lit (literals), Num
        public ulong Emitted;           // Object: keys emitted
        public ulong Cand;              // Object key-match candidates / StrEnum / Lit candidates
        public int MatchLen;            // chars into current key / literal
        public bool SeenDigit;          // Num
        public bool SeenDot;            // Num
        public bool SeenSign;           // Num
    }

    // Object sub-states.
    private const int OExpectKeyOrClose = 0; // key or '}' (if all required emitted)
    private const int OMatchKey = 1;         // matching a key name (byte-level)
    private const int OExpectCommaOrClose = 2;
    private const int OExpectKey = 3;        // after ',', a key is required ('}' not allowed)
    private const int OExpectOpenBrace = 4;  // top-level object engaged before '{' (early-engage path)

    // Array sub-states.
    private const int AExpectOpen = 0;       // '['
    private const int AExpectItemOrClose = 1;
    private const int AExpectCommaOrClose = 2;
    private const int AExpectItem = 3;       // after ',', an item is required

    // Str sub-states.
    private const int SExpectOpen = 0;       // open quote (token-level)
    private const int SContent = 1;          // free content (token-level)

    // StrEnum sub-states.
    private const int SeExpectOpen = 0;      // open quote (token-level)
    private const int SeMatch = 1;           // match enum bytes; close quote ends (mixed)

    // Num sub-states.
    private const int NStart = 0;
    private const int NIntDigits = 1;
    private const int NFracStart = 2;
    private const int NFracDigits = 3;

    // Lit sub-state: single matching state.
    private const int LMatch = 0;

    private void PushObject(CompiledObject obj)
    {
        ref var f = ref _stack[_depth++];
        f = default;
        f.Kind = FK.Object; f.State = OExpectKeyOrClose; f.Obj = obj; f.Emitted = 0;
    }

    /// <summary>Pushes a nested-object VALUE frame: unlike <see cref="PushObject"/> (top-level body,
    /// entered past the brace) the opening <c>{</c> hasn't been consumed yet, so it starts in
    /// <see cref="OExpectOpenBrace"/>.</summary>
    private bool PushObjectValue(CompiledObject obj)
    {
        ref var f = ref _stack[_depth++];
        f = default;
        f.Kind = FK.Object; f.State = OExpectOpenBrace; f.Obj = obj; f.Emitted = 0;
        return true;
    }

    private bool PushValue(CompiledNode node)
    {
        // An object value gets a dedicated Object frame (with its own emitted/required tracking),
        // dispatched before reserving a slot here so we don't write a frame we'd immediately replace.
        if (node.Kind == JsonSchemaKind.Object)
            return node.Object is not null && _depth < MaxDepth && PushObjectValue(node.Object);

        if (_depth >= MaxDepth) return false;
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
        if (IsConstraining)
        {
            bool ok = RunToken(token);
            // Under both greedy and temperature sampling the engine draws from the masked logits, so
            // a permitted token must replay cleanly. A rejection here is an invariant violation (a
            // Filter/Accept divergence — e.g. a mask-vs-simulate bug), distinct from the normal
            // end-of-object transition (ok && _depth == 0). Flag it once; either way, stop constraining.
            if (!ok)
                WarnOnce("accept-divergence",
                    "a sampled token permitted by the mask was rejected by the grammar — constraint "
                    + "disabled for the rest of this call (possible Filter/Accept divergence).");
            if (!ok || _depth == 0) { _depth = 0; _armed = false; _nameBuf.Clear(); }
            return;
        }

        // Watching.
        if (token == _openMarkerId)
        {
            _armed = true; _nameBuf.Clear();
            return;
        }
        if (!_armed) return;

        var bytes = _vocab.TokenBytes(token);
        // Append decoded text; the name region is ASCII (call:NAME) so a direct byte→char append
        // is exact for the parts we care about. Watch for the '{' that opens the argument object.
        for (int i = 0; i < bytes.Length; i++)
        {
            char c = (char)bytes[i];
            if (c == '{')
            {
                StartConstraintBody(bytes[(i + 1)..]);
                return;
            }
            _nameBuf.Append(c);
        }

        // Early engage: the instant the accumulated name exactly matches a (prefix-safe) tool, begin
        // constraining BEFORE the '{' arrives. This forces the model to open with '{' + a valid first
        // key, so a merged "{}" token can't drop the required arguments (the issue's get_weather → {}
        // failure mode). Names that are a prefix of another tool aren't engaged here (ambiguous).
        if (CurrentName() is { } name && _engageNames.Contains(name) && _tools.TryGetValue(name, out var obj))
        {
            _armed = false; _nameBuf.Clear();
            _depth = 0;
            PushObject(obj);
            _stack[0].State = OExpectOpenBrace;   // expect the opening '{' as the first constrained token
            return;
        }

        if (_nameBuf.Length > MaxNameScan) { _armed = false; _nameBuf.Clear(); }
    }

    /// <summary>The bare tool name accumulated so far (leading "call:" stripped), or null if empty.</summary>
    private string? CurrentName()
    {
        string region = _nameBuf.ToString();
        int colon = region.IndexOf("call:", StringComparison.Ordinal);
        string name = (colon >= 0 ? region[(colon + "call:".Length)..] : region).Trim();
        return name.Length == 0 ? null : name;
    }

    /// <summary>Begins constraining at the argument-object body (the '{' was already emitted, possibly
    /// merged into the triggering token whose trailing bytes are replayed here).</summary>
    private void StartConstraintBody(ReadOnlySpan<byte> trailing)
    {
        var name = CurrentName();
        _nameBuf.Clear();
        _armed = false;

        if (name is null || !_tools.TryGetValue(name, out var obj)) return;  // unknown / non-constrainable

        _depth = 0;
        PushObject(obj);     // starts in OExpectKeyOrClose (already past '{')

        // Rare: the '{' token carried trailing argument bytes (e.g. "{location"). Replay them now;
        // if they don't fit the grammar, abandon constraining rather than wedge generation. (The
        // early-engage path avoids this entirely for prefix-unambiguous tool names; a name that's a
        // prefix of another tool relies on this replay, so surface a failure once.)
        if (!trailing.IsEmpty && !FeedRawBytes(trailing))
        {
            _depth = 0;
            WarnOnce($"trailing-replay:{name}",
                $"could not replay argument bytes after '{{' for tool '{name}' — arguments generate "
                + "unconstrained for this call.");
        }
    }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        if (_depth == 0) return logits;

        var masked = _masked ??= new float[_vocab.VocabSize];
        if (masked.Length != logits.Length) return logits;     // vocab mismatch — never wedge
        logits.CopyTo(masked);

        int allowedCount = ComputeMask(masked);
        // Dead state (no legal token): leave logits untouched so generation never wedges. A valid
        // grammar state always has at least one legal token (a key, '}', a value byte, or free
        // content), so reaching zero indicates the model is at an off-schema point OR a grammar bug —
        // either way the call continues unconstrained, so flag it once rather than fail silently.
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

    /// <summary>Sets every forbidden token to -inf in <paramref name="buf"/>; returns the count kept.</summary>
    private int ComputeMask(float[] buf)
    {
        ref var top = ref _stack[_depth - 1];
        int kept = 0;

        // Token-level tops: free string content and the open/close quote. Handle without a
        // per-token byte simulation for speed.
        if (top.Kind == FK.Str && top.State == SContent)
        {
            // Any non-forbidden token stays in content; the quote token closes. Forbid only EOG —
            // a tiny set, so mask those ids directly rather than testing all 262k tokens against it.
            int forbidden = 0;
            foreach (int id in _forbidden)
                if ((uint)id < (uint)buf.Length && !float.IsNegativeInfinity(buf[id]))
                { buf[id] = float.NegativeInfinity; forbidden++; }
            return buf.Length - forbidden;
        }
        if ((top.Kind == FK.Str && top.State == SExpectOpen)
            || (top.Kind == FK.StrEnum && top.State == SeExpectOpen))
        {
            for (int id = 0; id < buf.Length; id++)
            {
                if (id == _quoteId) kept++;
                else buf[id] = float.NegativeInfinity;
            }
            return kept;
        }

        // General path: prune by allowed first byte, then full-simulate the survivors. The quote
        // token starts with '<' (outside the structural byte set), so whether it's a legal next token
        // — opening/closing a string, or opening a string array item — is determined explicitly.
        Array.Clear(_firstByteOk);
        bool quoteAllowed = CollectFirstBytes(_depth - 1, _firstByteOk) || IsQuoteAccepted();

        for (int id = 0; id < buf.Length; id++)
        {
            var bytes = _vocab.TokenBytes(id);
            bool candidate;
            if (id == _quoteId) candidate = quoteAllowed;
            else if (bytes.Length == 0) candidate = false;
            else candidate = _firstByteOk[bytes[0]];

            if (candidate && SimulateToken(id))
                kept++;
            else
                buf[id] = float.NegativeInfinity;
        }
        return kept;
    }

    private bool SimulateToken(int token)
    {
        int savedDepth = _depth;
        Array.Copy(_stack, _scratch, _depth);
        bool ok = RunToken(token);
        Array.Copy(_scratch, _stack, savedDepth);
        _depth = savedDepth;
        return ok;
    }

    // ── Token execution ───────────────────────────────────────────────────────

    /// <summary>Replays one token's bytes/identity through the machine, mutating the stack. Returns
    /// false if the token isn't legal from the current state.</summary>
    private bool RunToken(int tokenId)
    {
        if (_depth == 0) return false;

        // The quote is a single special token that delimits a string everywhere it can appear (value
        // open/close, string-enum open/close, string array-item open). Centralise it so it's matched
        // at the token level rather than byte-walking its '<|"|>' bytes.
        if (tokenId == _quoteId) return HandleQuote();

        ref var top = ref _stack[_depth - 1];
        switch (top.Kind)
        {
            case FK.Str when top.State == SExpectOpen:    // string value needs its opening quote
            case FK.StrEnum when top.State == SeExpectOpen:
                return false;
            case FK.Str when top.State == SContent:       // free content: any non-EOG token stays
                return !_forbidden.Contains(tokenId);
            case FK.StrEnum when top.State == SeMatch:
                break;                                    // enum content → byte-walk below
        }

        // Byte-level walk.
        return FeedRawBytes(_vocab.TokenBytes(tokenId));
    }

    /// <summary>Handles the quote special token from the current top frame: opens/closes a string or
    /// string-enum value, or opens a string array item (the one value whose first token is the quote
    /// rather than a structural byte). Returns false where a quote isn't legal.</summary>
    private bool HandleQuote()
    {
        ref var top = ref _stack[_depth - 1];
        switch (top.Kind)
        {
            case FK.Str:
                if (top.State == SExpectOpen) { top.State = SContent; return true; }
                _depth--; return PostValue();                          // SContent → close
            case FK.StrEnum:
                if (top.State == SeExpectOpen) { top.State = SeMatch; return true; }
                if (!AnyComplete(top.Node!.Literals!, top.Cand, top.MatchLen)) return false;
                _depth--; return PostValue();                          // SeMatch (complete) → close
            case FK.Array when top.State is AExpectItemOrClose or AExpectItem:
            {
                var item = top.Node!.Items!;
                if (item.Kind != JsonSchemaKind.String) return false;  // only a string item opens on a quote
                top.State = AExpectCommaOrClose;                       // resume here after the item
                if (_depth >= MaxDepth) return false;
                ref var f = ref _stack[_depth++];
                f = default; f.Node = item;
                if (item.Literals is not null) { f.Kind = FK.StrEnum; f.State = SeMatch; f.Cand = AllBits(item.Literals.Length); }
                else { f.Kind = FK.Str; f.State = SContent; }
                return true;
            }
            default:
                return false;                                          // quote not legal here
        }
    }

    /// <summary>Read-only mirror of <see cref="HandleQuote"/>: whether the quote token is a legal
    /// next token from the current top frame (used by mask pruning).</summary>
    private bool IsQuoteAccepted()
    {
        ref var top = ref _stack[_depth - 1];
        return top.Kind switch
        {
            FK.Str => top.State is SExpectOpen or SContent,
            FK.StrEnum => top.State == SeExpectOpen
                          || (top.State == SeMatch && AnyComplete(top.Node!.Literals!, top.Cand, top.MatchLen)),
            FK.Array => top.State is AExpectItemOrClose or AExpectItem
                        && top.Node!.Items!.Kind == JsonSchemaKind.String,
            _ => false,
        };
    }

    /// <summary>Byte-walks a raw byte span through the structural automaton. Used for ordinary
    /// tokens and for the rare trailing bytes after the opening brace.</summary>
    private bool FeedRawBytes(ReadOnlySpan<byte> bytes)
    {
        int i = 0;
        while (i < bytes.Length)
        {
            if (_depth == 0) return false;                 // closed the object but bytes remain
            ref var top = ref _stack[_depth - 1];

            // A token-level state reached mid-token (e.g. a string value's open quote) means the
            // remaining bytes belong to a separate token — not legal within one token.
            if (IsTokenLevel(top)) return false;

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

    private static bool IsTokenLevel(in Frame f) =>
        (f.Kind == FK.Str) || (f.Kind == FK.StrEnum && f.State == SeExpectOpen);

    private enum Step { Consume, Retry, Reject }

    private Step StepByte(ref Frame top, byte b)
    {
        switch (top.Kind)
        {
            case FK.Object: return StepObject(ref top, b);
            case FK.Array: return StepArray(ref top, b);
            case FK.Num: return StepNum(ref top, b);
            case FK.Lit: return StepLit(ref top, b);
            case FK.StrEnum: return StepStrEnum(ref top, b);   // SeMatch only reaches here
            default: return Step.Reject;
        }
    }

    private static bool IsWs(byte b) => b is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\r';

    // After a value frame pops, resume the parent (object/array) at its post-value state.
    private bool PostValue()
    {
        if (_depth == 0) return true;                      // top-level value (shouldn't happen here)
        ref var parent = ref _stack[_depth - 1];
        if (parent.Kind == FK.Object) { parent.State = OExpectCommaOrClose; return true; }
        if (parent.Kind == FK.Array) { parent.State = AExpectCommaOrClose; return true; }
        return false;
    }

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
                // Begin matching a not-yet-emitted key whose first byte == b.
                ulong cand = 0;
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Emitted & (1UL << i)) == 0 && obj.KeyBytes[i].Length > 0 && obj.KeyBytes[i][0] == b)
                        cand |= 1UL << i;
                if (cand == 0) return Step.Reject;
                f.Cand = cand; f.MatchLen = 1; f.State = OMatchKey;
                return Step.Consume;

            case OMatchKey:
            {
                // Continue matching candidate keys.
                ulong narrowed = 0;
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen && obj.KeyBytes[i][f.MatchLen] == b)
                        narrowed |= 1UL << i;
                if (narrowed != 0) { f.Cand = narrowed; f.MatchLen++; return Step.Consume; }

                // b doesn't continue any candidate: a completed key + ':' finalizes; ws waits.
                int complete = CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen);
                if (IsWs(b)) return complete >= 0 ? Step.Consume : Step.Reject;
                if (b == (byte)':' && complete >= 0)
                {
                    f.Emitted |= 1UL << complete;
                    f.State = OExpectCommaOrClose;          // resume here after the value
                    return PushValue(obj.Values[complete]) ? Step.Consume : Step.Reject;
                }
                return Step.Reject;
            }

            case OExpectCommaOrClose:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)',') { f.State = OExpectKey; return Step.Consume; }
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

    // Object frame popped: either the whole call is done (depth 0) or it was a nested value.
    private Step PostValueOrDone()
    {
        if (_depth == 0) return Step.Consume;               // top-level object done
        return PostValue() ? Step.Consume : Step.Reject;
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
                if (f.SeenDigit) { _depth--; return PostValueRetry(); }   // number ended — parent handles b
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

    // A bare value (number / literal) ends without consuming its delimiter; pop and let the parent
    // re-process the current byte.
    private Step PostValueRetry() => PostValue() ? Step.Retry : Step.Reject;

    private Step StepLit(ref Frame f, byte b)
    {
        var lits = f.Node!.Literals!;
        // If a candidate is already complete and b doesn't continue any, the literal is done.
        ulong narrowed = 0;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen && lits[i][f.MatchLen] == b)
                narrowed |= 1UL << i;
        if (narrowed != 0) { f.Cand = narrowed; f.MatchLen++; return Step.Consume; }
        if (CompleteIndex(lits, f.Cand, f.MatchLen) >= 0) { _depth--; return PostValueRetry(); }
        return Step.Reject;
    }

    private Step StepStrEnum(ref Frame f, byte b)
    {
        // SeMatch byte content: narrow the enum candidates.
        var lits = f.Node!.Literals!;
        ulong narrowed = 0;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen && lits[i][f.MatchLen] == b)
                narrowed |= 1UL << i;
        if (narrowed == 0) return Step.Reject;
        f.Cand = narrowed; f.MatchLen++; return Step.Consume;
    }

    // ── First-byte collection (mask pruning) ──────────────────────────────────

    /// <summary>Marks every byte that could legally start the next token from frame
    /// <paramref name="d"/>, following pop-through for bare values. Returns whether the quote token
    /// is also a legal next token (string-enum close).</summary>
    private bool CollectFirstBytes(int d, bool[] set)
    {
        bool quote = false;
        while (d >= 0)
        {
            ref var f = ref _stack[d];
            bool popThrough = false;
            switch (f.Kind)
            {
                case FK.Object:
                    popThrough = CollectObject(ref f, set);
                    break;
                case FK.Array:
                    popThrough = CollectArray(ref f, set);
                    break;
                case FK.Num:
                    popThrough = CollectNum(ref f, set);
                    break;
                case FK.Lit:
                    popThrough = CollectLit(ref f, set);
                    break;
                case FK.StrEnum:
                    // SeMatch: enum content bytes continue; quote closes if a candidate is complete.
                    CollectStrEnum(ref f, set, ref quote);
                    popThrough = false;
                    break;
            }
            if (!popThrough) break;
            d--;                                            // bare value can end here — parent's bytes also start a token
        }
        return quote;
    }

    private static bool CollectObject(ref Frame f, bool[] set)
    {
        var obj = f.Obj!;
        switch (f.State)
        {
            case OExpectOpenBrace:
                MarkWs(set);
                set['{'] = true;
                return false;
            case OExpectKeyOrClose:
            case OExpectKey:
                MarkWs(set);
                if (f.State == OExpectKeyOrClose && (f.Emitted & obj.RequiredMask) == obj.RequiredMask)
                    set['}'] = true;
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Emitted & (1UL << i)) == 0 && obj.KeyBytes[i].Length > 0)
                        set[obj.KeyBytes[i][0]] = true;
                return false;
            case OMatchKey:
                MarkWs(set);
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen)
                        set[obj.KeyBytes[i][f.MatchLen]] = true;
                if (CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen) >= 0) set[':'] = true;
                return false;
            case OExpectCommaOrClose:
                MarkWs(set);
                set[','] = true;
                if ((f.Emitted & obj.RequiredMask) == obj.RequiredMask) set['}'] = true;
                return false;
            default:
                return false;
        }
    }

    private static bool CollectArray(ref Frame f, bool[] set)
    {
        switch (f.State)
        {
            case AExpectOpen: MarkWs(set); set['['] = true; return false;
            case AExpectItemOrClose:
                MarkWs(set); set[']'] = true;
                // An item value starts next — mark its possible first bytes via a probe push.
                MarkValueFirstBytes(f.Node!.Items!, set);
                return false;
            case AExpectCommaOrClose: MarkWs(set); set[','] = true; set[']'] = true; return false;
            case AExpectItem: MarkWs(set); MarkValueFirstBytes(f.Node!.Items!, set); return false;
            default: return false;
        }
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
                return f.SeenDigit;                          // can end → parent bytes also start tokens
            case NFracStart:
                MarkDigits(set);
                return false;
            case NFracDigits:
                MarkDigits(set);
                return true;                                 // can end
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

    private static void CollectStrEnum(ref Frame f, bool[] set, ref bool quote)
    {
        var lits = f.Node!.Literals!;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen)
                set[lits[i][f.MatchLen]] = true;
        if (AnyComplete(lits, f.Cand, f.MatchLen)) quote = true;
    }

    /// <summary>Marks the possible first bytes of a value node (for array item pruning) by
    /// inspecting the node type — a read-only sibling of <see cref="PushValue"/>.</summary>
    private static void MarkValueFirstBytes(CompiledNode node, bool[] set)
    {
        switch (node.Kind)
        {
            case JsonSchemaKind.Array: set['['] = true; break;
            case JsonSchemaKind.Object: set['{'] = true; break;
            case JsonSchemaKind.String: break;             // opens with the quote token, not a byte
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

    private static bool AnyComplete(byte[][] lits, ulong cand, int matchLen) => CompleteIndex(lits, cand, matchLen) >= 0;

    // One diagnostic per distinct event per process (mirrors JinjaChatTemplate.WarnUnsupportedOnce):
    // Console.Error is the only channel from this dependency-free Core type, deduped so a recurring
    // condition can't spam the decode loop. Used for the otherwise-silent degradation paths — an
    // opt-in best-effort feature must never wedge generation, but a *silent* no-op defeats the point
    // of enabling it, so each abandonment leaves exactly one breadcrumb.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> s_warned = new();
    private static void WarnOnce(string key, string message)
    {
        if (s_warned.TryAdd(key, 0))
            Console.Error.WriteLine($"[SharpInference.ToolGrammar] {message}");
    }
}
