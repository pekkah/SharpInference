using System.Text;

namespace SharpInference.Core.Grammar;

/// <summary>
/// Constrained-decoding state machine (issue #383) for Qwen3-Coder's <b>XML</b> tool-call wire
/// format — the one remaining family not covered by the JSON
/// (<see cref="JsonToolArgumentConstraint"/>) or Gemma (<see cref="GemmaToolArgumentConstraint"/>)
/// constraints. Qwen3-Coder emits arguments as nested tags rather than JSON:
/// <code>
/// &lt;tool_call&gt;
/// &lt;function=get_weather&gt;
/// &lt;parameter=location&gt;
/// Paris
/// &lt;/parameter&gt;
/// &lt;/function&gt;
/// &lt;/tool_call&gt;
/// </code>
///
/// <para>
/// It watches the emitted stream until a call opens (<c>&lt;tool_call&gt;</c>) and a known tool's
/// <c>&lt;function=NAME&gt;</c> tag is seen, then constrains the function body so that only declared
/// parameters appear (as <c>&lt;parameter=KEY&gt;</c>), every required parameter appears exactly once,
/// and each value region matches its declared shape <em>where checkable</em>: a Coder value is bare
/// text between the open/close parameter tags, so typed-string / array / object / untyped values are
/// free content, while numbers, booleans, null, and enums are constrained. The closing
/// <c>&lt;/function&gt;</c> ends constraint and the machine returns to watching, so the trailing
/// <c>&lt;/tool_call&gt;</c> envelope and any later text are unconstrained.
/// </para>
///
/// <para>
/// Like the JSON sibling, the structural skeleton (<c>&lt;function=</c>, <c>&lt;parameter=</c>,
/// <c>&lt;/parameter&gt;</c>, <c>&lt;/function&gt;</c>) is matched entirely at the <b>byte</b> level —
/// these are ordinary BPE tokens the model freely merges with surrounding whitespace and content (one
/// token can carry <c>&gt;\n&lt;parameter=</c>, another <c>&lt;/parameter&gt;\n&lt;/function&gt;</c>).
/// A token is permitted iff replaying its bytes from the current state keeps the machine alive. The
/// constraint engages the instant the <c>&gt;</c> closing <c>&lt;function=NAME&gt;</c> is seen —
/// BEFORE the first <c>&lt;parameter=&gt;</c> — so a merged <c>&gt;&lt;/function&gt;</c> token can't
/// drop the required parameters (the analogue of the JSON/Gemma merged-<c>{}</c> early-engage trick).
/// </para>
///
/// <para>Default-off byte-identical: if no supplied tool is constrainable, or the
/// <c>&lt;tool_call&gt;</c> arming token isn't in the vocabulary, the constraint is inert and
/// generation is unconstrained.</para>
/// </summary>
public sealed class QwenCoderToolArgumentConstraint : ITokenConstraint
{
    private const int MaxDepth = 8;          // Func + one value frame; ample headroom (no nesting)
    private const int MaxNameScan = 256;     // give up watching a call whose name region runs this long
    private const int MaxPreambleScan = 512; // give up watching a call whose preamble runs this long

    // Structural literals (ordinary BPE byte tokens, not special tokens).
    private static readonly byte[] s_funcOpen   = Encoding.UTF8.GetBytes("<function=");
    private static readonly byte[] s_paramOpen  = Encoding.UTF8.GetBytes("<parameter=");
    private static readonly byte[] s_paramClose = Encoding.UTF8.GetBytes("</parameter>");
    private static readonly byte[] s_funcClose  = Encoding.UTF8.GetBytes("</function>");

    private readonly GrammarVocabulary _vocab;
    private readonly Dictionary<string, CompiledObject> _tools;
    private readonly HashSet<int> _forbidden;   // EOG ids — never legal inside the function body
    private readonly int _toolCallOpenId;       // <tool_call> — arms watching
    private readonly int _toolCallCloseId;      // </tool_call> — disarms (optional)

    // Watching / preamble state.
    private bool _armed;                         // saw <tool_call>, scanning for <function=NAME>
    private int _watchState;                     // preamble FSM state (W* constants)
    private int _funcMatchLen;                   // bytes of "<function=" matched in WFuncTag
    private readonly StringBuilder _nameBuf = new();
    private int _preambleLen;                    // bytes scanned since arming (runaway guard)

    // Constraining state: an explicit frame stack (pushdown automaton).
    private Frame[] _stack = new Frame[MaxDepth];
    private Frame[] _scratch = new Frame[MaxDepth];
    private int _depth;

    // Masking scratch — a reusable full-vocab logits buffer (allocated on first constrained step).
    private float[]? _masked;
    private readonly bool[] _firstByteOk = new bool[256];

    internal QwenCoderToolArgumentConstraint(GrammarVocabulary vocab, IReadOnlyList<ToolSchema> tools)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(tools);
        _vocab = vocab;

        _forbidden = new HashSet<int>(vocab.EogTokenIds);
        _ = vocab.TryGetSpecialToken(QwenCoderToolCallAdapter.ArmMarker, out _toolCallOpenId);
        _toolCallCloseId = -1;
        _ = vocab.TryGetSpecialToken(QwenCoderToolCallAdapter.ArmCloseMarker, out _toolCallCloseId);

        _tools = new Dictionary<string, CompiledObject>(StringComparer.Ordinal);
        if (_toolCallOpenId > 0)
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
            // The caller asked for a constraint but this vocabulary doesn't define Qwen's
            // <tool_call> arming token — the constraint is inert. Surface it once so an operator who
            // enabled tool-grammar on a mismatched model isn't left wondering why arguments are still
            // unconstrained.
            WarnOnce("no-tool-call-marker",
                "arming token <tool_call> not found in this vocabulary — tool-grammar is inert for this "
                + "model (Qwen3-Coder arguments generate unconstrained).");
        }
    }

    /// <summary>Whether this constraint can ever restrict anything (any tool was constrainable).</summary>
    public bool HasConstrainableTools => _tools.Count > 0;

    public bool IsConstraining => _depth > 0;

    public void Reset()
    {
        _armed = false;
        _depth = 0;
        ResetPreamble();
    }

    private void ResetPreamble()
    {
        _watchState = WSeekTag;
        _funcMatchLen = 0;
        _nameBuf.Clear();
        _preambleLen = 0;
    }

    // ── Frame stack ───────────────────────────────────────────────────────────

    private enum FK : byte { Func, Free, Num, Lit }

    private struct Frame
    {
        public FK Kind;
        public int State;
        public CompiledObject? Obj;     // Func frame
        public CompiledNode? Node;      // Lit (literals) / Num
        public ulong Emitted;           // Func: parameters emitted
        public ulong Cand;              // Func: tag-match / key-match candidates; Lit: literal candidates
        public int MatchLen;            // bytes into the current tag / key / literal / close-tag match
        public bool SeenDigit;          // Num
        public bool SeenDot;            // Num
        public bool SeenSign;           // Num
    }

    // Func sub-states.
    private const int FSeekTag = 0;          // skip ws; '<' opens a tag (<parameter= or </function>)
    private const int FMatchOpenTag = 1;     // matching <parameter= / </function> (candidate bitmask)
    private const int FParamKey = 2;         // matching a declared KEY up to '>'
    private const int FParamClose = 3;       // after a typed scalar value: ws* then </parameter>

    // Open-tag candidate bits (FMatchOpenTag).
    private const ulong TagParam = 1UL << 0; // <parameter=
    private const ulong TagFunc  = 1UL << 1; // </function>

    // Free value sub-state: rolling-match </parameter> over free content.
    private const int FrContent = 0;

    // Num sub-states (lead ws skipped in NStart).
    private const int NStart = 0;
    private const int NIntDigits = 1;
    private const int NFracStart = 2;
    private const int NFracDigits = 3;

    // Lit sub-states.
    private const int LStart = 0;            // skip leading ws
    private const int LMatch = 1;            // match one of the bare literals

    // Preamble (watching) sub-states.
    private const int WSeekTag = 0;          // skip ws; '<' begins the <function= literal
    private const int WFuncTag = 1;          // match the rest of "<function="
    private const int WName = 2;             // accumulate NAME up to '>'

    private void PushFunc(CompiledObject obj)
    {
        ref var f = ref _stack[_depth++];
        f = default;
        f.Kind = FK.Func; f.State = FSeekTag; f.Obj = obj; f.Emitted = 0;
    }

    /// <summary>Pushes the value frame for a parameter's declared node and pre-sets the parent Func's
    /// post-value resume state (so mask pop-through sees the right state). Free values consume their own
    /// <c>&lt;/parameter&gt;</c> and resume at <see cref="FSeekTag"/>; typed scalars resume at
    /// <see cref="FParamClose"/> to match the close tag after the value. Returns false if the stack is
    /// full (the value is then left unconstrained but the function structure stays enforced).</summary>
    private bool PushValue(CompiledNode node, ref Frame parent)
    {
        if (_depth >= MaxDepth) return false;
        parent.MatchLen = 0;    // entering a post-value resume state — clear the key-match cursor so
                                // FParamClose / FSeekTag start the close tag from byte 0, not mid-key.
        // Enum (any base type) / boolean / null → a bare literal from a set. Number/Integer → bare
        // number. Everything else (string, array, object, Any/free) → free content until </parameter>.
        if (node.Literals is not null)
        {
            parent.State = FParamClose;
            ref var f = ref _stack[_depth++];
            f = default; f.Kind = FK.Lit; f.State = LStart; f.Node = node; f.Cand = AllBits(node.Literals.Length);
            return true;
        }
        if (node.Kind is JsonSchemaKind.Number or JsonSchemaKind.Integer)
        {
            parent.State = FParamClose;
            ref var f = ref _stack[_depth++];
            f = default; f.Kind = FK.Num; f.State = NStart; f.Node = node;
            return true;
        }
        parent.State = FSeekTag;
        ref var ff = ref _stack[_depth++];
        ff = default; ff.Kind = FK.Free; ff.State = FrContent; ff.MatchLen = 0;
        return true;
    }

    private static ulong AllBits(int n) => n >= 64 ? ulong.MaxValue : (1UL << n) - 1;

    // ── Public lifecycle ──────────────────────────────────────────────────────

    public void Accept(int token)
    {
        if (IsConstraining)
        {
            bool ok = !_forbidden.Contains(token) && RunToken(token);
            // The engine draws from the masked logits, so a permitted token must replay cleanly. A
            // rejection here is an invariant violation (a Filter/Accept divergence); flag it once and
            // stop constraining either way. A normal end-of-function is ok && _depth == 0 — re-arm so a
            // second <function=…> in the same <tool_call> still engages.
            if (!ok)
            {
                WarnOnce("accept-divergence",
                    "a sampled token permitted by the mask was rejected by the grammar — constraint "
                    + "disabled for the rest of this call (possible Filter/Accept divergence).");
                _depth = 0; _armed = false; ResetPreamble();
            }
            else if (_depth == 0)
            {
                _armed = true; ResetPreamble();                 // function closed → keep watching the block
            }
            return;
        }

        // Watching.
        if (!_armed)
        {
            if (token == _toolCallOpenId) { _armed = true; ResetPreamble(); }
            return;
        }
        if (token == _toolCallCloseId) { Disarm(); return; }    // block ended without (another) call
        if (token == _toolCallOpenId) { ResetPreamble(); return; } // defensive: a fresh block opened

        // Walk the token's bytes through the preamble FSM; engage at the '>' closing <function=NAME>
        // so the following token (which opens the body) is masked — a merged "></function>" can't slip
        // the required parameters past. The '>' is consumed by the watcher; any remaining token bytes
        // replay into the constrained machine.
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
                        "could not replay body bytes after <function=NAME> — arguments generate "
                        + "unconstrained for this call.");
                }
                return;
            }
        }
    }

    private void Disarm() { _armed = false; ResetPreamble(); }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
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

    /// <summary>Sets every forbidden token to -inf in <paramref name="buf"/>; returns the count kept.</summary>
    private int ComputeMask(float[] buf)
    {
        Array.Clear(_firstByteOk);
        CollectFirstBytes(_depth - 1, _firstByteOk);

        // Fast-path free content. A free value's only structural byte is '<' (the start of
        // </parameter>), so CollectFree marks all 256 first-bytes — without this every step would
        // SimulateToken the WHOLE vocabulary. A token carrying no '<' is pure content that can only keep
        // the value open, so it's legal without simulation; only tokens that could begin the close tag
        // need the full replay.
        bool fastFree = _stack[_depth - 1].Kind == FK.Free;

        int kept = 0;
        for (int id = 0; id < buf.Length; id++)
        {
            var bytes = _vocab.TokenBytes(id);
            // Empty-byte tokens (EOG / control) never advance the structure — forbidding them keeps an
            // end-of-generation token from truncating the call mid-body.
            if (bytes.Length == 0 || !_firstByteOk[bytes[0]]) { buf[id] = float.NegativeInfinity; continue; }
            bool ok = (fastFree && !ContainsLt(bytes)) || SimulateToken(id);
            if (ok) kept++;
            else buf[id] = float.NegativeInfinity;
        }

        // Belt-and-suspenders: forbid every EOG id regardless of its bytes (a tokenizer whose EOS
        // decodes to ordinary text could otherwise pass first-byte pruning inside a free value).
        foreach (int id in _forbidden)
            if ((uint)id < (uint)buf.Length && !float.IsNegativeInfinity(buf[id]))
            { buf[id] = float.NegativeInfinity; kept--; }

        return kept;
    }

    private static bool ContainsLt(ReadOnlySpan<byte> bytes)
    {
        for (int i = 0; i < bytes.Length; i++)
            if (bytes[i] == (byte)'<') return true;
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
            if (_depth == 0) return false;                 // closed the function but bytes remain
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
            case FK.Func: return StepFunc(ref top, b);
            case FK.Free: return StepFree(ref top, b);
            case FK.Num: return StepNum(ref top, b);
            case FK.Lit: return StepLit(ref top, b);
            default: return Step.Reject;
        }
    }

    private Step StepFunc(ref Frame f, byte b)
    {
        var obj = f.Obj!;
        switch (f.State)
        {
            case FSeekTag:
                if (IsWs(b)) return Step.Consume;
                if (b == (byte)'<')
                {
                    // Both <parameter= and </function> open with '<'. A parameter tag is a candidate
                    // only while a declared key is still unemitted; the close tag only once every
                    // required key is emitted. (At least one always holds, so '<' is never a dead end.)
                    f.Cand = (UnemittedMask(f) != 0 ? TagParam : 0) | (RequiredSatisfied(f) ? TagFunc : 0);
                    f.MatchLen = 1;
                    f.State = FMatchOpenTag;
                    return Step.Consume;
                }
                return Step.Reject;

            case FMatchOpenTag:
            {
                ulong narrowed = 0;
                if ((f.Cand & TagParam) != 0 && s_paramOpen.Length > f.MatchLen && s_paramOpen[f.MatchLen] == b) narrowed |= TagParam;
                if ((f.Cand & TagFunc)  != 0 && s_funcClose.Length > f.MatchLen && s_funcClose[f.MatchLen] == b) narrowed |= TagFunc;
                if (narrowed == 0) return Step.Reject;
                f.Cand = narrowed; f.MatchLen++;
                // Exactly one candidate survives past index 1 ('p' vs '/'); act on completion.
                if ((f.Cand & TagParam) != 0 && f.MatchLen == s_paramOpen.Length)
                {
                    f.Cand = UnemittedMask(f); f.MatchLen = 0; f.State = FParamKey;
                    return Step.Consume;
                }
                if ((f.Cand & TagFunc) != 0 && f.MatchLen == s_funcClose.Length)
                {
                    _depth--;                               // </function> → the call is complete
                    return Step.Consume;
                }
                return Step.Consume;
            }

            case FParamKey:
            {
                if (b == (byte)'>')
                {
                    int complete = CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen);
                    if (complete < 0) return Step.Reject;   // '>' at a non-key boundary (incl. empty key)
                    f.Emitted |= 1UL << complete;
                    return PushValue(obj.Values[complete], ref f) ? Step.Consume : Step.Reject;
                }
                ulong narrowed = 0;
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen && obj.KeyBytes[i][f.MatchLen] == b)
                        narrowed |= 1UL << i;
                if (narrowed == 0) return Step.Reject;
                f.Cand = narrowed; f.MatchLen++;
                return Step.Consume;
            }

            case FParamClose:
                // After a typed scalar value: optional trailing ws, then </parameter>.
                if (f.MatchLen == 0 && IsWs(b)) return Step.Consume;
                if (s_paramClose.Length > f.MatchLen && s_paramClose[f.MatchLen] == b)
                {
                    f.MatchLen++;
                    if (f.MatchLen == s_paramClose.Length) { f.State = FSeekTag; f.MatchLen = 0; }
                    return Step.Consume;
                }
                return Step.Reject;

            default:
                return Step.Reject;
        }
    }

    /// <summary>Free value: any bytes are content until the <c>&lt;/parameter&gt;</c> close, matched by
    /// a rolling counter. <c>&lt;/parameter&gt;</c> has no repeated prefix (only index 0 is '&lt;'), so
    /// a failed match restarts cleanly. On the full match the value is complete; the parent Func resumes
    /// at <see cref="FSeekTag"/> (the close was already consumed here).</summary>
    private Step StepFree(ref Frame f, byte b)
    {
        if (b == s_paramClose[f.MatchLen])
        {
            f.MatchLen++;
            if (f.MatchLen == s_paramClose.Length)
            {
                _depth--;                                   // value + </parameter> consumed
                return _depth == 0 ? Step.Reject : Step.Consume;   // a Free value is always inside Func
            }
            return Step.Consume;
        }
        f.MatchLen = b == s_paramClose[0] ? 1 : 0;          // restart the rolling match
        return Step.Consume;                                // ordinary content byte
    }

    private Step StepNum(ref Frame f, byte b)
    {
        bool digit = b is >= (byte)'0' and <= (byte)'9';
        switch (f.State)
        {
            case NStart:
                if (IsWs(b)) return Step.Consume;           // leading ws (the template's newline)
                if (b == (byte)'-' && !f.SeenSign) { f.SeenSign = true; return Step.Consume; }
                if (digit) { f.SeenDigit = true; f.State = NIntDigits; return Step.Consume; }
                return Step.Reject;
            case NIntDigits:
                if (digit) return Step.Consume;
                if (b == (byte)'.' && !f.Node!.IntegerOnly && !f.SeenDot) { f.SeenDot = true; f.State = NFracStart; return Step.Consume; }
                if (f.SeenDigit) { _depth--; return Step.Retry; }   // number ended — parent (FParamClose) handles b
                return Step.Reject;
            case NFracStart:
                if (digit) { f.State = NFracDigits; return Step.Consume; }
                return Step.Reject;
            case NFracDigits:
                if (digit) return Step.Consume;
                _depth--; return Step.Retry;
            default:
                return Step.Reject;
        }
    }

    private Step StepLit(ref Frame f, byte b)
    {
        var lits = f.Node!.Literals!;
        if (f.State == LStart)
        {
            if (IsWs(b)) return Step.Consume;               // leading ws
            f.State = LMatch;                               // fall through to matching
        }
        ulong narrowed = 0;
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen && lits[i][f.MatchLen] == b)
                narrowed |= 1UL << i;
        if (narrowed != 0) { f.Cand = narrowed; f.MatchLen++; return Step.Consume; }
        if (CompleteIndex(lits, f.Cand, f.MatchLen) >= 0) { _depth--; return Step.Retry; }  // literal done
        return Step.Reject;
    }

    private static bool IsWs(byte b) => b is (byte)' ' or (byte)'\t' or (byte)'\n' or (byte)'\r';

    private static bool RequiredSatisfied(in Frame f) => (f.Emitted & f.Obj!.RequiredMask) == f.Obj!.RequiredMask;

    private static ulong UnemittedMask(in Frame f)
    {
        ulong cand = 0;
        for (int i = 0; i < f.Obj!.Count; i++)
            if ((f.Emitted & (1UL << i)) == 0) cand |= 1UL << i;
        return cand;
    }

    // ── Preamble (watching) byte FSM ──────────────────────────────────────────

    private enum WatchResult { Continue, Engage, Disarm }

    /// <summary>Walks one preamble byte while armed: skip ws, match the literal <c>&lt;function=</c>,
    /// accumulate NAME up to '>', then engage on a known constrainable tool.</summary>
    private WatchResult WatchByte(byte b)
    {
        switch (_watchState)
        {
            case WSeekTag:
                if (IsWs(b)) return WatchResult.Continue;
                if (b == s_funcOpen[0]) { _funcMatchLen = 1; _watchState = WFuncTag; return WatchResult.Continue; }
                return WatchResult.Disarm;                  // something other than <function= after <tool_call>

            case WFuncTag:
                if (s_funcOpen.Length > _funcMatchLen && s_funcOpen[_funcMatchLen] == b)
                {
                    _funcMatchLen++;
                    if (_funcMatchLen == s_funcOpen.Length) { _nameBuf.Clear(); _watchState = WName; }
                    return WatchResult.Continue;
                }
                return WatchResult.Disarm;

            case WName:
                if (b == (byte)'>')
                    return EngageOnTool() ? WatchResult.Engage : WatchResult.Disarm;
                if (_nameBuf.Length >= MaxNameScan) return WatchResult.Disarm;
                _nameBuf.Append((char)b);                   // function names are ASCII
                return WatchResult.Continue;

            default:
                return WatchResult.Disarm;
        }
    }

    /// <summary>Engages the constrained machine on the named tool's function body (the '>' closing
    /// <c>&lt;function=NAME&gt;</c> was just consumed). Returns false (caller disarms) for an unknown /
    /// non-constrainable tool.</summary>
    private bool EngageOnTool()
    {
        string name = _nameBuf.ToString().Trim();
        if (!_tools.TryGetValue(name, out var obj)) return false;
        _depth = 0;
        PushFunc(obj);
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
                case FK.Func: CollectFunc(ref f, set); break;
                case FK.Free: CollectFree(set); break;
                case FK.Num: popThrough = CollectNum(ref f, set); break;
                case FK.Lit: popThrough = CollectLit(ref f, set); break;
            }
            if (!popThrough) break;
            d--;                                            // a bare scalar can end here — parent bytes also start a token
        }
    }

    private static void CollectFunc(ref Frame f, bool[] set)
    {
        var obj = f.Obj!;
        switch (f.State)
        {
            case FSeekTag:
                MarkWs(set);
                set['<'] = true;                            // <parameter= or </function>
                break;
            case FMatchOpenTag:
                if ((f.Cand & TagParam) != 0 && s_paramOpen.Length > f.MatchLen) set[s_paramOpen[f.MatchLen]] = true;
                if ((f.Cand & TagFunc)  != 0 && s_funcClose.Length > f.MatchLen) set[s_funcClose[f.MatchLen]] = true;
                break;
            case FParamKey:
                for (int i = 0; i < obj.Count; i++)
                    if ((f.Cand & (1UL << i)) != 0 && obj.KeyBytes[i].Length > f.MatchLen)
                        set[obj.KeyBytes[i][f.MatchLen]] = true;
                if (CompleteIndex(obj.KeyBytes, f.Cand, f.MatchLen) >= 0) set['>'] = true;
                break;
            case FParamClose:
                if (f.MatchLen == 0) MarkWs(set);
                if (s_paramClose.Length > f.MatchLen) set[s_paramClose[f.MatchLen]] = true;
                break;
        }
    }

    private static void CollectFree(bool[] set)
    {
        // A free value admits any content; '<' may begin </parameter>. Mark everything and let the
        // (fast-pathed) simulate pass decide which '<'-bearing tokens stay alive.
        for (int i = 0; i < 256; i++) set[i] = true;
    }

    private static bool CollectNum(ref Frame f, bool[] set)
    {
        switch (f.State)
        {
            case NStart:
                MarkWs(set);
                if (!f.SeenSign) set['-'] = true;
                MarkDigits(set);
                return false;
            case NIntDigits:
                MarkDigits(set);
                if (!f.SeenDot && !f.Node!.IntegerOnly) set['.'] = true;
                return f.SeenDigit;                          // can end → parent (FParamClose) bytes too
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
        if (f.State == LStart) MarkWs(set);                 // leading ws still allowed
        for (int i = 0; i < lits.Length; i++)
            if ((f.Cand & (1UL << i)) != 0 && lits[i].Length > f.MatchLen)
                set[lits[i][f.MatchLen]] = true;
        // A complete literal can end (parent FParamClose handles the next byte), unless we're still in
        // the leading-ws state with nothing matched yet.
        return f.State == LMatch && CompleteIndex(lits, f.Cand, f.MatchLen) >= 0;
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
