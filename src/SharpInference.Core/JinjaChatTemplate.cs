using System.Text;

namespace SharpInference.Core;

/// <summary>
/// Minimal Jinja2 template engine for LLM chat templates (read from GGUF tokenizer.chat_template
/// or tokenizer_config.json). Handles all constructs used by Qwen3, Llama3/4, SmolLM, Gemma, etc.
///
/// Supported syntax:
///   {{ expr }}         — output expression ({{- strips left whitespace, -}} strips right)
///   {% if cond %}      — conditional
///   {% elif cond %}    — else-if branch
///   {% else %}         — else branch
///   {% endif %}        — end if
///   {% for v in iter %}— loop (loop.index0 / loop.first / loop.last available)
///   {% endfor %}       — end for
///   {% set x = expr %} — variable assignment (also ns.attr = value for namespace mutation)
///   {# comment #}      — ignored
///   {%- -%} / {{-     — whitespace control
///
/// Supported expressions:
///   string/int/bool/null literals, identifiers, attribute access (.attr), indexing ([n]),
///   slicing ([::-1] reverse etc.), arithmetic (+/-), concatenation (~), comparisons
///   (== != < > <= >= in not-in), logical (and or not), filters (| length | tojson | trim …),
///   method calls (.strip() .split() .startswith() etc.), type tests (is string / is defined),
///   function calls (namespace(key=val) / raise_exception(msg)).
/// </summary>
public sealed class JinjaChatTemplate
{
    private readonly List<INode> _nodes;

    public JinjaChatTemplate(string source) => _nodes = ParseTemplate(source);

    /// <summary>
    /// Render the template with the provided context variables.
    /// Common keys: "messages" (List&lt;Dictionary&lt;string,object?&gt;&gt;),
    /// "add_generation_prompt" (bool), "tools" (null or list).
    /// </summary>
    public string Render(IReadOnlyDictionary<string, object?> context)
    {
        var ctx = new Dictionary<string, object?>(context, StringComparer.Ordinal);
        var sb  = new StringBuilder();
        EvalNodes(_nodes, ctx, sb);
        return sb.ToString();
    }

    // ── AST ───────────────────────────────────────────────────────────────

    private interface INode { }
    private sealed record TextNode(string Text) : INode;
    private sealed record OutputNode(IExpr Expr) : INode;
    private sealed record IfNode(IExpr Cond, List<INode> Then,
        List<(IExpr Cond, List<INode> Body)> ElseIfs, List<INode>? Else) : INode;
    private sealed record ForNode(string Var, IExpr Iter, List<INode> Body) : INode;
    private sealed record SetNode(string[] Path, IExpr Value) : INode;

    private interface IExpr { }
    private sealed record LiteralExpr(object? Val) : IExpr;
    private sealed record NameExpr(string Name) : IExpr;
    private sealed record AttrExpr(IExpr Obj, string Attr) : IExpr;
    private sealed record IndexExpr(IExpr Obj, IExpr Idx) : IExpr;
    private sealed record SliceExpr(IExpr Obj, IExpr? Start, IExpr? Stop, IExpr? Step) : IExpr;
    private sealed record BinExpr(string Op, IExpr L, IExpr R) : IExpr;
    private sealed record UnaryExpr(string Op, IExpr Operand) : IExpr;
    private sealed record CallExpr(string Name, List<IExpr> Args, List<(string Key, IExpr Val)> Kwargs) : IExpr;
    private sealed record MethodExpr(IExpr Obj, string Method, List<IExpr> Args) : IExpr;
    private sealed record FilterExpr(IExpr Value, string Filter) : IExpr;
    private sealed record IsTestExpr(IExpr Value, string Test, bool Negated) : IExpr;

    // ── Lexer ─────────────────────────────────────────────────────────────

    private enum TokenKind { Text, Block, Expr, Comment }
    private readonly record struct Token(TokenKind Kind, string Content, bool StripL, bool StripR);

    private static List<Token> Lex(string src)
    {
        var tokens = new List<Token>();
        int pos = 0;

        while (pos < src.Length)
        {
            int bi = src.IndexOf("{%", pos, StringComparison.Ordinal);
            int ei = src.IndexOf("{{", pos, StringComparison.Ordinal);
            int ci = src.IndexOf("{#", pos, StringComparison.Ordinal);

            int next = Min3(bi >= 0 ? bi : int.MaxValue,
                            ei >= 0 ? ei : int.MaxValue,
                            ci >= 0 ? ci : int.MaxValue);

            if (next == int.MaxValue)
            {
                if (pos < src.Length) tokens.Add(new(TokenKind.Text, src[pos..], false, false));
                break;
            }

            if (next > pos) tokens.Add(new(TokenKind.Text, src[pos..next], false, false));

            if (next == bi)
            {
                int end = src.IndexOf("%}", next + 2, StringComparison.Ordinal);
                if (end < 0) throw new FormatException("Unclosed {%");
                bool sl = src[next + 2] == '-', sr = src[end - 1] == '-';
                string inner = src[(next + 2 + (sl ? 1 : 0))..(end - (sr ? 1 : 0))].Trim();
                tokens.Add(new(TokenKind.Block, inner, sl, sr));
                pos = end + 2;
            }
            else if (next == ei)
            {
                int end = src.IndexOf("}}", next + 2, StringComparison.Ordinal);
                if (end < 0) throw new FormatException("Unclosed {{");
                bool sl = src[next + 2] == '-', sr = src[end - 1] == '-';
                string inner = src[(next + 2 + (sl ? 1 : 0))..(end - (sr ? 1 : 0))].Trim();
                tokens.Add(new(TokenKind.Expr, inner, sl, sr));
                pos = end + 2;
            }
            else
            {
                int end = src.IndexOf("#}", next + 2, StringComparison.Ordinal);
                if (end < 0) throw new FormatException("Unclosed {#");
                bool sl = src[next + 2] == '-', sr = src[end - 1] == '-';
                tokens.Add(new(TokenKind.Comment, "", sl, sr));
                pos = end + 2;
            }
        }

        ApplyStripping(tokens);
        return tokens;
    }

    private static void ApplyStripping(List<Token> tokens)
    {
        for (int i = 0; i < tokens.Count; i++)
        {
            var tok = tokens[i];
            if (tok.Kind is TokenKind.Text) continue;

            if (tok.StripL && i > 0 && tokens[i - 1].Kind == TokenKind.Text)
            {
                var prev = tokens[i - 1];
                tokens[i - 1] = prev with { Content = prev.Content.TrimEnd() };
            }
            if (tok.StripR && i + 1 < tokens.Count && tokens[i + 1].Kind == TokenKind.Text)
            {
                var nxt = tokens[i + 1];
                string c = nxt.Content;
                if (c.Length > 0 && c[0] == '\r') c = c[1..];
                if (c.Length > 0 && c[0] == '\n') c = c[1..];
                tokens[i + 1] = nxt with { Content = c };
            }
        }
    }

    private static int Min3(int a, int b, int c) => Math.Min(a, Math.Min(b, c));

    // ── Parser ────────────────────────────────────────────────────────────

    private static List<INode> ParseTemplate(string src)
    {
        var tokens = Lex(src);
        int pos = 0;
        return ParseNodes(tokens, ref pos);
    }

    private static List<INode> ParseNodes(List<Token> tokens, ref int pos)
    {
        var nodes = new List<INode>();

        while (pos < tokens.Count)
        {
            var tok = tokens[pos];
            switch (tok.Kind)
            {
                case TokenKind.Text:
                    if (tok.Content.Length > 0) nodes.Add(new TextNode(tok.Content));
                    pos++;
                    break;

                case TokenKind.Comment:
                    pos++;
                    break;

                case TokenKind.Expr:
                    nodes.Add(new OutputNode(ParseExpr(tok.Content)));
                    pos++;
                    break;

                case TokenKind.Block:
                    string tag = tok.Content;
                    if (IsEndTag(tag)) return nodes;

                    if (tag.StartsWith("if ", StringComparison.Ordinal))
                    {
                        pos++;
                        nodes.Add(ParseIf(tokens, ref pos, tag[3..].Trim()));
                    }
                    else if (tag.StartsWith("for ", StringComparison.Ordinal))
                    {
                        pos++;
                        nodes.Add(ParseFor(tokens, ref pos, tag[4..].Trim()));
                    }
                    else if (tag.StartsWith("set ", StringComparison.Ordinal))
                    {
                        pos++;
                        nodes.Add(ParseSet(tag[4..].Trim()));
                    }
                    else if (tag is "else" || tag.StartsWith("elif ", StringComparison.Ordinal))
                    {
                        return nodes; // caller handles
                    }
                    else if (tag.StartsWith("macro ", StringComparison.Ordinal)
                          || tag.StartsWith("block ", StringComparison.Ordinal)
                          || tag == "raw")
                    {
                        // Consume the body up to {% endmacro %} / {% endblock %} / {% endraw %} and discard.
                        // Macros are only invoked from inside branches we don't enter (e.g. tool-call rendering
                        // when tools is empty); calls to undefined macros emit "" via CallFunction's null fallback.
                        pos++;
                        ParseNodes(tokens, ref pos);
                        if (pos < tokens.Count && tokens[pos].Kind == TokenKind.Block
                            && IsEndTag(tokens[pos].Content)) pos++;
                    }
                    else
                    {
                        pos++; // skip unknown single-line tag
                    }
                    break;
            }
        }

        return nodes;
    }

    private static bool IsEndTag(string tag) =>
        tag is "endif" or "endfor" or "endblock" or "endmacro" or "endraw";

    private static IfNode ParseIf(List<Token> tokens, ref int pos, string condStr)
    {
        var cond = ParseExpr(condStr);
        var then = ParseNodes(tokens, ref pos);
        var elseIfs = new List<(IExpr, List<INode>)>();
        List<INode>? elsePart = null;

        while (pos < tokens.Count && tokens[pos].Kind == TokenKind.Block)
        {
            string tag = tokens[pos].Content;
            if (tag.StartsWith("elif ", StringComparison.Ordinal))
            {
                pos++;
                var elifCond = ParseExpr(tag[5..].Trim());
                elseIfs.Add((elifCond, ParseNodes(tokens, ref pos)));
            }
            else if (tag == "else")
            {
                pos++;
                elsePart = ParseNodes(tokens, ref pos);
            }
            else if (tag == "endif")
            {
                pos++;
                break;
            }
            else break;
        }

        return new IfNode(cond, then, elseIfs, elsePart);
    }

    private static ForNode ParseFor(List<Token> tokens, ref int pos, string spec)
    {
        int inIdx = FindWordBoundary(spec, "in");
        if (inIdx < 0) throw new FormatException($"Invalid for spec: {spec}");
        string varName = spec[..inIdx].Trim();
        var iter = ParseExpr(spec[(inIdx + 2)..].Trim());
        var body = ParseNodes(tokens, ref pos);
        if (pos < tokens.Count && tokens[pos].Kind == TokenKind.Block && tokens[pos].Content == "endfor")
            pos++;
        return new ForNode(varName, iter, body);
    }

    private static SetNode ParseSet(string spec)
    {
        // Find first '=' that is not part of '==', '!=', '<=', '>='
        int eq = -1;
        for (int i = 0; i < spec.Length; i++)
        {
            char c = spec[i];
            if (c == '=' && (i == 0 || spec[i - 1] is not ('!' or '<' or '>' or '='))
                         && (i + 1 >= spec.Length || spec[i + 1] != '='))
            { eq = i; break; }
        }
        if (eq < 0) throw new FormatException($"Invalid set: {spec}");
        string lhs = spec[..eq].Trim();
        string rhs = spec[(eq + 1)..].Trim();
        return new SetNode(lhs.Split('.'), ParseExpr(rhs));
    }

    // ── Expression Parser ─────────────────────────────────────────────────

    private static IExpr ParseExpr(string s) => new ExprParser(s).Parse();

    private sealed class ExprParser(string src)
    {
        private readonly string _s = src;
        public int Pos;

        public IExpr Parse() { var e = ParseOr(); return e; }

        // Operator precedence (low → high): or → and → not → comparison → add → unary → postfix → primary

        private IExpr ParseOr()
        {
            var left = ParseAnd();
            while (MatchWord("or"))
                left = new BinExpr("or", left, ParseAnd());
            return left;
        }

        private IExpr ParseAnd()
        {
            var left = ParseNot();
            while (MatchWord("and"))
                left = new BinExpr("and", left, ParseNot());
            return left;
        }

        private IExpr ParseNot()
        {
            Skip();
            if (PeekWord("not"))
            {
                Pos += 3; // consume "not"
                Skip();
                if (Pos < _s.Length && _s[Pos] == '(')
                    return new UnaryExpr("not", ParsePrimary()); // not(expr)
                return new UnaryExpr("not", ParseNot()); // not expr (recursive for "not not x")
            }
            return ParseComparison();
        }

        private IExpr ParseComparison()
        {
            var left = ParseAdd();
            Skip();

            // Type test: "is" / "is not"
            if (Peek("is ") || Peek("is\t"))
            {
                Pos += 2; Skip();
                bool negated = false;
                if (MatchWord("not")) negated = true;
                Skip();
                string test = ReadIdent();
                return new IsTestExpr(left, test, negated);
            }

            string op = "";
            if      (Peek("=="))    { op = "==";    Pos += 2; }
            else if (Peek("!="))    { op = "!=";    Pos += 2; }
            else if (Peek(">="))    { op = ">=";    Pos += 2; }
            else if (Peek("<="))    { op = "<=";    Pos += 2; }
            else if (Peek(">") && !Peek("}}")) { op = ">";  Pos++; }
            else if (Peek("<") && !Peek("{%")) { op = "<";  Pos++; }
            else if (PeekWord("not") && PeekAtOffset(4, "in"))
                                    { op = "not in"; Pos += 6; }
            else if (PeekWord("in")) { op = "in";  Pos += 2; }

            if (op.Length > 0)
            {
                Skip();
                return new BinExpr(op, left, ParseAdd());
            }

            return left;
        }

        private IExpr ParseAdd()
        {
            var left = ParseUnary();
            Skip();
            while (Pos < _s.Length && (_s[Pos] == '+' || _s[Pos] == '-' || _s[Pos] == '~'))
            {
                // Don't eat '-' that starts a negative number following whitespace (ambiguous)
                char op = _s[Pos++];
                Skip();
                var right = ParseUnary();
                left = new BinExpr(op == '~' ? "~" : op.ToString(), left, right);
                Skip();
            }
            return left;
        }

        private IExpr ParseUnary()
        {
            Skip();
            if (Pos < _s.Length && _s[Pos] == '-') { Pos++; return new UnaryExpr("-", ParsePostfix()); }
            return ParsePostfix();
        }

        private IExpr ParsePostfix()
        {
            var expr = ParsePrimary();

            while (true)
            {
                Skip();
                if (Pos >= _s.Length) break;
                char c = _s[Pos];

                if (c == '.')
                {
                    Pos++;
                    string attr = ReadIdent();
                    Skip();
                    if (Pos < _s.Length && _s[Pos] == '(')
                    {
                        Pos++;
                        expr = new MethodExpr(expr, attr, ParseArgList(')'));
                    }
                    else
                        expr = new AttrExpr(expr, attr);
                }
                else if (c == '[')
                {
                    Pos++;
                    expr = ParseSubscript(expr);
                }
                else if (c == '|')
                {
                    Pos++; Skip();
                    string filter = ReadIdent();
                    expr = new FilterExpr(expr, filter);
                }
                else break;
            }

            return expr;
        }

        private IExpr ParseSubscript(IExpr obj)
        {
            // Handles [n], [n:m], [n:m:step], [::-1], etc.
            Skip();
            if (Pos < _s.Length && _s[Pos] == ':')
            {
                // [:...] — slice with no start
                Pos++;
                IExpr? stop = null, step = null;
                Skip();
                if (Pos < _s.Length && _s[Pos] is not ':' and not ']')
                    stop = ParseOr();
                Skip();
                if (Pos < _s.Length && _s[Pos] == ':')
                {
                    Pos++; Skip();
                    if (Pos < _s.Length && _s[Pos] != ']')
                        step = ParseOr();
                }
                ExpectChar(']');
                return new SliceExpr(obj, null, stop, step);
            }

            var idx = ParseOr();
            Skip();
            if (Pos < _s.Length && _s[Pos] == ':')
            {
                // [start:...] — slice with start
                Pos++;
                IExpr? stop = null, step = null;
                Skip();
                if (Pos < _s.Length && _s[Pos] is not ':' and not ']')
                    stop = ParseOr();
                Skip();
                if (Pos < _s.Length && _s[Pos] == ':')
                {
                    Pos++; Skip();
                    if (Pos < _s.Length && _s[Pos] != ']')
                        step = ParseOr();
                }
                ExpectChar(']');
                return new SliceExpr(obj, idx, stop, step);
            }

            ExpectChar(']');
            return new IndexExpr(obj, idx);
        }

        public IExpr ParsePrimary()
        {
            Skip();
            if (Pos >= _s.Length) return new LiteralExpr(null);
            char c = _s[Pos];

            if (c == '(')
            {
                Pos++;
                var inner = ParseOr();
                Skip(); ExpectChar(')');
                return inner;
            }

            if (c is '\'' or '"')
                return new LiteralExpr(ReadString());

            if (char.IsDigit(c))
                return new LiteralExpr((long)ReadInt());

            string ident = ReadIdent();
            if (ident.Length == 0) return new LiteralExpr(null);

            Skip();
            if (Pos < _s.Length && _s[Pos] == '(')
            {
                // Function call: name(args, key=val, …)
                Pos++;
                var args = new List<IExpr>();
                var kwargs = new List<(string, IExpr)>();
                while (Pos < _s.Length && _s[Pos] != ')')
                {
                    Skip();
                    if (_s[Pos] == ')') break;
                    int saved = Pos;
                    string kwName = ReadIdent();
                    Skip();
                    if (Pos < _s.Length && _s[Pos] == '=' && (Pos + 1 >= _s.Length || _s[Pos + 1] != '='))
                    {
                        Pos++; Skip();
                        kwargs.Add((kwName, ParseOr()));
                    }
                    else
                    {
                        Pos = saved;
                        args.Add(ParseOr());
                    }
                    Skip();
                    if (Pos < _s.Length && _s[Pos] == ',') Pos++;
                }
                ExpectChar(')');
                return new CallExpr(ident, args, kwargs);
            }

            return ident switch
            {
                "true"  or "True"  => new LiteralExpr(true),
                "false" or "False" => new LiteralExpr(false),
                "none"  or "None"  or "null" or "Null" => new LiteralExpr(null),
                _ => new NameExpr(ident)
            };
        }

        private List<IExpr> ParseArgList(char close)
        {
            var args = new List<IExpr>();
            while (Pos < _s.Length && _s[Pos] != close)
            {
                Skip();
                if (Pos < _s.Length && _s[Pos] == close) break;
                args.Add(ParseOr());
                Skip();
                if (Pos < _s.Length && _s[Pos] == ',') Pos++;
            }
            ExpectChar(close);
            return args;
        }

        // ── Helpers ───────────────────────────────────────────────────────

        private void Skip() { while (Pos < _s.Length && char.IsWhiteSpace(_s[Pos])) Pos++; }

        private bool Peek(string s) => Pos + s.Length <= _s.Length &&
            _s.AsSpan(Pos, s.Length).SequenceEqual(s.AsSpan());

        private bool PeekWord(string w) => Peek(w) &&
            (Pos + w.Length >= _s.Length || !char.IsLetterOrDigit(_s[Pos + w.Length]));

        private bool PeekAtOffset(int offset, string s) =>
            Pos + offset + s.Length <= _s.Length &&
            _s.AsSpan(Pos + offset, s.Length).SequenceEqual(s.AsSpan());

        private bool MatchWord(string w)
        {
            Skip();
            if (!PeekWord(w)) return false;
            Pos += w.Length; Skip();
            return true;
        }

        public string ReadIdent()
        {
            int start = Pos;
            while (Pos < _s.Length && (char.IsLetterOrDigit(_s[Pos]) || _s[Pos] == '_')) Pos++;
            return _s[start..Pos];
        }

        private string ReadString()
        {
            char q = _s[Pos++];
            var sb = new StringBuilder();
            while (Pos < _s.Length && _s[Pos] != q)
            {
                if (_s[Pos] == '\\')
                {
                    Pos++;
                    if (Pos < _s.Length)
                        sb.Append(_s[Pos] switch
                        {
                            'n' => '\n', 't' => '\t', 'r' => '\r',
                            '\\' => '\\', '\'' => '\'', '"' => '"', _ => _s[Pos]
                        });
                }
                else sb.Append(_s[Pos]);
                Pos++;
            }
            if (Pos < _s.Length) Pos++;
            return sb.ToString();
        }

        private long ReadInt()
        {
            int start = Pos;
            while (Pos < _s.Length && char.IsDigit(_s[Pos])) Pos++;
            return long.Parse(_s[start..Pos]);
        }

        private void ExpectChar(char c) { Skip(); if (Pos < _s.Length && _s[Pos] == c) Pos++; }
    }

    // ── Evaluator ─────────────────────────────────────────────────────────

    private static void EvalNodes(List<INode> nodes, Dictionary<string, object?> ctx, StringBuilder sb)
    {
        foreach (var node in nodes)
        {
            switch (node)
            {
                case TextNode t:
                    sb.Append(t.Text);
                    break;

                case OutputNode o:
                    sb.Append(Stringify(Eval(o.Expr, ctx)));
                    break;

                case IfNode ifn:
                    if (Truthy(Eval(ifn.Cond, ctx)))
                        EvalNodes(ifn.Then, ctx, sb);
                    else
                    {
                        bool handled = false;
                        foreach (var (ec, eb) in ifn.ElseIfs)
                            if (Truthy(Eval(ec, ctx))) { EvalNodes(eb, ctx, sb); handled = true; break; }
                        if (!handled && ifn.Else != null)
                            EvalNodes(ifn.Else, ctx, sb);
                    }
                    break;

                case ForNode fn:
                    var items = AsList(Eval(fn.Iter, ctx));
                    for (int i = 0; i < items.Count; i++)
                    {
                        var loopCtx = new Dictionary<string, object?>(ctx, StringComparer.Ordinal)
                        {
                            [fn.Var] = items[i],
                            ["loop"] = new LoopCtx(i, items.Count)
                        };
                        EvalNodes(fn.Body, loopCtx, sb);
                        // Propagate NamespaceObj mutations back to outer scope
                        foreach (var kv in loopCtx)
                            if (ctx.ContainsKey(kv.Key) && kv.Value is NamespaceObj) ctx[kv.Key] = kv.Value;
                    }
                    break;

                case SetNode sn:
                    var val = Eval(sn.Value, ctx);
                    if (sn.Path.Length == 1)
                        ctx[sn.Path[0]] = val;
                    else if (sn.Path.Length == 2
                        && ctx.TryGetValue(sn.Path[0], out var nsObj) && nsObj is NamespaceObj ns)
                        ns.Set(sn.Path[1], val);
                    else
                        ctx[string.Join(".", sn.Path)] = val;
                    break;
            }
        }
    }

    private static object? Eval(IExpr expr, Dictionary<string, object?> ctx)
    {
        switch (expr)
        {
            case LiteralExpr l: return l.Val;
            case NameExpr n:    return ctx.TryGetValue(n.Name, out var v) ? v : null;
            case AttrExpr a:    return GetAttr(Eval(a.Obj, ctx), a.Attr);
            case IndexExpr ix:  return GetIndex(Eval(ix.Obj, ctx), Eval(ix.Idx, ctx));
            case SliceExpr sl:
                return GetSlice(Eval(sl.Obj, ctx),
                                sl.Start != null ? Eval(sl.Start, ctx) : null,
                                sl.Stop  != null ? Eval(sl.Stop,  ctx) : null,
                                sl.Step  != null ? Eval(sl.Step,  ctx) : null);

            case BinExpr { Op: "and" } b:
                { var lv = Eval(b.L, ctx); return Truthy(lv) ? Eval(b.R, ctx) : lv; }
            case BinExpr { Op: "or" } b:
                { var lv = Eval(b.L, ctx); return Truthy(lv) ? lv : Eval(b.R, ctx); }
            case BinExpr b:
                return EvalBin(b.Op, Eval(b.L, ctx), Eval(b.R, ctx));

            case UnaryExpr { Op: "not" } u: return !Truthy(Eval(u.Operand, ctx));
            case UnaryExpr { Op: "-"   } u:
                var ov = Eval(u.Operand, ctx);
                return ov is long il ? -il : -(long)Convert.ToInt64(ov);

            case IsTestExpr it:
                bool r = EvalIsTest(Eval(it.Value, ctx), it.Test, ctx);
                return it.Negated ? !r : r;

            case FilterExpr f:  return ApplyFilter(Eval(f.Value, ctx), f.Filter);

            case MethodExpr m:
                var mArgs = m.Args.Select(a => Eval(a, ctx)).ToList();
                return CallMethod(Eval(m.Obj, ctx), m.Method, mArgs);

            case CallExpr c:
                var cArgs   = c.Args.Select(a => Eval(a, ctx)).ToList();
                var cKwargs = c.Kwargs.Select(kv => (kv.Key, Eval(kv.Val, ctx))).ToList();
                return CallFunction(c.Name, cArgs, cKwargs);

            default: return null;
        }
    }

    private static object? EvalBin(string op, object? l, object? r)
    {
        return op switch
        {
            "+"  => (l is string || r is string) ? Stringify(l) + Stringify(r)
                    : (object)(ToLong(l) + ToLong(r)),
            "-"  => (object)(ToLong(l) - ToLong(r)),
            "~"  => Stringify(l) + Stringify(r),
            "==" => EqValues(l, r),
            "!=" => !EqValues(l, r),
            ">"  => Compare(l, r) > 0,
            "<"  => Compare(l, r) < 0,
            ">=" => Compare(l, r) >= 0,
            "<=" => Compare(l, r) <= 0,
            "in" => ContainsOp(l, r),
            "not in" => !ContainsOp(l, r),
            _ => null
        };
    }

    private static bool EvalIsTest(object? val, string test, Dictionary<string, object?> ctx) =>
        test switch
        {
            "string"    => val is string,
            "defined"   => val is not null,
            "undefined" => val is null,
            "none"      => val is null,
            "integer" or "number" => val is long or int or double,
            "iterable"  => val is System.Collections.IList,
            "mapping"   => val is System.Collections.IDictionary,
            "true"      => Truthy(val),
            "false"     => !Truthy(val),
            _           => false
        };

    private static object? GetAttr(object? obj, string attr)
    {
        switch (obj)
        {
            case NamespaceObj ns: return ns.Get(attr);
            case LoopCtx lc:
                return attr switch
                {
                    "index0" => (object)(long)lc.Index,
                    "index"  => (long)(lc.Index + 1),
                    "first"  => lc.Index == 0,
                    "last"   => lc.Index == lc.Total - 1,
                    "length" => (long)lc.Total,
                    "revindex0" => (long)(lc.Total - lc.Index - 1),
                    _ => null
                };
            case Dictionary<string, object?> d:
                return d.TryGetValue(attr, out var v) ? v : null;
            case IReadOnlyDictionary<string, object?> rd:
                return rd.TryGetValue(attr, out var v2) ? v2 : null;
            default: return null;
        }
    }

    private static object? GetIndex(object? obj, object? idx)
    {
        switch (obj)
        {
            case List<object?> list:
            {
                int i = (int)ToLong(idx);
                if (i < 0) i += list.Count;
                return i >= 0 && i < list.Count ? list[i] : null;
            }
            case Dictionary<string, object?> d:
                return idx is string k && d.TryGetValue(k, out var v) ? v : null;
            case IReadOnlyDictionary<string, object?> rd:
                return idx is string ks && rd.TryGetValue(ks, out var v2) ? v2 : null;
            default: return null;
        }
    }

    private static object? GetSlice(object? obj, object? start, object? stop, object? step)
    {
        if (obj is not List<object?> list) return null;
        int count  = list.Count;
        int stepN  = step  != null ? (int)ToLong(step)  : 1;
        int startN = start != null ? (int)ToLong(start) : (stepN >= 0 ? 0 : count - 1);
        int stopN  = stop  != null ? (int)ToLong(stop)  : (stepN >= 0 ? count : -count - 1);

        if (startN < 0) startN = Math.Max(stepN >= 0 ? 0 : -1, count + startN);
        if (stopN  < 0) stopN  = count + stopN + (stepN < 0 ? 1 : 0);
        stopN = Math.Min(stopN, count);

        var result = new List<object?>();
        if (stepN > 0) { for (int i = startN; i < stopN; i += stepN) result.Add(list[i]); }
        else           { for (int i = startN; i >= Math.Max(stopN, 0); i += stepN) result.Add(list[i]); }
        return result;
    }

    private static object? ApplyFilter(object? val, string filter) => filter switch
    {
        "length"    => val is List<object?> l ? (object)(long)l.Count : val is string s ? (long)s.Length : 0L,
        "tojson"    => ToJson(val),
        "trim"      => val is string sv ? (object)sv.Trim()            : val,
        "upper"     => val is string su ? (object)su.ToUpperInvariant() : val,
        "lower"     => val is string sl ? (object)sl.ToLowerInvariant() : val,
        "first"     => val is List<object?> lf && lf.Count > 0 ? lf[0] : null,
        "last"      => val is List<object?> ll && ll.Count > 0 ? ll[ll.Count - 1] : null,
        "reverse"   => val is List<object?> lr ? (object)Enumerable.Reverse(lr).ToList() : val,
        "sort"      => val is List<object?> ls ? (object)ls.OrderBy(Stringify).ToList() : val,
        "unique"    => val is List<object?> lu ? (object)lu.Distinct().ToList() : val,
        "list"      => val is List<object?> ? val : (val is string sv2 ? sv2.Select(c => (object?)(string.Concat(c))).ToList() : null),
        "string"    => (object?)Stringify(val),
        "int"       => (object?)ToLong(val),
        "float"     => Convert.ToDouble(val),
        "abs"       => val is long lv ? (object)Math.Abs(lv) : val,
        "e" or "escape" or "forceescape" => val is string se ? (object)HtmlEscape(se) : val,
        _ => val
    };

    private static object? CallMethod(object? obj, string method, List<object?> args)
    {
        if (obj is string s)
        {
            char[]? chars = args.Count > 0 ? Stringify(args[0]).ToCharArray() : null;
            return method switch
            {
                "strip"      => s.Trim(chars),
                "lstrip"     => s.TrimStart(chars),
                "rstrip"     => s.TrimEnd(chars),
                "startswith" => args.Count > 0 && s.StartsWith(Stringify(args[0]), StringComparison.Ordinal),
                "endswith"   => args.Count > 0 && s.EndsWith(Stringify(args[0]), StringComparison.Ordinal),
                "split" => args.Count > 0
                    ? (object)s.Split(Stringify(args[0])).Select(x => (object?)x).ToList()
                    : s.Split().Select(x => (object?)x).ToList(),
                "replace" => args.Count >= 2 ? s.Replace(Stringify(args[0]), Stringify(args[1])) : (object?)s,
                "upper"   => s.ToUpperInvariant(),
                "lower"   => s.ToLowerInvariant(),
                "title"   => System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s.ToLower()),
                "count"   => args.Count > 0 ? (object)(long)CountOccurrences(s, Stringify(args[0])) : 0L,
                "find"    => args.Count > 0 ? (object)(long)s.IndexOf(Stringify(args[0]), StringComparison.Ordinal) : -1L,
                "format"  => s, // approximate: skip format calls
                _ => null
            };
        }
        if (obj is List<object?> list)
        {
            return method switch
            {
                "append" => AppendList(list, args.Count > 0 ? args[0] : null),
                "extend" => ExtendList(list, args.Count > 0 ? AsList(args[0]) : []),
                _ => null
            };
        }
        return null;
    }

    private static object? CallFunction(string name, List<object?> args, List<(string Key, object? Val)> kwargs)
    {
        if (name == "namespace")
        {
            var ns = new NamespaceObj();
            foreach (var (k, v) in kwargs) ns.Set(k, v);
            return ns;
        }
        if (name is "raise_exception" or "raise")
        {
            string msg = args.Count > 0 ? Stringify(args[0]) : "Template error";
            throw new InvalidOperationException($"Template error: {msg}");
        }
        if (name == "range")
        {
            long stop = args.Count > 0 ? ToLong(args[0]) : 0;
            return Enumerable.Range(0, (int)stop).Select(i => (object?)(long)i).ToList();
        }
        if (name == "dict")
        {
            var d = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var (k, v) in kwargs) d[k] = v;
            return d;
        }
        return null;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static bool Truthy(object? val) => val switch
    {
        null               => false,
        bool b             => b,
        long l             => l != 0,
        int i              => i != 0,
        double d           => d != 0.0,
        string s           => s.Length > 0,
        List<object?> lst  => lst.Count > 0,
        _                  => true
    };

    private static string Stringify(object? val) => val switch
    {
        null    => "",
        string s => s,
        bool b  => b ? "True" : "False",
        _       => val.ToString() ?? ""
    };

    private static long ToLong(object? val) => val switch
    {
        long l  => l,
        int i   => i,
        bool b  => b ? 1L : 0L,
        string s => long.TryParse(s, out var n) ? n : 0L,
        _       => Convert.ToInt64(val ?? 0L)
    };

    private static bool EqValues(object? a, object? b)
    {
        if (a is null && b is null) return true;
        if (a is null || b is null) return false;
        if (a is string sa && b is string sb) return sa == sb;
        if (a is bool   ba && b is bool   bb) return ba == bb;
        if (a is bool bba) return bba == Truthy(b);
        if (b is bool bbb) return Truthy(a) == bbb;
        if (a is long || b is long) return ToLong(a) == ToLong(b);
        return a.Equals(b);
    }

    private static int Compare(object? a, object? b)
    {
        if (a is long la && b is long lb) return la.CompareTo(lb);
        return string.Compare(Stringify(a), Stringify(b), StringComparison.Ordinal);
    }

    private static bool ContainsOp(object? needle, object? haystack) => haystack switch
    {
        string h          => h.Contains(Stringify(needle), StringComparison.Ordinal),
        List<object?> lst => lst.Any(item => EqValues(item, needle)),
        _                 => false
    };

    private static List<object?> AsList(object? val) => val switch
    {
        List<object?> l => l,
        null            => [],
        _               => []
    };

    private static int CountOccurrences(string s, string sub)
    {
        if (sub.Length == 0) return 0;
        int count = 0, idx = 0;
        while ((idx = s.IndexOf(sub, idx, StringComparison.Ordinal)) >= 0) { count++; idx += sub.Length; }
        return count;
    }

    private static object? AppendList(List<object?> list, object? item) { list.Add(item); return null; }
    private static object? ExtendList(List<object?> list, List<object?> other) { list.AddRange(other); return null; }

    private static string HtmlEscape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;")
         .Replace("\"", "&quot;").Replace("'", "&#39;");

    /// <summary>
    /// NativeAOT-safe recursive JSON serializer for the limited types used in Jinja2 template rendering.
    /// Handles: null, bool, long, double, string, List&lt;object?&gt;, Dictionary&lt;string,object?&gt;.
    /// </summary>
    private static string ToJson(object? val)
    {
        if (val is null) return "null";
        if (val is bool b) return b ? "true" : "false";
        if (val is long l) return l.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (val is int i) return i.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (val is double d) return d.ToString("G", System.Globalization.CultureInfo.InvariantCulture);
        if (val is string s)
        {
            var sb = new System.Text.StringBuilder(s.Length + 2);
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"':  sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n");  break;
                    case '\r': sb.Append("\\r");  break;
                    case '\t': sb.Append("\\t");  break;
                    default:
                        if (c < 0x20) sb.Append($"\\u{(int)c:x4}");
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }
        if (val is List<object?> list)
        {
            var sb = new System.Text.StringBuilder("[");
            for (int k = 0; k < list.Count; k++)
            {
                if (k > 0) sb.Append(',');
                sb.Append(ToJson(list[k]));
            }
            sb.Append(']');
            return sb.ToString();
        }
        if (val is Dictionary<string, object?> dict)
        {
            var sb = new System.Text.StringBuilder("{");
            bool first = true;
            foreach (var kv in dict)
            {
                if (!first) sb.Append(',');
                first = false;
                sb.Append(ToJson(kv.Key));
                sb.Append(':');
                sb.Append(ToJson(kv.Value));
            }
            sb.Append('}');
            return sb.ToString();
        }
        // Fallback for NamespaceObj or other types
        return ToJson(Stringify(val));
    }

    private static int FindWordBoundary(string s, string word)
    {
        int idx = 0;
        while (idx <= s.Length - word.Length)
        {
            int found = s.IndexOf(word, idx, StringComparison.Ordinal);
            if (found < 0) return -1;
            bool leftOk  = found == 0 || !char.IsLetterOrDigit(s[found - 1]) && s[found - 1] != '_';
            bool rightOk = found + word.Length >= s.Length ||
                           !char.IsLetterOrDigit(s[found + word.Length]) && s[found + word.Length] != '_';
            if (leftOk && rightOk) return found;
            idx = found + 1;
        }
        return -1;
    }

    // ── Special objects ───────────────────────────────────────────────────

    private sealed class NamespaceObj
    {
        private readonly Dictionary<string, object?> _d = new(StringComparer.Ordinal);
        public void Set(string key, object? val) => _d[key] = val;
        public object? Get(string key) => _d.TryGetValue(key, out var v) ? v : null;
    }

    private sealed class LoopCtx(int index, int total)
    {
        public int Index { get; } = index;
        public int Total { get; } = total;
    }

    // ── Convenience ───────────────────────────────────────────────────────

    /// <summary>
    /// Build a messages list suitable for passing to Render().
    /// </summary>
    public static List<object?> BuildMessages(
        string userContent, string? systemContent = null, string? assistantPreamble = null)
    {
        var messages = new List<object?>();
        if (systemContent != null)
            messages.Add(new Dictionary<string, object?> { ["role"] = "system",    ["content"] = systemContent });
        messages.Add(new Dictionary<string, object?> { ["role"] = "user",      ["content"] = userContent });
        if (assistantPreamble != null)
            messages.Add(new Dictionary<string, object?> { ["role"] = "assistant", ["content"] = assistantPreamble });
        return messages;
    }
}
