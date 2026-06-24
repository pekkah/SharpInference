using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-independent decode-time conformance for the Qwen3-Coder XML tool-argument grammar constraint
/// (issue #383) using <see cref="FakeCoderTokenizer"/>, so the byte-level masking is covered in CI
/// without the multi-gigabyte GGUF. The XML sibling of <see cref="JsonToolGrammarMockTests"/>: required
/// parameter, foreign-key rejection, bare enum/number/boolean values, free-text strings, the
/// early-engage that blocks an immediate <c>&lt;/function&gt;</c> when a required parameter is missing,
/// partially-typed free values, multi-parameter required-once, and re-arming across functions.
/// </summary>
public sealed class CoderToolGrammarMockTests
{
    private static (QwenCoderToolArgumentConstraint c, FakeCoderTokenizer tok, int vocab) Build(
        string schemaJson, string toolName)
    {
        var tok = new FakeCoderTokenizer();
        var vocab = new GrammarVocabulary(tok);
        using var doc = JsonDocument.Parse(schemaJson);
        var schema = ToolSchema.FromOpenAiFunction(toolName, doc.RootElement.Clone());
        var c = new QwenCoderToolCallAdapter().BuildArgumentConstraint([schema], vocab);
        Assert.NotNull(c);
        return ((QwenCoderToolArgumentConstraint)c!, tok, vocab.VocabSize);
    }

    private static void Feed(ITokenConstraint c, FakeCoderTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    // The canonical Coder preamble up to and including the '>' closing <function=NAME> — engage point.
    private static string Preamble(string name) => $"<tool_call>\n<function={name}>";

    [Fact]
    public void EngagesAtFunctionTag_BeforeFirstParameter()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather"));
        Assert.True(c.IsConstraining);                    // engaged at the '>', BEFORE any <parameter=

        // At the function body root only whitespace or a tag-opening '<' is legal — never bare text.
        Assert.True(Allowed(c, vocab, tok.Char('<')));
        Assert.True(Allowed(c, vocab, tok.Char('\n')));
        Assert.False(Allowed(c, vocab, tok.Char('x')));
        Assert.False(Allowed(c, vocab, tok.Char('>')));
    }

    [Fact]
    public void EarlyEngage_RequiredParam_BlocksImmediateFunctionClose()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<");      // now matching an open tag after '<'
        // Only <parameter= may continue ('p'); </function> is forbidden ('/') because the required
        // 'location' hasn't been emitted — the merged-{} analogue: the call can't close empty.
        Assert.True(Allowed(c, vocab, tok.Char('p')));
        Assert.False(Allowed(c, vocab, tok.Char('/')));
    }

    [Fact]
    public void MergedClosingBracket_EngagesMidToken()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        // Feed up to the name WITHOUT the '>', then deliver a merged ">\n" token (the realistic
        // post-tag merge). Engagement must happen mid-token on the '>', with the trailing '\n' replayed
        // into the (now constrained) function body.
        Feed(c, tok, "<tool_call>\n<function=get_weather");
        Assert.False(c.IsConstraining);
        c.Accept(tok.Merged(">\n"));
        Assert.True(c.IsConstraining);

        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('p')));
        Assert.False(Allowed(c, vocab, tok.Char('/')));   // still can't close — required missing
    }

    [Fact]
    public void OnlyDeclaredKeys_AreReachable()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}""",
            "web_search");

        Feed(c, tok, Preamble("web_search") + "<parameter=");
        // Inside the parameter key now: only 'q' (query) continues; never 'i' (a hallucinated 'queries').
        Assert.True(Allowed(c, vocab, tok.Char('q')));
        Assert.False(Allowed(c, vocab, tok.Char('z')));
        Feed(c, tok, "quer");
        Assert.True(Allowed(c, vocab, tok.Char('y')));
        Assert.False(Allowed(c, vocab, tok.Char('i')));
        Feed(c, tok, "y");
        Assert.True(Allowed(c, vocab, tok.Char('>')));    // 'query' complete → close the key tag
        Assert.False(Allowed(c, vocab, tok.Char('s')));   // can't extend past a declared key
    }

    [Fact]
    public void FreeStringValue_AcceptsContent_ClosesOnParamTag()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=location>");
        // A string value is free content: any byte stays, EOS forbidden mid-call.
        Assert.True(Allowed(c, vocab, tok.Char('B')));
        Assert.True(Allowed(c, vocab, tok.Char('\n')));
        Assert.True(Allowed(c, vocab, tok.Char('<')));    // may begin the </parameter> close
        Assert.False(Allowed(c, vocab, FakeCoderTokenizer.Eos));

        Feed(c, tok, "\nParis\n</parameter>");
        // Back at the function body: required 'location' satisfied → may close, or open another tag.
        Assert.True(Allowed(c, vocab, tok.Char('<')));
        Feed(c, tok, "</function>");
        Assert.False(c.IsConstraining);                   // function closed → back to watching
    }

    [Fact]
    public void Enum_RestrictsToDeclaredValues_BareNotQuoted()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["unit"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=unit>\n");
        // Coder enum values are BARE text (no quotes): only declared prefixes after the leading newline.
        Assert.True(Allowed(c, vocab, tok.Char('c')));    // celsius
        Assert.True(Allowed(c, vocab, tok.Char('f')));    // fahrenheit
        Assert.False(Allowed(c, vocab, tok.Char('x')));   // neither
        Assert.False(Allowed(c, vocab, tok.Char('"')));   // not quoted

        Feed(c, tok, "celsius");
        Assert.False(Allowed(c, vocab, tok.Char('z')));   // can't extend past the enum
        Assert.True(Allowed(c, vocab, tok.Char('<')));    // value complete → start </parameter>
        Assert.True(Allowed(c, vocab, tok.Char('\n')));   // …or trailing whitespace first
    }

    [Fact]
    public void NumberValue_AcceptsDigits_ThenCloses()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=days>\n");
        Assert.True(Allowed(c, vocab, tok.Char('3')));
        Assert.False(Allowed(c, vocab, tok.Char('.')));   // integer → no decimal point
        Assert.False(Allowed(c, vocab, tok.Char('x')));

        Feed(c, tok, "3");
        Assert.True(Allowed(c, vocab, tok.Char('0')));    // more digits
        Assert.True(Allowed(c, vocab, tok.Char('<')));    // …or begin the close
    }

    [Fact]
    public void BooleanValue_RestrictedToTrueFalse()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"metric":{"type":"boolean"}},"required":["metric"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=metric>\n");
        Assert.True(Allowed(c, vocab, tok.Char('t')));    // true
        Assert.True(Allowed(c, vocab, tok.Char('f')));    // false
        Assert.False(Allowed(c, vocab, tok.Char('y')));   // not a boolean literal
    }

    [Fact]
    public void MultipleParams_RequiredOnce_NoRepeat()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","unit"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=location>\nParis\n</parameter>");
        Assert.True(c.IsConstraining);                    // close tag consumed cleanly, still in body
        // 'location' done but 'unit' still required → cannot close the function yet.
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('p')));    // another <parameter=
        Assert.False(Allowed(c, vocab, tok.Char('/')));   // </function> still forbidden ('unit' missing)

        Feed(c, tok, "parameter=");
        // The already-emitted 'location' must be unreachable; only 'unit' remains.
        Assert.True(Allowed(c, vocab, tok.Char('u')));
        Assert.False(Allowed(c, vocab, tok.Char('l')));   // 'location' can't repeat

        Feed(c, tok, "unit>\ncelsius\n</parameter>");
        Assert.True(c.IsConstraining);                    // both values + close tags consumed cleanly
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('/')));    // all required emitted → may close now
    }

    [Fact]
    public void AllParamsEmitted_OnlyCloseRemains()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        // The single declared parameter is emitted; another <parameter= would have no key to match, so
        // only </function> remains legal (prevents a dead-state from opening a keyless parameter).
        Feed(c, tok, Preamble("get_weather") + "<parameter=location>\nParis\n</parameter>");
        Assert.True(c.IsConstraining);
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('/')));    // </function>
        Assert.False(Allowed(c, vocab, tok.Char('p')));   // no <parameter= — all keys used
    }

    [Fact]
    public void PartiallyTyped_FreeValue_StillEnforcesRequired()
    {
        // 'context' is an open object (no properties) → a free value; 'location' stays a required
        // string. The free value's content is unconstrained but the surrounding structure isn't.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"},"context":{"type":"object"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=context>\n{\"any\":[1,2],\"k\":\"v\"}\n</parameter>");
        Assert.True(c.IsConstraining);
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('p')));    // another <parameter=
        Assert.False(Allowed(c, vocab, tok.Char('/')));   // required 'location' STILL missing after free value

        Feed(c, tok, "parameter=location>\nParis\n</parameter>");
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('/')));    // required satisfied → may close
        Feed(c, tok, "/function>");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void ClosesAndReArms_ForSecondFunctionInBlock()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        // First call closes; without a </tool_call> the constraint stays armed for a second function.
        Feed(c, tok, Preamble("get_weather") + "<parameter=location>\nParis\n</parameter>\n</function>");
        Assert.False(c.IsConstraining);                   // function closed

        Feed(c, tok, "\n<function=get_weather>");
        Assert.True(c.IsConstraining);                    // re-engaged on the second function
        Feed(c, tok, "<");
        Assert.False(Allowed(c, vocab, tok.Char('/')));   // required enforced again

        // The </tool_call> envelope token disarms entirely.
        c.Reset();
        Feed(c, tok, "<tool_call>");
        Feed(c, tok, "</tool_call>");
        Feed(c, tok, "<function=get_weather>");
        Assert.False(c.IsConstraining);                   // disarmed by </tool_call> → no engage
    }

    [Fact]
    public void UnknownTool_IsNotConstrained()
    {
        var (c, tok, _) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<tool_call>\n<function=some_other_tool>");
        Assert.False(c.IsConstraining);                   // name not in the constrainable set → passive
    }

    [Fact]
    public void Reset_ReturnsToWatching()
    {
        var (c, tok, _) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, Preamble("get_weather") + "<parameter=");
        Assert.True(c.IsConstraining);
        c.Reset();
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void NonConstrainableTool_BuildsNoConstraint()
    {
        var tok = new FakeCoderTokenizer();
        var vocab = new GrammarVocabulary(tok);
        // An open object (no properties) isn't constrainable → adapter returns null.
        using var doc = JsonDocument.Parse("""{"type":"object"}""");
        var schema = ToolSchema.FromOpenAiFunction("noop", doc.RootElement.Clone());
        Assert.Null(new QwenCoderToolCallAdapter().BuildArgumentConstraint([schema], vocab));
    }

    [Fact]
    public void DefaultOff_NoToolCall_NeverEngages()
    {
        var (c, tok, _) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        // A bare <function=…> with no <tool_call> arming token must stay inert (the constraint never
        // arms on raw text, so non-tool generation is byte-identical to unconstrained).
        Feed(c, tok, "here is some code: <function=get_weather><parameter=location>");
        Assert.False(c.IsConstraining);
    }
}
