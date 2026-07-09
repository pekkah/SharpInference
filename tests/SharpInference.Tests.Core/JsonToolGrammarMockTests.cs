using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-independent decode-time conformance for the JSON tool-argument grammar constraint
/// (issue #376) using <see cref="FakeJsonTokenizer"/>, so the byte-level masking is covered in CI
/// without a multi-gigabyte GGUF. Mirrors <see cref="ToolGrammarMockTests"/> (the Gemma sibling) for
/// the standard-JSON families: required key, foreign-key rejection, enum, string-with-escapes,
/// number, nested object, array, the merged-{} early-engage, and the Qwen/Llama/DeepSeek envelopes.
/// </summary>
public sealed class JsonToolGrammarMockTests
{
    private static (ITokenConstraint c, FakeJsonTokenizer tok, int vocab) Build(
        string schemaJson, string toolName, IToolCallAdapter? adapter = null)
    {
        var tok = new FakeJsonTokenizer();
        var vocab = new GrammarVocabulary(tok);
        using var doc = JsonDocument.Parse(schemaJson);
        var schema = ToolSchema.FromOpenAiFunction(toolName, doc.RootElement.Clone());
        var c = (adapter ?? new QwenToolCallAdapter("qwen3")).BuildArgumentConstraint([schema], vocab);
        Assert.NotNull(c);
        return (c!, tok, vocab.VocabSize);
    }

    private static void Feed(ITokenConstraint c, FakeJsonTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    // The canonical Qwen preamble up to and including the args key's colon — engagement point.
    private const string QwenPreamble = "<tool_call>{\"name\":\"get_weather\",\"arguments\":";

    [Fact]
    public void EngagesAtArgsColon_EarlyEngage_RejectsMergedEmptyObject()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble);     // ends right after `"arguments":`
        Assert.True(c.IsConstraining);  // engaged at the colon, BEFORE the '{'

        // Only an explicit '{' (optionally as part of a merged '{"') may open the object — a merged
        // "{}" token (Qwen's empty-args encoding) is rejected because it would drop the required key.
        Assert.True(Allowed(c, vocab, tok.Char('{')));
        Assert.True(Allowed(c, vocab, tok.Merged("{\"")));
        Assert.False(Allowed(c, vocab, tok.Merged("{}")));
        Assert.False(Allowed(c, vocab, tok.Char('}')));
    }

    [Fact]
    public void RequiredKey_CannotClose_UntilEmitted()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{");
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // required key missing
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // a key opens with '"'
        Assert.False(Allowed(c, vocab, tok.Char('x')));   // a key cannot start with a bare letter
    }

    [Fact]
    public void OnlyDeclaredKeys_AreReachable()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}""",
            "web_search");

        Feed(c, tok, "<tool_call>{\"name\":\"web_search\",\"arguments\":{\"");
        // Inside the key now: only 'q' (query) continues; never 'i' (a hallucinated 'queries').
        Assert.True(Allowed(c, vocab, tok.Char('q')));
        Assert.False(Allowed(c, vocab, tok.Char('z')));
        Feed(c, tok, "quer");
        Assert.True(Allowed(c, vocab, tok.Char('y')));
        Assert.False(Allowed(c, vocab, tok.Char('i')));
    }

    [Fact]
    public void StringValue_OpensWithQuote_FreeContent_Closes()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{\"location\"");
        Assert.True(Allowed(c, vocab, tok.Char(':')));
        Feed(c, tok, ":");
        // Value is a string → only '"' (or ws) may open it; not a bare letter or '}'.
        Assert.True(Allowed(c, vocab, tok.Char('"')));
        Assert.False(Allowed(c, vocab, tok.Char('}')));
        Assert.False(Allowed(c, vocab, tok.Char('B')));

        Feed(c, tok, "\"");
        // Free content: any byte stays; '"' closes; EOS forbidden mid-call.
        Assert.True(Allowed(c, vocab, tok.Char('B')));
        Assert.True(Allowed(c, vocab, tok.Char('"')));
        Assert.False(Allowed(c, vocab, FakeJsonTokenizer.Eos));

        Feed(c, tok, "Berlin\"");
        Assert.True(Allowed(c, vocab, tok.Char('}')));    // required satisfied → may close
        // Every declared key is emitted, so ',' would commit to a key that cannot exist — masked
        // (issue #425's comma gate), where it previously livelocked in a whitespace-only
        // OExpectKey state ('}' isn't legal there and EOG is forbidden mid-call).
        Assert.False(Allowed(c, vocab, tok.Char(',')));

        Feed(c, tok, "}");
        Assert.False(c.IsConstraining);                   // object closed → back to watching
    }

    [Fact]
    public void EscapedQuote_DoesNotCloseStringEarly()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}""",
            "echo");

        Feed(c, tok, "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"a");
        Assert.True(c.IsConstraining);
        // A backslash-escaped quote stays INSIDE the string. If the escape were ignored, this '"'
        // would close the string and the 'b' below would diverge and disable the constraint — so the
        // post-'b' IsConstraining check is what proves the escape was honored. ('}' is meaningless
        // here: inside a free string it is ordinary content, not an object close.)
        Feed(c, tok, "\\\"");                             // the two bytes \ and "
        Assert.True(c.IsConstraining);
        Feed(c, tok, "b\"");                              // content 'b', then an unescaped closing '"'
        Assert.True(c.IsConstraining);                    // cleanly back at object level — not disabled
        Assert.True(Allowed(c, vocab, tok.Char('}')));    // required satisfied → the object may close
        Feed(c, tok, "}");
        Assert.False(c.IsConstraining);                   // clean close
    }

    [Fact]
    public void Enum_RestrictsToDeclaredValues()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["unit"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{\"unit\":\"");
        Assert.True(Allowed(c, vocab, tok.Char('c')));    // celsius
        Assert.True(Allowed(c, vocab, tok.Char('f')));    // fahrenheit
        Assert.False(Allowed(c, vocab, tok.Char('x')));   // neither

        Feed(c, tok, "celsius");
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // value complete → close
        Assert.False(Allowed(c, vocab, tok.Char('z')));   // can't extend past the enum
    }

    [Fact]
    public void NumberValue_AcceptsDigits_ThenCloses()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{\"days\":");
        Assert.True(Allowed(c, vocab, tok.Char('3')));
        Assert.False(Allowed(c, vocab, tok.Char('"')));   // integer is bare, not quoted
        Assert.False(Allowed(c, vocab, tok.Char('.')));   // integer → no decimal point

        Feed(c, tok, "3");
        Assert.True(Allowed(c, vocab, tok.Char('0')));    // more digits
        Assert.True(Allowed(c, vocab, tok.Char('}')));    // or close (required satisfied)
    }

    [Fact]
    public void NestedObjectValue_ExpectsBrace_ThenConstrainsInnerKeys()
    {
        var (c, tok, vocab) = Build(
            """
            {"type":"object","properties":{
               "filter":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}},
             "required":["filter"]}
            """,
            "search");

        Feed(c, tok, "<tool_call>{\"name\":\"search\",\"arguments\":{\"filter\":");
        Assert.True(Allowed(c, vocab, tok.Char('{')));    // object value opens with '{'
        Assert.False(Allowed(c, vocab, tok.Char('"')));   // not a string
        Assert.False(Allowed(c, vocab, tok.Char('3')));   // not a number

        Feed(c, tok, "{");
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // inner required "city" missing
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // inner key opens with '"'
    }

    [Fact]
    public void ArrayValue_OpensWithBracket_ConstrainsItems()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}},"required":["tags"]}""",
            "tagger");

        Feed(c, tok, "<tool_call>{\"name\":\"tagger\",\"arguments\":{\"tags\":");
        Assert.True(Allowed(c, vocab, tok.Char('[')));    // array opens with '['
        Assert.False(Allowed(c, vocab, tok.Char('"')));

        Feed(c, tok, "[");
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // a string item opens with '"'
        Assert.True(Allowed(c, vocab, tok.Char(']')));    // or close (empty array)
        Assert.False(Allowed(c, vocab, tok.Char('5')));   // a bare number isn't a string item
    }

    [Fact]
    public void UnknownTool_IsNotConstrained()
    {
        var (c, tok, _) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<tool_call>{\"name\":\"some_other_tool\",\"arguments\":");
        Assert.False(c.IsConstraining);   // name not in the constrainable set → stays passive
    }

    [Fact]
    public void Reset_ReturnsToWatching()
    {
        var (c, tok, _) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{");
        Assert.True(c.IsConstraining);
        c.Reset();
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void Llama_PythonTagEnvelope_ParametersKey()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather", new LlamaToolCallAdapter());

        Feed(c, tok, "<|python_tag|>{\"name\":\"get_weather\",\"parameters\":");
        Assert.True(c.IsConstraining);                    // Llama uses "parameters", not "arguments"
        Assert.False(Allowed(c, vocab, tok.Merged("{}"))); // still early-engaged
        Assert.True(Allowed(c, vocab, tok.Char('{')));
    }

    [Fact]
    public void DeepSeek_NameThenSeparatorEnvelope()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather", new DeepSeekToolCallAdapter());

        // DeepSeek: <|tool_call_begin|>NAME<|tool_sep|>{json}. Name is bare text up to the separator.
        Feed(c, tok, "<|tool_call_begin|>get_weather<|tool_sep|>");
        Assert.True(c.IsConstraining);                    // engaged on the separator
        Assert.True(Allowed(c, vocab, tok.Char('{')));
        Assert.False(Allowed(c, vocab, tok.Merged("{}"))); // required key still enforced
    }

    [Fact]
    public void PartiallyTyped_FreeValue_StillEnforcesTypedRequiredKey()
    {
        // 'context' is an open object (no properties) → a free value; 'location' stays a required
        // string. Before #378 the whole tool was dropped (Build would return null); now it compiles.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"},"context":{"type":"object"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, QwenPreamble + "{");
        Assert.True(c.IsConstraining);
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // required 'location' still missing

        // Emit the loosely-typed 'context' first — its value may be an object, string, or bare scalar.
        Feed(c, tok, "\"context\":");
        Assert.True(Allowed(c, vocab, tok.Char('{')));
        Assert.True(Allowed(c, vocab, tok.Char('"')));
        Assert.True(Allowed(c, vocab, tok.Char('5')));
        Assert.False(Allowed(c, vocab, tok.Char(',')));   // a value can't be empty

        // A free object with arbitrary inner keys/nesting is accepted whole.
        Feed(c, tok, "{\"anything\":42,\"nested\":{\"x\":[1,2]}}");
        Assert.True(c.IsConstraining);
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // 'location' STILL required after the free value

        Feed(c, tok, ",\"location\":\"Paris\"");
        Assert.True(Allowed(c, vocab, tok.Char('}')));    // required satisfied → may close
        Feed(c, tok, "}");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void PartiallyTyped_AnyValue_AndUntypedArray_AreFree()
    {
        // 'meta' has no type (Any) and 'tags' is an untyped array — both free; 'id' stays required int.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"id":{"type":"integer"},"meta":{},"tags":{"type":"array"}},"required":["id"]}""",
            "save");

        Feed(c, tok, "<tool_call>{\"name\":\"save\",\"arguments\":{");
        Assert.True(c.IsConstraining);

        Feed(c, tok, "\"meta\":");
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // Any → free: string ok
        Assert.True(Allowed(c, vocab, tok.Char('[')));    // …or array
        Feed(c, tok, "\"x\",\"tags\":[1,\"a\",{\"k\":2}]");  // free string, then free untyped array
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // required 'id' still missing
        Feed(c, tok, ",\"id\":7");
        Assert.True(Allowed(c, vocab, tok.Char('}')));
    }

    [Fact]
    public void TypedArray_OfFreeItems_AcceptsAnyItemShape()
    {
        // A typed array whose ITEM type is loose ({}) — the array structure is enforced, each item is
        // free. Regression: the first-byte prune must admit non-numeric free items (string/object), not
        // only numbers.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"items":{"type":"array","items":{}}},"required":["items"]}""",
            "save");

        Feed(c, tok, "<tool_call>{\"name\":\"save\",\"arguments\":{\"items\":[");
        Assert.True(c.IsConstraining);
        Assert.True(Allowed(c, vocab, tok.Char('"')));   // string item
        Assert.True(Allowed(c, vocab, tok.Char('{')));   // object item
        Assert.True(Allowed(c, vocab, tok.Char('5')));   // number item
        Assert.True(Allowed(c, vocab, tok.Char(']')));   // or close (empty)

        Feed(c, tok, "1,\"a\",{\"k\":2}]");              // mixed free items
        Assert.True(Allowed(c, vocab, tok.Char('}')));   // array done, 'items' satisfied
    }

    [Fact]
    public void NonConstrainableTool_BuildsNoConstraint()
    {
        var tok = new FakeJsonTokenizer();
        var vocab = new GrammarVocabulary(tok);
        // An open object (no properties) isn't constrainable → adapter returns null.
        using var doc = JsonDocument.Parse("""{"type":"object"}""");
        var schema = ToolSchema.FromOpenAiFunction("noop", doc.RootElement.Clone());
        Assert.Null(new QwenToolCallAdapter("qwen3").BuildArgumentConstraint([schema], vocab));
    }

    // The Qwen adapter overlays the JSON (#376) and Qwen3-Coder XML (#383) constraints in a
    // CompositeToolArgumentConstraint, because the same architecture hosts both formats (#383). These
    // exercise the composite's format dispatch model-free, so a CI runner without the GGUFs covers it.

    [Fact]
    public void Composite_DispatchesToXmlSub_OnCoderOutput()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");
        Assert.IsType<CompositeToolArgumentConstraint>(c);

        // Coder XML output engages the XML sub through the composite and enforces the required param.
        Feed(c, tok, "<tool_call>\n<function=get_weather>");
        Assert.True(c.IsConstraining);
        Feed(c, tok, "<");
        Assert.True(Allowed(c, vocab, tok.Char('p')));    // <parameter=
        Assert.False(Allowed(c, vocab, tok.Char('/')));   // </function> forbidden — 'location' missing
    }

    [Fact]
    public void Composite_DispatchesToJsonSub_OnJsonOutput()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");
        Assert.IsType<CompositeToolArgumentConstraint>(c);

        // JSON output engages the JSON sub through the composite — unchanged from #376.
        Feed(c, tok, QwenPreamble);
        Assert.True(c.IsConstraining);
        Assert.False(Allowed(c, vocab, tok.Merged("{}")));  // merged empty-object still rejected
        Assert.True(Allowed(c, vocab, tok.Char('{')));

        // Reset returns the whole composite to the watching (pass-through) state.
        c.Reset();
        Assert.False(c.IsConstraining);
    }
}
