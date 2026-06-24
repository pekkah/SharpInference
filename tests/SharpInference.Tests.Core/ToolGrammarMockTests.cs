using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-independent decode-time conformance for the Gemma argument-grammar constraint (issue #374)
/// using <see cref="FakeGemmaTokenizer"/>, so the core masking logic is covered in CI without the
/// GGUF. Mirrors the failure modes from the issue: dropped required key, foreign/hallucinated key,
/// out-of-enum value.
/// </summary>
public sealed class ToolGrammarMockTests
{
    private static (ITokenConstraint c, FakeGemmaTokenizer tok, int vocab) Build(string schemaJson, string toolName)
    {
        var tok = new FakeGemmaTokenizer();
        var vocab = new GrammarVocabulary(tok);
        using var doc = JsonDocument.Parse(schemaJson);
        var schema = ToolSchema.FromOpenAiFunction(toolName, doc.RootElement.Clone());
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab);
        Assert.NotNull(c);
        return (c!, tok, vocab.VocabSize);
    }

    private static void Feed(ITokenConstraint c, FakeGemmaTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    [Fact]
    public void RequiredKey_CannotClose_UntilEmitted()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{");
        Assert.True(c.IsConstraining);

        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));   // required key missing
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('l')));    // 'l' starts "location"
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('x')));   // no key starts with 'x'
    }

    [Fact]
    public void OnlyDeclaredKeys_AreReachable()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}""",
            "web_search");

        Feed(c, tok, "<|tool_call>call:web_search{");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('q')));    // 'q' starts "query"

        // Walk "quer"; the only legal continuation is 'y' (query) — never 'i' (queries).
        Feed(c, tok, "quer");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('y')));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('i')));
    }

    [Fact]
    public void StringValue_OpensWithQuote_FreeContent_Closes()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{location:");
        // Value is a string → only the quote may follow.
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));

        c.Accept(FakeGemmaTokenizer.Quote);
        // Free content: any non-EOS token; the quote closes; EOS forbidden mid-call.
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('B')));
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Eos));

        Feed(c, tok, "Berlin");
        c.Accept(FakeGemmaTokenizer.Quote);                 // close string
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));   // required satisfied
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char(',')));

        c.Accept(FakeGemmaTokenizer.Char('}'));
        Assert.False(c.IsConstraining);                     // object closed → watching
    }

    [Fact]
    public void Enum_RestrictsToDeclaredValues()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["unit"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{unit:");
        c.Accept(FakeGemmaTokenizer.Quote);                 // open the enum string

        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('c')));   // celsius
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('f')));   // fahrenheit
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('x')));  // neither

        Feed(c, tok, "celsius");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));       // value complete → close
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('z')));  // can't extend past the enum
    }

    [Fact]
    public void NumberValue_AcceptsDigits_ThenCloses()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{days:");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('3')));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Quote));      // integer is bare, not quoted
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('.')));  // integer → no decimal point

        c.Accept(FakeGemmaTokenizer.Char('3'));
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('0')));   // more digits
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));   // or close (required satisfied)
    }

    [Fact]
    public void EarlyEngage_ForcesOpenBrace_BeforeMergedEmptyObject()
    {
        // The constraint must engage the instant the tool name completes — BEFORE the '{' — so a
        // merged "{}" token (Gemma's empty-args encoding) can't slip past. Here we stop right after
        // the name and assert the object can only be OPENED (never closed-empty).
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather");   // note: no '{' yet
        Assert.True(c.IsConstraining);                  // engaged early on the name match

        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('{')));   // may open the object
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));  // may NOT close it (none opened)
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('l')));  // a key can't precede '{'
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

        Feed(c, tok, "<|tool_call>call:search{filter:");
        // The value is an object → it must open with '{', not a quote or a bare scalar.
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('{')));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Quote));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('3')));

        c.Accept(FakeGemmaTokenizer.Char('{'));
        // Inner object: required "city" missing → can't close; 'c' begins the key.
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('c')));
    }

    [Fact]
    public void ArrayValue_OpensWithBracket_ConstrainsItems()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}},"required":["tags"]}""",
            "tagger");

        Feed(c, tok, "<|tool_call>call:tagger{tags:");
        // Array value → opens with '[', not a quote.
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('[')));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Quote));

        c.Accept(FakeGemmaTokenizer.Char('['));
        // Inside the array: a string item opens with a quote, or the array closes (empty).
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char(']')));
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('x')));   // bare scalar not valid for a string item
    }

    [Fact]
    public void UnknownTool_IsNotConstrained()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        // A call to a tool the constraint doesn't know about stays passive.
        Feed(c, tok, "<|tool_call>call:some_other_tool{");
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void Reset_ReturnsToWatching()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{");
        Assert.True(c.IsConstraining);
        c.Reset();
        Assert.False(c.IsConstraining);
    }

    [Fact]
    public void TypedArray_OfFreeItems_AcceptsStringAndObjectItems()
    {
        // A typed array of loose items ({}): a free string item opens on the <|"|> quote, a free
        // object on '{', a scalar bare. Regression for the array-item first-byte / quote path.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"items":{"type":"array","items":{}}},"required":["items"]}""",
            "save");

        Feed(c, tok, "<|tool_call>call:save{items:[");
        Assert.True(c.IsConstraining);
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));     // free string item opens on the quote
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('{'))); // free object item
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('5'))); // bare scalar item
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char(']'))); // or close (empty)

        c.Accept(FakeGemmaTokenizer.Quote); Feed(c, tok, "a"); c.Accept(FakeGemmaTokenizer.Quote);
        Feed(c, tok, ",{k:3}]");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));
    }

    [Fact]
    public void PartiallyTyped_FreeValue_StillEnforcesTypedRequiredKey()
    {
        // 'context' is an open object → a free value; 'location' stays a required string. Before #378
        // the whole tool was dropped (Build would return null); now it compiles and enforces the
        // typed/required parts while leaving 'context' free.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"location":{"type":"string"},"context":{"type":"object"}},"required":["location"]}""",
            "get_weather");

        Feed(c, tok, "<|tool_call>call:get_weather{");
        Assert.True(c.IsConstraining);
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));   // required 'location' missing

        // Emit the loosely-typed 'context' first — its value may be a string, object, array, or scalar.
        Feed(c, tok, "context:");
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Quote));        // <|"|> string
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('{')));    // object
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('[')));    // array
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('3')));    // bare scalar

        // A free object with a <|"|>-string value and a nested array is balanced whole.
        Feed(c, tok, "{a:");
        c.Accept(FakeGemmaTokenizer.Quote); Feed(c, tok, "x"); c.Accept(FakeGemmaTokenizer.Quote);
        Feed(c, tok, ",b:[1,2]}");
        Assert.True(c.IsConstraining);
        Assert.False(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));   // 'location' STILL required

        Feed(c, tok, ",location:");
        c.Accept(FakeGemmaTokenizer.Quote); Feed(c, tok, "Paris"); c.Accept(FakeGemmaTokenizer.Quote);
        Assert.True(Allowed(c, vocab, FakeGemmaTokenizer.Char('}')));    // required satisfied
        c.Accept(FakeGemmaTokenizer.Char('}'));
        Assert.False(c.IsConstraining);                                  // clean close
    }
}
