using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using Xunit.Abstractions;

namespace SharpInference.Tests.Core;

/// <summary>
/// Decode-time conformance for the Gemma 4 tool-argument grammar constraint (issue #374). Drives a
/// real Gemma vocabulary through the constraint and asserts the per-token mask forbids the exact
/// failure modes the issue describes (dropped required key, hallucinated <c>queries</c> array,
/// out-of-enum value). Model-gated — skips when the GGUF is absent.
/// </summary>
public sealed class ToolGrammarConstraintTests(ITestOutputHelper output)
{
    private const string ModelPath = @"E:\models\gemma-4-12b-it-qat-q4_0.gguf";

    private static GgufTokenizer? Tok()
    {
        if (!File.Exists(ModelPath)) return null;
        var m = GgufModel.Open(ModelPath);
        return GgufTokenizer.FromGgufModel(m);
    }

    private static ToolSchema Schema(string name, string parametersJson)
    {
        using var doc = JsonDocument.Parse(parametersJson);
        return ToolSchema.FromOpenAiFunction(name, doc.RootElement.Clone());
    }

    // Feed every token of `text` into the constraint via Accept.
    private static void Feed(ITokenConstraint c, GgufTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, GgufTokenizer tok, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];   // all zeros
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    private static int Id(GgufTokenizer tok, string single)
    {
        var ids = tok.Encode(single);
        Assert.Single(ids);
        return ids[0];
    }

    [Fact]
    public void RequiredKey_CannotBeOmitted()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);

        var schema = Schema("get_weather",
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""");
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab);
        Assert.NotNull(c);

        Feed(c!, tok, "<|tool_call>call:get_weather{");
        Assert.True(c!.IsConstraining);

        int closeBrace = Id(tok, "}");
        int locationKey = Id(tok, "location");
        // Required `location` not yet emitted → the model may NOT close the object.
        Assert.False(Allowed(c, tok, vocab.VocabSize, closeBrace));
        // …and the `location` key IS reachable.
        Assert.True(Allowed(c, tok, vocab.VocabSize, locationKey));
    }

    [Fact]
    public void WebSearch_ForcesQueryKey_RejectsForeignKeys()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);

        var schema = Schema("web_search",
            """{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}""");
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab)!;

        Feed(c, tok, "<|tool_call>call:web_search{");
        Assert.True(c.IsConstraining);

        Assert.True(Allowed(c, tok, vocab.VocabSize, Id(tok, "query")));
        // A key from a DIFFERENT tool's schema is unreachable — only declared keys allowed.
        Assert.False(Allowed(c, tok, vocab.VocabSize, Id(tok, "location")));
        // The hallucinated array form `queries` cannot even begin: after consuming `quer`, the only
        // continuation toward a declared key is `y` (query), never `i` (queries).
    }

    [Fact]
    public void StringValue_FreeContentThenCloseQuote()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        int quote = vocab.TryGetSpecialToken("<|\"|>", out int q) ? q : -1;
        Assert.True(quote > 0);

        var schema = Schema("get_weather",
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""");
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab)!;

        Feed(c, tok, "<|tool_call>call:get_weather{location:");
        // Value is a string → only the open quote may follow.
        Assert.True(Allowed(c, tok, vocab.VocabSize, quote));
        Assert.False(Allowed(c, tok, vocab.VocabSize, Id(tok, "}")));

        c.Accept(quote);                              // open the string
        // Inside the string: free content (e.g. "Berlin") allowed; EOS forbidden; quote closes.
        Assert.True(Allowed(c, tok, vocab.VocabSize, tok.Encode("Berlin")[0]));
        Assert.True(Allowed(c, tok, vocab.VocabSize, quote));
        Assert.False(Allowed(c, tok, vocab.VocabSize, tok.EosTokenId));

        Feed(c, tok, "Berlin");
        c.Accept(quote);                              // close the string
        // location satisfied → `}` and `,` both legal now.
        Assert.True(Allowed(c, tok, vocab.VocabSize, Id(tok, "}")));
        Assert.True(Allowed(c, tok, vocab.VocabSize, Id(tok, ",")));

        c.Accept(Id(tok, "}"));
        Assert.False(c.IsConstraining);               // object closed → back to watching
    }

    [Fact]
    public void EnumValue_RestrictedToDeclaredSet()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        int quote = vocab.TryGetSpecialToken("<|\"|>", out int q) ? q : -1;

        var schema = Schema("get_weather",
            """{"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["unit"]}""");
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab)!;

        Feed(c, tok, "<|tool_call>call:get_weather{unit:");
        c.Accept(quote);                              // open the enum string

        // Both enum values begin with tokens that are reachable; a foreign letter is not.
        Assert.True(Allowed(c, tok, vocab.VocabSize, tok.Encode("celsius")[0]));   // 'c…'
        Assert.True(Allowed(c, tok, vocab.VocabSize, tok.Encode("fahrenheit")[0])); // 'fahren'
        Assert.False(Allowed(c, tok, vocab.VocabSize, tok.Encode("xenon")[0]));     // 'x…' not an enum prefix

        Feed(c, tok, "celsius");
        Assert.True(Allowed(c, tok, vocab.VocabSize, quote));   // completed value → close quote allowed
    }

    [Fact]
    public void PartiallyTyped_RequiredTypedKey_Enforced_LooseValueFree()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);

        // 'location' is a required string; 'context' is an open object (free value). Issue #378: the
        // tool is now constrained on its typed/required parts instead of being dropped wholesale.
        var schema = Schema("get_weather",
            """{"type":"object","properties":{"location":{"type":"string"},"context":{"type":"object"}},"required":["location"]}""");
        var c = new Gemma4ToolCallAdapter().BuildArgumentConstraint([schema], vocab);
        Assert.NotNull(c);

        Feed(c!, tok, "<|tool_call>call:get_weather{");
        Assert.True(c!.IsConstraining);
        // Required 'location' not yet emitted → may not close; both declared keys reachable.
        Assert.False(Allowed(c, tok, vocab.VocabSize, Id(tok, "}")));
        Assert.True(Allowed(c, tok, vocab.VocabSize, tok.Encode("location")[0]));
        Assert.True(Allowed(c, tok, vocab.VocabSize, tok.Encode("context")[0]));
    }

    [Fact]
    public void NonGemmaAdapter_BuildsNoConstraint()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        var schema = Schema("get_weather", """{"type":"object","properties":{"location":{"type":"string"}}}""");
        Assert.Null(ToolCallAdapterRegistry.Get("qwen3").BuildArgumentConstraint([schema], vocab));
    }
}
