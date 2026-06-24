using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using Xunit.Abstractions;

namespace SharpInference.Tests.Core;

/// <summary>
/// Decode-time conformance for the Qwen3-Coder XML tool-argument grammar constraint (issue #383)
/// against a REAL Qwen3-Coder vocabulary, so the byte-level matching is exercised over the actual BPE
/// merges (where a single token can carry a whole <c>&lt;parameter=</c> tag, or close one parameter
/// and open the next). Model-gated — skips when the GGUF is absent. The grammar logic itself is covered
/// model-free by <see cref="CoderToolGrammarMockTests"/>; this asserts the same invariants survive a
/// production tokenizer.
/// </summary>
public sealed class CoderToolGrammarConstraintTests(ITestOutputHelper output)
{
    private static readonly string[] s_modelPaths =
    [
        @"E:\models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
    ];

    private static GgufTokenizer? Tok()
    {
        foreach (var p in s_modelPaths)
            if (File.Exists(p))
            {
                var m = GgufModel.Open(p);
                return GgufTokenizer.FromGgufModel(m);
            }
        return null;
    }

    private static ToolSchema Schema(string name, string parametersJson)
    {
        using var doc = JsonDocument.Parse(parametersJson);
        return ToolSchema.FromOpenAiFunction(name, doc.RootElement.Clone());
    }

    private static void Feed(ITokenConstraint c, GgufTokenizer tok, string text)
    {
        foreach (int id in tok.Encode(text)) c.Accept(id);
    }

    private static bool Allowed(ITokenConstraint c, int vocab, int tokenId)
    {
        Span<float> logits = new float[vocab];
        var masked = c.Filter(logits);
        return !float.IsNegativeInfinity(masked[tokenId]);
    }

    // The first token that emits exactly `s` (handles multi-token strings: returns the leading token).
    private static int First(GgufTokenizer tok, string s) => tok.Encode(s)[0];

    private const string Weather =
        """{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]},"days":{"type":"integer"}},"required":["location"]}""";

    private QwenCoderToolArgumentConstraint? Constraint(GgufTokenizer tok, out GrammarVocabulary vocab)
    {
        vocab = new GrammarVocabulary(tok);
        return (QwenCoderToolArgumentConstraint?)
            new QwenCoderToolCallAdapter().BuildArgumentConstraint([Schema("get_weather", Weather)], vocab);
    }

    [Fact]
    public void EngagesAtFunctionTag_BlocksImmediateClose()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var c = Constraint(tok, out var vocab);
        Assert.NotNull(c);   // <tool_call> resolved as a special token → constrainable

        Feed(c!, tok, "<tool_call>\n<function=get_weather>");
        Assert.True(c!.IsConstraining);                    // engaged at the '>' closing the function tag

        // At the body root, a '<' (tag open) and whitespace are legal; bare text is not.
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "<")));
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "x")));

        Feed(c, tok, "<");
        // Only <parameter= may continue; </function> ('/') is forbidden — required 'location' missing.
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "p")));
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "/")));
    }

    [Fact]
    public void RequiredKey_CannotBeOmitted_ForeignKeyRejected()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var c = Constraint(tok, out var vocab)!;

        Feed(c, tok, "<tool_call>\n<function=get_weather>\n<parameter=");
        // Now matching a parameter key: declared names are reachable; a foreign letter is not.
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "location")));
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "unit")));
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "zzz")));
    }

    [Fact]
    public void EnumValue_RestrictedToDeclaredSet_BareNotQuoted()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var c = Constraint(tok, out var vocab)!;

        Feed(c, tok, "<tool_call>\n<function=get_weather>\n<parameter=unit>\n");
        Assert.True(c.IsConstraining);
        // Coder enum values are bare text — a quote is not part of the value.
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "\"")));
        // 'c' (celsius) / 'f' (fahrenheit) are reachable; 'x' is not.
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "c")));
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "f")));
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "x")));
    }

    [Fact]
    public void NumberValue_RejectsNonDigit()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var c = Constraint(tok, out var vocab)!;

        Feed(c, tok, "<tool_call>\n<function=get_weather>\n<parameter=days>\n");
        Assert.True(Allowed(c, vocab.VocabSize, First(tok, "3")));
        Assert.False(Allowed(c, vocab.VocabSize, First(tok, "x")));   // a non-numeric value is illegal
    }

    [Fact]
    public void JsonOutput_DoesNotEngage()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var c = Constraint(tok, out _)!;

        // If the model emitted a JSON envelope instead of the XML <function=…> shape, the Coder
        // constraint must stay inert (it arms on <tool_call> but only engages on <function=NAME>).
        Feed(c, tok, "<tool_call>\n{\"name\": \"get_weather\"}");
        Assert.False(c.IsConstraining);
    }
}
