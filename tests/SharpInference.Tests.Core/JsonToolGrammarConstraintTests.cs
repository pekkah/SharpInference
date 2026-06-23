using System.Text.Json;
using SharpInference.Core;
using SharpInference.Core.Grammar;
using Xunit.Abstractions;

namespace SharpInference.Tests.Core;

/// <summary>
/// Decode-time conformance for the JSON tool-argument grammar constraint (issue #376) against a REAL
/// Qwen vocabulary, so the byte-level matching is exercised over the actual BPE merges (where a
/// single token carries <c>{"</c>, <c>":</c>, <c>"}}</c>, or the whole <c>{}</c>). Model-gated —
/// skips when the GGUF is absent. The grammar logic itself is covered model-free by
/// <see cref="JsonToolGrammarMockTests"/>; this asserts the same invariants survive a production
/// tokenizer.
/// </summary>
public sealed class JsonToolGrammarConstraintTests(ITestOutputHelper output)
{
    private static readonly string[] s_modelPaths =
    [
        @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
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

    // The single token that emits exactly `s` (asserts it's one token so the test reasons about it).
    private static int Single(GgufTokenizer tok, string s)
    {
        var ids = tok.Encode(s);
        Assert.Single(ids);
        return ids[0];
    }

    private const string Weather =
        """{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location"]}""";

    [Fact]
    public void EngagesAtArgsColon_RejectsMergedEmptyObject()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        var c = new QwenToolCallAdapter("qwen3").BuildArgumentConstraint([Schema("get_weather", Weather)], vocab);
        Assert.NotNull(c);

        Feed(c!, tok, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ");
        Assert.True(c!.IsConstraining);   // engaged at the args-key colon

        // The merged "{}" empty-object token must be rejected (it would drop the required location);
        // the merged '{"' that opens the object and a key is allowed.
        Assert.False(Allowed(c, vocab.VocabSize, Single(tok, "{}")));
        Assert.True(Allowed(c, vocab.VocabSize, Single(tok, "{\"")));
    }

    [Fact]
    public void RequiredKey_CannotBeOmitted_ForeignKeyRejected()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        var c = new QwenToolCallAdapter("qwen3").BuildArgumentConstraint([Schema("get_weather", Weather)], vocab)!;

        Feed(c, tok, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"");
        // Now matching a key. 'location'/'unit' are declared; a foreign key letter must be unreachable.
        var loc = tok.Encode("location");
        var unit = tok.Encode("unit");
        Assert.True(Allowed(c, vocab.VocabSize, loc[0]));   // 'location…' reachable
        Assert.True(Allowed(c, vocab.VocabSize, unit[0]));  // 'unit…' reachable
        // A key from a different tool's schema ('city') is not a declared key here.
        Assert.False(Allowed(c, vocab.VocabSize, tok.Encode("zzz")[0]));
    }

    [Fact]
    public void EnumValue_RestrictedToDeclaredSet()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        var c = new QwenToolCallAdapter("qwen3").BuildArgumentConstraint([Schema("get_weather", Weather)], vocab)!;

        // Reach the 'unit' enum value's opening quote, then assert only enum prefixes are allowed.
        Feed(c, tok, "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": ");
        Assert.True(c.IsConstraining);
        // A bare (unquoted) value is illegal for a string enum — the value must open with a quote.
        Assert.False(Allowed(c, vocab.VocabSize, tok.Encode("5")[0]));
    }

    [Fact]
    public void NonJsonOutput_DoesNotEngage()
    {
        var tok = Tok();
        if (tok is null) { output.WriteLine("missing model — skip"); return; }
        var vocab = new GrammarVocabulary(tok);
        var c = new QwenToolCallAdapter("qwen3").BuildArgumentConstraint([Schema("get_weather", Weather)], vocab)!;

        // Qwen3.6 also emits an XML <function=…> shape — the JSON constraint must stay inert on it.
        Feed(c, tok, "<tool_call>\n<function=get_weather>");
        Assert.False(c.IsConstraining);
    }
}
