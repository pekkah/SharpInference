using System.Text.Json;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Model-independent tests for <see cref="JsonSchemaOutputConstraint"/> (issue #423 follow-up) using
/// <see cref="FakeJsonTokenizer"/>, mirroring <see cref="JsonToolGrammarMockTests"/>'s pattern. Unlike
/// the tool-argument constraint, there's no envelope/marker to arm on -- the constraint constrains
/// the response from the very first token.
/// </summary>
public sealed class JsonSchemaOutputConstraintTests
{
    private static ToolSchemaObject Schema(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return ToolSchema.FromOpenAiFunction("_", doc.RootElement.Clone()).Arguments;
    }

    private static (ITokenConstraint c, FakeJsonTokenizer tok, int vocab) Build(string schemaJson)
    {
        var tok = new FakeJsonTokenizer();
        var vocab = new GrammarVocabulary(tok);
        var c = new JsonSchemaOutputConstraint(vocab, Schema(schemaJson));
        return (c, tok, vocab.VocabSize);
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

    [Fact]
    public void NonObjectSchema_Throws()
    {
        var vocab = new GrammarVocabulary(new FakeJsonTokenizer());
        var ex = Assert.Throws<ArgumentException>(
            () => new JsonSchemaOutputConstraint(vocab, Schema("""{"type":"string"}""")));
        Assert.Contains("object schema", ex.Message);
    }

    [Fact]
    public void EmptyObjectSchema_Throws()
    {
        var vocab = new GrammarVocabulary(new FakeJsonTokenizer());
        Assert.Throws<ArgumentException>(
            () => new JsonSchemaOutputConstraint(vocab, Schema("""{"type":"object"}""")));
    }

    [Fact]
    public void ConstrainsFromTheFirstToken_NoAcceptNeeded()
    {
        var (c, _, _) = Build(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");
        Assert.True(c.IsConstraining);
    }

    [Fact]
    public void RequiredKey_CannotClose_UntilEmitted()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");

        Feed(c, tok, "{");
        Assert.False(Allowed(c, vocab, tok.Char('}')));   // required key missing
        Assert.True(Allowed(c, vocab, tok.Char('"')));    // opens the "answer" key
    }

    [Fact]
    public void OnlyDeclaredKeys_AreReachable()
    {
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");

        Feed(c, tok, "{\"");
        Assert.True(Allowed(c, vocab, tok.Char('a')));    // 'a' starts "answer"
        Assert.False(Allowed(c, vocab, tok.Char('x')));   // no key starts with 'x'
    }

    [Fact]
    public void FullObject_Closes_ThenForcesEndOfGeneration()
    {
        // "Constrains the ENTIRE response" means nothing may follow the object -- once it closes,
        // only an end-of-generation token is legal (forcing the model to stop), not free text.
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");

        Feed(c, tok, "{\"answer\":\"hi\"}");
        Assert.True(c.IsConstraining);                              // still constraining -- EOG-only now
        Assert.True(Allowed(c, vocab, FakeJsonTokenizer.Eos));       // EOG is the one legal token
        Assert.False(Allowed(c, vocab, tok.Char('x')));              // no further content is legal
        Assert.False(Allowed(c, vocab, tok.Char('{')));              // not even a second object
    }

    [Fact]
    public void Reset_ReEngagesImmediately_RegressionForMultiTurnReuse()
    {
        // Regression test: Reset() must re-engage the whole-body root immediately (not return to a
        // watching-idle state that never arms, since there is no envelope marker to watch for).
        var (c, tok, vocab) = Build(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");

        Feed(c, tok, "{\"answer\":\"hi\"}");
        Assert.True(c.IsConstraining);       // done with turn 1 -- now forcing EOG-only
        Assert.False(Allowed(c, vocab, tok.Char('{')));

        c.Reset();
        Assert.True(c.IsConstraining);        // turn 2 must be constrained from token 1 too
        Assert.True(Allowed(c, vocab, tok.Char('{')));   // back to expecting a fresh object, not EOG-only
    }
}
