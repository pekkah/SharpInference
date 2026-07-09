using System.Collections.Immutable;
using SharpInference.Cli;
using SharpInference.Core;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="RunCommand.TryLoadJsonSchemaConstraint"/> -- the llama.cpp-style
/// <c>-j/--json-schema</c> and <c>--json-schema-file/--jf</c> flags (issue #423 follow-up). Mirrors
/// <see cref="CpuMoeFlagsTests"/>'s direct-call pattern. Deep grammar correctness is covered by
/// <c>SharpInference.Tests.Core.JsonSchemaOutputConstraintTests</c>; these only test the CLI-level
/// validation/wiring (mutual exclusivity, file loading, JSON parsing, schema compilation errors).
/// </summary>
public sealed class JsonSchemaFlagsTests
{
    /// <summary>Bare-minimum tokenizer -- these tests only exercise construction, never Accept/Filter.</summary>
    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => 4;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int UnknownTokenId => 0;
        public int PadTokenId => 0;
        public bool AddBosToken => false;
        public ImmutableArray<int> EogTokenIds => [0];
        public IReadOnlyDictionary<string, int> SpecialTokens { get; } = new Dictionary<string, int>();
        public byte[] DecodeBytes(int token) => [];
        public IReadOnlyList<int> Encode(string text) => [];
        public string Decode(IEnumerable<int> tokens) => "";
    }

    private static readonly GrammarVocabulary Vocab = new(new StubTokenizer());

    [Fact]
    public void NeitherFlag_IsNoOp()
    {
        bool ok = RunCommand.TryLoadJsonSchemaConstraint(null, null, Vocab, out var constraint, out var error);

        Assert.True(ok);
        Assert.Null(constraint);
        Assert.Null(error);
    }

    [Fact]
    public void BothFlags_IsError()
    {
        bool ok = RunCommand.TryLoadJsonSchemaConstraint("{}", "somefile.json", Vocab, out var constraint, out var error);

        Assert.False(ok);
        Assert.Null(constraint);
        Assert.Contains("mutually exclusive", error);
    }

    [Fact]
    public void FileNotFound_IsError()
    {
        bool ok = RunCommand.TryLoadJsonSchemaConstraint(
            null, "definitely-does-not-exist.json", Vocab, out var constraint, out var error);

        Assert.False(ok);
        Assert.Null(constraint);
        Assert.Contains("not found", error);
    }

    [Fact]
    public void InlineSchema_MalformedJson_IsError()
    {
        bool ok = RunCommand.TryLoadJsonSchemaConstraint("{not valid json", null, Vocab, out var constraint, out var error);

        Assert.False(ok);
        Assert.Null(constraint);
        Assert.Contains("could not parse", error);
    }

    [Fact]
    public void InlineSchema_UncompilableSchema_IsError()
    {
        // Not an object schema (bare string) -- compiles to null, reported as an error rather than
        // silently generating unconstrained output.
        bool ok = RunCommand.TryLoadJsonSchemaConstraint(
            """{"type":"string"}""", null, Vocab, out var constraint, out var error);

        Assert.False(ok);
        Assert.Null(constraint);
        Assert.Contains("could not be compiled", error);
    }

    [Fact]
    public void InlineSchema_ValidObjectSchema_Succeeds()
    {
        bool ok = RunCommand.TryLoadJsonSchemaConstraint(
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""",
            null, Vocab, out var constraint, out var error);

        Assert.True(ok);
        Assert.NotNull(constraint);
        Assert.Null(error);
    }

    [Fact]
    public void OrderedFlag_IsThreadedIntoConstraint()
    {
        // --json-schema-ordered (issue #425): declaration-order property emission, default off.
        const string schema =
            """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""";

        bool ok = RunCommand.TryLoadJsonSchemaConstraint(
            schema, null, Vocab, out var constraint, out var error, ordered: true);
        Assert.True(ok);
        Assert.Null(error);
        Assert.True(Assert.IsType<JsonSchemaOutputConstraint>(constraint).OrderedProperties);

        ok = RunCommand.TryLoadJsonSchemaConstraint(schema, null, Vocab, out constraint, out error);
        Assert.True(ok);
        Assert.False(Assert.IsType<JsonSchemaOutputConstraint>(constraint).OrderedProperties);
    }

    [Fact]
    public void SchemaFile_ValidObjectSchema_Succeeds()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path,
                """{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}""");
            bool ok = RunCommand.TryLoadJsonSchemaConstraint(null, path, Vocab, out var constraint, out var error);

            Assert.True(ok);
            Assert.NotNull(constraint);
            Assert.Null(error);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
