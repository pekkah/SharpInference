using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Tekken (Mistral-Nemo family, <c>tokenizer.ggml.pre = "tekken"</c>) tokenization.
/// The split-pattern theory is the specification — it needs no model file and pins where Tekken
/// diverges from GPT-2. The model-backed facts then prove the full loop against a real 131k vocab,
/// and skip silently when the GGUF is absent (same convention as <see cref="GgufTokenizerTests"/>).
/// </summary>
public sealed class TekkenTokenizerTests
{
    // Resolved once — the scan opens GGUF headers, and every fact below asks for the tokenizer.
    private static readonly Lazy<GgufTokenizer?> s_tokenizer = new(FindTekkenTokenizer);

    private static GgufTokenizer? CreateTokenizer() => s_tokenizer.Value;

    /// <summary>
    /// Finds any locally available GGUF declaring <c>tokenizer.ggml.pre = "tekken"</c>, rather than
    /// naming one file — the behaviour under test belongs to the tokenizer family, so any member of
    /// it is a valid fixture and the suite doesn't rot when a particular checkpoint moves.
    /// Returns null (tests skip) when none is present.
    /// </summary>
    private static GgufTokenizer? FindTekkenTokenizer()
    {
        foreach (var dir in CandidateModelDirs())
        {
            foreach (var path in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                try
                {
                    using var model = GgufModel.Open(path);
                    if (model.Metadata.TryGetValue("tokenizer.ggml.pre", out var pre)
                        && pre as string == "tekken")
                        return GgufTokenizer.FromGgufModel(model);
                }
                catch
                {
                    // Unreadable / partial download — just keep looking.
                }
            }
        }
        return null;
    }

    private static IEnumerable<string> CandidateModelDirs()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var models = Path.Combine(dir, "models");
            if (Directory.Exists(models)) yield return models;
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        // Secondary location used by the other model-gated suites in this project.
        if (Directory.Exists(@"E:\models")) yield return @"E:\models";
    }

    [Theory]
    // A word keeps its leading space — the property `\w+|[^\w\s]+` cannot express, and the
    // reason the old encode-then-split order stranded the space marker on the wrong token.
    [InlineData("Hello world", "Hello| world")]
    [InlineData("Hello  world", "Hello| | world")]
    // Accented letters are \p{Ll}, so a word never splits mid-character; the old order tore
    // the two UTF-8 bytes of ï apart and they could never re-merge.
    [InlineData("café résumé", "café| résumé")]
    // Tekken's `\p{N}` has no quantifier, unlike GPT-2's ` ?\p{N}+`.
    [InlineData("is 123", "is| |1|2|3")]
    // Punctuation absorbs the newlines that follow it; newline runs are their own token.
    [InlineData("end.\n", "end|.\n")]
    [InlineData("a\n\nb", "a|\n\n|b")]
    // `upper* lower+` is greedy on the upper run, so an acronym keeps the next word's capital.
    [InlineData("XMLHttpRequest", "XMLHttp|Request")]
    public void PreTokenizer_SplitsLikeTekken(string text, string expectedPipeJoined)
    {
        var actual = GgufTokenizer.TekkenPreTokenizer().Matches(text).Select(m => m.Value);
        Assert.Equal(expectedPipeJoined, string.Join("|", actual));
    }

    [Fact]
    public void PreTokenizer_CoversEveryCharacter_LeavingNoGaps()
    {
        // A gap would mean a merge could not span it, so assert the pattern is total.
        const string text = "Hi\tthere\r\n  x=1; café — 日本語 🎉 end/";
        int pos = 0;
        foreach (System.Text.RegularExpressions.Match m in GgufTokenizer.TekkenPreTokenizer().Matches(text))
        {
            Assert.Equal(pos, m.Index);
            pos += m.Length;
        }
        Assert.Equal(text.Length, pos);
    }

    [Theory]
    [InlineData("The answer is 12345.")]
    [InlineData("def f(a,b):\n\treturn a+b\n\n\n")]
    [InlineData("Hello  world   !!!\n\n")]
    // Katakana キ is E3 82 AD — the byte GPT-2 lifts to U+0143 rather than leaving at U+00AD.
    // Treating it as identity left it with no vocab entry and produced an unknown token.
    [InlineData("café 日本語のテキスト 🎉")]
    public void EncodeDecode_RoundTripsExactly(string text)
    {
        var t = CreateTokenizer();
        if (t is null) return;

        var ids = t.Encode(text);
        Assert.All(ids, id => Assert.InRange(id, 0, t.VocabSize - 1));
        Assert.Equal(text, t.Decode(ids));
    }

    [Fact]
    public void Encode_ReachesWholeWordTokens_ForSpacedAndAccentedWords()
    {
        var t = CreateTokenizer();
        if (t is null) return;

        // This vocab holds " naïve" as one entry; reaching it requires splitting before
        // byte-encoding. And "Hello world" must reuse the very id that " world" alone yields —
        // proof the space attached to the word rather than to the preceding token.
        Assert.Single(t.Encode(" naïve"));

        var word = t.Encode(" world");
        var sentence = t.Encode("Hello world");
        Assert.Single(word);
        Assert.Equal(2, sentence.Count);
        Assert.Equal(word[0], sentence[1]);
    }

    [Fact]
    public void Encode_InstructMarkers_BecomeSingleControlTokens()
    {
        var t = CreateTokenizer();
        if (t is null) return;

        Assert.Equal(131072, t.VocabSize);
        var ids = t.Encode("[INST]Hi[/INST]");
        Assert.Equal(t.SpecialTokens["[INST]"], ids[0]);
        Assert.Equal(t.SpecialTokens["[/INST]"], ids[^1]);
        Assert.Equal("[INST]Hi[/INST]", t.Decode(ids));
    }

    [Fact]
    public void DecodeBytes_StreamedPerToken_ReassemblesMultiByteText()
    {
        var t = CreateTokenizer();
        if (t is null) return;

        // Streaming goes byte-at-a-time through Utf8StreamDecoder; a token boundary inside a
        // multi-byte character must not yield U+FFFD.
        const string text = "café 日本語 🎉";
        var decoder = new Utf8StreamDecoder();
        var sb = new System.Text.StringBuilder();
        foreach (int id in t.Encode(text))
            sb.Append(decoder.Append(t.DecodeBytes(id)));
        sb.Append(decoder.Flush());

        Assert.Equal(text, sb.ToString());
    }
}
