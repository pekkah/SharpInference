using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// A tiny deterministic tokenizer that mimics the facets of a real Qwen3-Coder BPE vocabulary the XML
/// tool-argument grammar depends on (issue #383): the <c>&lt;tool_call&gt;</c>/<c>&lt;/tool_call&gt;</c>
/// envelope tokens are single specials (the constraint's arming gate), every byte is also a single-char
/// token, AND a handful of <b>merged</b> structural tokens (<c>&lt;function=</c>, <c>&lt;parameter=</c>,
/// <c>&lt;/parameter&gt;</c>, <c>&lt;/function&gt;</c>, <c>&gt;\n</c>, …) mirror how a real BPE fuses the
/// XML tags and trailing newlines into single tokens. Those merges are why byte-level matching is
/// mandatory — a single token can carry a whole tag, or close one parameter and open the next. Unlike
/// Gemma/Qwen-JSON, the Coder tags are NOT special tokens; they are ordinary text the constraint
/// byte-walks, so they live in the merged table here.
/// </summary>
public sealed class FakeCoderTokenizer : ITokenizer
{
    public const int Pad = 0, Bos = 1, Eos = 2;

    private readonly Dictionary<string, int> _specials = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _merged = new(StringComparer.Ordinal);
    private readonly int _singleBase;

    // Realistic BPE-style merges: whole XML tags plus tags fused with the template's newlines. The
    // post-tag ">\n" merge is what exercises the early-engage (the '>' that closes <function=NAME>
    // arrives merged with the following newline, so engagement happens mid-token).
    private static readonly string[] MergedPieces =
    [
        "<function=", "<parameter=", "</parameter>", "</function>",
        ">\n", "</parameter>\n<parameter=", "</parameter>\n</function>",
    ];

    public FakeCoderTokenizer()
    {
        int id = 3;
        foreach (var s in new[] { "<tool_call>", "</tool_call>" })
            _specials[s] = id++;
        foreach (var s in MergedPieces)
            _merged[s] = id++;
        _singleBase = id;
    }

    public int VocabSize => _singleBase + 256;
    public int BosTokenId => Bos;
    public int EosTokenId => Eos;
    public int UnknownTokenId => Pad;
    public int PadTokenId => Pad;
    public bool AddBosToken => false;
    public ImmutableArray<int> EogTokenIds => [Eos];
    public IReadOnlyDictionary<string, int> SpecialTokens => _specials;

    /// <summary>Token id for a single ASCII byte.</summary>
    public int Char(char c) => _singleBase + (byte)c;

    /// <summary>Token id for a registered merged structural piece (e.g. <c>&lt;parameter=</c>).</summary>
    public int Merged(string piece) => _merged[piece];

    public byte[] DecodeBytes(int token)
    {
        foreach (var (s, id) in _specials) if (id == token) return Encoding.UTF8.GetBytes(s);
        foreach (var (s, id) in _merged) if (id == token) return Encoding.UTF8.GetBytes(s);
        if (token is Bos or Eos or Pad) return [];
        if (token >= _singleBase && token < _singleBase + 256) return [(byte)(token - _singleBase)];
        return [];
    }

    /// <summary>Greedy longest-match tokenization over specials ∪ merged pieces, else single chars.</summary>
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        int i = 0;
        while (i < text.Length)
        {
            int matched = -1, matchLen = 0;
            foreach (var table in new[] { _specials, _merged })
                foreach (var (s, id) in table)
                    if (s.Length > matchLen && i + s.Length <= text.Length
                        && text.AsSpan(i, s.Length).SequenceEqual(s))
                    { matched = id; matchLen = s.Length; }

            if (matched >= 0) { ids.Add(matched); i += matchLen; }
            else { ids.Add(Char(text[i])); i++; }
        }
        return ids;
    }

    public string Decode(IEnumerable<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (int t in tokens) sb.Append(Encoding.UTF8.GetString(DecodeBytes(t)));
        return sb.ToString();
    }
}
