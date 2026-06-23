using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// A tiny deterministic tokenizer that mimics the relevant facets of Gemma 4's vocabulary for
/// model-independent grammar tests (issue #374): the tool-call markers and the <c>&lt;|"|&gt;</c>
/// quote are single special tokens; every other byte is its own single-char token. This lets the
/// decode-time conformance tests run in CI without the multi-gigabyte GGUF, while still exercising
/// the multi-token key/enum matching (each key letter arrives as a separate token).
/// </summary>
public sealed class FakeGemmaTokenizer : ITokenizer
{
    public const int Pad = 0, Bos = 1, Eos = 2, OpenMarker = 3, CloseMarker = 4, Quote = 5;
    private const int FirstChar = 6;   // single-char tokens occupy [FirstChar, FirstChar+256)

    private readonly Dictionary<string, int> _specials = new(StringComparer.Ordinal)
    {
        ["<|tool_call>"] = OpenMarker,
        ["<tool_call|>"] = CloseMarker,
        ["<|\"|>"] = Quote,
    };

    public int VocabSize => FirstChar + 256;
    public int BosTokenId => Bos;
    public int EosTokenId => Eos;
    public int UnknownTokenId => Pad;
    public int PadTokenId => Pad;
    public bool AddBosToken => false;
    public ImmutableArray<int> EogTokenIds => [Eos];
    public IReadOnlyDictionary<string, int> SpecialTokens => _specials;

    /// <summary>Token id for a single ASCII byte.</summary>
    public static int Char(char c) => FirstChar + (byte)c;

    public byte[] DecodeBytes(int token)
    {
        if (token == OpenMarker) return Encoding.UTF8.GetBytes("<|tool_call>");
        if (token == CloseMarker) return Encoding.UTF8.GetBytes("<tool_call|>");
        if (token == Quote) return Encoding.UTF8.GetBytes("<|\"|>");
        if (token is Bos or Eos or Pad) return [];
        if (token >= FirstChar && token < FirstChar + 256) return [(byte)(token - FirstChar)];
        return [];
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        int i = 0;
        while (i < text.Length)
        {
            int matched = -1, matchLen = 0;
            foreach (var (s, id) in _specials)
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
