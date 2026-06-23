using System.Collections.Immutable;
using System.Text;
using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// A tiny deterministic tokenizer that mimics the facets of a real Qwen/Llama BPE vocabulary the
/// JSON tool-argument grammar depends on (issue #376): the call markers are single special tokens,
/// every byte is also available as a single-char token, AND a handful of <b>merged</b> structural
/// tokens (<c>{"</c>, <c>":</c>, <c>"}</c>, <c>{}</c>, …) mirror how a real BPE fuses JSON
/// punctuation with adjacent quotes/whitespace. The merged tokens are what make byte-level matching
/// (rather than token-level) mandatory — a single token can open a string and start its content, or
/// be the whole empty object <c>{}</c>. <see cref="EncodeMerged"/> lets a test request the
/// realistic merged tokenization explicitly.
/// </summary>
public sealed class FakeJsonTokenizer : ITokenizer
{
    public const int Pad = 0, Bos = 1, Eos = 2;

    private readonly Dictionary<string, int> _specials = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _merged = new(StringComparer.Ordinal);
    private readonly int _singleBase;

    // Realistic BPE-style merges of JSON punctuation with adjacent quotes / whitespace / braces.
    private static readonly string[] MergedPieces =
    [
        "{\"", "\":", "\",", "\"}", "\"]", "{}", "[]", "\":\"", ",\"", " {\"", "\"}}", "\": ", "\", ",
    ];

    public FakeJsonTokenizer()
    {
        int id = 3;
        foreach (var s in new[] { "<tool_call>", "</tool_call>", "<|python_tag|>", "<|tool_sep|>", "<|tool_call_begin|>", "<|tool_call_end|>" })
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

    /// <summary>Token id for a registered merged structural piece (e.g. <c>{"</c> or <c>{}</c>).</summary>
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

    /// <summary>Encodes <paramref name="text"/> but forces the named merged pieces to be used (the
    /// default greedy Encode already prefers them; this is an explicit alias for test readability).</summary>
    public IReadOnlyList<int> EncodeMerged(string text) => Encode(text);

    public string Decode(IEnumerable<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (int t in tokens) sb.Append(Encoding.UTF8.GetString(DecodeBytes(t)));
        return sb.ToString();
    }
}
