using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace SharpInference.Core;

/// <summary>
/// Tokenizer that loads BPE vocab and merges from GGUF metadata
/// and delegates to Microsoft.ML.Tokenizers.CodeGenTokenizer (GPT-2 style byte-level BPE).
/// </summary>
public sealed class GgufTokenizer : ITokenizer
{
    private readonly Tokenizer _inner;
    private readonly Dictionary<string, int> _specialTokens;

    public int VocabSize { get; }
    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnknownTokenId { get; }
    public int PadTokenId { get; }
    public bool AddBosToken { get; }

    private GgufTokenizer(
        Tokenizer inner,
        Dictionary<string, int> specialTokens,
        int vocabSize,
        int bosTokenId,
        int eosTokenId,
        int unknownTokenId,
        int padTokenId,
        bool addBosToken)
    {
        _inner = inner;
        _specialTokens = specialTokens;
        VocabSize = vocabSize;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        UnknownTokenId = unknownTokenId;
        PadTokenId = padTokenId;
        AddBosToken = addBosToken;
    }

    /// <summary>
    /// Creates a tokenizer from GGUF model metadata.
    /// Expects tokenizer.ggml.tokens, tokenizer.ggml.merges, and special token IDs.
    /// </summary>
    public static GgufTokenizer FromGgufModel(GgufModel model)
    {
        // Extract vocab tokens (array of strings indexed by token ID)
        var tokensArray = model.Metadata.TryGetValue("tokenizer.ggml.tokens", out var tokensObj)
            ? (object[])tokensObj
            : throw new InvalidDataException("GGUF metadata missing 'tokenizer.ggml.tokens'");

        // Extract merge rules
        var mergesArray = model.Metadata.TryGetValue("tokenizer.ggml.merges", out var mergesObj)
            ? (object[])mergesObj
            : [];

        // Extract special token IDs
        var bosTokenId = GetMetadataInt(model, "tokenizer.ggml.bos_token_id", 1);
        var eosTokenId = GetMetadataInt(model, "tokenizer.ggml.eos_token_id", 2);
        var unknownTokenId = GetMetadataInt(model, "tokenizer.ggml.unknown_token_id", 0);
        var padTokenId = GetMetadataInt(model, "tokenizer.ggml.padding_token_id", eosTokenId);
        var addBosToken = GetMetadataBool(model, "tokenizer.ggml.add_bos_token", false);

        // Identify special tokens (control tokens, type 3)
        var specialTokens = new Dictionary<string, int>();
        if (model.Metadata.TryGetValue("tokenizer.ggml.token_type", out var tokenTypeObj))
        {
            var tokenTypes = (object[])tokenTypeObj;
            for (int i = 0; i < tokenTypes.Length && i < tokensArray.Length; i++)
            {
                if (Convert.ToInt32(tokenTypes[i]) == 3)
                    specialTokens[(string)tokensArray[i]] = i;
            }
        }

        // Build vocab and merges as byte arrays (tokenizer constructors may dispose streams)
        byte[] vocabBytes;
        {
            using var vocabStream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(vocabStream))
            {
                writer.WriteStartObject();
                for (int i = 0; i < tokensArray.Length; i++)
                    writer.WriteNumber((string)tokensArray[i], i);
                writer.WriteEndObject();
            }
            vocabBytes = vocabStream.ToArray();
        }

        byte[] mergesBytes;
        {
            using var mergesStream = new MemoryStream();
            using (var sw = new StreamWriter(mergesStream, Encoding.UTF8, leaveOpen: true))
            {
                for (int i = 0; i < mergesArray.Length; i++)
                    sw.WriteLine((string)mergesArray[i]);
            }
            mergesBytes = mergesStream.ToArray();
        }

        // Get token strings for special tokens.
        // If the unknown token is a control/special token (type 3), don't pass it to
        // CodeGenTokenizer as it won't be in the BPE vocab and will throw.
        string? unknownToken = null;
        if (unknownTokenId >= 0 && unknownTokenId < tokensArray.Length)
        {
            bool isControl = model.Metadata.TryGetValue("tokenizer.ggml.token_type", out var ttObj)
                && Convert.ToInt32(((object[])ttObj)[unknownTokenId]) == 3;
            if (!isControl)
                unknownToken = (string)tokensArray[unknownTokenId];
        }
        string? bosToken = bosTokenId >= 0 && bosTokenId < tokensArray.Length
            ? (string)tokensArray[bosTokenId]
            : null;
        string? eosToken = eosTokenId >= 0 && eosTokenId < tokensArray.Length
            ? (string)tokensArray[eosTokenId]
            : null;

        IReadOnlyDictionary<string, int>? specialTokensDict =
            specialTokens.Count > 0 ? specialTokens : null;

        // Try CodeGenTokenizer first (better decode quality for GPT-2 style models).
        // Fall back to BpeTokenizer if CodeGenTokenizer fails (e.g., Llama 3.1 where
        // the default unknown token <|endoftext|> is not in the BPE vocab).
        Tokenizer inner;
        try
        {
            using var vs1 = new MemoryStream(vocabBytes);
            using var ms1 = new MemoryStream(mergesBytes);
            inner = CodeGenTokenizer.Create(vs1, ms1,
                addPrefixSpace: false,
                addBeginOfSentence: false,
                addEndOfSentence: false);
        }
        catch
        {
            using var vs2 = new MemoryStream(vocabBytes);
            using var ms2 = new MemoryStream(mergesBytes);
            inner = BpeTokenizer.Create(vs2, ms2,
                specialTokens: specialTokensDict,
                unknownToken: unknownToken);
        }

        return new GgufTokenizer(
            inner,
            specialTokens,
            tokensArray.Length,
            bosTokenId,
            eosTokenId,
            unknownTokenId,
            padTokenId,
            addBosToken);
    }

    public IReadOnlyList<int> Encode(string text)
    {
        if (_specialTokens.Count == 0)
            return _inner.EncodeToIds(text);

        // Split text on special token boundaries and encode each segment,
        // inserting special token IDs directly.
        var result = new List<int>();
        int pos = 0;
        while (pos < text.Length)
        {
            // Find the earliest special token match from current position
            int bestStart = text.Length;
            string? bestToken = null;
            foreach (var st in _specialTokens.Keys)
            {
                int idx = text.IndexOf(st, pos, StringComparison.Ordinal);
                if (idx >= 0 && idx < bestStart)
                {
                    bestStart = idx;
                    bestToken = st;
                }
            }

            if (bestToken is null)
            {
                // No more special tokens — encode the rest
                if (pos < text.Length)
                    result.AddRange(_inner.EncodeToIds(text[pos..]));
                break;
            }

            // Encode text before the special token
            if (bestStart > pos)
                result.AddRange(_inner.EncodeToIds(text[pos..bestStart]));

            // Insert the special token ID
            result.Add(_specialTokens[bestToken]);
            pos = bestStart + bestToken.Length;
        }
        return result;
    }

    public string Decode(IEnumerable<int> tokens)
    {
        var text = _inner.Decode(tokens) ?? string.Empty;

        // BpeTokenizer may output GPT-2 byte-level BPE artifacts:
        // Ġ (U+0120) = space, Ċ (U+010A) = newline, etc.
        // Convert them back to actual bytes if present.
        if (text.Contains('\u0120') || text.Contains('\u010A'))
            text = DecodeGpt2Bytes(text);

        return text;
    }

    /// <summary>
    /// Convert GPT-2 byte-level BPE Unicode characters back to actual bytes.
    /// GPT-2 maps bytes 0x00-0xFF to Unicode chars starting at various offsets.
    /// The most common: Ġ (U+0120) = space (0x20), Ċ (U+010A) = newline (0x0A).
    /// </summary>
    private static string DecodeGpt2Bytes(string text)
    {
        var bytes = new List<byte>(text.Length);
        foreach (char c in text)
        {
            if (c < 256)
            {
                bytes.Add((byte)c);
            }
            else
            {
                // GPT-2 byte mapping: chars 0x100-0x1FF map to bytes via lookup
                // The mapping: printable ASCII stays as-is, non-printable gets offset
                // Simplified: U+0100+n maps to byte n for n < 256
                int mapped = c - 0x100;
                if (mapped >= 0 && mapped < 256)
                    bytes.Add((byte)mapped);
                else
                {
                    // Not a byte token — encode as UTF-8
                    foreach (byte b in System.Text.Encoding.UTF8.GetBytes(new[] { c }))
                        bytes.Add(b);
                }
            }
        }
        return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
    }

    private static int GetMetadataInt(GgufModel model, string key, int defaultValue)
    {
        if (!model.Metadata.TryGetValue(key, out var value))
            return defaultValue;
        return Convert.ToInt32(value);
    }

    private static bool GetMetadataBool(GgufModel model, string key, bool defaultValue)
    {
        if (!model.Metadata.TryGetValue(key, out var value))
            return defaultValue;
        return Convert.ToBoolean(value);
    }
}
