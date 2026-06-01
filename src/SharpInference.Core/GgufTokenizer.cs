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
    private readonly Dictionary<int, string> _specialTokensById;
    private readonly Dictionary<string, int> _vocab;
    // Vocab strings indexed by token ID — for byte-level BPE these are the raw
    // GPT-2 byte-level encoded forms, used by DecodeBytes to recover exact bytes
    // without the inner tokenizer's lossy U+FFFD substitution on partial sequences.
    private readonly string[] _idToToken;
    private readonly bool _needsByteEncoding;
    // SentencePiece-style BPE (Gemma/Llama): the vocab encodes spaces as U+2581 (▁).
    // Pre-encode by replacing ' ' with ▁ on input and ▁ back to ' ' on output.
    private readonly bool _isSpmBpe;

    public int VocabSize { get; }
    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnknownTokenId { get; }
    public int PadTokenId { get; }
    public bool AddBosToken { get; }

    /// <summary>All special (control) tokens keyed by their string representation.</summary>
    public IReadOnlyDictionary<string, int> SpecialTokens => _specialTokens;

    /// <summary>The type name of the inner tokenizer (for diagnostics).</summary>
    public string InnerTokenizerType => _inner.GetType().Name;

    /// <summary>
    /// Jinja2 chat template parsed from GGUF tokenizer.chat_template metadata, if present.
    /// Use this to format messages into a prompt string for any model without hardcoding templates.
    /// </summary>
    public JinjaChatTemplate? ChatTemplate { get; private set; }

    private GgufTokenizer(
        Tokenizer inner,
        Dictionary<string, int> specialTokens,
        Dictionary<int, string> specialTokensById,
        Dictionary<string, int> vocab,
        string[] idToToken,
        int vocabSize,
        int bosTokenId,
        int eosTokenId,
        int unknownTokenId,
        int padTokenId,
        bool addBosToken,
        bool needsByteEncoding,
        bool isSpmBpe)
    {
        _inner = inner;
        _specialTokens = specialTokens;
        _specialTokensById = specialTokensById;
        _vocab = vocab;
        _idToToken = idToToken;
        _needsByteEncoding = needsByteEncoding;
        _isSpmBpe = isSpmBpe;
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

        // Identify special tokens (control tokens type 3, and user-defined type 4 like <think>)
        var specialTokens = new Dictionary<string, int>();
        if (model.Metadata.TryGetValue("tokenizer.ggml.token_type", out var tokenTypeObj))
        {
            var tokenTypes = (object[])tokenTypeObj;
            for (int i = 0; i < tokenTypes.Length && i < tokensArray.Length; i++)
            {
                int ttype = Convert.ToInt32(tokenTypes[i]);
                if (ttype == 3 || ttype == 4)
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

        // SPM-family tokenizers (Gemma, Llama SentencePiece) encode spaces as U+2581 (▁)
        // and may have merges containing literal newlines/whitespace. The stream-based
        // BpeTokenizer.Create reads merges via StreamReader, so embedded newlines split
        // a merge across multiple lines and the parser throws ("Invalid merger file
        // format"). Use the in-memory BpeOptions API for these models — no file parsing.
        var tokenizerModel = model.Metadata.TryGetValue("tokenizer.ggml.model", out var tmObj)
            ? (string)tmObj
            : "";
        bool isSpmModel = tokenizerModel is "gemma" or "gemma2" or "gemma3" or "gemma4" or "llama";

        Tokenizer? inner = null;
        bool needsByteEncoding = false;
        bool isSpmBpe = false;

        // Try CodeGenTokenizer first (better decode quality for GPT-2 style models).
        // CodeGenTokenizer handles GPT-2 byte-level BPE encoding internally.
        // Skip for SPM models — their merges contain whitespace that breaks the parser.
        if (!isSpmModel)
        {
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
                inner = null;
            }

            if (inner is null)
            {
                try
                {
                    using var vs2 = new MemoryStream(vocabBytes);
                    using var ms2 = new MemoryStream(mergesBytes);
                    inner = BpeTokenizer.Create(vs2, ms2,
                        specialTokens: specialTokensDict,
                        unknownToken: unknownToken);
                    needsByteEncoding = true;
                }
                catch
                {
                    inner = null;
                }
            }
        }

        if (inner is null)
        {
            var vocabDict = new Dictionary<string, int>(tokensArray.Length, StringComparer.Ordinal);
            for (int i = 0; i < tokensArray.Length; i++)
                vocabDict.TryAdd((string)tokensArray[i], i);

            var mergesList = new List<string>(mergesArray.Length);
            for (int i = 0; i < mergesArray.Length; i++)
                mergesList.Add((string)mergesArray[i]);

            var bpeOptions = new BpeOptions(vocabDict)
            {
                Merges = mergesList,
                SpecialTokens = specialTokensDict,
                UnknownToken = unknownToken,
            };
            inner = BpeTokenizer.Create(bpeOptions);
            needsByteEncoding = false;
            isSpmBpe = isSpmModel;
        }

        var idToToken = new string[tokensArray.Length];
        for (int i = 0; i < tokensArray.Length; i++)
            idToToken[i] = (string)tokensArray[i];

        var tokenizer = new GgufTokenizer(
            inner,
            specialTokens,
            specialTokens.ToDictionary(kv => kv.Value, kv => kv.Key),
            BuildVocabLookup(tokensArray),
            idToToken,
            tokensArray.Length,
            bosTokenId,
            eosTokenId,
            unknownTokenId,
            padTokenId,
            addBosToken,
            needsByteEncoding,
            isSpmBpe);

        if (model.Metadata.TryGetValue("tokenizer.chat_template", out var tmpl) && tmpl is string tmplStr)
        {
            try { tokenizer.ChatTemplate = new JinjaChatTemplate(tmplStr); }
            catch { /* malformed template — ChatTemplate stays null */ }
        }

        return tokenizer;
    }

    private static Dictionary<string, int> BuildVocabLookup(object[] tokensArray)
    {
        var vocab = new Dictionary<string, int>(tokensArray.Length, StringComparer.Ordinal);
        for (int i = 0; i < tokensArray.Length; i++)
            vocab.TryAdd((string)tokensArray[i], i);
        return vocab;
    }

    public IReadOnlyList<int> Encode(string text)
    {
        if (_specialTokens.Count == 0)
            return EncodeTextSegment(text);

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
                    result.AddRange(EncodeTextSegment(text[pos..]));
                break;
            }

            // Encode text before the special token
            if (bestStart > pos)
                result.AddRange(EncodeTextSegment(text[pos..bestStart]));

            // Insert the special token ID
            result.Add(_specialTokens[bestToken]);
            pos = bestStart + bestToken.Length;
        }
        return result;
    }

    private IReadOnlyList<int> EncodeTextSegment(string text)
    {
        if (text.Length == 0) return [];

        // BpeTokenizer doesn't do GPT-2 byte-level encoding internally —
        // we must convert raw bytes to GPT-2 Unicode before BPE lookup.
        // CodeGenTokenizer handles this automatically.
        if (_needsByteEncoding)
            text = EncodeToGpt2Bytes(text);
        else if (_isSpmBpe)
            text = text.Replace(' ', '▁');

        var ids = _inner.EncodeToIds(text);

        if (ids.Count > 0) return ids;

        var result = new List<int>(text.Length);
        foreach (char c in text)
        {
            // Text is already in GPT-2 / SPM encoding if a transform was applied above.
            char bpe = _needsByteEncoding ? c : EncodeByteToGpt2(c);
            if (_vocab.TryGetValue(bpe.ToString(), out int id))
                result.Add(id);
        }
        return result;
    }

    /// <summary>
    /// Converts a UTF-8 string to GPT-2 byte-level BPE Unicode representation.
    /// Each byte in the UTF-8 encoding is mapped to its GPT-2 Unicode codepoint.
    /// </summary>
    private static string EncodeToGpt2Bytes(string text)
    {
        byte[] utf8 = Encoding.UTF8.GetBytes(text);
        var sb = new StringBuilder(utf8.Length);
        foreach (byte b in utf8)
            sb.Append(EncodeByteToGpt2((char)b));
        return sb.ToString();
    }

    /// <summary>
    /// Maps a single character to its GPT-2 byte-level BPE Unicode representation.
    /// Printable ASCII (0x21–0x7E) and extended printable (0xA1–0xFF) are unchanged.
    /// Control and non-printable bytes map to U+0100–U+0142.
    /// </summary>
    private static char EncodeByteToGpt2(char c)
    {
        if (c is >= '!' and <= '~') return c;   // printable ASCII: unchanged
        if (c >= '\u00A1') return c;             // extended printable: unchanged
        if (c <= '\u0020') return (char)(c + 0x100); // 0x00–0x20 → U+0100–U+0120
        return (char)(c - 0x7F + 0x121);        // 0x7F–0xA0 → U+0121–U+0142
    }

    public string Decode(IEnumerable<int> tokens)
    {
        // For single-token decode of a special token, bypass the inner tokenizer
        // which doesn't know about special token IDs (type 3/4).
        if (tokens is IReadOnlyList<int> list && list.Count == 1 &&
            _specialTokensById.TryGetValue(list[0], out var specialStr))
            return specialStr;

        var text = _inner.Decode(tokens) ?? string.Empty;

        if (_isSpmBpe)
            return text.Replace('▁', ' ');

        // BpeTokenizer may output GPT-2 byte-level BPE artifacts:
        // Ġ (U+0120) = space, Ċ (U+010A) = newline, etc.
        // Convert them back to actual bytes if present.
        if (text.Contains('\u0120') || text.Contains('\u010A'))
            text = Encoding.UTF8.GetString(Gpt2CharsToBytes(text));

        return text;
    }

    /// <inheritdoc/>
    public byte[] DecodeBytes(int token)
    {
        // Special tokens (control / user-defined) decode to their literal UTF-8 bytes.
        if (_specialTokensById.TryGetValue(token, out var specialStr))
            return Encoding.UTF8.GetBytes(specialStr);

        // Look up the raw vocab string by ID rather than going through _inner.Decode,
        // which silently substitutes U+FFFD for incomplete UTF-8 sequences and so
        // loses the exact bytes a single token contributes to the stream.
        if ((uint)token >= (uint)_idToToken.Length)
            return [];

        if (_isSpmBpe)
            return Encoding.UTF8.GetBytes(_idToToken[token].Replace('▁', ' '));

        return Gpt2CharsToBytes(_idToToken[token]);
    }

    /// <summary>
    /// Convert GPT-2 byte-level BPE Unicode characters back to raw bytes.
    /// Bytes 0x21-0x7E and 0xA1-0xFF stay as-is; 0x00-0x20 and 0x7F-0xA0 are
    /// remapped to U+0100-U+0142 to keep them printable.
    ///
    /// Returns the raw byte sequence — caller is responsible for UTF-8 decoding,
    /// which may need to span multiple tokens to assemble multi-byte characters.
    /// </summary>
    private static byte[] Gpt2CharsToBytes(string text)
    {
        // Inverse of EncodeByteToGpt2 — must match exactly:
        //   0x21-0x7E       → identity
        //   0xA1-0xFF       → identity
        //   U+0100-U+0120   → 0x00-0x20  (subtract 0x100)
        //   U+0121-U+0142   → 0x7F-0xA0  (subtract 0xA2 = 0x121 - 0x7F)
        var bytes = new List<byte>(text.Length);
        foreach (char c in text)
        {
            if (c is >= '!' and <= '~')
                bytes.Add((byte)c);
            else if (c is >= '¡' and <= 'ÿ')
                bytes.Add((byte)c);
            else if (c is >= 'Ā' and <= 'Ġ')
                bytes.Add((byte)(c - 0x100));
            else if (c is >= 'ġ' and <= 'ł')
                bytes.Add((byte)(c - 0xA2));
            else
            {
                // Not a byte token — encode as UTF-8 (rare fallback)
                foreach (byte b in Encoding.UTF8.GetBytes(new[] { c }))
                    bytes.Add(b);
            }
        }
        return bytes.ToArray();
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
