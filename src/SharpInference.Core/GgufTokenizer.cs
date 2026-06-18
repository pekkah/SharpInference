using System.Collections.Immutable;
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
    // For SPM mode, merges as (left, right) → priority (lower = higher priority).
    // Microsoft.ML.Tokenizers' BPE applies a built-in pre-tokenizer that splits on
    // ▁ boundaries so merges like "▁ capital → ▁capital" never fire. We do the
    // merge loop ourselves directly against this map.
    private readonly Dictionary<(string, string), int>? _spmMerges;

    public int VocabSize { get; }
    public int BosTokenId { get; }
    public int EosTokenId { get; }
    public int UnknownTokenId { get; }
    public int PadTokenId { get; }
    public bool AddBosToken { get; }

    /// <summary>
    /// All end-of-generation token IDs: the configured EOS plus any well-known EOG marker
    /// present in this vocab (e.g. Gemma 4's <c>&lt;eos&gt;</c> at id 1, which is distinct from
    /// its configured EOS <c>&lt;turn|&gt;</c> at id 106). Generation should stop on ANY of these;
    /// otherwise a model that emits an alternate end token decodes it as literal text and runs on.
    /// Always contains at least <see cref="EosTokenId"/>. See <see cref="BuildEogTokenIds"/> for
    /// the resolution rules. Immutable, so the published stop set can't be tampered with.
    /// </summary>
    public ImmutableArray<int> EogTokenIds { get; }

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
        bool isSpmBpe,
        Dictionary<(string, string), int>? spmMerges,
        ImmutableArray<int> eogTokenIds)
    {
        _inner = inner;
        _specialTokens = specialTokens;
        _specialTokensById = specialTokensById;
        _vocab = vocab;
        _idToToken = idToToken;
        _needsByteEncoding = needsByteEncoding;
        _isSpmBpe = isSpmBpe;
        _spmMerges = spmMerges;
        VocabSize = vocabSize;
        BosTokenId = bosTokenId;
        EosTokenId = eosTokenId;
        UnknownTokenId = unknownTokenId;
        PadTokenId = padTokenId;
        AddBosToken = addBosToken;
        EogTokenIds = eogTokenIds;
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

        Dictionary<(string, string), int>? spmMerges = null;
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

            // SPM mode: build a (left,right)→priority map for our manual BPE.
            // The first space-separated pair per merge line is the merge rule.
            if (isSpmBpe)
            {
                spmMerges = new Dictionary<(string, string), int>(mergesArray.Length);
                for (int i = 0; i < mergesArray.Length; i++)
                {
                    var line = (string)mergesArray[i];
                    int sp = line.IndexOf(' ');
                    if (sp <= 0 || sp >= line.Length - 1) continue;
                    var left = line[..sp];
                    var right = line[(sp + 1)..];
                    spmMerges.TryAdd((left, right), i);
                }
            }
        }

        var idToToken = new string[tokensArray.Length];
        for (int i = 0; i < tokensArray.Length; i++)
            idToToken[i] = (string)tokensArray[i];

        var vocabLookup = BuildVocabLookup(tokensArray);
        var eogIds = BuildEogTokenIds(vocabLookup, new HashSet<int>(specialTokens.Values), eosTokenId);

        var tokenizer = new GgufTokenizer(
            inner,
            specialTokens,
            specialTokens.ToDictionary(kv => kv.Value, kv => kv.Key),
            vocabLookup,
            idToToken,
            tokensArray.Length,
            bosTokenId,
            eosTokenId,
            unknownTokenId,
            padTokenId,
            addBosToken,
            needsByteEncoding,
            isSpmBpe,
            spmMerges,
            eogIds);

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

    // Canonical end-of-sequence names accepted even when typed NORMAL rather than control —
    // Gemma 4 ships <eos> (id 1) as a normal token, distinct from its end-of-turn token,
    // which these GGUFs name <turn|> (id 106) — NOT the HF-canonical <end_of_turn> (which
    // doesn't exist as a single vocab token here; it splits into pieces). E4B declares
    // eos_token_id=106 so it stops anyway, but the 12B QAT model declares eos_token_id=1,
    // so without <turn|> in this list its turn-end (106) never enters the EOG set and
    // generation runs past the turn (leaking <turn|>…<channel|> garbage). Including all
    // three names makes the turn-end a stop regardless of the (model-varying) eos metadata.
    // A normal token literally named <eos>/<end_of_turn>/<turn|> appearing in generated
    // content is not a real-world case, so treating it as a stop is safe.
    private static readonly string[] s_canonicalEosNames = ["<eos>", "<end_of_turn>", "<turn|>"];

    // Bracket-style end-of-turn markers (Llama, Mistral, Phi, ChatML, ...). These are only
    // accepted as stops when the vocab types them as control/user-defined tokens, so a model
    // that legitimately uses one of these strings as ordinary text isn't silently truncated.
    private static readonly string[] s_controlEogNames =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|eot|>", "<|eom|>", "<|end|>", "<|endoftext|>"];

    /// <summary>
    /// Builds the end-of-generation token set: the configured EOS plus any well-known EOG marker
    /// present in <paramref name="vocabLookup"/>. llama.cpp stops on any EOG; mirroring that
    /// prevents a model from decoding an alternate end token as literal text and running on past
    /// its turn. Canonical EOS-family names (<see cref="s_canonicalEosNames"/>) are accepted even
    /// when typed NORMAL (Gemma 4's <c>&lt;eos&gt;</c> is id 1, NORMAL-typed); the bracket-style
    /// markers are accepted only when the vocab types them as control (id present in
    /// <paramref name="specialIds"/>) so an ordinary token that happens to share the string isn't
    /// silently turned into a stop. Always contains <paramref name="eosTokenId"/>.
    /// </summary>
    internal static ImmutableArray<int> BuildEogTokenIds(
        IReadOnlyDictionary<string, int> vocabLookup, IReadOnlySet<int> specialIds, int eosTokenId)
    {
        var eogIds = new List<int> { eosTokenId };

        void TryAdd(string name, bool requireControl)
        {
            if (vocabLookup.TryGetValue(name, out int id) && id > 0 && !eogIds.Contains(id)
                && (!requireControl || specialIds.Contains(id)))
                eogIds.Add(id);
        }

        foreach (var name in s_canonicalEosNames) TryAdd(name, requireControl: false);
        foreach (var name in s_controlEogNames)   TryAdd(name, requireControl: true);
        return [.. eogIds];
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
        {
            text = text.Replace(' ', '▁');
            if (_spmMerges is not null)
                return EncodeSpm(text);
        }

        var ids = _inner.EncodeToIds(text);

        if (ids.Count > 0) return RemapOutOfVocab(ids);

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
    /// Guarantees every emitted id is &lt; <see cref="VocabSize"/>, i.e. addressable in the
    /// model's embedding table. <see cref="CodeGenTokenizer"/> injects model-independent
    /// consecutive-whitespace tokens (ids beyond the supplied vocab — e.g. 2–8-space runs at
    /// ids 49152+) that this GGUF has no embedding row for; feeding one to the GPU embedding
    /// gather reads out of bounds and aborts the CUDA context with error 700 (issue #267), while
    /// the CPU silently reads an adjacent tensor. Any such id is decomposed into its constituent
    /// single-byte tokens (always present in a byte-level BPE vocab), so the whitespace is
    /// preserved as in-vocab tokens rather than dropped. The fast path returns the input
    /// unchanged when every id is already in range, so normal models pay only one scan.
    /// </summary>
    private IReadOnlyList<int> RemapOutOfVocab(IReadOnlyList<int> ids)
    {
        int oob = -1;
        for (int i = 0; i < ids.Count; i++)
            if ((uint)ids[i] >= (uint)VocabSize) { oob = i; break; }
        if (oob < 0) return ids; // common case: nothing to remap

        // Last-resort id for a byte not found in the vocab. UnknownTokenId is normally 0 and
        // in-vocab, but guard so a pathological model (unk = -1 or ≥ VocabSize) can't make the
        // remap itself emit an out-of-range id — that would defeat the whole point.
        int unk = (uint)UnknownTokenId < (uint)VocabSize ? UnknownTokenId : 0;

        var result = new List<int>(ids.Count + 4);
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if ((uint)id < (uint)VocabSize) { result.Add(id); continue; }

            // Decode the offending token and map each GPT-2 byte-level char to its single-byte
            // vocab token (the base alphabet of a byte-level BPE — always in-vocab). The two inner
            // tokenizers decode differently (see Decode): CodeGenTokenizer returns clean UTF-8, so
            // re-encode each byte to its GPT-2 char; the BpeTokenizer fallback already returns the
            // GPT-2 byte-level form, so use it as-is (re-encoding would double-encode it).
            string piece = _inner.Decode(new[] { id }) ?? string.Empty;
            string gpt2 = _needsByteEncoding ? piece : EncodeToGpt2Bytes(piece);
            foreach (char ch in gpt2)
            {
                result.Add(_vocab.TryGetValue(ch.ToString(), out int byteId) && byteId < VocabSize
                    ? byteId
                    : unk);
            }
        }
        return result;
    }

    /// <summary>
    /// SentencePiece-style BPE: split into unicode code points, then iteratively
    /// apply the highest-priority adjacent merge (lowest index in merges array)
    /// until no merges apply. Matches llama.cpp's <c>llm_tokenizer_spm</c> behaviour.
    /// </summary>
    private IReadOnlyList<int> EncodeSpm(string text)
    {
        // Split into Unicode text elements (code points / graphemes), each a start symbol.
        var pieces = new List<string>(text.Length);
        var en = System.Globalization.StringInfo.GetTextElementEnumerator(text);
        while (en.MoveNext()) pieces.Add((string)en.Current);

        var merged = SpmMergePieces(pieces, _spmMerges!);

        var ids = new List<int>(merged.Count);
        foreach (var piece in merged)
        {
            if (_vocab.TryGetValue(piece, out int id))
                ids.Add(id);
            else if (UnknownTokenId >= 0)
                ids.Add(UnknownTokenId);
        }
        return ids;
    }

    /// <summary>
    /// Core SentencePiece merge: given starting symbols (one per code point) and a
    /// bigram→rank merge table, repeatedly apply the lowest-rank adjacent merge (leftmost on
    /// a rank tie) until none apply, returning the surviving pieces in order. Matches
    /// llama.cpp's <c>llm_tokenizer_spm</c>.
    ///
    /// <para>O(n log n) via a min-heap of merge candidates over a symbol doubly-linked list.
    /// The previous implementation rescanned every adjacent pair to find the single best merge
    /// and applied one merge per pass — O(n²) — which made a 40k-token prompt take minutes to
    /// tokenize (the dominant server-request latency). Slots are never reused and a symbol's
    /// text only ever grows via merges, so a slot's current Length uniquely identifies its
    /// state — used as an O(1) staleness check to discard queued candidates whose operands
    /// have since merged away. <c>internal</c> so the parity tests can exercise it directly
    /// with a synthetic merge table (no model file needed).</para>
    /// </summary>
    internal static List<string> SpmMergePieces(
        List<string> pieces, IReadOnlyDictionary<(string, string), int> merges)
    {
        int n = pieces.Count;
        if (n == 0) return pieces;

        var sym = new string?[n];
        var prev = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            sym[i] = pieces[i];
            prev[i] = i - 1;
            next[i] = i + 1 < n ? i + 1 : -1;
        }

        // Priority packs merge rank (primary, lower wins — the merges-file order) with the
        // left slot index (secondary, so ties resolve leftmost, matching the old linear scan).
        var pq = new PriorityQueue<(int l, int r, int llen, int rlen), long>();

        void TryAdd(int l, int r)
        {
            if (l < 0 || r < 0) return;
            if (sym[l] is not { } ls || sym[r] is not { } rs) return;
            if (merges.TryGetValue((ls, rs), out int rank))
                pq.Enqueue((l, r, ls.Length, rs.Length), ((long)rank << 32) | (uint)l);
        }

        for (int i = 0; i + 1 < n; i++) TryAdd(i, i + 1);

        while (pq.Count > 0)
        {
            var (l, r, llen, rlen) = pq.Dequeue();
            // Skip a stale candidate: either operand already consumed (null) or grown (len
            // changed), so this queued pair no longer reflects the current adjacency.
            if (sym[l] is not { } ls || sym[r] is not { } rs || ls.Length != llen || rs.Length != rlen)
                continue;

            // Merge r into l and unlink r from the chain.
            sym[l] = ls + rs;
            sym[r] = null;
            int nn = next[r];
            next[l] = nn;
            if (nn >= 0) prev[nn] = l;

            // Re-evaluate the two adjacencies created around the merged symbol.
            TryAdd(prev[l], l);
            TryAdd(l, nn);
        }

        // Walk the surviving symbols in order (slot 0 is always the live head — nothing
        // precedes it, so it is never removed as a right operand).
        var result = new List<string>();
        for (int s = 0; s >= 0; s = next[s])
            if (sym[s] is { } piece) result.Add(piece);
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
