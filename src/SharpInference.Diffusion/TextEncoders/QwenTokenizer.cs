using System.Text;
using System.Text.Json;
using SharpInference.Core;

namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// Minimal Qwen3 BPE tokenizer for Z-Image text encoding.
///
/// Loads a HuggingFace tokenizer.json (ByteLevel BPE format as used by Qwen3).
/// If tokenizer_config.json (same directory) contains a chat_template key,
/// it is parsed as a Jinja2 template via <see cref="JinjaChatTemplate"/> and used in
/// <see cref="EncodeWithTemplate"/>. Otherwise falls back to the hardcoded Qwen3 ChatML template.
///
/// Download tokenizer.json from:
///   https://huggingface.co/Tongyi-MAI/Z-Image-Turbo  →  tokenizer/tokenizer.json
///
/// Special token IDs (Qwen3):
///   &lt;|im_start|&gt;  = 151644
///   &lt;|im_end|&gt;    = 151645
/// </summary>
public sealed class QwenTokenizer
{
    // Byte-level pre-tokenization mapping (GPT-2 style)
    private static readonly string[] _bytesAsTokens = BuildBytesTable();

    private readonly Dictionary<string, int> _vocab;
    // merge → rank for O(1) lookup during BPE
    private readonly Dictionary<(string, string), int> _mergeRanks;
    // Optional parsed Jinja2 chat template from tokenizer_config.json
    private readonly JinjaChatTemplate? _chatTemplate;

    private QwenTokenizer(Dictionary<string, int> vocab,
                          Dictionary<(string, string), int> mergeRanks,
                          JinjaChatTemplate? chatTemplate)
    {
        _vocab        = vocab;
        _mergeRanks   = mergeRanks;
        _chatTemplate = chatTemplate;
    }

    public static QwenTokenizer FromFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc    = JsonDocument.Parse(stream);
        var root         = doc.RootElement;

        // Vocabulary: model.vocab  { token: id }
        var vocabEl = root.GetProperty("model").GetProperty("vocab");
        var vocab   = new Dictionary<string, int>(160_000, StringComparer.Ordinal);
        foreach (var prop in vocabEl.EnumerateObject())
            vocab[prop.Name] = prop.Value.GetInt32();

        // Also read added_tokens for special tokens like <|im_start|>
        if (root.TryGetProperty("added_tokens", out var addedToks))
        {
            foreach (var tok in addedToks.EnumerateArray())
            {
                string content = tok.GetProperty("content").GetString()!;
                int    id      = tok.GetProperty("id").GetInt32();
                vocab[content] = id;
            }
        }

        // Merges: model.merges  — each entry is either a string "a b" or an array ["a","b"]
        // Build rank dictionary for O(1) BPE lookup
        var mergesEl   = root.GetProperty("model").GetProperty("merges");
        int mergeCount = mergesEl.GetArrayLength();
        var mergeRanks = new Dictionary<(string, string), int>(mergeCount);
        int rank       = 0;
        foreach (var el in mergesEl.EnumerateArray())
        {
            string a, b;
            if (el.ValueKind == JsonValueKind.String)
            {
                var s  = el.GetString()!;
                int sp = s.IndexOf(' ');
                a = s[..sp];
                b = s[(sp + 1)..];
            }
            else // Array ["a","b"]
            {
                a = el[0].GetString()!;
                b = el[1].GetString()!;
            }
            mergeRanks.TryAdd((a, b), rank++);
        }

        // Try to load the Jinja2 chat template from tokenizer_config.json (same directory)
        JinjaChatTemplate? chatTemplate = null;
        var configPath = Path.Combine(Path.GetDirectoryName(path)!, "tokenizer_config.json");
        if (File.Exists(configPath))
        {
            try
            {
                using var cfgStream = File.OpenRead(configPath);
                using var cfgDoc    = JsonDocument.Parse(cfgStream);
                if (cfgDoc.RootElement.TryGetProperty("chat_template", out var tmplEl))
                {
                    var tmplStr = tmplEl.GetString();
                    if (!string.IsNullOrWhiteSpace(tmplStr))
                        chatTemplate = new JinjaChatTemplate(tmplStr);
                }
            }
            catch { /* ignore — will use hardcoded fallback */ }
        }

        return new QwenTokenizer(vocab, mergeRanks, chatTemplate);
    }

    /// <summary>
    /// Encode a text prompt using the model's chat template with thinking mode enabled.
    /// If a Jinja2 template is available (from tokenizer_config.json), it is used.
    /// Otherwise falls back to the hardcoded Qwen3 ChatML + &lt;think&gt; format.
    /// Returns token IDs (excludes the final EOS that the LLM would generate).
    /// </summary>
    public int[] EncodeWithTemplate(string prompt)
    {
        string text;
        if (_chatTemplate != null)
        {
            // Render via Jinja2 with add_generation_prompt=true
            var messages = JinjaChatTemplate.BuildMessages(prompt, systemContent: null);
            text = _chatTemplate.Render(new Dictionary<string, object?>
            {
                ["messages"]              = messages,
                ["add_generation_prompt"] = true,
                ["tools"]                 = null,
            });
            // Z-Image encoder always runs in thinking mode: append <think>\n so the model
            // produces structured reasoning that the vision head can use.
            if (!text.Contains("<think>"))
                text += "<think>\n";
        }
        else
        {
            // Hardcoded Qwen3 ChatML with thinking mode
            text = $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";
        }

        return Encode(text);
    }

    /// <summary>Encode raw text without chat template.</summary>
    public int[] Encode(string text)
    {
        var result = new List<int>(text.Length);

        // Split on special tokens first
        foreach (var chunk in SplitOnSpecialTokens(text))
        {
            if (_vocab.TryGetValue(chunk, out int specialId))
            {
                result.Add(specialId);
            }
            else
            {
                // Byte-level BPE encode
                var words = ByteLevelPreTokenize(chunk);
                foreach (var word in words)
                    result.AddRange(BpeEncode(word));
            }
        }

        return result.ToArray();
    }

    // ── BPE ───────────────────────────────────────────────────────────────

    private IEnumerable<int> BpeEncode(string[] byteTokens)
    {
        if (byteTokens.Length == 0) yield break;
        if (byteTokens.Length == 1)
        {
            if (_vocab.TryGetValue(byteTokens[0], out int id)) yield return id;
            yield break;
        }

        // BPE: repeatedly merge the highest-priority pair using O(1) rank lookup
        var tokens = new List<string>(byteTokens);

        while (tokens.Count >= 2)
        {
            int bestRank = int.MaxValue;
            int bestPos  = -1;

            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_mergeRanks.TryGetValue((tokens[i], tokens[i + 1]), out int r) && r < bestRank)
                {
                    bestRank = r;
                    bestPos  = i;
                }
            }

            if (bestPos < 0) break;

            tokens[bestPos] = tokens[bestPos] + tokens[bestPos + 1];
            tokens.RemoveAt(bestPos + 1);
        }

        foreach (var tok in tokens)
            if (_vocab.TryGetValue(tok, out int id)) yield return id;
    }

    // ── Pre-tokenization ──────────────────────────────────────────────────

    // GPT-2 / Qwen byte-level pre-tokenization: map each byte of each "word" to
    // a printable unicode character, keeping whitespace semantics.
    private static string[][] ByteLevelPreTokenize(string text)
    {
        // Split on whitespace boundaries like GPT-2 (simplified: word-based)
        // Prefix non-first words with Ġ (unicode 0x0120, represents space)
        if (string.IsNullOrEmpty(text)) return [];

        var words = new List<string[]>();
        bool atStart = true;
        var sb = new StringBuilder();
        var bytes = Encoding.UTF8.GetBytes(text);

        // We use a simple word split: words are sequences of non-space chars,
        // space prefixed with Ġ for GPT-2 byte-level encoding
        bool prevSpace = true;
        sb.Clear();
        foreach (byte b in bytes)
        {
            bool isSpace = b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D;
            if (isSpace)
            {
                if (sb.Length > 0)
                {
                    words.Add(ToByteTokens(sb.ToString()));
                    sb.Clear();
                }
                prevSpace = true;
            }
            else
            {
                if (prevSpace && !atStart)
                    sb.Append('\u0120'); // Ġ = space prefix
                sb.Append(_bytesAsTokens[b]);
                prevSpace = false;
                atStart   = false;
            }
        }
        if (sb.Length > 0) words.Add(ToByteTokens(sb.ToString()));

        return words.ToArray();
    }

    private static string[] ToByteTokens(string word)
    {
        // Split UTF-8-encoded word into individual character tokens (already byte-mapped)
        var chars = new string[word.Length];
        for (int i = 0; i < word.Length; i++) chars[i] = word[i].ToString();
        return chars;
    }

    // ── Special token splitting ────────────────────────────────────────────

    private static readonly string[] _specialTokens =
    [
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<think>", "</think>", "<|vision_start|>", "<|vision_end|>",
    ];

    private static IEnumerable<string> SplitOnSpecialTokens(string text)
    {
        int pos = 0;
        while (pos < text.Length)
        {
            int foundAt  = int.MaxValue;
            string? found = null;

            foreach (var st in _specialTokens)
            {
                int idx = text.IndexOf(st, pos, StringComparison.Ordinal);
                if (idx >= 0 && idx < foundAt) { foundAt = idx; found = st; }
            }

            if (found == null)
            {
                yield return text[pos..];
                yield break;
            }

            if (foundAt > pos) yield return text[pos..foundAt];
            yield return found;
            pos = foundAt + found.Length;
        }
    }

    // ── Byte table ────────────────────────────────────────────────────────

    private static string[] BuildBytesTable()
    {
        // GPT-2 byte-to-unicode mapping
        var table = new string[256];
        int n = 0;
        for (int i = 0; i < 256; i++)
        {
            if ((i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF))
                table[i] = ((char)i).ToString();
            else
                table[i] = ((char)(256 + n++)).ToString();
        }
        return table;
    }
}
