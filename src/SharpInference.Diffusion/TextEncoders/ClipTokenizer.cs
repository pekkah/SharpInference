using System.Text;
using System.Text.Json;

namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// CLIP BPE tokenizer compatible with OpenAI CLIP / Stable Diffusion.
/// Loads vocabulary and merges from a tokenizer.json (HuggingFace format),
/// which is commonly distributed alongside clip_l.safetensors.
///
/// Output tokens are in range [0, 49407] with:
///   BOS = 49406 (start of text)
///   EOS = 49407 (end of text, used as pooling token)
///   PAD = 0
/// Maximum sequence length: 77.
/// </summary>
public sealed class ClipTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly List<(string first, string second)> _merges;
    private const int BosToken = 49406;
    private const int EosToken = 49407;
    private const int MaxLen   = 77;

    private ClipTokenizer(Dictionary<string, int> vocab, List<(string, string)> merges)
    {
        _vocab  = vocab;
        _merges = merges;
    }

    /// <summary>Load from a HuggingFace tokenizer.json file.</summary>
    public static ClipTokenizer FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model  = doc.RootElement.GetProperty("model");
        var vocabEl = model.GetProperty("vocab");
        var mergesEl = model.GetProperty("merges");

        var vocab = new Dictionary<string, int>(vocabEl.GetArrayLength(), StringComparer.Ordinal);
        foreach (var kv in vocabEl.EnumerateObject())
            vocab[kv.Name] = kv.Value.GetInt32();

        var merges = new List<(string, string)>(mergesEl.GetArrayLength());
        foreach (var m in mergesEl.EnumerateArray())
        {
            var parts = m.GetString()!.Split(' ', 2);
            merges.Add((parts[0], parts[1]));
        }

        return new ClipTokenizer(vocab, merges);
    }

    /// <summary>
    /// Tokenize a text string. Returns token ids padded to MaxLen.
    /// Includes BOS at position 0 and EOS immediately after text tokens.
    /// </summary>
    public int[] Tokenize(string text)
    {
        var ids = new List<int> { BosToken };

        // Simple whitespace + punctuation split (CLIP style)
        var words = SplitWords(text.ToLowerInvariant());
        foreach (var word in words)
        {
            var wordTokens = BpeEncode(word + "</w>");
            foreach (var tok in wordTokens)
            {
                if (ids.Count >= MaxLen - 1) break;
                if (_vocab.TryGetValue(tok, out int id)) ids.Add(id);
            }
            if (ids.Count >= MaxLen - 1) break;
        }

        ids.Add(EosToken);

        // Pad to MaxLen
        while (ids.Count < MaxLen) ids.Add(0);
        return ids.ToArray();
    }

    private List<string> BpeEncode(string token)
    {
        if (token.Length == 0) return new List<string>();

        // Initialize as individual characters (with </w> at end, already appended)
        var pairs = new List<string>();
        for (int i = 0; i < token.Length; i++)
            pairs.Add(token[i].ToString());

        // Apply BPE merges greedily in priority order
        var mergeRank = new Dictionary<(string, string), int>(_merges.Count);
        for (int i = 0; i < _merges.Count; i++)
            mergeRank[_merges[i]] = i;

        while (pairs.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestIdx  = -1;
            for (int i = 0; i < pairs.Count - 1; i++)
            {
                if (mergeRank.TryGetValue((pairs[i], pairs[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }
            if (bestIdx < 0) break;

            string merged = pairs[bestIdx] + pairs[bestIdx + 1];
            pairs.RemoveAt(bestIdx);
            pairs[bestIdx] = merged;
        }
        return pairs;
    }

    private static IEnumerable<string> SplitWords(string text)
    {
        // Basic tokenization matching CLIP's regex loosely:
        // letters+numbers, apostrophes contracted, punctuation as own tokens
        var sb = new StringBuilder();
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c) || c == '\'')
            {
                sb.Append(c);
            }
            else
            {
                if (sb.Length > 0) { yield return sb.ToString(); sb.Clear(); }
                if (!char.IsWhiteSpace(c)) yield return c.ToString();
            }
        }
        if (sb.Length > 0) yield return sb.ToString();
    }
}
