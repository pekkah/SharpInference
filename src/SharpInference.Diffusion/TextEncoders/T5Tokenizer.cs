using System.Text;
using System.Text.Json;

namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// Minimal SentencePiece-compatible T5 tokenizer.
/// Loads vocabulary from a tokenizer.json (HuggingFace format) distributed
/// alongside t5xxl_fp16.safetensors as tokenizer.json / spiece.model.
///
/// Supports the unigram tokenizer model used by T5.
/// BOS = 0, EOS = 1, UNK = 2, PAD = 0.
/// Maximum output length: 77 tokens (FLUX convention).
/// </summary>
public sealed class T5Tokenizer
{
    private readonly string[] _idToToken;
    private readonly Dictionary<string, int> _tokenToId;
    private const int EosToken = 1;
    private const int MaxLen   = 77;

    private T5Tokenizer(string[] idToToken, Dictionary<string, int> tokenToId)
    {
        _idToToken  = idToToken;
        _tokenToId  = tokenToId;
    }

    /// <summary>Load from a HuggingFace tokenizer.json file.</summary>
    public static T5Tokenizer FromFile(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var model   = doc.RootElement.GetProperty("model");
        var vocabEl = model.GetProperty("vocab");

        // vocab: array of [token_string, score] pairs
        var tokens = new List<string> { "<pad>", "</s>", "<unk>" };
        var tokenToId = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["<pad>"]  = 0,
            ["</s>"]   = 1,
            ["<unk>"]  = 2,
        };

        foreach (var entry in vocabEl.EnumerateArray())
        {
            string tok = entry[0].GetString()!;
            if (!tokenToId.ContainsKey(tok))
            {
                tokenToId[tok] = tokens.Count;
                tokens.Add(tok);
            }
        }

        return new T5Tokenizer(tokens.ToArray(), tokenToId);
    }

    /// <summary>
    /// Tokenize text using greedy maximal-munch on SentencePiece vocabulary.
    /// Returns token ids (no padding — caller truncates/pads as needed).
    /// </summary>
    public int[] Tokenize(string text)
    {
        var ids = new List<int>();
        // Normalize: T5 prepends a space to the input
        string normalized = "\u2581" + text.Replace(" ", "\u2581");

        int pos = 0;
        while (pos < normalized.Length && ids.Count < MaxLen - 1)
        {
            // Greedy longest match
            int bestLen = 0, bestId = 2; // UNK
            for (int len = Math.Min(normalized.Length - pos, 32); len >= 1; len--)
            {
                string sub = normalized.Substring(pos, len);
                if (_tokenToId.TryGetValue(sub, out int id))
                {
                    bestLen = len;
                    bestId  = id;
                    break;
                }
            }
            ids.Add(bestId);
            pos += bestLen == 0 ? 1 : bestLen;
        }

        ids.Add(EosToken);
        return ids.ToArray();
    }
}
