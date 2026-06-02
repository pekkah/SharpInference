namespace SharpInference.Core;

/// <summary>
/// Tokenizer abstraction. Backed by Microsoft.ML.Tokenizers (BPE / SentencePiece).
/// </summary>
public interface ITokenizer
{
    /// <summary>Encode text into token IDs.</summary>
    IReadOnlyList<int> Encode(string text);

    /// <summary>Decode token IDs back to text.</summary>
    string Decode(IEnumerable<int> tokens);

    /// <summary>
    /// Decode a single token to its raw UTF-8 bytes. For byte-level BPE tokenizers
    /// the bytes are exactly the token's contribution to the output stream — they
    /// may form an incomplete UTF-8 sequence on their own. Stream-decode through
    /// <see cref="Utf8StreamDecoder"/> to reassemble multi-byte characters
    /// split across tokens.
    /// </summary>
    byte[] DecodeBytes(int token);

    int VocabSize { get; }
    int BosTokenId { get; }
    int EosTokenId { get; }
    int UnknownTokenId { get; }
    int PadTokenId { get; }

    /// <summary>
    /// All end-of-generation token IDs (the configured EOS plus any alternate EOG control
    /// tokens this vocab defines). Generation stops on ANY of these. Defaults to just
    /// <see cref="EosTokenId"/> for tokenizers that don't distinguish a broader set.
    /// </summary>
    int[] EogTokenIds => [EosTokenId];

    /// <summary>Whether BOS token should be automatically prepended.</summary>
    bool AddBosToken { get; }
}
