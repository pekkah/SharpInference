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

    int VocabSize { get; }
    int BosTokenId { get; }
    int EosTokenId { get; }
    int UnknownTokenId { get; }
    int PadTokenId { get; }

    /// <summary>Whether BOS token should be automatically prepended.</summary>
    bool AddBosToken { get; }
}
