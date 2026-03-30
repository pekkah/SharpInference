namespace SharpInference.Core;

/// <summary>
/// Tokenizer abstraction. Backed by Microsoft.ML.Tokenizers (BPE / SentencePiece / Tiktoken).
/// </summary>
public interface ITokenizer
{
    /// <summary>Encode text into token IDs.</summary>
    ReadOnlyMemory<int> Encode(ReadOnlySpan<char> text);

    /// <summary>Decode token IDs back to text, streaming piece by piece.</summary>
    IEnumerable<string> Decode(ReadOnlySpan<int> tokens);

    int VocabSize { get; }
    int BosTokenId { get; }
    int EosTokenId { get; }
    int PadTokenId { get; }
}
