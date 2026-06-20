using System.Collections.Immutable;

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
    /// Immutable so the published stop set can't be tampered with by a consumer.
    /// </summary>
    ImmutableArray<int> EogTokenIds => [EosTokenId];

    /// <summary>Whether BOS token should be automatically prepended.</summary>
    bool AddBosToken { get; }

    /// <summary>
    /// The <c>(Open, Close)</c> special-token IDs that bracket this model's reasoning stream, or
    /// <c>(-1, -1)</c> when the vocabulary defines none. An engine uses these to split the
    /// reasoning channel out of the user-facing text stream (boundary tokens themselves are
    /// consumed, never emitted). Covers both the ChatML <c>&lt;think&gt;</c>/<c>&lt;/think&gt;</c>
    /// convention and Gemma 4's <c>&lt;|channel&gt;</c>/<c>&lt;channel|&gt;</c> "thought" channel.
    /// Default: none — only a vocab-backed tokenizer (e.g. <see cref="GgufTokenizer"/>) resolves a
    /// real pair, so every consumer that constructs an engine gets the same split without
    /// re-deriving the convention itself.
    /// </summary>
    (int Open, int Close) ReasoningTokens => (-1, -1);
}
