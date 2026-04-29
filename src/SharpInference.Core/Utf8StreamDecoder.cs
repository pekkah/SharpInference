using System.Text;

namespace SharpInference.Core;

/// <summary>
/// Stateful UTF-8 decoder for streaming token output.
///
/// Byte-level BPE models split a single multi-byte UTF-8 character (CJK, emoji,
/// curly quotes, em-dash, etc.) across several tokens. Decoding each token's
/// bytes independently with <c>Encoding.UTF8.GetString</c> yields U+FFFD
/// replacement characters because each call sees an incomplete sequence.
///
/// This helper wraps <see cref="System.Text.Decoder"/>, which retains incomplete
/// trailing bytes between calls so partial sequences are reassembled across
/// token boundaries.
/// </summary>
public sealed class Utf8StreamDecoder
{
    private readonly Decoder _decoder = Encoding.UTF8.GetDecoder();

    /// <summary>
    /// Feed token bytes to the decoder and return any complete codepoints
    /// produced so far. Trailing partial bytes are buffered for the next call.
    /// </summary>
    public string Append(ReadOnlySpan<byte> bytes)
    {
        if (bytes.IsEmpty) return string.Empty;
        // GetCharCount mutates decoder state, so skip it and size the output
        // buffer using the worst-case bound: each input byte plus an optional
        // surrogate trailing char from completing a buffered 4-byte sequence.
        int maxChars = bytes.Length + 4;
        Span<char> buf = maxChars <= 256 ? stackalloc char[256] : new char[maxChars];
        int written = _decoder.GetChars(bytes, buf, flush: false);
        return written == 0 ? string.Empty : new string(buf[..written]);
    }

    /// <summary>
    /// End of stream — surface any final partial bytes. Truly incomplete
    /// sequences are emitted as a single U+FFFD replacement character.
    /// </summary>
    public string Flush()
    {
        Span<char> buf = stackalloc char[8];
        int written = _decoder.GetChars(ReadOnlySpan<byte>.Empty, buf, flush: true);
        return written == 0 ? string.Empty : new string(buf[..written]);
    }
}
