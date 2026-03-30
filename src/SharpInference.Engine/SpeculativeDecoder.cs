using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Speculative decoding: a small draft model generates K tokens which the
/// target model verifies in a single forward pass, accepting or rejecting each.
/// </summary>
public sealed class SpeculativeDecoder
{
    private readonly InferenceEngine _target;
    private readonly InferenceEngine _draft;
    private readonly int _lookahead;

    public SpeculativeDecoder(InferenceEngine target, InferenceEngine draft, int lookahead = 4)
    {
        _target = target;
        _draft = draft;
        _lookahead = lookahead;
    }

    /// <summary>
    /// Produce the next accepted token using speculative decoding.
    /// </summary>
    public ValueTask<int> NextTokenAsync(
        ReadOnlyMemory<int> context,
        SamplingParams sampling,
        CancellationToken ct = default)
    {
        // TODO: draft K tokens, verify with target, accept/reject
        throw new NotImplementedException();
    }
}
