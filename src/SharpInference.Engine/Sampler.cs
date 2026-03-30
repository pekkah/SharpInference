using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Token sampler: greedy, top-k, top-p (nucleus), temperature, and min-p.
/// Operates on a logit tensor and returns the next token ID.
/// </summary>
public static class Sampler
{
    public static int Sample(Tensor logits, SamplingParams p, Random? rng = null)
    {
        rng ??= Random.Shared;
        // TODO: apply temperature, top-k, top-p, min-p filters then sample
        throw new NotImplementedException();
    }

    public static int Greedy(Tensor logits)
    {
        // TODO: argmax over logits
        throw new NotImplementedException();
    }
}

public sealed record SamplingParams
{
    public float Temperature { get; init; } = 1.0f;
    public int TopK { get; init; } = 0;
    public float TopP { get; init; } = 1.0f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int MaxNewTokens { get; init; } = 512;
    public int[]? StopTokenIds { get; init; }
}
