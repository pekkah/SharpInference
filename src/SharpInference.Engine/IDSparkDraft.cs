namespace SharpInference.Engine;

/// <summary>
/// One DSpark draft round's output: <see cref="Tokens"/> are the
/// <c>BlockSize</c> greedily-drafted tokens (Markov-corrected, sequential) and
/// <see cref="Confidences"/> their per-position predicted acceptance
/// probabilities (sigmoid of the confidence head; all 1s when the head is
/// absent). The decoder trims the verified prefix by confidence threshold.
/// </summary>
public readonly record struct DSparkProposal(int[] Tokens, float[] Confidences);

/// <summary>
/// A DSpark draft head as consumed by <see cref="DSparkDecoder"/>: an
/// EAGLE-3-style block-parallel drafter conditioned on target hidden-state
/// taps. The decoder feeds it fused-input taps for every committed context
/// position (<see cref="AppendContext"/>, exactly once per position, in order)
/// and asks for one draft block per decode round (<see cref="ProposeBlock"/>).
/// Split from <see cref="DSparkDraftModel"/> so decoder logic is testable with
/// a scripted fake.
/// </summary>
public interface IDSparkDraft : IDisposable
{
    /// <summary>Draft positions per round (config block_size).</summary>
    int BlockSize { get; }

    /// <summary>Vocabulary size — must match the target model's.</summary>
    int VocabSize { get; }

    /// <summary>Expected tap row width: taps-per-position handed to <see cref="AppendContext"/>.</summary>
    int TapDim { get; }

    /// <summary>Context positions currently held (== the next AppendContext startPos).</summary>
    int ContextLength { get; }

    /// <summary>
    /// Hard cap on context positions the draft can hold (its RoPE window /
    /// max_position_embeddings). The decoder stops drafting at this bound —
    /// it may be smaller than the target's window.
    /// </summary>
    int MaxContext { get; }

    /// <summary>
    /// Consume target taps for positions <paramref name="startPos"/> ..
    /// <paramref name="startPos"/>+<paramref name="count"/>-1 (row-major
    /// [count, TapDim]) into the per-layer context K/V cache.
    /// <paramref name="startPos"/> must equal <see cref="ContextLength"/>.
    /// </summary>
    void AppendContext(ReadOnlySpan<float> taps, int startPos, int count);

    /// <summary>
    /// Draft one block anchored at <paramref name="anchorPos"/> (the position of
    /// <paramref name="anchorToken"/>, the newest committed-but-unprocessed token).
    /// Requires <see cref="ContextLength"/> == <paramref name="anchorPos"/> — the
    /// block attends over all cached context plus itself (bidirectionally).
    /// Greedy (temperature 0) drafting only.
    /// </summary>
    DSparkProposal ProposeBlock(int anchorToken, int anchorPos);

    /// <summary>Drop context K/V for positions &gt;= <paramref name="length"/>.</summary>
    void TruncateContext(int length);

    /// <summary>Drop all context (new conversation).</summary>
    void ResetContext();
}
