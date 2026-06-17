namespace SharpInference.Engine;

/// <summary>
/// Lane→token mapping for the width-4 batched-verify (MTP) duplicated-input tail
/// (issues #30/#209). The k draft tokens of a <c>BatchVerify</c> step are processed in
/// groups of four sharing one CPU mmap weight read (<see cref="Cpu.SimdKernels.MatVec4In"/>).
/// Only the final group can be partial; it duplicates its last real token into the
/// empty lanes so the quad kernel always sees four valid inputs, and the duplicate
/// lanes' outputs are routed to a sink and discarded by the caller. Because MatVec4In
/// is bit-identical per token slot, every real token's bits are independent of the
/// batch width k — the per-position k-parity-independence contract the MTP greedy
/// parity test pins. Centralised here so the three former hand-copied tail idioms (the
/// CUDA dense-FFN loop, and the CPU pass's FFN + lm_head loops) share one definition.
/// </summary>
internal static class MtpBatchTail
{
    /// <summary>
    /// For the width-4 group starting at <paramref name="i"/> over a batch of
    /// <paramref name="k"/> tokens (<c>0 ≤ i &lt; k</c>), yields the four lane→token
    /// indices and <paramref name="nReal"/>, the count of real (non-duplicate) lanes
    /// in this group (1..4). In-range lanes map to their own token; past-the-end lanes
    /// clamp to the group's last real token (<c>k-1</c>) so the kernel reads valid
    /// inputs and the caller can route the duplicate output to a sink.
    /// </summary>
    public static void Group4(int i, int k,
        out int j0, out int j1, out int j2, out int j3, out int nReal)
    {
        int last = k - 1;
        j0 = i;                              // i < k by the loop guard
        j1 = i + 1 <= last ? i + 1 : last;
        j2 = i + 2 <= last ? i + 2 : last;
        j3 = i + 3 <= last ? i + 3 : last;
        nReal = Math.Min(4, k - i);
    }
}
