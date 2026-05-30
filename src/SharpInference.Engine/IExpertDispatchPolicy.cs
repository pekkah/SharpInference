namespace SharpInference.Engine;

/// <summary>
/// Decision returned by <see cref="IExpertDispatchPolicy"/> for a single routed
/// MoE expert on the CUDA hybrid path. The dispatcher consults the policy
/// whenever the SLRU expert cache misses on the GPU side: instead of
/// unconditionally paying the sync PCIe upload (the "Upload-stall"), the
/// policy may decide to run the expert on CPU using mmap'd weights instead.
/// </summary>
public enum ExpertDispatchDecision
{
    /// <summary>
    /// Block the decode thread on the SLRU sync upload (current behaviour).
    /// Cheapest when the upload completes before the CPU could finish, or
    /// when the CPU path is unavailable.
    /// </summary>
    WaitForGpu,

    /// <summary>
    /// Bypass the GPU upload for this token and evaluate the expert on CPU
    /// over its mmap'd weights (Fiddler, ICLR 2025 / arXiv:2402.07033). The
    /// expert is NOT inserted into the SLRU as a side-effect — a future
    /// access still races the same upload decision.
    /// </summary>
    RunOnCpu,
}

/// <summary>
/// Inputs to <see cref="IExpertDispatchPolicy.Decide"/>. Layout is a readonly
/// struct so the dispatch site can construct one per (layer, expert) without
/// heap allocation on the MoE hot path.
/// </summary>
public readonly record struct ExpertDispatchContext(
    int Layer,
    int ExpertId,
    long ExpertSizeBytes,
    bool SlotCacheHit,
    bool PrefetchReady);

/// <summary>
/// Per-expert dispatch policy for the CUDA hybrid MoE path.
///
/// <para>
/// On SLRU miss the dispatcher calls <see cref="Decide"/> with the current
/// (expert ID, predicted GPU-arrival time, predicted CPU compute time) and
/// follows the returned decision. Default
/// (<see cref="WaitForGpuPolicy"/>) preserves the pre-Fiddler "stall on the
/// sync upload" behaviour.
/// </para>
///
/// <para>
/// Implementations are expected to be cheap (a few comparisons + maybe a
/// EWMA update) — the dispatch site is on the MoE per-expert hot path and
/// touches the policy once per (layer × top-K) per token.
/// </para>
/// </summary>
public interface IExpertDispatchPolicy
{
    /// <summary>
    /// Decide where to evaluate one routed expert. Called only on SLRU miss;
    /// hits bypass the policy entirely.
    /// </summary>
    ExpertDispatchDecision Decide(in ExpertDispatchContext ctx);

    /// <summary>
    /// Record the wall-clock cost of the most recent CPU-fallback evaluation
    /// for this (layer, expert) so the policy can refine its CPU-time
    /// estimate. Called only when <see cref="ExpertDispatchDecision.RunOnCpu"/>
    /// was returned and actually executed.
    /// </summary>
    void RecordCpuLatency(int layer, int expertId, double milliseconds);

    /// <summary>
    /// Record the wall-clock cost of the most recent GPU SLRU-miss
    /// sync-upload path for this (layer, expert). Used to refine the
    /// GPU-arrival estimate.
    /// </summary>
    void RecordGpuLatency(int layer, int expertId, double milliseconds);
}

/// <summary>
/// Default policy: always wait for the GPU sync upload. Equivalent to the
/// pre-Fiddler dispatch path — produces no behavioural change.
/// </summary>
public sealed class WaitForGpuPolicy : IExpertDispatchPolicy
{
    public static readonly WaitForGpuPolicy Instance = new();

    public ExpertDispatchDecision Decide(in ExpertDispatchContext ctx) =>
        ExpertDispatchDecision.WaitForGpu;

    public void RecordCpuLatency(int layer, int expertId, double milliseconds) { }
    public void RecordGpuLatency(int layer, int expertId, double milliseconds) { }
}

/// <summary>
/// Fiddler-style policy: on SLRU miss, dispatch the expert to CPU when the
/// estimated CPU compute time is less than the estimated GPU arrival time;
/// otherwise wait for the GPU upload.
///
/// <para>
/// The CPU-time estimate is a per-(layer, expert-size) EWMA seeded with a
/// caller-supplied initial guess and updated through
/// <see cref="RecordCpuLatency"/>. The GPU-arrival estimate is, for now, a
/// constant placeholder (<paramref name="initialGpuMillis"/> in the
/// constructor) — issue #78 will refine this once real PCIe-queue-depth
/// telemetry is available from the async upload subsystem. The constant
/// placeholder lets the seam ship today and intentionally biases the
/// default-tuned policy toward WaitForGpu (the pre-Fiddler behaviour) until
/// per-expert CPU samples accumulate.
/// </para>
/// </summary>
public sealed class FiddlerDispatchPolicy : IExpertDispatchPolicy
{
    private readonly double _initialCpuMs;
    private readonly double _initialGpuMs;
    private readonly double _ewmaAlpha;
    private readonly Dictionary<long, double> _cpuLatencyEwma = new();

    /// <param name="initialCpuMillis">Seed estimate for CPU compute per
    /// expert; replaced by EWMA once samples arrive.</param>
    /// <param name="initialGpuMillis">Constant placeholder for predicted GPU
    /// arrival time. Until #78 lands the dispatcher has no PCIe-queue depth
    /// to refine this from, so any value sets the bias of the default
    /// policy: smaller → favour CPU more aggressively.</param>
    /// <param name="ewmaAlpha">Smoothing factor for the CPU EWMA; 0.2 keeps
    /// roughly the last five samples in the running estimate.</param>
    public FiddlerDispatchPolicy(double initialCpuMillis = 4.0,
        double initialGpuMillis = 2.0,
        double ewmaAlpha = 0.2)
    {
        _initialCpuMs = initialCpuMillis;
        _initialGpuMs = initialGpuMillis;
        _ewmaAlpha = ewmaAlpha;
    }

    public ExpertDispatchDecision Decide(in ExpertDispatchContext ctx)
    {
        double cpuMs = _cpuLatencyEwma.TryGetValue(Key(ctx.Layer, ctx.ExpertId), out var ewma)
            ? ewma
            : _initialCpuMs;
        return cpuMs < _initialGpuMs
            ? ExpertDispatchDecision.RunOnCpu
            : ExpertDispatchDecision.WaitForGpu;
    }

    public void RecordCpuLatency(int layer, int expertId, double milliseconds)
    {
        long k = Key(layer, expertId);
        if (_cpuLatencyEwma.TryGetValue(k, out var prev))
            _cpuLatencyEwma[k] = prev + _ewmaAlpha * (milliseconds - prev);
        else
            _cpuLatencyEwma[k] = milliseconds;
    }

    public void RecordGpuLatency(int layer, int expertId, double milliseconds) { }

    private static long Key(int layer, int expertId) =>
        ((long)layer << 32) | (uint)expertId;
}
