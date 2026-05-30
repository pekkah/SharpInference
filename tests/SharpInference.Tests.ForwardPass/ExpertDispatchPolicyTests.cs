using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for the per-expert dispatch policy (issue #54). These cover
/// the policy types in isolation — the integration with
/// <see cref="CudaHybridGdnForwardPass"/> is exercised in the
/// CUDA-dependent test suite.
/// </summary>
public sealed class ExpertDispatchPolicyTests
{
    private static ExpertDispatchContext MakeCtx(int layer = 0, int expert = 0,
        long sizeBytes = 1_800_000, bool hit = false, bool prefetch = false) =>
        new(layer, expert, sizeBytes, hit, prefetch);

    [Fact]
    public void WaitForGpuPolicy_AlwaysReturnsWaitForGpu()
    {
        var policy = WaitForGpuPolicy.Instance;
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx()));
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx(layer: 7, expert: 42)));
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx(hit: true)));
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx(prefetch: true)));
    }

    [Fact]
    public void WaitForGpuPolicy_LatencyRecordersAreNoOp()
    {
        var policy = WaitForGpuPolicy.Instance;
        policy.RecordCpuLatency(0, 0, 1000.0);
        policy.RecordGpuLatency(0, 0, 1000.0);
        // Decision must remain WaitForGpu regardless of any recorded telemetry.
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx()));
    }

    [Fact]
    public void FiddlerDispatchPolicy_DefaultsFavorCpuWhenInitialCpuLowerThanGpu()
    {
        var policy = new FiddlerDispatchPolicy(initialCpuMillis: 1.0, initialGpuMillis: 5.0);
        Assert.Equal(ExpertDispatchDecision.RunOnCpu, policy.Decide(MakeCtx()));
    }

    [Fact]
    public void FiddlerDispatchPolicy_FavorsGpuWhenCpuEstimateExceedsGpuPlaceholder()
    {
        var policy = new FiddlerDispatchPolicy(initialCpuMillis: 10.0, initialGpuMillis: 2.0);
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx()));
    }

    [Fact]
    public void FiddlerDispatchPolicy_CpuLatencyEwmaPullsDecisionTowardLatestSample()
    {
        // Initial 10 ms CPU vs 5 ms GPU → starts on GPU. Feed several short
        // CPU samples; EWMA should pull below 5 ms and flip to CPU.
        var policy = new FiddlerDispatchPolicy(initialCpuMillis: 10.0,
            initialGpuMillis: 5.0, ewmaAlpha: 0.5);
        var ctx = MakeCtx(layer: 3, expert: 17);

        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(in ctx));

        // First sample (no prior key) seeds the EWMA at the sample value.
        policy.RecordCpuLatency(3, 17, 1.0);
        Assert.Equal(ExpertDispatchDecision.RunOnCpu, policy.Decide(in ctx));

        // A different (layer, expert) still falls back to the initial estimate.
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx(layer: 3, expert: 18)));
    }

    [Fact]
    public void FiddlerDispatchPolicy_GpuLatencyRecorderIsAcceptedButCurrentlyIgnored()
    {
        var policy = new FiddlerDispatchPolicy(initialCpuMillis: 10.0, initialGpuMillis: 2.0);
        // The CUDA hybrid path will call RecordGpuLatency once #78 lands; today
        // the policy ignores it. Calling must not throw.
        policy.RecordGpuLatency(0, 0, 7.5);
        Assert.Equal(ExpertDispatchDecision.WaitForGpu, policy.Decide(MakeCtx()));
    }

    /// <summary>
    /// Drop-in test stub for the dispatch policy used in higher-level tests
    /// (the CUDA hybrid suite injects one of these to force the CPU fallback
    /// branch on real model weights). Keeping it here keeps the surface
    /// minimal — just verifies that a custom policy implementation is wired
    /// in correctly.
    /// </summary>
    private sealed class RecordingPolicy : IExpertDispatchPolicy
    {
        public ExpertDispatchDecision Returns { get; init; }
        public int DecideCalls { get; private set; }
        public int CpuRecordCalls { get; private set; }
        public int GpuRecordCalls { get; private set; }

        public ExpertDispatchDecision Decide(in ExpertDispatchContext ctx)
        {
            DecideCalls++;
            return Returns;
        }

        public void RecordCpuLatency(int layer, int expertId, double milliseconds) => CpuRecordCalls++;
        public void RecordGpuLatency(int layer, int expertId, double milliseconds) => GpuRecordCalls++;
    }

    [Fact]
    public void CustomPolicy_DecideCountsContextsCorrectly()
    {
        var p = new RecordingPolicy { Returns = ExpertDispatchDecision.RunOnCpu };
        Assert.Equal(ExpertDispatchDecision.RunOnCpu, p.Decide(MakeCtx()));
        Assert.Equal(ExpertDispatchDecision.RunOnCpu, p.Decide(MakeCtx(layer: 1)));
        Assert.Equal(2, p.DecideCalls);
        p.RecordCpuLatency(0, 0, 1.0);
        Assert.Equal(1, p.CpuRecordCalls);
        p.RecordGpuLatency(0, 0, 1.0);
        Assert.Equal(1, p.GpuRecordCalls);
    }
}
