using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Bit-parity oracle for CUDA Graph decode on the <b>non-Gemma dense</b> path (issue #158,
/// the Item-B analogue of <see cref="Gemma4CudaGraphParityTests"/>). The captured-and-
/// replayed Qwen3-8B Q4_K decode region (<c>CudaForwardPass.RunDeviceRegion</c>) runs the
/// identical kernels in the identical order as the direct-launch path — only the launch
/// mechanism differs — so graph-on vs graph-off logits must be <b>bit-identical</b> at every
/// decode step, not merely argmax-equal.
///
/// Silent-skip pattern: no-ops when CUDA isn't available or the GGUF isn't on disk, matching
/// the sibling Qwen3Cuda* / Gemma4Cuda* test files.
/// </summary>
public sealed class Qwen3CudaGraphParityTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    // "Hello, world! I am a virtual model. 2 3 4" — a mixed-vocab prompt of ordinary ids so
    // the post-prefill decode loop produces a non-degenerate token stream to compare.
    private static readonly int[] Tokens =
        { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13, 220, 17, 18, 19 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absolute = { $@"E:\models\{ModelFile}", $@"C:\p\sharpi\models\{ModelFile}" };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Headline #158 check: graph-on decode on a dense Q4_K model must be bit-identical to
    /// graph-off, and the graph must actually engage (no silent fallback). Prefill runs the
    /// per-token loop (BatchedPrefillEnabled = false) so the very first decode token captures
    /// a clean, eviction-free region.
    /// </summary>
    [Fact]
    public void Qwen3_8B_CudaGraph_AllGpu_BitMatchesDirectLaunch()
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        // Sanity: a dense, non-Gemma SwiGLU model (no PLE / per-layer head_dim / MoE).
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.HasPerLayerTokenEmbd);
        Assert.False(hp.IsMoE);

        const int NSteps = 8;

        // Reference: graph OFF (direct launches). Capture full logits per decode step.
        var refLogits = new float[NSteps][];
        var refTokens = new int[NSteps];
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512)
                { UseCudaGraph = false, BatchedPrefillEnabled = false };
            var logits = fwd.Prefill(Tokens);
            refLogits[0] = logits.ToArray();
            refTokens[0] = Argmax(logits);
            int pos = Tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                var step = fwd.Forward(refTokens[i - 1], pos++);
                refLogits[i] = step.ToArray();
                refTokens[i] = Argmax(step);
            }
        }

        // Candidate: graph ON (capture on the first decode token, replay after).
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512)
                { UseCudaGraph = true, BatchedPrefillEnabled = false };
            var logits = fwd.Prefill(Tokens);
            AssertBitIdentical(refLogits[0], logits, step: 0);
            int pos = Tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                // Drive the decode with the REFERENCE tokens so both runs see the same input
                // sequence even if (hypothetically) an early step diverged.
                var step = fwd.Forward(refTokens[i - 1], pos++);
                AssertBitIdentical(refLogits[i], step, step: i);
            }

            // Guard against a silent fallback: the graph path must actually have engaged.
            Assert.True(gpu.GraphReady,
                "CUDA graph was never captured — the parity test silently ran direct launches " +
                "on both sides and proves nothing. Check TryRunDeviceRegionViaGraph gating.");
        }
    }

    /// <summary>
    /// Regression guard for the SnapKV gate: a configured-but-unevicted budget keeps
    /// <c>_kvEvictedCount == 0</c>, so the cache fills sequentially and graphs MUST engage
    /// and match bit-for-bit. Eviction (prompt &gt; budget) is the separate bail; here the
    /// 14-token prompt fits the 512 budget.
    /// </summary>
    [Fact]
    public void Qwen3_8B_CudaGraph_SnapKvConfiguredNoEvict_BitMatches()
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "512"); // configured, prompt << 512
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            const int NSteps = 8;

            var refLogits = new float[NSteps][];
            var refTokens = new int[NSteps];
            using (var gpu = TryCreate())
            {
                if (gpu is null) return;
                using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024)
                    { UseCudaGraph = false, BatchedPrefillEnabled = false };
                var logits = fwd.Prefill(Tokens);
                refLogits[0] = logits.ToArray();
                refTokens[0] = Argmax(logits);
                int pos = Tokens.Length;
                for (int i = 1; i < NSteps; i++)
                {
                    var step = fwd.Forward(refTokens[i - 1], pos++);
                    refLogits[i] = step.ToArray();
                    refTokens[i] = Argmax(step);
                }
            }

            using (var gpu = TryCreate())
            {
                if (gpu is null) return;
                using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024)
                    { UseCudaGraph = true, BatchedPrefillEnabled = false };
                var logits = fwd.Prefill(Tokens);
                AssertBitIdentical(refLogits[0], logits, step: 0);
                int pos = Tokens.Length;
                for (int i = 1; i < NSteps; i++)
                    AssertBitIdentical(refLogits[i], fwd.Forward(refTokens[i - 1], pos++), step: i);

                Assert.True(gpu.GraphReady,
                    "Graphs must engage when a SnapKV budget is set but no eviction occurs " +
                    "(prompt fits the budget). GraphReady=false means the _kvEvictedCount gate " +
                    "is over-broad or the _snapKvCaptureSlot bail leaked into pure decode.");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    private static void AssertBitIdentical(float[] expected, ReadOnlySpan<float> actual, int step)
    {
        Assert.Equal(expected.Length, actual.Length);
        int diffs = 0, firstIdx = -1;
        float firstE = 0, firstA = 0;
        for (int k = 0; k < expected.Length; k++)
        {
            if (BitConverter.SingleToInt32Bits(expected[k]) != BitConverter.SingleToInt32Bits(actual[k]))
            {
                if (firstIdx < 0) { firstIdx = k; firstE = expected[k]; firstA = actual[k]; }
                diffs++;
            }
        }
        Assert.True(diffs == 0,
            $"CUDA graph decode diverged from direct launch at step {step}: {diffs}/{expected.Length} " +
            $"logits differ; first at idx {firstIdx} (direct={firstE:R} graph={firstA:R}). " +
            "Graph replay must be bit-identical to direct launches.");
    }
}
