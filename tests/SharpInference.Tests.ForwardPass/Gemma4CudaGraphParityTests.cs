using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Bit-parity oracle for CUDA Graph decode (issue #136). The captured-and-replayed
/// Gemma 4 decode region runs the identical kernels in the identical order as the
/// direct-launch path — only the launch mechanism differs — so graph-on vs graph-off
/// logits must be <b>bit-identical</b> at every decode step, not merely argmax-equal.
///
/// Silent-skip pattern: no-ops when CUDA isn't available or the GGUF isn't on disk,
/// matching the sibling Gemma4Cuda* test files.
/// </summary>
public sealed class Gemma4CudaGraphParityTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{ModelFile}",
            $@"C:\p\sharpi\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
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

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    [Fact]
    public void Gemma4_E4B_CudaGraph_AllGpu_BitMatchesDirectLaunch()
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 }; // "The capital of France is"
        const int NSteps = 8;

        // Reference: graph OFF (direct launches). Capture full logits per decode step.
        var refLogits = new float[NSteps][];
        var refTokens = new int[NSteps];
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512) { UseCudaGraph = false };
            var logits = fwd.Prefill(tokens);
            refLogits[0] = logits.ToArray();
            refTokens[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                var step = fwd.Forward(refTokens[i - 1], pos++);
                refLogits[i] = step.ToArray();
                refTokens[i] = Argmax(step);
            }
        }

        // Candidate: graph ON (capture on first decode token, replay after).
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512) { UseCudaGraph = true };
            var logits = fwd.Prefill(tokens);
            AssertBitIdentical(refLogits[0], logits, step: 0);
            int pos = tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                // Drive the decode with the REFERENCE tokens so both runs see the same
                // input sequence even if (hypothetically) an early step diverged.
                var step = fwd.Forward(refTokens[i - 1], pos++);
                AssertBitIdentical(refLogits[i], step, step: i);
            }

            // Guard against a silent fallback: the graph path must actually have engaged.
            Assert.True(gpu.GraphReady,
                "CUDA graph was never captured — the parity test silently ran direct launches " +
                "on both sides and proves nothing. Check TryRunGemma4DeviceRegionViaGraph gating.");
        }
    }

    [Fact]
    public void Gemma4_E4B_CudaGraph_Hybrid_BitMatchesDirectLaunch()
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 };
        const int NSteps = 8;
        const int SafeGpuLayers = 22; // matches the bench -g 22 split; passes the KV-share guard

        LayerPlacement Placement() => new(
            GpuLayers: SafeGpuLayers,
            CpuLayers: hp.NumLayers - SafeGpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 512);

        // Reference: graph OFF.
        var refLogits = new float[NSteps][];
        var refTokens = new int[NSteps];
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaHybridForwardPass(model, gpu, hp, Placement()) { UseCudaGraph = false };
            var logits = fwd.Prefill(tokens);
            refLogits[0] = logits.ToArray();
            refTokens[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                var step = fwd.Forward(refTokens[i - 1], pos++);
                refLogits[i] = step.ToArray();
                refTokens[i] = Argmax(step);
            }
        }

        // Candidate: graph ON.
        using (var gpu = TryCreate())
        {
            if (gpu is null) return;
            using var fwd = new CudaHybridForwardPass(model, gpu, hp, Placement()) { UseCudaGraph = true };
            var logits = fwd.Prefill(tokens);
            AssertBitIdentical(refLogits[0], logits, step: 0);
            int pos = tokens.Length;
            for (int i = 1; i < NSteps; i++)
            {
                var step = fwd.Forward(refTokens[i - 1], pos++);
                AssertBitIdentical(refLogits[i], step, step: i);
            }

            Assert.True(gpu.GraphReady,
                "CUDA graph was never captured for the hybrid GPU layer loop — the parity test " +
                "silently ran direct launches on both sides. Check TryRunGpuLayersGemma4ViaGraph gating.");
        }
    }

    [Fact]
    public void Gemma4_E4B_CudaGraph_AllGpu_SnapKvConfiguredNoEvict_BitMatches()
    {
        // Regression guard: a configured SHARPI_SNAPKV_BUDGET must not break Gemma 4 graph
        // decode. SnapKV is force-disabled for Gemma-4-style models (SWA ring caches +
        // per-layer head_dim can't be SnapKV-compacted), so the cache fills sequentially
        // and graphs MUST engage and match bit-for-bit. (Even on a full-attention model
        // where SnapKV stays enabled, a configured-but-unevicted budget keeps
        // _kvEvictedCount == 0, so the same invariant holds.)
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "512"); // configured, but prompt << 512
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            Assert.NotNull(hp.LayerHeadDim);

            int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
            var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 }; // 6 tokens — no eviction
            const int NSteps = 8;

            var refLogits = new float[NSteps][];
            var refTokens = new int[NSteps];
            using (var gpu = TryCreate())
            {
                if (gpu is null) return;
                using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024) { UseCudaGraph = false };
                var logits = fwd.Prefill(tokens);
                refLogits[0] = logits.ToArray();
                refTokens[0] = Argmax(logits);
                int pos = tokens.Length;
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
                using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024) { UseCudaGraph = true };
                var logits = fwd.Prefill(tokens);
                AssertBitIdentical(refLogits[0], logits, step: 0);
                int pos = tokens.Length;
                for (int i = 1; i < NSteps; i++)
                    AssertBitIdentical(refLogits[i], fwd.Forward(refTokens[i - 1], pos++), step: i);

                Assert.True(gpu.GraphReady,
                    "Graphs must engage when a SnapKV budget is set but no eviction occurs " +
                    "(SnapKV is disabled for Gemma 4, or the prompt fits the budget). " +
                    "GraphReady=false means the _kvEvictedCount gate is over-broad.");
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
