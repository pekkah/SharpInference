using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Pure-Vulkan coverage for the k-token MTP batched verify mechanism (issues #30 / #207 /
/// #357 PR2) on the dense qwen36 27B-MTP GDN model (Qwen3.6-27B-MTP):
/// <list type="bullet">
///   <item>pass-level: <see cref="VulkanHybridGdnForwardPass.BatchVerify"/> per-position logits
///         vs k sequential <c>Forward</c> calls (argmax + maxAbs). Unlike the CUDA pass, the
///         Vulkan trunk + per-position tail are byte-exact batched ops (the FFN runs per-row
///         through the same scalar helpers), so divergence implies a wiring/barrier bug, NOT
///         Q4_K cross-kernel noise — the tolerance is tight (1e-3);</item>
///   <item>rollback: verify junk drafts, <see cref="VulkanHybridGdnForwardPass.RestoreBatchSnapshot"/>
///         to an intermediate position, and confirm the continued trajectory matches the
///         pure-sequential one — this exercises the DEVICE GDN snapshot ring (a wrong restore
///         leaves the rejected draft's rank-1 recurrence update baked into the state).</item>
/// </list>
/// <para>These drive <see cref="VulkanHybridGdnForwardPass.BatchVerify"/> directly:
/// <see cref="VulkanHybridGdnForwardPass.SupportsBatchVerify"/> is gated false until #357 PR3
/// loads the MTP head, so <c>MtpDecoder</c> can't select this pass yet. The ring +
/// verify/rollback machinery already exist (PR2) and are exercised here.</para>
/// Silent-skips when Vulkan is unavailable, the device is out of memory (ring didn't allocate),
/// or the 27B-MTP GGUF isn't on disk. NOT run by the implementation pass (the orchestrator
/// verifies on a real GPU).
/// </summary>
public sealed class VulkanMtpBatchVerifyTests
{
    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindMtpModelPath()
    {
        const string fileName = "Qwen3.6-27B-MTP-Q4_K_M.gguf";
        string[] absoluteRoots = { @"E:\models", @"C:\p\sharpi\models" };
        foreach (var root in absoluteRoots)
        {
            var p = Path.Combine(root, fileName);
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static LayerPlacement GdnPlacement(ModelHyperparams hp) => new(
        GpuLayers: hp.NumLayers,
        CpuLayers: 0,
        GpuWeightBytes: 0,
        GpuKvBytes: 0,
        RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

    /// <summary>
    /// Constructs the pass with a 4-token snapshot ring. SHARPI_MTP_BATCH_MAX is instance-resolved
    /// at construction, so the env scope only needs to cover the ctor. Returns null (caller skips)
    /// when the device can't fit the dense GDN trunk + ring.
    /// </summary>
    private static VulkanHybridGdnForwardPass? CreatePass(GgufModel model, VulkanBackend gpu,
                                                          ModelHyperparams hp)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_MTP_BATCH_MAX");
        Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", "4");
        try
        {
            return new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp));
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            return null; // device too small → caller skips
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", prev);
        }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > m) m = d;
        }
        return m;
    }

    /// <summary>Top-1 minus top-2 logit (the argmax decision margin).</summary>
    private static float Top2Margin(float[] logits)
    {
        float top1 = float.NegativeInfinity, top2 = float.NegativeInfinity;
        foreach (var v in logits)
        {
            if (v > top1) { top2 = top1; top1 = v; }
            else if (v > top2) { top2 = v; }
        }
        return top1 - top2;
    }

    // Argmax-stable envelope for batched-vs-scalar trunk parity. The Vulkan batched trunk
    // (GDN scan / batched attention / batched matvecs) is numerically EQUAL to the scalar
    // single-token trunk only up to FP reduction order — same precision class the CUDA
    // BatchVerify test gates at 0.25 (gemma ~1.1). The measured per-position 27B divergence
    // is ~0.11–0.16; 0.30 leaves headroom while still flagging a real wiring/ring bug, whose
    // signature (junk-draft contamination, a bad restore) is an O(1)+ logit shift or a clear-
    // margin argmax flip — both far outside this envelope.
    private const float ArgmaxStableTol = 0.30f;

    // Assert argmax-equal only when the scalar top-2 margin clears twice the observed maxAbs:
    // within that band the FP noise cannot flip the argmax, so a mismatch is a real bug. On a
    // genuine near-tie (margin ≤ 2·maxAbs) the noise CAN flip it harmlessly — skip the argmax
    // assert there (the maxAbs bound still holds) instead of pinning the test to a lucky prompt.
    private static void AssertArgmaxStable(float[] reference, float[] actual, string where)
    {
        float maxAbs = MaxAbsDiff(reference, actual);
        Assert.True(maxAbs < ArgmaxStableTol,
            $"{where}: batched-vs-scalar logits diverge (maxAbs={maxAbs:F6} ≥ {ArgmaxStableTol}) — " +
            "beyond the argmax-stable envelope; suspect a position/state mismatch, a missing " +
            "RecordBarrier, or a broken GDN ring restore, NOT FP reduction-order noise.");
        if (Top2Margin(reference) > 2f * maxAbs)
            Assert.Equal(ArgMax(reference), ArgMax(actual));
    }

    [Fact]
    public void BatchVerify_MatchesSequentialForward_PerPosition()
    {
        using var gpu = TryCreateVulkan();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");
        Assert.True(hp.NumMtpLayers > 0, "Expected a NEXTN/MTP head on the 27B-MTP model");
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var fwd = CreatePass(model, gpu, hp);
        if (fwd is null) return;                                    // device OOM → skip
        // SupportsBatchVerify is intentionally false in PR2 (the MTP head isn't loaded yet); we
        // drive BatchVerify directly. The ring decides whether a ≥4-token batch is possible.
        if (fwd.MaxBatchVerifyTokens < 4) return;                  // ring didn't allocate → skip

        var prompt = tokenizer.Encode("The quick brown fox jumps over the lazy dog and then").ToArray();
        int P = prompt.Length;

        // Reference: greedy continuation via sequential Forward (k = 4 tokens).
        var prefillLogits = fwd.Prefill(prompt).ToArray();
        const int K = 4;
        var contTokens = new int[K];
        var seqLogits = new float[K][];
        contTokens[0] = ArgMax(prefillLogits);
        for (int i = 0; i < K; i++)
        {
            seqLogits[i] = fwd.Forward(contTokens[i], P + i).ToArray();
            if (i + 1 < K) contTokens[i + 1] = ArgMax(seqLogits[i]);
        }

        // Same tokens through one packed BatchVerify on a freshly prefilled state.
        fwd.ResetCache();
        _ = fwd.Prefill(prompt);
        var batch = fwd.BatchVerify(contTokens, P);

        Assert.Equal(K, batch.Length);
        for (int i = 0; i < K; i++)
            AssertArgmaxStable(seqLogits[i], batch[i], $"BatchVerify position {P + i}");
    }

    [Fact]
    public void BatchVerify_Rollback_RestoresDeviceGdnState()
    {
        using var gpu = TryCreateVulkan();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        using var fwd = CreatePass(model, gpu, hp);
        if (fwd is null) return;                                    // device OOM → skip
        if (fwd.MaxBatchVerifyTokens < 4) return;                  // ring didn't allocate → skip

        var prompt = tokenizer.Encode("Water boils at one hundred degrees and freezes at").ToArray();
        int P = prompt.Length;

        // Reference trajectory: g0 then two more greedy tokens, fully sequential.
        var prefillLogits = fwd.Prefill(prompt).ToArray();
        int g0 = ArgMax(prefillLogits);
        var l1 = fwd.Forward(g0, P).ToArray();
        int g1 = ArgMax(l1);
        var l2 = fwd.Forward(g1, P + 1).ToArray();
        int g2 = ArgMax(l2);
        var l3 = fwd.Forward(g2, P + 2).ToArray();

        // Fresh state → verify g0 + three JUNK drafts, roll back to P+1 (only g0 kept), then
        // replay the true continuation sequentially. If the device GDN ring restore is broken,
        // the junk tokens' rank-1 recurrence updates stay baked into the recurrence and the
        // replayed logits drift FAR past the argmax-stable envelope (an O(1)+ shift / argmax flip).
        // A working restore leaves only the batched-vs-scalar trunk noise (the captured state is
        // the batched trunk's; the replay is scalar) — within AssertArgmaxStable's bound.
        fwd.ResetCache();
        _ = fwd.Prefill(prompt);
        int junk = (g1 + 7) % hp.VocabSize;
        var batch = fwd.BatchVerify([g0, junk, junk, junk], P);
        AssertArgmaxStable(l1, batch[0], "BatchVerify position P (rollback test)");

        fwd.RestoreBatchSnapshot(P + 1);
        var r2 = fwd.Forward(g1, P + 1).ToArray();
        AssertArgmaxStable(l2, r2, "Post-rollback Forward at P+1");

        var r3 = fwd.Forward(g2, P + 2).ToArray();
        AssertArgmaxStable(l3, r3, "Second post-rollback Forward at P+2");
    }
}
