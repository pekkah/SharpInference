using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #195: CUDA continuous batching for Gemma 4. <see cref="CudaForwardPass"/> now accepts
/// Gemma-4 models on the batched-serving path — <see cref="CudaForwardPass.CreateCache"/> sizes
/// per-layer head_dim + SWA ring + shared-KV aliasing, and the batched decode
/// (<c>RunBatchedTrunkGemma4</c>) applies per-layer geometry, SWA-vs-global attention, k_eq_v,
/// the PLE pre-pass + injection, sandwich post-norms, layer_output_scale, and the final softcap.
///
/// These oracles validate the batched-serving entry points against the trusted single-user
/// <see cref="CudaForwardPass.Prefill"/> / <see cref="CudaForwardPass.Forward"/>
/// (<c>ForwardGemma4</c>) loop on <b>Gemma 4 E4B Q8_0</b> — which exercises per-layer head_dim
/// (256), SWA rings, the 18-layer shared-KV tail, and PLE. (The only local 12B is q4_0, whose
/// weights aren't GEMM-N-batchable, so k_eq_v's batched path isn't covered by a runnable test;
/// it mirrors the single-token <c>RunGemma4DeviceRegion</c> exactly.)
///
/// The batched decode routes its matmuls through the cuBLAS GEMM (fp16 weights/activations), so
/// it is argmax-stable, not bit-exact, vs the fp32 per-token loop — the same contract the Gemma 4
/// batched-prefill oracles hold. One ~8 GB instance per test; silent-skips when CUDA or the GGUF
/// is absent. Mirrors <see cref="CudaBatchForwardMultiTests"/>.
/// </summary>
public sealed class Gemma4CudaBatchForwardMultiTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // SnapKV pinned off: an active budget makes continuous batching unsupported (it throws), and
    // VRAM-scaled auto-SnapKV could otherwise engage and make these oracles machine-dependent.
    private static CudaForwardPass NewFwd(GgufModel model, CudaBackend gpu, ModelHyperparams hp, int ctx = 512)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev); }
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

    private static HashSet<int> TopKSet(ReadOnlySpan<float> logits, int k)
    {
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        var arr = logits.ToArray();
        Array.Sort(idx, (a, b) => arr[b].CompareTo(arr[a]));
        var set = new HashSet<int>();
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    private static int Overlap(float[] reference, float[] candidate, int k)
    {
        var r = TopKSet(reference, k);
        int o = 0;
        foreach (var t in TopKSet(candidate, k)) if (r.Contains(t)) o++;
        return o;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    // Argmax parity tolerant of a precision-driven near-tie (the cuBLAS GEMM rounds to fp16, so a
    // top-2 reference near-tie can flip) — accepted ONLY when provably a near-tie in the reference.
    private static void AssertArgmaxOrNearTie(float[] reference, float[] candidate, float tieEps, string label)
    {
        int rArg = Argmax(reference), cArg = Argmax(candidate);
        if (rArg == cArg) return;
        float gap = MathF.Abs(reference[rArg] - reference[cArg]);
        Assert.True(gap < tieEps,
            $"{label}: batched argmax {cArg} != single-user {rArg}, NOT a near-tie (reference gap {gap:F3} ≥ {tieEps:F1}) " +
            "— a real wiring divergence (per-layer geometry / SWA ring / shared-KV / PLE), not fp16 rounding.");
    }

    /// <summary>
    /// Precondition + gate: E4B is a Gemma-4 model (per-layer head_dim, PLE, shared-KV tail) and
    /// <see cref="CudaForwardPass.SupportsContinuousBatching"/> is now true for it (issue #195).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_SupportsContinuousBatching()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);          // per-layer head_dim
        Assert.True(hp.HasPerLayerTokenEmbd);     // PLE
        Assert.NotNull(hp.KvSourceLayer);         // shared-KV tail

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsContinuousBatching,
            "Gemma 4 E4B should support continuous batching after #195 (SnapKV pinned off).");
    }

    /// <summary>
    /// <see cref="CudaForwardPass.PrefillWithCache"/> into a per-sequence
    /// <see cref="CudaSequenceKvCache"/> must reproduce the single-user
    /// <see cref="CudaForwardPass.Prefill"/> into the owned cache. Validates that
    /// <see cref="CudaForwardPass.CreateCache"/> builds Gemma-4 per-layer geometry + SWA ring +
    /// shared-KV aliasing identically to the owned cache (identical kernels, only cache pointers
    /// differ → near bit-identical).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_PrefillWithCache_MatchesSingleUserPrefill()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int[] prompt = { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222 };

        fwd.ResetCache();
        float[] reference = fwd.Prefill(prompt).ToArray();

        using var cache = fwd.CreateCache();
        float[] viaCache = fwd.PrefillWithCache(prompt, cache).ToArray();

        Assert.Equal(reference.Length, viaCache.Length);
        Assert.Equal(Argmax(reference), Argmax(viaCache));
        Assert.Equal(prompt.Length, cache.Length);
        float maxAbs = MaxAbs(reference, viaCache);
        Assert.True(maxAbs < 1e-2f,
            $"Gemma 4 PrefillWithCache (per-sequence cache) diverged from single-user Prefill: maxAbs={maxAbs}.");
    }

    /// <summary>
    /// Headline #195 oracle: prefill two prompts into per-sequence caches, then one batched
    /// <see cref="CudaForwardPass.BatchForwardMulti"/> decode step must reproduce two independent
    /// single-user prefill+decode passes (argmax-stable within the fp16-GEMM tolerance).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_BatchForwardMulti_N2_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int[] promptA = { 2, 651, 6037, 576, 6081, 603, 1234 };
        int[] promptB = { 2, 1079, 4108, 1614, 13, 222, 333, 444 };

        // Single-user reference: prefill, greedy token, one decode step.
        fwd.ResetCache();
        int tokA = Argmax(fwd.Prefill(promptA));
        float[] refA = fwd.Forward(tokA, promptA.Length).ToArray();
        fwd.ResetCache();
        int tokB = Argmax(fwd.Prefill(promptB));
        float[] refB = fwd.Forward(tokB, promptB.Length).ToArray();

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(promptA, cacheA);
        fwd.PrefillWithCache(promptB, cacheB);

        float[][] batch = fwd.BatchForwardMulti(
            [tokA, tokB], [promptA.Length, promptB.Length], [cacheA, cacheB]);

        Assert.Equal(2, batch.Length);

        AssertArgmaxOrNearTie(refA, batch[0], tieEps: 1.0f, "Seq A");
        Assert.True(Overlap(refA, batch[0], 5) >= 4, $"Seq A top-5 overlap (maxAbs={MaxAbs(refA, batch[0])}).");
        Assert.True(MaxAbs(refA, batch[0]) < 1.0f, $"Seq A maxAbs={MaxAbs(refA, batch[0])}.");

        AssertArgmaxOrNearTie(refB, batch[1], tieEps: 1.0f, "Seq B");
        Assert.True(Overlap(refB, batch[1], 5) >= 4, $"Seq B top-5 overlap (maxAbs={MaxAbs(refB, batch[1])}).");
        Assert.True(MaxAbs(refB, batch[1]) < 1.0f, $"Seq B maxAbs={MaxAbs(refB, batch[1])}.");
    }

    /// <summary>
    /// Two batched decode steps (positions advance) must track the single-user continuation —
    /// catches a per-sequence KV-append / SWA-ring / position-indexing bug a single step would miss
    /// (the first step's KV is reused, a second token appended at the new position). The batched run
    /// follows its own greedy trajectory; each step's logits are compared to the single-user step at
    /// the same position only while the trajectory still matches (a near-tie flip ends the chain).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_BatchForwardMulti_TwoSteps_MatchSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int[] promptA = { 2, 651, 6037, 576, 6081, 603, 1234 };
        int[] promptB = { 2, 1079, 4108, 1614, 13, 222, 333, 444 };

        // Single-user reference: capture the step-1 and step-2 logits (and the greedy tokens) per
        // prompt, so each batched step can be compared to the right reference logits.
        fwd.ResetCache();
        int a0 = Argmax(fwd.Prefill(promptA));
        float[] refA1 = fwd.Forward(a0, promptA.Length).ToArray();
        int a1 = Argmax(refA1);
        float[] refA2 = fwd.Forward(a1, promptA.Length + 1).ToArray();
        fwd.ResetCache();
        int b0 = Argmax(fwd.Prefill(promptB));
        float[] refB1 = fwd.Forward(b0, promptB.Length).ToArray();
        int b1 = Argmax(refB1);
        float[] refB2 = fwd.Forward(b1, promptB.Length + 1).ToArray();

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(promptA, cacheA);
        fwd.PrefillWithCache(promptB, cacheB);

        var step1 = fwd.BatchForwardMulti([a0, b0], [promptA.Length, promptB.Length], [cacheA, cacheB]);
        AssertArgmaxOrNearTie(refA1, step1[0], tieEps: 1.0f, "Seq A step1");
        AssertArgmaxOrNearTie(refB1, step1[1], tieEps: 1.0f, "Seq B step1");
        Assert.True(MaxAbs(refA1, step1[0]) < 1.0f, $"Seq A step1 maxAbs={MaxAbs(refA1, step1[0])}.");
        Assert.True(MaxAbs(refB1, step1[1]) < 1.0f, $"Seq B step1 maxAbs={MaxAbs(refB1, step1[1])}.");
        int ba1 = Argmax(step1[0]);
        int bb1 = Argmax(step1[1]);

        var step2 = fwd.BatchForwardMulti(
            [ba1, bb1], [promptA.Length + 1, promptB.Length + 1], [cacheA, cacheB]);

        // Compare the 2nd step to the single-user 2nd step only where the trajectory tracked.
        if (ba1 == a1)
        {
            AssertArgmaxOrNearTie(refA2, step2[0], tieEps: 1.0f, "Seq A step2");
            Assert.True(MaxAbs(refA2, step2[0]) < 1.0f, $"Seq A step2 maxAbs={MaxAbs(refA2, step2[0])}.");
        }
        if (bb1 == b1)
        {
            AssertArgmaxOrNearTie(refB2, step2[1], tieEps: 1.0f, "Seq B step2");
            Assert.True(MaxAbs(refB2, step2[1]) < 1.0f, $"Seq B step2 maxAbs={MaxAbs(refB2, step2[1])}.");
        }
    }
}
