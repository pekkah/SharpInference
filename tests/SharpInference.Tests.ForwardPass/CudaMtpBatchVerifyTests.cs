using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CUDA-hybrid coverage for the k-token MTP batched verify (issues #30 / #207
/// goal 4) on the qwen35 27B-MTP model:
/// <list type="bullet">
///   <item>pass-level: <see cref="CudaHybridGdnForwardPass.BatchVerify"/> per-position
///         logits vs k sequential <c>Forward</c> calls (argmax + maxAbs — the
///         BatchForward2 precision class: the CPU mmap FFN layers run MatVec2In in
///         the batch vs MatVecDual sequentially, so bit-equality is not expected);</item>
///   <item>rollback: verify junk drafts, <see cref="CudaHybridGdnForwardPass.RestoreBatchSnapshot"/>
///         to an intermediate position, and confirm the continued trajectory matches
///         the pure-sequential one — this exercises the DEVICE GDN snapshot ring
///         (pre-#207 the GPU-GDN reject path restored stale host state and the
///         rejected draft's rank-1 update stayed baked into the recurrence);</item>
///   <item>e2e: <see cref="MtpDecoder"/> batched greedy decode is coherent and the
///         chained drafts actually get accepted.</item>
/// </list>
/// Skipped silently when CUDA is unavailable or the 27B-MTP GGUF isn't on disk.
/// </summary>
public sealed class CudaMtpBatchVerifyTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindMtpModelPath()
    {
        string[] absoluteCandidates =
        {
            @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "Qwen3.6-27B-MTP-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Constructs the pass with a 4-token snapshot ring (the production default is
    /// 2 — the measured k=2 optimum — but these tests exercise k=4 batches).
    /// SHARPI_MTP_BATCH_MAX is instance-resolved at construction, so the env scope
    /// only needs to cover the ctor.
    /// </summary>
    private static CudaHybridGdnForwardPass CreatePass(GgufModel model, CudaBackend gpu,
                                                       ModelHyperparams hp)
    {
        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers,
            CpuLayers: 0,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));
        var prev = Environment.GetEnvironmentVariable("SHARPI_MTP_BATCH_MAX");
        Environment.SetEnvironmentVariable("SHARPI_MTP_BATCH_MAX", "4");
        try
        {
            return new CudaHybridGdnForwardPass(model, gpu, hp, placement);
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

    [Fact]
    public void BatchVerify_MatchesSequentialForward_PerPosition()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.NumMtpLayers > 0);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = CreatePass(model, gpu, hp);
        Assert.True(fwd.HasMtpHead);
        Assert.True(fwd.SupportsBatchVerify,
            "27B-MTP without SnapKV must support batched verify (GDN ring must have allocated).");
        Assert.True(fwd.MaxBatchVerifyTokens >= 4,
            $"Default ring should allow ≥4-token batches; got {fwd.MaxBatchVerifyTokens}.");

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
        {
            Assert.Equal(ArgMax(seqLogits[i]), ArgMax(batch[i]));
            float maxAbs = MaxAbsDiff(seqLogits[i], batch[i]);
            Assert.True(maxAbs < 0.25f,
                $"BatchVerify logits at position {P + i} diverge from sequential Forward " +
                $"(maxAbs={maxAbs:F4}) — beyond the MatVec2In-vs-MatVecDual noise envelope; " +
                "suspect a position/state mismatch in the batched trunk.");
        }
    }

    [Fact]
    public void BatchVerify_Rollback_RestoresDeviceGdnState()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = CreatePass(model, gpu, hp);
        if (!fwd.SupportsBatchVerify || fwd.MaxBatchVerifyTokens < 4) return;

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

        // Fresh state → verify g0 + three JUNK drafts, roll back to P+1 (only g0
        // kept), then replay the true continuation sequentially. If the device GDN
        // ring restore is broken, the junk tokens' rank-1 recurrence updates stay
        // baked in and the replayed logits drift far beyond kernel noise.
        fwd.ResetCache();
        _ = fwd.Prefill(prompt);
        int junk = (g1 + 7) % hp.VocabSize;
        var batch = fwd.BatchVerify([g0, junk, junk, junk], P);
        Assert.Equal(ArgMax(l1), ArgMax(batch[0]));

        fwd.RestoreBatchSnapshot(P + 1);
        var r2 = fwd.Forward(g1, P + 1).ToArray();
        Assert.Equal(ArgMax(l2), ArgMax(r2));
        float d2 = MaxAbsDiff(l2, r2);
        Assert.True(d2 < 0.25f,
            $"Post-rollback Forward at P+1 diverges from the sequential trajectory " +
            $"(maxAbs={d2:F4}) — the GDN snapshot ring did not restore the device state.");

        var r3 = fwd.Forward(g2, P + 2).ToArray();
        Assert.Equal(ArgMax(l3), ArgMax(r3));
        float d3 = MaxAbsDiff(l3, r3);
        Assert.True(d3 < 0.25f,
            $"Second post-rollback Forward diverges (maxAbs={d3:F4}); residual junk-draft " +
            "contamination in the GDN recurrence.");
    }

    [Fact]
    public void MtpDecoder_BatchedGreedy_CoherentWithAcceptedDrafts()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = CreatePass(model, gpu, hp);
        if (!fwd.SupportsBatchVerify || fwd.MaxBatchVerifyTokens < 4) return;

        var prompt = tokenizer.Encode(
            "Write a Python function that sorts a list using the quicksort algorithm:").ToArray();
        var logits = fwd.Prefill(prompt);

        var decoder = new MtpDecoder(fwd);
        decoder.Initialize(prompt.Length, logits);
        fwd.PrefillMtp(prompt);

        var produced = new List<int>(24);
        int[] stops = tokenizer.EogTokenIds.ToArray();
        decoder.Decode(24, stops, produced.Add, pMin: 1f, draftN: 3);

        Assert.True(produced.Count >= 8,
            $"Batched MTP decode stopped after {produced.Count} tokens — unexpectedly early EOS.");
        Assert.True(produced.Distinct().Count() >= 2,
            $"Degenerate decode: [{string.Join(",", produced)}]");
        // Chained drafting must actually land accepts (the 27B head accepts 95-100%
        // at depth 1; depth-3 chains compound but anything below ~30% means the
        // chain/self-hidden wiring is broken even if output stays correct).
        Assert.True(decoder.TotalDraftsEmitted > 0);
        Assert.True(decoder.AcceptanceRate >= 0.3f,
            $"Chained-draft acceptance {decoder.AcceptanceRate:P0} " +
            $"({decoder.TotalDraftsAccepted}/{decoder.TotalDraftsEmitted}) is far below the " +
            "depth-1 reference (95-100%); MtpLastHidden chaining or the MTP KV refresh is off.");
    }
}
