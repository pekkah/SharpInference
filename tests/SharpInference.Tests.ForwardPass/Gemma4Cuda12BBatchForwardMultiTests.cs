using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #283 — closes the #276 synthetic-vs-real gap for the Gemma 4 12B-global
/// <c>attention_k_eq_v</c> batched decode. #276 validated the k_eq_v branch
/// (<c>GpuLayerBatchedDecodeGemma4</c>: <c>if (kEqV) CopyDevice(vAll, kAll)</c>, V reuses the raw K
/// projection) only on a tiny <b>synthetic</b> all-global F32 fixture
/// (<see cref="Gemma4CudaKEqVBatchedDecodeTests"/>), because the only local 12B was Q4_0 (not
/// GEMM-N-batchable → single-user fallback) and a Q8_0 12B (~13 GB) wouldn't fit the 4070 Ti's 12 GB.
/// With a GEMM-N-batchable <b>Q4_K_M 12B</b> (<c>gemma-4-12b-it-Q4_K_M.gguf</c>, ~7.4 GB) on disk,
/// these oracles drive the REALISTIC 12B pairing — k_eq_v on the global layers + real <c>attn_v</c>
/// on the SWA layers, in ONE model — through the batched serving path and compare it to the trusted
/// single-user <c>ForwardGemma4</c> loop on the same model.
///
/// <para>Mirrors <see cref="Gemma4CudaBatchForwardMultiTests"/> (E4B Q8_0) but on the 12B: per-layer
/// KV heads (8 GQA on SWA / 1 MQA on global), the k_eq_v global layers (V = raw K projection + pure
/// V-norm), the SWA/global split and softcaps; the 12B has NO PLE
/// (<c>embedding_length_per_layer_input = 0</c>). The Q4_K trunk weights are SoA-repacked at load
/// (dense path), so the N≥5 batched step exercises the #201/#206 int8 decode-MMQ tile end-to-end on
/// the k_eq_v layers — correctness coverage that complements the throughput bench
/// (<see cref="Gemma4CudaBatchedDecodeBench"/>, #283). Argmax-stable, not bit-exact: the batched WS
/// matvec / int8 decode-MMQ round differently from the fp32 single-user matvec, exactly the contract
/// the E4B oracles hold. KV pinned to fp32 + SnapKV pinned off so the path is deterministic and
/// continuous-batching-eligible. One ~7.4 GB instance per test; silent-skips without CUDA / the GGUF.
/// </para>
/// </summary>
public sealed class Gemma4Cuda12BBatchForwardMultiTests
{
    private const string ModelFile = "gemma-4-12b-it-Q4_K_M.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // SnapKV pinned off (an active budget makes continuous batching unsupported / machine-dependent)
    // and KV pinned to fp32 so the batched-vs-single comparison is deterministic across cards (since
    // #185 the 12B auto-narrows the KV dtype when fp32 won't fit — at ctx 512 it stays fp32 anyway,
    // but pin it so a larger-VRAM box doesn't silently change the path). Both restored after the ctor
    // reads them (the KV dtype + SnapKV budget are fixed at construction).
    private static CudaForwardPass NewFwd(GgufModel model, CudaBackend gpu, ModelHyperparams hp, int ctx = 512)
    {
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        var prevKv = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "fp32");
        try { return new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevKv);
        }
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

    // Argmax parity tolerant of a precision-driven near-tie (the batched WS matvec / int8 decode-MMQ
    // round differently from the single-user fp32 matvec, so a top-2 reference near-tie can flip) —
    // accepted ONLY when provably a near-tie in the reference.
    private static void AssertArgmaxOrNearTie(float[] reference, float[] candidate, float tieEps, string label)
    {
        int rArg = Argmax(reference), cArg = Argmax(candidate);
        if (rArg == cArg) return;
        float gap = MathF.Abs(reference[rArg] - reference[cArg]);
        Assert.True(gap < tieEps,
            $"{label}: batched argmax {cArg} != single-user {rArg}, NOT a near-tie (reference gap {gap:F3} ≥ {tieEps:F1}) " +
            "— a real k_eq_v / per-layer-KV / SWA wiring divergence, not WS/decode-MMQ rounding.");
    }

    // Distinct BOS-led mid-vocab prompts; arbitrary tokens (the oracle compares batched vs single-user
    // on the SAME model, so a meaningful prompt is unnecessary — only the per-sequence trajectory needs
    // to differ). Lengths vary so the batched positions[] are non-uniform.
    private static int[][] MakePrompts(int bos) =>
    [
        [bos, 818, 5279, 529, 7001, 563, 1234, 4567],
        [bos, 1079, 4108, 1614, 13, 222, 333, 444, 8901],
        [bos, 651, 6037, 576, 6081, 603, 99, 7777, 12, 34],
        [bos, 2024, 11, 512, 9000, 71, 6, 4242, 88],
        [bos, 314, 159, 265, 358, 979, 32, 384, 626, 433, 8],
        [bos, 700, 5005, 31, 2718, 281, 1828, 1, 9, 17, 5],
    ];

    /// <summary>
    /// Precondition + the gap-closer: the 12B is a Gemma-4 model that supports continuous batching
    /// (#195), AND it carries the realistic <c>k_eq_v-on-global + real-V-on-SWA</c> pairing in ONE
    /// model — at least one global layer omits <c>attn_v</c> (the k_eq_v branch fires) and at least
    /// one SWA layer keeps it (the real-V branch). The synthetic #276 fixture is all-global, so this
    /// is the arrangement it could not reproduce.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q4KM_SupportsContinuousBatching_WithKEqVOnGlobalAndRealVOnSwa()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.AttentionKEqV, "expected attention_k_eq_v=true for the 12B model.");
        Assert.NotNull(hp.LayerKvHeads);   // per-layer KV heads (8 GQA SWA / 1 MQA global)
        Assert.NotNull(hp.IsSwaLayer);     // SWA/global split

        int globalKEqV = 0, swaRealV = 0;
        for (int i = 0; i < hp.NumLayers; i++)
        {
            bool isSwa = hp.IsSwaLayer![i];
            bool hasV = model.FindTensor($"blk.{i}.attn_v.weight") is not null;
            // The batched decode's kEqV predicate is `AttentionKEqV && !isSwa && _wv[layer] is null`.
            if (!isSwa && !hasV) globalKEqV++;
            if (isSwa && hasV) swaRealV++;
        }
        Assert.True(globalKEqV > 0,
            $"expected ≥1 global k_eq_v layer (attn_v absent) — the realistic 12B pairing; found {globalKEqV}.");
        Assert.True(swaRealV > 0,
            $"expected ≥1 SWA layer with a real attn_v — the realistic 12B pairing; found {swaRealV}.");

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsContinuousBatching,
            "Q4_K_M 12B (GEMM-N-batchable, dense Gemma-4) should support continuous batching (#195/#283).");
    }

    /// <summary>
    /// <see cref="CudaForwardPass.PrefillWithCache"/> into a per-sequence cache must reproduce the
    /// single-user <see cref="CudaForwardPass.Prefill"/> into the owned cache (identical kernels, only
    /// cache pointers differ → near bit-identical) — including the global k_eq_v layers' V=rawK copy.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q4KM_PrefillWithCache_MatchesSingleUserPrefill()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int bos = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int[] prompt = MakePrompts(bos)[0];

        fwd.ResetCache();
        float[] reference = fwd.Prefill(prompt).ToArray();

        using var cache = fwd.CreateCache();
        float[] viaCache = fwd.PrefillWithCache(prompt, cache).ToArray();

        Assert.Equal(reference.Length, viaCache.Length);
        Assert.Equal(Argmax(reference), Argmax(viaCache));
        Assert.Equal(prompt.Length, cache.Length);
        float maxAbs = MaxAbs(reference, viaCache);
        Assert.True(maxAbs < 1e-2f,
            $"12B PrefillWithCache (per-sequence cache) diverged from single-user Prefill: maxAbs={maxAbs}.");
    }

    /// <summary>
    /// Headline #283 oracle: prefill two prompts into per-sequence caches, then one batched
    /// <see cref="CudaForwardPass.BatchForwardMulti"/> decode step must reproduce two independent
    /// single-user prefill+decode passes (argmax-stable within the WS-matvec tolerance). N=2 stays on
    /// the WS matvec (decode-MMQ engages only at N≥5), so this is the tight-tolerance k_eq_v check.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q4KM_BatchForwardMulti_N2_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int bos = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var prompts = MakePrompts(bos);
        int[] promptA = prompts[0], promptB = prompts[1];

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
    /// Two batched decode steps (positions advance) must track the single-user continuation — catches
    /// a k_eq_v KV-append / SWA-ring / position-indexing bug a single step would miss (the first
    /// step's K-as-V is reused, a second token appended at the new position). Compared to the
    /// single-user step at the same position only while the greedy trajectory still matches.
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q4KM_BatchForwardMulti_TwoSteps_MatchSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        int bos = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var prompts = MakePrompts(bos);
        int[] promptA = prompts[0], promptB = prompts[1];

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

    /// <summary>
    /// N=6 batched decode (≥5) — engages the #201/#206 int8 decode-MMQ tile for the big Q4_K trunk
    /// shapes (q/o-proj, gate/up, down, lm-head; rows≥2048, cols%256), so the global k_eq_v layers
    /// run end-to-end through the decode-MMQ path and must still match each sequence's single-user
    /// decode. This is the correctness companion to the throughput bench (<see
    /// cref="Gemma4CudaBatchedDecodeBench"/>). The int8 MMQ rounds more coarsely than the WS matvec
    /// (tracks fp32 to ~3% of RMS, see <see cref="CudaDecodeMmqTests"/>), so the tolerance is looser:
    /// argmax-or-near-tie + top-5 overlap are the load-bearing checks (a real k_eq_v wiring bug
    /// scrambles both); the maxAbs bound is a secondary net (the 12B softcaps logits to about +/-30,
    /// so it only catches a delta beyond 5 on that bounded scale, not unbounded divergence).
    /// </summary>
    [Fact]
    public void Gemma4_12B_Q4KM_BatchForwardMulti_N6_DecodeMmq_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsContinuousBatching);

        int bos = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var prompts = MakePrompts(bos);
        const int N = 6;
        Assert.Equal(N, prompts.Length);

        // Per-sequence single-user reference (greedy first token + one decode step).
        var refLogits = new float[N][];
        var firstTok = new int[N];
        for (int s = 0; s < N; s++)
        {
            fwd.ResetCache();
            firstTok[s] = Argmax(fwd.Prefill(prompts[s]));
            refLogits[s] = fwd.Forward(firstTok[s], prompts[s].Length).ToArray();
        }

        // Batched: per-sequence caches, one N=6 batched decode step (decode-MMQ for the big shapes).
        var caches = new CudaSequenceKvCache[N];
        try
        {
            var toks = new int[N];
            var poss = new int[N];
            for (int s = 0; s < N; s++)
            {
                caches[s] = fwd.CreateCache();
                fwd.PrefillWithCache(prompts[s], caches[s]);
                toks[s] = firstTok[s];
                poss[s] = prompts[s].Length;
            }

            float[][] batch = fwd.BatchForwardMulti(toks, poss, caches);
            Assert.Equal(N, batch.Length);

            for (int s = 0; s < N; s++)
            {
                AssertArgmaxOrNearTie(refLogits[s], batch[s], tieEps: 1.5f, $"Seq {s} (N=6 decode-MMQ)");
                Assert.True(Overlap(refLogits[s], batch[s], 5) >= 3,
                    $"Seq {s} (N=6 decode-MMQ) top-5 overlap < 3 (maxAbs={MaxAbs(refLogits[s], batch[s])}).");
                Assert.True(MaxAbs(refLogits[s], batch[s]) < 5.0f,
                    $"Seq {s} (N=6 decode-MMQ) maxAbs={MaxAbs(refLogits[s], batch[s])} exceeds the gross-divergence " +
                    "bound — a k_eq_v wiring bug, not int8 MMQ rounding.");
            }
        }
        finally
        {
            foreach (var c in caches) c?.Dispose();
        }
    }
}
