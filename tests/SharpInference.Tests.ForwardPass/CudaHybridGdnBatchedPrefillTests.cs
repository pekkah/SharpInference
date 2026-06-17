using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #110: byte-parity guard for batched prompt prefill on the CPU-MoE
/// GDN-hybrid CUDA path. The batched path (<see cref="CudaHybridGdnForwardPass"/>
/// per-layer prefill, routed experts grouped by selection) must produce
/// bit-identical prefill logits and bit-identical MTP draft logits to the
/// sequential per-token loop — the same DispatchDot/DispatchDotQ8K kernels run
/// with identical per-token top-k accumulation order. A divergence here means
/// the batching reordered a floating-point reduction (the failure mode the MTP
/// greedy-parity oracles trip on).
///
/// Targets the Carnice APEX MTP model (Q3_K + Q8_0 routed experts → exercises
/// the Q8_KS-prepacked batched path and the MTP hidden-history population).
/// Skipped silently when CUDA is unavailable or the model isn't on disk.
/// </summary>
public sealed class CudaHybridGdnBatchedPrefillTests : IDisposable
{
    // Issue #162: these oracles assert the BATCHED prefill is bit-identical to the
    // sequential loop — they validate batching's reduction order, not the matmul kernel
    // choice. The compute-bound prefill path (Q8_0/Q4_K int8 MMQ, Q6_K/Q5_K dequant→fp16
    // GEMM) is argmax-stable, NOT byte-exact, so pin it OFF for the whole class to keep
    // the batched side on the byte-exact GEMM-N matvec. The MMQ/GEMM kernels' correctness
    // is covered separately by CudaMmqQ4K/Q8_0Tests + CudaGemmQ6K/Q5KTests. Restored in Dispose.
    private readonly bool _prevGdnCompute = CudaHybridGdnForwardPass.GdnPrefillComputeEnabled;
    public CudaHybridGdnBatchedPrefillTests() => CudaHybridGdnForwardPass.GdnPrefillComputeEnabled = false;
    public void Dispose() => CudaHybridGdnForwardPass.GdnPrefillComputeEnabled = _prevGdnCompute;

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindCarnicePath()
    {
        string[] candidates =
        {
            @"E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
            @"E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        };
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
    }

    private static string? FirstExisting(params string[] candidates)
    {
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
    }

    // Dense GDN-hybrid (no MoE routing) — issue #119 path 1.
    private static string? FindDense27BMtpPath() => FirstExisting(
        @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf");

    // GDN-hybrid MoE model with Q4_K experts (GPU-SLRU friendly) — issue #119 path 2.
    private static string? FindGpuSlruMoePath() => FirstExisting(
        @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        @"E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf");

    [Fact]
    public void BatchedPrefill_BitwiseMatchesSequential_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return; // batched path only applies to CPU MoE
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0,
                GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

            // A long-enough prompt that many experts collide across tokens, so the
            // grouped-by-expert path is actually exercised (N >= 2 triggers it).
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly.");
            Assert.True(tokens.Count >= 8, $"Prompt tokenized to only {tokens.Count} tokens.");

            // ── Sequential reference ──────────────────────────────────────────
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = false;
            float[] seqLogits;
            float[]? seqMtp = null;
            using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            {
                seqLogits = fwd.Prefill(tokens).ToArray();
                if (fwd.HasMtpHead)
                {
                    fwd.PrefillMtp(tokens);
                    int t1 = Sampler.Greedy(seqLogits);
                    seqMtp = fwd.MtpForward(t1, tokens.Count,
                        new ReadOnlySpan<float>(/* prev hidden */ GetLastHidden(fwd))).ToArray();
                }
            }

            // ── Batched ───────────────────────────────────────────────────────
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            float[] batLogits;
            float[]? batMtp = null;
            using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            {
                batLogits = fwd.Prefill(tokens).ToArray();
                if (fwd.HasMtpHead)
                {
                    fwd.PrefillMtp(tokens);
                    int t1 = Sampler.Greedy(batLogits);
                    batMtp = fwd.MtpForward(t1, tokens.Count,
                        new ReadOnlySpan<float>(GetLastHidden(fwd))).ToArray();
                }
            }

            Assert.Equal(seqLogits.Length, batLogits.Length);
            int firstDiff = -1;
            for (int i = 0; i < seqLogits.Length; i++)
                if (BitConverter.SingleToInt32Bits(seqLogits[i]) != BitConverter.SingleToInt32Bits(batLogits[i]))
                { firstDiff = i; break; }

            Assert.True(firstDiff < 0,
                $"Batched prefill logits diverge from sequential at index {firstDiff}: " +
                $"seq={(firstDiff >= 0 ? seqLogits[firstDiff] : 0)} bat={(firstDiff >= 0 ? batLogits[firstDiff] : 0)}. " +
                "Batched prefill must be bit-identical to the sequential per-token loop " +
                "(see the K/V MatVecDual MTP-parity regression note).");

            Assert.Equal(Sampler.Greedy(seqLogits), Sampler.Greedy(batLogits));

            if (seqMtp is not null && batMtp is not null)
            {
                int mtpDiff = -1;
                for (int i = 0; i < seqMtp.Length; i++)
                    if (BitConverter.SingleToInt32Bits(seqMtp[i]) != BitConverter.SingleToInt32Bits(batMtp[i]))
                    { mtpDiff = i; break; }
                Assert.True(mtpDiff < 0,
                    $"Batched-prefill MTP draft logits diverge from sequential at index {mtpDiff}. " +
                    "The MTP hidden-history (PrefillMtp reads _mtpPrefillHiddens) must be " +
                    "populated identically under batching.");
            }
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    [Fact]
    public void BatchVerifyCpuMoe_BitwiseMatchesPerToken_Carnice()
    {
        // Issue #210: BatchVerify's routed-expert FFN, grouped by selected expert
        // (BatchVerifyCpuMoe), must be bit-identical to the per-token CpuMoeFfnCore
        // loop it replaces. Toggle ONLY SHARPI_MTP_BATCHED_MOE_VERIFY between the two
        // runs — the trunk, shared expert, and router are shared, so any logit
        // divergence is the grouped-routed-expert reduction reorder (the greedy-parity
        // failure mode). Greedy verify (pMin=1) relies on this exactness to keep the
        // accept/reject decision identical to a sequential decode.
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatchedMoeVerify = CudaHybridGdnForwardPass.BatchedMoeVerifyEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return; // grouped routed-expert path is CPU-MoE only
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0,
                GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

            // A prompt long enough that adjacent draft positions collide on experts,
            // so the grouped-by-expert path actually amortizes a shared read.
            var prompt = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs.");
            Assert.True(prompt.Count >= 4, $"Prompt tokenized to only {prompt.Count} tokens.");

            // Run an identical prefill + k-token verify batch under each toggle.
            float[][] RunVerify(bool batchedMoeVerify)
            {
                CudaHybridGdnForwardPass.BatchedMoeVerifyEnabled = batchedMoeVerify;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                if (!fwd.SupportsBatchVerify) return Array.Empty<float[]>();
                var pf = fwd.Prefill(prompt).ToArray();
                int k = Math.Min(4, fwd.MaxBatchVerifyTokens);
                if (k < 2) return Array.Empty<float[]>();
                // Verify computes logits at every position regardless of acceptance;
                // only the token identities must match across the two runs. Token 0 is
                // the greedy continuation; the rest are a fixed arbitrary chain.
                var batch = new int[k];
                batch[0] = Sampler.Greedy(pf);
                for (int i = 1; i < k; i++)
                    batch[i] = (batch[i - 1] * 131 + 7) % hp.VocabSize;
                return fwd.BatchVerify(batch, prompt.Count);
            }

            var perTok = RunVerify(false);
            var batched = RunVerify(true);
            if (perTok.Length == 0 || batched.Length == 0) return; // verify unsupported here

            Assert.Equal(perTok.Length, batched.Length);
            for (int t = 0; t < perTok.Length; t++)
            {
                Assert.Equal(perTok[t].Length, batched[t].Length);
                int firstDiff = -1;
                for (int i = 0; i < perTok[t].Length; i++)
                    if (BitConverter.SingleToInt32Bits(perTok[t][i])
                        != BitConverter.SingleToInt32Bits(batched[t][i]))
                    { firstDiff = i; break; }
                Assert.True(firstDiff < 0,
                    $"Batched-MoE verify logits diverge from per-token at position {t}, " +
                    $"index {firstDiff}: seq={(firstDiff >= 0 ? perTok[t][firstDiff] : 0)} " +
                    $"bat={(firstDiff >= 0 ? batched[t][firstDiff] : 0)}. BatchVerifyCpuMoe must " +
                    "be bit-identical to the per-token CpuMoeFfnCore loop.");
            }

            // Greedy parity at every position — the issue's acceptance contract.
            for (int t = 0; t < perTok.Length; t++)
                Assert.Equal(Sampler.Greedy(perTok[t]), Sampler.Greedy(batched[t]));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedMoeVerifyEnabled = prevBatchedMoeVerify;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    // The MTP head needs the pre-output-norm hidden of the last prompt token as
    // prevHidden. After Prefill, that is exposed via LastHidden.
    private static float[] GetLastHidden(CudaHybridGdnForwardPass fwd) =>
        fwd.LastHidden.ToArray();

    /// <summary>Repeat a varied seed until the tokenizer emits ≥ <paramref name="approx"/> tokens.</summary>
    private static List<int> LongPrompt(GgufTokenizer tokenizer, int approx)
    {
        const string seed =
            "The quick brown fox jumps over the lazy dog. " +
            "Sphinx of black quartz, judge my vow. " +
            "Pack my box with five dozen liquor jugs. " +
            "How razorback-jumping frogs can level six piqued gymnasts. ";
        var sb = new System.Text.StringBuilder();
        while (true)
        {
            sb.Append(seed);
            var attempt = tokenizer.Encode(sb.ToString());
            if (attempt.Count >= approx) return attempt.ToList();
            if (sb.Length > 200_000)
                throw new InvalidOperationException("Tokenizer not packing enough tokens.");
        }
    }

    /// <summary>
    /// Issue #118: the wave-based >4096 batched-query SDPA must be bit-identical to the
    /// per-position attention loop. Prefills a &gt;4096-token prompt (so the chunk exits
    /// the 4096 shared-scores window and takes the wave path) under batched prefill +
    /// batched trunk, toggling only <c>BatchedAttnEnabled</c> (true = wave-based batched
    /// SDPA; false = per-position global-scratch loop, the bit-exact reference). A small
    /// wave budget on the batched arm also forces the multi-wave loop. This is the
    /// model-level counterpart to <see cref="CudaGdnBatchedTrunkTests.AttentionBatchedWave_F32_BitwiseMatchesSequential"/>,
    /// exercising the AttnBlockBatched host wiring at &gt;4096.
    /// </summary>
    [Fact]
    public void BatchedAttnWave_BitwiseMatchesPerPosition_Over4096_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatchedPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevBatchedTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevAttn = CudaHybridGdnForwardPass.BatchedAttnEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            if (hp.ContextLength < 4400) return; // need room past the 4096 window
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, 4200);
            Assert.True(tokens.Count > 4096, $"Prompt only {tokens.Count} tokens; need >4096.");
            int ctx = Math.Min(hp.ContextLength, tokens.Count + 64);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: ctx);

            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;

            float[] RunWith(bool batchedAttn, string? budgetMb)
            {
                CudaHybridGdnForwardPass.BatchedAttnEnabled = batchedAttn;
                Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", budgetMb);
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                return fwd.Prefill(tokens).ToArray();
            }

            float[] perPos = RunWith(false, null);  // per-position global-scratch loop (reference)
            float[] wave   = RunWith(true, "8");     // wave-based SDPA, forced multi-wave (8 MiB)

            Assert.Equal(perPos.Length, wave.Length);
            int firstDiff = -1;
            for (int i = 0; i < perPos.Length; i++)
                if (BitConverter.SingleToInt32Bits(perPos[i]) != BitConverter.SingleToInt32Bits(wave[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Wave-based >4096 SDPA diverges from the per-position path at index {firstDiff} " +
                $"(N={tokens.Count}). AttentionBatchedWave host wiring must be bit-identical to the per-position loop.");
            Assert.Equal(Sampler.Greedy(perPos), Sampler.Greedy(wave));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatchedPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevBatchedTrunk;
            CudaHybridGdnForwardPass.BatchedAttnEnabled = prevAttn;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
            Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", prevBudget);
        }
    }

    /// <summary>
    /// Issue #111: the GEMM-batched trunk (<c>TrunkLayerBatched</c>, default) must be
    /// bit-identical to the per-token <c>TrunkLayerSequential</c> fallback
    /// (<c>SHARPI_BATCHED_TRUNK=0</c>). Both run under batched prefill; only the trunk
    /// strategy differs. This is the equivalence users rely on when bisecting with the
    /// env var — otherwise asserted only by prose.
    /// </summary>
    [Fact]
    public void BatchedTrunk_BitwiseMatchesSequentialTrunk_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatchedPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevBatchedTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts!");
            Assert.True(tokens.Count >= 8);

            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;

            float[] RunWithTrunk(bool batchedTrunk)
            {
                CudaHybridGdnForwardPass.BatchedTrunkEnabled = batchedTrunk;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                return fwd.Prefill(tokens).ToArray();
            }

            float[] seqTrunk = RunWithTrunk(false);
            float[] batTrunk = RunWithTrunk(true);

            Assert.Equal(seqTrunk.Length, batTrunk.Length);
            int firstDiff = -1;
            for (int i = 0; i < seqTrunk.Length; i++)
                if (BitConverter.SingleToInt32Bits(seqTrunk[i]) != BitConverter.SingleToInt32Bits(batTrunk[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Batched trunk diverges from sequential trunk at index {firstDiff}. " +
                "TrunkLayerBatched must be bit-identical to TrunkLayerSequential (SHARPI_BATCHED_TRUNK=0).");
            Assert.Equal(Sampler.Greedy(seqTrunk), Sampler.Greedy(batTrunk));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatchedPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevBatchedTrunk;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    /// <summary>
    /// Issue #114-B: the batched GDN trunk's <b>fused sequential-scan recurrence</b>
    /// (<c>BatchedGdnScanEnabled</c>) and <b>batched-query SDPA</b>
    /// (<c>BatchedAttnEnabled</c>), both default-on, must be bit-identical to the
    /// per-position View-loop fallback (the path taken when those flags are off — the
    /// pre-#114-B reference). This is the only oracle that exercises the host glue in
    /// <c>GdnBlockBatched</c> / <c>AttnBlockBatched</c> — the conv→silu→L2norm→tile→scan
    /// strides/offsets and the KV-append/SDPA wiring — at the model level; the per-kernel
    /// tests (<see cref="CudaGdnBatchedTrunkTests"/>) prove each kernel in isolation but
    /// not the argument plumbing that strings them together. Both arms run under batched
    /// prefill + batched trunk (#110/#111); only the #114-B sub-paths differ.
    /// </summary>
    [Fact]
    public void BatchedGdnScanAndAttn_BitwiseMatchesPerPosition_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatchedPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevBatchedTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevScan = CudaHybridGdnForwardPass.BatchedGdnScanEnabled;
        bool prevAttn = CudaHybridGdnForwardPass.BatchedAttnEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts!");
            Assert.True(tokens.Count >= 8);

            // Hold the #110/#111 batched paths on; only toggle the #114-B sub-paths.
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;

            float[] RunWith(bool fused)
            {
                CudaHybridGdnForwardPass.BatchedGdnScanEnabled = fused;
                CudaHybridGdnForwardPass.BatchedAttnEnabled = fused;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                return fwd.Prefill(tokens).ToArray();
            }

            float[] perPos = RunWith(false);  // per-position recurrence + SDPA (View loops)
            float[] fused = RunWith(true);     // fused sequential-scan + batched-query SDPA

            Assert.Equal(perPos.Length, fused.Length);
            int firstDiff = -1;
            for (int i = 0; i < perPos.Length; i++)
                if (BitConverter.SingleToInt32Bits(perPos[i]) != BitConverter.SingleToInt32Bits(fused[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Fused GDN-scan + batched-query SDPA diverges from the per-position path at index {firstDiff}. " +
                "GdnBlockBatched/AttnBlockBatched host wiring (strides/offsets) must be bit-identical to the View-loop fallback.");
            Assert.Equal(Sampler.Greedy(perPos), Sampler.Greedy(fused));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatchedPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevBatchedTrunk;
            CudaHybridGdnForwardPass.BatchedGdnScanEnabled = prevScan;
            CudaHybridGdnForwardPass.BatchedAttnEnabled = prevAttn;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    /// <summary>
    /// Issue #121: the GEMM-N-batched dense FFN (<c>BatchedGpuDenseFfn</c>, default) must be
    /// bit-identical to the per-token FFN fallback (<c>SHARPI_BATCHED_FFN=0</c>). Both arms
    /// run under batched prefill + batched trunk; only the FFN strategy differs. This is the
    /// FFN counterpart to <see cref="BatchedTrunk_BitwiseMatchesSequentialTrunk_Carnice"/> —
    /// the equivalence a user relies on when bisecting a parity regression with the env var,
    /// otherwise asserted only by prose. (The GPU-SLRU grouped-MoE FFN equivalence is covered
    /// by <see cref="BatchedTrunkGpuFfn_BitwiseMatchesSequential_GpuSlruMoe"/>, which pins both
    /// batched flags on.)
    /// </summary>
    [Fact]
    public void BatchedFfn_BitwiseMatchesPerTokenFfn_Dense27BMtp()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindDense27BMtpPath();
        if (path is null) return;

        bool prevPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevFfn = CudaHybridGdnForwardPass.BatchedFfnEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (hp.IsMoE) return; // dense-only path
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts!");
            Assert.True(tokens.Count >= 8);

            // Hold prefill + trunk batching on; toggle ONLY the FFN strategy.
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;

            float[] RunWithFfn(bool batchedFfn)
            {
                CudaHybridGdnForwardPass.BatchedFfnEnabled = batchedFfn;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                return fwd.Prefill(tokens).ToArray();
            }

            float[] perTokenFfn = RunWithFfn(false);
            float[] batchedFfn  = RunWithFfn(true);

            Assert.Equal(perTokenFfn.Length, batchedFfn.Length);
            int firstDiff = -1;
            for (int i = 0; i < perTokenFfn.Length; i++)
                if (BitConverter.SingleToInt32Bits(perTokenFfn[i]) != BitConverter.SingleToInt32Bits(batchedFfn[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Batched dense FFN diverges from the per-token FFN at index {firstDiff}. " +
                "BatchedGpuDenseFfn must be bit-identical to the per-token GpuDenseFfn loop (SHARPI_BATCHED_FFN=0).");
            Assert.Equal(Sampler.Greedy(perTokenFfn), Sampler.Greedy(batchedFfn));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevTrunk;
            CudaHybridGdnForwardPass.BatchedFfnEnabled = prevFfn;
        }
    }

    /// <summary>
    /// Issue #119: batched-trunk prefill for the <b>dense</b> GDN-hybrid path
    /// (<c>!hp.IsMoE</c>, Qwen3.6-27B-MTP) — <c>_cpuMoe == false</c>, FFN runs per token
    /// on GPU/CPU. The batched path (<c>PrefillBatchedTrunkGpuFfn</c>) must be bit-identical
    /// to the sequential per-token <c>Forward</c> loop (<c>SHARPI_BATCHED_PREFILL=0</c>) for
    /// both the final-token logits and the MTP draft logits (the trunk batching and MTP
    /// hidden-history population must not reorder any FP reduction).
    /// </summary>
    [Fact]
    public void BatchedTrunkGpuFfn_BitwiseMatchesSequential_Dense27BMtp()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindDense27BMtpPath();
        if (path is null) return;

        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (hp.IsMoE) return; // dense-only path
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts!");
            Assert.True(tokens.Count >= 8);

            float[] seqLogits; float[]? seqMtp = null;
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = false;
            using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            {
                seqLogits = fwd.Prefill(tokens).ToArray();
                if (fwd.HasMtpHead)
                {
                    fwd.PrefillMtp(tokens);
                    int t1 = Sampler.Greedy(seqLogits);
                    seqMtp = fwd.MtpForward(t1, tokens.Count, new ReadOnlySpan<float>(GetLastHidden(fwd))).ToArray();
                }
            }

            float[] batLogits; float[]? batMtp = null;
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = true;
            using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            {
                batLogits = fwd.Prefill(tokens).ToArray();
                if (fwd.HasMtpHead)
                {
                    fwd.PrefillMtp(tokens);
                    int t1 = Sampler.Greedy(batLogits);
                    batMtp = fwd.MtpForward(t1, tokens.Count, new ReadOnlySpan<float>(GetLastHidden(fwd))).ToArray();
                }
            }

            Assert.Equal(seqLogits.Length, batLogits.Length);
            int firstDiff = -1;
            for (int i = 0; i < seqLogits.Length; i++)
                if (BitConverter.SingleToInt32Bits(seqLogits[i]) != BitConverter.SingleToInt32Bits(batLogits[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Dense batched-trunk prefill diverges from sequential at index {firstDiff}. " +
                "PrefillBatchedTrunkGpuFfn must be bit-identical to the sequential Forward loop.");
            Assert.Equal(Sampler.Greedy(seqLogits), Sampler.Greedy(batLogits));

            if (seqMtp is not null && batMtp is not null)
            {
                int mtpDiff = -1;
                for (int i = 0; i < seqMtp.Length; i++)
                    if (BitConverter.SingleToInt32Bits(seqMtp[i]) != BitConverter.SingleToInt32Bits(batMtp[i]))
                    { mtpDiff = i; break; }
                Assert.True(mtpDiff < 0,
                    $"Dense batched-trunk MTP draft logits diverge from sequential at index {mtpDiff}.");
            }
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
        }
    }

    /// <summary>
    /// Issue #119 (coverage): the dense <c>PrefillBatchedTrunkGpuFfn</c> path across two
    /// chunks (<c>[0,k)</c> then <c>[k,N)</c> with <c>startPos=k</c>). Exercises the paths
    /// the single-chunk dense test misses — <c>startPos &gt; 0</c>, cross-chunk KV/GDN
    /// continuity, and the exact-size <c>EnsureStreamAll</c> reallocation between two chunks
    /// of different length (the realloc that historically produced the UploadInto
    /// element-count mismatch). Final-token logits must be bit-identical to the sequential
    /// per-token <c>Forward</c> loop.
    /// </summary>
    [Fact]
    public void BatchedTrunkGpuFfn_MultiChunk_BitwiseMatchesSequential_Dense27BMtp()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindDense27BMtpPath();
        if (path is null) return;

        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.");
            int N = tokens.Count;
            Assert.True(N >= 12, $"Prompt tokenized to only {N} tokens.");
            int k1 = N / 3;

            float[] RunTwoChunks(bool batched)
            {
                CudaHybridGdnForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                fwd.Prefill(tokens.Take(k1).ToList(), 0);
                return fwd.Prefill(tokens.Skip(k1).ToList(), k1).ToArray();
            }

            float[] seq = RunTwoChunks(false);
            float[] bat = RunTwoChunks(true);

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(bat[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Dense multi-chunk batched-trunk prefill diverges from sequential at index {firstDiff} " +
                $"(N={N}, split at {k1}). Cross-chunk KV/GDN continuity or the startPos>0 path " +
                "(EnsureStreamAll realloc) is not bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
        }
    }

    /// <summary>
    /// Issue #119 + #122 (coverage): SnapKV-active prefill on the dense
    /// <c>PrefillBatchedTrunkGpuFfn</c> path. SnapKV (issue #58) engages the trailing-window
    /// Q-capture inside <c>AttnBlockBatched</c>. Post-#122 (with <c>BatchedAttnEnabled</c> on,
    /// the default), the batched arm captures Q in one batched <c>CopyDeviceRegion</c> from
    /// <c>qAll</c> and runs the batched SDPA — so this oracle now exercises the #122 batched
    /// capture path (before #122 the SnapKV-active branch fell back to the per-position loop).
    /// With a long-enough prompt and a small <c>SHARPI_SNAPKV_BUDGET</c> the batched path must
    /// still return final-token logits bit-identical to the sequential per-token <c>Forward</c>
    /// loop (both capture Q and evict identically; sequential <c>Forward</c> is the
    /// deterministic reference). This is the dense ≤4096 counterpart to
    /// <see cref="SnapKvActive_BatchedWave_BitwiseMatchesSequential_Over4096_Carnice"/>.
    /// </summary>
    [Fact]
    public void BatchedTrunkGpuFfn_SnapKvActive_BitwiseMatchesSequential_Dense27BMtp()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindDense27BMtpPath();
        if (path is null) return;

        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "128");
        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
            // Prompt comfortably longer than the 128-token budget so SnapKV engages.
            var tokens = LongPrompt(tokenizer, 400);
            Assert.True(tokens.Count > 128);

            float[] Run(bool batched)
            {
                CudaHybridGdnForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                var logits = fwd.Prefill(tokens).ToArray();
                // Cache shrinks to exactly the budget post-eviction → confirms SnapKV
                // actually engaged (otherwise the test would be vacuous).
                Assert.Equal(128, fwd.Cache.Length);
                return logits;
            }

            float[] seq = Run(false);
            float[] bat = Run(true);

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(bat[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"SnapKV-active dense batched-trunk prefill diverges from sequential at index {firstDiff}. " +
                "The per-position Q-capture path under batched prefill must be bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
        }
    }

    /// <summary>
    /// Issue #122 (&gt;4096): the wave-based SDPA (issue #118) must run with SnapKV active.
    /// Before #122, SnapKV forced the per-position KV-append + <c>Attention</c> loop (the
    /// only place the trailing-window Q was captured into <c>_snapKvQCapture</c>), so a
    /// SnapKV-active &gt;4096 prefill got neither the wave SDPA nor batching. Now
    /// <c>AttnBlockBatched</c> does a single batched Q-capture from <c>qAll</c> first, then
    /// runs the wave SDPA. This pins the batched-prefill path (wave SDPA + batched capture,
    /// <c>SHARPI_BATCHED_PREFILL=1</c>) against the sequential per-token <c>Forward</c> loop
    /// (<c>=0</c>) — the deterministic reference — and asserts the final-token logits are
    /// <b>bit-identical</b>.
    /// <para>
    /// Why bit-identical (not just greedy-token): the sequential <c>Forward</c> reference is
    /// run-to-run deterministic under SnapKV (verified — two sequential runs are bit-equal),
    /// the batched Q-capture copies the same contiguous <c>qAll</c> rows <c>[wStart,N)</c> the
    /// per-position loop copies, and <c>AttentionBatchedWave</c> is already proven bit-identical
    /// to the per-position SDPA by the non-SnapKV wave oracle — so the captured Q, the
    /// eviction keep-set, and the post-eviction logits all match exactly. (The
    /// <c>BatchedAttnEnabled=false</c> per-position fallback under batched prefill is itself
    /// nondeterministic and is deliberately NOT used as the reference here.) A small wave
    /// budget on the batched arm forces the multi-wave loop. Asserts SnapKV engaged
    /// (<c>Cache.Length == 128</c>) on both arms so the batched Q-capture actually ran.
    /// </para>
    /// </summary>
    [Fact]
    public void SnapKvActive_BatchedWave_BitwiseMatchesSequential_Over4096_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB");
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "128");
        bool prevBatchedPrefill = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        bool prevBatchedTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevAttn = CudaHybridGdnForwardPass.BatchedAttnEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            if (hp.ContextLength < 4400) return; // need room past the 4096 window
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, 4200);
            Assert.True(tokens.Count > 4096, $"Prompt only {tokens.Count} tokens; need >4096.");
            int ctx = Math.Min(hp.ContextLength, tokens.Count + 64);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: ctx);

            CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;
            CudaHybridGdnForwardPass.BatchedAttnEnabled = true;

            float[] Run(bool batchedPrefill, string? budgetMb)
            {
                CudaHybridGdnForwardPass.BatchedPrefillEnabled = batchedPrefill;
                Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", budgetMb);
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                var logits = fwd.Prefill(tokens).ToArray();
                // Confirms SnapKV engaged on both arms → the batched Q-capture ran.
                Assert.Equal(128, fwd.Cache.Length);
                return logits;
            }

            float[] seq  = Run(false, null);  // sequential per-token Forward (deterministic reference)
            float[] wave = Run(true, "8");      // #122 batched capture + wave SDPA (forced multi-wave)

            Assert.Equal(seq.Length, wave.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(wave[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"SnapKV-active wave batched prefill diverges from sequential at index {firstDiff} " +
                $"(N={tokens.Count}). The batched Q-capture + AttentionBatchedWave must be bit-identical " +
                "to the sequential per-token Forward loop's per-position capture + SDPA.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(wave));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatchedPrefill;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevBatchedTrunk;
            CudaHybridGdnForwardPass.BatchedAttnEnabled = prevAttn;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
            Environment.SetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB", prevBudget);
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
        }
    }

    /// <summary>
    /// Issue #119: batched-trunk prefill for the <b>GPU-SLRU MoE</b> path — a GDN-hybrid
    /// MoE model forced onto the on-GPU routed-expert path with <c>SHARPI_CPU_MOE=0</c>
    /// (<c>_cpuMoe == false</c>; routed experts run via <c>GpuMoeFfn</c> per token). The
    /// batched path must be bit-identical to the sequential per-token <c>Forward</c> loop:
    /// the trunk batching only collapses the GDN/attn launches; the SLRU FFN runs the same
    /// single-token kernel sequence and loads identical expert weights regardless of cache
    /// access order. Large model — skipped if not on disk or if construction fails (VRAM).
    /// </summary>
    [Fact]
    public void BatchedTrunkGpuFfn_BitwiseMatchesSequential_GpuSlruMoe()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindGpuSlruMoePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "0"); // force GPU-SLRU MoE
        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        // The Batched*Enabled flags are static and mutated by sibling tests; this oracle's
        // batched arm must run the #121 grouped-by-expert BatchedGpuMoeFfn (which requires
        // both trunk + FFN batching on), not a per-token fallback left behind by another
        // test. Pin them on explicitly and restore in finally so the parity check isn't vacuous.
        bool prevTrunk = CudaHybridGdnForwardPass.BatchedTrunkEnabled;
        bool prevFfn = CudaHybridGdnForwardPass.BatchedFfnEnabled;
        CudaHybridGdnForwardPass.BatchedTrunkEnabled = true;
        CudaHybridGdnForwardPass.BatchedFfnEnabled = true;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));
            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs.");
            Assert.True(tokens.Count >= 8);

            float[] RunOrNull(bool batched)
            {
                CudaHybridGdnForwardPass.BatchedPrefillEnabled = batched;
                CudaHybridGdnForwardPass fwd;
                try
                {
                    // Only construction is allowed to skip (a box without the VRAM to host
                    // this 22 GB model under GPU-SLRU, or a config this class can't load).
                    // A failure inside Prefill itself must FAIL the test, not silently skip —
                    // that is exactly the kind of regression this oracle exists to catch.
                    fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                }
                catch (NotSupportedException) { return Array.Empty<float>(); }
                catch (InvalidOperationException) { return Array.Empty<float>(); } // OOM / VRAM at construction
                using (fwd)
                {
                    if (fwd.IsMoeOnCpu) return Array.Empty<float>(); // not the SLRU path on this box
                    return fwd.Prefill(tokens).ToArray();
                }
            }

            float[] seq = RunOrNull(false);
            float[] bat = RunOrNull(true);
            if (seq.Length == 0 || bat.Length == 0) return; // couldn't exercise SLRU here

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(bat[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"GPU-SLRU batched-trunk prefill diverges from sequential at index {firstDiff}. " +
                "PrefillBatchedTrunkGpuFfn must be bit-identical to the sequential Forward loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
            CudaHybridGdnForwardPass.BatchedTrunkEnabled = prevTrunk;
            CudaHybridGdnForwardPass.BatchedFfnEnabled = prevFfn;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }

    /// <summary>
    /// Multi-chunk parity: prefill the prompt in two segments (<c>[0,k)</c> then
    /// <c>[k,N)</c>, the second with <c>startPos=k</c>) and assert the final-token
    /// logits are bit-identical between batched and sequential. Exercises the paths
    /// the single-chunk test misses: <c>startPos &gt; 0</c>, cross-chunk KV/GDN
    /// continuity, and the exact-size <c>_gpuStreamAll</c> reallocation between two
    /// chunks of different length (the bug that produced the UploadInto element-count
    /// mismatch during development).
    /// </summary>
    [Fact]
    public void BatchedPrefill_MultiChunk_BitwiseMatchesSequential_Carnice()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindCarnicePath();
        if (path is null) return;

        var prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        bool prevBatched = CudaHybridGdnForwardPass.BatchedPrefillEnabled;
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            if (!hp.IsMoE) return;
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0,
                GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

            var tokens = tokenizer.Encode(
                "The quick brown fox jumps over the lazy dog. " +
                "Pack my box with five dozen liquor jugs. " +
                "How razorback-jumping frogs can level six piqued gymnasts! " +
                "The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow.");
            int N = tokens.Count;
            Assert.True(N >= 12, $"Prompt tokenized to only {N} tokens.");
            // Split into two unequal chunks so the second has startPos>0 and the
            // device stream buffer must resize from k1 to k2 tokens.
            int k1 = N / 3;

            float[] RunTwoChunks(bool batched)
            {
                CudaHybridGdnForwardPass.BatchedPrefillEnabled = batched;
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
                fwd.Prefill(tokens.Take(k1).ToList(), 0);
                return fwd.Prefill(tokens.Skip(k1).ToList(), k1).ToArray();
            }

            float[] seq = RunTwoChunks(false);
            float[] bat = RunTwoChunks(true);

            Assert.Equal(seq.Length, bat.Length);
            int firstDiff = -1;
            for (int i = 0; i < seq.Length; i++)
                if (BitConverter.SingleToInt32Bits(seq[i]) != BitConverter.SingleToInt32Bits(bat[i]))
                { firstDiff = i; break; }
            Assert.True(firstDiff < 0,
                $"Multi-chunk batched prefill diverges from sequential at index {firstDiff} " +
                $"(N={N}, split at {k1}). Cross-chunk KV/GDN continuity or the startPos>0 " +
                "path is not bit-identical to the sequential loop.");
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(bat));
        }
        finally
        {
            CudaHybridGdnForwardPass.BatchedPrefillEnabled = prevBatched;
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }
}
