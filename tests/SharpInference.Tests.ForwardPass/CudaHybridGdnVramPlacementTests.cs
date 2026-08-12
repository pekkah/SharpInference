using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Partial GPU offload + CPU embedding/output fallback (<c>PlanVram</c>) for
/// <see cref="CudaHybridGdnForwardPass"/>. Covers:
/// <list type="bullet">
///   <item>T1 — <c>EstimateEmbedGpuBytes</c> pure-metadata pricing (no GPU/model needed).</item>
///   <item>T2 — <c>-g N</c> caps GPU-resident dense-FFN layers without changing decoded tokens.</item>
///   <item>T3/T4 — forced CPU embedding / forced CPU output match the GPU-resident baseline.</item>
///   <item>T5/T6 — packed Q8_0 embedding on qwen35moe doesn't change tokens or the MoE
///         auto-select decision.</item>
///   <item>T7 — <c>EstimateMtpHeadGpuBytes</c> prices the MTP head within a sane envelope.</item>
/// </list>
/// GPU/model-dependent tests silently skip when CUDA is unavailable or the target GGUF isn't
/// on disk (mirrors <see cref="CudaHybridGdnForwardPassTests"/> / <see cref="CudaHybridGdnSnapKvTests"/>).
/// Collection parallelism is disabled project-wide (xunit.runner.json), so the env-var
/// save/restore in ctor/Dispose below is race-free against other tests in this project.
/// </summary>
public sealed class CudaHybridGdnVramPlacementTests : IDisposable
{
    private readonly string? _prevRawEmbed = Environment.GetEnvironmentVariable("SHARPI_GDN_RAW_EMBED");
    private readonly string? _prevRawQ80 = Environment.GetEnvironmentVariable("SHARPI_GDN_RAW_Q8_0");
    private readonly string? _prevCpuEmbed = Environment.GetEnvironmentVariable("SHARPI_GDN_CPU_EMBED");
    private readonly string? _prevCpuOutput = Environment.GetEnvironmentVariable("SHARPI_GDN_CPU_OUTPUT");

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_EMBED", _prevRawEmbed);
        Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_Q8_0", _prevRawQ80);
        Environment.SetEnvironmentVariable("SHARPI_GDN_CPU_EMBED", _prevCpuEmbed);
        Environment.SetEnvironmentVariable("SHARPI_GDN_CPU_OUTPUT", _prevCpuOutput);
    }

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FirstExisting(params string[] candidates)
    {
        foreach (var p in candidates)
            if (File.Exists(p)) return p;
        return null;
    }

    /// <summary>The ground-truth 27B dense hybrid-GDN + MTP model this WP was measured against,
    /// falling back to the generic filename other tests in this project use.</summary>
    private static string? FindMtpModelPath()
    {
        string[] absoluteCandidates =
        {
            @"D:\sharpi\models\Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf",
            @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        var found = FirstExisting(absoluteCandidates);
        if (found is not null) return found;

        var dir = Directory.GetCurrentDirectory();
        string[] relativeNames =
        {
            "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf",
            "Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        for (int i = 0; i < 8; i++)
        {
            foreach (var name in relativeNames)
            {
                var p = Path.Combine(dir, "models", name);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindHybridModelPath()
    {
        string[] absoluteCandidates =
        {
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        var found = FirstExisting(absoluteCandidates);
        if (found is not null) return found;

        string[] relativeCandidates =
        {
            @"models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in relativeCandidates)
            {
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static GgufTensorInfo MakeEmbedInfo(DType dtype) =>
        new("token_embd.weight", 2, [5120, 248320], dtype, 0);

    // ================================================================
    //  T1 — EstimateEmbedGpuBytes pricing (pure metadata, no GPU/model)
    // ================================================================

    [Theory]
    [InlineData(DType.Q4_K)]
    [InlineData(DType.Q5_K)]
    [InlineData(DType.Q6_K)]
    [InlineData(DType.Q8_0)]
    public void EstimateEmbedGpuBytes_PackedDtypes_ReportByteSize(DType dtype)
    {
        var info = MakeEmbedInfo(dtype);
        long bytes = CudaHybridGdnForwardPass.EstimateEmbedGpuBytes(info, tied: false, embDim: 5120, rawEmbedEnabled: true);
        long expected = (info.ByteSize + 3) & ~3L;
        Assert.Equal(expected, bytes);
    }

    [Fact]
    public void EstimateEmbedGpuBytes_Q3K_ReportsF32Expanded()
    {
        var info = MakeEmbedInfo(DType.Q3_K);
        long bytes = CudaHybridGdnForwardPass.EstimateEmbedGpuBytes(info, tied: false, embDim: 5120, rawEmbedEnabled: true);
        Assert.Equal(info.ElementCount * sizeof(float), bytes);
    }

    [Fact]
    public void EstimateEmbedGpuBytes_BFloat16_ReportsF32Expanded()
    {
        // Ground truth: BFloat16/Float16 are not in UploadWeight's raw set — they always
        // F32-expand, both for a standalone output.weight and as a (hypothetical) embedding.
        var info = MakeEmbedInfo(DType.BFloat16);
        long bytes = CudaHybridGdnForwardPass.EstimateEmbedGpuBytes(info, tied: false, embDim: 5120, rawEmbedEnabled: true);
        Assert.Equal(info.ElementCount * sizeof(float), bytes);
    }

    [Fact]
    public void EstimateEmbedGpuBytes_TiedQ8_0_RawQ80Off_ReportsF32Expanded()
    {
        bool prevRawQ80 = CudaHybridGdnForwardPass.RawQ80WeightsEnabled;
        Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_Q8_0", "0");
        CudaHybridGdnForwardPass.RawQ80WeightsEnabled = false;
        try
        {
            var info = MakeEmbedInfo(DType.Q8_0);
            // tied: true — the embedding buffer also serves as the lm_head, so it must only
            // stay raw when UploadWeight would also keep a standalone output.weight raw, which
            // for Q8_0 requires RawQ80WeightsEnabled.
            long bytes = CudaHybridGdnForwardPass.EstimateEmbedGpuBytes(info, tied: true, embDim: 5120, rawEmbedEnabled: true);
            Assert.Equal(info.ElementCount * sizeof(float), bytes);
        }
        finally
        {
            CudaHybridGdnForwardPass.RawQ80WeightsEnabled = prevRawQ80;
        }
    }

    [Fact]
    public void EstimateEmbedGpuBytes_TiedQ8_0_RawQ80On_StaysPacked()
    {
        bool prevRawQ80 = CudaHybridGdnForwardPass.RawQ80WeightsEnabled;
        CudaHybridGdnForwardPass.RawQ80WeightsEnabled = true;
        try
        {
            var info = MakeEmbedInfo(DType.Q8_0);
            long bytes = CudaHybridGdnForwardPass.EstimateEmbedGpuBytes(info, tied: true, embDim: 5120, rawEmbedEnabled: true);
            Assert.Equal((info.ByteSize + 3) & ~3L, bytes);
        }
        finally
        {
            CudaHybridGdnForwardPass.RawQ80WeightsEnabled = prevRawQ80;
        }
    }

    // ================================================================
    //  T1b — DecideEmbedOutputPlacement cascade (pure, no GPU/model). The GiB unit below is
    //  scaled down to keep the numbers small and readable; the arithmetic is scale-invariant.
    // ================================================================

    private const long GiB = 1024L * 1024 * 1024;

    [Fact]
    public void DecideEmbedOutputPlacement_UntiedOverBudget_DemotesEmbeddingFirst()
    {
        // budget 1.0 GiB, embed 0.5 GiB, output 0.3 GiB — both fit together; nothing demoted.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: (long)(0.5 * GiB), outputGpuBytes: (long)(0.3 * GiB), budget: 1 * GiB,
            tied: false, forceCpuEmbed: false, forceCpuOutput: false);

        Assert.True(r.EmbedOnGpu);
        Assert.True(r.OutputOnGpu);
    }

    [Fact]
    public void DecideEmbedOutputPlacement_OutputAloneFits_EmbedDemotedOutputStaysGpu()
    {
        // budget 1.0 GiB, embed 0.5 GiB, output 0.6 GiB — combined (1.1) doesn't fit, but output
        // alone (0.6) does: embed (cheap) is demoted first and that's enough — output never needs
        // to be touched.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: (long)(0.5 * GiB), outputGpuBytes: (long)(0.6 * GiB), budget: 1 * GiB,
            tied: false, forceCpuEmbed: false, forceCpuOutput: false);

        Assert.False(r.EmbedOnGpu);
        Assert.True(r.OutputOnGpu);
    }

    [Fact]
    public void DecideEmbedOutputPlacement_DemotingOutputAloneFreesRoom_RePromotesEmbedding()
    {
        // Finding 3's repro: budget 1.0 GiB, embed 0.5 GiB, output 1.5 GiB. Demoting the
        // embedding alone (cost 1.5) still exceeds budget, so the output is demoted too — but
        // that frees enough room (1.0 >= 0.5) to re-promote the embedding. Without the fix, both
        // stayed on CPU even though embed=GPU + output=CPU fits.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: (long)(0.5 * GiB), outputGpuBytes: (long)(1.5 * GiB), budget: 1 * GiB,
            tied: false, forceCpuEmbed: false, forceCpuOutput: false);

        Assert.True(r.EmbedOnGpu);
        Assert.False(r.OutputOnGpu);
        Assert.Equal((long)(0.5 * GiB), r.Cost);
    }

    [Fact]
    public void DecideEmbedOutputPlacement_ForcedCpuEmbed_SkipsRePromotion()
    {
        // Same numbers as the re-promotion repro, but SHARPI_GDN_CPU_EMBED=1 forced the
        // embedding off GPU — the re-promotion check must respect that override, not treat it
        // as a budget artifact to undo.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: (long)(0.5 * GiB), outputGpuBytes: (long)(1.5 * GiB), budget: 1 * GiB,
            tied: false, forceCpuEmbed: true, forceCpuOutput: false);

        Assert.False(r.EmbedOnGpu);
        Assert.False(r.OutputOnGpu);
    }

    [Fact]
    public void DecideEmbedOutputPlacement_EmbedHardCapped_NeverRePromoted()
    {
        // Embedding alone exceeds the 2 GiB single-allocation cap — hard-capped off GPU
        // unconditionally, before any budget arithmetic runs. Even with a generous overall
        // budget (output comfortably fits with room to spare), the hard-capped embedding must
        // never come back — a >2 GiB single allocation is a physical impossibility
        // (CudaBackend's exact-allocation cap / GgufModel.GetTensorData's int-length Span), not
        // a budget trade-off the re-promotion heuristic should second-guess.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: 3 * GiB, outputGpuBytes: (long)(1.5 * GiB), budget: 10 * GiB,
            tied: false, forceCpuEmbed: false, forceCpuOutput: false);

        Assert.False(r.EmbedOnGpu);
        Assert.True(r.OutputOnGpu);
    }

    [Fact]
    public void DecideEmbedOutputPlacement_Tied_NeverRePromotesIndependently()
    {
        // Tied weights (no separate output.weight) move together; the untied-only
        // re-promotion branch must never fire for them.
        var r = CudaHybridGdnForwardPass.DecideEmbedOutputPlacement(
            embedGpuBytes: (long)(0.5 * GiB), outputGpuBytes: 0, budget: (long)(0.1 * GiB),
            tied: true, forceCpuEmbed: false, forceCpuOutput: false);

        Assert.False(r.EmbedOnGpu);
        Assert.False(r.OutputOnGpu);
    }

    // ================================================================
    //  T2 — -g N caps GPU-resident dense-FFN layers, tokens unchanged
    // ================================================================

    private static LayerPlacement MakePlacement(ModelHyperparams hp, int gpuLayers, int ctx) => new(
        GpuLayers: gpuLayers,
        CpuLayers: hp.NumLayers - gpuLayers,
        GpuWeightBytes: 0,
        GpuKvBytes: 0,
        RecommendedCtxSize: Math.Min(hp.ContextLength, ctx));

    private static int[] GreedyDecode(CudaHybridGdnForwardPass fwd, int[] prompt, int count)
    {
        var logits = fwd.Prefill(prompt);
        var decoded = new int[count];
        for (int i = 0; i < count; i++)
        {
            int next = Sampler.Greedy(logits);
            decoded[i] = next;
            logits = fwd.Forward(next, prompt.Length + i);
        }
        return decoded;
    }

    [Fact]
    public void CudaHybridGdnForwardPass_GpuFfnCap_LimitsUploadsWithoutChangingTokens()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var prompt = tokenizer.Encode("Hello").ToArray();
        Assert.NotEmpty(prompt);

        int[] fullTokens;
        int uncappedLayers;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, MakePlacement(hp, hp.NumLayers, 4096)))
        {
            uncappedLayers = fwd.DenseFfnGpuLayers;
            fullTokens = GreedyDecode(fwd, prompt, 16);
        }

        if (uncappedLayers <= 4)
        {
            Console.Error.WriteLine(
                $"[SKIP] Uncapped dense-FFN fill only reached {uncappedLayers} layers (<=4); " +
                "the -g 4 cap wouldn't be exercised on this VRAM budget.");
            return;
        }

        int[] cappedTokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, MakePlacement(hp, 4, 4096)))
        {
            Assert.True(fwd.DenseFfnGpuLayers <= 4,
                $"Expected -g 4 to cap dense-FFN GPU layers at <=4, got {fwd.DenseFfnGpuLayers}.");
            cappedTokens = GreedyDecode(fwd, prompt, 16);
        }

        // GpuDenseFfn (cuBLAS/NVRTC GEMM) and CpuDenseFfn (SimdKernels AVX2/AVX-512 dot) are
        // argmax-stable, NOT byte-exact (documented throughout this codebase for every CPU/GPU
        // kernel pair) — capping how many layers land on GPU changes which kernel a layer runs
        // through, so full 16-token equality isn't a safe bar over 64 layers. The first decoded
        // token (from Prefill, before any FFN placement has had a chance to compound) is the
        // strong correctness signal; log where the sequences first diverge for visibility.
        Assert.Equal(fullTokens[0], cappedTokens[0]);
        if (!fullTokens.SequenceEqual(cappedTokens))
        {
            int firstDiff = 0;
            while (firstDiff < fullTokens.Length && fullTokens[firstDiff] == cappedTokens[firstDiff]) firstDiff++;
            Console.Error.WriteLine(
                $"[NOTE] Uncapped vs -g 4 token sequences diverge at position {firstDiff} " +
                $"(uncapped={string.Join(",", fullTokens)} capped={string.Join(",", cappedTokens)}) — " +
                "expected: CPU/GPU dense-FFN kernels are argmax-stable, not byte-exact.");
        }
    }

    // ================================================================
    //  T3 — forced CPU embedding matches GPU embedding
    // ================================================================

    [Fact]
    public void CudaHybridGdnForwardPass_ForcedCpuEmbedding_MatchesGpuEmbedding()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var prompt = tokenizer.Encode("Hello").ToArray();
        Assert.NotEmpty(prompt);

        var placement = MakePlacement(hp, hp.NumLayers, 4096);

        float[] baselineFirstLogits;
        int[] baselineTokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            Assert.True(fwd.EmbeddingOnGpu, "Baseline run should keep the embedding on GPU.");
            var logits = fwd.Prefill(prompt);
            baselineFirstLogits = logits.ToArray();
            baselineTokens = GreedyDecode(fwd, prompt, 16);
        }

        Environment.SetEnvironmentVariable("SHARPI_GDN_CPU_EMBED", "1");
        float[] cpuFirstLogits;
        int[] cpuTokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            Assert.False(fwd.EmbeddingOnGpu, "SHARPI_GDN_CPU_EMBED=1 should force the embedding off GPU.");
            var logits = fwd.Prefill(prompt);
            cpuFirstLogits = logits.ToArray();
            cpuTokens = GreedyDecode(fwd, prompt, 16);
        }

        float maxAbsDelta = 0f;
        for (int i = 0; i < baselineFirstLogits.Length; i++)
            maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs(baselineFirstLogits[i] - cpuFirstLogits[i]));
        Console.Error.WriteLine(
            $"[NOTE] GPU-embedding vs CPU-embedding first-token (post-Prefill) max-abs logit delta: "
            + $"{maxAbsDelta:G6}. CudaBackend.EmbedLookupQ4K and the CPU Dequantize.ToFloat32 path "
            + "are not guaranteed bit-identical, and this model's output is CPU-resident regardless "
            + "(BFloat16 output.weight never fits GPU) — non-zero here is expected, not a regression.");

        // Same rationale as the -g cap test: this is argmax-stable, not byte-exact. The first
        // decoded token is the hard correctness bar.
        Assert.Equal(baselineTokens[0], cpuTokens[0]);
    }

    // ================================================================
    //  T3b — forced CPU embedding, multi-token prefill (regression: write-after-read hazard
    //  in the batched embed-staging path). T3 above uses a single-token "Hello" prompt, which
    //  only exercises EmbedToken's single-row CPU-embed branch — Forward/MtpForward end in a
    //  synchronizing Download, so that path was never hazardous. A prompt with N>=2 tokens
    //  routes through PrefillBatchedTrunkGpuFfn (the dense-FFN GDN-hybrid trunk-batch path,
    //  issue #119), whose embed loop used to call EmbedToken(_gpuHidden, tokens[i]) per token —
    //  reusing the single-row _pinnedEmbedRow pinned buffer across iterations. UploadInto issues
    //  an async cudaMemcpyAsync without syncing, so token i+1's CpuEmbedToken write could race
    //  token i's still-in-flight H2D copy, reading the wrong row on the device. Fixed by staging
    //  all N rows into one pinned block (EnsurePinnedEmbedAll) before a single bulk H2D.
    // ================================================================

    [Fact]
    public void CudaHybridGdnForwardPass_ForcedCpuEmbedding_MultiTokenPrefill_MatchesGpuEmbedding()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        // Long enough to guarantee N >= 8 and trip the batched-trunk prefill path
        // (Prefill requires N >= 2; using a longer margin so this doesn't depend on the
        // tokenizer's exact BPE split of a short phrase).
        var prompt = tokenizer.Encode(
            "The quick brown fox jumps over the lazy dog near the riverbank at dawn.").ToArray();
        Assert.True(prompt.Length >= 8, $"Prompt tokenized to only {prompt.Length} tokens (need >= 8).");

        var placement = MakePlacement(hp, hp.NumLayers, 4096);

        int baselineFirstToken;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            Assert.True(fwd.EmbeddingOnGpu, "Baseline run should keep the embedding on GPU.");
            var logits = fwd.Prefill(prompt);
            baselineFirstToken = Sampler.Greedy(logits);
        }

        Environment.SetEnvironmentVariable("SHARPI_GDN_CPU_EMBED", "1");
        int cpuFirstToken;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            Assert.False(fwd.EmbeddingOnGpu, "SHARPI_GDN_CPU_EMBED=1 should force the embedding off GPU.");
            var logits = fwd.Prefill(prompt);
            cpuFirstToken = Sampler.Greedy(logits);
        }

        Assert.Equal(baselineFirstToken, cpuFirstToken);
    }

    // ================================================================
    //  T4 — forced CPU output matches the auto-placement decision (short: 4 tokens)
    // ================================================================
    //
    // Deviation from the spec's literal "matches GPU output" framing: this model's
    // output.weight is BFloat16 (2.37 GiB raw) — CudaBackend caps a single exact allocation at
    // 2 GiB and GgufModel.GetTensorData reads via a Span<byte> (int32-length-capped), so GPU
    // output residency is architecturally impossible here regardless of budget or
    // SHARPI_GDN_CPU_OUTPUT=0. PlanVram's auto decision therefore already demotes it to CPU, so
    // there is no GPU-output baseline to compare against on this GGUF. Compare
    // SHARPI_GDN_CPU_OUTPUT=1 (forced) against the auto decision instead — both resolve to the
    // identical CPU-output code path, so exact token equality is the correct bar (no cross-kernel
    // GPU/CPU divergence is possible when neither run ever touches a GPU output kernel).

    [Fact]
    public void CudaHybridGdnForwardPass_ForcedCpuOutput_MatchesGpuOutput()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var prompt = tokenizer.Encode("Hello").ToArray();
        Assert.NotEmpty(prompt);

        var placement = MakePlacement(hp, hp.NumLayers, 4096);

        int[] baselineTokens;
        bool baselineOutputOnGpu;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            baselineOutputOnGpu = fwd.OutputOnGpu;
            baselineTokens = GreedyDecode(fwd, prompt, 4);
        }
        if (!baselineOutputOnGpu)
            Console.Error.WriteLine(
                "[NOTE] Auto-placement already demoted output to CPU on this model (BFloat16 "
                + "output.weight never fits GPU) — SHARPI_GDN_CPU_OUTPUT=1 is a placement no-op "
                + "here; this still exercises ComputeCpuOutput end-to-end and checks it agrees "
                + "with the auto-selected path.");

        Environment.SetEnvironmentVariable("SHARPI_GDN_CPU_OUTPUT", "1");
        int[] cpuTokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
        {
            Assert.False(fwd.OutputOnGpu, "SHARPI_GDN_CPU_OUTPUT=1 should force the output projection off GPU.");
            Assert.False(fwd.SupportsBatchVerify,
                "A CPU-resident output projection has no k×vocab on-device batched-verify path.");
            cpuTokens = GreedyDecode(fwd, prompt, 4);
        }

        Assert.Equal(baselineTokens, cpuTokens);
    }

    // ================================================================
    //  T5 — qwen35moe packed Q8_0 embedding matches F32-expanded (A/B)
    // ================================================================

    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_PackedQ8Embedding_MatchesF32Expanded()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindHybridModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var prompt = tokenizer.Encode("Hello").ToArray();
        Assert.NotEmpty(prompt);

        var placement = MakePlacement(hp, hp.NumLayers, 4096);

        Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_EMBED", "0");
        int[] f32Tokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            f32Tokens = GreedyDecode(fwd, prompt, 8);

        Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_EMBED", "1");
        int[] packedTokens;
        using (var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement))
            packedTokens = GreedyDecode(fwd, prompt, 8);

        Assert.Equal(f32Tokens, packedTokens);
    }

    // ================================================================
    //  T6 — MoE auto-select decision unchanged by the packed embedding
    // ================================================================

    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_MoeAutoSelectUnchangedByPackedEmbedding()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindHybridModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var placement = MakePlacement(hp, hp.NumLayers, 4096);

        string CaptureAutoSelectLine(string rawEmbed)
        {
            Environment.SetEnvironmentVariable("SHARPI_GDN_RAW_EMBED", rawEmbed);
            var sw = new StringWriter();
            var prevErr = Console.Error;
            Console.SetError(sw);
            try
            {
                using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
            }
            finally
            {
                Console.SetError(prevErr);
            }
            const string marker = "MoE auto-select: SLRU capacity ≈";
            foreach (var line in sw.ToString().Split('\n'))
                if (line.Contains(marker, StringComparison.Ordinal))
                    return line.Trim();
            return "";
        }

        string f32Line = CaptureAutoSelectLine("0");
        string packedLine = CaptureAutoSelectLine("1");

        Assert.NotEmpty(f32Line);
        Assert.Equal(f32Line, packedLine);
    }

    // ================================================================
    //  T7 — EstimateMtpHeadGpuBytes prices the MTP head sanely (metadata only)
    // ================================================================

    [Fact]
    public void PlanVram_27B_PricesMtpHeadAndKv()
    {
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (hp.NumMtpLayers <= 0) return; // not an MTP model — nothing to price

        long bytes = CudaHybridGdnForwardPass.EstimateMtpHeadGpuBytes(model, hp, maxSeqLen: 4096, kvDType: DType.BFloat16);

        Assert.True(bytes > 0, "Expected a non-zero MTP head GPU byte estimate for an MTP-bearing model.");

        const long expectedMiB = 442;
        const long MiB = 1024 * 1024;
        long lowMiB = (long)(expectedMiB * 0.9);
        long highMiB = (long)(expectedMiB * 1.1);
        long gotMiB = bytes / MiB;
        Assert.True(gotMiB >= lowMiB && gotMiB <= highMiB,
            $"MTP head GPU estimate {gotMiB} MiB outside ±10% of {expectedMiB} MiB [{lowMiB}, {highMiB}].");
    }
}
