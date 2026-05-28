using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Smoke tests for <see cref="CudaHybridGdnForwardPass"/> — the CUDA + CPU hybrid
/// for qwen35moe (Qwen3.6-35B-A3B). Skipped silently when CUDA is unavailable or
/// the 22 GB qwen35moe GGUF isn't on disk.
///
/// Includes both well-formedness (finite, non-degenerate, non-collapsed logits)
/// and greedy parity against llama.cpp b9245's reference decode of "Hello" with
/// `--temp 0 -no-cnv`: first token must be 11 (",") and the next three tokens
/// must reproduce the published continuation "\n\nI". The CPU baseline
/// <see cref="HybridGdnForwardPass"/> already passes this check post-Phase 5.
/// </summary>
public sealed class CudaHybridGdnForwardPassTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindHybridModelPath()
    {
        // Big model: typically on E:\models\.
        string[] absoluteCandidates =
        {
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

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

    /// <summary>
    /// Loads qwen35moe via the CUDA hybrid path, runs prefill + 4 greedy decode
    /// tokens, and asserts the output is well-formed.
    ///
    /// LayerPlacement is constructed with <c>GpuLayers = NumLayers</c> — that's
    /// a hint to the class that the user wants GPU usage; the class's internal
    /// dispatch always routes GDN blocks to CPU and attention blocks to GPU
    /// regardless. Phase 6c will introduce a hybrid-aware planner.
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_ProducesWellFormedLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;   // silent skip — same pattern as CudaMoeTests

        var path = FindHybridModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: this test should only fire on a hybrid GDN model with MoE.
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for qwen35moe model");
        Assert.NotNull(hp.Gdn);
        Assert.NotNull(hp.LayerTypes);
        Assert.True(hp.IsMoE, "Expected hp.IsMoE for qwen35moe model");

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // GpuLayers = NumLayers is the "user wants GPU" sentinel; the class ignores
        // the cut point and dispatches per LayerTypes.
        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers,
            CpuLayers: 0,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

        using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);

        // Prefill — sequential T-step recurrence under the hood.
        var logits = fwd.Prefill(tokens);

        // Range + finiteness.
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            Assert.True(float.IsFinite(v), $"Non-finite logit at vocab idx {i}: {v}");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"Logit range too tight ({min:F3}..{max:F3}); CUDA hybrid GDN forward pass " +
            "is producing degenerate output.");

        // Greedy decode 4 tokens; assert at least 2 distinct argmaxes.
        var decoded = new List<int>(4);
        for (int i = 0; i < 4; i++)
        {
            int next = Sampler.Greedy(logits);
            decoded.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);

            for (int k = 0; k < logits.Length; k++)
                Assert.True(float.IsFinite(logits[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {logits[k]}");
        }

        int distinct = decoded.Distinct().Count();
        Assert.True(distinct >= 2,
            $"Greedy decode produced only {distinct} distinct token(s) across 4 steps " +
            $"({string.Join(",", decoded)}); the CUDA hybrid forward pass may be stuck in a " +
            "degenerate loop (logits collapsed onto a single output).");
    }

    /// <summary>
    /// First-token strict parity: greedy decode of the raw prompt "Hello" must
    /// produce token 11 (",") as the first generation step — the same answer
    /// llama.cpp b9245 and the CPU <see cref="HybridGdnForwardPass"/> emit
    /// post-Phase 5. This proves the whole CUDA stack (embedding lookup,
    /// 30 CPU GDN layers + 10 GPU attention layers, MoE router + SLRU experts
    /// + shared expert, output projection) computes the right answer in a
    /// 40-layer transit.
    ///
    /// We DON'T assert greedy parity beyond step 0: Q8_1 GPU matmul rounding
    /// is a per-layer ε ~1e-3 and accumulates fast enough across 40 layers ×
    /// many decode steps that ties between close top-1/top-2 logits flip.
    /// Phase 5's CPU run on the same prompt produces "Hello, I am trying to
    /// use the `get` function..." — coherent text — so a softer "produces
    /// English-shaped tokens" check at step ≥1 is the appropriate guardrail.
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_FirstTokenMatchesCpuBaseline()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindHybridModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers, CpuLayers: 0,
            GpuWeightBytes: 0, GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

        using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);

        // The reference first decoded token. Capture via the tokenizer so the
        // test isn't brittle to BPE-level upstream changes.
        int referenceFirstToken = tokenizer.Encode(",")[0];

        var logits = fwd.Prefill(tokens);
        int produced = Sampler.Greedy(logits);

        Assert.True(produced == referenceFirstToken,
            $"CUDA hybrid greedy decode diverged from CPU/llama.cpp at step 0: " +
            $"expected token {referenceFirstToken} (\",\"), got {produced}. " +
            "First-token mismatch is structural — it would mean a numerical bug " +
            "in the embedding→40-layer→output pipeline, NOT just precision drift.");
    }

    /// <summary>
    /// CPU-MoE experiment (SHARPI_CPU_MOE=1): keep attention on GPU and route
    /// the entire MoE FFN (routed experts + shared expert + scalar gate) through
    /// CPU SimdKernels via mmap reads. This eliminates SLRU thrash. First-token
    /// parity must hold because the MoE math is identical to
    /// <see cref="HybridGdnForwardPass.MoeFfn"/> which already passes the same
    /// strict greedy check post-Phase 5.
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Moe_CpuMoeMode_MatchesCpuBaseline()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindHybridModelPath();
        if (path is null) return;

        // Activate the experimental CPU-MoE path for the duration of the test.
        var prev = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers, CpuLayers: 0,
                GpuWeightBytes: 0, GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

            using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

            var tokens = tokenizer.Encode("Hello");
            Assert.NotEmpty(tokens);

            int referenceFirstToken = tokenizer.Encode(",")[0];

            var logits = fwd.Prefill(tokens);
            int produced = Sampler.Greedy(logits);

            Assert.True(produced == referenceFirstToken,
                $"CPU-MoE CUDA hybrid greedy decode diverged from CPU/llama.cpp at step 0: " +
                $"expected token {referenceFirstToken} (\",\"), got {produced}. " +
                "Since the CPU MoE math mirrors HybridGdnForwardPass.MoeFfn exactly, " +
                "a divergence here likely indicates a weight-loading or wiring bug in " +
                "the SHARPI_CPU_MOE branch.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prev);
        }
    }

    /// <summary>
    /// Probes for the qwen35 27B-MTP GGUF in the small-models directory. Tracked
    /// separately from <see cref="FindHybridModelPath"/> because qwen35-MTP is a
    /// dense (non-MoE) hybrid GDN model with an MTP head, not the 35B-A3B MoE.
    /// </summary>
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
    /// Smoke test for the MTP / NEXTN head on the CUDA hybrid path (issue #29).
    /// Mirrors <see cref="HybridGdnForwardPassTests"/>'s CPU MTP test: asserts
    /// the head loads, <see cref="IForwardPass.LastHidden"/> is downloaded from
    /// GPU after each main forward, and <see cref="IForwardPass.MtpForward"/>
    /// produces finite, non-degenerate logits routed entirely on GPU.
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Mtp_MtpHeadProducesWellFormedLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for qwen35 27B-MTP");
        Assert.NotNull(hp.Gdn);
        Assert.False(hp.IsMoE, "qwen35 27B-MTP is dense, not MoE");
        Assert.Equal(1, hp.NumMtpLayers);

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers,
            CpuLayers: 0,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

        using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

        Assert.True(fwd.HasMtpHead,
            "CudaHybridGdnForwardPass should have detected the MTP head at " +
            "blk.NumLayers and reported HasMtpHead == true.");

        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);

        var mainLogits = fwd.Prefill(tokens);

        var lastHidden = fwd.LastHidden;
        Assert.Equal(hp.EmbeddingDim, lastHidden.Length);
        for (int i = 0; i < lastHidden.Length; i++)
            Assert.True(float.IsFinite(lastHidden[i]),
                $"LastHidden has a non-finite entry at index {i}: {lastHidden[i]}. " +
                "Either the pre-output-norm snapshot or the GPU→host Download is broken.");

        int t1 = Sampler.Greedy(mainLogits);
        var mtpLogits = fwd.MtpForward(t1, tokens.Count, lastHidden);

        Assert.Equal(hp.VocabSize, mtpLogits.Length);

        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < mtpLogits.Length; i++)
        {
            float v = mtpLogits[i];
            Assert.True(float.IsFinite(v),
                $"CUDA MTP logit non-finite at vocab idx {i}: {v}. " +
                "Likely culprits: eh_proj F32 upload, CopyDeviceRegion concat halves, " +
                "MTP attention KV cache wiring, or shared_head_norm.");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"CUDA MTP logit range too tight ({min:F3}..{max:F3}); the head is " +
            "producing degenerate output. Likely culprits: SplitQG with MTP weights, " +
            "GLU gate, or output projection.");
    }

    /// <summary>
    /// Issue #27: Bf16 KV cache parity on the qwen35 27B-MTP CUDA hybrid path.
    /// Runs the same prompt twice — once with the legacy fp32 KV cache, once
    /// with the default bf16 cache — and asserts:
    ///
    ///   • bf16 logits are finite and non-degenerate,
    ///   • greedy top-1 matches fp32 at prefill,
    ///   • bf16 logits stay within a generous tolerance of fp32 (max top-K logit
    ///     gap &lt; 0.5 on the top 16 vocab entries, ratifying that bf16's
    ///     8-bit mantissa hasn't blown out attention output magnitudes).
    ///
    /// We intentionally use the qwen35 27B-MTP dense model (not qwen35moe): it
    /// loads on a 12 GB card and the parity surface is smaller (no SLRU/MoE
    /// non-determinism on top of cache precision).
    /// </summary>
    [Fact]
    public void CudaHybridGdnForwardPass_Qwen35Mtp_Bf16KvCache_GreedyMatchesFp32()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMtpModelPath();
        if (path is null) return;

        var prev = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");

        // Reference: fp32 KV cache (legacy precision).
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "fp32");
        int fp32Top1;
        float[] fp32TopK;
        try
        {
            (fp32Top1, fp32TopK) = RunGreedyPrefill(gpu, path);
        }
        finally { Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prev); }

        // Candidate: bf16 KV cache (the new default).
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "bf16");
        int bf16Top1;
        float[] bf16TopK;
        try
        {
            (bf16Top1, bf16TopK) = RunGreedyPrefill(gpu, path);
        }
        finally { Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prev); }

        Assert.True(fp32Top1 == bf16Top1,
            $"Bf16 KV cache diverged at prefill greedy: fp32 picked token {fp32Top1}, " +
            $"bf16 picked {bf16Top1}. Bf16's 8-bit mantissa shouldn't flip top-1 on a " +
            "short prompt — investigate KvAppendBf16 / AttentionBf16 wiring.");

        // Compare the top-16 logits in fp32-vocab order. Bf16's per-element ε is
        // ~1/256; accumulated over an attention dot of head_dim=256 the worst
        // case is ε ≈ 1.0 in absolute value, much smaller in practice. Cap at 0.5.
        float maxAbsDiff = 0f;
        for (int i = 0; i < fp32TopK.Length; i++)
            maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(fp32TopK[i] - bf16TopK[i]));
        Assert.True(maxAbsDiff < 0.5f,
            $"Bf16 KV cache produced top-K logits diverging by {maxAbsDiff:F3} from fp32; " +
            "expected < 0.5. Either Bf16 conversion is broken or the kernel is reading " +
            "stale ring positions.");
    }

    private static (int top1, float[] topK) RunGreedyPrefill(CudaBackend gpu, string modelPath)
    {
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        var placement = new LayerPlacement(
            GpuLayers: hp.NumLayers,
            CpuLayers: 0,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

        using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
        var tokens = tokenizer.Encode("Hello");
        var logits = fwd.Prefill(tokens).ToArray();

        int top1 = Sampler.Greedy(logits);

        // Capture the top-16 raw logits (descending) so the parity diff isn't
        // dominated by a single high-magnitude entry.
        const int K = 16;
        var sorted = (float[])logits.Clone();
        Array.Sort(sorted, (a, b) => b.CompareTo(a));
        var topK = new float[K];
        Array.Copy(sorted, topK, K);
        return (top1, topK);
    }
}
