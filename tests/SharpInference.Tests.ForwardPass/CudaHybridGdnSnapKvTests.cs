using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// SnapKV (issue #58) CUDA hybrid GDN port. Mirrors <see cref="SnapKvTests"/>'s
/// CPU coverage on the qwen35 27B-MTP model — the dense hybrid GDN that fits in
/// 12 GB VRAM. Asserts:
/// <list type="bullet">
///   <item>cache.Length shrinks to the configured budget after a long-prompt prefill,</item>
///   <item>cache.LogicalLength stays at the original prompt length (decode RoPE
///         stays in the correct position frame),</item>
///   <item>decode produces finite, non-degenerate logits and ≥2 distinct argmaxes,</item>
///   <item>with the env var unset, the cache is untouched (default == disabled).</item>
/// </list>
///
/// Skipped silently when CUDA is unavailable or the 27B-MTP GGUF isn't on disk.
/// </summary>
public sealed class CudaHybridGdnSnapKvTests
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
    /// Repeat a few sentences until the tokenizer encodes at least
    /// <paramref name="approxTokenCount"/> tokens. Picks varied content so the
    /// SnapKV scoring kernel sees a non-degenerate attention pattern.
    /// </summary>
    private static int[] LongPrompt(GgufTokenizer tokenizer, int approxTokenCount)
    {
        const string seed =
            "The quick brown fox jumps over the lazy dog. " +
            "Sphinx of black quartz, judge my vow. " +
            "Pack my box with five dozen liquor jugs. ";
        var sb = new System.Text.StringBuilder();
        while (true)
        {
            sb.Append(seed);
            var attempt = tokenizer.Encode(sb.ToString());
            if (attempt.Count >= approxTokenCount) return attempt.ToArray();
            if (sb.Length > 100_000)
                throw new InvalidOperationException("Tokenizer not packing enough — unexpected for Qwen3.");
        }
    }

    [Fact]
    public void CudaHybridGdnSnapKv_LongPrompt_CacheShrinksToBudget_DecodeStaysWellFormed()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMtpModelPath();
        if (path is null) return;

        const int budget = 256;
        const int promptTargetLen = 384;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers,
                CpuLayers: 0,
                GpuWeightBytes: 0,
                GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

            using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);

            var tokens = LongPrompt(tokenizer, promptTargetLen);
            Assert.True(tokens.Length >= budget + SnapKvSelector.DefaultWindow,
                $"Prompt too short ({tokens.Length}) — SnapKV gate requires it to exceed budget + window.");

            var logits = fwd.Prefill(tokens).ToArray();

            Assert.Equal(budget, fwd.Cache.Length);
            Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.True(float.IsFinite(logits[i]),
                    $"Non-finite logit at vocab idx {i}: {logits[i]} — likely a kernel " +
                    "bug in llm_snapkv_score / llm_kv_compact or a slot-vs-position " +
                    "mismatch in the post-eviction Attention seqLen.");
                if (logits[i] < min) min = logits[i];
                if (logits[i] > max) max = logits[i];
            }
            Assert.True(max - min > 0.5f,
                $"Post-SnapKV logit range collapsed to {min:F3}..{max:F3}; GPU eviction is " +
                "producing degenerate output.");

            var produced = new List<int>(4);
            for (int i = 0; i < 4; i++)
            {
                int next = Sampler.Greedy(logits);
                produced.Add(next);
                logits = fwd.Forward(next, tokens.Length + i).ToArray();
                for (int k = 0; k < logits.Length; k++)
                    Assert.True(float.IsFinite(logits[k]),
                        $"Non-finite logit at decode step {i}, vocab idx {k}: {logits[k]}");
            }

            int distinct = produced.Distinct().Count();
            Assert.True(distinct >= 2,
                $"Greedy decode under SnapKV produced only {distinct} distinct token(s): " +
                $"[{string.Join(",", produced)}]. Likely a kvPosition vs LogicalLength " +
                "mismatch — Forward's kvPosition should be _kvCache.Length post-compaction.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    /// <summary>
    /// With <c>SHARPI_SNAPKV_BUDGET</c> unset and a small configured context,
    /// the cache size sits below <see cref="SnapKvConfig.AutoEnableMinCacheBytes"/>
    /// and the auto-budget stays disabled. Verifies the cache-size threshold
    /// keeps small-context smoke runs lossless even though SnapKV is otherwise
    /// auto-enabled on the CUDA hybrid GDN path.
    /// </summary>
    [Fact]
    public void CudaHybridGdnSnapKv_SmallCtxDefault_AutoBudgetStaysOff()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMtpModelPath();
        if (path is null) return;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            // ctx=2048 on Qwen3.6-27B-MTP bf16 → full cache ≈ 40 MiB, well below
            // the 256 MiB auto-enable threshold.
            var placement = new LayerPlacement(
                GpuLayers: hp.NumLayers,
                CpuLayers: 0,
                GpuWeightBytes: 0,
                GpuKvBytes: 0,
                RecommendedCtxSize: Math.Min(hp.ContextLength, 2048));

            using var fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
            var tokens = LongPrompt(tokenizer, 384);
            _ = fwd.Prefill(tokens);

            Assert.Equal(tokens.Length, fwd.Cache.Length);
            Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    /// <summary>
    /// Auto-budget formula coverage — direct unit-level checks on the heuristic
    /// without paying the ~30 s cost of loading the 27B-MTP GGUF.
    /// </summary>
    [Fact]
    public void SnapKvConfig_ComputeAutoBudget_RespectsCacheSizeAndCaps()
    {
        // Cache below the auto-enable threshold → returns 0.
        Assert.Equal(0, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 2048, fullCacheBytes: 40L * 1024 * 1024));

        // Cache above threshold and maxSeqLen in the "floor regime" (≤ 4096):
        // candidate = maxSeqLen/4 (= 1024 at 4096) is exactly the floor.
        Assert.Equal(1024, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 4096, fullCacheBytes: 512L * 1024 * 1024));

        // maxSeqLen=8192 → candidate = 2048 (above floor, below cap).
        Assert.Equal(2048, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 8192, fullCacheBytes: 1024L * 1024 * 1024));

        // maxSeqLen=16384 → candidate = 4096 — exactly at the cap.
        Assert.Equal(4096, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 16384, fullCacheBytes: 2L * 1024 * 1024 * 1024));

        // Cap holds at very large maxSeqLen.
        Assert.Equal(4096, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 65536, fullCacheBytes: 4L * 1024 * 1024 * 1024));

        // Unknown cache size (0) bypasses the threshold check — used for
        // backends that can't measure ahead of time.
        Assert.Equal(2048, SnapKvConfig.ComputeAutoBudget(
            maxSeqLen: 8192, fullCacheBytes: 0));
    }

    /// <summary>
    /// When <c>SHARPI_SNAPKV_BUDGET</c> is explicitly set to <c>0</c>, eviction
    /// stays disabled regardless of whether the auto-budget would otherwise
    /// have fired. Distinguishes "env unset" (auto) from "env=0" (off).
    /// </summary>
    [Fact]
    public void SnapKvConfig_FromEnvironment_DistinguishesUnsetFromZero()
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        try
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
            var unset = SnapKvConfig.FromEnvironment();
            Assert.False(unset.IsBudgetExplicit);
            Assert.False(unset.Enabled);

            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
            var explicitZero = SnapKvConfig.FromEnvironment();
            Assert.True(explicitZero.IsBudgetExplicit);
            Assert.False(explicitZero.Enabled);

            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "1024");
            var explicitValue = SnapKvConfig.FromEnvironment();
            Assert.True(explicitValue.IsBudgetExplicit);
            Assert.True(explicitValue.Enabled);
            Assert.Equal(1024, explicitValue.Budget);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }
}
