using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// SnapKV (issue #59) GpuForwardPass coverage — the Vulkan full-GPU dense path.
/// Mirrors <see cref="CudaForwardPassSnapKvTests"/> for the Vulkan backend. Asserts:
/// <list type="bullet">
///   <item>KvLength shrinks to the configured budget after a long-prompt prefill,</item>
///   <item>decode produces finite, non-degenerate logits and ≥2 distinct argmaxes,</item>
///   <item>with the env var unset on a tiny context, the cache is untouched
///         (auto-budget gated off below the cache-size threshold),</item>
///   <item>a short prompt (under window size) skips eviction even with the env
///         var set — the SnapKV gate excludes it.</item>
/// </list>
///
/// Skipped silently when Vulkan is unavailable or the Qwen3-8B GGUF isn't on disk.
/// </summary>
public sealed class GpuForwardPassSnapKvTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{ModelFile}",
            $@"E:\models\{ModelFile}",
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
    public void GpuForwardPassSnapKv_LongPrompt_CacheShrinksToBudget_DecodeStaysWellFormed()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        const int promptTargetLen = 768;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            // Keep ctx tight so the prefill (and the matching budget gate) stays
            // small — we're testing the eviction logic, not the throughput path.
            int ctx = Math.Min(hp.ContextLength, 2048);
            using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: ctx);

            var tokens = LongPrompt(tokenizer, promptTargetLen);
            Assert.True(tokens.Length >= budget + SnapKvSelector.DefaultWindow,
                $"Prompt too short ({tokens.Length}) — SnapKV gate requires it to exceed budget + window.");

            var logits = fwd.Prefill(tokens).ToArray();

            Assert.Equal(budget, fwd.KvLength);

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.True(float.IsFinite(logits[i]),
                    $"Non-finite logit at vocab idx {i}: {logits[i]} — likely a shader " +
                    "bug in SnapKvScore / KvCompact or a slot-vs-position mismatch in " +
                    "the post-eviction Attention seqLen.");
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
                "mismatch — Forward's position arg should track tokens.Length + i regardless " +
                "of the compacted slot count.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    /// <summary>
    /// With <c>SHARPI_SNAPKV_BUDGET</c> unset and a small configured context,
    /// the full-cache byte size sits below
    /// <see cref="SnapKvConfig.AutoEnableMinCacheBytes"/> and the auto-budget
    /// stays disabled. Verifies the threshold keeps small-context smoke runs
    /// lossless even though SnapKV is otherwise auto-enabled on GpuForwardPass.
    /// </summary>
    [Fact]
    public void GpuForwardPassSnapKv_EnvUnset_SmallCtx_CacheUntouched()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindModelPath();
        if (path is null) return;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            // ctx=512 on Qwen3-8B (8 KV heads × 128 = 1024 kv_dim, 32 layers, fp32)
            // sits below the auto-enable threshold.
            int ctx = Math.Min(hp.ContextLength, 512);
            using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: ctx);

            var tokens = LongPrompt(tokenizer, 384);
            _ = fwd.Prefill(tokens);

            Assert.Equal(tokens.Length, fwd.KvLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    /// <summary>
    /// A short prompt (≤ budget or ≤ window) skips eviction even with the budget
    /// explicitly set. Covers the upper guards in the Prefill SnapKV gate.
    /// </summary>
    [Fact]
    public void GpuForwardPassSnapKv_ShortPrompt_BudgetNotTriggered_CacheUntouched()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;

        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            int ctx = Math.Min(hp.ContextLength, 2048);
            using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: ctx);

            // Prompt is short enough that N <= budget — gate excludes it.
            var tokens = tokenizer.Encode("Hello world, this is a tiny prompt.").ToArray();
            Assert.True(tokens.Length <= budget,
                $"Test pre-condition: prompt is {tokens.Length} tokens, should be ≤ {budget}.");

            _ = fwd.Prefill(tokens);

            Assert.Equal(tokens.Length, fwd.KvLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }
}
