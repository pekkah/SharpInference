using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for SnapKV prefill-time KV eviction (issue #51).
///
/// The unit-level Compact tests live in <see cref="PagedKvCacheTests"/>; this
/// suite exercises the end-to-end env-driven path on a real (small) model and
/// asserts the user-visible invariants:
/// <list type="bullet">
///   <item>cache.Length shrinks to the budget after a long-prompt prefill,</item>
///   <item>cache.LogicalLength stays at the original prompt length so RoPE on
///         subsequent decode tokens lands at the right angle,</item>
///   <item>decode produces finite, non-degenerate logits and distinct argmaxes
///         (the eviction didn't lobotomise the model into a single-token loop).</item>
/// </list>
/// Skipped silently when the small-model file isn't on disk.
/// </summary>
public sealed class SnapKvTests
{
    private static string? FindModelPath(string filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Build a prompt longer than the budget so eviction actually fires.
    /// The exact content doesn't matter for well-formedness — we just need
    /// >budget tokens of real-ish text so the scoring sees varied attention
    /// patterns instead of a degenerate uniform distribution.
    /// </summary>
    private static int[] LongPrompt(GgufTokenizer tokenizer, int approxTokenCount)
    {
        // Repeat a few sentences to reach the target length; SmolLM2's BPE
        // gives ~25 tokens for this sentence, so 16 repeats ≈ 400 tokens.
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
                throw new InvalidOperationException("Tokenizer not packing enough — unexpected for SmolLM2.");
        }
    }

    [Fact]
    public void SnapKv_LongPrompt_CacheShrinksToBudget_DecodeStaysWellFormed()
    {
        var path = FindModelPath();
        if (path is null) return;

        // 256 / 384 split: prompt is 384 tokens, eviction keeps 256 (~33% drop).
        // Window (last-W queries used for scoring) and recency (always-kept
        // trailing positions) default to 64 each.
        const int budget = 256;
        const int promptTargetLen = 384;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, promptTargetLen);
            Assert.True(tokens.Length >= budget + SnapKvSelector.DefaultWindow,
                $"Prompt too short ({tokens.Length}) — SnapKV gate requires it to exceed budget + window.");

            var logits = fwd.Prefill(tokens).ToArray();

            // Eviction should have compacted to the budget.
            Assert.Equal(budget, fwd.Cache.Length);
            Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);

            // Logits well-formed.
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.True(float.IsFinite(logits[i]),
                    $"Non-finite logit at vocab idx {i}: {logits[i]} — SnapKV compaction or " +
                    "the slot-length-aware Attention seqLen formula is broken.");
                if (logits[i] < min) min = logits[i];
                if (logits[i] > max) max = logits[i];
            }
            Assert.True(max - min > 0.5f,
                $"Post-SnapKV logit range collapsed to {min:F3}..{max:F3}; eviction is " +
                "producing degenerate output.");

            // Decode 4 tokens, asserting at least 2 distinct argmaxes — same
            // guardrail used by the CUDA hybrid GDN smoke tests.
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
                $"[{string.Join(",", produced)}]. Likely a slot-vs-position mismatch in the " +
                "Attention seqLen clamp or in PagedKvCache.Compact.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    [Fact]
    public void SnapKv_DisabledByDefault_NoCacheShrink()
    {
        var path = FindModelPath();
        if (path is null) return;

        // Force-clear the env var so this test is independent of how it was
        // run (e.g. in a session where the long-prompt test set it).
        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var tokens = LongPrompt(tokenizer, 384);
            _ = fwd.Prefill(tokens);

            // No eviction → cache.Length == prompt length == LogicalLength.
            Assert.Equal(tokens.Length, fwd.Cache.Length);
            Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    [Fact]
    public void SnapKv_BudgetExceedsPrompt_NoEviction()
    {
        var path = FindModelPath();
        if (path is null) return;

        // Budget larger than prompt → gating should skip eviction entirely.
        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "8192");
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var tokens = LongPrompt(tokenizer, 128);
            _ = fwd.Prefill(tokens);

            Assert.Equal(tokens.Length, fwd.Cache.Length);
            Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }
}
