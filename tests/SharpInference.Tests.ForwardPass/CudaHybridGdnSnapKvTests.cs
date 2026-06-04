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
    /// Issue #130 regression: with SnapKV eviction active on an MTP model, the
    /// first MTP batched-verify decode iteration used to throw
    /// <c>BatchForward2: _kvCache.Length=K != startPos=N</c> because eviction
    /// leaves the physical slot count (<see cref="PagedKvCache.Length"/>) at the
    /// budget K while the logical RoPE position (<see cref="PagedKvCache.LogicalLength"/>)
    /// stays at the prompt length N, and the decoder passes the logical position as
    /// <c>startPos</c>. The fix gates <see cref="CudaHybridGdnForwardPass.SupportsBatchVerify"/>
    /// to false once the cache is compacted, so <see cref="MtpDecoder"/> falls back
    /// to the eviction-safe sequential <c>Forward</c> path.
    ///
    /// This is a no-crash + coherence test (not bit-parity): decode must complete
    /// without throwing and produce non-degenerate output (first argmax != EOS,
    /// ≥2 distinct tokens) — IsFinite alone passes on all-EOS degenerate output.
    /// </summary>
    [Fact]
    public void CudaHybridGdnSnapKv_MtpDecode_AfterEviction_DoesNotCrash_StaysCoherent()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var path = FindMtpModelPath();
        if (path is null) return;

        const int budget = 128;
        const int promptTargetLen = 320;

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

            CudaHybridGdnForwardPass fwd;
            try
            {
                fwd = new CudaHybridGdnForwardPass(model, gpu, hp, placement);
            }
            catch (NotSupportedException) { return; }   // unsupported config — skip
            catch (InvalidOperationException) { return; } // VRAM / construction — skip

            using (fwd)
            {
                // Model must ship an MTP head for this path to be exercised.
                if (!fwd.HasMtpHead) return;

                var tokens = LongPrompt(tokenizer, promptTargetLen);
                Assert.True(tokens.Length >= budget + SnapKvSelector.DefaultWindow,
                    $"Prompt too short ({tokens.Length}) — SnapKV gate requires it to exceed budget + window.");

                var prefillLogits = fwd.Prefill(tokens).ToArray();

                // NON-VACUOUS: eviction must actually have occurred, otherwise we'd
                // never exercise the evicted-cache decode path this test targets.
                Assert.Equal(budget, fwd.Cache.Length);
                Assert.Equal(tokens.Length, fwd.Cache.LogicalLength);

                // The gate must have engaged: Length != LogicalLength ⇒ SupportsBatchVerify
                // is false even though the model otherwise supports batched verify.
                Assert.False(fwd.SupportsBatchVerify,
                    "Expected the #130 gate to disable batched-verify on a compacted cache.");

                var decoder = new MtpDecoder(fwd);
                decoder.Initialize(tokens.Length, prefillLogits);

                var produced = new List<int>(32);
                int[] stops = tokenizer.EogTokenIds.ToArray();

                // PRIMARY assertion: this must NOT throw. Pre-fix it threw the
                // BatchForward2 InvalidOperationException on the first iteration.
                decoder.Decode(
                    maxTokens: 24,
                    stopTokenIds: stops,
                    emitToken: produced.Add);

                // COHERENCE: first emitted token must not be EOS, and there must be
                // multi-token variety (IsFinite-only passes on degenerate all-EOS).
                Assert.NotEmpty(produced);
                Assert.DoesNotContain(produced[0], stops);
                int distinct = produced.Distinct().Count();
                Assert.True(distinct >= 2,
                    $"MTP decode after eviction produced only {distinct} distinct token(s): " +
                    $"[{string.Join(",", produced)}]. Degenerate output suggests the " +
                    "post-eviction sequential Forward fallback is broken.");
            }
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

            // Gate false-positive guard (issue #130): with no eviction the cache is not
            // compacted (Length == LogicalLength), so the #130 gate must NOT fire — an
            // MTP model still reports batched-verify as available here.
            if (fwd.HasMtpHead)
                Assert.True(fwd.SupportsBatchVerify,
                    "SupportsBatchVerify must stay true when the cache was not evicted; " +
                    "the #130 gate should only disable batched-verify on a compacted cache.");
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
