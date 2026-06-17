using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #196: SnapKV-aware CUDA continuous batching. Before #196, an active SnapKV budget made
/// <see cref="CudaForwardPass.SupportsContinuousBatching"/> false (long-context concurrent serving
/// silently fell back to single-user). #196 ships two things:
/// <list type="bullet">
///   <item><b>Option 1</b> — per-sequence eviction: <see cref="CudaForwardPass.PrefillWithCache"/>
///     scores + compacts each <see cref="CudaSequenceKvCache"/> at the end of its own prefill and
///     records the logical-minus-physical delta on the cache; the batched decode maps a logical
///     position to the physical slot <c>pos - EvictedCount</c> (RoPE keeps the logical pos), so
///     eviction composes with batching.</item>
///   <item><b>Option 2</b> — when batching is preferred (<c>preferBatchingOverAutoSnapKv</c>), the
///     VRAM-scaled SnapKV AUTO-enable is suppressed so a server doesn't silently route every
///     sequence through the slower per-sequence-eviction decode; an explicit budget still wins.</item>
/// </list>
///
/// Oracle: the batched decode of an evicted per-sequence cache must reproduce the single-user
/// SnapKV decode (owned cache) on dense <b>Qwen3-8B Q4_K</b>. Silent-skips when CUDA / the GGUF is
/// absent. Mirrors <see cref="CudaForwardPassSnapKvTests"/> + <see cref="CudaBatchForwardMultiTests"/>.
/// </summary>
public sealed class CudaSnapKvBatchingTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absolute = { $@"C:\p\sharpi\models\{ModelFile}", $@"E:\models\{ModelFile}" };
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

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static (float maxAbs, int overlap) Compare(float[] reference, float[] candidate)
    {
        Assert.Equal(reference.Length, candidate.Length);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - candidate[i]));
        var refTop = new HashSet<int>();
        {
            var idx = new int[reference.Length];
            for (int i = 0; i < idx.Length; i++) idx[i] = i;
            Array.Sort(idx, (a, b) => reference[b].CompareTo(reference[a]));
            for (int i = 0; i < 5 && i < idx.Length; i++) refTop.Add(idx[i]);
        }
        int overlap = 0;
        {
            var idx = new int[candidate.Length];
            for (int i = 0; i < idx.Length; i++) idx[i] = i;
            Array.Sort(idx, (a, b) => candidate[b].CompareTo(candidate[a]));
            for (int i = 0; i < 5 && i < idx.Length; i++) if (refTop.Contains(idx[i])) overlap++;
        }
        return (maxAbs, overlap);
    }

    private static IDisposable SnapKvEnv(int budget, int window)
    {
        return new EnvScope(
            ("SHARPI_SNAPKV_BUDGET", budget.ToString()),
            ("SHARPI_SNAPKV_WINDOW", window.ToString()),
            ("SHARPI_PREFIX_SLOTS", null)); // ensure multi-slot doesn't suppress the budget
    }

    private sealed class EnvScope : IDisposable
    {
        private readonly (string Key, string? Prev)[] _saved;
        public EnvScope(params (string Key, string? Value)[] vars)
        {
            _saved = new (string, string?)[vars.Length];
            for (int i = 0; i < vars.Length; i++)
            {
                _saved[i] = (vars[i].Key, Environment.GetEnvironmentVariable(vars[i].Key));
                Environment.SetEnvironmentVariable(vars[i].Key, vars[i].Value);
            }
        }
        public void Dispose()
        {
            foreach (var (key, prev) in _saved)
                Environment.SetEnvironmentVariable(key, prev);
        }
    }

    /// <summary>
    /// Option 1 gate: with an explicit SnapKV budget, the model now supports continuous batching
    /// (it was disqualified before #196). SnapKvEnabled stays true (the budget is honored).
    /// </summary>
    [Fact]
    public void Qwen3_8B_ExplicitSnapKv_SupportsContinuousBatching()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var env = SnapKvEnv(budget: 512, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);

        Assert.True(fwd.SnapKvEnabled, "explicit SHARPI_SNAPKV_BUDGET should keep SnapKV active.");
        Assert.True(fwd.SupportsContinuousBatching,
            "SnapKV + dense should now support continuous batching via per-sequence eviction (#196).");
    }

    /// <summary>
    /// Option 1 oracle (N=1): a prompt long enough to trigger SnapKV eviction, prefilled into a
    /// per-sequence cache, then a batched decode step — must reproduce the single-user SnapKV
    /// decode (owned cache). Asserts the per-sequence cache actually evicted (EvictedCount &gt; 0,
    /// physical Length &lt; prompt) and that physical+delta == logical.
    /// </summary>
    [Fact]
    public void Qwen3_8B_SnapKvBatchedDecode_N1_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        using var env = SnapKvEnv(budget, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);
        Assert.True(fwd.SnapKvEnabled);

        int[] prompt = LongPrompt(tokenizer, 700);

        // Single-user reference: owned-cache SnapKV prefill (evicts) + one decode step.
        fwd.ResetCache();
        int tok = Argmax(fwd.Prefill(prompt));
        Assert.True(fwd.KvLength < prompt.Length, "single-user prefill should have evicted.");
        float[] refLogits = fwd.Forward(tok, prompt.Length).ToArray();
        fwd.ResetCache(); // clear the owned-cache eviction state so the batched decode guard passes

        // Batched: per-sequence cache SnapKV prefill (evicts THIS cache) + batched decode step.
        using var cache = fwd.CreateCache();
        int tok2 = Argmax(fwd.PrefillWithCache(prompt, cache));
        Assert.Equal(tok, tok2); // identical eviction + prefill
        Assert.True(cache.EvictedCount > 0, "per-sequence prefill should have evicted.");
        Assert.True(cache.Length < prompt.Length);
        Assert.Equal(prompt.Length, cache.Length + cache.EvictedCount); // physical + delta == logical

        float[][] batch = fwd.BatchForwardMulti([tok2], [prompt.Length], [cache]);
        Assert.Single(batch);
        var (maxAbs, overlap) = Compare(refLogits, batch[0]);
        Assert.Equal(Argmax(refLogits), Argmax(batch[0]));
        Assert.True(overlap >= 4, $"SnapKV batched decode top-5 overlap {overlap}/5 (maxAbs={maxAbs}).");
        Assert.True(maxAbs < 1.0f, $"SnapKV batched decode vs single-user maxAbs={maxAbs}.");
    }

    /// <summary>
    /// Option 1 oracle (N=2): two prompts of DIFFERENT lengths evict to the same budget but with
    /// DIFFERENT deltas, so this exercises per-sequence eviction state (each cache's own
    /// EvictedCount drives its physical-slot mapping). Both sequences' batched decode must match
    /// their single-user references.
    /// </summary>
    [Fact]
    public void Qwen3_8B_SnapKvBatchedDecode_N2_PerSequenceEviction()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        using var env = SnapKvEnv(budget, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);

        int[] promptA = LongPrompt(tokenizer, 600);
        int[] promptB = LongPrompt(tokenizer, 1000);
        Assert.True(promptB.Length > promptA.Length);

        // Single-user references.
        fwd.ResetCache();
        int tokA = Argmax(fwd.Prefill(promptA));
        float[] refA = fwd.Forward(tokA, promptA.Length).ToArray();
        fwd.ResetCache();
        int tokB = Argmax(fwd.Prefill(promptB));
        float[] refB = fwd.Forward(tokB, promptB.Length).ToArray();
        fwd.ResetCache();

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        int tokA2 = Argmax(fwd.PrefillWithCache(promptA, cacheA));
        int tokB2 = Argmax(fwd.PrefillWithCache(promptB, cacheB));
        Assert.Equal(tokA, tokA2);
        Assert.Equal(tokB, tokB2);
        Assert.True(cacheA.EvictedCount > 0 && cacheB.EvictedCount > 0);
        Assert.NotEqual(cacheA.EvictedCount, cacheB.EvictedCount); // different prompt lengths

        float[][] batch = fwd.BatchForwardMulti(
            [tokA2, tokB2], [promptA.Length, promptB.Length], [cacheA, cacheB]);
        Assert.Equal(2, batch.Length);

        var (maxAbsA, overlapA) = Compare(refA, batch[0]);
        Assert.Equal(Argmax(refA), Argmax(batch[0]));
        Assert.True(overlapA >= 4, $"Seq A SnapKV batched top-5 {overlapA}/5 (maxAbs={maxAbsA}).");
        Assert.True(maxAbsA < 1.0f, $"Seq A SnapKV batched maxAbs={maxAbsA}.");

        var (maxAbsB, overlapB) = Compare(refB, batch[1]);
        Assert.Equal(Argmax(refB), Argmax(batch[1]));
        Assert.True(overlapB >= 4, $"Seq B SnapKV batched top-5 {overlapB}/5 (maxAbs={maxAbsB}).");
        Assert.True(maxAbsB < 1.0f, $"Seq B SnapKV batched maxAbs={maxAbsB}.");
    }

    /// <summary>
    /// Option 1 mixed batch: one SnapKV-evicted sequence (long prompt) and one un-evicted sequence
    /// (short prompt below the budget) in the SAME <see cref="CudaForwardPass.BatchForwardMulti"/>
    /// call. The forced per-sequence loop must apply <c>physSlot = pos - EvictedCount</c> per cache
    /// — the evicted one offset, the un-evicted one at <c>pos</c> — and both must match their
    /// single-user references.
    /// </summary>
    [Fact]
    public void Qwen3_8B_SnapKvBatchedDecode_MixedEvictedAndNot()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        using var env = SnapKvEnv(budget, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);

        int[] longPrompt = LongPrompt(tokenizer, 900);          // > budget → evicts
        int[] shortPrompt = { 9707, 11, 1879, 0, 358, 1079 };   // << budget → no eviction

        fwd.ResetCache();
        int tokL = Argmax(fwd.Prefill(longPrompt));
        float[] refL = fwd.Forward(tokL, longPrompt.Length).ToArray();
        fwd.ResetCache();
        int tokS = Argmax(fwd.Prefill(shortPrompt));
        float[] refS = fwd.Forward(tokS, shortPrompt.Length).ToArray();
        fwd.ResetCache();

        using var cacheL = fwd.CreateCache();
        using var cacheS = fwd.CreateCache();
        int tokL2 = Argmax(fwd.PrefillWithCache(longPrompt, cacheL));
        int tokS2 = Argmax(fwd.PrefillWithCache(shortPrompt, cacheS));
        Assert.True(cacheL.EvictedCount > 0, "long prompt should have evicted.");
        Assert.Equal(0, cacheS.EvictedCount);   // short prompt below budget — never evicted
        Assert.Equal(shortPrompt.Length, cacheS.Length);

        float[][] batch = fwd.BatchForwardMulti(
            [tokL2, tokS2], [longPrompt.Length, shortPrompt.Length], [cacheL, cacheS]);
        Assert.Equal(2, batch.Length);

        var (maxAbsL, overlapL) = Compare(refL, batch[0]);
        Assert.Equal(Argmax(refL), Argmax(batch[0]));
        Assert.True(overlapL >= 4, $"Evicted seq top-5 {overlapL}/5 (maxAbs={maxAbsL}).");
        Assert.True(maxAbsL < 1.0f, $"Evicted seq maxAbs={maxAbsL}.");

        var (maxAbsS, overlapS) = Compare(refS, batch[1]);
        Assert.Equal(Argmax(refS), Argmax(batch[1]));
        Assert.True(overlapS >= 4, $"Un-evicted seq top-5 {overlapS}/5 (maxAbs={maxAbsS}).");
        Assert.True(maxAbsS < 1.0f, $"Un-evicted seq maxAbs={maxAbsS}.");
    }

    /// <summary>
    /// Issue #277: a SnapKV-evicted batch now stays on the #197 ragged fast path (the
    /// <c>anyEvicted</c> → per-sequence-loop forcing is gone). The ragged KV-append + attention
    /// take the PHYSICAL slot <c>pos - EvictedCount</c> while RoPE keeps the logical position — so
    /// the ragged-evicted decode must be BIT-IDENTICAL to the per-sequence-loop decode it replaces.
    ///
    /// A batched decode step is idempotent on the same caches at the same positions (it overwrites
    /// its own next slot with identical K/V and bounds attention by <c>pos - EvictedCount</c>, never
    /// by <c>Length</c>), and both paths share every GEMM — so running one instance's ragged path
    /// then flipping to the per-sequence loop on the SAME caches isolates the attention block and
    /// asserts exact equality. Two different-length prompts give two different eviction deltas, so a
    /// physical-slot mis-index would diverge here. (A cross-instance compare can't assert bit-
    /// identity — prefill isn't bit-exact across instances.)
    /// </summary>
    [Fact]
    public void Qwen3_8B_RaggedEvictedDecode_BitIdentical_To_PerSequenceLoop_N2()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        using var env = SnapKvEnv(budget, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);
        Assert.True(fwd.SnapKvEnabled);
        Assert.True(fwd.BatchDecodeRaggedForTest, "ragged decode must be enabled to exercise #277.");

        int[] promptA = LongPrompt(tokenizer, 600);
        int[] promptB = LongPrompt(tokenizer, 1000);

        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        int tokA = Argmax(fwd.PrefillWithCache(promptA, cacheA));
        int tokB = Argmax(fwd.PrefillWithCache(promptB, cacheB));
        Assert.True(cacheA.EvictedCount > 0 && cacheB.EvictedCount > 0);
        Assert.NotEqual(cacheA.EvictedCount, cacheB.EvictedCount); // different deltas

        int[] toks = { tokA, tokB };
        int[] poss = { promptA.Length, promptB.Length };

        // Ragged path (default): handles eviction via the physical-slot array (#277).
        fwd.BatchDecodeRaggedForTest = true;
        float[][] ragged = fwd.BatchForwardMulti(toks, poss, [cacheA, cacheB]);

        // Per-sequence loop (#190) on the SAME caches at the SAME positions — idempotent re-run.
        fwd.BatchDecodeRaggedForTest = false;
        float[][] perSeq = fwd.BatchForwardMulti(toks, poss, [cacheA, cacheB]);

        for (int n = 0; n < 2; n++)
        {
            float maxAbs = MaxAbs(ragged[n], perSeq[n]);
            Assert.Equal(Argmax(perSeq[n]), Argmax(ragged[n]));
            Assert.True(maxAbs == 0f,
                $"Seq {n}: ragged-evicted decode must be bit-identical to the per-sequence loop " +
                $"(maxAbs={maxAbs}); a nonzero delta means the physical-slot threading diverged.");
        }
    }

    /// <summary>
    /// Issue #277, mixed batch: one SnapKV-evicted sequence and one un-evicted sequence in the same
    /// ragged decode. The physical-slot array must offset only the evicted one (<c>pos - delta</c>)
    /// and leave the un-evicted one at <c>pos</c>; the ragged path must still be bit-identical to the
    /// per-sequence loop for BOTH. Guards the per-sequence <c>slots[n]</c> construction.
    /// </summary>
    [Fact]
    public void Qwen3_8B_RaggedEvictedDecode_BitIdentical_To_PerSequenceLoop_MixedEvictedAndNot()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 512;
        using var env = SnapKvEnv(budget, window: 32);
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048);

        int[] longPrompt = LongPrompt(tokenizer, 900);          // > budget → evicts
        int[] shortPrompt = { 9707, 11, 1879, 0, 358, 1079 };   // << budget → no eviction

        using var cacheL = fwd.CreateCache();
        using var cacheS = fwd.CreateCache();
        int tokL = Argmax(fwd.PrefillWithCache(longPrompt, cacheL));
        int tokS = Argmax(fwd.PrefillWithCache(shortPrompt, cacheS));
        Assert.True(cacheL.EvictedCount > 0);
        Assert.Equal(0, cacheS.EvictedCount);

        int[] toks = { tokL, tokS };
        int[] poss = { longPrompt.Length, shortPrompt.Length };

        fwd.BatchDecodeRaggedForTest = true;
        float[][] ragged = fwd.BatchForwardMulti(toks, poss, [cacheL, cacheS]);
        fwd.BatchDecodeRaggedForTest = false;
        float[][] perSeq = fwd.BatchForwardMulti(toks, poss, [cacheL, cacheS]);

        for (int n = 0; n < 2; n++)
        {
            float maxAbs = MaxAbs(ragged[n], perSeq[n]);
            Assert.Equal(Argmax(perSeq[n]), Argmax(ragged[n]));
            Assert.True(maxAbs == 0f,
                $"Seq {n} (mixed): ragged decode must be bit-identical to the per-sequence loop (maxAbs={maxAbs}).");
        }
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    /// <summary>
    /// Option 2: when continuous batching is preferred (<c>preferBatchingOverAutoSnapKv</c>), the
    /// VRAM-scaled SnapKV AUTO-enable is suppressed — so a batching server gets the ragged-decode
    /// fast path, not silent lossy eviction. Verified at a context where auto-SnapKV WOULD engage.
    /// </summary>
    [Fact]
    public void Qwen3_8B_PreferBatching_SuppressesAutoSnapKv()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        // Unset the explicit budget so the auto-enable path is exercised.
        using var env = new EnvScope(
            ("SHARPI_SNAPKV_BUDGET", null),
            ("SHARPI_PREFIX_SLOTS", null),
            ("SHARPI_KV_DTYPE", null));
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // ctx 4096 makes the fp32 KV cache exceed the auto-enable threshold (and maxSeqLen/4 lands
        // in the auto-budget band), so auto-SnapKV engages by default.
        bool autoEngaged;
        using (var autoFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 4096,
            preferBatchingOverAutoSnapKv: false))
            autoEngaged = autoFwd.SnapKvEnabled;

        using (var preferFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 4096,
            preferBatchingOverAutoSnapKv: true))
            Assert.False(preferFwd.SnapKvEnabled,
                "preferBatchingOverAutoSnapKv should suppress the SnapKV auto-enable (#196 Option 2).");

        Assert.True(autoEngaged,
            "expected auto-SnapKV to engage at ctx 4096 for Qwen3-8B; if not, the suppression test " +
            "needs a larger context to be meaningful.");
    }
}
