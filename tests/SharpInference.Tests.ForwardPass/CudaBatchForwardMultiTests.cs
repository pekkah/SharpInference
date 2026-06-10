using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #190: CUDA continuous batching on the dense path. <see cref="CudaForwardPass"/> now
/// implements <see cref="IBatchedForwardPass"/> — the continuous-batching engine drives
/// per-sequence GPU KV caches (<see cref="CudaSequenceKvCache"/>) through prefill + true
/// batched decode. These oracles validate that the batched-serving entry points agree with
/// the trusted single-user <see cref="CudaForwardPass.Prefill"/> / <see cref="CudaForwardPass.Forward"/>
/// loop on a dense <b>Qwen3-8B Q4_K</b> (QK-norm, NEOX RoPE, SwiGLU, no PLE/SWA/MoE).
///
/// One ~5 GB model instance per test (a second instance would double VRAM beyond 12 GB), so
/// the single-user reference runs first (with a <c>ResetCache</c> between sequences) and the
/// batched run follows. Silent-skips when CUDA or the GGUF is absent — mirrors
/// <see cref="Qwen3CudaBatchedPrefillTests"/>.
/// </summary>
public sealed class CudaBatchForwardMultiTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    // Two distinct ≥2-token prompts (so prefill takes the batched trunk, not the N==1
    // per-token Forward fallback). Ordinary Qwen3 vocab ids.
    private static readonly int[] PromptA = { 9707, 11, 1879, 0, 358 };
    private static readonly int[] PromptB = { 1079, 264, 4108, 1614, 13, 220, 17 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // Construct with SnapKV pinned off: continuous batching is unsupported under an active
    // SnapKV budget (it throws), and VRAM-scaled auto-SnapKV could otherwise engage at this
    // context on a smaller GPU and make these oracles non-deterministic across machines.
    private static CudaForwardPass NewFwd(GgufModel model, CudaBackend gpu, ModelHyperparams hp, int ctx = 512)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev); }
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

    private static (float maxAbs, int overlap) Compare(float[] reference, float[] candidate)
    {
        Assert.Equal(reference.Length, candidate.Length);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - candidate[i]));
        var refTop = TopKSet(reference, 5);
        var candTop = TopKSet(candidate, 5);
        int overlap = 0;
        foreach (var t in candTop) if (refTop.Contains(t)) overlap++;
        return (maxAbs, overlap);
    }

    /// <summary>
    /// Headline #190 oracle: a 2-sequence <see cref="CudaForwardPass.BatchForwardMulti"/>
    /// decode step (each sequence at its own position against its own per-sequence cache,
    /// weight reads amortized 2×) must reproduce the next-token logits of two independent
    /// single-user prefill+decode passes. Same backend / dtype, so argmax must match and the
    /// batched GEMM-N reassociation stays well within the cross-path tolerance (the contract
    /// the prefill batched-trunk oracles hold).
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchForwardMulti_N2_MatchesSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        // Precondition: dense, non-Gemma, non-MoE — the supported batching path.
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);

        // ── Single-user reference: prefill each prompt, greedy-decode one token, decode it. ──
        fwd.ResetCache();
        int tokA = Argmax(fwd.Prefill(PromptA));
        float[] refLogitsA = fwd.Forward(tokA, PromptA.Length).ToArray();

        fwd.ResetCache();
        int tokB = Argmax(fwd.Prefill(PromptB));
        float[] refLogitsB = fwd.Forward(tokB, PromptB.Length).ToArray();

        // ── Batched: prefill two per-sequence caches, then one batched decode step. ──
        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(PromptA, cacheA);
        fwd.PrefillWithCache(PromptB, cacheB);

        float[][] batch = fwd.BatchForwardMulti(
            [tokA, tokB],
            [PromptA.Length, PromptB.Length],
            [cacheA, cacheB]);

        Assert.Equal(2, batch.Length);

        var (maxAbsA, overlapA) = Compare(refLogitsA, batch[0]);
        Assert.Equal(Argmax(refLogitsA), Argmax(batch[0]));
        Assert.True(overlapA >= 4,
            $"Seq A batched top-5 overlaps the single-user reference in only {overlapA}/5 slots (maxAbs={maxAbsA}).");
        Assert.True(maxAbsA < 1.0f,
            $"Seq A batched vs single-user logits diverged beyond tolerance: maxAbs={maxAbsA}.");

        var (maxAbsB, overlapB) = Compare(refLogitsB, batch[1]);
        Assert.Equal(Argmax(refLogitsB), Argmax(batch[1]));
        Assert.True(overlapB >= 4,
            $"Seq B batched top-5 overlaps the single-user reference in only {overlapB}/5 slots (maxAbs={maxAbsB}).");
        Assert.True(maxAbsB < 1.0f,
            $"Seq B batched vs single-user logits diverged beyond tolerance: maxAbs={maxAbsB}.");
    }

    /// <summary>
    /// A second batched decode step (positions advance by one) must still track the single-user
    /// continuation — catches a per-sequence cache-append / position-indexing bug that a single
    /// decode step would miss (the first step's KV is reused, a second token is appended).
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchForwardMulti_TwoDecodeSteps_MatchSingleUser()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = NewFwd(model, gpu, hp);

        // Single-user reference: two greedy decode steps for each prompt.
        fwd.ResetCache();
        int a0 = Argmax(fwd.Prefill(PromptA));
        int a1 = Argmax(fwd.Forward(a0, PromptA.Length));
        float[] refA = fwd.Forward(a1, PromptA.Length + 1).ToArray();

        fwd.ResetCache();
        int b0 = Argmax(fwd.Prefill(PromptB));
        int b1 = Argmax(fwd.Forward(b0, PromptB.Length));
        float[] refB = fwd.Forward(b1, PromptB.Length + 1).ToArray();

        // Batched: prefill, then two decode steps; sample greedily between steps to follow
        // the same trajectory as the single-user reference.
        using var cacheA = fwd.CreateCache();
        using var cacheB = fwd.CreateCache();
        fwd.PrefillWithCache(PromptA, cacheA);
        fwd.PrefillWithCache(PromptB, cacheB);

        var step1 = fwd.BatchForwardMulti([a0, b0], [PromptA.Length, PromptB.Length], [cacheA, cacheB]);
        int ba1 = Argmax(step1[0]);
        int bb1 = Argmax(step1[1]);
        Assert.Equal(a1, ba1);
        Assert.Equal(b1, bb1);

        var step2 = fwd.BatchForwardMulti([ba1, bb1], [PromptA.Length + 1, PromptB.Length + 1], [cacheA, cacheB]);

        var (maxAbsA, overlapA) = Compare(refA, step2[0]);
        Assert.Equal(Argmax(refA), Argmax(step2[0]));
        Assert.True(overlapA >= 4, $"Seq A 2nd-step top-5 overlap {overlapA}/5 (maxAbs={maxAbsA}).");
        Assert.True(maxAbsA < 1.0f, $"Seq A 2nd-step maxAbs={maxAbsA}.");

        var (maxAbsB, overlapB) = Compare(refB, step2[1]);
        Assert.Equal(Argmax(refB), Argmax(step2[1]));
        Assert.True(overlapB >= 4, $"Seq B 2nd-step top-5 overlap {overlapB}/5 (maxAbs={maxAbsB}).");
        Assert.True(maxAbsB < 1.0f, $"Seq B 2nd-step maxAbs={maxAbsB}.");
    }

    /// <summary>
    /// <see cref="CudaForwardPass.PrefillWithCache"/> into a per-sequence
    /// <see cref="CudaSequenceKvCache"/> must produce the same final-token logits as the
    /// single-user <see cref="CudaForwardPass.Prefill"/> into the owned cache — identical
    /// kernels, only the cache pointers differ, so this should be near bit-identical.
    /// </summary>
    [Fact]
    public void Qwen3_8B_PrefillWithCache_MatchesSingleUserPrefill()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = NewFwd(model, gpu, hp);

        fwd.ResetCache();
        float[] reference = fwd.Prefill(PromptA).ToArray();

        using var cache = fwd.CreateCache();
        float[] viaCache = fwd.PrefillWithCache(PromptA, cache).ToArray();

        Assert.Equal(reference.Length, viaCache.Length);
        Assert.Equal(Argmax(reference), Argmax(viaCache));
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - viaCache[i]));
        Assert.True(maxAbs < 1e-2f,
            $"PrefillWithCache (per-sequence cache) diverged from single-user Prefill: maxAbs={maxAbs}.");

        // The cache's logical length must equal the prompt length after prefill.
        Assert.Equal(PromptA.Length, cache.Length);
    }

    /// <summary>
    /// Chunked <see cref="CudaForwardPass.PrefillWithCache"/> (advancing <c>startPos</c>) into
    /// one per-sequence cache must match a single whole-prompt prefill — the path the engine's
    /// chunked admission drives. Validates that prior-chunk KV is read correctly by later chunks.
    /// </summary>
    [Fact]
    public void Qwen3_8B_PrefillWithCache_Chunked_MatchesFull()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        int[] prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13, 220, 17, 18, 19 };

        using var fwd = NewFwd(model, gpu, hp);

        using var refCache = fwd.CreateCache();
        float[] reference = fwd.PrefillWithCache(prompt, refCache).ToArray();

        using var cache = fwd.CreateCache();
        float[] chunked = Array.Empty<float>();
        const int chunk = 4;
        for (int start = 0; start < prompt.Length; start += chunk)
        {
            int len = Math.Min(chunk, prompt.Length - start);
            var segment = new ArraySegment<int>(prompt, start, len);
            chunked = fwd.PrefillWithCache(segment, cache, startPos: start).ToArray();
        }

        Assert.Equal(prompt.Length, cache.Length);
        Assert.Equal(Argmax(reference), Argmax(chunked));
        var (maxAbs, overlap) = Compare(reference, chunked);
        Assert.True(overlap >= 4, $"Chunked-vs-full top-5 overlap {overlap}/5 (maxAbs={maxAbs}).");
        Assert.True(maxAbs < 1.0f, $"Chunked-vs-full prefill maxAbs={maxAbs}.");
    }

    /// <summary>Empty token list and empty batch are rejected / no-op, matching the CPU path.</summary>
    [Fact]
    public void Qwen3_8B_BatchForwardMulti_EmptyBatch_ReturnsEmpty()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = NewFwd(model, gpu, hp);

        Assert.Empty(fwd.BatchForwardMulti([], [], []));

        using var cache = fwd.CreateCache();
        Assert.Throws<ArgumentException>(() => fwd.PrefillWithCache([], cache));
    }
}
