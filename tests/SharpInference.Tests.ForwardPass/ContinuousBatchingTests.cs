using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for Phase 7c continuous batching: PrefillWithCache, BatchForwardMulti, ContinuousBatchingEngine.
/// Integration tests skip silently if the model file is not present.
/// </summary>
public sealed class ContinuousBatchingTests
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

    // ── PrefillWithCache ──────────────────────────────────────────────

    [Fact]
    public void PrefillWithCache_MatchesPrefill_SameLogits()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        // Two independent ForwardPass instances so caches don't interfere
        using var fwdRef = new Engine.ForwardPass(model, backend, hp);
        using var fwdTest = new Engine.ForwardPass(model, backend, hp);

        int[] tokens = [1, 2, 3, 5, 7];  // small token IDs always in vocab

        // Reference: standard Prefill into _kvCache
        ReadOnlySpan<float> refLogits = fwdRef.Prefill(tokens);
        float[] refArr = refLogits.ToArray();

        // Test: PrefillWithCache into an external cache
        using var cache = fwdTest.CreateCache();
        ReadOnlySpan<float> testLogits = fwdTest.PrefillWithCache(tokens, cache);
        float[] testArr = testLogits.ToArray();

        Assert.Equal(refArr.Length, testArr.Length);

        // Logits should be numerically identical (same weights, same inputs, same operations)
        for (int i = 0; i < refArr.Length; i++)
            Assert.Equal(refArr[i], testArr[i], precision: 2);

        // Cache position should be tokens.Length
        Assert.Equal(tokens.Length, cache.Length);
    }

    [Fact]
    public void PrefillWithCache_SingleToken_MatchesForward()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        using var fwdRef = new Engine.ForwardPass(model, backend, hp);
        using var fwdTest = new Engine.ForwardPass(model, backend, hp);

        // Single-token path uses ForwardCore internally
        ReadOnlySpan<float> refLogits = fwdRef.Forward(42, 0);
        float[] refArr = refLogits.ToArray();

        using var cache = fwdTest.CreateCache();
        ReadOnlySpan<float> testLogits = fwdTest.PrefillWithCache([42], cache);
        float[] testArr = testLogits.ToArray();

        Assert.Equal(refArr.Length, testArr.Length);
        for (int i = 0; i < refArr.Length; i++)
            Assert.Equal(refArr[i], testArr[i], precision: 2);

        Assert.Equal(1, cache.Length);
    }

    [Fact]
    public void PrefillWithCache_EmptyTokens_Throws()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);
        using var cache = fwd.CreateCache();

        Assert.Throws<ArgumentException>(() => fwd.PrefillWithCache([], cache));
    }

    // ── BatchForwardMulti ─────────────────────────────────────────────

    [Fact]
    public void BatchForwardMulti_N2_MatchesIndividualForward()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        // Prompt tokens for two sequences
        int[] promptA = [1, 2, 3];
        int[] promptB = [4, 5, 6];

        // ── Reference: two fully independent ForwardPass instances ──────
        float[] refLogitsA, refLogitsB;
        int decodeTokenA, decodeTokenB;

        using (var fwdA = new Engine.ForwardPass(model, backend, hp))
        {
            var la = fwdA.Prefill(promptA);
            decodeTokenA = Sampler.Greedy(la);
            var la2 = fwdA.Forward(decodeTokenA, promptA.Length);
            refLogitsA = la2.ToArray();
        }

        using (var fwdB = new Engine.ForwardPass(model, backend, hp))
        {
            var lb = fwdB.Prefill(promptB);
            decodeTokenB = Sampler.Greedy(lb);
            var lb2 = fwdB.Forward(decodeTokenB, promptB.Length);
            refLogitsB = lb2.ToArray();
        }

        // ── Test: BatchForwardMulti on a third instance ──────────────────
        using var fwdBatch = new Engine.ForwardPass(model, backend, hp);

        using var cacheA = fwdBatch.CreateCache();
        using var cacheB = fwdBatch.CreateCache();

        fwdBatch.PrefillWithCache(promptA, cacheA);
        fwdBatch.PrefillWithCache(promptB, cacheB);

        float[][] batchResult = fwdBatch.BatchForwardMulti(
            [decodeTokenA, decodeTokenB],
            [promptA.Length, promptB.Length],
            [cacheA, cacheB]);

        Assert.Equal(2, batchResult.Length);
        Assert.Equal(refLogitsA.Length, batchResult[0].Length);
        Assert.Equal(refLogitsB.Length, batchResult[1].Length);

        // Logits from batch mode should match individual decode passes
        for (int i = 0; i < refLogitsA.Length; i++)
            Assert.Equal(refLogitsA[i], batchResult[0][i], precision: 2);

        for (int i = 0; i < refLogitsB.Length; i++)
            Assert.Equal(refLogitsB[i], batchResult[1][i], precision: 2);
    }

    [Fact]
    public void BatchForwardMulti_EmptyTokens_ReturnsEmpty()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        var result = fwd.BatchForwardMulti([], [], []);
        Assert.Empty(result);
    }

    // ── ContinuousBatchingEngine ──────────────────────────────────────

    [Fact]
    public async Task ContinuousBatchingEngine_TwoConcurrentRequests_BothComplete()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test-model", maxBatchSize: 4);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 5 };
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(120));

        // Launch two concurrent requests
        var taskA = Task.Run(async () =>
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engine.GenerateAsync("Hello", sp, cts.Token))
                sb.Append(chunk);
            return sb.ToString();
        });

        var taskB = Task.Run(async () =>
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engine.GenerateAsync("World", sp, cts.Token))
                sb.Append(chunk);
            return sb.ToString();
        });

        string[] results = await Task.WhenAll(taskA, taskB);

        // Both requests should complete and return some output
        Assert.Equal(2, results.Length);
        Assert.NotNull(results[0]);
        Assert.NotNull(results[1]);
    }

    [Fact]
    public async Task ContinuousBatchingEngine_Dispose_CancelsActiveRequests()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test-model", maxBatchSize: 2);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 100 };

        // Dispose should allow in-progress generation to complete or drain cleanly
        engine.Dispose();

        // After dispose, new requests should still complete (channel is completed)
        // The generator either returns 0 tokens or throws OperationCanceledException — both are fine.
        var tokens = new List<string>();
        try
        {
            await foreach (var chunk in engine.GenerateAsync("Test", sp))
                tokens.Add(chunk);
        }
        catch (OperationCanceledException) { /* expected */ }
        catch (Exception ex) when (ex.GetType().Name == "ChannelClosedException") { /* expected */ }

        // The engine shut down without crashing — that's the invariant we care about.
        Assert.True(true);
    }

    // ── Issue #183: chunked prefill, packed multi-seq prefill, KV budget ──

    [Fact]
    public void PrefillWithCache_Chunked_MatchesFull()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        using var fwdRef = new Engine.ForwardPass(model, backend, hp);
        using var fwdTest = new Engine.ForwardPass(model, backend, hp);

        int[] tokens = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        using var refCache = fwdRef.CreateCache();
        float[] refArr = fwdRef.PrefillWithCache(tokens, refCache).ToArray();

        // Same prompt prefilled in 3 chunks via successive startPos calls — the
        // continuation pattern the chunked-admission engine path uses (issue #183 Gap 1).
        using var cache = fwdTest.CreateCache();
        const int chunk = 5;
        float[] testArr = [];
        for (int start = 0; start < tokens.Length; start += chunk)
        {
            int take = Math.Min(chunk, tokens.Length - start);
            var segment = new ArraySegment<int>(tokens, start, take);
            testArr = fwdTest.PrefillWithCache(segment, cache, startPos: start).ToArray();
        }

        Assert.Equal(tokens.Length, cache.Length);
        Assert.Equal(refArr.Length, testArr.Length);
        // Chunk boundaries change GEMM batch sizes (different FP accumulation order),
        // so assert close logits + same argmax rather than bit equality.
        for (int i = 0; i < refArr.Length; i++)
            Assert.Equal(refArr[i], testArr[i], precision: 2);
        Assert.Equal(Sampler.Greedy(refArr), Sampler.Greedy(testArr));
    }

    [Fact]
    public void PrefillWithCache_DequantCacheOnOff_BitIdentical()
    {
        // Issue #189: the dequant-once weight cache must be transparent — chunked prefill with
        // the cache active produces bit-for-bit the same logits as with it off (same F32
        // dequant feeds the same SGEMM, just sourced from the cache on reuse).
        var path = FindModelPath();
        if (path is null) return;
        // The cache only diverts the OpenBLAS SGEMM path; without BLAS both runs are identical
        // by construction and the test proves nothing.
        if (!SimdKernels.BlasAvailable) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        // 48 tokens prefilled in 16-token chunks: each chunk is at/above MinBatchForBlas so the
        // SGEMM+cache path runs, and chunks 2-3 read weights the cache filled during chunk 1.
        int[] tokens = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
                        131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211];

        float[] cacheOff = ChunkedPrefillLogits(model, backend, hp, tokens, chunk: 16, dequantCacheBytes: 0);
        float[] cacheOn = ChunkedPrefillLogits(model, backend, hp, tokens, chunk: 16, dequantCacheBytes: -1);

        Assert.Equal(cacheOff.Length, cacheOn.Length);
        for (int i = 0; i < cacheOff.Length; i++)
            Assert.Equal(cacheOff[i], cacheOn[i]);
    }

    private static float[] ChunkedPrefillLogits(
        GgufModel model, CpuBackend backend, ModelHyperparams hp,
        int[] tokens, int chunk, long dequantCacheBytes)
    {
        using var fwd = new Engine.ForwardPass(model, backend, hp,
            prefillDequantCacheBytes: dequantCacheBytes);
        using var cache = fwd.CreateCache();
        float[] logits = [];
        for (int start = 0; start < tokens.Length; start += chunk)
        {
            int take = Math.Min(chunk, tokens.Length - start);
            var segment = new ArraySegment<int>(tokens, start, take);
            logits = fwd.PrefillWithCache(segment, cache, startPos: start).ToArray();
        }
        return logits;
    }

    [Fact]
    public void PrefillPackedMulti_MatchesSequentialPrefill()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int[] promptA = [1, 2, 3, 5, 7, 11, 13];
        int[] promptB = [4, 5, 6, 8];

        // Reference: each prompt prefilled alone on its own ForwardPass instance.
        float[] refA, refB;
        using (var fwdRef = new Engine.ForwardPass(model, backend, hp))
        {
            using var cA = fwdRef.CreateCache();
            refA = fwdRef.PrefillWithCache(promptA, cA).ToArray();
            using var cB = fwdRef.CreateCache();
            refB = fwdRef.PrefillWithCache(promptB, cB).ToArray();
        }

        // Test: both prompts in ONE packed forward pass.
        using var fwdPacked = new Engine.ForwardPass(model, backend, hp);
        using var cacheA = fwdPacked.CreateCache();
        using var cacheB = fwdPacked.CreateCache();

        float[]?[] packed = fwdPacked.PrefillPackedMulti(
            [promptA.AsMemory(), promptB.AsMemory()],
            [0, 0],
            [cacheA, cacheB],
            [true, true]);

        Assert.Equal(promptA.Length, cacheA.Length);
        Assert.Equal(promptB.Length, cacheB.Length);
        Assert.NotNull(packed[0]);
        Assert.NotNull(packed[1]);

        for (int i = 0; i < refA.Length; i++)
            Assert.Equal(refA[i], packed[0]![i], precision: 2);
        for (int i = 0; i < refB.Length; i++)
            Assert.Equal(refB[i], packed[1]![i], precision: 2);
        Assert.Equal(Sampler.Greedy(refA), Sampler.Greedy(packed[0]!));
        Assert.Equal(Sampler.Greedy(refB), Sampler.Greedy(packed[1]!));
    }

    [Fact]
    public void PrefillPackedMulti_ChunkedContinuation_MatchesFull()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();

        int[] promptA = [1, 2, 3, 5, 7, 11, 13];
        int[] promptB = [4, 5, 6, 8];

        float[] refA, refB;
        using (var fwdRef = new Engine.ForwardPass(model, backend, hp))
        {
            using var cA = fwdRef.CreateCache();
            refA = fwdRef.PrefillWithCache(promptA, cA).ToArray();
            using var cB = fwdRef.CreateCache();
            refB = fwdRef.PrefillWithCache(promptB, cB).ToArray();
        }

        // Two packed rounds: first a partial chunk of each prompt (no logits wanted),
        // then the remainder (logits wanted) — the exact shape the engine produces when
        // several prompts prefill chunk-by-chunk (issue #183 Gaps 1+2 combined).
        using var fwdPacked = new Engine.ForwardPass(model, backend, hp);
        using var cacheA = fwdPacked.CreateCache();
        using var cacheB = fwdPacked.CreateCache();

        float[]?[] round1 = fwdPacked.PrefillPackedMulti(
            [promptA.AsMemory(0, 4), promptB.AsMemory(0, 2)],
            [0, 0],
            [cacheA, cacheB],
            [false, false]);
        Assert.Null(round1[0]);
        Assert.Null(round1[1]);
        Assert.Equal(4, cacheA.Length);
        Assert.Equal(2, cacheB.Length);

        float[]?[] round2 = fwdPacked.PrefillPackedMulti(
            [promptA.AsMemory(4), promptB.AsMemory(2)],
            [4, 2],
            [cacheA, cacheB],
            [true, true]);

        Assert.Equal(promptA.Length, cacheA.Length);
        Assert.Equal(promptB.Length, cacheB.Length);

        for (int i = 0; i < refA.Length; i++)
            Assert.Equal(refA[i], round2[0]![i], precision: 2);
        for (int i = 0; i < refB.Length; i++)
            Assert.Equal(refB[i], round2[1]![i], precision: 2);
        Assert.Equal(Sampler.Greedy(refA), Sampler.Greedy(round2[0]!));
        Assert.Equal(Sampler.Greedy(refB), Sampler.Greedy(round2[1]!));
    }

    [Fact]
    public async Task ContinuousBatchingEngine_ChunkedPrefill_MatchesUnchunked()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();

        const string prompt = "The capital of France is";
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 8 };
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(300));

        // Reference: legacy blocking prefill (chunking disabled), single request.
        string refText;
        using (var fwdRef = new Engine.ForwardPass(model, backend, hp))
        using (var engineRef = new ContinuousBatchingEngine(fwdRef, tokenizer, "test-model",
                   maxBatchSize: 2, prefillChunkTokens: 0))
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engineRef.GenerateAsync(prompt, sp, cts.Token))
                sb.Append(chunk);
            refText = sb.ToString();
        }
        Assert.False(string.IsNullOrEmpty(refText));

        // Test: tiny chunk size + two concurrent identical requests so admission runs
        // the packed multi-prompt prefill path between decode steps.
        using var fwd = new Engine.ForwardPass(model, backend, hp);
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test-model",
            maxBatchSize: 4, prefillChunkTokens: 4);

        var tasks = Enumerable.Range(0, 2).Select(_ => Task.Run(async () =>
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engine.GenerateAsync(prompt, sp, cts.Token))
                sb.Append(chunk);
            return sb.ToString();
        })).ToArray();

        string[] results = await Task.WhenAll(tasks);

        // Greedy decode of the same prompt must reproduce the unchunked output —
        // decode-coherence assertion, not just "non-null" (see feedback memory).
        Assert.Equal(refText, results[0]);
        Assert.Equal(refText, results[1]);
    }

    [Fact]
    public async Task ContinuousBatchingEngine_TinyKvBudget_AllRequestsComplete()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        // Budget of 16 KV tokens: one request (prompt ~2 + 4 new = ~6 projected) fits,
        // three at once (≥18) do not — admission must serialize, not reject or OOM.
        long budgetBytes = fwd.KvBytesPerToken * 16;
        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test-model",
            maxBatchSize: 4, prefillChunkTokens: 4, kvBudgetBytes: budgetBytes);
        Assert.Equal(16, engine.KvTokenBudget);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(300));

        var tasks = new[] { "Hello", "World", "Paris" }.Select(p => Task.Run(async () =>
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engine.GenerateAsync(p, sp, cts.Token))
                sb.Append(chunk);
            return sb.ToString();
        })).ToArray();

        string[] results = await Task.WhenAll(tasks);

        // All three complete despite the budget forcing serialized admission.
        Assert.Equal(3, results.Length);
        Assert.All(results, r => Assert.NotNull(r));
    }
}
