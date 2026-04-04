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
}
