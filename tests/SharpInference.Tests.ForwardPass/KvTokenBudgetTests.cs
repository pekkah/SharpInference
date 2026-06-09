using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for the KV-cache admission budget (issue #183, gap 3): the
/// <see cref="HardwareProfile.EstimateKvTokenBudget"/> autotune helper,
/// <see cref="Engine.ForwardPass.KvBytesPerToken"/>, and the token-budget
/// backpressure in <see cref="ContinuousBatchingEngine"/>.
///
/// The pure-math tests run everywhere; the engine integration tests skip
/// silently when the model file is absent (same convention as
/// <see cref="ContinuousBatchingTests"/>).
/// </summary>
public sealed class KvTokenBudgetTests
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

    private static HardwareProfile Profile(long ramBytes) =>
        new(VramBytes: 0, RamBytes: ramBytes, CpuCores: 8,
            EstPcieBandwidthGBps: 0, HasAvx512: false);

    // ── EstimateKvTokenBudget (pure math) ─────────────────────────────

    [Fact]
    public void EstimateKvTokenBudget_NormalCase_HalvesAvailableRam()
    {
        // 8 GiB RAM, 1 MiB/token, default 0.5 fraction → 4 GiB / 1 MiB = 4096 tokens.
        long ram = 8L * 1024 * 1024 * 1024;
        long perToken = 1L * 1024 * 1024;
        Assert.Equal(4096, Profile(ram).EstimateKvTokenBudget(perToken));
    }

    [Fact]
    public void EstimateKvTokenBudget_CustomFraction_Scales()
    {
        long ram = 8L * 1024 * 1024 * 1024;
        long perToken = 1L * 1024 * 1024;
        // 0.25 fraction → 2 GiB / 1 MiB = 2048 tokens.
        Assert.Equal(2048, Profile(ram).EstimateKvTokenBudget(perToken, memoryFraction: 0.25));
    }

    [Fact]
    public void EstimateKvTokenBudget_FractionClampedToCeiling()
    {
        long ram = 10L * 1024 * 1024 * 1024;
        long perToken = 1L * 1024 * 1024 * 1024; // 1 GiB/token
        // Fraction 5.0 clamps to 0.9 → 9 GiB / 1 GiB = 9 tokens.
        Assert.Equal(9, Profile(ram).EstimateKvTokenBudget(perToken, memoryFraction: 5.0));
    }

    [Fact]
    public void EstimateKvTokenBudget_NonPositiveInputs_ReturnZero()
    {
        long ram = 8L * 1024 * 1024 * 1024;
        Assert.Equal(0, Profile(ram).EstimateKvTokenBudget(0));          // no per-token cost
        Assert.Equal(0, Profile(ram).EstimateKvTokenBudget(-1));         // negative per-token cost
        Assert.Equal(0, Profile(0).EstimateKvTokenBudget(1024));         // no RAM figure
        Assert.Equal(0, Profile(ram).EstimateKvTokenBudget(1024, 0.0));  // zero fraction
        Assert.Equal(0, Profile(ram).EstimateKvTokenBudget(1024, -1.0)); // negative fraction clamps to 0
    }

    [Fact]
    public void EstimateKvTokenBudget_PerTokenLargerThanBudget_ReturnsZero()
    {
        long ram = 1L * 1024 * 1024 * 1024;
        long perToken = 2L * 1024 * 1024 * 1024; // single token costs more than all of RAM
        Assert.Equal(0, Profile(ram).EstimateKvTokenBudget(perToken));
    }

    // ── KvBytesPerToken (needs model) ─────────────────────────────────

    [Fact]
    public void KvBytesPerToken_MatchesPagedKvCacheLayout()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        long bytes = fwd.KvBytesPerToken;
        Assert.True(bytes > 0);
        // keys + values, fp32 → must be a multiple of 2 * sizeof(float) per layer.
        Assert.Equal(0, bytes % (2L * sizeof(float)));
        // A non-trivial autotune budget should fall out of a realistic RAM figure.
        Assert.True(Profile(16L * 1024 * 1024 * 1024).EstimateKvTokenBudget(bytes) > 0);
    }

    // ── ContinuousBatchingEngine budget backpressure (needs model) ────

    [Fact]
    public void ContinuousBatchingEngine_DefaultBudget_IsUnlimited()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        using var engine = new ContinuousBatchingEngine(fwd, tokenizer, "test-model", maxBatchSize: 4);
        Assert.Equal(0, engine.TokenBudget);
        Assert.Equal(0, engine.ActiveTokens);
    }

    [Fact]
    public async Task ContinuousBatchingEngine_TinyBudget_ThrottlesButCompletesAll()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        // Budget of 1 token forces serialization: with any sequence already active, a new
        // prompt always exceeds the ceiling and is parked until the active one retires.
        // The invariant under test is liveness — every request still completes, no deadlock.
        using var engine = new ContinuousBatchingEngine(
            fwd, tokenizer, "test-model", maxBatchSize: 4, maxActiveTokens: 1);

        Assert.Equal(1, engine.TokenBudget);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(180));

        async Task<string> Run(string prompt)
        {
            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in engine.GenerateAsync(prompt, sp, cts.Token))
                sb.Append(chunk);
            return sb.ToString();
        }

        string[] results = await Task.WhenAll(Run("Hello"), Run("World"), Run("Again"));

        Assert.Equal(3, results.Length);
        Assert.All(results, r => Assert.NotNull(r));
        // Budget fully reclaimed once every sequence retired.
        Assert.Equal(0, engine.ActiveTokens);
    }
}
