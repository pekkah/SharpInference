using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Ad-hoc decode-throughput measurement for the CUDA backend, with and without TurboQuant
/// enabled. Skipped automatically when CUDA isn't available or the Qwen3-8B model isn't
/// present on disk. Not gated on a numeric threshold — its purpose is to surface t/s in
/// CI logs so regressions are easy to spot.
///
/// Run a single number with:
///   dotnet test tests/SharpInference.Tests.ForwardPass -c Release \
///     --filter "FullyQualifiedName~Decode" --logger "console;verbosity=normal"
/// </summary>
public sealed class CudaTurboQuantBench
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    private readonly ITestOutputHelper _out;

    public CudaTurboQuantBench(ITestOutputHelper outHelper) { _out = outHelper; }

    // Also tee to stderr so long-running tests stream progress in real time. xUnit
    // buffers ITestOutputHelper until the test method returns, which is unhelpful when
    // a single prefill takes 4 minutes.
    private void Log(string line)
    {
        _out.WriteLine(line);
        Console.Error.WriteLine(line);
    }

    // Opt-in via env var so this only fires when explicitly requested. xUnit's `Skip`
    // attribute can't be cleared with --filter, so we gate on the runtime environment
    // instead — set SHARPI_BENCH_CUDA_TQ=1 to actually run the timing.
    private static bool BenchEnabled => Environment.GetEnvironmentVariable("SHARPI_BENCH_CUDA_TQ") == "1";

    [Theory]
    [InlineData(16, 64)]
    [InlineData(512, 64)]
    [InlineData(1024, 64)]
    public void Decode_NoTq(int prefillTokens, int decodeTokens) => Measure(enableTq: false, prefillTokens, decodeTokens);

    [Theory]
    [InlineData(16, 64)]
    [InlineData(512, 64)]
    [InlineData(1024, 64)]
    [InlineData(4096, 64)]   // straddles the fast-path / recompute boundary
    [InlineData(8192, 64)]   // purely recompute path
    [InlineData(16384, 64)]  // long-context regime — the headline TQ use case
    public void Decode_Tq(int prefillTokens, int decodeTokens) => Measure(enableTq: true, prefillTokens, decodeTokens);

    [Theory]
    [InlineData(16, 64, false)]
    [InlineData(1024, 64, false)]
    [InlineData(16, 64, true)]
    [InlineData(1024, 64, true)]
    public void Decode_Vulkan(int prefillTokens, int decodeTokens, bool enableTq)
        => MeasureVulkan(prefillTokens, decodeTokens, enableTq);

    /// <summary>
    /// Reports the VRAM-derived max-context estimate for FP32 vs TurboQuant across
    /// both backends. This is the "how much extra room does TQ buy you?" measurement —
    /// useful when the kernel itself can't be exercised at the full estimate (the
    /// CUDA TQ attention shader caps stored scores at 4096).
    /// </summary>
    [Fact]
    public void Estimate_Context()
    {
        if (!BenchEnabled) { _out.WriteLine("Set SHARPI_BENCH_CUDA_TQ=1 to run; skipping."); return; }
        var path = FindModelPath();
        if (path is null) { _out.WriteLine($"{ModelFile} not found; skipping."); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        Log($"Model: {hp.NumLayers}L numKvHeads={hp.NumKvHeads} headDim={hp.HeadDim} modelMaxCtx={hp.ContextLength}");

        if (CudaBackend.IsAvailable())
        {
            using var cuda = CudaBackend.Create();
            int cudaFp32 = CudaForwardPass.EstimateMaxContext(model, cuda, hp);
            int cudaTq3  = CudaForwardPass.EstimateMaxContextTq(model, cuda, hp, fp32WindowSize: 256, bits: 3);
            Log($"CUDA  VRAM={cuda.VramBytes / (1024.0*1024*1024):F2} GiB | FP32 est={cudaFp32} | TQ3 est={cudaTq3} | ratio={(double)cudaTq3/cudaFp32:F2}×");
        }
        else
        {
            Log("CUDA unavailable.");
        }

        using var vk = new VulkanBackend();
        int vkFp32 = GetVkFp32Estimate(model, vk, hp);
        int vkTq3  = GpuForwardPass.EstimateMaxContextTq(model, vk, hp, fp32WindowSize: 256, bits: 3);
        Log($"Vulkan VRAM={vk.VramBytes / (1024.0*1024*1024):F2} GiB | FP32 est={vkFp32} | TQ3 est={vkTq3} | ratio={(double)vkTq3/vkFp32:F2}×");
    }

    // Vulkan's FP32 estimator is private; pull it out by constructing a forward pass with
    // maxContextLength=0 and reading back MaxSeqLen. The forward-pass construction also
    // uploads the weights, so this is more expensive than the CUDA static-method call,
    // but it's the only way to get the Vulkan estimator's answer without changing visibility.
    private static int GetVkFp32Estimate(GgufModel model, VulkanBackend vk, ModelHyperparams hp)
    {
        using var fwd = new GpuForwardPass(model, vk, hp, maxContextLength: 0, enableTurboQuant: false);
        return fwd.MaxSeqLen;
    }

    private void MeasureVulkan(int prefillTokens, int decodeTokens, bool enableTq)
    {
        if (!BenchEnabled) { _out.WriteLine("Set SHARPI_BENCH_CUDA_TQ=1 to run; skipping."); return; }
        var path = FindModelPath();
        if (path is null) { _out.WriteLine($"{ModelFile} not found; skipping."); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        using var gpu = new VulkanBackend();
        int requestCtx = enableTq
            ? Math.Min(4096, Math.Max(prefillTokens + decodeTokens + 256, 1024))
            : 0;
        using var fwd = new GpuForwardPass(model, gpu, hp, requestCtx, enableTurboQuant: enableTq);

        string tag = enableTq ? "VK/TQ3" : "VK/FP32";
        Log($"[{tag}] context={fwd.MaxSeqLen} prefill={prefillTokens} decode={decodeTokens}");

        var tokens = new int[prefillTokens];
        var rng = new Random(42);
        for (int i = 0; i < prefillTokens; i++) tokens[i] = rng.Next(1000, 50_000);

        ReadOnlySpan<float> warmLogits = default;
        for (int i = 0; i < tokens.Length; i++)
            warmLogits = fwd.Forward(tokens[i], i);
        int warmNext = ArgMax(warmLogits);
        for (int i = 0; i < 8; i++)
        {
            warmLogits = fwd.Forward(warmNext, tokens.Length + i);
            warmNext = ArgMax(warmLogits);
        }
        fwd.ResetCache();

        var sw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Length; i++)
            logits = fwd.Forward(tokens[i], i);
        double prefillMs = sw.Elapsed.TotalMilliseconds;

        int next = ArgMax(logits);
        sw.Restart();
        for (int i = 0; i < decodeTokens; i++)
        {
            logits = fwd.Forward(next, tokens.Length + i);
            next = ArgMax(logits);
        }
        double decodeMs = sw.Elapsed.TotalMilliseconds;

        Log(
            $"[{tag}] Prefill {prefillTokens} tok in {prefillMs:F1} ms → {prefillTokens / (prefillMs / 1000.0):F2} t/s | " +
            $"Decode {decodeTokens} tok in {decodeMs:F1} ms → {decodeTokens / (decodeMs / 1000.0):F2} t/s");
    }

    private void Measure(bool enableTq, int prefillTokens, int decodeTokens)
    {
        if (!BenchEnabled) { _out.WriteLine("Set SHARPI_BENCH_CUDA_TQ=1 to run; skipping."); return; }
        if (!CudaBackend.IsAvailable()) { _out.WriteLine("CUDA unavailable; skipping."); return; }
        var path = FindModelPath();
        if (path is null) { _out.WriteLine($"{ModelFile} not found; skipping."); return; }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        // For TQ, request enough context to cover prefill + decode plus the FP32 window;
        // for the FP32 baseline, let the estimator pick the largest context the card supports.
        int requestCtx = enableTq
            ? Math.Max(prefillTokens + decodeTokens + 256, 1024)
            : 0;

        using var gpu = CudaBackend.Create();
        using var fwd = new CudaForwardPass(model, gpu, hp,
            maxContextLength: requestCtx,
            enableTurboQuant: enableTq);

        Log($"[{(enableTq ? "TQ3" : "FP32")}] context={fwd.MaxSeqLen} prefill={prefillTokens} decode={decodeTokens}");

        var tokens = new int[prefillTokens];
        var rng = new Random(42);
        for (int i = 0; i < prefillTokens; i++) tokens[i] = rng.Next(1000, 50_000);

        // Warm-up: prime kernel JIT and HBM caches by running one full decode round.
        // Without this the first measurement carries cudaMalloc + first-launch JIT cost.
        WarmUp(fwd, tokens, warmDecode: 8);

        fwd.ResetCache();
        var sw = Stopwatch.StartNew();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Length; i++)
            logits = fwd.Forward(tokens[i], i);
        double prefillMs = sw.Elapsed.TotalMilliseconds;

        int next = ArgMax(logits);

        sw.Restart();
        for (int i = 0; i < decodeTokens; i++)
        {
            logits = fwd.Forward(next, tokens.Length + i);
            next = ArgMax(logits);
        }
        double decodeMs = sw.Elapsed.TotalMilliseconds;

        double prefillTps = prefillTokens / (prefillMs / 1000.0);
        double decodeTps = decodeTokens / (decodeMs / 1000.0);
        Log(
            $"[{(enableTq ? "TQ3" : "FP32")}] Prefill {prefillTokens} tok in {prefillMs:F1} ms → {prefillTps:F2} t/s | " +
            $"Decode {decodeTokens} tok in {decodeMs:F1} ms → {decodeTps:F2} t/s");
    }

    private static void WarmUp(CudaForwardPass fwd, int[] tokens, int warmDecode)
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Length; i++)
            logits = fwd.Forward(tokens[i], i);
        int next = ArgMax(logits);
        for (int i = 0; i < warmDecode; i++)
        {
            logits = fwd.Forward(next, tokens.Length + i);
            next = ArgMax(logits);
        }
        fwd.ResetCache();
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0;
        float bv = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var c = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(c)) return c;
            var p = Directory.GetParent(dir);
            if (p is null) break;
            dir = p.FullName;
        }
        return null;
    }
}
