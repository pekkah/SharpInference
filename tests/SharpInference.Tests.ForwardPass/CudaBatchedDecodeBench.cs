using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #190: aggregate decode-throughput measurement for CUDA continuous batching. Compares
/// the single-user per-token <see cref="CudaForwardPass.Forward"/> loop against batched
/// <see cref="CudaForwardPass.BatchForwardMulti"/> at several batch sizes on Qwen3-8B Q4_K_M —
/// the whole point of the feature is that batched decode amortizes weight reads across N
/// sequences, so aggregate t/s should climb well above the single-user baseline (~75 t/s on a
/// 4070 Ti). Surfaces numbers, asserts no threshold (mirrors <see cref="CudaTurboQuantBench"/>).
///
/// Opt-in (so it never runs in the normal suite) and silent-skip without CUDA/model:
///   $env:SHARPI_BENCH_BATCH=1; dotnet test tests/SharpInference.Tests.ForwardPass -c Release `
///     --filter "FullyQualifiedName~CudaBatchedDecodeBench" --logger "console;verbosity=normal"
/// </summary>
public sealed class CudaBatchedDecodeBench
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";
    private readonly ITestOutputHelper _out;
    public CudaBatchedDecodeBench(ITestOutputHelper outHelper) { _out = outHelper; }

    private void Log(string line) { _out.WriteLine(line); Console.Error.WriteLine(line); }

    private static bool BenchEnabled => Environment.GetEnvironmentVariable("SHARPI_BENCH_BATCH") == "1";

    // Profiler ergonomics (#197 follow-up): SHARPI_BENCH_BATCH_N="8" runs only those batch
    // sizes (comma list; default 1,2,4,8), SHARPI_BENCH_SINGLE=0 skips the single-user
    // baseline, and SHARPI_BENCH_STEPS shrinks the timed loop (ncu's launch interception
    // makes the full 128-step loop crawl), so a capture contains exactly one configuration.
    private static int[] BatchSizesToRun =>
        (Environment.GetEnvironmentVariable("SHARPI_BENCH_BATCH_N") ?? "1,2,4,8")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(int.Parse).ToArray();
    private static bool SingleUserEnabled =>
        Environment.GetEnvironmentVariable("SHARPI_BENCH_SINGLE") != "0";
    private static int DecodeSteps =>
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_BENCH_STEPS"), out int s) && s > 0 ? s : 128;

    private static readonly int[] Prompt =
        { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13, 220, 17, 18, 19, 20, 21,
          22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
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
        int best = 0; float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    [Fact]
    public void Decode_Throughput_SingleUser_vs_Batched()
    {
        if (!BenchEnabled) return;
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Pin SnapKV off: at this context VRAM auto-SnapKV would otherwise engage and (correctly)
        // disable batching — but the point of this bench is to measure the batched decode path.
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        CudaForwardPass fwdTmp;
        try { fwdTmp = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap); }
        using var fwd = fwdTmp;

        int decodeSteps = DecodeSteps;
        const int warmup = 8;

        // ── Single-user baseline (per-token Forward, CUDA graphs on by default) ──
        double singleTps = double.NaN;
        if (SingleUserEnabled)
        {
            fwd.ResetCache();
            int tok = Argmax(fwd.Prefill(Prompt));
            int pos = Prompt.Length;
            for (int i = 0; i < warmup; i++) { tok = Argmax(fwd.Forward(tok, pos)); pos++; }
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < decodeSteps; i++) { tok = Argmax(fwd.Forward(tok, pos)); pos++; }
            sw.Stop();
            singleTps = decodeSteps / sw.Elapsed.TotalSeconds;
            Log($"[bench-190] single-user Forward: {singleTps:F1} t/s ({decodeSteps} steps in {sw.Elapsed.TotalMilliseconds:F0} ms)");
        }

        // ── Batched BatchForwardMulti at N ∈ {1,2,4,8} (CreateCache disables graphs) ──
        foreach (int n in BatchSizesToRun)
        {
            var caches = new CudaSequenceKvCache[n];
            try
            {
                var toks = new int[n];
                var poss = new int[n];
                for (int s = 0; s < n; s++)
                {
                    caches[s] = fwd.CreateCache();
                    toks[s] = Argmax(fwd.PrefillWithCache(Prompt, caches[s]));
                    poss[s] = Prompt.Length;
                }

                // #205/#206: SHARPI_BENCH_ARGMAX=1 measures the on-device argmax tail
                // (BatchForwardMultiArgmax — rows*8-byte D2H instead of the full N×vocab download).
                // All sequences here are greedy, so this is the all-greedy upper bound of the win.
                bool useArgmax = Environment.GetEnvironmentVariable("SHARPI_BENCH_ARGMAX") == "1";
                for (int i = 0; i < warmup; i++)
                {
                    if (useArgmax)
                    {
                        var am = fwd.BatchForwardMultiArgmax(toks, poss, caches);
                        for (int s = 0; s < n; s++) { toks[s] = am[s].Token; poss[s]++; }
                    }
                    else
                    {
                        var lg = fwd.BatchForwardMulti(toks, poss, caches);
                        for (int s = 0; s < n; s++) { toks[s] = Argmax(lg[s]); poss[s]++; }
                    }
                }
                var sw2 = Stopwatch.StartNew();
                for (int i = 0; i < decodeSteps; i++)
                {
                    if (useArgmax)
                    {
                        var am = fwd.BatchForwardMultiArgmax(toks, poss, caches);
                        for (int s = 0; s < n; s++) { toks[s] = am[s].Token; poss[s]++; }
                    }
                    else
                    {
                        var lg = fwd.BatchForwardMulti(toks, poss, caches);
                        for (int s = 0; s < n; s++) { toks[s] = Argmax(lg[s]); poss[s]++; }
                    }
                }
                sw2.Stop();
                double aggTps = (double)n * decodeSteps / sw2.Elapsed.TotalSeconds;
                double perSeq = (double)decodeSteps / sw2.Elapsed.TotalSeconds;
                Log($"[bench-190] batched N={n}{(useArgmax ? " [argmax]" : "")}: aggregate {aggTps:F1} t/s ({perSeq:F1} t/s/seq), " +
                    $"{aggTps / singleTps:F2}× single-user");
            }
            finally
            {
                foreach (var c in caches) c?.Dispose();
            }
        }
    }

    /// <summary>
    /// Issue #277: aggregate decode throughput for a SnapKV-EVICTED batch on the ragged fast path
    /// vs the #190 per-sequence loop it used to be forced onto. Before #277 any evicted cache in the
    /// batch routed the whole batch through O(N) per-op launches; #277 keeps it on the O(1) ragged
    /// kernels (physical-slot threaded). The per-sequence loop is reachable here via the
    /// <see cref="CudaForwardPass.BatchDecodeRaggedForTest"/> seam, so this A/Bs both on one warm
    /// instance. Long prompts force eviction (budget 512). Surfaces t/s, asserts no threshold.
    ///
    /// Opt-in: $env:SHARPI_BENCH_BATCH=1; dotnet test ... --filter
    ///   "FullyQualifiedName~Decode_Throughput_SnapKvEvicted_Ragged_vs_PerSeq"
    /// </summary>
    [Fact]
    public void Decode_Throughput_SnapKvEvicted_Ragged_vs_PerSeq()
    {
        if (!BenchEnabled) return;
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Long prompt that exceeds the SnapKV budget so every per-sequence cache evicts.
        const string seed = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow. ";
        var sb = new System.Text.StringBuilder();
        while (tokenizer.Encode(sb.ToString()).Count < 1200) sb.Append(seed);
        int[] prompt = tokenizer.Encode(sb.ToString()).ToArray();

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        var prevWindow = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_WINDOW");
        var prevSlots = Environment.GetEnvironmentVariable("SHARPI_PREFIX_SLOTS");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "512");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_WINDOW", "32");
        Environment.SetEnvironmentVariable("SHARPI_PREFIX_SLOTS", null);
        CudaForwardPass fwdTmp;
        try { fwdTmp = new CudaForwardPass(model, gpu, hp, maxContextLength: 2048); }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_WINDOW", prevWindow);
            Environment.SetEnvironmentVariable("SHARPI_PREFIX_SLOTS", prevSlots);
        }
        using var fwd = fwdTmp;

        int decodeSteps = DecodeSteps;
        const int warmup = 8;
        foreach (int n in BatchSizesToRun)
        {
            var caches = new CudaSequenceKvCache[n];
            try
            {
                var toks = new int[n];
                var poss = new int[n];
                for (int s = 0; s < n; s++)
                {
                    caches[s] = fwd.CreateCache();
                    toks[s] = Argmax(fwd.PrefillWithCache(prompt, caches[s]));
                    poss[s] = prompt.Length;
                }
                if (caches[0].EvictedCount == 0)
                    throw new InvalidOperationException("bench expects SnapKV eviction; prompt too short for the budget.");

                double Run(bool ragged)
                {
                    fwd.BatchDecodeRaggedForTest = ragged;
                    var t = (int[])toks.Clone();
                    var p = (int[])poss.Clone();
                    // Idempotent re-decode at fixed positions keeps the timed window pinned to one
                    // cache geometry (no cache growth across steps) — a clean per-op A/B.
                    for (int i = 0; i < warmup; i++) fwd.BatchForwardMulti(t, p, caches);
                    var sw = Stopwatch.StartNew();
                    for (int i = 0; i < decodeSteps; i++) fwd.BatchForwardMulti(t, p, caches);
                    sw.Stop();
                    return (double)n * decodeSteps / sw.Elapsed.TotalSeconds;
                }

                double perSeq = Run(ragged: false);
                double raggedTps = Run(ragged: true);
                Log($"[bench-277] evicted N={n}: ragged {raggedTps:F1} t/s vs per-seq {perSeq:F1} t/s " +
                    $"({raggedTps / perSeq:F2}× ragged, EvictedCount={caches[0].EvictedCount})");
            }
            finally
            {
                fwd.BatchDecodeRaggedForTest = true;
                foreach (var c in caches) c?.Dispose();
            }
        }
    }
}
