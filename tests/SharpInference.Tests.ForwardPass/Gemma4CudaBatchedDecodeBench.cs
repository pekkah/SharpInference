using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #275/#283: aggregate decode-throughput measurement for the Gemma 4 batched decode. #195
/// shipped Gemma 4 continuous batching with every trunk matmul on the cuBLAS GEMM
/// (<c>GpuMatMulBatched</c>); #275 routes the trunk matmuls (Q/K/V/O, gate/up/down, lm-head) through
/// <see cref="CudaForwardPass.BatchForwardMulti"/>'s decode router (<c>BatchDecodeMatMul</c> — the
/// #194 weight-stationary matvec / #201/#206 decode-MMQ), like the dense path, while the PLE matmuls
/// stay on GEMM. cuBLAS GEMM is compute-bound and known to lose to WS/GEMM-N for small-N decode
/// (#190), so this is the A/B that proves the win.
///
/// <para>Default model is <b>Gemma 4 E4B Q8_0</b> (per-layer head_dim / SWA / shared-KV / PLE). On
/// Q8_0 the #201/#206 int8 decode-MMQ tile (Q4_K/Q6_K-only) always falls back to the WS matvec, so
/// E4B Q8_0 only measures the WS path. <b>Issue #283</b>: point this at a GEMM-N-batchable
/// <b>Q4_K 12B</b> (<c>SHARPI_BENCH_MODEL=gemma-4-12b-it-Q4_K_M.gguf</c>) so the decode-MMQ tile
/// (N≥5, rows≥2048, cols%256) actually engages for the big Q4_K trunk shapes (q/o-proj, gate/up,
/// down, lm-head), and confirm it beats the WS matvec at higher N. The 12B is the realistic
/// k_eq_v-on-global + real-V-on-SWA model.</para>
///
/// The routing is selected at construction by the ambient env, so A/B by running twice:
///   # default routing (WS for E4B Q8_0; WS for N&lt;5 + decode-MMQ for N≥5 big Q4_K shapes on the 12B):
///   $env:SHARPI_BENCH_BATCH=1; dotnet test ... --filter "FullyQualifiedName~Gemma4CudaBatchedDecodeBench"
///   # decode-MMQ vs WS A/B on the Q4_K 12B (#283): force WS everywhere with SHARPI_BATCH_DECODE_MMQ=0
///   $env:SHARPI_BENCH_BATCH=1; $env:SHARPI_BENCH_MODEL="gemma-4-12b-it-Q4_K_M.gguf";
///   $env:SHARPI_BENCH_BATCH_N="2,4,5,6,8"; dotnet test ... (default = MMQ on, then re-run with
///   $env:SHARPI_BATCH_DECODE_MMQ="0" = WS everywhere)
///   # old all-GEMM routing (#195 baseline):
///   $env:SHARPI_BENCH_BATCH=1; $env:SHARPI_BATCH_DECODE_GEMM=1; dotnet test ... (same filter)
/// Surfaces t/s, asserts no threshold (mirrors <see cref="CudaBatchedDecodeBench"/>). Silent-skips
/// without CUDA / the GGUF.
/// </summary>
public sealed class Gemma4CudaBatchedDecodeBench
{
    // Default E4B Q8_0; override with SHARPI_BENCH_MODEL (filename resolved against the model dirs,
    // or an absolute path) to point at the Q4_K 12B for the #283 decode-MMQ validation.
    private static string ModelFile =>
        Environment.GetEnvironmentVariable("SHARPI_BENCH_MODEL") is { Length: > 0 } m
            ? m
            : "gemma-4-E4B-it-Q8_0.gguf";
    private readonly ITestOutputHelper _out;
    public Gemma4CudaBatchedDecodeBench(ITestOutputHelper outHelper) { _out = outHelper; }

    private void Log(string line) { _out.WriteLine(line); Console.Error.WriteLine(line); }

    private static bool BenchEnabled => Environment.GetEnvironmentVariable("SHARPI_BENCH_BATCH") == "1";

    private static int[] BatchSizesToRun =>
        (Environment.GetEnvironmentVariable("SHARPI_BENCH_BATCH_N") ?? "1,2,4,8")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(int.Parse).ToArray();
    private static int DecodeSteps =>
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_BENCH_STEPS"), out int s) && s > 0 ? s : 128;
    // The decode loop only touches ~prompt+steps positions, so a smaller ctx is fine and lets the
    // bigger-KV Q4_K 12B fit N=8 caches in 12 GB (SHARPI_BENCH_CTX, default 2048 — unchanged for E4B).
    private static int MaxCtx =>
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_BENCH_CTX"), out int c) && c > 0 ? c : 2048;

    private static readonly int[] Prompt =
        { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444, 555, 666, 777, 888 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        // SHARPI_BENCH_MODEL may be an absolute path; honor it directly.
        if (Path.IsPathRooted(ModelFile)) return File.Exists(ModelFile) ? ModelFile : null;

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
    public void Gemma4_E4B_Decode_Throughput_Batched()
    {
        if (!BenchEnabled) return;
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Pin SnapKV off (the constructor already forces it off for Gemma 4, but be explicit).
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        CudaForwardPass fwdTmp;
        try { fwdTmp = new CudaForwardPass(model, gpu, hp, maxContextLength: MaxCtx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap); }
        using var fwd = fwdTmp;

        bool gemm = Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_GEMM") == "1";
        bool mmqKill = Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_MMQ") == "0";
        string routing = gemm
            ? "all-GEMM (#195 baseline)"
            : mmqKill ? "WS only (decode-MMQ forced off)" : "WS + decode-MMQ@N≥5 (#275 default)";
        Log($"[bench-275] model = {Path.GetFileName(path)} · routing = {routing}");

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
                    toks[s] = Argmax(fwd.PrefillWithCache(Prompt, caches[s]));
                    poss[s] = Prompt.Length;
                }
                for (int i = 0; i < warmup; i++)
                {
                    var lg = fwd.BatchForwardMulti(toks, poss, caches);
                    for (int s = 0; s < n; s++) { toks[s] = Argmax(lg[s]); poss[s]++; }
                }
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < decodeSteps; i++)
                {
                    var lg = fwd.BatchForwardMulti(toks, poss, caches);
                    for (int s = 0; s < n; s++) { toks[s] = Argmax(lg[s]); poss[s]++; }
                }
                sw.Stop();
                double aggTps = (double)n * decodeSteps / sw.Elapsed.TotalSeconds;
                double perSeq = (double)decodeSteps / sw.Elapsed.TotalSeconds;
                Log($"[bench-275] N={n}: aggregate {aggTps:F1} t/s ({perSeq:F1} t/s/seq)");
            }
            finally
            {
                foreach (var c in caches) c?.Dispose();
            }
        }
    }
}
