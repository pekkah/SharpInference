using System.Diagnostics;
using BenchmarkDotNet.Running;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion;
using SharpInference.ImageBench;

// ── BenchmarkDotNet micro benchmarks ─────────────────────────────────────────
if (args.Length > 0 && args[0] == "--bench")
{
    BenchmarkSwitcher.FromAssembly(typeof(CudaTransferBenchmarks).Assembly)
                     .Run(args[1..]);
    return 0;
}

// ── Locate model files ────────────────────────────────────────────────────────

static string? FindFile(string relative)
{
    // Walk up from the executable looking for the repo root (contains models/)
    var dir = AppContext.BaseDirectory;
    for (int i = 0; i < 8; i++)
    {
        var candidate = Path.Combine(dir, relative);
        if (File.Exists(candidate)) return candidate;
        var parent = Directory.GetParent(dir);
        if (parent is null) break;
        dir = parent.FullName;
    }
    return null;
}

string? ditPath   = FindFile(Path.Combine("models", "z_image_turbo-Q5_K_M.gguf"));
string? qwenPath  = FindFile(Path.Combine("models", "Z-Image-AbliteratedV1.Q5_K_M.gguf"));
string? vaePath   = FindFile(Path.Combine("models", "z-image-turbo", "vae", "diffusion_pytorch_model.safetensors"));
string? tokPath   = FindFile(Path.Combine("models", "z-image-turbo", "tokenizer", "tokenizer.json"));

if (ditPath is null || qwenPath is null || vaePath is null || tokPath is null)
{
    Console.WriteLine("ERROR: Z-Image-Turbo model files not found.");
    Console.WriteLine("  Run:  .\\scripts\\download-model.ps1 -Model z-image-turbo");
    return 1;
}

const string Prompt       = "a serene mountain lake at sunrise, photorealistic";
const int    Width        = 512;
const int    Height       = 512;
const int    Steps        = 9;
const int    Seed         = 42;
const int    WarmupRuns   = 1;
const int    TimedRuns    = 2;

string outDir = Path.Combine(AppContext.BaseDirectory, "bench-output");
Directory.CreateDirectory(outDir);

// ── Backend configurations to benchmark ──────────────────────────────────────

bool cudaAvailable = CudaBackend.IsAvailable();

var configs = new List<(string Label, Func<IComputeBackend> Factory)>();

if (cudaAvailable)
{
    configs.Add(("CUDA fp8 E4M3",  () => CudaBackend.Create(SgemmPrecision.Fp8E4M3)));
    configs.Add(("CUDA bf16",      () => CudaBackend.Create(SgemmPrecision.Bf16)));
    configs.Add(("CUDA fp16",      () => CudaBackend.Create(SgemmPrecision.Fp16)));
    configs.Add(("CUDA fp32",      () => CudaBackend.Create(SgemmPrecision.Fp32)));
}
configs.Add(("CPU (baseline)", () => new CpuBackend()));

// ── Results table ─────────────────────────────────────────────────────────────

var results = new List<(string Label, double AvgMs, double StepsPerSec)>();

Console.WriteLine($"Z-Image-Turbo Image Generation Benchmark");
Console.WriteLine($"  GPU: {(cudaAvailable ? "CUDA available" : "not available — CPU only")}");
Console.WriteLine($"  Resolution: {Width}×{Height}  Steps: {Steps}  Seed: {Seed}");
Console.WriteLine($"  Prompt: \"{Prompt}\"");
Console.WriteLine($"  Warmup: {WarmupRuns} run(s)  Timed: {TimedRuns} run(s) per config");
Console.WriteLine();

foreach (var (label, factory) in configs)
{
    Console.Write($"[{label}] Loading pipeline... ");
    Console.Out.Flush();

    IComputeBackend? backend = null;
    ZImagePipeline? pipeline = null;
    try
    {
        // Create backend once; pass the same instance to the pipeline
        backend  = factory();
        pipeline = ZImagePipeline.Load(ditPath, vaePath, qwenPath, tokPath, backend);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"SKIP (load failed: {ex.Message})");
        backend?.Dispose();
        pipeline?.Dispose();
        continue;
    }

    string? skipReason = null;

    // Warmup — loads weights into GPU caches; first run is always slower
    for (int w = 0; w < WarmupRuns && skipReason is null; w++)
    {
        Console.Write($"warmup {w + 1}/{WarmupRuns}... ");
        Console.Out.Flush();
        string warmupOut = Path.Combine(outDir, $"warmup_{label.Replace(' ', '_').Replace('/', '_')}.png");
        try   { pipeline.Generate(Prompt, Width, Height, Steps, Seed, warmupOut); }
        catch (Exception ex) { skipReason = ex.Message; }
    }

    if (skipReason is not null)
    {
        Console.WriteLine($"SKIP ({skipReason})");
        pipeline.Dispose();
        backend.Dispose();
        continue;
    }

    // Timed runs
    var timings = new List<double>();
    for (int r = 0; r < TimedRuns; r++)
    {
        Console.Write($"run {r + 1}/{TimedRuns}... ");
        Console.Out.Flush();
        string runOut = Path.Combine(outDir, $"result_{label.Replace(' ', '_').Replace('/', '_')}_r{r}.png");

        var sw = Stopwatch.StartNew();
        pipeline.Generate(Prompt, Width, Height, Steps, Seed, runOut);
        sw.Stop();
        timings.Add(sw.Elapsed.TotalMilliseconds);
    }

    pipeline.Dispose();
    backend.Dispose();

    double avgMs       = timings.Average();
    double stepsPerSec = Steps / (avgMs / 1000.0);
    results.Add((label, avgMs, stepsPerSec));

    Console.WriteLine($"avg={avgMs / 1000:F2}s  ({stepsPerSec:F2} steps/s)");
}

// ── Print comparison table ────────────────────────────────────────────────────

Console.WriteLine();
Console.WriteLine("┌─────────────────┬───────────┬─────────────┬──────────┐");
Console.WriteLine("│ Backend         │ Avg time  │  Steps/sec  │ Speedup  │");
Console.WriteLine("├─────────────────┼───────────┼─────────────┼──────────┤");

double? cpuMs = results.FirstOrDefault(r => r.Label.Contains("CPU")).AvgMs;
if (cpuMs == 0) cpuMs = null;  // not measured

foreach (var (label, avgMs, stepsPerSec) in results)
{
    string speedupStr;
    if (label.Contains("CPU") || cpuMs is null)
        speedupStr = label.Contains("CPU") ? "1.00×" : "  n/a ";
    else
        speedupStr = $"{cpuMs.Value / avgMs:F2}×";
    Console.WriteLine($"│ {label,-15} │ {avgMs / 1000,6:F2}s    │ {stepsPerSec,9:F2}     │ {speedupStr,7}  │");
}

Console.WriteLine("└─────────────────┴───────────┴─────────────┴──────────┘");
Console.WriteLine();
Console.WriteLine($"Output images saved to: {outDir}");

return 0;
