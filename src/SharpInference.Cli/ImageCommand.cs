using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Diffusion;
using SharpInference.Vulkan;

namespace SharpInference.Cli;

/// <summary>
/// Image generation command. Supports two native pipelines:
///
///   Z-Image-Turbo (auto-detected when model path contains "z_image" or "zimage"):
///     sharpi image -m models/z_image_turbo-Q5_K_M.gguf --vae vae/
///                  --qwen-encoder Z-Image-AbliteratedV1.Q5_K_M.gguf
///                  --qwen-tokenizer tokenizer.json -p "a cat" -W 1024 -H 1024 -o out.png
///
///   Recommended models (jayn7/Z-Image-Turbo-GGUF  +  BennyDaBall abliterated encoder):
///     DiT Q5_K_M: z_image_turbo-Q5_K_M.gguf  (5.52 GB, highest quality)
///     DiT Q4_K_M: z_image_turbo-Q4_K_M.gguf  (4.50 GB, good quality, faster)
///     Encoder:    Z-Image-AbliteratedV1.Q5_K_M.gguf  (2.89 GB, uncensored)
///
///   FLUX.1-schnell / FLUX.1-dev:
///     sharpi image -m flux1-schnell-q4_k.gguf --vae ae.safetensors
///                  --clip-l clip_l.safetensors --clip-tokenizer tokenizer_clip.json
///                  --t5xxl t5xxl_fp16.safetensors --t5-tokenizer tokenizer_t5.json
///                  -p "a cinematic photograph of a cat" -W 512 -H 512 -o out.png
///
/// Model downloads:
///   Z-Image-Turbo GGUF (Q5_K_M + Q4_K_M): https://huggingface.co/jayn7/Z-Image-Turbo-GGUF
///   Z-Image-Turbo full:                    https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
///   Text encoder (uncensored Q4/Q5):       https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1
///   Qwen3-4B GGUF:                         https://huggingface.co/Qwen/Qwen3-4B-GGUF
///   FLUX.1-schnell GGUF:                   https://huggingface.co/city96/FLUX.1-schnell-gguf
///   VAE + CLIP + T5:                       https://huggingface.co/comfyanonymous/flux_text_encoders
/// </summary>
public sealed class ImageCommand : Command<ImageCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to diffusion model GGUF or safetensors directory (FLUX.1, Z-Image-Turbo, …)")]
        public string? ModelPath { get; init; }

        [CommandOption("-p|--prompt")]
        [Description("Text prompt describing the image to generate")]
        public string? Prompt { get; init; }

        [CommandOption("--negative-prompt")]
        [Description("Negative prompt — what to avoid in the generated image")]
        public string? NegativePrompt { get; init; }

        [CommandOption("--vae")]
        [Description("Path to VAE safetensors file or directory (ae.safetensors or vae/ dir)")]
        public string? VaePath { get; init; }

        // ── Z-Image options ───────────────────────────────────────────────

        [CommandOption("--qwen-encoder")]
        [Description("(Z-Image) Path to Qwen3-4B GGUF text encoder (from Qwen/Qwen3-4B-GGUF)")]
        public string? QwenEncoderPath { get; init; }

        [CommandOption("--qwen-tokenizer")]
        [Description("(Z-Image) Path to Qwen3 tokenizer.json")]
        public string? QwenTokenizerPath { get; init; }

        [CommandOption("-g|--n-gpu-layers")]
        [Description("(Z-Image) GPU acceleration: -1 = auto (CUDA→Vulkan→CPU, default), 0 = CPU only")]
        [DefaultValue(-1)]
        public int NGpuLayers { get; init; }

        [CommandOption("--backend")]
        [Description("(Z-Image) Force compute backend: auto (default), cuda, vulkan, cpu")]
        public string? Backend { get; init; }

        // ── FLUX options ──────────────────────────────────────────────────

        [CommandOption("--clip-l")]
        [Description("(FLUX) Path to CLIP-L encoder safetensors")]
        public string? ClipLPath { get; init; }

        [CommandOption("--clip-tokenizer")]
        [Description("(FLUX) Path to CLIP tokenizer.json")]
        public string? ClipTokenizerPath { get; init; }

        [CommandOption("--t5xxl")]
        [Description("(FLUX) Path to T5-XXL encoder safetensors")]
        public string? T5XXLPath { get; init; }

        [CommandOption("--t5-tokenizer")]
        [Description("(FLUX) Path to T5 tokenizer.json")]
        public string? T5TokenizerPath { get; init; }

        // ── Common options ────────────────────────────────────────────────

        [CommandOption("-W|--width")]
        [Description("Output image width in pixels — must be divisible by 16 (default: 512)")]
        [DefaultValue(512)]
        public int Width { get; init; }

        [CommandOption("-H|--height")]
        [Description("Output image height in pixels — must be divisible by 16 (default: 512)")]
        [DefaultValue(512)]
        public int Height { get; init; }

        [CommandOption("--steps")]
        [Description("Denoising steps (default: 4 for Z-Image-Turbo, 4 for FLUX schnell, 20 for dev)")]
        [DefaultValue(0)]
        public int Steps { get; init; }

        [CommandOption("--cfg-scale")]
        [Description("Guidance scale — not used for Z-Image (distilled), 1.0 for FLUX schnell (default: auto)")]
        [DefaultValue(0f)]
        public float CfgScale { get; init; }

        [CommandOption("-s|--seed")]
        [Description("RNG seed (-1 = random, default: -1)")]
        [DefaultValue(-1)]
        public int Seed { get; init; }

        [CommandOption("-o|--output")]
        [Description("Output PNG file path (default: output.png)")]
        public string? OutputPath { get; init; }

        [CommandOption("-v|--verbose")]
        [Description("Show per-step timing and progress")]
        [DefaultValue(false)]
        public bool Verbose { get; init; }

        // ── sd-cli fallback ───────────────────────────────────────────────

        [CommandOption("--use-sdcpp")]
        [Description("Delegate to stable-diffusion.cpp sd-cli instead of native pipeline (for comparison)")]
        [DefaultValue(false)]
        public bool UseSdCpp { get; init; }

        [CommandOption("--sd-cli")]
        [Description("Path to sd-cli executable used when --use-sdcpp is set (overrides SHARPI_SDCPP env var)")]
        public string? SdCliPath { get; init; }

        [CommandOption("--text-encoder")]
        [Description("(sd-cli mode only) Path to LLM-style text encoder GGUF")]
        public string? TextEncoderPath { get; init; }
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        if (settings.UseSdCpp)
            return RunSdCpp(settings);

        return RunNative(settings);
    }

    // ── Native pipeline ───────────────────────────────────────────────────

    private static int RunNative(Settings s)
    {
        var modelPath = ResolveModelPath(s.ModelPath);
        if (modelPath is null)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No diffusion model found. Use [yellow]-m <path>[/]");
            PrintModelDownloadHint();
            return 1;
        }

        if (string.IsNullOrWhiteSpace(s.Prompt))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Prompt required. Use [yellow]-p \"your prompt\"[/]");
            return 1;
        }

        return IsZImage(modelPath) ? RunZImage(s, modelPath) : RunFlux(s, modelPath);
    }

    private static int RunZImage(Settings s, string modelPath)
    {
        // Auto-discover component paths from models/ directory if not explicitly provided
        string? vaePath       = s.VaePath       ?? ResolveZImageVae();
        string? encoderPath   = s.QwenEncoderPath   ?? ResolveZImageEncoder();
        string? tokenizerPath = s.QwenTokenizerPath ?? ResolveZImageTokenizer();

        if (!RequirePathExists(vaePath, "--vae", "models/z-image-turbo/vae/")) return 1;
        if (!RequireFile(encoderPath,   "--qwen-encoder",   "models/Z-Image-AbliteratedV1.Q5_K_M.gguf")) return 1;
        if (!RequireFile(tokenizerPath, "--qwen-tokenizer", "models/z-image-turbo/tokenizer/tokenizer.json")) return 1;

        string output = s.OutputPath ?? "output.png";
        // Pass -1 when no explicit --steps given so ZImagePipeline uses ZImageParams.DefaultSteps (4)
        int steps = s.Steps > 0 ? s.Steps : -1;

        AnsiConsole.MarkupLine("[bold]Z-Image-Turbo[/] (S3-DiT + Qwen3-4B)");
        AnsiConsole.MarkupLine($"[dim]DiT:[/]      {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]VAE:[/]      {Markup.Escape(vaePath!)}");
        AnsiConsole.MarkupLine($"[dim]Encoder:[/]  {Markup.Escape(encoderPath!)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]     {s.Width}×{s.Height}  steps={steps}  seed={s.Seed}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]   {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start("Loading Z-Image models…", ctx =>
                {
                    ctx.Status("Loading DiT + VAE + Qwen3-4B…");

                    IComputeBackend? gpu = null;
                    try
                    {
                        // Resolve which backend to use:
                        //   --backend cuda|vulkan|cpu  → forced
                        //   -g 0                       → CPU only
                        //   default (-1)               → CUDA → Vulkan → CPU fallback
                        string backendChoice = (s.Backend ?? "auto").ToLowerInvariant();
                        bool forceCpu    = s.NGpuLayers == 0 || backendChoice == "cpu";
                        bool forceCuda   = backendChoice == "cuda";
                        bool forceVulkan = backendChoice == "vulkan";

                        if (!forceCpu)
                        {
                            if (forceCuda || (!forceVulkan && CudaBackend.IsAvailable()))
                            {
                                gpu = CudaBackend.Create();
                                AnsiConsole.MarkupLine("[dim]Backend:[/]  GPU (CUDA cuBLAS)");
                            }
                            else
                            {
                                try
                                {
                                    var vulkan = new VulkanBackend();
                                    gpu = vulkan;
                                    vulkan.PrintDeviceInfo();
                                    AnsiConsole.MarkupLine("[dim]Backend:[/]  GPU (Vulkan SGEMM)");
                                }
                                catch
                                {
                                    AnsiConsole.MarkupLine("[dim]Backend:[/]  CPU (no GPU detected)");
                                }
                            }
                        }
                        else
                        {
                            AnsiConsole.MarkupLine("[dim]Backend:[/]  CPU");
                        }

                        var pipeline = ZImagePipeline.Load(
                            modelPath,
                            vaePath!,
                            encoderPath!,
                            tokenizerPath!,
                            gpu);

                        using (pipeline)
                        {
                            var stepSw = Stopwatch.StartNew();
                            pipeline.Generate(
                                s.Prompt!, s.Width, s.Height, steps, s.Seed, output,
                                progress: (step, total) =>
                                {
                                    ctx.Status($"Step {step}/{total} — {stepSw.Elapsed.TotalSeconds:F1}s elapsed…");
                                    stepSw.Restart();
                                },
                                statusCallback: s => ctx.Status(s));
                        }

                        AnsiConsole.MarkupLine($"[green]✓[/] Done in [cyan]{sw.Elapsed.TotalSeconds:F1}s[/]");
                        AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
                    }
                    finally
                    {
                        if (gpu is IDisposable d) d.Dispose();
                    }
                });
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        return 0;
    }

    private static int RunFlux(Settings s, string modelPath)
    {
        if (!RequireFile(s.VaePath,           "--vae",            "ae.safetensors"))        return 1;
        if (!RequireFile(s.ClipLPath,         "--clip-l",         "clip_l.safetensors"))     return 1;
        if (!RequireFile(s.ClipTokenizerPath, "--clip-tokenizer", "tokenizer_clip.json"))    return 1;
        if (!RequireFile(s.T5XXLPath,         "--t5xxl",          "t5xxl_fp16.safetensors")) return 1;
        if (!RequireFile(s.T5TokenizerPath,   "--t5-tokenizer",   "tokenizer_t5.json"))      return 1;

        string output = s.OutputPath ?? "output.png";
        int steps     = s.Steps > 0 ? s.Steps : IsDistilled(modelPath) ? 4 : 20;
        float cfg     = s.CfgScale > 0f ? s.CfgScale : 1.0f;

        AnsiConsole.MarkupLine("[bold]FLUX.1[/] (MM-DiT + CLIP-L + T5-XXL)");
        AnsiConsole.MarkupLine($"[dim]DiT:[/]     {Markup.Escape(modelPath)}");
        AnsiConsole.MarkupLine($"[dim]VAE:[/]     {Markup.Escape(s.VaePath!)}");
        AnsiConsole.MarkupLine($"[dim]CLIP-L:[/]  {Markup.Escape(s.ClipLPath!)}");
        AnsiConsole.MarkupLine($"[dim]T5-XXL:[/]  {Markup.Escape(s.T5XXLPath!)}");
        AnsiConsole.MarkupLine($"[dim]Size:[/]    {s.Width}×{s.Height}  steps={steps}  cfg={cfg:F1}  seed={s.Seed}");
        AnsiConsole.MarkupLine($"[dim]Output:[/]  {Markup.Escape(output)}");
        AnsiConsole.WriteLine();

        try
        {
            var sw = Stopwatch.StartNew();
            AnsiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .SpinnerStyle(Style.Parse("blue"))
                .Start("Loading FLUX models…", ctx =>
                {
                    ctx.Status("Loading DiT + VAE + CLIP-L + T5-XXL…");
                    var pipeline = ImagePipeline.Load(
                        modelPath,
                        s.VaePath!,
                        s.ClipLPath!,   s.ClipTokenizerPath!,
                        s.T5XXLPath!,   s.T5TokenizerPath!);

                    using (pipeline)
                    {
                        ctx.Status($"Generating {s.Width}×{s.Height} image…");
                        var stepSw = Stopwatch.StartNew();
                        pipeline.Generate(
                            s.Prompt!, s.Width, s.Height, steps, cfg, s.Seed, output,
                            progress: (step, total) =>
                            {
                                ctx.Status($"Step {step}/{total} — {stepSw.Elapsed.TotalSeconds:F1}s elapsed…");
                                stepSw.Restart();
                            });
                    }

                    AnsiConsole.MarkupLine($"[green]✓[/] Done in [cyan]{sw.Elapsed.TotalSeconds:F1}s[/]");
                    AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
                });
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }

        return 0;
    }

    // ── sd-cli fallback (for research comparison) ─────────────────────────

    private static int RunSdCpp(Settings s)
    {
        string? sdCli = FindSdCli(s.SdCliPath);
        if (sdCli is null)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] sd-cli not found. Set [yellow]SHARPI_SDCPP[/] or place binary in [cyan]tools/sd-cli.exe[/].");
            AnsiConsole.MarkupLine("Download: [link]https://github.com/leejet/stable-diffusion.cpp/releases[/]");
            return 1;
        }

        var modelPath = ResolveModelPath(s.ModelPath);
        if (modelPath is null || string.IsNullOrWhiteSpace(s.Prompt))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Model (-m) and prompt (-p) are required.");
            return 1;
        }

        string output = s.OutputPath ?? "output.png";
        int steps     = s.Steps > 0 ? s.Steps : IsDistilled(modelPath) ? 4 : 20;
        float cfg     = s.CfgScale > 0f ? s.CfgScale : IsFlowMatching(modelPath) ? 1.0f : 3.5f;

        var args = BuildSdCppArgs(modelPath, output, steps, cfg, s);

        AnsiConsole.MarkupLine($"[dim](sd-cli mode)[/] {Markup.Escape(sdCli)}");

        var psi = new ProcessStartInfo(sdCli, args) { UseShellExecute = false,
            RedirectStandardOutput = true, RedirectStandardError = true };
        using var proc = Process.Start(psi);
        if (proc is null) { AnsiConsole.MarkupLine("[red]Error:[/] Failed to launch sd-cli."); return 1; }

        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();
        AnsiConsole.Status().Spinner(Spinner.Known.Dots).SpinnerStyle(Style.Parse("blue"))
            .Start($"Generating {s.Width}×{s.Height} image…", _ => proc.WaitForExit());

        if (proc.ExitCode != 0)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] sd-cli exited with code {proc.ExitCode}.");
            return proc.ExitCode;
        }
        AnsiConsole.MarkupLine($"[green]✓[/] Image saved: [cyan]{Markup.Escape(Path.GetFullPath(output))}[/]");
        return 0;
    }

    private static string BuildSdCppArgs(string modelPath, string output, int steps, float cfg, Settings s)
    {
        var parts = new List<string> { "--diffusion-model", Q(modelPath) };
        if (s.VaePath is not null)            { parts.Add("--vae");   parts.Add(Q(s.VaePath)); }
        if (s.T5XXLPath is not null)          { parts.Add("--t5xxl"); parts.Add(Q(s.T5XXLPath)); }
        if (s.ClipLPath is not null)          { parts.Add("--clip_l"); parts.Add(Q(s.ClipLPath)); }
        if (s.QwenEncoderPath is not null)    { parts.Add("--llm");   parts.Add(Q(s.QwenEncoderPath)); }
        if (s.TextEncoderPath is not null)    { parts.Add("--llm");   parts.Add(Q(s.TextEncoderPath)); }
        parts.Add("-p"); parts.Add(Q(s.Prompt!));
        if (!string.IsNullOrWhiteSpace(s.NegativePrompt)) { parts.Add("-n"); parts.Add(Q(s.NegativePrompt)); }
        parts.Add("-W"); parts.Add(s.Width.ToString());
        parts.Add("-H"); parts.Add(s.Height.ToString());
        parts.Add("--steps"); parts.Add(steps.ToString());
        parts.Add("--cfg-scale"); parts.Add(cfg.ToString("F1", CultureInfo.InvariantCulture));
        if (s.Seed != -1) { parts.Add("-s"); parts.Add(s.Seed.ToString()); }
        parts.Add("-o"); parts.Add(Q(output));
        return string.Join(" ", parts);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static bool RequireFile(string? path, string flag, string example)
    {
        if (!string.IsNullOrWhiteSpace(path) && File.Exists(path)) return true;
        AnsiConsole.MarkupLine($"[red]Error:[/] Missing [yellow]{flag} <path>[/] (e.g. [cyan]{example}[/])");
        return false;
    }

    private static bool RequirePathExists(string? path, string flag, string example)
    {
        if (!string.IsNullOrWhiteSpace(path) && (File.Exists(path) || Directory.Exists(path))) return true;
        AnsiConsole.MarkupLine($"[red]Error:[/] Missing [yellow]{flag} <path>[/] (e.g. [cyan]{example}[/])");
        return false;
    }

    private static string? ResolveModelPath(string? given)
    {
        if (given is not null) return (File.Exists(given) || Directory.Exists(given)) ? given : null;
        foreach (var c in new[] {
            "models/z_image_turbo-Q5_K_M.gguf", "models/z_image_turbo-Q4_K_M.gguf", "models/z_image_turbo-Q8_0.gguf",
            "models/flux1-schnell-q4_k.gguf",   "models/flux1-dev-q4_k.gguf" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveZImageVae()
    {
        foreach (var c in new[] {
            "models/z-image-turbo/vae",
            "models/z-image-turbo/vae/diffusion_pytorch_model.safetensors",
            "models/ae.safetensors" })
            if (File.Exists(c) || Directory.Exists(c)) return c;
        return null;
    }

    private static string? ResolveZImageEncoder()
    {
        foreach (var c in new[] {
            "models/Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "models/Z-Image-AbliteratedV1.Q4_K_M.gguf",
            "models/Z-Image-AbliteratedV1.Q8_0.gguf",
            "models/Z-Image-AbliteratedV1.F16.gguf" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static string? ResolveZImageTokenizer()
    {
        foreach (var c in new[] {
            "models/z-image-turbo/tokenizer/tokenizer.json",
            "models/tokenizer.json" })
            if (File.Exists(c)) return c;
        return null;
    }

    private static void PrintModelDownloadHint()
    {
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("Z-Image-Turbo GGUF (Q5_K_M + Q4_K_M):");
        AnsiConsole.MarkupLine("  [link]https://huggingface.co/jayn7/Z-Image-Turbo-GGUF[/]");
        AnsiConsole.MarkupLine("  z_image_turbo-Q5_K_M.gguf  (5.52 GB, best quality)");
        AnsiConsole.MarkupLine("  z_image_turbo-Q4_K_M.gguf  (4.50 GB, good quality, faster dequant)");
        AnsiConsole.MarkupLine("Text encoder (uncensored):");
        AnsiConsole.MarkupLine("  [link]https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1[/]");
        AnsiConsole.MarkupLine("  Z-Image-AbliteratedV1.Q5_K_M.gguf or Q4_K_M.gguf");
        AnsiConsole.MarkupLine("FLUX.1-schnell GGUF: [link]https://huggingface.co/city96/FLUX.1-schnell-gguf[/]");
    }

    private static bool IsZImage(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("z_image") || n.Contains("zimage") || n.Contains("z-image")) return true;
        // Also check if it's a directory named "transformer" with z-image sibling vae
        if (Directory.Exists(path))
        {
            string parent = Path.GetDirectoryName(Path.GetFullPath(path)) ?? "";
            string lp     = parent.ToLowerInvariant();
            return lp.Contains("z_image") || lp.Contains("zimage") || lp.Contains("z-image");
        }
        return false;
    }

    private static bool IsDistilled(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("schnell") || n.Contains("turbo") || n.Contains("lcm");
    }

    private static bool IsFlowMatching(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        return n.Contains("flux") || n.Contains("z_image") || n.Contains("z-image");
    }

    private static string? FindSdCli(string? explicit_)
    {
        if (explicit_ is not null) return File.Exists(explicit_) ? explicit_ : null;
        var env = Environment.GetEnvironmentVariable("SHARPI_SDCPP");
        if (env is not null && File.Exists(env)) return env;
        string[] names = OperatingSystem.IsWindows() ? ["sd-cli.exe", "sd.exe"] : ["sd-cli", "sd"];
        foreach (var b in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
            foreach (var sub in new[] { "tools", "tools/sd", "." })
                foreach (var n in names) { var p = Path.Combine(b, sub, n); if (File.Exists(p)) return p; }
        var pathVar = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in pathVar.Split(Path.PathSeparator))
            foreach (var n in names) { var p = Path.Combine(dir, n); if (File.Exists(p)) return p; }
        return null;
    }

    private static string Q(string s) =>
        s.Contains(' ') || s.Contains('"') ? $"\"{s.Replace("\"", "\\\"")}\"" : s;
}
