using System.ComponentModel;
using System.Diagnostics;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Vulkan;

namespace SharpInference.Cli;

/// <summary>
/// Main inference command. Parameter names match llama-cli where applicable.
/// Usage: sharpi-cli -m model.gguf -p "Hello" -n 128 --temp 0.7
/// </summary>
public sealed class RunCommand : Command<RunCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("-p|--prompt")]
        [Description("Input prompt (default: interactive chat)")]
        public string? Prompt { get; init; }

        [CommandOption("-n|--n-predict")]
        [Description("Number of tokens to predict (default: 512)")]
        [DefaultValue(512)]
        public int NPredict { get; init; }

        [CommandOption("--temp")]
        [Description("Temperature (0 = greedy, default: 0.7)")]
        [DefaultValue(0.7f)]
        public float Temperature { get; init; }

        [CommandOption("--top-k")]
        [Description("Top-k sampling (0 = disabled, default: 40)")]
        [DefaultValue(40)]
        public int TopK { get; init; }

        [CommandOption("--top-p")]
        [Description("Top-p nucleus sampling (default: 0.95)")]
        [DefaultValue(0.95f)]
        public float TopP { get; init; }

        [CommandOption("--min-p")]
        [Description("Min-p sampling (default: 0.05)")]
        [DefaultValue(0.05f)]
        public float MinP { get; init; }

        [CommandOption("-s|--seed")]
        [Description("RNG seed (-1 = random, default: -1)")]
        [DefaultValue(-1)]
        public int Seed { get; init; }

        [CommandOption("--single-turn")]
        [Description("Generate one response and exit")]
        [DefaultValue(false)]
        public bool SingleTurn { get; init; }

        [CommandOption("--system-prompt")]
        [Description("System prompt")]
        public string? SystemPrompt { get; init; }

        [CommandOption("--no-display-prompt")]
        [Description("Don't echo the prompt")]
        [DefaultValue(false)]
        public bool NoDisplayPrompt { get; init; }

        [CommandOption("--verbose-prompt")]
        [Description("Print token IDs before generating")]
        [DefaultValue(false)]
        public bool VerbosePrompt { get; init; }

        [CommandOption("--n-gpu-layers|-g")]
        [Description("Layers on GPU (0=CPU only, -1=all, default: 0)")]
        [DefaultValue(0)]
        public int NGpuLayers { get; init; }

        [CommandOption("-c|--ctx-size")]
        [Description("Context size / max sequence length (0 = model default)")]
        [DefaultValue(0)]
        public int CtxSize { get; init; }

        [CommandOption("--tq")]
        [Description("Enable TurboQuant KV cache compression (3-bit, reduces VRAM ~5x)")]
        [DefaultValue(false)]
        public bool TurboQuant { get; init; }
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        var modelPath = settings.ModelPath;
        if (modelPath is null)
        {
            foreach (var candidate in new[] { "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "model.gguf" })
                if (File.Exists(candidate)) { modelPath = candidate; break; }
        }
        if (modelPath is null || !File.Exists(modelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No model file found. Use [yellow]-m <path>[/]");
            return 1;
        }

        AnsiConsole.MarkupLine($"[dim]Loading model:[/] {modelPath}");
        var sw = Stopwatch.StartNew();
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        int ctxSize = settings.CtxSize; // 0 = auto (GPU will estimate from VRAM, CPU uses model default)
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var cpuBackend = new CpuBackend();
        using var fwd = new ForwardPass(model, cpuBackend, hp);

        // Create backend-specific forward pass
        IDisposable? gpuBackend = null;
        IDisposable? gpuFwd = null;

        Func<int, int, ReadOnlySpan<float>> forward;
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill;
        Action resetCache;

        bool useGpu = settings.NGpuLayers != 0;
        if (useGpu)
        {
            var gpu = new VulkanBackend();
            gpuBackend = gpu;
            gpu.PrintDeviceInfo();

            var gfwd = new GpuForwardPass(model, gpu, hp, ctxSize,
                enableTurboQuant: settings.TurboQuant);
            if (settings.TurboQuant)
                AnsiConsole.MarkupLine($"[dim]TurboQuant: [green]enabled[/] (3-bit, context: {gfwd.MaxSeqLen})[/]");
            gpuFwd = gfwd;
            forward = gfwd.Forward;
            prefill = tokens => { ReadOnlySpan<float> l = default; for (int i = 0; i < tokens.Count; i++) l = gfwd.Forward(tokens[i], i); return l; };
            resetCache = gfwd.ResetCache;
            AnsiConsole.MarkupLine($"[dim]Backend: [green]GPU[/] ({gpu.Name})[/]");
        }
        else
        {
            if (settings.TurboQuant)
            {
                fwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
            }
            forward = fwd.Forward;
            prefill = fwd.Prefill;
            resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
            AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/][/]");
        }

        AnsiConsole.MarkupLine($"[dim]Model loaded in {sw.Elapsed.TotalSeconds:F1}s — " +
            $"{hp.NumLayers}L, {hp.EmbeddingDim}d, {hp.VocabSize} vocab, ctx={hp.ContextLength}[/]");

        var sp = new SamplingParams
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            MaxNewTokens = settings.NPredict,
            StopTokenIds = [tokenizer.EosTokenId],
        };
        var rng = settings.Seed >= 0 ? new Random(settings.Seed) : new Random();

        try
        {
            if (settings.Prompt is not null)
                return RunSinglePrompt(settings, forward, prefill, tokenizer, sp, rng);
            return RunInteractive(settings, forward, prefill, resetCache, tokenizer, sp, rng);
        }
        finally
        {
            gpuFwd?.Dispose();
            gpuBackend?.Dispose();
        }
    }

    private static int RunSinglePrompt(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt);
        var tokens = tok.Encode(prompt);

        if (s.VerbosePrompt)
            AnsiConsole.MarkupLine($"[dim]Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}[/]");

        var sw = Stopwatch.StartNew();
        var logits = prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        sw.Restart();
        int generated = 0;
        for (int i = 0; i < sp.MaxNewTokens; i++)
        {
            int next = sp.Temperature <= 0 ? Sampler.Greedy(logits) : Sampler.Sample(logits, sp, rng);
            if (sp.StopTokenIds.Contains(next)) break;
            Console.Write(tok.Decode([next]));
            generated++;
            logits = forward(next, tokens.Count + i);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {generated} tokens, {generated / (decodeMs / 1000):F1} t/s[/]");
        return 0;
    }

    private static int RunInteractive(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        Action resetCache,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        AnsiConsole.MarkupLine("[green]Interactive chat.[/] Type a message, or [yellow]/exit[/] to quit.\n");

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt);
            var tokens = tok.Encode(prompt);

            resetCache();
            var sw = Stopwatch.StartNew();
            var logits = prefill(tokens);

            sw.Restart();
            int generated = 0;
            for (int i = 0; i < sp.MaxNewTokens; i++)
            {
                int next = sp.Temperature <= 0 ? Sampler.Greedy(logits) : Sampler.Sample(logits, sp, rng);
                if (sp.StopTokenIds.Contains(next)) break;
                Console.Write(tok.Decode([next]));
                generated++;
                logits = forward(next, tokens.Count + i);
            }
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{generated} tokens, {generated / (decodeMs / 1000):F1} t/s[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static string FormatPrompt(string userMessage, string? systemPrompt)
    {
        var sb = new System.Text.StringBuilder();
        if (systemPrompt is not null)
            sb.Append($"<|im_start|>system\n{systemPrompt}<|im_end|>\n");
        sb.Append($"<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n");
        return sb.ToString();
    }
}
