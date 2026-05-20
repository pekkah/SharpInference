using System.ComponentModel;
using System.Diagnostics;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
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

        [CommandOption("--draft-model")]
        [Description("Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0)")]
        public string? DraftModelPath { get; init; }

        [CommandOption("--spec-lookahead")]
        [Description("Number of draft tokens per speculative step (default: 4)")]
        [DefaultValue(4)]
        public int SpecLookahead { get; init; }

        [CommandOption("--min-batch-blas")]
        [Description("Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via SHARPI_MIN_BATCH_BLAS env var.")]
        [DefaultValue(0)]
        public int MinBatchBlas { get; init; }

        [CommandOption("--rep-penalty")]
        [Description("Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1)")]
        [DefaultValue(1.1f)]
        public float RepPenalty { get; init; }

        [CommandOption("--backend")]
        [Description("GPU backend: auto, vulkan, cuda. Default: auto (prefers CUDA when -g is set and CUDA is available, otherwise Vulkan).")]
        [DefaultValue("auto")]
        public string Backend { get; init; } = "auto";

        [CommandOption("--no-thinking")]
        [Description("Disable reasoning mode (sets enable_thinking=false in the chat template)")]
        [DefaultValue(false)]
        public bool NoThinking { get; init; }

        [CommandOption("--hide-thinking")]
        [Description("Hide reasoning output (the model still reasons; only the answer is shown)")]
        [DefaultValue(false)]
        public bool HideThinking { get; init; }

        [CommandOption("--max-thinking-tokens")]
        [Description("Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path.")]
        [DefaultValue(0)]
        public int MaxThinkingTokens { get; init; }
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        if (settings.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = settings.MinBatchBlas;

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
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        s_arch = model.Metadata.TryGetValue("general.architecture", out var archVal) ? (string)archVal : "qwen2";
        int ctxSize = settings.CtxSize; // 0 = auto (GPU will estimate from VRAM, CPU uses model default)
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        s_jinja = tokenizer.ChatTemplate;

        // Reasoning models (Qwen3, DeepSeek-R1, SmolLM3, ...) register <think>/</think>
        // as control tokens in their GGUF. The decode loops no-op when these IDs are -1.
        if (tokenizer.SpecialTokens.TryGetValue("<think>", out int thinkId)
            && tokenizer.SpecialTokens.TryGetValue("</think>", out int endThinkId)
            && thinkId > 0 && endThinkId > 0)
        {
            s_thinkTokenId = thinkId;
            s_endThinkTokenId = endThinkId;
        }

        // Greedy on a reasoning model tends to "wait, but actually" itself into infinite
        // loops; --no-thinking sidesteps the issue since the model won't reason at all.
        if (s_thinkTokenId > 0 && settings.Temperature == 0f && !settings.NoThinking)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Greedy decoding (--temp 0) on a reasoning model often produces");
            AnsiConsole.MarkupLine("infinite \"wait, but actually\" loops. Consider [yellow]--temp 0.6 --top-p 0.95 --top-k 20[/].");
        }

        using var cpuBackend = new CpuBackend();

        // Hybrid GDN models (qwen35moe) currently only have a CPU forward pass; reject
        // GPU offload and TurboQuant up front so we don't burn the time to build a
        // ForwardPass that the dispatch won't use.
        if (hp.IsHybridSsm && settings.NGpuLayers != 0)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Hybrid GDN+MoE models (qwen35moe) currently run CPU-only. Use [yellow]-g 0[/] (the default).");
            return 1;
        }
        if (hp.IsHybridSsm && settings.TurboQuant)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers).");
            return 1;
        }
        if (hp.IsHybridSsm && settings.DraftModelPath is not null)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Speculative decoding is not supported for hybrid GDN models (GDN state is destructively updated and cannot be rewound).");
            return 1;
        }

        // Build the dense/MoE CPU forward pass for non-hybrid models. For hybrid GDN
        // models we use HybridGdnForwardPass below; the ForwardPass-typed `fwd` stays
        // null so any code path that depends on it (TurboQuant, speculative decoding)
        // is unreachable for hybrid.
        ForwardPass? fwd = null;
        HybridGdnForwardPass? hybridFwd = null;
        if (hp.IsHybridSsm)
            hybridFwd = new HybridGdnForwardPass(model, cpuBackend, hp);
        else
            fwd = new ForwardPass(model, cpuBackend, hp);

        // Create backend-specific forward pass
        IDisposable? gpuBackend = null;
        IDisposable? gpuFwd = null;

        Func<int, int, ReadOnlySpan<float>> forward;
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill;
        Action resetCache;

        // Validate TurboQuant head-dimension compatibility before any GPU allocation
        if (settings.TurboQuant)
        {
            int headDim = hp.HeadDim;
            if (headDim is not 128 and not 256)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] TurboQuant requires head dimension 128 or 256; this model has head dim {headDim}. Remove [yellow]--tq[/] to run without KV compression.");
                return 1;
            }
        }

        int nGpuLayers = settings.NGpuLayers;

        // Issue #2 (MoE on hybrid GPU+CPU produced NaN/garbled output) was resolved by
        // fixing the descriptor-set reuse hazard in ComputePipeline.RecordWith.
        // The MoE+hybrid path is now exercised by
        // SharpInference.Tests.ForwardPass.VulkanShaderTests.HybridForwardPass_MoE_ProducesFiniteLogits.

        // Resolve which GPU backend to use when -g is non-zero. CUDA is preferred only
        // when the user explicitly opted in (--backend cuda) or auto-detection finds it
        // and Vulkan is not the explicit choice. The CUDA forward pass currently covers
        // dense (non-MoE) models with all layers on GPU; MoE and hybrid -g N stay on the
        // Vulkan path.
        bool wantCuda = false;
        string backendStr = (settings.Backend ?? "auto").Trim().ToLowerInvariant();
        if (nGpuLayers != 0)
        {
            switch (backendStr)
            {
                case "cuda":
                    wantCuda = true;
                    break;
                case "vulkan":
                    wantCuda = false;
                    break;
                case "auto":
                case "":
                    // Auto: pick CUDA when available. CudaForwardPass handles full-offload
                    // (dense + MoE); CudaHybridForwardPass handles partial-offload (dense or
                    // MoE with eager per-layer expert loading). TQ on CUDA requires
                    // head_dim ∈ {128, 256}.
                    bool tqHeadDimOk = hp.HeadDim is 128 or 256;
                    wantCuda = (!settings.TurboQuant || tqHeadDimOk)
                        && CudaBackend.IsAvailable();
                    break;
                default:
                    AnsiConsole.MarkupLine($"[red]Error:[/] Unknown --backend value '{settings.Backend}'. Expected one of: auto, vulkan, cuda.");
                    return 1;
            }
            if (wantCuda && settings.TurboQuant && hp.HeadDim is not 128 and not 256)
            {
                AnsiConsole.MarkupLine($"[yellow]Note:[/] --backend cuda TurboQuant requires head_dim ∈ {{128, 256}} (model head_dim={hp.HeadDim}); falling back to Vulkan.");
                wantCuda = false;
            }
        }

        if (nGpuLayers == 0)
        {
            // CPU only
            if (hybridFwd is not null)
            {
                forward = hybridFwd.Forward;
                prefill = tokens => hybridFwd.Prefill(tokens);
                resetCache = hybridFwd.ResetCache;
                AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/] (hybrid GDN+MoE)[/]");
            }
            else
            {
                if (settings.TurboQuant)
                {
                    fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                    AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                }
                forward = fwd!.Forward;
                prefill = tokens => fwd.Prefill(tokens);
                resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/][/]");
            }
        }
        else if (wantCuda)
        {
            var cuda = CudaBackend.Create();
            gpuBackend = cuda;
            try
            {
                // For -g -1 (auto), run TierPlanner against CUDA's VRAM and use the
                // resulting layer split — same logic as the Vulkan branch. Without this,
                // a model bigger than VRAM (e.g. Qwen3-Coder 30B-A3B in 12 GB) would
                // attempt full-offload via CudaForwardPass and silently OOM.
                int cudaGpuLayers;
                if (nGpuLayers == -1)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize);
                    cudaGpuLayers = placement.GpuLayers;
                }
                else
                {
                    cudaGpuLayers = nGpuLayers;
                }

                bool wantHybrid = cudaGpuLayers > 0 && cudaGpuLayers < hp.NumLayers;
                if (wantHybrid)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize);
                    if (nGpuLayers != -1)
                        placement = placement with { GpuLayers = cudaGpuLayers, CpuLayers = hp.NumLayers - cudaGpuLayers };

                    var chfwd = new CudaHybridForwardPass(model, cuda, hp, placement, settings.TurboQuant);
                    gpuFwd = chfwd;
                    forward = chfwd.Forward;
                    prefill = tokens => chfwd.Prefill(tokens);
                    resetCache = chfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA hybrid[/] ({cuda.Name}, {placement.GpuLayers} GPU + {placement.CpuLayers} CPU layers)[/]");
                }
                else if (cudaGpuLayers == 0)
                {
                    // Model doesn't fit any GPU layer — fall back to CPU forward pass.
                    // (Hybrid GDN models were rejected before reaching here.)
                    cuda.Dispose();
                    gpuBackend = null;
                    if (settings.TurboQuant)
                    {
                        fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                        AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                    }
                    forward = fwd!.Forward;
                    prefill = tokens => fwd.Prefill(tokens);
                    resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                    AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/] (CUDA fallback: no GPU-capable layers)[/]");
                }
                else
                {
                    var cfwd = new CudaForwardPass(model, cuda, hp, ctxSize,
                        enableTurboQuant: settings.TurboQuant);
                    if (settings.TurboQuant)
                        AnsiConsole.MarkupLine($"[dim]TurboQuant: [green]enabled[/] (3-bit, context: {cfwd.MaxSeqLen})[/]");
                    gpuFwd = cfwd;
                    forward = cfwd.Forward;
                    prefill = tokens => cfwd.Prefill(tokens);
                    resetCache = cfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA[/] ({cuda.Name}, all {hp.NumLayers} layers)[/]");
                }
            }
            catch
            {
                gpuFwd?.Dispose();
                gpuBackend?.Dispose();
                gpuFwd = null;
                gpuBackend = null;
                throw;
            }
        }
        else
        {
            var gpu = new VulkanBackend();
            gpuBackend = gpu;
            try
            {
                gpu.PrintDeviceInfo();

                var hwProfile = HardwareProfile.Detect(gpu);
                AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");

                // Auto-detect layer count when -g -1
                if (nGpuLayers == -1)
                {
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant, requestedCtxSize: ctxSize);
                    nGpuLayers = placement.GpuLayers;
                    if (nGpuLayers == 0)
                    {
                        // Hybrid GDN models were rejected before reaching this Vulkan branch.
                        if (settings.TurboQuant)
                        {
                            fwd!.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                            AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                        }

                        forward = fwd!.Forward;
                        prefill = tokens => fwd.Prefill(tokens);
                        resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
                        AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/] (auto fallback: no GPU-capable layers for this model/path)[/]");
                        goto backendConfigured;
                    }
                }

                if (nGpuLayers >= hp.NumLayers)
                {
                    // All layers on GPU
                    var gfwd = new GpuForwardPass(model, gpu, hp, ctxSize,
                        enableTurboQuant: settings.TurboQuant);
                    if (settings.TurboQuant)
                        AnsiConsole.MarkupLine($"[dim]TurboQuant: [green]enabled[/] (3-bit, context: {gfwd.MaxSeqLen})[/]");
                    gpuFwd = gfwd;
                    forward = gfwd.Forward;
                    prefill = tokens => gfwd.Prefill(tokens);
                    resetCache = gfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]GPU[/] ({gpu.Name}, all {hp.NumLayers} layers)[/]");
                }
                else
                {
                    // Hybrid: N layers GPU, rest CPU
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize);
                    // Override with explicit -g N if user specified it
                    if (settings.NGpuLayers > 0)
                        placement = placement with { GpuLayers = nGpuLayers, CpuLayers = hp.NumLayers - nGpuLayers };

                    var hfwd = new HybridForwardPass(model, gpu, hp, placement, settings.TurboQuant);
                    gpuFwd = hfwd;
                    forward = hfwd.Forward;
                    prefill = tokens => hfwd.Prefill(tokens);
                    resetCache = hfwd.ResetCache;
                    AnsiConsole.MarkupLine($"[dim]Backend: [yellow]Hybrid[/] ({gpu.Name}, {placement.GpuLayers} GPU + {placement.CpuLayers} CPU layers)[/]");
                }
            }
            catch
            {
                gpuFwd?.Dispose();
                gpuBackend?.Dispose();
                gpuFwd = null;
                gpuBackend = null;
                throw;
            }
        }

    backendConfigured:
        AnsiConsole.MarkupLine($"[dim]Model loaded in {sw.Elapsed.TotalSeconds:F1}s — " +
            $"{hp.NumLayers}L, {hp.EmbeddingDim}d, headDim={hp.HeadDim}, {hp.VocabSize} vocab, ctx={hp.ContextLength}[/]");

        var sp = new SamplingParams
        {
            Temperature = settings.Temperature,
            TopK = settings.TopK,
            TopP = settings.TopP,
            MinP = settings.MinP,
            MaxNewTokens = settings.NPredict,
            StopTokenIds = [.. BuildStopTokenIds(tokenizer)],
            RepetitionPenalty = settings.RepPenalty,
        };
        var rng = settings.Seed >= 0 ? new Random(settings.Seed) : new Random();

        // Speculative decoding path (requires --draft-model and --temp 0)
        if (settings.DraftModelPath is not null)
        {
            if (settings.Temperature > 0f)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Speculative decoding requires greedy sampling (--temp 0). Falling back to normal generation.");
            }
            else if (nGpuLayers != 0)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Speculative decoding is only supported for CPU (--n-gpu-layers 0). Falling back to normal generation.");
            }
            else if (!File.Exists(settings.DraftModelPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] Draft model not found: {settings.DraftModelPath}");
                return 1;
            }
            else
            {
                try
                {
                    AnsiConsole.MarkupLine($"[dim]Loading draft model:[/] {settings.DraftModelPath}");
                    using var draftModel = GgufModel.Open(settings.DraftModelPath);
                    var draftHp = ModelHyperparams.FromGgufMetadata(draftModel.Metadata, draftModel);
                    using var draftCpuBackend = new CpuBackend();
                    using var draftFwd = new ForwardPass(draftModel, draftCpuBackend, draftHp);
                    AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d | Lookahead k={settings.SpecLookahead}[/]");

                    if (settings.Prompt is not null)
                        return RunSpeculativeSinglePrompt(settings, fwd!, draftFwd, tokenizer, sp);
                    return RunSpeculativeInteractive(settings, fwd!, draftFwd, tokenizer, sp);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex);
                    return 1;
                }
                finally
                {
                    gpuFwd?.Dispose();
                    gpuBackend?.Dispose();
                    fwd?.Dispose();
                    hybridFwd?.Dispose();
                }
            }
        }

        try
        {
            if (settings.Prompt is not null)
                return RunSinglePrompt(settings, forward, prefill, tokenizer, sp, rng);
            return RunInteractive(settings, forward, prefill, resetCache, tokenizer, sp, rng);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
        finally
        {
            gpuFwd?.Dispose();
            gpuBackend?.Dispose();
            fwd?.Dispose();
            hybridFwd?.Dispose();
        }
    }

    private static int RunSpeculativeSinglePrompt(Settings s,
        ForwardPass target, ForwardPass draft,
        GgufTokenizer tok, SamplingParams sp)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s.NoThinking);
        var tokens = tok.Encode(prompt);

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        // Prefill both models with the same prompt
        ReadOnlySpan<float> targetLogits = default;
        ReadOnlySpan<float> draftLogits = default;
        for (int i = 0; i < tokens.Count; i++)
        {
            targetLogits = target.Forward(tokens[i], i);
            draftLogits = draft.Forward(tokens[i], i);
        }
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        var spec = new SpeculativeDecoder(target, draft, s.SpecLookahead);
        spec.Initialize(tokens.Count, targetLogits, draftLogits);

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        spec.Decode(sp.MaxNewTokens, sp.StopTokenIds ?? [], token =>
        {
            if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
        });
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking)) Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            $" | Acceptance rate: {spec.AcceptanceRate:P0}[/]");
        return 0;
    }

    private static int RunSpeculativeInteractive(Settings s,
        ForwardPass target, ForwardPass draft,
        GgufTokenizer tok, SamplingParams sp)
    {
        AnsiConsole.MarkupLine("[green]Interactive chat (speculative decoding).[/] Type a message, or [yellow]/exit[/] to quit.\n");
        var spec = new SpeculativeDecoder(target, draft, s.SpecLookahead);

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s.NoThinking);
            var tokens = tok.Encode(prompt);

            target.Cache.Reset();
            draft.Cache.Reset();

            ReadOnlySpan<float> targetLogits = default;
            ReadOnlySpan<float> draftLogits = default;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < tokens.Count; i++)
            {
                targetLogits = target.Forward(tokens[i], i);
                draftLogits = draft.Forward(tokens[i], i);
            }

            spec.Initialize(tokens.Count, targetLogits, draftLogits);

            sw.Restart();
            int generated = 0;
            int totalDecoded = 0;
            bool inThinking = false;
            var streamDec = new Utf8StreamDecoder();
            bool hideThinking = s.HideThinking;
            spec.Decode(sp.MaxNewTokens, sp.StopTokenIds ?? [], token =>
            {
                if (EmitToken(token, tok, streamDec, ref inThinking, hideThinking)) generated++;
                totalDecoded++;
            });
            var tail = streamDec.Flush();
            if (!(hideThinking && inThinking)) Console.Write(tail);
            if (inThinking) Console.Write("\x1b[0m");
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                $" | Accept: {spec.AcceptanceRate:P0}[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static int RunSinglePrompt(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s.NoThinking);
        var tokens = tok.Encode(prompt);

        if (s.VerbosePrompt)
        {
            var escaped = prompt.Replace("\n", "\\n").Replace("\r", "\\r");
            AnsiConsole.MarkupLine($"[dim]Prompt (escaped): {Markup.Escape(escaped)}[/]");
            AnsiConsole.MarkupLine($"[dim]Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}[/]");
        }

        var sw = Stopwatch.StartNew();
        var logits = prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        sw.Restart();
        var (generated, totalDecoded) = DecodeLoop(forward, logits, tokens.Count, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens);
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            "[/]");
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

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s.NoThinking);
            var tokens = tok.Encode(prompt);

            resetCache();
            var sw = Stopwatch.StartNew();
            var logits = prefill(tokens);

            sw.Restart();
            var (generated, totalDecoded) = DecodeLoop(forward, logits, tokens.Count, tok, sp, rng, hideThinking: s.HideThinking, maxThinkingTokens: s.MaxThinkingTokens);
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
                (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
                "[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static (int generated, int totalDecoded) DecodeLoop(
        Func<int, int, ReadOnlySpan<float>> forward,
        ReadOnlySpan<float> initialLogits,
        int startPos,
        GgufTokenizer tok,
        SamplingParams sp,
        Random rng,
        bool verbosePromptLogging = false,
        bool hideThinking = false,
        int maxThinkingTokens = 0)
    {
        var logits = initialLogits;
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        int thinkingTokenCount = 0;
        var recentTokens = new List<int>(64);
        var streamDec = new Utf8StreamDecoder();
        for (int i = 0; i < sp.MaxNewTokens; i++)
        {
            var spWithHistory = sp.RepetitionPenalty != 1.0f && recentTokens.Count > 0
                ? sp with { PreviousTokens = recentTokens }
                : sp;
            int next;
            if (inThinking && maxThinkingTokens > 0 && thinkingTokenCount >= maxThinkingTokens && s_endThinkTokenId > 0)
            {
                // Force </think> to exit a runaway reasoning block; the close tag still
                // goes through forward() below so the model continues from the post-think state.
                next = s_endThinkTokenId;
            }
            else
            {
                next = sp.Temperature <= 0 ? Sampler.Greedy(logits) : Sampler.Sample(logits, spWithHistory, rng);
            }
            if (verbosePromptLogging)
            {
                var logitsArr = logits.ToArray();
                var top5 = Enumerable.Range(0, logitsArr.Length).OrderByDescending(j => logitsArr[j]).Take(5)
                    .Select(j => $"{j}({logitsArr[j]:F2})");
                Console.Error.WriteLine($"[DBG] tok={i} next={next}('{tok.Decode([next])}') stop={sp.StopTokenIds.Contains(next)} top5:{string.Join(" ", top5)}");
            }
            if (sp.StopTokenIds.Contains(next)) break;
            // Counter resets on each <think> open (in case the model opens multiple blocks)
            // and counts every token emitted while inThinking is true on entry, including
            // the boundary tokens themselves — that keeps the budget predictable: N tokens
            // of reasoning content trip the force-close on iteration N+1.
            if (next == s_thinkTokenId) thinkingTokenCount = 0;
            else if (inThinking) thinkingTokenCount++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking)) generated++;
            totalDecoded++;
            recentTokens.Add(next);
            if (recentTokens.Count > 64) recentTokens.RemoveAt(0);
            logits = forward(next, startPos + i);
        }
        // When hiding reasoning, the decoder may still hold an in-thinking tail —
        // flush it through the same gate so nothing leaks to stdout.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
            Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");
        return (generated, totalDecoded);
    }

    /// <summary>
    /// Writes <paramref name="next"/> to stdout, handling the &lt;think&gt;/&lt;/think&gt; boundary tokens
    /// and dim-styling everything inside. Returns true when the emitted token counts toward the
    /// visible decode total (i.e. not a thinking-mode token and not a boundary marker).
    /// </summary>
    private static bool EmitToken(int next, GgufTokenizer tok, Utf8StreamDecoder streamDec, ref bool inThinking, bool hideThinking = false)
    {
        if (next == s_thinkTokenId)
        {
            inThinking = true;
            // No trailing \n: the model often emits its own leading newline inside the block,
            // and a double break before the reasoning starts looks noisy.
            Console.Write("\x1b[2m[Thinking...] ");
            return false;
        }
        if (next == s_endThinkTokenId && inThinking)
        {
            inThinking = false;
            Console.Write("\x1b[0m\n");
            return false;
        }
        // Stream through the same UTF-8 decoder regardless of mode so multibyte
        // sequences split across thinking/visible boundaries stay intact. When
        // hideThinking is set we still consume the bytes (so the decoder stays in
        // sync across the boundary) but discard the rendered output.
        var rendered = streamDec.Append(tok.DecodeBytes(next));
        if (!(hideThinking && inThinking))
            Console.Write(rendered);
        return !inThinking;
    }

    private static string s_arch = "qwen2"; // set during model load
    private static int s_thinkTokenId = -1;    // <think> token for any model using the <think>/</think> special-token convention
    private static int s_endThinkTokenId = -1; // </think> token for any model using the <think>/</think> special-token convention
    private static JinjaChatTemplate? s_jinja;  // parsed from GGUF tokenizer.chat_template

    /// <summary>
    /// Builds the stop token ID list: EOS plus any end-of-turn special tokens
    /// (<|eot_id|>, <|eom_id|>, <|end|>, <|im_end|>) present in the model vocabulary.
    /// </summary>
    private static IReadOnlyList<int> BuildStopTokenIds(GgufTokenizer tokenizer)
    {
        var stops = new HashSet<int> { tokenizer.EosTokenId };
        // End-of-turn tokens used by Llama 3/4, Mistral, Phi, etc.
        foreach (var name in new[] { "<|eot_id|>", "<|eom_id|>", "<|eot|>", "<|eom|>", "<|end|>", "<|im_end|>", "<|endoftext|>" })
            if (tokenizer.SpecialTokens.TryGetValue(name, out int id) && id != tokenizer.EosTokenId)
                stops.Add(id);
        return [.. stops];
    }

    private static string FormatPrompt(string userMessage, string? systemPrompt, bool enableThinking = true)
    {
        // SHARPI_RAW_PROMPT=1 bypasses the chat template entirely. Used for parity testing
        // against llama.cpp's --no-conversation mode (raw text completion). Not for normal use.
        if (Environment.GetEnvironmentVariable("SHARPI_RAW_PROMPT") == "1")
            return userMessage;

        // Use the model's own Jinja2 chat template when available (read from GGUF metadata).
        if (s_jinja != null)
        {
            // Qwen3 (dense) chat models behave poorly without a system message — they
            // end the turn after a few tokens for short prompts. The hardcoded fallback
            // path (below) injects this default; mirror it here for the same arch.
            // Note: qwen3moe is intentionally excluded — Qwen3-Coder appears to be
            // tuned to operate without a system prompt and gets HIGH-confidence on
            // <|endoftext|> when one is forced (logit ~29 vs ~14 with no system).
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3" ? "You are a helpful assistant." : null);
            var messages = JinjaChatTemplate.BuildMessages(userMessage, systemContent: effectiveSystemPrompt);
            return s_jinja.Render(new Dictionary<string, object?>
            {
                ["messages"]             = messages,
                ["add_generation_prompt"] = true,
                ["tools"]                = null,
                ["enable_thinking"]      = enableThinking,
            });
        }

        // Fallback: hardcoded templates for known architectures.
        var sb = new System.Text.StringBuilder();

        if (s_arch is "llama4")
        {
            // Llama 4: <|begin_of_text|><|header_start|>role<|header_end|>\n\nmessage<|eot|>
            sb.Append("<|begin_of_text|>");
            if (systemPrompt is not null)
                sb.Append($"<|header_start|>system<|header_end|>\n\n{systemPrompt}<|eot|>");
            sb.Append($"<|header_start|>user<|header_end|>\n\n{userMessage}<|eot|>");
            sb.Append("<|header_start|>assistant<|header_end|>\n\n");
        }
        else if (s_arch is "llama")
        {
            // Llama 3/3.1: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\nmessage<|eot_id|>
            sb.Append("<|begin_of_text|>");
            if (systemPrompt is not null)
                sb.Append($"<|start_header_id|>system<|end_header_id|>\n\n{systemPrompt}<|eot_id|>");
            sb.Append($"<|start_header_id|>user<|end_header_id|>\n\n{userMessage}<|eot_id|>");
            sb.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        else
        {
            // ChatML (Qwen, SmolLM, default): <|im_start|>role\nmessage<|im_end|>
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3moe" or "qwen3" ? "You are a helpful assistant." : null);
            if (effectiveSystemPrompt is not null)
                sb.Append($"<|im_start|>system\n{effectiveSystemPrompt}<|im_end|>\n");
            sb.Append($"<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n");
        }

        return sb.ToString();
    }
}
