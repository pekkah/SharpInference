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

        // Look up Qwen3 thinking-mode tokens for use in the decode loops
        if (s_arch is "qwen3moe" or "qwen3")
        {
            tokenizer.SpecialTokens.TryGetValue("<think>", out int thinkId);
            tokenizer.SpecialTokens.TryGetValue("</think>", out int endThinkId);
            s_thinkTokenId = thinkId > 0 ? thinkId : -1;
            s_endThinkTokenId = endThinkId > 0 ? endThinkId : -1;
        }
        using var cpuBackend = new CpuBackend();
        using var fwd = new ForwardPass(model, cpuBackend, hp);

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

        // MoE models are not yet supported on the hybrid GPU+CPU path. The GpuMoeFfn path
        // produces NaN/degenerate output (see https://github.com/pekkah/SharpInference/issues/2).
        // Fall back to CPU automatically when -g is -1 (auto), refuse explicit -g N for MoE.
        if (hp.IsMoE && nGpuLayers != 0)
        {
            if (nGpuLayers > 0)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] MoE models (this one is [yellow]{s_arch}[/]) are not yet supported on the hybrid GPU+CPU path; output is degenerate. Tracking issue: https://github.com/pekkah/SharpInference/issues/2");
                AnsiConsole.MarkupLine("Re-run without [yellow]-g[/] (or [yellow]-g 0[/]) to use the working CPU path. [yellow]--tq[/] is supported on CPU.");
                return 1;
            }
            // -g -1 (auto) → silently fall back to CPU
            AnsiConsole.MarkupLine("[yellow]Note:[/] [yellow]-g -1[/] requested, but MoE models run only on CPU until issue #2 is fixed. Falling back to CPU.");
            nGpuLayers = 0;
        }

        if (nGpuLayers == 0)
        {
            // CPU only
            if (settings.TurboQuant)
            {
                fwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
            }
            forward = fwd.Forward;
            prefill = tokens => fwd.Prefill(tokens);
            resetCache = settings.TurboQuant ? fwd.TqCache!.Reset : fwd.Cache.Reset;
            AnsiConsole.MarkupLine("[dim]Backend: [blue]CPU[/][/]");
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
                        if (settings.TurboQuant)
                        {
                            fwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
                            AnsiConsole.MarkupLine("[dim]TurboQuant: [green]enabled[/] (3-bit, window=256)[/]");
                        }

                        forward = fwd.Forward;
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
                        return RunSpeculativeSinglePrompt(settings, fwd, draftFwd, tokenizer, sp);
                    return RunSpeculativeInteractive(settings, fwd, draftFwd, tokenizer, sp);
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
        }
    }

    private static int RunSpeculativeSinglePrompt(Settings s,
        ForwardPass target, ForwardPass draft,
        GgufTokenizer tok, SamplingParams sp)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt);
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
        spec.Decode(sp.MaxNewTokens, sp.StopTokenIds ?? [], token =>
        {
            Console.Write(tok.Decode([token]));
            generated++;
        });
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {generated} tokens, {generated / (decodeMs / 1000):F1} t/s | " +
            $"Acceptance rate: {spec.AcceptanceRate:P0}[/]");
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

            var prompt = FormatPrompt(input, s.SystemPrompt);
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
            spec.Decode(sp.MaxNewTokens, sp.StopTokenIds ?? [], token =>
            {
                Console.Write(tok.Decode([token]));
                generated++;
            });
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{generated} tokens, {generated / (decodeMs / 1000):F1} t/s | Accept: {spec.AcceptanceRate:P0}[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static int RunSinglePrompt(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt);
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
        int generated = 0;
        bool inThinking = false; // set to true when model generates <think> token
        var recentTokens = new List<int>(64);
        for (int i = 0; i < sp.MaxNewTokens; i++)
        {
            var spWithHistory = sp.RepetitionPenalty != 1.0f && recentTokens.Count > 0
                ? sp with { PreviousTokens = recentTokens }
                : sp;
            int next = sp.Temperature <= 0 ? Sampler.Greedy(logits) : Sampler.Sample(logits, spWithHistory, rng);
            if (s.VerbosePrompt)
            {
                var logitsArr = logits.ToArray();
                var top5 = Enumerable.Range(0, logitsArr.Length).OrderByDescending(j => logitsArr[j]).Take(5)
                    .Select(j => $"{j}({logitsArr[j]:F2})");
                Console.Error.WriteLine($"[DBG] tok={i} next={next}('{tok.Decode([next])}') stop={sp.StopTokenIds.Contains(next)} top5:{string.Join(" ", top5)}");
            }
            if (sp.StopTokenIds.Contains(next)) break;
            if (next == s_thinkTokenId)
            {
                inThinking = true;
                Console.Write("\x1b[2m[Thinking...]\n");
            }
            else if (next == s_endThinkTokenId && inThinking)
            {
                inThinking = false;
                Console.Write("\x1b[0m\n");
            }
            else if (!inThinking)
            {
                Console.Write(tok.Decode([next]));
                generated++;
            }
            recentTokens.Add(next);
            if (recentTokens.Count > 64) recentTokens.RemoveAt(0);
            logits = forward(next, tokens.Count + i);
        }
        if (inThinking) Console.Write("\x1b[0m"); // reset if thinking was never closed
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
            bool inThinking = false; // set to true when model generates <think> token
            var recentTokens = new List<int>(64);
            for (int i = 0; i < sp.MaxNewTokens; i++)
            {
                var spWithHistory = sp.RepetitionPenalty != 1.0f && recentTokens.Count > 0
                    ? sp with { PreviousTokens = recentTokens }
                    : sp;
                int next = sp.Temperature <= 0 ? Sampler.Greedy(logits) : Sampler.Sample(logits, spWithHistory, rng);
                if (sp.StopTokenIds.Contains(next)) break;
                if (next == s_thinkTokenId)
                {
                    inThinking = true;
                    Console.Write("\x1b[2m[Thinking...]\n");
                }
                else if (next == s_endThinkTokenId && inThinking)
                {
                    inThinking = false;
                    Console.Write("\x1b[0m\n");
                }
                else if (!inThinking)
                {
                    Console.Write(tok.Decode([next]));
                    generated++;
                }
                recentTokens.Add(next);
                if (recentTokens.Count > 64) recentTokens.RemoveAt(0);
                logits = forward(next, tokens.Count + i);
            }
            if (inThinking) Console.Write("\x1b[0m"); // reset if thinking was never closed
            var decodeMs = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine();
            AnsiConsole.MarkupLine($"[dim]{generated} tokens, {generated / (decodeMs / 1000):F1} t/s[/]\n");

            if (s.SingleTurn) break;
        }
        return 0;
    }

    private static string s_arch = "qwen2"; // set during model load
    private static int s_thinkTokenId = -1;    // <think> token for Qwen3 thinking mode
    private static int s_endThinkTokenId = -1; // </think> token for Qwen3 thinking mode
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

    private static string FormatPrompt(string userMessage, string? systemPrompt)
    {
        // Use the model's own Jinja2 chat template when available (read from GGUF metadata).
        if (s_jinja != null)
        {
            // Qwen3 instruction-tuned models behave poorly without a system message —
            // they end the turn after a few tokens for short prompts. The hardcoded
            // fallback path (below) injects this same default; mirror it here so both
            // paths produce comparable output.
            string? effectiveSystemPrompt = systemPrompt
                ?? (s_arch is "qwen3moe" or "qwen3" ? "You are a helpful assistant." : null);
            var messages = JinjaChatTemplate.BuildMessages(userMessage, systemContent: effectiveSystemPrompt);
            return s_jinja.Render(new Dictionary<string, object?>
            {
                ["messages"]             = messages,
                ["add_generation_prompt"] = true,
                ["tools"]                = null,
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
