using System.ComponentModel;
using System.Diagnostics;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vision;
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
        public string? Prompt { get; set; }

        [CommandOption("-f|--file")]
        [Description("Read the prompt from a file (llama.cpp -f/--file). Overrides -p when both are given; useful for prompts longer than the shell's command-line limit.")]
        public string? PromptFile { get; init; }

        [CommandOption("--image <PATH>")]
        [Description("Path to a PNG image for multimodal input (Gemma 4 encoder-free vision). Repeatable for multiple images; reference each with an <image> marker in -p (left-to-right), or omit markers to prepend them. Requires --mmproj and a text prompt (-p). CPU only for now (-g 0).")]
        public string[]? ImagePaths { get; init; }

        [CommandOption("--mmproj")]
        [Description("Path to the multimodal projector GGUF (mmproj-*.gguf). Required with --image. Mirrors llama.cpp's --mmproj.")]
        public string? MmprojPath { get; init; }

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

        [CommandOption("--ngl|--n-gpu-layers|--gpu-layers|-g")]
        [Description("Layers on GPU (0=CPU only, -1=all, default: 0). Mirrors llama.cpp's --n-gpu-layers/--ngl.")]
        [DefaultValue(0)]
        public int NGpuLayers { get; init; }

        [CommandOption("--device")]
        [Description("GPU device to offload to: index (0,1,…), name (CUDA0, Vulkan1), or 'none' for CPU. " +
            "Default: auto. Single-device only (no multi-GPU split). Mirrors llama.cpp's --device.")]
        public string? Device { get; init; }

        [CommandOption("-c|--ctx-size")]
        [Description("Context size / max sequence length (0 = model default)")]
        [DefaultValue(0)]
        public int CtxSize { get; init; }

        [CommandOption("--tq")]
        [Description("Enable TurboQuant KV cache compression (3-bit, reduces VRAM ~5x)")]
        [DefaultValue(false)]
        public bool TurboQuant { get; init; }

        [CommandOption("--kv-type")]
        [Description("KV-cache element type for the CUDA backend: fp32 (default), bf16 (half the KV VRAM → ~2x context), or q8_0 (quarter → ~4x). Like llama.cpp --cache-type-k/v. Env: SHARPI_KV_DTYPE.")]
        public string? KvType { get; init; }

        [CommandOption("--model-draft|--draft-model")]
        [Description("Path to a smaller draft model for speculative decoding (greedy only, requires --temp 0). Mirrors llama.cpp's --model-draft.")]
        public string? DraftModelPath { get; init; }

        [CommandOption("--spec-lookahead|--draft-tokens")]
        [Description("Number of draft tokens per speculative step with --draft-model (default: 4)")]
        [DefaultValue(4)]
        public int SpecLookahead { get; init; }

        [CommandOption("--draft-lookup")]
        [Description("Speculative decoding via prompt-lookup (n-gram) drafting — proposes tokens by matching the generated tail against prompt+history; no draft model needed (greedy only, requires --temp 0)")]
        [DefaultValue(false)]
        public bool DraftLookup { get; init; }

        [CommandOption("--spec-type")]
        [Description("Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp). Mirrors llama.cpp.")]
        [DefaultValue("auto")]
        public string SpecTypeStr { get; init; } = "auto";

        [CommandOption("--spec-draft-n-max")]
        [Description("Max draft tokens per MTP step (issue #30 batched verify). Unset resolves via SHARPI_MTP_DRAFT_N, then defaults to 1 (a 2-token verify batch — the measured optimum). Values > 1 also need snapshot-ring slots: set SHARPI_MTP_BATCH_MAX >= drafts+1 (default 2; each extra slot costs ~150 MiB VRAM on 27B). Mirrors llama.cpp.")]
        [DefaultValue(0)]
        public int SpecDraftNMax { get; init; }

        [CommandOption("--spec-draft-n-min")]
        [Description("Min draft tokens per MTP step (default: 0). Mirrors llama.cpp. Currently rejected at parse time when > 0 since N=1 is the only supported draft length; issue #37.")]
        [DefaultValue(0)]
        public int SpecDraftNMin { get; init; }

        [CommandOption("--spec-draft-p-min")]
        [Description("Min draft probability for MTP probabilistic accept (default: 1.0 = strict argmax-match, byte-identical to no-MTP baseline). 0.75 mirrors llama.cpp; values in (0, 1) accept drafts whose softmax probability under the verifier meets the threshold even when they aren't argmax (issue #38).")]
        [DefaultValue(1.0f)]
        public float SpecDraftPMin { get; init; }

        [CommandOption("--min-batch-blas")]
        [Description("Minimum batch size to use OpenBLAS SGEMM in MatMulBatched (default: 16, crossover for Q4_K_M weights). Also settable via SHARPI_MIN_BATCH_BLAS env var.")]
        [DefaultValue(0)]
        public int MinBatchBlas { get; init; }

        [CommandOption("--prefill-dequant-cache-mb")]
        [Description("Dequant-once BLAS weight-cache budget in MiB for CPU prefill (issue #189): caches the F32 dequant per projection weight so chunked prefill re-pays no dequant (bit-identical). Auto (env SHARPI_PREFILL_DEQUANT_MB / fit-25%-RAM) by default; 0 = off, negative = unlimited. CPU only.")]
        [DefaultValue(long.MinValue)]
        public long PrefillDequantCacheMb { get; init; }

        [CommandOption("--repeat-penalty|--rep-penalty")]
        [Description("Repetition penalty (1.0 = disabled, >1.0 penalizes repeated tokens, default: 1.1). Mirrors llama.cpp's --repeat-penalty.")]
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

        [CommandOption("--thinking")]
        [Description("Enable reasoning mode (sets enable_thinking=true). Needed for Gemma 4 reasoning " +
            "finetunes, which default off because stock Gemma 4 instruct models aren't reasoning-trained.")]
        [DefaultValue(false)]
        public bool Thinking { get; init; }

        [CommandOption("--hide-thinking")]
        [Description("Hide reasoning output (the model still reasons; only the answer is shown)")]
        [DefaultValue(false)]
        public bool HideThinking { get; init; }

        [CommandOption("--max-thinking-tokens")]
        [Description("Maximum reasoning tokens before forcing </think>. 0 = unlimited (default). Not honored on the speculative-decode path.")]
        [DefaultValue(0)]
        public int MaxThinkingTokens { get; init; }

        // ── MoE expert-cache tuning (offloaded MoE models) ──
        // Good defaults are automatic: frequency-aware SLRU eviction, VRAM-sized cache,
        // and next-layer predictive prefetch are all ON without any flag. These knobs only
        // tune/disable that behaviour. Each is also settable via the named env var.
        [CommandOption("--no-moe-predict-prefetch")]
        [Description("MoE: disable next-layer predictive expert prefetch (Vulkan; on by default). Env: SHARPI_MOE_PREDICT_PREFETCH=0.")]
        [DefaultValue(false)]
        public bool NoMoePredictPrefetch { get; init; }

        [CommandOption("--moe-warmpin")]
        [Description("MoE: also pin the top-N hottest experts per layer into the GPU cache after warmup (default 0 = off; frequency-aware eviction already retains hot experts). Env: SHARPI_MOE_WARMPIN.")]
        public int? MoeWarmPin { get; init; }

        [CommandOption("--moe-warmpin-after")]
        [Description("MoE: expert accesses to observe before warm-pinning selects the hot set (default 512). Only used with --moe-warmpin. Env: SHARPI_MOE_WARMPIN_AFTER.")]
        [DefaultValue(0L)]
        public long MoeWarmPinAfter { get; init; }

        [CommandOption("--expert-stats")]
        [Description("MoE: write GPU expert-cache (SLRU) hit-rate stats to this file on exit. Env: SHARPI_EXPERT_STATS.")]
        public string? ExpertStatsPath { get; init; }

        // ── MoE expert placement (CPU vs GPU), issue #80. Wraps the existing all-or-nothing
        // SHARPI_CPU_MOE override the engine reads at forward-pass construction.
        [CommandOption("--cpu-moe|--cmoe")]
        [Description("MoE: keep ALL routed expert weights on the CPU (llama.cpp --cpu-moe). Sets SHARPI_CPU_MOE=1, overriding the VRAM-fit auto-select; SHARPI_CPU_MOE=0 in the env still forces on-GPU experts. Alias --cmoe (llama.cpp's single-dash -cmoe isn't representable: Spectre short options must be one character).")]
        [DefaultValue(false)]
        public bool CpuMoe { get; init; }

        [CommandOption("--n-cpu-moe|--ncmoe <N>")]
        [Description("MoE: keep the routed experts of N layers on the CPU (llama.cpp --n-cpu-moe). DEFERRED / not yet supported — SharpInference's expert placement is all-or-nothing (no per-layer split in the engine), so passing any value errors with that rationale. Use --cpu-moe (all on CPU) or omit (auto).")]
        public int? NCpuMoe { get; init; }
    }

    /// <summary>
    /// Translates the llama.cpp-style MoE placement flags (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>,
    /// issue #80) into the <c>SHARPI_CPU_MOE</c> override the engine reads when it builds the
    /// hybrid forward pass. <paramref name="cpuMoe"/> forces every routed expert onto the CPU
    /// (equivalent to <c>SHARPI_CPU_MOE=1</c> and the server's <c>CpuMoe=true</c>, issue #93); an
    /// explicit flag wins over an inherited env var, and its absence leaves the env (hence the
    /// engine's VRAM-fit auto-select) untouched. <paramref name="nCpuMoe"/> (partial per-layer
    /// placement) is <b>deferred</b>: the engine override is all-or-nothing, so any value is
    /// rejected via <paramref name="error"/>. Returns <c>false</c> (with <paramref name="error"/>
    /// set) when the caller should abort; the env side effect mirrors <see cref="GpuDevice.Resolve"/>.
    /// </summary>
    internal static bool TryApplyCpuMoeFlags(bool cpuMoe, int? nCpuMoe, out string? error)
    {
        if (nCpuMoe is int n)
        {
            error =
                $"--n-cpu-moe/--ncmoe ({n}) is not supported yet: SharpInference places routed MoE " +
                "experts all-or-nothing (the SHARPI_CPU_MOE override the engine reads has no per-layer " +
                "granularity), so a partial per-layer split can't be honored. Use --cpu-moe to keep all " +
                "routed experts on the CPU, or omit it to let VRAM fit auto-select (SHARPI_CPU_MOE=0 " +
                "forces on-GPU experts). Tracked in issue #80.";
            return false;
        }

        if (cpuMoe)
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");

        error = null;
        return true;
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        // --file/-f (llama.cpp): load the prompt from a file. Overrides -p; lets prompts exceed
        // the shell command-line length limit. Read as-is (no trailing-newline stripping).
        if (settings.PromptFile is { Length: > 0 } promptFile)
        {
            if (!File.Exists(promptFile))
            {
                AnsiConsole.MarkupLine($"[red]Prompt file not found:[/] {Markup.Escape(promptFile)}");
                return 1;
            }
            // Read failures (locked file, permissions, bad path) should fail loud + clean, not
            // throw a stack trace; Escape the message since paths can carry Spectre markup chars.
            try
            {
                settings.Prompt = File.ReadAllText(promptFile);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException
                                          or System.Security.SecurityException or NotSupportedException)
            {
                AnsiConsole.MarkupLine($"[red]Error reading prompt file:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        if (settings.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = settings.MinBatchBlas;

        // Resolve --device before any GPU call (it may set CUDA_VISIBLE_DEVICES, which the CUDA
        // driver only reads at first init; Vulkan takes the index explicitly below). `--device none`
        // forces the CPU path, overriding --n-gpu-layers.
        int gpuDeviceIndex;
        bool deviceNone;
        try
        {
            gpuDeviceIndex = GpuDevice.Resolve(settings.Device, out deviceNone);
        }
        catch (InvalidOperationException ex)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
        if (deviceNone && settings.NGpuLayers != 0)
            AnsiConsole.MarkupLine("[yellow]Note:[/] --device none overrides --ngl/-g; running on CPU.");
        int effNGpuLayers = deviceNone ? 0 : settings.NGpuLayers;

        // MoE expert-cache knobs are read from the environment inside the engine
        // (WarmPinConfig / HybridForwardPass / slot-manager dispose). Surface them as
        // CLI flags by setting the env var here — before any forward pass is built —
        // so an explicit flag overrides, and env-only use still works.
        if (settings.MoeWarmPin is int warmPin)  // explicitly passed (incl. 0 to force off)
            Environment.SetEnvironmentVariable("SHARPI_MOE_WARMPIN", warmPin.ToString());
        if (settings.MoeWarmPinAfter > 0)
            Environment.SetEnvironmentVariable("SHARPI_MOE_WARMPIN_AFTER", settings.MoeWarmPinAfter.ToString());
        if (settings.NoMoePredictPrefetch)
            Environment.SetEnvironmentVariable("SHARPI_MOE_PREDICT_PREFETCH", "0");

        // MoE expert placement (#80): --cpu-moe sets SHARPI_CPU_MOE=1; --n-cpu-moe is deferred
        // (the engine override is all-or-nothing) and fails fast with the rationale. Done here,
        // before any forward pass is built, so the engine constructor sees the override.
        if (!TryApplyCpuMoeFlags(settings.CpuMoe, settings.NCpuMoe, out string? cpuMoeError))
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(cpuMoeError!)}");
            return 1;
        }

        // KV-cache dtype (issue #179): surface SHARPI_KV_DTYPE as a flag. Set before
        // any forward pass is built so an explicit flag overrides; env-only use still
        // works. The CudaForwardPass constructor validates the value (fp32|bf16|q8_0).
        if (settings.KvType is { Length: > 0 })
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", settings.KvType);
        if (!string.IsNullOrEmpty(settings.ExpertStatsPath))
            Environment.SetEnvironmentVariable("SHARPI_EXPERT_STATS", settings.ExpertStatsPath);

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
        // Gemma 4 brackets its reasoning in <|channel>thought … <channel|> instead. Route it
        // through the same think/end-think machinery so the markers don't leak into output.
        else if (tokenizer.SpecialTokens.TryGetValue("<|channel>", out int channelId)
            && tokenizer.SpecialTokens.TryGetValue("<channel|>", out int endChannelId)
            && channelId > 0 && endChannelId > 0)
        {
            s_thinkTokenId = channelId;
            s_endThinkTokenId = endChannelId;
        }

        // Gemma 4's stock instruct models (E4B-it, 12B-it) bracket a <|channel>thought block in
        // their chat template but are NOT trained to reason — rendering enable_thinking=true makes
        // them try to fill a think section they weren't trained for and the output degenerates. So
        // Gemma 4 defaults thinking OFF (its recommended config). Reasoning FINETUNES that share the
        // same arch/template (e.g. the agentic v2) DO need it on — and nothing in the GGUF metadata
        // distinguishes a reasoning-trained Gemma 4 from a stock one (identical chat template, tokens,
        // sampling hints), so --thinking is the explicit opt-in. (--no-thinking forces it off for any
        // model and wins if both are passed.)
        bool gemma4DefaultsThinkingOff = s_arch == "gemma4" && !settings.Thinking;
        s_noThinking = settings.NoThinking || gemma4DefaultsThinkingOff;
        if (gemma4DefaultsThinkingOff && !settings.NoThinking)
            AnsiConsole.MarkupLine("[dim]Gemma 4 defaults to --no-thinking (stock instruct models aren't " +
                "reasoning-trained). For a reasoning finetune pass --thinking " +
                "(recommended: --temp 1.0 --top-k 64 --top-p 0.95).[/]");

        // Greedy on a reasoning model tends to "wait, but actually" itself into infinite
        // loops; --no-thinking sidesteps the issue since the model won't reason at all.
        if (s_thinkTokenId > 0 && settings.Temperature == 0f && !s_noThinking)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Greedy decoding (--temp 0) on a reasoning model often produces");
            AnsiConsole.MarkupLine("infinite \"wait, but actually\" loops. Consider [yellow]--temp 0.6 --top-p 0.95 --top-k 20[/].");
        }

        // Image input (issue #250): encoder-free Gemma 4 vision. CPU-only single-prompt path.
        // Validate the preconditions up front so we fail fast before building any forward pass.
        if (settings.ImagePaths is { Length: > 0 } imagePaths)
        {
            if (s_arch != "gemma4")
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] --image is only supported for Gemma 4 models (model arch: {s_arch}).");
                return 1;
            }
            if (settings.MmprojPath is not { Length: > 0 })
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --image requires --mmproj <mmproj.gguf> (the multimodal projector).");
                return 1;
            }
            if (!File.Exists(settings.MmprojPath))
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] mmproj file not found: {Markup.Escape(settings.MmprojPath)}");
                return 1;
            }
            foreach (var imgPath in imagePaths)
            {
                if (!File.Exists(imgPath))
                {
                    AnsiConsole.MarkupLine($"[red]Error:[/] image file not found: {Markup.Escape(imgPath)}");
                    return 1;
                }
            }
            if (settings.Prompt is not { Length: > 0 })
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --image requires a text prompt ([yellow]-p \"...\"[/]); interactive image chat is not supported yet.");
                return 1;
            }
        }

        using var cpuBackend = new CpuBackend();

        // Hybrid GDN models (qwen35moe) run via the dedicated HybridGdnForwardPass
        // (CPU) or CudaHybridGdnForwardPass (GPU). Features that touch the per-token
        // GDN state are not supported because the rank-1 recurrence is destructive.
        if (hp.IsHybridSsm && settings.TurboQuant)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] TurboQuant is not supported for hybrid GDN models (no KV cache on GDN layers).");
            return 1;
        }
        if (hp.IsHybridSsm && (settings.DraftModelPath is not null || settings.DraftLookup))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] Speculative decoding is not supported for hybrid GDN models (GDN state is destructively updated and cannot be rewound).");
            return 1;
        }

        // Build the appropriate CPU forward pass. The GPU branches below construct
        // their own (CudaHybridGdnForwardPass for hybrid + CUDA; the existing Cuda/
        // CudaHybrid paths for non-hybrid). For hybrid + GPU we still build the
        // CPU baseline so the bridge can reuse small helpers, but it stays unused.
        ForwardPass? fwd = null;
        HybridGdnForwardPass? hybridFwd = null;
        // IForwardPass handle for MtpDecoder integration (issue #32). Captured when the
        // chosen forward pass ships an MTP head. The actual MTP gating happens later in
        // RunSinglePrompt / RunInteractive based on sp.SpecType.
        IForwardPass? mtpFwd = null;
        if (hp.IsHybridSsm && effNGpuLayers == 0)
        {
            hybridFwd = new HybridGdnForwardPass(model, cpuBackend, hp);
            if (hybridFwd.HasMtpHead) mtpFwd = hybridFwd;
        }
        else if (!hp.IsHybridSsm)
        {
            // #189 dequant cache: only the pure-CPU path (no GPU offload) runs the batched
            // CPU prefill that consults it; under -g it would be a wasted F32 model copy.
            long dequantBytes = effNGpuLayers != 0
                ? 0
                : settings.PrefillDequantCacheMb == long.MinValue
                    ? long.MinValue // auto / SHARPI_PREFILL_DEQUANT_MB
                    : ForwardPass.MbToBudgetBytes(settings.PrefillDequantCacheMb);
            fwd = new ForwardPass(model, cpuBackend, hp, prefillDequantCacheBytes: dequantBytes);
        }

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

        int nGpuLayers = effNGpuLayers;

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
                    // MoE; routed experts stream through the CudaExpertSlotManager SLRU).
                    // TQ on CUDA requires head_dim ∈ {128, 256}.
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
                string ffnKindCpu = hp.IsMoE ? "MoE" : "dense FFN";
                AnsiConsole.MarkupLine($"[dim]Backend: [blue]CPU[/] (hybrid GDN + {ffnKindCpu})[/]");
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
                // qwen35moe (hybrid GDN+MoE) takes a dedicated CUDA forward pass that
                // routes the 30 recurrent blocks to CPU and the 10 attention layers +
                // MoE FFN to GPU via the CudaExpertSlotManager SLRU. Layer placement is
                // implicit (driven by hp.LayerTypes), so we skip TierPlanner here.
                if (hp.IsHybridSsm)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = new LayerPlacement(
                        GpuLayers: hp.NumLayers,
                        CpuLayers: 0,
                        GpuWeightBytes: 0,
                        GpuKvBytes: 0,
                        RecommendedCtxSize: ctxSize > 0 ? ctxSize : Math.Min(hp.ContextLength, 4096));
                    var chgdn = new CudaHybridGdnForwardPass(model, cuda, hp, placement);
                    gpuFwd = chgdn;
                    if (chgdn.HasMtpHead) mtpFwd = chgdn;
                    forward = chgdn.Forward;
                    prefill = tokens => chgdn.Prefill(tokens);
                    resetCache = chgdn.ResetCache;
                    int gdnLayers = 0, attnLayers = 0;
                    for (int i = 0; i < hp.NumLayers; i++)
                        if (hp.LayerTypes![i] == LayerType.Attention) attnLayers++; else gdnLayers++;
                    string ffnKind = hp.IsMoE
                        ? (chgdn.IsMoeOnCpu ? "MoE on CPU" : "MoE on GPU")
                        : "dense FFN on CPU";
                    AnsiConsole.MarkupLine($"[dim]Backend: [green]CUDA hybrid GDN[/] ({cuda.Name}, {gdnLayers} GDN + {attnLayers} attn on GPU + {ffnKind})[/]");
                }
                else
                {

                // For -g -1 (auto), run TierPlanner against CUDA's VRAM and use the
                // resulting layer split — same logic as the Vulkan branch. Without this,
                // a model bigger than VRAM (e.g. Qwen3-Coder 30B-A3B in 12 GB) would
                // attempt full-offload via CudaForwardPass and silently OOM.
                int cudaGpuLayers;
                bool moeAutoNeedsHybrid = false;
                if (nGpuLayers == -1)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, kvDtype: CudaForwardPass.ResolveConfiguredKvDType());
                    cudaGpuLayers = placement.GpuLayers;

                    // Gemma 4 KV-share constraint: the shared-KV source layers (E4B:
                    // 22 and 23) must live on the same tier as the shared-KV tail
                    // layers (24..41) because cross-tier KV reads are not wired.
                    // TierPlanner doesn't model this and may return a value that
                    // straddles the boundary (e.g. 30). Clamp UP to NumLayers when
                    // possible — TierPlanner's per-layer KV budget ignores that
                    // shared-KV-aliased layers don't grow their own cache, so it's
                    // pessimistic by ~18 layers × full-ctx-KV; full offload almost
                    // always fits when the auto value already exceeded the safe max.
                    if (hp.KvSourceLayer is { } ksl)
                    {
                        int minSrc = int.MaxValue;
                        for (int i = 0; i < hp.NumLayers; i++)
                            if (ksl[i] >= 0 && ksl[i] < minSrc) minSrc = ksl[i];
                        if (minSrc != int.MaxValue
                            && cudaGpuLayers > minSrc
                            && cudaGpuLayers < hp.NumLayers)
                        {
                            AnsiConsole.MarkupLine(
                                $"[dim]TierPlanner returned -g {cudaGpuLayers}, which would " +
                                $"cross the Gemma 4 KV-share boundary (sources <= {minSrc}); " +
                                $"promoting to full offload (-g {hp.NumLayers}). " +
                                $"Pass -g {minSrc} explicitly if VRAM is tight.[/]");
                            cudaGpuLayers = hp.NumLayers;
                        }
                    }

                    // Issue #215: a MoE model whose routed experts can't all stay resident must use the
                    // hybrid path (which streams experts via SLRU or runs them on CPU), even though the
                    // planner places the whole attention trunk on GPU (GpuLayers == NumLayers). Without
                    // this, auto (-g -1) falls through to full-offload CudaForwardPass and thrashes/OOMs —
                    // the very case TierPlanner was added to avoid.
                    moeAutoNeedsHybrid = hp.IsMoE
                        && cudaGpuLayers == hp.NumLayers
                        && placement.MoeRoutedExpertBytes > placement.ExpertCacheBudgetBytes;
                    if (moeAutoNeedsHybrid)
                    {
                        AnsiConsole.MarkupLine(
                            $"[dim]MoE routed experts ({placement.MoeRoutedExpertBytes / (1024.0 * 1024):F0} MB) " +
                            $"exceed the GPU expert-cache budget ({placement.ExpertCacheBudgetBytes / (1024.0 * 1024):F0} MB); " +
                            $"using the hybrid path (CPU-MoE / SLRU streaming) instead of full offload.[/]");
                    }
                }
                else
                {
                    cudaGpuLayers = nGpuLayers;
                }

                bool wantHybrid = (cudaGpuLayers > 0 && cudaGpuLayers < hp.NumLayers) || moeAutoNeedsHybrid;
                if (wantHybrid)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    // pinGpuLayers prices the expert-cache budget (read by the MoE CPU-vs-SLRU
                    // auto-decision) for this exact split. cudaGpuLayers equals the auto value on
                    // the -g -1 path, so pinning is a no-op there and only matters on explicit -g N
                    // (#224). A `with { GpuLayers = }` override would leave the budget stale.
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, kvDtype: CudaForwardPass.ResolveConfiguredKvDType(),
                        pinGpuLayers: cudaGpuLayers);

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
                } // end !IsHybridSsm
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
            // Vulkan path. There is no Vulkan equivalent of CudaHybridGdnForwardPass yet,
            // so qwen35moe must use --backend cuda or -g 0.
            if (hp.IsHybridSsm)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Hybrid GDN models (qwen35moe) are not supported on the Vulkan backend yet. Use [yellow]--backend cuda[/] or [yellow]-g 0[/] (CPU).");
                return 1;
            }

            var gpu = new VulkanBackend(gpuDeviceIndex);
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
                    // Hybrid: N layers GPU, rest CPU. nGpuLayers is the auto value on -g -1 and the
                    // explicit count otherwise; pinGpuLayers prices weights/KV/budget for it (#224).
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize, pinGpuLayers: nGpuLayers);

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
            SpecType = ParseSpecType(settings.SpecTypeStr),
            SpecDraftNMax = settings.SpecDraftNMax,
            SpecDraftNMin = settings.SpecDraftNMin,
            SpecDraftPMin = settings.SpecDraftPMin,
        };
        var rng = settings.Seed >= 0 ? new Random(settings.Seed) : new Random();

        // Speculative decoding path (requires --draft-model and --temp 0). Supported
        // targets: pure CPU (-g 0) and full CUDA offload of a dense model (issue #207 —
        // packed k-token verify via CudaForwardPass.BatchVerify). Vulkan and the partial-
        // offload hybrids fall back to normal generation: without a batched verify,
        // speculation costs k sequential target forwards per step and is never a win.
        if (settings.DraftModelPath is not null || settings.DraftLookup)
        {
            bool cudaSpecTarget = gpuFwd is CudaForwardPass { SupportsBatchVerify: true };
            // Sampled speculative decoding (issue #178): temp>0 now drives distribution-preserving
            // spec sampling on the model-draft path (greedy at temp 0 stays byte-stable). Gated to
            // model drafts (lookup proposals expose no q), to non-penalized/-biased sampling (draft
            // and target must agree on the distribution), and bypassable via SHARPI_SPEC_SAMPLE=0.
            bool sampledSpec = settings.Temperature > 0f;
            bool specSampleDisabled = Environment.GetEnvironmentVariable("SHARPI_SPEC_SAMPLE") == "0";
            bool hasPenalty = sp.RepetitionPenalty != 1f && sp.PreviousTokens is { Count: > 0 };
            bool hasBias = sp.LogitBias is { Count: > 0 };
            if (settings.DraftModelPath is not null && settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] --draft-model and --draft-lookup are mutually exclusive.");
                return 1;
            }
            if (nGpuLayers != 0 && !cudaSpecTarget)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] Speculative decoding requires pure CPU (-g 0) or full CUDA offload of a dense or Gemma-4 model. Falling back to normal generation.");
            }
            else if (sampledSpec && settings.DraftLookup)
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] --draft-lookup supports greedy (--temp 0) only; sampled speculative decoding needs --draft-model. Falling back to normal generation.");
            }
            else if (sampledSpec && specSampleDisabled)
            {
                AnsiConsole.MarkupLine("[yellow]Note:[/] SHARPI_SPEC_SAMPLE=0 — sampled speculative decoding disabled; using normal sampled generation.");
            }
            else if (sampledSpec && (hasPenalty || hasBias))
            {
                AnsiConsole.MarkupLine("[yellow]Warning:[/] sampled speculative decoding does not yet support --repeat-penalty / logit bias (draft and target must share the same distribution); falling back to normal generation.");
            }
            else if (settings.DraftLookup)
            {
                // Prompt-lookup drafting (issue #207): no draft model — proposals come from
                // n-gram matches against prompt + generated history, verified by the same
                // batched-verify step. Floor is ~baseline (no match → plain decode step).
                try
                {
                    IForwardPass lookupTarget = cudaSpecTarget ? (CudaForwardPass)gpuFwd! : fwd!;
                    AnsiConsole.MarkupLine($"[dim]Speculative decoding: prompt-lookup (n-gram) drafting | Lookahead k={settings.SpecLookahead}[/]");
                    if (settings.Prompt is not null)
                        return RunSpeculativeSinglePrompt(settings, lookupTarget, null, tokenizer, sp, rng);
                    return RunSpeculativeInteractive(settings, lookupTarget, null, tokenizer, sp, rng);
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
                    if (cudaSpecTarget)
                    {
                        var target = (CudaForwardPass)gpuFwd!;
                        // The draft gets its OWN CudaBackend: graph capture state is one
                        // exec graph per backend instance, so sharing the target's backend
                        // would have the draft's decode graph clobber the target's.
                        //
                        // Clamp the draft's context: the decoder advances both passes in
                        // lockstep, so the draft never sees a position past the target's
                        // window — and unless the user pinned -c explicitly, cap it at 4096
                        // (the decode runners bound generation by BOTH windows, so a smaller
                        // draft ring only caps session length, never indexes out of range).
                        // Passing 0 would size the draft's KV from the VRAM left AFTER the
                        // target loaded — measured on the 12 GB 4070 Ti: the 0.6B draft
                        // grabbed a 34K-ctx / ~7 GB ring next to the 8B target (decode
                        // 75 → 13 t/s, WDDM paging); even a target-matched 12K fp32 ring
                        // (~2.8 GB) left so little headroom that the draft's weights paged
                        // in and out every step (draft forward 2.9 → ~15 ms, decode 34 t/s).
                        int draftCtx = ctxSize > 0 ? target.MaxSeqLen : Math.Min(target.MaxSeqLen, 4096);
                        using var draftCuda = CudaBackend.Create();
                        using var draftFwd = new CudaForwardPass(draftModel, draftCuda, draftHp, draftCtx);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([green]CUDA[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, target, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, target, draftFwd, tokenizer, sp, rng);
                    }
                    else
                    {
                        using var draftCpuBackend = new CpuBackend();
                        using var draftFwd = new ForwardPass(draftModel, draftCpuBackend, draftHp);
                        AnsiConsole.MarkupLine($"[dim]Draft model: {draftHp.NumLayers}L, {draftHp.EmbeddingDim}d ([blue]CPU[/]) | Lookahead k={settings.SpecLookahead}[/]");
                        if (settings.Prompt is not null)
                            return RunSpeculativeSinglePrompt(settings, fwd!, draftFwd, tokenizer, sp, rng);
                        return RunSpeculativeInteractive(settings, fwd!, draftFwd, tokenizer, sp, rng);
                    }
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
            if (settings.ImagePaths is { Length: > 0 })
                return RunImagePrompt(settings, (gpuFwd as IForwardPass) ?? fwd!, tokenizer, hp, sp, rng);
            if (settings.Prompt is not null)
                return RunSinglePrompt(settings, forward, prefill, tokenizer, sp, rng, mtpFwd);
            return RunInteractive(settings, forward, prefill, resetCache, tokenizer, sp, rng, mtpFwd);
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

    /// <summary>
    /// True when a prompt of <paramref name="promptTokens"/> tokens leaves no room to
    /// speculate inside BOTH context windows (prompt + lookahead + 1 correction token).
    /// Prints an actionable error: the typical trigger is the CUDA draft's 4096-token
    /// KV ring cap when <c>-c</c> isn't pinned, where prefilling past the ring would
    /// write K/V out of range and a tail prompt would silently emit zero tokens.
    /// </summary>
    private static bool SpecWindowExhausted(int promptTokens,
        IForwardPass target, IForwardPass? draft, int lookahead)
    {
        int window = Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue);
        if (promptTokens + lookahead + 1 < window) return false;
        AnsiConsole.MarkupLine(
            $"[red]Error:[/] prompt ({promptTokens} tokens) + lookahead ({lookahead}) does not fit the " +
            $"speculative context window ({window} tokens" +
            (draft is not null && draft.MaxSeqLen < target.MaxSeqLen
                ? $", limited by the draft model's KV ring — pass -c to size it explicitly"
                : "") +
            "). Shorten the prompt, raise -c, or drop --draft-model/--draft-lookup.");
        return true;
    }

    private static int RunSpeculativeSinglePrompt(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        // The prompt must fit BOTH context windows BEFORE any prefill runs — the
        // draft's ring may be much smaller than the target's (the CUDA spec path
        // caps it at 4096 when -c isn't pinned), and a too-long prompt would write
        // K/V past the ring's end during draft.Prefill, not merely cap generation.
        if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
            return 1;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        var sw = Stopwatch.StartNew();
        // Prefill (batched-trunk path — the per-token Forward loop this replaces was
        // ~30× slower on the CUDA target). A null draft means prompt-lookup mode.
        ReadOnlySpan<float> targetLogits = target.Prefill(tokens);
        ReadOnlySpan<float> draftLogits = draft is not null ? draft.Prefill(tokens) : default;
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        SpeculativeDecoder spec;
        if (draft is not null)
        {
            // temp>0 → sampled (distribution-preserving) accept; temp 0 → greedy (byte-stable).
            spec = sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead)
                : new SpeculativeDecoder(target, draft, s.SpecLookahead);
            spec.Initialize(tokens.Count, targetLogits, draftLogits);
        }
        else
        {
            spec = new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead);
            spec.Initialize(tokens, targetLogits);
        }

        // Bound generation by BOTH context windows (the draft's may be smaller — the CUDA
        // spec path caps its KV ring), leaving lookahead headroom for the last spec step.
        // The guard above ensures maxNew >= 1 here.
        int maxNew = Math.Min(sp.MaxNewTokens,
            Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);
        if (maxNew < sp.MaxNewTokens)
            AnsiConsole.MarkupLine($"[yellow]Note:[/] generation capped at {maxNew} tokens by the context window.");

        sw.Restart();
        int generated = 0;
        int totalDecoded = 0;
        bool inThinking = false;
        var streamDec = new Utf8StreamDecoder();
        bool hideThinking = s.HideThinking;
        spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
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
            $" | Acceptance rate: {spec.AcceptanceRate:P0} | " +
            $"draft {spec.DraftMs:F0}ms / verify {spec.VerifyMs:F0}ms / commit {spec.CommitMs:F0}ms[/]");
        return 0;
    }

    private static int RunSpeculativeInteractive(Settings s,
        IForwardPass target, IForwardPass? draft,
        GgufTokenizer tok, SamplingParams sp, Random rng)
    {
        AnsiConsole.MarkupLine("[green]Interactive chat (speculative decoding).[/] Type a message, or [yellow]/exit[/] to quit.\n");
        var spec = draft is not null
            ? (sp.Temperature > 0f
                ? new SpeculativeDecoder(target, draft, sp, rng, s.SpecLookahead)
                : new SpeculativeDecoder(target, draft, s.SpecLookahead))
            : new SpeculativeDecoder(target, new PromptLookupDraft(), s.SpecLookahead);

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
            var tokens = tok.Encode(prompt);

            // Same pre-prefill window guard as the single-prompt runner: the draft
            // ring may be smaller than the target's window, and prefilling past it
            // writes K/V out of range rather than just capping generation.
            if (SpecWindowExhausted(tokens.Count, target, draft, s.SpecLookahead))
                continue;

            target.ResetCache();
            draft?.ResetCache();

            var sw = Stopwatch.StartNew();
            ReadOnlySpan<float> targetLogits = target.Prefill(tokens);

            if (draft is not null)
                spec.Initialize(tokens.Count, targetLogits, draft.Prefill(tokens));
            else
                spec.Initialize(tokens, targetLogits);

            int maxNew = Math.Min(sp.MaxNewTokens,
                Math.Min(target.MaxSeqLen, draft?.MaxSeqLen ?? int.MaxValue) - tokens.Count - s.SpecLookahead - 1);

            sw.Restart();
            int generated = 0;
            int totalDecoded = 0;
            bool inThinking = false;
            var streamDec = new Utf8StreamDecoder();
            bool hideThinking = s.HideThinking;
            spec.Decode(maxNew, sp.StopTokenIds ?? [], token =>
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
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s_noThinking);
        var tokens = tok.Encode(prompt);

        // SHARPI_RAW_PROMPT bypasses the chat template, so we need to add BOS
        // manually for models that expect it (e.g. Gemma 4 with add_bos_token=true).
        // The chat-template path already injects bos_token via Jinja.
        bool isRaw = Environment.GetEnvironmentVariable("SHARPI_RAW_PROMPT") == "1";
        if (isRaw && tok.AddBosToken && tok.BosTokenId >= 0
            && (tokens.Count == 0 || tokens[0] != tok.BosTokenId))
        {
            var withBos = new List<int>(tokens.Count + 1) { tok.BosTokenId };
            withBos.AddRange(tokens);
            tokens = withBos;
        }

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

        // MTP self-speculative decode (issue #32). Activates when the model ships an
        // MTP head AND sampling is greedy AND the user disabled thinking on the chat
        // template (--no-thinking) AND sp.SpecType permits. SHARPI_DISABLE_MTP=1 is
        // a back-compat off-switch that wins.
        bool useMtp = ResolveCliMtp(mtpFwd, sp, s_noThinking, out string? mtpReject);
        if (mtpReject != null)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] {Markup.Escape(mtpReject)}");
            return 1;
        }

        sw.Restart();
        int generated, totalDecoded;
        float? acceptanceRate = null;
        long mtpAccepted = 0, mtpEmitted = 0;
        if (useMtp)
        {
            (generated, totalDecoded, acceptanceRate, mtpAccepted, mtpEmitted) =
                DecodeLoopMtp(mtpFwd!, tokens, logits, tok, sp, s.HideThinking, s.VerbosePrompt);
        }
        else
        {
            (generated, totalDecoded) =
                DecodeLoop(forward, logits, tokens.Count, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {tokens.Count} tokens, {tokens.Count / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            (acceptanceRate is float ar ? $" | MTP accept: {ar:P0} ({mtpAccepted}/{mtpEmitted})" : "") +
            "[/]");
        return 0;
    }

    /// <summary>User-facing prompt marker for an image position (mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder before templating). One per <c>--image</c>, left-to-right.</summary>
    private const string ImageMarker = "<image>";

    /// <summary>
    /// Single-prompt image→text for Gemma 4 (issue #250), one or more images. Each image is
    /// preprocessed and run through the encoder-free projector to soft tokens, then spliced
    /// into the decoder via <see cref="ForwardPass.ForwardEmbedding"/>, wrapped in the runtime
    /// markers (<c>&lt;|image&gt;</c> … soft tokens … <c>&lt;image|&gt;</c>).
    ///
    /// Placement: each <c>&lt;image&gt;</c> marker in the prompt is mapped to the model's
    /// <c>&lt;|image|&gt;</c> placeholder (id 258880), the prompt is rendered through the model's
    /// own chat template (so BOS / <c>&lt;|turn&gt;</c> / thinking handling matches the text path —
    /// Gemma 4 uses <c>&lt;|turn&gt;role\n…&lt;turn|&gt;</c>, NOT Gemma 3's <c>&lt;start_of_turn&gt;</c>),
    /// then each placeholder token in the token stream is expanded with its image's soft tokens
    /// in order. With no markers, the images are prepended to the user turn. CPU-only: the
    /// embedding-injection seam lives on <see cref="ForwardPass"/>.
    /// </summary>
    private static int RunImagePrompt(Settings s,
        IForwardPass fwd, GgufTokenizer tok, ModelHyperparams hp,
        SamplingParams sp, Random rng)
    {
        if (!fwd.SupportsEmbeddingInput)
        {
            AnsiConsole.MarkupLine("[red]Error:[/] the selected backend does not support image embedding input. " +
                "Image input runs on CPU ([yellow]-g 0[/]) or full CUDA offload ([yellow]-g -1[/] of a model that fits VRAM); " +
                "partial-offload hybrids and the Vulkan backend are not supported yet.");
            return 1;
        }

        var imagePaths = s.ImagePaths!;
        int nImages = imagePaths.Length;

        // Reconcile the number of <image> markers in the prompt with the number of --image
        // files. No markers → prepend one placeholder per image (in --image order). Otherwise
        // the counts must match so the i-th marker pairs with the i-th --image.
        int markerCount = CountOccurrences(s.Prompt!, ImageMarker);
        string userMsg;
        if (markerCount == 0)
        {
            userMsg = string.Concat(Enumerable.Repeat("<|image|>", nImages)) + s.Prompt;
        }
        else if (markerCount == nImages)
        {
            userMsg = s.Prompt!.Replace(ImageMarker, "<|image|>");
        }
        else
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] prompt has {markerCount} '{ImageMarker}' marker(s) but " +
                $"{nImages} --image file(s) were given; the counts must match (or omit markers to prepend the images).");
            return 1;
        }

        using var vision = VisionModel.Open(s.MmprojPath!);
        var embedder = new GemmaUvVisionEmbedder(vision);
        int embd = hp.EmbeddingDim;

        // Project every image to its soft-token block up front, in --image order.
        var blocks = new (float[] Soft, int NTok)[nImages];
        int totalSoft = 0;
        for (int i = 0; i < nImages; i++)
        {
            byte[] rgb;
            int srcW, srcH;
            try
            {
                rgb = ImageIO.LoadRgb(imagePaths[i], out srcW, out srcH);
            }
            catch (Exception ex) when (ex is IOException or NotSupportedException or InvalidDataException
                                          or UnauthorizedAccessException or System.Security.SecurityException)
            {
                AnsiConsole.MarkupLine($"[red]Error reading image[/] {Markup.Escape(imagePaths[i])}: {Markup.Escape(ex.Message)}");
                return 1;
            }
            var img = ImagePreprocessor.Preprocess(rgb, srcW, srcH, vision);
            var soft = embedder.Forward(img.Chw, img.Height, img.Width, out int nTok);
            blocks[i] = (soft, nTok);
            totalSoft += nTok;
            AnsiConsole.MarkupLine($"[dim]Image {i + 1}/{nImages}: {srcW}x{srcH} -> {img.Width}x{img.Height} -> {nTok} soft tokens[/]");
        }

        int imgOpen = tok.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
        int imgClose = tok.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;
        int placeholder = tok.SpecialTokens.TryGetValue("<|image|>", out var ph) ? ph : 258880;

        var prompt = FormatPrompt(userMsg, s.SystemPrompt, enableThinking: !s_noThinking);
        var allTokens = tok.Encode(prompt).ToList();
        int placeholdersFound = allTokens.Count(t => t == placeholder);
        if (placeholdersFound != nImages)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] expected {nImages} image placeholder token(s) (<|image|>, {placeholder}) " +
                $"after templating but found {placeholdersFound}; this model may not support image input.");
            return 1;
        }

        var sw = Stopwatch.StartNew();
        int pos = 0;
        int imgIdx = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in allTokens)
        {
            if (id == placeholder)
            {
                var (soft, nTok) = blocks[imgIdx++];
                logits = fwd.Forward(imgOpen, pos++);
                for (int t = 0; t < nTok; t++)
                    logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
                logits = fwd.Forward(imgClose, pos++);
            }
            else
            {
                logits = fwd.Forward(id, pos++);
            }
        }
        var prefillMs = sw.Elapsed.TotalMilliseconds;

        if (!s.NoDisplayPrompt)
            Console.Write(s.Prompt);

        sw.Restart();
        var (generated, totalDecoded) =
            DecodeLoop(fwd.Forward, logits, pos, tok, sp, rng, s.VerbosePrompt, s.HideThinking, s.MaxThinkingTokens);
        var decodeMs = sw.Elapsed.TotalMilliseconds;

        Console.WriteLine();
        AnsiConsole.MarkupLine($"\n[dim]Prefill: {pos} tokens ({totalSoft} image + {pos - totalSoft} text), " +
            $"{pos / (prefillMs / 1000):F1} t/s | " +
            $"Decode: {totalDecoded} tokens, {totalDecoded / (decodeMs / 1000):F1} t/s" +
            (totalDecoded > generated ? $" ({generated} visible, {totalDecoded - generated} thinking)" : "") +
            "[/]");
        return 0;
    }

    /// <summary>Count non-overlapping occurrences of <paramref name="needle"/> in <paramref name="haystack"/>.</summary>
    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0, idx = 0;
        while ((idx = haystack.IndexOf(needle, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += needle.Length;
        }
        return count;
    }

    // Decides whether to engage the MTP self-speculative path on the CLI side.
    // Mirrors the InferenceEngine gate but reads `--no-thinking` from CLI settings
    // (vs. the engine which inspects whether the model has think tokens registered).
    // The CLI path can engage MTP even on models that registered <think>/</think>,
    // as long as the user passed --no-thinking — in that case the template renders
    // with enable_thinking=false and no think tokens appear in the prompt or output.
    private static bool ResolveCliMtp(IForwardPass? mtpFwd, SamplingParams sp, bool noThinking, out string? rejectReason)
    {
        rejectReason = null;
        bool envDisabled = Environment.GetEnvironmentVariable("SHARPI_DISABLE_MTP") == "1";
        bool eligible = mtpFwd is not null
                        && mtpFwd.HasMtpHead
                        && sp.Temperature <= 0f
                        && noThinking
                        && !envDisabled;

        // Spec-decode CLI flag validation:
        //   --spec-draft-p-min: clamped to [0, 1] at the MtpDecoder boundary. Accepted
        //     on any spec path. 1.0 (or 0 / unset) = pure argmax-match; p ∈ (0,1) is
        //     llama.cpp's probabilistic-accept rule from #38. Reject obviously bad input.
        //   --spec-draft-n-min: accepted as a no-op under N=2 batched verify.
        //   --spec-draft-n-max == 2 enables #30 batched verify; > 2 still rejected.
        if (sp.SpecType != SpecType.None)
        {
            if (sp.SpecDraftPMin > 1f)
            {
                rejectReason = $"--spec-draft-p-min={sp.SpecDraftPMin} must be in [0, 1].";
                return false;
            }
        }

        // Max drafts per step = batch capacity − 1 (the certain token rides in the
        // batch). The pass's snapshot-ring capacity bounds the batch (issue #30);
        // without batched verify the sequential path drafts exactly 1 per step.
        int maxDraftN = (mtpFwd is not null && mtpFwd.SupportsBatchVerify)
            ? Math.Max(1, mtpFwd.MaxBatchVerifyTokens - 1)
            : 1;

        switch (sp.SpecType)
        {
            case SpecType.None:
                return false;
            case SpecType.Mtp:
                if (envDisabled) { rejectReason = "--spec-type mtp conflicts with SHARPI_DISABLE_MTP=1."; return false; }
                if (mtpFwd is null || !mtpFwd.HasMtpHead) { rejectReason = "--spec-type mtp requires a model with an MTP head (nextn tensors)."; return false; }
                if (sp.Temperature > 0f) { rejectReason = "--spec-type mtp requires greedy sampling (--temp 0)."; return false; }
                if (!noThinking) { rejectReason = "--spec-type mtp requires --no-thinking (chat template must render with enable_thinking=false)."; return false; }
                WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return true;
            default: // Auto
                if (eligible)
                    WarnIfDraftNClamped(sp.SpecDraftNMax, maxDraftN);
                return eligible;
        }
    }

    /// <summary>
    /// A draft chain deeper than the snapshot ring's capacity is CLAMPED, not rejected
    /// (rejecting would disable MTP entirely and run SLOWER — the silent-baseline trap
    /// the old SpecDraftNMax&gt;1 throw existed to prevent). Warn so the user knows the
    /// effective depth and the knob that raises it; MtpDecoder clamps per step.
    /// </summary>
    private static void WarnIfDraftNClamped(int requested, int maxDraftN)
    {
        if (requested > maxDraftN)
            AnsiConsole.MarkupLine(
                $"[yellow]Note:[/] --spec-draft-n-max={requested} exceeds the snapshot-ring capacity; " +
                $"running {maxDraftN} draft(s)/step. Set SHARPI_MTP_BATCH_MAX={requested + 1} to go deeper " +
                "(each ring slot costs ~150 MiB VRAM on 27B-class models).");
    }

    // MTP self-speculative decode path. Reuses the same UTF-8 streaming + EmitToken
    // logic as the baseline DecodeLoop but drives token emission via MtpDecoder.
    // Requires --no-thinking, so no thinking-mode bookkeeping here.
    private static (int generated, int totalDecoded, float acceptanceRate, long accepted, long emitted) DecodeLoopMtp(
        IForwardPass mtpFwd, IReadOnlyList<int> promptTokens, ReadOnlySpan<float> initialLogits,
        GgufTokenizer tok, SamplingParams sp, bool hideThinking, bool verbosePromptLogging = false)
    {
        var mtpDec = new MtpDecoder(mtpFwd);
        mtpDec.Initialize(promptTokens.Count, initialLogits);
        // Populate the MTP KV cache for the full prompt. Cost: ~1.6%/token; only paid
        // on the MTP-enabled run.
        mtpFwd.PrefillMtp(promptTokens, 0);

        var streamDec = new Utf8StreamDecoder();
        bool inThinking = false;
        int generated = 0;
        int totalDecoded = 0;

        // Materialize stop ids once (MtpDecoder takes ReadOnlySpan<int>).
        int[] stopIds = sp.StopTokenIds ?? [];

        mtpDec.Decode(sp.MaxNewTokens, stopIds.AsSpan(), next =>
        {
            if (verbosePromptLogging)
                Console.Error.WriteLine($"[DBG] tok={totalDecoded} next={next}('{tok.Decode([next])}')");
            totalDecoded++;
            if (EmitToken(next, tok, streamDec, ref inThinking, hideThinking)) generated++;
        }, pMin: sp.SpecDraftPMin, draftN: MtpDecoder.ResolveDraftN(sp.SpecDraftNMax),
           ct: CancellationToken.None);

        if (Environment.GetEnvironmentVariable("SHARPI_TRACE_MTP") == "1" && mtpDec.TotalDraftsEmitted > 0)
            Console.Error.WriteLine(
                $"[mtp] phase ms: draft={mtpDec.DraftMs:F0} verify={mtpDec.VerifyMs:F0} commit={mtpDec.CommitMs:F0}");

        // Flush the UTF-8 decoder tail, applying the same hide-thinking gate as DecodeLoop.
        var tail = streamDec.Flush();
        if (!(hideThinking && inThinking))
            Console.Write(tail);
        if (inThinking) Console.Write("\x1b[0m");

        return (generated, totalDecoded, mtpDec.AcceptanceRate, mtpDec.TotalDraftsAccepted, mtpDec.TotalDraftsEmitted);
    }

    private static int RunInteractive(Settings s,
        Func<int, int, ReadOnlySpan<float>> forward,
        Func<IReadOnlyList<int>, ReadOnlySpan<float>> prefill,
        Action resetCache,
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd)
    {
        // mtpFwd reserved for interactive MTP wiring (follow-up to #32). Today the
        // interactive loop stays on the baseline decode path; the bench surface and
        // single-prompt runs (RunSinglePrompt above) are what exercise MTP.
        _ = mtpFwd;
        AnsiConsole.MarkupLine("[green]Interactive chat.[/] Type a message, or [yellow]/exit[/] to quit.\n");

        while (true)
        {
            AnsiConsole.Markup("[bold]> [/]");
            var input = Console.ReadLine();
            if (input is null or "/exit" or "/quit") break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input, s.SystemPrompt, enableThinking: !s_noThinking);
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
    // Effective "thinking off" state: --no-thinking OR a model whose recommended config
    // disables reasoning (Gemma 4 E4B-it is not a reasoning model). Set during model load.
    private static bool s_noThinking;
    private static int s_thinkTokenId = -1;    // <think> token for any model using the <think>/</think> special-token convention
    private static int s_endThinkTokenId = -1; // </think> token for any model using the <think>/</think> special-token convention
    private static JinjaChatTemplate? s_jinja;  // parsed from GGUF tokenizer.chat_template

    /// <summary>
    /// Builds the stop token ID list. Delegates to <see cref="GgufTokenizer.EogTokenIds"/> —
    /// the single source of truth for end-of-generation tokens (EOS plus the end-of-turn
    /// variants used by Llama 3/4, Mistral, Phi, Gemma, etc.) — so the CLI and server stop on
    /// exactly the same set. Notably this is what lets the CLI halt on Gemma 4's <c>&lt;eos&gt;</c>
    /// (id 1, distinct from its configured EOS <c>&lt;turn|&gt;</c> at id 106) instead of decoding
    /// it as literal text.
    /// </summary>
    private static IReadOnlyList<int> BuildStopTokenIds(GgufTokenizer tokenizer) => tokenizer.EogTokenIds;

    // Accept llama.cpp's "draft-mtp" alongside the shorter "mtp" so existing command
    // lines copy-paste over. Unknown values fall back to auto with a console warning.
    private static SpecType ParseSpecType(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return SpecType.Auto;
        return value.Trim().ToLowerInvariant() switch
        {
            "auto" or "" => SpecType.Auto,
            "none" or "off" or "disabled" => SpecType.None,
            "mtp" or "draft-mtp" => SpecType.Mtp,
            _ => WarnUnknownSpecType(value),
        };

        static SpecType WarnUnknownSpecType(string v)
        {
            AnsiConsole.MarkupLine($"[yellow]Warning:[/] Unknown --spec-type [yellow]{Markup.Escape(v)}[/]; expected auto|none|mtp. Falling back to auto.");
            return SpecType.Auto;
        }
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
