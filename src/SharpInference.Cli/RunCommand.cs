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
        [Description("Number of draft tokens per speculative step with --draft-model (default: 4)")]
        [DefaultValue(4)]
        public int SpecLookahead { get; init; }

        [CommandOption("--spec-type")]
        [Description("Speculative decoding type: auto (default; enables MTP when supported), none, mtp (alias: draft-mtp). Mirrors llama.cpp.")]
        [DefaultValue("auto")]
        public string SpecTypeStr { get; init; } = "auto";

        [CommandOption("--spec-draft-n-max")]
        [Description("Max draft tokens per MTP step (default: 1). Currently only N=1 is supported on the MTP path; issue #30 will lift this. Mirrors llama.cpp.")]
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
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        if (settings.MinBatchBlas > 0)
            SimdKernels.MinBatchForBlas = settings.MinBatchBlas;

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

        // Greedy on a reasoning model tends to "wait, but actually" itself into infinite
        // loops; --no-thinking sidesteps the issue since the model won't reason at all.
        if (s_thinkTokenId > 0 && settings.Temperature == 0f && !settings.NoThinking)
        {
            AnsiConsole.MarkupLine("[yellow]Warning:[/] Greedy decoding (--temp 0) on a reasoning model often produces");
            AnsiConsole.MarkupLine("infinite \"wait, but actually\" loops. Consider [yellow]--temp 0.6 --top-p 0.95 --top-k 20[/].");
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
        if (hp.IsHybridSsm && settings.DraftModelPath is not null)
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
        if (hp.IsHybridSsm && settings.NGpuLayers == 0)
        {
            hybridFwd = new HybridGdnForwardPass(model, cpuBackend, hp);
            if (hybridFwd.HasMtpHead) mtpFwd = hybridFwd;
        }
        else if (!hp.IsHybridSsm)
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
                if (nGpuLayers == -1)
                {
                    var hwProfile = HardwareProfile.Detect(cuda);
                    AnsiConsole.MarkupLine($"[dim]Hardware: {hwProfile.Summary()}[/]");
                    var placement = TierPlanner.Plan(model, hp, hwProfile, settings.TurboQuant,
                        requestedCtxSize: ctxSize);
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
            SpecType = ParseSpecType(settings.SpecTypeStr),
            SpecDraftNMax = settings.SpecDraftNMax,
            SpecDraftNMin = settings.SpecDraftNMin,
            SpecDraftPMin = settings.SpecDraftPMin,
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
        GgufTokenizer tok, SamplingParams sp, Random rng,
        IForwardPass? mtpFwd)
    {
        var prompt = FormatPrompt(s.Prompt!, s.SystemPrompt, enableThinking: !s.NoThinking);
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
        bool useMtp = ResolveCliMtp(mtpFwd, sp, s.NoThinking, out string? mtpReject);
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

        int maxBatchN = (mtpFwd is not null && mtpFwd.SupportsBatchVerify) ? 2 : 1;

        switch (sp.SpecType)
        {
            case SpecType.None:
                return false;
            case SpecType.Mtp:
                if (envDisabled) { rejectReason = "--spec-type mtp conflicts with SHARPI_DISABLE_MTP=1."; return false; }
                if (mtpFwd is null || !mtpFwd.HasMtpHead) { rejectReason = "--spec-type mtp requires a model with an MTP head (nextn tensors)."; return false; }
                if (sp.Temperature > 0f) { rejectReason = "--spec-type mtp requires greedy sampling (--temp 0)."; return false; }
                if (!noThinking) { rejectReason = "--spec-type mtp requires --no-thinking (chat template must render with enable_thinking=false)."; return false; }
                if (sp.SpecDraftNMax > maxBatchN)
                {
                    rejectReason = $"--spec-draft-n-max={sp.SpecDraftNMax} exceeds the supported N=2 batched verify ceiling (issue #30); N>2 is still TODO.";
                    return false;
                }
                return true;
            default: // Auto
                if (eligible && sp.SpecDraftNMax > maxBatchN)
                {
                    rejectReason = $"--spec-draft-n-max={sp.SpecDraftNMax} exceeds the supported N=2 batched verify ceiling (issue #30); N>2 is still TODO.";
                    return false;
                }
                return eligible;
        }
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
        }, pMin: sp.SpecDraftPMin, ct: CancellationToken.None);

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
