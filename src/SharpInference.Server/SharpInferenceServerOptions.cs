using SharpInference.Engine;

namespace SharpInference.Server;

/// <summary>
/// Configuration for <see cref="ServiceCollectionExtensions.AddSharpInference"/>. Mirrors the
/// SharpInference CLI surface (<c>sharpi-cli run</c>) so an operator can express any tuning
/// the CLI exposes in <c>appsettings.json</c> / <c>appsettings.Local.json</c> instead of
/// editing <c>Program.cs</c>. Fields are grouped by concern; defaults are the CLI's defaults.
///
/// <para>CLI-only flags that have no server analogue are intentionally omitted:</para>
/// <list type="bullet">
///   <item><c>--prompt</c>, <c>--system-prompt</c> — supplied per-request via the chat API</item>
///   <item><c>--single-turn</c> — server is always multi-turn</item>
///   <item><c>--no-display-prompt</c>, <c>--verbose-prompt</c>, <c>--hide-thinking</c> — terminal UX</item>
///   <item><c>--draft-model</c> — uses a CPU-only side-engine that the singleton
///     <see cref="IInferenceEngine"/> abstraction can't host; MTP speculative decoding
///     (<see cref="ServerSpecType.Mtp"/>) is the supported equivalent.</item>
/// </list>
/// </summary>
public sealed class SharpInferenceServerOptions
{
    // ── Model loading ────────────────────────────────────────────────────────

    /// <summary>
    /// Path to the GGUF model file. Required unless <see cref="EngineFactory"/> is supplied.
    /// Relative paths resolve against the current directory, the entry-assembly directory,
    /// and a handful of parent directories.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Architecture hint used by <see cref="ChatTemplateRenderer"/> as a fallback when the
    /// model's GGUF metadata is missing <c>general.architecture</c> and no Jinja template
    /// is bundled. Defaults to <c>"qwen2"</c> (ChatML).
    /// </summary>
    public string Architecture { get; set; } = "qwen2";

    /// <summary>
    /// Optional escape hatch: build the engine programmatically instead of loading a GGUF
    /// file. When set, every other field on this options object that affects load behaviour
    /// (<see cref="Backend"/>, <see cref="NGpuLayers"/>, <see cref="TurboQuant"/>, ...) is
    /// the factory's responsibility — the built-in loader is bypassed entirely.
    /// </summary>
    public Func<IServiceProvider, LoadedEngine>? EngineFactory { get; set; }

    // ── Backend / hardware ───────────────────────────────────────────────────

    /// <summary>
    /// GPU backend selection. Mirrors the CLI's <c>--backend</c>. <c>Auto</c> picks CUDA when
    /// available, falls through to Vulkan, then CPU. Only consulted when
    /// <see cref="NGpuLayers"/> is non-zero.
    /// </summary>
    public ServerBackend Backend { get; set; } = ServerBackend.Auto;

    /// <summary>
    /// Number of model layers to offload to the GPU. Mirrors the CLI's <c>--n-gpu-layers</c>
    /// (<c>-g</c>): <c>0</c> = CPU only, <c>-1</c> = let TierPlanner size the split from
    /// available VRAM, <c>N</c> = explicit. Default <c>0</c>.
    /// </summary>
    public int NGpuLayers { get; set; } = 0;

    /// <summary>
    /// Context size / max sequence length. <c>0</c> = use the model's GGUF default.
    /// Mirrors <c>--ctx-size</c>.
    /// </summary>
    public int ContextSize { get; set; } = 0;

    /// <summary>
    /// Enable TurboQuant 3-bit KV-cache compression. Mirrors <c>--tq</c>. Requires head
    /// dimension ∈ {128, 256}; the loader falls back to non-TQ otherwise.
    /// </summary>
    public bool TurboQuant { get; set; } = false;

    /// <summary>
    /// Minimum batch size before <see cref="SharpInference.Cpu.SimdKernels.MinBatchForBlas"/>
    /// promotes the inner loop to OpenBLAS SGEMM. Mirrors <c>--min-batch-blas</c> /
    /// <c>SHARPI_MIN_BATCH_BLAS</c>. <c>0</c> = leave the engine default.
    /// </summary>
    public int MinBatchBlas { get; set; } = 0;

    // ── Concurrency ──────────────────────────────────────────────────────────

    /// <summary>
    /// Maximum concurrent decode sequences. Values &gt; 1 select
    /// <see cref="ContinuousBatchingEngine"/>; ≤ 1 selects <see cref="InferenceEngine"/>.
    /// </summary>
    public int MaxBatchSize { get; set; } = 1;

    // ── MoE expert-cache tuning ──────────────────────────────────────────────
    //
    // These knobs only have effect on Vulkan-hybrid MoE today; the engine reads them from
    // the SHARPI_* environment, so the loader translates the options into env vars before
    // model load. CUDA-hybrid MoE drives its own SLRU and ignores these settings.

    /// <summary>
    /// Pin the top-N hottest experts per layer after warmup. <c>null</c> = disabled
    /// (frequency-aware SLRU eviction is sufficient on its own). Mirrors
    /// <c>--moe-warmpin</c> / <c>SHARPI_MOE_WARMPIN</c>.
    /// </summary>
    public int? MoeWarmPin { get; set; }

    /// <summary>
    /// Number of expert accesses to observe before warm-pin selects the hot set. Only
    /// meaningful when <see cref="MoeWarmPin"/> is set. Mirrors <c>--moe-warmpin-after</c>.
    /// </summary>
    public long MoeWarmPinAfter { get; set; } = 0;

    /// <summary>
    /// Next-layer predictive expert prefetch on the Vulkan path. Mirrors
    /// <c>--no-moe-predict-prefetch</c> (defaulting to <c>true</c> here — set <c>false</c>
    /// to disable, equivalent to <c>SHARPI_MOE_PREDICT_PREFETCH=0</c>).
    /// </summary>
    public bool MoePredictPrefetch { get; set; } = true;

    /// <summary>
    /// Path to write GPU expert-cache (SLRU) hit-rate stats to on process exit. Mirrors
    /// <c>--expert-stats</c> / <c>SHARPI_EXPERT_STATS</c>.
    /// </summary>
    public string? ExpertStatsPath { get; set; }

    // ── Speculative decoding defaults ────────────────────────────────────────

    /// <summary>
    /// Speculative decoding mode. Mirrors <c>--spec-type</c>: <c>Auto</c> enables MTP when
    /// supported; <c>None</c> forces single-token; <c>Mtp</c> requires an MTP head.
    /// Applied as a per-request default when the request doesn't override.
    /// </summary>
    public ServerSpecType SpecType { get; set; } = ServerSpecType.Auto;

    /// <summary>Max draft tokens per speculative step. Mirrors <c>--spec-draft-n-max</c>.</summary>
    public int SpecDraftNMax { get; set; } = 0;

    /// <summary>Min draft tokens per speculative step. Mirrors <c>--spec-draft-n-min</c>.</summary>
    public int SpecDraftNMin { get; set; } = 0;

    /// <summary>
    /// Minimum draft probability for probabilistic accept under MTP verification. Mirrors
    /// <c>--spec-draft-p-min</c>. <c>1.0</c> = strict argmax-match (byte-identical to no-MTP).
    /// </summary>
    public float SpecDraftPMin { get; set; } = 1f;

    // ── Per-request sampling defaults ────────────────────────────────────────

    /// <summary>
    /// Sampling parameters applied when the inbound request omits them. The HTTP request
    /// fields (e.g. OpenAI <c>temperature</c>, Anthropic <c>top_p</c>) still take precedence
    /// — these are only the fallback when the client didn't say.
    /// </summary>
    public SamplingDefaults Sampling { get; set; } = new();
}

/// <summary>GPU backend selector. String values bind from <c>appsettings.json</c>.</summary>
public enum ServerBackend
{
    /// <summary>Prefer CUDA, fall through to Vulkan, then CPU.</summary>
    Auto = 0,
    /// <summary>Force CPU; ignore <see cref="SharpInferenceServerOptions.NGpuLayers"/>.</summary>
    Cpu = 1,
    /// <summary>Force CUDA; error if unavailable.</summary>
    Cuda = 2,
    /// <summary>Force Vulkan; error if unavailable.</summary>
    Vulkan = 3,
}

/// <summary>Speculative-decoding type. Public mirror of the engine's <c>SpecType</c>.</summary>
public enum ServerSpecType
{
    /// <summary>Engine picks: enable MTP when the model supports it and sampling is greedy.</summary>
    Auto = 0,
    /// <summary>Disable speculative decoding.</summary>
    None = 1,
    /// <summary>MTP self-speculative decoding (errors if the model has no MTP head).</summary>
    Mtp = 2,
}

/// <summary>
/// Default sampling parameters applied when the inbound HTTP request omits them.
/// Mirrors the per-request fields the CLI surfaces via <c>--temp</c> / <c>--top-k</c> / etc.
/// </summary>
public sealed class SamplingDefaults
{
    /// <summary>Temperature. <c>0</c> = greedy. Mirrors <c>--temp</c>.</summary>
    public float Temperature { get; set; } = 1f;

    /// <summary>Top-k truncation. <c>0</c> = disabled. Mirrors <c>--top-k</c>.</summary>
    public int TopK { get; set; } = 0;

    /// <summary>Top-p (nucleus) cutoff. <c>1.0</c> = disabled. Mirrors <c>--top-p</c>.</summary>
    public float TopP { get; set; } = 1f;

    /// <summary>Min-p cutoff. <c>0</c> = disabled. Mirrors <c>--min-p</c>.</summary>
    public float MinP { get; set; } = 0f;

    /// <summary>Repetition penalty. <c>1.0</c> = disabled. Mirrors <c>--rep-penalty</c>.</summary>
    public float RepetitionPenalty { get; set; } = 1f;

    /// <summary>
    /// Cap on generated tokens when the request doesn't specify <c>max_tokens</c>.
    /// Mirrors <c>--n-predict</c>.
    /// </summary>
    public int MaxNewTokens { get; set; } = 512;

    /// <summary>
    /// Maximum reasoning tokens before the engine forces <c>&lt;/think&gt;</c>. <c>0</c> =
    /// unlimited. Mirrors <c>--max-thinking-tokens</c>.
    /// </summary>
    public int MaxThinkingTokens { get; set; } = 0;

    /// <summary>
    /// Apply these defaults to a freshly-built <see cref="SamplingParams"/>. Used by the
    /// HTTP endpoints when the request omits a field. Callers pass per-request overrides
    /// after this returns.
    /// </summary>
    internal SamplingParams ToSamplingParams() => new()
    {
        Temperature       = Temperature,
        TopK              = TopK,
        TopP              = TopP,
        MinP              = MinP,
        RepetitionPenalty = RepetitionPenalty,
        MaxNewTokens      = MaxNewTokens,
        MaxThinkingTokens = MaxThinkingTokens,
    };
}

/// <summary>
/// Bundle returned by <see cref="SharpInferenceServerOptions.EngineFactory"/> and by the
/// built-in GGUF loader. Carries the engine plus the model metadata needed to render
/// chat prompts for that specific model.
/// </summary>
/// <param name="Engine">The engine instance to register as <see cref="IInferenceEngine"/>.</param>
/// <param name="Architecture">
/// Model architecture (e.g. <c>"llama"</c>, <c>"qwen2"</c>) used to pick the hardcoded
/// chat-template fallback when <paramref name="ChatTemplate"/> is null.
/// </param>
/// <param name="ChatTemplate">
/// Compiled Jinja chat template from the model's GGUF metadata, or null when the model
/// has no <c>tokenizer.chat_template</c> key.
/// </param>
public sealed record LoadedEngine(
    IInferenceEngine Engine,
    string Architecture,
    SharpInference.Core.JinjaChatTemplate? ChatTemplate);
