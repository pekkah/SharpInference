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
    /// <summary>
    /// Globally disable reasoning for every request (server-side <c>--no-thinking</c> /
    /// <c>SHARPI_NO_THINKING</c>). For agentic clients that never send the per-request opt-out.
    /// </summary>
    public bool DisableThinking { get; set; }

    /// <summary>
    /// Enable schema/grammar-constrained decoding for tool-call arguments (issue #374). When on and
    /// a tool-active request is served by a family with constraint support (Gemma 4 today), the
    /// sampler is restricted to tokens that satisfy the supplied JSON Schema in the model's native
    /// call syntax — required keys can't be dropped, only declared keys/enum values appear, value
    /// shapes match the declared type. Default off → byte-identical to unconstrained decoding. Also
    /// turned on by the <c>SHARPI_TOOL_GRAMMAR=1</c> environment variable.
    /// </summary>
    public bool ToolGrammar { get; set; }

    // ── Model loading ────────────────────────────────────────────────────────

    /// <summary>
    /// Path to the GGUF model file. Required unless <see cref="EngineFactory"/> is supplied.
    /// Relative paths resolve against the current directory, the entry-assembly directory,
    /// and a handful of parent directories.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Optional path to a multimodal projector GGUF (mmproj-*.gguf) enabling image input
    /// (issue #253). Only Gemma 4 <c>gemma4uv</c> projectors are supported today, and only on a
    /// backend whose forward pass accepts precomputed-embedding input (CPU, i.e.
    /// <see cref="NGpuLayers"/>=0, or full CUDA offload, <see cref="NGpuLayers"/>=-1, of a model
    /// that fits VRAM). Set via this property, the <c>SHARPI_MMPROJ</c> environment variable, or
    /// the <c>SharpInference:MmprojPath</c> configuration key. When unset, image content blocks
    /// are rejected with a clear error.
    /// </summary>
    public string? MmprojPath { get; set; }

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
    /// KV-cache element type for the CUDA dense path. Mirrors the CLI's <c>--kv-type</c> /
    /// <c>SHARPI_KV_DTYPE</c>: <c>"fp32"</c> (default), <c>"bf16"</c> (half the KV VRAM →
    /// ~2× context), or <c>"q8_0"</c> (block-quantized, ~quarter → ~4× context; restores
    /// full decode speed at very long context where bf16 spills the tied embed table). The
    /// loader forwards this to <c>SHARPI_KV_DTYPE</c> before the forward pass is built; the
    /// value is validated then (an unrecognized string fails model load). <c>null</c>/empty
    /// leaves any externally-set <c>SHARPI_KV_DTYPE</c> untouched. Has effect only on the
    /// CUDA dense forward pass; ignored by CPU/Vulkan and rejected with TurboQuant or an
    /// explicit SnapKV budget (issue #179).
    /// </summary>
    public string? KvType { get; set; }

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

    /// <summary>
    /// Maximum number of generation requests allowed in flight at once before the server
    /// fast-rejects with HTTP 429 (issue #109). <c>null</c> (default) keeps the legacy
    /// behaviour: the single-user <see cref="InferenceEngine"/> serializes overlapping
    /// requests on an internal gate, so a concurrent request silently blocks for the full
    /// duration of the in-flight one — which an agentic client (e.g. Claude Code firing
    /// near-simultaneous requests) experiences as an unbounded hang. Set to <c>1</c> to make
    /// the single-user limit explicit: overlapping requests get an immediate 429 pointing at
    /// <c>SHARPI_MAX_BATCH</c> instead of queuing. Leave unset (or use a value matching
    /// <see cref="MaxBatchSize"/>) when running <see cref="ContinuousBatchingEngine"/>, which
    /// serves concurrency itself. Mirrors <c>SHARPI_MAX_CONCURRENT</c>.
    /// </summary>
    public int? MaxConcurrentRequests { get; set; }

    /// <summary>
    /// Prompt tokens prefilled per batcher iteration under continuous batching
    /// (issue #183 Gap 1). Active sequences advance one decode step between chunks, so
    /// a long inbound prompt no longer freezes every in-flight generation. <c>0</c>
    /// disables chunking (each prompt prefills in one blocking call). <c>-1</c> = auto
    /// (issue #189): <c>64</c> when <see cref="PrefillDequantCacheMb"/> makes small chunks
    /// free, else <c>256</c>. Only consulted when <see cref="MaxBatchSize"/> &gt; 1. Mirrors
    /// <c>SHARPI_PREFILL_CHUNK</c>.
    /// </summary>
    public int PrefillChunkTokens { get; set; } = -1;

    /// <summary>
    /// Dequant-once BLAS weight-cache budget in MiB (issue #189). The CPU batched-prefill
    /// path re-dequantizes each projection weight to F32 on every call, so small prefill
    /// chunks re-pay the full dequant repeatedly; caching the F32 dequant per weight removes
    /// that cost (bit-identical). <c>null</c> = auto (cache the model iff a full F32 copy fits
    /// a quarter of available RAM), <c>0</c> = off, negative = unlimited, positive = explicit
    /// MiB. Only meaningful for the CPU forward pass. Mirrors <c>SHARPI_PREFILL_DEQUANT_MB</c>.
    /// </summary>
    public long? PrefillDequantCacheMb { get; set; }

    /// <summary>
    /// KV-cache memory budget in MiB gating request admission under continuous batching
    /// (issue #183 Gap 3). Each admitted sequence reserves
    /// <c>promptTokens + max_tokens</c> worth of KV; when the next request would exceed
    /// the budget it waits in the queue instead of risking memory exhaustion.
    /// <c>0</c> = auto (half of available system RAM), negative = unlimited, positive =
    /// explicit MiB. Only consulted when <see cref="MaxBatchSize"/> &gt; 1. Mirrors
    /// <c>SHARPI_KV_BUDGET_MB</c>.
    /// </summary>
    public long KvBudgetMb { get; set; } = 0;

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

    /// <summary>
    /// Force all routed MoE experts onto the CPU side — the engine's all-or-nothing
    /// <c>SHARPI_CPU_MOE</c> override, read by <see cref="CudaHybridGdnForwardPass"/> and
    /// <see cref="CudaHybridForwardPass"/> at construction. Server analogue of the CLI's
    /// <c>--cpu-moe</c> (issue #80 / #93). <c>true</c> → <c>SHARPI_CPU_MOE=1</c> (all routed
    /// experts on CPU; e.g. the <c>Qwen3.6-35B-A3B-MTP</c> hybrid only reaches its headline
    /// decode rate this way). <c>false</c> → <c>SHARPI_CPU_MOE=0</c> (force the on-GPU SLRU
    /// expert cache). <c>null</c> (default) writes nothing, preserving the engine's VRAM-fit
    /// auto-select and any <c>SHARPI_CPU_MOE</c> already set in the process environment. Has
    /// effect only on the CUDA hybrid MoE paths; ignored by CPU/Vulkan and non-MoE models.
    /// </summary>
    public bool? CpuMoe { get; set; }

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
/// <param name="ToolBoundaryStopTokenIds">
/// Token IDs that mark the end of the tool-call section for this model family (Gemma 4:
/// <c>&lt;|tool_response&gt;</c>), resolved from the adapter's
/// <see cref="SharpInference.Core.IToolCallAdapter.ToolBoundaryStopMarkers"/> against the vocab.
/// The chat endpoints add these to the stop set on tool-active requests so the model halts the
/// instant it finishes its tool calls (issue #304). Null/empty when the family needs none.
/// </param>
/// <param name="Grammar">
/// Vocabulary view used to build grammar-constrained tool-call decoders (issue #374), or null when
/// the family has no constraint support. The built-in GGUF loader always supplies one; a custom
/// <see cref="SharpInferenceServerOptions.EngineFactory"/> may leave it null (tool-grammar then
/// unavailable for that engine).
/// </param>
public sealed record LoadedEngine(
    IInferenceEngine Engine,
    string Architecture,
    SharpInference.Core.JinjaChatTemplate? ChatTemplate,
    IReadOnlyList<int>? ToolBoundaryStopTokenIds = null,
    SharpInference.Core.Grammar.GrammarVocabulary? Grammar = null);
