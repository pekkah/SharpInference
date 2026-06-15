using SharpInference.Server;

var builder = WebApplication.CreateSlimBuilder(args);

// Per-developer overrides (not committed) layered on top of appsettings.json. Anything you
// don't want in git — local model paths, credentials, port pinning — goes here. The file is
// listed in .gitignore so it never accidentally ships.
builder.Configuration.AddJsonFile("appsettings.Local.json", optional: true, reloadOnChange: true);

// Bind SharpInferenceServerOptions from the "SharpInference" config section first,
// then layer environment-variable overrides for backward compatibility with the original
// SHARPI_MODEL / SHARPI_MAX_BATCH knobs. Inline configure runs last → wins.
builder.Services.AddSharpInference(builder.Configuration, opts =>
{
    var envModel = Environment.GetEnvironmentVariable("SHARPI_MODEL");
    if (!string.IsNullOrWhiteSpace(envModel))
        opts.ModelPath = envModel;

    // Multimodal projector for image input (issue #253). Mirrors the CLI's --mmproj.
    var envMmproj = Environment.GetEnvironmentVariable("SHARPI_MMPROJ");
    if (!string.IsNullOrWhiteSpace(envMmproj))
        opts.MmprojPath = envMmproj;

    if (int.TryParse(Environment.GetEnvironmentVariable("SHARPI_MAX_BATCH"), out int maxBatch) && maxBatch > 0)
        opts.MaxBatchSize = maxBatch;

    // Continuous-batching scheduling knobs (issue #183): prefill chunk size (Gap 1)
    // and KV admission budget in MiB (Gap 3). Same precedence as SHARPI_MAX_BATCH.
    if (int.TryParse(Environment.GetEnvironmentVariable("SHARPI_PREFILL_CHUNK"), out int prefillChunk) && prefillChunk >= 0)
        opts.PrefillChunkTokens = prefillChunk;

    if (long.TryParse(Environment.GetEnvironmentVariable("SHARPI_KV_BUDGET_MB"), out long kvBudgetMb) && kvBudgetMb != 0)
        opts.KvBudgetMb = kvBudgetMb;

    // Dequant-once BLAS weight-cache budget in MiB (issue #189). null/unset = auto.
    if (long.TryParse(Environment.GetEnvironmentVariable("SHARPI_PREFILL_DEQUANT_MB"), out long dequantMb))
        opts.PrefillDequantCacheMb = dequantMb;

    // SHARPI_BACKEND ∈ {auto, cpu, cuda, vulkan} — case-insensitive. Lets a
    // smoke test or ad-hoc run override the appsettings.Local.json backend
    // without editing the file (matches the SHARPI_MODEL pattern above).
    var envBackend = Environment.GetEnvironmentVariable("SHARPI_BACKEND");
    if (!string.IsNullOrWhiteSpace(envBackend)
        && Enum.TryParse<SharpInference.Server.ServerBackend>(envBackend, ignoreCase: true, out var backend))
    {
        opts.Backend = backend;
    }

    if (int.TryParse(Environment.GetEnvironmentVariable("SHARPI_N_GPU_LAYERS"), out int nGpuLayers))
        opts.NGpuLayers = nGpuLayers;

    // SHARPI_KV_DTYPE ∈ {fp32, bf16, q8_0} — CUDA dense KV-cache element type (#179).
    // Mirrors the SHARPI_MODEL/SHARPI_BACKEND override pattern; the loader forwards it
    // back to the env var the forward pass reads. Validated at model load.
    var envKvType = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
    if (!string.IsNullOrWhiteSpace(envKvType))
        opts.KvType = envKvType;

    // SHARPI_NO_THINKING ∈ {1, true} globally disables reasoning (server-side --no-thinking),
    // for agentic clients that never send the per-request opt-out.
    var envNoThink = Environment.GetEnvironmentVariable("SHARPI_NO_THINKING");
    if (!string.IsNullOrWhiteSpace(envNoThink)
        && (envNoThink == "1" || envNoThink.Equals("true", StringComparison.OrdinalIgnoreCase)))
    {
        opts.DisableThinking = true;
    }
});

var app = builder.Build();

app.MapSharpInference();

app.Run();

// Required for WebApplicationFactory<Program> in integration tests.
public partial class Program { }
