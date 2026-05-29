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

    if (int.TryParse(Environment.GetEnvironmentVariable("SHARPI_MAX_BATCH"), out int maxBatch) && maxBatch > 0)
        opts.MaxBatchSize = maxBatch;
});

var app = builder.Build();

app.MapSharpInference();

app.Run();

// Required for WebApplicationFactory<Program> in integration tests.
public partial class Program { }
