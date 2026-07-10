using System.Text;
using Spectre.Console.Cli;
using SharpInference.Cli;

// Force UTF-8 for stdin/stdout. On Windows the console defaults to the OEM
// code page, which mangles multi-byte UTF-8 output (CJK, emoji, smart quotes)
// into '?' or replacement glyphs.
Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("sharpi-cli");
    config.SetApplicationVersion("0.1.0");
    config.AddCommand<ListMetadataCommand>("list-metadata")
        .WithDescription("Print all GGUF metadata key/value pairs from a model file");
    config.AddCommand<ListTensorsCommand>("list-tensors")
        .WithDescription("Print the tensor index (name, dtype, shape, bytes) from a model file");
    config.AddCommand<PerplexityCommand>("perplexity")
        .WithDescription("Teacher-forced perplexity over a text file (llama.cpp llama-perplexity analogue; CPU only). Reports mean NLL, perplexity, and position-bucket NLLs — the TurboQuant/KVarN accuracy gate (issue #180).");
    config.AddCommand<ImageCommand>("image")
        .WithDescription("Generate an image from a text prompt using a native FLUX or Z-Image-Turbo diffusion pipeline (VAE + CLIP-L + T5-XXL + DiT GGUF). See 'sharpi-cli image --help' for required model paths.");
});

return app.Run(args);
