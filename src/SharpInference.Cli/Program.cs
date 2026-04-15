using Spectre.Console.Cli;
using SharpInference.Cli;

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("sharpi-cli");
    config.SetApplicationVersion("0.1.0");
    config.AddCommand<ListMetadataCommand>("list-metadata")
        .WithDescription("Print all GGUF metadata key/value pairs from a model file");
    config.AddCommand<ImageCommand>("image")
        .WithDescription("Generate an image from a text prompt using a native FLUX or Z-Image-Turbo diffusion pipeline (VAE + CLIP-L + T5-XXL + DiT GGUF). See 'sharpi-cli image --help' for required model paths.");
});

return app.Run(args);
