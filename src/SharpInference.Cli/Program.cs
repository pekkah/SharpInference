using Spectre.Console.Cli;
using SharpInference.Cli;

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("sharpi-cli");
    config.SetApplicationVersion("0.1.0");
    config.AddCommand<ListMetadataCommand>("list-metadata")
        .WithDescription("Print all GGUF metadata key/value pairs from a model file");
});

return app.Run(args);
