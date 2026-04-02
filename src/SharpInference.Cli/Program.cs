using Spectre.Console.Cli;
using SharpInference.Cli;

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("sharpi-cli");
    config.SetApplicationVersion("0.1.0");
});

return app.Run(args);
