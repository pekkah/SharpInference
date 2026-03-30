using SharpInference.Cli;

if (args is ["bench", ..])
    await BenchRunner.RunAsync(args[1..]);
else
    await ChatRepl.RunAsync(args);
