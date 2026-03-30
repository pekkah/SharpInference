using SharpInference.Engine;

namespace SharpInference.Cli;

/// <summary>Interactive chat REPL with readline-style editing.</summary>
public static class ChatRepl
{
    public static async Task RunAsync(string[] args)
    {
        // TODO: parse args (--model, --ctx-len, --temp, etc.)
        // TODO: load model and create InferenceEngine
        // TODO: run prompt/response loop, streaming tokens to Console

        Console.WriteLine("SharpInference chat REPL - not yet implemented.");
        await Task.CompletedTask;
    }
}
