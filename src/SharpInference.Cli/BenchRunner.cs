using SharpInference.Engine;

namespace SharpInference.Cli;

/// <summary>Standalone benchmark runner (prompt/token throughput, TTFT, memory).</summary>
public static class BenchRunner
{
    public static async Task RunAsync(string[] args)
    {
        // TODO: run warmup pass, measure prefill/decode throughput, report results
        Console.WriteLine("SharpInference bench runner - not yet implemented.");
        await Task.CompletedTask;
    }
}
