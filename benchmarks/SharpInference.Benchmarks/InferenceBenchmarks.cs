using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using SharpInference.Engine;

namespace SharpInference.Benchmarks;

[SimpleJob]
[MemoryDiagnoser]
[HideColumns("Error", "StdDev", "Median", "RatioSD")]
public class InferenceBenchmarks
{
    private InferenceEngine _engine = null!;

    [Params(128, 512, 2048)]
    public int PromptLength { get; set; }

    [Params(64)]
    public int NewTokens { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        // TODO: load a test model for benchmarking
    }

    [GlobalCleanup]
    public async Task Cleanup()
    {
        if (_engine is not null)
            await _engine.DisposeAsync();
    }

    [Benchmark(Description = "Prefill throughput (tokens/s)")]
    public Task Prefill()
    {
        // TODO: measure prefill (prompt processing) speed
        throw new NotImplementedException();
    }

    [Benchmark(Description = "Decode throughput (tokens/s)")]
    public Task Decode()
    {
        // TODO: measure decode (token generation) speed
        throw new NotImplementedException();
    }
}
