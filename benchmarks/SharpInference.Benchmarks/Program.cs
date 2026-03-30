using BenchmarkDotNet.Running;
using SharpInference.Benchmarks;

BenchmarkSwitcher.FromAssembly(typeof(InferenceBenchmarks).Assembly).Run(args);
