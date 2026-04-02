# SharpInference

A high-performance LLM inference engine written in C# 14 / .NET 10.

## Projects

| Project | Type | Description |
|---|---|---|
| SharpInference.Core | classlib | GGUF parser, tokenizer, tensor types, model graph, IComputeBackend |
| SharpInference.Cpu | classlib | CPU backend with SIMD (AVX2 / AVX-512) |
| SharpInference.Vulkan | classlib | Vulkan compute backend via Vortice.Vulkan |
| SharpInference.TurboQuant | classlib | KV-cache compression (Lloyd-Max scalar quantisation) |
| SharpInference.Pipeline | classlib | Memory hierarchy, tier placement, expert cache, prefetching |
| SharpInference.Engine | classlib | Inference engine, forward pass, speculative decoding, sampling |
| SharpInference.Server | web | ASP.NET Core Minimal API � OpenAI + Anthropic compatible endpoints |
| SharpInference.Cli | console | Interactive chat REPL + benchmark runner |
| SharpInference.Benchmarks | console | BenchmarkDotNet harness |

## Build

    dotnet restore
    dotnet build

## Test

    dotnet test

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
