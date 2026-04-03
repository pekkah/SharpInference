# SharpInference

A high-performance LLM inference engine written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders).

## Quick Start

```bash
# Build
dotnet build -c Release

# Run inference (CPU)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" --temp 0

# Run inference (GPU — all layers on VRAM)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" --temp 0 -g -1

# Interactive chat
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

## CLI Usage (llama.cpp-compatible flags)

```
sharpi-cli [OPTIONS]

OPTIONS:
    -m, --model            Path to GGUF model file
    -p, --prompt           Input prompt (omit for interactive chat)
    -n, --n-predict        Tokens to generate (default: 512)
        --temp             Temperature (0 = greedy, default: 0.7)
        --top-k            Top-k sampling (default: 40)
        --top-p            Top-p nucleus sampling (default: 0.95)
        --min-p            Min-p sampling (default: 0.05)
    -s, --seed             RNG seed (-1 = random)
    -g, --n-gpu-layers     GPU layers (0 = CPU, -1 = all on GPU)
        --single-turn      Generate one response and exit
        --system-prompt     System prompt
        --no-display-prompt Don't echo the prompt
        --verbose-prompt    Print token IDs before generating
```

## Performance

Benchmarked on AMD Zen 4 (12c/24t) + RTX 4070 Ti, SmolLM2-1.7B Q4_K_M:

| Backend | Decode (t/s) | Notes |
|---------|-------------|-------|
| CPU (AVX2 SIMD) | 48.6 | Fused dequant-matvec, multi-threaded |
| GPU (Vulkan) | 87.4 | Compute shaders, VRAM-resident weights |
| llama.cpp (reference) | 45.1 | CPU decode on same hardware |

## Projects

| Project | Type | Description |
|---|---|---|
| SharpInference.Core | classlib | GGUF parser, tokenizer, tensor types, model graph |
| SharpInference.Cpu | classlib | CPU backend: AVX2/AVX-512 SIMD, fused dequant-matvec |
| SharpInference.Vulkan | classlib | GPU backend: Vulkan compute shaders via Vortice.Vulkan |
| SharpInference.Engine | classlib | Forward pass (CPU + GPU), KV cache, sampling |
| SharpInference.Cli | console | CLI tool (`sharpi-cli`) with NativeAOT support |
| SharpInference.TurboQuant | classlib | KV-cache compression (Phase 3) |
| SharpInference.Pipeline | classlib | Memory hierarchy, tier placement (Phase 4) |
| SharpInference.Server | web | API server (Phase 7) |

## Build & Test

```bash
dotnet build              # Debug build
dotnet build -c Release   # Release build (enables IlcOptimizationPreference=Speed)
dotnet test               # Run all tests (85 tests)

# Publish NativeAOT binary
dotnet publish src/SharpInference.Cli -c Release -r win-x64

# Run all benchmarks (requires every benchmark model to be present)
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'

# Run one model/backend suite (only that model is needed)
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*SmolLM2CpuBenchmarks*'
```

## Setup

```bash
# Download OpenBLAS (optional, for batched prefill GEMM)
powershell scripts/setup-openblas.ps1

# Download a model
# Place GGUF files in the models/ directory
```

## Architecture

See [docs/SharpInference-Design.md](docs/SharpInference-Design.md).
