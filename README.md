# SharpInference

A high-performance LLM inference engine written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders).
Includes an OpenAI- and Anthropic-compatible API server.

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

# Start API server
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release
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
        --draft-model       Draft model path for speculative decoding (requires --temp 0)
        --spec-lookahead    Draft tokens per speculative step (default: 4)
        --min-batch-blas    Min batch size for BLAS GEMM (default: 16)
```

## API Server

Starts an HTTP server compatible with OpenAI and Anthropic clients:

```bash
# Set model path and start
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release
# Defaults to http://localhost:5000

# OpenAI — chat completions (streaming)
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Anthropic — messages (non-streaming)
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'

# List loaded model
curl http://localhost:5000/v1/models

# Health check
curl http://localhost:5000/health

# Prometheus metrics
curl http://localhost:5000/metrics
```

### Server environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARPI_MODEL` | `model.gguf` | Path to GGUF model file |
| `SHARPI_N_GPU_LAYERS` | `0` | GPU layers (0 = CPU only) |
| `SHARPI_MIN_BATCH_BLAS` | `16` | BLAS GEMM threshold for batched MatMul |

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
| SharpInference.Engine | classlib | Forward pass (CPU + GPU), KV cache, sampling, speculative decoding |
| SharpInference.Cli | console | CLI tool (`sharpi-cli`) with NativeAOT support |
| SharpInference.TurboQuant | classlib | KV-cache compression (3-bit Lloyd-Max codebooks) |
| SharpInference.Pipeline | classlib | Memory hierarchy, SLRU expert cache, async prefetcher |
| SharpInference.Server | web | OpenAI + Anthropic API server with NativeAOT support |

## Build & Test

```bash
dotnet build              # Debug build
dotnet build -c Release   # Release build (enables IlcOptimizationPreference=Speed)
dotnet test               # Run all tests (132 tests across 5 projects)

# Publish NativeAOT binary
dotnet publish src/SharpInference.Cli -c Release -r win-x64
dotnet publish src/SharpInference.Server -c Release -r win-x64

# Run all benchmarks (requires every benchmark model to be present)
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'
```

## Architecture

See [docs/SharpInference-Design.md](docs/SharpInference-Design.md).

