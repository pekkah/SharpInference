# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SharpInference is a high-performance LLM inference engine in C# 14 / .NET 10. It reads GGUF model files and runs transformer inference on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders). Targets NativeAOT for single-binary deployment.

## Build & Test Commands

```bash
dotnet build                # Debug build
dotnet build -c Release     # Release (NativeAOT opts: IlcOptimizationPreference=Speed)
dotnet test                 # Run all tests (132 tests across 5 projects)
dotnet test --filter "FullyQualifiedName~SomeTest"  # Run a single test

# Run CLI inference
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0

# GPU backend (all layers offloaded)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0 -g -1

# Start API server (OpenAI + Anthropic compatible)
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release

# NativeAOT publish
dotnet publish src/SharpInference.Cli -c Release -r win-x64
dotnet publish src/SharpInference.Server -c Release -r win-x64

# Benchmarks
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'
```

## Architecture

Four-layer stack, bottom-up:

1. **Core** (`SharpInference.Core`) — GGUF parser (memory-mapped), BPE tokenizer (`Microsoft.ML.Tokenizers`), tensor types, model graph. Everything depends on this.
2. **Compute Backends** — Two implementations of `IComputeBackend`:
   - `SharpInference.Cpu` — AVX2/AVX-512 SIMD kernels, Q4_K_M dequantization, optional OpenBLAS GEMM
   - `SharpInference.Vulkan` — Vulkan compute via `Vortice.Vulkan`, SPIR-V shaders, GPU buffer pool
3. **Engine** (`SharpInference.Engine`) — Forward pass orchestration, KV cache, temperature/top-k/top-p/min-p sampling, speculative decoding. Depends on Core + both backends.
4. **Frontends** — CLI (`Spectre.Console.Cli`, llama.cpp-compatible flags) and API Server (ASP.NET Core Minimal API with `/v1/messages` Anthropic + `/v1/chat/completions` OpenAI endpoints).

Supporting libraries:
- **TurboQuant** — KV cache compression using Lloyd-Max codebooks (3-4 bit). Codebook data lives in `codebooks/`.
- **Pipeline** — 3-tier memory hierarchy (VRAM → pinned RAM → NVMe), SLRU expert cache, async prefetcher.

## Key Interfaces & Patterns

- `IComputeBackend` (in Core) is the central abstraction — defines MatMul, RmsNorm, RoPE, Softmax, SiLU, Attention, and memory management. CPU and Vulkan backends implement it.
- `IForwardPass` (in Core) — per-token forward pass; implemented by `ForwardPass` (CPU), `GpuForwardPass`, `HybridForwardPass`. Has `Forward`, `TruncateTo`, `ResetCache`, `VocabSize`, `MaxSeqLen`.
- `IInferenceEngine` (in Engine) — top-level generation interface used by the server: `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`. Serializes concurrent requests.
- `ForwardPass` / `GpuForwardPass` in Engine dispatch the transformer layer sequence.
- Hot paths use `NativeMemory`, `Span<T>`, and Vulkan buffers — no managed heap allocations.
- Unsafe code is used throughout for performance. `AllowUnsafeBlocks` is enabled globally.

## Build Constraints

- **TreatWarningsAsErrors** is enabled globally — all warnings must be resolved.
- **Trim and AOT analyzers** are enabled — code must be NativeAOT-compatible (no reflection-heavy patterns, no dynamic code generation).
- **InvariantGlobalization** is on — no culture-specific string operations.
- Vulkan shaders (GLSL) are in `shaders/` and compiled to SPIR-V.

## Test Projects

| Test Project | Covers |
|---|---|
| Tests.Core | GGUF parsing, tokenizer, model graph |
| Tests.ForwardPass | Forward pass, KV cache, sampler, Vulkan |
| Tests.Pipeline | Memory hierarchy |
| Tests.TurboQuant | KV cache compression |
| Tests.Server | API endpoints |

## Design Documentation

Detailed architecture doc at `docs/SharpInference-Design.md` covering all subsystems, algorithms, and data layouts.
