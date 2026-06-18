# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SharpInference is a high-performance LLM inference engine and image generation pipeline in C# 14 / .NET 10. It reads GGUF model files and runs transformer inference on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders, CUDA/cuBLAS). Also supports text-to-image generation via Z-Image-Turbo (S3-DiT + Qwen3-4B + FLUX VAE) and 4× image upscaling via RRDBNet (Real-ESRGAN). Targets NativeAOT for single-binary deployment.

## Build & Test Commands

```bash
dotnet build                # Debug build
dotnet build -c Release     # Release (NativeAOT opts: IlcOptimizationPreference=Speed)
dotnet test                 # Run all tests (207 tests across 5 projects)
dotnet test --filter "FullyQualifiedName~SomeTest"  # Run a single test

# Run CLI inference
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0

# GPU backend (all layers offloaded)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0 -g -1

# VibeThinker-1.5B (Qwen2-based math/reasoning, issue #282). Loads as a standard
# qwen2 GGUF (QKV bias but no output-projection bias, no QK-norm, 28 layers / 2 KV
# heads, ChatML, tied embeddings). Default download is Q8_0 (near-lossless; `-Model
# vibethinker-q4` for the smaller quant). Recommended sampling: temp 0.6, top_p 0.95,
# top_k 0, and no system prompt (the chat template supplies the math one). Emits a long
# <think> chain-of-thought then a \boxed{} answer (handled by the generic think machinery).
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/VibeThinker-1.5B.Q8_0.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  -p "If 5x + 3 = 2x + 18, what is x? Show your reasoning."

# Start API server (OpenAI + Anthropic compatible). SharpInference.Server is the
# ASP.NET Core library that ships AddSharpInference() / MapSharpInference();
# SharpInference.Server.Host is the runnable demo host you'd publish.
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server.Host -c Release

# NativeAOT publish
dotnet publish src/SharpInference.Cli -c Release -r win-x64
dotnet publish src/SharpInference.Server.Host -c Release -r win-x64

# Benchmarks
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'

# Image generation with upscaling (Z-Image-Turbo + RRDBNet)
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  --upscaler models/RealESRGAN_x4plus.safetensors \
  --upscale-blend 0.8 \
  -p "a serene mountain lake at sunrise" -W 512 -H 512 --steps 4 -o out.png

# Image generation micro-benchmarks
dotnet run --project benchmarks/SharpInference.ImageBench -c Release -- --bench --filter '*'
```

## Architecture

Four-layer stack, bottom-up:

1. **Core** (`SharpInference.Core`) — GGUF parser (memory-mapped), BPE tokenizer (`Microsoft.ML.Tokenizers`), tensor types, model graph. Everything depends on this.
2. **Compute Backends** — Three implementations of `IComputeBackend`; `CudaBackend` also implements `IImageOpsBackend` for convolutional image ops:
   - `SharpInference.Cpu` — AVX2/AVX-512 SIMD kernels, Q4_K_M dequantization, optional OpenBLAS GEMM
   - `SharpInference.Vulkan` — Vulkan compute via `Vortice.Vulkan`, SPIR-V shaders, GPU buffer pool
   - `SharpInference.Cuda` — cuBLAS GEMM (fp32/bf16/fp16/fp8) + NVRTC custom kernels (im2col, element-wise ops) for DiT and RRDBNet; includes `GpuBufferPool` to eliminate per-GEMM `cudaMalloc`/`cudaFree` overhead
3. **Engine** (`SharpInference.Engine`) — Forward pass orchestration, KV cache, temperature/top-k/top-p/min-p sampling, speculative decoding. Depends on Core + both backends.
4. **Frontends** — CLI (`Spectre.Console.Cli`, llama.cpp-compatible flags) and API Server. The API server is split: `SharpInference.Server` is an ASP.NET Core class library exposing `AddSharpInference()` / `MapSharpInference()` with the `SharpInferenceServerOptions` options pattern (`/v1/messages` Anthropic + `/v1/chat/completions` OpenAI endpoints); `SharpInference.Server.Host` is the runnable demo host (one `Program.cs`, AOT-published) that consumes it.

Supporting libraries:
- **TurboQuant** — KV cache compression using Lloyd-Max codebooks (3-4 bit). Codebook data lives in `codebooks/`.
- **Pipeline** — 3-tier memory hierarchy (VRAM → pinned RAM → NVMe), SLRU expert cache, async prefetcher.

## Key Interfaces & Patterns

- `IComputeBackend` (in Core) is the central abstraction — defines MatMul, RmsNorm, RoPE, Softmax, SiLU, Attention, and memory management. CPU and Vulkan backends implement it.
- `IImageOpsBackend` (in Core) — extends `IComputeBackend` with convolutional image ops (Conv2d, LeakyRelu, CatChannels, PixelShuffle, Upsample2x). Implemented by `CudaBackend` and `VulkanBackend` for the RRDBNet upscaler.
- `IForwardPass` (in Core) — per-token forward pass; implemented by `ForwardPass` (CPU), `GpuForwardPass`, `HybridForwardPass`. Has `Forward`, `Prefill`, `TruncateTo`, `ResetCache`, `VocabSize`, `MaxSeqLen`.
- `PagedKvCache` (in Engine) — lazily allocated paged KV cache used by `ForwardPass`. Pages (16 positions) allocated on first write; `TruncateTo` is a soft operation (enables prefix reuse); `Reset` returns pages to warm pool.
- `IInferenceEngine` (in Engine) — top-level generation interface used by the server: `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`. Implemented by `InferenceEngine` (single-user, prefix caching) and `ContinuousBatchingEngine` (multi-user batching, activated via `SHARPI_MAX_BATCH`).
- `ForwardPass.BatchForwardMulti(tokens[], positions[], caches[])` — batched multi-sequence decode; amortizes weight reads N× across concurrent users. Each sequence has its own `PagedKvCache`. Not supported for MoE or TurboQuant.
- `ForwardPass.PrefillWithCache(tokens, cache, startPos)` — prefills a per-sequence cache (used by `ContinuousBatchingEngine` during request admission). Admission is chunked (`SHARPI_PREFILL_CHUNK`, default 256 tokens) and interleaved with decode steps; multiple in-flight prompts prefill as one packed pass via `ForwardPass.PrefillPackedMulti` and admission is gated by a KV token budget (`SHARPI_KV_BUDGET_MB`) — issue #183.
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
