# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Context is layered: this file holds only the always-relevant core. Directory-level
`CLAUDE.md` files (`src/SharpInference.Engine/`, `src/SharpInference.Vulkan/`,
`src/SharpInference.TurboQuant/`) load automatically when working in those subtrees,
and model-specific CLI run recipes live in the `run-models` skill.

## Project Overview

SharpInference is a high-performance LLM inference engine and image generation pipeline in C# 14 / .NET 10. It reads GGUF model files and runs transformer inference on CPU (AVX2/AVX-512 SIMD), Vulkan compute shaders, and CUDA/cuBLAS. Architectures supported include `llama`/`llama4`, `qwen2`, `qwen3`, `qwen3moe`, `qwen35moe` (hybrid Gated-DeltaNet + attention + MoE), `gemma`/`gemma2`/`gemma3`/`gemma4`, `phi2`/`phi3`, `deepseek2`, and OLMoE. It also supports text-to-image generation (Z-Image-Turbo and FLUX.1), 4× image upscaling via RRDBNet (Real-ESRGAN), and Gemma 4 encoder-free multimodal vision. Targets NativeAOT for single-binary deployment.

## Build & Test Commands

```bash
dotnet build                # Debug build
dotnet build -c Release     # Release (NativeAOT opts: IlcOptimizationPreference=Speed, IlcInstructionSet=native)
dotnet test                 # Run all tests (1,000+ tests across 7 test projects)
dotnet test --filter "FullyQualifiedName~SomeTest"  # Run a single test
dotnet test tests/SharpInference.Tests.ForwardPass  # Run one test project

# Run CLI inference (RunCommand is the implicit default command)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0

# GPU backend (all layers offloaded; -g/-1 = all layers, --backend cuda|vulkan|auto)
# Append: -g -1

# Inspect a GGUF file
dotnet run --project src/SharpInference.Cli -c Release -- list-metadata -m model.gguf
dotnet run --project src/SharpInference.Cli -c Release -- list-tensors  -m model.gguf

# Start API server (OpenAI + Anthropic compatible; Server = library, Server.Host = demo host)
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server.Host -c Release

# NativeAOT publish (packable frontends)
dotnet publish src/SharpInference.Cli -c Release -r win-x64
dotnet publish src/SharpInference.Server.Host -c Release -r win-x64

# Benchmarks
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'

# Models: pwsh scripts/download-model.ps1 -Model <preset>  (see script header for presets)
```

Model-specific recipes — VibeThinker, Ornith-1.0, Gemma 4 vision (`--image`/`--mmproj`),
image generation + upscaling, perplexity gating (`perplexity`), whole-turn JSON-schema
structured output (`-j`), DSpark speculative decoding — live in the `run-models` skill
(`.claude/skills/run-models/SKILL.md`).

## Architecture

The solution (`SharpInference.slnx`) is a four-layer stack, bottom-up:

1. **Core** (`SharpInference.Core`) — GGUF parser (memory-mapped), BPE/SPM tokenizer (`Microsoft.ML.Tokenizers`), Jinja chat templates (`JinjaChatTemplate`), tool-call adapter, UTF-8 stream decoder, tensor types, model graph (`ModelGraph.cs` — architecture dispatch), and grammar-constrained decoding (`Grammar/`: `ITokenConstraint`, `JsonSchemaOutputConstraint` for whole-turn structured output, per-family tool-argument constraints, `ToolSchemaCompiler` — issues #423/#425). Everything depends on this. Defines the central interfaces (`IComputeBackend`, `IImageOpsBackend`, `IForwardPass`, `ITokenizer`, `ITokenConstraint`).
2. **Compute Backends** — three implementations of `IComputeBackend` (`CudaBackend` and `VulkanBackend` also implement `IImageOpsBackend` for convolutional image ops):
   - `SharpInference.Cpu` — AVX2/AVX-512 SIMD kernels (`SimdKernels`), Q4_K_M/Q6_K/Q8 dequantization (`Dequantize`), Gated-DeltaNet kernels (`GdnKernels`), optional OpenBLAS GEMM (`BlasInterop`)
   - `SharpInference.Vulkan` — Vulkan compute via `Vortice.Vulkan`, SPIR-V shaders (precompiled — see that project's CLAUDE.md), GPU buffer pool
   - `SharpInference.Cuda` — cuBLAS GEMM + NVRTC runtime-compiled kernels (`CudaTextKernels`, `CudaKernels` for image ops, `CudaWsKernels` weight-stationary batched decode, `CudaRaggedKernels` for SnapKV-evicted caches, `GpuBufferPool`)
3. **Engine** (`SharpInference.Engine`) — forward-pass orchestration, KV caches, sampling, speculative decoding, MoE expert offloading, continuous batching. See `src/SharpInference.Engine/CLAUDE.md` for the type-level map.
4. **Frontends** —
   - **CLI** (`SharpInference.Cli`, `Spectre.Console.Cli`, llama.cpp-compatible flags): `RunCommand` (default; text/vision + `-j` structured output), `ImageCommand`, `PerplexityCommand`, `ListMetadataCommand`, `ListTensorsCommand`.
   - **API Server**: `SharpInference.Server` (ASP.NET Core library — `AddSharpInference()` / `MapSharpInference()`, OpenAI `/v1/chat/completions` + `/v1/models`, Anthropic `/v1/messages`, OpenAI Responses, `/health`, `/metrics`); `SharpInference.Server.Host` is the runnable AOT-published demo host.

Supporting libraries: **SharpInference.Diffusion** (Z-Image-Turbo `ZImagePipeline` and FLUX `ImagePipeline` DiTs, `VaeDecoder`, `RRDBNet` upscaler, `EulerFlowScheduler`, text encoders); **SharpInference.Vision** (Gemma 4 `gemma4uv` encoder-free vision projector); **SharpInference.TurboQuant** (KV cache compression — KVarN + Lloyd-Max; see that project's CLAUDE.md); **SharpInference.Pipeline** (3-tier VRAM→RAM→NVMe memory hierarchy, SLRU expert cache, async prefetcher).

## Key Interfaces & Patterns

- `IComputeBackend` (Core) — the central abstraction: MatMul, RmsNorm, RoPE, Softmax, SiLU, Attention, memory management. `IImageOpsBackend` extends it with Conv2d/LeakyRelu/CatChannels/PixelShuffle/Upsample2x for the upscaler and VAE.
- `IForwardPass` (Core) — per-token forward pass (`Forward`, `Prefill`, `TruncateTo`, `ResetCache`). Engine implementations: `ForwardPass` (CPU), `GpuForwardPass` (Vulkan), `CudaForwardPass` (CUDA), `Hybrid*`/`CudaHybrid*` (MoE expert offload), `HybridGdnForwardPass`/`CudaHybridGdnForwardPass` (hybrid Gated-DeltaNet). A numeric change in one usually needs the siblings updated too.
- `IInferenceEngine` (Engine) — `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`; `InferenceEngine` (single-user, prefix caching) and `ContinuousBatchingEngine` (multi-user, `SHARPI_MAX_BATCH`).
- KV caches: `PagedKvCache` (default, lazily paged), `KvCache`, `CudaSequenceKvCache`, `TurboQuantKvCache` (compressed), behind `IMultiSlotKvCache`; `SnapKvSelector` (prefill eviction), `GdnStateCache` (MTP rollback).
- Speculative decoding: `SpeculativeDecoder` (draft model), `MtpDecoder` (self-speculative MTP/NEXTN), `PromptLookupDraft`, `DSparkDecoder` (DeepSeek DSpark heads). CLI toggles `--mtp` / `--draft-model` / `--dspark-model`; server `SpecType`.
- `Sampler` (Engine) — temperature, top-k/top-p/min-p, repetition penalty, logit bias, and per-step `ITokenConstraint` masking for grammar-constrained output.
- Hot paths use `NativeMemory`, `Span<T>`, and GPU buffers — no managed heap allocations. Unsafe code is normal; `AllowUnsafeBlocks` is global.

Full detail: `src/SharpInference.Engine/CLAUDE.md` (type-level engine map) and `docs/SharpInference-Design.md` (authoritative subsystem reference).

## Build Constraints

Shared settings live in `Directory.Build.props` (net10.0, LangVersion 14, Nullable enable, ImplicitUsings):

- **TreatWarningsAsErrors** is enabled globally — all warnings must be resolved.
- **Trim and AOT analyzers** are enabled (`IsTrimmable`, `EnableTrimAnalyzer`, `EnableAotAnalyzer`, warnings not suppressed) — code must be NativeAOT-compatible (no reflection-heavy patterns, no dynamic code generation). Server JSON uses a source-generated `SharpInferenceJsonContext`.
- **InvariantGlobalization** is on — no culture-specific string operations.
- Vulkan shaders are precompiled to a committed SPIR-V table; editing any GLSL const in `src/SharpInference.Vulkan/Shaders.cs` requires regenerating it with `scripts/gen-spirv.ps1` or `VulkanPrecompiledShaderTests` fails on drift. Details in `src/SharpInference.Vulkan/CLAUDE.md`.
- Versioning is MinVer-derived from git tags (`v*`); only the `SharpInference` meta-package, `SharpInference.Server`, and `SharpInference.Cli` are packable.

## Test Projects

Over 1,000 tests across 7 projects (xUnit, `[Fact]`/`[Theory]`):

| Test Project | Covers |
|---|---|
| Tests.Core | GGUF parsing, tokenizer (SPM/BPE), Jinja chat templates, model graph, tool-call adapter, grammar constraints / JSON-schema structured output, UTF-8 stream decode |
| Tests.ForwardPass | Forward pass (CPU/Vulkan/CUDA), KV cache, sampler, batched/ragged decode, MTP, SnapKV, quantization parity (largest suite, ~100 files) |
| Tests.Pipeline | Memory hierarchy, image pipeline integration |
| Tests.TurboQuant | KV cache compression (codebooks, encode/decode parity) |
| Tests.Server | API endpoints (OpenAI/Anthropic compatibility) |
| Tests.Cli | GPU device queries, CLI flags (e.g. `--cpu-moe`) |
| Tests.Vision | Gemma 4 vision embedder parity, image I/O, mmproj GGUF loading |

Shared test data lives in `tests/fixtures/`.

## Samples, Scripts & Docs

- `samples/` — `Sample.Chat` (streaming chat via the library), `Sample.ToolCall` (function-calling flow).
- `benchmarks/` — `SharpInference.Bench` (text), `SharpInference.ImageBench` (image), `SnapKvEval` (eviction quality harness).
- `scripts/` — benchmark drivers (`bench-*.ps1`), `download-model.ps1`, `setup-openblas.ps1` / `setup-llamacpp.ps1`, llama.cpp cross-check helpers (see the `parity-check` skill).
- `docs/SharpInference-Design.md` — authoritative architecture doc; `docs/*-plan.md` + `docs/research/` hold per-feature design notes.
