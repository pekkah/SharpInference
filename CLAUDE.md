# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0 -g -1

# Inspect a GGUF file
dotnet run --project src/SharpInference.Cli -c Release -- list-metadata -m model.gguf
dotnet run --project src/SharpInference.Cli -c Release -- list-tensors  -m model.gguf

# VibeThinker-1.5B (Qwen2-based math/reasoning, issue #282). Loads as a standard
# qwen2 GGUF (QKV bias but no output-projection bias, no QK-norm, 28 layers / 2 KV
# heads, ChatML, tied embeddings). `download-model.ps1 -Model vibethinker` fetches the
# default Q8_0 (near-lossless); `-Model vibethinker-q4` is the smaller quant. Recommended
# sampling: temp 0.6, top_p 0.95, top_k 0, and no system prompt (the chat template supplies
# the math one). Emits a long <think> chain-of-thought then a \boxed{} answer.
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/VibeThinker-1.5B.Q8_0.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  -p "If 5x + 3 = 2x + 18, what is x? Show your reasoning."

# Gemma 4 encoder-free vision (issue #250): pass one or more PNGs with --image and
# --mmproj (the gemma4uv projector GGUF). CPU-only single-prompt path for now (-g 0).
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/gemma-4-E4B-it.gguf --mmproj models/gemma4-mmproj.gguf -g 0 \
  --image photo.png -p "Describe <image>"

# Start API server (OpenAI + Anthropic compatible). SharpInference.Server is the
# ASP.NET Core library that ships AddSharpInference() / MapSharpInference();
# SharpInference.Server.Host is the runnable demo host you'd publish.
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server.Host -c Release

# NativeAOT publish (the three packable frontends + libraries)
dotnet publish src/SharpInference.Cli -c Release -r win-x64
dotnet publish src/SharpInference.Server.Host -c Release -r win-x64

# Benchmarks
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'

# Models: scripts/download-model.ps1 fetches known presets (smollm2, vibethinker,
# qwen3-8b, olmoe-1b-7b, qwen3-coder-30b-a3b, qwen36-35b-a3b[-mtp], gemma4-12b-qat,
# gemma4-e4b-qat, llama4-scout, z-image-turbo[-q8], realesrgan-x4, ...). Run with
# `-Model <name>` (PowerShell). See the script header for the full ValidateSet.

# Image generation with upscaling (Z-Image-Turbo + RRDBNet). ImageCommand auto-detects
# Z-Image vs FLUX from the model. Z-Image uses a Qwen3-4B text encoder; FLUX uses CLIP-L + T5.
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

The solution (`SharpInference.slnx`) is a four-layer stack, bottom-up:

1. **Core** (`SharpInference.Core`) — GGUF parser (memory-mapped), BPE/SPM tokenizer (`Microsoft.ML.Tokenizers`), Jinja chat templates (`JinjaChatTemplate`), tool-call adapter, UTF-8 stream decoder, tensor types, model graph. Everything depends on this. Defines the central interfaces (`IComputeBackend`, `IImageOpsBackend`, `IForwardPass`, `ITokenizer`).
2. **Compute Backends** — Three implementations of `IComputeBackend`; `CudaBackend` and `VulkanBackend` also implement `IImageOpsBackend` for convolutional image ops:
   - `SharpInference.Cpu` — AVX2/AVX-512 SIMD kernels (`SimdKernels`), Q4_K_M/Q6_K/Q8 dequantization (`Dequantize`), Gated-DeltaNet kernels (`GdnKernels`), optional OpenBLAS GEMM (`BlasInterop`)
   - `SharpInference.Vulkan` — Vulkan compute via `Vortice.Vulkan`, SPIR-V shaders, GPU buffer pool
   - `SharpInference.Cuda` — cuBLAS GEMM + NVRTC runtime-compiled kernels. `CudaTextKernels` (RMSNorm/RoPE/softmax/GQA attention/Q4_K-Q6_K-F32 matvecs/KV-append), `CudaKernels` (im2col + conv for DiT/RRDBNet), `CudaWsKernels` (weight-stationary batched-decode matvecs, issue #194), `CudaRaggedKernels` (ragged batched decode for SnapKV-evicted caches), plus `GpuBufferPool` to eliminate per-GEMM `cudaMalloc`/`cudaFree` overhead
3. **Engine** (`SharpInference.Engine`) — Forward-pass orchestration, KV cache, sampling, speculative decoding, MoE expert offloading, continuous batching. Depends on Core + backends.
4. **Frontends** —
   - **CLI** (`SharpInference.Cli`, `Spectre.Console.Cli`, llama.cpp-compatible flags): `RunCommand` (default text/vision inference), `ImageCommand` (`image` subcommand), `ListMetadataCommand` (`list-metadata`), `ListTensorsCommand` (`list-tensors`).
   - **API Server**: `SharpInference.Server` is an ASP.NET Core class library exposing `AddSharpInference()` / `MapSharpInference()` with the `SharpInferenceServerOptions` options pattern (OpenAI `/v1/chat/completions` + `/v1/models`, Anthropic `/v1/messages`, OpenAI Responses, `/health`, `/metrics`). `SharpInference.Server.Host` is the runnable demo host (one `Program.cs`, AOT-published) that consumes it.

Supporting libraries:
- **SharpInference.Diffusion** — Native image-generation pipelines. `ZImagePipeline` (Z-Image-Turbo: `ZImageDiT` single-stream S3-DiT + Qwen3-4B encoder + FLUX VAE) and `ImagePipeline` (`FluxDiT` multi-stream MMDiT + CLIP-L/T5 encoders). Includes `VaeDecoder`, `RRDBNet` (Real-ESRGAN 4× upscaler), `EulerFlowScheduler`, 2D RoPE, FP8 conversion, and Safetensors/GGUF weight loaders. Text encoders live in `TextEncoders/`.
- **SharpInference.Vision** — Gemma 4 encoder-free vision projector (`gemma4uv`). `VisionModel` loads the mmproj GGUF; `GemmaUvVisionEmbedder` does im2col patches → projection → soft tokens; `ImagePreprocessor`/`ImageIO` handle image loading.
- **SharpInference.TurboQuant** — KV cache compression using Lloyd-Max codebooks (3-4 bit). Codebook data lives in `codebooks/`.
- **SharpInference.Pipeline** — 3-tier memory hierarchy (VRAM → pinned RAM → NVMe), SLRU expert cache, async prefetcher.

## Key Interfaces & Patterns

- `IComputeBackend` (in Core) is the central abstraction — defines MatMul, RmsNorm, RoPE, Softmax, SiLU, Attention, and memory management. CPU, Vulkan, and CUDA backends implement it.
- `IImageOpsBackend` (in Core) — extends `IComputeBackend` with convolutional image ops (Conv2d, LeakyRelu, CatChannels, PixelShuffle, Upsample2x). Implemented by `CudaBackend` and `VulkanBackend` for the RRDBNet upscaler and VAE.
- `IForwardPass` (in Core) — per-token forward pass. Implementations in Engine: `ForwardPass` (CPU dense), `GpuForwardPass` (Vulkan), `CudaForwardPass` (CUDA dense), `HybridForwardPass`/`CudaHybridForwardPass` (dense + MoE expert offload), `HybridGdnForwardPass`/`CudaHybridGdnForwardPass` (qwen35moe hybrid Gated-DeltaNet + MoE). Has `Forward`, `Prefill`, `TruncateTo`, `ResetCache`, `VocabSize`, `MaxSeqLen`.
- `IBatchedForwardPass` (in Engine) — multi-token batched prefill/decode used by continuous batching.
- `PagedKvCache` (in Engine) — lazily allocated paged KV cache used by `ForwardPass`. Pages (16 positions) allocated on first write; `TruncateTo` is a soft operation (enables prefix reuse); `Reset` returns pages to a warm pool. Other cache types: `KvCache` (simple), `CudaSequenceKvCache` (per-sequence GPU), `TurboQuantKvCache` (3-bit compressed). `IMultiSlotKvCache` abstracts per-sequence/multi-slot caches. `SnapKvSelector` does prefill-time SnapKV eviction; `GdnStateCache` snapshots Gated-DeltaNet state for MTP rollback.
- `IInferenceEngine` (in Engine) — top-level generation interface used by the server: `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`. Implemented by `InferenceEngine` (single-user, prefix caching) and `ContinuousBatchingEngine` (multi-user batching, activated via `SHARPI_MAX_BATCH`).
- `ForwardPass.BatchForwardMulti(tokens[], positions[], caches[])` — batched multi-sequence decode; amortizes weight reads N× across concurrent users. Each sequence has its own `PagedKvCache`. Not supported for MoE or TurboQuant.
- `ForwardPass.PrefillWithCache(tokens, cache, startPos)` — prefills a per-sequence cache (used by `ContinuousBatchingEngine` during request admission). Admission is chunked (`SHARPI_PREFILL_CHUNK`, default 256 tokens) and interleaved with decode steps; multiple in-flight prompts prefill as one packed pass via `ForwardPass.PrefillPackedMulti` and admission is gated by a KV token budget (`SHARPI_KV_BUDGET_MB`) — issue #183.
- **Speculative decoding** — `SpeculativeDecoder` (general draft-model speculation), `MtpDecoder` + `MtpBatchTail` (self-speculative Multi-Token Prediction / NEXTN heads, e.g. Qwen3.6-27B-MTP, with folded k-token batched verify, issue #207), and `PromptLookupDraft` (prompt-lookup draft). Toggle from the CLI (`--mtp`, `--draft-model`) or server (`SpecType`).
- `Sampler` (in Engine) — temperature, top-k, top-p (nucleus), min-p, repetition penalty, logit bias.
- MoE expert offload: `ExpertSlotManager`/`CudaExpertSlotManager` (SLRU VRAM expert cache), `MoEPrefetcher` (async SSD→RAM→VRAM), `TierPlanner` + `HardwareProfile` (three-tier placement), `MmapPrefault`, `WarmPinConfig`. `--cpu-moe` / `SHARPI_CPU_MOE` keeps routed experts on the CPU (issues #80/#93).
- Hot paths use `NativeMemory`, `Span<T>`, and GPU buffers — no managed heap allocations.
- Unsafe code is used throughout for performance. `AllowUnsafeBlocks` is enabled globally.

## Build Constraints

Shared settings live in `Directory.Build.props` (net10.0, LangVersion 14, Nullable enable, ImplicitUsings):

- **TreatWarningsAsErrors** is enabled globally — all warnings must be resolved.
- **Trim and AOT analyzers** are enabled (`IsTrimmable`, `EnableTrimAnalyzer`, `EnableAotAnalyzer`, warnings not suppressed) — code must be NativeAOT-compatible (no reflection-heavy patterns, no dynamic code generation). Server JSON uses a source-generated `SharpInferenceJsonContext`.
- **InvariantGlobalization** is on — no culture-specific string operations.
- Vulkan shaders (GLSL) are in `shaders/` and compiled to SPIR-V.
- Versioning is MinVer-derived from git tags (`v*`); only the `SharpInference` meta-package, `SharpInference.Server`, and `SharpInference.Cli` are packable.

## Test Projects

Over 1,000 tests across 7 projects (xUnit, `[Fact]`/`[Theory]`):

| Test Project | Covers |
|---|---|
| Tests.Core | GGUF parsing, tokenizer (SPM/BPE), Jinja chat templates, model graph, tool-call adapter, UTF-8 stream decode |
| Tests.ForwardPass | Forward pass (CPU/Vulkan/CUDA), KV cache, sampler, batched/ragged decode, MTP, SnapKV, quantization parity (largest suite, ~100 files) |
| Tests.Pipeline | Memory hierarchy, image pipeline integration |
| Tests.TurboQuant | KV cache compression (codebooks, encode/decode parity) |
| Tests.Server | API endpoints (OpenAI/Anthropic compatibility) |
| Tests.Cli | GPU device queries, CLI flags (e.g. `--cpu-moe`) |
| Tests.Vision | Gemma 4 vision embedder parity, image I/O, mmproj GGUF loading |

Shared test data lives in `tests/fixtures/`.

## Samples & Scripts

- `samples/SharpInference.Sample.Chat` — minimal streaming chat using the library directly.
- `samples/SharpInference.Sample.ToolCall` — tool/function-calling flow.
- `scripts/` — PowerShell benchmark drivers (`bench-*.ps1`), `download-model.ps1` (model fetcher), `setup-openblas.ps1` / `setup-llamacpp.ps1`, and Python reference-generation helpers for llama.cpp cross-checking (`gemma4uv_ref.py`, `extract_reference.py`, `compare_tokens.py`).

## Design Documentation

Detailed architecture doc at `docs/SharpInference-Design.md` covering all subsystems, algorithms, data layouts, and the (mostly completed) implementation phases. `docs/research/` and the various `docs/*-plan.md` files hold per-feature design notes (Gemma 4, qwen35moe, MoE offloading, KV-compression feasibility).
