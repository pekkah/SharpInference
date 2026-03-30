# SharpInference — Architecture

## Overview

SharpInference is a multi-backend LLM inference engine targeting native AOT deployment.
It supports CPU (SIMD) and Vulkan GPU backends with a three-tier memory hierarchy
(VRAM → RAM → NVMe), speculative decoding, and KV-cache compression via Lloyd-Max
scalar quantisation (TurboQuant).

---

## Project Dependency Graph

```
SharpInference.Server   SharpInference.Cli
         └─────────────────┘
                  │
         SharpInference.Engine
          │    │    │    │    │
      Core  Cpu Vulkan TurboQuant Pipeline
          └────┴────┴──────┴────────┘
                  │
         SharpInference.Core
```

---

## Core (`SharpInference.Core`)

- **`GgufReader`** — streaming GGUF v1/v2/v3 parser using `System.IO.Pipelines`.
- **`IComputeBackend`** — compute abstraction: `MatMul`, `AddInPlace`, `RmsNorm`,
  `Softmax`, `SiLU`, `RoPE`, `Upload`, `Download`, `Synchronize`.
- **`Tensor` / `TensorShape` / `DType`** — lightweight tensor descriptors; actual
  memory is backend-owned.
- **`ModelGraph`** — layer list + weight index built from a parsed GGUF file.
- **`ITokenizer`** — BPE / SentencePiece / Tiktoken via `Microsoft.ML.Tokenizers`.

---

## CPU Backend (`SharpInference.Cpu`)

- **`CpuBackend`** — implements `IComputeBackend` using `System.Runtime.Intrinsics`
  (AVX2 / AVX-512 / NEON). Unsafe native memory for zero-copy tensor storage.

---

## Vulkan Backend (`SharpInference.Vulkan`)

- **`VulkanBackend`** — dispatches precompiled SPIR-V compute shaders via Silk.NET.Vulkan.
- Shaders live in `shaders/` (GLSL source + compiled `.spv` bytecode).
- Operations: `matmul.comp`, `add_inplace.comp`, `rmsnorm.comp`, `softmax.comp`,
  `silu.comp`, `rope.comp`.

---

## TurboQuant (`SharpInference.TurboQuant`)

- **`LloydMaxCodebook`** — precomputed quantisation codebooks (JSON in `codebooks/`).
- **`KvCacheCompressor`** — compress/decompress KV-cache tensors using scalar quantisation.
- Target: 4–8x KV-cache size reduction with < 0.5 ppl degradation.

---

## Pipeline (`SharpInference.Pipeline`)

- **`MemoryHierarchy`** — three-tier (VRAM / RAM / NVMe) tensor placement and LRU eviction.
- **`ExpertCache`** — per-expert LRU cache for MoE models.
- **`Prefetcher`** — async background worker that promotes tensors to GPU before they
  are needed, hiding I/O latency behind compute.

---

## Engine (`SharpInference.Engine`)

- **`InferenceEngine`** — top-level orchestrator: prefill + decode loop.
- **`ForwardPass`** — stateless attention + FFN kernels (RoPE, SwiGLU, GQA).
- **`KvCache`** — preallocated per-layer key/value tensors.
- **`Sampler`** — greedy, top-k, top-p, min-p, temperature, repetition penalty.
- **`SpeculativeDecoder`** — draft model generates K tokens; target model verifies in
  one pass.

---

## Server (`SharpInference.Server`)

ASP.NET Core Minimal API, compiled with NativeAOT (`PublishAot=true`).
Uses `WebApplication.CreateSlimBuilder` for minimal startup overhead.

- **`/v1/completions`** and **`/v1/chat/completions`** — OpenAI-compatible.
- **`/v1/messages`** — Anthropic Messages API-compatible.
- **`/health`** — liveness probe.
- SSE streaming for token-by-token output.

---

## CLI (`SharpInference.Cli`)

- **`ChatRepl`** — interactive chat loop with streaming output.
- **`BenchRunner`** — measures prefill/decode throughput, TTFT, memory usage.

---

## Benchmarks (`SharpInference.Benchmarks`)

BenchmarkDotNet harness targeting `net10.0`. Measures:

- Prefill throughput (tokens/s)
- Decode throughput (tokens/s)
- Time to first token (ms)
- Peak memory (GB)

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `IComputeBackend` abstraction | Swap CPU/Vulkan/future CUDA without touching inference logic |
| `System.IO.Pipelines` for GGUF | Backpressure-aware zero-copy streaming for multi-GB model files |
| NativeAOT server | Sub-10ms cold start, minimal memory overhead for container deployments |
| Lloyd-Max KV compression | Asymmetric quantisation tuned per layer; outperforms uniform quantisation |
| Speculative decoding | 2–4x decode speedup with no quality loss using small draft model |
| Three-tier memory | Enables 70B+ models on consumer hardware (24 GB VRAM + 64 GB RAM + NVMe) |
