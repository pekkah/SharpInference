# SharpInference

A high-performance LLM inference engine and image generation pipeline written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS).
Includes an OpenAI- and Anthropic-compatible API server and native pipelines for
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and FLUX.1 image generation.

> **Status: spike.** A quick experiment to see how LLM tooling can be built
> from scratch in .NET. Things may be broken or not work as advertised. No warranty — see [LICENSE](LICENSE).

## Prerequisites

### Required

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- x86-64 CPU with **AVX2** support (Haswell / Zen 1 or newer)

### Optional native dependencies

| Feature | Dependency | Notes |
|---------|-----------|-------|
| **Faster batched GEMM (CPU)** | [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS/releases) | Place `libopenblas.dll` in `tools/openblas/` or system PATH. Auto-detected at startup; silently skipped if absent. |
| **GPU inference (Vulkan)** | Vulkan-capable GPU + drivers | Works on AMD/Intel/NVIDIA. No extra install on Windows — just up-to-date GPU drivers. The `VULKAN_SDK` env var is used for shader recompilation only. |
| **GPU inference (CUDA)** | [CUDA Toolkit 11.x](https://developer.nvidia.com/cuda-toolkit) | Requires `cublas64_11.dll` and `cudart64_110.dll` on PATH (CUDA 11 runtime). NVRTC resolver additionally tries `nvrtc64_120_0.dll` (CUDA 12.x), then `nvrtc64_112_0.dll`, then `nvrtc64_11*.dll`. NVIDIA GPU only. Used for image generation pipelines. |
| **Image upscaling (RRDBNet)** | CUDA (above) | Real-ESRGAN ×2/×4 upscaler. Falls back to bicubic if CUDA is unavailable. |

## Getting Models

All models use the [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) and are downloaded from [Hugging Face](https://huggingface.co).

### Text generation models

The fastest way to download is with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
pip install huggingface_hub
mkdir -p models

# SmolLM2 1.7B — fast, low memory, great for testing (~1 GB)
huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF \
  SmolLM2-1.7B-Instruct-Q4_K_M.gguf --local-dir models

# Qwen3 8B — general purpose, fits in 6 GB VRAM (~5 GB)
huggingface-cli download Qwen/Qwen3-8B-GGUF \
  qwen3-8b-q4_k_m.gguf --local-dir models

# Qwen3-Coder 30B-A3B — MoE coding model, ~20 t/s CPU (~17 GB)
huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --local-dir models

# Llama 4 Scout 109B-16E — MoE, ~5 t/s CPU on DDR4-3200 (~65 GB)
huggingface-cli download unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF \
  Llama-4-Scout-17B-16E-Instruct-Q4_K_M.gguf --local-dir models
```

### Image generation models (Z-Image-Turbo)

```bash
# DiT model (choose one quant)
huggingface-cli download jayn7/Z-Image-Turbo-GGUF \
  z_image_turbo-Q5_K_M.gguf --local-dir models        # 5.5 GB, best quality
  # z_image_turbo-Q4_K_M.gguf --local-dir models      # 4.5 GB, slightly faster

# VAE + tokenizer (inside the same repo)
huggingface-cli download jayn7/Z-Image-Turbo-GGUF \
  --include "vae/*" "tokenizer/*" --local-dir models/z-image-turbo

# Text encoder — uncensored Qwen3-4B fine-tune (~2.9 GB)
huggingface-cli download BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1 \
  Z-Image-AbliteratedV1.Q5_K_M.gguf --local-dir models
```

### Image generation models (FLUX.1)

```bash
# FLUX.1-schnell GGUF (~7–9 GB depending on quant)
huggingface-cli download city96/FLUX.1-schnell-gguf \
  flux1-schnell-Q4_K_S.gguf --local-dir models

# VAE + CLIP-L + T5-XXL encoders
huggingface-cli download comfyanonymous/flux_text_encoders \
  ae.safetensors clip_l.safetensors t5xxl_fp16.safetensors --local-dir models/flux
```

## Quick Start

```bash
# Build in release mode
dotnet build -c Release

# Single-turn inference (CPU)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" --temp 0

# Single-turn inference (GPU — all layers in VRAM)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "What is 2+2?" --temp 0 -g -1

# Interactive chat session
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf

# MoE coding model (~20 t/s CPU)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  -p "Implement a binary search tree in C#" --temp 0

# Speculative decoding (draft + target model, ~2× faster at temp 0)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf \
  --draft-model models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "Write a quicksort in Python" --temp 0

# Start API server
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release
```

## CLI Reference

The CLI is run with `dotnet run` until a NuGet package is published:

```
dotnet run --project src/SharpInference.Cli -c Release -- [COMMAND] [OPTIONS]
```

### Text inference (default command)

Flag names are intentionally compatible with llama.cpp / llama-cli.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | auto-detect | Path to GGUF model file |
| `-p, --prompt` | — | Input prompt; omit to enter interactive chat |
| `-n, --n-predict` | `512` | Maximum tokens to generate |
| `-c, --ctx-size` | model default | Context / max sequence length (0 = model default) |
| `--temp` | `0.7` | Sampling temperature (`0` = greedy / deterministic) |
| `--top-k` | `40` | Top-k sampling (`0` = disabled) |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--min-p` | `0.05` | Min-p sampling |
| `--rep-penalty` | `1.1` | Repetition penalty (`1.0` = disabled) |
| `-s, --seed` | `-1` | RNG seed (`-1` = random) |
| `-g, --n-gpu-layers` | `0` | Layers to offload to GPU (`0` = CPU only, `-1` = all) |
| `--tq` | off | Enable TurboQuant KV-cache compression (3-bit, ~5× less VRAM) |
| `--single-turn` | off | Generate one response and exit (non-interactive) |
| `--system-prompt` | — | System prompt prepended to conversation |
| `--no-display-prompt` | off | Suppress echoing the prompt |
| `--verbose-prompt` | off | Print token IDs before generating |
| `--draft-model` | — | Path to draft model for speculative decoding (requires `--temp 0`) |
| `--spec-lookahead` | `4` | Draft tokens per speculative step |
| `--min-batch-blas` | `16` | Minimum batch size to use OpenBLAS SGEMM (also: `SHARPI_MIN_BATCH_BLAS` env var) |

### `image` — image generation

Supports two native pipelines: **Z-Image-Turbo** (auto-detected from model filename) and **FLUX.1**.

#### Z-Image-Turbo example

```bash
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "a serene mountain lake at sunrise" \
  -W 1024 -H 1024 --steps 4 -o landscape.png -v
```

#### FLUX.1-schnell example

```bash
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/flux1-schnell-Q4_K_S.gguf \
  --vae models/flux/ae.safetensors \
  --clip-l models/flux/clip_l.safetensors \
  --clip-tokenizer models/flux/tokenizer_clip.json \
  --t5xxl models/flux/t5xxl_fp16.safetensors \
  --t5-tokenizer models/flux/tokenizer_t5.json \
  -p "a cinematic photograph of a mountain lake" \
  -W 512 -H 512 --steps 4 -o out.png
```

#### All image options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | — | Diffusion model GGUF |
| `-p, --prompt` | — | Text prompt describing the image |
| `--negative-prompt` | — | What to avoid in the image |
| `--vae` | — | VAE safetensors file or `vae/` directory |
| `--qwen-encoder` | — | *(Z-Image)* Qwen3-4B GGUF text encoder |
| `--qwen-tokenizer` | — | *(Z-Image)* Qwen3 `tokenizer.json` |
| `--clip-l` | — | *(FLUX)* CLIP-L encoder safetensors |
| `--clip-tokenizer` | — | *(FLUX)* CLIP `tokenizer.json` |
| `--t5xxl` | — | *(FLUX)* T5-XXL encoder safetensors |
| `--t5-tokenizer` | — | *(FLUX)* T5 `tokenizer.json` |
| `-W, --width` | `512` | Output width in pixels (must be divisible by 16) |
| `-H, --height` | `512` | Output height in pixels (must be divisible by 16) |
| `--steps` | `4` | Denoising steps (4 optimal for Z-Image-Turbo/FLUX schnell) |
| `--cfg-scale` | auto | Guidance scale (not used for distilled models) |
| `-s, --seed` | `-1` | RNG seed (`-1` = random) |
| `-g, --n-gpu-layers` | `-1` | GPU accel: `-1` = auto (CUDA→Vulkan→CPU), `0` = CPU only |
| `--backend` | `auto` | Force backend: `auto`, `cuda`, `vulkan`, `cpu` |
| `--upscaler` | — | Path to ESRGAN/Real-ESRGAN weights (`.safetensors`) for ×2/×4 upscale |
| `--upscale-blend` | `1.0` | Blend factor for upscaling (`1.0` = sharpest, lower = softer) |
| `-o, --output` | `output.png` | Output PNG path |
| `-v, --verbose` | off | Show per-step timing |

#### Z-Image-Turbo GPU acceleration timing

Benchmarked on AMD Zen 4 + RTX 4070 Ti:

| Stage | First run | Subsequent runs |
|-------|-----------|-----------------|
| Text encoder (Qwen3-4B, cuBLAS bf16) | ~90 s (weights cached in VRAM) | ~0 s (prompt cache) |
| DiT denoising — 4 steps (cuBLAS bf16) | ~4 s | ~4 s |
| VAE decoder (cuBLAS fp32 im2col) | ~23 s (weights cached in VRAM) | ~2 s |
| **Total** | **~117 s** | **~30 s** |

### `list-metadata` — inspect a GGUF file

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  list-metadata -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

Prints all GGUF metadata key/value pairs in a table (architecture, context length, rope settings, tokenizer vocab, etc.).

## API Server

> **Note:** The ASP.NET host hasn't been exercised end-to-end — it builds and the
> endpoint handlers have unit tests, but running against real clients has not been
> validated. Expect it to need fixes.

Starts an HTTP server compatible with OpenAI and Anthropic clients. Defaults to `http://localhost:5000`.

```bash
# Start (CPU)
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release

# OpenAI chat completions (streaming)
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Anthropic messages (non-streaming)
curl http://localhost:5000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'

# OpenAI Responses API
curl http://localhost:5000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"smollm2","input":"Hello"}'

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
| `SHARPI_MAX_BATCH` | `1` | Enable continuous batching for N concurrent users (`> 1` activates `ContinuousBatchingEngine`) |
| `SHARPI_MIN_BATCH_BLAS` | `16` | Minimum batch size to use OpenBLAS SGEMM in `MatMulBatched` |

## Supported & Tested Models

### Text generation

| Model | HuggingFace repo | Architecture | Quant | File size | Notes |
|-------|-----------------|--------------|-------|-----------|-------|
| SmolLM2 1.7B Instruct | [HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | llama | Q4_K_M | ~1 GB | Fast, low RAM, great for testing |
| Qwen3 8B | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | qwen3 | Q4_K_M | ~5 GB | General purpose; fits in 6 GB VRAM |
| Qwen3-Coder 30B-A3B Instruct | [Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) | qwen3moe | Q4_K_M | ~17 GB | MoE, 128 experts / 8 active, ~20 t/s CPU |
| Llama 4 Scout 109B-16E Instruct | [unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) | llama4 | Q4_K_M | ~65 GB | MoE, 16 experts, ~5 t/s on DDR4-3200 |

Any GGUF model with architecture `llama`, `llama4`, `qwen3`, or `qwen3moe` should work.

### Image generation

| Model | HuggingFace repo | Quant | File size | Notes |
|-------|-----------------|-------|-----------|-------|
| Z-Image-Turbo DiT | [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) | Q5_K_M | 5.5 GB | Best quality; also Q4_K_M (4.5 GB) |
| Z-Image-Turbo text encoder | [BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1](https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1) | Q5_K_M | 2.9 GB | Uncensored fine-tune of Qwen3-4B |
| FLUX.1-schnell | [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) | Q4_K_S | ~7 GB | 4-step distilled; VAE+encoders from comfyanonymous/flux_text_encoders |

## Performance

Benchmarked on AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti:

| Model | Backend | Decode (t/s) | Notes |
|-------|---------|-------------|-------|
| SmolLM2 1.7B Q4_K_M | GPU (Vulkan) | 131.3 | Multi-row compute shaders + subgroupAdd |
| SmolLM2 1.7B Q4_K_M | CPU (AVX2) | 48.6 | Fused dequant-matvec, multi-threaded |
| Qwen3 8B Q4_K_M | GPU (Vulkan) | 43.5 | Full VRAM fit |
| Qwen3 8B Q4_K_M | CPU | 13.5 | 1.23× llama.cpp |
| Llama 4 Scout Q4_K_M | CPU | 5.3 | Bandwidth-limited on 65 GB DDR4 |
| Qwen3-Coder 30B-A3B Q4_K_M | CPU | 20.8 | MoE: only 8/128 experts active per token |
| llama.cpp SmolLM2 1.7B | CPU (reference) | 45.1 | Same hardware |

## Build & Test

```bash
dotnet build              # Debug build
dotnet build -c Release   # Release (IlcOptimizationPreference=Speed)
dotnet test               # Run all tests (207 tests across 5 projects)

# NativeAOT single-binary publish
dotnet publish src/SharpInference.Cli    -c Release -r win-x64
dotnet publish src/SharpInference.Server -c Release -r win-x64

# Benchmarks (requires benchmark models to be present)
dotnet run --project benchmarks/SharpInference.Bench -c Release -- --filter '*'
```

## Helper Scripts

The `scripts/` directory contains optional helpers for development and validation. The PowerShell scripts target Windows; the Python scripts require [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python).

| Script | Purpose |
|--------|---------|
| `download-model.ps1` | Downloads GGUF models into `models/` from Hugging Face. Accepts `-Model <name>` for any of `smollm2`, `qwen3-8b`, `llama31-70b`, `qwen3-coder-30b-a3b`, `llama4-scout`, `z-image-turbo`, `z-image-turbo-q8`, `realesrgan-x4`. Skips files already present. |
| `setup-openblas.ps1` | Downloads OpenBLAS (default `0.3.28`) and installs `libopenblas.dll` into `tools/openblas/` for the optional CPU GEMM acceleration path. |
| `setup-llamacpp.ps1` | Downloads prebuilt llama.cpp binaries into `tools/llama.cpp/`. Variants: `cpu` (default), `vulkan`, `cuda-12.4`, `cuda-13.1`. Used as the reference implementation for forward-pass validation. |
| `generate-reference-logits.ps1` | Runs llama.cpp with `--logits-all` on a fixed prompt and writes reference logits to `tests/reference-data/` for comparison against the SharpInference forward pass. Requires `setup-llamacpp.ps1` and `download-model.ps1 -Model smollm2` to have been run first. |
| `compare_tokens.py` | Python helper that tokenizes a chat prompt with `llama-cpp-python` and prints top-5 logits at each step. Used to debug divergence against Llama 4 Scout. |
| `extract_reference.py` | Python helper that prints model metadata (`n_vocab`, `n_ctx_train`, `n_embd`) and token IDs for prompt fragments. Useful when investigating tokenizer disagreements. |

Typical first-time setup on Windows:

```powershell
# From repo root
.\scripts\setup-openblas.ps1                  # optional, enables OpenBLAS GEMM
.\scripts\download-model.ps1 -Model smollm2   # fetch a small test model
.\scripts\setup-llamacpp.ps1                  # optional, for reference validation
.\scripts\generate-reference-logits.ps1       # optional, regenerates tests/reference-data/
```

## Projects

| Project | Description |
|---------|-------------|
| `SharpInference.Core` | GGUF parser, BPE tokenizer, tensor types, model graph |
| `SharpInference.Cpu` | CPU backend: AVX2/AVX-512 SIMD, Q4_K_M dequantization, optional OpenBLAS GEMM |
| `SharpInference.Vulkan` | GPU backend: Vulkan compute shaders via Vortice.Vulkan |
| `SharpInference.Cuda` | GPU backend: CUDA cuBLAS P/Invoke, NVRTC custom kernels (im2col, element-wise ops) |
| `SharpInference.Engine` | Forward pass (CPU/GPU/Hybrid), paged KV cache, sampling, speculative decoding |
| `SharpInference.Diffusion` | Z-Image-Turbo + FLUX.1 pipeline: DiT, VAE decoder, Qwen3 + CLIP-L + T5-XXL encoders |
| `SharpInference.TurboQuant` | KV-cache compression using 3-bit Lloyd-Max codebooks |
| `SharpInference.Pipeline` | 3-tier memory hierarchy (VRAM → RAM → NVMe), SLRU expert cache, async prefetcher |
| `SharpInference.Cli` | CLI tool (`sharpi-cli`) with NativeAOT support |
| `SharpInference.Server` | OpenAI + Anthropic + Responses API server with NativeAOT support |

## Architecture

See [docs/SharpInference-Design.md](docs/SharpInference-Design.md).

## License

Released under the [MIT License](LICENSE).

