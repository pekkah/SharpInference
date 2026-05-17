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
```

```bash
# SmolLM2 1.7B — fast, low memory, great for testing (~1 GB)
huggingface-cli download bartowski/SmolLM2-1.7B-Instruct-GGUF SmolLM2-1.7B-Instruct-Q4_K_M.gguf --local-dir models
```

```bash
# Qwen3 8B — general purpose, fits in 6 GB VRAM (~5 GB)
huggingface-cli download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models
```

```bash
# Qwen3-Coder 30B-A3B — MoE coding model, ~20 t/s CPU (~17 GB)
huggingface-cli download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --local-dir models
```

```bash
# Llama 4 Scout 109B-16E — MoE, ~5 t/s CPU on DDR4-3200 (~61 GB, 2 shards)
huggingface-cli download unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF --include "Q4_K_M/*" --local-dir models
```

### Image generation models (Z-Image-Turbo)

```bash
# DiT model — Q5_K_M (5.5 GB, best quality); use z_image_turbo-Q4_K_M.gguf for ~4.5 GB
huggingface-cli download jayn7/Z-Image-Turbo-GGUF z_image_turbo-Q5_K_M.gguf --local-dir models
```

```bash
# VAE + tokenizer (from the original Tongyi-MAI repo)
huggingface-cli download Tongyi-MAI/Z-Image-Turbo --include "vae/*" "tokenizer/*" --local-dir models/z-image-turbo
```

```bash
# Text encoder — uncensored Qwen3-4B fine-tune (~2.9 GB)
huggingface-cli download BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1 Z-Image-AbliteratedV1.Q5_K_M.gguf --local-dir models
```

### Image generation models (FLUX.1)

```bash
# FLUX.1-schnell GGUF (~7–9 GB depending on quant)
huggingface-cli download city96/FLUX.1-schnell-gguf flux1-schnell-Q4_K_S.gguf --local-dir models
```

```bash
# VAE + CLIP-L + T5-XXL encoders
huggingface-cli download comfyanonymous/flux_text_encoders ae.safetensors clip_l.safetensors t5xxl_fp16.safetensors --local-dir models/flux
```

## Quick Start

Build first:

```bash
dotnet build -c Release
```

Each command below is on a single line so you can copy-paste straight into your terminal.

**SmolLM2 1.7B — single-turn, CPU:**

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "What is 2+2?" --temp 0
```

**SmolLM2 1.7B — single-turn, all-GPU:**

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "What is 2+2?" --temp 0 -g -1
```

**SmolLM2 1.7B — interactive chat:**

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

**Qwen3-Coder 30B-A3B (MoE) — CPU + KV-cache compression:**

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --tq -p "Implement a binary search tree in C#" --temp 0
```

**Speculative decoding (draft + target, ~2× faster at temp 0):**

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/Qwen3-8B-Q4_K_M.gguf --draft-model models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Write a quicksort in Python" --temp 0
```

**Start API server (CPU):**

```bash
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf dotnet run --project src/SharpInference.Server -c Release
```

PowerShell equivalent for the server line:

```powershell
$env:SHARPI_MODEL='models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'; dotnet run --project src/SharpInference.Server -c Release
```

## CLI Reference

Install the CLI as a .NET global tool:

```
dotnet tool install -g SharpInference.Cli
sharpi-cli [COMMAND] [OPTIONS]
```

Or run from a source checkout without installing:

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
| `--backend` | `auto` | GPU backend selection: `auto`, `cuda`, or `vulkan`. `auto` prefers CUDA on dense models with full offload, otherwise Vulkan. |
| `--tq` | off | Enable TurboQuant KV-cache compression (3-bit, ~5× less VRAM). CPU & Vulkan only — CUDA falls back to Vulkan when set. |
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
dotnet run --project src/SharpInference.Cli -c Release -- image -m models/z_image_turbo-Q5_K_M.gguf --vae models/z-image-turbo/vae --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -p "a serene mountain lake at sunrise" -W 1024 -H 1024 --steps 4 -o landscape.png -v
```

#### FLUX.1-schnell example

```bash
dotnet run --project src/SharpInference.Cli -c Release -- image -m models/flux1-schnell-Q4_K_S.gguf --vae models/flux/ae.safetensors --clip-l models/flux/clip_l.safetensors --clip-tokenizer models/flux/tokenizer_clip.json --t5xxl models/flux/t5xxl_fp16.safetensors --t5-tokenizer models/flux/tokenizer_t5.json -p "a cinematic photograph of a mountain lake" -W 512 -H 512 --steps 4 -o out.png
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
dotnet run --project src/SharpInference.Cli -c Release -- list-metadata -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
```

Prints all GGUF metadata key/value pairs in a table (architecture, context length, rope settings, tokenizer vocab, etc.).

## API Server

> **Note:** The ASP.NET host hasn't been exercised end-to-end — it builds and the
> endpoint handlers have unit tests, but running against real clients has not been
> validated. Expect it to need fixes.

Starts an HTTP server compatible with OpenAI and Anthropic clients. Defaults to `http://localhost:5000`.

Start the server (CPU):

```bash
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf dotnet run --project src/SharpInference.Server -c Release
```

PowerShell:

```powershell
$env:SHARPI_MODEL='models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf'; dotnet run --project src/SharpInference.Server -c Release
```

OpenAI chat completions (streaming):

```bash
curl http://localhost:5000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

Anthropic messages (non-streaming):

```bash
curl http://localhost:5000/v1/messages -H "Content-Type: application/json" -d '{"model":"smollm2","messages":[{"role":"user","content":"Hello"}],"max_tokens":256}'
```

OpenAI Responses API:

```bash
curl http://localhost:5000/v1/responses -H "Content-Type: application/json" -d '{"model":"smollm2","input":"Hello"}'
```

List loaded model:

```bash
curl http://localhost:5000/v1/models
```

Health check:

```bash
curl http://localhost:5000/health
```

Prometheus metrics:

```bash
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
| Llama 4 Scout 109B-16E Instruct | [unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) | llama4 | Q4_K_M | ~61 GB (2 shards) | MoE, 16 experts, ~5 t/s on DDR4-3200 |

Any GGUF model with architecture `llama`, `llama4`, `qwen3`, or `qwen3moe` should work.

### Image generation

| Model | HuggingFace repo | Quant | File size | Notes |
|-------|-----------------|-------|-----------|-------|
| Z-Image-Turbo DiT | [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) | Q5_K_M | 5.5 GB | Best quality; also Q4_K_M (4.5 GB) |
| Z-Image-Turbo text encoder | [BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1](https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1) | Q5_K_M | 2.9 GB | Uncensored fine-tune of Qwen3-4B |
| FLUX.1-schnell | [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) | Q4_K_S | ~7 GB | 4-step distilled; VAE+encoders from comfyanonymous/flux_text_encoders |

### What works today

Verified on AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB VRAM):

#### SmolLM2 1.7B Instruct (`llama`, headDim 64)

| Backend | `--tq` | Status | Decode (t/s) |
|---------|--------|--------|--------------|
| CPU (default) | — | ✓ works | ~49 |
| CPU | `--tq` | ✗ refused — `headDim 64 != 128/256` (clear error) | — |
| GPU (`-g -1`) | — | ✓ works | ~163 |
| GPU (`-g -1`) | `--tq` | ✗ refused — same headDim guard | — |
| Hybrid (`-g N`, 1 ≤ N < layers) | — | ✓ works | ~53 (`-g 8`) |
| Hybrid | `--tq` | ✗ refused — same headDim guard | — |

Copy-paste:

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Hello" --temp 0
```

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Hello" --temp 0 -g -1
```

#### Qwen3-Coder 30B-A3B Instruct (`qwen3moe`, headDim 128, MoE 128/8)

| Backend | `--tq` | Status | Decode (t/s) |
|---------|--------|--------|--------------|
| CPU (default) | — | ✓ works | ~21 |
| CPU | `--tq` | ✓ works | ~20 |
| GPU (`-g -1`) | any | ⚠ auto-fallback to CPU with a warning (issue [#2](https://github.com/pekkah/SharpInference/issues/2)) | — |
| Hybrid (`-g N`) | any | ✗ refused with error pointing to issue [#2](https://github.com/pekkah/SharpInference/issues/2) | — |
| Hybrid (`-g 1..9`, debug only) | any | ✓ works with `SHARPI_ALLOW_BROKEN_MOE_HYBRID=1` after the [#2](https://github.com/pekkah/SharpInference/issues/2) host-barrier fix | ~15 (`-g 1 --tq`) |

Copy-paste (recommended config for this machine):

```bash
dotnet run --project src/SharpInference.Cli -c Release -- -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --tq -p "Write a Python quicksort." --temp 0
```

#### Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

The local file (`models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf`, 269 MB) is incomplete — a real 70B Q4_K_M is ~40 GB across multiple GGUF shards. Loading it crashes with `AccessViolationException` in `PrefaultWeights`. Re-download as a multi-shard set, e.g.:

```bash
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF --include "Meta-Llama-3.1-70B-Instruct-Q4_K_M*.gguf" --local-dir models
```

#### Z-Image-Turbo (image generation)

Models on disk: `models/z_image_turbo-Q5_K_M.gguf`, `models/Z-Image-AbliteratedV1.Q5_K_M.gguf`, `models/z-image-turbo/{vae,tokenizer}/`. The optional `RealESRGAN_x4plus.safetensors` is not present — upscaling falls back to bicubic unless you fetch it (`scripts/download-model.ps1 -Model realesrgan-x4`).

Image generation has not been re-tested in this round; the existing example below should still work end-to-end:

```bash
dotnet run --project src/SharpInference.Cli -c Release -- image -m models/z_image_turbo-Q5_K_M.gguf --vae models/z-image-turbo/vae --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -p "a serene mountain lake at sunrise" -W 1024 -H 1024 --steps 4 -o landscape.png -v
```

### Known limitations

- **MoE on GPU**: any `qwen3moe`/`llama4` model with `-g N` (N > 0) is rejected with an explicit error. `-g -1` (auto) silently falls back to CPU. Tracking: [#2](https://github.com/pekkah/SharpInference/issues/2). The compute→host visibility bug (Bug 1 in #2) that produced NaN at low `-g N` is fixed; the residual prefetcher-path corruption past `~-g 9` keeps the guard in place. Set `SHARPI_ALLOW_BROKEN_MOE_HYBRID=1` to bypass the guard for debugging.
- **GPU embedding lookup in `HybridForwardPass`**: works around a shader bug by always doing the per-token embedding row dequant on CPU. Cost is one row of dequant per token (negligible). Tracking: [#3](https://github.com/pekkah/SharpInference/issues/3).
- **`--backend` flag**: now wired into text inference too — accepts `auto` (default), `cuda`, or `vulkan`. CUDA is picked automatically when the model is dense and you ask for full GPU offload (`-g -1` or `-g >= NumLayers`); MoE models stay on the Vulkan path. Tracking: [#4](https://github.com/pekkah/SharpInference/issues/4).
- **`--tq`** (TurboQuant): requires the model to have `headDim == 128` (most Qwen3 / Llama 3) or `256` (Llama 70B). SmolLM2's `headDim 64` is rejected with a clear error. Supported on the CPU and Vulkan paths; **CUDA `--tq` is not yet implemented** and auto-falls-back to Vulkan with a warning.

## Performance

Benchmarked on AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB).
Each row is a single run at `--temp 0`, `-n 80`–`-n 200`, from a short prompt (`"The capital of France is"`). Coherent output verified in every run. **Prefill** is the rate at which prompt tokens are scored before generation; **Decode** is the steady-state per-token generation rate after the prompt is consumed.

| Model (Q4_K_M) | Backend | Prefill t/s | Decode t/s | Auto-ctx | Notes |
|---|---|---:|---:|---:|---|
| SmolLM2 1.7B | CPU | 23.6 | 53.0 | n/a | AVX2 fused dequant-matvec, multi-threaded |
| SmolLM2 1.7B | Vulkan, `-g -1` | 35.4 | 156.7 | 8192 | GLSL compute shaders, `subgroupAdd` reduce |
| SmolLM2 1.7B | **CUDA**, `--backend cuda -g -1` | **172.5** | **150.0** | 8192 | NVRTC `__dp4a` + Q8_1 / cuBLAS bf16 |
| Qwen3 8B | Vulkan | 18.8 | 13.0 | 11 399 | |
| Qwen3 8B | Vulkan, `--tq` | 17.7 | 13.0 | **40 960** | 3-bit KV → full model max-ctx |
| Qwen3 8B | **CUDA** | **65.1** | 14.3 | 12 080 | **~3.4× Vulkan prefill** |
| Qwen3-Coder 30B-A3B (MoE) | Vulkan | 8.8 | 22.2 | varies | 128 experts / 8 active per token |
| Qwen3-Coder 30B-A3B (MoE) | Vulkan, `--tq` | 9.5 | 22.0 | varies | |
| Qwen3-Coder 30B-A3B (MoE) | `--backend cuda` → falls back | 8.7 | 21.8 | varies | CUDA path doesn't yet support MoE |

Notes on the CUDA path:

- **Q4_K matvec** uses an `__dp4a` (int8×4 → int32) cooperative kernel with a per-call Q8_1 input quantization pre-pass, modelled on llama.cpp's `mul_mat_vec_q4_K_q8_1`. Achieves ~400–500 GB/s effective HBM bandwidth on a 4070 Ti.
- **VRAM headroom** matters: the auto-context picker reserves `max(VRAM/3, 2 GiB)` so late weight allocations (notably the 600 MiB `lm-head` of Qwen3) stay in HBM. An earlier conservative `max(VRAM/5, 1 GiB)` left ~24 MiB free with Qwen3-8B and pushed `lm-head` into system RAM, where it streamed at ~30 GB/s over PCIe and prefill collapsed to ~4 t/s.
- **TurboQuant on CUDA is not yet implemented**: `--tq` currently auto-falls-back to Vulkan with a warning. Vulkan + `--tq` already unlocks the full 40 960-token context on Qwen3-8B at essentially the same throughput as without `--tq`.
- **MoE on CUDA** falls back to Vulkan; the CUDA forward pass only handles dense (`qwen3` / `llama`) architectures.

The two `_BENCH` environment variables for reproducing the kernel-level numbers:
`SHARPI_CUDA_PROFILE=1` dumps per-phase timings at process exit;
`SHARPI_CUDA_MATVEC_BENCH=1` runs a pure-HBM memcpy baseline plus a per-shape Q4_K matvec microbench at backend init.

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

## Releasing

Two NuGet packages are published from this repo:

| Package | Contents |
|---------|----------|
| `SharpInference` | Library: all 8 inference / image-gen assemblies in a single package |
| `SharpInference.Cli` | `dotnet tool` exposing `sharpi-cli` |

The server (`SharpInference.Server`) is not published.

Versioning is handled by [MinVer](https://github.com/adamralph/minver) and driven by git tags (`v` prefix):

- **Preview**: every push to `master` publishes `0.X.Y-alpha.0.N` (where `N` is the commit height since the last tag). NuGet hides prereleases by default.
- **Release**: pushing a tag like `v0.2.0` publishes the stable version `0.2.0`.

```bash
# Cut a release
git tag v0.2.0
git push origin v0.2.0
```

The [`.github/workflows/release.yml`](.github/workflows/release.yml) workflow runs `dotnet pack` + `dotnet nuget push --skip-duplicate` and uses the `NUGET_KEY` repo secret.

## License

Released under the [MIT License](LICENSE).

