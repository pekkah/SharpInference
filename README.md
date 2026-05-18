# SharpInference

A high-performance LLM inference engine and image generation pipeline written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS).
Includes an OpenAI- and Anthropic-compatible API server and native pipelines for
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and FLUX.1.

> **Status: spike.** A quick experiment to see how LLM tooling can be built
> from scratch in .NET. Things may be broken or not work as advertised. No warranty — see [LICENSE](LICENSE).

**Requirements:** .NET 10 SDK, x86-64 CPU with AVX2.
Optional: Vulkan-capable GPU (drivers), CUDA Toolkit 11.x/12.x for NVIDIA paths,
OpenBLAS in `tools/openblas/` for faster batched GEMM. Build with `dotnet build -c Release`.

## Text generation

Supported architectures: `llama`, `llama4`, `qwen3`, `qwen3moe`. Benchmarked on
AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB) at `--temp 0`, prompt
`"The capital of France is"`, Q4_K_M quant unless noted. Cross-engine top-1
parity vs llama.cpp b8585 verified on Qwen3-8B (24-token prefill + 60-token
greedy decode byte-identical with matching chat template).

| Model | Repo | Size | Backend | Prefill t/s | Decode t/s | Auto-ctx | Notes |
|---|---|---:|---|---:|---:|---:|---|
| SmolLM2 1.7B Instruct | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | 1 GB | CPU | 23.6 | 53.0 | — | AVX2 fused dequant-matvec |
| SmolLM2 1.7B Instruct | (same) | 1 GB | Vulkan `-g -1` | 35.4 | 156.7 | 8 192 | GLSL `subgroupAdd` reduce |
| SmolLM2 1.7B Instruct | (same) | 1 GB | **CUDA** `-g -1` | **172.5** | **150.0** | 8 192 | NVRTC `__dp4a` + Q8_1 |
| Qwen3 8B | [Qwen](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5 GB | Vulkan | 18.8 | 13.0 | 11 399 | |
| Qwen3 8B | (same) | 5 GB | Vulkan `--tq` | 17.7 | 13.0 | **40 960** | 3-bit KV → full ctx |
| Qwen3 8B | (same) | 5 GB | **CUDA** | **65.1** | 14.3 | 12 080 | ~3.4× Vulkan prefill |
| Qwen3-Coder 30B-A3B (MoE) | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 17 GB | CPU | — | 21.0 | varies | 128 experts / 8 active |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | Vulkan | 8.8 | 22.2 | varies | MoE on GPU only [#2](https://github.com/pekkah/SharpInference/issues/2) |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | CPU `--tq` | — | 20.0 | varies | 3-bit KV; CUDA path falls back |
| Llama 4 Scout 109B-16E | [unsloth](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) | 61 GB | CPU | — | ~5 | varies | DDR4-3200 bandwidth-bound |

`--backend auto` (default) picks CUDA for dense models with full offload (`-g -1`),
Vulkan otherwise. `--tq` enables 3-bit TurboQuant KV compression (CPU + Vulkan;
requires `headDim ∈ {128, 256}`). MoE on GPU is rejected by default — see [#2](https://github.com/pekkah/SharpInference/issues/2);
hybrid embed lookup runs on CPU per [#19](https://github.com/pekkah/SharpInference/issues/19).

### CLI examples

```bash
# CPU, single-turn, greedy
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "What is 2+2?" --temp 0

# Full GPU offload (auto-picks CUDA on dense + full offload)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf -p "Write a quicksort in Python" --temp 0 -g -1

# MoE on CPU with 3-bit KV compression (5× less VRAM, full ctx)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --tq -p "Implement a BST in C#" --temp 0

# Interactive chat (no -p)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf

# Speculative decoding (~2× faster at temp 0)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf --draft-model models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "Write a binary search in Rust" --temp 0

# API server (OpenAI /v1/chat/completions + Anthropic /v1/messages, port 5000)
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release
```

## Image generation

Two pipelines, auto-detected from model filename. Benchmarked on AMD Zen 4
+ RTX 4070 Ti (CUDA backend, 4 denoising steps, 1024×1024 output).

| Pipeline | Components (repo • file • size) | First run | Cached | Notes |
|---|---|---:|---:|---|
| **Z-Image-Turbo** | DiT: [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) `z_image_turbo-Q5_K_M.gguf` 5.5 GB<br/>Encoder: [BennyDaBall/...-AbliteratedV1](https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1) `Z-Image-AbliteratedV1.Q5_K_M.gguf` 2.9 GB<br/>VAE + tokenizer: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) `vae/` `tokenizer/` | **~117 s** | **~30 s** | First-run text-encoder warmup ~90 s; cached prompts skip it. DiT ~4 s, VAE ~2 s once cached. |
| **FLUX.1-schnell** | DiT: [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) `flux1-schnell-Q4_K_S.gguf` ~7 GB<br/>Encoders + VAE: [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) `clip_l.safetensors` + `t5xxl_fp16.safetensors` + `ae.safetensors` | — | — | 4-step distilled; not re-benchmarked this cycle |

Optional **4× upscale** via Real-ESRGAN (`RealESRGAN_x4plus.safetensors`):
runs on CUDA when available, falls back to bicubic.

### CLI examples

```bash
# Z-Image-Turbo (auto-detects pipeline from filename containing "z_image")
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "a serene mountain lake at sunrise" -W 1024 -H 1024 --steps 4 -o landscape.png

# FLUX.1-schnell
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/flux1-schnell-Q4_K_S.gguf \
  --vae models/flux/ae.safetensors \
  --clip-l models/flux/clip_l.safetensors --clip-tokenizer models/flux/tokenizer_clip.json \
  --t5xxl models/flux/t5xxl_fp16.safetensors --t5-tokenizer models/flux/tokenizer_t5.json \
  -p "a cinematic photograph of a mountain lake" -W 512 -H 512 --steps 4 -o out.png

# With 4× Real-ESRGAN upscale + blend
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  --upscaler models/RealESRGAN_x4plus.safetensors --upscale-blend 0.8 \
  -p "a fox in autumn forest" -W 512 -H 512 --steps 4 -o fox.png
```

## More

- Architecture & algorithms: [docs/SharpInference-Design.md](docs/SharpInference-Design.md)
- All CLI flags: `sharpi-cli --help`, `sharpi-cli image --help`
- Model downloads: `scripts/download-model.ps1 -Model <smollm2|qwen3-8b|qwen3-coder-30b-a3b|llama4-scout|z-image-turbo|realesrgan-x4|…>`
- Tests: `dotnet test` (207 tests across 5 projects)
- NativeAOT publish: `dotnet publish src/SharpInference.Cli -c Release -r win-x64`

## License

Released under the [MIT License](LICENSE).
