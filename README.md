# SharpInference

A high-performance LLM inference engine and image generation pipeline written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS).
Includes an OpenAI- and Anthropic-compatible API server and native pipelines for
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and FLUX.1.

**Requirements:** .NET 10 SDK, x86-64 CPU with AVX2.
Optional: Vulkan-capable GPU (drivers), CUDA Toolkit 11.x/12.x for NVIDIA paths,
OpenBLAS in `tools/openblas/` for faster batched GEMM. Build with `dotnet build -c Release`.

## Text generation

Supported architectures: `llama`, `llama4`, `qwen3`, `qwen3moe`, `qwen35moe`
(hybrid Gated-DeltaNet + attention + MoE). Benchmarked on
AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB), Q4_K_M, `--temp 0`,
`-n 80`, prompt `"Write a Python function that sorts a list using the quicksort algorithm:"`.
Decode rate is **forward-pass iterations / decode time**, so it counts
thinking-mode tokens too. Outputs spot-checked for coherence
(`scripts/bench-all.ps1`); MoE on Vulkan hybrid is currently a known
broken row — see ⚠ note. Cross-engine top-1 parity vs llama.cpp b8585
verified on Qwen3-8B (byte-identical 60-token greedy decode with
matching chat template).

| Model | Repo | Size | Backend | Prefill t/s | Decode t/s | Notes |
|---|---|---:|---|---:|---:|---|
| SmolLM2 1.7B Instruct | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | 1 GB | CPU | 16.6 | 42.5 | AVX2 fused dequant-matvec |
| SmolLM2 1.7B Instruct | (same) | 1 GB | Vulkan `-g -1` | 41.9 | **138.7** | GLSL `subgroupAdd` reduce |
| SmolLM2 1.7B Instruct | (same) | 1 GB | **CUDA** `-g -1` | **180.7** | **156.7** | NVRTC `__dp4a` + Q8_1 |
| Qwen3 8B | [Qwen](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5 GB | Vulkan `-g -1` | 22.7 | 45.4 | 11.4K auto-ctx |
| Qwen3 8B | (same) | 5 GB | Vulkan `-g -1 --tq` | 21.6 | 45.7 | 3-bit KV → 40 960 ctx |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1` | **63.5** | **57.2** | ~2.8× Vulkan prefill |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --no-thinking` | **63.4** | **57.1** | Same per-token rate; reasoning suppressed in chat template, so all decoded tokens are visible answer |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq` | **65.7** | **57.7** | 3-bit KV → 40 960 ctx; 17 t/s @ 8K, 10 t/s @ 16K |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq --no-thinking` | **64.9** | **56.8** | Same per-token rate as `--tq` alone; reasoning suppressed |
| Qwen3-Coder 30B-A3B (MoE) | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 17 GB | CPU | 13.6 | 20.6 | 128 experts / 8 active |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | CPU `--tq` | 12.3 | 20.7 | 3-bit KV |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | Vulkan `-g -1` (hybrid) | 1.0 | 9.1 | ⚠ output incoherent on this path — under investigation |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | **CUDA** `-g -1` (hybrid) | **15.2** | **21.7** | 29 GPU + 19 CPU layers (auto), ~2.4× Vulkan decode |
| Llama-4 Scout 17B-16E (MoE) | [meta-llama](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 61 GB | CPU | 1.4 | 3.3 | 48 layers, 17B active params; split GGUF (Q4_K_M) |
| Llama-4 Scout 17B-16E (MoE) | (same) | 61 GB | CUDA `-g -1` (hybrid) | 0.8 | 1.9 | 7 GPU + 41 CPU layers — model dwarfs the 12 GB card, PCIe cost > GPU speedup so CPU-only wins here |
| Qwen3.6-35B-A3B (GDN+MoE) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 22 GB | CPU | 4.4 | 7.9 | hybrid GDN/attn, 256 experts / 8 active |
| Qwen3.6-35B-A3B (GDN+MoE) | (same) | 22 GB | **CUDA** `-g -1` (hybrid) | **10.6** | **22.7** | 10 attn + 30 GDN on GPU; MoE auto-routed to CPU, batched-expert dispatch (8 experts × 3 ops into 2 Parallel.For sweeps), shared expert kept on GPU and overlapped with the CPU routed loop |

`--backend auto` (default) picks CUDA when available, sizing the GPU/CPU split from
VRAM via TierPlanner; falls through to Vulkan only when CUDA isn't present.
`--tq` enables 3-bit TurboQuant KV compression (CPU, Vulkan, CUDA; requires
`headDim ∈ {128, 256}`). MoE runs on GPU (full-offload or partial hybrid) on
both Vulkan and CUDA backends.

For hybrid SSM/attention models (`qwen35moe`), the CUDA backend keeps the
attention KV cache, the 30 Gated-DeltaNet layers (conv1d + rank-1 outer-product
recurrence), and the shared expert resident in VRAM; routed-expert dispatch
auto-selects between an SLRU GPU cache and CPU mmap reads based on what
fraction of experts can be cached at boot. Override with `SHARPI_CPU_MOE=0|1`.

### Reasoning models

Models that emit `<think>...</think>` (Qwen3, DeepSeek-R1, SmolLM3, …) are
detected automatically from their special tokens — no flag needed. The CLI
dims the reasoning stream as it generates. Use `--no-thinking` to disable
reasoning at the chat-template level, `--hide-thinking` to keep it on but
hide the stream, and `--max-thinking-tokens N` to force-close runaway
reasoning. Greedy decoding (`--temp 0`) on these models often loops, so
the CLI warns and recommends `--temp 0.6 --top-p 0.95 --top-k 20`.

The API server surfaces reasoning per each protocol's convention: Anthropic
`/v1/messages` emits a `thinking` content block before `text`; OpenAI
`/v1/chat/completions` exposes `reasoning_content` alongside `content`
(vLLM / DeepSeek style). Anthropic's `thinking.budget_tokens` and an OpenAI
extension `max_thinking_tokens` both map to the same engine-side budget.
Prior assistant turns in chat history have their `<think>` blocks stripped
before templating (Qwen3 and friends are trained without them).

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

# Reasoning model: stream shows dimmed <think>...</think>, then the answer
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf -g -1 --temp 0.6 --top-p 0.95 --top-k 20 \
  -p "What's 17 × 23?" --max-thinking-tokens 1024

# API server (OpenAI /v1/chat/completions + Anthropic /v1/messages, port 5000)
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server -c Release
```

## Image generation

Two pipelines, auto-detected from model filename. Benchmarked on AMD Zen 4
+ RTX 4070 Ti (CUDA backend, 4 denoising steps, 512×512 output). The CLI is
a one-shot binary, so each invocation pays the full load + text-encoder
warmup. The "cached" column is the steady-state cost when the same encoder
weights stay resident — e.g., re-rendering inside the server or interactive
loop after the first prompt.

| Pipeline | Components (repo • file • size) | Per-run | Cached prompt | Notes |
|---|---|---:|---:|---|
| **Z-Image-Turbo** | DiT: [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) `z_image_turbo-Q5_K_M.gguf` 5.5 GB<br/>Encoder: [BennyDaBall/...-AbliteratedV1](https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1) `Z-Image-AbliteratedV1.Q5_K_M.gguf` 2.9 GB<br/>VAE + tokenizer: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) `vae/` `tokenizer/` | **~108 s** | **~30 s** | Most of the per-run cost is text-encoder warmup (~90 s); DiT ~4 s, VAE ~18 s once weights are hot. Output verified visually. |
| **FLUX.1-schnell** | DiT: [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) `flux1-schnell-Q4_K_S.gguf` ~7 GB<br/>Encoders + VAE: [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) `clip_l.safetensors` + `t5xxl_fp16.safetensors` + `ae.safetensors` | — | — | 4-step distilled; model not on this benchmark machine |

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
- Tests: `dotnet test`
- NativeAOT publish: `dotnet publish src/SharpInference.Cli -c Release -r win-x64`

## License

Released under the [MIT License](LICENSE).
