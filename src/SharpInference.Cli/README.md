# SharpInference.Cli

`sharpi-cli` — a command-line tool for LLM inference and image generation, powered by [SharpInference](https://www.nuget.org/packages/SharpInference). Reads GGUF models and runs transformer inference on CPU (AVX2/AVX-512 SIMD) or GPU (Vulkan / CUDA).

## Install

```
dotnet tool install -g SharpInference.Cli
```

Or update:

```
dotnet tool update -g SharpInference.Cli
```

## Usage

```
# Text generation (CPU)
sharpi-cli -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time" --temp 0.7

# All layers on GPU (Vulkan or CUDA, auto-selected)
sharpi-cli -m models/Qwen3-8B-Q4_K_M.gguf -p "Explain mmap" -g -1

# Interactive chat (omit -p to enter chat mode)
sharpi-cli -m models/Qwen3-8B-Q4_K_M.gguf

# Image generation (Z-Image-Turbo, requires CUDA)
sharpi-cli image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "a serene mountain lake at sunrise" -W 512 -H 512 --steps 4 -o out.png
```

Flag names are intentionally compatible with `llama.cpp` / `llama-cli`.

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | auto-detect | Path to GGUF model file |
| `-p, --prompt` | (interactive) | Input prompt; omit to enter chat |
| `-n, --n-predict` | `512` | Maximum tokens to generate |
| `--temp` | `0.7` | Sampling temperature (`0` = greedy) |
| `--top-k` | `40` | Top-k sampling |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--min-p` | `0.05` | Min-p sampling |
| `-g, --n-gpu-layers` | `0` | Layers on GPU (`0` = CPU only, `-1` = all) |
| `-c, --ctx-size` | model default | Context / max sequence length |
| `--tq` | off | TurboQuant KV cache compression (3-bit, ~5× VRAM reduction) |

Run `sharpi-cli --help` for the full reference.

## Requirements

- .NET 10 runtime (the tool installs framework-dependent)
- x86-64 CPU with **AVX2** support
- For GPU inference: Vulkan-capable GPU (any vendor) or NVIDIA GPU with CUDA 11.x / 12.x

## Links

- [Library package](https://www.nuget.org/packages/SharpInference)
- [Repository & docs](https://github.com/pekkah/SharpInference)

## License

MIT. Copyright (c) 2026 Pekka Heikura.
