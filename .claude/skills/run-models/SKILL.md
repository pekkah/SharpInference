---
name: run-models
description: CLI run recipes for specific models and modes — VibeThinker, Ornith-1.0, Gemma 4 vision, image generation + upscaling, perplexity gating, whole-turn JSON-schema structured output, DSpark speculative decoding, download-model presets. Use when actually running inference for one of these.
---

# Model-specific run recipes

Basic run/GPU/inspect/server commands are in the root CLAUDE.md. These are the
model- and mode-specific invocations.

## Perplexity (accuracy gate for KV compression, issue #180)

Supports `--tq`/`--tq-mode` exactly like the run command (auto = KVarN where
supported, else Lloyd-Max).

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  perplexity -m model.gguf -f corpus.txt -c 2048 --tq
```

## Whole-turn structured output (grammar-constrained decoding, issues #423/#425)

Mirrors llama.cpp's `-j`/`--json-schema`; the entire response is constrained to the
schema. The root must be an object schema with at least one property;
`--json-schema-ordered` emits keys in declared order. The server exposes the same
via OpenAI/Anthropic `response_format:json_schema`.

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m model.gguf --temp 0 -p "Extract name and age from: Alice is 30." \
  -j '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}'
```

## VibeThinker-1.5B (Qwen2-based math/reasoning, issue #282)

Loads as a standard qwen2 GGUF (QKV bias but no output-projection bias, no QK-norm,
28 layers / 2 KV heads, ChatML, tied embeddings). `download-model.ps1 -Model
vibethinker` fetches the default Q8_0 (near-lossless); `-Model vibethinker-q4` is
the smaller quant. Recommended sampling: temp 0.6, top_p 0.95, top_k 0, and no
system prompt (the chat template supplies the math one). Emits a long `<think>`
chain-of-thought then a `\boxed{}` answer.

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/VibeThinker-1.5B.Q8_0.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 0 \
  -p "If 5x + 3 = 2x + 18, what is x? Show your reasoning."
```

## Gemma 4 encoder-free vision (issue #250)

Pass one or more PNGs with `--image` and `--mmproj` (the gemma4uv projector GGUF).
CPU-only single-prompt path for now (`-g 0`).

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/gemma-4-E4B-it.gguf --mmproj models/gemma4-mmproj.gguf -g 0 \
  --image photo.png -p "Describe <image>"
```

## Ornith-1.0 (DeepReinforce, MIT)

Agentic-coding "self-scaffolding" RL finetunes of Qwen3.5 / Gemma 4 bases, NOT a
new architecture — self-scaffolding is a training-time technique; at inference
they're ordinary transformers. GGUF arches reduce to ones already dispatched:
9B = `qwen35`, 35B/397B = `qwen35moe`. Validated end-to-end (issue #411): the
bartowski 9B Q4_K_M GGUF actually ships GDN tensors, so `_sharpi.is_hybrid_ssm`
auto-activates and it takes the SAME hybrid Gated-DeltaNet + attention path as the
35B/397B MoE variants (24 GDN + 8 full-attention layers,
full_attention_interval=4) — not a plain dense transformer as the arch name alone
suggests. Full CUDA offload (`-g -1`) fits comfortably in 8 GB VRAM (~3 GB weights
uploaded; GDN state + dense FFN run on CPU by design of
`CudaHybridGdnForwardPass`). Chat template loads via `JinjaChatTemplate` and tool
calls parse via the qwen35moe-style `QwenToolCallAdapter` (Qwen3.6 XML
`<function=..><parameter=..>` inside `<tool_call>`). They're tagged
image-text-to-text, but the Qwen3.5 vision projector is unimplemented — the text
GGUF path is text-only, which is what the coding use case needs.
`download-model.ps1 -Model ornith-9b` (Q4_K_M, 5.5 GB).

```bash
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 20 -p "Write a Python LRU cache."
```

## Image generation with upscaling (Z-Image-Turbo + RRDBNet)

`ImageCommand` auto-detects Z-Image vs FLUX from the model. Z-Image uses a
Qwen3-4B text encoder; FLUX uses CLIP-L + T5.

```bash
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

## DSpark speculative decoding (docs/dspark-plan.md, PR #413)

Greedy only: `--dspark-model <safetensors-or-dir> --temp 0` with `-g 0` or
`-g -1`. Placement via `DSparkPlacementPlanner` / `--dspark-place` /
`SHARPI_DSPARK_*` (GPU draft needs a CUDA target and free VRAM — pass `-c` to
bound the target's KV solve). Fetch heads with
`download-model.ps1 -Model dspark-qwen3-4b`. Server: `SHARPI_DSPARK_MODEL` on the
single-user engine (`MaxBatchSize` 1), engaging on greedy
`enable_thinking:false` requests. On a 4B target the un-graphed verify pass caps
DSpark below plain graph-replayed decode — see the plan doc's Phase-4 numbers
before benchmarking. Internals: `src/SharpInference.Engine/CLAUDE.md`.

## Model downloads

`pwsh scripts/download-model.ps1 -Model <preset>` fetches known presets
(smollm2, vibethinker[-q4], qwen3-8b, olmoe-1b-7b, qwen3-coder-30b-a3b,
qwen36-35b-a3b[-mtp], ornith-9b/-35b, gemma4-12b-qat, gemma4-e4b-qat,
llama4-scout, z-image-turbo[-q8], realesrgan-x4, dspark-qwen3-4b, ...). See the
script header for the full ValidateSet. Models land in `models/` (gitignored).
