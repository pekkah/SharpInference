# SharpInference

A high-performance LLM inference engine and image generation pipeline written in C# 14 / .NET 10.
Runs GGUF models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS).
Includes an OpenAI- and Anthropic-compatible API server and native pipelines for
[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) and FLUX.1.

**Requirements:** .NET 10 SDK, x86-64 CPU with AVX2.
Optional: Vulkan-capable GPU (drivers), CUDA Toolkit 11.x/12.x for NVIDIA paths,
OpenBLAS in `tools/openblas/` for faster batched GEMM. Build with `dotnet build -c Release`.

## Text generation

Supported architectures: `llama`, `llama4`, `olmoe`, `qwen3`, `qwen3moe`, `qwen35moe`
(hybrid Gated-DeltaNet + attention + MoE). Benchmarked on
AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB), Q4_K_M, `--temp 0`,
`-n 80`, prompt `"Write a Python function that sorts a list using the quicksort algorithm:"`.
Decode rate is **forward-pass iterations / decode time**, so it counts
thinking-mode tokens too. All outputs verified coherent
(`scripts/bench-all.ps1`). Cross-engine top-1 parity vs llama.cpp b8585
verified on Qwen3-8B (byte-identical 60-token greedy decode with
matching chat template).

| Model | Repo | Size | Backend | Prefill t/s | Decode t/s | Notes |
|---|---|---:|---|---:|---:|---|
| SmolLM2 1.7B Instruct | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | 1 GB | CPU | 16.6 | 38.9 | AVX2 fused dequant-matvec |
| SmolLM2 1.7B Instruct | (same) | 1 GB | Vulkan `-g -1` | 42.0 | **139.7** | GLSL `subgroupAdd` reduce |
| SmolLM2 1.7B Instruct | (same) | 1 GB | **CUDA** `-g -1` | **181.1** | **158.1** | NVRTC `__dp4a` + Q8_1 |
| Qwen3 8B | [Qwen](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5 GB | Vulkan `-g -1` | 23.0 | 45.8 | 11.4K auto-ctx |
| Qwen3 8B | (same) | 5 GB | Vulkan `-g -1 --tq` | 21.7 | 45.5 | 3-bit KV → 40 960 ctx |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1` | **65.9** | **58.6** | ~2.8× Vulkan prefill |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --no-thinking` | **66.0** | **58.2** | Same per-token rate; reasoning suppressed in chat template, so all decoded tokens are visible answer |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq` | **65.9** | **58.4** | 3-bit KV → 40 960 ctx; 17 t/s @ 8K, 10 t/s @ 16K |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq --no-thinking` | **66.1** | **58.1** | Same per-token rate as `--tq` alone; reasoning suppressed |
| OLMoE 1B-7B Instruct (MoE) | [allenai](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF) | 4 GB | CPU | 21.6 | 55.7 | 64 experts / 8 active; per-channel QK-norm; `norm_topk_prob=false` |
| OLMoE 1B-7B Instruct (MoE) | (same) | 4 GB | Vulkan `-g -1` | 18.9 | **121.2** | 16 layers all on VRAM; greedy on this prompt is unstable across backends — use `--temp 0.6 --top-p 0.95` for usable output |
| OLMoE 1B-7B Instruct (MoE) | (same) | 4 GB | **CUDA** `-g -1` | **117.4** | **111.7** | Same; greedy varies, sampling coherent |
| Qwen3-Coder 30B-A3B (MoE) | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 17 GB | CPU | 15.1 | 21.2 | 128 experts / 8 active |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | CPU `--tq` | 12.0 | 21.1 | 3-bit KV |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | Vulkan `-g -1` (hybrid) | 1.0 | 5.8 | 29 GPU + 19 CPU layers, SLRU expert slot cache |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | **CUDA** `-g -1` (hybrid) | **13.9** | **22.7** | 29 GPU + 19 CPU layers (auto), ~2.2× Vulkan decode |
| Llama-4 Scout 17B-16E (MoE) | [meta-llama](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 61 GB | CPU | 1.9 | 3.9 | 48 layers, 17B active params; split GGUF (Q4_K_M) |
| Llama-4 Scout 17B-16E (MoE) | (same) | 61 GB | CUDA `-g -1` (hybrid) | 0.9 | 2.1 | 7 GPU + 41 CPU layers — model dwarfs the 12 GB card, PCIe cost > GPU speedup so CPU-only wins here |
| Qwen3.6-35B-A3B (GDN+MoE) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 22 GB | CPU | 4.3 | 7.8 | hybrid GDN/attn, 256 experts / 8 active |
| Qwen3.6-35B-A3B (GDN+MoE) | (same) | 22 GB | **CUDA** `-g -1` (hybrid) | **11.2** | **23.8** | 10 attn + 30 GDN on GPU; MoE auto-routed to CPU, batched-expert dispatch (8 experts × 3 ops into 2 Parallel.For sweeps), shared expert kept on GPU and overlapped with the CPU routed loop |
| Qwen3.6-27B-MTP (GDN) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | 16 GB | CPU `--no-thinking` | 2.8 | **3.8** | dense 27B, hybrid GDN/attn, native MTP head; auto-engages MTP self-spec (issue #25) at greedy + `--no-thinking`. 95% draft acceptance (38/40); batched N=2 verify (#30) + fused Q6_K·Q8_K 2-input dot (#42) lift decode from 2.7 (sequential N=1) to 3.8 — 1.4× over MTP-off baseline |
| Qwen3.6-27B-MTP (GDN) | (same) | 16 GB | **CUDA** `-g -1 --no-thinking` (hybrid) | **5.8** | **10.4** | 20/64 dense FFN layers on GPU (3.3 GB) + GDN + attn KV resident; 44/64 FFN layers on CPU mmap. 95% draft acceptance; batched verify lifts decode from 6.1 to 10.4 (1.70× over MTP-off baseline). The CPU FFN majority batches via `CpuDenseFfn2` and the on-GPU FFN layers now batch via `MatMulN2` (issue #43 — one weight read per row, two outputs); the small additional Q4_K gain over the previous CPU-only batching reflects the 31% on-GPU FFN share at 12 GB |
| Qwen3.6-27B-MTP (GDN) | (same) | 19 GB | CPU `--no-thinking` `Q5_K_M` | 2.5 | **3.5** | Q5_K_M variant, ~10% slower than Q4_K_M as expected from weight bandwidth. 100% draft acceptance (40/40) on this prompt; batched verify lifts decode from 2.4 to 3.5 (1.46×) |
| Qwen3.6-27B-MTP (GDN) | (same) | 19 GB | **CUDA** `-g -1 --no-thinking` `Q5_K_M` (hybrid) | 1.1 | **7.6** | 13/64 FFN layers on GPU (2.4 GB) + GDN + attn KV resident; 51/64 FFN on CPU mmap. Uses `llm_embed_lookup_q5k` direct-read kernel (issue #39) and the Q5_K `MatMulN2` kernel (issue #43) for the on-GPU layers. 98% draft acceptance; batched verify lifts decode from 4.1 to 7.6 (1.85×) |
| Qwen3.6-35B-A3B-MTP (GDN+MoE) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | 22 GB | CPU `--no-thinking` | 6.9 | 8.0 | hybrid GDN/attn + 256-expert MoE + native MTP head (issue #44). 100% draft acceptance. Issue #45 enables `BatchForward2` for MoE MTP — attn/GDN/lm_head batch across t1/t2 but routed experts run sequentially per token (top-K differs), so the win is small (lm_head save + frame overhead). At parity with the 8.2 MTP-off baseline within CPU jitter |
| Qwen3.6-35B-A3B-MTP (GDN+MoE) | (same) | 22 GB | **CUDA** `-g -1 --no-thinking` (hybrid) | **13.0** | **22.9** | Requires `SHARPI_CPU_MOE=1`: 30 GDN + 10 attn + shared expert on GPU, MoE routed experts + MTP MoE FFN mmap'd on CPU. 100% draft acceptance. Issue #45 lifts decode from 22.9 (sequential MTP at 0.99× MTP-off baseline) to at-or-above MTP-off — modest because routed-expert weight reads can't share between tokens; the bandwidth-bound CPU MoE FFN runs sequentially per token. Issues #47 (async UploadViaStaging) + #49 (overlap `_lastHidden` D2H with lm_head MatMul) shave further µs/layer |

`--backend auto` (default) picks CUDA when available, sizing the GPU/CPU split from
VRAM via TierPlanner; falls through to Vulkan only when CUDA isn't present.
`--tq` enables 3-bit TurboQuant KV compression (CPU, Vulkan, CUDA; requires
`headDim ∈ {128, 256}`). MoE runs on GPU (full-offload or partial hybrid) on
both Vulkan and CUDA backends.

CPU TurboQuant K-scoring and V-aggregation use a FastScan-derived AVX2
kernel (issue #34): KV positions are packed into 32-position tiles with
codes stored as 4-bit nibbles, a per-query i8 LUT is built once per
(layer, kv-head), and each `dim` step reduces to a `vpshufb` against
the LUT instead of `vpgatherdd`. V-aggregation defers the per-position
sign-flip + inverse Walsh-Hadamard transform to one call per kv-head
(commutes through the Σ_t accumulation), so the IWHT cost goes from
`O(tqLength · dim log dim)` per token down to `O(dim log dim)`. On
Ryzen 9 7900X, per (layer, kv-head) cost of the combined K+V attention
hot path vs the previous per-block AVX2 path
(`TurboQuantOps.DequantDot4Avx2`):

| TQ positions | per-block K+V | FastScan K+V | speedup |
|---:|---:|---:|---:|
| 1 024 | 479 µs | 26 µs | 18× |
| 4 096 | 1 931 µs | 98 µs | 20× |
| 8 192 | 3 936 µs | 193 µs | 20× |
| 16 384 | 8 216 µs | 390 µs | 21× |

End-to-end gain on decode tracks the K+V share of token cost: small at
short context (a few percent at 256 ≤ ctx ≤ 1K, dominated by the FFN /
QKV weight reads on dense models) and growing with context length —
Qwen3-8B Q4_K_M CPU `--tq` 2K-position decode measures 10.9 t/s, and
the FastScan ratio projects roughly 2× at 8K and 3× at 16K vs the
per-block path (where the per-token K+V cost alone would dominate
decode time).

Session-lifetime weights (per-layer projections, expert FFNs, embedding,
output) on all three CUDA forward passes (`CudaForwardPass`,
`CudaHybridForwardPass`, `CudaHybridGdnForwardPass`) bypass the GPU buffer
pool and go through `cudaMalloc`/`cudaFree` at the exact tensor size.
The pool's power-of-2 round-up was wasting hundreds of MiB on big-tensor
layouts (a 17 MiB attn_gate rounds to 32 MiB; across 60+ layers that's a
couple of GiB — the difference between fitting one more FFN layer on a
12 GB card or not, see issues #25 / #26). Scratch and KV-cache allocations
stay pooled.

For hybrid SSM/attention models (`qwen35moe`), the CUDA backend keeps the
attention KV cache, the 30 Gated-DeltaNet layers (conv1d + rank-1 outer-product
recurrence), and the shared expert resident in VRAM; routed-expert dispatch
auto-selects between an SLRU GPU cache and CPU mmap reads based on what
fraction of experts can be cached at boot. Override with `SHARPI_CPU_MOE=0|1`.

On Ampere+ (sm_80+) the CUDA backend auto-selects bf16 cuBLAS GEMM, which
matches fp32 for almost all workloads. `SHARPI_CUDA_PRECISION=fp32|fp16|bf16|fp8`
overrides the compute type — handy for bisecting whether an output divergence
is mantissa-precision (changes between fp32 and bf16) or algorithmic
(unchanged). Custom NVRTC kernels keep their fp32 accumulators regardless.

### Multi-Token Prediction (MTP)

Models that ship native MTP heads (Qwen3.6-27B-MTP, Qwen3.5 / Qwen3.6 A3B-MTP,
DeepSeek V3/R1, …) get self-speculative decoding for free — no separate
draft model. The MTP path engages automatically when the forward pass reports
`HasMtpHead`, sampling is greedy (`--temp 0`), and the chat template renders
with `enable_thinking=false` (`--no-thinking`). The CLI prints `MTP accept: N%`
at the end of the run so the acceptance gap is visible.

CLI surface mirrors llama.cpp: `--spec-type <auto|none|mtp|draft-mtp>` forces
on/off (default `auto` matches the eligibility check above); `--spec-draft-n-max
<int>` sets the draft depth per spec step (1 or 2; >2 isn't implemented yet —
that needs tree drafts);  `--spec-draft-p-min <0..1>` accepts a draft when
`softmax(main)[draft] ≥ p` even if it isn't the argmax (lossy but higher
acceptance). `SHARPI_DISABLE_MTP=1` is the back-compat off-switch;
`SHARPI_DISABLE_BATCH_VERIFY=1` forces the legacy sequential N=1 path for
parity bisection. Batched N=2 verify (issue #30) is the default for dense
MTP models; MoE MTP models (Qwen3.6-35B-A3B-MTP) also engage batched verify
since issue #45 — attn/GDN/norms/lm_head amortise across t1/t2 while the
routed-expert FFN runs sequentially per token (per-token top-K diverges).
`--spec-draft-n-min` / `--spec-draft-p-min` are not yet wired (issues #37, #38).

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
