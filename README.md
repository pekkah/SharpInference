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
(hybrid Gated-DeltaNet + attention + MoE), `gemma4` (per-layer head_dim, SWA + global, PLE).
Benchmarked on AMD Zen 4 (12c/24t) + RTX 4070 Ti (12 GB), Q4_K_M, using each model's recommended sampling
(the per-model commands below; MTP rows run greedy, which is what engages MTP self-speculative decoding).
Prefill t/s is the warm-cache rate at a realistic ~2K-token working-context prompt; decode t/s is the
near-zero-ctx headline rate (forward-pass iterations / time, so thinking tokens count) — comparable to
llama.cpp `tg128`; the per-model long-context falloff lives in the **Long-context decode** section below.
Outputs verified coherent; Qwen3-8B is byte-identical to llama.cpp b8585 under greedy (60-token decode).

Tables are grouped per model, ordered fastest decode first; within each model the rows also run fastest
decode first. Each model lists a ready-to-run command with that model's recommended sampling (matching the
upstream / llama.cpp defaults). Commands use the published `sharpi-cli` binary — from source, swap it for
`dotnet run --project src/SharpInference.Cli -c Release --`.

#### SmolLM2 1.7B Instruct — [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) · 1 GB

```bash
sharpi-cli -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -g -1 \
  --temp 0.7 --top-p 0.95 --top-k 40 -p "Write a haiku about autumn"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1` | **194** | **268.0** | NVRTC `__dp4a` + Q8_1 |
| Vulkan `-g -1` | 128.6 | 184.3 | GLSL `subgroupAdd` reduce |
| CPU | 39.6 | 42.9 | AVX2 fused dequant-matvec |

#### OLMoE 1B-7B Instruct (MoE) — [allenai](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF) · 4 GB

```bash
# greedy is unstable on OLMoE across backends — sample instead
sharpi-cli -m models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 -p "Explain mixture-of-experts routing in two sentences"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1` | **97.4** | **126.0** | greedy varies, sampling coherent |
| Vulkan `-g -1` | 119.6 | **143.8** | greedy unstable across backends — use `--temp 0.6 --top-p 0.95` |
| CPU | 51.7 | 60.5 | 64 experts / 8 active; per-channel QK-norm; `norm_topk_prob=false` |

#### Gemma 4 E4B-it QAT q4_0 — [google](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf) · 5 GB

```bash
# Gemma 4 is not a reasoning model — the CLI auto-sets --no-thinking
sharpi-cli -m models/gemma-4-E4B_q4_0-it.gguf -g -1 -c 2048 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0 -p "Summarize the water cycle"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 -c 2048` | **3666** | **100.4** | QAT q4_0 (5.15 GB): **~1.4× decode** vs Q8 at near-identical quality, frees ~3 GB for wider KV/context. Same gemma4 path; the shared-KV tail layers omit `attn_k`/`attn_v`/`attn_k_norm` (loaded conditionally). `download-model.ps1 -Model gemma4-e4b-qat` |
| Vulkan `-g -1 -c 2048` | 35 | 39.5 | full gemma4 trunk incl. PLE injection + shared-KV tail (#351); per-token prefill (no batched-prefill path). `--kv-type bf16\|q8_0` for long-context KV (byte-identical greedy vs fp32). The 5 GB q4_0 fits Vulkan full-offload (the 8 GB Q8 build OOMs on a 12 GB card) |

#### Qwen3 8B — [Qwen](https://huggingface.co/Qwen/Qwen3-8B-GGUF) · 5 GB

```bash
# Qwen3 reasoning defaults; add --no-thinking for the faster non-reasoning rows
sharpi-cli -m models/Qwen3-8B-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 20 -p "Write a quicksort in Python"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 --no-thinking` | **2309** | **74.8** | reasoning suppressed; same path |
| **CUDA** `-g -1` | **2287** | **74.5** | int8 tensor-core MMQ prefill (weight read once as int8) + dp4a/Q8_1 decode matvec + CUDA-graph replay; all argmax-stable. (llama.cpp b8585 pp1008 5764 — gap is its cp.async MMQ + flash attn) |
| **CUDA** `-g -1 --tq --no-thinking` | 54.0 | **70.4** | as `--tq`, reasoning suppressed |
| **CUDA** `-g -1 --tq` | 54.1 | **70.2** | 3-bit KV → 40 960 ctx; 17 t/s @ 8K, 10 @ 16K |
| Vulkan `-g -1` | 43.5 | 51.2 | int8 dp4a/Q8_1 matvec + split-KV flash-decode |
| Vulkan `-g -1 --tq` | 36.0 | 51.0 | 3-bit KV → 40 960 ctx |
| CPU | 10.0 | 13.2 | dense, no KV compression |
| CPU `--tq` | 9.4 | 13.2 | 3-bit KV → 40 960 ctx; FastScan K+V (#34) keeps long-ctx decode ~flat (10.2 @ 3K, 9.4 @ 6K) |

#### Gemma 4 E4B-it Q8 — [unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) · 8 GB

```bash
sharpi-cli -m models/gemma-4-E4B-it-Q8_0.gguf -g -1 -c 2048 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0 -p "Summarize the water cycle"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 -c 2048` | **4145** | **70.5** | all 42 layers fit at `-c 2048`; KV-share alias + per-layer SWA/global split. Prefill: int8 tensor-core MMQ + tensor-core flash attention + SoA Q8_0 weight repack. Prompts >4096 use a real SWA KV ring on the chunked-flash path (fixes correctness past the 512 window). Decode: dp4a/Q8_1 + CUDA-graph replay; argmax-stable. (llama.cpp ~8475 prefill / ~78 decode) |
| **CUDA** `-g 22 -c 2048` (hybrid) | 6.8 | 7.0 | 22 GPU + 20 CPU layers. `-g ≤ 22` required so the CPU shared-KV tail can read its own-KV source layers; CPU dense-FFN dominates decode (bandwidth-bound). `SHARPI_CUDA_PROFILE=1` for per-phase breakdown |
| Vulkan `-g -1` | — | — | full gemma4 trunk runs on Vulkan (#351), but the 8 GB Q8 weights **OOM on a 12 GB card** at full-offload (CUDA fits it via a tighter SWA KV ring). Use the q4_0 QAT build above for Vulkan; Vulkan partial-offload (`-g N`) is not yet wired for gemma4 (#252) |
| CPU | 5.0 | 5.1 | dense 42-layer gemma4: per-layer head_dim (256 SWA / 512 global), dual-RoPE, KV-share tail (18 layers), 5:1 SWA:global, logit softcap 30, PLE-256 injection (~4.2 GB mmap-resident) |

#### Gemma 4 12B-it QAT Q4_0 — [google](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) · 7 GB

```bash
sharpi-cli -m models/gemma-4-12b-it-qat-q4_0.gguf -g -1 -c 2048 \
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0 -p "Describe a sunset in three sentences"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 -c 2048` | **1714** | **54.1** | dense 48-layer gemma4, all bulk weights Q4_0 + tied Q6_K embedding; fits 12 GB full-offload. `attention_k_eq_v` (8 global MQA layers reuse K as V), per-layer KV heads (8 GQA / 1 MQA), pure V-norm. Prefill: Q4_0 int8 tensor-core MMQ over a SoA repack; decode: Q4_0 dp4a/Q8_1, argmax-stable, within ~6% of llama.cpp (57 t/s). **Long context:** `--kv-type bf16` halves / `q8_0` quarters the K/V store (fp32 kernel math, argmax-stable) + a bookkeeping-only host KV cache → **`-c 131072` (128K) within 12 GB** (fp32 `cudaMalloc`-fails at 64K). At 128K, q8_0 keeps the tied embed table resident (~53 t/s) where bf16 spills it (~19 t/s). Default fp32; opt-in `--kv-type bf16\|q8_0` |
| Vulkan `-g -1 -c 2048` | 17.0 | 19.1 | per-token prefill (no batched-prefill path). The 12B is PLE-free; the E4B family's PLE + shared-KV are now supported on Vulkan too (#351) |

#### Qwen3-Coder 30B-A3B (MoE) — [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) · 17 GB

```bash
sharpi-cli -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf -g -1 \
  --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 -p "Implement a binary search tree in C#"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1` (hybrid) | **29.4** | **28.0** | 29 GPU + 19 CPU layers; routed experts stream through `CudaExpertSlotManager` SLRU (#72/#77). Batched-trunk prefill (#123, bit-identical; `SHARPI_BATCHED_PREFILL=0` to bisect); `SHARPI_EXPERT_STATS=path` for hit rates |
| CPU `--tq` | 19.6 | 22.6 | 3-bit KV; FastScan (#34) → 15.5 t/s decode @ 3.2K ctx |
| CPU | 19.8 | 22.4 | 128 experts / 8 active |
| Vulkan `-g -1` (hybrid) | 1.2 | 4.9 | 29 GPU + 19 CPU layers, SLRU expert cache + predictive prefetch (`--no-moe-predict-prefetch` to disable). PCIe/expert-stream bound; short-ctx values |

#### Carnice (Qwen3.6-35B-A3B-MTP finetune) — [mudler](https://huggingface.co/mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF) · 17 GB

```bash
# greedy + --no-thinking engages MTP self-speculative decoding
sharpi-cli -m models/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf -g -1 \
  --temp 0 --no-thinking -p "List three uses for a paperclip"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | **144.3** | **26.5** | agentic finetune; 80% acceptance (`bench-carnice.ps1`). APEX mixed-precision (Q3_K + Q8_0 experts); Q8_KS per-32 int dots auto-enable |

#### Qwen3.6-35B-A3B (GDN+MoE) — [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · 22 GB

```bash
sharpi-cli -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -g -1 \
  --temp 0.6 --top-p 0.95 --top-k 20 -p "Prove that the square root of 2 is irrational"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1` (hybrid) | **67.5** | **23.7** | 10 attn + 30 GDN on GPU; routed MoE on CPU, shared expert GPU-overlapped. Fused GDN scan + batched SDPA, bit-identical. `SHARPI_CPU_MOE=0` forces on-GPU experts |
| CPU | **11.3** | 9.3 | hybrid GDN/attn, 256 experts / 8 active. Chunk-parallel (FlashQLA) GDN prefill default-on (1.35× over the 8.4 t/s per-token scan; `SHARPI_GDN_CHUNKED_PREFILL=0` to disable). MTP variants keep the byte-exact scan (FP-reorder flips the thinking-boundary token) |

#### Qwen3.6-35B-A3B-MTP (GDN+MoE) — [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · 22 GB

```bash
# SHARPI_CPU_MOE=1 keeps GDN/attn + shared expert on GPU, routed experts on CPU
SHARPI_CPU_MOE=1 sharpi-cli -m models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf -g -1 \
  --temp 0 --no-thinking -p "Write a binary search in Rust"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | **67.2** | **21.4** | needs `SHARPI_CPU_MOE=1`: 30 GDN + 10 attn + shared expert on GPU, routed experts CPU mmap. 100% acceptance |
| CPU `--no-thinking` | 9.1 | **8.5** | GDN/attn + 256-expert MoE + MTP head (#44). 100% acceptance; MoE-MTP batched verify (#45) — routed experts sequential per token, so ~MTP-off parity |

#### Qwen3.6-27B-MTP (GDN) — [unsloth](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) · 16 GB (Q4_K_M) / 19 GB (Q5_K_M)

```bash
sharpi-cli -m models/Qwen3.6-27B-MTP-Q4_K_M.gguf -g -1 \
  --temp 0 --no-thinking -p "Explain the CAP theorem"
```

| Backend | Size | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---:|---|
| **CUDA** `-g -1 --no-thinking` (hybrid) | 16 GB | **10.0** | **12.3** | GDN/attn KV resident on GPU, dense FFN on CPU mmap (the k=4 ring reclaims the VRAM the old k=2 default spent on 22 GPU FFN layers). 84% acceptance; 4-input CPU-FFN `MatVec4In` (#209) moves the verify optimum from k=2 → k=4 — **1.9× over MTP-off (6.5)**, +22% over the old k=2 default (10.1) |
| **CUDA** `-g -1 --no-thinking` `Q5_K_M` (hybrid) | 19 GB | 6.2 | **5.5** | 13/64 FFN on GPU, 51/64 CPU mmap. 98% acceptance; batched trunk (#119) bit-identical |
| CPU `--no-thinking` | 16 GB | 3.0 | **3.6** | dense 27B GDN/attn + native MTP head; auto MTP self-spec (#25) at greedy + `--no-thinking`. 90% draft acceptance; folded k-token batched verify (#30/#207) — 1.2× over MTP-off (3.0) |
| CPU `--no-thinking` `Q5_K_M` | 19 GB | 2.8 | **3.5** | ~10% slower than Q4_K_M; 100% acceptance |

#### Llama-4 Scout 17B-16E (MoE) — [meta-llama](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) · 61 GB

```bash
# split GGUF — point -m at shard 1; CPU-only wins on a 12 GB card (drop -g)
sharpi-cli -m models/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf \
  --temp 0.6 --top-p 0.9 -p "Summarize the plot of Hamlet"
```

| Backend | Prefill t/s | Decode t/s | Notes |
|---|---:|---:|---|
| CPU | 2.1 | 4.3 | 48 layers, 17B active; split GGUF (not on bench machine) |
| CUDA `-g -1` (hybrid) | 1.2 | 2.6 | 7 GPU + 41 CPU layers — model dwarfs the 12 GB card so CPU-only wins; per-expert SLRU streaming (#72/#77) still lifts both (not on bench machine) |

_CUDA columns re-measured 2026-06-16 with each model's recommended sampling (`scripts/bench-allrows-1k.ps1
-CudaOnly`): prefill warm at a realistic ~2K-token working context, decode at near-zero ctx (`-NearZero`),
each after a discarded warm-clock warm-up. The CPU and Vulkan rows are from the prior 2026-06 sweep (prefill
~1K ctx), except the Gemma 4 E4B q4_0 Vulkan row, freshly measured 2026-06-22 with the same convention after
the #351 gemma4-on-Vulkan work (PLE + shared-KV + narrowed KV). Vulkan rows remain ~35% below their earlier
numbers — an unexplained regression (CUDA improved on the same box). Llama-4 Scout and Qwen3-Coder
Vulkan-hybrid keep prior values (not re-runnable here). The Qwen3.6 family (Gated-DeltaNet + MoE/MTP:
35B-A3B, 27B-MTP, 35B-A3B-MTP, Carnice) has no Vulkan rows — GDN is not yet implemented on Vulkan (#356;
the MTP self-spec speedup is the #357 follow-up), so those models require `--backend cuda` or `-g 0` (CPU)._

**Long-context decode** uses flash-decoding (split-KV) on all CUDA paths (dense + MoE/GDN hybrids): the
per-token KV read parallelizes across SMs, so decode no longer collapses with context (Gemma 4 E4B q8
11→45 t/s @32K, up to ~2× on the hybrids). `SHARPI_SPLIT_DECODE=0` reverts.

**Sampling for Gemma 4 E4B-it:** `--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0` (Gemma 3/4 defaults).
It is **not** a reasoning model — the CLI auto-sets `enable_thinking=false`, else the template renders an
empty `<think>` block and output degenerates. Greedy is not recommended; use the values above.

`--backend auto` (default) picks CUDA when available, sizing the GPU/CPU split from VRAM via TierPlanner,
else Vulkan. For hybrid `qwen35moe` the CUDA backend keeps attention KV, GDN layers, and the shared expert
in VRAM; routed experts auto-select SLRU GPU cache vs CPU mmap by fit (`SHARPI_CPU_MOE=0|1`). On Ampere+ it
uses bf16 cuBLAS GEMM (`SHARPI_CUDA_PRECISION` to bisect; NVRTC kernels keep fp32 accumulators). GDN paths
store bf16 KV by default (`SHARPI_KV_DTYPE=fp32`).

MoE expert-cache knobs (`--moe-warmpin`, `--moe-warmpin-after`, `--no-moe-predict-prefetch`,
`--expert-stats`) are CLI-only; the server reads the equivalent `SHARPI_MOE_WARMPIN*`,
`SHARPI_MOE_PREDICT_PREFETCH=0`, `SHARPI_EXPERT_STATS=<path>` env vars.

MoE expert *placement* matches llama.cpp: `--cpu-moe` (alias `--cmoe`) keeps **all** routed
experts on the CPU, equivalent to `SHARPI_CPU_MOE=1` (omit for VRAM-fit auto-select;
`SHARPI_CPU_MOE=0` forces the on-GPU SLRU cache). On the server set `SharpInference:CpuMoe`
(`true`/`false`) in `appsettings.json`, or export `SHARPI_CPU_MOE` directly. llama.cpp's
`--n-cpu-moe` (per-layer split) is **not yet supported** — SharpInference's override is
all-or-nothing, so the flag is recognized but errors with that rationale rather than silently
behaving like `--cpu-moe`. The llama.cpp single-dash short forms `-cmoe`/`-ncmoe` aren't
available because Spectre.Console requires single-dash options to be one character; the
double-dash `--cmoe`/`--ncmoe` aliases stand in.

### SnapKV (prefill-time KV eviction, issue #51)

On CPU, CUDA (dense + hybrid GDN), and Vulkan. After prefill it scores each prompt position by softmaxed
attention from the last `W` queries, keeps top-K + a recency window, and compacts the K/V ring in place;
decode is unchanged (`LogicalLength` stays at the prompt length so RoPE on new tokens lands correctly). GPU
paths auto-enable above ~256 MiB KV (budget `min(maxSeqLen/4, 4096)` floored at 1024); `SHARPI_SNAPKV_BUDGET=N`
forces it (`=0` off), `_WINDOW`/`_RECENCY` tune the probe/keep zone (CPU is opt-in via the budget). Composes
with TurboQuant on CPU (~16× total KV). Eval: `benchmarks/SnapKvEval` (needle-in-haystack sweep).

### TurboQuant (`--tq`, 3-bit KV compression)

CPU/Vulkan/CUDA; requires `headDim ∈ {128, 256}`. K-scoring and V-aggregation use a FastScan AVX2 kernel:
KV packs into 32-position tiles with 4-bit codes, a per-query i8 LUT reduces each step to a `vpshufb`, and
the IWHT defers to one call per kv-head. Combined K+V hot-path cost vs the prior per-block path (Ryzen 9 7900X):

| TQ positions | per-block K+V | FastScan K+V | speedup |
|---:|---:|---:|---:|
| 1 024 | 479 µs | 26 µs | 18× |
| 4 096 | 1 931 µs | 98 µs | 20× |
| 8 192 | 3 936 µs | 193 µs | 20× |
| 16 384 | 8 216 µs | 390 µs | 21× |

End-to-end gain tracks the K+V share of token cost — small at short ctx, growing with length. Qwen3-8B CPU
`--tq` decode drops only ~22% from 30→6050 ctx (12.0→9.4 t/s) — ~1.9× the per-block path at 6K.

### Multi-Token Prediction (MTP)

Models with native MTP heads (Qwen3.6-27B-MTP, Qwen3.5/3.6 A3B-MTP, DeepSeek V3/R1) get self-speculative
decoding with no separate draft model. Engages automatically with greedy (`--temp 0`) + `--no-thinking`; the
CLI prints `MTP accept: N%`. Default is a folded k-token batched verify: the certain token plus a chained
draft run through one batched trunk pass, rejections rolled back via a per-token GDN snapshot ring. CLI
mirrors llama.cpp: `--spec-type`, `--spec-draft-n-max <N>` (default 1; deeper chains need
`SHARPI_MTP_BATCH_MAX≥N+1`, ~150 MiB VRAM/slot), `--spec-draft-p-min`. Off: `SHARPI_DISABLE_MTP=1`.

### Chat-continuation cache

Multi-turn requests reuse the prior turn's state instead of re-prefilling. GDN-hybrid passes snapshot
recurrent state at the history boundary and restore on a prefix match; MTP runs also snapshot the MTP KV +
hidden-history, so agentic tool loops skip the per-round re-prefill. `/metrics` exposes
`sharpi_prefill_tokens_reused_total`.

### Reasoning models

Models that emit `<think>...</think>` are detected from their special tokens (no flag); the CLI dims the
stream. `--no-thinking` disables it at the template level, `--hide-thinking` keeps it hidden,
`--max-thinking-tokens N` force-closes runaway reasoning. Greedy often loops — the CLI recommends
`--temp 0.6 --top-p 0.95 --top-k 20`. The server emits reasoning per protocol (Anthropic `thinking`,
OpenAI `reasoning_content`).

### CLI examples

```bash
# CPU, single-turn, greedy
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "What is 2+2?" --temp 0

# Full GPU offload (auto-picks CUDA)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf -p "Write a quicksort in Python" --temp 0 -g -1

# MoE on CPU with 3-bit KV compression
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --tq -p "Implement a BST in C#" --temp 0

# Speculative decoding (~2× faster at temp 0)
dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/Qwen3-8B-Q4_K_M.gguf --draft-model models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  -p "Write a binary search in Rust" --temp 0

# API server (OpenAI /v1/chat/completions + Anthropic /v1/messages, port 5000)
# Multi-user: SHARPI_MAX_BATCH=8 enables continuous batching (CPU or CUDA backend). On CUDA,
# batched decode uses int8 tensor-core matmuls by default at N>=5 (Qwen3-8B Q4_K_M @ 4070 Ti:
# +11% to +28% aggregate t/s from N=5 to N=8); SHARPI_BATCH_DECODE_MMQ=0 forces the bit-exact
# path. Long-prompt admission prefills in SHARPI_PREFILL_CHUNK-token slices (default 256,
# 0 = blocking) interleaved with decode, packed across prompts; SHARPI_KV_BUDGET_MB caps total
# KV memory for admission backpressure (default: half of RAM). See issues #183 / #206.
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server.Host -c Release
```

GPU flags mirror llama.cpp: `-g`/`--ngl`/`--n-gpu-layers` are interchangeable, and `--device <0|CUDA0|none>`
pins a single GPU (no multi-GPU split).

## Image generation

Two pipelines, auto-detected from model filename. Benchmarked on AMD Zen 4 + RTX 4070 Ti (CUDA, 4 steps,
512×512). The CLI is one-shot, so each run pays the full load + encoder warmup; the "cached" column is the
steady-state cost when encoder weights stay resident (server or interactive loop after the first prompt).

| Pipeline | Components (repo • file • size) | Per-run | Cached prompt | Notes |
|---|---|---:|---:|---|
| **Z-Image-Turbo** | DiT: [jayn7/Z-Image-Turbo-GGUF](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF) `z_image_turbo-Q5_K_M.gguf` 5.5 GB<br/>Encoder: [BennyDaBall/...-AbliteratedV1](https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1) `Z-Image-AbliteratedV1.Q5_K_M.gguf` 2.9 GB<br/>VAE + tokenizer: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) `vae/` `tokenizer/` | **~108 s** | **~30 s** | Most per-run cost is text-encoder warmup (~90 s); DiT ~4 s, VAE ~18 s once hot |
| **FLUX.1-schnell** | DiT: [city96/FLUX.1-schnell-gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf) `flux1-schnell-Q4_K_S.gguf` ~7 GB<br/>Encoders + VAE: [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) `clip_l.safetensors` + `t5xxl_fp16.safetensors` + `ae.safetensors` | — | — | 4-step distilled; not on this bench machine |

Optional **4× upscale** via Real-ESRGAN (`RealESRGAN_x4plus.safetensors`): runs on CUDA when available,
falls back to bicubic.

### CLI examples

```bash
# Z-Image-Turbo (auto-detects pipeline from filename containing "z_image")
dotnet run --project src/SharpInference.Cli -c Release -- image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "a serene mountain lake at sunrise" -W 1024 -H 1024 --steps 4 -o landscape.png

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
