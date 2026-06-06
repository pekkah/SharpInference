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
Benchmarked on AMD Zen 4 (12c/24t, DDR4-3200) + RTX 4070 Ti (12 GB), Q4_K_M, `--temp 0`.
**Prefill t/s** is the warm-cache rate at a ~1K-token prompt; **decode t/s** is the near-zero-ctx
generation rate (forward-pass iterations / time, so thinking tokens count). All outputs verified
coherent (`scripts/bench-all.ps1`); top-1 parity vs llama.cpp b8585 verified on Qwen3-8B (byte-identical
60-token greedy decode).

| Model | Repo | Size | Backend | Prefill t/s | Decode t/s | Notes |
|---|---|---:|---|---:|---:|---|
| SmolLM2 1.7B Instruct | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF) | 1 GB | CPU | 40.4 | 38.9 | AVX2 fused dequant-matvec |
| SmolLM2 1.7B Instruct | (same) | 1 GB | Vulkan `-g -1` | 123.2 | **139.7** | GLSL `subgroupAdd` reduce |
| SmolLM2 1.7B Instruct | (same) | 1 GB | **CUDA** `-g -1` | **163.1** | **158.1** | NVRTC `__dp4a` + Q8_1 |
| Qwen3 8B | [Qwen](https://huggingface.co/Qwen/Qwen3-8B-GGUF) | 5 GB | CPU | 9.9 | 11.7 | dense, no KV compression |
| Qwen3 8B | (same) | 5 GB | CPU `--tq` | 9.5 | **11.9** | 3-bit KV → 40 960 ctx; FastScan K+V (#34) keeps long-ctx decode ~flat (10.2 @ 3K, 9.4 @ 6K) |
| Qwen3 8B | (same) | 5 GB | Vulkan `-g -1` | 45.4 | 45.8 | 11.4K auto-ctx |
| Qwen3 8B | (same) | 5 GB | Vulkan `-g -1 --tq` | 40.7 | 45.5 | 3-bit KV → 40 960 ctx |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1` | **432** | **70.0** | Compute-bound Q4_K prefill (#156 Item C) 119.8 → 432 t/s (3.6×, same-session A/B). **C2 (default):** an int8 **tensor-core MMQ** matmul (`llm_mmq_q4k`, `mma.m16n8k32.s8`) reads each Q4_K weight once as int8 — nibble-expanded, `get_scale_min_k4` decode, asymmetric min-bias — with **no fp16 dequant temp to HBM**; `SHARPI_PREFILL_MMQ=0` reverts to **C1**, the dequant→fp16→cuBLAS GEMM (`llm_dequant_q4k_to_f16`); both read the weight once per batch and replace the memory-bound matvec GEMM-N (weight re-streamed per token); both argmax-stable (`SHARPI_PREFILL_GEMM=0` reverts to the bit-exact matvec). MMQ vs C1 same-session A/B: **+25% at ~100-tok prompts** (284 → 355 t/s, where C1 still pays its fp16-temp write), converging to a tie by ~1K ctx as cuBLAS amortizes that temp (430 → 432 @1008) — so the 1K column is unchanged but short-context prefill and prefill VRAM both improve. (llama.cpp b8585 pp1008 5764 t/s — the remaining ~13× is its cp.async-pipelined MMQ, which hides the weight re-read across token tiles cuBLAS amortizes via L2.) Builds on batched-trunk prefill + flash attention (#156 A). Decode CUDA graphs (#158) capture/replay the per-token device region, 65 → 70 t/s (+7%), `SHARPI_CUDA_GRAPH=0` to bisect |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --no-thinking` | **427** | **70.0** | reasoning suppressed in template; same Q4_K prefill GEMM + decode-graph path as the row above |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq` | **57.4** | **58.4** | 3-bit KV → 40 960 ctx; 17 t/s @ 8K, 10 @ 16K |
| Qwen3 8B | (same) | 5 GB | **CUDA** `-g -1 --tq --no-thinking` | **57.5** | **58.1** | as `--tq`, reasoning suppressed |
| OLMoE 1B-7B Instruct (MoE) | [allenai](https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF) | 4 GB | CPU | 51.6 | 55.7 | 64 experts / 8 active; per-channel QK-norm; `norm_topk_prob=false` |
| OLMoE 1B-7B Instruct (MoE) | (same) | 4 GB | Vulkan `-g -1` | 112.3 | **121.2** | greedy unstable across backends — use `--temp 0.6 --top-p 0.95` |
| OLMoE 1B-7B Instruct (MoE) | (same) | 4 GB | **CUDA** `-g -1` | **117.6** | **111.7** | greedy varies, sampling coherent |
| Qwen3-Coder 30B-A3B (MoE) | [Qwen](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 17 GB | CPU | 19.4 | 21.1 | 128 experts / 8 active |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | CPU `--tq` | 18.8 | 21.0 | 3-bit KV; FastScan (#34) → 15.5 t/s decode @ 3.2K ctx |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | Vulkan `-g -1` (hybrid) | 1.1 | 5.3 | 29 GPU + 19 CPU layers, SLRU expert cache; predictive prefetch (#50/#77) on by default (`--no-moe-predict-prefetch`). Prefill is the original short-ctx run (Vulkan-hybrid errored on the ~1K prompt) |
| Qwen3-Coder 30B-A3B (MoE) | (same) | 17 GB | **CUDA** `-g -1` (hybrid) | **30.1** | **25.0** | 29 GPU + 19 CPU layers; routed experts stream through `CudaExpertSlotManager` SLRU (#72/#77). Batched-trunk prefill (#123, bit-identical; `SHARPI_BATCHED_PREFILL=0` to bisect). `SHARPI_EXPERT_STATS=path` for hit rates |
| Llama-4 Scout 17B-16E (MoE) | [meta-llama](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | 61 GB | CPU | 2.1 | 4.3 | 48 layers, 17B active; split GGUF (not on bench machine) |
| Llama-4 Scout 17B-16E (MoE) | (same) | 61 GB | CUDA `-g -1` (hybrid) | 1.2 | 2.6 | 7 GPU + 41 CPU layers — model dwarfs the 12 GB card so CPU-only wins; per-expert SLRU streaming (#72/#77) still lifts both |
| Qwen3.6-35B-A3B (GDN+MoE) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 22 GB | CPU | 8.4 | 8.5 | hybrid GDN/attn, 256 experts / 8 active |
| Qwen3.6-35B-A3B (GDN+MoE) | (same) | 22 GB | **CUDA** `-g -1` (hybrid) | **63.7** | **23.2** | 10 attn + 30 GDN on GPU; MoE auto-routed to CPU, shared expert on GPU overlapped with the routed loop. Fused GDN scan + batched-query SDPA (#114-B/#118), bit-identical, win grows with ctx. Forcing on-GPU experts (`SHARPI_CPU_MOE=0`, non-default) gets the #129 fused MoE-reduce kernel: GPU-SLRU prefill +20% (45.3 → 54.3 t/s) |
| Qwen3.6-27B-MTP (GDN) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | 16 GB | CPU `--no-thinking` | 3.2 | **3.8** | dense 27B GDN/attn + native MTP head; auto MTP self-spec (#25) at greedy + `--no-thinking`. 95% draft acceptance; batched N=2 verify (#30) → 1.4× over MTP-off |
| Qwen3.6-27B-MTP (GDN) | (same) | 16 GB | **CUDA** `-g -1 --no-thinking` (hybrid) | **8.3** | **10.7** | 20/64 dense FFN on GPU + GDN/attn KV resident, 44/64 FFN CPU mmap. 95% acceptance; batched verify → 1.73×. Batched trunk + on-GPU dense-FFN (#119/#121), bit-identical |
| Qwen3.6-27B-MTP (GDN) | (same) | 19 GB | CPU `--no-thinking` `Q5_K_M` | 2.8 | **3.5** | ~10% slower than Q4_K_M; 100% acceptance; batched verify → 1.46× |
| Qwen3.6-27B-MTP (GDN) | (same) | 19 GB | **CUDA** `-g -1 --no-thinking` `Q5_K_M` (hybrid) | 5.4 | **7.9** | 13/64 FFN on GPU, 51/64 CPU mmap. 98% acceptance; batched verify → 1.84×. Batched trunk (#119) bit-identical; FFN batching prefill-neutral here (CPU-mmap bound) |
| Qwen3.6-35B-A3B-MTP (GDN+MoE) | [unsloth](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | 22 GB | CPU `--no-thinking` | 8.5 | 8.0 | GDN/attn + 256-expert MoE + MTP head (#44). 100% acceptance; MoE-MTP batched verify (#45) — routed experts sequential per token, so ~MTP-off parity |
| Qwen3.6-35B-A3B-MTP (GDN+MoE) | (same) | 22 GB | **CUDA** `-g -1 --no-thinking` (hybrid) | **65.0** | **22.9** | Requires `SHARPI_CPU_MOE=1`: 30 GDN + 10 attn + shared expert on GPU, routed experts CPU mmap. 100% acceptance. Fused GDN scan + batched SDPA (#114-B/#118), bit-identical, grows with ctx |
| Carnice (Qwen3.6-35B-A3B-MTP finetune) | [mudler](https://huggingface.co/mudler/Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-GGUF) | 17 GB | **CUDA** `-g -1 --no-thinking` (hybrid) | **43.6** | **25.0** | agentic finetune of 35B-A3B-MTP; 77% acceptance (`bench-carnice.ps1` — the default prompt 1-token-EOSes on this terser tune). APEX mixed-precision (Q3_K + Q8_0 experts); Q8_KS per-32 int dots auto-enable at load (#99/#101/#107), +4.6% decode at ~4× tighter parity vs plain Q8_K (`SHARPI_Q3K_Q8K=0`/`SHARPI_Q8_0_Q8K=0` to disable). Fused GDN scan + wave SDPA (#114-B/#118) bit-identical past 4096 |
| Gemma 4 E4B-it Q8 | [unsloth](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | 8 GB | CPU | 4.9 | 5.0 | dense 42-layer gemma4: per-layer head_dim (256 SWA / 512 global), dual-RoPE, KV-share tail (18 layers), 5:1 SWA:global, logit softcap 30, PLE-256 injection (~4.2 GB mmap-resident) |
| Gemma 4 E4B-it Q8 | (same) | 8 GB | **CUDA** `-g -1 -c 2048` | **3698** | **59** | all 42 layers fit at `-c 2048`. KV-share alias + SWA/global split per layer; PLE projections (~215 MB) upload at construction. **Prefill (#141):** int8 **tensor-core MMQ** matmul (`mma.m16n8k32.s8`, each Q8_0 weight read once as int8 — beats the dequant→fp16→cuBLAS GEMM, drops its fp16 HBM temp; `SHARPI_PREFILL_MMQ=0` reverts) + a **tensor-core flash-attention** prefill (#146/#147): both QK^T and P·V on the mma cores (`mma.m16n8k16.f16`), multi-warp **d-split** so the O tile stays register-resident — replaces the scalar O(n²) per-query attention (which re-streamed each query's K/V window up to ~512×) and is **+27% at ~1K / +40% at 1.8K** over the earlier half2 flash kernel (`SHARPI_PREFILL_FLASH_TC=0` reverts to half2, `=…_FLASH=0` to scalar) + a **SoA Q8_0 weight repack** (#149): all Q8_0 readers (MMQ, dp4a, fp32 matvec, GEMM-N, dequant) read the quants 16-byte-aligned with the fp16 scales split out, killing the `qs` 2-byte-misalignment funnelshift tax — **+10-12% prefill, bit-identical**; `SHARPI_MMQ_SOA=0` reverts) + a batched Q8_0 embedding lookup. **109→3698 at ~1K ctx, →4240 at 1.8K** — profiling showed *attention*, then the matmul inner-loop efficiency, were the dominant prefill costs at realistic prompt lengths. **Decode (#142):** dp4a/Q8_1 int8 matvec (`SHARPI_Q80_DP4A=0` to bisect) + CUDA-graph capture/replay default-on (`SHARPI_CUDA_GRAPH=0` to bisect). All prefill/decode fast paths are argmax-stable vs the fp32 path, not bit-exact (the SoA repack is bit-identical). Remaining gap to llama.cpp (~8475 prefill / ~78 decode): cp.async-pipelined MMQ on the SoA layout + decode matvec work |
| Gemma 4 E4B-it Q8 | (same) | 8 GB | **CUDA** `-g 22 -c 2048` (hybrid) | 6.6 | 6.8 | 22 GPU + 20 CPU layers. `-g ≤ 22` required so the CPU shared-KV tail can read its own-KV source layers; CPU dense-FFN dominates decode (bandwidth-bound). `SHARPI_CUDA_PROFILE=1` for per-phase breakdown |

_Numbers re-measured across every on-disk row at ~1K ctx so the prefill column is comparable; per-issue
before/after figures in the notes are historical. Llama-4 Scout and Qwen3-Coder Vulkan-hybrid keep their
prior values (not re-runnable on the bench machine)._

**Recommended sampling for Gemma 4 E4B-it:** `--temp 1.0 --top-k 64 --top-p 0.95 --min-p 0`
(the Gemma 3/4 family defaults). Gemma 4 E4B-it is **not** a reasoning model, so the CLI now
defaults `enable_thinking=false` for it automatically (no `--no-thinking` needed) — otherwise the chat
template renders a `<think>` block the model wasn't trained to fill and the output degenerates. Greedy
(`--temp 0`) is not recommended for it either; use the sampling values above.

`--backend auto` (default) picks CUDA when available, sizing the GPU/CPU split from VRAM via TierPlanner;
falls through to Vulkan only when CUDA isn't present. For hybrid `qwen35moe` models the CUDA backend keeps
attention KV, the GDN layers, and the shared expert in VRAM; routed-expert dispatch auto-selects between
an SLRU GPU cache and CPU mmap based on how many experts fit at boot (`SHARPI_CPU_MOE=0|1` to override).
On Ampere+ it auto-selects bf16 cuBLAS GEMM (`SHARPI_CUDA_PRECISION=fp32|fp16|bf16|fp8` to bisect; custom
NVRTC kernels keep fp32 accumulators). The GPU KV cache stores bf16 by default on GDN paths
(`SHARPI_KV_DTYPE=fp32` to bisect).

MoE expert-cache knobs (`--moe-warmpin`, `--moe-warmpin-after`, `--no-moe-predict-prefetch`,
`--expert-stats`) are CLI-only; the server reads the equivalent `SHARPI_MOE_WARMPIN*`,
`SHARPI_MOE_PREDICT_PREFETCH=0`, `SHARPI_EXPERT_STATS=<path>` env vars.

### SnapKV (prefill-time KV eviction, issue #51)

Ships on CPU `ForwardPass`, CUDA hybrid GDN, dense CUDA, and Vulkan. After prefill the model scores each
prompt position by softmaxed attention from the last `W` queries, keeps top-K + a trailing recency window,
and compacts the K/V ring in place. Decode is unchanged — `LogicalLength` stays at the original prompt
length so RoPE on new tokens lands correctly.

The GPU paths auto-enable when the full attention KV cache would exceed ~256 MiB; auto-budget is
`min(maxSeqLen/4, 4096)` floored at 1024. `SHARPI_SNAPKV_BUDGET=N` forces a budget (`=0` disables);
`_WINDOW`/`_RECENCY` (default 64) tune the probe and must-keep zone. CPU keeps the explicit-opt-in
convention (set the budget to engage). SnapKV composes with TurboQuant on CPU for ~16× total KV reduction
(#68). Long-context eval: `dotnet run --project benchmarks/SnapKvEval -- --model <gguf>` runs a
needle-in-haystack sweep across budgets.

### TurboQuant (`--tq`, 3-bit KV compression)

CPU/Vulkan/CUDA; requires `headDim ∈ {128, 256}`. K-scoring and V-aggregation use a FastScan AVX2 kernel
(#34): KV positions pack into 32-position tiles with 4-bit codes, a per-query i8 LUT reduces each step to a
`vpshufb`, and the IWHT is deferred to one call per kv-head. Per (layer, kv-head) cost of the combined K+V
hot path vs the prior per-block AVX2 path on a Ryzen 9 7900X:

| TQ positions | per-block K+V | FastScan K+V | speedup |
|---:|---:|---:|---:|
| 1 024 | 479 µs | 26 µs | 18× |
| 4 096 | 1 931 µs | 98 µs | 20× |
| 8 192 | 3 936 µs | 193 µs | 20× |
| 16 384 | 8 216 µs | 390 µs | 21× |

End-to-end gain tracks the K+V share of token cost — small at short context, growing with length. Qwen3-8B
CPU `--tq` decode degrades only ~22% from 30 → 6 050 ctx (12.0 → 9.4 t/s); the per-block path would drop
to ~5 t/s at 6K, so FastScan is ~1.9× decode there.

### Multi-Token Prediction (MTP)

Models with native MTP heads (Qwen3.6-27B-MTP, Qwen3.5/3.6 A3B-MTP, DeepSeek V3/R1) get self-speculative
decoding with no separate draft model. It engages automatically when the pass reports `HasMtpHead`, sampling
is greedy (`--temp 0`), and thinking is off (`--no-thinking`); the CLI prints `MTP accept: N%`. Batched N=2
verify (#30) is the default for dense MTP; MoE MTP also batches the trunk while routed experts run per token
(#45). CLI mirrors llama.cpp: `--spec-type`, `--spec-draft-n-max <1|2>`, `--spec-draft-p-min <0..1>`
(lossy probabilistic accept). `SHARPI_DISABLE_MTP=1` / `SHARPI_DISABLE_BATCH_VERIFY=1` are the off-switches.

### Chat-continuation cache

Multi-turn requests reuse the prior turn's state instead of re-prefilling. GDN-hybrid passes snapshot their
recurrent state at the history boundary (#102) and restore on a prefix match; MTP runs also snapshot the MTP
attention KV + hidden-history (#106), so agentic tool loops skip the per-round-trip re-prefill. `/metrics`
exposes `sharpi_prefill_tokens_reused_total` (verify with `scripts/test-snapshot-reuse.ps1`).

### Reasoning models

Models that emit `<think>...</think>` are detected from their special tokens — no flag. The CLI dims the
reasoning stream. `--no-thinking` disables it at the template level, `--hide-thinking` keeps it on but
hidden, `--max-thinking-tokens N` force-closes runaway reasoning. Greedy on these models often loops, so the
CLI recommends `--temp 0.6 --top-p 0.95 --top-k 20`. The server emits reasoning per protocol convention
(Anthropic `thinking` block; OpenAI `reasoning_content`).

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
SHARPI_MODEL=models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  dotnet run --project src/SharpInference.Server.Host -c Release
```

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
