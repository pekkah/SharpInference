# SharpInference CUDA vs llama.cpp CUDA — 2026-06-26

> **Update 2026-08-05 — issue #407 fixed.** The `!IsNeoxRope` prefill gate below is gone:
> `llm_rope_norm_partial_batched` gives NORM/interleaved-RoPE models a batched RoPE kernel,
> so `llama`-arch models now take the batched trunk. Re-measured on the same box:
> SmolLM2-1.7B prefill **220.2 → 1282.9 t/s** (5.8×) and Rocinante-X-12B (Mistral-Nemo)
> **8.3 → 526.1 t/s** (63×), both byte-identical to the per-token path with decode unchanged.
> The SmolLM2 row in the tables below is the pre-fix measurement and is kept as the record.
> #406 (`_isMoE`) and the `_hasAttnBias` sibling are still open.

Fresh, same-session, same-box head-to-head of SharpInference's CUDA and CUDA-hybrid
forward passes against llama.cpp CUDA. Both engines measured warm, back-to-back, on the
same hardware so the **ratio** is thermally robust (see the Carnice/35B notes on why the
raw decode ratio overstates the sustained gap for the CPU-MoE hybrids).

## Setup

- **Hardware:** RTX 4070 Ti (12 GB) + Ryzen 9 7900X (Zen 4).
- **llama.cpp:** CUDA build `b9529` (`llama-bench`), `-p 2048 -n 128` → `pp2048` / `tg128`.
  - Full offload: `-ngl 99`.
  - MoE hybrid (routed experts on CPU, matches our `--cpu-moe`): `-ngl 99 -ncmoe 99`.
  - Dense-GDN 27B (attn/GDN on GPU, dense FFN on CPU, matches our split): `-ngl 99 -ot ffn=CPU`.
- **SharpInference:** `scripts/bench-allrows-1k.ps1 -CudaOnly` (prefill, warm ~2K-token
  working context) + `-CudaOnly -NearZero` (decode headline, near-zero ctx). Each row gets
  a page-cache warm + a warm-clock GPU warm-up before the measured run. Default kernels
  (`SHARPI_GDN_DECODE_FAST` off, etc.). MTP rows run greedy `--no-thinking` (engages MTP
  self-speculative decode); their decode includes the MTP speedup.
- **Conventions** match the README: prefill = ~2K-ctx warm rate; decode = near-zero-ctx
  headline (≈ llama.cpp `tg128`).
- **Gap** = llama.cpp / SharpInference. `>1` = llama.cpp faster; `<1` = **SharpInference faster**.

## Full GPU offload (`-g -1` / `-ngl 99`)

| Model | sharpi pf | sharpi dec | llama.cpp pf | llama.cpp dec | **Prefill gap** | **Decode gap** |
|---|--:|--:|--:|--:|--:|--:|
| SmolLM2-1.7B Q4_K_M | 193.1 | 274.5 | 20 358 | 331.6 | **105×** | 1.21× |
| VibeThinker-1.5B Q8_0 (Qwen2) | 148.3 | 174.6 | 23 552 | 234.6 | **159×** | 1.34× |
| OLMoE-1B-7B Q4_K_M (MoE) | 110.9 | 127.7 | 17 194 | 434.0 | **155×** | **3.40×** |
| Qwen3-8B Q4_K_M | 2 304.9 | 77.8 | 6 014 | 90.2 | 2.61× | 1.16× |
| Gemma4-E4B q4_0 | 3 601.0 | 98.4 | 9 224 | 124.0 | 2.56× | 1.26× |
| Gemma4-E4B Q8 | 3 949.2 | 70.3 | 8 581 | 77.3 | 2.17× | 1.10× |
| Gemma4-12B q4_0 | 1 726.2 | 54.0 | 4 205 | 57.3 | 2.44× | 1.06× |

## CUDA-hybrid (CPU-MoE / CPU-FFN split)

| Model | split | sharpi pf | sharpi dec | llama.cpp pf | llama.cpp dec | **Prefill gap** | **Decode gap** |
|---|---|--:|--:|--:|--:|--:|--:|
| Qwen3-Coder-30B-A3B Q4_K_M | `-ncmoe 99` | 31.0 | 26.8 | 588 | 34.9 | **19×** | 1.30× |
| Carnice 35B-A3B-MTP-ft | `-ncmoe 99` | 544.5 | 27.6 † | 740 | 60.4 | 1.36× | 2.19× † |
| Qwen3.6-35B-A3B-UD | `-ncmoe 99` | 466.2 | 30.0 | 618 | 50.2 | 1.33× | 1.67× † |
| Qwen3.6-35B-A3B-MTP | `-ncmoe 99` | 482.8 | 32.5 † | 649 | 50.6 | 1.34× | 1.56× † |
| Qwen3.6-27B-MTP Q4_K_M | `-ot ffn=CPU` | 9.9 | 9.9 † | 615 | 4.46 | **62×** | **0.45× (we win 2.2×)** |
| Qwen3.6-27B-MTP Q5_K_M | `-ot ffn=CPU` | 6.2 | 5.3 † | 566 | 3.89 | **91×** | **0.73× (we win 1.4×)** |

† Our decode for these rows is a **sustained** 60-token generation; llama.cpp `tg128` is a
tight isolated loop that stays warmer / L2-resident and so reads ~25–30% high for the CPU-MoE
GDN hybrids (see `project_decode_carnice_cpumoe`: sustained-vs-sustained, our plain decode is
~80% of llama.cpp, i.e. ~1.25×, not the ~1.6–2.2× the raw `tg128` ratio implies). The MTP rows
(Carnice, 35B-MTP, 27B-MTP) additionally include our MTP self-speculative speedup, which
llama-bench does not use. The 27B-MTP **decode win is real** under both effects.

> Not matched to a llama.cpp run: our `gemma4-cuda-hyb` (E4B Q8 `-g 22`, 22 GPU + 20 CPU
> layers) = 7.0 / 7.0 t/s — a partial-offload row with no apples-to-apples llama.cpp config.

## What the gaps mean

### Prefill — three distinct causes
1. **Batched int8-MMQ trunk prefill is gated OFF** for whole model families →
   per-token prefill, **100–160×** slower. Root cause `CudaForwardPass.IsBatchedPrefillSupported()`
   (`src/SharpInference.Engine/CudaForwardPass.cs`, the gate at the top of that method):
   ```csharp
   if (_isMoE || _tqEnabled || _hasAttnBias || !_hp.IsNeoxRope) return false;
   ```
   - `_isMoE` → **OLMoE** (155×). → issue #406
   - `!_hp.IsNeoxRope` (Llama NORM rope) → **SmolLM2** (105×). → issue #407 — **FIXED**, see
     the update note at the top; the clause no longer exists.
   - `_hasAttnBias` (Qwen2 QKV bias) → **VibeThinker** (159×). → new, sibling of #406/#407
   The tell: all three prefill *below* their own decode rate.
2. **CPU-side FFN/expert work isn't batched during prefill** → **19–91×** slower.
   - **27B-MTP** dense FFN runs per-token on CPU during prefill (9.9 t/s) while llama.cpp
     batches it (615 t/s).
   - **Coder-30B** prefill is bottlenecked on SLRU expert streaming (31 t/s); it never got
     the GPU MoE op-offload (#390) the 35B GDN models did.
3. **Residual kernel gap (~2–2.6×)** on the models that *do* batch (Qwen3-8B, Gemma4) —
   llama.cpp's cp.async double-buffered MMQ + fused flash-attn prefill. → issue #409

### Decode
- Dense full-offload: **~1.05–1.35×** (near ceiling). Fine.
- **OLMoE 3.4×** is the one bad decode — on-GPU MoE expert matvec vs llama.cpp's fused
  `mul_mat_id`. → issue #408
- GDN/MoE hybrids: raw ratio 1.6–2.2× but ~1.25× sustained-vs-sustained (the experts are
  RAM-bandwidth-floored, equal on both engines; the gap is our trunk + per-layer
  GPU↔CPU coordination). See `project_decode_carnice_cpumoe`.
- **27B-MTP: we win** (MTP self-spec + faster CPU-FFN decode).

## Biggest differences (ratio, prefill or decode)

| # | Model | Axis | Gap | Cause |
|---|---|---|--:|---|
| 1 | VibeThinker-1.5B | prefill | 159× | prefill gate `_hasAttnBias` (Qwen2) |
| 2 | OLMoE-1B-7B | prefill | 155× | prefill gate `_isMoE` |
| 3 | SmolLM2-1.7B | prefill | 105× | prefill gate `!IsNeoxRope` |
| 4 | Qwen3.6-27B-MTP Q5 | prefill | 91× | CPU dense-FFN not batched in prefill |
| 5 | Qwen3.6-27B-MTP Q4 | prefill | 62× | CPU dense-FFN not batched in prefill |
| 6 | Qwen3-Coder-30B | prefill | 19× | SLRU expert-stream-bound prefill (no op-offload) |
| 7 | OLMoE-1B-7B | decode | 3.4× | on-GPU MoE expert matvec |

The earlier-filed top-4 (#406/#407/#408/#409) still hold; this fresh full run adds
VibeThinker (a third prefill-gate instance) and the 27B-MTP / Coder CPU-prefill-batching gap.

## Tracking issues
- **#405** umbrella · **#406** MoE prefill-gate · ~~**#407** NORM-rope prefill-gate~~ (fixed) ·
  **#408** OLMoE MoE decode · **#409** dense residual prefill gap.

## Raw data
- SharpInference: `tools/bench/allrows-1k.csv` (prefill), `tools/bench/allrows-nz.csv` (decode).
- llama.cpp: `llama-bench b9529 -p 2048 -n 128`, configs above. Reproduce with
  `scripts/bench-allrows-1k.ps1 -CudaOnly [-NearZero]` (ours) and the per-row `llama-bench`
  invocations in the Setup section.
