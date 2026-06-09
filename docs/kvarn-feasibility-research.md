# KVarN KV-Cache Quantization: Feasibility for SharpInference

> Research note, 2026-06-09. Evaluates whether to implement **KVarN**
> (Müller et al., Huawei CSL, arXiv:2606.03458, 2 Jun 2026) in SharpInference,
> audits how it maps onto our existing TurboQuant KV-cache machinery, and gives
> an effort estimate and a recommendation.
>
> Sources:
> - Paper: *KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error
>   Accumulation in Reasoning Tasks* — <https://arxiv.org/abs/2606.03458>
>   (HF: <https://hf.co/papers/2606.03458>)
> - Reference impl (vLLM fork, Apache-2.0): <https://github.com/huawei-csl/KVarN>

---

## 1. TL;DR

**Recommendation: worth a scoped CPU prototype, _not_ a from-scratch parallel
subsystem.** KVarN's core idea sits one step beyond what `SharpInference.TurboQuant`
already does, and ~70% of the supporting infrastructure (Hadamard rotation,
tile-staged hybrid FP32+compressed cache, SnapKV eviction, per-head/per-layer
fused dequant-dot attention) is already in the tree. The genuinely new piece is a
small, well-defined transform (**dual-axis "Sinkhorn" variance normalization**)
plus a switch from Lloyd-Max codebooks to **asymmetric round-to-nearest (RTN)**
with per-channel-key / per-token-value scales.

The catch is the same one we hit with TurboQuant: the headline *"throughput above
FP16"* number only materializes with good fused GPU kernels. KVarN ships **Triton
only** (a vLLM fork) — none of it is portable to our CPU/AVX2 + Vulkan + CUDA
stack, so the GPU kernel work is on us and is the bulk of the cost.

**Verdict:** Prototype the quantizer on CPU first (reusing TurboQuant's Hadamard +
cache staging), validate the 2-bit accuracy claim on a reasoning benchmark, and
only then decide whether to fund the CUDA/Vulkan kernels. Frame it as an
*alternative quantizer inside the existing TurboQuant cache*, not a second
KV-quant stack to maintain.

---

## 2. What KVarN actually is

KVarN is a **calibration-free** KV-cache quantizer. The paper's framing: prior
KV-quant methods (KIVI, TurboQuant, etc.) are evaluated in prefill-like settings,
but under long **autoregressive decoding** quantization errors *accumulate across
timesteps*, driven mostly by **incorrect per-token scales**. KVarN fixes the
token-scale outliers and so reduces drift over long generations — which is why it
targets reasoning / test-time-scaling / agentic workloads specifically.

### 2.1 The pipeline (per 128-token tile)

A tile = one vLLM block = 128 tokens. For each tile of K (and separately V):

1. **Hadamard rotation** along the channel/head dimension — orthonormal mixing to
   spread outlier channels. *Attention scores are preserved* because the rotation
   is orthonormal (Qᵀ(HK) = (HQ)ᵀK).
2. **Iterative variance normalization (the novel part).** A Sinkhorn-like
   alternation, in **log space**, of column-wise (per-channel) and row-wise
   (per-token) standard-deviation normalization. A few iterations drive both axes
   toward unit variance, equalizing the dynamic range a fixed-width grid has to
   cover and killing the per-token scale outliers that cause drift.
3. **Asymmetric quantization (RTN).** Low-bit round-to-nearest with a zero-point;
   **keys quantized per-channel**, **values per-token**, group size 128.
4. **Scale folding at read time.** The normalization/quant scales are reapplied
   when the cache is read back for attention.

### 2.2 Shipped configuration & results

- Preset `kvarn_k4v2_g128`: **4-bit keys, 2-bit values, group 128**, compute in
  fp16. Keys get more bits because they matter more for attention error
  (consistent with KVTuner/KIVI findings).
- Qwen3-32B (AIME25, 16K, TP=2): ~**4× KV capacity**, throughput **≥ FP16**,
  accuracy at FP16 parity.
- GLM-4.7-Flash (MLA): **2.77×** capacity vs bf16, **0.94×** throughput, accuracy
  maintained. Claims the first sub-8-bit KV-quant compatible with **MLA**
  (Multi-head Latent Attention) models.
- New SOTA at **2-bit** on MATH500 / AIME24 / HumanEval. Reported example:
  Qwen3-4B MATH500 79.2% (vs KIVI 77.8%), AIME24 55.5%→60.0%.

### 2.3 How it differs from what we have (TurboQuant)

| Aspect | TurboQuant (ours, shipping) | KVarN |
|---|---|---|
| Outlier handling | Walsh-Hadamard + per-head sign flip | Hadamard rotation |
| Scale correction | norm per block, Lloyd-Max codebook | **dual-axis Sinkhorn variance norm** |
| Quantizer | Lloyd-Max codebooks (3–4 bit), offline-generated, hardcoded for dim 128/256 | **asymmetric RTN**, per-channel K / per-token V, calibration-free |
| Attention | fused dequant-dot via FastScan i8 LUT (PSHUFB) | dequant + standard matmul, scales folded |
| Bit floor | 3–4 bit | **2-bit values** (its headline) |
| Kernels | CPU AVX2 + Vulkan + CUDA (ours) | Triton only (vLLM) |

The **Hadamard step is shared in spirit** — we already have
`WalshHadamard.Transform` (self-inverse, AVX2). KVarN's distinctive contribution
over TurboQuant is step 2 (Sinkhorn variance normalization) and the
calibration-free RTN that reaches 2-bit. It is explicitly positioned as a *Pareto
improvement over TurboQuant*, so this is a natural "next gen" for our existing
subsystem rather than an unrelated technique.

---

## 3. How it maps onto SharpInference

Reusable today (file references):

- **Hadamard rotation** — `src/SharpInference.TurboQuant/WalshHadamard.cs`
  (`Transform`, normalized self-inverse, AVX2). Directly usable for step 1.
- **Tile-staged hybrid cache** — `src/SharpInference.Engine/TurboQuantKvCache.cs`
  already buffers an FP32 ring window (recent tokens) and *promotes* older tokens
  into compressed tiles (`Append`, the `_tqKeyStaging`/`_tqValueStaging` →
  `_tqKeyTiles`/`_tqValueTiles` flow). KVarN's per-tile, dual-axis normalization
  **requires the whole 128-token tile assembled before it can quantize** — and we
  already have exactly that staging pattern (today at 32-position tiles).
- **Compressor wrapper shape** — `src/SharpInference.TurboQuant/KvCacheCompressor.cs`
  (`Compress`/`Decompress`/`RotateQuery`/`DequantDot`) is the API template a
  `KVarNCompressor` would mirror.
- **Attention dispatch** — `ForwardPass.TqAttention` (CPU, ~`ForwardPass.cs:1606`)
  already has the query-rotate → K-score → softmax → V-aggregate shape, split
  across a compressed region and an FP32 window. A KVarN path slots in as a
  sibling branch.
- **SnapKV eviction** — `TurboQuantKvCache.Compact` / `PagedKvCache.Compact`
  re-quantize on eviction; KVarN would reuse the same hook.
- **Backend abstraction** — `IComputeBackend` / `IForwardPass` give us the
  CPU/Vulkan/CUDA dispatch seam.

What is genuinely new work:

- **Sinkhorn variance normalization** — small and self-contained: per-tile column
  σ and row σ, log-space alternation, ~3–5 iterations. No calibration data, no
  offline codebook generation (a real operational simplification over TurboQuant's
  hardcoded Lloyd-Max tables, and it generalizes to *any* head dim, not just
  128/256).
- **Asymmetric RTN with per-channel-K / per-token-V scales + zero-points** —
  simpler math than Lloyd-Max, but a *different* packed layout than our current
  codebook-index FastScan tiles, so the tile format and the scoring kernel are new.
- **2-bit value packing** — we currently do 3–4 bit; 2-bit packing + the
  per-token V scale fold is new.
- **Fused score/aggregate kernels** — to beat FP16 we'd need the dequant-dot fused
  for KVarN's RTN+scale layout on **CPU (AVX2), CUDA, and Vulkan**. KVarN's Triton
  is reference-only. This is the dominant cost.

---

## 4. Effort estimate

| Phase | Scope | Effort |
|---|---|---|
| **P0 — CPU reference + accuracy validation** | `KVarNCompressor` (Hadamard reuse + Sinkhorn norm + asymmetric RTN K4V2, group 128); plug into a TurboQuant-style hybrid cache; scalar dequant-dot path; wire a `TqAttention`-sibling branch behind an env flag. Validate 2-bit accuracy vs FP32 and vs our TurboQuant 3–4 bit on MATH500/GSM8K-style runs. | **~1.5–2.5 wks**. Mostly the quantizer + harness; correctness-first, perf-second. |
| **P1 — CPU AVX2 fused kernels** | i8/LUT-free score + aggregate kernels for the RTN+scale layout (analogous to `FastScan`); 2-bit unpack. | **~1–2 wks** |
| **P2 — CUDA kernels** | NVRTC quantize / dequant-dot / V-aggregate for K4V2; integrate with `GpuBufferPool` and the existing TQ GPU cache state in `CudaForwardPass`. This is where "throughput ≥ FP16" is won or lost. | **~3–4 wks** |
| **P3 — Vulkan kernels (optional)** | SPIR-V ports of the above. | **~2–3 wks** |

A credible **go/no-go milestone is end of P0**: a few days of validation tells us
whether KVarN's 2-bit accuracy claim holds on the models we actually serve. If it
does, P2 is the high-value follow-on; P3 is optional and can lag.

---

## 5. Is it worth it?

**Arguments for:**

- The win is real and *on-target*. 2-bit KV at FP16 accuracy → 3–5× context is
  exactly what SharpInference's long-context / agentic / reasoning users want, and
  we've already signaled this is a priority by investing in TurboQuant + SnapKV.
- **High infrastructure leverage.** Hadamard, tile staging, hybrid FP32 window,
  eviction, and the attention dispatch seam already exist. KVarN is closer to a
  TurboQuant *upgrade* than a new subsystem.
- **Calibration-free is an operational win.** TurboQuant currently ships hardcoded
  Lloyd-Max codebooks for dims 128/256 (`TurboQuantCodebooks.cs`). KVarN needs no
  offline codebooks and works at any head dim — less to maintain, broader model
  coverage.
- It is explicitly a Pareto improvement over our current method, with peer-paper
  evidence at 2-bit.

**Arguments against / risks:**

- **GPU kernel cost dominates.** Without P2, we get capacity but a *slower* path
  than FP16 — the opposite of the paper's selling point. The capacity-only win is
  still useful for VRAM-bound single-user desktop (our stated target), but the
  marquee throughput claim needs real kernel work.
- **Two KV-quant systems risk.** Done carelessly this doubles maintenance. Mitigate
  by building KVarN *inside* the TurboQuant cache machinery as a selectable
  quantizer, and treat it as a candidate to eventually **supersede** TurboQuant
  rather than run alongside it indefinitely.
- **Newness.** Paper is days old (2 Jun 2026); results are on specific models
  (Qwen3, GLM-MLA). Independent replication is thin — P0 validation is the hedge.
- **MLA support is largely irrelevant to us today** — it matters for
  DeepSeek/GLM-style latent attention, which SharpInference doesn't target. Don't
  let the MLA headline inflate the perceived value for our use case.
- **License/porting.** The repo is Apache-2.0 but it's a vLLM fork with Triton
  kernels and CUDA C++ that is overwhelmingly inherited vLLM code; it's an
  algorithm reference, not a code source. Implement clean-room from the paper.

---

## 6. Recommendation

1. **Greenlight P0 only.** Build `KVarNCompressor` reusing `WalshHadamard`, add the
   Sinkhorn variance normalization and asymmetric RTN (K4V2, g128), and validate
   2-bit accuracy against FP32 and our existing TurboQuant on a reasoning
   benchmark. Gate it behind an env flag like the existing TQ path.
2. **Decide at the P0 gate.** If 2-bit accuracy holds on models we serve, fund
   **P2 (CUDA)** as the primary perf path; P1 (AVX2) for the CPU story; P3 (Vulkan)
   last.
3. **Build it as a quantizer option inside `TurboQuantKvCache`/`TqAttention`,** not
   a separate cache type — explicitly positioning KVarN to *replace* TurboQuant's
   codebook path if it wins, to avoid carrying two systems.

Bottom line: the algorithm is implementable and unusually well-aligned with code
we already own; the open question is purely empirical (does 2-bit hold for our
models?) and engineering throughput (CUDA kernels). A ~2-week prototype answers
the first cheaply before we commit to the expensive part.
