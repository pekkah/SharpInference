# MoE & Expert Offloading: State of the Art vs. SharpInference

> Research note, 2026-05. Surveys the current literature on Mixture-of-Experts
> (MoE) inference and expert offloading, audits what SharpInference does today
> (with file references), and identifies concrete gaps and opportunities tailored
> to our target deployment (single-user desktop, ~12 GB VRAM e.g. RTX 4070 Ti,
> NativeAOT, GGUF).

---

## 1. TL;DR

SharpInference already has a **solid offloading skeleton** that maps onto several
SOTA ideas — an SLRU expert cache, an async prefetcher, CPU-fallback compute on
cache miss (the core "Fiddler" trick), and per-expert access profiling. That
puts us ahead of naive "swap on demand" systems.

The implementation is uneven across paths. The audit found:

1. **Per-expert SLRU offloading exists on two of three hybrid paths, but not the
   third.** The **GDN CUDA path** (`CudaHybridGdnForwardPass`, used for
   qwen35moe) and the **Vulkan path** (`HybridForwardPass`) both stream experts
   through the SLRU cache (`CudaExpertSlotManager.GetOrLoad` /
   `ExpertSlotManager.TryGetCached`) with CPU-fallback compute on miss. But the
   **non-GDN CUDA hybrid path** (`CudaHybridForwardPass`, used for Mixtral /
   Qwen3-30B-A3B / Qwen3-Coder when they don't fit VRAM) does **whole-layer**
   offload only: every expert of a GPU-tier layer is uploaded resident, and its
   `_expertSlotManager`/`_prefetcher` fields are declared but never assigned (dead
   code). So for the big non-GDN MoEs, a "GPU layer" must hold its *entire* expert
   set in VRAM — there is no per-expert streaming, only the coarse CPU-layer /
   GPU-layer split that `TierPlanner` decides.
2. **Cached experts are quantized — except Q5_K in two spots.** Good news first:
   experts are cached in native quant (Q4_K/Q6_K everywhere; Q5_K too on
   `CudaExpertSlotManager`), so we are *not* generally paying an F32 premium. But
   the **Vulkan `ExpertSlotManager`** and the non-GDN CUDA resident path
   **dequantize Q5_K to F32** (`ExpertSlotManager.cs:156`,
   `CudaHybridForwardPass.cs:1428`). qwen35moe stores `ffn_down_exps` as Q5_K, so
   on Vulkan every cached down-projection expert is 4 B/element — 4× its source.
   `CudaExpertSlotManager` already keeps Q5_K raw (`UploadRaw`); the other two
   paths just need to mirror it.
3. **Prefetching is reactive, not predictive.** The Vulkan path re-enqueues the
   experts the router *just* selected, betting the next token reuses them at the
   same layer (1-token, same-layer temporal locality). Every SOTA system instead
   predicts the *next layer's* or *next token's* experts ahead of time. *(Already
   tracked as issue #50 — pre-gated / PreScope-style predictive prefetch.)*
4. **Caching is recency-only; the profiler is diagnostic-only.** The SLRU evicts
   by recency. `ExpertAccessProfiler` tracks per-expert hit/miss and prints stats
   (`CudaHybridGdnForwardPass` dump), but nothing feeds hotness back into eviction
   priority or warm-pins hot experts at load. `TierPlanner` places layers by
   footprint, not access frequency.

The highest-leverage, lowest-risk wins for our use case are: **(a)** bring the
non-GDN CUDA hybrid path to per-expert SLRU parity with the GDN path (so big
non-GDN MoEs fit in less VRAM), **(b)** stop dequantizing Q5_K experts on the
Vulkan/resident paths, **(c)** add next-layer expert *prediction* to drive the
prefetcher (#50), **(d)** make eviction/placement activation-aware using the
profiler we already built, and **(e)** a fast CPU expert GEMM (KTransformers-style,
related to #54). Details and priorities in §5.

---

## 2. What SharpInference does today

Verified against the source on this branch.

### 2.1 MoE model support
- Architecture/hparam detection in `src/SharpInference.Core/ModelGraph.cs` and
  `ModelHyperparams.cs` (`IsMoE`, `NumExperts`, `NumActiveExperts`,
  `ExpertIntermediateDim`, `HasSharedExpert`, `NormalizeMoeTopKWeights`,
  `UseSigmoidGating`).
- Covers Mixtral (top-2), Qwen3-MoE / qwen35moe (256 experts, top-8, shared
  expert, GDN-hybrid), OLMoE (top-1), DeepSeek-V2 family, Llama4-style MoE.
- GGUF stores experts as packed per-layer tensors
  (`blk.{L}.ffn_{gate,up,down}_exps.weight`) plus optional shared-expert and
  router (`ffn_gate_inp`) tensors; loaded zero-copy via mmap.

### 2.2 Forward-pass MoE
- **CPU** (`ForwardPass.cs`): router GEMV → softmax/sigmoid → `SelectTopK` →
  optional shared expert → sparse routed experts via pointer-sliced mmap weights,
  SIMD `MatVec`. Solid, correct, the reference path.
- **Vulkan hybrid** (`HybridForwardPass.GpuMoeFfn`, ~line 1293): router on GPU,
  top-k on CPU, then **per-selected-expert SLRU cache lookup**
  (`_expertSlotManager.TryGetCached`). **Cache miss → compute that expert on the
  CPU while the GPU is idle** (`GpuMoeFfnCpuFallback`), accumulate, upload, GPU
  `AddInPlace`. This is the Fiddler idea (compute on CPU rather than block on a
  transfer) and is genuinely good.
- **CUDA GDN hybrid** (`CudaHybridGdnForwardPass.cs`, for qwen35moe): the most
  developed path. Experts served by `CudaExpertSlotManager` SLRU
  (`GetOrLoad`, line ~2257), keeps Q4_K/Q5_K/Q6_K quantized, has a CPU-MoE mode
  (`SHARPI_CPU_MOE=1`), and dumps `ExpertAccessProfiler` stats on dispose.
- **CUDA non-GDN hybrid** (`CudaHybridForwardPass.cs`, for Mixtral / Qwen3-MoE /
  Qwen3-Coder too big for VRAM): GPU-tier layers upload **all** experts to VRAM as
  `Tensor[][] _gpuWGateExps/...` (line ~297) and index them directly (line ~1348);
  CPU-tier layers compute on CPU (`CpuMoeFfn`). Offload granularity is the whole
  layer (`TierPlanner` split) — there is **no per-expert SLRU streaming** here. The
  `_expertSlotManager`/`_prefetcher` fields (lines 108–109) are declared and
  disposed but **never assigned** → the dynamic cache path is dead on this path.
  Experts are kept in native quant for Q4_K/Q6_K (line ~1418) but Q5_K is
  dequantized to F32 (line ~1428).

### 2.3 Offloading infrastructure (`SharpInference.Pipeline` + Engine)
- `SlruCache<K,V>` — segmented LRU, 25% probationary / 75% protected, evicts
  probationary tail. `ExpertCache<T>` wraps it keyed by `(layer, expertId)`.
- `ExpertSlotManager` / `CudaExpertSlotManager` — VRAM expert slot cache;
  `TryGetCached`/`GetOrLoad`, `Preload`, eviction callback frees GPU tensors.
  Keeps experts in native quant — **except** the Vulkan `ExpertSlotManager`
  dequantizes Q5_K (and exotic dtypes) to F32 (`ExpertSlotManager.cs:156`), while
  `CudaExpertSlotManager` keeps Q4_K/Q5_K/Q6_K all raw (`UploadRaw`, line ~162).
- `MoEPrefetcher` — bounded channel + background worker calling
  `slotManager.Preload`. Drops oldest when full. Wired **only** in the Vulkan
  path, and only with `EnqueuePrefetch(layer, selectedExperts)` — i.e. the
  experts already selected for the current layer/token.
- `ExpertAccessProfiler` — per-`(layer,expert)` hit/miss counters, `OverallHitRate`,
  `GetTopExperts`. Diagnostic only; not consumed by placement or eviction.
- `TierPlanner` — greedy layer placement by **footprint** + KV budget. Not
  access-aware.
- `MemoryHierarchy` — 3-tier (VRAM → pinned RAM → NVMe) design; L3/NVMe +
  io_uring is a stub (`NotImplementedYet`).

---

## 3. State of the art (2024–2026)

Expert offloading exists because MoE activates only k-of-N experts per token, so
most expert weights can live in slow memory (host RAM / SSD / CPU) and only the
active few need to be in fast memory (VRAM). The whole game is **hiding the cost
of getting the right experts into fast memory in time**, or avoiding the move
entirely. The literature attacks this along six axes.

### Axis A — Static placement / partitioning
Decide once, offline, what lives where.
- **KTransformers** (SOSP'25) — partition by *arithmetic intensity*: attention
  and frequently-used experts on GPU, the rest computed on CPU with
  highly-optimized kernels (AMX / AVX-512, llamafile-style sgemm). Reports
  1.25–1.93× over llama.cpp, much more on quantized models; runs DeepSeek-R1/V3
  (671B) on a single 24 GB GPU + big DRAM. Key lesson: **CPU expert compute is a
  first-class path, not just a fallback** — with good kernels you don't move the
  weights at all.
- **llama.cpp** `--cpu-moe` / `--n-cpu-moe` / `-ot "exps=CPU"` — the practical
  baseline: keep attention + shared experts (always active) on GPU, routed
  experts on CPU. This is essentially what our CUDA path does statically.
- **Local Routing Consistency** ("Not All Models Suit Expert Offloading", 2505.16056)
  — *which* models even benefit from caching. Metrics SRP and SCH over 20 MoE
  LLMs: models that put MoE on **every** layer and use **no shared expert** have
  the highest locality (best cache hit rates); shared experts and dense-then-MoE
  layouts hurt locality. Most models do well with a cache ≈ **2× the active
  expert count**. Directly relevant to choosing cache sizes per model.

### Axis B — Caching policy
What to keep resident and what to evict.
- LRU/LFU/SLRU (what we have) vs. **activation-aware** caches.
- **MoE-Infinity** (2401.14361) — sequence-level activation *tracing* to capture
  temporal locality, then prioritize caching experts by predicted activation
  ratio; 4–20× latency reduction vs. baselines.
- **HybriMoE** (2504.05897) — *score-based* caching + dynamic intra-layer
  CPU/GPU scheduling, built on KTransformers; handles expert-activation
  instability across tokens.

### Axis C — Predictive prefetching (the big one)
Move experts *before* you need them, overlapping I/O with compute.
- **Pre-gated MoE** (ISCA'24) — add a "pre-gate" so layer L computes layer L+1's
  expert selection, giving a full layer of prefetch lead time. Algorithm+system
  co-design.
- **Cross-Layer Gate / "Fate"** (2502.12224) — predict future-layer experts from
  *current* layer's gate inputs; offloading system with prefetch + caching +
  quantization, tuned for edge/memory budgets.
- **ProMoE** (2410.22134) — proactive caching that predicts and preloads expert
  usage to cut cache misses, separating prefill/decode behavior.
- **AdapMoE** (2408.10284), **fMoE** (2502.05370), **ExpertFlow** (2410.17954) —
  sensitivity-based gating, fine-grained prefetch+cache, and predictive
  routing-path offload with token reordering (up to 93% VRAM savings, 2–10×).

### Axis D — Speculation-driven offloading
Use a draft/speculative process to predict experts many tokens ahead.
- **MoE-SpeQ** (2511.14102) — small on-device draft model predicts the *sequence*
  of experts for future tokens; a runtime orchestrator prefetches them from host
  memory to overlap I/O with compute. Introduces an "Amortization Roofline Model"
  to tune the speculation window for throughput.
- **SP-MoE** (2510.10302), **MoE-SpAc** (2603.09983) — speculative decoding +
  prefetch co-design; speculation doubles as a memory-management signal.
- **OD-MoE** reports up to **99.94%** expert-activation prediction with shadow
  networks; single-layer lookahead alone gives ~84–91%.

### Axis E — Mixed-precision / compression of experts
Don't pay full precision for cold experts.
- **HOBBIT** (2411.01433) — mixed-precision expert offloading: load *less
  important* experts at lower precision (cheaper transfer + less VRAM), critical
  experts at full precision; token-/layer-/sequence-level prefetch + caching.
  Built on llama.cpp; significant speedups with negligible quality loss. **Most
  directly relevant to our F32-cache problem.**
- **PreMoe** (2505.17639) — probabilistic expert *pruning* + task-adaptive
  retrieval to fit big MoEs in constrained memory.

### Axis F — Cache-miss tolerance & batching
- **BuddyMoE** (2511.10054) — on a cache miss, substitute a *redundant/similar*
  expert already resident rather than stalling, exploiting expert redundancy.
- **ExpertFlow** token reordering / expert buffering ("Towards MoE Deployment",
  2303.06182) — reorder tokens so a batch activates fewer distinct experts.
  (Most relevant once we have continuous batching for MoE, which we don't yet.)

---

## 4. Gap analysis

| SOTA capability | SharpInference today | Gap |
|---|---|---|
| Per-expert offloading across all backends | Vulkan + CUDA-GDN: per-expert SLRU. CUDA non-GDN: whole-layer offload, all experts resident. | Non-GDN CUDA MoE (Mixtral/Qwen3-30B-A3B/Coder) can't stream experts → must fit a layer's full expert set in VRAM. |
| Mixed-precision / quantized cache (HOBBIT) | Experts cached in native quant — **except Q5_K → F32** on Vulkan SLRU + non-GDN CUDA resident path | Q5_K (qwen35moe `ffn_down_exps`) costs 4 B/elem on Vulkan. No *down*-quantization of cold experts (true HOBBIT). |
| Activation-aware caching (MoE-Infinity, HybriMoE) | SLRU (recency only) + diagnostic-only `ExpertAccessProfiler` | Profiler doesn't drive eviction/placement or warm-pin hot experts. SLRU ≠ frequency-aware. |
| **Predictive prefetch** (Pre-gated, Cross-Layer Gate, ProMoE) | Reactive 1-token, **same-layer** re-enqueue (Vulkan) | No next-layer/next-token prediction. *Tracked: #50.* |
| Cache-miss CPU compute (Fiddler) | CPU-fallback compute on Vulkan + (via `SHARPI_CPU_MOE`) GDN paths | *Per-dispatch* CPU/GPU decision tracked as #54. |
| Speculative expert prefetch (MoE-SpeQ, SP-MoE) | speculative *decoding* (MTP) exists, not used for expert prediction | Not started; natural extension of #50 once draft accepts are available. |
| Fast CPU expert GEMM (KTransformers AMX) | SIMD `MatVec` (GEMV) CPU fallback | No AMX / blocked sgemm; CPU treated as fallback, not a peer compute tier (related to #54). |
| Cache-miss substitution (BuddyMoE) | block on CPU compute instead | Fine for single-user; no redundancy reuse. |
| Model-aware cache sizing (Local Routing Consistency) | fixed slot capacity | No per-model locality measurement; could size cache ≈2× active experts, skip caching for low-locality models. |
| L3 NVMe tier / io_uring | designed, stubbed | Not shipped; only matters for models > host RAM. |
| Batched MoE + token reorder (ExpertFlow) | MoE batching disabled | Out of scope for single-user target. |

---

## 5. Recommendations (prioritized for our target)

Ordered by leverage/effort for the single-user desktop, ~12 GB VRAM, GGUF case.
Most of these reuse infrastructure we already have.

### P0 — Bring the non-GDN CUDA hybrid path to per-expert SLRU parity
Wire `CudaExpertSlotManager` + `MoEPrefetcher` + CPU-fallback into
`CudaHybridForwardPass` (the fields are already there, just unassigned) so it
streams experts per-token like the GDN path (`CudaHybridGdnForwardPass`) and the
Vulkan path — instead of forcing every expert of a GPU-tier layer to be resident.
Today a non-GDN MoE bigger than VRAM (Mixtral, Qwen3-30B-A3B, Qwen3-Coder-30B) can
only offload at whole-layer granularity, wasting VRAM on cold experts in the
GPU-tier layers. The GDN path is the reference implementation to mirror.
*Risk:* medium — but the exact pattern already exists in-repo to copy.

### P1 — Stop dequantizing Q5_K experts (Vulkan + non-GDN CUDA resident path)
`CudaExpertSlotManager` already keeps Q5_K raw via `UploadRaw`; mirror that in the
Vulkan `ExpertSlotManager` (`ExpertSlotManager.cs:156`) and the
`CudaHybridForwardPass` resident upload (`:1428`). qwen35moe's `ffn_down_exps` is
Q5_K, so this 4×'s the cached down-proj footprint on Vulkan today. Cheap, local,
high-certainty.
*Risk:* low — Q5_K dequant-in-matmul kernels already exist.

### P1 — Predictive prefetch (Cross-Layer Gate / Pre-gated) — *issue #50*
Already filed. Replace the same-layer 1-token re-enqueue with **next-layer
prediction**: run the *next* MoE layer's router on the *current* hidden state (an
`embDim×numExperts` GEMV — tiny) to prefetch a layer ahead. ~84–91% accuracy for
single-layer lookahead is plenty to convert blocking misses into overlapped loads.
Biggest latency win in the offloaded regime; purely a prefetch hint.

### P1 — Feed the profiler into eviction & warm-pinning
We built `ExpertAccessProfiler` but only print it. Use it two ways: **(1)** at
load, warm the cache / pin the top-N experts per layer (KTransformers/MoE-Infinity
"hot experts on GPU"); **(2)** bias SLRU so high-frequency experts resist eviction
(frequency-aware, not pure recency).
*Risk:* low.

### P2 — Treat CPU as a compute peer (KTransformers-style) — *relates to #54*
Add a blocked/multi-threaded expert GEMM (and explore AVX-512/AMX where available)
so CPU-resident experts are computed cheaply rather than something we always try
to avoid by moving to GPU. Pairs naturally with the CPU-fallback dispatch policy
in #54 — it makes "compute on CPU" the *intended* steady state for cold experts,
à la KTransformers, rather than a stall.
*Risk:* medium (kernel work), high payoff for big-model offload.

### P2 — Model-aware cache policy (Local Routing Consistency)
Measure/lookup per-model routing locality and size the expert cache accordingly
(≈2× active experts is a good default), and detect low-locality models (shared
expert + dense-then-MoE) where caching helps little — pin shared experts, don't
over-invest cache slots on routed ones.
*Risk:* low.

### P3 — Speculative expert prefetch (MoE-SpeQ) — *extends #50*
We already have speculative decoding (MTP). Reuse its drafted tokens to prefetch
the experts those tokens will need, several tokens ahead — a natural extension of
the #50 predictor. Big win only in the heavily-offloaded regime; defer until the
P0/P1 items land.

### Out of scope for now
L3/NVMe tier (only matters when model > host RAM), continuous-batching MoE +
token reordering (single-user target), GDN GPU kernels (orthogonal to offloading).

---

## 6. Bottom line

We're not missing the *concept* of expert offloading — the bones are good, the
GDN CUDA path is genuinely SOTA-aligned (per-expert SLRU, CPU fallback, quantized
cache, profiling), and predictive prefetch (#50) and Fiddler dispatch (#54) are
already on the board. What's missing is *parity and follow-through*: the non-GDN
CUDA path still does coarse whole-layer offload, Q5_K experts get needlessly
expanded to F32 on two paths, and the profiler we built doesn't yet steer caching.
Those are mostly wiring over infrastructure we already have, and they're what
stands between "MoE that fits in VRAM" and "MoE that's bigger than VRAM but still
fast."

---

## 7. References

- KTransformers — CPU/GPU Hybrid Inference for MoE (SOSP'25): https://dl.acm.org/doi/10.1145/3731569.3764843
- llama.cpp MoE offload guide (`-ot`/`--cpu-moe`): https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide
- Not All Models Suit Expert Offloading (Local Routing Consistency): https://hf.co/papers/2505.16056
- MoE-Infinity — Activation-Aware Expert Offloading: https://hf.co/papers/2401.14361
- HybriMoE — Hybrid CPU-GPU Scheduling & Cache: https://hf.co/papers/2504.05897
- Pre-gated MoE (ISCA'24): https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf
- Cross-Layer Gate / Fate — Accurate Expert Predictions: https://hf.co/papers/2502.12224
- ProMoE — Proactive Caching: https://hf.co/papers/2410.22134
- AdapMoE — Sensitivity-based Gating/Management: https://hf.co/papers/2408.10284
- fMoE — Fine-Grained Expert Offloading: https://hf.co/papers/2502.05370
- ExpertFlow — Predictive Routing + Token Scheduling: https://hf.co/papers/2410.17954
- HOBBIT — Mixed-Precision Expert Offloading: https://hf.co/papers/2411.01433
- PreMoe — Expert Pruning & Retrieval: https://hf.co/papers/2505.17639
- MoE-SpeQ — Speculative Quantized Decoding + Prefetch: https://arxiv.org/abs/2511.14102
- SP-MoE — Speculative Decoding & Prefetching: https://arxiv.org/pdf/2510.10302
- BuddyMoE — Expert Redundancy for Cache Misses: https://arxiv.org/html/2511.10054v1
- Towards MoE Deployment (Expert Buffering / token reorder): https://hf.co/papers/2303.06182
