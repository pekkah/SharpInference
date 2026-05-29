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

But the audit found three things worth knowing:

1. **The CUDA hybrid path doesn't actually use the cache.** It statically
   uploads *every* expert of every GPU-tier layer to VRAM as **F32** and computes
   them resident. The `ExpertSlotManager`/`MoEPrefetcher` fields exist but are
   never assigned — dead code on the CUDA path. So on CUDA we are effectively
   doing llama.cpp's static `--cpu-moe` split, not dynamic offloading, and paying
   a 4×–8× VRAM premium by storing experts dequantized.
2. **Prefetching is reactive, not predictive.** The Vulkan path re-enqueues the
   experts the router *just* selected, betting the next token reuses them at the
   same layer (1-token, same-layer temporal locality). Every SOTA system instead
   predicts the *next layer's* or *next token's* experts ahead of time. We
   capture none of that lookahead.
3. **Cache placement ignores the profiler we already built.** `TierPlanner`
   places layers by memory footprint; `ExpertAccessProfiler` tracks hot/cold
   experts but nothing feeds it back into placement or eviction priority.

The highest-leverage, lowest-risk wins for our use case are: **(a)** keep cached
experts quantized instead of F32, **(b)** wire the SLRU cache into the CUDA path
so it stops being all-resident, **(c)** add cross-layer / next-layer expert
*prediction* to drive the prefetcher, and **(d)** add a fast CPU expert GEMM path
(KTransformers-style) so CPU-resident experts are cheap to compute rather than
something to avoid. Details and priorities in §5.

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
- **CUDA hybrid** (`CudaHybridForwardPass.cs`): GPU-tier layers upload **all**
  experts to VRAM as `Tensor[][] _gpuWGateExps/...` (line ~297) and index them
  directly (line ~1348); CPU-tier layers compute on CPU (`CpuMoeFfn`). The
  `_expertSlotManager`/`_prefetcher` fields (lines 108–109) are declared and
  disposed but **never assigned** → the dynamic cache path is dead here.

### 2.3 Offloading infrastructure (`SharpInference.Pipeline` + Engine)
- `SlruCache<K,V>` — segmented LRU, 25% probationary / 75% protected, evicts
  probationary tail. `ExpertCache<T>` wraps it keyed by `(layer, expertId)`.
- `ExpertSlotManager` / `CudaExpertSlotManager` — VRAM expert slot cache;
  `TryGetCached`, `Preload`, eviction callback frees GPU tensors. **Dequantizes
  experts to F32 on upload** (`ExpertSlotManager.cs` ~line 136/158) — so a cached
  expert costs 4 B/element regardless of its Q4_K source.
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
| Static hot/cold placement (KTransformers, llama.cpp) | CUDA: all GPU-layer experts resident (F32). Vulkan: SLRU. | CUDA has no offloading at all; neither path uses **profiled hotness** for placement. |
| Activation-aware caching (MoE-Infinity, HybriMoE) | SLRU (recency only) + unused `ExpertAccessProfiler` | Profiler exists but doesn't drive eviction/placement. SLRU ≠ activation-frequency-aware. |
| **Predictive prefetch** (Pre-gated, Cross-Layer Gate, ProMoE) | Reactive 1-token, **same-layer** re-enqueue (Vulkan only) | No next-layer or next-token prediction. The single biggest missing idea. |
| Speculative expert prefetch (MoE-SpeQ, SP-MoE) | none (we have speculative *decoding* for some models, not used for expert prediction) | Not started. Natural extension once a draft model exists. |
| **Mixed-precision experts** (HOBBIT) | Cached experts dequantized to **F32** | We pay max VRAM/transfer for every cached expert; opposite of SOTA. |
| Fast CPU expert GEMM (KTransformers AMX) | SIMD `MatVec` (GEMV) CPU fallback | Adequate for single-token, but no AMX / blocked sgemm; CPU is treated as fallback, not a peer compute tier. |
| Cache-miss substitution (BuddyMoE) | CPU-fallback compute (good!) | We block on CPU compute instead of substituting; fine for single-user, but no redundancy reuse. |
| Model-aware cache sizing (Local Routing Consistency) | fixed slot capacity | No per-model locality measurement; could size cache ≈2× active experts and skip caching for low-locality models. |
| L3 NVMe tier / io_uring | designed, stubbed | Not shipped; only matters for models > host RAM. |
| Batched/continuous-batching MoE + token reorder (ExpertFlow) | MoE batching disabled | Out of scope for single-user target; revisit if server multi-user MoE is wanted. |

---

## 5. Recommendations (prioritized for our target)

Ordered by leverage/effort for the single-user desktop, ~12 GB VRAM, GGUF case.
Most of these reuse infrastructure we already have.

### P0 — Stop the F32 cache bleed (HOBBIT-lite)
Cache experts **in their native quantized form** (Q4_K/Q5_K/Q6_K) in VRAM and
dequantize per-GEMM in the shader/kernel (we already dequantize Q4_K_M for dense
weights). This 3–4×'s the number of experts that fit in the cache for free and
cuts upload bytes proportionally. Optional next step: a true mixed-precision tier
(hot experts native, cold experts re-quantized lower) à la HOBBIT.
*Touches:* `ExpertSlotManager`/`CudaExpertSlotManager` upload path, GPU MoE matmul.
*Risk:* low — quantized matmul already exists.

### P0 — Make the CUDA hybrid path actually offload
Wire `CudaExpertSlotManager` + `MoEPrefetcher` into `CudaHybridForwardPass` (the
fields are already there, just unassigned) so CUDA does dynamic SLRU caching with
CPU fallback like the Vulkan path, instead of statically pinning every expert as
F32. This is what unlocks running larger MoEs (Qwen3-235B-class, DeepSeek) on a
12 GB card. Today CUDA can only run what fits fully resident.
*Risk:* medium — needs the Vulkan path's miss/fallback logic mirrored, but the
pattern exists to copy.

### P1 — Predictive prefetch (Cross-Layer Gate / Pre-gated)
Replace the same-layer 1-token re-enqueue with **next-layer prediction**. Cheapest
viable version: run the *next* MoE layer's router on the *current* layer's hidden
state (it's only an `embDim×numExperts` GEMV — tiny) to get an early, approximate
top-k and prefetch those experts a full layer ahead. ~84–91% accuracy is reported
for single-layer lookahead, which is plenty to convert blocking misses into
overlapped loads. This is the single biggest latency win for the offloaded regime.
*Touches:* prefetcher API (`EnqueuePrefetch` already exists), forward-pass loop.
*Risk:* low–medium; purely a prefetch hint, never affects correctness.

### P1 — Feed the profiler into placement & eviction
We built `ExpertAccessProfiler` but ignore it. Use it two ways: **(1)** at load,
warm the cache / pin the top-N experts per layer (offline-profile or first-N-token
profile) — KTransformers/MoE-Infinity "hot experts on GPU"; **(2)** bias SLRU so
high-frequency experts resist eviction (frequency-aware, not pure recency).
*Risk:* low.

### P2 — Treat CPU as a compute peer (KTransformers-style)
Add a blocked/multi-threaded expert GEMM (and explore AVX-512/AMX where available)
so CPU-resident experts are computed cheaply rather than something we always try
to avoid by moving to GPU. Pairs naturally with the existing CPU-fallback path —
it makes "compute on CPU" the *intended* steady state for cold experts, à la
KTransformers, rather than a stall.
*Risk:* medium (kernel work), high payoff for big-model offload.

### P2 — Model-aware cache policy (Local Routing Consistency)
Measure/lookup per-model routing locality and size the expert cache accordingly
(≈2× active experts is a good default), and detect low-locality models (shared
expert + dense-then-MoE) where caching helps little — pin shared experts, don't
over-invest cache slots on routed ones.
*Risk:* low.

### P3 — Speculative expert prefetch (MoE-SpeQ)
Once a draft model is wired for speculative *decoding*, reuse its accepted draft
tokens to prefetch the experts those tokens will need, several tokens ahead. Big
win only in the heavily-offloaded regime; defer until P0/P1 land.

### Out of scope for now
L3/NVMe tier (only matters when model > host RAM), continuous-batching MoE +
token reordering (single-user target), GDN GPU kernels (orthogonal to offloading).

---

## 6. Bottom line

We're not missing the *concept* of expert offloading — the bones are good and the
CPU-fallback-on-miss design is genuinely SOTA-aligned. We're missing the parts
that make it *pay off on a small GPU*: keeping the cache quantized, using the
cache on CUDA at all, and predicting experts ahead of time instead of reacting.
Those three (P0/P0/P1) are mostly wiring over infrastructure we already built, and
they're what stands between "MoE that fits in VRAM" and "MoE that's bigger than
VRAM but still fast."

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
