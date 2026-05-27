# Implementation Plan: `qwen35moe` Architecture for SharpInference

*Drafted 2026-05-19.*

## Executive Summary

Adding `qwen35moe` is a substantial multi-week project (~9–13 working weeks, 45–65 person-days). It bolts on a second block type (SSM/Mamba-2 style) that didn't previously exist in SharpInference, plus changes to RoPE, MoE shapes, and tokenization. The core engine has good architectural runway — the `ForwardPass` family is metadata-driven (no `arch == "qwen3moe"` switches), MoE expert handling is dimension-generic, and `PagedKvCache` has clean lifecycle hooks. The risk is concentrated in SSM kernel correctness (selective scan numerics) and the per-sequence SSM state interacting with batched/multi-sequence code paths.

The CUDA hybrid path is the only realistic deployment target (22 GB model, 12 GB VRAM RTX 4070 Ti), so prioritize CPU correctness → CUDA hybrid; defer pure-CUDA and Vulkan.

## Motivation

User downloaded `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.1 GB, at `E:\models\`) expecting a Qwen3-MoE successor. It's actually a different architecture: a hybrid Mamba-style SSM + sparse attention + MoE model. The current engine fails at load on `blk.0.attn_q.weight` because 3 of every 4 layers are SSM blocks with no `attn_q` tensor.

## GGUF metadata (from `sharpi-cli list-metadata`)

- `general.architecture = qwen35moe`
- `qwen35moe.block_count = 40`
- `qwen35moe.full_attention_interval = 4` → layers 0, 4, 8, 12, … are full attention; the other 30 layers are SSM blocks
- `qwen35moe.attention.head_count = 16`, `head_count_kv = 2`, `key_length = 256`, `value_length = 256` (GQA, head dim 256)
- `qwen35moe.rope.dimension_count = 64`, `rope.dimension_sections = [11, 11, 10, 0]`, `rope.freq_base = 1e7` — **partial RoPE** (64 of 256 head dims rotated)
- `qwen35moe.embedding_length = 2048`, `vocab_size = 248320`, `context_length = 262144`
- MoE: `expert_count = 256`, `expert_used_count = 8`, `expert_feed_forward_length = 512`, `expert_shared_feed_forward_length = 512` (shared expert present)
- SSM block params:
  - `ssm.conv_kernel = 4` (1D depthwise causal conv, kernel size 4)
  - `ssm.group_count = 16`
  - `ssm.inner_size = 4096`
  - `ssm.state_size = 128`
  - `ssm.time_step_rank = 32`

---

## Phase-by-Phase Rollout

### Phase 0 — Discovery & Tensor Naming (1–2 days)

**Goal:** Confirm exact tensor names; this de-risks the entire plan because file naming determines what we need to load.

**Tasks:**
- Add a `list-tensors` CLI command mirroring `ListMetadataCommand` (`src/SharpInference.Cli/ListMetadataCommand.cs:12`) that iterates `model.Tensors` and prints `Name | DType | Shape | ByteSize`. Wire it into `Program.cs` next to `list-metadata`.
- Run against `E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Capture the actual SSM tensor names and shapes.
- Cross-check against llama.cpp's mamba/qwen3next conventions; `ModelGraph.cs:148-150` already references `qwen3next`, `lfm2`, `lfm2moe`, `falcon-h1` — these are existing hybrid SSM architectures whose tensor naming is the likely template. Probable names per SSM layer: `ssm_in.weight`, `ssm_conv1d.weight`, `ssm_conv1d.bias`, `ssm_x.weight` (combined dt+B+C projection in Mamba-2), `ssm_dt.weight`, `ssm_dt.bias`, `ssm_a` (log A, shape `[group_count]` or `[ssm_inner]`), `ssm_d` (per-channel bias, `[ssm_inner]`), `ssm_norm.weight`, `ssm_out.weight`.
- Confirm full-attention layers (0, 4, 8, …, 36) carry the standard `attn_q/k/v/output.weight` and `attn_norm.weight` (likely yes; the load failure at `blk.0.attn_q.weight` from the user's report only fails on SSM layers 1–3).
- Document which layers carry which tensors → this fixes the layer-type mask.

**Files to touch:**
- `src/SharpInference.Cli/Program.cs` (register new command)

**Files to create:**
- `src/SharpInference.Cli/ListTensorsCommand.cs`

**Effort:** 1–2 days.

---

### Phase 1 — Architecture Registration & Hyperparameter Loading (2–3 days)

**Goal:** `ModelGraph` understands `qwen35moe`; metadata loads cleanly; the architecture can be opened in `list-metadata` without falling through to llama defaults.

**Key design decisions:**

1. **Add layer-type mask to `ModelHyperparams`.**
   Introduce `IReadOnlyList<LayerType> LayerTypes` (or a bitmask `bool[] IsAttentionLayer`). Populated at `FromGgufMetadata` time using `qwen35moe.full_attention_interval`. This makes the per-layer dispatch a simple `_layerTypes[i]` lookup rather than re-deriving `(i % 4 == 0)` everywhere.
   **Trade-off:** Adds 40 bytes to hp; trivial. Worth it because some hybrid architectures (qwen3next, falcon-h1) may use *different* patterns and we want a generic representation.

2. **Add SSM hyperparams** as a nested record `SsmConfig?` (null when not hybrid):
   `ConvKernel`, `GroupCount`, `InnerSize`, `StateSize`, `TimeStepRank`. Keep separate from attention dims to keep existing transformer paths uncluttered.

3. **Partial RoPE: add `RopeDim` (separate from `HeadDim`).**
   Current code assumes `headDim = ropeDim`. For `qwen35moe`: `headDim=256`, `ropeDim=64`. Plumb through every RoPE call. Default `RopeDim == HeadDim` for backward compat.
   **Trade-off on `dimension_sections`:** For text-only inference, multi-section M-RoPE collapses to a single position scalar — sections only differ when the model receives multimodal position IDs (temporal, height, width). Treat as single-section partial RoPE on the first 64 dims with `freq_base=1e7`. Sections `[11,11,10,0]` sum to 32 frequency pairs = 64 rotated dims, which matches `rope.dimension_count=64`. **Validate this empirically against llama.cpp's logits in Phase 5.**

4. **Add `IsNeoxRope` entry for `qwen35moe`** at `ModelGraph.cs:137` (the qwen family already maps to NEOX).

5. **Detect SSM presence** at load time the same way `HasAttnBias` is detected (`GgufModel.cs:127`): probe for `blk.1.ssm_in.weight` (knowing layer 1 is an SSM block under `full_attention_interval=4`). Inject synthetic `_sharpi.is_hybrid_ssm = true` to keep `FromGgufMetadata` pure.

6. **MoE shape generality.** `ExpertSlotManager.cs` and the MoE code in `ForwardPass.MoeFfn` use `_hp.NumExperts`, `_hp.NumActiveExperts`, `_hp.ExpertIntermediateDim` — no 128-baked constants. **The 256-expert path "just works"** at the data-structure level. *However* `SelectTopK` (`ForwardPass.cs:1179`) is O(numExperts × k) = O(256 × 8) = 2048 cmp/layer/token, fine for now. CUDA's `MoeFfn` downloads `numExperts` floats to do top-K on CPU (`CudaForwardPass.cs:~580`) — for 256 experts vs 128, this doubles the per-layer CPU↔GPU sync; consider implementing a GPU-side top-K in a later phase.

**Files to touch:**
- `src/SharpInference.Core/ModelGraph.cs` — add `LayerTypes`, `SsmConfig`, `RopeDim`, `IsHybridSsm`; extend `FromGgufMetadata` to parse `qwen35moe.*` keys; extend the NEOX list.
- `src/SharpInference.Core/GgufModel.cs:124-129` — extend synthetic-metadata injection to detect SSM tensors.
- `src/SharpInference.Core/IComputeBackend.cs:93` — add overload `RoPE(Tensor x, int position, int headDim, int ropeDim, float ropeTheta, bool neox)`. Old signature stays as a forwarding wrapper.

**Effort:** 2–3 days. Mostly mechanical, but `LayerTypes` propagation touches every forward pass file.

---

### Phase 2 — CPU SSM Kernels in Isolation (5–8 days)

**Goal:** A standalone CPU implementation of the SSM block (depthwise conv, dt projection, selective scan) that we can unit-test against a known reference *before* integrating into ForwardPass.

**Why this ordering:** Selective scan numerical correctness is the single biggest risk. We must be able to run it on tiny shapes with synthetic inputs and compare against either (a) a Python/numpy reference or (b) llama.cpp's CPU mamba implementation, before any of it goes into a 1500-line forward pass.

**Key design decisions:**

1. **New kernel module** `SsmKernels` in `SharpInference.Cpu` (separate file `src/SharpInference.Cpu/SsmKernels.cs`). Keeps the SSM math co-located and out of `SimdKernels.cs` (already 1978 lines).

2. **Kernels needed (CPU prefill + decode):**
   - `CausalDepthwiseConv1d(x[T, D], weight[D, K], bias[D], scratch[D*(K-1)], out[T, D])` — kernel size K=4. For decode: stateful version that keeps last 3 inputs in `scratch` (the "conv state").
   - `DtProjectionAndSoftplus(dtRank[T, R], dtWeight[D, R], dtBias[D], out[T, D])` — small matmul + `softplus(x + bias) = log(1 + exp(x + bias))`.
   - `SelectiveScanDecode(x[D], dt[D], A_log[D], B[N], C[N], D_skip[D], state[D, N], out[D])` — single-token. Computes `Δ = dt`, `Ā = exp(Δ * (-exp(A_log)))`, `B̄ = Δ * B`, `h = Ā ⊙ h + B̄ ⊙ x`, `y = h · C + D_skip ⊙ x`. The state evolves in place. This is the **decode kernel**.
   - `SelectiveScanPrefill(x[T, D], dt[T, D], A_log[D], B[T, N], C[T, N], D_skip[D], stateIn[D, N], stateOut[D, N], out[T, D])` — sequential T-step recurrence; same math but loops T times. Sequential, not "parallel scan" — for first version, simplicity over speed.
   - Reuse existing `SiLuMul`, `AddInPlace`, `RmsNorm` for surrounding glue.

3. **SIMD strategy.** Vectorize across the `D` dimension (= `ssm_inner = 4096`, divisible by 16/8). The recurrence is sequential in T but parallel across D — fits AVX2/AVX-512 nicely. The scan inner loop is `state[d,n] = Ā[d] * state[d,n] + B̄[d,n] * x[d]` — vectorize across N (=128) with D as outer loop.

4. **Group structure.** `group_count=16`, `ssm_inner=4096` → 256 channels per group. `B` and `C` are per-group (shape `[T, group_count, state_size]`); the channel-to-group mapping is `d / 256`. Apply per-group `B` and `C` to the per-channel state. This is Mamba-2 style "multi-input SSM" — each group of channels shares the same B/C.

5. **Quantization.** SSM weights from GGUF will be `Q4_K_M` for the big projections (`ssm_in`, `ssm_out`, `ssm_x`) and `F32` for the small per-channel `ssm_a`, `ssm_d`, `ssm_dt.bias`, `ssm_conv1d.bias`. Reuse `SimdKernels.MatVec`/`MatVecBatched` for projections; the scan itself runs on F32 (state, dt, A, B, C are all small and worth keeping in F32 for stability).

6. **Numerical stability.** Use `log1p(exp(x))` for softplus when `x < 20`, fallback to `x` directly when `x ≥ 20` (overflow). Compute `exp(-exp(A_log) * Δ)` carefully — `A` is always ≤0 so `exp(A) > 0`, then `Δ * exp(A)` is positive, then `exp(-positive)` is in `(0, 1]`. Add an assertion in DEBUG builds that the discretized `Ā` ∈ `(0, 1]`.

**Tests (in `tests/SharpInference.Tests.Core` or a new `Tests.Ssm`):**
- Each kernel against hand-computed values on `D=4, N=2, T=3` shapes.
- `ConvDecode + ConvDecode + ...` equivalence to `ConvPrefill` over the same T tokens.
- `ScanDecode × T` equivalence to `ScanPrefill(T)`.
- Numerical stability: scan with extreme `A_log` and `dt` values doesn't produce NaN/Inf.

**Files to create:**
- `src/SharpInference.Cpu/SsmKernels.cs`
- `tests/SharpInference.Tests.Core/SsmKernelsTests.cs` (or a new project `Tests.Ssm`)

**Files to touch:**
- `src/SharpInference.Cpu/Dequantize.cs` if any new SSM-specific tensor layouts need dequant (unlikely; the standard rowmajor Q4_K_M dequant covers projections).

**Effort:** 5–8 days. The scan math is fiddly; expect a full day of debugging numerical drift even with a reference.

---

### Phase 3 — SSM State Cache (2–3 days)

**Goal:** A per-sequence SSM state container with the same lifecycle hooks as `PagedKvCache` so it can be plumbed alongside.

**Key design decisions:**

1. **Layout per layer per sequence:**
   - Conv state: `[ssm_inner * (conv_kernel - 1)] = 4096 × 3 = 12,288 floats = 48 KB` per SSM layer.
   - SSM scan state: `[ssm_inner * ssm_state_size] = 4096 × 128 = 524,288 floats = 2 MiB` per SSM layer.
   - Per sequence, across 30 SSM layers: `30 × (48 KB + 2 MiB) ≈ 62 MiB`. That's substantial but fixed per sequence (unlike KV cache which grows with position).

2. **Lifecycle.** SSM state evolves token-by-token but has no notion of "page" — it's a flat tensor that gets overwritten in place. `TruncateTo(length)` becomes problematic: you cannot rewind a destructive update without snapshotting.

   **Design call:** For Phase 3, implement `TruncateTo` as **"only valid if length == current length or 0"** — throw on partial rewind. Speculative decoding rewind for SSM is genuinely hard (needs checkpoint/restore), and Qwen3.6 isn't a draft-model use case yet. Document the limitation. `ContinuousBatchingEngine` doesn't rewind (`TruncateTo(currentLength)` is a no-op there), so this only kills speculative decoding for hybrid models. Acceptable for v1.

3. **Allocation.** Eager — at construction, allocate one block per (SSM layer × sequence). Unlike `PagedKvCache.Reset` which returns pages to a warm pool, `SsmStateCache.Reset` just zeros the existing buffers. No page table needed.

4. **API symmetry.**
   - `SsmStateCache(numSsmLayers, ssmInner, stateSize, convKernel)` — single sequence.
   - `ConvStateAt(int ssmLayerIndex) → float*`
   - `ScanStateAt(int ssmLayerIndex) → float*`
   - `Reset()` — zeros all buffers.
   - `TruncateTo(int length)` — throws unless length ∈ {0, _length}.
   - `IncrementPosition()` — bumps `_length`; called once per token after all layers update.

5. **Layer indexing.** Use *SSM-layer index* (0..29) not absolute layer index (0..39). Compute the mapping once at construction: `int[] ssmLayerOfBlock = [-1, 0, 1, 2, -1, 3, 4, 5, -1, ...]`. Attention layers get -1. This avoids reserving 40 slots when only 30 are SSM.

**Files to create:**
- `src/SharpInference.Engine/SsmStateCache.cs`

**Tests:**
- `SsmStateCacheTests.cs` — alloc, write, read, reset, illegal truncate throws.

**Effort:** 2–3 days.

---

### Phase 4 — CPU ForwardPass Integration (4–6 days)

**Goal:** `ForwardPass.Forward` and `Prefill` work end-to-end for `qwen35moe` on CPU. Logits are finite, decode produces non-EOS first token.

**Key design decisions:**

1. **Block-type dispatch.** Add a per-layer `if (_hp.LayerTypes[layer] == LayerType.Attention) { AttnBlock(layer, ...); } else { SsmBlock(ssmIndex, ...); }` in `ForwardCore`. The attention path uses existing code unchanged.

   **Trade-off — extracting block methods vs inline:** Extract `AttnBlock(layer, position)` and `SsmBlock(ssmLayerIdx, position)` as private methods on `ForwardPass`. Avoids doubling code. The extraction touches the layer body (`ForwardPass.cs:773-839`) but is mostly cut-paste.

2. **Tensor reference storage.** Add parallel arrays alongside the existing `_wq, _wk, ...`:

   ```
   _ssmIn, _ssmConv1d, _ssmConv1dBias, _ssmX, _ssmDt, _ssmDtBias,
   _ssmA, _ssmD, _ssmNorm, _ssmOut  — each of length numSsmLayers
   ```

   Resolved in the constructor only for `_hp.LayerTypes[i] == Ssm`. Empty arrays when not hybrid.

3. **SSM state cache wiring.** Allocate `_ssmCache = new SsmStateCache(...)` when `_hp.IsHybridSsm`. Reset/truncate must touch *both* caches: extend `ResetCache()` to call `_ssmCache?.Reset()` and `TruncateTo(0)` to call `_ssmCache?.TruncateTo(0)`. For non-zero truncate, throw (per Phase 3).

4. **Prefill path.** Existing `Prefill` already falls through to sequential `Forward` for MoE (`ForwardPass.cs:333-339`). **Keep this for hybrid** — the batched prefill path requires more thought (per-token SSM state evolution doesn't trivially batch). The qwen3.6-35b model is MoE, so this falls into the sequential path automatically. Add an `_hp.IsHybridSsm` clause to the MoE check, or generalize to `_hp.IsMoE || _hp.IsHybridSsm`.

   **Future optimization:** Parallel scan for prefill (Mamba's `parallel_scan` kernel). Out of scope for v1.

5. **BatchForwardMulti and PrefillWithCache.** Both currently throw for MoE. Extend the throw condition to include `_hp.IsHybridSsm`. This means hybrid models don't get continuous batching in v1. Acceptable — single-user is the target use case for a 22 GB model on a personal RTX 4070 Ti.

6. **Partial RoPE.** Modify `ApplyRope` (`ForwardPass.cs:864`) to take a `ropeDim` parameter. The cos/sin table is sized by `ropeDim/2` not `headDim/2`. In `SimdKernels.ApplyRoPECachedNeox`, the rotation only touches the first `ropeDim` floats of each head; the remaining `headDim - ropeDim` dims pass through. Add a new method `ApplyRoPECachedNeoxPartial(x, cos, sin, numHeads, headDim, ropeDim)` rather than mutating the existing kernel.

   The math: NEOX-style splits rotated dims into pairs `(i, i + ropeDim/2)` for `i ∈ [0, ropeDim/2)`. Dims `[ropeDim, headDim)` are untouched. Verify this against llama.cpp's mamba2/qwen3next implementation.

7. **Per-layer residual stream sharing.** Both attention and SSM blocks contribute via residual to the same `_hidden` stream. The FFN/MoE FFN is the same in both block types (MoE FFN runs *after* either attention or SSM). Looking at common Mamba-hybrid layouts: the attention layer pattern is `x = x + Attn(norm(x)); x = x + MoeFfn(norm(x))`, the SSM layer pattern is `x = x + Ssm(norm(x)); x = x + MoeFfn(norm(x))`. So MoeFfn dispatch is unchanged. **Verify this against the model's actual tensor naming** in Phase 0 — if SSM blocks have `ssm_norm` and *no* `ffn_norm`/expert tensors for those layers, the assumption is wrong and layers 1–3 are pure SSM with no MoE. The metadata's `expert_count=256` suggests every layer has experts, but check.

**Files to touch:**
- `src/SharpInference.Engine/ForwardPass.cs` — major: add SSM tensor arrays, refactor `ForwardCore` to dispatch on layer type, add `SsmBlock` method, extend prefill MoE guard, extend `TruncateTo`/`ResetCache`.
- `src/SharpInference.Cpu/SimdKernels.cs` — add `ApplyRoPECachedNeoxPartial` (and the LLaMA variant for symmetry).
- `src/SharpInference.Core/IComputeBackend.cs` — already covered in Phase 1.

**Tests:**
- `Tests.ForwardPass.HybridSsmTests.HybridSsm_Decode_ProducesNonEosFiniteLogits` — load tiny model if available, otherwise smoke-test against the 22 GB model with `-n 5` cap. Assert `argmax(logits) != EOS && IsFinite(top)`.
- `Tests.ForwardPass.HybridSsmTests.HybridSsm_Prefill_MatchesSequentialForward` — feed the same 8-token prompt through `Prefill` and through 8 `Forward` calls, assert logits match.

**Effort:** 4–6 days.

---

### Phase 5 — Logit Parity vs llama.cpp (2–4 days)

**Goal:** CPU forward pass matches llama.cpp's output token-for-token on a fixed greedy seed.

**Key tasks:**
- Build/install llama.cpp with `qwen35moe` support locally; capture greedy-decode tokens for a 1-token prompt across 20 positions; capture logits at `temperature=0`.
- Add a `--dump-logits` flag to `sharpi-cli` (or wire it into existing tracing infra at `ForwardPass.cs:77`, env `SHARPI_TRACE_NORMS`). Dump first-N logits at each position.
- Compare: top-5 token IDs and their logits should match within `~1e-3` (Q4_K_M quantization roundoff). If they don't, bisect:
  - Disable RoPE — does the SSM scan agree?
  - Disable SSM — does the attention path agree?
  - Disable MoE shared expert — does the routed-only path agree?

**This phase is where bugs surface.** Budget time for re-litigating Phase 2 numerics or Phase 4 routing.

**Likely failure modes to anticipate:**
- Partial RoPE rotation pair convention (NEOX vs LLaMA) on dims `[0, ropeDim/2)` paired with `[ropeDim/2, ropeDim)` instead of `[headDim/2, headDim/2 + ropeDim/2)`.
- SSM `ssm_a` stored as raw `A` vs `log(A)` (llama.cpp conventions vary).
- Group dim ordering in `ssm_x`: is the combined projection `[dt | B | C]` concatenated along output, and what's the slicing math?
- MoE router using sigmoid vs softmax (qwen3moe uses softmax — confirm qwen35moe doesn't switch). Check `qwen35moe.expert_gating_func` metadata key if present.

**Effort:** 2–4 days, depending on how many bugs surface.

---

### Phase 6 — CUDA Hybrid Forward Pass (8–12 days)

**Goal:** `CudaHybridForwardPass` runs `qwen35moe` with the SSM layers on whichever tier (GPU or CPU) they land in via TierPlanner.

**Key design decisions:**

1. **Strongly recommend constraining SSM layers to one tier in v1.** The placement is currently a single `nGpuLayers` cut at layer N. With a hybrid arch, you have two choices:

   - **(A)** Keep the cut, dispatch SSM kernels on both sides — each side needs CPU and GPU SSM implementations and the placement is whatever TierPlanner chose.
   - **(B)** Always run all SSM layers on CPU, all attention layers on the side TierPlanner picks — but with `full_attention_interval=4` and 10 attention + 30 SSM layers, attention is only 25% of compute and most weights live in MoE experts anyway.
   - **(C)** Always run all SSM layers on GPU, place attention layers per TierPlanner.

   **Recommend (A) but with CPU SSM forward as the always-available baseline.** Even when an SSM layer is "on GPU," fall back to CPU SSM for v1; transfer hidden state across the boundary the same way `CudaHybridForwardPass` already handles per-token GPU↔CPU transfer (`_pinnedHidden` at `CudaHybridForwardPass.cs:80`). This means SSM weights for SSM-on-GPU-tier layers stay CPU-resident, freeing VRAM for MoE experts. The CPU SSM scan kernel is the bottleneck-defining piece anyway (only 30 layers × per-token state of 2 MiB × decode = manageable).

2. **CUDA SSM kernels (deferred to Phase 7 / future work).** A real CUDA SSM scan kernel (NVRTC-compiled, like the existing kernels in `CudaTextKernels.cs`) is 2–4 days of work — `selective_scan_decode` and `causal_conv1d_decode`. Defer to Phase 7. Use option (A) above with CPU SSM for v1; this avoids blocking on CUDA kernels.

3. **Per-token GPU↔CPU transfer overhead.** Each SSM layer on the GPU tier triggers a `download(hidden) → CPU SSM → upload(hidden)` round trip. With ~30 SSM layers and `embDim=2048` floats = 8 KB transfer × 2 directions, that's 16 KB × 30 = 480 KB per token over PCIe. At PCIe 4.0 ×16 (~30 GB/s) this is ~16 µs/token — negligible. **Caveat:** kernel launch overhead dominates; ensure each transfer fuses with the boundary operations rather than triggering individual `cudaMemcpyAsync` calls.

4. **MoE in CUDA hybrid.** Currently `CudaHybridForwardPass` *refuses MoE+GPU at construction time* (`CudaHybridForwardPass.cs:103-109`). For Qwen3.6, MoE *is* on both sides, and the user's 12 GB card cannot fit all 256 experts × 30 layers × ~512×2048 Q4_K weights (~5 GB just for one weight matrix × 3 × 30 = ~14 GB). **Must enable MoE on the GPU side of the hybrid** — either:
   - Eager: all experts for GPU-tier layers stay VRAM-resident. With qwen35moe MoE on 10 attention + 30 SSM = all 40 layers, this is infeasible.
   - Lazy SLRU eviction: use `ExpertSlotManager` (Vulkan-only today). Port to CUDA. Each layer's per-token decode does `expertSlotManager.GetOrLoad(layer, expertId)` for each of the 8 active experts, with the slot manager triggering an async upload on miss.

   **Recommend SLRU.** It's the only way the 22 GB model fits a 12 GB card. The Vulkan equivalent already exists at `src/SharpInference.Engine/ExpertSlotManager.cs` and integrates with `HybridForwardPass`. Port the pattern to CUDA: needs `cudaMemcpyAsync` background uploads on a dedicated stream, plus a `MoEPrefetcher` analogue.

5. **SSM cache on CPU side only.** Since (A) says CPU runs all SSM blocks, the `SsmStateCache` lives only on the CPU. No GPU mirror needed for v1. When CUDA SSM kernels arrive in Phase 7, mirror it.

**Files to touch:**
- `src/SharpInference.Engine/CudaHybridForwardPass.cs` — add SSM dispatch on CPU side; remove the "MoE not supported" guard if porting SLRU; route SSM layers through CPU regardless of `_placement`.
- `src/SharpInference.Engine/ExpertSlotManager.cs` — either generalize over `IComputeBackend` (introduces lots of churn) or duplicate as `CudaExpertSlotManager`. **Recommend duplicate** to avoid destabilizing the working Vulkan SLRU during this work.
- `src/SharpInference.Engine/TierPlanner.cs` — needs to account for SSM weight footprint differently (SSM `ssm_in` + `ssm_out` are dense linear projections sized `embDim → ssm_inner` = 2048×4096 = ~8 MiB Q4_K each; 30 layers × 4 SSM tensors × 8 MiB ≈ 1 GiB).

**Files to create:**
- `src/SharpInference.Engine/CudaExpertSlotManager.cs` (if porting SLRU)

**Effort:** 8–12 days. The SLRU port is the dominant cost. If we punt on SLRU and just say "Qwen3.6 needs CPU-only or a 24 GB card," cut to 4–5 days.

---

### Phase 7 — CUDA SSM Kernels (Optional, 5–8 days)

**Goal:** SSM blocks run on the GPU tier when placed there, eliminating per-token transfer cost.

**Key tasks:**

1. **Kernels in `CudaTextKernels.cs`:**
   - `llm_ssm_conv1d_decode` — causal 1D conv with state update. Block: `(blockDim.x = ssmInner / threadsPerBlock)`. Each thread handles a few channels of the conv (shift state, multiply by 4-element kernel, write output).
   - `llm_ssm_selective_scan_decode` — input-dependent recurrence. Each thread block handles one channel `d`; threads within the block parallelize over the state dim `n` (=128, perfect for a warp + half). Loads `state[d, :]` to shared memory, computes `Ā[d] * state[d, n] + B̄[d, n] * x[d]`, writes back, and dot-products with `C[d, n]` for the output.
   - `llm_ssm_dt_softplus` — small kernel, fuse with the `ssm_x` projection if convenient.

2. **No cuBLAS use** — the scan is recurrent and tiny enough that NVRTC custom kernels are the right tool. The projections (`ssm_in`, `ssm_out`, `ssm_x`) reuse existing `llm_matvec_q4k`.

3. **Prefill variant.** A separate `llm_ssm_selective_scan_prefill(T)` that loops T steps internally. Eventually replace with a true parallel scan, but sequential T-loop is fine for v1 prefill if `T < 1024`. The user's primary case is decode anyway (T=1 per layer per token).

4. **Validation.** Same parity strategy as Phase 5 but with CUDA kernels — log SSM output before/after, compare to CPU.

**Files to touch:**
- `src/SharpInference.Cuda/CudaTextKernels.cs` — append new kernels.
- `src/SharpInference.Cuda/CudaBackend.cs` — bind kernels, add methods `SsmConvDecode`, `SsmScanDecode`, `SsmConvPrefill`, `SsmScanPrefill`.
- `src/SharpInference.Engine/CudaHybridForwardPass.cs` — switch GPU-tier SSM blocks to use these kernels; create `_gpuSsmConvState[]` and `_gpuSsmScanState[]` mirrors of `SsmStateCache`.

**Effort:** 5–8 days. Defer until Phase 6 ships and is stable.

---

### Phase 8 — Vulkan & End-to-End Polish (Deferred)

**Recommendation: defer Vulkan indefinitely for `qwen35moe`.**

**Rationale:**
- The user's target hardware is RTX 4070 Ti (CUDA). Vulkan exists as an alternative GPU backend; CUDA hybrid covers the deployment target.
- Vulkan adds ~2500 lines of GLSL (`Shaders.cs` is already 2158 lines). Three new shaders for SSM (conv decode, scan decode, scan prefill) would be ~600 lines of GLSL plus ComputePipeline plumbing.
- No NVIDIA-only customer faces a deficit if Vulkan lags.

**If Vulkan becomes a requirement:** mirror Phase 7's CUDA kernels in GLSL. Workgroup size 256 across `ssm_inner / 16` channel groups; `state_size=128` fits perfectly in shared memory (`128 floats × 4 bytes = 512 bytes`). Estimate 5–7 days additional.

**Files (if pursued):**
- `src/SharpInference.Vulkan/Shaders.cs` — append `SsmConv1dDecode`, `SsmScanDecode` GLSL.
- `src/SharpInference.Vulkan/VulkanBackend.cs` — bind shaders.
- `src/SharpInference.Engine/HybridForwardPass.cs` — SSM dispatch.

**Effort:** 5–7 days (if/when needed).

---

### Phase 9 — Chat Template & Tokenizer Validation (1–2 days)

**Goal:** The 248K vocab loads correctly; chat template formats prompts without vision tokens corrupting text-only generation.

**Tasks:**
- Smoke-test `GgufTokenizer.FromGgufModel` on the 22 GB file. Existing code is vocab-agnostic (no 152K constants in `GgufTokenizer.cs`) — should just work.
- Inspect `tokenizer.chat_template` (the full Jinja chunk that didn't fit in the metadata dump). Look for vision-specific tags (`<|vision_start|>`, `<image>`, etc.).
- Test the existing `JinjaChatTemplate` parser handles the multimodal-aware template gracefully when no images are present. If Jinja branches like `{% if image %}` are unsupported by our parser, add a conditional skip or hard-code text-only mode.
- Confirm `<think>` token detection (`RunCommand.cs:161`) works on this model — Qwen 3.x family uses reasoning tokens.

**Files to touch (if Jinja parser falls short):**
- `src/SharpInference.Core/JinjaChatTemplate.cs`

**Effort:** 1–2 days.

---

### Phase 10 — Testing Coverage & Documentation (2–3 days)

**Tests to add:**
- `Tests.ForwardPass.HybridSsmTests`:
  - `Decode_FirstTokenIsNotEos_OnCpu`
  - `Decode_FirstTokenIsNotEos_OnCudaHybrid` (skip when no CUDA)
  - `Prefill_LogitsMatchSequentialForward` (CPU)
  - `LayerTypeMaskIsCorrect` — assert layers 0,4,8,...,36 are attention and the rest are SSM
  - `SsmStateCache_ResetZeroesState`
  - `PartialRoPE_DimsBeyondRopeDimAreUnchanged`
- `Tests.Ssm` (new project or under Tests.Core):
  - Kernels: conv decode vs prefill equivalence, scan decode vs prefill equivalence, group/B/C broadcast correctness.
- Documentation: update `docs/SharpInference-Design.md` with a "Hybrid SSM Architecture" section. Note the limitations (no speculative decoding rewind, no continuous batching, no Vulkan).

**Effort:** 2–3 days.

---

## Summary Effort Estimate

| Phase | Effort | Cumulative |
|---|---|---|
| 0 — Discovery & tensor naming | 1–2 days | 2 |
| 1 — Arch registration & hparams | 2–3 days | 5 |
| 2 — CPU SSM kernels in isolation | 5–8 days | 13 |
| 3 — SSM state cache | 2–3 days | 16 |
| 4 — CPU ForwardPass integration | 4–6 days | 22 |
| 5 — Logit parity vs llama.cpp | 2–4 days | 26 |
| 6 — CUDA hybrid forward | 8–12 days | 38 |
| 7 — CUDA SSM kernels (optional) | 5–8 days | 46 |
| 8 — Vulkan (deferred) | — | — |
| 9 — Tokenizer/chat | 1–2 days | 48 |
| 10 — Testing & docs | 2–3 days | 51 |

**Realistic ship-to-CUDA-hybrid (Phases 0–6, 9, 10): 32–45 days = 6.5–9 weeks of focused work.**

Adding CUDA SSM kernels (Phase 7) brings it to 38–53 days = 8–11 weeks.

---

## Risk Callouts

1. **Selective scan numerical stability across quantization.** Highest risk. Q4_K_M dequant of `ssm_in`/`ssm_out` projections + F32 scan + Q4_K_M MoE projections is a long chain. Plan: keep all *scan state* in F32 (don't quantize), validate with parity testing at Phase 5, expect to add per-channel scale clamps if scan output explodes. The model's `ssm_a` is per-channel and stored as `log(A)` — confirm storage convention against llama.cpp because `exp(log(A))` overflow when `log(A)` is positive (it shouldn't be — `A` is constrained negative — but quantization may have artifacts).

2. **Per-sequence SSM state interaction with batched paths.** `BatchForwardMulti` and `PrefillWithCache` are now disabled for hybrid SSM (Phase 4). This *removes* `ContinuousBatchingEngine` support for Qwen3.6 — confirm acceptable. If continuous batching is needed later, each sequence needs its own `SsmStateCache`, and the batched-layer routine would need to dispatch the SSM kernels per-sequence (no batching benefit, since each sequence's state is independent — at which point sequential per-sequence decode is the right call anyway).

3. **Partial RoPE section math.** Misreading `dimension_sections = [11, 11, 10, 0]` as a 4-way frequency partition (instead of single-section partial) would yield silent garbage. Mitigation: validate Phase 5 against llama.cpp at the K vector before attention, not just final logits. If section math truly matters for text, build a section-aware RoPE; otherwise treat as straight partial on first 64 dims.

4. **MoE 256-expert performance pathology.** Existing `SelectTopK` does O(n×k) selection; fine for k=8 over n=256 (2048 cmps). The router GEMM is `embDim=2048 → 256 experts` = 524 KB Q4_K, small. The *real* MoE cost is 8 expert evaluations × layer × token; with 40 layers × 8 experts = 320 expert MatVecs per token. Currently each expert MatVec reads its slice from a packed `ffn_*_exps` tensor (`ForwardPass.cs:1149-1157`). At Q4_K_M with `expertDim=512, embDim=2048`, one expert gate/up = 512×2048×0.5625 = 600 KB. 320 reads × 600 KB = 192 MB/token of weight traffic — at DRAM bandwidth of ~50 GB/s effective, that's 4 ms/token *just for MoE on CPU*. The model's "active 3B" sells well, but token rate will be DRAM-bound.

5. **CUDA expert-slot porting.** `ExpertSlotManager` is Vulkan-specific today; porting introduces new code in the hot path with thread-safety and cudaStream coordination concerns. Mitigation: ship CPU-only first (Phase 5), get logit parity, then take the CUDA hybrid SLRU work as a second milestone.

6. **TruncateTo restrictions cascade.** Disabling speculative decoding rewind disables one of SharpInference's distinctive features for this model. Confirm acceptable upfront with the user. Implementing checkpoint/restore for SSM state is doable (snapshot the 62 MiB per-sequence state on each pre-draft point, restore on rejection) but takes ~3 days.

7. **Synthetic-metadata probe ordering.** `GgufModel.cs:127` probes `blk.0.attn_q.bias`. The new SSM probe would target `blk.1.ssm_in.weight` (knowing the interleave pattern), but the *interleave step* itself is metadata-driven (`full_attention_interval`). Need to parse that metadata key before doing the tensor probe, or probe at multiple layers. Mitigation: probe `blk.0`, `blk.1`, `blk.2`, `blk.3` for `ssm_in.weight` — if any match, mark hybrid; then the layer-type mask uses the explicit `full_attention_interval` metadata.

8. **GGUF Q4_K_M tensor shapes for grouped SSM tensors.** GGUF stores tensors row-major with row length divisible by the Q4_K block size (32). If `ssm_x` is `[ssm_inner, dt_rank + 2 * group * state] = [4096, 32 + 2*16*128] = [4096, 4128]`, 4128 isn't divisible by 32 by happenstance (it is: 4128/32=129, OK) — but the convention may instead pack `[group, dt+B+C, ssm_inner]` and 32 might not divide cleanly. Confirm at Phase 0.

---

## Critical Files for Implementation

- `src/SharpInference.Core/ModelGraph.cs`
- `src/SharpInference.Engine/ForwardPass.cs`
- `src/SharpInference.Engine/CudaHybridForwardPass.cs`
- `src/SharpInference.Cpu/SsmKernels.cs` (new)
- `src/SharpInference.Engine/SsmStateCache.cs` (new)

---

## Phase 11 — Qwen3.6-27B-MTP on CUDA + CUDA-Hybrid

*Added 2026-05-25. Tracks GitHub issue #25.*

The 27B-MTP is a dense (no MoE) hybrid GDN+attention model with native MTP heads. Architecturally close to qwen35moe but distinct enough to require its own code paths and a per-token GDN state snapshot facility that the qwen35moe work explicitly deferred (Risk #6 above).

**Hyperparameters (from `Qwen/Qwen3.6-27B/config.json`):**

- `model_type = qwen3_5`, `architectures = Qwen3_5ForConditionalGeneration`
- 64 layers, `full_attention_interval = 4` → 48 GDN + 16 attention (same pattern as qwen35moe, more layers)
- `hidden_size = 5120`, `head_dim = 256`, `num_attention_heads = 24`, `num_key_value_heads = 4` (GQA)
- `intermediate_size = 17408` — **dense FFN**, no MoE
- `linear_num_value_heads = 48`, `linear_num_key_heads = 16`, `linear_value_head_dim = 128`, `linear_conv_kernel_dim = 4` — wider GDN than qwen35moe's 32-VHead variant
- `attn_output_gate = true`, `rope_theta = 1e7`, `partial_rotary_factor = 0.25` (64 of 256 dims rotated, same as qwen35moe)
- `mtp_num_hidden_layers = 1` — single MTP head, shares `embed_tokens` and `lm_head` with the main model
- `vocab_size = 248320`, `max_position_embeddings = 262144`

**Hardware target:** RTX 4070 Ti 12 GB. Q4_K_M file is 17.1 GB → must partially CPU-offload. Lower quants (Q3_K, IQ3_*) are not supported by `CudaBackend.MatMul` (Q4_K / Q5_K / Q6_K / F32 only, see `CudaBackend.cs:706-724`), so Q4_K_M with per-layer offload is the only Cuda-hybrid configuration. Pure all-GPU Cuda is infeasible at any supported quant on 12 GB.

**MTP tensors (per llama.cpp PR #20533):**

| GGUF name | Maps to (transformers) | Purpose |
|---|---|---|
| `mtp.fc` | `model.layers.{bid}.eh_proj` | Concat(embed, hidden) → hidden projection |
| `mtp.pre_fc_norm_embedding` | `model.layers.{bid}.enorm` | Pre-fc norm on input embedding |
| `mtp.pre_fc_norm_hidden` | `model.layers.{bid}.hnorm` | Pre-fc norm on last-layer hidden state |
| `mtp.norm` | `model.layers.{bid}.shared_head.norm` | Post-MTP-block norm |
| (`output.weight` reused) | `shared_head.head` | Shared lm_head — no separate MTP head weight |
| (`token_embd.weight` reused) | `model.embed_tokens` | Shared input embedding |

Plus a standard transformer block (attention + FFN of the same arch as main layers) per MTP head.

GGUF metadata key: `{arch}.nextn_predict_layers` (= 1 for 27B-MTP).

### Empirical confirmation from `list-metadata` / `list-tensors`

**Run on 2026-05-25 against `C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf` (17.11 GB, 866 tensors, 54 metadata keys).**

GGUF arch string is **`qwen35`** (not `qwen3_5` as the transformers config implies, not `qwen35moe`). All hyperparams live under the `qwen35.*` prefix and align cleanly with the existing GDN config reader:

| Key | Value | Maps to |
|---|---|---|
| `qwen35.block_count` | 65 | 64 main layers + 1 MTP head at `blk.64` |
| `qwen35.full_attention_interval` | 4 | `[GDN, GDN, GDN, attn] × 16` → 48 GDN + 16 attn |
| `qwen35.embedding_length` | 5120 | hidden_size |
| `qwen35.feed_forward_length` | 17408 | dense FFN intermediate — **no MoE** |
| `qwen35.attention.head_count` / `head_count_kv` | 24 / 4 | GQA |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 | head_dim |
| `qwen35.rope.dimension_count` | 64 | partial NEOX RoPE (64 of 256) |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` | M-RoPE sections; collapse to single-section for text |
| `qwen35.rope.freq_base` | 1e7 | same as qwen35moe |
| `qwen35.ssm.conv_kernel` | 4 | same as qwen35moe |
| `qwen35.ssm.group_count` | 16 | same as qwen35moe |
| `qwen35.ssm.inner_size` | 6144 | wider than qwen35moe (4096) |
| `qwen35.ssm.state_size` | 128 | same as qwen35moe |
| `qwen35.ssm.time_step_rank` | 48 | wider than qwen35moe (32) — = `linear_num_value_heads` |
| `qwen35.nextn_predict_layers` | 1 | one MTP head |
| `_sharpi.is_hybrid_ssm` | true | **synthetic probe trips correctly** — no arch hardcode needed for hybrid detection |

**GDN tensor naming is IDENTICAL to `qwen35moe`** — the existing GDN kernels in `HybridGdnForwardPass` / `CudaHybridGdnForwardPass` will work unchanged. Per-layer GDN tensors at `blk.0` (a GDN layer):

```
blk.0.attn_gate.weight       Q4_K    [5120, 6144]    16.9 MiB
blk.0.attn_norm.weight       F32     [5120]          20.0 KiB
blk.0.attn_qkv.weight        Q6_K    [5120, 10240]   41.0 MiB    Q(2048)+K(2048)+V(6144)
blk.0.ffn_down.weight        Q6_K    [17408, 5120]   69.7 MiB    dense FFN
blk.0.ffn_gate.weight        Q4_K    [5120, 17408]   47.8 MiB
blk.0.ffn_up.weight          Q4_K    [5120, 17408]   47.8 MiB
blk.0.post_attention_norm    F32     [5120]
blk.0.ssm_a                  F32     [48]
blk.0.ssm_alpha.weight       F32     [5120, 48]
blk.0.ssm_beta.weight        F32     [5120, 48]
blk.0.ssm_conv1d.weight      F32     [4, 10240]
blk.0.ssm_dt.bias            F32     [48]
blk.0.ssm_norm.weight        F32     [128]
blk.0.ssm_out.weight         Q5_K    [6144, 5120]    20.6 MiB
```

**MTP head at `blk.64`** — one full standard attention block (`attn_q/k/v/output` separate, not `attn_qkv` joint, because the MTP head is a regular attention block not a GDN block) **plus 4 `nextn.*` tensors:**

```
blk.64.attn_k.weight                   Q4_K     [5120, 1024]
blk.64.attn_k_norm.weight              F32      [256]
blk.64.attn_norm.weight                F32      [5120]
blk.64.attn_output.weight              Q4_K     [6144, 5120]
blk.64.attn_q.weight                   Q4_K     [5120, 12288]    Q‖gate interleaved (gated attention)
blk.64.attn_q_norm.weight              F32      [256]
blk.64.attn_v.weight                   Q6_K     [5120, 1024]
blk.64.ffn_down.weight                 Q6_K     [17408, 5120]
blk.64.ffn_gate.weight                 Q4_K     [5120, 17408]
blk.64.ffn_up.weight                   Q4_K     [5120, 17408]
blk.64.nextn.eh_proj.weight            Q8_0     [10240, 5120]    concat(enorm(e)‖hnorm(h)) → 5120
blk.64.nextn.enorm.weight              F32      [5120]           pre-fc norm on embedding
blk.64.nextn.hnorm.weight              F32      [5120]           pre-fc norm on hidden
blk.64.nextn.shared_head_norm.weight   F32      [5120]           pre-output norm
blk.64.post_attention_norm.weight      F32      [5120]
```

MTP head total: **15 tensors, 276 MiB** (≈ same cost as one main layer; expected since it IS one main-style layer plus 4 norms and the eh_proj concat).

**MTP forward shape:**
```
                  embedding of next decoded token e ∈ R^5120    last-layer hidden h ∈ R^5120
                          │                                        │
                       enorm(e)                                 hnorm(h)
                          └───────────── concat ──────────────────┘
                                        ∈ R^10240
                                          │
                                       eh_proj  ∈ R^5120
                                          │
                          standard attention block + dense FFN (full-attn, 24×256 heads, GQA 4 KV, gated Q‖gate)
                                          │
                                 shared_head_norm
                                          │
                                output.weight (shared lm_head)
                                          │
                                logits ∈ R^248320 for position p+1
```

**Tensor-name lookup for code:** the GGUF prefix is `nextn.` (NOT `mtp.` as PR #20533 might suggest — that was the internal canonical, the on-disk name is `nextn.`).

### Updated assessment of structural blockers

- Blocker #1 (`CudaHybridGdnForwardPass.cs:275-276` requires MoE) — **CONFIRMED**, still needs dense-FFN generalization.
- Blocker #2 (CUDA dequant set) — **CONFIRMED**: quants in the 27B-MTP file are Q4_K (most), Q5_K (`ssm_out`), Q6_K (`ffn_down`, some `attn_*`), Q8_0 (`nextn.eh_proj`), F32 (norms). All but Q8_0 are supported by `CudaBackend.MatMul`. The `nextn.eh_proj` weight is Q8_0 — needs either dequant on host or a Q8_0 CUDA matvec kernel. Q8_0 dequant is trivial (8 bits + per-block float scale); easiest fix is to dequant to F16/F32 once at load.
- Blocker #3 (`qwen35` arch unknown to `ModelGraph`) — **CONFIRMED**: needs to be added to the NEOX-rope arch list. Hybrid detection via synthetic `_sharpi.is_hybrid_ssm` probe **already works** (verified by metadata dump showing `_sharpi.is_hybrid_ssm = true`).

### Implementation landed 2026-05-25

- **11.1 arch registration** — `qwen35` added to NEOX-rope list at `ModelGraph.cs:179`. Hybrid detection auto-triggers via tensor probe in `GgufModel.cs:132-136`.
- **11.0/11.1 MTP layer accounting** — `NumLayers = block_count - nextn_predict_layers` in `ModelGraph.cs:202-209`. New `ModelHyperparams.NumMtpLayers` field at `ModelGraph.cs:121-128`. Main forward loop iterates 0..NumLayers-1; MTP block at `blk.NumLayers` is loaded only by the MTP head path (now wired, below).
- **11.2 dense-FFN on the hybrid GDN path** (CPU + CUDA hybrid) — `HybridGdnForwardPass` and `CudaHybridGdnForwardPass` now accept `!hp.IsMoE`. MoE-only state (router logits, expert scratch, shared-expert weights, SLRU manager) is gated behind `hp.IsMoE`. New dense FFN code paths: `DenseFfn(layer)` in CPU pass and `CpuDenseFfn(layer)` in CUDA hybrid (download GPU norm → CPU mmap matvec → upload). Backend label updated to distinguish dense-FFN from MoE.
- **11.3 per-layer FFN-on-GPU offload — NOT LANDED.** See blocker below.
- **11.5 MTP head tensor loading** (CPU only) — `HybridGdnForwardPass` now loads the 4 `nextn.*` tensors plus the full attention + dense FFN block at `blk.{NumLayers}` when `hp.NumMtpLayers > 0`. `nextn.eh_proj` (Q8_0) is dequantized to F32 at load (~200 MiB residence; ~52 M MACs per draft step). Per-head Q/K norms loaded into native F32 buffers. Separate `PagedKvCache(numLayers=1)` for the MTP attention block.
- **11.6 MTP head forward path** (CPU only) — new `IForwardPass.MtpForward(token, position, prevHidden)` + `HasMtpHead` + `LastHidden` interface surface. Pre-output-norm `_hidden` is snapshot into `_lastHidden` at the end of each main `Forward` so the next MTP draft has access. `MtpForward` runs: embed → enorm+hnorm → concat([hnorm(h), enorm(e)]) → eh_proj → standard gated attention with MTP KV cache → dense FFN → shared_head_norm → shared lm_head. Smoke test `HybridGdnForwardPass_Qwen35Mtp_MtpHeadProducesWellFormedLogits` asserts finite, non-degenerate logits on the 27B-MTP model.
- **11.8 MTP verify-and-accept loop** — new `MtpDecoder` class (sibling to `SpeculativeDecoder` since GDN's `SupportsPartialRewind == false` blocks reuse). Sequential N=1 algorithm: emit `t1 = argmax(saved_main_logits)`, MTP-draft `t2_draft`, main-verify via `Forward(t1, P)`, accept iff `argmax(main_logits) == t2_draft`, emit `t2`. Both caches advance through committed tokens; **no GDN snapshot needed for N=1 sequential** because the rejected draft never enters the main cache.
- **11.9 `SHARPI_DISABLE_MTP=1` env switch** — `InferenceEngine.GenerateChunksAsync` routes through `MtpDecoder` when `fwd.HasMtpHead && SHARPI_DISABLE_MTP != "1" && sp.Temperature <= 0f && !thinkingEnabled`. Acceptance rate tracing under `SHARPI_TRACE_MTP=1`.

### What did NOT land yet (follow-ups)

- **CUDA hybrid MTP** (mirror of 11.5/11.6 on `CudaHybridGdnForwardPass`). `HasMtpHead` defaults to `false` on the CUDA hybrid pass, so `-g -1` decoding of the 27B-MTP model silently falls through to baseline. Path: replicate the CPU MTP setup with GPU-resident MTP weights + GPU MTP KV cache + GPU `_lastHidden` mirror.
- **11.7 per-token GDN snapshot ring** — only required when batched verify lands (see below). N=1 sequential MTP doesn't need it.
- **Batched main verify** (Phase 7 work). Without it, sequential N=1 MTP costs 2 main forwards / 2 tokens — same as baseline. The issue #25 ≥1.3× speedup criterion requires this.
- **11.10 llama.cpp parity** — needs a side-by-side dump of greedy decode from llama.cpp `--spec-type draft-mtp --spec-draft-n-max 2`. Smoke test currently asserts only finite/non-degenerate logits; greedy parity is the next correctness step.
- **CLI integration** — `RunCommand` uses its own decode loop, not `InferenceEngine`. CLI runs against MTP-enabled models silently bypass `MtpDecoder`. Wire `RunCommand.DecodeLoop` to optionally route through `MtpDecoder` (or move CLI to `InferenceEngine`).
- **Thinking-mode + MTP combined** — current gate disables MTP when the model exposes `<think>` / `</think>` tokens, because `MtpDecoder` doesn't know how to split into reasoning vs text chunks. Workaround: `--no-thinking` on the CLI side. Eventual fix: thread `thinkId`/`endThinkId` through `MtpDecoder` (or have the engine wrap the emit callback to flip chunk kind).

### Measured baselines (RTX 4070 Ti 12 GB)

| Backend | Model | Config | Decode t/s | Notes |
|---|---|---|---|---|
| CPU | qwen36-27b-mtp Q4_K_M | — | 3.1 | Single-thread RmsNorm + parallel matvec |
| CUDA hybrid | qwen36-27b-mtp Q4_K_M | all FFN on CPU | 4.0 | Before exact-size patch; GDN + attn on GPU |
| CUDA hybrid | qwen36-27b-mtp Q4_K_M | 2 FFN layers on GPU | 4.3 | Before exact-size patch; ceiling was VRAM-bound |
| CUDA hybrid | qwen36-27b-mtp Q4_K_M | **21 FFN layers on GPU (exact-size)** | **6.3** | **+58 % over baseline.** SHARPI_DENSE_FFN_GPU_MARGIN_MB=32, ctx ≤ 2048 |
| CUDA hybrid | qwen36-35b-a3b-mtp UD-Q4_K_M | — | 22.3 | Regression check — existing MoE path intact |
| CUDA all-GPU | qwen3-8b Q4_K_M | — | 59.7 | Dense GPU regression check vs 65 t/s prior baseline |

### Root cause of the unaccounted VRAM — found + fixed

**`GpuBufferPool.RoundUp` rounded every device allocation to the next power of two** (`CudaBackend.cs:2120-2127`). Per the diagnostic checkpoints (run with `SHARPI_TRACE_VRAM=1`):

| Stage | Before exact-size | After exact-size | Reclaimed |
|---|---|---|---|
| Constructor entry | 10696 MiB free | 10696 MiB free | — |
| After embedding upload (Q4_K 715 MiB → ?) | 9672 MiB | 10012 MiB | **+340 MiB** (715→1024 GiB bucket eliminated) |
| After output upload (Q6_K 994 MiB → ?) | 8648 MiB | 9016 MiB | **+28 MiB** (994→1024 GiB bucket) |
| After per-layer weight upload | 525 MiB | 3576 MiB | **+3051 MiB** (48 GDN+16 attn layers' avg 50 % rounding waste) |
| FFN-on-GPU layers uploaded | 2 of 64 | **21 of 64** | — |

The pool's purpose is to **reuse** allocations on the hot path (per-token scratch), but weight uploads are session-lifetime and never freed/realloc'd — pooling is pure waste for them.

**Fix landed this session:**
- `IComputeBackend.Allocate` / `Upload` / `UploadRaw` now take `bool exact = false` (default false preserves pool semantics for scratch).
- `CudaBackend` exact path: bypass `_pool.Rent`, allocate via `cudaMalloc(byteSize)` with no rounding; track in `_exactHandles` for direct `cudaFree` on `Free()`.
- `CudaHybridGdnForwardPass.UploadWeight` and `UploadEmbeddingWeight` pass `exact: true` for every weight tensor.
- Same hook is available to other backends (CPU/Vulkan implementations no-op the hint for now; they don't pool).

### Remaining levers (smaller wins, deferred unless needed)

1. **Bf16 KV cache** — halves the 512 MiB KV-cache footprint at ctx 4096 → frees ~256 MiB → 1-2 more FFN layers → ~6.5 t/s estimate.
2. **MTP head + verify-and-accept (issue #25)** — with 1/3 of FFN now on GPU, MTP's batched verify can amortize attention+GDN work across draft positions. Realistic gain 1.2–1.5× → ~8 t/s estimate.
3. **Q8_0 CUDA matvec or load-time dequant for `nextn.eh_proj`** — needed for any GPU MTP forward path.
4. **Per-token GDN snapshot ring** — needed if MTP verify ever needs to roll back across multiple tokens (current draft-N is 1, so snapshot-before-draft + restore-on-reject suffices without a ring).

### Structural blockers (must fix before Phase 11 proper)

1. **`CudaHybridGdnForwardPass` rejects non-MoE models** (`CudaHybridGdnForwardPass.cs:275-276`): hard `throw` on `!hp.IsMoE || !hp.HasSharedExpert`. Either generalize (add dense FFN code paths gated on `hp.IsMoE`) or create a sibling `CudaHybridDenseGdnForwardPass` that shares the GDN/attention dispatch but uses a dense FFN. Sibling is cleaner; generalization risks regressing the well-tuned 35B-A3B path.

2. **`ModelGraph` doesn't know `qwen3_5`** (`ModelGraph.cs:161-181`): not in the NEOX-rope architecture list; hybrid detection only triggers via the synthetic `_sharpi.is_hybrid_ssm` tensor probe (`GgufModel.cs:132-136`). Add `qwen3_5` to the NEOX list. The tensor probe at layers 0-3 will trip on `blk.0.ssm_conv1d.weight` (layers 0-2 are GDN under `full_attention_interval=4`), so hybrid auto-detection should work without an arch hardcode.

3. **Per-token GDN state snapshot facility doesn't exist** (`GdnStateCache.cs:243-313` has end-of-decode snapshot only). MTP verify-and-accept needs to restore the scan state to the acceptance point after rejecting draft tokens. Risk #6 above estimated 3 days for this.

### Work order

| # | Task | Files | Notes |
|---|---|---|---|
| 11.0 | Empirical confirmation | run `list-tensors`/`list-metadata` on `models/Qwen3.6-27B-MTP-Q4_K_M.gguf`; append output to this doc | Confirm tensor naming and `qwen3_5.*` metadata key prefixes |
| 11.1 | Arch registration | `ModelGraph.cs` (add `qwen3_5` to NEOX list; verify `qwen3_5.ssm.*` key reads) | Small patch; rely on existing tensor-probe for hybrid detection |
| 11.2 | Generalize CudaHybridGdnForwardPass to dense FFN | new `CudaHybridDenseGdnForwardPass.cs` or refactor existing | ~80% shared with MoE sibling; pick sibling-class approach to avoid 35B regression |
| 11.3 | Per-layer FFN offload for 27B | `CudaHybridDenseGdnForwardPass.cs` | Layer-level placement (~half FFN layers to CPU on 12 GB); pattern from `CudaHybridForwardPass` |
| 11.4 | Baseline t/s without MTP | `bench-textgen.ps1`/`bench-all.ps1` rows for `qwen36-27b-mtp-{cuda,cuda-hybrid}` | Denominator for MTP speedup measurement |
| 11.5 | MTP head tensor loading | `ModelGraph.cs`, model loader | Read `{arch}.nextn_predict_layers`, load N MTP layers per the table above |
| 11.6 | MTP head forward path | new `MtpHead.cs` or `MtpForwardPass.cs` | Single transformer block + 2 pre-norms + fc projection + lm_head; reuses main-model attention/FFN kernels |
| 11.7 | Per-token GDN snapshot ring | `GdnStateCache.cs`, `HybridGdnForwardPass.cs`, `CudaHybridGdnForwardPass.cs` | N copies (N=2 → ~280 MiB extra VRAM for 27B's 140 MiB/snapshot); swap pointers on accept/reject |
| 11.8 | MTP verify-and-accept loop | new `MtpDecoder.cs` (sibling to `SpeculativeDecoder.cs`, not a refactor) | Cannot reuse `SpeculativeDecoder` — it requires `SupportsPartialRewind` on both passes |
| 11.9 | `SHARPI_DISABLE_MTP=1` env switch | wiring at `InferenceEngine`/`RunCommand` | Mirrors `SHARPI_BYPASS_GDN`/`SHARPI_CPU_GDN` |
| 11.10 | Parity vs llama.cpp `--spec-type draft-mtp` | manual + test | ≥60-token greedy parity per issue #25 acceptance criteria |
| 11.11 | README perf table update | `README.md`, `bench-all.ps1` | `qwen36-27b-mtp-cuda-hybrid` row with `-MTP` and `+MTP` columns; ≥1.3× target |

The dense-FFN generalization (11.2–11.3) and the snapshot ring (11.7) are the biggest unknowns. MTP forward + verify is well-understood since llama.cpp's algorithm is documented.

### Out of scope for Phase 11

- N > 2 MTP draft length (start at N=2 to match llama.cpp's `--spec-draft-n-max 2` default; raise once acceptance rate is measured).
- Vulkan hybrid path (separate issue; qwen35moe-Vulkan is a known-broken row).
- Multimodal vision path (the `mmproj` files are not bundled with `qwen36-27b-mtp`).
- `ContinuousBatchingEngine` integration (single-sequence only, same restriction as qwen35moe).
