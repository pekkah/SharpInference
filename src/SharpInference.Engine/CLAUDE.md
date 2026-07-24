# SharpInference.Engine — type-level map

Loaded when working in the Engine. The root CLAUDE.md has the condensed version;
`docs/SharpInference-Design.md` has the algorithms.

## Forward passes

- `IForwardPass` (defined in Core) — per-token forward pass: `Forward`, `Prefill`,
  `TruncateTo`, `ResetCache`, `VocabSize`, `MaxSeqLen`.
- Implementations: `ForwardPass` (CPU dense), `GpuForwardPass` (Vulkan),
  `CudaForwardPass` (CUDA dense), `HybridForwardPass`/`CudaHybridForwardPass`
  (dense + MoE expert offload), `HybridGdnForwardPass`/`CudaHybridGdnForwardPass`/
  `VulkanHybridGdnForwardPass` (qwen35moe hybrid Gated-DeltaNet + MoE).
- `IBatchedForwardPass` — multi-token batched prefill/decode used by continuous
  batching.
- `ForwardPass.BatchForwardMulti(tokens[], positions[], caches[])` — batched
  multi-sequence decode; amortizes weight reads N× across concurrent users. Each
  sequence has its own `PagedKvCache`. Not supported for MoE or TurboQuant.
- `ForwardPass.PrefillWithCache(tokens, cache, startPos)` — prefills a
  per-sequence cache (used by `ContinuousBatchingEngine` during request
  admission). Admission is chunked (`SHARPI_PREFILL_CHUNK`, default 256 tokens)
  and interleaved with decode steps; multiple in-flight prompts prefill as one
  packed pass via `ForwardPass.PrefillPackedMulti` and admission is gated by a KV
  token budget (`SHARPI_KV_BUDGET_MB`) — issue #183.

## KV caches

- `PagedKvCache` — default for `ForwardPass`; lazily allocated pages of 16
  positions, allocated on first write. `TruncateTo` is a soft operation (enables
  prefix reuse); `Reset` returns pages to a warm pool.
- `KvCache` (simple), `CudaSequenceKvCache` (per-sequence GPU),
  `TurboQuantKvCache` (KVarN 4/2-bit or Lloyd-Max 3-4 bit compressed — see
  `src/SharpInference.TurboQuant/CLAUDE.md`).
- `IMultiSlotKvCache` abstracts per-sequence/multi-slot caches.
- `SnapKvSelector` — prefill-time SnapKV eviction. `GdnStateCache` — snapshots
  Gated-DeltaNet state for MTP rollback.

## Engines

- `IInferenceEngine` — `GenerateAsync(prompt, sp, ct) → IAsyncEnumerable<string>`,
  used by the server. `InferenceEngine` (single-user, prefix caching);
  `ContinuousBatchingEngine` (multi-user batching, activated via
  `SHARPI_MAX_BATCH`).

## Speculative decoding

- `SpeculativeDecoder` — general draft-model speculation (`--draft-model`).
- `MtpDecoder` + `MtpBatchTail` — self-speculative Multi-Token Prediction / NEXTN
  heads (e.g. Qwen3.6-27B-MTP) with folded k-token batched verify, issue #207
  (`--mtp`).
- `PromptLookupDraft` — prompt-lookup draft.
- `DSparkDecoder` + `DSparkDraftModel`/`CudaDSparkDraftModel` — DeepSeek DSpark
  block-parallel safetensors draft heads (docs/dspark-plan.md, PR #413):
  EAGLE-3-style backbone conditioned on target hidden-state taps via
  `IForwardPass.EnableHiddenTaps` (CPU and dense-CUDA targets both capture);
  rank-256 Markov re-bias + confidence-trimmed verify on the host
  (`DSparkHostHeads`); greedy only. Placement via `DSparkPlacementPlanner` /
  `--dspark-place` / `SHARPI_DSPARK_*`. Run recipes: `run-models` skill.
- Server selects via `SpecType`.

## Sampling

- `Sampler` — temperature, top-k, top-p (nucleus), min-p, repetition penalty,
  logit bias, and grammar-constrained decoding (applies an `ITokenConstraint`
  token mask per step — tool-argument grammars and whole-turn JSON-schema
  structured output).

## MoE expert offload

- `ExpertSlotManager`/`CudaExpertSlotManager` — SLRU VRAM expert cache.
- `MoEPrefetcher` — async SSD→RAM→VRAM prefetch.
- `TierPlanner` + `HardwareProfile` — three-tier placement; `MmapPrefault`,
  `WarmPinConfig`.
- `--cpu-moe` / `SHARPI_CPU_MOE` keeps routed experts on the CPU (issues #80/#93).

## Rules

Hot paths (forward passes, caches, sampler) are allocation-free: `NativeMemory`,
`Span<T>`, GPU buffers — no LINQ/closures/boxing per token. A numeric change in
one forward-pass variant usually requires the CPU/CUDA/Vulkan siblings to change
too, or an explicit note that parity is intentionally not required.
