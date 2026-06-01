# Gemma 4 E4B Q8 CUDA — Implementation Plan

Status: planning, no code written yet. GGUF downloaded (8.2 GB at `E:\models\gemma-4-E4B-it-Q8_0.gguf`) and header verified. This plan supersedes the high-level surface in issues #82–#90 with the actual tensor layout from the unsloth GGUF.

## Confirmed architecture (from real GGUF header)

| Key | Value |
|---|---|
| `general.architecture` | `gemma4` (already in NEOX RoPE allowlist at `ModelGraph.cs:208`) |
| `gemma4.block_count` | 42 |
| `gemma4.embedding_length` | 2560 |
| `gemma4.feed_forward_length` | 10240 |
| `gemma4.attention.head_count` | 8 |
| `gemma4.attention.head_count_kv` | 2 (GQA, 4:1 ratio) |
| `gemma4.rope.freq_base` | 1,000,000 (global layers) |
| `gemma4.rope.freq_base_swa` | 10,000 (SWA layers) |
| `gemma4.rope.dimension_count` | 512 (global) |
| `gemma4.rope.dimension_count_swa` | 256 (SWA) |
| `gemma4.attention.key_length` | 512 (global head_dim) |
| `gemma4.attention.value_length` | 512 (global head_dim) |
| `gemma4.attention.key_length_swa` | 256 (SWA head_dim) |
| `gemma4.attention.value_length_swa` | 256 (SWA head_dim) |
| `gemma4.attention.sliding_window` | 512 |
| `gemma4.attention.sliding_window_pattern` | 42-element bool[], 5 SWA : 1 global (T,T,T,T,T,F,…) |
| `gemma4.attention.shared_kv_layers` | 18 (last N layers share earlier K/V) |
| `gemma4.embedding_length_per_layer_input` | 256 (PLE width) |
| `gemma4.final_logit_softcapping` | 30.0 |
| `gemma4.attention.layer_norm_rms_epsilon` | 1e-6 |
| `gemma4.context_length` | 131072 |

**Critical surprise:** per-layer head_dim VARIES. SWA layers use 256, global layers use 512. Every per-layer `attn_q/k/v/output` weight matrix has different shape depending on layer type:
- SWA (e.g. blk.0): `attn_q=(2560,2048)`, `attn_k=attn_v=(2560,512)`, `attn_output=(2048,2560)`
- Global (e.g. blk.5): `attn_q=(2560,4096)`, `attn_k=attn_v=(2560,1024)`, `attn_output=(4096,2560)`

This is unlike anything else in the codebase. ForwardPass currently assumes a single `_qDim/_kvDim`; this becomes per-layer.

## Confirmed tensor inventory

```
Global:
  token_embd.weight              (2560, 262144) Q8_0
  output_norm.weight             (2560,)        F32
  per_layer_token_embd.weight    (10752, 262144) Q8_0   [PLE table, 4.2GB at Q8 — MUST stay CPU]
  per_layer_model_proj.weight    (2560, 10752)  BF16
  per_layer_proj_norm.weight     (256,)         F32
  rope_freqs.weight              (256,)         F32     [likely the long-context scaling table]

Per layer (varies by SWA vs global as noted):
  attn_q/k/v/output.weight       Q8_0
  attn_q_norm.weight             (256 or 512,)  F32   [per-head head_dim norm]
  attn_k_norm.weight             (256 or 512,)  F32
  attn_norm.weight               (2560,)        F32
  post_attention_norm.weight     (2560,)        F32
  ffn_norm.weight                (2560,)        F32
  post_ffw_norm.weight           (2560,)        F32
  post_norm.weight               (2560,)        F32   [PLE post-norm]
  inp_gate.weight                (2560, 256)    F32
  proj.weight                    (256, 2560)    F32
  ffn_gate/up/down.weight        Q8_0
  layer_output_scale.weight      (1,)           F32   [single scalar per layer]
```

## NOT present in this GGUF (issue #82/#90 listed them, but unsloth E4B omits)

- **AltUp**: `altup_proj`, `altup_unembd_proj`, `altup_correct_coef`, `altup_correct_scale`, `altup_predict_coef`, `altup_router`, `altup_router_norm` — NONE present
- **LAuReL**: `laurel_l`, `laurel_r`, `laurel_post_norm` — NONE present

This drastically reduces the scope of issue #90. The Gemma-3n stack reduces to **PLE only**.

## Phased plan (revised against actual GGUF)

### Phase 0 — CUDA Q8_0 matvec + embed-lookup kernels (PREREQUISITE)

Without native Q8_0 GPU kernels, weights dequant to F32 on upload — ~2.1× VRAM blowup; E4B Q8 won't fit 12 GB. Required first.

- `src/SharpInference.Cuda/CudaTextKernels.cs` — add `llm_matvec_q8_0` (clone of `llm_matvec_q4_k_m` pattern), `llm_embed_lookup_q8_0`
- `src/SharpInference.Cuda/CudaBackend.cs:1157-1390` — extend MatMul dispatch with Q8_0 branch
- `src/SharpInference.Cuda/CudaBackend.cs:2598-2650` — resolve via GetKernelFunc with extern-C names
- `src/SharpInference.Cuda/CudaBackend.cs:2575-2590` — add to ForceEagerJit
- `src/SharpInference.Engine/CudaForwardPass.cs:1145-1192` — UploadExpertWeight Q8_0 raw-upload branch (not dequant-to-F32)
- `src/SharpInference.Engine/CudaForwardPass.cs:1282-1287` — EstimateGpuTensorBytes Q8_0 sizing
- Tests: parity test for Q8_0 matvec vs CPU `Dequantize.ToFloat32` + FP32 matmul

**Risk:** medium. Cloning the existing pattern is straightforward; CUDA NVRTC compile/cache must work.

### Phase 1 — ModelHyperparams Gemma 4 fields

Pure metadata parsing, no behaviour change.

- `src/SharpInference.Core/ModelGraph.cs:33-34` — add scalar properties: `EmbeddingScale`, `FinalLogitSoftcap`, `RopeThetaSwa`
- `src/SharpInference.Core/ModelGraph.cs:86-92` — add: `SlidingWindowSize`, `PerLayerEmbeddingWidth`, `HasPostAttnNorm`, `HasPostFfwNorm`, `HasPerLayerTokenEmbd`, `FfnActivation`, `IReadOnlyList<bool>? IsSwaLayer`, `IReadOnlyList<int>? KvSourceLayer`, `IReadOnlyList<int>? LayerHeadDim`, `IReadOnlyList<int>? LayerKvHeadDim`, `IReadOnlyList<int>? LayerRopeDim`
- `src/SharpInference.Core/ModelGraph.cs:184-191` — `bool isGemma4` block
- `src/SharpInference.Core/ModelGraph.cs:229-269` — parse all `gemma4.attention.*`, `gemma4.rope.*_swa`, `gemma4.final_logit_softcapping`, `gemma4.embedding_length_per_layer_input`, build per-layer `IsSwaLayer[]` from `gemma4.attention.sliding_window_pattern` bool[], derive `KvSourceLayer[]` from `gemma4.attention.shared_kv_layers` (llama.cpp rule: `il<NLayerKvFromStart → -1`, else `NLayerKvFromStart-2` if SWA else `NLayerKvFromStart-1`), populate per-layer head_dim arrays
- `src/SharpInference.Core/ModelGraph.cs:271-305` — initializer block
- `src/SharpInference.Core/ModelGraph.cs:330+` — add `enum FfnActivation { Silu, GeluApprox }`
- `src/SharpInference.Core/ModelGraph.cs:308-312` — add `GetBoolArray` helper
- `src/SharpInference.Core/GgufModel.cs:123-137` — synthetic `_sharpi.has_ple`, `_sharpi.has_post_attn_norm`, etc.
- Tests: `Gemma4_PopulatesAllFields` in Tests.Core mirroring `Qwen35Moe_PopulatesGdnConfigAndLayerTypeMask`

**Risk:** low. Verify against the real GGUF dump in this doc.

### Phase 2 — CPU GeluTanhMul + ScaleInPlace + SoftcapInPlace kernels

Independent of model wiring; can land standalone with unit tests.

- `src/SharpInference.Cpu/SimdKernels.cs` — `GeluTanhMul(gate, up, n)` (AVX2 + scalar fallback for parity tests), `ScaleInPlace(x, n, scale)`, `SoftcapInPlace(x, n, cap)`
- Tests: `SimdKernelsGeluTanhMulTests.cs` (AVX2 vs scalar parity + numerical properties)

**Risk:** low. Pure kernel work, well-isolated.

### Phase 3 — CPU ForwardPass: gemma4 trunk (high-risk integration)

- Per-layer head_dim refactor — replace `_qDim/_kvDim` scalars with per-layer arrays
- Embedding scale after lookup (`× sqrt(2560) ≈ 50.59`)
- Post-attn RmsNorm before residual add
- Post-FFN RmsNorm before residual add
- Per-layer `layer_output_scale` scalar multiply
- Dual-RoPE table (build both 10K + 1M tables, select per layer)
- Per-layer windowed attention (`kStart = max(0, position+1-window)` when SWA)
- KV-share dispatch (skip K/V proj + cache.Append; route Attention to source layer's pages)
- GeluTanhMul replacing SiLuMul
- Final logit softcap

PLE deferred to Phase 4 (additive — output is at least non-garbage without it, but won't match llama.cpp).

**Risk:** high. Per-layer head_dim variance is a load-bearing refactor; KV-share invariants in PagedKvCache will need care.

### Phase 4 — CPU PLE injection

- Load `per_layer_token_embd` (4.2 GB Q8_0) via mmap zero-copy, NEVER to GPU
- Load `per_layer_model_proj` (BF16 → F32 dequant on first use)
- Per-token: `proj_per_layer = per_layer_model_proj @ hidden / sqrt(2560)` → reshape `[42, 256]` → norm each slice → add PLE-row slice → `× 1/sqrt(2)`
- Per layer, after post-ffn-norm + residual: `cur = inp_gate[layer] @ hidden → GeluTanh → × proj_per_layer[layer] → proj[layer] @ → post_norm[layer] → AddInPlace`

**Risk:** high. Confirm tensor row layout vs column layout before wiring.

### Phase 5 — PagedKvCache per-layer alias + window

Refactor `PagedKvCache` so each layer can be (a) aliased to another layer's pages or (b) ring-buffered to `window` positions only.

- Breaks the "all layers share slot index" invariant — must update all four call-sites: `Forward`, `PrefillWithCache`, `BatchPrefill`, `BatchPrefill2`, `BatchForwardMulti`

**Risk:** high. Pure refactor; tests are the safety net.

### Phase 6 — CUDA arch plumbing (inert)

- `src/SharpInference.Engine/CudaForwardPass.cs:268-313` — per-layer KV cache allocation with per-layer window + alias-mode (track aliased layers to skip double-free at Dispose)
- `src/SharpInference.Engine/CudaForwardPass.cs:152-216` — load PLE refs as CPU-resident (existing `_cpuEmbedding` precedent for large tables)
- `src/SharpInference.Engine/TierPlanner.cs:32-47` — PLE on CPU unconditionally
- `src/SharpInference.Engine/CudaForwardPass.cs:1215-1244` — EstimateMaxContext: subtract PLE bytes; size per-layer KV by `min(maxSeqLen, window)`

**Risk:** medium. No behaviour change; safe to commit before kernel work.

### Phase 7 — CUDA kernels

- `llm_gelu_tanh_mul` — replace SiLU body with tanh-GELU
- `llm_attention_swa` — clone attention with `window_start/window_end` loop bounds (keep 4096 shared-score fast path when window ≤ 4096)
- `llm_softcap_inplace` — `x[i] = tanhf(x[i]/cap) * cap`
- Add to ForceEagerJit and GetKernelFunc resolution

**Risk:** medium. NVRTC kernel work, parity-testable in isolation.

### Phase 8 — CUDA Forward integration end-to-end

Tie everything together in `CudaForwardPass.Forward`: embedding scale, post-attn-norm, per-layer SWA-vs-full Attention dispatch, KV-source-layer skip+alias, post-ffn-norm, per-layer `layer_output_scale`, dual-RoPE per layer, PLE gather (CPU → upload pinned → on-GPU matmul/norm/add), GeluTanhMul, final SoftcapInPlace.

**Acceptance:** greedy-decode parity vs llama.cpp for 32 tokens on "The capital of France is".

**Risk:** high. Integration commit; lots of moving parts.

### Phase 9 — Server + CLI smoke

`-g -1` and `/v1/messages` are already arch-agnostic. The only risk is silent JinjaChatTemplate parse failure (`GgufTokenizer.cs:196-200` swallows exceptions). Add a smoke test asserting non-empty coherent output and verify the GGUF's `tokenizer.chat_template` renders correctly.

## Honest scope estimate

Phases 0+1+2 (kernels + hparams + CPU helper kernels) are the **realistic single-session scope** for me as an autonomous agent. They land foundation work without touching any forward-pass integration, are well-isolated, and have clean unit tests.

Phases 3–8 require the load-bearing forward-pass refactor + iterative parity debugging against llama.cpp — that loop is sequential and judgment-heavy, and would consume many hours of human-level engineering to land cleanly.

Phase 9 server+CLI smoke is final acceptance, only meaningful after Phase 8 works.

## Continuation strategy

After Phase 0+1+2 land + are tested + committed:
1. Capture llama.cpp greedy reference (8–32 tokens) for "The capital of France is" — checked into `tests/fixtures/gemma4_e4b_greedy.json`
2. Start Phase 3 in a fresh session with this plan + reference fixture as input
3. Validate Phase 3 CPU coherence before touching CUDA (Phase 6+)
4. Each phase produces a green commit; rollback boundary at every phase

## Reference

- Issue #82 epic (sub-issues #83–#90)
- Real GGUF header dump (in this document)
- llama.cpp gemma4 build function reference (in research workflow output `wjt8c4nga.output`)
