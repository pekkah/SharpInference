# Gemma 4 12B (dense `gemma4_unified`) — CUDA / CUDA-Hybrid Implementation Plan

Status: planning, no code written yet. Targets the **dense 12B** member of the Gemma 4
family released 2026-06-03. CUDA-Hybrid + full-CUDA are the first iteration; CPU is second,
Vulkan last.

> **TL;DR** — Most of the work is already done. The merged `gemma4` path (issue #82 epic,
> built and validated against **E4B**) is metadata/tensor-driven and already implements the
> hard parts the 12B needs: per-layer head-dim variance, dual-RoPE, SWA, shared-KV layers,
> post-attn/post-ffn norms, GeLU-tanh FFN, final-logit softcap, and proportional-RoPE
> (`rope_freqs`). The dense 12B differs from E4B mainly by what it **omits** (no PLE), plus
> a larger sliding window and a possible architecture-string change. The bulk of this plan is
> therefore **validation of the never-exercised dense (no-PLE) code path** and decode parity,
> not new kernels.

---

## 1. GGUF availability — confirmed

Multiple GGUF conversions of `google/gemma-4-12B-it` already exist (verified 2026-06-04 via HF Hub):

| Repo | Notes |
|---|---|
| `unsloth/gemma-4-12b-it-GGUF` | Q4_K_M / Q5_K_M / Q8_0; tagged `gemma4`, `gemma4_unified` |
| `lmstudio-community/gemma-4-12B-it-GGUF` | LM Studio community conversions |

**Primary iteration-1 target: `gemma-4-12b-it-Q4_K_M.gguf`** (~7.3 GB) — fits the 12 GB VRAM
target (RTX 3060/4070) with full-GPU offload plus KV at a practical 8K–32K context. Q4_K_M
matvec kernels already exist in the CUDA backend (the workhorse `llm_matvec_q4_k_m` path), so no
new quant kernel is required for iteration 1. Higher quants are follow-ups: Q5_K_M (~8.4 GB)
still fits the GPU; Q8_0 (~12.5 GB) needs CUDA-Hybrid layer offload (or a 16 GB card) and the
native Q8_0 CUDA matvec kernel landed for E4B.

> ⚠️ **Network note:** `huggingface.co` is **not** in this environment's egress allowlist
> (`curl` → `403 host_not_allowed`). The GGUF must be fetched on a machine with HF access and
> the header dumped there; the parity work in Phases 3–4 below requires the real file locally.

---

## 2. Architecture research — Gemma 4 12B

Gemma 4 is a family: on-device `E2B`/`E4B` (effective-param, MatFormer + PLE), and the dense
`12B` / `26B-A4B` (MoE) / `31B` consumer-GPU tier. The 12B is **dense**, `gemma4_unified`
(encoder-free multimodal), 256K context, ~11.96 B params. Its LLM trunk is "rather similar to
the 31B dense model."

### Decoder trunk (text) — what matters for GGUF inference

| Property | Gemma 4 dense (12B) | Already handled in repo? |
|---|---|---|
| Norm | RMSNorm, ε from metadata | ✅ |
| Pre + post attention norm, pre + post FFN norm (4 norms/layer) | yes | ✅ `HasPostAttnNorm` / `HasPostFfwNorm` |
| QK-norm (per-head RMSNorm on Q and K) | yes | ✅ `HasQkNorm` |
| FFN activation | **GeLU-tanh (GeGLU)** | ✅ `FfnActivation.GeluApprox` → `GeluTanhMul` |
| Embedding scale | × √hidden | ✅ `EmbeddingScale = sqrt(embDim)` |
| Hybrid attention | 5 sliding-window : 1 global, **final layer always global** | ✅ `IsSwaLayer[]` from `sliding_window_pattern` |
| Sliding window | **1024** (12B/26B/31B) vs 512 (E2B/E4B) | ✅ metadata-driven; CUDA SWA fast path covers window ≤ 4096 |
| Per-layer head-dim | **256 on SWA layers, 512 on global** | ✅ `LayerHeadDim[]` / `LayerRopeDim[]` |
| GQA | 8 query / 4 KV heads (pattern; confirm counts from GGUF) | ✅ metadata-driven |
| Dual RoPE base | 10 000 (SWA) / 1 000 000 (global) | ✅ `RopeTheta` / `RopeThetaSwa` |
| Proportional RoPE (p-RoPE, p≈0.25) on global layers | high freqs rotated, low-freq pairs frozen via `rope_freqs.weight` | ✅ `_gpuRopeFreqs`, applied to global layers only |
| Shared-KV tail layers | yes (`shared_kv_layers`) | ✅ `KvSourceLayer[]` |
| Final-logit softcap | 30.0 | ✅ `FinalLogitSoftcap` + `SoftcapInPlace` |
| **Per-Layer Embeddings (PLE)** | **NOT present** (dense) — `per_layer_*` tensors omitted | ⚠️ gated on `HasPerLayerTokenEmbd`, but the **false** branch is untested |
| **`layer_output_scale`** | likely absent on dense (confirm) | ⚠️ gated on `HasLayerOutputScale`; false branch untested |
| Vocab / tokenizer | 262 144 SentencePiece, Gemma chat template | ✅ existing `Gemma4TokenizerTests` / `JinjaChatTemplate` |
| Context length | 262 144 (256K) | metadata-driven; see §5 KV-budget note |

### Multimodal (encoder-free) — explicitly OUT of scope for iteration 1

The 12B projects raw image patches (48×48) and 16 kHz audio (40 ms / 640-float frames)
directly into the embedding space via lightweight linear layers (no ViT/audio encoder). In the
llama.cpp ecosystem these live in a **separate `mmproj` GGUF**; the text GGUF loads and decodes
text-only without it. Iteration 1 = **text-only**. Vision/audio is a follow-up epic.

### Net delta vs. the already-merged E4B path

The dense 12B is, for our engine, **E4B minus PLE minus `layer_output_scale`, plus a larger
sliding window, plus a possible arch-string rename**. There are no new architectural mechanisms.
The risk is entirely in (a) code paths that E4B always exercised with PLE on, and (b) end-to-end
decode parity.

---

## 3. Gap analysis — what actually needs doing

### G1 — Architecture string: `gemma4` vs `gemma4_unified` *(must verify first)*
HF reports `general.architecture = gemma4_unified` for the 12B, but the unsloth GGUF is also
tagged `gemma4`. Our detection is an exact-match allowlist:
- `src/SharpInference.Core/ModelGraph.cs:289` (NEOX-RoPE allowlist) — lists `gemma4`
- `src/SharpInference.Core/ModelGraph.cs:352` — `isGemma4 = arch.Equals("gemma4", …)`
- `src/SharpInference.Engine/CudaHybridForwardPass.cs` — `_isGemma4Like`

**Action:** dump the real 12B GGUF `general.architecture`. If it is `gemma4_unified`, add that
string to (a) the NEOX allowlist, (b) `isGemma4`, (c) `_isGemma4Like`, and the metadata-key
prefix logic so keys resolve as `gemma4_unified.*` (or normalize the arch to `gemma4`). If the
converter already emits `gemma4`, this is a no-op. **Verify, don't assume.**

### G2 — Dense (no-PLE) path has never been exercised
PLE is correctly gated on `HasPerLayerTokenEmbd` in all three forward passes
(`ForwardPass.cs:380/1218/1386/2385`, `CudaForwardPass.cs:580/940/1060/1307/1403`,
`CudaHybridForwardPass.cs:640/1203/1421/1584`), but **every existing gemma4 test uses E4B**,
which always has PLE. The 12B will take the `false` branch end-to-end for the first time.
**Action:** add dense-config unit + forward-pass tests; walk each `HasPerLayerTokenEmbd` /
`HasLayerOutputScale` branch to confirm the no-PLE / no-scale trunk produces the bare
residual stream (embed → attn → ffn → norm → logits + softcap) with no dangling PLE buffers.

### G3 — `layer_output_scale` absence
Gated on `HasLayerOutputScale` (`ForwardPass.cs:307`, `CudaForwardPass.cs:456`,
`CudaHybridForwardPass.cs:630`). Confirm the dense GGUF omits `blk.*.layer_output_scale.weight`
and that the false branch is a clean skip (no `× 1.0` no-op cost in the hot loop is fine).

### G4 — Sliding window = 1024
Larger than E4B's 512. Pure metadata (`gemma4.attention.sliding_window`). The CUDA `AttentionSwa`
kernel keeps its shared-score fast path for `window ≤ 4096`, so 1024 is covered. **Action:**
confirm no place hard-codes 512; add a 1024-window SWA parity test.

### G5 — Proportional RoPE for the dense global layers
`rope_freqs.weight` handling already exists (`CudaForwardPass.cs:559-568`, applied to global
layers only). p-RoPE (p≈0.25) is encoded in that table by the converter. **Action:** confirm the
dense GGUF ships `rope_freqs.weight` sized `maxHeadDim/2` and that global-layer RoPE matches
llama.cpp (this is the single most likely source of slow positional drift — validate at long
positions, not just position 0).

### G6 — KV / context budget at 256K
`EstimateMaxContext` must account for: no PLE table to subtract (frees ~GBs vs E4B), per-layer KV
sized by `min(maxSeqLen, window)` for SWA layers, full length for global layers, and shared-KV
tail aliasing. **Action:** sanity-check VRAM estimate so a Q4_K_M 12B + reasonable context fits
12–16 GB; expose the usual context cap.

### G7 — Tokenizer / chat template / EOG
Same Gemma tokenizer + template family as E4B. **Action:** smoke-test that the 12B GGUF's
embedded `tokenizer.chat_template` renders via `JinjaChatTemplate` and EOG token ids resolve
(reuse `Gemma4TokenizerTests` / `EogTokenIdsTests` patterns). `GgufTokenizer.cs` swallows
template-parse exceptions, so assert non-empty coherent output rather than trusting silence.

---

## 4. Phased plan (CUDA-Hybrid + CUDA first)

Ordering follows the user's priority: **CUDA-Hybrid & CUDA → CPU → Vulkan.** Because the trunk
already exists, phases are validation-and-fill rather than greenfield.

### Phase 0 — Acquire + dump the real 12B GGUF header  *(prerequisite, off-box)*
On a host with HF access, fetch `gemma-4-12b-it-Q4_K_M.gguf`, dump `general.architecture`, all
`gemma4*.{block_count,embedding_length,feed_forward_length,attention.*,rope.*}` keys, the
`sliding_window_pattern` bool[], and the full tensor inventory (confirm **absence** of
`per_layer_*` and `layer_output_scale`, **presence** of `rope_freqs.weight`). Check this dump
into `tests/fixtures/gemma4_12b_header.md`. **Everything below is gated on this ground truth.**
*Risk: low (data-gathering). Blocker: needs HF egress.*

### Phase 1 — Architecture-string + metadata wiring (G1)
Make `gemma4_unified` (if that's the GGUF arch) a first-class alias of `gemma4` in the NEOX
allowlist, `isGemma4`, `_isGemma4Like`, and metadata-prefix resolution. Add
`Gemma4_12B_Dense_PopulatesAllFields` to `Tests.Core` asserting the dumped metadata maps to
`ModelHyperparams` with `HasPerLayerTokenEmbd == false`, `IsSwaLayer` length == block_count,
`LayerHeadDim` = {256 SWA, 512 global}, `SlidingWindowSize == 1024`, softcap 30, dual RoPE bases.
*Risk: low. Pure parsing; verifiable against the Phase 0 dump.*

### Phase 2 — Dense (no-PLE) CUDA + CUDA-Hybrid decode (G2/G3/G4/G5)
No new kernels expected. Walk the `HasPerLayerTokenEmbd == false` / `HasLayerOutputScale == false`
branches in `CudaForwardPass` and `CudaHybridForwardPass`:
- embed → `× √hidden` (no PLE pre-pass) → per-layer trunk → `output_norm` → logits → softcap;
- per-layer SWA-vs-global dispatch with window 1024, dual-RoPE + `rope_freqs` on global layers,
  shared-KV tail aliasing, post-attn/post-ffn norms, GeLU-tanh FFN;
- TierPlanner: with no PLE table, allow the full trunk on GPU when VRAM permits; verify the
  hybrid CPU/GPU split still produces bit-comparable results to full-CUDA.
*Risk: medium. The mechanisms exist; the untested combination is the no-PLE trunk + tier split.*

### Phase 3 — CUDA / CUDA-Hybrid decode parity vs llama.cpp  *(acceptance gate)*
Capture llama.cpp greedy reference (32 tokens, temp 0) for a fixed prompt
(`"The capital of France is"`), check into `tests/fixtures/gemma4_12b_greedy.json`. Assert:
- full-CUDA `-g -1` greedy == reference;
- CUDA-Hybrid greedy == reference;
- **long-position** check (decode past the 1024 window and past a few global layers) to catch
  SWA-boundary and p-RoPE drift, which position-0 tests miss.
*Risk: high — this is the judgment-heavy loop. Most likely failure points: G1 arch keys, G5
global-layer p-RoPE, SWA window boundary, shared-KV aliasing on the no-PLE trunk.*

### Phase 4 — Server + CLI smoke (G7)
`-g -1` and `/v1/messages` are arch-agnostic. Add a server smoke (mirror
`scripts/smoke-gemma4-server.ps1`) asserting coherent non-empty output and correct chat-template
render + EOG stop for the 12B. *Risk: low.*

### Phase 5 — CPU backend parity *(second priority)*
Exercise the dense no-PLE branch in `ForwardPass.cs` (CPU). Same parity fixture as Phase 3. The
CPU gemma4 trunk already exists (GeluTanhMul, dual-RoPE tables, SWA, post-norms); this is
validation + any no-PLE-specific fixes. *Risk: medium.*

### Phase 6 — Vulkan backend *(last priority)*
Bring `GpuForwardPass` (Vulkan) to gemma4 dense parity: confirm SPIR-V shaders exist for
GeLU-tanh FFN, SWA, dual-RoPE, softcap (some may already be present from E4B), fill gaps, parity
test. *Risk: medium-high; deferred by design.*

### Phase 7 (future epic) — Encoder-free multimodal (vision + audio)
Separate `mmproj` GGUF load + linear patch/audio projection into the embedding stream. Out of
scope for this plan; tracked separately.

---

## 5. Open questions to resolve from the Phase 0 dump
1. Exact `general.architecture` string — `gemma4` or `gemma4_unified`? (drives G1 scope)
2. `block_count`, `embedding_length`, `feed_forward_length`, head counts — fill the §2 table.
3. Confirm `per_layer_*` tensors are **absent** and `layer_output_scale` absent (dense).
4. Confirm `rope_freqs.weight` present and its length (`maxHeadDim/2` ⇒ 256).
5. `sliding_window == 1024`? `sliding_window_pattern` length == block_count, final entry global?
6. `shared_kv_layers` count for the dense trunk (drives KV aliasing layout).
7. Does the text GGUF embed any multimodal projection tensors, or is `mmproj` fully separate?

## 6. Honest scope estimate
- **Phases 0–1** (acquire + arch/metadata wiring + dense hparam test): single-session, low risk,
  but **blocked on HF egress** for the real header.
- **Phases 2–3** (dense CUDA/Hybrid decode + llama.cpp parity): the real work; sequential,
  judgment-heavy parity debugging. Plan for iterative loops against the reference fixture.
- **Phases 4–6** (server smoke, CPU, Vulkan): incremental, each a green-commit boundary.
- **Phase 7** (multimodal): separate epic.

## 7. References
- Existing E4B plan: `docs/gemma4-e4b-implementation-plan.md`
- Gemma 4 12B announcement (2026-06-03): developers.googleblog.com / blog.google
- A Visual Guide to Gemma 4 12B — newsletter.maartengrootendorst.com
- GGUF: `unsloth/gemma-4-12b-it-GGUF`, `lmstudio-community/gemma-4-12B-it-GGUF`
- llama.cpp `gemma4` arch (proportional-RoPE via large `rope_freqs` tail dims)
