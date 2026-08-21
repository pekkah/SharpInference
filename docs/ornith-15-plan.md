# Ornith-1.5 Support Plan

**Status:** research + plan (no code changes yet). Written 2026-08-21.
**Upstream:** [ornith-ai](https://huggingface.co/ornith-ai) · MIT · released 2026-08-18 ·
[blog](https://ornith.ai/ornith_1_5.html)

## Executive Summary

Ornith-1.5 is **not a new architecture, and not even a new architecture *variant***.
It is a second-generation self-improvement RL post-train of the same Qwen3.5 bases
Ornith-1.0 was built on: the `config.json` of `ornith-ai/Ornith-1.5-9B` and
`deepreinforce-ai/Ornith-1.0-9B` agree on every architectural field (arch class,
layer types, head geometry, RoPE, GDN dims, vocab, MTP). New weights, same graph.

Consequence: **all three text variants run on SharpInference today with zero code
changes**, on exactly the paths issue #411 validated for Ornith-1.0. The genuinely
new thing in 1.5 is that `ornith-ai` publishes **first-party GGUF *and* `mmproj`
vision projectors** for every size — so the Qwen3.5 vision tower, deferred as
out-of-scope in #411, is now reachable in GGUF form for the first time.

So the work splits cleanly:

* **Small (days):** plumbing + validation for the three text variants and the MTP
  community quants. No new kernels.
* **Large (weeks, separate issue):** the `qwen3_5` vision projector + interleaved
  M-RoPE, which is the only real engineering in the family.

## Variant Matrix

| Variant | HF arch | GGUF arch (expected) | SharpInference path | Q4_K_M | Verdict |
|---|---|---|---|---|---|
| **Ornith-1.5-9B** | `qwen3_5` | `qwen35` | `HybridGdnForwardPass` / `CudaHybridGdnForwardPass` / `VulkanHybridGdnForwardPass` — 24 GDN + 8 full-attn of 32 | 5.63 GB | **Supported today.** Primary target; `-g -1` fits 8 GB VRAM (measured for 1.0-9B, same shape) |
| **Ornith-1.5-35B-A3B** | `qwen3_5_moe` | `qwen35moe` | same hybrid GDN path + MoE FFN, incl. `--cpu-moe` expert offload | 21.7 GB | **Supported today**, unvalidated on real weights (inherits #411's open 35B item) |
| **Ornith-1.5-397B** | `qwen3_5_moe` | `qwen35moe` | same, but only via `SharpInference.Pipeline` VRAM→RAM→NVMe tiering | 240 GB (single file) | Supported *in principle*; not a practical target on any dev machine here |
| MTP community quants (`protoLabsAI/Ornith-1.5-9B-MTP-GGUF`, `mudler/…-APEX-MTP-GGUF`, `SC117/…-MTP-APEX-GGUF`) | — | `qwen35`/`qwen35moe` + `nextn_predict_layers=1` | `MtpDecoder` (`--mtp`) | Should work unmodified; unvalidated |
| **Vision** (`mmproj-Ornith-1.5-*-BF16.gguf`, ~900 MB each) | `qwen3_5_vision` | `clip` | — | — | **Not supported.** See Phase 4 |

Official quants per size: BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0 + one `mmproj`
(all single-file except the 35B/397B BF16 exports).

## Evidence

### 1.0 vs 1.5 configs are architecturally identical

`ornith-ai/Ornith-1.5-9B/config.json` vs `deepreinforce-ai/Ornith-1.0-9B/config.json`:
both `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`, both carry a
`vision_config` (`qwen3_5_vision`, depth 27, hidden 1152), and every text field
matches — 32 layers, hidden 4096, 16 heads / 4 KV heads, `head_dim` 256,
`intermediate_size` 12288, `full_attention_interval` 4, `layer_types` = 3×
`linear_attention` + 1× `full_attention` repeated, `linear_conv_kernel_dim` 4,
`linear_num_key_heads` 16, `linear_num_value_heads` 32, `linear_*_head_dim` 128,
`partial_rotary_factor` 0.25, `rope_theta` 1e7, `mrope_interleaved` true,
`mrope_section` [11,11,10], `mtp_num_hidden_layers` 1, vocab 248320,
`max_position_embeddings` 262144. The only deltas are `transformers_version`
(5.8.1 → 5.12.1) and the weights themselves.

`Ornith-1.5-35B-A3B`: 40 layers, hidden 2048, 16 heads / 2 KV heads, `head_dim` 256,
256 experts / 8 active, `moe_intermediate_size` 512, `shared_expert_intermediate_size` 512.
**These are exactly the numbers `Ornith10ArchitectureTests.Ornith35BMoe_RoutesToHybridSsmMoEPath`
already pins**, so the existing routing test doubles as a 1.5 test.

`Ornith-1.5-397B`: `qwen3_5_moe`, hidden 4096, `head_dim` 256,
`full_attention_interval` 4 — same family, larger.

### Everything the text path needs already exists

* `ModelGraph.FromGgufMetadata`: `qwen35`/`qwen35moe` are in the NEOX-RoPE set;
  partial RoPE via `{arch}.rope.dimension_count` (64 of 256); hybrid GDN via
  `_sharpi.is_hybrid_ssm` (auto-probed from GDN tensors) or the arch name;
  `LayerTypes` built from `full_attention_interval`; MTP blocks stripped via
  `{arch}.nextn_predict_layers`.
* `HasSharedExpert` is honoured across `ForwardPass` / `HybridForwardPass` /
  `TierPlanner` — the 35B's `shared_expert_intermediate_size` is covered.
* Chat template (fetched from `Ornith-1.5-9B/chat_template.jinja`) is the Qwen3.6
  wire format we already handle: `<tool_call><function=…><parameter=…>` XML →
  `QwenToolCallAdapter` (registered for both `qwen35` and `qwen35moe`),
  `<|im_start|>`/`<think>` reasoning with an `enable_thinking` switch → our
  `ChatTemplate` / `--no-thinking` plumbing. The generation prompt pre-opens
  `<think>\n`, which #411 already exercised on 1.0.
* Recommended sampling (temp 0.6–1.0, top_p 0.95, top_k 20, min_p 0,
  presence_penalty 1.5 for general tasks) maps 1:1 onto `Sampler`'s existing
  `PresencePenalty` / `FrequencyPenalty` / `RepetitionPenalty` knobs.

### What could NOT be verified here

The session's egress policy blocks `huggingface.co` for direct download
(`CONNECT … 403`), so the real GGUF **metadata was not inspected** — the
`general.architecture` values in the matrix above are inferred from the identical
1.0 configs plus #411's confirmation that the real 1.0-9B GGUF reports
`general.architecture = qwen35` with GDN tensors present. Phase 1 starts by
confirming this with `list-metadata` on a downloaded file. Two specific things to
look for: an arch string other than `qwen35`/`qwen35moe` (if the converter now
tags VL-wrapped checkpoints differently), and `*.rope.mrope_sections` keys.

## Gaps

| # | Gap | Impact | Phase |
|---|---|---|---|
| G1 | No download presets / docs / tests for 1.5 | discoverability only | 0 |
| G2 | Text path unvalidated on real 1.5 weights | unknown-unknowns in the GGUF header | 1–2 |
| G3 | MTP community quants unvalidated | no self-speculative speedup | 3 |
| G4 | **`qwen3_5` vision projector unimplemented** — `VisionModel` hard-fails on any `projector_type` ≠ `gemma4uv` | image input impossible | 4 |
| G5 | **Interleaved M-RoPE unimplemented** — `ModelGraph` explicitly notes MROPE/IMROPE are unsupported | blocks G4 (text-only is unaffected: with equal t/h/w positions M-RoPE degenerates to the plain NEOX RoPE we already run) | 4 |
| G6 | No YaRN RoPE scaling | can't reach the 1M-token window the card documents (`factor: 4.0` over 262144); the native 262K window works | 5 |
| G7 | 397B is a 240 GB single file | needs `Pipeline` tiering + NVMe; untested at this scale | 5 |

## Phased Plan

### Phase 0 — Plumbing (½ day, no weights needed)

* `scripts/download-model.ps1`: add `ornith15-9b` (Q4_K_M, 5.63 GB) and
  `ornith15-35b` (Q4_K_M, 21.7 GB) presets pointing at `ornith-ai/*-GGUF`
  (extend the `ValidateSet`, set `SizeGB` so the free-disk guard fires); optionally
  `ornith15-9b-mmproj` once Phase 4 lands.
* `tests/SharpInference.Tests.Core/Ornith10ArchitectureTests.cs` → rename to
  `OrnithArchitectureTests`, and fix the 9B fixture: it currently uses synthetic
  48 layers / 32 heads / `key_length` 128, whereas the real 9B is 32 layers /
  16 heads / 4 KV / `key_length` 256 / `rope.dimension_count` 64. Add a 1.5-9B
  case with the real numbers and assert `LayerTypes` is 24 GDN + 8 attention.
* `CLAUDE.md` + `.claude/skills/run-models/SKILL.md`: fold 1.5 into the Ornith
  section (same path as 1.0, new org/URLs, note vision still unsupported).

**Done when:** `dotnet test tests/SharpInference.Tests.Core --filter Ornith` green.

### Phase 1 — Validate 9B on real weights (1 day, needs network + GPU)

1. `list-metadata` / `list-tensors` on `Ornith-1.5-9B-Q4_K_M.gguf` — confirm
   `general.architecture`, `qwen35.nextn_predict_layers`, GDN tensor presence, and
   whether any `mrope`/`imrope` keys appear (record the dump in this doc, as
   `docs/qwen35moe-plan.md` does).
2. Greedy cross-check vs llama.cpp at `--temp 0` (`parity-check` skill, level 2),
   then a perplexity gate (level 4).
3. Tool-call round trip through `QwenToolCallAdapter`, and a server smoke test with
   `enable_thinking:false`.

**Done when:** greedy tokens match llama.cpp on ≥1 prompt and a tool call executes end to end.

### Phase 2 — Validate 35B-A3B (1 day, needs a big-RAM box)

Same as Phase 1 on the MoE path, plus `--cpu-moe` expert offload. This also closes
the 35B item left open in #411.

### Phase 3 — MTP quants (1–2 days)

Run `protoLabsAI/Ornith-1.5-9B-MTP-GGUF` with `--mtp`. `HybridGdnForwardPass`
loads the head when `NumMtpLayers > 0` and the `blk.{NumLayers}.nextn.*` tensors
exist, so the expectation is zero code. Verify the acceptance rate and that
`GdnStateCache` rollback behaves; if a community quant lays the MTP block out
differently (APEX repacks), document rather than special-case.

### Phase 4 — Qwen3.5 vision (3–4 weeks; **file as its own issue**)

The only real engineering. Sub-steps, CPU-first:

* **4a — mmproj loader.** Generalize `VisionModel` from its single hardcoded
  `gemma4uv` projector into a projector-type dispatch, and read the `qwen3_5`
  ViT config (`clip.vision.*`). BF16 tensors are already handled by
  `Dequantize.DequantBF16`.
* **4b — preprocessing.** Reuse `ImagePreprocessor.CalcSizePreservedRatio` with
  `align = patch_size × spatial_merge = 32` (it already mirrors llama.cpp's Qwen
  smart-resize), swap in the Qwen normalization constants, and duplicate frames for
  `temporal_patch_size = 2` on stills.
* **4c — ViT forward.** 27 layers, hidden 1152, 16 heads, intermediate 4304,
  `gelu_pytorch_tanh`, full (non-causal) attention, learned position table of 2304
  (48×48) interpolated to the actual grid. Note `deepstack_visual_indexes` is
  **empty**, so the Qwen3-VL deepstack multi-level merge is *not* needed — a plain
  ViT. Build on `SimdKernels`; GPU after CPU parity.
* **4d — merger.** 2×2 spatial merge → MLP → `out_hidden_size` (4096 for 9B,
  2048 for 35B), producing soft tokens for `<|image_pad|>` slots.
* **4e — interleaved M-RoPE (the hard part).** Per-token (t,h,w) position triples
  with sections [11,11,10] over the 64 rotated dims. Touches position bookkeeping
  in the KV cache and the rope kernels in **all four** forward passes (CPU,
  Vulkan, CUDA, and the hybrid GDN siblings) — per CLAUDE.md, a numeric change in
  one needs the siblings updated. Gate it on image tokens actually being present so
  the text path stays bit-identical.
* **4f — prompt plumbing.** `<|vision_start|><|image_pad|><|vision_end|>` expansion
  in the template renderer, CLI `--image`/`--mmproj` wiring (already exists for
  Gemma 4), server image content parts.

**Done when:** soft-token embeddings match llama.cpp `mtmd` on a fixed image within
tolerance, and a caption smoke test reads correctly.

### Phase 5 — Optional / deferred

* **YaRN** (G6) — only if someone wants >262K context.
* **397B** (G7) — 240 GB Q4_K_M through the `Pipeline` tiering; treat as a
  scale experiment, not a supported configuration.

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | GGUF arch string isn't `qwen35`/`qwen35moe` (converter may tag VL-wrapped checkpoints differently) | Phase 1 step 1 is `list-metadata`; everything else is contingent on it |
| R2 | Text GGUF carries M-RoPE metadata we silently ignore | Harmless while positions are equal across t/h/w, but confirm in Phase 1 and assert in a test |
| R3 | Phase 4e's rope change regresses text-only numerics across 4 forward passes | Gate on image presence; parity-check level 2 before/after on a text-only prompt |
| R4 | MTP community repacks (APEX) deviate from the `nextn.*` layout | Document and skip rather than special-case |
| R5 | No local llama.cpp reference for the `qwen3_5` vision projector | Confirm the `tools/mtmd` implementation exists upstream before starting Phase 4 |

## Recommendation

Phases 0–3 are cheap and mostly validation — do them as one issue ("Ornith-1.5:
validate end-to-end", mirroring #411). Phase 4 is a Qwen3.5-vision project that
happens to be motivated by Ornith; file it separately so it can be scheduled
against issue #126 (Gemma 4 vision) rather than blocking 1.5 text support.
