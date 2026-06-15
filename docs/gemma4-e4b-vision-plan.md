# Gemma 4 E4B Multimodal (Vision) — Research & Implementation Plan

Status: **research / planning, no code written yet.** Tracked by **issue #126**.

> ## ⚠️ Verification update (2026-06-15) — architecture confirmed from the real mmproj
> The §1 "verification debt" is now **retired**: the E4B mmproj header was dumped
> (`E:\models\gemma-4-E4B-it-mmproj.gguf`, ~992 MB). **The Gemma-3n / MobileNet-V5 assumption
> below is WRONG.** What E4B actually ships:
> - **Vision** `clip.vision.projector_type = gemma4v` — a **transformer ViT** encoder (NOT a
>   conv MobileNet): `block_count=16`, `embedding_length=768`, `head_count=12` (head_dim 64,
>   with QK-norm), GeGLU FFN (`feed_forward_length=3072`), conv patch-embed `v.patch_embd.weight
>   [16,16,3,768]`, learned 2D position table `v.position_embd.weight [768,10240,2]`, `image_size=224`,
>   `patch_size=16`, image_mean=0/std=1.
> - **Audio** `clip.audio.projector_type = gemma4a` — a separate ~12-block conformer-style encoder
>   (`a.*` tensors, `num_mel_bins=128`); `clip.has_audio_encoder=True`.
> - Projectors: `mm.input_projection [768→2560]` (vision) and `mm.a.input_projection [1536→2560]`
>   (audio), into the E4B text embed dim 2560.
>
> **Net:** the plan's *conclusion* (E4B is encoder-FULL and needs a real encoder forward pass,
> unlike the 12B) holds; the *specifics* (MobileNet-V5 conv stack, gemma3n, 768² input, 256 fixed
> tokens, `<start_of_image>` markers) do not. The good news: a ViT reuses our existing
> attention/MLP/RMSNorm kernels almost directly — this is the plan's §4 "SigLIP fallback" path,
> which turns out to be the actual architecture. Phase V2 should target the `gemma4v` ViT, not
> MobileNet-V5; rewrite §1/§2/§3 hyperparameters against the header facts above before coding.
>
> **This is NOT the 12B path.** The Gemma 4 **12B** uses encoder-free `gemma4uv` (raw patches →
> linear projection, no ViT) and is implemented in `src/SharpInference.Vision` (issue #250, see the
> gemma4uv section of `docs/SharpInference-Design.md`). E4B (`gemma4v`+`gemma4a`) remains unimplemented.

This doc scopes adding **image input** to the already-working Gemma 4 E4B text path. Audio (the
other E-model modality) is noted but deferred. It is the multimodal counterpart to
`docs/gemma4-e4b-implementation-plan.md` (whose *text* phasing is now stale — the gemma4 text
trunk is implemented in `ForwardPass.cs`: embedding scale, PLE, dual-RoPE, SWA, cross-layer
KV-share, GeGLU, final-logit softcap are all present).

> **This plan is provisional and expected to change.** The vision hyperparameters and the encoder
> graph below are reconstructed from the Gemma-3n lineage and llama.cpp's `clip`/`mtmd` convention,
> **not** from a dumped E4B `mmproj` header (network policy blocked the binary pull while drafting).
> Phase V0 retires that verification debt; the later phases will be revised once the real model is
> inspected. Treat the structure as a direction, not a contract.

## TL;DR

- Gemma 4 is **natively multimodal** (all sizes: text+image; E2B/E4B add audio). The user's
  instinct was right.
- SharpInference is currently **text-only for LLM inference** — vision/audio were explicitly
  declared out of scope in issue #82 ("vision/audio encoders — text-only GGUF weights run
  standalone"). Zero vision code exists in `src/` today.
- **Key architectural finding:** the Gemma 4 **E-models (E2B/E4B) use a MobileNet-V5-300M
  convolutional vision encoder** (the Gemma-3n lineage), **not** the SigLIP ViT that Gemma 3's
  big models (and Gemma 4 26B/31B) use. This is good for us — a conv encoder maps onto the
  `IImageOpsBackend.Conv2d` infrastructure we already built for the diffusion/RRDBNet pipeline —
  but MobileNet-V5 is an unusual, conv-heavy architecture (inverted residuals, depthwise-separable
  convs, mobile MQA blocks) rather than a plain transformer.
- llama.cpp supports Gemma 4 E4B image input from day one (`llama-mtmd-cli` + an `mmproj` GGUF),
  so we have a **reference implementation to parity-debug against**.

## 1. How Gemma 4 multimodal works (the parts we must replicate)

A multimodal Gemma model is **two GGUF files**:

1. the text model we already load (`-m ...gemma-4-E4B-it-*.gguf`), and
2. a **multimodal projector** `mmproj-*.gguf` — the **vision encoder + projector** weights.
   (Available alongside the text GGUF, e.g. `ggml-org/gemma-4-E4B-it-GGUF`,
   `unsloth/gemma-4-E4B-it-GGUF`.)

Pipeline (image → text-model input):

1. **Preprocess** the image: decode → RGB float → resize to the encoder's fixed input
   (Gemma-3n/E-model: **768×768**, *to confirm against the E4B mmproj header*) → normalize.
   Optionally **Pan & Scan**: tile wide/tall images into extra crops + a global thumbnail, each
   encoded independently.
2. **Vision encoder** (MobileNet-V5-300M): conv stem → inverted-residual / depthwise-separable
   blocks → multi-scale feature fusion → a feature map that is pooled/projected to a fixed budget
   of **256 soft vision tokens per image/crop** (the Gemma-3n number; *confirm for E4B*).
3. **Projector MLP** (`mm.*` tensors): maps vision features into the **text embedding dim**
   (E4B `embedding_length` = 2560).
4. **Splice** the 256 embeddings into the token sequence: the chat template emits placeholder
   image tokens wrapped in `<start_of_image>` / `<end_of_image>`; those placeholder positions are
   **overwritten with the projected vision embeddings** (fed as raw input `embd`, not token IDs).
   The image then occupies 256 real positions in the KV cache; position IDs advance by 256.
5. The combined sequence flows through the **existing** gemma4 text decoder unchanged, except for
   attention masking (below).

**Attention over image tokens:** in the Gemma family, image soft-tokens attend **bidirectionally
within their own span**, while text remains causal. This interacts with `PagedKvCache` and the
causal mask and needs explicit handling (build a causal mask that is bidirectional inside each
image's 256-token block). *Confirm Gemma-4-E behavior matches Gemma-3 here.*

> **Verification debt:** image size (768²), exact soft-token count (256), normalization
> constants, projector type string, and the bidirectional-mask rule are stated from the Gemma-3n
> lineage. They MUST be confirmed against a real E4B `mmproj` GGUF header dump and the llama.cpp
> `clip.cpp` / `mtmd` gemma3n path before Phase V2 coding. We could not dump the mmproj binary in
> this session (network policy blocks direct HF binary pulls).

### mmproj GGUF structure (llama.cpp `clip` convention)

- Metadata: `clip.has_vision_encoder`, `clip.projector_type` (expect `gemma3n`),
  `clip.vision.image_size`, `clip.vision.patch_size`, `clip.vision.embedding_length`,
  `clip.vision.projection_dim`, `clip.vision.block_count`, plus MobileNet-specific keys.
- Vision tensors: `v.*` (patch/stem conv, per-block conv/attn/norm weights, `v.post_ln`).
- Projector tensors: `mm.*` (the projection MLP / `mm.input_projection`).
- Audio (E-models): `a.*` + `clip.has_audio_encoder` — **out of scope for the first pass.**

## 2. What we already have to build on

| Asset | Location | Reuse |
|---|---|---|
| GGUF parser (mmap, multi-shard, metadata) | `Core/GgufModel.cs` | Load the `mmproj` as a 2nd model handle |
| Conv2d / activations / pixel-shuffle / upsample (GPU) | `Core/IImageOpsBackend.cs`; `Cuda/CudaBackend.cs:3621`, `Vulkan/VulkanBackend.cs:1584` | MobileNet-V5 conv stack |
| Conv2D / GroupNorm / LayerNorm / Gelu / resize (CPU) | `Diffusion/DiffusionOps.cs` | CPU vision path + image resize |
| GEMM / RMSNorm / attention / GeLU SIMD kernels | `Cpu/SimdKernels.cs`, backends | Projector MLP, any attn blocks in MobileNet-V5 |
| gemma4 text decoder (PLE, SWA, KV-share, dual-RoPE, softcap) | `Engine/ForwardPass.cs` | Unchanged consumer of spliced embeddings |
| **Embedding entry point** | `ForwardPass.cs:1212` `EmbedToken` call site / `1824` `EmbedToken` wrapper / `1885` `EmbedTokenInto(token, dest)` definition; scale at `:1215` | **The splice seam** — add an overload that writes a precomputed embedding instead of a `token_embd` lookup |

The two toolkits we need — a GGUF transformer runtime and a convolutional image-ops backend —
already exist; they're just siloed (the image ops serve text-to-image diffusion / RRDBNet today).
The vision encoder is essentially "the conv toolkit feeding the transformer toolkit."

## 3. Phased implementation plan

Suggested new module: **`src/SharpInference.Vision`** (mmproj loader, preprocessing, encoder,
projector), keeping vision concerns out of `Core`/`Engine` until the seam is stable.

### Phase V0 — mmproj/clip GGUF loader (low risk)
Parse `clip.*` metadata + `v.*`/`mm.*` tensors into a `VisionModel` handle. Dump the real E4B
mmproj header and **reconcile every assumption in §1** (image size, token count, projector type,
tensor inventory, MobileNet block config). Smoke test: load, resolve all tensors, print config.

### Phase V1 — image preprocessing (low risk)
Decode (PNG/JPEG → RGB), resize to encoder input (reuse `DiffusionOps` bilinear), normalize, and
implement **Pan & Scan** crop generation (cap crops; global thumbnail + crops). Unit-test against
fixed fixtures (deterministic resize/normalize output).

### Phase V2 — MobileNet-V5 vision encoder forward pass (HIGH risk)
The load-bearing piece. Implement the MobileNet-V5-300M graph (conv stem, inverted-residual /
depthwise-separable blocks, mobile-MQA attention blocks, multi-scale fusion) on CPU first using
`DiffusionOps` + new depthwise-conv kernels, then GPU via `IImageOpsBackend`. **Parity-gate each
stage** against llama.cpp's `clip` gemma3n encoder (capture intermediate tensors from
`llama-mtmd-cli`). Risk drivers: depthwise-separable convs and the mobile-attention blocks are not
in our kernel set yet; MobileNet-V5 has architecture quirks (stem bias, GELU-tanh) that bit timm.

### Phase V3 — token reduction + projector MLP (medium risk)
Pool/condense the encoder feature map to 256 tokens and run the `mm.*` projector MLP to the text
embed dim (2560). Reuses MatMul. Parity-gate the 256×2560 output against llama.cpp
`mtmd_get_output_embd`.

### Phase V4 — embedding splice + bidirectional mask (HIGH risk)
- Add `ForwardPass`/`Prefill` support to accept **precomputed input embeddings** at given
  positions (overload of `EmbedTokenInto`; skip `token_embd` lookup; decide embedding-scale
  handling for image rows — text tokens get `× sqrt(2560)`, image embeddings come pre-scaled from
  the projector, *confirm*).
- Build the **causal-except-bidirectional-within-image** attention mask; verify interaction with
  SWA layers and cross-layer KV-share in `PagedKvCache`.
- Chat-template rendering of `<start_of_image>`/`<end_of_image>` + the soft-token placeholders
  (`GgufTokenizer` Jinja path).
- Acceptance: greedy-decode parity vs `llama-mtmd-cli` on a fixed image+prompt (e.g. "describe
  this image") for N tokens.

### Phase V5 — CLI + API surface (medium risk)
- CLI: `--image <path>` (and Pan & Scan toggle) on the existing `Spectre.Console.Cli` frontend.
- Server: image **content blocks** in `/v1/messages` (Anthropic) and `/v1/chat/completions`
  (OpenAI) — base64 / URL image parts → preprocess → encode → splice. Multi-image support.
- Smoke tests in `Tests.Server`.

### (Deferred) Phase V6 — audio
E2B/E4B audio via the `a.*` encoder (USM/conformer). Separate epic; not required for image.

## 4. Risks & de-risking

- **MobileNet-V5 is the long pole** (Phase V2). It is conv-heavy and idiosyncratic; our kernels
  cover standard Conv2d but not depthwise-separable / mobile-MQA yet. Mitigation: stage-by-stage
  tensor parity against llama.cpp; CPU-first before GPU.
- **Verification debt** (§1): confirm all vision hyperparameters against a real E4B mmproj header
  *before* V2.
- **Bidirectional mask × SWA × KV-share** (V4) is a delicate interaction in `PagedKvCache`.
- **Fallback / de-risking option:** if MobileNet-V5 parity proves too costly, the **SigLIP ViT**
  path (Gemma 3 4B, or Gemma 4 26B/31B big models) is a *much* simpler encoder (a plain
  bidirectional ViT that reuses our existing attention/MLP kernels almost directly, 896×896, 14px
  patches, 4×4 pool → 256 tokens). Phases V0/V1/V3/V4/V5 are largely shared; only the encoder
  (V2) differs. A SigLIP PoC would validate the whole splice pipeline end-to-end fastest, then the
  MobileNet-V5 encoder slots in for E4B. (User has chosen E4B-direct; this remains the documented
  fallback if V2 stalls.)

## 5. References

- Issue #82 — Gemma 4 family support (text); vision marked out of scope.
- `docs/gemma4-e4b-implementation-plan.md` — text trunk plan (phasing now stale; trunk landed).
- HF docs: [Gemma 4 (transformers)](https://huggingface.co/docs/transformers/main/model_doc/gemma4),
  [Gemma 3n](https://huggingface.co/docs/transformers/main/model_doc/gemma3n),
  [Welcome Gemma 4 (blog)](https://huggingface.co/blog/gemma4).
- [timm changelog](https://huggingface.co/docs/timm/changes) — "MobileNetV5 backbone … for Gemma
  3n image encoder" (the conv-encoder confirmation).
- llama.cpp: [multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
  [mtmd README + `mtmd.h`](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd),
  [gguf-py `constants.py`](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
  (`clip.*` keys, projector types, `v.*`/`mm.*`/`a.*` tensor names).
- GGUFs: `ggml-org/gemma-4-E4B-it-GGUF`, `unsloth/gemma-4-E4B-it-GGUF` (text + mmproj).
