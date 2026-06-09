# Nemotron-3.5-ASR-Streaming-0.6B — Support Feasibility & Implementation Plan

Research note for adding `nvidia/nemotron-3.5-asr-streaming-0.6b` to SharpInference.

- **Status:** Research / planning only. No code yet.
- **Branch:** `claude/nemotron-asr-support-jjpula`
- **TL;DR:** This is a *new modality* (audio → text) built on a *new architecture family*
  (FastConformer encoder + RNN-T transducer decoder) shipped in a *new file format* (`.nemo`).
  None of the existing GGUF text-LLM or image-diffusion infrastructure is reusable end-to-end.
  It is a multi-stage, parallel pipeline — comparable in scope to the diffusion subsystem,
  not a "new arch string in `ModelGraph`" change. Recommend a staged build behind an
  offline `.nemo → safetensors` conversion step.

---

## 1. What the model is

| Property | Value |
|---|---|
| Repo | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| Task | Automatic Speech Recognition (streaming) |
| Framework | **NVIDIA NeMo** (`library: nemo`, PyTorch) |
| Params | ~0.6 B |
| Encoder | **FastConformer** (cache-aware, limited-context attention) |
| Decoder | **RNN-T / Transducer** (Parakeet family) |
| Languages | 35+ (en, es, de, fr, it, ar, ja, ko, zh, hi, …) |
| Audio in | 16 kHz mono PCM |
| Output | Text transcript (streaming-capable) |
| License | `other` — NVIDIA Open Model License class; **verify redistribution terms** |
| Refs | arXiv:2312.17279 (FastConformer), arXiv:2305.05084 (cache-aware streaming Conformer) |

### 1.1 Architecture in detail

**FastConformer encoder**
- Conv subsampling front-end with **depthwise-separable convolutions, 8× time
  downsampling** (FastConformer's distinguishing feature vs. the 4× in vanilla Conformer).
- Stack of Conformer blocks, each a *macaron* sandwich:
  `½·FFN → MHSA(relative positional encoding) → Conv module (pointwise → GLU → depthwise → BatchNorm → SiLU → pointwise) → ½·FFN → LayerNorm`.
- "Cache-aware" = trained with limited right-context so the same weights run in chunked
  streaming mode at several latency settings (e.g. 0 / 80 / 480 / 1040 ms lookahead),
  carrying an **encoder activation cache** across chunks.

**RNN-T transducer decoder**
- **Prediction network**: LSTM over previously emitted tokens (stateful, autoregressive on text).
- **Joint network**: combines encoder frame embedding + prediction state → vocab logits
  including a **blank** symbol.
- **Decoding**: greedy or beam transducer search (emit-or-advance loop), *not* the
  LLM autoregressive sampler.

**Front-end features**
- Log-mel filterbank: 16 kHz, ~25 ms window / 10 ms hop, 80 or 128 mel bins
  (exact bin count, normalization, dither, pre-emphasis **must be read from
  `model_config.yaml`**).

**Tokenizer**
- SentencePiece (unified multilingual subword vocab; RNN-T blank is a separate index).
  Vocab size to be confirmed from the checkpoint.

> ⚠️ The HF model card and raw config files were **not reachable** from this sandbox
> (network allowlist + the HF MCP file-list call dropped). Architecture above is from the
> repo tags + NeMo FastConformer/Parakeet knowledge. Before implementation, the exact
> `model_config.yaml` must be pulled and the *italicized* values locked down.

### 1.2 `.nemo` file format

A `.nemo` file is a **tar archive** containing:
- `model_config.yaml` — full architecture + preprocessor + tokenizer config,
- `model_weights.ckpt` — PyTorch state dict (a zip of pickled tensors),
- tokenizer artifacts (SentencePiece `.model` / vocab).

Parsing pickled PyTorch checkpoints in C# under NativeAOT is hostile (pickle VM, no
reflection). **Do not** attempt to read `.nemo` directly at runtime.

---

## 2. Gap analysis vs. SharpInference today

The codebase is exclusively GGUF text transformers + image diffusion. A grep for
`conformer | rnnt | transducer | spectrogram | mel | \.nemo | audio` finds **no** audio
code (the `wave` hits are attention-SDPA "wave" scheduling, unrelated to audio waveforms).

| Capability needed | Exists today? | Notes |
|---|---|---|
| `.nemo` loading | ❌ | Tar + pickle. Solve **offline** (convert to safetensors). |
| Safetensors weight load | ✅ partial | `SafetensorsLoader` lives in `SharpInference.Diffusion`; would need promoting to `Core`. |
| Audio decode (WAV/PCM) | ❌ | New. 16 kHz mono, resample. |
| Log-mel feature extraction | ❌ | New DSP: framing → window → FFT → mel filterbank → log. Needs an FFT. |
| Conv subsampling / depthwise conv1d | ⚠️ | `IImageOpsBackend.Conv2d` (CUDA/Vulkan) exists for RRDBNet — adaptable, but it's **2D and GPU-only**; need CPU + 1D paths. |
| Relative-position MHSA | ❌ | `IComputeBackend.Attention` is RoPE/causal LLM attention. Conformer uses Transformer-XL-style **relative** positional attention — different math. |
| BatchNorm / GLU / SiLU-conv module | ⚠️ | `SiLU` exists; BatchNorm and GLU do not. |
| LSTM cell (prediction net) | ❌ | New. No recurrent cell anywhere in the engine. |
| RNN-T joint + transducer search | ❌ | New decode loop with blank handling; the `Sampler` / KV-cache / `IForwardPass` machinery does **not** apply. |
| Cache-aware streaming state | ❌ | Encoder activation cache across chunks — distinct from `PagedKvCache`. |
| Audio CLI / API surface | ❌ | CLI is chat-oriented; no `transcribe` verb, no `/v1/audio/transcriptions`. |

**Conclusion:** essentially a third top-level pipeline (`SharpInference.Asr`) alongside the
LLM engine and the diffusion pipeline. Reusable pieces are limited to: the `IComputeBackend`
MatMul/Softmax/SiLU primitives, the GPU `Conv2d` (with adaptation), and `SafetensorsLoader`.

---

## 3. Recommended staged plan

Each stage is independently testable and validated against a NeMo reference where possible.

**Stage 0 — Offline conversion + config capture** (`scripts/`)
- Python script using `nemo_toolkit` to load the `.nemo` and export:
  `weights.safetensors` + `config.json` + SentencePiece tokenizer.
- Dump the real `model_config.yaml` to pin: n_mels, hop/window, subsampling factor,
  encoder depth/width/heads, conv kernel size, vocab size, blank id, streaming chunk sizes.
- Capture reference tensors (mel features, encoder output, a full transcript) on a sample
  WAV for golden tests.

**Stage 1 — Audio front-end** (new `SharpInference.Audio` or under Core)
- WAV/PCM reader, mono downmix, resample to 16 kHz.
- Log-mel filterbank extractor + FFT. Test mel output bit-against Stage-0 reference (atol).

**Stage 2 — FastConformer encoder (CPU first)**
- Conv subsampling, Conformer blocks (macaron FFN, relative-pos MHSA, conv module w/ BatchNorm).
- Promote `SafetensorsLoader` to Core; map NeMo tensor names → graph.
- Validate encoder output vs. Stage-0 reference.

**Stage 3 — RNN-T decoder + offline transcription**
- LSTM prediction net, joint network, **greedy** transducer search with blank.
- End-to-end offline (full-utterance) transcription; validate WER on the sample.

**Stage 4 — Cache-aware streaming**
- Chunked encoder with activation cache; expose latency presets.

**Stage 5 — Acceleration + frontends**
- GPU kernels (reuse/extend `Conv2d`, add conv1d/BatchNorm); CLI `transcribe` verb;
  optional `/v1/audio/transcriptions` server endpoint.

### Suggested project layout
```
src/SharpInference.Audio/        # WAV + log-mel features (FFT)
src/SharpInference.Asr/          # FastConformer encoder, RNN-T decoder, streaming, pipeline
scripts/convert_nemo.py          # .nemo -> safetensors + config.json + tokenizer
tests/SharpInference.Tests.Asr/  # golden tests vs NeMo reference (features, encoder, WER)
```

---

## 4. Risks & open questions

1. **License** — `license: other`. Confirm NVIDIA terms permit redistributing converted
   weights / shipping a loader before publishing anything.
2. **Exact config** — every italicized value in §1.1 is currently inferred, not read.
   Stage 0 is a hard prerequisite.
3. **NativeAOT** — the whole repo is AOT + TreatWarningsAsErrors + InvariantGlobalization.
   FFT, BatchNorm, LSTM must be allocation-light, reflection-free, culture-invariant.
4. **Relative-position attention & RNN-T search** are the two genuinely novel algorithms;
   budget the most validation time there.
5. **Scope** — this is multi-week. Confirm appetite before building Stage 2+, and decide
   whether streaming (Stage 4) is in-scope for v1 or whether offline transcription ships first.

---

## 5. Recommendation

Proceed **Stage 0 first** (conversion + config capture + golden references) — it's low-risk,
unblocks accurate dimensioning, and answers the license question. Gate Stages 1–5 on an
explicit go/no-go after Stage 0, since the full build is a substantial new subsystem.
