# Gemma 4 12B (dense) — QAT q4_0 GGUF header dump (Phase 0 ground truth)

Source file: `gemma-4-12b-it-qat-q4_0.gguf` (6,975,877,728 bytes ≈ 6.98 GB)
Repo: `google/gemma-4-12B-it-qat-q4_0-gguf` (official Google QAT weights)
Dumped: 2026-06-08, via `sharpi-cli list-metadata` / `list-tensors` on the real file.
GGUF v3 — **667 tensors**, 49 metadata keys.

> ⚠️ **This dense 12B is NOT "E4B minus PLE".** Phase 0 surfaced three architectural
> mechanisms the merged gemma4 (E4B) path never had to handle. See "Deviations" below.
> The plan's premise ("no new architectural mechanisms") does not hold.

---

## 1. Core metadata (`general.*`, `gemma4.*`)

| Key | Value |
|---|---|
| `general.architecture` | **`gemma4`** (NOT `gemma4_unified` — G1 is a no-op) |
| `general.file_type` | 2 (= llama.cpp `MOSTLY_Q4_0`) |
| `general.name` / `general.finetune` | `12B_qat_it_dequant_safetensors` |
| `general.quantization_version` | 2 |
| `general.size_label` | 12B |
| `gemma4.block_count` | **48** |
| `gemma4.embedding_length` | **3840** |
| `gemma4.feed_forward_length` | **15360** |
| `gemma4.context_length` | 262144 (256K) |
| `gemma4.vocab_size` | 262144 |
| `gemma4.attention.head_count` | **16** |
| `gemma4.attention.head_count_kv` | **per-layer ARRAY[48]** → `8` on SWA, `1` on global (see §3) |
| `gemma4.attention.key_length` | 512 (global) |
| `gemma4.attention.key_length_swa` | 256 (SWA) |
| `gemma4.attention.value_length` | 512 |
| `gemma4.attention.value_length_swa` | 256 |
| `gemma4.attention.sliding_window` | **1024** |
| `gemma4.attention.shared_kv_layers` | **0** ✓ |
| `gemma4.attention.layer_norm_rms_epsilon` | 1e-6 |
| `gemma4.rope.dimension_count` | 512 (global) |
| `gemma4.rope.dimension_count_swa` | 256 (SWA) |
| `gemma4.rope.freq_base` | 1,000,000 (global) |
| `gemma4.rope.freq_base_swa` | 10,000 (SWA) |
| `gemma4.final_logit_softcapping` | **30.0** ✓ |
| `gemma4.embedding_length_per_layer_input` | **0** (no PLE) ✓ |

Tokenizer: `tokenizer.ggml.model = gemma4`, BOS=2, EOS=1, pad=0, unk=3, mask=4,
add_bos=true, add_space_prefix=false, 262144 tokens, 514906 merges.
Recommended sampling (`general.sampling.*`): temp 1, top_k 64, top_p 0.95.
A full SentencePiece `tokenizer.chat_template` (Jinja, tool-calling + thinking
channels) is embedded.

Synthetic probes injected by `GgufModel.Open` (from tensor inspection):
`_sharpi.has_post_attn_norm=true`, `_sharpi.has_post_ffw_norm=true`,
`_sharpi.has_qk_norm=true`, **`_sharpi.has_layer_output_scale=true`**.
There is **no** `_sharpi.has_ple` (PLE absent — confirmed).

---

## 2. Tensor inventory & per-tensor quant types (G0)

| Tensor | Count | DType | Sample shape `[ne0, ne1]` |
|---|---|---|---|
| `blk.*.ffn_down.weight`   | 48 | **Q4_0** | [15360, 3840] |
| `blk.*.ffn_gate.weight`   | 48 | **Q4_0** | [3840, 15360] |
| `blk.*.ffn_up.weight`     | 48 | **Q4_0** | [3840, 15360] |
| `blk.*.attn_q.weight`     | 48 | **Q4_0** | SWA [3840, 4096] / global [3840, 8192] |
| `blk.*.attn_output.weight`| 48 | **Q4_0** | SWA [4096, 3840] / global [8192, 3840] |
| `blk.*.attn_k.weight`     | 48 | **Q4_0** | SWA [3840, 2048] / global [3840, **512**] |
| `blk.*.attn_v.weight`     | **40** | **Q4_0** | SWA only [3840, 2048] — **absent on the 8 global layers** |
| `token_embd.weight`       | 1 | **Q6_K** | [3840, 262144] (tied — used as output too) |
| `blk.*.attn_norm.weight`           | 48 | F32 | [3840] |
| `blk.*.ffn_norm.weight`            | 48 | F32 | [3840] |
| `blk.*.post_attention_norm.weight` | 48 | F32 | [3840] |
| `blk.*.post_ffw_norm.weight`       | 48 | F32 | [3840] |
| `blk.*.attn_q_norm.weight` | 48 | F32 | SWA [256] / global [512] |
| `blk.*.attn_k_norm.weight` | 48 | F32 | SWA [256] / global [512] |
| `blk.*.layer_output_scale.weight` | 48 | F32 | [1] |
| `output_norm.weight` | 1 | F32 | [3840] |
| `rope_freqs.weight`  | 1 | F32 | **[256]** ✓ (= maxHeadDim/2) |

Count check: 40 SWA×14 + 8 global×13 + 3 top-level (`token_embd`, `output_norm`,
`rope_freqs`) = **667** ✓ — fully accounted for. **No** `output.weight` (tied
embeddings), **no** `per_layer_*` (no PLE), **no** `mmproj` (vision is a separate file).

**G0 result — the bulk matmul weights are `Q4_0`:** `ffn_{down,gate,up}` and
`attn_{q,k,v,o}`. `token_embd` is `Q6_K`; everything else is F32.

---

## 3. Layer-type map (from `sliding_window_pattern` + tensor shapes)

`sliding_window_pattern[48]` (True = SWA, False = global):
```
[T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F,
 T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F, T,T,T,T,T,F]
```
→ **Global (full-attention) layers: 5, 11, 17, 23, 29, 35, 41, 47** — every 6th,
`(i+1)%6==0`. 40 SWA + 8 global. Last layer (47) is global. Same 5:1 formula as E4B.

`head_count_kv[48]`:
```
[8,8,8,8,8,1, 8,8,8,8,8,1, 8,8,8,8,8,1, 8,8,8,8,8,1,
 8,8,8,8,8,1, 8,8,8,8,8,1, 8,8,8,8,8,1, 8,8,8,8,8,1]
```

| | SWA layers (40) | Global layers (8) |
|---|---|---|
| head_dim | 256 | 512 |
| Q heads | 16 → q[3840,4096] | 16 → q[3840,8192] |
| KV heads | 8 (GQA) → k/v[3840,2048] | **1 (MQA)** → k[3840,512] |
| V projection | separate `attn_v` | **none — K is reused as V** |
| rope base | 10,000 | 1,000,000 (+ `rope_freqs` p-RoPE) |
| q/k norm size | 256 | 512 |

---

## 4. Deviations from the plan (§2) — flag before Phase 2

1. **`attention_k_eq_v = true` (global layers).** Authoritative HF
   `google/gemma-4-12B-it/config.json` (`text_config.attention_k_eq_v=true`).
   The 8 global layers have **no `attn_v.weight`**; V is the K projection output.
   The merged gemma4 path unconditionally loads `attn_v` and runs separate K/V —
   it will throw `Missing tensor: blk.5.attn_v.weight`. **New mechanism.**
2. **Per-layer `head_count_kv` (8 SWA / 1 global).** Stored as an array, not a
   scalar. Two consequences:
   - `ModelHyperparams.GetInt("…head_count_kv")` does `Convert.ToInt32(object[])`
     → **throws on load** today.
   - A single `NumKvHeads` can't represent 8-vs-1; global layers are MQA. Needs a
     per-layer KV-head array (mirroring `LayerHeadDim`).
3. **`layer_output_scale` is PRESENT** (`blk.*.layer_output_scale.weight`, F32,
   48×). Plan G3 expected it absent on dense. Already gated by `HasLayerOutputScale`,
   so handled — but the dense premise was wrong; the **false** branch is still
   untested (no model exercises it).
4. **`token_embd` is `Q6_K`, tied** (no `output.weight`). CUDA embed-lookup keeps
   only Q4_K/Q8_0 packed; Q6_K falls to F32 dequant (~4 GB VRAM for the 3840×262144
   table). Acceptable functionally; a Q6_K embed-lookup kernel would reclaim VRAM.

### G0 backend support for `Q4_0` (verified in code)
- **CPU**: supported — `Dequantize.cs` + `SimdKernels.cs` have q4_0 dequant + matvec.
- **CUDA (full offload)**: **no native Q4_0 matvec**. `CudaBackend.MatMul` switch
  throws `NotSupportedException` for Q4_0; `CudaForwardPass.UploadWeight` routes
  Q4_0 to the F32-dequant fallback → ~4× VRAM (a 7 GB q4_0 model → ~28 GB F32,
  does **not** fit 12 GB). This defeats the QAT "fits full offload" premise.
- **CUDA-Hybrid**: GPU side keeps only {Q4_K, Q6_K} packed; Q4_0 → F32 dequant on
  the GPU-resident layers. CPU-resident layers use native CPU q4_0.

→ **"Implement optimized Q4" = add a native packed Q4_0 CUDA matvec kernel + keep
Q4_0 packed on upload**, so the QAT model actually fits and runs fast on the 4070 Ti.
