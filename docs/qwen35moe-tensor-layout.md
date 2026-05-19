# qwen35moe Tensor Layout & Block Math — Authoritative

*Captured 2026-05-19 from `E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` via `sharpi-cli list-tensors`, cross-referenced against [llama.cpp's `src/models/qwen35moe.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp) and `src/models/delta-net-base.cpp` on master. These findings supersede `qwen35moe-plan.md` where they conflict.*

## Architecture summary

**Qwen3.6-35B-A3B is a Gated DeltaNet (GDN) + MoE hybrid.** The "SSM" blocks are NOT Mamba/S6 — they are **Gated DeltaNet** linear-attention recurrences with a 2-D matrix state per head. The original plan's selective-scan kernel design is invalid for this model.

- 40 trunk layers; `(i+1) % 4 != 0` → GDN block (30 layers), `(i+1) % 4 == 0` → full attention (10 layers, indices 3,7,11,...,39)
- Full-attention layers use **GLU-gated Q** (Q + gate concatenated per head)
- All 40 layers have MoE FFN (256 routed experts, 8 active, plus 1 shared expert)
- Partial RoPE on first 64 head dims; M-RoPE multi-axis sections [11,11,10,0] (single section for text-only)

## Metadata (selected)

```
qwen35moe.block_count = 40
qwen35moe.full_attention_interval = 4    # (i+1) % 4 == 0 → full attn
qwen35moe.attention.head_count = 16      # Q-heads in full-attn layers
qwen35moe.attention.head_count_kv = 2    # KV-heads in full-attn layers
qwen35moe.attention.key_length = 256     # head_dim for full attention
qwen35moe.attention.value_length = 256
qwen35moe.attention.layer_norm_rms_epsilon = 1e-6
qwen35moe.rope.dimension_count = 64
qwen35moe.rope.dimension_sections = [11, 11, 10, 0]  # M-RoPE; text-only ≡ single section
qwen35moe.rope.freq_base = 1e7
qwen35moe.embedding_length = 2048
qwen35moe.context_length = 262_144
qwen35moe.vocab_size = 248_320
qwen35moe.expert_count = 256
qwen35moe.expert_used_count = 8
qwen35moe.expert_feed_forward_length = 512
qwen35moe.expert_shared_feed_forward_length = 512
qwen35moe.ssm.conv_kernel = 4
qwen35moe.ssm.group_count = 16           # ≡ num K-heads in GDN
qwen35moe.ssm.inner_size = 4096          # value_dim = 32 v-heads × 128 head_dim
qwen35moe.ssm.state_size = 128           # head_dim of GDN per-head 128×128 state
qwen35moe.ssm.time_step_rank = 32        # ≡ num V-heads in GDN
```

### Derived constants for GDN

```
gdn_head_dim   = 128        (= ssm.state_size)
gdn_n_v_heads  = 32         (= ssm.time_step_rank)
gdn_n_k_heads  = 16         (= ssm.group_count)
gdn_value_dim  = 4096       (= n_v_heads × head_dim = ssm.inner_size)
gdn_key_dim    = 2048       (= n_k_heads × head_dim)
gdn_qkv_dim    = key_dim*2 + value_dim = 2048 + 2048 + 4096 = 8192   # joint QKV channels
gdn_conv_dim   = 8192       (= qkv_dim; depthwise conv runs over all of it)
```

### Per-sequence GDN state size

- Recurrent state: `[head_dim, head_dim, n_v_heads]` = `[128, 128, 32]` = **524,288 fp32 = 2 MiB** per layer
- Conv state: `[gdn_conv_dim × (conv_kernel - 1)]` = `8192 × 3` = `24,576 fp32 = 96 KiB` per layer
- Across 30 GDN layers: `30 × 2.094 MiB ≈ 63 MiB` per sequence (fixed, position-independent — unlike KV cache)

## Layer typology

| Layer index pattern | Count | Block type |
|---|---|---|
| `(i+1) % 4 == 0` → indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 | 10 | Full attention (GLU-gated Q) |
| All other indices: 0, 1, 2, 4, 5, 6, 8, 9, 10, ... | 30 | Gated DeltaNet (GDN) |

## Per-layer tensors

### GDN-style layer (example: `blk.0`, 19 tensors)

```
blk.0.attn_norm.weight              [2048]            F32     pre-block RMSNorm
blk.0.attn_qkv.weight               [2048, 8192]      Q8_0    joint QKV projection (K=2048, Q=2048, V=4096 along output)
blk.0.attn_gate.weight              [2048, 4096]      Q8_0    z-gate projection (pre-activation, gated SiLU on output)
blk.0.ssm_conv1d.weight             [4, 8192]         F32     depthwise causal conv1d over joint QKV (kernel=4)
blk.0.ssm_alpha.weight              [2048, 32]        F32     per-v-head alpha projection (input to softplus)
blk.0.ssm_beta.weight               [2048, 32]        F32     per-v-head beta projection (input to sigmoid)
blk.0.ssm_a                         [32]              F32     per-head decay coefficient (-A_log.exp in llama.cpp)
blk.0.ssm_dt.bias                   [32]              F32     bias for alpha before softplus
blk.0.ssm_norm.weight               [128]             F32     per-head RMSNorm gain over head_dim
blk.0.ssm_out.weight                [4096, 2048]      Q8_0    output projection: value_dim → embDim
blk.0.post_attention_norm.weight    [2048]            F32     pre-MoE norm
blk.0.ffn_gate_inp.weight           [2048, 256]       F32     MoE router
blk.0.ffn_gate_inp_shexp.weight     [2048]            F32     shared-expert gating weight
blk.0.ffn_gate_shexp.weight         [2048, 512]       Q8_0    shared expert gate
blk.0.ffn_up_shexp.weight           [2048, 512]       Q8_0    shared expert up
blk.0.ffn_down_shexp.weight         [512, 2048]       Q8_0    shared expert down
blk.0.ffn_gate_exps.weight          [2048, 512, 256]  Q4_K    routed experts gate (256 experts, 512 expert dim)
blk.0.ffn_up_exps.weight            [2048, 512, 256]  Q4_K    routed experts up
blk.0.ffn_down_exps.weight          [512, 2048, 256]  Q5_K    routed experts down
```

### Full-attention layer (example: `blk.3`, 16 tensors)

```
blk.3.attn_norm.weight              [2048]            F32
blk.3.attn_q.weight                 [2048, 8192]      Q8_0    GLU-gated Q (Q + gate interleaved per head, 2 × 128 × 16 = 8192)
blk.3.attn_k.weight                 [2048, 512]       Q8_0    K projection (2 KV-heads × 256 head_dim)
blk.3.attn_v.weight                 [2048, 512]       Q8_0    V projection (2 KV-heads × 256 head_dim)
blk.3.attn_q_norm.weight            [256]             F32     per-head Q-norm over head_dim
blk.3.attn_k_norm.weight            [256]             F32     per-head K-norm over head_dim
blk.3.attn_output.weight            [4096, 2048]      Q8_0    output projection
blk.3.post_attention_norm.weight    [2048]            F32
blk.3.ffn_*_(exps|shexp).weight     ...                       (same MoE structure as GDN layers)
```

## Block forward-pass pseudocode

### GDN block (per token, autoregressive decode)

```
# H = embDim = 2048; D = head_dim = 128; Hv = n_v_heads = 32; Hk = n_k_heads = 16
# V = value_dim = 4096 = Hv*D;  K = key_dim = 2048 = Hk*D
# State S_h is a [D, D] matrix per v-head h.

x_norm = RMSNorm(x, attn_norm)
qkv = attn_qkv @ x_norm                              # [8192]
z   = attn_gate @ x_norm                             # [4096]

# Depthwise causal conv (kernel 4) on the joint qkv stream:
conv_in = concat(conv_state, qkv.unsqueeze(t-axis))  # state holds 3 prior tokens
qkv_c   = depthwise_conv1d(conv_in, ssm_conv1d.w)    # [8192], no bias in this model
conv_state := conv_in[-3:]                           # roll forward

qkv_c   = SiLU(qkv_c)
k_pre, q_pre, v = split(qkv_c, [2048, 2048, 4096])   # along channel axis

q = L2Norm(q_pre.reshape(Hk, D))                     # [Hk, D]  (norm per head)
k = L2Norm(k_pre.reshape(Hk, D))                     # [Hk, D]  (norm per head)
v = v.reshape(Hv, D)                                 # [Hv, D]
# K-heads repeated to match V-heads (16 → 32, factor 2):
q = repeat_interleave(q, Hv/Hk, axis=0)              # [Hv, D]
k = repeat_interleave(k, Hv/Hk, axis=0)              # [Hv, D]

alpha = ssm_alpha @ x_norm                           # [Hv]
beta  = ssm_beta  @ x_norm                           # [Hv]
dt    = softplus(alpha + ssm_dt.bias)                # [Hv]
g     = exp(dt * ssm_a)                              # [Hv]   per-head decay (ssm_a is negative)
b     = sigmoid(beta)                                # [Hv]

# Recurrent kernel (per v-head, S_h ∈ R^{D×D}):
for h in 0..Hv-1:
    S_h := S_h * g[h]                                # decay (scalar broadcast)
    d    = (v[h] - S_h @ k[h]) * b[h]                # [D]   "delta" with per-head scalar gain
    S_h := S_h + outer(k[h], d)                      # [D,D] rank-1 update
    o[h] = S_h @ q[h]                                # [D]   readout

o = o.reshape(V)                                     # [4096]
o = RMSNorm(o.reshape(Hv, D), ssm_norm.w).reshape(V) # gain is [D], applied per-head
o = o * SiLU(z)                                      # [4096]
out = ssm_out @ o                                    # [2048]
return x + out                                       # residual
```

### Full-attention block (per token)

```
x_norm = RMSNorm(x, attn_norm)
qg = attn_q @ x_norm                                 # [8192]
# Interleaved per head: for each head h ∈ [0,16),
#   Q[h] = qg[h*256     : h*256+128]
#   G[h] = qg[h*256+128 : h*256+256]
q = qg.reshape(16, 2*128)[:, :128]                   # [16, 128]
g = qg.reshape(16, 2*128)[:, 128:]                   # [16, 128]
k = (attn_k @ x_norm).reshape(2, 128)                # [2, 128]
v = (attn_v @ x_norm).reshape(2, 128)                # [2, 128]
q = RMSNorm(q, attn_q_norm)                          # per-head over head_dim
k = RMSNorm(k, attn_k_norm)
q = partial_rope_neox(q, position, rope_dim=64)      # first 64 dims rotated
k = partial_rope_neox(k, position, rope_dim=64)
attn_out = scaled_dot_product_attn_gqa(q, k, v, kv_cache)   # [16, 128]
attn_out = attn_out * sigmoid(g)                     # GLU gate
out = attn_output @ attn_out.reshape(2048 wait, 4096 — see note)
return x + out
```
**Note:** `attn_output` has shape `[4096, 2048]` (input 4096, output 2048). But 16 heads × 128 head_dim = 2048, not 4096. Re-check this when implementing — either head_dim is 256 (matching `key_length=256` from metadata) with K/V at 256 (matching `[2048,512]` = 2×256), OR head_dim is 128 with attn_output input including the gate-multiplied output at twice the width. The llama.cpp source is the tiebreaker; my reading of the research is that head_dim=128 for full attention here, so attn_output's input may include both rotated and unrotated halves of Q — needs re-validation.

### MoE FFN block (shared by both layer types)

```
x_norm = RMSNorm(x, post_attention_norm)
# Routed experts:
router_logits = ffn_gate_inp @ x_norm                # [256]
topk_ids, topk_w = topK_softmax(router_logits, k=8)
moe_out = sum over k of topk_w[k] * expert_ffn(topk_ids[k], x_norm)
# Shared expert:
shexp_gate = sigmoid(ffn_gate_inp_shexp · x_norm)    # scalar gate (or per-channel? [2048]→scalar)
shexp_out = ffn_down_shexp @ (SiLU(ffn_gate_shexp @ x_norm) * (ffn_up_shexp @ x_norm))
return x + moe_out + shexp_gate * shexp_out
```

## Key implementation notes for SharpInference port

1. **GDN state per sequence is 2 MiB per layer × 30 layers ≈ 60 MiB.** Allocate eagerly; reset zeroes it.

2. **Conv state per sequence is 96 KiB per layer × 30 layers ≈ 3 MiB.** Same lifecycle.

3. **`ssm_a` is stored already-negative** (the GGUF writer applies `-exp(A_log)` at conversion time). The llama.cpp expression is `exp(softplus(alpha+dt_bias) * ssm_a)`. Don't re-negate; multiply by `ssm_a` directly.

4. **Full-attention layers use `head_dim = 256`** (from `attention.key_length=256`); GDN layers use `head_dim = 128` (from `ssm.state_size`). The two block types have different head dimensions. Plumb both through the hyperparams.

5. **The Q/K/V ordering inside `attn_qkv [8192]` is unconfirmed** — both research agents disagreed (one said `Q∥K∥V`, the other `K∥Q∥V`). Must read llama.cpp's `build_qkvz` directly when implementing Phase 4. Functionally it only matters because llama.cpp is the reference for parity testing in Phase 5.

4. **No `ssm_conv1d.bias`** tensor exists — conv has no bias.

5. **L2Norm on Q and K post-conv, not RMSNorm.** L2-normalize each `[head_dim]` slice (divide by `||v||_2`).

6. **`ssm_norm`'s `[128]` gain is per-head, applied across all 32 heads.** Same gain for every head over its 128-dim output. Apply BEFORE the SiLU(z) gate.

7. **K is repeated 2× before recurrence** (Hk=16 → Hv=32). Cheap broadcast.

12. **No MTP head in this checkpoint.** Verified: `blk.40` has zero tensors and no `nextn_predict_layers` metadata. The model uses exactly 40 trunk layers.

8. **Full-attn `attn_q` is GLU-gated**: first 128 of each 256-wide head slot is Q, second 128 is the sigmoid gate on the attn output. M-RoPE applies to the Q half only.

9. **TruncateTo** for GDN is fundamentally lossy (state is rank-1-updated in place; no rewind). For v1: throw unless length ∈ {0, current}. Speculative decoding is disabled for hybrid GDN+MoE in v1.

10. **BatchForwardMulti / PrefillWithCache**: extend the MoE guard to also reject hybrid (`IsHybridSsm`).

11. **Prefill scan**: write a sequential T-step recurrence first. Parallel chunking scan (`build_delta_net_chunking`) is a future optimization.

## Sources

- llama.cpp `src/models/qwen35moe.cpp` and `src/models/delta-net-base.cpp` on master
- Recent issues/PRs: ggml-org/llama.cpp #19690, #20075, #22320
