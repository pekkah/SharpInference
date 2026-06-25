# KVarN P0 — CPU Reference Implementation

> Implementation note for issue #180, phase **P0** (CPU reference + accuracy
> gate). Builds on the feasibility analysis in `kvarn-feasibility-research.md`.
> Algorithm: KVarN (Müller et al., Huawei CSL, arXiv:2606.03458). Clean-room
> from the paper — the published reference is a Triton/vLLM fork used for
> algorithm reference only.

## What P0 ships

A correctness-first, scalar CPU implementation of the `kvarn_k4v2_g128` preset
(4-bit keys / 2-bit values, group = 128-token tile), wired as a selectable KV
cache behind an env flag, plus an accuracy harness.

| Piece | File |
|---|---|
| Quantizer core (Hadamard reuse + Sinkhorn norm + asymmetric RTN + fused dequant-dot / V-aggregate) | `src/SharpInference.TurboQuant/KVarN.cs` |
| Compressed tile container (4-bit / 2-bit token-major codes + folded scales) | `src/SharpInference.TurboQuant/KVarNTile.cs` |
| Hybrid FP32-window + tiled cache | `src/SharpInference.Engine/KVarNKvCache.cs` |
| `EnableKVarN` + `KVarNAttention` sibling branch | `src/SharpInference.Engine/ForwardPass.cs` |
| CLI opt-in (`SHARPI_KVARN=1`, CPU path) | `src/SharpInference.Cli/RunCommand.cs` |
| Algorithm tests (round-trip, Sinkhorn, fold parity) | `tests/SharpInference.Tests.TurboQuant/KVarNTests.cs` |
| Cache tests (needle retrieval, full-attention parity vs FP32) | `tests/SharpInference.Tests.ForwardPass/KVarNCacheTests.cs` |

## Algorithm (per 128-token tile, per kv-head)

1. **Randomized Hadamard rotation** along the head-dim axis (reuses
   `WalshHadamard` + a per-head sign flip). Orthonormal, so attention scores are
   preserved: `qᵀk = (S·H·q)ᵀ(S·H·k)`.
2. **Dual-axis Sinkhorn variance normalization** (the novel piece): a log-space
   alternation of per-channel (column) and per-token (row) RMS normalization.
   A few iterations drive both axes toward unit variance, equalizing the dynamic
   range and killing the per-token scale outliers that drive error accumulation
   over long autoregressive decoding. Pure dual-axis rescaling, so it inverts
   exactly: `rotated[t,d] = y[t,d]·cscale[d]·rscale[t]`.
3. **Asymmetric RTN**: per-channel keys at 4-bit, per-token values at 2-bit,
   group = tile.
4. **Scales folded at read time** — no decompressed tile is materialized on the
   attention hot path:
   - K-score: `score[t] = rscale[t]·(Σ_d a[d]·code[t,d] + b)`, with
     `a[d] = q_rot[d]·cscale[d]·qscale[d]`, `b = Σ_d q_rot[d]·cscale[d]·zero[d]`.
   - V-aggregate: `out_rot[d] = cscale[d]·(Σ_t wt[t]·code[t,d] + zsum)`, with
     `wt[t] = w[t]·rscale[t]·qscale[t]`, then a single deferred sign-flip + IWHT.

## Cache shape

`KVarNKvCache` keeps a full-precision FP32 window of recent tokens; once
`TileSize (128)` tokens have aged past the window they are quantized together
into one tile (the dual-axis norm needs the whole tile assembled first). The
public surface (`Append`, `ComputeKScores`, `ComputeVAggregation`, `Fp32KeyAt`,
`TruncateTo`, `Reset`) mirrors `TurboQuantKvCache` so the `ForwardPass`
attention dispatch hosts `KVarNAttention` as a sibling of `TqAttention`.

Decode and (sequential) prefill route through `ForwardPass.Forward`; the batched
prefill fast path is deferred to P1.

## Accuracy gate status

Validated at the algorithm and cache level (the cheap go/no-go signal):

- **Fold parity (exact):** `KScore` reproduces `q·k` against the quantized
  reconstruction, and `VAggregate` reproduces `Σ w·v`, both to fp tolerance —
  the read-time scale folding is algebraically exact.
- **Needle-in-haystack:** a distinctive key remains top-1% retrievable through
  4-bit-key compression at 1K–4K context.
- **Full-attention output vs exact FP32:** cosine > 0.9 with a peaked softmax,
  i.e. the 2-bit value error largely cancels in the weighted aggregate.

End-to-end model-level validation (MATH500 / GSM8K-style, vs FP32 and vs
TurboQuant 3–4 bit) is the remaining P0 milestone item and needs a GGUF model
run with `SHARPI_KVARN=1`.

## Usage

```bash
# CPU path, opt-in via env flag (mutually exclusive with --tq)
SHARPI_KVARN=1 dotnet run --project src/SharpInference.Cli -c Release -- \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "prompt" --temp 0
```

Requires a power-of-two head dimension (Walsh-Hadamard constraint).

## Not in P0 (follow-on phases)

- **P1** — AVX2 fused score/aggregate kernels + native packed tile layout
  (replaces the managed `KVarNTile`); batched prefill path.
- **P2** — CUDA (NVRTC) quantize / dequant-dot / V-aggregate; this is where
  "throughput ≥ FP16" is won.
- **P3** — Vulkan SPIR-V ports.

Per the issue's scope note, KVarN is positioned to eventually fold into the
TurboQuant cache machinery as a selectable quantizer (and supersede the codebook
path if it wins); P0 keeps it as a separate reference cache to avoid disturbing
the shipping TurboQuant path during validation.
