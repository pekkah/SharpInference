# Qwen3.6-27B-MTP parity at pos 12, post Q6_K · Q8_K fix
## (regenerated 2026-05-26 with current sharpi build, after the falsified GDN dot-order experiment was reverted)

Prompt: 13-token chat-template wrapping of "The capital of France is" → ends in `<|im_start|>assistant\n\n`.
Sharpi side: `Repro_Pos13Parity.DumpPos13Logits` with `SHARPI_TRACE_LAYERS=1` →
`tmp/sharpi_layer_trace.txt` (current build, sha `f8d8229`).
Llama side: `llama-eval-callback.exe` with SHARPI-COMPARE patch on `common/debug.cpp` →
`tmp/llama_eval_cb.txt` (unchanged since 2026-05-25 22:02).

## L0 per-tensor (pos = 12, row 12 of prefill)

| Tensor (sharpi name / llama name)            | l2(sharpi) | l2(llama) | Δl2(abs) | sum(sharpi) | sum(llama) | Δsum(abs) | Notes |
|----------------------------------------------|-----------|-----------|----------|-------------|------------|-----------|-------|
| input_embed                                  | 0.8013    | 0.8013    | 0        | -0.5595     | -0.5595    | 0         | exact ✓ |
| gdn-pre-norm / attn_norm-0                   | 73.9299   | 73.9300   | <0.001   | -54.4127    | -54.4127   | <0.001    | exact ✓ |
| gdn-qkv-mixed (Q6_K · attn_norm)             | 437.259   | 437.2586  | <0.001   | -633.074    | -633.0740  | <0.001    | **Q6_K fix holds** ✓ |
| gdn-conv-raw / conv_output_raw               | 132.134   | 132.1341  | <0.001   | 668.881     | 668.8809   | <0.001    | exact ✓ |
| gdn-conv-silu / conv_output_silu             | 124.277   | 124.2769  | <0.001   | 1015.41     | 1015.4067  | <0.001    | exact ✓ |
| gdn-alpha (node_34, Q6_K)                    | 43.7799   | 43.7799   | <0.001   | 95.071      | 95.0710    | <0.001    | exact ✓ |
| gdn-beta (node_40, Q6_K)                     | 11.5202   | 11.5202   | <0.001   | 42.6115     | 42.6115    | <0.001    | exact ✓ |
| gdn-z (Q4_K · attn_norm)                     | 285.345   | 285.5026  | 0.158    | -15684.4    | -15682.371 | 2.03      | tiny Q4_K residue (0.06%/0.013%) |
| **gdn-out (final_output-0, GDN scan)**       | 3.86054   | 3.8582    | 0.0023   | -7.79791    | -7.7774    | **0.0205**| **first real divergence** (Δsum 0.26%) |
| gdn-proj (linear_attn_out, Q4_K · gdn-out)   | 9.23459   | 9.2236    | 0.011    | 9.23648     | 9.1572     | 0.079     | amplified by ssm_out matmul |
| gdn-resid (attn_residual-0)                  | 9.25703   | 9.2462    | 0.011    | 8.67696     | 8.5977     | 0.079     | L0 net |

**L0 conclusion:** The first divergence at L0 is unchanged. Every upstream tensor remains
bit-exact to ~4 decimals; the only meaningful drift originates inside the
`GATED_DELTA_NET` scan (sharpi `GdnKernels.GdnRecurrenceDecode`). The post-Q6_K
parity table was already correct at L0 — no upstream tensor has silently drifted.

## Per-layer cumulative residual drift at pos 12 (new analysis)

Sharpi vs llama, absolute `Δsum` and `Δl2` at the post-Attention residual of every
layer (sharpi: `gdn-resid` for GDN layers / `attn-resid` for attention layers;
llama: `attn_residual-N`). Attention layers (every 4th, starting at L3) are bolded.

| L | l2(sharpi) | sum(sharpi) | l2(llama) | sum(llama) | |Δl2| | |Δsum| |
|---|-----------|-------------|-----------|------------|------|--------|
| 0  | 9.257    | 8.677    | 9.246   | 8.598   | 0.011 | 0.079 |
| 1  | 11.731   | 11.585   | 11.650  | 11.446  | 0.082 | 0.140 |
| 2  | 11.974   | 11.094   | 11.900  | 11.044  | 0.074 | 0.050 |
| **3**  | 15.595 | 14.793 | 15.503 | 14.790 | 0.092 | **0.003** |
| 4  | 18.179   | 19.578   | 18.074  | 19.709  | 0.105 | 0.132 |
| 5  | 19.699   | 21.366   | 19.608  | 21.527  | 0.091 | 0.161 |
| 6  | 20.907   | 21.278   | 20.846  | 21.445  | 0.061 | 0.167 |
| **7**  | 22.164 | 21.889 | 22.087 | 22.182 | 0.077 | 0.293 |
| 11 | 28.000   | 20.715   | 27.952  | 20.876  | 0.048 | 0.161 |
| 15 | 34.936   | 24.728   | 34.845  | 25.020  | 0.092 | 0.292 |
| 19 | 40.325   | 31.200   | 40.196  | 30.559  | 0.129 | 0.641 |
| 20 | 35.955   | 13.197   | 35.871  | 12.387  | 0.084 | 0.810 |
| 22 | 37.177   | 10.144   | 37.207  | 7.517   | 0.029 | 2.627 |
| **23** | 40.364 | -2.798 | 40.341 | -4.536 | 0.022 | 1.738 |
| 24 | 43.031   | 28.111   | 43.135  | 25.413  | 0.104 | 2.698 |
| **27** | 48.301 | 40.420 | 48.592 | 36.799 | 0.291 | 3.621 |
| **31** | 49.500 | 50.975 | 49.531 | 53.394 | 0.030 | 2.419 |
| **35** | 58.133 | -17.541 | 58.440 | -16.585 | 0.306 | 0.955 |
| 42 | 61.122   | 43.809   | 61.377  | 47.367  | 0.254 | 3.559 |
| **43** | 64.488 | 51.630 | 64.588 | 53.441 | 0.100 | 1.811 |
| **47** | 74.006 | -8.454 | 73.911 | -6.669 | 0.095 | 1.785 |
| 48 | 78.019   | 24.425   | 76.963  | 24.927  | **1.055** | 0.502 |
| 50 | 83.363   | -11.788  | 82.815  | -20.604 | 0.548 | 8.816 |
| **51** | 92.307 | 34.170 | 92.542 | 23.640 | 0.234 | **10.530** |
| **55** | 150.771 | 172.096 | 150.499 | 168.986 | 0.272 | 3.111 |
| 57 | 179.073  | 416.515  | 178.777 | 386.111 | 0.296 | 30.404 |
| 58 | 184.371  | 442.712  | 183.865 | 411.255 | 0.506 | 31.457 |
| **59** | 195.898 | 537.887 | 195.706 | 515.155 | 0.192 | 22.732 |
| 60 | 204.661  | 508.661  | 205.565 | 490.162 | 0.904 | 18.499 |
| 61 | 214.672  | 493.735  | 215.848 | 477.105 | 1.176 | 16.631 |
| 62 | 227.099  | 546.228  | 227.933 | 532.478 | 0.834 | 13.750 |
| **63** | 248.414 | 398.027 | 249.983 | 381.901 | 1.569 | 16.127 |

## Interpretation

1. **Magnitude is preserved throughout.** |Δl2| stays under 0.3 for layers 0–47
   (|l2| range 9 → 74). Only at L48+ does |Δl2| exceed 0.5, and even at L63 it is
   1.6 in 248 = 0.6%. This is exactly the "same magnitude, drifted direction"
   signature of f32-accumulation-order error.

2. **Direction drift is non-monotonic** — see L2 (Δsum 0.050) → L4 (0.132) →
   L6 (0.167) → L7 (0.293) → … → L24 (2.7) → L25 (0.39) → L26 (0.018) → L27 (3.6).
   Random rounding occasionally cancels. So this is not a single localized bug —
   it's diffuse f32-rounding-order drift.

3. **L3 is essentially bit-exact** (Δsum 0.003 in 14.8). The attention
   computation at the *first* attention layer is not introducing visible drift on
   top of its (already-near-bit-exact) input. That argues *against* MRoPE-at-pos>0
   being a primary cause, at least in early layers — the small L0 GDN drift does
   not get amplified by attention's RoPE+softmax until much later (L48+).

4. **The breakpoint is around L48–L51**, where |Δl2| first exceeds 1 and |Δsum|
   first exceeds 10. By L57+ Δsum sits in the 14–31 range and stays. This matches
   the previously-recorded rank trajectory (L35 rank 7 → L47 rank 2 → L59 rank 0).

5. **Attention layers are not the sole locus.** L50 (GDN) already has Δsum 8.8
   before the L51 attention adds another 1.7 to reach 10.5. The drift compounds
   through both GDN and attention layers; attention amplifies but does not
   originate it.

## Why the 2026-05-26 GDN dot-order falsification was premature

The transposed-scratch ggml-order dot port was tested only at L0 pos 12, where
the gdn-out drift is 0.26% (Δsum 0.02 absolute). At that scale, every reduction
order produces ~the same result within fp32 precision: the state is 0 after
13 small rank-1 updates and the dot magnitudes are tiny. The falsification
holds for L0 but says nothing about L40+ where state magnitudes grow large
enough that reduction order matters. Re-running the experiment with end-to-end
pos-12 evaluation (not L0-snapshot) would test the hypothesis correctly.

## Best next step (refined recommendation)

Given the picture above, option **(3) — extend tracing into GDN internals at
multiple layers** is the highest-information probe. Concrete:

- Add `SHARPI_TRACE_GDN_INTERNAL=1` to `GdnKernels.GdnRecurrenceDecode` that
  dumps, for the first head at the *current pos*, the four scan
  intermediates: `dvec` (per-head decay), `state_after_decay` (sum), `S_outer`
  (rank-1 update sum), and `p`-readout (q · S). Emit at L0, L20, L40, L60.
- On the llama side, extend the SHARPI-COMPARE patch in `common/debug.cpp`
  to emit the matching intermediates from `ggml_compute_forward_gated_delta_net_one_chunk`
  (the per-head loop at `ops.cpp:10475`).
- Diff at the same layers. If sharpi/llama agree at L0 but diverge at L40,
  the inner dot ordering or the state-decay multiplication is the source.
  If they diverge at L0 too, the source is upstream of GDN-scan altogether
  (which the L0 per-tensor table here would already have shown — and it does
  not, so the divergence really does emerge inside the scan).

Option **(2)** — MRoPE rotation at pos>0 — is lower priority. L3 attn is
near-bit-exact (Δsum 0.003), which would not be the case if MRoPE at pos 12
had a systematic bug in the attention path: the attention output's residual
would carry an attention-specific drift signature. Verify the math
algebraically only if (3) clears the GDN as a source.
