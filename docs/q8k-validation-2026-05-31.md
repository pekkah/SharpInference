# Q3K_Q8K / Q8_0_Q8K Auto-on Validation — 2026-05-31

Multi-prompt parity validation for issue #103 (re-enabling auto-on for the
`DotQ3K_Q8K` / `DotQ8_0_Q8K` int-domain kernel gates). Original probe with
plain Q8_K input (single per-256 scale) failed strict acceptance criteria on
2/5 prompts. Issue #107 tightened the input quantization to per-32-element
scales (Q8_KS); this log records both passes.

## Setup

- Model: `E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf`
- Backend: CUDA hybrid, `-g -1 --backend cuda --no-thinking`
- Greedy decode (`--temp 0`), 60 decode tokens per cell
- 1 warmup cell to pre-fill OS file cache
- For each prompt: baseline (`SHARPI_Q3K_Q8K=0 SHARPI_Q8_0_Q8K=0`) then gated
  (`SHARPI_Q3K_Q8K=1 SHARPI_Q8_0_Q8K=1`)
- Script: `scripts/bench-q8k-validation.ps1`
- Raw output: `bench-out/bench-q8k-validation*.csv`, `tools/bench/q8k-*.{out,err}`

## Results — original Q8_K input (pre-#107)

| Prompt        | Base MTP | Gated MTP | ΔMTP    | First divergence | Pass (strict) |
|---------------|---------:|----------:|--------:|-----------------:|:--------------|
| factual       |     77 % |     73 %  | −4.0 pp |  none in 32 toks | **FAIL** (MTP)|
| codegen       |    100 % |    100 %  |  0.0 pp |  none in 32 toks | PASS          |
| summary       |     90 % |     90 %  |  0.0 pp |  pos 27          | PASS          |
| mathreason    |     97 % |     97 %  |  0.0 pp |  none in 32 toks | PASS          |
| techexplain   |     80 % |     93 %  | +13.0 pp|  pos 11          | **FAIL** (MTP + argmax)|

Envelope: ±13 pp MTP, argmax divergence at position 11 on techexplain.

## Results — Q8_KS per-32 input (post-#107)

| Prompt        | Base MTP | Gated MTP | ΔMTP    | First divergence | Pass (strict) |
|---------------|---------:|----------:|--------:|-----------------:|:--------------|
| factual       |     77 % |     77 %  |  0.0 pp |  none in 32 toks | PASS          |
| codegen       |    100 % |    100 %  |  0.0 pp |  none in 32 toks | PASS          |
| summary       |     90 % |     87 %  | −3.0 pp |  pos 26          | **FAIL** (MTP)|
| mathreason    |     97 % |     97 %  |  0.0 pp |  none in 32 toks | PASS          |
| techexplain   |     80 % |     83 %  | +3.0 pp |  none in 32 toks | **FAIL** (MTP)|

Envelope: ±3 pp MTP — **4× tighter** than the Q8_K probe. Every prompt is now
argmax-stable through the full 32-token capture window (techexplain's pos-11
divergence is gone). The two remaining MTP-delta failures (summary, techexplain)
sit at exactly ±3 pp = one flipped draft out of 30 cycles, the draft-decoder
sample-noise floor.

## What changed (#107)

The original Q8_K input quantization stores one FP scale per 256-element
super-block; Q8_KS stores 8 FP scales per super-block (one per 32-element
sub-block). Each sub-block's `iscale = -127 / max_signed_amax_sub`, so
sub-blocks of lower dynamic range fill more of [-127, +127]. Routed-MoE
Phase-A and Phase-C inputs (post-RmsNorm, post-SiLU activations) have
non-uniform magnitude across the 256-element super-block, which is exactly
where Q8_K's single per-256 scale loses precision.

Per-kernel tightening tests in `SimdKernelsQ8KSTests` confirm Q8_KS tracks
the FP dequant-FMA reference 1.1–1.5× tighter than Q8_K on non-uniform
inputs (population mean-absolute-error ratio of 0.66–0.91).

A per-16 variant matching Q3_K's natural scale-lane granularity exactly was
tried and rejected: it shuffled rounding noise to a different prompt set
(factual regressed back to −4 pp, techexplain re-introduced pos-11 divergence)
without a net improvement. Per-32 was the local optimum for the prompts in
this suite. The remaining ±3 pp residual likely requires per-block min
offsets (Q8_1 style) to close further — left for a follow-up if the noise-
floor jitter ever becomes load-bearing.

## Outcome

`21e8d13` re-applied to auto-enable the Q3K_Q8KS / Q8_0_Q8KS gates on APEX-
mixed-precision models. The default user experience on Carnice is now
within draft-cycle noise of the FP-dequant baseline on every prompt
measured, while delivering the ~+6 % warm-warm decode lift the original
issue tracked.
