# Q3K_Q8K / Q8_0_Q8K Auto-on Validation — 2026-05-31

Multi-prompt parity validation for issue #103 (re-enabling auto-on for the
`DotQ3K_Q8K` / `DotQ8_0_Q8K` int-domain kernel gates). Reverted commit was
`21e8d13`; this log records why we did NOT re-apply it.

## Setup

- Model: `E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf`
- Backend: CUDA hybrid, `-g -1 --backend cuda --no-thinking`
- Greedy decode (`--temp 0`), 60 decode tokens per cell
- 1 warmup cell to pre-fill OS file cache
- For each prompt: baseline (`SHARPI_Q3K_Q8K=0 SHARPI_Q8_0_Q8K=0`) then gated
  (`SHARPI_Q3K_Q8K=1 SHARPI_Q8_0_Q8K=1`)
- Script: `scripts/bench-q8k-validation.ps1`
- Raw output: `bench-out/bench-q8k-validation.csv`, `tools/bench/q8k-*.{out,err}`

## Results

| Prompt        | Base t/s | Gated t/s | Base MTP | Gated MTP | ΔMTP    | First divergence | Pass (strict) |
|---------------|---------:|----------:|---------:|----------:|--------:|-----------------:|:--------------|
| factual       |    10.5  |    24.6   |     77 % |     73 %  | −4.0 pp |  none in 32 toks | **FAIL** (MTP)|
| codegen       |    11.5  |    26.5   |    100 % |    100 %  |  0.0 pp |  none in 32 toks | PASS          |
| summary       |    17.8  |    25.6   |     90 % |     90 %  |  0.0 pp |  pos 27          | PASS          |
| mathreason    |    17.8  |    27.7   |     97 % |     97 %  |  0.0 pp |  none in 32 toks | PASS          |
| techexplain   |    20.8  |    27.0   |     80 % |     93 %  | +13.0 pp|  pos 11          | **FAIL** (MTP)|

Note on decode t/s: cells run sequentially, so baseline cells take the cold-page
hit first and gated cells run warmer. The absolute lift figures here are inflated
by that ordering bias. The MTP-accept and argmax-divergence figures are
ordering-independent (greedy decode is deterministic given identical inputs and
kernel outputs).

## Failure analysis

### `factual` — output bit-identical but MTP −4 pp

Decoded continuation is **byte-identical** between baseline and gated for the
full 60-token cell:

```
Rainbows are a fascinating optical phenomenon that occurs when sunlight
interacts with water droplets in the atmosphere. Here's a detailed explanation
of how rainbows are formed:

1. **Light Refraction**: When sunlight enters a water droplet, it undergoes
refraction, which is the bending of light
```

But MTP accept dropped 23/30 → 22/30 = −3.3 pp ≈ −4 pp. The MTP head's draft
token differed at exactly one position out of 30 draft cycles, even though the
verifier ended up emitting the same final token. The verifier is the main
forward pass so the final argmax was unchanged; the draft argmax flipped on a
single position with a close margin.

This is a true positive for the kernels' parity gap: the int-domain dots
quantize the input vector to Q8_K (per-block dynamic-range encoding) and that
quantization is not bit-exact with the FP dequant path. One flipped argmax in
30 draft cycles is within "expected" given ~1 % per-kernel tolerance, but it
moves MTP-accept past the ±2 pp criterion #103 set.

### `techexplain` — divergence at position 11

argmax stable for 11 tokens (just past the "~10 tokens" floor), then diverges:

Baseline:
```
The CAP theorem, also known as Brewer's theorem, states that it is impossible
for a distributed computer system to simultaneously provide all three of the
following guarantees:
```

Gated:
```
The CAP theorem, also known as Brewer's theorem, is a fundamental principle in
distributed systems design. It states that a distributed system can only
guarantee two out of the following three properties at any given time:
```

Both are correct and fluent. After divergence the two sequences are no longer
apples-to-apples for MTP comparison, so the +13 pp delta does not directly
indicate "gated is better" — it just reflects different generated content.

### Other prompts

`codegen`, `summary`, `mathreason` all have either byte-identical 32-token
output or divergence well past position 10 with 0 pp MTP delta. These pass
cleanly.

## Why we did NOT re-apply `21e8d13`

The issue's criteria are:

> All cells should show the kernel-gated path within +/-2pp of baseline MTP
> accept and within fp32-noise of baseline argmax through the first ~10
> tokens (or document where divergence is acceptable).

Strict reading: `factual` (−4 pp) and `techexplain` (+13 pp) fail the MTP
tolerance.

Charitable reading: `factual` is output-identical (the strongest possible
parity result) and the MTP delta reflects one flipped draft argmax;
`techexplain` clears the 10-token argmax floor by one position.

We chose the strict reading because:

1. The kernels' parity gap is **demonstrably real** at the prompt level — not
   just at per-kernel ~1 % tolerance. `techexplain` produces semantically-
   equivalent but **textually different** output, which is exactly the
   "cumulative trunk drift" failure mode `feedback_q4k_q8k_no_parity_win`
   warned about.
2. Auto-on flips the default for every Carnice user with no env var set. Out
   of 5 prompts measured, one diverged at position 11 and one had a
   measurable MTP-accept delta. We don't know the failure rate beyond the
   sample.
3. The +6 % decode lift is real but the kernel is **already shipped behind
   `SHARPI_Q3K_Q8K=1` / `SHARPI_Q8_0_Q8K=1`** — users who want it can opt in.

## Follow-up

Kernel-tightening work is needed before re-validation:

- Investigate higher-precision input quantization (e.g. Q8_1-style with a min
  offset, or per-block FP accumulator) to close the parity gap with the FP
  dequant path.
- Once tightened, re-run `scripts/bench-q8k-validation.ps1` and target zero
  prompts with MTP delta outside ±2 pp.

Tracked in the new follow-up issue (see #103 for the link).
