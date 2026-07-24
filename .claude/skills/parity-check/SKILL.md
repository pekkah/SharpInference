---
name: parity-check
description: Cross-check SharpInference numerics against llama.cpp and run accuracy gates. Use when output quality or numerics are in question — new architecture bring-up, kernel or quantization changes, KV-compression work, "model produces garbage" reports.
---

# Numeric parity & accuracy gating

Escalate through these levels; stop at the first one that answers the question.

## 1. Unit-level parity (no model files needed)

`tests/SharpInference.Tests.ForwardPass` holds quantization and kernel parity
suites (CPU vs CUDA vs Vulkan, Q4_K/Q6_K/Q8 dequant round-trips). Run the
narrowest filter first — this catches most kernel regressions without any GGUF.

## 2. Greedy token cross-check vs llama.cpp

`pwsh scripts/xcheck-llamacpp.ps1 -Model <gguf> -Prompt "..." -NTokens 30 -Tag <tag>`

- Requires `tools/llama.cpp` (fetch via `pwsh scripts/setup-llamacpp.ps1`) and a
  Release build of the CLI (`dotnet build src/SharpInference.Cli -c Release`).
- Runs both engines at `--temp 0` and prints SharpInference's greedy token IDs;
  outputs land in `tools/xcheck_<tag>_*.txt` for diffing.
- First divergent token position is the lead — trace that position, not the last.

## 3. Reference logits comparison

`pwsh scripts/generate-reference-logits.ps1` plus `scripts/extract_reference.py` /
`scripts/compare_tokens.py` produce and compare per-position logits when token-level
divergence needs a numeric root cause.

## 4. Perplexity gate (statistical accuracy)

```
dotnet run --project src/SharpInference.Cli -c Release -- \
  perplexity -m <model.gguf> -f <corpus.txt> -c 2048 [--tq] [--tq-mode kvarn|lloydmax]
```

The accuracy gate for KV compression and lossy changes (issue #180). Compare
against a baseline run without the change; corpus helpers live in
`scripts/kvarn-gate/` (`get-wikitext2.ps1`, `math-eval.ps1`).

## Notes

- Model files go in `models/`; fetch known presets with
  `pwsh scripts/download-model.ps1 -Model <preset>` (see script header for the list).
- Levels 2–4 need real model files and are slow — prefer delegating to a
  background agent and continuing other work.
