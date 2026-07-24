---
name: new-arch
description: Checklist for adding support for a new GGUF model architecture (or a variant of an existing one) to SharpInference.
---

# Adding a new model architecture

First check it isn't already covered: many "new" models reduce to an existing arch
(e.g. Ornith-1.0 is qwen35/qwen35moe — see CLAUDE.md). Inspect the GGUF before
writing any code:

```
dotnet run --project src/SharpInference.Cli -c Release -- list-metadata -m model.gguf
dotnet run --project src/SharpInference.Cli -c Release -- list-tensors  -m model.gguf
```

## Order of work

1. **Plan doc** — write `docs/<arch>-plan.md` before coding; `docs/qwen35moe-plan.md`
   is the template (tensor layout, hparams, forward-pass math, phased milestones).
   This is an architect-agent task.
2. **Core** — `src/SharpInference.Core/ModelGraph.cs`: arch name dispatch, hparam
   mapping, tensor-name wiring. Tokenizer + chat template usually come free via
   `Microsoft.ML.Tokenizers` + `JinjaChatTemplate`; add a `ToolCallAdapter` entry
   only if the model family has its own tool-call wire format.
3. **Engine** — reuse an existing `IForwardPass` implementation whenever the math
   allows: dense → `ForwardPass`/`CudaForwardPass`/`GpuForwardPass`; MoE →
   `HybridForwardPass`/`CudaHybridForwardPass`; hybrid Gated-DeltaNet →
   `HybridGdnForwardPass`/`CudaHybridGdnForwardPass`/`VulkanHybridGdnForwardPass`.
   A genuinely new block type means new kernels in all backends you support —
   scope CPU-first, GPU after parity.
4. **Tests** — Tests.Core for tokenizer/template/GGUF metadata; Tests.ForwardPass
   parity against reference logits (see the `parity-check` skill and
   `scripts/generate-reference-logits.ps1`).
5. **Plumbing** — `scripts/download-model.ps1` preset (extend the ValidateSet),
   CLAUDE.md architecture list, README.
6. **Validation** — greedy cross-check vs llama.cpp, then a perplexity run
   (`parity-check` skill, levels 2 and 4). New-arch work is not done until greedy
   tokens match llama.cpp at temp 0 on at least one real model.

## Delegation pattern

Plan doc and design → `architect` agent. Steps 2–5 → `implementer` agent(s), one
per step, sequential (each depends on the previous). Validation → `test-runner`
plus the `parity-check` skill in a background agent.
