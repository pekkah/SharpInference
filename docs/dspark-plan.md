# DSpark Speculative Decoding — Feasibility & Adaptive-Placement Spec

*Drafted 2026-07-01, on branch `claude/dspark-feasibility-61y4au`.*

> **Implementation status (2026-07-07):** Phases 0–3 are implemented. Phase 0 findings
> (exact backbone math, tensor schema, inference protocol reverse-engineered from
> `deepspec/modeling/dspark/` + `deepspec/eval/`) corrected one §6 assumption: the DFlash
> backbone is NOT a k-sequential drafter — it is an EAGLE-3-style block drafter whose
> per-layer context K/V is projected from TARGET hidden-state taps
> (`target_layer_ids`, fused via `fc` + RMSNorm), with mask-token block positions decoded
> bidirectionally in one pass. Landed pieces: `SafetensorsLoader` moved to Core (+
> `ReadRaw`), hidden-state taps on `IForwardPass`/`ForwardPass`
> (`EnableHiddenTaps`/`HiddenTapsAt`, captured in Forward/Prefill/BatchVerify),
> `DSparkConfig`, `DSparkDraftModel` (CPU backbone + vanilla Markov head + confidence
> head), `DSparkDecoder` (folded batched verify, greedy-parity), `DSparkPlacementPlanner`
> (+ shared `TierPlanner.ReservedVramBytes`, `LayerPlacement.CpuWeightBytes`), the four
> CLI flags/env vars from §5, and `SpecType.DSpark`. The confidence-threshold trim (§7's
> CLI reduction) shipped with the decoder. Still open: Phase 4 (GPU draft path +
> heterogeneous benchmarking — `--dspark-place gpu` currently downgrades to CPU with a
> note), Phase 5 (load-aware scheduling beyond the threshold trim), Phase 6 (server).
> Fetch weights with `download-model.ps1 -Model qwen3-4b` + `-Model dspark-qwen3-4b`.

## 1. Background

[DSpark](https://github.com/deepseek-ai/DeepSpec) ("Confidence-Scheduled Speculative
Decoding with Semi-Autoregressive Generation") is DeepSeek's June 2026 speculative-decoding
framework, released as part of `DeepSpec` (MIT). Three pieces:

1. **DFlash parallel backbone** — a distilled multi-block transformer (HF repos are named
   `dspark_<model>_block7`, i.e. 7 blocks) that predicts draft logits for **all** k draft
   positions in one forward pass, instead of a token-at-a-time chain.
2. **Markov head** — a lightweight rank-256 factorization applied sequentially on top of the
   parallel logits, adding a cheap prefix-dependent bias per position. This is what makes it
   "semi-autoregressive": most of the quality of a full autoregressive draft, without paying
   for k sequential transformer forwards.
3. **Confidence head + hardware-aware scheduler** — predicts per-position acceptance
   probability and dynamically trims how many draft positions the target actually verifies,
   based on live load.

Released draft heads (checked against the HF hub) target architectures we already load:

| Repo | Params | Target arch |
|---|---|---|
| `deepseek-ai/dspark_qwen3_4b_block7` | 1.39B | qwen3 |
| `deepseek-ai/dspark_qwen3_8b_block7` | 2.37B | qwen3 |
| `deepseek-ai/dspark_qwen3_14b_block7` | 3.42B | qwen3 |
| `deepseek-ai/dspark_gemma4_12b_block7` | 3.43B | gemma4_text |

Important correction vs. the "lightweight" marketing framing: these heads are **not** a
few-KB bias vector. They're safetensors checkpoints in the 1.4B–3.4B parameter range — closer
in weight-loading cost to a small second model than to the practically-free NEXTN/MTP head we
already support (a single block fused into the target checkpoint, sharing embeddings/lm_head,
see `MtpDecoder.cs`). That has direct consequences for a 12 GB-class card: the draft head is a
real VRAM/RAM line item, not a rounding error, so **where it runs is a first-class decision**,
not an afterthought. This spec is about that decision.

## 2. What we already have to build on

- `SpeculativeDecoder` (`src/SharpInference.Engine/SpeculativeDecoder.cs`) already accepts two
  independent `IForwardPass` instances (target, draft) as long as vocab sizes match, and
  already supports **mixed backends** in principle — nothing in its constructor requires
  target and draft to share a backend.
- `RunCommand.cs`'s existing `--draft-model` wiring (~line 994–1034) is the one precedent for
  choosing a draft backend today, and it's **naive**: for a CUDA target it spins up a second
  `CudaBackend` for the draft; for a CPU target it uses a CPU draft. For a **Vulkan** target,
  `--draft-model` is rejected outright (the `vulkanSpecTarget` guard at ~line 942–946 warns
  and falls back to normal, non-speculative generation) — it does *not* fall back to a CPU
  draft. So today there are exactly two reachable combinations (CUDA+CUDA, CPU+CPU), never a
  mixed CUDA-target/CPU-draft or Vulkan-target/CPU-draft pair, even though nothing in
  `SpeculativeDecoder` itself prevents it (see below). There is no "GPU target + CPU draft"
  option today even though the draft model is usually much smaller than the target and would
  often fit better on the side with spare capacity.
- `HardwareProfile.Detect(...)` (`src/SharpInference.Engine/HardwareProfile.cs`) already
  auto-detects `VramBytes`, `RamBytes`, `CpuCores`, `HasAvx512`, and measured PCIe bandwidth —
  everything a placement decision needs is already collected once per run.
- `TierPlanner.Plan(...)` (`src/SharpInference.Engine/TierPlanner.cs`) is the existing
  precedent for "auto-decide, but let an explicit user value pin it exactly" — the
  `pinGpuLayers` parameter overrides the greedy auto-packer and the result carries a
  human-readable `Summary()`. The DSpark placement planner should follow the same shape.
- `SpecType` (parsed in `RunCommand.ParseSpecType`, ~line 1924) is the existing `auto|none|mtp`
  CLI enum pattern; adding a `DSpark` case is the natural extension point.

## 3. Placement options

A DSpark draft head can run in one of four modes:

| Mode | Where the draft backbone runs | When it makes sense |
|---|---|---|
| `Off` | n/a (DSpark disabled) | No head available for this target arch, or neither GPU nor CPU has room, or user opted out. |
| `Gpu` | Same GPU as the target, own backend instance (mirrors the existing `--draft-model` CUDA branch's "own `CudaBackend`, own exec-graph state" lesson) | Target's resident footprint (weights + KV at requested ctx) leaves enough free VRAM for the draft head + its own KV/scratch. |
| `Cpu` | `CpuBackend`, using system RAM (mirrors the existing `--draft-model` CPU branch) | VRAM is too tight to add the draft head, but RAM has room and CPU throughput (AVX2/AVX-512 cores) can keep the draft chain from becoming the bottleneck. **New**: usable even when the target itself is on GPU — today's code only reaches the CPU draft branch when the target is *also* CPU. |
| `Auto` (default) | Planner decides between `Gpu`/`Cpu`/`Off` per the algorithm below | No explicit user choice. |

## 4. Auto-placement algorithm

New type, `DSparkPlacementPlanner` in `SharpInference.Engine`, mirroring `TierPlanner`'s shape.
It should call a shared `TierPlanner.ReservedVramBytes(long vramTotal)` helper (factor the
existing inline `Math.Max(vramTotal / 10, 512L * 1024 * 1024)` out of `TierPlanner.Plan` into
one) rather than re-deriving the 10%-or-512MB floor independently — otherwise the two planners
drift the moment either heuristic changes.

```csharp
public enum DSparkPlacement { Auto, Gpu, Cpu, Off }

public sealed record DSparkPlacementDecision(
    DSparkPlacement Placement,
    string Reason,          // human-readable rationale, printed like LayerPlacement.Summary()
    long DraftHeadBytes,    // resident cost at its native quant
    long HeadroomBytes);    // free budget in the chosen location after the decision

public static class DSparkPlacementPlanner
{
    public static DSparkPlacementDecision Plan(
        HardwareProfile hardware,
        LayerPlacement targetPlacement,   // the TierPlanner.Plan result already computed for the target
        long draftHeadBytesGpuQuant,      // draft head resident size if placed on GPU (native quant)
        long draftHeadBytesCpuQuant,      // usually == GGUF/safetensors on-disk size (mmap, no dequant)
        DSparkPlacement userOverride = DSparkPlacement.Auto)
    { /* ... */ }
}
```

Decision logic (the `Auto` branch; `userOverride != Auto` short-circuits straight to the
requested mode, see §5):

1. **No head for this architecture → `Off`.** (Checked by the caller before invoking the
   planner at all — e.g. no `dspark_*` release exists for `qwen35moe`/MoE targets today.)
2. **Compute GPU headroom.** `vramFree = hardware.VramBytes - targetPlacement.GpuWeightBytes
   - targetPlacement.GpuKvBytes - targetPlacement.ExpertCacheBudgetBytes (if MoE) - reserved`,
   using the same 10%-or-512MB `reserved` floor `TierPlanner` already uses. If
   `vramFree >= draftHeadBytesGpuQuant * 1.15` (15% margin for the draft's own KV/scratch,
   mirroring the scratch-reservation pattern in `TierPlanner.Plan`) → **`Gpu`**.
3. **Else compute RAM headroom.** `LayerPlacement` today only exposes `GpuWeightBytes`/
   `GpuKvBytes`/`CpuLayers` (a count, not a byte size) — it has no per-CPU-layer weight-byte
   field, so this step needs one new piece of plumbing: either add a `CpuWeightBytes` field to
   `LayerPlacement` (computed the same way `TierPlanner.Plan` already sums `GpuWeightBytes`,
   just for the layers it *didn't* place on GPU) or have the caller re-derive it via
   `TierPlanner`'s existing `MeasureLayerBytes` helper for `targetPlacement.CpuLayers` layers.
   With that available: `ramFree = hardware.RamBytes - cpuResidentTrunkBytes - reserved`. If
   `ramFree >= draftHeadBytesCpuQuant * 1.15` **and** `hardware.CpuCores >= 4` (a floor below
   which a sequential-ish per-token draft chain would itself become the bottleneck) → **`Cpu`**.
4. **Else `Off`**, with a `Reason` explaining which budget was insufficient (VRAM, RAM, or
   both) — printed the same way `TierPlanner`'s silent-clamp warnings are today
   (`RunCommand.cs` prints when a requested draft chain exceeds ring capacity; DSpark should
   do the same instead of failing silently).

Two refinements worth calling out because they're **not** just "bigger number wins":

- **PCIe cost of `Gpu` vs `Cpu` isn't symmetric with `--draft-model`'s existing CPU case.**
  When target=GPU and draft=CPU, every draft step round-trips the current hidden
  state/token across PCIe and back for verification — unlike the existing all-CPU draft
  path (target and draft both CPU, no PCIe at all) or the existing all-GPU draft path (no
  host round-trip). `hardware.EstPcieBandwidthGBps` (already measured by
  `HardwareProfile.Detect(CudaBackend)` via a real pinned-copy probe, not just guessed) feeds
  a rough per-step latency estimate; if that estimate would exceed the batched-verify time
  saved, `Auto` should prefer `Off` over a `Cpu` placement that's a net loss. This needs
  calibration against a real acceptance-rate benchmark before it's trustworthy — flag as
  Phase 4 work, ship a conservative always-`Gpu`-else-`Off` `Auto` first and add the
  `Cpu`-when-VRAM-too-tight branch once we've measured real numbers on the 4070 Ti.
- **Runtime-measured VRAM, not just the static estimate.** `TierPlanner.Plan` already notes
  its expert-cache budget is "diagnostic only; the runtime SLRU/CPU-MoE decision is made later
  from actual free VRAM" (issue #215). DSpark placement should follow the same two-stage
  pattern: `Auto` picks a *starting* placement from the static estimate before any model is
  loaded, but the actual draft-head allocation should re-check real free VRAM (CUDA
  `cudaMemGetInfo`-equivalent) right before allocating, and fall back to `Cpu`/`Off` if the
  static estimate undershot fragmentation/driver overhead. Concretely, this means
  `DSparkPlacementDecision` is not the last word: the code that actually constructs the draft
  head's backend (the `DSparkDecoder` init path, Phase 2/3) must re-run the free-VRAM check
  immediately before allocating and downgrade `Gpu → Cpu → Off` on the spot if the static
  decision no longer holds — the planner's record isn't self-enforcing, so this recheck has to
  be called out as a concrete step in Phase 3, not left implicit.

## 5. User overrides

Mirroring the existing `--spec-draft-n-max`/`--spec-draft-p-min`/`SHARPI_MTP_*` conventions
(CLI flag + env var fallback + explicit value always wins over auto):

| CLI flag | Env var fallback | Meaning | Precedent |
|---|---|---|---|
| `--dspark-model <path>` | — | Path to the converted draft-head weights (see §6 loader). Presence + a matching target arch is what turns DSpark on at all. | `--draft-model` |
| `--dspark-place auto\|gpu\|cpu\|off` | `SHARPI_DSPARK_PLACE` | Overrides `DSparkPlacementPlanner.Plan`'s decision outright — skips the VRAM/RAM math entirely, same as `pinGpuLayers` skips `TierPlanner`'s greedy packer. | `-g N` pinning `TierPlanner`'s `pinGpuLayers` |
| `--dspark-verify-len <n>` | `SHARPI_DSPARK_VERIFY_LEN` | Caps the confidence scheduler's per-step verify length (§7). `0`/unset = scheduler decides. | `--spec-draft-n-max` |
| `--dspark-min-confidence <p>` | `SHARPI_DSPARK_MIN_CONFIDENCE` | Floor on the confidence head's acceptance-probability estimate below which the scheduler trims a position off the verify batch. | `--spec-draft-p-min` |

Precedence, identical to the existing `pinGpuLayers` / `SpecDraftPMin` pattern: **explicit
flag > explicit env var > auto-detected/scheduled value > built-in default (`Off`)**. An
explicit `--dspark-place gpu` on a card too small for it is the user's call — same philosophy
`TierPlanner.Plan` documents for `pinGpuLayers` ("a pin that exceeds VRAM is the user's
explicit choice; the forward pass enforces real fit"). It should still print what it's doing
(free-VRAM headroom, or lack of it) rather than silently OOMing, same as the existing
`[mtp] requested draft chain ... clamping ...` console warnings in `RunCommand.cs`.

`SpecTypeStr` gains a `dspark` value alongside today's `auto|none|mtp`, parsed the same way in
`ParseSpecType`.

## 6. Non-placement work still required (context for scoping)

Placement is the piece this request is scoped to, but it only matters once DSpark can run at
all. For completeness, the surrounding work:

- **Safetensors loader.** The main inference path (Core/Engine/backends) is GGUF-only, but
  `SharpInference.Diffusion` already ships a `SafetensorsLoader` (single-file and multi-shard,
  used for FLUX/Z-Image weights) — reuse/extract that rather than writing a second, unrelated
  safetensors reader. It only needs to grow enough to pull the DFlash backbone + Markov head +
  confidence head tensors, not become a general safetensors-to-GGUF converter. Per CLAUDE.md's
  Build Constraints (trim/AOT analyzers enabled, no reflection-heavy patterns), the tensor-index
  JSON (`model.safetensors.index.json`) must be parsed via a source-generated `JsonSerializerContext`
  (extending `SharpInferenceJsonContext` or a sibling context) rather than reflection-based
  `System.Text.Json` deserialization — check how the existing `SafetensorsLoader` handles this
  before assuming it already does.
- **`DSparkForwardPass`** (or an extension of the existing MTP-head capability surface on
  `IForwardPass`): parallel backbone forward producing k positions' logits in one pass, then
  the rank-256 sequential bias correction. The exact Markov-head math isn't in any of the
  public write-ups we found — it needs to be reverse-engineered from the safetensors state-dict
  shapes/names (`DSpark_paper.pdf` in the DeepSpec repo is the primary source; not yet fetched
  in this pass since it's a binary PDF).
- **`DSparkDecoder`**, analogous to `MtpDecoder`/`SpeculativeDecoder`: reuses the existing
  `BatchVerify`/`BatchVerifyArgmax`/`RestoreBatchSnapshot` machinery for the verify+rollback
  step — that part needs no new engine capability, it's the same folded k-token batched verify
  already built for MTP.
- **Confidence-scheduled verify length (§7 below is placement-adjacent; the scheduler itself
  is separate work).**

## 7. Confidence-scheduled verify length (brief, since placement is the focus here)

DSpark's scheduler picks k (how many draft positions to verify) per step from the confidence
head's predicted acceptance probabilities plus a load signal. For the CLI single-user path,
load is ~constant, so it reduces to: trim trailing draft positions whose predicted confidence
is below `--dspark-min-confidence` before calling `BatchVerify`. For the server
(`ContinuousBatchingEngine`), load = current batch occupancy — a natural follow-on once the
CLI path is validated, analogous to how `SHARPI_KV_BUDGET_MB` already gates admission by a
live resource signal rather than a static config.

Note the coupling back to §4: the placement decision's `draftHeadBytesGpuQuant`/`Cpu` margin
math assumes a fixed `--dspark-verify-len` batch width `k`. Once the scheduler can trim `k`
per step, a placement that was marginal at the configured max `k` may in practice run at a
smaller effective `k` most of the time — Phase 5 should re-validate (not just extend) the
Phase 3 placement heuristics rather than treat them as already-settled.

## 8. Phased rollout

1. **Phase 0 — Discovery.** Fetch `DSpark_paper.pdf` from `deepseek-ai/DeepSpec`; inspect a
   `dspark_qwen3_4b_block7` safetensors state dict (tensor names/shapes) to confirm the
   backbone-block-count and Markov-head factorization match the paper's description.
2. **Phase 1 — Safetensors loader + `DSparkForwardPass` (CPU only).** Get a single draft step
   numerically working against Qwen3-4B on CPU, no placement logic yet (always CPU).
3. **Phase 2 — `DSparkDecoder`** wired to the existing `BatchVerify` rollback machinery;
   correctness parity tests against greedy CPU decoding (byte-identical output), mirroring
   `MtpDecoder`'s greedy-parity guarantee.
4. **Phase 3 — `DSparkPlacementPlanner` + CLI/env overrides** (this spec, §3–§5): `Gpu`/`Cpu`
   auto-decision from `HardwareProfile` + `TierPlanner`'s existing `LayerPlacement`, plus the
   four new flags and `SpecType.DSpark`.
5. **Phase 4 — CUDA draft path + real heterogeneous placement benchmarking** on the 4070
   Ti/64GB reference rig: measure actual PCIe round-trip cost for `Cpu`-draft/`Gpu`-target to
   calibrate the §4 "is `Cpu` even worth it" heuristic instead of shipping it as a guess.
6. **Phase 5 — Confidence-scheduled verify length** (§7), CLI first.
7. **Phase 6 — Server integration** (`ContinuousBatchingEngine`, load-aware scheduling).

Phases 0–3 are the minimum for "it runs and picks a sensible placement automatically, with
overrides." Phases 4–6 are where the reported 60–85% speedup claim would actually get
validated on our hardware instead of taken on faith.

## 9. Open risks

- Markov-head math is not publicly documented in detail; Phase 0 may reveal it needs more
  reverse-engineering than a rank-256 bias add, which would push Phase 1 effort up.
- No DSpark head exists for MoE architectures (`qwen35moe`, `qwen3-coder-30b-a3b`) — this
  entire feature is inapplicable to those targets regardless of placement smarts.
- The §4 PCIe-cost heuristic for `Cpu` placement is a guess until Phase 4 produces real
  numbers; shipping `Auto` before that should default conservatively (prefer `Gpu`-or-`Off`,
  treat `Cpu` as opt-in-only via `--dspark-place cpu`) rather than assume it's always a win.
