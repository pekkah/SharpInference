# Re-benchmark every on-disk README row at a uniform "normal" working context
# (~2K-token realistic prompt, see below), warm-cache, current code (#114-B batched
# trunk on by default). Goal: make the README "Prefill t/s" column consistent (warm
# @ a realistic ctx) instead of the old ~10-token launch-overhead cells. The default
# run gives the prefill column; the README "Decode t/s" column is the near-zero-ctx
# headline from the -NearZero run (see below) — the default run's decode is at ~2K
# ctx and is captured only for reference.
#
# Each row carries that model's recommended sampling (Temp + Samp), matching the
# README per-model command blocks (upstream / llama.cpp defaults) instead of a
# uniform greedy --temp 0. Throughput is ~independent of sampling values, but this
# keeps the bench faithful to the published run commands. MTP rows stay at --temp 0
# --no-thinking on purpose: greedy is what engages MTP self-speculative decoding.
#
# -CudaOnly restricts the sweep to the CUDA rows (the README CPU/Vulkan numbers are
# unaffected by this code path, and re-running large models on CPU is slow/noisy).
#
# -NearZero swaps the ~2K prompt for a short one so the *decode* rate is measured at
# near-zero context (the README "Decode t/s" headline metric, comparable to llama.cpp
# tg128). Prefill from a near-zero run is launch-overhead-dominated and must be
# ignored — take the prefill column from the default (~2K) run and the decode column
# from the -NearZero run. The per-model long-context falloff lives in the separate
# "Long-context decode" README section, not in the table.
param([switch]$CudaOnly, [switch]$NearZero)
$ErrorActionPreference = "Continue"

$C = "C:\p\sharpi\models"
$E = "E:\models"

# Realistic ~2K-token prompt: a detailed design-review / analysis request in varied
# prose — representative of real interactive / agentic working context, not artificial
# repetition. Prefill is measured at this "normal" working-context size. NOTE: this
# prompt deliberately avoids a verbatim source-code block. A pasted code block (heavy
# indentation) reproducibly crashes the SmolLM2 CUDA prefill path with CUDA error 700
# (illegal address) while CPU handles it fine — a model-specific token-id bug tracked
# separately. Keeping the bench prompt code-free makes it uniform across all models.
$prompt = @'
You are advising the team behind a high-performance language-model inference engine
written in C#. They are about to merge a substantial rework of the key/value cache
and want a thorough design review before it lands. Read the description below and
respond with a structured analysis: the main correctness risks, the concurrency
hazards, the performance trade-offs, and the ways the new abstraction could make
planned future work harder than it needs to be. Be concrete and prioritize.

First, the context the change lives in. During generation the engine is dominated by
memory bandwidth rather than raw arithmetic. Every decode step has to stream the
weights of every layer through the memory system to produce a single token, so the
formats used to store weights, and the way the cache of past attention state is laid
out and allocated, both have first-order effects on throughput. The cache of keys
and values is read on every attention layer for every token generated, and it grows
in proportion to the length of the conversation. A cache layout that is friendly to
the hardware on short prompts can become a bottleneck on long ones, and an allocation
strategy that is cheap at steady state can still stall the very first token if it has
to reserve a large contiguous region before generation can begin.

The previous implementation reserved, for each layer, a single contiguous buffer in
full single-precision sized to the maximum supported context length. This was simple
and predictable, but it had two serious drawbacks. For the common case of a short
chat of a few hundred tokens it wasted gigabytes of device memory that were never
touched, and for the largest supported context windows it simply failed to fit inside
the memory budget of a twelve-gigabyte accelerator, so those configurations could not
be served at all. The team's goal with the rework is to cut steady-state memory for
short conversations dramatically while leaving decode throughput on long contexts
unchanged, and to make the very large context windows fit by spending memory only on
the positions that are actually written.

The new design breaks each layer's cache into fixed-size pages, each holding a small
run of consecutive positions. A page is allocated only on the first write that lands
inside it, and the pages belonging to a sequence are tracked in a per-sequence page
table that maps a logical position to the physical page that backs it. Two operations
make the scheme more than a simple bump allocator. The first is truncation. When a
new turn in a conversation reuses a prefix that was already processed, the engine
truncates the sequence back to the shared prefix instead of recomputing it. This
truncation is deliberately soft: the logical length is moved back, but the physical
pages past the new end are kept alive, so a later turn that extends the sequence again
can write straight back into them without paying for reallocation. The second is
pooling. When a sequence finishes for good, its pages are not freed immediately;
instead they are pushed onto a process-wide warm pool, organized per layer, and the
next sequence that needs a page for a given layer takes one from the pool before
asking the device allocator for fresh memory. The pool is capped, and pages beyond the
cap are returned to the allocator so that the resident set does not grow without bound.

There are three interactions the review should examine closely. The first is sharing
a prefix across two simultaneous requests. If two requests begin with the same long
system preamble, it is tempting to let them share the physical pages that hold the
prefilled state for that preamble, but the page table records raw page references by
value per sequence, so it is not obvious whether a later write performed on behalf of
one request can quietly modify a page that the other request still treats as frozen
and shared. The second is the bookkeeping for the warm pool. The count that enforces
the cap is consulted in one code path while holding a lock and in another without one,
which raises the question of whether two sequences finishing or starting at the same
moment can corrupt the count, double-return a page, or hand the same page to two
different sequences. The third is the relationship between the logical length and the
physical extent after a soft truncation. Because truncation leaves pages in place, the
physical span of allocated positions can exceed the logical length, and the attention
computation must scan exactly the logical length and never the stale tail; whether
that holds depends on which quantity is treated as the authority when the per-token
work is dispatched.

There are also two performance questions that sit underneath the correctness ones,
because a fix that is correct but slow will not survive contact with the throughput
targets. The first concerns the granularity of the pages. Small pages keep the wasted
tail at the end of a short conversation tiny, but they multiply the number of separate
allocations and the length of the per-layer page table, and they scatter the cache
across many non-contiguous regions, which can hurt the access pattern of the attention
kernel that walks the keys and values during decode. Large pages do the opposite: they
restore locality and shorten the page table, but they coarsen the wasted tail and make
the warm pool less able to satisfy an arbitrary request from recycled memory. The team
picked a small page size mostly on intuition, and they would like a principled way to
reason about the trade-off, ideally one that ties the choice to the head dimension, the
number of layers, and the distribution of conversation lengths they actually observe in
production rather than to a round number that looked reasonable.

The second performance question concerns the interaction with the rest of the runtime.
The engine already supports batched decode, where several independent sequences advance
one token at a time inside a single dispatch so that the cost of reading each layer's
weights is amortized across all of them. With the old contiguous cache the address of
any position in any sequence was a simple function of a base pointer and a stride, which
made it cheap to hand the kernel everything it needed as a few scalar arguments. With
the paged cache the mapping from a logical position to a physical address now involves
a per-sequence, per-layer indirection through the page table, and that table has to be
communicated to the kernel somehow on every step. The reviewer should think about how
the page tables for all the in-flight sequences are marshalled to the device each step,
whether that marshalling can become the new bottleneck once the weight reads are well
amortized, and whether the indirection defeats any of the coalescing the old layout got
for free. It is worth being explicit about whether the proposed design keeps batched
decode viable at all, or whether it quietly forces a fallback to one-sequence-at-a-time
generation whenever the paged cache is in use, because that would erase a large fraction
of the multi-user throughput the engine is supposed to deliver.

With that background, please answer the following. Is the warm-pool accounting correct
and safe when several sequences share the single process-wide pool concurrently, and
if not, what is the smallest synchronization change that fixes it without serializing
the common fast path? Does the combination of soft truncation and lazy page allocation
create any situation in which a reused prefix is read after it has been partially
overwritten, or in which a sequence reads a page it does not actually own? When a
finished sequence returns its pages to the pool, is there any window in which a page
that is still referenced by work that has been dispatched but not yet completed could
be handed out to a new sequence, and how would you close that window? Is the logical
length the right and only authority for how many positions the attention step should
consider, and what invariant would you add so that a future change cannot accidentally
let the physical extent leak into that calculation? How should the team choose the page
size, and does the page-table indirection threaten the batched-decode path? For each
issue, say whether you consider it blocking for the merge or acceptable as a tracked
follow-up, and recommend the minimal set of changes that closes the real correctness
gaps while preserving both the lazy allocation and the prefix-reuse fast path.
'@

# Near-zero-ctx decode: a short open-ended prompt so generation starts at ~near-zero
# context. Decode rate is ~independent of prompt content (only ctx length matters), so
# one neutral prompt suffices across models; it is open-ended enough that reasoning
# models think and non-reasoning models answer without an immediate EOS.
if ($NearZero) {
    $prompt = "Explain, step by step, how a modern CPU executes a single instruction, from fetch to retire."
}
$runSuffix = if ($NearZero) { "nz" } else { "1k" }

# Recommended sampling presets (mirror the README per-model command blocks).
$sQwen   = @("--top-p","0.95","--top-k","20")          # Qwen3 / Qwen3.6 reasoning defaults
$sOlmoe  = @("--top-p","0.95")                          # OLMoE: greedy unstable, sample
$sSmol   = @("--top-p","0.95","--top-k","40")          # SmolLM2 instruct
$sCoder  = @("--top-p","0.8","--top-k","20","--repeat-penalty","1.05")  # Qwen3-Coder
$sGemma  = @("--top-k","64","--top-p","0.95","--min-p","0")             # Gemma 3/4 defaults

# Job table: Tag, Model, Args, recommended Temp/Samp, Env (CPU_MOE), Timeout. Order
# groups by model file so the first run of each model warms the OS page cache for
# the ones after it. MTP rows keep Temp=0 + --no-thinking (engages MTP self-spec).
$jobs = @(
  @{ Tag="smol-cpu";        M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@();                                               Temp="0.7"; Samp=$sSmol;  T=300 }
  @{ Tag="smol-vulkan";     M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");                 Temp="0.7"; Samp=$sSmol;  T=300 }
  @{ Tag="smol-cuda";       M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                   Temp="0.7"; Samp=$sSmol;  T=300 }

  @{ Tag="qwen3-cpu";       M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@();                                                            Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-cpu-tq";    M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("--tq");                                                      Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-vulkan";    M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");                             Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-vulkan-tq"; M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan","--tq");                      Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-cuda";      M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                               Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-cuda-nt";   M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");               Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-cuda-tq";   M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--tq");                        Temp="0.6"; Samp=$sQwen;  T=400 }
  @{ Tag="qwen3-cuda-tq-nt";M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--tq","--no-thinking");        Temp="0.6"; Samp=$sQwen;  T=400 }

  @{ Tag="olmoe-cpu";       M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@();                                          Temp="0.6"; Samp=$sOlmoe; T=400 }
  @{ Tag="olmoe-vulkan";    M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");            Temp="0.6"; Samp=$sOlmoe; T=400 }
  @{ Tag="olmoe-cuda";      M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");              Temp="0.6"; Samp=$sOlmoe; T=400 }

  @{ Tag="coder-cpu";       M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@();                                       Temp="0.7"; Samp=$sCoder; T=600 }
  @{ Tag="coder-cpu-tq";    M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("--tq");                                 Temp="0.7"; Samp=$sCoder; T=600 }
  @{ Tag="coder-vulkan";    M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");         Temp="0.7"; Samp=$sCoder; T=1800 }
  @{ Tag="coder-cuda";      M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");           Temp="0.7"; Samp=$sCoder; T=600 }

  @{ Tag="qwen36-35b-cpu";  M="$E\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"; A=@();                                                 Temp="0.6"; Samp=$sQwen;  T=900 }
  @{ Tag="qwen36-35b-cuda"; M="$E\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                     Temp="0.6"; Samp=$sQwen;  T=900 }

  @{ Tag="27b-mtp-q4-cpu";  M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf"; A=@("--no-thinking");                                     Temp="0"; Samp=@();       T=1200 }
  @{ Tag="27b-mtp-q4-cuda"; M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");        Temp="0"; Samp=@();       T=1200 }
  @{ Tag="27b-mtp-q5-cpu";  M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf"; A=@("--no-thinking");                                     Temp="0"; Samp=@();       T=1200 }
  @{ Tag="27b-mtp-q5-cuda"; M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");        Temp="0"; Samp=@();       T=1200 }

  @{ Tag="35b-mtp-cpu";     M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("--no-thinking");                       Temp="0"; Samp=@(); CpuMoe=$true; T=900 }
  @{ Tag="35b-mtp-cuda";    M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking"); Temp="0"; Samp=@(); CpuMoe=$true; T=900 }

  @{ Tag="carnice-cuda";    M="$E\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking"); Temp="0"; Samp=@(); CpuMoe=$true; T=900 }

  @{ Tag="gemma4-e4b-q4-cuda"; M="$E\gemma-4-E4B_q4_0-it.gguf"; A=@("-g","-1","--backend","cuda","--ctx-size","2048");    Temp="1.0"; Samp=$sGemma; T=900 }

  @{ Tag="gemma4-cpu";      M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@();                                                       Temp="1.0"; Samp=$sGemma; T=900 }
  @{ Tag="gemma4-cuda";     M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@("-g","-1","--backend","cuda","--ctx-size","2048");       Temp="1.0"; Samp=$sGemma; T=900 }
  @{ Tag="gemma4-cuda-hyb"; M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@("-g","22","--backend","cuda","--ctx-size","2048");       Temp="1.0"; Samp=$sGemma; T=900 }

  @{ Tag="gemma4-12b-cuda"; M="$E\gemma-4-12b-it-qat-q4_0.gguf"; A=@("-g","-1","--backend","cuda","--ctx-size","2048");  Temp="1.0"; Samp=$sGemma; T=900 }
)

$warmed = @{}
$rows = @()
foreach ($j in $jobs) {
    if ($CudaOnly -and ($j.A -notcontains "cuda")) { continue }
    if (-not (Test-Path $j.M)) { Write-Host "[skip] $($j.Tag): $($j.M) missing" -ForegroundColor Yellow; continue }
    if ($j.CpuMoe) { $env:SHARPI_CPU_MOE = "1" } else { Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue }

    # Recommended sampling: backend/-g args + this model's top-p/top-k/min-p/etc.
    $runArgs = $j.A + $j.Samp

    # Warm the OS page cache for this model file once (short prompt, discarded).
    if (-not $warmed.ContainsKey($j.M)) {
        Write-Host "--- warming $($j.Tag) model ---" -ForegroundColor DarkGray
        $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-warm1k" -NTokens 8 -Prompt "Hello, world." -Temp $j.Temp -TimeoutSec $j.T -ExtraArgs $runArgs
        $warmed[$j.M] = $true
    }

    # GPU jobs need a per-config warm-up: the first launch of each backend in this
    # session runs at idle boost clocks and under-measures by ~30%. Run the full
    # measured prompt once and discard it, then keep the second (warm-clock) run.
    # CPU jobs don't boost-warm and the page cache is already hot, so skip it for them.
    if ($j.A -contains "-g") {
        $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-warmclk" -NTokens 60 -Prompt $prompt -Temp $j.Temp -TimeoutSec $j.T -ExtraArgs $runArgs
    }

    $r = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-$runSuffix" -NTokens 60 -Prompt $prompt -Temp $j.Temp -TimeoutSec $j.T -ExtraArgs $runArgs
    Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue
    $rows += [PSCustomObject]@{ Tag=$j.Tag; PrefTok=$r.PrefillTok; PrefillTps=$r.PrefillTps; DecodeTps=$r.DecodeTps; Mtp=$r.MtpAccept; Wall=$r.WallSec; TO=$r.TimedOut }
    Write-Host ("  {0,-18} pref={1,7} t/s  dec={2,6} t/s  ({3} tok, {4}s{5})" -f $j.Tag,$r.PrefillTps,$r.DecodeTps,$r.PrefillTok,$r.WallSec,($(if($r.TimedOut){" TIMEOUT"}else{""}))) -ForegroundColor Green
}
Write-Host ""
$ctxLabel = if ($NearZero) { "near-zero ctx (decode headline)" } else { "~2K ctx (warm)" }
Write-Host "=== All-rows @ $ctxLabel ===" -ForegroundColor Cyan
$rows | Format-Table -AutoSize
$rows | Export-Csv -NoTypeInformation -Path "tools\bench\allrows-$runSuffix.csv"
Write-Host "CSV: tools\bench\allrows-$runSuffix.csv"
