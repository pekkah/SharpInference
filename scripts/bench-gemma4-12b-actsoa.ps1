# Track A (#124/#173): Gemma 4 12B QAT q4_0 all-GPU activation-SoA A/B.
# SHARPI_ACT_SOA=1 (SoA Q8_1 activations: llm_quantize_q8_1_soa + llm_mmq_*_soa_acts)
# vs unset (interleaved 36-B AoS Q8_1). Weights stay SoA (SHARPI_MMQ_SOA default on), so
# this isolates the ACTIVATION layout. Phase A keeps the SAME load mapping → greedy
# samples MUST match bit-for-bit; prefill t/s expected ~neutral (the coalescing win is
# Phase B). Warm-run-twice, keep the 2nd.
param(
    [string]$ModelDir = "E:\models"
)
$ErrorActionPreference = "Continue"
$model = Join-Path $ModelDir "gemma-4-12b-it-qat-q4_0.gguf"

$para = @(
"Modern large language model inference is dominated by memory bandwidth rather than raw compute.",
"Each decode step streams the full weight matrix for every layer, so quantization formats like Q4_0 and Q4_K trade a small accuracy loss for a large reduction in bytes moved per token.",
"Mixture-of-experts models complicate this: only a handful of the hundreds of experts fire per token, and which fire varies token to token, defeating simple weight caching.",
"Gated DeltaNet layers replace quadratic attention with a linear recurrent state update, bounding per-token cost as context grows, at the price of a strictly sequential scan.",
"Hybrid placement keeps the attention and recurrent trunk on the accelerator while streaming routed-expert weights from host memory, overlapping the two so neither stalls."
) -join " "
$sb = [System.Text.StringBuilder]::new()
[void]$sb.Append("Read the following engineering notes and then write a concise technical summary.`n`n")
for ($i = 1; $i -le 6; $i++) { [void]$sb.Append("Section $i. $para`n`n") }
[void]$sb.Append("Summarize the main performance trade-offs across the sections above.")
$prompt = $sb.ToString()

$cudaArgs = @("-g","-1","--backend","cuda","--ctx-size","4096")

Write-Host "--- warming model ---" -ForegroundColor DarkGray
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-warm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec 1200 -ExtraArgs $cudaArgs

# ACT-SoA OFF (interleaved AoS Q8_1 activations) — the current default.
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-acts0-w" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
$rOff = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-acts0" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs

# ACT-SoA ON (SoA Q8_1 activations).
$env:SHARPI_ACT_SOA = "1"
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-acts1-w" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
$rOn = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-acts1" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
Remove-Item Env:\SHARPI_ACT_SOA

Write-Host ""
Write-Host "=== Gemma 4 12B QAT q4_0 all-GPU activation-SoA A/B (Track A, #124/#173) ===" -ForegroundColor Cyan
@(
    [PSCustomObject]@{ Cfg="ACT-SoA-OFF (AoS Q8_1)"; PrefillTok=$rOff.PrefillTok; PrefillTps=$rOff.PrefillTps; DecodeTps=$rOff.DecodeTps }
    [PSCustomObject]@{ Cfg="ACT-SoA-ON  (SoA Q8_1)"; PrefillTok=$rOn.PrefillTok;  PrefillTps=$rOn.PrefillTps;  DecodeTps=$rOn.DecodeTps }
) | Format-Table -AutoSize

$match = $rOn.Sample -eq $rOff.Sample
Write-Host "Greedy sample match (Phase A bit-identical): $match" -ForegroundColor $(if ($match) {"Green"} else {"Red"})
Write-Host "  OFF: $($rOff.Sample)" -ForegroundColor DarkGray
Write-Host "  ON : $($rOn.Sample)"  -ForegroundColor DarkGray
