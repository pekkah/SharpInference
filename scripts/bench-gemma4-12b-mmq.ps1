# Issue #124/#173: Gemma 4 12B QAT q4_0 all-GPU prefill MMQ A/B.
# MMQ ON (default) vs SHARPI_PREFILL_MMQ=0 (dequant->fp16->cuBLAS GEMM). Greedy
# samples must match (argmax-stable). Warm-run-twice, keep the 2nd.
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

# MMQ ON (default).
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-mmq1-w" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
$rOn = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-mmq1" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs

# MMQ OFF (dequant->fp16->cuBLAS GEMM).
$env:SHARPI_PREFILL_MMQ = "0"
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-mmq0-w" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
$rOff = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-12b-mmq0" -NTokens 60 -Prompt $prompt -TimeoutSec 1200 -ExtraArgs $cudaArgs
Remove-Item Env:\SHARPI_PREFILL_MMQ

Write-Host ""
Write-Host "=== Gemma 4 12B QAT q4_0 all-GPU prefill MMQ A/B (issue #124/#173) ===" -ForegroundColor Cyan
@(
    [PSCustomObject]@{ Cfg="MMQ-ON (int8 TC)";   PrefillTok=$rOn.PrefillTok;  PrefillTps=$rOn.PrefillTps;  DecodeTps=$rOn.DecodeTps }
    [PSCustomObject]@{ Cfg="MMQ-OFF (fp16 GEMM)"; PrefillTok=$rOff.PrefillTok; PrefillTps=$rOff.PrefillTps; DecodeTps=$rOff.DecodeTps }
) | Format-Table -AutoSize

$match = $rOn.Sample -eq $rOff.Sample
Write-Host "Greedy sample match (argmax-stable): $match" -ForegroundColor $(if ($match) {"Green"} else {"Red"})
Write-Host "  ON : $($rOn.Sample)"  -ForegroundColor DarkGray
Write-Host "  OFF: $($rOff.Sample)" -ForegroundColor DarkGray
