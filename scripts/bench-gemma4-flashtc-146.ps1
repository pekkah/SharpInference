# A/B: half2 flash prefill (default) vs tensor-core flash prefill (#146,
# SHARPI_PREFILL_FLASH_TC=1). Same ~1K prompt + warm-discard as bench-gemma4-136.ps1,
# all-GPU (-g -1). Reports prefill t/s for each so the TC kernel can be judged
# against the half2 baseline (re-baselined at 2589 t/s on merged master).
param(
    [string]$ModelDir = "E:\models",
    [int]$Sections = 6
)
$ErrorActionPreference = "Continue"
$model = Join-Path $ModelDir "gemma-4-E4B-it-Q8_0.gguf"

$para = @(
"Modern large language model inference is dominated by memory bandwidth rather than raw compute.",
"Each decode step streams the full weight matrix for every layer, so quantization formats like Q4_K and Q5_K trade a small accuracy loss for a large reduction in bytes moved per token.",
"Mixture-of-experts models complicate this: only a handful of the hundreds of experts fire per token, and which fire varies token to token, defeating simple weight caching.",
"Gated DeltaNet layers replace quadratic attention with a linear recurrent state update, bounding per-token cost as context grows, at the price of a strictly sequential scan.",
"Hybrid placement keeps the attention and recurrent trunk on the accelerator while streaming routed-expert weights from host memory, overlapping the two so neither stalls."
) -join " "
$sb = [System.Text.StringBuilder]::new()
[void]$sb.Append("Read the following engineering notes and then write a concise technical summary.`n`n")
for ($i = 1; $i -le $Sections; $i++) { [void]$sb.Append("Section $i. $para`n`n") }
[void]$sb.Append("Summarize the main performance trade-offs across the sections above.")
$prompt = $sb.ToString()

$cudaArgs = @("-g","-1","--backend","cuda","--ctx-size","2048")

Write-Host "--- warming model ---" -ForegroundColor DarkGray
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "ftc-warm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec 900 -ExtraArgs $cudaArgs

$rows = @()
function Run($tag, $tcOn) {
    try {
        if ($tcOn) { $env:SHARPI_PREFILL_FLASH_TC = "1" } else { Remove-Item env:SHARPI_PREFILL_FLASH_TC -ErrorAction SilentlyContinue }
        $null = .\scripts\bench-textgen.ps1 -Model $model -Tag "$tag-w" -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $cudaArgs
        $r = .\scripts\bench-textgen.ps1 -Model $model -Tag $tag -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $cudaArgs
        $script:rows += [PSCustomObject]@{ Tag=$tag; PrefTok=$r.PrefillTok; PrefillTps=$r.PrefillTps; DecodeTps=$r.DecodeTps; Wall=$r.WallSec; TO=$r.TimedOut; Sample=$r.Sample }
        Write-Host ("  {0,-18} pref={1,7} t/s  dec={2,6} t/s  ({3} tok)" -f $tag,$r.PrefillTps,$r.DecodeTps,$r.PrefillTok) -ForegroundColor Green
    } finally {
        Remove-Item env:SHARPI_PREFILL_FLASH_TC -ErrorAction SilentlyContinue
    }
}

Run "flash-half2" $false
Run "flash-tc"    $true

Write-Host ""
Write-Host "=== Gemma 4 flash half2 vs TC (#146, ~1K ctx, warm) ===" -ForegroundColor Cyan
$rows | Format-Table Tag,PrefTok,PrefillTps,DecodeTps,Wall,TO -AutoSize
$rows | ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Tag, $_.Sample) -ForegroundColor DarkGray }
