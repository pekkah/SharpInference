# Focused re-baseline of the all-GPU Gemma 4 prefill/decode on merged master,
# before the #146 full-TC-flash work. Same ~1K prompt + warm-discard methodology
# as bench-gemma4-136.ps1, but only the all-GPU (-g -1) cell, run twice (keep 2nd).
param(
    [string]$ModelDir = "E:\models"
)
$ErrorActionPreference = "Continue"
$model = Join-Path $ModelDir "gemma-4-E4B-it-Q8_0.gguf"

# Identical ~1K-token prompt to bench-allrows-1k.ps1 / bench-gemma4-136.ps1.
$para = @(
"Modern large language model inference is dominated by memory bandwidth rather than raw compute.",
"Each decode step streams the full weight matrix for every layer, so quantization formats like Q4_K and Q5_K trade a small accuracy loss for a large reduction in bytes moved per token.",
"Mixture-of-experts models complicate this: only a handful of the hundreds of experts fire per token, and which fire varies token to token, defeating simple weight caching.",
"Gated DeltaNet layers replace quadratic attention with a linear recurrent state update, bounding per-token cost as context grows, at the price of a strictly sequential scan.",
"Hybrid placement keeps the attention and recurrent trunk on the accelerator while streaming routed-expert weights from host memory, overlapping the two so neither stalls."
) -join " "
$sb = [System.Text.StringBuilder]::new()
[void]$sb.Append("Read the following engineering notes and then write a concise technical summary.`n`n")
for ($i = 1; $i -le 6; $i++) { [void]$sb.Append("Section $i. $para`n`n") }
[void]$sb.Append("Summarize the main performance trade-offs across the sections above.")
$prompt = $sb.ToString()

$cudaArgs = @("-g","-1","--backend","cuda","--ctx-size","2048")

Write-Host "--- warming model ---" -ForegroundColor DarkGray
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-rb-warm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec 900 -ExtraArgs $cudaArgs

# Run twice, keep the second (warm).
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-rb-w" -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $cudaArgs
$r = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-rb" -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $cudaArgs

Write-Host ""
Write-Host "=== Gemma 4 all-GPU re-baseline (merged master, ~1K ctx, warm) ===" -ForegroundColor Cyan
[PSCustomObject]@{
    PrefillTok = $r.PrefillTok
    PrefillTps = $r.PrefillTps
    DecodeTps  = $r.DecodeTps
    WallSec    = $r.WallSec
    TimedOut   = $r.TimedOut
    Sample     = $r.Sample
} | Format-List
