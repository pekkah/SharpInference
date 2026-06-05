# Targeted re-measure of the all-GPU Gemma 4 README row after #136 (batched-trunk
# prefill + launch fusions). Same ~1K prompt + warm-cache methodology as
# bench-allrows-1k.ps1. Captures batched-on (default), batched-off (A/B for the
# prefill delta), and the hybrid row (expected flat).
$ErrorActionPreference = "Continue"
$E = "E:\models"
$model = "$E\gemma-4-E4B-it-Q8_0.gguf"

# Identical ~1K-token prompt to bench-allrows-1k.ps1.
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
$hybArgs  = @("-g","22","--backend","cuda","--ctx-size","2048")

# Warm OS page cache once (discarded).
Write-Host "--- warming model ---" -ForegroundColor DarkGray
$null = .\scripts\bench-textgen.ps1 -Model $model -Tag "g4-warm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec 900 -ExtraArgs $cudaArgs

$rows = @()
function Run($tag, $a, $envOff) {
    if ($envOff) { $env:SHARPI_BATCHED_PREFILL = "0" } else { Remove-Item env:SHARPI_BATCHED_PREFILL -ErrorAction SilentlyContinue }
    # Run twice, keep the second (warm) — matches the discard-first guidance.
    $null = .\scripts\bench-textgen.ps1 -Model $model -Tag "$tag-w" -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $a
    $r = .\scripts\bench-textgen.ps1 -Model $model -Tag $tag -NTokens 60 -Prompt $prompt -TimeoutSec 900 -ExtraArgs $a
    Remove-Item env:SHARPI_BATCHED_PREFILL -ErrorAction SilentlyContinue
    $script:rows += [PSCustomObject]@{ Tag=$tag; PrefTok=$r.PrefillTok; PrefillTps=$r.PrefillTps; DecodeTps=$r.DecodeTps; Wall=$r.WallSec; TO=$r.TimedOut }
    Write-Host ("  {0,-22} pref={1,7} t/s  dec={2,6} t/s  ({3} tok)" -f $tag,$r.PrefillTps,$r.DecodeTps,$r.PrefillTok) -ForegroundColor Green
}

Run "gemma4-cuda-batched"    $cudaArgs $false
Run "gemma4-cuda-seq"        $cudaArgs $true
Run "gemma4-cuda-hyb"        $hybArgs  $false

Write-Host ""
Write-Host "=== Gemma 4 #136 re-measure (~1K ctx, warm) ===" -ForegroundColor Cyan
$rows | Format-Table -AutoSize
