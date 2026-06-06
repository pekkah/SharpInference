# Issue #156 Item C2 A/B: Qwen3-8B Q4_K CUDA prefill, C1 (dequant->fp16->cuBLAS GEMM,
# SHARPI_PREFILL_MMQ=0) vs C2 (int8 MMQ, default), same ~1K prompt, warm cache.
$ErrorActionPreference = "Continue"
$M = "C:\p\sharpi\models\Qwen3-8B-Q4_K_M.gguf"
$exe = "C:\p\sharpi\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.exe"

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

function Run($tag, $mmq) {
    $env:SHARPI_PREFILL_MMQ = $mmq
    $out = & $exe -m $M -p $prompt --temp 0 -g -1 --backend cuda -n 8 2>&1 | Out-String
    $pre = ($out | Select-String -Pattern "Prefill:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s").Matches
    if ($pre.Count -gt 0) {
        $ntok = $pre[0].Groups[1].Value; $tps = $pre[0].Groups[2].Value
        Write-Host ("[{0}] MMQ={1}  prefill {2} tok @ {3} t/s" -f $tag, $mmq, $ntok, $tps) -ForegroundColor Green
    } else { Write-Host "[$tag] MMQ=$mmq  (no prefill line)`n$out" -ForegroundColor Red }
}

Write-Host "--- warm ---" -ForegroundColor DarkGray
$env:SHARPI_PREFILL_MMQ = "1"
$null = & $exe -m $M -p "Hello, world." --temp 0 -g -1 --backend cuda -n 4 2>&1 | Out-String

Run "C1-gemm"  "0"
Run "C2-mmq"   "1"
Run "C1-gemm"  "0"
Run "C2-mmq"   "1"
