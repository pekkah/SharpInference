# Issue #156 decode A/B: Qwen3-8B Q4_K CUDA, interleaved (SHARPI_Q4K_SOA=0) vs
# scale-pre-unpacked SoA decode matvec + prefill MMQ (SHARPI_Q4K_SOA=1). Same prompt,
# warm, --temp 0, 160-token decode. Reports prefill + decode t/s per config.
$ErrorActionPreference = "Continue"
$M = "C:\p\sharpi\models\Qwen3-8B-Q4_K_M.gguf"
$exe = "C:\p\sharpi\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.exe"
$prompt = "Explain in detail how memory bandwidth, not arithmetic throughput, sets the ceiling on autoregressive LLM decode speed, and what quantization and kernel techniques narrow the gap."

function Run($tag, $soa) {
    $env:SHARPI_Q4K_SOA = $soa
    $out = & $exe -m $M -p $prompt --temp 0 -g -1 --backend cuda -n 160 2>&1 | Out-String
    $pre = ($out | Select-String -Pattern "Prefill:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s").Matches
    $dec = ($out | Select-String -Pattern "Decode:\s*(\d+)\s*tokens,\s*([\d.]+)\s*t/s").Matches
    $p = if ($pre.Count) { "{0} tok @ {1} t/s" -f $pre[0].Groups[1].Value, $pre[0].Groups[2].Value } else { "n/a" }
    $d = if ($dec.Count) { "{0} tok @ {1} t/s" -f $dec[0].Groups[1].Value, $dec[0].Groups[2].Value } else { "n/a" }
    Write-Host ("[{0}] SOA={1}  prefill {2} | decode {3}" -f $tag, $soa, $p, $d) -ForegroundColor Green
}

Write-Host "--- warm ---" -ForegroundColor DarkGray
$env:SHARPI_Q4K_SOA = "0"
$null = & $exe -m $M -p "Hello." --temp 0 -g -1 --backend cuda -n 4 2>&1 | Out-String

Run "AoS" "0"
Run "SoA" "1"
Run "AoS" "0"
Run "SoA" "1"
