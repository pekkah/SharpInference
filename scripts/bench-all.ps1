param(
    [int]$NTokens = 80,
    [string]$Prompt = "Write a Python function that sorts a list using the quicksort algorithm:"
)

$results = @()

# SmolLM2 1.7B — dense, headDim 64 (no TQ), 3 backends
$smol = "models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
$results += .\scripts\bench-textgen.ps1 -Model $smol -Tag "smol-cpu"      -NTokens $NTokens -Prompt $Prompt -TimeoutSec 120
$results += .\scripts\bench-textgen.ps1 -Model $smol -Tag "smol-vulkan"   -NTokens $NTokens -Prompt $Prompt -TimeoutSec 120 -ExtraArgs @("-g","-1","--backend","vulkan")
$results += .\scripts\bench-textgen.ps1 -Model $smol -Tag "smol-cuda"     -NTokens $NTokens -Prompt $Prompt -TimeoutSec 120 -ExtraArgs @("-g","-1","--backend","cuda")

# Qwen3 8B — dense, headDim 128, supports TQ, reasoning model
$qwen = "models\Qwen3-8B-Q4_K_M.gguf"
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-vulkan"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","vulkan")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-vk-tq"          -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","vulkan","--tq")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-cuda"           -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","cuda")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-cuda-nothink"   -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-cuda-tq"        -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","cuda","--tq")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-cuda-tq-nothink" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","cuda","--tq","--no-thinking")

# Qwen3-Coder 30B-A3B MoE — CPU + Vulkan/CUDA hybrid (17 GB model doesn't fit a 12 GB card in full offload).
$moe = "models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-cpu"           -NTokens $NTokens -Prompt $Prompt -TimeoutSec 360
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-cpu-tq"        -NTokens $NTokens -Prompt $Prompt -TimeoutSec 360 -ExtraArgs @("--tq")
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-vulkan-hybrid" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 600 -ExtraArgs @("-g","-1","--backend","vulkan")
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-cuda-hybrid"   -NTokens $NTokens -Prompt $Prompt -TimeoutSec 600 -ExtraArgs @("-g","-1","--backend","cuda")

# Qwen3.6-35B-A3B GDN+MoE — hybrid Gated-DeltaNet/attention with 256 experts × 8 active.
# 22 GB on E:; doesn't fit a 12 GB card so MoE auto-routes to CPU under the CUDA hybrid path.
$qwen36 = "E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
if (Test-Path $qwen36) {
    $results += .\scripts\bench-textgen.ps1 -Model $qwen36 -Tag "qwen36-cpu"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 600
    $results += .\scripts\bench-textgen.ps1 -Model $qwen36 -Tag "qwen36-cuda-hybrid" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 600 -ExtraArgs @("-g","-1","--backend","cuda")
}

# Qwen3.6-27B-MTP — dense 27B with native MTP head (issue #25, parity oracle).
# Vulkan can't run hybrid GDN, so CPU and CUDA-hybrid only. The MTP path engages
# automatically when the model has a HasMtpHead, sampling is greedy, and
# --no-thinking is set (chat template renders enable_thinking=false). Memory
# project_mtp_n1_no_speedup documents that N=1 sequential gives no decode
# speedup vs MTP-off; we still bench both to make the gap concrete (issue #28).
$qwen36mtpQ4 = "E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf"
$qwen36mtpQ5 = "E:\models\Qwen3.6-27B-MTP-Q5_K_M.gguf"
foreach ($pair in @(
        @{ Path = $qwen36mtpQ4; QuantTag = "q4" },
        @{ Path = $qwen36mtpQ5; QuantTag = "q5" })) {
    if (Test-Path $pair.Path) {
        $qt = $pair.QuantTag
        $results += .\scripts\bench-textgen.ps1 -Model $pair.Path -Tag "qwen36-27b-mtp-$qt-cpu-mtp"             -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
        $results += .\scripts\bench-textgen.ps1 -Model $pair.Path -Tag "qwen36-27b-mtp-$qt-cuda-hybrid-mtp"     -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
        # MTP-disabled pair, same chat template, to quantify the N=1 no-speedup gap.
        $env:SHARPI_DISABLE_MTP = "1"
        $results += .\scripts\bench-textgen.ps1 -Model $pair.Path -Tag "qwen36-27b-mtp-$qt-cpu-nomtp"           -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
        $results += .\scripts\bench-textgen.ps1 -Model $pair.Path -Tag "qwen36-27b-mtp-$qt-cuda-hybrid-nomtp"   -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
        Remove-Item Env:\SHARPI_DISABLE_MTP -ErrorAction SilentlyContinue
    }
}

# Llama-4-Scout 17B-16E MoE — split GGUF on E: drive (~61 GB total at Q4_K_M; CPU-only on a 12 GB card).
$scout = "E:\models\Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf"
if (Test-Path $scout) {
    $results += .\scripts\bench-textgen.ps1 -Model $scout -Tag "scout-cpu"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 1200
    $results += .\scripts\bench-textgen.ps1 -Model $scout -Tag "scout-cuda-hybrid" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 1200 -ExtraArgs @("-g","-1","--backend","cuda")
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$results | Format-Table Tag, PrefillTok, PrefillTps, DecodeTok, DecodeTps, MtpAccept, WallSec, TimedOut -AutoSize
Write-Host ""
Write-Host "=== Decode samples (first 12 tokens) ===" -ForegroundColor Cyan
$results | ForEach-Object { "[{0,-14}] {1}" -f $_.Tag, $_.Sample } | ForEach-Object { Write-Host $_ }

$results | ConvertTo-Json | Set-Content "tools\bench\summary.json"
Write-Host ""
Write-Host "Results written to tools\bench\summary.json" -ForegroundColor DarkGray
