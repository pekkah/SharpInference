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

# Qwen3 8B — dense, headDim 128, supports TQ, 3 backends
$qwen = "models\Qwen3-8B-Q4_K_M.gguf"
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-vulkan"  -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","vulkan")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-vk-tq"   -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","vulkan","--tq")
$results += .\scripts\bench-textgen.ps1 -Model $qwen -Tag "qwen3-cuda"    -NTokens $NTokens -Prompt $Prompt -TimeoutSec 240 -ExtraArgs @("-g","-1","--backend","cuda")

# Qwen3-Coder 30B-A3B MoE — CPU only (MoE GPU is gated by issue #2)
$moe = "models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-cpu"        -NTokens $NTokens -Prompt $Prompt -TimeoutSec 360
$results += .\scripts\bench-textgen.ps1 -Model $moe -Tag "moe-cpu-tq"     -NTokens $NTokens -Prompt $Prompt -TimeoutSec 360 -ExtraArgs @("--tq")

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$results | Format-Table Tag, PrefillTok, PrefillTps, DecodeTok, DecodeTps, WallSec, TimedOut -AutoSize
Write-Host ""
Write-Host "=== Decode samples (first 12 tokens) ===" -ForegroundColor Cyan
$results | ForEach-Object { "[{0,-14}] {1}" -f $_.Tag, $_.Sample } | ForEach-Object { Write-Host $_ }

$results | ConvertTo-Json | Set-Content "tools\bench\summary.json"
Write-Host ""
Write-Host "Results written to tools\bench\summary.json" -ForegroundColor DarkGray
