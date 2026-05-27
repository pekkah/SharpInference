# Bench harness for Qwen3.6-35B-A3B-MTP (closes-#44 follow-up to issue #28).
# MoE MTP head requires SHARPI_CPU_MOE=1 for CUDA-hybrid (the SLRU expert cache
# isn't sized for the extra MTP block). Mirrors bench-27b-mtp.ps1's matrix.
param(
    [int]$NTokens = 80,
    [string]$Prompt = "Write a Python function that sorts a list using the quicksort algorithm:"
)

$model = "E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
if (-not (Test-Path $model)) {
    Write-Host "[skip] $model not present"
    return
}

$env:SHARPI_CPU_MOE = "1"
$results = @()
try {
    # MTP-on (auto)
    $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-35b-a3b-mtp-q4km-cpu-mtp"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
    $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-35b-a3b-mtp-q4km-cuda-hybrid-mtp" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")

    # MTP-off (SHARPI_DISABLE_MTP=1)
    $env:SHARPI_DISABLE_MTP = "1"
    try {
        $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-35b-a3b-mtp-q4km-cpu-nomtp"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
        $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-35b-a3b-mtp-q4km-cuda-hybrid-nomtp" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
    }
    finally {
        Remove-Item Env:\SHARPI_DISABLE_MTP -ErrorAction SilentlyContinue
    }
}
finally {
    Remove-Item Env:\SHARPI_CPU_MOE -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$results | Format-Table Tag, PrefillTok, PrefillTps, DecodeTok, DecodeTps, MtpAccept, WallSec, TimedOut -AutoSize
