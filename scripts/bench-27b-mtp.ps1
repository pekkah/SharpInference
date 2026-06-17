# One-shot bench harness for Qwen3.6-27B-MTP (issue #28). Runs both MTP-on (default)
# and MTP-off (SHARPI_DISABLE_MTP=1) on CPU and CUDA-hybrid for one or more quants,
# so the README can quote both numbers and the no-speedup gap is visible.
#
# Issue #209: the MTP-on rows now exercise the k=4 verify batch (the new default:
# SHARPI_MTP_BATCH_MAX=4 / SHARPI_MTP_DRAFT_N=3) — the 4-input CPU-FFN MatVec4In
# amortizes the dominant CPU mmap weight read four ways, moving the optimum out from
# the old pairwise k=2. Measured Q4_K_M CUDA-hybrid: k=4 12.3 vs k=2 10.1 vs k=6 10.4
# vs MTP-off 6.5 t/s. Set SHARPI_MTP_BATCH_MAX / SHARPI_MTP_DRAFT_N to sweep other k.
param(
    [string[]]$Quants = @("Q4_K_M", "Q5_K_M"),
    [int]$NTokens = 80,
    [string]$Prompt = "Write a Python function that sorts a list using the quicksort algorithm:"
)

$results = @()
foreach ($q in $Quants) {
    $model = "E:\models\Qwen3.6-27B-MTP-$q.gguf"
    if (-not (Test-Path $model)) {
        Write-Host "[skip] $model not present"
        continue
    }
    $tagQ = $q.ToLowerInvariant().Replace("_", "")  # q4km / q5km

    # MTP-on (auto)
    $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-27b-mtp-$tagQ-cpu-mtp"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
    $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-27b-mtp-$tagQ-cuda-hybrid-mtp" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")

    # MTP-off (SHARPI_DISABLE_MTP=1)
    $env:SHARPI_DISABLE_MTP = "1"
    try {
        $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-27b-mtp-$tagQ-cpu-nomtp"         -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("--no-thinking")
        $results += .\scripts\bench-textgen.ps1 -Model $model -Tag "qwen36-27b-mtp-$tagQ-cuda-hybrid-nomtp" -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
    }
    finally {
        Remove-Item Env:\SHARPI_DISABLE_MTP -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
$results | Format-Table Tag, PrefillTok, PrefillTps, DecodeTok, DecodeTps, MtpAccept, WallSec, TimedOut -AutoSize
