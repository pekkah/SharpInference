# Issue #210 A/B: grouped-by-expert MoE verify batching for MTP on the CUDA-hybrid
# Qwen3.6-35B-A3B-MTP path.
#
# Cells (all MTP-on except baseline):
#   baseline  — SHARPI_DISABLE_MTP=1                     (MTP off; acceptance denominator)
#   pertoken  — MTP on, SHARPI_MTP_BATCHED_MOE_VERIFY=0  (old per-token verify FFN)
#   batched   — MTP on, default                          (issue #210 grouped-by-expert verify)
#
# The cells are INTERLEAVED across $Reps rounds with a cooldown between runs so GPU
# thermal/clock drift is shared evenly (sequential blocks let the last cell run hot
# and throttled). The reported number per cell is the MEDIAN warm decode t/s.
# Acceptance (#210): median(batched) / median(baseline) > 1.15.
param(
    [int]$NTokens = 80,
    [int]$Reps = 3,
    [int]$CooldownSec = 20,
    [string]$Prompt = "Write a Python function that sorts a list using the quicksort algorithm:"
)

$model = "E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"
if (-not (Test-Path $model)) { Write-Host "[skip] $model not present"; return }

$base = @("-g","-1","--backend","cuda","--no-thinking")
$env:SHARPI_CPU_MOE = "1"

function RunCell([string]$tag, [hashtable]$envSet) {
    foreach ($k in $envSet.Keys) { Set-Item -Path "Env:\$k" -Value $envSet[$k] }
    try {
        $r = .\scripts\bench-textgen.ps1 -Model $model -Tag $tag -NTokens $NTokens -Prompt $Prompt -TimeoutSec 900 -ExtraArgs $base
    } finally {
        foreach ($k in $envSet.Keys) { Remove-Item -Path "Env:\$k" -ErrorAction SilentlyContinue }
    }
    Start-Sleep -Seconds $CooldownSec   # let GPU boost clocks recover before the next cell
    return $r
}

$cells = [ordered]@{
    baseline = @{ SHARPI_DISABLE_MTP = "1" }
    pertoken = @{ SHARPI_MTP_BATCHED_MOE_VERIFY = "0" }
    batched  = @{}
}
$dec = @{ baseline = @(); pertoken = @(); batched = @() }
$acc = @{ baseline = @(); pertoken = @(); batched = @() }

try {
    # Warm the OS page cache + GPU context once (discarded).
    $null = .\scripts\bench-textgen.ps1 -Model $model -Tag "210-warmup" -NTokens 16 -Prompt $Prompt -TimeoutSec 900 -ExtraArgs $base
    Start-Sleep -Seconds $CooldownSec

    for ($rep = 1; $rep -le $Reps; $rep++) {
        foreach ($name in $cells.Keys) {
            $r = RunCell "210-$name-r$rep" $cells[$name]
            $dec[$name] += $r.DecodeTps
            if ($null -ne $r.MtpAccept) { $acc[$name] += $r.MtpAccept }
            Write-Host ("  rep{0} {1,-9} decode={2,6:F1} t/s" -f $rep, $name, $r.DecodeTps) -ForegroundColor DarkGray
        }
    }
}
finally {
    Remove-Item Env:\SHARPI_CPU_MOE -ErrorAction SilentlyContinue
}

function Median([double[]]$xs) {
    if ($xs.Count -eq 0) { return 0.0 }
    $s = $xs | Sort-Object
    $n = $s.Count
    if ($n % 2 -eq 1) { return [double]$s[[int](($n-1)/2)] }
    return ([double]$s[$n/2 - 1] + [double]$s[$n/2]) / 2.0
}

$mBase = Median ([double[]]$dec.baseline)
$mPer  = Median ([double[]]$dec.pertoken)
$mBat  = Median ([double[]]$dec.batched)

Write-Host ""
Write-Host "=== #210 Summary (median warm decode over $Reps reps, interleaved) ===" -ForegroundColor Cyan
Write-Host ("baseline (MTP off)       : {0,6:F1} t/s   reps=[{1}]" -f $mBase, ($dec.baseline -join ', '))
Write-Host ("MTP per-token verify     : {0,6:F1} t/s   reps=[{1}]" -f $mPer,  ($dec.pertoken -join ', '))
Write-Host ("MTP batched verify (#210): {0,6:F1} t/s   reps=[{1}]" -f $mBat,  ($dec.batched  -join ', ')) -ForegroundColor Green
if ($mBase -gt 0) {
    Write-Host ("batched / baseline       : {0,5:F3}x" -f ($mBat / $mBase)) -ForegroundColor Green
}
if ($mPer -gt 0) {
    Write-Host ("batched / per-token      : {0,5:F3}x" -f ($mBat / $mPer)) -ForegroundColor Green
}
$pass = ($mBase -gt 0) -and (($mBat / $mBase) -gt 1.15)
Write-Host ("Acceptance (>1.15x baseline): {0}" -f ($(if ($pass) { "PASS" } else { "FAIL" }))) -ForegroundColor $(if ($pass) { "Green" } else { "Red" })
