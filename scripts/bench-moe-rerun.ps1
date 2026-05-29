param(
    [int]$NTokens = 80,
    [int]$Repeats = 3,
    [string]$Prompt = "Write a Python function that sorts a list using the quicksort algorithm:",
    [string]$OutJson = "tools\bench\moe-rerun.json"
)

# Same model paths as scripts/bench-all.ps1
$moe = "models\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"

$cases = @(
    @{ Tag = "moe-cpu";           Extra = @();                                   Timeout = 360 },
    @{ Tag = "moe-cpu-tq";        Extra = @("--tq");                             Timeout = 360 },
    @{ Tag = "moe-vulkan-hybrid"; Extra = @("-g","-1","--backend","vulkan");     Timeout = 600 },
    @{ Tag = "moe-cuda-hybrid";   Extra = @("-g","-1","--backend","cuda");       Timeout = 600 }
)

$all = @()
foreach ($case in $cases) {
    for ($i = 1; $i -le $Repeats; $i++) {
        $runTag = "$($case.Tag)-r$i"
        Write-Host ""
        Write-Host "=== $runTag ($($i)/$Repeats) ===" -ForegroundColor Yellow
        $r = .\scripts\bench-textgen.ps1 -Model $moe -Tag $runTag -NTokens $NTokens -Prompt $Prompt `
                -TimeoutSec $case.Timeout -ExtraArgs $case.Extra
        $r | Add-Member -NotePropertyName Group -NotePropertyValue $case.Tag -Force
        $r | Add-Member -NotePropertyName Run   -NotePropertyValue $i        -Force
        $all += $r
        Write-Host ("  prefill={0} t/s  decode={1} t/s  wall={2}s" -f $r.PrefillTps, $r.DecodeTps, $r.WallSec)
    }
}

# Aggregate per (group, metric)
function Stats($values) {
    if ($values.Count -eq 0) { return @{ Mean = 0.0; Std = 0.0 } }
    $mean = ($values | Measure-Object -Average).Average
    if ($values.Count -lt 2) { return @{ Mean = [Math]::Round($mean,2); Std = 0.0 } }
    $sumsq = 0.0
    foreach ($v in $values) { $sumsq += ($v - $mean) * ($v - $mean) }
    $std = [Math]::Sqrt($sumsq / ($values.Count - 1))
    return @{ Mean = [Math]::Round($mean,2); Std = [Math]::Round($std,3) }
}

$baselines = @{
    "moe-cpu"           = @{ Prefill = 14.9; Decode = 21.4 }
    "moe-cpu-tq"        = @{ Prefill = 12.4; Decode = 21.4 }
    "moe-vulkan-hybrid" = @{ Prefill = 1.0;  Decode = 5.5  }
    "moe-cuda-hybrid"   = @{ Prefill = 16.1; Decode = 22.4 }
}

$summary = @()
foreach ($case in $cases) {
    $rows = $all | Where-Object { $_.Group -eq $case.Tag }
    $pf = Stats ($rows | ForEach-Object { [double]$_.PrefillTps })
    $dc = Stats ($rows | ForEach-Object { [double]$_.DecodeTps })
    $bp = $baselines[$case.Tag].Prefill
    $bd = $baselines[$case.Tag].Decode
    # Z-scores: (mean - baseline) / std.  Negative => below baseline.
    $zp = if ($pf.Std -gt 0) { [Math]::Round(($pf.Mean - $bp) / $pf.Std, 2) } else { 0.0 }
    $zd = if ($dc.Std -gt 0) { [Math]::Round(($dc.Mean - $bd) / $dc.Std, 2) } else { 0.0 }
    $summary += [PSCustomObject]@{
        Tag           = $case.Tag
        BasePf        = $bp
        BaseDc        = $bd
        PfMean        = $pf.Mean
        PfStd         = $pf.Std
        PfZ           = $zp
        PfDeltaPct    = [Math]::Round(100.0 * ($pf.Mean - $bp) / $bp, 1)
        DcMean        = $dc.Mean
        DcStd         = $dc.Std
        DcZ           = $zd
        DcDeltaPct    = [Math]::Round(100.0 * ($dc.Mean - $bd) / $bd, 1)
    }
}

Write-Host ""
Write-Host "=== Per-run rates ===" -ForegroundColor Cyan
$all | Format-Table Group, Run, PrefillTps, DecodeTps, WallSec, TimedOut -AutoSize

Write-Host ""
Write-Host "=== Mean ± stddev vs baseline ===" -ForegroundColor Cyan
$summary | Format-Table Tag, BasePf, PfMean, PfStd, PfZ, PfDeltaPct, BaseDc, DcMean, DcStd, DcZ, DcDeltaPct -AutoSize

$payload = [PSCustomObject]@{
    Runs    = $all
    Summary = $summary
}
$payload | ConvertTo-Json -Depth 5 | Set-Content $OutJson
Write-Host ""
Write-Host "Results written to $OutJson" -ForegroundColor DarkGray
