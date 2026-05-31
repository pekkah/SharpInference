# Multi-prompt validation for issue #103. Runs Carnice in two configurations:
# baseline (gates=0, legacy FP DotQ3K / DotQ8_0) vs gated (gates=1, the
# int-domain DotQ3K_Q8K / DotQ8_0_Q8K kernels that auto-on would enable).
# For each prompt, captures decode t/s, MTP accept rate, and the first
# DivergeCheckTokens decoded token IDs so we can compute the first-divergence
# position between baseline and gated.
#
# Pass criteria (issue #103):
#   - |MTP accept gated - baseline| <= 2 percentage points
#   - first-divergence position >= 10 (i.e. first ~10 tokens are bit-identical),
#     OR no divergence at all within the captured window
#
# Notes on warmth: Carnice routed-expert weights are mmap'd; the first cell
# pulls them into the OS file cache, all subsequent cells are warm. We run a
# throwaway warmup cell first so even the very first measurement cell is warm.
# This matches feedback_moe_bench_cache_warmth's "always run twice, discard
# the first" rule.
param(
    [string]$Model = "E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
    [int]$NTokens = 60,
    [int]$DivergeCheckTokens = 32,
    [int]$TimeoutSec = 900,
    [string]$OutCsv = "bench-out\bench-q8k-validation.csv"
)

if (-not (Test-Path $Model)) {
    Write-Error "Model not found: $Model"
    exit 1
}

# 5 representative prompts covering categories the issue calls out:
# factual recall (elaborated to avoid Carnice's 1-token EOS), code generation,
# summarisation, math/reasoning, technical explanation. Multi-turn and
# tool-calling are out of scope for the single-prompt CLI surface.
$prompts = @(
    @{ Name = "factual";     Prompt = "Explain in detail how rainbows are formed. Cover light refraction, water droplets, and color dispersion." }
    @{ Name = "codegen";     Prompt = "Write a Python function that computes the SHA-256 hash of a file by reading it in 64KB chunks. Include type hints and a docstring." }
    @{ Name = "summary";     Prompt = "Summarize the key differences between TCP and UDP in three paragraphs, covering reliability, ordering, and typical use cases." }
    @{ Name = "mathreason";  Prompt = "If a train leaves Boston at 3pm traveling 60mph east, and another leaves New York City at 4pm traveling 80mph west, when and where do they meet? Boston-NYC is 215 miles. Show your reasoning step by step." }
    @{ Name = "techexplain"; Prompt = "Explain the CAP theorem and why it's relevant to distributed systems design. Give a concrete example of a system that picks AP over CP." }
)

if (-not (Test-Path "bench-out")) { New-Item -ItemType Directory -Path "bench-out" | Out-Null }

# Parses [DBG] tok=N next=ID(...) lines from a stderr file, returning the
# first $Max token IDs in order. The DecodeLoopMtp path emits one such line
# per accepted token under --verbose-prompt (added with this validation).
function Get-DecodedTokenIds([string]$ErrPath, [int]$Max) {
    if (-not (Test-Path $ErrPath)) { return @() }
    $ids = @()
    foreach ($line in Get-Content $ErrPath) {
        if ($line -match '^\[DBG\]\s+tok=\d+\s+next=(\d+)') {
            $ids += [int]$matches[1]
            if ($ids.Count -ge $Max) { break }
        }
    }
    return ,$ids
}

# Returns the 0-based position of the first differing token between two
# sequences, or -1 if one is a prefix of the other (no divergence in the
# overlapping range).
function Get-FirstDivergence($a, $b) {
    $n = [Math]::Min($a.Count, $b.Count)
    for ($i = 0; $i -lt $n; $i++) {
        if ($a[$i] -ne $b[$i]) { return $i }
    }
    return -1
}

$env:SHARPI_CPU_MOE = "1"
Remove-Item env:SHARPI_Q3K_Q8K  -ErrorAction SilentlyContinue
Remove-Item env:SHARPI_Q8_0_Q8K -ErrorAction SilentlyContinue

try {
    # Warmup: short cell to pull routed-expert mmap pages into OS cache so
    # the first measurement cell isn't cold. Gates default off here.
    Write-Host "[warmup] priming OS file cache..." -ForegroundColor DarkCyan
    $env:SHARPI_Q3K_Q8K  = "0"
    $env:SHARPI_Q8_0_Q8K = "0"
    [void] (.\scripts\bench-textgen.ps1 -Model $Model -Tag "q8k-warmup" `
        -NTokens 10 -Prompt "Hello." -TimeoutSec $TimeoutSec `
        -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking"))

    $results = @()
    foreach ($p in $prompts) {
        # Baseline: gates explicitly off
        $env:SHARPI_Q3K_Q8K  = "0"
        $env:SHARPI_Q8_0_Q8K = "0"
        $base = .\scripts\bench-textgen.ps1 -Model $Model -Tag "q8k-$($p.Name)-baseline" `
            -NTokens $NTokens -Prompt $p.Prompt -TimeoutSec $TimeoutSec `
            -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
        $baseTokens = Get-DecodedTokenIds "tools\bench\q8k-$($p.Name)-baseline.err" $DivergeCheckTokens

        # Gated: both Q8K-input kernels on (the future auto-on default for Carnice)
        $env:SHARPI_Q3K_Q8K  = "1"
        $env:SHARPI_Q8_0_Q8K = "1"
        $gated = .\scripts\bench-textgen.ps1 -Model $Model -Tag "q8k-$($p.Name)-gated" `
            -NTokens $NTokens -Prompt $p.Prompt -TimeoutSec $TimeoutSec `
            -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
        $gatedTokens = Get-DecodedTokenIds "tools\bench\q8k-$($p.Name)-gated.err" $DivergeCheckTokens

        $divPos = Get-FirstDivergence $baseTokens $gatedTokens

        $mtpDelta = $null
        if ($null -ne $base.MtpAccept -and $null -ne $gated.MtpAccept) {
            $mtpDelta = [Math]::Round($gated.MtpAccept - $base.MtpAccept, 1)
        }

        # Pass when (a) MTP accept delta within ±2pp AND (b) divergence past
        # position 9 (i.e. first 10 tokens stable), or no divergence at all.
        $mtpOk = ($null -eq $mtpDelta) -or ([Math]::Abs($mtpDelta) -le 2.0)
        $argmaxOk = ($divPos -eq -1) -or ($divPos -ge 10)
        $pass = $mtpOk -and $argmaxOk

        $results += [PSCustomObject]@{
            Prompt           = $p.Name
            BaselineDecodeTps= $base.DecodeTps
            GatedDecodeTps   = $gated.DecodeTps
            DecodeTpsLift    = if ($base.DecodeTps -gt 0) { [Math]::Round(($gated.DecodeTps - $base.DecodeTps) / $base.DecodeTps * 100, 1) } else { $null }
            BaselineMtp      = $base.MtpAccept
            GatedMtp         = $gated.MtpAccept
            MtpDeltaPp       = $mtpDelta
            FirstDivergeAt   = $divPos
            BaselineTokens   = ($baseTokens  -join ",")
            GatedTokens      = ($gatedTokens -join ",")
            MtpOk            = $mtpOk
            ArgmaxOk         = $argmaxOk
            Pass             = $pass
        }
    }
}
finally {
    Remove-Item env:SHARPI_Q3K_Q8K  -ErrorAction SilentlyContinue
    Remove-Item env:SHARPI_Q8_0_Q8K -ErrorAction SilentlyContinue
}

$results | Format-Table Prompt, BaselineDecodeTps, GatedDecodeTps, DecodeTpsLift, BaselineMtp, GatedMtp, MtpDeltaPp, FirstDivergeAt, MtpOk, ArgmaxOk, Pass
$results | Export-Csv -NoTypeInformation -Path $OutCsv

$allPass = ($results | Where-Object { -not $_.Pass } | Measure-Object).Count -eq 0
Write-Host ""
if ($allPass) {
    Write-Host "All $($results.Count) prompts PASS the issue #103 criteria." -ForegroundColor Green
} else {
    $failed = ($results | Where-Object { -not $_.Pass } | Select-Object -ExpandProperty Prompt) -join ", "
    Write-Host "FAIL: $failed did not meet the issue #103 criteria." -ForegroundColor Red
}
Write-Host "Results written to $OutCsv" -ForegroundColor DarkGray
