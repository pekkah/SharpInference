# Issue #106 end-to-end verification: chat-completion snapshot reuse on MTP runs.
#
# Drives a 2-turn /v1/chat/completions conversation, then reads /metrics to
# confirm sharpi_prefill_tokens_reused_total > 0 — the metric that stayed at 0
# pre-#106 and is the user-visible proof that the snapshot path now fires on
# MTP chat continuations.
param(
    [string]$ServerUrl = "http://localhost:5000",
    [int]$TimeoutSec = 600,
    [int]$MaxTokens = 32
)

function Get-ReusedTokens {
    param([string]$Url)
    $metrics = Invoke-RestMethod -Uri "$Url/metrics" -TimeoutSec 10
    foreach ($line in $metrics -split "`n") {
        if ($line -match '^sharpi_prefill_tokens_reused_total\s+(\d+)') {
            return [long]$matches[1]
        }
    }
    return -1
}

function Send-Chat {
    param([string]$Url, $Messages, [int]$MaxTokens)
    $body = @{
        model = "any"
        messages = $Messages
        max_tokens = $MaxTokens
        temperature = 0.0
        enable_thinking = $false
    } | ConvertTo-Json -Depth 4 -Compress
    return Invoke-RestMethod -Uri "$Url/v1/chat/completions" -Method Post `
        -Body $body -ContentType "application/json" -TimeoutSec $TimeoutSec
}

Write-Host "[1/4] Probing $ServerUrl/metrics for baseline counter..."
$baseline = Get-ReusedTokens -Url $ServerUrl
Write-Host "      baseline sharpi_prefill_tokens_reused_total = $baseline"
if ($baseline -lt 0) {
    Write-Error "Could not read prefill-reused metric from $ServerUrl/metrics"
    exit 1
}

$turn1Messages = @(
    @{ role = "system"; content = "You are a concise technical assistant." },
    @{ role = "user";   content = "What is the time complexity of merge sort? Answer in one sentence." }
)

Write-Host "[2/4] Sending turn 1 (cold start)..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp1 = Send-Chat -Url $ServerUrl -Messages $turn1Messages -MaxTokens $MaxTokens
$sw.Stop()
$assistant1 = $resp1.choices[0].message.content
Write-Host "      turn1 wall: $($sw.Elapsed.TotalSeconds.ToString('F2'))s"
Write-Host "      turn1 assistant: $assistant1"

$afterTurn1 = Get-ReusedTokens -Url $ServerUrl
Write-Host "      after turn1: reused = $afterTurn1 (delta $($afterTurn1 - $baseline))"

# Turn 2 reuses turn 1's history + adds a follow-up. The snapshot taken at
# turn 1's canonical-history boundary must let the engine skip re-prefilling
# the first user message.
$turn2Messages = $turn1Messages + @(
    @{ role = "assistant"; content = $assistant1 },
    @{ role = "user";      content = "And the worst-case space complexity?" }
)

Write-Host "[3/4] Sending turn 2 (snapshot reuse path)..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp2 = Send-Chat -Url $ServerUrl -Messages $turn2Messages -MaxTokens $MaxTokens
$sw.Stop()
Write-Host "      turn2 wall: $($sw.Elapsed.TotalSeconds.ToString('F2'))s"
Write-Host "      turn2 assistant: $($resp2.choices[0].message.content)"

$afterTurn2 = Get-ReusedTokens -Url $ServerUrl
Write-Host "      after turn2: reused = $afterTurn2 (delta $($afterTurn2 - $afterTurn1))"

Write-Host "[4/4] Verdict"
$delta = $afterTurn2 - $afterTurn1
if ($delta -gt 0) {
    Write-Host "      PASS — sharpi_prefill_tokens_reused_total advanced by $delta on turn 2." -ForegroundColor Green
    Write-Host "             Issue #106 fix is live on this server: MTP chat continuations now reuse the snapshot."
    exit 0
} else {
    Write-Host "      FAIL — counter did not advance. Snapshot reuse did NOT fire." -ForegroundColor Red
    Write-Host "             Suspect: useMtp gating, _prevTokens drift, or canonical-prefix mismatch." -ForegroundColor Red
    exit 2
}
