param([int[]]$Sizes = @(200,500,1000,2000,4000))

$url = "http://127.0.0.1:5000/v1/messages"

function Invoke-Prefill([int]$n) {
    $prompt = ("data " * $n).Trim()
    $body = @{
        model = "carnice"
        max_tokens = 1
        temperature = 0
        stream = $false
        messages = @(@{ role = "user"; content = $prompt })
    } | ConvertTo-Json -Depth 6
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $resp = Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1200
    $sw.Stop()
    $inTok = $resp.usage.input_tokens
    $wall = $sw.Elapsed.TotalSeconds
    [pscustomobject]@{ Words=$n; InputTokens=$inTok; WallSec=[math]::Round($wall,2); PrefillTokPerSec=[math]::Round($inTok/$wall,1) }
}

# Warmup (also triggers lazy model load)
Write-Host "Warmup..." -ForegroundColor Yellow
$null = Invoke-Prefill 50

$results = foreach ($s in $Sizes) {
    $r = Invoke-Prefill $s
    Write-Host ("{0,5} words -> {1,5} tok, {2,7}s, {3,7} tok/s" -f $r.Words,$r.InputTokens,$r.WallSec,$r.PrefillTokPerSec)
    $r
}
$results | Format-Table -AutoSize
