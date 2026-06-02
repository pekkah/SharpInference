# Smoke test: launch SharpInference.Server.Host with Gemma 4 E4B Q8 and hit
# both OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) endpoints.
# Reports OK / FAIL per endpoint and prints the first ~80 chars of each
# response so a coherent reply is visually verifiable.
#
# Skips silently when the GGUF isn't present (matches the test-suite convention).
param(
    [string]$ModelPath = "E:\models\gemma-4-E4B-it-Q8_0.gguf",
    [int]$Port = 8181,
    [int]$StartupTimeoutSec = 120
)

if (-not (Test-Path $ModelPath)) {
    Write-Host "[smoke-gemma4-server] Model not found at $ModelPath — skipping." -ForegroundColor Yellow
    exit 0
}

$env:SHARPI_MODEL = $ModelPath
$env:ASPNETCORE_URLS = "http://127.0.0.1:$Port"

# Force CPU backend via SHARPI_BACKEND env var. The default hybrid-CUDA path
# can't fit Gemma 4 E4B Q8 in 12 GB VRAM and doesn't yet implement the
# per-layer head_dim trunk, so it produces `<pad>` tokens.
$env:SHARPI_BACKEND = "Cpu"
$env:SHARPI_N_GPU_LAYERS = "0"

Write-Host "[smoke-gemma4-server] Launching server (port $Port, CPU)..." -ForegroundColor DarkGray
$proc = Start-Process -FilePath "dotnet" `
    -ArgumentList @("run", "--project", "src/SharpInference.Server.Host", "-c", "Release") `
    -PassThru -NoNewWindow -RedirectStandardOutput "tools/smoke-gemma4-server.out" `
                          -RedirectStandardError  "tools/smoke-gemma4-server.err"

try {
    # Wait until /health responds (or timeout).
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $ready = $false
    while ($sw.Elapsed.TotalSeconds -lt $StartupTimeoutSec) {
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction Stop
            if ($h.status -eq "ok") { $ready = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 1000
    }
    if (-not $ready) {
        Write-Host "[smoke-gemma4-server] /health never reported ok within ${StartupTimeoutSec}s." -ForegroundColor Red
        exit 2
    }
    Write-Host "[smoke-gemma4-server] Server up ($([int]$sw.Elapsed.TotalSeconds)s)." -ForegroundColor Green

    # ── OpenAI /v1/chat/completions ────────────────────────────────────────
    $openaiBody = @{
        model = "gemma4"
        messages = @(@{ role = "user"; content = "What is the capital of France? Reply with one word." })
        max_tokens = 40
        temperature = 0
        stream = $false
    } | ConvertTo-Json -Depth 6

    $openaiResp = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" `
        -Method Post -ContentType "application/json" -Body $openaiBody -TimeoutSec 600
    $openaiText = $openaiResp.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($openaiText)) {
        Write-Host "[smoke-gemma4-server] OpenAI: FAIL — empty content" -ForegroundColor Red
        $rc = 3
    } elseif ($openaiText -match '^(<pad>|<unk>|<eos>)+\s*$') {
        Write-Host "[smoke-gemma4-server] OpenAI: FAIL — garbage tokens '$($openaiText.Substring(0, [Math]::Min(60,$openaiText.Length)))'" -ForegroundColor Red
        $rc = 3
    } else {
        $head = $openaiText.Substring(0, [Math]::Min(80, $openaiText.Length)).Replace("`n", " ")
        Write-Host "[smoke-gemma4-server] OpenAI : OK — '$head'" -ForegroundColor Green
        $rc = 0
    }

    # ── Anthropic /v1/messages ─────────────────────────────────────────────
    $anthBody = @{
        model = "gemma4"
        messages = @(@{ role = "user"; content = "What is the capital of France? Reply with one word." })
        max_tokens = 40
        temperature = 0
        stream = $false
    } | ConvertTo-Json -Depth 6

    $anthResp = Invoke-RestMethod "http://127.0.0.1:$Port/v1/messages" `
        -Method Post -ContentType "application/json" -Body $anthBody -TimeoutSec 600
    $anthText = $anthResp.content[0].text
    if ([string]::IsNullOrWhiteSpace($anthText)) {
        Write-Host "[smoke-gemma4-server] Anthropic: FAIL — empty content" -ForegroundColor Red
        if ($rc -eq 0) { $rc = 4 }
    } elseif ($anthText -match '^(<pad>|<unk>|<eos>)+\s*$') {
        Write-Host "[smoke-gemma4-server] Anthropic: FAIL — garbage tokens '$($anthText.Substring(0, [Math]::Min(60,$anthText.Length)))'" -ForegroundColor Red
        if ($rc -eq 0) { $rc = 4 }
    } else {
        $head = $anthText.Substring(0, [Math]::Min(80, $anthText.Length)).Replace("`n", " ")
        Write-Host "[smoke-gemma4-server] Anthropic: OK — '$head'" -ForegroundColor Green
    }

    exit $rc
}
finally {
    if ($proc -and -not $proc.HasExited) {
        try { $proc.Kill($true) } catch {}
    }
}
