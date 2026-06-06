param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][string]$Tag,
    [string[]]$ExtraArgs = @(),
    [string]$Prompt = "The capital of France is",
    [int]$NTokens = 60,
    [string]$OutDir = "tools\bench",
    [int]$TimeoutSec = 300
)

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$stdoutPath = Join-Path $OutDir "$Tag.out"
$stderrPath = Join-Path $OutDir "$Tag.err"

$cliDll = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
# NOTE: --verbose-prompt is deliberately NOT passed. Its per-token debug logging does a
# full-vocabulary LINQ OrderByDescending (262144 elements for Gemma 4) every decode step,
# which adds ~1.5-2.5 ms/token of CPU overhead and badly understates the measured decode
# t/s. Benchmarks must run without it to reflect real generation throughput.
$argList = @("$cliDll", "-m", "$Model", "-p", "$Prompt", "--temp", "0",
             "-n", "$NTokens", "--single-turn") + $ExtraArgs

Write-Host "[$Tag] $($ExtraArgs -join ' ') (timeout ${TimeoutSec}s)" -ForegroundColor DarkGray
$dotnetExe = (Get-Command dotnet).Source
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $dotnetExe
$psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)
# Async-read both streams so child output buffers don't block
$stdoutTask = $proc.StandardOutput.ReadToEndAsync()
$stderrTask = $proc.StandardError.ReadToEndAsync()
$timedOut = $false
if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
    Write-Host "[$Tag] TIMEOUT after ${TimeoutSec}s — killing PID $($proc.Id)" -ForegroundColor Red
    try { $proc.Kill($true) } catch {}
    $timedOut = $true
}
$sw.Stop()
$elapsed = $sw.Elapsed
$stdoutTask.Wait(2000) | Out-Null
$stderrTask.Wait(2000) | Out-Null
Set-Content -Path $stdoutPath -Value $stdoutTask.Result
Set-Content -Path $stderrPath -Value $stderrTask.Result

$stdout = $stdoutTask.Result; if ($null -eq $stdout) { $stdout = "" }
$stderr = $stderrTask.Result; if ($null -eq $stderr) { $stderr = "" }
$inv = [System.Globalization.CultureInfo]::InvariantCulture

function Parse-Double([string]$s) {
    return [double]::Parse($s, $inv)
}

# Prefill metrics from stdout line: "Prefill: X tokens, Y t/s | Decode: A tokens, B t/s"
$prefillTok = 0; $prefillTps = 0.0
if ($stdout -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') {
    $prefillTok = [int]$matches[1]; $prefillTps = Parse-Double $matches[2]
}
# Decode metrics from stdout. CLI now reports total forward iterations including
# thinking-mode tokens, so this counter is correct for all models.
$totalDecodeTok = 0; $decodeTps = 0.0
if ($stdout -match 'Decode:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') {
    $totalDecodeTok = [int]$matches[1]; $decodeTps = Parse-Double $matches[2]
}

# MTP acceptance rate (when MTP engages). Captured as a percentage so the bench
# summary can show the draft-acceptance gap without extra plumbing.
$mtpAccept = $null
if ($stdout -match 'MTP accept:\s+([\d\.]+)\s*%') {
    $mtpAccept = Parse-Double $matches[1]
}

# Generation sample for a sanity check. Without --verbose-prompt there are no per-token
# debug lines, so take the decoded text from stdout: strip the prompt echo and the
# trailing "Prefill: ... | Decode: ..." stats line, then keep a short snippet.
$genText = $stdout -replace "`r",""
$genText = ($genText -split "Prefill:")[0]
if ($genText.StartsWith($Prompt)) { $genText = $genText.Substring($Prompt.Length) }
$sample = ($genText -replace "`n","\\n").Trim()
if ($sample.Length -gt 80) { $sample = $sample.Substring(0, 80) }

[PSCustomObject]@{
    Tag         = $Tag
    PrefillTok  = $prefillTok
    PrefillTps  = [Math]::Round($prefillTps, 1)
    DecodeTok   = $totalDecodeTok
    DecodeTps   = [Math]::Round($decodeTps, 1)
    MtpAccept   = if ($null -eq $mtpAccept) { $null } else { [Math]::Round($mtpAccept, 1) }
    WallSec     = [Math]::Round($elapsed.TotalSeconds, 1)
    TimedOut    = $timedOut
    Sample      = $sample
}
