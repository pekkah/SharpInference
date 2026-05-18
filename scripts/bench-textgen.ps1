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
$argList = @("$cliDll", "-m", "$Model", "-p", "$Prompt", "--temp", "0",
             "-n", "$NTokens", "--verbose-prompt", "--single-turn") + $ExtraArgs

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

# First 12 decoded tokens for sanity check
$decTexts = @()
[regex]::Matches($stderr, "next=\d+\('([^']*)'\)") | Select-Object -First 12 | ForEach-Object {
    $decTexts += $_.Groups[1].Value
}
$sample = ($decTexts -join "") -replace "`r","" -replace "`n","\\n"

[PSCustomObject]@{
    Tag         = $Tag
    PrefillTok  = $prefillTok
    PrefillTps  = [Math]::Round($prefillTps, 1)
    DecodeTok   = $totalDecodeTok
    DecodeTps   = [Math]::Round($decodeTps, 1)
    WallSec     = [Math]::Round($elapsed.TotalSeconds, 1)
    TimedOut    = $timedOut
    Sample      = $sample
}
