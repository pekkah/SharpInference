# Issue #390: GPU op-offload MoE prefill — pinned-buffer A/B for the default-on decision.
# Measures prefill (long prompt) AND decode (same run) for three configs:
#   off      — SHARPI_MOE_GPU_PREFILL=0 (CPU MoE prefill; the decode-friendly baseline)
#   copy     — op-offload ON, 14 GB cudaMallocHost copy   (SHARPI_MOE_PIN_MODE=copy)     [#387 behavior]
#   register — op-offload ON, cudaHostRegister mmap in place (SHARPI_MOE_PIN_MODE=register) [#390 fix]
# Goal: register decode within noise of off, while keeping (most of) copy's prefill win.
param(
    [string]$Model = "E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
    [string]$PromptFile = "C:\Users\pekka\AppData\Local\Temp\claude\C--p-sharpi\e585495f-574a-494e-820c-3cfea3d513e6\scratchpad\prefill_prompt.txt",
    [int]$NTokens = 80,
    [int]$TimeoutSec = 900,
    [string]$Configs = "off,copy,register",   # comma-joined so it survives `pwsh -File` arg tokenizing
    [switch]$Warmup
)
$ConfigList = $Configs -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }

if (-not (Test-Path $Model))      { Write-Error "Model not found: $Model"; exit 1 }
if (-not (Test-Path $PromptFile)) { Write-Error "Prompt file not found: $PromptFile"; exit 1 }

$cliDll  = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
$dotnet  = (Get-Command dotnet).Source
$prompt  = Get-Content -Raw $PromptFile
$env:SHARPI_CPU_MOE = "1"

function Run-One([string]$cfg, [string]$tag) {
    if ($cfg -eq "off") {
        $env:SHARPI_MOE_GPU_PREFILL = "0"
        Remove-Item env:SHARPI_MOE_PIN_MODE -ErrorAction SilentlyContinue
    } else {
        $env:SHARPI_MOE_GPU_PREFILL = "1"
        $env:SHARPI_MOE_PIN_MODE = $cfg   # copy | register
    }
    $argList = @("$cliDll","-m","$Model","-f","$PromptFile","--temp","0","-n","$NTokens",
                 "--single-turn","-g","-1","--backend","cuda","--no-thinking")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $dotnet
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $proc = [System.Diagnostics.Process]::Start($psi)
    $so = $proc.StandardOutput.ReadToEndAsync(); $se = $proc.StandardError.ReadToEndAsync()
    $to = $false
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) { try { $proc.Kill($true) } catch {}; $to = $true }
    $sw.Stop(); $so.Wait(2000)|Out-Null; $se.Wait(2000)|Out-Null
    $stdout = $so.Result; if ($null -eq $stdout) { $stdout = "" }
    $stderr = $se.Result; if ($null -eq $stderr) { $stderr = "" }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $pTok=0;$pTps=0.0;$dTok=0;$dTps=0.0;$mtp=$null
    if ($stdout -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $pTok=[int]$matches[1]; $pTps=[double]::Parse($matches[2],$inv) }
    if ($stdout -match 'Decode:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s')  { $dTok=[int]$matches[1]; $dTps=[double]::Parse($matches[2],$inv) }
    if ($stdout -match 'MTP accept:\s+([\d\.]+)\s*%') { $mtp=[double]::Parse($matches[1],$inv) }
    # surface the pin-mode banner the engine prints to stderr
    $pin = ($stderr -split "`n" | Where-Object { $_ -match 'moe-offload' } | Select-Object -First 1)
    [PSCustomObject]@{
        Tag=$tag; Prefill=$pTok; PrefillTps=[Math]::Round($pTps,1)
        Decode=$dTok; DecodeTps=[Math]::Round($dTps,1)
        Mtp=if($null -eq $mtp){$null}else{[Math]::Round($mtp,1)}
        Wall=[Math]::Round($sw.Elapsed.TotalSeconds,1); TimedOut=$to
        Pin=if($pin){$pin.Trim()}else{""}
    }
}

$results = @()
try {
    if ($Warmup) { Write-Host "[warmup] off (page-cache warm)" -ForegroundColor DarkGray; Run-One "off" "warmup" | Out-Null }
    foreach ($c in $ConfigList) {
        Write-Host "[$c] running..." -ForegroundColor Cyan
        $results += Run-One $c $c
    }
} finally {
    Remove-Item env:SHARPI_MOE_GPU_PREFILL -ErrorAction SilentlyContinue
    Remove-Item env:SHARPI_MOE_PIN_MODE    -ErrorAction SilentlyContinue
}
$results | Format-Table Tag,Prefill,PrefillTps,Decode,DecodeTps,Mtp,Wall,TimedOut -AutoSize
$results | ForEach-Object { Write-Host ("  {0,-9} {1}" -f $_.Tag, $_.Pin) -ForegroundColor DarkGray }
if (-not (Test-Path "bench-out")) { New-Item -ItemType Directory -Path "bench-out" | Out-Null }
$results | Export-Csv -NoTypeInformation -Path "bench-out\bench-390-ab.csv"
Write-Host "`nResults -> bench-out\bench-390-ab.csv" -ForegroundColor Green
