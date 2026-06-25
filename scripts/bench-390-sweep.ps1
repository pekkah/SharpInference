# Issue #390: prefill crossover sweep — find the token count where register-mode op-offload
# starts beating the CPU MoE prefill (sets the default-on token gate). NTokens kept small so
# the cell is dominated by prefill, not decode.
param(
    [string]$Model = "E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
    [string]$PromptDir = "C:\Users\pekka\AppData\Local\Temp\claude\C--p-sharpi\e585495f-574a-494e-820c-3cfea3d513e6\scratchpad",
    [string]$Prompts = "p96.txt,p192.txt,p384.txt,p768.txt,p1536.txt,prefill_prompt.txt",
    [string]$Configs = "off,register",
    [int]$NTokens = 4,
    [int]$TimeoutSec = 600
)
$cliDll = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
$dotnet = (Get-Command dotnet).Source
$env:SHARPI_CPU_MOE = "1"
$promptList = $Prompts -split ',' | ForEach-Object { $_.Trim() }
$configList = $Configs -split ',' | ForEach-Object { $_.Trim() }

function Run-One([string]$cfg, [string]$pf) {
    if ($cfg -eq "off") { $env:SHARPI_MOE_GPU_PREFILL = "0"; Remove-Item env:SHARPI_MOE_PIN_MODE -ErrorAction SilentlyContinue }
    else { $env:SHARPI_MOE_GPU_PREFILL = "1"; $env:SHARPI_MOE_PIN_MODE = $cfg }
    $argList = @("$cliDll","-m","$Model","-f","$pf","--temp","0","-n","$NTokens",
                 "--single-turn","-g","-1","--backend","cuda","--no-thinking")
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $dotnet
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $so = $proc.StandardOutput.ReadToEndAsync(); $se = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) { try { $proc.Kill($true) } catch {} }
    $so.Wait(2000)|Out-Null; $se.Wait(2000)|Out-Null
    $stdout = $so.Result; if ($null -eq $stdout) { $stdout = "" }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $pTok=0;$pTps=0.0
    if ($stdout -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $pTok=[int]$matches[1]; $pTps=[double]::Parse($matches[2],$inv) }
    [PSCustomObject]@{ Config=$cfg; Prompt=(Split-Path $pf -Leaf); Tokens=$pTok; PrefillTps=[Math]::Round($pTps,1) }
}

$results = @()
try {
    foreach ($p in $promptList) {
        $pf = Join-Path $PromptDir $p
        if (-not (Test-Path $pf)) { Write-Host "skip missing $pf" -ForegroundColor Yellow; continue }
        foreach ($c in $configList) {
            Write-Host "[$c] $p" -ForegroundColor Cyan
            $results += Run-One $c $pf
        }
    }
} finally {
    Remove-Item env:SHARPI_MOE_GPU_PREFILL -ErrorAction SilentlyContinue
    Remove-Item env:SHARPI_MOE_PIN_MODE    -ErrorAction SilentlyContinue
}
$results | Sort-Object Tokens, Config | Format-Table Tokens, Config, PrefillTps -AutoSize
if (-not (Test-Path "bench-out")) { New-Item -ItemType Directory -Path "bench-out" | Out-Null }
$results | Export-Csv -NoTypeInformation -Path "bench-out\bench-390-sweep.csv"
Write-Host "`nResults -> bench-out\bench-390-sweep.csv" -ForegroundColor Green
