# Issue #390: re-measure the three CUDA GDN-hybrid CPU-MoE README rows with op-offload now
# DEFAULT-ON (register pin mode + token gate) vs forced-off (the prior CPU-path baseline).
# 2K-token working-context prompt (matches the README "Prefill t/s" definition). Same session
# so the off/on delta is hardware/thermal-consistent.
param(
    [string]$PromptFile = "C:\Users\pekka\AppData\Local\Temp\claude\C--p-sharpi\e585495f-574a-494e-820c-3cfea3d513e6\scratchpad\prefill_prompt.txt",
    [int]$NTokens = 60,
    [int]$TimeoutSec = 900
)
$cliDll = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
$dotnet = (Get-Command dotnet).Source
$env:SHARPI_CPU_MOE = "1"
$models = @(
    @{ Name="Carnice";        Path="E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf" }
    @{ Name="35B-A3B";        Path="E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" }
    @{ Name="35B-A3B-MTP";    Path="E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf" }
)

function Run-One([string]$name, [string]$model, [string]$mode) {
    if ($mode -eq "off") { $offArg = @("--gpu-moe-prefill","false") } else { $offArg = @() }   # on = rely on new default
    $argList = @("$cliDll","-m","$model","-f","$PromptFile","--temp","0","-n","$NTokens",
                 "--single-turn","-g","-1","--backend","cuda","--no-thinking") + $offArg
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $dotnet
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $so = $proc.StandardOutput.ReadToEndAsync(); $se = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) { try { $proc.Kill($true) } catch {} }
    $so.Wait(2000)|Out-Null; $se.Wait(2000)|Out-Null
    $stdout = $so.Result; if ($null -eq $stdout) { $stdout = "" }
    $stderr = $se.Result; if ($null -eq $stderr) { $stderr = "" }
    $inv = [System.Globalization.CultureInfo]::InvariantCulture
    $pTok=0;$pTps=0.0;$dTok=0;$dTps=0.0;$mtp=$null
    if ($stdout -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $pTok=[int]$matches[1]; $pTps=[double]::Parse($matches[2],$inv) }
    if ($stdout -match 'Decode:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s')  { $dTok=[int]$matches[1]; $dTps=[double]::Parse($matches[2],$inv) }
    if ($stdout -match 'MTP accept:\s+([\d\.]+)\s*%') { $mtp=[double]::Parse($matches[1],$inv) }
    $pin = ($stderr -split "`n" | Where-Object { $_ -match 'moe-offload' } | Select-Object -First 1)
    [PSCustomObject]@{
        Model=$name; Mode=$mode; Prefill=$pTok; PrefillTps=[Math]::Round($pTps,1)
        DecodeTps=[Math]::Round($dTps,1); Mtp=if($null -eq $mtp){$null}else{[Math]::Round($mtp,1)}
        Pin=if($pin){($pin.Trim() -replace '\[moe-offload\] ','')}else{"(cpu)"}
    }
}

$results = @()
try {
    foreach ($m in $models) {
        if (-not (Test-Path $m.Path)) { Write-Host "skip missing $($m.Path)" -ForegroundColor Yellow; continue }
        foreach ($mode in @("off","on")) {
            Write-Host "[$($m.Name)/$mode]..." -ForegroundColor Cyan
            $results += Run-One $m.Name $m.Path $mode
        }
    }
} finally {
    Remove-Item env:SHARPI_MOE_GPU_PREFILL -ErrorAction SilentlyContinue
    Remove-Item env:SHARPI_MOE_PIN_MODE    -ErrorAction SilentlyContinue
}
$results | Format-Table Model, Mode, Prefill, PrefillTps, DecodeTps, Mtp -AutoSize
$results | ForEach-Object { Write-Host ("  {0,-13} {1,-3} {2}" -f $_.Model, $_.Mode, $_.Pin) -ForegroundColor DarkGray }
if (-not (Test-Path "bench-out")) { New-Item -ItemType Directory -Path "bench-out" | Out-Null }
$results | Export-Csv -NoTypeInformation -Path "bench-out\bench-390-models.csv"
Write-Host "`nResults -> bench-out\bench-390-models.csv" -ForegroundColor Green
