# Issue #388 (router-on-GPU lever): measure the CPU-MoE prefill router moved to the GPU.
# Op-offload stays ON (the #390 default) throughout; we toggle SHARPI_MOE_GPU_ROUTER.
# 2K-token prompt. Prefill is the win; decode should be unchanged (decode uses the CPU router).
param(
    [string]$PromptFile = "C:\Users\pekka\AppData\Local\Temp\claude\C--p-sharpi\e585495f-574a-494e-820c-3cfea3d513e6\scratchpad\prefill_prompt.txt",
    [int]$NTokens = 40,
    [int]$TimeoutSec = 900
)
$cliDll = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
$dotnet = (Get-Command dotnet).Source
$env:SHARPI_CPU_MOE = "1"
$models = @(
    @{ Name="Carnice";     Path="E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf" }
    @{ Name="35B-A3B";     Path="E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" }
    @{ Name="35B-A3B-MTP"; Path="E:\models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf" }
)
function Run-One([string]$name, [string]$model, [string]$router) {
    if ($router -eq "off") { $env:SHARPI_MOE_GPU_ROUTER = "0" } else { Remove-Item env:SHARPI_MOE_GPU_ROUTER -ErrorAction SilentlyContinue }
    $argList = @("$cliDll","-m","$model","-f","$PromptFile","--temp","0","-n","$NTokens",
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
    $pTok=0;$pTps=0.0;$dTps=0.0
    if ($stdout -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $pTok=[int]$matches[1]; $pTps=[double]::Parse($matches[2],$inv) }
    if ($stdout -match 'Decode:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s')  { $dTps=[double]::Parse($matches[2],$inv) }
    [PSCustomObject]@{ Model=$name; Router=$router; Prefill=$pTok; PrefillTps=[Math]::Round($pTps,1); DecodeTps=[Math]::Round($dTps,1) }
}
$results = @()
try {
    foreach ($m in $models) {
        if (-not (Test-Path $m.Path)) { Write-Host "skip $($m.Path)" -ForegroundColor Yellow; continue }
        foreach ($r in @("off","on")) { Write-Host "[$($m.Name)/router-$r]..." -ForegroundColor Cyan; $results += Run-One $m.Name $m.Path $r }
    }
} finally { Remove-Item env:SHARPI_MOE_GPU_ROUTER -ErrorAction SilentlyContinue }
$results | Format-Table Model, Router, Prefill, PrefillTps, DecodeTps -AutoSize
if (-not (Test-Path "bench-out")) { New-Item -ItemType Directory -Path "bench-out" | Out-Null }
$results | Export-Csv -NoTypeInformation -Path "bench-out\bench-388-router.csv"
Write-Host "`nResults -> bench-out\bench-388-router.csv" -ForegroundColor Green
