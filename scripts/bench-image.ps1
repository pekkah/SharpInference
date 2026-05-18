param(
    [string]$Prompt = "a serene mountain lake at sunrise",
    [int]$Width = 512,
    [int]$Height = 512,
    [int]$Steps = 4,
    [string]$OutDir = "tools\bench",
    [int]$TimeoutSec = 600
)

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$dotnetExe = (Get-Command dotnet).Source
$cliDll    = ".\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"

$model     = "models\z_image_turbo-Q5_K_M.gguf"
$vae       = "models\z-image-turbo\vae"
$encoder   = "models\Z-Image-AbliteratedV1.Q5_K_M.gguf"
$tokenizer = "models\z-image-turbo\tokenizer\tokenizer.json"

function Run-One([string]$tag, [string]$outPng) {
    $stdoutPath = Join-Path $OutDir "img-$tag.out"
    $stderrPath = Join-Path $OutDir "img-$tag.err"
    $argList = @("$cliDll", "image",
        "-m", "$model",
        "--vae", "$vae",
        "--qwen-encoder", "$encoder",
        "--qwen-tokenizer", "$tokenizer",
        "-p", "$Prompt",
        "-W", "$Width", "-H", "$Height",
        "--steps", "$Steps",
        "-o", "$outPng",
        "-v")
    Write-Host "[img-$tag] $Width x $Height, $Steps steps (timeout ${TimeoutSec}s)" -ForegroundColor DarkGray
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = $dotnetExe
    $psi.Arguments = ($argList | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $psi.UseShellExecute = $false
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    $timedOut = $false
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        Write-Host "[img-$tag] TIMEOUT — killing" -ForegroundColor Red
        try { $proc.Kill($true) } catch {}
        $timedOut = $true
    }
    $sw.Stop()
    $stdoutTask.Wait(2000) | Out-Null
    $stderrTask.Wait(2000) | Out-Null
    Set-Content -Path $stdoutPath -Value $stdoutTask.Result
    Set-Content -Path $stderrPath -Value $stderrTask.Result
    return [PSCustomObject]@{
        Tag         = $tag
        WallSec     = [Math]::Round($sw.Elapsed.TotalSeconds, 1)
        TimedOut    = $timedOut
        OutPng      = $outPng
        OutSize     = if (Test-Path $outPng) { (Get-Item $outPng).Length } else { 0 }
    }
}

$r1 = Run-One "first" "tools\bench\img-first.png"
$r2 = Run-One "cached" "tools\bench\img-cached.png"

Write-Host ""
Write-Host "=== Z-Image-Turbo timing ===" -ForegroundColor Cyan
@($r1, $r2) | Format-Table Tag, WallSec, TimedOut, OutPng, OutSize -AutoSize

@($r1, $r2) | ConvertTo-Json | Set-Content "tools\bench\image-summary.json"
