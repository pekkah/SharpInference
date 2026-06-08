# Focused re-measure of the 7 MTP rows in the README. MTP self-speculative decode is
# prompt-sensitive (draft acceptance varies with content), so the generic essay prompt
# used by bench-allrows-decode.ps1 under-measures these cells. Here each model uses a
# coding/explanation prompt that elicits a long, high-acceptance generation, with the
# same warm-clock discipline (a discarded full run first) as the other sweeps.
$ErrorActionPreference = "Continue"
$E = "E:\models"

# Coding prompt for the dense/MoE MTP models (matches bench-27b-mtp.ps1); the elaborate
# explanation prompt for Carnice (its agentic tune 1-token-EOSes on short factual recall).
# Code-only prompt (matches bench-27b-mtp.ps1 exactly). An "...then explain step by step"
# suffix was tried and dropped: the prose continuation is less predictable than code, so
# the MTP head's draft acceptance fell ~7 pp (27B-CUDA 95%→88%) and decode with it.
$code = "Write a Python function that sorts a list using the quicksort algorithm:"
$expl = "Explain in detail how rainbows are formed. Cover light refraction, water droplets, and color dispersion."

$jobs = @(
  @{ Tag="27b-mtp-q4-cpu";  M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf";     A=@("--no-thinking");                                     P=$code; T=900 }
  @{ Tag="27b-mtp-q4-cuda"; M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf";     A=@("-g","-1","--backend","cuda","--no-thinking");        P=$code; T=900 }
  @{ Tag="27b-mtp-q5-cpu";  M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf";     A=@("--no-thinking");                                     P=$code; T=900 }
  @{ Tag="27b-mtp-q5-cuda"; M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf";     A=@("-g","-1","--backend","cuda","--no-thinking");        P=$code; T=900 }
  @{ Tag="35b-mtp-cpu";     M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("--no-thinking");                                  P=$code; CpuMoe=$true; T=900 }
  @{ Tag="35b-mtp-cuda";    M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");     P=$code; CpuMoe=$true; T=900 }
  @{ Tag="carnice-cuda";    M="$E\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking"); P=$expl; CpuMoe=$true; T=900 }
)

$warmed = @{}
$rows = @()
foreach ($j in $jobs) {
    if (-not (Test-Path $j.M)) { Write-Host "[skip] $($j.Tag): $($j.M) missing" -ForegroundColor Yellow; continue }
    if ($j.CpuMoe) { $env:SHARPI_CPU_MOE = "1" } else { Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue }

    if (-not $warmed.ContainsKey($j.M)) {
        Write-Host "--- warming $($j.Tag) model ---" -ForegroundColor DarkGray
        $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-mwarm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec $j.T -ExtraArgs $j.A
        $warmed[$j.M] = $true
    }
    $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-mwarmclk" -NTokens 80 -Prompt $j.P -TimeoutSec $j.T -ExtraArgs $j.A

    $r = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-mtp" -NTokens 80 -Prompt $j.P -TimeoutSec $j.T -ExtraArgs $j.A
    Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue
    $rows += [PSCustomObject]@{ Tag=$j.Tag; DecodeTps=$r.DecodeTps; Mtp=$r.MtpAccept; DecTok=$r.DecodeTok; Wall=$r.WallSec; TO=$r.TimedOut }
    Write-Host ("  {0,-18} dec={1,6} t/s  mtp={2,4}%  ({3} tok, {4}s{5})" -f $j.Tag,$r.DecodeTps,$r.MtpAccept,$r.DecodeTok,$r.WallSec,($(if($r.TimedOut){" TIMEOUT"}else{""}))) -ForegroundColor Green
}
Write-Host ""
Write-Host "=== MTP rows near-zero decode + acceptance (warm, proper prompt) ===" -ForegroundColor Cyan
$rows | Format-Table -AutoSize
$rows | Export-Csv -NoTypeInformation -Path "tools\bench\allrows-mtp.csv"
Write-Host "CSV: tools\bench\allrows-mtp.csv"
