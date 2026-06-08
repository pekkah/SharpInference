# Companion to bench-allrows-1k.ps1: re-measure the README "Decode t/s" column at
# near-zero context (short prompt) so it stays comparable to llama.cpp tg128 and the
# per-issue near-zero figures. Same job table, same warm-clock discipline (a discarded
# full run for GPU configs so boost clocks have ramped), 60 generated tokens.
$ErrorActionPreference = "Continue"

$C = "C:\p\sharpi\models"
$E = "E:\models"

# Near-zero-ctx prompt: short to keep prefill negligible, but OPEN-ENDED so every model
# generates the full 60 tokens. A factual prompt ("capital of France") makes the model
# emit a one-line answer and hit EOS after a few tokens, so decode t/s is then computed
# over a handful of tokens dominated by fixed first-token cost (SmolLM2 mis-measured at
# ~14 t/s instead of ~38). An essay prompt guarantees a long, EOS-free generation.
$prompt = "Write a detailed multi-paragraph explanation of how the water cycle works on Earth."

$jobs = @(
  @{ Tag="smol-cpu";        M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@();                                               T=300 }
  @{ Tag="smol-vulkan";     M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");                 T=300 }
  @{ Tag="smol-cuda";       M="$C\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                   T=300 }

  @{ Tag="qwen3-cpu";       M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@();                                                            T=400 }
  @{ Tag="qwen3-cpu-tq";    M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("--tq");                                                      T=400 }
  @{ Tag="qwen3-vulkan";    M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");                             T=400 }
  @{ Tag="qwen3-vulkan-tq"; M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan","--tq");                      T=400 }
  @{ Tag="qwen3-cuda";      M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                               T=400 }
  @{ Tag="qwen3-cuda-nt";   M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");               T=400 }
  @{ Tag="qwen3-cuda-tq";   M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--tq");                        T=400 }
  @{ Tag="qwen3-cuda-tq-nt";M="$C\Qwen3-8B-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--tq","--no-thinking");        T=400 }

  @{ Tag="olmoe-cpu";       M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@();                                          T=400 }
  @{ Tag="olmoe-vulkan";    M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");            T=400 }
  @{ Tag="olmoe-cuda";      M="$C\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");              T=400 }

  @{ Tag="coder-cpu";       M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@();                                       T=600 }
  @{ Tag="coder-cpu-tq";    M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("--tq");                                 T=600 }
  @{ Tag="coder-cuda";      M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");           T=600 }

  @{ Tag="qwen36-35b-cpu";  M="$E\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"; A=@();                                                 T=900 }
  @{ Tag="qwen36-35b-cuda"; M="$E\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda");                     T=900 }

  @{ Tag="27b-mtp-q4-cpu";  M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf"; A=@("--no-thinking");                                     T=900 }
  @{ Tag="27b-mtp-q4-cuda"; M="$E\Qwen3.6-27B-MTP-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");        T=900 }
  @{ Tag="27b-mtp-q5-cpu";  M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf"; A=@("--no-thinking");                                     T=900 }
  @{ Tag="27b-mtp-q5-cuda"; M="$E\Qwen3.6-27B-MTP-Q5_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking");        T=900 }

  @{ Tag="35b-mtp-cpu";     M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("--no-thinking");                       CpuMoe=$true; T=900 }
  @{ Tag="35b-mtp-cuda";    M="$E\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking"); CpuMoe=$true; T=900 }

  @{ Tag="carnice-cuda";    M="$E\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf"; A=@("-g","-1","--backend","cuda","--no-thinking"); CpuMoe=$true; T=900 }

  @{ Tag="gemma4-cpu";      M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@();                                                       T=900 }
  @{ Tag="gemma4-cuda";     M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@("-g","-1","--backend","cuda","--ctx-size","2048");       T=900 }
  @{ Tag="gemma4-cuda-hyb"; M="$E\gemma-4-E4B-it-Q8_0.gguf"; A=@("-g","22","--backend","cuda","--ctx-size","2048");       T=900 }

  @{ Tag="gemma4-12b-cuda"; M="$E\gemma-4-12b-it-qat-q4_0.gguf"; A=@("-g","-1","--backend","cuda","--ctx-size","2048");  T=900 }
)

$warmed = @{}
$rows = @()
foreach ($j in $jobs) {
    if (-not (Test-Path $j.M)) { Write-Host "[skip] $($j.Tag): $($j.M) missing" -ForegroundColor Yellow; continue }
    if ($j.CpuMoe) { $env:SHARPI_CPU_MOE = "1" } else { Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue }

    if (-not $warmed.ContainsKey($j.M)) {
        Write-Host "--- warming $($j.Tag) model ---" -ForegroundColor DarkGray
        $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-dwarm" -NTokens 8 -Prompt "Hello, world." -TimeoutSec $j.T -ExtraArgs $j.A
        $warmed[$j.M] = $true
    }
    # Warm-up measured run, discarded: GPU configs need it for boost clocks, and CPU
    # configs need it for JIT — a short near-zero decode on a freshly-JITted process
    # under-measures the first tokens (e.g. SmolLM2 finishes in ~4 s, before hot paths
    # compile). The 1K-prefill sweep masks this; here we must warm explicitly.
    $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-dwarmclk" -NTokens 60 -Prompt $prompt -TimeoutSec $j.T -ExtraArgs $j.A

    $r = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-dec" -NTokens 60 -Prompt $prompt -TimeoutSec $j.T -ExtraArgs $j.A
    Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue
    $rows += [PSCustomObject]@{ Tag=$j.Tag; PrefTok=$r.PrefillTok; DecodeTps=$r.DecodeTps; Mtp=$r.MtpAccept; Wall=$r.WallSec; TO=$r.TimedOut }
    Write-Host ("  {0,-18} dec={1,6} t/s  ({2} tok, {3}s{4})" -f $j.Tag,$r.DecodeTps,$r.PrefillTok,$r.WallSec,($(if($r.TimedOut){" TIMEOUT"}else{""}))) -ForegroundColor Green
}
Write-Host ""
Write-Host "=== All-rows near-zero-ctx decode (warm) ===" -ForegroundColor Cyan
$rows | Format-Table -AutoSize
$rows | Export-Csv -NoTypeInformation -Path "tools\bench\allrows-decode.csv"
Write-Host "CSV: tools\bench\allrows-decode.csv"
