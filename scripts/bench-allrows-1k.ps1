# Re-benchmark every on-disk README row at a uniform ~1K-token "normal" context,
# warm-cache, current code (#114-B batched trunk on by default). Goal: make the
# README "Prefill t/s" column consistent (warm @~1K ctx) instead of the old
# ~10-token launch-overhead cells. Decode is captured for reference but the README
# decode column / notes stay as the near-zero-ctx generation rate.
$ErrorActionPreference = "Continue"

$C = "C:\p\sharpi\models"
$E = "E:\models"

# ~1K-token prompt: a few varied technical sections + a task (not pure repetition).
$para = @(
"Modern large language model inference is dominated by memory bandwidth rather than raw compute.",
"Each decode step streams the full weight matrix for every layer, so quantization formats like Q4_K and Q5_K trade a small accuracy loss for a large reduction in bytes moved per token.",
"Mixture-of-experts models complicate this: only a handful of the hundreds of experts fire per token, and which fire varies token to token, defeating simple weight caching.",
"Gated DeltaNet layers replace quadratic attention with a linear recurrent state update, bounding per-token cost as context grows, at the price of a strictly sequential scan.",
"Hybrid placement keeps the attention and recurrent trunk on the accelerator while streaming routed-expert weights from host memory, overlapping the two so neither stalls."
) -join " "
$sb = [System.Text.StringBuilder]::new()
[void]$sb.Append("Read the following engineering notes and then write a concise technical summary.`n`n")
for ($i = 1; $i -le 6; $i++) { [void]$sb.Append("Section $i. $para`n`n") }
[void]$sb.Append("Summarize the main performance trade-offs across the sections above.")
$prompt = $sb.ToString()

# Job table: Tag, Model, Args, Env (CPU_MOE), Timeout. Order groups by model file so
# the first run of each model warms the OS page cache for the ones after it.
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
  @{ Tag="coder-vulkan";    M="$C\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"; A=@("-g","-1","--backend","vulkan");         T=1800 }
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
)

$warmed = @{}
$rows = @()
foreach ($j in $jobs) {
    if (-not (Test-Path $j.M)) { Write-Host "[skip] $($j.Tag): $($j.M) missing" -ForegroundColor Yellow; continue }
    if ($j.CpuMoe) { $env:SHARPI_CPU_MOE = "1" } else { Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue }

    # Warm the OS page cache for this model file once (short prompt, discarded).
    if (-not $warmed.ContainsKey($j.M)) {
        Write-Host "--- warming $($j.Tag) model ---" -ForegroundColor DarkGray
        $null = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-warm1k" -NTokens 8 -Prompt "Hello, world." -TimeoutSec $j.T -ExtraArgs $j.A
        $warmed[$j.M] = $true
    }

    $r = .\scripts\bench-textgen.ps1 -Model $j.M -Tag "$($j.Tag)-1k" -NTokens 60 -Prompt $prompt -TimeoutSec $j.T -ExtraArgs $j.A
    Remove-Item env:SHARPI_CPU_MOE -ErrorAction SilentlyContinue
    $rows += [PSCustomObject]@{ Tag=$j.Tag; PrefTok=$r.PrefillTok; PrefillTps=$r.PrefillTps; DecodeTps=$r.DecodeTps; Mtp=$r.MtpAccept; Wall=$r.WallSec; TO=$r.TimedOut }
    Write-Host ("  {0,-18} pref={1,7} t/s  dec={2,6} t/s  ({3} tok, {4}s{5})" -f $j.Tag,$r.PrefillTps,$r.DecodeTps,$r.PrefillTok,$r.WallSec,($(if($r.TimedOut){" TIMEOUT"}else{""}))) -ForegroundColor Green
}
Write-Host ""
Write-Host "=== All-rows @~1K ctx (warm) ===" -ForegroundColor Cyan
$rows | Format-Table -AutoSize
$rows | Export-Csv -NoTypeInformation -Path "tools\bench\allrows-1k.csv"
Write-Host "CSV: tools\bench\allrows-1k.csv"
