# Issue #129 A/B: GPU-SLRU MoE prefill (SHARPI_CPU_MOE=0) — this branch vs master.
# Forces the on-GPU routed-expert path that #129's fused weighted-scatter-reduce kernel
# optimizes (the default auto-routes MoE to CPU, so no standard bench row exercises it).
# Confirms Backend: CUDA hybrid + GPU-SLRU in EACH run before trusting the number
# (a prior $args/PowerShell collision silently fell back to CPU).
param(
  [string]$Model = "E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  [int]$NTokens = 8
)

# Identical ~1K-token prompt to bench-allrows-1k.ps1.
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

$dlls = [ordered]@{
  "master"   = "C:\p\sharpi-master\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
  "branch129" = "C:\p\sharpi\src\SharpInference.Cli\bin\Release\net10.0\sharpi-cli.dll"
}

$env:SHARPI_CPU_MOE = "0"   # force GPU-SLRU on-GPU routed experts (#129 path)
$dotnet = (Get-Command dotnet).Source
$outDir = "C:\p\sharpi\tools\bench"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

function Run([string]$label, [string]$dll, [bool]$warm) {
  $args = @($dll, "-m", $Model, "-p", $prompt, "--temp", "0", "-n", "$NTokens",
            "--ngl", "-1", "--backend", "cuda", "--single-turn", "--verbose-prompt")
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $dotnet
  $psi.Arguments = ($args | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
  $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true; $psi.UseShellExecute = $false
  $p = [System.Diagnostics.Process]::Start($psi)
  $so = $p.StandardOutput.ReadToEndAsync(); $se = $p.StandardError.ReadToEndAsync()
  $p.WaitForExit(900*1000) | Out-Null
  $so.Wait(3000) | Out-Null; $se.Wait(3000) | Out-Null
  $out = $so.Result + "`n" + $se.Result
  if ($warm) { return }
  Set-Content -Path (Join-Path $outDir "129ab-$label.txt") -Value $out
  # Backend confirmation (contract: must be CUDA hybrid + GPU-SLRU, NOT CPU fallback).
  $slru = ($out | Select-String -Pattern "SLRU expert cache").Matches.Count
  $cpuMoe = ($out | Select-String -Pattern "MoE.*on CPU|auto-routed to CPU|Dense FFN mode").Matches.Count
  $backendLine = ($out -split "`n" | Where-Object { $_ -match "Backend|SLRU expert cache|GDN:|MoE:" } | Select-Object -First 4) -join " | "
  $prefill = "n/a"
  if ($out -match 'Prefill:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $prefill = "$($matches[1]) tok @ $($matches[2]) t/s" }
  $decode = "n/a"
  if ($out -match 'Decode:\s+(\d+)\s+tokens,\s+([\d\.]+)\s+t/s') { $decode = "$($matches[2]) t/s" }
  [PSCustomObject]@{ Label=$label; Prefill=$prefill; Decode=$decode; SLRUlines=$slru; CpuMoeHits=$cpuMoe; Backend=$backendLine }
}

Write-Host "=== Warming OS page cache for $Model ===" -ForegroundColor DarkGray
Run "warm" $dlls["branch129"] $true

$results = @()
foreach ($kv in $dlls.GetEnumerator()) {
  Write-Host "=== Running $($kv.Key) (SHARPI_CPU_MOE=0, GPU-SLRU) ===" -ForegroundColor Cyan
  $results += Run $kv.Key $kv.Value $false
}
$results | Format-List
