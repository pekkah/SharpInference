# Bench Carnice (Qwen3.6-35B-A3B-MTP APEX-MTP-I-Compact, mudler) CUDA hybrid in
# three configurations: Q8K gates OFF (current baseline), SHARPI_Q3K_Q8K=1
# alone (rank-1 #101), and both Q3K + Q8_0 gates on (rank-2 #99). Carnice
# routed experts always run on CPU (qwen35moe MoE path with mixed-precision
# weights: Q3_K + Q8_0 + Q4_K / Q5_K), so SHARPI_CPU_MOE=1 is set throughout.
param(
    [string]$Model = "E:\models\Carnice-Qwen3.6-MoE-35B-A3B-APEX-MTP-I-Compact.gguf",
    # Default prompt is more elaborate than the bench-all "capital of France"
    # one-liner because Carnice's agentic fine-tune terminates after a single
    # token on short factual recall ("Paris.<|im_end|>"), giving a useless
    # 1-decode-token bench cell. This prompt reliably elicits >=60 tokens.
    [string]$Prompt = "Explain in detail how rainbows are formed. Cover light refraction, water droplets, and color dispersion.",
    [int]$NTokens = 60,
    [int]$TimeoutSec = 900
)

if (-not (Test-Path $Model)) {
    Write-Error "Model not found: $Model"
    exit 1
}

$results = @()
$env:SHARPI_CPU_MOE = "1"

# Ensure both Q8K gates start OFF
Remove-Item env:SHARPI_Q3K_Q8K -ErrorAction SilentlyContinue
Remove-Item env:SHARPI_Q8_0_Q8K -ErrorAction SilentlyContinue

try {
    # Baseline: legacy DotQ3K / DotQ8_0 (f32 dequant-FMA)
    $results += .\scripts\bench-textgen.ps1 -Model $Model -Tag "carnice-cuda-hybrid-baseline" `
        -NTokens $NTokens -Prompt $Prompt -TimeoutSec $TimeoutSec `
        -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")

    # Q3K gate only (rank 1 #101)
    $env:SHARPI_Q3K_Q8K = "1"
    $results += .\scripts\bench-textgen.ps1 -Model $Model -Tag "carnice-cuda-hybrid-q3k-q8k" `
        -NTokens $NTokens -Prompt $Prompt -TimeoutSec $TimeoutSec `
        -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")

    # Both gates (rank 1 + rank 2 #99)
    $env:SHARPI_Q8_0_Q8K = "1"
    $results += .\scripts\bench-textgen.ps1 -Model $Model -Tag "carnice-cuda-hybrid-both-q8k" `
        -NTokens $NTokens -Prompt $Prompt -TimeoutSec $TimeoutSec `
        -ExtraArgs @("-g","-1","--backend","cuda","--no-thinking")
}
finally {
    Remove-Item env:SHARPI_Q3K_Q8K  -ErrorAction SilentlyContinue
    Remove-Item env:SHARPI_Q8_0_Q8K -ErrorAction SilentlyContinue
}

$results | Format-Table Tag, PrefillTok, PrefillTps, DecodeTok, DecodeTps, MtpAccept, WallSec, TimedOut
$results | Export-Csv -NoTypeInformation -Path "bench-out\bench-carnice.csv"
Write-Host ""
Write-Host "Results written to bench-out\bench-carnice.csv" -ForegroundColor Green
