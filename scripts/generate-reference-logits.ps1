<#
.SYNOPSIS
    Generates reference logits from llama.cpp for forward pass validation.
.DESCRIPTION
    Runs llama.cpp with --logits-all on a fixed prompt and saves the output
    for comparison against SharpInference's forward pass implementation.
.PARAMETER Prompt
    The test prompt (default: "The capital of France is")
.PARAMETER NumTokens
    Number of tokens to generate (default: 10)
#>
param(
    [string]$Prompt = "The capital of France is",
    [int]$NumTokens = 10
)

$RepoRoot = Split-Path $PSScriptRoot -Parent
$LlamaCli = Join-Path $RepoRoot "tools\llama.cpp\llama-cli.exe"
$ModelPath = Join-Path $RepoRoot "models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
$OutputDir = Join-Path $RepoRoot "tests\reference-data"
$LogitsFile = Join-Path $OutputDir "smollm2-1.7b-logits.bin"

if (-not (Test-Path $LlamaCli)) {
    Write-Error "llama-cli.exe not found. Run setup-llamacpp.ps1 first."
    exit 1
}

if (-not (Test-Path $ModelPath)) {
    Write-Error "Model not found. Run download-model.ps1 first."
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

Write-Host "Generating reference logits..."
Write-Host "Prompt: `"$Prompt`""
Write-Host "Tokens: $NumTokens"
Write-Host ""

# Generate with greedy sampling (temp=0) for deterministic output
& $LlamaCli `
    -m $ModelPath `
    -p $Prompt `
    -n $NumTokens `
    --temp 0 `
    --logits-all `
    --no-display-prompt `
    -o $LogitsFile `
    2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Error "llama-cli failed with exit code $LASTEXITCODE"
    exit 1
}

if (Test-Path $LogitsFile) {
    $sizeMB = [math]::Round((Get-Item $LogitsFile).Length / 1MB, 2)
    Write-Host ""
    Write-Host "Reference logits saved: $LogitsFile ($sizeMB MB)"
} else {
    Write-Warning "Logits file was not created. Your llama.cpp build may not support --logits-all with -o."
    Write-Host "The greedy-decoded text output above can still be used for output-identity validation."
}
