<#
.SYNOPSIS
    Downloads the SmolLM2 1.7B Instruct GGUF model (Q4_K_M) for Phase 1 development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
#>

$ModelDir = Join-Path $PSScriptRoot "..\models"
$ModelFile = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
$ModelPath = Join-Path $ModelDir $ModelFile
$Url = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/$ModelFile"

if (Test-Path $ModelPath) {
    Write-Host "Model already exists: $ModelPath"
    Write-Host "Size: $([math]::Round((Get-Item $ModelPath).Length / 1MB, 1)) MB"
    exit 0
}

if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

Write-Host "Downloading $ModelFile (~1.1 GB)..."
Write-Host "From: $Url"
Write-Host "To:   $ModelPath"
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $Url -OutFile $ModelPath -UseBasicParsing
    $ProgressPreference = 'Continue'

    $size = [math]::Round((Get-Item $ModelPath).Length / 1MB, 1)
    Write-Host "Download complete: $size MB"
} catch {
    Write-Error "Download failed: $_"
    if (Test-Path $ModelPath) { Remove-Item $ModelPath }
    exit 1
}
