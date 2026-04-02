<#
.SYNOPSIS
    Downloads GGUF models for SharpInference development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
    Supports: smollm2 (default), qwen3-8b
.PARAMETER Model
    Which model to download: "smollm2" (1.1 GB) or "qwen3-8b" (4.9 GB). Default: smollm2.
.PARAMETER All
    Download all supported models.
.EXAMPLE
    .\download-model.ps1                    # SmolLM2 1.7B (default)
    .\download-model.ps1 -Model qwen3-8b   # Qwen3 8B
    .\download-model.ps1 -All              # All models
#>
param(
    [ValidateSet("smollm2", "qwen3-8b")]
    [string]$Model
)

$Models = @{
    "smollm2" = @{
        File = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
        Url  = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
        Size = "~1.1 GB"
    }
    "qwen3-8b" = @{
        File = "Qwen3-8B-Q4_K_M.gguf"
        Url  = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
        Size = "~4.9 GB"
    }
}

$ModelDir = Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

function Download-Model($key) {
    $info = $Models[$key]
    $path = Join-Path $ModelDir $info.File

    if (Test-Path $path) {
        Write-Host "[$key] Already exists: $path"
        Write-Host "  Size: $([math]::Round((Get-Item $path).Length / 1MB, 1)) MB"
        return
    }

    Write-Host "[$key] Downloading $($info.File) ($($info.Size))..."
    Write-Host "  From: $($info.Url)"
    Write-Host "  To:   $path"
    Write-Host ""

    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $info.Url -OutFile $path -UseBasicParsing
        $ProgressPreference = 'Continue'

        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "[$key] Download complete: $size MB"
    } catch {
        Write-Error "[$key] Download failed: $_"
        if (Test-Path $path) { Remove-Item $path }
    }
}

if ($Model) {
    Download-Model $Model
} else {
    foreach ($key in $Models.Keys) {
        Download-Model $key
    }
}
