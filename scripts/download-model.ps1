<#
.SYNOPSIS
    Downloads GGUF models for SharpInference development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
    Supports: smollm2, qwen3-8b, llama31-70b, qwen3-30b-a3b
.PARAMETER Model
    Which model to download. Default: downloads all.
.EXAMPLE
    .\download-model.ps1                           # All models
    .\download-model.ps1 -Model smollm2            # SmolLM2 1.7B (1.1 GB)
    .\download-model.ps1 -Model qwen3-8b           # Qwen3 8B (4.9 GB)
    .\download-model.ps1 -Model llama31-70b        # Llama 3.1 70B (40.8 GB) - Phase 4
    .\download-model.ps1 -Model qwen3-30b-a3b      # Qwen3 30B-A3B MoE (17.2 GB) - Phase 5
#>
param(
    [ValidateSet("smollm2", "qwen3-8b", "llama31-70b", "qwen3-30b-a3b")]
    [string]$Model
)

$Models = @{
    "smollm2" = @{
        File  = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
        Url   = "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
        Size  = "1.1 GB"
        Phase = "1-2"
    }
    "qwen3-8b" = @{
        File  = "Qwen3-8B-Q4_K_M.gguf"
        Url   = "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
        Size  = "4.9 GB"
        Phase = "2b-3"
    }
    "llama31-70b" = @{
        File  = "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
        Url   = "https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
        Size  = "40.8 GB"
        Phase = "4"
    }
    "qwen3-30b-a3b" = @{
        File  = "Qwen3-30B-A3B-Q4_K_M.gguf"
        Url   = "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf"
        Size  = "17.2 GB"
        Phase = "5"
    }
}

$ModelDir = Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

function Download-Model {
    param([string]$key)
    $info = $Models[$key]
    $path = Join-Path $ModelDir $info.File

    if (Test-Path $path) {
        $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "[$key] Already exists: $($info.File) ($sizeMB MB) - Phase $($info.Phase)"
        return
    }

    Write-Host "[$key] Downloading $($info.File) ($($info.Size)) - Phase $($info.Phase)..."
    Write-Host "  From: $($info.Url)"
    Write-Host "  To:   $path"
    Write-Host ""

    try {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            & curl.exe -L -o $path -C - --progress-bar $info.Url
            if ($LASTEXITCODE -ne 0) { throw "curl exited with code $LASTEXITCODE" }
        }
        else {
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $info.Url -OutFile $path -UseBasicParsing
            $ProgressPreference = 'Continue'
        }

        $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "[$key] Download complete: $sizeMB MB"
    }
    catch {
        Write-Error "[$key] Download failed: $_"
        if (Test-Path $path) { Remove-Item $path }
    }
}

if ($Model) {
    Download-Model -key $Model
}
else {
    foreach ($key in $Models.Keys) {
        Download-Model -key $key
    }
}
