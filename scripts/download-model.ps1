<#
.SYNOPSIS
    Downloads GGUF models for SharpInference development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
    Supports: smollm2, qwen3-8b, llama31-70b, qwen3-30b-a3b, llama4-scout
.PARAMETER Model
    Which model to download. Default: downloads all.
.EXAMPLE
    .\download-model.ps1                           # All models
    .\download-model.ps1 -Model smollm2            # SmolLM2 1.7B (1.1 GB)
    .\download-model.ps1 -Model qwen3-8b           # Qwen3 8B (4.9 GB)
    .\download-model.ps1 -Model llama31-70b        # Llama 3.1 70B (40.8 GB)
    .\download-model.ps1 -Model llama4-scout       # Llama 4 Scout Q4_K_M (60.9 GB, 2 shards)
#>
param(
    [ValidateSet("smollm2", "qwen3-8b", "llama31-70b", "qwen3-30b-a3b", "llama4-scout")]
    [string]$Model
)

$Models = @{
    "smollm2" = @{
        Files = @("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
        Size  = "1.1 GB"
        Phase = "1-2"
    }
    "qwen3-8b" = @{
        Files = @("Qwen3-8B-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf")
        Size  = "4.9 GB"
        Phase = "2b-3"
    }
    "llama31-70b" = @{
        Files = @("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
        Size  = "40.8 GB"
        Phase = "4"
    }
    "qwen3-30b-a3b" = @{
        Files = @("Qwen3-30B-A3B-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf")
        Size  = "6.2 GB"
        Phase = "5a"
    }
    "llama4-scout" = @{
        Files = @(
            "Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf"
        )
        Urls  = @(
            "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF/resolve/main/Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf"
        )
        Size  = "60.9 GB (2 shards: 46.4 GB + 14.5 GB)"
        Phase = "5b"
    }
}

$ModelDir = Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

function Download-File {
    param([string]$url, [string]$path)
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        & curl.exe -L -o $path -C - --progress-bar $url
        if ($LASTEXITCODE -ne 0) { throw "curl exited with code $LASTEXITCODE" }
    }
    else {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $path -UseBasicParsing
        $ProgressPreference = 'Continue'
    }
}

function Download-Model {
    param([string]$key)
    $info = $Models[$key]

    # Check free disk space
    $drive   = Split-Path -Qualifier (Resolve-Path $ModelDir)
    $freeGB  = [math]::Round((Get-PSDrive ($drive.TrimEnd(':'))).Free / 1GB, 1)
    Write-Host "[$key] Free disk: $freeGB GB"

    $allPresent = $true
    foreach ($file in $info.Files) {
        if (-not (Test-Path (Join-Path $ModelDir $file))) { $allPresent = $false; break }
    }

    if ($allPresent) {
        $totalMB = ($info.Files | ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } | Measure-Object -Sum).Sum / 1MB
        Write-Host "[$key] Already complete: $($info.Files -join ', ') ($([math]::Round($totalMB, 1)) MB total) - Phase $($info.Phase)"
        return
    }

    Write-Host "[$key] Downloading $($info.Size) - Phase $($info.Phase)"
    Write-Host "  Files: $($info.Files -join ', ')"
    Write-Host ""

    for ($i = 0; $i -lt $info.Files.Count; $i++) {
        $file = $info.Files[$i]
        $url  = $info.Urls[$i]
        $path = Join-Path $ModelDir $file

        if (Test-Path $path) {
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  Shard $($i+1)/$($info.Files.Count): already present ($sizeMB MB), skipping"
            continue
        }

        Write-Host "  Shard $($i+1)/$($info.Files.Count): $file"
        Write-Host "  From: $url"
        Write-Host ""

        try {
            Download-File -url $url -path $path
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  Shard $($i+1) complete: $sizeMB MB"
        }
        catch {
            Write-Error "[$key] Download failed for shard $($i+1): $_"
            if (Test-Path $path) { Remove-Item $path }
            return
        }
    }

    $totalMB = ($info.Files | ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } | Measure-Object -Sum).Sum / 1MB
    Write-Host "[$key] All shards complete: $([math]::Round($totalMB, 1)) MB total"
}

if ($Model) {
    Download-Model -key $Model
}
else {
    foreach ($key in $Models.Keys) {
        Download-Model -key $key
    }
}
