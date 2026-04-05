<#
.SYNOPSIS
    Downloads GGUF models for SharpInference development.
.DESCRIPTION
    Downloads from HuggingFace to the models/ directory. Skips if already present.
    Supports: smollm2, qwen3-8b, llama31-70b, qwen3-coder-30b-a3b, llama4-scout,
              z-image-turbo, z-image-turbo-q8
.PARAMETER Model
    Which model to download. Default: downloads all text models (skips large image models).
.EXAMPLE
    .\download-model.ps1                                # All text models
    .\download-model.ps1 -Model smollm2                 # SmolLM2 1.7B (1.1 GB)
    .\download-model.ps1 -Model qwen3-8b                # Qwen3 8B (4.9 GB)
    .\download-model.ps1 -Model llama31-70b             # Llama 3.1 70B (40.8 GB)
    .\download-model.ps1 -Model qwen3-coder-30b-a3b     # Qwen3-Coder 30B-A3B Q4_K_M (18.6 GB)
    .\download-model.ps1 -Model llama4-scout            # Llama 4 Scout Q4_K_M (60.9 GB, 2 shards)
    .\download-model.ps1 -Model z-image-turbo           # Z-Image-Turbo Q5_K_M + abliterated encoder (~8.5 GB)
    .\download-model.ps1 -Model z-image-turbo-q8        # Z-Image-Turbo Q8_0 + abliterated encoder Q8_0 (~12 GB)
#>
param(
    [ValidateSet("smollm2", "qwen3-8b", "llama31-70b", "qwen3-coder-30b-a3b", "llama4-scout",
                 "z-image-turbo", "z-image-turbo-q8")]
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
    "qwen3-coder-30b-a3b" = @{
        Files = @("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
        Urls  = @("https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
        Size  = "18.6 GB"
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
    # ── Image generation ──────────────────────────────────────────────────────
    # Z-Image-Turbo Q5_K_M (recommended balance of quality and size)
    #   DiT:      jayn7/Z-Image-Turbo-GGUF        (5.52 GB)
    #   Encoder:  BennyDaBall abliterated Qwen3-4B (2.89 GB, uncensored)
    #   VAE:      Tongyi-MAI/Z-Image-Turbo vae/    (0.33 GB)
    #   Tokenizer: Tongyi-MAI/Z-Image-Turbo tokenizer/ (11 MB)
    "z-image-turbo" = @{
        Files = @(
            "z_image_turbo-Q5_K_M.gguf",
            "Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "z-image-turbo\vae\diffusion_pytorch_model.safetensors",
            "z-image-turbo\tokenizer\tokenizer.json"
        )
        Urls  = @(
            "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q5_K_M.gguf",
            "https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1/resolve/main/Z-Image-AbliteratedV1.Q5_K_M.gguf",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/tokenizer/tokenizer.json"
        )
        Size  = "~8.5 GB (DiT 5.52 GB + encoder 2.89 GB + VAE 0.33 GB + tokenizer)"
        Phase = "image"
        IsImage = $true
    }
    # Z-Image-Turbo Q8_0 (maximum quality, needs ~16 GB VRAM)
    #   DiT:      jayn7/Z-Image-Turbo-GGUF Q8_0   (7.22 GB)
    #   Encoder:  BennyDaBall abliterated Q8_0     (4.28 GB)
    #   VAE + Tokenizer: same as above
    "z-image-turbo-q8" = @{
        Files = @(
            "z_image_turbo-Q8_0.gguf",
            "Z-Image-AbliteratedV1.Q8_0.gguf",
            "z-image-turbo\vae\diffusion_pytorch_model.safetensors",
            "z-image-turbo\tokenizer\tokenizer.json"
        )
        Urls  = @(
            "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf",
            "https://huggingface.co/BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1/resolve/main/Z-Image-AbliteratedV1.Q8_0.gguf",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/vae/diffusion_pytorch_model.safetensors",
            "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/main/tokenizer/tokenizer.json"
        )
        Size  = "~12 GB (DiT 7.22 GB + encoder 4.28 GB + VAE 0.33 GB + tokenizer)"
        Phase = "image"
        IsImage = $true
    }
}

$ModelDir= Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
}

function Download-File {
    param([string]$url, [string]$path)
    # Ensure parent directory exists (needed for z-image-turbo\vae\ etc.)
    $parent = Split-Path $path -Parent
    if (-not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
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

    # Component labels for image bundles
    $labels = @{
        "z_image_turbo-Q5_K_M.gguf"                         = "DiT (image model)"
        "z_image_turbo-Q8_0.gguf"                           = "DiT (image model)"
        "Z-Image-AbliteratedV1.Q5_K_M.gguf"                 = "Text encoder (abliterated Qwen3-4B)"
        "Z-Image-AbliteratedV1.Q8_0.gguf"                   = "Text encoder (abliterated Qwen3-4B)"
        "z-image-turbo\vae\diffusion_pytorch_model.safetensors" = "VAE decoder"
        "z-image-turbo\tokenizer\tokenizer.json"             = "Tokenizer"
    }

    for ($i = 0; $i -lt $info.Files.Count; $i++) {
        $file  = $info.Files[$i]
        $url   = $info.Urls[$i]
        $path  = Join-Path $ModelDir $file
        $label = if ($labels.ContainsKey($file)) { " ($($labels[$file]))" } else { "" }

        if (Test-Path $path) {
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  File $($i+1)/$($info.Files.Count)$label`: already present ($sizeMB MB), skipping"
            continue
        }

        Write-Host "  File $($i+1)/$($info.Files.Count)$label`: $file"
        Write-Host "  From: $url"
        Write-Host ""

        try {
            Download-File -url $url -path $path
            $sizeMB = [math]::Round((Get-Item $path).Length / 1MB, 1)
            Write-Host "  File $($i+1) complete: $sizeMB MB"
        }
        catch {
            Write-Error "[$key] Download failed for file $($i+1): $_"
            if (Test-Path $path) { Remove-Item $path }
            return
        }
    }

    $totalMB = ($info.Files | ForEach-Object { (Get-Item (Join-Path $ModelDir $_)).Length } | Measure-Object -Sum).Sum / 1MB
    Write-Host "[$key] All files complete: $([math]::Round($totalMB, 1)) MB total"

    # Print ready-to-use command for image models
    if ($info.IsImage) {
        Write-Host ""
        Write-Host "[$key] Ready to generate images:"
        if ($key -like "z-image-turbo*") {
            $ditFile = $info.Files[0]
            $encFile = $info.Files[1]
            Write-Host "  dotnet run --project src/SharpInference.Cli -c Release -- image \"
            Write-Host "    -m models/$ditFile \"
            Write-Host "    --vae models/z-image-turbo/vae \"
            Write-Host "    --qwen-encoder models/$encFile \"
            Write-Host "    --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \"
            Write-Host "    -p `"your prompt here`" -W 1024 -H 1024 --steps 9 -o output.png"
        }
    }
}

if ($Model) {
    Download-Model -key $Model
}
else {
    # Default: download text models only (image models are large and optional)
    $textModels = $Models.Keys | Where-Object { -not $Models[$_].IsImage } | Sort-Object
    foreach ($key in $textModels) {
        Download-Model -key $key
    }
    Write-Host ""
    Write-Host "Image models not downloaded by default (large). Run explicitly:"
    Write-Host "  .\download-model.ps1 -Model z-image-turbo      # Q5_K_M (~8.5 GB)"
    Write-Host "  .\download-model.ps1 -Model z-image-turbo-q8   # Q8_0   (~12 GB)"
}
