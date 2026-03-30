<#
.SYNOPSIS
    Downloads and extracts llama.cpp prebuilt binaries for reference logit generation.
.DESCRIPTION
    Downloads the latest llama.cpp Windows release from GitHub to tools/llama.cpp/.
    Used to generate reference logits for validating SharpInference forward pass correctness.
.PARAMETER Variant
    Build variant: cpu, vulkan, cuda-12.4, cuda-13.1 (default: cpu)
.PARAMETER Version
    Release tag (default: b8585)
.EXAMPLE
    .\setup-llamacpp.ps1
    .\setup-llamacpp.ps1 -Variant vulkan
    .\setup-llamacpp.ps1 -Variant cuda-12.4 -Version b8585
#>
param(
    [ValidateSet("cpu", "vulkan", "cuda-12.4", "cuda-13.1")]
    [string]$Variant = "cpu",
    [string]$Version = "b8585"
)

$ToolsDir = Join-Path $PSScriptRoot "..\tools\llama.cpp"
$MarkerFile = Join-Path $ToolsDir "VERSION"
$ZipName = "llama-$Version-bin-win-$Variant-x64.zip"
$Url = "https://github.com/ggml-org/llama.cpp/releases/download/$Version/$ZipName"
$ZipPath = Join-Path $env:TEMP $ZipName

# Check if already installed with this version+variant
if (Test-Path $MarkerFile) {
    $installed = Get-Content $MarkerFile -Raw
    if ($installed.Trim() -eq "$Version-$Variant") {
        Write-Host "llama.cpp $Version ($Variant) already installed at: $ToolsDir"
        $cli = Join-Path $ToolsDir "llama-cli.exe"
        if (Test-Path $cli) {
            & $cli --version 2>$null
        }
        exit 0
    }
}

# Clean previous install
if (Test-Path $ToolsDir) {
    Write-Host "Removing previous llama.cpp installation..."
    Remove-Item -Recurse -Force $ToolsDir
}
New-Item -ItemType Directory -Path $ToolsDir -Force | Out-Null

Write-Host "Downloading llama.cpp $Version ($Variant)..."
Write-Host "From: $Url"
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath -UseBasicParsing
    $ProgressPreference = 'Continue'
} catch {
    Write-Error "Download failed: $_"
    exit 1
}

$sizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "Downloaded: $sizeMB MB"
Write-Host "Extracting to: $ToolsDir"

try {
    Expand-Archive -Path $ZipPath -DestinationPath $ToolsDir -Force

    # llama.cpp zips often have a nested folder — flatten if needed
    $nested = Get-ChildItem $ToolsDir -Directory
    if ($nested.Count -eq 1 -and (Test-Path (Join-Path $nested[0].FullName "llama-cli.exe"))) {
        $nestedPath = $nested[0].FullName
        Get-ChildItem $nestedPath | Move-Item -Destination $ToolsDir -Force
        Remove-Item $nestedPath -Force
    }
} catch {
    Write-Error "Extraction failed: $_"
    exit 1
} finally {
    Remove-Item $ZipPath -Force -ErrorAction SilentlyContinue
}

# Write version marker
"$Version-$Variant" | Set-Content $MarkerFile

# Verify
$cli = Join-Path $ToolsDir "llama-cli.exe"
if (Test-Path $cli) {
    Write-Host ""
    Write-Host "llama.cpp installed successfully."
    & $cli --version 2>$null
} else {
    Write-Warning "llama-cli.exe not found. Contents of $ToolsDir :"
    Get-ChildItem $ToolsDir | Select-Object Name
}
