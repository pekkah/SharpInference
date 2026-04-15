# Downloads and extracts OpenBLAS for Windows x64.
# Only bin/libopenblas.dll is needed for P/Invoke at runtime.

param(
    [string]$Version = "0.3.28",
    [string]$OutputDir = "$PSScriptRoot\..\tools\openblas"
)

$ErrorActionPreference = "Stop"

$url = "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$Version/OpenBLAS-$Version-x64.zip"
$zipFile = "$env:TEMP\OpenBLAS-$Version-x64.zip"
$dllPath = "$OutputDir\libopenblas.dll"

if (Test-Path $dllPath) {
    Write-Host "OpenBLAS already present at $dllPath"
    exit 0
}

Write-Host "Downloading OpenBLAS $Version..."
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Invoke-WebRequest -Uri $url -OutFile $zipFile -UseBasicParsing

Write-Host "Extracting libopenblas.dll..."
$tempDir = "$env:TEMP\openblas-extract"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
Expand-Archive -Path $zipFile -DestinationPath $tempDir

# Find and copy just the DLL
$dll = Get-ChildItem -Path $tempDir -Recurse -Filter "libopenblas.dll" | Select-Object -First 1
if (-not $dll) {
    Write-Error "libopenblas.dll not found in archive"
    exit 1
}
Copy-Item $dll.FullName $dllPath
Write-Host "Installed: $dllPath ($([math]::Round($dll.Length / 1MB, 1)) MB)"

# Cleanup
Remove-Item -Force $zipFile
Remove-Item -Recurse -Force $tempDir

Write-Host "Done. OpenBLAS ready at $OutputDir"
