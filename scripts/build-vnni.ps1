# Builds the native AVX-512-VNNI CPU kernel DLL (sharpi_cpu_vnni.dll) with clang-cl.
#
# Produces native/cpu_vnni/sharpi_cpu_vnni.dll from native/cpu_vnni/q8k_vnni.c.
# This DLL is OPTIONAL: if clang-cl / the MSVC toolset / the Windows SDK is
# absent, the build is skipped (exit 0) and the managed AVX2 path remains the
# fallback at runtime. So this never fails a `dotnet build`.
#
# Mirrors the style of scripts/setup-openblas.ps1.

param(
    [string]$Source = "$PSScriptRoot\..\native\cpu_vnni\q8k_vnni.c",
    [string]$Output = "$PSScriptRoot\..\native\cpu_vnni\sharpi_cpu_vnni.dll"
)

$ErrorActionPreference = "Stop"

function Write-Warn([string]$msg) { Write-Host "[build-vnni] $msg" -ForegroundColor Yellow }
function Write-Info([string]$msg) { Write-Host "[build-vnni] $msg" }

# ---- Locate clang-cl --------------------------------------------------------
$clangcl = $null
$clangCandidates = @(
    "C:\Program Files\LLVM\bin\clang-cl.exe",
    "C:\Program Files (x86)\LLVM\bin\clang-cl.exe"
)
$onPath = Get-Command clang-cl -ErrorAction SilentlyContinue
if ($onPath) { $clangcl = $onPath.Source }
if (-not $clangcl) {
    foreach ($c in $clangCandidates) { if (Test-Path $c) { $clangcl = $c; break } }
}
if (-not $clangcl) {
    Write-Warn "clang-cl not found (looked on PATH and in LLVM install dirs)."
    Write-Warn "Skipping native VNNI build; managed AVX2 path will be used."
    exit 0
}
Write-Info "clang-cl: $clangcl"

# ---- Locate the MSVC toolset (prefer vswhere, then 2022 BuildTools) ----------
function Get-LatestSubdir([string]$parent) {
    if (-not (Test-Path $parent)) { return $null }
    Get-ChildItem -Path $parent -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
}

$vcRoot = $null
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsInstall = & $vswhere -latest -property installationPath 2>$null
    if ($vsInstall -and (Test-Path "$vsInstall\VC\Tools\MSVC")) {
        $vcRoot = "$vsInstall\VC\Tools\MSVC"
    }
}
if (-not $vcRoot) {
    foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
        $cand = "C:\Program Files (x86)\Microsoft Visual Studio\2022\$edition\VC\Tools\MSVC"
        if (Test-Path $cand) { $vcRoot = $cand; break }
        $cand2 = "C:\Program Files\Microsoft Visual Studio\2022\$edition\VC\Tools\MSVC"
        if (Test-Path $cand2) { $vcRoot = $cand2; break }
    }
}
if (-not $vcRoot) {
    Write-Warn "MSVC toolset (VC\Tools\MSVC) not found. Skipping native VNNI build."
    exit 0
}
$vcToolset = Get-LatestSubdir $vcRoot
if (-not $vcToolset) {
    Write-Warn "No MSVC toolset version dir under $vcRoot. Skipping native VNNI build."
    exit 0
}
$vc = $vcToolset.FullName
Write-Info "MSVC toolset: $vc"

# ---- Locate the Windows SDK -------------------------------------------------
$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
if (-not (Test-Path "$sdkRoot\Include")) {
    Write-Warn "Windows 10/11 SDK Include dir not found at $sdkRoot. Skipping native VNNI build."
    exit 0
}
$sdkVerDir = Get-LatestSubdir "$sdkRoot\Include"
if (-not $sdkVerDir) {
    Write-Warn "No Windows SDK version dir under $sdkRoot\Include. Skipping native VNNI build."
    exit 0
}
$sdkVer = $sdkVerDir.Name
Write-Info "Windows SDK: $sdkVer"

# ---- Compose INCLUDE / LIB --------------------------------------------------
$env:INCLUDE = @(
    "$vc\include",
    "$sdkRoot\Include\$sdkVer\ucrt",
    "$sdkRoot\Include\$sdkVer\shared",
    "$sdkRoot\Include\$sdkVer\um"
) -join ";"

$env:LIB = @(
    "$vc\lib\x64",
    "$sdkRoot\Lib\$sdkVer\ucrt\x64",
    "$sdkRoot\Lib\$sdkVer\um\x64"
) -join ";"

if (-not (Test-Path $Source)) {
    Write-Warn "Source not found: $Source. Nothing to build."
    exit 0
}

$outDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Info "Compiling $Source -> $Output"

# Build into the output dir so the .obj/.lib/.exp land alongside (cleaned after).
Push-Location $outDir
try {
    & $clangcl /nologo /LD /O2 `
        -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni `
        "/Fe:$Output" "$Source"
    $code = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($code -ne 0) {
    Write-Warn "clang-cl failed (exit $code). Native VNNI DLL not produced; AVX2 fallback remains."
    exit 0
}

# Clean up intermediates (keep only the DLL).
foreach ($ext in @("obj", "lib", "exp")) {
    Get-ChildItem -Path $outDir -Filter "*.$ext" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

if (Test-Path $Output) {
    $sizeKb = [math]::Round((Get-Item $Output).Length / 1KB, 1)
    Write-Info "Built: $Output ($sizeKb KB)"
} else {
    Write-Warn "Build reported success but $Output is missing. AVX2 fallback remains."
}

exit 0
