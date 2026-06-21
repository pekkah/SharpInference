# Regenerates src/SharpInference.Vulkan/Shaders.Precompiled.g.cs from the GLSL shader
# consts in Shaders.cs. Requires the Vulkan SDK (glslc) — the committed output it produces
# compiles with no glslc, so this only runs on a dev machine after editing a shader.
#
# Usage: scripts/gen-spirv.ps1
#
# Whenever you ADD, REMOVE, or EDIT a shader const in Shaders.cs, run this to refresh the
# precompiled SPIR-V table (the VulkanPrecompiledShaderTests will otherwise fail on drift).

param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path "$PSScriptRoot\.."

Write-Host "Building SharpInference.Vulkan ($Configuration)..."
dotnet build "$repoRoot\src\SharpInference.Vulkan" -c $Configuration | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Vulkan build failed." }

Write-Host "Running SpirvGen (compiles every shader via glslc)..."
dotnet run --project "$repoRoot\tools\SpirvGen" -c $Configuration
if ($LASTEXITCODE -ne 0) { throw "SpirvGen failed." }

Write-Host "Done. Rebuild SharpInference.Vulkan to pick up the regenerated table."
