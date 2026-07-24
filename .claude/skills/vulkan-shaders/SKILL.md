---
name: vulkan-shaders
description: Required workflow when adding, editing, or removing Vulkan GLSL shader consts — the committed precompiled SPIR-V table must be regenerated or tests fail on drift.
paths: src/SharpInference.Vulkan/**
---

# Vulkan shader editing workflow

Shaders are GLSL `const string`s in `src/SharpInference.Vulkan/Shaders.cs`,
precompiled to SPIR-V committed in `Shaders.Precompiled.g.cs` (keyed by FNV-1a
`ShaderCompiler.StableHash`) so the NativeAOT binary needs no glslc at runtime.
`ShaderCompiler.Compile` falls back to glslc only on a table miss.

After ANY add/edit/remove of a shader const:

1. Regenerate the table: `pwsh scripts/gen-spirv.ps1`
   (builds SharpInference.Vulkan, then runs `tools/SpirvGen`, which compiles every
   shader via glslc — requires the Vulkan SDK on PATH).
2. Verify: `dotnet test tests/SharpInference.Tests.ForwardPass --filter "FullyQualifiedName~VulkanPrecompiledShaderTests"`
   — this suite fails on any drift between `Shaders.cs` and the precompiled table.
3. Commit the regenerated `Shaders.Precompiled.g.cs` together with the shader change.

Caveats:

- Never hand-edit `Shaders.Precompiled.g.cs`.
- If glslc / the Vulkan SDK is unavailable in the current environment, make the
  shader change, state clearly that the table regen is pending, and expect
  `VulkanPrecompiledShaderTests` to fail until someone runs the script on a dev
  machine. Do not fake table entries to get green.
- `SgemmBf16` and `SgemmFp8` need extensions the bundled glslc lacks; they are
  listed in `SkippedShaders` and fall back to runtime compilation by design —
  don't "fix" their absence from the table.
