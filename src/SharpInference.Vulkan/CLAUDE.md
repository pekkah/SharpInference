# SharpInference.Vulkan — shader workflow

Vulkan compute via `Vortice.Vulkan`; GPU buffer pool; implements both
`IComputeBackend` and `IImageOpsBackend`.

Shaders are GLSL `const string`s in `Shaders.cs`, precompiled to SPIR-V committed
in `Shaders.Precompiled.g.cs` (keyed by an FNV-1a `ShaderCompiler.StableHash`) so
the NativeAOT binary needs no glslc at runtime; `ShaderCompiler.Compile` falls
back to glslc only on a table miss.

- After adding/editing/removing a shader const, regenerate the table with
  `pwsh scripts/gen-spirv.ps1` (runs `tools/SpirvGen`, needs the Vulkan SDK) —
  `VulkanPrecompiledShaderTests` fails on drift. Commit the regenerated file with
  the shader change. Never hand-edit `Shaders.Precompiled.g.cs`.
- Shaders needing extensions the bundled glslc lacks (`SgemmBf16`, `SgemmFp8`)
  are recorded in `SkippedShaders` and fall back at runtime by design.
- Full procedure and no-Vulkan-SDK fallback: the `vulkan-shaders` skill
  (auto-activates on files in this directory).
