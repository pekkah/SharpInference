# SharpInference

A high-performance LLM inference engine and image generation pipeline for .NET 10. Reads GGUF model files and runs transformer inference on CPU (AVX2/AVX-512 SIMD) or GPU (Vulkan compute shaders / CUDA cuBLAS). Includes Z-Image-Turbo text-to-image and Real-ESRGAN upscaling.

This is the **library** package. For a command-line tool, install [`SharpInference.Cli`](https://www.nuget.org/packages/SharpInference.Cli) instead.

## Install

```
dotnet add package SharpInference
```

## Quick start

```csharp
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

var model = GgufModelLoader.Load("models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
var backend = new CpuBackend();
var forward = new ForwardPass(model, backend);
var engine = new InferenceEngine(forward, model.Tokenizer);

await foreach (var token in engine.GenerateAsync("Hello, ", new SamplingParams { Temperature = 0.7f }))
{
    Console.Write(token);
}
```

For GPU inference, swap `CpuBackend` for `VulkanBackend` or `CudaBackend`, or use `HybridForwardPass` to offload selected layers.

## What's in the package

This is a **meta-package**: installing it pulls in every SharpInference sub-package via transitive NuGet dependencies. The sub-packages can also be installed individually if you only need a subset (e.g., no GPU backend).

| Sub-package | Purpose |
|-------------|---------|
| [`SharpInference.Core`](https://www.nuget.org/packages/SharpInference.Core) | GGUF parsing, BPE tokenizer, tensor types, model graph |
| [`SharpInference.Cpu`](https://www.nuget.org/packages/SharpInference.Cpu) | CPU backend (AVX2/AVX-512 SIMD, Q4_K_M dequant, optional OpenBLAS) |
| [`SharpInference.Vulkan`](https://www.nuget.org/packages/SharpInference.Vulkan) | Vulkan compute backend |
| [`SharpInference.Cuda`](https://www.nuget.org/packages/SharpInference.Cuda) | CUDA / cuBLAS backend + NVRTC kernels |
| [`SharpInference.Engine`](https://www.nuget.org/packages/SharpInference.Engine) | Forward pass, paged KV cache, samplers, speculative decoding |
| [`SharpInference.Diffusion`](https://www.nuget.org/packages/SharpInference.Diffusion) | Z-Image-Turbo + FLUX.1 image generation |
| [`SharpInference.Pipeline`](https://www.nuget.org/packages/SharpInference.Pipeline) | 3-tier VRAM → RAM → NVMe memory hierarchy |
| [`SharpInference.TurboQuant`](https://www.nuget.org/packages/SharpInference.TurboQuant) | 3-bit KV-cache compression |

## Optional native dependencies

- **OpenBLAS** (CPU GEMM acceleration) — auto-detected on PATH, silently skipped if absent.
- **Vulkan drivers** — up-to-date GPU drivers (AMD / Intel / NVIDIA). No extra install on Windows.
- **CUDA Toolkit 11.x or 12.x** — `cublas64_*.dll` and `cudart64_*.dll` on PATH. NVIDIA only.

## NativeAOT

All assemblies are trim-safe and NativeAOT-compatible. To publish a single-binary application:

```
dotnet publish -c Release -r win-x64
```

## Links

- [Repository & docs](https://github.com/pekkah/SharpInference)
- [Design document](https://github.com/pekkah/SharpInference/blob/master/docs/SharpInference-Design.md)
- [Issues](https://github.com/pekkah/SharpInference/issues)

## License

MIT. Copyright (c) 2026 Pekka Heikura.
