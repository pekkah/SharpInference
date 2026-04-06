using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Diffusion;

namespace SharpInference.ImageBench;

// ── Micro benchmarks for CUDA backend hot paths ────────────────────────────────
//
// Run with:
//   dotnet run --project benchmarks/SharpInference.ImageBench -c Release -- --bench
//
// These benchmarks exercise the operations that dominate Z-Image-Turbo inference:
//   1. Host-to-device upload (activation tensors, per-GEMM)
//   2. Device-to-host download (result tensors, per-GEMM)
//   3. cuBLAS SGEMM (the core compute kernel)
//   4. cudaMalloc / cudaFree round-trip (per-GEMM allocation overhead)
//   5. Full DiT forward pass (end-to-end single-step latency)
//
// All benchmarks use synthetic data so no model file is required.
// ──────────────────────────────────────────────────────────────────────────────

/// <summary>Benchmark CUDA upload/download bandwidth at realistic activation sizes.</summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public class CudaTransferBenchmarks
{
    // Typical activation sizes seen in Z-Image-Turbo:
    //   nTok=1088 (1024 image + 64 text), dim=3840 → 4.2 MB fp32 / 2.1 MB bf16
    //   FFN intermediate: nTok × ffnHidden = 1088 × 10240 → 11.2 MB
    [Params(1024 * 64, 1024 * 3840, 1024 * 10240)]
    public int FloatCount { get; set; }

    private CudaBackend _backend = null!;
    private float[]  _hostBuf  = null!;
    private ushort[] _hostBf16 = null!;
    private Core.Tensor? _gpuTensor;

    [GlobalSetup]
    public void Setup()
    {
        if (!CudaBackend.IsAvailable())
            throw new InvalidOperationException("CUDA not available — skip transfer benchmarks");
        _backend  = CudaBackend.Create();
        _hostBuf  = new float[FloatCount];
        _hostBf16 = new ushort[FloatCount];
        var rng = new Random(42);
        for (int i = 0; i < FloatCount; i++)
        {
            _hostBuf[i]  = (float)rng.NextDouble();
            uint bits = BitConverter.SingleToUInt32Bits(_hostBuf[i]);
            _hostBf16[i] = (ushort)(bits >> 16);
        }
        // Pre-allocate a device tensor for the download benchmark
        _gpuTensor = _backend.Upload(_hostBuf.AsSpan(), TensorShape.D1(FloatCount));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_gpuTensor is not null) _backend.Free(_gpuTensor);
        _backend.Dispose();
    }

    [Benchmark(Description = "Upload fp32 H→D")]
    public Core.Tensor UploadFp32()
    {
        var t = _backend.Upload(_hostBuf.AsSpan(), TensorShape.D1(FloatCount));
        _backend.Free(t);
        return t;
    }

    [Benchmark(Description = "Upload bf16 H→D")]
    public Core.Tensor UploadBf16()
    {
        var t = _backend.UploadBf16(_hostBf16.AsSpan(), TensorShape.D1(FloatCount));
        _backend.Free(t);
        return t;
    }

    [Benchmark(Description = "Download fp32 D→H")]
    public void DownloadFp32()
    {
        _backend.Download(_gpuTensor!, _hostBuf.AsSpan());
    }
}

/// <summary>Benchmark cuBLAS SGEMM at realistic DiT weight shapes.</summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public class CudaSgemmBenchmarks
{
    // Representative GEMM shapes from Z-Image-Turbo (nTok=1088, dim=3840):
    //   QKV projection: M=1088, K=3840, N=3×3840=11520
    //   FFN gate/up:    M=1088, K=3840, N=10240
    //   FFN down:       M=1088, K=10240, N=3840
    //   Output proj:    M=1088, K=3840, N=3840
    [Params("qkv", "ffn_gate", "ffn_down", "out_proj")]
    public string Shape { get; set; } = "qkv";

    private CudaBackend _backend = null!;
    private Core.Tensor? _A, _B, _C;
    private int _M, _K, _N;

    [GlobalSetup]
    public void Setup()
    {
        if (!CudaBackend.IsAvailable())
            throw new InvalidOperationException("CUDA not available — skip SGEMM benchmarks");
        _backend = CudaBackend.Create(SgemmPrecision.Bf16);

        (_M, _K, _N) = Shape switch
        {
            "qkv"      => (1088, 3840, 11520),
            "ffn_gate" => (1088, 3840, 10240),
            "ffn_down" => (1088, 10240, 3840),
            "out_proj" => (1088, 3840, 3840),
            _          => (1088, 3840, 3840),
        };

        var rng = new Random(42);
        var aData = new ushort[_M * _K];
        var bData = new ushort[_N * _K];
        for (int i = 0; i < aData.Length; i++) { float v = (float)rng.NextDouble(); aData[i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16); }
        for (int i = 0; i < bData.Length; i++) { float v = (float)rng.NextDouble(); bData[i] = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16); }

        _A = _backend.UploadBf16(aData.AsSpan(), TensorShape.D1(aData.Length));
        _B = _backend.UploadBf16(bData.AsSpan(), TensorShape.D1(bData.Length));
        _C = _backend.Allocate(TensorShape.D1(_M * _N), DType.BFloat16);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_A is not null) _backend.Free(_A);
        if (_B is not null) _backend.Free(_B);
        if (_C is not null) _backend.Free(_C);
        _backend.Dispose();
    }

    [Benchmark(Description = "SGEMM bf16")]
    public void SgemmBf16()
    {
        _backend.Sgemm(_C!, _A!, _B!, _M, _K, _N);
        _backend.Synchronize();
    }
}

/// <summary>
/// Benchmark the overhead of cudaMalloc + cudaFree vs the GPU buffer pool.
/// This measures the allocation tax paid per-GEMM when no pool is used.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public class CudaAllocBenchmarks
{
    // Size representative of an activation tensor (nTok=1088, dim=3840, bf16 → ~8 MB)
    private const int TensorFloats = 1088 * 3840;

    private CudaBackend _backend = null!;

    [GlobalSetup]
    public void Setup()
    {
        if (!CudaBackend.IsAvailable())
            throw new InvalidOperationException("CUDA not available");
        _backend = CudaBackend.Create();
    }

    [GlobalCleanup]
    public void Cleanup() => _backend.Dispose();

    [Benchmark(Description = "Allocate+Free activation tensor")]
    public void AllocFree()
    {
        var t = _backend.Allocate(TensorShape.D1(TensorFloats));
        _backend.Free(t);
    }

    [Benchmark(Description = "Allocate+Free ×6 (one block's worth)")]
    public void AllocFree6()
    {
        // Simulate one DiT block: 4 MatQ calls × 2 tensors (xGpu + cGpu)
        // + adaLN MatQ × 1 tensor pair = 5 pairs ≈ 10 alloc/free pairs
        var t = new Core.Tensor[6];
        for (int i = 0; i < 6; i++)
            t[i] = _backend.Allocate(TensorShape.D1(TensorFloats));
        for (int i = 0; i < 6; i++)
            _backend.Free(t[i]);
    }
}

/// <summary>
/// End-to-end single DiT forward-pass latency benchmark (no VAE, no text encode).
/// Uses synthetic activations; measures pure transformer compute per denoising step.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class DiTForwardPassBenchmarks
{
    // 512×512 image → 64×64 latent → 32×32 patches (patch size=2)
    private const int PatchH = 32, PatchW = 32;
    private const int NImg   = PatchH * PatchW;  // 1024
    private const int NTxt   = 64;               // typical Qwen3 token count

    private ZImageDiT?     _dit;
    private IComputeBackend? _backend;
    private float[]? _imgPatches;
    private float[]? _txtEmbeds;
    private int[]?   _imgPosIds;
    private int[]?   _txtPosIds;
    private ZImageParams _p = new();

    [Params("cpu", "cuda_bf16", "cuda_fp32")]
    public string Backend { get; set; } = "cpu";

    [GlobalSetup]
    public void Setup()
    {
        string? ditPath = BenchmarkHelper.FindFile(Path.Combine("models", "z_image_turbo-Q5_K_M.gguf"));
        if (ditPath is null)
            throw new FileNotFoundException("z_image_turbo-Q5_K_M.gguf not found — skipping DiT benchmarks");

        _backend = Backend switch
        {
            "cuda_bf16" when CudaBackend.IsAvailable() => CudaBackend.Create(SgemmPrecision.Bf16),
            "cuda_fp32" when CudaBackend.IsAvailable() => CudaBackend.Create(SgemmPrecision.Fp32),
            _                                           => new SharpInference.Cpu.CpuBackend(),
        };

        var loader = SharpInference.Diffusion.GgufWeightLoader.Open(ditPath);
        _dit = new ZImageDiT(loader, _p, _backend);

        var rng = new Random(42);
        _imgPatches = new float[NImg * _p.PatchDim];
        _txtEmbeds  = new float[NTxt * _p.CapFeatDim];
        for (int i = 0; i < _imgPatches.Length; i++) _imgPatches[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < _txtEmbeds.Length;  i++) _txtEmbeds[i]  = (float)(rng.NextDouble() - 0.5);

        _imgPosIds = ZImageRoPE.ImagePosIds(NTxt, PatchH, PatchW);
        _txtPosIds = ZImageRoPE.TextPosIds(NTxt);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _dit?.Dispose();
        (_backend as IDisposable)?.Dispose();
    }

    [Benchmark(Description = "DiT single forward step")]
    public float[] DiTForward()
        => _dit!.Forward(_imgPatches!, _imgPosIds!, _txtEmbeds!, _txtPosIds!, t: 0.5f);
}

/// <summary>Helper for resolving model/repo paths inside benchmarks.</summary>
public static class BenchmarkHelper
{
    public static string? FindFile(string relative)
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, relative);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
