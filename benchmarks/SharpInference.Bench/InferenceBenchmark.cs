using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.TurboQuant;

namespace SharpInference.Bench;

[MemoryDiagnoser]
[WarmupCount(2)]
[IterationCount(10)]
public class InferenceBenchmark
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private GgufTokenizer _tokenizer = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("SmolLM2-1.7B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, _hp);

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    // ================================================================
    //  Decode N tokens — measures realistic sustained throughput
    //  as KV cache grows, attention cost increases per token.
    // ================================================================

    [Params(1, 32, 128)]
    public int TokenCount { get; set; }

    [IterationSetup(Target = nameof(DecodeTokens))]
    public void DecodeIterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "SmolLM2 Decode N tokens")]
    public int DecodeTokens()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < TokenCount; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    // ================================================================
    //  Prefill
    // ================================================================

    [IterationSetup(Targets = new[] { nameof(PrefillSequential), nameof(PrefillBatched) })]
    public void PrefillIterSetup()
    {
        _fwd.Cache.Reset();
    }

    [Benchmark(Description = "SmolLM2 Prefill sequential")]
    public int PrefillSequential()
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _fwd.Forward(_promptTokens[i], i);
        return Sampler.Greedy(logits);
    }

    [Benchmark(Description = "SmolLM2 Prefill batched")]
    public int PrefillBatched()
    {
        var logits = _fwd.Prefill(_promptTokens);
        return Sampler.Greedy(logits);
    }

    // ================================================================
    //  GPU Decode (SmolLM2)
    // ================================================================

    private Vulkan.VulkanBackend _gpu = null!;
    private Engine.GpuForwardPass _gpuFwd = null!;
    private int _gpuDecodePos;
    private int _gpuLastToken;

    [GlobalSetup(Targets = new[] { nameof(GpuDecodeTokens) })]
    public void GpuSetup()
    {
        var path = FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Model not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);

        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new Engine.GpuForwardPass(_model, _gpu, _hp);

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup(Target = nameof(GpuDecodeTokens))]
    public void GpuDecodeIterSetup()
    {
        _gpuFwd.ResetCache();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);
        _gpuLastToken = Sampler.Greedy(logits);
        _gpuDecodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "SmolLM2 GPU Decode 32 tokens")]
    [Arguments(32)]
    public int GpuDecodeTokens(int tokenCount)
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_gpuLastToken, _gpuDecodePos++);
        int lastToken = Sampler.Greedy(logits);
        for (int i = 1; i < tokenCount; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _gpuDecodePos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [GlobalCleanup(Targets = new[] { nameof(GpuDecodeTokens) })]
    public void GpuCleanup()
    {
        _gpuFwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }

    private static string? FindModelPath(string filename)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}

/// <summary>
/// Qwen3 8B benchmarks — separate class so BenchmarkDotNet can run them independently.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(5)]
public class Qwen3Benchmark
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private GgufTokenizer _tokenizer = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, _hp);

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [Params(1, 32)]
    public int TokenCount { get; set; }

    [IterationSetup(Target = nameof(DecodeTokens))]
    public void DecodeIterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Qwen3-8B Decode N tokens")]
    public int DecodeTokens()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < TokenCount; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [IterationSetup(Target = nameof(PrefillBatched))]
    public void PrefillIterSetup() => _fwd.Cache.Reset();

    [Benchmark(Description = "Qwen3-8B Prefill batched")]
    public int PrefillBatched()
    {
        var logits = _fwd.Prefill(_promptTokens);
        return Sampler.Greedy(logits);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }

    // ================================================================
    //  GPU Decode (Qwen3 8B)
    // ================================================================

    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwd = null!;
    private int _gpuDecodePos;
    private int _gpuLastToken;

    [GlobalSetup(Targets = new[] { nameof(GpuDecodeTokens) })]
    public void GpuSetup()
    {
        var path = FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);

        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new GpuForwardPass(_model, _gpu, _hp);

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup(Target = nameof(GpuDecodeTokens))]
    public void GpuDecodeIterSetup()
    {
        _gpuFwd.ResetCache();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);
        _gpuLastToken = Sampler.Greedy(logits);
        _gpuDecodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "Qwen3-8B GPU Decode 32 tokens")]
    [Arguments(32)]
    public int GpuDecodeTokens(int tokenCount)
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_gpuLastToken, _gpuDecodePos++);
        int lastToken = Sampler.Greedy(logits);
        for (int i = 1; i < tokenCount; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _gpuDecodePos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [GlobalCleanup(Targets = new[] { nameof(GpuDecodeTokens) })]
    public void GpuCleanup()
    {
        _gpuFwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }

    private static string? FindModelPath(string filename)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}

// ================================================================
//  Qwen3 8B TurboQuant CPU Benchmarks
// ================================================================

/// <summary>
/// Benchmarks Qwen3 8B decode with TurboQuant KV cache compression on CPU.
/// Compares FP32 baseline vs TQ3 (3-bit) to measure compression overhead.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(5)]
public class Qwen3TqCpuBenchmark
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private GgufTokenizer _tokenizer = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private ForwardPass _fwdTq = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);
        _backend = new CpuBackend();

        // FP32 baseline
        _fwd = new ForwardPass(_model, _backend, _hp);

        // TurboQuant enabled
        _fwdTq = new ForwardPass(_model, _backend, _hp);
        _fwdTq.EnableTurboQuant(fp32WindowSize: 256, bits: 3);

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [Params(32)]
    public int TokenCount { get; set; }

    // ── FP32 Baseline ──

    [IterationSetup(Target = nameof(DecodeBaseline))]
    public void BaselineIterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Baseline = true, Description = "Qwen3-8B CPU Decode (FP32 KV)")]
    public int DecodeBaseline()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < TokenCount; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    // ── TurboQuant TQ3 ──

    [IterationSetup(Target = nameof(DecodeTq3))]
    public void Tq3IterSetup()
    {
        _fwdTq.TqCache!.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwdTq.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Qwen3-8B CPU Decode (TQ3 KV)")]
    public int DecodeTq3()
    {
        ReadOnlySpan<float> logits = _fwdTq.Forward(
            Sampler.Greedy(_fwdTq.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < TokenCount; i++)
        {
            logits = _fwdTq.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _fwdTq.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}

// ================================================================
//  Qwen3 8B TurboQuant GPU Benchmarks
// ================================================================

/// <summary>
/// Benchmarks Qwen3 8B GPU decode: FP32 baseline.
/// Also reports estimated max context for FP32 vs TQ3 modes.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(5)]
public class Qwen3TqGpuBenchmark
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private GgufTokenizer _tokenizer = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _gpuDecodePos;
    private int _gpuLastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _tokenizer = GgufTokenizer.FromGgufModel(_model);

        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new GpuForwardPass(_model, _gpu, _hp);

        // Report estimated context sizes for comparison
        int fp32Ctx = _gpuFwd.MaxSeqLen;
        int tqCtx = GpuForwardPass.EstimateMaxContextTq(_model, _gpu, _hp);
        Console.Error.WriteLine($"[Qwen3TqGpuBenchmark] FP32 auto context: {fp32Ctx}, TQ3 estimated context: {tqCtx}");

        _promptTokens = _tokenizer.Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup(Target = nameof(GpuDecodeFp32))]
    public void Fp32IterSetup()
    {
        _gpuFwd.ResetCache();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);
        _gpuLastToken = Sampler.Greedy(logits);
        _gpuDecodePos = _promptTokens.Count;
    }

    [Benchmark(Baseline = true, Description = "Qwen3-8B GPU Decode 32t (FP32 KV)")]
    public int GpuDecodeFp32()
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_gpuLastToken, _gpuDecodePos++);
        int lastToken = Sampler.Greedy(logits);
        for (int i = 1; i < 32; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _gpuDecodePos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _gpuFwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}

// ================================================================
//  TurboQuant Micro-Benchmarks
// ================================================================

/// <summary>
/// Micro-benchmarks for TurboQuant primitive operations:
/// WHT, quantize, dequant-dot (scalar vs AVX2), bit packing.
/// No model file required.
/// </summary>
[MemoryDiagnoser]
public unsafe class TurboQuantMicroBenchmarks
{
    private const int Dim = 128;

    private float[] _inputVec = null!;
    private float[] _queryVec = null!;
    private float[] _rotatedQuery = null!;
    private float[] _signPattern = null!;
    private float[] _centroids3 = null!;
    private float[] _boundaries3 = null!;
    private float[] _centroids4 = null!;
    private float[] _boundaries4 = null!;
    private byte[] _compressed3 = null!;
    private byte[] _compressed4 = null!;
    private float[] _decompressed = null!;
    private float[] _whtOutput = null!;
    private byte[][] _compressedBatch = null!;

    [Params(256, 1024, 4096)]
    public int SeqLen { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _inputVec = new float[Dim];
        _queryVec = new float[Dim];
        _rotatedQuery = new float[Dim];
        _decompressed = new float[Dim];
        _whtOutput = new float[Dim];

        float norm = 0;
        for (int i = 0; i < Dim; i++)
        {
            _inputVec[i] = (float)(rng.NextDouble() * 2 - 1);
            _queryVec[i] = (float)(rng.NextDouble() * 2 - 1);
            norm += _inputVec[i] * _inputVec[i];
        }
        norm = MathF.Sqrt(norm);
        for (int i = 0; i < Dim; i++) _inputVec[i] /= norm;

        _signPattern = WalshHadamard.GenerateSignPattern(Dim, 0);
        _centroids3 = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        _boundaries3 = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();
        _centroids4 = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        _boundaries4 = TurboQuantCodebooks.Boundaries4Bit_D128.ToArray();

        // Pre-compress for dequant benchmarks
        _compressed3 = new byte[TurboQuantOps.BlockSize(3, Dim)];
        _compressed4 = new byte[TurboQuantOps.BlockSize(4, Dim)];
        TurboQuantOps.Quantize(_inputVec, _compressed3, _signPattern, _centroids3, _boundaries3, 3, Dim);
        TurboQuantOps.Quantize(_inputVec, _compressed4, _signPattern, _centroids4, _boundaries4, 4, Dim);

        // Pre-rotate query
        TurboQuantOps.RotateQuery(_queryVec, _rotatedQuery, _signPattern, Dim);

        // Pre-compress batch for attention simulation
        int blockSize = TurboQuantOps.BlockSize(3, Dim);
        float[] vec = new float[Dim];
        _compressedBatch = new byte[8192][];
        for (int i = 0; i < _compressedBatch.Length; i++)
        {
            float n = 0;
            for (int d = 0; d < Dim; d++) { vec[d] = (float)(rng.NextDouble() * 2 - 1); n += vec[d] * vec[d]; }
            n = MathF.Sqrt(n);
            for (int d = 0; d < Dim; d++) vec[d] /= n;
            _compressedBatch[i] = new byte[blockSize];
            TurboQuantOps.Quantize(vec, _compressedBatch[i], _signPattern, _centroids3, _boundaries3, 3, Dim);
        }
    }

    // ── Single-operation benchmarks ──

    [Benchmark(Description = "WHT d=128")]
    public void WalshHadamardTransform()
    {
        WalshHadamard.Transform(_inputVec, _whtOutput, Dim);
    }

    [Benchmark(Description = "RotateQuery d=128")]
    public void RotateQuery()
    {
        TurboQuantOps.RotateQuery(_queryVec, _rotatedQuery, _signPattern, Dim);
    }

    [Benchmark(Description = "Quantize 3-bit d=128")]
    public void Quantize3Bit()
    {
        TurboQuantOps.Quantize(_inputVec, _compressed3, _signPattern, _centroids3, _boundaries3, 3, Dim);
    }

    [Benchmark(Description = "Quantize 4-bit d=128")]
    public void Quantize4Bit()
    {
        TurboQuantOps.Quantize(_inputVec, _compressed4, _signPattern, _centroids4, _boundaries4, 4, Dim);
    }

    [Benchmark(Description = "Dequantize 3-bit d=128")]
    public void Dequantize3Bit()
    {
        TurboQuantOps.Dequantize(_compressed3, _decompressed, _signPattern, _centroids3, 3, Dim);
    }

    [Benchmark(Description = "DequantDot 3-bit scalar d=128")]
    public float DequantDot3Scalar()
    {
        return TurboQuantOps.DequantDot(_compressed3, _rotatedQuery, _centroids3, 3, Dim);
    }

    [Benchmark(Description = "DequantDot 3-bit AVX2 d=128")]
    public float DequantDot3Avx2()
    {
        fixed (byte* pComp = _compressed3)
        fixed (float* pQuery = _rotatedQuery, pCentroids = _centroids3)
            return TurboQuantOps.DequantDot3Avx2(pComp, pQuery, pCentroids, Dim);
    }

    [Benchmark(Description = "DequantDot 4-bit scalar d=128")]
    public float DequantDot4Scalar()
    {
        return TurboQuantOps.DequantDot(_compressed4, _rotatedQuery, _centroids4, 4, Dim);
    }

    [Benchmark(Description = "DequantDot 4-bit AVX2 d=128")]
    public float DequantDot4Avx2()
    {
        fixed (byte* pComp = _compressed4)
        fixed (float* pQuery = _rotatedQuery, pCentroids = _centroids4)
            return TurboQuantOps.DequantDot4Avx2(pComp, pQuery, pCentroids, Dim);
    }

    // ── Batch: simulate attention scoring over N cached positions ──

    [Benchmark(Description = "Batch DequantDot 3-bit AVX2 (N positions)")]
    public float BatchDequantDot3Avx2()
    {
        float sum = 0;
        for (int t = 0; t < SeqLen; t++)
        {
            fixed (byte* pComp = _compressedBatch[t])
            fixed (float* pQuery = _rotatedQuery, pCentroids = _centroids3)
                sum += TurboQuantOps.DequantDot3Avx2(pComp, pQuery, pCentroids, Dim);
        }
        return sum;
    }
}

// ================================================================
//  Shared helper
// ================================================================

internal static class BenchmarkHelper
{
    public static string? FindModelPath(string filename)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
