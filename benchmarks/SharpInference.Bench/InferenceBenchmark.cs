using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.TurboQuant;

namespace SharpInference.Bench;

// Each benchmark class owns a single model/backend setup so BenchmarkDotNet
// only keeps one large model resident at a time.

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class SmolLM2CpuBenchmarks
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [Params(1, 32, 128)]
    public int TokenCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("SmolLM2-1.7B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

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

    [IterationSetup(Targets = [nameof(PrefillSequential), nameof(PrefillBatched)])]
    public void PrefillIterSetup() => _fwd.Cache.Reset();

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

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class SmolLM2GpuDecodeBenchmark
{
    private GgufModel _model = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("SmolLM2-1.7B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new GpuForwardPass(_model, _gpu, hp);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _gpuFwd.ResetCache();

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);

        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "SmolLM2 GPU Decode 32 tokens")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_lastToken, _decodePos++);
        int lastToken = Sampler.Greedy(logits);

        for (int i = 1; i < 32; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _decodePos++);
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

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Qwen3CpuBenchmarks
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [Params(1, 32)]
    public int TokenCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

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
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Qwen3GpuDecodeBenchmark
{
    private GgufModel _model = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new GpuForwardPass(_model, _gpu, hp);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _gpuFwd.ResetCache();

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);

        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "Qwen3-8B GPU Decode 32 tokens")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_lastToken, _decodePos++);
        int lastToken = Sampler.Greedy(logits);

        for (int i = 1; i < 32; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _decodePos++);
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

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Qwen3TqCpuBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private ForwardPass _fwdTq = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [Params(32)]
    public int TokenCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);
        _fwdTq = new ForwardPass(_model, _backend, hp);
        _fwdTq.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

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

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Qwen3TqGpuBenchmark
{
    private GgufModel _model = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _gpu = new Vulkan.VulkanBackend();
        _gpuFwd = new GpuForwardPass(_model, _gpu, hp);

        int tqCtx = GpuForwardPass.EstimateMaxContextTq(_model, _gpu, hp);
        Console.Error.WriteLine(
            $"[Qwen3TqGpuBenchmark] FP32 context: {_gpuFwd.MaxSeqLen}, TQ3 estimated context: {tqCtx}");

        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _gpuFwd.ResetCache();

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwd.Forward(_promptTokens[i], i);

        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "Qwen3-8B GPU Decode 32t (FP32 KV)")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _gpuFwd.Forward(_lastToken, _decodePos++);
        int lastToken = Sampler.Greedy(logits);

        for (int i = 1; i < 32; i++)
        {
            logits = _gpuFwd.Forward(lastToken, _decodePos++);
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

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Qwen3TqGpuDecodeBenchmark
{
    private GgufModel _model = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private GpuForwardPass _gpuFwdTq = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Qwen3-8B-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Qwen3-8B-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata);
        _gpu = new Vulkan.VulkanBackend();
        _gpuFwdTq = new GpuForwardPass(_model, _gpu, hp, enableTurboQuant: true);

        Console.Error.WriteLine($"[Qwen3TqGpuDecodeBenchmark] TQ3 context: {_gpuFwdTq.MaxSeqLen}");

        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _gpuFwdTq.ResetCache();

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _gpuFwdTq.Forward(_promptTokens[i], i);

        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "Qwen3-8B GPU Decode 32t (TQ3 KV)")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _gpuFwdTq.Forward(_lastToken, _decodePos++);
        int lastToken = Sampler.Greedy(logits);

        for (int i = 1; i < 32; i++)
        {
            logits = _gpuFwdTq.Forward(lastToken, _decodePos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _gpuFwdTq?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}

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
        for (int i = 0; i < Dim; i++)
            _inputVec[i] /= norm;

        _signPattern = WalshHadamard.GenerateSignPattern(Dim, 0);
        _centroids3 = TurboQuantCodebooks.Centroids3Bit_D128.ToArray();
        _boundaries3 = TurboQuantCodebooks.Boundaries3Bit_D128.ToArray();
        _centroids4 = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        _boundaries4 = TurboQuantCodebooks.Boundaries4Bit_D128.ToArray();

        _compressed3 = new byte[TurboQuantOps.BlockSize(3, Dim)];
        _compressed4 = new byte[TurboQuantOps.BlockSize(4, Dim)];
        TurboQuantOps.Quantize(_inputVec, _compressed3, _signPattern, _centroids3, _boundaries3, 3, Dim);
        TurboQuantOps.Quantize(_inputVec, _compressed4, _signPattern, _centroids4, _boundaries4, 4, Dim);
        TurboQuantOps.RotateQuery(_queryVec, _rotatedQuery, _signPattern, Dim);

        int blockSize = TurboQuantOps.BlockSize(3, Dim);
        float[] vec = new float[Dim];
        _compressedBatch = new byte[8192][];
        for (int i = 0; i < _compressedBatch.Length; i++)
        {
            float n = 0;
            for (int d = 0; d < Dim; d++)
            {
                vec[d] = (float)(rng.NextDouble() * 2 - 1);
                n += vec[d] * vec[d];
            }

            n = MathF.Sqrt(n);
            for (int d = 0; d < Dim; d++)
                vec[d] /= n;

            _compressedBatch[i] = new byte[blockSize];
            TurboQuantOps.Quantize(vec, _compressedBatch[i], _signPattern, _centroids3, _boundaries3, 3, Dim);
        }
    }

    [Benchmark(Description = "WHT d=128")]
    public void WalshHadamardTransform() => WalshHadamard.Transform(_inputVec, _whtOutput, Dim);

    [Benchmark(Description = "RotateQuery d=128")]
    public void RotateQuery() => TurboQuantOps.RotateQuery(_queryVec, _rotatedQuery, _signPattern, Dim);

    [Benchmark(Description = "Quantize 3-bit d=128")]
    public void Quantize3Bit() =>
        TurboQuantOps.Quantize(_inputVec, _compressed3, _signPattern, _centroids3, _boundaries3, 3, Dim);

    [Benchmark(Description = "Quantize 4-bit d=128")]
    public void Quantize4Bit() =>
        TurboQuantOps.Quantize(_inputVec, _compressed4, _signPattern, _centroids4, _boundaries4, 4, Dim);

    [Benchmark(Description = "Dequantize 3-bit d=128")]
    public void Dequantize3Bit() =>
        TurboQuantOps.Dequantize(_compressed3, _decompressed, _signPattern, _centroids3, 3, Dim);

    [Benchmark(Description = "DequantDot 3-bit scalar d=128")]
    public float DequantDot3Scalar() => TurboQuantOps.DequantDot(_compressed3, _rotatedQuery, _centroids3, 3, Dim);

    [Benchmark(Description = "DequantDot 3-bit AVX2 d=128")]
    public float DequantDot3Avx2()
    {
        fixed (byte* pComp = _compressed3)
        fixed (float* pQuery = _rotatedQuery, pCentroids = _centroids3)
            return TurboQuantOps.DequantDot3Avx2(pComp, pQuery, pCentroids, Dim);
    }

    [Benchmark(Description = "DequantDot 4-bit scalar d=128")]
    public float DequantDot4Scalar() => TurboQuantOps.DequantDot(_compressed4, _rotatedQuery, _centroids4, 4, Dim);

    [Benchmark(Description = "DequantDot 4-bit AVX2 d=128")]
    public float DequantDot4Avx2()
    {
        fixed (byte* pComp = _compressed4)
        fixed (float* pQuery = _rotatedQuery, pCentroids = _centroids4)
            return TurboQuantOps.DequantDot4Avx2(pComp, pQuery, pCentroids, Dim);
    }

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

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama70bCpuBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Llama-70B CPU Decode 10t")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < 10; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama70bHybridBenchmark
{
    private GgufModel _model = null!;
    private Vulkan.VulkanBackend _gpu = null!;
    private HybridForwardPass _hfwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf")
            ?? throw new FileNotFoundException("Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _gpu = new Vulkan.VulkanBackend();

        var hwProfile = HardwareProfile.Detect(_gpu);
        var placement = TierPlanner.Plan(_model, hp, hwProfile);
        _hfwd = new HybridForwardPass(_model, _gpu, hp, placement);

        Console.Error.WriteLine($"[Llama70bHybridBenchmark] {placement.Summary()}");

        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _hfwd.ResetCache();
        for (int i = 0; i < _promptTokens.Count; i++)
            _hfwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Llama-70B Hybrid Decode 10t")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _hfwd.Forward(
            Sampler.Greedy(_hfwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);

        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;

        for (int i = 1; i < 10; i++)
        {
            logits = _hfwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _hfwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama4ScoutCpuBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(BenchmarkHelper.Llama4ScoutModelFile)
            ?? throw new FileNotFoundException($"{BenchmarkHelper.Llama4ScoutModelFile} not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp, maxContextLength: BenchmarkHelper.ScoutBenchmarkContext);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(BenchmarkHelper.LlamaChatPrompt);
    }

    [IterationSetup]
    public void IterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Llama-4-Scout CPU Decode 10t")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);
        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;
        for (int i = 1; i < 10; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama4ScoutCpuTqBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(BenchmarkHelper.Llama4ScoutModelFile)
            ?? throw new FileNotFoundException($"{BenchmarkHelper.Llama4ScoutModelFile} not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp, maxContextLength: BenchmarkHelper.ScoutBenchmarkContext);
        _fwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(BenchmarkHelper.LlamaChatPrompt);
    }

    [IterationSetup]
    public void IterSetup()
    {
        _fwd.TqCache!.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Llama-4-Scout CPU Decode 10t (TQ3 KV)")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(
            Sampler.Greedy(_fwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);
        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;
        for (int i = 1; i < 10; i++)
        {
            logits = _fwd.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd.Dispose();
        _backend.Dispose();
        _model.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama4ScoutHybridBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend? _cpuBackend;
    private ForwardPass? _cpuFwd;
    private Vulkan.VulkanBackend? _gpu;
    private HybridForwardPass? _hfwd;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(BenchmarkHelper.Llama4ScoutModelFile)
            ?? throw new FileNotFoundException($"{BenchmarkHelper.Llama4ScoutModelFile} not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _gpu = new Vulkan.VulkanBackend();

        var hwProfile = HardwareProfile.Detect(_gpu);
        var placement = TierPlanner.Plan(_model, hp, hwProfile, requestedCtxSize: BenchmarkHelper.ScoutBenchmarkContext);
        if (placement.GpuLayers == 0)
        {
            _gpu.Dispose();
            _gpu = null;
            _cpuBackend = new CpuBackend();
            _cpuFwd = new ForwardPass(_model, _cpuBackend, hp, maxContextLength: BenchmarkHelper.ScoutBenchmarkContext);
            Console.Error.WriteLine("[Llama4ScoutHybridBenchmark] Auto fallback to CPU (no GPU-capable MoE layers yet)");
        }
        else
        {
            _hfwd = new HybridForwardPass(_model, _gpu, hp, placement);
            Console.Error.WriteLine($"[Llama4ScoutHybridBenchmark] {placement.Summary()}");
        }

        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(BenchmarkHelper.LlamaChatPrompt);
    }

    [IterationSetup]
    public void IterSetup()
    {
        if (_cpuFwd is not null)
            _cpuFwd.Cache.Reset();
        else
            _hfwd!.ResetCache();

        for (int i = 0; i < _promptTokens.Count; i++)
        {
            if (_cpuFwd is not null)
                _cpuFwd.Forward(_promptTokens[i], i);
            else
                _hfwd!.Forward(_promptTokens[i], i);
        }
    }

    [Benchmark(Description = "Llama-4-Scout Auto Decode 10t")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _cpuFwd is not null
            ? _cpuFwd.Forward(
                Sampler.Greedy(_cpuFwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
                _promptTokens.Count)
            : _hfwd!.Forward(
                Sampler.Greedy(_hfwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
                _promptTokens.Count);
        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;
        for (int i = 1; i < 10; i++)
        {
            logits = _cpuFwd is not null
                ? _cpuFwd.Forward(lastToken, pos++)
                : _hfwd!.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cpuFwd?.Dispose();
        _cpuBackend?.Dispose();
        _hfwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Llama4ScoutHybridTqBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend? _cpuBackend;
    private ForwardPass? _cpuFwd;
    private Vulkan.VulkanBackend? _gpu;
    private HybridForwardPass? _hfwd;
    private IReadOnlyList<int> _promptTokens = null!;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(BenchmarkHelper.Llama4ScoutModelFile)
            ?? throw new FileNotFoundException($"{BenchmarkHelper.Llama4ScoutModelFile} not found");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _gpu = new Vulkan.VulkanBackend();

        var hwProfile = HardwareProfile.Detect(_gpu);
        var placement = TierPlanner.Plan(_model, hp, hwProfile, turboQuant: true, requestedCtxSize: BenchmarkHelper.ScoutBenchmarkContext);
        if (placement.GpuLayers == 0)
        {
            _gpu.Dispose();
            _gpu = null;
            _cpuBackend = new CpuBackend();
            _cpuFwd = new ForwardPass(_model, _cpuBackend, hp, maxContextLength: BenchmarkHelper.ScoutBenchmarkContext);
            _cpuFwd.EnableTurboQuant(fp32WindowSize: 256, bits: 3);
            Console.Error.WriteLine("[Llama4ScoutHybridTqBenchmark] Auto fallback to CPU [TQ3] (no GPU-capable MoE layers yet)");
        }
        else
        {
            _hfwd = new HybridForwardPass(_model, _gpu, hp, placement, enableTq: true);
            Console.Error.WriteLine($"[Llama4ScoutHybridTqBenchmark] {placement.Summary()} [TQ3]");
        }

        _promptTokens = GgufTokenizer.FromGgufModel(_model).Encode(BenchmarkHelper.LlamaChatPrompt);
    }

    [IterationSetup]
    public void IterSetup()
    {
        if (_cpuFwd is not null)
            _cpuFwd.TqCache!.Reset();
        else
            _hfwd!.ResetCache();

        for (int i = 0; i < _promptTokens.Count; i++)
        {
            if (_cpuFwd is not null)
                _cpuFwd.Forward(_promptTokens[i], i);
            else
                _hfwd!.Forward(_promptTokens[i], i);
        }
    }

    [Benchmark(Description = "Llama-4-Scout Auto Decode 10t (TQ3 KV)")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _cpuFwd is not null
            ? _cpuFwd.Forward(
                Sampler.Greedy(_cpuFwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
                _promptTokens.Count)
            : _hfwd!.Forward(
                Sampler.Greedy(_hfwd.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
                _promptTokens.Count);
        int lastToken = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;
        for (int i = 1; i < 10; i++)
        {
            logits = _cpuFwd is not null
                ? _cpuFwd.Forward(lastToken, pos++)
                : _hfwd!.Forward(lastToken, pos++);
            lastToken = Sampler.Greedy(logits);
        }

        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _cpuFwd?.Dispose();
        _cpuBackend?.Dispose();
        _hfwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public unsafe class Llama4ScoutMoeMicroBenchmarks
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private ScoutTensorRef _routerWeight;
    private ScoutTensorRef _sharedGate;
    private ScoutTensorRef _sharedUp;
    private ScoutTensorRef _sharedDown;
    private ScoutTensorRef _expertGate;
    private ScoutTensorRef _expertUp;
    private ScoutTensorRef _expertDown;

    private float* _normBuf;
    private float* _routerLogits;
    private float* _sharedGateBuf;
    private float* _sharedUpBuf;
    private float* _sharedOutBuf;
    private float* _expertGateBuf;
    private float* _expertUpBuf;
    private float* _expertOutBuf;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(BenchmarkHelper.Llama4ScoutModelFile)
            ?? throw new FileNotFoundException($"{BenchmarkHelper.Llama4ScoutModelFile} not found");

        _model = GgufModel.Open(path);
        _hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        if (!_hp.IsMoE)
            throw new InvalidOperationException("Llama 4 Scout microbenchmarks require an MoE model.");

        _routerWeight = ResolveTensor("blk.0.ffn_gate_inp.weight");
        _expertGate = ResolveTensor("blk.0.ffn_gate_exps.weight");
        _expertUp = ResolveTensor("blk.0.ffn_up_exps.weight");
        _expertDown = ResolveTensor("blk.0.ffn_down_exps.weight");
        if (_hp.HasSharedExpert)
        {
            _sharedGate = ResolveTensor("blk.0.ffn_gate_shexp.weight");
            _sharedUp = ResolveTensor("blk.0.ffn_up_shexp.weight");
            _sharedDown = ResolveTensor("blk.0.ffn_down_shexp.weight");
        }

        _normBuf = Alloc(_hp.EmbeddingDim);
        _routerLogits = Alloc(_hp.NumExperts);
        _sharedGateBuf = Alloc(_hp.ExpertIntermediateDim);
        _sharedUpBuf = Alloc(_hp.ExpertIntermediateDim);
        _sharedOutBuf = Alloc(_hp.EmbeddingDim);
        _expertGateBuf = Alloc(_hp.ExpertIntermediateDim);
        _expertUpBuf = Alloc(_hp.ExpertIntermediateDim);
        _expertOutBuf = Alloc(_hp.EmbeddingDim);

        var rng = new Random(42);
        for (int i = 0; i < _hp.EmbeddingDim; i++)
            _normBuf[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    [Benchmark(Description = "Llama-4-Scout MoE Router+TopK")]
    public int RouterTopK()
    {
        SimdKernels.MatVec(_routerLogits, _routerWeight.DataPtr, _normBuf, _hp.NumExperts, _hp.EmbeddingDim, _routerWeight.DType);
        SimdKernels.SoftmaxInPlace(_routerLogits, _hp.NumExperts);

        Span<int> selectedExperts = stackalloc int[_hp.NumActiveExperts];
        Span<float> expertWeights = stackalloc float[_hp.NumActiveExperts];
        SelectTopK(_routerLogits, _hp.NumExperts, _hp.NumActiveExperts, selectedExperts, expertWeights);
        return selectedExperts[0];
    }

    [Benchmark(Description = "Llama-4-Scout MoE FFN layer")]
    public float MoeFfnLayer()
    {
        SimdKernels.MatVec(_routerLogits, _routerWeight.DataPtr, _normBuf, _hp.NumExperts, _hp.EmbeddingDim, _routerWeight.DType);
        SimdKernels.SoftmaxInPlace(_routerLogits, _hp.NumExperts);

        Span<int> selectedExperts = stackalloc int[_hp.NumActiveExperts];
        Span<float> expertWeights = stackalloc float[_hp.NumActiveExperts];
        SelectTopK(_routerLogits, _hp.NumExperts, _hp.NumActiveExperts, selectedExperts, expertWeights);

        new Span<float>(_expertOutBuf, _hp.EmbeddingDim).Clear();

        if (_hp.HasSharedExpert)
        {
            SimdKernels.MatVec(_sharedGateBuf, _sharedGate.DataPtr, _normBuf, _hp.ExpertIntermediateDim, _hp.EmbeddingDim, _sharedGate.DType);
            SimdKernels.MatVec(_sharedUpBuf, _sharedUp.DataPtr, _normBuf, _hp.ExpertIntermediateDim, _hp.EmbeddingDim, _sharedUp.DType);
            SimdKernels.SiLuMul(_sharedGateBuf, _sharedUpBuf, _hp.ExpertIntermediateDim);
            SimdKernels.MatVec(_sharedOutBuf, _sharedDown.DataPtr, _sharedGateBuf, _hp.EmbeddingDim, _hp.ExpertIntermediateDim, _sharedDown.DType);
        }

        for (int i = 0; i < _hp.NumActiveExperts; i++)
        {
            int expertIdx = selectedExperts[i];
            float expertWeight = expertWeights[i];
            ExpertMatVec(_expertGateBuf, _expertGate, expertIdx, _hp.ExpertIntermediateDim, _hp.EmbeddingDim, _normBuf);
            ExpertMatVec(_expertUpBuf, _expertUp, expertIdx, _hp.ExpertIntermediateDim, _hp.EmbeddingDim, _normBuf);
            SimdKernels.SiLuMul(_expertGateBuf, _expertUpBuf, _hp.ExpertIntermediateDim);
            ExpertMatVecDown(_expertOutBuf, _expertDown, expertIdx, _hp.EmbeddingDim, _hp.ExpertIntermediateDim, _expertGateBuf, expertWeight);
        }

        if (_hp.HasSharedExpert)
            SimdKernels.AddInPlace(_expertOutBuf, _sharedOutBuf, _hp.EmbeddingDim);

        return _expertOutBuf[0];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_normBuf != null) NativeMemory.Free(_normBuf);
        if (_routerLogits != null) NativeMemory.Free(_routerLogits);
        if (_sharedGateBuf != null) NativeMemory.Free(_sharedGateBuf);
        if (_sharedUpBuf != null) NativeMemory.Free(_sharedUpBuf);
        if (_sharedOutBuf != null) NativeMemory.Free(_sharedOutBuf);
        if (_expertGateBuf != null) NativeMemory.Free(_expertGateBuf);
        if (_expertUpBuf != null) NativeMemory.Free(_expertUpBuf);
        if (_expertOutBuf != null) NativeMemory.Free(_expertOutBuf);
        _model.Dispose();
    }

    private ScoutTensorRef ResolveTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new ScoutTensorRef(info.Name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private static void ExpertMatVec(float* output, in ScoutTensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    private static void ExpertMatVecDown(float* output, in ScoutTensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;

        float* temp = Alloc(rows);
        try
        {
            SimdKernels.MatVec(temp, expertData, input, rows, cols, packedTensor.DType);
            for (int i = 0; i < rows; i++)
                output[i] += weight * temp[i];
        }
        finally
        {
            NativeMemory.Free(temp);
        }
    }

    private static void SelectTopK(float* logits, int n, int k, Span<int> indices, Span<float> weights)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                {
                    if (indices[j] != i)
                        continue;

                    alreadySelected = true;
                    break;
                }

                if (!alreadySelected && logits[i] > bestVal)
                {
                    bestVal = logits[i];
                    bestIdx = i;
                }
            }

            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        if (k <= 1)
            return;

        float sum = 0;
        for (int i = 0; i < k; i++)
            sum += weights[i];
        if (sum <= 0)
            return;
        for (int i = 0; i < k; i++)
            weights[i] /= sum;
    }

    private readonly unsafe struct ScoutTensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public ScoutTensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name;
            Info = info;
            DType = dtype;
            DataPtr = dataPtr;
        }
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count, (nuint)sizeof(float));
}

/// <summary>
/// Speculative decoding benchmark: SmolLM2-1.7B as target, SmolLM2-360M as draft.
/// Sweeps lookahead k and MinBatchForBlas threshold to find the best configuration.
/// Skipped automatically when either model file is not present in models/.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class SpeculativeDecodingBenchmark
{
    private GgufModel _targetModel = null!;
    private GgufModel _draftModel = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _target = null!;
    private ForwardPass _draft = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private bool _skip;

    private const string TargetModelFile = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
    private const string DraftModelFile  = "SmolLM2-360M-Instruct-Q4_K_M.gguf";

    [Params(4, 8)]
    public int Lookahead { get; set; }

    [Params(1, 4, 8, 32)]
    public int MinBatchBlas { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var targetPath = BenchmarkHelper.FindModelPath(TargetModelFile);
        var draftPath  = BenchmarkHelper.FindModelPath(DraftModelFile);

        if (targetPath is null || draftPath is null)
        {
            Console.Error.WriteLine(
                $"[SpeculativeDecodingBenchmark] Skipping: {TargetModelFile} or {DraftModelFile} not found in models/");
            _skip = true;
            return;
        }

        _backend = new CpuBackend();

        _targetModel = GgufModel.Open(targetPath);
        var targetHp = ModelHyperparams.FromGgufMetadata(_targetModel.Metadata);
        _target = new ForwardPass(_targetModel, _backend, targetHp);

        _draftModel = GgufModel.Open(draftPath);
        var draftHp = ModelHyperparams.FromGgufMetadata(_draftModel.Metadata);
        _draft = new ForwardPass(_draftModel, _backend, draftHp);

        _promptTokens = GgufTokenizer.FromGgufModel(_targetModel).Encode(
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

        Console.Error.WriteLine(
            $"[SpeculativeDecodingBenchmark] Target: {targetHp.NumLayers}L {targetHp.EmbeddingDim}d, " +
            $"Draft: {draftHp.NumLayers}L {draftHp.EmbeddingDim}d");
    }

    [IterationSetup(Targets = [nameof(DecodeBaseline), nameof(DecodeSpeculative)])]
    public void IterSetup()
    {
        if (_skip) return;
        SimdKernels.MinBatchForBlas = MinBatchBlas;
        _target.Cache.Reset();
        _draft.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
        {
            _target.Forward(_promptTokens[i], i);
            _draft.Forward(_promptTokens[i], i);
        }
    }

    [Benchmark(Baseline = true, Description = "Greedy baseline 32t")]
    public int DecodeBaseline()
    {
        if (_skip) return -1;
        ReadOnlySpan<float> logits = _target.Forward(
            Sampler.Greedy(_target.Forward(_promptTokens[^1], _promptTokens.Count - 1)),
            _promptTokens.Count);
        int token = Sampler.Greedy(logits);
        int pos = _promptTokens.Count + 1;
        for (int i = 1; i < 32; i++)
        {
            logits = _target.Forward(token, pos++);
            token = Sampler.Greedy(logits);
        }
        return token;
    }

    [Benchmark(Description = "Speculative 32t")]
    public int DecodeSpeculative()
    {
        if (_skip) return -1;

        var targetLogits = _target.Forward(_promptTokens[^1], _promptTokens.Count - 1);
        var draftLogits  = _draft.Forward(_promptTokens[^1], _promptTokens.Count - 1);

        var spec = new SpeculativeDecoder(_target, _draft, lookahead: Lookahead);
        spec.Initialize(_promptTokens.Count, targetLogits, draftLogits);

        int lastToken = -1;
        spec.Decode(32, [], token => { lastToken = token; });
        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_skip) return;
        _target.Dispose();
        _draft.Dispose();
        _backend.Dispose();
        _targetModel.Dispose();
        _draftModel.Dispose();
    }
}

/// <summary>
/// Micro-benchmark for MatMulBatched at typical transformer weight dimensions.
/// Compares sequential MatVec vs OpenBLAS SGEMM path at different batch sizes.
/// Uses SmolLM2-1.7B FFN gate weight: [8192 × 2048] Q4_K_M.
/// </summary>
[WarmupCount(2)]
[IterationCount(5)]
public unsafe class MatMulBatchedThresholdBenchmark
{
    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private byte* _weights;
    private float* _input;
    private float* _output;
    private int _rows, _cols;
    private DType _dtype;
    private bool _skip;

    private const string ModelFile = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf";

    [Params(1, 2, 4, 8, 16, 32)]
    public int BatchSize { get; set; }

    [Params(false, true)]
    public bool UseBlas { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(ModelFile);
        if (path is null) { _skip = true; return; }

        _model = GgufModel.Open(path);
        _backend = new CpuBackend();

        // Use FFN gate weight of first layer: [intermediate_dim × embed_dim]
        const string tensorName = "blk.0.ffn_gate.weight";
        var info = _model.FindTensor(tensorName);
        if (info is null) { _skip = true; return; }

        _weights = _model.GetTensorDataPtr(info.Value);
        _dtype   = info.Value.DType;
        _rows    = (int)info.Value.Dimensions[0];
        _cols    = (int)info.Value.Dimensions[1];

        _input  = (float*)NativeMemory.AllocZeroed((nuint)(32 * _cols * sizeof(float)));
        _output = (float*)NativeMemory.AllocZeroed((nuint)(32 * _rows * sizeof(float)));

        // Fill input with small values
        var rng = new Random(42);
        for (int i = 0; i < 32 * _cols; i++)
            _input[i] = (float)(rng.NextDouble() * 0.1);

        Console.Error.WriteLine($"[MatMulBatched] {tensorName}: [{_rows}×{_cols}] {_dtype}");
    }

    [IterationSetup]
    public void IterSetup()
    {
        if (_skip) return;
        SimdKernels.MinBatchForBlas = UseBlas ? 1 : int.MaxValue;
    }

    [Benchmark]
    public float MatMulBatch()
    {
        if (_skip) return 0f;
        SimdKernels.MatMulBatched(_output, _weights, _input, BatchSize, _rows, _cols, _dtype);
        return _output[0];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_skip) return;
        if (_input  != null) NativeMemory.Free(_input);
        if (_output != null) NativeMemory.Free(_output);
        _backend.Dispose();
        _model.Dispose();
    }
}

internal static class BenchmarkHelper
{
    public const string Llama4ScoutModelFile = "Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf";
    public const int ScoutBenchmarkContext = 2048;
    public const string LlamaChatPrompt =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    public static string? FindModelPath(string filename)
    {
        // Conventional out-of-tree location for large (>20 GB) GGUFs on this host.
        if (OperatingSystem.IsWindows() && File.Exists($@"E:\models\{filename}"))
            return $@"E:\models\{filename}";

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate))
                return candidate;

            var parent = Directory.GetParent(dir);
            if (parent == null)
                break;

            dir = parent.FullName;
        }

        return null;
    }
}

internal static class BenchmarkPromptHelper
{
    /// <summary>
    /// Gemma 4 raw-completion prompt for parity vs llama.cpp `--no-conversation`:
    /// prepend the BOS token id, then encode the user text. The tokenizer's
    /// Encode() does not auto-prepend BOS; the CLI's SHARPI_RAW_PROMPT path
    /// inlines the same step.
    /// </summary>
    public static IReadOnlyList<int> BuildGemma4PromptTokens(GgufModel model, string text)
    {
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        int bosId = 2;
        if (model.Metadata.TryGetValue("tokenizer.ggml.bos_token_id", out var bosObj))
            bosId = Convert.ToInt32(bosObj);
        var encoded = tokenizer.Encode(text);
        var result = new int[encoded.Count + 1];
        result[0] = bosId;
        for (int i = 0; i < encoded.Count; i++) result[i + 1] = encoded[i];
        return result;
    }
}

// ── Gemma 4 E4B Q8 ────────────────────────────────────────────────────────────
// Phase 9: smoke-bench so per-release t/s comparisons against llama.cpp have a
// reference row. The model is 8.2 GB Q8_0; the CUDA path requires fitting all
// 42 layers in VRAM (use a small context to leave headroom for KV).

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Gemma4E4BCpuBenchmarks
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private GgufModel _model = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;

    [Params(1, 32)]
    public int TokenCount { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(ModelFile)
            ?? throw new FileNotFoundException($"{ModelFile} not found (drop it in models/ or E:\\models\\).");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _backend = new CpuBackend();
        _fwd = new ForwardPass(_model, _backend, hp);
        // Gemma 4's GGUF carries a BOS token; mirror llama.cpp `--no-conversation`
        // raw-completion mode used for the parity baseline.
        _promptTokens = BenchmarkPromptHelper.BuildGemma4PromptTokens(_model, "The capital of France is");
    }

    [IterationSetup(Target = nameof(DecodeTokens))]
    public void DecodeIterSetup()
    {
        _fwd.Cache.Reset();
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [Benchmark(Description = "Gemma 4 E4B Q8 CPU Decode N tokens")]
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

    [Benchmark(Description = "Gemma 4 E4B Q8 CPU Prefill batched")]
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
}

[MemoryDiagnoser]
[WarmupCount(1)]
[IterationCount(3)]
public class Gemma4E4BCudaDecodeBenchmark
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private GgufModel _model = null!;
    private Cuda.CudaBackend _gpu = null!;
    private CudaForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

    [GlobalSetup]
    public void Setup()
    {
        var path = BenchmarkHelper.FindModelPath(ModelFile)
            ?? throw new FileNotFoundException($"{ModelFile} not found (drop it in models/ or E:\\models\\).");

        _model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(_model.Metadata, _model);
        _gpu = Cuda.CudaBackend.Create();
        // 512-token context keeps the full 42-layer Gemma 4 E4B Q8 model in VRAM
        // on a 12 GB card without spilling to CudaHybridForwardPass.
        _fwd = new CudaForwardPass(_model, _gpu, hp, maxContextLength: 512);
        _promptTokens = BenchmarkPromptHelper.BuildGemma4PromptTokens(_model, "The capital of France is");
    }

    [IterationSetup]
    public void IterSetup()
    {
        _fwd.ResetCache();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _fwd.Forward(_promptTokens[i], i);
        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    [Benchmark(Description = "Gemma 4 E4B Q8 CUDA Decode 32 tokens")]
    public int Decode()
    {
        ReadOnlySpan<float> logits = _fwd.Forward(_lastToken, _decodePos++);
        int lastToken = Sampler.Greedy(logits);
        for (int i = 1; i < 32; i++)
        {
            logits = _fwd.Forward(lastToken, _decodePos++);
            lastToken = Sampler.Greedy(logits);
        }
        return lastToken;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _fwd?.Dispose();
        _gpu?.Dispose();
        _model?.Dispose();
    }
}
