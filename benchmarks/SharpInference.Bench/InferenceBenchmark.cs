using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.TurboQuant;

namespace SharpInference.Bench;

// Each benchmark class owns a single model/backend setup so BenchmarkDotNet
// only keeps one large model resident at a time.

[MemoryDiagnoser]
[WarmupCount(2)]
[IterationCount(10)]
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
[IterationCount(5)]
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
[IterationCount(5)]
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
[IterationCount(5)]
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
[IterationCount(5)]
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
[IterationCount(5)]
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
[IterationCount(5)]
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

internal static class BenchmarkHelper
{
    public static string? FindModelPath(string filename)
    {
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
