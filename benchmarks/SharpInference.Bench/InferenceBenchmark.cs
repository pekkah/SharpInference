using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

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
        var path = FindModelPath()
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

    [Benchmark(Description = "Decode N tokens")]
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

    [Benchmark(Description = "Prefill 10 sequential")]
    public int PrefillSequential()
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _fwd.Forward(_promptTokens[i], i);
        return Sampler.Greedy(logits);
    }

    [Benchmark(Description = "Prefill 10 batched")]
    public int PrefillBatched()
    {
        var logits = _fwd.Prefill(_promptTokens);
        return Sampler.Greedy(logits);
    }

    // ================================================================
    //  GPU Decode
    // ================================================================

    private Vulkan.VulkanBackend _gpu = null!;
    private Engine.GpuForwardPass _gpuFwd = null!;
    private int _gpuDecodePos;
    private int _gpuLastToken;

    [GlobalSetup(Targets = new[] { nameof(GpuDecodeTokens) })]
    public void GpuSetup()
    {
        var path = FindModelPath()
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

    [Benchmark(Description = "GPU Decode 32 tokens")]
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

    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 10; i++)
        {
            var candidate = Path.Combine(dir, "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
