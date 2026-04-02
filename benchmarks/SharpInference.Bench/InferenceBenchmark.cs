using BenchmarkDotNet.Attributes;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Bench;

[MemoryDiagnoser]
[WarmupCount(2)]
[IterationCount(20)]
public class InferenceBenchmark
{
    private GgufModel _model = null!;
    private ModelHyperparams _hp = null!;
    private GgufTokenizer _tokenizer = null!;
    private CpuBackend _backend = null!;
    private ForwardPass _fwd = null!;
    private IReadOnlyList<int> _promptTokens = null!;
    private int _decodePos;
    private int _lastToken;

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

        // Prefill prompt
        for (int i = 0; i < _promptTokens.Count; i++)
            _fwd.Forward(_promptTokens[i], i);
    }

    [IterationSetup(Target = nameof(DecodeOneToken))]
    public void DecodeIterSetup()
    {
        _fwd.Cache.Reset();
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _fwd.Forward(_promptTokens[i], i);
        _lastToken = Sampler.Greedy(logits);
        _decodePos = _promptTokens.Count;
    }

    /// <summary>
    /// Benchmark a single decode step. To get tokens/sec: 1000 / Mean(ms).
    /// </summary>
    [Benchmark(Description = "Decode 1 token")]
    public int DecodeOneToken()
    {
        var logits = _fwd.Forward(_lastToken, _decodePos++);
        return Sampler.Greedy(logits);
    }

    [IterationSetup(Target = nameof(PrefillPrompt))]
    public void PrefillIterSetup()
    {
        _fwd.Cache.Reset();
    }

    /// <summary>
    /// Benchmark prefill of 10-token prompt. Tokens/sec = 10 * 1000 / Mean(ms).
    /// </summary>
    [Benchmark(Description = "Prefill 10 tokens")]
    public int PrefillPrompt()
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < _promptTokens.Count; i++)
            logits = _fwd.Forward(_promptTokens[i], i);
        return Sampler.Greedy(logits);
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
