using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

public sealed class DebugForwardPass
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ListLayer0TensorNames()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        foreach (var t in model.Tensors.Where(t => t.Name.StartsWith("blk.0.") || !t.Name.StartsWith("blk.")))
            Console.WriteLine($"{t.Name}: [{string.Join(",", t.Dimensions.Take(t.NDimensions))}] {t.DType}");
    }

    [Fact]
    public void VerifyEmbeddingLookup()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var info = model.FindTensor("token_embd.weight")!.Value;
        var rawData = model.GetTensorData(info);

        // Dequantize embedding table
        int totalElements = (int)info.ElementCount;
        var floats = new float[totalElements];
        Dequantize.ToFloat32(rawData, floats, info.DType, totalElements);

        // Token 1 (BOS) embedding - first 10 values
        int embDim = 2048;
        Console.WriteLine("Token 1 embedding (first 10):");
        for (int i = 0; i < 10; i++)
            Console.Write($"{floats[1 * embDim + i]:F4} ");
        Console.WriteLine();

        // Check norm of embedding
        float norm = 0;
        for (int i = 0; i < embDim; i++)
            norm += floats[1 * embDim + i] * floats[1 * embDim + i];
        norm = MathF.Sqrt(norm);
        Console.WriteLine($"Token 1 embedding L2 norm: {norm:F4}");
    }

    [Fact]
    public void DebugGeneration()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        // Simple prompt: just "Hi"
        var prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        var tokens = tokenizer.Encode(prompt);
        Console.WriteLine($"Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}");

        // Prefill
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
        {
            logits = fwd.Forward(tokens[i], i);
            if (i == 0)
            {
                int top = Engine.Sampler.Greedy(logits);
                Console.WriteLine($"After token 0 ({tokens[0]}): top prediction = {top} ({tokenizer.Decode([top])})");
            }
        }

        // Generate 15 tokens (enough for "Hello! How can I assist you today?")
        var generated = new List<int>();
        Console.Write("Generated: ");
        for (int i = 0; i < 15; i++)
        {
            int next = Engine.Sampler.Greedy(logits);
            generated.Add(next);
            Console.Write($"[{next}:{tokenizer.Decode([next])}]");
            logits = fwd.Forward(next, tokens.Count + i);
        }
        Console.WriteLine();
        Console.WriteLine($"Full text: {tokenizer.Decode(generated)}");
    }

    [Fact]
    public void VerifyRmsNorm()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        // Get embedding for token 1
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        var embData = model.GetTensorData(embInfo);
        var embFloats = new float[(int)embInfo.ElementCount];
        Dequantize.ToFloat32(embData, embFloats, embInfo.DType, embInfo.ElementCount);

        var hidden = embFloats.AsSpan().Slice(1 * hp.EmbeddingDim, hp.EmbeddingDim).ToArray();

        // Get norm weight
        var normInfo = model.FindTensor("blk.0.attn_norm.weight")!.Value;
        var normData = model.GetTensorData(normInfo);
        var normWeight = new float[hp.EmbeddingDim];
        Dequantize.ToFloat32(normData, normWeight, normInfo.DType, normInfo.ElementCount);

        // Apply RmsNorm
        float sumSq = 0;
        for (int i = 0; i < hp.EmbeddingDim; i++)
            sumSq += hidden[i] * hidden[i];
        float rms = MathF.Sqrt(sumSq / hp.EmbeddingDim + hp.RmsNormEps);
        float scale = 1.0f / rms;

        var normed = new float[hp.EmbeddingDim];
        for (int i = 0; i < hp.EmbeddingDim; i++)
            normed[i] = hidden[i] * scale * normWeight[i];

        Console.WriteLine($"Pre-norm hidden (first 5): {string.Join(" ", hidden.Take(5).Select(v => $"{v:F4}"))}");
        Console.WriteLine($"Norm weight (first 5): {string.Join(" ", normWeight.Take(5).Select(v => $"{v:F4}"))}");
        Console.WriteLine($"RMS value: {rms:F4}");
        Console.WriteLine($"Post-norm (first 5): {string.Join(" ", normed.Take(5).Select(v => $"{v:F4}"))}");

        // The normed values should be reasonable (not all zeros, not huge)
        Assert.True(normed.Any(v => MathF.Abs(v) > 0.001f), "Normed output should not be all near-zero");
    }
}
