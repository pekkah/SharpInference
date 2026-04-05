using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

public sealed class DebugForwardPass
{
    private static string? FindModelPath(string filename = "SmolLM2-1.7B-Instruct-Q4_K_M.gguf")
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindQwen3Path() => FindModelPath("Qwen3-8B-Q4_K_M.gguf");
    private static string? FindQwen3CoderPath() => FindModelPath("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf");

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

        // Prefill with batched method (uses OpenBLAS GEMM when available)
        var sw = System.Diagnostics.Stopwatch.StartNew();
        ReadOnlySpan<float> logits = fwd.Prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Prefill (batched): {tokens.Count} tokens in {prefillMs:F0}ms ({tokens.Count / (prefillMs / 1000):F1} t/s)");

        // Generate 30 tokens with timing
        sw.Restart();
        var generated = new List<int>();
        for (int i = 0; i < 30; i++)
        {
            int next = Engine.Sampler.Greedy(logits);
            generated.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Decode: {generated.Count} tokens in {decodeMs:F0}ms ({generated.Count / (decodeMs / 1000):F1} t/s)");
        Console.WriteLine($"Output: {tokenizer.Decode(generated)}");
    }

    [Fact]
    public void Qwen3_ParsesHyperparams()
    {
        var path = FindQwen3Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);

        Assert.Equal(151936, hp.VocabSize);
        Assert.Equal(4096, hp.EmbeddingDim);
        Assert.Equal(36, hp.NumLayers);
        Assert.Equal(32, hp.NumHeads);
        Assert.Equal(8, hp.NumKvHeads);
        Assert.Equal(12288, hp.IntermediateDim);
        Assert.Equal(1_000_000f, hp.RopeTheta);
        Assert.True(hp.HasQkNorm);
        Assert.False(hp.HasAttnBias);
    }

    [Fact]
    public void Qwen3_ListLayer0TensorNames()
    {
        var path = FindQwen3Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        foreach (var t in model.Tensors.Where(t => t.Name.StartsWith("blk.0.") || !t.Name.StartsWith("blk.")))
            Console.WriteLine($"{t.Name}: [{string.Join(",", t.Dimensions.Take(t.NDimensions))}] {t.DType}");

        // Verify Qwen3-specific tensors exist
        Assert.NotNull(model.FindTensor("blk.0.attn_q_norm.weight"));
        Assert.NotNull(model.FindTensor("blk.0.attn_k_norm.weight"));
    }

    [Fact]
    public void Qwen3_CpuGeneration()
    {
        var path = FindQwen3Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        var tokens = tokenizer.Encode(prompt);
        Console.WriteLine($"Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}");

        // Prefill
        var sw = System.Diagnostics.Stopwatch.StartNew();
        ReadOnlySpan<float> logits = fwd.Prefill(tokens);
        var prefillMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Prefill: {tokens.Count} tokens in {prefillMs:F0}ms ({tokens.Count / (prefillMs / 1000):F1} t/s)");

        // Generate 10 tokens
        sw.Restart();
        var generated = new List<int>();
        for (int i = 0; i < 10; i++)
        {
            int next = Engine.Sampler.Greedy(logits);
            generated.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);
        }
        var decodeMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"Decode: {generated.Count} tokens in {decodeMs:F0}ms ({generated.Count / (decodeMs / 1000):F1} t/s)");
        Console.WriteLine($"Output: {tokenizer.Decode(generated)}");

        // Should produce non-empty, non-garbage output
        Assert.True(generated.Count == 10);
        Assert.True(generated.Any(t => t != generated[0]), "All tokens identical — likely broken");
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

    [Fact]
    public void Qwen3Coder_ListLayer0TensorNames()
    {
        var path = FindQwen3CoderPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        foreach (var t in model.Tensors.Where(t => t.Name.StartsWith("blk.0.") || !t.Name.StartsWith("blk.")))
            Console.WriteLine($"{t.Name}: [{string.Join(",", t.Dimensions.Take(t.NDimensions))}] {t.DType}");

        // Verify MoE-specific tensors exist
        Assert.NotNull(model.FindTensor("blk.0.ffn_gate_exps.weight"));
        Assert.NotNull(model.FindTensor("blk.0.ffn_gate_inp.weight"));
    }

    [Fact]
    public void Qwen3Coder_ParsesHyperparams()
    {
        var path = FindQwen3CoderPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        Console.WriteLine($"IsMoE={hp.IsMoE}, NumExperts={hp.NumExperts}, NumActive={hp.NumActiveExperts}");
        Console.WriteLine($"ExpertIntermediateDim={hp.ExpertIntermediateDim}, IntermediateDim={hp.IntermediateDim}");
        Console.WriteLine($"EmbDim={hp.EmbeddingDim}, HeadDim={hp.HeadDim}, NumHeads={hp.NumHeads}, NumKvHeads={hp.NumKvHeads}");
        Console.WriteLine($"HasQkNorm={hp.HasQkNorm}, UseL2QkNorm={hp.UseL2QkNorm}, RopeTheta={hp.RopeTheta}");

        Assert.True(hp.IsMoE);
        Assert.Equal(128, hp.NumExperts);
        Assert.Equal(8, hp.NumActiveExperts);
        Assert.Equal(768, hp.ExpertIntermediateDim);
        Assert.Equal(2048, hp.EmbeddingDim);
        Assert.Equal(128, hp.HeadDim);
        Assert.Equal(32, hp.NumHeads);
        Assert.Equal(4, hp.NumKvHeads);
        Assert.True(hp.HasQkNorm);
        Assert.False(hp.UseL2QkNorm);
    }

    [Fact]
    public void Qwen3Coder_CpuFirstToken()
    {
        var path = FindQwen3CoderPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        var tokens = tokenizer.Encode(prompt);
        Console.WriteLine($"Prompt tokens ({tokens.Count}): {string.Join(", ", tokens)}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        ReadOnlySpan<float> logits = fwd.Prefill(tokens);
        Console.WriteLine($"Prefill: {tokens.Count} tokens in {sw.Elapsed.TotalMilliseconds:F0}ms");

        var logitsArr = logits.ToArray();
        var top10 = Enumerable.Range(0, logitsArr.Length).OrderByDescending(j => logitsArr[j]).Take(10).ToArray();
        Console.WriteLine("Top-10 logits after prefill:");
        foreach (var idx in top10)
        {
            string decoded = tokenizer.Decode([idx]);
            Console.WriteLine($"  token {idx} ('{decoded}') = {logitsArr[idx]:F4}");
        }

        // First generated token
        int firstToken = Engine.Sampler.Greedy(logits);
        Console.WriteLine($"First token: {firstToken} ('{tokenizer.Decode([firstToken])}')");

        // Generate 10 tokens
        sw.Restart();
        var generated = new List<int> { firstToken };
        var curLogits = fwd.Forward(firstToken, tokens.Count);
        for (int i = 1; i < 10; i++)
        {
            int next = Engine.Sampler.Greedy(curLogits);
            generated.Add(next);
            curLogits = fwd.Forward(next, tokens.Count + i);
        }
        Console.WriteLine($"Decode: {generated.Count} tokens in {sw.Elapsed.TotalMilliseconds:F0}ms");
        Console.WriteLine($"Output: '{tokenizer.Decode(generated)}'");
        Console.WriteLine($"Token IDs: {string.Join(", ", generated)}");

        // The model should generate something reasonable (not all same token)
        Assert.True(generated.Count == 10);
        Assert.True(generated.Any(t => t != generated[0]), "All tokens identical — likely broken");
    }
}
