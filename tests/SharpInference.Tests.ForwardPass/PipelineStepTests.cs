using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests that verify each pipeline step produces correct results.
/// SIMD kernel tests run without a model; pipeline tests require model files.
/// Run with: dotnet test --filter "FullyQualifiedName~PipelineStepTests"
/// </summary>
public sealed class PipelineStepTests : IDisposable
{
    private readonly List<IntPtr> _allocations = [];

    public unsafe void Dispose()
    {
        foreach (var p in _allocations)
            NativeMemory.AlignedFree((void*)p);
    }

    private unsafe float* AllocFloats(int count)
    {
        var ptr = (float*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
        NativeMemory.Clear(ptr, (nuint)(count * sizeof(float)));
        _allocations.Add((IntPtr)ptr);
        return ptr;
    }

    private unsafe float* AllocFrom(float[] data)
    {
        var ptr = AllocFloats(data.Length);
        data.AsSpan().CopyTo(new Span<float>(ptr, data.Length));
        return ptr;
    }

    // ================================================================
    //  SigmoidInPlace
    // ================================================================

    [Theory]
    [InlineData(new float[] { 0f }, new float[] { 0.5f })]
    [InlineData(new float[] { -10f, -5f, 0f, 5f, 10f },
                new float[] { 0.0000454f, 0.006693f, 0.5f, 0.993307f, 0.999955f })]
    public unsafe void SigmoidInPlace_MatchesReference(float[] input, float[] expected)
    {
        var x = AllocFrom(input);
        SimdKernels.SigmoidInPlace(x, input.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(x[i], expected[i] - 1e-3f, expected[i] + 1e-3f);
    }

    [Fact]
    public unsafe void SigmoidInPlace_LlamaRouterLogits_MatchesPython()
    {
        float[] logits = { -1.2633f, -1.2399f, -0.8720f, -0.8698f, -1.0758f,
                           -1.2337f, -1.7947f, -1.4085f, -1.7304f, -0.8622f,
                           -0.8824f, -0.8447f, -0.9513f, -0.9423f, -1.2873f,
                           -2.2396f };
        float[] expected = new float[16];
        for (int i = 0; i < 16; i++)
            expected[i] = 1.0f / (1.0f + MathF.Exp(-logits[i]));

        var x = AllocFrom(logits);
        SimdKernels.SigmoidInPlace(x, 16);

        for (int i = 0; i < 16; i++)
        {
            float err = MathF.Abs(x[i] - expected[i]);
            Assert.True(err < 1e-3f,
                $"Sigmoid[{i}]: got {x[i]:F6}, expected {expected[i]:F6}, err={err:E2}");
        }
        Assert.InRange(x[11], 0.299f, 0.302f);
    }

    [Fact]
    public unsafe void SigmoidInPlace_LargeArray_MatchesScalar()
    {
        int n = 256;
        float[] data = new float[n];
        float[] reference = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            data[i] = (float)(rng.NextDouble() * 20 - 10);
            reference[i] = 1.0f / (1.0f + MathF.Exp(-data[i]));
        }

        var x = AllocFrom(data);
        SimdKernels.SigmoidInPlace(x, n);

        for (int i = 0; i < n; i++)
        {
            float err = MathF.Abs(x[i] - reference[i]);
            Assert.True(err < 5e-3f,
                $"Sigmoid[{i}] (input={data[i]:F4}): got {x[i]:F6}, expected {reference[i]:F6}, err={err:E2}");
        }
    }

    // ================================================================
    //  SoftmaxInPlace
    // ================================================================

    [Fact]
    public unsafe void SoftmaxInPlace_Uniform_ProducesEqualProbs()
    {
        float[] input = { 1, 1, 1, 1 };
        var x = AllocFrom(input);
        SimdKernels.SoftmaxInPlace(x, 4);
        for (int i = 0; i < 4; i++)
            Assert.InRange(x[i], 0.2499f, 0.2501f);
    }

    [Fact]
    public unsafe void SoftmaxInPlace_SumsToOne()
    {
        float[] input = { 1.5f, -0.3f, 2.7f, 0.0f, -1.2f };
        var x = AllocFrom(input);
        SimdKernels.SoftmaxInPlace(x, 5);

        float sum = 0;
        for (int i = 0; i < 5; i++) sum += x[i];
        Assert.InRange(sum, 0.999f, 1.001f);
    }

    [Fact]
    public unsafe void SoftmaxInPlace_SingleElement_IsOne()
    {
        float[] input = { -0.8447f };
        var x = AllocFrom(input);
        SimdKernels.SoftmaxInPlace(x, 1);
        Assert.InRange(x[0], 0.999f, 1.001f);
    }

    [Fact]
    public unsafe void SoftmaxInPlace_PreservesArgmax()
    {
        float[] input = { -1.2633f, -1.2399f, -0.8720f, -0.8698f, -1.0758f,
                          -1.2337f, -1.7947f, -1.4085f, -1.7304f, -0.8622f,
                          -0.8824f, -0.8447f, -0.9513f, -0.9423f, -1.2873f,
                          -2.2396f };
        var x = AllocFrom(input);
        SimdKernels.SoftmaxInPlace(x, 16);

        int argmax = 0;
        for (int i = 1; i < 16; i++)
            if (x[i] > x[argmax]) argmax = i;
        Assert.Equal(11, argmax);
    }

    // ================================================================
    //  SiLuMul
    // ================================================================

    [Fact]
    public unsafe void SiLuMul_MatchesScalar()
    {
        int n = 64;
        float[] gate = new float[n];
        float[] up = new float[n];
        float[] expected = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
        {
            gate[i] = (float)(rng.NextDouble() * 10 - 5);
            up[i] = (float)(rng.NextDouble() * 10 - 5);
            float sig = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            expected[i] = gate[i] * sig * up[i];
        }

        var g = AllocFrom(gate);
        var u = AllocFrom(up);
        SimdKernels.SiLuMul(g, u, n);

        for (int i = 0; i < n; i++)
        {
            float err = MathF.Abs(g[i] - expected[i]);
            float relErr = expected[i] != 0 ? err / MathF.Abs(expected[i]) : err;
            Assert.True(relErr < 0.01f || err < 1e-3f,
                $"SiLuMul[{i}]: got {g[i]:F6}, expected {expected[i]:F6}, relErr={relErr:P1}");
        }
    }

    [Fact]
    public unsafe void SiLuMul_WeightBeforeFFN_DiffersFromWeightAfter()
    {
        float w = 0.3f;
        float[] gate = { 2.0f, -1.0f, 0.5f, 3.0f, -2.5f, 1.0f, -0.1f, 4.0f };
        float[] up = { 1.0f, 2.0f, 3.0f, -1.0f, 0.5f, -2.0f, 1.5f, -0.5f };

        float[] gateA = (float[])gate.Clone();
        float[] upA = (float[])up.Clone();
        for (int i = 0; i < 8; i++) { gateA[i] *= w; upA[i] *= w; }
        var gA = AllocFrom(gateA);
        var uA = AllocFrom(upA);
        SimdKernels.SiLuMul(gA, uA, 8);

        var gB = AllocFrom(gate);
        var uB = AllocFrom(up);
        SimdKernels.SiLuMul(gB, uB, 8);

        bool anyDifferent = false;
        for (int i = 0; i < 8; i++)
        {
            if (MathF.Abs(gA[i] - w * gB[i]) > 1e-4f) anyDifferent = true;
        }
        Assert.True(anyDifferent, "Weight-before vs weight-after should differ due to SiLU non-linearity");
    }

    // ================================================================
    //  RmsNorm
    // ================================================================

    [Fact]
    public unsafe void RmsNorm_KnownValues()
    {
        float[] input = { 1, 2, 3, 4 };
        float[] weight = { 1, 1, 1, 1 };
        float rms = MathF.Sqrt(7.5f);
        float[] expected = { 1 / rms, 2 / rms, 3 / rms, 4 / rms };

        var inp = AllocFrom(input);
        var w = AllocFrom(weight);
        var outp = AllocFloats(4);
        SimdKernels.RmsNorm(outp, inp, w, 4, 1e-5f);

        for (int i = 0; i < 4; i++)
            Assert.InRange(outp[i], expected[i] - 1e-4f, expected[i] + 1e-4f);
    }

    [Fact]
    public unsafe void PureRmsNorm_ProducesUnitRms()
    {
        int n = 128;
        float[] data = new float[n];
        var rng = new Random(42);
        for (int i = 0; i < n; i++)
            data[i] = (float)(rng.NextDouble() * 10 - 5);

        var inp = AllocFrom(data);
        var outp = AllocFloats(n);
        SimdKernels.PureRmsNorm(outp, inp, n, 1e-5f);

        float sumSq = 0;
        for (int i = 0; i < n; i++) sumSq += outp[i] * outp[i];
        float rms = MathF.Sqrt(sumSq / n);
        Assert.InRange(rms, 0.99f, 1.01f);
    }

    // ================================================================
    //  ScaleInPlace / AddInPlace
    // ================================================================

    [Fact]
    public unsafe void ScaleInPlace_Works()
    {
        float[] data = { 1, 2, 3, 4, 5, 6, 7, 8 };
        var x = AllocFrom(data);
        SimdKernels.ScaleInPlace(x, 0.5f, 8);
        for (int i = 0; i < 8; i++)
            Assert.InRange(x[i], data[i] * 0.5f - 1e-6f, data[i] * 0.5f + 1e-6f);
    }

    [Fact]
    public unsafe void AddInPlace_Works()
    {
        float[] a = { 1, 2, 3, 4 };
        float[] b = { 10, 20, 30, 40 };
        var pa = AllocFrom(a);
        var pb = AllocFrom(b);
        SimdKernels.AddInPlace(pa, pb, 4);
        Assert.Equal(11f, pa[0]);
        Assert.Equal(22f, pa[1]);
        Assert.Equal(33f, pa[2]);
        Assert.Equal(44f, pa[3]);
    }

    // ================================================================
    //  PureRmsNorm output properties
    // ================================================================

    [Theory]
    [InlineData(128)]
    [InlineData(64)]
    [InlineData(256)]
    public unsafe void PureRmsNorm_OutputNorm_IsSqrtDim(int dim)
    {
        float[] data = new float[dim];
        var rng = new Random(42);
        for (int i = 0; i < dim; i++)
            data[i] = (float)(rng.NextDouble() * 10 - 5);

        var inp = AllocFrom(data);
        var outp = AllocFloats(dim);
        SimdKernels.PureRmsNorm(outp, inp, dim, 1e-5f);

        float sumSq = 0;
        for (int i = 0; i < dim; i++) sumSq += outp[i] * outp[i];
        float outputNorm = MathF.Sqrt(sumSq);
        float expectedNorm = MathF.Sqrt(dim);
        Assert.InRange(outputNorm, expectedNorm - 0.1f, expectedNorm + 0.1f);
    }

    // ================================================================
    //  Llama-4 model-dependent tests
    // ================================================================

    private static string? FindModelPath(string filename)
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

    private static string? FindLlama4Path() =>
        FindModelPath("Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf");

    [Fact]
    public void Llama4_Tokenizer_MatchesReference()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        int[] expected = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                          30, 43, 200008, 200005, 140680, 200006, 368];

        var tokens = tokenizer.Encode(
            "<|begin_of_text|><|header_start|>user<|header_end|>\n\n" +
            "What is 2+2?" +
            "<|eot|><|header_start|>assistant<|header_end|>\n\n");

        // Dump actual tokens for diagnosis
        var actualStr = string.Join(", ", tokens);
        Assert.True(tokens.Count == expected.Length,
            $"Token count: got {tokens.Count}, expected {expected.Length}.\n" +
            $"Actual:   [{actualStr}]\n" +
            $"Expected: [{string.Join(", ", expected)}]");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(tokens[i] == expected[i],
                $"Token[{i}]: got {tokens[i]}, expected {expected[i]}");
    }

    // ================================================================
    //  Llama-4 Hyperparameter verification
    // ================================================================

    [Fact]
    public void Llama4_Hyperparams_QkNormEnabled()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.HasQkNorm, "Llama-4 must have QK-norm enabled (L2 pure RMS norm)");
        Assert.True(hp.UseL2QkNorm, "Llama-4 must use L2 (pure) QK-norm, not weighted");
    }

    [Fact]
    public void Llama4_Hyperparams_NoPEStep()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.Equal(4, hp.NoRopeLayerStep);
        // Layers 3,7,11,15,... skip RoPE (NoPE layers)
        Assert.True((3 + 1) % hp.NoRopeLayerStep == 0, "Layer 3 should be NoPE");
        Assert.False((0 + 1) % hp.NoRopeLayerStep == 0, "Layer 0 should use RoPE");
    }

    [Fact]
    public void Llama4_Hyperparams_SigmoidGating()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.UseSigmoidGating, "Llama-4 must use sigmoid gating for MoE router");
        Assert.True(hp.IsMoE, "Llama-4 must be identified as MoE");
        Assert.Equal(16, hp.NumExperts);
        Assert.Equal(1, hp.NumActiveExperts);
    }

    [Fact]
    public void Llama4_Hyperparams_ModelDimensions()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.Equal(5120, hp.EmbeddingDim);
        Assert.Equal(48, hp.NumLayers);
        Assert.Equal(40, hp.NumHeads);
        Assert.Equal(8, hp.NumKvHeads);
        Assert.Equal(128, hp.EmbeddingDim / hp.NumHeads); // headDim
    }

    // ================================================================
    //  Llama-4 Per-position pipeline verification
    // ================================================================

    /// <summary>
    /// Verifies logit norms and top tokens at key positions match llama-cpp-python reference.
    /// Reference values from sigmoid mode (default for Llama-4).
    /// Tolerance is generous to account for Q4_K_M quantization differences.
    /// </summary>
    [Fact]
    public void Llama4_Sigmoid_PerPosition_LogitNormsMatchReference()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] tokens = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        // Reference norms from llama-cpp-python (sigmoid mode, per-token eval)
        // pos: (referenceNorm, tolerancePct)
        var referenceNorms = new (int pos, float norm, float tol)[]
        {
            (0,  3406.48f, 0.10f),  // BOS token
            (1,  1959.60f, 0.10f),  // header_start — first real attention
            (2,  4865.08f, 0.10f),  // "user"
            (16, 5717.30f, 0.10f),  // final \n\n — produces answer
        };

        // Run forward pass per-token
        var norms = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            var logits = (i == 0)
                ? fwd.Forward(tokens[i], i)
                : fwd.Forward(tokens[i], i);
            float norm = 0;
            for (int j = 0; j < logits.Length; j++) norm += logits[j] * logits[j];
            norms[i] = MathF.Sqrt(norm);
        }

        foreach (var (pos, refNorm, tol) in referenceNorms)
        {
            float relErr = MathF.Abs(norms[pos] - refNorm) / refNorm;
            Assert.True(relErr < tol,
                $"pos={pos}: logit norm={norms[pos]:F2}, reference={refNorm:F2}, relErr={relErr:P1} (tol={tol:P0})");
        }
    }

    /// <summary>
    /// Verifies the top predicted tokens at each position match reference.
    /// This catches attention, RoPE, QK-norm, and MoE routing bugs.
    /// </summary>
    [Fact]
    public void Llama4_Sigmoid_PerPosition_TopTokensMatchReference()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] tokens = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        // Reference top-1 tokens from llama-cpp-python (sigmoid mode)
        var referenceTop1 = new (int pos, int expectedArgmax)[]
        {
            (0,  954),    // BOS → predicts common continuation
            (1,  29),     // header_start
            (2,  200008), // "user" → predicts eot
            (16, 30),     // final \n\n → predicts "2" (answer)
        };

        var argmaxes = new int[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            var logits = fwd.Forward(tokens[i], i);
            argmaxes[i] = ArgMax(logits);
        }

        foreach (var (pos, expectedArgmax) in referenceTop1)
        {
            Assert.True(argmaxes[pos] == expectedArgmax,
                $"pos={pos}: argmax={argmaxes[pos]}, expected={expectedArgmax}");
        }
    }

    [Fact]
    public void Llama4_Softmax_FirstToken_MatchesReference()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model) with { UseSigmoidGating = false };
        Assert.False(hp.UseSigmoidGating);
        Assert.True(hp.HasQkNorm, "QK-norm must be enabled for Llama-4");
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] tokens = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        var logits = fwd.Prefill(tokens);
        int argmax = ArgMax(logits);

        var topStr = FormatTop5(logits);
        Assert.True(argmax == 30 || argmax == 954,
            $"Softmax: expected argmax=30 or 954, got {argmax}.\nTOP5: {topStr}");
        Assert.InRange(logits[argmax], 30f, 70f);
    }

    [Fact]
    public void Llama4_Sigmoid_FirstToken_MatchesReference()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.UseSigmoidGating);
        Assert.True(hp.HasQkNorm, "QK-norm must be enabled for Llama-4");
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] tokens = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        var logits = fwd.Prefill(tokens);
        int argmax = ArgMax(logits);

        var topStr = FormatTop5(logits);
        Assert.True(argmax == 30,
            $"Sigmoid: expected argmax=30 ('2'), got {argmax}.\n" +
            $"TOP5: {topStr}\n" +
            $"Logit for [30]={logits[30]:F2} (reference: ~54.12)");
        Assert.InRange(logits[30], 40f, 70f);
    }

    [Fact]
    public void Llama4_Sigmoid_FullSequence_2Plus2()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] prompt = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        // Reference from llama-cpp-python (sigmoid): "2 + 2 = 4<eot>"
        // Q4_K_M quantization may add trailing "." before eot; math content must match
        int[] expectedPrefix = [30, 584, 220, 30, 319, 220, 32]; // "2 + 2 = 4"

        var logits = fwd.Prefill(prompt);
        var generated = new List<int>();

        for (int step = 0; step < 12; step++)
        {
            int argmax = ArgMax(logits);
            generated.Add(argmax);
            if (argmax == 200008) break;
            logits = fwd.Forward(argmax, prompt.Length + step);
        }

        // Verify the math answer prefix
        for (int i = 0; i < expectedPrefix.Length; i++)
        {
            Assert.True(i < generated.Count,
                $"Generated only {generated.Count} tokens, expected at least {expectedPrefix.Length}");
            Assert.True(generated[i] == expectedPrefix[i],
                $"Step {i}: expected token {expectedPrefix[i]}, got {generated[i]}. " +
                $"Generated: [{string.Join(", ", generated)}]");
        }

        // Must eventually produce eot
        Assert.Contains(200008, generated);
    }

    [Fact]
    public void Llama4_Softmax_FullSequence_2Plus2()
    {
        var path = FindLlama4Path();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model) with { UseSigmoidGating = false };
        using var backend = new CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp, maxContextLength: 512);

        int[] prompt = [200000, 200005, 1556, 200006, 368, 3668, 373, 220, 30, 23,
                        30, 43, 200008, 200005, 140680, 200006, 368];

        // Softmax gating (non-default for Llama-4) produces "2 + 2 = 4." then eot
        // The math tokens must match; trailing punctuation may vary from sigmoid mode
        int[] expectedPrefix = [30, 584, 220, 30, 319, 220, 32]; // "2 + 2 = 4"

        var logits = fwd.Prefill(prompt);
        var generated = new List<int>();

        for (int step = 0; step < 12; step++)
        {
            int argmax = ArgMax(logits);
            generated.Add(argmax);
            if (argmax == 200008) break;
            logits = fwd.Forward(argmax, prompt.Length + step);
        }

        // Verify the math answer prefix
        for (int i = 0; i < expectedPrefix.Length; i++)
        {
            Assert.True(i < generated.Count,
                $"Generated only {generated.Count} tokens, expected at least {expectedPrefix.Length}");
            Assert.True(generated[i] == expectedPrefix[i],
                $"Step {i}: expected token {expectedPrefix[i]}, got {generated[i]}. " +
                $"Generated: [{string.Join(", ", generated)}]");
        }

        // Must eventually produce eot
        Assert.Contains(200008, generated);
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static string FormatTop5(ReadOnlySpan<float> logits)
    {
        var top5 = new (int idx, float val)[5];
        for (int i = 0; i < 5; i++) top5[i] = (-1, float.MinValue);
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            for (int j = 0; j < 5; j++)
            {
                if (v > top5[j].val)
                {
                    for (int k = 4; k > j; k--) top5[k] = top5[k - 1];
                    top5[j] = (i, v);
                    break;
                }
            }
        }
        return string.Join(", ", top5.Select(t => $"[{t.idx}]={t.val:F2}"));
    }
}
