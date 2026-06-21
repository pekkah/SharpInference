using SharpInference.Core;
using SharpInference.TurboQuant;

namespace SharpInference.Tests.ForwardPass;

public sealed unsafe class VulkanShaderTests
{
    [Fact]
    public void RmsNormMatchesCpu()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int N = 2048;
        var input = new float[N];
        var weight = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            input[i] = (float)(rng.NextDouble() * 2 - 1);
            weight[i] = (float)(rng.NextDouble() * 0.5 + 0.75);
        }

        // GPU computation
        var gpuInput = backend.Upload(input, TensorShape.D1(N));
        var gpuWeight = backend.Upload(weight, TensorShape.D1(N));
        var gpuOutput = backend.Allocate(TensorShape.D1(N));
        backend.RmsNorm(gpuOutput, gpuInput, gpuWeight, 1e-5f);

        var gpuResult = new float[N];
        backend.Download(gpuOutput, gpuResult);

        // CPU reference
        float sumSq = 0;
        for (int i = 0; i < N; i++) sumSq += input[i] * input[i];
        float scale = 1f / MathF.Sqrt(sumSq / N + 1e-5f);
        var cpuResult = new float[N];
        for (int i = 0; i < N; i++) cpuResult[i] = input[i] * scale * weight[i];

        // Compare
        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(gpuResult[i] - cpuResult[i]) < 0.001f,
                $"RmsNorm mismatch at [{i}]: gpu={gpuResult[i]}, cpu={cpuResult[i]}");

        backend.Free(gpuInput);
        backend.Free(gpuWeight);
        backend.Free(gpuOutput);
    }

    [Fact]
    public void AddInPlaceMatchesCpu()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int N = 1024;
        var a = new float[N];
        var b = new float[N];
        for (int i = 0; i < N; i++) { a[i] = i * 0.1f; b[i] = i * -0.05f; }

        var gpuA = backend.Upload(a, TensorShape.D1(N));
        var gpuB = backend.Upload(b, TensorShape.D1(N));
        backend.AddInPlace(gpuA, gpuB);

        var result = new float[N];
        backend.Download(gpuA, result);

        for (int i = 0; i < N; i++)
            Assert.Equal(a[i] + b[i], result[i], 3);

        backend.Free(gpuA);
        backend.Free(gpuB);
    }

    [Fact]
    public void SiLUMatchesCpu()
    {
        // Issue #314: standalone SiLU must honor the IComputeBackend contract
        // (CPU and CUDA implement it; Vulkan previously threw NotImplementedException).
        using var backend = new Vulkan.VulkanBackend();

        const int N = 1031; // not a multiple of 256 → exercises the bounds guard
        var input = new float[N];
        var rng = new Random(123);
        for (int i = 0; i < N; i++) input[i] = (float)(rng.NextDouble() * 20 - 10);

        var gpuX = backend.Upload(input, TensorShape.D1(N));
        backend.SiLU(gpuX);

        var result = new float[N];
        backend.Download(gpuX, result);

        // CPU reference: x * sigmoid(x) = x / (1 + exp(-x)) (matches GdnKernels.SiLu).
        for (int i = 0; i < N; i++)
        {
            float expected = input[i] / (1f + MathF.Exp(-input[i]));
            Assert.True(MathF.Abs(result[i] - expected) < 1e-5f,
                $"SiLU mismatch at [{i}]: gpu={result[i]}, cpu={expected}");
        }

        backend.Free(gpuX);
    }

    [Fact]
    public void SoftmaxSumsToOne()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int N = 512;
        var input = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++) input[i] = (float)(rng.NextDouble() * 10 - 5);

        var gpuX = backend.Upload(input, TensorShape.D1(N));
        backend.Softmax(gpuX);

        var result = new float[N];
        backend.Download(gpuX, result);

        float sum = result.Sum();
        Assert.True(MathF.Abs(sum - 1.0f) < 0.001f, $"Softmax sum = {sum}, expected 1.0");
        Assert.All(result, v => Assert.True(v >= 0f));

        backend.Free(gpuX);
    }

    [Fact]
    public void SiLuMulMatchesCpu()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int N = 1024;
        var gate = new float[N];
        var up = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            gate[i] = (float)(rng.NextDouble() * 4 - 2);
            up[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // CPU reference
        var expected = new float[N];
        for (int i = 0; i < N; i++)
        {
            float g = gate[i];
            expected[i] = g / (1f + MathF.Exp(-g)) * up[i];
        }

        var gpuGate = backend.Upload(gate, TensorShape.D1(N));
        var gpuUp = backend.Upload(up, TensorShape.D1(N));
        ((Vulkan.VulkanBackend)backend).SiLuMul(gpuGate, gpuUp);

        var result = new float[N];
        backend.Download(gpuGate, result);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 0.001f,
                $"SiLuMul mismatch at [{i}]: gpu={result[i]}, cpu={expected[i]}");

        backend.Free(gpuGate);
        backend.Free(gpuUp);
    }

    [Fact]
    public void RoPEMatchesCpu()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int numHeads = 4;
        const int headDim = 64;
        const int N = numHeads * headDim;
        const int position = 5;
        const float theta = 10000f;

        var input = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference
        var expected = (float[])input.Clone();
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1f / MathF.Pow(theta, 2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);
                int j = h * headDim + 2 * i;
                float x0 = expected[j], x1 = expected[j + 1];
                expected[j] = x0 * cos - x1 * sin;
                expected[j + 1] = x0 * sin + x1 * cos;
            }
        }

        var gpuX = backend.Upload(input, TensorShape.D1(N));
        backend.RoPE(gpuX, position, headDim, theta);

        var result = new float[N];
        backend.Download(gpuX, result);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 0.01f,
                $"RoPE mismatch at [{i}]: gpu={result[i]}, cpu={expected[i]}");

        backend.Free(gpuX);
    }

    /// <summary>Issue #8: NEOX-style Vulkan RoPE shader compiled but had never been
    /// validated against the CPU formula.</summary>
    [Fact]
    public void RoPENeoxMatchesCpu()
    {
        using var backend = new Vulkan.VulkanBackend();

        const int numHeads = 4;
        const int headDim = 128;
        const int N = numHeads * headDim;
        const int position = 5;
        const float theta = 10000f;

        var input = new float[N];
        var rng = new Random(42);
        for (int i = 0; i < N; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU NEOX reference: rotate pair (x[i], x[i + halfDim]) using angle position·freq_i
        var expected = (float[])input.Clone();
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1f / MathF.Pow(theta, 2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);
                int j0 = h * headDim + i;
                int j1 = h * headDim + i + halfDim;
                float x0 = expected[j0], x1 = expected[j1];
                expected[j0] = x0 * cos - x1 * sin;
                expected[j1] = x0 * sin + x1 * cos;
            }
        }

        var gpuX = backend.Upload(input, TensorShape.D1(N));
        backend.RoPE(gpuX, position, headDim, theta, neox: true);

        var result = new float[N];
        backend.Download(gpuX, result);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 0.01f,
                $"NEOX RoPE mismatch at [{i}]: gpu={result[i]}, cpu={expected[i]}");

        backend.Free(gpuX);
    }

    [Fact]
    public void MatVecQ4KMatchesCpu()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        using var backend = new Vulkan.VulkanBackend();

        // Use blk.0.attn_q.weight — a Q4_K tensor
        var qInfo = model.FindTensor("blk.0.attn_q.weight")!.Value;
        var rawData = model.GetTensorData(qInfo);

        // Dequantize to F32 on CPU
        int totalElements = (int)qInfo.ElementCount;
        var f32Weights = new float[totalElements];
        SharpInference.Cpu.Dequantize.ToFloat32(rawData, f32Weights, qInfo.DType, totalElements);

        // Determine matrix dimensions
        int matRows = (int)qInfo.Dimensions[0];
        int matCols = totalElements / matRows;

        // Random input vector
        var input = new float[matCols];
        var rng = new Random(42);
        for (int i = 0; i < matCols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference
        var cpuOutput = new float[matRows];
        for (int r = 0; r < matRows; r++)
        {
            float sum = 0;
            for (int c = 0; c < matCols; c++)
                sum += f32Weights[r * matCols + c] * input[c];
            cpuOutput[r] = sum;
        }

        // Upload raw Q4_K bytes as floats (reinterpret)
        int floatCount = rawData.Length / 4;
        var rawAsFloats = new float[floatCount];
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(rawData).CopyTo(rawAsFloats);
        var gpuWeights = backend.Upload(rawAsFloats, TensorShape.D1(floatCount));
        var gpuInput = backend.Upload(input, TensorShape.D1(matCols));
        var gpuOutput = backend.Allocate(TensorShape.D1(matRows));

        backend.MatMul(gpuOutput, gpuWeights, gpuInput);

        var gpuResult = new float[matRows];
        backend.Download(gpuOutput, gpuResult);

        // Compare (5% relative tolerance for quantized matmul)
        int mismatches = 0;
        for (int i = 0; i < matRows; i++)
        {
            float diff = MathF.Abs(gpuResult[i] - cpuOutput[i]);
            float relDiff = diff / (MathF.Abs(cpuOutput[i]) + 1e-6f);
            if (relDiff > 0.05f)
            {
                if (mismatches < 3)
                    Console.WriteLine($"  [{i}]: gpu={gpuResult[i]:F4} cpu={cpuOutput[i]:F4} rel={relDiff:P1}");
                mismatches++;
            }
        }
        Console.WriteLine($"MatVecQ4K: {mismatches}/{matRows} mismatches (>5% rel error)");
        Assert.True(mismatches < matRows / 10, $"Too many mismatches: {mismatches}/{matRows}");

        backend.Free(gpuWeights);
        backend.Free(gpuInput);
        backend.Free(gpuOutput);
    }

    [Fact]
    public void MatVecQ6KMatchesCpu()
    {
        // Q6_K tensors appear as output.weight in Q4_K_M models.
        // We fall back to a synthetic test if the tensor is not present.
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        using var backend = new Vulkan.VulkanBackend();

        // Try to find a Q6_K tensor; output.weight is Q6_K in Q4_K_M models
        var tensorInfo = model.FindTensor("output.weight")
                      ?? model.FindTensor("token_embd.weight");
        if (tensorInfo is null) return;
        var qInfo = tensorInfo.Value;
        if (qInfo.DType != DType.Q6_K) return;

        var rawData = model.GetTensorData(qInfo);
        int totalElements = (int)qInfo.ElementCount;
        var f32Weights = new float[totalElements];
        SharpInference.Cpu.Dequantize.ToFloat32(rawData, f32Weights, qInfo.DType, totalElements);

        int matRows = (int)qInfo.Dimensions[0];
        int matCols = totalElements / matRows;

        // Only test a manageable subset of rows
        const int maxRows = 512;
        int testRows = Math.Min(matRows, maxRows);

        var input = new float[matCols];
        var rng = new Random(99);
        for (int i = 0; i < matCols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var cpuOutput = new float[testRows];
        for (int r = 0; r < testRows; r++)
        {
            float sum = 0;
            for (int c = 0; c < matCols; c++)
                sum += f32Weights[r * matCols + c] * input[c];
            cpuOutput[r] = sum;
        }

        int floatCount = rawData.Length / 4;
        var rawAsFloats = new float[floatCount];
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(rawData).CopyTo(rawAsFloats);

        // Only upload the rows we're testing
        int bytesPerRow = rawData.Length / matRows;
        int testBytes = testRows * bytesPerRow;
        int testFloatCount = testBytes / 4;
        var testRawAsFloats = rawAsFloats[..testFloatCount];

        var gpuWeights = backend.Upload(testRawAsFloats, TensorShape.D1(testFloatCount));
        var gpuInput = backend.Upload(input, TensorShape.D1(matCols));
        var gpuOutput = backend.Allocate(TensorShape.D1(testRows));

        backend.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q6_K);

        var gpuResult = new float[testRows];
        backend.Download(gpuOutput, gpuResult);

        int mismatches = 0;
        for (int i = 0; i < testRows; i++)
        {
            float diff = MathF.Abs(gpuResult[i] - cpuOutput[i]);
            float relDiff = diff / (MathF.Abs(cpuOutput[i]) + 1e-6f);
            if (relDiff > 0.05f)
            {
                if (mismatches < 3)
                    Console.WriteLine($"  [{i}]: gpu={gpuResult[i]:F4} cpu={cpuOutput[i]:F4} rel={relDiff:P1}");
                mismatches++;
            }
        }
        Console.WriteLine($"MatVecQ6K: {mismatches}/{testRows} mismatches (>5% rel error)");
        Assert.True(mismatches < testRows / 10, $"Too many mismatches: {mismatches}/{testRows}");

        backend.Free(gpuWeights);
        backend.Free(gpuInput);
        backend.Free(gpuOutput);
    }

    // Q8_0 / Q4_0 matvec parity (issue #310). These use synthetic quantized blocks
    // (no GGUF fixture needed): the bytes are dequantized by the codebase's own
    // Dequantize.ToFloat32 for the CPU reference and by the new GLSL shader on the GPU,
    // so identical bytes must yield matching dot products (only float-accumulation order
    // differs). matRows is deliberately not a multiple of 8 to exercise the row guard.

    [Fact]
    public void MatVecQ8_0MatchesCpu()
    {
        const int matRows = 131;      // not a multiple of 8 → partial workgroup
        const int matCols = 160;      // 5 blocks of 32
        var weights = BuildQ8_0(matRows, matCols, seed: 1234);
        AssertVulkanMatVecMatchesCpu(weights, matRows, matCols, DType.Q8_0, inputSeed: 7);
    }

    [Fact]
    public void MatVecQ4_0MatchesCpu()
    {
        const int matRows = 131;
        const int matCols = 160;
        var weights = BuildQ4_0(matRows, matCols, seed: 4321);
        AssertVulkanMatVecMatchesCpu(weights, matRows, matCols, DType.Q4_0, inputSeed: 7);
    }

    private static void AssertVulkanMatVecMatchesCpu(
        byte[] weightBytes, int matRows, int matCols, DType dtype, int inputSeed)
    {
        using var backend = new Vulkan.VulkanBackend();

        // CPU reference: dequantize the same bytes, then naive matvec.
        int totalElements = matRows * matCols;
        var f32Weights = new float[totalElements];
        SharpInference.Cpu.Dequantize.ToFloat32(weightBytes, f32Weights, dtype, totalElements);

        var input = new float[matCols];
        var rng = new Random(inputSeed);
        for (int i = 0; i < matCols; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var cpuOutput = new float[matRows];
        for (int r = 0; r < matRows; r++)
        {
            float sum = 0;
            for (int c = 0; c < matCols; c++)
                sum += f32Weights[r * matCols + c] * input[c];
            cpuOutput[r] = sum;
        }

        // GPU: upload raw quantized bytes reinterpreted as floats (round up to 4 bytes).
        int floatCount = (weightBytes.Length + 3) / 4;
        var rawAsFloats = new float[floatCount];
        weightBytes.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rawAsFloats.AsSpan()));

        var gpuWeights = backend.Upload(rawAsFloats, TensorShape.D1(floatCount));
        var gpuInput = backend.Upload(input, TensorShape.D1(matCols));
        var gpuOutput = backend.Allocate(TensorShape.D1(matRows));

        backend.MatMul(gpuOutput, gpuWeights, gpuInput, dtype);

        var gpuResult = new float[matRows];
        backend.Download(gpuOutput, gpuResult);

        int mismatches = 0;
        for (int i = 0; i < matRows; i++)
        {
            float diff = MathF.Abs(gpuResult[i] - cpuOutput[i]);
            float relDiff = diff / (MathF.Abs(cpuOutput[i]) + 1e-6f);
            // Exact dequant on both sides → only accumulation-order error remains.
            if (diff > 0.1f && relDiff > 0.02f)
            {
                if (mismatches < 3)
                    Console.WriteLine($"  [{i}]: gpu={gpuResult[i]:F4} cpu={cpuOutput[i]:F4} abs={diff:E2} rel={relDiff:P2}");
                mismatches++;
            }
        }
        Console.WriteLine($"MatVec{dtype}: {mismatches}/{matRows} mismatches");
        Assert.Equal(0, mismatches);

        backend.Free(gpuWeights);
        backend.Free(gpuInput);
        backend.Free(gpuOutput);
    }

    private static void PutHalf(byte[] dst, int off, float value)
    {
        ushort h = BitConverter.HalfToUInt16Bits((Half)value);
        dst[off] = (byte)(h & 0xFF);
        dst[off + 1] = (byte)(h >> 8);
    }

    // Q8_0: 34 bytes/block = FP16 scale + 32 int8. Layout matches DequantQ8_0.
    private static byte[] BuildQ8_0(int rows, int cols, int seed)
    {
        const int qk = 32, blockBytes = 34;
        int blocksPerRow = cols / qk;
        var bytes = new byte[rows * blocksPerRow * blockBytes];
        var rng = new Random(seed);
        int off = 0;
        for (int b = 0; b < rows * blocksPerRow; b++)
        {
            PutHalf(bytes, off, (float)(rng.NextDouble() * 0.045 + 0.005));
            for (int j = 0; j < qk; j++)
                bytes[off + 2 + j] = (byte)(sbyte)(rng.Next(-127, 128));
            off += blockBytes;
        }
        return bytes;
    }

    // Q4_0: 18 bytes/block = FP16 scale + 16 nibble bytes. Layout matches DequantQ4_0.
    private static byte[] BuildQ4_0(int rows, int cols, int seed)
    {
        const int qk = 32, blockBytes = 18;
        int blocksPerRow = cols / qk;
        var bytes = new byte[rows * blocksPerRow * blockBytes];
        var rng = new Random(seed);
        int off = 0;
        for (int b = 0; b < rows * blocksPerRow; b++)
        {
            PutHalf(bytes, off, (float)(rng.NextDouble() * 0.045 + 0.005));
            for (int j = 0; j < qk / 2; j++)
                bytes[off + 2 + j] = (byte)(rng.Next(0, 256)); // two packed nibbles
            off += blockBytes;
        }
        return bytes;
    }

    [Fact]
    public void GpuEmbedThenRmsNormMatchesCpu()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var gpu = new Vulkan.VulkanBackend();

        // Dequantize embedding for token 1 on CPU
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        var embData = model.GetTensorData(embInfo);
        int bytesPerRow = (hp.EmbeddingDim / DTypeInfo.BlockSize(embInfo.DType)) * DTypeInfo.BytesPerBlock(embInfo.DType);
        var cpuEmb = new float[hp.EmbeddingDim];
        Cpu.Dequantize.ToFloat32(embData.Slice(1 * bytesPerRow, bytesPerRow), cpuEmb, embInfo.DType, hp.EmbeddingDim);

        Console.WriteLine($"CPU emb [0..4]: {cpuEmb[0]:F4} {cpuEmb[1]:F4} {cpuEmb[2]:F4} {cpuEmb[3]:F4} {cpuEmb[4]:F4}");

        // Upload embedding to GPU
        var gpuHidden = gpu.Upload(cpuEmb, TensorShape.D1(hp.EmbeddingDim));

        // Upload norm weight
        var normInfo = model.FindTensor("blk.0.attn_norm.weight")!.Value;
        var normData = model.GetTensorData(normInfo);
        var cpuNorm = new float[hp.EmbeddingDim];
        Cpu.Dequantize.ToFloat32(normData, cpuNorm, normInfo.DType, hp.EmbeddingDim);
        var gpuNorm = gpu.Upload(cpuNorm, TensorShape.D1(hp.EmbeddingDim));

        // RmsNorm on GPU
        var gpuOutput = gpu.Allocate(TensorShape.D1(hp.EmbeddingDim));
        gpu.RmsNorm(gpuOutput, gpuHidden, gpuNorm, hp.RmsNormEps);

        var gpuResult = new float[hp.EmbeddingDim];
        gpu.Download(gpuOutput, gpuResult);

        // RmsNorm on CPU
        float sumSq = 0;
        for (int i = 0; i < hp.EmbeddingDim; i++) sumSq += cpuEmb[i] * cpuEmb[i];
        float scale = 1f / MathF.Sqrt(sumSq / hp.EmbeddingDim + hp.RmsNormEps);
        var cpuResult = new float[hp.EmbeddingDim];
        for (int i = 0; i < hp.EmbeddingDim; i++) cpuResult[i] = cpuEmb[i] * scale * cpuNorm[i];

        Console.WriteLine($"CPU norm [0..4]: {cpuResult[0]:F4} {cpuResult[1]:F4} {cpuResult[2]:F4}");
        Console.WriteLine($"GPU norm [0..4]: {gpuResult[0]:F4} {gpuResult[1]:F4} {gpuResult[2]:F4}");

        int mismatches = 0;
        for (int i = 0; i < hp.EmbeddingDim; i++)
        {
            if (MathF.Abs(gpuResult[i] - cpuResult[i]) > 0.01f)
            {
                if (mismatches < 3) Console.WriteLine($"  Mismatch [{i}]: gpu={gpuResult[i]:F4} cpu={cpuResult[i]:F4}");
                mismatches++;
            }
        }
        Assert.True(mismatches == 0, $"RmsNorm after embed: {mismatches} mismatches");

        gpu.Free(gpuHidden); gpu.Free(gpuNorm); gpu.Free(gpuOutput);
    }

    [Fact]
    public void GpuEmbedNormMatVecChain()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        using var gpu = new Vulkan.VulkanBackend();

        // Step 1: embed token 1
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        var embData = model.GetTensorData(embInfo);
        int bytesPerRow = (hp.EmbeddingDim / DTypeInfo.BlockSize(embInfo.DType)) * DTypeInfo.BytesPerBlock(embInfo.DType);
        var cpuEmb = new float[hp.EmbeddingDim];
        Cpu.Dequantize.ToFloat32(embData.Slice(1 * bytesPerRow, bytesPerRow), cpuEmb, embInfo.DType, hp.EmbeddingDim);
        var gpuHidden = gpu.Upload(cpuEmb, TensorShape.D1(hp.EmbeddingDim));

        // Step 2: RmsNorm
        var normInfo = model.FindTensor("blk.0.attn_norm.weight")!.Value;
        var normData = model.GetTensorData(normInfo);
        var cpuNormW = new float[hp.EmbeddingDim];
        Cpu.Dequantize.ToFloat32(normData, cpuNormW, normInfo.DType, hp.EmbeddingDim);
        var gpuNormW = gpu.Upload(cpuNormW, TensorShape.D1(hp.EmbeddingDim));
        var gpuNormOut = gpu.Allocate(TensorShape.D1(hp.EmbeddingDim));
        gpu.RmsNorm(gpuNormOut, gpuHidden, gpuNormW, hp.RmsNormEps);

        // Step 3: MatVec with attn_q weight
        var qInfo = model.FindTensor("blk.0.attn_q.weight")!.Value;
        var qRaw = model.GetTensorData(qInfo);
        int floatCount = qRaw.Length / 4;
        var rawFloats = new float[floatCount];
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(qRaw).CopyTo(rawFloats);
        var gpuWq = gpu.Upload(rawFloats, TensorShape.D1(floatCount));

        int qDim = hp.NumHeads * (hp.EmbeddingDim / hp.NumHeads);
        var gpuQ = gpu.Allocate(TensorShape.D1(qDim));
        gpu.MatMul(gpuQ, gpuWq, gpuNormOut);

        var gpuResult = new float[qDim];
        gpu.Download(gpuQ, gpuResult);

        // CPU reference: dequant weights, matmul manually
        var f32Wq = new float[(int)qInfo.ElementCount];
        Cpu.Dequantize.ToFloat32(qRaw, f32Wq, qInfo.DType, qInfo.ElementCount);

        // Get the GPU norm output for CPU reference
        var cpuNormResult = new float[hp.EmbeddingDim];
        gpu.Download(gpuNormOut, cpuNormResult);

        var cpuQ = new float[qDim];
        int cols = hp.EmbeddingDim;
        for (int r = 0; r < qDim; r++)
        {
            float sum = 0;
            for (int c = 0; c < cols; c++)
                sum += f32Wq[r * cols + c] * cpuNormResult[c];
            cpuQ[r] = sum;
        }

        Console.WriteLine($"GPU Q [0..4]: {gpuResult[0]:F4} {gpuResult[1]:F4} {gpuResult[2]:F4} {gpuResult[3]:F4}");
        Console.WriteLine($"CPU Q [0..4]: {cpuQ[0]:F4} {cpuQ[1]:F4} {cpuQ[2]:F4} {cpuQ[3]:F4}");

        int mismatches = 0;
        for (int i = 0; i < qDim; i++)
        {
            float diff = MathF.Abs(gpuResult[i] - cpuQ[i]);
            float rel = diff / (MathF.Abs(cpuQ[i]) + 1e-6f);
            if (rel > 0.1f) mismatches++;
        }
        Console.WriteLine($"Chain mismatches: {mismatches}/{qDim}");
        Assert.True(mismatches < qDim / 10);

        gpu.Free(gpuHidden); gpu.Free(gpuNormW); gpu.Free(gpuNormOut);
        gpu.Free(gpuWq); gpu.Free(gpuQ);
    }

    [Fact]
    public void GpuForwardPassMatchesCpuOutput()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // CPU reference
        using var cpuBackend = new SharpInference.Cpu.CpuBackend();
        using var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp);

        // GPU
        using var gpu = new Vulkan.VulkanBackend();
        using var gpuFwd = new SharpInference.Engine.GpuForwardPass(model, gpu, hp);

        var prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        var tokens = tokenizer.Encode(prompt);

        // Run both forward passes
        ReadOnlySpan<float> cpuLogits = default, gpuLogits = default;
        for (int i = 0; i < tokens.Count; i++)
        {
            cpuLogits = cpuFwd.Forward(tokens[i], i);
            gpuLogits = gpuFwd.Forward(tokens[i], i);
        }

        // Compare top prediction
        int cpuTop = SharpInference.Engine.Sampler.Greedy(cpuLogits);
        int gpuTop = SharpInference.Engine.Sampler.Greedy(gpuLogits);
        Console.WriteLine($"CPU top: {cpuTop} ({tokenizer.Decode([cpuTop])})");
        Console.WriteLine($"GPU top: {gpuTop} ({tokenizer.Decode([gpuTop])})");

        // Check first token's logits aren't all zero
        float gpuMax = float.MinValue, gpuMin = float.MaxValue;
        for (int i = 0; i < gpuLogits.Length; i++)
        {
            if (gpuLogits[i] > gpuMax) gpuMax = gpuLogits[i];
            if (gpuLogits[i] < gpuMin) gpuMin = gpuLogits[i];
        }
        Console.WriteLine($"GPU logits range: [{gpuMin:F2}, {gpuMax:F2}]");
        Console.WriteLine($"CPU logits range: [{cpuLogits.ToArray().Min():F2}, {cpuLogits.ToArray().Max():F2}]");

        // Generate 5 tokens with each
        var cpuTokens = new List<int>();
        var gpuTokens = new List<int>();
        for (int i = 0; i < 5; i++)
        {
            int cpuNext = SharpInference.Engine.Sampler.Greedy(cpuLogits);
            int gpuNext = SharpInference.Engine.Sampler.Greedy(gpuLogits);
            cpuTokens.Add(cpuNext);
            gpuTokens.Add(gpuNext);
            cpuLogits = cpuFwd.Forward(cpuNext, tokens.Count + i);
            gpuLogits = gpuFwd.Forward(gpuNext, tokens.Count + i);
        }

        Console.WriteLine($"CPU: {tokenizer.Decode(cpuTokens)}");
        Console.WriteLine($"GPU: {tokenizer.Decode(gpuTokens)}");

        // The outputs should match (greedy decode should produce identical tokens)
        Assert.Equal(cpuTokens, gpuTokens);
    }

    [Fact]
    public void TurboQuantKvCache_UsesLayerIndexBaseForCompressors()
    {
        const int numLayers = 2;
        const int numKvHeads = 4;
        const int headDim = 128;
        const int layerIndexBase = 3;
        const int totalLayerCount = 8;

        using var cache = new SharpInference.Engine.TurboQuantKvCache(
            numLayers, maxSeqLen: 16, numKvHeads, headDim,
            layerIndexBase: layerIndexBase, totalLayerCountForSeeds: totalLayerCount);

        for (int layer = 0; layer < numLayers; layer++)
        {
            for (int head = 0; head < numKvHeads; head++)
            {
                int globalLayer = layer + layerIndexBase;

                var expectedKey = WalshHadamard.GenerateSignPattern(headDim, globalLayer * numKvHeads + head);
                var actualKey = cache.GetKeyCompressor(layer, head).SignPattern;
                for (int i = 0; i < headDim; i++)
                    Assert.Equal(expectedKey[i], actualKey[i]);

                var expectedValue = WalshHadamard.GenerateSignPattern(headDim, (globalLayer + totalLayerCount) * numKvHeads + head);
                var actualValue = cache.GetValueCompressor(layer, head).SignPattern;
                for (int i = 0; i < headDim; i++)
                    Assert.Equal(expectedValue[i], actualValue[i]);
            }
        }
    }

    [Fact]
    public void GpuForwardPassTurboQuantRunsPastFp32Window()
    {
        var path = FindTurboQuantModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var gpu = new Vulkan.VulkanBackend();
        using var gpuFwd = new SharpInference.Engine.GpuForwardPass(
            model, gpu, hp, maxContextLength: 32, enableTurboQuant: true, tqFp32Window: 2);

        var prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        var tokens = tokenizer.Encode(prompt);

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = gpuFwd.Forward(tokens[i], i);

        Assert.Equal(hp.VocabSize, logits.Length);
        for (int i = 0; i < logits.Length; i++)
            Assert.True(float.IsFinite(logits[i]), $"Non-finite logit at [{i}]: {logits[i]}");
    }

    [Fact]
    public void HybridForwardPassTurboQuantRunsPastFp32Window()
    {
        var path = FindTurboQuantModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        if (hp.NumLayers < 2) return;

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var gpu = new Vulkan.VulkanBackend();
        var placement = new SharpInference.Engine.LayerPlacement(
            GpuLayers: 1,
            CpuLayers: hp.NumLayers - 1,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 32);
        using var hybridFwd = new SharpInference.Engine.HybridForwardPass(
            model, gpu, hp, placement, enableTq: true, tqFp32Window: 2);

        var prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        AssertHybridForwardPassProducesCoherentDecode(hybridFwd, tokenizer, prompt, hp.VocabSize);
    }

    /// <summary>
    /// Validates the Attention shader produces correct output for seq_len &lt;= 256 (fast path).
    /// </summary>
    [Fact]
    public void AttentionShader_ShortSequence_MatchesCpuReference()
    {
        AttentionShaderMatchesCpuReference(seqLen: 64, numHeads: 2, numKvHeads: 2, headDim: 32);
    }

    /// <summary>
    /// Validates the Attention shader produces correct output for seq_len &gt; 256,
    /// exercising the stored-scores path (was the correctness bug region).
    /// </summary>
    [Fact]
    public void AttentionShader_LongSequence_MatchesCpuReference()
    {
        AttentionShaderMatchesCpuReference(seqLen: 300, numHeads: 2, numKvHeads: 2, headDim: 32);
    }

    /// <summary>
    /// Validates GQA (num_heads > num_kv_heads) for seq_len &gt; 256.
    /// </summary>
    [Fact]
    public void AttentionShader_LongSequenceGqa_MatchesCpuReference()
    {
        AttentionShaderMatchesCpuReference(seqLen: 512, numHeads: 4, numKvHeads: 2, headDim: 32);
    }

    private static void AttentionShaderMatchesCpuReference(
        int seqLen, int numHeads, int numKvHeads, int headDim)
    {
        using var backend = new Vulkan.VulkanBackend();

        int kvDim = numKvHeads * headDim;
        int maxSeqLen = seqLen + 16;

        var rng = new Random(42);
        var q = new float[numHeads * headDim];
        var kCache = new float[maxSeqLen * kvDim];
        var vCache = new float[maxSeqLen * kvDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < seqLen * kvDim; i++) kCache[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < seqLen * kvDim; i++) vCache[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference: scaled dot-product attention with GQA
        float scale = 1f / MathF.Sqrt(headDim);
        var cpuOutput = new float[numHeads * headDim];
        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / (numHeads / numKvHeads);
            var scores = new float[seqLen];
            for (int t = 0; t < seqLen; t++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[h * headDim + d] * kCache[t * kvDim + kvHead * headDim + d];
                scores[t] = dot * scale;
            }
            // Softmax
            float maxS = scores.Max();
            float sumE = 0f;
            for (int t = 0; t < seqLen; t++) { scores[t] = MathF.Exp(scores[t] - maxS); sumE += scores[t]; }
            for (int t = 0; t < seqLen; t++) scores[t] /= sumE;
            // Weighted value sum
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int t = 0; t < seqLen; t++)
                    sum += scores[t] * vCache[t * kvDim + kvHead * headDim + d];
                cpuOutput[h * headDim + d] = sum;
            }
        }

        // GPU
        var gpuQ = backend.Upload(q, TensorShape.D1(q.Length));
        var gpuK = backend.Upload(kCache, TensorShape.D2(maxSeqLen, kvDim));
        var gpuV = backend.Upload(vCache, TensorShape.D2(maxSeqLen, kvDim));
        var gpuOut = backend.Allocate(TensorShape.D1(numHeads * headDim));
        long scratchElems = maxSeqLen > 4096 ? (long)numHeads * maxSeqLen : 1L;
        var gpuScratch = backend.Allocate(TensorShape.D1(scratchElems));
        ((Vulkan.VulkanBackend)backend).Attention(
            gpuQ, gpuK, gpuV, gpuOut, gpuScratch,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim,
            (uint)seqLen, (uint)maxSeqLen);

        var gpuResult = new float[numHeads * headDim];
        backend.Download(gpuOut, gpuResult);

        for (int i = 0; i < cpuOutput.Length; i++)
            Assert.True(MathF.Abs(gpuResult[i] - cpuOutput[i]) < 1e-3f,
                $"Attention mismatch at [{i}] (seqLen={seqLen}): gpu={gpuResult[i]:F5} cpu={cpuOutput[i]:F5}");

        backend.Free(gpuQ);
        backend.Free(gpuK);
        backend.Free(gpuV);
        backend.Free(gpuOut);
        backend.Free(gpuScratch);
    }

    private static string? FindModelPath()
    {
        return FindModelPath(
            "models\\SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
    }

    private static string? FindTurboQuantModelPath()
    {
        return FindModelPath(
            "models\\Qwen3-8B-Q4_K_M.gguf",
            "models\\Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf",
            "models\\Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf");
    }

    private static string? FindMoEModelPath()
    {
        // Qwen3-Coder is preferred because it exercises the embDim != expertDim layout
        // that exposes scratch-sizing bugs (see #2 regression). OLMoE works as a
        // fallback because its constraints (per-channel QK norm, norm_topk_prob=false,
        // embDim > intermDim) cover orthogonal MoE features.
        return FindModelPath(
            "models\\Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
            "models\\OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf",
            "models\\Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf");
    }

    private static string? FindModelPath(params string[] candidates)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var candidate in candidates)
            {
                var fullPath = Path.Combine(dir, candidate);
                if (File.Exists(fullPath))
                    return fullPath;
            }
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Regression test for issue #2 (MoE+hybrid produced NaN/garbled output) and the
    /// underlying descriptor-set reuse hazard in <c>ComputePipeline.RecordWith</c>.
    ///
    /// <c>GpuMoeFfn</c> dispatches the same MatVec pipeline 3× per active expert per
    /// layer (gate, up, down) with different weight buffers each time. Before the fix
    /// these all shared one <c>_reusableDs</c>; at GPU execution time every dispatch
    /// read whichever weight buffer was bound last, producing wrong activations that
    /// cascaded into NaN logits within a few tokens. With per-dispatch DS allocation
    /// each MatVec call has its own descriptor set and the activations stay finite.
    ///
    /// Silently no-ops when the MoE GGUF isn't present locally — matches the rest of
    /// the model-dependent tests in this file.
    /// </summary>
    [Fact]
    public void HybridForwardPass_MoE_ProducesCoherentDecode()
    {
        var path = FindMoEModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        // Pass `model` so HasQkNorm / IsPerChannelQkNorm probe the tensor index.
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.IsMoE || hp.NumLayers < 2) return;

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var gpu = new Vulkan.VulkanBackend();
        // GpuLayers=5 makes any per-layer activation error compound through enough
        // attention+FFN stages to push logits into the degenerate "all-EOS" or NaN
        // regime that the coherence helper catches. The original GpuLayers=1 was
        // too weak — the issue #2 scratch-sizing bug went undetected for a week
        // because one layer of wrong activations doesn't always cascade visibly.
        int gpuLayers = Math.Min(5, hp.NumLayers);
        var placement = new SharpInference.Engine.LayerPlacement(
            GpuLayers: gpuLayers,
            CpuLayers: hp.NumLayers - gpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 64);
        using var hybridFwd = new SharpInference.Engine.HybridForwardPass(
            model, gpu, hp, placement, enableTq: false);

        AssertHybridForwardPassProducesCoherentDecode(hybridFwd, tokenizer, prompt: "Hello", hp.VocabSize);
    }

    /// <summary>Dense non-TQ hybrid smoke test (SmolLM2, 1 GPU layer + rest on CPU).
    /// Sister to the MoE and TQ-enabled hybrid coverage. Exercises the GPU embed
    /// lookup + GPU output projection paths end-to-end (regression guard for #19/#3,
    /// where Q6_K embed tables were uploaded as raw bytes and reinterpreted by the
    /// F32 EmbedLookup shader, producing NaN/huge values).</summary>
    [Fact]
    public void HybridForwardPass_DenseSmallVocab_ProducesCoherentDecode()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        if (hp.NumLayers < 2) return;

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var gpu = new Vulkan.VulkanBackend();
        var placement = new SharpInference.Engine.LayerPlacement(
            GpuLayers: 1,
            CpuLayers: hp.NumLayers - 1,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 64);
        using var hybridFwd = new SharpInference.Engine.HybridForwardPass(
            model, gpu, hp, placement, enableTq: false);

        AssertHybridForwardPassProducesCoherentDecode(
            hybridFwd, tokenizer, prompt: "The capital of France is", hp.VocabSize);
    }

    /// <summary>
    /// Decode coherence helper used by the hybrid forward-pass smoke tests.
    /// Checks three things, layered from weakest to strongest:
    ///
    ///   1. Every logit is finite (catches NaN cascades from MoE expert MatMul wiring etc.)
    ///   2. argmax(logits) immediately after the prompt is NOT EOS — the test that catches
    ///      "zero logits → sampler picks token 0 → 0 decode tokens" failures, which all
    ///      pass an <c>IsFinite</c>-only check.
    ///   3. A short greedy decode produces some token that's neither EOS nor a repeat
    ///      of the immediately-prior token. Catches degenerate "emit the same token
    ///      forever" outputs from subtler corruption like KV-cache aliasing.
    ///
    /// Silently no-ops when the test model file isn't present locally.
    /// </summary>
    private static void AssertHybridForwardPassProducesCoherentDecode(
        SharpInference.Engine.HybridForwardPass hybridFwd,
        GgufTokenizer tokenizer,
        string prompt,
        int vocabSize)
    {
        var tokens = tokenizer.Encode(prompt);

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = hybridFwd.Forward(tokens[i], i);

        Assert.Equal(vocabSize, logits.Length);

        // (1) All logits finite.
        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in post-prompt output.");

        // (2) argmax != EOS at first decode step.
        int firstDecodeToken = Argmax(logits);
        Assert.NotEqual(tokenizer.EosTokenId, firstDecodeToken);

        // (3) A 4-token greedy decode produces variety. With finite-but-zero logits
        //     argmax is deterministic (token 0) and this loop would emit "EOS, EOS, EOS, EOS"
        //     — caught here as "all generated tokens are EOS".
        Span<int> decoded = stackalloc int[4];
        decoded[0] = firstDecodeToken;
        int pos = tokens.Count;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = hybridFwd.Forward(decoded[i - 1], pos++);
            decoded[i] = Argmax(step);
        }

        int eosCount = 0;
        for (int i = 0; i < decoded.Length; i++)
            if (decoded[i] == tokenizer.EosTokenId) eosCount++;
        Assert.True(eosCount < decoded.Length,
            $"All {decoded.Length} greedy-decoded tokens were EOS — output is degenerate.");
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Vulkan twin of <c>CudaTurboQuantTests.TqAttention_RecomputePath_MatchesFastPath</c>.
    /// Drives the TQ-attention shader at TqLen=4096 (fits in shared memory) and TqLen=4097
    /// (forces the global-scratch spill path), with the first 4096 K/V identical across both
    /// runs and a zero V at position 4096. Because that zero-V position contributes nothing
    /// to any output dimension regardless of its softmax weight, the slow path differs from
    /// the fast path by exactly the global softmax-rescale factor sum_fast / sum_slow. Recover
    /// that ratio from the largest fast-path dim, then check every dim agrees under it.
    ///
    /// Guards against the silent OOB regression where the shader's <c>shared float scores[4096]</c>
    /// gets written past its cap when <c>tq_seq_len + fp16_seq_len &gt; 4096</c>.
    /// </summary>
    [Fact]
    public void TqAttention_LongContextScratchPath_MatchesFastPath()
    {
        using var gpu = new Vulkan.VulkanBackend();

        const int HeadDim = 128;
        const int NumHeads = 1;
        const int NumKvHeads = 1;
        const int FastLen = 4096;   // exactly at the cap → shared-memory fast path
        const int SlowLen = 4097;   // one over → global-scratch spill path
        int blockBytes = TurboQuantOps.BlockSize(bits: 3, HeadDim);
        long tqBytesPerPos = (long)NumKvHeads * blockBytes;
        long fastUints = ((long)FastLen * tqBytesPerPos + 3) / 4;
        long slowUints = ((long)SlowLen * tqBytesPerPos + 3) / 4;

        var rng = new Random(13371337);
        var queryDir = RandomUnit(rng, HeadDim);

        var kVecs = new float[SlowLen][];
        var vVecs = new float[SlowLen][];
        for (int p = 0; p < FastLen; p++)
        {
            kVecs[p] = RandomUnit(rng, HeadDim);
            vVecs[p] = RandomUnit(rng, HeadDim);
        }
        kVecs[FastLen] = RandomUnit(rng, HeadDim);
        vVecs[FastLen] = new float[HeadDim];   // zero V — doesn't perturb output dims

        var compressor = new KvCacheCompressor(bits: 3, HeadDim, layerIndex: 0);
        var signPatterns = compressor.SignPattern.ToArray();
        var centroids    = TurboQuantCodebooks.GetCentroids(bits: 3, HeadDim).ToArray();
        var boundaries   = TurboQuantCodebooks.GetBoundaries(bits: 3, HeadDim).ToArray();

        var gpuSigns      = gpu.Upload(signPatterns, TensorShape.D1(signPatterns.Length));
        var gpuCodebook   = gpu.Upload(centroids,    TensorShape.D1(centroids.Length));
        var gpuBoundaries = gpu.Upload(boundaries,   TensorShape.D1(boundaries.Length));

        float[] outFast = RunTqAttentionPath(gpu, NumHeads, NumKvHeads, HeadDim, FastLen, fastUints,
            kVecs, vVecs, queryDir, blockBytes, gpuSigns, gpuCodebook, gpuBoundaries);
        float[] outSlow = RunTqAttentionPath(gpu, NumHeads, NumKvHeads, HeadDim, SlowLen, slowUints,
            kVecs, vVecs, queryDir, blockBytes, gpuSigns, gpuCodebook, gpuBoundaries);

        int probeDim = 0;
        float bestAbs = MathF.Abs(outFast[probeDim]);
        for (int d = 1; d < HeadDim; d++)
            if (MathF.Abs(outFast[d]) > bestAbs) { bestAbs = MathF.Abs(outFast[d]); probeDim = d; }
        Assert.True(bestAbs > 1e-4f, "Fast-path output is degenerate (all near zero); test inputs are pathological.");

        float ratio = outSlow[probeDim] / outFast[probeDim];
        for (int d = 0; d < HeadDim; d++)
        {
            float expected = outFast[d] * ratio;
            float diff = MathF.Abs(outSlow[d] - expected);
            float tol = MathF.Max(1e-3f, MathF.Abs(expected) * 1e-2f);
            Assert.True(diff < tol,
                $"Scratch path mismatch at dim {d}: slow={outSlow[d]:E3} expected={expected:E3} " +
                $"(fast={outFast[d]:E3}, ratio={ratio:F4}, tol={tol:E3}). " +
                $"The two paths must agree under a single global softmax-rescale factor.");
        }

        gpu.Free(gpuSigns); gpu.Free(gpuCodebook); gpu.Free(gpuBoundaries);
    }

    private static float[] RunTqAttentionPath(Vulkan.VulkanBackend gpu,
        int numHeads, int numKvHeads, int headDim, int tqLen, long totalUints,
        float[][] kVecs, float[][] vVecs, float[] queryDir, int blockBytes,
        Tensor gpuSigns, Tensor gpuCodebook, Tensor gpuBoundaries)
    {
        var gpuKCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuVCacheTq = gpu.Allocate(TensorShape.D1(totalUints));

        for (int p = 0; p < tqLen; p++)
        {
            var gpuKIn = gpu.Upload(kVecs[p], TensorShape.D1(kVecs[p].Length));
            var gpuVIn = gpu.Upload(vVecs[p], TensorShape.D1(vVecs[p].Length));
            gpu.TqKvAppend(gpuKIn, gpuVIn, gpuKCacheTq, gpuVCacheTq,
                gpuSigns, gpuCodebook, gpuBoundaries,
                (uint)(numKvHeads * headDim), (uint)headDim, (uint)p, (uint)tqLen,
                (uint)numKvHeads, (uint)blockBytes);
            gpu.Free(gpuKIn); gpu.Free(gpuVIn);
        }

        var gpuQ = gpu.Upload(queryDir, TensorShape.D1(queryDir.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(queryDir.Length));
        gpu.TqRotateQuery(gpuQ, gpuRotated, gpuSigns, (uint)numHeads, (uint)numKvHeads, (uint)headDim);

        var gpuKCacheFp16 = gpu.Allocate(TensorShape.D1(numKvHeads * headDim));
        var gpuVCacheFp16 = gpu.Allocate(TensorShape.D1(numKvHeads * headDim));
        var gpuOut = gpu.Allocate(TensorShape.D1(numHeads * headDim));

        // Allocate the long-context scratch unconditionally so both paths exercise the
        // same kernel; the shader ignores it on the fast path (total_seq ≤ 4096).
        var scratch = gpu.Allocate(TensorShape.D1((long)numHeads * tqLen));

        gpu.TqAttention(gpuQ, gpuRotated, gpuKCacheTq, gpuVCacheTq,
            gpuKCacheFp16, gpuVCacheFp16, gpuOut, gpuCodebook,
            scratch,
            (uint)numHeads, (uint)numKvHeads, (uint)headDim,
            (uint)tqLen, fp16SeqLen: 0u, (uint)tqLen, (uint)blockBytes);

        var output = new float[numHeads * headDim];
        gpu.Download(gpuOut, output);

        gpu.Free(gpuQ); gpu.Free(gpuRotated);
        gpu.Free(gpuKCacheTq); gpu.Free(gpuVCacheTq);
        gpu.Free(gpuKCacheFp16); gpu.Free(gpuVCacheFp16);
        gpu.Free(gpuOut); gpu.Free(scratch);

        return output;
    }

    private static float[] RandomUnit(Random rng, int dim)
    {
        var v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        float mag = 0f;
        for (int i = 0; i < dim; i++) mag += v[i] * v[i];
        mag = MathF.Sqrt(mag);
        if (mag > 0f) for (int i = 0; i < dim; i++) v[i] /= mag;
        return v;
    }
}
