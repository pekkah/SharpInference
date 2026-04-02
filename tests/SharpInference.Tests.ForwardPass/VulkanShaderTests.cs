using SharpInference.Core;

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
}
