using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for the three NVRTC kernels added to support the qwen35moe GPU
/// attention block: <see cref="CudaBackend.RoPEPartial"/>,
/// <see cref="CudaBackend.SigmoidMulInPlace"/>, and
/// <see cref="CudaBackend.SplitQG"/>. Each test silently no-ops on hosts
/// without CUDA, mirroring <c>CudaTurboQuantTests</c>.
/// </summary>
public sealed unsafe class CudaAttnKernelsTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void RoPEPartial_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Single head, headDim=8, ropeDim=4 (rotate first 4 dims, leave last 4 alone).
        const int NumHeads = 1;
        const int HeadDim = 8;
        const int RopeDim = 4;
        const int Position = 5;
        const float Theta = 10000f;

        var rng = new Random(424242);
        var input = new float[NumHeads * HeadDim];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference using the same per-position cos/sin table the engine uses.
        // BuildRopeTable expects buffers of size [position * (ropeDim/2)] per position.
        int halfRope = RopeDim / 2;
        int ctxLen = Position + 1;
        var cosTab = new float[ctxLen * halfRope];
        var sinTab = new float[ctxLen * halfRope];
        fixed (float* pCos = cosTab)
        fixed (float* pSin = sinTab)
        {
            SimdKernels.BuildRopeTable(pCos, pSin, ctxLen, RopeDim, Theta);
        }

        var expected = (float[])input.Clone();
        fixed (float* pExp = expected)
        fixed (float* pCos = cosTab)
        fixed (float* pSin = sinTab)
        {
            SimdKernels.ApplyRoPECachedNeoxPartial(
                pExp, pCos + Position * halfRope, pSin + Position * halfRope,
                NumHeads, HeadDim, RopeDim);
        }

        var gpuX = gpu.Upload(input, TensorShape.D1(input.Length));
        gpu.RoPEPartial(gpuX, Position, HeadDim, RopeDim, Theta, neox: true);
        gpu.Synchronize();
        var gpuResult = new float[input.Length];
        gpu.Download(gpuX, gpuResult);
        gpu.Free(gpuX);

        for (int i = 0; i < input.Length; i++)
            Assert.True(MathF.Abs(gpuResult[i] - expected[i]) < 1e-5f,
                $"RoPEPartial mismatch at [{i}]: gpu={gpuResult[i]} cpu={expected[i]}");

        // Dims [ropeDim, headDim) must pass through unchanged.
        for (int i = RopeDim; i < HeadDim; i++)
            Assert.True(MathF.Abs(gpuResult[i] - input[i]) < 1e-6f,
                $"RoPEPartial touched non-rope dim [{i}]: gpu={gpuResult[i]} input={input[i]}");
    }

    [Fact]
    public void RoPEPartial_NonNeoxThrows()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var x = gpu.Allocate(TensorShape.D1(8));
        Assert.Throws<ArgumentException>(() =>
            gpu.RoPEPartial(x, position: 0, headDim: 8, ropeDim: 4, ropeTheta: 10000f, neox: false));
        gpu.Free(x);
    }

    [Fact]
    public void SigmoidMulInPlace_MatchesScalarReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 257;   // odd, non-block-aligned size to exercise tail handling
        var rng = new Random(9001);
        var x = new float[N];
        var gate = new float[N];
        for (int i = 0; i < N; i++)
        {
            x[i] = (float)(rng.NextDouble() * 4 - 2);
            gate[i] = (float)(rng.NextDouble() * 6 - 3);
        }

        var expected = new float[N];
        for (int i = 0; i < N; i++)
        {
            float sig = 1.0f / (1.0f + MathF.Exp(-gate[i]));
            expected[i] = x[i] * sig;
        }

        var gpuX = gpu.Upload(x, TensorShape.D1(N));
        var gpuGate = gpu.Upload(gate, TensorShape.D1(N));
        gpu.SigmoidMulInPlace(gpuX, gpuGate);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuX, result);
        gpu.Free(gpuX);
        gpu.Free(gpuGate);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"SigmoidMulInPlace mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    [Fact]
    public void SplitQG_MatchesHandComputed()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 3;
        const int HeadDim = 4;
        // Per head: 4 floats Q followed by 4 floats G. Total = NumHeads * HeadDim * 2.
        var qg = new float[NumHeads * HeadDim * 2];
        for (int i = 0; i < qg.Length; i++) qg[i] = i + 0.5f;

        var expectedQ = new float[NumHeads * HeadDim];
        var expectedG = new float[NumHeads * HeadDim];
        for (int h = 0; h < NumHeads; h++)
        {
            int srcBase = h * HeadDim * 2;
            for (int j = 0; j < HeadDim; j++)
            {
                expectedQ[h * HeadDim + j] = qg[srcBase + j];
                expectedG[h * HeadDim + j] = qg[srcBase + HeadDim + j];
            }
        }

        var gpuQg = gpu.Upload(qg, TensorShape.D1(qg.Length));
        var gpuQ = gpu.Allocate(TensorShape.D1((long)NumHeads * HeadDim));
        var gpuG = gpu.Allocate(TensorShape.D1((long)NumHeads * HeadDim));

        gpu.SplitQG(gpuQ, gpuG, gpuQg, NumHeads, HeadDim);
        gpu.Synchronize();

        var resultQ = new float[NumHeads * HeadDim];
        var resultG = new float[NumHeads * HeadDim];
        gpu.Download(gpuQ, resultQ);
        gpu.Download(gpuG, resultG);

        gpu.Free(gpuQg);
        gpu.Free(gpuQ);
        gpu.Free(gpuG);

        for (int i = 0; i < expectedQ.Length; i++)
        {
            Assert.Equal(expectedQ[i], resultQ[i]);
            Assert.Equal(expectedG[i], resultG[i]);
        }
    }

    [Fact]
    public void ElementwiseMul_MatchesScalarReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 513;
        var rng = new Random(31415);
        var a = new float[N];
        var b = new float[N];
        for (int i = 0; i < N; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2 - 1);
            b[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        var gpuA = gpu.Upload(a, TensorShape.D1(N));
        var gpuB = gpu.Upload(b, TensorShape.D1(N));
        var gpuOut = gpu.Allocate(TensorShape.D1(N));
        gpu.ElementwiseMul(gpuOut, gpuA, gpuB);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuOut, result);
        gpu.Free(gpuA);
        gpu.Free(gpuB);
        gpu.Free(gpuOut);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - a[i] * b[i]) < 1e-6f,
                $"ElementwiseMul mismatch at [{i}]: gpu={result[i]} cpu={a[i] * b[i]}");
    }
}
