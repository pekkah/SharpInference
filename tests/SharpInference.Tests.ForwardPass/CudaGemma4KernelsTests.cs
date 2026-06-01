using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Unit tests for the three NVRTC kernels added in Phase 7 of the Gemma 4
/// implementation: <see cref="CudaBackend.GeluTanhMul"/>,
/// <see cref="CudaBackend.SoftcapInPlace"/>, and
/// <see cref="CudaBackend.AttentionSwa"/>. Each test silently no-ops on
/// hosts without CUDA, mirroring <see cref="CudaAttnKernelsTests"/>.
/// </summary>
public sealed unsafe class CudaGemma4KernelsTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void GeluTanhMul_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 4096;
        var rng = new Random(0xCafe);
        var gate = new float[N];
        var up = new float[N];
        for (int i = 0; i < N; i++)
        {
            gate[i] = (float)(rng.NextDouble() * 6.0 - 3.0);   // gate ~ U[-3, 3]
            up[i] = (float)(rng.NextDouble() * 4.0 - 2.0);     // up ~ U[-2, 2]
        }

        // CPU reference: use the scalar implementation to avoid AVX exp
        // approximation drift influencing the comparison.
        var expected = new float[N];
        fixed (float* pGate = gate)
        fixed (float* pUp = up)
        fixed (float* pOut = expected)
        {
            SimdKernels.GeluTanhMul_Scalar(pGate, pUp, pOut, N);
        }

        var gpuGate = gpu.Upload(gate, TensorShape.D1(N));
        var gpuUp = gpu.Upload(up, TensorShape.D1(N));
        gpu.GeluTanhMul(gpuGate, gpuUp);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuGate, result);
        gpu.Free(gpuGate);
        gpu.Free(gpuUp);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-4f,
                $"GeluTanhMul mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    [Fact]
    public void SoftcapInPlace_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int N = 1024;
        const float Cap = 30.0f;
        var rng = new Random(0xBeef);
        var x = new float[N];
        for (int i = 0; i < N; i++)
            x[i] = (float)(rng.NextDouble() * 100.0 - 50.0);   // ±50, exercises the cap

        var expected = new float[N];
        for (int i = 0; i < N; i++)
            expected[i] = MathF.Tanh(x[i] / Cap) * Cap;

        var gpuX = gpu.Upload(x, TensorShape.D1(N));
        gpu.SoftcapInPlace(gpuX, Cap);
        gpu.Synchronize();
        var result = new float[N];
        gpu.Download(gpuX, result);
        gpu.Free(gpuX);

        for (int i = 0; i < N; i++)
            Assert.True(MathF.Abs(result[i] - expected[i]) < 1e-5f,
                $"SoftcapInPlace mismatch at [{i}]: gpu={result[i]} cpu={expected[i]}");
    }

    /// <summary>
    /// SWA attention with window=8 over a 65-position prefix (position=64).
    /// The kernel must (a) match a manual full-attention computation restricted
    /// to the windowed range [57, 65), and (b) DIFFER from a full-attention
    /// computation over the entire range [0, 65). Both checks together
    /// demonstrate that the windowing is real and that the result is
    /// mathematically correct over the window.
    /// </summary>
    [Fact]
    public void AttentionSwa_WindowedReadsSubsetOfFullAttention()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 4;
        const int NumKvHeads = 4;
        const int HeadDim = 64;
        const int Position = 64;
        const int WindowSize = 8;
        const int MaxSeqLen = 128;
        int seqLen = Position + 1;
        int kvDim = NumKvHeads * HeadDim;

        var rng = new Random(0xC0FFEE);
        var q = new float[NumHeads * HeadDim];
        var kCache = new float[MaxSeqLen * kvDim];
        var vCache = new float[MaxSeqLen * kvDim];
        for (int i = 0; i < q.Length; i++)
            q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < seqLen * kvDim; i++)
        {
            kCache[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vCache[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        int windowStart = seqLen - WindowSize;   // 57
        var expectedWindowed = ReferenceAttention(q, kCache, vCache,
            NumHeads, NumKvHeads, HeadDim, windowStart, seqLen);
        var fullAttention = ReferenceAttention(q, kCache, vCache,
            NumHeads, NumKvHeads, HeadDim, 0, seqLen);

        var gpuQ = gpu.Upload(q, TensorShape.D1(q.Length));
        var gpuK = gpu.Upload(kCache, TensorShape.D1(kCache.Length));
        var gpuV = gpu.Upload(vCache, TensorShape.D1(vCache.Length));
        var gpuOut = gpu.Allocate(TensorShape.D1(NumHeads * HeadDim));
        // Scores scratch: not needed for window=8 (fits in shared) but pass a
        // valid buffer anyway so the kernel arg can be non-null on any backend
        // configuration. Sized [num_heads * max_seq_len].
        var gpuScratch = gpu.Allocate(TensorShape.D1((long)NumHeads * MaxSeqLen));

        gpu.AttentionSwa(gpuQ, gpuK, gpuV, gpuOut, gpuScratch,
            Position, WindowSize, HeadDim, NumHeads, NumKvHeads, MaxSeqLen);
        gpu.Synchronize();

        var result = new float[NumHeads * HeadDim];
        gpu.Download(gpuOut, result);
        gpu.Free(gpuQ);
        gpu.Free(gpuK);
        gpu.Free(gpuV);
        gpu.Free(gpuOut);
        gpu.Free(gpuScratch);

        // (a) GPU result equals CPU reference computed over the windowed range.
        for (int i = 0; i < result.Length; i++)
            Assert.True(MathF.Abs(result[i] - expectedWindowed[i]) < 1e-4f,
                $"AttentionSwa windowed mismatch at [{i}]: gpu={result[i]} cpu={expectedWindowed[i]}");

        // (b) GPU windowed result must differ from full attention over [0, 65).
        // We only need to see a meaningful gap on at least one element — the
        // window discards 57/65 positions, so the V-weighted sums for any
        // non-uniform input cannot agree to 1e-3 across the whole vector.
        float maxDiff = 0f;
        for (int i = 0; i < result.Length; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(result[i] - fullAttention[i]));
        Assert.True(maxDiff > 1e-3f,
            $"AttentionSwa output should differ from full attention but maxDiff={maxDiff}");
    }

    /// <summary>
    /// CPU reference: scaled dot-product attention with GQA over a windowed
    /// range of positions [tStart, tEnd). Layout matches the GPU kernel: Q is
    /// [num_heads * head_dim], K/V cache is [max_seq_len * (num_kv_heads *
    /// head_dim)] indexed as (t * num_kv_heads + kv_head, d), output is
    /// [num_heads * head_dim].
    /// </summary>
    private static float[] ReferenceAttention(
        float[] q, float[] kCache, float[] vCache,
        int numHeads, int numKvHeads, int headDim,
        int tStart, int tEnd)
    {
        int kvDim = numKvHeads * headDim;
        int effSeq = tEnd - tStart;
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[numHeads * headDim];

        var scores = new float[effSeq];
        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / (numHeads / numKvHeads);
            int qOff = h * headDim;

            float maxScore = float.NegativeInfinity;
            for (int t = 0; t < effSeq; t++)
            {
                int absT = t + tStart;
                int kOff = absT * kvDim + kvHead * headDim;
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[qOff + d] * kCache[kOff + d];
                scores[t] = dot * scale;
                if (scores[t] > maxScore) maxScore = scores[t];
            }

            float sum = 0f;
            for (int t = 0; t < effSeq; t++)
            {
                scores[t] = MathF.Exp(scores[t] - maxScore);
                sum += scores[t];
            }
            float invSum = 1.0f / sum;
            for (int t = 0; t < effSeq; t++)
                scores[t] *= invSum;

            for (int d = 0; d < headDim; d++)
            {
                float acc = 0f;
                for (int t = 0; t < effSeq; t++)
                {
                    int absT = t + tStart;
                    int vOff = absT * kvDim + kvHead * headDim;
                    acc += scores[t] * vCache[vOff + d];
                }
                output[qOff + d] = acc;
            }
        }
        return output;
    }
}
