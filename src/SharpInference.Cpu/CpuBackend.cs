using System.Buffers;
using System.Runtime.InteropServices;
using SharpInference.Core;

namespace SharpInference.Cpu;

/// <summary>
/// CPU compute backend. Phase 1: scalar reference implementation for correctness.
/// All tensors are backed by NativeMemory-allocated float buffers.
/// </summary>
public sealed unsafe class CpuBackend : IComputeBackend
{
    public string Name => "CPU (Scalar)";
    public SgemmPrecision BestSgemmPrecision => SgemmPrecision.Fp32;

    // --- Memory management ---

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32)
    {
        if (dtype != DType.Float32)
            throw new NotSupportedException($"CpuBackend Phase 1 only supports Float32, got {dtype}");

        var count = shape.ElementCount;
        var ptr = NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));
        return new Tensor(shape, dtype, (nint)ptr);
    }

    public void Free(Tensor tensor)
    {
        if (tensor.Handle != 0)
            NativeMemory.Free((void*)tensor.Handle);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        if (data.Length != shape.ElementCount)
            throw new ArgumentException(
                $"Data length ({data.Length}) doesn't match shape element count ({shape.ElementCount})");

        var tensor = Allocate(shape);
        data.CopyTo(new Span<float>((void*)tensor.Handle, data.Length));
        return tensor;
    }

    public void Download(Tensor src, Span<float> dst)
    {
        var count = (int)src.ElementCount;
        if (dst.Length < count)
            throw new ArgumentException($"Destination too small ({dst.Length} < {count})");

        new ReadOnlySpan<float>((void*)src.Handle, count).CopyTo(dst);
    }

    public Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape) =>
        throw new NotSupportedException("CpuBackend does not support fp16 upload");

    public void DownloadHalf(Tensor src, Span<Half> dst) =>
        throw new NotSupportedException("CpuBackend does not support fp16 download");

    public Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape) =>
        throw new NotSupportedException("CpuBackend does not support bf16 upload");

    public void DownloadBf16(Tensor src, Span<ushort> dst) =>
        throw new NotSupportedException("CpuBackend does not support bf16 download");

    public Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape) =>
        throw new NotSupportedException("CpuBackend does not support fp8 upload");

    public void DownloadFp8(Tensor src, Span<byte> dst) =>
        throw new NotSupportedException("CpuBackend does not support fp8 download");

    public Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("CpuBackend does not support raw quantized upload");

    public bool SupportsGpuDequant => false;

    public void DequantQ5KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CpuBackend does not support GPU dequantization");

    public void DequantQ4KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CpuBackend does not support GPU dequantization");

    // --- Core math operations ---

    /// <summary>
    /// Matrix-vector multiply: output[i] = sum_j(matrix[i,j] * vector[j]).
    /// Matrix shape: [rows, cols], vector shape: [cols], output shape: [rows].
    /// </summary>
    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        var rows = (int)matrix.Shape.Dims[0];
        var cols = (int)matrix.Shape.Dims[1];
        var m = (float*)matrix.Handle;
        var v = (float*)vector.Handle;
        var o = (float*)output.Handle;

        for (int i = 0; i < rows; i++)
        {
            float sum = 0;
            var row = m + (long)i * cols;
            for (int j = 0; j < cols; j++)
                sum += row[j] * v[j];
            o[i] = sum;
        }
    }

    /// <summary>Element-wise addition in-place: dst += src.</summary>
    public void AddInPlace(Tensor dst, Tensor src)
    {
        var count = (int)dst.ElementCount;
        var d = (float*)dst.Handle;
        var s = (float*)src.Handle;

        for (int i = 0; i < count; i++)
            d[i] += s[i];
    }

    /// <summary>Element-wise multiplication: output = a * b.</summary>
    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        var count = (int)a.ElementCount;
        var o = (float*)output.Handle;
        var pa = (float*)a.Handle;
        var pb = (float*)b.Handle;

        for (int i = 0; i < count; i++)
            o[i] = pa[i] * pb[i];
    }

    /// <summary>
    /// RMS normalization: output = (x / rms(x)) * weight.
    /// rms(x) = sqrt(mean(x^2) + eps)
    /// </summary>
    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        var count = (int)x.ElementCount;
        var px = (float*)x.Handle;
        var pw = (float*)weight.Handle;
        var po = (float*)output.Handle;

        // Compute mean of squares
        float sumSq = 0;
        for (int i = 0; i < count; i++)
            sumSq += px[i] * px[i];

        float rms = MathF.Sqrt(sumSq / count + eps);
        float scale = 1.0f / rms;

        // Normalize and scale by weight
        for (int i = 0; i < count; i++)
            po[i] = px[i] * scale * pw[i];
    }

    /// <summary>
    /// Numerically stable softmax in-place along the full tensor.
    /// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    /// </summary>
    public void Softmax(Tensor x)
    {
        var count = (int)x.ElementCount;
        var px = (float*)x.Handle;

        // Find max for numerical stability
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
            if (px[i] > max) max = px[i];

        // Compute exp(x - max) and sum
        float sum = 0;
        for (int i = 0; i < count; i++)
        {
            px[i] = MathF.Exp(px[i] - max);
            sum += px[i];
        }

        // Normalize
        float invSum = 1.0f / sum;
        for (int i = 0; i < count; i++)
            px[i] *= invSum;
    }

    /// <summary>
    /// SiLU (Swish) activation in-place: x = x * sigmoid(x).
    /// </summary>
    public void SiLU(Tensor x)
    {
        var count = (int)x.ElementCount;
        var px = (float*)x.Handle;

        for (int i = 0; i < count; i++)
        {
            float val = px[i];
            px[i] = val / (1.0f + MathF.Exp(-val));
        }
    }

    /// <summary>
    /// Rotary positional embedding (RoPE) in-place.
    /// Applies rotation to pairs of elements: (x[2i], x[2i+1]) rotated by position * freq_i.
    /// freq_i = 1 / (theta ^ (2i / headDim))
    /// </summary>
    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f)
    {
        var count = (int)x.ElementCount;
        var px = (float*)x.Handle;
        int halfDim = headDim / 2;

        // Process each head in the tensor
        int numHeads = count / headDim;
        for (int h = 0; h < numHeads; h++)
        {
            var head = px + h * headDim;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(ropeTheta, 2.0f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                float x0 = head[i];
                float x1 = head[i + halfDim];
                head[i] = x0 * cos - x1 * sin;
                head[i + halfDim] = x0 * sin + x1 * cos;
            }
        }
    }

    public void Synchronize() { /* CPU operations are synchronous */ }

    /// <summary>
    /// General matrix multiply: C[M,N] = A[M,K] × B[N,K]^T using CBLAS SGEMM.
    /// Falls back to a scalar loop if OpenBLAS is not available.
    /// </summary>
    public unsafe void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N)
    {
        var a = (float*)A.Handle;
        var b = (float*)B.Handle;
        var c = (float*)C.Handle;

        if (BlasInterop.IsAvailable)
        {
            BlasInterop.Sgemm(
                BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                M, N, K,
                1.0f, a, K,
                b, K,
                0.0f, c, N);
        }
        else
        {
            // Scalar fallback: C[i,j] = sum_k A[i,k] * B[j,k]
            for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++)
                    acc += a[i * K + k] * b[j * K + k];
                c[i * N + j] = acc;
            }
        }
    }

    /// <summary>
    /// Full-sequence self-attention.
    /// Layout: element (tok, head, d) at index tok*nHeads*headDim + head*headDim + d.
    /// </summary>
    public unsafe void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                                        int nTok, int nHeads, int headDim, float scale)
    {
        var qPtr = (float*)q.Handle;
        var kPtr = (float*)k.Handle;
        var vPtr = (float*)v.Handle;
        var oPtr = (float*)output.Handle;

        int scoreCount = nHeads * nTok * nTok;
        float[] scoresBuf = ArrayPool<float>.Shared.Rent(scoreCount);
        try
        {
            for (int h = 0; h < nHeads; h++)
            {
                int sBase = h * nTok * nTok;
                for (int i = 0; i < nTok; i++)
                {
                    float* qi = qPtr + ((long)i * nHeads + h) * headDim;
                    int sRow = sBase + i * nTok;
                    for (int j = 0; j < nTok; j++)
                    {
                        float* kj = kPtr + ((long)j * nHeads + h) * headDim;
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                            dot += qi[d] * kj[d];
                        scoresBuf[sRow + j] = dot * scale;
                    }
                    // Softmax over the nTok scores for query i
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < nTok; j++)
                        if (scoresBuf[sRow + j] > max) max = scoresBuf[sRow + j];
                    float sum = 0f;
                    for (int j = 0; j < nTok; j++)
                    {
                        scoresBuf[sRow + j] = MathF.Exp(scoresBuf[sRow + j] - max);
                        sum += scoresBuf[sRow + j];
                    }
                    float invSum = 1f / sum;
                    for (int j = 0; j < nTok; j++)
                        scoresBuf[sRow + j] *= invSum;
                }

                // Gather V for this head, accumulate weighted output
                float[] vhBuf = ArrayPool<float>.Shared.Rent(nTok * headDim);
                try
                {
                    fixed (float* vhPtr = vhBuf)
                    {
                        for (int j = 0; j < nTok; j++)
                        {
                            float* src = vPtr + ((long)j * nHeads + h) * headDim;
                            float* dst = vhPtr + j * headDim;
                            for (int d = 0; d < headDim; d++)
                                dst[d] = src[d];
                        }
                        for (int i = 0; i < nTok; i++)
                        {
                            int sRow = sBase + i * nTok;
                            float* outRow = oPtr + ((long)i * nHeads + h) * headDim;
                            for (int d = 0; d < headDim; d++)
                                outRow[d] = 0f;
                            for (int j = 0; j < nTok; j++)
                            {
                                float w = scoresBuf[sRow + j];
                                float* vj = vhPtr + j * headDim;
                                for (int d = 0; d < headDim; d++)
                                    outRow[d] += w * vj[d];
                            }
                        }
                    }
                }
                finally { ArrayPool<float>.Shared.Return(vhBuf); }
            }
        }
        finally { ArrayPool<float>.Shared.Return(scoresBuf); }
    }

    public void Dispose() { }
}
