using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;

namespace SharpInference.Cpu;

/// <summary>
/// CPU compute backend using SIMD intrinsics (AVX2/AVX-512/NEON via System.Runtime.Intrinsics).
/// All operations execute on the thread pool; uses unsafe memory for zero-copy tensor storage.
/// </summary>
public sealed unsafe class CpuBackend : IComputeBackend
{
    public string Name => $"CPU ({(Avx512F.IsSupported ? "AVX-512" : Avx2.IsSupported ? "AVX2" : "Scalar")})";

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape)
    {
        // TODO: allocate aligned native memory and copy data
        throw new NotImplementedException();
    }

    public void Download(Tensor src, Span<float> dst)
    {
        // TODO: copy from native memory back to managed span
        throw new NotImplementedException();
    }

    public void MatMul(Tensor lhs, Tensor rhs, Tensor output)
    {
        // TODO: GEMM with SIMD (AVX2 f32x8 / AVX-512 f32x16 lanes)
        throw new NotImplementedException();
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        // TODO: vectorised element-wise add
        throw new NotImplementedException();
    }

    public void RmsNorm(Tensor x, Tensor weight, float eps = 1e-5f)
    {
        // TODO: RMS normalisation kernel
        throw new NotImplementedException();
    }

    public void Softmax(Tensor x)
    {
        // TODO: numerically stable softmax
        throw new NotImplementedException();
    }

    public void SiLU(Tensor x)
    {
        // TODO: sigmoid linear unit
        throw new NotImplementedException();
    }

    public void RoPE(Tensor x, int position, int headDim)
    {
        // TODO: rotary positional embeddings
        throw new NotImplementedException();
    }

    public void Synchronize() { /* CPU operations are synchronous */ }

    public void Dispose() { }
}
