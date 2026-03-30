namespace SharpInference.Core;

/// <summary>
/// Abstraction over a compute device (CPU, Vulkan GPU, etc.).
/// Implementations perform tensor operations on their respective hardware.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }

    /// <summary>Perform matrix multiplication: out = lhs @ rhs.</summary>
    void MatMul(Tensor lhs, Tensor rhs, Tensor output);

    /// <summary>Element-wise addition in-place: dst += src.</summary>
    void AddInPlace(Tensor dst, Tensor src);

    /// <summary>Apply RMS-norm in-place.</summary>
    void RmsNorm(Tensor x, Tensor weight, float eps = 1e-5f);

    /// <summary>Softmax in-place along the last dimension.</summary>
    void Softmax(Tensor x);

    /// <summary>Apply SiLU activation in-place.</summary>
    void SiLU(Tensor x);

    /// <summary>Rope positional embedding in-place.</summary>
    void RoPE(Tensor x, int position, int headDim);

    /// <summary>Copy data to the backend device.</summary>
    Tensor Upload(ReadOnlySpan<float> data, TensorShape shape);

    /// <summary>Copy data back from the backend device.</summary>
    void Download(Tensor src, Span<float> dst);

    /// <summary>Wait for all queued operations to complete.</summary>
    void Synchronize();
}
