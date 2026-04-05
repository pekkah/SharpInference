namespace SharpInference.Core;

/// <summary>
/// Abstraction over a compute device (CPU, Vulkan GPU, etc.).
/// Implementations perform tensor operations on their respective hardware.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }

    // --- Memory management ---

    /// <summary>Allocate a tensor of the given shape, initialized to zero.</summary>
    Tensor Allocate(TensorShape shape, DType dtype = DType.Float32);

    /// <summary>Free a tensor's backing memory.</summary>
    void Free(Tensor tensor);

    /// <summary>Copy data to the backend device, returning a new tensor.</summary>
    Tensor Upload(ReadOnlySpan<float> data, TensorShape shape);

    /// <summary>Copy data back from the backend device.</summary>
    void Download(Tensor src, Span<float> dst);

    // --- Core math operations ---

    /// <summary>Matrix-vector multiply: output[i] = sum_j(matrix[i,j] * vector[j]).</summary>
    void MatMul(Tensor output, Tensor matrix, Tensor vector);

    /// <summary>Element-wise addition in-place: dst += src.</summary>
    void AddInPlace(Tensor dst, Tensor src);

    /// <summary>Element-wise multiplication: output = a * b.</summary>
    void ElementwiseMul(Tensor output, Tensor a, Tensor b);

    /// <summary>Apply RMS-norm: x = (x / rms(x)) * weight.</summary>
    void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f);

    /// <summary>Softmax in-place along the last dimension.</summary>
    void Softmax(Tensor x);

    /// <summary>Apply SiLU (x * sigmoid(x)) activation in-place.</summary>
    void SiLU(Tensor x);

    /// <summary>Apply rotary positional embedding in-place.</summary>
    void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f);

    /// <summary>
    /// General matrix multiply: C[M,N] = A[M,K] × B[N,K]^T
    /// A is activations [M rows, K cols], B is weight matrix [N rows, K cols] stored row-major.
    /// Used for transformer projections where nBatch > 1.
    /// </summary>
    void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N);

    /// <summary>
    /// Full-sequence self-attention: softmax(Q×K^T / sqrt(headDim)) × V
    /// q, k, v: [nTok, nHeads * headDim] with interleaved layout [tok*nHeads + head, headDim]
    /// output: [nTok, nHeads * headDim] same layout
    /// </summary>
    void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                          int nTok, int nHeads, int headDim, float scale);

    /// <summary>Wait for all queued operations to complete.</summary>
    void Synchronize();
}
