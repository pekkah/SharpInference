namespace SharpInference.Core;

/// <summary>Reduced-precision variants available for SGEMM dispatch.</summary>
public enum SgemmPrecision
{
    Fp32,
    Fp16,
    Bf16,
    Int8Fp16,
    Fp8E4M3,
}

/// <summary>
/// Abstraction over a compute device (CPU, Vulkan GPU, etc.).
/// Implementations perform tensor operations on their respective hardware.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }

    /// <summary>Best SGEMM precision the backend supports for ZImage DiT.</summary>
    SgemmPrecision BestSgemmPrecision { get; }

    // --- Memory management ---

    /// <summary>Allocate a tensor of the given shape, initialized to zero.</summary>
    Tensor Allocate(TensorShape shape, DType dtype = DType.Float32);

    /// <summary>Free a tensor's backing memory.</summary>
    void Free(Tensor tensor);

    /// <summary>Copy data to the backend device, returning a new tensor.</summary>
    Tensor Upload(ReadOnlySpan<float> data, TensorShape shape);

    /// <summary>Copy data back from the backend device.</summary>
    void Download(Tensor src, Span<float> dst);

    /// <summary>Copy fp16 data to the backend device, returning a Float16 tensor.</summary>
    Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape);

    /// <summary>Copy fp16 data back from a Float16 tensor on the backend device.</summary>
    void DownloadHalf(Tensor src, Span<Half> dst);

    /// <summary>Copy bf16 data (as raw ushort bits) to the backend device, returning a BFloat16 tensor.</summary>
    Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape);

    /// <summary>Copy bf16 data back from a BFloat16 tensor on the backend device.</summary>
    void DownloadBf16(Tensor src, Span<ushort> dst);

    /// <summary>Copy fp8 E4M3 data (one byte per element) to the backend device, returning a Float8E4M3 tensor.</summary>
    Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape);

    /// <summary>Copy fp8 E4M3 data back from a Float8E4M3 tensor on the backend device.</summary>
    void DownloadFp8(Tensor src, Span<byte> dst);

    /// <summary>Upload raw quantized bytes to a device-local GPU buffer.</summary>
    Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype);

    /// <summary>True if the backend supports GPU-side dequantization of Q4_K/Q5_K weights.</summary>
    bool SupportsGpuDequant { get; }

    /// <summary>GPU-side dequantize Q5_K raw bytes → fp16 output.</summary>
    void DequantQ5KM(Tensor src, Tensor dst, int numBlocks);

    /// <summary>GPU-side dequantize Q4_K raw bytes → fp16 output.</summary>
    void DequantQ4KM(Tensor src, Tensor dst, int numBlocks);

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
