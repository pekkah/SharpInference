namespace SharpInference.Core;

/// <summary>
/// Abstraction over weight storage backends (safetensors or GGUF).
/// Diffusion pipelines (ZImageDiT, VaeDecoder) and the DSpark draft-head loader
/// depend on this interface rather than concrete loaders.
/// </summary>
public interface IWeightLoader : IDisposable
{
    /// <summary>Returns true if the named tensor exists.</summary>
    bool Contains(string name);

    /// <summary>Read a tensor and return it as a float32 array (dequantized if necessary).
    /// Use for small tensors (norms, biases, embeddings). For large weight matrices
    /// prefer <see cref="TryGetRaw"/> to avoid allocating multi-hundred-MB float arrays.</summary>
    float[] ReadF32(string name);

    /// <summary>
    /// For GGUF backends: returns a direct pointer into the memory-mapped file data,
    /// along with dtype and shape (rows = output features = ne1, cols = input features = ne0).
    /// The pointer is valid for the lifetime of this loader — no allocation, no copy.
    /// Returns <c>false</c> for safetensors or 1-D tensors; caller falls back to
    /// <see cref="ReadF32"/> plus a dense F32 linear.
    /// </summary>
    unsafe bool TryGetRaw(string name,
        out nint dataPtr, out long byteLen,
        out DType dtype, out int rows, out int cols);
}
