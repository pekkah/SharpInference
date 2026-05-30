using SharpInference.Core;

namespace SharpInference.Cuda;

/// <summary>
/// Result of an async <see cref="CudaBackend.UploadBackground(System.ReadOnlySpan{float}, TensorShape, bool)"/>
/// or <see cref="CudaBackend.UploadBackgroundRaw"/>.
/// <para>
/// The DMA is issued on the backend's dedicated upload stream and may still be
/// in flight when this struct is returned. Before any kernel on the compute
/// stream reads <see cref="Tensor"/>, the consumer MUST insert a wait via
/// <see cref="CudaBackend.WaitForUpload(CudaUploadHandle)"/> — otherwise the
/// read can race ahead of the transfer and observe pre-DMA contents.
/// </para>
/// <para>
/// The owned <see cref="UploadEvent"/> is a <c>cudaEvent_t</c>. Destroy it via
/// <see cref="CudaBackend.ReleaseUploadHandle(CudaUploadHandle)"/> once the
/// readiness no longer needs to be tracked; freeing the underlying tensor goes
/// through <see cref="CudaBackend.Free(Tensor)"/> like any other.
/// </para>
/// </summary>
public readonly record struct CudaUploadHandle(Tensor Tensor, nint UploadEvent);
