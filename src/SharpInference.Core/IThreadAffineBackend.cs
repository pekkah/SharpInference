namespace SharpInference.Core;

/// <summary>
/// A backend whose work must run on a single, consistent OS thread (issue #302). Both the
/// per-token forward-pass interface (<see cref="IForwardPass"/>) and the engine's batched-forward
/// surface extend this, so an inference engine can pin all of a backend's work to one owned thread
/// without knowing which concrete backend it drives.
/// </summary>
public interface IThreadAffineBackend
{
    /// <summary>
    /// Make the backend's thread-affine context current on the calling thread. A CUDA context is
    /// thread-affine: the engine that drives the forward pass must call this on the worker thread
    /// that issues the backend's calls, before the first one. Otherwise — in non-interactive
    /// sessions, where the driver does not keep the device's primary context current on freshly-
    /// scheduled threads — the first CUDA call on an unbound thread can hang forever. Backends with
    /// no thread-affine context (CPU, Vulkan) leave this a no-op. Idempotent and cheap after the
    /// first call per thread.
    /// </summary>
    void BindToCurrentThread() { }
}
