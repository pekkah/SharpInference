namespace SharpInference.Core;

/// <summary>
/// Extended compute-backend interface for image-processing operations (conv2d, activations,
/// spatial rearrangements). Implemented by backends that can run a full convolutional forward
/// pass without CPU round-trips.
///
/// Inherits from <see cref="IComputeBackend"/> so callers receive Upload/Download/Free/AddInPlace
/// from the same object.
///
/// Tensor layout convention: all spatial tensors are CHW (channels-first), flat float32.
/// </summary>
public interface IImageOpsBackend : IComputeBackend
{
    /// <summary>
    /// 2D convolution (stride=1, same padding by default).
    /// input  [inCh, H, W], weight [outCh, inCh, k, k], bias [outCh]
    /// → output [outCh, H, W]
    /// </summary>
    Tensor Conv2d(Tensor input, Tensor weight, Tensor bias,
                  int inCh, int outCh, int h, int w, int ksize, int padding = -1);

    /// <summary>LeakyReLU in-place: x[i] = x[i] >= 0 ? x[i] : negSlope * x[i]</summary>
    void LeakyReluInPlace(Tensor x, float negSlope);

    /// <summary>Scale in-place: x[i] *= scale</summary>
    void ScaleInPlace(Tensor x, float scale);

    /// <summary>Scaled add in-place: dst[i] += src[i] * scale</summary>
    void AddScaledInPlace(Tensor dst, Tensor src, float scale);

    /// <summary>Clamp in-place: x[i] = clamp(x[i], min, max)</summary>
    void ClampInPlace(Tensor x, float min, float max);

    /// <summary>
    /// Channel concatenation along C axis.
    /// a [aCh, hw], b [bCh, hw] → output [(aCh+bCh), hw]
    /// </summary>
    Tensor CatChannels(Tensor a, int aCh, Tensor b, int bCh, int hw);

    /// <summary>
    /// Pixel shuffle: [inCh, H, W] → [inCh/r², H*r, W*r]
    /// where r = upscaleFactor.
    /// </summary>
    Tensor PixelShuffleGpu(Tensor input, int inCh, int h, int w, int upscaleFactor);

    /// <summary>
    /// Pixel unshuffle (inverse of pixel shuffle): [inCh, H*r, W*r] → [inCh*r², H, W]
    /// where r = downscaleFactor.
    /// </summary>
    Tensor PixelUnshuffleGpu(Tensor input, int inCh, int h, int w, int downscaleFactor);

    /// <summary>
    /// Nearest-neighbor 2× upsample: [ch, H, W] → [ch, 2H, 2W]
    /// </summary>
    Tensor Upsample2xGpu(Tensor input, int ch, int h, int w);

    // ── Batch recording ──────────────────────────────────────────────────────
    // BeginBatch/EndBatch enable dispatching the entire forward pass as a single
    // GPU command buffer submission, eliminating per-dispatch queue-submit overhead.
    // Free() calls between Begin/EndBatch are automatically deferred until after submit.

    /// <summary>Begin recording: subsequent dispatches are batched into one submission.</summary>
    void BeginBatch();

    /// <summary>
    /// Insert a compute→compute memory barrier (all prior writes visible to subsequent reads).
    /// Required between dispatches that have data dependencies in batch recording mode.
    /// No-op when not batching.
    /// </summary>
    void BatchBarrier();

    /// <summary>End batch: submit all recorded dispatches, wait for completion, process deferred frees.</summary>
    void EndBatch();
}
