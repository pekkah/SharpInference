using System.Numerics.Tensors;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Vision;

/// <summary>
/// Encoder-free vision embedder for the Gemma 4 "unified" projector (gemma4uv).
/// Turns a preprocessed RGB image into LLM soft-token embeddings.
///
/// Mirrors llama.cpp <c>tools/mtmd/models/gemma4uv.cpp::build()</c> exactly:
/// <code>
///   im2col(patch=48, stride=48)            // per patch -> 6912 vec, order [c*48*48 + ky*48 + kx]
///   LayerNorm(patch_norm_1, eps=1e-5)      // 6912-dim, BEFORE the linear
///   matmul(v.patch_embd: 6912->3840) + bias
///   LayerNorm(patch_norm_2, eps=1e-5)      // 3840-dim
///   + pos_x[col] + pos_y[row]              // learned 2D position tables; col=i%nCols, row=i/nCols
///   LayerNorm(patch_norm_3, eps=1e-5)      // "pos_norm"
///   rms_norm(eps=1e-6)                      // embedding_pre_projection_norm (no weight)
///   matmul(mm.input_projection: 3840->3840)
/// </code>
/// There is no ViT, pooling, or std-affine (block_count=0). One soft token per 48×48 patch,
/// so <c>nTokens = (W/48) * (H/48)</c>, raster-ordered.
/// </summary>
public sealed class GemmaUvVisionEmbedder
{
    private readonly VisionModel _m;
    private readonly int _embd;       // 3840
    private readonly int _patch;      // 48
    private readonly int _patchVec;   // 6912 = 3*48*48
    private readonly int _posSize;    // 1120 (per-axis position table length)
    private readonly float _lnEps, _rmsEps;

    private readonly float[] _n1w, _n1b, _n2w, _n2b, _n3w, _n3b;
    private readonly float[] _peBias;
    private readonly float[] _pos;    // [2 * posSize * embd] : x-table then y-table

    private readonly unsafe byte* _peW;   // v.patch_embd.weight    F32  [embd, patchVec]
    private readonly unsafe byte* _mmW;   // mm.input_projection    BF16 [embd, embd]

    /// <remarks>
    /// The embedder borrows <paramref name="m"/>'s memory-mapped weights (it caches raw
    /// pointers into them), so <paramref name="m"/> MUST outlive this embedder and must not
    /// be disposed before the last <see cref="Forward"/> call. <see cref="Forward"/> guards
    /// against a disposed model and throws <see cref="ObjectDisposedException"/>.
    /// </remarks>
    public GemmaUvVisionEmbedder(VisionModel m)
    {
        _m = m;
        _embd = m.EmbeddingLength;
        _patch = m.PatchSize;
        _patchVec = _patch * _patch * 3;
        _posSize = (int)m.PositionEmbd.Dimensions[1];
        _lnEps = m.LayerNormEps;
        _rmsEps = m.RmsNormEps;

        _n1w = m.LoadFloats(m.PatchNorm1W); _n1b = m.LoadFloats(m.PatchNorm1B);
        _n2w = m.LoadFloats(m.PatchNorm2W); _n2b = m.LoadFloats(m.PatchNorm2B);
        _n3w = m.LoadFloats(m.PatchNorm3W); _n3b = m.LoadFloats(m.PatchNorm3B);
        _peBias = m.LoadFloats(m.PatchEmbdBias);
        _pos = m.LoadFloats(m.PositionEmbd);

        unsafe
        {
            _peW = m.Gguf.GetTensorDataPtr(m.PatchEmbdWeight);
            _mmW = m.Gguf.GetTensorDataPtr(m.MmInputProjection);
        }
    }

    /// <summary>Soft-token count for an image of the given (already-resized) dimensions.</summary>
    public int TokenCount(int height, int width) => (height / _patch) * (width / _patch);

    /// <summary>
    /// Run the projector. <paramref name="chw"/> is the preprocessed image in planar CHW order
    /// (3 channels × height × width, values in [0,1]); width and height must be multiples of 48.
    /// Returns <c>nTokens × EmbeddingLength</c> soft-token embeddings, row-major.
    /// </summary>
    public float[] Forward(ReadOnlySpan<float> chw, int height, int width, out int nTokens)
    {
        _m.EnsureNotDisposed();
        if (width % _patch != 0 || height % _patch != 0)
            throw new ArgumentException($"image dims ({width}x{height}) must be multiples of {_patch}.");
        int gx = width / _patch, gy = height / _patch;
        // The learned position tables hold _posSize (1120) entries per axis; a grid larger
        // than that has no position embedding. Preprocessing keeps tokens within
        // [MinImageTokens, MaxImageTokens], so this only trips on un-preprocessed input.
        if (gx > _posSize || gy > _posSize)
            throw new ArgumentException(
                $"image {width}x{height} -> {gx}x{gy} patches exceeds position-table capacity {_posSize} per axis; " +
                $"preprocess the image first (max {_m.MaxImageTokens} tokens).");
        nTokens = gx * gy;
        int hw = height * width;

        // 1. im2col: per patch a 6912 vector laid out [c*patch*patch + ky*patch + kx]
        var P = new float[nTokens * _patchVec];
        for (int p = 0; p < nTokens; p++)
        {
            int pr = p / gx, pc = p % gx;
            int baseY = pr * _patch, baseX = pc * _patch;
            int dst = p * _patchVec;
            for (int c = 0; c < 3; c++)
            {
                int cPlane = c * hw;
                int cOff = dst + c * _patch * _patch;
                for (int ky = 0; ky < _patch; ky++)
                {
                    int srcRow = cPlane + (baseY + ky) * width + baseX;
                    int dstRow = cOff + ky * _patch;
                    for (int kx = 0; kx < _patch; kx++)
                        P[dstRow + kx] = chw[srcRow + kx];
                }
            }
        }

        // 2. patch_norm.1 (LayerNorm over 6912), then linear patch_embd + bias
        LayerNorm(P, _n1w, _n1b, _patchVec, _lnEps);
        var E = new float[nTokens * _embd];
        unsafe
        {
            fixed (float* pP = P, pE = E)
                for (int p = 0; p < nTokens; p++)
                    SimdKernels.MatVec(pE + p * _embd, _peW, pP + p * _patchVec, _embd, _patchVec, DType.Float32);
        }
        for (int p = 0; p < nTokens; p++)
        {
            var row = E.AsSpan(p * _embd, _embd);
            TensorPrimitives.Add(row, _peBias, row);
        }

        // 3. patch_norm.2, then add learned 2D position embeddings, then patch_norm.3 (pos_norm)
        LayerNorm(E, _n2w, _n2b, _embd, _lnEps);
        int yBase = _posSize * _embd;   // y-table starts after the x-table
        for (int p = 0; p < nTokens; p++)
        {
            int pr = p / gx, pc = p % gx;
            var row = E.AsSpan(p * _embd, _embd);
            TensorPrimitives.Add(row, _pos.AsSpan(pc * _embd, _embd), row);            // + pos_x[col]
            TensorPrimitives.Add(row, _pos.AsSpan(yBase + pr * _embd, _embd), row);    // + pos_y[row]
        }
        LayerNorm(E, _n3w, _n3b, _embd, _lnEps);

        // 4. embedding_pre_projection_norm (RMSNorm, no weight), then mm.input_projection
        var outp = new float[nTokens * _embd];
        unsafe
        {
            fixed (float* pE = E, pO = outp)
            {
                for (int p = 0; p < nTokens; p++)
                    SimdKernels.PureRmsNorm(pE + p * _embd, pE + p * _embd, _embd, _rmsEps);
                for (int p = 0; p < nTokens; p++)
                    SimdKernels.MatVec(pO + p * _embd, _mmW, pE + p * _embd, _embd, _embd, DType.BFloat16);
            }
        }
        return outp;
    }

    /// <summary>
    /// Per-row LayerNorm matching ggml_norm: (x-mean)/sqrt(var+eps) then *weight + bias,
    /// with population variance over the last <paramref name="dim"/> elements. In-place.
    /// </summary>
    private static void LayerNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias, int dim, float eps)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var r = x.Slice(row * dim, dim);
            float mean = TensorPrimitives.Sum<float>(r) / dim;
            TensorPrimitives.Subtract(r, mean, r);
            float var = TensorPrimitives.Dot<float>(r, r) / dim;
            float scale = 1f / MathF.Sqrt(var + eps);
            TensorPrimitives.Multiply(r, scale, r);
            TensorPrimitives.Multiply(r, weight, r);
            TensorPrimitives.Add(r, bias, r);
        }
    }
}
