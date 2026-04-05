using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpInference.Cpu;

namespace SharpInference.Diffusion;

/// <summary>
/// CPU-only primitive operations needed by the VAE decoder and text encoders.
/// All methods operate on flat float[] arrays with explicit shape parameters.
/// Tensors are NCHW (batch × channels × height × width) for spatial ops.
/// </summary>
internal static class DiffusionOps
{
    // ── Activation functions ──────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Gelu(float x)
    {
        // Tanh GELU approximation — matches PyTorch default
        const float c = 0.044715f;
        float v = 0.7978845608028654f * (x + c * x * x * x);
        return 0.5f * x * (1.0f + MathF.Tanh(v));
    }

    public static void GeluInPlace(Span<float> x)
    {
        for (int i = 0; i < x.Length; i++)
            x[i] = Gelu(x[i]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x) => x / (1f + MathF.Exp(-x));

    public static void SiluInPlace(Span<float> x)
    {
        // SiLU(x) = x * sigmoid(x). Pool a temp buffer to avoid heap pressure.
        var tempArr = ArrayPool<float>.Shared.Rent(x.Length);
        var temp    = tempArr.AsSpan(0, x.Length);
        TensorPrimitives.Sigmoid(x, temp);   // temp = sigmoid(x)
        TensorPrimitives.Multiply(x, temp, x);  // x   *= temp  → SiLU
        ArrayPool<float>.Shared.Return(tempArr);
    }

    // ── Normalization ─────────────────────────────────────────────────────

    /// <summary>
    /// Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias.
    /// Operates on the last axis of length <paramref name="dim"/>.
    /// </summary>
    public static void LayerNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int dim, float eps = 1e-5f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var row_ = x.Slice(row * dim, dim);
            float mean = TensorPrimitives.Sum(row_) / dim;
            // Shift so we can compute variance as dot(shifted, shifted)
            TensorPrimitives.Subtract(row_, mean, row_);
            float var  = TensorPrimitives.Dot<float>(row_, row_) / dim;
            float scale = 1f / MathF.Sqrt(var + eps);
            TensorPrimitives.Multiply(row_, scale, row_);
            TensorPrimitives.Multiply(row_, weight, row_);
            TensorPrimitives.Add(row_, bias, row_);
        }
    }

    /// <summary>
    /// Group Normalization: groups of channels along C axis.
    /// Input layout: [N, C, H, W] flattened. Normalizes within each group.
    /// </summary>
    public static void GroupNorm(Span<float> x, ReadOnlySpan<float> weight, ReadOnlySpan<float> bias,
                                 int n, int c, int h, int w, int groups, float eps = 1e-5f)
    {
        int chansPerGroup = c / groups;
        int spatialSize   = h * w;
        int groupElements = chansPerGroup * spatialSize;

        for (int b = 0; b < n; b++)
        {
            int bOff = b * c * spatialSize;
            for (int g = 0; g < groups; g++)
            {
                // Compute mean and variance over this group
                int gOff = bOff + g * groupElements;
                float mean = 0f;
                for (int i = 0; i < groupElements; i++) mean += x[gOff + i];
                mean /= groupElements;

                float var = 0f;
                for (int i = 0; i < groupElements; i++)
                { float d = x[gOff + i] - mean; var += d * d; }
                float invStd = 1f / MathF.Sqrt(var / groupElements + eps);

                for (int gc = 0; gc < chansPerGroup; gc++)
                {
                    int c_abs = g * chansPerGroup + gc;
                    int cOff  = bOff + c_abs * spatialSize;
                    for (int s = 0; s < spatialSize; s++)
                    {
                        float v = (x[cOff + s] - mean) * invStd;
                        x[cOff + s] = v * weight[c_abs] + bias[c_abs];
                    }
                }
            }
        }
    }

    // ── Convolution ───────────────────────────────────────────────────────

    /// <summary>
    /// 2D convolution. Inputs/outputs: [N, C, H, W] (NCHW flat arrays).
    /// Kernel: [outC, inC, kH, kW].  Bias: [outC] (nullable).
    /// Supports stride=1 or stride=2, padding computed as "same" for stride=1.
    /// </summary>
    public static float[] Conv2D(float[] input, float[] kernel, float[]? bias,
                                  int n, int inC, int inH, int inW,
                                  int outC, int kH, int kW,
                                  int stride = 1, int padding = -1)
    {
        if (padding < 0) padding = (kH - 1) / 2; // "same" padding for odd kernels

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        var output = new float[n * outC * outH * outW];

        for (int b = 0; b < n; b++)
        {
            for (int oc = 0; oc < outC; oc++)
            {
                float biasVal = bias is not null ? bias[oc] : 0f;
                int outBase = b * outC * outH * outW + oc * outH * outW;

                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        float sum = biasVal;
                        for (int ic = 0; ic < inC; ic++)
                        {
                            int kBase = (oc * inC + ic) * kH * kW;
                            int inBase = b * inC * inH * inW + ic * inH * inW;
                            for (int kh = 0; kh < kH; kh++)
                            {
                                int ih = oh * stride - padding + kh;
                                if ((uint)ih >= (uint)inH) continue;
                                for (int kw = 0; kw < kW; kw++)
                                {
                                    int iw = ow * stride - padding + kw;
                                    if ((uint)iw >= (uint)inW) continue;
                                    sum += input[inBase + ih * inW + iw] * kernel[kBase + kh * kW + kw];
                                }
                            }
                        }
                        output[outBase + oh * outW + ow] = sum;
                    }
                }
            }
        }
        return output;
    }

    // ── Spatial ops ───────────────────────────────────────────────────────

    /// <summary>
    /// Nearest-neighbor 2× upsample. Input: [N, C, H, W]. Output: [N, C, 2H, 2W].
    /// </summary>
    public static float[] Upsample2x(float[] input, int n, int c, int h, int w)
    {
        int oh = h * 2, ow = w * 2;
        var output = new float[n * c * oh * ow];
        for (int b = 0; b < n; b++)
        {
            int bIn = b * c * h * w, bOut = b * c * oh * ow;
            for (int ch = 0; ch < c; ch++)
            {
                int cIn = bIn + ch * h * w, cOut = bOut + ch * oh * ow;
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        float v = input[cIn + y * w + x];
                        output[cOut + (2*y)   * ow + 2*x    ] = v;
                        output[cOut + (2*y)   * ow + 2*x + 1] = v;
                        output[cOut + (2*y+1) * ow + 2*x    ] = v;
                        output[cOut + (2*y+1) * ow + 2*x + 1] = v;
                    }
                }
            }
        }
        return output;
    }

    // ── Linear (dense) layer helpers ──────────────────────────────────────

    /// <summary>
    /// Dense layer for float32 weights (VAE decoder): out[i] = sum_j(weight[i,j] * x[j]) + bias[i].
    /// weight: [outDim, inDim], result: [n, outDim].
    /// Uses TensorPrimitives.Dot for hardware-accelerated SIMD — no GCHandle, no unsafe.
    /// For large quantized GGUF weights use SimdKernels.MatMulBatched via IWeightLoader.TryGetRaw.
    /// </summary>
    public static float[] Linear(float[] x, float[] weight, float[]? bias, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];

        Parallel.For(0, outDim, o =>
        {
            float b0   = bias is not null ? bias[o] : 0f;
            var   wRow = weight.AsSpan(o * inDim, inDim);
            for (int b = 0; b < n; b++)
            {
                var xRow = x.AsSpan(b * inDim, inDim);
                result[b * outDim + o] = b0 + TensorPrimitives.Dot<float>(xRow, wRow);
            }
        });

        return result;
    }

    /// <summary>RMS normalization along last axis (in-place). T5LayerNorm = no bias/mean centering.</summary>
    public static void RmsNorm(Span<float> x, ReadOnlySpan<float> weight, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var row_ = x.Slice(row * dim, dim);
            float ss     = TensorPrimitives.Dot<float>(row_, row_);
            float invRms = 1f / MathF.Sqrt(ss / dim + eps);
            TensorPrimitives.Multiply(row_, weight, row_);   // row_ *= weight
            TensorPrimitives.Multiply(row_, invRms, row_);   // row_ *= invRms
        }
    }

    /// <summary>Softmax over the last axis of length <paramref name="dim"/> (in-place).</summary>
    public static void Softmax(Span<float> x, int dim)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var s = x.Slice(row * dim, dim);
            float max = TensorPrimitives.Max(s);
            TensorPrimitives.Subtract(s, max, s);
            TensorPrimitives.Exp(s, s);
            float sum = TensorPrimitives.Sum(s);
            TensorPrimitives.Divide(s, sum, s);
        }
    }

    /// <summary>In-place element-wise: x[i] = x[i] * (1 + scale[i]) + shift[i].</summary>
    public static void ScaleShiftInPlace(Span<float> x, ReadOnlySpan<float> scale, ReadOnlySpan<float> shift, int dim)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            int off = row * dim;
            for (int i = 0; i < dim; i++)
                x[off + i] = x[off + i] * (1f + scale[i]) + shift[i];
        }
    }

    /// <summary>Per-row addition: a[i] += b[i].</summary>
    public static void AddRows(float[] a, float[] b, int n, int dim)
    {
        TensorPrimitives.Add(a.AsSpan(0, n * dim), b.AsSpan(0, n * dim), a.AsSpan(0, n * dim));
    }

    // ── Aliases for consistent PascalCase naming ──────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float SiLU(float x) => Silu(x);

    public static void SiLUInPlace(Span<float> x) => SiluInPlace(x);

    // ── Offset-based overloads (used by ZImageDiT, QwenTextEncoder) ───────

    /// <summary>
    /// RMS-norm one row: dst[dstOff..dstOff+dim] = RMSNorm(src[srcOff..srcOff+dim]).
    /// </summary>
    public static void RmsNorm(float[] src, int srcOff, int dim,
                                float[] weight, float eps,
                                float[] dst, int dstOff)
    {
        var srcSlice = src.AsSpan(srcOff, dim);
        var dstSlice = dst.AsSpan(dstOff, dim);
        float ss     = TensorPrimitives.Dot<float>(srcSlice, srcSlice);
        float invRms = 1f / MathF.Sqrt(ss / dim + eps);
        TensorPrimitives.Multiply(srcSlice, weight.AsSpan(0, dim), dstSlice); // dst = src * weight
        TensorPrimitives.Multiply(dstSlice, invRms, dstSlice);                // dst *= invRms
    }

    /// <summary>
    /// Softmax over scores[offset .. offset+n] in-place.
    /// </summary>
    public static void Softmax(float[] scores, int offset, int n)
    {
        var s = scores.AsSpan(offset, n);
        float max = TensorPrimitives.Max(s);
        TensorPrimitives.Subtract(s, max, s);
        TensorPrimitives.Exp(s, s);
        float sum = TensorPrimitives.Sum(s);
        TensorPrimitives.Divide(s, sum, s);
    }

    public static unsafe void SoftmaxInPlace(float* scores, int n)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < n; i++) { scores[i] = MathF.Exp(scores[i] - max); sum += scores[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < n; i++) scores[i] *= inv;
    }
}
