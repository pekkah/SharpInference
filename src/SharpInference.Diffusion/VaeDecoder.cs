using System.Numerics.Tensors;

namespace SharpInference.Diffusion;

/// <summary>
/// FLUX VAE decoder: decodes 16-channel latent tensor [1, 16, H, W] → RGB image [1, 3, H*8, W*8].
///
/// Loads weights from ae.safetensors (from black-forest-labs/FLUX.1-schnell).
/// Architecture:
///   post_quant_conv:  Conv2D(16→16, 1×1)
///   decoder.conv_in:  Conv2D(16→512, 3×3)
///   decoder.mid_block: ResBlock × 2 + AttnBlock
///   decoder.up_blocks:
///     up[3]: 512→512, ResBlock×3, Upsample (nearest 2×)
///     up[2]: 512→512, ResBlock×3, Upsample
///     up[1]: 512→256, ResBlock×3, Upsample
///     up[0]: 256→128, ResBlock×3
///   decoder.norm_out: GroupNorm(32, 128) + SiLU
///   decoder.conv_out: Conv2D(128→3, 3×3)
///
/// The VAE scaling constant 0.3611 and shift 0.1159 are applied before decoding.
/// </summary>
public sealed class VaeDecoder : IDisposable
{
    private readonly IWeightLoader _st;

    // FLUX VAE scale/shift applied to latent before decode
    private const float VaeScale = 1f / 0.3611f;
    private const float VaeShift = 0.1159f;

    public VaeDecoder(string path) => _st = SafetensorsLoader.Open(path);

    /// <summary>Create from a pre-opened IWeightLoader (SafetensorsLoader or GgufWeightLoader).</summary>
    public VaeDecoder(IWeightLoader st) => _st = st;

    /// <summary>
    /// Decode latent [C=16, H, W] → RGB float [3, H*8, W*8], values in [0,1].
    /// </summary>
    public float[] Decode(float[] latent, int latH, int latW)
    {
        // Rescale latent: z = latent / scale_factor + shift_factor
        var z = new float[latent.Length];
        for (int i = 0; i < z.Length; i++)
            z[i] = latent[i] / 0.3611f + VaeShift;

        // post_quant_conv: Conv2D(16→16, 1×1) — present in FLUX ae.safetensors, absent in Z-Image VAE
        if (_st.Contains("post_quant_conv.weight"))
            z = ConvBlock("post_quant_conv", z, 1, 16, latH, latW, 16, 1, padding: 0);

        // conv_in: Conv2D(16→512, 3×3)
        z = ConvBlock("decoder.conv_in", z, 1, 16, latH, latW, 512, 3);
        int ch = 512, h = latH, w = latW;

        // mid block: 2× ResBlock with single-head spatial attention in between
        z = ResBlock("decoder.mid_block.resnets.0", z, 1, ch, h, w);
        z = MidAttn("decoder.mid_block.attentions.0", z, 1, ch, h, w);
        z = ResBlock("decoder.mid_block.resnets.1", z, 1, ch, h, w);

        // Up blocks (numbered in reverse in safetensors: up_blocks.0 = deepest)
        // FLUX VAE up_blocks order: 0=512→512, 1=512→512, 2=512→256, 3=256→128
        (z, ch, h, w) = UpBlock(z, 1, ch, h, w, "decoder.up_blocks.0", outCh: 512, upsample: true);
        (z, ch, h, w) = UpBlock(z, 1, ch, h, w, "decoder.up_blocks.1", outCh: 512, upsample: true);
        (z, ch, h, w) = UpBlock(z, 1, ch, h, w, "decoder.up_blocks.2", outCh: 256, upsample: true);
        (z, ch, h, w) = UpBlock(z, 1, ch, h, w, "decoder.up_blocks.3", outCh: 128, upsample: false);

        // norm_out: GroupNorm — named "decoder.conv_norm_out" in Z-Image, "decoder.norm_out" in FLUX ae
        string normName = _st.Contains("decoder.conv_norm_out.weight") ? "decoder.conv_norm_out"
                                                                        : "decoder.norm_out";
        var gnW = _st.ReadF32($"{normName}.weight");
        var gnB = _st.ReadF32($"{normName}.bias");
        DiffusionOps.GroupNorm(z, gnW, gnB, 1, ch, h, w, groups: 32);
        DiffusionOps.SiluInPlace(z);

        // conv_out: Conv2D(128→3)
        z = ConvBlock("decoder.conv_out", z, 1, ch, h, w, 3, 3);

        // Clamp to [0, 1]
        for (int i = 0; i < z.Length; i++)
            z[i] = Math.Clamp((z[i] + 1f) * 0.5f, 0f, 1f);

        return z;  // shape: [3, h*8, w*8]
    }

    // ── Building blocks ───────────────────────────────────────────────────

    private (float[] z, int ch, int h, int w) UpBlock(
        float[] z, int n, int inCh, int h, int w,
        string prefix, int outCh, bool upsample)
    {
        // 3 ResBlocks in each up block
        for (int r = 0; r < 3; r++)
        {
            string resPrefix = $"{prefix}.resnets.{r}";
            // First ResBlock may change channels (inCh → outCh)
            z = ResBlock(resPrefix, z, n, r == 0 ? inCh : outCh, h, w, outCh);
        }
        int ch = outCh;

        if (upsample)
        {
            z = DiffusionOps.Upsample2x(z, n, ch, h, w);
            h *= 2; w *= 2;
            // Conv after upsample
            z = ConvBlock($"{prefix}.upsamplers.0.conv", z, n, ch, h, w, ch, 3);
        }
        return (z, ch, h, w);
    }

    private float[] ResBlock(string prefix, float[] x, int n, int inCh, int h, int w, int outCh = -1)
    {
        if (outCh < 0) outCh = inCh;

        // norm1 + silu + conv1
        var gnW1 = _st.ReadF32($"{prefix}.norm1.weight");
        var gnB1 = _st.ReadF32($"{prefix}.norm1.bias");
        var h1 = x.ToArray();
        DiffusionOps.GroupNorm(h1, gnW1, gnB1, n, inCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv1", h1, n, inCh, h, w, outCh, 3);

        // norm2 + silu + conv2
        var gnW2 = _st.ReadF32($"{prefix}.norm2.weight");
        var gnB2 = _st.ReadF32($"{prefix}.norm2.bias");
        DiffusionOps.GroupNorm(h1, gnW2, gnB2, n, outCh, h, w, groups: 32);
        DiffusionOps.SiluInPlace(h1);
        h1 = ConvBlock($"{prefix}.conv2", h1, n, outCh, h, w, outCh, 3);

        // Skip connection: project input if channels differ
        float[] skip = x;
        if (inCh != outCh)
            skip = ConvBlock($"{prefix}.conv_shortcut", x, n, inCh, h, w, outCh, 1, padding: 0);

        for (int i = 0; i < h1.Length; i++) h1[i] += skip[i];
        return h1;
    }

    private float[] MidAttn(string prefix, float[] x, int n, int ch, int h, int w)
    {
        // GroupNorm + single-head self-attention over spatial positions
        var normed = x.ToArray();
        var gnW = _st.ReadF32($"{prefix}.group_norm.weight");
        var gnB = _st.ReadF32($"{prefix}.group_norm.bias");
        DiffusionOps.GroupNorm(normed, gnW, gnB, n, ch, h, w, groups: 32);

        int hw = h * w;

        var wQ  = _st.ReadF32($"{prefix}.to_q.weight"); // [ch, ch]
        var wK  = _st.ReadF32($"{prefix}.to_k.weight");
        var wV  = _st.ReadF32($"{prefix}.to_v.weight");
        var wOut  = _st.ReadF32($"{prefix}.to_out.0.weight"); // [ch, ch]
        var bOut  = _st.ReadF32($"{prefix}.to_out.0.bias");

        float scale = 1f / MathF.Sqrt(ch);
        var result = x.ToArray();

        for (int b = 0; b < n; b++)
        {
            // Build Q, K, V via matrix multiply: normedT[hw, ch] @ W[ch, ch]^T → [hw, ch]
            // normed is NCHW; transpose to [hw, ch] for matmul
            var tokens = new float[hw * ch]; // [hw, ch]
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
                tokens[pos * ch + c2] = normed[b * ch * hw + c2 * hw + pos];

            var q = LinearHW(tokens, wQ, hw, ch);
            var k = LinearHW(tokens, wK, hw, ch);
            var v = LinearHW(tokens, wV, hw, ch);

            // Attention: for each query row, dot with all key rows, softmax, weighted V sum
            var attnOut = new float[hw * ch];
            var scores  = new float[hw];

            Parallel.For(0, hw, i =>
            {
                var qi = q.AsSpan(i * ch, ch);
                var localScores = new float[hw];
                for (int j = 0; j < hw; j++)
                    localScores[j] = TensorPrimitives.Dot(qi, k.AsSpan(j * ch, ch)) * scale;

                // Softmax
                float maxS = TensorPrimitives.Max(localScores.AsSpan());
                TensorPrimitives.Subtract(localScores.AsSpan(), maxS, localScores.AsSpan());
                TensorPrimitives.Exp(localScores.AsSpan(), localScores.AsSpan());
                TensorPrimitives.Divide(localScores.AsSpan(),
                    TensorPrimitives.Sum(localScores.AsSpan()), localScores.AsSpan());

                // Weighted V sum
                var outRow = attnOut.AsSpan(i * ch, ch);
                outRow.Clear();
                for (int j = 0; j < hw; j++)
                    TensorPrimitives.MultiplyAdd<float>(v.AsSpan(j * ch, ch), localScores[j], outRow, outRow);
            });

            // Output projection + residual (convert back to NCHW)
            var proj = LinearHW(attnOut, wOut, hw, ch);
            for (int pos = 0; pos < hw; pos++)
            for (int c2 = 0; c2 < ch; c2++)
            {
                int nchwOff = b * ch * hw + c2 * hw + pos;
                result[nchwOff] += proj[pos * ch + c2] + bOut[c2];
            }
        }

        return result;
    }

    /// <summary>Matrix multiply tokens[hw, ch] @ W[ch, ch]^T → out[hw, ch].</summary>
    private static float[] LinearHW(float[] tokens, float[] w, int hw, int ch)
    {
        var result = new float[hw * ch];
        Parallel.For(0, hw, pos =>
        {
            var rowIn  = tokens.AsSpan(pos * ch, ch);
            var rowOut = result.AsSpan(pos * ch, ch);
            for (int oc = 0; oc < ch; oc++)
                rowOut[oc] = TensorPrimitives.Dot(rowIn, w.AsSpan(oc * ch, ch));
        });
        return result;
    }

    private float[] ConvBlock(string name, float[] x, int n, int inCh, int h, int w, int outCh, int k, int padding = -1)
    {
        var weight = _st.ReadF32($"{name}.weight");  // [outCh, inCh, k, k]
        var bias   = _st.Contains($"{name}.bias") ? _st.ReadF32($"{name}.bias") : null;
        return DiffusionOps.Conv2D(x, weight, bias, n, inCh, h, w, outCh, k, k, stride: 1, padding: padding);
    }

    public void Dispose() => _st.Dispose();
}
