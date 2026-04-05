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
        // Reshape to [B, hw, ch] for attention — iterate over batch
        var q  = new float[n * hw * ch];
        var k  = new float[n * hw * ch];
        var v  = new float[n * hw * ch];

        var wQ  = _st.ReadF32($"{prefix}.to_q.weight"); // [ch, ch]
        var wK  = _st.ReadF32($"{prefix}.to_k.weight");
        var wV  = _st.ReadF32($"{prefix}.to_v.weight");

        // Linear projection: normed is [n, ch, h, w] → treat each spatial pos as a token
        for (int b = 0; b < n; b++)
        for (int pos = 0; pos < hw; pos++)
        {
            // Build input vector at (b, pos): gather from [b, :, pos//w, pos%w]
            int inOff = b * (ch * hw) + pos;
            int outRowQ = (b * hw + pos) * ch;

            for (int oc = 0; oc < ch; oc++)
            {
                float sumQ = 0f, sumK = 0f, sumV = 0f;
                int wRow = oc * ch;
                for (int ic = 0; ic < ch; ic++)
                {
                    float xVal = normed[b * ch * hw + ic * hw + pos];
                    sumQ += wQ[wRow + ic] * xVal;
                    sumK += wK[wRow + ic] * xVal;
                    sumV += wV[wRow + ic] * xVal;
                }
                q[outRowQ + oc] = sumQ;
                k[outRowQ + oc] = sumK;
                v[outRowQ + oc] = sumV;
            }
        }

        // Scaled dot-product attention [n, hw, hw]
        float scale  = 1f / MathF.Sqrt(ch);
        var   scores = new float[hw];
        var   attnOut = new float[n * hw * ch];

        for (int b = 0; b < n; b++)
        for (int i = 0; i < hw; i++)
        {
            int qi = (b * hw + i) * ch;
            for (int j = 0; j < hw; j++)
            {
                int kj = (b * hw + j) * ch;
                float s = 0f;
                for (int d = 0; d < ch; d++) s += q[qi + d] * k[kj + d];
                scores[j] = s * scale;
            }

            // Softmax
            float maxS = scores[0];
            for (int j = 1; j < hw; j++) if (scores[j] > maxS) maxS = scores[j];
            float sumExp = 0f;
            for (int j = 0; j < hw; j++) { scores[j] = MathF.Exp(scores[j] - maxS); sumExp += scores[j]; }
            for (int j = 0; j < hw; j++) scores[j] /= sumExp;

            // Weighted V
            int outI = (b * hw + i) * ch;
            for (int d = 0; d < ch; d++)
            {
                float acc = 0f;
                for (int j = 0; j < hw; j++) acc += scores[j] * v[(b * hw + j) * ch + d];
                attnOut[outI + d] = acc;
            }
        }

        // Output projection
        var wOut  = _st.ReadF32($"{prefix}.to_out.0.weight"); // [ch, ch]
        var bOut  = _st.ReadF32($"{prefix}.to_out.0.bias");
        var proj  = new float[n * hw * ch];
        for (int b = 0; b < n; b++)
        for (int pos = 0; pos < hw; pos++)
        {
            int rowOff = (b * hw + pos) * ch;
            for (int oc = 0; oc < ch; oc++)
            {
                float s = bOut[oc];
                int wRow = oc * ch;
                for (int ic = 0; ic < ch; ic++) s += wOut[wRow + ic] * attnOut[rowOff + ic];
                proj[rowOff + oc] = s;
            }
        }

        // Residual: convert proj back to [n, ch, h, w] and add to x
        var result = x.ToArray();
        for (int b = 0; b < n; b++)
        for (int c2 = 0; c2 < ch; c2++)
        for (int pos = 0; pos < hw; pos++)
            result[b * ch * hw + c2 * hw + pos] += proj[(b * hw + pos) * ch + c2];

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
