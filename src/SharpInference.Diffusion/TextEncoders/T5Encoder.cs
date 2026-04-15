namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// T5-XXL encoder-only transformer.
/// Produces context embeddings [seq, 4096] for the FLUX DiT.
///
/// Loads weights from t5xxl_fp16.safetensors or t5xxl_fp8.safetensors.
/// Architecture: 24 encoder layers, d_model=4096, 64 heads, head_dim=64, d_ff=10240.
/// Notable differences from standard transformers:
///   - T5LayerNorm = RMSNorm without mean-centering or bias
///   - Relative position bias (bias[head, query_pos - key_pos + max_dist])
///   - SiLU gated FFN (Flan-T5 / T5v1.1 style): h = silu(w1*x) * (w3*x); out = w2*h
///   - No absolute position embeddings
/// </summary>
public sealed class T5Encoder : IDisposable
{
    private const int Layers      = 24;
    private const int Dim         = 4096;
    private const int Heads       = 64;
    private const int HeadDim     = 64;
    private const int FfDim       = 10240;
    private const int VocabSize   = 32128;
    private const int RelPosBuckets = 32;
    private const int MaxRelPos   = 128;

    private readonly SafetensorsLoader _st;
    private float[]? _relPosBias; // lazy cached

    public T5Encoder(string path) => _st = SafetensorsLoader.Open(path);

    /// <summary>
    /// Encode token ids → context embeddings [seq, 4096].
    /// </summary>
    public float[] Encode(int[] tokens)
    {
        int seq = tokens.Length;
        var tokEmb = _st.ReadF32("shared.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int off = tokens[t] * Dim;
            x.AsSpan().Slice(t * Dim, Dim).Fill(0);
            tokEmb.AsSpan(off, Dim).CopyTo(x.AsSpan(t * Dim, Dim));
        }

        // Precompute relative position bias from first block (shared across all blocks)
        if (_relPosBias is null)
        {
            var rpW = _st.ReadF32("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            // rpW: [num_buckets, num_heads] = [32, 64]
            _relPosBias = ComputeRelPosBias(rpW, seq, Heads);
        }
        else if (_relPosBias.Length != seq * seq * Heads)
        {
            // Recompute if sequence length changed
            var rpW = _st.ReadF32("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
            _relPosBias = ComputeRelPosBias(rpW, seq, Heads);
        }

        for (int i = 0; i < Layers; i++)
            x = EncoderBlock(x, _relPosBias, seq, i);

        // Final layer norm
        var fnW = _st.ReadF32("encoder.final_layer_norm.weight");
        DiffusionOps.RmsNorm(x, fnW, Dim);
        return x;  // [seq, 4096]
    }

    private float[] EncoderBlock(float[] x, float[] relPosBias, int seq, int blockIdx)
    {
        string p = $"encoder.block.{blockIdx}.layer";

        // Self-attention sub-layer
        var lnW0 = _st.ReadF32($"{p}.0.layer_norm.weight");
        var xNorm = x.ToArray();
        DiffusionOps.RmsNorm(xNorm, lnW0, Dim);
        var attn = SelfAttention(xNorm, relPosBias, seq, $"{p}.0.SelfAttention", blockIdx);
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        // Feed-forward sub-layer
        var lnW1 = _st.ReadF32($"{p}.1.layer_norm.weight");
        var xNorm2 = x.ToArray();
        DiffusionOps.RmsNorm(xNorm2, lnW1, Dim);
        var ff = FeedForward(xNorm2, seq, $"{p}.1.DenseReluDense");
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private float[] SelfAttention(float[] x, float[] relBias, int seq, string p, int blockIdx)
    {
        var qW = _st.ReadF32($"{p}.q.weight");
        var kW = _st.ReadF32($"{p}.k.weight");
        var vW = _st.ReadF32($"{p}.v.weight");
        var oW = _st.ReadF32($"{p}.o.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var k = DiffusionOps.Linear(x, kW, null, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, null, seq, Dim, Dim);

        float scale = 1f / MathF.Sqrt(HeadDim);
        var attnOut = new float[seq * Dim];

        for (int h = 0; h < Heads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                for (int j = 0; j < seq; j++)
                {
                    float dot = 0f;
                    int qOff = i * Dim + h * HeadDim;
                    int kOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    // Add relative position bias
                    scores[i * seq + j] = dot * scale + relBias[(h * seq + i) * seq + j];
                }
            }
            DiffusionOps.Softmax(scores, seq);

            for (int i = 0; i < seq; i++)
            {
                int outOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    float w = scores[i * seq + j];
                    int vOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) attnOut[outOff + d] += w * v[vOff + d];
                }
            }
        }

        return DiffusionOps.Linear(attnOut, oW, null, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        // Flan-T5 gated SiLU FFN: h = silu(wi_0 * x) * (wi_1 * x); out = wo * h
        var wi0W = _st.ReadF32($"{p}.wi_0.weight");  // [FfDim, Dim]
        var wi1W = _st.ReadF32($"{p}.wi_1.weight");  // [FfDim, Dim]
        var woW  = _st.ReadF32($"{p}.wo.weight");    // [Dim, FfDim]

        var gate = DiffusionOps.Linear(x, wi0W, null, seq, Dim, FfDim);
        var val  = DiffusionOps.Linear(x, wi1W, null, seq, Dim, FfDim);

        // h = silu(gate) * val
        for (int i = 0; i < gate.Length; i++)
            gate[i] = DiffusionOps.Silu(gate[i]) * val[i];

        return DiffusionOps.Linear(gate, woW, null, seq, FfDim, Dim);
    }

    /// <summary>Compute T5 relative position bias for a sequence length.</summary>
    private static float[] ComputeRelPosBias(float[] biasWeight, int seq, int nHeads)
    {
        // biasWeight: [RelPosBuckets, nHeads] = [32, 64]
        var bias = new float[nHeads * seq * seq];
        for (int i = 0; i < seq; i++)
        {
            for (int j = 0; j < seq; j++)
            {
                int bucket = RelPosBucket(i - j);
                for (int h = 0; h < nHeads; h++)
                    bias[(h * seq + i) * seq + j] = biasWeight[bucket * nHeads + h];
            }
        }
        return bias;
    }

    private static int RelPosBucket(int relPos)
    {
        // T5 bidirectional relative position bucketing
        bool negative = relPos < 0;
        int pos = negative ? -relPos : relPos;
        int maxExact = RelPosBuckets / 2;
        if (pos < maxExact) return negative ? RelPosBuckets - pos : pos;

        float log = MathF.Log((float)pos / maxExact) / MathF.Log((float)MaxRelPos / maxExact);
        int bucket = maxExact + (int)(log * (RelPosBuckets / 2 - maxExact));
        bucket = Math.Min(bucket, RelPosBuckets / 2 - 1);
        return negative ? RelPosBuckets - bucket : bucket;
    }

    public void Dispose() => _st.Dispose();
}
