namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// CLIP-L (ViT-L/14) text encoder.
/// Produces:
///   - last_hidden_state [seq, 768] — used as secondary conditioning in some models
///   - pooled_output [768]          — the [EOS] token embedding, main conditioning for FLUX vector_in
///
/// Loads weights from clip_l.safetensors (comfyanonymous/flux_text_encoders).
/// Architecture: 12 transformer encoder layers, 768-dim, 12 heads, head_dim=64.
/// Activation: QuickGELU (x * sigmoid(1.702*x)).
/// </summary>
public sealed class ClipLEncoder : IDisposable
{
    private const int Layers    = 12;
    private const int Dim       = 768;
    private const int Heads     = 12;
    private const int HeadDim   = 64;
    private const int MlpDim    = 3072;
    private const int MaxSeqLen = 77;
    private const int VocabSize = 49408;

    private readonly SafetensorsLoader _st;

    public ClipLEncoder(string path) => _st = SafetensorsLoader.Open(path);

    /// <summary>
    /// Encode a list of token ids (length ≤ 77, padded to 77 with 0).
    /// Returns (lastHiddenState [77, 768], pooledOutput [768]).
    /// </summary>
    public (float[] lastHidden, float[] pooled) Encode(int[] tokens)
    {
        int seq = MaxSeqLen;
        // Pad/truncate to MaxSeqLen
        var ids = new int[seq];
        int copy = Math.Min(tokens.Length, seq);
        Array.Copy(tokens, ids, copy);

        // Token + position embeddings
        var tokEmb = _st.ReadF32("text_model.embeddings.token_embedding.weight");
        var posEmb = _st.ReadF32("text_model.embeddings.position_embedding.weight");

        var x = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int tokOff = ids[t] * Dim;
            int posOff = t * Dim;
            int xOff   = t * Dim;
            for (int d = 0; d < Dim; d++)
                x[xOff + d] = tokEmb[tokOff + d] + posEmb[posOff + d];
        }

        // Build causal mask (CLIP uses causal masking like GPT)
        var mask = BuildCausalMask(seq);

        // 12 encoder layers
        for (int i = 0; i < Layers; i++)
            x = EncoderLayer(x, mask, seq, i);

        // Final layer norm
        var lnW = _st.ReadF32("text_model.final_layer_norm.weight");
        var lnB = _st.ReadF32("text_model.final_layer_norm.bias");
        DiffusionOps.LayerNorm(x, lnW, lnB, Dim);

        // Pooled = EOS token (last non-pad token, or last position)
        // CLIP uses the EOS token embedding at position of the EOT (49407)
        int eosPos = copy - 1;
        for (int t = 0; t < copy; t++)
            if (ids[t] == 49407) { eosPos = t; break; }

        var pooled = x.AsSpan(eosPos * Dim, Dim).ToArray();
        return (x, pooled);
    }

    private float[] EncoderLayer(float[] x, float[] mask, int seq, int layerIdx)
    {
        string p = $"text_model.encoder.layers.{layerIdx}";

        // Self-attention with residual
        var lnW1 = _st.ReadF32($"{p}.layer_norm1.weight");
        var lnB1 = _st.ReadF32($"{p}.layer_norm1.bias");
        var xNorm = x.ToArray();
        DiffusionOps.LayerNorm(xNorm, lnW1, lnB1, Dim);

        var attn = SelfAttention(xNorm, mask, seq, p);
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        // MLP with residual
        var lnW2 = _st.ReadF32($"{p}.layer_norm2.weight");
        var lnB2 = _st.ReadF32($"{p}.layer_norm2.bias");
        var xNorm2 = x.ToArray();
        DiffusionOps.LayerNorm(xNorm2, lnW2, lnB2, Dim);

        var mlp = Mlp(xNorm2, seq, p);
        for (int i = 0; i < x.Length; i++) x[i] += mlp[i];

        return x;
    }

    private float[] SelfAttention(float[] x, float[] mask, int seq, string p)
    {
        var qW = _st.ReadF32($"{p}.self_attn.q_proj.weight");
        var kW = _st.ReadF32($"{p}.self_attn.k_proj.weight");
        var vW = _st.ReadF32($"{p}.self_attn.v_proj.weight");
        var qB = _st.ReadF32($"{p}.self_attn.q_proj.bias");
        var kB = _st.ReadF32($"{p}.self_attn.k_proj.bias");
        var vB = _st.ReadF32($"{p}.self_attn.v_proj.bias");
        var oW = _st.ReadF32($"{p}.self_attn.out_proj.weight");
        var oB = _st.ReadF32($"{p}.self_attn.out_proj.bias");

        var q = DiffusionOps.Linear(x, qW, qB, seq, Dim, Dim);  // [seq, Dim]
        var k = DiffusionOps.Linear(x, kW, kB, seq, Dim, Dim);
        var v = DiffusionOps.Linear(x, vW, vB, seq, Dim, Dim);

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
                    scores[i * seq + j] = dot * scale + mask[i * seq + j];
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

        return DiffusionOps.Linear(attnOut, oW, oB, seq, Dim, Dim);
    }

    private float[] Mlp(float[] x, int seq, string p)
    {
        var fc1W = _st.ReadF32($"{p}.mlp.fc1.weight");
        var fc1B = _st.ReadF32($"{p}.mlp.fc1.bias");
        var fc2W = _st.ReadF32($"{p}.mlp.fc2.weight");
        var fc2B = _st.ReadF32($"{p}.mlp.fc2.bias");

        var h = DiffusionOps.Linear(x, fc1W, fc1B, seq, Dim, MlpDim);
        // QuickGELU activation: x * sigmoid(1.702 * x)
        for (int i = 0; i < h.Length; i++)
            h[i] = h[i] * (1f / (1f + MathF.Exp(-1.702f * h[i])));
        return DiffusionOps.Linear(h, fc2W, fc2B, seq, MlpDim, Dim);
    }

    private static float[] BuildCausalMask(int seq)
    {
        var mask = new float[seq * seq];
        for (int i = 0; i < seq; i++)
            for (int j = i + 1; j < seq; j++)
                mask[i * seq + j] = float.NegativeInfinity;
        return mask;
    }

    public void Dispose() => _st.Dispose();
}
