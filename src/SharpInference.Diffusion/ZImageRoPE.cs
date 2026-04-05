namespace SharpInference.Diffusion;

/// <summary>
/// 3-axis Rotary Position Embedding for Z-Image.
///
/// Three axes:  t (time/sequence position), h (row), w (column).
/// Per-axis dimensions from config: [32, 48, 48] — sum = 128 = head_dim.
/// Max positions per axis: [1536, 512, 512] (from axes_lens in config.json).
/// Base theta: 256 (much lower than typical LLM RoPE — spatial locality bias).
///
/// Implementation follows the original Z-Image code:
///   freqs[i] = 1 / (theta ^ (2i / d))
///   freqs_cis[pos] = e^{i * pos * freqs}  (complex exponential)
///   apply: rotate pairs (a,b) by angle — (a*cos − b*sin, a*sin + b*cos)
///
/// Usage:
///   var rope = new ZImageRoPE(params);
///   float[] freqs = rope.Compute(posIds, nTokens); // [nTokens, headDim/2 * 2]
///   rope.Apply(q, nTokens, nHeads, headDim, freqs);
///   rope.Apply(k, nTokens, nHeads, headDim, freqs);
/// </summary>
public sealed class ZImageRoPE
{
    private readonly int _headDim;   // 128
    private readonly int[] _dims;    // [32, 48, 48] half-dims per axis → actual half-dims = [16, 24, 24]
    private readonly float[][] _freqs; // precomputed freqs per axis [axisLen, dim/2]

    public ZImageRoPE(ZImageParams p)
    {
        _headDim = p.HeadDim;
        _dims    = p.AxesDims;

        _freqs = new float[3][];
        for (int ax = 0; ax < 3; ax++)
        {
            int d        = _dims[ax];          // e.g. 32
            int halfD    = d / 2;              // 16
            int maxLen   = p.AxesLens[ax];     // e.g. 1536
            float theta  = p.RopeTheta;        // 256

            var table = new float[maxLen * halfD * 2]; // cos and sin
            for (int pos = 0; pos < maxLen; pos++)
            {
                for (int j = 0; j < halfD; j++)
                {
                    float freq = 1f / MathF.Pow(theta, 2f * j / d);
                    float angle = pos * freq;
                    table[(pos * halfD + j) * 2 + 0] = MathF.Cos(angle); // cos
                    table[(pos * halfD + j) * 2 + 1] = MathF.Sin(angle); // sin
                }
            }
            _freqs[ax] = table;
        }
    }

    /// <summary>
    /// Build per-token rotation cosines/sines from 3D position IDs.
    /// </summary>
    /// <param name="posIds">Flat array [nTokens * 3] of (t, h, w) positions.</param>
    /// <param name="nTokens">Number of tokens.</param>
    /// <returns>Flat [nTokens * headDim] of interleaved (cos, sin) pairs for all 3 axes.</returns>
    public float[] BuildFreqs(int[] posIds, int nTokens)
    {
        // Total half_dim for full head = sum(dims)/2 = (32+48+48)/2 = 64
        int totalHalfDim = (_dims[0] + _dims[1] + _dims[2]) / 2;
        var result = new float[nTokens * totalHalfDim * 2];

        for (int t = 0; t < nTokens; t++)
        {
            int tPos = posIds[t * 3 + 0];
            int hPos = posIds[t * 3 + 1];
            int wPos = posIds[t * 3 + 2];

            int outOffset = t * totalHalfDim * 2;
            int[] axisPos = [tPos, hPos, wPos];

            for (int ax = 0; ax < 3; ax++)
            {
                int halfD   = _dims[ax] / 2;
                int pos     = axisPos[ax];
                var freqTab = _freqs[ax];

                for (int j = 0; j < halfD; j++)
                {
                    result[outOffset + 0] = freqTab[(pos * halfD + j) * 2 + 0]; // cos
                    result[outOffset + 1] = freqTab[(pos * halfD + j) * 2 + 1]; // sin
                    outOffset += 2;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Apply RoPE in-place to Q or K tensor.
    /// </summary>
    /// <param name="qk">Flat [nTokens * nHeads * headDim].</param>
    /// <param name="nTokens">Sequence length.</param>
    /// <param name="nHeads">Number of attention heads.</param>
    /// <param name="freqs">Output of BuildFreqs: [nTokens * headDim].</param>
    public void Apply(float[] qk, int nTokens, int nHeads, float[] freqs)
    {
        int halfDim = _headDim / 2;

        for (int t = 0; t < nTokens; t++)
        {
            int freqBase = t * halfDim * 2; // 2 floats per pair (cos, sin)

            for (int h = 0; h < nHeads; h++)
            {
                int qkBase = (t * nHeads + h) * _headDim;

                for (int j = 0; j < halfDim; j++)
                {
                    float cos = freqs[freqBase + j * 2 + 0];
                    float sin = freqs[freqBase + j * 2 + 1];
                    float a   = qk[qkBase + j * 2 + 0];
                    float b   = qk[qkBase + j * 2 + 1];
                    qk[qkBase + j * 2 + 0] = a * cos - b * sin;
                    qk[qkBase + j * 2 + 1] = a * sin + b * cos;
                }
            }
        }
    }

    /// <summary>
    /// Build position IDs for text tokens: t = 1..nTxt, h = 0, w = 0.
    /// </summary>
    public static int[] TextPosIds(int nTxt)
    {
        var ids = new int[nTxt * 3];
        for (int i = 0; i < nTxt; i++)
        {
            ids[i * 3 + 0] = i + 1; // t axis starts at 1
            ids[i * 3 + 1] = 0;
            ids[i * 3 + 2] = 0;
        }
        return ids;
    }

    /// <summary>
    /// Build position IDs for image patches: t = tOffset, h = 0..nRows-1, w = 0..nCols-1.
    /// tOffset = nTxt + 1  (image tokens start right after text).
    /// </summary>
    public static int[] ImagePosIds(int tOffset, int nRows, int nCols)
    {
        int nImg = nRows * nCols;
        var ids  = new int[nImg * 3];
        for (int r = 0; r < nRows; r++)
        for (int c = 0; c < nCols; c++)
        {
            int idx = r * nCols + c;
            ids[idx * 3 + 0] = tOffset;
            ids[idx * 3 + 1] = r;
            ids[idx * 3 + 2] = c;
        }
        return ids;
    }
}
