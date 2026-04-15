using System.Runtime.CompilerServices;

namespace SharpInference.Diffusion;

/// <summary>
/// 2D Rotary Position Embeddings for FLUX image patches.
///
/// Unlike 1D LLM RoPE, FLUX encodes each patch by its (row, col) grid position.
/// Each head has head_dim dimensions split as: [dim/2 for row-freq, dim/2 for col-freq].
/// The precomputed cosine/sine tables are shaped [n_patches, head_dim/2] per axis.
/// </summary>
internal static class Flux2DRoPE
{
    /// <summary>
    /// Build (cos, sin) frequency tables for a sequence of 2D positions.
    /// </summary>
    /// <param name="positions">Array of (row, col) pairs, length nPatches×2.</param>
    /// <param name="headDim">Head dimension (must be even; dim/2 used per axis).</param>
    /// <param name="theta">RoPE base frequency (default 10000).</param>
    /// <returns>cos[nPatches, headDim] and sin[nPatches, headDim] interleaved as (row|col).</returns>
    public static (float[] cos, float[] sin) BuildFreqs(
        ReadOnlySpan<int> positions, int nPatches, int headDim, float theta = 10000f)
    {
        // Each axis gets headDim/2 frequencies; total = headDim per head
        int halfDim = headDim / 2;
        int quarterDim = halfDim / 2;  // frequencies per axis

        var cos = new float[nPatches * headDim];
        var sin = new float[nPatches * headDim];

        // Precompute inverse frequencies for each axis
        var invFreq = new float[quarterDim];
        for (int i = 0; i < quarterDim; i++)
            invFreq[i] = 1f / MathF.Pow(theta, 2f * i / headDim);

        for (int p = 0; p < nPatches; p++)
        {
            int row = positions[p * 2];
            int col = positions[p * 2 + 1];
            int outBase = p * headDim;

            // Row axis → first half of headDim
            for (int f = 0; f < quarterDim; f++)
            {
                float angle = row * invFreq[f];
                cos[outBase + f * 2]     = MathF.Cos(angle);
                cos[outBase + f * 2 + 1] = MathF.Cos(angle);
                sin[outBase + f * 2]     = MathF.Sin(angle);
                sin[outBase + f * 2 + 1] = MathF.Sin(angle);
            }

            // Col axis → second half of headDim
            int colBase = outBase + halfDim;
            for (int f = 0; f < quarterDim; f++)
            {
                float angle = col * invFreq[f];
                cos[colBase + f * 2]     = MathF.Cos(angle);
                cos[colBase + f * 2 + 1] = MathF.Cos(angle);
                sin[colBase + f * 2]     = MathF.Sin(angle);
                sin[colBase + f * 2 + 1] = MathF.Sin(angle);
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Build patch position ids for an image of (heightPatches × widthPatches).
    /// Returns flat array of row,col pairs: [nPatches × 2].
    /// </summary>
    public static int[] ImagePatchIds(int heightPatches, int widthPatches)
    {
        int n = heightPatches * widthPatches;
        var ids = new int[n * 2];
        for (int r = 0; r < heightPatches; r++)
            for (int c = 0; c < widthPatches; c++)
            {
                int p = r * widthPatches + c;
                ids[p * 2]     = r;
                ids[p * 2 + 1] = c;
            }
        return ids;
    }

    /// <summary>
    /// Apply RoPE to Q or K tensor in-place.
    /// x layout: [nSeq, nHeads, headDim], cos/sin: [nSeq, headDim].
    /// Only applied to the first <paramref name="nImgPatches"/> positions (image patches);
    /// text positions are left unchanged.
    /// </summary>
    public static void ApplyInPlace(float[] x, float[] cos, float[] sin,
                                    int nSeq, int nHeads, int headDim, int nImgPatches)
    {
        for (int s = 0; s < nImgPatches && s < nSeq; s++)
        {
            int freqOff = s * headDim;
            for (int h = 0; h < nHeads; h++)
            {
                int xOff = (s * nHeads + h) * headDim;
                RotateHalfInPlace(x, xOff, cos, sin, freqOff, headDim);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RotateHalfInPlace(float[] x, int xOff,
                                           float[] cos, float[] sin, int freqOff,
                                           int headDim)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            float x0 = x[xOff + i];
            float x1 = x[xOff + i + half];
            float c = cos[freqOff + i];
            float s = sin[freqOff + i];
            x[xOff + i]        = x0 * c - x1 * s;
            x[xOff + i + half] = x1 * c + x0 * s;
        }
    }
}
