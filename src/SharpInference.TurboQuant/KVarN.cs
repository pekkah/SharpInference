using System.Runtime.CompilerServices;

namespace SharpInference.TurboQuant;

/// <summary>
/// KVarN: variance-normalized, calibration-free KV-cache quantization
/// (Müller et al., Huawei CSL, arXiv:2606.03458). Clean-room implementation
/// from the paper — the published reference is a Triton/vLLM fork and is used
/// for algorithm reference only.
///
/// This is the P0 CPU reference (issue #180): correctness-first, scalar kernels.
/// The pipeline, per 128-token tile and per kv-head, is:
///
///   1. Randomized Hadamard rotation along the channel (head_dim) axis. The
///      rotation is orthonormal so attention scores are preserved:
///      qᵀk = (S·H·q)ᵀ(S·H·k). Reuses <see cref="WalshHadamard"/>.
///   2. Dual-axis "Sinkhorn" variance normalization (the novel piece): a
///      log-space alternation of per-channel (column) and per-token (row)
///      standard-deviation normalization. A few iterations drive both axes
///      toward unit variance, equalizing the dynamic range a fixed-width grid
///      must cover and killing the per-token scale outliers that drive error
///      accumulation over long autoregressive decoding.
///   3. Asymmetric round-to-nearest quantization with a zero point:
///      keys per-channel at 4-bit, values per-token at 2-bit, group size = tile.
///   4. Scales folded at read time (see <see cref="KScore"/> /
///      <see cref="VAggregate"/>), so no decompressed tile is ever materialized
///      on the attention hot path.
///
/// The reconstruction of a rotated coordinate is exact-up-to-quantization:
///     X_rot[t,d] ≈ (code[t,d]·qscale + zero) · cscale[d] · rscale[t]
/// where (qscale, zero) are per-channel for keys / per-token for values, and
/// (cscale, rscale) are the Sinkhorn column/row scales.
/// </summary>
public static class KVarN
{
    /// <summary>Tile size in tokens (matches the paper's 128-token block / quant group).</summary>
    public const int TileSize = 128;

    /// <summary>Default number of Sinkhorn alternation iterations.</summary>
    public const int DefaultSinkhornIters = 5;

    /// <summary>Key quantization bit width.</summary>
    public const int KeyBits = 4;

    /// <summary>Value quantization bit width.</summary>
    public const int ValueBits = 2;

    private const float Eps = 1e-8f;

    /// <summary>
    /// Rotate a single head_dim vector into the KVarN domain: normalized
    /// Walsh-Hadamard transform followed by the per-head sign flip. Self-inverse
    /// and orthonormal, so it is also used to pre-rotate the query before
    /// <see cref="KScore"/> and to un-rotate the aggregated value output.
    /// </summary>
    public static void Rotate(ReadOnlySpan<float> vec, Span<float> dst,
        ReadOnlySpan<float> signPattern, int dim)
    {
        WalshHadamard.Transform(vec, dst, dim);
        WalshHadamard.ApplySignFlip(dst, signPattern);
    }

    /// <summary>
    /// Compress one key tile (<paramref name="t"/> tokens × <paramref name="headDim"/>
    /// channels, token-major) into a <see cref="KVarNTile"/>: rotate, Sinkhorn
    /// normalize, then per-channel 4-bit asymmetric RTN.
    /// </summary>
    /// <param name="src">Tile rows in the original domain, layout <c>src[t*headDim + d]</c>.</param>
    /// <param name="t">Number of tokens in the tile (1..TileSize).</param>
    /// <param name="headDim">Channel count (power of two).</param>
    /// <param name="signPattern">Per-head sign flip, length headDim.</param>
    /// <param name="iters">Sinkhorn iterations.</param>
    public static KVarNTile CompressKeyTile(ReadOnlySpan<float> src, int t, int headDim,
        ReadOnlySpan<float> signPattern, int iters = DefaultSinkhornIters)
    {
        var tile = new KVarNTile(t, headDim, perChannel: true);
        float[] y = RotateAndNormalize(src, t, headDim, signPattern, iters,
            tile.CScale, tile.RScale);

        // Per-channel asymmetric RTN at 4 bits over the whole tile (group = tile).
        const int levels = (1 << KeyBits) - 1;
        for (int d = 0; d < headDim; d++)
        {
            float minv = float.PositiveInfinity, maxv = float.NegativeInfinity;
            for (int i = 0; i < t; i++)
            {
                float v = y[i * headDim + d];
                if (v < minv) minv = v;
                if (v > maxv) maxv = v;
            }
            float qscale = (maxv - minv) / levels;
            float invq = qscale > Eps ? 1f / qscale : 0f;
            for (int i = 0; i < t; i++)
            {
                int code = (int)MathF.Round((y[i * headDim + d] - minv) * invq);
                if (code < 0) code = 0; else if (code > levels) code = levels;
                tile.SetKeyCode(i, d, code);
            }
            // Fold the per-channel quant scale/zero with the Sinkhorn column
            // scale so the read path only needs cscale·qscale and cscale·zero.
            tile.KQScale[d] = qscale * tile.CScale[d];
            tile.KZero[d] = minv * tile.CScale[d];
        }
        return tile;
    }

    /// <summary>
    /// Compress one value tile into a <see cref="KVarNTile"/>: rotate, Sinkhorn
    /// normalize, then per-token 2-bit asymmetric RTN.
    /// </summary>
    public static KVarNTile CompressValueTile(ReadOnlySpan<float> src, int t, int headDim,
        ReadOnlySpan<float> signPattern, int iters = DefaultSinkhornIters)
    {
        var tile = new KVarNTile(t, headDim, perChannel: false);
        float[] y = RotateAndNormalize(src, t, headDim, signPattern, iters,
            tile.CScale, tile.RScale);

        // Per-token asymmetric RTN at 2 bits over the whole row (group = headDim).
        const int levels = (1 << ValueBits) - 1;
        for (int i = 0; i < t; i++)
        {
            float minv = float.PositiveInfinity, maxv = float.NegativeInfinity;
            for (int d = 0; d < headDim; d++)
            {
                float v = y[i * headDim + d];
                if (v < minv) minv = v;
                if (v > maxv) maxv = v;
            }
            float qscale = (maxv - minv) / levels;
            float invq = qscale > Eps ? 1f / qscale : 0f;
            for (int d = 0; d < headDim; d++)
            {
                int code = (int)MathF.Round((y[i * headDim + d] - minv) * invq);
                if (code < 0) code = 0; else if (code > levels) code = levels;
                tile.SetValueCode(i, d, code);
            }
            // Fold per-token quant scale/zero with the Sinkhorn row scale.
            tile.VQScale[i] = qscale * tile.RScale[i];
            tile.VZero[i] = minv * tile.RScale[i];
        }
        return tile;
    }

    /// <summary>
    /// Rotate every token of a tile and apply dual-axis Sinkhorn variance
    /// normalization. Returns the normalized matrix (token-major) and fills the
    /// per-channel / per-token scales so that
    /// <c>rotated[t,d] = y[t,d]·cscale[d]·rscale[t]</c>.
    /// </summary>
    private static float[] RotateAndNormalize(ReadOnlySpan<float> src, int t, int headDim,
        ReadOnlySpan<float> signPattern, int iters, float[] cscale, float[] rscale)
    {
        float[] y = new float[t * headDim];
        Span<float> rotated = headDim <= 512 ? stackalloc float[headDim] : new float[headDim];
        for (int i = 0; i < t; i++)
        {
            Rotate(src.Slice(i * headDim, headDim), rotated, signPattern, headDim);
            rotated.CopyTo(y.AsSpan(i * headDim, headDim));
        }

        Sinkhorn(y, t, headDim, cscale, rscale, iters);
        return y;
    }

    /// <summary>
    /// Dual-axis Sinkhorn variance normalization. Operates in place on the
    /// token-major matrix <paramref name="y"/> (t×headDim), accumulating the
    /// per-channel column scales into <paramref name="cscale"/> and the
    /// per-token row scales into <paramref name="rscale"/>. After the loop the
    /// matrix has approximately unit RMS along both axes and the original value
    /// is <c>y[t,d]·cscale[d]·rscale[t]</c>.
    /// </summary>
    public static void Sinkhorn(Span<float> y, int t, int headDim,
        Span<float> cscale, Span<float> rscale, int iters)
    {
        cscale.Slice(0, headDim).Fill(1f);
        rscale.Slice(0, t).Fill(1f);

        for (int it = 0; it < iters; it++)
        {
            // Column pass: normalize each channel by its RMS across tokens.
            for (int d = 0; d < headDim; d++)
            {
                float sumSq = 0f;
                for (int i = 0; i < t; i++)
                {
                    float v = y[i * headDim + d];
                    sumSq += v * v;
                }
                float s = MathF.Sqrt(sumSq / t);
                if (s > Eps)
                {
                    float inv = 1f / s;
                    for (int i = 0; i < t; i++)
                        y[i * headDim + d] *= inv;
                    cscale[d] *= s;
                }
            }

            // Row pass: normalize each token by its RMS across channels.
            for (int i = 0; i < t; i++)
            {
                float sumSq = 0f;
                int base_ = i * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float v = y[base_ + d];
                    sumSq += v * v;
                }
                float s = MathF.Sqrt(sumSq / headDim);
                if (s > Eps)
                {
                    float inv = 1f / s;
                    for (int d = 0; d < headDim; d++)
                        y[base_ + d] *= inv;
                    rscale[i] *= s;
                }
            }
        }
    }

    /// <summary>
    /// Fused dequant-dot key scoring: computes the raw attention score
    /// <c>q·k[t] = Σ_d rotatedQuery[d]·rotated_k[t,d]</c> for every token in the
    /// tile, without materializing the decompressed keys. Writes into
    /// <paramref name="scores"/> (length ≥ tile.T), scaled by
    /// <paramref name="attnScale"/>.
    /// </summary>
    /// <param name="rotatedQuery">Query pre-rotated via <see cref="Rotate"/> with the key sign pattern.</param>
    public static void KScore(KVarNTile tile, ReadOnlySpan<float> rotatedQuery,
        float attnScale, Span<float> scores)
    {
        int headDim = tile.HeadDim;
        int t = tile.T;

        // score[t] = rscale[t]·( Σ_d a[d]·code[t,d] + b )
        //   a[d] = q_rot[d]·cscale[d]·qscale[d]   (= q_rot[d]·KQScale[d])
        //   b    = Σ_d q_rot[d]·cscale[d]·zero[d] (= Σ_d q_rot[d]·KZero[d])
        Span<float> a = headDim <= 512 ? stackalloc float[headDim] : new float[headDim];
        float b = 0f;
        for (int d = 0; d < headDim; d++)
        {
            a[d] = rotatedQuery[d] * tile.KQScale[d];
            b += rotatedQuery[d] * tile.KZero[d];
        }

        for (int i = 0; i < t; i++)
        {
            float acc = 0f;
            for (int d = 0; d < headDim; d++)
                acc += a[d] * tile.GetKeyCode(i, d);
            scores[i] = (acc + b) * tile.RScale[i] * attnScale;
        }
    }

    /// <summary>
    /// Fused value aggregation: accumulates <c>Σ_t weights[t]·value[t]</c> over
    /// the tile into <paramref name="outAcc"/> (length ≥ headDim), in the
    /// original (un-rotated) domain. The sign flip and inverse Hadamard are
    /// applied once at the end, amortized across all tokens.
    /// </summary>
    /// <param name="weights">Attention weights for the tile's tokens (length ≥ tile.T).</param>
    /// <param name="signPattern">The value sign pattern for this head.</param>
    public static void VAggregate(KVarNTile tile, ReadOnlySpan<float> weights,
        ReadOnlySpan<float> signPattern, Span<float> outAcc)
    {
        int headDim = tile.HeadDim;
        int t = tile.T;

        // out_rot[d] = cscale[d]·( Σ_t wt[t]·code[t,d] + zsum )
        //   wt[t] = weights[t]·rscale[t]·qscale[t]  (= weights[t]·VQScale[t])
        //   zsum  = Σ_t weights[t]·rscale[t]·zero[t] (= Σ_t weights[t]·VZero[t])
        Span<float> rot = headDim <= 512 ? stackalloc float[headDim] : new float[headDim];
        rot.Clear();
        float zsum = 0f;
        for (int i = 0; i < t; i++)
        {
            float wt = weights[i] * tile.VQScale[i];
            zsum += weights[i] * tile.VZero[i];
            for (int d = 0; d < headDim; d++)
                rot[d] += wt * tile.GetValueCode(i, d);
        }
        for (int d = 0; d < headDim; d++)
            rot[d] = (rot[d] + zsum) * tile.CScale[d];

        // Un-rotate: sign flip then inverse (= forward, self-inverse) Hadamard.
        WalshHadamard.ApplySignFlip(rot, signPattern);
        WalshHadamard.Transform(rot, rot, headDim);
        for (int d = 0; d < headDim; d++)
            outAcc[d] += rot[d];
    }

    /// <summary>
    /// Reconstruct one key vector in the original domain (for tests / fallback).
    /// </summary>
    public static void ReconstructKey(KVarNTile tile, int tokenIdx,
        ReadOnlySpan<float> signPattern, Span<float> dst)
    {
        int headDim = tile.HeadDim;
        float r = tile.RScale[tokenIdx];
        Span<float> rot = headDim <= 512 ? stackalloc float[headDim] : new float[headDim];
        for (int d = 0; d < headDim; d++)
            rot[d] = (tile.GetKeyCode(tokenIdx, d) * tile.KQScale[d] + tile.KZero[d]) * r;
        WalshHadamard.ApplySignFlip(rot, signPattern);
        WalshHadamard.Transform(rot, rot, headDim);
        rot.Slice(0, headDim).CopyTo(dst);
    }

    /// <summary>
    /// Reconstruct one value vector in the original domain (for tests / fallback).
    /// </summary>
    public static void ReconstructValue(KVarNTile tile, int tokenIdx,
        ReadOnlySpan<float> signPattern, Span<float> dst)
    {
        int headDim = tile.HeadDim;
        float q = tile.VQScale[tokenIdx];
        float z = tile.VZero[tokenIdx];
        Span<float> rot = headDim <= 512 ? stackalloc float[headDim] : new float[headDim];
        for (int d = 0; d < headDim; d++)
            rot[d] = (tile.GetValueCode(tokenIdx, d) * q + z) * tile.CScale[d];
        WalshHadamard.ApplySignFlip(rot, signPattern);
        WalshHadamard.Transform(rot, rot, headDim);
        rot.Slice(0, headDim).CopyTo(dst);
    }

    /// <summary>
    /// Generate a sign pattern for a (layer, head) pair. Thin wrapper over
    /// <see cref="WalshHadamard.GenerateSignPattern"/> so KVarN seeds stay
    /// distinct from the TurboQuant codebook path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float[] GenerateSignPattern(int dim, int seed) =>
        WalshHadamard.GenerateSignPattern(dim, seed);
}
