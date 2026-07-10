using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInference.TurboQuant;

/// <summary>
/// KVarN KV-cache quantizer, preset <c>kvarn_k4v2_g128</c>: per-tile Hadamard
/// rotation + dual-axis ("Sinkhorn") variance normalization + asymmetric
/// round-to-nearest quantization. Keys are quantized to 4 bits per-channel,
/// values to 2 bits per-token in channel groups of <see cref="ValueGroupSize"/>.
/// Clean-room implementation from the algorithm description in
/// docs/kvarn-feasibility-research.md (Müller et al., arXiv:2606.03458); no
/// code was ported from the reference vLLM fork.
///
/// A tile is <see cref="TileTokens"/> = 128 consecutive token vectors of one
/// kv-head, row-major [token, channel]. The whole tile must be assembled
/// before compression because the dual-axis normalization needs both token
/// and channel statistics — the same staging pattern TurboQuantKvCache
/// already uses for its 32-position FastScan tiles.
///
/// Write pipeline (per tile):
///  1. Walsh-Hadamard rotation of every token vector along the channel axis.
///     The normalized WHT is orthonormal and self-inverse, so attention
///     scores are preserved: (Hq)·(Hk) = q·k. Both K and V are rotated —
///     see the V note below. Unlike TurboQuant no per-layer sign flip is
///     applied: the Sinkhorn step below replaces it as the outlier equalizer,
///     and skipping it keeps the rotation a pure involution.
///  2. Sinkhorn variance normalization (<see cref="SinkhornNormalize"/>):
///     alternate per-channel (column) and per-token (row) RMS normalization,
///     accumulating the factors in log space. A few iterations drive both
///     axes toward unit variance, so the fixed low-bit RTN grid no longer has
///     to cover per-token or per-channel scale outliers (the error source
///     KVarN identifies as the driver of long-decode drift).
///  3. Asymmetric RTN: keys per-channel (min/step over the tile's 128 tokens,
///     i.e. group size 128 along the token axis), values per-token in channel
///     groups of 128 (clamped to headDim when headDim &lt; 128). Codes are
///     unsigned (K: 0..15, V: 0..3); dequant is <c>min + step · code</c>.
///  4. Scale folding: the Sinkhorn factor of the quantized axis is folded
///     into the stored RTN affine (K: channel factor into step/min; V: token
///     factor into step/min), so read paths apply exactly one leftover factor
///     per element — rowScale_t for K, colScale_c for V:
///       score_t = rowScale_t · Σ_c q'_c · (chanMin_c + chanStep_c · code)
///       out'_c += colScale_c · Σ_t p_t · (tokMin_tg + tokStep_tg · code)
///     For K the per-channel terms fold into the query once per tile
///     (<see cref="KeyScores"/>), not once per token.
///
/// Why V is rotated too: V never meets a rotated query — it is aggregated by
/// softmax weights — so rotation is not needed for score preservation. It is
/// kept because (a) at 2 bits a per-token min/max grid is dominated by the
/// token's largest-magnitude channels, and Sinkhorn's column factors are
/// tile-wide: they equalize channel variance across the tile but cannot
/// spread one token's energy concentration, while the Hadamard mixes each
/// token's channels and tightens its range toward Gaussian; (b) aggregation
/// is linear, so all tiles accumulate in the rotated domain and the inverse
/// transform runs once per head per decode step
/// (<see cref="UnrotateOutput"/>) — the same deferred-inverse pattern the
/// FastScan V path uses; (c) it keeps the K and V write pipelines identical.
/// The aggregate-parity and round-trip tests prove the un-rotation is exact.
///
/// Instances are not thread-safe for concurrent Compress* calls (they share
/// per-instance scratch buffers, allocated once at construction). The read
/// methods are stateless (stackalloc only) and safe to call concurrently.
/// </summary>
public sealed class KVarNCompressor
{
    /// <summary>Tokens per tile. Matches the paper's vLLM block size and the K quant group.</summary>
    public const int TileTokens = 128;

    /// <summary>Channel group size for per-token V quantization (clamped to headDim).</summary>
    public const int ValueGroupSize = 128;

    /// <summary>Key quantization bit width.</summary>
    public const int KeyBits = 4;

    /// <summary>Value quantization bit width.</summary>
    public const int ValueBits = 2;

    /// <summary>Default Sinkhorn iteration count (paper: ~3-5 suffice).</summary>
    public const int DefaultSinkhornIterations = 4;

    // Rows/columns whose RMS falls at or below this are left unscaled (their
    // log factor contribution is 0), so all-zero rows/columns survive the
    // normalization untouched and reconstruct exactly.
    private const float RmsEpsilon = 1e-12f;

    /// <summary>
    /// Test hook: forces the scalar read kernels (<see cref="KeyScores"/> /
    /// <see cref="AggregateValues"/>) even when AVX2 is available so the parity
    /// suite and micro-benchmarks can compare both implementations on the same
    /// hardware. Not intended for production use; both paths compute the same
    /// algebra (the AVX2 path reassociates the floating-point accumulation and
    /// contracts mul+add into FMA — one rounding instead of two).
    /// </summary>
    internal static bool ForceScalar;

    private readonly int _headDim;
    private readonly int _valueGroups;
    private readonly int _sinkhornIterations;

    // Scratch reused across Compress* calls (write path only; see thread-safety note).
    private readonly float[] _work;        // TileTokens × headDim rotated/normalized tile
    private readonly float[] _axisScratch; // Sinkhorn factors of the folded-away axis

    // ── Packed K tile layout (channel-major codes) ──────────────────────────
    //
    //   rowScale[T]  : T × f32 — per-token Sinkhorn factor (the only per-token
    //                  multiply left on the score path)
    //   chanStep[D]  : D × f32 — colScale_c · rtnStep_c (Sinkhorn channel
    //                  factor folded into the RTN step)
    //   chanMin[D]   : D × f32 — colScale_c · rtnMin_c (folded asymmetric
    //                  zero-point, stored as a real-valued minimum)
    //   codes        : D contiguous runs of T/2 bytes, channel-major. Run c
    //                  holds channel c's 128 4-bit codes across the tile's
    //                  tokens (token t in nibble t of the run: even t → low
    //                  nibble of byte t/2, odd t → high nibble — the
    //                  BitPacking.PackBits4 convention), so a tile-scores
    //                  kernel walks each channel's codes sequentially while
    //                  accumulating into scores[0..127] — the FastScan-style
    //                  loop inversion <see cref="KeyScoresAvx2"/> vectorizes
    //                  along tokens (P1).
    //
    //   dequant:  k̂'[t,c] = rowScale[t] · (chanMin[c] + chanStep[c] · code[c,t])
    //
    //   Total bytes = 4·T + 8·D + D·T/2. For T=128, D=128: 9728 bytes for
    //   16384 elements → 4.75 effective bits/element (scales are f32 in this
    //   P0 reference; f16 scales in P1 would shave ~0.7 bits).
    private readonly int _kChanStepF32;  // f32 index of chanStep[]
    private readonly int _kChanMinF32;   // f32 index of chanMin[]
    private readonly int _kCodesOffset;  // byte offset of codes

    // ── Packed V tile layout (token-major codes) ────────────────────────────
    //
    //   colScale[D]   : D × f32 — per-channel Sinkhorn factor (the only
    //                   per-channel multiply left on the aggregate path)
    //   tokStep[T·G]  : T·G × f32 — rowScale_t · rtnStep_tg folded, token-major
    //                   (index t·G + g), G = ceil(D / ValueGroupSize)
    //   tokMin[T·G]   : T·G × f32 — rowScale_t · rtnMin_tg folded
    //   codes         : T contiguous runs of D/4 bytes, token-major. Run t
    //                   holds token t's D 2-bit codes (channel c in bits
    //                   (c&amp;3)·2 of byte c/4 — the BitPacking.PackBits2
    //                   convention), so the V-aggregate walks token rows
    //                   sequentially: outer loop over tokens, broadcast
    //                   p_t · tokStep, inner run walk over channels.
    //
    //   dequant:  v̂'[t,c] = colScale[c] · (tokMin[t,g(c)] + tokStep[t,g(c)] · code[t,c])
    //
    //   Total bytes = 4·D + 8·T·G + T·D/4. For T=128, D=128 (G=1): 5632 bytes
    //   for 16384 elements → 2.75 effective bits/element.
    private readonly int _vTokStepF32;   // f32 index of tokStep[]
    private readonly int _vTokMinF32;    // f32 index of tokMin[]
    private readonly int _vCodesOffset;  // byte offset of codes

    /// <summary>Head dimension (channels per token).</summary>
    public int HeadDim => _headDim;

    /// <summary>Bytes per compressed K tile (128 tokens × headDim, 4-bit).</summary>
    public int KeyTileBytes { get; }

    /// <summary>Bytes per compressed V tile (128 tokens × headDim, 2-bit).</summary>
    public int ValueTileBytes { get; }

    public KVarNCompressor(int headDim, int sinkhornIterations = DefaultSinkhornIterations)
    {
        ValidateHeadDim(headDim);
        if (sinkhornIterations is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(sinkhornIterations), "Sinkhorn iterations must be in [1, 32].");

        _headDim = headDim;
        _sinkhornIterations = sinkhornIterations;
        _valueGroups = (headDim + ValueGroupSize - 1) / ValueGroupSize;

        _kChanStepF32 = TileTokens;
        _kChanMinF32 = TileTokens + headDim;
        _kCodesOffset = (TileTokens + 2 * headDim) * sizeof(float);
        KeyTileBytes = _kCodesOffset + headDim * (TileTokens / 2);

        _vTokStepF32 = headDim;
        _vTokMinF32 = headDim + TileTokens * _valueGroups;
        _vCodesOffset = (headDim + 2 * TileTokens * _valueGroups) * sizeof(float);
        ValueTileBytes = _vCodesOffset + TileTokens * (headDim / 4);

        _work = new float[TileTokens * headDim];
        _axisScratch = new float[Math.Max(TileTokens, headDim)];
    }

    /// <summary>Bytes per compressed K tile for the given head dimension.</summary>
    public static int KeyTileSize(int headDim)
    {
        ValidateHeadDim(headDim);
        return (TileTokens + 2 * headDim) * sizeof(float) + headDim * (TileTokens / 2);
    }

    /// <summary>Bytes per compressed V tile for the given head dimension.</summary>
    public static int ValueTileSize(int headDim)
    {
        ValidateHeadDim(headDim);
        int groups = (headDim + ValueGroupSize - 1) / ValueGroupSize;
        return (headDim + 2 * TileTokens * groups) * sizeof(float) + TileTokens * (headDim / 4);
    }

    private static void ValidateHeadDim(int headDim)
    {
        if (!BitOperations.IsPow2(headDim) || headDim is < 8 or > 1024)
            throw new ArgumentException("Head dimension must be a power of 2 in [8, 1024].", nameof(headDim));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Sinkhorn variance normalization
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Dual-axis variance normalization: alternately divides each channel
    /// (column) and each token (row) of the tile by its RMS, accumulating the
    /// factors in log space, so a few iterations drive both axes toward unit
    /// variance. RMS (std about zero) is used rather than mean-centered std so
    /// the normalization stays a pure per-row/per-column scaling — the only
    /// form that folds back into dot products and weighted sums exactly.
    /// Rows/columns with RMS ≤ epsilon (e.g. all-zero) are left unscaled.
    ///
    /// On return: original[t,c] == tile[t,c] · rowScaleOut[t] · colScaleOut[c]
    /// (up to fp rounding). Columns are normalized first and rows last, so the
    /// token axis — whose scale outliers drive decode drift — ends exactly
    /// unit-RMS.
    /// </summary>
    /// <param name="tile">Row-major [tokens, dim] tile, normalized in place.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="dim">Column count.</param>
    /// <param name="rowScaleOut">Receives the per-token factors (length ≥ tokens).</param>
    /// <param name="colScaleOut">Receives the per-channel factors (length ≥ dim).</param>
    /// <param name="iterations">Alternation count (~3-5 suffice).</param>
    public static void SinkhornNormalize(
        Span<float> tile,
        int tokens,
        int dim,
        Span<float> rowScaleOut,
        Span<float> colScaleOut,
        int iterations)
    {
        if (tokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), "tokens must be positive.");
        if (dim <= 0)
            throw new ArgumentOutOfRangeException(nameof(dim), "dim must be positive.");
        if (tile.Length < tokens * dim)
            throw new ArgumentException($"tile must hold {tokens}×{dim} floats.", nameof(tile));
        if (rowScaleOut.Length < tokens)
            throw new ArgumentException($"rowScaleOut must hold {tokens} floats.", nameof(rowScaleOut));
        if (colScaleOut.Length < dim)
            throw new ArgumentException($"colScaleOut must hold {dim} floats.", nameof(colScaleOut));
        if (iterations < 1)
            throw new ArgumentOutOfRangeException(nameof(iterations));

        Span<float> logRow = rowScaleOut.Slice(0, tokens);
        Span<float> logCol = colScaleOut.Slice(0, dim);
        logRow.Clear();
        logCol.Clear();

        for (int iter = 0; iter < iterations; iter++)
        {
            // Channels (columns) first ...
            for (int c = 0; c < dim; c++)
            {
                float sumSq = 0f;
                for (int t = 0; t < tokens; t++)
                {
                    float v = tile[t * dim + c];
                    sumSq += v * v;
                }
                float rms = MathF.Sqrt(sumSq / tokens);
                if (rms <= RmsEpsilon)
                    continue;
                float inv = 1f / rms;
                for (int t = 0; t < tokens; t++)
                    tile[t * dim + c] *= inv;
                logCol[c] += MathF.Log(rms);
            }

            // ... tokens (rows) last, so per-token RMS ends exactly at 1.
            for (int t = 0; t < tokens; t++)
            {
                Span<float> row = tile.Slice(t * dim, dim);
                float sumSq = 0f;
                for (int c = 0; c < dim; c++)
                    sumSq += row[c] * row[c];
                float rms = MathF.Sqrt(sumSq / dim);
                if (rms <= RmsEpsilon)
                    continue;
                float inv = 1f / rms;
                for (int c = 0; c < dim; c++)
                    row[c] *= inv;
                logRow[t] += MathF.Log(rms);
            }
        }

        // Log domain → linear scale factors.
        for (int t = 0; t < tokens; t++)
            logRow[t] = MathF.Exp(logRow[t]);
        for (int c = 0; c < dim; c++)
            logCol[c] = MathF.Exp(logCol[c]);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Write path
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Compress one K tile: <see cref="TileTokens"/> key vectors (row-major
    /// [token, channel]) into the packed 4-bit per-channel layout.
    /// </summary>
    /// <param name="keys">TileTokens × headDim floats, original (un-rotated) domain.</param>
    /// <param name="tileOut">Destination, at least <see cref="KeyTileBytes"/> bytes.</param>
    public void CompressKeyTile(ReadOnlySpan<float> keys, Span<byte> tileOut)
    {
        int d = _headDim;
        if (keys.Length < TileTokens * d)
            throw new ArgumentException($"keys must hold {TileTokens}×{d} floats.", nameof(keys));
        if (tileOut.Length < KeyTileBytes)
            throw new ArgumentException($"tileOut must be at least {KeyTileBytes} bytes.", nameof(tileOut));

        Span<float> work = _work.AsSpan(0, TileTokens * d);
        RotateRows(keys, work, d);

        Span<float> f32 = MemoryMarshal.Cast<byte, float>(tileOut);
        Span<float> rowScale = f32.Slice(0, TileTokens);            // stored raw
        Span<float> colScale = _axisScratch.AsSpan(0, d);           // folded below
        SinkhornNormalize(work, TileTokens, d, rowScale, colScale, _sinkhornIterations);

        Span<float> chanStep = f32.Slice(_kChanStepF32, d);
        Span<float> chanMin = f32.Slice(_kChanMinF32, d);
        Span<byte> codes = tileOut.Slice(_kCodesOffset, d * (TileTokens / 2));

        const int levels = (1 << KeyBits) - 1; // 15
        for (int c = 0; c < d; c++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int t = 0; t < TileTokens; t++)
            {
                float v = work[t * d + c];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float step = (max - min) / levels;
            float invStep = step > 0f ? 1f / step : 0f;
            chanStep[c] = colScale[c] * step;
            chanMin[c] = colScale[c] * min;

            Span<byte> run = codes.Slice(c * (TileTokens / 2), TileTokens / 2);
            for (int t = 0; t < TileTokens; t++)
            {
                int q = (int)MathF.Round((work[t * d + c] - min) * invStep);
                if (q < 0) q = 0;
                else if (q > levels) q = levels;
                BitPacking.PackBits4(run, 0, t, q);
            }
        }
    }

    /// <summary>
    /// Compress one V tile: <see cref="TileTokens"/> value vectors (row-major
    /// [token, channel]) into the packed 2-bit per-token layout.
    /// </summary>
    /// <param name="values">TileTokens × headDim floats, original (un-rotated) domain.</param>
    /// <param name="tileOut">Destination, at least <see cref="ValueTileBytes"/> bytes.</param>
    public void CompressValueTile(ReadOnlySpan<float> values, Span<byte> tileOut)
    {
        int d = _headDim;
        if (values.Length < TileTokens * d)
            throw new ArgumentException($"values must hold {TileTokens}×{d} floats.", nameof(values));
        if (tileOut.Length < ValueTileBytes)
            throw new ArgumentException($"tileOut must be at least {ValueTileBytes} bytes.", nameof(tileOut));

        Span<float> work = _work.AsSpan(0, TileTokens * d);
        RotateRows(values, work, d);

        Span<float> f32 = MemoryMarshal.Cast<byte, float>(tileOut);
        Span<float> colScale = f32.Slice(0, d);                     // stored raw
        Span<float> rowScale = _axisScratch.AsSpan(0, TileTokens);  // folded below
        SinkhornNormalize(work, TileTokens, d, rowScale, colScale, _sinkhornIterations);

        int groups = _valueGroups;
        Span<float> tokStep = f32.Slice(_vTokStepF32, TileTokens * groups);
        Span<float> tokMin = f32.Slice(_vTokMinF32, TileTokens * groups);
        Span<byte> codes = tileOut.Slice(_vCodesOffset, TileTokens * (d / 4));

        const int levels = (1 << ValueBits) - 1; // 3
        for (int t = 0; t < TileTokens; t++)
        {
            Span<float> row = work.Slice(t * d, d);
            Span<byte> run = codes.Slice(t * (d / 4), d / 4);

            for (int g = 0; g < groups; g++)
            {
                int c0 = g * ValueGroupSize;
                int c1 = Math.Min(c0 + ValueGroupSize, d);

                float min = float.MaxValue, max = float.MinValue;
                for (int c = c0; c < c1; c++)
                {
                    float v = row[c];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                float step = (max - min) / levels;
                float invStep = step > 0f ? 1f / step : 0f;
                tokStep[t * groups + g] = rowScale[t] * step;
                tokMin[t * groups + g] = rowScale[t] * min;

                for (int c = c0; c < c1; c++)
                {
                    int q = (int)MathF.Round((row[c] - min) * invStep);
                    if (q < 0) q = 0;
                    else if (q > levels) q = levels;
                    BitPacking.PackBits2(run, 0, c, q);
                }
            }
        }
    }

    /// <summary>Rotate each token row along the channel axis (normalized WHT).</summary>
    private static void RotateRows(ReadOnlySpan<float> source, Span<float> dest, int dim)
    {
        for (int t = 0; t < TileTokens; t++)
            WalshHadamard.Transform(source.Slice(t * dim, dim), dest.Slice(t * dim, dim), dim);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Read path
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Pre-rotate a query vector for use with <see cref="KeyScores"/> /
    /// <see cref="DequantDot"/>. Call once per head, reuse across all tiles.
    /// </summary>
    public void RotateQuery(ReadOnlySpan<float> query, Span<float> rotatedQuery)
    {
        if (query.Length < _headDim)
            throw new ArgumentException($"query must hold {_headDim} floats.", nameof(query));
        if (rotatedQuery.Length < _headDim)
            throw new ArgumentException($"rotatedQuery must hold {_headDim} floats.", nameof(rotatedQuery));
        WalshHadamard.Transform(query, rotatedQuery, _headDim);
    }

    /// <summary>
    /// Compute q·k̂ for every token in a K tile. The per-channel factors
    /// (chanStep/chanMin, which already carry the Sinkhorn channel scale) are
    /// folded into the query once for the whole tile; the code walk is
    /// channel-major, matching the packed layout. Dispatches to a fused AVX2
    /// kernel when available (<see cref="KeyScoresAvx2"/>); the scalar path is
    /// the reference implementation. The two paths differ only in
    /// floating-point accumulation order.
    /// </summary>
    /// <param name="tile">Compressed K tile.</param>
    /// <param name="rotatedQuery">Pre-rotated query (<see cref="RotateQuery"/>).</param>
    /// <param name="scoresOut">Receives <see cref="TileTokens"/> scores.</param>
    public void KeyScores(ReadOnlySpan<byte> tile, ReadOnlySpan<float> rotatedQuery, Span<float> scoresOut)
    {
        int d = _headDim;
        if (tile.Length < KeyTileBytes)
            throw new ArgumentException($"tile must be at least {KeyTileBytes} bytes.", nameof(tile));
        if (rotatedQuery.Length < d)
            throw new ArgumentException($"rotatedQuery must hold {d} floats.", nameof(rotatedQuery));
        if (scoresOut.Length < TileTokens)
            throw new ArgumentException($"scoresOut must hold {TileTokens} floats.", nameof(scoresOut));

        if (Avx2.IsSupported && Fma.IsSupported && !ForceScalar)
            KeyScoresAvx2(tile, rotatedQuery, scoresOut);
        else
            KeyScoresScalar(tile, rotatedQuery, scoresOut);
    }

    private void KeyScoresScalar(ReadOnlySpan<byte> tile, ReadOnlySpan<float> rotatedQuery, Span<float> scoresOut)
    {
        int d = _headDim;
        ReadOnlySpan<float> f32 = MemoryMarshal.Cast<byte, float>(tile);
        ReadOnlySpan<float> rowScale = f32.Slice(0, TileTokens);
        ReadOnlySpan<float> chanStep = f32.Slice(_kChanStepF32, d);
        ReadOnlySpan<float> chanMin = f32.Slice(_kChanMinF32, d);
        ReadOnlySpan<byte> codes = tile.Slice(_kCodesOffset, d * (TileTokens / 2));

        // Fold per-channel factors into the query once per tile:
        //   score_t = rowScale_t · (Σ_c q_c·chanMin_c  +  Σ_c (q_c·chanStep_c)·code[c,t])
        Span<float> acc = stackalloc float[TileTokens];
        acc.Clear();
        float bias = 0f;

        for (int c = 0; c < d; c++)
        {
            float q = rotatedQuery[c];
            bias += q * chanMin[c];
            float qw = q * chanStep[c];
            if (qw == 0f)
                continue;

            ReadOnlySpan<byte> run = codes.Slice(c * (TileTokens / 2), TileTokens / 2);
            for (int b = 0; b < TileTokens / 2; b++)
            {
                byte pair = run[b];
                acc[b * 2] += qw * (pair & 0x0F);
                acc[b * 2 + 1] += qw * ((pair >> 4) & 0x0F);
            }
        }

        for (int t = 0; t < TileTokens; t++)
            scoresOut[t] = rowScale[t] * (acc[t] + bias);
    }

    /// <summary>
    /// AVX2 K-score kernel. Same algebra as <see cref="KeyScoresScalar"/> —
    /// per-channel factors folded into the query, channel-major code walk —
    /// but the 128 token scores are held in Vector256 accumulators, processed
    /// in four chunks of 32 tokens so the working set (4 accumulators + unpack
    /// temporaries) stays within the 16 YMM registers. Per channel and chunk:
    /// one 16-byte load covers 32 nibbles (byte b = tokens 2b low / 2b+1 high,
    /// the PackBits4 convention); low/high nibble extraction + byte interleave
    /// (UnpackLow/High) restores token order, then vpmovzxbd + cvtdq2ps widens
    /// to f32 for four FMAs against the broadcast folded query weight.
    /// Numerics differ from scalar by FP reassociation + FMA contraction only.
    /// </summary>
    private unsafe void KeyScoresAvx2(ReadOnlySpan<byte> tile, ReadOnlySpan<float> rotatedQuery, Span<float> scoresOut)
    {
        int d = _headDim;
        const int runBytes = TileTokens / 2;   // 64 code bytes per channel
        const int chunkBytes = 16;             // bytes per chunk = 32 tokens

        fixed (byte* tilePtr = tile)
        fixed (float* qPtr = rotatedQuery)
        fixed (float* scoresPtr = scoresOut)
        {
            float* f32 = (float*)tilePtr;
            float* rowScale = f32;                 // [TileTokens]
            float* chanStep = f32 + _kChanStepF32; // [d]
            float* chanMin = f32 + _kChanMinF32;   // [d]
            byte* codes = tilePtr + _kCodesOffset;

            // bias = Σ_c q_c · chanMin_c — vectorized; d is a power of 2 ≥ 8,
            // so the 8-lane loop has no tail.
            var biasAcc = Vector256<float>.Zero;
            for (int c = 0; c + 8 <= d; c += 8)
                biasAcc = Fma.MultiplyAdd(Avx.LoadVector256(qPtr + c), Avx.LoadVector256(chanMin + c), biasAcc);
            float bias = Vector256.Sum(biasAcc);

            float* acc = stackalloc float[TileTokens];
            var nibbleMask = Vector128.Create((byte)0x0F);

            for (int chunk = 0; chunk < TileTokens / 32; chunk++)
            {
                byte* chunkCodes = codes + chunk * chunkBytes;
                var a0 = Vector256<float>.Zero;
                var a1 = Vector256<float>.Zero;
                var a2 = Vector256<float>.Zero;
                var a3 = Vector256<float>.Zero;

                for (int c = 0; c < d; c++)
                {
                    float qw = qPtr[c] * chanStep[c];
                    if (qw == 0f)
                        continue;

                    var pairs = Sse2.LoadVector128(chunkCodes + c * runBytes);
                    var lo = Sse2.And(pairs, nibbleMask); // even tokens
                    var hi = Sse2.And(                     // odd tokens
                        Sse2.ShiftRightLogical(pairs.AsInt16(), 4).AsByte(),
                        nibbleMask);
                    var t01 = Sse2.UnpackLow(lo, hi);  // chunk tokens 0..15, in order
                    var t23 = Sse2.UnpackHigh(lo, hi); // chunk tokens 16..31

                    var qwv = Vector256.Create(qw);
                    a0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(t01)), qwv, a0);
                    a1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(
                        Sse2.ShiftRightLogical128BitLane(t01, 8))), qwv, a1);
                    a2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(t23)), qwv, a2);
                    a3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(
                        Sse2.ShiftRightLogical128BitLane(t23, 8))), qwv, a3);
                }

                int baseT = chunk * 32;
                Avx.Store(acc + baseT, a0);
                Avx.Store(acc + baseT + 8, a1);
                Avx.Store(acc + baseT + 16, a2);
                Avx.Store(acc + baseT + 24, a3);
            }

            // score_t = rowScale_t · (acc_t + bias)
            var biasV = Vector256.Create(bias);
            for (int t = 0; t < TileTokens; t += 8)
            {
                Avx.Store(scoresPtr + t, Avx.Multiply(
                    Avx.LoadVector256(rowScale + t),
                    Avx.Add(Avx.LoadVector256(acc + t), biasV)));
            }
        }
    }

    /// <summary>
    /// Compute q·k̂ for a single token of a K tile. Equivalent to
    /// dot(query, decompressed row); prefer <see cref="KeyScores"/> when
    /// scoring the whole tile so the channel folding is amortized.
    /// </summary>
    public float DequantDot(ReadOnlySpan<byte> tile, ReadOnlySpan<float> rotatedQuery, int tokenIndex)
    {
        int d = _headDim;
        if (tile.Length < KeyTileBytes)
            throw new ArgumentException($"tile must be at least {KeyTileBytes} bytes.", nameof(tile));
        if (rotatedQuery.Length < d)
            throw new ArgumentException($"rotatedQuery must hold {d} floats.", nameof(rotatedQuery));
        if ((uint)tokenIndex >= TileTokens)
            throw new ArgumentOutOfRangeException(nameof(tokenIndex));

        ReadOnlySpan<float> f32 = MemoryMarshal.Cast<byte, float>(tile);
        ReadOnlySpan<float> chanStep = f32.Slice(_kChanStepF32, d);
        ReadOnlySpan<float> chanMin = f32.Slice(_kChanMinF32, d);
        ReadOnlySpan<byte> codes = tile.Slice(_kCodesOffset, d * (TileTokens / 2));

        int byteInRun = tokenIndex >> 1;
        bool high = (tokenIndex & 1) != 0;

        float dot = 0f;
        for (int c = 0; c < d; c++)
        {
            byte pair = codes[c * (TileTokens / 2) + byteInRun];
            int code = high ? (pair >> 4) & 0x0F : pair & 0x0F;
            dot += rotatedQuery[c] * (chanMin[c] + chanStep[c] * code);
        }

        return f32[tokenIndex] * dot; // rowScale[tokenIndex]
    }

    /// <summary>
    /// Accumulate the softmax-weighted V aggregate for one tile into
    /// <paramref name="rotatedAccInOut"/> (rotated domain). Tiles share the
    /// same rotation, so fold any number of tiles into the same accumulator
    /// and call <see cref="UnrotateOutput"/> once at the end. The per-token
    /// factors (tokStep/tokMin, which already carry the Sinkhorn token scale)
    /// are folded into the weight once per token; the code walk is
    /// token-major, matching the packed layout.
    /// </summary>
    /// <param name="tile">Compressed V tile.</param>
    /// <param name="weights">Softmax weights, one per tile token.</param>
    /// <param name="rotatedAccInOut">Accumulator of headDim floats (rotated domain).</param>
    public void AggregateValues(ReadOnlySpan<byte> tile, ReadOnlySpan<float> weights, Span<float> rotatedAccInOut)
    {
        int d = _headDim;
        if (tile.Length < ValueTileBytes)
            throw new ArgumentException($"tile must be at least {ValueTileBytes} bytes.", nameof(tile));
        if (weights.Length < TileTokens)
            throw new ArgumentException($"weights must hold {TileTokens} floats.", nameof(weights));
        if (rotatedAccInOut.Length < d)
            throw new ArgumentException($"rotatedAccInOut must hold {d} floats.", nameof(rotatedAccInOut));

        // The AVX2 kernel consumes 16 code bytes (64 channels) per step, so it
        // needs every channel group to span a multiple of 64 channels — true
        // for all shipping head dims (64/128/256); d ∈ {8,16,32} falls back.
        if (Avx2.IsSupported && Fma.IsSupported && !ForceScalar && (d & 63) == 0)
            AggregateValuesAvx2(tile, weights, rotatedAccInOut);
        else
            AggregateValuesScalar(tile, weights, rotatedAccInOut);
    }

    private void AggregateValuesScalar(ReadOnlySpan<byte> tile, ReadOnlySpan<float> weights, Span<float> rotatedAccInOut)
    {
        int d = _headDim;
        int groups = _valueGroups;
        ReadOnlySpan<float> f32 = MemoryMarshal.Cast<byte, float>(tile);
        ReadOnlySpan<float> colScale = f32.Slice(0, d);
        ReadOnlySpan<float> tokStep = f32.Slice(_vTokStepF32, TileTokens * groups);
        ReadOnlySpan<float> tokMin = f32.Slice(_vTokMinF32, TileTokens * groups);
        ReadOnlySpan<byte> codes = tile.Slice(_vCodesOffset, TileTokens * (d / 4));

        //   out'_c += colScale_c · ( Σ_t w_t·tokMin_tg  +  Σ_t (w_t·tokStep_tg)·code[t,c] )
        // The min term is constant per channel group, so it accumulates once
        // per (token, group) instead of once per element.
        Span<float> acc = stackalloc float[d];
        acc.Clear();
        Span<float> groupBias = stackalloc float[groups];
        groupBias.Clear();

        int bytesPerToken = d / 4;
        for (int t = 0; t < TileTokens; t++)
        {
            float w = weights[t];
            if (w == 0f)
                continue;

            ReadOnlySpan<byte> run = codes.Slice(t * bytesPerToken, bytesPerToken);
            for (int g = 0; g < groups; g++)
            {
                groupBias[g] += w * tokMin[t * groups + g];
                float ws = w * tokStep[t * groups + g];
                if (ws == 0f)
                    continue;

                int b0 = (g * ValueGroupSize) >> 2;
                int b1 = Math.Min((g + 1) * ValueGroupSize, d) >> 2;
                for (int b = b0; b < b1; b++)
                {
                    byte quad = run[b];
                    int c = b << 2;
                    acc[c] += ws * (quad & 0x3);
                    acc[c + 1] += ws * ((quad >> 2) & 0x3);
                    acc[c + 2] += ws * ((quad >> 4) & 0x3);
                    acc[c + 3] += ws * ((quad >> 6) & 0x3);
                }
            }
        }

        for (int c = 0; c < d; c++)
            rotatedAccInOut[c] += colScale[c] * (acc[c] + groupBias[c / ValueGroupSize]);
    }

    /// <summary>
    /// AVX2 V-aggregate kernel. Same algebra and skip semantics as
    /// <see cref="AggregateValuesScalar"/> (w == 0 tokens and ws == 0 groups
    /// are skipped; the group-min term accumulates once per token/group), but
    /// the token-major code walk unpacks 16 bytes = 64 channels per step: four
    /// crumb planes (bits 0-1/2-3/4-5/6-7, the PackBits2 convention) are
    /// extracted with shift+mask, then a two-level byte/word interleave
    /// (UnpackLow/High) restores channel order before vpmovzxbd + cvtdq2ps and
    /// an FMA against the broadcast w·tokStep into the stack accumulator.
    /// Requires d to be a multiple of 64 (see the dispatch in
    /// <see cref="AggregateValues"/>). Numerics differ from scalar by FP
    /// reassociation + FMA contraction only.
    /// </summary>
    private unsafe void AggregateValuesAvx2(ReadOnlySpan<byte> tile, ReadOnlySpan<float> weights, Span<float> rotatedAccInOut)
    {
        int d = _headDim;
        int groups = _valueGroups;
        int bytesPerToken = d / 4;

        fixed (byte* tilePtr = tile)
        fixed (float* wPtr = weights)
        fixed (float* outPtr = rotatedAccInOut)
        {
            float* f32 = (float*)tilePtr;
            float* colScale = f32;                // [d]
            float* tokStep = f32 + _vTokStepF32;  // [T·G]
            float* tokMin = f32 + _vTokMinF32;    // [T·G]
            byte* codes = tilePtr + _vCodesOffset;

            float* acc = stackalloc float[d];     // d ≤ 1024 (validated) → ≤ 4 KiB
            new Span<float>(acc, d).Clear();
            float* groupBias = stackalloc float[groups];
            new Span<float>(groupBias, groups).Clear();

            var crumbMask = Vector128.Create((byte)0x03);

            for (int t = 0; t < TileTokens; t++)
            {
                float w = wPtr[t];
                if (w == 0f)
                    continue;

                byte* run = codes + t * bytesPerToken;
                for (int g = 0; g < groups; g++)
                {
                    groupBias[g] += w * tokMin[t * groups + g];
                    float ws = w * tokStep[t * groups + g];
                    if (ws == 0f)
                        continue;

                    var wsv = Vector256.Create(ws);
                    int b0 = (g * ValueGroupSize) >> 2;
                    int b1 = Math.Min((g + 1) * ValueGroupSize, d) >> 2;
                    for (int b = b0; b < b1; b += 16)
                    {
                        // Byte j of the load holds channels 4j..4j+3 in crumbs.
                        var quads = Sse2.LoadVector128(run + b);
                        var c0 = Sse2.And(quads, crumbMask);
                        var c1 = Sse2.And(Sse2.ShiftRightLogical(quads.AsInt16(), 2).AsByte(), crumbMask);
                        var c2 = Sse2.And(Sse2.ShiftRightLogical(quads.AsInt16(), 4).AsByte(), crumbMask);
                        var c3 = Sse2.And(Sse2.ShiftRightLogical(quads.AsInt16(), 6).AsByte(), crumbMask);

                        // Byte interleave → channel pairs (4j, 4j+1) / (4j+2, 4j+3),
                        // then word interleave → channels in order, 16 per vector.
                        var p01Lo = Sse2.UnpackLow(c0, c1);
                        var p23Lo = Sse2.UnpackLow(c2, c3);
                        var p01Hi = Sse2.UnpackHigh(c0, c1);
                        var p23Hi = Sse2.UnpackHigh(c2, c3);

                        var q0 = Sse2.UnpackLow(p01Lo.AsInt16(), p23Lo.AsInt16()).AsByte();  // ch +0..15
                        var q1 = Sse2.UnpackHigh(p01Lo.AsInt16(), p23Lo.AsInt16()).AsByte(); // ch +16..31
                        var q2 = Sse2.UnpackLow(p01Hi.AsInt16(), p23Hi.AsInt16()).AsByte();  // ch +32..47
                        var q3 = Sse2.UnpackHigh(p01Hi.AsInt16(), p23Hi.AsInt16()).AsByte(); // ch +48..63

                        float* accBase = acc + (b << 2);
                        FmaCodes16(accBase, q0, wsv);
                        FmaCodes16(accBase + 16, q1, wsv);
                        FmaCodes16(accBase + 32, q2, wsv);
                        FmaCodes16(accBase + 48, q3, wsv);
                    }
                }
            }

            // out_c += colScale_c · (acc_c + groupBias_{c/G}) — the group index
            // is constant across each 8-lane block because ValueGroupSize (128,
            // clamped to d ≥ 64 here) is a multiple of 8.
            for (int c = 0; c < d; c += 8)
            {
                var gb = Vector256.Create(groupBias[c / ValueGroupSize]);
                var v = Avx.Add(Avx.LoadVector256(acc + c), gb);
                var prev = Avx.LoadVector256(outPtr + c);
                Avx.Store(outPtr + c, Fma.MultiplyAdd(Avx.LoadVector256(colScale + c), v, prev));
            }
        }
    }

    /// <summary>
    /// Widen 16 in-order code bytes to f32 and FMA them (× the broadcast
    /// weight) into 16 consecutive accumulator floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void FmaCodes16(float* accBase, Vector128<byte> codes16, Vector256<float> wsv)
    {
        var lo = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(codes16));
        var hi = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(
            Sse2.ShiftRightLogical128BitLane(codes16, 8)));
        Avx.Store(accBase, Fma.MultiplyAdd(lo, wsv, Avx.LoadVector256(accBase)));
        Avx.Store(accBase + 8, Fma.MultiplyAdd(hi, wsv, Avx.LoadVector256(accBase + 8)));
    }

    /// <summary>
    /// Undo the channel rotation on a V aggregate accumulated via
    /// <see cref="AggregateValues"/>. Call once per head after all tiles have
    /// been folded in (the normalized WHT is its own inverse).
    /// </summary>
    public void UnrotateOutput(Span<float> rotatedAcc)
    {
        if (rotatedAcc.Length < _headDim)
            throw new ArgumentException($"rotatedAcc must hold {_headDim} floats.", nameof(rotatedAcc));
        WalshHadamard.Transform(rotatedAcc.Slice(0, _headDim), rotatedAcc.Slice(0, _headDim), _headDim);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Debug / test path
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reconstruct a K tile to full precision (original, un-rotated domain).
    /// Test/debug path — the hot path never materializes this.
    /// </summary>
    /// <param name="tile">Compressed K tile.</param>
    /// <param name="keysOut">TileTokens × headDim floats, row-major.</param>
    public void DecompressKeyTile(ReadOnlySpan<byte> tile, Span<float> keysOut)
    {
        int d = _headDim;
        if (tile.Length < KeyTileBytes)
            throw new ArgumentException($"tile must be at least {KeyTileBytes} bytes.", nameof(tile));
        if (keysOut.Length < TileTokens * d)
            throw new ArgumentException($"keysOut must hold {TileTokens}×{d} floats.", nameof(keysOut));

        ReadOnlySpan<float> f32 = MemoryMarshal.Cast<byte, float>(tile);
        ReadOnlySpan<float> rowScale = f32.Slice(0, TileTokens);
        ReadOnlySpan<float> chanStep = f32.Slice(_kChanStepF32, d);
        ReadOnlySpan<float> chanMin = f32.Slice(_kChanMinF32, d);
        ReadOnlySpan<byte> codes = tile.Slice(_kCodesOffset, d * (TileTokens / 2));

        for (int t = 0; t < TileTokens; t++)
        {
            Span<float> row = keysOut.Slice(t * d, d);
            int byteInRun = t >> 1;
            bool high = (t & 1) != 0;

            for (int c = 0; c < d; c++)
            {
                byte pair = codes[c * (TileTokens / 2) + byteInRun];
                int code = high ? (pair >> 4) & 0x0F : pair & 0x0F;
                row[c] = rowScale[t] * (chanMin[c] + chanStep[c] * code);
            }

            WalshHadamard.Transform(row, row, d); // involution → inverse rotation
        }
    }

    /// <summary>
    /// Reconstruct a V tile to full precision (original, un-rotated domain).
    /// Test/debug path — the hot path never materializes this.
    /// </summary>
    /// <param name="tile">Compressed V tile.</param>
    /// <param name="valuesOut">TileTokens × headDim floats, row-major.</param>
    public void DecompressValueTile(ReadOnlySpan<byte> tile, Span<float> valuesOut)
    {
        int d = _headDim;
        int groups = _valueGroups;
        if (tile.Length < ValueTileBytes)
            throw new ArgumentException($"tile must be at least {ValueTileBytes} bytes.", nameof(tile));
        if (valuesOut.Length < TileTokens * d)
            throw new ArgumentException($"valuesOut must hold {TileTokens}×{d} floats.", nameof(valuesOut));

        ReadOnlySpan<float> f32 = MemoryMarshal.Cast<byte, float>(tile);
        ReadOnlySpan<float> colScale = f32.Slice(0, d);
        ReadOnlySpan<float> tokStep = f32.Slice(_vTokStepF32, TileTokens * groups);
        ReadOnlySpan<float> tokMin = f32.Slice(_vTokMinF32, TileTokens * groups);
        ReadOnlySpan<byte> codes = tile.Slice(_vCodesOffset, TileTokens * (d / 4));

        int bytesPerToken = d / 4;
        for (int t = 0; t < TileTokens; t++)
        {
            Span<float> row = valuesOut.Slice(t * d, d);
            ReadOnlySpan<byte> run = codes.Slice(t * bytesPerToken, bytesPerToken);

            for (int c = 0; c < d; c++)
            {
                int g = c / ValueGroupSize;
                int code = BitPacking.UnpackBits2(run, 0, c);
                row[c] = colScale[c] * (tokMin[t * groups + g] + tokStep[t * groups + g] * code);
            }

            WalshHadamard.Transform(row, row, d);
        }
    }
}
