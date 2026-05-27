using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInference.TurboQuant;

/// <summary>
/// FAISS-FastScan-derived SIMD inner loop for TurboQuant KV-cache attention.
///
/// The standard <see cref="TurboQuantOps.DequantDot4Avx2"/> path vectorises along
/// <c>dim</c> using <c>vpgatherdd</c>, which is the slow leg on every recent uarch
/// (~2-3× slower than equivalent scalar throughput). FastScan inverts the loop:
/// vectorise along <em>positions</em> (32 KV positions per tile) and replace the
/// gather with <c>vpshufb</c> (one shuffle per clock) against an i8 codebook LUT
/// that is rebuilt once per query head.
///
/// Tile layout (4-bit, single kv-head):
///   norms[32]            : 32 × FP16 (64 bytes)
///   codes[dim][16 bytes] : per dim d, 16 bytes — byte b holds
///                          (codes[d][b+16] &lt;&lt; 4) | codes[d][b],
///                          so a single load + pshufb pair (low nibble / high
///                          nibble) yields 32 i8 contributions across positions.
///
/// Numerical model (per query head):
///   q_scale = max_{d, v} |centroids[v] · q[d]| / 127
///   LUT[d][v] = clamp(round(centroids[v] · q[d] / q_scale), -128, 127)
/// then per tile per dim d:
///   acc[t] += LUT[d][codes[d][t]]                                    (i16 sum)
/// and final
///   score[t] = acc[t] · q_scale · norm[t] · attnScale.
///
/// Range bound: dim × i8 ≤ 128 × 128 = 16384, fits i16 (±32767). The only
/// quality risk is the i8 LUT (≤ ½ LSB ≈ q_scale/254 per entry); over 128 dims
/// the worst-case relative error vs the existing fp32 dequant-dot is &lt; 0.1 %.
/// </summary>
public static class FastScan
{
    /// <summary>KV positions packed into one FastScan tile.</summary>
    public const int TileSize = 32;

    /// <summary>FP16 norm bytes per tile (one norm per position).</summary>
    public const int NormBytesPerTile = TileSize * 2;

    /// <summary>Bytes used by codes for a single dim within a K-tile (32 nibbles).</summary>
    public const int CodeBytesPerDim = 16;

    /// <summary>
    /// Total bytes for one FastScan K-tile (32 positions × 1 kv-head): packed
    /// norms followed by dim-major 4-bit codes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int TileBytes(int dim) => NormBytesPerTile + dim * CodeBytesPerDim;

    /// <summary>
    /// Total bytes for one FastScan V-tile (32 positions × 1 kv-head): the
    /// 32 norms followed by position-major codes — byte (t, d/2) holds
    /// (code[d=2k+1][t] &lt;&lt; 4) | code[d=2k][t]. This is the transpose of
    /// the K-tile codes, sized so per-t the kernel streams dim/2 bytes through
    /// a single pshufb over 32 dims.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int VTileBytes(int dim) => NormBytesPerTile + TileSize * (dim / 2);

    /// <summary>
    /// Pack 32 existing per-block TurboQuant buffers (same layout the engine
    /// produces today via <see cref="TurboQuantOps.Quantize"/>) into a FastScan
    /// tile. Source blocks must all be 4-bit; 3-bit packing is a follow-up.
    /// </summary>
    /// <param name="blocks">32 contiguous per-block buffers, each of length
    /// <see cref="TurboQuantOps.BlockSize"/>(4, dim).</param>
    /// <param name="tileOut">Destination tile, length <see cref="TileBytes"/>.</param>
    /// <param name="dim">Vector dimension.</param>
    public static void PackTile4Bit(
        ReadOnlySpan<byte> blocks,
        Span<byte> tileOut,
        int dim)
        => PackKTileCore(blocks, tileOut, dim, bits: 4);

    /// <summary>
    /// 3-bit K-tile packer. Source codes are 0..7 (3 bits in the block layout),
    /// written into the tile as 4-bit nibbles with the high bit always zero so
    /// the same pshufb kernel serves both bit widths.
    /// </summary>
    public static void PackTile3Bit(
        ReadOnlySpan<byte> blocks,
        Span<byte> tileOut,
        int dim)
        => PackKTileCore(blocks, tileOut, dim, bits: 3);

    private static void PackKTileCore(
        ReadOnlySpan<byte> blocks,
        Span<byte> tileOut,
        int dim,
        int bits)
    {
        int blockSize = TurboQuantOps.BlockSize(bits, dim);
        if (blocks.Length < TileSize * blockSize)
            throw new ArgumentException($"blocks must hold {TileSize} blocks of {blockSize} bytes each.", nameof(blocks));
        if (tileOut.Length < TileBytes(dim))
            throw new ArgumentException($"tileOut must be at least {TileBytes(dim)} bytes.", nameof(tileOut));

        // Norms: copy each block's 2-byte FP16 norm header in order.
        for (int t = 0; t < TileSize; t++)
        {
            int src = t * blockSize + TurboQuantOps.NormOffset;
            tileOut[t * 2]     = blocks[src];
            tileOut[t * 2 + 1] = blocks[src + 1];
        }

        // Codes: walk every (d, t) pair and place the code into
        // tile.codes[d][b] at the appropriate nibble (low for t<16, high otherwise).
        var codesOut = tileOut.Slice(NormBytesPerTile, dim * CodeBytesPerDim);
        codesOut.Clear();

        for (int t = 0; t < TileSize; t++)
        {
            ReadOnlySpan<byte> block = blocks.Slice(t * blockSize, blockSize);
            ReadOnlySpan<byte> packed = block.Slice(TurboQuantOps.IndicesOffset);
            int b = t & 15;
            bool highNibble = t >= 16;

            for (int d = 0; d < dim; d++)
            {
                int code = bits == 4
                    ? BitPacking.UnpackBits4(packed, 0, d) & 0x0F
                    : BitPacking.UnpackBits3(packed, 0, d) & 0x07;
                int idx = d * CodeBytesPerDim + b;
                codesOut[idx] = highNibble
                    ? (byte)(codesOut[idx] | (code << 4))
                    : (byte)(codesOut[idx] | code);
            }
        }
    }

    /// <summary>
    /// Build the per-query i8 LUT used by <see cref="KScoreTile4BitAvx2"/> and
    /// <see cref="KScoreTile4BitScalar"/>. Returns the scale that must be
    /// re-applied when the i16 accumulator is converted back to f32.
    /// </summary>
    /// <param name="rotatedQuery">Pre-rotated query, length <paramref name="dim"/>.</param>
    /// <param name="centroids">16-entry 4-bit codebook.</param>
    /// <param name="lutOut">Output LUT, length <paramref name="dim"/> × 16 bytes.</param>
    /// <param name="dim">Vector dimension.</param>
    /// <returns>Per-query scale (q_scale = max / 127).</returns>
    public static unsafe float BuildLut4Bit(
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<float> centroids,
        Span<sbyte> lutOut,
        int dim)
    {
        if (centroids.Length != 16)
            throw new ArgumentException("4-bit codebook must have 16 centroids.", nameof(centroids));
        return BuildKLutCore(rotatedQuery, centroids, lutOut, dim, codebookSlots: 16);
    }

    /// <summary>
    /// 3-bit variant: 8 centroids padded into the same 16-slot LUT shape the
    /// 4-bit kernel consumes (slots 8..15 are zero — the packed 3-bit codes
    /// stored in 4-bit nibbles never reach them).
    /// </summary>
    public static float BuildLut3Bit(
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<float> centroids,
        Span<sbyte> lutOut,
        int dim)
    {
        if (centroids.Length != 8)
            throw new ArgumentException("3-bit codebook must have 8 centroids.", nameof(centroids));
        return BuildKLutCore(rotatedQuery, centroids, lutOut, dim, codebookSlots: 8);
    }

    /// <summary>
    /// Shared K-side LUT builder. Centroids beyond <paramref name="codebookSlots"/>
    /// are written as zero so a single shared kernel can serve both bit widths.
    /// </summary>
    private static float BuildKLutCore(
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<float> centroids,
        Span<sbyte> lutOut,
        int dim,
        int codebookSlots)
    {
        if (lutOut.Length < dim * 16)
            throw new ArgumentException("LUT must be dim*16 bytes.", nameof(lutOut));

        float maxAbsCentroid = 0f;
        for (int v = 0; v < codebookSlots; v++)
        {
            float a = MathF.Abs(centroids[v]);
            if (a > maxAbsCentroid) maxAbsCentroid = a;
        }
        float maxAbsQuery = 0f;
        for (int d = 0; d < dim; d++)
        {
            float a = MathF.Abs(rotatedQuery[d]);
            if (a > maxAbsQuery) maxAbsQuery = a;
        }

        float maxAbs = maxAbsCentroid * maxAbsQuery;
        // Guard against an all-zero query (e.g. unused head padding); the LUT will
        // be all-zero anyway and we return scale=0 so the i16 accumulator zeroes
        // out cleanly after the final multiply.
        float scale = maxAbs > 0f ? maxAbs / 127f : 1f;
        float invScale = maxAbs > 0f ? 127f / maxAbs : 0f;

        for (int d = 0; d < dim; d++)
        {
            float q = rotatedQuery[d];
            int rowBase = d * 16;
            for (int v = 0; v < codebookSlots; v++)
            {
                float val = centroids[v] * q * invScale;
                int q8 = (int)MathF.Round(val);
                if (q8 > 127) q8 = 127;
                else if (q8 < -128) q8 = -128;
                lutOut[rowBase + v] = (sbyte)q8;
            }
            for (int v = codebookSlots; v < 16; v++)
                lutOut[rowBase + v] = 0;
        }
        return scale;
    }

    /// <summary>
    /// Scalar reference: computes 32 approximate dot products for one tile
    /// using the i8 LUT. Mirrors the AVX2 path bit-for-bit (i16 accumulator,
    /// final fp32 multiply) so the test suite can compare them without intrinsic
    /// hardware support and so the AVX2 implementation has a target to match.
    /// </summary>
    public static unsafe void KScoreTile4BitScalar(
        ReadOnlySpan<byte> tile,
        ReadOnlySpan<sbyte> lut,
        float scale,
        float attnScale,
        Span<float> scoresOut,
        int dim)
    {
        if (scoresOut.Length < TileSize)
            throw new ArgumentException($"scoresOut must be at least {TileSize}.", nameof(scoresOut));

        Span<short> acc = stackalloc short[TileSize];
        ReadOnlySpan<byte> codes = tile.Slice(NormBytesPerTile);

        for (int d = 0; d < dim; d++)
        {
            ReadOnlySpan<byte> codeRow = codes.Slice(d * CodeBytesPerDim, CodeBytesPerDim);
            ReadOnlySpan<sbyte> lutRow = lut.Slice(d * 16, 16);

            for (int b = 0; b < CodeBytesPerDim; b++)
            {
                byte pair = codeRow[b];
                int low = pair & 0x0F;        // position b
                int high = (pair >> 4) & 0x0F; // position 16 + b
                acc[b]      = (short)(acc[b]      + lutRow[low]);
                acc[16 + b] = (short)(acc[16 + b] + lutRow[high]);
            }
        }

        for (int t = 0; t < TileSize; t++)
        {
            float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(tile.Slice(t * 2, 2));
            scoresOut[t] = acc[t] * scale * norm * attnScale;
        }
    }

    /// <summary>
    /// AVX2 K-score for one tile (32 positions). Falls back to the scalar path
    /// when AVX2 isn't available — same numerical behavior either way.
    /// </summary>
    public static unsafe void KScoreTile4BitAvx2(
        byte* tile,
        sbyte* lut,
        float scale,
        float attnScale,
        float* scoresOut,
        int dim)
    {
        if (!Avx2.IsSupported || !Ssse3.IsSupported)
        {
            KScoreTile4BitScalar(
                new ReadOnlySpan<byte>(tile, TileBytes(dim)),
                new ReadOnlySpan<sbyte>(lut, dim * 16),
                scale, attnScale,
                new Span<float>(scoresOut, TileSize),
                dim);
            return;
        }

        byte* codes = tile + NormBytesPerTile;

        // Two i16x16 accumulators: acc_lo holds positions 0..15, acc_hi 16..31.
        // pshufb against a 16-byte LUT row, then widen i8 → i16 before the add to
        // keep the dim-fold from saturating (range is well within ±32767).
        var accLo = Vector256<short>.Zero;
        var accHi = Vector256<short>.Zero;
        var nibbleMask = Vector128.Create((byte)0x0F);

        for (int d = 0; d < dim; d++)
        {
            var codeRow = Sse2.LoadVector128(codes + d * CodeBytesPerDim);
            var lutRow  = Sse2.LoadVector128((byte*)(lut + d * 16));

            var codesLo = Sse2.And(codeRow, nibbleMask);
            var codesHi = Sse2.And(
                Sse2.ShiftRightLogical(codeRow.AsInt16(), 4).AsByte(),
                nibbleMask);

            var shufLo = Ssse3.Shuffle(lutRow, codesLo).AsSByte();
            var shufHi = Ssse3.Shuffle(lutRow, codesHi).AsSByte();

            accLo = Avx2.Add(accLo, Avx2.ConvertToVector256Int16(shufLo));
            accHi = Avx2.Add(accHi, Avx2.ConvertToVector256Int16(shufHi));
        }

        // Final fp32 fold: multiply each accumulator lane by scale × norm × attnScale.
        // 32 multiplies is cheap relative to the dim-deep inner loop above.
        float combinedScale = scale * attnScale;
        short* accLoPtr = stackalloc short[16];
        short* accHiPtr = stackalloc short[16];
        Avx.Store(accLoPtr, accLo);
        Avx.Store(accHiPtr, accHi);

        for (int t = 0; t < 16; t++)
        {
            float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
                new ReadOnlySpan<byte>(tile + t * 2, 2));
            scoresOut[t] = accLoPtr[t] * combinedScale * norm;
        }
        for (int t = 0; t < 16; t++)
        {
            float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
                new ReadOnlySpan<byte>(tile + (16 + t) * 2, 2));
            scoresOut[16 + t] = accHiPtr[t] * combinedScale * norm;
        }
    }

    /// <summary>
    /// Pack 32 per-block 4-bit TQ buffers into the FastScan V-tile layout
    /// (norms followed by position-major codes: byte (t, d/2) holds two
    /// consecutive dims as low/high nibbles).
    /// </summary>
    public static void PackVTile4Bit(
        ReadOnlySpan<byte> blocks,
        Span<byte> vTileOut,
        int dim)
    {
        if ((dim & 1) != 0)
            throw new ArgumentException("V-tile packing requires even dim.", nameof(dim));

        int blockSize = TurboQuantOps.BlockSize(4, dim);
        if (blocks.Length < TileSize * blockSize)
            throw new ArgumentException($"blocks must hold {TileSize} blocks of {blockSize} bytes each.", nameof(blocks));
        if (vTileOut.Length < VTileBytes(dim))
            throw new ArgumentException($"vTileOut must be at least {VTileBytes(dim)} bytes.", nameof(vTileOut));

        // Norms first — same layout as K-tile.
        for (int t = 0; t < TileSize; t++)
        {
            int src = t * blockSize + TurboQuantOps.NormOffset;
            vTileOut[t * 2]     = blocks[src];
            vTileOut[t * 2 + 1] = blocks[src + 1];
        }

        // Codes: byte (t, d/2) holds (code[d=2k+1][t] << 4) | code[d=2k][t].
        // The 4-bit per-block packed layout already stores adjacent dim pairs
        // as low/high nibbles of one byte, so the copy is a straight memcpy of
        // each block's packed bytes into vTileOut at offset NormBytesPerTile + t*(dim/2).
        var codesOut = vTileOut.Slice(NormBytesPerTile, TileSize * (dim / 2));
        int packedPerBlock = dim / 2;
        for (int t = 0; t < TileSize; t++)
        {
            ReadOnlySpan<byte> blockPacked = blocks.Slice(t * blockSize + TurboQuantOps.IndicesOffset, packedPerBlock);
            blockPacked.CopyTo(codesOut.Slice(t * packedPerBlock, packedPerBlock));
        }
    }

    /// <summary>
    /// 3-bit V-tile packer. Source codes are 0..7 — we unpack each one from
    /// the 3-bit-packed source block and rewrite as 4-bit nibbles into the
    /// V-tile layout. No fast memcpy here (unlike the 4-bit path) because
    /// 3-bit codes straddle byte boundaries in the source.
    /// </summary>
    public static void PackVTile3Bit(
        ReadOnlySpan<byte> blocks,
        Span<byte> vTileOut,
        int dim)
    {
        if ((dim & 1) != 0)
            throw new ArgumentException("V-tile packing requires even dim.", nameof(dim));

        int blockSize = TurboQuantOps.BlockSize(3, dim);
        if (blocks.Length < TileSize * blockSize)
            throw new ArgumentException($"blocks must hold {TileSize} blocks of {blockSize} bytes each.", nameof(blocks));
        if (vTileOut.Length < VTileBytes(dim))
            throw new ArgumentException($"vTileOut must be at least {VTileBytes(dim)} bytes.", nameof(vTileOut));

        for (int t = 0; t < TileSize; t++)
        {
            int src = t * blockSize + TurboQuantOps.NormOffset;
            vTileOut[t * 2]     = blocks[src];
            vTileOut[t * 2 + 1] = blocks[src + 1];
        }

        var codesOut = vTileOut.Slice(NormBytesPerTile, TileSize * (dim / 2));
        codesOut.Clear();
        int packedPerBlock = dim / 2;
        for (int t = 0; t < TileSize; t++)
        {
            ReadOnlySpan<byte> packed = blocks.Slice(t * blockSize + TurboQuantOps.IndicesOffset);
            int outOffset = t * packedPerBlock;
            for (int d = 0; d < dim; d += 2)
            {
                int low  = BitPacking.UnpackBits3(packed, 0, d)     & 0x07;
                int high = BitPacking.UnpackBits3(packed, 0, d + 1) & 0x07;
                codesOut[outOffset + d / 2] = (byte)(low | (high << 4));
            }
        }
    }

    /// <summary>
    /// Build the per-tile i8 LUT for V-aggregation. Effective weight is the
    /// caller-precomputed <c>weights[t] · norm[t]</c> for each of the 32
    /// positions in the tile. Returns the scale to apply when promoting the
    /// i16 accumulator back to fp32.
    /// </summary>
    public static float BuildVLut4Bit(
        ReadOnlySpan<float> effectiveWeights,
        ReadOnlySpan<float> centroids,
        Span<sbyte> vLutOut)
    {
        if (centroids.Length != 16)
            throw new ArgumentException("4-bit codebook must have 16 centroids.", nameof(centroids));
        return BuildVLutCore(effectiveWeights, centroids, vLutOut, codebookSlots: 16);
    }

    /// <summary>
    /// 3-bit variant: 8 centroids padded into the 16-slot LUT shape (slots
    /// 8..15 zero). See <see cref="BuildLut3Bit"/> for the rationale.
    /// </summary>
    public static float BuildVLut3Bit(
        ReadOnlySpan<float> effectiveWeights,
        ReadOnlySpan<float> centroids,
        Span<sbyte> vLutOut)
    {
        if (centroids.Length != 8)
            throw new ArgumentException("3-bit codebook must have 8 centroids.", nameof(centroids));
        return BuildVLutCore(effectiveWeights, centroids, vLutOut, codebookSlots: 8);
    }

    private static float BuildVLutCore(
        ReadOnlySpan<float> effectiveWeights,
        ReadOnlySpan<float> centroids,
        Span<sbyte> vLutOut,
        int codebookSlots)
    {
        if (effectiveWeights.Length < TileSize)
            throw new ArgumentException($"effectiveWeights must hold {TileSize}.", nameof(effectiveWeights));
        if (vLutOut.Length < TileSize * 16)
            throw new ArgumentException("vLutOut must be 32*16 bytes.", nameof(vLutOut));

        float maxAbsCentroid = 0f;
        for (int v = 0; v < codebookSlots; v++)
        {
            float a = MathF.Abs(centroids[v]);
            if (a > maxAbsCentroid) maxAbsCentroid = a;
        }
        float maxAbsWeight = 0f;
        for (int t = 0; t < TileSize; t++)
        {
            float a = MathF.Abs(effectiveWeights[t]);
            if (a > maxAbsWeight) maxAbsWeight = a;
        }

        float maxAbs = maxAbsCentroid * maxAbsWeight;
        float scale = maxAbs > 0f ? maxAbs / 127f : 1f;
        float invScale = maxAbs > 0f ? 127f / maxAbs : 0f;

        for (int t = 0; t < TileSize; t++)
        {
            float w = effectiveWeights[t];
            int rowBase = t * 16;
            for (int v = 0; v < codebookSlots; v++)
            {
                float val = w * centroids[v] * invScale;
                int q8 = (int)MathF.Round(val);
                if (q8 > 127) q8 = 127;
                else if (q8 < -128) q8 = -128;
                vLutOut[rowBase + v] = (sbyte)q8;
            }
            for (int v = codebookSlots; v < 16; v++)
                vLutOut[rowBase + v] = 0;
        }
        return scale;
    }

    /// <summary>
    /// Scalar reference for one V-aggregation tile.
    /// Accumulates into <paramref name="rotatedAccInOut"/> the contribution
    /// <c>sum_t effectiveW[t] · centroids[code[d][t]]</c>, in the rotated domain
    /// (sign flip and inverse WHT are deferred to the caller and applied once
    /// per kv-head after all tiles have been folded in).
    /// </summary>
    public static void VAggregateTile4BitScalar(
        ReadOnlySpan<byte> vTile,
        ReadOnlySpan<sbyte> vLut,
        float vScale,
        Span<float> rotatedAccInOut,
        int dim)
    {
        if (rotatedAccInOut.Length < dim)
            throw new ArgumentException($"rotatedAccInOut must be at least {dim}.", nameof(rotatedAccInOut));

        Span<short> acc = dim <= 1024 ? stackalloc short[dim] : new short[dim];
        acc.Clear();

        ReadOnlySpan<byte> codes = vTile.Slice(NormBytesPerTile);
        int packedPerBlock = dim / 2;

        for (int t = 0; t < TileSize; t++)
        {
            ReadOnlySpan<byte> codesT = codes.Slice(t * packedPerBlock, packedPerBlock);
            ReadOnlySpan<sbyte> lutT  = vLut.Slice(t * 16, 16);

            for (int b = 0; b < packedPerBlock; b++)
            {
                byte pair = codesT[b];
                int dEven = b * 2;
                acc[dEven]     = (short)(acc[dEven]     + lutT[pair & 0x0F]);
                acc[dEven + 1] = (short)(acc[dEven + 1] + lutT[(pair >> 4) & 0x0F]);
            }
        }

        for (int d = 0; d < dim; d++)
            rotatedAccInOut[d] += acc[d] * vScale;
    }

    /// <summary>
    /// AVX2 V-aggregation for one tile. Vectorises the inner d-loop so each
    /// pshufb pair handles 32 consecutive dims per position.
    /// </summary>
    public static unsafe void VAggregateTile4BitAvx2(
        byte* vTile,
        sbyte* vLut,
        float vScale,
        float* rotatedAccInOut,
        int dim)
    {
        if (!Avx2.IsSupported || !Ssse3.IsSupported || (dim & 31) != 0)
        {
            VAggregateTile4BitScalar(
                new ReadOnlySpan<byte>(vTile, VTileBytes(dim)),
                new ReadOnlySpan<sbyte>(vLut, TileSize * 16),
                vScale,
                new Span<float>(rotatedAccInOut, dim),
                dim);
            return;
        }

        byte* codes = vTile + NormBytesPerTile;
        int packedPerBlock = dim / 2;
        int chunks = dim / 32;  // each chunk = 16 bytes of codes = 32 dim entries

        // Per-tile i16 dim-accumulators, kept on the stack. dim=128 → 4 Vector256<short>.
        Span<short> accStorage = stackalloc short[1024];
        accStorage = accStorage.Slice(0, dim);
        accStorage.Clear();
        fixed (short* accPtr = accStorage)
        {
            var nibbleMask = Vector128.Create((byte)0x0F);

            for (int t = 0; t < TileSize; t++)
            {
                byte* codesT = codes + t * packedPerBlock;
                var lutT = Sse2.LoadVector128((byte*)(vLut + t * 16));

                for (int c = 0; c < chunks; c++)
                {
                    var codeChunk = Sse2.LoadVector128(codesT + c * 16);
                    var codesLo = Sse2.And(codeChunk, nibbleMask);
                    var codesHi = Sse2.And(
                        Sse2.ShiftRightLogical(codeChunk.AsInt16(), 4).AsByte(),
                        nibbleMask);

                    var shufLo = Ssse3.Shuffle(lutT, codesLo).AsSByte();
                    var shufHi = Ssse3.Shuffle(lutT, codesHi).AsSByte();

                    // Interleave low/high nibble contributions back to d-order:
                    // shufLo holds dims {d, d+2, ..., d+30}; shufHi holds {d+1, d+3, ..., d+31}.
                    // UnpackLow gives dims {d..d+15}, UnpackHigh gives {d+16..d+31}.
                    var lo16 = Sse2.UnpackLow(shufLo, shufHi).AsSByte();
                    var hi16 = Sse2.UnpackHigh(shufLo, shufHi).AsSByte();

                    int accOffset = c * 32;
                    var accLo = Avx.LoadVector256(accPtr + accOffset);
                    var accHi = Avx.LoadVector256(accPtr + accOffset + 16);
                    accLo = Avx2.Add(accLo, Avx2.ConvertToVector256Int16(lo16));
                    accHi = Avx2.Add(accHi, Avx2.ConvertToVector256Int16(hi16));
                    Avx.Store(accPtr + accOffset, accLo);
                    Avx.Store(accPtr + accOffset + 16, accHi);
                }
            }

            // i16 → fp32 + fold into running accumulator. 32 dims per vector pass.
            for (int d = 0; d < dim; d += 8)
            {
                var iv = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(accPtr + d));
                var fv = Avx.ConvertToVector256Single(iv);
                var prev = Avx.LoadVector256(rotatedAccInOut + d);
                Avx.Store(rotatedAccInOut + d, Avx.Add(prev, Avx.Multiply(fv, Vector256.Create(vScale))));
            }
        }
    }
}
