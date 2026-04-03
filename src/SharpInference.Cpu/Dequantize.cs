using System.Runtime.InteropServices;
using SharpInference.Core;

namespace SharpInference.Cpu;

/// <summary>
/// Scalar dequantization routines for GGML quantized formats.
/// Matches the reference implementation in ggml-quants.c exactly.
/// </summary>
public static class Dequantize
{
    /// <summary>
    /// Dequantize a tensor from any supported quantized format to Float32.
    /// </summary>
    public static void ToFloat32(ReadOnlySpan<byte> src, Span<float> dst, DType dtype, long elementCount)
    {
        switch (dtype)
        {
            case DType.Float32:
                MemoryMarshal.Cast<byte, float>(src).Slice(0, (int)elementCount).CopyTo(dst);
                break;
            case DType.Q4_K:
                DequantQ4K(src, dst, elementCount);
                break;
            case DType.Q6_K:
                DequantQ6K(src, dst, elementCount);
                break;
            case DType.Q5_K:
                DequantQ5K(src, dst, elementCount);
                break;
            case DType.Q2_K:
                DequantQ2K(src, dst, elementCount);
                break;
            case DType.Q3_K:
                DequantQ3K(src, dst, elementCount);
                break;
            default:
                throw new NotSupportedException($"Dequantization not implemented for {dtype}");
        }
    }

    /// <summary>
    /// Q4_K dequantization. Block size = 256, type size = 144 bytes.
    /// Layout per block (block_q4_K in ggml):
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    ///   - 12 bytes: packed 6-bit scales and mins
    ///   - 128 bytes: 4-bit quantized values
    ///
    /// Reference: dequantize_row_q4_K in ggml-quants.c
    /// </summary>
    private static void DequantQ4K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 144;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);

            var scales = x.Slice(4, 12);  // K_SCALE_SIZE = 12
            var qs = x.Slice(16, 128);    // QK_K/2 = 128

            int qIdx = 0;
            int scaleIdx = 0;

            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(scaleIdx, scales, out byte sc1, out byte m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;
                GetScaleMinK4(scaleIdx + 1, scales, out byte sc2, out byte m2);
                float d2 = d * sc2;
                float dm2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    y[j + l] = d1 * (qs[qIdx + l] & 0xF) - dm1;
                    y[j + l + 32] = d2 * (qs[qIdx + l] >> 4) - dm2;
                }
                qIdx += 32;
                scaleIdx += 2;
            }
        }
    }

    /// <summary>
    /// Decode one 6-bit scale and min from the packed 12-byte scale/min buffer.
    /// Matches get_scale_min_k4 in ggml-quants.c.
    /// </summary>
    private static void GetScaleMinK4(int j, ReadOnlySpan<byte> q, out byte scale, out byte min)
    {
        if (j < 4)
        {
            scale = (byte)(q[j] & 63);
            min = (byte)(q[j + 4] & 63);
        }
        else
        {
            scale = (byte)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
            min = (byte)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
        }
    }

    /// <summary>
    /// Q6_K dequantization. Block size = 256, type size = 210 bytes.
    /// Layout per block (block_q6_K in ggml):
    ///   - 128 bytes: ql — lower 4 bits of 6-bit quants
    ///   - 64 bytes: qh — upper 2 bits of 6-bit quants
    ///   - 16 bytes: int8 scales (one per 16-element sub-block)
    ///   - 2 bytes: FP16 d (super-block scale)
    ///
    /// Reference: dequantize_row_q6_K in ggml-quants.c
    /// </summary>
    private static void DequantQ6K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 210;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[208], x[209]);

            int qlOff = 0;   // into ql (bytes 0..127)
            int qhOff = 128; // into qh (bytes 128..191)
            int scOff = 192; // into scales (bytes 192..207)
            int yOff = 0;

            int scBase = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16; // 0 for l<16, 1 for l>=16

                    int q1 = ((x[qlOff + l] & 0xF) | (((x[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((x[qlOff + l + 32] & 0xF) | (((x[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((x[qlOff + l] >> 4) | (((x[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((x[qlOff + l + 32] >> 4) | (((x[qhOff + l] >> 6) & 3) << 4)) - 32;

                    y[yOff + l] = d * (sbyte)x[scOff + scBase + isc] * q1;
                    y[yOff + l + 32] = d * (sbyte)x[scOff + scBase + isc + 2] * q2;
                    y[yOff + l + 64] = d * (sbyte)x[scOff + scBase + isc + 4] * q3;
                    y[yOff + l + 96] = d * (sbyte)x[scOff + scBase + isc + 6] * q4;
                }
                yOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }
        }
    }

    /// <summary>
    /// Q5_K dequantization. Block size = 256, type size = 176 bytes.
    /// Layout per block (block_q5_K in ggml):
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    ///   - 12 bytes: packed 6-bit scales and mins (same as Q4_K)
    ///   - 32 bytes: qh — high bits (one bit per element, packed)
    ///   - 128 bytes: ql — lower 4 bits (two elements per byte)
    ///
    /// Reference: dequantize_row_q5_K in ggml-quants.c
    /// </summary>
    private static void DequantQ5K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 176;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            var y = dst.Slice((int)(block * QK_K), QK_K);

            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);

            var scales = x.Slice(4, 12);  // K_SCALE_SIZE = 12
            var qh = x.Slice(16, 32);     // high bits: 256 bits = 32 bytes
            var ql = x.Slice(48, 128);    // QK_K/2 = 128

            int qIdx = 0;
            int scaleIdx = 0;

            // qh bit layout per byte: bits 0,1 for j=0; bits 2,3 for j=64;
            // bits 4,5 for j=128; bits 6,7 for j=192.
            // u1 masks the low-nibble high bit, u2 masks the high-nibble high bit.
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                GetScaleMinK4(scaleIdx, scales, out byte sc1, out byte m1);
                float d1 = d * sc1;
                float dm1 = dmin * m1;
                GetScaleMinK4(scaleIdx + 1, scales, out byte sc2, out byte m2);
                float d2 = d * sc2;
                float dm2 = dmin * m2;

                for (int l = 0; l < 32; l++)
                {
                    int hLo = (qh[l] & u1) != 0 ? 16 : 0;
                    int hHi = (qh[l] & u2) != 0 ? 16 : 0;
                    y[j + l] = d1 * ((ql[qIdx + l] & 0xF) + hLo) - dm1;
                    y[j + l + 32] = d2 * ((ql[qIdx + l] >> 4) + hHi) - dm2;
                }
                qIdx += 32;
                scaleIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    /// <summary>
    /// Q2_K dequantization. Block size = 256, type size = 84 bytes.
    /// Layout per block (block_q2_K in ggml):
    ///   - 16 bytes: scales (4 bits each, packed as nibbles)
    ///   - 64 bytes: qs (2-bit quantized values, 4 per byte)
    ///   - 2 bytes: FP16 d (super-block scale)
    ///   - 2 bytes: FP16 dmin (super-block min)
    /// Reference: dequantize_row_q2_K in ggml-quants.c
    /// </summary>
    /// <summary>
    /// Q2_K: matches ggml dequantize_row_q2_K exactly.
    /// Layout: [scales:16][qs:64][d:FP16][dmin:FP16] = 84 bytes / 256 elements.
    /// The 64 qs bytes are read 4 times with shifts 0,2,4,6 per 128-element group.
    /// </summary>
    private static void DequantQ2K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 84;
        long numBlocks = elementCount / QK_K;

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            int yOff = (int)(block * QK_K);

            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);

            int qOff = 16; // qs at byte 16
            int isIdx = 0;
            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc = x[isIdx++]; // scales at byte 0
                    float dl = d * (sc & 0xF);
                    float ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        dst[yOff++] = dl * ((x[qOff + l] >> shift) & 3) - ml;

                    sc = x[isIdx++];
                    dl = d * (sc & 0xF);
                    ml = min * (sc >> 4);
                    for (int l = 0; l < 16; l++)
                        dst[yOff++] = dl * ((x[qOff + l + 16] >> shift) & 3) - ml;

                    shift += 2;
                }
                qOff += 32;
            }
        }
    }

    /// <summary>
    /// Q3_K dequantization. Block size = 256, type size = 110 bytes.
    /// Layout per block (block_q3_K in ggml):
    ///   - 32 bytes: hmask (high bit per element)
    ///   - 64 bytes: qs (lower 2 bits, 4 per byte)
    ///   - 12 bytes: packed scales
    ///   - 2 bytes: FP16 d
    /// Reference: dequantize_row_q3_K in ggml-quants.c
    /// </summary>
    /// <summary>
    /// Q3_K: matches ggml dequantize_row_q3_K exactly.
    /// Layout: [hmask:32][qs:64][scales:12][d:FP16] = 110 bytes / 256 elements.
    /// Uses the aux[] uint32 manipulation for scale unpacking.
    /// </summary>
    private static void DequantQ3K(ReadOnlySpan<byte> src, Span<float> dst, long elementCount)
    {
        const int QK_K = 256;
        const int bytesPerBlock = 110;
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        long numBlocks = elementCount / QK_K;

        Span<uint> aux = stackalloc uint[4];

        for (long block = 0; block < numBlocks; block++)
        {
            var x = src.Slice((int)(block * bytesPerBlock), bytesPerBlock);
            int yOff = (int)(block * QK_K);

            float dAll = HalfToFloat(x[108], x[109]);

            // Unpack scales: copy 12 bytes at offset 96 into aux[0..2], then manipulate
            aux[0] = (uint)(x[96] | (x[97] << 8) | (x[98] << 16) | (x[99] << 24));
            aux[1] = (uint)(x[100] | (x[101] << 8) | (x[102] << 16) | (x[103] << 24));
            uint tmp = (uint)(x[104] | (x[105] << 8) | (x[106] << 16) | (x[107] << 24));

            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            // aux now contains 16 signed 6-bit scales as bytes (subtract 32 when used)
            int isIdx = 0;
            int qOff = 32; // qs at byte 32
            byte m = 1;    // hmask bit

            for (int n = 0; n < QK_K; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    // Scale as signed int8 from aux bytes
                    int scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    float dl = dAll * (scByte - 32);
                    isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((x[qOff + l] >> shift) & 3) - ((x[l] & m) != 0 ? 0 : 4);
                        dst[yOff++] = dl * q;
                    }

                    scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    dl = dAll * (scByte - 32);
                    isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((x[qOff + l + 16] >> shift) & 3) - ((x[l + 16] & m) != 0 ? 0 : 4);
                        dst[yOff++] = dl * q;
                    }

                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }
        }
    }

    /// <summary>Convert two bytes (little-endian) to FP16, then to float.</summary>
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }
}
