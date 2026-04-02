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

    /// <summary>Convert two bytes (little-endian) to FP16, then to float.</summary>
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }
}
