namespace SharpInference.Diffusion;

/// <summary>
/// Float ↔ fp8 E4M3FN conversion helpers (IEEE-compatible finite + NaN, no infinities).
///
/// Format: 1 sign | 4 exponent (bias=7) | 3 mantissa
/// Range:  ±448.0   (max normal = 2^8 × 1.875)
/// NaN:    0x7F and 0xFF (only NaN encodings; no infinities)
/// </summary>
internal static class Fp8Converter
{
    /// <summary>Convert a float32 value to fp8 E4M3FN, saturating out-of-range values.</summary>
    internal static byte FloatToFp8E4M3(float value)
    {
        if (float.IsNaN(value)) return 0x7F;

        // Saturate to fp8 representable range
        const float fp8Max = 448.0f;
        if (value > fp8Max)  value = fp8Max;
        if (value < -fp8Max) value = -fp8Max;

        uint bits = BitConverter.SingleToUInt32Bits(value);
        byte sign = (byte)(bits >> 31);
        int fp32Exp = (int)((bits >> 23) & 0xFF);
        uint fp32Mant = bits & 0x7FFFFF;

        // ±0
        if (fp32Exp == 0 && fp32Mant == 0) return (byte)(sign << 7);

        // Rebias exponent: fp32 bias=127, fp8 bias=7
        int exp8 = fp32Exp - 127 + 7;

        if (exp8 <= 0)
        {
            // Subnormal fp8: include implicit leading 1 and shift right
            int shift = 1 - exp8;
            if (shift > 4) return (byte)(sign << 7); // underflow → ±0
            uint sub = (fp32Mant | 0x800000u) >> (20 + shift);
            // Round-to-nearest: look at the bit being dropped
            uint rnd = ((fp32Mant | 0x800000u) >> (19 + shift)) & 1u;
            sub += rnd;
            return (byte)(((uint)sign << 7) | (sub & 7u));
        }

        // Round mantissa from 23 bits to 3 bits (round-to-nearest)
        uint mant3 = (fp32Mant + (1u << 19)) >> 20;
        if (mant3 >= 8) { mant3 = 0; exp8++; }

        // Prevent encoding NaN (0x7F / 0xFF): if result would be 0x7F, saturate to max 0x7E
        if (exp8 > 15 || (exp8 == 15 && mant3 >= 7))
            return (byte)((sign << 7) | 0x7E); // max = 448

        return (byte)((sign << 7) | (exp8 << 3) | (int)mant3);
    }

    /// <summary>Convert an fp8 E4M3FN byte back to float32.</summary>
    internal static float Fp8E4M3ToFloat(byte fp8)
    {
        // NaN encodings
        if ((fp8 & 0x7F) == 0x7F) return float.NaN;

        byte sign = (byte)(fp8 >> 7);
        int exp8 = (fp8 >> 3) & 0xF;
        int mant8 = fp8 & 0x7;

        float magnitude;
        if (exp8 == 0)
        {
            if (mant8 == 0) return 0.0f; // ±0
            // Subnormal: 2^(-6) × (mant/8)
            magnitude = mant8 * MathF.Pow(2.0f, -9.0f); // mant/8 × 2^(-6)
        }
        else
        {
            // Normal: 2^(exp8−7) × (1 + mant/8)
            magnitude = (1.0f + mant8 / 8.0f) * MathF.Pow(2.0f, exp8 - 7);
        }

        return sign == 0 ? magnitude : -magnitude;
    }

    /// <summary>Batch-convert float32 values to fp8 E4M3FN.</summary>
    internal static void ConvertToFp8(ReadOnlySpan<float> src, Span<byte> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = FloatToFp8E4M3(src[i]);
    }
}
