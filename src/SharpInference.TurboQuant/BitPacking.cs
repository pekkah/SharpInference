using System.Runtime.CompilerServices;

namespace SharpInference.TurboQuant;

/// <summary>
/// 2-bit, 3-bit and 4-bit packing/unpacking for TurboQuant/KVarN compressed KV cache.
/// 128 2-bit values pack into 32 bytes (256 bits).
/// 128 3-bit values pack into 48 bytes (384 bits).
/// 128 4-bit values pack into 64 bytes (512 bits).
/// </summary>
public static class BitPacking
{
    /// <summary>Bytes needed for 128 packed 2-bit indices.</summary>
    public const int PackedBytes2Bit = 32; // 128 * 2 / 8

    /// <summary>Bytes needed for 128 packed 3-bit indices.</summary>
    public const int PackedBytes3Bit = 48; // 128 * 3 / 8

    /// <summary>Bytes needed for 128 packed 4-bit indices.</summary>
    public const int PackedBytes4Bit = 64; // 128 * 4 / 8

    /// <summary>
    /// Pack a 2-bit index (0..3) at the given position into a byte buffer.
    /// Position p maps to bits (p &amp; 3) * 2 of byte p / 4. The target field is
    /// masked before writing, so no pre-clear of the buffer is required.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PackBits2(Span<byte> buffer, int byteOffset, int position, int value)
    {
        int byteIdx = byteOffset + (position >> 2);
        int bitShift = (position & 3) << 1;
        buffer[byteIdx] = (byte)((buffer[byteIdx] & ~(0x3 << bitShift)) | ((value & 0x3) << bitShift));
    }

    /// <summary>
    /// Unpack a 2-bit index (0..3) from the given position in a byte buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int UnpackBits2(ReadOnlySpan<byte> buffer, int byteOffset, int position)
    {
        int byteIdx = byteOffset + (position >> 2);
        int bitShift = (position & 3) << 1;
        return (buffer[byteIdx] >> bitShift) & 0x3;
    }

    /// <summary>
    /// Pack a 3-bit index (0..7) at the given position into a byte buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PackBits3(Span<byte> buffer, int byteOffset, int position, int value)
    {
        int bitOffset = position * 3;
        int byteIdx = byteOffset + (bitOffset >> 3);
        int bitShift = bitOffset & 7;

        // Value occupies up to 2 bytes when straddling a byte boundary
        buffer[byteIdx] |= (byte)(value << bitShift);
        if (bitShift > 5) // 3 bits starting at bit 6 or 7 will overflow
            buffer[byteIdx + 1] |= (byte)(value >> (8 - bitShift));
    }

    /// <summary>
    /// Unpack a 3-bit index (0..7) from the given position in a byte buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int UnpackBits3(ReadOnlySpan<byte> buffer, int byteOffset, int position)
    {
        int bitOffset = position * 3;
        int byteIdx = byteOffset + (bitOffset >> 3);
        int bitShift = bitOffset & 7;

        int raw = buffer[byteIdx] >> bitShift;
        if (bitShift > 5)
            raw |= buffer[byteIdx + 1] << (8 - bitShift);

        return raw & 0x7;
    }

    /// <summary>
    /// Pack a 4-bit index (0..15) at the given position into a byte buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void PackBits4(Span<byte> buffer, int byteOffset, int position, int value)
    {
        int byteIdx = byteOffset + (position >> 1);
        if ((position & 1) == 0)
            buffer[byteIdx] = (byte)((buffer[byteIdx] & 0xF0) | (value & 0x0F));
        else
            buffer[byteIdx] = (byte)((buffer[byteIdx] & 0x0F) | ((value & 0x0F) << 4));
    }

    /// <summary>
    /// Unpack a 4-bit index (0..15) from the given position in a byte buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int UnpackBits4(ReadOnlySpan<byte> buffer, int byteOffset, int position)
    {
        int byteIdx = byteOffset + (position >> 1);
        if ((position & 1) == 0)
            return buffer[byteIdx] & 0x0F;
        else
            return (buffer[byteIdx] >> 4) & 0x0F;
    }

    /// <summary>
    /// Returns the number of packed bytes for the given bit width and dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PackedBytesForBlock(int bits, int dim) =>
        (dim * bits + 7) / 8;
}
