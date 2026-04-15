using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInference.TurboQuant;

/// <summary>
/// Core quantization/dequantization operations implementing TurboQuant Algorithm 1
/// (MSE-optimal) from Zandieh et al., "TurboQuant: Online Vector Quantization with
/// Near-optimal Distortion Rate" (arXiv:2504.19874).
///
/// Block format for one head_dim-sized vector:
///   [FP16 norm (2 bytes)] [packed N-bit indices] [padding to align]
///
/// For 3-bit, dim=128: 2 + 48 + 2 = 52 bytes per block.
/// For 4-bit, dim=128: 2 + 64 + 2 = 68 bytes per block.
///
/// NOTE: This implements only Algorithm 1 (TurboQuant_mse). The paper's Algorithm 2
/// (TurboQuant_prod) adds a 1-bit QJL correction on the residual for unbiased inner
/// product estimation. At 3-4 bits the MSE-optimal bias is small (~2-5%), but adding
/// QJL would improve long-context fidelity at the cost of ~1 extra bit per channel.
/// </summary>
public static class TurboQuantOps
{
    /// <summary>Norm offset within a block (always 0).</summary>
    public const int NormOffset = 0;

    /// <summary>Packed indices start at byte 2.</summary>
    public const int IndicesOffset = 2;

    /// <summary>
    /// Computes the total byte size of one TQ block for the given bit width and dimension.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int BlockSize(int bits, int dim)
    {
        int packedBytes = BitPacking.PackedBytesForBlock(bits, dim);
        // 2 bytes norm + packed indices + 2 bytes padding (align to 4 bytes)
        int raw = 2 + packedBytes;
        return (raw + 3) & ~3; // round up to 4-byte alignment
    }

    /// <summary>
    /// Quantize a float vector to TurboQuant compressed representation.
    /// </summary>
    /// <param name="input">Input vector of length <paramref name="dim"/>.</param>
    /// <param name="output">Output buffer, must be at least <see cref="BlockSize"/> bytes.</param>
    /// <param name="signPattern">Per-layer sign flip pattern, length <paramref name="dim"/>.</param>
    /// <param name="centroids">Lloyd-Max centroids.</param>
    /// <param name="boundaries">Lloyd-Max decision boundaries.</param>
    /// <param name="bits">Quantization bits (3 or 4).</param>
    /// <param name="dim">Vector dimension (must be power of 2).</param>
    public static void Quantize(
        ReadOnlySpan<float> input,
        Span<byte> output,
        ReadOnlySpan<float> signPattern,
        ReadOnlySpan<float> centroids,
        ReadOnlySpan<float> boundaries,
        int bits,
        int dim)
    {
        // Allocate rotated buffer on stack for typical dimensions
        Span<float> rotated = dim <= 512
            ? stackalloc float[dim]
            : new float[dim];

        // Step 1: Walsh-Hadamard transform + sign flip
        WalshHadamard.Transform(input, rotated, dim);
        WalshHadamard.ApplySignFlip(rotated, signPattern);

        // Step 2: Compute L2 norm and store as FP16
        float norm = ComputeNorm(rotated);
        BinaryPrimitives.WriteHalfLittleEndian(output, (Half)norm);

        // Step 3: Normalize and quantize each coordinate
        float invNorm = norm > 0 ? 1.0f / norm : 0f;

        // Clear the packed indices region
        int blockSize = BlockSize(bits, dim);
        output.Slice(IndicesOffset, blockSize - IndicesOffset).Clear();

        for (int i = 0; i < dim; i++)
        {
            float normalized = rotated[i] * invNorm;
            int index = FindBin(normalized, boundaries);

            if (bits == 3)
                BitPacking.PackBits3(output, IndicesOffset, i, index);
            else
                BitPacking.PackBits4(output, IndicesOffset, i, index);
        }
    }

    /// <summary>
    /// Dequantize a TurboQuant block back to a float vector.
    /// Used primarily for testing — the hot path uses <see cref="DequantDot"/>.
    /// </summary>
    public static void Dequantize(
        ReadOnlySpan<byte> compressed,
        Span<float> output,
        ReadOnlySpan<float> signPattern,
        ReadOnlySpan<float> centroids,
        int bits,
        int dim)
    {
        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(compressed);

        // Unpack and reconstruct in the rotated domain
        for (int i = 0; i < dim; i++)
        {
            int index = bits == 3
                ? BitPacking.UnpackBits3(compressed, IndicesOffset, i)
                : BitPacking.UnpackBits4(compressed, IndicesOffset, i);
            output[i] = centroids[index] * norm;
        }

        // Undo sign flip
        WalshHadamard.ApplySignFlip(output, signPattern);

        // Inverse WHT (WHT is its own inverse when normalized)
        WalshHadamard.Transform(output, output, dim);
    }

    /// <summary>
    /// Fused dequant-dot-product: computes dot(compressed_kv, query) without
    /// fully materializing the decompressed vector. This is the attention hot path.
    /// </summary>
    /// <param name="compressed">TQ-compressed block.</param>
    /// <param name="rotatedQuery">Pre-rotated query (WHT + sign flip applied once per head).</param>
    /// <param name="centroids">Lloyd-Max centroids.</param>
    /// <param name="bits">Quantization bits (3 or 4).</param>
    /// <param name="dim">Vector dimension.</param>
    /// <returns>Approximate dot product.</returns>
    public static float DequantDot(
        ReadOnlySpan<byte> compressed,
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<float> centroids,
        int bits,
        int dim)
    {
        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(compressed);
        float dot = 0f;

        for (int i = 0; i < dim; i++)
        {
            int index = bits == 3
                ? BitPacking.UnpackBits3(compressed, IndicesOffset, i)
                : BitPacking.UnpackBits4(compressed, IndicesOffset, i);
            dot += centroids[index] * rotatedQuery[i];
        }

        return dot * norm;
    }

    /// <summary>
    /// Prepares a query vector for use with <see cref="DequantDot"/>:
    /// applies WHT + sign flip. Call once per head, reuse across all cached positions.
    /// </summary>
    public static void RotateQuery(
        ReadOnlySpan<float> query,
        Span<float> rotatedQuery,
        ReadOnlySpan<float> signPattern,
        int dim)
    {
        WalshHadamard.Transform(query, rotatedQuery, dim);
        WalshHadamard.ApplySignFlip(rotatedQuery, signPattern);
    }

    /// <summary>
    /// AVX2-optimized fused dequant-dot for 3-bit TQ blocks.
    /// Uses a permute-based 8-entry LUT to avoid gather instructions.
    /// </summary>
    public static unsafe float DequantDot3Avx2(
        byte* compressed,
        float* rotatedQuery,
        float* centroids,
        int dim)
    {
        if (!Avx2.IsSupported)
            return DequantDot3Scalar(compressed, rotatedQuery, centroids, dim);

        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
            new ReadOnlySpan<byte>(compressed, 2));

        // Load 8 centroids into an AVX2 register for permute-based lookup
        var lut = Avx.LoadVector256(centroids);

        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;

        byte* packed = compressed + IndicesOffset;
        int i = 0;

        // Process 8 elements at a time
        for (; i + 8 <= dim; i += 8)
        {
            // Unpack 8 consecutive 3-bit indices
            var indices = Unpack8Indices3(packed, i);

            // Permute-based LUT: use indices to select centroids
            var centroidVals = Avx2.PermuteVar8x32(lut, indices);

            // Load 8 rotated query values
            var qVec = Avx.LoadVector256(rotatedQuery + i);

            // FMA: acc += centroid * query
            if (Fma.IsSupported)
                acc0 = Fma.MultiplyAdd(centroidVals, qVec, acc0);
            else
                acc0 = Avx.Add(acc0, Avx.Multiply(centroidVals, qVec));
        }

        // Horizontal sum
        float dot = HorizontalSum256(Avx.Add(acc0, acc1));

        // Scalar tail
        for (; i < dim; i++)
        {
            int idx = BitPacking.UnpackBits3(new ReadOnlySpan<byte>(packed, (dim * 3 + 7) / 8), 0, i);
            dot += centroids[idx] * rotatedQuery[i];
        }

        return dot * norm;
    }

    /// <summary>
    /// AVX2-optimized fused dequant-dot for 4-bit TQ blocks.
    /// 4-bit indices are half-byte aligned, making extraction trivial.
    /// </summary>
    public static unsafe float DequantDot4Avx2(
        byte* compressed,
        float* rotatedQuery,
        float* centroids,
        int dim)
    {
        if (!Avx2.IsSupported)
            return DequantDot4Scalar(compressed, rotatedQuery, centroids, dim);

        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
            new ReadOnlySpan<byte>(compressed, 2));

        byte* packed = compressed + IndicesOffset;
        var acc = Vector256<float>.Zero;
        int i = 0;

        for (; i + 8 <= dim; i += 8)
        {
            // Unpack 8 consecutive 4-bit indices (4 bytes = 8 nibbles)
            var indices = Unpack8Indices4(packed, i);

            // Gather centroids (16-entry table doesn't fit in permute, use gather)
            var centroidVals = Avx2.GatherVector256(centroids, indices, 4);

            var qVec = Avx.LoadVector256(rotatedQuery + i);

            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(centroidVals, qVec, acc);
            else
                acc = Avx.Add(acc, Avx.Multiply(centroidVals, qVec));
        }

        float dot = HorizontalSum256(acc);

        for (; i < dim; i++)
        {
            int idx = BitPacking.UnpackBits4(new ReadOnlySpan<byte>(packed, (dim + 1) / 2), 0, i);
            dot += centroids[idx] * rotatedQuery[i];
        }

        return dot * norm;
    }

    /// <summary>
    /// Scalar fallback for 3-bit dequant-dot using raw pointers (no span overhead).
    /// </summary>
    public static unsafe float DequantDot3Scalar(
        byte* compressed,
        float* rotatedQuery,
        float* centroids,
        int dim)
    {
        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
            new ReadOnlySpan<byte>(compressed, 2));
        byte* packed = compressed + IndicesOffset;
        float dot = 0f;

        for (int i = 0; i < dim; i++)
        {
            int bitOffset = i * 3;
            int byteIdx = bitOffset >> 3;
            int bitShift = bitOffset & 7;
            int raw = packed[byteIdx] >> bitShift;
            if (bitShift > 5)
                raw |= packed[byteIdx + 1] << (8 - bitShift);
            int idx = raw & 0x7;
            dot += centroids[idx] * rotatedQuery[i];
        }

        return dot * norm;
    }

    /// <summary>
    /// Scalar fallback for 4-bit dequant-dot using raw pointers.
    /// </summary>
    public static unsafe float DequantDot4Scalar(
        byte* compressed,
        float* rotatedQuery,
        float* centroids,
        int dim)
    {
        float norm = (float)BinaryPrimitives.ReadHalfLittleEndian(
            new ReadOnlySpan<byte>(compressed, 2));
        byte* packed = compressed + IndicesOffset;
        float dot = 0f;

        for (int i = 0; i < dim; i++)
        {
            int byteIdx = i >> 1;
            int idx = ((i & 1) == 0)
                ? packed[byteIdx] & 0x0F
                : (packed[byteIdx] >> 4) & 0x0F;
            dot += centroids[idx] * rotatedQuery[i];
        }

        return dot * norm;
    }

    /// <summary>
    /// Unpack 8 consecutive 3-bit indices starting at the given element position.
    /// Returns a Vector256&lt;int&gt; with values 0..7.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> Unpack8Indices3(byte* packed, int startPos)
    {
        // Extract 8 3-bit values. Each spans at most 2 bytes.
        // For simplicity, extract each individually (the permute lookup amortizes this).
        int* indices = stackalloc int[8];
        for (int j = 0; j < 8; j++)
        {
            int pos = startPos + j;
            int bitOffset = pos * 3;
            int byteIdx = bitOffset >> 3;
            int bitShift = bitOffset & 7;
            int raw = packed[byteIdx] >> bitShift;
            if (bitShift > 5)
                raw |= packed[byteIdx + 1] << (8 - bitShift);
            indices[j] = raw & 0x7;
        }
        return Avx.LoadVector256(indices);
    }

    /// <summary>
    /// Unpack 8 consecutive 4-bit indices starting at the given element position.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> Unpack8Indices4(byte* packed, int startPos)
    {
        int* indices = stackalloc int[8];
        for (int j = 0; j < 8; j++)
        {
            int pos = startPos + j;
            int byteIdx = pos >> 1;
            indices[j] = ((pos & 1) == 0)
                ? packed[byteIdx] & 0x0F
                : (packed[byteIdx] >> 4) & 0x0F;
        }
        return Avx.LoadVector256(indices);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalSum256(Vector256<float> v)
    {
        // Sum upper and lower 128-bit lanes
        var lo = v.GetLower();
        var hi = v.GetUpper();
        var sum128 = lo + hi;
        // Horizontal add within 128 bits
        sum128 = Sse3.IsSupported
            ? Sse3.HorizontalAdd(sum128, sum128)
            : sum128; // fallback below
        if (Sse3.IsSupported)
        {
            sum128 = Sse3.HorizontalAdd(sum128, sum128);
            return sum128.ToScalar();
        }
        // Scalar fallback
        return sum128.GetElement(0) + sum128.GetElement(1) +
               sum128.GetElement(2) + sum128.GetElement(3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeNorm(ReadOnlySpan<float> data)
    {
        float sumSq = 0f;
        for (int i = 0; i < data.Length; i++)
            sumSq += data[i] * data[i];
        return MathF.Sqrt(sumSq);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FindBin(float value, ReadOnlySpan<float> boundaries)
    {
        // Linear search for small codebooks (7 or 15 boundaries)
        int bin = 0;
        for (int i = 0; i < boundaries.Length; i++)
        {
            if (value >= boundaries[i])
                bin = i + 1;
            else
                break;
        }
        return bin;
    }
}
