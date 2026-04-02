using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpInference.TurboQuant;

/// <summary>
/// Walsh-Hadamard Transform (WHT) with scalar and AVX2 implementations.
/// Used by TurboQuant to rotate KV vectors into a distribution suitable
/// for Lloyd-Max scalar quantization.
/// </summary>
public static class WalshHadamard
{
    /// <summary>
    /// Applies the normalized Walsh-Hadamard transform: output = WHT(input) / sqrt(dim).
    /// The transform is its own inverse (involution) when normalized.
    /// </summary>
    /// <param name="input">Input vector of length <paramref name="dim"/>.</param>
    /// <param name="output">Output buffer of length <paramref name="dim"/>. May alias input.</param>
    /// <param name="dim">Dimension, must be a power of 2.</param>
    public static void Transform(ReadOnlySpan<float> input, Span<float> output, int dim)
    {
        if (!BitOperations.IsPow2(dim))
            throw new ArgumentException("Dimension must be a power of 2.", nameof(dim));

        // Copy input to output for in-place transform
        if (!Unsafe.AreSame(ref MemoryMarshal.GetReference(input),
                            ref MemoryMarshal.GetReference(output)))
        {
            input.CopyTo(output);
        }

        if (Avx2.IsSupported && dim >= 8)
            TransformAvx2(output, dim);
        else
            TransformScalar(output, dim);

        // Normalize by 1/sqrt(dim)
        float scale = 1.0f / MathF.Sqrt(dim);
        ScaleInPlace(output, scale);
    }

    /// <summary>
    /// In-place WHT using the iterative butterfly algorithm (scalar path).
    /// </summary>
    internal static void TransformScalar(Span<float> data, int dim)
    {
        for (int stride = dim >> 1; stride >= 1; stride >>= 1)
        {
            for (int i = 0; i < dim; i += stride << 1)
            {
                for (int j = i; j < i + stride; j++)
                {
                    float a = data[j];
                    float b = data[j + stride];
                    data[j] = a + b;
                    data[j + stride] = a - b;
                }
            }
        }
    }

    /// <summary>
    /// In-place WHT using AVX2 SIMD for strides >= 8, scalar for smaller strides.
    /// </summary>
    internal static void TransformAvx2(Span<float> data, int dim)
    {
        ref float dataRef = ref MemoryMarshal.GetReference(data);

        // Large strides: process 8 pairs at a time with AVX2
        for (int stride = dim >> 1; stride >= 8; stride >>= 1)
        {
            for (int i = 0; i < dim; i += stride << 1)
            {
                for (int j = i; j < i + stride; j += 8)
                {
                    ref float pA = ref Unsafe.Add(ref dataRef, j);
                    ref float pB = ref Unsafe.Add(ref dataRef, j + stride);

                    var a = Vector256.LoadUnsafe(ref pA);
                    var b = Vector256.LoadUnsafe(ref pB);

                    Vector256.StoreUnsafe(Avx.Add(a, b), ref pA);
                    Vector256.StoreUnsafe(Avx.Subtract(a, b), ref pB);
                }
            }
        }

        // Small strides (< 8): scalar butterfly
        for (int stride = Math.Min(dim >> 1, 4); stride >= 1; stride >>= 1)
        {
            for (int i = 0; i < dim; i += stride << 1)
            {
                for (int j = i; j < i + stride; j++)
                {
                    float a = data[j];
                    float b = data[j + stride];
                    data[j] = a + b;
                    data[j + stride] = a - b;
                }
            }
        }
    }

    /// <summary>
    /// Applies a deterministic sign-flip pattern to a vector.
    /// The sign pattern is precomputed once per layer using a seeded PRNG.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplySignFlip(Span<float> data, ReadOnlySpan<float> signs)
    {
        if (Avx2.IsSupported && data.Length >= 8)
        {
            ref float dRef = ref MemoryMarshal.GetReference(data);
            ref float sRef = ref MemoryMarshal.GetReference(signs);
            int i = 0;
            for (; i + 8 <= data.Length; i += 8)
            {
                var d = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, i));
                var s = Vector256.LoadUnsafe(ref Unsafe.Add(ref sRef, i));
                Vector256.StoreUnsafe(Avx.Multiply(d, s), ref Unsafe.Add(ref dRef, i));
            }
            for (; i < data.Length; i++)
                data[i] *= signs[i];
        }
        else
        {
            for (int i = 0; i < data.Length; i++)
                data[i] *= signs[i];
        }
    }

    /// <summary>
    /// Generates a deterministic sign pattern (+1/-1) for a given layer index.
    /// Uses a simple xorshift64 PRNG seeded with the layer index.
    /// </summary>
    public static float[] GenerateSignPattern(int dim, int layerIndex)
    {
        var signs = new float[dim];
        ulong state = (ulong)(layerIndex + 1) * 0x9E3779B97F4A7C15UL;
        for (int i = 0; i < dim; i++)
        {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            signs[i] = (state & 1) == 0 ? 1.0f : -1.0f;
        }
        return signs;
    }

    private static void ScaleInPlace(Span<float> data, float scale)
    {
        if (Avx2.IsSupported && data.Length >= 8)
        {
            ref float dRef = ref MemoryMarshal.GetReference(data);
            var vScale = Vector256.Create(scale);
            int i = 0;
            for (; i + 8 <= data.Length; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, i));
                Vector256.StoreUnsafe(Avx.Multiply(v, vScale), ref Unsafe.Add(ref dRef, i));
            }
            for (; i < data.Length; i++)
                data[i] *= scale;
        }
        else
        {
            for (int i = 0; i < data.Length; i++)
                data[i] *= scale;
        }
    }
}
