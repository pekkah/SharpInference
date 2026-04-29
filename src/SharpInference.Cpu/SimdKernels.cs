using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;

namespace SharpInference.Cpu;

/// <summary>
/// AVX2-optimized compute kernels with fused dequantization and multi-threading.
/// All methods expect properly sized, non-null inputs. No bounds checking.
/// </summary>
public static unsafe class SimdKernels
{
    private const int MinRowsForParallel = 64;
    private static readonly ParallelOptions s_parallelOpts = new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount
    };

    // ================================================================
    //  Batched GEMM (for prefill)
    // ================================================================

    // Reusable dequant buffer for GEMM (one weight matrix at a time)
    [ThreadStatic] private static nint t_dequantBuf;
    [ThreadStatic] private static int t_dequantBufSize;

    private static bool s_blasLogged;

    /// <summary>
    /// Minimum batch size to engage OpenBLAS SGEMM in MatMulBatched.
    /// Below this threshold, sequential fused MatVec (dequant in registers) is used.
    /// Default 16 is the empirical crossover where SGEMM amortizes F32 dequantization cost
    /// over the per-token compute (measured on Ryzen 9 7900X with Q4_K_M 8192×2048 weights).
    /// Override via SHARPI_MIN_BATCH_BLAS environment variable.
    /// </summary>
    public static int MinBatchForBlas { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("SHARPI_MIN_BATCH_BLAS"), out var v) && v >= 1
            ? v
            : 16;

    /// <summary>
    /// Batched matrix multiply: output[batchSize, rows] = input[batchSize, cols] × W[rows, cols]^T
    /// Uses OpenBLAS sgemm when available (dequant weights to F32 temp buffer, then GEMM).
    /// Falls back to sequential MatVec per batch element.
    /// </summary>
    public static void MatMulBatched(float* output, byte* weights, float* input,
        int batchSize, int rows, int cols, DType dtype)
    {
        if (!s_blasLogged)
        {
            Console.Error.WriteLine($"[SharpInference] OpenBLAS: {(BlasInterop.IsAvailable ? "LOADED" : "not found (fallback to sequential)")}");
            s_blasLogged = true;
        }
        // For small batches, fused MatVec is faster (no dequant overhead)
        // BLAS only wins when N is large enough to amortize F32 dequantization

        if (batchSize < MinBatchForBlas || !BlasInterop.IsAvailable)
        {
            // Sequential fused MatVec (dequant in registers, no temp buffer)
            for (int n = 0; n < batchSize; n++)
                MatVec(output + n * rows, weights, input + n * cols, rows, cols, dtype);
            return;
        }

        // OpenBLAS GEMM path for large batches: dequant weights to F32, then sgemm
        if (dtype != DType.Float32)
        {
            int weightElements = rows * cols;

            // Ensure thread-local dequant buffer is large enough
            if (t_dequantBufSize < weightElements)
            {
                if (t_dequantBuf != 0) NativeMemory.Free((void*)t_dequantBuf);
                t_dequantBuf = (nint)NativeMemory.AllocZeroed((nuint)(weightElements * sizeof(float)));
                t_dequantBufSize = weightElements;
            }
            var wf32 = (float*)t_dequantBuf;

            // Dequantize full weight matrix to F32
            long totalBytes = DTypeInfo.ByteSize(weightElements, dtype);
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(weights, (int)totalBytes),
                new Span<float>(wf32, weightElements),
                dtype, weightElements);

            // sgemm: C[M,N] = A[M,K] * B[K,N]
            // We want: output[batchSize, rows] = input[batchSize, cols] * W[rows, cols]^T
            // In row-major: C = input * W^T
            // sgemm(RowMajor, NoTrans, Trans, M=batchSize, N=rows, K=cols,
            //        alpha=1, A=input, lda=cols, B=W, ldb=cols, beta=0, C=output, ldc=rows)
            BlasInterop.Sgemm(
                BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                batchSize, rows, cols,
                1.0f, input, cols,
                wf32, cols,
                0.0f, output, rows);
            return;
        }

        // F32 weights with BLAS
        if (BlasInterop.IsAvailable && dtype == DType.Float32)
        {
            BlasInterop.Sgemm(
                BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
                batchSize, rows, cols,
                1.0f, input, cols,
                (float*)weights, cols,
                0.0f, output, rows);
            return;
        }

    }

    // ================================================================
    //  Dispatchers
    // ================================================================

    /// <summary>
    /// Fused matrix-vector multiply. For quantized dtypes, dequantization
    /// happens in registers — no intermediate F32 buffer is allocated.
    /// </summary>
    public static void MatVec(float* output, byte* weights, float* input,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Float32:
                MatVecF32(output, (float*)weights, input, rows, cols);
                break;
            case DType.Q4_K:
                MatVecQ4K(output, weights, input, rows, cols);
                break;
            case DType.Q6_K:
                MatVecQ6K(output, weights, input, rows, cols);
                break;
            case DType.Q5_K:
                MatVecQ5K(output, weights, input, rows, cols);
                break;
            case DType.Q2_K:
                MatVecQ2K(output, weights, input, rows, cols);
                break;
            case DType.Q3_K:
                MatVecQ3K(output, weights, input, rows, cols);
                break;
            default:
                MatVecDequantFallback(output, weights, input, rows, cols, dtype);
                break;
        }
    }

    /// <summary>
    /// Compute two matrix-vector products sharing the same input in a single Parallel.For,
    /// halving thread-dispatch overhead for fused gate+up FFN projections.
    /// Both weight matrices must have the same dtype, rows, and cols.
    /// Falls back to two sequential MatVec calls if dtypes differ.
    /// </summary>
    public static void MatVecDual(
        float* output1, byte* weights1,
        float* output2, byte* weights2,
        float* input, int rows, int cols, DType dtype1, DType dtype2)
    {
        if (dtype1 != dtype2)
        {
            MatVec(output1, weights1, input, rows, cols, dtype1);
            MatVec(output2, weights2, input, rows, cols, dtype2);
            return;
        }

        switch (dtype1)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ4K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ4K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ4K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ4K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ6K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ6K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ6K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ6K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ5K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ5K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ5K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ5K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q3_K:
            {
                int bpr = (cols / 256) * 110;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ3K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ3K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ3K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ3K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Q2_K:
            {
                int bpr = (cols / 256) * 84;
                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ2K(w1 + (long)r * bpr, inp, c);
                        o2[r] = DotQ2K(w2 + (long)r * bpr, inp, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ2K(weights1 + (long)r * bpr, input, cols);
                        output2[r] = DotQ2K(weights2 + (long)r * bpr, input, cols);
                    }
                }
                break;
            }
            case DType.Float32:
            {
                if (rows >= MinRowsForParallel)
                {
                    var m1 = (float*)weights1; var m2 = (float*)weights2; var inp = input;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotF32(m1 + (long)r * c, inp, c);
                        o2[r] = DotF32(m2 + (long)r * c, inp, c);
                    });
                }
                else
                {
                    var m1 = (float*)weights1; var m2 = (float*)weights2;
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotF32(m1 + (long)r * cols, input, cols);
                        output2[r] = DotF32(m2 + (long)r * cols, input, cols);
                    }
                }
                break;
            }
            default:
                MatVec(output1, weights1, input, rows, cols, dtype1);
                MatVec(output2, weights2, input, rows, cols, dtype2);
                break;
        }
    }

    private static void MatVecDequantFallback(float* output, byte* weights, float* input,
        int rows, int cols, DType dtype)
    {
        int blockSize = DTypeInfo.BlockSize(dtype);
        int bytesPerBlock = DTypeInfo.BytesPerBlock(dtype);
        int blocksPerRow = cols / blockSize;
        int bytesPerRow = blocksPerRow * bytesPerBlock;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            int bpr = bytesPerRow; int c = cols;
            var dt = dtype;
            Parallel.For(0, rows, s_parallelOpts, () =>
                (nint)NativeMemory.Alloc((nuint)(c * sizeof(float))),
                (r, _, bufPtr) =>
                {
                    float* rowBuf = (float*)bufPtr;
                    byte* rowPtr = w + (long)r * bpr;
                    Dequantize.ToFloat32(new ReadOnlySpan<byte>(rowPtr, bpr),
                        new Span<float>(rowBuf, c), dt, c);
                    outp[r] = DotF32(rowBuf, inp, c);
                    return bufPtr;
                },
                bufPtr => NativeMemory.Free((void*)bufPtr)
            );
        }
        else
        {
            float* rowBuf = (float*)NativeMemory.Alloc((nuint)(cols * sizeof(float)));
            try
            {
                for (int r = 0; r < rows; r++)
                {
                    byte* rowPtr = weights + (long)r * bytesPerRow;
                    Dequantize.ToFloat32(new ReadOnlySpan<byte>(rowPtr, bytesPerRow),
                        new Span<float>(rowBuf, cols), dtype, cols);
                    output[r] = DotF32(rowBuf, input, cols);
                }
            }
            finally { NativeMemory.Free(rowBuf); }
        }
    }

    // ================================================================
    //  F32 MatVec
    // ================================================================

    public static void MatVecF32(float* output, float* matrix, float* input, int rows, int cols)
    {
        if (rows >= MinRowsForParallel)
        {
            var m = matrix; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotF32(m + (long)i * cols, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotF32(matrix + (long)i * cols, input, cols);
        }
    }

    // ================================================================
    //  Q4_K Fused MatVec
    // ================================================================

    public static void MatVecQ4K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 144;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ4K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ4K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q6_K Fused MatVec
    // ================================================================

    public static void MatVecQ6K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 210;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ6K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ6K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  F32 Dot Product  (4-way unrolled FMA)
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotF32(float* a, float* b, int n)
    {
        if (Fma.IsSupported && n >= 32)
        {
            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            int i = 0;
            for (; i + 32 <= n; i += 32)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), acc1);
                acc2 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 16), Avx.LoadVector256(b + i + 16), acc2);
                acc3 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 24), Avx.LoadVector256(b + i + 24), acc3);
            }
            acc0 = Avx.Add(Avx.Add(acc0, acc1), Avx.Add(acc2, acc3));

            for (; i + 8 <= n; i += 8)
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0);

            float sum = HSum256(acc0);
            for (; i < n; i++) sum += a[i] * b[i];
            return sum;
        }

        {
            float sum = 0;
            for (int i = 0; i < n; i++) sum += a[i] * b[i];
            return sum;
        }
    }

    // ================================================================
    //  Q4_K Fused Dequant-Dot  (one row)
    // ================================================================

    public static float DotQ4K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
            return DotQ4K_Avx512(row, input, cols, numBlocks);

        if (!Fma.IsSupported)
            return DotQ4K_Scalar(row, input, cols);

        // Two independent accumulators to break FMA dependency chains
        var accLo = Vector256<float>.Zero;
        var accHi = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector256.Create(d * sc1);
                var negDm1 = Vector256.Create(-(dmin * m1));
                var d2 = Vector256.Create(d * sc2);
                var negDm2 = Vector256.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                // Interleaved: process both nibbles from same bytes, into separate accumulators
                for (int l = 0; l < 32; l += 8)
                {
                    var bytes = LoadBytes8(qs + qIdx + l);
                    var ints = Avx2.ConvertToVector256Int32(bytes);

                    // Lower nibble → accLo
                    var lo = Avx2.And(ints, mask0F);
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(lo), negDm1);
                    accLo = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + l), accLo);

                    // Upper nibble → accHi (independent chain)
                    var hi = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(hi), negDm2);
                    accHi = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + l), accHi);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        return HSum256(Avx.Add(accLo, accHi));
    }

    private static float DotQ4K_Avx512(byte* row, float* input, int cols, int numBlocks)
    {
        var accLo = Vector512<float>.Zero;
        var accHi = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            int qIdx = 0;
            int scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2 = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                // Process 16 elements per iteration (vs 8 with AVX2)
                for (int l = 0; l < 32; l += 16)
                {
                    // Load 16 quantized bytes → 16 int32s via vpmovzxbd
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble → accLo
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    accLo = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input + bo + l)), accLo);

                    // Upper nibble → accHi
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input + bo + 32 + l)), accHi);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        return HSum512(Avx512F.Add(accLo, accHi));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum512(Vector512<float> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        return HSum256(Avx.Add(lo, hi));
    }

    private static float DotQ4K_Scalar(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;
            int qIdx = 0, scIdx = 0;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);
                float d1 = d * sc1, dm1 = dmin * m1;
                float d2 = d * sc2, dm2 = dmin * m2;
                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l++)
                {
                    acc += (d1 * (qs[qIdx + l] & 0xF) - dm1) * input[bo + l];
                    acc += (d2 * (qs[qIdx + l] >> 4) - dm2) * input[bo + 32 + l];
                }
                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q5_K Fused MatVec
    // ================================================================

    public static void MatVecQ5K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 176;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ5K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ5K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q5_K Fused Dequant-Dot  (one row)
    // ================================================================

    /// <summary>
    /// Fused Q5_K dequantize-dot product using AVX2 FMA.
    /// Q5_K block (176 bytes per 256 elements):
    ///   [0:1] FP16 d, [2:3] FP16 dmin, [4:15] scales (12 bytes),
    ///   [16:47] qh (32 bytes, 1 high bit per element), [48:175] ql (128 bytes, 4 bits).
    /// </summary>
    public static float DotQ5K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
            return DotQ5K_Avx512(row, input, cols, numBlocks);

        if (!Fma.IsSupported)
            return DotQ5K_Scalar(row, input, cols);

        var accLo = Vector256<float>.Zero;
        var accHi = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        var bit16 = Vector256.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0;
            int scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector256.Create(d * sc1);
                var negDm1 = Vector256.Create(-(dmin * m1));
                var d2 = Vector256.Create(d * sc2);
                var negDm2 = Vector256.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 8)
                {
                    // Load 8 ql bytes and extract nibbles
                    var bytes = LoadBytes8(ql + qIdx + l);
                    var ints = Avx2.ConvertToVector256Int32(bytes);
                    var loNibble = Avx2.And(ints, mask0F);
                    var hiNibble = Avx2.And(Avx2.ShiftRightLogical(ints, 4), mask0F);

                    // Load 8 qh bytes and extract high bits for this chunk
                    var qhBytes = LoadBytes8(qh + l);
                    var qhInts = Avx2.ConvertToVector256Int32(qhBytes);

                    // High bit for low nibble: (qh & u1) != 0 → 16
                    var hLoMask = Avx2.And(qhInts, Vector256.Create((int)u1));
                    var hLo = Avx2.And(
                        Avx2.CompareGreaterThan(hLoMask, Vector256<int>.Zero),
                        bit16);
                    var q5Lo = Avx2.Add(loNibble, hLo);

                    // High bit for high nibble: (qh & u2) != 0 → 16
                    var hHiMask = Avx2.And(qhInts, Vector256.Create((int)u2));
                    var hHi = Avx2.And(
                        Avx2.CompareGreaterThan(hHiMask, Vector256<int>.Zero),
                        bit16);
                    var q5Hi = Avx2.Add(hiNibble, hHi);

                    // Dequant: d1 * q5Lo - dm1
                    var deqLo = Fma.MultiplyAdd(d1, Avx.ConvertToVector256Single(q5Lo), negDm1);
                    accLo = Fma.MultiplyAdd(deqLo, Avx.LoadVector256(input + bo + l), accLo);

                    // Dequant: d2 * q5Hi - dm2
                    var deqHi = Fma.MultiplyAdd(d2, Avx.ConvertToVector256Single(q5Hi), negDm2);
                    accHi = Fma.MultiplyAdd(deqHi, Avx.LoadVector256(input + bo + 32 + l), accHi);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        return HSum256(Avx.Add(accLo, accHi));
    }

    private static float DotQ5K_Avx512(byte* row, float* input, int cols, int numBlocks)
    {
        var accLo = Vector512<float>.Zero;
        var accHi = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16 = Vector512.Create(16);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;

            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);

                var d1 = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2 = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);

                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input + bo + l)), accLo);

                    // High nibble + high bit
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input + bo + 32 + l)), accHi);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        return HSum512(Avx512F.Add(accLo, accHi));
    }

    private static float DotQ5K_Scalar(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 176;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qh = x + 16;
            byte* ql = x + 48;
            int qIdx = 0, scIdx = 0;
            byte u1 = 1, u2 = 2;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(scIdx, sc, out byte sc1, out byte m1);
                GetScaleMinK4(scIdx + 1, sc, out byte sc2, out byte m2);
                float d1 = d * sc1, dm1 = dmin * m1;
                float d2 = d * sc2, dm2 = dmin * m2;
                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l++)
                {
                    int hLo = (qh[l] & u1) != 0 ? 16 : 0;
                    int hHi = (qh[l] & u2) != 0 ? 16 : 0;
                    acc += (d1 * ((ql[qIdx + l] & 0xF) + hLo) - dm1) * input[bo + l];
                    acc += (d2 * ((ql[qIdx + l] >> 4) + hHi) - dm2) * input[bo + 32 + l];
                }
                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q3_K Fused MatVec
    // ================================================================

    public static void MatVecQ3K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 110;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ3K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ3K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    /// <summary>
    /// Fused Q3_K dequant-dot with AVX2.
    /// Block = 110 bytes / 256 elements: [hmask:32][qs:64][scales:12][d:FP16].
    /// Uses aux[] uint32 scale unpacking matching ggml exactly.
    /// </summary>
    public static float DotQ3K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (!Fma.IsSupported)
            return DotQ3K_Scalar(row, input, cols, numBlocks);

        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;

        var acc = Vector256<float>.Zero;
        var mask03 = Vector256.Create(0x03);
        var four = Vector256.Create(4);
        int elemOff = 0;

        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);

            // Unpack scales using aux[] manipulation (matching ggml)
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            // Extract 16 scale bytes from aux
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32; // qs at byte 32
            byte* hm = x;       // hmask at byte 0
            int qOff = 0;
            int isIdx = 0;
            byte m = 1;

            for (int n = 0; n < 256; n += 128)
            {
                for (int j = 0; j < 4; j++)
                {
                    float dl = dAll * (scales[isIdx++] - 32);
                    var vDl = Vector256.Create(dl);

                    // First 16 elements
                    for (int l = 0; l < 16; l += 8)
                    {
                        var qBytes = LoadBytes8(qs + qOff + l);
                        var qInts = Avx2.ConvertToVector256Int32(qBytes);
                        var shifted = j switch {
                            0 => qInts,
                            1 => Avx2.ShiftRightLogical(qInts, 2),
                            2 => Avx2.ShiftRightLogical(qInts, 4),
                            _ => Avx2.ShiftRightLogical(qInts, 6),
                        };
                        var q2 = Avx2.And(shifted, mask03);

                        // High bit from hmask: subtract 4 if hmask bit is NOT set
                        var hmBytes = LoadBytes8(hm + l);
                        var hmInts = Avx2.ConvertToVector256Int32(hmBytes);
                        var hmBit = Avx2.And(hmInts, Vector256.Create((int)m));
                        // If bit set → 0, if not set → 4
                        var sub = Avx2.And(Avx2.CompareEqual(hmBit, Vector256<int>.Zero), four);
                        var q3 = Avx2.Subtract(q2, sub);

                        var deq = Avx.Multiply(vDl, Avx.ConvertToVector256Single(q3));
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff), acc);
                        elemOff += 8;
                    }

                    // Second 16 elements (qs + 16, hm + 16)
                    dl = dAll * (scales[isIdx++] - 32);
                    vDl = Vector256.Create(dl);

                    for (int l = 0; l < 16; l += 8)
                    {
                        var qBytes = LoadBytes8(qs + qOff + 16 + l);
                        var qInts = Avx2.ConvertToVector256Int32(qBytes);
                        var shifted = j switch {
                            0 => qInts,
                            1 => Avx2.ShiftRightLogical(qInts, 2),
                            2 => Avx2.ShiftRightLogical(qInts, 4),
                            _ => Avx2.ShiftRightLogical(qInts, 6),
                        };
                        var q2 = Avx2.And(shifted, mask03);

                        var hmBytes = LoadBytes8(hm + 16 + l);
                        var hmInts = Avx2.ConvertToVector256Int32(hmBytes);
                        var hmBit = Avx2.And(hmInts, Vector256.Create((int)m));
                        var sub = Avx2.And(Avx2.CompareEqual(hmBit, Vector256<int>.Zero), four);
                        var q3 = Avx2.Subtract(q2, sub);

                        var deq = Avx.Multiply(vDl, Avx.ConvertToVector256Single(q3));
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff), acc);
                        elemOff += 8;
                    }

                    m <<= 1;
                }
                qOff += 32;
            }
        }

        return HSum256(acc);
    }

    private static float DotQ3K_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float acc = 0;
        int elemOff = 0;
        Span<uint> aux = stackalloc uint[4];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);

            aux[0] = *(uint*)(x + 96); aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

            byte* qs = x + 32; byte* hm = x;
            int qOff = 0; int isIdx = 0; byte m = 1;

            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    float dl = dAll * (scByte - 32); isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((qs[qOff + l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4);
                        acc += dl * q * input[elemOff++];
                    }
                    scByte = (int)(byte)((aux[isIdx / 4] >> ((isIdx % 4) * 8)) & 0xFF);
                    dl = dAll * (scByte - 32); isIdx++;
                    for (int l = 0; l < 16; l++)
                    {
                        int q = ((qs[qOff + l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4);
                        acc += dl * q * input[elemOff++];
                    }
                    shift += 2; m <<= 1;
                }
                qOff += 32;
            }
        }
        return acc;
    }

    // ================================================================
    //  Q2_K Fused MatVec
    // ================================================================

    public static void MatVecQ2K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 84;

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var inp = input; var outp = output;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ2K(w + (long)i * bytesPerRow, inp, cols);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ2K(weights + (long)i * bytesPerRow, input, cols);
        }
    }

    // ================================================================
    //  Q2_K Fused Dequant-Dot  (one row)
    // ================================================================

    /// <summary>
    /// Fused Q2_K dequant-dot with AVX2. Block = 84 bytes / 256 elements.
    /// Layout: [scales:16][qs:64][d:FP16][dmin:FP16].
    /// The 64 qs bytes are read 4 times with shifts 0,2,4,6 per 128-element group.
    /// </summary>
    public static float DotQ2K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (!Fma.IsSupported)
            return DotQ2K_Scalar(row, input, cols, numBlocks);

        var acc = Vector256<float>.Zero;
        var mask03 = Vector256.Create(0x03);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 84;
            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);
            byte* sc = x;       // scales at byte 0
            byte* qs = x + 16;  // qs at byte 16

            int qOff = 0;
            int isIdx = 0;
            for (int n = 0; n < 256; n += 128)
            {
                // Unrolled: 4 shifts (0, 2, 4, 6) as constants
                for (int j = 0; j < 4; j++)
                {
                    byte scByte = sc[isIdx++];
                    var dl = Vector256.Create(d * (scByte & 0xF));
                    var negMl = Vector256.Create(-(min * (scByte >> 4)));

                    for (int l = 0; l < 16; l += 8)
                    {
                        var bytes = LoadBytes8(qs + qOff + l);
                        var ints = Avx2.ConvertToVector256Int32(bytes);
                        // Shift by constant: j=0→0, j=1→2, j=2→4, j=3→6
                        var shifted = j switch {
                            0 => ints,
                            1 => Avx2.ShiftRightLogical(ints, 2),
                            2 => Avx2.ShiftRightLogical(ints, 4),
                            _ => Avx2.ShiftRightLogical(ints, 6),
                        };
                        var q = Avx2.And(shifted, mask03);
                        var deq = Fma.MultiplyAdd(dl, Avx.ConvertToVector256Single(q), negMl);
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff + n + j * 32 + l), acc);
                    }

                    scByte = sc[isIdx++];
                    dl = Vector256.Create(d * (scByte & 0xF));
                    negMl = Vector256.Create(-(min * (scByte >> 4)));

                    for (int l = 0; l < 16; l += 8)
                    {
                        var bytes = LoadBytes8(qs + qOff + 16 + l);
                        var ints = Avx2.ConvertToVector256Int32(bytes);
                        var shifted = j switch {
                            0 => ints,
                            1 => Avx2.ShiftRightLogical(ints, 2),
                            2 => Avx2.ShiftRightLogical(ints, 4),
                            _ => Avx2.ShiftRightLogical(ints, 6),
                        };
                        var q = Avx2.And(shifted, mask03);
                        var deq = Fma.MultiplyAdd(dl, Avx.ConvertToVector256Single(q), negMl);
                        acc = Fma.MultiplyAdd(deq, Avx.LoadVector256(input + elemOff + n + j * 32 + 16 + l), acc);
                    }
                }
                qOff += 32;
            }
            elemOff += 256;
        }

        return HSum256(acc);
    }

    private static float DotQ2K_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 84;
            float d = HalfToFloat(x[80], x[81]);
            float min = HalfToFloat(x[82], x[83]);
            byte* sc = x;
            byte* qs = x + 16;

            int qOff = 0;
            int isIdx = 0;
            int yOff = elemOff;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte scByte = sc[isIdx++];
                    float dl = d * (scByte & 0xF);
                    float ml = min * (scByte >> 4);
                    for (int l = 0; l < 16; l++)
                        acc += (dl * ((qs[qOff + l] >> shift) & 3) - ml) * input[yOff++];

                    scByte = sc[isIdx++];
                    dl = d * (scByte & 0xF);
                    ml = min * (scByte >> 4);
                    for (int l = 0; l < 16; l++)
                        acc += (dl * ((qs[qOff + l + 16] >> shift) & 3) - ml) * input[yOff++];

                    shift += 2;
                }
                qOff += 32;
            }
            elemOff += 256;
        }
        return acc;
    }

    // ================================================================
    //  Q6_K Fused Dequant-Dot  (one row)
    // ================================================================

    public static float DotQ6K(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
            return DotQ6K_Avx512(row, input, cols, numBlocks);

        if (!Fma.IsSupported)
            return DotQ6K_Scalar(row, input, cols);

        // Four independent accumulators (one per output group) to break dependency chains
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;
        var acc4 = Vector256<float>.Zero;
        var mask0F = Vector256.Create(0x0F);
        var mask03 = Vector256.Create(0x03);
        var sub32 = Vector256.Create(32);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 210;
            byte* ql = x;
            byte* qh = x + 128;
            byte* sc = x + 192;
            float d = HalfToFloat(x[208], x[209]);

            int qlOff = 0, qhOff = 0, scBase = 0;

            for (int half = 0; half < 2; half++)
            {
                for (int phase = 0; phase < 2; phase++)
                {
                    int lStart = phase * 16;
                    var s1v = Vector256.Create(d * (sbyte)sc[scBase + phase]);
                    var s2v = Vector256.Create(d * (sbyte)sc[scBase + phase + 2]);
                    var s3v = Vector256.Create(d * (sbyte)sc[scBase + phase + 4]);
                    var s4v = Vector256.Create(d * (sbyte)sc[scBase + phase + 6]);

                    for (int l = lStart; l < lStart + 16; l += 8)
                    {
                        var qlA = Avx2.ConvertToVector256Int32(LoadBytes8(ql + qlOff + l));
                        var qlB = Avx2.ConvertToVector256Int32(LoadBytes8(ql + qlOff + 32 + l));
                        var qhV = Avx2.ConvertToVector256Int32(LoadBytes8(qh + qhOff + l));

                        // Group 1 → acc1
                        var g1 = Avx2.Subtract(
                            Avx2.Or(Avx2.And(qlA, mask0F),
                                Avx2.ShiftLeftLogical(Avx2.And(qhV, mask03), 4)),
                            sub32);
                        acc1 = Fma.MultiplyAdd(
                            Avx.Multiply(s1v, Avx.ConvertToVector256Single(g1)),
                            Avx.LoadVector256(input + elemOff + l), acc1);

                        // Group 2 → acc2
                        var g2 = Avx2.Subtract(
                            Avx2.Or(Avx2.And(qlB, mask0F),
                                Avx2.ShiftLeftLogical(Avx2.And(
                                    Avx2.ShiftRightLogical(qhV, 2), mask03), 4)),
                            sub32);
                        acc2 = Fma.MultiplyAdd(
                            Avx.Multiply(s2v, Avx.ConvertToVector256Single(g2)),
                            Avx.LoadVector256(input + elemOff + 32 + l), acc2);

                        // Group 3 → acc3
                        var g3 = Avx2.Subtract(
                            Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(qlA, 4), mask0F),
                                Avx2.ShiftLeftLogical(Avx2.And(
                                    Avx2.ShiftRightLogical(qhV, 4), mask03), 4)),
                            sub32);
                        acc3 = Fma.MultiplyAdd(
                            Avx.Multiply(s3v, Avx.ConvertToVector256Single(g3)),
                            Avx.LoadVector256(input + elemOff + 64 + l), acc3);

                        // Group 4 → acc4
                        var g4 = Avx2.Subtract(
                            Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(qlB, 4), mask0F),
                                Avx2.ShiftLeftLogical(Avx2.And(
                                    Avx2.ShiftRightLogical(qhV, 6), mask03), 4)),
                            sub32);
                        acc4 = Fma.MultiplyAdd(
                            Avx.Multiply(s4v, Avx.ConvertToVector256Single(g4)),
                            Avx.LoadVector256(input + elemOff + 96 + l), acc4);
                    }
                }
                elemOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }
        }

        return HSum256(Avx.Add(Avx.Add(acc1, acc2), Avx.Add(acc3, acc4)));
    }

    private static float DotQ6K_Avx512(byte* row, float* input, int cols, int numBlocks)
    {
        var acc1 = Vector512<float>.Zero;
        var acc2 = Vector512<float>.Zero;
        var acc3 = Vector512<float>.Zero;
        var acc4 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var mask03 = Vector512.Create(0x03);
        var sub32 = Vector512.Create(32);
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 210;
            byte* ql = x;
            byte* qh = x + 128;
            byte* sc = x + 192;
            float d = HalfToFloat(x[208], x[209]);

            int qlOff = 0, qhOff = 0, scBase = 0;

            for (int half = 0; half < 2; half++)
            {
                for (int phase = 0; phase < 2; phase++)
                {
                    int lStart = phase * 16;
                    var s1v = Vector512.Create(d * (sbyte)sc[scBase + phase]);
                    var s2v = Vector512.Create(d * (sbyte)sc[scBase + phase + 2]);
                    var s3v = Vector512.Create(d * (sbyte)sc[scBase + phase + 4]);
                    var s4v = Vector512.Create(d * (sbyte)sc[scBase + phase + 6]);

                    // 16 elements per Vector512 — entire phase in one iteration
                    var qlA = Avx512F.ConvertToVector512Int32(
                        Vector128.LoadUnsafe(ref *(ql + qlOff + lStart)));
                    var qlB = Avx512F.ConvertToVector512Int32(
                        Vector128.LoadUnsafe(ref *(ql + qlOff + 32 + lStart)));
                    var qhV = Avx512F.ConvertToVector512Int32(
                        Vector128.LoadUnsafe(ref *(qh + qhOff + lStart)));

                    var g1 = Avx512F.Subtract(
                        Avx512F.Or(Avx512F.And(qlA, mask0F),
                            Avx512F.ShiftLeftLogical(Avx512F.And(qhV, mask03), 4)),
                        sub32);
                    acc1 = Avx512F.FusedMultiplyAdd(
                        Avx512F.Multiply(s1v, Avx512F.ConvertToVector512Single(g1)),
                        Vector512.LoadUnsafe(ref *(input + elemOff + lStart)), acc1);

                    var g2 = Avx512F.Subtract(
                        Avx512F.Or(Avx512F.And(qlB, mask0F),
                            Avx512F.ShiftLeftLogical(Avx512F.And(
                                Avx512F.ShiftRightLogical(qhV, 2), mask03), 4)),
                        sub32);
                    acc2 = Avx512F.FusedMultiplyAdd(
                        Avx512F.Multiply(s2v, Avx512F.ConvertToVector512Single(g2)),
                        Vector512.LoadUnsafe(ref *(input + elemOff + 32 + lStart)), acc2);

                    var g3 = Avx512F.Subtract(
                        Avx512F.Or(Avx512F.And(Avx512F.ShiftRightLogical(qlA, 4), mask0F),
                            Avx512F.ShiftLeftLogical(Avx512F.And(
                                Avx512F.ShiftRightLogical(qhV, 4), mask03), 4)),
                        sub32);
                    acc3 = Avx512F.FusedMultiplyAdd(
                        Avx512F.Multiply(s3v, Avx512F.ConvertToVector512Single(g3)),
                        Vector512.LoadUnsafe(ref *(input + elemOff + 64 + lStart)), acc3);

                    var g4 = Avx512F.Subtract(
                        Avx512F.Or(Avx512F.And(Avx512F.ShiftRightLogical(qlB, 4), mask0F),
                            Avx512F.ShiftLeftLogical(Avx512F.And(
                                Avx512F.ShiftRightLogical(qhV, 6), mask03), 4)),
                        sub32);
                    acc4 = Avx512F.FusedMultiplyAdd(
                        Avx512F.Multiply(s4v, Avx512F.ConvertToVector512Single(g4)),
                        Vector512.LoadUnsafe(ref *(input + elemOff + 96 + lStart)), acc4);
                }
                elemOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }
        }

        return HSum512(Avx512F.Add(Avx512F.Add(acc1, acc2), Avx512F.Add(acc3, acc4)));
    }

    private static float DotQ6K_Scalar(byte* row, float* input, int cols)
    {
        int numBlocks = cols / 256;
        float acc = 0;
        int elemOff = 0;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 210;
            byte* ql = x;
            byte* qh = x + 128;
            byte* sc = x + 192;
            float d = HalfToFloat(x[208], x[209]);

            int qlOff = 0, qhOff = 0, scBase = 0;

            for (int half = 0; half < 2; half++)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16;
                    int q1 = ((ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                    acc += d * (sbyte)sc[scBase + isc] * q1 * input[elemOff + l];
                    acc += d * (sbyte)sc[scBase + isc + 2] * q2 * input[elemOff + 32 + l];
                    acc += d * (sbyte)sc[scBase + isc + 4] * q3 * input[elemOff + 64 + l];
                    acc += d * (sbyte)sc[scBase + isc + 6] * q4 * input[elemOff + 96 + l];
                }
                elemOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }
        }
        return acc;
    }

    // ================================================================
    //  RMS Norm (AVX2)
    // ================================================================

    public static void RmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        if (Fma.IsSupported && size >= 8)
        {
            // Pass 1: sum of squares
            var sumSq = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                sumSq = Fma.MultiplyAdd(v, v, sumSq);
            }
            float ss = HSum256(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector256.Create(scale);

            // Pass 2: scale and weight
            i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                var w = Avx.LoadVector256(weight + i);
                Avx.Store(output + i, Avx.Multiply(Avx.Multiply(v, scaleV), w));
            }
            for (; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
        else
        {
            float ss = 0;
            for (int i = 0; i < size; i++) ss += input[i] * input[i];
            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            for (int i = 0; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
    }

    /// <summary>
    /// RMS normalization without learned weights (pure L2 normalize).
    /// Used for Llama4TextL2Norm in QK-norm.
    /// </summary>
    public static void PureRmsNorm(float* output, float* input, int size, float eps)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var sumSq = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(input + i);
                sumSq = Fma.MultiplyAdd(v, v, sumSq);
            }
            float ss = HSum256(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector256.Create(scale);

            i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(output + i, Avx.Multiply(Avx.LoadVector256(input + i), scaleV));
            for (; i < size; i++)
                output[i] = input[i] * scale;
        }
        else
        {
            float ss = 0;
            for (int i = 0; i < size; i++) ss += input[i] * input[i];
            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            for (int i = 0; i < size; i++)
                output[i] = input[i] * scale;
        }
    }

    // ================================================================
    //  Softmax (AVX2 with scalar exp)
    // ================================================================

    public static void SoftmaxInPlace(float* x, int size)
    {
        if (Avx.IsSupported && size >= 8)
        {
            // Pass 1: find max
            var maxV = Vector256.Create(float.NegativeInfinity);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                maxV = Avx.Max(maxV, Avx.LoadVector256(x + i));
            float max = HMax256(maxV);
            for (; i < size; i++)
                if (x[i] > max) max = x[i];

            // Pass 2: exp(x - max) and sum
            var maxBcast = Vector256.Create(max);
            var sumV = Vector256<float>.Zero;
            i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.Subtract(Avx.LoadVector256(x + i), maxBcast);
                var e = ExpApprox256(v);
                Avx.Store(x + i, e);
                sumV = Avx.Add(sumV, e);
            }
            float sum = HSum256(sumV);
            for (; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }

            // Pass 3: normalize
            var invSum = Vector256.Create(1.0f / sum);
            i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), invSum));
            float invSumS = 1.0f / sum;
            for (; i < size; i++)
                x[i] *= invSumS;
        }
        else
        {
            float max = float.NegativeInfinity;
            for (int i = 0; i < size; i++)
                if (x[i] > max) max = x[i];
            float sum = 0;
            for (int i = 0; i < size; i++)
            {
                x[i] = MathF.Exp(x[i] - max);
                sum += x[i];
            }
            float inv = 1.0f / sum;
            for (int i = 0; i < size; i++) x[i] *= inv;
        }
    }

    /// <summary>
    /// In-place element-wise sigmoid: x[i] = 1 / (1 + exp(-x[i])).
    /// Used for Llama-4 MoE router gating.
    /// </summary>
    public static void SigmoidInPlace(float* x, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var one = Vector256.Create(1.0f);
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var v = Avx.LoadVector256(x + i);
                var negV = Avx.Subtract(Vector256<float>.Zero, v);
                var expNeg = ExpApprox256(negV);
                var sig = Avx.Divide(one, Avx.Add(one, expNeg));
                Avx.Store(x + i, sig);
            }
            for (; i < size; i++)
                x[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
        else
        {
            for (int i = 0; i < size; i++)
                x[i] = 1.0f / (1.0f + MathF.Exp(-x[i]));
        }
    }

    // ================================================================
    //  Fused SiLU(gate) * up  (AVX2)
    // ================================================================

    public static void SiLuMul(float* gate, float* up, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var one = Vector256.Create(1.0f);
            int i = 0;
            for (; i + 8 <= size; i += 8)
            {
                var g = Avx.LoadVector256(gate + i);
                var u = Avx.LoadVector256(up + i);
                // sigmoid(g) = 1 / (1 + exp(-g))
                var negG = Avx.Subtract(Vector256<float>.Zero, g);
                var expNg = ExpApprox256(negG);
                var sigmoid = Avx.Divide(one, Avx.Add(one, expNg));
                // SiLU = g * sigmoid(g) * up
                Avx.Store(gate + i, Avx.Multiply(Avx.Multiply(g, sigmoid), u));
            }
            for (; i < size; i++)
            {
                float g = gate[i];
                gate[i] = g / (1.0f + MathF.Exp(-g)) * up[i];
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                float g = gate[i];
                gate[i] = g / (1.0f + MathF.Exp(-g)) * up[i];
            }
        }
    }

    // ================================================================
    //  Add in-place (AVX2)
    // ================================================================

    public static void AddInPlace(float* dst, float* src, int size)
    {
        if (Avx.IsSupported)
        {
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(dst + i, Avx.Add(Avx.LoadVector256(dst + i), Avx.LoadVector256(src + i)));
            for (; i < size; i++) dst[i] += src[i];
        }
        else
        {
            for (int i = 0; i < size; i++) dst[i] += src[i];
        }
    }

    /// <summary>Multiply every element of <paramref name="x"/> by a scalar.</summary>
    public static void ScaleInPlace(float* x, float scale, int size)
    {
        if (Avx.IsSupported)
        {
            var sv = Vector256.Create(scale);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(x + i, Avx.Multiply(Avx.LoadVector256(x + i), sv));
            for (; i < size; i++) x[i] *= scale;
        }
        else
        {
            for (int i = 0; i < size; i++) x[i] *= scale;
        }
    }

    /// <summary>Weighted accumulate in-place: dst[i] += weight * src[i].</summary>
    public static void WeightedAddInPlace(float* dst, float* src, float weight, int size)
    {
        if (Fma.IsSupported && size >= 8)
        {
            var wv = Vector256.Create(weight);
            int i = 0;
            for (; i + 8 <= size; i += 8)
                Avx.Store(dst + i, Fma.MultiplyAdd(wv, Avx.LoadVector256(src + i), Avx.LoadVector256(dst + i)));
            for (; i < size; i++)
                dst[i] += weight * src[i];
        }
        else
        {
            for (int i = 0; i < size; i++)
                dst[i] += weight * src[i];
        }
    }

    // ================================================================
    //  RoPE (precomputed sin/cos, SIMD rotation)
    // ================================================================

    /// <summary>
    /// Apply RoPE using precomputed cos/sin tables (avoids recomputing trig 48× per token).
    /// </summary>
    public static void ApplyRoPECached(float* x, float* cosTab, float* sinTab, int numHeads, int headDim)
    {
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            if (Avx.IsSupported && halfDim >= 4)
            {
                int i = 0;
                for (; i + 4 <= halfDim; i += 4)
                {
                    var v = Avx.LoadVector256(head + 2 * i);
                    var c = Vector256.Create(cosTab[i], cosTab[i], cosTab[i + 1], cosTab[i + 1],
                                             cosTab[i + 2], cosTab[i + 2], cosTab[i + 3], cosTab[i + 3]);
                    var s = Vector256.Create(sinTab[i], sinTab[i], sinTab[i + 1], sinTab[i + 1],
                                             sinTab[i + 2], sinTab[i + 2], sinTab[i + 3], sinTab[i + 3]);
                    var swapped = Avx.Shuffle(v, v, 0b10_11_00_01);
                    var signMask = Vector256.Create(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
                    var result = Fma.MultiplyAdd(v, c, Avx.Multiply(swapped, Avx.Multiply(s, signMask)));
                    Avx.Store(head + 2 * i, result);
                }
                for (; i < halfDim; i++)
                {
                    float x0 = head[2 * i], x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
            else
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i], x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
        }
    }

    /// <summary>
    /// NEOX-style RoPE (used by Qwen, Phi, Gemma, Falcon, etc.):
    /// rotates dim pair (i, i + headDim/2) instead of consecutive (2i, 2i+1).
    /// </summary>
    public static void ApplyRoPECachedNeox(float* x, float* cosTab, float* sinTab, int numHeads, int headDim)
    {
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            int i = 0;
            if (Avx.IsSupported)
            {
                for (; i + 8 <= halfDim; i += 8)
                {
                    var x0 = Avx.LoadVector256(head + i);
                    var x1 = Avx.LoadVector256(head + i + halfDim);
                    var c = Avx.LoadVector256(cosTab + i);
                    var s = Avx.LoadVector256(sinTab + i);
                    var r0 = Fma.MultiplySubtract(x0, c, Avx.Multiply(x1, s));
                    var r1 = Fma.MultiplyAdd(x0, s, Avx.Multiply(x1, c));
                    Avx.Store(head + i, r0);
                    Avx.Store(head + i + halfDim, r1);
                }
            }
            for (; i < halfDim; i++)
            {
                float x0 = head[i], x1 = head[i + halfDim];
                head[i] = x0 * cosTab[i] - x1 * sinTab[i];
                head[i + halfDim] = x0 * sinTab[i] + x1 * cosTab[i];
            }
        }
    }

    public static void ApplyRoPE(float* x, int position, int numHeads, int headDim, float theta)
    {
        int halfDim = headDim / 2;

        // Precompute cos/sin tables (shared across all heads)
        float* cosTab = stackalloc float[halfDim];
        float* sinTab = stackalloc float[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            float freq = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
            float angle = position * freq;
            cosTab[i] = MathF.Cos(angle);
            sinTab[i] = MathF.Sin(angle);
        }

        // Apply rotation to all heads
        // Interleaved pairs: rotate (x[2i], x[2i+1])
        // Reinterpret as pairs and apply rotation using cos/sin tables
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;

            if (Avx.IsSupported && halfDim >= 4)
            {
                // Process 4 pairs (8 floats) at a time
                int i = 0;
                for (; i + 4 <= halfDim; i += 4)
                {
                    // Load 8 consecutive floats: (x0,x1, x2,x3, x4,x5, x6,x7)
                    var v = Avx.LoadVector256(head + 2 * i);
                    var c = Vector256.Create(cosTab[i], cosTab[i], cosTab[i + 1], cosTab[i + 1],
                                             cosTab[i + 2], cosTab[i + 2], cosTab[i + 3], cosTab[i + 3]);
                    var s = Vector256.Create(sinTab[i], sinTab[i], sinTab[i + 1], sinTab[i + 1],
                                             sinTab[i + 2], sinTab[i + 2], sinTab[i + 3], sinTab[i + 3]);

                    // Even elements (x0, x2, x4, x6) and odd elements (x1, x3, x5, x7)
                    // x0' = x0*cos - x1*sin,  x1' = x0*sin + x1*cos
                    // Shuffle to get (x1,x0, x3,x2, x5,x4, x7,x6)
                    var swapped = Avx.Shuffle(v, v, 0b10_11_00_01);
                    // Signs: (-sin, sin, -sin, sin, ...) for even positions,
                    //        (cos, cos, cos, cos, ...) already correct
                    // Actually: result = v*cos + swapped * (-sin_even, sin_odd, ...)
                    var signMask = Vector256.Create(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
                    var sFlipped = Avx.Multiply(s, signMask);
                    var result = Fma.MultiplyAdd(v, c, Avx.Multiply(swapped, sFlipped));
                    Avx.Store(head + 2 * i, result);
                }
                // Scalar remainder
                for (; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
            else
            {
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTab[i] - x1 * sinTab[i];
                    head[2 * i + 1] = x0 * sinTab[i] + x1 * cosTab[i];
                }
            }
        }
    }

    /// <summary>
    /// Precompute RoPE cos/sin tables for all positions [0, maxSeqLen).
    /// cosOut and sinOut must each point to maxSeqLen * (headDim / 2) floats.
    /// </summary>
    public static void BuildRopeTable(float* cosOut, float* sinOut, int maxSeqLen, int headDim, float theta)
    {
        int halfDim = headDim / 2;
        float* freqs = stackalloc float[halfDim];
        for (int i = 0; i < halfDim; i++)
            freqs[i] = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);

        for (int p = 0; p < maxSeqLen; p++)
        {
            float* c = cosOut + (long)p * halfDim;
            float* s = sinOut + (long)p * halfDim;
            for (int i = 0; i < halfDim; i++)
            {
                float angle = p * freqs[i];
                c[i] = MathF.Cos(angle);
                s[i] = MathF.Sin(angle);
            }
        }
    }

    // ================================================================
    //  Single-row dequantization (for embedding lookup)
    // ================================================================

    /// <summary>
    /// Dequantize a single row from a quantized 2D tensor.
    /// rowData points to (cols/blockSize)*bytesPerBlock bytes.
    /// </summary>
    public static void DequantRow(byte* rowData, float* output, int cols, DType dtype)
    {
        Dequantize.ToFloat32(
            new ReadOnlySpan<byte>(rowData, (cols / DTypeInfo.BlockSize(dtype)) * DTypeInfo.BytesPerBlock(dtype)),
            new Span<float>(output, cols),
            dtype, cols);
    }

    // ================================================================
    //  Helpers
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v)
    {
        var hi = Avx.ExtractVector128(v, 1);
        var lo = v.GetLower();
        var sum = Sse.Add(lo, hi);
        sum = Sse.Add(sum, Sse.MoveHighToLow(sum, sum));
        sum = Sse.AddScalar(sum, Sse.Shuffle(sum, sum, 1));
        return sum.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HMax256(Vector256<float> v)
    {
        var hi = Avx.ExtractVector128(v, 1);
        var lo = v.GetLower();
        var m = Sse.Max(lo, hi);
        m = Sse.Max(m, Sse.MoveHighToLow(m, m));
        m = Sse.Max(m, Sse.Shuffle(m, m, 1));
        return m.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HalfToFloat(byte lo, byte hi)
    {
        ushort bits = (ushort)(lo | (hi << 8));
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void GetScaleMinK4(int j, byte* q, out byte scale, out byte min)
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

    /// <summary>Load 8 bytes into a Vector128 for vpmovzxbd.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> LoadBytes8(byte* ptr)
    {
        return Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(ptr)).AsByte();
    }

    /// <summary>
    /// Fast exp approximation for Vector256 using the standard
    /// Cephes-style range reduction + polynomial.
    /// Max relative error ~1.5e-7 in [-87, 88].
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ExpApprox256(Vector256<float> x)
    {
        // Clamp to avoid overflow/underflow
        x = Avx.Max(x, Vector256.Create(-87.3365f));
        x = Avx.Min(x, Vector256.Create(88.7228f));

        // t = x / ln(2)
        var t = Avx.Multiply(x, Vector256.Create(1.44269504088896341f));

        // Round to nearest integer
        var ti = Avx.RoundToNearestInteger(t);
        var n = Avx.ConvertToVector256Int32(ti);

        // Fractional part: f = t - round(t)
        var f = Avx.Subtract(t, ti);

        // Polynomial approximation of 2^f on [-0.5, 0.5]
        // Coefficients: Taylor series of 2^x = e^(x·ln2), i.e. (ln2)^k / k!
        var p = Vector256.Create(1.5403530e-4f);
        p = Fma.MultiplyAdd(p, f, Vector256.Create(1.3333558e-3f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(9.6181291e-3f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(5.5504109e-2f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(2.4022651e-1f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(6.9314718e-1f));
        p = Fma.MultiplyAdd(p, f, Vector256.Create(1.0f));

        // 2^n via IEEE 754 exponent manipulation
        var pow2n = Avx2.ShiftLeftLogical(Avx2.Add(n, Vector256.Create(127)), 23).AsSingle();
        return Avx.Multiply(p, pow2n);
    }
}
