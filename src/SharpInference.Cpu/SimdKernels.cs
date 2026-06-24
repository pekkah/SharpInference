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

    /// <summary>Whether OpenBLAS was found, i.e. whether the SGEMM batched-prefill path is live.</summary>
    public static bool BlasAvailable => BlasInterop.IsAvailable;

    /// <summary>
    /// Batched matrix multiply against an <b>already-dequantized F32</b> weight matrix —
    /// the dequant-free twin of <see cref="MatMulBatched"/>. Issue #189: chunked prompt
    /// admission re-walks the same layer weights every chunk, so <see cref="MatMulBatched"/>
    /// re-pays the full Q→F32 dequant on every call. When a caller (ForwardPass) holds the
    /// F32 dequant of a weight in a reuse cache, it routes here to skip dequant entirely.
    /// Bit-identical to <see cref="MatMulBatched"/>'s BLAS path: same F32 weights, same SGEMM.
    /// </summary>
    public static void MatMulBatchedF32(float* output, float* weightsF32, float* input,
        int batchSize, int rows, int cols)
    {
        if (batchSize < MinBatchForBlas || !BlasInterop.IsAvailable)
        {
            // Mirror the small-batch / no-BLAS fallback, but the weights are already F32.
            for (int n = 0; n < batchSize; n++)
                MatVecF32(output + n * rows, weightsF32, input + n * cols, rows, cols);
            return;
        }

        BlasInterop.Sgemm(
            BlasInterop.RowMajor, BlasInterop.NoTrans, BlasInterop.Trans,
            batchSize, rows, cols,
            1.0f, input, cols,
            weightsF32, cols,
            0.0f, output, rows);
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
                int scratchBytes = Q8KScratchBytes(cols);
                byte* scratch = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input, cols, scratch);

                if (rows >= MinRowsForParallel)
                {
                    var w1 = weights1; var w2 = weights2; var s = scratch;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        o1[r] = DotQ6K_Q8K(w1 + (long)r * bpr, s, c);
                        o2[r] = DotQ6K_Q8K(w2 + (long)r * bpr, s, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        output1[r] = DotQ6K_Q8K(weights1 + (long)r * bpr, scratch, cols);
                        output2[r] = DotQ6K_Q8K(weights2 + (long)r * bpr, scratch, cols);
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

    /// <summary>
    /// Compute two matrix-vector products sharing the same weight matrix against
    /// two distinct inputs in a single Parallel.For sweep:
    /// <c>output1 = weights @ input1</c> and <c>output2 = weights @ input2</c>.
    /// Each weight row is touched once per row iteration; the second dot reads the
    /// just-loaded row from L1, halving the effective weight-bandwidth cost of the
    /// pair vs two sequential <see cref="MatVec"/> calls. Used by the MTP batched
    /// verify path (issue #30) where both tokens share the same FFN weights.
    /// </summary>
    public static void MatVec2In(
        float* output1, float* output2,
        byte* weights, float* input1, float* input2,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ4K_2In(row, i1, i2, c, out float s1, out float s2);
                        o1[r] = s1; o2[r] = s2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ4K_2In(row, input1, input2, cols, out float s1, out float s2);
                        output1[r] = s1; output2[r] = s2;
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ5K_2In(row, i1, i2, c, out float s1, out float s2);
                        o1[r] = s1; o2[r] = s2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ5K_2In(row, input1, input2, cols, out float s1, out float s2);
                        output1[r] = s1; output2[r] = s2;
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                int scratchBytes = Q8KScratchBytes(cols);
                // Two Q8_K scratches (one per input); stack-alloc when small enough,
                // heap fallback for large cols (Q8_K scratch is ~262 B per 256 elems).
                byte* sc1 = stackalloc byte[scratchBytes];
                byte* sc2 = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input1, cols, sc1);
                QuantizeRowToQ8K(input2, cols, sc2);

                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var s1 = sc1; var s2 = sc2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ6K_Q8K_2In(row, s1, s2, c, out float v1, out float v2);
                        o1[r] = v1;
                        o2[r] = v2;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ6K_Q8K_2In(row, sc1, sc2, cols, out float v1, out float v2);
                        output1[r] = v1;
                        output2[r] = v2;
                    }
                }
                break;
            }
            case DType.Float32:
            {
                var m = (float*)weights;
                if (rows >= MinRowsForParallel)
                {
                    var i1 = input1; var i2 = input2;
                    var o1 = output1; var o2 = output2; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        float* row = m + (long)r * c;
                        o1[r] = DotF32(row, i1, c);
                        o2[r] = DotF32(row, i2, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = m + (long)r * cols;
                        output1[r] = DotF32(row, input1, cols);
                        output2[r] = DotF32(row, input2, cols);
                    }
                }
                break;
            }
            default:
                // Fallback: two sequential MatVec calls. Loses the weight-bandwidth
                // benefit but stays correct for dtypes we haven't specialised yet.
                MatVec(output1, weights, input1, rows, cols, dtype);
                MatVec(output2, weights, input2, rows, cols, dtype);
                break;
        }
    }

    /// <summary>
    /// Four-input fused mat-vec (issue #209): for each weight row
    /// <c>output{0..3}[r] = weights[r] · input{0..3}</c>. Decodes each weight row
    /// ONCE and dots it against four token columns in the same pass — one weight
    /// HBM/L2 read per four tokens versus <see cref="MatVec2In"/>'s one-per-two. This
    /// is the dominant lever on the 27B-MTP CUDA-hybrid decode path, where 46/64 dense
    /// FFN layers are CPU-mmap'd and re-read once per draft token. Per-token
    /// accumulation order is identical to <see cref="MatVec2In"/> / single
    /// <see cref="MatVec"/>, so each position's bits are independent of the batch width
    /// k (the duplicated-input-tail contract — see the BatchVerify callers, which fill
    /// past-the-end lanes with a duplicate token routed to a sink).
    /// </summary>
    public static void MatVec4In(
        float* output0, float* output1, float* output2, float* output3,
        byte* weights,
        float* input0, float* input1, float* input2, float* input3,
        int rows, int cols, DType dtype)
    {
        switch (dtype)
        {
            case DType.Q4_K:
            {
                int bpr = (cols / 256) * 144;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ4K_4In(row, i0, i1, i2, i3, c, out float s0, out float s1, out float s2, out float s3);
                        o0[r] = s0; o1[r] = s1; o2[r] = s2; o3[r] = s3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ4K_4In(row, input0, input1, input2, input3, cols, out float s0, out float s1, out float s2, out float s3);
                        output0[r] = s0; output1[r] = s1; output2[r] = s2; output3[r] = s3;
                    }
                }
                break;
            }
            case DType.Q5_K:
            {
                int bpr = (cols / 256) * 176;
                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ5K_4In(row, i0, i1, i2, i3, c, out float s0, out float s1, out float s2, out float s3);
                        o0[r] = s0; o1[r] = s1; o2[r] = s2; o3[r] = s3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ5K_4In(row, input0, input1, input2, input3, cols, out float s0, out float s1, out float s2, out float s3);
                        output0[r] = s0; output1[r] = s1; output2[r] = s2; output3[r] = s3;
                    }
                }
                break;
            }
            case DType.Q6_K:
            {
                int bpr = (cols / 256) * 210;
                int scratchBytes = Q8KScratchBytes(cols);
                // One Q8_K scratch per input; same stack-alloc discipline as MatVec2In.
                byte* sc0 = stackalloc byte[scratchBytes];
                byte* sc1 = stackalloc byte[scratchBytes];
                byte* sc2 = stackalloc byte[scratchBytes];
                byte* sc3 = stackalloc byte[scratchBytes];
                QuantizeRowToQ8K(input0, cols, sc0);
                QuantizeRowToQ8K(input1, cols, sc1);
                QuantizeRowToQ8K(input2, cols, sc2);
                QuantizeRowToQ8K(input3, cols, sc3);

                if (rows >= MinRowsForParallel)
                {
                    var w = weights; var s0 = sc0; var s1 = sc1; var s2 = sc2; var s3 = sc3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        byte* row = w + (long)r * bpr;
                        DotQ6K_Q8K_4In(row, s0, s1, s2, s3, c, out float v0, out float v1, out float v2, out float v3);
                        o0[r] = v0; o1[r] = v1; o2[r] = v2; o3[r] = v3;
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        byte* row = weights + (long)r * bpr;
                        DotQ6K_Q8K_4In(row, sc0, sc1, sc2, sc3, cols, out float v0, out float v1, out float v2, out float v3);
                        output0[r] = v0; output1[r] = v1; output2[r] = v2; output3[r] = v3;
                    }
                }
                break;
            }
            case DType.Float32:
            {
                var m = (float*)weights;
                if (rows >= MinRowsForParallel)
                {
                    var i0 = input0; var i1 = input1; var i2 = input2; var i3 = input3;
                    var o0 = output0; var o1 = output1; var o2 = output2; var o3 = output3; int c = cols;
                    Parallel.For(0, rows, s_parallelOpts, r =>
                    {
                        float* row = m + (long)r * c;
                        o0[r] = DotF32(row, i0, c);
                        o1[r] = DotF32(row, i1, c);
                        o2[r] = DotF32(row, i2, c);
                        o3[r] = DotF32(row, i3, c);
                    });
                }
                else
                {
                    for (int r = 0; r < rows; r++)
                    {
                        float* row = m + (long)r * cols;
                        output0[r] = DotF32(row, input0, cols);
                        output1[r] = DotF32(row, input1, cols);
                        output2[r] = DotF32(row, input2, cols);
                        output3[r] = DotF32(row, input3, cols);
                    }
                }
                break;
            }
            default:
                // Fallback: two MatVec2In pairs (each falls back per-dtype as needed).
                // Never worse than the prior pairwise path, still correct. Q8_0 lands
                // here deliberately — the single-token dense FFN decode (MatVec /
                // MatVecDual) and the old MatVec2In path both take the dequant→DotF32
                // fallback for Q8_0, so routing the quad here keeps the verify path
                // bit-identical to single-token decode (specialising it would also
                // require moving the single-token path to DotQ8_0 and re-validating).
                MatVec2In(output0, output1, weights, input0, input1, rows, cols, dtype);
                MatVec2In(output2, output3, weights, input2, input3, rows, cols, dtype);
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
    //  Q5_K Fused two-input dot (issue #30) — mirror of DotQ4K_2In for
    //  Q5_K weights (gate/up in 27B-MTP-Q5_K_M).
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ5K_2In(byte* row, float* input1, float* input2, int cols,
                                  out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx512F.IsSupported)
        {
            DotQ5K_2In_Avx512(row, input1, input2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ5K(row, input1, cols);
        sum2 = DotQ5K(row, input2, cols);
    }

    private static void DotQ5K_2In_Avx512(byte* row, float* input1, float* input2,
                                          int cols, int numBlocks,
                                          out float sum1, out float sum2)
    {
        var accLo1 = Vector512<float>.Zero;
        var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero;
        var accHi2 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16  = Vector512.Create(16);
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

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);
                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit u1 → q5Lo
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);

                    // High nibble + high bit u2 → q5Hi
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
    }

    // ================================================================
    //  Q4_K Fused two-input dot (issue #30) — decode each block ONCE
    //  in registers and dot against TWO inputs in the same loop pass.
    //  Halves both the dequant work AND the weight L2/DRAM reads vs
    //  two sequential DotQ4K calls — the actual bandwidth win that
    //  MatVec2In can't get from naive double-dispatch alone.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ4K_2In(byte* row, float* input1, float* input2, int cols,
                                  out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ4K_2In_Avx512(row, input1, input2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        // Fallback: two scalar/AVX2 dots (cache-friendly but no fused-dequant win).
        sum1 = DotQ4K(row, input1, cols);
        sum2 = DotQ4K(row, input2, cols);
    }

    private static void DotQ4K_2In_Avx512(byte* row, float* input1, float* input2,
                                          int cols, int numBlocks,
                                          out float sum1, out float sum2)
    {
        // Four accumulators: lo/hi nibble × input1/input2.
        var accLo1 = Vector512<float>.Zero;
        var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero;
        var accHi2 = Vector512<float>.Zero;
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

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    // Single byte→int load shared by both inputs.
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble: dequant once.
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    // FMA against both inputs.
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo,
                                Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);

                    // Upper nibble: dequant once.
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi,
                                Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
    }

    // ================================================================
    //  Q4_K Fused four-input dot (issue #114) — register-tiled extension
    //  of DotQ4K_2In: decode each block ONCE and FMA against FOUR inputs
    //  in the same pass. Amortizes the nibble unpack decode/4 instead of
    //  decode/2. Each input's lo/hi accumulators follow the single-input
    //  order exactly, so the result is bit-identical to four DotQ4K calls.
    // ================================================================

    public static void DotQ4K_4In(byte* row,
        float* input0, float* input1, float* input2, float* input3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ4K_4In_Avx512(row, input0, input1, input2, input3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        // Fallback: four single dots (no fused-dequant win, still correct).
        sum0 = DotQ4K(row, input0, cols);
        sum1 = DotQ4K(row, input1, cols);
        sum2 = DotQ4K(row, input2, cols);
        sum3 = DotQ4K(row, input3, cols);
    }

    private static void DotQ4K_4In_Avx512(byte* row,
        float* input0, float* input1, float* input2, float* input3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        // Two accumulators (lo/hi nibble) per input.
        var accLo0 = Vector512<float>.Zero; var accHi0 = Vector512<float>.Zero;
        var accLo1 = Vector512<float>.Zero; var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero; var accHi2 = Vector512<float>.Zero;
        var accLo3 = Vector512<float>.Zero; var accHi3 = Vector512<float>.Zero;
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

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    // Single byte→int load shared by all inputs.
                    var bytes16 = Vector128.LoadUnsafe(ref *(qs + qIdx + l));
                    var ints = Avx512F.ConvertToVector512Int32(bytes16);

                    // Lower nibble: dequant once, FMA against all four inputs.
                    var lo = Avx512F.And(ints, mask0F);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(lo), negDm1);
                    accLo0 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input0 + bo + l)), accLo0);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);
                    accLo3 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input3 + bo + l)), accLo3);

                    // Upper nibble: dequant once, FMA against all four inputs.
                    var hi = Avx512F.And(Avx512F.ShiftRightLogical(ints, 4), mask0F);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(hi), negDm2);
                    accHi0 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input0 + bo + 32 + l)), accHi0);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                    accHi3 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input3 + bo + 32 + l)), accHi3);
                }

                qIdx += 32;
                scIdx += 2;
            }
            elemOff += 256;
        }

        sum0 = HSum512(Avx512F.Add(accLo0, accHi0));
        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
        sum3 = HSum512(Avx512F.Add(accLo3, accHi3));
    }

    // ================================================================
    //  Q5_K Fused four-input dot (issue #209) — register-tiled extension
    //  of DotQ5K_2In: decode each block ONCE and FMA against FOUR inputs.
    //  Each input's lo/hi accumulator chain matches the single-input order
    //  exactly, so the result is bit-identical to four DotQ5K calls.
    // ================================================================

    public static void DotQ5K_4In(byte* row,
        float* input0, float* input1, float* input2, float* input3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx512F.IsSupported)
        {
            DotQ5K_4In_Avx512(row, input0, input1, input2, input3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ5K(row, input0, cols);
        sum1 = DotQ5K(row, input1, cols);
        sum2 = DotQ5K(row, input2, cols);
        sum3 = DotQ5K(row, input3, cols);
    }

    private static void DotQ5K_4In_Avx512(byte* row,
        float* input0, float* input1, float* input2, float* input3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        var accLo0 = Vector512<float>.Zero; var accHi0 = Vector512<float>.Zero;
        var accLo1 = Vector512<float>.Zero; var accHi1 = Vector512<float>.Zero;
        var accLo2 = Vector512<float>.Zero; var accHi2 = Vector512<float>.Zero;
        var accLo3 = Vector512<float>.Zero; var accHi3 = Vector512<float>.Zero;
        var mask0F = Vector512.Create(0x0F);
        var bit16  = Vector512.Create(16);
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

                var d1     = Vector512.Create(d * sc1);
                var negDm1 = Vector512.Create(-(dmin * m1));
                var d2     = Vector512.Create(d * sc2);
                var negDm2 = Vector512.Create(-(dmin * m2));

                int bo = elemOff + chunk * 64;

                for (int l = 0; l < 32; l += 16)
                {
                    var qlBytes = Vector128.LoadUnsafe(ref *(ql + qIdx + l));
                    var qlInts = Avx512F.ConvertToVector512Int32(qlBytes);
                    var qhBytes = Vector128.LoadUnsafe(ref *(qh + l));
                    var qhInts = Avx512F.ConvertToVector512Int32(qhBytes);

                    // Low nibble + high bit u1 → q5Lo, dequant once.
                    var loNib = Avx512F.And(qlInts, mask0F);
                    var hLoMask = Avx512F.And(qhInts, Vector512.Create((int)u1));
                    var hLo = Avx512F.And(
                        Avx512F.CompareGreaterThan(hLoMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Lo = Avx512F.Add(loNib, hLo);
                    var deqLo = Avx512F.FusedMultiplyAdd(d1, Avx512F.ConvertToVector512Single(q5Lo), negDm1);
                    accLo0 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input0 + bo + l)), accLo0);
                    accLo1 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input1 + bo + l)), accLo1);
                    accLo2 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input2 + bo + l)), accLo2);
                    accLo3 = Avx512F.FusedMultiplyAdd(deqLo, Vector512.LoadUnsafe(ref *(input3 + bo + l)), accLo3);

                    // High nibble + high bit u2 → q5Hi, dequant once.
                    var hiNib = Avx512F.And(Avx512F.ShiftRightLogical(qlInts, 4), mask0F);
                    var hHiMask = Avx512F.And(qhInts, Vector512.Create((int)u2));
                    var hHi = Avx512F.And(
                        Avx512F.CompareGreaterThan(hHiMask, Vector512<int>.Zero).AsInt32(),
                        bit16);
                    var q5Hi = Avx512F.Add(hiNib, hHi);
                    var deqHi = Avx512F.FusedMultiplyAdd(d2, Avx512F.ConvertToVector512Single(q5Hi), negDm2);
                    accHi0 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input0 + bo + 32 + l)), accHi0);
                    accHi1 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input1 + bo + 32 + l)), accHi1);
                    accHi2 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input2 + bo + 32 + l)), accHi2);
                    accHi3 = Avx512F.FusedMultiplyAdd(deqHi, Vector512.LoadUnsafe(ref *(input3 + bo + 32 + l)), accHi3);
                }

                qIdx += 32;
                scIdx += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
            elemOff += 256;
        }

        sum0 = HSum512(Avx512F.Add(accLo0, accHi0));
        sum1 = HSum512(Avx512F.Add(accLo1, accHi1));
        sum2 = HSum512(Avx512F.Add(accLo2, accHi2));
        sum3 = HSum512(Avx512F.Add(accLo3, accHi3));
    }

    // ================================================================
    //  Q6_K Fused MatVec
    // ================================================================

    public static void MatVecQ6K(float* output, byte* weights, float* input, int rows, int cols)
    {
        int bytesPerRow = (cols / 256) * 210;
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);

        if (rows >= MinRowsForParallel)
        {
            var w = weights; var s = scratch; var outp = output; int c = cols;
            Parallel.For(0, rows, s_parallelOpts, i =>
            {
                outp[i] = DotQ6K_Q8K(w + (long)i * bytesPerRow, s, c);
            });
        }
        else
        {
            for (int i = 0; i < rows; i++)
                output[i] = DotQ6K_Q8K(weights + (long)i * bytesPerRow, scratch, cols);
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
    //  Q8_0 Fused MatVec — 32-element blocks, [d:FP16 | qs:32×int8].
    //  AVX2 path expands 32 int8 → 4× 8 f32 per block and FMAs against
    //  the f32 input. APEX-mixed quants (e.g. Carnice MoE) interleave
    //  Q8_0 with K-quants, so this lives next to DotQ4K/DotQ5K/DotQ6K
    //  for use by the routed-expert DispatchDot path.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotQ8_0(byte* row, float* input, int cols)
    {
        const int QK = 32;
        const int bytesPerBlock = 34;
        int numBlocks = cols / QK;

        if (!Fma.IsSupported)
            return DotQ8_0_Scalar(row, input, cols, numBlocks);

        var acc = Vector256<float>.Zero;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            var dvec = Vector256.Create(d);
            sbyte* qs = (sbyte*)(block + 2);
            float* inp = input + b * QK;

            // Two halves of 16 sbytes each. Each half: low 8 → 8 i32 → 8 f32,
            // high 8 → 8 i32 → 8 f32. FMA scaled-qs × input into the accumulator.
            for (int half = 0; half < 2; half++)
            {
                var qs16 = Sse2.LoadVector128((byte*)(qs + half * 16)).AsSByte();
                var lo32 = Avx2.ConvertToVector256Int32(qs16);
                var hi32 = Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(qs16.AsByte(), 8).AsSByte());
                var loF  = Avx.ConvertToVector256Single(lo32);
                var hiF  = Avx.ConvertToVector256Single(hi32);
                var inpLo = Avx.LoadVector256(inp + half * 16);
                var inpHi = Avx.LoadVector256(inp + half * 16 + 8);
                acc = Fma.MultiplyAdd(Avx.Multiply(loF, dvec), inpLo, acc);
                acc = Fma.MultiplyAdd(Avx.Multiply(hiF, dvec), inpHi, acc);
            }
        }
        return HSum256(acc);
    }

    private static float DotQ8_0_Scalar(byte* row, float* input, int cols, int numBlocks)
    {
        const int QK = 32;
        const int bytesPerBlock = 34;
        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = HalfToFloat(block[0], block[1]);
            sbyte* qs = (sbyte*)(block + 2);
            float* inp = input + b * QK;
            float blockSum = 0;
            for (int i = 0; i < QK; i++)
                blockSum += qs[i] * inp[i];
            acc += d * blockSum;
        }
        return (float)acc;
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
    //  Q8_K Input Quantization (used by Q6_K dot for parity with ggml)
    // ================================================================
    // Scratch layout, one entry per super-block of 256 input floats (nb = cols/256):
    //   [0 .. nb*4):                          float d[nb]
    //   [nb*4 .. nb*4 + nb*256):              sbyte qs[nb*256]
    //   [nb*4 + nb*256 .. nb*4 + nb*256 + nb*32):  short bsums[nb*16]
    //
    // Mirrors ggml's block_q8_K but laid out as SoA so each array is contiguous
    // across super-blocks (cheaper to load in the dot kernel).

    public static int Q8KScratchBytes(int cols)
    {
        int nb = cols / 256;
        return nb * 4 + nb * 256 + nb * 32;
    }

    /// <summary>
    /// Quantize a row of float input to Q8_K format, mirroring ggml's
    /// quantize_row_q8_K_ref. Scale is iscale = -127/max where max is the
    /// signed element with largest |·|. Single FP rounding per element.
    /// </summary>
    public static void QuantizeRowToQ8K(float* input, int cols, byte* scratch)
    {
        int nb = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + nb * 4);
        short* bsumsArr = (short*)(scratch + nb * 4 + nb * 256);

        for (int b = 0; b < nb; b++)
        {
            float* x = input + b * 256;
            sbyte* qs = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            float max = 0f, amax = 0f;
            for (int j = 0; j < 256; j++)
            {
                float ax = MathF.Abs(x[j]);
                if (ax > amax) { amax = ax; max = x[j]; }
            }

            if (amax == 0f)
            {
                dArr[b] = 0f;
                for (int j = 0; j < 256; j++) qs[j] = 0;
                for (int j = 0; j < 16; j++) bsums[j] = 0;
                continue;
            }

            float iscale = -127.0f / max;
            for (int j = 0; j < 256; j++)
            {
                int v = (int)MathF.Round(iscale * x[j], MidpointRounding.ToEven);
                if (v > 127) v = 127;
                qs[j] = (sbyte)v;
            }
            for (int j = 0; j < 16; j++)
            {
                int sum = 0;
                for (int ii = 0; ii < 16; ii++) sum += qs[j * 16 + ii];
                bsums[j] = (short)sum;
            }
            dArr[b] = 1.0f / iscale;
        }
    }

    // ================================================================
    //  Q6_K · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_q6_K_q8_K. The crucial difference vs the legacy
    // dequant-FMA path is that the input is quantized to int8 once per super-
    // block (one FP rounding per input element), then the inner dot is done
    // entirely in int domain (u8·i8 → i16 via maddubs, ×i8 scale → i32 via
    // madd), with a single FP multiply by d_super = d_w * d_y at the end.
    // This collapses 256 per-element FP rounding steps to ~1 per super-block,
    // which matches what llama.cpp produces and removes the Q6_K direction
    // drift that caused the Qwen3.6-27B-MTP pos-12 argmax flip.

    public static float DotQ6K_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ6K_Q8K_Avx2(row, scratch, cols, numBlocks);

        return DotQ6K_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ6K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ6K_Q8K(row, scratch, cols);
    }

    private static float DotQ6K_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        float acc = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dy = dArr[b];
            float dSuper = dw * dy;

            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            // Integer accumulator over the whole super-block
            int sumi = 0;
            // -32 offset correction: sum(32 * sc[i] * bsums[i]) over 16 sub-groups
            int offsetCorr = 0;
            for (int g = 0; g < 16; g++)
                offsetCorr += (int)sc[g] * bsums[g];
            offsetCorr <<= 5;  // × 32

            int qlOff = 0, qhOff = 0, scBase = 0, qOff = 0;
            for (int half = 0; half < 2; half++)
            {
                for (int l = 0; l < 32; l++)
                {
                    int isc = l / 16;
                    // Unsigned 6-bit reconstruction (no -32 offset; subtracted via offsetCorr)
                    int q1u = (ql[qlOff + l] & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4);
                    int q2u = (ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4);
                    int q3u = (ql[qlOff + l] >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4);
                    int q4u = (ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4);

                    sumi += (int)sc[scBase + isc] * q1u * q8[qOff + l];
                    sumi += (int)sc[scBase + isc + 2] * q2u * q8[qOff + 32 + l];
                    sumi += (int)sc[scBase + isc + 4] * q3u * q8[qOff + 64 + l];
                    sumi += (int)sc[scBase + isc + 6] * q4u * q8[qOff + 96 + l];
                }
                qOff += 128;
                qlOff += 64;
                qhOff += 32;
                scBase += 8;
            }

            acc += dSuper * (sumi - offsetCorr);
        }
        return acc;
    }

    private static float DotQ6K_Q8K_Avx2(byte* row, byte* scratch, int cols, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc = Vector256<float>.Zero;

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper = dw * dArr[i];

            // q8sclsub = (bsums · scales_int16) << 5  →  8 int32
            // bsums: 16 int16. scales: 16 int8 → cvtepi8_epi16 → 16 int16.
            var q8sums = Vector256.LoadUnsafe(ref *(bsumsArr + i * 16));
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            var scales16 = Avx2.ConvertToVector256Int16(scales128);
            var q8sclsub = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);

            var sumi = Vector256<int>.Zero;
            sbyte* q8 = (sbyte*)(qsArr + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values
                var q4h_0 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(
                    Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15),
                    q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15),
                    q4h_3);

                var q8_0 = Vector256.LoadUnsafe(ref *(q8 + j * 128)).AsSByte();
                var q8_1 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 32)).AsSByte();
                var q8_2 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 64)).AsSByte();
                var q8_3 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 96)).AsSByte();

                // u8 × i8 → i16 pairs (no saturation: |u6×i8| ≤ 63×127 = 8001, pairs ≤ 16002)
                var p16_0 = Avx2.MultiplyAddAdjacent(q4_0, q8_0);
                var p16_1 = Avx2.MultiplyAddAdjacent(q4_1, q8_1);
                var p16_2 = Avx2.MultiplyAddAdjacent(q4_2, q8_2);
                var p16_3 = Avx2.MultiplyAddAdjacent(q4_3, q8_3);

                // Apply 8 per-16-element scales (2 per q4_k). Each scale broadcast
                // to 8 int16 lanes; madd pairs adjacent lanes → 4 int32 outputs
                // per q4_k all sharing the same scale within a 16-elem sub-group.
                int isc = j * 8;
                var sc16_0 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 0]),
                    Vector128.Create((short)sc[isc + 1]));
                var sc16_1 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 2]),
                    Vector128.Create((short)sc[isc + 3]));
                var sc16_2 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 4]),
                    Vector128.Create((short)sc[isc + 5]));
                var sc16_3 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 6]),
                    Vector128.Create((short)sc[isc + 7]));

                var s0 = Avx2.MultiplyAddAdjacent(sc16_0, p16_0);
                var s1 = Avx2.MultiplyAddAdjacent(sc16_1, p16_1);
                var s2 = Avx2.MultiplyAddAdjacent(sc16_2, p16_2);
                var s3 = Avx2.MultiplyAddAdjacent(sc16_3, p16_3);

                sumi = Avx2.Add(sumi, Avx2.Add(Avx2.Add(s0, s1), Avx2.Add(s2, s3)));
            }

            acc = Fma.MultiplyAdd(
                Vector256.Create(dSuper),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi, q8sclsub)),
                acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q3_K · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Mirrors ggml_vec_dot_q3_K_q8_K. Same int-domain strategy as
    // DotQ6K_Q8K: weights are reconstructed as unsigned 3-bit values
    // (qu ∈ [0,7]) and the signed offset is amortised across the super-
    // block via the Q8_K bsums. The per-sub-group dl factor in scalar
    // Q3_K is `dAll * (scales[is] - 32)`; here we bake the -32 into the
    // i8 scale so the inner dot stays in int domain.
    //
    // Decomposition:
    //   q3 = qu - 4    (qu = ((qs>>shift)&3) + 4*hmask_bit, in [0,7])
    //   dl * q3 * y    = dl*qu*y  -  4*dl*y
    //   Sum over a 16-element sub-group:
    //     dl * (Σ qu·y)  -  4 * dl * bsums_is
    //   With dl = dAll*(scale-32):
    //     dAll * [(scale-32)*Σ(qu·y)  -  4*(scale-32)*bsums_is]
    //   The second term, summed over all 16 sub-blocks, is the
    //   `q8sclsub` correction = ((bsums·scales_adj) << 2).
    //
    // The auto-on parity gap that #103 surfaced was NOT in this dot —
    // per-kernel int-domain reference matches ggml at 1e-4 rel. The gap
    // is in the Q8_K input quantization itself (per-256-element single
    // scale). The Q8_KS-input variant (Q3K_Q8KS) ships alongside and
    // is what the gated MoE path actually dispatches when both gates
    // resolve to on; see DotQ3K_Q8KS in this file for the per-32 scale
    // path and HybridGdnForwardPass for the routing.

    public static float DotQ3K_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ3K_Q8K_Avx2(row, scratch, numBlocks);

        return DotQ3K_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ3K_Q8K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ3K_Q8K(row, scratch, cols);
    }

    internal static float DotQ3K_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        float acc = 0f;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float dy = dArr[b];
            float dSuper = dAll * dy;

            // Unpack 16 6-bit scales via the ggml aux[] pattern.
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32;
            byte* hm = x;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            // -32 offset correction: 4 * Σ (scale-32) * bsums_is, scaled by dSuper.
            int offsetCorr = 0;
            for (int g = 0; g < 16; g++)
                offsetCorr += ((int)scales[g] - 32) * bsums[g];
            offsetCorr <<= 2; // × 4

            int sumi = 0;
            int qOff = 0;
            int isIdx = 0;
            int qOut = 0;
            byte m = 1;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int sc0 = (int)scales[isIdx++] - 32;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + l] >> shift) & 3) + ((hm[l] & m) != 0 ? 4 : 0);
                        sumi += sc0 * qu * q8[qOut + l];
                    }
                    int sc1 = (int)scales[isIdx++] - 32;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + 16 + l] >> shift) & 3) + ((hm[16 + l] & m) != 0 ? 4 : 0);
                        sumi += sc1 * qu * q8[qOut + 16 + l];
                    }
                    qOut += 32;
                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }

            acc += dSuper * (sumi - offsetCorr);
        }
        return acc;
    }

    private static float DotQ3K_Q8K_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        short* bsumsArr = (short*)(scratch + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float dSuper = dAll * dArr[b];

            // Unpack 16 6-bit scales via the ggml aux[] pattern.
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            // q8sclsub = (bsums · scales_adj) << 2  →  8 int32
            // scales_adj ∈ [-32, +31] fits in i8; bsums sub-group sums fit in i16.
            var q8sums = Vector256.LoadUnsafe(ref *(bsumsArr + b * 16));
            var scales128 = Vector128.LoadUnsafe(ref scales[0]);
            var scales16 = Avx2.ConvertToVector256Int16(scales128);
            var q8sclsub = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums, scales16), 2);

            // hmask is shared across both halves; bit-plane indexed by (half*4 + j).
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));

            var sumi = Vector256<int>.Zero;
            sbyte* q8 = qsArr + b * 256;
            byte* qs = x + 32;

            // Two halves × four j-iterations. The qs/hm shift amounts are
            // selected via switch on j (and (half,j)) so each AVX2 shift sees
            // a compile-time-constant immediate (CA1857). Each j contributes
            // 32 unsigned 3-bit weights spanning two 16-element sub-groups.
            for (int half = 0; half < 2; half++)
            {
                // 32 packed qs bytes for this half (4 weights per byte via shifts 0,2,4,6)
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    // qlo = (qs_v >> shift) & 0x03  (per-byte low-2-bits extraction)
                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    // hbit = ((hm_v >> hbitPos) & 1) << 2   → 0 or 4 per byte
                    // hbitPos = half*4 + j  ∈ [0..7]
                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit); // u3 in [0,7] per byte

                    // q3u carries two 16-element sub-groups: lanes [0..15] and [16..31]
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + half * 128 + j * 32)).AsSByte();

                    // u3·i8 → i16 pairs (no saturation: |u3·i8| ≤ 7·127 = 889, pairs ≤ 1778)
                    var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);

                    // Two scale lanes — one per 16-element sub-group within q3u
                    int isc = half * 8 + 2 * j;
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scales[isc + 0]),
                        Vector128.Create((short)scales[isc + 1]));

                    var s = Avx2.MultiplyAddAdjacent(sc16, p16);
                    sumi = Avx2.Add(sumi, s);
                }
            }

            acc = Fma.MultiplyAdd(
                Vector256.Create(dSuper),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi, q8sclsub)),
                acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q8_0 · Q8_K Dot Product  (one row, pre-quantized input)
    // ================================================================
    // Q8_0 is a 32-element / 34-byte block: [d:FP16 | qs:32×int8]. The
    // legacy DotQ8_0(float* input) path dequant-expands each block to 32
    // FP32 lanes and FMAs against the f32 input — that's 32 FP multiplies
    // and 4 widen-and-convert sequences per block (256 FP rounding events
    // per 256-element super-block).
    //
    // The Q8_K-input fusion keeps the inner dot entirely in int domain:
    //   - 32 i8·i8 products per sub-block via two VPMADDWD chains
    //     (16 i16 + 16 i16 → 8 i32 + 8 i32 → lane-add to 8 i32 partials)
    //   - one FP multiply per Q8_0 sub-block (d_w[sub] × 8-lane int partials)
    //   - one FP multiply per Q8_K super-block (d_y[b] × Σ_sub)
    // Eight Q8_0 weight blocks span one Q8_K super-block (8 × 32 = 256
    // elements), so we collapse 256 FP roundings to 9 (8 inner + 1 outer)
    // per super-block — same direction-of-improvement as DotQ6K_Q8K and
    // DotQ3K_Q8K. cols must be a multiple of 256 (every model dim in
    // the codebase already satisfies this).
    //
    // The Q8_K bsums region is intentionally unused: Q8_0 is signed-
    // symmetric with no -32 offset to amortise, so no `q8sclsub`
    // correction is needed (cf. DotQ6K/DotQ3K which both subtract a
    // bsums-based correction). The bsums bytes are dead weight in this
    // path — see briefing notes for rank-2 design.
    //
    // Dual-acc-chain VPMADDWD pattern: each Q8_0 sub-block reduces its
    // 32 i8·i8 products via two independent MultiplyAddAdjacent chains
    // (low 16 + high 16 of the sub-block), matching the throughput
    // template of DotQ6K_Q8K_Avx2 / DotQ3K_Q8K_Avx2.

    public static float DotQ8_0_Q8K(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ8_0_Q8K_Avx2(row, scratch, numBlocks);

        return DotQ8_0_Q8K_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ8_0_Q8K(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8K(input, cols, scratch);
        return DotQ8_0_Q8K(row, scratch, cols);
    }

    internal static float DotQ8_0_Q8K_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        // bsums region after qsArr is unused for Q8_0 — see header.

        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float dy = dArr[b];
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            float subAcc = 0f;
            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                int intDot = 0;
                for (int i = 0; i < 32; i++)
                    intDot += qw[i] * qy[i];

                subAcc += dw * intDot;
            }

            acc += dy * subAcc;
        }
        return (float)acc;
    }

    private static float DotQ8_0_Q8K_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);

        var acc = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            float dy = dArr[b];
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            // 8 inner i32 dots → scale by d_w[sub] into subAccF; one outer
            // FMA by d_y[b] folds in the Q8_K super-block scale (see header).
            var subAccF = Vector256<float>.Zero;
            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                // 32 i8 → two halves of 16 i8 → widen to i16. AVX2 path:
                // SSE2 load 128b → ConvertToVector256Int16 sign-extends.
                var qw_lo128 = Sse2.LoadVector128((byte*)qw).AsSByte();           // lanes 0..15
                var qw_hi128 = Sse2.LoadVector128((byte*)(qw + 16)).AsSByte();    // lanes 16..31
                var qy_lo128 = Sse2.LoadVector128((byte*)qy).AsSByte();
                var qy_hi128 = Sse2.LoadVector128((byte*)(qy + 16)).AsSByte();

                var qw_lo = Avx2.ConvertToVector256Int16(qw_lo128);
                var qw_hi = Avx2.ConvertToVector256Int16(qw_hi128);
                var qy_lo = Avx2.ConvertToVector256Int16(qy_lo128);
                var qy_hi = Avx2.ConvertToVector256Int16(qy_hi128);

                // Two independent VPMADDWD chains: i16·i16 → i32 pair-sum.
                // |i16·i16| ≤ 127·127 = 16129; pair ≤ 32258, no saturation.
                var p_lo = Avx2.MultiplyAddAdjacent(qw_lo, qy_lo); // 8 i32
                var p_hi = Avx2.MultiplyAddAdjacent(qw_hi, qy_hi); // 8 i32
                var p_sum = Avx2.Add(p_lo, p_hi);                  // 8 i32 partials

                // Scale this sub-block's 8 i32 partials by d_w[sub] and
                // accumulate into the super-block FP accumulator.
                var pF = Avx.ConvertToVector256Single(p_sum);
                subAccF = Fma.MultiplyAdd(Vector256.Create(dw), pF, subAccF);
            }

            // Fold in the Q8_K super-block input scale d_y[b].
            acc = Fma.MultiplyAdd(Vector256.Create(dy), subAccF, acc);
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q8_KS — Q8_K with per-32-element sub-block scales (issue #107)
    // ================================================================
    // Same int8 qs as Q8_K, but eight FP scales per 256-element super-
    // block (one per 32-element sub-block) instead of a single per-256
    // scale. Each sub-block's iscale is computed from its own amax, so
    // sub-blocks of low dynamic range get higher resolution (qs fills
    // more of [-127, +127]).
    //
    // Motivation (validation log docs/q8k-validation-*.md): the Q8_K
    // per-256 scale loses precision on inputs with non-uniform magnitude
    // across the super-block (post-SiLU activations, attention outputs).
    // Per-kernel parity vs ggml matches at 1e-4 rel; the trunk drift
    // that flips occasional argmaxes lives entirely in the input
    // quantization step. Per-32 scales cut the quantization-noise
    // envelope ~4× on Carnice (validation envelope was ±13 pp with
    // plain Q8_K, drops to ±3 pp with Q8_KS). A per-16 variant matching
    // Q3_K's scale-lane granularity was tried; it shuffles FP rounding
    // noise to different prompts (mathreason and factual lose what
    // techexplain gains) and was not strictly better — see
    // bench-q8k-validation-per16.csv for the comparison. Per-32 is the
    // local optimum until a finer-grained approach (e.g. Q8_1 with
    // per-block min offset) is investigated.
    //
    // Scratch layout, one entry per 256-input-float super-block (nb = cols/256):
    //   [0 .. nb*32):                                float d[nb*8]    (per-32 scales)
    //   [nb*32 .. nb*32 + nb*256):                    sbyte qs[nb*256]
    //   [nb*32 + nb*256 .. nb*32 + nb*256 + nb*32):   short bsums[nb*16]  (per-16 sums, for Q3K -32 offset)
    //
    // Total: nb * 320 bytes (vs nb * 292 for Q8_K). The extra 28 B/sb
    // is comfortably under the routed-MoE expert-scratch budget.

    public static int Q8KSScratchBytes(int cols)
    {
        int nb = cols / 256;
        return nb * 32 + nb * 256 + nb * 32;
    }

    /// <summary>
    /// Quantize a row of float input to Q8_KS format (per-32-element
    /// sub-block scales). Each 32-element sub-block computes its own
    /// iscale = -127 / max_signed_amax_sub, single FP rounding per element.
    /// bsums[g] keeps the unscaled int-sum over each 16-element sub-group
    /// so the Q3_K -32 offset correction can fold across two adjacent
    /// sub-groups per sub-block.
    /// </summary>
    public static void QuantizeRowToQ8KS(float* input, int cols, byte* scratch)
    {
        int nb = cols / 256;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + nb * 32);
        short* bsumsArr = (short*)(scratch + nb * 32 + nb * 256);

        for (int b = 0; b < nb; b++)
        {
            float* x = input + b * 256;
            sbyte* qs = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;
            float* d = dArr + b * 8;

            for (int sub = 0; sub < 8; sub++)
            {
                float max = 0f, amax = 0f;
                for (int j = 0; j < 32; j++)
                {
                    float ax = MathF.Abs(x[sub * 32 + j]);
                    if (ax > amax) { amax = ax; max = x[sub * 32 + j]; }
                }

                if (amax == 0f)
                {
                    d[sub] = 0f;
                    for (int j = 0; j < 32; j++) qs[sub * 32 + j] = 0;
                }
                else
                {
                    float iscale = -127.0f / max;
                    for (int j = 0; j < 32; j++)
                    {
                        int v = (int)MathF.Round(iscale * x[sub * 32 + j], MidpointRounding.ToEven);
                        if (v > 127) v = 127;
                        qs[sub * 32 + j] = (sbyte)v;
                    }
                    d[sub] = 1.0f / iscale;
                }
            }

            for (int g = 0; g < 16; g++)
            {
                int sum = 0;
                for (int ii = 0; ii < 16; ii++) sum += qs[g * 16 + ii];
                bsums[g] = (short)sum;
            }
        }
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Same int-domain strategy as Q3K_Q8K but each 32-element sub-block
    // (= 2 Q3_K 16-element sub-groups) has its own FP scale d_y[sub] in
    // place of the single per-super-block d_y. Per-sub-block FMA pattern
    // is identical to DotQ8_0_Q8K (which already accumulates per-sub),
    // so the extra cost is 7 extra FP FMAs per super-block — invisible
    // against the inner u3·i8/i16·i16 work.

    public static float DotQ3K_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ3K_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ3K_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ3K_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ3K_Q8KS(row, scratch, cols);
    }

    internal static float DotQ3K_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        float acc = 0f;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub = dArr + b * 8;

            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)(byte)(aux[i] >> 0);
                scales[i * 4 + 1] = (sbyte)(byte)(aux[i] >> 8);
                scales[i * 4 + 2] = (sbyte)(byte)(aux[i] >> 16);
                scales[i * 4 + 3] = (sbyte)(byte)(aux[i] >> 24);
            }

            byte* qs = x + 32;
            byte* hm = x;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            int qOff = 0;
            int isIdx = 0;
            int qOut = 0;
            byte m = 1;
            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;
                    int sc0 = (int)scales[isIdx++] - 32;
                    int sub0 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + l] >> shift) & 3) + ((hm[l] & m) != 0 ? 4 : 0);
                        sub0 += qu * q8[qOut + l];
                    }
                    int sc1 = (int)scales[isIdx++] - 32;
                    int sub1 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int qu = ((qs[qOff + 16 + l] >> shift) & 3) + ((hm[16 + l] & m) != 0 ? 4 : 0);
                        sub1 += qu * q8[qOut + 16 + l];
                    }

                    int subInt = sc0 * sub0 + sc1 * sub1
                               - 4 * (sc0 * bsums[isIdx - 2] + sc1 * bsums[isIdx - 1]);
                    acc += (dAll * dSub[sub]) * subInt;

                    qOut += 32;
                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }
        }
        return acc;
    }

    private static float DotQ3K_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub = dArr + b * 8;

            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums = bsumsArr + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8 = qsArr + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);

                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + half * 128 + j * 32)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);

                    // Per-sub-block offset correction folded into lane 0:
                    //   sub_corr = 4 * (scA * bsums[isc] + scB * bsums[isc+1])
                    int subCorr = ((int)scA * bsums[isc] + (int)scB * bsums[isc + 1]) << 2;
                    sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);

                    var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                    float scaleSub = dAll * dSub[sub];
                    acc = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc);
                }
            }
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product — two-input dequant-once (issue #112)
    // ================================================================
    // Decodes the Q3_K weight row ONCE (the 3-bit unpack + 6-bit scale decode is
    // the expensive part) and dots it against two Q8_KS-prepacked inputs. Each
    // input's accumulation is byte-for-byte identical to <see cref="DotQ3K_Q8KS"/>
    // — same sub-block order, same int MAdd / offset-correction / FP FMA chain —
    // so it is bit-identical to two single dots. Used by the batched routed-MoE
    // path to amortize the unpack across token pairs routing to the same expert.
    public static void DotQ3K_Q8KS_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ3K_Q8KS_2In_Avx2(row, scratch1, scratch2, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ3K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ3K_Q8KS_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ3K_Q8KS_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int numBlocks, out float sum1, out float sum2)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub1 = dArr1 + b * 8;
            float* dSub2 = dArr2 + b * 8;

            // Scales decode (shared between both inputs).
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums1 = bsumsArr1 + b * 16;
            short* bsums2 = bsumsArr2 + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8a = qsArr1 + b * 256;
            sbyte* q8b = qsArr2 + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);   // shared weight quants

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    // Input 1 — same accumulation as the single-input kernel.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8a + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums1[isc] + (int)scB * bsums1[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub1[sub];
                        acc1 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc1);
                    }
                    // Input 2 — reuses decoded q3u / sc16.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8b + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums2[isc] + (int)scB * bsums2[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub2[sub];
                        acc2 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc2);
                    }
                }
            }
        }
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
    }

    // ================================================================
    //  Q3_K · Q8_KS Dot Product — four-input dequant-once (issue #114)
    // ================================================================
    // Generalizes <see cref="DotQ3K_Q8KS_2In"/> to a register-tiled tile of FOUR
    // Q8_KS-prepacked inputs: the 3-bit unpack + 6-bit scale decode is done ONCE
    // per sub-block and reused across all four inputs, so the (dominant) weight
    // decode is amortized decode/4 instead of decode/2. Each input's accumulation
    // is byte-for-byte identical to <see cref="DotQ3K_Q8KS"/> — same sub-block
    // order, same int MAdd / offset-correction / FP FMA chain — so the result is
    // bit-identical to four single dots. Used by the batched routed-MoE path to
    // amortize the unpack across token quads routing to the same expert.
    public static void DotQ3K_Q8KS_4In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ3K_Q8KS_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ3K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ3K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ3K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ3K_Q8KS_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ3K_Q8KS_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m1 = Vector256.Create((byte)0x01);
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            float dAll = HalfToFloat(x[108], x[109]);
            float* dSub0 = dArr0 + b * 8;
            float* dSub1 = dArr1 + b * 8;
            float* dSub2 = dArr2 + b * 8;
            float* dSub3 = dArr3 + b * 8;

            // Scales decode (shared between all four inputs).
            aux[0] = *(uint*)(x + 96);
            aux[1] = *(uint*)(x + 100);
            uint tmp = *(uint*)(x + 104);
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                scales[i * 4 + 0] = (sbyte)((byte)(aux[i] >> 0) - 32);
                scales[i * 4 + 1] = (sbyte)((byte)(aux[i] >> 8) - 32);
                scales[i * 4 + 2] = (sbyte)((byte)(aux[i] >> 16) - 32);
                scales[i * 4 + 3] = (sbyte)((byte)(aux[i] >> 24) - 32);
            }

            short* bsums0 = bsumsArr0 + b * 16;
            short* bsums1 = bsumsArr1 + b * 16;
            short* bsums2 = bsumsArr2 + b * 16;
            short* bsums3 = bsumsArr3 + b * 16;
            var hm_v = Vector256.LoadUnsafe(ref *(x + 0));
            sbyte* q8_0 = qsArr0 + b * 256;
            sbyte* q8_1 = qsArr1 + b * 256;
            sbyte* q8_2 = qsArr2 + b * 256;
            sbyte* q8_3 = qsArr3 + b * 256;
            byte* qs = x + 32;

            for (int half = 0; half < 2; half++)
            {
                var qs_v = Vector256.LoadUnsafe(ref *(qs + half * 32));

                for (int j = 0; j < 4; j++)
                {
                    int sub = half * 4 + j;

                    var qloShifted = j switch
                    {
                        0 => qs_v,
                        1 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 2).AsByte(),
                        2 => Avx2.ShiftRightLogical(qs_v.AsInt16(), 4).AsByte(),
                        _ => Avx2.ShiftRightLogical(qs_v.AsInt16(), 6).AsByte(),
                    };
                    var qlo = Avx2.And(qloShifted, m3);

                    var hmShifted = (half, j) switch
                    {
                        (0, 0) => hm_v,
                        (0, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 1).AsByte(),
                        (0, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 2).AsByte(),
                        (0, 3) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 3).AsByte(),
                        (1, 0) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 4).AsByte(),
                        (1, 1) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 5).AsByte(),
                        (1, 2) => Avx2.ShiftRightLogical(hm_v.AsInt16(), 6).AsByte(),
                        _      => Avx2.ShiftRightLogical(hm_v.AsInt16(), 7).AsByte(),
                    };
                    var hbit = Avx2.ShiftLeftLogical(
                        Avx2.And(hmShifted, m1).AsInt16(), 2).AsByte();
                    var q3u = Avx2.Or(qlo, hbit);   // shared weight quants

                    int isc = half * 8 + 2 * j;
                    sbyte scA = scales[isc + 0];
                    sbyte scB = scales[isc + 1];
                    var sc16 = Vector256.Create(
                        Vector128.Create((short)scA),
                        Vector128.Create((short)scB));

                    // Each input — same accumulation as the single-input kernel,
                    // reusing the decoded q3u / sc16.
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_0 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums0[isc] + (int)scB * bsums0[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub0[sub];
                        acc0 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc0);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_1 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums1[isc] + (int)scB * bsums1[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub1[sub];
                        acc1 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc1);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_2 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums2[isc] + (int)scB * bsums2[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub2[sub];
                        acc2 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc2);
                    }
                    {
                        var q8_v = Vector256.LoadUnsafe(ref *(q8_3 + half * 128 + j * 32)).AsSByte();
                        var p16 = Avx2.MultiplyAddAdjacent(q3u, q8_v);
                        var sub_i32 = Avx2.MultiplyAddAdjacent(sc16, p16);
                        int subCorr = ((int)scA * bsums3[isc] + (int)scB * bsums3[isc + 1]) << 2;
                        sub_i32 = sub_i32.WithElement(0, sub_i32.GetElement(0) - subCorr);
                        var sub_fp = Avx.ConvertToVector256Single(sub_i32);
                        float scaleSub = dAll * dSub3[sub];
                        acc3 = Fma.MultiplyAdd(Vector256.Create(scaleSub), sub_fp, acc3);
                    }
                }
            }
        }
        sum0 = HSum256(acc0);
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
        sum3 = HSum256(acc3);
    }

    // ================================================================
    //  Q8_0 · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Q8_0 block (32 elements / 34 bytes) naturally pairs 1:1 with one
    // Q8_KS sub-block. Per-sub-block dot is qw·qy summed in i32, scaled
    // by d_w[sub] × d_y[sub], accumulated across 8 sub-blocks per super-
    // block. The per-32 d_y dramatically reduces the FP-vs-quantized
    // envelope vs Q8_0_Q8K's per-256 d_y for activations with non-
    // uniform magnitude.

    public static float DotQ8_0_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ8_0_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ8_0_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ8_0_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ8_0_Q8KS(row, scratch, cols);
    }

    internal static float DotQ8_0_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);

        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                int intDot = 0;
                for (int i = 0; i < 32; i++)
                    intDot += qw[i] * qy[i];

                acc += (dw * dSub[sub]) * intDot;
            }
        }
        return (float)acc;
    }

    private static float DotQ8_0_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);

        var acc = Vector256<float>.Zero;

        for (int b = 0; b < numBlocks; b++)
        {
            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                float dw = HalfToFloat(block[0], block[1]);
                sbyte* qw = (sbyte*)(block + 2);
                sbyte* qy = q8 + sub * 32;

                var qw_lo128 = Sse2.LoadVector128((byte*)qw).AsSByte();
                var qw_hi128 = Sse2.LoadVector128((byte*)(qw + 16)).AsSByte();
                var qy_lo128 = Sse2.LoadVector128((byte*)qy).AsSByte();
                var qy_hi128 = Sse2.LoadVector128((byte*)(qy + 16)).AsSByte();

                var qw_lo = Avx2.ConvertToVector256Int16(qw_lo128);
                var qw_hi = Avx2.ConvertToVector256Int16(qw_hi128);
                var qy_lo = Avx2.ConvertToVector256Int16(qy_lo128);
                var qy_hi = Avx2.ConvertToVector256Int16(qy_hi128);

                var p_lo = Avx2.MultiplyAddAdjacent(qw_lo, qy_lo);
                var p_hi = Avx2.MultiplyAddAdjacent(qw_hi, qy_hi);
                var p_sum = Avx2.Add(p_lo, p_hi);

                var pF = Avx.ConvertToVector256Single(p_sum);
                float scaleSub = dw * dSub[sub];
                acc = Fma.MultiplyAdd(Vector256.Create(scaleSub), pF, acc);
            }
        }
        return HSum256(acc);
    }

    // ================================================================
    //  Q4_K · Q8_KS Dot Product  (one row, per-32 prequantized input)
    // ================================================================
    // Q4_K super-block = 256 elements / 144 bytes. Each 32-element
    // sub-block (8 per super-block) has its own 6-bit scale `sc` and
    // 6-bit min `m` (decoded by GetScaleMinK4) plus the per-super-block
    // FP `d` / `dmin`. The 4-bit weight nibble is UNSIGNED [0,15], so the
    // inner Σ nibble·q8 is the same u8·s8 (vpmaddubsw) pattern as the
    // Q3_K kernel — but the offset correction is `-dmin·m·Σq8` (a true
    // per-sub-block min, NOT Q3_K's constant -4). The per-32 activation
    // scale `dSub[sub]` folds into the per-sub-block FP FMA, so the dot
    // is dw-quantized in the int domain and FP-scaled per sub-block:
    //   acc += dSub·( d·sc·Σ(nibble·q8) − dmin·m·Σq8 ).
    // This replaces the slow f32-dequant fallback (DotQ4K) for routed
    // Q4_K MoE experts on the int8 path (Carnice: 9/41 expert layers).

    public static float DotQ4K_Q8KS(byte* row, byte* scratch, int cols)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
            return DotQ4K_Q8KS_Avx2(row, scratch, numBlocks);

        return DotQ4K_Q8KS_Scalar(row, scratch, numBlocks);
    }

    public static float DotQ4K_Q8KS(byte* row, float* input, int cols)
    {
        int scratchBytes = Q8KSScratchBytes(cols);
        byte* scratch = stackalloc byte[scratchBytes];
        QuantizeRowToQ8KS(input, cols, scratch);
        return DotQ4K_Q8KS(row, scratch, cols);
    }

    internal static float DotQ4K_Q8KS_Scalar(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        float acc = 0f;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out byte m1);     // s0 (low nibbles)
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out byte m2); // s1 (high nibbles)

                // Sub-block s0 (32 elems at chunk*64).
                int sub0 = 0;
                int eo0 = chunk * 64;
                for (int l = 0; l < 32; l++)
                    sub0 += (qs[chunk * 32 + l] & 0x0F) * (int)q8[eo0 + l];
                int s0 = 2 * chunk;
                int bsum0 = (int)bsums[2 * s0] + (int)bsums[2 * s0 + 1];
                acc += dSub[s0] * (d * sc1 * sub0 - dmin * m1 * bsum0);

                // Sub-block s1 (32 elems at chunk*64+32).
                int sub1 = 0;
                int eo1 = chunk * 64 + 32;
                for (int l = 0; l < 32; l++)
                    sub1 += (qs[chunk * 32 + l] >> 4) * (int)q8[eo1 + l];
                int s1 = 2 * chunk + 1;
                int bsum1 = (int)bsums[2 * s1] + (int)bsums[2 * s1 + 1];
                acc += dSub[s1] * (d * sc2 * sub1 - dmin * m2 * bsum1);
            }
        }
        return acc;
    }

    private static float DotQ4K_Q8KS_Avx2(byte* row, byte* scratch, int numBlocks)
    {
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 32);
        short* bsumsArr = (short*)(scratch + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        float acc = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub = dArr + b * 8;
            sbyte* q8 = qsArr + b * 256;
            short* bsums = bsumsArr + b * 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out byte m1);     // s0 (low nibbles)
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out byte m2); // s1 (high nibbles)

                // 32 packed nibble-bytes for this chunk → low/high nibble halves.
                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;

                // Sub-block s0 (low nibbles · q8 at chunk*64).
                {
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(lo, q8_v);                 // u8·s8 → i16 pairs
                    int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16, one16));
                    int bsum0 = (int)bsums[2 * s0] + (int)bsums[2 * s0 + 1];
                    acc += dSub[s0] * (d * sc1 * sub0 - dmin * m1 * bsum0);
                }
                // Sub-block s1 (high nibbles · q8 at chunk*64+32).
                {
                    var q8_v = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
                    var p16 = Avx2.MultiplyAddAdjacent(hi, q8_v);
                    int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(p16, one16));
                    int bsum1 = (int)bsums[2 * s1] + (int)bsums[2 * s1 + 1];
                    acc += dSub[s1] * (d * sc2 * sub1 - dmin * m2 * bsum1);
                }
            }
        }
        return acc;
    }

    // ================================================================
    //  Q4_K · Q8_KS Dot Product — two/four-input dequant-once (#112/#114)
    // ================================================================
    // Decodes the Q4_K weight row ONCE (the nibble unpack + 6-bit scale/min
    // decode) and dots it against 2/4 Q8_KS-prepacked inputs. Each input's
    // accumulation is byte-for-byte identical to <see cref="DotQ4K_Q8KS"/> —
    // same sub-block order, same int MAdd / min-correction / FP FMA chain —
    // so it is bit-identical to N single dots. Used by the batched routed-MoE
    // path to amortize the unpack across tokens routing to the same expert.
    public static void DotQ4K_Q8KS_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ4K_Q8KS_2In_Avx2(row, scratch1, scratch2, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ4K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ4K_Q8KS_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ4K_Q8KS_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int numBlocks, out float sum1, out float sum2)
    {
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        float acc1 = 0f;
        float acc2 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub1 = dArr1 + b * 8;
            sbyte* q8a = qsArr1 + b * 256;
            short* bsums1 = bsumsArr1 + b * 16;
            float* dSub2 = dArr2 + b * 8;
            sbyte* q8b = qsArr2 + b * 256;
            short* bsums2 = bsumsArr2 + b * 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out byte m1);
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out byte m2);

                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));   // shared weight nibbles
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;
                float dsc1 = d * sc1, dm1 = dmin * m1;
                float dsc2 = d * sc2, dm2 = dmin * m2;

                // Input 1.
                {
                    var qlo = Vector256.LoadUnsafe(ref *(q8a + chunk * 64)).AsSByte();
                    var qhi = Vector256.LoadUnsafe(ref *(q8a + chunk * 64 + 32)).AsSByte();
                    int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(lo, qlo), one16));
                    int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(hi, qhi), one16));
                    int bsum0 = (int)bsums1[2 * s0] + (int)bsums1[2 * s0 + 1];
                    int bsum1 = (int)bsums1[2 * s1] + (int)bsums1[2 * s1 + 1];
                    acc1 += dSub1[s0] * (dsc1 * sub0 - dm1 * bsum0);
                    acc1 += dSub1[s1] * (dsc2 * sub1 - dm2 * bsum1);
                }
                // Input 2 — reuses decoded lo/hi.
                {
                    var qlo = Vector256.LoadUnsafe(ref *(q8b + chunk * 64)).AsSByte();
                    var qhi = Vector256.LoadUnsafe(ref *(q8b + chunk * 64 + 32)).AsSByte();
                    int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(lo, qlo), one16));
                    int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(hi, qhi), one16));
                    int bsum0 = (int)bsums2[2 * s0] + (int)bsums2[2 * s0 + 1];
                    int bsum1 = (int)bsums2[2 * s1] + (int)bsums2[2 * s1 + 1];
                    acc2 += dSub2[s0] * (dsc1 * sub0 - dm1 * bsum0);
                    acc2 += dSub2[s1] * (dsc2 * sub1 - dm2 * bsum1);
                }
            }
        }
        sum1 = acc1;
        sum2 = acc2;
    }

    public static void DotQ4K_Q8KS_4In(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int cols,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ4K_Q8KS_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ4K_Q8KS_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ4K_Q8KS_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ4K_Q8KS_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ4K_Q8KS_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ4K_Q8KS_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 32);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 32 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 32);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 32 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 32);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 32 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 32);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 32 + numBlocks * 256);

        var m0F = Vector256.Create((byte)0x0F);
        var one16 = Vector256.Create((short)1);
        float acc0 = 0f;
        float acc1 = 0f;
        float acc2 = 0f;
        float acc3 = 0f;

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 144;
            float d = HalfToFloat(x[0], x[1]);
            float dmin = HalfToFloat(x[2], x[3]);
            byte* sc = x + 4;
            byte* qs = x + 16;

            float* dSub0 = dArr0 + b * 8;
            sbyte* q8_0 = qsArr0 + b * 256;
            short* bsums0 = bsumsArr0 + b * 16;
            float* dSub1 = dArr1 + b * 8;
            sbyte* q8_1 = qsArr1 + b * 256;
            short* bsums1 = bsumsArr1 + b * 16;
            float* dSub2 = dArr2 + b * 8;
            sbyte* q8_2 = qsArr2 + b * 256;
            short* bsums2 = bsumsArr2 + b * 16;
            float* dSub3 = dArr3 + b * 8;
            sbyte* q8_3 = qsArr3 + b * 256;
            short* bsums3 = bsumsArr3 + b * 16;

            for (int chunk = 0; chunk < 4; chunk++)
            {
                GetScaleMinK4(2 * chunk, sc, out byte sc1, out byte m1);
                GetScaleMinK4(2 * chunk + 1, sc, out byte sc2, out byte m2);

                var qbytes = Vector256.LoadUnsafe(ref *(qs + chunk * 32));   // shared weight nibbles
                var lo = Avx2.And(qbytes, m0F);
                var hi = Avx2.And(Avx2.ShiftRightLogical(qbytes.AsInt16(), 4).AsByte(), m0F);

                int s0 = 2 * chunk;
                int s1 = 2 * chunk + 1;
                float dsc1 = d * sc1, dm1 = dmin * m1;
                float dsc2 = d * sc2, dm2 = dmin * m2;

                AccumQ4KInput(lo, hi, one16, q8_0, bsums0, dSub0, chunk, s0, s1, dsc1, dm1, dsc2, dm2, ref acc0);
                AccumQ4KInput(lo, hi, one16, q8_1, bsums1, dSub1, chunk, s0, s1, dsc1, dm1, dsc2, dm2, ref acc1);
                AccumQ4KInput(lo, hi, one16, q8_2, bsums2, dSub2, chunk, s0, s1, dsc1, dm1, dsc2, dm2, ref acc2);
                AccumQ4KInput(lo, hi, one16, q8_3, bsums3, dSub3, chunk, s0, s1, dsc1, dm1, dsc2, dm2, ref acc3);
            }
        }
        sum0 = acc0;
        sum1 = acc1;
        sum2 = acc2;
        sum3 = acc3;
    }

    // One input's per-chunk accumulation for the Q4_K_Q8KS quad kernel. The two
    // sub-block terms (s0 then s1) are added to the running `acc` left-to-right,
    // identical to the single-input kernel's `acc += s0term; acc += s1term;`
    // ordering — so each input is bit-identical (not just FP-close) to a single
    // <see cref="DotQ4K_Q8KS"/> dot, the per-token k-independence the routed-MoE
    // byte-parity oracle relies on.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumQ4KInput(Vector256<byte> lo, Vector256<byte> hi, Vector256<short> one16,
        sbyte* q8, short* bsums, float* dSub, int chunk, int s0, int s1,
        float dsc1, float dm1, float dsc2, float dm2, ref float acc)
    {
        var qlo = Vector256.LoadUnsafe(ref *(q8 + chunk * 64)).AsSByte();
        var qhi = Vector256.LoadUnsafe(ref *(q8 + chunk * 64 + 32)).AsSByte();
        int sub0 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(lo, qlo), one16));
        int sub1 = HSumI32_256(Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(hi, qhi), one16));
        int bsum0 = (int)bsums[2 * s0] + (int)bsums[2 * s0 + 1];
        int bsum1 = (int)bsums[2 * s1] + (int)bsums[2 * s1 + 1];
        acc += dSub[s0] * (dsc1 * sub0 - dm1 * bsum0);
        acc += dSub[s1] * (dsc2 * sub1 - dm2 * bsum1);
    }

    // ================================================================
    //  Q6_K · Q8_K Fused two-input dot (issue #42) — decode each Q6_K
    //  super-block ONCE in registers and inner-int-product it against
    //  TWO pre-quantized Q8_K inputs in the same pass. Mirrors the
    //  DotQ4K_2In / DotQ5K_2In pattern but stays in AVX2 since the
    //  one-input kernel is AVX2 (u8·i8 maddubs).
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ6K_Q8K_2In(byte* row, byte* scratch1, byte* scratch2, int cols,
                                       out float sum1, out float sum2)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ6K_Q8K_2In_Avx2(row, scratch1, scratch2, cols, numBlocks, out sum1, out sum2);
            return;
        }
        sum1 = DotQ6K_Q8K_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ6K_Q8K_Scalar(row, scratch2, numBlocks);
    }

    private static void DotQ6K_Q8K_2In_Avx2(byte* row, byte* scratch1, byte* scratch2,
                                             int cols, int numBlocks,
                                             out float sum1, out float sum2)
    {
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 4);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 4 + numBlocks * 256);

        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 4);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper1 = dw * dArr1[i];
            float dSuper2 = dw * dArr2[i];

            // Scales (int16) — shared between both inputs.
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            var scales16 = Avx2.ConvertToVector256Int16(scales128);

            // Per-input offset corrections.
            var q8sums1 = Vector256.LoadUnsafe(ref *(bsumsArr1 + i * 16));
            var q8sclsub1 = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums1, scales16), 5);
            var q8sums2 = Vector256.LoadUnsafe(ref *(bsumsArr2 + i * 16));
            var q8sclsub2 = Avx2.ShiftLeftLogical(
                Avx2.MultiplyAddAdjacent(q8sums2, scales16), 5);

            var sumi1 = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero;
            sbyte* q8a = (sbyte*)(qsArr1 + i * 256);
            sbyte* q8b = (sbyte*)(qsArr2 + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values — shared.
                var q4h_0 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(
                    Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(
                    Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15),
                    q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15),
                    q4h_3);

                int isc = j * 8;
                var sc16_0 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 0]),
                    Vector128.Create((short)sc[isc + 1]));
                var sc16_1 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 2]),
                    Vector128.Create((short)sc[isc + 3]));
                var sc16_2 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 4]),
                    Vector128.Create((short)sc[isc + 5]));
                var sc16_3 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 6]),
                    Vector128.Create((short)sc[isc + 7]));

                // Input 1 pass — q4_X stays live for input 2.
                {
                    var qa0 = Vector256.LoadUnsafe(ref *(q8a + j * 128)).AsSByte();
                    var qa1 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 32)).AsSByte();
                    var qa2 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 64)).AsSByte();
                    var qa3 = Vector256.LoadUnsafe(ref *(q8a + j * 128 + 96)).AsSByte();

                    var pa0 = Avx2.MultiplyAddAdjacent(q4_0, qa0);
                    var pa1 = Avx2.MultiplyAddAdjacent(q4_1, qa1);
                    var pa2 = Avx2.MultiplyAddAdjacent(q4_2, qa2);
                    var pa3 = Avx2.MultiplyAddAdjacent(q4_3, qa3);

                    var sa0 = Avx2.MultiplyAddAdjacent(sc16_0, pa0);
                    var sa1 = Avx2.MultiplyAddAdjacent(sc16_1, pa1);
                    var sa2 = Avx2.MultiplyAddAdjacent(sc16_2, pa2);
                    var sa3 = Avx2.MultiplyAddAdjacent(sc16_3, pa3);

                    sumi1 = Avx2.Add(sumi1, Avx2.Add(Avx2.Add(sa0, sa1), Avx2.Add(sa2, sa3)));
                }

                // Input 2 pass — reuses decoded q4_X and sc16_X.
                {
                    var qb0 = Vector256.LoadUnsafe(ref *(q8b + j * 128)).AsSByte();
                    var qb1 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 32)).AsSByte();
                    var qb2 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 64)).AsSByte();
                    var qb3 = Vector256.LoadUnsafe(ref *(q8b + j * 128 + 96)).AsSByte();

                    var pb0 = Avx2.MultiplyAddAdjacent(q4_0, qb0);
                    var pb1 = Avx2.MultiplyAddAdjacent(q4_1, qb1);
                    var pb2 = Avx2.MultiplyAddAdjacent(q4_2, qb2);
                    var pb3 = Avx2.MultiplyAddAdjacent(q4_3, qb3);

                    var sb0 = Avx2.MultiplyAddAdjacent(sc16_0, pb0);
                    var sb1 = Avx2.MultiplyAddAdjacent(sc16_1, pb1);
                    var sb2 = Avx2.MultiplyAddAdjacent(sc16_2, pb2);
                    var sb3 = Avx2.MultiplyAddAdjacent(sc16_3, pb3);

                    sumi2 = Avx2.Add(sumi2, Avx2.Add(Avx2.Add(sb0, sb1), Avx2.Add(sb2, sb3)));
                }
            }

            acc1 = Fma.MultiplyAdd(
                Vector256.Create(dSuper1),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi1, q8sclsub1)),
                acc1);
            acc2 = Fma.MultiplyAdd(
                Vector256.Create(dSuper2),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi2, q8sclsub2)),
                acc2);
        }
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
    }

    // ================================================================
    //  Q6_K · Q8_K Fused four-input dot (issue #209) — register-tiled
    //  extension of DotQ6K_Q8K_2In: decode each Q6_K super-block ONCE and
    //  inner-int-product it against FOUR pre-quantized Q8_K inputs. Each
    //  input's sumi/acc chain matches the single-input order exactly, so the
    //  result is bit-identical to four DotQ6K_Q8K calls.
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotQ6K_Q8K_4In(byte* row, byte* scratch0, byte* scratch1,
                                       byte* scratch2, byte* scratch3, int cols,
                                       out float sum0, out float sum1, out float sum2, out float sum3)
    {
        int numBlocks = cols / 256;

        if (Avx2.IsSupported && Fma.IsSupported)
        {
            DotQ6K_Q8K_4In_Avx2(row, scratch0, scratch1, scratch2, scratch3, numBlocks,
                out sum0, out sum1, out sum2, out sum3);
            return;
        }
        sum0 = DotQ6K_Q8K_Scalar(row, scratch0, numBlocks);
        sum1 = DotQ6K_Q8K_Scalar(row, scratch1, numBlocks);
        sum2 = DotQ6K_Q8K_Scalar(row, scratch2, numBlocks);
        sum3 = DotQ6K_Q8K_Scalar(row, scratch3, numBlocks);
    }

    private static void DotQ6K_Q8K_4In_Avx2(byte* row,
        byte* scratch0, byte* scratch1, byte* scratch2, byte* scratch3, int numBlocks,
        out float sum0, out float sum1, out float sum2, out float sum3)
    {
        float* dArr0 = (float*)scratch0;
        sbyte* qsArr0 = (sbyte*)(scratch0 + numBlocks * 4);
        short* bsumsArr0 = (short*)(scratch0 + numBlocks * 4 + numBlocks * 256);
        float* dArr1 = (float*)scratch1;
        sbyte* qsArr1 = (sbyte*)(scratch1 + numBlocks * 4);
        short* bsumsArr1 = (short*)(scratch1 + numBlocks * 4 + numBlocks * 256);
        float* dArr2 = (float*)scratch2;
        sbyte* qsArr2 = (sbyte*)(scratch2 + numBlocks * 4);
        short* bsumsArr2 = (short*)(scratch2 + numBlocks * 4 + numBlocks * 256);
        float* dArr3 = (float*)scratch3;
        sbyte* qsArr3 = (sbyte*)(scratch3 + numBlocks * 4);
        short* bsumsArr3 = (short*)(scratch3 + numBlocks * 4 + numBlocks * 256);

        var m3 = Vector256.Create((byte)0x03);
        var m12 = Vector256.Create((byte)0x0C);
        var m48 = Vector256.Create((byte)0x30);
        var m192 = Vector256.Create((byte)0xC0);
        var m15 = Vector256.Create((byte)0x0F);
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        var acc2 = Vector256<float>.Zero;
        var acc3 = Vector256<float>.Zero;

        for (int i = 0; i < numBlocks; i++)
        {
            byte* x = row + i * 210;
            byte* ql = x;
            byte* qh = x + 128;
            sbyte* sc = (sbyte*)(x + 192);
            float dw = HalfToFloat(x[208], x[209]);
            float dSuper0 = dw * dArr0[i];
            float dSuper1 = dw * dArr1[i];
            float dSuper2 = dw * dArr2[i];
            float dSuper3 = dw * dArr3[i];

            // Scales (int16) — shared between all inputs.
            var scales128 = Vector128.LoadUnsafe(ref *(byte*)sc).AsSByte();
            var scales16 = Avx2.ConvertToVector256Int16(scales128);

            // Per-input offset corrections.
            var q8sums0 = Vector256.LoadUnsafe(ref *(bsumsArr0 + i * 16));
            var q8sclsub0 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums0, scales16), 5);
            var q8sums1 = Vector256.LoadUnsafe(ref *(bsumsArr1 + i * 16));
            var q8sclsub1 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums1, scales16), 5);
            var q8sums2 = Vector256.LoadUnsafe(ref *(bsumsArr2 + i * 16));
            var q8sclsub2 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums2, scales16), 5);
            var q8sums3 = Vector256.LoadUnsafe(ref *(bsumsArr3 + i * 16));
            var q8sclsub3 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums3, scales16), 5);

            var sumi0 = Vector256<int>.Zero;
            var sumi1 = Vector256<int>.Zero;
            var sumi2 = Vector256<int>.Zero;
            var sumi3 = Vector256<int>.Zero;
            sbyte* q8a = (sbyte*)(qsArr0 + i * 256);
            sbyte* q8b = (sbyte*)(qsArr1 + i * 256);
            sbyte* q8c = (sbyte*)(qsArr2 + i * 256);
            sbyte* q8d = (sbyte*)(qsArr3 + i * 256);

            for (int j = 0; j < 2; j++)
            {
                var q4bits1 = Vector256.LoadUnsafe(ref *(ql + j * 64));
                var q4bits2 = Vector256.LoadUnsafe(ref *(ql + j * 64 + 32));
                var q4bitsH = Vector256.LoadUnsafe(ref *(qh + j * 32));

                // Reconstruct 4 sets of 32 unsigned 6-bit values — shared across inputs.
                var q4h_0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsInt16(), 4).AsByte();
                var q4h_1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsInt16(), 2).AsByte();
                var q4h_2 = Avx2.And(q4bitsH, m48);
                var q4h_3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, m192).AsInt16(), 2).AsByte();

                var q4_0 = Avx2.Or(Avx2.And(q4bits1, m15), q4h_0);
                var q4_1 = Avx2.Or(Avx2.And(q4bits2, m15), q4h_1);
                var q4_2 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsInt16(), 4).AsByte(), m15), q4h_2);
                var q4_3 = Avx2.Or(
                    Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsInt16(), 4).AsByte(), m15), q4h_3);

                int isc = j * 8;
                var sc16_0 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 0]), Vector128.Create((short)sc[isc + 1]));
                var sc16_1 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 2]), Vector128.Create((short)sc[isc + 3]));
                var sc16_2 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 4]), Vector128.Create((short)sc[isc + 5]));
                var sc16_3 = Vector256.Create(
                    Vector128.Create((short)sc[isc + 6]), Vector128.Create((short)sc[isc + 7]));

                Q6KAccumInput(q8a, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi0);
                Q6KAccumInput(q8b, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi1);
                Q6KAccumInput(q8c, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi2);
                Q6KAccumInput(q8d, j, q4_0, q4_1, q4_2, q4_3, sc16_0, sc16_1, sc16_2, sc16_3, ref sumi3);
            }

            acc0 = Fma.MultiplyAdd(Vector256.Create(dSuper0),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi0, q8sclsub0)), acc0);
            acc1 = Fma.MultiplyAdd(Vector256.Create(dSuper1),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi1, q8sclsub1)), acc1);
            acc2 = Fma.MultiplyAdd(Vector256.Create(dSuper2),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi2, q8sclsub2)), acc2);
            acc3 = Fma.MultiplyAdd(Vector256.Create(dSuper3),
                Avx.ConvertToVector256Single(Avx2.Subtract(sumi3, q8sclsub3)), acc3);
        }
        sum0 = HSum256(acc0);
        sum1 = HSum256(acc1);
        sum2 = HSum256(acc2);
        sum3 = HSum256(acc3);
    }

    /// <summary>One Q8_K input's contribution to a Q6_K super-block half-pair, using
    /// the already-decoded weight sextets (<paramref name="q4_0"/>..<c>q4_3</c>) and
    /// per-group scales. Matches the inline input pass of <see cref="DotQ6K_Q8K_2In_Avx2"/>
    /// exactly so the 4-input path stays bit-identical to the single/pair paths.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q6KAccumInput(sbyte* q8, int j,
        Vector256<byte> q4_0, Vector256<byte> q4_1, Vector256<byte> q4_2, Vector256<byte> q4_3,
        Vector256<short> sc16_0, Vector256<short> sc16_1, Vector256<short> sc16_2, Vector256<short> sc16_3,
        ref Vector256<int> sumi)
    {
        var qa0 = Vector256.LoadUnsafe(ref *(q8 + j * 128)).AsSByte();
        var qa1 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 32)).AsSByte();
        var qa2 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 64)).AsSByte();
        var qa3 = Vector256.LoadUnsafe(ref *(q8 + j * 128 + 96)).AsSByte();

        var pa0 = Avx2.MultiplyAddAdjacent(q4_0, qa0);
        var pa1 = Avx2.MultiplyAddAdjacent(q4_1, qa1);
        var pa2 = Avx2.MultiplyAddAdjacent(q4_2, qa2);
        var pa3 = Avx2.MultiplyAddAdjacent(q4_3, qa3);

        var sa0 = Avx2.MultiplyAddAdjacent(sc16_0, pa0);
        var sa1 = Avx2.MultiplyAddAdjacent(sc16_1, pa1);
        var sa2 = Avx2.MultiplyAddAdjacent(sc16_2, pa2);
        var sa3 = Avx2.MultiplyAddAdjacent(sc16_3, pa3);

        sumi = Avx2.Add(sumi, Avx2.Add(Avx2.Add(sa0, sa1), Avx2.Add(sa2, sa3)));
    }

    // ================================================================
    //  RMS Norm (AVX2)
    // ================================================================

    /// <summary>
    /// Wide-vector RmsNorm (AVX-512, 16 floats/iter). Falls through to the AVX2 path
    /// when Avx512F isn't available so callers can use it unconditionally. The
    /// reduction order differs by ~ULP vs the AVX2 path; only use from forward
    /// passes whose parity tests target internal-only argmax (Gemma 4) rather than
    /// byte-exact llama.cpp output (Qwen3.6-MTP — see
    /// feedback_qkv_matvecdual_breaks_mtp_parity).
    /// </summary>
    public static void RmsNormWide(float* output, float* input, float* weight, int size, float eps)
    {
        if (Avx512F.IsSupported && size >= 16)
        {
            var sumSq = Vector512<float>.Zero;
            int i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                sumSq = Avx512F.FusedMultiplyAdd(v, v, sumSq);
            }
            float ss = HSum512(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector512.Create(scale);

            i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                var w = Avx512F.LoadVector512(weight + i);
                Avx512F.Store(output + i, Avx512F.Multiply(Avx512F.Multiply(v, scaleV), w));
            }
            for (; i < size; i++)
                output[i] = input[i] * scale * weight[i];
        }
        else
        {
            RmsNorm(output, input, weight, size, eps);
        }
    }

    /// <summary>
    /// Wide-vector PureRmsNorm (AVX-512, 16 floats/iter). See <see cref="RmsNormWide"/>
    /// for parity caveats — only use from forward passes whose tests target
    /// internal-only argmax.
    /// </summary>
    public static void PureRmsNormWide(float* output, float* input, int size, float eps)
    {
        if (Avx512F.IsSupported && size >= 16)
        {
            var sumSq = Vector512<float>.Zero;
            int i = 0;
            for (; i + 16 <= size; i += 16)
            {
                var v = Avx512F.LoadVector512(input + i);
                sumSq = Avx512F.FusedMultiplyAdd(v, v, sumSq);
            }
            float ss = HSum512(sumSq);
            for (; i < size; i++) ss += input[i] * input[i];

            float scale = 1.0f / MathF.Sqrt(ss / size + eps);
            var scaleV = Vector512.Create(scale);

            i = 0;
            for (; i + 16 <= size; i += 16)
                Avx512F.Store(output + i, Avx512F.Multiply(Avx512F.LoadVector512(input + i), scaleV));
            for (; i < size; i++)
                output[i] = input[i] * scale;
        }
        else
        {
            PureRmsNorm(output, input, size, eps);
        }
    }

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
    //  Fused GELU-tanh(gate) * up   (AVX2 + scalar fallback)
    // ================================================================

    /// <summary>
    /// Fused tanh-approximate GELU on <paramref name="gate"/> multiplied by
    /// <paramref name="up"/>, written to <paramref name="outp"/>:
    /// <c>outp[i] = gelu_tanh(gate[i]) * up[i]</c> where
    /// <c>gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))</c>.
    /// Used by Gemma-style models (Gemma 4 FFN activation).
    /// </summary>
    public static void GeluTanhMul(float* gate, float* up, float* outp, int n)
    {
        // sqrt(2/π) ≈ 0.7978845608028654
        const float kAlpha = 0.7978845608028654f;
        const float kBeta = 0.044715f;

        if (Fma.IsSupported && n >= 8)
        {
            var half = Vector256.Create(0.5f);
            var one = Vector256.Create(1.0f);
            var two = Vector256.Create(2.0f);
            var alpha = Vector256.Create(kAlpha);
            var beta = Vector256.Create(kBeta);
            // Clamp 2*inner before exp so |inner|>~10 (e.g. ~ ±20 gate inputs from a
            // wide-dim trunk like Gemma 4) doesn't overflow ExpApprox256 to inf and
            // cascade to (inf-1)/(inf+1)=NaN. Safe range for float32 exp is ~[-88, 88];
            // |2*inner|>20 already saturates tanh to ±1 well within float precision.
            var clampHi = Vector256.Create(20.0f);
            var clampLo = Vector256.Create(-20.0f);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var g = Avx.LoadVector256(gate + i);
                var u = Avx.LoadVector256(up + i);
                // inner = alpha * (g + beta * g^3) = alpha * g * (1 + beta * g^2)
                var g2 = Avx.Multiply(g, g);
                var inner = Avx.Multiply(alpha,
                    Avx.Multiply(g, Fma.MultiplyAdd(beta, g2, one)));
                // tanh(inner) via (exp(2x) - 1) / (exp(2x) + 1)
                var twoInner = Avx.Max(clampLo, Avx.Min(clampHi, Avx.Multiply(two, inner)));
                var e2x = ExpApprox256(twoInner);
                var tanh = Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one));
                // 0.5 * g * (1 + tanh) * u
                var gelu = Avx.Multiply(half, Avx.Multiply(g, Avx.Add(one, tanh)));
                Avx.Store(outp + i, Avx.Multiply(gelu, u));
            }
            for (; i < n; i++)
            {
                float gs = gate[i];
                float inner = kAlpha * (gs + kBeta * gs * gs * gs);
                outp[i] = 0.5f * gs * (1.0f + MathF.Tanh(inner)) * up[i];
            }
        }
        else
        {
            GeluTanhMul_Scalar(gate, up, outp, n);
        }
    }

    /// <summary>
    /// Scalar reference for <see cref="GeluTanhMul"/> used by parity tests.
    /// Uses <see cref="MathF.Tanh"/> directly (no exp approximation).
    /// </summary>
    internal static void GeluTanhMul_Scalar(float* gate, float* up, float* outp, int n)
    {
        const float kAlpha = 0.7978845608028654f;
        const float kBeta = 0.044715f;
        for (int i = 0; i < n; i++)
        {
            float gs = gate[i];
            float inner = kAlpha * (gs + kBeta * gs * gs * gs);
            outp[i] = 0.5f * gs * (1.0f + MathF.Tanh(inner)) * up[i];
        }
    }

    // ================================================================
    //  Final-logit softcap (AVX2 + scalar)
    // ================================================================

    /// <summary>
    /// Apply <c>x[i] = tanh(x[i] / cap) * cap</c> in place. Used for the
    /// Gemma 4 final-logit softcap (cap=30) to clip extreme logits while
    /// preserving a smooth gradient near the boundary.
    /// </summary>
    public static void SoftcapInPlace(float* x, int n, float cap)
    {
        if (Fma.IsSupported && n >= 8)
        {
            var one = Vector256.Create(1.0f);
            var two = Vector256.Create(2.0f);
            var capV = Vector256.Create(cap);
            var invCap = Vector256.Create(1.0f / cap);
            // Clamp 2*arg before exp so an extreme pre-softcap logit doesn't overflow
            // ExpApprox256 to inf. |2*arg|>20 already saturates tanh to ±1 well within
            // float precision so the clamp is invisible to the final cap*tanh result.
            var clampHi = Vector256.Create(20.0f);
            var clampLo = Vector256.Create(-20.0f);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var v = Avx.LoadVector256(x + i);
                var arg = Avx.Multiply(v, invCap);
                // tanh(arg) = (exp(2*arg) - 1) / (exp(2*arg) + 1)
                var twoArg = Avx.Max(clampLo, Avx.Min(clampHi, Avx.Multiply(two, arg)));
                var e2x = ExpApprox256(twoArg);
                var tanh = Avx.Divide(Avx.Subtract(e2x, one), Avx.Add(e2x, one));
                Avx.Store(x + i, Avx.Multiply(tanh, capV));
            }
            for (; i < n; i++)
                x[i] = MathF.Tanh(x[i] / cap) * cap;
        }
        else
        {
            for (int i = 0; i < n; i++)
                x[i] = MathF.Tanh(x[i] / cap) * cap;
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

    /// <summary>
    /// NEOX-style RoPE with PARTIAL rotation. Rotates only the first <paramref name="ropeDim"/>
    /// dims of each head; dims <c>[ropeDim, headDim)</c> pass through unchanged.
    ///
    /// Pair convention: for each head and <c>i ∈ [0, ropeDim/2)</c>, the pair
    /// <c>(x[i], x[i + ropeDim/2])</c> is rotated by <c>(cosTab[i], sinTab[i])</c>.
    /// Both <paramref name="cosTab"/> and <paramref name="sinTab"/> must point at the
    /// per-position slice of a table sized with <c>BuildRopeTable(..., ropeDim, theta)</c>
    /// (i.e. <c>ropeDim/2</c> entries).
    ///
    /// Matches llama.cpp's <c>ggml_compute_forward_rope_flt</c> NEOX path with
    /// <c>n_dims=ropeDim</c>: the tail dims are passed through (see ggml ops.cpp:
    /// "fill the remain channels with data from src tensor").
    /// </summary>
    /// <remarks>
    /// Used by hybrid models with partial RoPE (notably qwen35moe: ropeDim=64, headDim=256).
    /// The scalar inner loop is sufficient for the small ropeDim/2 typical for these models;
    /// SIMD on the partial path is a future optimization.
    /// </remarks>
    public static void ApplyRoPECachedNeoxPartial(
        float* x, float* cosTab, float* sinTab,
        int heads, int headDim, int ropeDim)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim", nameof(ropeDim));
        int halfRope = ropeDim / 2;
        for (int h = 0; h < heads; h++)
        {
            float* head = x + h * headDim;
            for (int i = 0; i < halfRope; i++)
            {
                float x0 = head[i];
                float x1 = head[i + halfRope];
                head[i]            = x0 * cosTab[i] - x1 * sinTab[i];
                head[i + halfRope] = x0 * sinTab[i] + x1 * cosTab[i];
            }
            // Dims [ropeDim, headDim) pass through unchanged — nothing to do.
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
        => BuildRopeTable(cosOut, sinOut, maxSeqLen, headDim, theta, null);

    /// <summary>
    /// Variant accepting a per-pair frequency factor array (e.g. Gemma 4
    /// <c>rope_freqs.weight</c> for global layers). When non-null, the raw
    /// inverse frequency is divided by <c>freqFactors[i]</c> for pair i,
    /// so a factor of 1e30 zeros out the rotation for that pair (identity).
    /// </summary>
    public static void BuildRopeTable(float* cosOut, float* sinOut, int maxSeqLen, int headDim, float theta, float* freqFactors)
    {
        int halfDim = headDim / 2;
        float* freqs = stackalloc float[halfDim];
        for (int i = 0; i < halfDim; i++)
        {
            float inv = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
            if (freqFactors != null) inv /= freqFactors[i];
            freqs[i] = inv;
        }

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

    /// <summary>Horizontal sum of a Vector256&lt;int&gt; to a single int (exact, no FP).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int HSumI32_256(Vector256<int> v)
    {
        var lo = v.GetLower();
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse2.Add(lo, hi);
        s = Sse2.Add(s, Sse2.Shuffle(s, 0x4E)); // [2,3,0,1]
        s = Sse2.Add(s, Sse2.Shuffle(s, 0xB1)); // [1,0,3,2]
        return s.ToScalar();
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
