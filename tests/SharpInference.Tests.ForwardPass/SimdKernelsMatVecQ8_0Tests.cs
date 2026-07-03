using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Kernel oracles for <see cref="SimdKernels.MatVecQ8_0"/> and the issue #417
/// dispatcher wiring that routes <see cref="DType.Q8_0"/> through it in
/// <see cref="SimdKernels.MatVec"/>, <see cref="SimdKernels.MatVecDual"/>,
/// <see cref="SimdKernels.MatVec2In"/>, and <see cref="SimdKernels.MatVec4In"/>.
///
/// Q8_0 weight rows are synthesized by hand (34-byte block = [d:FP16 | qs:32×int8]),
/// the same construction as <see cref="SimdKernelsQ8_0Q8KTests"/>. Assertions come
/// in two strengths:
///
///   * BITWISE — MatVecQ8_0 vs per-row <see cref="SimdKernels.DotQ8_0"/>, and each
///     multi-input dispatcher vs N single MatVec calls. These pin the #415
///     re-admission contract: the batched CPU prefill paths dispatch per row via
///     the same DotQ8_0 (DispatchDot / DispatchDot2In / DispatchDot4In fall back
///     to sequential single dots for Q8_0), so any accumulation-order divergence
///     here would silently break the batched-vs-per-token byte-parity oracles.
///
///   * TOLERANCED — MatVecQ8_0 vs a test-local double-accumulation scalar
///     reference (1e-4 either-trip, the same envelope the other Dot* kernel
///     oracles use for vector-vs-scalar reduction-order noise), and vs the legacy
///     dequant→DotF32 fallback route that MatVec used for Q8_0 before #417
///     (1e-3 either-trip — both are float-domain evaluations of the same values,
///     differing only in multiply/reduction order; this documents the
///     argmax-stable numerics change the issue calls out).
///
/// cols ∈ {96, 256, 2048, 4096} covers a non-multiple-of-256 row (Q8_0's block is
/// 32, unlike the 256-elem K-quants) and multi-KB rows; rows ∈ {3, 33, 96} spans
/// both sides of the MinRowsForParallel=64 threshold so the Parallel.For tier and
/// the sequential tier are both exercised.
/// </summary>
public sealed unsafe class SimdKernelsMatVecQ8_0Tests
{
    private static readonly (int Rows, int Cols)[] s_cases =
    {
        (3, 96),
        (33, 256),
        (8, 2048),
        (96, 512),   // rows ≥ MinRowsForParallel(64) → Parallel.For tier
        (96, 4096),
    };

    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/> Q8_0-encoded
    /// values. Block layout (34 bytes / 32 elements): [d:FP16][qs:32 × int8],
    /// small positive FP16 per-block scale — same recipe as
    /// <see cref="SimdKernelsQ8_0Q8KTests"/> minus its 256-multiple restriction
    /// (MatVecQ8_0 only needs cols % 32 == 0).
    /// </summary>
    private static byte[] BuildQ8_0Matrix(int rows, int cols, Random rng)
    {
        if ((cols & 0x1f) != 0)
            throw new ArgumentException("cols must be a multiple of 32.");
        const int bytesPerBlock = 34;
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;

                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = BitConverter.HalfToUInt16Bits((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);

                for (int i = 0; i < 32; i++)
                    bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        }
        return bytes;
    }

    private static float[] RandomInput(int cols, Random rng)
    {
        var input = new float[cols];
        for (int i = 0; i < cols; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);
        return input;
    }

    /// <summary>
    /// Test-local scalar reference: per block, dequantize d·qs[i] in FP32 (the
    /// same per-element rounding as the production kernels) and accumulate the
    /// products in double. Independent of every production code path so a shared
    /// kernel bug can't self-verify.
    /// </summary>
    private static float DotQ8_0_LocalScalar(byte* row, float* input, int cols)
    {
        const int bytesPerBlock = 34;
        int numBlocks = cols / 32;
        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            byte* block = row + b * bytesPerBlock;
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(block[0] | (block[1] << 8)));
            sbyte* qs = (sbyte*)(block + 2);
            float* inp = input + b * 32;
            for (int i = 0; i < 32; i++)
                acc += (float)(d * qs[i]) * (double)inp[i];
        }
        return (float)acc;
    }

    /// <summary>
    /// MatVecQ8_0 must be BIT-IDENTICAL to a per-row DotQ8_0 sweep in both the
    /// sequential and Parallel.For tiers (rows are independent, so the parallel
    /// tier cannot legally reorder any row's accumulation). This is the invariant
    /// the #415 batched paths' byte-parity rests on.
    /// </summary>
    [Fact]
    public void MatVecQ8_0_BitwiseMatchesPerRowDotQ8_0()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(20260703 + rows * 31 + cols);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            float[] input = RandomInput(cols, rng);

            int bytesPerRow = (cols / 32) * 34;
            var matVecOut = new float[rows];
            var perRowOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = matVecOut)
            {
                SimdKernels.MatVecQ8_0(oPtr, wPtr, iPtr, rows, cols);
                for (int r = 0; r < rows; r++)
                    perRowOut[r] = SimdKernels.DotQ8_0(wPtr + (long)r * bytesPerRow, iPtr, cols);
            }

            for (int r = 0; r < rows; r++)
                Assert.True(matVecOut[r].Equals(perRowOut[r]),
                    $"MatVecQ8_0 not bit-identical to per-row DotQ8_0 rows={rows} cols={cols} r={r}: " +
                    $"matvec={matVecOut[r]:R} dot={perRowOut[r]:R}");
            Console.WriteLine($"MatVecQ8_0 bitwise-vs-DotQ8_0 rows={rows} cols={cols}: OK");
        }
    }

    /// <summary>
    /// The MatVec dispatcher must route DType.Q8_0 to the fused kernel — pinned
    /// as bit-equality with MatVecQ8_0. Before #417, Q8_0 took the dequant→DotF32
    /// fallback, whose reduction order differs from DotQ8_0, so this assertion
    /// discriminates the wiring (it would fail on the old route).
    /// </summary>
    [Fact]
    public void MatVec_DispatchesQ8_0_ToFusedKernel()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0x417 ^ (rows * 131 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            float[] input = RandomInput(cols, rng);

            var dispatchOut = new float[rows];
            var fusedOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* dPtr = dispatchOut)
            fixed (float* fPtr = fusedOut)
            {
                SimdKernels.MatVec(dPtr, wPtr, iPtr, rows, cols, DType.Q8_0);
                SimdKernels.MatVecQ8_0(fPtr, wPtr, iPtr, rows, cols);
            }

            for (int r = 0; r < rows; r++)
                Assert.True(dispatchOut[r].Equals(fusedOut[r]),
                    $"MatVec(Q8_0) not routed to MatVecQ8_0 rows={rows} cols={cols} r={r}: " +
                    $"dispatch={dispatchOut[r]:R} fused={fusedOut[r]:R}");
        }
    }

    /// <summary>
    /// Vector-vs-scalar oracle: MatVecQ8_0 against the local double-accumulation
    /// scalar at the usual 1e-4 either-trip envelope (only float-reduction
    /// ordering separates them).
    /// </summary>
    [Fact]
    public void MatVecQ8_0_MatchesLocalScalarReference()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0xBADC0DE ^ (rows * 257 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            float[] input = RandomInput(cols, rng);

            int bytesPerRow = (cols / 32) * 34;
            var kernelOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = kernelOut)
            {
                SimdKernels.MatVecQ8_0(oPtr, wPtr, iPtr, rows, cols);
                for (int r = 0; r < rows; r++)
                    scalarOut[r] = DotQ8_0_LocalScalar(wPtr + (long)r * bytesPerRow, iPtr, cols);
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(kernelOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-4f && rel > 1e-4f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: kernel={kernelOut[r]:F6} scalar={scalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"MatVecQ8_0 vs local scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"MatVecQ8_0 diverges from scalar reference ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Documents the issue #417 numerics change: the fused route vs the legacy
    /// dequant→DotF32 fallback MatVec previously took for Q8_0. Both are
    /// float-domain evaluations of the same dequantized values — only the
    /// multiply/reduction order differs — so a loose 1e-3 either-trip bound
    /// holds; a violation would mean the new route shifted magnitude rather
    /// than reduction order (argmax-UNstable, a real bug).
    /// </summary>
    [Fact]
    public void MatVecQ8_0_MatchesLegacyDequantFallbackWithinFpEnvelope()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0x5CA1A5 ^ (rows * 97 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            float[] input = RandomInput(cols, rng);

            int bytesPerRow = (cols / 32) * 34;
            var fusedOut = new float[rows];
            var legacyOut = new float[rows];
            var rowF32 = new float[cols];

            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = fusedOut)
            fixed (float* rPtr = rowF32)
            {
                SimdKernels.MatVecQ8_0(oPtr, wPtr, iPtr, rows, cols);

                // Legacy route: per-row Dequantize.ToFloat32 → DotF32 (what
                // MatVecDequantFallback did for Q8_0 before #417).
                for (int r = 0; r < rows; r++)
                {
                    Dequantize.ToFloat32(
                        new ReadOnlySpan<byte>(wPtr + (long)r * bytesPerRow, bytesPerRow),
                        rowF32, DType.Q8_0, cols);
                    legacyOut[r] = SimdKernels.DotF32(rPtr, iPtr, cols);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(fusedOut[r] - legacyOut[r]);
                float rel = diff / (MathF.Abs(legacyOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-3f && rel > 1e-3f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: fused={fusedOut[r]:F6} legacy={legacyOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"MatVecQ8_0 vs legacy dequant-fallback rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"MatVecQ8_0 outside FP envelope of legacy dequant route ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>MatVecDual's Q8_0 case must be bit-identical to two MatVec calls.</summary>
    [Fact]
    public void MatVecDual_Q8_0_BitwiseMatchesTwoMatVecs()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0xD0A1 ^ (rows * 31 + cols));
            byte[] w1 = BuildQ8_0Matrix(rows, cols, rng);
            byte[] w2 = BuildQ8_0Matrix(rows, cols, rng);
            float[] input = RandomInput(cols, rng);

            var dual1 = new float[rows]; var dual2 = new float[rows];
            var single1 = new float[rows]; var single2 = new float[rows];

            fixed (byte* w1Ptr = w1)
            fixed (byte* w2Ptr = w2)
            fixed (float* iPtr = input)
            fixed (float* d1 = dual1)
            fixed (float* d2 = dual2)
            fixed (float* s1 = single1)
            fixed (float* s2 = single2)
            {
                SimdKernels.MatVecDual(d1, w1Ptr, d2, w2Ptr, iPtr, rows, cols,
                    DType.Q8_0, DType.Q8_0);
                SimdKernels.MatVec(s1, w1Ptr, iPtr, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s2, w2Ptr, iPtr, rows, cols, DType.Q8_0);
            }

            for (int r = 0; r < rows; r++)
            {
                Assert.True(dual1[r].Equals(single1[r]) && dual2[r].Equals(single2[r]),
                    $"MatVecDual(Q8_0) not bit-identical to two MatVecs rows={rows} cols={cols} r={r}: " +
                    $"dual=({dual1[r]:R},{dual2[r]:R}) single=({single1[r]:R},{single2[r]:R})");
            }
        }
    }

    /// <summary>
    /// MatVec2In's Q8_0 case must be bit-identical to two single MatVec calls —
    /// the duplicated-input-tail contract of the MTP batched-verify callers, and
    /// the same shape as the batched paths' DispatchDot2In Q8_0 fallback.
    /// </summary>
    [Fact]
    public void MatVec2In_Q8_0_BitwiseMatchesTwoMatVecs()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0x21D ^ (rows * 61 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            float[] input1 = RandomInput(cols, rng);
            float[] input2 = RandomInput(cols, rng);

            var two1 = new float[rows]; var two2 = new float[rows];
            var single1 = new float[rows]; var single2 = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (float* i1 = input1)
            fixed (float* i2 = input2)
            fixed (float* t1 = two1)
            fixed (float* t2 = two2)
            fixed (float* s1 = single1)
            fixed (float* s2 = single2)
            {
                SimdKernels.MatVec2In(t1, t2, wPtr, i1, i2, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s1, wPtr, i1, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s2, wPtr, i2, rows, cols, DType.Q8_0);
            }

            for (int r = 0; r < rows; r++)
            {
                Assert.True(two1[r].Equals(single1[r]) && two2[r].Equals(single2[r]),
                    $"MatVec2In(Q8_0) not bit-identical to two MatVecs rows={rows} cols={cols} r={r}: " +
                    $"2in=({two1[r]:R},{two2[r]:R}) single=({single1[r]:R},{single2[r]:R})");
            }
        }
    }

    /// <summary>
    /// MatVec4In's Q8_0 case must be bit-identical to four single MatVec calls
    /// (per-position bits independent of batch width k — the BatchVerify callers'
    /// duplicated-input-tail contract).
    /// </summary>
    [Fact]
    public void MatVec4In_Q8_0_BitwiseMatchesFourMatVecs()
    {
        foreach ((int rows, int cols) in s_cases)
        {
            var rng = new Random(0x41D ^ (rows * 43 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);
            var inputs = new float[4][];
            for (int k = 0; k < 4; k++) inputs[k] = RandomInput(cols, rng);

            var quadOut = new float[4][];
            var singleOut = new float[4][];
            for (int k = 0; k < 4; k++) { quadOut[k] = new float[rows]; singleOut[k] = new float[rows]; }

            fixed (byte* wPtr = weightBytes)
            fixed (float* i0 = inputs[0])
            fixed (float* i1 = inputs[1])
            fixed (float* i2 = inputs[2])
            fixed (float* i3 = inputs[3])
            fixed (float* q0 = quadOut[0])
            fixed (float* q1 = quadOut[1])
            fixed (float* q2 = quadOut[2])
            fixed (float* q3 = quadOut[3])
            fixed (float* s0 = singleOut[0])
            fixed (float* s1 = singleOut[1])
            fixed (float* s2 = singleOut[2])
            fixed (float* s3 = singleOut[3])
            {
                SimdKernels.MatVec4In(q0, q1, q2, q3, wPtr, i0, i1, i2, i3,
                    rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s0, wPtr, i0, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s1, wPtr, i1, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s2, wPtr, i2, rows, cols, DType.Q8_0);
                SimdKernels.MatVec(s3, wPtr, i3, rows, cols, DType.Q8_0);
            }

            for (int k = 0; k < 4; k++)
                for (int r = 0; r < rows; r++)
                    Assert.True(quadOut[k][r].Equals(singleOut[k][r]),
                        $"MatVec4In(Q8_0) not bit-identical to four MatVecs rows={rows} cols={cols} k={k} r={r}: " +
                        $"4in={quadOut[k][r]:R} single={singleOut[k][r]:R}");
        }
    }
}
