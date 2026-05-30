using System.Runtime.Intrinsics.X86;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity tests for the new <c>SimdKernels.DotQ8_0_Q8K_Avx2</c> rank-2
/// dual-chain VPMADDWD kernel and its <c>DotQ8_0_Q8K_Scalar</c> production
/// fallback.
///
/// Q8_0 weight rows are synthesized by hand: each 34-byte block holds an
/// FP16 super-block scale d_w plus 32 signed int8 quantized weights. Eight
/// consecutive Q8_0 blocks (256 elements, 272 bytes) span one Q8_K input
/// super-block, so cols must be a multiple of 256 (already true for every
/// model dim in the codebase). A float input row is Q8_K-quantized once
/// via <see cref="SimdKernels.QuantizeRowToQ8K"/> and the dot is then
/// compared across four independent implementations:
///
///   1. <see cref="SimdKernels.DotQ8_0"/> — the legacy per-element FP32
///      dequant-FMA path (4× 8 f32 expand + FMA per 32-elem block). Used
///      as a loose FP reference: the Q8_K path quantizes the entire
///      input row once per 256-element super-block (single per-iscale
///      rounding event), while the legacy path applies per-element
///      FP rounding inside the FMA, so the two diverge by up to ~1.5%
///      on uniformly random rows at cols=256 — that's the
///      FP-vs-quantized envelope, not a kernel bug. Bit-exact int-domain
///      correctness is pinned by (3) and (4) below at 1e-4 rel. There
///      is no env-var swap of <c>DotQ8_0</c> → <c>DotQ8_0_Q8K</c> in
///      the engine today; only <c>SHARPI_Q3K_Q8K</c> is wired.
///
///   2. <see cref="SimdKernels.DotQ8_0_Q8K"/> (dispatcher → AVX2 on this
///      host) against the Q8_K-prequantized input.
///
///   3. <c>SimdKernels.DotQ8_0_Q8K_Scalar</c> — production scalar
///      fallback, exposed as <c>internal</c> with InternalsVisibleTo set
///      on the Cpu csproj. Asserting equality between the AVX2 path,
///      production scalar, and the test-local scalar pins the
///      i8·i8 → i32 inner-product, the per-sub-block d_w multiply, and
///      the per-super-block d_y multiply at FP-noise tolerance.
///
///   4. <see cref="DotQ8_0_Q8K_LocalScalar"/> — test-local int-domain
///      scalar reference, kept independent of production so AVX2 vs
///      scalar can be diffed even when both live in the same source file.
///
/// Cases mirror <see cref="SimdKernelsQ3KQ8KTests"/>: cols ∈ {256, 512,
/// 1024, 2048, 4096} exercises 1, 2, 4, 8, 16 super-blocks per row;
/// row counts {8, 33, 64} include a non-multiple-of-8 to catch any future
/// row-block loop off-by-one. A hand-picked single-super-block regression
/// pins the per-sub-block d_w multiply at the 4-decimal scalar.
/// A near-saturation case stresses the Q8_K per-block amax/iscale
/// rounding at the extremes of the i8·i8 product.
/// </summary>
public sealed unsafe class SimdKernelsQ8_0Q8KTests
{
    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/>
    /// Q8_0-encoded values. Block layout (34 bytes / 32 elements):
    /// [d:FP16][qs:32 × int8]. Eight blocks span one 256-elem Q8_K
    /// super-block. All fields are filled from <paramref name="rng"/>
    /// with a small positive FP16 per-block scale.
    /// </summary>
    private static byte[] BuildQ8_0Matrix(int rows, int cols, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
        const int bytesPerBlock = 34;
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;

                // d (FP16, per-block scale) at byte 0..1. Range
                // (0.01, 0.1] mirrors the Q3_K test's plausible super-block
                // magnitudes.
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);

                // qs at byte 2..33: 32 signed int8 quantized weights.
                for (int i = 0; i < 32; i++)
                    bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        }
        return bytes;
    }

    /// <summary>
    /// Build a near-saturation Q8_0 matrix where every quantized weight is
    /// ±127. This stresses the i8·i8 → i32 pair-sum at its maximum
    /// magnitude (|i16·i16| ≤ 16129, pair ≤ 32258) and the Q8_K iscale
    /// rounding when the input is also at saturation.
    /// </summary>
    private static byte[] BuildQ8_0MatrixSaturated(int rows, int cols, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
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
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);

                // ±127 (signed saturation) at every lane.
                for (int i = 0; i < 32; i++)
                {
                    sbyte v = (sbyte)(rng.Next(2) == 0 ? 127 : -127);
                    bytes[off + 2 + i] = (byte)v;
                }
            }
        }
        return bytes;
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>
    /// Float-reference sanity: both the FP32 dequant-FMA path
    /// (<see cref="SimdKernels.DotQ8_0"/>) and the int-domain
    /// <see cref="SimdKernels.DotQ8_0_Q8K"/> path produce finite values
    /// of comparable magnitude on random Q8_0 rows. Unlike Q3_K, Q8_0
    /// has no per-row optimizer, so we use a tighter 1.2× population-max
    /// bound (we'd expect well within 5% in practice; the slack absorbs
    /// the one extra Q8_K input quantization rounding per super-block).
    /// The load-bearing kernel-correctness assertions live in
    /// <c>DotQ8_0_Q8K_Avx2_MatchesIntDomainScalar</c> and
    /// <c>DotQ8_0_Q8K_ProductionScalar_MatchesLocalScalar</c>; the
    /// direction-drift guard against the legacy FP32 path lives in
    /// <c>DotQ8_0_Q8K_MatchesLegacyDotQ8_0</c>.
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_FloatReferenceSanity()
    {
        // The new kernel is AVX2+FMA. On hosts without it the dispatcher
        // falls through to DotQ8_0_Q8K_Scalar, which is exercised
        // directly by DotQ8_0_Q8K_ProductionScalar_MatchesLocalScalar.
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[]
        {
            (8, 256),
            (33, 512),
            (64, 1024),
            (16, 2048),
            (8, 4096),
        })
        {
            var rng = new Random(20260530 + rows * 31 + cols);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 32) * 34;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var refOutput = new float[rows];
            var newOutput = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                    refOutput[r] = SimdKernels.DotQ8_0(wPtr + (long)r * bytesPerRow, iPtr, cols);

                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                for (int r = 0; r < rows; r++)
                    newOutput[r] = SimdKernels.DotQ8_0_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
            }

            float maxAbsRef = 0f, maxAbsNew = 0f;
            for (int r = 0; r < rows; r++)
            {
                Assert.True(float.IsFinite(refOutput[r]),
                    $"DotQ8_0 produced non-finite output rows={rows} cols={cols} r={r}: {refOutput[r]}");
                Assert.True(float.IsFinite(newOutput[r]),
                    $"DotQ8_0_Q8K produced non-finite output rows={rows} cols={cols} r={r}: {newOutput[r]}");
                if (MathF.Abs(refOutput[r]) > maxAbsRef) maxAbsRef = MathF.Abs(refOutput[r]);
                if (MathF.Abs(newOutput[r]) > maxAbsNew) maxAbsNew = MathF.Abs(newOutput[r]);
            }

            // Q8_0 weights are signed-symmetric (no per-row optimizer);
            // only divergence vs the FP path is the one Q8_K input rounding
            // per super-block. Population maximum tracks within ~1.2×.
            Assert.True(maxAbsNew <= 1.2f * maxAbsRef + 1e-3f,
                $"DotQ8_0_Q8K row-pop |max| grew rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
            Assert.True(maxAbsRef <= 1.2f * maxAbsNew + 1e-3f,
                $"DotQ8_0 row-pop |max| grew rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
            Console.WriteLine(
                $"DotQ8_0_Q8K sanity rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
        }
    }

    /// <summary>
    /// Exact-match parity between the AVX2 dispatcher entry and a local
    /// reimplementation of the int-domain scalar reference. This catches
    /// vector-vs-scalar bugs in <c>DotQ8_0_Q8K_Avx2</c>'s dual VPMADDWD
    /// chains at near-zero abs tolerance (modulo float-reduction
    /// ordering, where we still allow a tiny 1e-4 relative slack).
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_Avx2_MatchesIntDomainScalar()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[]
        {
            (4, 256),
            (5, 512),
            (8, 2048),
            (3, 4096),
        })
        {
            var rng = new Random(0xBADC0DE ^ (rows * 131 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 32) * 34;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var avxOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);

                for (int r = 0; r < rows; r++)
                {
                    avxOut[r] = SimdKernels.DotQ8_0_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
                    scalarOut[r] = DotQ8_0_Q8K_LocalScalar(wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(avxOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                // Both paths compute the same integer i8·i8 inner-product
                // and the same per-sub-block d_w / per-super-block d_y FP
                // multiplies; only the accumulation order differs. Either-
                // trip predicate so a large abs error on a large reference
                // can't hide behind a small relative ratio.
                if (diff > 1e-4f || rel > 1e-4f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: avx={avxOut[r]:F6} scalar={scalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"DotQ8_0_Q8K avx2-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ8_0_Q8K AVX2 vs int-scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Exercise the production <c>DotQ8_0_Q8K_Scalar</c> directly (the
    /// fallback path taken on non-AVX2 hosts). Production scalar and
    /// LocalScalar share the same algebra (i8·i8 → i32 inner product,
    /// d_w per sub-block, d_y per super-block, no bsums correction since
    /// Q8_0 is signed-symmetric with no -32 offset), so equality holds at
    /// FP-noise tolerance (1e-4 rel). We also cross-check against the
    /// legacy FP32 dequant-FMA <see cref="SimdKernels.DotQ8_0"/> at the
    /// loose <c>1e-2 + 2e-1 * |fp|</c> bound — the algebraic difference
    /// is one whole-row Q8_K input quantization per super-block, which
    /// produces the FP-vs-quantized envelope (~1.5% on random rows at
    /// cols=256), so this sub-assertion is only a magnitude sanity check.
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_ProductionScalar_MatchesLocalScalar()
    {
        foreach ((int rows, int cols) in new[]
        {
            (4, 256),
            (5, 512),
            (8, 2048),
            (3, 4096),
        })
        {
            var rng = new Random(0x5CA1A5 ^ (rows * 257 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 32) * 34;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var prodScalarOut = new float[rows];
            var localScalarOut = new float[rows];
            var fpRefOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);

                for (int r = 0; r < rows; r++)
                {
                    prodScalarOut[r] = SimdKernels.DotQ8_0_Q8K_Scalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                    localScalarOut[r] = DotQ8_0_Q8K_LocalScalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                    fpRefOut[r] = SimdKernels.DotQ8_0(
                        wPtr + (long)r * bytesPerRow, iPtr, cols);
                }
            }

            // (a) Production scalar vs LocalScalar: same algebra, FP-noise
            //     tolerance.
            int intMismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(prodScalarOut[r] - localScalarOut[r]);
                float rel = diff / (MathF.Abs(localScalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-4f || rel > 1e-4f)
                {
                    if (intMismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: prod={prodScalarOut[r]:F6} local={localScalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    intMismatches++;
                }
            }
            Console.WriteLine(
                $"DotQ8_0_Q8K prod-scalar-vs-local rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={intMismatches}/{rows}");
            Assert.True(intMismatches == 0,
                $"Production DotQ8_0_Q8K_Scalar diverges from local int-domain scalar ({intMismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");

            // (b) Production scalar vs FP32 dequant-FMA: loose sanity bound
            //     (one extra Q8_K input rounding per super-block).
            int fpMismatches = 0;
            float fpMaxAbs = 0, fpMaxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(prodScalarOut[r] - fpRefOut[r]);
                float rel = diff / (MathF.Abs(fpRefOut[r]) + 1e-6f);
                if (diff > fpMaxAbs) fpMaxAbs = diff;
                if (rel > fpMaxRel) fpMaxRel = rel;
                if (diff > 1e-2f + 2e-1f * MathF.Abs(fpRefOut[r]))
                {
                    if (fpMismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: prod={prodScalarOut[r]:F5} fp={fpRefOut[r]:F5} diff={diff:E2} rel={rel:E2}");
                    fpMismatches++;
                }
            }
            Console.WriteLine(
                $"DotQ8_0_Q8K prod-scalar-vs-fp rows={rows} cols={cols}: maxAbs={fpMaxAbs:E2} maxRel={fpMaxRel:E2} mismatches={fpMismatches}/{rows}");
            Assert.True(fpMismatches == 0,
                $"Production DotQ8_0_Q8K_Scalar diverges from FP32 reference beyond sanity bound ({fpMismatches}/{rows}) rows={rows} cols={cols}, maxAbs={fpMaxAbs:E3}, maxRel={fpMaxRel:E3}");
        }
    }

    /// <summary>
    /// Direction-drift guard: parity test for
    /// <see cref="SimdKernels.DotQ8_0_Q8K"/> against the legacy FP32
    /// dequant-FMA <see cref="SimdKernels.DotQ8_0"/>. The Q8_K path
    /// quantizes the entire FP32 input row once per 256-element
    /// super-block (single rounding event with a per-super-block iscale),
    /// while the legacy FP32 path applies per-element rounding implicit
    /// in the FMA accumulation. On uniformly random rows at cols=256
    /// (a single super-block) the path divergence can reach ~1.5%
    /// even though both kernels are bit-correct in their own domain;
    /// this is the FP-vs-quantized rounding floor, not a kernel bug.
    /// Bit-exact int-domain correctness is pinned by
    /// <c>DotQ8_0_Q8K_Avx2_MatchesIntDomainScalar</c> and
    /// <c>DotQ8_0_Q8K_ProductionScalar_MatchesLocalScalar</c> at 1e-4
    /// rel; this test only guards against the kernel silently shifting
    /// magnitude or sign — hence the loose
    /// <c>5e-2 abs + 5e-2 * |legacy|</c> bound, matched to the
    /// empirically-defensible FP-vs-quantized envelope.
    ///
    /// No env-var swap of <c>DotQ8_0</c> → <c>DotQ8_0_Q8K</c> ships
    /// today: only <c>SHARPI_Q3K_Q8K</c> is wired in HybridGdnForwardPass
    /// and CudaHybridGdnForwardPass; the DType.Q8_0 routed-expert branch
    /// still calls <c>DotQ8_0</c> unconditionally. Any future
    /// <c>SHARPI_Q8_0_Q8K</c> gate would dispatch to the same kernel
    /// covered here.
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_MatchesLegacyDotQ8_0()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[]
        {
            (8, 256),
            (33, 512),
            (16, 1024),
            (8, 2048),
            (4, 4096),
        })
        {
            var rng = new Random(0xCA12E ^ (rows * 97 + cols));
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 32) * 34;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var legacyOut = new float[rows];
            var newOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                    legacyOut[r] = SimdKernels.DotQ8_0(wPtr + (long)r * bytesPerRow, iPtr, cols);

                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                for (int r = 0; r < rows; r++)
                    newOut[r] = SimdKernels.DotQ8_0_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
            }

            // Population-level direction-drift guard. The FP-vs-quantized
            // path divergence isn't bounded usefully per-row (the legacy
            // path's per-element FP rounding and the Q8_K path's single
            // per-super-block iscale rounding produce outliers that can
            // reach ~10% on small reference magnitudes in random rows),
            // so we instead assert:
            //   * sign agrees on every row (no direction flips)
            //   * population MAE / mean(|legacy|) ≤ 5%
            //   * at most 1 row with rel > 10% (allow a single FP-vs-
            //     quantized outlier from a near-zero reference)
            // Bit-exact int-domain correctness lives in
            // DotQ8_0_Q8K_Avx2_MatchesIntDomainScalar and
            // DotQ8_0_Q8K_ProductionScalar_MatchesLocalScalar at 1e-4 rel.
            float maxAbs = 0, maxRel = 0;
            double sumAbsDiff = 0, sumAbsRef = 0;
            int signFlips = 0, outliers = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(legacyOut[r] - newOut[r]);
                float denom = MathF.Abs(legacyOut[r]) + 1e-6f;
                float rel = diff / denom;
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                sumAbsDiff += diff;
                sumAbsRef += MathF.Abs(legacyOut[r]);
                // A sign flip on a non-trivial reference (|legacy| > 1e-3)
                // would indicate the kernel is computing the wrong
                // direction, which would shift argmax at the engine level.
                if (MathF.Abs(legacyOut[r]) > 1e-3f &&
                    MathF.Sign(legacyOut[r]) != MathF.Sign(newOut[r]))
                {
                    signFlips++;
                    if (signFlips <= 3)
                        Console.WriteLine(
                            $"  SIGN FLIP rows={rows} cols={cols} [{r}]: legacy={legacyOut[r]:F6} new={newOut[r]:F6}");
                }
                if (rel > 0.10f)
                    outliers++;
            }
            double meanRel = sumAbsRef > 1e-9 ? sumAbsDiff / sumAbsRef : 0;
            Console.WriteLine(
                $"DotQ8_0_Q8K vs legacy DotQ8_0 rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} meanRel={meanRel:E2} signFlips={signFlips} outliers>10%={outliers}/{rows}");
            Assert.True(signFlips == 0,
                $"DotQ8_0_Q8K direction drift vs legacy DotQ8_0: {signFlips} sign flips rows={rows} cols={cols}");
            Assert.True(meanRel <= 0.05,
                $"DotQ8_0_Q8K population MAE/mean(|legacy|) = {meanRel:F4} > 5% rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
            Assert.True(outliers <= 1,
                $"DotQ8_0_Q8K has {outliers} rows with rel>10% (allowed: 1) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Near-saturation stress: every Q8_0 weight is ±127 (signed
    /// saturation). This drives the i8·i8 → i32 pair-sum at the maximum
    /// magnitude (|i16·i16| ≤ 16129, pair ≤ 32258 fits i16, sum-of-8
    /// pairs ≤ 258064 fits i32) and exercises the Q8_K iscale rounding
    /// at the input distribution extreme. AVX2, production scalar, and
    /// LocalScalar all use the same int-domain dot, so they must all
    /// agree at FP-noise tolerance even when the i32 accumulator is
    /// pushed to the high end.
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_NearSaturation()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        foreach ((int rows, int cols) in new[]
        {
            (4, 512),
            (8, 2048),
        })
        {
            var rng = new Random(unchecked((int)0xDEADBEEF) ^ (rows * 17 + cols));
            byte[] weightBytes = BuildQ8_0MatrixSaturated(rows, cols, rng);

            // Saturated input as well: values in [-1, 1] with most of the
            // mass at the tails so Q8_K iscale lands close to its limit.
            var input = new float[cols];
            for (int i = 0; i < cols; i++)
            {
                double u = rng.NextDouble();
                input[i] = (float)(u < 0.5 ? -1.0 + u * 0.1 : 1.0 - (u - 0.5) * 0.1);
            }

            int bytesPerRow = (cols / 32) * 34;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var avxOut = new float[rows];
            var prodScalarOut = new float[rows];
            var localScalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);

                for (int r = 0; r < rows; r++)
                {
                    avxOut[r] = SimdKernels.DotQ8_0_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
                    prodScalarOut[r] = SimdKernels.DotQ8_0_Q8K_Scalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                    localScalarOut[r] = DotQ8_0_Q8K_LocalScalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                Assert.True(float.IsFinite(avxOut[r]),
                    $"DotQ8_0_Q8K avx produced non-finite output rows={rows} cols={cols} r={r}: {avxOut[r]}");
                float diff = MathF.Abs(avxOut[r] - localScalarOut[r]);
                float rel = diff / (MathF.Abs(localScalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-3f || rel > 1e-4f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: avx={avxOut[r]:F6} local={localScalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }

                // Production scalar must also agree with LocalScalar at
                // FP-noise tolerance under saturation.
                float diffProd = MathF.Abs(prodScalarOut[r] - localScalarOut[r]);
                float relProd = diffProd / (MathF.Abs(localScalarOut[r]) + 1e-6f);
                Assert.True(diffProd < 1e-3f || relProd < 1e-4f,
                    $"DotQ8_0_Q8K_Scalar vs local under saturation rows={rows} cols={cols} r={r}: prod={prodScalarOut[r]:F6} local={localScalarOut[r]:F6} diff={diffProd:E2} rel={relProd:E2}");
            }
            Console.WriteLine(
                $"DotQ8_0_Q8K saturation rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ8_0_Q8K AVX2 vs local-scalar under saturation ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Hand-picked single-super-block regression that pins the
    /// per-sub-block d_w multiply at the 4-decimal scalar, independent of
    /// the float reference path. Every Q8_0 sub-block has d_w = 1.0 and
    /// qs[i] = 1; constant 0.5f input → <see cref="SimdKernels.QuantizeRowToQ8K"/>
    /// emits ±127 q8 with d_y = ±0.5/127.
    /// Expected per sub-block: int dot = 32 * 1 * q8_value = 32 * q8.
    /// Summed over 8 sub-blocks: 8 * 32 * q8 = 256 * q8.
    /// Scaled by d_w (=1) and d_y: result = d_y * 1.0 * (256 * q8).
    /// All three implementations (AVX2 dispatcher, production scalar,
    /// LocalScalar) must agree exactly to 4 decimals.
    /// </summary>
    [Fact]
    public void DotQ8_0_Q8K_HandPicked_ConstScale_ConstInput()
    {
        const int cols = 256;
        const int bytesPerBlock = 34;
        const int numQ8_0Blocks = 8; // one Q8_K super-block = 8 Q8_0 blocks
        int bytesPerRow = numQ8_0Blocks * bytesPerBlock;

        var row = new byte[bytesPerRow];

        // For every Q8_0 block: d_w = 1.0 (FP16), qs[i] = 1 for all 32 lanes.
        ushort dHalf = HalfToUshort((Half)1.0f);
        for (int sub = 0; sub < numQ8_0Blocks; sub++)
        {
            int off = sub * bytesPerBlock;
            row[off + 0] = (byte)(dHalf & 0xff);
            row[off + 1] = (byte)(dHalf >> 8);
            for (int i = 0; i < 32; i++)
                row[off + 2 + i] = 1; // signed int8 = +1
        }

        // Constant 0.5f input → Q8_K iscale = -127/0.5 = -254
        // → q8[i] = round(-254 * 0.5) = -127 for every lane, d_y = -0.5/127.
        // Inner i8·i8 dot per sub-block = Σ 1*(-127) = 32 * -127 = -4064.
        // d_w * intDot per sub-block = 1.0 * -4064 = -4064.
        // Sub-block sum over 8 sub-blocks = -32512.
        // Final = d_y * -32512.
        // Derived numerically from the actual q8 below.
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = 0.5f;

        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        var scratch = new byte[scratchBytes];

        float prodScalar, localScalar, dispatcher;
        fixed (byte* wPtr = row)
        fixed (byte* sPtr = scratch)
        fixed (float* iPtr = input)
        {
            SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);

            // Verify the precondition: constant 0.5f input must round to a
            // single ±127 q8 value across every lane.
            sbyte* qsArr = (sbyte*)(sPtr + 4);
            sbyte q8Value = qsArr[0];
            Assert.True(q8Value == 127 || q8Value == -127,
                $"Expected Q8_K to produce ±127 for constant input 0.5, got {q8Value}");
            for (int i = 0; i < cols; i++)
                Assert.Equal(q8Value, qsArr[i]);
            float dy = *(float*)sPtr;

            // Per-sub-block int dot: Σ qw[i] * qy[i] = 32 * 1 * q8Value = 32 * q8.
            // Summed over 8 sub-blocks: 256 * q8. d_w = 1.0 per sub-block.
            // Outer multiply by d_y[0]:
            float expected = dy * (256.0f * q8Value);

            prodScalar = SimdKernels.DotQ8_0_Q8K_Scalar(wPtr, sPtr, 1);
            localScalar = DotQ8_0_Q8K_LocalScalar(wPtr, sPtr, 1);
            dispatcher = SimdKernels.DotQ8_0_Q8K(wPtr, sPtr, cols);

            Assert.Equal(expected, prodScalar, 4);
            Assert.Equal(expected, localScalar, 4);
            Assert.Equal(expected, dispatcher, 4);
        }

        Console.WriteLine(
            $"DotQ8_0_Q8K hand-picked: prod={prodScalar:F6} local={localScalar:F6} avx={dispatcher:F6}");
    }

    /// <summary>
    /// Test-local reimplementation of the int-domain scalar Q8_0·Q8_K
    /// dot, kept independent of the production code so AVX2 vs scalar
    /// can be diffed even when both live in the same source file.
    /// Algebra mirrors the production scalar exactly:
    ///   per Q8_K super-block b:
    ///     subAcc = Σ_{sub=0..7} d_w[sub] * (Σ_{i=0..31} qw[sub,i] · qy[sub,i])
    ///     acc   += d_y[b] * subAcc
    /// where the 8 Q8_0 sub-blocks within one Q8_K super-block live at
    /// row + (b*8 + sub)*34 and qy[sub,i] = scratch.qs[b*256 + sub*32 + i].
    /// The Q8_K bsums region is unused (Q8_0 has no -32 offset).
    /// </summary>
    private static float DotQ8_0_Q8K_LocalScalar(byte* row, byte* scratch, int numBlocks)
    {
        const int bytesPerBlock = 34;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);

        double acc = 0;
        for (int b = 0; b < numBlocks; b++)
        {
            float dy = dArr[b];
            sbyte* q8 = qsArr + b * 256;
            byte* superBase = row + (long)b * 8 * bytesPerBlock;

            double subAcc = 0;
            for (int sub = 0; sub < 8; sub++)
            {
                byte* block = superBase + sub * bytesPerBlock;
                ushort dHalf = (ushort)(block[0] | (block[1] << 8));
                float dw = (float)BitConverter.UInt16BitsToHalf(dHalf);
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
}
