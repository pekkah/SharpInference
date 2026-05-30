using System.Runtime.Intrinsics.X86;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity tests for the new <c>SimdKernels.DotQ3K_Q8K_Avx2</c> kernel and
/// its <c>DotQ3K_Q8K_Scalar</c> production fallback.
///
/// Q3_K weight rows are synthesized by hand (random hmask / qs / scales /
/// FP16 super-block d). A float input row is Q8_K-quantized once via
/// <see cref="SimdKernels.QuantizeRowToQ8K"/> and the dot is then compared
/// across three independent implementations:
///
///   1. <see cref="SimdKernels.DotQ3K"/> — per-element FP32 dequant-FMA
///      reference. Used only as a loose sanity bound: random unconstrained
///      Q3_K rows are pathological vs ggml-encoded rows (qs/scales jointly
///      optimised), so the FP and int-domain paths can legitimately diverge
///      by several percent without a kernel bug. The FP cross-check uses
///      <c>relTol = 0.2</c>; the load-bearing correctness assertions are
///      the AVX2-vs-scalar and scalar-vs-LocalScalar tests below.
///
///   2. <see cref="SimdKernels.DotQ3K_Q8K"/> (dispatcher → AVX2 on this
///      host) against the Q8_K-prequantized input.
///
///   3. <c>SimdKernels.DotQ3K_Q8K_Scalar</c> — production scalar fallback,
///      exposed as <c>internal</c> with InternalsVisibleTo. This path uses
///      a different algebra than the test-local <see cref="DotQ3K_Q8K_LocalScalar"/>:
///      production sums <c>qu * sc * y</c> and applies <c>offsetCorr =
///      ((scale-32) · bsums) &lt;&lt; 2</c> as a super-block correction,
///      while LocalScalar folds the <c>-4</c> into per-element math and
///      ignores bsums. Asserting equality between the two scalar
///      implementations pins the bsums emission in QuantizeRowToQ8K and the
///      offsetCorr indexing/shift in production scalar at 0 abs tolerance,
///      so non-AVX2 CI runners cannot ship a silent regression.
///
/// Cases mirror <see cref="CudaMatVecQ5KTests"/>: cols ∈ {256, 512, 1024,
/// 2048, 4096} exercises 1, 2, 4, 8 and 16 super-blocks per row; row counts
/// {8, 33, 64} include a non-multiple-of-8 to catch any future row-block
/// loop off-by-one. A near-saturation case stresses the Q8_K per-block
/// amax/iscale rounding and the −32 scale-bias correction at the extremes
/// of the int8·u3 inner product. A hand-picked single-sub-block regression
/// case pins the −32 constant at zero tolerance, independent of the float
/// reference path.
/// </summary>
public sealed unsafe class SimdKernelsQ3KQ8KTests
{
    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/>
    /// Q3_K-encoded values. Block layout (110 bytes / 256 elements):
    /// [hmask:32][qs:64][scales:12][d:fp16]. All fields are filled from
    /// <paramref name="rng"/> with a small positive FP16 super-block scale.
    /// </summary>
    private static byte[] BuildQ3KMatrix(int rows, int cols, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.");
        int blocksPerRow = cols / 256;
        const int bytesPerBlock = 110;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;

                // hmask at byte 0..31
                for (int i = 0; i < 32; i++)
                    bytes[off + i] = (byte)rng.Next(256);

                // qs at byte 32..95
                for (int i = 0; i < 64; i++)
                    bytes[off + 32 + i] = (byte)rng.Next(256);

                // scales at byte 96..107 (12 packed 6-bit fields). The ggml
                // aux[] unpack handles any random bit pattern, so fill all
                // 12 bytes with random data.
                for (int i = 0; i < 12; i++)
                    bytes[off + 96 + i] = (byte)rng.Next(256);

                // d (FP16, super-block scale) at byte 108..109. Range
                // (0.01, 0.1] mirrors the Q5_K test's plausible super-block
                // magnitudes.
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 108] = (byte)(dHalf & 0xFF);
                bytes[off + 109] = (byte)(dHalf >> 8);
            }
        }
        return bytes;
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>
    /// Loose smoke check: both the FP32 dequant-FMA path
    /// (<see cref="SimdKernels.DotQ3K"/>) and the int-domain
    /// <see cref="SimdKernels.DotQ3K_Q8K"/> path produce finite,
    /// same-sign values of broadly comparable magnitude on random Q3_K
    /// rows. Random unconstrained Q3_K rows are pathological vs ggml-
    /// encoded rows (where qs and scales are jointly optimised to
    /// minimise FP error against the original FP32 row), so a per-row
    /// tight tolerance is not achievable; we only assert that both paths
    /// agree on the order of magnitude across the row population (max
    /// |new| within 2× max |ref|). The load-bearing kernel-correctness
    /// assertions live in <c>DotQ3K_Q8K_Avx2_MatchesIntDomainScalar</c>
    /// and <c>DotQ3K_Q8K_ProductionScalar_MatchesLocalScalar</c>.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8K_FloatReferenceSanity()
    {
        // The new kernel is AVX2+FMA. On hosts without it the dispatcher
        // falls through to DotQ3K_Q8K_Scalar, which would just be a
        // self-check — silently skip and let CI on AVX2 hosts exercise it.
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
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 256) * 110;
            int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var refOutput = new float[rows];
            var newOutput = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                    refOutput[r] = SimdKernels.DotQ3K(wPtr + (long)r * bytesPerRow, iPtr, cols);

                SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);
                for (int r = 0; r < rows; r++)
                    newOutput[r] = SimdKernels.DotQ3K_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
            }

            float maxAbsRef = 0f, maxAbsNew = 0f;
            for (int r = 0; r < rows; r++)
            {
                Assert.True(float.IsFinite(refOutput[r]),
                    $"DotQ3K produced non-finite output rows={rows} cols={cols} r={r}: {refOutput[r]}");
                Assert.True(float.IsFinite(newOutput[r]),
                    $"DotQ3K_Q8K produced non-finite output rows={rows} cols={cols} r={r}: {newOutput[r]}");
                if (MathF.Abs(refOutput[r]) > maxAbsRef) maxAbsRef = MathF.Abs(refOutput[r]);
                if (MathF.Abs(newOutput[r]) > maxAbsNew) maxAbsNew = MathF.Abs(newOutput[r]);
            }

            // Order-of-magnitude bound across the row population. Q8_K
            // quantization of the input plus pathological unconstrained
            // Q3_K weight rows can shift any individual row by tens of
            // percent, but the population maximum should track within
            // ~2× either direction.
            Assert.True(maxAbsNew <= 2.5f * maxAbsRef + 1e-3f,
                $"DotQ3K_Q8K row-pop |max| grew rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
            Assert.True(maxAbsRef <= 2.5f * maxAbsNew + 1e-3f,
                $"DotQ3K row-pop |max| grew rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
            Console.WriteLine(
                $"DotQ3K_Q8K sanity rows={rows} cols={cols}: refMax={maxAbsRef:E2} newMax={maxAbsNew:E2}");
        }
    }

    /// <summary>
    /// Exact-match parity between the AVX2 dispatcher entry and a local
    /// reimplementation of the int-domain scalar reference. This catches
    /// vector-vs-scalar bugs in <c>DotQ3K_Q8K_Avx2</c> at 0 abs tolerance
    /// (modulo float-reduction ordering, where we still allow a tiny
    /// 1e-4 relative slack).
    /// </summary>
    [Fact]
    public void DotQ3K_Q8K_Avx2_MatchesIntDomainScalar()
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
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 256) * 110;
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
                    avxOut[r] = SimdKernels.DotQ3K_Q8K(wPtr + (long)r * bytesPerRow, sPtr, cols);
                    scalarOut[r] = DotQ3K_Q8K_LocalScalar(wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
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
                // int-domain dot is exact; only FP32 reduction-ordering noise
                // on the d_super * sumi step. Either-trip predicate so a
                // large absolute error on a large reference can't hide
                // behind a small relative ratio.
                if (diff > 1e-4f || rel > 1e-4f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: avx={avxOut[r]:F6} scalar={scalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"DotQ3K_Q8K avx2-vs-scalar rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ3K_Q8K AVX2 vs int-scalar mismatch ({mismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Exercise the production <c>DotQ3K_Q8K_Scalar</c> directly (the
    /// fallback path taken on non-AVX2 hosts). Production scalar uses a
    /// substantively different algebra than <see cref="DotQ3K_Q8K_LocalScalar"/>:
    /// it sums <c>qu * sc * y</c> with <c>qu ∈ [0,7]</c> and applies the
    /// <c>(scale-32)·bsums &lt;&lt; 2</c> super-block correction separately,
    /// while LocalScalar folds the <c>-4</c> high-bit subtraction into
    /// per-element math (<c>q = q2 - hb</c>) and ignores bsums. Asserting
    /// equality between the two pins:
    ///   (a) the bsums emission in <see cref="SimdKernels.QuantizeRowToQ8K"/>,
    ///   (b) the bsums indexing and <c>&lt;&lt; 2</c> shift in the
    ///       production scalar's offsetCorr term,
    ///   (c) the sign of offsetCorr (subtracted, not added).
    /// We also cross-check against the FP32 reference at the loose sanity
    /// bound used by <see cref="DotQ3K_Q8K_FloatReferenceSanity"/>.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8K_ProductionScalar_MatchesLocalScalar()
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
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 256) * 110;
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
                    prodScalarOut[r] = SimdKernels.DotQ3K_Q8K_Scalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                    localScalarOut[r] = DotQ3K_Q8K_LocalScalar(
                        wPtr + (long)r * bytesPerRow, sPtr, cols / 256);
                    fpRefOut[r] = SimdKernels.DotQ3K(
                        wPtr + (long)r * bytesPerRow, iPtr, cols);
                }
            }

            // (a) Production scalar vs LocalScalar: 0 abs tolerance.
            int intMismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(prodScalarOut[r] - localScalarOut[r]);
                float rel = diff / (MathF.Abs(localScalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                // Both implementations sum the same integer products and
                // multiply by the same FP super-block scale; only the
                // accumulation order differs, so allow tiny FP slack.
                if (diff > 1e-4f || rel > 1e-4f)
                {
                    if (intMismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: prod={prodScalarOut[r]:F6} local={localScalarOut[r]:F6} diff={diff:E2} rel={rel:E2}");
                    intMismatches++;
                }
            }
            Console.WriteLine(
                $"DotQ3K_Q8K prod-scalar-vs-local rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={intMismatches}/{rows}");
            Assert.True(intMismatches == 0,
                $"Production DotQ3K_Q8K_Scalar diverges from local int-domain scalar ({intMismatches}/{rows}) rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");

            // (b) Production scalar vs FP32 dequant-FMA: loose sanity bound.
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
                $"DotQ3K_Q8K prod-scalar-vs-fp rows={rows} cols={cols}: maxAbs={fpMaxAbs:E2} maxRel={fpMaxRel:E2} mismatches={fpMismatches}/{rows}");
            Assert.True(fpMismatches == 0,
                $"Production DotQ3K_Q8K_Scalar diverges from FP32 reference beyond sanity bound ({fpMismatches}/{rows}) rows={rows} cols={cols}, maxAbs={fpMaxAbs:E3}, maxRel={fpMaxRel:E3}");
        }
    }

    /// <summary>
    /// Hand-picked single-super-block regression that pins the
    /// <c>scale - 32</c> bias at zero tolerance, independent of the float
    /// reference path. A copy-paste regression that flips the constant to
    /// <c>scale - 31</c> in both the AVX2 and LocalScalar paths together
    /// would slip past every randomized test under loose FP tolerance.
    /// Here we construct one super-block where every sub-group has fixed
    /// scale, qu, q8 and bsums values, compute the expected accumulator
    /// from first principles (Σ_g (scale-32) * Σ_l (qu * q8) over the 16
    /// sub-groups, scaled by dSuper) and assert exact equality against
    /// every implementation we ship.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8K_ScaleBiasMinus32_HandPicked()
    {
        const int cols = 256;
        const int bytesPerRow = 110;

        // We want fully deterministic scales after the ggml aux[] unpack,
        // so build the 12 scale bytes such that all 16 unpacked values are
        // the same constant. Easiest: choose every unpacked value = 0x21 (33).
        // (scale - 32) = 1 then, so the integer accumulator becomes simply
        // Σ_g Σ_l (qu * q8) over all 256 elements.
        const sbyte kScale = 33;
        const int kScaleAdj = kScale - 32; // = 1

        var row = new byte[bytesPerRow];

        // hmask = 0 everywhere → high bit always 0 → qu = (qs >> shift) & 3
        // qs[i] = 0x55 = 0b01010101. For shift=0,2,4,6 → low 2 bits are 1.
        // So qu = 1 for every element regardless of shift.
        for (int i = 0; i < 64; i++) row[32 + i] = 0x55;

        // 6-bit packed scales (12 bytes). The ggml aux[] unpacker recovers
        // 16 6-bit scales from this layout:
        //   aux[0..3] are 16-bit values, packed into the 12 bytes as
        //   described in the production code. The simplest way to get all
        //   16 scales = kScale (33 = 0b100001) is to derive the byte pattern
        //   that decodes to {33, 33, ..., 33}. We compute it inline using
        //   the same algebra as the unpacker.
        {
            const uint kmask1 = 0x03030303;
            const uint kmask2 = 0x0f0f0f0f;
            // Forward pack: each scale is 6 bits = low4 (in aux[0..1]) +
            // high2 (in tmp). For all-33 we want low4 = 0x1 and high2 = 0x2.
            // After packing:
            //   aux[0..1].byte = (low4_lo | low4_hi << 4)       (two 4-bit lanes)
            //   tmp.byte  bits = (high2_a | high2_b << 2 | ...) (four 2-bit lanes)
            // For all lanes = 33 = 0b100001:
            //   low4 lanes (in aux[0..7]) = 0x1
            //   high2 lanes (in aux[8..15]) = 0x2
            // → aux[0..1] bytes = 0x11; tmp bytes = 0xAA.
            uint a0 = 0x11111111u;
            uint a1 = 0x11111111u;
            // tmp packs the high 2 bits of all 16 scales: lanes 0..3 in
            // bits 0..7, lanes 4..7 in bits 8..15, lanes 8..11 reuse the
            // same bytes via shifts >> 4 then >> 6. The unpacker is:
            //   aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            //   aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            //   aux[0] = (aux[0]      & kmask2) | (((tmp >> 0) & kmask1) << 4);
            //   aux[1] = (aux[1]      & kmask2) | (((tmp >> 2) & kmask1) << 4);
            // We need post-unpack each byte == 0x21. So:
            //   (aux[i_raw] & 0x0f) == 0x1   → low nibble = 1 for all bytes
            //   ((tmp_shift) & 0x03) == 0x2  → high 2 bits = 2 for all lanes
            // So pre-pack: aux[0..1] both 0x11111111 (each byte = 0x11),
            // and tmp must encode 0x2 in EVERY 2-bit lane, for every shift
            // (>> 0, >> 2, >> 4, >> 6). That means every byte of tmp is
            // 0b10101010 = 0xAA.
            uint tmp = 0xAAAAAAAAu;

            // Pre-unpack form: aux[0..1] hold low nibbles; the unpacker
            // splits them into aux[0,1,2,3] via the shifts above. So we
            // store the inputs the unpacker expects.
            unchecked
            {
                row[96] = (byte)(a0 & 0xff);
                row[97] = (byte)((a0 >> 8) & 0xff);
                row[98] = (byte)((a0 >> 16) & 0xff);
                row[99] = (byte)((a0 >> 24) & 0xff);
                row[100] = (byte)(a1 & 0xff);
                row[101] = (byte)((a1 >> 8) & 0xff);
                row[102] = (byte)((a1 >> 16) & 0xff);
                row[103] = (byte)((a1 >> 24) & 0xff);
                row[104] = (byte)(tmp & 0xff);
                row[105] = (byte)((tmp >> 8) & 0xff);
                row[106] = (byte)((tmp >> 16) & 0xff);
                row[107] = (byte)((tmp >> 24) & 0xff);
            }

            // Sanity: the test-local scale-unpack should agree.
            Span<uint> aux = stackalloc uint[4];
            aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            for (int i = 0; i < 4; i++)
            {
                for (int k = 0; k < 4; k++)
                {
                    int s = (int)((aux[i] >> (k * 8)) & 0xff);
                    Assert.Equal(0x21, s);
                }
            }
        }

        // dSuper = 1.0 (FP16) → makes the expected output an integer.
        {
            ushort dHalf = HalfToUshort((Half)1.0f);
            row[108] = (byte)(dHalf & 0xff);
            row[109] = (byte)(dHalf >> 8);
        }

        // Input: choose all input elements = a constant after Q8_K so the
        // expected sum is trivially computable. Q8_K quantises each
        // sub-block by amax → iscale = -127/-max_signed_amax. For constant
        // input x[i] = 0.5, max = 0.5 → iscale = -254, q8[i] = round(0.5 *
        // -254) = -127. To get q8 = +127 (matching qu's sign) use a NEGATIVE
        // constant: x[i] = -0.5 → iscale = -127/0.5 = -254, q8[i] =
        // round(-0.5 * -254) = +127. Actually ggml's reference implementation
        // computes iscale = -128.0 / max where max preserves sign; we
        // sidestep the sign tangle by asserting the actual q8 value below
        // and deriving kExpected from it dynamically.
        var input = new float[cols];
        for (int i = 0; i < cols; i++) input[i] = 0.5f;

        // Expected (LocalScalar algebra), parameterised by the q8 value
        // that QuantizeRowToQ8K actually emits for a constant input of
        // 0.5: q = q2 - hb, where q2 = 1 (qs[*] = 0x55 → low 2 bits = 1
        // for every shift) and hb = 4 (hmask = 0 → high bit clear). So
        // q = -3 for every element. scaleAdj = 1. dSuper = dAll * dy.
        //   per element: scaleAdj * q * q8 = -3 * q8
        //   whole super-block: 256 * -3 * q8 = -768 * q8
        //   acc = dSuper * sumi = dSuper * (-768 * q8)
        // The production scalar arrives at the same value via offsetCorr;
        // see comment block in DotQ3K_Q8K_Scalar.

        int scratchBytes = SimdKernels.Q8KScratchBytes(cols);
        var scratch = new byte[scratchBytes];

        float prodScalar, localScalar, dispatcher;
        fixed (byte* wPtr = row)
        fixed (byte* sPtr = scratch)
        fixed (float* iPtr = input)
        {
            SimdKernels.QuantizeRowToQ8K(iPtr, cols, sPtr);

            // Verify the Q8_K quantization produced a constant ±127 q8
            // value (precondition for the closed-form kExpected). If this
            // ever fails the rest of the math collapses and we want to
            // see it as a precondition failure, not a kernel mismatch.
            sbyte* qsArr = (sbyte*)(sPtr + 4);
            sbyte q8Value = qsArr[0];
            Assert.True(q8Value == 127 || q8Value == -127,
                $"Expected Q8_K to produce ±127 for constant input 0.5, got {q8Value}");
            for (int i = 0; i < cols; i++)
                Assert.Equal(q8Value, qsArr[i]);
            float dy = *(float*)sPtr;
            // With dAll = 1.0, dSuper = dy. Per-element sumi contribution
            // is scaleAdj * (q2-hb) * q8 = 1 * -3 * q8 = -3 * q8; over 256
            // elements sumi = -768 * q8.
            float expected = dy * (-768.0f * q8Value);

            prodScalar = SimdKernels.DotQ3K_Q8K_Scalar(wPtr, sPtr, 1);
            localScalar = DotQ3K_Q8K_LocalScalar(wPtr, sPtr, 1);
            dispatcher = SimdKernels.DotQ3K_Q8K(wPtr, sPtr, cols);

            Assert.Equal(expected, prodScalar, 4);
            Assert.Equal(expected, localScalar, 4);
            Assert.Equal(expected, dispatcher, 4);
        }

        Console.WriteLine(
            $"DotQ3K_Q8K scale-bias hand-picked: prod={prodScalar:F6} local={localScalar:F6} avx={dispatcher:F6}");
        // Silence "kScaleAdj unused" if the compiler can fold it: the
        // constant is documentary (it pins the math kExpected was derived
        // under). One static check that it equals 1 makes that explicit.
        Assert.Equal(1, kScaleAdj);
    }

    /// <summary>
    /// Test-local reimplementation of the int-domain scalar Q3_K·Q8_K
    /// dot, kept independent of the production code so AVX2 vs scalar
    /// can be diffed even when both live in the same source file.
    /// Mirrors ggml_vec_dot_q3_K_q8_K's algebra exactly:
    ///   per super-block: dSuper * (Σ_g sc_adj_g · Σ_l (qu·q8))
    ///   where sc_adj_g = scale[g] - 32 and qu in [0,7] = (qs>>shift)&3 + 4*hmaskBit.
    /// </summary>
    private static float DotQ3K_Q8K_LocalScalar(byte* row, byte* scratch, int numBlocks)
    {
        const uint kmask1 = 0x03030303;
        const uint kmask2 = 0x0f0f0f0f;
        float* dArr = (float*)scratch;
        sbyte* qsArr = (sbyte*)(scratch + numBlocks * 4);
        // bsums are not needed by the int-domain reference; the scale-bias
        // correction is folded into the (scale-32) multiply per sub-group.

        float acc = 0f;
        Span<uint> aux = stackalloc uint[4];
        Span<sbyte> scales = stackalloc sbyte[16];

        for (int b = 0; b < numBlocks; b++)
        {
            byte* x = row + b * 110;
            ushort dHalf = (ushort)(x[108] | (x[109] << 8));
            float dAll = (float)BitConverter.UInt16BitsToHalf(dHalf);
            float dy = dArr[b];
            float dSuper = dAll * dy;

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

            int sumi = 0;
            int qOff = 0;
            int isIdx = 0;
            int qOut = 0;
            byte m = 1;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    int sc0 = (int)scales[isIdx++] - 32;
                    int sub0 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int q2 = (qs[qOff + l] >> shift) & 3;
                        int hb = (hm[l] & m) != 0 ? 0 : 4;
                        int q = q2 - hb;
                        sub0 += q * q8[qOut + l];
                    }
                    sumi += sc0 * sub0;

                    int sc1 = (int)scales[isIdx++] - 32;
                    int sub1 = 0;
                    for (int l = 0; l < 16; l++)
                    {
                        int q2 = (qs[qOff + 16 + l] >> shift) & 3;
                        int hb = (hm[16 + l] & m) != 0 ? 0 : 4;
                        int q = q2 - hb;
                        sub1 += q * q8[qOut + 16 + l];
                    }
                    sumi += sc1 * sub1;

                    qOut += 32;
                    shift += 2;
                    m <<= 1;
                }
                qOff += 32;
            }

            acc += dSuper * sumi;
        }
        return acc;
    }
}
