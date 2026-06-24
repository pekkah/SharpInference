using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity tests for the native AVX-512-VNNI Q3_K · Q8_KS dot kernel
/// (<c>sharpi_cpu_vnni.dll</c>, slice 1 of perf/carnice-vnni-moe).
///
/// The native kernel computes the integer sub-block sums with vpdpbusd; only the
/// final per-sub-block float scale FMA is FP, accumulated in the same sub-block
/// order as <see cref="SimdKernels.DotQ3K_Q8KS_Scalar"/>. So the native result
/// must match the scalar reference to FP-noise (the integer <c>subInt</c> is
/// bit-identical by construction).
///
/// All tests early-return (pass, not fail) when <c>Q8VnniInterop.IsAvailable</c>
/// is false — i.e. on non-VNNI hardware, when the DLL is absent, or when the
/// SHARPI_CPU_VNNI kill switch is set — so CI never breaks on machines without
/// AVX512_VNNI. On a VNNI host (where this slice is developed) IsAvailable is
/// true and the body runs.
/// </summary>
public sealed unsafe class Q8VnniParityTests
{
    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>
    /// Builds a matrix of synthetic-but-valid Q3_K rows (110 bytes/super-block).
    /// Identical construction to <c>SimdKernelsQ8KSTests.BuildQ3KMatrix</c>: the
    /// hmask/qs/scale bytes are unconstrained random (the kernel decodes them the
    /// same way regardless of value) and dAll is a small positive half.
    /// </summary>
    private static byte[] BuildQ3KMatrix(int rows, int cols, Random rng)
    {
        if ((cols & 0xff) != 0)
            throw new ArgumentException("cols must be a multiple of 256.", nameof(cols));
        int blocksPerRow = cols / 256;
        const int bytesPerBlock = 110;
        int bytesPerRow = blocksPerRow * bytesPerBlock;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * bytesPerBlock;
                for (int i = 0; i < 32; i++) bytes[off + i] = (byte)rng.Next(256);         // hmask
                for (int i = 0; i < 64; i++) bytes[off + 32 + i] = (byte)rng.Next(256);     // qs
                for (int i = 0; i < 12; i++) bytes[off + 96 + i] = (byte)rng.Next(256);     // scales
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 108] = (byte)(dHalf & 0xFF);
                bytes[off + 109] = (byte)(dHalf >> 8);
            }
        }
        return bytes;
    }

    /// <summary>
    /// The native vpdpbusd kernel and the production scalar reference must agree
    /// across cols ∈ {256, 512, 2048} (1, 2, 8 super-blocks). Uniform random
    /// activations in [-1, 1]. The integer sub-block sums are bit-identical; the
    /// only divergence is FP rounding in the per-sub-block scale FMA, so a tight
    /// absolute+relative tolerance holds.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8KS_Native_MatchesScalar()
    {
        if (!Q8VnniInterop.IsAvailable) return; // skip on non-VNNI hosts / absent DLL

        foreach ((int rows, int cols) in new[] { (4, 256), (5, 512), (8, 2048) })
        {
            var rng = new Random(unchecked((int)0x5174_71E5) ^ (rows * 131 + cols));
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 256) * 110;
            int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var nativeOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8KS(iPtr, cols, sPtr);
                for (int r = 0; r < rows; r++)
                {
                    byte* rowP = wPtr + (long)r * bytesPerRow;
                    // Goes through the dispatcher, which selects the native VNNI
                    // path when IsAvailable (asserted above).
                    nativeOut[r] = SimdKernels.DotQ3K_Q8KS(rowP, sPtr, cols);
                    scalarOut[r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, sPtr, cols / 256);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(nativeOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-5f && rel > 1e-5f) mismatches++;
            }
            Console.WriteLine(
                $"DotQ3K_Q8KS native-vs-scalar rows={rows} cols={cols}: " +
                $"maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ3K_Q8KS native VNNI vs scalar mismatch ({mismatches}/{rows}) " +
                $"rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Non-uniform activation magnitudes (10× variance across sub-blocks, the
    /// post-SiLU routed-MoE shape) stress the per-sub-block scale handling: the
    /// native kernel must still match the scalar reference. This is the case
    /// that motivated Q8_KS over Q8_K, and the one most likely to expose a
    /// sub-block ordering or scale-decode bug in the native kernel.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8KS_Native_MatchesScalar_NonUniform()
    {
        if (!Q8VnniInterop.IsAvailable) return;

        foreach ((int rows, int cols) in new[] { (6, 512), (4, 2048) })
        {
            var rng = new Random(unchecked((int)0x4E07_1F02) ^ (rows * 257 + cols));
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input = new float[cols];
            int nb = cols / 256;
            for (int b = 0; b < nb; b++)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    float mag = (sub % 2 == 0) ? 0.5f : 5.0f;
                    for (int j = 0; j < 32; j++)
                        input[b * 256 + sub * 32 + j] = mag * (float)(rng.NextDouble() * 2 - 1);
                }
            }

            int bytesPerRow = (cols / 256) * 110;
            int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
            var scratch = new byte[scratchBytes];

            var nativeOut = new float[rows];
            var scalarOut = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* sPtr = scratch)
            fixed (float* iPtr = input)
            {
                SimdKernels.QuantizeRowToQ8KS(iPtr, cols, sPtr);
                for (int r = 0; r < rows; r++)
                {
                    byte* rowP = wPtr + (long)r * bytesPerRow;
                    nativeOut[r] = SimdKernels.DotQ3K_Q8KS(rowP, sPtr, cols);
                    scalarOut[r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, sPtr, cols / 256);
                }
            }

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(nativeOut[r] - scalarOut[r]);
                float rel = diff / (MathF.Abs(scalarOut[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-5f && rel > 1e-5f) mismatches++;
            }
            Console.WriteLine(
                $"DotQ3K_Q8KS native-vs-scalar (non-uniform) rows={rows} cols={cols}: " +
                $"maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ3K_Q8KS native VNNI vs scalar (non-uniform) mismatch ({mismatches}/{rows}) " +
                $"rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// The native batched two-input kernel (dequant-once, one native dot per
    /// input) must match the production scalar reference for BOTH inputs across
    /// cols ∈ {256, 512, 2048}. The two inputs use independent random
    /// activations, so a mix-up of the per-input scratch/bsums/d pointers (the
    /// most likely batched-kernel bug) would surface as a mismatch on one input.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8KS_Native_2In_MatchesScalar()
    {
        if (!Q8VnniInterop.IsAvailable) return; // skip on non-VNNI hosts / absent DLL

        foreach ((int rows, int cols) in new[] { (4, 256), (5, 512), (8, 2048) })
        {
            var rng = new Random(unchecked((int)0x2A19_C3D7) ^ (rows * 137 + cols));
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            var input0 = new float[cols];
            var input1 = new float[cols];
            for (int i = 0; i < cols; i++)
            {
                input0[i] = (float)(rng.NextDouble() * 2 - 1);
                input1[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            int bytesPerRow = (cols / 256) * 110;
            int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
            var scratch0 = new byte[scratchBytes];
            var scratch1 = new byte[scratchBytes];

            var native0 = new float[rows];
            var native1 = new float[rows];
            var scalar0 = new float[rows];
            var scalar1 = new float[rows];

            fixed (byte* wPtr = weightBytes)
            fixed (byte* s0 = scratch0)
            fixed (byte* s1 = scratch1)
            fixed (float* i0 = input0)
            fixed (float* i1 = input1)
            {
                SimdKernels.QuantizeRowToQ8KS(i0, cols, s0);
                SimdKernels.QuantizeRowToQ8KS(i1, cols, s1);
                for (int r = 0; r < rows; r++)
                {
                    byte* rowP = wPtr + (long)r * bytesPerRow;
                    // Goes through the dispatcher, which selects the native VNNI
                    // batched path when IsAvailable (asserted above).
                    SimdKernels.DotQ3K_Q8KS_2In(rowP, s0, s1, cols,
                        out native0[r], out native1[r]);
                    scalar0[r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s0, cols / 256);
                    scalar1[r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s1, cols / 256);
                }
            }

            AssertBatchMatches("2In", rows, cols,
                (native0, scalar0), (native1, scalar1));
        }
    }

    /// <summary>
    /// The native batched four-input kernel (dequant-once, one native dot per
    /// input) must match the production scalar reference for ALL FOUR inputs
    /// across cols ∈ {256, 512, 2048}, including the non-uniform 10×-variance
    /// activation shape that stresses the per-sub-block scale handling. Four
    /// independent activations catch any per-input pointer/accumulator swap.
    /// </summary>
    [Fact]
    public void DotQ3K_Q8KS_Native_4In_MatchesScalar()
    {
        if (!Q8VnniInterop.IsAvailable) return;

        foreach ((int rows, int cols) in new[] { (4, 256), (6, 512), (8, 2048) })
        {
            var rng = new Random(unchecked((int)0x71B0_55AAu) ^ (rows * 251 + cols));
            byte[] weightBytes = BuildQ3KMatrix(rows, cols, rng);

            int nb = cols / 256;
            var inputs = new float[4][];
            for (int k = 0; k < 4; k++)
            {
                var arr = new float[cols];
                // Inputs 0/2 uniform; inputs 1/3 non-uniform (10× variance across
                // sub-blocks, the post-SiLU routed-MoE shape).
                bool nonUniform = (k & 1) == 1;
                for (int b = 0; b < nb; b++)
                    for (int sub = 0; sub < 8; sub++)
                    {
                        float mag = nonUniform ? ((sub % 2 == 0) ? 0.5f : 5.0f) : 1.0f;
                        for (int j = 0; j < 32; j++)
                            arr[b * 256 + sub * 32 + j] = mag * (float)(rng.NextDouble() * 2 - 1);
                    }
                inputs[k] = arr;
            }

            int bytesPerRow = nb * 110;
            int scratchBytes = SimdKernels.Q8KSScratchBytes(cols);
            var scratches = new byte[4][];
            for (int k = 0; k < 4; k++) scratches[k] = new byte[scratchBytes];

            var native = new float[4][];
            var scalar = new float[4][];
            for (int k = 0; k < 4; k++) { native[k] = new float[rows]; scalar[k] = new float[rows]; }

            fixed (byte* wPtr = weightBytes)
            fixed (byte* s0 = scratches[0])
            fixed (byte* s1 = scratches[1])
            fixed (byte* s2 = scratches[2])
            fixed (byte* s3 = scratches[3])
            fixed (float* i0 = inputs[0])
            fixed (float* i1 = inputs[1])
            fixed (float* i2 = inputs[2])
            fixed (float* i3 = inputs[3])
            {
                SimdKernels.QuantizeRowToQ8KS(i0, cols, s0);
                SimdKernels.QuantizeRowToQ8KS(i1, cols, s1);
                SimdKernels.QuantizeRowToQ8KS(i2, cols, s2);
                SimdKernels.QuantizeRowToQ8KS(i3, cols, s3);
                for (int r = 0; r < rows; r++)
                {
                    byte* rowP = wPtr + (long)r * bytesPerRow;
                    SimdKernels.DotQ3K_Q8KS_4In(rowP, s0, s1, s2, s3, cols,
                        out native[0][r], out native[1][r], out native[2][r], out native[3][r]);
                    scalar[0][r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s0, nb);
                    scalar[1][r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s1, nb);
                    scalar[2][r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s2, nb);
                    scalar[3][r] = SimdKernels.DotQ3K_Q8KS_Scalar(rowP, s3, nb);
                }
            }

            AssertBatchMatches("4In", rows, cols,
                (native[0], scalar[0]), (native[1], scalar[1]),
                (native[2], scalar[2]), (native[3], scalar[3]));
        }
    }

    /// <summary>
    /// Asserts every (native, scalar) input pair agrees to FP-noise (same tight
    /// tolerance as the single-input parity tests: relErr &lt; 1e-5, 0
    /// mismatches), reporting per-input maxAbs/maxRel/mismatches.
    /// </summary>
    private static void AssertBatchMatches(string label, int rows, int cols,
        params (float[] native, float[] scalar)[] pairs)
    {
        for (int p = 0; p < pairs.Length; p++)
        {
            (float[] native, float[] scalar) = pairs[p];
            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(native[r] - scalar[r]);
                float rel = diff / (MathF.Abs(scalar[r]) + 1e-6f);
                if (diff > maxAbs) maxAbs = diff;
                if (rel > maxRel) maxRel = rel;
                if (diff > 1e-5f && rel > 1e-5f) mismatches++;
            }
            Console.WriteLine(
                $"DotQ3K_Q8KS native {label} input{p} rows={rows} cols={cols}: " +
                $"maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"DotQ3K_Q8KS native {label} input{p} vs scalar mismatch " +
                $"({mismatches}/{rows}) rows={rows} cols={cols}, " +
                $"maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Confirms the test is actually exercising the native path on this host.
    /// <see cref="Q8VnniInterop.HasVnniSupport"/> is the native CPUID probe (the
    /// same one <see cref="Q8VnniInterop.IsAvailable"/> uses) — when the CPU
    /// reports AVX512_VNNI and the kill switch is off, the native path MUST be
    /// available, else the parity tests above are silently no-op'ing and the
    /// slice is unverified. On non-VNNI hardware / CI this is legitimately
    /// false and the assertion is skipped, so the test never fails there.
    /// </summary>
    [Fact]
    public void Native_IsAvailable_OnVnniHost()
    {
        bool killed = Environment.GetEnvironmentVariable("SHARPI_CPU_VNNI") == "0";
        if (Q8VnniInterop.HasVnniSupport && !killed)
        {
            Assert.True(Q8VnniInterop.IsAvailable,
                "AVX512_VNNI is supported by the CPU and the native " +
                "sharpi_cpu_vnni.dll loads, but the native path is not available " +
                "— did the DLL fail to build/copy next to the test binary?");
        }
    }
}
