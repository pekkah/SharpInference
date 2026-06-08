using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity test for the Q4_0 CUDA matvec kernel (<c>llm_matvec_q4_0</c>), the
/// native packed path for Gemma 4 12B QAT weights (issue #124). Without it q4_0
/// weights fall to the F32-dequant upload (~4× VRAM), defeating full GPU offload.
///
/// Synthesizes Q4_0-encoded weight rows by hand (FP16 d + 16 packed nibble bytes
/// per 32-element block) and compares <see cref="CudaBackend.MatMul"/> against the
/// CPU reference <see cref="SimdKernels.MatVec"/> (Q4_0 → dequant fallback) over
/// the SAME raw bytes — so this validates the GPU dequant-dot semantics, not a
/// quantizer's choice of d. Silently skips on hosts without CUDA.
/// </summary>
public sealed unsafe class CudaMatVecQ40Tests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/> Q4_0-encoded
    /// values. Layout per 32-element block (18 bytes): [d:fp16][qs:16 × uint8],
    /// two signed nibbles per byte. Value = (nibble - 8) * d.
    /// </summary>
    private static byte[] BuildQ4_0Matrix(int rows, int cols, Random rng)
    {
        if ((cols & 0x1f) != 0)
            throw new ArgumentException("cols must be a multiple of 32.");
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 18;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 18;
                // d ∈ (0, 0.1]. Plausible Q4_0-style scale.
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                // 16 nibble bytes: random 4-bit values in both halves.
                for (int i = 0; i < 16; i++)
                    bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        }
        return bytes;
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    [Fact]
    public void MatVecQ4_0_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        // Validate the fp32-decode kernel (llm_matvec_q4_0): exact-byte Q4_0 weights
        // dotted with fp32 activations, so only fp16-d rounding + reduction order differ
        // from the CPU reference. Pin off the dp4a path (issue #124), which quantizes the
        // activation to int8 and is covered by the looser test below.
        gpu.Q40Dp4aEnabled = false;

        // Mix of shapes: small single-block rows, an odd row count (exercises the
        // tail of the 8-rows/block grid), and Gemma 4 12B's real attn/ffn widths.
        foreach ((int rows, int cols) in new[] { (8, 32), (33, 128), (64, 3840), (40, 15360) })
        {
            var rng = new Random(20260608 + rows * 31 + cols);
            byte[] weightBytes = BuildQ4_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: SimdKernels.MatVec routes Q4_0 through the dequant
            // fallback over the identical raw bytes.
            var cpuOutput = new float[rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = cpuOutput)
            {
                SimdKernels.MatVec(oPtr, wPtr, iPtr, rows, cols, DType.Q4_0);
            }

            // GPU: upload raw Q4_0 bytes packed and dispatch the native matvec.
            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_0);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q4_0);
            gpu.Synchronize();

            var gpuResult = new float[rows];
            gpu.Download(gpuOutput, gpuResult);

            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            // Dequant is exact for matching bytes (fp16 d identical on both paths);
            // only float reduction ordering differs. Tolerance 1e-3 abs or rel.
            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(gpuResult[r] - cpuOutput[r]);
                float rel = diff / (MathF.Abs(cpuOutput[r]) + 1e-6f);
                maxAbs = MathF.Max(maxAbs, diff);
                maxRel = MathF.Max(maxRel, rel);
                if (diff > 1e-3f && rel > 1e-3f) mismatches++;
            }
            Console.WriteLine(
                $"MatVecQ4_0 rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"Q4_0 matvec mismatches ({mismatches}/{rows}) for rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Issue #124: the dp4a Q4_0 matvec (<c>llm_matvec_q4_0_dp4a</c>) quantizes the
    /// activation to int8 (Q8_1) before the int8·int8 dp4a dot, using the asymmetric
    /// −8·Σq centering trick (Q4_0 is symmetric). That introduces ~Q8 activation-quant
    /// error (~1%), so it tracks the fp32 CPU reference to a loose relative tolerance,
    /// not 1e-3. The aggregate dot must still be accurate enough to be argmax-stable,
    /// which this bounds. Mirrors <c>CudaQ8_0Tests.MatVec_Q8_0_Dp4a_TracksCpuReference</c>.
    /// </summary>
    [Fact]
    public void MatVecQ4_0_Dp4a_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        gpu.Q40Dp4aEnabled = true;

        foreach ((int rows, int cols) in new[] { (256, 256), (1024, 1024), (64, 3840), (40, 15360) })
        {
            var rng = new Random(20260608 + rows * 31 + cols);
            byte[] weightBytes = BuildQ4_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            var cpuOutput = new float[rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            fixed (float* oPtr = cpuOutput)
            {
                SimdKernels.MatVec(oPtr, wPtr, iPtr, rows, cols, DType.Q4_0);
            }

            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_0);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));
            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q4_0);
            gpu.Synchronize();
            var gpuResult = new float[rows];
            gpu.Download(gpuOutput, gpuResult);
            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            // Per-row magnitude scale for a relative bound (random ±1 activations over
            // `cols` Q4_0 weights → dot stddev ~ sqrt(cols)·scale).
            float refRms = 0f;
            for (int r = 0; r < rows; r++) refRms += cpuOutput[r] * cpuOutput[r];
            refRms = MathF.Sqrt(refRms / rows);

            int mismatches = 0;
            float maxAbs = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(gpuResult[r] - cpuOutput[r]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 0.02f * refRms) mismatches++;
            }
            Console.WriteLine($"MatVecQ4_0-dp4a rows={rows} cols={cols}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches <= rows / 100 + 1,
                $"dp4a Q4_0 matvec drifted from fp32 reference: {mismatches}/{rows} rows beyond 2% of row RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
