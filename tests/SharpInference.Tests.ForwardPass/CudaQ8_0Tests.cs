using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity tests for the Phase-0 Q8_0 CUDA kernels: <c>llm_matvec_q8_0</c> and
/// <c>llm_embed_lookup_q8_0</c>. Mirrors <see cref="CudaMatVecQ5KTests"/>: a
/// small Q8_0-encoded matrix is built by hand (FP16 d + 32 int8 quants per
/// block), then dispatched both through the CPU reference
/// (<see cref="SimdKernels.DotQ8_0"/> / <see cref="Dequantize.ToFloat32"/>)
/// and through the GPU kernels, with element-wise tolerance check.
///
/// Silently no-ops on hosts without CUDA, matching the rest of the Cuda* test
/// files. Q8_0 dequant is exact for matching bytes (no quantization step here),
/// so the only error sources are fp16 rounding of d (already identical between
/// CPU and GPU) and float reduction ordering — tolerance 1e-3 absolute or
/// 1e-3 relative.
/// </summary>
public sealed unsafe class CudaQ8_0Tests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    /// <summary>
    /// Build <paramref name="rows"/> rows of <paramref name="cols"/> Q8_0-encoded
    /// values. Layout per 32-element block (34 bytes): [d:fp16][qs:32 × int8].
    /// d is drawn from (0, 0.1] (plausible Q8_0-style scale), qs is signed 8-bit
    /// uniform over [-127, 127].
    /// </summary>
    private static byte[] BuildQ8_0Matrix(int rows, int cols, Random rng)
    {
        if ((cols & 0x1f) != 0)
            throw new ArgumentException("cols must be a multiple of 32 (Q8_0 block size).");
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];

        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 34;
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 32; i++)
                    bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        }
        return bytes;
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    [Fact]
    public void MatVec_Q8_0_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        // Validate the fp32-decode kernel (llm_matvec_q8_0): exact-byte Q8_0 weights
        // dotted with fp32 activations, so only fp16-d rounding + reduction order
        // differ from the CPU reference. Pin off the dp4a path (issue #142), which
        // quantizes the activation to int8 and is covered by the looser test below.
        gpu.Q80Dp4aEnabled = false;

        foreach ((int rows, int cols) in new[] { (256, 256), (1024, 1024), (33, 512) })
        {
            var rng = new Random(20260601 + rows * 31 + cols);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: row-by-row Q8_0 dot product. Q8_0 has no
            // SimdKernels.MatVecQ8_0 entry point yet, but DotQ8_0 + the
            // public 34-byte-per-block stride is enough to compute a
            // per-row reference identical to what a fused matvec would emit.
            int bytesPerRow = (cols / 32) * 34;
            var cpuOutput = new float[rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                    cpuOutput[r] = SimdKernels.DotQ8_0(wPtr + r * bytesPerRow, iPtr, cols);
            }

            // GPU: upload raw Q8_0 bytes and dispatch matvec via the new dtype branch.
            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));

            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q8_0);
            gpu.Synchronize();

            var gpuResult = new float[rows];
            gpu.Download(gpuOutput, gpuResult);

            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            int mismatches = 0;
            float maxAbs = 0, maxRel = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(gpuResult[r] - cpuOutput[r]);
                float rel = diff / (MathF.Abs(cpuOutput[r]) + 1e-6f);
                maxAbs = MathF.Max(maxAbs, diff);
                maxRel = MathF.Max(maxRel, rel);
                if (diff > 1e-3f && rel > 1e-3f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  rows={rows} cols={cols} [{r}]: gpu={gpuResult[r]:F5} cpu={cpuOutput[r]:F5} diff={diff:E2} rel={rel:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"MatVecQ8_0 rows={rows} cols={cols}: maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches == 0,
                $"Q8_0 matvec mismatches ({mismatches}/{rows}) for rows={rows} cols={cols}, maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
        }
    }

    /// <summary>
    /// Issue #142: the dp4a Q8_0 matvec (<c>llm_matvec_q8_0_dp4a</c>) quantizes the
    /// activation to int8 (Q8_1) before the int8·int8 dp4a dot — exactly llama.cpp's
    /// decode matvec. That introduces ~Q8 activation-quant error (~1%), so it tracks
    /// the fp32 CPU reference to a loose relative tolerance, not 1e-3. The aggregate
    /// dot must still be accurate enough to be argmax-stable, which this bounds.
    /// </summary>
    [Fact]
    public void MatVec_Q8_0_Dp4a_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        gpu.Q80Dp4aEnabled = true;

        foreach ((int rows, int cols) in new[] { (256, 256), (1024, 1024), (64, 2560) })
        {
            var rng = new Random(20260605 + rows * 31 + cols);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            int bytesPerRow = (cols / 32) * 34;
            var cpuOutput = new float[rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* iPtr = input)
            {
                for (int r = 0; r < rows; r++)
                    cpuOutput[r] = SimdKernels.DotQ8_0(wPtr + r * bytesPerRow, iPtr, cols);
            }

            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));
            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q8_0);
            gpu.Synchronize();
            var gpuResult = new float[rows];
            gpu.Download(gpuOutput, gpuResult);
            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            // Per-row magnitude scale for a relative bound (the dot of random ±1
            // activations over `cols` int8 weights has stddev ~ sqrt(cols)).
            float refRms = 0f;
            for (int r = 0; r < rows; r++) refRms += cpuOutput[r] * cpuOutput[r];
            refRms = MathF.Sqrt(refRms / rows);

            int mismatches = 0;
            float maxAbs = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(gpuResult[r] - cpuOutput[r]);
                maxAbs = MathF.Max(maxAbs, diff);
                // Q8 activation quant: per-element error ~ scale/2; allow 2% of the
                // typical row magnitude as the absolute envelope.
                if (diff > 0.02f * refRms) mismatches++;
            }
            Console.WriteLine($"MatVecQ8_0-dp4a rows={rows} cols={cols}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{rows}");
            Assert.True(mismatches <= rows / 100 + 1,
                $"dp4a Q8_0 matvec drifted from fp32 reference: {mismatches}/{rows} rows beyond 2% of row RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }

    /// <summary>
    /// Issue #405: the high-MLP mmvq Q8_0 decode matvec (<c>llm_matvec_q8_0_mmvq</c>,
    /// gated by <see cref="CudaBackend.TrunkMatVecFast"/>) is a faithful port of
    /// llama.cpp's <c>mul_mat_vec_q&lt;Q8_0,1&gt;</c>. It quantizes the activation to the
    /// SAME Q8_1 layout and runs the SAME int8 __dp4a dot as <c>llm_matvec_q8_0_dp4a</c>,
    /// only with a higher-MLP load/reduce schedule (128 thr/block, 1 row/block, vdr=2
    /// independent loads). The two int8 dot results are therefore the same integer sum
    /// per row; they can differ only in the float scale/accumulation ORDER across blocks.
    /// So the mmvq output must match the dp4a output to a tight relative envelope (much
    /// tighter than either's drift from the fp32 reference) — and be exactly argmax-stable.
    /// </summary>
    [Fact]
    public void MatVec_Q8_0_Mmvq_MatchesDp4a()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        gpu.Q80Dp4aEnabled = true;   // both paths live under the dp4a branch

        // Shapes mirror the real Qwen3.6-35B-A3B trunk Q8_0 matrices (#405): attn_qkv
        // 2048→8192, attn_gate 2048→4096, ssm_out 4096→2048, ffn shexp 2048→512 / 512→2048.
        foreach ((int rows, int cols) in new[]
                 { (8192, 2048), (4096, 2048), (2048, 4096), (512, 2048), (2048, 512), (33, 512) })
        {
            var rng = new Random(20260625 + rows * 31 + cols);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var input = new float[cols];
            for (int i = 0; i < cols; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            var gpuWeights = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
            var gpuInput = gpu.Upload(input, TensorShape.D1(cols));
            var gpuOutput = gpu.Allocate(TensorShape.D1(rows));

            // Reference: the existing dp4a kernel.
            gpu.TrunkMatVecFast = false;
            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q8_0);
            gpu.Synchronize();
            var dp4aResult = new float[rows];
            gpu.Download(gpuOutput, dp4aResult);

            // New path: the mmvq kernel.
            gpu.TrunkMatVecFast = true;
            gpu.MatMul(gpuOutput, gpuWeights, gpuInput, DType.Q8_0);
            gpu.Synchronize();
            var mmvqResult = new float[rows];
            gpu.Download(gpuOutput, mmvqResult);
            gpu.TrunkMatVecFast = false;

            gpu.Free(gpuWeights);
            gpu.Free(gpuInput);
            gpu.Free(gpuOutput);

            // Per-row magnitude scale for a relative bound.
            float refRms = 0f;
            for (int r = 0; r < rows; r++) refRms += dp4aResult[r] * dp4aResult[r];
            refRms = MathF.Sqrt(refRms / rows);

            int mismatches = 0;
            float maxAbs = 0;
            for (int r = 0; r < rows; r++)
            {
                float diff = MathF.Abs(mmvqResult[r] - dp4aResult[r]);
                maxAbs = MathF.Max(maxAbs, diff);
                // Same int8 dot, only float accumulation order differs → ~1e-4 of row RMS.
                if (diff > 1e-3f * refRms + 1e-4f) mismatches++;
            }
            // argmax must be identical (decode picks the max row).
            int aDp4a = ArgMax(dp4aResult), aMmvq = ArgMax(mmvqResult);

            Console.WriteLine(
                $"MatVecQ8_0-mmvq vs dp4a rows={rows} cols={cols}: maxAbs={maxAbs:E2} refRms={refRms:E2} " +
                $"mismatches={mismatches}/{rows} argmax dp4a={aDp4a} mmvq={aMmvq}");
            Assert.True(mismatches == 0,
                $"mmvq Q8_0 matvec diverged from dp4a: {mismatches}/{rows} rows beyond 0.1% of row RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
            Assert.True(aDp4a == aMmvq,
                $"mmvq Q8_0 matvec argmax {aMmvq} != dp4a argmax {aDp4a} for rows={rows} cols={cols}.");
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    [Fact]
    public void EmbedLookup_Q8_0_MatchesCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // emb_dim must be a multiple of 256 — match the constraint shared by the
        // Q4_K / Q5_K embed-lookup kernels. Two configurations exercise 1 and 4
        // outer iterations of the cooperative-load loop.
        foreach ((int vocab, int embDim) in new[] { (128, 256), (64, 1024) })
        {
            var rng = new Random(20260601 + vocab * 7 + embDim);
            byte[] tableBytes = BuildQ8_0Matrix(vocab, embDim, rng);
            int bytesPerRow = (embDim / 32) * 34;

            // CPU reference: dequant the chosen row via the shared
            // SharpInference.Cpu Dequantize entry point. Identical layout +
            // identical fp16 d decoding means GPU output should be bit-exact
            // for the dequant path (no reduction → no ordering effects).
            int tokenId = 17 % vocab;
            var cpuRow = new float[embDim];
            Dequantize.ToFloat32(
                tableBytes.AsSpan(tokenId * bytesPerRow, bytesPerRow),
                cpuRow, DType.Q8_0, embDim);

            var gpuTable = gpu.UploadRaw(tableBytes, TensorShape.D1(tableBytes.Length), DType.Q8_0);
            var gpuOutput = gpu.Allocate(TensorShape.D1(embDim));
            gpu.EmbedLookupQ8_0(gpuTable, gpuOutput, tokenId, embDim);
            gpu.Synchronize();
            var gpuResult = new float[embDim];
            gpu.Download(gpuOutput, gpuResult);
            gpu.Free(gpuTable);
            gpu.Free(gpuOutput);

            float maxAbs = 0;
            int mismatches = 0;
            for (int i = 0; i < embDim; i++)
            {
                float diff = MathF.Abs(gpuResult[i] - cpuRow[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 1e-5f)
                {
                    if (mismatches < 3)
                        Console.WriteLine(
                            $"  vocab={vocab} embDim={embDim} [{i}]: gpu={gpuResult[i]:F6} cpu={cpuRow[i]:F6} diff={diff:E2}");
                    mismatches++;
                }
            }
            Console.WriteLine(
                $"EmbedLookupQ8_0 vocab={vocab} embDim={embDim}: maxAbs={maxAbs:E2} mismatches={mismatches}/{embDim}");
            Assert.True(mismatches == 0,
                $"Q8_0 embed-lookup mismatches ({mismatches}/{embDim}) for vocab={vocab} embDim={embDim}, maxAbs={maxAbs:E3}");
        }
    }
}
