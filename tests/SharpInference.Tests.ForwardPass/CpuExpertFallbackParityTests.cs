using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity test for the CPU expert fallback path (issue #54). Synthesizes a
/// single MoE routed expert with Q4_K-encoded gate / up / down weights and
/// confirms the CPU evaluation pipeline (DotQ4K + SiLuMul) matches the
/// CUDA evaluation pipeline (CudaBackend.MatMul + SiLuMul) for the same
/// input. This is the kernel-level analogue of running an SLRU miss
/// through the dispatch-policy CPU fallback vs the GPU-resident SLRU
/// path — the byte layouts are identical so any divergence is purely
/// reduction-order roundoff.
///
/// Silently skips on hosts without CUDA, same pattern as other Cuda* tests.
/// </summary>
public sealed unsafe class CpuExpertFallbackParityTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                // Keep d/dmin tiny so the per-row dot stays in a plausible
                // MoE intermediate range. SiLU is nonlinear, so per-row
                // sums larger than O(10) magnify reduction-order roundoff
                // enough to break a parity comparison even when both
                // kernels are individually correct.
                float d    = (float)(rng.NextDouble() * 0.002 + 0.0005);
                float dmin = (float)(rng.NextDouble() * 0.001 + 0.0005);
                ushort dh = HalfToUshort((Half)d);
                ushort dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF);
                bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF);
                bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12;  i++) bytes[off +  4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off + 16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void CpuExpertFallback_MatchesGpuExpertEvaluation_OnSyntheticQ4K()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int embDim = 512;
        const int expertDim = 256;
        var rng = new Random(20260530);

        byte[] gateBytes = BuildQ4KMatrix(expertDim, embDim, rng);
        byte[] upBytes   = BuildQ4KMatrix(expertDim, embDim, rng);
        byte[] downBytes = BuildQ4KMatrix(embDim, expertDim, rng);

        var input = new float[embDim];
        for (int i = 0; i < embDim; i++)
            input[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        int bprE = (embDim    / 256) * 144;
        int bprX = (expertDim / 256) * 144;

        // CPU expert evaluation: gate + up MatVec, SiLuMul, then down MatVec.
        var cpuGate = new float[expertDim];
        var cpuUp   = new float[expertDim];
        var cpuOut  = new float[embDim];
        fixed (byte* gP = gateBytes)
        fixed (byte* uP = upBytes)
        fixed (byte* dP = downBytes)
        fixed (float* inP = input)
        fixed (float* gOut = cpuGate)
        fixed (float* uOut = cpuUp)
        fixed (float* outP = cpuOut)
        {
            for (int r = 0; r < expertDim; r++)
            {
                gOut[r] = SimdKernels.DotQ4K(gP + (long)r * bprE, inP, embDim);
                uOut[r] = SimdKernels.DotQ4K(uP + (long)r * bprE, inP, embDim);
            }
            SimdKernels.SiLuMul(gOut, uOut, expertDim);
            for (int r = 0; r < embDim; r++)
                outP[r] = SimdKernels.DotQ4K(dP + (long)r * bprX, gOut, expertDim);
        }

        // GPU expert evaluation against the same byte layouts.
        var gpuGateW = gpu.UploadRaw(gateBytes, TensorShape.D1(gateBytes.Length), DType.Q4_K);
        var gpuUpW   = gpu.UploadRaw(upBytes,   TensorShape.D1(upBytes.Length),   DType.Q4_K);
        var gpuDownW = gpu.UploadRaw(downBytes, TensorShape.D1(downBytes.Length), DType.Q4_K);
        var gpuIn    = gpu.Upload(input, TensorShape.D1(embDim));
        var gpuGate  = gpu.Allocate(TensorShape.D1(expertDim));
        var gpuUp    = gpu.Allocate(TensorShape.D1(expertDim));
        var gpuOut   = gpu.Allocate(TensorShape.D1(embDim));

        gpu.MatMul(gpuGate, gpuGateW, gpuIn, DType.Q4_K);
        gpu.MatMul(gpuUp,   gpuUpW,   gpuIn, DType.Q4_K);
        gpu.SiLuMul(gpuGate, gpuUp);
        gpu.MatMul(gpuOut, gpuDownW, gpuGate, DType.Q4_K);
        gpu.Synchronize();

        var gpuResult = new float[embDim];
        gpu.Download(gpuOut, gpuResult);

        gpu.Free(gpuGateW);
        gpu.Free(gpuUpW);
        gpu.Free(gpuDownW);
        gpu.Free(gpuIn);
        gpu.Free(gpuGate);
        gpu.Free(gpuUp);
        gpu.Free(gpuOut);

        // Three-stage pipeline (Q4_K MatVec → SiLuMul → Q4_K MatVec). The
        // CUDA Q4_K matvec quantizes the activation to Q8_1 before the dot,
        // while the CPU path runs fp32 dequant·fp32 dot — so even within a
        // single matvec there's a small divergence (proven within 1e-3 by
        // the per-kernel parity tests). After SiLuMul magnifies near-zero
        // values and the second matvec re-reduces, the per-row absolute
        // error budget grows with sqrt(expertDim). Accept either a small
        // absolute or small relative diff; bound max abs to a fraction of
        // the dynamic range of the GPU output for a meaningful check.
        float gpuRange = 0;
        for (int r = 0; r < embDim; r++)
            gpuRange = MathF.Max(gpuRange, MathF.Abs(gpuResult[r]));

        float absTol = MathF.Max(1e-3f, 0.05f * gpuRange);
        const float relTol = 0.05f;

        int mismatches = 0;
        float maxAbs = 0, maxRel = 0;
        for (int r = 0; r < embDim; r++)
        {
            float diff = MathF.Abs(gpuResult[r] - cpuOut[r]);
            float rel = diff / (MathF.Abs(cpuOut[r]) + 1e-6f);
            maxAbs = MathF.Max(maxAbs, diff);
            maxRel = MathF.Max(maxRel, rel);
            if (diff > absTol && rel > relTol)
            {
                if (mismatches < 3)
                    Console.WriteLine(
                        $"  [{r}]: gpu={gpuResult[r]:F5} cpu={cpuOut[r]:F5} diff={diff:E2} rel={rel:E2}");
                mismatches++;
            }
        }
        Console.WriteLine(
            $"CpuExpertFallback parity: gpuRange={gpuRange:E2} absTol={absTol:E2} maxAbs={maxAbs:E2} maxRel={maxRel:E2} mismatches={mismatches}/{embDim}");
        Assert.True(mismatches == 0,
            $"CPU vs GPU expert evaluation mismatched ({mismatches}/{embDim}), maxAbs={maxAbs:E3}, maxRel={maxRel:E3}");
    }
}
