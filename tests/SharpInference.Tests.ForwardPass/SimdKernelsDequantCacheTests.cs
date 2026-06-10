using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #189: the dequant-once weight cache routes BLAS-path batched matmuls through
/// <see cref="SimdKernels.MatMulBatchedF32"/> with a pre-dequantized F32 weight instead of
/// <see cref="SimdKernels.MatMulBatched"/> (which dequantizes every call). The substitution
/// must be bit-for-bit identical: same F32 weights feed the same SGEMM. This kernel-level
/// test proves that without needing a model file.
/// </summary>
public sealed unsafe class SimdKernelsDequantCacheTests
{
    [Fact]
    public void MatMulBatchedF32_EqualsMatMulBatched_OnBlasPath_BitIdentical()
    {
        // The cache only diverts the OpenBLAS SGEMM path; below the threshold both fall back
        // to fused MatVec, which is a *different* (register-dequant) kernel — out of scope here.
        if (!SimdKernels.BlasAvailable) return;

        const int rows = 96, cols = 160;
        int batch = SimdKernels.MinBatchForBlas + 8; // ensure the SGEMM path is taken
        var rng = new Random(20260610);

        // bf16 weights are trivially constructible and dequantize exactly (zero-extend the
        // mantissa), so MatMulBatched's internal Dequantize.ToFloat32 yields exactly wF32.
        var wBf16 = new ushort[rows * cols];
        var wF32 = new float[rows * cols];
        for (int i = 0; i < wBf16.Length; i++)
        {
            float f = (float)(rng.NextDouble() * 2 - 1);
            ushort bf = (ushort)(BitConverter.SingleToUInt32Bits(f) >> 16);
            wBf16[i] = bf;
            wF32[i] = BitConverter.UInt32BitsToSingle((uint)bf << 16);
        }

        var input = new float[batch * cols];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 2 - 1);

        var outPerCallDequant = new float[batch * rows];
        var outCachedF32 = new float[batch * rows];

        fixed (ushort* wp = wBf16)
        fixed (float* wf = wF32)
        fixed (float* ip = input)
        fixed (float* o1 = outPerCallDequant)
        fixed (float* o2 = outCachedF32)
        {
            SimdKernels.MatMulBatched(o1, (byte*)wp, ip, batch, rows, cols, DType.BFloat16);
            SimdKernels.MatMulBatchedF32(o2, wf, ip, batch, rows, cols);
        }

        for (int i = 0; i < outPerCallDequant.Length; i++)
            Assert.Equal(outPerCallDequant[i], outCachedF32[i]);
    }
}
