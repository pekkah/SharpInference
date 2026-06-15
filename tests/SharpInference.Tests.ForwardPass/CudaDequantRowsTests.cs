using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #247: the Gemma-4 PLE batched pre-pass dequants the chunk's gathered packed
/// rows on the GPU (<see cref="CudaBackend.DequantRowsQ8_0"/> /
/// <see cref="CudaBackend.DequantRowsQ6K"/>) instead of a CPU <c>Parallel.For</c> + a
/// 4×-larger f32 upload. The batched-prefill bit-exact oracle
/// (<c>Gemma4_E4B_BatchedPrefill_GemmOff_MatchesSequentialBitExact</c>) only holds if the
/// GPU dequant is byte-for-byte identical to the CPU <see cref="Dequantize.ToFloat32"/>
/// that the per-token reference uses. These tests prove that directly on synthetic rows —
/// model-free and fast, so no large GGUF or heavy inference is needed.
///
/// Both kernels compute <c>(d·scale)·q</c> with an exact <c>cvt.f32.f16</c> scale, the
/// same arithmetic and order as the CPU dequant, so the result must be bit-identical (not
/// merely close). Silently no-ops on hosts without CUDA, like the rest of the Cuda* tests.
/// </summary>
public sealed unsafe class CudaDequantRowsTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>
    /// Build <paramref name="rows"/> contiguous Q8_0 rows of <paramref name="rowDim"/>
    /// elements. Layout per 32-element block (34 bytes): [d:fp16][qs:32 × int8]. d is a
    /// finite plausible scale; qs is signed 8-bit. Mirrors the Q8_0 PLE-table layout.
    /// </summary>
    private static byte[] BuildQ8_0Rows(int rows, int rowDim, Random rng)
    {
        int blocksPerRow = rowDim / 32;
        int bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 34;
                ushort dHalf = HalfToUshort((Half)(rng.NextDouble() * 0.18 - 0.09));
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 32; i++)
                    bytes[off + 2 + i] = (byte)(sbyte)(rng.Next(255) - 127);
            }
        return bytes;
    }

    /// <summary>
    /// Build <paramref name="rows"/> contiguous Q6_K rows of <paramref name="rowDim"/>
    /// elements. Block layout (210 bytes / 256 elems): [ql:128][qh:64][scales:16 int8][d:fp16].
    /// All bytes are valid Q6_K data (random ql/qh/int8-scales), with a finite fp16 block
    /// scale so neither path produces a NaN that a bit comparison would treat specially.
    /// </summary>
    private static byte[] BuildQ6KRows(int rows, int rowDim, Random rng)
    {
        int blocksPerRow = rowDim / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 210;
                for (int i = 0; i < 208; i++)            // ql + qh + int8 scales
                    bytes[off + i] = (byte)rng.Next(256);
                ushort dHalf = HalfToUshort((Half)(rng.NextDouble() * 0.02 + 0.001));
                bytes[off + 208] = (byte)(dHalf & 0xFF);
                bytes[off + 209] = (byte)(dHalf >> 8);
            }
        return bytes;
    }

    private static void AssertBitIdentical(CudaBackend gpu, byte[] packed, DType dtype,
        int rows, int rowDim, bool q8)
    {
        long count = (long)rows * rowDim;
        var cpu = new float[count];
        Dequantize.ToFloat32(packed, cpu, dtype, count);

        var src = gpu.UploadRaw(packed, TensorShape.D1(packed.Length), dtype);
        var dst = gpu.Allocate(TensorShape.D1(count));
        if (q8) gpu.DequantRowsQ8_0(src, dst, rows, rowDim);
        else    gpu.DequantRowsQ6K(src, dst, rows, rowDim);
        gpu.Synchronize();

        var got = new float[count];
        gpu.Download(dst, got);
        gpu.Free(src);
        gpu.Free(dst);

        int mismatches = 0;
        float maxAbs = 0;
        for (long i = 0; i < count; i++)
        {
            if (BitConverter.SingleToInt32Bits(cpu[i]) != BitConverter.SingleToInt32Bits(got[i]))
            {
                if (mismatches < 3)
                    Console.WriteLine($"  {dtype} rows={rows} dim={rowDim} [{i}]: gpu={got[i]:G9} cpu={cpu[i]:G9}");
                mismatches++;
            }
            maxAbs = MathF.Max(maxAbs, MathF.Abs(cpu[i] - got[i]));
        }
        Assert.True(mismatches == 0,
            $"{dtype} dequant-rows not bit-identical to CPU: {mismatches}/{count} differ " +
            $"(rows={rows}, dim={rowDim}, maxAbs={maxAbs:E3}).");
    }

    [Theory]
    [InlineData(1, 256)]
    [InlineData(3, 512)]
    [InlineData(13, 10752)]   // the real Gemma-4 E4B PLE row (L=42 × pleWidth=256)
    public void DequantRowsQ8_0_BitIdenticalToCpu(int rows, int rowDim)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var rng = new Random(20260615 + rows * 131 + rowDim);
        AssertBitIdentical(gpu, BuildQ8_0Rows(rows, rowDim, rng), DType.Q8_0, rows, rowDim, q8: true);
    }

    [Theory]
    [InlineData(1, 256)]
    [InlineData(3, 512)]
    [InlineData(13, 10752)]   // the real Gemma-4 E4B (q4_0) PLE row
    public void DequantRowsQ6K_BitIdenticalToCpu(int rows, int rowDim)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var rng = new Random(20260615 + rows * 137 + rowDim);
        AssertBitIdentical(gpu, BuildQ6KRows(rows, rowDim, rng), DType.Q6_K, rows, rowDim, q8: false);
    }
}
