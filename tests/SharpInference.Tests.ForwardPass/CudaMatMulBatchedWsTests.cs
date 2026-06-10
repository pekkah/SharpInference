using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #194 bit-exactness tests for <see cref="CudaBackend.MatMulBatchedWeightStationary"/>.
/// The weight-stationary kernels keep the per-(row, token) reduction chain of the GEMM-N
/// matvec and only move the token loop inside the thread block, so the output must be
/// <b>bit-identical</b> to N sequential <see cref="CudaBackend.MatMul"/> calls — the
/// independent per-token reference, NOT the GEMM-N path the kernels were derived from
/// (a path validated only against the path built to mirror it isn't validated).
///
/// Batch sizes exercise every compile-time capacity variant (2/4/8/16) including
/// non-capacity sizes that round up with predicated-off tokens (3, 5, 11), the N=1 and
/// N&gt;16 delegates to the GEMM-N path, and rows not divisible by the 8-row block group.
///
/// Silently skips on hosts without CUDA, mirroring the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMatMulBatchedWsTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    private static void AssertBitIdentical(string label, int rows, int nTok,
                                           float[] batched, float[][] reference)
    {
        for (int t = 0; t < nTok; t++)
            for (int r = 0; r < rows; r++)
            {
                float bat = batched[(long)t * rows + r];
                float refv = reference[t][r];
                if (BitConverter.SingleToInt32Bits(bat) != BitConverter.SingleToInt32Bits(refv))
                    Assert.Fail(
                        $"{label}: token {t} row {r} WS={bat} (0x{BitConverter.SingleToInt32Bits(bat):X8}) " +
                        $"!= sequential GEMV={refv} (0x{BitConverter.SingleToInt32Bits(refv):X8}). " +
                        "MatMulBatchedWeightStationary must be bit-identical to N sequential MatMul calls.");
            }
    }

    /// <summary>Run WS over [nTok × cols] random inputs and compare to nTok sequential
    /// per-token MatMul calls against the same weight tensor, bit-for-bit.</summary>
    private static void RunCase(CudaBackend gpu, Tensor gpuW, DType dtype, string label,
                                int rows, int cols, int nTok, Random rng)
    {
        var inAll = new float[(long)nTok * cols];
        for (int i = 0; i < inAll.Length; i++) inAll[i] = (float)(rng.NextDouble() * 2 - 1);

        var gpuInAll = gpu.Upload(inAll, TensorShape.D1((long)nTok * cols));
        var gpuOutAll = gpu.Allocate(TensorShape.D1((long)nTok * rows));

        gpu.MatMulBatchedWeightStationary(gpuOutAll, gpuW, gpuInAll, nTok, dtype);
        gpu.Synchronize();
        var batched = new float[(long)nTok * rows];
        gpu.Download(gpuOutAll, batched);

        var reference = new float[nTok][];
        for (int t = 0; t < nTok; t++)
        {
            var inT = new float[cols];
            Array.Copy(inAll, (long)t * cols, inT, 0, cols);
            var gpuInT = gpu.Upload(inT, TensorShape.D1(cols));
            var gpuRefT = gpu.Allocate(TensorShape.D1(rows));
            gpu.MatMul(gpuRefT, gpuW, gpuInT, dtype);
            gpu.Synchronize();
            reference[t] = new float[rows];
            gpu.Download(gpuRefT, reference[t]);
            gpu.Free(gpuInT); gpu.Free(gpuRefT);
        }

        gpu.Free(gpuInAll); gpu.Free(gpuOutAll);

        AssertBitIdentical($"{label} rows={rows} cols={cols} nTok={nTok}", rows, nTok, batched, reference);
    }

    // Batch sizes covering each capacity variant, the round-up predication, and the
    // N=1 / N>16 GEMM-N delegates.
    private static readonly int[] BatchSizes = { 1, 2, 3, 4, 5, 8, 11, 16, 17 };

    /// Q4_K layout: 144 bytes per 256-element super-block (matches the GGUF path).
    private static byte[] BuildQ4KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 144;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 144;
                float d    = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.03 + 0.005);
                ushort dh = HalfToUshort((Half)d), dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF); bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF); bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 0; i < 12;  i++) bytes[off +   4 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 128; i++) bytes[off +  16 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void Ws_Q4K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (33, 512), (64, 1024) })
        {
            var rng = new Random(20260610 + rows * 31 + cols * 7);
            byte[] weights = BuildQ4KMatrix(rows, cols, rng);
            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q4_K);
            foreach (int nTok in BatchSizes)
                RunCase(gpu, gpuW, DType.Q4_K, "Q4_K", rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }

    [Fact]
    public void Ws_Q4K_Soa_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int rows = 33, cols = 512;
        var rng = new Random(20260610 + 156);
        byte[] weights = BuildQ4KMatrix(rows, cols, rng);
        var gpuWAos = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q4_K);
        // Repack marks the new handle; both the WS dispatch and the sequential
        // MatMul reference auto-route to their SoA readers.
        var gpuW = gpu.RepackQ4KSoa(gpuWAos, rows, cols);
        foreach (int nTok in BatchSizes)
            RunCase(gpu, gpuW, DType.Q4_K, "Q4_K-SoA", rows, cols, nTok, rng);
        gpu.Free(gpuW);
    }

    /// Q6_K layout: 210 bytes per 256-element super-block.
    private static byte[] BuildQ6KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 210;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 210;
                for (int i = 0; i < 128; i++) bytes[off + i] = (byte)rng.Next(256);
                for (int i = 0; i < 64;  i++) bytes[off + 128 + i] = (byte)rng.Next(256);
                for (int i = 0; i < 16;  i++) bytes[off + 192 + i] = (byte)(rng.Next(33) - 16);
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                ushort dh = HalfToUshort((Half)d);
                bytes[off + 208] = (byte)(dh & 0xFF);
                bytes[off + 209] = (byte)(dh >> 8);
            }
        return bytes;
    }

    [Fact]
    public void Ws_Q6K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (33, 512), (64, 1024) })
        {
            var rng = new Random(20260610 + rows * 37 + cols * 11);
            byte[] weights = BuildQ6KMatrix(rows, cols, rng);
            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q6_K);
            foreach (int nTok in BatchSizes)
                RunCase(gpu, gpuW, DType.Q6_K, "Q6_K", rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }

    /// Q5_K layout: 176 bytes per 256-element super-block
    /// ([d:fp16][dmin:fp16][scales:12][qh:32][ql:128]).
    private static byte[] BuildQ5KMatrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 256;
        int bytesPerRow = blocksPerRow * 176;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 176;
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                float dmin = (float)(rng.NextDouble() * 0.02);
                ushort dh = HalfToUshort((Half)d);
                ushort dmh = HalfToUshort((Half)dmin);
                bytes[off + 0] = (byte)(dh & 0xFF);
                bytes[off + 1] = (byte)(dh >> 8);
                bytes[off + 2] = (byte)(dmh & 0xFF);
                bytes[off + 3] = (byte)(dmh >> 8);
                for (int i = 4; i < 176; i++) bytes[off + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void Ws_Q5K_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (33, 512), (64, 1024) })
        {
            var rng = new Random(20260610 + rows * 41 + cols * 13);
            byte[] weights = BuildQ5KMatrix(rows, cols, rng);
            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q5_K);
            foreach (int nTok in BatchSizes)
                RunCase(gpu, gpuW, DType.Q5_K, "Q5_K", rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }

    /// Q8_0 layout: 34 bytes per 32-element block ([d:fp16][32×int8]).
    private static byte[] BuildQ80Matrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 34;
                float d = (float)(rng.NextDouble() * 0.05 + 0.005);
                ushort dh = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dh & 0xFF);
                bytes[off + 1] = (byte)(dh >> 8);
                for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void Ws_Q8_0_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        // The WS kernels are fp32; pin the sequential MatMul reference to the fp32
        // kernel too (the default dp4a path quantizes the activation to int8 —
        // argmax-stable, not bit-exact). Issue #142.
        gpu.Q80Dp4aEnabled = false;

        foreach ((int rows, int cols) in new[] { (33, 512), (64, 1024) })
        {
            var rng = new Random(20260610 + rows * 17 + cols * 5);
            byte[] weights = BuildQ80Matrix(rows, cols, rng);
            var gpuW = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q8_0);
            foreach (int nTok in BatchSizes)
                RunCase(gpu, gpuW, DType.Q8_0, "Q8_0", rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }

    [Fact]
    public void Ws_Q8_0_Soa_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        gpu.Q80Dp4aEnabled = false;

        const int rows = 33, cols = 512;
        var rng = new Random(20260610 + 149);
        byte[] weights = BuildQ80Matrix(rows, cols, rng);
        var gpuWAos = gpu.UploadRaw(weights, TensorShape.D1(weights.Length), DType.Q8_0);
        var gpuW = gpu.RepackQ8_0Soa(gpuWAos, rows, cols);
        foreach (int nTok in BatchSizes)
            RunCase(gpu, gpuW, DType.Q8_0, "Q8_0-SoA", rows, cols, nTok, rng);
        gpu.Free(gpuW);
    }

    [Fact]
    public void Ws_F32_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        foreach ((int rows, int cols) in new[] { (33, 500), (64, 1024) })
        {
            var rng = new Random(20260610 + rows * 13 + cols * 3);
            var weights = new float[(long)rows * cols];
            for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 2 - 1);
            var gpuW = gpu.Upload(weights, TensorShape.D1((long)rows * cols));
            foreach (int nTok in BatchSizes)
                RunCase(gpu, gpuW, DType.Float32, "F32", rows, cols, nTok, rng);
            gpu.Free(gpuW);
        }
    }
}
