using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #141 (MMQ): parity for the int8 tensor-core Q8_0×Q8_1 batched matmul
/// (<see cref="CudaBackend.MatMulBatchedMmq"/>, kernel <c>llm_mmq_q8_0</c>).
/// A small Q8_0 weight matrix [rows×cols] and an fp32 activation batch [nTok×cols]
/// are multiplied both on the GPU (MMQ) and on the CPU (<see cref="SimdKernels.DotQ8_0"/>,
/// fp32 reference). MMQ quantizes the activation to int8 (Q8_1) before the int8 mma,
/// so — like the dp4a decode matvec — it tracks the fp32 reference to a loose
/// per-row-RMS tolerance rather than bit-exactly. The int32 dot itself is exact; the
/// only error is the ~Q8 activation quantization.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMmqQ8_0Tests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static ushort HalfToUshort(Half h) =>
        BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>Build <paramref name="rows"/>×<paramref name="cols"/> Q8_0 (34 B/block).</summary>
    private static byte[] BuildQ8_0Matrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 34;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
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
        return bytes;
    }

    [Fact]
    public void MatMulBatchedMmq_Q8_0_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Cover: square, non-multiple-of-16 rows + non-multiple-of-8 tokens (partial
        // tile guards), and a wide single-block-token batch.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (33, 256, 5), (128, 2560, 64) })
        {
            var rng = new Random(20260605 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ8_0Matrix(rows, cols, rng);

            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, exact-byte Q8_0).
            int bytesPerRow = (cols / 32) * 34;
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            {
                for (int t = 0; t < nTok; t++)
                    for (int r = 0; r < rows; r++)
                        cpuOut[t * rows + r] = SimdKernels.DotQ8_0(wPtr + r * bytesPerRow, aPtr + t * cols, cols);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q8_0);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q8_0);
            gpu.Synchronize();

            var gpuOut = new float[nTok * rows];
            gpu.Download(gpuY, gpuOut);
            gpu.Free(gpuW);
            gpu.Free(gpuX);
            gpu.Free(gpuY);

            // Per-row magnitude scale for the relative bound (dot of ±1 acts over
            // `cols` int8 weights has stddev ~ sqrt(cols)).
            double sumSq = 0;
            for (int i = 0; i < cpuOut.Length; i++) sumSq += (double)cpuOut[i] * cpuOut[i];
            float refRms = (float)Math.Sqrt(sumSq / cpuOut.Length);

            int mismatches = 0;
            float maxAbs = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(gpuOut[i] - cpuOut[i]);
                maxAbs = MathF.Max(maxAbs, diff);
                if (diff > 0.02f * refRms) mismatches++;
            }
            Console.WriteLine(
                $"MMQ rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"MMQ Q8_0 drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 2% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
