using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #124/#173 (MMQ): parity for the int8 tensor-core Q4_0×Q8_1 batched matmul
/// (<see cref="CudaBackend.MatMulBatchedMmq"/>, kernel <c>llm_mmq_q4_0</c>) — the
/// Gemma 4 12B QAT prefill path. A small Q4_0 weight matrix [rows×cols] and an fp32
/// activation batch [nTok×cols] are multiplied both on the GPU (MMQ) and on the CPU
/// (<see cref="SimdKernels.MatVec"/>, Q4_0 dequant fallback, fp32 reference) over the
/// SAME raw bytes. MMQ quantizes the activation to int8 (Q8_1) before the int8 mma
/// and dots the RAW Q4_0 nibbles with the −8·d_w·Σq symmetric centering trick, so —
/// like the dp4a decode matvec — it tracks the fp32 reference to a loose per-row-RMS
/// tolerance rather than bit-exactly. The int32 dot itself is exact; the only error is
/// the ~Q8 activation quantization. Mirrors <see cref="CudaMmqQ8_0Tests"/>.
///
/// Silent no-op on hosts without CUDA, matching the other Cuda* test files.
/// </summary>
public sealed unsafe class CudaMmqQ40Tests
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
    /// Build <paramref name="rows"/>×<paramref name="cols"/> Q4_0 (18 B/block):
    /// [d:fp16][qs:16 × uint8], two nibbles per byte. Value = (nibble − 8) · d.
    /// </summary>
    private static byte[] BuildQ4_0Matrix(int rows, int cols, Random rng)
    {
        int blocksPerRow = cols / 32;
        int bytesPerRow = blocksPerRow * 18;
        var bytes = new byte[rows * bytesPerRow];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * bytesPerRow + b * 18;
                float d = (float)(rng.NextDouble() * 0.09 + 0.01);
                ushort dHalf = HalfToUshort((Half)d);
                bytes[off + 0] = (byte)(dHalf & 0xFF);
                bytes[off + 1] = (byte)(dHalf >> 8);
                for (int i = 0; i < 16; i++)
                    bytes[off + 2 + i] = (byte)rng.Next(256);
            }
        return bytes;
    }

    [Fact]
    public void MatMulBatchedMmq_Q4_0_TracksCpuReference()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        // Cover: square, non-multiple-of-16 rows + non-multiple-of-8 tokens (partial
        // tile guards), a wide single-block-token batch, and Gemma 4 12B's ffn width.
        foreach ((int rows, int cols, int nTok) in new[]
                 { (256, 256, 8), (1024, 512, 12), (33, 256, 5), (128, 2560, 64), (40, 15360, 16) })
        {
            var rng = new Random(20260608 + rows * 31 + cols * 7 + nTok);
            byte[] weightBytes = BuildQ4_0Matrix(rows, cols, rng);

            var acts = new float[nTok * cols];
            for (int i = 0; i < acts.Length; i++)
                acts[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference: out[t*rows + r] = Σ W[r]·acts[t]  (fp32, Q4_0 dequant).
            var cpuOut = new float[nTok * rows];
            fixed (byte* wPtr = weightBytes)
            fixed (float* aPtr = acts)
            fixed (float* oPtr = cpuOut)
            {
                for (int t = 0; t < nTok; t++)
                    SimdKernels.MatVec(oPtr + t * rows, wPtr, aPtr + t * cols, rows, cols, DType.Q4_0);
            }

            var gpuW = gpu.UploadRaw(weightBytes, TensorShape.D1(weightBytes.Length), DType.Q4_0);
            var gpuX = gpu.Upload(acts, TensorShape.D1(acts.Length));
            var gpuY = gpu.Allocate(TensorShape.D1((long)nTok * rows));

            gpu.MatMulBatchedMmq(gpuY, gpuW, gpuX, nTok, DType.Q4_0);
            gpu.Synchronize();

            var gpuOut = new float[nTok * rows];
            gpu.Download(gpuY, gpuOut);
            gpu.Free(gpuW);
            gpu.Free(gpuX);
            gpu.Free(gpuY);

            // Per-row magnitude scale for the relative bound (dot of ±1 acts over `cols`
            // Q4_0 weights has stddev ~ sqrt(cols)·scale).
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
                $"MMQ-Q4_0 rows={rows} cols={cols} nTok={nTok}: maxAbs={maxAbs:E2} refRms={refRms:E2} mismatches={mismatches}/{cpuOut.Length}");
            Assert.True(mismatches <= cpuOut.Length / 100 + 1,
                $"MMQ Q4_0 drifted from fp32 reference: {mismatches}/{cpuOut.Length} beyond 2% of RMS ({refRms:E3}), maxAbs={maxAbs:E3}.");
        }
    }
}
