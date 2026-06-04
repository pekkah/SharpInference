using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #129 per-kernel bit-exactness tests for the batched GPU-SLRU MoE prefill
/// kernels (<c>llm_moe_weighted_reduce</c>, <c>llm_scale_rows_inplace</c>). Each must be
/// <b>bit-identical</b> to the sequence of single-element primitives it replaced — not
/// just within tolerance. These run on any CUDA GPU with no model load, so they guard
/// the bit-parity claim that the heavy 22 GB GPU-SLRU oracle
/// (<c>BatchedTrunkGpuFfn_BitwiseMatchesSequential_GpuSlruMoe</c>) can only cover on the
/// dev box. Mirrors <see cref="CudaGdnBatchedTrunkTests"/>; silently skips without CUDA.
/// </summary>
public sealed unsafe class CudaMoeReduceKernelTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static float[] Rand(long n, Random rng)
    {
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static int Bits(float f) => BitConverter.SingleToInt32Bits(f);

    private static void AssertBitId(string label, float[] batched, float[] reference)
    {
        Assert.Equal(reference.Length, batched.Length);
        for (int i = 0; i < reference.Length; i++)
            if (Bits(batched[i]) != Bits(reference[i]))
                Assert.Fail($"{label}: index {i} batched={batched[i]} (0x{Bits(batched[i]):X8}) " +
                            $"!= sequential={reference[i]} (0x{Bits(reference[i]):X8}).");
    }

    /// <summary>
    /// <c>MoeWeightedReduce</c> must reproduce, byte-for-byte, the per-token sequential
    /// reduce it replaced: <c>Clear</c> → na× <c>AddScaledInPlace(acc, partial_k, w_k)</c>
    /// (routed slots in k=0..na-1 order) → <c>AddInPlace(acc, shared)</c> (shared LAST).
    /// The per-k FMA contraction and the final plain add must match the device primitives.
    /// </summary>
    [Fact]
    public void MoeWeightedReduce_BitwiseMatchesSequentialReduce()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int na = 8;       // active experts (Qwen3.6-A3B uses 8)
        const int embDim = 512; // GDN-MoE embedding dim
        // Includes N=1 (single token) and N=257 to stress the kernel's per-row indexing
        // and the (long)N*embDim → int launch-size cast at a realistic prefill N.
        foreach (int N in new[] { 1, 5, 33, 257 })
        {
            var rng = new Random(129 + N);
            var partial = Rand((long)N * na * embDim, rng);   // unweighted down partials
            var weights = Rand((long)N * na, rng);            // top-k weights
            var sharedScaled = Rand((long)N * embDim, rng);   // already scaled+rounded shared

            // ── Sequential reference: the exact primitive sequence the kernel replaced ──
            var gpuPartialRef = gpu.Upload(partial, TensorShape.D1((long)N * na * embDim));
            var gpuSharedRef = gpu.Upload(sharedScaled, TensorShape.D1((long)N * embDim));
            var acc = gpu.Allocate(TensorShape.D1(embDim));
            var refOut = new float[(long)N * embDim];
            for (int i = 0; i < N; i++)
            {
                gpu.Clear(acc);
                for (int k = 0; k < na; k++)
                {
                    var pv = gpu.View(gpuPartialRef, ((long)i * na + k) * embDim, embDim);
                    gpu.AddScaledInPlace(acc, pv, weights[(long)i * na + k]);
                    gpu.Free(pv);
                }
                var sv = gpu.View(gpuSharedRef, (long)i * embDim, embDim);
                gpu.AddInPlace(acc, sv);
                gpu.Free(sv);
                gpu.Synchronize();
                var ot = new float[embDim]; gpu.Download(acc, ot);
                Array.Copy(ot, 0, refOut, (long)i * embDim, embDim);
            }
            gpu.Free(acc); gpu.Free(gpuPartialRef); gpu.Free(gpuSharedRef);

            // ── Batched: one MoeWeightedReduce launch (shared buffer is in/out) ──
            var gpuPartialBat = gpu.Upload(partial, TensorShape.D1((long)N * na * embDim));
            var gpuWeights = gpu.Upload(weights, TensorShape.D1((long)N * na));
            var gpuSharedBat = gpu.Upload(sharedScaled, TensorShape.D1((long)N * embDim));
            gpu.MoeWeightedReduce(gpuPartialBat, gpuWeights, gpuSharedBat, N, na, embDim);
            gpu.Synchronize();
            var batOut = new float[(long)N * embDim]; gpu.Download(gpuSharedBat, batOut);
            gpu.Free(gpuPartialBat); gpu.Free(gpuWeights); gpu.Free(gpuSharedBat);

            AssertBitId($"moe reduce N={N}", batOut, refOut);
        }
    }

    /// <summary>
    /// <c>ScaleRowsInPlace</c> must be bit-identical to calling <c>ScaleInPlace</c> once
    /// per row with that row's scalar — a single float multiply per element, so the
    /// shared-expert sigmoid-gate scale is rounded to float before the Phase-3 plain add.
    /// </summary>
    [Fact]
    public void ScaleRowsInPlace_BitwiseMatchesPerRowScaleInPlace()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int cols = 512;
        foreach (int rows in new[] { 1, 7, 64, 300 })
        {
            var rng = new Random(529 + rows);
            var buf = Rand((long)rows * cols, rng);
            var scales = Rand(rows, rng);

            // ── Reference: per-row ScaleInPlace ──
            var gpuRef = gpu.Upload(buf, TensorShape.D1((long)rows * cols));
            for (int r = 0; r < rows; r++)
            {
                var rv = gpu.View(gpuRef, (long)r * cols, cols);
                gpu.ScaleInPlace(rv, scales[r]);
                gpu.Free(rv);
            }
            gpu.Synchronize();
            var refOut = new float[(long)rows * cols]; gpu.Download(gpuRef, refOut);
            gpu.Free(gpuRef);

            // ── Batched: one ScaleRowsInPlace launch ──
            var gpuBuf = gpu.Upload(buf, TensorShape.D1((long)rows * cols));
            var gpuScales = gpu.Upload(scales, TensorShape.D1(rows));
            gpu.ScaleRowsInPlace(gpuBuf, gpuScales, rows, cols);
            gpu.Synchronize();
            var batOut = new float[(long)rows * cols]; gpu.Download(gpuBuf, batOut);
            gpu.Free(gpuBuf); gpu.Free(gpuScales);

            AssertBitId($"scale_rows rows={rows}", batOut, refOut);
        }
    }
}
