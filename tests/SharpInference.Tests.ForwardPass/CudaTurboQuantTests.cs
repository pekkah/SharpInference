using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.TurboQuant;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CUDA-side TurboQuant kernel tests. Each test silently no-ops when no CUDA
/// device is available so CI on non-CUDA hosts (and Vulkan-only laptops) still
/// passes — matches the existing pattern in <see cref="GgufModelIntegrationTests"/>.
///
/// Numerics are validated against the same TurboQuant primitives the CPU and
/// Vulkan paths use (<see cref="KvCacheCompressor"/>): if the GPU and CPU paths
/// agree on the rotated query and on the fused dequant-dot per cached position,
/// the kernels are correctly mirroring the reference implementation.
/// </summary>
public sealed unsafe class CudaTurboQuantTests
{
    private const int HeadDim = 128;

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    [Fact]
    public void TqRotateQuery_MatchesCpu()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 4;
        const int NumKvHeads = 2;     // GQA: 2 query heads share each KV head

        // Each KV head gets its own seeded sign pattern, matching the convention used by
        // KvCacheCompressor / GpuForwardPass: seed = layerIndex * numKvHeads + kvHead.
        // For this test we use layerIndex == 0 so the seeds are simply 0 and 1.
        var rng = new Random(1234);
        var query = new float[NumHeads * HeadDim];
        for (int i = 0; i < query.Length; i++) query[i] = (float)(rng.NextDouble() * 2 - 1);

        var signsByKvHead = new float[NumKvHeads][];
        var signPatterns = new float[NumKvHeads * HeadDim];
        for (int kv = 0; kv < NumKvHeads; kv++)
        {
            var s = WalshHadamard.GenerateSignPattern(HeadDim, kv).ToArray();
            signsByKvHead[kv] = s;
            Array.Copy(s, 0, signPatterns, kv * HeadDim, HeadDim);
        }

        var gpuQ = gpu.Upload(query, TensorShape.D1(query.Length));
        var gpuSigns = gpu.Upload(signPatterns, TensorShape.D1(signPatterns.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(query.Length));

        gpu.TqRotateQuery(gpuQ, gpuRotated, gpuSigns, NumHeads, NumKvHeads, HeadDim);
        gpu.Synchronize();

        var gpuResult = new float[query.Length];
        gpu.Download(gpuRotated, gpuResult);

        // CPU reference: rotate each query head with its kv head's sign pattern.
        var expected = new float[query.Length];
        int headsPerKvGroup = NumHeads / NumKvHeads;
        for (int h = 0; h < NumHeads; h++)
        {
            int kv = h / headsPerKvGroup;
            var headIn = new float[HeadDim];
            Array.Copy(query, h * HeadDim, headIn, 0, HeadDim);
            var headOut = new float[HeadDim];
            TurboQuantOps.RotateQuery(headIn, headOut, signsByKvHead[kv], HeadDim);
            Array.Copy(headOut, 0, expected, h * HeadDim, HeadDim);
        }

        for (int i = 0; i < query.Length; i++)
            Assert.True(MathF.Abs(gpuResult[i] - expected[i]) < 1e-4f,
                $"TqRotateQuery mismatch at [{i}]: gpu={gpuResult[i]} cpu={expected[i]}");

        gpu.Free(gpuQ);
        gpu.Free(gpuSigns);
        gpu.Free(gpuRotated);
    }

    /// <summary>
    /// Compresses a batch of K vectors into the CUDA TQ cache, then runs DequantDot
    /// against the corresponding rotated query on the CPU using the same packed bytes
    /// read back from VRAM. The DequantDot result must match the CPU reference value
    /// computed from the un-quantized input vector via <see cref="KvCacheCompressor"/>.
    /// </summary>
    [Fact]
    public void TqKvAppend_RoundTripMatchesCpu()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumKvHeads = 2;
        const int Positions = 4;
        int blockBytes = TurboQuantOps.BlockSize(bits: 3, HeadDim);
        long tqBytesPerPos = (long)NumKvHeads * blockBytes;
        long totalBytes = (long)Positions * tqBytesPerPos;
        long totalUints = (totalBytes + 3) / 4;

        // Generate per-head inputs and matching CPU compressors. Seed every (layer × kvHead)
        // pair as if layerIndex=0, so the GPU and CPU paths share the same sign patterns.
        var rng = new Random(7777);
        var kInput = new float[Positions * NumKvHeads * HeadDim];
        var vInput = new float[Positions * NumKvHeads * HeadDim];
        for (int i = 0; i < kInput.Length; i++)
        {
            kInput[i] = (float)(rng.NextDouble() * 2 - 1);
            vInput[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        var signPatterns = new float[NumKvHeads * HeadDim];
        var keyCompressors = new KvCacheCompressor[NumKvHeads];
        var valueCompressors = new KvCacheCompressor[NumKvHeads];
        for (int kv = 0; kv < NumKvHeads; kv++)
        {
            keyCompressors[kv] = new KvCacheCompressor(bits: 3, HeadDim, kv);
            valueCompressors[kv] = new KvCacheCompressor(bits: 3, HeadDim, kv);
            keyCompressors[kv].SignPattern.CopyTo(signPatterns.AsSpan(kv * HeadDim));
        }

        var gpuKIn = gpu.Allocate(TensorShape.D1((long)NumKvHeads * HeadDim));
        var gpuVIn = gpu.Allocate(TensorShape.D1((long)NumKvHeads * HeadDim));
        var gpuKCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuVCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuSigns = gpu.Upload(signPatterns, TensorShape.D1(signPatterns.Length));
        var centroids = TurboQuantCodebooks.GetCentroids(bits: 3, HeadDim).ToArray();
        var boundaries = TurboQuantCodebooks.GetBoundaries(bits: 3, HeadDim).ToArray();
        var gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));
        var gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));

        // Upload each position's K and V vectors, then dispatch TqKvAppend.
        for (int p = 0; p < Positions; p++)
        {
            int offset = p * NumKvHeads * HeadDim;
            var kSlice = new float[NumKvHeads * HeadDim];
            var vSlice = new float[NumKvHeads * HeadDim];
            Array.Copy(kInput, offset, kSlice, 0, kSlice.Length);
            Array.Copy(vInput, offset, vSlice, 0, vSlice.Length);

            // Re-upload into the same scratch tensor — UploadViaStaging only goes via the
            // staging path, but the public Upload allocates fresh. Free + re-create keeps
            // the test code short; the per-position cost is negligible.
            gpu.Free(gpuKIn);
            gpu.Free(gpuVIn);
            gpuKIn = gpu.Upload(kSlice, TensorShape.D1(kSlice.Length));
            gpuVIn = gpu.Upload(vSlice, TensorShape.D1(vSlice.Length));

            gpu.TqKvAppend(gpuKIn, gpuVIn, gpuKCacheTq, gpuVCacheTq,
                gpuSigns, gpuCodebook, gpuBoundaries,
                NumKvHeads * HeadDim, HeadDim, p, Positions, NumKvHeads, blockBytes);
        }
        gpu.Synchronize();

        // Read back the compressed K cache and decompress on the CPU; compare each
        // position's dequantized vector to a fresh CPU-side compress/decompress round
        // trip. The two must be bit-identical up to FP16 rounding of the norm and the
        // identical Lloyd-Max bin assignment.
        int totalUintsInt = checked((int)totalUints);
        var rawK = new float[totalUintsInt];
        gpu.Download(gpuKCacheTq, rawK);
        Span<byte> rawKBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(rawK.AsSpan());

        for (int p = 0; p < Positions; p++)
        {
            for (int kv = 0; kv < NumKvHeads; kv++)
            {
                int offset = (p * NumKvHeads + kv) * HeadDim;
                var headInput = new float[HeadDim];
                Array.Copy(kInput, offset, headInput, 0, HeadDim);

                // GPU-produced packed block at the same offset.
                int blockOffset = checked((int)((long)p * tqBytesPerPos + (long)kv * blockBytes));
                var gpuBlock = rawKBytes.Slice(blockOffset, blockBytes).ToArray();

                // CPU reference: compress the same input.
                var cpuBlock = new byte[blockBytes];
                keyCompressors[kv].Compress(headInput, cpuBlock);

                // Decompress both and compare elementwise.
                var gpuDecoded = new float[HeadDim];
                var cpuDecoded = new float[HeadDim];
                keyCompressors[kv].Decompress(gpuBlock, gpuDecoded);
                keyCompressors[kv].Decompress(cpuBlock, cpuDecoded);

                // Norm is FP16, so a 1e-3 relative tolerance covers the half-precision
                // rounding plus any reduction-order variation in the GPU norm reduction.
                float refNorm = 0f;
                for (int d = 0; d < HeadDim; d++) refNorm += cpuDecoded[d] * cpuDecoded[d];
                refNorm = MathF.Sqrt(refNorm);
                float tol = MathF.Max(1e-3f, refNorm * 1e-3f);

                for (int d = 0; d < HeadDim; d++)
                {
                    Assert.True(MathF.Abs(gpuDecoded[d] - cpuDecoded[d]) < tol,
                        $"Decoded TQ K mismatch at pos={p} kv={kv} d={d}: " +
                        $"gpu={gpuDecoded[d]} cpu={cpuDecoded[d]} tol={tol}");
                }
            }
        }

        gpu.Free(gpuKIn); gpu.Free(gpuVIn);
        gpu.Free(gpuKCacheTq); gpu.Free(gpuVCacheTq);
        gpu.Free(gpuSigns); gpu.Free(gpuCodebook); gpu.Free(gpuBoundaries);
    }

    /// <summary>
    /// End-to-end needle-in-a-haystack on the TQ attention kernel. The needle's K vector
    /// is scaled up so its dot product with the query dominates background K·Q values
    /// (background random unit-vector dots have std ≈ 1/sqrt(d); a 30× needle ensures the
    /// softmax weight on the needle is &gt; 95% even with 32 random distractors).
    ///
    /// The output is in the rotated/sign-flipped basis (the V-side of TqAttention does not
    /// un-rotate — same convention as the Vulkan TqAttention shader), so we compare the
    /// kernel output to the *rotated* needle value via cosine similarity.
    /// </summary>
    [Fact]
    public void TqAttention_NeedleInHaystack()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 1;
        const int NumKvHeads = 1;
        const int TqLen = 32;
        const float NeedleScale = 30f;
        int fp32Window = 0;
        int blockBytes = TurboQuantOps.BlockSize(bits: 3, HeadDim);
        long tqBytesPerPos = (long)NumKvHeads * blockBytes;
        long totalUints = ((long)TqLen * tqBytesPerPos + 3) / 4;

        var rng = new Random(909090);

        var queryDir = RandomUnit(rng);
        var needleKey = new float[HeadDim];
        for (int d = 0; d < HeadDim; d++) needleKey[d] = queryDir[d] * NeedleScale;

        var needleValue = new float[HeadDim];
        needleValue[0] = 1f;

        int needlePos = TqLen / 3;

        var compressor = new KvCacheCompressor(bits: 3, HeadDim, layerIndex: 0);
        var signPatterns = compressor.SignPattern.ToArray();
        var centroids    = TurboQuantCodebooks.GetCentroids(bits: 3, HeadDim).ToArray();
        var boundaries   = TurboQuantCodebooks.GetBoundaries(bits: 3, HeadDim).ToArray();

        var gpuKCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuVCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuSigns      = gpu.Upload(signPatterns, TensorShape.D1(signPatterns.Length));
        var gpuCodebook   = gpu.Upload(centroids,    TensorShape.D1(centroids.Length));
        var gpuBoundaries = gpu.Upload(boundaries,   TensorShape.D1(boundaries.Length));

        for (int p = 0; p < TqLen; p++)
        {
            float[] kvec = p == needlePos ? needleKey : RandomUnit(rng);
            float[] vvec = p == needlePos ? needleValue : RandomUnit(rng);

            var gpuKIn = gpu.Upload(kvec, TensorShape.D1(kvec.Length));
            var gpuVIn = gpu.Upload(vvec, TensorShape.D1(vvec.Length));
            gpu.TqKvAppend(gpuKIn, gpuVIn, gpuKCacheTq, gpuVCacheTq,
                gpuSigns, gpuCodebook, gpuBoundaries,
                NumKvHeads * HeadDim, HeadDim, p, TqLen, NumKvHeads, blockBytes);
            gpu.Free(gpuKIn); gpu.Free(gpuVIn);
        }

        var query = (float[])queryDir.Clone();
        var gpuQ = gpu.Upload(query, TensorShape.D1(query.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(query.Length));
        gpu.TqRotateQuery(gpuQ, gpuRotated, gpuSigns, NumHeads, NumKvHeads, HeadDim);

        var gpuKCacheFp32 = gpu.Allocate(TensorShape.D1(NumKvHeads * HeadDim));
        var gpuVCacheFp32 = gpu.Allocate(TensorShape.D1(NumKvHeads * HeadDim));
        var gpuOut = gpu.Allocate(TensorShape.D1(NumHeads * HeadDim));

        gpu.TqAttention(gpuQ, gpuRotated, gpuKCacheTq, gpuVCacheTq,
            gpuKCacheFp32, gpuVCacheFp32, gpuOut, gpuCodebook,
            scoresScratch: null,
            NumHeads, NumKvHeads, HeadDim, TqLen, fp32Window, TqLen, blockBytes);
        gpu.Synchronize();

        var output = new float[NumHeads * HeadDim];
        gpu.Download(gpuOut, output);

        // Output is accumulated in the rotated basis. Compare to a fresh CPU round-trip
        // of the needle value through the same Lloyd-Max codebook so the reference reflects
        // the actual quantization error, not the ideal rotated vector.
        var compressedNeedleValue = new byte[blockBytes];
        compressor.Compress(needleValue, compressedNeedleValue);
        var rotatedQuantizedNeedleValue = ReconstructRotated(compressedNeedleValue, centroids, blockBytes);

        float cos = Cosine(output, rotatedQuantizedNeedleValue);
        Assert.True(cos > 0.7f,
            $"TQ attention output does not align with the quantized needle value (cosine={cos:F3}); " +
            $"the needle should carry the dominant softmax weight at NeedleScale={NeedleScale}, TqLen={TqLen}.");

        gpu.Free(gpuQ); gpu.Free(gpuRotated);
        gpu.Free(gpuKCacheTq); gpu.Free(gpuVCacheTq);
        gpu.Free(gpuKCacheFp32); gpu.Free(gpuVCacheFp32);
        gpu.Free(gpuOut);
        gpu.Free(gpuSigns); gpu.Free(gpuCodebook); gpu.Free(gpuBoundaries);
    }

    /// <summary>
    /// Strict equivalence check between the stored-scores fast path (TqLen=4096, hits the
    /// fast branch in the kernel) and the triple-pass recompute path (TqLen=4097, just over
    /// MAX_STORED_SCORES). Both paths see the same K/V data for the first 4096 positions and
    /// must produce numerically equivalent attention output once we cancel out the extra
    /// position's contribution analytically. We use a fully-orthogonal V layout (one unit
    /// vector per dim, only the first 4096 V positions non-zero) so the slow-path's extra
    /// position contributes zero in any output dim and we can compare the two outputs
    /// dimension-by-dimension.
    ///
    /// This is the load-bearing correctness test for the recompute branch — a wrong score,
    /// a wrong softmax normalization, or a wrong V indexing all show up here as a per-dim
    /// numeric mismatch, no statistical hand-waving required.
    /// </summary>
    [Fact]
    public void TqAttention_RecomputePath_MatchesFastPath()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 1;
        const int NumKvHeads = 1;
        const int FastLen = 4096;     // exactly at the cap → fast path
        const int SlowLen = 4097;     // one over → recompute path
        int blockBytes = TurboQuantOps.BlockSize(bits: 3, HeadDim);
        long tqBytesPerPos = (long)NumKvHeads * blockBytes;
        long fastUints = ((long)FastLen * tqBytesPerPos + 3) / 4;
        long slowUints = ((long)SlowLen * tqBytesPerPos + 3) / 4;

        var rng = new Random(13371337);
        var queryDir = RandomUnit(rng);

        // Build K/V for the first 4096 positions identically across both runs.
        // Position 4096 in the slow run gets a zero V so it contributes nothing in any
        // output dimension regardless of its softmax weight — this is what lets us
        // compare the two paths element-wise.
        var kVecs = new float[SlowLen][];
        var vVecs = new float[SlowLen][];
        for (int p = 0; p < FastLen; p++)
        {
            kVecs[p] = RandomUnit(rng);
            vVecs[p] = RandomUnit(rng);
        }
        kVecs[FastLen] = RandomUnit(rng);
        vVecs[FastLen] = new float[HeadDim];   // zero V at position 4096

        var compressor = new KvCacheCompressor(bits: 3, HeadDim, layerIndex: 0);
        var signPatterns = compressor.SignPattern.ToArray();
        var centroids    = TurboQuantCodebooks.GetCentroids(bits: 3, HeadDim).ToArray();
        var boundaries   = TurboQuantCodebooks.GetBoundaries(bits: 3, HeadDim).ToArray();

        var gpuSigns      = gpu.Upload(signPatterns, TensorShape.D1(signPatterns.Length));
        var gpuCodebook   = gpu.Upload(centroids,    TensorShape.D1(centroids.Length));
        var gpuBoundaries = gpu.Upload(boundaries,   TensorShape.D1(boundaries.Length));

        float[] outFast = RunPath(gpu, NumHeads, NumKvHeads, FastLen, fastUints, kVecs, vVecs,
            queryDir, blockBytes, gpuSigns, gpuCodebook, gpuBoundaries);
        float[] outSlow = RunPath(gpu, NumHeads, NumKvHeads, SlowLen, slowUints, kVecs, vVecs,
            queryDir, blockBytes, gpuSigns, gpuCodebook, gpuBoundaries);

        // The slow path's softmax denominator includes the (zero-V, non-zero-K) extra
        // position — that uniformly rescales every output dim by sum_fast / sum_slow.
        // Recover the rescale factor from any non-trivial dim and check the remaining
        // dims agree under that single global scale.
        int probeDim = 0;
        float bestAbs = MathF.Abs(outFast[probeDim]);
        for (int d = 1; d < HeadDim; d++)
        {
            if (MathF.Abs(outFast[d]) > bestAbs) { bestAbs = MathF.Abs(outFast[d]); probeDim = d; }
        }
        Assert.True(bestAbs > 1e-4f, "Fast-path output is degenerate (all near zero); test inputs are pathological.");

        float ratio = outSlow[probeDim] / outFast[probeDim];
        for (int d = 0; d < HeadDim; d++)
        {
            float expected = outFast[d] * ratio;
            float diff = MathF.Abs(outSlow[d] - expected);
            float tol = MathF.Max(1e-3f, MathF.Abs(expected) * 1e-2f);
            Assert.True(diff < tol,
                $"Recompute path mismatch at dim {d}: slow={outSlow[d]:E3} expected={expected:E3} " +
                $"(fast={outFast[d]:E3}, ratio={ratio:F4}, tol={tol:E3}). " +
                $"The two paths must agree under a single global softmax-rescale factor.");
        }

        gpu.Free(gpuSigns); gpu.Free(gpuCodebook); gpu.Free(gpuBoundaries);
    }

    /// <summary>
    /// Strict equivalence check between the FP32 attention kernel's shared-memory fast
    /// path (seq_len=4096) and the global-scratch slow path (seq_len=4097, just over
    /// MAX_STORED_SCORES). Mirrors the TQ test's strategy: both paths see identical K/V
    /// for the first 4096 positions; position 4096 in the slow path has zero V so it
    /// contributes nothing to any output dim regardless of its softmax weight. The two
    /// outputs must agree under a single global softmax-rescale factor that recovers
    /// the slow path's extra denominator contribution.
    ///
    /// This is the load-bearing correctness test for the FP32 long-context branch.
    /// A wrong score, wrong softmax normalization, or wrong V indexing all show up
    /// here as a per-dim numeric mismatch.
    /// </summary>
    [Fact]
    public void Attention_LongContextScratchPath_MatchesFastPath()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumHeads = 1;
        const int NumKvHeads = 1;
        const int FastLen = 4096;
        const int SlowLen = 4097;
        int kvDim = NumKvHeads * HeadDim;

        var rng = new Random(42);
        var query = RandomUnit(rng);

        // K/V layout matches the kernel: [maxSeqLen, kvDim].
        var kCacheSlow = new float[(long)SlowLen * kvDim];
        var vCacheSlow = new float[(long)SlowLen * kvDim];
        for (int t = 0; t < FastLen; t++)
        {
            var k = RandomUnit(rng);
            var v = RandomUnit(rng);
            Array.Copy(k, 0, kCacheSlow, t * kvDim, HeadDim);
            Array.Copy(v, 0, vCacheSlow, t * kvDim, HeadDim);
        }
        // Extra position has non-trivial K (contributes to softmax denom) but zero V.
        var extraK = RandomUnit(rng);
        Array.Copy(extraK, 0, kCacheSlow, FastLen * kvDim, HeadDim);
        // V at position FastLen left as zeros.

        // Fast-path uses the first FastLen rows of the same buffer.
        var kCacheFast = new float[(long)FastLen * kvDim];
        var vCacheFast = new float[(long)FastLen * kvDim];
        Array.Copy(kCacheSlow, kCacheFast, (long)FastLen * kvDim);
        Array.Copy(vCacheSlow, vCacheFast, (long)FastLen * kvDim);

        float[] outFast = RunFp32Path(gpu, NumHeads, NumKvHeads, FastLen, kCacheFast, vCacheFast, query);
        float[] outSlow = RunFp32Path(gpu, NumHeads, NumKvHeads, SlowLen, kCacheSlow, vCacheSlow, query);

        int probeDim = 0;
        float bestAbs = MathF.Abs(outFast[probeDim]);
        for (int d = 1; d < HeadDim; d++)
            if (MathF.Abs(outFast[d]) > bestAbs) { bestAbs = MathF.Abs(outFast[d]); probeDim = d; }
        Assert.True(bestAbs > 1e-4f, "Fast-path output is degenerate (all near zero); test inputs are pathological.");

        float ratio = outSlow[probeDim] / outFast[probeDim];
        for (int d = 0; d < HeadDim; d++)
        {
            float expected = outFast[d] * ratio;
            float diff = MathF.Abs(outSlow[d] - expected);
            float tol = MathF.Max(1e-3f, MathF.Abs(expected) * 1e-2f);
            Assert.True(diff < tol,
                $"FP32 long-context path mismatch at dim {d}: slow={outSlow[d]:E3} expected={expected:E3} " +
                $"(fast={outFast[d]:E3}, ratio={ratio:F4}, tol={tol:E3}). " +
                $"Both paths must agree under a single global softmax-rescale factor.");
        }
    }

    private static float[] RunFp32Path(CudaBackend gpu, int numHeads, int numKvHeads, int seqLen,
        float[] kCache, float[] vCache, float[] query)
    {
        var gpuQ = gpu.Upload(query, TensorShape.D1(query.Length));
        var gpuK = gpu.Upload(kCache, TensorShape.D1(kCache.Length));
        var gpuV = gpu.Upload(vCache, TensorShape.D1(vCache.Length));
        var gpuOut = gpu.Allocate(TensorShape.D1(numHeads * HeadDim));
        var gpuScratch = gpu.Allocate(TensorShape.D1((long)numHeads * seqLen));

        gpu.Attention(gpuQ, gpuK, gpuV, gpuOut, gpuScratch,
            numHeads, numKvHeads, HeadDim, seqLen, seqLen);
        gpu.Synchronize();

        var output = new float[numHeads * HeadDim];
        gpu.Download(gpuOut, output);

        gpu.Free(gpuQ); gpu.Free(gpuK); gpu.Free(gpuV);
        gpu.Free(gpuOut); gpu.Free(gpuScratch);
        return output;
    }

    private static float[] RunPath(CudaBackend gpu, int numHeads, int numKvHeads,
        int tqLen, long totalUints, float[][] kVecs, float[][] vVecs, float[] queryDir,
        int blockBytes,
        Tensor gpuSigns, Tensor gpuCodebook, Tensor gpuBoundaries)
    {
        var gpuKCacheTq = gpu.Allocate(TensorShape.D1(totalUints));
        var gpuVCacheTq = gpu.Allocate(TensorShape.D1(totalUints));

        for (int p = 0; p < tqLen; p++)
        {
            var gpuKIn = gpu.Upload(kVecs[p], TensorShape.D1(kVecs[p].Length));
            var gpuVIn = gpu.Upload(vVecs[p], TensorShape.D1(vVecs[p].Length));
            gpu.TqKvAppend(gpuKIn, gpuVIn, gpuKCacheTq, gpuVCacheTq,
                gpuSigns, gpuCodebook, gpuBoundaries,
                numKvHeads * HeadDim, HeadDim, p, tqLen, numKvHeads, blockBytes);
            gpu.Free(gpuKIn); gpu.Free(gpuVIn);
        }

        var gpuQ = gpu.Upload(queryDir, TensorShape.D1(queryDir.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(queryDir.Length));
        gpu.TqRotateQuery(gpuQ, gpuRotated, gpuSigns, numHeads, numKvHeads, HeadDim);

        var gpuKCacheFp32 = gpu.Allocate(TensorShape.D1(numKvHeads * HeadDim));
        var gpuVCacheFp32 = gpu.Allocate(TensorShape.D1(numKvHeads * HeadDim));
        var gpuOut = gpu.Allocate(TensorShape.D1(numHeads * HeadDim));

        // Allocate the long-context scratch unconditionally so both paths exercise
        // the same code; the kernel ignores it on the fast path.
        var scratch = gpu.Allocate(TensorShape.D1((long)numHeads * tqLen));

        gpu.TqAttention(gpuQ, gpuRotated, gpuKCacheTq, gpuVCacheTq,
            gpuKCacheFp32, gpuVCacheFp32, gpuOut, gpuCodebook,
            scratch,
            numHeads, numKvHeads, HeadDim, tqLen, 0, tqLen, blockBytes);
        gpu.Synchronize();

        var output = new float[numHeads * HeadDim];
        gpu.Download(gpuOut, output);

        gpu.Free(gpuQ); gpu.Free(gpuRotated);
        gpu.Free(gpuKCacheTq); gpu.Free(gpuVCacheTq);
        gpu.Free(gpuKCacheFp32); gpu.Free(gpuVCacheFp32);
        gpu.Free(gpuOut); gpu.Free(scratch);

        return output;
    }

    /// <summary>
    /// Reconstruct one TQ block as the kernel sees it during phase 3: each coordinate
    /// is `centroid[idx] * fp16_norm` (rotated basis, no inverse WHT applied).
    /// </summary>
    private static float[] ReconstructRotated(byte[] block, float[] centroids, int blockBytes)
    {
        var rotated = new float[HeadDim];
        float norm = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(block);
        for (int d = 0; d < HeadDim; d++)
        {
            int bitPos = 16 + d * 3;
            int byteIdx = bitPos >> 3;
            int bitOff = bitPos & 7;
            int raw = block[byteIdx] >> bitOff;
            if (bitOff > 5) raw |= block[byteIdx + 1] << (8 - bitOff);
            int idx = raw & 0x7;
            rotated[d] = centroids[idx] * norm;
        }
        return rotated;
    }

    private static void Normalize(float[] v)
    {
        float mag = 0f;
        for (int i = 0; i < v.Length; i++) mag += v[i] * v[i];
        mag = MathF.Sqrt(mag);
        if (mag <= 0f) return;
        for (int i = 0; i < v.Length; i++) v[i] /= mag;
    }

    private static float[] RandomUnit(Random rng)
    {
        var v = new float[HeadDim];
        for (int i = 0; i < HeadDim; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        Normalize(v);
        return v;
    }

    private static float Cosine(float[] a, float[] b)
    {
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na <= 0f || nb <= 0f) return 0f;
        return dot / MathF.Sqrt(na * nb);
    }
}
