using SharpInference.Core;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #197 bit-exactness tests for the ragged-batched decode kernels
/// (<see cref="CudaBackend.RoPEBatchedRagged"/>, <see cref="CudaBackend.KvAppendBatchedRagged"/>,
/// <see cref="CudaBackend.AttentionBatchedRagged"/>, <see cref="CudaBackend.AddBiasBatched"/>).
/// The ragged kernels keep the per-element / per-(head, position) computation chain of their
/// single-sequence counterparts and only batch the row/cache indirection onto the grid, so each
/// sequence's result must be <b>bit-identical</b> to the matching sequential per-token kernel
/// call — the independent per-token reference, NOT a path built to mirror the ragged one
/// (a path validated only against its own mirror isn't validated).
///
/// Batch sizes cross the by-value struct parameter capacity (16) so the host-side chunking into
/// multiple launches is exercised (17, 20), plus N=1 and odd sizes. Positions are deliberately
/// ragged (every sequence at a different position). Cache buffers are compared only at the rows
/// the appends wrote (allocations are not zero-initialized).
///
/// Silently skips on hosts without CUDA, mirroring the other Cuda* test files.
/// </summary>
public sealed class CudaRaggedDecodeKernelTests
{
    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static readonly int[] BatchSizes = { 1, 2, 3, 5, 8, 16, 17, 20 };

    private static float[] RandomFloats(int n, Random rng)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Ragged positions: distinct, non-monotonic, includes 0.</summary>
    private static int[] RaggedPositions(int n, int maxSeqLen, Random rng)
    {
        var pos = new int[n];
        for (int i = 0; i < n; i++)
            pos[i] = i == 0 ? 0 : rng.Next(maxSeqLen);
        return pos;
    }

    private static void AssertBitIdentical(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail(
                    $"{label}: element {i} ragged={actual[i]} (0x{BitConverter.SingleToInt32Bits(actual[i]):X8}) " +
                    $"!= sequential={expected[i]} (0x{BitConverter.SingleToInt32Bits(expected[i]):X8}). " +
                    "Ragged kernels must be bit-identical to the sequential per-token kernels.");
    }

    // ── RoPE ────────────────────────────────────────────────────────────────

    private void RopeCase(bool neox)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int numHeads = 8, headDim = 64;
        int rowDim = numHeads * headDim;
        const float theta = 1_000_000f;   // Qwen3's rope_theta scale

        foreach (int n in BatchSizes)
        {
            var rng = new Random(197_001 + n);
            int[] positions = RaggedPositions(n, 5000, rng);
            float[] xAll = RandomFloats(n * rowDim, rng);

            // Ragged: one call over all rows.
            var gpuXAll = gpu.Upload(xAll, TensorShape.D1((long)n * rowDim));
            gpu.RoPEBatchedRagged(gpuXAll, positions, numHeads, headDim, theta, neox);
            gpu.Synchronize();
            var ragged = new float[n * rowDim];
            gpu.Download(gpuXAll, ragged);
            gpu.Free(gpuXAll);

            // Sequential per-token reference: each row through the independent RoPE kernel.
            for (int t = 0; t < n; t++)
            {
                var row = new float[rowDim];
                Array.Copy(xAll, (long)t * rowDim, row, 0, rowDim);
                var gpuRow = gpu.Upload(row, TensorShape.D1(rowDim));
                gpu.RoPE(gpuRow, positions[t], headDim, theta, neox);
                gpu.Synchronize();
                var expected = new float[rowDim];
                gpu.Download(gpuRow, expected);
                gpu.Free(gpuRow);

                var actual = new float[rowDim];
                Array.Copy(ragged, (long)t * rowDim, actual, 0, rowDim);
                AssertBitIdentical($"RoPE(neox={neox}) N={n} seq={t} pos={positions[t]}", expected, actual);
            }
        }
    }

    [Fact]
    public void Rope_Neox_Ragged_BitwiseMatchesSequential() => RopeCase(neox: true);

    [Fact]
    public void Rope_Interleaved_Ragged_BitwiseMatchesSequential() => RopeCase(neox: false);

    // ── KV append ───────────────────────────────────────────────────────────

    /// <summary>Bytes one cache row occupies for the given KV dtype (fp32/bf16/q8_0).</summary>
    private static int RowBytes(int kvDim, DType dtype) => dtype switch
    {
        DType.Float32  => kvDim * 4,
        DType.BFloat16 => kvDim * 2,
        DType.Q8_0     => kvDim / 32 * 34,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    /// <summary>Download a cache tensor's raw storage as uint words for bit comparison.</summary>
    private static uint[] DownloadWords(CudaBackend gpu, Tensor cache, int totalBytes)
    {
        var f = new float[totalBytes / 4];
        gpu.Download(cache, f);
        var w = new uint[f.Length];
        for (int i = 0; i < f.Length; i++) w[i] = (uint)BitConverter.SingleToInt32Bits(f[i]);
        return w;
    }

    private void KvAppendCase(DType dtype)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int kvDim = 128, maxSeqLen = 64;
        int rowBytes = RowBytes(kvDim, dtype);
        int totalBytes = maxSeqLen * rowBytes;
        int rowWords = rowBytes / 4;

        foreach (int n in BatchSizes)
        {
            var rng = new Random(197_100 + n);
            int[] positions = RaggedPositions(n, maxSeqLen, rng);
            float[] kAll = RandomFloats(n * kvDim, rng);
            float[] vAll = RandomFloats(n * kvDim, rng);

            var refK = new Tensor[n]; var refV = new Tensor[n];
            var ragK = new Tensor[n]; var ragV = new Tensor[n];
            for (int t = 0; t < n; t++)
            {
                refK[t] = gpu.Allocate(TensorShape.D1(maxSeqLen * kvDim), dtype);
                refV[t] = gpu.Allocate(TensorShape.D1(maxSeqLen * kvDim), dtype);
                ragK[t] = gpu.Allocate(TensorShape.D1(maxSeqLen * kvDim), dtype);
                ragV[t] = gpu.Allocate(TensorShape.D1(maxSeqLen * kvDim), dtype);
            }
            try
            {
                // Ragged: one call appends every row into its own cache pair.
                var gpuKAll = gpu.Upload(kAll, TensorShape.D1((long)n * kvDim));
                var gpuVAll = gpu.Upload(vAll, TensorShape.D1((long)n * kvDim));
                gpu.KvAppendBatchedRagged(gpuKAll, gpuVAll, ragK, ragV, positions, kvDim, maxSeqLen, dtype);
                gpu.Synchronize();
                gpu.Free(gpuKAll); gpu.Free(gpuVAll);

                // Sequential per-token reference appends into the independent cache set.
                for (int t = 0; t < n; t++)
                {
                    var kRow = new float[kvDim]; var vRow = new float[kvDim];
                    Array.Copy(kAll, (long)t * kvDim, kRow, 0, kvDim);
                    Array.Copy(vAll, (long)t * kvDim, vRow, 0, kvDim);
                    var gpuKRow = gpu.Upload(kRow, TensorShape.D1(kvDim));
                    var gpuVRow = gpu.Upload(vRow, TensorShape.D1(kvDim));
                    switch (dtype)
                    {
                        case DType.Float32:
                            gpu.KvAppend(gpuKRow, gpuVRow, refK[t], refV[t], kvDim, positions[t], maxSeqLen);
                            break;
                        case DType.BFloat16:
                            gpu.KvAppendBf16(gpuKRow, gpuVRow, refK[t], refV[t], kvDim, positions[t], maxSeqLen);
                            break;
                        default:
                            gpu.KvAppendQ8_0(gpuKRow, gpuVRow, refK[t], refV[t], kvDim, positions[t], maxSeqLen);
                            break;
                    }
                    gpu.Synchronize();
                    gpu.Free(gpuKRow); gpu.Free(gpuVRow);
                }

                // Compare only the written row (allocations are not zeroed; untouched rows
                // hold unrelated garbage that differs between the two allocation sets).
                for (int t = 0; t < n; t++)
                {
                    int wordOff = positions[t] * rowWords;
                    foreach ((Tensor reference, Tensor candidate, string which) in
                             new[] { (refK[t], ragK[t], "K"), (refV[t], ragV[t], "V") })
                    {
                        uint[] expected = DownloadWords(gpu, reference, totalBytes);
                        uint[] actual   = DownloadWords(gpu, candidate, totalBytes);
                        for (int w = 0; w < rowWords; w++)
                            if (expected[wordOff + w] != actual[wordOff + w])
                                Assert.Fail(
                                    $"KvAppend({dtype}) N={n} seq={t} pos={positions[t]} {which}: word {w} " +
                                    $"ragged=0x{actual[wordOff + w]:X8} != sequential=0x{expected[wordOff + w]:X8}.");
                    }
                }
            }
            finally
            {
                for (int t = 0; t < n; t++)
                {
                    gpu.Free(refK[t]); gpu.Free(refV[t]);
                    gpu.Free(ragK[t]); gpu.Free(ragV[t]);
                }
            }
        }
    }

    [Fact]
    public void KvAppend_Ragged_F32_BitwiseMatchesSequential() => KvAppendCase(DType.Float32);

    [Fact]
    public void KvAppend_Ragged_Bf16_BitwiseMatchesSequential() => KvAppendCase(DType.BFloat16);

    [Fact]
    public void KvAppend_Ragged_Q8_0_BitwiseMatchesSequential() => KvAppendCase(DType.Q8_0);

    // ── Single-query attention ──────────────────────────────────────────────

    private static ushort HalfToUshort(Half h) => BitConverter.ToUInt16(BitConverter.GetBytes(h), 0);

    /// <summary>Upload a random cache of <paramref name="elems"/> KV elements in the given
    /// dtype. The same tensor is read by both the ragged and the per-token kernels, so
    /// contents only need to be valid for the dtype, not derived from appends.</summary>
    private static Tensor UploadRandomCache(CudaBackend gpu, int elems, DType dtype, Random rng)
    {
        switch (dtype)
        {
            case DType.Float32:
                return gpu.Upload(RandomFloats(elems, rng), TensorShape.D1(elems));
            case DType.BFloat16:
            {
                var u = new ushort[elems];
                for (int i = 0; i < elems; i++)
                    u[i] = (ushort)((uint)BitConverter.SingleToInt32Bits(
                        (float)(rng.NextDouble() * 2 - 1)) >> 16);
                return gpu.UploadBf16(u, TensorShape.D1(elems));
            }
            default:
            {
                // block_q8_0: fp16 scale + 32 int8 quants per 32 elements.
                int blocks = elems / 32;
                var bytes = new byte[blocks * 34];
                for (int b = 0; b < blocks; b++)
                {
                    int off = b * 34;
                    ushort d = HalfToUshort((Half)(rng.NextDouble() * 0.05 + 0.005));
                    bytes[off] = (byte)(d & 0xFF); bytes[off + 1] = (byte)(d >> 8);
                    for (int i = 0; i < 32; i++) bytes[off + 2 + i] = (byte)rng.Next(256);
                }
                return gpu.UploadRaw(bytes, TensorShape.D1(bytes.Length), DType.Q8_0);
            }
        }
    }

    private void AttentionCase(DType dtype, int maxSeqLen, int[] seqLens, bool withScratch)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int numHeads = 8, numKvHeads = 2, headDim = 64;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        int n = seqLens.Length;
        int cacheElems = maxSeqLen * kvDim;
        var rng = new Random(197_200 + n + maxSeqLen);

        var positions = new int[n];
        for (int t = 0; t < n; t++) positions[t] = seqLens[t] - 1;

        var kCaches = new Tensor[n]; var vCaches = new Tensor[n];
        for (int t = 0; t < n; t++)
        {
            kCaches[t] = UploadRandomCache(gpu, cacheElems, dtype, rng);
            vCaches[t] = UploadRandomCache(gpu, cacheElems, dtype, rng);
        }
        float[] qAll = RandomFloats(n * qDim, rng);
        var gpuQAll = gpu.Upload(qAll, TensorShape.D1((long)n * qDim));
        var gpuOutAll = gpu.Allocate(TensorShape.D1((long)n * qDim));
        Tensor? raggedScratch = withScratch
            ? gpu.Allocate(TensorShape.D1((long)n * numHeads * maxSeqLen)) : null;
        Tensor? refScratch = withScratch
            ? gpu.Allocate(TensorShape.D1((long)numHeads * maxSeqLen)) : null;

        try
        {
            // Ragged: one launch, all sequences' single-query attentions concurrent.
            gpu.AttentionBatchedRagged(gpuQAll, kCaches, vCaches, gpuOutAll, raggedScratch,
                numHeads, numKvHeads, headDim, positions, maxSeqLen, -1f, dtype);
            gpu.Synchronize();
            var ragged = new float[n * qDim];
            gpu.Download(gpuOutAll, ragged);

            // Sequential per-token reference against the same caches.
            for (int t = 0; t < n; t++)
            {
                var qRow = new float[qDim];
                Array.Copy(qAll, (long)t * qDim, qRow, 0, qDim);
                var gpuQRow = gpu.Upload(qRow, TensorShape.D1(qDim));
                var gpuOutRow = gpu.Allocate(TensorShape.D1(qDim));
                switch (dtype)
                {
                    case DType.Float32:
                        gpu.Attention(gpuQRow, kCaches[t], vCaches[t], gpuOutRow, refScratch,
                            numHeads, numKvHeads, headDim, seqLens[t], maxSeqLen);
                        break;
                    case DType.BFloat16:
                        gpu.AttentionBf16(gpuQRow, kCaches[t], vCaches[t], gpuOutRow, refScratch,
                            numHeads, numKvHeads, headDim, seqLens[t], maxSeqLen);
                        break;
                    default:
                        gpu.AttentionQ8_0(gpuQRow, kCaches[t], vCaches[t], gpuOutRow, refScratch,
                            numHeads, numKvHeads, headDim, seqLens[t], maxSeqLen);
                        break;
                }
                gpu.Synchronize();
                var expected = new float[qDim];
                gpu.Download(gpuOutRow, expected);
                gpu.Free(gpuQRow); gpu.Free(gpuOutRow);

                var actual = new float[qDim];
                Array.Copy(ragged, (long)t * qDim, actual, 0, qDim);
                AssertBitIdentical(
                    $"Attention({dtype}) N={n} seq={t} seqLen={seqLens[t]} maxSeq={maxSeqLen}",
                    expected, actual);
            }
        }
        finally
        {
            for (int t = 0; t < n; t++) { gpu.Free(kCaches[t]); gpu.Free(vCaches[t]); }
            gpu.Free(gpuQAll); gpu.Free(gpuOutAll);
            if (raggedScratch is { } rs) gpu.Free(rs);
            if (refScratch is { } fs) gpu.Free(fs);
        }
    }

    /// <summary>Ragged lengths under the 4096 shared-memory fast path, batch sizes crossing
    /// the 16-sequence chunk boundary.</summary>
    [Fact]
    public void Attention_Ragged_F32_BitwiseMatchesSequential()
    {
        foreach (int n in new[] { 1, 3, 8, 17 })
        {
            var rng = new Random(197_300 + n);
            var lens = new int[n];
            for (int t = 0; t < n; t++) lens[t] = 1 + rng.Next(300);
            AttentionCase(DType.Float32, maxSeqLen: 320, lens, withScratch: false);
        }
    }

    /// <summary>seqLen &gt; 4096 forces the per-(sequence, head) spill-scratch rows — mixed
    /// with short sequences so both paths run in one launch.</summary>
    [Fact]
    public void Attention_Ragged_F32_SpillPath_BitwiseMatchesSequential() =>
        AttentionCase(DType.Float32, maxSeqLen: 4500, seqLens: [4400, 17, 4100], withScratch: true);

    [Fact]
    public void Attention_Ragged_Bf16_BitwiseMatchesSequential() =>
        AttentionCase(DType.BFloat16, maxSeqLen: 320, seqLens: [200, 1, 64, 300, 7], withScratch: false);

    [Fact]
    public void Attention_Ragged_Q8_0_BitwiseMatchesSequential() =>
        AttentionCase(DType.Q8_0, maxSeqLen: 320, seqLens: [200, 1, 64, 300, 7], withScratch: false);

    // ── Broadcast bias add ──────────────────────────────────────────────────

    [Fact]
    public void AddBiasBatched_BitwiseMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int dim = 1234;
        foreach (int n in new[] { 1, 5, 17 })
        {
            var rng = new Random(197_400 + n);
            float[] xAll = RandomFloats(n * dim, rng);
            float[] bias = RandomFloats(dim, rng);
            var gpuBias = gpu.Upload(bias, TensorShape.D1(dim));

            var gpuXAll = gpu.Upload(xAll, TensorShape.D1((long)n * dim));
            gpu.AddBiasBatched(gpuXAll, gpuBias, dim, n);
            gpu.Synchronize();
            var ragged = new float[n * dim];
            gpu.Download(gpuXAll, ragged);
            gpu.Free(gpuXAll);

            for (int t = 0; t < n; t++)
            {
                var row = new float[dim];
                Array.Copy(xAll, (long)t * dim, row, 0, dim);
                var gpuRow = gpu.Upload(row, TensorShape.D1(dim));
                gpu.AddInPlace(gpuRow, gpuBias);
                gpu.Synchronize();
                var expected = new float[dim];
                gpu.Download(gpuRow, expected);
                gpu.Free(gpuRow);

                var actual = new float[dim];
                Array.Copy(ragged, (long)t * dim, actual, 0, dim);
                AssertBitIdentical($"AddBiasBatched N={n} row={t}", expected, actual);
            }
            gpu.Free(gpuBias);
        }
    }
}
