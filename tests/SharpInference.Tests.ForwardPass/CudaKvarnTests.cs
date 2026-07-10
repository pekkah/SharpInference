using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.TurboQuant;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CUDA KVarN kernel + decode-integration tests (issue #180 Task 5a). Each test
/// silently no-ops when no CUDA device is available (same convention as
/// <see cref="CudaTurboQuantTests"/>).
///
/// The CPU <see cref="KVarNCompressor"/> is the oracle throughout:
/// <list type="bullet">
///   <item>rotate-query parity is elementwise (abs 1e-4 — expected ~1e-7: both
///         sides run the same butterfly with the same 1/sqrt(dim) scale);</item>
///   <item>compress-tile parity is BYTE-EXACT on the packed RTN codes (the
///         kernel uses correctly-rounded single ops + round-to-even, so codes
///         depend only on ops that are bit-identical to the CPU compressor)
///         and rel-1e-5 on the stored scale floats (logf/expf may differ from
///         MathF.Log/Exp in the last ulp);</item>
///   <item>attention parity vs the CPU KeyScores/AggregateValues/UnrotateOutput
///         pipeline is per-dim max(1e-3, 1e-2·|expected|) — the same band the
///         TQ CUDA tests use.</item>
/// </list>
/// </summary>
public sealed unsafe class CudaKvarnTests(ITestOutputHelper output)
{
    private const int Tile = KVarNCompressor.TileTokens; // 128

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Rotate query
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(64, 4)]
    [InlineData(128, 4)]
    [InlineData(256, 2)]
    public void KvarnRotateQuery_MatchesCpu(int headDim, int numHeads)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        var rng = new Random(4242 + headDim);
        var query = new float[numHeads * headDim];
        for (int i = 0; i < query.Length; i++) query[i] = (float)(rng.NextDouble() * 2 - 1);

        var gpuQ = gpu.Upload(query, TensorShape.D1(query.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(query.Length));

        gpu.KvarnRotateQuery(gpuQ, gpuRotated, numHeads, headDim);
        gpu.Synchronize();

        var gpuResult = new float[query.Length];
        gpu.Download(gpuRotated, gpuResult);

        var comp = new KVarNCompressor(headDim);
        var expected = new float[query.Length];
        for (int h = 0; h < numHeads; h++)
            comp.RotateQuery(query.AsSpan(h * headDim, headDim), expected.AsSpan(h * headDim, headDim));

        for (int i = 0; i < query.Length; i++)
            Assert.True(MathF.Abs(gpuResult[i] - expected[i]) < 1e-4f,
                $"KvarnRotateQuery mismatch at [{i}]: gpu={gpuResult[i]} cpu={expected[i]}");

        gpu.Free(gpuQ); gpu.Free(gpuRotated);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Compress tile
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void KvarnCompressTile_MatchesCpu(int headDim)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        const int NumKvHeads = 2;
        const int NumTiles = 2;   // exercises the tile-index output placement
        const int WindowRows = 256;
        int kvDim = NumKvHeads * headDim;
        var comp = new KVarNCompressor(headDim);
        int kStride = comp.KeyTileBytes;
        int vStride = comp.ValueTileBytes;

        var gpuKTiles = gpu.Allocate(TensorShape.D1((long)NumTiles * NumKvHeads * kStride / 4));
        var gpuVTiles = gpu.Allocate(TensorShape.D1((long)NumTiles * NumKvHeads * vStride / 4));
        var gpuWork = gpu.Allocate(TensorShape.D1((long)NumKvHeads * 2 * Tile * headDim));

        var rng = new Random(20260710 + headDim);
        var kWindows = new float[NumTiles][];
        var vWindows = new float[NumTiles][];

        // The kernel always compresses window rows [0, 128) into the given tile
        // index (the host shifts the window between promotions), so each tile
        // gets its own window upload.
        for (int tile = 0; tile < NumTiles; tile++)
        {
            kWindows[tile] = Gaussian(WindowRows * kvDim, rng);
            vWindows[tile] = Gaussian(WindowRows * kvDim, rng);
            var gpuKw = gpu.Upload(kWindows[tile], TensorShape.D1(kWindows[tile].Length));
            var gpuVw = gpu.Upload(vWindows[tile], TensorShape.D1(vWindows[tile].Length));
            gpu.KvarnCompressTile(gpuKw, gpuVw, gpuKTiles, gpuVTiles, gpuWork,
                kvDim, headDim, tile, NumKvHeads, kStride, vStride,
                KVarNCompressor.DefaultSinkhornIterations);
            gpu.Synchronize();
            gpu.Free(gpuKw); gpu.Free(gpuVw);
        }

        byte[] gpuKBytes = DownloadBytes(gpu, gpuKTiles, NumTiles * NumKvHeads * kStride);
        byte[] gpuVBytes = DownloadBytes(gpu, gpuVTiles, NumTiles * NumKvHeads * vStride);

        int kScaleFloats = Tile + 2 * headDim;
        int groups = (headDim + 127) / 128;
        int vScaleFloats = headDim + 2 * Tile * groups;

        var gather = new float[Tile * headDim];
        var cpuTile = new byte[Math.Max(kStride, vStride)];
        var gpuDec = new float[Tile * headDim];
        var cpuDec = new float[Tile * headDim];

        for (int tile = 0; tile < NumTiles; tile++)
        {
            for (int head = 0; head < NumKvHeads; head++)
            {
                // K tile.
                GatherHead(kWindows[tile], gather, head, headDim, kvDim);
                comp.CompressKeyTile(gather, cpuTile);
                var gpuKTile = gpuKBytes.AsSpan((tile * NumKvHeads + head) * kStride, kStride);
                AssertTileMatch(gpuKTile, cpuTile.AsSpan(0, kStride), kScaleFloats,
                    $"K tile {tile} head {head} (headDim {headDim})");

                comp.DecompressKeyTile(gpuKTile, gpuDec);
                comp.DecompressKeyTile(cpuTile.AsSpan(0, kStride), cpuDec);
                AssertDecompressedClose(gpuDec, cpuDec, $"K tile {tile} head {head}");

                // V tile.
                GatherHead(vWindows[tile], gather, head, headDim, kvDim);
                comp.CompressValueTile(gather, cpuTile);
                var gpuVTile = gpuVBytes.AsSpan((tile * NumKvHeads + head) * vStride, vStride);
                AssertTileMatch(gpuVTile, cpuTile.AsSpan(0, vStride), vScaleFloats,
                    $"V tile {tile} head {head} (headDim {headDim})");

                comp.DecompressValueTile(gpuVTile, gpuDec);
                comp.DecompressValueTile(cpuTile.AsSpan(0, vStride), cpuDec);
                AssertDecompressedClose(gpuDec, cpuDec, $"V tile {tile} head {head}");
            }
        }

        gpu.Free(gpuKTiles); gpu.Free(gpuVTiles); gpu.Free(gpuWork);
    }

    /// <summary>
    /// Packed-tile comparison: stored scale floats (the header region) to rel
    /// 1e-5 — logf/expf on the device can differ from MathF.Log/Exp by a ulp —
    /// and the RTN code region BYTE-EXACT (codes depend only on correctly-rounded
    /// ops that are bit-identical between the CPU and the kernel).
    /// </summary>
    private static void AssertTileMatch(ReadOnlySpan<byte> gpuTile, ReadOnlySpan<byte> cpuTile,
        int scaleFloats, string what)
    {
        var gpuF = MemoryMarshal.Cast<byte, float>(gpuTile.Slice(0, scaleFloats * 4));
        var cpuF = MemoryMarshal.Cast<byte, float>(cpuTile.Slice(0, scaleFloats * 4));
        for (int i = 0; i < scaleFloats; i++)
        {
            float tol = MathF.Max(1e-6f, MathF.Abs(cpuF[i]) * 1e-5f);
            Assert.True(MathF.Abs(gpuF[i] - cpuF[i]) <= tol,
                $"{what}: scale float [{i}] mismatch: gpu={gpuF[i]:G9} cpu={cpuF[i]:G9} tol={tol:E2}");
        }

        int mismatches = 0;
        int firstMismatch = -1;
        for (int b = scaleFloats * 4; b < cpuTile.Length; b++)
        {
            if (gpuTile[b] != cpuTile[b])
            {
                if (firstMismatch < 0) firstMismatch = b;
                mismatches++;
            }
        }
        Assert.True(mismatches == 0,
            $"{what}: {mismatches} code byte(s) differ (first at byte {firstMismatch}: " +
            $"gpu=0x{(firstMismatch >= 0 ? gpuTile[firstMismatch] : 0):X2} " +
            $"cpu=0x{(firstMismatch >= 0 ? cpuTile[firstMismatch] : 0):X2}). " +
            "Codes must be bit-identical to the CPU compressor.");
    }

    private static void AssertDecompressedClose(float[] gpuDec, float[] cpuDec, string what)
    {
        for (int i = 0; i < cpuDec.Length; i++)
        {
            float tol = MathF.Max(1e-4f, MathF.Abs(cpuDec[i]) * 1e-4f);
            Assert.True(MathF.Abs(gpuDec[i] - cpuDec[i]) <= tol,
                $"{what}: decompressed [{i}] mismatch: gpu={gpuDec[i]:G9} cpu={cpuDec[i]:G9}");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Attention
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(128, 97, 1, 1, 128)]    // one tile + partial window
    [InlineData(256, 256, 2, 4, 128)]   // two tiles + full window, GQA 2 q heads / kv head
    [InlineData(384, 130, 2, 2, 64)]    // three tiles, head_dim 64
    public void KvarnAttention_MatchesCpuOracle(int tqLen, int fp32Len, int kvHeads, int qHeads, int headDim)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        RunAttentionParity(gpu, tqLen, fp32Len, kvHeads, qHeads, headDim, useScratch: false,
            seed: 991 + tqLen + headDim);
    }

    /// <summary>
    /// Long-context branch: tq 4096 + window 64 = 4160 positions &gt; the 4096
    /// shared-scores cap, so phase 1-3 run against the global scores scratch.
    /// Same CPU oracle, same tolerance — a wrong scratch index, softmax, or V
    /// walk shows up as a per-dim mismatch.
    /// </summary>
    [Fact]
    public void KvarnAttention_ScratchPath_MatchesCpuOracle()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        RunAttentionParity(gpu, tqLen: 4096, fp32Len: 64, kvHeads: 1, qHeads: 1, headDim: 128,
            useScratch: true, seed: 777);
    }

    private void RunAttentionParity(CudaBackend gpu, int tqLen, int fp32Len,
        int kvHeads, int qHeads, int headDim, bool useScratch, int seed)
    {
        Assert.Equal(0, tqLen % Tile);
        int numTiles = tqLen / Tile;
        int kvDim = kvHeads * headDim;
        int totalSeq = tqLen + fp32Len;
        var comp = new KVarNCompressor(headDim);
        int kStride = comp.KeyTileBytes;
        int vStride = comp.ValueTileBytes;

        var rng = new Random(seed);
        var keys = new float[totalSeq][];
        var values = new float[totalSeq][];
        for (int t = 0; t < totalSeq; t++)
        {
            keys[t] = Gaussian(kvDim, rng);
            values[t] = Gaussian(kvDim, rng);
        }

        // CPU-compress the tq region into the packed tile layout the kernel reads
        // (byte-compatible by construction — proven by KvarnCompressTile_MatchesCpu).
        var kTileBytes = new byte[numTiles * kvHeads * kStride];
        var vTileBytes = new byte[numTiles * kvHeads * vStride];
        var gather = new float[Tile * headDim];
        for (int tile = 0; tile < numTiles; tile++)
        {
            for (int head = 0; head < kvHeads; head++)
            {
                for (int t = 0; t < Tile; t++)
                    keys[tile * Tile + t].AsSpan(head * headDim, headDim)
                        .CopyTo(gather.AsSpan(t * headDim, headDim));
                comp.CompressKeyTile(gather, kTileBytes.AsSpan((tile * kvHeads + head) * kStride, kStride));

                for (int t = 0; t < Tile; t++)
                    values[tile * Tile + t].AsSpan(head * headDim, headDim)
                        .CopyTo(gather.AsSpan(t * headDim, headDim));
                comp.CompressValueTile(gather, vTileBytes.AsSpan((tile * kvHeads + head) * vStride, vStride));
            }
        }

        // FP32 window rows are the trailing positions, linear (slot 0 = oldest).
        var kWindow = new float[fp32Len * kvDim];
        var vWindow = new float[fp32Len * kvDim];
        for (int t = 0; t < fp32Len; t++)
        {
            keys[tqLen + t].CopyTo(kWindow.AsSpan(t * kvDim, kvDim));
            values[tqLen + t].CopyTo(vWindow.AsSpan(t * kvDim, kvDim));
        }

        var query = Gaussian(qHeads * headDim, rng);

        var gpuKTiles = gpu.Upload(BytesToFloats(kTileBytes), TensorShape.D1(kTileBytes.Length / 4));
        var gpuVTiles = gpu.Upload(BytesToFloats(vTileBytes), TensorShape.D1(vTileBytes.Length / 4));
        var gpuKw = gpu.Upload(kWindow, TensorShape.D1(kWindow.Length));
        var gpuVw = gpu.Upload(vWindow, TensorShape.D1(vWindow.Length));
        var gpuQ = gpu.Upload(query, TensorShape.D1(query.Length));
        var gpuRotated = gpu.Allocate(TensorShape.D1(query.Length));
        var gpuOut = gpu.Allocate(TensorShape.D1(qHeads * headDim));
        var scratch = useScratch ? gpu.Allocate(TensorShape.D1((long)qHeads * totalSeq)) : (Tensor?)null;

        gpu.KvarnRotateQuery(gpuQ, gpuRotated, qHeads, headDim);
        gpu.KvarnAttention(gpuQ, gpuRotated, gpuKTiles, gpuVTiles, gpuKw, gpuVw, gpuOut,
            scratch, qHeads, kvHeads, headDim, tqLen, fp32Len, totalSeq, kStride, vStride);
        gpu.Synchronize();

        var gpuResult = new float[qHeads * headDim];
        gpu.Download(gpuOut, gpuResult);

        // CPU oracle: the exact pipeline ForwardPass.TqAttention drives in KVarN
        // mode — RotateQuery → per-tile KeyScores (attn scale folded after) →
        // FP32 window dots → softmax → per-tile AggregateValues (rotated domain)
        // → one UnrotateOutput → FP32 window V accumulation.
        var expected = new float[qHeads * headDim];
        int hpkg = qHeads / kvHeads;
        float scale = 1f / MathF.Sqrt(headDim);
        var scores = new float[totalSeq];
        var rotated = new float[headDim];
        var rotAcc = new float[headDim];

        for (int h = 0; h < qHeads; h++)
        {
            int kvHead = h / hpkg;
            var qHead = query.AsSpan(h * headDim, headDim);
            comp.RotateQuery(qHead, rotated);

            for (int tile = 0; tile < numTiles; tile++)
            {
                comp.KeyScores(kTileBytes.AsSpan((tile * kvHeads + kvHead) * kStride, kStride),
                    rotated, scores.AsSpan(tile * Tile, Tile));
                for (int t = 0; t < Tile; t++) scores[tile * Tile + t] *= scale;
            }
            for (int t = 0; t < fp32Len; t++)
            {
                float dot = 0f;
                var kRow = keys[tqLen + t].AsSpan(kvHead * headDim, headDim);
                for (int d = 0; d < headDim; d++) dot += qHead[d] * kRow[d];
                scores[tqLen + t] = dot * scale;
            }

            Softmax(scores);

            Array.Clear(rotAcc);
            for (int tile = 0; tile < numTiles; tile++)
                comp.AggregateValues(vTileBytes.AsSpan((tile * kvHeads + kvHead) * vStride, vStride),
                    scores.AsSpan(tile * Tile, Tile), rotAcc);
            comp.UnrotateOutput(rotAcc);

            var outHead = expected.AsSpan(h * headDim, headDim);
            rotAcc.AsSpan(0, headDim).CopyTo(outHead);
            for (int t = 0; t < fp32Len; t++)
            {
                float w = scores[tqLen + t];
                var vRow = values[tqLen + t].AsSpan(kvHead * headDim, headDim);
                for (int d = 0; d < headDim; d++) outHead[d] += w * vRow[d];
            }
        }

        float cos = Cosine(expected, gpuResult);
        output.WriteLine($"tq={tqLen} fp32={fp32Len} kv={kvHeads} q={qHeads} d={headDim} " +
            $"scratch={useScratch}: cosine vs CPU oracle {cos:F6}");
        Assert.True(cos > 0.999f, $"KVarN attention output cosine vs CPU oracle too low: {cos:F4}");

        for (int i = 0; i < expected.Length; i++)
        {
            float tol = MathF.Max(1e-3f, MathF.Abs(expected[i]) * 1e-2f);
            Assert.True(MathF.Abs(gpuResult[i] - expected[i]) <= tol,
                $"KVarN attention mismatch at [{i}]: gpu={gpuResult[i]:G6} cpu={expected[i]:G6} tol={tol:E2}");
        }

        gpu.Free(gpuKTiles); gpu.Free(gpuVTiles);
        gpu.Free(gpuKw); gpu.Free(gpuVw);
        gpu.Free(gpuQ); gpu.Free(gpuRotated); gpu.Free(gpuOut);
        if (scratch is { } s) gpu.Free(s);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CudaForwardPass integration (model-gated: skipped when neither CUDA nor
    // the fixture model is present)
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Whole-tile promotion cadence on the real decode path, mirroring the CPU
    /// KVarNKvCacheTests: window 256 → the ring stays FP32 through position 255,
    /// the append at position 256 promotes the first 128 positions, the append
    /// at position 384 promotes the next 128 — the TQ length only ever grows in
    /// 128-token steps and CPU/CUDA agree position-for-position. Also covers
    /// TruncateTo semantics and the batching guard on the same instance.
    /// </summary>
    [Fact]
    public void CudaForwardPass_Kvarn_PromotionCadence_And_Guards()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 640,
            enableTurboQuant: true, tqFp32Window: 256, tqQuantizer: TqQuantizer.KVarN);

        // Guards: KVarN inherits every TurboQuant exclusivity constraint.
        Assert.False(fwd.SupportsContinuousBatching);
        Assert.Throws<NotSupportedException>(() => fwd.CreateCache());
        Assert.False(fwd.SupportsHiddenTaps);

        int token = 100 % hp.VocabSize;
        ReadOnlySpan<float> logits = default;
        for (int pos = 0; pos < 400; pos++)
        {
            logits = fwd.Forward(token, pos);

            // Promotion cadence (window 256): fp32Count reaches 256 at the append
            // of position 256 → tq 0→128; again at position 384 → tq 128→256.
            int expectedTq = pos < 256 ? 0 : pos < 384 ? 128 : 256;
            Assert.Equal(expectedTq, fwd.TqCompressedLength);
            Assert.Equal(pos + 1 - expectedTq, fwd.TqFp32Count);
        }

        for (int i = 0; i < logits.Length; i++)
            Assert.True(float.IsFinite(logits[i]), $"Non-finite logit at idx {i} after 400 KVarN decode steps");

        // TruncateTo: rewinding inside the FP32 window resyncs the linear window
        // count; rewinding into the compressed region throws.
        fwd.TruncateTo(300);
        Assert.Equal(256, fwd.TqCompressedLength);
        Assert.Equal(44, fwd.TqFp32Count);
        Assert.Throws<NotSupportedException>(() => fwd.TruncateTo(200));

        // Decode continues coherently after the rewind, keeping the cadence.
        for (int pos = 300; pos < 340; pos++)
        {
            logits = fwd.Forward(token, pos);
            Assert.Equal(256, fwd.TqCompressedLength);
        }
        for (int i = 0; i < logits.Length; i++)
            Assert.True(float.IsFinite(logits[i]), $"Non-finite logit at idx {i} after post-truncate decode");
    }

    /// <summary>Construction-time guard parity with TQ: SnapKV and narrowed-KV combos throw.</summary>
    [Fact]
    public void CudaForwardPass_Kvarn_RejectsSnapKvAndNarrowedKv()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        var prevDtype = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        try
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "256");
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", null);
            var ex = Assert.Throws<NotSupportedException>(() => new CudaForwardPass(
                model, gpu, hp, maxContextLength: 640,
                enableTurboQuant: true, tqQuantizer: TqQuantizer.KVarN));
            Assert.Contains("SnapKV", ex.Message);

            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "bf16");
            var ex2 = Assert.Throws<NotSupportedException>(() => new CudaForwardPass(
                model, gpu, hp, maxContextLength: 640,
                enableTurboQuant: true, tqQuantizer: TqQuantizer.KVarN));
            Assert.Contains("TurboQuant", ex2.Message);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevDtype);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static void GatherHead(float[] window, float[] gather, int head, int headDim, int kvDim)
    {
        for (int t = 0; t < Tile; t++)
            window.AsSpan(t * kvDim + head * headDim, headDim)
                .CopyTo(gather.AsSpan(t * headDim, headDim));
    }

    private static byte[] DownloadBytes(CudaBackend gpu, Tensor tensor, int byteCount)
    {
        var floats = new float[(byteCount + 3) / 4];
        gpu.Download(tensor, floats);
        var bytes = new byte[byteCount];
        MemoryMarshal.AsBytes(floats.AsSpan()).Slice(0, byteCount).CopyTo(bytes);
        return bytes;
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        var floats = new float[bytes.Length / 4];
        bytes.AsSpan().CopyTo(MemoryMarshal.AsBytes(floats.AsSpan()));
        return floats;
    }

    private static void Softmax(Span<float> x)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < x.Length; i++) if (x[i] > max) max = x[i];
        double sum = 0;
        for (int i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        float inv = (float)(1.0 / sum);
        for (int i = 0; i < x.Length; i++) x[i] *= inv;
    }

    private static float[] Gaussian(int n, Random rng)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    private static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom > 0 ? (float)(dot / denom) : 0f;
    }

    private static string? FindModelPath(string file = "Qwen3-0.6B-Q8_0.gguf")
    {
        string[] absolute =
        {
            $@"C:\models\{file}",
            $@"E:\models\{file}",
        };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", file);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
