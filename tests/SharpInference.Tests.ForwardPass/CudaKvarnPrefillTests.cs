using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.TurboQuant;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Chunked batched KVarN prefill tests (issue #180 Task 6). Two levels:
/// <list type="bullet">
///   <item><b>Kernel</b> — <c>llm_kvarn_prefill_attention</c> (M chunk queries ×
///         all tiles + causal fp32 window, streaming softmax) against the CPU
///         <see cref="KVarNCompressor"/> oracle, per-dim max(1e-3, 1e-2·|x|) —
///         the same band as the per-token KVarN attention tests.</item>
///   <item><b>Integration</b> — <see cref="CudaForwardPass.Prefill"/> chunked vs
///         per-token on a real model: identical promotion cadence, near-identical
///         tiles (the RTN codes are byte-exact given identical inputs, but the
///         chunked trunk computes K/V via batched MMQ GEMMs and flash attention,
///         which are argmax-stable — not bit-identical — vs the per-token matvec
///         path, so a small fraction of codes may shift by ±1 step), final-logit
///         parity, and greedy-continuation agreement.</item>
/// </list>
/// Every test silently no-ops when CUDA (or the fixture model) is unavailable,
/// matching <see cref="CudaKvarnTests"/>.
/// </summary>
public sealed unsafe class CudaKvarnPrefillTests(ITestOutputHelper output)
{
    private const int Tile = KVarNCompressor.TileTokens; // 128

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Kernel-level parity vs the CPU oracle
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(384, 128, 128, 2, 4, 128)]   // steady-state chunk: 3 tiles + window 128, M=128, GQA 2q/kv
    [InlineData(256, 37, 44, 1, 2, 64)]      // tail chunk: odd f0, M=44 (last q-block has 12 queries), head_dim 64
    [InlineData(256, 64, 45, 1, 2, 128)]     // ODD tail (last q-block mq=13): half-active warp around the sub-warp shuffles
    [InlineData(4224, 128, 32, 1, 1, 128)]   // deep walk: 33 tiles, > 4096 total positions (no score-storage cap)
    public void KvarnPrefillAttention_MatchesCpuOracle(
        int tqLen, int f0, int m, int kvHeads, int qHeads, int headDim)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;

        Assert.Equal(0, tqLen % Tile);
        int numTiles = tqLen / Tile;
        int kvDim = kvHeads * headDim;
        int qDim = qHeads * headDim;
        int wTotal = f0 + m;                 // window rows after the chunk append
        var comp = new KVarNCompressor(headDim);
        int kStride = comp.KeyTileBytes;
        int vStride = comp.ValueTileBytes;

        var rng = new Random(20260711 + tqLen + headDim);
        var keys = new float[tqLen + wTotal][];
        var values = new float[tqLen + wTotal][];
        for (int t = 0; t < tqLen + wTotal; t++)
        {
            keys[t] = Gaussian(kvDim, rng);
            values[t] = Gaussian(kvDim, rng);
        }

        // CPU-compress the tq region into the packed tile layout (byte-compatible
        // with the GPU store by KvarnCompressTile_MatchesCpu).
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

        // Linear fp32 window (slot 0 = oldest): rows for positions tqLen .. tqLen+wTotal.
        var kWindow = new float[wTotal * kvDim];
        var vWindow = new float[wTotal * kvDim];
        for (int t = 0; t < wTotal; t++)
        {
            keys[tqLen + t].CopyTo(kWindow.AsSpan(t * kvDim, kvDim));
            values[tqLen + t].CopyTo(vWindow.AsSpan(t * kvDim, kvDim));
        }

        // Chunk queries, token-major [m × qDim].
        var queries = Gaussian(m * qDim, rng);

        var gpuKTiles = gpu.Upload(BytesToFloats(kTileBytes), TensorShape.D1(kTileBytes.Length / 4));
        var gpuVTiles = gpu.Upload(BytesToFloats(vTileBytes), TensorShape.D1(vTileBytes.Length / 4));
        var gpuKw = gpu.Upload(kWindow, TensorShape.D1(kWindow.Length));
        var gpuVw = gpu.Upload(vWindow, TensorShape.D1(vWindow.Length));
        var gpuQ = gpu.Upload(queries, TensorShape.D1(queries.Length));
        var gpuRot = gpu.Allocate(TensorShape.D1(queries.Length));
        var gpuOut = gpu.Allocate(TensorShape.D1(m * qDim));

        // Batched rotation: the per-token WHT kernel launched over m·qHeads head-rows
        // (rows are contiguous in the token-major layout) — same call the trunk makes.
        gpu.KvarnRotateQuery(gpuQ, gpuRot, qHeads * m, headDim);
        gpu.KvarnAttentionPrefill(gpuQ, gpuRot, gpuKTiles, gpuVTiles, gpuKw, gpuVw, gpuOut,
            qHeads, kvHeads, headDim, tqLen, f0, m, kStride, vStride);
        gpu.Synchronize();

        var gpuResult = new float[m * qDim];
        gpu.Download(gpuOut, gpuResult);
        gpu.Synchronize();

        // CPU oracle: per query i, ONE softmax over [tiles + window slots 0..f0+i]
        // (RotateQuery → KeyScores per tile → window dots → softmax → AggregateValues
        // rotated → UnrotateOutput → window V) — the per-token TqAttention pipeline.
        int hpkg = qHeads / kvHeads;
        float scale = 1f / MathF.Sqrt(headDim);
        var rotated = new float[headDim];
        var rotAcc = new float[headDim];
        var expected = new float[m * qDim];
        float worstCos = 1f;

        for (int i = 0; i < m; i++)
        {
            int fp32Len = f0 + i + 1;
            var scores = new float[tqLen + fp32Len];
            for (int h = 0; h < qHeads; h++)
            {
                int kvHead = h / hpkg;
                var qHead = queries.AsSpan(i * qDim + h * headDim, headDim);
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

                var outHead = expected.AsSpan(i * qDim + h * headDim, headDim);
                rotAcc.AsSpan(0, headDim).CopyTo(outHead);
                for (int t = 0; t < fp32Len; t++)
                {
                    float w = scores[tqLen + t];
                    var vRow = values[tqLen + t].AsSpan(kvHead * headDim, headDim);
                    for (int d = 0; d < headDim; d++) outHead[d] += w * vRow[d];
                }

                float cos = Cosine(outHead, gpuResult.AsSpan(i * qDim + h * headDim, headDim));
                if (cos < worstCos) worstCos = cos;
            }
        }

        output.WriteLine($"tq={tqLen} f0={f0} m={m} kv={kvHeads} q={qHeads} d={headDim}: " +
            $"worst per-(query,head) cosine vs CPU oracle {worstCos:F6}");
        Assert.True(worstCos > 0.999f, $"KVarN prefill attention cosine too low: {worstCos:F4}");

        for (int i = 0; i < expected.Length; i++)
        {
            float tol = MathF.Max(1e-3f, MathF.Abs(expected[i]) * 1e-2f);
            Assert.True(MathF.Abs(gpuResult[i] - expected[i]) <= tol,
                $"KVarN prefill attention mismatch at [{i}] (query {i / qDim}): " +
                $"gpu={gpuResult[i]:G6} cpu={expected[i]:G6} tol={tol:E2}");
        }

        gpu.Free(gpuKTiles); gpu.Free(gpuVTiles);
        gpu.Free(gpuKw); gpu.Free(gpuVw);
        gpu.Free(gpuQ); gpu.Free(gpuRot); gpu.Free(gpuOut);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CudaForwardPass integration: chunked vs per-token prefill
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Full-path A/B (window 256, prompt 640 → promotions at positions 256/384/512 → 3
    /// compressed tiles): the chunked driver must land the SAME promotion cadence and tile
    /// position ranges as the per-token loop, produce near-identical packed tiles (codes ±1
    /// step under batched-GEMM/flash noise), parity logits at the last prompt position, and
    /// the same greedy continuation.
    ///
    /// <para>
    /// Run over both RoPE conventions. Qwen3-0.6B is NEOX (head_dim 128); SmolLM2-1.7B is
    /// `llama`-arch NORM/interleaved (head_dim 64) and only reaches this driver because the
    /// #407 follow-up dropped the NEOX-only gate — the chunk trunk's RoPE already dispatched
    /// on the model's convention, so the gate, not the kernels, was what forced every
    /// llama-arch KVarN prompt onto the per-token loop. Covering NORM here is what keeps a
    /// future edit from silently mis-rotating those chunks (the failure #407 found latent in
    /// PrefillPackedTrunkMulti): a wrong rotation moves K/V into different tiles entirely,
    /// which the tile-content and continuation asserts below catch loudly.
    /// </para>
    /// </summary>
    [Theory]
    [InlineData("Qwen3-0.6B-Q8_0.gguf")]                // NEOX rope, head_dim 128
    [InlineData("SmolLM2-1.7B-Instruct-Q4_K_M.gguf")]   // NORM/interleaved rope, head_dim 64
    public void CudaForwardPass_KvarnChunkedPrefill_MatchesPerToken(string modelFile)
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath(modelFile);
        if (path is null) return;

        const int PromptLen = 640;
        const int DecodeSteps = 20;
        const int ExpectTq = 384;                 // 3 whole tiles (window 256: promotions at 256/384/512)

        var (tilesK1, tilesV1, logits1, cont1, wasBatched1, layers1) =
            RunPrefill(path, chunked: false, PromptLen, DecodeSteps, ExpectTq / Tile);
        var (tilesK2, tilesV2, logits2, cont2, wasBatched2, layers2) =
            RunPrefill(path, chunked: true, PromptLen, DecodeSteps, ExpectTq / Tile);

        Assert.False(wasBatched1);
        Assert.True(wasBatched2, "chunked KVarN prefill did not engage the batched trunk");
        Assert.Equal(layers1, layers2);

        // 1. Final-prompt-position logit parity (argmax + cosine). Position 639's
        // attention reads all 3 tiles + the full window, so wrong tile contents or
        // positions would blow this check grossly. The cosine band is calibrated to
        // the known cross-path noise: per-token K/V flows through dp4a matvecs while
        // the chunked trunk uses MMQ/flash, and each flipped 2-bit V code moves a
        // dequantized value by ~2× the normalized row RMS.
        int argmax1 = ArgMax(logits1), argmax2 = ArgMax(logits2);
        float cos = Cosine(logits1, logits2);
        float maxAbs = 0, atMax1 = 0, atMax2 = 0;
        for (int i = 0; i < logits1.Length; i++)
        {
            float d = MathF.Abs(logits1[i] - logits2[i]);
            if (d > maxAbs) { maxAbs = d; atMax1 = logits1[i]; atMax2 = logits2[i]; }
        }
        output.WriteLine($"final-position logits: argmax {argmax1} vs {argmax2}, cosine {cos:F6}, " +
            $"max |Δ| {maxAbs:E3} (per-token {atMax1:G6} vs chunked {atMax2:G6})");

        // 2. Greedy continuation (crosses the position-640 promotion on decode).
        output.WriteLine($"greedy continuation (per-token): {string.Join(", ", cont1)}");
        output.WriteLine($"greedy continuation (chunked):   {string.Join(", ", cont2)}");
        int firstDiff = -1;
        for (int i = 0; i < DecodeSteps; i++)
            if (cont1[i] != cont2[i]) { firstDiff = i; break; }

        Assert.Equal(argmax1, argmax2);
        Assert.True(cos > 0.995f, $"final-position logit cosine too low: {cos:F6}");
        Assert.True(firstDiff < 0,
            $"greedy continuation diverged at decode step {firstDiff}: " +
            $"per-token={cont1[Math.Max(firstDiff, 0)]} chunked={cont2[Math.Max(firstDiff, 0)]}");

        // 3. Tile parity. Byte-identity is NOT attainable end-to-end: the per-token
        // oracle computes K/V through dp4a matvecs (Q8_1-quantized activations,
        // issue #142) while the chunked trunk uses the MMQ/flash batched kernels —
        // both argmax-stable, neither bit-identical to the other (the same
        // pre-existing property that makes fp32 batched prefill "argmax-stable,
        // not bit-exact"). The compress kernel itself IS byte-exact given identical
        // inputs (CudaKvarnTests.KvarnCompressTile_MatchesCpu), so these thresholds
        // are calibrated to separate that K/V input noise — codes shift ±1-2 steps
        // at quantization boundaries; the coarse 2-bit V steps (~2× the Sinkhorn-
        // normalized row RMS) amplify each flip, so V rel-L2 runs tens of percent
        // while K (4-bit) stays far tighter — from a chunk-schedule bug (wrong rows
        // in a tile → ~94% uncorrelated codes with |Δ| up to 15, cosine ≈ 0).
        var comp = new KVarNCompressor(GetHeadDim(path));
        long codeTotal = 0, codeMismatch = 0; int maxDelta = 0;
        float maxScaleRel = 0f;
        double kMaxRelL2 = 0, kMinCos = 1, vMaxRelL2 = 0, vMinCos = 1;
        var dec1 = new float[Tile * comp.HeadDim];
        var dec2 = new float[Tile * comp.HeadDim];
        for (int layer = 0; layer < layers1; layer++)
        {
            CompareTiles(tilesK1[layer], tilesK2[layer], comp.KeyTileBytes,
                Tile + 2 * comp.HeadDim, bits: 4,
                ref codeTotal, ref codeMismatch, ref maxDelta, ref maxScaleRel);
            int groups = (comp.HeadDim + 127) / 128;
            CompareTiles(tilesV1[layer], tilesV2[layer], comp.ValueTileBytes,
                comp.HeadDim + 2 * Tile * groups, bits: 2,
                ref codeTotal, ref codeMismatch, ref maxDelta, ref maxScaleRel);

            for (int slot = 0; slot < tilesK1[layer].Length / comp.KeyTileBytes; slot++)
            {
                comp.DecompressKeyTile(tilesK1[layer].AsSpan(slot * comp.KeyTileBytes, comp.KeyTileBytes), dec1);
                comp.DecompressKeyTile(tilesK2[layer].AsSpan(slot * comp.KeyTileBytes, comp.KeyTileBytes), dec2);
                AccumulateTileError(dec1, dec2, ref kMaxRelL2, ref kMinCos);
                comp.DecompressValueTile(tilesV1[layer].AsSpan(slot * comp.ValueTileBytes, comp.ValueTileBytes), dec1);
                comp.DecompressValueTile(tilesV2[layer].AsSpan(slot * comp.ValueTileBytes, comp.ValueTileBytes), dec2);
                AccumulateTileError(dec1, dec2, ref vMaxRelL2, ref vMinCos);
            }
        }
        double mismatchFrac = codeTotal > 0 ? (double)codeMismatch / codeTotal : 0;
        output.WriteLine($"tile codes: {codeMismatch}/{codeTotal} mismatched ({mismatchFrac:P4}), " +
            $"max |Δcode| = {maxDelta}, max scale rel-diff = {maxScaleRel:E3}");
        output.WriteLine($"decompressed K tiles: worst cosine {kMinCos:F6}, worst rel-L2 {kMaxRelL2:P3}");
        output.WriteLine($"decompressed V tiles: worst cosine {vMinCos:F6}, worst rel-L2 {vMaxRelL2:P3}");
        Assert.True(maxDelta <= 3,
            $"tile code deltas up to {maxDelta} steps — beyond quantization-boundary noise (schedule bug?)");
        Assert.True(mismatchFrac < 0.10,
            $"tile code mismatch fraction {mismatchFrac:P3} ≥ 10% — beyond batched-GEMM noise (schedule bug?)");
        Assert.True(kMinCos > 0.98,
            $"decompressed K tile cosine {kMinCos:F4} ≤ 0.98 — tiles hold different content (schedule bug?)");
        Assert.True(vMinCos > 0.85,
            $"decompressed V tile cosine {vMinCos:F4} ≤ 0.85 — tiles hold different content (schedule bug?)");
    }

    /// <summary>
    /// Tile byte-identity under re-chunking, and startPos-continuation composition
    /// (the TruncateTo / multi-call shape). One shot prefills 640 tokens as chunks
    /// [0,256)+[256,384)+[384,512)+[512,640); the split call (300 + 340 @300) turns
    /// the second tile epoch into 44 + 84-token chunks. Tiles must depend ONLY on
    /// the positions they cover — never on how those positions were batched — and
    /// since both runs use the SAME kernels (per-query/per-row deterministic math,
    /// no cross-path dp4a-vs-MMQ noise), the packed tiles must match BYTE-EXACTLY.
    /// This is the strongest schedule check: any off-by-one in the promotion
    /// cadence or append slots would shift a whole row into the wrong tile.
    ///
    /// <para>
    /// Deliberately pinned to the Q8_0 fixture. Byte-exactness across chunk plans needs a
    /// chunk-size-INVARIANT trunk matmul, which the Q8_0 MMQ path gives and the Q4_K one does
    /// not: re-running this comparison on Q4_K models lands ~5-7% of tile codes off by a step
    /// or two purely from the different per-launch N (measured at 5.2% on Qwen3-8B-Q4_K_M,
    /// NEOX, and 7.1% on SmolLM2-1.7B-Q4_K_M, NORM — the dtype, not the rope convention).
    /// That is the same argmax-stable band the cross-path A/B above tolerates, so it is noise
    /// rather than a schedule bug — but it means a Q4_K case added here would assert a
    /// property the kernels never promised. NORM-rope schedule coverage lives in that A/B
    /// test instead, where the thresholds are calibrated for it.
    /// </para>
    /// </summary>
    [Fact]
    public void CudaForwardPass_KvarnChunkedPrefill_SplitCall_MatchesSingleShot()
    {
        if (!CudaBackend.IsAvailable()) return;
        var path = FindModelPath();
        if (path is null) return;

        const int PromptLen = 640;
        const int NumTiles = 3;
        var tokens = MakeTokens(path, PromptLen);

        using var gpu = TryCreate() ?? throw new InvalidOperationException("checked above");
        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        int single;
        var tilesK1 = new byte[hp.NumLayers][];
        var tilesV1 = new byte[hp.NumLayers][];
        using (var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 768,
                   enableTurboQuant: true, tqFp32Window: 256, tqQuantizer: TqQuantizer.KVarN))
        {
            fwd.KvarnBatchedPrefillEnabled = true;
            var logits = fwd.Prefill(tokens);
            Assert.True(fwd.LastPrefillWasBatched);
            single = ArgMax(logits);
            for (int layer = 0; layer < hp.NumLayers; layer++)
            {
                tilesK1[layer] = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: false, NumTiles);
                tilesV1[layer] = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: true, NumTiles);
            }
        }

        using (var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 768,
                   enableTurboQuant: true, tqFp32Window: 256, tqQuantizer: TqQuantizer.KVarN))
        {
            fwd.KvarnBatchedPrefillEnabled = true;
            fwd.Prefill(tokens.Take(300).ToArray());
            Assert.Equal(128, fwd.TqCompressedLength);   // promotion at position 256
            Assert.Equal(172, fwd.TqFp32Count);
            var logits = fwd.Prefill(tokens.Skip(300).ToArray(), startPos: 300);
            Assert.True(fwd.LastPrefillWasBatched);
            Assert.Equal(384, fwd.TqCompressedLength);
            Assert.Equal(256, fwd.TqFp32Count);
            Assert.Equal(single, ArgMax(logits));

            long diffBytes = 0, totalBytes = 0;
            for (int layer = 0; layer < hp.NumLayers; layer++)
            {
                var k2 = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: false, NumTiles);
                var v2 = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: true, NumTiles);
                for (int b = 0; b < k2.Length; b++) { totalBytes++; if (tilesK1[layer][b] != k2[b]) diffBytes++; }
                for (int b = 0; b < v2.Length; b++) { totalBytes++; if (tilesV1[layer][b] != v2[b]) diffBytes++; }
            }
            output.WriteLine($"re-chunked tile bytes: {diffBytes}/{totalBytes} differ");
            Assert.True(diffBytes == 0,
                $"{diffBytes}/{totalBytes} tile bytes differ between chunk plans — tiles must depend " +
                "only on the positions they cover, never on chunk boundaries (schedule bug).");
        }
    }

    private (byte[][] TilesK, byte[][] TilesV, float[] Logits, int[] Continuation, bool WasBatched, int Layers)
        RunPrefill(string modelPath, bool chunked, int promptLen, int decodeSteps, int numTiles)
    {
        using var gpu = TryCreate() ?? throw new InvalidOperationException("CUDA availability checked by caller");
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 768,
            enableTurboQuant: true, tqFp32Window: 256, tqQuantizer: TqQuantizer.KVarN);
        fwd.KvarnBatchedPrefillEnabled = chunked;

        var tokens = MakeTokens(modelPath, promptLen);
        var logits = fwd.Prefill(tokens).ToArray();
        bool wasBatched = fwd.LastPrefillWasBatched;

        // Promotion cadence must be IDENTICAL on both paths (window 256, prompt 640:
        // promotions before appending positions 256/384/512 → tq 384, window full).
        Assert.Equal(numTiles * Tile, fwd.TqCompressedLength);
        Assert.Equal(promptLen - numTiles * Tile, fwd.TqFp32Count);

        var tilesK = new byte[hp.NumLayers][];
        var tilesV = new byte[hp.NumLayers][];
        for (int layer = 0; layer < hp.NumLayers; layer++)
        {
            tilesK[layer] = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: false, numTiles);
            tilesV[layer] = fwd.DownloadKvarnTileBytesForTest(layer, valueTiles: true, numTiles);
        }

        // Greedy continuation from the prefill logits (exercises the decode path on
        // top of the prefilled cache, crossing the position-640 promotion).
        var continuation = new int[decodeSteps];
        int next = ArgMax(logits);
        for (int step = 0; step < decodeSteps; step++)
        {
            continuation[step] = next;
            next = ArgMax(fwd.Forward(next, promptLen + step));
        }

        return (tilesK, tilesV, logits, continuation, wasBatched, hp.NumLayers);
    }

    /// <summary>
    /// Per-head packed-tile comparison: scale-header floats tracked as max relative
    /// difference, code region nibble-by-nibble (K, 4-bit) / crumb-by-crumb (V, 2-bit)
    /// accumulating mismatch count and max step delta into the caller's counters.
    /// (No hard per-element assert here — the caller applies calibrated thresholds and
    /// prints the aggregate stats, since the chunked trunk's MMQ/flash K/V inputs are
    /// argmax-stable rather than bit-identical vs the per-token path.)
    /// </summary>
    private static void CompareTiles(byte[] a, byte[] b, int stride, int scaleFloats, int bits,
        ref long codeTotal, ref long codeMismatch, ref int maxDelta, ref float maxScaleRel)
    {
        Assert.Equal(a.Length, b.Length);
        int tiles = a.Length / stride;
        for (int tile = 0; tile < tiles; tile++)
        {
            var ta = a.AsSpan(tile * stride, stride);
            var tb = b.AsSpan(tile * stride, stride);

            var fa = MemoryMarshal.Cast<byte, float>(ta.Slice(0, scaleFloats * 4));
            var fb = MemoryMarshal.Cast<byte, float>(tb.Slice(0, scaleFloats * 4));
            for (int i = 0; i < scaleFloats; i++)
            {
                float denom = MathF.Max(1e-3f, MathF.Abs(fa[i]));
                float rel = MathF.Abs(fa[i] - fb[i]) / denom;
                if (rel > maxScaleRel) maxScaleRel = rel;
            }

            int mask = (1 << bits) - 1;
            int per = 8 / bits;
            for (int byteIdx = scaleFloats * 4; byteIdx < stride; byteIdx++)
            {
                int ba = ta[byteIdx], bb = tb[byteIdx];
                for (int k = 0; k < per; k++)
                {
                    int ca = (ba >> (k * bits)) & mask;
                    int cb = (bb >> (k * bits)) & mask;
                    codeTotal++;
                    if (ca != cb)
                    {
                        codeMismatch++;
                        int d = Math.Abs(ca - cb);
                        if (d > maxDelta) maxDelta = d;
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Relative L2 error and cosine between two decompressed tiles.</summary>
    private static void AccumulateTileError(float[] a, float[] b, ref double maxRelL2, ref double minCos)
    {
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            diff2 += d * d;
            ref2 += (double)a[i] * a[i];
        }
        if (ref2 > 0)
        {
            double rel = Math.Sqrt(diff2 / ref2);
            if (rel > maxRelL2) maxRelL2 = rel;
        }
        double cos = Cosine(a, b);
        if (cos < minCos) minCos = cos;
    }

    private static int[] MakeTokens(string modelPath, int count)
    {
        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var rng = new Random(6161);
        var tokens = new int[count];
        for (int i = 0; i < count; i++)
            tokens[i] = rng.Next(0, Math.Min(hp.VocabSize, 32000));
        return tokens;
    }

    private static int GetHeadDim(string modelPath)
    {
        using var model = GgufModel.Open(modelPath);
        return ModelHyperparams.FromGgufMetadata(model.Metadata, model).HeadDim;
    }

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
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

    private static float[] BytesToFloats(byte[] bytes)
    {
        var floats = new float[bytes.Length / 4];
        bytes.AsSpan().CopyTo(MemoryMarshal.AsBytes(floats.AsSpan()));
        return floats;
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
