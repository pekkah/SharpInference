using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.TurboQuant;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for the KVarN quantizer mode of <see cref="TurboQuantKvCache"/>
/// (issue #180 Task 2: KVarN wired into the TurboQuant cache + CPU TqAttention
/// path as a selectable quantizer).
///
/// Covers:
/// <list type="bullet">
///   <item>End-to-end attention parity over a KVarN-mode cache vs (a) an exact
///         reference computed on the decompressed tiles (wiring parity — tight
///         bound) and (b) the FP32 truth (quantization quality — loose bound,
///         numbers reported; KVarN V is 2-bit so ~0.5 relative error on Gaussian
///         data is the expected codec floor, see KVarNCompressorTests).</item>
///   <item>Cache mechanics: whole-tile promotion at the 128 boundary, Reset
///         reuse, TruncateTo semantics, and config-time rejection of the
///         unsupported combos (SnapKV, sub-tile FP32 window, non-pow2 head dim).</item>
/// </list>
/// </summary>
public sealed class KVarNKvCacheTests(ITestOutputHelper output)
{
    private const int Tile = KVarNCompressor.TileTokens; // 128

    // ─────────────────────────────────────────────────────────────────────────
    // End-to-end attention parity
    // ─────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(100, 128, 1, 1)]   // shorter than window — all FP32, must be exact
    [InlineData(129, 128, 1, 1)]   // exactly one compressed tile (+1 in the window)
    [InlineData(256, 128, 1, 1)]   // one tile + a full FP32 window
    [InlineData(700, 192, 2, 4)]   // several tiles + non-tile-aligned FP32 remainder, GQA (4 q / 2 kv)
    public void Attention_KVarNCache_MatchesReferences(int seqLen, int window, int kvHeads, int qHeads)
    {
        const int headDim = 128;
        var (cache, keys, values) = BuildKVarNCache(seqLen, window, kvHeads, headDim, seed: 913 + seqLen);
        using var _ = cache;

        var rng = new Random(4711);
        float[] q = Gaussian(qHeads * headDim, rng);

        float[] actual = CacheAttention(cache, q, qHeads);
        float[] fp32Ref = ReferenceAttention(keys, values, q, qHeads, kvHeads, headDim);

        int tqLen = cache.GetTqLength(0);
        float relFp32 = RelL2(fp32Ref, actual);
        float cosFp32 = Cosine(fp32Ref, actual);
        output.WriteLine($"seqLen={seqLen} window={window} tq={tqLen} fp32={seqLen - tqLen} " +
            $"kv={kvHeads} q={qHeads}: rel-L2 vs FP32 truth {relFp32:F4}, cosine {cosFp32:F4}");

        if (tqLen == 0)
        {
            // No compressed positions: the whole path is plain FP32 math.
            Assert.True(relFp32 < 1e-5f, $"All-FP32 case should be numerically exact, got rel-L2 {relFp32:E3}");
            return;
        }

        Assert.Equal(0, tqLen % Tile); // KVarN invariant: whole tiles only

        // Wiring parity: attention recomputed over the decompressed tiles + FP32
        // window must match the fused cache path almost exactly (KeyScores and
        // AggregateValues are proven against decompression in KVarNCompressorTests
        // to <1e-4; softmax propagation keeps this in the 1e-3 band).
        var (effKeys, effValues) = DecompressedKv(cache, keys, values);
        float[] decompRef = ReferenceAttention(effKeys, effValues, q, qHeads, kvHeads, headDim);
        float relWiring = RelL2(decompRef, actual);
        output.WriteLine($"  rel-L2 vs decompressed reference (wiring parity): {relWiring:E3}");
        Assert.True(relWiring < 5e-3f,
            $"Cache attention deviates from the decompressed-tile reference: {relWiring:E3} — staging/promotion/aggregation wiring is broken.");

        // Quality gate vs FP32 truth: dominated by the 2-bit V codec floor
        // (~0.5 rel on Gaussian data per KVarNCompressorTests) plus a small
        // softmax perturbation from the 4-bit K scores. Loose bound by design —
        // the reported numbers above are the real deliverable.
        Assert.True(relFp32 < 0.65f, $"KVarN attention error vs FP32 truth too high: {relFp32:F4}");
        Assert.True(cosFp32 > 0.75f, $"KVarN attention cosine vs FP32 truth too low: {cosFp32:F4}");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cache mechanics
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void Append_PromotesWholeTiles_AtWindowOverflow()
    {
        const int headDim = 64;
        using var cache = new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: 512, numKvHeads: 1, headDim: headDim,
            fp32WindowSize: 128, quantizer: TqQuantizer.KVarN);

        var rng = new Random(7);
        for (int pos = 0; pos < 300; pos++)
        {
            cache.Append(0, Gaussian(headDim, rng), Gaussian(headDim, rng));
            cache.IncrementPosition();

            // Promotion fires inside the Append that finds the window full:
            // fp32Count reaches 128 at pos 128 (tq 0→128) and again at pos 256.
            int expectedTq = pos < 128 ? 0 : pos < 256 ? 128 : 256;
            Assert.Equal(expectedTq, cache.GetTqLength(0));
            Assert.Equal(0, cache.GetTqLength(0) % Tile);
            Assert.Equal(0, cache.KeyStagingCount(0));
        }

        Assert.Equal(300, cache.Length);
        Assert.Equal(2, cache.NumKeyTiles(0));
        Assert.Equal(256, cache.TqLength);
        Assert.Equal(44, cache.Fp32Length);
        Assert.Equal(Tile, cache.TileSizeTokens);
        Assert.Equal(TqQuantizer.KVarN, cache.Quantizer);
    }

    [Fact]
    public void Reset_AllowsReuse_WithCorrectPromotionAndFiniteAttention()
    {
        const int headDim = 128;
        var (cache, _, _) = BuildKVarNCache(seqLen: 200, window: 128, kvHeads: 1, headDim: headDim, seed: 11);
        using var _d = cache;
        Assert.Equal(128, cache.GetTqLength(0));

        cache.Reset();
        Assert.Equal(0, cache.Length);
        Assert.Equal(0, cache.GetTqLength(0));

        // Refill after Reset: promotion cadence must be identical and the
        // attention path must produce finite output from the reused buffers.
        var rng = new Random(12);
        for (int pos = 0; pos < 200; pos++)
        {
            cache.Append(0, Gaussian(headDim, rng), Gaussian(headDim, rng));
            cache.IncrementPosition();
        }
        Assert.Equal(200, cache.Length);
        Assert.Equal(128, cache.GetTqLength(0));

        float[] outVec = CacheAttention(cache, Gaussian(headDim, rng), numQHeads: 1);
        Assert.All(outVec, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void TruncateTo_WithinWindow_Ok_IntoCompressedRegion_Throws()
    {
        var (cache, _, _) = BuildKVarNCache(seqLen: 200, window: 128, kvHeads: 1, headDim: 128, seed: 21);
        using var _d = cache;
        Assert.Equal(128, cache.GetTqLength(0));

        cache.TruncateTo(150); // within the FP32 window (tqLen=128 <= 150)
        Assert.Equal(150, cache.Length);
        Assert.Equal(128, cache.GetTqLength(0));

        Assert.Throws<NotSupportedException>(() => cache.TruncateTo(100));

        // Appends after a truncate keep the whole-tile promotion cadence.
        var rng = new Random(22);
        for (int pos = 150; pos < 260; pos++)
        {
            cache.Append(0, Gaussian(128, rng), Gaussian(128, rng));
            cache.IncrementPosition();
        }
        Assert.Equal(260, cache.Length);
        Assert.Equal(256, cache.GetTqLength(0)); // promoted again at total=256
    }

    [Fact]
    public void Compact_Throws_InKVarNMode()
    {
        var (cache, _, _) = BuildKVarNCache(seqLen: 200, window: 128, kvHeads: 1, headDim: 128, seed: 31);
        using var _d = cache;

        int[] keep = new int[100];
        for (int i = 0; i < keep.Length; i++) keep[i] = i * 2;

        var ex = Assert.Throws<NotSupportedException>(() => cache.Compact(keep, 200));
        Assert.Contains("KVarN", ex.Message);
    }

    [Fact]
    public void Ctor_Rejects_WindowSmallerThanTile_WhenCompressedRegionPossible()
    {
        // A compressed region can exist (maxSeqLen > window) but the window can
        // never assemble a full 128-token tile → reject at construction.
        var ex = Assert.Throws<ArgumentException>(() => new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: 512, numKvHeads: 1, headDim: 128,
            fp32WindowSize: 64, quantizer: TqQuantizer.KVarN));
        Assert.Contains("128", ex.Message);

        // No compressed region possible (maxSeqLen <= window) → any window is fine.
        using var ok = new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: 64, numKvHeads: 1, headDim: 128,
            fp32WindowSize: 64, quantizer: TqQuantizer.KVarN);
        Assert.Equal(0, ok.Length);
    }

    [Fact]
    public void Ctor_Rejects_NonPow2HeadDim()
    {
        var ex = Assert.Throws<ArgumentException>(() => new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: 512, numKvHeads: 1, headDim: 96,
            fp32WindowSize: 128, quantizer: TqQuantizer.KVarN));
        Assert.Contains("power of 2", ex.Message);
    }

    [Fact]
    public void LloydMaxCompressorAccessors_Throw_InKVarNMode()
    {
        using var cache = new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: 512, numKvHeads: 1, headDim: 128,
            fp32WindowSize: 128, quantizer: TqQuantizer.KVarN);

        Assert.Throws<InvalidOperationException>(() => cache.GetKeyCompressor(0, 0));
        Assert.Throws<InvalidOperationException>(() => cache.GetValueCompressor(0, 0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ForwardPass-level integration (model-gated, skipped when the fixture
    // model is absent — same convention as SnapKvTurboQuantTests)
    // ─────────────────────────────────────────────────────────────────────────

    [Fact]
    public void ForwardPass_KVarN_WithSnapKvEnv_ThrowsAtEnable()
    {
        var path = FindModelPath();
        if (path is null) return;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "256");
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);

            var ex = Assert.Throws<NotSupportedException>(() =>
                fwd.EnableTurboQuant(fp32WindowSize: 128, bits: 3, quantizer: TqQuantizer.KVarN));
            Assert.Contains("SnapKV", ex.Message);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    [Fact]
    public void ForwardPass_KVarN_PrefillAndDecode_StaysCoherent()
    {
        var path = FindModelPath();
        if (path is null) return;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", null);
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            // Window 128 (the KVarN minimum) so a ~384-token prompt ends with
            // two whole compressed tiles: promotions at totals 128 and 256.
            fwd.EnableTurboQuant(fp32WindowSize: 128, bits: 3, quantizer: TqQuantizer.KVarN);

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, approxTokenCount: 384);
            var logits = fwd.Prefill(tokens).ToArray();

            Assert.NotNull(fwd.TqCache);
            Assert.Equal(TqQuantizer.KVarN, fwd.TqCache!.Quantizer);
            Assert.Equal(0, fwd.TqCache.GetTqLength(0) % Tile);
            Assert.True(fwd.TqCache.GetTqLength(0) >= 2 * Tile,
                $"Expected >= 2 compressed tiles after a {tokens.Length}-token prefill, got tqLen={fwd.TqCache.GetTqLength(0)}.");

            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.True(float.IsFinite(logits[i]), $"Non-finite prefill logit at idx {i}: {logits[i]}");
                if (logits[i] < min) min = logits[i];
                if (logits[i] > max) max = logits[i];
            }
            Assert.True(max - min > 0.5f, $"Post-KVarN logit range collapsed to {min:F3}..{max:F3}.");

            var produced = new List<int>(4);
            for (int i = 0; i < 4; i++)
            {
                int next = Sampler.Greedy(logits);
                produced.Add(next);
                logits = fwd.Forward(next, tokens.Length + i).ToArray();
                for (int k = 0; k < logits.Length; k++)
                    Assert.True(float.IsFinite(logits[k]), $"Non-finite logit at decode step {i}, idx {k}");
            }
            int distinct = produced.Distinct().Count();
            Assert.True(distinct >= 2,
                $"Greedy decode under KVarN produced only {distinct} distinct token(s): [{string.Join(",", produced)}].");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Builds a KVarN-mode cache with <paramref name="seqLen"/> random Gaussian K/V rows, stashing the originals.</summary>
    private static (TurboQuantKvCache cache, float[][] keys, float[][] values) BuildKVarNCache(
        int seqLen, int window, int kvHeads, int headDim, int seed)
    {
        var cache = new TurboQuantKvCache(
            numLayers: 1, maxSeqLen: seqLen + Tile, numKvHeads: kvHeads, headDim: headDim,
            fp32WindowSize: window, quantizer: TqQuantizer.KVarN);

        var rng = new Random(seed);
        int kvDim = kvHeads * headDim;
        var keys = new float[seqLen][];
        var values = new float[seqLen][];
        for (int pos = 0; pos < seqLen; pos++)
        {
            keys[pos] = Gaussian(kvDim, rng);
            values[pos] = Gaussian(kvDim, rng);
            cache.Append(0, keys[pos], values[pos]);
            cache.IncrementPosition();
        }
        return (cache, keys, values);
    }

    /// <summary>
    /// Attention over the cache exactly as ForwardPass.TqAttention drives it:
    /// per-head RotateQuery → ComputeKScores over the compressed region → FP32
    /// dot products over the window → softmax → ComputeVAggregation (rotated
    /// domain, one deferred un-rotation inside) → FP32-window V accumulation
    /// in the original domain.
    /// </summary>
    private static unsafe float[] CacheAttention(TurboQuantKvCache cache, float[] q, int numQHeads)
    {
        int hd = cache.HeadDim;
        int hpkg = numQHeads / cache.NumKvHeads;
        int seqLen = cache.Length;
        int tqLen = cache.GetTqLength(0);
        float scale = 1f / MathF.Sqrt(hd);

        var outBuf = new float[numQHeads * hd];
        var scores = new float[seqLen];
        var rotated = new float[hd];

        for (int h = 0; h < numQHeads; h++)
        {
            int kvHead = h / hpkg;
            var qHead = q.AsSpan(h * hd, hd);
            cache.RotateQuery(0, kvHead, qHead, rotated);

            fixed (float* rotPtr = rotated)
            fixed (float* scoresPtr = scores)
                cache.ComputeKScores(0, kvHead, rotPtr, scale, scoresPtr);

            for (int t = tqLen; t < seqLen; t++)
            {
                float* kVec = cache.Fp32KeyAt(0, t) + kvHead * hd;
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += qHead[d] * kVec[d];
                scores[t] = dot * scale;
            }

            Softmax(scores);

            fixed (float* scoresPtr = scores)
            fixed (float* outPtr = &outBuf[h * hd])
                cache.ComputeVAggregation(0, kvHead, scoresPtr, outPtr);

            for (int t = tqLen; t < seqLen; t++)
            {
                float* vVec = cache.Fp32ValueAt(0, t) + kvHead * hd;
                float w = scores[t];
                for (int d = 0; d < hd; d++)
                    outBuf[h * hd + d] += w * vVec[d];
            }
        }
        return outBuf;
    }

    /// <summary>Plain FP32 GQA attention over per-position K/V rows of kvDim floats.</summary>
    private static float[] ReferenceAttention(
        float[][] keys, float[][] values, float[] q, int numQHeads, int numKvHeads, int hd)
    {
        int hpkg = numQHeads / numKvHeads;
        int seqLen = keys.Length;
        float scale = 1f / MathF.Sqrt(hd);

        var outBuf = new float[numQHeads * hd];
        var scores = new float[seqLen];

        for (int h = 0; h < numQHeads; h++)
        {
            int off = (h / hpkg) * hd;
            for (int t = 0; t < seqLen; t++)
            {
                float dot = 0f;
                for (int d = 0; d < hd; d++) dot += q[h * hd + d] * keys[t][off + d];
                scores[t] = dot * scale;
            }
            Softmax(scores);
            for (int t = 0; t < seqLen; t++)
                for (int d = 0; d < hd; d++)
                    outBuf[h * hd + d] += scores[t] * values[t][off + d];
        }
        return outBuf;
    }

    /// <summary>
    /// Materializes the cache's effective K/V: compressed positions reconstructed
    /// via KVarNCompressor.Decompress*Tile (read straight from the cache's tile
    /// storage), FP32-window positions taken from the stashed originals (the
    /// window stores exact copies).
    /// </summary>
    private static unsafe (float[][] keys, float[][] values) DecompressedKv(
        TurboQuantKvCache cache, float[][] origKeys, float[][] origValues)
    {
        int hd = cache.HeadDim;
        int kvHeads = cache.NumKvHeads;
        int kvDim = kvHeads * hd;
        int seqLen = cache.Length;
        int tqLen = cache.GetTqLength(0);
        var comp = new KVarNCompressor(hd);

        var keys = new float[seqLen][];
        var values = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            keys[t] = new float[kvDim];
            values[t] = new float[kvDim];
        }

        int numTiles = tqLen / Tile;
        var decK = new float[Tile * hd];
        var decV = new float[Tile * hd];
        for (int tileIdx = 0; tileIdx < numTiles; tileIdx++)
        {
            for (int head = 0; head < kvHeads; head++)
            {
                comp.DecompressKeyTile(
                    new ReadOnlySpan<byte>(cache.KeyTileAt(0, tileIdx, head), comp.KeyTileBytes), decK);
                comp.DecompressValueTile(
                    new ReadOnlySpan<byte>(cache.ValueTileAt(0, tileIdx, head), comp.ValueTileBytes), decV);
                for (int t = 0; t < Tile; t++)
                {
                    Array.Copy(decK, t * hd, keys[tileIdx * Tile + t], head * hd, hd);
                    Array.Copy(decV, t * hd, values[tileIdx * Tile + t], head * hd, hd);
                }
            }
        }

        for (int t = tqLen; t < seqLen; t++)
        {
            origKeys[t].CopyTo(keys[t], 0);
            origValues[t].CopyTo(values[t], 0);
        }
        return (keys, values);
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
            // Box-Muller; 1 - NextDouble() avoids log(0).
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return v;
    }

    private static float RelL2(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        double errSq = 0, refSq = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double e = actual[i] - reference[i];
            errSq += e * e;
            refSq += (double)reference[i] * reference[i];
        }
        return refSq > 0 ? (float)Math.Sqrt(errSq / refSq) : (float)Math.Sqrt(errSq);
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

    private static string? FindModelPath(string filename = "Qwen3-8B-Q4_K_M.gguf")
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models", filename);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int[] LongPrompt(GgufTokenizer tokenizer, int approxTokenCount)
    {
        const string seed =
            "The quick brown fox jumps over the lazy dog. " +
            "Sphinx of black quartz, judge my vow. " +
            "Pack my box with five dozen liquor jugs. ";
        var sb = new System.Text.StringBuilder();
        while (true)
        {
            sb.Append(seed);
            var attempt = tokenizer.Encode(sb.ToString());
            if (attempt.Count >= approxTokenCount) return attempt.ToArray();
            if (sb.Length > 100_000)
                throw new InvalidOperationException("Tokenizer not packing enough tokens — unexpected.");
        }
    }
}
