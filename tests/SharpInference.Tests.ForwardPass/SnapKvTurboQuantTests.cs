using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.TurboQuant;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Tests for SnapKV + TurboQuant composition (issue #60 Phase 1 / sub-issue #68).
///
/// Covers:
/// <list type="bullet">
///   <item><see cref="TurboQuantKvCache.Compact"/> round-trip preserves the
///         decompressed survivors with high cosine similarity vs the original
///         input — the dequant→requant pass on previously-TQ positions is the
///         most lossy step, and we verify it stays inside the codec's
///         steady-state error budget.</item>
///   <item>End-to-end CPU ForwardPass with both SnapKV and TurboQuant enabled:
///         the TQ cache shrinks to the budget after a long-prompt prefill,
///         decode stays coherent, and env-unset disables eviction.</item>
/// </list>
/// </summary>
public sealed class SnapKvTurboQuantTests
{
    private const int HeadDim = 128;
    private const int NumLayers = 1;
    private const int NumKvHeads = 1;

    // Qwen3-8B has headDim=128 which is in the shipped TurboQuant codebook set
    // (3-bit @ d=128 / 256). SmolLM2 has headDim=64 so it can't be used here.
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
                throw new InvalidOperationException("Tokenizer not packing enough — unexpected for Qwen3-8B.");
        }
    }

    /// <summary>
    /// Cosine similarity over the per-position decompressed K vector. After a
    /// single quant→dequant pass the TQ codec preserves direction with high
    /// fidelity (the 3-bit codebook achieves ≥0.95 cosine on Gaussian inputs);
    /// after a Compact-time dequant→requant→dequant round trip the loss
    /// compounds but stays in the same band for codes that don't fall on a
    /// boundary on the requantize pass.
    /// </summary>
    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        if (denom <= 0) return 0f;
        return (float)(dot / denom);
    }

    [Fact]
    public unsafe void Compact_RoundTrip_PreservesKeptValues()
    {
        const int totalPositions = 256;
        const int keepCount = 128;
        const int fp32Window = 64;
        const int maxSeqLen = totalPositions + 32;

        using var cache = new TurboQuantKvCache(
            numLayers: NumLayers,
            maxSeqLen: maxSeqLen,
            numKvHeads: NumKvHeads,
            headDim: HeadDim,
            fp32WindowSize: fp32Window,
            bits: 3);

        var rng = new Random(20260528);
        // Stash the input K and V vectors so we can compare post-compaction.
        var inputKeys = new float[totalPositions][];
        var inputValues = new float[totalPositions][];

        for (int pos = 0; pos < totalPositions; pos++)
        {
            var k = new float[HeadDim];
            var v = new float[HeadDim];
            // Normalized random unit vectors — matches what RMS-normed
            // attention K/V look like in practice.
            double normK = 0, normV = 0;
            for (int d = 0; d < HeadDim; d++)
            {
                k[d] = (float)(rng.NextDouble() * 2 - 1);
                v[d] = (float)(rng.NextDouble() * 2 - 1);
                normK += k[d] * k[d];
                normV += v[d] * v[d];
            }
            float invK = (float)(1.0 / Math.Sqrt(normK));
            float invV = (float)(1.0 / Math.Sqrt(normV));
            for (int d = 0; d < HeadDim; d++) { k[d] *= invK; v[d] *= invV; }

            inputKeys[pos] = k;
            inputValues[pos] = v;
            cache.Append(0, k, v);
            cache.IncrementPosition();
        }

        Assert.Equal(totalPositions, cache.Length);
        Assert.True(cache.GetTqLength(0) > 0, "Steady-state TQ region should be non-empty");

        // Keep a striped subset: every other position. This stresses both TQ
        // and FP32 source classes since the keep set spans the entire range.
        var keep = new int[keepCount];
        for (int i = 0; i < keepCount; i++) keep[i] = i * 2;
        cache.Compact(keep, totalPositions);

        Assert.Equal(keepCount, cache.Length);

        // For each kept position, decompress its survivor and compare to the
        // original input. We accept ≥0.95 cosine similarity — the same
        // threshold the needle-in-a-haystack tests use as the codec's effective
        // direction-preservation guarantee at 3-bit on dim=128.
        var compressor = cache.GetKeyCompressor(0, 0);
        var valueCompressor = cache.GetValueCompressor(0, 0);
        int tqLenNew = cache.GetTqLength(0);
        int tileSize = cache.FastScanTileSize;
        int numFullTilesNew = tqLenNew / tileSize;
        int stagingCountNew = tqLenNew % tileSize;

        float minKeyCos = 1f, minValCos = 1f;
        var decompKey = new float[HeadDim];
        var decompVal = new float[HeadDim];

        for (int i = 0; i < keepCount; i++)
        {
            int srcPos = keep[i];

            if (i < tqLenNew)
            {
                // Survivor landed in the TQ region — pull from tile or staging.
                ReadOnlySpan<byte> keyBlock;
                ReadOnlySpan<byte> valBlock;
                var keyBlockArr = new byte[cache.TqBlockSize];
                var valBlockArr = new byte[cache.TqBlockSize];

                if (i < numFullTilesNew * tileSize)
                {
                    int tileIdx = i / tileSize;
                    int slot = i % tileSize;
                    // Unpack via dequant of the tile's nibble for this (slot, dim)
                    // pair. Direct K-tile reconstruction.
                    byte* kTile = cache.KeyTileAt(0, tileIdx, 0);
                    byte* vTile = cache.ValueTileAt(0, tileIdx, 0);

                    // K-tile reconstruction: norm + per-dim nibble
                    keyBlockArr[0] = kTile[slot * 2];
                    keyBlockArr[1] = kTile[slot * 2 + 1];
                    byte* kCodes = kTile + 64; // FastScan.NormBytesPerTile
                    int byteInRow = slot & 15;
                    bool highNibble = slot >= 16;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        byte pair = kCodes[d * 16 + byteInRow];
                        int code = highNibble ? (pair >> 4) & 0x0F : pair & 0x0F;
                        BitPacking.PackBits3(keyBlockArr, TurboQuantOps.IndicesOffset, d, code & 0x07);
                    }

                    // V-tile reconstruction: norm + per-position row of dim/2 bytes
                    valBlockArr[0] = vTile[slot * 2];
                    valBlockArr[1] = vTile[slot * 2 + 1];
                    byte* vCodes = vTile + 64;
                    byte* vRow = vCodes + (long)slot * (HeadDim / 2);
                    for (int d = 0; d < HeadDim; d += 2)
                    {
                        byte pair = vRow[d / 2];
                        BitPacking.PackBits3(valBlockArr, TurboQuantOps.IndicesOffset, d,     pair & 0x07);
                        BitPacking.PackBits3(valBlockArr, TurboQuantOps.IndicesOffset, d + 1, (pair >> 4) & 0x07);
                    }

                    keyBlock = keyBlockArr;
                    valBlock = valBlockArr;
                }
                else
                {
                    int stagingIdx = i - numFullTilesNew * tileSize;
                    keyBlock = new ReadOnlySpan<byte>(cache.KeyStagingBlockAt(0, stagingIdx, 0), cache.TqBlockSize);
                    valBlock = new ReadOnlySpan<byte>(cache.ValueStagingBlockAt(0, stagingIdx, 0), cache.TqBlockSize);
                }

                compressor.Decompress(keyBlock, decompKey);
                valueCompressor.Decompress(valBlock, decompVal);
            }
            else
            {
                int fp32SlotDst = i - tqLenNew;
                // FP32 path — should round-trip exactly for positions that
                // were originally FP32-resident, with small loss for those
                // promoted from TQ.
                var fp32K = cache.Fp32KeyAt(0, tqLenNew + fp32SlotDst);
                var fp32V = cache.Fp32ValueAt(0, tqLenNew + fp32SlotDst);
                new ReadOnlySpan<float>(fp32K, HeadDim).CopyTo(decompKey);
                new ReadOnlySpan<float>(fp32V, HeadDim).CopyTo(decompVal);
            }

            float kc = CosineSimilarity(decompKey, inputKeys[srcPos]);
            float vc = CosineSimilarity(decompVal, inputValues[srcPos]);
            if (kc < minKeyCos) minKeyCos = kc;
            if (vc < minValCos) minValCos = vc;
        }

        Assert.True(minKeyCos >= 0.95f,
            $"Min K cosine across survivors {minKeyCos:F4} < 0.95 — Compact path is degrading TQ accuracy more than the codec's steady-state error budget.");
        Assert.True(minValCos >= 0.95f,
            $"Min V cosine across survivors {minValCos:F4} < 0.95 — Compact path is degrading TQ accuracy more than the codec's steady-state error budget.");
    }

    [Fact]
    public void SnapKvTqPath_LongPrompt_CacheShrinksToBudget()
    {
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 256;
        const int promptTargetLen = 384;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            // FP32 window 64 ensures most of the prompt ends up TQ-compressed
            // by the end of prefill, exercising the TQ-side scoring + the
            // promotion-on-overflow path in Compact.
            fwd.EnableTurboQuant(fp32WindowSize: 64, bits: 3);

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, promptTargetLen);
            Assert.True(tokens.Length >= budget + SnapKvSelector.DefaultWindow,
                $"Prompt too short ({tokens.Length}) — SnapKV gate requires it to exceed budget + window.");

            _ = fwd.Prefill(tokens).ToArray();

            Assert.NotNull(fwd.TqCache);
            Assert.Equal(budget, fwd.TqCache!.Length);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    [Fact]
    public void SnapKvTqPath_DecodeStaysCoherent()
    {
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 256;
        const int promptTargetLen = 384;

        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new Engine.ForwardPass(model, backend, hp);
            fwd.EnableTurboQuant(fp32WindowSize: 64, bits: 3);

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, promptTargetLen);
            var logits = fwd.Prefill(tokens).ToArray();

            // Logits must be finite and have non-trivial spread.
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                Assert.True(float.IsFinite(logits[i]),
                    $"Non-finite prefill logit at idx {i}: {logits[i]} — SnapKV+TQ composition is broken.");
                if (logits[i] < min) min = logits[i];
                if (logits[i] > max) max = logits[i];
            }
            Assert.True(max - min > 0.5f,
                $"Post-(SnapKV+TQ) logit range collapsed to {min:F3}..{max:F3}.");

            // Decode 4 tokens; assert finite + ≥2 distinct argmaxes.
            var produced = new List<int>(4);
            for (int i = 0; i < 4; i++)
            {
                int next = Sampler.Greedy(logits);
                produced.Add(next);
                logits = fwd.Forward(next, tokens.Length + i).ToArray();
                for (int k = 0; k < logits.Length; k++)
                    Assert.True(float.IsFinite(logits[k]),
                        $"Non-finite logit at decode step {i}, idx {k}: {logits[k]}");
            }
            int distinct = produced.Distinct().Count();
            Assert.True(distinct >= 2,
                $"Greedy decode under SnapKV+TQ produced only {distinct} distinct token(s): " +
                $"[{string.Join(",", produced)}].");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    [Fact]
    public void SnapKvTqPath_EnvUnset_NoEviction()
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
            fwd.EnableTurboQuant(fp32WindowSize: 64, bits: 3);

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var tokens = LongPrompt(tokenizer, 384);
            _ = fwd.Prefill(tokens);

            // Env unset → no eviction → cache holds the full prompt.
            Assert.NotNull(fwd.TqCache);
            Assert.Equal(tokens.Length, fwd.TqCache!.Length);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }
}
