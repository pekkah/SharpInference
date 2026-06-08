using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CudaHybridForwardPass integration tests for Gemma 4 E4B. Exercises the
/// per-layer head_dim, dual-RoPE, SWA/full split, KV-share dispatch, post-attn
/// and post-ffw norms, PLE injection, GeluTanhMul FFN, layer_output_scale and
/// final-logit softcap across both tiers (GPU + CPU layers).
///
/// Silent-skip when CUDA isn't available or the GGUF isn't on disk, matching
/// the other Cuda* test files.
/// </summary>
public sealed class Gemma4CudaHybridForwardPassTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{ModelFile}",
            $@"C:\p\sharpi\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    // The largest -g value that keeps Gemma 4 E4B's KV-share sources on the CPU
    // side. Derived from `shared_kv_layers = 18` (E4B has 42 layers): the
    // own-KV sources are layers 22 and 23, so any split with -g <= 22 keeps
    // them on CPU together with the shared layers (24..41).
    private const int SafeGpuLayers = 22;

    /// <summary>
    /// Greedy decode coherence: finite logits + non-EOS first decode + at least
    /// 2 distinct tokens across a 4-step run. Cheapest signal that the trunk is
    /// wired correctly without requiring full bit-exact CPU parity.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CudaHybridForward_ProducesNonGarbageLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        var placement = new LayerPlacement(
            GpuLayers: SafeGpuLayers,
            CpuLayers: hp.NumLayers - SafeGpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 512);

        using var fwd = new CudaHybridForwardPass(model, gpu, hp, placement);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in post-prompt output.");

        int firstDecode = Argmax(logits);
        if (eosId >= 0)
            Assert.NotEqual(eosId, firstDecode);

        Span<int> decoded = stackalloc int[4];
        decoded[0] = firstDecode;
        int pos = tokens.Length;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
            for (int k = 0; k < step.Length; k++)
                Assert.True(float.IsFinite(step[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {step[k]}");
            decoded[i] = Argmax(step);
        }

        int distinct = 0;
        for (int i = 0; i < decoded.Length; i++)
        {
            bool seen = false;
            for (int j = 0; j < i; j++) if (decoded[j] == decoded[i]) { seen = true; break; }
            if (!seen) distinct++;
        }
        Assert.True(distinct >= 2,
            $"Hybrid greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"({string.Join(",", decoded.ToArray())}); Gemma 4 hybrid forward is degenerate.");
    }

    /// <summary>
    /// First-decode argmax parity vs CPU. Cumulative FP drift across both tiers
    /// and the GPU↔CPU hidden-state round-trip means downstream tokens can
    /// diverge; first-step argmax is the tightest practical signal that the
    /// CPU and hybrid paths agree on the trunk transformation.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CudaHybridForward_MatchesCpuArgmax()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        int cpuFirstArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
        {
            var cpuLogits = cpuFwd.Prefill(tokens);
            cpuFirstArgmax = Argmax(cpuLogits);
        }

        var placement = new LayerPlacement(
            GpuLayers: SafeGpuLayers,
            CpuLayers: hp.NumLayers - SafeGpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 512);

        using var hybridFwd = new CudaHybridForwardPass(model, gpu, hp, placement);
        var hybridLogits = hybridFwd.Prefill(tokens);
        int hybridFirstArgmax = Argmax(hybridLogits);

        if (cpuFirstArgmax != hybridFirstArgmax)
        {
            var msg = new System.Text.StringBuilder();
            msg.AppendLine($"CPU first argmax: {cpuFirstArgmax}");
            msg.AppendLine($"Hybrid first argmax: {hybridFirstArgmax}");
            int[] top3 = TopK(hybridLogits, 3);
            msg.Append("Hybrid top-3: ");
            for (int i = 0; i < top3.Length; i++)
                msg.Append($"{top3[i]}({hybridLogits[top3[i]]:F2}) ");
            Assert.Fail(msg.ToString());
        }
    }

    /// <summary>
    /// Strict cross-backend logit parity for the E4B hybrid split vs the independent CPU
    /// <see cref="ForwardPass"/> reference. This is the oracle the argmax-only
    /// <see cref="Gemma4_E4B_CudaHybridForward_MatchesCpuArgmax"/> can't give: it pins the
    /// V-norm consistency between the two hybrid tiers. llama.cpp gemma4.cpp:227 V-norms
    /// every KV-owning gemma4 layer (E4B too); the CPU half (CpuLayerGemma4) always did, but
    /// the GPU half used to gate V-norm on AttentionKEqV — so an E4B split V-normed its CPU
    /// layers and not its GPU layers (split-internal inconsistency), diverging from the CPU
    /// reference by ~14 logits even while first-argmax held. Q8_0 is argmax-stable but not
    /// bit-exact, so we assert a single prefill: argmax + top-5 overlap + a loose maxAbs bound.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CpuMatchesCudaHybridLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        var placement = new LayerPlacement(
            GpuLayers: SafeGpuLayers,
            CpuLayers: hp.NumLayers - SafeGpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 512);

        float[] hybridLogits;
        using (var hybridFwd = new CudaHybridForwardPass(model, gpu, hp, placement))
            hybridLogits = hybridFwd.Prefill(tokens).ToArray();

        float[] cpuLogits;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
            cpuLogits = cpuFwd.Prefill(tokens).ToArray();

        Assert.Equal(hybridLogits.Length, cpuLogits.Length);

        float maxAbs = 0f;
        for (int i = 0; i < cpuLogits.Length; i++)
        {
            float d = MathF.Abs(cpuLogits[i] - hybridLogits[i]);
            if (d > maxAbs) maxAbs = d;
        }

        int cpuArgmax = Argmax(cpuLogits);
        int hybridArgmax = Argmax(hybridLogits);
        var cpuTop5 = TopK(cpuLogits, 5);
        var hybridTop5 = TopK(hybridLogits, 5);
        int overlap = 0;
        foreach (var t in cpuTop5) if (Array.IndexOf(hybridTop5, t) >= 0) overlap++;

        Assert.True(cpuArgmax == hybridArgmax,
            $"CPU↔CUDA-hybrid E4B argmax disagree: CPU={cpuArgmax} hybrid={hybridArgmax}, maxAbs={maxAbs:F3}, " +
            $"CPU top5=[{string.Join(",", cpuTop5)}] hybrid top5=[{string.Join(",", hybridTop5)}]. " +
            "A V-norm / per-layer-KV inconsistency across the hybrid tiers, not Q8_0 noise.");
        Assert.True(overlap >= 4,
            $"CPU↔CUDA-hybrid E4B top-5 overlap only {overlap}/5 (maxAbs={maxAbs:F3}); " +
            $"CPU=[{string.Join(",", cpuTop5)}] hybrid=[{string.Join(",", hybridTop5)}].");
        Assert.True(maxAbs < 5.0f,
            $"CPU↔CUDA-hybrid E4B logit maxAbs {maxAbs:F3} exceeds the structural bound (Q8_0 cross-backend " +
            "noise stays well under 5; this magnitude indicates a tier-inconsistent V-norm).");
    }

    /// <summary>
    /// Validation gate: hybrid splits that would cross a KV-share source/dependent
    /// tier boundary must be rejected up front with an actionable message.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CudaHybridForward_RejectsCrossTierKvShare()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // -g 30 places own-KV source layers 22, 23 on the GPU but their shared-KV
        // dependents (layers 24..29 on GPU, 30..41 on CPU) live across the tier
        // boundary — must reject.
        var bad = new LayerPlacement(
            GpuLayers: 30,
            CpuLayers: hp.NumLayers - 30,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 512);

        var ex = Assert.Throws<NotSupportedException>(() =>
        {
            using var fwd = new CudaHybridForwardPass(model, gpu, hp, bad);
        });
        Assert.Contains("KV", ex.Message);
    }

    private static int[] TopK(ReadOnlySpan<float> logits, int k)
    {
        var result = new int[k];
        var taken = new System.Collections.Generic.HashSet<int>();
        for (int ki = 0; ki < k; ki++)
        {
            int best = -1; float bestVal = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (taken.Contains(i)) continue;
                if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
            }
            result[ki] = best;
            taken.Add(best);
        }
        return result;
    }
}
