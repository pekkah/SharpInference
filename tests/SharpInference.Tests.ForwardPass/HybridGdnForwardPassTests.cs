using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Smoke tests for <see cref="HybridGdnForwardPass"/> (qwen35moe, Phase 4 wiring).
///
/// Like <see cref="CudaMoeTests"/> the heavy test is skipped silently when the
/// 22 GB qwen35moe GGUF isn't on disk. Until Phase-5 parity work lands, we only
/// assert pipeline well-formedness: finite logits, non-degenerate range, and at
/// least two distinct argmax tokens over a short decode window. Greedy output may
/// be garbled in v1 — the synth doesn't have to be coherent, only non-collapsed.
/// </summary>
public sealed class HybridGdnForwardPassTests
{
    /// <summary>
    /// Probes the two known disks where the qwen35moe GGUF could live. The model is
    /// 22 GB so it's typically on E:\models (the "large models" tier from
    /// reference_model_locations); we also check the project-local models/ for
    /// CI symlinks.
    /// </summary>
    private static string? FindHybridModelPath()
    {
        string[] absoluteCandidates =
        {
            @"E:\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        string[] relativeCandidates =
        {
            @"models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            @"models\Qwen3.6-35B-A3B-Q4_K_M.gguf",
        };
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var c in relativeCandidates)
            {
                var p = Path.Combine(dir, c);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Loads the qwen35moe GGUF on CPU, runs prefill + 4 greedy decode tokens, and
    /// asserts the output is well-formed: finite logits, a non-degenerate range,
    /// and at least two distinct argmax tokens across the decode window.
    ///
    /// The "two distinct tokens" assertion is the load-bearing check — when GDN
    /// recurrence, partial-RoPE, or the GLU attention gate is wired wrong, the
    /// logits collapse and every step picks the same token (the "all-EOS" failure
    /// mode <see cref="CudaMoeTests.CudaMoeForwardPass_ProducesWellFormedLogits"/>
    /// guards against on the CUDA side).
    /// </summary>
    [Fact]
    public void HybridGdnForwardPass_Qwen35Moe_ProducesWellFormedLogits()
    {
        var path = FindHybridModelPath();
        if (path is null) return;   // silent skip — same pattern as CudaMoeTests

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: this test should only fire on a hybrid GDN model with MoE.
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for qwen35moe model");
        Assert.NotNull(hp.Gdn);
        Assert.NotNull(hp.LayerTypes);
        Assert.True(hp.IsMoE, "Expected hp.IsMoE for qwen35moe model");

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);

        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);

        // Prefill — sequential T-step recurrence under the hood.
        var logits = fwd.Prefill(tokens);

        // Range + finiteness on the final prefill position's logits.
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            Assert.True(float.IsFinite(v), $"Non-finite logit at vocab idx {i}: {v}");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"Logit range too tight ({min:F3}..{max:F3}); hybrid GDN forward pass " +
            "is producing degenerate output. Likely culprits: GDN recurrence, conv1d " +
            "weight transpose, or partial-RoPE indexing.");

        // Greedy decode 4 tokens; assert at least 2 distinct argmaxes.
        var decoded = new List<int>(4);
        for (int i = 0; i < 4; i++)
        {
            int next = Sampler.Greedy(logits);
            decoded.Add(next);
            logits = fwd.Forward(next, tokens.Count + i);

            // Re-check finiteness at every decode step — catches drift in the
            // GDN scan state (which is destructively accumulated in place).
            for (int k = 0; k < logits.Length; k++)
                Assert.True(float.IsFinite(logits[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {logits[k]}");
        }

        int distinct = decoded.Distinct().Count();
        Assert.True(distinct >= 2,
            $"Greedy decode produced only {distinct} distinct token(s) across 4 steps " +
            $"({string.Join(",", decoded)}); the hybrid forward pass may be stuck in a " +
            "degenerate loop (logits collapsed onto a single output).");
    }

    /// <summary>
    /// Probes for the qwen35 27B-MTP GGUF in the small-models directory. Tracked
    /// separately from <see cref="FindHybridModelPath"/> because (a) qwen35-MTP
    /// is a different architecture (dense FFN + MTP head, not MoE) and (b) the
    /// file lives in <c>C:\p\sharpi\models</c> (17 GB), not <c>E:\models</c>.
    /// </summary>
    private static string? FindMtpModelPath()
    {
        string[] absoluteCandidates =
        {
            @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "Qwen3.6-27B-MTP-Q4_K_M.gguf");
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Smoke test for the MTP / NEXTN head on CPU <see cref="HybridGdnForwardPass"/>
    /// (issue #25). Asserts the head loads, <see cref="IForwardPass.LastHidden"/>
    /// is refreshed by the main forward pass, and <see cref="IForwardPass.MtpForward"/>
    /// returns finite, non-degenerate logits.
    ///
    /// We do NOT assert greedy parity vs llama.cpp here — that's a separate test
    /// requiring an external reference dump. The "no all-NaN, no flat logits"
    /// guard is enough to catch wholesale wiring bugs (wrong tensor names, bad
    /// concat order, missed norm).
    /// </summary>
    [Fact]
    public void HybridGdnForwardPass_Qwen35Mtp_MtpHeadProducesWellFormedLogits()
    {
        var path = FindMtpModelPath();
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for qwen35 hybrid GDN model");
        Assert.NotNull(hp.Gdn);
        Assert.NotNull(hp.LayerTypes);
        Assert.False(hp.IsMoE, "qwen35 27B-MTP is dense, not MoE");
        Assert.Equal(1, hp.NumMtpLayers);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);

        Assert.True(fwd.HasMtpHead,
            "HybridGdnForwardPass should have detected the MTP head at blk.NumLayers " +
            "and reported HasMtpHead == true.");

        // Run prefill on a tiny prompt to populate LastHidden.
        var tokens = tokenizer.Encode("Hello");
        Assert.NotEmpty(tokens);
        var mainLogits = fwd.Prefill(tokens);

        var lastHidden = fwd.LastHidden;
        Assert.Equal(hp.EmbeddingDim, lastHidden.Length);
        for (int i = 0; i < lastHidden.Length; i++)
            Assert.True(float.IsFinite(lastHidden[i]),
                $"LastHidden has a non-finite entry at index {i}: {lastHidden[i]}. " +
                "Pre-output-norm hidden capture is broken.");

        // The first decode-position token (= argmax of last prefill logits) feeds
        // into MtpForward at position tokens.Count to draft position tokens.Count+1.
        int t1 = Sampler.Greedy(mainLogits);
        var mtpLogits = fwd.MtpForward(t1, tokens.Count, lastHidden);

        Assert.Equal(hp.VocabSize, mtpLogits.Length);

        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < mtpLogits.Length; i++)
        {
            float v = mtpLogits[i];
            Assert.True(float.IsFinite(v),
                $"MTP logit non-finite at vocab idx {i}: {v}. " +
                "Likely culprits: eh_proj dequant, enorm/hnorm wiring, " +
                "MTP attention KV cache state, or shared_head_norm.");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"MTP logit range too tight ({min:F3}..{max:F3}); the head is " +
            "producing degenerate output. Likely culprits: concat order " +
            "(hnorm vs enorm halves), eh_proj weight orientation, or a " +
            "missed per-head norm in the MTP attention block.");
    }
}
