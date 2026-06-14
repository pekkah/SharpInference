using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// CPU forward-pass smoke tests for Gemma 4 E4B (Phase 3 of the gemma4 plan).
/// Confirms that the per-layer head_dim refactor, dual-RoPE table, post-attn/post-ffn
/// norms, layer_output_scale, SWA windowed attention, KV-share dispatch, GeluTanhMul,
/// and final-logit softcap together produce a non-garbage decode stream on the real
/// 8.2 GB unsloth GGUF. Also runs a Qwen3-MoE coherence check to confirm the same
/// refactor didn't regress non-Gemma 4 models.
/// </summary>
public sealed class Gemma4CpuForwardPassTests
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            Path.Combine(@"E:\models", fileName),
            Path.Combine(@"C:\p\sharpi\models", fileName),
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Gemma4_E4B_CpuForward_ProducesNonGarbageLogits()
    {
        var path = FindModelPath("gemma-4-E4B-it-Q8_0.gguf");
        if (path is null) return;   // silent skip — same pattern as HybridGdnForwardPassTests

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: this test must only fire against an actual gemma4 GGUF where the
        // Phase-1 hyperparam fields are all populated. Catches accidental file swaps.
        Assert.NotNull(hp.IsSwaLayer);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.KvSourceLayer);
        Assert.Equal(hp.NumLayers, hp.IsSwaLayer!.Count);
        Assert.Equal(hp.NumLayers, hp.LayerHeadDim!.Count);
        Assert.Equal(hp.NumLayers, hp.KvSourceLayer!.Count);
        Assert.Equal(FfnActivation.GeluApprox, hp.FfnActivation);
        Assert.True(hp.FinalLogitSoftcap > 0f);
        Assert.True(hp.EmbeddingScale > 1f);

        // Gemma uses a SentencePiece vocab that GgufTokenizer (BPE / CodeGen) doesn't
        // currently load. Drive the smoke test with the model's BOS + an arbitrary
        // mid-vocab token sequence so the forward-pass path is still exercised end-to-end.
        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        // Arbitrary mid-vocab token IDs to exercise prefill + multi-position attention.
        // With Phase 4 PLE wired the model is expected to produce a varied decode stream,
        // so variety is asserted via AssertCoherentDecode(requireVariety: true).
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        Assert.True(hp.HasPerLayerTokenEmbd,
            "gemma4 E4B GGUF must carry per_layer_token_embd for the PLE smoke test.");

        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        AssertCoherentDecode(fwd, eosId, logits, tokens.Length, hp.VocabSize, requireVariety: true);
    }

    [Fact]
    public void Gemma4_E4B_Q4_0_CpuForward_LoadsWithAbsentSharedKvNorm()
    {
        // #211: Google's official E4B QAT q4_0 GGUF omits attn_k/attn_v/attn_k_norm for the
        // 18 shared-KV tail layers (the Q8_0 ships dead, never-read copies). The CPU loader
        // used to require attn_k_norm unconditionally and threw
        //   "Missing bias tensor: blk.24.attn_k_norm.weight".
        // It now skips the K-norm for KV-share layers (where ApplyQkNormLayer passes k=null),
        // so the file loads and decodes coherently.
        var path = FindModelPath("gemma-4-E4B_q4_0-it.gguf");
        if (path is null) return;   // silent skip — file only present on the dev box

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Regression precondition: this file must actually have a KV-share layer whose
        // attn_k_norm is genuinely absent (else the test wouldn't exercise the fix). Confirm
        // attn_q_norm is still present for that layer — only the shared layers' K-norm is omitted.
        Assert.NotNull(hp.KvSourceLayer);
        int sharedLayer = -1;
        for (int i = 0; i < hp.KvSourceLayer!.Count; i++)
            if (hp.KvSourceLayer[i] >= 0) { sharedLayer = i; break; }
        Assert.True(sharedLayer >= 0, "expected a KV-share layer in the E4B q4_0 GGUF — wrong file?");
        Assert.Null(model.FindTensor($"blk.{sharedLayer}.attn_k_norm.weight"));
        Assert.NotNull(model.FindTensor($"blk.{sharedLayer}.attn_q_norm.weight"));

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        using var backend = new CpuBackend();
        // Pre-#211 this constructor threw on blk.24.attn_k_norm.weight.
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);
        AssertCoherentDecode(fwd, eosId, logits, tokens.Length, hp.VocabSize, requireVariety: true);
    }

    [Fact]
    public void Gemma4_E4B_CpuForward_PleProducesVariety()
    {
        // Phase 4 acceptance signal: with PLE correctly injected, greedy decode over
        // four tokens must produce ≥ 2 distinct token IDs. A degenerate single-token
        // loop (e.g. <pad> spam) signals broken PLE wiring (wrong row layout, missing
        // norm offset, wrong scaling, etc.).
        var path = FindModelPath("gemma-4-E4B-it-Q8_0.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        if (!hp.HasPerLayerTokenEmbd) return;

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var logits = fwd.Prefill(tokens);

        Span<int> decoded = stackalloc int[4];
        decoded[0] = Argmax(logits);
        int pos = tokens.Length;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
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
            $"PLE-on greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"({string.Join(",", decoded.ToArray())}); Phase 4 PLE injection is not producing variety.");
    }

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    [Fact]
    public void NonGemma_Qwen3_CpuForward_StillCoherent()
    {
        // Regression guard: the Phase-3 per-layer-head-dim refactor must NOT change
        // behaviour on plain (non-gemma4) models. We pick a small dense Qwen3 GGUF
        // that's already known-coherent on the canonical CPU Forward path.
        string[] candidates =
        {
            "Qwen3-1.7B-Instruct-Q4_K_M.gguf",
            "Qwen3-0.6B-Instruct-Q4_K_M.gguf",
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        };
        string? path = null;
        foreach (var f in candidates)
        {
            path = FindModelPath(f);
            if (path is not null) break;
        }
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);
        Assert.Null(hp.IsSwaLayer);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp);

        var tokens = tokenizer.Encode("The capital of France is");
        Assert.NotEmpty(tokens);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        AssertCoherentDecode(fwd, tokenizer.EosTokenId, logits, tokens.Count, hp.VocabSize, requireVariety: true);
    }

    /// <summary>
    /// Inline twin of <c>VulkanShaderTests.AssertHybridForwardPassProducesCoherentDecode</c>
    /// for <see cref="ForwardPass"/>. Asserts finite logits, argmax of the post-prompt
    /// step is NOT EOS, and (when <paramref name="requireVariety"/> is true) a 4-token
    /// greedy decode produces at least two distinct tokens — the load-bearing check
    /// per the feedback_forward_pass_tests memory entry. The variety check is skipped
    /// for Gemma 4 Phase 3 (no PLE) because the model can legitimately produce a single
    /// repeated token until Phase 4 adds the Per-Layer-Embedding injection.
    /// </summary>
    private static void AssertCoherentDecode(
        SharpInference.Engine.ForwardPass fwd, int eosTokenId, ReadOnlySpan<float> postPromptLogits,
        int promptLen, int vocabSize, bool requireVariety)
    {
        Assert.Equal(vocabSize, postPromptLogits.Length);

        int nonFinite = 0;
        for (int i = 0; i < postPromptLogits.Length; i++)
            if (!float.IsFinite(postPromptLogits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in post-prompt output.");

        int firstDecodeToken = Argmax(postPromptLogits);
        if (eosTokenId >= 0)
            Assert.NotEqual(eosTokenId, firstDecodeToken);

        Span<int> decoded = stackalloc int[4];
        decoded[0] = firstDecodeToken;
        int pos = promptLen;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
            for (int k = 0; k < step.Length; k++)
                Assert.True(float.IsFinite(step[k]),
                    $"Non-finite logit at decode step {i}, vocab idx {k}: {step[k]}");
            decoded[i] = Argmax(step);
        }

        if (eosTokenId >= 0)
        {
            int eosCount = 0;
            for (int i = 0; i < decoded.Length; i++)
                if (decoded[i] == eosTokenId) eosCount++;
            Assert.True(eosCount < decoded.Length,
                $"All {decoded.Length} greedy-decoded tokens were EOS — output is degenerate.");
        }

        if (requireVariety)
        {
            int distinct = 0;
            for (int i = 0; i < decoded.Length; i++)
            {
                bool seen = false;
                for (int j = 0; j < i; j++) if (decoded[j] == decoded[i]) { seen = true; break; }
                if (!seen) distinct++;
            }
            Assert.True(distinct >= 2,
                $"Greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
                $"({string.Join(",", decoded.ToArray())}); forward pass may be stuck in a degenerate loop.");
        }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
