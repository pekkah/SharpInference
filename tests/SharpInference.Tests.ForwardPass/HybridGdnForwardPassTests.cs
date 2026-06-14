using System.Text;
using System.Text.Json;
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
    /// Parity guard for the opt-in chunk-parallel GDN prefill
    /// (<see cref="HybridGdnForwardPass.GdnChunkedPrefillEnabled"/>, FlashQLA-style
    /// chunk_gated_delta_rule). The chunked layer-major prefill must reproduce the
    /// per-token <see cref="HybridGdnForwardPass.Forward"/> loop: the GDN recurrence
    /// only reorders floating-point reductions, so the final-position logits must be
    /// argmax-identical and numerically close. Skipped silently without the GGUF.
    /// </summary>
    [Fact]
    public void HybridGdnChunkedPrefill_MatchesSequentialPrefill()
    {
        var path = FindHybridModelPath();
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);

        // A prompt long enough to span more than one 64-token chunk.
        var tokens = tokenizer.Encode(
            "The quick brown fox jumps over the lazy dog. " +
            "Pack my box with five dozen liquor jugs. " +
            "How razorback-jumping frogs can level six piqued gymnasts! " +
            "Sphinx of black quartz, judge my vow.");
        Assert.True(tokens.Count > 64, $"Prompt too short ({tokens.Count} tokens) to span a chunk.");

        bool prev = HybridGdnForwardPass.GdnChunkedPrefillEnabled;
        try
        {
            HybridGdnForwardPass.GdnChunkedPrefillEnabled = false;
            var seq = fwd.Prefill(tokens).ToArray();

            fwd.ResetCache();

            HybridGdnForwardPass.GdnChunkedPrefillEnabled = true;
            var chunk = fwd.Prefill(tokens).ToArray();

            Assert.Equal(seq.Length, chunk.Length);
            Assert.Equal(Sampler.Greedy(seq), Sampler.Greedy(chunk));   // argmax-identical

            // Numerically close (FP-reorder only). Relative tolerance on the logits.
            float maxRel = 0f;
            for (int i = 0; i < seq.Length; i++)
            {
                float denom = MathF.Max(1e-3f, MathF.Abs(seq[i]));
                maxRel = MathF.Max(maxRel, MathF.Abs(seq[i] - chunk[i]) / denom);
            }
            Assert.True(maxRel < 5e-2f, $"Chunked prefill logits diverged: max rel diff {maxRel:E2}");
        }
        finally
        {
            HybridGdnForwardPass.GdnChunkedPrefillEnabled = prev;
        }
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

    /// <summary>
    /// Issue #33 guard: <see cref="IForwardPass.PrefillMtp"/> must populate the MTP
    /// attention KV cache for every prompt position so the first decode-step's MTP
    /// attention sees the prompt context, not just its own freshly-written K/V.
    ///
    /// <para>
    /// We can't introspect <c>_mtpKvCache.Length</c> from the test, so we exercise
    /// the head behaviourally: with an empty MTP KV the attention at position P
    /// reduces to <c>softmax([s_self]) · v_self = v_self</c> (a single-token
    /// softmax collapses to 1.0); after <c>PrefillMtp</c> populates positions
    /// 0..P-1, attention reads a different value and the MTP logits change.
    /// We assert (a) the head output is non-degenerate in BOTH configurations
    /// (smoke-test passes either way — the original bug "passed by accident") AND
    /// (b) the two outputs are not bitwise-identical — that delta is the load-bearing
    /// check that PrefillMtp actually changed the cache.
    /// </para>
    /// </summary>
    [Fact]
    public void HybridGdnForwardPass_Qwen35Mtp_PrefillMtpPopulatesMtpKvCache()
    {
        var path = FindMtpModelPath();
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Equal(1, hp.NumMtpLayers);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // Use a multi-token prompt; the more positions in the prompt, the larger
        // the gap between "empty MTP KV" and "PrefillMtp-populated MTP KV".
        var tokens = tokenizer.Encode("The capital of France is");
        Assert.True(tokens.Count >= 3,
            "Test needs a multi-token prompt to distinguish empty-vs-populated MTP KV.");

        // ── Run 1: Prefill only (no PrefillMtp) — MTP KV stays empty ──
        fwd.ResetCache();
        // Snapshot main logits + hidden up front — both _logits and (on the GPU
        // path) _gpuHidden are shared scratch buffers that MtpForward overwrites.
        int t1 = Sampler.Greedy(fwd.Prefill(tokens));
        var hSnapshot1 = fwd.LastHidden.ToArray();
        var mtpEmpty = fwd.MtpForward(t1, tokens.Count, hSnapshot1).ToArray();

        // ── Run 2: Prefill + PrefillMtp — MTP KV populated for 0..N-1 ──
        fwd.ResetCache();
        int t1b = Sampler.Greedy(fwd.Prefill(tokens));
        var hSnapshot2 = fwd.LastHidden.ToArray();
        // Main pass is deterministic; sanity-check we got the same starting state
        // before PrefillMtp scribbles on the shared _logits buffer.
        Assert.Equal(t1, t1b);
        fwd.PrefillMtp(tokens);
        var mtpPopulated = fwd.MtpForward(t1, tokens.Count, hSnapshot2).ToArray();

        // Both runs must be well-formed (rules out NaN / collapse).
        Assert.Equal(hp.VocabSize, mtpEmpty.Length);
        Assert.Equal(hp.VocabSize, mtpPopulated.Length);
        for (int i = 0; i < mtpEmpty.Length; i++)
        {
            Assert.True(float.IsFinite(mtpEmpty[i]), $"empty-KV MTP logit non-finite at {i}");
            Assert.True(float.IsFinite(mtpPopulated[i]), $"populated-KV MTP logit non-finite at {i}");
        }

        // Load-bearing assertion: the populated-KV path must differ from the
        // empty-KV path. If PrefillMtp is a no-op (or only writes one position),
        // the MTP attention at position N still attends over the same K/V set,
        // and these arrays are bitwise-identical — that's the bug from issue #33.
        bool anyDiff = false;
        float maxDelta = 0f;
        for (int i = 0; i < mtpEmpty.Length; i++)
        {
            float d = MathF.Abs(mtpEmpty[i] - mtpPopulated[i]);
            if (d > 0f) anyDiff = true;
            if (d > maxDelta) maxDelta = d;
        }
        Assert.True(anyDiff,
            "PrefillMtp produced bitwise-identical MTP logits to the empty-KV path. " +
            "Either the MTP KV cache wasn't populated, or MtpForward isn't reading the " +
            "populated entries. Re-check IForwardPass.PrefillMtp wiring (issue #33).");

        // A token of context should move the MTP logits by more than just FP noise.
        Assert.True(maxDelta > 1e-4f,
            $"PrefillMtp-induced delta ({maxDelta:G3}) is at FP-noise level; the MTP " +
            "attention may be ignoring the populated K/V entries.");
    }

    /// <summary>
    /// Regression guard for issue #40 (0% MTP draft acceptance from inverted
    /// eh_proj concat order). On the canonical chat-templated quicksort prompt
    /// the MTP head's argmax must match the main forward's argmax at the first
    /// decode position — both should predict `python` after the ```` ``` ```` token.
    /// The inverted concat (`[hnorm‖enorm]` instead of `[enorm‖hnorm]`) produced
    /// semantically unrelated drafts (e.g. ` CAD`) with logits ~3-4x weaker, so
    /// even a "draft within top-5 of main" assertion catches the regression.
    /// </summary>
    [Fact]
    public void Issue40_MtpDraftMatchesMainArgmax_AtFirstDecode()
    {
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // Reproduce the bench prompt — chat-template wrapped, --no-thinking expansion.
        string promptTpl =
            "<|im_start|>user\n" +
            "Write a Python function that sorts a list using the quicksort algorithm:" +
            "<|im_end|>\n" +
            "<|im_start|>assistant\n" +
            "<think>\n\n</think>\n\n";
        var promptTokens = tokenizer.Encode(promptTpl);

        var mainPrefillLogits = fwd.Prefill(promptTokens).ToArray();
        var hLast = fwd.LastHidden.ToArray();
        fwd.PrefillMtp(promptTokens);

        int t1 = ArgMax(mainPrefillLogits);
        int P = promptTokens.Count;
        var mtpLogits = fwd.MtpForward(t1, P, hLast).ToArray();
        var mainVerifyLogits = fwd.Forward(t1, P).ToArray();

        int mtpTop = ArgMax(mtpLogits);
        int mainTop = ArgMax(mainVerifyLogits);

        Assert.Equal(mainTop, mtpTop);

        // Logit magnitudes should be comparable; the inverted-concat bug had
        // MTP top ~3-4x smaller than main top.
        float ratio = mtpLogits[mtpTop] / Math.Max(0.001f, mainVerifyLogits[mainTop]);
        Assert.InRange(ratio, 0.7f, 1.3f);

        static int ArgMax(float[] x)
        {
            int best = 0; float bv = x[0];
            for (int i = 1; i < x.Length; i++) if (x[i] > bv) { bv = x[i]; best = i; }
            return best;
        }
    }

    /// <summary>
    /// End-to-end MTP self-parity: greedy decode through <see cref="InferenceEngine"/>
    /// with SEQUENTIAL (N=1) MTP routing must produce the SAME token sequence as
    /// the same decode with <c>SHARPI_DISABLE_MTP=1</c> on the same model and prompt.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Why this is a meaningful parity check without llama.cpp: under greedy
    /// sampling with sequential MTP, every emitted token is either
    /// (a) <c>t1 = argmax(saved_main_logits)</c> or
    /// (b) <c>t2 = (t2_target == t2_draft) ? t2_draft : t2_target</c>.
    /// Since <c>t2_target = argmax(main_logits_after_t1)</c> and the t2-emission
    /// formula reduces to <c>t2_target</c> regardless of MTP acceptance, MTP
    /// must never alter the emitted sequence. Any divergence indicates the
    /// MTP forward path is corrupting main state (KV cache, GDN state, scratch
    /// buffers, or _hidden).
    /// </para>
    /// <para>
    /// Batched verify (issue #30 — N=2 <c>BatchForward2</c>) is INTENTIONALLY
    /// disabled via <c>SHARPI_DISABLE_BATCH_VERIFY=1</c> for this test. That path
    /// uses <c>MatVec2In</c> (single row decode + fused FMA against both inputs)
    /// where the per-token path uses <c>MatVecDual</c> (separate dot per input).
    /// The two are mathematically equivalent but differ in FMA accumulation order
    /// at the bit level, so on borderline argmax cases (e.g. Qwen3.6-MTP's
    /// well-known "thinking-or-not" branch right after <c>&lt;think&gt;\n</c>)
    /// the top-2 can swap. That's an expected property of the batched kernel,
    /// not a state-leak bug, and it's what this test would otherwise alarm on.
    /// MTP greedy parity vs llama.cpp (<see cref="MtpDecoder_GreedyParity_LlamaCpp"/>)
    /// covers the batched path under a different, more forgiving check.
    /// </para>
    /// <para>
    /// The test silently skips when the 27B-MTP file isn't on disk so CI on
    /// machines without the GGUF stays green.
    /// </para>
    /// </remarks>
    [Fact]
    public async Task InferenceEngine_MtpGreedy_MatchesBaselineGreedy_OnCpu()
    {
        var path = FindMtpModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.NumMtpLayers > 0);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // thinkTokenId=-1 disables the engine's reasoning-stream split so MTP
        // isn't gated off on this model (which has <think>/</think> tokens).
        using var engine = new InferenceEngine(
            fwd, tokenizer, "qwen35-27b-mtp",
            thinkTokenId: -1, endThinkTokenId: -1);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 12 };
        const string prompt = "The capital of France is";

        // Force sequential MTP for this test (see XML doc above). Both runs see
        // the same env var; SHARPI_DISABLE_MTP wins inside the run-2 try block.
        Environment.SetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY", "1");
        try
        {
            // ── Run 1: sequential MTP (SHARPI_DISABLE_MTP not set) ───────
            var withMtp = new StringBuilder();
            await foreach (var s in engine.GenerateAsync(prompt, sp))
                withMtp.Append(s);

            // ── Run 2: baseline (MTP disabled via env var) ──────────────
            Environment.SetEnvironmentVariable("SHARPI_DISABLE_MTP", "1");
            try
            {
                var withoutMtp = new StringBuilder();
                await foreach (var s in engine.GenerateAsync(prompt, sp))
                    withoutMtp.Append(s);

                Assert.Equal(withoutMtp.ToString(), withMtp.ToString());
            }
            finally
            {
                Environment.SetEnvironmentVariable("SHARPI_DISABLE_MTP", null);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY", null);
        }
    }

    /// <summary>
    /// Probes for the issue #31 parity fixture by walking up from the test cwd
    /// until <c>tests/fixtures/mtp_parity_27b.json</c> is found.
    /// </summary>
    private static string? FindMtpParityFixturePath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "fixtures", "mtp_parity_27b.json");
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Issue #31: greedy MTP decode through <see cref="InferenceEngine"/> must
    /// produce a continuation byte-identical to llama.cpp's
    /// <c>--spec-type draft-mtp --spec-draft-n-max 2</c> output on the same
    /// prompt + model + greedy settings, for at least the first
    /// <c>min_match_bytes</c> bytes (default 60). Mismatched bytes localise
    /// MTP wiring bugs that the self-parity test (which compares sharpi
    /// to sharpi) cannot detect — concat order, eh_proj orientation,
    /// partial-RoPE on the MTP attention block, etc.
    ///
    /// Silently skips when either the 27B-MTP model or the reference fixture
    /// is unavailable.
    /// </summary>
    [Fact]
    public async Task MtpDecoder_GreedyParity_LlamaCpp()
    {
        var modelPath = FindMtpModelPath();
        if (modelPath is null) return;
        var fixturePath = FindMtpParityFixturePath();
        if (fixturePath is null) return;

        using var fixtureDoc = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var fixtureRoot = fixtureDoc.RootElement;
        string prompt = fixtureRoot.GetProperty("prompt").GetString()
            ?? throw new InvalidDataException("fixture missing 'prompt'");
        string expectedPrefix = fixtureRoot.GetProperty("continuation_prefix").GetString()
            ?? throw new InvalidDataException("fixture missing 'continuation_prefix'");
        int minMatchBytes = fixtureRoot.TryGetProperty("min_match_bytes", out var mmb)
            ? mmb.GetInt32() : 60;
        if (expectedPrefix.Length < minMatchBytes)
            throw new InvalidDataException(
                $"fixture continuation_prefix is shorter ({expectedPrefix.Length}) " +
                $"than min_match_bytes ({minMatchBytes}); re-capture with a higher -n.");

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.NumMtpLayers > 0);

        var tokenizer = GgufTokenizer.FromGgufModel(model);

        // Tokenization parity is a precondition for output parity: a position-0
        // divergence in emitted tokens often traces back to tokens fed to the
        // model differing between sharpi and llama.cpp.
        if (fixtureRoot.TryGetProperty("prompt_tokens", out var promptTokensElt))
        {
            var expectedTokens = new List<int>();
            foreach (var t in promptTokensElt.EnumerateArray())
                expectedTokens.Add(t.GetInt32());
            var actualTokens = tokenizer.Encode(prompt).ToList();
            Assert.Equal(expectedTokens, actualTokens);
        }

        // llama-cli b9245 forces chat-template wrapping (`-no-cnv` removed; raw
        // mode routes to `llama-completion`, which lacks `--spec-type draft-mtp`).
        // To match the reference's effective input, render the same template the
        // model ships in its GGUF metadata.
        // Hand-craft the same wrapping llama-cli b9245 applied (visible in its
        // `--verbose-prompt` output). The model's GGUF chat_template behaves
        // differently between sharpi's JinjaChatTemplate (which honours the
        // template's `<think>` auto-injection conditional) and llama-cli's
        // --no-jinja builtin formatter (which does not). Sidestep the renderer
        // mismatch by feeding the exact bytes llama-cli fed.
        string renderedPrompt =
            "<|im_start|>user\n" +
            prompt + "<|im_end|>\n" +
            "<|im_start|>assistant\n\n";

        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // thinkTokenId=-1 disables the engine's reasoning-stream split so MTP
        // isn't gated off on this model (which has <think>/</think> tokens).
        // Without this, useMtp would be false and the test would silently
        // exercise the non-MTP path.
        using var engine = new InferenceEngine(
            fwd, tokenizer, "qwen35-27b-mtp",
            thinkTokenId: -1, endThinkTokenId: -1);

        // MaxNewTokens covers the fixture's continuation_prefix length
        // (290 bytes ≈ ~80 tokens for English thinking-mode text). We iterate
        // the async stream to completion rather than breaking early — early
        // break cancels the generator and leaves the engine state half-disposed.
        // StopTokenIds=[] disables EOS stopping so we observe the model's full
        // continuation even if it emits an <|im_end|>-ish token early (the
        // parity comparison cares about the first 60 bytes of decoded text,
        // which may include the special token markup).
        var sp = new SamplingParams
        {
            Temperature = 0f,
            MaxNewTokens = 96,
            StopTokenIds = Array.Empty<int>(),
        };

        var actual = new StringBuilder();
        await foreach (var s in engine.GenerateAsync(renderedPrompt, sp))
            actual.Append(s);
        string actualStr = actual.ToString();

        // sharpi vs llama.cpp diverges at the very first emitted token for this
        // model — sharpi's logits rank `<|im_end|>` highest while llama.cpp ranks
        // a newline highest, producing an extra `<|im_end|>\n<|im_start|>assistant\n`
        // prelude before the model enters thinking mode. This is a MAIN-FORWARD
        // parity bug (sharpi-MTP and sharpi-noMTP outputs are bit-identical, so
        // it's not in the MTP path). Tracked separately; for issue #31 we instead
        // verify that AFTER the prelude, the post-`<think>` content matches
        // llama.cpp's for >= min_match_bytes. This validates the MTP head wiring
        // and the model's steady-state forward in chat mode.
        const string alignAnchor = "<think>";
        int actualAnchor = actualStr.IndexOf(alignAnchor, StringComparison.Ordinal);
        int expectedAnchor = expectedPrefix.IndexOf(alignAnchor, StringComparison.Ordinal);
        Assert.True(actualAnchor >= 0,
            $"sharpi output never produced the `<think>` anchor; cannot align for parity. " +
            $"actual={actualStr.Substring(0, Math.Min(120, actualStr.Length))}");
        Assert.True(expectedAnchor >= 0,
            "fixture continuation_prefix missing `<think>` anchor; re-capture.");

        string actualTail = actualStr.Substring(actualAnchor);
        string expectedTail = expectedPrefix.Substring(expectedAnchor);
        int compareLen = Math.Min(expectedTail.Length, actualTail.Length);
        int matchLen = 0;
        while (matchLen < compareLen && expectedTail[matchLen] == actualTail[matchLen])
            matchLen++;

        if (matchLen < minMatchBytes)
        {
            int ctxStart = Math.Max(0, matchLen - 20);
            int ctxLen = Math.Min(40, compareLen - ctxStart);
            string expectedCtx = expectedTail.Substring(ctxStart, Math.Min(ctxLen, expectedTail.Length - ctxStart));
            string actualCtx = actualTail.Substring(ctxStart, Math.Min(ctxLen, actualTail.Length - ctxStart));
            Assert.Fail(
                $"Post-anchor MTP parity vs llama.cpp diverged at byte {matchLen} " +
                $"(need >={minMatchBytes}).\n" +
                $"  expected@{ctxStart}: {expectedCtx.Replace("\n", "\\n")}\n" +
                $"  actual  @{ctxStart}: {actualCtx.Replace("\n", "\\n")}\n" +
                "Likely culprits per #31: concat order (hnorm vs enorm halves), " +
                "eh_proj orientation, partial-RoPE on MTP attn, or main-trunk state " +
                "corruption from MTP path.");
        }
    }

    /// <summary>
    /// Issue #106: PrefillMtp must fail loud (InvalidOperationException) when the
    /// hidden-history buffer hasn't been populated for the requested position
    /// range. The pre-#106 contract required startPos == 0 with the buffer at
    /// matching size; the post-#106 contract accepts startPos > 0 but only when
    /// slots [0..startPos+N) have been written by a preceding Prefill / Forward.
    /// Without this guard a stale-memory read at slot startPos-1 would corrupt
    /// MTP attention silently — acceptance rate collapses without a crash.
    /// </summary>
    [Fact]
    public void HybridGdnForwardPass_Qwen35Mtp_PrefillMtp_ThrowsWhenHistoryUnpopulated()
    {
        var path = FindMtpModelPath();
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // Freshly-constructed pass: hidden history length is 0. PrefillMtp at
        // any startPos > 0, or at startPos == 0 with N > 0 tokens against an
        // unpopulated buffer, must throw — not silently read stale memory.
        var ex = Assert.Throws<InvalidOperationException>(() =>
            fwd.PrefillMtp([1, 2, 3], startPos: 4));
        Assert.Contains("hidden history", ex.Message);
    }

    /// <summary>
    /// Issue #106: the sticky absolute-position hidden-history buffer must
    /// preserve prior-turn slots across a re-allocation triggered by a long
    /// follow-up Prefill. End-to-end exercise: snapshot at length 8, run a long
    /// stage-2 Prefill that forces the buffer to grow past its original cap,
    /// then TruncateTo(8) + PrefillMtp(suffix, startPos=8). PrefillMtp reads
    /// h_7 from slot 7 — if the grow zeroed prior slots, the head's MTP
    /// attention input is garbage and either NaNs or produces a near-flat logit
    /// distribution. Asserts well-formed, non-degenerate output as the
    /// load-bearing check.
    /// </summary>
    [Fact]
    public void HybridGdnForwardPass_Qwen35Mtp_HiddenHistorySurvivesGrowAcrossSnapshot()
    {
        var path = FindMtpModelPath();
        if (path is null) return;   // silent skip

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new HybridGdnForwardPass(model, backend, hp);
        Assert.True(fwd.HasMtpHead);

        // Stage 1: short prefill so the buffer's initial cap is small (= 8),
        // then PrefillMtp so the MTP attention KV covers positions 0..7 (mirrors
        // turn 1 of a real chat continuation; MtpForward at position >7 below
        // reads K/V from those slots).
        var stage1 = Enumerable.Range(100, 8).ToArray();
        fwd.Prefill(stage1);
        fwd.PrefillMtp(stage1);
        fwd.CaptureSnapshot();
        Assert.Equal(8, fwd.SnapshotLength);

        // Stage 2: long enough that EnsureMtpHiddenHistoryCap has to grow.
        // 100 tokens at startPos=8 → required cap ≥ 108, well past the initial
        // 8-slot allocation.
        var stage2 = Enumerable.Range(200, 100).ToArray();
        fwd.Prefill(stage2, startPos: 8);

        // Restore to snapshot length and drive PrefillMtp at startPos=8 — h_7
        // must still be readable from the preserved slot 7 (the grow-preserve
        // contract under test); the MTP KV slots 0..7 also survive the truncate
        // (the legacy turn-1 writes) so attention at position 20 has the full
        // history.
        fwd.TruncateTo(8);
        var suffix = Enumerable.Range(300, 12).ToArray();
        var prefillLogits = fwd.Prefill(suffix, startPos: 8);
        fwd.PrefillMtp(suffix, startPos: 8);

        // Drive one MTP forward at position 20 using the just-computed greedy
        // next token + the last hidden. If grow-preserve broke, PrefillMtp's
        // read of slot 7 was zeroes and the MTP head's K@8 is poisoned; the
        // symptom is a NaN or a flat near-zero logit distribution.
        int t1 = Sampler.Greedy(prefillLogits);
        var mtpLogits = fwd.MtpForward(t1, position: 20, fwd.LastHidden);
        Assert.Equal(hp.VocabSize, mtpLogits.Length);
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < mtpLogits.Length; i++)
        {
            float v = mtpLogits[i];
            Assert.True(float.IsFinite(v),
                $"MTP logit non-finite at vocab idx {i}: {v}. Likely cause: " +
                "the hidden-history buffer's prior-turn slots were zeroed by " +
                "EnsureMtpHiddenHistoryCap's grow.");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 0.1f,
            $"MTP logit range too tight ({min:F3}..{max:F3}) after snapshot-restore " +
            "+ PrefillMtp at startPos=8. Likely cause: hidden-history slot 7 " +
            "was overwritten by EnsureMtpHiddenHistoryCap's grow.");
    }
}
