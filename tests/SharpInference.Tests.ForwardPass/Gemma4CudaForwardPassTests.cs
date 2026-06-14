using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Phase-8 CUDA forward-pass integration tests for Gemma 4 E4B. Confirms the
/// load-bearing trunk wiring — per-layer head_dim variance, dual-RoPE selection,
/// KV-share dispatch, SWA / full Attention split, post-attn / post-ffn norms,
/// PLE injection, GeluTanhMul FFN, layer_output_scale, final-logit softcap —
/// together produce a non-garbage decode stream on the real 8.2 GB unsloth GGUF
/// AND that the CUDA result tracks the CPU reference at first-decode argmax.
///
/// Silent-skip pattern: if CUDA isn't available OR the GGUF isn't present on
/// disk these tests no-op, mirroring the rest of the Cuda* test files.
/// </summary>
public sealed class Gemma4CudaForwardPassTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath() => FindModelPath(ModelFile);

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{fileName}",
            $@"C:\p\sharpi\models\{fileName}",
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

    [Fact]
    public void Gemma4_E4B_CudaForward_ProducesNonGarbageLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: the test only makes sense against a real gemma4 GGUF.
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.IsSwaLayer);
        Assert.True(hp.HasPerLayerTokenEmbd);
        Assert.True(hp.FinalLogitSoftcap > 0f);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        // Finite logits + non-EOS first decode + ≥2 distinct tokens across 4 steps.
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
            $"CUDA greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"({string.Join(",", decoded.ToArray())}); Gemma 4 CUDA forward integration is degenerate.");

        if (eosId >= 0)
        {
            int eosCount = 0;
            for (int i = 0; i < decoded.Length; i++)
                if (decoded[i] == eosId) eosCount++;
            Assert.True(eosCount < decoded.Length,
                $"All {decoded.Length} greedy-decoded tokens were EOS — Gemma 4 CUDA output is degenerate.");
        }
    }

    [Fact]
    public void Gemma4_E4B_CudaForward_MatchesCpuArgmax()
    {
        // The Phase 8 keystone: greedy argmax of the first decoded token after
        // a 9-token prefill must match between CPU and CUDA. Cumulative FP drift
        // across 42 layers + softcap means later tokens can diverge; the first
        // step is the tightest parity signal we can demand without bit-exactness.
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        // CPU reference.
        int cpuFirstArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
        {
            var cpuLogits = cpuFwd.Prefill(tokens);
            cpuFirstArgmax = Argmax(cpuLogits);
        }

        // CUDA path.
        using var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);
        var cudaLogits = cudaFwd.Prefill(tokens);
        int cudaFirstArgmax = Argmax(cudaLogits);

        // Diagnostic: top-3 each side so a divergence message is actionable.
        if (cpuFirstArgmax != cudaFirstArgmax)
        {
            var msg = new System.Text.StringBuilder();
            msg.AppendLine($"CPU first argmax: {cpuFirstArgmax}");
            msg.AppendLine($"CUDA first argmax: {cudaFirstArgmax}");

            int[] cudaTop3 = TopK(cudaLogits, 3);
            msg.Append("CUDA top-3: ");
            for (int i = 0; i < cudaTop3.Length; i++)
                msg.Append($"{cudaTop3[i]}({cudaLogits[cudaTop3[i]]:F2}) ");
            Assert.Fail(msg.ToString());
        }
    }

    /// <summary>
    /// Long-decode variety check vs the "degenerate 2-cycle" failure mode that
    /// a 4-step variety test misses: an attention scale mismatch (kernel
    /// applies 1/sqrt(head_dim) but Gemma 4 wants 1.0) produces a correct
    /// first-decode argmax then collapses into a 2-token repeat ("the of the
    /// of of of..."). Real coherent output produces ≥8 distinct tokens in a
    /// 16-step decode; the bug produces ≤3. The first 4 tokens are also
    /// checked against the CPU reference for exact-match, which catches the
    /// failure before drift kicks in.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CudaForward_LongDecodeIsCoherent()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 }; // "The capital of France is"

        // CPU reference for the first 4 decode tokens — pre-drift the argmax
        // should match across CPU and CUDA.
        var cpuFirst4 = new int[4];
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
        {
            var logits = cpuFwd.Prefill(tokens);
            cpuFirst4[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < cpuFirst4.Length; i++)
            {
                var step = cpuFwd.Forward(cpuFirst4[i - 1], pos++);
                cpuFirst4[i] = Argmax(step);
            }
        }

        // CUDA long decode.
        const int NSteps = 16;
        var cudaDecoded = new int[NSteps];
        using (var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512))
        {
            var logits = cudaFwd.Prefill(tokens);
            cudaDecoded[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < cudaDecoded.Length; i++)
            {
                var step = cudaFwd.Forward(cudaDecoded[i - 1], pos++);
                cudaDecoded[i] = Argmax(step);
            }
        }

        // First 4 should match CPU exactly — drift hasn't built up yet.
        for (int i = 0; i < cpuFirst4.Length; i++)
            Assert.True(cpuFirst4[i] == cudaDecoded[i],
                $"CPU/CUDA diverge at decode step {i}: CPU={cpuFirst4[i]} CUDA={cudaDecoded[i]}. " +
                $"Full CUDA decode: [{string.Join(",", cudaDecoded)}]. " +
                "Likely an attention scale / RoPE / KV-cache mismatch — the kind that lets " +
                "first-argmax parity pass while the decode loop degenerates.");

        // ≥8 distinct tokens out of 16 — coherent output never collapses to a
        // 2-cycle repeat, but a partial repeat (e.g. "the X the Y the Z") is
        // still allowed by greedy sampling on a short factual prompt.
        int distinct = 0;
        for (int i = 0; i < cudaDecoded.Length; i++)
        {
            bool seen = false;
            for (int j = 0; j < i; j++) if (cudaDecoded[j] == cudaDecoded[i]) { seen = true; break; }
            if (!seen) distinct++;
        }
        Assert.True(distinct >= 8,
            $"CUDA 16-step decode collapsed to {distinct} distinct token(s): " +
            $"[{string.Join(",", cudaDecoded)}]. Output is degenerate.");
    }

    /// <summary>
    /// Cross-backend (CPU↔CUDA) logit-level parity on a single E4B prefill — the strict
    /// oracle the argmax-only <see cref="Gemma4_E4B_CudaForward_MatchesCpuArgmax"/> /
    /// <see cref="Gemma4_E4B_CudaForward_LongDecodeIsCoherent"/> checks cannot provide
    /// (top-1 token sequences are robust to sub-argmax perturbations, per
    /// feedback_cross_backend_parity_test). The CPU <see cref="ForwardPass"/> is the
    /// independent reference (mirrors HF/llama.cpp math, shares no GPU kernels): llama.cpp
    /// gemma4.cpp:227 plain-RMS-norms V on EVERY KV-owning gemma4 layer, E4B included, and
    /// the CPU per-token Forward does too — so a CUDA path that skips V-norm for E4B (the
    /// pre-#124 AttentionKEqV gate) diverges here at the logit level even while argmax holds.
    /// Q8_0 is argmax-stable but not bit-exact across backends, so we assert on a SINGLE
    /// prefill: argmax agreement, top-5 overlap, and a loose maxAbs bound (a real trunk
    /// math divergence is many logits; Q8_0 cross-backend noise stays well under 5).
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CpuMatchesCudaLogits()
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

        float[] cudaLogits;
        using (var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512))
            cudaLogits = cudaFwd.Prefill(tokens).ToArray();

        float[] cpuLogits;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
            cpuLogits = cpuFwd.Prefill(tokens).ToArray();

        Assert.Equal(cudaLogits.Length, cpuLogits.Length);

        float maxAbs = 0f;
        for (int i = 0; i < cpuLogits.Length; i++)
        {
            float d = MathF.Abs(cpuLogits[i] - cudaLogits[i]);
            if (d > maxAbs) maxAbs = d;
        }

        int cpuArgmax = Argmax(cpuLogits);
        int cudaArgmax = Argmax(cudaLogits);
        var cpuTop5 = TopK(cpuLogits, 5);
        var cudaTop5 = TopK(cudaLogits, 5);
        int overlap = 0;
        foreach (var t in cpuTop5) if (Array.IndexOf(cudaTop5, t) >= 0) overlap++;

        Assert.True(cpuArgmax == cudaArgmax,
            $"CPU↔CUDA E4B argmax disagree: CPU={cpuArgmax} CUDA={cudaArgmax}, maxAbs={maxAbs:F3}, " +
            $"CPU top5=[{string.Join(",", cpuTop5)}] CUDA top5=[{string.Join(",", cudaTop5)}]. " +
            "A structural V-norm / per-layer-KV ordering bug, not Q8_0 noise.");
        Assert.True(overlap >= 4,
            $"CPU↔CUDA E4B top-5 overlap only {overlap}/5 (maxAbs={maxAbs:F3}); " +
            $"CPU=[{string.Join(",", cpuTop5)}] CUDA=[{string.Join(",", cudaTop5)}].");
        Assert.True(maxAbs < 5.0f,
            $"CPU↔CUDA E4B logit maxAbs {maxAbs:F3} exceeds the structural bound (Q8_0 cross-backend " +
            "noise stays well under 5; this magnitude indicates a trunk math divergence — e.g. V-norm " +
            "applied on one backend but not the other).");
    }

    /// <summary>
    /// SnapKV must be force-disabled for Gemma-4-style models: their SWA layers use
    /// sliding-window ring caches and layers carry per-layer head_dim, so the full-context
    /// scoring + uniform-kvDim compaction in ApplySnapKvEviction would mis-index the cache.
    /// With an explicit budget set and an over-budget prompt, KvLength must equal the full
    /// prompt length (no eviction) rather than collapsing to the budget.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_CudaForward_SnapKvDisabled_NoEvictionEvenOverBudget()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int budget = 64;
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", budget.ToString());
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            Assert.NotNull(hp.LayerHeadDim); // confirms this is the Gemma-4-like path

            // Build an over-budget prompt (N > budget AND N > SnapKV window) from valid
            // mid-vocab token ids — enough to trip the eviction gate if it were enabled.
            int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
            int[] cycle = { 818, 5279, 529, 7001, 563, 1234, 4567, 8901 };
            var tokens = new int[160];
            tokens[0] = bosId;
            for (int i = 1; i < tokens.Length; i++) tokens[i] = cycle[i % cycle.Length];

            using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 1024);
            var logits = fwd.Prefill(tokens);

            Assert.Equal(hp.VocabSize, logits.Length);
            // The decisive assertion: SnapKV stayed off, so the cache keeps every token.
            Assert.Equal(tokens.Length, fwd.KvLength);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev);
        }
    }

    /// <summary>
    /// #211: the E4B QAT q4_0 GGUF omits attn_k / attn_v / attn_k_norm for the 18 KV-share
    /// tail layers (the Q8_0 ships dead copies). CudaForwardPass already guarded the upload
    /// behind <c>!kvShared</c>; this pins that the full <c>-g -1</c> path loads the file and
    /// tracks the CPU reference at first-decode argmax. Dev-box-only (silent-skips without the
    /// file or CUDA), like every other gemma4 test here.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Q4_0_CudaForward_LoadsAndMatchesCpuArgmax()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath("gemma-4-E4B_q4_0-it.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);

        // Precondition: a KV-share layer whose attn_k_norm is genuinely absent (the case that
        // threw pre-#211 on the CPU pass, and that CudaForwardPass's !kvShared guard skips).
        Assert.NotNull(hp.KvSourceLayer);
        int sharedLayer = -1;
        for (int i = 0; i < hp.KvSourceLayer!.Count; i++)
            if (hp.KvSourceLayer[i] >= 0) { sharedLayer = i; break; }
        Assert.True(sharedLayer >= 0, "expected a KV-share layer in the E4B q4_0 GGUF — wrong file?");
        Assert.Null(model.FindTensor($"blk.{sharedLayer}.attn_k_norm.weight"));

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        // CPU reference (the pass that was fixed) — independent of any GPU kernel.
        int cpuArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
            cpuArgmax = Argmax(cpuFwd.Prefill(tokens));

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);
        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in E4B q4_0 CUDA output.");

        int cudaArgmax = Argmax(logits);
        if (eosId >= 0)
            Assert.NotEqual(eosId, cudaArgmax);
        Assert.True(cpuArgmax == cudaArgmax,
            $"CPU↔CUDA E4B q4_0 first-argmax disagree: CPU={cpuArgmax} CUDA={cudaArgmax}.");
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
