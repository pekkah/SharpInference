using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// First automated coverage of the Gemma 4 12B QAT (dense Q4_0) forward path on BOTH
/// the CUDA and CPU backends (issue #124/#173). The 12B trunk — per-layer KV heads
/// (8 GQA / 1 MQA), the <c>attention_k_eq_v</c> global layers (V reuses the raw K
/// projection + a pure V-norm), the packed Q6_K tied embedding, SWA/global split,
/// softcaps — was only validated by hand via the CLI. These pin it as a regression guard.
///
/// Mirrors the E4B integration tests: a synthetic prompt-token sequence drives a
/// prefill, then we assert the post-prompt logits are finite and the greedy decode is
/// non-degenerate (≥2 distinct tokens, not all EOS). This catches NaN/degenerate-output
/// regressions (attention-scale, softcap, k_eq_v, per-layer-KV, embed bugs) without
/// depending on the exact chat template or a meaningful prompt.
///
/// The cross-backend test compares a single prefill's logits CPU↔CUDA at the logit
/// level (argmax + top-5 overlap + maxAbs) — the independent oracle per
/// feedback_cross_backend_parity_test, since GPU-vs-GPU batched-vs-sequential checks
/// only prove an optimization is faithful to a path, not that the path is correct. The
/// CPU <see cref="ForwardPass"/> mirrors HF/llama.cpp math with no shared GPU kernels.
///
/// Silent-skip: if CUDA isn't available (CPU-only tests still run) OR the GGUF isn't on
/// disk, the relevant tests no-op.
/// </summary>
public sealed class Gemma4Cuda12BForwardPassTests
{
    private const string ModelFile = "gemma-4-12b-it-qat-q4_0.gguf";

    // Synthetic BOS-led mid-vocab prompt: the 12B IT model emits a 1-token EOS on real
    // factual prompts, so coherence is asserted on arbitrary tokens (see task notes).
    private static int[] SyntheticPrompt(int bosId) =>
        new[] { bosId, 818, 5279, 529, 7001, 563, 1234, 4567, 8901 };

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

    [Fact]
    public void Gemma4_12B_CudaForward_ProducesCoherentDecode()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: only meaningful against the real 12B k_eq_v GGUF.
        Assert.True(hp.AttentionKEqV, "expected attention_k_eq_v=true for the 12B QAT model");
        Assert.NotNull(hp.LayerKvHeads);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = SyntheticPrompt(bosId);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 4096);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite}/{logits.Length} non-finite logits after the 12B prefill.");

        int first = Argmax(logits);
        Assert.NotEqual(eosId, first);

        Span<int> decoded = stackalloc int[6];
        decoded[0] = first;
        int pos = tokens.Length;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
            for (int k = 0; k < step.Length; k++)
                Assert.True(float.IsFinite(step[k]), $"non-finite logit at decode step {i}, idx {k}");
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
            $"12B CUDA greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"([{string.Join(",", decoded.ToArray())}]); the 12B forward integration is degenerate.");

        int eosCount = 0;
        for (int i = 0; i < decoded.Length; i++)
            if (decoded[i] == eosId) eosCount++;
        Assert.True(eosCount < decoded.Length, $"All {decoded.Length} greedy tokens were EOS — 12B output is degenerate.");
    }

    /// <summary>
    /// CPU forward coverage for the 12B k_eq_v trunk. The CPU <see cref="ForwardPass"/>
    /// per-token <c>Forward</c> path now mirrors the CUDA semantics: per-layer KV heads
    /// (8 GQA on SWA, 1 MQA on global), the global k_eq_v layers (V = raw K projection,
    /// pure V-norm, no RoPE), and per-layer-head-dim attention. Before this it dereferenced
    /// a null attn_v weight (NRE) / mis-sized the MQA attention. Synthetic prompt → assert
    /// finite + non-degenerate decode (the load-bearing coherence check, not just IsFinite).
    /// </summary>
    [Fact]
    public void Gemma4_12B_CpuForward_ProducesCoherentDecode()
    {
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.AttentionKEqV, "expected attention_k_eq_v=true for the 12B QAT model");
        Assert.NotNull(hp.LayerKvHeads);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = SyntheticPrompt(bosId);

        using var backend = new CpuBackend();
        using var fwd = new SharpInference.Engine.ForwardPass(model, backend, hp, maxContextLength: 4096);

        var logits = fwd.Prefill(tokens);
        Assert.Equal(hp.VocabSize, logits.Length);

        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite}/{logits.Length} non-finite logits after the 12B CPU prefill.");

        int first = Argmax(logits);
        Assert.NotEqual(eosId, first);

        Span<int> decoded = stackalloc int[6];
        decoded[0] = first;
        int pos = tokens.Length;
        for (int i = 1; i < decoded.Length; i++)
        {
            var step = fwd.Forward(decoded[i - 1], pos++);
            for (int k = 0; k < step.Length; k++)
                Assert.True(float.IsFinite(step[k]), $"non-finite logit at CPU decode step {i}, idx {k}");
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
            $"12B CPU greedy decode produced only {distinct} distinct token(s) over {decoded.Length} steps " +
            $"([{string.Join(",", decoded.ToArray())}]); the CPU 12B k_eq_v integration is degenerate.");

        int eosCount = 0;
        for (int i = 0; i < decoded.Length; i++)
            if (decoded[i] == eosId) eosCount++;
        Assert.True(eosCount < decoded.Length, $"All {decoded.Length} CPU greedy tokens were EOS — 12B output is degenerate.");
    }

    /// <summary>
    /// Cross-backend (CPU↔CUDA) logit-level parity on a single 12B prefill. Per
    /// feedback_cross_backend_parity_test, the CPU path is the independent oracle (it
    /// mirrors HF/llama.cpp math, shares no GPU kernels), so this is what would expose a
    /// k_eq_v / per-layer-KV ordering bug that a GPU-batched-vs-sequential check can't.
    /// q4_0 is argmax-stable but not bit-exact across backends (CPU dequant→f32 matvec vs
    /// CUDA native q4_0/dp4a), so we assert on a SINGLE prefill (no long greedy drift):
    /// argmax agreement, top-5 overlap, and a loose maxAbs structural bound. A real trunk
    /// bug diverges by many logits (issue #157 was 9.2); q4_0 noise stays well under 5.
    /// </summary>
    [Fact]
    public void Gemma4_12B_CpuMatchesCudaLogits()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.AttentionKEqV, "expected attention_k_eq_v=true for the 12B QAT model");

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = SyntheticPrompt(bosId);

        float[] cudaLogits;
        using (var cudaFwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 4096))
            cudaLogits = cudaFwd.Prefill(tokens).ToArray();

        float[] cpuLogits;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp, maxContextLength: 4096))
            cpuLogits = cpuFwd.Prefill(tokens).ToArray();

        Assert.Equal(cudaLogits.Length, cpuLogits.Length);

        // maxAbs over the full vocab — the gross structural-divergence guard.
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
            $"CPU↔CUDA 12B argmax disagree: CPU={cpuArgmax} CUDA={cudaArgmax}, maxAbs={maxAbs:F3}, " +
            $"CPU top5=[{string.Join(",", cpuTop5)}] CUDA top5=[{string.Join(",", cudaTop5)}]. " +
            "A structural k_eq_v / per-layer-KV ordering bug, not q4_0 noise.");
        Assert.True(overlap >= 4,
            $"CPU↔CUDA 12B top-5 overlap only {overlap}/5 (maxAbs={maxAbs:F3}); " +
            $"CPU=[{string.Join(",", cpuTop5)}] CUDA=[{string.Join(",", cudaTop5)}].");
        Assert.True(maxAbs < 5.0f,
            $"CPU↔CUDA 12B logit maxAbs {maxAbs:F3} exceeds the structural bound (q4_0 cross-backend " +
            "noise stays well under 5; this magnitude indicates a trunk math divergence).");
    }

    /// <summary>Indices of the top-<paramref name="k"/> logits, descending by value.</summary>
    private static int[] TopK(ReadOnlySpan<float> logits, int k)
    {
        var idx = new int[k];
        var val = new float[k];
        for (int j = 0; j < k; j++) { val[j] = float.NegativeInfinity; idx[j] = -1; }
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (v <= val[k - 1]) continue;
            int p = k - 1;
            while (p > 0 && val[p - 1] < v) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
            val[p] = v; idx[p] = i;
        }
        return idx;
    }
}
