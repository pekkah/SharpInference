using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// First automated coverage of the Gemma 4 12B QAT (dense Q4_0) CUDA forward path
/// (issue #124/#173). Until now the 12B trunk — per-layer KV heads (8 GQA / 1 MQA),
/// the <c>attention_k_eq_v</c> global layers (V reuses the raw K projection + a pure
/// V-norm), the packed Q6_K tied embedding, SWA/global split, softcaps — was only
/// validated by hand via the CLI. These pin it as a regression guard.
///
/// Mirrors the E4B integration tests: a synthetic prompt-token sequence drives a
/// prefill, then we assert the post-prompt logits are finite and the greedy decode is
/// non-degenerate (≥2 distinct tokens, not all EOS). This catches NaN/degenerate-output
/// regressions (attention-scale, softcap, k_eq_v, per-layer-KV, embed bugs) without
/// depending on the exact chat template or a meaningful prompt.
///
/// Silent-skip: if CUDA isn't available OR the GGUF isn't on disk these tests no-op.
/// </summary>
public sealed class Gemma4Cuda12BForwardPassTests
{
    private const string ModelFile = "gemma-4-12b-it-qat-q4_0.gguf";

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
        var tokens = new[] { bosId, 818, 5279, 529, 7001, 563, 1234, 4567, 8901 };

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
}
