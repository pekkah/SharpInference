using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// Regression coverage for Qwen2-family models that carry bias terms on the Q/K/V
/// attention projections but no QK-norm — the configuration VibeThinker-1.5B (a
/// fine-tune of Qwen2.5-Math-1.5B) loads with (issue #282). The Qwen2 quirk is that
/// <c>blk.*.attn_{q,k,v}.bias</c> tensors are present (driving <see cref="ModelHyperparams.HasAttnBias"/>)
/// while <c>blk.*.attn_q_norm.weight</c> is absent (so <see cref="ModelHyperparams.HasQkNorm"/>
/// stays false). These tests pin that the GGUF→<see cref="ModelHyperparams"/> mapping reports
/// bias on, qk-norm off, and the correct head/layer geometry.
/// </summary>
public sealed class Qwen2BiasModelTests
{
    /// <summary>
    /// Pure metadata test (always runs, no model file needed). Mirrors the load-path
    /// contract: <see cref="GgufModel.Open"/> injects the synthetic <c>_sharpi.has_attn_bias</c>
    /// key when it observes <c>blk.0.attn_q.bias</c> in the tensor index, and injects nothing
    /// for QK-norm when <c>blk.0.attn_q_norm.weight</c> is absent. The values below match
    /// VibeThinker-1.5B / Qwen2.5-Math-1.5B (28 layers, hidden 1536, 12 heads / 2 KV heads,
    /// head_dim 128, intermediate 8960, vocab 151936).
    /// </summary>
    [Fact]
    public void Qwen2WithAttnBias_ParsesBiasOnQkNormOffAndHeadCounts()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                   = "qwen2",
            ["qwen2.block_count"]                       = 28,
            ["qwen2.embedding_length"]                  = 1536,
            ["qwen2.context_length"]                    = 4096,
            ["qwen2.vocab_size"]                        = (ulong)151_936,
            ["qwen2.attention.head_count"]              = 12,
            ["qwen2.attention.head_count_kv"]           = 2,
            ["qwen2.feed_forward_length"]               = 8960,
            ["qwen2.attention.layer_norm_rms_epsilon"]  = 1e-6f,
            ["qwen2.rope.freq_base"]                    = 10_000f,
            // Synthetic key injected by GgufModel.Open when blk.0.attn_q.bias exists.
            // QK-norm key is deliberately omitted (Qwen2 has no attn_q_norm tensor).
            ["_sharpi.has_attn_bias"]                   = true,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.True(hp.HasAttnBias, "Qwen2 carries Q/K/V projection bias");
        Assert.False(hp.HasAttnOutputBias, "Qwen2 has no output-projection bias");
        Assert.False(hp.HasQkNorm, "Qwen2 has no QK-norm (unlike Qwen3)");
        Assert.False(hp.IsPerChannelQkNorm);
        Assert.False(hp.IsMoE);
        Assert.False(hp.IsHybridSsm);
        Assert.True(hp.IsNeoxRope, "Qwen2 uses NEOX RoPE");

        Assert.Equal(28, hp.NumLayers);
        Assert.Equal(1536, hp.EmbeddingDim);
        Assert.Equal(12, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(128, hp.HeadDim);          // 1536 / 12 (no key_length override)
        Assert.Equal(128, hp.RopeDim);          // full RoPE → equals HeadDim
        Assert.Equal(8960, hp.IntermediateDim);
        Assert.Equal(151_936, hp.VocabSize);
    }

    /// <summary>
    /// Integration test against the real VibeThinker-1.5B Q8_0 GGUF. Gated on the file
    /// being present (downloaded via <c>scripts/download-model.ps1 -Model vibethinker</c>),
    /// like the other model-dependent tests in this project — returns early (skips) when the
    /// file is absent so CI without the weights stays green. Exercises the real tensor-probe
    /// path: bias tensors present, attn_q_norm absent.
    /// </summary>
    [Fact]
    public void VibeThinkerGguf_ParsesAsQwen2WithBiasNoQkNorm()
    {
        var path = FindModelPath("models/VibeThinker-1.5B.Q8_0.gguf");
        if (path is null) return; // Model file not available — skip.

        using var model = GgufModel.Open(path);

        var arch = model.GetMetadata<string>("general.architecture");
        Assert.Equal("qwen2", arch);

        // Tensor-level ground truth for the bias/qk-norm probes.
        Assert.NotNull(model.FindTensor("blk.0.attn_q.bias"));
        Assert.NotNull(model.FindTensor("blk.0.attn_k.bias"));
        Assert.NotNull(model.FindTensor("blk.0.attn_v.bias"));
        Assert.Null(model.FindTensor("blk.0.attn_output.bias")); // Qwen2 has no o_proj bias
        Assert.Null(model.FindTensor("blk.0.attn_q_norm.weight"));

        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        Assert.True(hp.HasAttnBias);
        Assert.False(hp.HasAttnOutputBias);
        Assert.False(hp.HasQkNorm);
        Assert.False(hp.IsMoE);
        Assert.False(hp.IsHybridSsm);
        Assert.True(hp.IsNeoxRope);

        Assert.Equal(28, hp.NumLayers);
        Assert.Equal(1536, hp.EmbeddingDim);
        Assert.Equal(12, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(128, hp.HeadDim);
        Assert.Equal(8960, hp.IntermediateDim);
        Assert.Equal(151_936, hp.VocabSize);
    }

    /// <summary>Walks up from the test execution directory to find a repo-relative model file.</summary>
    private static string? FindModelPath(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, relativePath);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
