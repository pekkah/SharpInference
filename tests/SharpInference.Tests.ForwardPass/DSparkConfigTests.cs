using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// <see cref="DSparkConfig"/> parsing/validation against the real
/// dspark_qwen3_4b_block7 config.json shape (docs/dspark-plan.md, PR #413).
/// </summary>
public sealed class DSparkConfigTests
{
    /// <summary>The released head's config, verbatim shape (trimmed of HF noise fields).</summary>
    private const string RealConfig = """
        {
          "architectures": ["Qwen3DSparkModel"],
          "attention_bias": false,
          "block_size": 7,
          "confidence_head_with_markov": true,
          "dtype": "bfloat16",
          "enable_confidence_head": true,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 2560,
          "intermediate_size": 9728,
          "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
          "markov_head_type": "vanilla",
          "markov_rank": 256,
          "mask_token_id": 151669,
          "max_position_embeddings": 40960,
          "model_type": "qwen3",
          "num_anchors": 512,
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "num_target_layers": 36,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 1000000, "rope_type": "default" },
          "target_layer_ids": [1, 9, 17, 25, 33],
          "tie_word_embeddings": false,
          "vocab_size": 151936
        }
        """;

    [Fact]
    public void Parses_RealConfig()
    {
        var cfg = DSparkConfig.FromJson(RealConfig);

        Assert.Equal(2560, cfg.HiddenSize);
        Assert.Equal(128, cfg.HeadDim);
        Assert.Equal(32, cfg.NumHeads);
        Assert.Equal(8, cfg.NumKvHeads);
        Assert.Equal(9728, cfg.IntermediateSize);
        Assert.Equal(5, cfg.NumLayers);
        Assert.Equal(7, cfg.BlockSize);
        Assert.Equal(151669, cfg.MaskTokenId);
        Assert.Equal([1, 9, 17, 25, 33], cfg.TargetLayerIds);
        Assert.Equal(36, cfg.NumTargetLayers);
        Assert.Equal(256, cfg.MarkovRank);
        Assert.Equal("vanilla", cfg.MarkovHeadType);
        Assert.True(cfg.EnableConfidenceHead);
        Assert.True(cfg.ConfidenceHeadWithMarkov);
        Assert.Equal(151936, cfg.VocabSize);
        Assert.Equal(1e-6f, cfg.RmsNormEps);
        Assert.Equal(1_000_000f, cfg.RopeTheta);
        Assert.Equal(40960, cfg.MaxPositionEmbeddings);
        Assert.Equal(5 * 2560, cfg.TapDim);
    }

    [Fact]
    public void TopLevel_RopeTheta_Fallback()
    {
        var cfg = DSparkConfig.FromJson(Mutate(RealConfig,
            "\"rope_parameters\": { \"rope_theta\": 1000000, \"rope_type\": \"default\" }",
            "\"rope_theta\": 5000"));
        Assert.Equal(5000f, cfg.RopeTheta);
    }

    [Fact]
    public void MissingRopeTheta_DefaultsTo10K()
    {
        var cfg = DSparkConfig.FromJson(Mutate(RealConfig,
            "\"rope_parameters\": { \"rope_theta\": 1000000, \"rope_type\": \"default\" },", ""));
        Assert.Equal(10_000f, cfg.RopeTheta);
    }

    [Theory]
    [InlineData("\"markov_head_type\": \"vanilla\"", "\"markov_head_type\": \"gated\"")]
    [InlineData("\"markov_head_type\": \"vanilla\"", "\"markov_head_type\": \"rnn\"")]
    [InlineData("\"full_attention\"]", "\"sliding_attention\"]")]
    [InlineData("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")]
    [InlineData("\"attention_bias\": false", "\"attention_bias\": true")]
    [InlineData("[1, 9, 17, 25, 33]", "[9, 1, 17, 25, 33]")]
    [InlineData("[1, 9, 17, 25, 33]", "[1, 9, 17, 25, 36]")]
    [InlineData("[1, 9, 17, 25, 33]", "[-1, 9, 17, 25, 33]")]
    [InlineData("[1, 9, 17, 25, 33]", "[]")]
    [InlineData("\"mask_token_id\": 151669", "\"mask_token_id\": 151936")]
    [InlineData("\"mask_token_id\": 151669", "\"mask_token_id\": -1")]
    public void Rejects_UnsupportedVariants(string find, string replace) =>
        Assert.Throws<NotSupportedException>(() => DSparkConfig.FromJson(Mutate(RealConfig, find, replace)));

    /// <summary>
    /// Tapping the LAST target layer (id == num_target_layers-1) is accepted by
    /// our loader: the tap plumbing captures raw layer outputs (pre-final-norm)
    /// for every layer, so the reference's assert_no_final_target_layer concern
    /// (HF hidden_states[-1] being post-norm) doesn't apply here. No released
    /// head does this, but the config isn't rejected.
    /// </summary>
    [Fact]
    public void FinalLayerTap_IsAccepted()
    {
        var cfg = DSparkConfig.FromJson(Mutate(RealConfig, "[1, 9, 17, 25, 33]", "[1, 9, 17, 25, 35]"));
        Assert.Equal(35, cfg.TargetLayerIds[^1]);
    }

    [Fact]
    public void MarkovRank0_WithoutHeadType_IsVanillaDefault()
    {
        var json = Mutate(RealConfig, "\"markov_rank\": 256", "\"markov_rank\": 0");
        json = Mutate(json, "\"markov_head_type\": \"vanilla\",", "");
        json = Mutate(json, "\"confidence_head_with_markov\": true", "\"confidence_head_with_markov\": false");
        var cfg = DSparkConfig.FromJson(json);
        Assert.Equal(0, cfg.MarkovRank);
        Assert.Equal("vanilla", cfg.MarkovHeadType);
    }

    [Fact]
    public void ConfidenceWithMarkov_RequiresMarkovRank()
    {
        var json = Mutate(RealConfig, "\"markov_rank\": 256", "\"markov_rank\": 0");
        json = Mutate(json, "\"markov_head_type\": \"vanilla\",", "");
        Assert.Throws<NotSupportedException>(() => DSparkConfig.FromJson(json));
    }

    private static string Mutate(string json, string find, string replace)
    {
        Assert.Contains(find, json);
        return json.Replace(find, replace);
    }
}
