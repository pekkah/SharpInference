using SharpInference.Core;
namespace SharpInference.Tests.Core;

public sealed class ModelHyperparamsTests
{
    [Fact]
    public void Qwen35Moe_PopulatesGdnConfigAndLayerTypeMask()
    {
        // Minimal metadata mirroring the qwen35moe GGUF (Qwen3.6-35B-A3B).
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                            = "qwen35moe",
            ["qwen35moe.block_count"]                           = 40,
            ["qwen35moe.embedding_length"]                      = 2048,
            ["qwen35moe.context_length"]                        = 262_144,
            ["qwen35moe.vocab_size"]                            = (ulong)248_320,
            ["qwen35moe.attention.head_count"]                  = 16,
            ["qwen35moe.attention.head_count_kv"]               = 2,
            ["qwen35moe.attention.key_length"]                  = 256,
            ["qwen35moe.attention.value_length"]                = 256,
            ["qwen35moe.attention.layer_norm_rms_epsilon"]      = 1e-6f,
            ["qwen35moe.full_attention_interval"]               = 4,
            ["qwen35moe.rope.dimension_count"]                  = 64,
            ["qwen35moe.rope.freq_base"]                        = 1e7f,
            ["qwen35moe.expert_count"]                          = 256,
            ["qwen35moe.expert_used_count"]                     = 8,
            ["qwen35moe.expert_feed_forward_length"]            = 512,
            ["qwen35moe.ssm.conv_kernel"]                       = 4,
            ["qwen35moe.ssm.group_count"]                       = 16,
            ["qwen35moe.ssm.inner_size"]                        = 4096,
            ["qwen35moe.ssm.state_size"]                        = 128,
            ["qwen35moe.ssm.time_step_rank"]                    = 32,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.True(hp.IsHybridSsm);
        Assert.True(hp.IsMoE);
        Assert.True(hp.IsNeoxRope);
        Assert.Equal(64, hp.RopeDim);
        Assert.Equal(256, hp.HeadDim);                  // from key_length, full-attn head dim
        Assert.Equal(40, hp.NumLayers);
        Assert.Equal(256, hp.NumExperts);
        Assert.Equal(8, hp.NumActiveExperts);

        // Layer-type mask: full attention at (i+1) % 4 == 0.
        Assert.NotNull(hp.LayerTypes);
        Assert.Equal(40, hp.LayerTypes!.Count);
        int attnCount = 0, gdnCount = 0;
        for (int i = 0; i < 40; i++)
        {
            bool expectAttn = ((i + 1) % 4) == 0;
            var expected = expectAttn ? LayerType.Attention : LayerType.GatedDeltaNet;
            Assert.Equal(expected, hp.LayerTypes[i]);
            if (expectAttn) attnCount++; else gdnCount++;
        }
        Assert.Equal(10, attnCount);
        Assert.Equal(30, gdnCount);

        // Spot check the attn indices match the observed file.
        Assert.Equal(LayerType.Attention,     hp.LayerTypes[3]);
        Assert.Equal(LayerType.Attention,     hp.LayerTypes[39]);
        Assert.Equal(LayerType.GatedDeltaNet, hp.LayerTypes[0]);
        Assert.Equal(LayerType.GatedDeltaNet, hp.LayerTypes[38]);

        // GdnConfig: derived dims should match the observed tensor shapes.
        Assert.NotNull(hp.Gdn);
        var gdn = hp.Gdn!;
        Assert.Equal(16,   gdn.NumKHeads);
        Assert.Equal(32,   gdn.NumVHeads);
        Assert.Equal(128,  gdn.HeadDim);
        Assert.Equal(4096, gdn.InnerSize);
        Assert.Equal(4,    gdn.ConvKernel);
        Assert.Equal(4,    gdn.FullAttentionInterval);
        Assert.Equal(2048, gdn.KeyDim);                  // 16 * 128
        Assert.Equal(4096, gdn.ValueDim);                // 32 * 128
        Assert.Equal(8192, gdn.ConvChannels);            // 2*2048 + 4096 — matches ssm_conv1d [4, 8192]
    }

    [Fact]
    public void NonHybridModel_HasNullLayerTypesAndGdn()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                       = "qwen3",
            ["qwen3.block_count"]                          = 28,
            ["qwen3.embedding_length"]                     = 2048,
            ["qwen3.attention.head_count"]                 = 16,
            ["qwen3.attention.head_count_kv"]              = 8,
            ["qwen3.attention.key_length"]                 = 128,
            ["qwen3.attention.layer_norm_rms_epsilon"]     = 1e-5f,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.False(hp.IsHybridSsm);
        Assert.Null(hp.LayerTypes);
        Assert.Null(hp.Gdn);
        // RopeDim falls back to HeadDim when no rope.dimension_count is present.
        Assert.Equal(hp.HeadDim, hp.RopeDim);
    }
}
