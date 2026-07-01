using SharpInference.Core;
namespace SharpInference.Tests.Core;

// Ornith-1.0 (DeepReinforce, MIT) is an agentic-coding "self-scaffolding" RL
// post-train of existing Qwen3.5 / Gemma 4 bases — NOT a new architecture. After
// llama.cpp GGUF conversion its arch strings are the ones SharpInference already
// dispatches: `qwen35moe` (the 35B / 397B MoE variants) and dense `qwen35` (the 9B).
// These tests pin that routing so an Ornith GGUF lands on the existing hybrid
// Gated-DeltaNet + sparse-attention MoE path (35B/397B) and the dense path (9B)
// without any model-specific code.
public sealed class Ornith10ArchitectureTests
{
    // Ornith-1.0-35B / -397B: HF arch `qwen3_5_moe` → GGUF arch `qwen35moe`.
    // Hyperparams mirror the Qwen3.5 35B-A3B base it was trained from.
    [Fact]
    public void Ornith35BMoe_RoutesToHybridSsmMoEPath()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                       = "qwen35moe",
            ["qwen35moe.block_count"]                       = 40,
            ["qwen35moe.embedding_length"]                  = 2048,
            ["qwen35moe.context_length"]                    = 262_144,
            ["qwen35moe.attention.head_count"]              = 16,
            ["qwen35moe.attention.head_count_kv"]           = 2,
            ["qwen35moe.attention.key_length"]              = 256,
            ["qwen35moe.attention.layer_norm_rms_epsilon"]  = 1e-6f,
            ["qwen35moe.full_attention_interval"]           = 4,
            ["qwen35moe.rope.dimension_count"]              = 64,
            ["qwen35moe.expert_count"]                      = 256,
            ["qwen35moe.expert_used_count"]                 = 8,
            ["qwen35moe.ssm.conv_kernel"]                   = 4,
            ["qwen35moe.ssm.group_count"]                   = 16,
            ["qwen35moe.ssm.inner_size"]                    = 4096,
            ["qwen35moe.ssm.state_size"]                    = 128,
            ["qwen35moe.ssm.time_step_rank"]                = 32,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        // Ornith-35B must take the existing hybrid GDN + MoE path — no new arch handling.
        Assert.True(hp.IsHybridSsm);
        Assert.True(hp.IsMoE);
        Assert.True(hp.IsNeoxRope);
        Assert.Equal(256, hp.NumExperts);
        Assert.Equal(8, hp.NumActiveExperts);
        Assert.NotNull(hp.LayerTypes);
        Assert.NotNull(hp.Gdn);
        // 1-in-4 full-attention interleave (the rest Gated-DeltaNet).
        Assert.Equal(LayerType.Attention,     hp.LayerTypes![3]);
        Assert.Equal(LayerType.GatedDeltaNet, hp.LayerTypes[0]);
    }

    // Ornith-1.0-9B: HF arch `qwen3_5` (dense) → GGUF arch `qwen35`. Recognized as a
    // NEOX-RoPE dense transformer; not MoE, and not hybrid unless GDN tensors are present.
    [Fact]
    public void Ornith9BDense_IsRecognizedAsNeoxDense()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                   = "qwen35",
            ["qwen35.block_count"]                     = 48,
            ["qwen35.embedding_length"]                = 4096,
            ["qwen35.attention.head_count"]            = 32,
            ["qwen35.attention.head_count_kv"]         = 8,
            ["qwen35.attention.key_length"]            = 128,
            ["qwen35.attention.layer_norm_rms_epsilon"] = 1e-6f,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.True(hp.IsNeoxRope);     // qwen35 is in the NEOX rope set
        Assert.False(hp.IsMoE);
        Assert.False(hp.IsHybridSsm);   // no GDN tensors → plain dense path
        Assert.Null(hp.LayerTypes);
        Assert.Null(hp.Gdn);
        Assert.Equal(48, hp.NumLayers);
    }

    // If the 9B (or any dense qwen35) ships Gated-DeltaNet tensors, GgufModel.Open
    // injects `_sharpi.is_hybrid_ssm` and the hybrid path activates automatically.
    [Fact]
    public void Ornith9BDense_ActivatesHybridWhenGdnTensorsProbed()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                   = "qwen35",
            ["_sharpi.is_hybrid_ssm"]                  = true,
            ["qwen35.block_count"]                     = 48,
            ["qwen35.embedding_length"]                = 4096,
            ["qwen35.attention.head_count"]            = 32,
            ["qwen35.attention.head_count_kv"]         = 8,
            ["qwen35.attention.key_length"]            = 128,
            ["qwen35.attention.layer_norm_rms_epsilon"] = 1e-6f,
            ["qwen35.full_attention_interval"]         = 4,
            ["qwen35.ssm.conv_kernel"]                 = 4,
            ["qwen35.ssm.group_count"]                 = 16,
            ["qwen35.ssm.inner_size"]                  = 4096,
            ["qwen35.ssm.state_size"]                  = 128,
            ["qwen35.ssm.time_step_rank"]              = 32,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.True(hp.IsHybridSsm);
        Assert.NotNull(hp.LayerTypes);
        Assert.NotNull(hp.Gdn);
    }
}
