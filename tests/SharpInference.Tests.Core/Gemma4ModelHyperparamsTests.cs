using SharpInference.Core;
namespace SharpInference.Tests.Core;

public sealed class Gemma4ModelHyperparamsTests
{
    private static Dictionary<string, object> BuildE4BMetadata()
    {
        // 42-element 5-SWA : 1-global pattern. Encoded as object[] to mirror
        // the GGUF reader, which boxes bool arrays element-by-element.
        var pattern = new object[42];
        for (int i = 0; i < 42; i++) pattern[i] = ((i + 1) % 6) != 0;

        return new Dictionary<string, object>
        {
            ["general.architecture"]                         = "gemma4",
            ["gemma4.block_count"]                           = 42,
            ["gemma4.embedding_length"]                      = 2560,
            ["gemma4.feed_forward_length"]                   = 10240,
            ["gemma4.context_length"]                        = 131_072,
            ["gemma4.vocab_size"]                            = (ulong)262_144,
            ["gemma4.attention.head_count"]                  = 8,
            ["gemma4.attention.head_count_kv"]               = 2,
            ["gemma4.attention.key_length"]                  = 512,
            ["gemma4.attention.value_length"]                = 512,
            ["gemma4.attention.key_length_swa"]              = 256,
            ["gemma4.attention.value_length_swa"]            = 256,
            ["gemma4.attention.sliding_window"]              = 512,
            ["gemma4.attention.sliding_window_pattern"]      = pattern,
            ["gemma4.attention.shared_kv_layers"]            = 18,
            ["gemma4.attention.layer_norm_rms_epsilon"]      = 1e-6f,
            ["gemma4.rope.dimension_count"]                  = 512,
            ["gemma4.rope.dimension_count_swa"]              = 256,
            ["gemma4.rope.freq_base"]                        = 1_000_000f,
            ["gemma4.rope.freq_base_swa"]                    = 10_000f,
            ["gemma4.embedding_length_per_layer_input"]      = 256,
            ["gemma4.final_logit_softcapping"]               = 30.0f,
            ["_sharpi.has_ple"]                              = true,
            ["_sharpi.has_post_attn_norm"]                   = true,
            ["_sharpi.has_post_ffw_norm"]                    = true,
            ["_sharpi.has_layer_output_scale"]               = true,
        };
    }

    [Fact]
    public void Gemma4_PopulatesAllFields()
    {
        var md = BuildE4BMetadata();
        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.Equal(42, hp.NumLayers);
        Assert.Equal(2560, hp.EmbeddingDim);
        Assert.Equal(10240, hp.IntermediateDim);
        Assert.Equal(8, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(512, hp.HeadDim);
        Assert.Equal(512, hp.RopeDim);
        Assert.Equal(1_000_000f, hp.RopeTheta);
        Assert.True(hp.IsNeoxRope);

        Assert.Equal(10_000f, hp.RopeThetaSwa);
        Assert.Equal(512, hp.SlidingWindowSize);
        Assert.Equal(256, hp.PerLayerEmbeddingWidth);
        Assert.Equal(30.0f, hp.FinalLogitSoftcap);
        Assert.Equal(MathF.Sqrt(2560), hp.EmbeddingScale);
        Assert.Equal(FfnActivation.GeluApprox, hp.FfnActivation);

        Assert.True(hp.HasPostAttnNorm);
        Assert.True(hp.HasPostFfwNorm);
        Assert.True(hp.HasPerLayerTokenEmbd);
        Assert.True(hp.HasLayerOutputScale);

        Assert.NotNull(hp.IsSwaLayer);
        Assert.Equal(42, hp.IsSwaLayer!.Count);
        int swaCount = 0, globalCount = 0;
        for (int i = 0; i < 42; i++)
        {
            bool expectSwa = ((i + 1) % 6) != 0;
            Assert.Equal(expectSwa, hp.IsSwaLayer[i]);
            if (expectSwa) swaCount++; else globalCount++;
        }
        Assert.Equal(35, swaCount);
        Assert.Equal(7, globalCount);

        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.LayerRopeDim);
        Assert.Equal(42, hp.LayerHeadDim!.Count);
        Assert.Equal(42, hp.LayerRopeDim!.Count);
        for (int i = 0; i < 42; i++)
        {
            bool sw = hp.IsSwaLayer[i];
            Assert.Equal(sw ? 256 : 512, hp.LayerHeadDim[i]);
            Assert.Equal(sw ? 256 : 512, hp.LayerRopeDim[i]);
        }

        // shared_kv_layers = 18, numLayers = 42 → firstSharedLayer = 24.
        // Layers 0..23 own their KV (−1). Layers 24..41 alias the most recent
        // earlier layer (j < 24) of matching SWA/global type.
        Assert.NotNull(hp.KvSourceLayer);
        Assert.Equal(42, hp.KvSourceLayer!.Count);
        for (int i = 0; i < 24; i++) Assert.Equal(-1, hp.KvSourceLayer[i]);
        for (int i = 24; i < 42; i++)
        {
            int src = hp.KvSourceLayer[i];
            Assert.InRange(src, 0, 23);
            Assert.Equal(hp.IsSwaLayer[i], hp.IsSwaLayer[src]);
            int expected = -1;
            for (int j = 23; j >= 0; j--)
            {
                if (hp.IsSwaLayer[j] == hp.IsSwaLayer[i]) { expected = j; break; }
            }
            Assert.Equal(expected, src);
        }
    }

    /// <summary>
    /// Dense Gemma 4 12B (QAT q4_0) metadata — values from the real header dump in
    /// <c>tests/fixtures/gemma4_12b_header.md</c>. Distinct from E4B: <b>no PLE</b>,
    /// per-layer <c>head_count_kv</c> (8 GQA on SWA, 1 MQA on global), 1024 sliding
    /// window, <c>attention_k_eq_v</c> on global layers, and (deviating from the
    /// plan) <c>layer_output_scale</c> IS present.
    /// </summary>
    private static Dictionary<string, object> Build12BDenseMetadata()
    {
        // 48-layer 5-SWA : 1-global pattern → global at 5,11,17,23,29,35,41,47.
        var pattern = new object[48];
        for (int i = 0; i < 48; i++) pattern[i] = ((i + 1) % 6) != 0;

        // Per-layer KV head count: 8 on SWA, 1 (MQA) on the global layers. Boxed
        // element-by-element to mirror the GGUF reader's object[] array shape.
        var kvHeads = new object[48];
        for (int i = 0; i < 48; i++) kvHeads[i] = ((i + 1) % 6) != 0 ? 8 : 1;

        return new Dictionary<string, object>
        {
            ["general.architecture"]                         = "gemma4",
            ["gemma4.block_count"]                           = 48,
            ["gemma4.embedding_length"]                      = 3840,
            ["gemma4.feed_forward_length"]                   = 15360,
            ["gemma4.context_length"]                        = 262_144,
            ["gemma4.vocab_size"]                            = (ulong)262_144,
            ["gemma4.attention.head_count"]                  = 16,
            ["gemma4.attention.head_count_kv"]               = kvHeads,
            ["gemma4.attention.key_length"]                  = 512,
            ["gemma4.attention.value_length"]                = 512,
            ["gemma4.attention.key_length_swa"]              = 256,
            ["gemma4.attention.value_length_swa"]            = 256,
            ["gemma4.attention.sliding_window"]              = 1024,
            ["gemma4.attention.sliding_window_pattern"]      = pattern,
            ["gemma4.attention.shared_kv_layers"]            = 0,
            ["gemma4.attention.layer_norm_rms_epsilon"]      = 1e-6f,
            ["gemma4.rope.dimension_count"]                  = 512,
            ["gemma4.rope.dimension_count_swa"]              = 256,
            ["gemma4.rope.freq_base"]                        = 1_000_000f,
            ["gemma4.rope.freq_base_swa"]                    = 10_000f,
            ["gemma4.embedding_length_per_layer_input"]      = 0,
            ["gemma4.final_logit_softcapping"]               = 30.0f,
            // No "_sharpi.has_ple" — dense 12B has no PLE.
            ["_sharpi.has_post_attn_norm"]                   = true,
            ["_sharpi.has_post_ffw_norm"]                    = true,
            ["_sharpi.has_layer_output_scale"]               = true,   // present on dense (deviates from plan)
            ["_sharpi.attention_k_eq_v"]                     = true,   // global layers reuse K as V
        };
    }

    [Fact]
    public void Gemma4_12B_Dense_PopulatesAllFields()
    {
        var md = Build12BDenseMetadata();
        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.Equal(48, hp.NumLayers);
        Assert.Equal(3840, hp.EmbeddingDim);
        Assert.Equal(15360, hp.IntermediateDim);
        Assert.Equal(16, hp.NumHeads);
        // head_count_kv is a per-layer array; the scalar collapses to the first
        // element (8, a SWA layer) without throwing on the object[] value.
        Assert.Equal(8, hp.NumKvHeads);
        Assert.Equal(512, hp.HeadDim);       // global key_length
        Assert.Equal(512, hp.RopeDim);
        Assert.Equal(1_000_000f, hp.RopeTheta);
        Assert.Equal(10_000f, hp.RopeThetaSwa);
        Assert.True(hp.IsNeoxRope);

        Assert.Equal(1024, hp.SlidingWindowSize);
        Assert.Equal(0, hp.PerLayerEmbeddingWidth);
        Assert.Equal(30.0f, hp.FinalLogitSoftcap);
        Assert.Equal(MathF.Sqrt(3840), hp.EmbeddingScale);
        Assert.Equal(FfnActivation.GeluApprox, hp.FfnActivation);

        Assert.True(hp.HasPostAttnNorm);
        Assert.True(hp.HasPostFfwNorm);
        // Dense 12B has NO per-layer embeddings — the never-exercised false branch.
        Assert.False(hp.HasPerLayerTokenEmbd);
        // layer_output_scale IS present on the dense 12B (plan §2 expected absent).
        Assert.True(hp.HasLayerOutputScale);
        // Global layers omit attn_v and reuse K as V.
        Assert.True(hp.AttentionKEqV);

        // 48-layer 5:1 pattern → 40 SWA + 8 global (5,11,17,23,29,35,41,47).
        Assert.NotNull(hp.IsSwaLayer);
        Assert.Equal(48, hp.IsSwaLayer!.Count);
        int swaCount = 0, globalCount = 0;
        for (int i = 0; i < 48; i++)
        {
            bool expectSwa = ((i + 1) % 6) != 0;
            Assert.Equal(expectSwa, hp.IsSwaLayer[i]);
            if (expectSwa) swaCount++; else globalCount++;
        }
        Assert.Equal(40, swaCount);
        Assert.Equal(8, globalCount);

        // Per-layer head dim: 256 on SWA, 512 on global.
        Assert.NotNull(hp.LayerHeadDim);
        Assert.NotNull(hp.LayerRopeDim);
        Assert.Equal(48, hp.LayerHeadDim!.Count);
        Assert.Equal(48, hp.LayerRopeDim!.Count);
        for (int i = 0; i < 48; i++)
        {
            bool sw = hp.IsSwaLayer[i];
            Assert.Equal(sw ? 256 : 512, hp.LayerHeadDim[i]);
            Assert.Equal(sw ? 256 : 512, hp.LayerRopeDim[i]);
        }

        // Per-layer KV heads: 8 (GQA) on SWA, 1 (MQA) on global.
        Assert.NotNull(hp.LayerKvHeads);
        Assert.Equal(48, hp.LayerKvHeads!.Count);
        for (int i = 0; i < 48; i++)
            Assert.Equal(hp.IsSwaLayer[i] ? 8 : 1, hp.LayerKvHeads[i]);

        // shared_kv_layers = 0 → no cross-layer KV aliasing.
        Assert.Null(hp.KvSourceLayer);
    }

    [Fact]
    public void Gemma4_NumLayersStripsMtpHeads()
    {
        var md = BuildE4BMetadata();
        md["gemma4.nextn_predict_layers"] = 2;
        md["gemma4.block_count"] = 44;

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.Equal(42, hp.NumLayers);
        Assert.Equal(2, hp.NumMtpLayers);
        Assert.Equal(42, hp.IsSwaLayer!.Count);
        Assert.Equal(42, hp.LayerHeadDim!.Count);
    }

    [Fact]
    public void NonGemma4_AllGemma4FieldsAtDefaults()
    {
        var md = new Dictionary<string, object>
        {
            ["general.architecture"]                       = "llama",
            ["llama.block_count"]                          = 32,
            ["llama.embedding_length"]                     = 4096,
            ["llama.feed_forward_length"]                  = 11008,
            ["llama.attention.head_count"]                 = 32,
            ["llama.attention.head_count_kv"]              = 32,
            ["llama.attention.layer_norm_rms_epsilon"]     = 1e-5f,
        };

        var hp = ModelHyperparams.FromGgufMetadata(md);

        Assert.Equal(1f, hp.EmbeddingScale);
        Assert.Equal(0f, hp.FinalLogitSoftcap);
        Assert.Equal(0f, hp.RopeThetaSwa);
        Assert.Equal(0, hp.SlidingWindowSize);
        Assert.Equal(0, hp.PerLayerEmbeddingWidth);
        Assert.False(hp.HasPostAttnNorm);
        Assert.False(hp.HasPostFfwNorm);
        Assert.False(hp.HasPerLayerTokenEmbd);
        Assert.False(hp.HasLayerOutputScale);
        Assert.Equal(FfnActivation.Silu, hp.FfnActivation);
        Assert.Null(hp.IsSwaLayer);
        Assert.Null(hp.KvSourceLayer);
        Assert.Null(hp.LayerHeadDim);
        Assert.Null(hp.LayerRopeDim);
    }
}
