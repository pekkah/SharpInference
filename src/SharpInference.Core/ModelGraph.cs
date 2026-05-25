namespace SharpInference.Core;

/// <summary>
/// Represents the full computation graph of a loaded model.
/// Layers are stored in execution order; weights are resolved lazily.
/// </summary>
public sealed class ModelGraph
{
    public string Architecture { get; init; } = string.Empty;
    public ModelHyperparams Hyperparams { get; init; } = new();
    public IReadOnlyList<ModelLayer> Layers { get; init; } = [];
    public IReadOnlyDictionary<string, GgufTensorInfo> WeightIndex { get; init; } =
        new Dictionary<string, GgufTensorInfo>();
}

public sealed record ModelHyperparams
{
    public int VocabSize { get; init; }
    public int ContextLength { get; init; }
    public int EmbeddingDim { get; init; }
    public int NumLayers { get; init; }
    public int NumHeads { get; init; }
    public int NumKvHeads { get; init; }
    public int IntermediateDim { get; init; }

    /// <summary>
    /// Attention head dimension. For most models this equals EmbeddingDim / NumHeads,
    /// but some architectures (e.g. Qwen3-MoE) use a larger head dim stored in
    /// {arch}.attention.key_length metadata.
    /// </summary>
    public int HeadDim { get; init; }

    public float RmsNormEps { get; init; } = 1e-5f;
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>
    /// Whether the model has bias terms on Q/K/V/O attention projections (e.g. Qwen models).
    /// Detected at load time by probing for "blk.0.attn_q.bias" in the GGUF tensor index.
    /// </summary>
    public bool HasAttnBias { get; init; }

    /// <summary>
    /// Whether the model has per-head Q/K RMSNorm (e.g. Qwen3).
    /// Detected at load time by probing for "blk.0.attn_q_norm.weight" in the GGUF tensor index.
    /// </summary>
    public bool HasQkNorm { get; init; }

    // ── MoE (Mixture of Experts) ──

    /// <summary>Whether this model uses Mixture of Experts architecture.</summary>
    public bool IsMoE { get; init; }

    /// <summary>Total number of experts per layer (e.g. 16 for Llama 4 Scout).</summary>
    public int NumExperts { get; init; }

    /// <summary>Number of experts activated per token (e.g. 1 for Llama 4 Scout, 2 for Mixtral).</summary>
    public int NumActiveExperts { get; init; }

    /// <summary>FFN dimension per expert (may differ from IntermediateDim which is the shared FFN dim).</summary>
    public int ExpertIntermediateDim { get; init; }

    /// <summary>Whether the model has a shared expert that runs on every token (e.g. Llama 4, DeepSeek-V2).</summary>
    public bool HasSharedExpert { get; init; }

    // ── NoPE (No Positional Encoding) ──

    /// <summary>
    /// Every Nth layer skips RoPE (NoPE). 0 = all layers use RoPE.
    /// Llama-4: step=4 → layers 3,7,11,... (0-indexed where (layer+1)%4==0) use NoPE.
    /// </summary>
    public int NoRopeLayerStep { get; init; }

    /// <summary>
    /// Whether the MoE router uses sigmoid gating instead of softmax (e.g. Llama-4).
    /// </summary>
    public bool UseSigmoidGating { get; init; }

    /// <summary>
    /// Whether QK-norm uses pure RMS norm (L2 normalize) without learned weights.
    /// Llama-4 uses Llama4TextL2Norm (pure RMS norm); Qwen3 uses weighted RMS norm.
    /// </summary>
    public bool UseL2QkNorm { get; init; }

    /// <summary>
    /// True for NEOX-style RoPE (rotates dim pairs (i, i + headDim/2)).
    /// False for LLaMA-style "normal" RoPE (rotates consecutive pairs (2i, 2i+1)).
    /// Qwen2/Qwen3, Phi, Gemma, Falcon, and most non-LLaMA architectures use NEOX.
    /// LLaMA, Mistral, SmolLM, Granite, and DeepSeek use the interleaved convention.
    /// </summary>
    public bool IsNeoxRope { get; init; }

    /// <summary>
    /// Number of head dims that receive RoPE rotation. Default equals HeadDim (full RoPE).
    /// Some architectures (notably qwen35moe) use partial RoPE where only the first
    /// <see cref="RopeDim"/> dims of each head are rotated and the rest pass through.
    /// </summary>
    public int RopeDim { get; init; }

    // ── Hybrid Gated DeltaNet + Attention (qwen35moe) ──

    /// <summary>
    /// True for hybrid models whose trunk interleaves recurrent (Gated DeltaNet / SSM-named)
    /// blocks with full softmax-attention blocks. Drives layer-by-layer dispatch.
    /// </summary>
    public bool IsHybridSsm { get; init; }

    /// <summary>
    /// Per-layer block type. <c>null</c> for non-hybrid models (every layer is Attention).
    /// Indexed by absolute layer number (0..NumLayers-1).
    /// </summary>
    public IReadOnlyList<LayerType>? LayerTypes { get; init; }

    /// <summary>
    /// Gated DeltaNet configuration. Non-null iff <see cref="IsHybridSsm"/> is true.
    /// Holds per-head dims, group count, conv kernel, and rank — the parameters of the
    /// recurrent block. Despite the GGUF prefix <c>ssm.*</c>, the math is delta-rule
    /// linear attention with a 2D matrix state per head, NOT Mamba selective scan.
    /// </summary>
    public GdnConfig? Gdn { get; init; }

    /// <summary>
    /// Number of Multi-Token Prediction (MTP) head layers stored at the end of the GGUF
    /// block stack. Read from <c>{arch}.nextn_predict_layers</c> (default 0 when absent).
    /// On disk these live at block indices <c>NumLayers..NumLayers+NumMtpLayers-1</c> —
    /// <see cref="NumLayers"/> already excludes them so the main forward loop stays clean.
    /// Used by MTP self-speculative decoding to draft N-ahead tokens per main forward.
    /// </summary>
    public int NumMtpLayers { get; init; }

    /// <summary>
    /// Extract hyperparameters from GGUF metadata using the model's architecture prefix.
    /// Supports llama-family models (llama, mistral, qwen, smollm, etc.) and MoE variants.
    /// </summary>
    public static ModelHyperparams FromGgufMetadata(IReadOnlyDictionary<string, object> metadata)
        => FromGgufMetadata(metadata, null);

    public static ModelHyperparams FromGgufMetadata(IReadOnlyDictionary<string, object> metadata,
        GgufModel? model)
    {
        var arch = metadata.TryGetValue("general.architecture", out var a) ? (string)a : "llama";

        int numExperts = GetInt(metadata, $"{arch}.expert_count");
        int numActiveExperts = GetInt(metadata, $"{arch}.expert_used_count");
        bool isMoE = numExperts > 0;

        // Detect features by probing tensor names
        bool hasAttnBias = metadata.ContainsKey("_sharpi.has_attn_bias")
            || (model?.FindTensor("blk.0.attn_q.bias") is not null);
        bool hasQkNorm = metadata.ContainsKey("_sharpi.has_qk_norm")
            || (model?.FindTensor("blk.0.attn_q_norm.weight") is not null);
        bool hasSharedExpert = isMoE
            && (model?.FindTensor("blk.0.ffn_gate_shexp.weight") is not null);

        // Llama-4 (arch "llama4") uses NoPE: every 4th layer skips RoPE.
        // This is hardcoded in llama.cpp (not stored in GGUF metadata).
        bool isLlama4 = arch.Equals("llama4", StringComparison.OrdinalIgnoreCase);
        int noRopeStep = isLlama4 ? 4 : 0;
        // Llama-4 uses sigmoid gating with weight-before-FFN per Meta's reference impl.
        bool useSigmoidGating = isLlama4;
        // Llama-4 uses Llama4TextL2Norm for QK-norm: pure RMS norm without learned weights.
        // No attn_q_norm.weight tensor exists, so force hasQkNorm for Llama-4.
        bool useL2QkNorm = isLlama4;
        if (isLlama4) hasQkNorm = true;

        // RoPE convention: NEOX (pairs offset by headDim/2) vs NORM/interleaved (consecutive pairs).
        // Mirrors llama.cpp's llama_model_rope_type() in src/llama-model.cpp (NEOX block).
        // Architectures NOT listed here default to NORM (LLaMA-style interleaved).
        // Special rope types (MROPE for QWEN2VL/PADDLEOCR, IMROPE for QWEN3VL family, conditional
        // for GLM4/GLM4_MOE) are not currently supported and would need their own dispatch.
        bool isNeoxRope = arch switch
        {
            "falcon" or "falcon-h1" or "grok" or "dbrx" or
            "bert" or "jina-bert-v3" or "modern-bert" or "nomic-bert" or "nomic-bert-moe" or "eurobert" or
            "stablelm" or "bitnet" or
            "qwen" or "qwen2" or "dream" or "qwen2moe" or "qwen3" or "qwen3moe" or
            "llada-moe" or "rnd1" or
            "olmo2" or "olmoe" or
            "phi2" or "phi3" or "phimoe" or
            "plamo" or "plamo2" or "plamo3" or
            "gemma" or "gemma2" or "gemma3" or "gemma3n" or "gemma4" or "gemma-embedding" or
            "starcoder2" or "openelm" or "gptneox" or "codeshell" or "orion" or
            "nemotron" or "exaone" or "exaone4" or "exaone-moe" or
            "minicpm3" or "bailingmoe2" or "dots1" or
            "hunyuan-moe" or "hunyuan-dense" or
            "jais2" or "gpt-oss" or
            "lfm2" or "lfm2moe" or "smallthinker" or "seed_oss" or "grovemoe" or
            "apertus" or "minimax-m2" or "cogvlm" or "pangu-embedded" or "afmoe" or
            "qwen3next" or "qwen35moe" or "qwen35" or "mimo2" or "step35" => true,
            _ => false,
        };

        int embDim = GetInt(metadata, $"{arch}.embedding_length");
        int numHeads = GetInt(metadata, $"{arch}.attention.head_count");
        // Some models (e.g. Qwen3-MoE) use a head dim that differs from embDim/numHeads.
        // Read from metadata if available; fall back to computed value.
        int headDimFromMeta = GetInt(metadata, $"{arch}.attention.key_length");
        int headDim = headDimFromMeta > 0 ? headDimFromMeta : (numHeads > 0 ? embDim / numHeads : embDim);

        // Partial RoPE: rope.dimension_count, when present and smaller than headDim,
        // rotates only the first ropeDim dims of each head. qwen35moe rotates 64 of 256.
        int ropeDimFromMeta = GetInt(metadata, $"{arch}.rope.dimension_count");
        int ropeDim = ropeDimFromMeta > 0 ? ropeDimFromMeta : headDim;

        // Hybrid Gated-DeltaNet detection. qwen35moe (and similar future architectures)
        // interleave recurrent and attention blocks. We rely on metadata exclusively here;
        // the synthetic-metadata probe in GgufModel.Open injects _sharpi.is_hybrid_ssm
        // when GDN tensors are observed.
        bool isHybridSsm = metadata.ContainsKey("_sharpi.is_hybrid_ssm")
                        || arch == "qwen35moe";

        // {arch}.block_count is the total block count in the file, which on MTP-enabled
        // models (qwen35 27B-MTP, qwen35moe-MTP) includes the MTP head blocks appended
        // after the main layers. Strip them so NumLayers reflects only the main model;
        // MTP blocks are loaded separately by the MTP head logic.
        int totalBlocks = GetInt(metadata, $"{arch}.block_count");
        int numMtpLayers = GetInt(metadata, $"{arch}.nextn_predict_layers", 0);
        int numLayers = totalBlocks - numMtpLayers;

        IReadOnlyList<LayerType>? layerTypes = null;
        GdnConfig? gdn = null;
        if (isHybridSsm && numLayers > 0)
        {
            int fullAttnInterval = GetInt(metadata, $"{arch}.full_attention_interval", 4);
            var types = new LayerType[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                // qwen35moe: full attention when (i+1) % full_attention_interval == 0.
                // i.e. the LAST layer of each group of full_attention_interval is full attn.
                bool isFullAttn = fullAttnInterval > 0 && ((i + 1) % fullAttnInterval) == 0;
                types[i] = isFullAttn ? LayerType.Attention : LayerType.GatedDeltaNet;
            }
            layerTypes = types;

            gdn = new GdnConfig(
                NumKHeads:    GetInt(metadata, $"{arch}.ssm.group_count"),
                NumVHeads:    GetInt(metadata, $"{arch}.ssm.time_step_rank"),
                HeadDim:      GetInt(metadata, $"{arch}.ssm.state_size"),
                InnerSize:    GetInt(metadata, $"{arch}.ssm.inner_size"),
                ConvKernel:   GetInt(metadata, $"{arch}.ssm.conv_kernel"),
                FullAttentionInterval: fullAttnInterval);
        }

        return new ModelHyperparams
        {
            VocabSize = GetInt(metadata, $"{arch}.vocab_size"),
            ContextLength = GetInt(metadata, $"{arch}.context_length"),
            EmbeddingDim = embDim,
            NumLayers = numLayers,
            NumMtpLayers = numMtpLayers,
            NumHeads = numHeads,
            NumKvHeads = GetInt(metadata, $"{arch}.attention.head_count_kv",
                            GetInt(metadata, $"{arch}.attention.head_count")),
            IntermediateDim = GetInt(metadata, $"{arch}.feed_forward_length"),
            HeadDim = headDim,
            RmsNormEps = GetFloat(metadata, $"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
            RopeTheta = GetFloat(metadata, $"{arch}.rope.freq_base", 10_000f),
            HasAttnBias = hasAttnBias,
            HasQkNorm = hasQkNorm,
            IsMoE = isMoE,
            NumExperts = numExperts,
            NumActiveExperts = numActiveExperts,
            ExpertIntermediateDim = GetInt(metadata, $"{arch}.expert_feed_forward_length",
                                       GetInt(metadata, $"{arch}.feed_forward_length")),
            HasSharedExpert = hasSharedExpert,
            NoRopeLayerStep = noRopeStep,
            UseSigmoidGating = useSigmoidGating,
            UseL2QkNorm = useL2QkNorm,
            IsNeoxRope = isNeoxRope,
            RopeDim = ropeDim,
            IsHybridSsm = isHybridSsm,
            LayerTypes = layerTypes,
            Gdn = gdn,
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0) =>
        m.TryGetValue(key, out var v) ? Convert.ToInt32(v) : fallback;

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;
}

public abstract class ModelLayer
{
    public string Name { get; init; } = string.Empty;
}

public sealed class AttentionLayer : ModelLayer { }
public sealed class FeedForwardLayer : ModelLayer { }
public sealed class EmbeddingLayer : ModelLayer { }
public sealed class NormLayer : ModelLayer { }
public sealed class OutputLayer : ModelLayer { }

/// <summary>
/// Block type for one trunk layer. Hybrid models (qwen35moe) interleave the two
/// types according to a fixed interval; pure transformer models are all-Attention.
/// </summary>
public enum LayerType
{
    Attention = 0,
    GatedDeltaNet = 1,
}

/// <summary>
/// Hyperparameters for a Gated DeltaNet recurrent block (linear attention with
/// delta-rule rank-1 state update). Despite the GGUF prefix <c>ssm.*</c>, this
/// is NOT Mamba selective scan — there is no per-state-dim A vector and the
/// recurrent state is a per-head matrix.
/// </summary>
/// <param name="NumKHeads">Number of key heads (= <c>ssm.group_count</c>). Each K head is shared by
/// <c>NumVHeads / NumKHeads</c> value heads (GQA-style for the GDN block).</param>
/// <param name="NumVHeads">Number of value heads (= <c>ssm.time_step_rank</c>). The per-head decay
/// (alpha/A) and write rate (beta) are scalars indexed by v-head.</param>
/// <param name="HeadDim">Head dimension shared by Q, K, V, and the per-head matrix state
/// (= <c>ssm.state_size</c>). Each head's recurrent state is a <c>[HeadDim, HeadDim]</c> matrix.</param>
/// <param name="InnerSize">Total value channels = <c>NumVHeads * HeadDim</c> (= <c>ssm.inner_size</c>).</param>
/// <param name="ConvKernel">Depthwise causal conv1d kernel size, applied to the joint Q‖K‖V stream
/// (= <c>ssm.conv_kernel</c>; typically 4).</param>
/// <param name="FullAttentionInterval">Stride between full-attention layers. With value 4,
/// layers where <c>(i+1) % 4 == 0</c> are full attention and the rest are GDN.</param>
public sealed record GdnConfig(
    int NumKHeads,
    int NumVHeads,
    int HeadDim,
    int InnerSize,
    int ConvKernel,
    int FullAttentionInterval)
{
    /// <summary>Total key channels = <c>NumKHeads * HeadDim</c>.</summary>
    public int KeyDim => NumKHeads * HeadDim;

    /// <summary>Total value channels = <c>NumVHeads * HeadDim</c>; equals <see cref="InnerSize"/>.</summary>
    public int ValueDim => NumVHeads * HeadDim;

    /// <summary>
    /// Channels in the joint QKV stream that the depthwise conv1d operates on:
    /// <c>KeyDim*2 + ValueDim</c> (Q and K share KeyDim each, V is ValueDim).
    /// </summary>
    public int ConvChannels => KeyDim * 2 + ValueDim;
}
