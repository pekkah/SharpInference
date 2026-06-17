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
    /// Scalar multiplier applied to the token embeddings before they enter the
    /// transformer trunk. Gemma family multiplies by <c>sqrt(EmbeddingDim)</c>.
    /// Defaults to 1 (no scaling) for every other architecture.
    /// </summary>
    public float EmbeddingScale { get; init; } = 1f;

    /// <summary>
    /// Cap value for the final-logits softcap (<c>x = tanh(x/cap) * cap</c>).
    /// 0 disables softcapping (default for non-Gemma architectures). Gemma 4 = 30.0.
    /// </summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>
    /// RoPE base frequency used by sliding-window-attention layers. Gemma 4 mixes
    /// two RoPE bases: <see cref="RopeTheta"/> (1e6 for global layers) and this
    /// value (1e4 for SWA layers). 0 when the model has only one RoPE base.
    /// </summary>
    public float RopeThetaSwa { get; init; }

    /// <summary>
    /// Whether the model has bias terms on the Q/K/V attention projections (e.g. Qwen models).
    /// Detected at load time by probing for "blk.0.attn_q.bias" in the GGUF tensor index.
    /// </summary>
    public bool HasAttnBias { get; init; }

    /// <summary>
    /// Whether the model also carries a bias on the attention <em>output</em> projection
    /// (<c>blk.*.attn_output.bias</c>). Qwen2 has Q/K/V bias but no output-projection bias,
    /// so this is probed independently of <see cref="HasAttnBias"/> (and is only ever true
    /// when <see cref="HasAttnBias"/> is). Mirrors llama.cpp treating <c>bo</c> as optional.
    /// </summary>
    public bool HasAttnOutputBias { get; init; }

    /// <summary>
    /// Whether the model has per-head Q/K RMSNorm (e.g. Qwen3).
    /// Detected at load time by probing for "blk.0.attn_q_norm.weight" in the GGUF tensor index.
    /// </summary>
    public bool HasQkNorm { get; init; }

    /// <summary>
    /// Whether QK-norm uses a per-channel learned weight of size <c>numHeads * headDim</c>
    /// (OLMoE) rather than a single <c>headDim</c> vector shared across heads (Qwen3).
    /// Detected at load time from <c>blk.0.attn_q_norm.weight</c>'s element count.
    /// Only meaningful when <see cref="HasQkNorm"/> and <see cref="UseL2QkNorm"/> is false.
    /// </summary>
    public bool IsPerChannelQkNorm { get; init; }

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

    /// <summary>
    /// Whether MoE router top-k weights should be renormalized to sum to 1 after
    /// selecting the top-k experts. Most architectures (Qwen3-MoE, Mixtral) do.
    /// OLMoE was trained with <c>norm_topk_prob=false</c> and uses the raw
    /// post-softmax probabilities directly — renormalizing produces wrong outputs.
    /// </summary>
    public bool NormalizeMoeTopKWeights { get; init; } = true;

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

    // ── Gemma 4 (sliding-window + per-layer head-dim + PLE) ──

    /// <summary>
    /// Sliding window size (in tokens) used by SWA layers. 0 when the model has
    /// no sliding-window attention. Gemma 4 = 512.
    /// </summary>
    public int SlidingWindowSize { get; init; }

    /// <summary>
    /// Per-Layer-Embedding (PLE) projection width. Gemma 4 E4B = 256. 0 when the
    /// model has no PLE table.
    /// </summary>
    public int PerLayerEmbeddingWidth { get; init; }

    /// <summary>Whether each layer has a post-attention RMSNorm before residual add (Gemma 4).</summary>
    public bool HasPostAttnNorm { get; init; }

    /// <summary>Whether each layer has a post-FFN RMSNorm before residual add (Gemma 4).</summary>
    public bool HasPostFfwNorm { get; init; }

    /// <summary>Whether the model carries a <c>per_layer_token_embd.weight</c> table (Gemma 4 PLE).</summary>
    public bool HasPerLayerTokenEmbd { get; init; }

    /// <summary>
    /// Whether each layer has a learned <c>layer_output_scale.weight</c> scalar applied
    /// to the layer output (Gemma 4).
    /// </summary>
    public bool HasLayerOutputScale { get; init; }

    /// <summary>
    /// FFN activation function. <see cref="FfnActivation.Silu"/> for the vast majority of
    /// architectures (LLaMA/Mistral/Qwen/etc); <see cref="FfnActivation.GeluApprox"/> for Gemma 4.
    /// </summary>
    public FfnActivation FfnActivation { get; init; } = FfnActivation.Silu;

    /// <summary>
    /// Per-layer flag: <c>true</c> when the layer uses sliding-window attention,
    /// <c>false</c> for global (full-context) attention. <c>null</c> when every
    /// layer is global. Gemma 4 follows a 5-SWA : 1-global repeating pattern.
    /// </summary>
    public IReadOnlyList<bool>? IsSwaLayer { get; init; }

    /// <summary>
    /// Per-layer KV-cache source. <c>-1</c> means the layer owns its own K/V;
    /// otherwise the layer aliases another layer's KV pages (Gemma 4
    /// <c>shared_kv_layers</c> tail). <c>null</c> when no layer shares KV.
    /// </summary>
    public IReadOnlyList<int>? KvSourceLayer { get; init; }

    /// <summary>
    /// Per-layer attention head dimension (Gemma 4 mixes 256 for SWA and 512 for
    /// global). <c>null</c> when every layer uses <see cref="HeadDim"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerHeadDim { get; init; }

    /// <summary>
    /// Per-layer RoPE rotation dimension (Gemma 4 mixes 256 for SWA and 512 for
    /// global). <c>null</c> when every layer uses <see cref="RopeDim"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerRopeDim { get; init; }

    /// <summary>
    /// Per-layer KV head count. Gemma 4 12B (dense) mixes 8 (GQA) on SWA layers
    /// and 1 (MQA) on global layers; stored in the GGUF as a per-layer
    /// <c>attention.head_count_kv</c> array. <c>null</c> when every layer uses the
    /// scalar <see cref="NumKvHeads"/>.
    /// </summary>
    public IReadOnlyList<int>? LayerKvHeads { get; init; }

    /// <summary>
    /// When <c>true</c>, attention reuses the K projection as V on the layers that
    /// omit a <c>attn_v.weight</c> tensor (Gemma 4 12B global layers,
    /// <c>attention_k_eq_v=true</c> in the HF config). Such layers carry no V
    /// projection; the K output doubles as the value stream. <c>false</c> for the
    /// usual separate-K/V layout.
    /// </summary>
    public bool AttentionKEqV { get; init; }

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
        // The output-projection bias is optional even when Q/K/V bias is present (Qwen2 omits it).
        bool hasAttnOutputBias = hasAttnBias
            && (metadata.ContainsKey("_sharpi.has_attn_output_bias")
                || (model?.FindTensor("blk.0.attn_output.bias") is not null));
        bool hasQkNorm = metadata.ContainsKey("_sharpi.has_qk_norm")
            || (model?.FindTensor("blk.0.attn_q_norm.weight") is not null);
        bool perChannelQkNorm = false;
        if (hasQkNorm && model is not null)
        {
            var qNormInfo = model.FindTensor("blk.0.attn_q_norm.weight");
            int numHeadsTmp = GetInt(metadata, $"{arch}.attention.head_count");
            int embDimTmp = GetInt(metadata, $"{arch}.embedding_length");
            int headDimMetaTmp = GetInt(metadata, $"{arch}.attention.key_length");
            int headDimTmp = headDimMetaTmp > 0 ? headDimMetaTmp
                : (numHeadsTmp > 0 ? embDimTmp / numHeadsTmp : embDimTmp);
            if (qNormInfo is not null && headDimTmp > 0 && numHeadsTmp > 0)
                perChannelQkNorm = qNormInfo.Value.ElementCount >= (long)numHeadsTmp * headDimTmp;
        }
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

        bool isGemma4 = arch.Equals("gemma4", StringComparison.OrdinalIgnoreCase);

        int slidingWindow = 0;
        int perLayerEmbedWidth = 0;
        float finalLogitSoftcap = 0f;
        float ropeThetaSwa = 0f;
        float embeddingScale = 1f;
        bool hasPostAttnNorm = false;
        bool hasPostFfwNorm = false;
        bool hasPerLayerTokenEmbd = false;
        bool hasLayerOutputScale = false;
        FfnActivation ffnActivation = FfnActivation.Silu;
        IReadOnlyList<bool>? isSwaLayer = null;
        IReadOnlyList<int>? kvSourceLayer = null;
        IReadOnlyList<int>? layerHeadDim = null;
        IReadOnlyList<int>? layerRopeDim = null;
        IReadOnlyList<int>? layerKvHeads = null;
        bool attentionKEqV = false;

        if (isGemma4)
        {
            slidingWindow         = GetInt(metadata, $"{arch}.attention.sliding_window");
            perLayerEmbedWidth    = GetInt(metadata, $"{arch}.embedding_length_per_layer_input");
            finalLogitSoftcap     = GetFloat(metadata, $"{arch}.final_logit_softcapping");
            ropeThetaSwa          = GetFloat(metadata, $"{arch}.rope.freq_base_swa", 10_000f);
            int sharedKvLayers    = GetInt(metadata, $"{arch}.attention.shared_kv_layers");
            int keyLengthSwa      = GetInt(metadata, $"{arch}.attention.key_length_swa", headDim);
            int ropeDimSwa        = GetInt(metadata, $"{arch}.rope.dimension_count_swa", keyLengthSwa);

            embeddingScale = MathF.Sqrt(embDim);
            ffnActivation  = FfnActivation.GeluApprox;

            hasPostAttnNorm      = metadata.ContainsKey("_sharpi.has_post_attn_norm")
                || (model?.FindTensor("blk.0.post_attention_norm.weight") is not null);
            hasPostFfwNorm       = metadata.ContainsKey("_sharpi.has_post_ffw_norm")
                || (model?.FindTensor("blk.0.post_ffw_norm.weight") is not null);
            hasPerLayerTokenEmbd = metadata.ContainsKey("_sharpi.has_ple")
                || (model?.FindTensor("per_layer_token_embd.weight") is not null);
            hasLayerOutputScale  = metadata.ContainsKey("_sharpi.has_layer_output_scale")
                || (model?.FindTensor("blk.0.layer_output_scale.weight") is not null);

            // Gemma 4 12B (dense) global layers omit attn_v and reuse K as V
            // (attention_k_eq_v=true in the HF config; not a GGUF metadata key, so it
            // is detected from the tensor inventory via a GgufModel.Open probe).
            attentionKEqV = metadata.ContainsKey("_sharpi.attention_k_eq_v");

            if (numLayers > 0)
            {
                var pattern = GetBoolArray(metadata, $"{arch}.attention.sliding_window_pattern");
                var swa = new bool[numLayers];
                if (pattern is not null && pattern.Count > 0)
                {
                    for (int i = 0; i < numLayers; i++)
                        swa[i] = pattern[i % pattern.Count];
                }
                isSwaLayer = swa;

                var hdArr = new int[numLayers];
                var rdArr = new int[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    bool sw = swa[i];
                    hdArr[i] = sw ? keyLengthSwa : headDim;
                    rdArr[i] = sw ? ropeDimSwa : ropeDim;
                }
                layerHeadDim = hdArr;
                layerRopeDim = rdArr;

                // Per-layer KV head count (Gemma 4 12B: 8 on SWA, 1 on global).
                // Stored as a per-layer array in the GGUF; build the full vector so
                // forward passes can size each layer's KV independently. Falls back
                // to the scalar head_count_kv (broadcast) when stored as a scalar.
                var kvArr = GetIntArray(metadata, $"{arch}.attention.head_count_kv");
                if (kvArr is not null && kvArr.Count > 0)
                {
                    var lkv = new int[numLayers];
                    for (int i = 0; i < numLayers; i++)
                    {
                        // Guard a corrupt/0 KV head count → it would divide-by-zero in the
                        // attention group-size calc (_numHeads / kvHeads) downstream.
                        int val = kvArr[i % kvArr.Count];
                        lkv[i] = val > 0 ? val : 1;
                    }
                    layerKvHeads = lkv;
                }

                if (sharedKvLayers > 0)
                {
                    int firstSharedLayer = numLayers - sharedKvLayers;
                    var src = new int[numLayers];
                    for (int i = 0; i < numLayers; i++)
                    {
                        if (i < firstSharedLayer)
                        {
                            src[i] = -1;
                        }
                        else
                        {
                            int found = -1;
                            for (int j = firstSharedLayer - 1; j >= 0; j--)
                            {
                                if (swa[j] == swa[i]) { found = j; break; }
                            }
                            src[i] = found;
                        }
                    }
                    kvSourceLayer = src;
                }
            }
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
            HasAttnOutputBias = hasAttnOutputBias,
            HasQkNorm = hasQkNorm,
            IsPerChannelQkNorm = perChannelQkNorm,
            IsMoE = isMoE,
            NumExperts = numExperts,
            NumActiveExperts = numActiveExperts,
            ExpertIntermediateDim = GetInt(metadata, $"{arch}.expert_feed_forward_length",
                                       GetInt(metadata, $"{arch}.feed_forward_length")),
            HasSharedExpert = hasSharedExpert,
            // OLMoE was trained without top-k renormalization. Other softmax-gated
            // MoE architectures (Qwen3-MoE, Mixtral, qwen35moe) renormalize.
            NormalizeMoeTopKWeights = !arch.Equals("olmoe", StringComparison.OrdinalIgnoreCase),
            NoRopeLayerStep = noRopeStep,
            UseSigmoidGating = useSigmoidGating,
            UseL2QkNorm = useL2QkNorm,
            IsNeoxRope = isNeoxRope,
            RopeDim = ropeDim,
            IsHybridSsm = isHybridSsm,
            LayerTypes = layerTypes,
            Gdn = gdn,
            EmbeddingScale = embeddingScale,
            FinalLogitSoftcap = finalLogitSoftcap,
            RopeThetaSwa = ropeThetaSwa,
            SlidingWindowSize = slidingWindow,
            PerLayerEmbeddingWidth = perLayerEmbedWidth,
            HasPostAttnNorm = hasPostAttnNorm,
            HasPostFfwNorm = hasPostFfwNorm,
            HasPerLayerTokenEmbd = hasPerLayerTokenEmbd,
            HasLayerOutputScale = hasLayerOutputScale,
            FfnActivation = ffnActivation,
            IsSwaLayer = isSwaLayer,
            KvSourceLayer = kvSourceLayer,
            LayerHeadDim = layerHeadDim,
            LayerRopeDim = layerRopeDim,
            LayerKvHeads = layerKvHeads,
            AttentionKEqV = attentionKEqV,
        };
    }

    private static int GetInt(IReadOnlyDictionary<string, object> m, string key, int fallback = 0)
    {
        if (!m.TryGetValue(key, out var v)) return fallback;
        // Some keys are stored per-layer as an array (e.g. Gemma 4 12B's
        // gemma4.attention.head_count_kv = [8,8,8,8,8,1,…]). A plain Convert.ToInt32
        // throws on an array; collapse to the first element so the scalar reader
        // doesn't crash. IList covers the reader's object[] plus any typed array
        // (int[]/long[]). Per-layer consumers use GetIntArray instead.
        if (v is System.Collections.IList list) return list.Count > 0 ? Convert.ToInt32(list[0]) : fallback;
        return Convert.ToInt32(v);
    }

    private static float GetFloat(IReadOnlyDictionary<string, object> m, string key, float fallback = 0f) =>
        m.TryGetValue(key, out var v) ? Convert.ToSingle(v) : fallback;

    /// <summary>
    /// Reads a per-layer integer array (e.g. Gemma 4's per-layer
    /// <c>attention.head_count_kv</c>). Returns <c>null</c> when the key is absent
    /// or stored as a scalar (the caller then falls back to the scalar field).
    /// </summary>
    private static IReadOnlyList<int>? GetIntArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case IReadOnlyList<int> rl: return rl;          // int[]/List<int> — zero-copy
            case System.Collections.IList list:             // object[] (the reader's form), long[], … — convert
            {
                var result = new int[list.Count];
                for (int i = 0; i < list.Count; i++)
                    result[i] = Convert.ToInt32(list[i]);
                return result;
            }
            default: return null;
        }
    }

    private static IReadOnlyList<bool>? GetBoolArray(IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var v)) return null;
        switch (v)
        {
            case bool[] ba: return ba;
            case IReadOnlyList<bool> rl: return rl;
            case object[] oa:
            {
                var result = new bool[oa.Length];
                for (int i = 0; i < oa.Length; i++)
                    result[i] = Convert.ToBoolean(oa[i]);
                return result;
            }
            default: return null;
        }
    }
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
/// FFN inner activation. Most LLaMA-family models use SiLU/Swish gating; Gemma 4 uses
/// the tanh-approximation of GELU on the gate projection.
/// </summary>
public enum FfnActivation
{
    Silu = 0,
    GeluApprox = 1,
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
