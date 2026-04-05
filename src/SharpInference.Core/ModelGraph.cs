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

        int embDim = GetInt(metadata, $"{arch}.embedding_length");
        int numHeads = GetInt(metadata, $"{arch}.attention.head_count");
        // Some models (e.g. Qwen3-MoE) use a head dim that differs from embDim/numHeads.
        // Read from metadata if available; fall back to computed value.
        int headDimFromMeta = GetInt(metadata, $"{arch}.attention.key_length");
        int headDim = headDimFromMeta > 0 ? headDimFromMeta : (numHeads > 0 ? embDim / numHeads : embDim);

        return new ModelHyperparams
        {
            VocabSize = GetInt(metadata, $"{arch}.vocab_size"),
            ContextLength = GetInt(metadata, $"{arch}.context_length"),
            EmbeddingDim = embDim,
            NumLayers = GetInt(metadata, $"{arch}.block_count"),
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
