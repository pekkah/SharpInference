using System.Text.Json;

namespace SharpInference.Engine;

/// <summary>
/// Parsed config.json of a DSpark draft head (deepseek-ai/DeepSpec, e.g.
/// <c>dspark_qwen3_4b_block7</c>). The backbone is a small qwen3-style
/// transformer whose K/V context comes from the TARGET model's tapped hidden
/// states (<see cref="TargetLayerIds"/>) fused through an <c>fc</c> projection;
/// draft positions are mask tokens decoded in one bidirectional block of
/// <see cref="BlockSize"/> positions, then bias-corrected sequentially by a
/// rank-<see cref="MarkovRank"/> Markov head (docs/dspark-plan.md, PR #413).
/// Parsed with the JsonDocument DOM (trim/AOT-safe, no serializer context needed).
/// </summary>
public sealed record DSparkConfig
{
    public required int HiddenSize { get; init; }
    public required int HeadDim { get; init; }
    public required int NumHeads { get; init; }
    public required int NumKvHeads { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumLayers { get; init; }
    public required int BlockSize { get; init; }
    public required int MaskTokenId { get; init; }
    public required int[] TargetLayerIds { get; init; }
    public required int NumTargetLayers { get; init; }
    public required int MarkovRank { get; init; }
    public required string MarkovHeadType { get; init; }
    public required bool EnableConfidenceHead { get; init; }
    public required bool ConfidenceHeadWithMarkov { get; init; }
    public required int VocabSize { get; init; }
    public required float RmsNormEps { get; init; }
    public required float RopeTheta { get; init; }
    public required int MaxPositionEmbeddings { get; init; }

    /// <summary>Width of one target tap row: tapped layer count × target hidden size.</summary>
    public int TapDim => TargetLayerIds.Length * HiddenSize;

    public static DSparkConfig FromJsonFile(string path) => FromJson(File.ReadAllText(path));

    public static DSparkConfig FromJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        static int I(JsonElement e, string name) => e.GetProperty(name).GetInt32();
        static bool B(JsonElement e, string name, bool def = false) =>
            e.TryGetProperty(name, out var v) ? v.GetBoolean() : def;

        // rope_theta lives under rope_parameters on transformers >= 5.x configs,
        // top-level on older ones.
        float ropeTheta = 10_000f;
        if (root.TryGetProperty("rope_parameters", out var rp)
            && rp.TryGetProperty("rope_theta", out var rt))
            ropeTheta = rt.GetSingle();
        else if (root.TryGetProperty("rope_theta", out var rtTop))
            ropeTheta = rtTop.GetSingle();

        var idsEl = root.GetProperty("target_layer_ids");
        var targetLayerIds = new int[idsEl.GetArrayLength()];
        int ti = 0;
        foreach (var el in idsEl.EnumerateArray()) targetLayerIds[ti++] = el.GetInt32();

        var cfg = new DSparkConfig
        {
            HiddenSize = I(root, "hidden_size"),
            HeadDim = I(root, "head_dim"),
            NumHeads = I(root, "num_attention_heads"),
            NumKvHeads = I(root, "num_key_value_heads"),
            IntermediateSize = I(root, "intermediate_size"),
            NumLayers = I(root, "num_hidden_layers"),
            BlockSize = I(root, "block_size"),
            MaskTokenId = I(root, "mask_token_id"),
            TargetLayerIds = targetLayerIds,
            NumTargetLayers = I(root, "num_target_layers"),
            MarkovRank = I(root, "markov_rank"),
            MarkovHeadType = root.TryGetProperty("markov_head_type", out var mht)
                ? mht.GetString() ?? "vanilla" : "vanilla",
            EnableConfidenceHead = B(root, "enable_confidence_head"),
            ConfidenceHeadWithMarkov = B(root, "confidence_head_with_markov"),
            VocabSize = I(root, "vocab_size"),
            RmsNormEps = root.GetProperty("rms_norm_eps").GetSingle(),
            RopeTheta = ropeTheta,
            MaxPositionEmbeddings = I(root, "max_position_embeddings"),
        };

        cfg.Validate(root);
        return cfg;
    }

    private void Validate(JsonElement root)
    {
        if (BlockSize < 1)
            throw new NotSupportedException($"DSpark block_size must be >= 1, got {BlockSize}.");
        if (MaskTokenId < 0 || MaskTokenId >= VocabSize)
            throw new NotSupportedException(
                $"DSpark mask_token_id {MaskTokenId} outside [0, {VocabSize - 1}] — the mask " +
                "token indexes the head's own embedding table.");
        if (TargetLayerIds.Length == 0)
            throw new NotSupportedException("DSpark target_layer_ids must not be empty.");
        int prev = int.MinValue;
        foreach (int id in TargetLayerIds)
        {
            // -1 (embedding output) is legal per the reference but no released head
            // uses it; the tap plumbing only captures layer outputs, so reject it.
            if (id < 0 || id >= NumTargetLayers)
                throw new NotSupportedException(
                    $"DSpark target_layer_id {id} outside [0, {NumTargetLayers - 1}] " +
                    "(embedding taps, id -1, are not supported).");
            if (id <= prev)
                throw new NotSupportedException("DSpark target_layer_ids must be strictly increasing.");
            prev = id;
        }
        if (MarkovRank > 0 && !string.Equals(MarkovHeadType, "vanilla", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException(
                $"DSpark markov_head_type '{MarkovHeadType}' is not supported yet (only 'vanilla'; " +
                "'gated'/'rnn' heads need extra tensors and step state).");
        if (EnableConfidenceHead && ConfidenceHeadWithMarkov && MarkovRank <= 0)
            throw new NotSupportedException(
                "DSpark confidence_head_with_markov requires markov_rank > 0.");
        if (root.TryGetProperty("attention_bias", out var ab) && ab.GetBoolean())
            throw new NotSupportedException("DSpark attention_bias=true is not supported.");
        if (root.TryGetProperty("tie_word_embeddings", out var tie) && tie.GetBoolean())
            throw new NotSupportedException(
                "DSpark tie_word_embeddings=true is not supported (the loader expects lm_head.weight).");
        if (root.TryGetProperty("layer_types", out var lt))
            foreach (var t in lt.EnumerateArray())
                if (t.GetString() != "full_attention")
                    throw new NotSupportedException(
                        $"DSpark layer_type '{t.GetString()}' is not supported (full_attention only).");
    }
}
