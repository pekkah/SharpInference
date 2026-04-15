namespace SharpInference.Diffusion;

/// <summary>
/// Architecture constants for FLUX.1 models (schnell and dev share the same transformer shape).
/// Values are derived from the GGUF metadata where present; hard-coded defaults match FLUX.1.
/// </summary>
public sealed class FluxParams
{
    // Transformer dims
    public int HiddenSize    { get; init; } = 3072;   // image stream dim
    public int NumHeads      { get; init; } = 24;
    public int HeadDim       => HiddenSize / NumHeads; // 128

    // Block counts
    public int DoubleBlocks  { get; init; } = 19;
    public int SingleBlocks  { get; init; } = 38;

    // Patch / latent
    public int InChannels    { get; init; } = 64;     // 2×2 patch × 16 latent channels
    public int OutChannels   { get; init; } = 64;
    public int PatchSize     { get; init; } = 2;
    public int LatentChannels{ get; init; } = 16;

    // Text / conditioning
    public int ContextDim    { get; init; } = 4096;   // T5-XXL output dim
    public int VecDim        { get; init; } = 768;    // CLIP-L pooled dim

    // Timestep MLP hidden
    public int TimeEmbDim    => HiddenSize * 4;       // 12288
    public int VecEmbDim     => HiddenSize * 4;

    // QK-norm eps
    public float QkNormEps   { get; init; } = 1e-6f;

    // Whether the model has guidance conditioning (dev=yes, schnell=no)
    public bool HasGuidanceIn { get; init; } = false;

    // VAE spatial compression factor
    public int VaeScaleFactor { get; init; } = 8;

    /// <summary>Derive params from GGUF metadata keys (falls back to FLUX.1 defaults).</summary>
    public static FluxParams FromMetadata(IReadOnlyDictionary<string, object> meta)
    {
        int Get(string key, int def) =>
            meta.TryGetValue(key, out var v) && v is int i ? i : def;

        // FLUX GGUF stores these under the "flux" architecture prefix
        bool hasGuidance = meta.ContainsKey("flux.guidance_embed");

        return new FluxParams
        {
            HiddenSize    = Get("flux.hidden_size",           3072),
            NumHeads      = Get("flux.num_attention_heads",   24),
            DoubleBlocks  = Get("flux.num_double_layers",     19),
            SingleBlocks  = Get("flux.num_single_layers",     38),
            InChannels    = Get("flux.in_channels",           64),
            ContextDim    = Get("flux.context_in_dim",        4096),
            VecDim        = Get("flux.vec_in_dim",            768),
            HasGuidanceIn = hasGuidance,
        };
    }
}
