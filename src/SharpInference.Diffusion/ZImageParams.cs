namespace SharpInference.Diffusion;

/// <summary>
/// Hyperparameters for the Z-Image-Turbo / Z-Image family.
/// Values are taken from Tongyi-MAI/Z-Image-Turbo/transformer/config.json.
///
/// Architecture: Scalable Single-Stream DiT (S3-DiT)
///   — All tokens (text + image patches) concatenated into one sequence.
///   — 30 transformer blocks with adaLN modulation (256-dim bottleneck).
///   — 2 context refiners (text only, no modulation).
///   — 2 noise refiners (image only, with modulation).
///   — 3-axis RoPE: (t=time/sequence, h=row, w=col).
///
/// Text encoder: Qwen3-4B (36 layers, GQA 32/8, hidden=2560)
///   — pipeline takes hidden_states[-2] = layer 34 output (before final norm).
///
/// VAE: same FLUX ae.safetensors (scale=0.3611, shift=0.1159).
/// Steps: 8 (guidance=0, distilled via Decoupled-DMD).
/// </summary>
public sealed class ZImageParams
{
    // ── DiT ───────────────────────────────────────────────────────────────
    public int Dim            { get; init; } = 3840;
    public int NHeads         { get; init; } = 30;
    public int NLayers        { get; init; } = 30;
    public int NRefinerLayers { get; init; } = 2;
    public int CapFeatDim     { get; init; } = 2560;  // Qwen3-4B hidden size
    public int InChannels     { get; init; } = 16;
    public int PatchSize      { get; init; } = 2;
    public float NormEps      { get; init; } = 1e-5f;
    public float RopeTheta    { get; init; } = 256.0f;
    public float TScale       { get; init; } = 1000.0f;

    /// <summary>RoPE frequency dims per axis: [t, h, w].</summary>
    public int[] AxesDims { get; init; } = [32, 48, 48];

    /// <summary>Maximum position per axis: [t, h, w].</summary>
    public int[] AxesLens { get; init; } = [1536, 512, 512];

    /// <summary>AdaLN conditioning bottleneck dimensionality (timestep MLP output).</summary>
    public int AdalnEmbedDim { get; init; } = 256;

    // ── Derived DiT ───────────────────────────────────────────────────────
    public int HeadDim    => Dim / NHeads;                  // 128
    public int FfnHidden  => (int)(Dim * 8.0 / 3.0 + 0.5); // 10240
    public int PatchDim   => PatchSize * PatchSize * InChannels; // 64

    // ── Text encoder (Qwen3-4B) ───────────────────────────────────────────
    public int   QwenHiddenSize  { get; init; } = 2560;
    public int   QwenNumLayers   { get; init; } = 36;
    public int   QwenNumHeads    { get; init; } = 32;
    public int   QwenNumKvHeads  { get; init; } = 8;
    public int   QwenHeadDim     { get; init; } = 128;
    public int   QwenIntermSize  { get; init; } = 9728;
    public float QwenRopeTheta   { get; init; } = 1_000_000f;
    public float QwenRmsNormEps  { get; init; } = 1e-6f;

    /// <summary>
    /// Layer index whose output is used as text features (hidden_states[-2] = layer 34).
    /// The pipeline runs layers 0..QwenEncoderLayer inclusive, then returns.
    /// </summary>
    public int QwenEncoderLayer => QwenNumLayers - 2;  // 34

    // ── VAE ───────────────────────────────────────────────────────────────
    public int VaeScaleFactor { get; init; } = 8;
    public int LatentChannels { get; init; } = 16;

    // ── Inference defaults ────────────────────────────────────────────────
    /// <summary>8 NFEs (9 scheduler steps). guidance_scale = 0.</summary>
    public int   DefaultSteps    { get; init; } = 9;
    public float DefaultGuidance { get; init; } = 0.0f;
}
