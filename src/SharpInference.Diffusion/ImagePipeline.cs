using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Diffusion.TextEncoders;

namespace SharpInference.Diffusion;

/// <summary>
/// Full FLUX.1 image generation pipeline.
///
/// Components loaded:
///   ditPath        — FLUX DiT weights as GGUF (.gguf)
///   vaePath        — FLUX VAE as safetensors (.safetensors)
///   clipPath       — CLIP-L encoder as safetensors (.safetensors)
///   clipTokenizerPath — tokenizer.json for CLIP BPE tokenizer
///   t5Path         — T5-XXL encoder as safetensors (.safetensors)
///   t5TokenizerPath — tokenizer.json for T5 unigram tokenizer
///
/// Usage:
///   var pipeline = ImagePipeline.Load(...);
///   pipeline.Generate("a cat", width: 512, height: 512, steps: 4, output: "out.png");
/// </summary>
public sealed class ImagePipeline : IDisposable
{
    private readonly FluxDiT _dit;
    private readonly VaeDecoder _vae;
    private readonly ClipLEncoder _clip;
    private readonly T5Encoder _t5;
    private readonly ClipTokenizer _clipTok;
    private readonly T5Tokenizer _t5Tok;
    private readonly FluxParams _params;
    private readonly CpuBackend _backend;
    private bool _disposed;

    private ImagePipeline(FluxDiT dit, VaeDecoder vae, ClipLEncoder clip, T5Encoder t5,
                           ClipTokenizer clipTok, T5Tokenizer t5Tok,
                           FluxParams p, CpuBackend backend)
    {
        _dit     = dit;
        _vae     = vae;
        _clip    = clip;
        _t5      = t5;
        _clipTok = clipTok;
        _t5Tok   = t5Tok;
        _params  = p;
        _backend = backend;
    }

    /// <summary>
    /// Load all pipeline components. All paths are required unless noted.
    /// </summary>
    public static ImagePipeline Load(
        string ditPath,
        string vaePath,
        string clipPath,
        string clipTokenizerPath,
        string t5Path,
        string t5TokenizerPath)
    {
        var model   = GgufModel.Open(ditPath);
        var meta    = model.Metadata;
        var p       = FluxParams.FromMetadata(meta);
        var backend = new CpuBackend();
        var dit     = new FluxDiT(model, p, backend);
        var vae     = new VaeDecoder(vaePath);
        var clip    = new ClipLEncoder(clipPath);
        var t5      = new T5Encoder(t5Path);
        var clipTok = ClipTokenizer.FromFile(clipTokenizerPath);
        var t5Tok   = T5Tokenizer.FromFile(t5TokenizerPath);
        return new ImagePipeline(dit, vae, clip, t5, clipTok, t5Tok, p, backend);
    }

    /// <summary>
    /// Generate an image from a text prompt and save as PNG.
    /// </summary>
    /// <param name="prompt">Text description of the desired image.</param>
    /// <param name="width">Output width in pixels (must be divisible by 16).</param>
    /// <param name="height">Output height in pixels (must be divisible by 16).</param>
    /// <param name="steps">Denoising steps (4 = schnell quality, 20-28 = dev quality).</param>
    /// <param name="guidance">Guidance scale (1.0 for schnell; 3.5 for dev).</param>
    /// <param name="seed">Random seed (-1 = random).</param>
    /// <param name="outputPath">Path to write the output PNG.</param>
    /// <param name="progress">Optional progress callback (step, totalSteps).</param>
    public void Generate(
        string prompt,
        int width = 512, int height = 512,
        int steps = 4, float guidance = 1.0f,
        int seed = -1,
        string outputPath = "output.png",
        Action<int, int>? progress = null)
    {
        if (width % 16 != 0 || height % 16 != 0)
            throw new ArgumentException("Width and height must be divisible by 16.");

        int latH = height / _params.VaeScaleFactor;
        int latW = width  / _params.VaeScaleFactor;
        int latC = _params.LatentChannels;  // 16

        // ── 1. Encode text ────────────────────────────────────────────────
        var clipTokens = _clipTok.Tokenize(prompt);
        var (_, pooledEmbed) = _clip.Encode(clipTokens);   // [768]

        var t5Tokens   = _t5Tok.Tokenize(prompt);
        var txtEmbeds  = _t5.Encode(t5Tokens);             // [seq, 4096]
        int nTxt = t5Tokens.Length;

        // ── 2. Build position ids ─────────────────────────────────────────
        int pH = latH / _params.PatchSize;
        int pW = latW / _params.PatchSize;
        int nImg = pH * pW;

        var imgIds = Flux2DRoPE.ImagePatchIds(pH, pW);     // [nImg, 2]
        var txtIds = new int[nTxt * 2];                    // all zeros

        // ── 3. Sample initial noise ───────────────────────────────────────
        var noise = EulerFlowScheduler.SampleNoise(latC * latH * latW, seed);

        // Pack noise into patch sequence
        var noisePacked = EulerFlowScheduler.PackLatent(noise, latC, latH, latW, _params.PatchSize);

        // ── 4. Denoising loop ─────────────────────────────────────────────
        float timeShift = _params.HasGuidanceIn ? 3.0f : 1.0f; // dev vs schnell
        var scheduler = EulerFlowScheduler.Linear(steps, timeShift);

        var denoised = scheduler.Denoise(
            noisePacked,
            (x, t) => _dit.Forward(x, imgIds, txtEmbeds, txtIds, pooledEmbed, t, guidance),
            progress);

        // ── 5. Unpack and decode ──────────────────────────────────────────
        var latent  = EulerFlowScheduler.UnpackLatent(denoised, latC, latH, latW, _params.PatchSize);
        var pixels  = _vae.Decode(latent, latH, latW);    // [3, H, W] in [0,1]

        // ── 6. Write PNG ──────────────────────────────────────────────────
        PngWriter.Write(outputPath, pixels, width, height);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _dit.Dispose();
            _vae.Dispose();
            _clip.Dispose();
            _t5.Dispose();
            _backend.Dispose();
        }
    }
}
