using SharpInference.Core;
using SharpInference.Diffusion.TextEncoders;

namespace SharpInference.Diffusion;

/// <summary>
/// Full Z-Image-Turbo image generation pipeline.
///
/// Components:
///   ditDir         — transformer/ directory (multi-shard safetensors)
///                    OR single model.safetensors file
///   vaeDir         — vae/ directory (ae.safetensors or multi-shard)
///   qwenPath       — Qwen3-4B GGUF file (text encoder)
///   tokenizerPath  — tokenizer.json for Qwen3 BPE tokenizer
///
/// Model download (HuggingFace):
///   https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
///   https://huggingface.co/unsloth/Z-Image-Turbo-GGUF  (quantized DiT)
///   Qwen3-4B GGUF: https://huggingface.co/Qwen/Qwen3-4B-GGUF
///
/// Usage:
///   var pipeline = ZImagePipeline.Load(ditDir, vaeDir, qwenPath, tokenizerPath);
///   pipeline.Generate("a cat", width: 512, height: 512, output: "out.png");
/// </summary>
public sealed class ZImagePipeline : IDisposable
{
    private readonly ZImageDiT _dit;
    private readonly VaeDecoder _vae;
    private readonly QwenTextEncoder _encoder;
    private readonly QwenTokenizer _tokenizer;
    private readonly ZImageParams _p;
    private string? _cachedPrompt;
    private float[]? _cachedEmbeds;
    private bool _disposed;

    private ZImagePipeline(ZImageDiT dit, VaeDecoder vae,
                            QwenTextEncoder encoder, QwenTokenizer tokenizer,
                            ZImageParams p)
    {
        _dit       = dit;
        _vae       = vae;
        _encoder   = encoder;
        _tokenizer = tokenizer;
        _p         = p;
    }

    /// <summary>
    /// Load all Z-Image pipeline components.
    /// </summary>
    /// <param name="ditPath">
    ///   Path to the DiT weights: either a directory containing safetensors shards
    ///   (e.g. the Z-Image transformer/ dir) or a single .safetensors file.
    /// </param>
    /// <param name="vaePath">
    ///   Path to the VAE weights: directory or single .safetensors file.
    /// </param>
    /// <param name="qwenPath">
    ///   Path to the Qwen3-4B GGUF file used as text encoder.
    /// </param>
    /// <param name="tokenizerPath">
    ///   Path to the Qwen3 tokenizer.json file.
    /// </param>
    /// <param name="backend">
    ///   Optional GPU compute backend. When supplied, large weight projections are
    ///   accelerated via batched SGEMM on the GPU.
    /// </param>
    public static ZImagePipeline Load(string ditPath, string vaePath,
                                       string qwenPath, string tokenizerPath,
                                       IComputeBackend? backend = null)
    {
        var p = new ZImageParams();

        IWeightLoader ditLoader;
        if (string.Equals(Path.GetExtension(ditPath), ".gguf", StringComparison.OrdinalIgnoreCase))
            ditLoader = GgufWeightLoader.Open(ditPath);
        else if (Directory.Exists(ditPath))
            ditLoader = SafetensorsLoader.OpenDirectory(ditPath);
        else
            ditLoader = SafetensorsLoader.Open(ditPath);

        var vaeLoader  = Directory.Exists(vaePath) ? SafetensorsLoader.OpenDirectory(vaePath)
                                                   : SafetensorsLoader.Open(vaePath);

        var dit       = new ZImageDiT(ditLoader, p, backend);
        var vae       = new VaeDecoder(vaeLoader, backend);
        var qwen      = new QwenTextEncoder(GgufModel.Open(qwenPath), p, backend);
        var tokenizer = QwenTokenizer.FromFile(tokenizerPath);

        return new ZImagePipeline(dit, vae, qwen, tokenizer, p);
    }

    // ── Main generation entry point ───────────────────────────────────────

    /// <summary>
    /// Generate an image from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the image.</param>
    /// <param name="width">Output image width in pixels (must be divisible by 16).</param>
    /// <param name="height">Output image height in pixels (must be divisible by 16).</param>
    /// <param name="steps">Number of denoising steps (default: 9 = 8 NFEs).</param>
    /// <param name="seed">RNG seed for reproducibility (-1 = random).</param>
    /// <param name="outputPath">Path to write the output PNG file.</param>
    /// <param name="progress">Optional progress callback (step, totalSteps).</param>
    /// <param name="statusCallback">Optional status text callback for spinner updates.</param>
    public void Generate(string prompt,
                          int width  = 512,
                          int height = 512,
                          int steps  = -1,
                          int seed   = -1,
                          string outputPath = "output.png",
                          Action<int, int>? progress = null,
                          Action<string>? statusCallback = null)
    {
        if (steps < 0) steps = _p.DefaultSteps;

        int latH = height / _p.VaeScaleFactor;  // e.g. 64 for 512px
        int latW = width  / _p.VaeScaleFactor;
        int patchH = latH / _p.PatchSize;         // e.g. 32
        int patchW = latW / _p.PatchSize;
        int nImg   = patchH * patchW;

        // ── 1. Encode text ────────────────────────────────────────────────
        progress?.Invoke(0, steps + 2);
        float[] txtEmbeds;
        int[] tokenIds;
        if (_cachedPrompt == prompt && _cachedEmbeds is not null)
        {
            statusCallback?.Invoke("Using cached text embeddings…");
            txtEmbeds = _cachedEmbeds;
            tokenIds  = _tokenizer.EncodeWithTemplate(prompt);
        }
        else
        {
            statusCallback?.Invoke("Encoding text prompt (Qwen3-4B)…");
            tokenIds  = _tokenizer.EncodeWithTemplate(prompt);
            int totalEncLayers = _p.QwenEncoderLayer + 1;
            txtEmbeds = _encoder.Encode(tokenIds,
                encodeProgress: (layer, _) =>
                    statusCallback?.Invoke($"Encoding text: layer {layer + 1}/{totalEncLayers}…"));
            _cachedPrompt  = prompt;
            _cachedEmbeds  = txtEmbeds;
        }
        int nTxt = tokenIds.Length;

        // ── 2. Build position IDs ─────────────────────────────────────────
        int[] txtPosIds = ZImageRoPE.TextPosIds(nTxt);
        int[] imgPosIds = ZImageRoPE.ImagePosIds(nTxt, patchH, patchW);

        // ── 3. Sample noise latent ────────────────────────────────────────
        var noiseLatent = EulerFlowScheduler.SampleNoise(_p.LatentChannels * latH * latW, seed);

        // Pack into patches [nImg, 64] — Z-Image uses spatial-first (ky,kx,ch) ordering
        var noisePacked = EulerFlowScheduler.PackLatentSpatialFirst(noiseLatent, _p.LatentChannels, latH, latW, _p.PatchSize);

        // ── 4. Denoise with Euler flow scheduler ──────────────────────────
        // Z-Image-Turbo uses shift=3 (from scheduler_config.json)
        var scheduler = EulerFlowScheduler.Linear(steps, shift: 3.0f);

        var resultPacked = scheduler.Denoise(noisePacked, (patches, t) =>
        {
            return _dit.Forward(patches, imgPosIds, txtEmbeds, txtPosIds, t);
        }, (step, total) =>
        {
            progress?.Invoke(step + 1, steps + 2);
            statusCallback?.Invoke($"Denoising step {step + 1}/{total}…");
        });

        // ── 5. Unpack and decode through VAE ──────────────────────────────
        progress?.Invoke(steps + 1, steps + 2);
        var latent = EulerFlowScheduler.UnpackLatentSpatialFirst(resultPacked, _p.LatentChannels, latH, latW, _p.PatchSize);

        // VaeDecoder.Decode() applies the FLUX VAE scaling internally:
        //   z = latent / 0.3611 + 0.1159
        // so we pass the raw scheduler output directly.
        float[] rgb = _vae.Decode(latent, latH, latW);

        // ── 6. Write PNG ──────────────────────────────────────────────────
        PngWriter.Write(outputPath, rgb, width, height);
        progress?.Invoke(steps + 2, steps + 2);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _dit.Dispose();
            _vae.Dispose();
            _encoder.Dispose();
        }
    }
}
