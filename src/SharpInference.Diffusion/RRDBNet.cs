using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Core;
using CoreTensor = SharpInference.Core.Tensor;

namespace SharpInference.Diffusion;

/// <summary>
/// ESRGAN / Real-ESRGAN image upscaler — RRDBNet architecture.
///
/// Loads weights from a .safetensors file and auto-detects all hyperparameters:
///   • num_feat     — feature channels (e.g. 64)
///   • num_block    — number of RRDB blocks (e.g. 23)
///   • num_grow_ch  — growth channels per dense layer (e.g. 32)
///   • scale        — upscale factor: 1, 2, or 4 (auto-detected from conv_first input channels)
///   • upsample style — "new" (nearest + conv) vs "old" (PixelShuffle via upconv keys)
///
/// Weight naming conventions handled:
///   • Direct:  conv_first.weight, body.0.rdb1.conv1.weight, …
///   • Prefixed: model.conv_first.weight, model.body.0.rdb1.conv1.weight, …
///
/// Scale-2 models produce the same spatial size as input from the neural network forward
/// pass. The <see cref="Upscale"/> method bilinear-resizes the output to 2× after inference,
/// matching the behaviour of the official Real-ESRGAN inference pipeline.
///
/// Tiled processing is automatically engaged for images wider or taller than
/// <c>tileSize</c> pixels, with configurable overlap to hide tile seams.
/// </summary>
public sealed class RRDBNet : IDisposable
{
    private readonly SafetensorsLoader _st;
    private readonly string _prefix;        // "" or "model."
    private readonly int _numFeat;          // feature channels (e.g. 64)
    private readonly int _numBlock;         // RRDB blocks (e.g. 23)
    private readonly int _numGrowCh;        // dense growth channels (e.g. 32)
    private readonly bool _usePixelShuffle; // true = old upconv style; false = new conv_up style
    private readonly IComputeBackend? _backend;
    private readonly IImageOpsBackend? _imageOps;
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly Dictionary<string, float[]> _weights = new(StringComparer.Ordinal);
    private bool _disposed;

    // Im2col chunk size: limit each col matrix upload to 128 MB (32M fp32 values)
    private const int MaxColChunkFloats = 32 * 1024 * 1024;

    /// <summary>Upscale factor: 1, 2, or 4.</summary>
    public int Scale { get; }

    // ── Construction ──────────────────────────────────────────────────────

    private RRDBNet(SafetensorsLoader st, string prefix,
                    int numFeat, int numBlock, int numGrowCh,
                    int scale, bool usePixelShuffle, IComputeBackend? backend)
    {
        _st              = st;
        _prefix          = prefix;
        _numFeat         = numFeat;
        _numBlock        = numBlock;
        _numGrowCh       = numGrowCh;
        Scale            = scale;
        _usePixelShuffle = usePixelShuffle;
        _backend         = backend;
        _imageOps        = backend as IImageOpsBackend;
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    /// <summary>
    /// Load an RRDBNet upscaler from a .safetensors model file.
    /// Architecture hyperparameters (scale, num_feat, num_block, num_grow_ch) are
    /// inferred automatically from weight shapes.
    /// </summary>
    /// <param name="path">Path to the .safetensors model file.</param>
    /// <param name="backend">
    ///   Optional GPU compute backend. When supplied, all Conv2D operations are
    ///   dispatched as SGEMM on the GPU via im2col, matching VaeDecoder's acceleration.
    ///   Weights are cached in VRAM after first upload.
    /// </param>
    public static RRDBNet Load(string path, IComputeBackend? backend = null)
    {
        var st = SafetensorsLoader.Open(path);
        try
        {
            var (prefix, numFeat, numBlock, numGrowCh, scale, usePixelShuffle) =
                DetectArchitecture(st);
            return new RRDBNet(st, prefix, numFeat, numBlock, numGrowCh, scale, usePixelShuffle, backend);
        }
        catch
        {
            st.Dispose();
            throw;
        }
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Upscale an image.
    /// </summary>
    /// <param name="image">CHW float32 values in [0, 1], layout [3, height, width].</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="tileSize">
    ///   Max tile dimension for large images. Images larger than this will be processed
    ///   tile-by-tile to bound peak memory usage.
    /// </param>
    /// <param name="overlap">
    ///   Overlap (in pixels) between adjacent tiles; larger values reduce visible seams
    ///   at the cost of extra computation.
    /// </param>
    /// <returns>
    ///   Upscaled CHW float32 image plus the actual output width and height.
    ///   For scale=4 these are width×4 and height×4; for scale=2 they are width×2 and height×2.
    /// </returns>
    public (float[] pixels, int outWidth, int outHeight) Upscale(
        float[] image, int width, int height,
        int tileSize = 512, int overlap = 16)
    {
        // For scale=4 the network itself doubles spatial twice → 4×.
        // For scale=2 the network outputs the same spatial size as input (pixel_unshuffle
        // inside forward cancels the single 2× upsample); we bilinear-resize afterwards.
        int modelScale = Scale == 4 ? 4 : 1;
        int outW = width  * Scale;
        int outH = height * Scale;

        float[] result = (width <= tileSize && height <= tileSize)
            ? Forward(image, height, width)
            : ProcessTiled(image, width, height, tileSize, overlap, modelScale);

        // scale=2: bilinear resize from [3, H, W] → [3, 2H, 2W]
        if (Scale == 2)
            result = DiffusionOps.UpsampleBilinear(result, 1, 3, height, width, outH, outW);

        return (result, outW, outH);
    }

    // ── Tiled processing ──────────────────────────────────────────────────

    private float[] ProcessTiled(float[] image, int width, int height,
                                  int tileSize, int overlap, int modelScale)
    {
        int outW = width  * modelScale;
        int outH = height * modelScale;
        var output = new float[3 * outH * outW];

        for (int tileY = 0; tileY < height; tileY += tileSize)
        for (int tileX = 0; tileX < width;  tileX += tileSize)
        {
            // Padded tile bounds (includes overlap, clamped to image edges)
            int padX1 = Math.Max(0, tileX - overlap);
            int padY1 = Math.Max(0, tileY - overlap);
            int padX2 = Math.Min(width,  tileX + tileSize + overlap);
            int padY2 = Math.Min(height, tileY + tileSize + overlap);
            int tw = padX2 - padX1;
            int th = padY2 - padY1;

            // Extract padded tile and run forward pass
            var tileIn  = ExtractTile(image, width, height, padX1, padY1, tw, th);
            var tileOut = Forward(tileIn, th, tw);

            // Region in tileOut that corresponds to the valid (non-overlap) input area
            int leftSkip = (tileX - padX1) * modelScale;
            int topSkip  = (tileY - padY1) * modelScale;
            int validW   = Math.Min(tileX + tileSize, width)  - tileX;
            int validH   = Math.Min(tileY + tileSize, height) - tileY;
            int validOutW = validW * modelScale;
            int validOutH = validH * modelScale;

            int tileOutW = tw * modelScale;
            int outX = tileX * modelScale;
            int outY = tileY * modelScale;

            for (int c = 0; c < 3; c++)
            for (int row = 0; row < validOutH; row++)
            {
                Array.Copy(
                    tileOut, c * th * modelScale * tileOutW + (topSkip + row) * tileOutW + leftSkip,
                    output,  c * outH * outW + (outY + row) * outW + outX,
                    validOutW);
            }
        }

        return output;
    }

    private static float[] ExtractTile(float[] image, int width, int height,
                                        int x1, int y1, int tw, int th)
    {
        var tile = new float[3 * th * tw];
        for (int c = 0; c < 3; c++)
        for (int row = 0; row < th; row++)
            Array.Copy(image, c * height * width + (y1 + row) * width + x1,
                       tile,  c * th * tw + row * tw, tw);
        return tile;
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    private float[] Forward(float[] x, int h, int w)
    {
        if (_imageOps is not null)
            return ForwardGpu(x, h, w);

        int n = 1;
        float[] feat;
        int featH = h, featW = w;

        // Pre-process: pixel unshuffle (only for scale=2 and scale=1 models)
        if (Scale == 2)
        {
            feat  = DiffusionOps.PixelUnshuffle(x, n, 3, h, w, 2);
            featH = h / 2;
            featW = w / 2;
        }
        else if (Scale == 1)
        {
            feat  = DiffusionOps.PixelUnshuffle(x, n, 3, h, w, 4);
            featH = h / 4;
            featW = w / 4;
        }
        else
        {
            feat = x;
        }

        // conv_first
        int inCh = feat.Length / (n * featH * featW);
        feat = ConvBlock($"{_prefix}conv_first", feat, n, inCh, featH, featW, _numFeat, 3);

        // RRDB body
        var bodyFeat = feat;
        for (int i = 0; i < _numBlock; i++)
            bodyFeat = RRDBBlock($"{_prefix}body.{i}", bodyFeat, n, featH, featW);

        // conv_body + residual connection
        bodyFeat = ConvBlock($"{_prefix}conv_body", bodyFeat, n, _numFeat, featH, featW, _numFeat, 3);
        TensorPrimitives.Add(feat.AsSpan(), bodyFeat.AsSpan(), feat.AsSpan());

        // Upsample
        int curH = featH, curW = featW;
        if (_usePixelShuffle)
        {
            // Old-style ESRGAN: Conv2D(num_feat, num_feat*4) → PixelShuffle(2)
            feat = ConvBlock($"{_prefix}upconv1", feat, n, _numFeat, curH, curW, _numFeat * 4, 3);
            feat = DiffusionOps.PixelShuffle(feat, n, _numFeat * 4, curH, curW, 2);
            DiffusionOps.LeakyReLUInPlace(feat, 0.2f);
            curH *= 2; curW *= 2;

            if (Scale == 4)
            {
                feat = ConvBlock($"{_prefix}upconv2", feat, n, _numFeat, curH, curW, _numFeat * 4, 3);
                feat = DiffusionOps.PixelShuffle(feat, n, _numFeat * 4, curH, curW, 2);
                DiffusionOps.LeakyReLUInPlace(feat, 0.2f);
                curH *= 2; curW *= 2;
            }
        }
        else
        {
            // New-style Real-ESRGAN: nearest-neighbor 2× → Conv2D
            feat = DiffusionOps.Upsample2x(feat, n, _numFeat, curH, curW);
            curH *= 2; curW *= 2;
            feat = ConvBlock($"{_prefix}conv_up1", feat, n, _numFeat, curH, curW, _numFeat, 3);
            DiffusionOps.LeakyReLUInPlace(feat, 0.2f);

            if (Scale == 4)
            {
                feat = DiffusionOps.Upsample2x(feat, n, _numFeat, curH, curW);
                curH *= 2; curW *= 2;
                feat = ConvBlock($"{_prefix}conv_up2", feat, n, _numFeat, curH, curW, _numFeat, 3);
                DiffusionOps.LeakyReLUInPlace(feat, 0.2f);
            }
        }

        // conv_hr + lrelu
        feat = ConvBlock($"{_prefix}conv_hr", feat, n, _numFeat, curH, curW, _numFeat, 3);
        DiffusionOps.LeakyReLUInPlace(feat, 0.2f);

        // conv_last
        feat = ConvBlock($"{_prefix}conv_last", feat, n, _numFeat, curH, curW, 3, 3);

        // Clamp output to [0, 1]
        for (int i = 0; i < feat.Length; i++)
            feat[i] = Math.Clamp(feat[i], 0f, 1f);

        return feat;
    }

    // ── GPU-native forward pass (IImageOpsBackend) ────────────────────────

    /// <summary>
    /// Full GPU-native forward pass: uploads the input once, keeps ALL intermediate
    /// tensors in VRAM (no PCIe round-trips between conv layers), downloads the result.
    /// Requires <see cref="_imageOps"/> to be non-null (VulkanBackend).
    /// </summary>
    private float[] ForwardGpu(float[] x, int h, int w)
    {
        var gpu = _imageOps!;
        int featH = h, featW = w;
        int inCh;

        var diagSw = System.Diagnostics.Stopwatch.StartNew();

        // Pre-warm: ensure all model weights are in VRAM before any RDB batch starts.
        // GetGpuWeight → Upload → CopyBuffer shares _transferCmd with the batch recording,
        // so all uploads must finish before the first BeginBatch() inside RDBBlockGpu.
        WarmupWeightsOnGpu(h, w);
        Console.Error.WriteLine($"[DIAG] WarmupWeights: {diagSw.Elapsed.TotalSeconds:F2}s");
        diagSw.Restart();

        // Upload input (uses CopyBuffer).
        var xGpu = gpu.Upload(x.AsSpan(), TensorShape.D1(x.Length));
        CoreTensor feat;

        // Pre-process: pixel unshuffle (scale=2 and scale=1 only)
        if (Scale == 2)
        {
            featH = h / 2; featW = w / 2;
            inCh  = 12;
            feat = gpu.PixelUnshuffleGpu(xGpu, 3, h, w, 2);
            gpu.Free(xGpu);
        }
        else if (Scale == 1)
        {
            featH = h / 4; featW = w / 4;
            inCh  = 48;
            feat = gpu.PixelUnshuffleGpu(xGpu, 3, h, w, 4);
            gpu.Free(xGpu);
        }
        else
        {
            inCh = 3;
            feat = xGpu; // scale=4: no unshuffle
        }

        // conv_first
        var convFirst = ConvBlockGpu($"{_prefix}conv_first", feat, inCh, featH, featW, _numFeat, 3);
        gpu.Free(feat);
        feat = convFirst;

        // RRDB body — each RRDB block internally batches per RDB for GPU throughput.
        var bodyFeat = feat;
        bool bodyFeatIsFeat = true;
        for (int i = 0; i < _numBlock; i++)
        {
            var next = RRDBBlockGpu($"{_prefix}body.{i}", bodyFeat, featH, featW);
            if (!bodyFeatIsFeat) gpu.Free(bodyFeat);
            bodyFeat      = next;
            bodyFeatIsFeat = false;
        }
        Console.Error.WriteLine($"[DIAG] RRDB body ({_numBlock} blocks): {diagSw.Elapsed.TotalSeconds:F2}s");
        diagSw.Restart();

        // conv_body + residual connection: feat += conv_body(bodyFeat)
        var convBodyOut = ConvBlockGpu($"{_prefix}conv_body", bodyFeat, _numFeat, featH, featW, _numFeat, 3);
        if (!bodyFeatIsFeat) gpu.Free(bodyFeat);
        gpu.AddInPlace(feat, convBodyOut);
        gpu.Free(convBodyOut);

        // Upsample
        int curH = featH, curW = featW;
        if (_usePixelShuffle)
        {
            // Old-style ESRGAN: conv(num_feat*4) → PixelShuffle(2) → LeakyReLU
            var up1In = ConvBlockGpu($"{_prefix}upconv1", feat, _numFeat, curH, curW, _numFeat * 4, 3);
            gpu.Free(feat);
            feat = gpu.PixelShuffleGpu(up1In, _numFeat * 4, curH, curW, 2);
            gpu.Free(up1In);
            gpu.LeakyReluInPlace(feat, 0.2f);
            curH *= 2; curW *= 2;

            if (Scale == 4)
            {
                var up2In = ConvBlockGpu($"{_prefix}upconv2", feat, _numFeat, curH, curW, _numFeat * 4, 3);
                gpu.Free(feat);
                feat = gpu.PixelShuffleGpu(up2In, _numFeat * 4, curH, curW, 2);
                gpu.Free(up2In);
                gpu.LeakyReluInPlace(feat, 0.2f);
                curH *= 2; curW *= 2;
            }
        }
        else
        {
            // New-style Real-ESRGAN: nearest-neighbor 2× → conv → LeakyReLU
            var up1 = gpu.Upsample2xGpu(feat, _numFeat, curH, curW);
            gpu.Free(feat);
            curH *= 2; curW *= 2;
            feat = ConvBlockGpu($"{_prefix}conv_up1", up1, _numFeat, curH, curW, _numFeat, 3);
            gpu.Free(up1);
            gpu.LeakyReluInPlace(feat, 0.2f);

            if (Scale == 4)
            {
                var up2 = gpu.Upsample2xGpu(feat, _numFeat, curH, curW);
                gpu.Free(feat);
                curH *= 2; curW *= 2;
                feat = ConvBlockGpu($"{_prefix}conv_up2", up2, _numFeat, curH, curW, _numFeat, 3);
                gpu.Free(up2);
                gpu.LeakyReluInPlace(feat, 0.2f);
            }
        }

        // conv_hr + LeakyReLU
        var hr = ConvBlockGpu($"{_prefix}conv_hr", feat, _numFeat, curH, curW, _numFeat, 3);
        gpu.Free(feat);
        gpu.LeakyReluInPlace(hr, 0.2f);

        // conv_last + clamp
        var last = ConvBlockGpu($"{_prefix}conv_last", hr, _numFeat, curH, curW, 3, 3);
        gpu.Free(hr);
        gpu.ClampInPlace(last, 0f, 1f);

        var result = new float[3 * curH * curW];
        gpu.Download(last, result.AsSpan());
        gpu.Free(last);
        Console.Error.WriteLine($"[DIAG] Upsample+hr+last+download: {diagSw.Elapsed.TotalSeconds:F2}s");
        return result;
    }

    /// <summary>GPU conv2d: uploads weights once (cached), returns a new VRAM tensor.</summary>
    private CoreTensor ConvBlockGpu(string key, CoreTensor input, int inCh, int h, int w, int outCh, int k)
    {
        var wGpu = GetGpuWeight($"{key}.weight", inCh * outCh * k * k);
        var bGpu = GetGpuWeight($"{key}.bias",   outCh);
        return _imageOps!.Conv2d(input, wGpu, bGpu, inCh, outCh, h, w, k);
    }

    /// <summary>Upload a weight to VRAM and cache it for reuse across tiles and blocks.</summary>
    private CoreTensor GetGpuWeight(string name, int zeroFallbackSize)
    {
        if (_gpuWeights!.TryGetValue(name, out var cached)) return cached;
        float[] data = _st.Contains(name) ? Wt(name) : new float[zeroFallbackSize];
        var gpu = _imageOps!.Upload(data.AsSpan(), TensorShape.D1(data.Length));
        _gpuWeights[name] = gpu;
        return gpu;
    }

    /// <summary>
    /// Ensure all model weights are uploaded to VRAM before batch recording begins.
    /// <see cref="Upload"/> uses <c>CopyBuffer</c> which shares the main command buffer
    /// with batch recording — any upload inside a batch would corrupt the recording.
    /// On the first tile this actually uploads; on subsequent tiles it's a cache-hit no-op.
    /// </summary>
    private void WarmupWeightsOnGpu(int h, int w)
    {
        int inCh = Scale == 2 ? 12 : Scale == 1 ? 48 : 3;
        UploadWeight($"{_prefix}conv_first", inCh, _numFeat, 3);
        for (int i = 0; i < _numBlock; i++)
        {
            for (int r = 1; r <= 3; r++)
            {
                string rdb = $"{_prefix}body.{i}.rdb{r}";
                UploadWeight($"{rdb}.conv1", _numFeat,          _numGrowCh, 3);
                UploadWeight($"{rdb}.conv2", _numFeat +   _numGrowCh, _numGrowCh, 3);
                UploadWeight($"{rdb}.conv3", _numFeat + 2*_numGrowCh, _numGrowCh, 3);
                UploadWeight($"{rdb}.conv4", _numFeat + 3*_numGrowCh, _numGrowCh, 3);
                UploadWeight($"{rdb}.conv5", _numFeat + 4*_numGrowCh, _numFeat,   3);
            }
        }
        UploadWeight($"{_prefix}conv_body", _numFeat, _numFeat, 3);
        if (_usePixelShuffle)
        {
            UploadWeight($"{_prefix}upconv1", _numFeat, _numFeat * 4, 3);
            if (Scale == 4) UploadWeight($"{_prefix}upconv2", _numFeat, _numFeat * 4, 3);
        }
        else
        {
            UploadWeight($"{_prefix}conv_up1", _numFeat, _numFeat, 3);
            if (Scale == 4) UploadWeight($"{_prefix}conv_up2", _numFeat, _numFeat, 3);
        }
        UploadWeight($"{_prefix}conv_hr",   _numFeat, _numFeat, 3);
        UploadWeight($"{_prefix}conv_last",  _numFeat, 3,        3);
    }

    private void UploadWeight(string key, int inCh, int outCh, int k)
    {
        GetGpuWeight($"{key}.weight", inCh * outCh * k * k);
        GetGpuWeight($"{key}.bias",   outCh);
    }

    /// <summary>
    /// RRDB block (GPU): out1→out2→out3 with residual scale 0.2.
    /// Caller owns x; returns a new VRAM tensor (caller must free it).
    /// </summary>
    private CoreTensor RRDBBlockGpu(string prefix, CoreTensor x, int h, int w)
    {
        var out1 = RDBBlockGpu($"{prefix}.rdb1", x,    h, w);
        var out2 = RDBBlockGpu($"{prefix}.rdb2", out1, h, w);
        _imageOps!.Free(out1);
        var out3 = RDBBlockGpu($"{prefix}.rdb3", out2, h, w);
        _imageOps.Free(out2);
        // out3 = out3 * 0.2 + x
        _imageOps.ScaleInPlace(out3, 0.2f);
        _imageOps.AddInPlace(out3, x);
        return out3;
    }

    /// <summary>
    /// RDB block (GPU): dense connections across 5 conv layers, residual scale 0.2.
    /// Batches all ~15 dispatches into a single command buffer submission to reduce
    /// vkQueueSubmit overhead. Intermediate tensors are deferred-freed after submit.
    /// Caller owns x; returns a new VRAM tensor (caller must free it).
    /// </summary>
    private CoreTensor RDBBlockGpu(string prefix, CoreTensor x, int h, int w)
    {
        int ch = _numFeat;
        int gc = _numGrowCh;
        int hw = h * w;
        var gpu = _imageOps!;

        // Batch all dispatches in this RDB block into one submission.
        // Intermediate frees (x1, cat1, ...) are deferred until EndBatch().
        // Peak extra VRAM ≈ (gc×4 + (ch+gc×4)×1) × H×W × 4 bytes per tile.
        gpu.BeginBatch();

        var x1 = ConvBlockGpu($"{prefix}.conv1", x, ch, h, w, gc, 3);
        gpu.LeakyReluInPlace(x1, 0.2f);

        var cat1 = gpu.CatChannels(x,    ch,      x1, gc, hw);
        gpu.Free(x1);

        var x2 = ConvBlockGpu($"{prefix}.conv2", cat1, ch + gc, h, w, gc, 3);
        gpu.LeakyReluInPlace(x2, 0.2f);

        var cat2 = gpu.CatChannels(cat1, ch + gc, x2, gc, hw);
        gpu.Free(x2);
        gpu.Free(cat1);

        var x3 = ConvBlockGpu($"{prefix}.conv3", cat2, ch + 2 * gc, h, w, gc, 3);
        gpu.LeakyReluInPlace(x3, 0.2f);

        var cat3 = gpu.CatChannels(cat2, ch + 2 * gc, x3, gc, hw);
        gpu.Free(x3);
        gpu.Free(cat2);

        var x4 = ConvBlockGpu($"{prefix}.conv4", cat3, ch + 3 * gc, h, w, gc, 3);
        gpu.LeakyReluInPlace(x4, 0.2f);

        var cat4 = gpu.CatChannels(cat3, ch + 3 * gc, x4, gc, hw);
        gpu.Free(x4);
        gpu.Free(cat3);

        var x5 = ConvBlockGpu($"{prefix}.conv5", cat4, ch + 4 * gc, h, w, ch, 3);
        gpu.Free(cat4);

        // x5 = x5 * 0.2 + x
        gpu.ScaleInPlace(x5, 0.2f);
        gpu.AddInPlace(x5, x);

        // Submit batch; deferred frees (x1, cat1 … cat4) are processed here.
        gpu.EndBatch();
        return x5;
    }

    // ── RRDB + RDB blocks ─────────────────────────────────────────────────

    /// <summary>
    /// RRDB (Residual-in-Residual Dense Block): 3 RDB blocks with residual scaling 0.2.
    /// Weight key pattern: {prefix}body.{i}.rdb{1|2|3}.conv{1-5}.{weight|bias}
    /// </summary>
    private float[] RRDBBlock(string prefix, float[] x, int n, int h, int w)
    {
        var out1 = ResidualDenseBlock($"{prefix}.rdb1", x,    n, h, w);
        var out2 = ResidualDenseBlock($"{prefix}.rdb2", out1, n, h, w);
        var out3 = ResidualDenseBlock($"{prefix}.rdb3", out2, n, h, w);

        // result = out3 * 0.2 + x
        var result = new float[out3.Length];
        for (int i = 0; i < out3.Length; i++)
            result[i] = out3[i] * 0.2f + x[i];
        return result;
    }

    /// <summary>
    /// RDB (Residual Dense Block): dense connections across 5 conv layers, residual scale 0.2.
    ///
    ///   x1 = lrelu(conv1(x))
    ///   x2 = lrelu(conv2(cat(x, x1)))
    ///   x3 = lrelu(conv3(cat(x, x1, x2)))
    ///   x4 = lrelu(conv4(cat(x, x1, x2, x3)))
    ///   x5 =       conv5(cat(x, x1, x2, x3, x4))
    ///   return x5 * 0.2 + x
    /// </summary>
    private float[] ResidualDenseBlock(string prefix, float[] x, int n, int h, int w)
    {
        int ch = _numFeat;
        int gc = _numGrowCh;

        var x1 = ConvBlock($"{prefix}.conv1", x, n, ch, h, w, gc, 3);
        DiffusionOps.LeakyReLUInPlace(x1, 0.2f);

        var cat1 = CatC(x, n, ch, x1, gc, h, w);
        var x2 = ConvBlock($"{prefix}.conv2", cat1, n, ch + gc, h, w, gc, 3);
        DiffusionOps.LeakyReLUInPlace(x2, 0.2f);

        var cat2 = CatC(cat1, n, ch + gc, x2, gc, h, w);
        var x3 = ConvBlock($"{prefix}.conv3", cat2, n, ch + 2 * gc, h, w, gc, 3);
        DiffusionOps.LeakyReLUInPlace(x3, 0.2f);

        var cat3 = CatC(cat2, n, ch + 2 * gc, x3, gc, h, w);
        var x4 = ConvBlock($"{prefix}.conv4", cat3, n, ch + 3 * gc, h, w, gc, 3);
        DiffusionOps.LeakyReLUInPlace(x4, 0.2f);

        var cat4 = CatC(cat3, n, ch + 3 * gc, x4, gc, h, w);
        var x5 = ConvBlock($"{prefix}.conv5", cat4, n, ch + 4 * gc, h, w, ch, 3);

        // result = x5 * 0.2 + x
        var result = new float[x5.Length];
        for (int i = 0; i < x5.Length; i++)
            result[i] = x5[i] * 0.2f + x[i];
        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// <summary>Concatenate two NCHW tensors along the C axis.</summary>
    private static float[] CatC(float[] a, int n, int aC, float[] b, int bC, int h, int w)
    {
        int hw   = h * w;
        int outC = aC + bC;
        var result = new float[n * outC * hw];
        for (int batch = 0; batch < n; batch++)
        {
            Array.Copy(a, batch * aC * hw, result, batch * outC * hw,          aC * hw);
            Array.Copy(b, batch * bC * hw, result, batch * outC * hw + aC * hw, bC * hw);
        }
        return result;
    }

    private float[] ConvBlock(string key, float[] input, int n, int inC, int h, int w, int outC, int k)
    {
        if (_backend is not null)
            return ConvGpu(key, input, inC, h, w, outC, k);
        var weight = Wt($"{key}.weight");
        float[]? bias = _st.Contains($"{key}.bias") ? Wt($"{key}.bias") : null;
        return DiffusionOps.Conv2D(input, weight, bias, n, inC, h, w, outC, k, k);
    }

    /// <summary>
    /// GPU Conv2D via im2col + SGEMM, mirroring VaeDecoder.ConvGpu.
    /// Weights are cached in VRAM after first upload and reused across tiles/blocks.
    /// Col matrices are chunked to stay within the MaxColChunkFloats budget.
    /// </summary>
    private float[] ConvGpu(string name, float[] x, int inCh, int h, int w, int outCh, int ksize)
    {
        int padding = (ksize - 1) / 2;
        int outH = h, outW = w; // stride=1, same padding
        int hw   = outH * outW;
        int kPts = inCh * ksize * ksize;

        string wKey = $"{name}.weight";
        if (!_gpuWeights!.TryGetValue(wKey, out var wGpu))
        {
            var wf = Wt(wKey);
            wGpu = _backend!.Upload(wf.AsSpan(), TensorShape.D1(wf.Length));
            _gpuWeights[wKey] = wGpu;
        }

        var biasArr = _st.Contains($"{name}.bias") ? Wt($"{name}.bias") : null;

        int chunkRows = kPts > 0 ? Math.Max(1, Math.Min(outH, MaxColChunkFloats / (outW * kPts))) : outH;
        var output = new float[outCh * hw];

        var colBuf    = ArrayPool<float>.Shared.Rent(chunkRows * outW * kPts);
        var resultBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * outCh);
        try
        {
            for (int rowStart = 0; rowStart < outH; rowStart += chunkRows)
            {
                int rowEnd  = Math.Min(rowStart + chunkRows, outH);
                int chunkH  = rowEnd - rowStart;
                int chunkHW = chunkH * outW;
                int colSize = chunkHW * kPts;

                Im2ColChunk(x, inCh, h, w, ksize, padding, rowStart, rowEnd, colBuf);

                var colGpu = _backend!.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu   = _backend.Allocate(TensorShape.D1(chunkHW * outCh));
                try
                {
                    _backend.Sgemm(cGpu, colGpu, wGpu, chunkHW, kPts, outCh);
                    _backend.Synchronize();
                    _backend.Download(cGpu, resultBuf.AsSpan(0, chunkHW * outCh));
                }
                finally
                {
                    _backend.Free(colGpu);
                    _backend.Free(cGpu);
                }

                int basePos = rowStart * outW;
                for (int pos = 0; pos < chunkHW; pos++)
                {
                    int absPos = basePos + pos;
                    for (int oc = 0; oc < outCh; oc++)
                        output[oc * hw + absPos] = resultBuf[pos * outCh + oc] + (biasArr?[oc] ?? 0f);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(colBuf);
            ArrayPool<float>.Shared.Return(resultBuf);
        }
        return output;
    }

    /// <summary>Im2col for a horizontal strip of output rows [rowStart, rowEnd).</summary>
    private static void Im2ColChunk(float[] x, int inCh, int h, int w,
                                     int ksize, int padding,
                                     int rowStart, int rowEnd, float[] col)
    {
        int outW = w;
        int idx  = 0;
        for (int oh = rowStart; oh < rowEnd; oh++)
        for (int ow = 0; ow < outW; ow++)
        {
            for (int ic = 0; ic < inCh; ic++)
            for (int kh = 0; kh < ksize; kh++)
            for (int kw = 0; kw < ksize; kw++)
            {
                int ih = oh + kh - padding;
                int iw = ow + kw - padding;
                col[idx++] = ((uint)ih < (uint)h && (uint)iw < (uint)w)
                    ? x[ic * h * w + ih * w + iw]
                    : 0f;
            }
        }
    }

    private float[] Wt(string name)
    {
        if (_weights.TryGetValue(name, out var cached)) return cached;
        var data = _st.ReadF32(name);
        _weights[name] = data;
        return data;
    }

    // ── Architecture detection ────────────────────────────────────────────

    private static (string prefix, int numFeat, int numBlock, int numGrowCh,
                    int scale, bool usePixelShuffle)
        DetectArchitecture(SafetensorsLoader st)
    {
        // Detect key prefix ("" or "model.")
        string prefix =
            st.Contains("conv_first.weight")       ? "" :
            st.Contains("model.conv_first.weight") ? "model." :
            throw new InvalidDataException(
                "Cannot detect RRDBNet weights — expected 'conv_first.weight' or " +
                "'model.conv_first.weight'. Verify the file is an ESRGAN/Real-ESRGAN model.");

        // num_feat and scale: inferred from conv_first.weight shape [outC, inC, kH, kW]
        var firstShape = st.GetShape($"{prefix}conv_first.weight");
        int numFeat = firstShape[0];
        int inCh    = firstShape[1];

        // inCh=3 → scale=4 (no pixel unshuffle)
        // inCh=12 → scale=2 (pixel unshuffle by 2: 3×4)
        // inCh=48 → scale=1 (pixel unshuffle by 4: 3×16)
        int scale = inCh == 3 ? 4 : inCh == 12 ? 2 : inCh == 48 ? 1 : 4;

        // num_grow_ch from body.0.rdb1.conv1.weight shape [gc, numFeat, kH, kW]
        var rdb1Shape = st.GetShape($"{prefix}body.0.rdb1.conv1.weight");
        int numGrowCh = rdb1Shape[0];

        // Count RRDB blocks
        int numBlock = 0;
        while (st.Contains($"{prefix}body.{numBlock}.rdb1.conv1.weight"))
            numBlock++;
        if (numBlock == 0)
            throw new InvalidDataException("No RRDB body blocks found in weight file.");

        // Upsample style: old (upconv1) vs new (conv_up1)
        bool usePixelShuffle = st.Contains($"{prefix}upconv1.weight");

        return (prefix, numFeat, numBlock, numGrowCh, scale, usePixelShuffle);
    }

    // ── IDisposable ───────────────────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_gpuWeights is not null)
            {
                foreach (var t in _gpuWeights.Values) _backend!.Free(t);
                _gpuWeights.Clear();
            }
            _st.Dispose();
        }
    }
}
