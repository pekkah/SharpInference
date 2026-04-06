using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Diffusion;

/// <summary>
/// Z-Image Scalable Single-Stream DiT (S3-DiT) forward pass.
///
/// Architecture (30 layers + 2 refiners each):
///   x_embedder:       Linear(64, 3840)   — project image patches
///   cap_embedder:     [RMSNorm(2560) → Linear(2560, 3840)]  — project Qwen3 text features
///   t_embedder:       TimestepMLP → [256]   — timestep conditioning
///   context_refiner × 2:  transformer blocks (no modulation) on text tokens only
///   noise_refiner × 2:    transformer blocks (with modulation) on image tokens only
///   layers × 30:          single-stream blocks on [txt | img] concat with adaLN(256-dim)
///   final_layer:      AdaLN + Linear(3840, 64)  — project image patches back
///
/// Weight names follow diffusers naming (Tongyi-MAI/Z-Image-Turbo/transformer/).
/// Supports both single-file safetensors and multi-shard directories.
/// </summary>
public sealed class ZImageDiT : IDisposable
{
    private readonly IWeightLoader _st;
    private readonly ZImageParams _p;
    private readonly ZImageRoPE _rope;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _cache = new(StringComparer.Ordinal);
    /// <summary>Raw quantized weight buffers cached on the GPU (lazy-populated by GPU dequant path).</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeights;
    /// <summary>bf16 dequantized weights cached on GPU — uploaded once on first use, reused every step.</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeightsBf16;
    /// <summary>fp8 E4M3 weights cached on GPU — uploaded once on first use, reused every step (sm_89+).</summary>
    private readonly Dictionary<string, Core.Tensor>? _gpuWeightsFp8;
    private bool _disposed;

    /// <summary>Minimum batch size to route a MatQ call through the GPU backend.</summary>
    private const int MinGpuBatch = 16;

    public ZImageDiT(IWeightLoader st, ZImageParams p, IComputeBackend? backend = null)
    {
        _st      = st;
        _p       = p;
        _rope    = new ZImageRoPE(p);
        _backend = backend;
        if (backend?.SupportsGpuDequant == true)
            _gpuWeights = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
        // Cache bf16 weights on GPU so each weight is uploaded only once across all denoising steps.
        if (backend?.BestSgemmPrecision == SgemmPrecision.Bf16)
            _gpuWeightsBf16 = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
        // Cache fp8 weights on GPU (sm_89+, 2× smaller than bf16, use fp8 tensor cores).
        if (backend?.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
            _gpuWeightsFp8 = new Dictionary<string, Core.Tensor>(StringComparer.Ordinal);
    }

    // ── Main forward pass ─────────────────────────────────────────────────

    /// <summary>
    /// Run the S3-DiT forward pass for one denoising step.
    /// </summary>
    /// <param name="imgPatches">Noisy image patches [nImg, 64].</param>
    /// <param name="imgPosIds">Image patch position IDs [nImg*3].</param>
    /// <param name="txtEmbeds">Qwen3 text hidden states [nTxt, 2560].</param>
    /// <param name="txtPosIds">Text position IDs [nTxt*3].</param>
    /// <param name="t">Fractional timestep in [0,1].</param>
    /// <returns>Predicted velocity (packed patches) [nImg, 64].</returns>
    public float[] Forward(
        float[] imgPatches, int[] imgPosIds,
        float[] txtEmbeds,  int[] txtPosIds,
        float t)
    {
        int nImg = imgPatches.Length / _p.PatchDim;
        int nTxt = txtEmbeds.Length / _p.CapFeatDim;
        int dim  = _p.Dim;

        // ── 1. Embed ──────────────────────────────────────────────────────
        var imgHid = MatQ(imgPatches, nImg, _p.PatchDim, "x_embedder.weight", "x_embedder.bias", dim);
        var txtHid = EmbedCap(txtEmbeds, nTxt);
        var adaln  = TimestepEmbed(t);           // [256]

        // ── 2. Build per-group RoPE freqs ─────────────────────────────────
        // Refiners use separate img/txt freqs; main layers use combined [img, txt].
        var imgFreqs = _rope.BuildFreqs(imgPosIds, nImg);
        var txtFreqs = _rope.BuildFreqs(txtPosIds, nTxt);

        // ── 3. Refine separately ──────────────────────────────────────────
        // Context refiner: 2 unmodulated blocks on text tokens with text RoPE
        for (int r = 0; r < _p.NRefinerLayers; r++)
            ApplyBlock($"context_refiner.{r}", txtHid, nTxt, txtFreqs, null, false);

        // Noise refiner: 2 modulated blocks on image tokens with image RoPE
        for (int r = 0; r < _p.NRefinerLayers; r++)
            ApplyBlock($"noise_refiner.{r}", imgHid, nImg, imgFreqs, adaln, true);

        // ── 4. Concatenate + combined RoPE ────────────────────────────────
        // Reference order: [img, txt] — image tokens first, text tokens after.
        int nTotal = nImg + nTxt;
        var x = new float[nTotal * dim];
        Array.Copy(imgHid, 0, x, 0,         nImg * dim);
        Array.Copy(txtHid, 0, x, nImg * dim, nTxt * dim);

        // Build combined position IDs and RoPE freqs: [imgPos, txtPos]
        var combinedPos = new int[(nImg + nTxt) * 3];
        Array.Copy(imgPosIds, 0, combinedPos, 0,        nImg * 3);
        Array.Copy(txtPosIds, 0, combinedPos, nImg * 3, nTxt * 3);
        var freqs = _rope.BuildFreqs(combinedPos, nTotal);

        // ── 4. 30 main transformer blocks ─────────────────────────────────
        for (int l = 0; l < _p.NLayers; l++)
            ApplyBlock($"layers.{l}", x, nTotal, freqs, adaln, true);

        // ── 5. Extract image portion (first nImg tokens) and apply final layer
        var imgOut = new float[nImg * dim];
        Array.Copy(x, 0, imgOut, 0, nImg * dim);

        return FinalLayer(imgOut, nImg, adaln);
    }

    // ── Embedding helpers ─────────────────────────────────────────────────

    private float[] EmbedCap(float[] txtFeats, int nTxt)
    {
        var normed = new float[txtFeats.Length];
        var normW  = W("cap_embedder.0.weight");
        for (int i = 0; i < nTxt; i++)
        {
            int off = i * _p.CapFeatDim;
            DiffusionOps.RmsNorm(txtFeats, off, _p.CapFeatDim, normW, _p.NormEps, normed, off);
        }
        return MatQ(normed, nTxt, _p.CapFeatDim, "cap_embedder.1.weight", "cap_embedder.1.bias", _p.Dim);
    }

    private float[] TimestepEmbed(float t)
    {
        // Reference convention: t_model=0 at pure noise, t_model=1 at clean image.
        // Our scheduler sigma=t: t=1 at noise, t=0 at clean.
        // So t_model = 1 - t.
        float tScaled = (1f - t) * _p.TScale;
        int   freqDim = 256;
        var   sinEmb  = new float[freqDim];
        int   half    = freqDim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000f) * i / half);
            float v    = tScaled * freq;
            sinEmb[i]        = MathF.Cos(v);
            sinEmb[i + half] = MathF.Sin(v);
        }
        var h = MatQ(sinEmb, 1, freqDim, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias", 1024);
        DiffusionOps.SiLUInPlace(h);
        return MatQ(h, 1, 1024, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias", _p.AdalnEmbedDim);
    }

    // ── Block implementation ───────────────────────────────────────────────

    private void ApplyBlock(string prefix, float[] x, int nTok,
                            float[]? freqs, float[]? adaln, bool modulated)
    {
        int dim = _p.Dim;

        float[] scaleMsa, gateMsa, scaleMlp, gateMlp;
        if (modulated)
        {
            // adaLN_modulation: Linear(256, 4*dim) + bias → 4 vectors each [dim]
            var mod = MatQ(adaln!, 1, _p.AdalnEmbedDim,
                           $"{prefix}.adaLN_modulation.0.weight",
                           $"{prefix}.adaLN_modulation.0.bias",
                           4 * dim);
            scaleMsa = mod.AsSpan(0,       dim).ToArray();
            gateMsa  = mod.AsSpan(dim,     dim).ToArray();
            scaleMlp = mod.AsSpan(2 * dim, dim).ToArray();
            gateMlp  = mod.AsSpan(3 * dim, dim).ToArray();
            TensorPrimitives.Tanh(gateMsa.AsSpan(), gateMsa.AsSpan());
            TensorPrimitives.Tanh(gateMlp.AsSpan(), gateMlp.AsSpan());
            TensorPrimitives.Add(scaleMsa.AsSpan(), 1f, scaleMsa.AsSpan());
            TensorPrimitives.Add(scaleMlp.AsSpan(), 1f, scaleMlp.AsSpan());
        }
        else
        {
            scaleMsa = scaleMlp = Ones(dim);
            gateMsa  = gateMlp  = Ones(dim);
        }

        // ── Attention sub-block ───────────────────────────────────────────
        var normW1 = W($"{prefix}.attention_norm1.weight");
        var attnIn = new float[nTok * dim];
        for (int t = 0; t < nTok; t++)
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(x, off, dim, normW1, _p.NormEps, attnIn, off);
            TensorPrimitives.Multiply(attnIn.AsSpan(off, dim), scaleMsa, attnIn.AsSpan(off, dim));
        }

        var attnOut = SelfAttention(prefix, attnIn, nTok, freqs);

        var normW2 = W($"{prefix}.attention_norm2.weight");
        for (int t = 0; t < nTok; t++)
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(attnOut, off, dim, normW2, _p.NormEps, attnOut, off);
            TensorPrimitives.MultiplyAdd<float>(attnOut.AsSpan(off, dim), gateMsa,
                                                x.AsSpan(off, dim), x.AsSpan(off, dim));
        }

        // ── FFN sub-block ─────────────────────────────────────────────────
        var normW3 = W($"{prefix}.ffn_norm1.weight");
        var ffnIn  = new float[nTok * dim];
        for (int t = 0; t < nTok; t++)
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(x, off, dim, normW3, _p.NormEps, ffnIn, off);
            TensorPrimitives.Multiply(ffnIn.AsSpan(off, dim), scaleMlp, ffnIn.AsSpan(off, dim));
        }

        // SiLU-gated FFN: w2(silu(w1(x)) * w3(x))
        var gate   = MatQ(ffnIn, nTok, dim, $"{prefix}.feed_forward.w1.weight", _p.FfnHidden);
        var up     = MatQ(ffnIn, nTok, dim, $"{prefix}.feed_forward.w3.weight", _p.FfnHidden);
        DiffusionOps.SiluInPlace(gate.AsSpan());                              // gate = SiLU(gate)
        TensorPrimitives.Multiply(gate.AsSpan(), up.AsSpan(), gate.AsSpan()); // gate *= up
        var ffnOut = MatQ(gate, nTok, _p.FfnHidden, $"{prefix}.feed_forward.w2.weight", dim);

        var normW4 = W($"{prefix}.ffn_norm2.weight");
        for (int t = 0; t < nTok; t++)
        {
            int off = t * dim;
            DiffusionOps.RmsNorm(ffnOut, off, dim, normW4, _p.NormEps, ffnOut, off);
            TensorPrimitives.MultiplyAdd<float>(ffnOut.AsSpan(off, dim), gateMlp,
                                                x.AsSpan(off, dim), x.AsSpan(off, dim));
        }
    }

    // ── Self-attention ────────────────────────────────────────────────────

    private float[] SelfAttention(string prefix, float[] x, int nTok, float[]? freqs)
    {
        int dim     = _p.Dim;
        int nHeads  = _p.NHeads;
        int headDim = _p.HeadDim;

        // Fused QKV: [nTok, 3*dim] — split into separate q, k, v
        var qkv = MatQ(x, nTok, dim, $"{prefix}.attention.qkv.weight", 3 * dim);
        var q = new float[nTok * dim];
        var k = new float[nTok * dim];
        var v = new float[nTok * dim];
        for (int tok = 0; tok < nTok; tok++)
        {
            int srcOff = tok * 3 * dim;
            int dstOff = tok * dim;
            qkv.AsSpan(srcOff,           dim).CopyTo(q.AsSpan(dstOff));
            qkv.AsSpan(srcOff + dim,     dim).CopyTo(k.AsSpan(dstOff));
            qkv.AsSpan(srcOff + 2 * dim, dim).CopyTo(v.AsSpan(dstOff));
        }

        // Per-head QK norm
        var qNormW = W($"{prefix}.attention.q_norm.weight");
        var kNormW = W($"{prefix}.attention.k_norm.weight");
        ApplyPerHeadRmsNorm(q, nTok, nHeads, headDim, qNormW);
        ApplyPerHeadRmsNorm(k, nTok, nHeads, headDim, kNormW);

        if (freqs != null)
        {
            _rope.Apply(q, nTok, nHeads, freqs);
            _rope.Apply(k, nTok, nHeads, freqs);
        }

        float scale      = 1f / MathF.Sqrt(headDim);
        var   attn       = new float[nTok * dim];
        int   scoreCount = nHeads * nTok * nTok;
        var   scoresBuf  = ArrayPool<float>.Shared.Rent(scoreCount);

        try
        {
            // Parallelise over heads — Span locals inside the lambda are fine
            Parallel.For(0, nHeads, h =>
            {
                int sBase = h * nTok * nTok;

                // QK scores using SIMD dot products
                for (int i = 0; i < nTok; i++)
                {
                    var qi   = q.AsSpan((i * nHeads + h) * headDim, headDim);
                    int sRow = sBase + i * nTok;
                    for (int j = 0; j < nTok; j++)
                    {
                        var kj = k.AsSpan((j * nHeads + h) * headDim, headDim);
                        scoresBuf[sRow + j] = TensorPrimitives.Dot<float>(qi, kj) * scale;
                    }
                    DiffusionOps.Softmax(scoresBuf, sRow, nTok);
                }

                // Extract contiguous per-head V for vectorized value aggregation
                var vhBuf = ArrayPool<float>.Shared.Rent(nTok * headDim);
                try
                {
                    for (int j = 0; j < nTok; j++)
                        v.AsSpan((j * nHeads + h) * headDim, headDim)
                         .CopyTo(vhBuf.AsSpan(j * headDim));

                    for (int i = 0; i < nTok; i++)
                    {
                        int    sRow  = sBase + i * nTok;
                        var    outSl = attn.AsSpan((i * nHeads + h) * headDim, headDim);
                        outSl.Clear();
                        for (int j = 0; j < nTok; j++)
                        {
                            TensorPrimitives.MultiplyAdd<float>(
                                vhBuf.AsSpan(j * headDim, headDim),
                                scoresBuf[sRow + j],
                                outSl, outSl);
                        }
                    }
                }
                finally { ArrayPool<float>.Shared.Return(vhBuf); }
            });
        }
        finally { ArrayPool<float>.Shared.Return(scoresBuf); }

        return MatQ(attn, nTok, dim, $"{prefix}.attention.out.weight", dim);
    }

    private static void ApplyPerHeadRmsNorm(float[] x, int nTok, int nHeads, int headDim,
                                            float[] normW)
    {
        float eps = 1e-5f;
        var   wSp = normW.AsSpan(0, headDim);
        for (int t = 0; t < nTok; t++)
        for (int h = 0; h < nHeads; h++)
        {
            var row  = x.AsSpan((t * nHeads + h) * headDim, headDim);
            float ss = TensorPrimitives.Dot<float>(row, row);
            float r  = 1f / MathF.Sqrt(ss / headDim + eps);
            TensorPrimitives.Multiply(row, wSp, row);
            TensorPrimitives.Multiply(row, r,   row);
        }
    }

    // ── Final layer ───────────────────────────────────────────────────────

    private float[] FinalLayer(float[] imgHid, int nImg, float[] adaln)
    {
        int dim    = _p.Dim;
        int outDim = _p.PatchDim;  // 64
        string fl  = "final_layer";

        // adaLN_modulation: SiLU(adaln) → Linear(256, dim) + bias → scale
        var adaln_silu = ArrayPool<float>.Shared.Rent(_p.AdalnEmbedDim);
        try
        {
            adaln.AsSpan(0, _p.AdalnEmbedDim).CopyTo(adaln_silu.AsSpan(0, _p.AdalnEmbedDim));
            DiffusionOps.SiluInPlace(adaln_silu.AsSpan(0, _p.AdalnEmbedDim));
            var scale = MatQ(adaln_silu, 1, _p.AdalnEmbedDim,
                             $"{fl}.adaLN_modulation.1.weight",
                             $"{fl}.adaLN_modulation.1.bias", dim);
            TensorPrimitives.Add(scale.AsSpan(), 1f, scale.AsSpan()); // scale = 1 + scale

            // LayerNorm (elementwise_affine=False) + scale
            var normed = new float[nImg * dim];
            for (int t = 0; t < nImg; t++)
            {
                var row = imgHid.AsSpan(t * dim, dim);
                float mean = TensorPrimitives.Sum(row) / dim;
                TensorPrimitives.Subtract(row, mean, row);
                float varr = TensorPrimitives.Dot<float>(row, row) / dim;
                float rstd = 1f / MathF.Sqrt(varr + 1e-6f);
                var dst = normed.AsSpan(t * dim, dim);
                TensorPrimitives.Multiply(row, rstd,  dst);
                TensorPrimitives.Multiply(dst, scale, dst);
            }

            return MatQ(normed, nImg, dim, $"{fl}.linear.weight", $"{fl}.linear.bias", outDim);
        }
        finally { ArrayPool<float>.Shared.Return(adaln_silu); }
    }

    // ── Weight and matmul helpers ─────────────────────────────────────────

    /// <summary>
    /// Quantized matrix multiply using zero-copy mmap pointer from GGUF.
    /// When a GPU backend is provided and the batch is large enough, dequantizes
    /// the weight to float32 and uses the backend's batched SGEMM.
    /// Falls back to DiffusionOps.Linear for float32 safetensors backends.
    /// </summary>
    private unsafe float[] MatQ(float[] x, int n, int inDim, string wName, int outDim,
                                 string? bName = null)
    {
        var result = new float[n * outDim];

        if (_st.TryGetRaw(wName, out nint ptr, out long byteLen, out DType dtype, out int rows, out int cols))
        {
            if (_backend != null && n >= MinGpuBatch)
            {
                if (_backend.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
                {
                    // fp8 E4M3 GPU path (sm_89+): weights stored as fp8 (1 byte/element, 2× smaller than fp16).
                    // Both A (activations) and B (weights) must be fp8 for cublasGemmEx; output C is fp32.
                    // Weights are uploaded once on first call and cached; reused on subsequent steps.
                    int wCount = rows * cols;
                    byte[] xFp8 = ArrayPool<byte>.Shared.Rent(n * cols);
                    try
                    {
                        // Convert activations to fp8
                        int xCount = n * cols;
                        for (int i = 0; i < xCount; i++)
                            xFp8[i] = Fp8Converter.FloatToFp8E4M3(x[i]);

                        // Get or upload the fp8 weight (cached after first call)
                        Core.Tensor wGpu;
                        if (_gpuWeightsFp8 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K) &&
                            _gpuWeightsFp8.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            byte[] wFp8 = ArrayPool<byte>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(
                                    new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                                    wBuf32.AsSpan(0, wCount), dtype, wCount);
                                for (int i = 0; i < wCount; i++)
                                    wFp8[i] = Fp8Converter.FloatToFp8E4M3(wBuf32[i]);
                                wGpu = _backend.UploadFp8(wFp8.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<byte>.Shared.Return(wFp8);
                            }
                            if (_gpuWeightsFp8 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K))
                                _gpuWeightsFp8[wName] = wGpu;
                        }

                        bool ownW = _gpuWeightsFp8 == null || !_gpuWeightsFp8.ContainsKey(wName);
                        var xGpu = _backend.UploadFp8(xFp8.AsSpan(0, xCount), TensorShape.D1(xCount));
                        // Output as fp32 to avoid bf16→fp32 conversion step on download
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.Float32);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(xFp8);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Bf16)
                {
                    // bf16 GPU path: dequant weight to bf16, upload once and cache on GPU.
                    // Each weight matrix is uploaded only on first call; reused every subsequent step.
                    int wCount = rows * cols;
                    ushort[] xBf16 = ArrayPool<ushort>.Shared.Rent(n * cols);
                    ushort[] cBf16 = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        // Convert activations to bf16
                        int xCount = n * cols;
                        for (int i = 0; i < xCount; i++)
                        {
                            uint bits = BitConverter.SingleToUInt32Bits(x[i]);
                            xBf16[i] = (ushort)(bits >> 16);
                        }

                        // Get or upload the bf16 weight (cached after first call)
                        Core.Tensor wGpu;
                        if (_gpuWeightsBf16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K) &&
                            _gpuWeightsBf16.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            ushort[] wBf16 = ArrayPool<ushort>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(
                                    new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                                    wBuf32.AsSpan(0, wCount), dtype, wCount);
                                for (int i = 0; i < wCount; i++)
                                {
                                    uint bits = BitConverter.SingleToUInt32Bits(wBuf32[i]);
                                    wBf16[i] = (ushort)(bits >> 16);
                                }
                                wGpu = _backend.UploadBf16(wBf16.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<ushort>.Shared.Return(wBf16);
                            }
                            if (_gpuWeightsBf16 != null && (dtype == DType.Q5_K || dtype == DType.Q4_K))
                                _gpuWeightsBf16[wName] = wGpu;
                        }

                        bool ownW = _gpuWeightsBf16 == null || !_gpuWeightsBf16.ContainsKey(wName);
                        var xGpu = _backend.UploadBf16(xBf16.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.DownloadBf16(cGpu, cBf16.AsSpan(0, n * rows));
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }

                        int cCount = n * rows;
                        for (int i = 0; i < cCount; i++)
                        {
                            uint bits = (uint)cBf16[i] << 16;
                            result[i] = BitConverter.UInt32BitsToSingle(bits);
                        }
                    }
                    finally
                    {
                        ArrayPool<ushort>.Shared.Return(xBf16);
                        ArrayPool<ushort>.Shared.Return(cBf16);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16 &&
                         _gpuWeights != null &&
                         (dtype == DType.Q5_K || dtype == DType.Q4_K))
                {
                    // GPU dequant path: upload raw quantized bytes once, dequant on GPU each call.
                    // SgemmF16: A=fp32 activations, B=fp16 weights, C=fp32 output (no fp16 overflow).
                    if (!_gpuWeights.TryGetValue(wName, out var rawGpu))
                    {
                        rawGpu = _backend.UploadRaw(
                            new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                            TensorShape.D1((long)byteLen), dtype);
                        _gpuWeights[wName] = rawGpu;
                    }

                    int numBlocks = rows * cols / 256;
                    int xCount   = n * cols;
                    int cCount   = n * rows;
                    var wGpu = _backend.Allocate(TensorShape.D1(rows * cols), DType.Float16);
                    try
                    {
                        if (dtype == DType.Q5_K)
                            _backend.DequantQ5KM(rawGpu, wGpu, numBlocks);
                        else
                            _backend.DequantQ4KM(rawGpu, wGpu, numBlocks);

                        // Upload activations as fp32 (not fp16) — avoids overflow for large intermediate values
                        // (e.g. SiLU(w1) * w3 can exceed fp16 range of 65504).
                        // SgemmF16 reads A as fp32, B as fp16 — weights get fp16 bandwidth savings only.
                        var xGpu = _backend.Upload(x.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(cCount), DType.Float32);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        _backend.Free(wGpu);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16)
                {
                    // fp16-weight GPU path: A=fp32 activations (no overflow), B=fp16 weights (bandwidth), C=fp32
                    int wCount = rows * cols;
                    float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                    Half[] wHalf   = ArrayPool<Half>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(
                            new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                            wBuf32.AsSpan(0, wCount), dtype, wCount);

                        int xCount = n * cols;
                        for (int i = 0; i < wCount; i++) wHalf[i] = (Half)wBuf32[i];

                        // Upload activations as fp32 — avoids overflow for large intermediates
                        var xGpu = _backend.Upload(x.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var wGpu = _backend.UploadHalf(wHalf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.Float32);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(wBuf32);
                        ArrayPool<Half>.Shared.Return(wHalf);
                    }
                }
                else
                {
                    // fp32 GPU path: dequantize weight on CPU, upload A + B, run SGEMM, download C
                    int wCount = rows * cols;
                    float[] wBuf = ArrayPool<float>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(
                            new ReadOnlySpan<byte>((byte*)ptr, (int)byteLen),
                            wBuf.AsSpan(0, wCount), dtype, wCount);

                        var xGpu = _backend.Upload(x.AsSpan(0, n * cols), TensorShape.D1(n * cols));
                        var wGpu = _backend.Upload(wBuf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows));
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result.AsSpan());
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally { ArrayPool<float>.Shared.Return(wBuf); }
                }
            }
            else
            {
                // CPU path
                fixed (float* xPtr = x, rPtr = result)
                    SimdKernels.MatMulBatched(rPtr, (byte*)ptr, xPtr, n, rows, cols, dtype);
            }
        }
        else
        {
            // Safetensors float32 path (VAE decoder, test mocks)
            DiffusionOps.Linear(x, _st.ReadF32(wName), null, n, inDim, outDim)
                        .AsSpan().CopyTo(result);
        }

        if (bName is not null)
        {
            var bias = _st.ReadF32(bName);
            for (int b = 0; b < n; b++)
                TensorPrimitives.Add(result.AsSpan(b * outDim, outDim),
                                     bias.AsSpan(), result.AsSpan(b * outDim, outDim));
        }

        return result;
    }

    private float[] MatQ(float[] x, int n, int inDim, string wName, string bName, int outDim)
        => MatQ(x, n, inDim, wName, outDim, bName);

    /// <summary>Read and cache a small weight (norm weights, q/k-norm weights).</summary>
    private float[] W(string name)
    {
        if (_cache.TryGetValue(name, out var c)) return c;
        var v = _st.ReadF32(name);
        _cache[name] = v;
        return v;
    }

    private static float[] Ones(int dim)
    {
        var v = new float[dim];
        v.AsSpan().Fill(1f);
        return v;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_gpuWeights != null && _backend != null)
            {
                foreach (var t in _gpuWeights.Values)
                    _backend.Free(t);
                _gpuWeights.Clear();
            }
            if (_gpuWeightsBf16 != null && _backend != null)
            {
                foreach (var t in _gpuWeightsBf16.Values)
                    _backend.Free(t);
                _gpuWeightsBf16.Clear();
            }
            if (_gpuWeightsFp8 != null && _backend != null)
            {
                foreach (var t in _gpuWeightsFp8.Values)
                    _backend.Free(t);
                _gpuWeightsFp8.Clear();
            }
            _cache.Clear();
            _st.Dispose();
        }
    }
}
