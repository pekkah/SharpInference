using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Core;
using SharpInference.Cpu;
using CoreTensor = SharpInference.Core.Tensor;

namespace SharpInference.Diffusion;

/// <summary>
/// FLUX Multi-Modal Diffusion Transformer (MM-DiT) forward pass.
///
/// Architecture (FLUX.1-schnell / FLUX.1-dev):
///   img_in:     Linear(64, 3072)           — project image patches
///   txt_in:     Linear(4096, 3072)          — project T5 text embeddings
///   time_in:    Timestep MLP → [3072]
///   vector_in:  Pooled CLIP MLP → [3072]
///   double_blocks × 19: img/txt separate streams with cross-attention
///   single_blocks × 38: concatenated img+txt stream
///   final_layer: AdaLN + Linear(3072, 64)
///
/// Weights are loaded from a GGUF file using the existing GgufModel infrastructure.
/// All matrix multiplications go through <see cref="IComputeBackend"/> (CpuBackend or VulkanBackend).
/// </summary>
public sealed class FluxDiT : IDisposable
{
    private readonly GgufModel _model;
    private readonly FluxParams _p;
    private readonly IComputeBackend _backend;
    private bool _disposed;

    // Cached tensor lookups (lazy on first use)
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);

    /// <summary>Minimum token batch size to route a MatQ call through the GPU backend.</summary>
    private const int MinGpuBatch = 16;
    /// <summary>bf16 weights cached on GPU — uploaded once on first denoising step, reused every step.</summary>
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsBf16;
    /// <summary>fp8 E4M3 weights cached on GPU — uploaded once on first step (sm_89+, 2× smaller than bf16).</summary>
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsFp8;

    public FluxParams Params => _p;

    public FluxDiT(GgufModel model, FluxParams p, IComputeBackend backend)
    {
        _model   = model;
        _p       = p;
        _backend = backend;
        if (backend?.BestSgemmPrecision == SgemmPrecision.Bf16)
            _gpuWeightsBf16 = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
        if (backend?.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
            _gpuWeightsFp8 = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    // ── Entry point ───────────────────────────────────────────────────────

    /// <summary>
    /// Single denoising forward pass.
    /// Returns predicted velocity field (same shape as <paramref name="imgLatent"/>).
    /// </summary>
    /// <param name="imgLatent">Packed image patches [nImg, inChannels] (= [H/2·W/2, 64]).</param>
    /// <param name="imgIds">Patch (row,col) position ids [nImg, 2].</param>
    /// <param name="txtEmbeds">T5 text embeddings [nTxt, 4096].</param>
    /// <param name="txtIds">Text token ids (all zero for FLUX) [nTxt, 2].</param>
    /// <param name="pooledEmbed">CLIP pooled embed [768].</param>
    /// <param name="timestep">Scalar timestep ∈ [0, 1].</param>
    /// <param name="guidance">Guidance scale (ignored for schnell, used by dev).</param>
    public float[] Forward(
        float[] imgLatent, int[] imgIds,
        float[] txtEmbeds, int[] txtIds,
        float[] pooledEmbed,
        float timestep, float guidance = 3.5f)
    {
        int nImg = imgIds.Length / 2;
        int nTxt = txtIds.Length / 2;
        int d    = _p.HiddenSize;   // 3072

        // ── Encode conditioning ───────────────────────────────────────────
        float[] vec = ComputeVec(timestep, pooledEmbed, guidance);     // [d]
        float[] txtHidden = ProjectTxt(txtEmbeds, nTxt);               // [nTxt, d]
        float[] imgHidden = ProjectImg(imgLatent, nImg);               // [nImg, d]

        // ── Build RoPE freqs ──────────────────────────────────────────────
        // Combine img and txt ids for single-stream RoPE
        int nSeq = nImg + nTxt;
        var allIds = new int[nSeq * 2];
        imgIds.CopyTo(allIds, 0);
        // text positions: all zeros (no spatial encoding)
        // (already zero from array init)
        var (ropeC, ropeS) = Flux2DRoPE.BuildFreqs(allIds, nSeq, _p.HeadDim);

        // ── Double stream blocks ──────────────────────────────────────────
        for (int i = 0; i < _p.DoubleBlocks; i++)
            DoubleBlock(i, imgHidden, txtHidden, vec, ropeC, ropeS, nImg, nTxt);

        // ── Single stream blocks ──────────────────────────────────────────
        // Concatenate img + txt → [nSeq, d]
        var x = new float[nSeq * d];
        imgHidden.AsSpan().CopyTo(x.AsSpan(0, nImg * d));
        txtHidden.AsSpan().CopyTo(x.AsSpan(nImg * d, nTxt * d));

        for (int i = 0; i < _p.SingleBlocks; i++)
            SingleBlock(i, x, vec, ropeC, ropeS, nSeq, nImg);

        // ── Final layer ───────────────────────────────────────────────────
        imgHidden = x.AsSpan(0, nImg * d).ToArray();
        return FinalLayer(imgHidden, vec, nImg);
    }

    // ── Conditioning embedding ────────────────────────────────────────────

    private float[] ComputeVec(float timestep, float[] pooled, float guidance)
    {
        int d = _p.HiddenSize;

        // Timestep sinusoidal embedding → MLP → [d]
        float[] tEmb  = TimestepEmbedding(timestep, 256);
        float[] tProj = MlpProj("model.diffusion_model.time_in", tEmb, 256, d);

        // CLIP pooled embedding → MLP → [d]
        float[] vProj = MlpProj("model.diffusion_model.vector_in", pooled, _p.VecDim, d);

        var vec = new float[d];
        for (int i = 0; i < d; i++) vec[i] = tProj[i] + vProj[i];

        if (_p.HasGuidanceIn)
        {
            float[] gEmb  = TimestepEmbedding(guidance, 256);
            float[] gProj = MlpProj("model.diffusion_model.guidance_in", gEmb, 256, d);
            for (int i = 0; i < d; i++) vec[i] += gProj[i];
        }
        return vec;
    }

    private float[] ProjectImg(float[] imgLatent, int nImg) =>
        MatQ(imgLatent, nImg, _p.InChannels, "model.diffusion_model.img_in.weight", _p.HiddenSize,
             W("model.diffusion_model.img_in.bias"));

    private float[] ProjectTxt(float[] txtEmb, int nTxt) =>
        MatQ(txtEmb, nTxt, _p.ContextDim, "model.diffusion_model.txt_in.weight", _p.HiddenSize,
             W("model.diffusion_model.txt_in.bias"));

    // ── Double stream block ───────────────────────────────────────────────

    private void DoubleBlock(int idx,
        float[] img, float[] txt, float[] vec,
        float[] ropeC, float[] ropeS,
        int nImg, int nTxt)
    {
        int d  = _p.HiddenSize;
        int nh = _p.NumHeads;
        int hd = _p.HeadDim;
        string pi = $"model.diffusion_model.double_blocks.{idx}";

        // adaLN modulation: Linear(d, 6d) × silu for each stream
        float[] imgMod = AdaLNMod($"{pi}.img_mod.lin", vec, d, 6);
        float[] txtMod = AdaLNMod($"{pi}.txt_mod.lin", vec, d, 6);

        // Img attention
        var imgNorm = RmsNormMod(img, imgMod, nImg, d, shift: 0, scale: 1);
        var (imgQ, imgK, imgV) = QKV($"{pi}.img_attn.qkv", $"{pi}.img_attn.norm", imgNorm, nImg, d, nh, hd);

        // Txt attention
        var txtNorm = RmsNormMod(txt, txtMod, nTxt, d, shift: 0, scale: 1);
        var (txtQ, txtK, txtV) = QKV($"{pi}.txt_attn.qkv", $"{pi}.txt_attn.norm", txtNorm, nTxt, d, nh, hd);

        // Apply 2D RoPE to img Q,K only
        Flux2DRoPE.ApplyInPlace(imgQ, ropeC, ropeS, nImg, nh, hd, nImg);
        Flux2DRoPE.ApplyInPlace(imgK, ropeC, ropeS, nImg, nh, hd, nImg);

        // Joint attention: img and txt attend to each other
        var (imgAttn, txtAttn) = JointAttention(imgQ, imgK, imgV, txtQ, txtK, txtV, nImg, nTxt, nh, hd);

        // Project + residual for img
        var imgOut = LinearBias($"{pi}.img_attn.proj", imgAttn, nImg, d, d);
        ScaleGateAdd(img, imgOut, imgMod, nImg, d, gateIdx: 2);

        // img MLP with adaLN
        var imgNorm2 = RmsNormMod(img, imgMod, nImg, d, shift: 3, scale: 4);
        var imgMlp   = GeluMlp($"{pi}.img_mlp", imgNorm2, nImg, d);
        ScaleGateAdd(img, imgMlp, imgMod, nImg, d, gateIdx: 5);

        // Project + residual for txt
        var txtOut = LinearBias($"{pi}.txt_attn.proj", txtAttn, nTxt, d, d);
        ScaleGateAdd(txt, txtOut, txtMod, nTxt, d, gateIdx: 2);

        // txt MLP
        var txtNorm2 = RmsNormMod(txt, txtMod, nTxt, d, shift: 3, scale: 4);
        var txtMlp   = GeluMlp($"{pi}.txt_mlp", txtNorm2, nTxt, d);
        ScaleGateAdd(txt, txtMlp, txtMod, nTxt, d, gateIdx: 5);
    }

    // ── Single stream block ───────────────────────────────────────────────

    private void SingleBlock(int idx, float[] x, float[] vec,
                              float[] ropeC, float[] ropeS, int nSeq, int nImg)
    {
        int d  = _p.HiddenSize;
        int nh = _p.NumHeads;
        int hd = _p.HeadDim;
        string p = $"model.diffusion_model.single_blocks.{idx}";

        // adaLN modulation: Linear(d, 3d)
        float[] mod = AdaLNMod($"{p}.modulation.lin", vec, d, 3);

        var xNorm = RmsNormMod(x, mod, nSeq, d, shift: 0, scale: 1);

        // Fused linear: [qkv | mlp_in], weight = [3d+4d, d] = [7d, d] ... actually FLUX single blocks
        // combine attn (3d) + MLP gate (4d) in one linear1: output [7d] per token
        var lin1 = LinearNoBias($"{p}.linear1", xNorm, nSeq, d, d * 3 + d * 4);
        // Split into q[d], k[d], v[d], mlp[4d]
        float[] q    = Slice(lin1, nSeq, 0,     d,    d * 7);
        float[] k    = Slice(lin1, nSeq, d,     d,    d * 7);
        float[] v    = Slice(lin1, nSeq, d * 2, d,    d * 7);
        float[] mlpH = Slice(lin1, nSeq, d * 3, d * 4, d * 7);

        // Reshape q,k,v to [nSeq, nh, hd]
        q = Reshape2Heads(q, nSeq, nh, hd);
        k = Reshape2Heads(k, nSeq, nh, hd);
        v = Reshape2Heads(v, nSeq, nh, hd);

        // QK norm (per-head)
        QKNorm($"{p}.norm", q, k, nSeq, nh, hd);

        // 2D RoPE on img part of q, k
        Flux2DRoPE.ApplyInPlace(q, ropeC, ropeS, nSeq, nh, hd, nImg);
        Flux2DRoPE.ApplyInPlace(k, ropeC, ropeS, nSeq, nh, hd, nImg);

        // Self-attention over full sequence
        float[] attnOut = SelfAttention(q, k, v, nSeq, nh, hd);

        // GEGLU activation on mlpH: split into two halves, y = x1 * gelu(x2)
        var mlpOut = Geglu(mlpH, nSeq, d * 4);

        // linear2: concatenate attn_out [d] + mlp_out [2d] → project to [d]
        var combined = new float[nSeq * (d + d * 2)];
        for (int i = 0; i < nSeq; i++)
        {
            attnOut.AsSpan(i * d, d).CopyTo(combined.AsSpan(i * (d + d * 2), d));
            mlpOut.AsSpan(i * d * 2, d * 2).CopyTo(combined.AsSpan(i * (d + d * 2) + d, d * 2));
        }
        // Actually linear2 is [d, d+4d/2] = [d, 3d]... let's recompute
        // FLUX single block: linear2 projects attn_out (d) + mlp_gated (d*2) back to d
        // But the standard FLUX single block fuses attention + MLP:
        //   combined output (d + mlp_out_d) → linear2 → x update
        // mlp_out_d = d*4/2 = d*2 (after GEGLU splits d*4 in half)
        // linear2: [d, d + d*2] = [d, 3d]... but some impls use [d, d] for attn only
        // Following the actual FLUX code: linear2 = [d, d+mlp_hidden] where mlp_hidden=d*4/2
        // Wait, let me reconsider. The combined is attn_out(d) + mlp_out(d*2) = 3d → linear2(d) = [d,3d]
        // Actually linear2 weight is [d, 3d] because it takes [attn_out, mlp_geglu_out] = [d + d*2]
        // And mlp hidden = 4*d, geglu cuts that in half = 2*d
        // So linear2: weight [d, 3d]
        var out_ = MatQ(combined, nSeq, d + d * 2, $"{p}.linear2", d, null);

        // Gate and residual
        ScaleGateAdd(x, out_, mod, nSeq, d, gateIdx: 2);
    }

    // ── Final layer ───────────────────────────────────────────────────────

    private float[] FinalLayer(float[] img, float[] vec, int nImg)
    {
        int d = _p.HiddenSize;
        string p = "model.diffusion_model.final_layer";

        // adaLN modulation: shift + scale
        var mod = AdaLNMod($"{p}.adaLN_modulation.1", vec, d, 2);
        var normed = RmsNormMod(img, mod, nImg, d, shift: 0, scale: 1);

        // Linear(d, outChannels)
        return MatQ(normed, nImg, d, $"{p}.linear", _p.OutChannels, W($"{p}.linear.bias"));
    }

    // ── Attention helpers ─────────────────────────────────────────────────

    private (float[] q, float[] k, float[] v) QKV(
        string qkvPath, string normPath,
        float[] x, int n, int d, int nh, int hd)
    {
        // Fused QKV projection [n, 3*d]
        var qkv = MatQ(x, n, d, $"{qkvPath}.weight", d * 3, null);

        var q = Reshape2Heads(Slice(qkv, n, 0,     d, d * 3), n, nh, hd);
        var k = Reshape2Heads(Slice(qkv, n, d,     d, d * 3), n, nh, hd);
        var v = Reshape2Heads(Slice(qkv, n, d * 2, d, d * 3), n, nh, hd);

        QKNorm(normPath, q, k, n, nh, hd);
        return (q, k, v);
    }

    private void QKNorm(string normPath, float[] q, float[] k, int n, int nh, int hd)
    {
        var qScale = W($"{normPath}.query_norm.scale");
        var kScale = W($"{normPath}.key_norm.scale");
        // Norm each head's q and k independently
        for (int i = 0; i < n * nh; i++)
        {
            DiffusionOps.RmsNorm(q.AsSpan(i * hd, hd), qScale, hd, _p.QkNormEps);
            DiffusionOps.RmsNorm(k.AsSpan(i * hd, hd), kScale, hd, _p.QkNormEps);
        }
    }

    private static (float[] imgAttn, float[] txtAttn) JointAttention(
        float[] iq, float[] ik, float[] iv,
        float[] tq, float[] tk, float[] tv,
        int nImg, int nTxt, int nh, int hd)
    {
        // Concatenate along sequence: [nImg+nTxt, nh, hd]
        int nSeq = nImg + nTxt;
        var q = CatSeq(iq, tq, nImg, nTxt, nh, hd);
        var k = CatSeq(ik, tk, nImg, nTxt, nh, hd);
        var v = CatSeq(iv, tv, nImg, nTxt, nh, hd);

        var attn = SelfAttention(q, k, v, nSeq, nh, hd);
        var imgA = attn.AsSpan(0, nImg * nh * hd).ToArray(); // shape [nImg, nh, hd]
        var txtA = attn.AsSpan(nImg * nh * hd, nTxt * nh * hd).ToArray();

        // Reshape back: [n, nh, hd] → [n, d]
        return (MergeHeads(imgA, nImg, nh, hd), MergeHeads(txtA, nTxt, nh, hd));
    }

    private static float[] SelfAttention(float[] q, float[] k, float[] v, int n, int nh, int hd)
    {
        float scale      = 1f / MathF.Sqrt(hd);
        var   attnOut    = new float[n * nh * hd];
        int   scoreCount = nh * n * n;
        var   scoresBuf  = ArrayPool<float>.Shared.Rent(scoreCount);

        try
        {
            Parallel.For(0, nh, h =>
            {
                int sBase = h * n * n;

                // SIMD QK dot products per query row
                for (int i = 0; i < n; i++)
                {
                    var qi   = q.AsSpan((i * nh + h) * hd, hd);
                    int sRow = sBase + i * n;
                    for (int j = 0; j < n; j++)
                        scoresBuf[sRow + j] = TensorPrimitives.Dot<float>(qi, k.AsSpan((j * nh + h) * hd, hd)) * scale;
                    DiffusionOps.Softmax(scoresBuf, sRow, n);
                }

                // Contiguous per-head V buffer for cache-friendly weighted sum
                var vhBuf = ArrayPool<float>.Shared.Rent(n * hd);
                try
                {
                    for (int j = 0; j < n; j++)
                        v.AsSpan((j * nh + h) * hd, hd).CopyTo(vhBuf.AsSpan(j * hd));

                    for (int i = 0; i < n; i++)
                    {
                        int sRow  = sBase + i * n;
                        var outSl = attnOut.AsSpan((i * nh + h) * hd, hd);
                        outSl.Clear();
                        for (int j = 0; j < n; j++)
                            TensorPrimitives.MultiplyAdd<float>(
                                vhBuf.AsSpan(j * hd, hd), scoresBuf[sRow + j], outSl, outSl);
                    }
                }
                finally { ArrayPool<float>.Shared.Return(vhBuf); }
            });
        }
        finally { ArrayPool<float>.Shared.Return(scoresBuf); }

        return attnOut;
    }

    // ── MLP helpers ───────────────────────────────────────────────────────

    private float[] GeluMlp(string prefix, float[] x, int n, int d)
    {
        // Two-layer MLP: fc1(GELU) → fc2. Expansion factor 4.
        int hidden = d * 4;
        var h = MatQ(x, n, d, $"{prefix}.0.weight", hidden, W($"{prefix}.0.bias"));
        DiffusionOps.GeluInPlace(h);
        return MatQ(h, n, hidden, $"{prefix}.2.weight", d, W($"{prefix}.2.bias"));
    }

    private static float[] Geglu(float[] x, int n, int totalDim)
    {
        // GEGLU: x has shape [n, totalDim]. Split into two halves: gate, val.
        // output = gate * gelu(val)
        int half = totalDim / 2;
        var out_ = new float[n * half];
        for (int i = 0; i < n; i++)
        {
            int off = i * totalDim;
            int o   = i * half;
            for (int j = 0; j < half; j++)
                out_[o + j] = DiffusionOps.Gelu(x[off + j]) * x[off + half + j];
        }
        return out_;
    }

    private float[] MlpProj(string prefix, float[] x, int inDim, int outDim)
    {
        var h = DiffusionOps.Linear(x, W($"{prefix}.in_layer.weight"), W($"{prefix}.in_layer.bias"), 1, inDim, outDim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, W($"{prefix}.out_layer.weight"), W($"{prefix}.out_layer.bias"), 1, outDim, outDim);
    }

    // ── adaLN helpers ─────────────────────────────────────────────────────

    private float[] AdaLNMod(string linPath, float[] vec, int d, int nOut)
    {
        var mod = DiffusionOps.Linear(vec, W($"{linPath}.weight"), W($"{linPath}.bias"), 1, d, nOut * d);
        DiffusionOps.SiluInPlace(mod);
        return mod;
    }

    private static float[] RmsNormMod(float[] x, float[] mod, int n, int d,
                                       int shift, int scale)
    {
        var normed = x.ToArray();
        var shiftV = mod.AsSpan(shift * d, d);
        var scaleV = mod.AsSpan(scale * d, d);

        // Apply adaLN: (rms_norm(x) * (1 + scale) + shift)
        // Here we use a simple per-channel RMSNorm with no learned weight (weight = 1)
        var ones = new float[d];
        Array.Fill(ones, 1f);
        DiffusionOps.RmsNorm(normed, ones, d);
        DiffusionOps.ScaleShiftInPlace(normed, scaleV, shiftV, d);
        return normed;
    }

    private static void ScaleGateAdd(float[] x, float[] update, float[] mod,
                                      int n, int d, int gateIdx)
    {
        var gate = mod.AsSpan(gateIdx * d, d);
        for (int i = 0; i < n; i++)
        {
            int off = i * d;
            for (int j = 0; j < d; j++)
                x[off + j] += gate[j] * update[off + j];
        }
    }

    // ── Shape utilities ───────────────────────────────────────────────────

    private static float[] Slice(float[] x, int n, int colStart, int colLen, int rowStride)
    {
        var out_ = new float[n * colLen];
        for (int i = 0; i < n; i++)
            x.AsSpan(i * rowStride + colStart, colLen).CopyTo(out_.AsSpan(i * colLen));
        return out_;
    }

    // Reshape [n, d] → [n, nh, hd], then transpose to [n, nh, hd] (same here, nh-first)
    private static float[] Reshape2Heads(float[] x, int n, int nh, int hd) => x; // already [n, nh*hd]

    private static float[] MergeHeads(float[] x, int n, int nh, int hd)
    {
        // [n, nh, hd] → [n, d=nh*hd], already contiguous in our layout
        return x;
    }

    private static float[] CatSeq(float[] a, float[] b, int na, int nb, int nh, int hd)
    {
        int d = nh * hd;
        var out_ = new float[(na + nb) * d];
        a.AsSpan(0, na * d).CopyTo(out_);
        b.AsSpan(0, nb * d).CopyTo(out_.AsSpan(na * d));
        return out_;
    }

    private float[] LinearNoBias(string path, float[] x, int n, int inDim, int outDim)
        => MatQ(x, n, inDim, $"{path}.weight", outDim, null);

    private float[] LinearBias(string path, float[] x, int n, int inDim, int outDim)
        => MatQ(x, n, inDim, $"{path}.weight", outDim, OptW($"{path}.bias"));

    // ── Timestep sinusoidal embedding ─────────────────────────────────────

    private static float[] TimestepEmbedding(float t, int dim)
    {
        // Standard sinusoidal embedding at fractional timestep t ∈ [0,1]
        var emb = new float[dim];
        int halfDim = dim / 2;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < halfDim; i++)
        {
            float freq = MathF.Exp(-logMax * i / (halfDim - 1));
            float v = t * 1000f * freq;   // scale t to [0, 1000]
            emb[i]           = MathF.Cos(v);
            emb[i + halfDim] = MathF.Sin(v);
        }
        return emb;
    }

    // ── Weight access ─────────────────────────────────────────────────────

    private float[] W(string name)
    {
        if (_weightCache.TryGetValue(name, out var cached)) return cached;
        var info = _model.FindTensor(name) ?? throw new KeyNotFoundException($"DiT weight not found: {name}");
        var data = DequantGguf(info);
        _weightCache[name] = data;
        return data;
    }

    private float[]? OptW(string name)
    {
        if (_weightCache.TryGetValue(name, out var cached)) return cached;
        var info = _model.FindTensor(name);
        if (info is null) return null;
        var data = DequantGguf(info.Value);
        _weightCache[name] = data;
        return data;
    }

    private float[] DequantGguf(GgufTensorInfo info)
    {
        var raw = _model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(raw, dst, info.DType, info.ElementCount);
        return dst;
    }

    // ── GPU-accelerated matmul with weight caching ────────────────────────

    /// <summary>
    /// Multiply <paramref name="x"/> [n × inDim] by weight tensor <paramref name="wName"/>
    /// [outDim × inDim], optionally adding bias, and return the result [n × outDim].
    ///
    /// On a CUDA backend: dequantizes the weight (fp8 / bf16 / fp16 / fp32 depending on
    /// device capability), uploads it once and caches on GPU, then dispatches cuBLAS SGEMM.
    /// Falls back to CPU (SimdKernels.MatMulBatched) when no GPU backend or n &lt; MinGpuBatch.
    /// </summary>
    private unsafe float[] MatQ(float[] x, int n, int inDim, string wName, int outDim,
                                  float[]? bias)
    {
        var result = new float[n * outDim];

        var info = _model.FindTensor(wName);
        if (info.HasValue)
        {
            var ti       = info.Value;
            int rows     = (int)ti.Dimensions[1];  // outDim — output features (ne1)
            int cols     = (int)ti.Dimensions[0];  // inDim  — input features  (ne0)
            var rawBytes = _model.GetTensorData(ti);

            if (_backend is not CpuBackend && n >= MinGpuBatch)
            {
                if (_backend.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    byte[]   xFp8     = ArrayPool<byte>.Shared.Rent(xCount);
                    ushort[] cBf16Buf = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        for (int i = 0; i < xCount; i++)
                            xFp8[i] = Fp8Converter.FloatToFp8E4M3(x[i]);

                        CoreTensor wGpu;
                        bool ownW;
                        if (_gpuWeightsFp8 != null && _gpuWeightsFp8.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW; ownW = false;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            byte[]  wFp8   = ArrayPool<byte>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
                                for (int i = 0; i < wCount; i++)
                                    wFp8[i] = Fp8Converter.FloatToFp8E4M3(wBuf32[i]);
                                wGpu = _backend.UploadFp8(wFp8.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<byte>.Shared.Return(wFp8);
                            }
                            if (_gpuWeightsFp8 != null)
                                _gpuWeightsFp8[wName] = wGpu;
                            ownW = _gpuWeightsFp8 == null;
                        }

                        var xGpu = _backend.UploadFp8(xFp8.AsSpan(0, xCount), TensorShape.D1(xCount));
                        // fp8 GEMM output must be bf16 (cuBLAS restriction); convert to fp32 on download
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.DownloadBf16(cGpu, cBf16Buf.AsSpan(0, n * rows));
                            int cCount = n * rows;
                            for (int i = 0; i < cCount; i++)
                            {
                                uint bits = (uint)cBf16Buf[i] << 16;
                                result[i] = BitConverter.UInt32BitsToSingle(bits);
                            }
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
                        ArrayPool<ushort>.Shared.Return(cBf16Buf);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Bf16)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    ushort[] xBf16 = ArrayPool<ushort>.Shared.Rent(xCount);
                    ushort[] cBf16 = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        for (int i = 0; i < xCount; i++)
                        {
                            uint bits = BitConverter.SingleToUInt32Bits(x[i]);
                            xBf16[i] = (ushort)(bits >> 16);
                        }

                        CoreTensor wGpu;
                        bool ownW;
                        if (_gpuWeightsBf16 != null && _gpuWeightsBf16.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW; ownW = false;
                        }
                        else
                        {
                            float[]  wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            ushort[] wBf16  = ArrayPool<ushort>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
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
                            if (_gpuWeightsBf16 != null)
                                _gpuWeightsBf16[wName] = wGpu;
                            ownW = _gpuWeightsBf16 == null;
                        }

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

                        for (int i = 0, cnt = n * rows; i < cnt; i++)
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
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                    Half[]  wHalf  = ArrayPool<Half>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
                        for (int i = 0; i < wCount; i++) wHalf[i] = (Half)wBuf32[i];

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
                    // fp32 GPU path
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    float[] wBuf = ArrayPool<float>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(rawBytes, wBuf.AsSpan(0, wCount), ti.DType, wCount);
                        var xGpu = _backend.Upload(x.AsSpan(0, xCount), TensorShape.D1(xCount));
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
                // CPU path: zero-copy via unsafe pointer into mmap'd buffer
                fixed (byte* rawPtr = rawBytes)
                fixed (float* xPtr = x, rPtr = result)
                    SimdKernels.MatMulBatched(rPtr, rawPtr, xPtr, n, rows, cols, ti.DType);
            }
        }
        else
        {
            // Fallback: dequantize + naive multiply (should not be reached in normal operation)
            var w = W(wName);
            DiffusionOps.Linear(x, w, null, n, inDim, outDim).AsSpan().CopyTo(result);
        }

        if (bias is not null)
        {
            for (int b = 0; b < n; b++)
                TensorPrimitives.Add(result.AsSpan(b * outDim, outDim),
                                     bias.AsSpan(), result.AsSpan(b * outDim, outDim));
        }

        return result;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weightCache.Clear();
            if (_gpuWeightsBf16 != null)
            {
                foreach (var t in _gpuWeightsBf16.Values) _backend.Free(t);
                _gpuWeightsBf16.Clear();
            }
            if (_gpuWeightsFp8 != null)
            {
                foreach (var t in _gpuWeightsFp8.Values) _backend.Free(t);
                _gpuWeightsFp8.Clear();
            }
        }
    }
}
