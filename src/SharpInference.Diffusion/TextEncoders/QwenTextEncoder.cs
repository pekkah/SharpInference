using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Diffusion.TextEncoders;

/// <summary>
/// Qwen3-4B text encoder for Z-Image.
///
/// Runs the Qwen3-4B LLM in "encoder mode" — a single full-sequence prefill pass
/// with no generation. Returns hidden_states[-2] (output of layer 34, second-to-last)
/// for all non-padding tokens. This is what ZImagePipeline.encode_prompt does:
///   prompt_embeds = text_encoder(...).hidden_states[-2]
///   kept tokens: where attention_mask == True (i.e., all real tokens)
///
/// Architecture: Qwen3 (LLaMA-style with GQA + QK-norm)
///   hidden_size=2560, num_layers=36, n_heads=32, n_kv_heads=8, head_dim=128
///   intermediate_size=9728, rope_theta=1e6, rms_norm_eps=1e-6
///   FFN: gate_proj * silu(up_proj) — standard SiLU-gated (no w3, just up+gate)
///
/// Performance: quantized matmuls use SimdKernels.MatMulBatched with zero-copy mmap
/// pointers (GgufModel.GetTensorDataPtr), so large weight matrices are never
/// dequantized to float32 upfront — on-the-fly dequant happens in SIMD registers.
/// </summary>
public sealed class QwenTextEncoder : IDisposable
{
    private readonly GgufModel _model;
    private readonly ZImageParams _p;
    // Cache only small tensors (norms, q/k-norm weights). Large weight matrices are
    // accessed via MatQ() → GetTensorDataPtr() → SimdKernels.MatMulBatched().
    private readonly Dictionary<string, float[]> _cache = new(StringComparer.Ordinal);

    private readonly float[] _ropeCos;
    private readonly float[] _ropeSin;
    private readonly int _maxSeqLen = 4096;
    private bool _disposed;

    public QwenTextEncoder(GgufModel model, ZImageParams p)
    {
        _model = model;
        _p     = p;
        (_ropeCos, _ropeSin) = BuildRopeTables(_maxSeqLen, p.QwenHeadDim, p.QwenRopeTheta);
    }

    // ── Text encoding ─────────────────────────────────────────────────────

    /// <summary>
    /// Encode token IDs and return hidden states [nTokens, hiddenSize].
    /// Runs layers 0..QwenEncoderLayer (35 layers, hidden_states[-2]).
    /// </summary>
    public float[] Encode(int[] tokenIds, Action<int, int>? encodeProgress = null)
    {
        int seqLen     = tokenIds.Length;
        int hidden     = _p.QwenHiddenSize;      // 2560
        int nHeads     = _p.QwenNumHeads;         // 32
        int nKvHeads   = _p.QwenNumKvHeads;       // 8
        int headDim    = _p.QwenHeadDim;           // 128
        int intermSize = _p.QwenIntermSize;        // 9728
        int nLayers    = _p.QwenEncoderLayer + 1; // 35 layers
        float rmsEps   = _p.QwenRmsNormEps;

        // Token embedding: per-row dequantization from mmap — avoids 1.5 GB float[] cache
        var h      = EmbedTokens(tokenIds, seqLen, hidden);
        var normed = new float[seqLen * hidden];

        for (int l = 0; l < nLayers; l++)
        {
            encodeProgress?.Invoke(l, nLayers);
            string blk = $"blk.{l}";

            // ── Attention ─────────────────────────────────────────────────
            var attnNormW = W($"{blk}.attn_norm.weight");
            for (int t = 0; t < seqLen; t++)
                DiffusionOps.RmsNorm(h, t * hidden, hidden, attnNormW, rmsEps, normed, t * hidden);

            // Q, K, V projections — quantized matmul via zero-copy mmap pointer
            var qFlat = MatQ(normed, seqLen, hidden, $"{blk}.attn_q.weight",      nHeads   * headDim);
            var kFlat = MatQ(normed, seqLen, hidden, $"{blk}.attn_k.weight",      nKvHeads * headDim);
            var vFlat = MatQ(normed, seqLen, hidden, $"{blk}.attn_v.weight",      nKvHeads * headDim);

            // Per-head QK-norm (Qwen3 feature)
            ApplyPerHeadRmsNorm(qFlat, seqLen, nHeads,   headDim, W($"{blk}.attn_q_norm.weight"), rmsEps);
            ApplyPerHeadRmsNorm(kFlat, seqLen, nKvHeads, headDim, W($"{blk}.attn_k_norm.weight"), rmsEps);

            // 1D RoPE
            ApplyRoPE1D(qFlat, seqLen, nHeads,   headDim);
            ApplyRoPE1D(kFlat, seqLen, nKvHeads, headDim);

            // Causal GQA attention
            var attnOut  = CausalGqaAttention(qFlat, kFlat, vFlat, seqLen, nHeads, nKvHeads, headDim);
            var attnProj = MatQ(attnOut, seqLen, nHeads * headDim, $"{blk}.attn_output.weight", hidden);

            // Residual
            TensorPrimitives.Add(h.AsSpan(0, seqLen * hidden),
                                 attnProj.AsSpan(0, seqLen * hidden),
                                 h.AsSpan(0, seqLen * hidden));

            // ── FFN ───────────────────────────────────────────────────────
            var ffnNormW = W($"{blk}.ffn_norm.weight");
            for (int t = 0; t < seqLen; t++)
                DiffusionOps.RmsNorm(h, t * hidden, hidden, ffnNormW, rmsEps, normed, t * hidden);

            var gate = MatQ(normed, seqLen, hidden, $"{blk}.ffn_gate.weight", intermSize);
            var up   = MatQ(normed, seqLen, hidden, $"{blk}.ffn_up.weight",   intermSize);

            // SiLU(gate) * up → in-place using TensorPrimitives
            DiffusionOps.SiluInPlace(gate.AsSpan());
            TensorPrimitives.Multiply(gate.AsSpan(), up.AsSpan(), gate.AsSpan());

            var down = MatQ(gate, seqLen, intermSize, $"{blk}.ffn_down.weight", hidden);

            TensorPrimitives.Add(h.AsSpan(0, seqLen * hidden),
                                 down.AsSpan(0, seqLen * hidden),
                                 h.AsSpan(0, seqLen * hidden));
        }

        return h;
    }

    // ── Token embedding via per-row dequantization ────────────────────────

    /// <summary>
    /// Embed tokens by dequantizing only the required rows from the mmap'd embedding table.
    /// Avoids the 1.5 GB float[] allocation that full vocab dequantization would require.
    /// </summary>
    private unsafe float[] EmbedTokens(int[] tokenIds, int seqLen, int hidden)
    {
        var info = _model.FindTensor("token_embd.weight")
                   ?? throw new KeyNotFoundException("token_embd.weight not found");

        byte* ptr         = _model.GetTensorDataPtr(info);
        long  vocabSize   = info.Dimensions[1];  // ne1 = vocab_size
        long  bytesTotal  = info.ByteSize;
        long  bytesPerRow = bytesTotal / vocabSize;
        DType dtype       = info.DType;

        var h      = new float[seqLen * hidden];
        var rowBuf = ArrayPool<float>.Shared.Rent(hidden);
        try
        {
            for (int i = 0; i < seqLen; i++)
            {
                long rowOff  = (long)tokenIds[i] * bytesPerRow;
                var  rowSpan = new ReadOnlySpan<byte>((void*)(ptr + rowOff), (int)bytesPerRow);
                Dequantize.ToFloat32(rowSpan, rowBuf, dtype, hidden);
                rowBuf.AsSpan(0, hidden).CopyTo(h.AsSpan(i * hidden, hidden));
            }
        }
        finally { ArrayPool<float>.Shared.Return(rowBuf); }

        return h;
    }

    // ── Causal GQA attention ──────────────────────────────────────────────

    private float[] CausalGqaAttention(float[] q, float[] k, float[] v,
                                        int seqLen, int nQ, int nKV, int headDim)
    {
        int groupSize = nQ / nKV;
        float scale   = 1f / MathF.Sqrt(headDim);
        var output    = new float[seqLen * nQ * headDim];

        Parallel.For(0, nQ, h =>
        {
            int kvHead    = h / groupSize;
            var scoresBuf = ArrayPool<float>.Shared.Rent(seqLen);
            try
            {
                for (int i = 0; i < seqLen; i++)
                {
                    var qi = q.AsSpan((i * nQ + h) * headDim, headDim);

                    // Compute causal QK scores via TensorPrimitives.Dot (SIMD)
                    for (int j = 0; j <= i; j++)
                    {
                        var kj       = k.AsSpan((j * nKV + kvHead) * headDim, headDim);
                        scoresBuf[j] = TensorPrimitives.Dot<float>(qi, kj) * scale;
                    }

                    // Softmax over [0..i]
                    var sRow = scoresBuf.AsSpan(0, i + 1);
                    float max = TensorPrimitives.Max(sRow);
                    TensorPrimitives.Subtract(sRow, max,          sRow);
                    TensorPrimitives.Exp(sRow,                    sRow);
                    TensorPrimitives.Divide(sRow, TensorPrimitives.Sum(sRow), sRow);

                    // Weighted sum of V using MultiplyAdd
                    int outBase = (i * nQ + h) * headDim;
                    output.AsSpan(outBase, headDim).Clear();
                    for (int j = 0; j <= i; j++)
                    {
                        TensorPrimitives.MultiplyAdd<float>(
                            v.AsSpan((j * nKV + kvHead) * headDim, headDim),
                            scoresBuf[j],
                            output.AsSpan(outBase, headDim),
                            output.AsSpan(outBase, headDim));
                    }
                }
            }
            finally { ArrayPool<float>.Shared.Return(scoresBuf); }
        });

        return output;
    }

    private static void ApplyPerHeadRmsNorm(float[] x, int seqLen, int nHeads, int headDim,
                                             float[] normW, float eps)
    {
        var wSp = normW.AsSpan(0, headDim);
        for (int t = 0; t < seqLen; t++)
        for (int h = 0; h < nHeads; h++)
        {
            var row  = x.AsSpan((t * nHeads + h) * headDim, headDim);
            float ss = TensorPrimitives.Dot<float>(row, row);
            float r  = 1f / MathF.Sqrt(ss / headDim + eps);
            TensorPrimitives.Multiply(row, wSp, row);
            TensorPrimitives.Multiply(row, r,   row);
        }
    }

    private void ApplyRoPE1D(float[] qk, int seqLen, int nHeads, int headDim)
    {
        int halfDim = headDim / 2;
        for (int t = 0; t < seqLen; t++)
        for (int h = 0; h < nHeads; h++)
        {
            int base_ = (t * nHeads + h) * headDim;
            for (int j = 0; j < halfDim; j++)
            {
                float cos = _ropeCos[t * halfDim + j];
                float sin = _ropeSin[t * halfDim + j];
                float a   = qk[base_ + j];
                float b   = qk[base_ + j + halfDim];
                qk[base_ + j]           = a * cos - b * sin;
                qk[base_ + j + halfDim] = a * sin + b * cos;
            }
        }
    }

    // ── Quantized matmul: x[n,k] @ W[outDim,k]^T → out[n,outDim] ─────────

    /// <summary>
    /// Runs the matrix multiply using SimdKernels.MatMulBatched with a zero-copy
    /// mmap pointer (GgufModel.GetTensorDataPtr). Dequantization happens on-the-fly
    /// inside SIMD kernels — no large float32 allocations.
    /// </summary>
    private unsafe float[] MatQ(float[] x, int n, int k, string wName, int outDim)
    {
        var result = new float[n * outDim];
        var info   = _model.FindTensor(wName)
                     ?? throw new KeyNotFoundException($"Qwen weight not found: {wName}");
        byte* wPtr = _model.GetTensorDataPtr(info);
        int   rows = (int)info.Dimensions[1]; // ne1 = output features
        int   cols = (int)info.Dimensions[0]; // ne0 = input features
        fixed (float* xPtr = x, rPtr = result)
            SimdKernels.MatMulBatched(rPtr, wPtr, xPtr, n, rows, cols, info.DType);
        return result;
    }

    // ── Weight access (cached, for small tensors only) ─────────────────────

    private float[] W(string name)
    {
        if (_cache.TryGetValue(name, out var c)) return c;
        var info = _model.FindTensor(name)
                   ?? throw new KeyNotFoundException($"Qwen weight not found: {name}");
        long count = info.ElementCount;
        var raw    = _model.GetTensorData(info);
        var dst    = new float[count];
        Dequantize.ToFloat32(raw, dst, info.DType, count);
        _cache[name] = dst;
        return dst;
    }

    // ── RoPE precompute ────────────────────────────────────────────────────

    private static (float[] cos, float[] sin) BuildRopeTables(int maxSeqLen, int headDim, float theta)
    {
        int halfDim = headDim / 2;
        var cos = new float[maxSeqLen * halfDim];
        var sin = new float[maxSeqLen * halfDim];
        for (int pos = 0; pos < maxSeqLen; pos++)
        for (int j = 0; j < halfDim; j++)
        {
            float freq  = 1f / MathF.Pow(theta, 2f * j / headDim);
            float angle = pos * freq;
            cos[pos * halfDim + j] = MathF.Cos(angle);
            sin[pos * halfDim + j] = MathF.Sin(angle);
        }
        return (cos, sin);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _cache.Clear();
        }
    }
}
