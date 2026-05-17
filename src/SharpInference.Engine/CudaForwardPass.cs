using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Engine;

/// <summary>
/// GPU-resident forward pass for dense LLaMA-family transformers driven by the
/// CUDA backend (cuBLAS + NVRTC compute kernels).
///
/// All weights live in VRAM (Q4_K / Q6_K raw bytes for projection matrices,
/// FP32 for norm/bias weights). One-token autoregressive decode runs the full
/// sequence on the GPU: embedding lookup, per-layer attention with paged-stride
/// KV cache, SwiGLU FFN, output projection, and a single logits download.
///
/// Limitations (intentional, scoped for Qwen3-8B-Q4_K_M first):
///   • Dense FFN only (no MoE — call site must pass a non-MoE model).
///   • No TurboQuant compressed KV cache.
///   • No NoPE layer skipping (NoRopeLayerStep is honored if set, but the
///     primary target Qwen3 uses RoPE on every layer).
///   • Embedding table accepted as Q4_K or F32; quantized variants are
///     dequantized to F32 on CPU when uploading (small one-time cost).
/// </summary>
public sealed unsafe class CudaForwardPass : IForwardPass
{
    private readonly CudaBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;

    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _intermDim;
    private readonly int _maxSeqLen;
    private int _kvLength;

    private readonly float[] _logitsBuf;

    // Scratch buffers in VRAM
    private readonly Tensor _hidden;
    private readonly Tensor _residual;
    private readonly Tensor _normBuf;
    private readonly Tensor _q;
    private readonly Tensor _k;
    private readonly Tensor _v;
    private readonly Tensor _attnOut;
    private readonly Tensor _ffnGate;
    private readonly Tensor _ffnUp;
    private readonly Tensor _logits;

    // Embedding table (Q4_K raw bytes or F32 row-major)
    private readonly Tensor _gpuEmbedding;
    private readonly bool _embIsQuantized;

    // Per-layer weights (VRAM)
    private readonly Tensor[] _wAttnNorm;
    private readonly Tensor[] _wq, _wk, _wv, _wo;
    private readonly Tensor[] _wFfnNorm;
    private readonly Tensor[] _wGate, _wUp, _wDown;
    private readonly Tensor _wOutputNorm;
    private readonly Tensor _wOutput;

    // Optional attention biases
    private readonly bool _hasAttnBias;
    private readonly Tensor[]? _bq, _bk, _bv, _bo;

    // Optional per-head QK norm (Qwen3)
    private readonly bool _hasQkNorm;
    private readonly Tensor[]? _wqNorm, _wkNorm;

    // Per-layer KV cache in VRAM (FP32, [maxSeqLen, numKvHeads*headDim])
    private readonly Tensor[] _gpuKCache;
    private readonly Tensor[] _gpuVCache;

    // CPU prefix cache kept only to satisfy IForwardPass.TruncateTo + InferenceEngine
    // prefix-reuse bookkeeping. CUDA cache state advances in lockstep via _kvLength.
    private readonly KvCache _kvCache;

    // Dtype dispatch for MatMul (mirrors GpuForwardPass._weightDTypes).
    private readonly Dictionary<nint, DType> _weightDTypes = new();

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeqLen;

    public CudaForwardPass(GgufModel model, CudaBackend gpu, ModelHyperparams hp, int maxContextLength = 0)
    {
        if (hp.IsMoE)
            throw new NotSupportedException("CudaForwardPass currently supports only dense (non-MoE) models. Use the Vulkan backend or CPU path for MoE.");

        _model = model;
        _gpu = gpu;
        _hp = hp;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        _maxSeqLen = maxContextLength > 0
            ? Math.Min(maxContextLength, hp.ContextLength)
            : EstimateMaxContext(model, gpu, hp);

        _kvCache = new KvCache(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.HeadDim);

        Console.Error.WriteLine($"[CudaForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength})");

        // Scratch
        _hidden    = gpu.Allocate(TensorShape.D1(_embDim));
        _residual  = gpu.Allocate(TensorShape.D1(_embDim));
        _normBuf   = gpu.Allocate(TensorShape.D1(_embDim));
        _q         = gpu.Allocate(TensorShape.D1((long)_numHeads * _headDim));
        _k         = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _headDim));
        _v         = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _headDim));
        _attnOut   = gpu.Allocate(TensorShape.D1((long)_numHeads * _headDim));
        _ffnGate   = gpu.Allocate(TensorShape.D1(_intermDim));
        _ffnUp     = gpu.Allocate(TensorShape.D1(_intermDim));
        _logits    = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _logitsBuf = new float[hp.VocabSize];

        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[hp.NumLayers];
        _gpuVCache = new Tensor[hp.NumLayers];
        for (int i = 0; i < hp.NumLayers; i++)
        {
            _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
        }

        // Upload weights to VRAM
        int L = hp.NumLayers;
        _wAttnNorm = new Tensor[L]; _wFfnNorm = new Tensor[L];
        _wq = new Tensor[L]; _wk = new Tensor[L]; _wv = new Tensor[L]; _wo = new Tensor[L];
        _wGate = new Tensor[L]; _wUp = new Tensor[L]; _wDown = new Tensor[L];

        _hasAttnBias = hp.HasAttnBias;
        if (_hasAttnBias)
        {
            _bq = new Tensor[L]; _bk = new Tensor[L];
            _bv = new Tensor[L]; _bo = new Tensor[L];
        }

        _hasQkNorm = hp.HasQkNorm;
        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            _wqNorm = new Tensor[L]; _wkNorm = new Tensor[L];
        }

        Console.Error.Write($"[CudaForwardPass] Uploading {L} layers to VRAM...");
        for (int i = 0; i < L; i++)
        {
            _wAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _wq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _wk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            _wv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _wo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _wFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            _wGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
            _wUp[i]   = UploadWeight($"blk.{i}.ffn_up.weight");
            _wDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");

            if (_hasAttnBias)
            {
                _bq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _bk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                _bv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                _bo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _wqNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _wkNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            Console.Error.Write(".");
        }

        // Embedding table
        Console.Error.Write(" emb...");
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        if (embInfo.DType == DType.Q4_K)
        {
            var embData = model.GetTensorData(embInfo);
            _gpuEmbedding = _gpu.UploadRaw(embData, TensorShape.D1(embData.Length), DType.Q4_K);
            _embIsQuantized = true;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Q4_K;
        }
        else
        {
            var embData = model.GetTensorData(embInfo);
            var embF32 = new float[(int)embInfo.ElementCount];
            Dequantize.ToFloat32(embData, embF32, embInfo.DType, embInfo.ElementCount);
            _gpuEmbedding = _gpu.Upload(embF32, TensorShape.D1(embF32.Length));
            _embIsQuantized = false;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Float32;
        }

        _wOutputNorm = UploadWeight("output_norm.weight");
        _wOutput = model.FindTensor("output.weight") is not null
            ? UploadWeight("output.weight")
            : _gpuEmbedding;

        Console.Error.WriteLine(" done.");

        // Warm up: synchronize so kernel compilation/caching latency isn't reported
        // as the first token's decode time.
        _gpu.Synchronize();

        if (Environment.GetEnvironmentVariable("SHARPI_CUDA_MATVEC_BENCH") == "1")
            BenchMatVec();
    }

    /// <summary>
    /// Microbench: time the Q4_K matvec kernel in isolation at the three FFN shapes
    /// (gate/up = rows×cols=12288×4096, down = 4096×12288, output = vocab×emb).
    /// Reports effective HBM bandwidth so we can tell whether the kernel is
    /// bandwidth-bound (good — anything > ~250 GB/s on RTX 4070 Ti is healthy)
    /// or compute/scheduling-bound (bad — much lower than that).
    /// </summary>
    private void BenchMatVec()
    {
        // Pure HBM bandwidth baseline — if this can't hit ≥ 200 GB/s, the GPU
        // or driver is the bottleneck, not the matvec kernel.
        {
            const int MB = 28 * 1024 * 1024;        // matches the bytes touched by gate matmul
            var src = _gpu.Allocate(TensorShape.D1(MB / 4));
            var dst = _gpu.Allocate(TensorShape.D1(MB / 4));
            nint srcPtr = _gpu.GetTensorDevicePtr(src);
            nint dstPtr = _gpu.GetTensorDevicePtr(dst);
            for (int i = 0; i < 16; i++) _gpu.RunBandwidthBaseline(srcPtr, dstPtr, MB);
            _gpu.Synchronize();
            const int BwIter = 500;
            var swb = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < BwIter; i++) _gpu.RunBandwidthBaseline(srcPtr, dstPtr, MB);
            _gpu.Synchronize();
            swb.Stop();
            double ms = swb.Elapsed.TotalMilliseconds / BwIter;
            double gbps = (double)MB / (ms / 1000.0) / 1e9;            // read only
            double gbpsRW = 2.0 * (double)MB / (ms / 1000.0) / 1e9;     // read+write
            Console.Error.WriteLine(
                $"[CudaForwardPass] HBM baseline (memcpy 28 MB × {BwIter}): {ms*1000:F1} µs/call → " +
                $"{gbps:F1} GB/s read, {gbpsRW:F1} GB/s read+write");
            _gpu.Free(src); _gpu.Free(dst);
        }

        Console.Error.WriteLine("[CudaForwardPass] matvec_q4k microbench (3000 iter/shape):");
        var shapes = new (int rows, int cols, string label)[]
        {
            (4096,   4096,  "qkv-Q     (4096×4096)"),
            (12288,  4096,  "ffn-gate  (12288×4096)"),
            (4096,   12288, "ffn-down  (4096×12288)"),
            (151936, 4096,  "lm-head   (151936×4096)"),
        };
        const int Iter = 3000;

        foreach (var (rows, cols, label) in shapes)
        {
            // Borrow real weights (always present in the upload set): use the first layer's FFN.
            Tensor weight = rows == 12288 ? _wGate[0]
                          : rows == 4096 && cols == 12288 ? _wDown[0]
                          : rows == 4096 && cols == 4096 ? _wq[0]
                          : _wOutput;
            Tensor input  = cols == 4096 ? _normBuf : _ffnGate;
            Tensor output = rows == 4096 ? _hidden
                          : rows == 12288 ? _ffnGate
                          : _logits;

            // Warm-up.
            for (int i = 0; i < 32; i++)
                _gpu.MatMul(output, weight, input, DType.Q4_K);
            _gpu.Synchronize();

            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < Iter; i++)
                _gpu.MatMul(output, weight, input, DType.Q4_K);
            _gpu.Synchronize();
            sw.Stop();

            double msPerCall = sw.Elapsed.TotalMilliseconds / Iter;
            double weightBytes = (long)rows * cols * 0.5625; // Q4_K = 4.5 bits/elem
            double gbPerSec = weightBytes / (msPerCall / 1000.0) / 1e9;
            Console.Error.WriteLine(
                $"  {label,-26} {msPerCall * 1000,7:F1} µs/call  →  {gbPerSec,6:F1} GB/s");
        }
    }

    // Profiling state (only used when SHARPI_CUDA_PROFILE is set).
    private static readonly bool s_profile =
        Environment.GetEnvironmentVariable("SHARPI_CUDA_PROFILE") == "1";
    private readonly double[] _phaseMs = new double[10];
    private readonly long[]   _phaseCount = new long[10];
    private const int PH_EMBED = 0, PH_QKV = 1, PH_ROPE_QKN = 2, PH_KV_ATTN = 3,
                      PH_O_RES = 4, PH_FFN = 5, PH_FINAL = 6;
    private static readonly string[] s_phaseName =
        ["embed", "qkv-matmul", "rope+qknorm", "kv+attn", "o-proj+res", "ffn", "final+download"];

    /// <inheritdoc/>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if (s_profile) return ForwardProfiled(token, position);

        // Embed token
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // residual = hidden
            CopyDevice(_residual, _hidden);

            // normBuf = rmsnorm(hidden, w_attn_norm)
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);

            // Q/K/V projections from normBuf
            GpuMatMul(_q, _wq[layer], _normBuf);
            GpuMatMul(_k, _wk[layer], _normBuf);
            GpuMatMul(_v, _wv[layer], _normBuf);

            if (_hasAttnBias)
            {
                _gpu.AddInPlace(_q, _bq![layer]);
                _gpu.AddInPlace(_k, _bk![layer]);
                _gpu.AddInPlace(_v, _bv![layer]);
            }

            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;
            if (useRoPE)
            {
                _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
            }

            if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
            {
                if (_hp.UseL2QkNorm)
                {
                    _gpu.HeadNormPure(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    _gpu.HeadNormPure(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
                else
                {
                    _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps);
                    _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }

            int kvDim = _numKvHeads * _headDim;
            _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, position, _maxSeqLen);

            _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                _numHeads, _numKvHeads, _headDim, position + 1, _maxSeqLen);

            GpuMatMul(_hidden, _wo[layer], _attnOut);
            if (_hasAttnBias)
                _gpu.AddInPlace(_hidden, _bo![layer]);

            _gpu.AddInPlace(_hidden, _residual);

            // FFN
            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);

            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp,   _wUp[layer],   _normBuf);
            _gpu.SiLuMul(_ffnGate, _ffnUp);
            GpuMatMul(_hidden, _wDown[layer], _ffnGate);

            _gpu.AddInPlace(_hidden, _residual);
        }

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        GpuMatMul(_logits, _wOutput, _hidden);

        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    private ReadOnlySpan<float> ForwardProfiled(int token, int position)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        long t0 = sw.ElapsedTicks;

        if (_embIsQuantized) _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        else                 _gpu.EmbedLookup   (_gpuEmbedding, _hidden, token, _embDim);
        _gpu.Synchronize();
        AccPhase(PH_EMBED, sw, ref t0);

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
            GpuMatMul(_q, _wq[layer], _normBuf);
            GpuMatMul(_k, _wk[layer], _normBuf);
            GpuMatMul(_v, _wv[layer], _normBuf);
            if (_hasAttnBias)
            {
                _gpu.AddInPlace(_q, _bq![layer]);
                _gpu.AddInPlace(_k, _bk![layer]);
                _gpu.AddInPlace(_v, _bv![layer]);
            }
            _gpu.Synchronize();
            AccPhase(PH_QKV, sw, ref t0);

            bool useRoPE = _hp.NoRopeLayerStep == 0 || (layer + 1) % _hp.NoRopeLayerStep != 0;
            if (useRoPE)
            {
                _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
            }
            if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
            {
                if (_hp.UseL2QkNorm)
                {
                    _gpu.HeadNormPure(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    _gpu.HeadNormPure(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
                else
                {
                    _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps);
                    _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }
            _gpu.Synchronize();
            AccPhase(PH_ROPE_QKN, sw, ref t0);

            int kvDim = _numKvHeads * _headDim;
            _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, position, _maxSeqLen);
            _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                _numHeads, _numKvHeads, _headDim, position + 1, _maxSeqLen);
            _gpu.Synchronize();
            AccPhase(PH_KV_ATTN, sw, ref t0);

            GpuMatMul(_hidden, _wo[layer], _attnOut);
            if (_hasAttnBias) _gpu.AddInPlace(_hidden, _bo![layer]);
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.Synchronize();
            AccPhase(PH_O_RES, sw, ref t0);

            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp,   _wUp[layer],   _normBuf);
            _gpu.SiLuMul(_ffnGate, _ffnUp);
            GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.Synchronize();
            AccPhase(PH_FFN, sw, ref t0);
        }

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        GpuMatMul(_logits, _wOutput, _hidden);
        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();
        AccPhase(PH_FINAL, sw, ref t0);

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    private void AccPhase(int idx, System.Diagnostics.Stopwatch sw, ref long t0)
    {
        long t1 = sw.ElapsedTicks;
        _phaseMs[idx] += (t1 - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _phaseCount[idx]++;
        t0 = t1;
    }

    /// <summary>Write accumulated per-phase timings to stderr (no-op when profiling disabled).</summary>
    public void DumpProfile()
    {
        if (!s_profile) return;
        Console.Error.WriteLine("[CudaForwardPass] Per-phase totals (ms):");
        double total = 0;
        for (int i = 0; i < s_phaseName.Length; i++) total += _phaseMs[i];
        for (int i = 0; i < s_phaseName.Length; i++)
        {
            if (_phaseCount[i] == 0) continue;
            double share = total > 0 ? 100.0 * _phaseMs[i] / total : 0;
            Console.Error.WriteLine(
                $"  {s_phaseName[i],-16} {_phaseMs[i],10:F2} ms  ({_phaseCount[i]} calls, " +
                $"{_phaseMs[i] / _phaseCount[i] * 1000:F1} µs/call, {share:F1}%)");
        }
        Console.Error.WriteLine($"  {"TOTAL",-16} {total,10:F2} ms");
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    /// <inheritdoc/>
    public void TruncateTo(int length)
    {
        _kvLength = length;
        _kvCache.TruncateTo(length);
    }

    /// <inheritdoc/>
    public void ResetCache()
    {
        _kvLength = 0;
        _kvCache.Reset();
    }

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        var dtype = _weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K);
        _gpu.MatMul(output, weights, input, dtype);
    }

    private void CopyDevice(Tensor dst, Tensor src) => _gpu.CopyDevice(dst, src);

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType);
            _weightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Less-common dtypes: dequantize on CPU and upload as F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    /// <summary>
    /// VRAM-based context-length estimator: subtract uploaded-weight bytes and a fixed
    /// scratch budget from total VRAM, then divide what's left between K and V caches
    /// (each FP32, [maxSeqLen, kvDim] per layer).
    /// </summary>
    public static int EstimateMaxContext(GgufModel model, CudaBackend gpu, ModelHyperparams hp)
    {
        long vramBytes = (long)gpu.VramBytes;
        if (vramBytes <= 0) vramBytes = 8L * 1024 * 1024 * 1024; // fallback assumption: 8 GB

        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += EstimateGpuTensorBytes(t);

        int headDim = hp.HeadDim;
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        // Reserve at least 2 GiB (or a third of total) for the driver, the cuBLAS
        // workspace, the Q8_1 quantization scratch, the pinned host buffer, the GPU
        // buffer pool's per-bucket reuse list, and CUDA's framebuffer-and-context
        // overhead. The previous max(vram/5, 1 GiB) left only ~24 MiB free on a
        // 12 GiB card running Qwen3-8B; the driver then mapped late weight
        // allocations (notably the 600 MiB lm-head) into system memory, where the
        // matvec ran at ~22 GB/s over PCIe instead of ~400 GB/s in HBM and prefill
        // collapsed from ~65 t/s to ~4 t/s.
        long reserved = Math.Max(vramBytes / 3, 2L * 1024 * 1024 * 1024);
        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024;

        long bytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float);
        int maxCtx = (int)(available / bytesPerToken);
        return Math.Clamp(maxCtx, 512, hp.ContextLength);
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K || tensor.DType == DType.Q6_K)
            return (tensor.ByteSize + 3) & ~3L;
        return tensor.ElementCount * sizeof(float);
    }

    public void Dispose()
    {
        DumpProfile();
        _gpu.Free(_hidden); _gpu.Free(_residual); _gpu.Free(_normBuf);
        _gpu.Free(_q); _gpu.Free(_k); _gpu.Free(_v); _gpu.Free(_attnOut);
        _gpu.Free(_ffnGate); _gpu.Free(_ffnUp); _gpu.Free(_logits);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_wAttnNorm[i]); _gpu.Free(_wFfnNorm[i]);
            _gpu.Free(_wq[i]); _gpu.Free(_wk[i]); _gpu.Free(_wv[i]); _gpu.Free(_wo[i]);
            _gpu.Free(_wGate[i]); _gpu.Free(_wUp[i]); _gpu.Free(_wDown[i]);

            if (_hasAttnBias)
            {
                _gpu.Free(_bq![i]); _gpu.Free(_bk![i]);
                _gpu.Free(_bv![i]); _gpu.Free(_bo![i]);
            }

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.Free(_wqNorm![i]); _gpu.Free(_wkNorm![i]);
            }
        }
        _gpu.Free(_wOutputNorm);
        if (_wOutput.Handle != _gpuEmbedding.Handle)
            _gpu.Free(_wOutput);
        _gpu.Free(_gpuEmbedding);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_gpuKCache[i]);
            _gpu.Free(_gpuVCache[i]);
        }

        _kvCache.Dispose();
    }
}
