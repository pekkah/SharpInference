using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.TurboQuant;
using SharpInference.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid GPU/CPU forward pass for models larger than VRAM.
/// First N layers run on GPU (Vulkan compute shaders), remaining layers on CPU (AVX2 SIMD).
/// Hidden state transfers via pinned host memory at GPU↔CPU boundaries.
/// </summary>
public sealed unsafe class HybridForwardPass : IDisposable
{
    private readonly GgufModel _model;
    private readonly VulkanBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly LayerPlacement _placement;

    // Dimensions
    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim;
    private readonly int _nGpuLayers, _nCpuLayers;

    // ── GPU resources (layers 0..nGpuLayers-1) ──
    private readonly Tensor _gpuHidden, _gpuResidual, _gpuNormBuf;
    private readonly Tensor _gpuQ, _gpuK, _gpuV, _gpuAttnOut;
    private readonly Tensor _gpuFfnGate, _gpuFfnUp;
    private readonly Tensor _gpuLogits;
    private readonly Tensor _gpuEmbedding, _gpuOutputWeight, _gpuOutputNorm;
    private readonly bool _embIsQuantized;
    private readonly Tensor[] _gpuAttnNorm, _gpuWq, _gpuWk, _gpuWv, _gpuWo;
    private readonly Tensor[] _gpuFfnNorm, _gpuWGate, _gpuWUp, _gpuWDown;
    private readonly Tensor[]? _gpuBq, _gpuBk, _gpuBv, _gpuBo;
    private readonly Tensor[]? _gpuQNorm, _gpuKNorm;
    private readonly Tensor[] _gpuKCache, _gpuVCache;
    private readonly Dictionary<nint, DType> _gpuWeightDTypes = new();

    // ── CPU resources (layers nGpuLayers..numLayers-1) ──
    private readonly float* _cpuHidden, _cpuResidual, _cpuNormBuf;
    private readonly float* _cpuQ, _cpuK, _cpuV, _cpuAttnOut;
    private readonly float* _cpuFfnGate, _cpuFfnUp;
    private readonly float* _cpuAttnScores;
    private readonly CpuWeightRef[] _cpuAttnNorm, _cpuWq, _cpuWk, _cpuWv, _cpuWo;
    private readonly CpuWeightRef[] _cpuFfnNorm, _cpuWGate, _cpuWUp, _cpuWDown;
    private readonly float*[] _cpuBq, _cpuBk, _cpuBv, _cpuBo;
    private readonly float*[] _cpuQNorm, _cpuKNorm;
    private readonly KvCache _cpuKvCache;
    private readonly TurboQuantKvCache? _cpuTqKvCache;
    private readonly float* _cpuRotatedQuery; // scratch for TQ query rotation [headDim]
    private readonly float* _cpuDecompBuf;    // scratch for TQ value decompress [headDim]
    private readonly Dictionary<string, nint> _cpuNormCache = new();

    // ── Shared ──
    private readonly Tensor _pinnedHidden; // host-visible buffer for GPU↔CPU transfer
    private readonly float[] _logitsBuf;
    private readonly bool _hasAttnBias, _hasQkNorm;
    private readonly bool _tqEnabled;
    private int _kvLength;
    private readonly int _maxSeqLen;

    public int MaxSeqLen => _maxSeqLen;
    public LayerPlacement Placement => _placement;

    public HybridForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        LayerPlacement placement, bool enableTq = false)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _placement = placement;
        _nGpuLayers = placement.GpuLayers;
        _nCpuLayers = placement.CpuLayers;
        _maxSeqLen = placement.RecommendedCtxSize;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.EmbeddingDim / hp.NumHeads;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;
        _hasAttnBias = hp.HasAttnBias;
        _hasQkNorm = hp.HasQkNorm;
        _tqEnabled = enableTq;

        Console.Error.WriteLine($"[HybridForwardPass] {placement.Summary()}{(enableTq ? " [TQ3]" : "")}");

        // ── Allocate GPU scratch buffers ──
        _gpuHidden = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuResidual = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuNormBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _gpuK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _gpuV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _gpuAttnOut = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _gpuFfnGate = gpu.Allocate(TensorShape.D1(_intermDim));
        _gpuFfnUp = gpu.Allocate(TensorShape.D1(_intermDim));
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _logitsBuf = new float[hp.VocabSize];

        // Pinned buffer for hidden state transfer (embDim floats)
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));

        // ── Upload GPU weights (embedding + output + first N layers) ──
        _gpuEmbedding = UploadWeight("token_embd.weight");
        _embIsQuantized = model.FindTensor("token_embd.weight")!.Value.DType != DType.Float32;

        _gpuOutputNorm = UploadWeight("output_norm.weight");
        _gpuOutputWeight = model.FindTensor("output.weight") is not null
            ? UploadWeight("output.weight")
            : _gpuEmbedding;

        _gpuAttnNorm = new Tensor[_nGpuLayers];
        _gpuWq = new Tensor[_nGpuLayers]; _gpuWk = new Tensor[_nGpuLayers];
        _gpuWv = new Tensor[_nGpuLayers]; _gpuWo = new Tensor[_nGpuLayers];
        _gpuFfnNorm = new Tensor[_nGpuLayers];
        _gpuWGate = new Tensor[_nGpuLayers]; _gpuWUp = new Tensor[_nGpuLayers]; _gpuWDown = new Tensor[_nGpuLayers];

        if (_hasAttnBias) { _gpuBq = new Tensor[_nGpuLayers]; _gpuBk = new Tensor[_nGpuLayers]; _gpuBv = new Tensor[_nGpuLayers]; _gpuBo = new Tensor[_nGpuLayers]; }
        if (_hasQkNorm) { _gpuQNorm = new Tensor[_nGpuLayers]; _gpuKNorm = new Tensor[_nGpuLayers]; }

        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[_nGpuLayers];
        _gpuVCache = new Tensor[_nGpuLayers];

        Console.Error.Write($"[HybridForwardPass] Uploading {_nGpuLayers} GPU layers...");
        for (int i = 0; i < _nGpuLayers; i++)
        {
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuWq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _gpuWk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            _gpuWv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _gpuWo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _gpuFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            _gpuWGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
            _gpuWUp[i] = UploadWeight($"blk.{i}.ffn_up.weight");
            _gpuWDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");

            if (_hasAttnBias)
            {
                _gpuBq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _gpuBk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                _gpuBv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                _gpuBo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }
            if (_hasQkNorm)
            {
                _gpuQNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _gpuKNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            Console.Error.Write(".");
        }
        Console.Error.WriteLine(" done.");

        // ── Resolve CPU weights (layers nGpuLayers..numLayers-1) ──
        _cpuHidden = Alloc(_embDim);
        _cpuResidual = Alloc(_embDim);
        _cpuNormBuf = Alloc(_embDim);
        _cpuQ = Alloc(_numHeads * _headDim);
        _cpuK = Alloc(_numKvHeads * _headDim);
        _cpuV = Alloc(_numKvHeads * _headDim);
        _cpuAttnOut = Alloc(_numHeads * _headDim);
        _cpuFfnGate = Alloc(_intermDim);
        _cpuFfnUp = Alloc(_intermDim);
        _cpuAttnScores = Alloc(_maxSeqLen);

        _cpuAttnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWq = new CpuWeightRef[_nCpuLayers]; _cpuWk = new CpuWeightRef[_nCpuLayers];
        _cpuWv = new CpuWeightRef[_nCpuLayers]; _cpuWo = new CpuWeightRef[_nCpuLayers];
        _cpuFfnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWGate = new CpuWeightRef[_nCpuLayers]; _cpuWUp = new CpuWeightRef[_nCpuLayers]; _cpuWDown = new CpuWeightRef[_nCpuLayers];
        _cpuBq = new float*[_nCpuLayers]; _cpuBk = new float*[_nCpuLayers];
        _cpuBv = new float*[_nCpuLayers]; _cpuBo = new float*[_nCpuLayers];
        _cpuQNorm = new float*[_nCpuLayers]; _cpuKNorm = new float*[_nCpuLayers];

        for (int ci = 0; ci < _nCpuLayers; ci++)
        {
            int li = ci + _nGpuLayers; // actual layer index
            _cpuAttnNorm[ci] = ResolveCpuWeight($"blk.{li}.attn_norm.weight");
            _cpuWq[ci] = ResolveCpuWeight($"blk.{li}.attn_q.weight");
            _cpuWk[ci] = ResolveCpuWeight($"blk.{li}.attn_k.weight");
            _cpuWv[ci] = ResolveCpuWeight($"blk.{li}.attn_v.weight");
            _cpuWo[ci] = ResolveCpuWeight($"blk.{li}.attn_output.weight");
            _cpuFfnNorm[ci] = ResolveCpuWeight($"blk.{li}.ffn_norm.weight");
            _cpuWGate[ci] = ResolveCpuWeight($"blk.{li}.ffn_gate.weight");
            _cpuWUp[ci] = ResolveCpuWeight($"blk.{li}.ffn_up.weight");
            _cpuWDown[ci] = ResolveCpuWeight($"blk.{li}.ffn_down.weight");

            if (_hasAttnBias)
            {
                _cpuBq[ci] = LoadCpuBias($"blk.{li}.attn_q.bias", _numHeads * _headDim);
                _cpuBk[ci] = LoadCpuBias($"blk.{li}.attn_k.bias", _numKvHeads * _headDim);
                _cpuBv[ci] = LoadCpuBias($"blk.{li}.attn_v.bias", _numKvHeads * _headDim);
                _cpuBo[ci] = LoadCpuBias($"blk.{li}.attn_output.bias", _embDim);
            }
            if (_hasQkNorm)
            {
                _cpuQNorm[ci] = LoadCpuBias($"blk.{li}.attn_q_norm.weight", _headDim);
                _cpuKNorm[ci] = LoadCpuBias($"blk.{li}.attn_k_norm.weight", _headDim);
            }
        }

        _cpuKvCache = new KvCache(_nCpuLayers, _maxSeqLen, _numKvHeads, _headDim);

        if (_tqEnabled && _nCpuLayers > 0)
        {
            _cpuTqKvCache = new TurboQuantKvCache(_nCpuLayers, _maxSeqLen, _numKvHeads, _headDim);
            _cpuRotatedQuery = Alloc(_headDim);
            _cpuDecompBuf = Alloc(_headDim);
        }
    }

    // ================================================================
    //  Forward Pass
    // ================================================================

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // ── Phase 1: GPU layers ──
        _gpu.BeginRecord();

        // Embed token on GPU
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding, _gpuHidden, (uint)token, (uint)_embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding, _gpuHidden, (uint)token, (uint)_embDim);
        _gpu.RecordBarrier();

        for (int i = 0; i < _nGpuLayers; i++)
        {
            GpuLayer(i, position);
        }

        if (_nCpuLayers > 0)
        {
            // Download hidden state to pinned buffer
            CopyGpuBuffer(_pinnedHidden, _gpuHidden);
            _gpu.RecordTransferBarrier();
        }

        _gpu.EndRecordAndSubmit();

        if (_nCpuLayers > 0)
        {
            // ── Phase 2: Transfer GPU hidden → CPU ──
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new Span<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);

            // ── Phase 3: CPU layers ──
            for (int ci = 0; ci < _nCpuLayers; ci++)
            {
                CpuLayer(ci, position);
            }

            // Increment CPU KV cache
            if (_cpuTqKvCache != null)
                _cpuTqKvCache.IncrementPosition();
            else
                _cpuKvCache.IncrementPosition();

            // ── Phase 4: Transfer CPU hidden → GPU ──
            pinned = _gpu.MapPinned(_pinnedHidden);
            new ReadOnlySpan<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(pinned, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);

            // Upload pinned → GPU hidden, then final norm + output
            _gpu.BeginRecord();
            CopyGpuBuffer(_gpuHidden, _pinnedHidden);
            _gpu.RecordTransferBarrier();
        }
        else
        {
            _gpu.BeginRecord();
        }

        // ── Phase 5: Final norm + output projection on GPU ──
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden);

        _gpu.EndRecordAndSubmit();
        _gpu.Download(_gpuLogits, _logitsBuf);

        _kvLength = position + 1;
        return _logitsBuf;
    }

    public void ResetCache()
    {
        _kvLength = 0;
        if (_cpuTqKvCache != null)
            _cpuTqKvCache.Reset();
        else
            _cpuKvCache.Reset();
    }

    // ================================================================
    //  GPU Layer (same pattern as GpuForwardPass)
    // ================================================================

    private void GpuLayer(int i, int position)
    {
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordTransferBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuQ, _gpuWq[i], _gpuNormBuf);
        GpuMatMul(_gpuK, _gpuWk[i], _gpuNormBuf);
        GpuMatMul(_gpuV, _gpuWv[i], _gpuNormBuf);
        _gpu.RecordBarrier();

        if (_hasAttnBias)
        {
            _gpu.AddInPlace(_gpuQ, _gpuBq![i]);
            _gpu.AddInPlace(_gpuK, _gpuBk![i]);
            _gpu.AddInPlace(_gpuV, _gpuBv![i]);
            _gpu.RecordBarrier();
        }

        if (_hasQkNorm)
        {
            _gpu.HeadNorm(_gpuQ, _gpuQNorm![i], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
            _gpu.HeadNorm(_gpuK, _gpuKNorm![i], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
            _gpu.RecordBarrier();
        }

        _gpu.RoPE(_gpuQ, position, _headDim, _hp.RopeTheta);
        _gpu.RoPE(_gpuK, position, _headDim, _hp.RopeTheta);
        _gpu.RecordBarrier();

        _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[i], _gpuVCache[i],
            (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        _gpu.Attention(_gpuQ, _gpuKCache[i], _gpuVCache[i], _gpuAttnOut,
            (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
            (uint)(position + 1), (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWo[i], _gpuAttnOut);
        if (_hasAttnBias)
        {
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuBo![i]);
        }
        _gpu.RecordBarrier();
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);

        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordTransferBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuFfnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuFfnGate, _gpuWGate[i], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp, _gpuWUp[i], _gpuNormBuf);
        _gpu.RecordBarrier();

        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWDown[i], _gpuFfnGate);
        _gpu.RecordBarrier();

        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
    }

    // ================================================================
    //  CPU Layer (same pattern as ForwardPass)
    // ================================================================

    private void CpuLayer(int ci, int position)
    {
        // Save residual
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-attention RMS norm
        var normW = GetCpuNormWeight(_cpuAttnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, normW, _embDim, _hp.RmsNormEps);

        // Q/K/V projections
        SimdKernels.MatVec(_cpuQ, _cpuWq[ci].DataPtr, _cpuNormBuf, _numHeads * _headDim, _embDim, _cpuWq[ci].DType);
        SimdKernels.MatVec(_cpuK, _cpuWk[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWk[ci].DType);
        SimdKernels.MatVec(_cpuV, _cpuWv[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWv[ci].DType);

        if (_hasAttnBias)
        {
            SimdKernels.AddInPlace(_cpuQ, _cpuBq[ci], _numHeads * _headDim);
            SimdKernels.AddInPlace(_cpuK, _cpuBk[ci], _numKvHeads * _headDim);
            SimdKernels.AddInPlace(_cpuV, _cpuBv[ci], _numKvHeads * _headDim);
        }

        if (_hasQkNorm)
        {
            PerHeadRmsNorm(_cpuQ, _cpuQNorm[ci], _numHeads, _headDim, _hp.RmsNormEps);
            PerHeadRmsNorm(_cpuK, _cpuKNorm[ci], _numKvHeads, _headDim, _hp.RmsNormEps);
        }

        // RoPE
        SimdKernels.ApplyRoPE(_cpuQ, position, _numHeads, _headDim, _hp.RopeTheta);
        SimdKernels.ApplyRoPE(_cpuK, position, _numKvHeads, _headDim, _hp.RopeTheta);

        // KV cache append (ci = CPU layer index)
        if (_cpuTqKvCache != null)
        {
            _cpuTqKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }
        else
        {
            _cpuKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }

        // Attention
        if (_cpuTqKvCache != null)
            CpuTqAttention(ci, position);
        else
            CpuAttention(ci, position);

        // Output projection
        SimdKernels.MatVec(_cpuHidden, _cpuWo[ci].DataPtr, _cpuAttnOut, _embDim, _numHeads * _headDim, _cpuWo[ci].DType);
        if (_hasAttnBias)
            SimdKernels.AddInPlace(_cpuHidden, _cpuBo[ci], _embDim);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);

        // Save residual for FFN
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-FFN RMS norm
        var ffnNormW = GetCpuNormWeight(_cpuFfnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, ffnNormW, _embDim, _hp.RmsNormEps);

        // SwiGLU FFN
        SimdKernels.MatVec(_cpuFfnGate, _cpuWGate[ci].DataPtr, _cpuNormBuf, _intermDim, _embDim, _cpuWGate[ci].DType);
        SimdKernels.MatVec(_cpuFfnUp, _cpuWUp[ci].DataPtr, _cpuNormBuf, _intermDim, _embDim, _cpuWUp[ci].DType);

        SimdKernels.SiLuMul(_cpuFfnGate, _cpuFfnUp, _intermDim);

        SimdKernels.MatVec(_cpuHidden, _cpuWDown[ci].DataPtr, _cpuFfnGate, _embDim, _intermDim, _cpuWDown[ci].DType);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);
    }

    private void CpuAttention(int ci, int position)
    {
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / _headsPerKvGroup;
            float* qHead = _cpuQ + h * _headDim;
            float* outHead = _cpuAttnOut + h * _headDim;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = _cpuKvCache.KeyAt(ci, t) + kvHead * _headDim;
                _cpuAttnScores[t] = SimdKernels.DotF32(qHead, kVec, _headDim) * scale;
            }

            SimdKernels.SoftmaxInPlace(_cpuAttnScores, seqLen);

            for (int d = 0; d < _headDim; d++) outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = _cpuKvCache.ValueAt(ci, t) + kvHead * _headDim;
                float w = _cpuAttnScores[t];
                if (Fma.IsSupported && _headDim >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= _headDim; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < _headDim; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < _headDim; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        }
    }

    private void CpuTqAttention(int ci, int position)
    {
        var tq = _cpuTqKvCache!;
        int seqLen = position + 1;
        int tqLen = tq.TqLength;
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / _headsPerKvGroup;
            float* qHead = _cpuQ + h * _headDim;
            float* outHead = _cpuAttnOut + h * _headDim;

            // Rotate query once per head for TQ dot products
            var keyCompressor = tq.GetKeyCompressor(ci, kvHead);
            keyCompressor.RotateQuery(
                new ReadOnlySpan<float>(qHead, _headDim),
                new Span<float>(_cpuRotatedQuery, _headDim));

            // TQ-compressed positions
            for (int t = 0; t < tqLen; t++)
            {
                byte* tqKey = tq.TqKeyAt(ci, t, kvHead);
                float dot = TurboQuantOps.DequantDot3Scalar(
                    tqKey, _cpuRotatedQuery,
                    (float*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(keyCompressor.Centroids)),
                    _headDim);
                _cpuAttnScores[t] = dot * scale;
            }

            // FP32 window positions
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(ci, t) + kvHead * _headDim;
                _cpuAttnScores[t] = SimdKernels.DotF32(qHead, kVec, _headDim) * scale;
            }

            SimdKernels.SoftmaxInPlace(_cpuAttnScores, seqLen);

            for (int d = 0; d < _headDim; d++) outHead[d] = 0;

            // TQ values: decompress and accumulate
            var valCompressor = tq.GetValueCompressor(ci, kvHead);
            for (int t = 0; t < tqLen; t++)
            {
                byte* tqVal = tq.TqValueAt(ci, t, kvHead);
                var decompSpan = new Span<float>(_cpuDecompBuf, _headDim);
                valCompressor.Decompress(
                    new ReadOnlySpan<byte>(tqVal, tq.TqBlockSize), decompSpan);
                float w = _cpuAttnScores[t];
                for (int d = 0; d < _headDim; d++)
                    outHead[d] += w * _cpuDecompBuf[d];
            }

            // FP32 values
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(ci, t) + kvHead * _headDim;
                float w = _cpuAttnScores[t];
                if (Fma.IsSupported && _headDim >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= _headDim; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < _headDim; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < _headDim; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        }
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private readonly unsafe struct CpuWeightRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public CpuWeightRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        { Name = name; Info = info; DType = dtype; DataPtr = dataPtr; }
    }

    private CpuWeightRef ResolveCpuWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new CpuWeightRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private float* LoadCpuBias(string name, int count)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing bias tensor: {name}");
        var data = _model.GetTensorData(info);
        var buf = Alloc(count);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

    private float* GetCpuNormWeight(in CpuWeightRef tensor)
    {
        if (_cpuNormCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;
        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);
        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);
        _cpuNormCache[tensor.Name] = (nint)buf;
        return buf;
    }

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
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        _gpu.MatMul(output, weights, input,
            _gpuWeightDTypes.TryGetValue(weights.Handle, out var dt) ? dt : DType.Float32);
    }

    private void CopyGpuBuffer(Tensor dst, Tensor src)
    {
        var srcBuf = _gpu.GetBuffer(src);
        var dstBuf = _gpu.GetBuffer(dst);
        Vortice.Vulkan.VkBufferCopy region = new() { size = srcBuf.Size };
        _gpu.Vkd.vkCmdCopyBuffer(_gpu.TransferCmd, srcBuf.Buffer, dstBuf.Buffer, 1, &region);
    }

    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count, (nuint)sizeof(float));

    // ================================================================
    //  Disposal
    // ================================================================

    public void Dispose()
    {
        _gpu.Free(_gpuHidden); _gpu.Free(_gpuResidual); _gpu.Free(_gpuNormBuf);
        _gpu.Free(_gpuQ); _gpu.Free(_gpuK); _gpu.Free(_gpuV); _gpu.Free(_gpuAttnOut);
        _gpu.Free(_gpuFfnGate); _gpu.Free(_gpuFfnUp); _gpu.Free(_gpuLogits);
        _gpu.Free(_pinnedHidden);

        for (int i = 0; i < _nGpuLayers; i++)
        {
            _gpu.Free(_gpuAttnNorm[i]); _gpu.Free(_gpuFfnNorm[i]);
            _gpu.Free(_gpuWq[i]); _gpu.Free(_gpuWk[i]); _gpu.Free(_gpuWv[i]); _gpu.Free(_gpuWo[i]);
            _gpu.Free(_gpuWGate[i]); _gpu.Free(_gpuWUp[i]); _gpu.Free(_gpuWDown[i]);
            _gpu.Free(_gpuKCache[i]); _gpu.Free(_gpuVCache[i]);

            if (_hasAttnBias)
            { _gpu.Free(_gpuBq![i]); _gpu.Free(_gpuBk![i]); _gpu.Free(_gpuBv![i]); _gpu.Free(_gpuBo![i]); }
            if (_hasQkNorm)
            { _gpu.Free(_gpuQNorm![i]); _gpu.Free(_gpuKNorm![i]); }
        }
        _gpu.Free(_gpuOutputNorm);
        if (_gpuOutputWeight.Handle != _gpuEmbedding.Handle)
            _gpu.Free(_gpuOutputWeight);
        _gpu.Free(_gpuEmbedding);

        NativeMemory.Free(_cpuHidden); NativeMemory.Free(_cpuResidual); NativeMemory.Free(_cpuNormBuf);
        NativeMemory.Free(_cpuQ); NativeMemory.Free(_cpuK); NativeMemory.Free(_cpuV); NativeMemory.Free(_cpuAttnOut);
        NativeMemory.Free(_cpuFfnGate); NativeMemory.Free(_cpuFfnUp); NativeMemory.Free(_cpuAttnScores);
        if (_cpuRotatedQuery != null) NativeMemory.Free(_cpuRotatedQuery);
        if (_cpuDecompBuf != null) NativeMemory.Free(_cpuDecompBuf);
        _cpuKvCache.Dispose();
        _cpuTqKvCache?.Dispose();
    }
}
