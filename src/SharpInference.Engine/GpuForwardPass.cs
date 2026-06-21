using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.TurboQuant;
using Vortice.Vulkan;
using SharpInference.Vulkan;
using static Vortice.Vulkan.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// GPU-accelerated forward pass for LLaMA-family transformers.
/// All weight data resides in VRAM. Compute shaders handle dequantization,
/// MatVec, normalization, attention, and FFN on the GPU.
///
/// For operations not yet GPU-accelerated (attention scoring/aggregation),
/// falls back to CPU with download/upload round-trips.
/// </summary>
public sealed unsafe class GpuForwardPass : IForwardPass
{
    private readonly VulkanBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;

    // Pre-allocated logits download buffer (avoids GC allocation per token)
    private readonly float[] _logitsBuf;

    // GPU scratch buffers
    private readonly Tensor _hidden;     // [embDim]
    private readonly Tensor _residual;   // [embDim]
    private readonly Tensor _normBuf;    // [embDim]
    private readonly Tensor _q;          // [numHeads * headDim]
    private readonly Tensor _k;          // [numKvHeads * headDim]
    private readonly Tensor _v;          // [numKvHeads * headDim]
    private readonly Tensor _attnOut;    // [numHeads * headDim]
    private readonly Tensor _ffnGate;    // [intermDim]
    private readonly Tensor _ffnUp;      // [intermDim]
    private readonly Tensor _logits;     // [vocabSize]

    // Embedding table in VRAM (quantized for large vocabs, F32 for small)
    private readonly Tensor _gpuEmbedding;
    private readonly bool _embIsQuantized;

    // GPU weight tensors (Q4_K/Q6_K bytes uploaded to VRAM)
    private readonly Tensor[] _wAttnNorm;
    private readonly Tensor[] _wq, _wk, _wv, _wo;
    private readonly Tensor[] _wFfnNorm;
    private readonly Tensor[] _wGate, _wUp, _wDown;
    private readonly Tensor[]? _wGateInp, _wGateShexp, _wUpShexp, _wDownShexp;
    private readonly Tensor[][]? _wGateExps, _wUpExps, _wDownExps;
    private readonly Tensor _wOutputNorm;
    private readonly Tensor _wOutput;

    // Optional attention biases in VRAM (Qwen models)
    private readonly bool _hasAttnBias;
    // Qwen2 has Q/K/V bias but no output-projection bias — probed separately.
    private readonly bool _hasAttnOutputBias;
    private readonly Tensor[]? _bq, _bk, _bv, _bo;

    // Optional per-head Q/K RMSNorm weights in VRAM (Qwen3)
    private readonly bool _hasQkNorm;
    private readonly Tensor[]? _wqNorm, _wkNorm;

    // KV cache in VRAM: per-layer K and V buffers [maxSeqLen, kvDim]
    private readonly Tensor[] _gpuKCache;  // per layer (FP32, or FP32 window when TQ)
    private readonly Tensor[] _gpuVCache;  // per layer
    private readonly int _maxSeqLen;
    private int _kvLength; // current sequence length in cache

    /// <summary>Maximum sequence length (context size) configured for this forward pass.</summary>
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>Vocabulary size of this model.</summary>
    public int VocabSize => _hp.VocabSize;

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// The GPU K/V cache is updated via the length counter (_kvLength); no VRAM data is erased
    /// since subsequent appends will overwrite those positions.
    /// </summary>
    public void TruncateTo(int length)
    {
        _kvLength = length;
        _tqCompressedLen = Math.Min(_tqCompressedLen, length);
        _kvCache.TruncateTo(length);
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => true;

    // CPU KV cache kept for fallback (not used when GPU attention works)
    private readonly Engine.KvCache _kvCache;

    // TurboQuant GPU state (null when TQ disabled)
    private readonly bool _tqEnabled;
    private readonly int _tqFp32Window;
    private readonly int _tqBlockBytes;
    private Tensor[]? _gpuTqKCache;    // per layer, compressed VRAM
    private Tensor[]? _gpuTqVCache;    // per layer, compressed VRAM
    private Tensor[]? _gpuSignPatterns;  // [numKvHeads * headDim] sign flips, per layer
    private Tensor? _gpuCodebook;      // [8] centroids (3-bit)
    private Tensor? _gpuBoundaries;    // [7] decision boundaries
    private Tensor? _rotatedQ;         // [numHeads * headDim] WHT-rotated query
    private Tensor? _evictK;           // [numKvHeads * headDim] scratch for evicted FP32 entry
    private Tensor? _evictV;
    private Tensor _attnScoresScratch = default!; // [numHeads * maxSeqLen] long-context softmax-score spill; 1-float placeholder for short contexts
    private Tensor? _routerLogits;
    private Tensor? _moeSharedOut;
    private Tensor? _moeExpertOut;
    private int _tqCompressedLen;      // positions in TQ storage
    private int _fp32WriteIdx;         // ring buffer write position in FP32 window
    private int _fp32Count;            // number of FP32 positions currently stored

    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim, _expertDim;
    private readonly bool _isMoE, _hasSharedExpert;
    private readonly float[]? _routerBuf;

    // SnapKV (#59) — prefill-time eviction by attention-weight scoring. Mirrors
    // the CudaForwardPass layout. Every layer in GpuForwardPass is an attention
    // layer, so no per-layer-type filter on the capture.
    private readonly SnapKvConfig _snapKvCfg;
    private readonly int _snapKvEffectiveBudget;
    private Tensor? _snapKvQCapture;     // [numLayers × W × qDim] f32, captured during Prefill
    private int _snapKvQCaptureW;        // cached W the buffer was sized for
    private Tensor? _snapKvScoreAccum;   // [maxSeqLen] f32, per-position importance accumulator
    private Tensor? _snapKvScoreScratch; // [numHeads × maxSeqLen] f32, lazy scratch for the score kernel
    private bool _snapKvScoreScratchOwned; // false if aliased to _attnScoresScratch
    private int _snapKvCaptureSlot = -1; // 0..W-1 for tokens in the capture window; -1 otherwise

    public GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int maxContextLength = 0, bool enableTurboQuant = false, int tqFp32Window = 256, int tqBits = 3)
    {
        // Gemma 4's per-layer head_dim, sliding-window attention, PLE injection, KV-share tail,
        // and final-logit softcap have no Vulkan implementation — only CUDA and CPU. Running it
        // here would silently produce garbage (generic dense path with wrong dims / no PLE / no
        // softcap), so reject up front. hp.LayerHeadDim is the gemma4 master switch.
        if (hp.LayerHeadDim is not null)
            throw new NotSupportedException(
                "Gemma 4 models are not supported on the Vulkan backend (no SWA / PLE / per-layer " +
                "head_dim / softcap kernels). Use the CUDA backend (-g) or CPU (NGpuLayers=0).");

        _model = model;
        _gpu = gpu;
        _hp = hp;
        _tqEnabled = enableTurboQuant;

        if (maxContextLength > 0)
        {
            _maxSeqLen = Math.Min(maxContextLength, hp.ContextLength);
        }
        else if (enableTurboQuant)
        {
            _maxSeqLen = EstimateMaxContextTq(model, gpu, hp, tqFp32Window, tqBits);
        }
        else
        {
            _maxSeqLen = EstimateMaxContext(model, gpu, hp);
        }

        _tqFp32Window = enableTurboQuant ? Math.Min(tqFp32Window, _maxSeqLen) : 0;
        _tqBlockBytes = enableTurboQuant ? TurboQuantOps.BlockSize(tqBits, hp.HeadDim) : 0;

        // Bookkeeping-only: KV lives in GPU buffers, this tracks only the position counter.
        // Allocating the full host K/V buffers is pure waste (tens of GB at long ctx, #179).
        _kvCache = Engine.KvCache.CreateBookkeepingOnly(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.HeadDim);
        Console.Error.WriteLine($"[GpuForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength}){(enableTurboQuant ? " [TQ3]" : "")}");

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;
        _expertDim = hp.IsMoE ? hp.ExpertIntermediateDim : 0;
        _isMoE = hp.IsMoE;
        _hasSharedExpert = hp.HasSharedExpert;
        if (_tqEnabled && _headDim is not 128 and not 256)
            throw new NotSupportedException($"TurboQuant currently supports head dimensions 128 and 256; model head dim is {_headDim}.");
        _routerBuf = _isMoE ? new float[hp.NumExperts] : null;

        // SnapKV (issue #59) — gated by SHARPI_SNAPKV_BUDGET. Buffers are lazily
        // allocated on the first active prefill in Prefill(). Composition with
        // TurboQuant requires per-block ring bookkeeping that doesn't yet exist
        // (issue #60); explicit opt-in + TQ is rejected up front, and the auto
        // path stays disabled when TQ is on. Mirrors CudaForwardPass.
        _snapKvCfg = SnapKvConfig.FromEnvironment();
        if (_tqEnabled && _snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0)
            throw new NotSupportedException(
                "SnapKV + TurboQuant composition is not yet implemented (issue #60). " +
                "Set SHARPI_SNAPKV_BUDGET=0 to disable or disable --tq.");
        if (_snapKvCfg.IsBudgetExplicit)
        {
            _snapKvEffectiveBudget = _snapKvCfg.Budget;
        }
        else if (_tqEnabled)
        {
            _snapKvEffectiveBudget = 0;
        }
        else
        {
            long fullCacheBytes = (long)_hp.NumLayers * _maxSeqLen
                                * _numKvHeads * _headDim * 2 * sizeof(float); // K + V, fp32
            _snapKvEffectiveBudget = SnapKvConfig.ComputeAutoBudget(_maxSeqLen, fullCacheBytes);
            if (_snapKvEffectiveBudget > 0)
            {
                Console.Error.WriteLine(
                    $"[GpuForwardPass] SnapKV auto-enabled: budget={_snapKvEffectiveBudget}, " +
                    $"window={_snapKvCfg.Window}, recency={_snapKvCfg.Recency} " +
                    $"(full cache ~{fullCacheBytes / (1024.0 * 1024.0):F0} MiB; " +
                    "set SHARPI_SNAPKV_BUDGET=0 to disable).");
            }
        }

        // Allocate GPU scratch buffers
        _hidden = gpu.Allocate(TensorShape.D1(_embDim));
        _residual = gpu.Allocate(TensorShape.D1(_embDim));
        _normBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _q = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _k = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _v = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _attnOut = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        // Match CudaForwardPass: size scratch to the path this model actually uses.
        // MatMul derives row count from output.ElementCount, so a buffer sized
        // max(_intermDim, _expertDim) would make an expert MatMul write _intermDim rows
        // when only _expertDim are valid — the MoE-on-Vulkan garble that's been chased
        // since #2. Pure-MoE and pure-dense models both fall out correctly from this.
        // Sizing is centralized in ComputeFfnScratchDim and enforced by
        // ValidateFfnScratchDim so the MoE-vs-dense distinction can't drift (issue #315).
        int ffnScratchDim = ComputeFfnScratchDim(_isMoE, _intermDim, _expertDim);
        ValidateFfnScratchDim(_isMoE, ffnScratchDim, _intermDim, _expertDim);
        _ffnGate = gpu.Allocate(TensorShape.D1(ffnScratchDim));
        _ffnUp = gpu.Allocate(TensorShape.D1(ffnScratchDim));
        _logits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _routerLogits = _isMoE ? gpu.Allocate(TensorShape.D1(hp.NumExperts)) : null;
        _moeSharedOut = _isMoE && _hasSharedExpert ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _moeExpertOut = _isMoE ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _logitsBuf = new float[hp.VocabSize];

        // Allocate VRAM KV cache
        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[hp.NumLayers];
        _gpuVCache = new Tensor[hp.NumLayers];

        if (_tqEnabled)
        {
            // FP32 window: only tqFp32Window positions
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
            }

            // TQ compressed cache: (maxSeqLen - fp32Window) positions
            int maxTqPositions = Math.Max(0, _maxSeqLen - _tqFp32Window);
            long tqBytesPerPos = (long)_numKvHeads * _tqBlockBytes;
            // Allocate as uint buffer (shader accesses via uint[])
            long tqUintsPerLayer = (maxTqPositions * tqBytesPerPos + 3) / 4;
            _gpuTqKCache = new Tensor[hp.NumLayers];
            _gpuTqVCache = new Tensor[hp.NumLayers];
            _gpuSignPatterns = new Tensor[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuTqKCache[i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuTqVCache[i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuSignPatterns[i] = UploadTqSignPatterns(i);
            }

            // Upload TQ constants to VRAM
            var centroids = TurboQuantCodebooks.GetCentroids(tqBits, _headDim).ToArray();
            _gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));

            var boundaries = TurboQuantCodebooks.GetBoundaries(tqBits, _headDim).ToArray();
            _gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));

            _rotatedQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
            _evictK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
            _evictV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));

        }
        else
        {
            // Full FP32 cache: [maxSeqLen, kvDim] per layer
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            }
        }

        // Both TQ and FP32 attention shaders spill softmax scores to VRAM once the
        // live context exceeds the 4096-slot shared-memory fast path. Vulkan still
        // requires the descriptor bound on the fast path, so a 1-float placeholder
        // is sufficient when _maxSeqLen ≤ 4096.
        {
            long scratchElems = _maxSeqLen > 4096 ? (long)_numHeads * _maxSeqLen : 1L;
            _attnScoresScratch = gpu.Allocate(TensorShape.D1(scratchElems));
        }

        // Upload all weights to VRAM
        int L = hp.NumLayers;
        _wAttnNorm = new Tensor[L]; _wFfnNorm = new Tensor[L];
        _wq = new Tensor[L]; _wk = new Tensor[L]; _wv = new Tensor[L]; _wo = new Tensor[L];
        _wGate = new Tensor[L]; _wUp = new Tensor[L]; _wDown = new Tensor[L];
        _wGateInp = _isMoE ? new Tensor[L] : null;
        _wGateExps = _isMoE ? new Tensor[L][] : null;
        _wUpExps = _isMoE ? new Tensor[L][] : null;
        _wDownExps = _isMoE ? new Tensor[L][] : null;
        _wGateShexp = _isMoE && _hasSharedExpert ? new Tensor[L] : null;
        _wUpShexp = _isMoE && _hasSharedExpert ? new Tensor[L] : null;
        _wDownShexp = _isMoE && _hasSharedExpert ? new Tensor[L] : null;

        _hasAttnBias = hp.HasAttnBias;
        _hasAttnOutputBias = hp.HasAttnOutputBias;
        if (_hasAttnBias)
        {
            _bq = new Tensor[L]; _bk = new Tensor[L];
            _bv = new Tensor[L]; _bo = new Tensor[L];
        }

        _hasQkNorm = hp.HasQkNorm;
        if (_hasQkNorm)
        {
            _wqNorm = new Tensor[L]; _wkNorm = new Tensor[L];
        }

        Console.Error.Write($"[GpuForwardPass] Uploading {L} layers to VRAM...");
        for (int i = 0; i < L; i++)
        {
            _wAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _wq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _wk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            _wv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _wo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _wFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            if (_isMoE)
            {
                _wGateInp![i] = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                _wGateExps![i] = UploadExpertWeights($"blk.{i}.ffn_gate_exps.weight", _expertDim, _embDim, hp.NumExperts);
                _wUpExps![i] = UploadExpertWeights($"blk.{i}.ffn_up_exps.weight", _expertDim, _embDim, hp.NumExperts);
                _wDownExps![i] = UploadExpertWeights($"blk.{i}.ffn_down_exps.weight", _embDim, _expertDim, hp.NumExperts);
                if (_hasSharedExpert)
                {
                    _wGateShexp![i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                    _wUpShexp![i] = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                    _wDownShexp![i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _wGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _wUp[i] = UploadWeight($"blk.{i}.ffn_up.weight");
                _wDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _bq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _bk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                _bv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                if (_hasAttnOutputBias)
                    _bo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _wqNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _wkNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            Console.Error.Write(".");
        }
        // Upload embedding table to VRAM — keep quantized for Q4_K to save VRAM
        Console.Error.Write(" emb...");
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        if (embInfo.DType == DType.Q4_K)
        {
            // Upload raw quantized bytes (reinterpret as uint32 for storage buffer)
            var embData = model.GetTensorData(embInfo);
            int floatCount = embData.Length / 4;
            var raw = new float[floatCount];
            embData.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
            _gpuEmbedding = gpu.Upload(raw, TensorShape.D1(floatCount));
            _embIsQuantized = true;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Q4_K;
        }
        else
        {
            // Small vocab or F32: dequantize to F32
            var embData = model.GetTensorData(embInfo);
            var embF32 = new float[(int)embInfo.ElementCount];
            Dequantize.ToFloat32(embData, embF32, embInfo.DType, embInfo.ElementCount);
            _gpuEmbedding = gpu.Upload(embF32, TensorShape.D1(embF32.Length));
            _embIsQuantized = false;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Float32;
        }

        _wOutputNorm = UploadWeight("output_norm.weight");
        _wOutput = model.FindTensor("output.weight") is not null
            ? UploadWeight("output.weight")
            : _gpuEmbedding;

        Console.Error.WriteLine(" done.");
    }

    public Engine.KvCache Cache => _kvCache;
    public int KvLength => _kvLength;

    public void ResetCache()
    {
        _kvLength = 0;
        _tqCompressedLen = 0;
        _fp32WriteIdx = 0;
        _fp32Count = 0;
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        int N = tokens.Count;

        // SnapKV (issue #59) gating: only run eviction when this is a fresh
        // prefill (startPos==0), the effective budget is positive, the prompt
        // is long enough that eviction would actually drop something, and TQ
        // is off (composition with the TQ ring is #60).
        bool snapKvActive = _snapKvEffectiveBudget > 0
                         && startPos == 0
                         && !_tqEnabled
                         && N > _snapKvEffectiveBudget
                         && N > _snapKvCfg.Window;
        int W = 0, wStart = 0;
        if (snapKvActive)
        {
            W = Math.Min(_snapKvCfg.Window, N);
            wStart = N - W;
            EnsureSnapKvCaptureBuffer(W);
        }

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < N; i++)
        {
            // Drive Q-capture for the last W tokens — Forward reads
            // _snapKvCaptureSlot and writes _q into _snapKvQCapture.
            _snapKvCaptureSlot = (snapKvActive && i >= wStart) ? (i - wStart) : -1;
            logits = Forward(tokens[i], startPos + i);
        }
        _snapKvCaptureSlot = -1;

        if (snapKvActive)
            ApplySnapKvEviction(N, W, wStart);

        return logits;
    }

    /// <summary>
    /// SnapKV (issue #59): score the captured trailing-W queries against the
    /// VRAM K cache for every layer (atomicAdd-pooled into a single per-position
    /// accumulator), download the accumulator, pick a keep set, then compact the
    /// GPU K/V rings + the host-side <see cref="_kvCache"/> length bookkeeping.
    /// Called once at the end of a SnapKV-active prefill. Mirrors
    /// <c>CudaForwardPass.ApplySnapKvEviction</c>.
    /// </summary>
    private void ApplySnapKvEviction(int N, int W, int wStart)
    {
        EnsureSnapKvScoreBuffers();
        // Zero only the prompt-prefix slice; the rest of the [maxSeqLen] buffer
        // doesn't participate in scoring and will not be downloaded.
        _gpu.ClearRegion(_snapKvScoreAccum!, 0, N);

        // Record ALL (layer, w) score dispatches into one command buffer and
        // submit once. The CAS-based atomicAdd on _snapKvScoreAccum (binding 2
        // of SnapKvScore) serialises writes globally, so dispatch order doesn't
        // affect the final accumulator. WAR/WAW hazards on the reused _q buffer
        // are handled by RecordBarrier between each (CopyRegion → SnapKvScore)
        // pair and after each SnapKvScore (before the next iter's CopyRegion).
        // For Qwen3-8B (36 × 64 = 2304 dispatches) this collapses ~50 ms of
        // per-submit overhead into a single submission. (Closes #66.)
        int qDim = _numHeads * _headDim;
        _gpu.BeginRecord();
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            for (int w = 0; w < W; w++)
            {
                long srcOffsetElems = ((long)layer * _snapKvQCaptureW + w) * qDim;
                _gpu.RecordComputeCopyRegion(_q, 0,
                    _snapKvQCapture!, srcOffsetElems * sizeof(float),
                    (long)qDim * sizeof(float));
                _gpu.RecordBarrier();

                int qAbsPos = wStart + w;
                _gpu.SnapKvScore(_q, _gpuKCache[layer],
                    _snapKvScoreAccum!, _snapKvScoreScratch!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                    (uint)N, (uint)qAbsPos, (uint)_maxSeqLen);
                _gpu.RecordBarrier();
            }
        }
        _gpu.EndRecordAndSubmit();

        // Download the prompt-length prefix of the accumulator and pick the keep set.
        var hostScores = new float[N];
        _gpu.Download(_snapKvScoreAccum!, hostScores);

        var selector = new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
        selector.LoadScores(hostScores, N);
        int[] keep = selector.SelectKeepSet(N, _snapKvEffectiveBudget, _snapKvCfg.Recency);
        int K = keep.Length;
        if (K >= N)
        {
            // No actual eviction — leave the GPU ring + bookkeeping alone.
            return;
        }

        // Upload the keep list to device (int32) for the gather kernels.
        ReadOnlySpan<byte> keepBytes = MemoryMarshal.AsBytes(keep.AsSpan());
        var keepDev = _gpu.UploadRaw(keepBytes, TensorShape.D1(K), DType.Int32);
        int kvDim = _numKvHeads * _headDim;
        var stage = _gpu.Allocate(TensorShape.D1((long)K * kvDim));
        try
        {
            long sliceBytes = (long)K * kvDim * sizeof(float);
            _gpu.BeginRecord();
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                // K: gather kept positions into stage, then copy stage back over
                // the cache's [0, K * kvDim) prefix. Same for V. Two-phase to
                // avoid src==dst race (workgroup ordering is undefined).
                _gpu.KvCompact(_gpuKCache[layer], stage, keepDev, (uint)K, (uint)kvDim);
                _gpu.RecordBarrier();
                _gpu.RecordComputeCopyRegion(_gpuKCache[layer], 0, stage, 0, sliceBytes);
                _gpu.RecordBarrier();
                _gpu.KvCompact(_gpuVCache[layer], stage, keepDev, (uint)K, (uint)kvDim);
                _gpu.RecordBarrier();
                _gpu.RecordComputeCopyRegion(_gpuVCache[layer], 0, stage, 0, sliceBytes);
                _gpu.RecordBarrier();
            }
            _gpu.EndRecordAndSubmit();
        }
        finally
        {
            _gpu.Free(stage);
            _gpu.Free(keepDev);
        }

        // Update host-side length bookkeeping. _kvCache is bookkeeping-only on
        // GpuForwardPass — actual data lives in _gpuKCache/_gpuVCache.
        _kvLength = K;
        _kvCache.TruncateTo(K);
    }

    private void EnsureSnapKvCaptureBuffer(int W)
    {
        if (_snapKvQCapture is not null && _snapKvQCaptureW >= W) return;
        if (_snapKvQCapture is { } old) _gpu.Free(old);
        int qDim = _numHeads * _headDim;
        long elems = (long)_hp.NumLayers * W * qDim;
        _snapKvQCapture = _gpu.Allocate(TensorShape.D1(elems));
        _snapKvQCaptureW = W;
    }

    private void EnsureSnapKvScoreBuffers()
    {
        if (_snapKvScoreAccum is null)
            _snapKvScoreAccum = _gpu.Allocate(TensorShape.D1(_maxSeqLen));

        // The SnapKvScore kernel only reads/writes scratch when prompt_len > 4096
        // (the shared-memory fast-path cap), but it always *binds* it. When
        // _maxSeqLen ≤ 4096 the kernel never touches scratch, so the 1-float
        // _attnScoresScratch placeholder is sufficient — same convention as
        // Attention / TqAttention. Above the cap, reuse the real attention
        // scratch when sized large enough; otherwise allocate a dedicated buffer.
        // Track ownership so Dispose doesn't double-free the aliased case.
        if (_snapKvScoreScratch is null)
        {
            if (_maxSeqLen <= 4096
                || (_attnScoresScratch.Handle != 0 && _attnScoresScratch.ElementCount >= (long)_numHeads * _maxSeqLen))
            {
                _snapKvScoreScratch = _attnScoresScratch;
                _snapKvScoreScratchOwned = false;
            }
            else
            {
                _snapKvScoreScratch = _gpu.Allocate(TensorShape.D1((long)_numHeads * _maxSeqLen));
                _snapKvScoreScratchOwned = true;
            }
        }
    }

    /// <summary>
    /// Run one token through the transformer on GPU. Returns logits span (downloaded from VRAM).
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // Record ALL dispatches into ONE command buffer
        _gpu.BeginRecord();

        // Embed token (GPU lookup from cached table — no PCIe transfer)
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
        _gpu.RecordBarrier();

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Copy hidden → residual + RmsNorm (barrier after both)
            CopyBuffer(_residual, _hidden);
            _gpu.RecordBarrier();

            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // Q/K/V all read normBuf (no conflict between them)
            GpuMatMul(_q, _wq[layer], _normBuf);
            GpuMatMul(_k, _wk[layer], _normBuf);
            GpuMatMul(_v, _wv[layer], _normBuf);
            _gpu.RecordBarrier(); // Q/K/V done → bias + RoPE

            if (_hasAttnBias)
            {
                _gpu.AddInPlace(_q, _bq![layer]);
                _gpu.AddInPlace(_k, _bk![layer]);
                _gpu.AddInPlace(_v, _bv![layer]);
                _gpu.RecordBarrier();
            }

            // NoPE: skip RoPE for NoPE layers
            {
                bool useRoPE = _hp.NoRopeLayerStep == 0
                    || (layer + 1) % _hp.NoRopeLayerStep != 0;

                // Ordering (issue #157): RoPE does NOT commute with per-channel-weighted
                // QK-norm (NEOX RoPE mixes channels i and i+d/2, which carry different
                // learned weights), so mirror the CPU ForwardPass / HF Qwen3 / llama.cpp
                // build_qwen3 order exactly:
                //   • weighted QK-norm (Qwen3, …): norm BEFORE RoPE
                //   • L2 QK-norm (Llama-4):        norm AFTER  RoPE (RoPE layers only)
                if (_hasQkNorm && !_hp.UseL2QkNorm)
                {
                    _gpu.HeadNorm(_q, _wqNorm![layer], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.HeadNorm(_k, _wkNorm![layer], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.RecordBarrier();
                }

                if (useRoPE)
                {
                    // RoPE on Q and K
                    _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    _gpu.RecordBarrier();
                }

                // L2 QK-norm (Llama-4): pure RMS norm without learned weights, after RoPE,
                // only on RoPE layers per llama.cpp.
                if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                {
                    _gpu.HeadNormPure(_q, (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.HeadNormPure(_k, (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.RecordBarrier();
                }
            }

            // SnapKV (issue #59): capture the post-RoPE / post-Q-norm query for
            // this (layer, token) into the scoring ring. Gated by the Prefill
            // wrapper — outside that path _snapKvCaptureSlot stays -1 and we skip.
            // Every layer here is an attention layer, so no per-layer-type filter.
            if (_snapKvCaptureSlot >= 0 && _snapKvQCapture is { } capBuf)
            {
                int qDim = _numHeads * _headDim;
                long dstOffsetElems = ((long)layer * _snapKvQCaptureW + _snapKvCaptureSlot) * qDim;
                _gpu.RecordComputeCopyRegion(capBuf, dstOffsetElems * sizeof(float),
                                             _q, 0, (long)qDim * sizeof(float));
                _gpu.RecordBarrier();
            }
            // KV append reads K/V (with or without RoPE)

            if (_tqEnabled)
            {
                int kvDimLocal = _numKvHeads * _headDim;
                long rowBytes = (long)kvDimLocal * sizeof(float);

                // If FP32 window is full, compress the oldest entry before overwriting
                if (_fp32Count >= _tqFp32Window)
                {
                    // Copy oldest FP32 row (at _fp32WriteIdx) to evict buffers
                    CopyBufferRegion(_evictK!, 0, _gpuKCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    CopyBufferRegion(_evictV!, 0, _gpuVCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    _gpu.RecordBarrier();

                    // Compress evicted entry into TQ cache
                    _gpu.TqKvAppend(_evictK!, _evictV!, _gpuTqKCache![layer], _gpuTqVCache![layer],
                        _gpuSignPatterns![layer], _gpuCodebook!, _gpuBoundaries!,
                        (uint)kvDimLocal, (uint)_headDim, (uint)_tqCompressedLen,
                        (uint)_maxSeqLen, (uint)_numKvHeads, (uint)_tqBlockBytes);
                    _gpu.RecordBarrier();
                }

                // Write new K/V into FP32 ring buffer at _fp32WriteIdx
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)kvDimLocal, (uint)_fp32WriteIdx, (uint)_tqFp32Window);
                _gpu.RecordBarrier();

                // Rotate query for TQ attention
                _gpu.TqRotateQuery(_q, _rotatedQ!, _gpuSignPatterns![layer],
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim);
                _gpu.RecordBarrier();

                // TQ Attention: handles both compressed and FP32 regions
                uint fp32SeqLen = (uint)Math.Min(_fp32Count + 1, _tqFp32Window);
                _gpu.TqAttention(_q, _rotatedQ!, _gpuTqKCache![layer], _gpuTqVCache![layer],
                    _gpuKCache[layer], _gpuVCache[layer], _attnOut, _gpuCodebook!,
                    _attnScoresScratch!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                    (uint)_tqCompressedLen, fp32SeqLen, (uint)_maxSeqLen, (uint)_tqBlockBytes);
            }
            else
            {
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
                _gpu.RecordBarrier();

                _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _attnScoresScratch,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                    (uint)(position + 1), (uint)_maxSeqLen);
            }
            _gpu.RecordBarrier(); // attnOut done → output projection

            GpuMatMul(_hidden, _wo[layer], _attnOut);
            if (_hasAttnOutputBias)
            {
                _gpu.RecordBarrier();
                _gpu.AddInPlace(_hidden, _bo![layer]);
            }
            _gpu.RecordBarrier(); // hidden written → add

            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier(); // hidden done → FFN copy reads it

            CopyBuffer(_residual, _hidden);
            _gpu.RecordBarrier();

            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            if (_isMoE)
                GpuMoeFfn(layer);
            else
                GpuDenseFfn(layer);
            _gpu.RecordBarrier(); // hidden written → add

            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier(); // hidden done → next layer's compute copy reads it
        }

        // Update TQ ring buffer state (after all layers used the same indices)
        if (_tqEnabled)
        {
            if (_fp32Count >= _tqFp32Window)
                _tqCompressedLen++;

            _fp32WriteIdx = (_fp32WriteIdx + 1) % _tqFp32Window;
            if (_fp32Count < _tqFp32Window)
                _fp32Count++;
        }

        // Final norm + output projection
        _gpu.RecordBarrier(); // last layer's AddInPlace → final norm
        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_logits, _wOutput, _hidden);

        // Fold the logits download into the main submit: no second command buffer needed.
        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_logits, _logitsBuf.Length);

        _gpu.EndRecordAndSubmit();

        _gpu.ReadFromStaging(_logitsBuf);
        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        var dtype = _weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K);
        _gpu.MatMul(output, weights, input, dtype);
    }

    private void GpuDenseFfn(int layer)
    {
        GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
        GpuMatMul(_ffnUp, _wUp[layer], _normBuf);
        _gpu.RecordBarrier();

        _gpu.SiLuMul(_ffnGate, _ffnUp);
        _gpu.RecordBarrier();

        GpuMatMul(_hidden, _wDown[layer], _ffnGate);
    }

    private void GpuMoeFfn(int layer)
    {
        int numActive = _hp.NumActiveExperts;

        GpuMatMul(_routerLogits!, _wGateInp![layer], _normBuf);
        _gpu.RecordBarrier();
        if (_hp.UseSigmoidGating)
            _gpu.Sigmoid(_routerLogits!);
        else
            _gpu.Softmax(_routerLogits!);
        _gpu.EndRecordAndSubmit();
        _gpu.Download(_routerLogits!, _routerBuf!);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerBuf!, numActive, selectedExperts, expertWeights, _hp.NormalizeMoeTopKWeights);

        _gpu.BeginRecord();

        if (_hasSharedExpert)
        {
            GpuMatMul(_ffnGate, _wGateShexp![layer], _normBuf);
            GpuMatMul(_ffnUp, _wUpShexp![layer], _normBuf);
            _gpu.RecordBarrier();
            _gpu.SiLuMul(_ffnGate, _ffnUp);
            _gpu.RecordBarrier();
            GpuMatMul(_moeSharedOut!, _wDownShexp![layer], _ffnGate);
            _gpu.RecordBarrier();
        }

        _gpu.Clear(_hidden);
        _gpu.RecordBarrier();

        for (int i = 0; i < numActive; i++)
        {
            int expertIdx = selectedExperts[i];
            float expertWeight = expertWeights[i];

            GpuMatMul(_ffnGate, _wGateExps![layer][expertIdx], _normBuf);
            GpuMatMul(_ffnUp, _wUpExps![layer][expertIdx], _normBuf);
            _gpu.RecordBarrier();

            if (_hp.UseSigmoidGating)
            {
                _gpu.ScaleInPlace(_ffnGate, expertWeight);
                _gpu.ScaleInPlace(_ffnUp, expertWeight);
                _gpu.RecordBarrier();
            }

            _gpu.SiLuMul(_ffnGate, _ffnUp);
            _gpu.RecordBarrier();
            GpuMatMul(_moeExpertOut!, _wDownExps![layer][expertIdx], _ffnGate);
            _gpu.RecordBarrier();

            if (_hp.UseSigmoidGating)
                _gpu.AddInPlace(_hidden, _moeExpertOut!);
            else
                _gpu.AddScaledInPlace(_hidden, _moeExpertOut!, expertWeight);
            _gpu.RecordBarrier();
        }

        if (_hasSharedExpert)
            _gpu.AddInPlace(_hidden, _moeSharedOut!);
    }

    private void EmbedToken(int token)
    {
        // Dequantize embedding row on CPU, upload to GPU
        var info = _model.FindTensor("token_embd.weight")!.Value;
        var data = _model.GetTensorData(info);
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        int rowOffset = token * bytesPerRow;

        var embedding = new float[_embDim];
        Dequantize.ToFloat32(
            data.Slice(rowOffset, bytesPerRow),
            embedding, info.DType, _embDim);

        UploadToExisting(_hidden, embedding);
    }

    private void UploadToExisting(Tensor gpuTensor, ReadOnlySpan<float> data)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(float));
        using var staging = GpuBuffer.CreateStaging(_gpu, byteSize,
            Vortice.Vulkan.VkBufferUsageFlags.TransferSrc);
        float* mapped = (float*)staging.Map();
        data.CopyTo(new Span<float>(mapped, data.Length));
        staging.Unmap();

        // Copy staging → device buffer via command buffer
        var vkd = _gpu.Vkd;
        var cmd = _gpu.TransferCmd;
        Vortice.Vulkan.VkCommandBufferBeginInfo beginInfo = new()
        {
            flags = Vortice.Vulkan.VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        vkd.vkBeginCommandBuffer(cmd, &beginInfo).CheckResult();

        Vortice.Vulkan.VkBufferCopy region = new() { size = byteSize };
        vkd.vkCmdCopyBuffer(cmd, staging.Buffer, _gpu.GetBuffer(gpuTensor).Buffer, 1, &region);

        vkd.vkEndCommandBuffer(cmd).CheckResult();
        Vortice.Vulkan.VkSubmitInfo submit = new()
        {
            commandBufferCount = 1,
            pCommandBuffers = &cmd,
        };
        var fence = _gpu.Fence;
        vkd.vkResetFences(1, &fence).CheckResult();
        vkd.vkQueueSubmit(_gpu.ComputeQueue, 1, &submit, fence).CheckResult();
        vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    /// <summary>Record a buffer copy into the current command buffer (must be in recording mode).</summary>
    private void CopyBuffer(Tensor dst, Tensor src)
    {
        _gpu.RecordComputeCopy(dst, src);
    }

    /// <summary>Copy a sub-region from src to dst (both in VRAM).</summary>
    private void CopyBufferRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
    {
        _gpu.RecordComputeCopyRegion(dst, dstOffsetBytes, src, srcOffsetBytes, sizeBytes);
    }

    // Track quantization type per weight tensor for MatMul dispatch
    private readonly Dictionary<nint, DType> _weightDTypes = new();

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
        else if (IsRawGpuQuant(info.DType))
        {
            // Upload raw quantized bytes (reinterpret as floats for storage buffer).
            // Round up: Q8_0 (34B) / Q4_0 (18B) blocks can make a non-4-multiple total.
            int floatCount = (data.Length + 3) / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _weightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Other types: dequantize to F32 on CPU
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private Tensor[] UploadExpertWeights(string name, int rows, int cols, int expertCount)
    {
        var tensors = new Tensor[expertCount];
        for (int expertIdx = 0; expertIdx < expertCount; expertIdx++)
            tensors[expertIdx] = UploadExpertWeight(name, rows, cols, expertIdx);
        return tensors;
    }

    private Tensor UploadTqSignPatterns(int layerIndex)
    {
        var fullSigns = new float[_numKvHeads * _headDim];
        for (int h = 0; h < _numKvHeads; h++)
        {
            var headSigns = WalshHadamard.GenerateSignPattern(_headDim, layerIndex * _numKvHeads + h);
            headSigns.CopyTo(fullSigns.AsSpan(h * _headDim));
        }

        return _gpu.Upload(fullSigns, TensorShape.D1(fullSigns.Length));
    }

    private Tensor UploadExpertWeight(string name, int rows, int cols, int expertIdx)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _weightDTypes[result.Handle] = DType.Float32;
            return result;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (IsRawGpuQuant(info.DType))
        {
            int floatCount = (expertData.Length + 3) / 4;
            var rawFloats = new float[floatCount];
            expertData.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            var result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _weightDTypes[result.Handle] = info.DType;
            return result;
        }
        else
        {
            int count = rows * cols;
            var f32 = new float[count];
            Dequantize.ToFloat32(expertData, f32, info.DType, count);
            var result = _gpu.Upload(f32, TensorShape.D1(count));
            _weightDTypes[result.Handle] = DType.Float32;
            return result;
        }
    }

    private static void SelectTopK(ReadOnlySpan<float> logits, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                {
                    if (indices[j] != i)
                        continue;

                    alreadySelected = true;
                    break;
                }

                if (!alreadySelected && logits[i] > bestVal)
                {
                    bestVal = logits[i];
                    bestIdx = i;
                }
            }

            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        if (!normalize || k <= 1)
            return;

        float sum = 0;
        for (int i = 0; i < k; i++)
            sum += weights[i];
        if (sum <= 0)
            return;
        for (int i = 0; i < k; i++)
            weights[i] /= sum;
    }

    /// <summary>
    /// Estimates max context when using TurboQuant compressed KV cache.
    /// TQ3 uses ~52 bytes per head per position vs 512 bytes (128 floats) for FP32.
    /// </summary>
    public static int EstimateMaxContextTq(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int fp32WindowSize = 256, int bits = 3)
    {
        long vramBytes = (long)gpu.VramBytes;
        int headDim = hp.HeadDim;
        int blockSize = TurboQuantOps.BlockSize(bits, headDim);

        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += EstimateGpuTensorBytes(t);

        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        // See note on the non-TQ path: keep ≥ 2 GiB free so late weight allocations
        // (notably lm-head) stay in HBM instead of getting paged to system memory.
        long reserved = Math.Max(vramBytes / 3, 2L * 1024 * 1024 * 1024);
        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024;

        // FP32 window: 2 * layers * kvDim * sizeof(float) per position
        long fp32Bytes = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float) * fp32WindowSize;

        // TQ: 2 * layers * numKvHeads * blockSize per position
        long tqBytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * blockSize;

        long availableForTq = available - fp32Bytes;
        if (availableForTq <= 0) availableForTq = 64L * 1024 * 1024;

        int maxTqPositions = (int)(availableForTq / tqBytesPerToken);
        int maxCtx = Math.Clamp(maxTqPositions + fp32WindowSize, 512, hp.ContextLength);

        return maxCtx;
    }

    private static int EstimateMaxContext(GgufModel model, VulkanBackend gpu, ModelHyperparams hp)
    {
        long vramBytes = (long)gpu.VramBytes;

        // Estimate weight VRAM: raw quantized bytes padded to 4-byte alignment per tensor
        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += EstimateGpuTensorBytes(t);

        // Scratch buffers (F32): hidden, residual, norm, Q, K, V, attnOut, ffnGate, ffnUp, logits
        int headDim = hp.HeadDim;
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        // Reserve at least 2 GiB (or a third of total) for the driver, staging buffers,
        // OS/desktop compositor, GPU buffer pool reuse, and any per-allocation overhead.
        // The previous max(vram/5, 1 GiB) leaves only ~24 MiB free on a 12 GiB card for
        // Qwen3-8B at the auto-picked context — anything allocated *late* (notably the
        // 600 MiB lm-head weight) then gets mapped into system memory by the driver,
        // and the kernel reads those weights at ~30 GB/s over PCIe instead of ~400 GB/s
        // in HBM. The CUDA backend hit this hard (4 t/s on Qwen3 prefill before the fix);
        // Vulkan currently escapes only because it has no eager image-ops buffer eating
        // 2.5 GiB at construction. Same trapdoor though, so close it here too.
        long reserved = Math.Max(vramBytes / 3, 2L * 1024 * 1024 * 1024);

        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024; // minimum fallback

        // KV cache: 2 (K+V) * numLayers * numKvHeads * headDim * sizeof(float) per token
        long bytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float);

        int maxCtx = (int)(available / bytesPerToken);
        maxCtx = Math.Clamp(maxCtx, 512, hp.ContextLength);

        return maxCtx;
    }

    public void Dispose()
    {
        _gpu.Free(_hidden); _gpu.Free(_residual); _gpu.Free(_normBuf);
        _gpu.Free(_q); _gpu.Free(_k); _gpu.Free(_v); _gpu.Free(_attnOut);
        _gpu.Free(_ffnGate); _gpu.Free(_ffnUp); _gpu.Free(_logits);
        if (_routerLogits is not null) _gpu.Free(_routerLogits);
        if (_moeSharedOut is not null) _gpu.Free(_moeSharedOut);
        if (_moeExpertOut is not null) _gpu.Free(_moeExpertOut);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_wAttnNorm[i]); _gpu.Free(_wFfnNorm[i]);
            _gpu.Free(_wq[i]); _gpu.Free(_wk[i]); _gpu.Free(_wv[i]); _gpu.Free(_wo[i]);
            if (_isMoE)
            {
                _gpu.Free(_wGateInp![i]);
                foreach (var t in _wGateExps![i]) _gpu.Free(t);
                foreach (var t in _wUpExps![i]) _gpu.Free(t);
                foreach (var t in _wDownExps![i]) _gpu.Free(t);
                if (_hasSharedExpert)
                {
                    _gpu.Free(_wGateShexp![i]);
                    _gpu.Free(_wUpShexp![i]);
                    _gpu.Free(_wDownShexp![i]);
                }
            }
            else
            {
                _gpu.Free(_wGate[i]); _gpu.Free(_wUp[i]); _gpu.Free(_wDown[i]);
            }

            if (_hasAttnBias)
            {
                _gpu.Free(_bq![i]); _gpu.Free(_bk![i]);
                _gpu.Free(_bv![i]);
                if (_hasAttnOutputBias) _gpu.Free(_bo![i]);
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

        if (_tqEnabled)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                _gpu.Free(_gpuTqKCache![i]);
                _gpu.Free(_gpuTqVCache![i]);
                _gpu.Free(_gpuSignPatterns![i]);
            }
            _gpu.Free(_gpuCodebook!);
            _gpu.Free(_gpuBoundaries!);
            _gpu.Free(_rotatedQ!);
            _gpu.Free(_evictK!);
            _gpu.Free(_evictV!);
        }
        _gpu.Free(_attnScoresScratch);

        // SnapKV (#59) buffers
        if (_snapKvQCapture is { } capBuf) _gpu.Free(capBuf);
        if (_snapKvScoreAccum is { } accBuf) _gpu.Free(accBuf);
        if (_snapKvScoreScratch is { } scrBuf && _snapKvScoreScratchOwned) _gpu.Free(scrBuf);

        _kvCache.Dispose();
    }

    /// <summary>
    /// FFN scratch dimension: MoE expert FFNs write _expertDim rows; dense FFNs write _intermDim.
    /// Centralized so the MoE-vs-dense distinction can't drift (issue #315).
    /// </summary>
    internal static int ComputeFfnScratchDim(bool isMoE, int intermDim, int expertDim)
        => isMoE ? expertDim : intermDim;

    /// <summary>
    /// Invariant guard for the FFN scratch buffers (issue #315 / "the MoE-on-Vulkan garble
    /// chased since #2"). VulkanBackend.MatMul derives output row count from
    /// output.ElementCount, so an expert MatMul writing into a buffer sized
    /// max(_intermDim,_expertDim) would silently corrupt expert output. Fail loudly instead.
    /// </summary>
    internal static void ValidateFfnScratchDim(bool isMoE, int scratchDim, int intermDim, int expertDim)
    {
        if (isMoE)
        {
            if (expertDim <= 0)
                throw new InvalidOperationException(
                    "MoE model (IsMoE=true) but ExpertIntermediateDim is 0 — check GGUF metadata.");
            if (scratchDim != expertDim)
                throw new InvalidOperationException(
                    $"MoE FFN scratch dim {scratchDim} must equal _expertDim {expertDim}, " +
                    $"not max(_intermDim={intermDim}, _expertDim={expertDim}). Expert MatMuls write " +
                    "_expertDim rows; an oversized buffer corrupts expert output (issue #315).");
        }
        else if (scratchDim != intermDim)
        {
            throw new InvalidOperationException(
                $"Dense FFN scratch dim {scratchDim} must equal _intermDim {intermDim} (issue #315).");
        }
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || IsRawGpuQuant(tensor.DType))
            return (tensor.ByteSize + 3) & ~3L;

        return tensor.ElementCount * sizeof(float);
    }

    /// <summary>
    /// Weight quantizations uploaded to the GPU as raw blocks (dequantized in-shader by the
    /// matching <c>VulkanBackend.MatMul</c> matvec dispatch) rather than expanded to F32 on
    /// the CPU. Keeping these quantized is the whole point of the GPU matvec shaders.
    /// </summary>
    private static bool IsRawGpuQuant(DType dtype) =>
        dtype is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Q4_0;
}
