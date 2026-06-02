using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// Optimized CPU forward pass for a dense LLaMA-family transformer.
/// Uses AVX2 SIMD, fused dequant-matvec, and multi-threading.
/// </summary>
public sealed unsafe class ForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly PagedKvCache _kvCache;
    private readonly int _ctxLen; // scratch buffer sizing (attnScores, TurboQuant)

    // Norm weight cache: only tiny F32 weights (2048 floats = 8KB each)
    private readonly Dictionary<string, nint> _normCache = new();

    // Preallocated scratch buffers
    private readonly float* _hidden;     // [embDim]
    private readonly float* _residual;   // [embDim]
    private readonly float* _normBuf;    // [embDim]
    private readonly float* _q;          // [numHeads * headDim]
    private readonly float* _k;          // [numKvHeads * headDim]
    private readonly float* _v;          // [numKvHeads * headDim]
    private readonly float* _attnOut;    // [numHeads * headDim]
    private readonly float* _ffnGate;    // [intermDim]
    private readonly float* _ffnUp;      // [intermDim]
    private readonly float* _logits;     // [vocabSize]
    private readonly float* _attnScores; // [numHeads * ctxLen] per-head score scratch

    private readonly int _embDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headsPerKvGroup;
    private readonly int _intermDim;

    // Per-layer head-dim variance (Gemma 4: SWA layers use 256, global layers use 512).
    // _layerHeadDim is non-null only when hp.LayerHeadDim is non-null; otherwise the
    // hot path uses the scalar _headDim. _maxHeadDim sizes scratch + the PagedKvCache
    // so per-layer slots fit; per-layer Attention reads/writes the leading
    // head_dim[layer] of each head, with the trailing bytes left as zeros via an
    // explicit Clear before the Q/K/V matvecs.
    private readonly int[]? _layerHeadDim;
    private readonly int[]? _layerRopeDim;
    private readonly int[]? _layerKvSrc;
    private readonly bool[]? _isSwaLayer;
    private readonly int _maxHeadDim;

    // Precomputed tensor metadata for hot-path access
    private readonly TensorRef _embTensor;
    private readonly TensorRef[] _attnNorm;
    private readonly TensorRef[] _wq, _wk, _wv, _wo;
    private readonly TensorRef[] _ffnNorm;
    private readonly TensorRef[] _wGate, _wUp, _wDown;
    private readonly TensorRef _outputNorm;
    private readonly TensorRef _outputWeight;

    // Optional attention biases (Qwen models)
    private readonly bool _hasAttnBias;
    private readonly float*[] _bq, _bk, _bv, _bo;

    // Optional per-head Q/K RMSNorm (Qwen3-style shared weights of size headDim,
    // or OLMoE-style per-channel weights of size numHeads*headDim / numKvHeads*headDim).
    private readonly bool _hasQkNorm;
    private readonly bool _perChannelQkNorm;
    private readonly float*[] _qNorm, _kNorm;

    // Wide-vector (AVX-512) RmsNorm fast path. Enabled only when the model has
    // per-layer head_dim (Gemma 4) — for other models the byte-parity oracles
    // (e.g. MtpDecoder_GreedyParity_LlamaCpp on Qwen3.6-27B-MTP) are sensitive
    // to the ~ULP reduction-order shift between the AVX2 and AVX-512 sum-of-
    // squares paths. See feedback_qkv_matvecdual_breaks_mtp_parity.
    private readonly bool _useWideNorms;

    // MoE (Mixture of Experts) — Phase 5
    private readonly TensorRef[]? _wGateInp;      // router weights [numExperts, embDim] per layer
    private readonly TensorRef[]? _wGateShexp, _wUpShexp, _wDownShexp; // shared expert per layer
    private readonly TensorRef[]? _wGateExps, _wUpExps, _wDownExps;   // packed expert weights per layer
    private readonly float* _routerLogits;  // [numExperts] scratch
    private readonly float* _sharedOut;     // [embDim] shared expert output
    private readonly float* _expertGate;    // [expertIntermDim] expert gate scratch
    private readonly float* _expertUp;      // [expertIntermDim] expert up scratch
    // Per-expert down-projection scratch — sized embDim because that's the row count
    // of the down MatVec. Most MoE models have intermDim >= embDim so _ffnUp would
    // suffice, but OLMoE has embDim=2048 / intermDim=1024 and overflows it.
    private readonly float* _moeDownTemp;

    // Optional TurboQuant KV cache (Phase 3)
    private TurboQuantKvCache? _tqKvCache;
    private float* _rotatedQuery;  // scratch for WHT-rotated query [headDim]
    private float* _decompBuf;     // scratch for decompressed TQ value [headDim]

    // SnapKV (issue #51) prefill-time KV eviction. Activated via the
    // SHARPI_SNAPKV_BUDGET env var; the selector is lazily allocated on the
    // first prefill long enough to require eviction.
    private readonly SnapKvConfig _snapKvCfg;
    private SnapKvSelector? _snapKv;

    // Diagnostic: per-layer residual L2-norm trace (env: SHARPI_TRACE_NORMS=1).
    private static readonly bool _traceNorms =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_NORMS") == "1";
    private float[]? _normTraceAttn;   // [numLayers] post-attn-residual L2 norm
    private float[]? _normTraceFfn;    // [numLayers] post-ffn-residual L2 norm
    // Optional MoE router probe (env: SHARPI_TRACE_ROUTERS=1 dumps top-k experts for every
    // MoE layer). To restrict to a single position (large MoE models), set SHARPI_TRACE_POS=<n>.
    private static readonly bool _traceRouters =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_ROUTERS") == "1";
    private static readonly int _traceRouterPos = ParseInt("SHARPI_TRACE_POS", -1);
    private static int ParseInt(string env, int def)
    {
        var s = Environment.GetEnvironmentVariable(env);
        return int.TryParse(s, System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : def;
    }
    // Tracks the position of the in-flight forward pass so MoeFfn can decide whether to log.
    private int _currentPos;

    // Precomputed RoPE cos/sin tables [maxSeqLen * halfDim]
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;
    private readonly int _ropeHalfDim;

    // Second RoPE table for Gemma 4 SWA layers (theta = RopeThetaSwa, e.g. 10K).
    // Non-null only when hp.RopeThetaSwa > 0. Halfdim follows the SWA layers' rope dim.
    private readonly float* _ropeCosTableSwa;
    private readonly float* _ropeSinTableSwa;
    private readonly int _ropeHalfDimSwa;

    // Gemma 4 per-layer norms + scale (null/empty on non-Gemma 4 models).
    private readonly TensorRef[]? _postAttnNorm;
    private readonly TensorRef[]? _postFfwNorm;
    private readonly float[]? _layerOutputScale;

    // Gemma 4 Per-Layer-Embedding (PLE) injection. Non-null only when
    // _hp.HasPerLayerTokenEmbd is true. _pleTokenEmbed stays mmap-resident
    // (~3 GB Q8_0). _perLayerModelProj is preloaded BF16→F32 once.
    private readonly TensorRef _pleTokenEmbed;
    private readonly float* _perLayerModelProj;
    private readonly TensorRef _perLayerProjNormTensor;
    private readonly TensorRef[]? _pleInpGate;
    private readonly TensorRef[]? _plePostProj;
    private readonly TensorRef[]? _plePostNorm;
    private readonly int _pleWidth;
    private readonly float* _pleRowBuf;       // [NumLayers * PleWidth] gathered+dequant per token
    private readonly float* _projPerLayer;    // [NumLayers * PleWidth] proj_layer cache per token
    private readonly float* _pleX;            // [PleWidth] inner scratch
    private readonly float* _pleY;            // [embDim] inner scratch

    public ForwardPass(GgufModel model, IComputeBackend backend, ModelHyperparams hp,
        int maxContextLength = 0)
    {
        _model = model;
        _hp = hp;
        // ctxLen only governs scratch buffer sizes; PagedKvCache allocates pages lazily.
        int ctxLen = maxContextLength > 0
            ? Math.Min(maxContextLength, hp.ContextLength)
            : Math.Min(hp.ContextLength, 32768);
        _ctxLen = ctxLen;
        _snapKvCfg = SnapKvConfig.FromEnvironment();

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        // Per-layer head-dim plumbing. Materialise the per-layer arrays once so the
        // hot loop reads from a plain int[] rather than IReadOnlyList<int>.
        if (hp.LayerHeadDim is { } lhd)
        {
            _layerHeadDim = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerHeadDim[i] = lhd[i];
        }
        if (hp.LayerRopeDim is { } lrd)
        {
            _layerRopeDim = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerRopeDim[i] = lrd[i];
        }
        if (hp.KvSourceLayer is { } ksl)
        {
            _layerKvSrc = new int[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _layerKvSrc[i] = ksl[i];
        }
        if (hp.IsSwaLayer is { } swa)
        {
            _isSwaLayer = new bool[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++) _isSwaLayer[i] = swa[i];
        }

        _maxHeadDim = _headDim;
        int maxRopeDim = hp.RopeDim > 0 ? hp.RopeDim : _headDim;
        if (_layerHeadDim is not null)
            for (int i = 0; i < hp.NumLayers; i++)
                if (_layerHeadDim[i] > _maxHeadDim) _maxHeadDim = _layerHeadDim[i];

        // Gemma 4 (per-layer head_dim) gets the wider AVX-512 RmsNorm path because
        // its parity test is internal CPU-vs-CUDA argmax, which is invariant to the
        // ~ULP reduction-order difference. Other architectures stick to AVX2.
        _useWideNorms = _layerHeadDim is not null && Avx512F.IsSupported;
        if (_layerRopeDim is not null)
            for (int i = 0; i < hp.NumLayers; i++)
                if (_layerRopeDim[i] > maxRopeDim) maxRopeDim = _layerRopeDim[i];

        _kvCache = new PagedKvCache(hp.NumLayers, hp.NumKvHeads, _maxHeadDim);

        // Allocate scratch
        _hidden = Alloc(_embDim);
        _residual = Alloc(_embDim);
        _normBuf = Alloc(_embDim);
        _q = Alloc(_numHeads * _maxHeadDim);
        _k = Alloc(_numKvHeads * _maxHeadDim);
        _v = Alloc(_numKvHeads * _maxHeadDim);
        _attnOut = Alloc(_numHeads * _maxHeadDim);
        _ffnGate = Alloc(_intermDim);
        _ffnUp = Alloc(_intermDim);
        _logits = Alloc(hp.VocabSize);
        _attnScores = Alloc(_numHeads * ctxLen);

        if (_traceNorms)
        {
            _normTraceAttn = new float[hp.NumLayers];
            _normTraceFfn = new float[hp.NumLayers];
        }

        // Precompute RoPE cos/sin tables for all positions [0, ctxLen).
        // For Gemma 4 the global table is sized for the largest rope dim across layers;
        // the SWA table is built separately at RopeThetaSwa with its own (smaller) halfDim.
        _ropeHalfDim = maxRopeDim / 2;
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));

        // Gemma 4 stores a top-level `rope_freqs.weight` (size halfDim) that masks
        // the global-layer RoPE frequencies: first 64 pairs = 1.0 (rotate), last
        // 192 = 1e30 (identity). Mirrors llama.cpp gemma4.cpp:191 which passes
        // `rope_freqs` only for non-SWA layers. The SWA table is built unscaled.
        float* globalFreqFactors = null;
        var ropeFreqsInfo = model.FindTensor("rope_freqs.weight");
        float[]? ropeFreqsBuf = null;
        if (ropeFreqsInfo is GgufTensorInfo rfi && rfi.DType == DType.Float32 && rfi.ElementCount == _ropeHalfDim)
        {
            var src = MemoryMarshal.Cast<byte, float>(model.GetTensorData(rfi));
            ropeFreqsBuf = new float[_ropeHalfDim];
            src.Slice(0, _ropeHalfDim).CopyTo(ropeFreqsBuf);
        }
        fixed (float* p = ropeFreqsBuf)
        {
            globalFreqFactors = p;
            SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, maxRopeDim, hp.RopeTheta, globalFreqFactors);
        }

        if (hp.RopeThetaSwa > 0f && _layerRopeDim is not null)
        {
            int swaRopeDim = maxRopeDim;
            for (int i = 0; i < hp.NumLayers; i++)
                if (_isSwaLayer![i]) { swaRopeDim = _layerRopeDim[i]; break; }
            _ropeHalfDimSwa = swaRopeDim / 2;
            _ropeCosTableSwa = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDimSwa * sizeof(float)));
            _ropeSinTableSwa = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDimSwa * sizeof(float)));
            SimdKernels.BuildRopeTable(_ropeCosTableSwa, _ropeSinTableSwa, ctxLen, swaRopeDim, hp.RopeThetaSwa);
        }

        // Pre-resolve all tensor references (avoids dictionary lookups in hot loop)
        _embTensor = ResolveTensor("token_embd.weight");

        int L = hp.NumLayers;
        _attnNorm = new TensorRef[L];
        _wq = new TensorRef[L]; _wk = new TensorRef[L];
        _wv = new TensorRef[L]; _wo = new TensorRef[L];
        _ffnNorm = new TensorRef[L];
        _wGate = new TensorRef[L]; _wUp = new TensorRef[L]; _wDown = new TensorRef[L];

        _hasAttnBias = hp.HasAttnBias;
        _bq = new float*[L]; _bk = new float*[L];
        _bv = new float*[L]; _bo = new float*[L];

        _hasQkNorm = hp.HasQkNorm;
        _perChannelQkNorm = hp.IsPerChannelQkNorm;
        _qNorm = new float*[L]; _kNorm = new float*[L];

        // MoE weight arrays
        if (hp.IsMoE)
        {
            _wGateInp = new TensorRef[L];
            _wGateExps = new TensorRef[L]; _wUpExps = new TensorRef[L]; _wDownExps = new TensorRef[L];
            if (hp.HasSharedExpert)
            {
                _wGateShexp = new TensorRef[L]; _wUpShexp = new TensorRef[L]; _wDownShexp = new TensorRef[L];
            }
            _routerLogits = Alloc(hp.NumExperts);
            _sharedOut = Alloc(_embDim);
            _expertGate = Alloc(hp.ExpertIntermediateDim);
            _expertUp = Alloc(hp.ExpertIntermediateDim);
            _moeDownTemp = Alloc(_embDim);
        }

        if (hp.HasPostAttnNorm) _postAttnNorm = new TensorRef[L];
        if (hp.HasPostFfwNorm) _postFfwNorm = new TensorRef[L];
        if (hp.HasLayerOutputScale) _layerOutputScale = new float[L];

        for (int i = 0; i < L; i++)
        {
            int layerHd = _layerHeadDim?[i] ?? _headDim;
            bool kvShared = _layerKvSrc is not null && _layerKvSrc[i] >= 0;

            _attnNorm[i] = ResolveTensor($"blk.{i}.attn_norm.weight");
            _wq[i] = ResolveTensor($"blk.{i}.attn_q.weight");
            _wo[i] = ResolveTensor($"blk.{i}.attn_output.weight");
            _ffnNorm[i] = ResolveTensor($"blk.{i}.ffn_norm.weight");

            // KV-share layers (Gemma 4 tail) don't carry their own attn_k/attn_v weights —
            // they alias the source layer's K/V pages. Skip the tensor lookup so missing-
            // tensor errors don't fire; the runtime path also skips these projections.
            if (!kvShared)
            {
                _wk[i] = ResolveTensor($"blk.{i}.attn_k.weight");
                _wv[i] = ResolveTensor($"blk.{i}.attn_v.weight");
            }

            if (hp.IsMoE)
            {
                _wGateInp![i] = ResolveTensor($"blk.{i}.ffn_gate_inp.weight");
                _wGateExps![i] = ResolveTensor($"blk.{i}.ffn_gate_exps.weight");
                _wUpExps![i] = ResolveTensor($"blk.{i}.ffn_up_exps.weight");
                _wDownExps![i] = ResolveTensor($"blk.{i}.ffn_down_exps.weight");
                if (hp.HasSharedExpert)
                {
                    _wGateShexp![i] = ResolveTensor($"blk.{i}.ffn_gate_shexp.weight");
                    _wUpShexp![i] = ResolveTensor($"blk.{i}.ffn_up_shexp.weight");
                    _wDownShexp![i] = ResolveTensor($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _wGate[i] = ResolveTensor($"blk.{i}.ffn_gate.weight");
                _wUp[i] = ResolveTensor($"blk.{i}.ffn_up.weight");
                _wDown[i] = ResolveTensor($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _bq[i] = LoadBias($"blk.{i}.attn_q.bias", _numHeads * layerHd);
                if (!kvShared)
                {
                    _bk[i] = LoadBias($"blk.{i}.attn_k.bias", _numKvHeads * layerHd);
                    _bv[i] = LoadBias($"blk.{i}.attn_v.bias", _numKvHeads * layerHd);
                }
                _bo[i] = LoadBias($"blk.{i}.attn_output.bias", _embDim);
            }

            if (_hasQkNorm && !hp.UseL2QkNorm)
            {
                int qNormSize = _perChannelQkNorm ? _numHeads * layerHd : layerHd;
                int kNormSize = _perChannelQkNorm ? _numKvHeads * layerHd : layerHd;
                _qNorm[i] = LoadBias($"blk.{i}.attn_q_norm.weight", qNormSize);
                _kNorm[i] = LoadBias($"blk.{i}.attn_k_norm.weight", kNormSize);
            }

            if (_postAttnNorm is not null)
                _postAttnNorm[i] = ResolveTensor($"blk.{i}.post_attention_norm.weight");
            if (_postFfwNorm is not null)
                _postFfwNorm[i] = ResolveTensor($"blk.{i}.post_ffw_norm.weight");
            if (_layerOutputScale is not null)
                _layerOutputScale[i] = LoadScalarF32($"blk.{i}.layer_output_scale.weight");
        }

        _outputNorm = ResolveTensor("output_norm.weight");
        _outputWeight = model.FindTensor("output.weight") is not null
            ? ResolveTensor("output.weight")
            : _embTensor; // tied embeddings

        if (hp.HasPerLayerTokenEmbd)
        {
            if (model.FindTensor("per_layer_token_embd.weight") is null
                || model.FindTensor("per_layer_model_proj.weight") is null
                || model.FindTensor("per_layer_proj_norm.weight") is null)
            {
                throw new InvalidOperationException(
                    "ModelHyperparams.HasPerLayerTokenEmbd is true but one or more PLE tensors " +
                    "(per_layer_token_embd / per_layer_model_proj / per_layer_proj_norm) are missing.");
            }

            _pleWidth = hp.PerLayerEmbeddingWidth;
            int stackedDim = L * _pleWidth;

            _pleTokenEmbed = ResolveTensor("per_layer_token_embd.weight");

            var projInfo = model.FindTensor("per_layer_model_proj.weight")!.Value;
            var projData = model.GetTensorData(projInfo);
            int projCount = (int)projInfo.ElementCount;
            _perLayerModelProj = Alloc(projCount);
            Dequantize.ToFloat32(projData, new Span<float>(_perLayerModelProj, projCount),
                projInfo.DType, projCount);

            _perLayerProjNormTensor = ResolveTensor("per_layer_proj_norm.weight");

            _pleInpGate = new TensorRef[L];
            _plePostProj = new TensorRef[L];
            _plePostNorm = new TensorRef[L];
            for (int i = 0; i < L; i++)
            {
                _pleInpGate[i] = ResolveTensor($"blk.{i}.inp_gate.weight");
                _plePostProj[i] = ResolveTensor($"blk.{i}.proj.weight");
                _plePostNorm[i] = ResolveTensor($"blk.{i}.post_norm.weight");
            }

            _pleRowBuf = Alloc(stackedDim);
            _projPerLayer = Alloc(stackedDim);
            _pleX = Alloc(_pleWidth);
            _pleY = Alloc(_embDim);
        }

        PrefaultWeights();
    }

    /// <summary>
    /// Touch every 4KB page of all weight tensors to force OS page-in,
    /// eliminating soft page faults during inference.
    /// </summary>
    private void PrefaultWeights()
    {
        var tensors = new List<TensorRef> { _embTensor, _outputNorm, _outputWeight };
        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            bool kvShared = _layerKvSrc is not null && _layerKvSrc[i] >= 0;
            tensors.Add(_attnNorm[i]);
            tensors.Add(_wq[i]); tensors.Add(_wo[i]);
            if (!kvShared) { tensors.Add(_wk[i]); tensors.Add(_wv[i]); }
            tensors.Add(_ffnNorm[i]);
            if (_postAttnNorm is not null) tensors.Add(_postAttnNorm[i]);
            if (_postFfwNorm is not null) tensors.Add(_postFfwNorm[i]);

            if (_hp.IsMoE)
            {
                tensors.Add(_wGateInp![i]);
                tensors.Add(_wGateExps![i]); tensors.Add(_wUpExps![i]); tensors.Add(_wDownExps![i]);
                if (_hp.HasSharedExpert)
                {
                    tensors.Add(_wGateShexp![i]); tensors.Add(_wUpShexp![i]); tensors.Add(_wDownShexp![i]);
                }
            }
            else
            {
                tensors.Add(_wGate[i]); tensors.Add(_wUp[i]); tensors.Add(_wDown[i]);
            }
        }

        long touchSum = 0;
        Parallel.ForEach(tensors, tensor =>
        {
            long size = tensor.Info.ByteSize;
            byte* ptr = tensor.DataPtr;
            long localSum = 0;
            for (long off = 0; off < size; off += 4096)
                localSum += ptr[off];
            if (size > 0)
                localSum += ptr[size - 1];
            Interlocked.Add(ref touchSum, localSum);
        });

        // Prevent dead-code elimination
        if (touchSum == long.MinValue) Console.Write(touchSum);
    }

    public PagedKvCache Cache => _kvCache;

    /// <summary>Vocabulary size of this model.</summary>
    public int VocabSize => _hp.VocabSize;

    /// <summary>Maximum supported sequence length.</summary>
    public int MaxSeqLen => _kvCache.MaxSeqLen;

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// Not supported when TurboQuant is enabled and the target length falls in the compressed range.
    /// </summary>
    public void TruncateTo(int length)
    {
        if (_tqKvCache != null)
            _tqKvCache.TruncateTo(length);
        else
            _kvCache.TruncateTo(length);
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => true;

    public void ResetCache()
    {
        if (_tqKvCache != null)
            _tqKvCache.Reset();
        else
            _kvCache.Reset();
    }

    /// <summary>
    /// Enables TurboQuant KV cache compression. Must be called before any forward pass.
    /// </summary>
    public void EnableTurboQuant(int fp32WindowSize = 256, int bits = 3)
    {
        _tqKvCache = new TurboQuantKvCache(
            _hp.NumLayers, _ctxLen, _numKvHeads, _headDim,
            Math.Min(fp32WindowSize, _ctxLen), bits,
            layerIndexBase: 0, totalLayerCountForSeeds: _hp.NumLayers);
        _rotatedQuery = Alloc(_numHeads * _headDim);
        _decompBuf = Alloc(_numHeads * _headDim);
    }

    /// <summary>The TurboQuant KV cache, if enabled.</summary>
    public TurboQuantKvCache? TqCache => _tqKvCache;

    /// <summary>
    /// Prefill: process all prompt tokens layer-by-layer.
    /// Weights stay hot in L3 cache across tokens within each layer,
    /// amortizing DRAM reads ~N× vs sequential Forward() calls.
    /// Returns logits for the last token.
    /// </summary>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        int N = tokens.Count;
        if (N == 0) throw new ArgumentException("Token list is empty");
        if (N == 1) return Forward(tokens[0], startPos);

        // MoE models: sequential prefill (batched FFN not yet supported for MoE).
        // Per-layer head-dim models (Gemma 4): the batched PrefillCore path assumes a
        // single qDim/kvDim across layers, so fall back to sequential Forward until
        // Phase 8 plumbs per-layer head_dim through the batched paths.
        if (_hp.IsMoE || _layerHeadDim is not null)
        {
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < N; i++)
                logits = Forward(tokens[i], startPos + i);
            return logits;
        }

        // TurboQuant uses a sibling batched path that populates _tqKvCache
        // (with FastScan tile + staging compression along the way) instead of
        // _kvCache. Without this, decode reads from _tqKvCache and only sees
        // the decode-token K/V — the prompt's K/V would land in _kvCache where
        // TqAttention can't reach it.
        if (_tqKvCache != null)
            return PrefillCoreTq(tokens, startPos);

        return PrefillCore(tokens, _kvCache, startPos);
    }

    /// <summary>
    /// Batched prefill core: processes N tokens layer-by-layer into the given cache.
    /// Used by <see cref="Prefill"/> (with _kvCache) and <see cref="PrefillWithCache"/> (with an external cache).
    /// </summary>
    private ReadOnlySpan<float> PrefillCore(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos)
    {
        int N = tokens.Count;

        // SnapKV gating (issue #51): only run eviction when this is a fresh
        // prefill (startPos==0), the budget is configured, AND the prompt is
        // long enough that eviction would actually drop something. On short
        // prompts the scoring cost outweighs the savings.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        // Batch hidden states: [N, embDim]
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            // Temp buffers for batched operations
            int qDim = _numHeads * _headDim;
            int kvDim = _numKvHeads * _headDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                // 2. Process layer-by-layer
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    cache.TruncateTo(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);

                    // Batch RMS norm for all tokens
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    // Batched Q/K/V projections (single GEMM per weight matrix)
                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    // Apply QKV biases per token (Qwen models)
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    // Per-head Q/K RMSNorm and RoPE — ordering and NoPE layers
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    // Per-token: RoPE, KV cache append, attention
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRope(qn, startPos + n, _numHeads);
                            ApplyRope(kn, startPos + n, _numKvHeads);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                        }

                        cache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        cache.IncrementPosition();

                        // Copy Q to scratch for Attention (it reads from _q)
                        Copy(_q, qn, qDim);
                        Attention(cache, layer, startPos + n);

                        // Copy attention output for batched output projection
                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    // SnapKV (issue #51): accumulate per-layer last-W query
                    // attention into the global score buffer. batchQ here is
                    // post-RoPE / post-Q-norm — the same vectors that just
                    // wrote scores against the K cache in the per-token loop
                    // above, so the scoring math is internally consistent.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    // Batched output projection
                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    // Apply output projection bias (Qwen models)
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    // Add output projection + residual → batchHidden
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    // FFN: batch norm, batched gate/up GEMM, per-token SiLU, batched down GEMM
                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    // Per-token SiLU(gate) * up
                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    // Residual add
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // Set KV cache length to startPos + N for subsequent decode calls.
                cache.TruncateTo(startPos + N);

                // SnapKV (issue #51): compact the cache to the selected keep
                // set. Runs once per prefill — the per-token decode path is
                // untouched and pays no extra cost. After compaction
                // cache.Length is the kept-slot count and cache.LogicalLength
                // is the original prompt length, so decode RoPE continues from
                // the right reference frame.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            // 3. Final norm + output projection on last token only
            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            var outNormW = GetNormWeight(_outputNorm);
            SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// TurboQuant variant of <see cref="PrefillCore"/>: identical batched matmul
    /// structure (QKV, attn output, FFN gate/up/down all run as one GEMM per
    /// weight matrix across N tokens), but routes K/V into the TQ cache and
    /// uses TqAttention per token. Between layers the global TQ position
    /// counter snaps back to <paramref name="startPos"/> while per-layer
    /// FastScan tile/staging/FP32 state stays intact — each layer's TQ window
    /// evolves independently as the N tokens stream through it.
    /// </summary>
    private ReadOnlySpan<float> PrefillCoreTq(IReadOnlyList<int> tokens, int startPos)
    {
        var cache = _tqKvCache!;
        int N = tokens.Count;

        // SnapKV (issue #60) gating: fresh prefill, explicit budget, prompt
        // long enough to drop something. TQ is fine to compose with — the
        // selector reads from the same cache the per-token TqAttention writes
        // and the compaction promotes the oldest FP32-window survivors into
        // the TQ region as needed.
        bool snapKvActive = _snapKvCfg.Enabled
                         && startPos == 0
                         && N > _snapKvCfg.Budget
                         && N > _snapKvCfg.Window;
        if (snapKvActive)
        {
            _snapKv ??= new SnapKvSelector(_numHeads, _numKvHeads, _headDim);
            _snapKv.Reset(N);
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            int qDim = _numHeads * _headDim;
            int kvDim = _numKvHeads * _headDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Snap the shared global position counter back to startPos
                    // for this layer's per-token loop. Per-layer TQ tile + FP32
                    // window state from prior layers is untouched.
                    cache.ResetTotalLengthForBatchedPrefill(startPos);
                    var normW = GetNormWeight(_attnNorm[layer]);

                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                            ApplyQkNorm(qn, kn, layer);

                        if (useRoPE)
                        {
                            ApplyRope(qn, startPos + n, _numHeads);
                            ApplyRope(kn, startPos + n, _numKvHeads);
                        }

                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                        }

                        cache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        cache.IncrementPosition();

                        Copy(_q, qn, qDim);
                        TqAttention(layer, startPos + n);

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    // SnapKV (issue #60): same shape as PrefillCore's call but
                    // against the TQ cache. batchQ is post-RoPE / post-Q-norm —
                    // the same vectors TqAttention just used to write scores
                    // against the TQ-compressed + FP32-ring K state.
                    if (snapKvActive)
                    {
                        _snapKv!.AccumulateLayer(batchQ, N, cache, layer, startPos,
                            _snapKvCfg.Window);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // _totalLength was advanced to startPos + N by the last layer's
                // per-token loop, which is the state subsequent decode calls expect.

                // SnapKV (issue #60): compact the TQ cache to the selected keep
                // set. Runs once per prefill — per-token decode is untouched.
                // After compaction Length is the kept-slot count; decode RoPE
                // for the next token continues from `startPos + N` (the caller's
                // position counter is unchanged), which is the right post-eviction
                // reference frame because RoPE depends on absolute position not
                // on cache slot index.
                if (snapKvActive)
                {
                    var keep = _snapKv!.SelectKeepSet(N, _snapKvCfg.Budget, _snapKvCfg.Recency);
                    if (keep.Length < N)
                    {
                        cache.Compact(keep, N);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            float* lastHidden = batchHidden + (long)(N - 1) * _embDim;
            var outNormW = GetNormWeight(_outputNorm);
            SimdKernels.RmsNorm(lastHidden, lastHidden, outNormW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, lastHidden, _hp.VocabSize, _embDim);

            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Batched verification for speculative decoding: processes <paramref name="tokens"/> starting
    /// at <paramref name="startPos"/> using the existing KV cache as context.
    /// All K/V entries are appended to the cache; caller must call TruncateTo to rewind on rejection.
    /// Returns <c>result[i]</c> = logits after processing <c>tokens[i]</c>.
    /// </summary>
    /// <exception cref="NotSupportedException">If TurboQuant KV cache is enabled.</exception>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchVerify is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on the batched BatchVerify path.");

        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();

        if (N == 1 || _hp.IsMoE)
        {
            // Single token or MoE: fall back to sequential Forward calls
            var seq = new float[N][];
            for (int i = 0; i < N; i++)
            {
                var logits = Forward(tokens[i], startPos + i);
                seq[i] = new float[_hp.VocabSize];
                logits.CopyTo(seq[i]);
            }
            return seq;
        }

        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            // 1. Embed all tokens
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);

            int qDim = _numHeads * _headDim;
            int kvDim = _numKvHeads * _headDim;
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));

            try
            {
                // 2. Process layer-by-layer (same batch structure as Prefill, starting at startPos)
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    // Restore cache length to startPos so K/V appends land at the right positions
                    _kvCache.TruncateTo(startPos);

                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }

                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;

                    // Sequential: RoPE (at startPos+n), K/V append, causal attention
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;

                        int pos = startPos + n;

                        // Qwen3 (weighted QK-norm): norm BEFORE RoPE
                        if (_hasQkNorm && !_hp.UseL2QkNorm)
                        {
                            ApplyQkNorm(qn, kn, layer);
                        }

                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }

                        // L2 QK-norm (Llama-4): norm AFTER RoPE, only on RoPE layers
                        if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                        {
                            PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                            PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                        }

                        _kvCache.Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        _kvCache.IncrementPosition();  // _length = startPos + n + 1

                        Copy(_q, qn, qDim);
                        Attention(_kvCache, layer, pos);  // seqLen = startPos + n + 1, uses K/V for 0..pos

                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }

                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);

                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        float* proj = batchNorm + (long)n * _embDim;
                        float* r = batchResidual + (long)n * _embDim;
                        Copy(h, proj, _embDim);
                        SimdKernels.AddInPlace(h, r, _embDim);
                    }

                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }

                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);

                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);

                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);

                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }

                // Ensure cache length is startPos + N
                _kvCache.TruncateTo(startPos);
                for (int i = 0; i < N; i++) _kvCache.IncrementPosition();
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }

            // 3. Final norm + output projection per position
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                float* h = batchHidden + (long)n * _embDim;
                SimdKernels.RmsNorm(h, h, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, h, _hp.VocabSize, _embDim);
                result[n] = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(result[n]);
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }

    /// <summary>
    /// Run one token through the full transformer. Returns logits span.
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        _currentPos = position;

        // 1. Embedding lookup (single-row dequant, no full table materialization)
        EmbedToken(token);

        // Gemma family scales embeddings by sqrt(EmbeddingDim) before the trunk.
        if (_hp.EmbeddingScale != 1f)
            SimdKernels.ScaleInPlace(_hidden, _hp.EmbeddingScale, _embDim);

        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjections(token);

        float embNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            int layerHd = _layerHeadDim?[layer] ?? _headDim;
            int qDimL = _numHeads * layerHd;
            int kvDimL = _numKvHeads * layerHd;
            int kvSrc = _layerKvSrc is not null ? _layerKvSrc[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            int windowSize = _isSwaLayer is not null && _isSwaLayer[layer] ? _hp.SlidingWindowSize : -1;

            // Save residual
            Copy(_residual, _hidden, _embDim);

            // Pre-attention RMS norm
            var normW = GetNormWeight(_attnNorm[layer]);
            FastRmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);

            // Q projection always runs on the active layer's weights.
            // For per-layer head_dim models the trailing bytes of _q/_k/_v are stale
            // from a wider prior layer; zero them so subsequent Attention reads (which
            // stride by the active layerHd) don't pick up garbage on heads beyond
            // numHeads*layerHd, and KV cache pages don't carry inter-layer pollution.
            if (_layerHeadDim is not null)
            {
                int qBytes = _numHeads * _maxHeadDim;
                int kvBytes = _numKvHeads * _maxHeadDim;
                new Span<float>(_q, qBytes).Clear();
                new Span<float>(_k, kvBytes).Clear();
                new Span<float>(_v, kvBytes).Clear();
            }
            FusedMatVec(_q, _wq[layer], _normBuf, qDimL, _embDim);
            if (!kvShared)
            {
                if (_layerHeadDim is not null)
                {
                    // Gemma 4: K and V share row count and dtype — fuse via
                    // MatVecDual so the row loops interleave and the input
                    // vector reads amortize. The row-interleave changes the FP
                    // ordering vs sequential matvecs by ~ULP; gated to the
                    // per-layer head_dim path because cumulative trunk drift
                    // breaks Qwen3.6-27B-MTP byte parity (see
                    // feedback_qkv_matvecdual_breaks_mtp_parity).
                    SimdKernels.MatVecDual(_k, _wk[layer].DataPtr, _v, _wv[layer].DataPtr,
                        _normBuf, kvDimL, _embDim, _wk[layer].DType, _wv[layer].DType);
                }
                else
                {
                    FusedMatVec(_k, _wk[layer], _normBuf, kvDimL, _embDim);
                    FusedMatVec(_v, _wv[layer], _normBuf, kvDimL, _embDim);
                }
            }

            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], qDimL);
                if (!kvShared)
                {
                    SimdKernels.AddInPlace(_k, _bk[layer], kvDimL);
                    SimdKernels.AddInPlace(_v, _bv[layer], kvDimL);
                }
            }

            // NoPE: skip RoPE for NoPE layers
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;

            // Qwen3 (weighted QK-norm): apply norm BEFORE RoPE (per reference implementation)
            // Llama-4 (L2 QK-norm): apply norm AFTER RoPE (per llama.cpp)
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                ApplyQkNormLayer(_q, kvShared ? null : _k, layer, layerHd);
            }

            // Gemma 4: V is plain per-head RmsNorm (no learned weight) before cache.
            // Matches llama.cpp src/models/gemma4.cpp line 227:
            //   Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)
            if (_layerHeadDim is not null && !kvShared)
            {
                PerHeadPureRmsNorm(_v, _numKvHeads, layerHd, _hp.RmsNormEps);
            }

            if (useRoPE)
            {
                ApplyRopeLayer(_q, position, _numHeads, layer, layerHd);
                if (!kvShared)
                    ApplyRopeLayer(_k, position, _numKvHeads, layer, layerHd);
            }

            // L2 QK-norm (Llama-4): only on RoPE layers, applied after RoPE
            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                PerHeadPureRmsNorm(_q, _numHeads, layerHd, _hp.RmsNormEps);
                if (!kvShared)
                    PerHeadPureRmsNorm(_k, _numKvHeads, layerHd, _hp.RmsNormEps);
            }

            // Store K, V in cache. KV-share layers don't append — the source layer's
            // cache slot is shared via effLayer in the Attention call below.
            // PagedKvCache.Append requires exactly cache.KvDim floats; for per-layer
            // head_dim models the trailing (KvDim - kvDimL) floats were just zeroed.
            if (!kvShared)
            {
                int appendLen = _layerHeadDim is not null ? _kvCache.KvDim : kvDimL;
                if (_tqKvCache != null)
                {
                    _tqKvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
                else
                {
                    _kvCache.Append(layer,
                        new ReadOnlySpan<float>(_k, appendLen),
                        new ReadOnlySpan<float>(_v, appendLen));
                }
            }

            // Attention
            if (_tqKvCache != null)
                TqAttention(layer, position);
            else
                Attention(_kvCache, effLayer, layer, position, layerHd, windowSize);

            // Output projection (input width is per-layer qDim).
            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, qDimL);
            if (_hasAttnBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);

            // Gemma 4: post-attention RmsNorm BEFORE the residual add.
            if (_postAttnNorm is not null)
            {
                var paNormW = GetNormWeight(_postAttnNorm[layer]);
                FastRmsNorm(_hidden, _hidden, paNormW, _embDim, _hp.RmsNormEps);
            }

            // Residual
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceNorms) _normTraceAttn![layer] = L2Norm(_hidden, _embDim);

            // Save residual for FFN
            Copy(_residual, _hidden, _embDim);

            // Pre-FFN RMS norm
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            FastRmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);

            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);

            // Gemma 4: post-FFN RmsNorm before the residual add.
            if (_postFfwNorm is not null)
            {
                var pfNormW = GetNormWeight(_postFfwNorm[layer]);
                FastRmsNorm(_hidden, _hidden, pfNormW, _embDim, _hp.RmsNormEps);
            }

            // Residual (post-attn output that includes its own residual).
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_hp.HasPerLayerTokenEmbd)
                ApplyPerLayerEmbedding(layer);

            // Gemma 4: per-layer learned output scale applies AFTER the PLE injection
            // (matches llama.cpp gemma4 build order — applying it before PLE breaks the
            // residual balance and produces unbounded hidden L2 growth).
            if (_layerOutputScale is not null)
                SimdKernels.ScaleInPlace(_hidden, _layerOutputScale[layer], _embDim);

            if (_traceNorms) _normTraceFfn![layer] = L2Norm(_hidden, _embDim);
        }

        // Increment KV cache position
        if (_tqKvCache != null)
            _tqKvCache.IncrementPosition();
        else
            _kvCache.IncrementPosition();

        float preFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 3. Final RMS norm
        var outNormW = GetNormWeight(_outputNorm);
        FastRmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);

        float postFinalNorm = _traceNorms ? L2Norm(_hidden, _embDim) : 0f;

        // 4. Output projection → logits (fused, no 400MB intermediate buffer)
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);

        // Gemma 4 final-logit softcap: x = tanh(x/cap) * cap.
        if (_hp.FinalLogitSoftcap > 0f)
            SimdKernels.SoftcapInPlace(_logits, _hp.VocabSize, _hp.FinalLogitSoftcap);

        if (_traceNorms)
            EmitNormTrace(token, position, embNorm, preFinalNorm, postFinalNorm);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    private void ApplyRope(float* x, int pos, int heads)
    {
        var cos = _ropeCosTable + (long)pos * _ropeHalfDim;
        var sin = _ropeSinTable + (long)pos * _ropeHalfDim;
        if (_hp.IsNeoxRope)
            SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, _headDim);
        else
            SimdKernels.ApplyRoPECached(x, cos, sin, heads, _headDim);
    }

    /// <summary>
    /// Per-layer-aware RoPE: selects the global or SWA cos/sin table for Gemma 4 and
    /// rotates each head's leading <paramref name="layerHd"/> dims.
    /// </summary>
    private void ApplyRopeLayer(float* x, int pos, int heads, int layer, int layerHd)
    {
        bool useSwa = _ropeCosTableSwa != null && _isSwaLayer is not null && _isSwaLayer[layer];
        int halfDim = useSwa ? _ropeHalfDimSwa : _ropeHalfDim;
        float* cosTab = useSwa ? _ropeCosTableSwa : _ropeCosTable;
        float* sinTab = useSwa ? _ropeSinTable    : _ropeSinTable;
        if (useSwa) sinTab = _ropeSinTableSwa;
        var cos = cosTab + (long)pos * halfDim;
        var sin = sinTab + (long)pos * halfDim;
        if (_hp.IsNeoxRope)
            SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, layerHd);
        else
            SimdKernels.ApplyRoPECached(x, cos, sin, heads, layerHd);
    }

    private static float L2Norm(float* x, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++) { double v = x[i]; s += v * v; }
        return (float)Math.Sqrt(s);
    }

    private void EmitNormTrace(int token, int position,
        float embNorm, float preFinalNorm, float postFinalNorm)
    {
        // Top-1 logit + index
        int topIdx = 0; float topVal = float.MinValue;
        for (int i = 0; i < _hp.VocabSize; i++)
            if (_logits[i] > topVal) { topVal = _logits[i]; topIdx = i; }

        var sb = new System.Text.StringBuilder(2048);
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        sb.Append("[norms pos=").Append(position)
          .Append(" tok=").Append(token)
          .Append(" emb=").Append(embNorm.ToString("F2", inv));
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            sb.Append(" L").Append(i).Append(":a=")
              .Append(_normTraceAttn![i].ToString("F1", inv))
              .Append("/f=").Append(_normTraceFfn![i].ToString("F1", inv));
        }
        sb.Append(" preFN=").Append(preFinalNorm.ToString("F2", inv))
          .Append(" postFN=").Append(postFinalNorm.ToString("F2", inv))
          .Append(" top=").Append(topIdx)
          .Append('@').Append(topVal.ToString("F2", inv));
        Console.Error.WriteLine(sb.ToString());
    }

    // ================================================================
    //  Attention
    // ================================================================

    private void Attention(PagedKvCache cache, int layer, int position)
        => Attention(cache, layer, layer, position, _headDim, windowSize: -1);

    /// <summary>
    /// Multi-head attention with optional per-layer head dim, KV-source aliasing, and
    /// sliding-window bound. <paramref name="readLayer"/> is the layer whose K/V pages
    /// to read (== <paramref name="ownLayer"/> for non-shared layers; the source layer
    /// when KV is aliased). <paramref name="windowSize"/> &gt; 0 restricts the score and
    /// V-aggregation loops to the last <paramref name="windowSize"/> positions.
    /// </summary>
    private void Attention(PagedKvCache cache, int readLayer, int ownLayer, int position,
        int hd, int windowSize)
    {
        // After SnapKV eviction (issue #51), the absolute position keeps
        // growing while the cache only stores `cache.Length` slots — `position`
        // would overshoot. The prefill loop increments cache.Length before
        // calling Attention (so position+1 == cache.Length); the decode loop
        // increments after (so position+1 == cache.Length+1). Clamping to
        // cache.Length+1 keeps the old answer for both prefill and the
        // un-evicted decode case while bounding the read to the actually
        // stored slots once eviction has shrunk the cache.
        _ = ownLayer;
        int endSeq = Math.Min(position + 1, cache.Length + 1);
        int startSeq = windowSize > 0 ? Math.Max(0, endSeq - windowSize) : 0;
        // Gemma 4 uses self.scaling = 1.0 (no pre-attention scaling); other archs
        // use 1/sqrt(head_dim). See llama.cpp src/models/gemma4.cpp:11
        //   hparams.f_attention_scale = 1.0f
        float scale = _layerHeadDim is not null ? 1.0f : 1.0f / MathF.Sqrt(hd);
        int ctxLen = _ctxLen; int hpkg = _headsPerKvGroup;
        // Per-layer K/V stride into the cache slot: when LayerHeadDim is active each
        // cache page row is _maxKvDim wide but only kvHead*layerHd is populated.
        int slotStride = _layerHeadDim is not null ? _numKvHeads * _maxHeadDim : _numKvHeads * hd;
        _ = slotStride;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        int rl = readLayer; int hdLocal = hd; int startLocal = startSeq;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hdLocal;
            float* outHead = attnOut + h * hdLocal;
            float* headScores = scores + (long)h * ctxLen;

            int scoreLen = endSeq - startLocal;
            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* kVec = cache.KeyAt(rl, t) + kvHead * hdLocal;
                headScores[i] = SimdKernels.DotF32(qHead, kVec, hdLocal) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, scoreLen);

            for (int d = 0; d < hdLocal; d++) outHead[d] = 0;

            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* vVec = cache.ValueAt(rl, t) + kvHead * hdLocal;
                float w = headScores[i];
                if (Fma.IsSupported && hdLocal >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hdLocal; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    // ================================================================
    //  TurboQuant Attention
    // ================================================================

    private void TqAttention(int layer, int position)
    {
        var tq = _tqKvCache!;
        // After SnapKV (issue #60) eviction the absolute position keeps
        // growing while the TQ cache only stores `tq.Length` slots. The
        // Forward decode path's Append runs before IncrementPosition so the
        // new K/V is at slot tq.Length (and visible via Fp32KeyAt(position=
        // tq.Length)), hence the `+1` — mirrors PagedKvCache.Attention.
        // Pre-eviction position+1 == tq.Length so the clamp is a no-op.
        int seqLen = Math.Min(position + 1, tq.Length + 1);
        int tqLen = tq.GetTqLength(layer);
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        int tqBlkSz = tq.TqBlockSize;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        var rotated = _rotatedQuery; var decomp = _decompBuf;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;
            float* headRotated = rotated + h * hd;
            float* headDecomp = decomp + h * hd;

            var keyCompressor = tq.GetKeyCompressor(layer, kvHead);
            keyCompressor.RotateQuery(
                new ReadOnlySpan<float>(qHead, hd),
                new Span<float>(headRotated, hd));

            // K-scoring via FastScan (issue #34): tile-walks full 32-position
            // tiles through an i8-LUT pshufb kernel and falls back to per-block
            // DequantDot on the <32 staging tail.
            tq.ComputeKScores(layer, kvHead, headRotated, scale, headScores);

            // Phase 1b: FP32 window positions
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            // FastScan V-aggregation (issue #34 Phase 3): tile-walks the
            // TQ-compressed positions with deferred sign-flip + IWHT, then
            // the FP32-window loop below accumulates the recent positions
            // on top in the original domain.
            tq.ComputeVAggregation(layer, kvHead, headScores, outHead);

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(layer, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    // ================================================================
    //  Dense FFN (non-MoE)
    // ================================================================

    private void DenseFfn(int layer)
    {
        SimdKernels.MatVecDual(_ffnGate, _wGate[layer].DataPtr, _ffnUp, _wUp[layer].DataPtr,
            _normBuf, _intermDim, _embDim, _wGate[layer].DType, _wUp[layer].DType);
        if (_hp.FfnActivation == FfnActivation.GeluApprox)
            SimdKernels.GeluTanhMul(_ffnGate, _ffnUp, _ffnGate, _intermDim);
        else
            SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(_hidden, _wDown[layer], _ffnGate, _embDim, _intermDim);
    }

    // ================================================================
    //  MoE FFN (Mixture of Experts)
    // ================================================================

    private void MoeFfn(int layer)
    {
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        // Step 1: Router — compute expert logits and select top-k
        FusedMatVec(_routerLogits, _wGateInp![layer], _normBuf, numExperts, _embDim);

        // Gating: sigmoid for Llama-4, softmax for others
        if (_hp.UseSigmoidGating)
            SimdKernels.SigmoidInPlace(_routerLogits, numExperts);
        else
            SimdKernels.SoftmaxInPlace(_routerLogits, numExperts);

        // Find top-k experts (for k=1, just argmax)
        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        if (_traceRouters && (_traceRouterPos < 0 || _traceRouterPos == _currentPos))
        {
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(512);
            sb.Append("[router pos=").Append(_currentPos).Append(" L").Append(layer).Append(']');
            float wsum = 0;
            for (int i = 0; i < numActive; i++)
            {
                sb.Append(' ').Append(selectedExperts[i]).Append('=')
                  .Append(expertWeights[i].ToString("F4", inv));
                wsum += expertWeights[i];
            }
            sb.Append(" sum=").Append(wsum.ToString("F4", inv));
            Console.Error.WriteLine(sb.ToString());
        }

        // Step 2: Shared expert (runs on every token if present)
        // Shared expert uses the same dim as routed experts (ExpertIntermediateDim)
        if (_hp.HasSharedExpert)
        {
            FusedMatVec(_expertGate, _wGateShexp![layer], _normBuf, expertDim, _embDim);
            FusedMatVec(_expertUp, _wUpShexp![layer], _normBuf, expertDim, _embDim);
            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            FusedMatVec(_sharedOut, _wDownShexp![layer], _expertGate, _embDim, expertDim);
        }

        // Step 3: Selected expert(s) — sparse execution
        // Zero the output accumulator
        new Span<float>(_hidden, _embDim).Clear();

        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            // Expert weights are packed: all experts concatenated in one tensor.
            // Each expert's gate/up is [expertDim, embDim], down is [embDim, expertDim].
            // Expert slice offset in packed tensor: expertIdx * expertDim * (bytes per row)
            ExpertMatVec(_expertGate, _wGateExps![layer], expertIdx, expertDim, _embDim, _normBuf);
            ExpertMatVec(_expertUp, _wUpExps![layer], expertIdx, expertDim, _embDim, _normBuf);

            if (_hp.UseSigmoidGating)
            {
                // Llama-4: apply sigmoid weight before FFN (scale gate/up ≡ scaling input)
                SimdKernels.ScaleInPlace(_expertGate, weight, expertDim);
                SimdKernels.ScaleInPlace(_expertUp, weight, expertDim);
                weight = 1.0f;
            }

            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            ExpertMatVecDown(_hidden, _wDownExps![layer], expertIdx, _embDim, expertDim, _expertGate, weight);
        }

        // Step 4: Add shared expert output
        if (_hp.HasSharedExpert)
            SimdKernels.AddInPlace(_hidden, _sharedOut, _embDim);
    }

    /// <summary>
    /// MatVec for a single expert slice from a packed expert tensor.
    /// The packed tensor has shape [numExperts * rows, cols]. Expert i's slice
    /// starts at row offset (i * rows).
    /// </summary>
    private void ExpertMatVec(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    /// <summary>
    /// MatVec for expert down projection, with weighted accumulation into output.
    /// output += weight * (expertDown[expertIdx] × input)
    /// </summary>
    private void ExpertMatVecDown(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;

        SimdKernels.MatVec(_moeDownTemp, expertData, input, rows, cols, packedTensor.DType);

        SimdKernels.WeightedAddInPlace(output, _moeDownTemp, weight, rows);
    }

    private static void SelectTopK(float* logits, int n, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        // Simple selection for small k (typically 1 or 2)
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool alreadySelected = false;
                for (int j = 0; j < ki; j++)
                    if (indices[j] == i) { alreadySelected = true; break; }
                if (!alreadySelected && logits[i] > bestVal)
                { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }

        // Renormalize selected weights to sum to 1 (Qwen3-MoE / Mixtral convention).
        // OLMoE skips this — its router uses raw post-softmax probabilities, so
        // unused mass on non-selected experts intentionally shrinks the MoE block's
        // contribution to the residual.
        if (normalize && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // ================================================================
    //  Embedding lookup (single-row dequant)
    // ================================================================

    private void EmbedToken(int token) => EmbedTokenInto(token, _hidden);

    // ================================================================
    //  Gemma 4 Per-Layer-Embedding (PLE)
    // ================================================================

    // Token-major layout: per_layer_token_embd shape (PleAll=10752, vocab=262144) stores
    // one row of length PleAll per token (GGUF dim[0] is row width). Gather + dequant
    // the row, then per-layer normalise + add projection + scale.
    private void BuildPerLayerProjections(int token)
    {
        int stackedDim = _hp.NumLayers * _pleWidth;

        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(_pleTokenEmbed.DType))
                        * DTypeInfo.BytesPerBlock(_pleTokenEmbed.DType);
        byte* rowPtr = _pleTokenEmbed.DataPtr + (long)token * bytesPerRow;
        if (_pleTokenEmbed.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, stackedDim)
                .CopyTo(new Span<float>(_pleRowBuf, stackedDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, _pleRowBuf, stackedDim, _pleTokenEmbed.DType);
        }

        // Gemma scales every embedding table by sqrt(its hidden dim). The PLE table's
        // hidden dim is PleWidth (256 → 16×), matching the trunk's sqrt(EmbeddingDim)
        // scale on token_embd.
        float pleScale = MathF.Sqrt(_pleWidth);
        SimdKernels.ScaleInPlace(_pleRowBuf, pleScale, stackedDim);

        SimdKernels.MatVec(_projPerLayer, (byte*)_perLayerModelProj,
            _hidden, stackedDim, _embDim, DType.Float32);

        float embScale = 1.0f / MathF.Sqrt(_embDim);
        SimdKernels.ScaleInPlace(_projPerLayer, embScale, stackedDim);

        float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);
        var projNormW = GetNormWeight(_perLayerProjNormTensor);
        for (int L = 0; L < _hp.NumLayers; L++)
        {
            float* slice = _projPerLayer + (long)L * _pleWidth;
            FastRmsNorm(slice, slice, projNormW, _pleWidth, _hp.RmsNormEps);
            SimdKernels.AddInPlace(slice, _pleRowBuf + (long)L * _pleWidth, _pleWidth);
            SimdKernels.ScaleInPlace(slice, invSqrt2, _pleWidth);
        }

    }

    private void ApplyPerLayerEmbedding(int layer)
    {
        float* slice = _projPerLayer + (long)layer * _pleWidth;
        FusedMatVec(_pleX, _pleInpGate![layer], _hidden, _pleWidth, _embDim);
        SimdKernels.GeluTanhMul(_pleX, slice, _pleX, _pleWidth);
        FusedMatVec(_pleY, _plePostProj![layer], _pleX, _embDim, _pleWidth);
        var postW = GetNormWeight(_plePostNorm![layer]);
        FastRmsNorm(_pleY, _pleY, postW, _embDim, _hp.RmsNormEps);
        SimdKernels.AddInPlace(_hidden, _pleY, _embDim);
    }

    private void EmbedTokenInto(int token, float* dest)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                        * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* rowPtr = _embTensor.DataPtr + (long)token * bytesPerRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dest, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, dest, _embDim, _embTensor.DType);
        }
    }

    // ================================================================
    //  Fused quantized MatVec (no intermediate F32 weight buffer)
    // ================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FusedMatVec(float* output, in TensorRef tensor, float* input, int rows, int cols)
    {
        SimdKernels.MatVec(output, tensor.DataPtr, input, rows, cols, tensor.DType);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastRmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.RmsNormWide(output, input, weight, size, eps);
        else               SimdKernels.RmsNorm    (output, input, weight, size, eps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void FastPureRmsNorm(float* output, float* input, int size, float eps)
    {
        if (_useWideNorms) SimdKernels.PureRmsNormWide(output, input, size, eps);
        else               SimdKernels.PureRmsNorm    (output, input, size, eps);
    }

    // ================================================================
    //  Norm weight cache (tiny F32 weights, cached permanently)
    // ================================================================

    private float* GetNormWeight(in TensorRef tensor)
    {
        if (_normCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;

        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);

        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);

        // GGUF converter for gemma family already bakes the HF "(1 + w)" RMSNorm
        // convention; we multiply by stored `w` directly (mirrors llama.cpp build_norm
        // in src/llama-graph.cpp). Verified vs actual GGUF: attn_norm ~8, attn_q_norm
        // ~0.98 — already final multipliers.

        _normCache[tensor.Name] = (nint)buf;
        return buf;
    }

    // ================================================================
    //  Tensor resolution
    // ================================================================

    private TensorRef ResolveTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new TensorRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    private float LoadScalarF32(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);
        float[] buf = new float[1];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, 1).CopyTo(buf);
        else
            Dequantize.ToFloat32(data, buf.AsSpan(), info.DType, 1);
        return buf[0];
    }

    private float* LoadBias(string name, int count)
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

    private readonly unsafe struct TensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public TensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name; Info = info; DType = dtype; DataPtr = dataPtr;
        }
    }

    // ================================================================
    //  Utilities
    // ================================================================

    /// <summary>
    /// Apply RMSNorm independently to each head-sized chunk.
    /// weight has [headDim] elements and is shared across all heads.
    /// </summary>
    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    private static void PerChannelRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight + h * headDim, headDim, eps);
    }

    private void ApplyQkNorm(float* q, float* k, int layer)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerChannelRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads,   _headDim, _hp.RmsNormEps);
            PerHeadRmsNorm(k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
        }
    }

    /// <summary>
    /// Per-layer-head-dim QK-norm. <paramref name="k"/> may be null on KV-share layers
    /// where the K projection didn't run (the source layer already normed its own K).
    /// </summary>
    private void ApplyQkNormLayer(float* q, float* k, int layer, int layerHd)
    {
        if (_perChannelQkNorm)
        {
            PerChannelRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerChannelRmsNorm(k, _kNorm[layer], _numKvHeads, layerHd, _hp.RmsNormEps);
        }
        else
        {
            PerHeadRmsNorm(q, _qNorm[layer], _numHeads, layerHd, _hp.RmsNormEps);
            if (k != null)
                PerHeadRmsNorm(k, _kNorm[layer], _numKvHeads, layerHd, _hp.RmsNormEps);
        }
    }

    private static void PerHeadPureRmsNorm(float* data, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.PureRmsNorm(data + h * headDim, data + h * headDim, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));

    private static void Copy(float* dst, float* src, int size) =>
        new ReadOnlySpan<float>(src, size).CopyTo(new Span<float>(dst, size));

    // ================================================================
    //  Continuous Batching API
    // ================================================================

    /// <summary>
    /// Creates a new empty <see cref="PagedKvCache"/> compatible with this model's layer/head dimensions.
    /// Used by <see cref="ContinuousBatchingEngine"/> to allocate per-sequence caches.
    /// </summary>
    public PagedKvCache CreateCache() =>
        new PagedKvCache(_hp.NumLayers, _hp.NumKvHeads, _headDim);

    /// <summary>
    /// Forward pass for a single token using the provided explicit cache (no TurboQuant).
    /// Used by <see cref="PrefillWithCache"/> for single-token prompts and MoE sequential prefill.
    /// </summary>
    private ReadOnlySpan<float> ForwardCore(int token, int pos, PagedKvCache cache)
    {
        EmbedToken(token);
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            Copy(_residual, _hidden, _embDim);
            var normW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);
            FusedMatVec(_q, _wq[layer], _normBuf, _numHeads * _headDim, _embDim);
            FusedMatVec(_k, _wk[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            FusedMatVec(_v, _wv[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], _numHeads * _headDim);
                SimdKernels.AddInPlace(_k, _bk[layer], _numKvHeads * _headDim);
                SimdKernels.AddInPlace(_v, _bv[layer], _numKvHeads * _headDim);
            }
            {
                bool useRoPE = _hp.NoRopeLayerStep == 0
                    || (layer + 1) % _hp.NoRopeLayerStep != 0;
                if (_hasQkNorm && !_hp.UseL2QkNorm)
                {
                    ApplyQkNorm(_q, _k, layer);
                }
                if (useRoPE)
                {
                    ApplyRope(_q, pos, _numHeads);
                    ApplyRope(_k, pos, _numKvHeads);
                }
                if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                {
                    PerHeadPureRmsNorm(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    PerHeadPureRmsNorm(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }
            cache.Append(layer,
                new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));
            Attention(cache, layer, pos);
            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, _numHeads * _headDim);
            if (_hasAttnBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
            Copy(_residual, _hidden, _embDim);
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);
            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
        }
        cache.IncrementPosition();
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);
        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    /// <summary>
    /// Prefill prompt tokens into an explicitly provided KV cache instead of the engine's primary cache.
    /// Used by <see cref="ContinuousBatchingEngine"/> to prefill per-sequence caches concurrently.
    /// Not supported when TurboQuant KV cache is enabled.
    /// </summary>
    /// <param name="tokens">Prompt token IDs to process.</param>
    /// <param name="cache">The KV cache to write into.</param>
    /// <param name="startPos">Starting position in the cache (default 0).</param>
    /// <returns>Logits for the last token.</returns>
    public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos = 0)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("PrefillWithCache is not supported when TurboQuant KV cache is enabled.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on PrefillWithCache.");
        int N = tokens.Count;
        if (N == 0) throw new ArgumentException("Token list is empty", nameof(tokens));
        if (N == 1 || _hp.IsMoE)
        {
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < N; i++)
                logits = ForwardCore(tokens[i], startPos + i, cache);
            return logits;
        }
        return PrefillCore(tokens, cache, startPos);
    }

    /// <summary>
    /// Batched decode step for N sequences simultaneously: one token per sequence, each with its own
    /// KV cache at the given position. Amortizes weight reads N× across concurrent users.
    /// Not supported when TurboQuant KV cache is enabled or for MoE models.
    /// </summary>
    /// <param name="tokens">Next token for each sequence (length N).</param>
    /// <param name="positions">Current decode position for each sequence (= cache.Length before this call).</param>
    /// <param name="caches">Per-sequence KV cache (length N).</param>
    /// <returns>Logits array for each sequence (length N × VocabSize).</returns>
    public float[][] BatchForwardMulti(int[] tokens, int[] positions, PagedKvCache[] caches)
    {
        if (_tqKvCache != null)
            throw new NotSupportedException("BatchForwardMulti is not supported when TurboQuant KV cache is enabled.");
        if (_hp.IsMoE)
            throw new NotSupportedException("BatchForwardMulti is not supported for MoE models; use individual ForwardCore calls.");
        if (_layerHeadDim is not null)
            throw new NotSupportedException(
                "gemma4 per-layer head_dim not yet supported on BatchForwardMulti.");
        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        var batchHidden = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        var batchResidual = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
        try
        {
            for (int n = 0; n < N; n++)
                EmbedTokenInto(tokens[n], batchHidden + (long)n * _embDim);
            var batchNorm = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _embDim * sizeof(float)));
            var batchQ = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchK = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchV = (float*)NativeMemory.AllocZeroed((nuint)((long)N * kvDim * sizeof(float)));
            var batchAttnOut = (float*)NativeMemory.AllocZeroed((nuint)((long)N * qDim * sizeof(float)));
            var batchFfnGate = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            var batchFfnUp = (float*)NativeMemory.AllocZeroed((nuint)((long)N * _intermDim * sizeof(float)));
            try
            {
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    var normW = GetNormWeight(_attnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, normW, _embDim, _hp.RmsNormEps);
                    }
                    SimdKernels.MatMulBatched(batchQ, _wq[layer].DataPtr, batchNorm,
                        N, qDim, _embDim, _wq[layer].DType);
                    SimdKernels.MatMulBatched(batchK, _wk[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wk[layer].DType);
                    SimdKernels.MatMulBatched(batchV, _wv[layer].DataPtr, batchNorm,
                        N, kvDim, _embDim, _wv[layer].DType);
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                        {
                            SimdKernels.AddInPlace(batchQ + (long)n * qDim, _bq[layer], qDim);
                            SimdKernels.AddInPlace(batchK + (long)n * kvDim, _bk[layer], kvDim);
                            SimdKernels.AddInPlace(batchV + (long)n * kvDim, _bv[layer], kvDim);
                        }
                    }
                    bool useRoPE = _hp.NoRopeLayerStep == 0
                        || (layer + 1) % _hp.NoRopeLayerStep != 0;
                    // Per-sequence: RoPE, KV append to individual cache, causal attention
                    for (int n = 0; n < N; n++)
                    {
                        float* qn = batchQ + (long)n * qDim;
                        float* kn = batchK + (long)n * kvDim;
                        float* vn = batchV + (long)n * kvDim;
                        int pos = positions[n];
                        // Soft-reset this layer's position so the Append lands at pos
                        caches[n].TruncateTo(pos);
                        if (useRoPE)
                        {
                            ApplyRope(qn, pos, _numHeads);
                            ApplyRope(kn, pos, _numKvHeads);
                        }
                        if (_hasQkNorm)
                        {
                            if (_hp.UseL2QkNorm)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
                            else
                            {
                                ApplyQkNorm(qn, kn, layer);
                            }
                        }
                        caches[n].Append(layer,
                            new ReadOnlySpan<float>(kn, kvDim),
                            new ReadOnlySpan<float>(vn, kvDim));
                        caches[n].IncrementPosition(); // _length = pos+1
                        Copy(_q, qn, qDim);
                        Attention(caches[n], layer, pos);
                        Copy(batchAttnOut + (long)n * qDim, _attnOut, qDim);
                    }
                    SimdKernels.MatMulBatched(batchNorm, _wo[layer].DataPtr, batchAttnOut,
                        N, _embDim, qDim, _wo[layer].DType);
                    if (_hasAttnBias)
                    {
                        for (int n = 0; n < N; n++)
                            SimdKernels.AddInPlace(batchNorm + (long)n * _embDim, _bo[layer], _embDim);
                    }
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                    var ffnNormW = GetNormWeight(_ffnNorm[layer]);
                    for (int n = 0; n < N; n++)
                    {
                        Copy(batchResidual + (long)n * _embDim, batchHidden + (long)n * _embDim, _embDim);
                        SimdKernels.RmsNorm(batchNorm + (long)n * _embDim,
                            batchHidden + (long)n * _embDim, ffnNormW, _embDim, _hp.RmsNormEps);
                    }
                    SimdKernels.MatMulBatched(batchFfnGate, _wGate[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wGate[layer].DType);
                    SimdKernels.MatMulBatched(batchFfnUp, _wUp[layer].DataPtr, batchNorm,
                        N, _intermDim, _embDim, _wUp[layer].DType);
                    for (int n = 0; n < N; n++)
                        SimdKernels.SiLuMul(batchFfnGate + (long)n * _intermDim,
                            batchFfnUp + (long)n * _intermDim, _intermDim);
                    SimdKernels.MatMulBatched(batchNorm, _wDown[layer].DataPtr, batchFfnGate,
                        N, _embDim, _intermDim, _wDown[layer].DType);
                    for (int n = 0; n < N; n++)
                    {
                        float* h = batchHidden + (long)n * _embDim;
                        Copy(h, batchNorm + (long)n * _embDim, _embDim);
                        SimdKernels.AddInPlace(h, batchResidual + (long)n * _embDim, _embDim);
                    }
                }
            }
            finally
            {
                NativeMemory.Free(batchNorm);
                NativeMemory.Free(batchQ);
                NativeMemory.Free(batchK);
                NativeMemory.Free(batchV);
                NativeMemory.Free(batchAttnOut);
                NativeMemory.Free(batchFfnGate);
                NativeMemory.Free(batchFfnUp);
            }
            var outNormW = GetNormWeight(_outputNorm);
            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                float* h = batchHidden + (long)n * _embDim;
                SimdKernels.RmsNorm(h, h, outNormW, _embDim, _hp.RmsNormEps);
                FusedMatVec(_logits, _outputWeight, h, _hp.VocabSize, _embDim);
                result[n] = new float[_hp.VocabSize];
                new ReadOnlySpan<float>(_logits, _hp.VocabSize).CopyTo(result[n]);
            }
            return result;
        }
        finally
        {
            NativeMemory.Free(batchHidden);
            NativeMemory.Free(batchResidual);
        }
    }


    public void Dispose()
    {
        NativeMemory.Free(_hidden);
        NativeMemory.Free(_residual);
        NativeMemory.Free(_normBuf);
        NativeMemory.Free(_q);
        NativeMemory.Free(_k);
        NativeMemory.Free(_v);
        NativeMemory.Free(_attnOut);
        NativeMemory.Free(_ffnGate);
        NativeMemory.Free(_ffnUp);
        NativeMemory.Free(_logits);
        NativeMemory.Free(_attnScores);
        NativeMemory.Free(_ropeCosTable);
        NativeMemory.Free(_ropeSinTable);
        if (_ropeCosTableSwa != null) NativeMemory.Free(_ropeCosTableSwa);
        if (_ropeSinTableSwa != null) NativeMemory.Free(_ropeSinTableSwa);

        foreach (var ptr in _normCache.Values)
            NativeMemory.Free((void*)ptr);
        _normCache.Clear();

        if (_hasAttnBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_bq[i] != null) NativeMemory.Free(_bq[i]);
                if (_bk[i] != null) NativeMemory.Free(_bk[i]);
                if (_bv[i] != null) NativeMemory.Free(_bv[i]);
                if (_bo[i] != null) NativeMemory.Free(_bo[i]);
            }
        }

        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                NativeMemory.Free(_qNorm[i]);
                NativeMemory.Free(_kNorm[i]);
            }
        }

        if (_hp.IsMoE)
        {
            NativeMemory.Free(_routerLogits);
            NativeMemory.Free(_sharedOut);
            NativeMemory.Free(_expertGate);
            NativeMemory.Free(_expertUp);
            NativeMemory.Free(_moeDownTemp);
        }

        if (_hp.HasPerLayerTokenEmbd)
        {
            NativeMemory.Free(_perLayerModelProj);
            NativeMemory.Free(_pleRowBuf);
            NativeMemory.Free(_projPerLayer);
            NativeMemory.Free(_pleX);
            NativeMemory.Free(_pleY);
        }

        _kvCache.Dispose();
    }
}
