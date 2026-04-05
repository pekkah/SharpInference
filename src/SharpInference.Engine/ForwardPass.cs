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

    // Optional per-head Q/K RMSNorm (Qwen3)
    private readonly bool _hasQkNorm;
    private readonly float*[] _qNorm, _kNorm;

    // MoE (Mixture of Experts) — Phase 5
    private readonly TensorRef[]? _wGateInp;      // router weights [numExperts, embDim] per layer
    private readonly TensorRef[]? _wGateShexp, _wUpShexp, _wDownShexp; // shared expert per layer
    private readonly TensorRef[]? _wGateExps, _wUpExps, _wDownExps;   // packed expert weights per layer
    private readonly float* _routerLogits;  // [numExperts] scratch
    private readonly float* _sharedOut;     // [embDim] shared expert output
    private readonly float* _expertGate;    // [expertIntermDim] expert gate scratch
    private readonly float* _expertUp;      // [expertIntermDim] expert up scratch

    // Optional TurboQuant KV cache (Phase 3)
    private TurboQuantKvCache? _tqKvCache;
    private float* _rotatedQuery;  // scratch for WHT-rotated query [headDim]
    private float* _decompBuf;     // scratch for decompressed TQ value [headDim]

    // Precomputed RoPE cos/sin tables [maxSeqLen * halfDim]
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;
    private readonly int _ropeHalfDim;

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
        _kvCache = new PagedKvCache(hp.NumLayers, hp.NumKvHeads, hp.EmbeddingDim / hp.NumHeads);

        _embDim = hp.EmbeddingDim;
        _headDim = hp.EmbeddingDim / hp.NumHeads;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        // Allocate scratch
        _hidden = Alloc(_embDim);
        _residual = Alloc(_embDim);
        _normBuf = Alloc(_embDim);
        _q = Alloc(_numHeads * _headDim);
        _k = Alloc(_numKvHeads * _headDim);
        _v = Alloc(_numKvHeads * _headDim);
        _attnOut = Alloc(_numHeads * _headDim);
        _ffnGate = Alloc(_intermDim);
        _ffnUp = Alloc(_intermDim);
        _logits = Alloc(hp.VocabSize);
        _attnScores = Alloc(_numHeads * ctxLen);

        // Precompute RoPE cos/sin tables for all positions [0, ctxLen)
        _ropeHalfDim = _headDim / 2;
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, _headDim, hp.RopeTheta);

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
        }

        for (int i = 0; i < L; i++)
        {
            _attnNorm[i] = ResolveTensor($"blk.{i}.attn_norm.weight");
            _wq[i] = ResolveTensor($"blk.{i}.attn_q.weight");
            _wk[i] = ResolveTensor($"blk.{i}.attn_k.weight");
            _wv[i] = ResolveTensor($"blk.{i}.attn_v.weight");
            _wo[i] = ResolveTensor($"blk.{i}.attn_output.weight");
            _ffnNorm[i] = ResolveTensor($"blk.{i}.ffn_norm.weight");

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
                _bq[i] = LoadBias($"blk.{i}.attn_q.bias", _numHeads * _headDim);
                _bk[i] = LoadBias($"blk.{i}.attn_k.bias", _numKvHeads * _headDim);
                _bv[i] = LoadBias($"blk.{i}.attn_v.bias", _numKvHeads * _headDim);
                _bo[i] = LoadBias($"blk.{i}.attn_output.bias", _embDim);
            }

            if (_hasQkNorm && !hp.UseL2QkNorm)
            {
                _qNorm[i] = LoadBias($"blk.{i}.attn_q_norm.weight", _headDim);
                _kNorm[i] = LoadBias($"blk.{i}.attn_k_norm.weight", _headDim);
            }
        }

        _outputNorm = ResolveTensor("output_norm.weight");
        _outputWeight = model.FindTensor("output.weight") is not null
            ? ResolveTensor("output.weight")
            : _embTensor; // tied embeddings

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
            tensors.Add(_attnNorm[i]);
            tensors.Add(_wq[i]); tensors.Add(_wk[i]);
            tensors.Add(_wv[i]); tensors.Add(_wo[i]);
            tensors.Add(_ffnNorm[i]);

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

        // MoE models: sequential prefill (batched FFN not yet supported for MoE)
        if (_hp.IsMoE)
        {
            ReadOnlySpan<float> logits = default;
            for (int i = 0; i < N; i++)
                logits = Forward(tokens[i], startPos + i);
            return logits;
        }

        return PrefillCore(tokens, _kvCache, startPos);
    }

    /// <summary>
    /// Batched prefill core: processes N tokens layer-by-layer into the given cache.
    /// Used by <see cref="Prefill"/> (with _kvCache) and <see cref="PrefillWithCache"/> (with an external cache).
    /// </summary>
    private ReadOnlySpan<float> PrefillCore(IReadOnlyList<int> tokens, PagedKvCache cache, int startPos)
    {
        int N = tokens.Count;
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

                        if (useRoPE)
                        {
                            SimdKernels.ApplyRoPECached(qn, _ropeCosTable + (long)(startPos + n) * _ropeHalfDim, _ropeSinTable + (long)(startPos + n) * _ropeHalfDim, _numHeads, _headDim);
                            SimdKernels.ApplyRoPECached(kn, _ropeCosTable + (long)(startPos + n) * _ropeHalfDim, _ropeSinTable + (long)(startPos + n) * _ropeHalfDim, _numKvHeads, _headDim);
                        }

                        // QK-norm: for L2 (Llama-4), only on RoPE layers (matches llama.cpp)
                        // For weighted QK-norm (Qwen3), applied to all layers
                        if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
                        {
                            if (_hp.UseL2QkNorm)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
                            else
                            {
                                PerHeadRmsNorm(qn, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadRmsNorm(kn, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
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

                        if (useRoPE)
                        {
                            SimdKernels.ApplyRoPECached(qn, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numHeads, _headDim);
                            SimdKernels.ApplyRoPECached(kn, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numKvHeads, _headDim);
                        }

                        if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
                        {
                            if (_hp.UseL2QkNorm)
                            {
                                PerHeadPureRmsNorm(qn, _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadPureRmsNorm(kn, _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
                            else
                            {
                                PerHeadRmsNorm(qn, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadRmsNorm(kn, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                            }
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
        // 1. Embedding lookup (single-row dequant, no full table materialization)
        EmbedToken(token);

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Save residual
            Copy(_residual, _hidden, _embDim);

            // Pre-attention RMS norm
            var normW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, normW, _embDim, _hp.RmsNormEps);

            // Q/K/V projections (fused dequant-matvec, no intermediate F32 buffer)
            FusedMatVec(_q, _wq[layer], _normBuf, _numHeads * _headDim, _embDim);
            FusedMatVec(_k, _wk[layer], _normBuf, _numKvHeads * _headDim, _embDim);
            FusedMatVec(_v, _wv[layer], _normBuf, _numKvHeads * _headDim, _embDim);

            if (_hasAttnBias)
            {
                SimdKernels.AddInPlace(_q, _bq[layer], _numHeads * _headDim);
                SimdKernels.AddInPlace(_k, _bk[layer], _numKvHeads * _headDim);
                SimdKernels.AddInPlace(_v, _bv[layer], _numKvHeads * _headDim);
            }

            // NoPE: skip RoPE for NoPE layers
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;

            if (useRoPE)
            {
                // RoPE
                SimdKernels.ApplyRoPECached(_q, _ropeCosTable + (long)position * _ropeHalfDim, _ropeSinTable + (long)position * _ropeHalfDim, _numHeads, _headDim);
                SimdKernels.ApplyRoPECached(_k, _ropeCosTable + (long)position * _ropeHalfDim, _ropeSinTable + (long)position * _ropeHalfDim, _numKvHeads, _headDim);
            }

            // QK-norm: for L2 (Llama-4), only on RoPE layers per llama.cpp
            if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
            {
                if (_hp.UseL2QkNorm)
                {
                    PerHeadPureRmsNorm(_q, _numHeads, _headDim, _hp.RmsNormEps);
                    PerHeadPureRmsNorm(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                }
                else
                {
                    PerHeadRmsNorm(_q, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
                    PerHeadRmsNorm(_k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }

            // Store K, V in cache
            if (_tqKvCache != null)
            {
                _tqKvCache.Append(layer,
                    new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                    new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));
            }
            else
            {
                _kvCache.Append(layer,
                    new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                    new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));
            }

            // Attention
            if (_tqKvCache != null)
                TqAttention(layer, position);
            else
                Attention(_kvCache, layer, position);

            // Output projection
            FusedMatVec(_hidden, _wo[layer], _attnOut, _embDim, _numHeads * _headDim);
            if (_hasAttnBias)
                SimdKernels.AddInPlace(_hidden, _bo[layer], _embDim);

            // Residual
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            // Save residual for FFN
            Copy(_residual, _hidden, _embDim);

            // Pre-FFN RMS norm
            var ffnNormW = GetNormWeight(_ffnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, ffnNormW, _embDim, _hp.RmsNormEps);

            if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);

            // Residual
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);
        }

        // Increment KV cache position
        if (_tqKvCache != null)
            _tqKvCache.IncrementPosition();
        else
            _kvCache.IncrementPosition();

        // 3. Final RMS norm
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);

        // 4. Output projection → logits (fused, no 400MB intermediate buffer)
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    // ================================================================
    //  Attention
    // ================================================================

    private void Attention(PagedKvCache cache, int layer, int position)
    {
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = cache.KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = cache.ValueAt(layer, t) + kvHead * hd;
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
    //  TurboQuant Attention
    // ================================================================

    private void TqAttention(int layer, int position)
    {
        var tq = _tqKvCache!;
        int seqLen = position + 1;
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

            // Phase 1a: TQ-compressed positions
            for (int t = 0; t < tqLen; t++)
            {
                byte* tqKey = tq.TqKeyAt(layer, t, kvHead);
                float dot = keyCompressor.DequantDot(
                    new ReadOnlySpan<byte>(tqKey, tqBlkSz),
                    new ReadOnlySpan<float>(headRotated, hd));
                headScores[t] = dot * scale;
            }

            // Phase 1b: FP32 window positions
            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            var valCompressor = tq.GetValueCompressor(layer, kvHead);
            for (int t = 0; t < tqLen; t++)
            {
                byte* tqVal = tq.TqValueAt(layer, t, kvHead);
                valCompressor.Decompress(
                    new ReadOnlySpan<byte>(tqVal, tqBlkSz),
                    new Span<float>(headDecomp, hd));
                float w = headScores[t];
                for (int d = 0; d < hd; d++)
                    outHead[d] += w * headDecomp[d];
            }

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
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights);

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

        // Compute into a temp buffer, then weighted-add to output
        float* temp = _ffnUp; // reuse buffer (embDim is always <= intermDim)
        SimdKernels.MatVec(temp, expertData, input, rows, cols, packedTensor.DType);

        // Weighted accumulate: output += weight * temp (SIMD)
        SimdKernels.WeightedAddInPlace(output, temp, weight, rows);
    }

    private static void SelectTopK(float* logits, int n, int k,
        Span<int> indices, Span<float> weights)
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

        // Renormalize weights for selected experts
        if (k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
        // For k=1, weight is the softmax probability (already normalized)
    }

    // ================================================================
    //  Embedding lookup (single-row dequant)
    // ================================================================

    private void EmbedToken(int token) => EmbedTokenInto(token, _hidden);

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
                if (useRoPE)
                {
                    SimdKernels.ApplyRoPECached(_q, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numHeads, _headDim);
                    SimdKernels.ApplyRoPECached(_k, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numKvHeads, _headDim);
                }
                if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
                {
                    if (_hp.UseL2QkNorm)
                    {
                        PerHeadPureRmsNorm(_q, _numHeads, _headDim, _hp.RmsNormEps);
                        PerHeadPureRmsNorm(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
                    }
                    else
                    {
                        PerHeadRmsNorm(_q, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
                        PerHeadRmsNorm(_k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
                    }
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
                            SimdKernels.ApplyRoPECached(qn, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numHeads, _headDim);
                            SimdKernels.ApplyRoPECached(kn, _ropeCosTable + (long)pos * _ropeHalfDim, _ropeSinTable + (long)pos * _ropeHalfDim, _numKvHeads, _headDim);
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
                                PerHeadRmsNorm(qn, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
                                PerHeadRmsNorm(kn, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);
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

        foreach (var ptr in _normCache.Values)
            NativeMemory.Free((void*)ptr);
        _normCache.Clear();

        if (_hasAttnBias)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                NativeMemory.Free(_bq[i]);
                NativeMemory.Free(_bk[i]);
                NativeMemory.Free(_bv[i]);
                NativeMemory.Free(_bo[i]);
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

        _kvCache.Dispose();
    }
}
