using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Engine;

/// <summary>
/// CPU implementation of the DSpark draft head (deepseek-ai/DeepSpec,
/// docs/dspark-plan.md / PR #413): a <see cref="DSparkConfig.NumLayers"/>-layer
/// qwen3-style backbone whose per-layer attention context K/V is projected from
/// the TARGET model's tapped hidden states — <c>fused = RMSNorm_hidden_norm(fc(taps))</c>,
/// the same fused vector feeding every layer's k/v_proj — plus the draft block's
/// own K/V. Block queries attend over ALL context positions and the whole block
/// bidirectionally (no causal mask; reference `create_dspark_attention_mask`).
/// Base logits for all <see cref="BlockSize"/> positions come from one parallel
/// pass; the rank-256 Markov head then re-biases each position sequentially on
/// the previously drafted token, and the confidence head scores
/// <c>[hidden ‖ markov_w1[prev]]</c> per position.
///
/// Weight storage: everything is converted BF16→F32 into native buffers except
/// the two row-gathered [vocab, *] tables (embed_tokens, markov_w1), which stay
/// in their storage dtype and dequantize per row — they are only ever read a
/// handful of rows per round.
///
/// The reference implementation appends the block K/V to its context cache and
/// crops it back after every round; here the block K/V lives in scratch and is
/// never cached — mathematically identical, no crop needed. Context K/V is
/// cached post-k_norm, post-RoPE at absolute positions, exactly like the
/// reference DynamicCache after `crop(start)`.
/// </summary>
public sealed unsafe class DSparkDraftModel : IDSparkDraft
{
    private readonly DSparkConfig _cfg;
    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _kvGroup;
    private readonly int _qDim, _kvDim, _interm, _vocab, _block, _rank, _tapDim;
    private readonly int _maxCtx;
    private readonly float _eps, _scale;

    // F32 weights (native, row-major [out, in] like the PyTorch state dict).
    private float* _fc;            // [embDim, tapDim]
    private float* _hiddenNormW;   // [embDim]
    private float* _finalNormW;    // [embDim]
    private float* _lmHead;        // [vocab, embDim]
    private float* _markovW2;      // [vocab, rank] (bias[v] = dot(w2[v], w1[prev]))
    private float* _confW;         // [embDim + rank] or [embDim]; null when no head
    private float _confB;
    private readonly float*[] _wq, _wk, _wv, _wo;          // [qDim|kvDim|kvDim, embDim], [embDim, qDim]
    private readonly float*[] _qNormW, _kNormW;            // [headDim]
    private readonly float*[] _inNormW, _ffnNormW;         // [embDim]
    private readonly float*[] _wGate, _wUp, _wDown;        // [interm, embDim] ×2, [embDim, interm]

    // Row-gathered tables in storage dtype (BF16 kept raw; F32 checkpoints kept F32).
    private ushort* _embedBf16; private float* _embedF32;      // [vocab, embDim]
    private ushort* _markovW1Bf16; private float* _markovW1F32; // [vocab, rank]

    // RoPE tables [maxCtx + block, headDim/2].
    private float* _ropeCos, _ropeSin;
    private readonly int _halfDim;

    // Context K/V cache: per layer [_ctxCap, kvDim], filled to _ctxLen. Grown
    // geometrically on append (the hard cap _maxCtx only sizes the RoPE tables —
    // eagerly allocating it would zero out gigabytes for long-context targets).
    private readonly float*[] _ctxK, _ctxV;
    private int _ctxCap;
    private int _ctxLen;

    // Block scratch (persistent; sized once).
    private float* _x, _resid, _norm;            // [block, embDim]
    private float* _q;                           // [block, qDim]
    private float* _kBlock, _vBlock;             // [block, kvDim]
    private float* _attnOut;                     // [block, qDim]
    private float* _gate, _up;                   // [block, interm]
    private float* _logits, _bias;               // [vocab]
    private float* _w1Rows;                      // [block, rank] — markov_w1[prev_j] per position
    private float* _confFeat;                    // [embDim + rank]
    private bool _disposed;

    public int BlockSize => _block;
    public int VocabSize => _vocab;
    public int TapDim => _tapDim;
    public int ContextLength => _ctxLen;
    public int MaxContext => _maxCtx;

    /// <summary>Layer ids to pass to <see cref="IForwardPass.EnableHiddenTaps"/> on the target.</summary>
    public int[] TargetLayerIds => _cfg.TargetLayerIds;

    public int MaskTokenId => _cfg.MaskTokenId;

    public DSparkDraftModel(DSparkConfig cfg, SafetensorsLoader weights, int maxContextLength)
    {
        _cfg = cfg;
        _embDim = cfg.HiddenSize;
        _headDim = cfg.HeadDim;
        _numHeads = cfg.NumHeads;
        _numKvHeads = cfg.NumKvHeads;
        _kvGroup = cfg.NumHeads / cfg.NumKvHeads;
        _qDim = _numHeads * _headDim;
        _kvDim = _numKvHeads * _headDim;
        _interm = cfg.IntermediateSize;
        _vocab = cfg.VocabSize;
        _block = cfg.BlockSize;
        _rank = cfg.MarkovRank;
        _tapDim = cfg.TapDim;
        _eps = cfg.RmsNormEps;
        _scale = 1f / MathF.Sqrt(_headDim);
        _maxCtx = Math.Min(maxContextLength, cfg.MaxPositionEmbeddings);
        if (_maxCtx < 1)
            throw new ArgumentOutOfRangeException(nameof(maxContextLength));
        _halfDim = _headDim / 2;

        int L = cfg.NumLayers;
        _wq = new float*[L]; _wk = new float*[L]; _wv = new float*[L]; _wo = new float*[L];
        _qNormW = new float*[L]; _kNormW = new float*[L];
        _inNormW = new float*[L]; _ffnNormW = new float*[L];
        _wGate = new float*[L]; _wUp = new float*[L]; _wDown = new float*[L];
        _ctxK = new float*[L]; _ctxV = new float*[L];

        try
        {
            _fc = LoadF32(weights, "fc.weight", [_embDim, _tapDim]);
            _hiddenNormW = LoadF32(weights, "hidden_norm.weight", [_embDim]);
            _finalNormW = LoadF32(weights, "norm.weight", [_embDim]);
            _lmHead = LoadF32(weights, "lm_head.weight", [_vocab, _embDim]);
            LoadRowTable(weights, "embed_tokens.weight", [_vocab, _embDim],
                out _embedBf16, out _embedF32);

            if (_rank > 0)
            {
                _markovW2 = LoadF32(weights, "markov_head.markov_w2.weight", [_vocab, _rank]);
                LoadRowTable(weights, "markov_head.markov_w1.weight", [_vocab, _rank],
                    out _markovW1Bf16, out _markovW1F32);
            }

            if (cfg.EnableConfidenceHead)
            {
                int confIn = _embDim + (cfg.ConfidenceHeadWithMarkov ? _rank : 0);
                _confW = LoadF32(weights, "confidence_head.proj.weight", [1, confIn]);
                var b = weights.ReadF32("confidence_head.proj.bias");
                if (b.Length != 1)
                    throw new InvalidDataException("confidence_head.proj.bias must be a scalar.");
                _confB = b[0];
            }

            for (int l = 0; l < L; l++)
            {
                _wq[l] = LoadF32(weights, $"layers.{l}.self_attn.q_proj.weight", [_qDim, _embDim]);
                _wk[l] = LoadF32(weights, $"layers.{l}.self_attn.k_proj.weight", [_kvDim, _embDim]);
                _wv[l] = LoadF32(weights, $"layers.{l}.self_attn.v_proj.weight", [_kvDim, _embDim]);
                _wo[l] = LoadF32(weights, $"layers.{l}.self_attn.o_proj.weight", [_embDim, _qDim]);
                _qNormW[l] = LoadF32(weights, $"layers.{l}.self_attn.q_norm.weight", [_headDim]);
                _kNormW[l] = LoadF32(weights, $"layers.{l}.self_attn.k_norm.weight", [_headDim]);
                _inNormW[l] = LoadF32(weights, $"layers.{l}.input_layernorm.weight", [_embDim]);
                _ffnNormW[l] = LoadF32(weights, $"layers.{l}.post_attention_layernorm.weight", [_embDim]);
                _wGate[l] = LoadF32(weights, $"layers.{l}.mlp.gate_proj.weight", [_interm, _embDim]);
                _wUp[l] = LoadF32(weights, $"layers.{l}.mlp.up_proj.weight", [_interm, _embDim]);
                _wDown[l] = LoadF32(weights, $"layers.{l}.mlp.down_proj.weight", [_embDim, _interm]);
            }

            int ropePositions = _maxCtx + _block;
            _ropeCos = Alloc((long)ropePositions * _halfDim);
            _ropeSin = Alloc((long)ropePositions * _halfDim);
            SimdKernels.BuildRopeTable(_ropeCos, _ropeSin, ropePositions, _headDim, cfg.RopeTheta);

            _x = Alloc((long)_block * _embDim);
            _resid = Alloc((long)_block * _embDim);
            _norm = Alloc((long)_block * _embDim);
            _q = Alloc((long)_block * _qDim);
            _kBlock = Alloc((long)_block * _kvDim);
            _vBlock = Alloc((long)_block * _kvDim);
            _attnOut = Alloc((long)_block * _qDim);
            _gate = Alloc((long)_block * _interm);
            _up = Alloc((long)_block * _interm);
            _logits = Alloc(_vocab);
            _bias = Alloc(_vocab);
            _w1Rows = Alloc((long)_block * Math.Max(_rank, 1));
            _confFeat = Alloc(_embDim + Math.Max(_rank, 1));
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    /// <summary>
    /// Approximate CPU-resident bytes of a loaded draft head at this config:
    /// F32 for everything except the two row-gather tables, which stay BF16.
    /// Used by the placement planner before any weight is read.
    /// </summary>
    public static long EstimateResidentBytes(DSparkConfig cfg)
    {
        long embDim = cfg.HiddenSize;
        long perLayer =
            (long)cfg.NumHeads * cfg.HeadDim * embDim * 2      // q_proj + o_proj
            + (long)cfg.NumKvHeads * cfg.HeadDim * embDim * 2  // k_proj + v_proj
            + (long)cfg.IntermediateSize * embDim * 3          // gate/up/down
            + cfg.HeadDim * 2 + embDim * 2;                    // qk norms + layer norms
        long f32Elems =
            embDim * (long)cfg.TapDim                          // fc
            + embDim * 2                                       // hidden_norm + norm
            + (long)cfg.VocabSize * embDim                     // lm_head
            + (cfg.MarkovRank > 0 ? (long)cfg.VocabSize * cfg.MarkovRank : 0)  // markov_w2
            + perLayer * cfg.NumLayers;
        long bf16Elems =
            (long)cfg.VocabSize * embDim                       // embed_tokens
            + (cfg.MarkovRank > 0 ? (long)cfg.VocabSize * cfg.MarkovRank : 0); // markov_w1
        return f32Elems * sizeof(float) + bf16Elems * sizeof(ushort);
    }

    // ── IDSparkDraft ─────────────────────────────────────────────────────

    public void AppendContext(ReadOnlySpan<float> taps, int startPos, int count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (count == 0) return;
        if (startPos != _ctxLen)
            throw new InvalidOperationException(
                $"AppendContext must be contiguous: startPos={startPos}, ContextLength={_ctxLen}.");
        if (startPos + count > _maxCtx)
            throw new InvalidOperationException(
                $"Context overflow: {startPos + count} > maxContextLength {_maxCtx}.");
        if (taps.Length != (long)count * _tapDim)
            throw new ArgumentException(
                $"Expected {count} × {_tapDim} tap floats, got {taps.Length}.", nameof(taps));

        EnsureCtxCapacity(startPos + count);

        // fused = RMSNorm_hidden_norm(fc @ tap) per position — reused by every layer.
        // Allocated inside the try so a failed later Alloc can't leak the earlier ones.
        float* fused = null, kBuf = null, vBuf = null;
        try
        {
            fused = (float*)NativeMemory.Alloc((nuint)((long)count * _embDim * sizeof(float)));
            kBuf = (float*)NativeMemory.Alloc((nuint)((long)count * _kvDim * sizeof(float)));
            vBuf = (float*)NativeMemory.Alloc((nuint)((long)count * _kvDim * sizeof(float)));
            fixed (float* tapsPtr = taps)
            {
                SimdKernels.MatMulBatchedF32(fused, _fc, tapsPtr, count, _embDim, _tapDim);
            }
            for (int i = 0; i < count; i++)
                SimdKernels.RmsNorm(fused + (long)i * _embDim, fused + (long)i * _embDim,
                    _hiddenNormW, _embDim, _eps);

            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                SimdKernels.MatMulBatchedF32(kBuf, _wk[l], fused, count, _kvDim, _embDim);
                SimdKernels.MatMulBatchedF32(vBuf, _wv[l], fused, count, _kvDim, _embDim);
                for (int i = 0; i < count; i++)
                {
                    float* k = kBuf + (long)i * _kvDim;
                    PerHeadRmsNorm(k, _kNormW[l], _numKvHeads);
                    Rope(k, startPos + i, _numKvHeads);
                    long dst = (long)(startPos + i) * _kvDim;
                    Copy(_ctxK[l] + dst, k, _kvDim);
                    Copy(_ctxV[l] + dst, vBuf + (long)i * _kvDim, _kvDim);
                }
            }
        }
        finally
        {
            if (fused != null) NativeMemory.Free(fused);
            if (kBuf != null) NativeMemory.Free(kBuf);
            if (vBuf != null) NativeMemory.Free(vBuf);
        }
        _ctxLen = startPos + count;
    }

    public DSparkProposal ProposeBlock(int anchorToken, int anchorPos)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (anchorPos != _ctxLen)
            throw new InvalidOperationException(
                $"ProposeBlock anchorPos={anchorPos} but ContextLength={_ctxLen} " +
                "(all taps below the anchor must be appended first).");
        if ((uint)anchorToken >= (uint)_vocab)
            throw new ArgumentOutOfRangeException(nameof(anchorToken));

        int B = _block;

        // Block inputs: [embed(anchor), embed(mask) × (B-1)] at positions anchorPos + j.
        // The mask row is dequantized once into slot 1 and copied to the rest.
        EmbedRow(anchorToken, _x);
        if (B > 1)
        {
            EmbedRow(_cfg.MaskTokenId, _x + _embDim);
            for (int j = 2; j < B; j++)
                Copy(_x + (long)j * _embDim, _x + _embDim, _embDim);
        }

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            // Attention.
            Copy(_resid, _x, (long)B * _embDim);
            for (int j = 0; j < B; j++)
                SimdKernels.RmsNorm(_norm + (long)j * _embDim, _x + (long)j * _embDim,
                    _inNormW[l], _embDim, _eps);

            SimdKernels.MatMulBatchedF32(_q, _wq[l], _norm, B, _qDim, _embDim);
            SimdKernels.MatMulBatchedF32(_kBlock, _wk[l], _norm, B, _kvDim, _embDim);
            SimdKernels.MatMulBatchedF32(_vBlock, _wv[l], _norm, B, _kvDim, _embDim);

            for (int j = 0; j < B; j++)
            {
                float* qj = _q + (long)j * _qDim;
                float* kj = _kBlock + (long)j * _kvDim;
                PerHeadRmsNorm(qj, _qNormW[l], _numHeads);
                PerHeadRmsNorm(kj, _kNormW[l], _numKvHeads);
                Rope(qj, anchorPos + j, _numHeads);
                Rope(kj, anchorPos + j, _numKvHeads);
            }

            BlockAttention(l, anchorPos);

            SimdKernels.MatMulBatchedF32(_norm, _wo[l], _attnOut, B, _embDim, _qDim);
            for (int j = 0; j < B; j++)
            {
                float* xj = _x + (long)j * _embDim;
                Copy(xj, _norm + (long)j * _embDim, _embDim);
                SimdKernels.AddInPlace(xj, _resid + (long)j * _embDim, _embDim);
            }

            // FFN (SwiGLU).
            Copy(_resid, _x, (long)B * _embDim);
            for (int j = 0; j < B; j++)
                SimdKernels.RmsNorm(_norm + (long)j * _embDim, _x + (long)j * _embDim,
                    _ffnNormW[l], _embDim, _eps);
            SimdKernels.MatMulBatchedF32(_gate, _wGate[l], _norm, B, _interm, _embDim);
            SimdKernels.MatMulBatchedF32(_up, _wUp[l], _norm, B, _interm, _embDim);
            for (int j = 0; j < B; j++)
                SimdKernels.SiLuMul(_gate + (long)j * _interm, _up + (long)j * _interm, _interm);
            SimdKernels.MatMulBatchedF32(_norm, _wDown[l], _gate, B, _embDim, _interm);
            for (int j = 0; j < B; j++)
            {
                float* xj = _x + (long)j * _embDim;
                Copy(xj, _norm + (long)j * _embDim, _embDim);
                SimdKernels.AddInPlace(xj, _resid + (long)j * _embDim, _embDim);
            }
        }

        // Final norm → block hidden states (kept in _x for the confidence head).
        for (int j = 0; j < B; j++)
            SimdKernels.RmsNorm(_x + (long)j * _embDim, _x + (long)j * _embDim,
                _finalNormW, _embDim, _eps);

        // Parallel base logits + sequential Markov correction, greedy. Each
        // position's markov_w1[prev_j] row is gathered once into _w1Rows and
        // reused by the confidence head below (same prev-token schedule:
        // anchor for j=0, tokens[j-1] after).
        // Note: the B lm_head matvecs re-stream the [vocab, embDim] weight per
        // position; a multi-input F32 kernel (MatVec4In-style) could amortize
        // that — Phase 4 perf work alongside the GPU draft path.
        var tokens = new int[B];
        int prev = anchorToken;
        for (int j = 0; j < B; j++)
        {
            SimdKernels.MatVecF32(_logits, _lmHead, _x + (long)j * _embDim, _vocab, _embDim);
            if (_rank > 0)
            {
                float* w1Row = _w1Rows + (long)j * _rank;
                MarkovW1Row(prev, w1Row);
                SimdKernels.MatVecF32(_bias, _markovW2, w1Row, _vocab, _rank);
                SimdKernels.AddInPlace(_logits, _bias, _vocab);
            }
            tokens[j] = Sampler.Greedy(new ReadOnlySpan<float>(_logits, _vocab));
            prev = tokens[j];
        }

        // Confidence head: features = [hidden_j ‖ markov_w1[prev_j]].
        var conf = new float[B];
        if (_confW != null)
        {
            int confIn = _embDim + (_cfg.ConfidenceHeadWithMarkov ? _rank : 0);
            for (int j = 0; j < B; j++)
            {
                Copy(_confFeat, _x + (long)j * _embDim, _embDim);
                if (_cfg.ConfidenceHeadWithMarkov)
                    Copy(_confFeat + _embDim, _w1Rows + (long)j * _rank, _rank);
                float logit = SimdKernels.DotF32(_confW, _confFeat, confIn) + _confB;
                conf[j] = 1f / (1f + MathF.Exp(-logit));
            }
        }
        else
        {
            Array.Fill(conf, 1f);
        }

        return new DSparkProposal(tokens, conf);
    }

    public void TruncateContext(int length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (length < 0 || length > _ctxLen)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"TruncateContext({length}) outside [0, {_ctxLen}].");
        _ctxLen = length;
    }

    public void ResetContext() => _ctxLen = 0;

    // ── Internals ────────────────────────────────────────────────────────

    /// <summary>
    /// Bidirectional GQA attention for the block: every block query attends over
    /// all cached context K/V (positions 0..ctxLen-1) plus every block position.
    /// </summary>
    private void BlockAttention(int layer, int anchorPos)
    {
        int B = _block;
        int ctxLen = _ctxLen;
        int total = ctxLen + B;
        float* ctxK = _ctxK[layer];
        float* ctxV = _ctxV[layer];

        Parallel.For(0, _numHeads * B, hj =>
        {
            int h = hj / B;
            int j = hj % B;
            int kvHead = h / _kvGroup;
            long kvOff = (long)kvHead * _headDim;
            float* q = _q + (long)j * _qDim + (long)h * _headDim;
            float[] scoreArr = System.Buffers.ArrayPool<float>.Shared.Rent(total);
            fixed (float* sc = scoreArr)
            {
                for (int c = 0; c < ctxLen; c++)
                    sc[c] = SimdKernels.DotF32(q, ctxK + (long)c * _kvDim + kvOff, _headDim) * _scale;
                for (int c = 0; c < B; c++)
                    sc[ctxLen + c] = SimdKernels.DotF32(q, _kBlock + (long)c * _kvDim + kvOff, _headDim) * _scale;

                SimdKernels.SoftmaxInPlace(sc, total);

                float* outp = _attnOut + (long)j * _qDim + (long)h * _headDim;
                new Span<float>(outp, _headDim).Clear();
                for (int c = 0; c < ctxLen; c++)
                    SimdKernels.WeightedAddInPlace(outp, ctxV + (long)c * _kvDim + kvOff, sc[c], _headDim);
                for (int c = 0; c < B; c++)
                    SimdKernels.WeightedAddInPlace(outp, _vBlock + (long)c * _kvDim + kvOff, sc[ctxLen + c], _headDim);
            }
            System.Buffers.ArrayPool<float>.Shared.Return(scoreArr);
        });
    }

    /// <summary>Grow the per-layer context K/V buffers to hold at least <paramref name="positions"/> rows.</summary>
    private void EnsureCtxCapacity(int positions)
    {
        if (positions <= _ctxCap) return;
        int newCap = Math.Max(_ctxCap == 0 ? 1024 : _ctxCap * 2, positions);
        newCap = Math.Min(newCap, _maxCtx);
        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            GrowBuffer(ref _ctxK[l], newCap);
            GrowBuffer(ref _ctxV[l], newCap);
        }
        _ctxCap = newCap;

        void GrowBuffer(ref float* buf, int cap)
        {
            var next = Alloc((long)cap * _kvDim);
            if (buf != null)
            {
                Copy(next, buf, (long)_ctxLen * _kvDim);
                NativeMemory.Free(buf);
            }
            buf = next;
        }
    }

    private void PerHeadRmsNorm(float* x, float* weight, int heads)
    {
        for (int h = 0; h < heads; h++)
            SimdKernels.RmsNorm(x + (long)h * _headDim, x + (long)h * _headDim,
                weight, _headDim, _eps);
    }

    private void Rope(float* x, int position, int heads)
    {
        float* cos = _ropeCos + (long)position * _halfDim;
        float* sin = _ropeSin + (long)position * _halfDim;
        SimdKernels.ApplyRoPECachedNeox(x, cos, sin, heads, _headDim);
    }

    private void EmbedRow(int token, float* dst)
    {
        if (_embedF32 != null)
        {
            Copy(dst, _embedF32 + (long)token * _embDim, _embDim);
        }
        else
        {
            var src = new ReadOnlySpan<byte>((byte*)(_embedBf16 + (long)token * _embDim),
                _embDim * sizeof(ushort));
            Dequantize.ToFloat32(src, new Span<float>(dst, _embDim), DType.BFloat16, _embDim);
        }
    }

    private void MarkovW1Row(int token, float* dst)
    {
        if (_markovW1F32 != null)
        {
            Copy(dst, _markovW1F32 + (long)token * _rank, _rank);
        }
        else
        {
            var src = new ReadOnlySpan<byte>((byte*)(_markovW1Bf16 + (long)token * _rank),
                _rank * sizeof(ushort));
            Dequantize.ToFloat32(src, new Span<float>(dst, _rank), DType.BFloat16, _rank);
        }
    }

    private static float* Alloc(long floats) =>
        (float*)NativeMemory.AllocZeroed((nuint)(floats * sizeof(float)));

    private static void Copy(float* dst, float* src, long floats) =>
        NativeMemory.Copy(src, dst, (nuint)(floats * sizeof(float)));

    private static void ValidateShape(SafetensorsLoader st, string name, int[] expectedShape)
    {
        var shape = st.GetShape(name);
        if (!shape.AsSpan().SequenceEqual(expectedShape))
            throw new InvalidDataException(
                $"DSpark tensor '{name}' has shape [{string.Join(",", shape)}], " +
                $"expected [{string.Join(",", expectedShape)}].");
    }

    private static float* LoadF32(SafetensorsLoader st, string name, int[] expectedShape)
    {
        ValidateShape(st, name, expectedShape);
        var managed = st.ReadF32(name);
        var buf = Alloc(managed.Length);
        managed.AsSpan().CopyTo(new Span<float>(buf, managed.Length));
        return buf;
    }

    /// <summary>
    /// Load a large row-gathered table in its storage dtype: BF16 stays BF16
    /// (half the resident bytes; rows are dequantized on access), anything else
    /// goes through the F32 conversion path.
    /// </summary>
    private static void LoadRowTable(SafetensorsLoader st, string name, int[] expectedShape,
        out ushort* bf16, out float* f32)
    {
        ValidateShape(st, name, expectedShape);

        long elems = (long)expectedShape[0] * expectedShape[1];
        var raw = st.ReadRaw(name, out string dtype);
        if (dtype == "BF16")
        {
            f32 = null;
            bf16 = (ushort*)NativeMemory.Alloc((nuint)raw.LongLength);
            raw.AsSpan().CopyTo(new Span<byte>(bf16, raw.Length));
        }
        else
        {
            bf16 = null;
            var managed = st.ReadF32(name);
            f32 = Alloc(elems);
            managed.AsSpan().CopyTo(new Span<float>(f32, managed.Length));
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        FreeAll();
    }

    private void FreeAll()
    {
        Free(ref _fc); Free(ref _hiddenNormW); Free(ref _finalNormW);
        Free(ref _lmHead); Free(ref _markovW2); Free(ref _confW);
        FreeArr(_wq); FreeArr(_wk); FreeArr(_wv); FreeArr(_wo);
        FreeArr(_qNormW); FreeArr(_kNormW); FreeArr(_inNormW); FreeArr(_ffnNormW);
        FreeArr(_wGate); FreeArr(_wUp); FreeArr(_wDown);
        FreeArr(_ctxK); FreeArr(_ctxV);
        if (_embedBf16 != null) { NativeMemory.Free(_embedBf16); _embedBf16 = null; }
        Free(ref _embedF32);
        if (_markovW1Bf16 != null) { NativeMemory.Free(_markovW1Bf16); _markovW1Bf16 = null; }
        Free(ref _markovW1F32);
        Free(ref _ropeCos); Free(ref _ropeSin);
        Free(ref _x); Free(ref _resid); Free(ref _norm);
        Free(ref _q); Free(ref _kBlock); Free(ref _vBlock); Free(ref _attnOut);
        Free(ref _gate); Free(ref _up); Free(ref _logits); Free(ref _bias);
        Free(ref _w1Rows); Free(ref _confFeat);

        static void Free(ref float* p) { if (p != null) { NativeMemory.Free(p); p = null; } }
        static void FreeArr(float*[]? arr)
        {
            if (arr is null) return;
            for (int i = 0; i < arr.Length; i++)
                if (arr[i] != null) { NativeMemory.Free(arr[i]); arr[i] = null; }
        }
    }
}
