using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Engine;

/// <summary>
/// CUDA implementation of the DSpark draft head (docs/dspark-plan.md Phase 4,
/// PR #413). Same math as <see cref="DSparkDraftModel"/> — an EAGLE-3-style
/// block drafter over fused target hidden-state taps — with the backbone on
/// the GPU and the sequential heads on the host:
///
/// <list type="bullet">
/// <item>Projections (fc, q/k/v/o, gate/up/down, lm_head) are resident fp16
/// (BF16→FP16 is exact for weight-range values) driven through one
/// <c>cublasGemmEx</c> each (<see cref="CudaBackend.MatMulBatchedGemmF16W"/>,
/// fp32 activations/accumulation). Norm weights stay f32.</item>
/// <item>Per-layer context K/V lives in device buffers, grown geometrically,
/// with <see cref="DSparkConfig.BlockSize"/> slack rows at the tail: the block's
/// own K/V is projected straight into rows [ctxLen, ctxLen+B), and the
/// mask-free <see cref="CudaBackend.Attention"/> kernel over seqLen = ctxLen+B
/// gives every block query bidirectional visibility of context + block —
/// no crop, identical semantics to the CPU model's scratch-block attention.</item>
/// <item>Base logits and final hiddens are downloaded once per block
/// (~[B×vocab] + [B×embDim] floats) and the shared <see cref="DSparkHostHeads"/>
/// applies the Markov re-bias, greedy chain, and confidence scoring on the
/// host — the semi-autoregressive part is tiny and inherently sequential.</item>
/// </list>
///
/// Draft-side numerics (fp16 GEMMs) affect acceptance rate only, never emitted
/// tokens — greedy parity is enforced by the target's verify. Must share the
/// TARGET's <see cref="CudaBackend"/> instance: one stream orders the tap
/// producer and this consumer implicitly, and taps arrive as host spans either way.
/// </summary>
public sealed unsafe class CudaDSparkDraftModel : IDSparkDraft
{
    private readonly DSparkConfig _cfg;
    private readonly CudaBackend _gpu;
    private readonly int _embDim, _headDim, _numHeads, _numKvHeads;
    private readonly int _qDim, _kvDim, _interm, _vocab, _block, _tapDim;
    private readonly int _maxCtx;
    private readonly float _eps;

    // fp16 resident projection weights.
    private Tensor? _fc;                       // [embDim, tapDim]
    private Tensor? _lmHead;                   // [vocab, embDim]
    private readonly Tensor?[] _wq, _wk, _wv, _wo, _wGate, _wUp, _wDown;

    // f32 norm weights.
    private Tensor? _hiddenNormW, _finalNormW;
    private readonly Tensor?[] _qNormW, _kNormW, _inNormW, _ffnNormW;

    // Host-side pieces shared with the CPU model.
    private DSparkHostHeads? _heads;
    private ushort* _embedBf16; private float* _embedF32;   // [vocab, embDim]

    // Device context K/V per layer: [_ctxCap, kvDim] f32, filled to _ctxLen,
    // with _block slack rows at the tail for the in-place block K/V.
    private readonly Tensor?[] _ctxK, _ctxV;
    private int _ctxCap;
    private int _ctxLen;
    private Tensor? _attnScratch;              // [numHeads × _ctxCap] once cap > 4096

    // Block scratch (fixed B rows).
    private Tensor? _x, _resid, _norm;         // [B × embDim]
    private Tensor? _q, _attnOut;              // [B × qDim]
    private Tensor? _gate, _up;                // [B × interm]
    private Tensor? _logitsDev;                // [B × vocab]
    private readonly float[] _xHost;           // [B × embDim] block input assembly
    private readonly float[] _baseLogitsHost;  // [B × vocab]
    private readonly float[] _blockHiddenHost; // [B × embDim]

    // AppendContext scratch, grown to the chunk size.
    private const int AppendChunkRows = 256;
    private Tensor? _tapsDev, _fusedDev;
    private float[] _tapClampScratch = [];
    private int _appendCap;
    private bool _disposed;

    public int BlockSize => _block;
    public int VocabSize => _vocab;
    public int TapDim => _tapDim;
    public int ContextLength => _ctxLen;
    public int MaxContext => _maxCtx;

    /// <summary>Layer ids to pass to <see cref="IForwardPass.EnableHiddenTaps"/> on the target.</summary>
    public int[] TargetLayerIds => _cfg.TargetLayerIds;

    public CudaDSparkDraftModel(DSparkConfig cfg, SafetensorsLoader weights, CudaBackend gpu,
        int maxContextLength)
    {
        _cfg = cfg;
        _gpu = gpu;
        _embDim = cfg.HiddenSize;
        _headDim = cfg.HeadDim;
        _numHeads = cfg.NumHeads;
        _numKvHeads = cfg.NumKvHeads;
        _qDim = _numHeads * _headDim;
        _kvDim = _numKvHeads * _headDim;
        _interm = cfg.IntermediateSize;
        _vocab = cfg.VocabSize;
        _block = cfg.BlockSize;
        _tapDim = cfg.TapDim;
        _eps = cfg.RmsNormEps;
        _maxCtx = Math.Min(maxContextLength, cfg.MaxPositionEmbeddings);
        if (_maxCtx < 1)
            throw new ArgumentOutOfRangeException(nameof(maxContextLength));

        int L = cfg.NumLayers;
        _wq = new Tensor?[L]; _wk = new Tensor?[L]; _wv = new Tensor?[L]; _wo = new Tensor?[L];
        _wGate = new Tensor?[L]; _wUp = new Tensor?[L]; _wDown = new Tensor?[L];
        _qNormW = new Tensor?[L]; _kNormW = new Tensor?[L];
        _inNormW = new Tensor?[L]; _ffnNormW = new Tensor?[L];
        _ctxK = new Tensor?[L]; _ctxV = new Tensor?[L];

        try
        {
            _fc = UploadF16(weights, "fc.weight", [_embDim, _tapDim]);
            _lmHead = UploadF16(weights, "lm_head.weight", [_vocab, _embDim]);
            _hiddenNormW = UploadF32(weights, "hidden_norm.weight", [_embDim]);
            _finalNormW = UploadF32(weights, "norm.weight", [_embDim]);
            DSparkWeightLoading.LoadRowTable(weights, "embed_tokens.weight", [_vocab, _embDim],
                out _embedBf16, out _embedF32);
            _heads = new DSparkHostHeads(cfg, weights);

            for (int l = 0; l < L; l++)
            {
                _wq[l] = UploadF16(weights, $"layers.{l}.self_attn.q_proj.weight", [_qDim, _embDim]);
                _wk[l] = UploadF16(weights, $"layers.{l}.self_attn.k_proj.weight", [_kvDim, _embDim]);
                _wv[l] = UploadF16(weights, $"layers.{l}.self_attn.v_proj.weight", [_kvDim, _embDim]);
                _wo[l] = UploadF16(weights, $"layers.{l}.self_attn.o_proj.weight", [_embDim, _qDim]);
                _wGate[l] = UploadF16(weights, $"layers.{l}.mlp.gate_proj.weight", [_interm, _embDim]);
                _wUp[l] = UploadF16(weights, $"layers.{l}.mlp.up_proj.weight", [_interm, _embDim]);
                _wDown[l] = UploadF16(weights, $"layers.{l}.mlp.down_proj.weight", [_embDim, _interm]);
                _qNormW[l] = UploadF32(weights, $"layers.{l}.self_attn.q_norm.weight", [_headDim]);
                _kNormW[l] = UploadF32(weights, $"layers.{l}.self_attn.k_norm.weight", [_headDim]);
                _inNormW[l] = UploadF32(weights, $"layers.{l}.input_layernorm.weight", [_embDim]);
                _ffnNormW[l] = UploadF32(weights, $"layers.{l}.post_attention_layernorm.weight", [_embDim]);
            }

            _x = AllocF32((long)_block * _embDim);
            _resid = AllocF32((long)_block * _embDim);
            _norm = AllocF32((long)_block * _embDim);
            _q = AllocF32((long)_block * _qDim);
            _attnOut = AllocF32((long)_block * _qDim);
            _gate = AllocF32((long)_block * _interm);
            _up = AllocF32((long)_block * _interm);
            _logitsDev = AllocF32((long)_block * _vocab);
            _xHost = new float[_block * _embDim];
            _baseLogitsHost = new float[_block * _vocab];
            _blockHiddenHost = new float[_block * _embDim];
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    /// <summary>
    /// Approximate VRAM-resident bytes of a loaded head at this config: fp16
    /// projections + f32 norms (embeddings and the Markov/confidence heads stay
    /// on the host). Context K/V and scratch grow separately.
    /// </summary>
    public static long EstimateGpuResidentBytes(DSparkConfig cfg)
    {
        long embDim = cfg.HiddenSize;
        long perLayer =
            (long)cfg.NumHeads * cfg.HeadDim * embDim * 2
            + (long)cfg.NumKvHeads * cfg.HeadDim * embDim * 2
            + (long)cfg.IntermediateSize * embDim * 3;
        long f16Elems = embDim * (long)cfg.TapDim
            + (long)cfg.VocabSize * embDim
            + perLayer * cfg.NumLayers;
        long f32Elems = (embDim * 2 + (cfg.HeadDim * 2 + embDim * 2) * (long)cfg.NumLayers);
        return f16Elems * 2 + f32Elems * 4;
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

        EnsureCtxCapacity(startPos + count + _block);

        // Bounded chunks keep the transient tap/fused device scratch small even
        // when a caller hands the whole prompt at once.
        int done = 0;
        while (done < count)
        {
            int n = Math.Min(AppendChunkRows, count - done);
            AppendChunk(taps.Slice(done * _tapDim, n * _tapDim), startPos + done, n);
            done += n;
        }
        _ctxLen = startPos + count;
    }

    private void AppendChunk(ReadOnlySpan<float> taps, int startPos, int n)
    {
        EnsureAppendScratch(n);

        // Clamp taps into fp16's finite range before upload: MatMulBatchedGemmF16W
        // rounds activations f32→f16 without saturation, so a residual-stream
        // outlier above ±65504 would become ±Inf and poison the fused vector (and
        // through it, every layer's context K/V for the position).
        int len = n * _tapDim;
        if (_tapClampScratch.Length < len) _tapClampScratch = new float[len];
        for (int i = 0; i < len; i++)
            _tapClampScratch[i] = Math.Clamp(taps[i], -65504f, 65504f);

        var tapsN = TapsView(n);
        var fused = _gpu.View(_fusedDev!, 0, (long)n * _embDim);
        try
        {
            _gpu.UploadInto(tapsN, _tapClampScratch.AsSpan(0, len));
            // fused = RMSNorm_hidden_norm(fc @ tap) — the same fused vector feeds
            // every layer's k/v projections (reference `_forward_backbone`).
            _gpu.MatMulBatchedGemmF16W(fused, _fc!, tapsN, n);
            _gpu.RmsNormBatched(fused, fused, _hiddenNormW!, n, _embDim, _eps);

            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                var kRows = _gpu.View(_ctxK[l]!, (long)startPos * _kvDim, (long)n * _kvDim);
                var vRows = _gpu.View(_ctxV[l]!, (long)startPos * _kvDim, (long)n * _kvDim);
                try
                {
                    _gpu.MatMulBatchedGemmF16W(kRows, _wk[l]!, fused, n);
                    _gpu.HeadNormBatched(kRows, _kNormW[l]!, _numKvHeads, _headDim, n, _eps);
                    _gpu.RoPEPartialBatched(kRows, startPos, _headDim, _headDim,
                        _cfg.RopeTheta, _numKvHeads, n, neox: true);
                    _gpu.MatMulBatchedGemmF16W(vRows, _wv[l]!, fused, n);
                }
                finally
                {
                    _gpu.Free(kRows);
                    _gpu.Free(vRows);
                }
            }
        }
        finally
        {
            _gpu.Free(fused);
            _gpu.Free(tapsN);
        }
    }

    private Tensor TapsView(int n) => _gpu.View(_tapsDev!, 0, (long)n * _tapDim);

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
        EnsureCtxCapacity(anchorPos + B);

        // Block inputs: [embed(anchor), embed(mask) × (B-1)] assembled on host.
        EmbedRowHost(anchorToken, _xHost.AsSpan(0, _embDim));
        if (B > 1)
        {
            EmbedRowHost(_cfg.MaskTokenId, _xHost.AsSpan(_embDim, _embDim));
            for (int j = 2; j < B; j++)
                _xHost.AsSpan(_embDim, _embDim).CopyTo(_xHost.AsSpan(j * _embDim, _embDim));
        }
        _gpu.UploadInto(_x!, _xHost);

        int seqLen = anchorPos + B;
        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            // Attention. Block K/V is projected directly into the ctx buffers'
            // rows [anchorPos, anchorPos+B) — the next AppendContext at
            // startPos=anchorPos overwrites them, mirroring the reference crop.
            _gpu.CopyDevice(_resid!, _x!);
            _gpu.RmsNormBatched(_norm!, _x!, _inNormW[l]!, B, _embDim, _eps);
            _gpu.MatMulBatchedGemmF16W(_q!, _wq[l]!, _norm!, B);

            var kRows = _gpu.View(_ctxK[l]!, (long)anchorPos * _kvDim, (long)B * _kvDim);
            var vRows = _gpu.View(_ctxV[l]!, (long)anchorPos * _kvDim, (long)B * _kvDim);
            try
            {
                _gpu.MatMulBatchedGemmF16W(kRows, _wk[l]!, _norm!, B);
                _gpu.MatMulBatchedGemmF16W(vRows, _wv[l]!, _norm!, B);
                _gpu.HeadNormQkBatched(_q!, _qNormW[l]!, kRows, _kNormW[l]!,
                    _numHeads, _numKvHeads, _headDim, B, _eps);
                _gpu.RoPEPartialBatched(_q!, anchorPos, _headDim, _headDim,
                    _cfg.RopeTheta, _numHeads, B, neox: true);
                _gpu.RoPEPartialBatched(kRows, anchorPos, _headDim, _headDim,
                    _cfg.RopeTheta, _numKvHeads, B, neox: true);
            }
            finally
            {
                _gpu.Free(kRows);
                _gpu.Free(vRows);
            }

            // Bidirectional GQA: the kernel has no causal mask — every block query
            // scans all seqLen = ctx+block keys.
            for (int j = 0; j < B; j++)
            {
                var qRow = _gpu.View(_q!, (long)j * _qDim, _qDim);
                var oRow = _gpu.View(_attnOut!, (long)j * _qDim, _qDim);
                try
                {
                    _gpu.Attention(qRow, _ctxK[l]!, _ctxV[l]!, oRow, _attnScratch,
                        _numHeads, _numKvHeads, _headDim, seqLen, _ctxCap);
                }
                finally
                {
                    _gpu.Free(qRow);
                    _gpu.Free(oRow);
                }
            }

            _gpu.MatMulBatchedGemmF16W(_x!, _wo[l]!, _attnOut!, B);
            _gpu.AddInPlace(_x!, _resid!);

            // FFN (SwiGLU).
            _gpu.CopyDevice(_resid!, _x!);
            _gpu.RmsNormBatched(_norm!, _x!, _ffnNormW[l]!, B, _embDim, _eps);
            _gpu.MatMulBatchedGemmF16W(_gate!, _wGate[l]!, _norm!, B);
            _gpu.MatMulBatchedGemmF16W(_up!, _wUp[l]!, _norm!, B);
            _gpu.SiLuMul(_gate!, _up!);
            _gpu.MatMulBatchedGemmF16W(_x!, _wDown[l]!, _gate!, B);
            _gpu.AddInPlace(_x!, _resid!);
        }

        _gpu.RmsNormBatched(_x!, _x!, _finalNormW!, B, _embDim, _eps);
        _gpu.MatMulBatchedGemmF16W(_logitsDev!, _lmHead!, _x!, B);

        _gpu.Download(_logitsDev!, _baseLogitsHost);
        _gpu.Download(_x!, _blockHiddenHost);

        fixed (float* logits = _baseLogitsHost)
        fixed (float* hidden = _blockHiddenHost)
        {
            return _heads!.GreedyBlock(logits, hidden, anchorToken);
        }
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

    private void EnsureCtxCapacity(int positions)
    {
        if (positions <= _ctxCap) return;
        int newCap = Math.Max(_ctxCap == 0 ? 1024 : _ctxCap * 2, positions);
        newCap = Math.Min(newCap, _maxCtx + _block);

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            GrowCtxBuffer(ref _ctxK[l], newCap);
            GrowCtxBuffer(ref _ctxV[l], newCap);
        }

        // The attention kernel spills scores to scratch past 4096 keys; size it
        // for the full capacity so seqLen can reach _ctxCap.
        if (newCap > 4096)
        {
            if (_attnScratch is { } old) _gpu.Free(old);
            _attnScratch = AllocF32((long)_numHeads * newCap);
        }
        _ctxCap = newCap;

        void GrowCtxBuffer(ref Tensor? buf, int cap)
        {
            var next = AllocF32((long)cap * _kvDim);
            if (buf is { } prev)
            {
                if (_ctxLen > 0)
                    _gpu.CopyDeviceRegion(next, 0, prev, 0, (long)_ctxLen * _kvDim * sizeof(float));
                _gpu.Free(prev);
            }
            buf = next;
        }
    }

    private void EnsureAppendScratch(int rows)
    {
        if (rows <= _appendCap) return;
        if (_tapsDev is { } t) _gpu.Free(t);
        if (_fusedDev is { } f) _gpu.Free(f);
        _tapsDev = AllocF32((long)rows * _tapDim);
        _fusedDev = AllocF32((long)rows * _embDim);
        _appendCap = rows;
    }

    private void EmbedRowHost(int token, Span<float> dst)
    {
        if (_embedF32 != null)
        {
            new ReadOnlySpan<float>(_embedF32 + (long)token * _embDim, _embDim).CopyTo(dst);
        }
        else
        {
            var src = new ReadOnlySpan<byte>((byte*)(_embedBf16 + (long)token * _embDim),
                _embDim * sizeof(ushort));
            Dequantize.ToFloat32(src, dst, DType.BFloat16, _embDim);
        }
    }

    private Tensor AllocF32(long elems) => _gpu.Allocate(TensorShape.D1(elems));

    private Tensor UploadF16(SafetensorsLoader st, string name, int[] expectedShape)
    {
        DSparkWeightLoading.ValidateShape(st, name, expectedShape);
        var f32 = st.ReadF32(name);
        var f16 = new Half[f32.Length];
        for (int i = 0; i < f32.Length; i++) f16[i] = (Half)f32[i];
        return _gpu.UploadHalf(f16, TensorShape.D1(f16.Length));
    }

    private Tensor UploadF32(SafetensorsLoader st, string name, int[] expectedShape)
    {
        DSparkWeightLoading.ValidateShape(st, name, expectedShape);
        var f32 = st.ReadF32(name);
        return _gpu.Upload(f32, TensorShape.D1(f32.Length));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        FreeAll();
    }

    private void FreeAll()
    {
        _heads?.Dispose(); _heads = null;
        if (_embedBf16 != null)
        {
            System.Runtime.InteropServices.NativeMemory.Free(_embedBf16);
            _embedBf16 = null;
        }
        if (_embedF32 != null)
        {
            System.Runtime.InteropServices.NativeMemory.Free(_embedF32);
            _embedF32 = null;
        }

        FreeTensor(ref _fc); FreeTensor(ref _lmHead);
        FreeTensor(ref _hiddenNormW); FreeTensor(ref _finalNormW);
        FreeArr(_wq); FreeArr(_wk); FreeArr(_wv); FreeArr(_wo);
        FreeArr(_wGate); FreeArr(_wUp); FreeArr(_wDown);
        FreeArr(_qNormW); FreeArr(_kNormW); FreeArr(_inNormW); FreeArr(_ffnNormW);
        FreeArr(_ctxK); FreeArr(_ctxV);
        FreeTensor(ref _attnScratch);
        FreeTensor(ref _x); FreeTensor(ref _resid); FreeTensor(ref _norm);
        FreeTensor(ref _q); FreeTensor(ref _attnOut);
        FreeTensor(ref _gate); FreeTensor(ref _up); FreeTensor(ref _logitsDev);
        FreeTensor(ref _tapsDev); FreeTensor(ref _fusedDev);

        void FreeTensor(ref Tensor? t)
        {
            if (t is { } tensor) { _gpu.Free(tensor); t = null; }
        }

        void FreeArr(Tensor?[]? arr)
        {
            if (arr is null) return;
            for (int i = 0; i < arr.Length; i++)
                if (arr[i] is { } tensor) { _gpu.Free(tensor); arr[i] = null; }
        }
    }
}
