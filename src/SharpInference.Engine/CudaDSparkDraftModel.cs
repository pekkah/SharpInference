using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Engine;

/// <summary>
/// CUDA implementation of the DSpark draft head (docs/dspark-plan.md Phase 4,
/// PR #413). Same math as <see cref="DSparkDraftModel"/> — an EAGLE-3-style
/// block drafter over fused target hidden-state taps — fully resident on the
/// GPU, sequential Markov/confidence heads included (issue #428):
///
/// <list type="bullet">
/// <item>Projections (fc, q/k/v/o, gate/up/down, lm_head) are resident fp16
/// (BF16→FP16 is exact for weight-range values) driven through one
/// <c>cublasGemmEx</c> each (<see cref="CudaBackend.MatMulBatchedGemmF16W"/>,
/// fp32 activations/accumulation). Norm weights stay f32.</item>
/// <item>Per-layer context K/V lives in device buffers, grown geometrically,
/// with <see cref="DSparkConfig.BlockSize"/> slack rows at the tail: the block's
/// own K/V is projected straight into rows [ctxLen, ctxLen+B), and ONE mask-free
/// <see cref="CudaBackend.AttentionBatchedRagged"/> launch per layer (every
/// query row aliased to this layer's cache, slot = ctxLen+B-1) gives every
/// block query bidirectional visibility of context + block — bit-identical per
/// (query, head) to the per-query <see cref="CudaBackend.Attention"/> loop it
/// replaces (issue #428: 35 launches → 5), no crop, identical semantics to the
/// CPU model's scratch-block attention.</item>
/// <item>The Markov re-bias, greedy chain, and confidence head run ON-DEVICE
/// (issue #428): per position, a gather kernel pulls markov_w1[prev] (prev read
/// on-stream from the previous position's argmax), one beta=1 GemmEx adds the
/// markov_w2 bias into the logits row, and llm_argmax_rows picks the token —
/// no host sync inside the chain. Measured host-side, the old
/// <see cref="DSparkHostHeads"/> chain re-streamed the [vocab × rank] markov_w2
/// 7× per round at ~17.6 ms — ~65% of the whole draft round; on-device the same
/// traffic runs at HBM bandwidth and the [B×vocab] logits download disappears —
/// only [B] tokens + [B] confidences cross PCIe, behind a single stream sync.</item>
/// <item>Launch-count trims (issue #428): residual adds ride the o/down GEMMs'
/// beta=1 epilogue (no CopyDevice + AddInPlace pairs), q+k RoPE is one fused
/// launch, and back-to-back projections of one normed block reuse the f32→f16
/// activation conversion.</item>
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

    // Draft-only embedding table stays host-side (two rows gathered per block).
    private ushort* _embedBf16; private float* _embedF32;   // [vocab, embDim]

    // Device-side Markov/confidence heads (issue #428; semantics mirror
    // DSparkHostHeads exactly — see its GreedyBlock).
    private readonly int _rank;
    private readonly bool _confWithMarkov;
    private Tensor? _markovW1F16, _markovW2F16;  // [vocab, rank] fp16
    private Tensor? _confWDev;                   // [embDim (+ rank)] f32
    private float _confB;
    private Tensor? _w1RowDev;                   // [rank] — the bias GEMV activation
    private Tensor? _w1RowsDev;                  // [B, rank] — confidence feature tails
    private Tensor? _argmaxDev;                  // [B*2] llm_argmax_rows (idx bits, value)
    private Tensor? _confDev;                    // [B]

    // Device context K/V per layer: [_ctxCap, kvDim] f32, filled to _ctxLen,
    // with _block slack rows at the tail for the in-place block K/V.
    private readonly Tensor?[] _ctxK, _ctxV;
    private int _ctxCap;
    private int _ctxLen;
    private Tensor? _attnScratch;              // [B × numHeads × _ctxCap] once cap > 4096

    // Block scratch (fixed B rows). No residual buffer: x itself carries the
    // residual — the o/down GEMMs accumulate onto it via beta=1 (issue #428).
    private Tensor? _x, _norm;                 // [B × embDim]
    private Tensor? _q, _attnOut;              // [B × qDim]
    private Tensor? _gate, _up;                // [B × interm]
    private Tensor? _logitsDev;                // [B × vocab]
    private readonly float[] _xHost;           // [B × embDim] block input assembly
    // Pinned host landing zones for the per-block result pair — one async +
    // one sync copy share a single StreamSynchronize (issue #49 pattern).
    private nint _pinnedArgmax;                // [B*2] floats (idx bits, value)
    private nint _pinnedConf;                  // [B] floats
    // One-launch ragged attention args: every block query attends this layer's
    // ctx cache over the same [0, ctxLen+B) range (issue #428).
    private readonly Tensor[] _raggedK, _raggedV;
    private readonly int[] _raggedSlots;

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

    private readonly System.Diagnostics.Stopwatch _phaseSw = new();

    /// <summary>Milliseconds spent waiting on the per-block result download —
    /// the stream sync collapses the whole enqueued GPU pipeline (backbone +
    /// device-side heads) into this window, so it reads as "GPU execution +
    /// D2H transfer" per round (issue #428).</summary>
    public double GpuWaitMs { get; private set; }

    /// <summary>Milliseconds spent host-side after the download (proposal
    /// assembly). The Markov/confidence chain itself runs on-device (issue
    /// #428), so this should stay near zero — a regression here means the
    /// heads fell back to the host somehow.</summary>
    public double HostHeadsMs { get; private set; }

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

            // Device-side Markov/confidence heads (issue #428): fp16 markov
            // tables (BF16→FP16 exact for weight-range values, like the
            // projections), f32 confidence projection. Loading mirrors
            // DSparkHostHeads' names, shapes and optionality.
            _rank = cfg.MarkovRank;
            _confWithMarkov = cfg.ConfidenceHeadWithMarkov;
            if (_rank > 0)
            {
                _markovW1F16 = UploadF16(weights, "markov_head.markov_w1.weight", [_vocab, _rank]);
                _markovW2F16 = UploadF16(weights, "markov_head.markov_w2.weight", [_vocab, _rank]);
            }
            if (cfg.EnableConfidenceHead)
            {
                int confIn = _embDim + (_confWithMarkov ? _rank : 0);
                _confWDev = UploadF32(weights, "confidence_head.proj.weight", [1, confIn]);
                var b = weights.ReadF32("confidence_head.proj.bias");
                if (b.Length != 1)
                    throw new InvalidDataException("confidence_head.proj.bias must be a scalar.");
                _confB = b[0];
            }

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
            _norm = AllocF32((long)_block * _embDim);
            _q = AllocF32((long)_block * _qDim);
            _attnOut = AllocF32((long)_block * _qDim);
            _gate = AllocF32((long)_block * _interm);
            _up = AllocF32((long)_block * _interm);
            _logitsDev = AllocF32((long)_block * _vocab);
            _xHost = new float[_block * _embDim];
            if (_rank > 0)
            {
                _w1RowDev = AllocF32(_rank);    // exactly rank: the GEMV derives cols from it
                _w1RowsDev = AllocF32((long)_block * _rank);
            }
            _argmaxDev = AllocF32((long)_block * 2);
            _confDev = AllocF32(_block);
            _pinnedArgmax = CudaBackend.AllocatePinnedHost((nuint)((long)_block * 2 * sizeof(float)));
            _pinnedConf = CudaBackend.AllocatePinnedHost((nuint)((long)_block * sizeof(float)));
            if (_pinnedArgmax == nint.Zero || _pinnedConf == nint.Zero)
                throw new InvalidOperationException(
                    "cudaMallocHost failed for the DSpark block result buffers " +
                    $"({(long)_block * 3 * sizeof(float)} bytes pinned).");
            _raggedK = new Tensor[_block];
            _raggedV = new Tensor[_block];
            _raggedSlots = new int[_block];
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    /// <summary>
    /// Approximate VRAM-resident bytes of a loaded head at this config: fp16
    /// projections + fp16 Markov tables (issue #428 device-side heads) + f32
    /// norms/confidence proj (embeddings stay on the host). Context K/V and
    /// scratch grow separately.
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
            + perLayer * cfg.NumLayers
            + 2L * cfg.VocabSize * cfg.MarkovRank;   // markov_w1 + markov_w2 (#428)
        long f32Elems = (embDim * 2 + (cfg.HeadDim * 2 + embDim * 2) * (long)cfg.NumLayers);
        if (cfg.EnableConfidenceHead)
            f32Elems += embDim + (cfg.ConfidenceHeadWithMarkov ? cfg.MarkovRank : 0);
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

            // All 2L projections below read the SAME normed `fused` rows: the first
            // k GEMM converts them f32→f16 once, the rest reuse the scratch (#428).
            bool converted = false;
            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                var kRows = _gpu.View(_ctxK[l]!, (long)startPos * _kvDim, (long)n * _kvDim);
                var vRows = _gpu.View(_ctxV[l]!, (long)startPos * _kvDim, (long)n * _kvDim);
                try
                {
                    _gpu.MatMulBatchedGemmF16W(kRows, _wk[l]!, fused, n,
                        reuseConvertedInput: converted);
                    converted = true;
                    _gpu.HeadNormBatched(kRows, _kNormW[l]!, _numKvHeads, _headDim, n, _eps);
                    _gpu.RoPEPartialBatched(kRows, startPos, _headDim, _headDim,
                        _cfg.RopeTheta, _numKvHeads, n, neox: true);
                    _gpu.MatMulBatchedGemmF16W(vRows, _wv[l]!, fused, n,
                        reuseConvertedInput: true);
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
        // One-launch ragged attention args: every block query attends the same
        // bidirectional [0, seqLen) range of this layer's ctx buffers (the ragged
        // kernel attends [0, slots[j]+1) of caches[j]; aliasing all B rows to one
        // cache with slot seqLen-1 is exactly the per-query Attention loop it
        // replaces, bit-identical per (query, head)).
        for (int j = 0; j < B; j++) _raggedSlots[j] = seqLen - 1;

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            // Attention. Block K/V is projected directly into the ctx buffers'
            // rows [anchorPos, anchorPos+B) — the next AppendContext at
            // startPos=anchorPos overwrites them, mirroring the reference crop.
            // `_x` stays untouched below the norm and carries the residual: the
            // o-projection accumulates onto it via beta=1 (no copy, no add).
            _gpu.RmsNormBatched(_norm!, _x!, _inNormW[l]!, B, _embDim, _eps);
            _gpu.MatMulBatchedGemmF16W(_q!, _wq[l]!, _norm!, B);

            var kRows = _gpu.View(_ctxK[l]!, (long)anchorPos * _kvDim, (long)B * _kvDim);
            var vRows = _gpu.View(_ctxV[l]!, (long)anchorPos * _kvDim, (long)B * _kvDim);
            try
            {
                _gpu.MatMulBatchedGemmF16W(kRows, _wk[l]!, _norm!, B, reuseConvertedInput: true);
                _gpu.MatMulBatchedGemmF16W(vRows, _wv[l]!, _norm!, B, reuseConvertedInput: true);
                _gpu.HeadNormQkBatched(_q!, _qNormW[l]!, kRows, _kNormW[l]!,
                    _numHeads, _numKvHeads, _headDim, B, _eps);
                _gpu.RoPEPartialBatchedQk(_q!, kRows, anchorPos, _headDim, _headDim,
                    _cfg.RopeTheta, _numHeads, _numKvHeads, B, neox: true);
            }
            finally
            {
                _gpu.Free(kRows);
                _gpu.Free(vRows);
            }

            // Bidirectional GQA, one launch for all B queries (no causal mask —
            // every block query scans all seqLen = ctx+block keys).
            _raggedK.AsSpan().Fill(_ctxK[l]!);
            _raggedV.AsSpan().Fill(_ctxV[l]!);
            _gpu.AttentionBatchedRagged(_q!, _raggedK, _raggedV, _attnOut!, _attnScratch,
                _numHeads, _numKvHeads, _headDim, _raggedSlots, _ctxCap);

            _gpu.MatMulBatchedGemmF16W(_x!, _wo[l]!, _attnOut!, B, beta: 1f);

            // FFN (SwiGLU); `_x` again carries the residual through the down
            // projection's beta=1, and gate/up share one activation conversion.
            _gpu.RmsNormBatched(_norm!, _x!, _ffnNormW[l]!, B, _embDim, _eps);
            _gpu.MatMulBatchedGemmF16W(_gate!, _wGate[l]!, _norm!, B);
            _gpu.MatMulBatchedGemmF16W(_up!, _wUp[l]!, _norm!, B, reuseConvertedInput: true);
            _gpu.SiLuMul(_gate!, _up!);
            _gpu.MatMulBatchedGemmF16W(_x!, _wDown[l]!, _gate!, B, beta: 1f);
        }

        _gpu.RmsNormBatched(_x!, _x!, _finalNormW!, B, _embDim, _eps);
        _gpu.MatMulBatchedGemmF16W(_logitsDev!, _lmHead!, _x!, B);

        // Device-side Markov chain (issue #428), semantics of
        // DSparkHostHeads.GreedyBlock: per position j, gather markov_w1[prev]
        // (prev = anchor at j=0, else position j-1's argmax, read on-stream),
        // bias the logits row in place via a beta=1 GEMV against markov_w2,
        // then argmax the row. No host sync anywhere in the chain.
        if (_rank > 0)
        {
            for (int j = 0; j < B; j++)
            {
                _gpu.DSparkGatherW1(_markovW1F16!, _argmaxDev!, j, anchorToken,
                    _w1RowDev!, _w1RowsDev!, _rank);
                var logitsRow = _gpu.View(_logitsDev!, (long)j * _vocab, _vocab);
                var amRow = _gpu.View(_argmaxDev!, (long)j * 2, 2);
                try
                {
                    _gpu.MatMulBatchedGemmF16W(logitsRow, _markovW2F16!, _w1RowDev!, 1, beta: 1f);
                    _gpu.ArgmaxRowsToDevice(logitsRow, amRow, 1, _vocab, _vocab);
                }
                finally
                {
                    _gpu.Free(logitsRow);
                    _gpu.Free(amRow);
                }
            }
        }
        else
        {
            // No Markov head: the block's argmaxes are independent — one launch.
            _gpu.ArgmaxRowsToDevice(_logitsDev!, _argmaxDev!, B, _vocab, _vocab);
        }

        bool hasConf = _confWDev is not null;
        _phaseSw.Restart();
        if (hasConf)
        {
            _gpu.DSparkConfidence(_x!, _w1RowsDev ?? _confDev!, _confWDev!, _confB, _confDev!,
                _embDim, _rank, withMarkov: _confWithMarkov && _rank > 0, B);
            _gpu.DownloadAsync(_confDev!, _pinnedConf, B);
        }
        // One sync drains the whole pipeline + both result copies (issue #49).
        _gpu.Download(_argmaxDev!, _pinnedArgmax, B * 2);
        GpuWaitMs += _phaseSw.Elapsed.TotalMilliseconds;

        _phaseSw.Restart();
        var tokens = new int[B];
        var conf = new float[B];
        int* am = (int*)_pinnedArgmax;
        for (int j = 0; j < B; j++) tokens[j] = am[j * 2];
        if (hasConf) new ReadOnlySpan<float>((float*)_pinnedConf, B).CopyTo(conf);
        else Array.Fill(conf, 1f);
        HostHeadsMs += _phaseSw.Elapsed.TotalMilliseconds;
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
        // for the full capacity so seqLen can reach _ctxCap. The one-launch ragged
        // kernel spills per (query, head) — B × numHeads × cap floats (issue #428).
        if (newCap > 4096)
        {
            if (_attnScratch is { } old) _gpu.Free(old);
            _attnScratch = AllocF32((long)_block * _numHeads * newCap);
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
        FreeTensor(ref _x); FreeTensor(ref _norm);
        FreeTensor(ref _q); FreeTensor(ref _attnOut);
        FreeTensor(ref _gate); FreeTensor(ref _up); FreeTensor(ref _logitsDev);
        FreeTensor(ref _tapsDev); FreeTensor(ref _fusedDev);
        FreeTensor(ref _markovW1F16); FreeTensor(ref _markovW2F16); FreeTensor(ref _confWDev);
        FreeTensor(ref _w1RowDev); FreeTensor(ref _w1RowsDev);
        FreeTensor(ref _argmaxDev); FreeTensor(ref _confDev);
        CudaBackend.FreePinnedHost(_pinnedArgmax); _pinnedArgmax = nint.Zero;
        CudaBackend.FreePinnedHost(_pinnedConf); _pinnedConf = nint.Zero;

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
