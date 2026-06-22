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

    // Embedding table in VRAM (kept raw-quantized for Q4_K/Q6_K large vocabs, F32 for small/other).
    // _embDType records which path was taken so the lookup dispatches the matching shader.
    private readonly Tensor _gpuEmbedding;
    private readonly DType _embDType;

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

    // ── Speculative-decode batched verify (issue #308) ───────────────────────────────────

    /// <summary>Maximum k for the batched-trunk verify path (matches MatMulBatched's nTok cap of 8
    /// and the lazily-sized <see cref="EnsureBatchVerifyScratch"/> buffers).</summary>
    private const int MaxBatchVerifyK = 8;

    /// <summary>
    /// Whether the weight-amortizing batched trunk (<see cref="BatchVerifyBatched"/>) is usable: a
    /// dense (non-MoE) model whose EVERY trunk matmul weight is Q4_K or Q6_K — the only dtypes
    /// <c>MatMulBatched</c> amortizes (every other dtype hits its per-token single-row fallback,
    /// which allocates/frees temp tensors mid-recording — a recording hazard, and no speedup). When
    /// false, <see cref="BatchVerify"/> uses the bit-exact <see cref="BatchVerifyKLoop"/>. Computed
    /// once in the ctor: Qwen3-8B-Q4_K_M ⇒ true; Qwen3-0.6B-Q8_0 ⇒ false (Q8_0 weights).
    /// </summary>
    private readonly bool _canBatchedTrunk;

    /// <summary>Test hook: whether <see cref="BatchVerify"/> takes the weight-amortizing batched
    /// trunk (true) or the K-loop fallback (false). True for an all-Q4_K/Q6_K dense model.</summary>
    internal bool CanBatchedTrunk => _canBatchedTrunk;

    // Batched-verify scratch buffers [K][dim] (token-major contiguous), lazily allocated by
    // EnsureBatchVerifyScratch on first use (K ≤ MaxBatchVerifyK) and freed in Dispose. Sized at
    // the single-query dims × the allocated K; reused across calls when K is non-decreasing.
    private int _bvK;                          // K the buffers below are currently sized for (0 = unallocated)
    private Tensor _hiddenK = default!;        // [K * embDim]
    private Tensor _residualK = default!;      // [K * embDim]
    private Tensor _normK = default!;          // [K * embDim]
    private Tensor _qK = default!;             // [K * numHeads * headDim]
    private Tensor _kK = default!;             // [K * numKvHeads * headDim]
    private Tensor _vK = default!;             // [K * numKvHeads * headDim]
    private Tensor _attnOutK = default!;       // [K * numHeads * headDim]
    private Tensor _ffnGateK = default!;       // [K * ffnScratchDim]
    private Tensor _ffnUpK = default!;         // [K * ffnScratchDim]
    private Tensor _logitsK = default!;        // [K * vocabSize]
    private float[]? _logitsKBuf;              // host download buffer [K * vocabSize]

    /// <summary>
    /// Whether <see cref="BatchVerify"/> can run on this Vulkan dense full-offload pass. Gated to
    /// the dense path (non-Gemma-4 — its per-layer head_dim / SWA rings / shared-KV / softcap need
    /// a separate batched path, a later PR — and non-TurboQuant, whose ring bookkeeping breaks the
    /// contiguous [startPos, startPos+k) layout) with an uncompacted cache. Mirrors
    /// <see cref="CudaForwardPass.SupportsBatchVerify"/>: a CONFIGURED SnapKV budget does not
    /// disable verify — only an actual prefill-time eviction does (then physical slot != logical
    /// position and the cache geometry the K-loop relies on no longer holds), so this flips false
    /// after such a prefill and the speculative decoder (which re-checks per step) degrades to
    /// sequential verify.
    /// </summary>
    public bool SupportsBatchVerify => !_isGemma4 && !_tqEnabled && !_kvEvicted;

    /// <summary>
    /// Batched k-token verify for single-user speculative decoding (issue #308): processes
    /// <paramref name="tokens"/> over the cache starting at <paramref name="startPos"/> (the cache
    /// must hold exactly <paramref name="startPos"/> positions), returning <c>result[i]</c> = logits
    /// after <c>tokens[i]</c>. All k K/V entries are appended at contiguous positions
    /// [<paramref name="startPos"/>, <paramref name="startPos"/> + k); the caller rewinds rejected
    /// tokens via <see cref="TruncateTo"/>.
    /// <para>The payoff path (issue #308 PR1c): when <see cref="_canBatchedTrunk"/> (a dense model
    /// whose every trunk matmul weight is Q4_K or Q6_K), this dispatches to
    /// <see cref="BatchVerifyBatched"/>, which streams the K draft tokens through ONE command buffer
    /// and reads each weight matrix from VRAM exactly once via <c>MatMulBatched</c> (the
    /// weight-amortization). Otherwise (mixed/other weight dtype) it falls back to
    /// <see cref="BatchVerifyKLoop"/>, the bit-exact K-sequential-<see cref="Forward"/> reference.
    /// k == 1 short-circuits to a single <see cref="Forward"/> (mirrors CUDA). The batched path is
    /// bit-exact to the K-loop by construction (<c>MatMulBatched</c> is bit-identical to single-row
    /// matvec, the gather/scatter copies are exact, and the per-token RmsNorm/QK-norm/RoPE/append/
    /// attention reuse the single-query shaders with the same positions/seqLens).</para>
    /// </summary>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (!SupportsBatchVerify)
            throw new NotSupportedException(
                "BatchVerify requires a dense (non-Gemma-4, non-TurboQuant) Vulkan pass with an " +
                "uncompacted cache. Check SupportsBatchVerify before calling.");
        int k = tokens.Length;
        if (k == 0) return Array.Empty<float[]>();
        if (startPos < 0 || startPos + k > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos),
                $"BatchVerify range [{startPos}, {startPos + k}) exceeds the context window (maxSeqLen={_maxSeqLen}).");

        // k == 1 has nothing to amortize — one Forward is strictly cheaper than the batched
        // gather/scatter machinery (and bit-identical). Mirrors CudaForwardPass.BatchVerify.
        if (k == 1)
        {
            TruncateTo(startPos);
            return [Forward(tokens[0], startPos).ToArray()];
        }

        if (_canBatchedTrunk && k <= MaxBatchVerifyK)
            return BatchVerifyBatched(tokens, startPos);

        return BatchVerifyKLoop(tokens, startPos);
    }

    /// <summary>
    /// FOUNDATION K-loop reference: loops the single-query <see cref="Forward"/> k times. Establishes
    /// the contiguous-append semantics and the rollback contract; bit-identical to k sequential
    /// <see cref="Forward"/> calls by construction (it IS those calls). Used as the fallback when the
    /// model's trunk weights are not all Q4_K/Q6_K, and as the parity oracle for the batched path.
    /// </summary>
    private float[][] BatchVerifyKLoop(int[] tokens, int startPos)
    {
        int k = tokens.Length;
        // The cache must hold exactly startPos positions; soft-truncate to make it so (the K-loop
        // then appends the k K/V rows over any stale rewound slots, position by position).
        TruncateTo(startPos);
        var result = new float[k][];
        for (int i = 0; i < k; i++)
            // Forward appends one K/V slot at startPos+i and returns logits-after-tokens[i]; the
            // returned span is reused across calls, so copy each. After the loop the cache holds
            // startPos+k positions — matching "all k K/V entries appended".
            result[i] = Forward(tokens[i], startPos + i).ToArray();
        return result;
    }

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

    // Flash-decoding split-KV (issue #312): OPT-IN, DEFAULT-OFF via SHARPI_VULKAN_SPLIT_DECODE.
    // When enabled (and maxSeqLen > 4096 with a representable split count), the fp32 decode
    // attention parallelizes its long-context KV scan across numHeads × nSplits workgroups and
    // LSE-merges the partials, mirroring CUDA's SHARPI_SPLIT_DECODE. Default-ON (measured ~2×
    // decode at 10.5K ctx vs the VRAM score-spill; the win grows with context); the ≤4096 path
    // is untouched. SHARPI_VULKAN_SPLIT_DECODE=0 reverts to the spill path (kill switch).
    private readonly bool _splitKvEnabled =
        Environment.GetEnvironmentVariable("SHARPI_VULKAN_SPLIT_DECODE") != "0";
    private Tensor? _splitKvPartialO;   // [numHeads * maxSplits * headDim] un-normalized weighted-V numerators
    private Tensor? _splitKvPartialMeta; // [numHeads * maxSplits * 2] (m_i, l_i) per (head, split)
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
    // True once a prefill-time SnapKV eviction has actually compacted the cache, so the
    // physical slot layout no longer equals the logical position. Mirrors CudaForwardPass's
    // `_kvEvictedCount > 0` guard: a CONFIGURED-but-unevicted budget keeps this false (and
    // BatchVerify available). Set in ApplySnapKvEviction, reset in ResetCache.
    private bool _kvEvicted;

    // KV-cache store dtype (issues #311 / #325). Float32 (default) keeps the original fp32 cache;
    // BFloat16 selects the half-width KV path (stored as IEEE fp16 packed two-per-uint — more
    // precise than bf16 for the small KV magnitudes, and packHalf2x16 is core GLSL so no device
    // extension is needed); Q8_0 selects the block-quantized store (34 bytes per 32 elements,
    // ~4× smaller than fp32) added in #325. Arithmetic everywhere stays fp32 — only the stored
    // value is narrowed.
    private readonly DType _kvDType;

    // ── Gemma 4 (issue #309) ────────────────────────────────────────────────
    // The whole gemma4 trunk is dispatched from ForwardGemma4 when _isGemma4. The
    // per-layer head_dim varies (256 SWA / 512 global), so _maxHeadDim sizes the
    // Q/K/V/attnOut scratch for the widest layer and per-layer Tensor VIEWS carve
    // out the active rows. This gemma4-v2 variant has PLE disabled
    // (embedding_length_per_layer_input=0) and shared_kv_layers=0 (no KV aliasing),
    // so neither path is wired here.
    private readonly bool _isGemma4;
    private readonly int _maxHeadDim;
    private readonly float _ropeThetaSwa;
    // Per-layer post-norms (sandwich norm), per-head Q/K norm, per-layer output scale.
    private readonly Tensor[]? _wPostAttnNorm;
    private readonly Tensor[]? _wPostFfwNorm;
    private readonly Tensor[]? _wQNormG4;   // per-layer attn_q_norm (gemma4; [layerHd])
    private readonly Tensor[]? _wKNormG4;   // per-layer attn_k_norm (gemma4; [layerHd])
    private readonly float[]? _layerOutputScale;
    // Optional rope_freqs.weight table (size = maxHeadDim/2), applied on global layers only.
    private readonly Tensor? _gpuRopeFreqs;

    public GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int maxContextLength = 0, bool enableTurboQuant = false, int tqFp32Window = 256, int tqBits = 3,
        DType kvDtype = DType.Float32)
    {
        // Gemma 4 master switch: hp.LayerHeadDim is non-null only for gemma4-family models.
        // The full gemma4 trunk (per-layer head_dim, SWA, dual RoPE + rope_freqs, sandwich
        // norms, V-norm, attn_scale=1.0, final softcap, k_eq_v globals) is implemented in
        // ForwardGemma4 / RunGemma4Layers (issue #309). PLE and shared-KV are NOT wired (this
        // gemma4-v2 variant has both disabled); reject up front if a future GGUF enables them.
        _isGemma4 = hp.LayerHeadDim is not null;
        if (_isGemma4)
        {
            if (hp.HasPerLayerTokenEmbd)
                throw new NotSupportedException(
                    "Gemma 4 PLE (embedding_length_per_layer_input > 0) is not implemented on the " +
                    "Vulkan backend (issue #309 scope: PLE-disabled gemma4 only). Use CUDA (-g) or CPU.");
            if (hp.KvSourceLayer is { } ksl)
                for (int i = 0; i < ksl.Count; i++)
                    if (ksl[i] >= 0)
                        throw new NotSupportedException(
                            "Gemma 4 shared-KV layers (shared_kv_layers > 0) are not implemented on the " +
                            "Vulkan backend (issue #309 scope). Use CUDA (-g) or CPU.");
            if (enableTurboQuant)
                throw new NotSupportedException("TurboQuant is not supported for Gemma 4 on Vulkan.");
            if (kvDtype != DType.Float32)
                throw new NotSupportedException(
                    "Gemma 4 on Vulkan supports only fp32 KV (the narrowed-KV append/attention shaders " +
                    "are not wired for per-layer head_dim / SWA). Use --kv-type fp32.");
        }

        _model = model;
        _gpu = gpu;
        _hp = hp;
        _tqEnabled = enableTurboQuant;
        _kvDType = kvDtype;

        // Per-layer-max head_dim (gemma4: 256 SWA / 512 global). Mirrors CudaForwardPass.
        _maxHeadDim = hp.HeadDim;
        if (hp.LayerHeadDim is { } lhdMax)
            for (int i = 0; i < hp.NumLayers; i++)
                if (lhdMax[i] > _maxHeadDim) _maxHeadDim = lhdMax[i];
        _ropeThetaSwa = hp.RopeThetaSwa;

        // Issues #311 / #325: the Vulkan KV cache supports fp32, a half-width (bf16) store, and a
        // block-quantized q8_0 store. Reject anything else up front rather than silently mis-store.
        bool kvNarrowed = _kvDType != DType.Float32;
        if (kvNarrowed && _kvDType is not (DType.BFloat16 or DType.Q8_0))
            throw new NotSupportedException(
                "Vulkan KV cache supports fp32, bf16, and q8_0 only (issue #325).");
        // The bf16 path packs two fp16 per uint, so each KV head must START on a uint-word
        // boundary for the AttentionBf16 two-at-a-time reads (issue #324) to address words
        // correctly: a head begins at kv_head*head_dim, which is word-even iff head_dim itself
        // is even. True for all supported head dims (64/128/256), but enforce it so a future
        // odd-head_dim model fails loudly instead of corrupting KV reads.
        if (_kvDType == DType.BFloat16 && (hp.HeadDim & 1) != 0)
            throw new NotSupportedException(
                $"bf16 KV requires an even head dimension; got {hp.HeadDim}. " +
                "Use fp32 KV (issue #324).");
        // The q8_0 path quantizes per 32-element block (one thread per block), so a KV row must
        // be a multiple of 32 for blocks to align to row boundaries (no straddling).
        if (_kvDType == DType.Q8_0 && ((hp.NumKvHeads * hp.HeadDim) & 31) != 0)
            throw new NotSupportedException(
                $"q8_0 KV requires kvDim % 32 == 0; got {hp.NumKvHeads}*{hp.HeadDim}. " +
                "Use fp32 or bf16 (issue #325).");
        // bf16 KV narrows the store the same way TurboQuant does; the two are mutually
        // exclusive (TQ owns the KV quantization).
        if (kvNarrowed && _tqEnabled)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE={_kvDType} + TurboQuant is not supported on Vulkan " +
                "(TQ owns the KV quantization). Use one or the other (issue #311).");

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
            _maxSeqLen = EstimateMaxContext(model, gpu, hp, _kvDType);
        }

        _tqFp32Window = enableTurboQuant ? Math.Min(tqFp32Window, _maxSeqLen) : 0;
        _tqBlockBytes = enableTurboQuant ? TurboQuantOps.BlockSize(tqBits, hp.HeadDim) : 0;

        // Bookkeeping-only: KV lives in GPU buffers, this tracks only the position counter.
        // Allocating the full host K/V buffers is pure waste (tens of GB at long ctx, #179).
        _kvCache = Engine.KvCache.CreateBookkeepingOnly(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.HeadDim);
        string kvTag = _kvDType switch
        {
            DType.BFloat16 => " [KV bf16]",
            DType.Q8_0 => " [KV q8_0]",
            _ => "",
        };
        Console.Error.WriteLine($"[GpuForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength})" +
            $"{(enableTurboQuant ? " [TQ3]" : "")}{kvTag}");

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
        // Issues #311 / #325: SnapKV's score/compact shaders read the cache as fp32 floats, so
        // they would read garbage from a narrowed buffer (fp16-packed for bf16, block-quantized
        // for q8_0). Narrowed KV + SnapKV is not yet wired; reject an explicit budget and force
        // the auto path off (the narrowed store already shrinks the KV footprint SnapKV chases).
        if (kvNarrowed && _snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE={_kvDType} + SnapKV is not yet implemented on Vulkan " +
                "(SnapKvScore/KvCompact read the cache as fp32, but the narrowed store packs it; " +
                "issue #325). Set SHARPI_SNAPKV_BUDGET=0 to disable SnapKV.");
        if (_isGemma4)
        {
            // SnapKV cannot compose with Gemma 4: its score/compaction shaders use the uniform
            // _numHeads/_headDim, but gemma4 caches are per-layer-dimensioned (head_dim 256 SWA /
            // 512 global, per-layer KV heads), so eviction would mis-index and corrupt the cache.
            // Force it off (mirrors CudaForwardPass); warn only if a budget was explicitly set.
            if (_snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0)
                Console.Error.WriteLine(
                    "[GpuForwardPass] SnapKV is not supported for Gemma 4 (per-layer head_dim / SWA); " +
                    "ignoring the budget and using the full KV cache.");
            _snapKvEffectiveBudget = 0;
        }
        else if (_snapKvCfg.IsBudgetExplicit)
        {
            _snapKvEffectiveBudget = _snapKvCfg.Budget;
        }
        else if (_tqEnabled)
        {
            _snapKvEffectiveBudget = 0;
        }
        else if (kvNarrowed)
        {
            // A narrowed store (bf16/q8_0) already shrinks the KV footprint (the memory win SnapKV
            // auto-enable chases), and the compaction shaders read fp32; don't auto-enable
            // eviction on the packed cache (issues #311 / #325).
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
        // Sized for the widest layer head_dim so gemma4 per-layer view tensors (256 SWA /
        // 512 global) can carve out the active rows. Identity to _headDim for non-gemma4.
        _q = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
        _k = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _v = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _attnOut = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
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
        else if (_kvDType == DType.BFloat16)
        {
            // Half-width KV (issue #311): allocate the cache as BFloat16 so byte accounting is
            // honest (2 bytes/elem = the packed-fp16-word byte count). The append/attention
            // shaders bind the buffer as uint[] (2 fp16 per word) regardless of the declared
            // dtype, and index it identically to the fp32 path (no ring modulo).
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), DType.BFloat16);
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), DType.BFloat16);
            }
        }
        else if (_kvDType == DType.Q8_0)
        {
            // Block-quantized KV (issue #325): the cache holds ggml block_q8_0 (34 bytes per 32
            // elements). Vulkan's Allocate(shape, DType) computes bytes via BytesPerElement, which
            // THROWS for quantized dtypes, so allocate a raw uint-word buffer sized to the exact
            // q8_0 byte count (rounded up to whole words). The append/attention shaders bind it as
            // uint[] and index it by block regardless of the declared dtype — same pattern as bf16.
            long q8Bytes = DTypeInfo.ByteSize((long)_maxSeqLen * kvDim, DType.Q8_0); // (count/32)*34
            long words = (q8Bytes + 3) / 4;
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1(words));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1(words));
            }
        }
        else if (_isGemma4)
        {
            // Gemma 4: per-layer KV geometry (8 GQA on SWA at head_dim 256, 1 MQA on global at
            // head_dim 512). The Vulkan KvAppend shader has NO ring modulo, so EVERY layer —
            // SWA included — is allocated at FULL context (more VRAM than CUDA's SWA ring, but
            // correct). Mirrors FillKvCacheArrays minus the ring/alias (shared_kv=0 here). fp32
            // is enforced for gemma4 above.
            for (int i = 0; i < hp.NumLayers; i++)
            {
                int layerHd = hp.LayerHeadDim![i];
                int layerKv = hp.LayerKvHeads is { } lkv ? lkv[i] : _numKvHeads;
                long layerKvDim = (long)layerKv * layerHd;
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * layerKvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * layerKvDim));
            }
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

        // Flash-decoding split-KV partial buffers (issue #312), allocated ONLY when the opt-in
        // gate is set, the context exceeds the 4096-slot shared-memory fast path, and the split
        // count fits the combine shader's 256-split shared array (CHUNK=512 ⇒ maxSplits =
        // ceil(maxSeqLen/512); the combine's sh_scale[256] caps it at 256). Otherwise we leave
        // these null and the decode falls back to the spill path. headDim%32==0 is also required
        // at dispatch time (the gate below) but does not affect buffer sizing.
        // maxSplits <= 256 (combine shader bound) ⇔ maxSeqLen <= 131072; bound it directly
        // so the (_maxSeqLen + 511) split-count math can't overflow on a pathological ctx.
        if (_splitKvEnabled && _maxSeqLen > 4096 && _maxSeqLen <= 131072)
        {
            int maxSplits = (_maxSeqLen + 511) / 512;
            _splitKvPartialO = gpu.Allocate(TensorShape.D1((long)_numHeads * maxSplits * _headDim));
            _splitKvPartialMeta = gpu.Allocate(TensorShape.D1((long)_numHeads * maxSplits * 2));
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

        // Gemma 4: per-layer post-norms (sandwich norm), per-head Q/K norm (gemma4 keeps these in
        // their own arrays so the per-layer-head_dim HeadNorm dims are unambiguous), and the
        // per-layer scalar output gain. PLE is disabled in this variant.
        if (_isGemma4)
        {
            if (hp.HasPostAttnNorm) _wPostAttnNorm = new Tensor[L];
            if (hp.HasPostFfwNorm) _wPostFfwNorm = new Tensor[L];
            if (hp.HasLayerOutputScale) _layerOutputScale = new float[L];
            if (_hasQkNorm) { _wQNormG4 = new Tensor[L]; _wKNormG4 = new Tensor[L]; }
        }

        Console.Error.Write($"[GpuForwardPass] Uploading {L} layers to VRAM...");
        for (int i = 0; i < L; i++)
        {
            _wAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _wq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _wk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            // Gemma 4 12B global layers omit attn_v (attention_k_eq_v): V reuses the raw K
            // projection — leave _wv[i] null and copy K→V at runtime. All other models always
            // carry attn_v. Mirrors the CUDA upload probe.
            if (!_isGemma4 || model.FindTensor($"blk.{i}.attn_v.weight") is not null)
                _wv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _wo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _wFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");

            if (_isGemma4)
            {
                if (_wPostAttnNorm is not null)
                    _wPostAttnNorm[i] = UploadWeight($"blk.{i}.post_attention_norm.weight");
                if (_wPostFfwNorm is not null)
                    _wPostFfwNorm[i] = UploadWeight($"blk.{i}.post_ffw_norm.weight");
                if (_layerOutputScale is not null)
                    _layerOutputScale[i] = LoadScalarF32($"blk.{i}.layer_output_scale.weight");
            }
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
                if (_isGemma4)
                {
                    // Gemma 4 has per-head Q/K norm too, but the weight is [layerHd] so it lives in
                    // a separate array indexed at the layer's head_dim (256 SWA / 512 global).
                    _wQNormG4![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                    _wKNormG4![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
                }
                else
                {
                    _wqNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                    _wkNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
                }
            }

            Console.Error.Write(".");
        }
        // Upload embedding table to VRAM — keep raw-quantized for Q4_K/Q6_K to save VRAM
        // (the F32 dequant of a [3840, 262144] Q6_K table would burn ~4 GB; raw stays ~787 MiB).
        Console.Error.Write(" emb...");
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        if (embInfo.DType is DType.Q4_K or DType.Q6_K)
        {
            // Upload raw quantized bytes (reinterpret as uint32 for storage buffer)
            var embData = model.GetTensorData(embInfo);
            int floatCount = embData.Length / 4;
            var raw = new float[floatCount];
            embData.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
            _gpuEmbedding = gpu.Upload(raw, TensorShape.D1(floatCount));
            _embDType = embInfo.DType;
            _weightDTypes[_gpuEmbedding.Handle] = embInfo.DType;
        }
        else
        {
            // Small vocab or F32: dequantize to F32
            var embData = model.GetTensorData(embInfo);
            var embF32 = new float[(int)embInfo.ElementCount];
            Dequantize.ToFloat32(embData, embF32, embInfo.DType, embInfo.ElementCount);
            _gpuEmbedding = gpu.Upload(embF32, TensorShape.D1(embF32.Length));
            _embDType = DType.Float32;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Float32;
        }

        _wOutputNorm = UploadWeight("output_norm.weight");
        _wOutput = model.FindTensor("output.weight") is not null
            ? UploadWeight("output.weight")
            : _gpuEmbedding;

        // Gemma 4: optional `rope_freqs.weight` (size = maxHeadDim/2) masks the global-layer RoPE
        // high-frequency tail (~identity for long context). CPU bakes this into its precomputed
        // RoPE table; here it's applied live via RoPEWithFactors on non-SWA layers. Uploaded as a
        // plain F32 tensor (matches CUDA's UploadWeight + RoPEWithFactors path).
        if (_isGemma4
            && model.FindTensor("rope_freqs.weight") is GgufTensorInfo rfInfo
            && rfInfo.DType == DType.Float32
            && rfInfo.ElementCount == _maxHeadDim / 2)
        {
            _gpuRopeFreqs = UploadWeight("rope_freqs.weight");
        }

        // Batched-trunk verify gate (issue #308 PR1c): only when this is a dense model and every
        // trunk matmul weight (Q/K/V/O + gate/up/down per layer, plus the lm-head) is Q4_K or
        // Q6_K — the dtypes MatMulBatched amortizes. Gemma 4 uses ForwardGemma4 (a separate batched
        // path is a later PR) so it never qualifies. Computed here, after all weights are uploaded
        // and _weightDTypes is populated.
        _canBatchedTrunk = ComputeCanBatchedTrunk();

        Console.Error.WriteLine(" done.");
    }

    /// <summary>
    /// Determine <see cref="_canBatchedTrunk"/>: dense (non-MoE, non-Gemma-4) with every trunk
    /// matmul weight in {Q4_K, Q6_K}. Reads the per-weight dtype recorded at upload in
    /// <see cref="_weightDTypes"/> (defaulting to Q4_K matches <see cref="GpuMatMul"/>).
    /// </summary>
    private bool ComputeCanBatchedTrunk()
    {
        // MoE (mid-trunk router submit), gemma4 (per-layer dims), and TQ are excluded.
        // QKV/output bias is excluded too: the batched trunk wires a bias gather/add/scatter
        // path but no local Q4_K bias model (e.g. a Q4 Qwen2) exercises it yet, so keep bias
        // models on the verified K-loop fallback until a parity test covers that path.
        if (_isMoE || _isGemma4 || _hasAttnBias || _hasAttnOutputBias) return false;

        bool IsBatchable(Tensor w) =>
            _weightDTypes.GetValueOrDefault(w.Handle, DType.Q4_K) is DType.Q4_K or DType.Q6_K;

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            if (!IsBatchable(_wq[i]) || !IsBatchable(_wk[i]) || !IsBatchable(_wv[i]) ||
                !IsBatchable(_wo[i]) || !IsBatchable(_wGate[i]) || !IsBatchable(_wUp[i]) ||
                !IsBatchable(_wDown[i]))
                return false;
        }
        return IsBatchable(_wOutput);
    }

    public Engine.KvCache Cache => _kvCache;
    public int KvLength => _kvLength;

    public void ResetCache()
    {
        _kvLength = 0;
        _tqCompressedLen = 0;
        _fp32WriteIdx = 0;
        _fp32Count = 0;
        _kvEvicted = false; // fresh sequence — standard sequential slot==position layout restored
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
        // The compaction broke the logical-position ↔ physical-slot identity, so the
        // contiguous-position assumption BatchVerify (and the future batched matvec) rely on
        // no longer holds. Flip the verify gate off, like CudaForwardPass's _kvEvictedCount.
        _kvEvicted = true;
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
        if (_isGemma4)
            return ForwardGemma4(token, position);

        // Record ALL dispatches into ONE command buffer
        _gpu.BeginRecord();

        // Embed token (GPU lookup from cached table — no PCIe transfer)
        DispatchEmbedLookup(token);
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
            else if (_kvDType == DType.BFloat16)
            {
                // Half-width KV (issue #311): same args as the fp32 path; the bf16 shaders
                // store/read the cache as fp16-packed uint[] but index it identically.
                _gpu.KvAppendBf16(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
                _gpu.RecordBarrier();

                // Flash-decoding split-KV (issue #332): bf16 cache, same gate as the fp32 path
                // (#312). Below it / split-disabled / headDim%32!=0 runs the byte-identical
                // single-workgroup bf16 spill path. The combine pass is dtype-agnostic.
                if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                {
                    _gpu.AttentionSplitKvBf16(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _splitKvPartialO, _splitKvPartialMeta!,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
                else
                {
                    _gpu.AttentionBf16(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _attnScoresScratch,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
            }
            else if (_kvDType == DType.Q8_0)
            {
                // Block-quantized KV (issue #325): same args as the fp32/bf16 paths; the q8_0
                // shaders quantize/dequantize the cache (block_q8_0) but index it identically.
                _gpu.KvAppendQ8_0(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
                _gpu.RecordBarrier();

                // Flash-decoding split-KV (issue #332): q8_0 cache, same gate as the fp32 path
                // (#312). q8_0 already guarantees kv_dim%32==0 so headDim%32==0 holds, but the
                // gate is kept uniform. Below it / split-disabled runs the byte-identical q8_0
                // single-workgroup spill path. The combine pass is dtype-agnostic.
                if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                {
                    _gpu.AttentionSplitKvQ8(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _splitKvPartialO, _splitKvPartialMeta!,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
                else
                {
                    _gpu.AttentionQ8_0(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _attnScoresScratch,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
            }
            else
            {
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
                _gpu.RecordBarrier();

                // Flash-decoding split-KV (issue #312, OPT-IN): only past the 4096-slot
                // shared-memory fast path (where the single-workgroup scan collapses), with
                // headDim a multiple of 32 (matches the CUDA gate) and the partial buffers
                // allocated (gate + maxSplits≤256). Otherwise the byte-identical spill path runs.
                if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                {
                    _gpu.AttentionSplitKv(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _splitKvPartialO, _splitKvPartialMeta!,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
                else
                {
                    _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _attnScoresScratch,
                        (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                        (uint)(position + 1), (uint)_maxSeqLen, window: 0u);
                }
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
    //  Gemma 4 (issue #309)
    // ================================================================

    /// <summary>
    /// Run one token through the Gemma 4 transformer on Vulkan. The faithful mirror of
    /// <see cref="CudaForwardPass"/>'s <c>ForwardGemma4</c> + <c>RunGemma4DeviceRegion</c>:
    /// per-layer head_dim (256 SWA / 512 global), per-head Q/K norm + plain V-norm, dual RoPE
    /// (global theta + rope_freqs vs SWA theta), attention_scale = 1.0 (via a √head_dim Q
    /// prescale so the shader's 1/√head_dim cancels), SWA windowing, k_eq_v global layers
    /// (V = raw K projection), sandwich norm (post-attn / post-ffw RMSNorm BEFORE the residual
    /// add), GELU-tanh FFN, per-layer output scale, and the final-logit softcap. PLE and
    /// shared-KV are disabled in this variant (rejected at construction). All dispatches are
    /// recorded into one command buffer and submitted once, like the dense Forward.
    /// </summary>
    private ReadOnlySpan<float> ForwardGemma4(int token, int position)
    {
        _gpu.BeginRecord();

        // 1. Embedding lookup → scale by sqrt(embDim).
        EmbedTokenGemma4(token);
        _gpu.RecordBarrier();
        if (_hp.EmbeddingScale != 1f)
        {
            _gpu.ScaleInPlace(_hidden, _hp.EmbeddingScale);
            _gpu.RecordBarrier();
        }

        // 2. Transformer layers.
        RunGemma4Layers(position);

        // 3. Final norm + output projection + softcap.
        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_logits, _wOutput, _hidden);
        if (_hp.FinalLogitSoftcap > 0f)
        {
            _gpu.RecordBarrier();
            _gpu.SoftcapInPlace(_logits, _hp.FinalLogitSoftcap);
        }

        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_logits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Gemma 4 only: image input (issue #252) splices vision soft tokens into the decode
    /// stream. Other arches and the partial-offload hybrids don't implement the seam.
    /// </remarks>
    public bool SupportsEmbeddingInput => _isGemma4;

    /// <summary>
    /// Forward one position from a PRECOMPUTED embedding (a vision soft token) instead of a
    /// token-table lookup. The Vulkan mirror of <see cref="ForwardGemma4"/> (and of
    /// <c>ForwardPass.ForwardEmbedding</c> / <c>CudaForwardPass.ForwardEmbedding</c>): the
    /// supplied embedding is uploaded into the device hidden buffer and — per the
    /// <see cref="IForwardPass.ForwardEmbedding"/> contract and gemma4.cpp:182 — the
    /// sqrt(EmbeddingDim) embedding scale is SKIPPED (the soft tokens arrive already final).
    /// The transformer trunk + final norm/output/softcap are identical to
    /// <see cref="ForwardGemma4"/>. Gemma 4 only (PLE/shared-KV are rejected at construction
    /// on this backend, so there is no per-layer-projection pre-pass to mirror). The upload
    /// runs BEFORE <c>BeginRecord</c> because <see cref="UploadToExisting"/> owns its own
    /// command-buffer begin/submit/wait and cannot be nested inside an in-progress record.
    /// </summary>
    public ReadOnlySpan<float> ForwardEmbedding(ReadOnlySpan<float> embedding, int position)
    {
        if (!_isGemma4)
            throw new NotSupportedException(
                "ForwardEmbedding (image input) is only supported for Gemma 4 on the Vulkan backend.");
        if (embedding.Length != _embDim)
            throw new ArgumentException(
                $"embedding length {embedding.Length} != model embedding dim {_embDim}.");

        // 1. Upload the precomputed embedding into _hidden. No sqrt(d) scale (gemma4.cpp:182).
        //    Done OUTSIDE the record block: UploadToExisting submits + waits its own command
        //    buffer, so it cannot be nested between BeginRecord and EndRecordAndSubmit.
        UploadToExisting(_hidden, embedding);

        // 2. Transformer layers + final norm/output/softcap (same device region as text decode).
        _gpu.BeginRecord();
        // The upload above is a transfer write in a prior (fence-waited) submission; insert a
        // transfer→compute barrier so the first layer's _hidden read is ordered after it and the
        // write is made visible to the shader (host fence-wait alone doesn't guarantee visibility
        // to this submission's compute reads). The barrier's first scope includes the earlier
        // transfer submission in queue order.
        _gpu.RecordTransferBarrier();
        RunGemma4Layers(position);

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_logits, _wOutput, _hidden);
        if (_hp.FinalLogitSoftcap > 0f)
        {
            _gpu.RecordBarrier();
            _gpu.SoftcapInPlace(_logits, _hp.FinalLogitSoftcap);
        }

        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_logits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    /// <summary>Gemma 4 embedding gather. Mirrors the CUDA path, but Vulkan ships only
    /// F32 + Q4_K + Q6_K embed-lookup shaders (the Q4_K_M tied embedding is Q4_K, the Q6_K
    /// variant stays raw via EmbedLookupQ6K; any other quant is dequantized to F32 at
    /// upload, taking the EmbedLookup path here).</summary>
    private void EmbedTokenGemma4(int token) => DispatchEmbedLookup(token);

    /// <summary>
    /// Dispatch the embedding lookup for <paramref name="token"/> into <see cref="_hidden"/>,
    /// choosing the shader by the embedding table's dtype. Q4_K and Q6_K tables are kept raw
    /// in VRAM (see embedding-upload branch); everything else was dequantized to F32 at upload.
    /// </summary>
    private void DispatchEmbedLookup(int token)
    {
        switch (_embDType)
        {
            case DType.Q4_K:
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
                break;
            case DType.Q6_K:
                _gpu.EmbedLookupQ6K(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
                break;
            default:
                _gpu.EmbedLookup(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
                break;
        }
    }

    /// <summary>
    /// The per-layer Gemma 4 trunk, mirroring CUDA's <c>RunGemma4DeviceRegion</c> op-for-op.
    /// Records into the in-progress command buffer (caller owns BeginRecord/EndRecordAndSubmit).
    /// </summary>
    private void RunGemma4Layers(int position)
    {
        int L = _hp.NumLayers;
        for (int layer = 0; layer < L; layer++)
        {
            int layerHd = _hp.LayerHeadDim![layer];
            int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
            int qDimL = _numHeads * layerHd;
            int kvDimL = layerKv * layerHd;
            bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];
            // Gemma 4 12B global layers carry no attn_v: V reuses the raw K projection
            // (attention_k_eq_v). These layers always own their KV (shared_kv_layers=0).
            bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer] is null;

            // Per-layer view tensors so each dispatch addresses exactly the active rows.
            var qView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _q.Handle);
            var kView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _k.Handle);
            var vView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _v.Handle);
            var attnOutView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _attnOut.Handle);

            // a. attn_norm.
            CopyBuffer(_residual, _hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // b. Q/K/V projections (read normBuf; no conflict between them).
            GpuMatMul(qView, _wq[layer], _normBuf);
            GpuMatMul(kView, _wk[layer], _normBuf);
            _gpu.RecordBarrier();
            if (kEqV)
                _gpu.RecordComputeCopy(vView, kView); // V = raw K projection (pre-norm, pre-RoPE)
            else
                GpuMatMul(vView, _wv[layer]!, _normBuf);
            _gpu.RecordBarrier();

            // c. Per-head Q/K norm (gemma4: shared [layerHd] weight per head, BEFORE RoPE).
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.HeadNorm(qView, _wQNormG4![layer], (uint)_numHeads, (uint)layerHd,
                    _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.HeadNorm(kView, _wKNormG4![layer], (uint)layerKv, (uint)layerHd,
                    _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.RecordBarrier();
            }

            // d. V-norm (REQUIRED): plain per-head RMSNorm (no learned weight) on every
            //    KV-owning layer (E4B and 12B alike). V is never RoPE'd.
            _gpu.HeadNormPure(vView, (uint)layerKv, (uint)layerHd, _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // e. RoPE: global (non-SWA) layers use the rope_freqs table; SWA layers use plain
            //    NEOX RoPE at the SWA theta. Gemma uses NEOX/half rotation.
            float ropeTheta = isSwa ? _ropeThetaSwa : _hp.RopeTheta;
            if (!isSwa && _gpuRopeFreqs is { } rfTbl)
            {
                _gpu.RoPEWithFactors(qView, position, layerHd, ropeTheta, rfTbl);
                _gpu.RoPEWithFactors(kView, position, layerHd, ropeTheta, rfTbl);
            }
            else
            {
                _gpu.RoPE(qView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(kView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
            }
            _gpu.RecordBarrier();

            // f. KV append (full context — the Vulkan shader has no SWA ring modulo).
            _gpu.KvAppend(kView, vView, _gpuKCache[layer], _gpuVCache[layer],
                (uint)kvDimL, (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();

            // g. attn_scale = 1.0: gemma4 does NOT use 1/sqrt(head_dim). The shader divides the
            //    score by sqrt(head_dim), so pre-scale Q by sqrt(head_dim) to cancel it to 1.0.
            _gpu.ScaleInPlace(qView, MathF.Sqrt(layerHd));
            _gpu.RecordBarrier();

            // h. Attention. SWA layers pass the sliding-window bound; global layers pass 0
            //    (full causal). Base fp32 Attention (Q4_K-free; fp32 KV).
            _gpu.Attention(qView, _gpuKCache[layer], _gpuVCache[layer], attnOutView,
                _attnScoresScratch,
                (uint)_numHeads, (uint)layerKv, (uint)layerHd,
                (uint)(position + 1), (uint)_maxSeqLen,
                window: isSwa ? (uint)_hp.SlidingWindowSize : 0u);
            _gpu.RecordBarrier();

            // i. Output projection: _wo[layer] is [embDim, qDimL]; attnOutView has qDimL rows.
            GpuMatMul(_hidden, _wo[layer], attnOutView);
            _gpu.RecordBarrier();

            // j. Sandwich norm: post-attn RMSNorm on the O output BEFORE the residual add.
            if (_wPostAttnNorm is not null)
            {
                _gpu.RmsNorm(_hidden, _hidden, _wPostAttnNorm[layer], _hp.RmsNormEps);
                _gpu.RecordBarrier();
            }
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();

            // k. FFN: ffn_norm → gate/up → GELU-tanh → down.
            CopyBuffer(_residual, _hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();
            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp, _wUp[layer], _normBuf);
            _gpu.RecordBarrier();
            _gpu.GeluTanhMul(_ffnGate, _ffnUp);
            _gpu.RecordBarrier();
            GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            _gpu.RecordBarrier();

            // l. Sandwich norm: post-ffw RMSNorm on the down output BEFORE the residual add.
            if (_wPostFfwNorm is not null)
            {
                _gpu.RmsNorm(_hidden, _hidden, _wPostFfwNorm[layer], _hp.RmsNormEps);
                _gpu.RecordBarrier();
            }
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();

            // m. Per-layer scalar output gain.
            if (_layerOutputScale is not null)
            {
                _gpu.ScaleInPlace(_hidden, _layerOutputScale[layer]);
                _gpu.RecordBarrier();
            }
        }
    }

    /// <summary>Load a single F32 scalar tensor (any source dtype) into a managed float. Used for
    /// Gemma 4's per-layer <c>layer_output_scale.weight</c>. Mirrors the CUDA helper.</summary>
    private float LoadScalarF32(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);
        Span<float> buf = stackalloc float[1];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, 1).CopyTo(buf);
        else
            Dequantize.ToFloat32(data, buf, info.DType, 1);
        return buf[0];
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        _gpu.MatMul(output, weights, input, WeightDType(weights));
    }

    /// <summary>Recorded weight dtype (defaults to Q4_K to match <see cref="GpuMatMul"/>).</summary>
    private DType WeightDType(Tensor weights) =>
        _weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K);

    /// <summary>
    /// Batched trunk for k-token speculative verify (issue #308 PR1c). Processes all k draft tokens
    /// through ONE command buffer, reading each weight matrix from VRAM exactly once via
    /// <c>MatMulBatched</c> (the weight-amortization). The per-token reductions (RmsNorm, QK-norm,
    /// RoPE) and the position-dependent KV-append + causal attention run as a K-loop with
    /// gather/scatter to single-token temp buffers (the existing single-query scratch), while the
    /// position-independent elementwise ops (residual copy/add, SiLuMul) run once over the whole
    /// [K][dim] buffer. Bit-exact to <see cref="BatchVerifyKLoop"/> by construction.
    /// <para>Op ordering MIRRORS the single-query <see cref="Forward"/> exactly (the #157 QK-norm /
    /// RoPE ordering branches included). RoPE position for token i is startPos+i; the attention
    /// seqLen for token i is startPos+i+1 (causal among the k tokens). Only the non-TQ fp32 KV path
    /// is reachable here — <see cref="SupportsBatchVerify"/> excludes TurboQuant, and
    /// <see cref="_canBatchedTrunk"/> excludes Gemma-4/MoE; bf16/q8_0 KV stores still work (the
    /// per-token append/attention dispatch the matching narrowed shaders).</para>
    /// </summary>
    private float[][] BatchVerifyBatched(int[] tokens, int startPos)
    {
        int k = tokens.Length;
        EnsureBatchVerifyScratch(k);

        // Cache must hold exactly startPos positions; soft-truncate (the per-token KvAppend below
        // overwrites any stale rewound slots at [startPos, startPos+k)).
        TruncateTo(startPos);

        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int embDim = _embDim;
        const int f32 = sizeof(float);

        // Per-token single-token temps (reuse the single-query scratch buffers as gather targets).
        // _hidden/_normBuf are [embDim], _q is [numHeads*headDim], _k/_v [numKvHeads*headDim],
        // _attnOut [numHeads*headDim], _ffnGate/_ffnUp [ffnScratchDim].

        _gpu.BeginRecord();

        // ── Embed: K-loop lookup into the [K][embDim] hidden buffer. ──
        // DispatchEmbedLookup writes _hidden (offset 0); copy each token's row into _hiddenK[i].
        for (int i = 0; i < k; i++)
        {
            DispatchEmbedLookup(tokens[i]);
            _gpu.RecordBarrier();
            _gpu.RecordComputeCopyRegion(_hiddenK, (long)i * embDim * f32, _hidden, 0, (long)embDim * f32);
            _gpu.RecordBarrier();
        }

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // residualK = hiddenK (whole buffer, elementwise).
            _gpu.RecordComputeCopy(_residualK, _hiddenK);
            _gpu.RecordBarrier();

            // attn RmsNorm per token: hiddenK[i] → _hidden temp → RmsNorm → normK[i].
            for (int i = 0; i < k; i++)
            {
                _gpu.RecordComputeCopyRegion(_hidden, 0, _hiddenK, (long)i * embDim * f32, (long)embDim * f32);
                _gpu.RecordBarrier();
                _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
                _gpu.RecordBarrier();
                _gpu.RecordComputeCopyRegion(_normK, (long)i * embDim * f32, _normBuf, 0, (long)embDim * f32);
                _gpu.RecordBarrier();
            }

            // Q/K/V projections: batched (weight read once for all k tokens).
            _gpu.MatMulBatched(_qK, _wq[layer], _normK, k, WeightDType(_wq[layer]));
            _gpu.MatMulBatched(_kK, _wk[layer], _normK, k, WeightDType(_wk[layer]));
            _gpu.MatMulBatched(_vK, _wv[layer], _normK, k, WeightDType(_wv[layer]));
            _gpu.RecordBarrier();

            if (_hasAttnBias)
            {
                // Bias is per-channel and identical across tokens; replicate per token via views.
                for (int i = 0; i < k; i++)
                {
                    _gpu.RecordComputeCopyRegion(_q, 0, _qK, (long)i * qDim * f32, (long)qDim * f32);
                    _gpu.RecordComputeCopyRegion(_k, 0, _kK, (long)i * kvDim * f32, (long)kvDim * f32);
                    _gpu.RecordComputeCopyRegion(_v, 0, _vK, (long)i * kvDim * f32, (long)kvDim * f32);
                    _gpu.RecordBarrier();
                    _gpu.AddInPlace(_q, _bq![layer]);
                    _gpu.AddInPlace(_k, _bk![layer]);
                    _gpu.AddInPlace(_v, _bv![layer]);
                    _gpu.RecordBarrier();
                    _gpu.RecordComputeCopyRegion(_qK, (long)i * qDim * f32, _q, 0, (long)qDim * f32);
                    _gpu.RecordComputeCopyRegion(_kK, (long)i * kvDim * f32, _k, 0, (long)kvDim * f32);
                    _gpu.RecordComputeCopyRegion(_vK, (long)i * kvDim * f32, _v, 0, (long)kvDim * f32);
                    _gpu.RecordBarrier();
                }
            }

            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (layer + 1) % _hp.NoRopeLayerStep != 0;

            // Per-token QK-norm / RoPE / L2-norm (issue #157 ordering, mirrors Forward). Each op
            // reduces or is position-dependent per token, so gather → temp → op → scatter.
            for (int i = 0; i < k; i++)
            {
                int position = startPos + i;
                _gpu.RecordComputeCopyRegion(_q, 0, _qK, (long)i * qDim * f32, (long)qDim * f32);
                _gpu.RecordComputeCopyRegion(_k, 0, _kK, (long)i * kvDim * f32, (long)kvDim * f32);
                _gpu.RecordBarrier();

                if (_hasQkNorm && !_hp.UseL2QkNorm)
                {
                    _gpu.HeadNorm(_q, _wqNorm![layer], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.HeadNorm(_k, _wkNorm![layer], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.RecordBarrier();
                }

                if (useRoPE)
                {
                    _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    _gpu.RecordBarrier();
                }

                if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                {
                    _gpu.HeadNormPure(_q, (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.HeadNormPure(_k, (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
                    _gpu.RecordBarrier();
                }

                _gpu.RecordComputeCopyRegion(_qK, (long)i * qDim * f32, _q, 0, (long)qDim * f32);
                _gpu.RecordComputeCopyRegion(_kK, (long)i * kvDim * f32, _k, 0, (long)kvDim * f32);
                _gpu.RecordBarrier();
            }

            // KvAppend + Attention per token, INTERLEAVED so token i attends to [0, startPos+i]
            // (causal among the k tokens). seqLen = startPos+i+1 — bit-identical to k sequential
            // Forwards. Appends the post-RoPE k/v at slot startPos+i, then attends.
            for (int i = 0; i < k; i++)
            {
                int position = startPos + i;
                _gpu.RecordComputeCopyRegion(_q, 0, _qK, (long)i * qDim * f32, (long)qDim * f32);
                _gpu.RecordComputeCopyRegion(_k, 0, _kK, (long)i * kvDim * f32, (long)kvDim * f32);
                _gpu.RecordComputeCopyRegion(_v, 0, _vK, (long)i * kvDim * f32, (long)kvDim * f32);
                _gpu.RecordBarrier();

                BatchVerifyAppendAttend(layer, position);

                _gpu.RecordComputeCopyRegion(_attnOutK, (long)i * qDim * f32, _attnOut, 0, (long)qDim * f32);
                _gpu.RecordBarrier();
            }

            // O projection: batched (weight read once). hiddenK = Wo · attnOutK.
            _gpu.MatMulBatched(_hiddenK, _wo[layer], _attnOutK, k, WeightDType(_wo[layer]));
            _gpu.RecordBarrier();

            if (_hasAttnOutputBias)
            {
                for (int i = 0; i < k; i++)
                {
                    _gpu.RecordComputeCopyRegion(_hidden, 0, _hiddenK, (long)i * embDim * f32, (long)embDim * f32);
                    _gpu.RecordBarrier();
                    _gpu.AddInPlace(_hidden, _bo![layer]);
                    _gpu.RecordBarrier();
                    _gpu.RecordComputeCopyRegion(_hiddenK, (long)i * embDim * f32, _hidden, 0, (long)embDim * f32);
                    _gpu.RecordBarrier();
                }
            }

            // + residual (whole buffer), then residualK = hiddenK for the FFN.
            _gpu.AddInPlace(_hiddenK, _residualK);
            _gpu.RecordBarrier();
            _gpu.RecordComputeCopy(_residualK, _hiddenK);
            _gpu.RecordBarrier();

            // ffn RmsNorm per token.
            for (int i = 0; i < k; i++)
            {
                _gpu.RecordComputeCopyRegion(_hidden, 0, _hiddenK, (long)i * embDim * f32, (long)embDim * f32);
                _gpu.RecordBarrier();
                _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
                _gpu.RecordBarrier();
                _gpu.RecordComputeCopyRegion(_normK, (long)i * embDim * f32, _normBuf, 0, (long)embDim * f32);
                _gpu.RecordBarrier();
            }

            // gate/up: batched. SiLuMul over the whole [K][ffnDim] buffer. down: batched.
            _gpu.MatMulBatched(_ffnGateK, _wGate[layer], _normK, k, WeightDType(_wGate[layer]));
            _gpu.MatMulBatched(_ffnUpK, _wUp[layer], _normK, k, WeightDType(_wUp[layer]));
            _gpu.RecordBarrier();
            _gpu.SiLuMul(_ffnGateK, _ffnUpK); // [K*ffnDim] elementwise, K-agnostic
            _gpu.RecordBarrier();
            _gpu.MatMulBatched(_hiddenK, _wDown[layer], _ffnGateK, k, WeightDType(_wDown[layer]));
            _gpu.RecordBarrier();

            _gpu.AddInPlace(_hiddenK, _residualK);
            _gpu.RecordBarrier();
        }

        // Final norm per token + batched output projection → logitsK.
        for (int i = 0; i < k; i++)
        {
            _gpu.RecordComputeCopyRegion(_hidden, 0, _hiddenK, (long)i * embDim * f32, (long)embDim * f32);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.RecordComputeCopyRegion(_hiddenK, (long)i * embDim * f32, _hidden, 0, (long)embDim * f32);
            _gpu.RecordBarrier();
        }
        _gpu.MatMulBatched(_logitsK, _wOutput, _hiddenK, k, WeightDType(_wOutput));

        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_logitsK, _logitsKBuf!.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsKBuf);

        // The per-token KvAppend wrote slots [startPos, startPos+k); advance the length counter.
        _kvLength = Math.Max(_kvLength, startPos + k);

        // Split into k logit rows.
        int vocab = _hp.VocabSize;
        var result = new float[k][];
        for (int i = 0; i < k; i++)
        {
            var row = new float[vocab];
            Array.Copy(_logitsKBuf, (long)i * vocab, row, 0, vocab);
            result[i] = row;
        }
        return result;
    }

    /// <summary>
    /// KvAppend the single-token K/V in <see cref="_k"/>/<see cref="_v"/> at <paramref name="position"/>
    /// then run causal attention into <see cref="_attnOut"/> for that token (seqLen = position+1).
    /// Dispatches the matching KV-store shaders for the active <see cref="_kvDType"/> (fp32 / bf16 /
    /// q8_0) — identical to the single-query <see cref="Forward"/> non-TQ branches. TurboQuant is
    /// excluded by <see cref="SupportsBatchVerify"/>, so the TQ path is unreachable here.
    /// </summary>
    private void BatchVerifyAppendAttend(int layer, int position)
    {
        int kvDim = _numKvHeads * _headDim;
        uint seqLen = (uint)(position + 1);

        if (_kvDType == DType.BFloat16)
        {
            _gpu.KvAppendBf16(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                (uint)kvDim, (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();
            if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                _gpu.AttentionSplitKvBf16(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _splitKvPartialO, _splitKvPartialMeta!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
            else
                _gpu.AttentionBf16(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _attnScoresScratch,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
        }
        else if (_kvDType == DType.Q8_0)
        {
            _gpu.KvAppendQ8_0(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                (uint)kvDim, (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();
            if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                _gpu.AttentionSplitKvQ8(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _splitKvPartialO, _splitKvPartialMeta!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
            else
                _gpu.AttentionQ8_0(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _attnScoresScratch,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
        }
        else
        {
            _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                (uint)kvDim, (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();
            if (_splitKvEnabled && position + 1 > 4096 && _headDim % 32 == 0 && _splitKvPartialO is not null)
                _gpu.AttentionSplitKv(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _splitKvPartialO, _splitKvPartialMeta!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
            else
                _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _attnScoresScratch,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, seqLen, (uint)_maxSeqLen, window: 0u);
        }
        _gpu.RecordBarrier();
    }

    /// <summary>
    /// Lazily allocate the batched-verify [K][dim] scratch (and the host logits buffer) for
    /// <paramref name="k"/> tokens. Sizes each buffer at the single-query dim × K; reused when the
    /// requested K is ≤ the currently-allocated K. Reallocates (freeing the old set) on a larger K.
    /// Freed in <see cref="Dispose"/>. K is bounded by <see cref="MaxBatchVerifyK"/>.
    /// </summary>
    private void EnsureBatchVerifyScratch(int k)
    {
        if (_bvK >= k) return;
        if (_bvK > 0) FreeBatchVerifyScratch(); // free the prior (fully-allocated) generation

        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int ffnDim = ComputeFfnScratchDim(_isMoE, _intermDim, _expertDim);

        // Track each allocation so a mid-way throw (e.g. OOM on a later buffer) frees the
        // partial set instead of leaking it (the fields would otherwise hold live-but-orphaned
        // tensors with _bvK==0).
        var allocated = new List<Tensor>(10);
        Tensor Alloc(long n) { var t = _gpu.Allocate(TensorShape.D1(n)); allocated.Add(t); return t; }
        try
        {
            _hiddenK = Alloc((long)k * _embDim);
            _residualK = Alloc((long)k * _embDim);
            _normK = Alloc((long)k * _embDim);
            _qK = Alloc((long)k * qDim);
            _kK = Alloc((long)k * kvDim);
            _vK = Alloc((long)k * kvDim);
            _attnOutK = Alloc((long)k * qDim);
            _ffnGateK = Alloc((long)k * ffnDim);
            _ffnUpK = Alloc((long)k * ffnDim);
            _logitsK = Alloc((long)k * _hp.VocabSize);
            _logitsKBuf = new float[k * _hp.VocabSize]; // k ≤ 8, vocab small → fits int
            _bvK = k;
        }
        catch
        {
            foreach (var t in allocated) _gpu.Free(t);
            _bvK = 0; // fields hold freed tensors; next call (re)allocates, Dispose skips (_bvK==0)
            throw;
        }
    }

    // Called only when fully allocated (_bvK > 0); the tensors are non-null here. A partial
    // allocation that threw is freed in EnsureBatchVerifyScratch's catch, not here (the fields
    // would be null/freed). Free dereferences tensor.Handle, so it is NOT null-safe.
    private void FreeBatchVerifyScratch()
    {
        _gpu.Free(_hiddenK); _gpu.Free(_residualK); _gpu.Free(_normK);
        _gpu.Free(_qK); _gpu.Free(_kK); _gpu.Free(_vK); _gpu.Free(_attnOutK);
        _gpu.Free(_ffnGateK); _gpu.Free(_ffnUpK); _gpu.Free(_logitsK);
        _logitsKBuf = null;
        _bvK = 0;
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

    private static int EstimateMaxContext(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        DType kvDType = DType.Float32)
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

        // KV cache: 2 (K+V) * numLayers * ByteSize(kvDim) per token. DTypeInfo.ByteSize handles
        // every store dtype (fp32 = kvDim*4, bf16 = kvDim*2, q8_0 = (kvDim/32)*34), so use it
        // instead of BytesPerElement (which THROWS for the quantized q8_0 store, issue #325).
        // bf16 (#311) ~halves the per-token store and q8_0 (#325) ~quarters it, so the
        // auto-context buys ~2×/~4× the tokens under --kv-type bf16/q8_0 respectively.
        long bytesPerToken;
        if (hp.LayerHeadDim is { } lhd)
        {
            // Gemma 4: per-layer KV geometry (head_dim 256/512, kv_heads 8/1). Every layer is
            // allocated at full context on Vulkan (no SWA ring), so sum the per-layer K+V bytes.
            long perToken = 0;
            for (int i = 0; i < hp.NumLayers; i++)
            {
                int layerKv = hp.LayerKvHeads is { } lkv ? lkv[i] : hp.NumKvHeads;
                perToken += 2L * DTypeInfo.ByteSize(layerKv * lhd[i], kvDType);
            }
            bytesPerToken = perToken;
        }
        else
        {
            bytesPerToken = 2L * hp.NumLayers * DTypeInfo.ByteSize(hp.NumKvHeads * headDim, kvDType);
        }

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
            _gpu.Free(_wq[i]); _gpu.Free(_wk[i]);
            // Gemma 4 k_eq_v global layers have no attn_v (V reuses raw K) — _wv[i] is null.
            if (_wv[i] is not null) _gpu.Free(_wv[i]);
            _gpu.Free(_wo[i]);
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
                if (_isGemma4)
                {
                    _gpu.Free(_wQNormG4![i]); _gpu.Free(_wKNormG4![i]);
                }
                else
                {
                    _gpu.Free(_wqNorm![i]); _gpu.Free(_wkNorm![i]);
                }
            }

            // Gemma 4 sandwich-norm weights.
            if (_wPostAttnNorm is not null) _gpu.Free(_wPostAttnNorm[i]);
            if (_wPostFfwNorm is not null) _gpu.Free(_wPostFfwNorm[i]);
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

        // Gemma 4 rope_freqs table (#309), null unless present + sized for the model.
        if (_gpuRopeFreqs is { } ropeFreqs) _gpu.Free(ropeFreqs);

        // Flash-decoding split-KV (#312) partial buffers (null unless the opt-in gate enabled them)
        if (_splitKvPartialO is { } skO) _gpu.Free(skO);
        if (_splitKvPartialMeta is { } skM) _gpu.Free(skM);

        // SnapKV (#59) buffers
        if (_snapKvQCapture is { } capBuf) _gpu.Free(capBuf);
        if (_snapKvScoreAccum is { } accBuf) _gpu.Free(accBuf);
        if (_snapKvScoreScratch is { } scrBuf && _snapKvScoreScratchOwned) _gpu.Free(scrBuf);

        // Batched-verify scratch (#308 PR1c). Guarded: only fully-allocated when _bvK > 0 (a
        // partial alloc that threw is freed in EnsureBatchVerifyScratch's catch, leaving _bvK==0).
        if (_bvK > 0) FreeBatchVerifyScratch();

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
    internal static bool IsRawGpuQuant(DType dtype) =>
        dtype is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Q4_0;
}
