using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// GPU-resident forward pass for dense LLaMA-family transformers driven by the
/// CUDA backend (cuBLAS + NVRTC compute kernels).
///
/// All weights live in VRAM (Q4_K / Q6_K / Q8_0 raw bytes for projection
/// matrices, FP32 for norm/bias weights). One-token autoregressive decode runs the full
/// sequence on the GPU: embedding lookup, per-layer attention with paged-stride
/// KV cache, SwiGLU FFN, output projection, and a single logits download.
///
/// Optional TurboQuant 3-bit KV cache compression mirrors the Vulkan path:
/// recent tokens stay in a small FP32 ring buffer, older tokens are compressed
/// to TQ blocks. Limited to head_dim ∈ {128, 256}. The CUDA TQ attention kernel
/// uses a stored-scores fast path up to 4096 positions and falls through to a
/// triple-pass recompute branch above that, so the full model context window
/// is supported (e.g. 40K tokens on Qwen3-8B, the unblocking step that closes
/// the 3.4× memory advantage TurboQuant offers on a 12 GiB card).
///
/// MoE (qwen3moe, olmoe, etc.) is supported as a full-VRAM offload: all expert
/// weights for every layer are resident, decode runs the router → top-K
/// softmax/sigmoid → per-expert SwiGLU → weighted combine pattern mirrored from
/// `GpuForwardPass`. Won't fit on cards with insufficient VRAM (Qwen3-Coder 30B
/// Q4_K_M needs ~17 GB just for weights — use OLMoE-1B-7B or smaller for
/// 12 GB validation, see scripts/download-model.ps1).
///
/// Limitations (intentional):
///   • No NoPE layer skipping (NoRopeLayerStep is honored if set, but the
///     primary target Qwen3 uses RoPE on every layer).
///   • Embedding table accepted as Q4_K or F32; quantized variants are
///     dequantized to F32 on CPU when uploading (small one-time cost).
///   • MoE + TurboQuant combination not validated (no test model has both;
///     should compose but the dispatch never sees the combination today).
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
    private readonly Tensor[] _wGate, _wUp, _wDown;            // dense FFN
    private readonly Tensor _wOutputNorm;
    private readonly Tensor _wOutput;

    // ── MoE state (null/empty when !_isMoE) ──────────────────────────────────
    // Mirrors GpuForwardPass: per-expert weight uploads, router projection, and
    // scratch buffers for the per-expert FFN accumulation. Top-K selection runs
    // on CPU after downloading router logits — the same pattern Vulkan uses.
    private readonly bool _isMoE, _hasSharedExpert;
    private readonly int _expertDim;
    private readonly Tensor[]? _wGateInp;                       // router projection per layer
    private readonly Tensor[][]? _wGateExps, _wUpExps, _wDownExps;
    private readonly Tensor[]? _wGateShexp, _wUpShexp, _wDownShexp; // shared-expert per layer
    private readonly Tensor? _routerLogits;                     // [numExperts]
    private readonly Tensor? _moeSharedOut;                     // [embDim] shared-expert output
    private readonly Tensor? _moeExpertOut;                     // [embDim] per-expert scratch
    private readonly float[]? _routerBuf;                       // CPU-side router logits

    // Optional attention biases
    private readonly bool _hasAttnBias;
    private readonly Tensor[]? _bq, _bk, _bv, _bo;

    // Optional per-head QK norm (Qwen3)
    private readonly bool _hasQkNorm;
    private readonly Tensor[]? _wqNorm, _wkNorm;

    // Per-layer KV cache in VRAM.
    // Non-TQ path: full FP32 cache [maxSeqLen, numKvHeads*headDim] per layer.
    // TQ path:     FP32 ring window [tqFp32Window, numKvHeads*headDim] per layer,
    //              plus TurboQuant-compressed storage for older positions.
    private readonly Tensor[] _gpuKCache;
    private readonly Tensor[] _gpuVCache;

    // Layers whose K/V buffers ALIAS another layer's pages (Gemma 4 shared_kv_layers
    // tail). Dispose must skip Free() for any handle that is shared with another
    // layer to avoid a double free / use-after-free. Empty for non-gemma4 models.
    private readonly HashSet<int> _kvAliasedLayers = new();

    // ── Gemma 4 plumbing ──────────────────────────────────────────────────
    // PLE table (per_layer_token_embd) stays mmap-resident — ~4.2 GB at Q8_0,
    // never uploaded to GPU. Forward dequants the active row per token into a
    // managed buffer, uploads it, and runs the per-layer projection on-GPU.
    // Per-layer model projection (per_layer_model_proj.weight) is BF16 on disk;
    // we dequant once into a CPU float[] (kept as a safety net) AND upload to
    // GPU at construction (~26 MB) so the per-token MatMul stays on-device.
    // The small per-layer F32 norms / projections (inp_gate / proj / post_norm /
    // per_layer_proj_norm) likewise upload at construction (~215 MB total) —
    // trivially small in VRAM and keeps the hot path free of CPU hops.
    private readonly CudaTensorRef? _cpuPleTokenEmbed;
    private readonly float[]? _cpuPerLayerModelProj;
    private readonly CudaTensorRef? _cpuPerLayerProjNorm;
    private readonly CudaTensorRef[]? _cpuInpGate;
    private readonly CudaTensorRef[]? _cpuPleProj;
    private readonly CudaTensorRef[]? _cpuPlePostNorm;

    // GPU-resident PLE weights (uploaded at construction).
    private readonly Tensor? _gpuPerLayerModelProj;     // [PleWidth*NumLayers, embDim] F32
    private readonly Tensor? _gpuPerLayerProjNorm;      // [PleWidth] F32 (Gemma w+1 baked in)
    private readonly Tensor[]? _gpuInpGate;             // per-layer [PleWidth, embDim] F32
    private readonly Tensor[]? _gpuPleProj;             // per-layer [embDim, PleWidth] F32
    private readonly Tensor[]? _gpuPlePostNorm;         // per-layer [embDim] F32 (Gemma w+1)

    // Per-token PLE scratch buffers (VRAM-resident, sized once at construction).
    private readonly Tensor? _gpuPleRow;        // [PleWidth*NumLayers]   F32 — dequant'd token row
    private readonly Tensor? _gpuProjPerLayer;  // [PleWidth*NumLayers]   F32 — per-layer projection
    private readonly Tensor? _gpuPleX;          // [PleWidth]             F32 — inner gate buffer
    private readonly Tensor? _gpuPleY;          // [embDim]               F32 — inner proj buffer
    // Non-owning per-layer slice views into _gpuProjPerLayer (offset layer*PleWidth),
    // built once so ApplyPerLayerEmbeddingGpu can read the proj slice without a copy.
    // Freed in Dispose.
    private readonly Tensor[]? _gpuProjSliceViews; // [L] view → _gpuProjPerLayer[layer]
    // Managed buffer for dequanting the active token's PLE row before upload.
    private readonly float[]? _pleRowHost;
    private readonly int _pleWidth;

    // Per-layer post-attention / post-FFN RmsNorm weights (Gemma 4). Each is a
    // [embDim] f32 vector. Uploaded with Gemma (w-1)→(w+1) offset baked in.
    private readonly Tensor[]? _wPostAttnNorm;
    private readonly Tensor[]? _wPostFfwNorm;

    // Per-layer scalar (layer_output_scale.weight); CPU-side.
    private readonly float[]? _layerOutputScale;

    // Per-layer head_dim flag — non-null on Gemma 4. Triggers the gemma4 Forward path.
    private readonly bool _isGemma4Like;
    private readonly int _maxHeadDim;

    // Softmax score scale passed to the attention kernels. Gemma 4 uses
    // attention_scale = 1.0 (no 1/sqrt(head_dim) prefactor); every other model uses
    // the kernel default (≤0 → 1/sqrt(head_dim)). Cached so the batched/per-token
    // attention call sites stay model-agnostic.
    private readonly float _attnScale;

    // Issue #136: all-GPU batched-trunk prefill scratch (Gemma 4). Lazily sized to
    // the current prompt length N; reallocated when N changes (prefill is infrequent).
    // The embDim/intermDim/pleWidth buffers hold exactly N rows, so element-count-driven
    // ops (Add/Scale/Copy) span exactly N tokens. The Q/K/V/attn buffers are sized for
    // max head_dim and accessed through per-layer _gpu.View slices at the active layer's
    // head_dim. Freed in Dispose.
    /// <summary>
    /// Gates the issue-#136 batched-trunk prefill. Initialised from
    /// <c>SHARPI_BATCHED_PREFILL</c> (default on); settable so tests can A/B the
    /// batched path against the bit-exact per-token loop on one model instance.
    /// </summary>
    public bool BatchedPrefillEnabled { get; set; }
    /// <summary>
    /// Issue #141: route Q8_0 trunk matmuls in the batched prefill through the
    /// compute-bound cuBLAS GEMM (<see cref="CudaBackend.MatMulBatchedGemm"/>)
    /// instead of the memory-bound matvec GEMM-N. Default on
    /// (<c>SHARPI_PREFILL_GEMM</c>); settable so tests can A/B the GEMM path
    /// against the bit-exact matvec path on one model instance. Non-Q8_0 trunk
    /// weights always fall back to the matvec GEMM-N regardless.
    /// </summary>
    public bool PrefillGemmEnabled { get; set; }
    /// <summary>
    /// Issue #141 (MMQ): route Q8_0 trunk matmuls through the int8 tensor-core MMQ
    /// kernel (<see cref="CudaBackend.MatMulBatchedMmq"/>) instead of the
    /// dequant→fp16→cuBLAS GEMM (<see cref="PrefillGemmEnabled"/>). Reads each Q8_0
    /// weight once as int8 with no fp16 HBM temp, on int8 tensor cores. Issue #156 C2
    /// extends the same MMQ path to Q4_K (nibble-expanded weights + asymmetric min-bias,
    /// kernel <c>llm_mmq_q4k</c>). Takes
    /// precedence over <see cref="PrefillGemmEnabled"/> when both are set. Default on
    /// (<c>SHARPI_PREFILL_MMQ</c>=0 reverts to the GEMM path). Argmax-stable, not
    /// bit-exact (both operands int8-quantized), like the GEMM path it replaces.
    /// </summary>
    public bool PrefillMmqEnabled { get; set; }
    /// <summary>
    /// Issue #141 (attention): route the batched-trunk prefill attention through the
    /// memory-efficient flash kernel (<see cref="CudaBackend.FlashAttentionPrefill"/>,
    /// shared K/V tiles + online softmax) instead of the scalar per-query
    /// <see cref="CudaBackend.AttentionBatched"/> / <see cref="CudaBackend.AttentionSwaBatched"/>
    /// (which re-read each query's whole K/V range from global — O(n²), the dominant
    /// prefill cost). fp32 KV only. Argmax-stable, not bit-exact (online softmax).
    /// Default on (<c>SHARPI_PREFILL_FLASH</c>=0 reverts to the scalar kernels).
    /// </summary>
    public bool PrefillFlashAttnEnabled { get; set; }

    /// <summary>
    /// Issues #146/#147: use the tensor-core flash-attention prefill (QK^T + P·V on the
    /// mma cores) instead of the half2 <see cref="PrefillFlashAttnEnabled"/> kernel. Takes
    /// precedence within the flash path. Requires head_dim % 16 == 0 (Gemma 4: 256/512);
    /// the multi-warp #147 kernel (head_dim % 64 == 0) is +27-40% over half2, the
    /// single-warp #146 fallback +5%. Default on (<c>SHARPI_PREFILL_FLASH_TC</c>=0 reverts
    /// to half2). Argmax-stable, not bit-exact (fp16 Q/K/V/P + online softmax).
    /// </summary>
    public bool PrefillFlashTcEnabled { get; set; }
    private bool _forceFlashTc1;             // #147 A/B: pin the single-warp TC kernel
    private readonly bool _mmqSoa;           // #149: repack 2-D Q8_0 weights to SoA at upload
    private int _bpCapacity;                 // current N the scratch is sized for (0 = none)
    private Tensor? _bpHidden, _bpResidual, _bpNorm;       // [N × embDim]
    private Tensor? _bpQ, _bpAttnOut;                      // [N × numHeads*maxHeadDim]
    private Tensor? _bpK, _bpV;                            // [N × numKvHeads*maxHeadDim]
    private Tensor? _bpFfnGate, _bpFfnUp;                  // [N × intermDim]
    private Tensor? _bpProjAll, _bpPleRowAll;             // [N × L*pleWidth]
    private Tensor? _bpPleGate;                            // [N × pleWidth]
    private Tensor? _bpPleY;                               // [N × embDim]
    private float[]? _bpPleRowHostAll;                     // managed [N × L*pleWidth]
    /// <summary>True if the most recent <see cref="Prefill"/> used the batched trunk.</summary>
    public bool LastPrefillWasBatched { get; private set; }

    // SWA RoPE freq base (10K for Gemma 4 SWA layers). 0 when no SWA layers.
    private readonly float _ropeThetaSwa;

    // Optional Gemma 4 RoPE frequency-scaling table (size = maxHeadDim/2). Non-null
    // when `rope_freqs.weight` is present; applied to non-SWA (global) layers only,
    // mirrors llama.cpp gemma4.cpp:191 and the CPU ForwardPass globalFreqFactors path.
    private readonly Tensor? _gpuRopeFreqs;

    // CPU prefix cache kept only to satisfy IForwardPass.TruncateTo + InferenceEngine
    // prefix-reuse bookkeeping. CUDA cache state advances in lockstep via _kvLength.
    private readonly KvCache _kvCache;

    // TurboQuant state (null/0 when disabled).
    private readonly bool _tqEnabled;
    private readonly int _tqFp32Window;
    private readonly int _tqBits;
    private readonly int _tqBlockBytes;
    private Tensor[]? _gpuTqKCache;
    private Tensor[]? _gpuTqVCache;
    private Tensor[]? _gpuSignPatterns;
    private Tensor? _gpuCodebook;
    private Tensor? _gpuBoundaries;
    private Tensor? _rotatedQ;
    private Tensor? _evictK;
    private Tensor? _evictV;
    // Per-query-head softmax-scores scratch in VRAM, sized [numHeads × maxSeqLen].
    // Used by both the TQ and FP32 attention kernels when the live context exceeds
    // their shared-memory fast-path cap (4096 positions). Allocated only when
    // _maxSeqLen > 4096; otherwise null and never touched.
    private Tensor? _attnScoresScratch;
    private int _tqCompressedLen;
    private int _fp32WriteIdx;
    private int _fp32Count;

    // Dtype dispatch for MatMul (mirrors GpuForwardPass._weightDTypes).
    private readonly Dictionary<nint, DType> _weightDTypes = new();

    // SnapKV (#59) — prefill-time eviction by attention-weight scoring.
    // Mirrors the CudaHybridGdnForwardPass layout, minus the GDN per-layer-type
    // indirection: every layer in CudaForwardPass is an attention layer.
    private readonly SnapKvConfig _snapKvCfg;
    private readonly int _snapKvEffectiveBudget;
    private Tensor? _snapKvQCapture;     // [numLayers × W × qDim] f32, captured during Prefill
    private int _snapKvQCaptureW;        // cached W the buffer was sized for
    private Tensor? _snapKvScoreAccum;   // [maxSeqLen] f32, per-position importance accumulator
    private Tensor? _snapKvScoreScratch; // [numHeads × maxSeqLen] f32, lazy scratch for the score kernel
    private bool _snapKvScoreScratchOwned; // false if aliased to _attnScoresScratch
    private int _snapKvCaptureSlot = -1; // 0..W-1 for tokens in the capture window; -1 otherwise

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeqLen;

    /// <summary>
    /// Test-only accessor for the post-prefill KV slot count. After a SnapKV-active
    /// prefill this is the keep-set size (≤ budget); otherwise it's the prompt length.
    /// </summary>
    internal int KvLength => _kvLength;

    public CudaForwardPass(GgufModel model, CudaBackend gpu, ModelHyperparams hp,
        int maxContextLength = 0,
        bool enableTurboQuant = false, int tqFp32Window = 256, int tqBits = 3,
        bool? mmqSoa = null)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        // Issue #149: repack 2-D Q8_0 weights into the SoA layout at upload so all the
        // Q8_0 readers (prefill MMQ, decode dp4a, fp32 matvec, GEMM-N, dequant) use
        // aligned loads instead of the qs-misalignment funnelshift — +10-12% prefill,
        // bit-identical. The backend auto-routes per repacked handle. Default on
        // (SHARPI_MMQ_SOA=0 reverts).
        _mmqSoa = mmqSoa ?? (Environment.GetEnvironmentVariable("SHARPI_MMQ_SOA") != "0");
        _tqEnabled = enableTurboQuant;
        _tqBits = enableTurboQuant ? tqBits : 0;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;
        _isMoE = hp.IsMoE;
        _hasSharedExpert = hp.HasSharedExpert;
        _expertDim = hp.IsMoE ? hp.ExpertIntermediateDim : 0;

        // Gemma 4 / per-layer head_dim path. _maxHeadDim sizes the Q/K/V/attnOut
        // scratch (per-layer view tensors carve out the active head_dim).
        _isGemma4Like = hp.LayerHeadDim is not null;
        // Gemma 4 attention has no 1/sqrt(head_dim) prefactor; everyone else lets the
        // kernel derive it (attnScale ≤ 0 → 1/sqrt(head_dim)).
        _attnScale = _isGemma4Like ? 1f : -1f;
        // Batched-trunk prefill is on by default; SHARPI_BATCHED_PREFILL=0 forces the
        // bit-exact per-token loop (useful for A/B and parity debugging).
        BatchedPrefillEnabled =
            Environment.GetEnvironmentVariable("SHARPI_BATCHED_PREFILL") != "0";
        PrefillGemmEnabled =
            Environment.GetEnvironmentVariable("SHARPI_PREFILL_GEMM") != "0";
        // Issue #141 (MMQ): default on — the int8 tensor-core MMQ beats the
        // dequant→fp16→cuBLAS GEMM matmul (~316ms vs ~332ms over a 1848-tok Gemma 4
        // prefill) and drops the fp16 weight HBM temp. SHARPI_PREFILL_MMQ=0 reverts to
        // the cuBLAS GEMM path. Gated under PrefillGemmEnabled (the compute-bound path).
        PrefillMmqEnabled =
            Environment.GetEnvironmentVariable("SHARPI_PREFILL_MMQ") != "0";
        // Issue #141 (attention): default on — the flash kernel cuts batched-prefill
        // attention ~929ms→~411ms (2.26×) at N=1848, lifting Gemma 4 prefill
        // ~1389→~2180 t/s. SHARPI_PREFILL_FLASH=0 reverts to the scalar kernels.
        PrefillFlashAttnEnabled =
            Environment.GetEnvironmentVariable("SHARPI_PREFILL_FLASH") != "0";
        // Issues #146/#147: tensor-core flash prefill — default on (the #147 multi-warp
        // kernel is +27-40% over half2 on Gemma 4 at d=512). SHARPI_PREFILL_FLASH_TC=0
        // reverts to the half2 kernel.
        PrefillFlashTcEnabled =
            Environment.GetEnvironmentVariable("SHARPI_PREFILL_FLASH_TC") != "0"
            && (_headDim & 15) == 0;
        // Issue #147 A/B: force the single-warp TC kernel even where #147 would apply.
        _forceFlashTc1 =
            Environment.GetEnvironmentVariable("SHARPI_PREFILL_FLASH_TC1") == "1";
        _maxHeadDim = _headDim;
        if (hp.LayerHeadDim is { } lhdMax)
            for (int i = 0; i < hp.NumLayers; i++)
                if (lhdMax[i] > _maxHeadDim) _maxHeadDim = lhdMax[i];
        _ropeThetaSwa = hp.RopeThetaSwa;

        if (_tqEnabled && _headDim is not 128 and not 256)
            throw new NotSupportedException(
                $"CUDA TurboQuant requires head_dim ∈ {{128, 256}} (model head_dim={_headDim}).");
        if (_tqEnabled && tqBits != 3)
            throw new NotSupportedException(
                $"CUDA TurboQuant only ships 3-bit kernels today (requested bits={tqBits}).");

        if (maxContextLength > 0)
            _maxSeqLen = Math.Min(maxContextLength, hp.ContextLength);
        else if (_tqEnabled)
            _maxSeqLen = EstimateMaxContextTq(model, gpu, hp, tqFp32Window, tqBits);
        else
            _maxSeqLen = EstimateMaxContext(model, gpu, hp);

        if (_tqEnabled)
        {
            _tqFp32Window = Math.Min(tqFp32Window, _maxSeqLen);
            _tqBlockBytes = TurboQuantOps.BlockSize(tqBits, _headDim);
            // The TQ attention kernel uses a stored-scores fast path up to 4096 positions
            // and a triple-pass recompute path above that. No per-context allocation cap.
        }

        _kvCache = new KvCache(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.HeadDim);

        // SnapKV (issue #59) — gated by SHARPI_SNAPKV_BUDGET. Buffers are lazily
        // allocated on the first active prefill in Prefill(). Composition with
        // TurboQuant requires per-block ring bookkeeping that doesn't yet exist
        // (issue #60); explicit opt-in + TQ is rejected up front, and the auto
        // path stays disabled when TQ is on.
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
            // SnapKV stacking on the TQ ring buffer needs separate design — see #60.
            // Defer auto-enable.
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
                    $"[CudaForwardPass] SnapKV auto-enabled: budget={_snapKvEffectiveBudget}, " +
                    $"window={_snapKvCfg.Window}, recency={_snapKvCfg.Recency} " +
                    $"(full cache ~{fullCacheBytes / (1024.0 * 1024.0):F0} MiB; " +
                    "set SHARPI_SNAPKV_BUDGET=0 to disable).");
            }
        }

        // SnapKV cannot compose with Gemma-4-like models: their SWA layers use
        // sliding-window ring caches and layers carry per-layer head_dim, so the
        // full-context scoring + uniform-kvDim compaction in ApplySnapKvEviction would
        // mis-index those caches (out-of-range gather, wrong row stride). Force it off
        // rather than silently corrupt the cache; warn only if it was explicitly asked for.
        if (_isGemma4Like && _snapKvEffectiveBudget > 0)
        {
            if (_snapKvCfg.IsBudgetExplicit)
                Console.Error.WriteLine(
                    "[CudaForwardPass] SnapKV is not supported for Gemma-4-style models " +
                    "(sliding-window ring caches + per-layer head_dim); ignoring the " +
                    "configured budget and using the full KV cache.");
            _snapKvEffectiveBudget = 0;
        }

        Console.Error.WriteLine($"[CudaForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength}){(_tqEnabled ? " [TQ3]" : "")}");

        bool vramTrace = Environment.GetEnvironmentVariable("SHARPI_TRACE_VRAM") == "1";
        void TraceVram(string label)
        {
            if (vramTrace)
                Console.Error.WriteLine($"[VRAM] {label}: free={gpu.FreeVramBytes / (1024 * 1024)} MiB");
        }
        TraceVram("constructor entry");

        // Scratch — sized for the widest layer head_dim so per-layer view tensors
        // (Gemma 4: 256 SWA / 512 global) can carve out the active rows.
        _hidden    = gpu.Allocate(TensorShape.D1(_embDim));
        _residual  = gpu.Allocate(TensorShape.D1(_embDim));
        _normBuf   = gpu.Allocate(TensorShape.D1(_embDim));
        _q         = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
        _k         = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _v         = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _attnOut   = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
        // FFN scratch sized for MoE expert dim when MoE; dense FFN uses _intermDim.
        int ffnScratchDim = _isMoE ? _expertDim : _intermDim;
        _ffnGate   = gpu.Allocate(TensorShape.D1(ffnScratchDim));
        _ffnUp     = gpu.Allocate(TensorShape.D1(ffnScratchDim));
        _logits    = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _logitsBuf = new float[hp.VocabSize];

        if (_isMoE)
        {
            _routerLogits = gpu.Allocate(TensorShape.D1(hp.NumExperts));
            _routerBuf    = new float[hp.NumExperts];
            _moeExpertOut = gpu.Allocate(TensorShape.D1(_embDim));
            _moeSharedOut = _hasSharedExpert ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        }

        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[hp.NumLayers];
        _gpuVCache = new Tensor[hp.NumLayers];
        if (_tqEnabled)
        {
            if (hp.LayerHeadDim is not null)
                throw new NotSupportedException(
                    "CUDA TurboQuant is not supported for per-layer head_dim architectures " +
                    "(e.g. Gemma 4). Disable --tq.");

            // FP32 window holds only the recent `tqFp32Window` positions per layer.
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
            }

            // TQ-compressed storage for older positions, stored as uint[] (one block per
            // (position, kv_head) at byte offset position*numKvHeads*blockBytes + ...).
            int maxTqPositions = Math.Max(0, _maxSeqLen - _tqFp32Window);
            long tqBytesPerPos = (long)_numKvHeads * _tqBlockBytes;
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

            // Upload TQ constants to VRAM.
            var centroids = TurboQuantCodebooks.GetCentroids(tqBits, _headDim).ToArray();
            _gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));

            var boundaries = TurboQuantCodebooks.GetBoundaries(tqBits, _headDim).ToArray();
            _gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));

            _rotatedQ = gpu.Allocate(TensorShape.D1((long)_numHeads * _headDim));
            _evictK   = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _headDim));
            _evictV   = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _headDim));

        }
        else
        {
            // Per-layer KV-cache sizing for gemma4 (LayerHeadDim != null): each layer's
            // K/V buffer takes its own head_dim, and SWA layers cap context at
            // SlidingWindowSize. KV-share layers (KvSourceLayer[L] >= 0) alias the
            // source layer's handle — _kvAliasedLayers tracks them so Dispose skips
            // the double-free.
            //
            // Non-gemma4 path is unchanged: every layer is full-context with the
            // model-wide head_dim.
            bool perLayerKv = hp.LayerHeadDim is not null;
            int swaWindow = hp.SlidingWindowSize > 0 ? hp.SlidingWindowSize : _maxSeqLen;
            for (int i = 0; i < hp.NumLayers; i++)
            {
                int kvSrc = hp.KvSourceLayer is { } ksl ? ksl[i] : -1;
                if (kvSrc >= 0)
                {
                    // Alias the source layer's K/V handles. Source must already be
                    // initialised — the GGUF puts shared_kv_layers at the tail.
                    _gpuKCache[i] = _gpuKCache[kvSrc];
                    _gpuVCache[i] = _gpuVCache[kvSrc];
                    _kvAliasedLayers.Add(i);
                    continue;
                }

                int layerHd = perLayerKv ? hp.LayerHeadDim![i] : _headDim;
                int layerKvDim = _numKvHeads * layerHd;
                int layerCtx = (perLayerKv && hp.IsSwaLayer is { } swa && swa[i])
                    ? Math.Min(_maxSeqLen, swaWindow)
                    : _maxSeqLen;
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim));
            }
        }

        // Long-context kernels (both TQ and FP32 attention) need a per-head softmax-scores
        // scratch buffer once seq_len exceeds the shared-memory fast-path cap (4096).
        // Skip the allocation when the whole context fits in shared memory — CUDA's
        // kernels accept a null pointer in that case.
        if (_maxSeqLen > 4096)
            _attnScoresScratch = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxSeqLen));

        // Upload weights to VRAM
        int L = hp.NumLayers;
        _wAttnNorm = new Tensor[L]; _wFfnNorm = new Tensor[L];
        _wq = new Tensor[L]; _wk = new Tensor[L]; _wv = new Tensor[L]; _wo = new Tensor[L];
        // Dense FFN arrays are still allocated for MoE so the layout matches; the
        // MoE-specific arrays below carry the expert/shared weights instead.
        _wGate = new Tensor[L]; _wUp = new Tensor[L]; _wDown = new Tensor[L];

        if (_isMoE)
        {
            _wGateInp   = new Tensor[L];
            _wGateExps  = new Tensor[L][];
            _wUpExps    = new Tensor[L][];
            _wDownExps  = new Tensor[L][];
            _wGateShexp = _hasSharedExpert ? new Tensor[L] : null;
            _wUpShexp   = _hasSharedExpert ? new Tensor[L] : null;
            _wDownShexp = _hasSharedExpert ? new Tensor[L] : null;
        }

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

        // Gemma 4: per-layer post-norms + per-layer output-scale scalars + PLE refs.
        if (hp.HasPostAttnNorm) _wPostAttnNorm = new Tensor[L];
        if (hp.HasPostFfwNorm)  _wPostFfwNorm  = new Tensor[L];
        if (hp.HasLayerOutputScale) _layerOutputScale = new float[L];

        TraceVram("before per-layer weight upload");
        Console.Error.Write($"[CudaForwardPass] Uploading {L} layers to VRAM...");
        for (int i = 0; i < L; i++)
        {
            bool kvShared = hp.KvSourceLayer is { } ksl && ksl[i] >= 0;

            _wAttnNorm[i] = UploadNormWeight($"blk.{i}.attn_norm.weight");
            _wq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            // KV-share layers (Gemma 4 tail) don't carry their own attn_k/attn_v
            // weights — they reuse the source layer's projections and aliased K/V
            // pages. Skip the lookup so missing-tensor errors don't fire.
            if (!kvShared)
            {
                _wk[i] = UploadWeight($"blk.{i}.attn_k.weight");
                _wv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            }
            _wo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _wFfnNorm[i] = UploadNormWeight($"blk.{i}.ffn_norm.weight");

            if (_wPostAttnNorm is not null)
                _wPostAttnNorm[i] = UploadNormWeight($"blk.{i}.post_attention_norm.weight");
            if (_wPostFfwNorm is not null)
                _wPostFfwNorm[i]  = UploadNormWeight($"blk.{i}.post_ffw_norm.weight");
            if (_layerOutputScale is not null)
                _layerOutputScale[i] = LoadScalarF32($"blk.{i}.layer_output_scale.weight");
            if (_isMoE)
            {
                _wGateInp![i]  = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                _wGateExps![i] = UploadExpertWeights($"blk.{i}.ffn_gate_exps.weight", _expertDim, _embDim,    hp.NumExperts);
                _wUpExps![i]   = UploadExpertWeights($"blk.{i}.ffn_up_exps.weight",   _expertDim, _embDim,    hp.NumExperts);
                _wDownExps![i] = UploadExpertWeights($"blk.{i}.ffn_down_exps.weight", _embDim,    _expertDim, hp.NumExperts);
                if (_hasSharedExpert)
                {
                    _wGateShexp![i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                    _wUpShexp![i]   = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                    _wDownShexp![i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _wGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _wUp[i]   = UploadWeight($"blk.{i}.ffn_up.weight");
                _wDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _bq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                if (!kvShared)
                {
                    _bk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                    _bv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                }
                _bo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                // Per-head Q/K norm weights are loaded RAW (no Gemma w+1 offset)
                // on every model — including Gemma 4. The CPU reference loads
                // them via LoadBias which does not apply the offset, so CUDA
                // must match.
                _wqNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                if (!kvShared)
                    _wkNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            Console.Error.Write(".");
        }

        // Embedding table — session-lifetime, use exact-size allocation (see #25/#26):
        // a Q4_K embedding can be 700+ MiB raw and would otherwise round to 1 GiB.
        Console.Error.Write(" emb...");
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        // Raw-upload branch: Q4_K and Q8_0 both have dedicated `EmbedLookup*`
        // kernels on the GPU side, so keep the packed representation in VRAM and
        // gather a single token's row at decode time. Q8_0 is the Gemma 4 case:
        // 2560 × 262144 dequanted to F32 would burn ~2.7 GB of VRAM on top of the
        // raw 700 MB table.
        if (embInfo.DType == DType.Q4_K || embInfo.DType == DType.Q8_0)
        {
            var embData = model.GetTensorData(embInfo);
            _gpuEmbedding = _gpu.UploadRaw(embData, TensorShape.D1(embData.Length), embInfo.DType, exact: true);
            _embIsQuantized = true;
            _weightDTypes[_gpuEmbedding.Handle] = embInfo.DType;
        }
        else
        {
            var embData = model.GetTensorData(embInfo);
            var embF32 = new float[(int)embInfo.ElementCount];
            Dequantize.ToFloat32(embData, embF32, embInfo.DType, embInfo.ElementCount);
            _gpuEmbedding = _gpu.Upload(embF32, TensorShape.D1(embF32.Length), exact: true);
            _embIsQuantized = false;
            _weightDTypes[_gpuEmbedding.Handle] = DType.Float32;
        }

        _wOutputNorm = UploadNormWeight("output_norm.weight");
        _wOutput = model.FindTensor("output.weight") is not null
            ? UploadWeight("output.weight")
            : _gpuEmbedding;

        // Gemma 4 / Gemma-3n: optional `rope_freqs.weight` table (size = maxHeadDim/2)
        // masks the global-layer RoPE high-frequency tail (~identity for long context).
        // CPU bakes this into its precomputed RoPE table; CUDA applies it live via
        // `RoPEWithFactors` on non-SWA layers — see ForwardGemma4 dispatch.
        if (_isGemma4Like
            && model.FindTensor("rope_freqs.weight") is GgufTensorInfo rfInfo
            && rfInfo.DType == DType.Float32
            && rfInfo.ElementCount == _maxHeadDim / 2)
        {
            _gpuRopeFreqs = UploadWeight("rope_freqs.weight");
        }

        // ── Gemma 4 PLE plumbing ───────────────────────────────────────────────
        // Per-layer-embedding table (~4.2 GB at Q8_0) MUST stay CPU-resident; the
        // matching TierPlanner branch excludes it from the GPU weight budget.
        // Forward gathers + dequants one row per token, pipes it through pinned
        // host memory, then runs the projection on-GPU.
        // The small per-layer F32 weights (inp_gate / proj / post_norm /
        // per_layer_model_proj / per_layer_proj_norm) all upload at construction —
        // ~215 MB total, trivially absorbed in VRAM and keeps the per-token hot
        // path free of CPU MatMul hops. The CPU refs stay as a safety net.
        _pleWidth = hp.HasPerLayerTokenEmbd ? hp.PerLayerEmbeddingWidth : 0;
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

            _cpuPleTokenEmbed = ResolveCpuTensor("per_layer_token_embd.weight");

            var projInfo = model.FindTensor("per_layer_model_proj.weight")!.Value;
            var projData = model.GetTensorData(projInfo);
            int projCount = (int)projInfo.ElementCount;
            _cpuPerLayerModelProj = new float[projCount];
            Dequantize.ToFloat32(projData, _cpuPerLayerModelProj.AsSpan(), projInfo.DType, projCount);

            _cpuPerLayerProjNorm = ResolveCpuTensor("per_layer_proj_norm.weight");

            _cpuInpGate     = new CudaTensorRef[L];
            _cpuPleProj     = new CudaTensorRef[L];
            _cpuPlePostNorm = new CudaTensorRef[L];
            for (int i = 0; i < L; i++)
            {
                _cpuInpGate[i]     = ResolveCpuTensor($"blk.{i}.inp_gate.weight");
                _cpuPleProj[i]     = ResolveCpuTensor($"blk.{i}.proj.weight");
                _cpuPlePostNorm[i] = ResolveCpuTensor($"blk.{i}.post_norm.weight");
            }

            // GPU uploads. per_layer_model_proj is [PleWidth*NumLayers, EmbDim] F32
            // (10752, 2560) ~ 26 MB. Per-layer F32 norms / projections each ~2.5 MB,
            // total ~215 MB across 42 layers — fits trivially in VRAM.
            _gpuPerLayerModelProj = _gpu.Upload(_cpuPerLayerModelProj,
                TensorShape.D1(_cpuPerLayerModelProj.Length), exact: true);
            _weightDTypes[_gpuPerLayerModelProj.Handle] = DType.Float32;

            _gpuPerLayerProjNorm = UploadNormWeight("per_layer_proj_norm.weight");

            _gpuInpGate     = new Tensor[L];
            _gpuPleProj     = new Tensor[L];
            _gpuPlePostNorm = new Tensor[L];
            for (int i = 0; i < L; i++)
            {
                _gpuInpGate[i]     = UploadWeight($"blk.{i}.inp_gate.weight");
                _gpuPleProj[i]     = UploadWeight($"blk.{i}.proj.weight");
                _gpuPlePostNorm[i] = UploadNormWeight($"blk.{i}.post_norm.weight");
            }

            int stackedDim = L * _pleWidth;
            _gpuPleRow       = _gpu.Allocate(TensorShape.D1(stackedDim));
            _gpuProjPerLayer = _gpu.Allocate(TensorShape.D1(stackedDim));
            _gpuPleX         = _gpu.Allocate(TensorShape.D1(_pleWidth));
            _gpuPleY         = _gpu.Allocate(TensorShape.D1(_embDim));
            _pleRowHost      = new float[stackedDim];

            // Precompute static per-layer proj-slice views (no per-token copy).
            _gpuProjSliceViews = new Tensor[L];
            for (int i = 0; i < L; i++)
                _gpuProjSliceViews[i] = _gpu.View(_gpuProjPerLayer, (long)i * _pleWidth, _pleWidth);
        }

        Console.Error.WriteLine(" done.");
        TraceVram("after all weight uploads");

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
    private static readonly bool s_prefillProfile =
        Environment.GetEnvironmentVariable("SHARPI_PREFILL_PROFILE") == "1";
    private readonly System.Diagnostics.Stopwatch _profSw = new();
    private double _profAttnMs;
    private readonly System.Diagnostics.Stopwatch _profMmSw = new();
    private double _profMatmulMs;
    private readonly double[] _phaseMs = new double[10];
    private readonly long[]   _phaseCount = new long[10];
    private const int PH_EMBED = 0, PH_QKV = 1, PH_ROPE_QKN = 2, PH_KV_ATTN = 3,
                      PH_O_RES = 4, PH_FFN = 5, PH_FINAL = 6, PH_PLE = 7;
    private static readonly string[] s_phaseName =
        ["embed", "qkv-matmul", "rope+qknorm", "kv+attn", "o-proj+res", "ffn", "final+download", "ple"];

    // CUDA Graph decode (issue #136): the Gemma 4 layer + output region has static
    // topology across decode tokens (only `position` varies), so capture it once on the
    // first decode token and replay per token — collapsing ~1k host launches/token into
    // one cuGraphLaunch + a handful of node-param updates. Falls back to direct launches
    // on any capture/replay failure.
    //
    // Default ON (issue #142): after the #137 launch fusions and the dp4a decode matvec,
    // graphs measure +9–10% at both low and ~1K context on the all-GPU Gemma 4 path (the
    // earlier short-context regression that kept #136 default-off is gone). SHARPI_CUDA_GRAPH=0
    // reverts to direct launches.
    private bool _useCudaGraph =
        Environment.GetEnvironmentVariable("SHARPI_CUDA_GRAPH") != "0";
    private bool _graphCaptured;
    // Number of KV entries SnapKV physically dropped when it compacted the cache
    // (= N - K at eviction, 0 otherwise). Decode indexes the compacted cache by the
    // *physical* slot `position - _kvEvictedCount` while RoPE keeps the logical
    // `position`; without this the post-eviction decode reads stale/duplicated slots
    // (cache-fill != absolute position). Also gates CUDA graphs off once > 0 — a
    // compacted cache breaks the captured seqLen == position+1 invariant. A configured
    // SnapKV budget that never evicts (prompt <= budget) leaves this 0, so the cache
    // fills sequentially and graphs stay valid. Reset on a full ResetCache.
    private int _kvEvictedCount;

    /// <summary>
    /// Enable/disable CUDA-graph capture+replay for the Gemma 4 decode loop. Defaults from
    /// the <c>SHARPI_CUDA_GRAPH</c> env var; set before the first decode (tests/bench).
    /// </summary>
    public bool UseCudaGraph { get => _useCudaGraph; set => _useCudaGraph = value; }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        if (_isGemma4Like) return s_profile ? ForwardProfiledGemma4(token, position) : ForwardGemma4(token, position);

        if (s_profile) return ForwardProfiled(token, position);

        // Embed token. Dispatch on stored dtype so Q8_0 (Gemma 4 / Phase 0) uses
        // its dedicated kernel; Q4_K keeps the legacy fast path; F32 the generic one.
        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0)
                _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);

        // Transformer layers + final norm/output — static topology across tokens (only
        // `position` varies), so optionally capture it once into a CUDA graph and replay
        // per token to kill the ~1k-launch/token host overhead (#136/#158). The token-
        // varying embedding ran above; the TQ ring-advance (host state) and logits
        // download run after.
        if (!TryRunDeviceRegionViaGraph(position))
            RunDeviceRegion(position);

        // After all layers have used the same FP32 indices for this token, advance the
        // TQ ring-buffer state (shared across layers). Pure host-state mutation for the
        // NEXT token — kept outside the captured region (it would break static topology,
        // and TQ already bails graphs off anyway).
        if (_tqEnabled)
        {
            if (_fp32Count >= _tqFp32Window)
                _tqCompressedLen++;
            _fp32WriteIdx = (_fp32WriteIdx + 1) % _tqFp32Window;
            if (_fp32Count < _tqFp32Window)
                _fp32Count++;
        }

        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    // Transformer layer loop + final norm/output — the pure on-device-compute region the
    // CUDA-graph path captures. Token-varying embedding runs before it; the TQ ring-advance
    // (host state) and logits download + sync run after. Only `position` varies across
    // tokens. Contains the TQ-hybrid and SnapKV-capture blocks for the direct-launch path;
    // both are host-gated off whenever graphs are active (see TryRunDeviceRegionViaGraph).
    private void RunDeviceRegion(int position)
    {
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
                    _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
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
                _gpu.CopyDeviceRegion(capBuf, dstOffsetElems * sizeof(float),
                                      _q, 0, (long)qDim * sizeof(float));
            }

            int kvDim = _numKvHeads * _headDim;

            if (_tqEnabled)
            {
                long rowBytes = (long)kvDim * sizeof(float);

                // Evict the oldest FP32 row to TQ storage if the ring is full.
                if (_fp32Count >= _tqFp32Window)
                {
                    _gpu.CopyDeviceRegion(_evictK!, 0,
                        _gpuKCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    _gpu.CopyDeviceRegion(_evictV!, 0,
                        _gpuVCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    _gpu.TqKvAppend(_evictK!, _evictV!,
                        _gpuTqKCache![layer], _gpuTqVCache![layer],
                        _gpuSignPatterns![layer], _gpuCodebook!, _gpuBoundaries!,
                        kvDim, _headDim, _tqCompressedLen,
                        _maxSeqLen, _numKvHeads, _tqBlockBytes);
                }

                // Append the fresh K/V into the ring buffer slot.
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    kvDim, _fp32WriteIdx, _tqFp32Window);

                int fp32SeqLen = Math.Min(_fp32Count + 1, _tqFp32Window);

                // Pre-eviction fast path: before the ring wraps, every cached row sits at
                // its natural position-index and there's no TQ-compressed history yet, so
                // the plain Attention kernel can read the FP32 cache directly. Skipping
                // TqRotateQuery + the larger TqAttention kernel (its codebook init,
                // K-block staging shared mem, and extra args) is worth several percent at
                // short context. Once any row has been evicted (_tqCompressedLen > 0),
                // fall through to the full hybrid TqAttention path.
                if (_tqCompressedLen == 0)
                {
                    _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                        _attnScoresScratch,
                        _numHeads, _numKvHeads, _headDim, fp32SeqLen, _tqFp32Window);
                }
                else
                {
                    // Rotate the query (per-layer sign pattern) for fused dequant-dot.
                    _gpu.TqRotateQuery(_q, _rotatedQ!, _gpuSignPatterns![layer],
                        _numHeads, _numKvHeads, _headDim);

                    _gpu.TqAttention(_q, _rotatedQ!,
                        _gpuTqKCache![layer], _gpuTqVCache![layer],
                        _gpuKCache[layer], _gpuVCache[layer], _attnOut, _gpuCodebook!,
                        _attnScoresScratch,
                        _numHeads, _numKvHeads, _headDim,
                        _tqCompressedLen, fp32SeqLen, _maxSeqLen, _tqBlockBytes);
                }
            }
            else
            {
                // Index the cache by the physical slot. After a SnapKV eviction the cache
                // is compacted to K entries, so the next write lands at
                // `position - _kvEvictedCount` (= position when nothing was evicted), and
                // attention reads that many + 1. RoPE above still uses the logical position.
                int kvSlot = position - _kvEvictedCount;
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, kvSlot, _maxSeqLen);
                _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    _attnScoresScratch,
                    _numHeads, _numKvHeads, _headDim, kvSlot + 1, _maxSeqLen);
            }

            GpuMatMul(_hidden, _wo[layer], _attnOut);
            if (_hasAttnBias)
                _gpu.AddInPlace(_hidden, _bo![layer]);

            _gpu.AddInPlace(_hidden, _residual);

            // FFN
            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);

            if (_isMoE)
            {
                MoeFfn(layer);
            }
            else
            {
                GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
                GpuMatMul(_ffnUp,   _wUp[layer],   _normBuf);
                _gpu.SiLuMul(_ffnGate, _ffnUp);
                GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            }

            _gpu.AddInPlace(_hidden, _residual);
        }

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        GpuMatMul(_logits, _wOutput, _hidden);
    }

    // CUDA-graph capture/replay for the non-Gemma dense decode region (#158, mirrors the
    // Gemma TryRunGemma4DeviceRegionViaGraph). Returns true if the logits were produced via
    // the graph (captured on the first eligible token, replayed after); false means the
    // caller must run the region with direct launches. Any capture/replay failure disables
    // graphs for the rest of the session and degrades to direct launches.
    private bool TryRunDeviceRegionViaGraph(int position)
    {
        if (!_useCudaGraph || !_gpu.GraphCaptureSupported)
            return false;

        // Graphs bake in the standard decode invariant (fixed topology, seqLen ==
        // position+1, no host-varying device offsets). Three things break that and must
        // bail to direct launches:
        //  • TurboQuant — extra host-synced rotate/compress ops + a ring whose advance is
        //    host state (the if/else attention branch also flips once a row evicts).
        //  • An actual SnapKV eviction — the cache is compacted, so cache-fill != absolute
        //    position. A configured-but-unevicted budget keeps _kvEvictedCount == 0 and is
        //    fine (the cache still fills sequentially).
        //  • An active SnapKV Q-capture window (_snapKvCaptureSlot >= 0, prefill scoring
        //    only) — its per-layer CopyDeviceRegion writes to a host-computed slot offset
        //    that would be baked wrong on replay. Pure decode keeps the slot at -1, so the
        //    capture block is dormant and never enters the captured graph.
        //  • MoE — MoeFfn does a router Download + Synchronize mid-layer, which is illegal
        //    during stream capture (it would error the stream).
        if (_tqEnabled || _kvEvictedCount > 0 || _snapKvCaptureSlot >= 0 || _isMoE)
            return false;

        // Steady state: graph already captured — just replay at the new position.
        if (_graphCaptured && _gpu.GraphReady)
        {
            try { _gpu.LaunchGraphForPosition(position); return true; }
            catch { _useCudaGraph = false; _graphCaptured = false; return false; }
        }

        if (_graphCaptured)
            return false; // latched but not ready — shouldn't happen; stay on direct launches

        // First eligible token: pre-grow the Q4_K/Q8_0 dp4a Q8_1 input scratch to the widest
        // decode matvec (FFN-down cols = intermDim, output proj cols = embDim) BEFORE capture
        // — DispatchMatVecQ4K/Dp4a grow it on demand via cudaMalloc, which capture forbids.
        _gpu.EnsureQ81Scratch(Math.Max(_embDim, _intermDim));

        // Capture the region (records onto the stream without executing) then launch it for
        // real. On any failure the region was NOT executed, so the caller re-runs it directly.
        try
        {
            if (!_gpu.TryBeginGraphCapture())
                return false;
            RunDeviceRegion(position);
            if (!_gpu.TryEndGraphCaptureAndInstantiate())
                return false;
            _gpu.LaunchGraphForPosition(position);
            _graphCaptured = true;
            return true;
        }
        catch
        {
            _gpu.AbortGraphCapture();
            _useCudaGraph = false;
            return false;
        }
    }

    /// <summary>
    /// Gemma 4 (E4B) forward path. Mirrors <see cref="ForwardPass.Forward"/> exactly:
    /// embedding scale, PLE pre-pass, per-layer head_dim variance, dual-RoPE (10K
    /// SWA / 1M global), KV-share dispatch, sliding-window attention, post-attn /
    /// post-ffn norms, GeluTanhMul FFN, PLE injection, layer_output_scale, final
    /// softcap on the logits.
    /// </summary>
    private ReadOnlySpan<float> ForwardGemma4(int token, int position)
    {
        if (s_regionProfile) return ForwardGemma4RegionProfiled(token, position);

        // 1. Embedding lookup
        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0)
                _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);

        if (_hp.EmbeddingScale != 1f)
            _gpu.ScaleInPlace(_hidden, _hp.EmbeddingScale);

        // 2. PLE pre-pass — build per-layer projection cache once per token.
        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjectionsGpu(token);

        // 3. Transformer layers + final norm/output/softcap. Static topology across
        //    tokens (only `position` varies), so optionally capture it once into a CUDA
        //    graph and replay per token to kill the ~1k-launch/token host overhead (#136).
        if (!TryRunGemma4DeviceRegionViaGraph(position))
            RunGemma4DeviceRegion(position);

        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    // Region profiler (SHARPI_DECODE_REGIONS=1): splits the graphs-ON decode token into
    // (a) embed+PLE-prepass, (b) graphed device region, (c) logits download+sync, each
    // bracketed by a Synchronize so the wall-clock is attributable. The syncs add ~µs of
    // overhead but reveal where the per-token time goes (the SHARPI_CUDA_PROFILE path runs
    // graphs-off, which inflates the launch-bound phases). Prints every 64 tokens.
    private static readonly bool s_regionProfile =
        Environment.GetEnvironmentVariable("SHARPI_DECODE_REGIONS") == "1";
    private readonly System.Diagnostics.Stopwatch _rpSw = new();
    private double _rpEmbedPle, _rpDevice, _rpDownload;
    private long _rpCount;

    private ReadOnlySpan<float> ForwardGemma4RegionProfiled(int token, int position)
    {
        _rpSw.Restart();
        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0) _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);
        if (_hp.EmbeddingScale != 1f) _gpu.ScaleInPlace(_hidden, _hp.EmbeddingScale);
        if (_hp.HasPerLayerTokenEmbd) BuildPerLayerProjectionsGpu(token);
        _gpu.Synchronize();
        _rpEmbedPle += _rpSw.Elapsed.TotalMilliseconds; _rpSw.Restart();

        if (!TryRunGemma4DeviceRegionViaGraph(position))
            RunGemma4DeviceRegion(position);
        _gpu.Synchronize();
        _rpDevice += _rpSw.Elapsed.TotalMilliseconds; _rpSw.Restart();

        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();
        _rpDownload += _rpSw.Elapsed.TotalMilliseconds;

        if (++_rpCount % 64 == 0)
        {
            double n = _rpCount;
            Console.Error.WriteLine(
                $"[regions] n={_rpCount} embed+ple={_rpEmbedPle / n:F3}ms device={_rpDevice / n:F3}ms " +
                $"download={_rpDownload / n:F3}ms total={(_rpEmbedPle + _rpDevice + _rpDownload) / n:F3}ms");
        }

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
    }

    // Transformer layer loop + final norm/output/softcap — the pure on-device-compute
    // region the CUDA-graph path captures. Token-varying embedding + PLE run before it;
    // the logits download + sync run after. Only `position` varies across tokens.
    private void RunGemma4DeviceRegion(int position)
    {
        int L = _hp.NumLayers;
        for (int layer = 0; layer < L; layer++)
        {
            int layerHd = _hp.LayerHeadDim![layer];
            int qDimL = _numHeads * layerHd;
            int kvDimL = _numKvHeads * layerHd;
            int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];

            // Per-layer view tensors so MatMul writes only qDimL/kvDimL rows.
            var qView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _q.Handle);
            var kView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _k.Handle);
            var vView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _v.Handle);
            var attnOutView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _attnOut.Handle);

            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);

            GpuMatMul(qView, _wq[layer], _normBuf);
            if (!kvShared)
            {
                GpuMatMul(kView, _wk[layer], _normBuf);
                GpuMatMul(vView, _wv[layer], _normBuf);
            }

            // Per-head Q/K norm (Gemma 4: shared headDim-sized weight per head).
            // CPU applies norm BEFORE RoPE (UseL2QkNorm == false).
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                if (!kvShared)
                    _gpu.HeadNormQk(qView, _wqNorm![layer], kView, _wkNorm![layer],
                        _numHeads, _numKvHeads, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                else
                    _gpu.HeadNorm(qView, _wqNorm![layer], _numHeads, layerHd,
                        _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }

            float ropeTheta = isSwa ? _ropeThetaSwa : _hp.RopeTheta;
            // Global (non-SWA) layers use rope_freqs.weight to mask high-frequency
            // pairs; SWA layers use the plain table. Mirrors llama.cpp gemma4.cpp:191.
            if (!isSwa && _gpuRopeFreqs is { } rfTbl)
            {
                _gpu.RoPEWithFactors(qView, position, layerHd, ropeTheta, rfTbl);
                if (!kvShared)
                    _gpu.RoPEWithFactors(kView, position, layerHd, ropeTheta, rfTbl);
            }
            else
            {
                _gpu.RoPE(qView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
                if (!kvShared)
                    _gpu.RoPE(kView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
            }

            // KV append on the owning layer only; shared layers read from the
            // source layer's pages via effLayer below.
            if (!kvShared)
            {
                int layerCtx = isSwa && _hp.SlidingWindowSize > 0
                    ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
                    : _maxSeqLen;
                _gpu.KvAppend(kView, vView, _gpuKCache[layer], _gpuVCache[layer],
                    kvDimL, position, layerCtx);
            }

            int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer]
                              && _hp.SlidingWindowSize > 0)
                ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
                : _maxSeqLen;

            // Gemma 4 uses attention_scale = 1.0 (no 1/sqrt(head_dim) prefactor). Pass
            // it explicitly so the kernel skips its rsqrtf(head_dim) — matching the CPU
            // path's `_layerHeadDim is not null ? 1f : 1/sqrt(hd)` exactly with no
            // prescale round-trip, and dropping a ScaleInPlace launch per layer.
            if (isSwa)
            {
                _gpu.AttentionSwa(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch,
                    position, _hp.SlidingWindowSize, layerHd,
                    _numHeads, _numKvHeads, effLayerCtx, attnScale: 1f);
            }
            else
            {
                _gpu.Attention(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch,
                    _numHeads, _numKvHeads, layerHd, position + 1, effLayerCtx, attnScale: 1f);
            }

            // Output projection: _wo[layer] is [embDim, qDimL] — pass attnOutView
            // with ElementCount = qDimL so the matvec reads exactly the active head_dim.
            GpuMatMul(_hidden, _wo[layer], attnOutView);

            // Gemma 4: post-attn RmsNorm before residual.
            if (_wPostAttnNorm is not null)
                _gpu.RmsNorm(_hidden, _hidden, _wPostAttnNorm[layer], _hp.RmsNormEps);

            _gpu.AddInPlace(_hidden, _residual);

            // FFN
            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);

            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp,   _wUp[layer],   _normBuf);
            _gpu.GeluTanhMul(_ffnGate, _ffnUp);
            GpuMatMul(_hidden, _wDown[layer], _ffnGate);

            // Gemma 4: post-ffn RmsNorm before residual.
            if (_wPostFfwNorm is not null)
                _gpu.RmsNorm(_hidden, _hidden, _wPostFfwNorm[layer], _hp.RmsNormEps);

            _gpu.AddInPlace(_hidden, _residual);

            // PLE injection (after post-FFN residual, before layer_output_scale).
            if (_hp.HasPerLayerTokenEmbd)
                ApplyPerLayerEmbeddingGpu(layer);

            // Per-layer scalar gain (after PLE — matches CPU ordering).
            if (_layerOutputScale is not null)
                _gpu.ScaleInPlace(_hidden, _layerOutputScale[layer]);
        }

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        GpuMatMul(_logits, _wOutput, _hidden);

        if (_hp.FinalLogitSoftcap > 0f)
            _gpu.SoftcapInPlace(_logits, _hp.FinalLogitSoftcap);
    }

    // CUDA-graph capture/replay for the Gemma 4 decode region. Returns true if the logits
    // were produced via the graph (captured on the first decode token, replayed after);
    // false means the caller must run the region with direct launches. Any capture/replay
    // failure disables graphs for the rest of the session and degrades to direct launches.
    private bool TryRunGemma4DeviceRegionViaGraph(int position)
    {
        if (!_useCudaGraph || !_gpu.GraphCaptureSupported)
            return false;

        // Graphs bake in the standard decode invariant (fixed topology, seqLen ==
        // position+1). TurboQuant (extra host-synced rotate/compress ops) and an *actual*
        // SnapKV eviction (cache compacted, so cache-fill != absolute position) break that.
        // A configured-but-unevicted SnapKV budget does not — the cache still fills
        // sequentially — so gate on an actual eviction (_kvEvictedCount > 0), not on the
        // budget being set.
        if (_kvEvictedCount > 0 || _tqEnabled)
            return false;

        // Steady state: graph already captured — just replay at the new position.
        if (_graphCaptured && _gpu.GraphReady)
        {
            try { _gpu.LaunchGraphForPosition(position); return true; }
            catch { _useCudaGraph = false; _graphCaptured = false; return false; }
        }

        if (_graphCaptured)
            return false; // latched but not ready — shouldn't happen; stay on direct launches

        // First decode token: capture the region (records onto the stream without executing)
        // then launch it for real. On any failure the region was NOT executed, so the caller
        // re-runs it directly.
        try
        {
            if (!_gpu.TryBeginGraphCapture())
                return false;
            RunGemma4DeviceRegion(position);
            if (!_gpu.TryEndGraphCaptureAndInstantiate())
                return false;
            _gpu.LaunchGraphForPosition(position);
            _graphCaptured = true;
            return true;
        }
        catch
        {
            _gpu.AbortGraphCapture();
            _useCudaGraph = false;
            return false;
        }
    }

    /// <summary>
    /// Build the per-layer Gemma-4 PLE projection cache once per token. Mirrors
    /// CPU <see cref="ForwardPass.BuildPerLayerProjections"/>: dequant the PLE
    /// row, scale by sqrt(PleWidth), project through per_layer_model_proj, scale
    /// by 1/sqrt(EmbeddingDim), per-layer RmsNorm + add row slice + scale by
    /// 1/sqrt(2).
    /// </summary>
    private void BuildPerLayerProjectionsGpu(int token)
    {
        int L = _hp.NumLayers;
        int stackedDim = L * _pleWidth;

        // CPU dequant of the active token's PLE row.
        var pleRef = _cpuPleTokenEmbed!.Value;
        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(pleRef.DType))
                        * DTypeInfo.BytesPerBlock(pleRef.DType);
        byte* rowPtr = pleRef.DataPtr + (long)token * bytesPerRow;
        var rowHost = _pleRowHost!.AsSpan(0, stackedDim);
        if (pleRef.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, stackedDim).CopyTo(rowHost);
        }
        else
        {
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(rowPtr, bytesPerRow),
                rowHost, pleRef.DType, stackedDim);
        }

        // Upload → _gpuPleRow.
        _gpu.UploadInto(_gpuPleRow!, rowHost);

        // Per-embedding-table scaling: sqrt(PleWidth) = 16 for Gemma 4.
        _gpu.ScaleInPlace(_gpuPleRow!, MathF.Sqrt(_pleWidth));

        // proj_per_layer = per_layer_model_proj @ hidden  → [stackedDim]
        GpuMatMul(_gpuProjPerLayer!, _gpuPerLayerModelProj!, _hidden);

        _gpu.ScaleInPlace(_gpuProjPerLayer!, 1.0f / MathF.Sqrt(_embDim));

        // Per-layer slice: RmsNorm each pleWidth row with per_layer_proj_norm (w+1
        // baked in), add the same-slice PLE row, scale by 1/sqrt(2). The norm is the
        // only per-slice op (one block per row, byte-identical to llm_rmsnorm); the
        // add and scale are pure elementwise and run over the whole [L*pleWidth]
        // buffer at once. Collapses the original per-slice loop (≈6 device ops × L —
        // 3 slice copies + norm/add/scale) to 3 whole-buffer launches.
        _gpu.RmsNormBatched(_gpuProjPerLayer!, _gpuProjPerLayer!, _gpuPerLayerProjNorm!,
            L, _pleWidth, _hp.RmsNormEps);
        _gpu.AddInPlace(_gpuProjPerLayer!, _gpuPleRow!);
        _gpu.ScaleInPlace(_gpuProjPerLayer!, 1.0f / MathF.Sqrt(2f));
    }

    /// <summary>
    /// Inject the layer's PLE residual: <c>gelu_tanh(inp_gate @ hidden) * proj_per_layer[L]
    /// → proj @ → post_norm → add to hidden</c>. Mirrors CPU
    /// <see cref="ForwardPass.ApplyPerLayerEmbedding"/>.
    /// </summary>
    private void ApplyPerLayerEmbeddingGpu(int layer)
    {
        // gate = inp_gate @ hidden  → [pleWidth]
        GpuMatMul(_gpuPleX!, _gpuInpGate![layer], _hidden);

        // up = proj_per_layer[layer], read directly via the static slice view.
        _gpu.GeluTanhMul(_gpuPleX!, _gpuProjSliceViews![layer]);

        // proj output (embDim).
        GpuMatMul(_gpuPleY!, _gpuPleProj![layer], _gpuPleX!);
        _gpu.RmsNorm(_gpuPleY!, _gpuPleY!, _gpuPlePostNorm![layer], _hp.RmsNormEps);
        _gpu.AddInPlace(_hidden, _gpuPleY!);
    }

    private ReadOnlySpan<float> ForwardProfiled(int token, int position)
    {
        if (_tqEnabled)
            throw new NotSupportedException(
                "SHARPI_CUDA_PROFILE per-phase profiling is not wired for the TurboQuant path. " +
                "Disable TurboQuant to use the profiler, or extend ForwardProfiled if you need per-phase TQ timings.");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        long t0 = sw.ElapsedTicks;

        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0)
                _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);
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
                    _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                }
            }
            _gpu.Synchronize();
            AccPhase(PH_ROPE_QKN, sw, ref t0);

            // SnapKV (issue #59) capture — same as Forward, mirrored into the profiled path.
            if (_snapKvCaptureSlot >= 0 && _snapKvQCapture is { } capBuf)
            {
                int qDimCap = _numHeads * _headDim;
                long dstOffsetElems = ((long)layer * _snapKvQCaptureW + _snapKvCaptureSlot) * qDimCap;
                _gpu.CopyDeviceRegion(capBuf, dstOffsetElems * sizeof(float),
                                      _q, 0, (long)qDimCap * sizeof(float));
            }

            int kvDim = _numKvHeads * _headDim;
            // Physical KV slot (= position unless SnapKV compacted the cache). See the
            // matching note in Forward.
            int kvSlot = position - _kvEvictedCount;
            _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, kvSlot, _maxSeqLen);
            _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                _attnScoresScratch,
                _numHeads, _numKvHeads, _headDim, kvSlot + 1, _maxSeqLen);
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

    /// <summary>
    /// Per-phase profiled mirror of ForwardGemma4. Buckets per-token GPU time into
    /// embed / PLE pre-pass / qkv-matmul / rope+qknorm / kv+attn / o-proj+res /
    /// ffn / final so SHARPI_CUDA_PROFILE=1 can decompose Gemma 4 kernel time.
    /// Throws on TQ (matches the dense profiler's precondition).
    /// </summary>
    private ReadOnlySpan<float> ForwardProfiledGemma4(int token, int position)
    {
        if (_tqEnabled)
            throw new NotSupportedException(
                "SHARPI_CUDA_PROFILE per-phase profiling is not wired for the TurboQuant path.");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        long t0 = sw.ElapsedTicks;

        // 1. Embedding lookup + scale.
        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0) _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else                         _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);
        if (_hp.EmbeddingScale != 1f) _gpu.ScaleInPlace(_hidden, _hp.EmbeddingScale);
        _gpu.Synchronize();
        AccPhase(PH_EMBED, sw, ref t0);

        if (_hp.HasPerLayerTokenEmbd)
        {
            BuildPerLayerProjectionsGpu(token);
            _gpu.Synchronize();
            AccPhase(PH_PLE, sw, ref t0);
        }

        int L = _hp.NumLayers;
        for (int layer = 0; layer < L; layer++)
        {
            int layerHd = _hp.LayerHeadDim![layer];
            int qDimL = _numHeads * layerHd;
            int kvDimL = _numKvHeads * layerHd;
            int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];

            var qView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _q.Handle);
            var kView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _k.Handle);
            var vView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _v.Handle);
            var attnOutView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _attnOut.Handle);

            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);

            GpuMatMul(qView, _wq[layer], _normBuf);
            if (!kvShared)
            {
                GpuMatMul(kView, _wk[layer], _normBuf);
                GpuMatMul(vView, _wv[layer], _normBuf);
            }
            _gpu.Synchronize();
            AccPhase(PH_QKV, sw, ref t0);

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                if (!kvShared)
                    _gpu.HeadNormQk(qView, _wqNorm![layer], kView, _wkNorm![layer],
                        _numHeads, _numKvHeads, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                else
                    _gpu.HeadNorm(qView, _wqNorm![layer], _numHeads, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }
            float ropeTheta = isSwa ? _ropeThetaSwa : _hp.RopeTheta;
            if (!isSwa && _gpuRopeFreqs is { } rfTbl)
            {
                _gpu.RoPEWithFactors(qView, position, layerHd, ropeTheta, rfTbl);
                if (!kvShared) _gpu.RoPEWithFactors(kView, position, layerHd, ropeTheta, rfTbl);
            }
            else
            {
                _gpu.RoPE(qView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
                if (!kvShared) _gpu.RoPE(kView, position, layerHd, ropeTheta, _hp.IsNeoxRope);
            }
            _gpu.Synchronize();
            AccPhase(PH_ROPE_QKN, sw, ref t0);

            if (!kvShared)
            {
                int layerCtx = isSwa && _hp.SlidingWindowSize > 0
                    ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
                    : _maxSeqLen;
                _gpu.KvAppend(kView, vView, _gpuKCache[layer], _gpuVCache[layer], kvDimL, position, layerCtx);
            }
            int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer]
                              && _hp.SlidingWindowSize > 0)
                ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
                : _maxSeqLen;
            if (isSwa)
                _gpu.AttentionSwa(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch, position, _hp.SlidingWindowSize, layerHd,
                    _numHeads, _numKvHeads, effLayerCtx, attnScale: 1f);
            else
                _gpu.Attention(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch, _numHeads, _numKvHeads, layerHd, position + 1, effLayerCtx, attnScale: 1f);
            _gpu.Synchronize();
            AccPhase(PH_KV_ATTN, sw, ref t0);

            GpuMatMul(_hidden, _wo[layer], attnOutView);
            if (_wPostAttnNorm is not null)
                _gpu.RmsNorm(_hidden, _hidden, _wPostAttnNorm[layer], _hp.RmsNormEps);
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.Synchronize();
            AccPhase(PH_O_RES, sw, ref t0);

            CopyDevice(_residual, _hidden);
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp,   _wUp[layer],   _normBuf);
            _gpu.GeluTanhMul(_ffnGate, _ffnUp);
            GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            if (_wPostFfwNorm is not null)
                _gpu.RmsNorm(_hidden, _hidden, _wPostFfwNorm[layer], _hp.RmsNormEps);
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.Synchronize();
            AccPhase(PH_FFN, sw, ref t0);

            if (_hp.HasPerLayerTokenEmbd)
            {
                ApplyPerLayerEmbeddingGpu(layer);
                _gpu.Synchronize();
                AccPhase(PH_PLE, sw, ref t0);
            }

            if (_layerOutputScale is not null)
                _gpu.ScaleInPlace(_hidden, _layerOutputScale[layer]);
        }

        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        GpuMatMul(_logits, _wOutput, _hidden);
        if (_hp.FinalLogitSoftcap > 0f)
            _gpu.SoftcapInPlace(_logits, _hp.FinalLogitSoftcap);
        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();
        AccPhase(PH_FINAL, sw, ref t0);

        _kvLength = Math.Max(_kvLength, position + 1);
        return _logitsBuf;
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
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        int N = tokens.Count;
        LastPrefillWasBatched = false;

        // Issue #142: pre-grow the dp4a Q8_1 input scratch to the widest decode
        // matvec (FFN down: cols = intermDim) before any CUDA-graph capture on the
        // first decode token — capture forbids cudaMalloc, so the buffer must
        // already be at max size. Harmless when the dp4a path is disabled.
        if (_isGemma4Like)
            _gpu.EnsureQ81Scratch(Math.Max(_embDim, _intermDim));

        // SnapKV (issue #59) gating: only run eviction when this is a fresh
        // prefill (startPos==0), the effective budget is positive (env-set or
        // VRAM-scaled auto), the prompt is long enough that eviction would drop
        // something, and TQ is off (composition with the TQ ring is #60).
        bool snapKvActive = _snapKvEffectiveBudget > 0
                         && startPos == 0
                         && !_tqEnabled
                         && N > _snapKvEffectiveBudget
                         && N > _snapKvCfg.Window;
        // Issue #136/#156: all-GPU batched-trunk prefill. Collapses the per-position
        // attention/FFN/PLE launches (whose count grows with N) into batched GEMM-N +
        // batched-attention launches. Originally Gemma-4-only; #156 opened it to any
        // dense model the batched kernels cover (e.g. Qwen3-8B Q4_K). Everything else
        // (MoE, SnapKV-active, TQ, >4096 context, non-NEOX RoPE, L2 QK-norm, attn bias,
        // unbatchable weight dtype) falls back to the per-token loop below.
        if (BatchedPrefillEnabled && !snapKvActive && N >= 2
            && startPos + N <= 4096 && IsBatchedPrefillSupported())
            return PrefillBatchedTrunk(tokens, startPos);

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

    private static bool BatchableDType(DType d) =>
        d is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Float32;

    private DType WDType(Tensor t) => _weightDTypes.GetValueOrDefault(t.Handle, DType.Q4_K);

    /// <summary>
    /// Fail-safe batchability check for the prefill gate: an unregistered handle is
    /// treated as NOT batchable (forcing the per-token fallback) rather than defaulting
    /// to a batchable dtype, so a future weight whose dtype isn't tracked can't slip
    /// into the GEMM-N path and be misdispatched.
    /// </summary>
    private bool BatchableWeight(Tensor t) =>
        _weightDTypes.TryGetValue(t.Handle, out var d) && BatchableDType(d);

    /// <summary>
    /// Whether the issue-#136/#156 batched-trunk prefill can run this model. Requires a
    /// dense model with NEOX RoPE, non-L2 QK-norm, no attention bias, TQ off, and every
    /// trunk (+ optional Gemma PLE) weight in a GEMM-N-batchable dtype. Gemma 4 satisfies
    /// these; so do dense Qwen3/Llama-style models (the batched body skips the Gemma-only
    /// PLE / shared-KV / SWA / post-norm steps via their null/flag guards).
    /// </summary>
    private bool IsBatchedPrefillSupported()
    {
        if (_isMoE || _tqEnabled || _hasAttnBias || !_hp.IsNeoxRope) return false;
        if (_hasQkNorm && _hp.UseL2QkNorm) return false;

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            bool kvShared = _hp.KvSourceLayer is { } ksl && ksl[i] >= 0;
            if (!BatchableWeight(_wq[i]) || !BatchableWeight(_wo[i]) ||
                !BatchableWeight(_wGate[i]) || !BatchableWeight(_wUp[i]) ||
                !BatchableWeight(_wDown[i]))
                return false;
            if (!kvShared && (!BatchableWeight(_wk[i]) || !BatchableWeight(_wv[i])))
                return false;
            if (_hp.HasPerLayerTokenEmbd &&
                (!BatchableWeight(_gpuInpGate![i]) || !BatchableWeight(_gpuPleProj![i])))
                return false;
        }
        if (_hp.HasPerLayerTokenEmbd && !BatchableWeight(_gpuPerLayerModelProj!))
            return false;
        return BatchableWeight(_wOutput);
    }

    private void GpuMatMulBatched(Tensor outAll, Tensor weights, Tensor inAll, int n)
    {
        if (s_prefillProfile) { _gpu.Synchronize(); _profMmSw.Restart(); }
        GpuMatMulBatchedCore(outAll, weights, inAll, n);
        if (s_prefillProfile) { _gpu.Synchronize(); _profMatmulMs += _profMmSw.Elapsed.TotalMilliseconds; }
    }

    private void GpuMatMulBatchedCore(Tensor outAll, Tensor weights, Tensor inAll, int n)
    {
        var dt = WDType(weights);
        // Issue #141: route the trunk matmuls through compute-bound cuBLAS GEMM
        // (each weight read once per batch) instead of the memory-bound matvec
        // GEMM-N (weight re-streamed per token). Q8_0 weights dequant to fp16
        // first; F32 weights (PLE projections) feed Sgemm's TF32 path directly.
        // ~argmax-stable, not byte-exact — fine for Gemma 4 prefill, which has no
        // MTP/GDN byte-parity oracle. Other dtypes keep the matvec GEMM-N.
        if (!PrefillGemmEnabled)
        {
            _gpu.MatMulBatched(outAll, weights, inAll, n, dt);
            return;
        }
        if (dt == DType.Q8_0)
        {
            if (PrefillMmqEnabled)
                _gpu.MatMulBatchedMmq(outAll, weights, inAll, n, dt);
            else
                _gpu.MatMulBatchedGemm(outAll, weights, inAll, n, dt);
        }
        else if (dt == DType.Q4_K)
        {
            // Issue #156 Item C: route Q4_K trunk matmuls through a compute-bound path
            // (weight read once per batch) instead of the memory-bound matvec GEMM-N
            // (weight re-streamed per token). Argmax-stable. Requires cols % 256; every
            // Q4_K hidden dim satisfies it, but fall back to the matvec path defensively
            // if not. C2 (PrefillMmqEnabled): the int8 MMQ reads the weight once as int8
            // with no fp16 dequant temp; C1 fallback: dequant→fp16→cuBLAS GEMM.
            int cols = (int)(inAll.ElementCount / n);
            if ((cols & 0xff) == 0)
            {
                if (PrefillMmqEnabled)
                    _gpu.MatMulBatchedMmq(outAll, weights, inAll, n, dt);
                else
                    _gpu.MatMulBatchedGemm(outAll, weights, inAll, n, dt);
            }
            else
                _gpu.MatMulBatched(outAll, weights, inAll, n, dt);
        }
        else if (dt == DType.Float32)
        {
            // C[n×rows] = A[n×cols] · B[rows×cols]ᵀ  →  Sgemm(C, A, B, M=n, K=cols, N=rows).
            int rows = (int)(outAll.ElementCount / n);
            int cols = (int)(inAll.ElementCount / n);
            _gpu.Sgemm(outAll, inAll, weights, n, cols, rows);
        }
        else
        {
            _gpu.MatMulBatched(outAll, weights, inAll, n, dt);
        }
    }

    /// <summary>(Re)allocate the batched-trunk scratch for a prompt of length N.</summary>
    private void EnsureBatchedTrunkScratch(int n)
    {
        if (_bpCapacity == n) return;
        FreeBatchedTrunkScratch();

        long emb = (long)n * _embDim;
        _bpHidden   = _gpu.Allocate(TensorShape.D1(emb));
        _bpResidual = _gpu.Allocate(TensorShape.D1(emb));
        _bpNorm     = _gpu.Allocate(TensorShape.D1(emb));
        _bpQ        = _gpu.Allocate(TensorShape.D1((long)n * _numHeads * _maxHeadDim));
        _bpAttnOut  = _gpu.Allocate(TensorShape.D1((long)n * _numHeads * _maxHeadDim));
        _bpK        = _gpu.Allocate(TensorShape.D1((long)n * _numKvHeads * _maxHeadDim));
        _bpV        = _gpu.Allocate(TensorShape.D1((long)n * _numKvHeads * _maxHeadDim));
        _bpFfnGate  = _gpu.Allocate(TensorShape.D1((long)n * _intermDim));
        _bpFfnUp    = _gpu.Allocate(TensorShape.D1((long)n * _intermDim));
        if (_hp.HasPerLayerTokenEmbd)
        {
            long stacked = (long)n * _hp.NumLayers * _pleWidth;
            _bpProjAll    = _gpu.Allocate(TensorShape.D1(stacked));
            _bpPleRowAll  = _gpu.Allocate(TensorShape.D1(stacked));
            _bpPleGate    = _gpu.Allocate(TensorShape.D1((long)n * _pleWidth));
            _bpPleY       = _gpu.Allocate(TensorShape.D1(emb));
            _bpPleRowHostAll = new float[stacked];
        }
        _bpCapacity = n;
    }

    private void FreeBatchedTrunkScratch()
    {
        foreach (var t in new[] { _bpHidden, _bpResidual, _bpNorm, _bpQ, _bpAttnOut,
                                  _bpK, _bpV, _bpFfnGate, _bpFfnUp,
                                  _bpProjAll, _bpPleRowAll, _bpPleGate, _bpPleY })
            if (t is { } v) _gpu.Free(v);
        _bpHidden = _bpResidual = _bpNorm = _bpQ = _bpAttnOut = _bpK = _bpV =
            _bpFfnGate = _bpFfnUp = _bpProjAll = _bpPleRowAll = _bpPleGate = _bpPleY = null;
        _bpPleRowHostAll = null;
        _bpCapacity = 0;
    }

    /// <summary>
    /// All-GPU batched-trunk prefill (issue #136; generalized to dense non-Gemma models
    /// in #156). Embeds all N tokens, builds the per-layer PLE projections batched (Gemma
    /// only), runs every transformer layer batched across N, then the final norm + output
    /// projection on the last token. Argmax-stable with the per-token
    /// <see cref="Forward"/> / <see cref="ForwardGemma4"/> loop (flash/GEMM are not byte-exact).
    /// </summary>
    private ReadOnlySpan<float> PrefillBatchedTrunk(IReadOnlyList<int> tokens, int startPos)
    {
        int N = tokens.Count;
        EnsureBatchedTrunkScratch(N);
        int embDim = _embDim;

        bool profile = s_prefillProfile;
        var swp = profile ? System.Diagnostics.Stopwatch.StartNew() : null;
        long tEmbed = 0, tPle = 0, tLayers = 0;

        // 1. Embed every token into _bpHidden, then batched embedding scale. The Q8_0
        // table (Gemma 4) goes through a single batched launch instead of 2·N host-
        // driven EmbedLookup + copy launches (bit-identical); other dtypes keep the loop.
        var embDType = _embIsQuantized
            ? _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K)
            : DType.Float32;
        if (_embIsQuantized && embDType == DType.Q8_0)
        {
            int[] ids = tokens as int[] ?? System.Linq.Enumerable.ToArray(tokens);
            var idTensor = _gpu.UploadRaw(
                System.Runtime.InteropServices.MemoryMarshal.AsBytes<int>(ids),
                TensorShape.D1(N), DType.Float32);
            _gpu.EmbedLookupQ8_0Batched(_gpuEmbedding, _bpHidden!, idTensor, N, embDim);
            _gpu.Free(idTensor);
        }
        else
        {
            for (int i = 0; i < N; i++)
            {
                EmbedTokenGpu(tokens[i]);   // writes _hidden
                _gpu.CopyDeviceRegion(_bpHidden!, (long)i * embDim * sizeof(float),
                                      _hidden, 0, (long)embDim * sizeof(float));
            }
        }
        if (_hp.EmbeddingScale != 1f)
            _gpu.ScaleInPlace(_bpHidden!, _hp.EmbeddingScale);
        if (profile) { _gpu.Synchronize(); tEmbed = swp!.ElapsedMilliseconds; }

        // 2. PLE pre-pass batched (builds _bpProjAll = [N × L*pleWidth]).
        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjectionsBatched(tokens);
        if (profile) { _gpu.Synchronize(); tPle = swp!.ElapsedMilliseconds; }

        // 3. Transformer layers, batched across N.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
            GpuLayerBatchedTrunk(layer, N, startPos);
        if (profile)
        {
            _gpu.Synchronize();
            tLayers = swp!.ElapsedMilliseconds;
            Console.Error.WriteLine($"[prefill-profile] N={N} embed={tEmbed}ms ple={tPle - tEmbed}ms " +
                $"layers={tLayers - tPle}ms (attn={_profAttnMs:F0}ms matmul={_profMatmulMs:F0}ms)");
            _profAttnMs = 0; _profMatmulMs = 0;
        }

        // 4. Final norm + output projection on the last token only.
        var lastHidden = _gpu.View(_bpHidden!, (long)(N - 1) * embDim, embDim);
        _gpu.RmsNorm(_hidden, lastHidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.Free(lastHidden);
        GpuMatMul(_logits, _wOutput, _hidden);
        if (_hp.FinalLogitSoftcap > 0f)
            _gpu.SoftcapInPlace(_logits, _hp.FinalLogitSoftcap);
        _gpu.Download(_logits, _logitsBuf);
        _gpu.Synchronize();

        _kvLength = Math.Max(_kvLength, startPos + N);
        LastPrefillWasBatched = true;
        return _logitsBuf;
    }

    /// <summary>Embed a single token into <c>_hidden</c> (mirrors ForwardGemma4 step 1).</summary>
    private void EmbedTokenGpu(int token)
    {
        if (_embIsQuantized)
        {
            var embDType = _weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
            if (embDType == DType.Q8_0) _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim);
            else                        _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim);
        }
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);
    }

    /// <summary>
    /// Batched PLE pre-pass: build <c>_bpProjAll</c> ([N × L*pleWidth]) for all N
    /// tokens. Mirrors <see cref="BuildPerLayerProjectionsGpu"/> kernel-for-kernel,
    /// batched: one proj GEMM-N, per-(token,layer) RmsNormBatched, full-buffer add/scale.
    /// </summary>
    private void BuildPerLayerProjectionsBatched(IReadOnlyList<int> tokens)
    {
        int N = tokens.Count, L = _hp.NumLayers;
        int stackedDim = L * _pleWidth;
        var pleRef = _cpuPleTokenEmbed!.Value;
        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(pleRef.DType))
                        * DTypeInfo.BytesPerBlock(pleRef.DType);

        // CPU dequant of each token's PLE row into the [N × stackedDim] host buffer.
        // Parallelized across tokens (each row is independent): for a long prompt
        // this serial gather+dequant of N×stackedDim Q8_0 elements was ~30% of the
        // whole batched prefill (issue #141 profiling).
        var host = _bpPleRowHostAll!;
        byte* basePtr = pleRef.DataPtr;
        var dtype = pleRef.DType;
        System.Threading.Tasks.Parallel.For(0, N, i =>
        {
            byte* rowPtr = basePtr + (long)tokens[i] * bytesPerRow;
            var dst = new Span<float>(host).Slice(i * stackedDim, stackedDim);
            if (dtype == DType.Float32)
                new ReadOnlySpan<float>((float*)rowPtr, stackedDim).CopyTo(dst);
            else
                Dequantize.ToFloat32(new ReadOnlySpan<byte>(rowPtr, bytesPerRow), dst, dtype, stackedDim);
        });
        _gpu.UploadInto(_bpPleRowAll!, host);
        _gpu.ScaleInPlace(_bpPleRowAll!, MathF.Sqrt(_pleWidth));

        // proj_per_layer = per_layer_model_proj @ hidden, for all N tokens.
        GpuMatMulBatched(_bpProjAll!, _gpuPerLayerModelProj!, _bpHidden!, N);
        _gpu.ScaleInPlace(_bpProjAll!, 1.0f / MathF.Sqrt(_embDim));

        // Per-(token,layer) slice RmsNorm, then full-buffer add of the PLE row + 1/sqrt(2).
        _gpu.RmsNormBatched(_bpProjAll!, _bpProjAll!, _gpuPerLayerProjNorm!, N * L, _pleWidth, _hp.RmsNormEps);
        _gpu.AddInPlace(_bpProjAll!, _bpPleRowAll!);
        _gpu.ScaleInPlace(_bpProjAll!, 1.0f / MathF.Sqrt(2f));
    }

    /// <summary>
    /// One transformer layer of the batched-trunk prefill, batched across N tokens.
    /// Mirrors the per-token body (<see cref="ForwardGemma4"/> for Gemma, the dense
    /// <see cref="Forward"/> loop otherwise); Gemma-only steps (PLE, shared-KV, SWA,
    /// sandwich post-norms, per-layer head_dim, layer_output_scale) are guarded off for
    /// models that lack them.
    /// </summary>
    private void GpuLayerBatchedTrunk(int layer, int N, int startPos)
    {
        int layerHd = _hp.LayerHeadDim is { } lhd ? lhd[layer] : _headDim;
        int qDimL = _numHeads * layerHd, kvDimL = _numKvHeads * layerHd;
        int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
        bool kvShared = kvSrc >= 0;
        int effLayer = kvShared ? kvSrc : layer;
        bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];
        int window = _hp.SlidingWindowSize;

        // Per-layer dense views (the buffers are sized for max head_dim).
        var qAll = _gpu.View(_bpQ!, 0, (long)N * qDimL);
        var kAll = _gpu.View(_bpK!, 0, (long)N * kvDimL);
        var vAll = _gpu.View(_bpV!, 0, (long)N * kvDimL);
        var attnAll = _gpu.View(_bpAttnOut!, 0, (long)N * qDimL);

        _gpu.CopyDevice(_bpResidual!, _bpHidden!);
        _gpu.RmsNormBatched(_bpNorm!, _bpHidden!, _wAttnNorm[layer], N, _embDim, _hp.RmsNormEps);

        GpuMatMulBatched(qAll, _wq[layer], _bpNorm!, N);
        if (!kvShared)
        {
            GpuMatMulBatched(kAll, _wk[layer], _bpNorm!, N);
            GpuMatMulBatched(vAll, _wv[layer], _bpNorm!, N);
        }

        // RoPE and per-head QK-norm must run in the SAME order as the matching per-token
        // oracle, because RoPE does not commute with per-channel-weighted RMSNorm: Gemma
        // applies QK-norm before RoPE (RunGemma4DeviceRegion), the dense path applies RoPE
        // before QK-norm (Forward). NoRopeLayerStep skips RoPE on the same layers as the
        // per-token path; QK-norm (weighted) always runs. L2 QK-norm is gated out upstream.
        bool useRoPE = _hp.NoRopeLayerStep == 0 || (layer + 1) % _hp.NoRopeLayerStep != 0;
        float ropeTheta = isSwa ? _ropeThetaSwa : _hp.RopeTheta;

        void ApplyQkNormBatched()
        {
            if (!_hasQkNorm || _hp.UseL2QkNorm) return;
            if (!kvShared)
                _gpu.HeadNormQkBatched(qAll, _wqNorm![layer], kAll, _wkNorm![layer],
                    _numHeads, _numKvHeads, layerHd, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            else
                _gpu.HeadNormBatched(qAll, _wqNorm![layer], _numHeads, layerHd, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        }

        void ApplyRopeBatched()
        {
            if (!useRoPE) return;
            if (!isSwa && _gpuRopeFreqs is { } rfTbl)
            {
                _gpu.RoPEWithFactorsBatched(qAll, startPos, layerHd, ropeTheta, rfTbl, _numHeads, N);
                if (!kvShared)
                    _gpu.RoPEWithFactorsBatched(kAll, startPos, layerHd, ropeTheta, rfTbl, _numKvHeads, N);
            }
            else
            {
                _gpu.RoPEPartialBatched(qAll, startPos, layerHd, layerHd, ropeTheta, _numHeads, N, neox: true);
                if (!kvShared)
                    _gpu.RoPEPartialBatched(kAll, startPos, layerHd, layerHd, ropeTheta, _numKvHeads, N, neox: true);
            }
        }

        if (_isGemma4Like) { ApplyQkNormBatched(); ApplyRopeBatched(); }
        else { ApplyRopeBatched(); ApplyQkNormBatched(); }

        if (!kvShared)
        {
            int layerCtx = isSwa && window > 0 ? Math.Min(_maxSeqLen, window) : _maxSeqLen;
            _gpu.KvAppendBatched(kAll, vAll, _gpuKCache[layer], _gpuVCache[layer], kvDimL, startPos, layerCtx, N);
        }

        int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer] && window > 0)
            ? Math.Min(_maxSeqLen, window) : _maxSeqLen;

        if (s_prefillProfile) { _gpu.Synchronize(); _profSw.Restart(); }
        // Gemma 4: attention_scale = 1.0, passed explicitly (kernel skips its rsqrtf).
        // Other models pass _attnScale = -1 so the kernel derives 1/sqrt(head_dim).
        if (PrefillFlashTcEnabled && (layerHd & 15) == 0)
        {
            // #147 multi-warp/d-split when head_dim is a multiple of 64 (W·16); else the
            // #146 single-warp kernel. SHARPI_PREFILL_FLASH_TC1=1 forces single-warp (A/B).
            if (!_forceFlashTc1 && (layerHd & 63) == 0)
                _gpu.FlashAttentionPrefillTc2(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, _numKvHeads, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
            else
                _gpu.FlashAttentionPrefillTc(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, _numKvHeads, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
        }
        else if (PrefillFlashAttnEnabled)
            _gpu.FlashAttentionPrefill(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, _numKvHeads, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
        else if (isSwa)
            _gpu.AttentionSwaBatched(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, _numKvHeads, layerHd, startPos, window, effLayerCtx, N, attnScale: _attnScale);
        else
            _gpu.AttentionBatched(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, _numKvHeads, layerHd, startPos, effLayerCtx, N, attnScale: _attnScale);
        if (s_prefillProfile) { _gpu.Synchronize(); _profAttnMs += _profSw.Elapsed.TotalMilliseconds; }

        GpuMatMulBatched(_bpHidden!, _wo[layer], attnAll, N);
        if (_wPostAttnNorm is not null)
            _gpu.RmsNormBatched(_bpHidden!, _bpHidden!, _wPostAttnNorm[layer], N, _embDim, _hp.RmsNormEps);
        _gpu.AddInPlace(_bpHidden!, _bpResidual!);

        // FFN.
        _gpu.CopyDevice(_bpResidual!, _bpHidden!);
        _gpu.RmsNormBatched(_bpNorm!, _bpHidden!, _wFfnNorm[layer], N, _embDim, _hp.RmsNormEps);
        GpuMatMulBatched(_bpFfnGate!, _wGate[layer], _bpNorm!, N);
        GpuMatMulBatched(_bpFfnUp!,   _wUp[layer],   _bpNorm!, N);
        // SwiGLU (Silu, Qwen/Llama) vs GEGLU (GeluApprox, Gemma 4). Both are
        // elementwise over the whole N·intermDim buffer, so the batched call is
        // identical to the per-token one bar the activation.
        if (_hp.FfnActivation == FfnActivation.GeluApprox)
            _gpu.GeluTanhMul(_bpFfnGate!, _bpFfnUp!);
        else
            _gpu.SiLuMul(_bpFfnGate!, _bpFfnUp!);
        GpuMatMulBatched(_bpHidden!, _wDown[layer], _bpFfnGate!, N);
        if (_wPostFfwNorm is not null)
            _gpu.RmsNormBatched(_bpHidden!, _bpHidden!, _wPostFfwNorm[layer], N, _embDim, _hp.RmsNormEps);
        _gpu.AddInPlace(_bpHidden!, _bpResidual!);

        // PLE injection, batched: gate = inp_gate @ hidden; gelu * proj-slice;
        // proj @; post-norm; add. proj-slice read with per-token stride via the
        // strided gelu, so no gather of the [N × L*pleWidth] projection buffer.
        if (_hp.HasPerLayerTokenEmbd)
        {
            GpuMatMulBatched(_bpPleGate!, _gpuInpGate![layer], _bpHidden!, N);
            _gpu.GeluTanhMulStrided(_bpPleGate!, _bpProjAll!, _pleWidth,
                (long)_hp.NumLayers * _pleWidth, (long)layer * _pleWidth, N);
            GpuMatMulBatched(_bpPleY!, _gpuPleProj![layer], _bpPleGate!, N);
            _gpu.RmsNormBatched(_bpPleY!, _bpPleY!, _gpuPlePostNorm![layer], N, _embDim, _hp.RmsNormEps);
            _gpu.AddInPlace(_bpHidden!, _bpPleY!);
        }

        if (_layerOutputScale is not null)
            _gpu.ScaleInPlace(_bpHidden!, _layerOutputScale[layer]);

        _gpu.Free(qAll); _gpu.Free(kAll); _gpu.Free(vAll); _gpu.Free(attnAll);
    }

    /// <summary>
    /// SnapKV (issue #59): score the captured trailing-W queries against the
    /// VRAM K cache for every layer (atomicAdd-pooled into a single per-position
    /// accumulator), download the accumulator, pick a keep set, then compact the
    /// GPU K/V rings + the host-side <see cref="_kvCache"/> length bookkeeping.
    /// Called once at the end of a SnapKV-active prefill.
    /// </summary>
    private void ApplySnapKvEviction(int N, int W, int wStart)
    {
        EnsureSnapKvScoreBuffers();
        // Zero only the prompt-prefix slice; the rest of the [maxSeqLen] buffer
        // doesn't participate in scoring and will not be downloaded.
        _gpu.ClearRegion(_snapKvScoreAccum!, 0, N);

        int qDim = _numHeads * _headDim;
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            for (int w = 0; w < W; w++)
            {
                // Stage the captured Q into _q so the scoring kernel can read a
                // contiguous [numHeads × headDim] vector at the same device
                // pointer it does during Forward.
                long srcOffsetElems = ((long)layer * _snapKvQCaptureW + w) * qDim;
                _gpu.CopyDeviceRegion(_q, 0,
                    _snapKvQCapture!, srcOffsetElems * sizeof(float),
                    (long)qDim * sizeof(float));

                int qAbsPos = wStart + w;
                _gpu.SnapKvScore(_q, _gpuKCache[layer],
                    _snapKvScoreAccum!, _snapKvScoreScratch!,
                    _numHeads, _numKvHeads, _headDim,
                    N, qAbsPos, _maxSeqLen);
            }
        }

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
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                // K: gather kept positions into stage, then copy stage back over
                // the cache's [0, K * kvDim) prefix. Same for V. Two-phase to
                // avoid src==dst race (kernel block ordering is undefined).
                _gpu.KvCompact(_gpuKCache[layer], stage, keepDev, K, kvDim);
                _gpu.CopyDeviceRegion(_gpuKCache[layer], 0, stage, 0, sliceBytes);
                _gpu.KvCompact(_gpuVCache[layer], stage, keepDev, K, kvDim);
                _gpu.CopyDeviceRegion(_gpuVCache[layer], 0, stage, 0, sliceBytes);
            }
        }
        finally
        {
            _gpu.Free(stage);
            _gpu.Free(keepDev);
        }

        // Update host-side length bookkeeping. _kvCache is bookkeeping-only on
        // CudaForwardPass — actual data lives in _gpuKCache/_gpuVCache.
        _kvLength = K;
        _kvCache.TruncateTo(K);
        // The cache is now compacted: K physical entries at slots [0, K) but the logical
        // RoPE positions continue at N, N+1, … So decode must index the cache by the
        // physical slot `position - _kvEvictedCount`. This also disables CUDA-graph replay
        // for the sequence (the seqLen == position+1 invariant no longer holds); both are
        // re-enabled on a full ResetCache.
        _kvEvictedCount = N - K;
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
        // (the shared-memory fast-path cap), but it always *indexes* into it.
        // Reuse the attention scratch when it's already allocated; otherwise
        // make a dedicated allocation sized [numHeads × maxSeqLen]. Track
        // ownership so Dispose doesn't double-free the aliased case.
        if (_snapKvScoreScratch is null)
        {
            if (_attnScoresScratch is { } existing)
            {
                _snapKvScoreScratch = existing;
                _snapKvScoreScratchOwned = false;
            }
            else
            {
                _snapKvScoreScratch = _gpu.Allocate(TensorShape.D1((long)_numHeads * _maxSeqLen));
                _snapKvScoreScratchOwned = true;
            }
        }
    }

    /// <inheritdoc/>
    public void TruncateTo(int length)
    {
        if (_tqEnabled && length < _tqCompressedLen)
            throw new NotSupportedException(
                $"TruncateTo({length}) cannot rewind into the TQ-compressed region " +
                $"(tqCompressedLen={_tqCompressedLen}). Speculative decoding can only " +
                "truncate inside the FP32 recent window.");
        _kvLength = length;
        _kvCache.TruncateTo(length);
        // _kvEvictedCount is intentionally NOT reset here. TruncateTo only rewinds *decode*
        // tokens (speculative decode rejects), whose logical positions stay >= the eviction
        // point, so the physical mapping `slot = position - _kvEvictedCount` remains valid.
        // (Rewinding below the eviction point would be nonsensical — those prompt tokens were
        // compacted away — and SnapKV does not compose with speculative decode in practice.)
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => true;

    /// <inheritdoc/>
    public void ResetCache()
    {
        _kvLength = 0;
        _kvCache.Reset();
        _tqCompressedLen = 0;
        _fp32WriteIdx = 0;
        _fp32Count = 0;
        _kvEvictedCount = 0; // fresh sequence — standard sequential cache state restored
    }

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        var dtype = _weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K);
        _gpu.MatMul(output, weights, input, dtype);
    }

    private void CopyDevice(Tensor dst, Tensor src) => _gpu.CopyDevice(dst, src);

    /// <summary>
    /// MoE FFN: project to router logits, softmax/sigmoid, top-K, then sum over
    /// the selected experts' SwiGLU outputs weighted by their router gates.
    /// Matches the Vulkan path's `GpuMoeFfn`; the CUDA stream model means we
    /// don't need explicit barriers between dependent ops, but the router
    /// download still needs `Synchronize` so top-K sees finished values.
    /// </summary>
    private void MoeFfn(int layer)
    {
        int numActive = _hp.NumActiveExperts;

        // Router: project hidden through gate_inp, then softmax or sigmoid in place.
        GpuMatMul(_routerLogits!, _wGateInp![layer], _normBuf);
        if (_hp.UseSigmoidGating) _gpu.Sigmoid(_routerLogits!);
        else                       _gpu.Softmax(_routerLogits!);

        _gpu.Download(_routerLogits!, _routerBuf!);
        _gpu.Synchronize();

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerBuf!, numActive, selectedExperts, expertWeights, _hp.NormalizeMoeTopKWeights);

        // Shared expert (always-active) runs once per layer when present.
        if (_hasSharedExpert)
        {
            GpuMatMul(_ffnGate, _wGateShexp![layer], _normBuf);
            GpuMatMul(_ffnUp,   _wUpShexp![layer],   _normBuf);
            _gpu.SiLuMul(_ffnGate, _ffnUp);
            GpuMatMul(_moeSharedOut!, _wDownShexp![layer], _ffnGate);
        }

        _gpu.Clear(_hidden);

        for (int i = 0; i < numActive; i++)
        {
            int expertIdx = selectedExperts[i];
            float expertWeight = expertWeights[i];

            GpuMatMul(_ffnGate, _wGateExps![layer][expertIdx], _normBuf);
            GpuMatMul(_ffnUp,   _wUpExps![layer][expertIdx],   _normBuf);

            // Sigmoid gating: pre-scale gate/up by expertWeight so the post-SiLU
            // accumulator picks up the gate without an extra scaled-add. Softmax
            // gating instead uses AddScaledInPlace on the down projection.
            if (_hp.UseSigmoidGating)
            {
                _gpu.ScaleInPlace(_ffnGate, expertWeight);
                _gpu.ScaleInPlace(_ffnUp,   expertWeight);
            }

            _gpu.SiLuMul(_ffnGate, _ffnUp);
            GpuMatMul(_moeExpertOut!, _wDownExps![layer][expertIdx], _ffnGate);

            if (_hp.UseSigmoidGating)
                _gpu.AddInPlace(_hidden, _moeExpertOut!);
            else
                _gpu.AddScaledInPlace(_hidden, _moeExpertOut!, expertWeight);
        }

        if (_hasSharedExpert)
            _gpu.AddInPlace(_hidden, _moeSharedOut!);
    }

    /// <summary>
    /// Top-K selection in descending order of logit value. Same algorithm as
    /// <see cref="GpuForwardPass"/>: O(k × n), trivial for k=8 / n=64..128.
    /// Selected indices stay in arrival order so weights[i] = logits[indices[i]].
    /// </summary>
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
                    if (indices[j] != i) continue;
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

        if (!normalize || k <= 1) return;
        float sum = 0;
        for (int i = 0; i < k; i++) sum += weights[i];
        if (sum <= 0) return;
        for (int i = 0; i < k; i++) weights[i] /= sum;
    }

    /// <summary>
    /// Upload all <paramref name="expertCount"/> slices of a stacked expert weight
    /// tensor — one Tensor per expert. Matches the Vulkan path so per-expert
    /// MatMul dispatches in MoeFfn use the same indexed layout.
    /// </summary>
    private Tensor[] UploadExpertWeights(string name, int rows, int cols, int expertCount)
    {
        var tensors = new Tensor[expertCount];
        for (int expertIdx = 0; expertIdx < expertCount; expertIdx++)
            tensors[expertIdx] = UploadExpertWeight(name, rows, cols, expertIdx);
        return tensors;
    }

    private Tensor UploadExpertWeight(string name, int rows, int cols, int expertIdx)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // exact=true on every branch: expert weights are session-lifetime, never freed
        // during decode. Pool's power-of-2 round-up wastes VRAM that the SLRU expert
        // cache could otherwise spend on more slots.
        if (info.DType == DType.Float32)
        {
            int elemOffset = expertIdx * rows * cols;
            var floats = MemoryMarshal.Cast<byte, float>(data).Slice(elemOffset, rows * cols);
            var result = _gpu.Upload(floats, TensorShape.D1(floats.Length), exact: true);
            _weightDTypes[result.Handle] = DType.Float32;
            return result;
        }

        int bytesPerRow = (cols / DTypeInfo.BlockSize(info.DType))
                        * DTypeInfo.BytesPerBlock(info.DType);
        int expertBytes = rows * bytesPerRow;
        int byteOffset = expertIdx * expertBytes;
        var expertData = data.Slice(byteOffset, expertBytes);

        if (info.DType == DType.Q4_K || info.DType == DType.Q6_K || info.DType == DType.Q8_0)
        {
            // Q8_0 raw-upload: same Phase-0 motivation as UploadWeight — keep the
            // 1.0625 byte/elem packed layout on the GPU, dispatch the native
            // llm_matvec_q8_0 kernel from MatMul. The dequant-to-F32 fallback
            // below would burn 4× the VRAM per expert.
            var result = _gpu.UploadRaw(expertData, TensorShape.D1(expertData.Length), info.DType, exact: true);
            _weightDTypes[result.Handle] = info.DType;
            return result;
        }
        else
        {
            // Less-common dtypes: dequantize on CPU and upload as F32.
            int count = rows * cols;
            var f32 = new float[count];
            Dequantize.ToFloat32(expertData, f32, info.DType, count);
            var result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _weightDTypes[result.Handle] = DType.Float32;
            return result;
        }
    }

    /// <summary>
    /// Upload an F32 RMSNorm weight tensor. The Gemma GGUF converter already bakes
    /// the HF "(1 + w)" RMSNorm convention into the stored weights, so we upload
    /// raw — same as the CPU <c>ForwardPass.GetNormWeight</c> path.
    /// </summary>
    private Tensor UploadNormWeight(string name) => UploadWeight(name);

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // exact=true: weights live for the entire decoding session. The pool's
        // power-of-2 round-up (e.g. 17 MiB → 32 MiB) is pure waste at this lifetime;
        // exact-path goes through cudaMalloc/cudaFree directly. See #25/#26.
        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length), exact: true);
            _weightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K || info.DType == DType.Q8_0)
        {
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            // #149: repack 2-D Q8_0 GEMM weights (norms/biases are 1-D; embedding uploads
            // elsewhere) into the SoA layout. Dimensions are GGUF ne order: [cols, rows].
            if (_mmqSoa && info.DType == DType.Q8_0 && info.NDimensions == 2)
            {
                int cols = (int)info.Dimensions[0];
                int rows = (int)info.Dimensions[1];
                result = _gpu.RepackQ8_0Soa(result, rows, cols);
            }
            _weightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Less-common dtypes: dequantize on CPU and upload as F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _weightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    /// <summary>
    /// Load a single F32 scalar tensor (any source dtype) into a managed float.
    /// Used for Gemma 4's per-layer <c>layer_output_scale.weight</c>.
    /// </summary>
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

    /// <summary>
    /// Resolve a tensor name to a mmap-resident CPU reference (no upload, no copy).
    /// Used for Gemma 4 PLE tensors that must NEVER hit VRAM.
    /// </summary>
    private CudaTensorRef ResolveCpuTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new CudaTensorRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    /// <summary>
    /// CPU-resident tensor handle backed by the GGUF mmap. Mirrors the private
    /// <c>TensorRef</c> struct in <see cref="ForwardPass"/>; kept here so the CUDA
    /// path can hold a reference to PLE / per-layer-norm tensors that stay on the
    /// host without dragging in the larger ForwardPass machinery.
    /// </summary>
    private readonly unsafe struct CudaTensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public CudaTensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name; Info = info; DType = dtype; DataPtr = dataPtr;
        }
    }

    private Tensor UploadTqSignPatterns(int layerIndex)
    {
        var fullSigns = new float[_numKvHeads * _headDim];
        for (int h = 0; h < _numKvHeads; h++)
        {
            // Match the per-(layer × kv_head) seeding used by KvCacheCompressor and
            // GpuForwardPass — the sign pattern is what binds a query rotation to its
            // matching cached keys, and the seeds must align across paths.
            var headSigns = WalshHadamard.GenerateSignPattern(_headDim, layerIndex * _numKvHeads + h);
            headSigns.CopyTo(fullSigns.AsSpan(h * _headDim));
        }
        return _gpu.Upload(fullSigns, TensorShape.D1(fullSigns.Length));
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

        // Gemma 4: the per-layer-embedding table (~4.2 GB at Q8_0) is loaded
        // mmap-only and never reaches VRAM. Subtract it out of the weight budget
        // before computing free VRAM — otherwise we'd reserve >8 GB of phantom
        // space and clamp the context window to a uselessly small number.
        if (hp.HasPerLayerTokenEmbd
            && model.FindTensor("per_layer_token_embd.weight") is { } pleInfo)
        {
            weightBytes -= EstimateGpuTensorBytes(pleInfo);
        }

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

        // Gemma 4 per-layer head-dim path: each layer's K/V buffer takes its own
        // head_dim, and SWA layers cap at SlidingWindowSize regardless of the
        // global context window. Solve for the largest maxCtx s.t. the summed
        // per-layer bytes still fit in `available`. Without this branch the
        // non-gemma4 formula (NumLayers × headDim × maxCtx) wildly under- or
        // over-counts depending on which side of the head-dim mix dominates.
        if (hp.LayerHeadDim is { } lhd && hp.IsSwaLayer is { } swa)
        {
            int swaWindow = hp.SlidingWindowSize > 0 ? hp.SlidingWindowSize : int.MaxValue;
            long globalKvDimPerToken = 0;
            long swaKvDimPerToken    = 0;
            for (int i = 0; i < hp.NumLayers; i++)
            {
                // KV-share layers don't allocate their own pages (the source layer
                // already counted). Skip from both buckets.
                if (hp.KvSourceLayer is { } ksl && ksl[i] >= 0) continue;
                long layerKvDim = 2L * hp.NumKvHeads * lhd[i] * sizeof(float);
                if (swa[i]) swaKvDimPerToken    += layerKvDim;
                else        globalKvDimPerToken += layerKvDim;
            }
            // For a given maxCtx C: bytes = globalKvDimPerToken * C
            //                            + swaKvDimPerToken    * min(C, swaWindow)
            // Solve for the largest C ≤ hp.ContextLength that fits in `available`.
            // Branch on whether C ≤ swaWindow:
            //   if C ≤ swaWindow: bytes = (global+swa) * C
            //   else:             bytes = global * C + swa * swaWindow
            long globalPlusSwa = globalKvDimPerToken + swaKvDimPerToken;
            int candA = globalPlusSwa > 0 ? (int)(available / globalPlusSwa) : int.MaxValue;
            int maxCtxL;
            if (candA <= swaWindow)
            {
                maxCtxL = candA;
            }
            else
            {
                long remain = available - swaKvDimPerToken * swaWindow;
                int candB = globalKvDimPerToken > 0 && remain > 0
                    ? (int)(remain / globalKvDimPerToken) : 0;
                maxCtxL = Math.Max(swaWindow, candB);
            }
            return Math.Clamp(maxCtxL, 512, hp.ContextLength);
        }

        long bytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float);
        int maxCtx = (int)(available / bytesPerToken);
        return Math.Clamp(maxCtx, 512, hp.ContextLength);
    }

    /// <summary>
    /// Context-length estimator for the TurboQuant path: the FP32 ring buffer is fixed
    /// at <paramref name="fp32WindowSize"/> positions, the remainder live in TQ blocks
    /// (~52 bytes for head_dim=128 vs 512 bytes for FP32 — about 10× smaller per token).
    /// </summary>
    public static int EstimateMaxContextTq(GgufModel model, CudaBackend gpu, ModelHyperparams hp,
        int fp32WindowSize = 256, int bits = 3)
    {
        long vramBytes = (long)gpu.VramBytes;
        if (vramBytes <= 0) vramBytes = 8L * 1024 * 1024 * 1024;

        int headDim = hp.HeadDim;
        int blockSize = TurboQuantOps.BlockSize(bits, headDim);

        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += EstimateGpuTensorBytes(t);

        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        long reserved = Math.Max(vramBytes / 3, 2L * 1024 * 1024 * 1024);
        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024;

        long fp32Bytes = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float) * fp32WindowSize;
        long tqBytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * blockSize;

        long availableForTq = available - fp32Bytes;
        if (availableForTq <= 0) availableForTq = 64L * 1024 * 1024;

        int maxTqPositions = (int)(availableForTq / tqBytesPerToken);
        return Math.Clamp(maxTqPositions + fp32WindowSize, 512, hp.ContextLength);
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        // Raw-upload dtypes (no CPU dequant): tensor lives on GPU at its native byte size,
        // padded up to the next 4-byte boundary to match UploadRaw's uint32-strided
        // layout. Q8_0 is included here (Phase 0 of the Gemma-4 plan) — ~1.0625 bytes
        // per element vs the 4 bytes/elem the F32-fallback path would burn.
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K
            || tensor.DType == DType.Q6_K  || tensor.DType == DType.Q8_0)
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
            bool kvShared = _hp.KvSourceLayer is { } ksl && ksl[i] >= 0;

            _gpu.Free(_wAttnNorm[i]); _gpu.Free(_wFfnNorm[i]);
            _gpu.Free(_wq[i]); _gpu.Free(_wo[i]);
            // KV-share layers (Gemma 4 tail) never owned their own K/V projections.
            if (!kvShared)
            {
                _gpu.Free(_wk[i]); _gpu.Free(_wv[i]);
            }
            // Gemma 4 per-layer post-norms (small, GPU-resident).
            if (_wPostAttnNorm is not null) _gpu.Free(_wPostAttnNorm[i]);
            if (_wPostFfwNorm  is not null) _gpu.Free(_wPostFfwNorm[i]);
            if (_isMoE)
            {
                _gpu.Free(_wGateInp![i]);
                for (int e = 0; e < _hp.NumExperts; e++)
                {
                    _gpu.Free(_wGateExps![i][e]);
                    _gpu.Free(_wUpExps![i][e]);
                    _gpu.Free(_wDownExps![i][e]);
                }
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
                _gpu.Free(_bq![i]);
                if (!kvShared)
                {
                    _gpu.Free(_bk![i]); _gpu.Free(_bv![i]);
                }
                _gpu.Free(_bo![i]);
            }

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.Free(_wqNorm![i]);
                if (!kvShared) _gpu.Free(_wkNorm![i]);
            }
        }

        if (_isMoE)
        {
            if (_routerLogits is { } rl) _gpu.Free(rl);
            if (_moeExpertOut is { } eo) _gpu.Free(eo);
            if (_moeSharedOut is { } so) _gpu.Free(so);
        }
        _gpu.Free(_wOutputNorm);
        if (_wOutput.Handle != _gpuEmbedding.Handle)
            _gpu.Free(_wOutput);
        _gpu.Free(_gpuEmbedding);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            // Skip KV-aliased layers (Gemma 4 shared_kv_layers tail) — their handles
            // are owned by the source layer, which already freed them on its iteration.
            // Without this guard the second Free() on the same device pointer would
            // hit a CUDA double-free / use-after-free.
            if (_kvAliasedLayers.Contains(i)) continue;
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

        // _attnScoresScratch is shared with SnapKV's score scratch when both are
        // allocated; the SnapKV side flags ownership so we free it exactly once
        // here regardless of which path allocated it.
        if (_attnScoresScratch is { } attnScratch) _gpu.Free(attnScratch);

        // SnapKV (issue #59) scratch (only allocated when active during a prefill).
        if (_snapKvQCapture is { } qc) _gpu.Free(qc);
        if (_snapKvScoreAccum is { } sa) _gpu.Free(sa);
        if (_snapKvScoreScratchOwned && _snapKvScoreScratch is { } ss) _gpu.Free(ss);

        // Gemma 4 PLE GPU buffers.
        if (_gpuPerLayerModelProj is { } pmp) _gpu.Free(pmp);
        if (_gpuPerLayerProjNorm  is { } ppn) _gpu.Free(ppn);
        if (_gpuInpGate is not null)
            for (int i = 0; i < _gpuInpGate.Length; i++) _gpu.Free(_gpuInpGate[i]);
        if (_gpuPleProj is not null)
            for (int i = 0; i < _gpuPleProj.Length; i++) _gpu.Free(_gpuPleProj[i]);
        if (_gpuPlePostNorm is not null)
            for (int i = 0; i < _gpuPlePostNorm.Length; i++) _gpu.Free(_gpuPlePostNorm[i]);
        FreeBatchedTrunkScratch();
        if (_gpuProjSliceViews is { } psv) foreach (var v in psv) _gpu.Free(v);
        if (_gpuPleRow       is { } pr) _gpu.Free(pr);
        if (_gpuProjPerLayer is { } ppl) _gpu.Free(ppl);
        if (_gpuPleX         is { } px) _gpu.Free(px);
        if (_gpuPleY         is { } py) _gpu.Free(py);

        if (_gpuRopeFreqs    is { } rf) _gpu.Free(rf);

        _kvCache.Dispose();
    }
}
