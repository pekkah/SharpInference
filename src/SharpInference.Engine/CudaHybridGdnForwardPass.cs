using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid GPU/CPU forward pass for the qwen35moe Gated-DeltaNet (GDN) + MoE
/// architecture (Qwen3.6-35B-A3B).
///
/// Placement (Phase 6, option A from <c>docs/qwen35moe-plan.md</c>):
/// <list type="bullet">
///   <item>GDN layers (30 of 40, indices where <c>(i+1) % 4 != 0</c>): the full
///         block — joint QKV projection, depthwise conv1d, L2-norm, delta-net
///         recurrence, ssm-out projection — runs on the CPU via
///         <see cref="GdnKernels"/>. Hidden state is downloaded to a pinned host
///         buffer before the block and uploaded back after.</item>
///   <item>Attention layers (10 of 40, indices 3, 7, …, 39): GLU-gated attention
///         runs on the GPU via <see cref="CudaBackend"/>. Per-head Q/K RMSNorm,
///         partial NEOX RoPE, GQA scaled-dot-product attention, sigmoid GLU gate.
///         Q/K/V/O weights stay VRAM-resident.</item>
///   <item>MoE FFN (every layer): 256-expert top-8 router runs on GPU; experts
///         are served by <see cref="CudaExpertSlotManager"/> (SLRU lazy load);
///         the shared expert and its per-token sigmoid gate run on GPU with
///         eager-resident weights.</item>
///   <item>Embedding / output projection: GPU if VRAM permits (mirrors
///         <c>CudaHybridForwardPass.ShouldKeepFixedWeightsOnCpu</c>); else CPU.</item>
///   <item>KV cache for the 10 attention layers: VRAM (<c>_gpuKCache[layer]</c>,
///         <c>_gpuVCache[layer]</c>) — same flat layout as
///         <see cref="CudaHybridForwardPass"/>.</item>
///   <item>GDN state cache: CPU only (<see cref="GdnStateCache"/>).</item>
/// </list>
///
/// <para>
/// Per-token GPU↔CPU transfer cost: 30 GDN blocks × 2 directions × <c>embDim×4 B</c>
/// = 480 KiB/token over PCIe; ~16 µs at PCIe 4.0 ×16. Negligible compared to MoE
/// expert evaluation.
/// </para>
///
/// <para>
/// Triage policy: where an op isn't directly available on <see cref="CudaBackend"/>,
/// this implementation prefers a download → CPU → upload fallback over adding a new
/// NVRTC kernel. Per-token overhead is small (a few KiB transfer per layer) and adding
/// kernels is deferred to Phase 7 (CUDA SSM kernels) and beyond. CPU-fallback sites are
/// marked with <c>TODO(Phase6c)</c> comments to drive future kernel work.
/// </para>
///
/// <para>v1 limitations (mirrors <see cref="HybridGdnForwardPass"/>):</para>
/// <list type="bullet">
///   <item>No speculative-decoding rewind (GDN state is destructively updated;
///         <see cref="TruncateTo"/> accepts only 0 or current length).</item>
///   <item>No batched prefill — <see cref="Prefill"/> walks tokens sequentially.</item>
///   <item>No TurboQuant — KV cache is plain FP32.</item>
///   <item>No continuous batching — single-sequence only.</item>
/// </list>
/// </summary>
public sealed unsafe class CudaHybridGdnForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly CudaBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly GdnConfig _gdn;
    private readonly LayerPlacement _placement;
    private readonly int _maxSeqLen;
    private readonly int _ctxLen;

    // ── Dimensions ─────────────────────────────────────────────────────
    private readonly int _embDim;
    private readonly int _headDim;          // attention head dim (256)
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headsPerKvGroup;
    private readonly int _ropeDim;          // 64 (partial NEOX RoPE)
    private readonly int _ropeHalfDim;
    private readonly int _gdnHeadDim;       // 128
    private readonly int _gdnNumVHeads;
    private readonly int _gdnNumKHeads;
    private readonly int _gdnKvRepeat;
    private readonly int _gdnValueDim;      // 4096
    private readonly int _gdnKeyDim;        // 2048
    private readonly int _gdnConvChannels;  // 8192
    private readonly int _gdnConvKernel;
    private readonly int _numExperts;
    private readonly int _numActiveExperts;
    private readonly int _expertDim;

    // ── GPU scratch ────────────────────────────────────────────────────
    private readonly Tensor _gpuHidden;
    private readonly Tensor _gpuResidual;
    private readonly Tensor _gpuNormBuf;
    private readonly Tensor _gpuQGate;       // [numHeads * headDim * 2] = 8192 (Q‖gate interleaved per head)
    private readonly Tensor _gpuQ;           // [numHeads * headDim] = 4096
    private readonly Tensor _gpuGate;        // [numHeads * headDim] = 4096 (pre-sigmoid)
    private readonly Tensor _gpuK;           // [numKvHeads * headDim] = 512
    private readonly Tensor _gpuV;           // [numKvHeads * headDim] = 512
    private readonly Tensor _gpuAttnOut;     // [numHeads * headDim] = 4096
    private readonly Tensor _gpuAttnScratch; // attention scores spill scratch
    private readonly Tensor _gpuRouterLogits;
    private readonly Tensor _gpuFfnGate;
    private readonly Tensor _gpuFfnUp;
    private readonly Tensor _gpuExpertOut;
    private readonly Tensor _gpuSharedOut;
    private readonly Tensor _gpuLogits;
    private readonly Tensor _pinnedHidden;   // host-mappable embDim float buffer for CPU↔GPU sync

    // ── Per-layer GPU weights (sized [NumLayers]; null/default slots for the
    //    block type that doesn't apply on that layer) ──────────────────────
    private readonly Tensor[] _gpuAttnNorm;       // [L] F32
    private readonly Tensor[] _gpuPostAttnNorm;   // [L] F32
    private readonly Tensor[] _gpuWGateInp;       // [L] router weight F32 [embDim, NumExperts]
    private readonly Tensor[] _gpuWGateInpShexp;  // [L] shared-expert gate F32 [embDim]
    private readonly Tensor[] _gpuWGateShexp;     // [L]
    private readonly Tensor[] _gpuWUpShexp;       // [L]
    private readonly Tensor[] _gpuWDownShexp;     // [L]

    // Attention-only (slots at GDN layers are unused)
    private readonly Tensor[] _gpuWQGate;        // [L] attn_q (GLU-gated, output 8192)
    private readonly Tensor[] _gpuWK;            // [L]
    private readonly Tensor[] _gpuWV;            // [L]
    private readonly Tensor[] _gpuWO;            // [L]
    private readonly Tensor[] _gpuQNorm;         // [L] [headDim] F32
    private readonly Tensor[] _gpuKNorm;         // [L] [headDim] F32

    // GPU KV cache (sized [numLayers]; only attention slots are allocated)
    private readonly Tensor?[] _gpuKCache;       // [L][maxSeq * kvDim] F32
    private readonly Tensor?[] _gpuVCache;       // [L][maxSeq * kvDim] F32

    // Embedding + output
    private readonly Tensor? _gpuEmbedding;
    private readonly bool _embIsQuantized;
    private readonly Tensor? _gpuOutputNorm;
    private readonly Tensor? _gpuOutputWeight;

    // Dtype map shared with CudaExpertSlotManager so MatMul dispatch picks
    // the right matvec variant for SLRU-loaded expert tensors.
    private readonly Dictionary<nint, DType> _gpuWeightDTypes = new();
    private readonly CudaExpertSlotManager? _expertSlotManager;

    // ── CPU-side state for GDN layers ──────────────────────────────────
    private readonly GdnStateCache _gdnStateCache;
    private readonly PagedKvCache _kvCache;     // bookkeeping (block table) for attention layers; data lives on GPU

    // Per-GDN-layer F32 weights (preloaded once, decode-only)
    // Tensor refs to GGUF mmap for the larger projections (run via SimdKernels.MatVec).
    // Used by the CPU GDN block path (SHARPI_CPU_GDN=1).
    private readonly CpuWeightRef[] _cpuWQkv;        // [L] attn_qkv (output 8192)
    private readonly CpuWeightRef[] _cpuWZGate;      // [L] attn_gate (output 4096)
    private readonly CpuWeightRef[] _cpuSsmOut;      // [L] ssm_out (input 4096, output 2048)
    private readonly CpuWeightRef[] _cpuSsmAlpha;    // [L] F32 [embDim, NumVHeads]
    private readonly CpuWeightRef[] _cpuSsmBeta;     // [L] F32 [embDim, NumVHeads]

    // Tiny preloaded F32 weights (CPU GDN path).
    private readonly float*[] _ssmConv1d;            // [L][kernel * channels] — transposed [k, c]
    private readonly float*[] _ssmA;                 // [L][NumVHeads]
    private readonly float*[] _ssmDtBias;            // [L][NumVHeads]
    private readonly float*[] _ssmNormW;             // [L][gdnHeadDim]

    // GPU-resident GDN weights (per layer, only populated for GDN-type layers).
    // The new GPU GDN block path (default) uses these.
    private readonly Tensor[] _gpuWAttnQkv;          // [L] Q4_K [conv_channels, embDim]
    private readonly Tensor[] _gpuWAttnGate;         // [L] Q4_K [value_dim, embDim]
    private readonly Tensor[] _gpuWSsmOut;           // [L] Q4_K [embDim, value_dim]
    private readonly Tensor[] _gpuWSsmAlpha;         // [L] F32 [num_v_heads, embDim]
    private readonly Tensor[] _gpuWSsmBeta;          // [L] F32 [num_v_heads, embDim]
    private readonly Tensor[] _gpuSsmA;              // [L] F32 [num_v_heads]
    private readonly Tensor[] _gpuSsmDtBias;         // [L] F32 [num_v_heads]
    private readonly Tensor[] _gpuSsmNormW;          // [L] F32 [head_dim]
    private readonly Tensor[] _gpuSsmConv1d;         // [L] F32 [kernel, channels] — transposed

    // GPU-resident GDN per-sequence state, indexed by GDN-layer index (0..numGdn-1),
    // not absolute layer index. Allocated lazily on first Forward call.
    private readonly Tensor?[] _gpuGdnScanState;     // [numGdn] F32 [num_v_heads, head_dim, head_dim]
    private readonly Tensor?[] _gpuGdnConvState;     // [numGdn] F32 [kernel-1, conv_channels] oldest-first

    // GPU scratch reused across GDN layers.
    private readonly Tensor _gpuGdnQkv;              // [conv_channels]
    private readonly Tensor _gpuGdnQkvConv;          // [conv_channels] post-conv1d + SiLU
    private readonly Tensor _gpuGdnZVec;             // [value_dim]
    private readonly Tensor _gpuGdnQHead;            // [value_dim] (tiled to num_v_heads)
    private readonly Tensor _gpuGdnKHead;            // [value_dim] (tiled to num_v_heads)
    private readonly Tensor _gpuGdnVHead;            // [value_dim] (V slice copied out of QkvConv)
    private readonly Tensor _gpuGdnAlpha;            // [num_v_heads]
    private readonly Tensor _gpuGdnBeta;             // [num_v_heads]
    private readonly Tensor _gpuGdnOut;              // [value_dim]

    // CPU scratch for GDN block
    private readonly float* _cpuNormBuf;       // [embDim]
    private readonly float* _cpuHiddenOut;     // [embDim] — output of GDN block, uploaded to _gpuHidden
    private readonly float* _qkv;              // [ConvChannels] = 8192
    private readonly float* _qkvConv;          // [ConvChannels] = 8192
    private readonly float* _zVec;             // [ValueDim] = 4096
    private readonly float* _qVHeads;          // [NumVHeads*HeadDim] = 4096
    private readonly float* _kVHeads;          // [NumVHeads*HeadDim] = 4096
    private readonly float* _alpha;            // [NumVHeads]
    private readonly float* _beta;             // [NumVHeads]
    private readonly float* _gdnOut;           // [ValueDim] = 4096

    // CPU scratch used by the shared-expert scalar gate sigmoid (line ~756).
    // The attention block now runs entirely on the GPU; no per-token CPU↔GPU
    // round-trip remains.
    private readonly float* _hostQ;            // [max(qDim, embDim)] scratch for shexp gate dot+sigmoid
    private readonly float* _ropeCosTable;     // [ctxLen * (ropeDim/2)] — retained for any future CPU fallback path
    private readonly float* _ropeSinTable;

    // Top-K router readback (CPU side)
    private readonly float[] _routerBuf;
    private readonly float[] _logitsBuf;

    // Shared-expert per-token scalar gate is `sigmoid(ffn_gate_inp_shexp · x)`.
    // We need a single scalar on the GPU but CudaBackend doesn't expose a
    // dot+sigmoid op; download _gpuNormBuf and compute on CPU.
    // This is reused across layers — embDim floats per token.
    private readonly float* _cpuNormReadback;

    // Diagnostic: per-layer activation trace (env: SHARPI_TRACE_LAYERS=1).
    private static readonly bool _traceLayers =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_LAYERS") == "1";

    // CPU MoE mode: routes the MoE FFN (routed + shared experts) through CPU
    // mmap reads instead of GPU SLRU. Auto-enabled when SLRU capacity covers
    // less than ~half the total experts — at that VRAM ratio the per-token
    // PCIe upload cost from SLRU misses exceeds the cost of running MoE on CPU.
    //
    // SHARPI_CPU_MOE override values:
    //   "1" — force CPU MoE on; "0" — force GPU SLRU MoE on; unset — auto.
    // Decided in the constructor after we know SLRU capacity (see _cpuMoe init).
    private readonly bool _cpuMoe;

    // SHARPI_CPU_GDN=1 forces the legacy CPU GDN block path (Phase 7a baseline).
    // Default (unset) is the new full-GPU GDN block (Phase 7e+). Useful for
    // bisecting parity bugs and confirming the GPU kernels match CPU output.
    private readonly bool _cpuGdn =
        Environment.GetEnvironmentVariable("SHARPI_CPU_GDN") == "1";

    // ── CPU MoE state (only allocated/populated when _cpuMoe == true) ──
    // Packed MoE weight refs (mmap pointers; routed experts stay quantized on disk).
    private readonly CpuWeightRef[]? _cpuFfnGateInp;       // [L] router F32 [embDim, numExperts]
    private readonly CpuWeightRef[]? _cpuFfnGateExps;      // [L] packed [numExperts, expertDim, embDim]
    private readonly CpuWeightRef[]? _cpuFfnUpExps;        // [L] packed
    private readonly CpuWeightRef[]? _cpuFfnDownExps;      // [L] packed
    private readonly float*[]? _cpuFfnGateInpShexp;        // [L][embDim] F32 (preloaded; small)

    // CPU scratch for the MoE FFN path.
    private readonly float* _cpuRouterLogits;   // [numExperts]
    private readonly float* _cpuSharedOut;      // [embDim]
    // Batched-expert intermediate buffers: gate/up for all 8 routed experts laid
    // out as [numActive × expertDim]. The routed loop folds 16 sequential
    // Parallel.For dispatches per layer into 2 (gate+up sweep, down+accumulate
    // sweep), amortising TPL dispatch over much larger work units.
    private readonly float* _cpuExpertGateAll;
    private readonly float* _cpuExpertUpAll;
    private readonly float* _cpuMoeHidden;      // [embDim] — accumulator written back to _gpuHidden

    // ── CPU dense FFN state (qwen35 27B-MTP and other dense hybrid GDN variants).
    //    Only allocated when !_hp.IsMoE. Mirrors the CPU MoE FFN pattern: weights
    //    stay mmap'd; per-token download GPU norm → CPU FFN → upload to GPU hidden.
    private readonly CpuWeightRef[]? _cpuWFfnGate;     // [L] ffn_gate.weight (Q4_K)
    private readonly CpuWeightRef[]? _cpuWFfnUp;       // [L] ffn_up.weight (Q4_K)
    private readonly CpuWeightRef[]? _cpuWFfnDown;     // [L] ffn_down.weight (Q6_K)
    // Per-layer GPU FFN slots. Populated lazily by TryUploadDenseFfnLayers when
    // VRAM headroom allows. Null slots → CpuDenseFfn(layer); non-null → GpuDenseFfn(layer).
    // Not readonly because populated inside a helper called from the constructor.
    private Tensor?[]? _gpuWFfnGate;          // [L] uploaded ffn_gate (Q4_K or matching dtype)
    private Tensor?[]? _gpuWFfnUp;            // [L] uploaded ffn_up
    private Tensor?[]? _gpuWFfnDown;          // [L] uploaded ffn_down (Q6_K typically)
    // GPU FFN scratch (allocated only when at least one layer is on GPU).
    private Tensor? _gpuFfnGateBufDense;      // [_intermDim] f32
    private Tensor? _gpuFfnUpBufDense;        // [_intermDim] f32
    private int _denseFfnGpuLayers;           // count of layers with FFN on GPU (diagnostic)
    private readonly float* _cpuFfnGateBuf;            // [_intermDim] scratch
    private readonly float* _cpuFfnUpBuf;              // [_intermDim] scratch
    private readonly int _intermDim;                   // hp.IntermediateDim (dense); 0 for MoE

    // ── MTP / NEXTN head (issue #25 / #29) ─────────────────────────────
    // Mirror of HybridGdnForwardPass MTP fields, plus GPU-resident equivalents
    // for the per-step forward path. Loaded when hp.NumMtpLayers > 0 AND the
    // expected nextn.* tensors are present in the GGUF.
    private readonly bool _hasMtp;
    private readonly PagedKvCache? _mtpKvCache;  // length bookkeeping; data lives on GPU

    // GPU-resident MTP block weights (standard attn+FFN layout, same as a main
    // full-attention layer; null when !_hasMtp).
    private readonly Tensor _gpuMtpAttnNorm;
    private readonly Tensor _gpuMtpWQGate;        // attn_q (Q‖gate interleaved, output qDim*2)
    private readonly Tensor _gpuMtpWK;
    private readonly Tensor _gpuMtpWV;
    private readonly Tensor _gpuMtpWO;
    private readonly Tensor _gpuMtpQNorm;
    private readonly Tensor _gpuMtpKNorm;
    private readonly Tensor _gpuMtpPostAttnNorm;
    private readonly Tensor _gpuMtpFfnGate;
    private readonly Tensor _gpuMtpFfnUp;
    private readonly Tensor _gpuMtpFfnDown;

    // nextn.* — pre-fc norms, eh_proj (Q8_0 → F32 at load), shared_head_norm.
    private readonly Tensor _gpuMtpEnorm;
    private readonly Tensor _gpuMtpHnorm;
    private readonly Tensor _gpuMtpSharedHeadNorm;
    private readonly Tensor _gpuMtpEhProj;        // F32 [embDim, embDim*2]

    // GPU MTP KV cache (single attention layer, flat ring layout same as
    // _gpuKCache[layer] in the trunk).
    private readonly Tensor? _gpuMtpKCache;
    private readonly Tensor? _gpuMtpVCache;

    // Per-step scratch on GPU.
    private readonly Tensor _gpuMtpEmbedBuf;      // [embDim]
    private readonly Tensor _gpuMtpEnormBuf;      // [embDim]
    private readonly Tensor _gpuMtpHnormBuf;      // [embDim]
    private readonly Tensor _gpuMtpConcatBuf;     // [embDim * 2]
    private readonly Tensor _gpuLastHidden;       // [embDim] — pre-output-norm hidden snapshot

    // Host LastHidden buffer; refreshed via Download after each main Forward so
    // IForwardPass.LastHidden returns a host span without an extra device read.
    private readonly float* _lastHidden;

    private bool _disposed;

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public LayerPlacement Placement => _placement;

    public CudaHybridGdnForwardPass(GgufModel model, CudaBackend gpu, ModelHyperparams hp,
        LayerPlacement placement, int maxContextLength = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(hp);
        ArgumentNullException.ThrowIfNull(placement);
        if (!hp.IsHybridSsm)
            throw new ArgumentException("CudaHybridGdnForwardPass requires hp.IsHybridSsm=true.", nameof(hp));
        if (hp.Gdn is null)
            throw new ArgumentException("CudaHybridGdnForwardPass requires hp.Gdn != null.", nameof(hp));
        if (hp.LayerTypes is null)
            throw new ArgumentException("CudaHybridGdnForwardPass requires hp.LayerTypes != null.", nameof(hp));
        if (hp.IsMoE && !hp.HasSharedExpert)
            throw new ArgumentException("CudaHybridGdnForwardPass with MoE requires a shared expert (qwen35moe layout).", nameof(hp));
        if (!hp.IsMoE && hp.IntermediateDim <= 0)
            throw new ArgumentException("CudaHybridGdnForwardPass dense FFN requires hp.IntermediateDim > 0 (qwen35 dense layout).", nameof(hp));

        _model = model;
        _gpu = gpu;
        _hp = hp;
        _gdn = hp.Gdn;
        _placement = placement;
        _maxSeqLen = placement.RecommendedCtxSize > 0
            ? placement.RecommendedCtxSize
            : Math.Min(hp.ContextLength, 32768);

        _ctxLen = _maxSeqLen;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = _numHeads / _numKvHeads;
        _ropeDim = hp.RopeDim;
        _ropeHalfDim = _ropeDim / 2;
        _gdnHeadDim = _gdn.HeadDim;
        _gdnNumVHeads = _gdn.NumVHeads;
        _gdnNumKHeads = _gdn.NumKHeads;
        _gdnKvRepeat = _gdnNumVHeads / _gdnNumKHeads;
        _gdnValueDim = _gdn.ValueDim;
        _gdnKeyDim = _gdn.KeyDim;
        _gdnConvChannels = _gdn.ConvChannels;
        _gdnConvKernel = _gdn.ConvKernel;
        _numExperts = hp.NumExperts;
        _numActiveExperts = hp.NumActiveExperts;
        _expertDim = hp.ExpertIntermediateDim;

        int L = hp.NumLayers;
        int qDim = _numHeads * _headDim;        // 4096
        int kvDim = _numKvHeads * _headDim;     // 512

        Console.Error.WriteLine($"[CudaHybridGdnForwardPass] layers={L} embDim={_embDim} headDim={_headDim} numHeads={_numHeads} ropeDim={_ropeDim} ctx={_ctxLen}");
        Console.Error.WriteLine($"[CudaHybridGdnForwardPass] GDN: heads={_gdnNumVHeads}v×{_gdnNumKHeads}k headDim={_gdnHeadDim} conv={_gdnConvChannels}×{_gdnConvKernel} MoE: {_numExperts}exp×{_numActiveExperts}active dim={_expertDim}");

        bool vramTrace = Environment.GetEnvironmentVariable("SHARPI_TRACE_VRAM") == "1";
        void TraceVram(string label)
        {
            if (vramTrace)
                Console.Error.WriteLine($"[VRAM] {label}: free={gpu.FreeVramBytes / (1024 * 1024)} MiB");
        }
        TraceVram("constructor entry");

        // ── Allocate GPU scratch ───────────────────────────────────────
        _gpuHidden = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuResidual = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuNormBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuQGate = gpu.Allocate(TensorShape.D1(qDim * 2));
        _gpuQ = gpu.Allocate(TensorShape.D1(qDim));
        _gpuGate = gpu.Allocate(TensorShape.D1(qDim));
        _gpuK = gpu.Allocate(TensorShape.D1(kvDim));
        _gpuV = gpu.Allocate(TensorShape.D1(kvDim));
        _gpuAttnOut = gpu.Allocate(TensorShape.D1(qDim));
        // Attention scores scratch — only needed when ctx > 4096; otherwise placeholder.
        long scratchElems = _maxSeqLen > 4096 ? (long)_numHeads * _maxSeqLen : 1L;
        _gpuAttnScratch = gpu.Allocate(TensorShape.D1(scratchElems));
        // MoE-only GPU scratch: router logits, expert intermediate buffers, shared
        // expert output. For dense FFN (qwen35 27B-MTP) these are unused and
        // _numExperts=_expertDim=0, so skip allocation entirely. Fields are
        // initialized to null! and gated by hp.IsMoE at every access site.
        if (hp.IsMoE)
        {
            _gpuRouterLogits = gpu.Allocate(TensorShape.D1(_numExperts));
            _gpuFfnGate = gpu.Allocate(TensorShape.D1(_expertDim));
            _gpuFfnUp = gpu.Allocate(TensorShape.D1(_expertDim));
            _gpuExpertOut = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuSharedOut = gpu.Allocate(TensorShape.D1(_embDim));
        }
        else
        {
            _gpuRouterLogits = null!;
            _gpuFfnGate = null!;
            _gpuFfnUp = null!;
            _gpuExpertOut = null!;
            _gpuSharedOut = null!;
        }
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));
        TraceVram("after GPU scratch + logits + pinned host");

        // GDN GPU scratch (per-layer; reused).
        _gpuGdnQkv     = gpu.Allocate(TensorShape.D1(_gdnConvChannels));
        _gpuGdnQkvConv = gpu.Allocate(TensorShape.D1(_gdnConvChannels));
        _gpuGdnZVec    = gpu.Allocate(TensorShape.D1(_gdnValueDim));
        _gpuGdnQHead   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads * _gdnHeadDim));
        _gpuGdnKHead   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads * _gdnHeadDim));
        _gpuGdnVHead   = gpu.Allocate(TensorShape.D1(_gdnValueDim));
        _gpuGdnAlpha   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads));
        _gpuGdnBeta    = gpu.Allocate(TensorShape.D1(_gdnNumVHeads));
        _gpuGdnOut     = gpu.Allocate(TensorShape.D1(_gdnValueDim));

        _routerBuf = new float[_numExperts];
        _logitsBuf = new float[hp.VocabSize];

        // ── CPU scratch ────────────────────────────────────────────────
        _cpuNormBuf = Alloc(_embDim);
        _cpuHiddenOut = Alloc(_embDim);
        _qkv = Alloc(_gdnConvChannels);
        _qkvConv = Alloc(_gdnConvChannels);
        _zVec = Alloc(_gdnValueDim);
        _qVHeads = Alloc(_gdnNumVHeads * _gdnHeadDim);
        _kVHeads = Alloc(_gdnNumVHeads * _gdnHeadDim);
        _alpha = Alloc(_gdnNumVHeads);
        _beta = Alloc(_gdnNumVHeads);
        _gdnOut = Alloc(_gdnValueDim);
        // _hostQ doubles as the shexp readback buffer (~embDim) and as occasional
        // attention-scratch for any future debugging download. Size it for whichever
        // is larger so the same allocation serves both.
        _hostQ = Alloc(Math.Max(qDim, _embDim));
        _cpuNormReadback = Alloc(_embDim);

        // RoPE tables for partial NEOX rotation (ropeDim/2 entries per position).
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)_ctxLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)_ctxLen * _ropeHalfDim * sizeof(float)));
        SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, _ctxLen, _ropeDim, hp.RopeTheta);

        // ── Caches ─────────────────────────────────────────────────────
        // PagedKvCache here is used purely for layer-0 ReserveBlock semantics +
        // length bookkeeping. The actual KV data lives on the GPU in _gpuKCache /
        // _gpuVCache (flat ring layout, position-indexed). We still need
        // ReserveBlock to be called once per token before any attention layer
        // appends, so reuse PagedKvCache rather than reimplement its accounting.
        _kvCache = new PagedKvCache(L, _numKvHeads, _headDim);
        _gdnStateCache = new GdnStateCache(hp.LayerTypes, _gdn);

        // ── Decide embedding/output placement (mirrors CudaHybridForwardPass.ShouldKeepFixedWeightsOnCpu) ──
        bool cpuFixedWeights = ShouldKeepFixedWeightsOnCpu(
            model.FindTensor("token_embd.weight")!.Value,
            model.FindTensor("output.weight"));

        if (!cpuFixedWeights)
        {
            TraceVram("before embedding upload");
            _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embIsQuantized);
            TraceVram("after embedding upload");
            _gpuOutputNorm = UploadWeight("output_norm.weight");
            _gpuOutputWeight = model.FindTensor("output.weight") is not null
                ? UploadWeight("output.weight")
                : _gpuEmbedding;
            TraceVram("after output.weight upload");
        }
        else
        {
            // Phase 6 limitation: this class assumes embedding+output fit on GPU.
            // The qwen35moe model with Q8_0 embedding has F32-expanded footprint
            // that may exceed the 2GB single-allocation limit on some drivers.
            // If we hit this, the CPU-embedding fallback path mirrors CudaHybridForwardPass.
            throw new NotSupportedException(
                "CudaHybridGdnForwardPass: embedding/output do not fit on GPU; " +
                "CPU embedding fallback is not implemented in v1. Reduce ctx size or " +
                "use HybridGdnForwardPass for CPU-only execution.");
        }

        // ── Auto-detect CPU-MoE vs GPU SLRU MoE ─────────────────────────
        // SLRU only pays off when most experts fit in VRAM. Predict the slot
        // count from the same formula the SLRU itself uses, before uploading
        // any MoE weights. Threshold: route MoE through CPU when fewer than
        // ~half of all experts can be cached on the GPU — at that point
        // per-token miss-driven PCIe uploads cost more than CPU mmap reads.
        // (Measured at 35 % capacity / 80 % hit rate: GPU SLRU 6.1 t/s,
        //  CPU MoE 11.8 t/s on Qwen3.6-35B-A3B on a 4070 Ti.)
        if (hp.IsMoE)
        {
            string? cpuMoeOverride = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
            if (cpuMoeOverride == "1")
            {
                _cpuMoe = true;
            }
            else if (cpuMoeOverride == "0")
            {
                _cpuMoe = false;
            }
            else
            {
                int predictedSlots = PredictSlruSlots(L);
                int totalExperts = L * _numExperts;
                double ratio = totalExperts > 0 ? (double)predictedSlots / totalExperts : 1.0;
                _cpuMoe = ratio < 0.5;
                Console.Error.WriteLine(
                    $"[CudaHybridGdnForwardPass] MoE auto-select: SLRU capacity ≈ {predictedSlots}/{totalExperts} ({ratio:P0}) → {(_cpuMoe ? "CPU" : "GPU SLRU")} MoE.  Override with SHARPI_CPU_MOE=0|1.");
            }
        }
        else
        {
            // Dense FFN variant (qwen35 27B-MTP): no MoE routing, no SLRU. FFN weights
            // stay mmap'd and run on CPU per layer; _cpuMoe is unused on this path.
            _cpuMoe = false;
            _intermDim = hp.IntermediateDim;
            Console.Error.WriteLine(
                $"[CudaHybridGdnForwardPass] Dense FFN mode (intermDim={_intermDim}): per-layer ffn_gate/up/down run on CPU from mmap; attn + GDN stay on GPU.");
        }

        // ── Per-layer tensor arrays ────────────────────────────────────
        _gpuAttnNorm = new Tensor[L];
        _gpuPostAttnNorm = new Tensor[L];
        _gpuWGateInp = new Tensor[L];
        _gpuWGateInpShexp = new Tensor[L];
        _gpuWGateShexp = new Tensor[L];
        _gpuWUpShexp = new Tensor[L];
        _gpuWDownShexp = new Tensor[L];
        _gpuWQGate = new Tensor[L];
        _gpuWK = new Tensor[L];
        _gpuWV = new Tensor[L];
        _gpuWO = new Tensor[L];
        _gpuQNorm = new Tensor[L];
        _gpuKNorm = new Tensor[L];
        _gpuKCache = new Tensor?[L];
        _gpuVCache = new Tensor?[L];

        _cpuWQkv = new CpuWeightRef[L];
        _cpuWZGate = new CpuWeightRef[L];
        _cpuSsmOut = new CpuWeightRef[L];
        _cpuSsmAlpha = new CpuWeightRef[L];
        _cpuSsmBeta = new CpuWeightRef[L];
        _ssmConv1d = new float*[L];
        _ssmA = new float*[L];
        _ssmDtBias = new float*[L];
        _ssmNormW = new float*[L];

        _gpuWAttnQkv = new Tensor[L];
        _gpuWAttnGate = new Tensor[L];
        _gpuWSsmOut = new Tensor[L];
        _gpuWSsmAlpha = new Tensor[L];
        _gpuWSsmBeta = new Tensor[L];
        _gpuSsmA = new Tensor[L];
        _gpuSsmDtBias = new Tensor[L];
        _gpuSsmNormW = new Tensor[L];
        _gpuSsmConv1d = new Tensor[L];
        _gpuGdnScanState = new Tensor?[L];
        _gpuGdnConvState = new Tensor?[L];

        if (_cpuMoe)
        {
            Console.Error.WriteLine(
                "[CudaHybridGdnForwardPass] CPU MoE mode: routed experts run on CPU (mmap, ~2.3 GB/s); shared expert stays on GPU, overlapped with the CPU routed loop. SLRU disabled.");
            _cpuFfnGateInp = new CpuWeightRef[L];
            _cpuFfnGateExps = new CpuWeightRef[L];
            _cpuFfnUpExps = new CpuWeightRef[L];
            _cpuFfnDownExps = new CpuWeightRef[L];
            _cpuFfnGateInpShexp = new float*[L];

            _cpuRouterLogits = Alloc(_numExperts);
            _cpuSharedOut = Alloc(_embDim);
            _cpuExpertGateAll = Alloc(_numActiveExperts * _expertDim);
            _cpuExpertUpAll = Alloc(_numActiveExperts * _expertDim);
            _cpuMoeHidden = Alloc(_embDim);
        }
        else if (!hp.IsMoE)
        {
            // Dense FFN variant: always run FFN on CPU. Need per-layer weight refs
            // and scratch sized to the dense intermediate dim. _cpuMoeHidden serves
            // as the upload buffer back to GPU after the FFN.
            _cpuWFfnGate = new CpuWeightRef[L];
            _cpuWFfnUp   = new CpuWeightRef[L];
            _cpuWFfnDown = new CpuWeightRef[L];
            _cpuFfnGateBuf = Alloc(_intermDim);
            _cpuFfnUpBuf   = Alloc(_intermDim);
            _cpuMoeHidden  = Alloc(_embDim);

            _cpuRouterLogits = null;
            _cpuSharedOut = null;
            _cpuExpertGateAll = null;
            _cpuExpertUpAll = null;
        }
        else
        {
            _cpuRouterLogits = null;
            _cpuSharedOut = null;
            _cpuExpertGateAll = null;
            _cpuExpertUpAll = null;
            _cpuMoeHidden = null;
        }

        Console.Error.Write("[CudaHybridGdnForwardPass] Uploading per-layer weights...");
        for (int i = 0; i < L; i++)
        {
            // Common (both block types): norms + MoE FFN weights live on GPU.
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuPostAttnNorm[i] = UploadWeight($"blk.{i}.post_attention_norm.weight");

            if (hp.IsMoE)
            {
                // Shared expert weights stay GPU-resident in both modes so the CPU MoE
                // path can fire them off in parallel with the routed expert loop
                // (saves ~5 MiB × NumLayers of CPU↔mem bandwidth per token).
                _gpuWGateShexp[i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                _gpuWUpShexp[i] = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                _gpuWDownShexp[i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");

                if (!_cpuMoe)
                {
                    _gpuWGateInp[i] = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                    _gpuWGateInpShexp[i] = UploadWeight($"blk.{i}.ffn_gate_inp_shexp.weight");
                }
                else
                {
                    _cpuFfnGateInp![i] = ResolveCpuWeight($"blk.{i}.ffn_gate_inp.weight");
                    _cpuFfnGateExps![i] = ResolveCpuWeight($"blk.{i}.ffn_gate_exps.weight");
                    _cpuFfnUpExps![i] = ResolveCpuWeight($"blk.{i}.ffn_up_exps.weight");
                    _cpuFfnDownExps![i] = ResolveCpuWeight($"blk.{i}.ffn_down_exps.weight");
                    _cpuFfnGateInpShexp![i] = LoadF32Tensor($"blk.{i}.ffn_gate_inp_shexp.weight", _embDim);
                }
            }
            else
            {
                // Dense FFN (qwen35 27B-MTP): resolve mmap refs only; CPU FFN reads
                // them per token. No GPU upload — 8.6 GB of dense FFN won't fit in
                // 12 GB VRAM alongside attention/GDN/embed/output weights.
                _cpuWFfnGate![i] = ResolveCpuWeight($"blk.{i}.ffn_gate.weight");
                _cpuWFfnUp![i]   = ResolveCpuWeight($"blk.{i}.ffn_up.weight");
                _cpuWFfnDown![i] = ResolveCpuWeight($"blk.{i}.ffn_down.weight");
            }

            if (hp.LayerTypes[i] == LayerType.Attention)
            {
                _gpuWQGate[i] = UploadWeight($"blk.{i}.attn_q.weight");
                _gpuWK[i] = UploadWeight($"blk.{i}.attn_k.weight");
                _gpuWV[i] = UploadWeight($"blk.{i}.attn_v.weight");
                _gpuWO[i] = UploadWeight($"blk.{i}.attn_output.weight");
                _gpuQNorm[i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _gpuKNorm[i] = UploadWeight($"blk.{i}.attn_k_norm.weight");

                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            }
            else
            {
                if (_cpuGdn)
                {
                    // CPU GDN path (regression / parity validation).
                    _cpuWQkv[i] = ResolveCpuWeight($"blk.{i}.attn_qkv.weight");
                    _cpuWZGate[i] = ResolveCpuWeight($"blk.{i}.attn_gate.weight");
                    _cpuSsmOut[i] = ResolveCpuWeight($"blk.{i}.ssm_out.weight");
                    _cpuSsmAlpha[i] = ResolveCpuWeight($"blk.{i}.ssm_alpha.weight");
                    _cpuSsmBeta[i] = ResolveCpuWeight($"blk.{i}.ssm_beta.weight");

                    _ssmA[i] = LoadF32Tensor($"blk.{i}.ssm_a", _gdnNumVHeads);
                    _ssmDtBias[i] = LoadF32Tensor($"blk.{i}.ssm_dt.bias", _gdnNumVHeads);
                    _ssmNormW[i] = LoadF32Tensor($"blk.{i}.ssm_norm.weight", _gdnHeadDim);
                    _ssmConv1d[i] = LoadConv1dTransposed($"blk.{i}.ssm_conv1d.weight",
                        _gdnConvKernel, _gdnConvChannels);
                }
                else
                {
                    // GPU GDN path (default; Phase 7e).
                    _gpuWAttnQkv[i]   = UploadWeight($"blk.{i}.attn_qkv.weight");
                    _gpuWAttnGate[i]  = UploadWeight($"blk.{i}.attn_gate.weight");
                    _gpuWSsmOut[i]    = UploadWeight($"blk.{i}.ssm_out.weight");
                    _gpuWSsmAlpha[i]  = UploadWeight($"blk.{i}.ssm_alpha.weight");
                    _gpuWSsmBeta[i]   = UploadWeight($"blk.{i}.ssm_beta.weight");

                    _gpuSsmA[i]       = UploadWeight($"blk.{i}.ssm_a");
                    _gpuSsmDtBias[i]  = UploadWeight($"blk.{i}.ssm_dt.bias");
                    _gpuSsmNormW[i]   = UploadWeight($"blk.{i}.ssm_norm.weight");
                    _gpuSsmConv1d[i]  = UploadConv1dTransposedToGpu($"blk.{i}.ssm_conv1d.weight",
                        _gdnConvKernel, _gdnConvChannels);

                    // Allocate per-layer GDN state on GPU (numVHeads × headDim × headDim scan + conv ring).
                    long scanFloats = (long)_gdnNumVHeads * _gdnHeadDim * _gdnHeadDim;
                    long convFloats = (long)(_gdnConvKernel - 1) * _gdnConvChannels;
                    var scan = gpu.Allocate(TensorShape.D1(scanFloats));
                    var conv = gpu.Allocate(TensorShape.D1(convFloats));
                    gpu.Clear(scan);
                    gpu.Clear(conv);
                    _gpuGdnScanState[i] = scan;
                    _gpuGdnConvState[i] = conv;
                }
            }
            if ((i % 4) == 3) Console.Error.Write(".");
        }
        Console.Error.WriteLine(" done.");
        TraceVram("after per-layer weight upload");

        // ── SLRU expert slot manager ───────────────────────────────────
        // Compute slot capacity from remaining VRAM. The plan calls for sizing
        // capacity by (remaining VRAM) / (per-expert bytes). For qwen35moe Q4_K_M:
        //   per expert ≈ 1.81 MiB (gate+up+down for one expert across 3 tensors)
        // We're conservative — most of the remaining budget is reserved for the
        // attention KV cache (10 layers × maxSeqLen × kvDim × 4 B × 2) and various
        // scratch. Use the GpuKvBytes from placement when the planner sized it.
        if (!hp.IsMoE)
        {
            // Dense FFN — no expert slot manager. Per-layer FFN-on-GPU upload runs
            // below after we know real free VRAM. Reads via GpuDenseFfn / CpuDenseFfn
            // depending on whether the layer's _gpuWFfn* slot was populated.
            _expertSlotManager = null;
            TryUploadDenseFfnLayers(gpu, hp, L);
        }
        else if (_cpuMoe)
        {
            _expertSlotManager = null;
        }
        else
        {
            ulong vramTotal = gpu.VramBytes;
            long allocated = EstimateUploadedVram(); // best-effort estimate of weights uploaded so far
            long remaining = (long)vramTotal - allocated - (2L << 30); // reserve 2 GiB for headroom
            long perExpertBytes = EstimatePerExpertBytes();
            int capacity = perExpertBytes > 0 ? (int)Math.Max(64, remaining / perExpertBytes) : 1024;
            // Cap at total expert count (all layers × num experts).
            int totalExperts = L * _numExperts;
            capacity = Math.Min(capacity, totalExperts);
            Console.Error.WriteLine($"[CudaHybridGdnForwardPass] SLRU expert cache: {capacity} slots / {totalExperts} total experts (per-expert ≈ {perExpertBytes / 1024} KiB, remaining VRAM ≈ {remaining / (1024 * 1024)} MiB).");
            _expertSlotManager = new CudaExpertSlotManager(gpu, model, hp, capacity, _gpuWeightDTypes);
        }

        // ── MTP / NEXTN head on GPU (issue #29; mirror of HybridGdnForwardPass) ──
        // Loaded when the GGUF reports nextn_predict_layers > 0 AND the expected
        // tensors exist at blk.{NumLayers}. Multi-head MTP (NumMtpLayers > 1) is
        // out of scope for v1; only the first head is loaded.
        _hasMtp = hp.NumMtpLayers > 0
                  && model.FindTensor($"blk.{hp.NumLayers}.nextn.eh_proj.weight") is not null;
        if (_hasMtp)
        {
            int mtpLayerIdx = hp.NumLayers;
            TraceVram("before MTP head upload");

            _gpuMtpAttnNorm       = UploadWeight($"blk.{mtpLayerIdx}.attn_norm.weight");
            _gpuMtpWQGate         = UploadWeight($"blk.{mtpLayerIdx}.attn_q.weight");
            _gpuMtpWK             = UploadWeight($"blk.{mtpLayerIdx}.attn_k.weight");
            _gpuMtpWV             = UploadWeight($"blk.{mtpLayerIdx}.attn_v.weight");
            _gpuMtpWO             = UploadWeight($"blk.{mtpLayerIdx}.attn_output.weight");
            _gpuMtpQNorm          = UploadWeight($"blk.{mtpLayerIdx}.attn_q_norm.weight");
            _gpuMtpKNorm          = UploadWeight($"blk.{mtpLayerIdx}.attn_k_norm.weight");
            _gpuMtpPostAttnNorm   = UploadWeight($"blk.{mtpLayerIdx}.post_attention_norm.weight");
            _gpuMtpFfnGate        = UploadWeight($"blk.{mtpLayerIdx}.ffn_gate.weight");
            _gpuMtpFfnUp          = UploadWeight($"blk.{mtpLayerIdx}.ffn_up.weight");
            _gpuMtpFfnDown        = UploadWeight($"blk.{mtpLayerIdx}.ffn_down.weight");

            _gpuMtpEnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.enorm.weight");
            _gpuMtpHnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.hnorm.weight");
            _gpuMtpSharedHeadNorm = UploadWeight($"blk.{mtpLayerIdx}.nextn.shared_head_norm.weight");
            // eh_proj is Q8_0 in GGUF; UploadWeight dequants to F32 on the path
            // for any dtype not in {F32, Q4_K, Q5_K, Q6_K}, so this lands as F32
            // and the CudaBackend.MatMul fp32 path serves it. ~200 MiB residence.
            _gpuMtpEhProj         = UploadWeight($"blk.{mtpLayerIdx}.nextn.eh_proj.weight");

            // MTP attention KV cache on GPU (one slot; same layout as trunk KV).
            int mtpKvDim = _numKvHeads * _headDim;
            _gpuMtpKCache = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * mtpKvDim));
            _gpuMtpVCache = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * mtpKvDim));
            gpu.Clear(_gpuMtpKCache);
            gpu.Clear(_gpuMtpVCache);

            // Per-step scratch.
            _gpuMtpEmbedBuf  = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuMtpEnormBuf  = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuMtpHnormBuf  = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuMtpConcatBuf = gpu.Allocate(TensorShape.D1(_embDim * 2));
            _gpuLastHidden   = gpu.Allocate(TensorShape.D1(_embDim));

            // Bookkeeping cache (PagedKvCache for layer-0 invariant + length tracking).
            _mtpKvCache = new PagedKvCache(numLayers: 1, _numKvHeads, _headDim);

            // Host LastHidden buffer (downloaded after each main Forward).
            _lastHidden = Alloc(_embDim);

            // GPU dense FFN scratch is allocated by TryUploadDenseFfnLayers only
            // when at least one trunk FFN layer lands on GPU. For MTP we need it
            // regardless — the MTP block's dense FFN runs on GPU even if all
            // trunk FFN layers stayed on CPU (which is unusual but possible
            // under tight VRAM). Allocate here when missing; the cost is
            // 2 × intermDim × 4 B = 140 KiB for 27B's intermDim=17408.
            if (!hp.IsMoE && _gpuFfnGateBufDense is null)
            {
                _gpuFfnGateBufDense = gpu.Allocate(TensorShape.D1(_intermDim));
                _gpuFfnUpBufDense   = gpu.Allocate(TensorShape.D1(_intermDim));
            }

            TraceVram("after MTP head upload");
        }
        else
        {
            // Null-forgiving assigns so the non-nullable struct fields satisfy
            // the constructor analyser. Every access site gates on _hasMtp.
            _gpuMtpAttnNorm = _gpuMtpWQGate = _gpuMtpWK = _gpuMtpWV = _gpuMtpWO =
                _gpuMtpQNorm = _gpuMtpKNorm = _gpuMtpPostAttnNorm =
                _gpuMtpFfnGate = _gpuMtpFfnUp = _gpuMtpFfnDown =
                _gpuMtpEnorm = _gpuMtpHnorm = _gpuMtpSharedHeadNorm =
                _gpuMtpEhProj =
                _gpuMtpEmbedBuf = _gpuMtpEnormBuf = _gpuMtpHnormBuf =
                _gpuMtpConcatBuf = _gpuLastHidden = null!;
            _lastHidden = null;
        }
    }

    // =================================================================
    //  IForwardPass surface
    // =================================================================

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    public void TruncateTo(int length)
    {
        if (length == _gdnStateCache.Length)
        {
            _kvCache.TruncateTo(length);
            return;
        }
        if (length == 0)
        {
            ResetCache();
            return;
        }
        if (length == _snapshotLength && _snapshotLength >= 0)
        {
            // Issue #21: restore from the end-of-decode snapshot.
            // 1. Pull the host-side cache from the snapshot buffer.
            _gdnStateCache.RestoreFrom(_snapshotBuf, _snapshotCap);
            // 2. On the GPU GDN path, upload each per-layer host buffer back to
            //    the device tensors. CPU GDN path has nothing extra to do — state
            //    already lives in _gdnStateCache.
            if (!_cpuGdn)
            {
                int scanFloats = _gdnStateCache.ScanStateFloatsPerLayer;
                int convFloats = _gdnStateCache.ConvStateFloatsPerLayer;
                for (int layer = 0; layer < _hp.NumLayers; layer++)
                {
                    int g = _gdnStateCache.GdnLayerOf(layer);
                    if (g < 0) continue;
                    if (_gpuGdnScanState[layer] is { } scanT && scanFloats > 0)
                    {
                        float* hostScan = _gdnStateCache.ScanStateAt(g);
                        _gpu.UploadInto(scanT, new ReadOnlySpan<float>(hostScan, scanFloats));
                    }
                    if (_gpuGdnConvState[layer] is { } convT && convFloats > 0)
                    {
                        float* hostConv = _gdnStateCache.ConvStateAt(g);
                        _gpu.UploadInto(convT, new ReadOnlySpan<float>(hostConv, convFloats));
                    }
                }
            }
            _kvCache.TruncateTo(length);
            return;
        }
        throw new NotSupportedException(
            $"CudaHybridGdnForwardPass.TruncateTo({length}): GDN state is destructively " +
            $"updated and cannot be partially rewound; only length == 0, current ({_gdnStateCache.Length}), " +
            $"or SnapshotLength ({_snapshotLength}) is supported. " +
            "SupportsPartialRewind == false on this pass — callers should check it before invoking " +
            "TruncateTo with an intermediate length.");
    }

    public void ResetCache()
    {
        _kvCache.Reset();
        _gdnStateCache.Reset();
        ClearSnapshot();
        if (!_cpuGdn)
        {
            // Zero GPU-resident scan + conv state for every GDN layer.
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_gpuGdnScanState[i] is { } scan) _gpu.Clear(scan);
                if (_gpuGdnConvState[i] is { } conv) _gpu.Clear(conv);
            }
        }
        if (_hasMtp)
        {
            _mtpKvCache?.Reset();
            if (_gpuMtpKCache is { } kT) _gpu.Clear(kT);
            if (_gpuMtpVCache is { } vT) _gpu.Clear(vT);
        }
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => false;

    // ── Snapshot / restore (issue #21) ─────────────────────────────────
    // Same Phase-1 design as HybridGdnForwardPass: one snapshot per forward-pass
    // instance, captured at end-of-decode by InferenceEngine. The GPU path
    // downloads device state into the host-side _gdnStateCache before writing
    // it into _snapshotBuf, then uploads back on restore.
    private byte* _snapshotBuf;
    private long _snapshotCap;
    private int _snapshotLength = -1;

    /// <inheritdoc />
    public int SnapshotLength => _snapshotLength;

    /// <inheritdoc />
    public void CaptureSnapshot()
    {
        EnsureSnapshotBuf();
        if (!_cpuGdn)
        {
            // GPU GDN path: device tensors hold the live state; the host-side
            // _gdnStateCache scan/conv buffers are stale. Download per-layer
            // before snapshotting. _gpu.Download self-syncs the stream so all
            // recurrence kernels for this token have committed before the copy.
            int scanFloats = _gdnStateCache.ScanStateFloatsPerLayer;
            int convFloats = _gdnStateCache.ConvStateFloatsPerLayer;
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                int g = _gdnStateCache.GdnLayerOf(layer);
                if (g < 0) continue;
                if (_gpuGdnScanState[layer] is { } scanT && scanFloats > 0)
                {
                    float* hostScan = _gdnStateCache.ScanStateAt(g);
                    _gpu.Download(scanT, new Span<float>(hostScan, scanFloats));
                }
                if (_gpuGdnConvState[layer] is { } convT && convFloats > 0)
                {
                    float* hostConv = _gdnStateCache.ConvStateAt(g);
                    _gpu.Download(convT, new Span<float>(hostConv, convFloats));
                }
            }
        }
        _gdnStateCache.SnapshotInto(_snapshotBuf, _snapshotCap);
        _snapshotLength = _gdnStateCache.Length;
    }

    /// <summary>Drop the currently held snapshot (if any).</summary>
    public void ClearSnapshot() => _snapshotLength = -1;

    private void EnsureSnapshotBuf()
    {
        long needed = _gdnStateCache.SnapshotBytes;
        if (_snapshotBuf != null && _snapshotCap >= needed)
            return;
        if (_snapshotBuf != null)
            NativeMemory.Free(_snapshotBuf);
        _snapshotBuf = (byte*)NativeMemory.Alloc((nuint)needed);
        _snapshotCap = needed;
    }

    /// <summary>Forward one token through the hybrid CUDA + CPU stack.</summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Embedding → _gpuHidden
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding!, _gpuHidden, token, _embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding!, _gpuHidden, token, _embDim);

        if (_traceLayers) TraceGpuTensor(position, -1, "emb", _gpuHidden, _embDim);

        // 2. Reserve KV cache page (layer-0 invariant; even if layer 0 is GDN).
        _kvCache.ReserveBlock();

        // 3. Trunk layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // ── Pre-block residual + norm on GPU ────────────────────
            _gpu.CopyDevice(_gpuResidual, _gpuHidden);
            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[layer], _hp.RmsNormEps);

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            if (isAttn)
            {
                GpuAttnBlock(layer, position);
            }
            else if (_cpuGdn)
            {
                CpuGdnBlock(layer, position);
            }
            else
            {
                GpuGdnBlock(layer, position);
            }

            // Residual add on GPU
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);

            if (_traceLayers) TraceGpuTensor(position, layer, isAttn ? "attn-resid" : "gdn-resid", _gpuHidden, _embDim);

            // ── Pre-MoE residual + norm on GPU ──────────────────────
            _gpu.CopyDevice(_gpuResidual, _gpuHidden);
            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuPostAttnNorm[layer], _hp.RmsNormEps);

            if (!_hp.IsMoE)
            {
                // Dense FFN (qwen35 27B-MTP): per-layer placement.
                // GPU slots populated by TryUploadDenseFfnLayers run entirely on GPU.
                // Layers without GPU slots fall back to the download/CPU/upload path.
                if (_gpuWFfnGate is not null && _gpuWFfnGate[layer] is not null)
                {
                    GpuDenseFfn(layer);
                }
                else
                {
                    _gpu.Download(_gpuNormBuf, new Span<float>(_cpuNormBuf, _embDim));
                    CpuDenseFfn(layer);
                    _gpu.UploadInto(_gpuHidden, new ReadOnlySpan<float>(_cpuMoeHidden, _embDim));
                }
            }
            else if (_cpuMoe)
            {
                // Download _gpuNormBuf → _cpuNormBuf, run MoE on CPU, upload result.
                // Download already syncs the stream (CudaMemcpyAsync + StreamSynchronize),
                // so an explicit Synchronize before it would just stall the host twice.
                _gpu.Download(_gpuNormBuf, new Span<float>(_cpuNormBuf, _embDim));
                CpuMoeFfn(layer);
                _gpu.UploadInto(_gpuHidden, new ReadOnlySpan<float>(_cpuMoeHidden, _embDim));
            }
            else
            {
                GpuMoeFfn(layer);
            }

            // Residual add
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);

            if (_traceLayers) TraceGpuTensor(position, layer, "moe-resid", _gpuHidden, _embDim);
        }

        // 4. Advance position counters
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();

        // 5. Capture pre-output-norm hidden for MTP (issue #29). _gpu.RmsNorm
        //    below overwrites _gpuHidden in place; snapshot to _gpuLastHidden
        //    first, then download to host so IForwardPass.LastHidden serves a
        //    host span. Cost: embDim×4 B = 20 KiB / token over PCIe (negligible).
        if (_hasMtp)
        {
            _gpu.CopyDevice(_gpuLastHidden, _gpuHidden);
            _gpu.Download(_gpuLastHidden, new Span<float>(_lastHidden, _embDim));
        }

        // 6. Final norm + output projection on GPU
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);

        // 6. Download logits to host (Download self-syncs the stream).
        _gpu.Download(_gpuLogits, _logitsBuf);

        if (_traceLayers) TraceLogits(position, _logitsBuf);

        return _logitsBuf;
    }

    // =================================================================
    //  GPU attention block — GLU-gated Q, partial NEOX RoPE on first 64 dims
    // =================================================================

    private void GpuAttnBlock(int layer, int position)
    {
        int kvDim = _numKvHeads * _headDim;

        // 1. Project: attn_q → [Q‖G] interleaved per head (output qDim*2 = 8192).
        GpuMatMul(_gpuQGate, _gpuWQGate[layer], _gpuNormBuf);
        GpuMatMul(_gpuK, _gpuWK[layer], _gpuNormBuf);
        GpuMatMul(_gpuV, _gpuWV[layer], _gpuNormBuf);

        // 2a. GPU strided de-interleave of [Q‖G] → _gpuQ, _gpuGate.
        _gpu.SplitQG(_gpuQ, _gpuGate, _gpuQGate, _numHeads, _headDim);

        // 2b. Per-head RMSNorm on Q and K (qwen35moe attn_q_norm / attn_k_norm).
        _gpu.HeadNorm(_gpuQ, _gpuQNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
        _gpu.HeadNorm(_gpuK, _gpuKNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);

        // 2c. Partial NEOX RoPE on Q and K (rotate first ropeDim of each head).
        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);

        // 3. Append K/V to GPU cache (position-indexed flat layout).
        _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[layer]!, _gpuVCache[layer]!,
            kvDim, position, _maxSeqLen);

        // 4. Scaled dot-product attention.
        _gpu.Attention(_gpuQ, _gpuKCache[layer]!, _gpuVCache[layer]!, _gpuAttnOut,
            _gpuAttnScratch,
            _numHeads, _numKvHeads, _headDim,
            (position + 1), _maxSeqLen);

        // 5. Apply GLU gate: attn_out *= sigmoid(gate). Single fused kernel.
        _gpu.SigmoidMulInPlace(_gpuAttnOut, _gpuGate);

        // 6. Output projection.
        GpuMatMul(_gpuHidden, _gpuWO[layer], _gpuAttnOut);
    }

    // =================================================================
    //  MTP / NEXTN head on GPU (issue #29)
    //  Mirror of HybridGdnForwardPass.MtpForward but with GPU-resident
    //  weights + MTP KV cache + shared scratch tensors.
    // =================================================================

    /// <inheritdoc />
    public bool HasMtpHead => _hasMtp;

    /// <inheritdoc />
    public ReadOnlySpan<float> LastHidden =>
        _hasMtp ? new ReadOnlySpan<float>(_lastHidden, _embDim) : default;

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
    {
        if (!_hasMtp)
            throw new InvalidOperationException(
                "MtpForward called on a CudaHybridGdnForwardPass that did not load an MTP head. " +
                "Check HasMtpHead before calling.");
        if (prevHidden.Length != _embDim)
            throw new ArgumentException(
                $"prevHidden length {prevHidden.Length} != EmbeddingDim {_embDim}.", nameof(prevHidden));

        // 1. Upload prevHidden into _gpuLastHidden. The end-of-Forward snapshot
        //    already populates _gpuLastHidden with the correct bits when the
        //    caller is MtpDecoder (which copies LastHidden → _savedHidden);
        //    re-upload for safety in case the caller passes a different buffer.
        _gpu.UploadInto(_gpuLastHidden, prevHidden);

        // 2. Embed token → _gpuMtpEmbedBuf.
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding!, _gpuMtpEmbedBuf, token, _embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding!, _gpuMtpEmbedBuf, token, _embDim);

        // 3. enorm(embedding) → _gpuMtpEnormBuf; hnorm(prevHidden) → _gpuMtpHnormBuf.
        _gpu.RmsNorm(_gpuMtpEnormBuf, _gpuMtpEmbedBuf,  _gpuMtpEnorm, _hp.RmsNormEps);
        _gpu.RmsNorm(_gpuMtpHnormBuf, _gpuLastHidden,   _gpuMtpHnorm, _hp.RmsNormEps);

        // 4. Concat [hnorm(h) ‖ enorm(e)] into _gpuMtpConcatBuf [embDim*2].
        //    Two device-side copies; total 40 KiB GPU memcpy for 27B.
        long embBytes = (long)_embDim * sizeof(float);
        _gpu.CopyDeviceRegion(_gpuMtpConcatBuf, dstByteOffset: 0,
                              _gpuMtpHnormBuf, srcByteOffset: 0, embBytes);
        _gpu.CopyDeviceRegion(_gpuMtpConcatBuf, dstByteOffset: embBytes,
                              _gpuMtpEnormBuf, srcByteOffset: 0, embBytes);

        // 5. eh_proj @ concat → _gpuHidden. F32 (UploadWeight dequant'd Q8_0 → F32).
        GpuMatMul(_gpuHidden, _gpuMtpEhProj, _gpuMtpConcatBuf);

        // 6. Residual + attn_norm.
        _gpu.CopyDevice(_gpuResidual, _gpuHidden);
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuMtpAttnNorm, _hp.RmsNormEps);

        // 7. MTP attention block (reuses _gpuQGate / _gpuQ / _gpuGate / _gpuK /
        //    _gpuV / _gpuAttnOut / _gpuAttnScratch scratch; writes _gpuHidden).
        GpuMtpAttnBlock(position);

        // 8. Residual add.
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);

        // 9. Residual + post_attention_norm.
        _gpu.CopyDevice(_gpuResidual, _gpuHidden);
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuMtpPostAttnNorm, _hp.RmsNormEps);

        // 10. Dense FFN on GPU. For qwen35 27B-MTP, _intermDim = 17408. Use the
        //     dense FFN scratch tensors that were allocated by TryUploadDenseFfnLayers
        //     when at least one trunk FFN layer ran on GPU. If those weren't
        //     allocated (the no-trunk-FFN-on-GPU case), allocate one-off scratch.
        var gateBuf = _gpuFfnGateBufDense
            ?? throw new InvalidOperationException(
                "MTP dense FFN on GPU requires _gpuFfnGateBufDense; the dense-FFN-on-GPU " +
                "path didn't initialise it. Re-check TryUploadDenseFfnLayers for the " +
                "27B-MTP model — at least one trunk FFN layer must land on GPU for the " +
                "scratch buffers to exist, OR allocate dedicated MTP FFN scratch.");
        var upBuf = _gpuFfnUpBufDense!;

        GpuMatMul(gateBuf,    _gpuMtpFfnGate, _gpuNormBuf);
        GpuMatMul(upBuf,      _gpuMtpFfnUp,   _gpuNormBuf);
        _gpu.SiLuMul(gateBuf, upBuf);
        GpuMatMul(_gpuHidden, _gpuMtpFfnDown, gateBuf);

        // 11. Residual add.
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);

        // 12. shared_head_norm (NOT main output_norm) → output.weight (shared lm_head).
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuMtpSharedHeadNorm, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);

        // 13. Download logits to host (Download self-syncs the stream).
        _gpu.Download(_gpuLogits, _logitsBuf);
        return _logitsBuf;
    }

    /// <summary>
    /// MTP attention block on GPU. Mirrors <see cref="GpuAttnBlock"/> but uses the
    /// MTP head's per-head norm, projection weights, and its own KV cache. Writes
    /// the post-attention residual contribution into <c>_gpuHidden</c>.
    /// </summary>
    private void GpuMtpAttnBlock(int position)
    {
        int kvDim = _numKvHeads * _headDim;
        var mtpCache = _mtpKvCache!;
        var kCache = _gpuMtpKCache!;
        var vCache = _gpuMtpVCache!;

        // 1. Project Q‖gate, K, V on GPU.
        GpuMatMul(_gpuQGate, _gpuMtpWQGate, _gpuNormBuf);
        GpuMatMul(_gpuK,     _gpuMtpWK,     _gpuNormBuf);
        GpuMatMul(_gpuV,     _gpuMtpWV,     _gpuNormBuf);

        // 2a. De-interleave Q‖gate per head → _gpuQ + _gpuGate.
        _gpu.SplitQG(_gpuQ, _gpuGate, _gpuQGate, _numHeads, _headDim);

        // 2b. Per-head RMSNorm on Q and K.
        _gpu.HeadNorm(_gpuQ, _gpuMtpQNorm, _numHeads,   _headDim, _hp.RmsNormEps);
        _gpu.HeadNorm(_gpuK, _gpuMtpKNorm, _numKvHeads, _headDim, _hp.RmsNormEps);

        // 2c. Partial NEOX RoPE.
        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);

        // 3. Layer-0 invariant: reserve a block on the MTP KV bookkeeping cache
        //    before any append at a new page boundary.
        mtpCache.ReserveBlock();
        _gpu.KvAppend(_gpuK, _gpuV, kCache, vCache, kvDim, position, _maxSeqLen);

        // 4. Scaled dot-product attention against the MTP cache.
        _gpu.Attention(_gpuQ, kCache, vCache, _gpuAttnOut, _gpuAttnScratch,
            _numHeads, _numKvHeads, _headDim, position + 1, _maxSeqLen);

        // 5. GLU gate.
        _gpu.SigmoidMulInPlace(_gpuAttnOut, _gpuGate);

        // 6. Output projection.
        GpuMatMul(_gpuHidden, _gpuMtpWO, _gpuAttnOut);

        mtpCache.IncrementPosition();
    }

    /// <inheritdoc />
    public void MtpResetCache()
    {
        if (!_hasMtp) return;
        _mtpKvCache?.Reset();
        if (_gpuMtpKCache is { } kT) _gpu.Clear(kT);
        if (_gpuMtpVCache is { } vT) _gpu.Clear(vT);
    }

    /// <inheritdoc />
    public void MtpTruncateTo(int length)
    {
        if (!_hasMtp) return;
        if (length == 0) { MtpResetCache(); return; }
        // Soft truncate — PagedKvCache.TruncateTo handles arbitrary lengths up
        // to current Length. Pages above the new length are reused on next
        // write; the device-side _gpuMtpKCache is a flat ring, so writes simply
        // overwrite stale slots on subsequent KvAppend calls.
        _mtpKvCache?.TruncateTo(length);
    }

    // =================================================================
    //  CPU GDN block — mirror of HybridGdnForwardPass.GdnBlock
    // =================================================================

    private void CpuGdnBlock(int layer, int position)
    {
        // Download _gpuNormBuf → _cpuNormBuf so the CPU GDN kernels can consume it
        // (Download self-syncs the stream).
        _gpu.Download(_gpuNormBuf, new Span<float>(_cpuNormBuf, _embDim));

        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        float* scanState = _gdnStateCache.ScanStateAt(gdnIdx);
        float* convState = _gdnStateCache.ConvStateAt(gdnIdx);
        int convStateLen = _gdnStateCache.ConvStateFloatsPerLayer;
        int scanStateLen = _gdnStateCache.ScanStateFloatsPerLayer;

        // 1. Joint QKV projection and z (gate) projection — CPU via SimdKernels.
        SimdKernels.MatVec(_qkv, _cpuWQkv[layer].DataPtr, _cpuNormBuf,
            _gdnConvChannels, _embDim, _cpuWQkv[layer].DType);
        SimdKernels.MatVec(_zVec, _cpuWZGate[layer].DataPtr, _cpuNormBuf,
            _gdnValueDim, _embDim, _cpuWZGate[layer].DType);

        // 2. Depthwise causal conv1d.
        GdnKernels.CausalDepthwiseConv1dDecode(
            new ReadOnlySpan<float>(_qkv, _gdnConvChannels),
            new Span<float>(convState, convStateLen),
            new ReadOnlySpan<float>(_ssmConv1d[layer], _gdnConvKernel * _gdnConvChannels),
            new Span<float>(_qkvConv, _gdnConvChannels),
            _gdnConvChannels, _gdnConvKernel);

        // 3. SiLU on conv output.
        GdnKernels.SiLu(
            new Span<float>(_qkvConv, _gdnConvChannels),
            new ReadOnlySpan<float>(_qkvConv, _gdnConvChannels));

        // 4. Split Q‖K‖V.
        var qPre = new Span<float>(_qkvConv, _gdnKeyDim);
        var kPre = new Span<float>(_qkvConv + _gdnKeyDim, _gdnKeyDim);
        var vV = new ReadOnlySpan<float>(_qkvConv + 2 * _gdnKeyDim, _gdnValueDim);

        // 5. Per-K-head L2 norm.
        GdnKernels.L2NormPerHead(qPre, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        GdnKernels.L2NormPerHead(kPre, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);

        // 6. Tile K→V head count.
        GdnKernels.TileHeads(qPre, new Span<float>(_qVHeads, _gdnNumVHeads * _gdnHeadDim),
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);
        GdnKernels.TileHeads(kPre, new Span<float>(_kVHeads, _gdnNumVHeads * _gdnHeadDim),
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);

        // 7. Alpha / Beta per-v-head projections.
        SimdKernels.MatVec(_alpha, _cpuSsmAlpha[layer].DataPtr, _cpuNormBuf,
            _gdnNumVHeads, _embDim, _cpuSsmAlpha[layer].DType);
        SimdKernels.MatVec(_beta, _cpuSsmBeta[layer].DataPtr, _cpuNormBuf,
            _gdnNumVHeads, _embDim, _cpuSsmBeta[layer].DType);

        // 8. Recurrence: rank-1 state update + per-head RMSNorm + SiLU(z) gate.
        GdnKernels.GdnRecurrenceDecode(
            q: new ReadOnlySpan<float>(_qVHeads, _gdnNumVHeads * _gdnHeadDim),
            k: new ReadOnlySpan<float>(_kVHeads, _gdnNumVHeads * _gdnHeadDim),
            v: vV,
            alphaIn: new ReadOnlySpan<float>(_alpha, _gdnNumVHeads),
            beta: new ReadOnlySpan<float>(_beta, _gdnNumVHeads),
            ssmA: new ReadOnlySpan<float>(_ssmA[layer], _gdnNumVHeads),
            dtBias: new ReadOnlySpan<float>(_ssmDtBias[layer], _gdnNumVHeads),
            normWeight: new ReadOnlySpan<float>(_ssmNormW[layer], _gdnHeadDim),
            z: new ReadOnlySpan<float>(_zVec, _gdnValueDim),
            state: new Span<float>(scanState, scanStateLen),
            output: new Span<float>(_gdnOut, _gdnValueDim),
            numVHeads: _gdnNumVHeads,
            headDim: _gdnHeadDim,
            normEps: 1e-6f);

        // 9. Output projection: ssm_out (input ValueDim, output embDim) → _cpuHiddenOut.
        SimdKernels.MatVec(_cpuHiddenOut, _cpuSsmOut[layer].DataPtr, _gdnOut,
            _embDim, _gdnValueDim, _cpuSsmOut[layer].DType);

        // Upload back to GPU.
        _gpu.UploadInto(_gpuHidden, new ReadOnlySpan<float>(_cpuHiddenOut, _embDim));
    }

    // =================================================================
    //  GPU GDN block — full-GPU mirror of CpuGdnBlock.
    //  Consumes _gpuNormBuf, writes the block output into _gpuHidden.
    //  No CPU↔GPU sync inside the block.
    // =================================================================

    private void GpuGdnBlock(int layer, int position)
    {
        var scanState = _gpuGdnScanState[layer]!;
        var convState = _gpuGdnConvState[layer]!;

        // 1. Joint QKV projection and z (gate) projection.
        GpuMatMul(_gpuGdnQkv, _gpuWAttnQkv[layer], _gpuNormBuf);
        GpuMatMul(_gpuGdnZVec, _gpuWAttnGate[layer], _gpuNormBuf);

        // 2. Depthwise causal conv1d (updates convState in place).
        _gpu.GdnConv1dDecode(_gpuGdnQkv, convState, _gpuSsmConv1d[layer], _gpuGdnQkvConv,
            _gdnConvChannels, _gdnConvKernel);

        // 3. SiLU on the conv output (whole 8192).
        _gpu.SiLUInPlace(_gpuGdnQkvConv);

        // 4. L2-norm per K-head on the Q and K slices (each [k_heads=16, head_dim=128]).
        //    Layout of _gpuGdnQkvConv:
        //      [0 .. key_dim)            → Q (k_heads × head_dim)
        //      [key_dim .. 2*key_dim)    → K
        //      [2*key_dim .. conv_chan)  → V (v_heads × head_dim)
        _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, 0,
            _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, _gdnKeyDim,
            _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);

        // 5. Tile K-heads → V-head count (Hk=16, Hv=32, repeat=2).
        _gpu.GdnTileHeads(_gpuGdnQkvConv, 0, _gpuGdnQHead, 0,
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);
        _gpu.GdnTileHeads(_gpuGdnQkvConv, _gdnKeyDim, _gpuGdnKHead, 0,
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);

        // 6. Alpha / Beta per-v-head projections.
        GpuMatMul(_gpuGdnAlpha, _gpuWSsmAlpha[layer], _gpuNormBuf);
        GpuMatMul(_gpuGdnBeta,  _gpuWSsmBeta[layer],  _gpuNormBuf);

        // 7. Copy the V slice (final 4096 floats of _gpuGdnQkvConv) into _gpuGdnVHead.
        //    This is required because the recurrence wrapper takes whole Tensors;
        //    16 KiB device copy per layer is negligible (~30 µs total over 30 layers).
        _gpu.CopyDeviceRegion(_gpuGdnVHead, 0,
            _gpuGdnQkvConv, 2L * _gdnKeyDim * sizeof(float),
            (long)_gdnValueDim * sizeof(float));

        // 8. Recurrence: rank-1 state update + per-head RMSNorm + SiLU(z) gate (GPU).
        _gpu.GdnRecurrenceDecode(
            scanState, _gpuGdnQHead, _gpuGdnKHead, _gpuGdnVHead,
            _gpuGdnAlpha, _gpuGdnBeta,
            _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
            _gpuGdnZVec, _gpuGdnOut,
            _gdnNumVHeads, _gdnHeadDim, normEps: 1e-6f);

        // 9. Output projection: ssm_out (input value_dim=4096, output emb_dim=2048).
        GpuMatMul(_gpuHidden, _gpuWSsmOut[layer], _gpuGdnOut);
    }

    // =================================================================
    //  MoE FFN on GPU (router + SLRU experts + shared expert)
    // =================================================================

    private void GpuMoeFfn(int layer)
    {
        // 1. Router on GPU.
        GpuMatMul(_gpuRouterLogits, _gpuWGateInp[layer], _gpuNormBuf);
        _gpu.Softmax(_gpuRouterLogits);

        // 2. Download to host and pick top-K (256 → 8).
        //    Per-layer cost: 1 KB readback — Download self-syncs the stream.
        _gpu.Download(_gpuRouterLogits, _routerBuf);

        Span<int> selectedExperts = stackalloc int[_numActiveExperts];
        Span<float> expertWeights = stackalloc float[_numActiveExperts];
        SelectTopK(_routerBuf, _numActiveExperts, selectedExperts, expertWeights);

        // 3. Shared expert: ffn_down @ (SiLU(gate @ x) * (up @ x))
        //    then per-token sigmoid-scalar gate via ffn_gate_inp_shexp · x.
        GpuMatMul(_gpuFfnGate, _gpuWGateShexp[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp, _gpuWUpShexp[layer], _gpuNormBuf);
        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        GpuMatMul(_gpuSharedOut, _gpuWDownShexp[layer], _gpuFfnGate);

        // 3a. Compute the shared-expert scalar gate on CPU.
        //     ffn_gate_inp_shexp is a [embDim] vector → scalar via dot product.
        //
        // TODO(Phase6c): add a dot-product + sigmoid + scale-in-place kernel.
        //   For v1: download _gpuNormBuf and the small weight, compute scalar, scale on GPU.
        //   Note: _gpuNormBuf is already populated; just need the readback for the dot
        //   (Download self-syncs the stream).
        _gpu.Download(_gpuNormBuf, new Span<float>(_cpuNormReadback, _embDim));
        _gpu.Download(_gpuWGateInpShexp[layer], new Span<float>(_hostQ, _embDim)); // reuse _hostQ
        // dot product on CPU.
        float dot = SimdKernels.DotF32(_hostQ, _cpuNormReadback, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-dot));
        _gpu.ScaleInPlace(_gpuSharedOut, shexpScale);

        // 4. Routed experts via SLRU.
        _gpu.Clear(_gpuHidden);
        for (int k = 0; k < _numActiveExperts; k++)
        {
            int expertIdx = selectedExperts[k];
            float expertWeight = expertWeights[k];
            var slot = _expertSlotManager!.GetOrLoad(layer, expertIdx);

            GpuMatMul(_gpuFfnGate, slot.Gate, _gpuNormBuf);
            GpuMatMul(_gpuFfnUp, slot.Up, _gpuNormBuf);
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuExpertOut, slot.Down, _gpuFfnGate);
            _gpu.AddScaledInPlace(_gpuHidden, _gpuExpertOut, expertWeight);
        }

        // 5. Add shared expert.
        _gpu.AddInPlace(_gpuHidden, _gpuSharedOut);
    }

    // =================================================================
    //  CPU dense FFN (qwen35 27B-MTP) — mirror of HybridGdnForwardPass.DenseFfn.
    //  Consumes _cpuNormBuf, produces _cpuMoeHidden. Weights stay mmap'd; per-token
    //  read traffic is ~8.6 GB for 27B at Q4_K_M, capping decode at DRAM bandwidth.
    // =================================================================

    private void CpuDenseFfn(int layer)
    {
        var wGate = _cpuWFfnGate![layer];
        var wUp   = _cpuWFfnUp![layer];
        var wDown = _cpuWFfnDown![layer];

        SimdKernels.MatVecDual(
            _cpuFfnGateBuf, wGate.DataPtr,
            _cpuFfnUpBuf,   wUp.DataPtr,
            _cpuNormBuf, _intermDim, _embDim,
            wGate.DType, wUp.DType);
        SimdKernels.SiLuMul(_cpuFfnGateBuf, _cpuFfnUpBuf, _intermDim);
        SimdKernels.MatVec(_cpuMoeHidden, wDown.DataPtr, _cpuFfnGateBuf,
            _embDim, _intermDim, wDown.DType);
    }

    // =================================================================
    //  GPU dense FFN — for layers whose ffn_gate/up/down were uploaded by
    //  TryUploadDenseFfnLayers. Consumes _gpuNormBuf, produces _gpuHidden.
    // =================================================================

    private void GpuDenseFfn(int layer)
    {
        var wGate = _gpuWFfnGate![layer]!;
        var wUp   = _gpuWFfnUp![layer]!;
        var wDown = _gpuWFfnDown![layer]!;
        var gateBuf = _gpuFfnGateBufDense!;
        var upBuf   = _gpuFfnUpBufDense!;

        GpuMatMul(gateBuf, wGate, _gpuNormBuf);
        GpuMatMul(upBuf,   wUp,   _gpuNormBuf);
        _gpu.SiLuMul(gateBuf, upBuf);
        GpuMatMul(_gpuHidden, wDown, gateBuf);
    }

    // =================================================================
    //  TryUploadDenseFfnLayers — opportunistically upload as many dense FFN
    //  layers' ffn_gate/up/down to GPU as fit in remaining VRAM. Reserves
    //  a safety margin for KV cache growth and scratch.
    //
    //  Called from the constructor for the dense-FFN (qwen35 27B-MTP) path.
    //  Layers not uploaded fall back to CpuDenseFfn (mmap reads per token).
    // =================================================================

    private void TryUploadDenseFfnLayers(CudaBackend gpu, ModelHyperparams hp, int L)
    {
        // Probe per-layer FFN cost from layer 0.
        var gateInfo = _model.FindTensor("blk.0.ffn_gate.weight");
        var upInfo   = _model.FindTensor("blk.0.ffn_up.weight");
        var downInfo = _model.FindTensor("blk.0.ffn_down.weight");
        if (gateInfo is null || upInfo is null || downInfo is null)
            return;

        long perLayerBytes = gateInfo.Value.ByteSize + upInfo.Value.ByteSize + downInfo.Value.ByteSize;
        // 64 MiB safety margin — empirically the minimum that survives subsequent
        // runtime growth (cuBLAS workspace, transient kernel scratch). KV cache and
        // GDN state are fully pre-allocated at construction so no runtime growth there.
        // Allocator overhead per upload is ~50 MiB (alignment/pool), already accounted
        // for by the per-iteration FreeVramBytes re-check inside the upload loop.
        // Override via SHARPI_DENSE_FFN_GPU_MARGIN_MB env var (set 0 to push to the wall).
        long safetyMarginBytes = 64L * 1024 * 1024;
        var marginOverride = Environment.GetEnvironmentVariable("SHARPI_DENSE_FFN_GPU_MARGIN_MB");
        if (marginOverride is not null && int.TryParse(marginOverride, out int marginMb) && marginMb >= 0)
            safetyMarginBytes = (long)marginMb * 1024 * 1024;

        ulong freeNow = gpu.FreeVramBytes;
        long budget = (long)freeNow - safetyMarginBytes;
        if (budget < perLayerBytes)
        {
            Console.Error.WriteLine(
                $"[CudaHybridGdnForwardPass] Dense FFN-on-GPU: free VRAM {freeNow / (1024 * 1024)} MiB < safety {safetyMarginBytes / (1024 * 1024)} MiB + per-layer {perLayerBytes / (1024 * 1024)} MiB. All FFN stays on CPU.");
            return;
        }
        int canUpload = (int)Math.Min(L, budget / perLayerBytes);

        _gpuWFfnGate = new Tensor?[L];
        _gpuWFfnUp   = new Tensor?[L];
        _gpuWFfnDown = new Tensor?[L];

        // Allocate GPU FFN scratch (intermDim floats × 2 = 17408 × 8 ≈ 140 KB).
        _gpuFfnGateBufDense = gpu.Allocate(TensorShape.D1(_intermDim));
        _gpuFfnUpBufDense   = gpu.Allocate(TensorShape.D1(_intermDim));

        int uploaded = 0;
        for (int i = 0; i < L; i++)
        {
            if (uploaded >= canUpload) break;
            // Re-check free VRAM each iteration: per-layer estimate may be off due to
            // alignment / pool overhead, so bail early if we're about to violate margin.
            ulong freeIter = gpu.FreeVramBytes;
            if ((long)freeIter - safetyMarginBytes < perLayerBytes) break;
            try
            {
                _gpuWFfnGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _gpuWFfnUp[i]   = UploadWeight($"blk.{i}.ffn_up.weight");
                _gpuWFfnDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
                uploaded++;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[CudaHybridGdnForwardPass] FFN-on-GPU upload aborted at layer {i}: {ex.Message}");
                // Roll back partial layer uploads to keep slot state consistent.
                if (_gpuWFfnGate[i] is { } gT) { gpu.Free(gT); _gpuWFfnGate[i] = null; }
                if (_gpuWFfnUp[i]   is { } uT) { gpu.Free(uT); _gpuWFfnUp[i]   = null; }
                break;
            }
        }
        _denseFfnGpuLayers = uploaded;
        Console.Error.WriteLine(
            $"[CudaHybridGdnForwardPass] Dense FFN-on-GPU: uploaded {uploaded}/{L} layers ({uploaded * perLayerBytes / (1024 * 1024)} MiB); {L - uploaded} stay on CPU. Free VRAM after: {gpu.FreeVramBytes / (1024 * 1024)} MiB.");
    }

    // =================================================================
    //  CPU MoE FFN (SHARPI_CPU_MOE=1) — mirror of HybridGdnForwardPass.MoeFfn
    //  Consumes _cpuNormBuf, produces _cpuMoeHidden.
    // =================================================================

    private void CpuMoeFfn(int layer)
    {
        int numExperts = _numExperts;
        int numActive = _numActiveExperts;
        int expertDim = _expertDim;

        // 1. Kick off the GPU shared expert (async; overlaps with CPU work below).
        //    _gpuNormBuf is already populated by the RmsNorm before this call;
        //    the launches return immediately and execute while the CPU runs router
        //    and routed experts. Sigmoid scalar gate is computed on CPU and applied
        //    via ScaleInPlace before the host blocks on Download.
        GpuMatMul(_gpuFfnGate, _gpuWGateShexp[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp, _gpuWUpShexp[layer], _gpuNormBuf);
        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        GpuMatMul(_gpuSharedOut, _gpuWDownShexp[layer], _gpuFfnGate);

        float shexpDot = SimdKernels.DotF32(_cpuFfnGateInpShexp![layer], _cpuNormBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        _gpu.ScaleInPlace(_gpuSharedOut, shexpScale);

        // 2. Router: ffn_gate_inp.weight is F32 [embDim, numExperts]; softmax then top-K.
        var routerW = _cpuFfnGateInp![layer];
        SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, _cpuNormBuf,
            numExperts, _embDim, routerW.DType);
        SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopKPtr(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights);

        // 3. Routed experts (sparse top-K). Two batched Parallel.For sweeps
        //    instead of 16 per-expert ones — gate+up across all 8 experts in
        //    one sweep, then down+weighted-accumulate across all 8 experts in
        //    another. Each worker thread does much more work per dispatch,
        //    amortising TPL barrier overhead.
        var gateExps = _cpuFfnGateExps![layer];
        var upExps = _cpuFfnUpExps![layer];
        var downExps = _cpuFfnDownExps![layer];

        int bprG = (_embDim    / DTypeInfo.BlockSize(gateExps.DType))
                 * DTypeInfo.BytesPerBlock(gateExps.DType);
        int bprU = (_embDim    / DTypeInfo.BlockSize(upExps.DType))
                 * DTypeInfo.BytesPerBlock(upExps.DType);
        int bprD = (expertDim  / DTypeInfo.BlockSize(downExps.DType))
                 * DTypeInfo.BytesPerBlock(downExps.DType);

        // Stackalloc small per-token arrays into native pointers so worker
        // threads can read them safely (Parallel.For is synchronous, so the
        // stack frame stays alive until all workers complete).
        int* sePtr = stackalloc int[numActive];
        float* ewPtr = stackalloc float[numActive];
        for (int i = 0; i < numActive; i++)
        {
            sePtr[i] = selectedExperts[i];
            ewPtr[i] = expertWeights[i];
        }

        // Local copies for lambda capture.
        byte*  gateP    = gateExps.DataPtr;
        byte*  upP      = upExps.DataPtr;
        byte*  downP    = downExps.DataPtr;
        DType  gateDt   = gateExps.DType;
        DType  upDt     = upExps.DType;
        DType  downDt   = downExps.DType;
        float* gateAll  = _cpuExpertGateAll;
        float* upAll    = _cpuExpertUpAll;
        float* normBuf  = _cpuNormBuf;
        float* moeOut   = _cpuMoeHidden;
        int    embDimL  = _embDim;
        int    expertDimL = expertDim;
        int    numActiveL = numActive;
        int    bprGL = bprG, bprUL = bprU, bprDL = bprD;

        // Phase A: gate + up rows for all (k, r) tuples.
        Parallel.For(0, numActiveL * expertDimL, s_moeParallelOpts, idx =>
        {
            int k = idx / expertDimL;
            int r = idx % expertDimL;
            int expertIdx = sePtr[k];
            long offG = (long)expertIdx * expertDimL * bprGL + (long)r * bprGL;
            long offU = (long)expertIdx * expertDimL * bprUL + (long)r * bprUL;
            gateAll[idx] = DispatchDot(gateP + offG, normBuf, embDimL, gateDt);
            upAll[idx]   = DispatchDot(upP   + offU, normBuf, embDimL, upDt);
        });

        // Phase B: one fused SiLuMul over (numActive × expertDim) contiguous
        // floats. SiLuMul is element-wise, so expert boundaries don't matter —
        // one AVX-vectorised call beats 8 with their own setup cost.
        SimdKernels.SiLuMul(_cpuExpertGateAll, _cpuExpertUpAll, numActive * expertDim);

        // Phase C: down × weight, fused across all 8 experts into one sweep over
        // embDim rows. Each thread owns its rows so there's no cross-expert race.
        // Note: dtype-specialised inner loops were tried but destabilised this
        // path (run-to-run jitter doubled). On the GPU-coordinated hybrid path
        // the extra closure variants seem to perturb the host↔stream launch
        // pacing — the per-iter DispatchDot switch is cheap relative to the
        // ~1.8 ms/layer GPU sync the host stays in lock-step with anyway.
        Parallel.For(0, embDimL, s_moeParallelOpts, r =>
        {
            float sum = 0f;
            for (int k = 0; k < numActiveL; k++)
            {
                int expertIdx = sePtr[k];
                float w = ewPtr[k];
                long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                sum += w * DispatchDot(downP + offD,
                                       gateAll + (long)k * expertDimL,
                                       expertDimL, downDt);
            }
            moeOut[r] = sum;
        });

        // 4. Wait for GPU shared expert, download, and combine into routed accumulator
        //    (Download self-syncs the stream).
        _gpu.Download(_gpuSharedOut, new Span<float>(_cpuSharedOut, _embDim));
        SimdKernels.AddInPlace(_cpuMoeHidden, _cpuSharedOut, _embDim);
    }

    // ParallelOptions for the routed-MoE Parallel.For sweeps. Matches the
    // CPU core count exactly so workers don't oversubscribe the CPU with
    // the (concurrently running) GPU shared-expert host launches.
    private static readonly ParallelOptions s_moeParallelOpts = new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DispatchDot(byte* row, float* input, int cols, DType dtype) =>
        dtype switch
        {
            DType.Q4_K    => SimdKernels.DotQ4K(row, input, cols),
            DType.Q5_K    => SimdKernels.DotQ5K(row, input, cols),
            DType.Q6_K    => SimdKernels.DotQ6K(row, input, cols),
            DType.Float32 => SimdKernels.DotF32((float*)row, input, cols),
            _ => throw new NotSupportedException($"Routed expert dtype {dtype} not supported in batched path"),
        };

    private static void SelectTopKPtr(float* logits, int n, int k,
        Span<int> indices, Span<float> weights)
    {
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
        if (k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // =================================================================
    //  Helpers — MatMul dispatch with dtype awareness
    // =================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GpuMatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        _gpu.MatMul(output, matrix, vector,
            _gpuWeightDTypes.TryGetValue(matrix.Handle, out var dt) ? dt : DType.Float32);
    }

    private static void SelectTopK(ReadOnlySpan<float> logits, int k,
        Span<int> indices, Span<float> weights)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
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
        if (k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0)
                for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // =================================================================
    //  Weight loading
    // =================================================================

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

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // exact=true: weights live for the entire decoding session. Pooling and the
        // associated power-of-2 round-up are pure VRAM waste — a 17 MiB attn_gate
        // rounds to 32 MiB; aggregated across 64 layers that's gigabytes of overhead.
        // Free()'s exact path returns the memory directly to the driver instead of
        // stranding it in a per-tensor pool bucket.
        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q5_K || info.DType == DType.Q6_K)
        {
            // CUDA matvec dispatches on Q4_K / Q5_K / Q6_K via dedicated kernels.
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Q8_0, Q3_K, etc. — CUDA matvec doesn't dispatch on these. Dequantize to F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private Tensor UploadEmbeddingWeight(string name, out bool isQuantized)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        if (info.DType == DType.Q4_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            // exact=true: embedding table is permanent for the session; skip the
            // power-of-2 round-up that would otherwise inflate a 715 MiB Q4_K embed
            // to a 1024 MiB GPU allocation.
            var result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Q4_K;
            isQuantized = true;
            return result;
        }
        int count = (int)info.ElementCount;
        var f32 = new float[count];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).CopyTo(f32);
        else
            Dequantize.ToFloat32(data, f32, info.DType, count);
        var t = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
        _gpuWeightDTypes[t.Handle] = DType.Float32;
        isQuantized = false;
        return t;
    }

    private float* LoadF32Tensor(string name, int expectedCount)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        int count = (int)info.ElementCount;
        if (count != expectedCount)
            throw new InvalidOperationException(
                $"Tensor {name}: expected {expectedCount} elements, got {count}.");
        var buf = Alloc(count);
        var data = _model.GetTensorData(info);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

    private float* LoadConv1dTransposed(string name, int kernel, int channels)
    {
        // GGUF stores conv1d as [channels, kernel] row-major; GdnKernels wants [kernel, channels].
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        int expected = kernel * channels;
        int count = (int)info.ElementCount;
        if (count != expected)
            throw new InvalidOperationException(
                $"Tensor {name}: expected {expected} elements ({kernel}*{channels}), got {count}.");
        var data = _model.GetTensorData(info);
        Span<float> src;
        float[]? tempArr = null;
        if (info.DType == DType.Float32)
            src = MemoryMarshal.Cast<byte, float>(data).Slice(0, count).ToArray();
        else
        {
            tempArr = new float[count];
            Dequantize.ToFloat32(data, tempArr, info.DType, count);
            src = tempArr;
        }
        var buf = Alloc(expected);
        for (int k = 0; k < kernel; k++)
            for (int c = 0; c < channels; c++)
                buf[k * channels + c] = src[c * kernel + k];
        return buf;
    }

    private Tensor UploadConv1dTransposedToGpu(string name, int kernel, int channels)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        int expected = kernel * channels;
        int count = (int)info.ElementCount;
        if (count != expected)
            throw new InvalidOperationException(
                $"Tensor {name}: expected {expected} elements ({kernel}*{channels}), got {count}.");
        var data = _model.GetTensorData(info);
        float[] src = new float[count];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(src);
        else
            Dequantize.ToFloat32(data, src, info.DType, count);

        var transposed = new float[expected];
        for (int k = 0; k < kernel; k++)
            for (int c = 0; c < channels; c++)
                transposed[k * channels + c] = src[c * kernel + k];
        var tensor = _gpu.Upload(transposed, TensorShape.D1(expected));
        _gpuWeightDTypes[tensor.Handle] = DType.Float32;
        return tensor;
    }

    private static bool ShouldKeepFixedWeightsOnCpu(GgufTensorInfo embedding, GgufTensorInfo? output)
    {
        const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;
        if (EstimateGpuEmbeddingBytes(embedding) > maxStorageBufferBytes)
            return true;
        if (output is not null && EstimateGpuTensorBytes(output.Value) > maxStorageBufferBytes)
            return true;
        return false;
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K
            || tensor.DType == DType.Q5_K || tensor.DType == DType.Q6_K)
            return (tensor.ByteSize + 3) & ~3L;
        return tensor.ElementCount * sizeof(float);
    }

    private static long EstimateGpuEmbeddingBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Q4_K)
            return (tensor.ByteSize + 3) & ~3L;
        return tensor.ElementCount * sizeof(float);
    }

    private long EstimateUploadedVram()
    {
        // Sum bytes of every GPU tensor we've allocated so far via the dtype map.
        // This is an overestimate (Q4_K stays raw bytes) but a fine input to slot sizing.
        long total = 0;
        foreach (var (_, _) in _gpuWeightDTypes)
        {
            // We don't track byte sizes per handle here; rely on the conservative
            // 2 GiB reservation in the slot capacity computation instead.
        }
        // Use a rough sum based on layer count and known tensor shapes.
        int L = _hp.NumLayers;
        // Per layer: norm (8KB) + post_attn_norm (8KB) + router (2KB×256=512KB F32) + shared
        // expert (~8.4 MiB at Q8_0→F32) + 3 ffn_*_shexp matrices + ffn_gate_inp_shexp (8KB).
        // Attention layers add ~50 MiB of Q8_0→F32 Q/K/V/O + KV cache (~ ctx × 2 × 512 × 4 B).
        long perLayer = (long)(_embDim * sizeof(float) * 2) // attn_norm + post_attn_norm
                     + (long)_numExperts * _embDim * sizeof(float) // ffn_gate_inp F32
                     + (long)_embDim * sizeof(float) // ffn_gate_inp_shexp F32
                     + (long)_embDim * _expertDim * sizeof(float) * 3; // shared expert (gate/up/down)
        total += L * perLayer;
        // Attention layers contribute Q/K/V/O (F32 expansion of Q8_0).
        int attnLayers = 0;
        for (int i = 0; i < L; i++)
            if (_hp.LayerTypes![i] == LayerType.Attention) attnLayers++;
        long attnPerLayer = (long)_embDim * _numHeads * _headDim * 2 * sizeof(float)  // q (output qDim*2)
                          + (long)_embDim * _numKvHeads * _headDim * sizeof(float) * 2 // k + v
                          + (long)_embDim * _numHeads * _headDim * sizeof(float)      // o
                          + (long)_maxSeqLen * _numKvHeads * _headDim * sizeof(float) * 2; // kv cache
        total += attnLayers * attnPerLayer;
        // Embedding + output.
        if (_gpuEmbedding is not null)
            total += (long)_hp.VocabSize * _embDim * sizeof(float);
        return total;
    }

    /// <summary>
    /// Predicts how many SLRU expert slots will fit in VRAM after the non-MoE
    /// weights (attention, GDN, norms, embeddings) are uploaded. Uses the same
    /// arithmetic the SLRU itself runs after the per-layer upload loop, just
    /// hoisted so the auto-MoE-routing decision can be made up front.
    /// </summary>
    private int PredictSlruSlots(int numLayers)
    {
        long perLayerNonMoeBytes = 0;
        // norms
        perLayerNonMoeBytes += 2L * _embDim * sizeof(float);
        // shared-expert weights stay on GPU regardless of routing
        perLayerNonMoeBytes += (long)_numExperts * _embDim * sizeof(float);   // router
        perLayerNonMoeBytes += (long)_embDim * sizeof(float);                  // shexp gate inp
        perLayerNonMoeBytes += 3L * _embDim * _expertDim * sizeof(float);      // shared gate/up/down (Q8_0 → F32 in current path)

        long attnPerLayer =
              (long)_embDim * _numHeads * _headDim * 2 * sizeof(float)         // q (output qDim*2)
            + (long)_embDim * _numKvHeads * _headDim * sizeof(float) * 2       // k + v
            + (long)_embDim * _numHeads * _headDim * sizeof(float)             // o
            + (long)_maxSeqLen * _numKvHeads * _headDim * sizeof(float) * 2;   // kv cache

        long gdnPerLayer = 0;
        if (!_cpuGdn)
        {
            // raw Q4_K bytes (CUDA matvec keeps these quantized)
            gdnPerLayer += (long)_gdnConvChannels * _embDim / 256 * 144;  // attn_qkv Q4_K
            gdnPerLayer += (long)_gdnValueDim * _embDim / 256 * 144;      // attn_gate Q4_K
            gdnPerLayer += (long)_embDim * _gdnValueDim / 256 * 144;      // ssm_out Q4_K
            gdnPerLayer += (long)_gdnNumVHeads * _embDim * sizeof(float); // ssm_alpha F32
            gdnPerLayer += (long)_gdnNumVHeads * _embDim * sizeof(float); // ssm_beta F32
            gdnPerLayer += (long)_gdnConvKernel * _gdnConvChannels * sizeof(float);  // conv1d
            gdnPerLayer += (long)_gdnNumVHeads * _gdnHeadDim * _gdnHeadDim * sizeof(float); // scan state
            gdnPerLayer += (long)(_gdnConvKernel - 1) * _gdnConvChannels * sizeof(float);   // conv state
        }

        int attnLayers = 0;
        for (int i = 0; i < numLayers; i++)
            if (_hp.LayerTypes![i] == LayerType.Attention) attnLayers++;
        int gdnLayers = numLayers - attnLayers;

        long total = numLayers * perLayerNonMoeBytes
                   + (long)attnLayers * attnPerLayer
                   + (long)gdnLayers * gdnPerLayer
                   + (long)_hp.VocabSize * _embDim * sizeof(float);    // embedding/output

        long vramTotal = (long)_gpu.VramBytes;
        long remaining = vramTotal - total - (2L << 30);
        long perExpert = EstimatePerExpertBytes();
        if (perExpert <= 0) return 1024;
        return (int)Math.Max(64, remaining / perExpert);
    }

    private long EstimatePerExpertBytes()
    {
        // Per expert: gate + up + down weight matrices. For qwen35moe Q4_K_M:
        //   gate/up: [embDim=2048, expertDim=512] each ≈ 588 KiB raw Q4_K
        //   down:    [expertDim=512, embDim=2048]     ≈ 588 KiB raw Q5_K — with the
        //   CUDA Q5_K matvec landed (Phase 7b) this matrix now stays as raw bytes
        //   instead of expanding 4× to F32, halving the per-expert footprint.
        // Sum: ~1.81 MiB raw. Fall back to F32 expansion only for dtypes the
        // CUDA matvec still can't handle (Q8_0, Q3_K, etc.).
        long bytes = 0;
        foreach (var name in new[] { "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight" })
        {
            var info = _model.FindTensor(name);
            if (info is null) continue;
            // bytesPerExpert in the packed tensor — total bytes / numExperts.
            long perExpert = info.Value.ByteSize / Math.Max(1, _numExperts);
            // If the dtype is something CUDA matvec can't handle, the SLRU will
            // dequantize to F32, increasing footprint. Compute conservative byte
            // size by assuming F32 when not Q4_K/Q5_K/Q6_K.
            if (info.Value.DType != DType.Q4_K && info.Value.DType != DType.Q5_K
                && info.Value.DType != DType.Q6_K && info.Value.DType != DType.Float32)
            {
                int rows = (int)(info.Value.ElementCount / _numExperts);
                perExpert = (long)rows * sizeof(float);
            }
            bytes += perExpert;
        }
        return bytes > 0 ? bytes : (long)(1.81 * 1024 * 1024);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count * (nuint)sizeof(float));

    // =================================================================
    //  Tracing (SHARPI_TRACE_LAYERS=1)
    // =================================================================

    private void TraceGpuTensor(int position, int layer, string tag, Tensor t, int n)
    {
        Span<float> buf = stackalloc float[Math.Min(n, 64)];
        // Download just enough to compute l2/sum/first8 cheaply.
        var hostBuf = new float[n];
        _gpu.Synchronize();
        _gpu.Download(t, hostBuf);
        TraceHost(position, layer, tag, hostBuf, n);
    }

    private static void TraceHost(int position, int layer, string tag, float[] buf, int n)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        double s = 0, sum = 0;
        for (int i = 0; i < n; i++) { double v = buf[i]; s += v * v; sum += v; }
        float l2 = (float)Math.Sqrt(s);
        var sb = new System.Text.StringBuilder(220);
        sb.Append("[pos=").Append(position).Append(" L");
        if (layer < 0) sb.Append("--"); else sb.Append(layer);
        sb.Append(' ').Append(tag).Append("] l2=")
          .Append(l2.ToString("G6", inv))
          .Append(" sum=").Append(((float)sum).ToString("G6", inv))
          .Append(" first8=[");
        int k = Math.Min(8, n);
        for (int i = 0; i < k; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(buf[i].ToString("G6", inv));
        }
        sb.Append(']');
        Console.Error.WriteLine(sb.ToString());
    }

    private static void TraceLogits(int position, float[] logits)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        const int K = 5;
        Span<int> idx = stackalloc int[K];
        Span<float> val = stackalloc float[K];
        for (int i = 0; i < K; i++) { idx[i] = -1; val[i] = float.MinValue; }
        for (int i = 0; i < logits.Length; i++)
        {
            float lv = logits[i];
            for (int j = 0; j < K; j++)
            {
                if (lv > val[j])
                {
                    for (int s = K - 1; s > j; s--) { val[s] = val[s - 1]; idx[s] = idx[s - 1]; }
                    val[j] = lv; idx[j] = i;
                    break;
                }
            }
        }
        var sb = new System.Text.StringBuilder(160);
        sb.Append("[pos=").Append(position).Append(" top5]");
        for (int j = 0; j < K; j++)
        {
            sb.Append(' ').Append(idx[j]).Append('@')
              .Append(val[j].ToString("G6", inv));
        }
        Console.Error.WriteLine(sb.ToString());
    }

    // =================================================================
    //  Dispose
    // =================================================================

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // CPU buffers
        NativeMemory.Free(_cpuNormBuf);
        NativeMemory.Free(_cpuHiddenOut);
        NativeMemory.Free(_qkv);
        NativeMemory.Free(_qkvConv);
        NativeMemory.Free(_zVec);
        NativeMemory.Free(_qVHeads);
        NativeMemory.Free(_kVHeads);
        NativeMemory.Free(_alpha);
        NativeMemory.Free(_beta);
        NativeMemory.Free(_gdnOut);
        NativeMemory.Free(_hostQ);
        NativeMemory.Free(_cpuNormReadback);
        NativeMemory.Free(_ropeCosTable);
        NativeMemory.Free(_ropeSinTable);

        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            if (_ssmConv1d[i] != null) NativeMemory.Free(_ssmConv1d[i]);
            if (_ssmA[i] != null) NativeMemory.Free(_ssmA[i]);
            if (_ssmDtBias[i] != null) NativeMemory.Free(_ssmDtBias[i]);
            if (_ssmNormW[i] != null) NativeMemory.Free(_ssmNormW[i]);
            if (_cpuFfnGateInpShexp is not null && _cpuFfnGateInpShexp[i] != null)
                NativeMemory.Free(_cpuFfnGateInpShexp[i]);
        }

        if (_cpuMoe)
        {
            if (_cpuRouterLogits != null) NativeMemory.Free(_cpuRouterLogits);
            if (_cpuSharedOut != null) NativeMemory.Free(_cpuSharedOut);
            if (_cpuExpertGateAll != null) NativeMemory.Free(_cpuExpertGateAll);
            if (_cpuExpertUpAll != null) NativeMemory.Free(_cpuExpertUpAll);
            if (_cpuMoeHidden != null) NativeMemory.Free(_cpuMoeHidden);
        }
        else if (!_hp.IsMoE)
        {
            // Dense FFN scratch (allocated alongside _cpuMoeHidden on the dense path).
            if (_cpuFfnGateBuf != null) NativeMemory.Free(_cpuFfnGateBuf);
            if (_cpuFfnUpBuf   != null) NativeMemory.Free(_cpuFfnUpBuf);
            if (_cpuMoeHidden  != null) NativeMemory.Free(_cpuMoeHidden);
            // Per-layer GPU FFN slots populated by TryUploadDenseFfnLayers.
            if (_gpuWFfnGate is not null)
            {
                for (int i = 0; i < _gpuWFfnGate.Length; i++)
                {
                    if (_gpuWFfnGate[i]  is { } gT) _gpu.Free(gT);
                    if (_gpuWFfnUp![i]   is { } uT) _gpu.Free(uT);
                    if (_gpuWFfnDown![i] is { } dT) _gpu.Free(dT);
                }
            }
            if (_gpuFfnGateBufDense is { } gB) _gpu.Free(gB);
            if (_gpuFfnUpBufDense   is { } uB) _gpu.Free(uB);
        }

        // GPU scratch
        _gpu.Free(_gpuHidden);
        _gpu.Free(_gpuResidual);
        _gpu.Free(_gpuNormBuf);
        _gpu.Free(_gpuQGate);
        _gpu.Free(_gpuQ);
        _gpu.Free(_gpuGate);
        _gpu.Free(_gpuK);
        _gpu.Free(_gpuV);
        _gpu.Free(_gpuAttnOut);
        _gpu.Free(_gpuAttnScratch);
        if (_hp.IsMoE)
        {
            _gpu.Free(_gpuRouterLogits);
            _gpu.Free(_gpuFfnGate);
            _gpu.Free(_gpuFfnUp);
            _gpu.Free(_gpuExpertOut);
            _gpu.Free(_gpuSharedOut);
        }
        _gpu.Free(_gpuLogits);
        _gpu.Free(_pinnedHidden);

        _gpu.Free(_gpuGdnQkv);
        _gpu.Free(_gpuGdnQkvConv);
        _gpu.Free(_gpuGdnZVec);
        _gpu.Free(_gpuGdnQHead);
        _gpu.Free(_gpuGdnKHead);
        _gpu.Free(_gpuGdnVHead);
        _gpu.Free(_gpuGdnAlpha);
        _gpu.Free(_gpuGdnBeta);
        _gpu.Free(_gpuGdnOut);

        for (int i = 0; i < L; i++)
        {
            _gpu.Free(_gpuAttnNorm[i]);
            _gpu.Free(_gpuPostAttnNorm[i]);
            if (_hp.IsMoE)
            {
                _gpu.Free(_gpuWGateShexp[i]);
                _gpu.Free(_gpuWUpShexp[i]);
                _gpu.Free(_gpuWDownShexp[i]);
                if (!_cpuMoe)
                {
                    _gpu.Free(_gpuWGateInp[i]);
                    _gpu.Free(_gpuWGateInpShexp[i]);
                }
            }
            if (_hp.LayerTypes![i] == LayerType.Attention)
            {
                _gpu.Free(_gpuWQGate[i]);
                _gpu.Free(_gpuWK[i]);
                _gpu.Free(_gpuWV[i]);
                _gpu.Free(_gpuWO[i]);
                _gpu.Free(_gpuQNorm[i]);
                _gpu.Free(_gpuKNorm[i]);
                if (_gpuKCache[i] is { } kc) _gpu.Free(kc);
                if (_gpuVCache[i] is { } vc) _gpu.Free(vc);
            }
            else if (!_cpuGdn)
            {
                _gpu.Free(_gpuWAttnQkv[i]);
                _gpu.Free(_gpuWAttnGate[i]);
                _gpu.Free(_gpuWSsmOut[i]);
                _gpu.Free(_gpuWSsmAlpha[i]);
                _gpu.Free(_gpuWSsmBeta[i]);
                _gpu.Free(_gpuSsmA[i]);
                _gpu.Free(_gpuSsmDtBias[i]);
                _gpu.Free(_gpuSsmNormW[i]);
                _gpu.Free(_gpuSsmConv1d[i]);
                if (_gpuGdnScanState[i] is { } gs) _gpu.Free(gs);
                if (_gpuGdnConvState[i] is { } gc) _gpu.Free(gc);
            }
        }

        if (_gpuEmbedding is not null) _gpu.Free(_gpuEmbedding);
        if (_gpuOutputNorm is not null) _gpu.Free(_gpuOutputNorm);
        if (_gpuOutputWeight is not null && _gpuOutputWeight.Handle != _gpuEmbedding?.Handle)
            _gpu.Free(_gpuOutputWeight);

        if (_expertSlotManager is not null)
        {
            // SHARPI_EXPERT_STATS=<path>: dump SLRU hit rate, per-layer hit rate, and
            // top-3 experts per layer to the given file. Used to investigate whether
            // the expert access pattern is highly skewed (caching strategies matter)
            // or uniformly random (only more VRAM helps).
            var statsPath = Environment.GetEnvironmentVariable("SHARPI_EXPERT_STATS");
            if (!string.IsNullOrEmpty(statsPath))
            {
                using var w = new StreamWriter(statsPath);
                _expertSlotManager.Profiler.PrintStats(w);
            }

            _expertSlotManager.Dispose();
        }
        if (_snapshotBuf != null)
        {
            NativeMemory.Free(_snapshotBuf);
            _snapshotBuf = null;
        }

        if (_hasMtp)
        {
            _gpu.Free(_gpuMtpAttnNorm);
            _gpu.Free(_gpuMtpWQGate);
            _gpu.Free(_gpuMtpWK);
            _gpu.Free(_gpuMtpWV);
            _gpu.Free(_gpuMtpWO);
            _gpu.Free(_gpuMtpQNorm);
            _gpu.Free(_gpuMtpKNorm);
            _gpu.Free(_gpuMtpPostAttnNorm);
            _gpu.Free(_gpuMtpFfnGate);
            _gpu.Free(_gpuMtpFfnUp);
            _gpu.Free(_gpuMtpFfnDown);
            _gpu.Free(_gpuMtpEnorm);
            _gpu.Free(_gpuMtpHnorm);
            _gpu.Free(_gpuMtpSharedHeadNorm);
            _gpu.Free(_gpuMtpEhProj);
            if (_gpuMtpKCache is { } mkT) _gpu.Free(mkT);
            if (_gpuMtpVCache is { } mvT) _gpu.Free(mvT);
            _gpu.Free(_gpuMtpEmbedBuf);
            _gpu.Free(_gpuMtpEnormBuf);
            _gpu.Free(_gpuMtpHnormBuf);
            _gpu.Free(_gpuMtpConcatBuf);
            _gpu.Free(_gpuLastHidden);
            if (_lastHidden != null) NativeMemory.Free(_lastHidden);
            _mtpKvCache?.Dispose();
        }

        _kvCache.Dispose();
        _gdnStateCache.Dispose();
    }
}
