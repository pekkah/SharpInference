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

    // Experimental: move the MoE FFN (routed + shared experts) back to the CPU
    // while keeping attention on the GPU. SLRU expert thrash dominates per-token
    // cost on the default GPU path (~87% misses), so routing MoE through CPU
    // mmap reads can be a net win even with the extra GPU↔CPU round-trip.
    // See docs/qwen35moe-plan.md Phase 7 part C. Read per-instance (not static)
    // so the env var can be toggled between constructions during testing.
    private readonly bool _cpuMoe =
        Environment.GetEnvironmentVariable("SHARPI_CPU_MOE") == "1";

    // SHARPI_CPU_GDN=1 forces the legacy CPU GDN block path (Phase 7a baseline).
    // Default (unset) is the new full-GPU GDN block (Phase 7e+). Useful for
    // bisecting parity bugs and confirming the GPU kernels match CPU output.
    private readonly bool _cpuGdn =
        Environment.GetEnvironmentVariable("SHARPI_CPU_GDN") == "1";

    // ── CPU MoE state (only allocated/populated when _cpuMoe == true) ──
    // Packed MoE weight refs (mmap pointers; routed experts stay quantized on disk).
    private readonly CpuWeightRef[]? _cpuFfnGateInp;       // [L] router F32 [embDim, numExperts]
    private readonly CpuWeightRef[]? _cpuFfnGateShexp;     // [L] shared-expert gate
    private readonly CpuWeightRef[]? _cpuFfnUpShexp;       // [L] shared-expert up
    private readonly CpuWeightRef[]? _cpuFfnDownShexp;     // [L] shared-expert down
    private readonly CpuWeightRef[]? _cpuFfnGateExps;      // [L] packed [numExperts, expertDim, embDim]
    private readonly CpuWeightRef[]? _cpuFfnUpExps;        // [L] packed
    private readonly CpuWeightRef[]? _cpuFfnDownExps;      // [L] packed
    private readonly float*[]? _cpuFfnGateInpShexp;        // [L][embDim] F32 (preloaded; small)

    // CPU scratch for the MoE FFN path.
    private readonly float* _cpuRouterLogits;   // [numExperts]
    private readonly float* _cpuSharedOut;      // [embDim]
    private readonly float* _cpuExpertGate;     // [expertDim]
    private readonly float* _cpuExpertUp;       // [expertDim]
    private readonly float* _cpuExpertContrib;  // [embDim] — per-expert down-proj scratch
    private readonly float* _cpuMoeHidden;      // [embDim] — accumulator written back to _gpuHidden

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
        if (!hp.IsMoE || !hp.HasSharedExpert)
            throw new ArgumentException("CudaHybridGdnForwardPass currently requires MoE FFN with shared expert (qwen35moe).", nameof(hp));

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
        _gpuRouterLogits = gpu.Allocate(TensorShape.D1(_numExperts));
        _gpuFfnGate = gpu.Allocate(TensorShape.D1(_expertDim));
        _gpuFfnUp = gpu.Allocate(TensorShape.D1(_expertDim));
        _gpuExpertOut = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuSharedOut = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));

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
            _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embIsQuantized);
            _gpuOutputNorm = UploadWeight("output_norm.weight");
            _gpuOutputWeight = model.FindTensor("output.weight") is not null
                ? UploadWeight("output.weight")
                : _gpuEmbedding;
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
                "[CudaHybridGdnForwardPass] CPU MoE mode: routed + shared experts run on CPU (~2.3 GB/s mmap reads); SLRU disabled.");
            _cpuFfnGateInp = new CpuWeightRef[L];
            _cpuFfnGateShexp = new CpuWeightRef[L];
            _cpuFfnUpShexp = new CpuWeightRef[L];
            _cpuFfnDownShexp = new CpuWeightRef[L];
            _cpuFfnGateExps = new CpuWeightRef[L];
            _cpuFfnUpExps = new CpuWeightRef[L];
            _cpuFfnDownExps = new CpuWeightRef[L];
            _cpuFfnGateInpShexp = new float*[L];

            _cpuRouterLogits = Alloc(_numExperts);
            _cpuSharedOut = Alloc(_embDim);
            _cpuExpertGate = Alloc(_expertDim);
            _cpuExpertUp = Alloc(_expertDim);
            _cpuExpertContrib = Alloc(_embDim);
            _cpuMoeHidden = Alloc(_embDim);
        }
        else
        {
            _cpuRouterLogits = null;
            _cpuSharedOut = null;
            _cpuExpertGate = null;
            _cpuExpertUp = null;
            _cpuExpertContrib = null;
            _cpuMoeHidden = null;
        }

        Console.Error.Write("[CudaHybridGdnForwardPass] Uploading per-layer weights...");
        for (int i = 0; i < L; i++)
        {
            // Common (both block types): norms + MoE FFN weights live on GPU.
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuPostAttnNorm[i] = UploadWeight($"blk.{i}.post_attention_norm.weight");
            if (!_cpuMoe)
            {
                _gpuWGateInp[i] = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                _gpuWGateInpShexp[i] = UploadWeight($"blk.{i}.ffn_gate_inp_shexp.weight");
                _gpuWGateShexp[i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                _gpuWUpShexp[i] = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                _gpuWDownShexp[i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");
            }
            else
            {
                _cpuFfnGateInp![i] = ResolveCpuWeight($"blk.{i}.ffn_gate_inp.weight");
                _cpuFfnGateShexp![i] = ResolveCpuWeight($"blk.{i}.ffn_gate_shexp.weight");
                _cpuFfnUpShexp![i] = ResolveCpuWeight($"blk.{i}.ffn_up_shexp.weight");
                _cpuFfnDownShexp![i] = ResolveCpuWeight($"blk.{i}.ffn_down_shexp.weight");
                _cpuFfnGateExps![i] = ResolveCpuWeight($"blk.{i}.ffn_gate_exps.weight");
                _cpuFfnUpExps![i] = ResolveCpuWeight($"blk.{i}.ffn_up_exps.weight");
                _cpuFfnDownExps![i] = ResolveCpuWeight($"blk.{i}.ffn_down_exps.weight");
                _cpuFfnGateInpShexp![i] = LoadF32Tensor($"blk.{i}.ffn_gate_inp_shexp.weight", _embDim);
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

        // ── SLRU expert slot manager ───────────────────────────────────
        // Compute slot capacity from remaining VRAM. The plan calls for sizing
        // capacity by (remaining VRAM) / (per-expert bytes). For qwen35moe Q4_K_M:
        //   per expert ≈ 1.81 MiB (gate+up+down for one expert across 3 tensors)
        // We're conservative — most of the remaining budget is reserved for the
        // attention KV cache (10 layers × maxSeqLen × kvDim × 4 B × 2) and various
        // scratch. Use the GpuKvBytes from placement when the planner sized it.
        if (_cpuMoe)
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
        throw new NotSupportedException(
            $"CudaHybridGdnForwardPass.TruncateTo({length}): GDN state is destructively " +
            $"updated and cannot be partially rewound; only length == 0 or current ({_gdnStateCache.Length}) is supported.");
    }

    public void ResetCache()
    {
        _kvCache.Reset();
        _gdnStateCache.Reset();
        if (!_cpuGdn)
        {
            // Zero GPU-resident scan + conv state for every GDN layer.
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                if (_gpuGdnScanState[i] is { } scan) _gpu.Clear(scan);
                if (_gpuGdnConvState[i] is { } conv) _gpu.Clear(conv);
            }
        }
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

            if (_cpuMoe)
            {
                // Download _gpuNormBuf → _cpuNormBuf, run MoE on CPU, upload result.
                _gpu.Synchronize();
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

        // 5. Final norm + output projection on GPU
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);

        // 6. Download logits to host
        _gpu.Synchronize();
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
    //  CPU GDN block — mirror of HybridGdnForwardPass.GdnBlock
    // =================================================================

    private void CpuGdnBlock(int layer, int position)
    {
        // Download _gpuNormBuf → _cpuNormBuf so the CPU GDN kernels can consume it.
        _gpu.Synchronize();
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
        //    Per-layer cost: 1 KB readback + sync — fine.
        _gpu.Synchronize();
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
        //   Note: _gpuNormBuf is already populated; just need the readback for the dot.
        _gpu.Synchronize();
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
    //  CPU MoE FFN (SHARPI_CPU_MOE=1) — mirror of HybridGdnForwardPass.MoeFfn
    //  Consumes _cpuNormBuf, produces _cpuMoeHidden.
    // =================================================================

    private void CpuMoeFfn(int layer)
    {
        int numExperts = _numExperts;
        int numActive = _numActiveExperts;
        int expertDim = _expertDim;

        // 1. Router: ffn_gate_inp.weight is F32 [embDim, numExperts]; softmax then top-K.
        var routerW = _cpuFfnGateInp![layer];
        SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, _cpuNormBuf,
            numExperts, _embDim, routerW.DType);
        SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopKPtr(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights);

        // 2. Shared expert: ffn_down @ (SiLU(ffn_gate @ x) * (ffn_up @ x)).
        var gateShexp = _cpuFfnGateShexp![layer];
        var upShexp = _cpuFfnUpShexp![layer];
        var downShexp = _cpuFfnDownShexp![layer];
        SimdKernels.MatVec(_cpuExpertGate, gateShexp.DataPtr, _cpuNormBuf,
            expertDim, _embDim, gateShexp.DType);
        SimdKernels.MatVec(_cpuExpertUp, upShexp.DataPtr, _cpuNormBuf,
            expertDim, _embDim, upShexp.DType);
        SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, expertDim);
        SimdKernels.MatVec(_cpuSharedOut, downShexp.DataPtr, _cpuExpertGate,
            _embDim, expertDim, downShexp.DType);

        // Per-token sigmoid scalar gate on the shared expert output.
        float shexpDot = SimdKernels.DotF32(_cpuFfnGateInpShexp![layer], _cpuNormBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        SimdKernels.ScaleInPlace(_cpuSharedOut, shexpScale, _embDim);

        // 3. Routed experts (sparse top-K). Accumulate into _cpuMoeHidden.
        new Span<float>(_cpuMoeHidden, _embDim).Clear();
        var gateExps = _cpuFfnGateExps![layer];
        var upExps = _cpuFfnUpExps![layer];
        var downExps = _cpuFfnDownExps![layer];
        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            ExpertMatVec(_cpuExpertGate, gateExps, expertIdx, expertDim, _embDim, _cpuNormBuf);
            ExpertMatVec(_cpuExpertUp, upExps, expertIdx, expertDim, _embDim, _cpuNormBuf);
            SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, expertDim);
            ExpertMatVecDown(_cpuMoeHidden, downExps, expertIdx, _embDim, expertDim,
                _cpuExpertGate, weight);
        }

        // 4. Add shared expert output.
        SimdKernels.AddInPlace(_cpuMoeHidden, _cpuSharedOut, _embDim);
    }

    private void ExpertMatVec(float* output, in CpuWeightRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    private void ExpertMatVecDown(float* output, in CpuWeightRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(_cpuExpertContrib, expertData, input, rows, cols, packedTensor.DType);
        SimdKernels.WeightedAddInPlace(output, _cpuExpertContrib, weight, rows);
    }

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

        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q5_K || info.DType == DType.Q6_K)
        {
            // CUDA matvec dispatches on Q4_K / Q5_K / Q6_K via dedicated kernels.
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType);
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Q8_0, Q3_K, etc. — CUDA matvec doesn't dispatch on these. Dequantize to F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
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
            var result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
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
        var t = _gpu.Upload(f32, TensorShape.D1(count));
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
            if (_cpuExpertGate != null) NativeMemory.Free(_cpuExpertGate);
            if (_cpuExpertUp != null) NativeMemory.Free(_cpuExpertUp);
            if (_cpuExpertContrib != null) NativeMemory.Free(_cpuExpertContrib);
            if (_cpuMoeHidden != null) NativeMemory.Free(_cpuMoeHidden);
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
        _gpu.Free(_gpuRouterLogits);
        _gpu.Free(_gpuFfnGate);
        _gpu.Free(_gpuFfnUp);
        _gpu.Free(_gpuExpertOut);
        _gpu.Free(_gpuSharedOut);
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
            if (!_cpuMoe)
            {
                _gpu.Free(_gpuWGateInp[i]);
                _gpu.Free(_gpuWGateInpShexp[i]);
                _gpu.Free(_gpuWGateShexp[i]);
                _gpu.Free(_gpuWUpShexp[i]);
                _gpu.Free(_gpuWDownShexp[i]);
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
        _kvCache.Dispose();
        _gdnStateCache.Dispose();
    }
}
