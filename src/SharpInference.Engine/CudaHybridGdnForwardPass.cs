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
    // Flash-decoding split-KV (#238) for the GPU attention layers' per-token decode. Same
    // mechanism as the dense/MoE-hybrid passes; null → single-block. GDN decode is GDN-scan +
    // MoE dominated with only the attention layers' KV growing, so this is measurement-gated.
    private Tensor? _splitKvPartialO;
    private Tensor? _splitKvPartialMeta;
    private readonly bool _splitDecodeEnabled =
        Environment.GetEnvironmentVariable("SHARPI_SPLIT_DECODE") != "0";
    private readonly string? _splitGroupedMode =
        Environment.GetEnvironmentVariable("SHARPI_SPLIT_DECODE_GROUPED");
    // GDN attention is a smaller share than the MoE hybrid (only ~10 attention layers vs the GDN
    // scan + MoE), so the split overhead amortizes only at genuinely long ctx. #238 GDN A/B
    // (Qwen3.6-35B-A3B, bf16, 4070 Ti, CPU-page-cache-warm) showed a clean +19% at 16K; the 8K
    // number was cold-cache-confounded (mmap'd CPU-MoE). Threshold conservatively at the
    // clean-win region (8192) since with so few attention layers a moderate-ctx split risks the
    // OLMoE-style overhead regression and 4096–8192 is unverified for this pass.
    private const int GdnSplitMinSeq = 8192;
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

    // GPU KV cache (sized [numLayers]; only attention slots are allocated).
    // Element dtype is governed by `_kvDType` — F32 (legacy) or BFloat16 (#27).
    private readonly Tensor?[] _gpuKCache;       // [L][maxSeq * kvDim]
    private readonly Tensor?[] _gpuVCache;       // [L][maxSeq * kvDim]

    // Issue #27: KV-cache element dtype. Default Bf16 — halves cache footprint
    // on the 16 attention layers (qwen35 27B-MTP / qwen35moe), freeing ~256 MiB
    // at ctx=4096 to admit more dense-FFN layers on GPU. `SHARPI_KV_DTYPE=fp32`
    // restores the legacy fp32 path for bisecting any precision regression.
    private readonly DType _kvDType;

    // Embedding + output
    private readonly Tensor? _gpuEmbedding;
    private readonly DType _embDType;  // dtype of the on-GPU embedding bytes (Q4_K, Q5_K, or Float32)
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
    private readonly Tensor[] _gpuWAttnQkv;          // [L] raw-quant/F32 [conv_channels, embDim]
    private readonly Tensor[] _gpuWAttnGate;         // [L] raw-quant/F32 [value_dim, embDim]
    private readonly Tensor[] _gpuWSsmOut;           // [L] raw-quant/F32 [embDim, value_dim]
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

    // Q3_K / Q8_0 routed-expert MoE rows can run through the int-domain
    // DotQ3K_Q8KS / DotQ8_0_Q8KS kernels instead of the FP dequant-FMA path.
    // The float input is prepacked to Q8_KS (per-32-element sub-block scales,
    // issue #107) once per CpuMoeFfnCore call (Phase A: cpuNormIn; Phase C:
    // each gateAll slice) and shared across all (numActive × expertDim) rows —
    // same prepack feeds both kernels, so either gate enables the scratch
    // allocation.
    //
    // Resolved per-instance from the model's routed-expert dtype mix: auto-on
    // when the model has at least one Q3_K (or Q8_0) routed-expert layer —
    // this is mainly the APEX mixed-precision tier (Carnice etc.) where the
    // legacy FP kernels are the decode bottleneck. SHARPI_Q3K_Q8K=1 / =0 and
    // SHARPI_Q8_0_Q8K=1 / =0 force the latch on / off respectively, overriding
    // auto-detect. Validation log docs/q8k-validation-2026-05-31.md: Q8_KS
    // cuts the MTP-accept drift envelope from ±13 pp (plain Q8_K) to ±3 pp
    // on Carnice, with every prompt argmax-stable through the full 32-token
    // capture window. The residual ±3 pp is at the draft-cycle noise floor
    // (1 flipped draft of 30); output text is functionally equivalent to
    // the FP path on every prompt measured.
    private readonly bool _q3kQ8KEnabled;
    private readonly bool _q8_0Q8KEnabled;
    private readonly bool _q4kQ8KEnabled;

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
    // Q8_KS prepacked inputs (per-32-element sub-block scales, issue #107) for
    // the routed-expert dot kernels (allocated only when _q3kQ8KEnabled).
    // _cpuNormInQ8K holds the post-RmsNorm hidden quantised to Q8_KS for
    // Phase A (gate+up) — rewritten per CpuMoeFfnCore call when gate or up is
    // Q3_K. _cpuExpertGateAllQ8K holds numActive contiguous Q8_KS-packed
    // expertDim slices of the post-SiLuMul gate buffer for Phase C (down) —
    // rewritten per CpuMoeFfnCore call when down is Q3_K.
    private readonly byte* _cpuNormInQ8K;
    private readonly byte* _cpuExpertGateAllQ8K;
    private readonly int _cpuExpertGateAllQ8KStride;
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
    // Token-2 GPU FFN scratch for the batched verify path (issue #30). Lazy-allocated
    // alongside the token-1 buffers so the GPU dense FFN can be invoked per token.
    private Tensor? _gpuFfnGateBufDense2;     // [_intermDim] f32
    private Tensor? _gpuFfnUpBufDense2;       // [_intermDim] f32
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

    // MTP MoE FFN tensors (qwen35moe 35B-A3B-MTP). Populated only when _mtpIsMoE.
    // Shared expert weights live on GPU (the shared expert MatMul fires in parallel
    // with the CPU routed loop). Routed experts mmap'd on CPU since the 256-expert
    // stack at the MTP block alone is ~470 MiB Q4_K/Q5_K — won't co-reside with the
    // trunk routed weights on a 12 GB-class card.
    private readonly bool _mtpIsMoE;
    private readonly Tensor _gpuMtpWGateShexp;
    private readonly Tensor _gpuMtpWUpShexp;
    private readonly Tensor _gpuMtpWDownShexp;
    private readonly CpuWeightRef _cpuMtpFfnGateInp;     // router F32 [embDim, numExperts]
    private readonly CpuWeightRef _cpuMtpFfnGateExps;    // [numExperts, expertDim, embDim]
    private readonly CpuWeightRef _cpuMtpFfnUpExps;
    private readonly CpuWeightRef _cpuMtpFfnDownExps;
    private readonly float* _cpuMtpFfnGateInpShexp;      // [embDim] F32 shared-expert sigmoid gate

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

    // ── Issue #30 batched-verify scratch (only when _hasMtp && !IsMoE) ─
    // Mirrors HybridGdnForwardPass; the second token gets its own residual
    // stream + norm buffer + output logits. Attn/GDN scratch is reused
    // across t1 and t2 within a layer iteration (sequential block calls).
    private readonly Tensor _gpuHidden2;          // [embDim]
    private readonly Tensor _gpuResidual2;        // [embDim]
    private readonly Tensor _gpuNormBuf2;         // [embDim]
    private readonly Tensor _gpuLogits2;          // [vocabSize]
    // BatchForward2 (SHARPI_CPU_GDN=1 debug trunk) snapshots t1's pre-norm hidden into
    // these so it can ride the queued DownloadAsync alongside t2's _gpuLastHidden, then
    // copies it into the MTP hidden history. Internal scratch for that path only — the
    // production k-token BatchVerify path writes the history straight from the device
    // stream (the dead public LastHiddenT1 accessor was removed in issue #209).
    private readonly Tensor _gpuLastHiddenT1;     // [embDim] — t1 hidden device snapshot
    private readonly float[] _logitsBuf2;         // host download for token 2 logits
    private readonly float* _cpuNormBuf2;         // [embDim] — t2's norm download for CPU FFN path
    private readonly float* _cpuMoeHidden2;       // [embDim] — t2's CPU FFN output
    private readonly float* _lastHiddenT1;        // [embDim] — pinned t1 hidden host target

    private byte* _batchSnapshotBuf;
    private long _batchSnapshotCap;
    private bool _batchSnapshotValid;
    private bool _bvArgmaxOnly;                                // #219 greedy-verify fast path
    private (int Index, float Value)[] _bvArgmaxResult = [];   // #219 result stashed by the tail
    private int _batchStartPos;        // startPos of the most recent batched verify
    private int _batchK;               // token count of the most recent batched verify

    // ── Device-side GDN snapshot ring (issues #30/#207 goal 4, #290) ────
    // On the GPU-GDN trunk (!_cpuGdn) the live recurrent state is the per-layer
    // _gpuGdnScanState/_gpuGdnConvState device tensors, so rollback snapshots must
    // be captured on-device: ring slot j holds every GDN layer's (scan, conv) state
    // AFTER batch token j, packed per layer at offset gdnIdx × per-layer-floats.
    // #290: the ring is one flat contiguous tensor per (scan, conv) — slot j sits
    // at j × (numGdn × per-layer-floats) — so the fused #114-B scan kernel can
    // stride across slots and dump each token's state in place (dropping the
    // per-position relaunch + the bulk CopyDeviceRegion fan-out). Allocated in the
    // constructor BEFORE TryUploadDenseFfnLayers fills VRAM (~60 MB/slot for 27B;
    // landing it in WDDM-paged memory would 5-10× the verify). _gdnRingSlots is
    // the achieved slot count (alloc retries with fewer slots on OOM);
    // SupportsBatchVerify requires ≥ 1 slot on this trunk. The host
    // _batchSnapshotBuf above serves the SHARPI_CPU_GDN=1 debug trunk only.
    private readonly Tensor? _gpuGdnRingScan;   // [slots × numGdn × scanFloatsPerLayer]
    private readonly Tensor? _gpuGdnRingConv;   // [slots × numGdn × convFloatsPerLayer]
    private readonly int _gdnRingSlots;

    // Batched-verify scratch (exact-k; reallocated when the batch size changes the
    // same way EnsureBatchedFfnScratch is — GEMM-N derives rows from ElementCount/k).
    private Tensor? _gpuBvLogitsAll;   // [k × vocab] all-position logits
    private Tensor? _gpuBvFfnAll;      // [k × embDim] CPU-FFN upload staging / combine
    private float* _bvNormHost;        // pinned [k × embDim] — moeNorm download for CPU FFN
    private float* _bvFfnHost;         // pinned [k × embDim] — CPU FFN outputs for upload
    private float[]? _bvLogitsHost;    // managed [k × vocab] download target
    private int _bvCap;

    // MTP block-out hidden of the most recent MtpForward (pre-shared-head-norm),
    // pinned so the capture rides the queued D2H stream (issue #30 draft chaining).
    private float* _mtpSelfHidden;

    // Max tokens per BatchVerify call = ring slots + 1. Each slot costs ~149 MiB
    // of VRAM that TryUploadDenseFfnLayers would otherwise fill with ~2 dense FFN
    // layers, so the default (4 → 3 slots) is the smallest ring that reaches the
    // measured k=4 optimum now that the 4-input CPU FFN kernel (issue #209) amortizes
    // the dominant mmap weight read four ways. Instance-resolved at construction so
    // tests can override per instance; the knob semantics live in one place
    // (GdnStateCache.ResolveMtpBatchMax) shared with the CPU pass.
    private readonly int _mtpBatchMax = GdnStateCache.ResolveMtpBatchMax();
    // Token-2 host FFN scratch (intermediate gate/up post-MatVec2In, pre-SiLuMul).
    private readonly float* _cpuFfnGateBuf2;
    private readonly float* _cpuFfnUpBuf2;
    // Lane-3/4 host FFN scratch (issue #209): CpuDenseFfn4 dots one CPU mmap weight
    // read against four draft tokens via MatVec4In, so it needs four distinct
    // gate/up scratch slabs (SiLU reads them per-lane before the down projection).
    private readonly float* _cpuFfnGateBuf3;
    private readonly float* _cpuFfnUpBuf3;
    private readonly float* _cpuFfnGateBuf4;
    private readonly float* _cpuFfnUpBuf4;

    // Host-side hidden history; see HybridGdnForwardPass field-level doc.
    private float* _mtpPrefillHiddens;     // [_mtpPrefillHiddensCap × embDim], slot p = h_p
    private int _mtpPrefillHiddensCap;     // allocated capacity in tokens
    private int _mtpHiddenHistoryLength;   // slots [0.._mtpHiddenHistoryLength) populated

    // ── SnapKV (issue #58): GPU prefill-eviction scratch ───────────────
    // Allocated lazily on the first SnapKV-active Prefill (Budget > 0 and prompt
    // exceeds budget + window). Sized for the current N — grow-only.
    //
    // _snapKvQCapture stores post-RoPE / post-Q-norm queries for the trailing
    // W tokens, per attention layer: layout [_numAttnLayers × W × qDim] floats.
    // Captured inline from GpuAttnBlockAt when _snapKvCaptureSlot >= 0.
    // _snapKvScoreAccum is the per-position importance accumulator, summed
    // across (queries × heads × attention layers) on GPU via atomicAdd.
    private readonly SnapKvConfig _snapKvCfg;
    private readonly int _snapKvEffectiveBudget; // resolved at construction: explicit env, or auto-derived from maxSeqLen + free VRAM
    private readonly int _numAttnLayers;
    private readonly int[] _attnLayerIndexOf; // [numLayers] — 0-based attn-layer index, -1 for GDN layers
    private Tensor? _snapKvQCapture;        // [_numAttnLayers × W × qDim] f32
    private int _snapKvQCaptureW;           // cached W the buffer was sized for
    private Tensor? _snapKvScoreAccum;      // [maxSeqLen] f32
    // Transient state set by the Prefill loop and read by GpuAttnBlockAt to
    // drive Q-capture without changing the per-token Forward signature.
    // Non-negative only inside SnapKV-active prefill in the capture window.
    private int _snapKvCaptureSlot;         // 0..W-1 for tokens in the capture window; -1 otherwise

    private bool _disposed;

    // Issue #110/#111: batched prefill is non-transactional — it mutates the GDN
    // scan/conv recurrent state and writes KV pages as it goes, deferring the
    // length-counter bookkeeping to the end. A throw mid-chunk (CUDA stream fault,
    // OOM) therefore leaves the recurrent state partially advanced while the length
    // counters still read startPos. Retrying any forward call would run on poisoned
    // state and silently produce wrong output. We latch this fault and refuse all
    // subsequent forward entries so the caller must discard the pass / reload the
    // model rather than getting garbage tokens.
    private bool _faulted;

    private void ThrowIfFaulted()
    {
        if (_faulted)
            throw new InvalidOperationException(
                "CudaHybridGdnForwardPass: a prior batched prefill faulted mid-chunk, leaving the " +
                "GDN recurrent state corrupted. This instance can no longer produce correct output — " +
                "discard it and reload the model.");
    }

    // Prefill profiling (SHARPI_PREFILL_PROFILE=1): accumulate synchronous CPU-MoE
    // wall time vs total Forward wall time to size the batching opportunity.
    private static readonly bool _prefillProfile =
        Environment.GetEnvironmentVariable("SHARPI_PREFILL_PROFILE") == "1";
    private long _profMoeTicks;
    private long _profTotalTicks;
    private int _profTokens;
    // Decode profiling (SHARPI_DECODE_PROFILE=1): per-token GPU-trunk vs CPU-MoE split. The
    // trunk launches are async and already drained at the per-layer MoE Download sync, so adding
    // an explicit sync right after the trunk just moves that drain earlier (≈no perturbation) and
    // attributes the GPU compute. Read the trunk:moe ratio to find the decode pole.
    private static readonly bool _decodeProfile =
        Environment.GetEnvironmentVariable("SHARPI_DECODE_PROFILE") == "1";
    private long _pdTrunkTicks, _pdMoeTicks;
    // Finer CPU-MoE decode breakdown (also gated on _decodeProfile): router (matvec+softmax+top-k),
    // Phase-A (gate+up dots), Phase-C (down dots), and the GPU shared-expert download+combine.
    // Pinpoints whether the MoE wall time is the RAM-bound expert dots or the coordination overhead.
    private long _pdRouterTicks, _pdPhaseATicks, _pdPhaseCTicks, _pdSharedTicks;
    private int _pdTokens;
    // #388: per-layer CUDA-graph of the decode GPU trunk block (eliminates ~600 kernel launches/token
    // that starve the launch-bound decode). Each layer's pure-GPU trunk (norms + attn/GDN block + residual
    // adds) is captured into a per-layer graph (keyed by layer index) on the first eligible decode token and
    // replayed after; the CPU-MoE (Download/CpuMoeFfn/Upload) stays OUTSIDE the graph between replays.
    // Gated SHARPI_DECODE_CUDA_GRAPH (default OFF). Falls back to direct launches on any capture failure.
    private static readonly bool _decodeCudaGraph =
        Environment.GetEnvironmentVariable("SHARPI_DECODE_CUDA_GRAPH") == "1";
    private bool[]? _layerGraphCaptured;       // per-layer: graph captured+ready
    private bool _decodeGraphDisabled;         // latched off after any capture failure
    private int _decodeTokensSeen;             // warmup counter: capture only after on-demand scratch settles (mirrors llama.cpp's 2-token warmup)
    private const int GraphWarmupTokens = 2;
    private bool _graphDiagLogged;             // one-shot [graph-diag] capture-coverage log
    // Sub-phase breakdown of the batched routed-MoE (BatchedRoutedExperts), accumulated
    // across the layer loop and printed once per chunk on the [moe-subphase] line when
    // SHARPI_PREFILL_PROFILE=1. Establishes whether routed-MoE prefill cost is in the
    // int8 dots (phaseA gate+up, phaseC down) or the per-token Q8_KS quantization /
    // bucketing overhead — the diagnostic that confirmed #112's routed-MoE is dot-bound
    // (96% phaseA+phaseC) and pinpoints the next mixed-quant cliff (issue #168). Zero
    // cost when profiling is off (all sites guarded by _prefillProfile).
    private long _profMoeNormQ, _profMoePhaseA, _profMoeSilu, _profMoeGateQ, _profMoePhaseC, _profMoeBucket;

    // ── Batched prefill (issue #110) ──────────────────────────────────────
    // Per-layer batched prompt prefill for the CPU-MoE GDN-hybrid path. The
    // trunk (attention + GDN) stays sequential per token (GDN recurrence and KV
    // append are positional), but the routed MoE experts — 78-83% of prefill
    // wall, DRAM-bound on mmap weight reads — are grouped by selected expert so
    // each expert's weight rows are read once per chunk and dotted against every
    // token routing to it, instead of re-reading per token. Byte-parity with the
    // sequential path is preserved: the same DispatchDot/DispatchDotQ8K kernels
    // run with identical per-token top-k accumulation order. Disable with
    // SHARPI_BATCHED_PREFILL=0 (falls back to the sequential Forward loop).
    // Settable (not readonly) so the A/B parity test can toggle batched vs
    // sequential prefill within one process; resolved from the env at class load.
    internal static bool BatchedPrefillEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_PREFILL") != "0";
    // Issue #162: route batched-prefill trunk matmuls through the compute-bound path
    // (weight read once per batch) instead of the per-token GEMM-N matvec that re-streams
    // the whole weight once per token. Q8_0/Q4_K → int8 MMQ; Q6_K/Q5_K (no MMQ kernel) →
    // dequant→fp16→cuBLAS GEMM. The GDN trunk is mostly Q4_K (attn q/k/o + dense FFN) with
    // Q6_K/Q5_K islands (e.g. Qwen3.6-27B-MTP: Q6_K attn_qkv/ffn_down, Q5_K ssm_out), so
    // routing ONLY Q6_K/Q5_K left the Q4_K bulk on the memory-bound fallback. Like the
    // all-GPU path (CudaForwardPass), MMQ/GEMM are argmax-stable, NOT byte-exact — so the
    // bit-parity oracle (CudaHybridGdnBatchedPrefillTests) pins this OFF to keep validating
    // the byte-exact matvec batching; the MMQ/GEMM kernels' correctness is covered by
    // CudaMmqQ4K/Q8_0Tests + CudaGemmQ6K/Q5KTests. Settable for that test;
    // SHARPI_GDN_PREFILL_COMPUTE=0 reverts to the byte-exact per-token matvec.
    internal static bool GdnPrefillComputeEnabled =
        Environment.GetEnvironmentVariable("SHARPI_GDN_PREFILL_COMPUTE") != "0";
    // Keep Q8_0 trunk weights raw on the GPU instead of dequantizing them to F32 at
    // upload. CUDA has the full raw-Q8_0 kernel suite for both phases — llm_matvec_q8_0
    // (+ dp4a int8) for decode and llm_mmq_q8_0 (int8 MMQ) / llm_matvec_q8_0_gemm_n for
    // batched prefill — so the only thing the legacy F32 dequant bought was the
    // memory-bound llm_matvec_f32_gemm_n path (the prefill #1 GPU cost on Q8_0-trunk
    // models such as Qwen3.6-35B-A3B-UD: 78% of GPU time) plus 4× the trunk VRAM. The
    // MMQ/dp4a kernels are argmax-stable, NOT byte-exact (Q8_1 int8 activations), so this
    // shares the same contract as GdnPrefillComputeEnabled; SHARPI_GDN_RAW_Q8_0=0 reverts
    // to the F32-dequant upload (the byte-exact reference the bit-parity oracle pins).
    internal static bool RawQ80WeightsEnabled =
        Environment.GetEnvironmentVariable("SHARPI_GDN_RAW_Q8_0") != "0";
    // Issue #210: route the k MTP-draft tokens' routed-expert FFN in BatchVerify
    // through the #110 group-by-expert core (BatchedRoutedExperts) instead of the
    // per-token CpuMoeFfnCore loop, so each selected expert's mmap'd gate/up/down
    // rows are read once and dotted against every draft that routed to it. The win
    // scales with expert overlap across the (adjacent-position) draft chain. The
    // routed output is bit-identical to the per-token path (same DispatchDot/
    // DispatchDotQ8K kernels, same top-k accumulation order) — the shared expert
    // and (routed+shared)+resid combine mirror the per-token operand order exactly.
    // SHARPI_MTP_BATCHED_MOE_VERIFY=0 reverts to the per-token loop for parity
    // bisection. Settable (not readonly) so the A/B parity test can toggle it.
    internal static bool BatchedMoeVerifyEnabled =
        Environment.GetEnvironmentVariable("SHARPI_MTP_BATCHED_MOE_VERIFY") != "0";
    private int _bCap;                 // token capacity the batched scratch is sized for (grow-only)
    private Tensor? _gpuStreamAll;     // [N × embDim] inter-layer residual stream for all tokens
    private float* _bResidAll;         // [bCap × embDim] pinned — per-token MoE residual (postBlock hidden)
    private float* _bNormAll;          // [bCap × embDim] pinned — per-token post-attn-norm (MoE input)
    private float* _btRouterAll;       // [N × numExperts] router-logit readback (CPU-MoE GPU-router, #388)
    private float* _bSharedAll;        // [bCap × embDim] pinned — per-token shared-expert out (unscaled)
    private float* _bHiddenAll;        // [bCap × embDim] pinned — combined hidden, uploaded to _gpuStreamAll
    private float* _bRoutedAll;        // [bCap × embDim] — routed-expert accumulator (host only)
    private float* _bGateAll;          // [bCap × numActive × expertDim] — gate projections
    private float* _bUpAll;            // [bCap × numActive × expertDim] — up projections
    private float* _bDownPartial;      // [bCap × numActive × embDim] — per-(token,slot) down dots (pre-reduce)
    private byte*  _bNormAllQ8K;       // [bCap × q8ksEmbBytes] — Q8_KS-packed norms (when q3k/q8_0 gate/up)
    private byte*  _bGateAllQ8K;       // [bCap × numActive × q8ksExpBytes] — Q8_KS-packed silu'd gate slices
    private int    _bQ8KEmbStride;     // Q8_KS bytes for an embDim row
    private int    _bQ8KExpStride;     // Q8_KS bytes for an expertDim row
    private int*   _bSelected;         // [bCap × numActive] — per-token selected experts
    private float* _bWeights;          // [bCap × numActive] — per-token expert weights
    private float* _bShexpScale;       // [bCap] — per-token shared-expert sigmoid gate
    private int*   _bExpStart;         // [numExperts+1] — CSR offsets into _bExpTokI/_bExpTokK
    private int*   _bExpCursor;        // [numExperts] — fill cursor (reused per layer)
    private int*   _bExpTokI;          // [bCap × numActive] — token index, grouped by expert
    private int*   _bExpTokK;          // [bCap × numActive] — slot (top-k rank), grouped by expert
    private int*   _bUsedExperts;      // [numExperts] — compact list of experts with ≥1 token this layer

    // ── Batched trunk (issue #111) ────────────────────────────────────────
    // Device-side per-token activation buffers for the GEMM-batched trunk. The
    // projection matvecs (GDN qkv/z/alpha/beta/ssm_out, attn q/k/v/o, shared
    // gate/up/down) run as single GEMM-N launches over all N tokens; the conv1d /
    // delta-net recurrence and KV-append / SDPA stay per-position, reading per-token
    // slices via CudaBackend.View. Token-major layout ([N × dim]) matches MatMulBatched.
    // Allocated grow-only by EnsureBatchedTrunkScratch; null when _cpuGdn (the CPU-GDN
    // debug path keeps the sequential per-token trunk). Disabled with
    // SHARPI_BATCHED_TRUNK=0 (falls back to the per-token trunk loop).
    internal static bool BatchedTrunkEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_TRUNK") != "0";
    private int _btCap;
    private Tensor? _gpuBtNorm;      // [N × embDim] attn-norm output (block input)
    private Tensor? _gpuBtBlockOut;  // [N × embDim] block output → postBlock (resid)
    private Tensor? _gpuBtMoeNorm;   // [N × embDim] post-attn-norm (MoE input)
    private Tensor? _gpuBtRouterAll; // [N × numExperts] batched router logits (CPU-MoE GPU-router, #388)
    private Tensor? _gpuBtShared;    // [N × embDim] shared-expert output (unscaled)
    private Tensor? _gpuBtQkv;       // [N × convChannels] GDN joint QKV
    private Tensor? _gpuBtZ;         // [N × valueDim] GDN z-gate
    private Tensor? _gpuBtAlpha;     // [N × numVHeads] GDN alpha
    private Tensor? _gpuBtBeta;      // [N × numVHeads] GDN beta
    private Tensor? _gpuBtGdnOut;    // [N × valueDim] GDN recurrence output
    private Tensor? _gpuBtQGate;     // [N × qDim*2] attn Q‖gate
    private Tensor? _gpuBtQ;         // [N × qDim] attn Q
    private Tensor? _gpuBtGate;      // [N × qDim] attn GLU gate
    private Tensor? _gpuBtK;         // [N × kvDim] attn K
    private Tensor? _gpuBtV;         // [N × kvDim] attn V
    private Tensor? _gpuBtAttnOut;   // [N × qDim] attn output (pre-O)
    private Tensor? _gpuBtSGate;     // [N × expertDim] shared-expert gate
    private Tensor? _gpuBtSUp;       // [N × expertDim] shared-expert up
    // Issue #114-B fused-GDN-scan scratch (only allocated when BatchedGdnScanEnabled).
    private Tensor? _gpuBtQkvConv;   // [N × convChannels] post-conv1d + SiLU
    private Tensor? _gpuBtQHead;     // [N × valueDim] tiled GDN query heads
    private Tensor? _gpuBtKHead;     // [N × valueDim] tiled GDN key heads

    // ── Batched FFN / MoE (issue #121) ─────────────────────────────────────
    // Device buffers for the GEMM-N-batched FFN stage of PrefillBatchedTrunkGpuFfn:
    // dense gate/up ([N × intermDim]) + hidden ([N × embDim]); MoE routed gather/
    // scatter scratch ([N × expertDim] gate/up, [N × na × embDim] down partials, plus
    // per-expert gather buffers). Sized exactly to N by EnsureBatchedFfnScratch alongside
    // EnsureBatchedTrunkScratch. Null on the CPU-MoE / CPU-dense-fallback paths.
    private int _bfCap;
    private Tensor? _gpuBfGateAll;   // dense: [N × intermDim] gate proj
    private Tensor? _gpuBfUpAll;     // dense: [N × intermDim] up proj
    private Tensor? _gpuBfHiddenAll; // dense+MoE: [N × embDim] FFN output / shared-expert out
    private Tensor? _gpuBfMoeDownPartial; // MoE: [N × na × embDim] unweighted down partials (pre-reduce)
    private Tensor? _gpuBfMoeGateGathN;   // MoE: [N × expertDim] gathered gate (per used expert, ≤N rows)
    private Tensor? _gpuBfMoeUpGathN;      // MoE: [N × expertDim] gathered up
    private Tensor? _gpuBfMoeNormGathN;    // MoE: [N × embDim] gathered routed-token norms (per used expert)
    private Tensor? _gpuBfMoeDownGathN;    // MoE: [N × embDim] gathered down output (per used expert)
    private Tensor? _gpuBfMoeRouterAll;    // MoE: [N × numExperts] batched router logits
    // Issue #129: batched shared-expert + single-launch reduce scratch (GPU-SLRU MoE).
    private Tensor? _gpuBfShGateAll;       // MoE: [N × expertDim] batched shared-expert gate
    private Tensor? _gpuBfShUpAll;         // MoE: [N × expertDim] batched shared-expert up
    private Tensor? _gpuBfShexpScaleDev;   // MoE: [N] per-token shared-expert sigmoid gate (device)
    private Tensor? _gpuBfMoeWeightsDev;   // MoE: [N × na] top-k weights (device, for the reduce kernel)
    // Host bucket bookkeeping for the GPU-SLRU grouped-by-expert routed pass (issue #121).
    // Mirrors the CPU-MoE _bExpStart/_bExpTokI/… arrays but lives on the GPU-FFN path,
    // which never calls EnsureBatchedScratch. Selection / weights / shexp-gate are host
    // arrays (top-k picked on CPU after a router-logit readback, as in GpuMoeFfn).
    private int*   _bfSelected;     // [N × na] selected experts (token-major)
    private float* _bfWeights;      // [N × na] expert weights
    private float* _bfShexpScale;   // [N] shared-expert sigmoid gate
    private float* _bfRouterAll;    // [N × numExperts] router-logit readback
    private float* _bfNormReadback; // [N × embDim] norm readback for shexp-gate dots
    private int*   _bfExpStart;     // [numExperts+1] CSR offsets
    private int*   _bfExpCursor;    // [numExperts] fill cursor
    private int*   _bfExpTokI;      // [N × na] token index, grouped by expert
    private int*   _bfExpTokK;      // [N × na] slot (top-k rank), grouped by expert
    private int*   _bfUsedExperts;  // [numExperts] compact list of experts with ≥1 token
    private int*   _bfGathTokI;     // [N] gather list: row r of gather buffer ← token _bfGathTokI[r]

    // ── GPU op-offload for the CPU-MoE routed prefill (perf/carnice-vnni-moe) ──
    // OPT-IN, default OFF (SHARPI_MOE_GPU_PREFILL=1; CLI --gpu-moe-prefill true / server
    // GpuMoePrefill=true): instead of running BatchedRoutedExperts' grouped int8/F32 dots on
    // the CPU, transiently upload each used expert's host-resident gate/up/down weight to a
    // reused GPU buffer and run the gather → GEMM-N gate/up → SiLuMul → GEMM-N down on the GPU
    // (mirrors llama.cpp uploading CPU-resident MoE weights for the batched prefill matmul) —
    // measured +15-44% prefill on the GDN-hybrid CPU-MoE models. Raw-quant dtypes (Q3_K via
    // the #100 in-kernel-dequant GEMM, Q4_K/Q5_K/Q6_K/Q8_0) upload raw bytes and dispatch the
    // quantized GEMM; only Float32 weights dequant-stage. NOT bit-exact — argmax-stable (the
    // GPU runs the MoE in F32, *more* precise than the CPU int8 path).
    //   *** Default ON (#390). The original opt-in #387 path traded decode for prefill: a ~14 GB
    //   pinned cudaMallocHost copy duplicated the experts in RAM (mmap + pinned), and that copy
    //   could evict the page cache single-token DECODE's CPU expert-streaming relies on (the
    //   measured ~-25% decode blocker). #390 fixes it two ways: (1) the register-in-place pin mode
    //   (SHARPI_MOE_PIN_MODE=register, now default) cudaHostRegisters the expert mmap pages instead
    //   of copying — no 14 GB duplicate, no page-cache eviction (measured decode within noise of the
    //   CPU path, clean A/B); (2) a token gate (_gpuMoePrefillMinTokens) keeps tiny prefills + all
    //   decode on the byte-exact CPU path, so op-offload engages only where it wins (+28-67% prefill
    //   from ~120 tokens up). SHARPI_MOE_GPU_PREFILL=0 restores the pure CPU MoE prefill. ***
    // Falls back to the CPU path (the field is cleared) if the scratch can't be allocated —
    // see the ctor. Not readonly: the ctor clears it on a setup failure.
    private bool _gpuMoePrefill = ResolveGate("SHARPI_MOE_GPU_PREFILL", true);
    // #388: run the CPU-MoE prefill router matvec on the GPU (batched GEMM over the on-GPU
    // post-attn norm) instead of the per-token CPU matvec (~16% of CPU-MoE prefill). RAW logits
    // download; softmax + top-k stay on the host. Argmax-stable vs the CPU matvec (same FP class
    // as BatchedGpuMoeFfn's already-default GPU router). SHARPI_MOE_GPU_ROUTER=0 forces CPU.
    private readonly bool _gpuRouterPrefill = ResolveGate("SHARPI_MOE_GPU_ROUTER", true);
    // Set per layer by TrunkLayerBatched: true when it issued the GPU router GEMM+download for
    // this layer, so PrefillBatchedCpuMoe's router loop reads _btRouterAll instead of matvec'ing.
    private bool _btRouterGpuValid;
    // #390: op-offload only engages for prefill batches of at least this many tokens. The op-offload
    // uploads the WHOLE expert tensor (~14 GB) per layer regardless of N, so a tiny batch goes
    // upload-bound and loses to the CPU MoE (measured register mode: trails CPU below ~50 tokens,
    // wins +28-67% from ~120 up). Below the gate — and for ALL single-token decode / MTP-verify
    // steps, which never reach this batched-prefill path anyway — the byte-exact CPU MoE runs.
    // Env override SHARPI_MOE_GPU_PREFILL_MIN_TOKENS (0 = always engage when op-offload is on).
    private readonly int _gpuMoePrefillMinTokens = ResolveIntGate("SHARPI_MOE_GPU_PREFILL_MIN_TOKENS", 64);
    private int     _goCap;            // token capacity the GPU-offload scratch is sized for
    private Tensor? _gpuOffNorm;       // [N × embDim] uploaded norm activations (per layer call)
    private Tensor? _gpuOffGather;     // [totalSel × embDim] CSR-ordered gathered routed-token norms (ONE gather)
    private Tensor? _gpuOffGate;       // [totalSel × expertDim] gate projection (CSR-ordered, per-expert slices)
    private Tensor? _gpuOffUp;         // [totalSel × expertDim] up projection (CSR-ordered, per-expert slices)
    private Tensor? _gpuOffDownCsr;    // [totalSel × embDim] CSR-ordered down output (per-expert slices; ONE scatter)
    private Tensor? _gpuOffWGate;      // reused transient gate weight buffer (max raw/F32 bytes; Float32 fallback only)
    private Tensor? _gpuOffWUp;        // reused transient up weight buffer (Float32 fallback only)
    private Tensor? _gpuOffWDown;      // reused transient down weight buffer (Float32 fallback only)
    // Whole-layer raw-quant weight buffers: one big contiguous UploadRawInto of the WHOLE
    // layer's ffn_*_exps tensor (all numExperts experts back-to-back) per layer call,
    // then per-expert ViewRawBytes carves each expert's matrix out for the GEMM. Replaces
    // ~30k tiny per-expert uploads with 3 big transfers/layer (the upload floor). Sized to
    // the MAX raw layer bytes over all MoE layers so every dtype/dim fits. Raw-quant only
    // (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0); Float32 falls back to the per-expert _gpuOffW* buffers above.
    // Double-buffered (ping-pong) whole-layer raw-quant weight buffers. Slot 0 is the legacy
    // single buffer (used for the synchronous path and the first MoE layer of every chunk);
    // when the expert weights are in the pinned cudaMallocHost buffer (_goHostPinned), the NEXT
    // layer's weights are DMA'd into the OTHER slot via UploadRawIntoAsyncDirect while THIS
    // layer's GEMMs run, overlapping the upload with compute. Doubles layer-weight VRAM (~330→660 MB).
    private readonly Tensor?[] _gpuLayerGate = new Tensor?[2]; // [slot][numExperts × expertDim × bprMaxG]
    private readonly Tensor?[] _gpuLayerUp   = new Tensor?[2]; // [slot][numExperts × expertDim × bprMaxU]
    private readonly Tensor?[] _gpuLayerDown = new Tensor?[2]; // [slot][numExperts × embDim × bprMaxD]
    // Async double-buffer prefetch state (only used when _goHostPinned):
    //   _goHostPinned     — the expert weights were copied into a truly-pinned cudaMallocHost
    //                       buffer → DMA reads run at full PCIe bandwidth + overlap compute.
    //   _goPinAttempted   — the pinned-buffer alloc+copy ran for the current scratch (one-time guard).
    //   _goCurSlot        — slot holding THIS layer's weights (the one the GEMMs read).
    //   _goPrefetchedLayer— layer index whose weights were prefetched into the OTHER slot (-1 = none).
    //   _goPrefetchSlot   — the slot the prefetch landed in (becomes _goCurSlot when we consume it).
    //   _goPrefetch{Gate,Up,Down}H — in-flight DMA handles for the prefetched layer (waited then released).
    private bool _goHostPinned;
    private bool _goPinAttempted;
    // The N-independent static scratch (the ~14 GB pinned weight copy, the whole-layer GPU weight
    // buffers, the F32 dequant staging) is allocated ONCE and survives a dynamic per-N regrow —
    // re-copying the pinned buffer on every batch-size growth caused multi-second spikes +
    // fragmentation risk (Gemini review). Only the per-N gather/scatter/GEMM scratch re-grows.
    private bool _goStaticAllocated;
    // Truly-pinned (cudaMallocHost) host copy of all MoE expert weights — the DMA source. A
    // file-backed mmap registered via cudaHostRegister overlaps but tops out at ~13 GB/s; a
    // genuine pinned allocation reaches full PCIe (~26 GB/s). Copied once at first scratch
    // alloc (~1s, pages already pre-faulted) and freed in FreeGpuOffloadScratch/Dispose.
    private nint    _goPinnedBuf;         // base of the big cudaMallocHost buffer (Zero = not pinned)
    // In register mode (#390) _goPinnedBuf is nint.Zero and these per-layer pointers alias the
    // mmap DataPtrs directly; in copy mode they point inside _goPinnedBuf.
    private byte*[]? _goPinnedGate;       // [L] per-layer gate_exps base (mmap DataPtr in register mode, inside _goPinnedBuf in copy mode)
    private byte*[]? _goPinnedUp;         // [L] per-layer up_exps base
    private byte*[]? _goPinnedDown;       // [L] per-layer down_exps base
    // #390 register-in-place mode: the page-aligned mmap ranges registered via cudaHostRegister
    // (no owned buffer). Tracked so FreeGpuOffloadScratch can cudaHostUnregister them. Null in
    // copy mode (which owns _goPinnedBuf instead). The two modes are mutually exclusive.
    private nint[]? _goRegisteredRanges;
    private int  _goCurSlot;
    private int  _goPrefetchedLayer = -1;
    private int  _goPrefetchSlot;
    private CudaUploadHandle _goPrefetchGateH;
    private CudaUploadHandle _goPrefetchUpH;
    private CudaUploadHandle _goPrefetchDownH;
    private bool _goPrefetchValid;   // a prefetch DMA is in flight and its handles are live
    private float*  _hGpuOffDeq;       // [expertDim·embDim] host F32 staging for Q3_K/F32 dequant
    private float*  _hGpuOffDownDl;    // [≤N·na × embDim] pinned host download buffer (final routed result)
    private Tensor? _gpuOffDownPartial; // [N × na × embDim] device unweighted down partials (GPU scatter target)
    private Tensor? _gpuOffRouted;     // [N × embDim] device weighted-reduce output (downloaded ONCE)
    private Tensor? _gpuOffWeightsDev; // [N × na] device top-k weights for the reduce kernel
    private Tensor? _gpuOffGatherIdx;  // [N·na] int32 device — CSR gather row indices (= expTokI[p])
    private Tensor? _gpuOffScatterIdx; // [N·na] int32 device — CSR scatter slot indices (= expTokI[p]*na + expTokK[p])
    private int*    _hGpuOffScatterIdx; // [N·na] host int32 staging for the scatter index upload
    // GPU-offload per-phase profiling (printed once per chunk under SHARPI_PREFILL_PROFILE).
    private long _profGoDequant, _profGoUpload, _profGoGather, _profGoGemm, _profGoDownScatter;

    // Issue #121: batch the per-token FFN/MoE stage of PrefillBatchedTrunkGpuFfn into
    // GEMM-N launches (dense gate/up/down over N) and a grouped-by-expert routed-MoE
    // pass (each cached expert loaded once, matmul'd against all its tokens), with a
    // per-token top-k-ordered reduce that keeps byte-parity with the sequential
    // per-token loop. Default on; SHARPI_BATCHED_FFN=0 forces the per-token FFN path
    // (the bit-exact reference). A/B-toggleable like SHARPI_BATCHED_TRUNK.
    internal static bool BatchedFfnEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_FFN") != "0";

    // Issue #114-B: fuse the per-position GDN recurrence launches into one
    // sequential-scan kernel + batched conv1d/L2norm/tile/silu over N. Default on;
    // SHARPI_BATCHED_GDN_SCAN=0 falls back to the per-position View loop inside
    // GdnBlockBatched (the pre-#114-B path). A/B-toggleable like SHARPI_BATCHED_TRUNK.
    internal static bool BatchedGdnScanEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_GDN_SCAN") != "0";

    // FlashQLA chunked GDN prefill (issue #211 follow-up): inside the batched-prefill
    // fast path, resolve the GDN recurrence with the chunk-parallel
    // chunk_gated_delta_rule kernel (CudaBackend.GdnChunkedPrefill) instead of the
    // sequential GdnRecurrenceScan. The chunked form is numerically equal to the scan
    // only up to FP reduction order (NOT byte-exact). It only engages on the prefill
    // (clean state-carry) path: GdnBlockBatched short-circuits to the byte-exact
    // ring-capturing scan whenever snapRing is set (if (snapRing) … else if (chunked) …),
    // so decode and batched-verify ring capture always stay on the scan.
    //   #388: this is now AUTO-ON when the GPU MoE op-offload is active (_gpuMoePrefill) —
    //   that prefill path is already argmax-stable (the routed MoE runs in F32/int8, not
    //   the CPU's byte-exact dots), so the chunked GDN's FP-reorder is consistent and free
    //   (measured ~+8% Carnice prefill). When op-offload is OFF (byte-exact CPU MoE) the
    //   scan is kept. Tri-state SHARPI_GDN_CHUNKED_PREFILL: "1" forces chunked even with
    //   op-offload off, "0" forces the byte-exact scan, unset = auto. The test sets the
    //   override directly. Mirrors HybridGdnForwardPass's chunked gate (which excludes MTP).
    internal static bool? GdnChunkedPrefillOverride =
        Environment.GetEnvironmentVariable("SHARPI_GDN_CHUNKED_PREFILL") switch
        { "1" => true, "0" => false, _ => null };

    // Issue #114-B: batch the per-position KV-append + SDPA into one launch each
    // (CudaBackend.AttentionBatched). Only used when the chunk stays on the
    // shared-scores path (startPos+N ≤ 4096) and SnapKV Q-capture is inactive;
    // otherwise AttnBlockBatched keeps the per-position loop. Default on;
    // SHARPI_BATCHED_ATTN=0 forces the per-position path.
    internal static bool BatchedAttnEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_ATTN") != "0";

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public LayerPlacement Placement => _placement;

    /// <summary>
    /// Bind this pass's CUDA context to the calling thread (issue #302). The engine calls it on
    /// the worker thread that drives the forward pass before any CUDA work, so a non-interactive
    /// session doesn't hang on the first unbound-thread CUDA call.
    /// </summary>
    public void BindToCurrentThread() => _gpu.BindContextToCurrentThread();

    /// <summary>
    /// Host-side bookkeeping cache. Holds slot/length state only — the actual K/V
    /// payload for attention layers lives on the GPU in <c>_gpuKCache</c> /
    /// <c>_gpuVCache</c>. Exposed so tests can assert SnapKV (issue #58)
    /// post-eviction invariants (<c>Length</c> == budget, <c>LogicalLength</c> ==
    /// original prompt length).
    /// </summary>
    public PagedKvCache Cache => _kvCache;

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

        // #388: per-layer decode-trunk graph readiness, only when the gate is on.
        _layerGraphCaptured = _decodeCudaGraph ? new bool[L] : null;

        // SHARPI_KV_DTYPE: fp32 | bf16 (default bf16). Issue #27.
        _kvDType = ParseKvDType(Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE"));

        // SnapKV (issue #58) — gated by SHARPI_SNAPKV_BUDGET. Buffers are
        // lazily allocated on the first active prefill in Prefill().
        _snapKvCfg = SnapKvConfig.FromEnvironment();
        _attnLayerIndexOf = new int[L];
        int numAttn = 0;
        for (int i = 0; i < L; i++)
        {
            if (hp.LayerTypes![i] == LayerType.Attention)
                _attnLayerIndexOf[i] = numAttn++;
            else
                _attnLayerIndexOf[i] = -1;
        }
        _numAttnLayers = numAttn;
        _snapKvCaptureSlot = -1;

        // Resolve the effective SnapKV budget (issue #58 follow-up):
        //   * env explicitly set        → use that value verbatim (0 = disabled)
        //   * env unset + attn layers   → maxSeqLen/4 cache-sized auto-budget
        //   * no attention layers       → 0 (nothing to evict)
        // The auto path uses the full-cache byte size as its threshold, so big-
        // context setups (where the cache is the dominant VRAM tenant — the
        // 12 GB target) trip the default on, while small-context smoke tests
        // stay untouched. Decided once here for stable per-request behaviour.
        if (_snapKvCfg.IsBudgetExplicit || _numAttnLayers == 0)
        {
            _snapKvEffectiveBudget = _snapKvCfg.Budget;
        }
        else
        {
            int kvElemBytes = DTypeInfo.BytesPerElement(_kvDType);
            long fullCacheBytes =
                (long)_numAttnLayers * _maxSeqLen * kvDim * 2 /*K+V*/ * kvElemBytes;
            _snapKvEffectiveBudget = SnapKvConfig.ComputeAutoBudget(_maxSeqLen, fullCacheBytes);
            if (_snapKvEffectiveBudget > 0)
                Console.Error.WriteLine(
                    $"[CudaHybridGdnForwardPass] SnapKV auto-enabled: budget={_snapKvEffectiveBudget}, "
                    + $"window={_snapKvCfg.Window}, recency={_snapKvCfg.Recency} "
                    + $"(set SHARPI_SNAPKV_BUDGET=0 to disable)");
        }

        Console.Error.WriteLine($"[CudaHybridGdnForwardPass] layers={L} embDim={_embDim} headDim={_headDim} numHeads={_numHeads} ropeDim={_ropeDim} ctx={_ctxLen} kvDType={_kvDType}");
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
        // Flash-decoding split-KV partials (#238) — allocate when ctx is long enough and within the
        // combine kernel's split bound. Uniform head_dim (no per-layer), so size at _headDim.
        if (_splitDecodeEnabled
            && _maxSeqLen > GdnSplitMinSeq && _maxSeqLen <= CudaForwardPass.SplitKvMaxCtx)
        {
            long nSplitsMax = (_maxSeqLen + CudaBackend.SplitKvChunk - 1) / CudaBackend.SplitKvChunk;
            _splitKvPartialO = gpu.Allocate(TensorShape.D1((long)_numHeads * nSplitsMax * _headDim));
            _splitKvPartialMeta = gpu.Allocate(TensorShape.D1((long)_numHeads * nSplitsMax * 2));
        }
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
        // _cpuNormBuf is the per-token GDN-pre-MoE-norm download target. Pinned
        // (cudaMallocHost) so CudaBackend's direct-pinned Download overload can
        // DMA into it without bouncing through the internal staging buffer
        // (issue #48). ~8 KiB per token; safe to pin.
        _cpuNormBuf = AllocPinned(_embDim);
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
            _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embDType);
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

        // Op-offload (GPU offload of the routed-MoE prefill, #390) only applies to CPU-MoE
        // models — there are no CPU-resident routed experts to offload on the dense or
        // experts-on-GPU (SLRU) paths. Clamp the gate to _cpuMoe so its eager scratch/pin
        // setup, per-call dispatch, AND the chunked-GDN auto-enable (which keys off
        // _gpuMoePrefill) never engage for those models. Without this, the #390 default-on
        // flip ran the eager GPU-scratch alloc for the dense 27B-MTP, perturbing cuBLAS
        // workspace/algo selection enough to break the dense batched-trunk bitwise-parity tests.
        _gpuMoePrefill &= _cpuMoe;

        // Resolve Q3_K_Q8K / Q8_0_Q8K kernel gates. Auto-on when the model has
        // routed-expert weights in that dtype (APEX mixed-precision tier — e.g.
        // Carnice). SHARPI_Q3K_Q8K / SHARPI_Q8_0_Q8K = "1" or "0" override.
        bool hasQ3KRouted  = HasRoutedExpertsOfDType(model, hp, DType.Q3_K);
        bool hasQ8_0Routed = HasRoutedExpertsOfDType(model, hp, DType.Q8_0);
        bool hasQ4KRouted  = HasRoutedExpertsOfDType(model, hp, DType.Q4_K);
        _q3kQ8KEnabled  = ResolveGate("SHARPI_Q3K_Q8K",  hasQ3KRouted);
        _q8_0Q8KEnabled = ResolveGate("SHARPI_Q8_0_Q8K", hasQ8_0Routed);
        // Q4_K int8 (DotQ4K_Q8KS) is AVX2-only, but the f32 DotQ4K has an AVX-512 path
        // (DotQ4K_Avx512). On AVX-512 hardware the f32-AVX512 dot + no activation-quant
        // overhead BEATS the int8-AVX2 dot — measured ~8% faster on a Q4_K_M model (Zen4).
        // So only auto-enable Q4_K int8 where AVX-512 is absent (there f32 falls to AVX2 and
        // the int8 dot can win); on AVX-512 default OFF. Still forceable via SHARPI_Q4K_Q8K=1.
        // (Q3_K/Q8_0 int8 have no f32-AVX512 competitor, so they stay auto-on.)
        _q4kQ8KEnabled  = ResolveGate("SHARPI_Q4K_Q8K",
            hasQ4KRouted && !System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated);
        if (_cpuMoe && (_q3kQ8KEnabled || _q8_0Q8KEnabled || _q4kQ8KEnabled))
        {
            var enabled = new List<string>(3);
            if (_q3kQ8KEnabled)  enabled.Add($"Q3_K_Q8K (Q3_K routed: {hasQ3KRouted})");
            if (_q8_0Q8KEnabled) enabled.Add($"Q8_0_Q8K (Q8_0 routed: {hasQ8_0Routed})");
            if (_q4kQ8KEnabled)  enabled.Add($"Q4_K_Q8K (Q4_K routed: {hasQ4KRouted})");
            Console.Error.WriteLine(
                $"[CudaHybridGdnForwardPass] Routed-MoE Q8_K-input kernels enabled: {string.Join(", ", enabled)}. Override with SHARPI_Q3K_Q8K=0 / SHARPI_Q8_0_Q8K=0 / SHARPI_Q4K_Q8K=0.");
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
            if (_q3kQ8KEnabled || _q8_0Q8KEnabled || _q4kQ8KEnabled)
            {
                // Q8_KS layout (per-32-element sub-block scales) closes the
                // parity gap that #103 surfaced — see DotQ3K_Q8KS / #107.
                _cpuExpertGateAllQ8KStride = SimdKernels.Q8KSScratchBytes(_expertDim);
                _cpuNormInQ8K = (byte*)NativeMemory.Alloc((nuint)SimdKernels.Q8KSScratchBytes(_embDim));
                _cpuExpertGateAllQ8K = (byte*)NativeMemory.Alloc(
                    (nuint)(_numActiveExperts * _cpuExpertGateAllQ8KStride));
            }
            else
            {
                _cpuNormInQ8K = null;
                _cpuExpertGateAllQ8K = null;
                _cpuExpertGateAllQ8KStride = 0;
            }
            // Pinned: source of the per-token UploadInto back to _gpuHidden after
            // the CPU MoE FFN runs (issue #48). ~8 KiB; safe to pin.
            _cpuMoeHidden = AllocPinned(_embDim);
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
            // Pinned: per-layer UploadInto source after CPU dense FFN (issue #48).
            _cpuMoeHidden  = AllocPinned(_embDim);

            _cpuRouterLogits = null;
            _cpuSharedOut = null;
            _cpuExpertGateAll = null;
            _cpuExpertUpAll = null;
            _cpuNormInQ8K = null;
            _cpuExpertGateAllQ8K = null;
            _cpuExpertGateAllQ8KStride = 0;
        }
        else
        {
            _cpuRouterLogits = null;
            _cpuSharedOut = null;
            _cpuExpertGateAll = null;
            _cpuExpertUpAll = null;
            _cpuNormInQ8K = null;
            _cpuExpertGateAllQ8K = null;
            _cpuExpertGateAllQ8KStride = 0;
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
                    // #388: also upload the small (F32, ~2 MiB/layer) router weight to GPU so the
                    // prefill router matvec can run on-GPU (SHARPI_MOE_GPU_ROUTER); the routed
                    // expert weights still stream from CPU mmap. _cpuFfnGateInp stays resolved for
                    // the CPU-router fallback (SHARPI_MOE_GPU_ROUTER=0 / sequential trunk).
                    if (_gpuRouterPrefill)
                        _gpuWGateInp[i] = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
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

                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), _kvDType);
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), _kvDType);
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
        // ── MTP detection + GDN snapshot ring reservation (issues #30/#207) ──
        // Decided HERE (not at the head-upload block below) because the ring must
        // be carved out of VRAM BEFORE TryUploadDenseFfnLayers greedily fills it
        // to a 64 MiB margin — a later allocation would land in WDDM-paged memory
        // and 5-10× every verify phase. SHARPI_DISABLE_MTP=1 skips the ring so
        // MTP-off baseline runs keep the VRAM for FFN layers.
        _hasMtp = hp.NumMtpLayers > 0
                  && model.FindTensor($"blk.{hp.NumLayers}.nextn.eh_proj.weight") is not null;
        if (_hasMtp && !_cpuGdn && _gdnStateCache.NumGdnLayers > 0
            && Environment.GetEnvironmentVariable("SHARPI_DISABLE_MTP") != "1")
        {
            int numGdn = _gdnStateCache.NumGdnLayers;
            int scanF = _gdnStateCache.ScanStateFloatsPerLayer;
            int convF = _gdnStateCache.ConvStateFloatsPerLayer;
            int want = _mtpBatchMax - 1;
            // #290: one flat contiguous tensor per (scan, conv) sized for `got`
            // slots, so the fused scan/conv-capture kernels can stride across slots.
            // A single allocation can't partially succeed, so retry with fewer slots
            // on OOM — total footprint matches the old slot-by-slot sum, preserving
            // the graceful-degradation semantics.
            Tensor? scanFlat = null, convFlat = null;
            int got = 0;
            for (int trySlots = want; trySlots >= 1; trySlots--)
            {
                Tensor? s = null, c = null;
                try
                {
                    s = gpu.Allocate(TensorShape.D1((long)trySlots * numGdn * scanF));
                    if (convF > 0)
                        c = gpu.Allocate(TensorShape.D1((long)trySlots * numGdn * convF));
                    scanFlat = s;
                    convFlat = c;
                    got = trySlots;
                    break;
                }
                catch (Exception ex)
                {
                    // Free any partial allocation (e.g. scan succeeded but conv threw)
                    // before retrying with fewer slots.
                    if (s is { } ps) gpu.Free(ps);
                    if (c is { } pc) gpu.Free(pc);
                    Console.Error.WriteLine(
                        $"[CudaHybridGdnForwardPass] GDN ring allocation for {trySlots} slot(s) failed " +
                        $"({ex.GetType().Name}); retrying with fewer.");
                }
            }
            _gpuGdnRingScan = scanFlat;
            _gpuGdnRingConv = convFlat;
            _gdnRingSlots = got;
            long slotBytes = (long)numGdn * (scanF + convF) * sizeof(float);
            Console.Error.WriteLine(
                $"[CudaHybridGdnForwardPass] MTP batched-verify GDN ring: {got} slot(s) × " +
                $"{slotBytes / (1024 * 1024)} MiB → max verify batch {got + 1} tokens.");
        }

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
        // out of scope for v1; only the first head is loaded. (_hasMtp itself is
        // decided earlier, before the dense-FFN VRAM fill, so the batched-verify
        // GDN ring could be reserved first.)
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

            // MoE-MTP vs dense-MTP probe (issue #44). Mirrors HybridGdnForwardPass.
            // For MoE MTP we require CPU MoE mode — the routed-expert stack at the
            // MTP block (~470 MiB Q4_K) won't co-reside with the trunk experts on a
            // 12 GB GPU, and the GPU MoE SLRU isn't sized for the extra layer.
            _mtpIsMoE = model.FindTensor($"blk.{mtpLayerIdx}.ffn_gate_exps.weight") is not null;
            if (_mtpIsMoE && !hp.IsMoE)
                throw new NotSupportedException(
                    "MoE MTP head requires trunk MoE (NumExperts/ExpertIntermediateDim from hyperparams). " +
                    "Dense-trunk + MoE-MTP-head is not a configuration we've seen.");
            if (_mtpIsMoE && !_cpuMoe)
                throw new NotSupportedException(
                    "MoE MTP head requires SHARPI_CPU_MOE=1. GPU MoE path (SLRU expert cache) " +
                    "doesn't reserve slots for the MTP block; enable CPU MoE mode to load this model.");

            if (_mtpIsMoE)
            {
                _gpuMtpWGateShexp   = UploadWeight($"blk.{mtpLayerIdx}.ffn_gate_shexp.weight");
                _gpuMtpWUpShexp     = UploadWeight($"blk.{mtpLayerIdx}.ffn_up_shexp.weight");
                _gpuMtpWDownShexp   = UploadWeight($"blk.{mtpLayerIdx}.ffn_down_shexp.weight");
                _cpuMtpFfnGateInp   = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_gate_inp.weight");
                _cpuMtpFfnGateExps  = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_gate_exps.weight");
                _cpuMtpFfnUpExps    = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_up_exps.weight");
                _cpuMtpFfnDownExps  = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_down_exps.weight");
                _cpuMtpFfnGateInpShexp = LoadF32Tensor($"blk.{mtpLayerIdx}.ffn_gate_inp_shexp.weight", _embDim);
                _gpuMtpFfnGate = _gpuMtpFfnUp = _gpuMtpFfnDown = null!;
            }
            else
            {
                _gpuMtpFfnGate    = UploadWeight($"blk.{mtpLayerIdx}.ffn_gate.weight");
                _gpuMtpFfnUp      = UploadWeight($"blk.{mtpLayerIdx}.ffn_up.weight");
                _gpuMtpFfnDown    = UploadWeight($"blk.{mtpLayerIdx}.ffn_down.weight");
                _gpuMtpWGateShexp = _gpuMtpWUpShexp = _gpuMtpWDownShexp = null!;
            }

            _gpuMtpEnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.enorm.weight");
            _gpuMtpHnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.hnorm.weight");
            _gpuMtpSharedHeadNorm = UploadWeight($"blk.{mtpLayerIdx}.nextn.shared_head_norm.weight");
            // eh_proj is Q8_0 in GGUF; UploadWeight dequants to F32 on the path
            // for any dtype not in {F32, Q4_K, Q5_K, Q6_K}, so this lands as F32
            // and the CudaBackend.MatMul fp32 path serves it. ~200 MiB residence.
            _gpuMtpEhProj         = UploadWeight($"blk.{mtpLayerIdx}.nextn.eh_proj.weight");

            // MTP attention KV cache on GPU (one slot; same layout as trunk KV).
            int mtpKvDim = _numKvHeads * _headDim;
            _gpuMtpKCache = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * mtpKvDim), _kvDType);
            _gpuMtpVCache = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * mtpKvDim), _kvDType);
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
            // Pinned (cudaMallocHost) so the snapshot D2H can be queued via
            // DownloadAsync and drained by the subsequent logits Download's
            // stream sync. This lets the lm_head MatMul launch in parallel with
            // the queued PCIe transfer (issue #49).
            _lastHidden = AllocPinned(_embDim);

            // MTP self-chaining hidden (issue #30): the MTP block's residual
            // output, captured in MtpForward before the in-place shared-head
            // norm. Pinned for the same queued-D2H reason as _lastHidden.
            _mtpSelfHidden = AllocPinned(_embDim);

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

            // Issue #30 / #45 batched-verify scratch. Token 2 gets its own residual
            // stream + norm + logits + a token-1 hidden snapshot for the MTP commit
            // step. Allocated for all MTP-bearing models (dense or MoE). The dense
            // intermediate buffers (_gpuFfnGateBufDense2 etc.) are MoE-skip; on the
            // MoE path the routed FFN runs sequentially per token via CpuMoeFfn,
            // which reuses _cpuExpertGateAll / _cpuExpertUpAll within each call.
            _gpuHidden2      = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuResidual2    = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuNormBuf2     = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuLogits2      = gpu.Allocate(TensorShape.D1(_hp.VocabSize));
            _gpuLastHiddenT1 = gpu.Allocate(TensorShape.D1(_embDim));
            _logitsBuf2      = new float[_hp.VocabSize];
            // Pinned: per-token batched-verify Download/UploadInto scratch
            // (issues #48/#49). _lastHiddenT1 is queued via DownloadAsync and
            // drained by the subsequent logits Download's sync, mirroring the
            // single-token path's overlap.
            _cpuNormBuf2     = AllocPinned(_embDim);
            _cpuMoeHidden2   = AllocPinned(_embDim);
            _lastHiddenT1    = AllocPinned(_embDim);
            if (!hp.IsMoE)
            {
                _gpuFfnGateBufDense2 = gpu.Allocate(TensorShape.D1(_intermDim));
                _gpuFfnUpBufDense2   = gpu.Allocate(TensorShape.D1(_intermDim));
                _cpuFfnGateBuf2  = Alloc(_intermDim);
                _cpuFfnUpBuf2    = Alloc(_intermDim);
                _cpuFfnGateBuf3  = Alloc(_intermDim);
                _cpuFfnUpBuf3    = Alloc(_intermDim);
                _cpuFfnGateBuf4  = Alloc(_intermDim);
                _cpuFfnUpBuf4    = Alloc(_intermDim);
            }

            // Host snapshot buffer for BatchForward2's between-token capture — only
            // the SHARPI_CPU_GDN=1 debug trunk uses it (the default GPU trunk's
            // batched-verify snapshots live in the device ring); skip the ~150 MB
            // host allocation otherwise.
            long perLayerBytes = _gdnStateCache.LayerSnapshotBytes;
            long totalSnapBytes = perLayerBytes * _gdnStateCache.NumGdnLayers;
            if (totalSnapBytes > 0 && _cpuGdn)
            {
                _batchSnapshotBuf = (byte*)NativeMemory.Alloc((nuint)totalSnapBytes);
                _batchSnapshotCap = totalSnapBytes;
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
                _gpuMtpWGateShexp = _gpuMtpWUpShexp = _gpuMtpWDownShexp =
                _gpuMtpEnorm = _gpuMtpHnorm = _gpuMtpSharedHeadNorm =
                _gpuMtpEhProj =
                _gpuMtpEmbedBuf = _gpuMtpEnormBuf = _gpuMtpHnormBuf =
                _gpuMtpConcatBuf = _gpuLastHidden = null!;
            _lastHidden = null;

            _gpuHidden2 = _gpuResidual2 = _gpuNormBuf2 =
                _gpuLogits2 = _gpuLastHiddenT1 = null!;
            _logitsBuf2 = Array.Empty<float>();
            _cpuNormBuf2 = _cpuMoeHidden2 = _lastHiddenT1 = null;
        }

        // Pre-fault CPU-resident mmap weight pages (issue #221). On the CPU-MoE config
        // (the auto-selected winner on 12 GB) the routed experts / dense FFN weights are
        // paged in lazily; without this the first request faults them all on the critical
        // path, ~5× slower than warm. MmapPrefault honours SHARPI_PREFAULT and the
        // RAM-fit heuristic, and no-ops when nothing is CPU-resident (full-GPU GDN).
        MmapPrefault.Run("CudaHybridGdnForwardPass", BuildCpuPrefaultRegions());

        // Op-offload (SHARPI_MOE_GPU_PREFILL): build ALL op-offload scratch at LOAD — the
        // ~14 GB pinned cudaMallocHost + copy AND the GPU gather/scatter/layer buffers —
        // not lazily on the first prefill, otherwise the one-time setup lands on the first
        // request's critical path (tanking its TTFT and polluting single-turn benchmarks).
        // EnsureGpuOffloadScratch is grow-only, so a larger chunk later just re-grows the
        // (small) GPU buffers; the dominant pinned-buffer cost is N-independent and done here.
        // This eager setup is independent of _gpuMoePrefillMinTokens by design: that gate only
        // affects whether a given per-call dispatch uses op-offload, not whether the scratch /
        // pin is built at load.
        if (_gpuMoePrefill)
        {
            int warmChunk = int.TryParse(Environment.GetEnvironmentVariable("SHARPI_PREFILL_CHUNK"),
                out int pc) && pc > 0 ? pc : 512;
            // Safety fallback (op-offload is opt-in but can still hit allocation limits): if the
            // scratch can't be allocated (low host RAM for the ~14 GB pinned buffer, or tight
            // VRAM for the GPU gather/scatter/layer buffers), disable op-offload and run the CPU
            // MoE prefill rather than failing model load. The pinned-buffer alloc already
            // self-falls-back to synchronous upload; this catches the harder allocation failures.
            try
            {
                EnsureGpuOffloadScratch(warmChunk);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"[CudaHybridGdnForwardPass] GPU op-offload setup failed ({ex.GetType().Name}: {ex.Message}); " +
                    "falling back to the CPU MoE prefill. Set SHARPI_MOE_GPU_PREFILL=0 to silence.");
                FreeGpuOffloadScratch();
                _gpuMoePrefill = false;
            }
        }
    }

    // =================================================================
    //  IForwardPass surface
    // =================================================================

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ThrowIfFaulted();
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        int N = tokens.Count;

        // Size the hidden buffer up front so Forward's per-step writes don't
        // each trigger a grow.
        if (_hasMtp)
            EnsureMtpHiddenHistoryCap(startPos + N);

        // SnapKV (issue #58) gating: only run eviction when this is a fresh
        // prefill (startPos==0), the effective budget is positive (env-set or
        // VRAM-scaled auto), and the prompt is long enough that eviction would
        // drop something. _snapKvEffectiveBudget already encodes the "user set
        // SHARPI_SNAPKV_BUDGET=0 to disable" intent (value is 0 in that case).
        bool snapKvActive = _snapKvEffectiveBudget > 0
                         && startPos == 0
                         && N > _snapKvEffectiveBudget
                         && N > _snapKvCfg.Window
                         && _numAttnLayers > 0;
        int W = 0, wStart = 0;
        if (snapKvActive)
        {
            W = Math.Min(_snapKvCfg.Window, N);
            wStart = N - W;
            EnsureSnapKvCaptureBuffer(W);
        }

        // Issue #110: batched prompt prefill for the CPU-MoE path. Amortises the
        // DRAM-bound routed-expert mmap reads across all prompt tokens. Falls back
        // to the sequential loop below for single tokens, non-CPU-MoE configs, or
        // when explicitly disabled.
        //
        // Two correctness guards gate the fast path (both fall back to the
        // sequential loop, which has neither limitation):
        //   • Length == startPos on both caches: the batched trunk writes the
        //     attention KV at explicit slot `startPos + i` and advances the GDN
        //     recurrence in token order, which is only equivalent to the
        //     sequential `_kvCache.Length`-driven append when the caches sit
        //     exactly at startPos. After a SnapKV compaction the physical KV
        //     length diverges from the logical RoPE frame (kvPosition != position),
        //     so the explicit-slot assumption breaks — defer to sequential.
        //   • int-safe element counts: BatchedRoutedExperts indexes its scratch
        //     with `int` element counts (e.g. SiLuMul over N×numActive×expertDim);
        //     guard against silent truncation when chunking is disabled and the
        //     chunk is enormous (SHARPI_PREFILL_CHUNK set very large).
        bool trunkBatchSafe = N >= 2
            && _kvCache.Length == startPos
            && _gdnStateCache.Length == startPos;
        bool cpuMoeBatchSafe = trunkBatchSafe
            && (long)N * _numActiveExperts * _expertDim <= int.MaxValue;
        if (BatchedPrefillEnabled && _cpuMoe && cpuMoeBatchSafe)
        {
            ReadOnlySpan<float> bLogits = PrefillBatchedCpuMoe(tokens, startPos, snapKvActive, W, wStart);
            _snapKvCaptureSlot = -1;
            if (snapKvActive)
                ApplySnapKvEviction(N, W, wStart);
            return bLogits;
        }

        // Issue #119: extend the batched trunk to the dense GDN-hybrid (!IsMoE) and
        // GPU-SLRU MoE (_cpuMoe==false) configs. The trunk batches identically; only
        // the FFN/MoE stage differs (per-token GPU/CPU FFN or GPU-SLRU routed experts).
        // Gated on the GPU-GDN trunk (the batched kernels require it) and the trunk-batch
        // toggle (the whole value here is the batched trunk; SHARPI_BATCHED_TRUNK=0 falls
        // through to the sequential per-token loop below, the parity reference). The
        // `!_cpuMoe` clause is load-bearing: a CPU-MoE chunk that tripped the int-overflow
        // guard above (cpuMoeBatchSafe false, trunkBatchSafe true) must NOT land here —
        // PrefillBatchedTrunkGpuFfn dispatches GpuMoeFfn, whose SLRU manager is null on the
        // CPU-MoE path. It falls through to the sequential per-token loop instead.
        if (BatchedPrefillEnabled && BatchedTrunkEnabled && !_cpuGdn && !_cpuMoe && trunkBatchSafe)
        {
            ReadOnlySpan<float> gLogits = PrefillBatchedTrunkGpuFfn(tokens, startPos, snapKvActive, W, wStart);
            _snapKvCaptureSlot = -1;
            if (snapKvActive)
                ApplySnapKvEviction(N, W, wStart);
            return gLogits;
        }

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < N; i++)
        {
            // Drive Q-capture for the last W tokens — GpuAttnBlockAt reads
            // _snapKvCaptureSlot and writes _gpuQ into _snapKvQCapture.
            _snapKvCaptureSlot = (snapKvActive && i >= wStart) ? (i - wStart) : -1;

            logits = Forward(tokens[i], startPos + i);
        }
        _snapKvCaptureSlot = -1;

        if (snapKvActive)
        {
            ApplySnapKvEviction(N, W, wStart);
        }

        if (_prefillProfile && _profTokens > 0)
            DumpPrefillProfile();

        return logits;
    }

    private void DumpPrefillProfile()
    {
        double totalMs = _profTotalTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        double moeMs   = _profMoeTicks   * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        Console.Error.WriteLine(
            $"[prefill-profile] tokens={_profTokens} total={totalMs:F0}ms " +
            $"({totalMs / _profTokens:F1}ms/tok) cpuMoe={moeMs:F0}ms ({100.0 * moeMs / totalMs:F0}%) " +
            $"trunk+other={totalMs - moeMs:F0}ms ({100.0 * (totalMs - moeMs) / totalMs:F0}%)");
        _profMoeTicks = _profTotalTicks = 0;
        _profTokens = 0;
    }

    // =================================================================
    //  Batched prefill (issue #110)
    // =================================================================

    /// <summary>
    /// Exact-size (re)allocation of the inter-layer residual-stream device buffer
    /// <see cref="_gpuStreamAll"/> for N tokens. <c>UploadInto</c> requires a whole-tensor
    /// element-count match, so chunks of differing length reallocate (at most twice per
    /// prompt: a full chunk then a remainder). Shared by the CPU-MoE
    /// (<see cref="EnsureBatchedScratch"/>) and dense/GPU-SLRU
    /// (<see cref="PrefillBatchedTrunkGpuFfn"/>, issue #119) batched-prefill paths.
    /// </summary>
    private void EnsureStreamAll(int N)
    {
        int embDim = _embDim;
        if (_gpuStreamAll is not { } gs || gs.ElementCount != (long)N * embDim)
        {
            if (_gpuStreamAll is { } old) { _gpu.Free(old); _gpuStreamAll = null; }
            _gpuStreamAll = _gpu.Allocate(TensorShape.D1((long)N * embDim));
        }
    }

    /// <summary>
    /// Grow-only allocation of the per-chunk CPU-MoE batched-prefill scratch, sized for
    /// <paramref name="N"/> tokens (calls <see cref="EnsureStreamAll"/> first). Host buffers
    /// fed to <c>Download</c>/<c>UploadInto</c> are pinned (cudaMallocHost); the routed-expert
    /// compute buffers and the expert→token bucket arrays are plain native memory.
    /// </summary>
    private void EnsureBatchedScratch(int N)
    {
        int embDim = _embDim;
        EnsureStreamAll(N);

        if (N <= _bCap) return;

        FreeBatchedHostScratch();

        int na = _numActiveExperts;
        long perTokEmb = (long)N * embDim;
        long perTokSel = (long)N * na;

        _bResidAll  = AllocPinnedL(perTokEmb);
        _bNormAll   = AllocPinnedL(perTokEmb);
        if (_cpuMoe && _gpuRouterPrefill)
            _btRouterAll = AllocPinnedL((long)N * _numExperts);   // #388: pinned for fast D2H of the GPU router logits
        _bSharedAll = AllocPinnedL(perTokEmb);
        _bHiddenAll = AllocPinnedL(perTokEmb);
        _bRoutedAll = AllocL(perTokEmb);
        _bGateAll   = AllocL(perTokSel * _expertDim);
        _bUpAll     = AllocL(perTokSel * _expertDim);
        _bDownPartial = AllocL(perTokSel * embDim);

        _bSelected   = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));
        _bWeights    = AllocL(perTokSel);
        _bShexpScale = AllocL(N);
        _bExpTokI    = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));
        _bExpTokK    = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));

        if (_q3kQ8KEnabled || _q8_0Q8KEnabled || _q4kQ8KEnabled)
        {
            _bQ8KEmbStride = SimdKernels.Q8KSScratchBytes(embDim);
            _bQ8KExpStride = SimdKernels.Q8KSScratchBytes(_expertDim);
            _bNormAllQ8K = (byte*)NativeMemory.Alloc((nuint)((long)N * _bQ8KEmbStride));
            _bGateAllQ8K = (byte*)NativeMemory.Alloc((nuint)(perTokSel * _bQ8KExpStride));
        }

        // Per-expert bucket bookkeeping (sized by numExperts, allocated once).
        if (_bExpStart == null)
        {
            _bExpStart    = (int*)NativeMemory.Alloc((nuint)(_numExperts + 1) * sizeof(int));
            _bExpCursor   = (int*)NativeMemory.Alloc((nuint)_numExperts * sizeof(int));
            _bUsedExperts = (int*)NativeMemory.Alloc((nuint)_numExperts * sizeof(int));
        }

        _bCap = N;
    }

    private static float* AllocL(long count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count * (nuint)sizeof(float));

    private static float* AllocPinnedL(long count)
    {
        nint ptr = CudaBackend.AllocatePinnedHost((nuint)count * sizeof(float));
        if (ptr == nint.Zero)
            throw new InvalidOperationException($"AllocatePinnedHost({count} floats) failed for batched prefill scratch.");
        return (float*)ptr;
    }

    private void FreeBatchedScratch()
    {
        if (_gpuStreamAll is { } s) { _gpu.Free(s); _gpuStreamAll = null; }
        FreeBatchedHostScratch();
        FreeBatchedTrunkScratch();
    }

    /// <summary>
    /// Grow-only (exact-size) allocation of the device batched-trunk activation
    /// buffers (issue #111). Sized to exactly <paramref name="N"/> tokens — the GEMM-N
    /// kernels derive <c>rows = output.ElementCount / nTok</c>, so over-allocation would
    /// misshape the launch. Reallocated when N changes (at most twice per prompt: a full
    /// chunk then a remainder). Only used on the GPU-GDN path (<c>!_cpuGdn</c>).
    /// </summary>
    private void EnsureBatchedTrunkScratch(int N)
    {
        if (_btCap == N) return;
        FreeBatchedTrunkScratch();

        int embDim = _embDim;
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        Tensor A(long elems) => _gpu.Allocate(TensorShape.D1(elems));

        _gpuBtNorm     = A((long)N * embDim);
        _gpuBtBlockOut = A((long)N * embDim);
        _gpuBtMoeNorm  = A((long)N * embDim);
        if (_cpuMoe && _gpuRouterPrefill)
            _gpuBtRouterAll = A((long)N * _numExperts);   // batched router logits (#388, CPU-MoE GPU-router)
        if (_cpuMoe)
            _gpuBtShared = A((long)N * embDim);   // shared-expert out (CPU-MoE combine only)
        _gpuBtQkv      = A((long)N * _gdnConvChannels);
        _gpuBtZ        = A((long)N * _gdnValueDim);
        _gpuBtAlpha    = A((long)N * _gdnNumVHeads);
        _gpuBtBeta     = A((long)N * _gdnNumVHeads);
        _gpuBtGdnOut   = A((long)N * _gdnValueDim);
        _gpuBtQGate    = A((long)N * qDim * 2);
        _gpuBtQ        = A((long)N * qDim);
        _gpuBtGate     = A((long)N * qDim);
        _gpuBtK        = A((long)N * kvDim);
        _gpuBtV        = A((long)N * kvDim);
        _gpuBtAttnOut  = A((long)N * qDim);
        // Shared-expert scratch is used only by the CPU-MoE TrunkLayerBatched combine;
        // the dense / GPU-SLRU batched-trunk path (issue #119) computes its FFN/MoE
        // per token and never touches these (and _expertDim is unset for dense models).
        if (_cpuMoe)
        {
            _gpuBtSGate = A((long)N * _expertDim);
            _gpuBtSUp   = A((long)N * _expertDim);
        }
        if (BatchedGdnScanEnabled)
        {
            _gpuBtQkvConv = A((long)N * _gdnConvChannels);
            _gpuBtQHead   = A((long)N * _gdnValueDim);
            _gpuBtKHead   = A((long)N * _gdnValueDim);
        }
        _btCap = N;
    }

    private void FreeBatchedTrunkScratch()
    {
        void F(ref Tensor? t) { if (t is { } v) { _gpu.Free(v); t = null; } }
        F(ref _gpuBtNorm); F(ref _gpuBtBlockOut); F(ref _gpuBtMoeNorm); F(ref _gpuBtRouterAll); F(ref _gpuBtShared);
        F(ref _gpuBtQkv); F(ref _gpuBtZ); F(ref _gpuBtAlpha); F(ref _gpuBtBeta); F(ref _gpuBtGdnOut);
        F(ref _gpuBtQGate); F(ref _gpuBtQ); F(ref _gpuBtGate); F(ref _gpuBtK); F(ref _gpuBtV); F(ref _gpuBtAttnOut);
        F(ref _gpuBtSGate); F(ref _gpuBtSUp);
        F(ref _gpuBtQkvConv); F(ref _gpuBtQHead); F(ref _gpuBtKHead);
        _btCap = 0;
        // Caller contract: EnsureBatchedFfnScratch(N) must always run AFTER this (the
        // FFN scratch is sized to the same N as the trunk scratch). PrefillBatchedTrunkGpuFfn
        // calls EnsureBatchedTrunkScratch then EnsureBatchedFfnScratch in that order, so an
        // N-change that re-frees the FFN scratch here is immediately re-allocated.
        FreeBatchedFfnScratch();
    }

    /// <summary>
    /// Issue #121: exact-size device scratch for the GEMM-N-batched FFN/MoE stage. Only
    /// allocated on the GPU-FFN batched-prefill path (dense GPU-FFN layers or GPU-SLRU
    /// MoE) — the CPU-MoE / CPU-dense-fallback paths never call this. Sized to exactly
    /// <paramref name="N"/> tokens so the GEMM-N kernels derive
    /// <c>rows = output.ElementCount / nTok</c> correctly. Reallocated when N changes.
    /// </summary>
    private void EnsureBatchedFfnScratch(int N)
    {
        if (_bfCap == N) return;
        FreeBatchedFfnScratch();

        int embDim = _embDim;
        Tensor A(long elems) => _gpu.Allocate(TensorShape.D1(elems));

        if (_hp.IsMoE)
        {
            // GPU-SLRU routed-MoE grouped-by-expert scratch.
            int na = _numActiveExperts;
            _gpuBfHiddenAll        = A((long)N * embDim);              // per-token shared-expert out
            _gpuBfMoeDownPartial   = A((long)N * na * embDim);         // per-(token,slot) down partials
            _gpuBfMoeNormGathN     = A((long)N * embDim);              // gathered routed-token norms
            _gpuBfMoeGateGathN     = A((long)N * _expertDim);          // gathered gate proj
            _gpuBfMoeUpGathN       = A((long)N * _expertDim);          // gathered up proj
            _gpuBfMoeDownGathN     = A((long)N * embDim);              // gathered down out
            _gpuBfMoeRouterAll     = A((long)N * _numExperts);         // batched router logits
            // Issue #129: batched shared-expert gate/up + per-token gate scalar + device
            // top-k weights for the single-launch reduce.
            _gpuBfShGateAll        = A((long)N * _expertDim);          // batched shared-expert gate
            _gpuBfShUpAll          = A((long)N * _expertDim);          // batched shared-expert up
            _gpuBfShexpScaleDev    = A((long)N);                       // per-token shexp sigmoid gate
            _gpuBfMoeWeightsDev    = A((long)N * na);                  // top-k weights (device)

            long perTokSel = (long)N * na;
            _bfSelected     = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));
            _bfWeights      = (float*)NativeMemory.Alloc((nuint)perTokSel * sizeof(float));
            _bfShexpScale   = (float*)NativeMemory.Alloc((nuint)N * sizeof(float));
            _bfRouterAll    = (float*)NativeMemory.Alloc((nuint)((long)N * _numExperts) * sizeof(float));
            _bfNormReadback = (float*)NativeMemory.Alloc((nuint)((long)N * embDim) * sizeof(float));
            _bfExpTokI      = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));
            _bfExpTokK      = (int*)NativeMemory.Alloc((nuint)perTokSel * sizeof(int));
            _bfGathTokI     = (int*)NativeMemory.Alloc((nuint)N * sizeof(int));
            if (_bfExpStart == null)
            {
                _bfExpStart    = (int*)NativeMemory.Alloc((nuint)(_numExperts + 1) * sizeof(int));
                _bfExpCursor   = (int*)NativeMemory.Alloc((nuint)_numExperts * sizeof(int));
                _bfUsedExperts = (int*)NativeMemory.Alloc((nuint)_numExperts * sizeof(int));
            }
        }
        else
        {
            // Dense GPU-FFN gate/up/down batched scratch.
            _gpuBfGateAll   = A((long)N * _intermDim);
            _gpuBfUpAll     = A((long)N * _intermDim);
            _gpuBfHiddenAll = A((long)N * embDim);
        }
        _bfCap = N;
    }

    private void FreeBatchedFfnScratch()
    {
        void F(ref Tensor? t) { if (t is { } v) { _gpu.Free(v); t = null; } }
        F(ref _gpuBfGateAll); F(ref _gpuBfUpAll); F(ref _gpuBfHiddenAll);
        F(ref _gpuBfMoeDownPartial); F(ref _gpuBfMoeGateGathN); F(ref _gpuBfMoeUpGathN);
        F(ref _gpuBfMoeNormGathN); F(ref _gpuBfMoeDownGathN); F(ref _gpuBfMoeRouterAll);
        F(ref _gpuBfShGateAll); F(ref _gpuBfShUpAll); F(ref _gpuBfShexpScaleDev); F(ref _gpuBfMoeWeightsDev);
        void FH(ref int* p) { if (p != null) { NativeMemory.Free(p); p = null; } }
        void FHf(ref float* p) { if (p != null) { NativeMemory.Free(p); p = null; } }
        FH(ref _bfSelected); FHf(ref _bfWeights); FHf(ref _bfShexpScale);
        FHf(ref _bfRouterAll); FHf(ref _bfNormReadback);
        FH(ref _bfExpTokI); FH(ref _bfExpTokK); FH(ref _bfGathTokI);
        // _bfExpStart/_bfExpCursor/_bfUsedExperts are sized by numExperts (N-independent);
        // keep them across reallocs and free only in Dispose.
        _bfCap = 0;
    }

    /// <summary>
    /// Sequential per-token trunk for one layer (the pre-#111 path). Used on the
    /// CPU-GDN debug path and when SHARPI_BATCHED_TRUNK=0. Produces the host
    /// <see cref="_bResidAll"/> / <see cref="_bNormAll"/> / <see cref="_bSharedAll"/>
    /// buffers and self-syncs the stream, exactly as the batched variant.
    /// </summary>
    private void TrunkLayerSequential(int layer, int N, int startPos, bool isAttn,
                                      bool snapKvActive, int wStart)
    {
        int embDim = _embDim;
        for (int i = 0; i < N; i++)
        {
            _gpu.CopyDeviceRegion(_gpuHidden, 0, _gpuStreamAll!,
                                  (long)i * embDim * sizeof(float), (long)embDim * sizeof(float));
            _gpu.CopyDevice(_gpuResidual, _gpuHidden);
            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[layer], _hp.RmsNormEps);

            _snapKvCaptureSlot = (snapKvActive && i >= wStart) ? (i - wStart) : -1;

            if (isAttn)
                GpuAttnBlockAt(layer, position: startPos + i, kvPosition: startPos + i,
                               normIn: _gpuNormBuf, hiddenOut: _gpuHidden);
            else if (_cpuGdn)
                CpuGdnBlockAt(layer, position: startPos + i, normInGpu: _gpuNormBuf,
                              hiddenOutGpu: _gpuHidden, cpuNormScratch: _cpuNormBuf,
                              cpuHiddenScratch: _cpuHiddenOut);
            else
                GpuGdnBlockAt(layer, position: startPos + i, normIn: _gpuNormBuf, hiddenOut: _gpuHidden);

            _gpu.AddInPlace(_gpuHidden, _gpuResidual);           // postBlock (MoE residual)
            _gpu.DownloadAsync(_gpuHidden, (nint)(_bResidAll + (long)i * embDim), embDim);

            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuPostAttnNorm[layer], _hp.RmsNormEps);
            _gpu.DownloadAsync(_gpuNormBuf, (nint)(_bNormAll + (long)i * embDim), embDim);

            GpuMatMul(_gpuFfnGate, _gpuWGateShexp[layer], _gpuNormBuf);
            GpuMatMul(_gpuFfnUp, _gpuWUpShexp[layer], _gpuNormBuf);
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuSharedOut, _gpuWDownShexp[layer], _gpuFfnGate);
            _gpu.DownloadAsync(_gpuSharedOut, (nint)(_bSharedAll + (long)i * embDim), embDim);
        }
        _snapKvCaptureSlot = -1;
        _gpu.Synchronize();   // drain all queued D2H: resid/norm/shared now host-valid
    }

    /// <summary>
    /// GEMM-batched trunk for one layer (issue #111). The projection matvecs —
    /// GDN qkv/z/alpha/beta/ssm_out, attention q/k/v/o, shared-expert gate/up/down —
    /// plus the attn-norm / post-attn-norm RmsNorms, per-head Q/K RMSNorm, RoPE,
    /// Q‖gate split and GLU gate run as single batched launches over all N tokens.
    /// The conv1d / delta-net recurrence and KV-append / scaled-dot-product attention
    /// were per-position over N (per-token slices via <see cref="CudaBackend.View"/>)
    /// in #111; issue #114-B now batches them too by default — a fused sequential-scan
    /// for the GDN recurrence and a batched-query SDPA — falling back to the per-position
    /// View loops under <c>SHARPI_BATCHED_GDN_SCAN=0</c> / <c>SHARPI_BATCHED_ATTN=0</c>
    /// (and, for attention, when the chunk exceeds the shared-scores window or SnapKV is
    /// active). Output is bit-identical to
    /// <see cref="TrunkLayerSequential"/>: every batched kernel runs the same per-row /
    /// per-element computation as its single-token counterpart, only collapsing the
    /// N per-token launches into one each — which is what removes the host launch
    /// overhead that dominates GDN-hybrid prefill.
    /// </summary>
    /// <summary>
    /// Config-independent batched trunk block for one layer: attn-norm → attention /
    /// GDN block → postBlock residual → post-attention norm, all batched over N tokens
    /// and left on the GPU. On return <see cref="_gpuBtBlockOut"/> holds the postBlock
    /// residual (MoE/FFN residual) and <see cref="_gpuBtMoeNorm"/> the post-attention
    /// norm (MoE/FFN input). The shared-expert / host-combine plumbing is layered on top
    /// by the per-config callers (<see cref="TrunkLayerBatched"/> for CPU-MoE;
    /// <see cref="PrefillBatchedTrunkGpuFfn"/> for the dense / GPU-SLRU paths, issue #119).
    /// Bit-identical to the per-token trunk in <see cref="TrunkLayerSequential"/> /
    /// <see cref="Forward"/>: every batched kernel runs the same per-row computation as
    /// its single-token counterpart.
    /// </summary>
    private void TrunkBlockBatched(int layer, int N, int startPos, bool isAttn,
                                   bool snapKvActive, int wStart, bool gdnSnapRing = false)
    {
        int embDim = _embDim;
        EnsureBatchedTrunkScratch(N);
        var stream   = _gpuStreamAll!;
        var norm     = _gpuBtNorm!;
        var blockOut = _gpuBtBlockOut!;
        var moeNorm  = _gpuBtMoeNorm!;

        // attn-norm (block input) over all tokens.
        _gpu.RmsNormBatched(norm, stream, _gpuAttnNorm[layer], N, embDim, _hp.RmsNormEps);

        if (isAttn)
            AttnBlockBatched(layer, N, startPos, snapKvActive, wStart, norm, blockOut);
        else
            GdnBlockBatched(layer, N, norm, blockOut, gdnSnapRing);

        // postBlock = blockOut + stream (the pre-block residual). blockOut now holds
        // the MoE/FFN residual.
        _gpu.AddInPlace(blockOut, stream);

        // post-attention norm (MoE/FFN input).
        _gpu.RmsNormBatched(moeNorm, blockOut, _gpuPostAttnNorm[layer], N, embDim, _hp.RmsNormEps);
    }

    private void TrunkLayerBatched(int layer, int N, int startPos, bool isAttn,
                                   bool snapKvActive, int wStart)
    {
        int embDim = _embDim;
        TrunkBlockBatched(layer, N, startPos, isAttn, snapKvActive, wStart);
        var blockOut = _gpuBtBlockOut!;
        var moeNorm  = _gpuBtMoeNorm!;
        var shared   = _gpuBtShared!;

        // Download the postBlock residual + post-attn norm to the host combine buffers.
        // (blockOut is unchanged by the moeNorm RmsNorm, so the queue order is moot.)
        _gpu.DownloadAsync(blockOut, (nint)_bResidAll, (int)((long)N * embDim));
        _gpu.DownloadAsync(moeNorm, (nint)_bNormAll, (int)((long)N * embDim));

        // Shared expert (unscaled) — gate/up/down batched over all tokens; scale
        // folded into the host combine, matching the sequential path's operand order.
        GpuMatMulBatched(_gpuBtSGate!, _gpuWGateShexp[layer], moeNorm, N);
        GpuMatMulBatched(_gpuBtSUp!,   _gpuWUpShexp[layer],   moeNorm, N);
        _gpu.SiLuMul(_gpuBtSGate!, _gpuBtSUp!);   // pointwise over N×expertDim
        GpuMatMulBatched(shared, _gpuWDownShexp[layer], _gpuBtSGate!, N);
        _gpu.DownloadAsync(shared, (nint)_bSharedAll, (int)((long)N * embDim));

        // #388: batched GPU router GEMM, overlapping the trunk's D2H. RAW logits → host (softmax +
        // top-k run on the host in PrefillBatchedCpuMoe, reading _btRouterAll). Skipped (→ CPU router)
        // when the router weight isn't batched-GEMM-supported. moeNorm (_gpuBtMoeNorm) is valid here.
        _btRouterGpuValid = _gpuRouterPrefill && _cpuMoe && _gpuBtRouterAll is not null
                            && BatchedMatMulSupported(_gpuWGateInp[layer]);
        if (_btRouterGpuValid)
        {
            GpuMatMulBatched(_gpuBtRouterAll!, _gpuWGateInp[layer], moeNorm, N);
            _gpu.DownloadAsync(_gpuBtRouterAll!, (nint)_btRouterAll, (int)((long)N * _numExperts));
        }

        _gpu.Synchronize();   // drain all queued D2H: resid/norm/shared (+ router) now host-valid
    }

    /// <summary>Batched GDN block: projections over N tokens; fused sequential-scan
    /// recurrence + batched conv1d/L2norm/tile by default (issue #114-B), or the
    /// per-position View loop under <c>SHARPI_BATCHED_GDN_SCAN=0</c>.
    /// <para><paramref name="snapRing"/> (issues #30/#290 batched verify) keeps the
    /// fused path and captures the post-token-i (scan, conv) state into device ring
    /// slot i AS IT SCANS: the sequential-scan kernel mirrors each token's
    /// post-update state into the ring (zero extra launches), and a single
    /// conv-capture launch dumps the per-token conv states. This replaces the old
    /// per-position relaunch (k×8 launches/layer) + the per-slot
    /// <see cref="CaptureGdnRingSlot"/> CopyDeviceRegion fan-out. Verify always uses
    /// the byte-exact scan (never the chunked prefill form), so it stays in the same
    /// precision class as prefill/Forward. The per-position View loop below remains
    /// the <c>SHARPI_BATCHED_GDN_SCAN=0</c> fallback and keeps its CopyDeviceRegion
    /// capture.</para></summary>
    private void GdnBlockBatched(int layer, int N, Tensor norm, Tensor blockOut, bool snapRing = false)
    {
        int convCh = _gdnConvChannels, valDim = _gdnValueDim, nVH = _gdnNumVHeads;
        int kDim = _gdnKeyDim, hd = _gdnHeadDim;
        var qkvAll = _gpuBtQkv!; var zAll = _gpuBtZ!;
        var alphaAll = _gpuBtAlpha!; var betaAll = _gpuBtBeta!; var gdnOutAll = _gpuBtGdnOut!;

        // Batched projections (one launch each over all tokens).
        GpuMatMulBatched(qkvAll,   _gpuWAttnQkv[layer],  norm, N);
        GpuMatMulBatched(zAll,     _gpuWAttnGate[layer], norm, N);
        GpuMatMulBatched(alphaAll, _gpuWSsmAlpha[layer], norm, N);
        GpuMatMulBatched(betaAll,  _gpuWSsmBeta[layer],  norm, N);

        var scanState = _gpuGdnScanState[layer]!;
        var convState = _gpuGdnConvState[layer]!;

        // Issue #114-B: fuse the per-position conv1d + delta-net recurrence into
        // one batched launch per stage + a single sequential-scan kernel. Output is
        // bit-identical to the per-position View loop below (same per-position math,
        // same reduction order; only the host launch overhead is removed). Issue
        // #290: this path also serves batched verify (snapRing) — the scan/conv
        // captures dump each token's state into the ring during the fused pass.
        if (BatchedGdnScanEnabled)
        {
            var qkvConvAll = _gpuBtQkvConv!;
            var qHeadAll = _gpuBtQHead!;
            var kHeadAll = _gpuBtKHead!;

            // #290 ring geometry (only when capturing): this layer's dense GDN index,
            // per-layer float counts, and the inter-slot stride (= numGdn × per-layer
            // floats). gdnIdx ≥ 0 here — GdnBlockBatched only runs for GDN layers.
            int gdnIdx = snapRing ? _gdnStateCache.GdnLayerOf(layer) : -1;
            int numGdn = _gdnStateCache.NumGdnLayers;
            int scanF = _gdnStateCache.ScanStateFloatsPerLayer;
            int convF = _gdnStateCache.ConvStateFloatsPerLayer;
            int nCapture = N - 1;   // slots [0, N-1): state after each non-final token

            // conv1d over all tokens (read-only state), capture the per-token conv
            // states (BEFORE advancing the live state), then advance the state.
            _gpu.GdnConv1dDecodeBatched(qkvAll, convState, _gpuSsmConv1d[layer], qkvConvAll,
                convCh, _gdnConvKernel, N);
            if (snapRing && nCapture > 0 && convF > 0 && _gpuGdnRingConv is { } ringConv)
                _gpu.GdnConv1dStateCaptureRing(qkvAll, convState, ringConv, (long)gdnIdx * convF,
                    convCh, _gdnConvKernel, numGdn * convF, nCapture);
            _gpu.GdnConv1dStateUpdateBatched(qkvAll, convState, convCh, _gdnConvKernel, N);
            // SiLU over the whole [N × convCh] (matches the per-token full-convCh SiLU).
            _gpu.SiLUInPlace(qkvConvAll);
            // L2-norm the Q (offset 0) and K (offset kDim) regions, per head, per token.
            _gpu.GdnL2NormPerHeadBatched(qkvConvAll, 0,    _gdnNumKHeads, hd, convCh, N, eps: 1e-6f);
            _gpu.GdnL2NormPerHeadBatched(qkvConvAll, kDim, _gdnNumKHeads, hd, convCh, N, eps: 1e-6f);
            // Tile Q and K heads (GQA broadcast) into the [N × valueDim] head buffers.
            _gpu.GdnTileHeadsBatched(qkvConvAll, 0,    qHeadAll, 0, _gdnNumKHeads, _gdnKvRepeat, hd, convCh, valDim, N);
            _gpu.GdnTileHeadsBatched(qkvConvAll, kDim, kHeadAll, 0, _gdnNumKHeads, _gdnKvRepeat, hd, convCh, valDim, N);
            // Recurrence: v read straight from the silu'd conv output's V region
            // (vHeadOff = 2*kDim, stride convCh); q/k from the tiled head buffers.
            // Verify (snapRing) always uses the byte-exact scan with ring capture;
            // GdnChunkedPrefill (opt-in, NOT byte-exact) only serves clean prefill.
            if (snapRing)
                _gpu.GdnRecurrenceScan(
                    scanState, qHeadAll, kHeadAll, qkvConvAll,
                    alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                    zAll, gdnOutAll,
                    nVH, hd, normEps: 1e-6f,
                    qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                    zStride: valDim, oStride: valDim, nTok: N,
                    ringScan: _gpuGdnRingScan, ringScanFloatOffset: (long)gdnIdx * scanF,
                    ringSlotStride: numGdn * scanF, nCapture: nCapture);
            else if (GdnChunkedPrefillOverride ?? _gpuMoePrefill)
                _gpu.GdnChunkedPrefill(
                    scanState, qHeadAll, kHeadAll, qkvConvAll,
                    alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                    zAll, gdnOutAll,
                    nVH, hd, normEps: 1e-6f,
                    qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                    zStride: valDim, oStride: valDim, nTok: N);
            else
                _gpu.GdnRecurrenceScan(
                    scanState, qHeadAll, kHeadAll, qkvConvAll,
                    alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                    zAll, gdnOutAll,
                    nVH, hd, normEps: 1e-6f,
                    qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                    zStride: valDim, oStride: valDim, nTok: N);

            GpuMatMulBatched(blockOut, _gpuWSsmOut[layer], gdnOutAll, N);
            return;
        }

        // Per-token conv1d + delta-net recurrence (positional → sequential). The
        // conv/L2/tile scratch (_gpuGdnQkvConv / _gpuGdnQHead / _gpuGdnKHead /
        // _gpuGdnVHead) is reused per token; only the batched-buffer inputs/output
        // need per-token views.
        for (int i = 0; i < N; i++)
        {
            var qkvIn = _gpu.View(qkvAll, (long)i * convCh, convCh);
            var zIn   = _gpu.View(zAll,   (long)i * valDim, valDim);
            var aIn   = _gpu.View(alphaAll, (long)i * nVH, nVH);
            var bIn   = _gpu.View(betaAll,  (long)i * nVH, nVH);
            var outV  = _gpu.View(gdnOutAll, (long)i * valDim, valDim);
            try
            {
                _gpu.GdnConv1dDecode(qkvIn, convState, _gpuSsmConv1d[layer], _gpuGdnQkvConv,
                    convCh, _gdnConvKernel);
                _gpu.SiLUInPlace(_gpuGdnQkvConv);
                _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, 0,    _gdnNumKHeads, hd, eps: 1e-6f);
                _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, kDim, _gdnNumKHeads, hd, eps: 1e-6f);
                _gpu.GdnTileHeads(_gpuGdnQkvConv, 0,    _gpuGdnQHead, 0, _gdnNumKHeads, _gdnKvRepeat, hd);
                _gpu.GdnTileHeads(_gpuGdnQkvConv, kDim, _gpuGdnKHead, 0, _gdnNumKHeads, _gdnKvRepeat, hd);
                _gpu.CopyDeviceRegion(_gpuGdnVHead, 0,
                    _gpuGdnQkvConv, 2L * kDim * sizeof(float), (long)valDim * sizeof(float));
                _gpu.GdnRecurrenceDecode(
                    scanState, _gpuGdnQHead, _gpuGdnKHead, _gpuGdnVHead, aIn, bIn,
                    _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer], zIn, outV,
                    nVH, hd, normEps: 1e-6f);
                // Batched verify (issue #30): capture the post-token-i state into
                // device ring slot i so a rejection at position startPos+i+1 can
                // restore it. Stream-ordered D2D, ~2 MB scan + conv per layer.
                if (snapRing && i < N - 1)
                    CaptureGdnRingSlot(slot: i, layer);
            }
            finally
            {
                _gpu.Free(qkvIn); _gpu.Free(zIn); _gpu.Free(aIn); _gpu.Free(bIn); _gpu.Free(outV);
            }
        }

        // Batched ssm_out projection: blockOut = WSsmOut @ gdnOutAll.
        GpuMatMulBatched(blockOut, _gpuWSsmOut[layer], gdnOutAll, N);
    }

    /// <summary>Batched attention block: projections + Q/K norm + RoPE over N tokens;
    /// batched KV-append + batched-query SDPA by default — the shared-scores fast path
    /// (issue #114-B) at <c>startPos+N ≤ 4096</c>, the wave-based global-scratch SDPA
    /// (issue #118) past it. Issue #122: the batched path also runs with SnapKV active,
    /// capturing the trailing-window Q in a single batched copy. Only
    /// <c>SHARPI_BATCHED_ATTN=0</c> takes the per-position KV-append + SDPA loop (which
    /// keeps its own per-position Q-capture).</summary>
    private void AttnBlockBatched(int layer, int N, int startPos, bool snapKvActive, int wStart,
                                  Tensor norm, Tensor blockOut)
    {
        int qDim = _numHeads * _headDim, kvDim = _numKvHeads * _headDim;
        var qGateAll = _gpuBtQGate!; var qAll = _gpuBtQ!; var gateAll = _gpuBtGate!;
        var kAll = _gpuBtK!; var vAll = _gpuBtV!; var attnOutAll = _gpuBtAttnOut!;

        // Batched projections + de-interleave + per-head norm + RoPE over all tokens.
        GpuMatMulBatched(qGateAll, _gpuWQGate[layer], norm, N);
        GpuMatMulBatched(kAll,     _gpuWK[layer],     norm, N);
        GpuMatMulBatched(vAll,     _gpuWV[layer],     norm, N);

        _gpu.SplitQGBatched(qAll, gateAll, qGateAll, _numHeads, _headDim, N);
        _gpu.HeadNormBatched(qAll, _gpuQNorm[layer], _numHeads,   _headDim, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNormBatched(kAll, _gpuKNorm[layer], _numKvHeads, _headDim, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.RoPEPartialBatched(qAll, startPos, _headDim, _ropeDim, _hp.RopeTheta, _numHeads,   N, neox: true);
        _gpu.RoPEPartialBatched(kAll, startPos, _headDim, _ropeDim, _hp.RopeTheta, _numKvHeads, N, neox: true);

        // Issue #114-B / #118 / #122: batch the KV-append + SDPA into one launch each.
        // ≤4096 uses the shared-scores fast path (AttentionBatched); past 4096 uses the
        // wave-based global-scratch SDPA (AttentionBatchedWave, issue #118) — both
        // bit-identical to the per-position loop below (each (head, query) block clones
        // llm_attention). Issue #122: this path now runs with SnapKV active too — the
        // trailing-window Q-capture is done as a single batched CopyDeviceRegion from
        // qAll BEFORE the SDPA. This is bit-identical to the per-position capture below:
        //   • Captured Q — copying contiguous rows [wStart,N) of qAll equals the
        //     per-position per-row copies of the same qAll rows → byte-identical, because
        //     the destination slots (i-wStart)=0..(N-1-wStart) for a fixed attnIdx are
        //     also contiguous (stride qDim).
        //   • Attention output — AttentionBatched/AttentionBatchedWave are already proven
        //     bit-identical to the per-position Attention loop; SnapKV only changes which
        //     Q rows are captured, not the attention math.
        // The per-position loop below remains the fallback for !BatchedAttnEnabled (and
        // keeps its own per-position capture for that path).
        if (BatchedAttnEnabled)
        {
            // Issue #122: single batched Q-capture for the trailing SnapKV window.
            // Source rows [wStart,N) of qAll are contiguous, and their destination slots
            // for this attnIdx are contiguous, so the whole window copies in one shot.
            if (snapKvActive && N > wStart && _snapKvQCapture is { } snapCapBuf)
            {
                int attnIdx = _attnLayerIndexOf[layer];
                if (attnIdx >= 0)
                {
                    long dstOff = (long)attnIdx * _snapKvQCaptureW * qDim;
                    _gpu.CopyDeviceRegion(snapCapBuf, dstOff * sizeof(float),
                                          qAll, (long)wStart * qDim * sizeof(float),
                                          (long)(N - wStart) * qDim * sizeof(float));
                }
            }
            bool sharedFast = startPos + N <= 4096;
            if (_kvDType == DType.BFloat16)
            {
                _gpu.KvAppendBatchedBf16(kAll, vAll, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, startPos, _maxSeqLen, N);
                if (sharedFast)
                    _gpu.AttentionBatchedBf16(qAll, _gpuKCache[layer]!, _gpuVCache[layer]!, attnOutAll,
                        _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, N);
                else
                    _gpu.AttentionBatchedWaveBf16(qAll, _gpuKCache[layer]!, _gpuVCache[layer]!, attnOutAll,
                        _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, N);
            }
            else
            {
                _gpu.KvAppendBatched(kAll, vAll, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, startPos, _maxSeqLen, N);
                if (sharedFast)
                    _gpu.AttentionBatched(qAll, _gpuKCache[layer]!, _gpuVCache[layer]!, attnOutAll,
                        _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, N);
                else
                    _gpu.AttentionBatchedWave(qAll, _gpuKCache[layer]!, _gpuVCache[layer]!, attnOutAll,
                        _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, N);
            }
            _gpu.SigmoidMulInPlace(attnOutAll, gateAll);
            GpuMatMulBatched(blockOut, _gpuWO[layer], attnOutAll, N);
            return;
        }

        // Fallback (!BatchedAttnEnabled): per-position KV-append + scaled-dot-product
        // attention (positional → sequential), plus SnapKV Q-capture for the trailing
        // window. Issue #122: the batched path above now handles the SnapKV-active case,
        // so this per-position capture only runs when batched attention is disabled.
        for (int i = 0; i < N; i++)
        {
            int pos = startPos + i;
            var qV = _gpu.View(qAll, (long)i * qDim, qDim);
            var kV = _gpu.View(kAll, (long)i * kvDim, kvDim);
            var vV = _gpu.View(vAll, (long)i * kvDim, kvDim);
            var oV = _gpu.View(attnOutAll, (long)i * qDim, qDim);
            try
            {
                if (snapKvActive && i >= wStart && _snapKvQCapture is { } capBuf)
                {
                    int attnIdx = _attnLayerIndexOf[layer];
                    if (attnIdx >= 0)
                    {
                        long dstOff = ((long)attnIdx * _snapKvQCaptureW + (i - wStart)) * qDim;
                        _gpu.CopyDeviceRegion(capBuf, dstOff * sizeof(float),
                                              qV, 0, (long)qDim * sizeof(float));
                    }
                }
                if (_kvDType == DType.BFloat16)
                {
                    _gpu.KvAppendBf16(kV, vV, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, pos, _maxSeqLen);
                    _gpu.AttentionBf16(qV, _gpuKCache[layer]!, _gpuVCache[layer]!, oV, _gpuAttnScratch,
                        _numHeads, _numKvHeads, _headDim, pos + 1, _maxSeqLen);
                }
                else
                {
                    _gpu.KvAppend(kV, vV, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, pos, _maxSeqLen);
                    _gpu.Attention(qV, _gpuKCache[layer]!, _gpuVCache[layer]!, oV, _gpuAttnScratch,
                        _numHeads, _numKvHeads, _headDim, pos + 1, _maxSeqLen);
                }
            }
            finally
            {
                _gpu.Free(qV); _gpu.Free(kV); _gpu.Free(vV); _gpu.Free(oV);
            }
        }

        // GLU gate (pointwise over N×qDim) then batched O projection.
        _gpu.SigmoidMulInPlace(attnOutAll, gateAll);
        GpuMatMulBatched(blockOut, _gpuWO[layer], attnOutAll, N);
    }

    private void FreeBatchedHostScratch()
    {
        if (_bResidAll  != null) { CudaBackend.FreePinnedHost((nint)_bResidAll);  _bResidAll = null; }
        if (_bNormAll   != null) { CudaBackend.FreePinnedHost((nint)_bNormAll);   _bNormAll = null; }
        if (_btRouterAll != null) { CudaBackend.FreePinnedHost((nint)_btRouterAll); _btRouterAll = null; }
        if (_bSharedAll != null) { CudaBackend.FreePinnedHost((nint)_bSharedAll); _bSharedAll = null; }
        if (_bHiddenAll != null) { CudaBackend.FreePinnedHost((nint)_bHiddenAll); _bHiddenAll = null; }
        if (_bRoutedAll != null) { NativeMemory.Free(_bRoutedAll); _bRoutedAll = null; }
        if (_bGateAll   != null) { NativeMemory.Free(_bGateAll);   _bGateAll = null; }
        if (_bUpAll     != null) { NativeMemory.Free(_bUpAll);     _bUpAll = null; }
        if (_bDownPartial != null) { NativeMemory.Free(_bDownPartial); _bDownPartial = null; }
        if (_bSelected   != null) { NativeMemory.Free(_bSelected);   _bSelected = null; }
        if (_bWeights    != null) { NativeMemory.Free(_bWeights);    _bWeights = null; }
        if (_bShexpScale != null) { NativeMemory.Free(_bShexpScale); _bShexpScale = null; }
        if (_bExpTokI    != null) { NativeMemory.Free(_bExpTokI);    _bExpTokI = null; }
        if (_bExpTokK    != null) { NativeMemory.Free(_bExpTokK);    _bExpTokK = null; }
        if (_bNormAllQ8K != null) { NativeMemory.Free(_bNormAllQ8K); _bNormAllQ8K = null; }
        if (_bGateAllQ8K != null) { NativeMemory.Free(_bGateAllQ8K); _bGateAllQ8K = null; }
        _bCap = 0;
    }

    /// <summary>
    /// Per-layer batched prompt prefill for the CPU-MoE GDN-hybrid path. The trunk
    /// (attention / GDN) runs sequentially per token on the GPU exactly as in
    /// <see cref="Forward"/> — the GDN recurrence and KV append are positional — but
    /// the routed MoE experts run once batched per layer (<see cref="BatchedRoutedExperts"/>),
    /// reading each selected expert's weight rows once and dotting them against every
    /// token that routed to it. Produces bit-identical KV cache, GDN state, MTP hidden
    /// history, and last-token logits as the sequential loop.
    ///
    /// <para><b>Not transactional.</b> The trunk mutates the GDN scan/conv state in
    /// place and writes KV pages as it goes, but defers the <c>IncrementPosition</c>
    /// bookkeeping to the end. A throw mid-chunk (e.g. a CUDA stream fault) therefore
    /// leaves the recurrent GDN state partially advanced while the length counters
    /// still read <paramref name="startPos"/> — the cache is NOT cleanly truncated as
    /// it would be after a failed sequential <see cref="Forward"/>. The caller must
    /// treat such a failure as fatal for this pass (discard it / reload the model);
    /// retrying the prefill would run on poisoned recurrent state. In practice the
    /// only throw sources past the entry guards are genuine CUDA/OOM faults, which
    /// are fatal anyway.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillBatchedCpuMoe(IReadOnlyList<int> tokens, int startPos,
                                                     bool snapKvActive, int W, int wStart)
    {
        int N = tokens.Count;
        int embDim = _embDim;
        int na = _numActiveExperts;
        EnsureBatchedScratch(N);

        // Hoisted out of the layer loop (CA2014: no stackalloc in a loop).
        Span<int> sel = stackalloc int[na];
        Span<float> wts = stackalloc float[na];

        long t0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        long trunkTicks = 0, routerTicks = 0, routedTicks = 0, combineTicks = 0;
        _profMoeNormQ = _profMoePhaseA = _profMoeSilu = _profMoeGateQ = _profMoePhaseC = _profMoeBucket = 0;
        _profGoDequant = _profGoUpload = _profGoGather = _profGoGemm = _profGoDownScatter = 0;

        // Pessimistic fault latch: the GDN recurrent-state mutation + deferred
        // length-counter bookkeeping below is non-transactional, so mark the pass
        // poisoned for the whole region and clear it only once the counters have
        // been advanced consistently (step 3). A throw anywhere in between leaves
        // _faulted set, and ThrowIfFaulted() blocks any retry on corrupt state.
        _faulted = true;

        // 1. Embed every token into the residual-stream buffer + reserve KV blocks.
        for (int i = 0; i < N; i++)
        {
            EmbedToken(_gpuHidden, tokens[i]);
            _gpu.CopyDeviceRegion(_gpuStreamAll!, (long)i * embDim * sizeof(float),
                                  _gpuHidden, 0, (long)embDim * sizeof(float));
            _kvCache.ReserveBlockAt(startPos + i);
        }

        // Chunk-start reset for the GPU op-offload double-buffer: drain any stale prefetch
        // (handles freed/released) and reset to slot 0 so layer 0 of THIS chunk always uploads
        // synchronously into slot 0 rather than consuming a prefetch from the previous chunk.
        if (_gpuMoePrefill)
        {
            DrainPrefetch();
            _goCurSlot = 0;
            _goPrefetchedLayer = -1;
        }

        // 2. Trunk + batched MoE, layer by layer.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            long lt0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── Trunk: GEMM-batched over N tokens (issue #111) on the GPU-GDN path;
            //    sequential per token on the CPU-GDN debug path / when disabled.
            //    Both produce host _bResidAll / _bNormAll / _bSharedAll, then the
            //    batched MoE below runs identically. The batched trunk is bit-identical
            //    to the sequential one (same kernels, same per-row FP reduction).
            // #388: cleared here so the sequential-trunk path (which never issues the GPU
            // router GEMM) leaves it false → the host router loop below matvecs on the CPU.
            // TrunkLayerBatched sets it true per layer when it ran the batched router.
            _btRouterGpuValid = false;
            if (BatchedTrunkEnabled && !_cpuGdn)
            {
                TrunkLayerBatched(layer, N, startPos, isAttn, snapKvActive, wStart);
            }
            else
            {
                TrunkLayerSequential(layer, N, startPos, isAttn, snapKvActive, wStart);
            }
            long lt1 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── Router + shared-expert gate per token (host).
            var routerW = _cpuFfnGateInp![layer];
            float* gateInpShexp = _cpuFfnGateInpShexp![layer];
            for (int i = 0; i < N; i++)
            {
                float* normI = _bNormAll + (long)i * embDim;
                // #388: router logits from the GPU batched GEMM (TrunkLayerBatched downloaded
                // them into _btRouterAll) when available; else the per-token CPU matvec.
                // Softmax + top-k run on the host either way; the shexp dot below is unchanged
                // (it still reads normI from the host post-attn norm).
                float* logitsI;
                if (_btRouterGpuValid)
                {
                    logitsI = _btRouterAll + (long)i * _numExperts;
                    SimdKernels.SoftmaxInPlace(logitsI, _numExperts);
                }
                else
                {
                    logitsI = _cpuRouterLogits;
                    SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, normI,
                        _numExperts, embDim, routerW.DType);
                    SimdKernels.SoftmaxInPlace(_cpuRouterLogits, _numExperts);
                }
                SelectTopKPtr(logitsI, _numExperts, na, sel, wts, _hp.NormalizeMoeTopKWeights);
                for (int k = 0; k < na; k++)
                {
                    _bSelected[(long)i * na + k] = sel[k];
                    _bWeights[(long)i * na + k] = wts[k];
                }
                float dot = SimdKernels.DotF32(gateInpShexp, normI, embDim);
                _bShexpScale[i] = 1.0f / (1.0f + MathF.Exp(-dot));
            }

            long lt2 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── Batched routed experts: CPU grouped-dot (default, byte-exact) or
            //    GPU op-offload (transient weight upload + GEMM, argmax-stable). The op-offload
            //    only engages for large-enough batches (_gpuMoePrefillMinTokens) — tiny prefills
            //    go upload-bound on the ~14 GB whole-tensor upload and lose to the CPU path.
            if (_gpuMoePrefill && N >= _gpuMoePrefillMinTokens)
                BatchedRoutedExpertsGpuOffload(layer, N);
            else
                BatchedRoutedExperts(layer, N);

            long lt3 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── Combine on host: (routed + shared*scale) + resid, matching the
            //    sequential AddInPlace(moe, shared) then GPU AddInPlace(hidden, resid)
            //    operand order exactly. Upload the new residual stream to the GPU.
            Parallel.For(0, N, s_moeParallelOpts, i =>
            {
                float* routed = _bRoutedAll + (long)i * embDim;
                float* shared = _bSharedAll + (long)i * embDim;
                float* resid  = _bResidAll  + (long)i * embDim;
                float* outp   = _bHiddenAll + (long)i * embDim;
                float scale   = _bShexpScale[i];
                for (int r = 0; r < embDim; r++)
                    outp[r] = (routed[r] + shared[r] * scale) + resid[r];
            });
            _gpu.UploadInto(_gpuStreamAll!, (nint)_bHiddenAll, (int)((long)N * embDim));
            if (_prefillProfile)
            {
                long lt4 = System.Diagnostics.Stopwatch.GetTimestamp();
                trunkTicks   += lt1 - lt0;
                routerTicks  += lt2 - lt1;
                routedTicks  += lt3 - lt2;
                combineTicks += lt4 - lt3;
            }
        }

        // 3. Advance the position counters by N (block table already reserved).
        for (int i = 0; i < N; i++)
        {
            _kvCache.IncrementPosition();
            _gdnStateCache.IncrementPosition();
        }
        // Recurrent state + length counters are now consistent — clear the fault latch.
        _faulted = false;

        // 4. MTP hidden history: _bHiddenAll holds the pre-output-norm hidden for
        //    every token after the final layer — mirror into the absolute-position
        //    history buffer so PrefillMtp can read h_{p-1}.
        if (_hasMtp)
        {
            for (int i = 0; i < N; i++)
                new ReadOnlySpan<float>(_bHiddenAll + (long)i * embDim, embDim).CopyTo(
                    new Span<float>(_mtpPrefillHiddens + (long)(startPos + i) * embDim, embDim));
            if (_mtpHiddenHistoryLength < startPos + N)
                _mtpHiddenHistoryLength = startPos + N;

            // Last token's pre-output-norm hidden — the MTP decoder reads
            // LastHidden after prefill as the first draft's prevHidden. Mirror
            // Forward's _lastHidden population so batched prefill drives MTP too.
            new ReadOnlySpan<float>(_bHiddenAll + (long)(N - 1) * embDim, embDim).CopyTo(
                new Span<float>(_lastHidden, embDim));
        }

        // 5. Last token: output norm + lm_head → logits.
        _gpu.CopyDeviceRegion(_gpuHidden, 0, _gpuStreamAll!,
                              (long)(N - 1) * embDim * sizeof(float), (long)embDim * sizeof(float));
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);
        _gpu.Download(_gpuLogits, _logitsBuf);

        if (_prefillProfile)
        {
            double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            double total = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * f;
            Console.Error.WriteLine(
                $"[batched-prefill] N={N} total={total:F0}ms ({total / N:F1}ms/tok) " +
                $"trunk={trunkTicks * f:F0}ms router={routerTicks * f:F0}ms " +
                $"routedMoE={routedTicks * f:F0}ms combine={combineTicks * f:F0}ms");
            Console.Error.WriteLine(
                $"[moe-subphase] bucket={_profMoeBucket * f:F0}ms normQ={_profMoeNormQ * f:F0}ms " +
                $"phaseA(gate+up)={_profMoePhaseA * f:F0}ms silu/reduce={_profMoeSilu * f:F0}ms " +
                $"gateQ={_profMoeGateQ * f:F0}ms phaseC(down)={_profMoePhaseC * f:F0}ms");
            if (_gpuMoePrefill)
                Console.Error.WriteLine(
                    $"[gpu-offload] dequant={_profGoDequant * f:F0}ms upload={_profGoUpload * f:F0}ms " +
                    $"gather={_profGoGather * f:F0}ms gemm={_profGoGemm * f:F0}ms " +
                    $"download+scatter={_profGoDownScatter * f:F0}ms");
        }
        return _logitsBuf;
    }

    /// <summary>
    /// Batched-trunk prompt prefill for the non-CPU-MoE GDN-hybrid configs (issue #119):
    /// the dense FFN path (<c>!hp.IsMoE</c>, e.g. Qwen3.6-27B-MTP) and the GPU-SLRU MoE
    /// path (<c>_cpuMoe == false</c> on a model whose experts mostly fit VRAM, e.g.
    /// Qwen3-Coder-30B forced with <c>SHARPI_CPU_MOE=0</c>). The trunk (attention / GDN)
    /// runs as the same GEMM-batched + fused-scan + batched-query-SDPA launches the
    /// CPU-MoE path uses (<see cref="TrunkBlockBatched"/>), collapsing the per-token
    /// GDN/attn launches that dominate long-context prefill. The FFN/MoE stage then runs
    /// per token on the GPU exactly as in <see cref="Forward"/> — there is no routed-expert
    /// DRAM amortization here (that is CPU-MoE-specific, #110/#112), only the trunk half.
    ///
    /// <para>Produces bit-identical KV cache, GDN state, MTP hidden history, and last-token
    /// logits to the sequential per-token <see cref="Forward"/> loop: every batched trunk
    /// kernel runs the same per-row math as its single-token counterpart, and the FFN/MoE
    /// is the identical single-token kernel sequence over the (batched) post-attn norm.</para>
    ///
    /// <para><b>Not transactional</b> — same caveat as <see cref="PrefillBatchedCpuMoe"/>:
    /// the GDN scan/conv state and KV pages are mutated in place while the length counters
    /// are advanced only at the end, so a mid-chunk throw leaves poisoned recurrent state
    /// (<c>_faulted</c> latched). Such a failure is fatal for this pass.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillBatchedTrunkGpuFfn(IReadOnlyList<int> tokens, int startPos,
                                                          bool snapKvActive, int W, int wStart)
    {
        int N = tokens.Count;
        int embDim = _embDim;
        EnsureStreamAll(N);
        EnsureBatchedTrunkScratch(N);
        // Issue #121: device + host scratch for the batched FFN/MoE stage. Allocated only
        // when the batched FFN path can actually run on this model (MoE, or ≥1 GPU dense
        // FFN layer); the CPU-dense-fallback-only / CPU-MoE paths never touch it.
        if (BatchedFfnEnabled && (_hp.IsMoE || _denseFfnGpuLayers > 0))
            EnsureBatchedFfnScratch(N);

        long t0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        long trunkTicks = 0, ffnTicks = 0;

        // Pessimistic fault latch (see PrefillBatchedCpuMoe): clear only once the
        // length counters have been advanced consistently with the recurrent state.
        _faulted = true;

        // 1. Embed every token into the residual-stream buffer + reserve KV blocks.
        for (int i = 0; i < N; i++)
        {
            EmbedToken(_gpuHidden, tokens[i]);
            _gpu.CopyDeviceRegion(_gpuStreamAll!, (long)i * embDim * sizeof(float),
                                  _gpuHidden, 0, (long)embDim * sizeof(float));
            _kvCache.ReserveBlockAt(startPos + i);
        }

        bool isMoe = _hp.IsMoE;
        var stream = _gpuStreamAll!;

        // 2. Trunk (batched) + per-token FFN/MoE, layer by layer.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            long lt0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── Batched trunk block → _gpuBtBlockOut (resid) + _gpuBtMoeNorm (FFN input).
            TrunkBlockBatched(layer, N, startPos, isAttn, snapKvActive, wStart);
            var blockOut = _gpuBtBlockOut!;
            var moeNorm  = _gpuBtMoeNorm!;
            long lt1 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            // ── FFN / MoE stage. Issue #121: batch the GPU FFN/MoE over all N tokens
            //    when possible (dense GPU-FFN layers → GEMM-N gate/up/down; GPU-SLRU MoE
            //    → grouped-by-expert with a top-k-ordered per-token reduce). Both write
            //    the combined FFN output into _gpuBfHiddenAll [N × embDim], add the
            //    postBlock residual batched, and scatter the new residual stream in one
            //    copy. The per-token fallback (below) covers !BatchedFfnEnabled and the
            //    dense CPU-mmap-fallback layers (_gpuWFfnGate[layer] == null).
            bool denseGpuLayer = !isMoe && _gpuWFfnGate is not null && _gpuWFfnGate[layer] is not null;
            // Issue #121: only batch when every weight that would hit MatMulBatched is a
            // GEMM-N-supported dtype (Q4_K/Q5_K/Q6_K/F32). Unsupported dtypes (e.g. a Q8_0
            // router) fall back to the bit-exact per-token loop instead of faulting. Routed
            // expert weights are constrained to supported dtypes by the slot manager.
            bool batchLayer = BatchedFfnEnabled
                && (isMoe ? BatchedMatMulSupported(_gpuWGateInp[layer])
                          : denseGpuLayer
                            && BatchedMatMulSupported(_gpuWFfnGate![layer]!)
                            && BatchedMatMulSupported(_gpuWFfnUp![layer]!)
                            && BatchedMatMulSupported(_gpuWFfnDown![layer]!));
            if (batchLayer)
            {
                if (isMoe)
                    BatchedGpuMoeFfn(layer, N, moeNorm, _gpuBfHiddenAll!);
                else
                    BatchedGpuDenseFfn(layer, N, moeNorm, _gpuBfHiddenAll!);

                // Batched residual add (postBlock) + scatter the whole stream slice.
                _gpu.AddInPlace(_gpuBfHiddenAll!, blockOut);
                _gpu.CopyDeviceRegion(stream, 0, _gpuBfHiddenAll!, 0, (long)N * embDim * sizeof(float));
            }
            else
            {
                for (int i = 0; i < N; i++)
                {
                    _gpu.CopyDeviceRegion(_gpuNormBuf, 0, moeNorm, (long)i * embDim * sizeof(float),
                                          (long)embDim * sizeof(float));

                    if (!isMoe)
                    {
                        if (denseGpuLayer)
                        {
                            GpuDenseFfn(layer);
                        }
                        else
                        {
                            _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, embDim);
                            CpuDenseFfn(layer);
                            _gpu.UploadInto(_gpuHidden, (nint)_cpuMoeHidden, embDim);
                        }
                    }
                    else
                    {
                        GpuMoeFfn(layer);
                    }

                    _gpu.CopyDeviceRegion(_gpuResidual, 0, blockOut, (long)i * embDim * sizeof(float),
                                          (long)embDim * sizeof(float));
                    _gpu.AddInPlace(_gpuHidden, _gpuResidual);
                    _gpu.CopyDeviceRegion(stream, (long)i * embDim * sizeof(float),
                                          _gpuHidden, 0, (long)embDim * sizeof(float));
                }
            }
            if (_prefillProfile)
            {
                long lt2 = System.Diagnostics.Stopwatch.GetTimestamp();
                trunkTicks += lt1 - lt0;
                ffnTicks   += lt2 - lt1;
            }
        }

        // 3. Advance the position counters by N (block table already reserved).
        for (int i = 0; i < N; i++)
        {
            _kvCache.IncrementPosition();
            _gdnStateCache.IncrementPosition();
        }
        _faulted = false;

        // 4. MTP hidden history: after the final layer, stream[i] holds the
        //    pre-output-norm hidden for token startPos+i. Mirror into the
        //    absolute-position history so PrefillMtp can read h_{p-1}.
        if (_hasMtp)
        {
            EnsureMtpHiddenHistoryCap(startPos + N);
            _gpu.Download(stream, (nint)(_mtpPrefillHiddens + (long)startPos * embDim), (int)((long)N * embDim));
            if (_mtpHiddenHistoryLength < startPos + N)
                _mtpHiddenHistoryLength = startPos + N;
            new ReadOnlySpan<float>(_mtpPrefillHiddens + (long)(startPos + N - 1) * embDim, embDim).CopyTo(
                new Span<float>(_lastHidden, embDim));
        }

        // 5. Last token: output norm + lm_head → logits.
        _gpu.CopyDeviceRegion(_gpuHidden, 0, stream,
                              (long)(N - 1) * embDim * sizeof(float), (long)embDim * sizeof(float));
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);
        _gpu.Download(_gpuLogits, _logitsBuf);

        if (_prefillProfile)
        {
            double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            double total = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * f;
            Console.Error.WriteLine(
                $"[batched-prefill-gpuffn] N={N} total={total:F0}ms ({total / N:F1}ms/tok) " +
                $"trunk={trunkTicks * f:F0}ms ffn={ffnTicks * f:F0}ms");
        }
        return _logitsBuf;
    }

    /// <summary>
    /// Batched routed-MoE FFN for <paramref name="N"/> prompt tokens at one layer.
    /// Groups tokens by selected expert so each expert's gate/up/down weight rows are
    /// read once per layer and dotted against every token routing to it (instead of
    /// per-token re-reads — the DRAM bottleneck the sequential path hits). Output is
    /// written to <see cref="_bRoutedAll"/>. Byte-parity with the per-token
    /// <see cref="CpuMoeFfnCore"/> is preserved: identical dot kernels, identical
    /// per-token top-k accumulation order in the final reduce.
    /// </summary>
    private void BatchedRoutedExperts(int layer, int N)
    {
        int embDim = _embDim;
        int na = _numActiveExperts;
        int expertDim = _expertDim;
        int numExperts = _numExperts;

        var gateExps = _cpuFfnGateExps![layer];
        var upExps   = _cpuFfnUpExps![layer];
        var downExps = _cpuFfnDownExps![layer];
        byte* gateP = gateExps.DataPtr; byte* upP = upExps.DataPtr; byte* downP = downExps.DataPtr;
        DType gateDt = gateExps.DType, upDt = upExps.DType, downDt = downExps.DType;

        int bprG = (embDim    / DTypeInfo.BlockSize(gateDt)) * DTypeInfo.BytesPerBlock(gateDt);
        int bprU = (embDim    / DTypeInfo.BlockSize(upDt))   * DTypeInfo.BytesPerBlock(upDt);
        int bprD = (expertDim / DTypeInfo.BlockSize(downDt)) * DTypeInfo.BytesPerBlock(downDt);

        bool useQ8KGate = (_q3kQ8KEnabled && gateDt == DType.Q3_K) || (_q8_0Q8KEnabled && gateDt == DType.Q8_0) || (_q4kQ8KEnabled && gateDt == DType.Q4_K);
        bool useQ8KUp   = (_q3kQ8KEnabled && upDt   == DType.Q3_K) || (_q8_0Q8KEnabled && upDt   == DType.Q8_0) || (_q4kQ8KEnabled && upDt   == DType.Q4_K);
        bool useQ8KDown = (_q3kQ8KEnabled && downDt == DType.Q3_K) || (_q8_0Q8KEnabled && downDt == DType.Q8_0) || (_q4kQ8KEnabled && downDt == DType.Q4_K);

        long sp0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        // Bucket (token, slot) pairs by selected expert (CSR layout).
        int* expStart = _bExpStart!; int* cursor = _bExpCursor!; int* used = _bUsedExperts!;
        int* selected = _bSelected!;
        for (int e = 0; e <= numExperts; e++) expStart[e] = 0;
        long totalSel = (long)N * na;
        for (long s = 0; s < totalSel; s++) expStart[selected[s] + 1]++;
        for (int e = 0; e < numExperts; e++) expStart[e + 1] += expStart[e];
        for (int e = 0; e < numExperts; e++) cursor[e] = expStart[e];
        int* expTokI = _bExpTokI!; int* expTokK = _bExpTokK!;
        for (int i = 0; i < N; i++)
            for (int k = 0; k < na; k++)
            {
                int e = selected[(long)i * na + k];
                int p = cursor[e]++;
                expTokI[p] = i; expTokK[p] = k;
            }
        int numUsed = 0;
        for (int e = 0; e < numExperts; e++)
            if (expStart[e + 1] > expStart[e]) used[numUsed++] = e;

        float* gateAll = _bGateAll!; float* upAll = _bUpAll!; float* downPartial = _bDownPartial!;
        float* normAll = _bNormAll;
        byte* normAllQ8K = _bNormAllQ8K; byte* gateAllQ8K = _bGateAllQ8K;
        int q8kEmbStride = _bQ8KEmbStride, q8kExpStride = _bQ8KExpStride;

        long sp1 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoeBucket += sp1 - sp0;
        // Q8_KS-prepack each token's norm once (shared across all gate/up rows).
        if (useQ8KGate || useQ8KUp)
            Parallel.For(0, N, s_moeParallelOpts, i =>
                SimdKernels.QuantizeRowToQ8KS(normAll + (long)i * embDim, embDim,
                    normAllQ8K + (long)i * q8kEmbStride));
        long sp2 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoeNormQ += sp2 - sp1;

        // Phase A: gate + up. Parallelize over (used expert, expert-row); read each
        // weight row once, dot against every token routing to this expert.
        int naL = na, expertDimL = expertDim, embDimL = embDim;
        Parallel.For(0, numUsed * expertDim, s_moeParallelOpts, idx =>
        {
            int u = idx / expertDimL;
            int r = idx % expertDimL;
            int e = used[u];
            byte* gateRow = gateP + (long)e * expertDimL * bprG + (long)r * bprG;
            byte* upRow   = upP   + (long)e * expertDimL * bprU + (long)r * bprU;
            int pStart = expStart[e], pEnd = expStart[e + 1];
            // Issue #114: dot each gate/up row against the expert's tokens in QUADS,
            // decoding the (Q4_K/Q3_K) weight row once per quad (decode/4). Then mop up
            // remaining tokens in PAIRS (issue #112, decode/2) and a final single.
            // Every tier is bit-identical to the per-token dot (the 4In/2In kernels
            // mirror the single accumulation order) — only the unpack is amortized.
            int p = pStart;
            for (; p + 3 < pEnd; p += 4)
            {
                int i0 = expTokI[p],     k0 = expTokK[p];
                int i1 = expTokI[p + 1], k1 = expTokK[p + 1];
                int i2 = expTokI[p + 2], k2 = expTokK[p + 2];
                int i3 = expTokI[p + 3], k3 = expTokK[p + 3];
                long o0 = ((long)i0 * naL + k0) * expertDimL + r;
                long o1 = ((long)i1 * naL + k1) * expertDimL + r;
                long o2 = ((long)i2 * naL + k2) * expertDimL + r;
                long o3 = ((long)i3 * naL + k3) * expertDimL + r;
                float a0, a1, a2, a3;
                if (useQ8KGate)
                    DispatchDotQ8K4In(gateRow, normAllQ8K + (long)i0 * q8kEmbStride,
                        normAllQ8K + (long)i1 * q8kEmbStride, normAllQ8K + (long)i2 * q8kEmbStride,
                        normAllQ8K + (long)i3 * q8kEmbStride, embDimL, gateDt, out a0, out a1, out a2, out a3);
                else
                    DispatchDot4In(gateRow, normAll + (long)i0 * embDimL, normAll + (long)i1 * embDimL,
                        normAll + (long)i2 * embDimL, normAll + (long)i3 * embDimL, embDimL, gateDt,
                        out a0, out a1, out a2, out a3);
                gateAll[o0] = a0; gateAll[o1] = a1; gateAll[o2] = a2; gateAll[o3] = a3;
                if (useQ8KUp)
                    DispatchDotQ8K4In(upRow, normAllQ8K + (long)i0 * q8kEmbStride,
                        normAllQ8K + (long)i1 * q8kEmbStride, normAllQ8K + (long)i2 * q8kEmbStride,
                        normAllQ8K + (long)i3 * q8kEmbStride, embDimL, upDt, out a0, out a1, out a2, out a3);
                else
                    DispatchDot4In(upRow, normAll + (long)i0 * embDimL, normAll + (long)i1 * embDimL,
                        normAll + (long)i2 * embDimL, normAll + (long)i3 * embDimL, embDimL, upDt,
                        out a0, out a1, out a2, out a3);
                upAll[o0] = a0; upAll[o1] = a1; upAll[o2] = a2; upAll[o3] = a3;
            }
            for (; p + 1 < pEnd; p += 2)
            {
                int i0 = expTokI[p],     k0 = expTokK[p];
                int i1 = expTokI[p + 1], k1 = expTokK[p + 1];
                long o0 = ((long)i0 * naL + k0) * expertDimL + r;
                long o1 = ((long)i1 * naL + k1) * expertDimL + r;
                float a0, a1;
                if (useQ8KGate)
                    DispatchDotQ8K2In(gateRow, normAllQ8K + (long)i0 * q8kEmbStride,
                        normAllQ8K + (long)i1 * q8kEmbStride, embDimL, gateDt, out a0, out a1);
                else
                    DispatchDot2In(gateRow, normAll + (long)i0 * embDimL,
                        normAll + (long)i1 * embDimL, embDimL, gateDt, out a0, out a1);
                gateAll[o0] = a0; gateAll[o1] = a1;
                if (useQ8KUp)
                    DispatchDotQ8K2In(upRow, normAllQ8K + (long)i0 * q8kEmbStride,
                        normAllQ8K + (long)i1 * q8kEmbStride, embDimL, upDt, out a0, out a1);
                else
                    DispatchDot2In(upRow, normAll + (long)i0 * embDimL,
                        normAll + (long)i1 * embDimL, embDimL, upDt, out a0, out a1);
                upAll[o0] = a0; upAll[o1] = a1;
            }
            if (p < pEnd) // odd remainder
            {
                int i = expTokI[p], k = expTokK[p];
                long outIdx = ((long)i * naL + k) * expertDimL + r;
                gateAll[outIdx] = useQ8KGate
                    ? DispatchDotQ8K(gateRow, normAllQ8K + (long)i * q8kEmbStride, embDimL, gateDt)
                    : DispatchDot(gateRow, normAll + (long)i * embDimL, embDimL, gateDt);
                upAll[outIdx] = useQ8KUp
                    ? DispatchDotQ8K(upRow, normAllQ8K + (long)i * q8kEmbStride, embDimL, upDt)
                    : DispatchDot(upRow, normAll + (long)i * embDimL, embDimL, upDt);
            }
        });

        long sp3 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoePhaseA += sp3 - sp2;
        // Phase B: SiLU(gate) * up over the whole contiguous (token × slot × expertDim) block.
        SimdKernels.SiLuMul(gateAll, upAll, (int)(totalSel * expertDim));
        long sp4 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoeSilu += sp4 - sp3;

        // Q8_KS-prepack each silu'd gate slice for the down dots.
        if (useQ8KDown)
            Parallel.For(0, (int)totalSel, s_moeParallelOpts, s =>
                SimdKernels.QuantizeRowToQ8KS(gateAll + (long)s * expertDim, expertDim,
                    gateAllQ8K + (long)s * q8kExpStride));
        long sp5 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoeGateQ += sp5 - sp4;

        // Phase C: down. Parallelize over (used expert, emb-row); read each down row
        // once, dot against every token's silu'd gate. Store unweighted partials.
        Parallel.For(0, numUsed * embDim, s_moeParallelOpts, idx =>
        {
            int u = idx / embDimL;
            int r = idx % embDimL;
            int e = used[u];
            byte* downRow = downP + (long)e * embDimL * bprD + (long)r * bprD;
            int pStart = expStart[e], pEnd = expStart[e + 1];
            // Issue #114: down row dotted against the expert's silu'd-gate slices in
            // QUADS (decode/4), then PAIRS (issue #112, decode/2), then a final single.
            // Every tier is bit-identical to the per-token dot.
            int p = pStart;
            for (; p + 3 < pEnd; p += 4)
            {
                long s0 = (long)expTokI[p]     * naL + expTokK[p];
                long s1 = (long)expTokI[p + 1] * naL + expTokK[p + 1];
                long s2 = (long)expTokI[p + 2] * naL + expTokK[p + 2];
                long s3 = (long)expTokI[p + 3] * naL + expTokK[p + 3];
                float d0, d1, d2, d3;
                if (useQ8KDown)
                    DispatchDotQ8K4In(downRow, gateAllQ8K + s0 * q8kExpStride,
                        gateAllQ8K + s1 * q8kExpStride, gateAllQ8K + s2 * q8kExpStride,
                        gateAllQ8K + s3 * q8kExpStride, expertDimL, downDt, out d0, out d1, out d2, out d3);
                else
                    DispatchDot4In(downRow, gateAll + s0 * expertDimL, gateAll + s1 * expertDimL,
                        gateAll + s2 * expertDimL, gateAll + s3 * expertDimL, expertDimL, downDt,
                        out d0, out d1, out d2, out d3);
                downPartial[s0 * embDimL + r] = d0;
                downPartial[s1 * embDimL + r] = d1;
                downPartial[s2 * embDimL + r] = d2;
                downPartial[s3 * embDimL + r] = d3;
            }
            for (; p + 1 < pEnd; p += 2)
            {
                long s0 = (long)expTokI[p]     * naL + expTokK[p];
                long s1 = (long)expTokI[p + 1] * naL + expTokK[p + 1];
                float d0, d1;
                if (useQ8KDown)
                    DispatchDotQ8K2In(downRow, gateAllQ8K + s0 * q8kExpStride,
                        gateAllQ8K + s1 * q8kExpStride, expertDimL, downDt, out d0, out d1);
                else
                    DispatchDot2In(downRow, gateAll + s0 * expertDimL,
                        gateAll + s1 * expertDimL, expertDimL, downDt, out d0, out d1);
                downPartial[s0 * embDimL + r] = d0;
                downPartial[s1 * embDimL + r] = d1;
            }
            if (p < pEnd) // odd remainder
            {
                long slot = (long)expTokI[p] * naL + expTokK[p];
                downPartial[slot * embDimL + r] = useQ8KDown
                    ? DispatchDotQ8K(downRow, gateAllQ8K + slot * q8kExpStride, expertDimL, downDt)
                    : DispatchDot(downRow, gateAll + slot * expertDimL, expertDimL, downDt);
            }
        });
        long sp6 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (_prefillProfile) _profMoePhaseC += sp6 - sp5;

        // Phase C reduce: per token, sum the numActive weighted down partials in
        // top-k order — bit-identical to the sequential `sum += w * dot` loop.
        float* weights = _bWeights!; float* routedAll = _bRoutedAll!;
        Parallel.For(0, N, s_moeParallelOpts, i =>
        {
            float* outp = routedAll + (long)i * embDimL;
            for (int r = 0; r < embDimL; r++)
            {
                float sum = 0f;
                for (int k = 0; k < naL; k++)
                {
                    long slot = (long)i * naL + k;
                    sum += weights[slot] * downPartial[slot * embDimL + r];
                }
                outp[r] = sum;
            }
        });
        // Fold the top-k reduce into the silu/reduce bucket (both are cheap element-wise passes).
        if (_prefillProfile) _profMoeSilu += System.Diagnostics.Stopwatch.GetTimestamp() - sp6;
    }

    /// <summary>
    /// GPU op-offload sibling of <see cref="BatchedRoutedExperts"/> (perf/carnice-vnni-moe).
    /// Same CSR bucketing + same UNWEIGHTED partial scatter + same weighted reduce, but the
    /// gate/up/down matmuls run on the GPU: <see cref="_bNormAll"/> is uploaded once per layer
    /// call, then ONE <see cref="CudaBackend.GatherRows"/> launch fills the whole
    /// <c>[totalSel × embDim]</c> CSR-ordered gathered-norm buffer (<see cref="_gpuOffGather"/>),
    /// and for each used expert its host-resident gate/up/down weight is transiently uploaded
    /// into a reused GPU buffer and the GEMM-N gate/up → SiLuMul → GEMM-N down runs over that
    /// expert's contiguous CSR slice (the <see cref="BatchedGpuMoeFfn"/> structure, but with a
    /// transient upload in place of the resident SLRU slab). Raw-quant dtypes (Q4_K/Q5_K/
    /// Q6_K/Q8_0) upload raw bytes and dispatch the quantized GEMM via <see cref="GpuMatMulBatched"/>;
    /// Q3_K and Float32 dequantize on the host into <see cref="_hGpuOffDeq"/> and upload F32
    /// (<see cref="CudaBackend.MatMulBatched"/> has no Q3_K kernel). The CSR-ordered down output
    /// (<see cref="_gpuOffDownCsr"/>) is scattered into <see cref="_gpuOffDownPartial"/> by ONE
    /// <see cref="CudaBackend.ScatterRows"/> launch (no per-row CopyDeviceRegion), then a single
    /// GPU weighted reduce (<see cref="CudaBackend.MoeWeightedReduce"/> over a zeroed
    /// <see cref="_gpuOffRouted"/>) produces the routed sum, which is downloaded ONCE into
    /// <see cref="_bRoutedAll"/> — eliminating the ~256·layers tiny per-expert downloads + host
    /// scatter. The downstream combine is untouched. Argmax-stable, NOT byte-exact (gated off).
    /// </summary>
    private void BatchedRoutedExpertsGpuOffload(int layer, int N)
    {
        int embDim = _embDim;
        int na = _numActiveExperts;
        int expertDim = _expertDim;
        int numExperts = _numExperts;

        var gateExps = _cpuFfnGateExps![layer];
        var upExps   = _cpuFfnUpExps![layer];
        var downExps = _cpuFfnDownExps![layer];
        byte* gateP = gateExps.DataPtr; byte* upP = upExps.DataPtr; byte* downP = downExps.DataPtr;
        DType gateDt = gateExps.DType, upDt = upExps.DType, downDt = downExps.DType;

        // Raw bytes per expert weight (gate/up are [expertDim × embDim]; down is
        // [embDim × expertDim] — both expertDim·embDim elements, same per-block layout).
        int bprG = (embDim    / DTypeInfo.BlockSize(gateDt)) * DTypeInfo.BytesPerBlock(gateDt);
        int bprU = (embDim    / DTypeInfo.BlockSize(upDt))   * DTypeInfo.BytesPerBlock(upDt);
        int bprD = (expertDim / DTypeInfo.BlockSize(downDt)) * DTypeInfo.BytesPerBlock(downDt);
        long gateRawBytes = (long)expertDim * bprG;   // whole gate matrix per expert
        long upRawBytes   = (long)expertDim * bprU;
        long downRawBytes = (long)embDim    * bprD;

        EnsureGpuOffloadScratch(N);

        long sp0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        // Bucket (token, slot) pairs by selected expert (CSR layout) — IDENTICAL to
        // BatchedRoutedExperts so the partial scatter + weighted reduce line up.
        int* expStart = _bExpStart!; int* cursor = _bExpCursor!; int* used = _bUsedExperts!;
        int* selected = _bSelected!;
        for (int e = 0; e <= numExperts; e++) expStart[e] = 0;
        long totalSel = (long)N * na;
        for (long s = 0; s < totalSel; s++) expStart[selected[s] + 1]++;
        for (int e = 0; e < numExperts; e++) expStart[e + 1] += expStart[e];
        for (int e = 0; e < numExperts; e++) cursor[e] = expStart[e];
        int* expTokI = _bExpTokI!; int* expTokK = _bExpTokK!;
        for (int i = 0; i < N; i++)
            for (int k = 0; k < na; k++)
            {
                int e = selected[(long)i * na + k];
                int p = cursor[e]++;
                expTokI[p] = i; expTokK[p] = k;
            }
        int numUsed = 0;
        for (int e = 0; e < numExperts; e++)
            if (expStart[e + 1] > expStart[e]) used[numUsed++] = e;
        if (_prefillProfile) _profGoUpload += System.Diagnostics.Stopwatch.GetTimestamp() - sp0;

        // Upload the host norm activations [N × embDim] once per layer call. The scratch is
        // grow-only (sized for _goCap ≥ N), so view it to exactly N·embDim for the pinned
        // upload (which requires floatCount == dst.ElementCount).
        long su0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        var normAll = _gpu.View(_gpuOffNorm!, 0, (long)N * embDim);
        _gpu.UploadInto(normAll, (nint)_bNormAll, N * embDim);
        if (_prefillProfile) _profGoUpload += System.Diagnostics.Stopwatch.GetTimestamp() - su0;
        var downPartialDev = _gpuOffDownPartial!;   // [N × na × embDim] device, GPU scatter target

        // ── Build + upload the CSR gather/scatter index arrays (one launch each below). ──
        // The CSR bucketing already orders the totalSel selections by expert, so expTokI[p]
        // is exactly the source token row for gathered position p, and expTokI[p]*na+expTokK[p]
        // is the (token,slot) partial slot p scatters into. Both index arrays are int32 device
        // buffers; the gather indices ARE _bExpTokI[0..totalSel), uploaded raw verbatim.
        long sg0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        for (long p = 0; p < totalSel; p++)
            _hGpuOffScatterIdx![p] = expTokI[p] * na + expTokK[p];
        var gatherIdxV  = _gpu.View(_gpuOffGatherIdx!,  0, totalSel, DType.Int32);
        var scatterIdxV = _gpu.View(_gpuOffScatterIdx!, 0, totalSel, DType.Int32);
        _gpu.UploadRawInto(gatherIdxV,  new ReadOnlySpan<byte>(expTokI,           (int)(totalSel * sizeof(int))));
        _gpu.UploadRawInto(scatterIdxV, new ReadOnlySpan<byte>(_hGpuOffScatterIdx, (int)(totalSel * sizeof(int))));

        // ONE gather: fill the whole [totalSel × embDim] CSR-ordered gathered-norm buffer.
        _gpu.GatherRows(_gpuOffGather!, normAll, gatherIdxV, (int)totalSel, embDim);
        _gpu.Free(gatherIdxV);
        if (_prefillProfile) _profGoGather += System.Diagnostics.Stopwatch.GetTimestamp() - sg0;

        // ── Whole-layer batched weight upload (raw-quant only), double-buffered ──────────
        // The host expert weights are contiguous per layer: gateP/upP/downP each point at the
        // WHOLE layer's ffn_*_exps tensor (all numExperts experts back-to-back). For raw-quant
        // dtypes (Q4_K/Q5_K/Q6_K/Q8_0) the bytes upload unchanged, so do ONE big contiguous
        // transfer per weight into the CURRENT ping-pong slot — the per-expert matrix is then a
        // non-owning ViewRawBytes into that slot. (One ~tens-of-MB transfer is bandwidth-bound;
        // thousands of tiny ones from non-pinned mmap are latency-bound — the upload floor.)
        // Only Float32 keeps the per-expert host-dequant fallback below.
        //
        // When the expert weights are in the pinned cudaMallocHost buffer (_goHostPinned):
        //   • If THIS layer was prefetched into the opposite slot by the previous call, just
        //     WaitForUpload its 3 DMA handles (cross-stream fence so the GEMMs can't race the
        //     copy), release them, and adopt that slot — no synchronous upload at all.
        //   • Otherwise DMA THIS layer synchronously into _goCurSlot from the pinned buffer
        //     (first MoE layer of the chunk) — direct, full-bandwidth, then host-wait.
        //   • Then issue a DIRECT async DMA of layer+1's weights into the OTHER slot
        //     (UploadRawIntoAsyncDirect straight from the pinned buffer — no staging), overlapping
        //     with this layer's GEMMs. routed-MoE ≈ max(upload, compute) instead of their sum.
        // When NOT pinned this is byte-identical to the old single-buffer synchronous path
        // (slot stays 0, no prefetch ever issued; UploadLayerRaw straight from the mmap).
        bool layerGate = IsRawOffloadQuant(gateDt);
        bool layerUp   = IsRawOffloadQuant(upDt);
        bool layerDown = IsRawOffloadQuant(downDt);
        long swu0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // Per-layer pinned-buffer bases (null for a Float32 layer or when not pinned → mmap path).
        byte* pinGate = _goPinnedGate != null ? _goPinnedGate[layer] : null;
        byte* pinUp   = _goPinnedUp   != null ? _goPinnedUp[layer]   : null;
        byte* pinDown = _goPinnedDown != null ? _goPinnedDown[layer] : null;
        bool layerPinned = _goHostPinned && pinGate != null && pinUp != null && pinDown != null;

        bool consumedPrefetch = _goHostPinned && _goPrefetchValid && _goPrefetchedLayer == layer;
        if (consumedPrefetch)
        {
            // THIS layer's weights are already (being) DMA'd into _goPrefetchSlot. Fence the
            // compute stream behind those copies, release the handles, and read from that slot.
            if (layerGate) _gpu.WaitForUpload(_goPrefetchGateH);
            if (layerUp)   _gpu.WaitForUpload(_goPrefetchUpH);
            if (layerDown) _gpu.WaitForUpload(_goPrefetchDownH);
            _gpu.ReleaseUploadHandle(_goPrefetchGateH);
            _gpu.ReleaseUploadHandle(_goPrefetchUpH);
            _gpu.ReleaseUploadHandle(_goPrefetchDownH);
            _goPrefetchValid = false;
            _goCurSlot = _goPrefetchSlot;
        }
        else if (layerPinned)
        {
            // First MoE layer of the chunk (or any non-prefetched raw-quant layer): DMA straight
            // from the pinned buffer into the current slot, then FENCE the compute stream behind
            // the copies (not a host block) so the GEMMs below (on _stream) read complete weights
            // while the CPU keeps launching kernels — mirrors the consumedPrefetch branch (Gemini
            // review). Direct (no staging) + full bandwidth. Waits stay unconditional to match the
            // three unconditional DMAs just issued.
            var gh = _gpu.UploadRawIntoAsyncDirect(_gpuLayerGate[_goCurSlot]!, pinGate, (long)numExperts * gateRawBytes);
            var uh = _gpu.UploadRawIntoAsyncDirect(_gpuLayerUp[_goCurSlot]!,   pinUp,   (long)numExperts * upRawBytes);
            var dh = _gpu.UploadRawIntoAsyncDirect(_gpuLayerDown[_goCurSlot]!, pinDown, (long)numExperts * downRawBytes);
            _gpu.WaitForUpload(gh); _gpu.WaitForUpload(uh); _gpu.WaitForUpload(dh);
            _gpu.ReleaseUploadHandle(gh); _gpu.ReleaseUploadHandle(uh); _gpu.ReleaseUploadHandle(dh);
        }
        else
        {
            // Not pinned (or Float32 layer): synchronous whole-layer upload straight from the
            // mmap into the current slot — the original op-offload path (byte-identical fallback).
            if (layerGate) UploadLayerRaw(_gpuLayerGate[_goCurSlot]!, gateP, (long)numExperts * gateRawBytes);
            if (layerUp)   UploadLayerRaw(_gpuLayerUp[_goCurSlot]!,   upP,   (long)numExperts * upRawBytes);
            if (layerDown) UploadLayerRaw(_gpuLayerDown[_goCurSlot]!, downP, (long)numExperts * downRawBytes);
        }
        // Tag the live-slot layer buffers so ResolveExpertWeight/GpuMatMulBatched dispatch on
        // the raw dtype (re-tag every call — the slot's handle persists but the tag could have
        // been dropped by a prior FreeExpertWeightView on the recycled handle space).
        Tensor liveGate = _gpuLayerGate[_goCurSlot]!;
        Tensor liveUp   = _gpuLayerUp[_goCurSlot]!;
        Tensor liveDown = _gpuLayerDown[_goCurSlot]!;
        if (layerGate) _gpuWeightDTypes[liveGate.Handle] = gateDt;
        if (layerUp)   _gpuWeightDTypes[liveUp.Handle]   = upDt;
        if (layerDown) _gpuWeightDTypes[liveDown.Handle] = downDt;

        // ── Prefetch layer+1 into the OTHER slot (direct DMA from the pinned buffer). ──
        // Only when pinned and a next MoE layer exists. The prefetch targets the opposite slot
        // so this layer's reads (from _goCurSlot) never see a half-written buffer. A layer is
        // prefetched only if all three of its weights are raw-quant AND were copied into the
        // pinned buffer (Float32 takes the host-dequant fallback, which the next call does
        // synchronously) — so the consume branch above can adopt the slot wholesale.
        if (_goHostPinned && layer + 1 < _hp.NumLayers)
        {
            int nl = layer + 1;
            byte* npGate = _goPinnedGate![nl], npUp = _goPinnedUp![nl], npDown = _goPinnedDown![nl];
            DType ngDt = _cpuFfnGateExps![nl].DType, nuDt = _cpuFfnUpExps![nl].DType, ndDt = _cpuFfnDownExps![nl].DType;
            if (npGate != null && npUp != null && npDown != null
                && IsRawOffloadQuant(ngDt) && IsRawOffloadQuant(nuDt) && IsRawOffloadQuant(ndDt))
            {
                int nbprG = (embDim    / DTypeInfo.BlockSize(ngDt)) * DTypeInfo.BytesPerBlock(ngDt);
                int nbprU = (embDim    / DTypeInfo.BlockSize(nuDt)) * DTypeInfo.BytesPerBlock(nuDt);
                int nbprD = (expertDim / DTypeInfo.BlockSize(ndDt)) * DTypeInfo.BytesPerBlock(ndDt);
                long ngBytes = (long)numExperts * expertDim * nbprG;
                long nuBytes = (long)numExperts * expertDim * nbprU;
                long ndBytes = (long)numExperts * embDim    * nbprD;
                int otherSlot = _goCurSlot ^ 1;
                _goPrefetchGateH = _gpu.UploadRawIntoAsyncDirect(_gpuLayerGate[otherSlot]!, npGate, ngBytes);
                _goPrefetchUpH   = _gpu.UploadRawIntoAsyncDirect(_gpuLayerUp[otherSlot]!,   npUp,   nuBytes);
                _goPrefetchDownH = _gpu.UploadRawIntoAsyncDirect(_gpuLayerDown[otherSlot]!, npDown, ndBytes);
                _goPrefetchSlot = otherSlot;
                _goPrefetchedLayer = nl;
                _goPrefetchValid = true;
            }
            else
            {
                // Next layer has a Float32 weight → no prefetch; it'll upload synchronously.
                _goPrefetchedLayer = -1;
            }
        }
        if (_prefillProfile) _profGoUpload += System.Diagnostics.Stopwatch.GetTimestamp() - swu0;

        for (int u = 0; u < numUsed; u++)
        {
            int e = used[u];
            int pStart = expStart[e], pEnd = expStart[e + 1];
            int cnt = pEnd - pStart;
            if (cnt == 0) continue;

            // a. Resolve each weight matrix for this expert. Raw-quant dtypes are a non-owning
            //    ViewRawBytes into the whole-layer buffer uploaded once above (no per-expert
            //    transfer). Float32 still host-dequant→F32 into the per-expert _gpuOffW* buffer
            //    (UploadOffloadWeight). The matrix tensor passed to the GEMM is the same in both
            //    cases; views additionally carry an entry in _gpuWeightDTypes so
            //    GpuMatMulBatched dispatches on the raw dtype.
            long sd0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            Tensor gateW = ResolveExpertWeight(layerGate, liveGate, _gpuOffWGate!,
                gateP, e, gateRawBytes, gateDt, expertDim, embDim, ref sd0);
            Tensor upW = ResolveExpertWeight(layerUp, liveUp, _gpuOffWUp!,
                upP, e, upRawBytes, upDt, expertDim, embDim, ref sd0);
            Tensor downW = ResolveExpertWeight(layerDown, liveDown, _gpuOffWDown!,
                downP, e, downRawBytes, downDt, embDim, expertDim, ref sd0);
            if (_prefillProfile) _profGoUpload += System.Diagnostics.Stopwatch.GetTimestamp() - sd0;

            // b. GEMM-N gate/up over this expert's CONTIGUOUS CSR slice [pStart..pEnd) of the
            //    pre-gathered norms → SiLuMul → GEMM-N down, writing each output CSR-ordered
            //    into its [pStart..pEnd) slice. No per-row gather/scatter inside the loop.
            long sm0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
            var normV = _gpu.View(_gpuOffGather!,  (long)pStart * embDim,    (long)cnt * embDim);
            var gateV = _gpu.View(_gpuOffGate!,    (long)pStart * expertDim, (long)cnt * expertDim);
            var upV   = _gpu.View(_gpuOffUp!,      (long)pStart * expertDim, (long)cnt * expertDim);
            var downV = _gpu.View(_gpuOffDownCsr!, (long)pStart * embDim,    (long)cnt * embDim);
            try
            {
                GpuMatMulBatched(gateV, gateW, normV, cnt);
                GpuMatMulBatched(upV,   upW,   normV, cnt);
                _gpu.SiLuMul(gateV, upV);
                GpuMatMulBatched(downV, downW, gateV, cnt);
            }
            finally
            {
                _gpu.Free(normV); _gpu.Free(gateV); _gpu.Free(upV); _gpu.Free(downV);
                // Per-expert weight views are non-owning (Free drops only the handle/devptr
                // registration; the layer buffer keeps the memory) — but also drop the
                // engine-side dtype tag so the recycled handle doesn't leak an entry.
                FreeExpertWeightView(layerGate, gateW);
                FreeExpertWeightView(layerUp,   upW);
                FreeExpertWeightView(layerDown, downW);
            }
            if (_prefillProfile) _profGoGemm += System.Diagnostics.Stopwatch.GetTimestamp() - sm0;
        }
        _gpu.Free(normAll);   // drop the norm view (non-owning; underlying buffer kept)

        // ONE scatter: the [totalSel × embDim] CSR down output → the per-(token,slot) partials
        // (gathered position p → slot expTokI[p]*na+expTokK[p]). Each slot is written exactly
        // once (the CSR covers every (token,slot) selection once — no atomics needed). Folded
        // into the download+scatter bucket below.
        long sx0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        _gpu.ScatterRows(downPartialDev, _gpuOffDownCsr!, scatterIdxV, (int)totalSel, embDim);
        _gpu.Free(scatterIdxV);
        if (_prefillProfile) _profGoDownScatter += System.Diagnostics.Stopwatch.GetTimestamp() - sx0;

        // Weighted reduce ON THE GPU: routedAll[i,r] = Σ_k weights[i,k]·downPartial[(i*na+k),r].
        // Reuse BatchedGpuMoeFfn's Phase-3 reduce (CudaHybridGdnForwardPass.cs:5244-5245):
        // upload the host top-k weights once, then ONE MoeWeightedReduce launch. That kernel
        // also adds its `shared` operand (acc += shared[i*embDim+e]); the offload path has no
        // shared term here (the downstream combine adds shared*scale separately), so we zero
        // _gpuOffRouted first → acc += 0, leaving the pure routed weighted sum. The result is
        // downloaded ONCE into _bRoutedAll — the single D2H replacing the old per-expert ones.
        long sr0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        var routedDev = _gpu.View(_gpuOffRouted!, 0, (long)N * embDim);
        var weightsDev = _gpu.View(_gpuOffWeightsDev!, 0, (long)N * na);
        try
        {
            _gpu.ClearRegion(routedDev, 0, N * embDim);
            _gpu.UploadInto(weightsDev, new ReadOnlySpan<float>(_bWeights!, (int)((long)N * na)));
            _gpu.MoeWeightedReduce(downPartialDev, weightsDev, routedDev, N, na, embDim);
            // Single final download of the [N × embDim] routed result into the pinned staging
            // buffer, then one bulk copy into _bRoutedAll (the downstream combine's input).
            _gpu.Download(routedDev, (nint)_hGpuOffDownDl, N * embDim);
        }
        finally { _gpu.Free(routedDev); _gpu.Free(weightsDev); }
        Buffer.MemoryCopy(_hGpuOffDownDl, _bRoutedAll!,
                          (long)N * embDim * sizeof(float), (long)N * embDim * sizeof(float));
        if (_prefillProfile) _profGoDownScatter += System.Diagnostics.Stopwatch.GetTimestamp() - sr0;
    }

    /// <summary>
    /// Resolves the GPU weight matrix for expert <paramref name="e"/> on the op-offload path.
    /// When <paramref name="isRawLayer"/> (raw-quant dtype whose whole layer was uploaded once
    /// into <paramref name="layerBuf"/>), returns a non-owning <see cref="CudaBackend.ViewRawBytes"/>
    /// into that layer buffer at this expert's byte offset (<c>e·rawBytes</c>) — no per-expert
    /// transfer — and tags the view handle in <see cref="_gpuWeightDTypes"/> so
    /// <see cref="GpuMatMulBatched"/> dispatches on <paramref name="dt"/>. Otherwise (Float32)
    /// host-dequantizes this expert into the per-expert <paramref name="perExpertBuf"/> via
    /// <see cref="UploadOffloadWeight"/> and returns that buffer, tagged with the effective dtype.
    /// </summary>
    private Tensor ResolveExpertWeight(bool isRawLayer, Tensor layerBuf, Tensor perExpertBuf,
                                       byte* basePtr, int e, long rawBytes, DType dt,
                                       int rows, int cols, ref long dequantTimer)
    {
        if (isRawLayer)
        {
            // Non-owning byte view into the whole-layer buffer at this expert's slot.
            var view = _gpu.ViewRawBytes(layerBuf, (long)e * rawBytes, rawBytes,
                                         TensorShape.D2(rows, cols), dt);
            // ViewRawBytes tags the backend-side _tensorDTypes, but GpuMatMulBatched reads the
            // engine-side _gpuWeightDTypes — tag the view handle there explicitly.
            _gpuWeightDTypes[view.Handle] = dt;
            return view;
        }
        // Float32: host-dequant → F32 into the reused per-expert buffer (over-sized to F32 max).
        DType eff = UploadOffloadWeight(perExpertBuf, basePtr + (long)e * rawBytes,
                                        rawBytes, dt, rows, cols, ref dequantTimer);
        _gpuWeightDTypes[perExpertBuf.Handle] = eff;
        return perExpertBuf;
    }

    /// <summary>
    /// Releases the per-expert weight matrix returned by <see cref="ResolveExpertWeight"/>.
    /// For the raw-quant view path the tensor is a non-owning <see cref="CudaBackend.ViewRawBytes"/>:
    /// drop its engine-side dtype tag and its backend handle registration (the layer buffer keeps
    /// the device memory). The Float32 per-expert buffer is reused across experts, so nothing
    /// to free there (its dtype tag is overwritten on the next expert / cleared on scratch free).
    /// </summary>
    private void FreeExpertWeightView(bool isRawLayer, Tensor weight)
    {
        if (!isRawLayer) return;
        _gpuWeightDTypes.Remove(weight.Handle);
        _gpu.Free(weight);   // non-owning view: drops handle registration only
    }

    /// <summary>
    /// Uploads one expert weight matrix into a reused GPU buffer for the offload path and
    /// returns the dtype the GPU GEMM should dispatch on. Raw-quant dtypes (Q4_K/Q5_K/Q6_K/
    /// Q8_0) upload the raw quantized bytes unchanged and return the same dtype. Q3_K and
    /// Float32 are dequantized on the host into <see cref="_hGpuOffDeq"/> (Q3_K has no MMQ/
    /// GEMM-N kernel) and uploaded as F32, returning Float32. <paramref name="rows"/>/
    /// <paramref name="cols"/> are the matrix dimensions (gate/up: expertDim×embDim;
    /// down: embDim×expertDim) used to size the F32 dequant.
    /// </summary>
    private DType UploadOffloadWeight(Tensor dst, byte* src, long rawBytes, DType dt,
                                      int rows, int cols, ref long dequantTimer)
    {
        if (dt is DType.Q3_K or DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0)
        {
            // Q3_K now has a raw GPU GEMM-N kernel (#100) — upload the compact quantized
            // bytes and dequant in-kernel, eliminating the host F32 dequant (was ~79% of
            // the offload wall). The buffer is over-allocated to F32-max size; UploadRawInto
            // is byte-based and tolerates the over-sized dst.
            _gpu.UploadRawInto(dst, new ReadOnlySpan<byte>(src, (int)rawBytes));
            return dt;
        }
        // Float32: dequantize the whole matrix to F32 on the host, then upload F32.
        long count = (long)rows * cols;
        long dq0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        Dequantize.ToFloat32(new ReadOnlySpan<byte>(src, (int)rawBytes),
                             new Span<float>(_hGpuOffDeq, (int)count), dt, count);
        if (_prefillProfile)
        {
            long dq1 = System.Diagnostics.Stopwatch.GetTimestamp();
            _profGoDequant += dq1 - dq0;
            dequantTimer += dq1 - dq0;   // exclude dequant from the upload bucket
        }
        // The weight buffer is over-allocated to the F32-max size; view it to exactly
        // count elements for the F32 upload (UploadInto requires src.Length == dst.ElementCount).
        // The subsequent GpuMatMulBatched derives shape from the input/output element counts,
        // not the matrix buffer's, so the over-sized backing buffer is fine.
        var dstV = _gpu.View(dst, 0, count);
        try { _gpu.UploadInto(dstV, new ReadOnlySpan<float>(_hGpuOffDeq, (int)count)); }
        finally { _gpu.Free(dstV); }
        return DType.Float32;
    }

    /// <summary>
    /// Grow-only allocation of the GPU op-offload routed-prefill scratch (sized for
    /// <paramref name="N"/> tokens). The GEMM gather buffers are sized to N·na rows
    /// (the worst case when every selection routes to one expert); the transient weight
    /// buffers are sized to the max raw-quant bytes OR the F32 dequant size (whichever is
    /// larger — F32 is the worst case for every quant) over gate/up and down. Frees the
    /// prior allocation on growth. Only reached on the opt-in CPU-MoE GPU-offload path.
    /// </summary>
    private void EnsureGpuOffloadScratch(int N)
    {
        int embDim = _embDim;
        int expertDim = _expertDim;
        int na = _numActiveExperts;

        // ── N-independent static scratch: allocate ONCE and keep across regrows ──
        // The pinned expert-weight DMA source (copy-mode ~14 GB cudaMallocHost buffer, or the
        // in-place cudaHostRegister of the mmap pages, per SHARPI_MOE_PIN_MODE) + the whole-layer
        // GPU weight buffers + the F32 dequant staging don't depend on the batch size N, so a later
        // token-count growth must NOT free and re-build them — re-building the pinned source is
        // multi-second and fragmentation-prone (Gemini review). Only the per-N gather/scatter/GEMM
        // scratch below re-grows.
        if (!_goStaticAllocated)
        {
            // Transient weight buffers: an F32 dequant of a [expertDim×embDim] (or [embDim×
            // expertDim]) matrix is expertDim·embDim·4 bytes, which dominates any raw-quant
            // byte count for the same matrix — size all three to that and reuse for raw too.
            long wBytes = (long)expertDim * embDim * sizeof(float);
            _gpuOffWGate = _gpu.AllocateRawBytes(wBytes, DType.Float32);
            _gpuOffWUp   = _gpu.AllocateRawBytes(wBytes, DType.Float32);
            _gpuOffWDown = _gpu.AllocateRawBytes(wBytes, DType.Float32);
            _hGpuOffDeq  = (float*)NativeMemory.Alloc((nuint)((long)expertDim * embDim) * sizeof(float));

            // Whole-layer raw-quant weight buffers (one big UploadRawInto per layer, then
            // per-expert ViewRawBytes). Sized to the MAX raw layer bytes over every MoE layer so
            // a single allocation fits each layer's dtype (e.g. mixed Q4_K / Q5_K / Q6_K K_M).
            // gate/up rows = expertDim, cols = embDim; down rows = embDim, cols = expertDim. The
            // per-row byte count depends on the column count's block layout, so it differs per
            // dtype — take the max raw bytes-per-row over all layers for each of the three.
            long maxLayerGateBytes = 0, maxLayerUpBytes = 0, maxLayerDownBytes = 0;
            for (int l = 0; l < _hp.NumLayers; l++)
            {
                DType gDt = _cpuFfnGateExps![l].DType;
                DType uDt = _cpuFfnUpExps![l].DType;
                DType dDt = _cpuFfnDownExps![l].DType;
                long bg = (long)(embDim    / DTypeInfo.BlockSize(gDt)) * DTypeInfo.BytesPerBlock(gDt);
                long bu = (long)(embDim    / DTypeInfo.BlockSize(uDt)) * DTypeInfo.BytesPerBlock(uDt);
                long bd = (long)(expertDim / DTypeInfo.BlockSize(dDt)) * DTypeInfo.BytesPerBlock(dDt);
                maxLayerGateBytes = Math.Max(maxLayerGateBytes, (long)_numExperts * expertDim * bg);
                maxLayerUpBytes   = Math.Max(maxLayerUpBytes,   (long)_numExperts * expertDim * bu);
                maxLayerDownBytes = Math.Max(maxLayerDownBytes, (long)_numExperts * embDim    * bd);
            }
            // Two ping-pong slots so the next layer can DMA into the idle slot while this layer's
            // GEMMs read the live one (double-buffer; ~330→660 MB). Slot 0 doubles as the legacy
            // single buffer used by the synchronous path / first MoE layer of every chunk.
            for (int s = 0; s < 2; s++)
            {
                _gpuLayerGate[s] = _gpu.AllocateRawBytes(maxLayerGateBytes, DType.Float32);
                _gpuLayerUp[s]   = _gpu.AllocateRawBytes(maxLayerUpBytes,   DType.Float32);
                _gpuLayerDown[s] = _gpu.AllocateRawBytes(maxLayerDownBytes, DType.Float32);
            }

            // Pin the expert weights as the DMA source for UploadRawIntoAsyncDirect. SHARPI_MOE_PIN_MODE
            // selects how: "register" (default, #390) cudaHostRegisters the mmap pages in place — zero
            // RAM copy, decode-safe; "copy" (#387) makes the ~14 GB cudaMallocHost duplicate (faster DMA
            // but evicts the page cache CPU decode streams from). On any failure it self-falls-back to
            // the synchronous mmap upload (_goHostPinned=false), with zero regression.
            EnsureExpertWeightsPinned();
            _goStaticAllocated = true;
        }

        if (N <= _goCap) return;
        FreeGpuOffloadScratch(freeStatic: false);   // free only the per-N scratch; keep the static buffers above

        long maxRows = (long)N * na;   // worst-case gathered token rows for a single expert
        Tensor A(long elems) => _gpu.Allocate(TensorShape.D1(elems));
        // maxRows == N·na == the worst-case totalSel, so a single CSR-ordered gather/down
        // buffer of maxRows rows holds every (token,slot) selection.
        _gpuOffNorm    = A((long)N * embDim);
        _gpuOffGather  = A(maxRows * embDim);     // [totalSel × embDim] CSR-ordered gathered norms
        _gpuOffGate    = A(maxRows * expertDim);  // [totalSel × expertDim]
        _gpuOffUp      = A(maxRows * expertDim);  // [totalSel × expertDim]
        _gpuOffDownCsr = A(maxRows * embDim);     // [totalSel × embDim] CSR-ordered down output

        // GPU-side scatter target + weighted-reduce output + device top-k weights, mirroring
        // BatchedGpuMoeFfn's _gpuBfMoeDownPartial / hiddenAll / _gpuBfMoeWeightsDev. The
        // down partials are per-(token,slot): N·na rows of embDim (= maxRows·embDim).
        _gpuOffDownPartial = A(maxRows * embDim);   // [N × na × embDim]
        _gpuOffRouted      = A((long)N * embDim);    // [N × embDim] reduce output (downloaded once)
        _gpuOffWeightsDev  = A(maxRows);             // [N × na] top-k weights (device)

        // Int32 CSR gather/scatter index buffers (one launch each fills the whole gathered /
        // scattered buffer). Sized to N·na (== maxRows) selections; uploaded raw per layer call.
        _gpuOffGatherIdx  = _gpu.AllocateRawBytes(maxRows * sizeof(int), DType.Int32, exact: true);
        _gpuOffScatterIdx = _gpu.AllocateRawBytes(maxRows * sizeof(int), DType.Int32, exact: true);
        _hGpuOffScatterIdx = (int*)NativeMemory.Alloc((nuint)maxRows * sizeof(int));

        nint dl = CudaBackend.AllocatePinnedHost((nuint)(maxRows * embDim) * sizeof(float));
        if (dl == nint.Zero)
            throw new InvalidOperationException($"AllocatePinnedHost({maxRows * embDim} floats) failed for GPU-offload download scratch.");
        _hGpuOffDownDl = (float*)dl;

        _goCap = N;
    }

    /// <summary>
    /// One-time setup of the pinned host DMA source for every MoE layer's gate/up/down expert
    /// weights, in one of two modes selected by <c>SHARPI_MOE_PIN_MODE</c>: <c>register</c>
    /// (default, #390) <c>cudaHostRegister</c>s the expert mmap pages in place — no RAM copy, so
    /// the CPU page cache the decode path streams experts from is never evicted; <c>copy</c>
    /// (#387) allocates a single big <c>cudaMallocHost</c> buffer (~14 GB) and copies every
    /// tensor in contiguously (higher DMA bandwidth but evicts that page cache). Either way the
    /// result is the DMA source for <see cref="CudaBackend.UploadRawIntoAsyncDirect"/>, so the
    /// H2D copy runs without per-call staging and overlaps compute. Sets <see cref="_goHostPinned"/>
    /// on success; on ANY failure (e.g. the copy-mode alloc returns Zero — typically ENOMEM for the
    /// ~14 GB) leaves it false so the path stays on the synchronous mmap upload (no regression).
    /// Only raw-quant layers are registered/copied/DMA'd — Float32 layers host-dequant from the
    /// mmap as before, so a Float32 layer leaves its per-layer pinned base null and falls back per
    /// call. Guarded by <see cref="_goPinAttempted"/> so it runs at most once per scratch sizing.
    /// The mmap (<see cref="_cpuFfnGateExps"/> etc.) is never touched — the CPU-MoE path keeps using it.
    /// </summary>
    private void EnsureExpertWeightsPinned()
    {
        if (_goPinAttempted) return;
        _goPinAttempted = true;

        int embDim = _embDim;
        int expertDim = _expertDim;
        int numExperts = _numExperts;
        int L = _hp.NumLayers;

        // Whole-layer raw byte length for one ffn_*_exps tensor (all experts back to back) — the
        // count we both allocate and copy. Zero (skip, mmap fallback) for a Float32 weight or a
        // missing tensor. "cols" is the per-row dim (embDim for gate/up, expertDim for down);
        // "rows" the outer count (expertDim for gate/up, embDim for down).
        long PinnedBytes(byte* src, DType dt, int rows, int cols)
        {
            if (src == null || !IsRawOffloadQuant(dt)) return 0;
            long bpr = (long)(cols / DTypeInfo.BlockSize(dt)) * DTypeInfo.BytesPerBlock(dt);
            return (long)numExperts * rows * bpr;
        }

        // 1. Sum the total bytes to allocate (raw-quant, present layers only). Uses the SAME
        //    predicate as the copy below so the buffer size and the copied bytes stay in lockstep.
        long total = 0;
        for (int l = 0; l < L; l++)
        {
            total += PinnedBytes(_cpuFfnGateExps![l].DataPtr, _cpuFfnGateExps![l].DType, expertDim, embDim);
            total += PinnedBytes(_cpuFfnUpExps![l].DataPtr,   _cpuFfnUpExps![l].DType,   expertDim, embDim);
            total += PinnedBytes(_cpuFfnDownExps![l].DataPtr, _cpuFfnDownExps![l].DType, embDim,    expertDim);
        }
        if (total <= 0) { _goHostPinned = false; return; }   // nothing raw-quant → mmap path

        // #390: pin mode. "register" (default) = cudaHostRegister the expert mmap pages in place
        // (zero RAM copy → no page-cache eviction → no decode regression; DMA ~13 GB/s). "copy" =
        // the original ~14 GB cudaMallocHost duplicate (DMA ~26 GB/s but evicts the page cache CPU
        // decode streams experts from → ~-25% decode). Default register so op-offload is decode-safe.
        string pinMode = (Environment.GetEnvironmentVariable("SHARPI_MOE_PIN_MODE") ?? "register")
            .Trim().ToLowerInvariant();

        _goPinnedGate = new byte*[L];
        _goPinnedUp   = new byte*[L];
        _goPinnedDown = new byte*[L];

        if (pinMode == "register")
        {
            // Set each raw-quant layer's DMA base straight to the mmap pointer (Float32 / missing →
            // null, per-layer host-dequant fallback). Collect each tensor's raw byte range, then
            // page-align + merge them via MergePageAlignedRanges for cudaHostRegister. GGUF tensors
            // are 32-byte aligned (not page-aligned), so adjacent tensors can share a page — merging
            // ensures no page is registered twice (a double register returns false → that range
            // silently DMAs as pageable; merging avoids it).
            var rawRanges = new List<(long ptr, long bytes)>(3 * L);
            void Reg(byte* src, DType dt, int rows, int cols, ref byte* slot)
            {
                long bytes = PinnedBytes(src, dt, rows, cols);
                if (bytes == 0) { slot = null; return; }
                slot = src;
                rawRanges.Add(((long)src, bytes));
            }
            for (int l = 0; l < L; l++)
            {
                Reg(_cpuFfnGateExps![l].DataPtr, _cpuFfnGateExps![l].DType, expertDim, embDim, ref _goPinnedGate[l]);
                Reg(_cpuFfnUpExps![l].DataPtr,   _cpuFfnUpExps![l].DType,   expertDim, embDim, ref _goPinnedUp[l]);
                Reg(_cpuFfnDownExps![l].DataPtr, _cpuFfnDownExps![l].DType, embDim,    expertDim, ref _goPinnedDown[l]);
            }
            var merged = MergePageAlignedRanges(rawRanges, 4096);
            var reg = new List<nint>(merged.Count);
            int ok = 0;
            foreach (var (s, e) in merged)
                if (_gpu.TryRegisterHostPinned((nint)s, e - s)) { reg.Add((nint)s); ok++; }
            if (ok < merged.Count)
            {
                // Partial registration (e.g. Linux RLIMIT_MEMLOCK / `ulimit -l`) would leave the
                // un-registered ranges pageable, silently degrading UploadRawIntoAsyncDirect to a
                // slow synchronous staged copy while _goHostPinned claims otherwise. Don't keep a
                // half-pinned state: unregister what succeeded and fall back wholesale to the
                // synchronous mmap upload (_goHostPinned=false) — correct, no perf cliff (Gemini).
                Console.Error.WriteLine($"[moe-offload] registered only {ok}/{merged.Count} mmap expert-weight ranges (locked-memory limit?) — falling back to synchronous mmap upload.");
                foreach (var p in reg) _gpu.UnregisterHostPinned(p);
                _goRegisteredRanges = null;
                _goPinnedBuf = nint.Zero;
                _goHostPinned = false;
                return;
            }
            _goRegisteredRanges = reg.ToArray();
            _goPinnedBuf = nint.Zero;
            _goHostPinned = true;
            Console.Error.WriteLine($"[moe-offload] registered {ok}/{merged.Count} mmap expert-weight ranges in place ({total / (1024.0 * 1024.0 * 1024.0):F2} GiB, cudaHostRegister) — no RAM copy, DMA source for op-offload prefill.");
            return;
        }

        // ── "copy" mode (#387): one big pinned cudaMallocHost buffer + copy (max DMA bandwidth) ──
        // 2. Allocate the big pinned buffer. On failure (ENOMEM) fall back to the mmap path.
        nint buf = CudaBackend.AllocatePinnedHost((nuint)total);
        if (buf == nint.Zero)
        {
            Console.Error.WriteLine($"[moe-offload] cudaMallocHost({total / (1024.0 * 1024.0):F0} MiB) for pinned expert weights FAILED — falling back to synchronous mmap upload.");
            _goHostPinned = false;
            return;
        }

        // 3. Copy each raw-quant layer's gate/up/down tensor into the buffer, contiguously, and
        //    record the per-layer base pointers the DMA source will use. Float32 layers leave a
        //    null base (per-layer sync host-dequant fallback in BatchedRoutedExpertsGpuOffload).
        byte* dst = (byte*)buf;
        long off = 0;
        void CopyOne(byte* src, DType dt, int rows, int cols, ref byte* slot)
        {
            long bytes = PinnedBytes(src, dt, rows, cols);
            if (bytes == 0) { slot = null; return; }   // Float32 / missing → mmap fallback
            byte* d = dst + off;
            Buffer.MemoryCopy(src, d, bytes, bytes);
            slot = d;
            off += bytes;
        }
        for (int l = 0; l < L; l++)
        {
            CopyOne(_cpuFfnGateExps![l].DataPtr, _cpuFfnGateExps![l].DType, expertDim, embDim, ref _goPinnedGate[l]);
            CopyOne(_cpuFfnUpExps![l].DataPtr,   _cpuFfnUpExps![l].DType,   expertDim, embDim, ref _goPinnedUp[l]);
            CopyOne(_cpuFfnDownExps![l].DataPtr, _cpuFfnDownExps![l].DType, embDim,    expertDim, ref _goPinnedDown[l]);
        }

        _goPinnedBuf = buf;
        _goHostPinned = true;
        Console.Error.WriteLine($"[moe-offload] pinned expert-weight buffer ready: {total / (1024.0 * 1024.0 * 1024.0):F2} GiB (cudaMallocHost) — DMA source for op-offload prefill.");
    }

    /// <summary>True when the offload weight dtype uploads raw quantized bytes unchanged
    /// (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0) — eligible for the whole-layer batched upload + per-expert
    /// view. Q3_K has an in-kernel-dequant GEMM-N (#100) so it too uploads compact raw bytes
    /// and dispatches a raw kernel. Only Float32 host-dequantizes and takes the per-expert
    /// fallback (mirrors the raw-vs-F32 split inside <see cref="UploadOffloadWeight"/>).</summary>
    private static bool IsRawOffloadQuant(DType dt) =>
        dt is DType.Q3_K or DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0;

    /// <summary>
    /// Uploads <paramref name="byteLength"/> raw bytes from host <paramref name="src"/> into the
    /// front of <paramref name="dst"/>. <see cref="CudaBackend.UploadRawInto"/> takes a
    /// <c>ReadOnlySpan&lt;byte&gt;</c> (int length), so a layer-weight tensor larger than 2 GiB is
    /// uploaded in ≤int.MaxValue chunks via byte-addressed destination views (none of the current
    /// MoE models reach this, but the chunk loop keeps the span length from silently truncating).
    /// </summary>
    private void UploadLayerRaw(Tensor dst, byte* src, long byteLength)
    {
        long off = 0;
        while (off < byteLength)
        {
            int chunk = (int)Math.Min(byteLength - off, int.MaxValue);
            if (off == 0 && byteLength <= int.MaxValue)
            {
                _gpu.UploadRawInto(dst, new ReadOnlySpan<byte>(src, chunk));
                return;
            }
            var dstV = _gpu.ViewRawBytes(dst, off, chunk, TensorShape.D1(chunk), DType.Int8);
            try { _gpu.UploadRawInto(dstV, new ReadOnlySpan<byte>(src + off, chunk)); }
            finally { _gpu.Free(dstV); }
            off += chunk;
        }
    }

    /// <param name="freeStatic">
    /// <c>false</c> (the per-N regrow path) frees ONLY the batch-size-dependent gather/scatter/GEMM
    /// scratch, keeping the N-independent static buffers — the pinned expert-weight DMA source
    /// (copy-mode ~14 GB buffer or the in-place mmap registration, per <c>SHARPI_MOE_PIN_MODE</c>),
    /// the whole-layer GPU weight buffers, the F32 dequant staging — so a token-count growth doesn't
    /// re-build the multi-second pinned source (Gemini review). <c>true</c> (full teardown / Dispose)
    /// additionally frees those static buffers and resets the pin + static-alloc state.
    /// </param>
    private void FreeGpuOffloadScratch(bool freeStatic = true)
    {
        // Drain any in-flight prefetch DMA before freeing the buffers it targets (else the
        // backend would free a device buffer with a live H2D copy still draining into it).
        DrainPrefetch();

        void F(ref Tensor? t) { if (t is { } v) { _gpu.Free(v); t = null; } }
        // Drop the engine-side dtype tag before freeing so the handle dict doesn't leak the
        // layer-buffer entry across a scratch regrow (the handle is recycled by the backend).
        void FW(ref Tensor? t) { if (t is { } v) { _gpuWeightDTypes.Remove(v.Handle); _gpu.Free(v); t = null; } }

        // ── Per-N dynamic scratch: always freed (re-grown by EnsureGpuOffloadScratch) ──
        F(ref _gpuOffNorm); F(ref _gpuOffGather); F(ref _gpuOffGate); F(ref _gpuOffUp); F(ref _gpuOffDownCsr);
        F(ref _gpuOffDownPartial); F(ref _gpuOffRouted); F(ref _gpuOffWeightsDev);
        F(ref _gpuOffGatherIdx); F(ref _gpuOffScatterIdx);
        if (_hGpuOffScatterIdx != null) { NativeMemory.Free(_hGpuOffScatterIdx); _hGpuOffScatterIdx = null; }
        if (_hGpuOffDownDl != null) { CudaBackend.FreePinnedHost((nint)_hGpuOffDownDl); _hGpuOffDownDl = null; }
        _goCurSlot = 0;
        _goPrefetchedLayer = -1;
        _goCap = 0;

        if (!freeStatic) return;

        // ── N-independent static buffers + the pinned expert-weight DMA source (copy-mode ~14 GB
        //    buffer or the in-place mmap registration, per SHARPI_MOE_PIN_MODE): only on full
        //    teardown ── (DrainPrefetch above ensured no DMA is still reading the slot buffers /
        //    pinned source.)
        F(ref _gpuOffWGate); F(ref _gpuOffWUp); F(ref _gpuOffWDown);
        if (_hGpuOffDeq != null) { NativeMemory.Free(_hGpuOffDeq); _hGpuOffDeq = null; }
        for (int s = 0; s < 2; s++) { FW(ref _gpuLayerGate[s]); FW(ref _gpuLayerUp[s]); FW(ref _gpuLayerDown[s]); }
        if (_goPinnedBuf != nint.Zero) { CudaBackend.FreePinnedHost(_goPinnedBuf); _goPinnedBuf = nint.Zero; }
        if (_goRegisteredRanges != null)
        {
            foreach (var p in _goRegisteredRanges) _gpu.UnregisterHostPinned(p);
            _goRegisteredRanges = null;
        }
        _goPinnedGate = null; _goPinnedUp = null; _goPinnedDown = null;
        _goHostPinned = false;
        _goPinAttempted = false;
        _goStaticAllocated = false;
    }

    /// <summary>
    /// Wait for any in-flight prefetch DMA to complete and release its handles. Idempotent —
    /// a no-op when no prefetch is outstanding. Called at chunk start (so a stale prefetch
    /// never bleeds across chunks) and before freeing the scratch buffers it targets.
    /// </summary>
    private void DrainPrefetch()
    {
        if (!_goPrefetchValid) return;
        // HOST-block until the DMAs have actually drained — DrainPrefetch precedes freeing /
        // regrowing the target buffers, so an async cross-stream fence (WaitForUpload) is not
        // enough; the copy must be complete from the host's perspective before we release them.
        _gpu.WaitForUploadHost(_goPrefetchGateH);
        _gpu.WaitForUploadHost(_goPrefetchUpH);
        _gpu.WaitForUploadHost(_goPrefetchDownH);
        _gpu.ReleaseUploadHandle(_goPrefetchGateH);
        _gpu.ReleaseUploadHandle(_goPrefetchUpH);
        _gpu.ReleaseUploadHandle(_goPrefetchDownH);
        _goPrefetchValid = false;
        _goPrefetchedLayer = -1;
    }

    /// <summary>
    /// SnapKV (issue #58): score the captured trailing-W queries against the
    /// VRAM K cache for every attention layer (atomicAdd-pooled into a single
    /// per-position accumulator), download the accumulator, pick a keep set,
    /// then compact the GPU K/V rings + the host-side <see cref="_kvCache"/>
    /// length bookkeeping. Called once at the end of a SnapKV-active prefill.
    /// </summary>
    private void ApplySnapKvEviction(int N, int W, int wStart)
    {
        EnsureSnapKvScoreBuffer();
        // Zero only the prompt-prefix slice; the rest of the [maxSeqLen] buffer
        // doesn't participate in scoring and will not be downloaded.
        _gpu.ClearRegion(_snapKvScoreAccum!, 0, N);

        int qDim = _numHeads * _headDim;
        bool bf16Kv = _kvDType == DType.BFloat16;
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            int attnIdx = _attnLayerIndexOf[layer];
            if (attnIdx < 0) continue;
            for (int w = 0; w < W; w++)
            {
                // Stage the captured Q into _gpuQ so the scoring kernel can
                // read a contiguous [numHeads × headDim] vector at the same
                // device pointer it does during Forward.
                long srcOffsetElems = ((long)attnIdx * _snapKvQCaptureW + w) * qDim;
                _gpu.CopyDeviceRegion(_gpuQ, 0,
                    _snapKvQCapture!, srcOffsetElems * sizeof(float),
                    (long)qDim * sizeof(float));

                int qAbsPos = wStart + w;
                if (bf16Kv)
                {
                    _gpu.SnapKvScoreBf16(_gpuQ, _gpuKCache[layer]!,
                        _snapKvScoreAccum!, _gpuAttnScratch,
                        _numHeads, _numKvHeads, _headDim,
                        N, qAbsPos, _maxSeqLen);
                }
                else
                {
                    _gpu.SnapKvScore(_gpuQ, _gpuKCache[layer]!,
                        _snapKvScoreAccum!, _gpuAttnScratch,
                        _numHeads, _numKvHeads, _headDim,
                        N, qAbsPos, _maxSeqLen);
                }
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
        // Stage buffer for one layer's gather output: [K × kvDim] of _kvDType.
        int kvDim = _numKvHeads * _headDim;
        var stage = _gpu.Allocate(TensorShape.D1((long)K * kvDim), _kvDType);
        try
        {
            long sliceBytes = (long)K * kvDim * DTypeInfo.BytesPerElement(_kvDType);
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                if (_attnLayerIndexOf[layer] < 0) continue;

                // K: gather kept positions into stage, then copy stage back over
                // the ring's [0, K * kvDim) prefix. Same for V. Two-phase to
                // avoid the src==dst race (kernel block ordering is undefined).
                if (bf16Kv)
                {
                    _gpu.KvCompactBf16(_gpuKCache[layer]!, stage, keepDev, K, kvDim);
                    _gpu.CopyDeviceRegion(_gpuKCache[layer]!, 0, stage, 0, sliceBytes);
                    _gpu.KvCompactBf16(_gpuVCache[layer]!, stage, keepDev, K, kvDim);
                    _gpu.CopyDeviceRegion(_gpuVCache[layer]!, 0, stage, 0, sliceBytes);
                }
                else
                {
                    _gpu.KvCompact(_gpuKCache[layer]!, stage, keepDev, K, kvDim);
                    _gpu.CopyDeviceRegion(_gpuKCache[layer]!, 0, stage, 0, sliceBytes);
                    _gpu.KvCompact(_gpuVCache[layer]!, stage, keepDev, K, kvDim);
                    _gpu.CopyDeviceRegion(_gpuVCache[layer]!, 0, stage, 0, sliceBytes);
                }
            }
        }
        finally
        {
            _gpu.Free(stage);
            _gpu.Free(keepDev);
        }

        // Compact the host-side length bookkeeping. _kvCache stores no payload
        // for this forward pass (the data lives in _gpuKCache/_gpuVCache), so
        // we only need the slot-count drop: Length → K, LogicalLength stays at
        // N, and trailing block slots return to the warm pool.
        _kvCache.CompactLengthOnly(K);
    }

    private void EnsureSnapKvCaptureBuffer(int W)
    {
        if (_snapKvQCapture is not null && _snapKvQCaptureW >= W)
            return;
        if (_snapKvQCapture is { } old)
            _gpu.Free(old);
        int qDim = _numHeads * _headDim;
        long elems = (long)_numAttnLayers * W * qDim;
        _snapKvQCapture = _gpu.Allocate(TensorShape.D1(elems));
        _snapKvQCaptureW = W;
    }

    private void EnsureSnapKvScoreBuffer()
    {
        if (_snapKvScoreAccum is not null) return;
        _snapKvScoreAccum = _gpu.Allocate(TensorShape.D1(_maxSeqLen));
    }

    private void EnsureMtpHiddenHistoryCap(int requiredTokens)
    {
        if (_mtpPrefillHiddensCap >= requiredTokens) return;
        // Grow by doubling — see HybridGdnForwardPass.EnsureMtpHiddenHistoryCap.
        int newCap = Math.Max(requiredTokens, _mtpPrefillHiddensCap * 2);
        long oldBytes = (long)_mtpHiddenHistoryLength * _embDim * sizeof(float);
        float* fresh = (float*)NativeMemory.Alloc(
            (nuint)((long)newCap * _embDim * sizeof(float)));
        if (_mtpPrefillHiddens != null)
        {
            if (oldBytes > 0)
                NativeMemory.Copy(_mtpPrefillHiddens, fresh, (nuint)oldBytes);
            NativeMemory.Free(_mtpPrefillHiddens);
        }
        _mtpPrefillHiddens = fresh;
        _mtpPrefillHiddensCap = newCap;
    }

    public void TruncateTo(int length)
    {
        if (length == _gdnStateCache.Length)
        {
            _kvCache.TruncateTo(length);
            // Mirror the CPU path: keep MTP attention KV in lockstep with the
            // trunk so a future RestoreBatchSnapshot-without-MtpTruncateTo
            // caller can't silently leave stale entries past `length`.
            _mtpKvCache?.TruncateTo(length);
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
            // Issue #106: rewind MTP KV (the device-side _gpuMtpKCache is a flat
            // ring, so future KvAppends just overwrite stale slots) and the
            // hidden-history length so PrefillMtp(suffix, startPos=length) sees
            // a consistent view.
            if (_hasMtp)
            {
                _mtpKvCache?.TruncateTo(length);
                if (_mtpHiddenHistoryLength > length)
                    _mtpHiddenHistoryLength = length;
            }
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
        // Always reset the hidden-history length regardless of _hasMtp, mirroring
        // the CPU pass — the field defaults to 0 on non-MTP passes so this is a
        // no-op there, but the unconditional form prevents a future MTP-late-bind
        // refactor from leaving stale state across resets.
        _mtpHiddenHistoryLength = 0;
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
    public bool SupportsSnapshot => true;

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

    /// <summary>
    /// Dispatch the per-token embedding lookup into <paramref name="dst"/> based on
    /// the on-GPU embedding table's dtype. Q4_K + Q5_K both have direct-read NVRTC
    /// kernels (issue #39); other dtypes are F32-expanded at load time and read via
    /// the F32 path.
    /// </summary>
    private void EmbedToken(Tensor dst, int token)
    {
        switch (_embDType)
        {
            case DType.Q4_K:
                _gpu.EmbedLookupQ4K(_gpuEmbedding!, dst, token, _embDim);
                break;
            case DType.Q5_K:
                _gpu.EmbedLookupQ5K(_gpuEmbedding!, dst, token, _embDim);
                break;
            default:
                _gpu.EmbedLookup(_gpuEmbedding!, dst, token, _embDim);
                break;
        }
    }

    /// <summary>
    /// #388: the pure-GPU decode "trunk block" for one layer — the unit that runs directly OR
    /// gets captured into a per-layer CUDA graph. Pre-block residual+norm, the attn/GDN block,
    /// the residual add, then the pre-MoE residual+norm; leaves <c>_gpuNormBuf</c> (the MoE input)
    /// and <c>_gpuResidual</c> ready for the CPU-MoE step that runs OUTSIDE the graph. Position
    /// dependence lives only in the attn block's RoPE/KvAppend/Attention kernels, which self-register
    /// position nodes via TrackPositionNode so graph replay rewrites the position; GDN decode kernels
    /// mutate state in place (position-independent → capture-once).
    /// NOTE: the embedded <c>_traceLayers</c> trace does a Synchronize+Download, which is illegal
    /// during stream capture — so SHARPI_TRACE_LAYERS + SHARPI_DECODE_CUDA_GRAPH together make the
    /// first capture fail and latch graphs off (graceful fallback to direct launches). Both are
    /// developer diagnostics; don't combine them when measuring the graph path.
    /// </summary>
    private void RunDecodeTrunkBlock(int layer, int position, bool isAttn)
    {
        // ── Pre-block residual + norm on GPU ────────────────────
        _gpu.CopyDevice(_gpuResidual, _gpuHidden);
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[layer], _hp.RmsNormEps);

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
    }

    /// <summary>Latch the decode CUDA-graph path off after a capture/replay failure, logging the
    /// cause once. The fallback (direct RunDecodeTrunkBlock) is always correct; this only surfaces
    /// WHY the opt-in graph disabled (a persistent root cause still re-throws via the direct re-run).</summary>
    private void DisableDecodeGraph(int layer, string reason)
    {
        if (!_decodeGraphDisabled)
            Console.Error.WriteLine($"[graph-diag] decode CUDA-graph disabled at layer {layer}: {reason}");
        _decodeGraphDisabled = true;
    }

    /// <summary>Forward one token through the hybrid CUDA + CPU stack.</summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        ThrowIfFaulted();
        if (_decodeCudaGraph) _decodeTokensSeen++;   // warmup counter for the per-layer trunk graph
        long fwdT0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        // 1. Embedding → _gpuHidden
        EmbedToken(_gpuHidden, token);

        if (_traceLayers) TraceGpuTensor(position, -1, "emb", _gpuHidden, _embDim);

        // 2. Reserve KV cache page (layer-0 invariant; even if layer 0 is GDN).
        _kvCache.ReserveBlock();

        // 3. Trunk layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            long ltp0 = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            // #388: run the pure-GPU trunk block directly, or via a per-layer CUDA graph
            // (capture once on the first eligible token, replay at the new position after).
            // The CPU-MoE stays OUTSIDE the graph (it does an illegal-in-capture Download).
            // TRADE-OFF (correctness-safe, perf-only; see #401): the attention kernel choice
            // (single-block vs split-KV) is frozen at capture-time context (short, just after
            // warmup), so a graphed run keeps the single-block kernel even past the split-KV
            // threshold. Single-block is the bit-exact reference for any seqLen ≤ maxSeqLen, so
            // output stays correct — but long-context decode silently forgoes the split-KV speedup.
            // Acceptable while opt-in/short-ctx; revisit before any default-on (#401).
            bool useGraph = _decodeCudaGraph && !_decodeGraphDisabled && !_cpuGdn
                && _hp.IsMoE && _layerGraphCaptured is not null
                && _decodeTokensSeen >= GraphWarmupTokens;   // warmup: let on-demand scratch settle before capturing a stable graph
            if (useGraph && _layerGraphCaptured![layer] && _gpu.GraphReadyFor(layer))
            {
                // Steady state: replay this layer's captured trunk at the new position.
                try { _gpu.LaunchGraphForPosition(layer, position); }
                catch (Exception ex) { DisableDecodeGraph(layer, $"replay: {ex.Message}"); RunDecodeTrunkBlock(layer, position, isAttn); }
            }
            else if (useGraph)
            {
                // First eligible token for this layer: pre-grow on-demand scratch (capture forbids
                // cudaMalloc), capture the trunk block (records, no execute), instantiate, then launch.
                // Size to the WIDEST vector the trunk quantizes for a dp4a/Q8_1 matvec — not just
                // _embDim: the attn o-proj input is qDim (_numHeads*_headDim) and the GDN out-proj
                // input is _gdnValueDim, both > _embDim. Undersizing here would let the first wide
                // matvec cudaFree+cudaMalloc the grow-only _q81Buf DURING capture; under RELAXED that
                // alloc passes through and the captured node binds a since-freed pointer → silent
                // garbage on replay. Sizing it correctly makes correctness independent of the warmup.
                _gpu.EnsureQ81Scratch(Math.Max(_embDim, Math.Max(_numHeads * _headDim, _gdnValueDim)));
                try
                {
                    if (_gpu.TryBeginGraphCapture(layer))
                    {
                        RunDecodeTrunkBlock(layer, position, isAttn);
                        if (_gpu.TryEndGraphCaptureAndInstantiate(layer))
                        {
                            _gpu.LaunchGraphForPosition(layer, position);
                            _layerGraphCaptured![layer] = true;
                        }
                        else { DisableDecodeGraph(layer, "TryEndGraphCaptureAndInstantiate returned false"); RunDecodeTrunkBlock(layer, position, isAttn); }
                    }
                    else { DisableDecodeGraph(layer, "TryBeginGraphCapture returned false"); RunDecodeTrunkBlock(layer, position, isAttn); }
                }
                catch (Exception ex) { _gpu.AbortGraphCapture(); DisableDecodeGraph(layer, $"capture/launch: {ex.Message}"); RunDecodeTrunkBlock(layer, position, isAttn); }
            }
            else
            {
                RunDecodeTrunkBlock(layer, position, isAttn);
            }

            if (_decodeProfile)
            {
                _gpu.Synchronize();   // drain the trunk to attribute its GPU time (the MoE Download would drain it anyway)
                _pdTrunkTicks += System.Diagnostics.Stopwatch.GetTimestamp() - ltp0;
            }
            long mtp0 = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

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
                    // Direct-pinned Download/UploadInto (issue #48): _cpuNormBuf
                    // and _cpuMoeHidden are cudaMallocHost'd, so cudaMemcpyAsync
                    // can DMA straight in/out without bouncing through _pinnedBuf.
                    _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, _embDim);
                    CpuDenseFfn(layer);
                    _gpu.UploadInto(_gpuHidden, (nint)_cpuMoeHidden, _embDim);
                }
            }
            else if (_cpuMoe)
            {
                // Download _gpuNormBuf → _cpuNormBuf, run MoE on CPU, upload result.
                // Download already syncs the stream (CudaMemcpyAsync + StreamSynchronize),
                // so an explicit Synchronize before it would just stall the host twice.
                // Pinned overloads (issue #48): skip the _pinnedBuf staging hop.
                long moeT0 = _prefillProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, _embDim);
                CpuMoeFfn(layer);
                _gpu.UploadInto(_gpuHidden, (nint)_cpuMoeHidden, _embDim);
                if (_prefillProfile)
                    _profMoeTicks += System.Diagnostics.Stopwatch.GetTimestamp() - moeT0;
            }
            else
            {
                GpuMoeFfn(layer);
            }

            // Residual add
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);

            if (_decodeProfile) _pdMoeTicks += System.Diagnostics.Stopwatch.GetTimestamp() - mtp0;

            if (_traceLayers) TraceGpuTensor(position, layer, "moe-resid", _gpuHidden, _embDim);
        }

        if (_decodeCudaGraph && !_graphDiagLogged && _decodeTokensSeen > GraphWarmupTokens && _layerGraphCaptured is not null)
        {
            int cap = 0; foreach (bool b in _layerGraphCaptured) if (b) cap++;
            Console.Error.WriteLine($"[graph-diag] captured {cap}/{_hp.NumLayers} layers, disabled={_decodeGraphDisabled}");
            _graphDiagLogged = true;
        }

        // 4. Advance position counters
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();

        // 5. Capture pre-output-norm hidden for MTP (issue #29). _gpu.RmsNorm
        //    below overwrites _gpuHidden in place; snapshot to _gpuLastHidden
        //    on the same stream so the snapshot is consistent. Issue #49: the
        //    _lastHidden D2H is queued via DownloadAsync (no sync) AFTER the
        //    lm_head MatMul has been enqueued. The MatMul kernel launch is no
        //    longer gated on the snapshot's PCIe transfer, and the logits
        //    Download below is the single sync that drains both. Pinning
        //    _lastHidden via cudaMallocHost (constructor) lets cudaMemcpyAsync
        //    run truly concurrently with the lm_head kernel — without pinning,
        //    the driver would silently route through the shared _pinnedBuf
        //    and serialise behind the next Download's sync anyway.
        if (_hasMtp)
            _gpu.CopyDevice(_gpuLastHidden, _gpuHidden);

        // 6. Final norm + output projection on GPU
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden,
            _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var outDt) ? outDt : DType.Float32);

        if (_hasMtp)
            _gpu.DownloadAsync(_gpuLastHidden, (nint)_lastHidden, _embDim);

        // 6. Download logits to host (Download self-syncs the stream — also
        // drains the _lastHidden async D2H above).
        _gpu.Download(_gpuLogits, _logitsBuf);

        // Issue #106: mirror the host _lastHidden into the absolute-position
        // hidden history buffer so future snapshot-restore + PrefillMtp(startPos =
        // past decode position) calls can read the right h_{p-1}. Must come AFTER
        // the synchronizing Download above so _lastHidden's async D2H has landed.
        if (_hasMtp)
        {
            EnsureMtpHiddenHistoryCap(position + 1);
            new ReadOnlySpan<float>(_lastHidden, _embDim).CopyTo(
                new Span<float>(_mtpPrefillHiddens + (long)position * _embDim, _embDim));
            if (_mtpHiddenHistoryLength < position + 1)
                _mtpHiddenHistoryLength = position + 1;
        }

        if (_traceLayers) TraceLogits(position, _logitsBuf);

        if (_prefillProfile)
        {
            _profTotalTicks += System.Diagnostics.Stopwatch.GetTimestamp() - fwdT0;
            _profTokens++;
        }

        if (_decodeProfile && ++_pdTokens % 50 == 0)
        {
            double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            Console.Error.WriteLine(
                $"[decode-profile] {_pdTokens} tok: trunk={_pdTrunkTicks * f / _pdTokens:F2} ms/tok " +
                $"moe(dl+cpu+ul)={_pdMoeTicks * f / _pdTokens:F2} ms/tok " +
                $"(trunk {100.0 * _pdTrunkTicks / (_pdTrunkTicks + _pdMoeTicks):F0}% / moe {100.0 * _pdMoeTicks / (_pdTrunkTicks + _pdMoeTicks):F0}%)");
            Console.Error.WriteLine(
                $"[decode-moe] router={_pdRouterTicks * f / _pdTokens:F2} phaseA(gate+up)={_pdPhaseATicks * f / _pdTokens:F2} " +
                $"phaseC(down)={_pdPhaseCTicks * f / _pdTokens:F2} shared(dl+comb)={_pdSharedTicks * f / _pdTokens:F2} ms/tok " +
                $"(sum of CPU-MoE per-layer phases; remainder of moe = norm download + result upload)");
        }

        return _logitsBuf;
    }

    // =================================================================
    //  BatchForward2 — MTP batched verify (issue #30, CUDA mirror of
    //  HybridGdnForwardPass.BatchForward2). Bandwidth win lives in
    //  CpuDenseFfn2 (MatVec2In on the CPU mmap FFN); GPU attn/GDN run
    //  sequentially per token — no cuBLAS batched GEMM here since most
    //  27B-MTP layers are CPU FFN under realistic 12 GB VRAM budgets.
    // =================================================================

    /// <summary>True when the attention KV cache has been SnapKV-compacted, i.e.
    /// the physical slot count (<see cref="PagedKvCache.Length"/>) has dropped
    /// below the logical RoPE position (<see cref="PagedKvCache.LogicalLength"/>).
    /// <c>IncrementPosition</c> advances both together and <c>TruncateTo</c>/<c>Reset</c>
    /// keep them equal, so this is an exact, stable "eviction occurred" signal that
    /// is false in all normal (non-evicted) operation (issue #130). Only meaningful
    /// when this config has attention layers (<c>_numAttnLayers &gt; 0</c>); a
    /// pure-GDN model never compacts. <c>_kvCache</c> is always constructed, but the
    /// null-guard mirrors the defensive style used elsewhere.</summary>
    private bool KvCacheCompacted =>
        _numAttnLayers > 0 && _kvCache is not null && _kvCache.Length != _kvCache.LogicalLength;

    /// <inheritdoc />
    /// Issue #45: MoE MTP is supported via the CPU-MoE path (SHARPI_CPU_MOE=1 or
    /// auto-routed on 12 GB-class cards). Full-GPU MoE (rare, ≥24 GB cards) still
    /// falls back to sequential decode — folding a 2-token batched-verify into
    /// GpuMoeFfn's SLRU loop is tracked separately.
    /// Issue #130: batched-verify (BatchForward2) cannot run on a SnapKV-evicted
    /// cache — its precondition requires _kvCache.Length == startPos (the logical
    /// RoPE position), but eviction leaves Length at the budget K while the logical
    /// position stays at the prompt length N. We gate off when the cache is compacted
    /// so MtpDecoder falls back to the eviction-safe sequential Forward path; making
    /// batched-verify coexist with eviction is the #130 follow-up.
    public bool SupportsBatchVerify => _hasMtp
        && (!_hp.IsMoE || _cpuMoe)
        && (_cpuGdn || _gdnRingSlots >= 1)
        && !KvCacheCompacted
        && Environment.GetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY") != "1";

    /// <inheritdoc/>
    // #219: the verify lm_head logits are produced on-device anyway, so the greedy verify can
    // reduce them to per-position argmaxes here instead of downloading k×vocab. Same kill switch.
    public bool SupportsBatchVerifyArgmax => SupportsBatchVerify && _gpu.GpuArgmaxEnabled;

    /// <inheritdoc />
    /// On the GPU-GDN trunk the ceiling is the device snapshot ring's capacity
    /// (slots + 1, reserved at construction — SHARPI_MTP_BATCH_MAX). The
    /// SHARPI_CPU_GDN=1 debug trunk keeps the legacy 2-token path.
    public int MaxBatchVerifyTokens => _cpuGdn ? 2 : _gdnRingSlots + 1;

    /// <inheritdoc />
    public ReadOnlySpan<float> HiddenAt(int position)
    {
        if (!_hasMtp || position < 0 || position >= _mtpHiddenHistoryLength)
            return default;
        return new ReadOnlySpan<float>(_mtpPrefillHiddens + (long)position * _embDim, _embDim);
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpLastHidden =>
        _mtpSelfHidden != null ? new ReadOnlySpan<float>(_mtpSelfHidden, _embDim) : default;

    /// <inheritdoc />
    public void BatchForward2(int t1, int t2, int startPos,
        out ReadOnlySpan<float> logits1, out ReadOnlySpan<float> logits2)
    {
        ThrowIfFaulted();
        if (!SupportsBatchVerify)
            throw new InvalidOperationException(
                "BatchForward2 is only supported on dense-FFN MTP passes. " +
                "Check SupportsBatchVerify before calling.");
        if (startPos < 0)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, "startPos must be >= 0.");
        if (_kvCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchForward2: _kvCache.Length={_kvCache.Length} != startPos={startPos}. " +
                "A SnapKV-evicted (compacted) cache is unsupported here (issue #130) — callers " +
                "must check SupportsBatchVerify, which returns false once the cache is compacted, " +
                "and fall back to the sequential Forward path.");
        if (_gdnStateCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchForward2: _gdnStateCache.Length={_gdnStateCache.Length} != startPos={startPos}.");

        // 1. Embed both tokens into independent residual streams.
        EmbedToken(_gpuHidden,  t1);
        EmbedToken(_gpuHidden2, t2);

        // 2. Reserve KV blocks covering both positions on the CPU-side block table.
        _kvCache.ReserveBlockAt(startPos);
        _kvCache.ReserveBlockAt(startPos + 1);

        _batchSnapshotValid = false;
        long layerSnapBytes = _gdnStateCache.LayerSnapshotBytes;

        // 3. Trunk layers — sequential per-token attn/GDN within a layer, then
        //    batched CPU FFN (where the weight-bandwidth win lives), or two
        //    sequential GPU dense FFNs (GPU layers benefit from L2 weight reuse
        //    between the two MatMuls).
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Pre-block residual + attn-norm for both.
            _gpu.CopyDevice(_gpuResidual,  _gpuHidden);
            _gpu.CopyDevice(_gpuResidual2, _gpuHidden2);
            _gpu.RmsNorm(_gpuNormBuf,  _gpuHidden,  _gpuAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RmsNorm(_gpuNormBuf2, _gpuHidden2, _gpuAttnNorm[layer], _hp.RmsNormEps);

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            if (isAttn)
            {
                GpuAttnBlockAt(layer, position: startPos,     kvPosition: startPos,
                               normIn: _gpuNormBuf,  hiddenOut: _gpuHidden);
                GpuAttnBlockAt(layer, position: startPos + 1, kvPosition: startPos + 1,
                               normIn: _gpuNormBuf2, hiddenOut: _gpuHidden2);
            }
            else if (_cpuGdn)
            {
                CpuGdnBlockAt(layer, position: startPos,     normInGpu: _gpuNormBuf,
                              hiddenOutGpu: _gpuHidden, cpuNormScratch: _cpuNormBuf,
                              cpuHiddenScratch: _cpuHiddenOut);
                // Snapshot this layer's GDN state right after t1's CPU GDN update.
                int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
                _gdnStateCache.SnapshotLayerInto(gdnIdx,
                    _batchSnapshotBuf + (long)gdnIdx * layerSnapBytes,
                    layerSnapBytes);
                CpuGdnBlockAt(layer, position: startPos + 1, normInGpu: _gpuNormBuf2,
                              hiddenOutGpu: _gpuHidden2, cpuNormScratch: _cpuNormBuf2,
                              cpuHiddenScratch: _cpuMoeHidden2);
            }
            else
            {
                GpuGdnBlockAt(layer, position: startPos,     normIn: _gpuNormBuf,  hiddenOut: _gpuHidden);
                // Snapshot t1's state into DEVICE ring slot 0 — the live state on
                // this trunk is the _gpuGdnScanState/_gpuGdnConvState tensors, not
                // the host _gdnStateCache (which is stale outside CaptureSnapshot;
                // the pre-#207 host SnapshotLayerInto here silently captured stale
                // bytes, so a rejected draft never actually rewound the GPU state).
                CaptureGdnRingSlot(slot: 0, layer);
                GpuGdnBlockAt(layer, position: startPos + 1, normIn: _gpuNormBuf2, hiddenOut: _gpuHidden2);
            }

            // Residual add for both.
            _gpu.AddInPlace(_gpuHidden,  _gpuResidual);
            _gpu.AddInPlace(_gpuHidden2, _gpuResidual2);

            // Pre-FFN residual + post-norm for both.
            _gpu.CopyDevice(_gpuResidual,  _gpuHidden);
            _gpu.CopyDevice(_gpuResidual2, _gpuHidden2);
            _gpu.RmsNorm(_gpuNormBuf,  _gpuHidden,  _gpuPostAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RmsNorm(_gpuNormBuf2, _gpuHidden2, _gpuPostAttnNorm[layer], _hp.RmsNormEps);

            // FFN dispatch.
            //   Dense GPU layer  → batched GpuDenseFfn2At (issue #43 — single
            //                      weight read per row, two outputs).
            //   Dense CPU layer  → batched CpuDenseFfn2 (MatVec2In win).
            //   MoE CPU layer    → two sequential CpuMoeFfn calls (issue #45).
            //   MoE GPU layer    → two sequential GpuMoeFfn calls.
            // The MoE per-token TopK usually differs across t1 and t2 so no
            // routed-expert weight sharing is possible; the wins for MoE come
            // from amortising lm_head + norms + KV pages, not the FFN itself.
            if (!_hp.IsMoE)
            {
                if (_gpuWFfnGate is not null && _gpuWFfnGate[layer] is not null)
                {
                    GpuDenseFfn2At(layer,
                        normIn1:    _gpuNormBuf,
                        normIn2:    _gpuNormBuf2,
                        hiddenOut1: _gpuHidden,
                        hiddenOut2: _gpuHidden2,
                        gateBuf1:   _gpuFfnGateBufDense!,
                        gateBuf2:   _gpuFfnGateBufDense2!,
                        upBuf1:     _gpuFfnUpBufDense!,
                        upBuf2:     _gpuFfnUpBufDense2!);
                }
                else
                {
                    // Direct-pinned overloads (issue #48).
                    _gpu.Download(_gpuNormBuf,  (nint)_cpuNormBuf,  _embDim);
                    _gpu.Download(_gpuNormBuf2, (nint)_cpuNormBuf2, _embDim);
                    CpuDenseFfn2(layer, _cpuNormBuf, _cpuNormBuf2,
                                 _cpuMoeHidden, _cpuMoeHidden2);
                    _gpu.UploadInto(_gpuHidden,  (nint)_cpuMoeHidden,  _embDim);
                    _gpu.UploadInto(_gpuHidden2, (nint)_cpuMoeHidden2, _embDim);
                }
            }
            else if (_cpuMoe)
            {
                // Direct-pinned overloads (issue #48).
                // t1
                _gpu.Download(_gpuNormBuf,  (nint)_cpuNormBuf,  _embDim);
                CpuMoeFfnCore(
                    _gpuWGateShexp[layer], _gpuWUpShexp[layer], _gpuWDownShexp[layer],
                    _cpuFfnGateInp![layer], _cpuFfnGateInpShexp![layer],
                    _cpuFfnGateExps![layer], _cpuFfnUpExps![layer], _cpuFfnDownExps![layer],
                    gpuNormIn: _gpuNormBuf, gpuSharedOut: _gpuSharedOut,
                    cpuNormIn: _cpuNormBuf, cpuMoeOut: _cpuMoeHidden);
                _gpu.UploadInto(_gpuHidden,  (nint)_cpuMoeHidden,  _embDim);
                // t2
                _gpu.Download(_gpuNormBuf2, (nint)_cpuNormBuf2, _embDim);
                CpuMoeFfnCore(
                    _gpuWGateShexp[layer], _gpuWUpShexp[layer], _gpuWDownShexp[layer],
                    _cpuFfnGateInp![layer], _cpuFfnGateInpShexp![layer],
                    _cpuFfnGateExps![layer], _cpuFfnUpExps![layer], _cpuFfnDownExps![layer],
                    gpuNormIn: _gpuNormBuf2, gpuSharedOut: _gpuSharedOut,
                    cpuNormIn: _cpuNormBuf2, cpuMoeOut: _cpuMoeHidden2);
                _gpu.UploadInto(_gpuHidden2, (nint)_cpuMoeHidden2, _embDim);
            }
            else
            {
                // Full-GPU MoE (rare; only on ≥24 GB cards). SupportsBatchVerify
                // gates this off so MtpDecoder falls back to sequential decode.
                throw new InvalidOperationException(
                    "BatchForward2 reached the full-GPU MoE branch — SupportsBatchVerify " +
                    "should have steered MtpDecoder to the sequential path. " +
                    "Folding 2-token verify into GpuMoeFfn's SLRU loop is a follow-up.");
            }

            // Post-FFN residual add for both.
            _gpu.AddInPlace(_gpuHidden,  _gpuResidual);
            _gpu.AddInPlace(_gpuHidden2, _gpuResidual2);
        }

        // 4. Advance both caches by 2.
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();
        _batchStartPos = startPos;
        _batchK = 2;

        // 5. Snapshot pre-output-norm hiddens for MTP commit + next iter draft.
        //    Issue #49: queue both snapshots via DownloadAsync (pinned host
        //    targets) so the lm_head MatMuls below run concurrently with the
        //    queued PCIe transfers. The logits Downloads at the end sync the
        //    stream and drain everything.
        _gpu.CopyDevice(_gpuLastHiddenT1, _gpuHidden);
        _gpu.CopyDevice(_gpuLastHidden,   _gpuHidden2);
        _gpu.DownloadAsync(_gpuLastHiddenT1, (nint)_lastHiddenT1, _embDim);
        _gpu.DownloadAsync(_gpuLastHidden,   (nint)_lastHidden,   _embDim);

        // 6. Final norm + output projection for both tokens.
        _gpu.RmsNorm(_gpuHidden,  _gpuHidden,  _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.RmsNorm(_gpuHidden2, _gpuHidden2, _gpuOutputNorm!, _hp.RmsNormEps);
        var outDt = _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var dt) ? dt : DType.Float32;
        _gpu.MatMul(_gpuLogits,  _gpuOutputWeight!, _gpuHidden,  outDt);
        _gpu.MatMul(_gpuLogits2, _gpuOutputWeight!, _gpuHidden2, outDt);

        _gpu.Download(_gpuLogits,  _logitsBuf);
        _gpu.Download(_gpuLogits2, _logitsBuf2);

        // Issue #106: mirror the two host hiddens (now landed via the syncing
        // Downloads above) into the absolute-position history buffer at slots
        // startPos and startPos+1. RestoreBatchSnapshot shrinks the count if
        // t2 is rejected; the follow-up Forward(corrected_t2, startPos+1) then
        // rewrites slot startPos+1.
        if (_hasMtp)
        {
            EnsureMtpHiddenHistoryCap(startPos + 2);
            new ReadOnlySpan<float>(_lastHiddenT1, _embDim).CopyTo(
                new Span<float>(_mtpPrefillHiddens + (long)startPos       * _embDim, _embDim));
            new ReadOnlySpan<float>(_lastHidden,   _embDim).CopyTo(
                new Span<float>(_mtpPrefillHiddens + (long)(startPos + 1) * _embDim, _embDim));
            if (_mtpHiddenHistoryLength < startPos + 2)
                _mtpHiddenHistoryLength = startPos + 2;
        }

        _batchSnapshotValid = true;
        logits1 = _logitsBuf;
        logits2 = _logitsBuf2;
    }

    /// <inheritdoc />
    /// <summary>
    /// Roll the caches back to an intermediate point of the most recent batched
    /// verify using the GDN snapshot ring: slot <c>lengthAfter - startPos - 1</c>
    /// holds the state after the batch token at position <c>lengthAfter - 1</c>.
    /// GPU-GDN trunk: device-to-device ring → live state tensors (the host
    /// _gdnStateCache stays stale, as in normal GPU operation — only its length
    /// is bookkept). SHARPI_CPU_GDN=1 trunk: host ring → host state.
    /// </summary>
    public void RestoreBatchSnapshot(int lengthAfter)
    {
        if (!_batchSnapshotValid)
            throw new InvalidOperationException(
                "RestoreBatchSnapshot: no batched-verify snapshot is held. " +
                "Call BatchForward2 or BatchVerify first.");
        int slot = lengthAfter - _batchStartPos - 1;
        if (slot < 0 || slot >= _batchK - 1)
            throw new ArgumentOutOfRangeException(nameof(lengthAfter), lengthAfter,
                $"RestoreBatchSnapshot: lengthAfter must be in [{_batchStartPos + 1}, " +
                $"{_batchStartPos + _batchK - 1}] — the most recent batched verify " +
                $"covered positions [{_batchStartPos}, {_batchStartPos + _batchK}).");

        if (_cpuGdn)
        {
            long layerSnapBytes = _gdnStateCache.LayerSnapshotBytes;
            long slotBytes = layerSnapBytes * _gdnStateCache.NumGdnLayers;
            for (int gdnIdx = 0; gdnIdx < _gdnStateCache.NumGdnLayers; gdnIdx++)
            {
                _gdnStateCache.RestoreLayerFrom(gdnIdx,
                    _batchSnapshotBuf + slot * slotBytes + (long)gdnIdx * layerSnapBytes,
                    layerSnapBytes);
            }
        }
        else
        {
            for (int layer = 0; layer < _hp.NumLayers; layer++)
                RestoreGdnRingSlot(slot, layer);
        }
        _gdnStateCache.SetLength(lengthAfter);
        _kvCache.TruncateTo(lengthAfter);
        // Atomic with the trunk rewind — see HybridGdnForwardPass.RestoreBatchSnapshot.
        _mtpKvCache?.TruncateTo(lengthAfter);
        if (_hasMtp && _mtpHiddenHistoryLength > lengthAfter)
            _mtpHiddenHistoryLength = lengthAfter;
        _batchSnapshotValid = false;
    }

    /// <summary>
    /// Copy one GDN layer's live device state (scan + conv) into ring slot
    /// <paramref name="slot"/> at the layer's packed offset. No-op for attention
    /// layers. Device-to-device, stream-ordered — callers issue it right after the
    /// layer's token-<c>slot</c> recurrence update.
    /// </summary>
    private void CaptureGdnRingSlot(int slot, int layer)
    {
        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        if (gdnIdx < 0) return;
        if (_gpuGdnRingScan is null || slot >= _gdnRingSlots)
            throw new InvalidOperationException(
                $"CaptureGdnRingSlot({slot}): the GDN snapshot ring has {_gdnRingSlots} slot(s). " +
                "Callers must clamp the batch to MaxBatchVerifyTokens.");
        long scanF = _gdnStateCache.ScanStateFloatsPerLayer;
        long convF = _gdnStateCache.ConvStateFloatsPerLayer;
        long numGdn = _gdnStateCache.NumGdnLayers;
        long scanBytes = scanF * sizeof(float);
        long convBytes = convF * sizeof(float);
        if (_gpuGdnScanState[layer] is { } scanT && scanBytes > 0)
            _gpu.CopyDeviceRegion(_gpuGdnRingScan, ((long)slot * numGdn * scanF + gdnIdx * scanF) * sizeof(float),
                scanT, 0, scanBytes);
        if (_gpuGdnConvState[layer] is { } convT && convBytes > 0 && _gpuGdnRingConv is { } convRing)
            _gpu.CopyDeviceRegion(convRing, ((long)slot * numGdn * convF + gdnIdx * convF) * sizeof(float),
                convT, 0, convBytes);
    }

    /// <summary>Inverse of <see cref="CaptureGdnRingSlot"/>: ring slot → live device state.</summary>
    private void RestoreGdnRingSlot(int slot, int layer)
    {
        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        if (gdnIdx < 0) return;
        if (_gpuGdnRingScan is null || slot >= _gdnRingSlots)
            throw new InvalidOperationException(
                $"RestoreGdnRingSlot({slot}): the GDN snapshot ring has {_gdnRingSlots} slot(s).");
        long scanF = _gdnStateCache.ScanStateFloatsPerLayer;
        long convF = _gdnStateCache.ConvStateFloatsPerLayer;
        long numGdn = _gdnStateCache.NumGdnLayers;
        long scanBytes = scanF * sizeof(float);
        long convBytes = convF * sizeof(float);
        if (_gpuGdnScanState[layer] is { } scanT && scanBytes > 0)
            _gpu.CopyDeviceRegion(scanT, 0, _gpuGdnRingScan, ((long)slot * numGdn * scanF + gdnIdx * scanF) * sizeof(float),
                scanBytes);
        if (_gpuGdnConvState[layer] is { } convT && convBytes > 0 && _gpuGdnRingConv is { } convRing)
            _gpu.CopyDeviceRegion(convT, 0, convRing, ((long)slot * numGdn * convF + gdnIdx * convF) * sizeof(float),
                convBytes);
    }

    /// <summary>
    /// k-token batched verify for the MTP folded decode loop (issues #30 /
    /// #207 goal 4). The trunk runs as the #111/#114-B batched-prefill launches —
    /// GEMM-batched projections, batched attention at contiguous positions
    /// <c>[startPos, startPos+k)</c>, per-position delta-net recurrence with the
    /// device GDN snapshot ring captured at every token boundary — followed by the
    /// FFN stage (GEMM-N on GPU dense layers; pair-batched <c>MatVec2In</c> on the
    /// CPU mmap layers that dominate 27B decode; per-token CPU MoE) and an
    /// all-position [k × vocab] lm_head. Returns <c>result[i]</c> = logits after
    /// <c>tokens[i]</c>; rollback is <see cref="RestoreBatchSnapshot"/>.
    /// </summary>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        ThrowIfFaulted();
        ArgumentNullException.ThrowIfNull(tokens);
        if (!SupportsBatchVerify)
            throw new InvalidOperationException(
                "BatchVerify requires an MTP pass with an uncompacted cache and an " +
                "available GDN snapshot ring. Check SupportsBatchVerify before calling.");
        int k = tokens.Length;
        if (k == 0) return Array.Empty<float[]>();
        if (startPos < 0 || startPos + k > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos),
                $"BatchVerify range [{startPos}, {startPos + k}) exceeds the context window (maxSeqLen={_maxSeqLen}).");
        if (k > MaxBatchVerifyTokens)
            throw new ArgumentOutOfRangeException(nameof(tokens), k,
                $"BatchVerify token count exceeds MaxBatchVerifyTokens ({MaxBatchVerifyTokens}); " +
                "raise SHARPI_MTP_BATCH_MAX (ring slots are reserved at construction).");
        if (k == 1)
        {
            // A single token amortizes nothing — plain Forward is strictly better.
            var l = Forward(tokens[0], startPos);
            return [l.ToArray()];
        }
        if (_cpuGdn)
        {
            // SHARPI_CPU_GDN=1 debug trunk: the host-side GDN state is live, so the
            // legacy 2-token path with its host snapshot is correct as-is.
            // MaxBatchVerifyTokens caps k at 2 in this config.
            BatchForward2(tokens[0], tokens[1], startPos, out var l1, out var l2);
            return [l1.ToArray(), l2.ToArray()];
        }
        if (_kvCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _kvCache.Length={_kvCache.Length} != startPos={startPos}. " +
                "Caches must sit exactly at startPos (a SnapKV-compacted cache is gated " +
                "off via SupportsBatchVerify).");
        if (_gdnStateCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _gdnStateCache.Length={_gdnStateCache.Length} != startPos={startPos}.");

        int embDim = _embDim;
        long embBytes = (long)embDim * sizeof(float);
        bool isMoe = _hp.IsMoE;
        // Snapshot the settable toggle once: the scratch alloc below and the FFN-branch
        // select in the layer loop must agree, or we'd allocate-without-use (or worse,
        // use-without-alloc) if the test flipped it mid-call.
        bool batchedMoeVerify = isMoe && BatchedMoeVerifyEnabled;

        EnsureStreamAll(k);
        EnsureBatchedTrunkScratch(k);
        if (BatchedFfnEnabled && !isMoe && _denseFfnGpuLayers > 0)
            EnsureBatchedFfnScratch(k);
        EnsureBatchVerifyScratch(k);
        // Group-by-expert routed FFN (issue #210) reuses the batched-prefill host
        // scratch (_bNormAll / _bSelected / _bRoutedAll / bucket buffers). Grow-only;
        // a no-op when a prior prefill already sized it past k.
        if (batchedMoeVerify)
            EnsureBatchedScratch(k);

        // Pessimistic fault latch — same contract as the batched prefills: a
        // mid-pass throw leaves the recurrent state partially advanced while the
        // length counters still read startPos; fatal for this pass.
        _faulted = true;
        _batchSnapshotValid = false;

        var stream = _gpuStreamAll!;

        // 1. Embed every token into the residual-stream buffer + reserve KV blocks.
        for (int i = 0; i < k; i++)
        {
            EmbedToken(_gpuHidden, tokens[i]);
            _gpu.CopyDeviceRegion(stream, i * embBytes, _gpuHidden, 0, embBytes);
            _kvCache.ReserveBlockAt(startPos + i);
        }

        // 2. Trunk (batched, ring-capturing) + FFN stage, layer by layer.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            TrunkBlockBatched(layer, k, startPos, isAttn, snapKvActive: false, wStart: 0,
                              gdnSnapRing: true);
            var blockOut = _gpuBtBlockOut!;
            var moeNorm  = _gpuBtMoeNorm!;

            bool denseGpuLayer = !isMoe && _gpuWFfnGate is not null && _gpuWFfnGate[layer] is not null;
            bool batchLayer = BatchedFfnEnabled && denseGpuLayer
                && BatchedMatMulSupported(_gpuWFfnGate![layer]!)
                && BatchedMatMulSupported(_gpuWFfnUp![layer]!)
                && BatchedMatMulSupported(_gpuWFfnDown![layer]!);
            if (batchLayer)
            {
                // GEMM-N gate/up/down over all k tokens (issue #121 machinery).
                BatchedGpuDenseFfn(layer, k, moeNorm, _gpuBfHiddenAll!);
                _gpu.AddInPlace(_gpuBfHiddenAll!, blockOut);
                _gpu.CopyDeviceRegion(stream, 0, _gpuBfHiddenAll!, 0, k * embBytes);
            }
            else if (!isMoe && !denseGpuLayer)
            {
                // CPU mmap dense FFN — the 27B/12GB decode cost center (~8.6 GB
                // weight reads per token). Quad-batched MatVec4In reads each weight
                // row once per four tokens (issue #209); the final partial group's
                // duplicated-tail lanes re-run the last real token with their output
                // routed to a shared sink, so every token's bits match the quad
                // kernel regardless of k parity (per-position k-parity independence).
                _gpu.Download(moeNorm, (nint)_bvNormHost, k * embDim);
                for (int i = 0; i < k; i += 4)
                {
                    MtpBatchTail.Group4(i, k, out int j0, out int j1, out int j2, out int j3, out int nReal);
                    CpuDenseFfn4(layer,
                        _bvNormHost + (long)j0 * embDim, _bvNormHost + (long)j1 * embDim,
                        _bvNormHost + (long)j2 * embDim, _bvNormHost + (long)j3 * embDim,
                        _bvFfnHost + (long)j0 * embDim,
                        nReal > 1 ? _bvFfnHost + (long)j1 * embDim : _cpuMoeHidden2,
                        nReal > 2 ? _bvFfnHost + (long)j2 * embDim : _cpuMoeHidden2,
                        nReal > 3 ? _bvFfnHost + (long)j3 * embDim : _cpuMoeHidden2);
                }
                _gpu.UploadInto(_gpuBvFfnAll!, (nint)_bvFfnHost, k * embDim);
                _gpu.AddInPlace(_gpuBvFfnAll!, blockOut);
                _gpu.CopyDeviceRegion(stream, 0, _gpuBvFfnAll!, 0, k * embBytes);
            }
            else if (batchedMoeVerify)
            {
                // Issue #210: group the k draft tokens by selected expert so each
                // expert's mmap'd rows are read once across the chain. Bit-identical
                // routed output to the per-token CpuMoeFfnCore loop below.
                BatchVerifyCpuMoe(layer, k, moeNorm, blockOut, stream);
            }
            else
            {
                // Per-token fallbacks: GPU dense layer with a non-GEMM-N weight
                // dtype, or CPU MoE (per-token routing, issue #45 — the wins come
                // from the batched trunk + lm_head, not the routed FFN itself;
                // SHARPI_MTP_BATCHED_MOE_VERIFY=0 forces this path for MoE too).
                // Full-GPU MoE never reaches here (SupportsBatchVerify gate).
                for (int i = 0; i < k; i++)
                {
                    _gpu.CopyDeviceRegion(_gpuNormBuf, 0, moeNorm, i * embBytes, embBytes);
                    if (!isMoe)
                    {
                        GpuDenseFfn(layer);
                    }
                    else
                    {
                        _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, embDim);
                        CpuMoeFfnCore(
                            _gpuWGateShexp[layer], _gpuWUpShexp[layer], _gpuWDownShexp[layer],
                            _cpuFfnGateInp![layer], _cpuFfnGateInpShexp![layer],
                            _cpuFfnGateExps![layer], _cpuFfnUpExps![layer], _cpuFfnDownExps![layer],
                            gpuNormIn: _gpuNormBuf, gpuSharedOut: _gpuSharedOut,
                            cpuNormIn: _cpuNormBuf, cpuMoeOut: _cpuMoeHidden);
                        _gpu.UploadInto(_gpuHidden, (nint)_cpuMoeHidden, embDim);
                    }
                    _gpu.CopyDeviceRegion(_gpuResidual, 0, blockOut, i * embBytes, embBytes);
                    _gpu.AddInPlace(_gpuHidden, _gpuResidual);
                    _gpu.CopyDeviceRegion(stream, i * embBytes, _gpuHidden, 0, embBytes);
                }
            }
        }

        // 3. Advance the position counters by k.
        for (int i = 0; i < k; i++)
        {
            _kvCache.IncrementPosition();
            _gdnStateCache.IncrementPosition();
        }
        _batchStartPos = startPos;
        _batchK = k;
        _faulted = false;

        // 4. MTP hidden history (issues #33/#106): stream[i] holds the
        //    pre-output-norm hidden for token startPos+i.
        EnsureMtpHiddenHistoryCap(startPos + k);
        _gpu.Download(stream, (nint)(_mtpPrefillHiddens + (long)startPos * embDim), k * embDim);
        if (_mtpHiddenHistoryLength < startPos + k)
            _mtpHiddenHistoryLength = startPos + k;
        new ReadOnlySpan<float>(_mtpPrefillHiddens + (long)(startPos + k - 1) * embDim, embDim)
            .CopyTo(new Span<float>(_lastHidden, embDim));

        // 5. All-position logits: batched output norm + GEMM-N lm_head when the
        //    weight dtype supports it, else a per-token MatMul loop.
        var normAll = _gpuBtNorm!;   // free to reuse after the last trunk layer
        _gpu.RmsNormBatched(normAll, stream, _gpuOutputNorm!, k, embDim, _hp.RmsNormEps);

        // #219 greedy verify (pMin=1.0): reduce the lm_head output to one (idx, value) per position
        // on-device — a k*8-byte download — instead of materializing and downloading k×vocab logits.
        if (_bvArgmaxOnly)
        {
            if (BatchedMatMulSupported(_gpuOutputWeight!))
            {
                GpuMatMulBatched(_gpuBvLogitsAll!, _gpuOutputWeight!, normAll, k);
                _bvArgmaxResult = _gpu.ArgmaxRows(_gpuBvLogitsAll!, k, _hp.VocabSize, _hp.VocabSize);
            }
            else
            {
                var outDtA = _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var dtA)
                    ? dtA : DType.Float32;
                _bvArgmaxResult = new (int, float)[k];
                for (int i = 0; i < k; i++)
                {
                    _gpu.CopyDeviceRegion(_gpuHidden, 0, normAll, i * embBytes, embBytes);
                    _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden, outDtA);
                    _bvArgmaxResult[i] = _gpu.Argmax(_gpuLogits);
                }
            }
            _batchSnapshotValid = true;
            return [];
        }

        var result = new float[k][];
        if (BatchedMatMulSupported(_gpuOutputWeight!))
        {
            GpuMatMulBatched(_gpuBvLogitsAll!, _gpuOutputWeight!, normAll, k);
            _gpu.Download(_gpuBvLogitsAll!, _bvLogitsHost.AsSpan(0, k * _hp.VocabSize));
            for (int i = 0; i < k; i++)
            {
                result[i] = new float[_hp.VocabSize];
                Array.Copy(_bvLogitsHost!, (long)i * _hp.VocabSize, result[i], 0, _hp.VocabSize);
            }
        }
        else
        {
            var outDt = _gpuWeightDTypes.TryGetValue(_gpuOutputWeight!.Handle, out var dt)
                ? dt : DType.Float32;
            for (int i = 0; i < k; i++)
            {
                _gpu.CopyDeviceRegion(_gpuHidden, 0, normAll, i * embBytes, embBytes);
                _gpu.MatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden, outDt);
                _gpu.Download(_gpuLogits, _logitsBuf);
                result[i] = (float[])_logitsBuf.Clone();
            }
        }

        _batchSnapshotValid = true;
        return result;
    }

    /// <summary>
    /// Routed-expert FFN for a <see cref="BatchVerify"/> draft batch, grouped by
    /// selected expert (issue #210). Mirrors <see cref="CpuMoeFfnCore"/> per token —
    /// GPU shared expert scaled by the sigmoid gate, host router top-K — but routes the
    /// dominant routed-expert dots through <see cref="BatchedRoutedExperts"/> so each
    /// selected expert's mmap'd gate/up/down rows are read once across the chain instead
    /// of re-read per token. Bit-identical to the per-token loop it replaces: the routed
    /// dots run the same DispatchDot kernels in the same top-k accumulation order, the
    /// shared expert uses the identical per-token GPU matvecs, and the (routed + shared)
    /// + resid combine keeps the per-token operand order (host add of routed+shared, GPU
    /// add of the block residual over all k tokens).
    /// </summary>
    private void BatchVerifyCpuMoe(int layer, int k, Tensor moeNorm, Tensor blockOut, Tensor stream)
    {
        int embDim = _embDim;
        long embBytes = (long)embDim * sizeof(float);
        int na = _numActiveExperts;

        // All k post-attn norms to host — the routed-expert dot input and the
        // router / shared-gate input. A pure memcpy, so byte-identical to the
        // per-token Download(_gpuNormBuf → _cpuNormBuf) the sequential path runs.
        _gpu.Download(moeNorm, (nint)_bNormAll, k * embDim);

        var routerW = _cpuFfnGateInp![layer];
        float* gateInpShexp = _cpuFfnGateInpShexp![layer];
        Tensor gateShexp = _gpuWGateShexp[layer];
        Tensor upShexp = _gpuWUpShexp[layer];
        Tensor downShexp = _gpuWDownShexp[layer];
        Span<int> sel = stackalloc int[na];
        Span<float> wts = stackalloc float[na];

        // ── Router top-K per token (host) into the grouped-expert bucket inputs.
        for (int i = 0; i < k; i++)
        {
            float* normI = _bNormAll + (long)i * embDim;
            SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, normI,
                _numExperts, embDim, routerW.DType);
            SimdKernels.SoftmaxInPlace(_cpuRouterLogits, _numExperts);
            SelectTopKPtr(_cpuRouterLogits, _numExperts, na, sel, wts,
                normalize: _hp.NormalizeMoeTopKWeights);
            for (int s = 0; s < na; s++)
            {
                _bSelected[(long)i * na + s] = sel[s];
                _bWeights[(long)i * na + s] = wts[s];
            }
        }

        // ── Kick the k GPU shared experts (scaled) onto the stream, staged into
        //    _gpuBvFfnAll. These are enqueued async and NOT synced here, so they run
        //    on the GPU while the host computes the routed experts below — mirroring
        //    CpuMoeFfnCore's GPU-shared / CPU-routed overlap, but in bulk. Reusing
        //    _gpuSharedOut / _gpuFfnGate / _gpuFfnUp across tokens is safe: the single
        //    stream serializes each token's matvec → scale → stage before the next
        //    token's matvec overwrites them.
        for (int i = 0; i < k; i++)
        {
            float* normI = _bNormAll + (long)i * embDim;
            _gpu.CopyDeviceRegion(_gpuNormBuf, 0, moeNorm, i * embBytes, embBytes);
            GpuMatMul(_gpuFfnGate, gateShexp, _gpuNormBuf);
            GpuMatMul(_gpuFfnUp, upShexp, _gpuNormBuf);
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuSharedOut, downShexp, _gpuFfnGate);
            float shexpDot = SimdKernels.DotF32(gateInpShexp, normI, embDim);
            float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
            _gpu.ScaleInPlace(_gpuSharedOut, shexpScale);
            _gpu.CopyDeviceRegion(_gpuBvFfnAll!, i * embBytes, _gpuSharedOut, 0, embBytes);
        }

        // ── Group-by-expert routed FFN (the issue's amortization, host). Reads each
        //    selected expert's rows once; output is bit-identical to the per-token
        //    routed accumulator CpuMoeFfnCore builds. Overlaps the in-flight GPU
        //    shared experts above.
        BatchedRoutedExperts(layer, k);

        // Scaled shared-expert outputs for every token → host (single sync; the GPU
        // work is largely hidden behind the routed compute that just ran).
        _gpu.Download(_gpuBvFfnAll!, (nint)_bSharedAll, k * embDim);

        // ── Combine: hidden = routed + sharedScaled on the host (matching
        //    CpuMoeFfnCore's AddInPlace(moeOut, _cpuSharedOut) operand order), then add
        //    the block residual on the GPU over all k tokens (element-wise → per-token
        //    bits). One flat pass over the contiguous [k×embDim] buffers: k is the tiny
        //    draft batch (2–4), so a sequential loop beats Parallel.For here — TPL
        //    scheduling plus the closure heap-alloc would dwarf the work. (The prefill
        //    combine parallelizes only because there N ≈ the prefill chunk size.)
        long combineCount = (long)k * embDim;
        for (long r = 0; r < combineCount; r++)
            _bHiddenAll[r] = _bRoutedAll[r] + _bSharedAll[r];
        _gpu.UploadInto(_gpuBvFfnAll!, (nint)_bHiddenAll, k * embDim);
        _gpu.AddInPlace(_gpuBvFfnAll!, blockOut);
        _gpu.CopyDeviceRegion(stream, 0, _gpuBvFfnAll!, 0, (long)k * embBytes);
    }

    /// <inheritdoc/>
    public (int Index, float Value)[] BatchVerifyArgmax(int[] tokens, int startPos)
    {
        _bvArgmaxOnly = true;
        try { BatchVerify(tokens, startPos); }   // identical trunk + cache effects; tail does the argmax
        finally { _bvArgmaxOnly = false; }
        return _bvArgmaxResult;
    }

    /// <summary>
    /// (Re)allocate the batched-verify scratch for an exact batch size of
    /// <paramref name="k"/> tokens: the [k × vocab] all-position logits tensor +
    /// managed download buffer, the [k × embDim] CPU-FFN staging tensor, and the
    /// pinned host norm/FFN roundtrip buffers. Exact-size (not grow-only) because
    /// the GEMM-N kernels derive their row count from <c>ElementCount / k</c>.
    /// </summary>
    private void EnsureBatchVerifyScratch(int k)
    {
        if (_bvCap == k) return;
        if (_gpuBvLogitsAll is { } l) { _gpu.Free(l); _gpuBvLogitsAll = null; }
        if (_gpuBvFfnAll is { } f) { _gpu.Free(f); _gpuBvFfnAll = null; }
        if (_bvNormHost != null) { CudaBackend.FreePinnedHost((nint)_bvNormHost); _bvNormHost = null; }
        if (_bvFfnHost != null) { CudaBackend.FreePinnedHost((nint)_bvFfnHost); _bvFfnHost = null; }
        _bvCap = -1;   // a mid-sequence alloc failure must not leave a stale cap
                       // matching a future k (early return on half-built scratch)
        long logitsTotal = (long)k * _hp.VocabSize;
        if (logitsTotal > int.MaxValue)
            throw new NotSupportedException(
                $"Batched verify logits buffer ({k}×{_hp.VocabSize}) exceeds int.MaxValue.");
        _gpuBvLogitsAll = _gpu.Allocate(TensorShape.D1(logitsTotal));
        _gpuBvFfnAll = _gpu.Allocate(TensorShape.D1((long)k * _embDim));
        _bvNormHost = AllocPinnedL((long)k * _embDim);
        _bvFfnHost = AllocPinnedL((long)k * _embDim);
        _bvLogitsHost = new float[(int)logitsTotal];
        _bvCap = k;
    }

    // =================================================================
    //  GPU attention block — GLU-gated Q, partial NEOX RoPE on first 64 dims
    // =================================================================

    private void GpuAttnBlock(int layer, int position) =>
        // kvPosition tracks the next physical slot in the ring; after SnapKV
        // (issue #58) compaction this diverges from `position` (the logical
        // RoPE frame). For the single-token decode path the next slot is
        // always _kvCache.Length, which holds in both compacted and
        // un-compacted runs.
        GpuAttnBlockAt(layer, position, kvPosition: _kvCache.Length,
                       normIn: _gpuNormBuf, hiddenOut: _gpuHidden);

    /// <summary>
    /// GPU attention block parameterised on input-norm / output-hidden tensors and
    /// RoPE/KV position. Used by both <see cref="Forward"/> and <see cref="BatchForward2"/>.
    /// </summary>
    private void GpuAttnBlockAt(int layer, int position, int kvPosition,
                                Tensor normIn, Tensor hiddenOut)
    {
        int kvDim = _numKvHeads * _headDim;

        GpuMatMul(_gpuQGate, _gpuWQGate[layer], normIn);
        GpuMatMul(_gpuK, _gpuWK[layer], normIn);
        GpuMatMul(_gpuV, _gpuWV[layer], normIn);

        _gpu.SplitQG(_gpuQ, _gpuGate, _gpuQGate, _numHeads, _headDim);

        _gpu.HeadNorm(_gpuQ, _gpuQNorm[layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNorm(_gpuK, _gpuKNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);

        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);

        // SnapKV (issue #58): capture the post-RoPE / post-Q-norm query for
        // this (layer, token) into the scoring ring. Gated by the Prefill
        // wrapper — outside that path _snapKvCaptureSlot stays -1 and we skip.
        if (_snapKvCaptureSlot >= 0 && _snapKvQCapture is { } capBuf)
        {
            int attnIdx = _attnLayerIndexOf[layer];
            if (attnIdx >= 0)
            {
                int qDim = _numHeads * _headDim;
                long dstOffsetElems = ((long)attnIdx * _snapKvQCaptureW + _snapKvCaptureSlot)
                                      * qDim;
                _gpu.CopyDeviceRegion(capBuf, dstOffsetElems * sizeof(float),
                                      _gpuQ, 0, (long)qDim * sizeof(float));
            }
        }

        // seqLen = kvPosition + 1: number of populated slots after this token's
        // KvAppend lands. Forward passes kvPosition = _kvCache.Length, which
        // post-SnapKV-compaction is the compacted slot count rather than the
        // logical prompt length — same +1 invariant. BatchForward2 passes
        // kvPosition = startPos[+1] for two sequential appends within one call.
        if (_kvDType == DType.BFloat16)
            _gpu.KvAppendBf16(_gpuK, _gpuV, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, kvPosition, _maxSeqLen);
        else
            _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[layer]!, _gpuVCache[layer]!, kvDim, kvPosition, _maxSeqLen);

        int seqLen = kvPosition + 1;
        // Flash-decoding split-KV (#238) at long ctx; else the single-block kernel. Uses the
        // #237 grouped auto-select like the dense/MoE-hybrid passes. (No `maxSeqLen <= _maxSeqLen`
        // guard as on the dense path: this site always passes _maxSeqLen as the cache extent, so
        // nSplits == the buffer's nSplitsMax exactly — there is no per-call maxSeqLen to bound.)
        if (_splitKvPartialO is { } splitO && _splitKvPartialMeta is { } splitMeta
            && seqLen > GdnSplitMinSeq)
        {
            bool grouped = CudaForwardPass.ShouldUseGroupedSplit(_splitGroupedMode, _kvDType, _numHeads, _numKvHeads, seqLen);
            _gpu.AttentionSplitKv(_gpuQ, _gpuKCache[layer]!, _gpuVCache[layer]!, _gpuAttnOut, splitO, splitMeta,
                _kvDType, _numHeads, _numKvHeads, _headDim, seqLen, _maxSeqLen, attnScale: -1f, grouped: grouped);
        }
        else if (_kvDType == DType.BFloat16)
        {
            _gpu.AttentionBf16(_gpuQ, _gpuKCache[layer]!, _gpuVCache[layer]!, _gpuAttnOut,
                _gpuAttnScratch, _numHeads, _numKvHeads, _headDim, seqLen, _maxSeqLen);
        }
        else
        {
            _gpu.Attention(_gpuQ, _gpuKCache[layer]!, _gpuVCache[layer]!, _gpuAttnOut,
                _gpuAttnScratch, _numHeads, _numKvHeads, _headDim, seqLen, _maxSeqLen);
        }

        _gpu.SigmoidMulInPlace(_gpuAttnOut, _gpuGate);

        GpuMatMul(hiddenOut, _gpuWO[layer], _gpuAttnOut);
    }

    // =================================================================
    //  MTP / NEXTN head on GPU (issue #29)
    //  Mirror of HybridGdnForwardPass.MtpForward but with GPU-resident
    //  weights + MTP KV cache + shared scratch tensors.
    // =================================================================

    /// <inheritdoc />
    public bool HasMtpHead => _hasMtp;

    // True when the routed/shared MoE FFN runs on CPU (mmap weights + per-token
    // PCIe norm/hidden roundtrip) rather than via the GPU SLRU expert cache.
    // Driven by SHARPI_CPU_MOE or auto-selected from SLRU capacity. Exposed so
    // the CLI banner can report the actual MoE routing instead of guessing.
    public bool IsMoeOnCpu => _cpuMoe;

    /// <inheritdoc />
    public ReadOnlySpan<float> LastHidden =>
        _hasMtp ? new ReadOnlySpan<float>(_lastHidden, _embDim) : default;

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
    {
        ThrowIfFaulted();
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
        EmbedToken(_gpuMtpEmbedBuf, token);

        // 3. enorm(embedding) → _gpuMtpEnormBuf; hnorm(prevHidden) → _gpuMtpHnormBuf.
        _gpu.RmsNorm(_gpuMtpEnormBuf, _gpuMtpEmbedBuf,  _gpuMtpEnorm, _hp.RmsNormEps);
        _gpu.RmsNorm(_gpuMtpHnormBuf, _gpuLastHidden,   _gpuMtpHnorm, _hp.RmsNormEps);

        // 4. Concat [enorm(e) ‖ hnorm(h)] into _gpuMtpConcatBuf [embDim*2].
        //    Two device-side copies; total 40 KiB GPU memcpy for 27B.
        //    Issue #40: matches the transformers `Qwen3NextNextNDecoderLayer`
        //    reference (`torch.cat([enormed, hnormed], dim=-1)`); the inverted
        //    order produces 0% draft acceptance (see CPU MtpForward note).
        long embBytes = (long)_embDim * sizeof(float);
        _gpu.CopyDeviceRegion(_gpuMtpConcatBuf, dstByteOffset: 0,
                              _gpuMtpEnormBuf, srcByteOffset: 0, embBytes);
        _gpu.CopyDeviceRegion(_gpuMtpConcatBuf, dstByteOffset: embBytes,
                              _gpuMtpHnormBuf, srcByteOffset: 0, embBytes);

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

        // 10. FFN — MoE (qwen35moe-A3B-MTP via CPU MoE) or dense (qwen35 27B-MTP on GPU).
        if (_mtpIsMoE)
        {
            // CPU MoE for the MTP head: download _gpuNormBuf to CPU, run the same
            // batched-expert path as the trunk CpuMoeFfn but with the MTP block's
            // tensors, upload the result back to _gpuHidden. Shared expert MatMuls
            // run on GPU and overlap with the CPU routed loop. Direct-pinned
            // overloads (issue #48).
            _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, _embDim);
            CpuMoeFfnCore(
                _gpuMtpWGateShexp, _gpuMtpWUpShexp, _gpuMtpWDownShexp,
                _cpuMtpFfnGateInp, _cpuMtpFfnGateInpShexp,
                _cpuMtpFfnGateExps, _cpuMtpFfnUpExps, _cpuMtpFfnDownExps,
                gpuNormIn: _gpuNormBuf, gpuSharedOut: _gpuSharedOut,
                cpuNormIn: _cpuNormBuf, cpuMoeOut: _cpuMoeHidden);
            _gpu.UploadInto(_gpuHidden, (nint)_cpuMoeHidden, _embDim);
        }
        else
        {
            // Dense FFN on GPU. For qwen35 27B-MTP, _intermDim = 17408. Use the
            // dense FFN scratch tensors that were allocated by TryUploadDenseFfnLayers
            // when at least one trunk FFN layer ran on GPU. If those weren't
            // allocated (the no-trunk-FFN-on-GPU case), allocate one-off scratch.
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
        }

        // 11. Residual add.
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);

        // 11b. Capture the MTP block's residual output BEFORE the in-place
        //      shared-head norm (issue #30 chained drafting). Queued D2H into the
        //      pinned _mtpSelfHidden; stream order serializes it ahead of the norm
        //      kernel, and the logits Download below syncs/drains it.
        _gpu.DownloadAsync(_gpuHidden, (nint)_mtpSelfHidden, _embDim);

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
        _gpu.HeadNorm(_gpuQ, _gpuMtpQNorm, _numHeads,   _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNorm(_gpuK, _gpuMtpKNorm, _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);

        // 2c. Partial NEOX RoPE.
        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);

        // 3. Layer-0 invariant: reserve a block on the MTP KV bookkeeping cache
        //    before any append at a new page boundary.
        mtpCache.ReserveBlock();
        // The MTP DRAFT head is a single attention layer, so split-KV (#238) is deliberately
        // not wired here — the occupancy win is negligible for one layer (the main-model verify
        // via GpuAttnBlockAt does get the split). Stays single-block.
        if (_kvDType == DType.BFloat16)
        {
            _gpu.KvAppendBf16(_gpuK, _gpuV, kCache, vCache, kvDim, position, _maxSeqLen);
            _gpu.AttentionBf16(_gpuQ, kCache, vCache, _gpuAttnOut, _gpuAttnScratch,
                _numHeads, _numKvHeads, _headDim, position + 1, _maxSeqLen);
        }
        else
        {
            _gpu.KvAppend(_gpuK, _gpuV, kCache, vCache, kvDim, position, _maxSeqLen);
            _gpu.Attention(_gpuQ, kCache, vCache, _gpuAttnOut, _gpuAttnScratch,
                _numHeads, _numKvHeads, _headDim, position + 1, _maxSeqLen);
        }

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

    /// <inheritdoc />
    /// <remarks>
    /// Issue #33 / #106 (CUDA mirror of <see cref="HybridGdnForwardPass.PrefillMtp"/>):
    /// Walks the prompt and calls <see cref="MtpForward"/> at each position to
    /// populate the GPU MTP KV cache. The previous hidden <c>h_{startPos+i-1}</c> is
    /// read from the absolute-position hidden history buffer populated by every
    /// preceding <see cref="Prefill"/> / <see cref="Forward"/> / <see cref="BatchForward2"/>;
    /// the snapshot branch in <see cref="TruncateTo"/> guarantees slot startPos-1
    /// survives a turn-boundary snapshot restore. MtpForward uploads each prev
    /// hidden back to <c>_gpuLastHidden</c> per step.
    /// </remarks>
    public void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ThrowIfFaulted();
        if (!_hasMtp) return;
        if (tokens is null || tokens.Count == 0) return;

        int N = tokens.Count;
        int requiredHistory = startPos + N;
        if (_mtpHiddenHistoryLength < requiredHistory)
            throw new InvalidOperationException(
                $"PrefillMtp({N} tokens, startPos={startPos}) requires a preceding Prefill / Forward " +
                $"sweep covering positions [0..{requiredHistory}); the hidden history only goes to " +
                $"{_mtpHiddenHistoryLength}.");

        // For position startPos+i, prevHidden = h_{startPos+i-1}:
        //   startPos+i == 0 → zero vector (sequence start)
        //   otherwise       → _mtpPrefillHiddens[(startPos+i-1) * embDim]
        float* zeroHidden = startPos == 0
            ? (float*)NativeMemory.AllocZeroed((nuint)(_embDim * sizeof(float)))
            : null;
        try
        {
            for (int i = 0; i < N; i++)
            {
                int absPos = startPos + i;
                float* prevH = absPos == 0
                    ? zeroHidden!
                    : _mtpPrefillHiddens + (long)(absPos - 1) * _embDim;
                _ = MtpForward(tokens[i], absPos, new ReadOnlySpan<float>(prevH, _embDim));
            }
        }
        finally
        {
            if (zeroHidden != null) NativeMemory.Free(zeroHidden);
        }
    }

    // =================================================================
    //  CPU GDN block — mirror of HybridGdnForwardPass.GdnBlock
    // =================================================================

    private void CpuGdnBlock(int layer, int position) =>
        CpuGdnBlockAt(layer, position, normInGpu: _gpuNormBuf, hiddenOutGpu: _gpuHidden,
                      cpuNormScratch: _cpuNormBuf, cpuHiddenScratch: _cpuHiddenOut);

    /// <summary>
    /// CPU-side GDN block parameterised on the GPU norm-input + GPU hidden-output
    /// tensors plus the CPU scratch pointers to use for the GPU↔CPU staging copies.
    /// Used by both <see cref="Forward"/> and <see cref="BatchForward2"/>.
    /// </summary>
    private void CpuGdnBlockAt(int layer, int position,
                               Tensor normInGpu, Tensor hiddenOutGpu,
                               float* cpuNormScratch, float* cpuHiddenScratch)
    {
        // Download normIn → cpuNormScratch so the CPU GDN kernels can consume it.
        _gpu.Download(normInGpu, new Span<float>(cpuNormScratch, _embDim));

        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        float* scanState = _gdnStateCache.ScanStateAt(gdnIdx);
        float* convState = _gdnStateCache.ConvStateAt(gdnIdx);
        int convStateLen = _gdnStateCache.ConvStateFloatsPerLayer;
        int scanStateLen = _gdnStateCache.ScanStateFloatsPerLayer;

        // 1. Joint QKV projection and z (gate) projection — CPU via SimdKernels.
        SimdKernels.MatVec(_qkv, _cpuWQkv[layer].DataPtr, cpuNormScratch,
            _gdnConvChannels, _embDim, _cpuWQkv[layer].DType);
        SimdKernels.MatVec(_zVec, _cpuWZGate[layer].DataPtr, cpuNormScratch,
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
        SimdKernels.MatVec(_alpha, _cpuSsmAlpha[layer].DataPtr, cpuNormScratch,
            _gdnNumVHeads, _embDim, _cpuSsmAlpha[layer].DType);
        SimdKernels.MatVec(_beta, _cpuSsmBeta[layer].DataPtr, cpuNormScratch,
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
            normEps: 1e-6f,
            layer: layer,
            position: position);

        // 9. Output projection: ssm_out (input ValueDim, output embDim) → cpuHiddenScratch.
        SimdKernels.MatVec(cpuHiddenScratch, _cpuSsmOut[layer].DataPtr, _gdnOut,
            _embDim, _gdnValueDim, _cpuSsmOut[layer].DType);

        // Upload back to GPU.
        _gpu.UploadInto(hiddenOutGpu, new ReadOnlySpan<float>(cpuHiddenScratch, _embDim));
    }

    // =================================================================
    //  GPU GDN block — full-GPU mirror of CpuGdnBlock.
    //  Consumes _gpuNormBuf, writes the block output into _gpuHidden.
    //  No CPU↔GPU sync inside the block.
    // =================================================================

    private void GpuGdnBlock(int layer, int position) =>
        GpuGdnBlockAt(layer, position, normIn: _gpuNormBuf, hiddenOut: _gpuHidden);

    /// <summary>
    /// GPU GDN block parameterised on input-norm / output-hidden tensors. The
    /// recurrent state is the layer-local <see cref="_gpuGdnScanState"/> /
    /// <see cref="_gpuGdnConvState"/> slot; the caller is responsible for
    /// snapshotting it between calls when running the batched verify path.
    /// </summary>
    private void GpuGdnBlockAt(int layer, int position, Tensor normIn, Tensor hiddenOut)
    {
        var scanState = _gpuGdnScanState[layer]!;
        var convState = _gpuGdnConvState[layer]!;

        // 1. Joint QKV projection and z (gate) projection.
        GpuMatMul(_gpuGdnQkv, _gpuWAttnQkv[layer], normIn);
        GpuMatMul(_gpuGdnZVec, _gpuWAttnGate[layer], normIn);

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
        GpuMatMul(_gpuGdnAlpha, _gpuWSsmAlpha[layer], normIn);
        GpuMatMul(_gpuGdnBeta,  _gpuWSsmBeta[layer],  normIn);

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
        GpuMatMul(hiddenOut, _gpuWSsmOut[layer], _gpuGdnOut);
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
        SelectTopK(_routerBuf, _numActiveExperts, selectedExperts, expertWeights, _hp.NormalizeMoeTopKWeights);

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

    private void CpuDenseFfn(int layer) =>
        CpuDenseFfnAt(layer, cpuNorm: _cpuNormBuf, cpuHiddenOut: _cpuMoeHidden);

    /// <summary>
    /// Single-token CPU dense FFN parameterised on the host norm + output scratch.
    /// </summary>
    private void CpuDenseFfnAt(int layer, float* cpuNorm, float* cpuHiddenOut)
    {
        var wGate = _cpuWFfnGate![layer];
        var wUp   = _cpuWFfnUp![layer];
        var wDown = _cpuWFfnDown![layer];

        SimdKernels.MatVecDual(
            _cpuFfnGateBuf, wGate.DataPtr,
            _cpuFfnUpBuf,   wUp.DataPtr,
            cpuNorm, _intermDim, _embDim,
            wGate.DType, wUp.DType);
        SimdKernels.SiLuMul(_cpuFfnGateBuf, _cpuFfnUpBuf, _intermDim);
        SimdKernels.MatVec(cpuHiddenOut, wDown.DataPtr, _cpuFfnGateBuf,
            _embDim, _intermDim, wDown.DType);
    }

    /// <summary>
    /// Batched two-token CPU dense FFN (issue #30). Each FFN weight row is read
    /// once and dotted against both tokens' inputs via
    /// <see cref="SimdKernels.MatVec2In"/>. Realises the bandwidth win on the
    /// CPU mmap FFN path that dominates 27B-MTP decode time on CUDA-hybrid.
    /// </summary>
    private void CpuDenseFfn2(int layer,
        float* cpuNorm1, float* cpuNorm2,
        float* cpuHiddenOut1, float* cpuHiddenOut2)
    {
        var wGate = _cpuWFfnGate![layer];
        var wUp   = _cpuWFfnUp![layer];
        var wDown = _cpuWFfnDown![layer];

        // gate: weight row read once, dotted with both norms.
        SimdKernels.MatVec2In(_cpuFfnGateBuf, _cpuFfnGateBuf2, wGate.DataPtr,
            cpuNorm1, cpuNorm2, _intermDim, _embDim, wGate.DType);
        // up: same pattern.
        SimdKernels.MatVec2In(_cpuFfnUpBuf, _cpuFfnUpBuf2, wUp.DataPtr,
            cpuNorm1, cpuNorm2, _intermDim, _embDim, wUp.DType);

        SimdKernels.SiLuMul(_cpuFfnGateBuf,  _cpuFfnUpBuf,  _intermDim);
        SimdKernels.SiLuMul(_cpuFfnGateBuf2, _cpuFfnUpBuf2, _intermDim);

        // down: silu'd gate buffers are the inputs, output dim = embDim.
        SimdKernels.MatVec2In(cpuHiddenOut1, cpuHiddenOut2, wDown.DataPtr,
            _cpuFfnGateBuf, _cpuFfnGateBuf2, _embDim, _intermDim, wDown.DType);
    }

    /// <summary>
    /// Batched four-token CPU dense FFN (issue #209). Each gate/up/down weight row is
    /// read once from the CPU mmap and dotted against all four tokens via
    /// <see cref="SimdKernels.MatVec4In"/> — one weight HBM read per four draft tokens
    /// versus <see cref="CpuDenseFfn2"/>'s one-per-two, halving the dominant decode
    /// cost on the 27B-MTP CUDA-hybrid path at k = 4. Per-token bits are identical to
    /// <see cref="CpuDenseFfn2"/> and single-token decode (MatVec4In is bit-identical
    /// per slot). Lanes that are duplicated-tail fillers point their <c>out</c> at a
    /// shared sink — the value is recomputed-but-discarded; the four gate/up scratch
    /// slabs stay distinct because SiLU consumes each lane before the down projection.
    /// </summary>
    private void CpuDenseFfn4(int layer,
        float* n0, float* n1, float* n2, float* n3,
        float* out0, float* out1, float* out2, float* out3)
    {
        var wGate = _cpuWFfnGate![layer];
        var wUp   = _cpuWFfnUp![layer];
        var wDown = _cpuWFfnDown![layer];

        SimdKernels.MatVec4In(_cpuFfnGateBuf, _cpuFfnGateBuf2, _cpuFfnGateBuf3, _cpuFfnGateBuf4,
            wGate.DataPtr, n0, n1, n2, n3, _intermDim, _embDim, wGate.DType);
        SimdKernels.MatVec4In(_cpuFfnUpBuf, _cpuFfnUpBuf2, _cpuFfnUpBuf3, _cpuFfnUpBuf4,
            wUp.DataPtr, n0, n1, n2, n3, _intermDim, _embDim, wUp.DType);

        SimdKernels.SiLuMul(_cpuFfnGateBuf,  _cpuFfnUpBuf,  _intermDim);
        SimdKernels.SiLuMul(_cpuFfnGateBuf2, _cpuFfnUpBuf2, _intermDim);
        SimdKernels.SiLuMul(_cpuFfnGateBuf3, _cpuFfnUpBuf3, _intermDim);
        SimdKernels.SiLuMul(_cpuFfnGateBuf4, _cpuFfnUpBuf4, _intermDim);

        SimdKernels.MatVec4In(out0, out1, out2, out3, wDown.DataPtr,
            _cpuFfnGateBuf, _cpuFfnGateBuf2, _cpuFfnGateBuf3, _cpuFfnGateBuf4,
            _embDim, _intermDim, wDown.DType);
    }

    // =================================================================
    //  GPU dense FFN — for layers whose ffn_gate/up/down were uploaded by
    //  TryUploadDenseFfnLayers. Consumes _gpuNormBuf, produces _gpuHidden.
    // =================================================================

    private void GpuDenseFfn(int layer) =>
        GpuDenseFfnAt(layer, normIn: _gpuNormBuf, hiddenOut: _gpuHidden,
                      gateBuf: _gpuFfnGateBufDense!, upBuf: _gpuFfnUpBufDense!);

    /// <summary>
    /// GPU dense FFN parameterised on the input-norm / output-hidden tensors and
    /// the gate/up scratch tensors. <see cref="BatchForward2"/> calls it twice
    /// per layer with its own scratch pair so token 1 and token 2 don't clobber
    /// each other's intermediate gate/up.
    /// </summary>
    private void GpuDenseFfnAt(int layer, Tensor normIn, Tensor hiddenOut,
                               Tensor gateBuf, Tensor upBuf)
    {
        var wGate = _gpuWFfnGate![layer]!;
        var wUp   = _gpuWFfnUp![layer]!;
        var wDown = _gpuWFfnDown![layer]!;

        GpuMatMul(gateBuf, wGate, normIn);
        GpuMatMul(upBuf,   wUp,   normIn);
        _gpu.SiLuMul(gateBuf, upBuf);
        GpuMatMul(hiddenOut, wDown, gateBuf);
    }

    /// <summary>
    /// Issue #43: two-token variant of <see cref="GpuDenseFfnAt"/>. Each of
    /// the three FFN MatMuls (gate, up, down) is dispatched as a single
    /// <c>MatMulN2</c> that reads the weight tensor once and accumulates into
    /// the two token-side outputs in lockstep. SiLuMul runs twice (cheap,
    /// purely element-wise on independent scratches).
    /// </summary>
    private void GpuDenseFfn2At(int layer,
                                Tensor normIn1, Tensor normIn2,
                                Tensor hiddenOut1, Tensor hiddenOut2,
                                Tensor gateBuf1, Tensor gateBuf2,
                                Tensor upBuf1,   Tensor upBuf2)
    {
        var wGate = _gpuWFfnGate![layer]!;
        var wUp   = _gpuWFfnUp![layer]!;
        var wDown = _gpuWFfnDown![layer]!;

        GpuMatMulN2(gateBuf1, gateBuf2, wGate, normIn1, normIn2);
        GpuMatMulN2(upBuf1,   upBuf2,   wUp,   normIn1, normIn2);
        _gpu.SiLuMul(gateBuf1, upBuf1);
        _gpu.SiLuMul(gateBuf2, upBuf2);
        GpuMatMulN2(hiddenOut1, hiddenOut2, wDown, gateBuf1, gateBuf2);
    }

    /// <summary>
    /// Issue #121: dense GPU FFN batched over all <paramref name="N"/> prompt tokens.
    /// gate/up/down each run as a single GEMM-N launch (<see cref="GpuMatMulBatched"/>)
    /// over the post-attn-norm rows <paramref name="normAll"/> ([N × embDim]) — one weight
    /// read per row applied to all N token columns — followed by an elementwise batched
    /// <c>SiLuMul</c>. Bit-identical to N sequential <see cref="GpuDenseFfn"/> calls: the
    /// GEMM-N kernels are proven row-for-row equal to the per-token matvec (#119), and
    /// SiLuMul is per-element. Output written to <paramref name="hiddenAll"/> [N × embDim].
    /// </summary>
    private void BatchedGpuDenseFfn(int layer, int N, Tensor normAll, Tensor hiddenAll)
    {
        var wGate = _gpuWFfnGate![layer]!;
        var wUp   = _gpuWFfnUp![layer]!;
        var wDown = _gpuWFfnDown![layer]!;
        var gateAll = _gpuBfGateAll!;
        var upAll   = _gpuBfUpAll!;

        GpuMatMulBatched(gateAll, wGate, normAll, N);
        GpuMatMulBatched(upAll,   wUp,   normAll, N);
        _gpu.SiLuMul(gateAll, upAll);
        GpuMatMulBatched(hiddenAll, wDown, gateAll, N);
    }

    /// <summary>
    /// Issue #121: GPU-SLRU routed-MoE FFN batched over <paramref name="N"/> prompt tokens
    /// at one layer, grouped by selected expert. Mirrors the CPU
    /// <see cref="BatchedRoutedExperts"/> structure on the GPU: each cached expert is
    /// loaded once (one <c>GetOrLoad</c> instead of N×na), its tokens' norm rows are
    /// gathered into a contiguous block, gate/up/down run as GEMM-N over that block, and
    /// the unweighted down outputs are scattered into a per-(token,slot) partial buffer.
    /// A final single-launch reduce (issue #129) then sums the na partials in top-k slot
    /// order (k=0..na-1) and adds the per-token shared expert — byte-identical to the
    /// sequential <see cref="GpuMoeFfn"/> accumulation. The shared expert itself is also
    /// computed batched (GEMM-N gate/up/down + one per-row scale launch) rather than the
    /// old per-token loop. Output written to <paramref name="hiddenAll"/>.
    ///
    /// <para>Bit-parity rests on: (1) GEMM-N over a gathered (or full-N) contiguous block is
    /// row-for-row equal to per-token <c>GpuMatMul</c> (#119/#121); (2) the gather/scatter
    /// are exact byte copies; (3) the <c>llm_moe_weighted_reduce</c> kernel visits slots in
    /// the same k=0..na-1 order with the same per-token weights and the same per-op rounding
    /// (FMA per term + plain shared add) as the sequential <c>AddScaledInPlace</c>×na +
    /// <c>AddInPlace</c>; (4) the shared expert + its sigmoid scalar gate are computed and
    /// applied (CPU dot → per-row <c>llm_scale_rows_inplace</c>) exactly as in
    /// <c>GpuMoeFfn</c>. SLRU access order does not affect loaded weights, so grouping is
    /// safe.</para>
    /// </summary>
    private void BatchedGpuMoeFfn(int layer, int N, Tensor normAll, Tensor hiddenAll)
    {
        int embDim = _embDim;
        int na = _numActiveExperts;
        int expertDim = _expertDim;
        int numExperts = _numExperts;

        // ── Phase 0: router (batched) + per-token top-k selection. The router matmul is
        //    a GEMM-N over normAll, bit-identical to N per-token GpuMatMul(_gpuRouterLogits)
        //    calls. Softmax is applied per row as N independent Softmax launches over
        //    numExperts-wide views — bit-identical to the per-token Softmax(_gpuRouterLogits).
        //    Download once, pick top-k per token on the host exactly as the sequential
        //    SelectTopK does.
        var routerAll = _gpuBfMoeRouterAll!;
        GpuMatMulBatched(routerAll, _gpuWGateInp[layer], normAll, N);
        for (int i = 0; i < N; i++)
        {
            var rowV = _gpu.View(routerAll, (long)i * numExperts, numExperts);
            try { _gpu.Softmax(rowV); }
            finally { _gpu.Free(rowV); }
        }
        _gpu.Download(routerAll, new Span<float>(_bfRouterAll, (int)((long)N * numExperts)));

        int* selected = _bfSelected!; float* weights = _bfWeights!;
        for (int i = 0; i < N; i++)
        {
            Span<int> sel = new(selected + (long)i * na, na);
            Span<float> wts = new(weights + (long)i * na, na);
            SelectTopK(new ReadOnlySpan<float>(_bfRouterAll + (long)i * numExperts, numExperts),
                       na, sel, wts, _hp.NormalizeMoeTopKWeights);
        }

        // ── Phase 1: shared expert (batched). Issue #129: instead of N per-token
        //    matvec+scale loops, run gate/up/down as GEMM-N over all N tokens — each is
        //    bit-identical to the per-token GpuMatMul by the #119/#121 GEMM-N invariant
        //    (also verified by the batched shared-expert in TrunkLayerBatched). The
        //    per-token sigmoid scalar gate is still computed on the CPU exactly as
        //    GpuMoeFfn (dot of ffn_gate_inp_shexp · norm_i over the host readback) to stay
        //    bit-identical, then applied with one llm_scale_rows_inplace launch — a single
        //    float multiply per element, identical to the per-token ScaleInPlace. This
        //    rounds the shared output to float BEFORE the Phase-3 plain add, exactly as the
        //    sequential ScaleInPlace-then-AddInPlace ordering requires.
        _gpu.Download(normAll, new Span<float>(_bfNormReadback, (int)((long)N * embDim)));
        _gpu.Download(_gpuWGateInpShexp[layer], new Span<float>(_hostQ, embDim)); // gate-inp weight (shared across tokens)
        float* shexpScale = _bfShexpScale!;
        for (int i = 0; i < N; i++)
        {
            float dot = SimdKernels.DotF32(_hostQ, _bfNormReadback + (long)i * embDim, embDim);
            shexpScale[i] = 1.0f / (1.0f + MathF.Exp(-dot));
        }
        // GEMM-N gate/up over all N norm rows → batched SiLuMul → GEMM-N down into hiddenAll
        // (UNSCALED shared output), then apply the per-row sigmoid gate in one launch.
        GpuMatMulBatched(_gpuBfShGateAll!, _gpuWGateShexp[layer], normAll, N);
        GpuMatMulBatched(_gpuBfShUpAll!,   _gpuWUpShexp[layer],   normAll, N);
        _gpu.SiLuMul(_gpuBfShGateAll!, _gpuBfShUpAll!);   // pointwise over N×expertDim
        GpuMatMulBatched(hiddenAll, _gpuWDownShexp[layer], _gpuBfShGateAll!, N);
        _gpu.UploadInto(_gpuBfShexpScaleDev!, new ReadOnlySpan<float>(shexpScale, N));
        _gpu.ScaleRowsInPlace(hiddenAll, _gpuBfShexpScaleDev!, N, embDim);

        // ── Phase 2: bucket (token, slot) by selected expert (CSR), identical to
        //    BatchedRoutedExperts. Then for each used expert, gather its tokens' norm
        //    rows contiguously, GEMM-N gate/up/down, and scatter the UNWEIGHTED down
        //    output into the per-(token,slot) partial buffer.
        int* expStart = _bfExpStart!; int* cursor = _bfExpCursor!; int* used = _bfUsedExperts!;
        for (int e = 0; e <= numExperts; e++) expStart[e] = 0;
        long totalSel = (long)N * na;
        for (long s = 0; s < totalSel; s++) expStart[selected[s] + 1]++;
        for (int e = 0; e < numExperts; e++) expStart[e + 1] += expStart[e];
        for (int e = 0; e < numExperts; e++) cursor[e] = expStart[e];
        int* expTokI = _bfExpTokI!; int* expTokK = _bfExpTokK!;
        for (int i = 0; i < N; i++)
            for (int k = 0; k < na; k++)
            {
                int e = selected[(long)i * na + k];
                int p = cursor[e]++;
                expTokI[p] = i; expTokK[p] = k;
            }
        int numUsed = 0;
        for (int e = 0; e < numExperts; e++)
            if (expStart[e + 1] > expStart[e]) used[numUsed++] = e;

        var downPartial = _gpuBfMoeDownPartial!;   // [N × na × embDim]
        var normGath = _gpuBfMoeNormGathN!;        // [≤N × embDim]
        var gateGath = _gpuBfMoeGateGathN!;        // [≤N × expertDim]
        var upGath   = _gpuBfMoeUpGathN!;          // [≤N × expertDim]
        var downGath = _gpuBfMoeDownGathN!;        // [≤N × embDim]
        int* gathTokI = _bfGathTokI!;

        for (int u = 0; u < numUsed; u++)
        {
            int e = used[u];
            int pStart = expStart[e], pEnd = expStart[e + 1];
            int cnt = pEnd - pStart;
            if (cnt == 0) continue;

            var slot = _expertSlotManager!.GetOrLoad(layer, e);

            // Gather this expert's token norm rows into a contiguous [cnt × embDim] block.
            for (int g = 0; g < cnt; g++)
            {
                int i = expTokI[pStart + g];
                gathTokI[g] = i;
                _gpu.CopyDeviceRegion(normGath, (long)g * embDim * sizeof(float),
                                      normAll, (long)i * embDim * sizeof(float),
                                      (long)embDim * sizeof(float));
            }

            // GEMM-N gate/up over the gathered block (bit-identical to per-token matvec),
            // batched SiLuMul, GEMM-N down. View the gather buffers to exactly cnt rows so
            // the GEMM-N row/col derivation matches.
            var normV = _gpu.View(normGath, 0, (long)cnt * embDim);
            var gateV = _gpu.View(gateGath, 0, (long)cnt * expertDim);
            var upV   = _gpu.View(upGath,   0, (long)cnt * expertDim);
            var downV = _gpu.View(downGath, 0, (long)cnt * embDim);
            try
            {
                GpuMatMulBatched(gateV, slot.Gate, normV, cnt);
                GpuMatMulBatched(upV,   slot.Up,   normV, cnt);
                _gpu.SiLuMul(gateV, upV);
                GpuMatMulBatched(downV, slot.Down, gateV, cnt);
            }
            finally
            {
                _gpu.Free(normV); _gpu.Free(gateV); _gpu.Free(upV); _gpu.Free(downV);
            }

            // Scatter the UNWEIGHTED down rows into per-(token,slot) partials. The reduce
            // (Phase 3) applies the top-k weights in slot order, so we keep them unscaled
            // here — exactly like the CPU Phase-C partials.
            for (int g = 0; g < cnt; g++)
            {
                int i = expTokI[pStart + g];
                int k = expTokK[pStart + g];
                long slotIdx = (long)i * na + k;
                _gpu.CopyDeviceRegion(downPartial, slotIdx * embDim * sizeof(float),
                                      downGath, (long)g * embDim * sizeof(float),
                                      (long)embDim * sizeof(float));
            }
        }

        // ── Phase 3: single-launch ordered reduce (issue #129 — RESOLVED). The previous
        //    host loop issued ~N·(na+2) tiny stream ops (Clear + na AddScaledInPlace +
        //    AddInPlace per token), whose launch overhead undercut the grouped-GEMM win
        //    on large-N GPU-SLRU prefill. We now upload the host top-k weights once and do
        //    the whole weighted scatter-reduce + shared add in ONE llm_moe_weighted_reduce
        //    launch over all N·embDim elements. Per (token i, element e) the kernel computes
        //    acc = Σ_{k=0..na-1} downPartial[(i*na+k)*embDim+e] * weights[i*na+k], then
        //    acc += hiddenAll[i*embDim+e] (the scaled+rounded shared output), writing back
        //    in place. The per-k FMA contraction (NVRTC fmad=true) reproduces the
        //    AddScaledInPlace fmaf rounding exactly; the final plain add reproduces
        //    AddInPlace; routed-first / shared-last order is preserved — byte-identical to
        //    the sequential Clear + AddScaledInPlace×na + AddInPlace. Gates only the
        //    on-GPU-experts (SHARPI_CPU_MOE=0) path.
        _gpu.UploadInto(_gpuBfMoeWeightsDev!, new ReadOnlySpan<float>(weights, (int)((long)N * na)));
        _gpu.MoeWeightedReduce(downPartial, _gpuBfMoeWeightsDev!, hiddenAll, N, na, embDim);
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

    private void CpuMoeFfn(int layer) =>
        CpuMoeFfnCore(
            gpuGateShexp:  _gpuWGateShexp[layer],
            gpuUpShexp:    _gpuWUpShexp[layer],
            gpuDownShexp:  _gpuWDownShexp[layer],
            cpuRouter:     _cpuFfnGateInp![layer],
            cpuGateInpShexp: _cpuFfnGateInpShexp![layer],
            cpuGateExps:   _cpuFfnGateExps![layer],
            cpuUpExps:     _cpuFfnUpExps![layer],
            cpuDownExps:   _cpuFfnDownExps![layer],
            gpuNormIn:     _gpuNormBuf,
            gpuSharedOut:  _gpuSharedOut,
            cpuNormIn:     _cpuNormBuf,
            cpuMoeOut:     _cpuMoeHidden);

    private void CpuMoeFfnCore(
        Tensor gpuGateShexp, Tensor gpuUpShexp, Tensor gpuDownShexp,
        CpuWeightRef cpuRouter, float* cpuGateInpShexp,
        CpuWeightRef cpuGateExps, CpuWeightRef cpuUpExps, CpuWeightRef cpuDownExps,
        Tensor gpuNormIn, Tensor gpuSharedOut,
        float* cpuNormIn, float* cpuMoeOut)
    {
        // Issue #45: gpuNormIn / cpuNormIn / cpuMoeOut / gpuSharedOut let
        // BatchForward2 dispatch CpuMoeFfn for token 2 (with _gpuNormBuf2 /
        // _cpuNormBuf2 / _cpuMoeHidden2). gpuSharedOut is reused for both
        // tokens because the GPU shared-expert kick is synchronously awaited
        // by the Download at the end of CpuMoeFfnCore.
        int numExperts = _numExperts;
        int numActive = _numActiveExperts;
        int expertDim = _expertDim;

        // 1. Kick off the GPU shared expert (async; overlaps with CPU work below).
        //    gpuNormIn is already populated by the RmsNorm before this call;
        //    the launches return immediately and execute while the CPU runs router
        //    and routed experts. Sigmoid scalar gate is computed on CPU and applied
        //    via ScaleInPlace before the host blocks on Download.
        GpuMatMul(_gpuFfnGate, gpuGateShexp, gpuNormIn);
        GpuMatMul(_gpuFfnUp, gpuUpShexp, gpuNormIn);
        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        GpuMatMul(gpuSharedOut, gpuDownShexp, _gpuFfnGate);

        float shexpDot = SimdKernels.DotF32(cpuGateInpShexp, cpuNormIn, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        _gpu.ScaleInPlace(gpuSharedOut, shexpScale);

        // 2. Router: ffn_gate_inp.weight is F32 [embDim, numExperts]; softmax then top-K.
        long sdRouter = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        var routerW = cpuRouter;
        SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, cpuNormIn,
            numExperts, _embDim, routerW.DType);
        SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopKPtr(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);
        if (_decodeProfile) _pdRouterTicks += System.Diagnostics.Stopwatch.GetTimestamp() - sdRouter;

        // 3. Routed experts (sparse top-K). Two batched Parallel.For sweeps
        //    instead of 16 per-expert ones — gate+up across all 8 experts in
        //    one sweep, then down+weighted-accumulate across all 8 experts in
        //    another. Each worker thread does much more work per dispatch,
        //    amortising TPL barrier overhead.
        var gateExps = cpuGateExps;
        var upExps = cpuUpExps;
        var downExps = cpuDownExps;

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
        float* normBuf  = cpuNormIn;
        float* moeOut   = cpuMoeOut;
        int    embDimL  = _embDim;
        int    expertDimL = expertDim;
        int    numActiveL = numActive;
        int    bprGL = bprG, bprUL = bprU, bprDL = bprD;

        // SHARPI_Q3K_Q8K=1 / SHARPI_Q8_0_Q8K=1: hoist a single Q8_K prepack of
        // the Phase-A input (cpuNormIn → _cpuNormInQ8K) so all numActive*expertDim
        // Q3_K / Q8_0 rows can dot against the int-domain DotQ3K_Q8K / DotQ8_0_Q8K.
        // BatchForward2 safety: each CpuMoeFfnCore call writes its own
        // _cpuNormInQ8K from its own cpuNormIn, so t1 and t2 do not collide.
        bool useQ8KGate = (_q3kQ8KEnabled  && gateDt == DType.Q3_K)
                       || (_q8_0Q8KEnabled && gateDt == DType.Q8_0)
                       || (_q4kQ8KEnabled  && gateDt == DType.Q4_K);
        bool useQ8KUp   = (_q3kQ8KEnabled  && upDt   == DType.Q3_K)
                       || (_q8_0Q8KEnabled && upDt   == DType.Q8_0)
                       || (_q4kQ8KEnabled  && upDt   == DType.Q4_K);
        byte* normInQ8K = _cpuNormInQ8K;
        long sdPhaseA = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        if (useQ8KGate || useQ8KUp)
            SimdKernels.QuantizeRowToQ8KS(cpuNormIn, _embDim, normInQ8K);

        // Phase A: gate + up rows for all (k, r) tuples.
        Parallel.For(0, numActiveL * expertDimL, s_moeParallelOpts, idx =>
        {
            int k = idx / expertDimL;
            int r = idx % expertDimL;
            int expertIdx = sePtr[k];
            long offG = (long)expertIdx * expertDimL * bprGL + (long)r * bprGL;
            long offU = (long)expertIdx * expertDimL * bprUL + (long)r * bprUL;
            gateAll[idx] = useQ8KGate
                ? DispatchDotQ8K(gateP + offG, normInQ8K, embDimL, gateDt)
                : DispatchDot(gateP + offG, normBuf, embDimL, gateDt);
            upAll[idx]   = useQ8KUp
                ? DispatchDotQ8K(upP + offU, normInQ8K, embDimL, upDt)
                : DispatchDot(upP   + offU, normBuf, embDimL, upDt);
        });
        if (_decodeProfile) _pdPhaseATicks += System.Diagnostics.Stopwatch.GetTimestamp() - sdPhaseA;
        long sdPhaseC = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // Phase B: one fused SiLuMul over (numActive × expertDim) contiguous
        // floats. SiLuMul is element-wise, so expert boundaries don't matter —
        // one AVX-vectorised call beats 8 with their own setup cost.
        SimdKernels.SiLuMul(_cpuExpertGateAll, _cpuExpertUpAll, numActive * expertDim);

        // SHARPI_Q3K_Q8K=1 / SHARPI_Q8_0_Q8K=1 Phase C prepack: each routed
        // expert k has its own post-SiLuMul gate slice (gateAll + k*expertDim),
        // so we quantise numActive slices into a stacked Q8_K buffer once before
        // the embDim-row Parallel.For, and the inner loop indexes by k * stride.
        bool useQ8KDown = (_q3kQ8KEnabled  && downDt == DType.Q3_K)
                       || (_q8_0Q8KEnabled && downDt == DType.Q8_0)
                       || (_q4kQ8KEnabled  && downDt == DType.Q4_K);
        byte* gateAllQ8K = _cpuExpertGateAllQ8K;
        int   gateAllQ8KStride = _cpuExpertGateAllQ8KStride;
        if (useQ8KDown)
        {
            for (int k = 0; k < numActiveL; k++)
                SimdKernels.QuantizeRowToQ8KS(
                    gateAll + (long)k * expertDimL,
                    expertDimL,
                    gateAllQ8K + (long)k * gateAllQ8KStride);
        }

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
                sum += w * (useQ8KDown
                    ? DispatchDotQ8K(downP + offD,
                                     gateAllQ8K + (long)k * gateAllQ8KStride,
                                     expertDimL, downDt)
                    : DispatchDot(downP + offD,
                                  gateAll + (long)k * expertDimL,
                                  expertDimL, downDt));
            }
            moeOut[r] = sum;
        });
        if (_decodeProfile) _pdPhaseCTicks += System.Diagnostics.Stopwatch.GetTimestamp() - sdPhaseC;

        // 4. Wait for GPU shared expert, download, and combine into routed accumulator
        //    (Download self-syncs the stream).
        long sdShared = _decodeProfile ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        _gpu.Download(gpuSharedOut, new Span<float>(_cpuSharedOut, _embDim));
        SimdKernels.AddInPlace(cpuMoeOut, _cpuSharedOut, _embDim);
        if (_decodeProfile) _pdSharedTicks += System.Diagnostics.Stopwatch.GetTimestamp() - sdShared;
    }

    // ParallelOptions for the routed-MoE Parallel.For sweeps. Defaults to the
    // logical processor count, but SHARPI_MOE_THREADS overrides it: the int8
    // dot sweeps are heavy on the per-core SIMD pipeline (especially the
    // AVX-512 VNNI path, where two SMT siblings share one 512-bit unit), so
    // pinning to the physical core count often beats oversubscribing all
    // logical processors. It also caps oversubscription against the
    // concurrently running GPU shared-expert host launches.
    private static readonly ParallelOptions s_moeParallelOpts = new()
    {
        MaxDegreeOfParallelism = ResolveMoeThreads()
    };

    private static int ResolveMoeThreads()
    {
        var v = Environment.GetEnvironmentVariable("SHARPI_MOE_THREADS");
        if (int.TryParse(v, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int n) && n > 0)
            return n;
        return Environment.ProcessorCount;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DispatchDot(byte* row, float* input, int cols, DType dtype) =>
        dtype switch
        {
            DType.Q3_K    => SimdKernels.DotQ3K(row, input, cols),
            DType.Q4_K    => SimdKernels.DotQ4K(row, input, cols),
            DType.Q5_K    => SimdKernels.DotQ5K(row, input, cols),
            DType.Q6_K    => SimdKernels.DotQ6K(row, input, cols),
            DType.Q8_0    => SimdKernels.DotQ8_0(row, input, cols),
            DType.Float32 => SimdKernels.DotF32((float*)row, input, cols),
            _ => throw new NotSupportedException($"Routed expert dtype {dtype} not supported in batched path"),
        };

    // Issue #112: dequant-once two-input dispatch. Decodes the quantized weight row
    // ONCE and dots it against two token inputs, amortizing the (Q4_K/Q5_K nibble)
    // unpack across the pair. Bit-identical to two <see cref="DispatchDot"/> calls —
    // the 2In kernels mirror the single-input accumulator structure exactly (proven
    // by the MTP batched-verify path) — so the routed-MoE byte-parity oracle still
    // holds. Dtypes without a 2In kernel fall back to two single dots (no win, still
    // correct).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DispatchDot2In(byte* row, float* in1, float* in2, int cols, DType dtype,
        out float v1, out float v2)
    {
        switch (dtype)
        {
            case DType.Q4_K: SimdKernels.DotQ4K_2In(row, in1, in2, cols, out v1, out v2); break;
            case DType.Q5_K: SimdKernels.DotQ5K_2In(row, in1, in2, cols, out v1, out v2); break;
            default:
                v1 = DispatchDot(row, in1, cols, dtype);
                v2 = DispatchDot(row, in2, cols, dtype);
                break;
        }
    }

    // Issue #112: dequant-once two-input dispatch for the Q8_KS-prepacked path.
    // Decodes the Q3_K weight row once and dots against two prepacked token inputs.
    // Bit-identical to two <see cref="DispatchDotQ8K"/> calls. Q8_0 has no expensive
    // unpack, so it falls back to two single dots.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DispatchDotQ8K2In(byte* row, byte* scr1, byte* scr2, int cols, DType dtype,
        out float v1, out float v2)
    {
        switch (dtype)
        {
            case DType.Q3_K: SimdKernels.DotQ3K_Q8KS_2In(row, scr1, scr2, cols, out v1, out v2); break;
            case DType.Q4_K: SimdKernels.DotQ4K_Q8KS_2In(row, scr1, scr2, cols, out v1, out v2); break;
            default:
                v1 = DispatchDotQ8K(row, scr1, cols, dtype);
                v2 = DispatchDotQ8K(row, scr2, cols, dtype);
                break;
        }
    }

    // Issue #114: dequant-once FOUR-input dispatch — register-tiled extension of
    // DispatchDot2In. Decodes the quantized weight row ONCE and dots it against four
    // token inputs (decode/4 vs the pairing's decode/2). Bit-identical to four
    // DispatchDot calls (the 4In kernels mirror the single-input accumulator order;
    // proven by the SimdKernelsQ8KSTests *_4In_BitwiseMatchesSingle oracles). Dtypes
    // without a 4In kernel fall back to two 2In pairs — never worse than the prior
    // pairing path, still correct.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void DispatchDot4In(byte* row, float* in0, float* in1, float* in2, float* in3,
        int cols, DType dtype, out float v0, out float v1, out float v2, out float v3)
    {
        switch (dtype)
        {
            case DType.Q4_K:
                SimdKernels.DotQ4K_4In(row, in0, in1, in2, in3, cols, out v0, out v1, out v2, out v3);
                break;
            default:
                DispatchDot2In(row, in0, in1, cols, dtype, out v0, out v1);
                DispatchDot2In(row, in2, in3, cols, dtype, out v2, out v3);
                break;
        }
    }

    // Issue #114: dequant-once four-input dispatch for the Q8_KS-prepacked path.
    // Decodes the Q3_K weight row once and dots against four prepacked token inputs.
    // Bit-identical to four DispatchDotQ8K calls. Other dtypes fall back to two 2In
    // pairs (Q8_0 has no expensive unpack; this keeps it correct without regressing).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void DispatchDotQ8K4In(byte* row, byte* s0, byte* s1, byte* s2, byte* s3,
        int cols, DType dtype, out float v0, out float v1, out float v2, out float v3)
    {
        switch (dtype)
        {
            case DType.Q3_K:
                SimdKernels.DotQ3K_Q8KS_4In(row, s0, s1, s2, s3, cols, out v0, out v1, out v2, out v3);
                break;
            case DType.Q4_K:
                SimdKernels.DotQ4K_Q8KS_4In(row, s0, s1, s2, s3, cols, out v0, out v1, out v2, out v3);
                break;
            default:
                DispatchDotQ8K2In(row, s0, s1, cols, dtype, out v0, out v1);
                DispatchDotQ8K2In(row, s2, s3, cols, dtype, out v2, out v3);
                break;
        }
    }

    // Same idea as DispatchDot but the input is already prepacked to Q8_KS
    // (per-32-element scales — issue #107) once per CpuMoeFfnCore call (Phase A:
    // cpuNormIn; Phase C: each gateAll slice), so individual rows hit the
    // int-domain dot kernels. Only Q3_K, Q8_0, and Q4_K are wired today — the
    // caller guards entry via the corresponding useQ8K* flag, so other dtypes
    // throw if they ever reach here.
    private static float DispatchDotQ8K(byte* row, byte* q8kScratch, int cols, DType dtype) =>
        dtype switch
        {
            DType.Q3_K => SimdKernels.DotQ3K_Q8KS(row, q8kScratch, cols),
            DType.Q8_0 => SimdKernels.DotQ8_0_Q8KS(row, q8kScratch, cols),
            DType.Q4_K => SimdKernels.DotQ4K_Q8KS(row, q8kScratch, cols),
            _ => throw new NotSupportedException($"Q8_KS-prepacked dispatch not implemented for dtype {dtype}"),
        };

    // True if any routed-expert weight tensor (trunk layers + MTP head if present)
    // is encoded in `target`. Used to auto-enable the matching Q8_K-input kernel
    // gate at model load — see _q3kQ8KEnabled / _q8_0Q8KEnabled. Scans the GGUF
    // tensor index without allocating, so it is cheap to call from the constructor.
    private static bool HasRoutedExpertsOfDType(GgufModel model, ModelHyperparams hp, DType target)
    {
        if (!hp.IsMoE) return false;
        int L = hp.NumLayers;
        for (int i = 0; i <= L; i++) // <= L so the MTP-head layer (index L) is included if present
        {
            if (model.FindTensor($"blk.{i}.ffn_gate_exps.weight")?.DType == target) return true;
            if (model.FindTensor($"blk.{i}.ffn_up_exps.weight")?.DType   == target) return true;
            if (model.FindTensor($"blk.{i}.ffn_down_exps.weight")?.DType == target) return true;
        }
        return false;
    }

    // Three-state env-var resolver: "1" forces on, "0" forces off, anything else
    // (including unset) falls through to the auto-detected default.
    private static bool ResolveGate(string envName, bool autoDetect)
    {
        var v = Environment.GetEnvironmentVariable(envName);
        if (v == "1") return true;
        if (v == "0") return false;
        // Also accept the natural string forms (true/false/True/False) so a direct-env user who
        // types SHARPI_..=true isn't silently defaulted — the CLI/server plumbing writes "1"/"0",
        // but humans set these by hand. Anything else still falls through to the auto-default.
        if (bool.TryParse(v, out bool b)) return b;
        return autoDetect;
    }

    /// <summary>Parse a non-negative integer env override, falling back to <paramref name="dflt"/>
    /// for an unset/blank/invalid value (used for the op-offload token gate).</summary>
    private static int ResolveIntGate(string envName, int dflt)
    {
        var v = Environment.GetEnvironmentVariable(envName);
        return int.TryParse(v, out int n) && n >= 0 ? n : dflt;
    }

    /// <summary>
    /// Convert raw host byte ranges <c>(ptr, bytes)</c> into page-aligned, sorted, merged
    /// <c>[start, end)</c> ranges suitable for <c>cudaHostRegister</c> — each input range is
    /// rounded out to whole <paramref name="pageSize"/> pages (floor start, ceil end), then
    /// overlapping or exactly-adjacent ranges are coalesced so a page shared by two
    /// 32-byte-aligned GGUF tensors is never registered twice. Ranges with non-positive
    /// <c>bytes</c> are skipped. Internal + static so it can be covered by a CPU-only unit test.
    /// </summary>
    internal static List<(long start, long end)> MergePageAlignedRanges(
        List<(long ptr, long bytes)> ranges, long pageSize)
    {
        // Work in UNSIGNED: a host pointer with the high bit set is negative as a signed long,
        // which would floor the wrong way under signed division and misorder under signed compare
        // (Gemini). The returned tuples carry the unsigned address bit-pattern in `long` (callers
        // cast to nint / take end−start, both bit-pattern-correct), so callers are unaffected.
        ulong upage = (ulong)pageSize;
        var aligned = new List<(ulong start, ulong end)>(ranges.Count);
        foreach (var (ptr, bytes) in ranges)
        {
            if (bytes <= 0) continue;
            ulong uptr = (ulong)ptr;
            ulong s = uptr / upage * upage;
            ulong e = (uptr + (ulong)bytes + upage - 1) / upage * upage;
            aligned.Add((s, e));
        }
        aligned.Sort((a, b) => a.start.CompareTo(b.start));   // unsigned compare
        var merged = new List<(ulong start, ulong end)>(aligned.Count);
        foreach (var r in aligned)
        {
            if (merged.Count > 0 && r.start <= merged[^1].end)
            {
                if (r.end > merged[^1].end) merged[^1] = (merged[^1].start, r.end);
            }
            else merged.Add(r);
        }
        var result = new List<(long start, long end)>(merged.Count);
        foreach (var (s, e) in merged) result.Add(((long)s, (long)e));
        return result;
    }

    private static void SelectTopKPtr(float* logits, int n, int k,
        Span<int> indices, Span<float> weights, bool normalize)
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
        if (normalize && k > 1)
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

    /// Issue #111: batched (GEMM-N) MatMul dispatch — one weight read applied to N
    /// token columns in a single launch, bit-identical to N sequential GpuMatMul calls.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GpuMatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll, int nTok)
    {
        var dt = _gpuWeightDTypes.TryGetValue(matrix.Handle, out var d) ? d : DType.Float32;
        // #162: in a batched prompt (nTok > 1), read each weight once per batch instead of
        // the per-token GEMM-N matvec re-stream. Q8_0/Q4_K → int8 MMQ (no fp16 temp);
        // Q6_K/Q5_K (no MMQ kernel) → dequant→fp16→cuBLAS GEMM. Argmax-stable, gated so the
        // bit-parity oracle keeps the matvec path. Q4_K/Q6_K/Q5_K need 256-aligned cols;
        // Q8_0 needs 32 — true for every projection dim, but guarded so we fall back
        // (never throw) on an odd shape.
        // The MMQ/dequant-GEMM compute path only amortizes its fixed per-call costs
        // (whole-weight dequant to an fp16 temp for Q6_K/Q5_K — 71 MB per 27B FFN
        // layer, ~600 MB for the lm_head; MMQ's activation re-quant) at prefill-scale
        // N. At decode-sized N — the #30 batched verify (k ≤ 8 by SHARPI_MTP_BATCH_MAX)
        // or a tiny prefill tail chunk — those temps land in WDDM-paged VRAM behind
        // the post-fill 64 MiB margin and 5-10× every step (measured on the verify:
        // 6.0 → 9.2 t/s from this threshold alone). Small N takes the temp-free
        // matvec re-stream below — the same decode kernels sequential Forward uses,
        // and the bit-exact reference path the compute kernels are validated against.
        if (GdnPrefillComputeEnabled && nTok > MatMulComputeBatchMinN)
        {
            int cols = (int)(inputAll.ElementCount / nTok);
            switch (dt)
            {
                case DType.Q8_0 when (cols & 31) == 0:
                    _gpu.MatMulBatchedMmq(outputAll, matrix, inputAll, nTok, dt); return;
                case DType.Q4_K when (cols & 0xff) == 0:
                    _gpu.MatMulBatchedMmq(outputAll, matrix, inputAll, nTok, dt); return;
                case DType.Q6_K when (cols & 0xff) == 0:
                case DType.Q5_K when (cols & 0xff) == 0:
                    _gpu.MatMulBatchedGemm(outputAll, matrix, inputAll, nTok, dt); return;
                case DType.Q3_K when Q3kDequantGemmEnabled && (cols & 0xff) == 0:
                    _gpu.MatMulBatchedGemm(outputAll, matrix, inputAll, nTok, dt); return;
            }
        }
        _gpu.MatMulBatched(outputAll, matrix, inputAll, nTok, dt);
    }

    // Crossover below which the MMQ/dequant-GEMM compute kernels' fixed per-call
    // costs exceed the matvec re-stream's k× weight reads. 8 = the verify-batch
    // ceiling; prefill chunks run at hundreds, so the regimes are well separated.
    private const int MatMulComputeBatchMinN = 8;

    // #388: route prefill-scale Q3_K (routed MoE experts) through the dequant→fp16→cuBLAS GEMM
    // (weight read once) instead of the per-token-re-reading GEMM-N. Argmax-stable (fp16-rounded
    // weight, same class as the Q5_K/Q6_K dequant-GEMM). SHARPI_Q3K_DEQUANT_GEMM=0 → GEMM-N.
    private static readonly bool Q3kDequantGemmEnabled =
        Environment.GetEnvironmentVariable("SHARPI_Q3K_DEQUANT_GEMM") != "0";

    /// Issue #121: true when <paramref name="matrix"/>'s dtype is one of the dtypes
    /// <see cref="CudaBackend.MatMulBatched"/> implements a GEMM-N kernel for. Gates the
    /// batched FFN/MoE path so an unsupported dtype falls back to the per-token loop
    /// instead of throwing NotSupportedException mid-prefill.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private bool BatchedMatMulSupported(Tensor matrix)
    {
        var dt = _gpuWeightDTypes.TryGetValue(matrix.Handle, out var d) ? d : DType.Float32;
        return dt is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Float32;
    }

    /// Issue #43: two-input MatMul dispatch — used by GpuDenseFfn2At so the
    /// FFN gate / up / down weights get read from VRAM once per (row, lane)
    /// for both draft tokens instead of twice.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void GpuMatMulN2(Tensor outputA, Tensor outputB,
                             Tensor matrix,
                             Tensor inputA, Tensor inputB)
    {
        _gpu.MatMulN2(outputA, outputB, matrix, inputA, inputB,
            _gpuWeightDTypes.TryGetValue(matrix.Handle, out var dt) ? dt : DType.Float32);
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
                    if (indices[j] == i) { alreadySelected = true; break; }
                if (!alreadySelected && logits[i] > bestVal)
                { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }
        if (normalize && k > 1)
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

    /// <summary>Collect every CPU-resident mmap weight region for the issue #221
    /// pre-fault sweep: the CPU-MoE routed experts (or dense FFN weights), the
    /// SHARPI_CPU_GDN debug GDN weights, and the MoE-MTP head experts. Arrays that
    /// aren't allocated for this config are null; unpopulated slots (e.g. the GDN
    /// arrays when not in CPU-GDN mode) have a null <c>DataPtr</c> and are skipped.
    /// Everything dequantized via LoadF32Tensor/LoadConv1d lives in separate buffers,
    /// not the mmap, and is excluded.</summary>
    private List<(nint Ptr, long Bytes)> BuildCpuPrefaultRegions()
    {
        var regions = new List<(nint, long)>();
        void Add1(CpuWeightRef w)
        {
            if (w.DataPtr != null) regions.Add(((nint)w.DataPtr, w.Info.ByteSize));
        }
        void Add(CpuWeightRef[]? arr)
        {
            if (arr is null) return;
            foreach (var w in arr) Add1(w);
        }

        // Trunk: CPU-MoE routed experts, or dense FFN weights (Qwen3.6-27B-MTP).
        Add(_cpuFfnGateInp); Add(_cpuFfnGateExps); Add(_cpuFfnUpExps); Add(_cpuFfnDownExps);
        Add(_cpuWFfnGate); Add(_cpuWFfnUp); Add(_cpuWFfnDown);

        // SHARPI_CPU_GDN=1 debug path (arrays always allocated, populated only then).
        Add(_cpuWQkv); Add(_cpuWZGate); Add(_cpuSsmOut); Add(_cpuSsmAlpha); Add(_cpuSsmBeta);

        // MoE-MTP head routed experts (one extra layer; null DataPtr when absent).
        Add1(_cpuMtpFfnGateInp); Add1(_cpuMtpFfnGateExps);
        Add1(_cpuMtpFfnUpExps); Add1(_cpuMtpFfnDownExps);

        return regions;
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
        else if (info.DType == DType.Q4_K || info.DType == DType.Q5_K || info.DType == DType.Q6_K
                 || (info.DType == DType.Q8_0 && RawQ80WeightsEnabled))
        {
            // CUDA matvec/MMQ dispatch on Q4_K / Q5_K / Q6_K / Q8_0 via dedicated kernels
            // (GpuMatMul → MatMul; GpuMatMulBatched → MatMulBatchedMmq). Keeping the weight
            // raw avoids the F32 dequant's 4× VRAM and the memory-bound F32 GEMM-N path.
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Q3_K, etc. (and Q8_0 under SHARPI_GDN_RAW_Q8_0=0) — no raw GPU matvec used
            // here, or the reverted reference path. Dequantize to F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private Tensor UploadEmbeddingWeight(string name, out DType embDType)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // Raw-bytes upload path for quants we can decode on-device. Q4_K and Q5_K
        // both have NVRTC embedding-lookup kernels (issues #25 fix, #39); other
        // dtypes fall through to F32 expansion (capped by ShouldKeepFixedWeightsOnCpu).
        if (info.DType == DType.Q4_K || info.DType == DType.Q5_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            // exact=true: embedding table is permanent for the session; skip the
            // power-of-2 round-up that would otherwise inflate a 715 MiB Q4_K embed
            // to a 1024 MiB GPU allocation.
            var result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount), exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
            embDType = info.DType;
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
        embDType = DType.Float32;
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

    // SHARPI_KV_DTYPE — issue #27. Default Bf16 on this forward pass; fp32 is
    // the bisect-only escape hatch. Anything else is rejected so a typo in the
    // env var doesn't silently fall back to the default.
    private static DType ParseKvDType(string? envValue) => envValue?.Trim().ToLowerInvariant() switch
    {
        null or ""    => DType.BFloat16,
        "bf16"        => DType.BFloat16,
        "fp32"        => DType.Float32,
        var other     => throw new ArgumentException(
            $"SHARPI_KV_DTYPE must be 'bf16' or 'fp32' (got '{other}').", nameof(envValue)),
    };

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
        // Quants with a direct-read embedding kernel stay raw on the GPU.
        if (tensor.DType == DType.Q4_K || tensor.DType == DType.Q5_K)
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
        int kvBytes = DTypeInfo.BytesPerElement(_kvDType);
        long attnPerLayer = (long)_embDim * _numHeads * _headDim * 2 * sizeof(float)  // q (output qDim*2)
                          + (long)_embDim * _numKvHeads * _headDim * sizeof(float) * 2 // k + v
                          + (long)_embDim * _numHeads * _headDim * sizeof(float)      // o
                          + (long)_maxSeqLen * _numKvHeads * _headDim * kvBytes * 2;  // kv cache
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

        int kvBytes = DTypeInfo.BytesPerElement(_kvDType);
        long attnPerLayer =
              (long)_embDim * _numHeads * _headDim * 2 * sizeof(float)         // q (output qDim*2)
            + (long)_embDim * _numKvHeads * _headDim * sizeof(float) * 2       // k + v
            + (long)_embDim * _numHeads * _headDim * sizeof(float)             // o
            + (long)_maxSeqLen * _numKvHeads * _headDim * kvBytes * 2;          // kv cache

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
        // Sum each role's MAX per-expert footprint over ALL layers (issue #216), via the same
        // CudaExpertSlotManager.MaxRoleExpertBytes the slab uses — so PredictSlruSlots' predicted
        // capacity equals what the slab actually allocates. Q4_K/Q5_K/Q6_K stay raw; other dtypes
        // (Q3_K/Q8_0/…) expand to F32. A role's dtype varies per layer in K_M / Unsloth "UD" quants,
        // so a later F32-expanding layer dominates the stride — blk.0-only sizing would wildly
        // under-count and over-commit VRAM. Dims match CudaExpertSlotManager.UploadExpert:
        // gate/up are [ExpertIntermediateDim, EmbeddingDim], down is [EmbeddingDim, ExpertIntermediateDim].
        long Max(string role, int rows, int cols) =>
            CudaExpertSlotManager.MaxRoleExpertBytes(_model, _hp.NumLayers, role, rows, cols);
        long bytes =
              Max("ffn_gate_exps", _hp.ExpertIntermediateDim, _hp.EmbeddingDim)
            + Max("ffn_up_exps",   _hp.ExpertIntermediateDim, _hp.EmbeddingDim)
            + Max("ffn_down_exps", _hp.EmbeddingDim,          _hp.ExpertIntermediateDim);
        return bytes > 0 ? bytes : (long)(1.81 * 1024 * 1024);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count * (nuint)sizeof(float));

    /// <summary>
    /// Pinned-host counterpart to <see cref="Alloc"/>. Allocates via
    /// <c>cudaMallocHost</c> so the buffer can be DMA'd directly via the
    /// <see cref="CudaBackend.Download(Tensor, nint, int)"/> /
    /// <see cref="CudaBackend.UploadInto(Tensor, nint, int)"/> overloads,
    /// skipping the internal staging hop (issues #48/#49). Zeroes the buffer
    /// to match the AllocZeroed semantics of the pageable helper.
    /// </summary>
    private static float* AllocPinned(int count)
    {
        nuint byteSize = (nuint)count * (nuint)sizeof(float);
        nint ptr = CudaBackend.AllocatePinnedHost(byteSize);
        if (ptr == nint.Zero)
            throw new InvalidOperationException(
                $"CudaBackend.AllocatePinnedHost({byteSize}) failed; cannot allocate pinned host scratch.");
        new Span<byte>((void*)ptr, (int)byteSize).Clear();
        return (float*)ptr;
    }

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

        // Batched-prefill scratch (issue #110).
        FreeBatchedScratch();
        if (_bExpStart    != null) NativeMemory.Free(_bExpStart);
        if (_bExpCursor   != null) NativeMemory.Free(_bExpCursor);
        if (_bUsedExperts != null) NativeMemory.Free(_bUsedExperts);
        // Issue #121: N-independent GPU-MoE bucket arrays (FreeBatchedFfnScratch keeps them).
        if (_bfExpStart    != null) NativeMemory.Free(_bfExpStart);
        if (_bfExpCursor   != null) NativeMemory.Free(_bfExpCursor);
        if (_bfUsedExperts != null) NativeMemory.Free(_bfUsedExperts);
        // GPU op-offload routed-prefill scratch (perf/carnice-vnni-moe).
        FreeGpuOffloadScratch();

        // CPU buffers — _cpuNormBuf is pinned (issue #48).
        CudaBackend.FreePinnedHost((nint)_cpuNormBuf);
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
            if (_cpuNormInQ8K != null) NativeMemory.Free(_cpuNormInQ8K);
            if (_cpuExpertGateAllQ8K != null) NativeMemory.Free(_cpuExpertGateAllQ8K);
            // _cpuMoeHidden is pinned (issue #48).
            if (_cpuMoeHidden != null) CudaBackend.FreePinnedHost((nint)_cpuMoeHidden);
        }
        else if (!_hp.IsMoE)
        {
            // Dense FFN scratch (allocated alongside _cpuMoeHidden on the dense path).
            if (_cpuFfnGateBuf != null) NativeMemory.Free(_cpuFfnGateBuf);
            if (_cpuFfnUpBuf   != null) NativeMemory.Free(_cpuFfnUpBuf);
            // _cpuMoeHidden is pinned (issue #48).
            if (_cpuMoeHidden  != null) CudaBackend.FreePinnedHost((nint)_cpuMoeHidden);
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
        if (_splitKvPartialO is { } spo) _gpu.Free(spo);
        if (_splitKvPartialMeta is { } spm) _gpu.Free(spm);
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
                else if (_gpuRouterPrefill && _gpuWGateInp[i] is { } routerW)
                {
                    _gpu.Free(routerW);   // #388: router weight uploaded for the GPU prefill router
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
            if (_mtpIsMoE)
            {
                _gpu.Free(_gpuMtpWGateShexp);
                _gpu.Free(_gpuMtpWUpShexp);
                _gpu.Free(_gpuMtpWDownShexp);
                if (_cpuMtpFfnGateInpShexp != null) NativeMemory.Free(_cpuMtpFfnGateInpShexp);
            }
            else
            {
                _gpu.Free(_gpuMtpFfnGate);
                _gpu.Free(_gpuMtpFfnUp);
                _gpu.Free(_gpuMtpFfnDown);
            }
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
            // _lastHidden is pinned (issue #49).
            if (_lastHidden != null) CudaBackend.FreePinnedHost((nint)_lastHidden);
            if (_mtpPrefillHiddens != null)
            {
                NativeMemory.Free(_mtpPrefillHiddens);
                _mtpPrefillHiddens = null;
                _mtpPrefillHiddensCap = 0;
            }
            // Issue #30 batched-verify scratch: GPU tensors + pinned host
            // buffers (_cpuNormBuf2/_cpuMoeHidden2/_lastHiddenT1) are allocated
            // for all MTP-bearing models; dense-only intermediate buffers
            // (_gpuFfnGateBufDense2 etc.) are MoE-skip.
            if (_gpuHidden2      is { } h2) _gpu.Free(h2);
            if (_gpuResidual2    is { } r2) _gpu.Free(r2);
            if (_gpuNormBuf2     is { } n2) _gpu.Free(n2);
            if (_gpuLogits2      is { } l2) _gpu.Free(l2);
            if (_gpuLastHiddenT1 is { } lh1) _gpu.Free(lh1);
            // Pinned (issues #48/#49).
            if (_cpuNormBuf2   != null) CudaBackend.FreePinnedHost((nint)_cpuNormBuf2);
            if (_cpuMoeHidden2 != null) CudaBackend.FreePinnedHost((nint)_cpuMoeHidden2);
            if (_lastHiddenT1  != null) CudaBackend.FreePinnedHost((nint)_lastHiddenT1);
            if (!_hp.IsMoE)
            {
                if (_gpuFfnGateBufDense2 is { } gB2) _gpu.Free(gB2);
                if (_gpuFfnUpBufDense2   is { } uB2) _gpu.Free(uB2);
                if (_cpuFfnGateBuf2 != null) NativeMemory.Free(_cpuFfnGateBuf2);
                if (_cpuFfnUpBuf2   != null) NativeMemory.Free(_cpuFfnUpBuf2);
                if (_cpuFfnGateBuf3 != null) NativeMemory.Free(_cpuFfnGateBuf3);
                if (_cpuFfnUpBuf3   != null) NativeMemory.Free(_cpuFfnUpBuf3);
                if (_cpuFfnGateBuf4 != null) NativeMemory.Free(_cpuFfnGateBuf4);
                if (_cpuFfnUpBuf4   != null) NativeMemory.Free(_cpuFfnUpBuf4);
                if (_batchSnapshotBuf != null)
                {
                    NativeMemory.Free(_batchSnapshotBuf);
                    _batchSnapshotBuf = null;
                }
            }
            // Issue #30/#207-goal-4 k-token batched verify: device GDN snapshot
            // ring + exact-k verify scratch + the MTP self-chaining hidden.
            if (_gpuGdnRingScan is { } ringScan) _gpu.Free(ringScan);
            if (_gpuGdnRingConv is { } ringConv) _gpu.Free(ringConv);
            if (_gpuBvLogitsAll is { } bvl) _gpu.Free(bvl);
            if (_gpuBvFfnAll is { } bvf) _gpu.Free(bvf);
            if (_bvNormHost != null) CudaBackend.FreePinnedHost((nint)_bvNormHost);
            if (_bvFfnHost != null) CudaBackend.FreePinnedHost((nint)_bvFfnHost);
            if (_mtpSelfHidden != null) CudaBackend.FreePinnedHost((nint)_mtpSelfHidden);
            _mtpKvCache?.Dispose();
        }

        // SnapKV (issue #58) scratch (only allocated when active during a prefill).
        if (_snapKvQCapture is { } qc) _gpu.Free(qc);
        if (_snapKvScoreAccum is { } sa) _gpu.Free(sa);

        _kvCache.Dispose();
        _gdnStateCache.Dispose();
    }
}
