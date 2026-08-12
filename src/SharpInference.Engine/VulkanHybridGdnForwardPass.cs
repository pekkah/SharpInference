using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// Vulkan + CPU hybrid forward pass for the qwen35moe / qwen36 Gated-DeltaNet (GDN)
/// architecture (Qwen3.6-27B-MTP dense, Qwen3.6-35B-A3B MoE). The Vulkan mirror of
/// <see cref="CudaHybridGdnForwardPass"/> — the op sequences, dims, weight layouts, GDN
/// device-state lifecycle, and the dense CPU-FFN boundary are copied op-for-op from the
/// CUDA pass; this file only swaps the backend and threads in the Vulkan record/submit
/// session model (see "Vulkan deviations" below).
///
/// <para>Placement (mirrors the CUDA pass):</para>
/// <list type="bullet">
///   <item>GDN blocks (joint QKV projection, depthwise conv1d, L2-norm, delta-rule
///         recurrence, ssm-out projection): all GPU via <see cref="VulkanBackend"/>'s
///         GDN op-kit (issue #356 PRs 1-3).</item>
///   <item>Gated-attention blocks: all GPU (per-head Q/K RMSNorm, partial NEOX RoPE,
///         GQA SDPA, sigmoid GLU gate). Q/K/V/O weights stay VRAM-resident; KV cache
///         lives in VRAM (fp32) at <c>_gpuKCache[layer]</c> / <c>_gpuVCache[layer]</c>.</item>
///   <item>Dense FFN (Qwen3.6-27B-MTP): per-layer ffn_gate/up/down run on GPU when VRAM
///         permits (<see cref="TryUploadDenseFfnLayers"/>); the remaining layers run on
///         CPU from the mmap weights, through a pinned-buffer download/upload boundary.</item>
///   <item>MoE FFN (Qwen3.6-35B-A3B): the shared expert always runs on GPU; routed experts
///         run either on the CPU (<see cref="CpuMoeFfn"/> — auto-selected when VRAM can't
///         cache most experts, the 12 GB path) or via a GPU SLRU expert cache
///         (<see cref="GpuMoeFfn"/> — ≥24 GB). A per-token sigmoid scalar gate
///         (<c>sigmoid(ffn_gate_inp_shexp · norm)</c>) scales the shared-expert output.</item>
/// </list>
///
/// <para>Vulkan deviations from the CUDA pass (drive the whole file):</para>
/// <list type="bullet">
///   <item>Ops only RECORD between <c>BeginRecord()</c> / <c>EndRecordAndSubmit()</c>;
///         a CUDA stream has implicit ordering, Vulkan does not — every read-after-write
///         needs an explicit <c>RecordBarrier()</c>.</item>
///   <item>The CPU↔GPU boundary goes through the pinned <c>_pinnedHidden</c> tensor +
///         <c>CopyGpuBuffer</c> in-session, then <c>RecordComputeToHostBarrier()</c>
///         before the host reads (mirrors <see cref="HybridForwardPass"/>).</item>
///   <item><c>SiLU(Tensor)</c> (in-place) replaces CUDA's <c>SiLUInPlace</c>.</item>
///   <item>No EmbedLookupQ5K on Vulkan — Q5_K (and any non-Q4_K/Q6_K/F32) embedding is
///         dequantized to F32 at upload and read via <c>EmbedLookup</c>.</item>
///   <item>KV cache is plain fp32.</item>
/// </list>
///
/// <para>Scope (issue #356 PR4): the DENSE GDN model (Round 1) and the GDN+MoE model
/// (Round 2). Batched prefill (#356 PR5) and the k-token MTP batched-verify MECHANISM
/// (<see cref="BatchVerify"/> + <see cref="RestoreBatchSnapshot"/> + the device GDN snapshot
/// ring, #357 PR2) are implemented. #357 PR3 wires the NEXTN/MTP head itself
/// (<see cref="MtpForward"/> + <see cref="GpuMtpAttnBlock"/> + <see cref="PrefillMtp"/> + the MTP
/// KV cache + the absolute-position hidden-history surface), so <see cref="HasMtpHead"/> and
/// <see cref="SupportsBatchVerify"/> now report true on an MTP-bearing GGUF and the Qwen3.6 -MTP
/// GDN models do self-speculative decoding on Vulkan. (<see cref="SupportsPartialRewind"/> stays
/// false — the GDN recurrence is still destructively updated; rollback goes through the snapshot ring.)</para>
/// </summary>
public sealed unsafe class VulkanHybridGdnForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly VulkanBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly GdnConfig _gdn;
    private readonly LayerPlacement _placement;
    // -g N caps GPU-resident dense-FFN trunk layers on this pass, not a trunk split (mirrors
    // CudaHybridGdnForwardPass._denseFfnGpuCap — GDN/attention stay GPU-resident regardless).
    private readonly int _denseFfnGpuCap;
    private readonly int _maxSeqLen;

    // ── Dimensions (mirror CudaHybridGdnForwardPass.cs:69-87) ───────────
    private readonly int _embDim;
    private readonly int _headDim;          // attention head dim (256)
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _ropeDim;          // 64 (partial NEOX RoPE)
    private readonly int _gdnHeadDim;       // 128
    private readonly int _gdnNumVHeads;     // 32
    private readonly int _gdnNumKHeads;     // 16
    private readonly int _gdnKvRepeat;      // 2
    private readonly int _gdnValueDim;      // 4096
    private readonly int _gdnKeyDim;        // 2048
    private readonly int _gdnConvChannels;  // 8192
    private readonly int _gdnConvKernel;    // 4
    private readonly int _intermDim;        // dense FFN intermediate dim
    private readonly int _numExperts;       // MoE: routed expert count (qwen35moe)
    private readonly int _numActiveExperts; // MoE: top-K active routed experts
    private readonly int _expertDim;        // MoE: per-expert intermediate dim

    // ── GPU scratch (mirror :90-121 minus split-KV) ─────────────────────
    private readonly Tensor _gpuHidden;
    private readonly Tensor _gpuResidual;
    private readonly Tensor _gpuNormBuf;
    private readonly Tensor _gpuQGate;       // [numHeads * headDim * 2] = 8192 (Q‖gate interleaved per head)
    private readonly Tensor _gpuQ;           // [numHeads * headDim] = 4096
    private readonly Tensor _gpuGate;        // [numHeads * headDim] = 4096 (pre-sigmoid)
    private readonly Tensor _gpuK;           // [numKvHeads * headDim] = 512
    private readonly Tensor _gpuV;           // [numKvHeads * headDim] = 512
    private readonly Tensor _gpuAttnOut;     // [numHeads * headDim] = 4096
    private readonly Tensor _gpuAttnScratch; // attention scores spill scratch (>4096 ctx) or 1-float placeholder
    private readonly Tensor _gpuLogits;
    private readonly Tensor _pinnedHidden;   // host-mappable embDim float buffer for CPU↔GPU sync

    // ── GDN GPU scratch (mirror :201-209) ───────────────────────────────
    private readonly Tensor _gpuGdnQkv;      // [conv_channels]
    private readonly Tensor _gpuGdnQkvConv;  // [conv_channels] post-conv1d + SiLU
    private readonly Tensor _gpuGdnZVec;     // [value_dim]
    private readonly Tensor _gpuGdnQHead;    // [value_dim] (tiled to num_v_heads)
    private readonly Tensor _gpuGdnKHead;    // [value_dim] (tiled to num_v_heads)
    private readonly Tensor _gpuGdnVHead;    // [value_dim] (V slice copied out of QkvConv)
    private readonly Tensor _gpuGdnAlpha;    // [num_v_heads]
    private readonly Tensor _gpuGdnBeta;     // [num_v_heads]
    private readonly Tensor _gpuGdnOut;      // [value_dim]

    // ── MoE GPU scratch (only allocated when hp.IsMoE) — mirror :116-120 ──
    private readonly Tensor? _gpuRouterLogits;  // [numExperts]
    private readonly Tensor? _gpuFfnGate;       // [expertDim] shared/routed expert gate
    private readonly Tensor? _gpuFfnUp;         // [expertDim] shared/routed expert up
    private readonly Tensor? _gpuExpertOut;     // [embDim] routed expert down output (GPU-SLRU path)
    private readonly Tensor? _gpuSharedOut;     // [embDim] shared expert output
    private readonly Tensor? _pinnedNorm;       // host-mappable embDim float buffer (MoE norm readback)
    private readonly Tensor? _pinnedFallback;   // host-coherent embDim float buffer (GPU-SLRU CPU fallback combine)

    // ── Per-layer GPU weights (sized [NumLayers]; null/default slots for the
    //    block type that doesn't apply on that layer) — mirror :126-139 + :185-193 ─
    private readonly Tensor[] _gpuAttnNorm;       // [L] F32
    private readonly Tensor[] _gpuPostAttnNorm;   // [L] F32

    // MoE shared-expert + router GPU weights (only populated when hp.IsMoE; router +
    // shexp-gate stay on GPU only when !_cpuMoe — mirror :128-132 + :1125-1141).
    private readonly Tensor[] _gpuWGateInp;       // [L] router weight (GPU SLRU path only)
    private readonly Tensor[] _gpuWGateInpShexp;  // [L] shared-expert scalar gate (GPU SLRU path only)
    private readonly Tensor[] _gpuWGateShexp;     // [L] shared expert gate (both modes)
    private readonly Tensor[] _gpuWUpShexp;       // [L] shared expert up (both modes)
    private readonly Tensor[] _gpuWDownShexp;     // [L] shared expert down (both modes)

    // Attention-only (slots at GDN layers are unused)
    private readonly Tensor[] _gpuWQGate;        // [L] attn_q (GLU-gated, output 8192)
    private readonly Tensor[] _gpuWK;            // [L]
    private readonly Tensor[] _gpuWV;            // [L]
    private readonly Tensor[] _gpuWO;            // [L]
    private readonly Tensor[] _gpuQNorm;         // [L] [headDim] F32
    private readonly Tensor[] _gpuKNorm;         // [L] [headDim] F32

    // GPU KV cache (sized [numLayers]; only attention slots are allocated). fp32.
    private readonly Tensor?[] _gpuKCache;       // [L][maxSeq * kvDim]
    private readonly Tensor?[] _gpuVCache;       // [L][maxSeq * kvDim]

    // GPU-resident GDN weights (per layer, only populated for GDN-type layers).
    private readonly Tensor[] _gpuWAttnQkv;          // [L] [conv_channels, embDim]
    private readonly Tensor[] _gpuWAttnGate;         // [L] [value_dim, embDim]
    private readonly Tensor[] _gpuWSsmOut;           // [L] [embDim, value_dim]
    private readonly Tensor[] _gpuWSsmAlpha;         // [L] F32 [num_v_heads, embDim]
    private readonly Tensor[] _gpuWSsmBeta;          // [L] F32 [num_v_heads, embDim]
    private readonly Tensor[] _gpuSsmA;              // [L] F32 [num_v_heads]
    private readonly Tensor[] _gpuSsmDtBias;         // [L] F32 [num_v_heads]
    private readonly Tensor[] _gpuSsmNormW;          // [L] F32 [head_dim]
    private readonly Tensor[] _gpuSsmConv1d;         // [L] F32 [kernel, channels] — transposed

    // GPU-resident GDN per-sequence state, indexed by ABSOLUTE layer index (GDN slots
    // populated, attention slots null). Allocated once + Clear in the ctor (mirror :1197-1205).
    private readonly Tensor?[] _gpuGdnScanState;     // [L] F32 [num_v_heads, head_dim, head_dim]
    private readonly Tensor?[] _gpuGdnConvState;     // [L] F32 [kernel-1, conv_channels] oldest-first

    // ── MTP batched-verify (#357 PR2) ────────────────────────────────────
    // Device GDN snapshot ring for k-token batched verify (issues #30/#207/#357). Two flat
    // contiguous tensors sized [slots × numGdn × floatsPerLayer] so the fused scan/conv-capture
    // can stride across slots. Reserved at construction BEFORE the dense-FFN greedy VRAM fill
    // (mirror CudaHybridGdnForwardPass :1220-1273) so it isn't paged. Null when no MTP head.
    private Tensor? _gpuGdnRingScan;     // [slots × numGdn × scanFloatsPerLayer]
    private Tensor? _gpuGdnRingConv;     // [slots × numGdn × convFloatsPerLayer]
    private int _gdnRingSlots;           // captured slots; MaxBatchVerifyTokens = slots + 1
    private readonly int _mtpBatchMax = GdnStateCache.ResolveMtpBatchMax();
    // GGUF declares a NEXTN/MTP head → reserve the verify ring. The head WEIGHTS + HasMtpHead +
    // SupportsBatchVerify are wired in #357 PR3; PR2 only builds the verify+rollback mechanism.
    private bool _hasMtp;

    // ── MTP / NEXTN head (#357 PR3; mirror CudaHybridGdnForwardPass :340-473) ──
    // Loaded by LoadMtpHead when _hasMtp; the head is a single attention+FFN block plus the
    // NEXTN enorm/hnorm/eh_proj/shared_head_norm fusion. Dense (27B-MTP) and MoE (35B-A3B-MTP)
    // FFN tensor sets are mutually exclusive: the unused set stays null.
    private bool _mtpIsMoE;                      // MTP block uses MoE FFN (else dense)
    private Tensor? _gpuMtpAttnNorm;             // attn_norm.weight
    private Tensor? _gpuMtpWQGate;               // attn_q (Q‖gate interleaved, output qDim*2)
    private Tensor? _gpuMtpWK;                   // attn_k.weight
    private Tensor? _gpuMtpWV;                   // attn_v.weight
    private Tensor? _gpuMtpWO;                   // attn_output.weight
    private Tensor? _gpuMtpQNorm;                // attn_q_norm.weight [headDim] F32
    private Tensor? _gpuMtpKNorm;                // attn_k_norm.weight [headDim] F32
    private Tensor? _gpuMtpPostAttnNorm;         // post_attention_norm.weight
    // Dense MTP FFN (only when !_mtpIsMoE).
    private Tensor? _gpuMtpFfnGate;              // ffn_gate.weight
    private Tensor? _gpuMtpFfnUp;                // ffn_up.weight
    private Tensor? _gpuMtpFfnDown;              // ffn_down.weight
    // MoE MTP FFN (only when _mtpIsMoE): shared expert on GPU, routed/router on CPU (mmap).
    private Tensor? _gpuMtpWGateShexp;           // ffn_gate_shexp.weight
    private Tensor? _gpuMtpWUpShexp;             // ffn_up_shexp.weight
    private Tensor? _gpuMtpWDownShexp;           // ffn_down_shexp.weight
    private CpuWeightRef _cpuMtpFfnGateInp;      // router F32 [embDim, numExperts]
    private CpuWeightRef _cpuMtpFfnGateExps;     // [numExperts, expertDim, embDim]
    private CpuWeightRef _cpuMtpFfnUpExps;
    private CpuWeightRef _cpuMtpFfnDownExps;
    private float* _cpuMtpFfnGateInpShexp;       // [embDim] F32 shared-expert sigmoid gate (preloaded)
    // NEXTN fusion weights.
    private Tensor? _gpuMtpEnorm;                // nextn.enorm.weight
    private Tensor? _gpuMtpHnorm;                // nextn.hnorm.weight
    private Tensor? _gpuMtpSharedHeadNorm;       // nextn.shared_head_norm.weight
    private Tensor? _gpuMtpEhProj;               // nextn.eh_proj.weight (Q8_0→F32, [embDim*2 → embDim])
    // MTP attention KV cache (one slot; same layout as a trunk attention layer). fp32.
    private Tensor? _gpuMtpKCache;               // [maxSeq × kvDim]
    private Tensor? _gpuMtpVCache;               // [maxSeq × kvDim]
    private PagedKvCache? _mtpKvCache;           // length bookkeeping; data lives on GPU
    // Per-step MTP scratch (device).
    private Tensor? _gpuMtpEmbedBuf;             // [embDim] embedded MTP token
    private Tensor? _gpuMtpEnormBuf;             // [embDim] enorm(embedding)
    private Tensor? _gpuMtpHnormBuf;             // [embDim] hnorm(prevHidden)
    private Tensor? _gpuMtpConcatBuf;            // [embDim*2] [enorm ‖ hnorm]
    private Tensor? _gpuLastHidden;              // [embDim] prevHidden upload target
    private Tensor? _gpuMtpSelfHiddenDev;        // [embDim] device capture of the MTP pre-shared-head-norm hidden
    private Tensor? _gpuMtpHistDev;              // [embDim] device capture of the pre-output-norm trunk hidden (Forward)
    private Tensor? _pinnedMtpHidden;            // [embDim] dedicated pinned buffer for MTP host downloads/uploads
    // MTP host buffers (plain native memory; freed in Dispose).
    private float* _lastHidden;                  // [embDim] pre-output-norm hidden of the last main Forward
    private float* _mtpSelfHidden;               // [embDim] MTP block residual output (issue #30 chained drafting)
    private float* _mtpPrefillHiddens;           // [_mtpPrefillHiddensCap × embDim], slot p = h_p
    private int _mtpPrefillHiddensCap;           // allocated capacity in tokens
    private int _mtpHiddenHistoryLength;         // slots [0.._mtpHiddenHistoryLength) populated

    // ── Embedding + output ──────────────────────────────────────────────
    private readonly Tensor _gpuEmbedding;
    private readonly DType _embDType;     // dtype of the on-GPU embedding bytes (Q4_K, Q6_K, or Float32)
    private readonly Tensor _gpuOutputNorm;
    private readonly Tensor _gpuOutputWeight;

    // Dtype map driving the MatMul dispatch for raw-quant weights.
    private readonly Dictionary<nint, DType> _gpuWeightDTypes = new();

    // ── Caches ──────────────────────────────────────────────────────────
    private readonly GdnStateCache _gdnStateCache;
    private readonly PagedKvCache _kvCache;     // bookkeeping (block table) for attention layers; data lives on GPU

    // ── Dense FFN state (only populated when !hp.IsMoE) ──────────────────
    // Per-layer mmap weight refs (CPU FFN reads them per token).
    private readonly CpuWeightRef[]? _cpuWFfnGate;    // [L] ffn_gate.weight (Q4_K)
    private readonly CpuWeightRef[]? _cpuWFfnUp;      // [L] ffn_up.weight (Q4_K)
    private readonly CpuWeightRef[]? _cpuWFfnDown;    // [L] ffn_down.weight (Q6_K)
    // Per-layer GPU FFN slots. Populated lazily by TryUploadDenseFfnLayers when VRAM
    // headroom allows. Null slot → CpuDenseFfn(layer); non-null → GpuDenseFfn(layer).
    private Tensor?[]? _gpuWFfnGate;
    private Tensor?[]? _gpuWFfnUp;
    private Tensor?[]? _gpuWFfnDown;
    private Tensor? _gpuFfnGateBufDense;      // [_intermDim] f32 (only when ≥1 layer on GPU)
    private Tensor? _gpuFfnUpBufDense;        // [_intermDim] f32
    private int _denseFfnGpuLayers;           // diagnostic

    // CPU scratch for the dense FFN boundary.
    private readonly float* _cpuNormBuf;      // [embDim] — GPU norm download target
    private readonly float* _cpuMoeHidden;    // [embDim] — CPU FFN output, uploaded back
    private readonly float* _cpuFfnGateBuf;   // [_intermDim] scratch (dense only; null for MoE)
    private readonly float* _cpuFfnUpBuf;     // [_intermDim] scratch (dense only; null for MoE)

    // ── MoE FFN routing ──────────────────────────────────────────────────
    // CPU-MoE vs GPU-SLRU MoE selection (mirror :252 + :961-981). On a 12 GB card the
    // 35B's experts won't fit in VRAM, so the auto-heuristic selects CPU-MoE.
    private readonly bool _cpuMoe;
    // GPU-SLRU expert cache (only when hp.IsMoE && !_cpuMoe). Instantiate this pass's own.
    private readonly ExpertSlotManager? _expertSlotManager;
    private readonly MoEPrefetcher? _prefetcher;

    // CPU-MoE weight refs (only when _cpuMoe). Routed experts read from mmap per token.
    private readonly CpuWeightRef[]? _cpuFfnGateInp;       // [L] router F32 [embDim, numExperts]
    private readonly CpuWeightRef[]? _cpuFfnGateExps;      // [L] packed [numExperts, expertDim, embDim]
    private readonly CpuWeightRef[]? _cpuFfnUpExps;        // [L] packed
    private readonly CpuWeightRef[]? _cpuFfnDownExps;      // [L] packed
    private readonly float*[]? _cpuFfnGateInpShexp;        // [L][embDim] F32 (preloaded; small)

    // CPU-MoE scratch (only when _cpuMoe).
    private readonly float* _cpuRouterLogits;   // [numExperts]
    private readonly float* _cpuExpertGateAll;  // [numActive * expertDim]
    private readonly float* _cpuExpertUpAll;    // [numActive * expertDim]

    // GPU-SLRU CPU-fallback scratch (only when hp.IsMoE && !_cpuMoe).
    private readonly float[]? _gpuRouterBuf;    // [numExperts] router readback
    private float[]? _cpuFallbackBuf;           // [embDim] CPU-fallback expert accumulator
    private float[]? _cpuFallbackGate;          // [expertDim]
    private float[]? _cpuFallbackUp;            // [expertDim]

    private readonly float[] _logitsBuf;

    // Best-effort running estimate of weight bytes uploaded to VRAM. Vulkan has no
    // free-VRAM query (unlike CUDA's FreeVramBytes), so TryUploadDenseFfnLayers budgets
    // against VramBytes − this estimate − a safety margin (mirrors CUDA's EstimateUploadedVram).
    private long _uploadedVramBytes;

    // ── Batched prefill (issue #356 PR5b) ───────────────────────────────
    // One dispatch per trunk stage over a chunk of tokens, amortizing weight reads + removing
    // per-token launch overhead. Gated by SHARPI_VULKAN_BATCHED_PREFILL (default ON; =0 forces
    // the sequential per-token Forward loop). Byte-identical to the sequential path by
    // composition (each batched op == N sequential single-token ops; verified per-op in PR5a/#308).
    // MatMulBatched caps nTok at 8, so the batched path processes the admissible N in sub-chunks
    // of at most MaxBatchChunk tokens; each sub-chunk advances the device state by its size.
    private const int MaxBatchChunk = 8;
    private readonly bool _batchedPrefillEnabled;
    // PR5c (#356): opt-in FlashQLA chunk-parallel recurrence scan (GdnChunkedPrefill) in place of
    // the byte-exact fused GdnRecurrenceScan during clean batched prefill. Default OFF
    // (SHARPI_VULKAN_GDN_CHUNKED_PREFILL=1 to enable) + requires the device to fit the ~34 KB
    // shared-mem tile (SupportsGdnChunkedPrefill). Argmax-stable, NOT byte-exact (FP reduction
    // order differs). NOTE: meaningful speedup needs 64-token chunks, but the batched trunk is
    // capped at MaxBatchChunk=8 by MatMulBatched (#308 acc[8]); this drop-in runs the chunked scan
    // over the ≤8-token sub-chunk, so the end-to-end win is small until a "decoupled-64" rewiring
    // (or a larger batched matvec) lands. The validated kernel ships now; the wiring is a follow-up.
    private readonly bool _chunkedPrefillEnabled;
    private int _btCap;   // currently-allocated batched-scratch capacity (tokens); 0 = unallocated.

    // Batched trunk residual / norm (all [_btCap × embDim]). The GDN/attn blocks write the block
    // output directly into _gpuBtHidden (mirroring the scalar blocks, which write _gpuHidden).
    private Tensor? _gpuBtHidden;
    private Tensor? _gpuBtResidual;
    private Tensor? _gpuBtNorm;
    // Batched GDN scratch.
    private Tensor? _gpuBtQkv;       // [_btCap × convChannels]
    private Tensor? _gpuBtQkvConv;   // [_btCap × convChannels]
    private Tensor? _gpuBtZ;         // [_btCap × valueDim]
    private Tensor? _gpuBtQHead;     // [_btCap × valueDim]
    private Tensor? _gpuBtKHead;     // [_btCap × valueDim]
    private Tensor? _gpuBtAlpha;     // [_btCap × numVHeads]
    private Tensor? _gpuBtBeta;      // [_btCap × numVHeads]
    private Tensor? _gpuBtGdnOut;    // [_btCap × valueDim]
    // Batched attention scratch.
    private Tensor? _gpuBtQGate;     // [_btCap × qDim*2]
    private Tensor? _gpuBtQ;         // [_btCap × qDim]
    private Tensor? _gpuBtGate;      // [_btCap × qDim]
    private Tensor? _gpuBtK;         // [_btCap × kvDim]
    private Tensor? _gpuBtV;         // [_btCap × kvDim]
    private Tensor? _gpuBtAttnOut;   // [_btCap × qDim]

    private bool _disposed;

    // Pessimistic fault latch (mirror CudaHybridGdnForwardPass._faulted): the batched prefill
    // mutates the GDN recurrent state + advances the host length counters non-transactionally,
    // so a mid-chunk failure (Vulkan device-lost / OOM) leaves the state corrupted. Latch true
    // for the whole batched region, clear only on consistent completion; ThrowIfFaulted() then
    // blocks any retry on the poisoned state (discard the instance + reload the model).
    private bool _faulted;

    // Most-recent batched-verify snapshot bookkeeping (mirror CUDA _batchSnapshotValid/_batchStartPos/_batchK).
    private bool _batchSnapshotValid;
    private int _batchStartPos;
    private int _batchK;

    // Batched-verify [k×vocab] logits scratch (exact-size: MatMulBatched derives rows/cols from ElementCount/k).
    private Tensor? _gpuBvLogitsAll;
    private float[]? _bvLogitsHost;
    private int _bvCap;

    private void ThrowIfFaulted()
    {
        if (_faulted)
            throw new InvalidOperationException(
                "VulkanHybridGdnForwardPass: a prior batched prefill faulted mid-chunk, leaving the " +
                "GDN recurrent state corrupted. This instance can no longer produce correct output — " +
                "discard it and reload the model.");
    }

    public VulkanHybridGdnForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        LayerPlacement placement, int maxContextLength = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(gpu);
        ArgumentNullException.ThrowIfNull(hp);
        ArgumentNullException.ThrowIfNull(placement);

        // ── Validate (mirror CudaHybridGdnForwardPass.cs:727-740) ──────
        if (!hp.IsHybridSsm)
            throw new ArgumentException("VulkanHybridGdnForwardPass requires hp.IsHybridSsm=true.", nameof(hp));
        if (hp.Gdn is null)
            throw new ArgumentException("VulkanHybridGdnForwardPass requires hp.Gdn != null.", nameof(hp));
        if (hp.LayerTypes is null)
            throw new ArgumentException("VulkanHybridGdnForwardPass requires hp.LayerTypes != null.", nameof(hp));
        if (hp.IsMoE && !hp.HasSharedExpert)
            throw new ArgumentException("VulkanHybridGdnForwardPass with MoE requires a shared expert (qwen35moe layout).", nameof(hp));
        if (!hp.IsMoE && hp.IntermediateDim <= 0)
            throw new ArgumentException("VulkanHybridGdnForwardPass dense FFN requires hp.IntermediateDim > 0 (qwen35 dense layout).", nameof(hp));
        // gemma4 guard: per-layer head-dim models (PLE / shared-KV) are a different trunk.
        if (hp.LayerHeadDim is not null)
            throw new NotSupportedException(
                "VulkanHybridGdnForwardPass does not support per-layer-head-dim (gemma4-style) models.");

        // headDim=128 hard constraint (GdnRecurrenceDecode throws otherwise).
        if (hp.Gdn.HeadDim != 128)
            throw new NotSupportedException(
                $"VulkanHybridGdnForwardPass requires gdn.HeadDim == 128 (the Vulkan GDN recurrence shader's " +
                $"specialization); got {hp.Gdn.HeadDim}.");

        _model = model;
        _gpu = gpu;
        _hp = hp;
        _gdn = hp.Gdn;
        _placement = placement;
        _maxSeqLen = placement.RecommendedCtxSize > 0
            ? placement.RecommendedCtxSize
            : Math.Min(hp.ContextLength, 32768);
        if (maxContextLength > 0)
            _maxSeqLen = Math.Min(_maxSeqLen, maxContextLength);

        // ── Dims (mirror :742-770) ─────────────────────────────────────
        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _ropeDim = hp.RopeDim;
        _gdnHeadDim = _gdn.HeadDim;
        _gdnNumVHeads = _gdn.NumVHeads;
        _gdnNumKHeads = _gdn.NumKHeads;
        _gdnKvRepeat = _gdnNumVHeads / _gdnNumKHeads;
        _gdnValueDim = _gdn.ValueDim;
        _gdnKeyDim = _gdn.KeyDim;
        _gdnConvChannels = _gdn.ConvChannels;
        _gdnConvKernel = _gdn.ConvKernel;
        _intermDim = hp.IntermediateDim;
        _numExperts = hp.NumExperts;
        _numActiveExperts = hp.NumActiveExperts;
        _expertDim = hp.ExpertIntermediateDim;

        int L = hp.NumLayers;
        int qDim = _numHeads * _headDim;        // 4096
        int kvDim = _numKvHeads * _headDim;     // 512

        // Negative GpuLayers (e.g. -1, the CLI's "auto") means no cap; 0 means zero dense-FFN
        // layers on GPU (all CPU); any other value clamps to [0, L] — mirrors
        // CudaHybridGdnForwardPass._denseFfnGpuCap.
        _denseFfnGpuCap = placement.GpuLayers < 0 ? L : Math.Min(placement.GpuLayers, L);
        if (hp.IsMoE && _denseFfnGpuCap < L)
            Console.Error.WriteLine(
                $"[VulkanHybridGdnForwardPass] -g {_denseFfnGpuCap} requested a dense-FFN GPU cap, but "
                + "this model is MoE (no dense-FFN-on-GPU path) — the cap is a no-op here.");

        Console.Error.WriteLine($"[VulkanHybridGdnForwardPass] layers={L} embDim={_embDim} headDim={_headDim} numHeads={_numHeads} ropeDim={_ropeDim} ctx={_maxSeqLen}");
        if (hp.IsMoE)
            Console.Error.WriteLine($"[VulkanHybridGdnForwardPass] GDN: heads={_gdnNumVHeads}v×{_gdnNumKHeads}k headDim={_gdnHeadDim} conv={_gdnConvChannels}×{_gdnConvKernel}  MoE: {_numExperts}exp×{_numActiveExperts}active dim={_expertDim} (CPU-MoE or GPU-SLRU per VRAM).");
        else
            Console.Error.WriteLine($"[VulkanHybridGdnForwardPass] GDN: heads={_gdnNumVHeads}v×{_gdnNumKHeads}k headDim={_gdnHeadDim} conv={_gdnConvChannels}×{_gdnConvKernel}  Dense FFN intermDim={_intermDim} (per-layer GPU/CPU placement).");

        // ── Caches (mirror :922-923) ───────────────────────────────────
        _kvCache = new PagedKvCache(L, _numKvHeads, _headDim);
        _gdnStateCache = new GdnStateCache(hp.LayerTypes, _gdn);

        // ── GPU scratch (HybridForwardPass.cs:205-228 + GDN scratch) ────
        _gpuHidden = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuResidual = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuNormBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuQGate = gpu.Allocate(TensorShape.D1(qDim * 2));
        _gpuQ = gpu.Allocate(TensorShape.D1(qDim));
        _gpuGate = gpu.Allocate(TensorShape.D1(qDim));
        _gpuK = gpu.Allocate(TensorShape.D1(kvDim));
        _gpuV = gpu.Allocate(TensorShape.D1(kvDim));
        _gpuAttnOut = gpu.Allocate(TensorShape.D1(qDim));
        // Attention scores scratch — only needed when ctx > 4096 (HybridForwardPass.cs:297-300).
        long scratchElems = _maxSeqLen > 4096 ? (long)_numHeads * _maxSeqLen : 1L;
        _gpuAttnScratch = gpu.Allocate(TensorShape.D1(scratchElems));
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));

        // MoE GPU scratch (mirror :856-863). Dense FFN leaves these null.
        if (hp.IsMoE)
        {
            _gpuRouterLogits = gpu.Allocate(TensorShape.D1(_numExperts));
            _gpuFfnGate = gpu.Allocate(TensorShape.D1(_expertDim));
            _gpuFfnUp = gpu.Allocate(TensorShape.D1(_expertDim));
            _gpuExpertOut = gpu.Allocate(TensorShape.D1(_embDim));
            _gpuSharedOut = gpu.Allocate(TensorShape.D1(_embDim));
            // Host-mappable norm readback (shared-expert scalar gate dot + CPU router/fallback)
            // and a host-coherent buffer for the GPU-SLRU CPU-fallback combine.
            _pinnedNorm = gpu.AllocatePinned(TensorShape.D1(_embDim));
            _pinnedFallback = gpu.AllocatePinned(TensorShape.D1(_embDim));
        }

        _gpuGdnQkv     = gpu.Allocate(TensorShape.D1(_gdnConvChannels));
        _gpuGdnQkvConv = gpu.Allocate(TensorShape.D1(_gdnConvChannels));
        _gpuGdnZVec    = gpu.Allocate(TensorShape.D1(_gdnValueDim));
        _gpuGdnQHead   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads * _gdnHeadDim));
        _gpuGdnKHead   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads * _gdnHeadDim));
        _gpuGdnVHead   = gpu.Allocate(TensorShape.D1(_gdnValueDim));
        _gpuGdnAlpha   = gpu.Allocate(TensorShape.D1(_gdnNumVHeads));
        _gpuGdnBeta    = gpu.Allocate(TensorShape.D1(_gdnNumVHeads));
        _gpuGdnOut     = gpu.Allocate(TensorShape.D1(_gdnValueDim));

        _logitsBuf = new float[hp.VocabSize];

        // ── Resolve CPU-MoE vs GPU-SLRU MoE (mirror :961-981) ──────────
        // SLRU only pays off when most experts fit in VRAM. On a 12 GB card the 35B's
        // 256 experts × 40 layers won't fit, so the auto-heuristic selects CPU-MoE.
        // SHARPI_CPU_MOE: "1" force CPU, "0" force GPU-SLRU, unset → auto.
        if (hp.IsMoE)
        {
            string? cpuMoeOverride = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
            if (cpuMoeOverride == "1") _cpuMoe = true;
            else if (cpuMoeOverride == "0") _cpuMoe = false;
            else
            {
                int predictedSlots = PredictSlruSlots(L);
                int totalExperts = L * _numExperts;
                double ratio = totalExperts > 0 ? (double)predictedSlots / totalExperts : 1.0;
                _cpuMoe = ratio < 0.5;
                Console.Error.WriteLine(
                    $"[VulkanHybridGdnForwardPass] MoE auto-select: SLRU capacity ≈ {predictedSlots}/{totalExperts} ({ratio:P0}) → {(_cpuMoe ? "CPU" : "GPU SLRU")} MoE.  Override with SHARPI_CPU_MOE=0|1.");
            }
        }
        else
        {
            _cpuMoe = false;
        }

        // ── CPU scratch (FFN boundary) ─────────────────────────────────
        _cpuNormBuf = Alloc(_embDim);
        _cpuMoeHidden = Alloc(_embDim);
        if (!hp.IsMoE)
        {
            // Dense FFN scratch (intermDim-sized). MoE leaves these null.
            _cpuFfnGateBuf = Alloc(_intermDim);
            _cpuFfnUpBuf = Alloc(_intermDim);
        }
        if (_cpuMoe)
        {
            _cpuRouterLogits = Alloc(_numExperts);
            _cpuExpertGateAll = Alloc(_numActiveExperts * _expertDim);
            _cpuExpertUpAll = Alloc(_numActiveExperts * _expertDim);
        }
        else if (hp.IsMoE)
        {
            // GPU-SLRU MoE: router readback buffer (CPU fallback scratch is lazy).
            _gpuRouterBuf = new float[_numExperts];
        }

        // ── Embedding / output upload (mirror :933-951) ────────────────
        // Q4_K/Q6_K kept raw (Vulkan EmbedLookupQ4K/Q6K exist); Q5_K/other dequant→F32
        // (no EmbedLookupQ5K on Vulkan).
        if (ShouldKeepFixedWeightsOnGpu(
                model.FindTensor("token_embd.weight")!.Value,
                model.FindTensor("output.weight")))
        {
            _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embDType);
            _gpuOutputNorm = UploadWeight("output_norm.weight");
            _gpuOutputWeight = model.FindTensor("output.weight") is not null
                ? UploadWeight("output.weight")
                : _gpuEmbedding;
        }
        else
        {
            // Keep the throw (27B/35B Q4_K fit; the 2 GB single-storage-buffer limit is the
            // only failure mode and CPU embedding fallback is out of scope for v1 — the CUDA
            // sibling's PlanVram CPU fallback needs a GLSL EmbedLookup/matvec kernel suite this
            // pass doesn't have; see src/SharpInference.Vulkan/CLAUDE.md for the SPIR-V workflow).
            var embInfo = model.FindTensor("token_embd.weight")!.Value;
            var outInfo = model.FindTensor("output.weight");
            long embBytes = EstimateEmbeddingGpuBytes(embInfo);
            long outBytes = outInfo is not null ? EstimateWeightGpuBytes(outInfo.Value) : 0;
            throw new NotSupportedException(
                $"VulkanHybridGdnForwardPass: embedding (dtype={embInfo.DType}, ~{embBytes / (1024 * 1024)} MiB) "
                + (outInfo is not null
                    ? $"and/or output (dtype={outInfo.Value.DType}, ~{outBytes / (1024 * 1024)} MiB) "
                    : "(tied output) ")
                + "do not fit in a single 2 GiB GPU storage buffer; CPU embedding/output fallback is not "
                + "implemented on this backend. Use --backend cuda (which has a CPU fallback via PlanVram), "
                + "-g 0 for CPU-only execution, or reduce ctx size.");
        }

        // ── Per-layer tensor arrays (mirror :1009-1045) ────────────────
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

        if (hp.IsMoE)
        {
            // CPU-MoE: resolve routed-expert + router mmap refs; shexp gate preloaded F32.
            if (_cpuMoe)
            {
                _cpuFfnGateInp = new CpuWeightRef[L];
                _cpuFfnGateExps = new CpuWeightRef[L];
                _cpuFfnUpExps = new CpuWeightRef[L];
                _cpuFfnDownExps = new CpuWeightRef[L];
                _cpuFfnGateInpShexp = new float*[L];
                Console.Error.WriteLine(
                    "[VulkanHybridGdnForwardPass] CPU MoE mode: routed experts run on CPU (mmap); shared expert stays on GPU. SLRU disabled.");
            }
        }
        else
        {
            _cpuWFfnGate = new CpuWeightRef[L];
            _cpuWFfnUp = new CpuWeightRef[L];
            _cpuWFfnDown = new CpuWeightRef[L];
        }

        // ── Per-layer upload loop (mirror :1113-1210) ──────────────────
        Console.Error.Write("[VulkanHybridGdnForwardPass] Uploading per-layer weights...");
        for (int i = 0; i < L; i++)
        {
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuPostAttnNorm[i] = UploadWeight($"blk.{i}.post_attention_norm.weight");

            if (hp.IsMoE)
            {
                // Shared-expert weights stay GPU-resident in both modes (the CPU-MoE path
                // fires them off in parallel with the routed-expert CPU loop) (mirror :1125-1141).
                _gpuWGateShexp[i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                _gpuWUpShexp[i]   = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                _gpuWDownShexp[i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");

                if (!_cpuMoe)
                {
                    _gpuWGateInp[i]      = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                    _gpuWGateInpShexp[i] = UploadWeight($"blk.{i}.ffn_gate_inp_shexp.weight");
                }
                else
                {
                    _cpuFfnGateInp![i]      = ResolveCpuWeight($"blk.{i}.ffn_gate_inp.weight");
                    _cpuFfnGateExps![i]     = ResolveCpuWeight($"blk.{i}.ffn_gate_exps.weight");
                    _cpuFfnUpExps![i]       = ResolveCpuWeight($"blk.{i}.ffn_up_exps.weight");
                    _cpuFfnDownExps![i]     = ResolveCpuWeight($"blk.{i}.ffn_down_exps.weight");
                    _cpuFfnGateInpShexp![i] = LoadF32Tensor($"blk.{i}.ffn_gate_inp_shexp.weight", _embDim);
                }
            }
            else
            {
                // Dense FFN (qwen35 27B-MTP): resolve mmap refs only; CPU FFN reads them per
                // token. GPU upload (when it fits) happens in TryUploadDenseFfnLayers below.
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

                _gpuKCache[i] = AllocateTracked(TensorShape.D1((long)_maxSeqLen * kvDim));
                _gpuVCache[i] = AllocateTracked(TensorShape.D1((long)_maxSeqLen * kvDim));
            }
            else
            {
                _gpuWAttnQkv[i]   = UploadWeight($"blk.{i}.attn_qkv.weight");
                _gpuWAttnGate[i]  = UploadWeight($"blk.{i}.attn_gate.weight");
                _gpuWSsmOut[i]    = UploadWeight($"blk.{i}.ssm_out.weight");
                _gpuWSsmAlpha[i]  = UploadWeight($"blk.{i}.ssm_alpha.weight");
                _gpuWSsmBeta[i]   = UploadWeight($"blk.{i}.ssm_beta.weight");

                _gpuSsmA[i]       = UploadWeight($"blk.{i}.ssm_a");
                _gpuSsmDtBias[i]  = UploadWeight($"blk.{i}.ssm_dt.bias");
                _gpuSsmNormW[i]   = UploadWeight($"blk.{i}.ssm_norm.weight");
                // GGUF [channels, kernel] → [kernel, channels] (mirror :1194-1195 / :5663-5686).
                _gpuSsmConv1d[i]  = UploadConv1dTransposedToGpu($"blk.{i}.ssm_conv1d.weight",
                    _gdnConvKernel, _gdnConvChannels);

                // Allocate per-layer GDN state on GPU + zero it (mirror :1197-1205).
                long scanFloats = (long)_gdnNumVHeads * _gdnHeadDim * _gdnHeadDim;
                long convFloats = (long)(_gdnConvKernel - 1) * _gdnConvChannels;
                var scan = AllocateTracked(TensorShape.D1(scanFloats));
                var conv = AllocateTracked(TensorShape.D1(convFloats));
                ClearBracketed(scan);
                ClearBracketed(conv);
                _gpuGdnScanState[i] = scan;
                _gpuGdnConvState[i] = conv;
            }
            if ((i % 4) == 3) Console.Error.Write(".");
        }
        Console.Error.WriteLine(" done.");

        // ── MTP detection + GDN snapshot-ring reservation (#357 PR2; mirror CUDA :1220-1273) ──
        // Decided here (before the dense-FFN VRAM fill) so the ring is carved out first. The ring
        // enables the k-token batched verify; PR3 loads the actual NEXTN head + flips the gates.
        // Vulkan has no _cpuGdn (GDN always runs on GPU here), so CUDA's `!_cpuGdn` term is dropped.
        _hasMtp = hp.NumMtpLayers > 0
                  && model.FindTensor($"blk.{hp.NumLayers}.nextn.eh_proj.weight") is not null;
        if (_hasMtp && _gdnStateCache.NumGdnLayers > 0
            && Environment.GetEnvironmentVariable("SHARPI_DISABLE_MTP") != "1")
        {
            int numGdn = _gdnStateCache.NumGdnLayers;
            int scanF = _gdnStateCache.ScanStateFloatsPerLayer;
            int convF = _gdnStateCache.ConvStateFloatsPerLayer;
            int want = _mtpBatchMax - 1;
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
                    scanFlat = s; convFlat = c; got = trySlots;
                    // Account the reserved bytes against the VRAM estimate so TryUploadDenseFfnLayers
                    // budgets around the ring (Vulkan has no free-VRAM query).
                    _uploadedVramBytes += (long)trySlots * numGdn * (scanF + convF) * sizeof(float);
                    break;
                }
                catch (Exception ex)
                {
                    if (s is { } ps) gpu.Free(ps);
                    if (c is { } pc) gpu.Free(pc);
                    Console.Error.WriteLine(
                        $"[VulkanHybridGdnForwardPass] GDN ring allocation for {trySlots} slot(s) failed " +
                        $"({ex.GetType().Name}); retrying with fewer.");
                }
            }
            _gpuGdnRingScan = scanFlat;
            _gpuGdnRingConv = convFlat;
            _gdnRingSlots = got;
            long slotBytes = (long)numGdn * (scanF + convF) * sizeof(float);
            Console.Error.WriteLine(
                $"[VulkanHybridGdnForwardPass] MTP batched-verify GDN ring: {got} slot(s) × " +
                $"{slotBytes / (1024 * 1024)} MiB → max verify batch {got + 1} tokens.");
        }

        // ── FFN routing setup (mirror :1275-1299) ──────────────────────
        if (!hp.IsMoE)
        {
            // Dense FFN — no expert slot manager; greedily fill GPU FFN layers.
            TryUploadDenseFfnLayers(gpu, L);
        }
        else if (!_cpuMoe)
        {
            // GPU-SLRU MoE (≥24 GB cards). Size capacity from remaining VRAM
            // (VramBytes has no free query, so budget against _uploadedVramBytes + margin).
            long remaining = (long)gpu.VramBytes - _uploadedVramBytes - (2L << 30);
            long perExpertBytes = EstimatePerExpertBytes();
            int capacity = perExpertBytes > 0 ? (int)Math.Max(64, remaining / perExpertBytes) : 1024;
            int totalExperts = L * _numExperts;
            capacity = Math.Min(capacity, totalExperts);
            Console.Error.WriteLine(
                $"[VulkanHybridGdnForwardPass] SLRU expert cache: {capacity} slots / {totalExperts} total experts (per-expert ≈ {perExpertBytes / 1024} KiB, remaining VRAM ≈ {remaining / (1024 * 1024)} MiB).");
            _expertSlotManager = new ExpertSlotManager(gpu, model, hp, capacity, _gpuWeightDTypes);
            _prefetcher = new MoEPrefetcher(_expertSlotManager);
        }
        // CPU-MoE: no SLRU manager; routed experts read from mmap per token.

        // ── MTP / NEXTN head (#357 PR3) ────────────────────────────────
        // Loaded AFTER the dense-FFN routing so _gpuFfnGateBufDense/_gpuFfnUpBufDense exist for the
        // dense MTP FFN (allocated by TryUploadDenseFfnLayers when ≥1 trunk FFN layer lands on GPU;
        // LoadMtpHead allocates them itself when no trunk FFN layer did). _hasMtp + the verify ring
        // are decided above; this loads the actual head weights + flips the public gates.
        LoadMtpHead(gpu);

        // Pre-fault the CPU-resident mmap FFN weight pages (issue #221). Mirrors the CUDA
        // pass + HybridForwardPass.cs:473.
        MmapPrefault.Run("VulkanHybridGdnForwardPass", BuildCpuPrefaultRegions());

        // Batched prefill gate (issue #356 PR5b): default ON; SHARPI_VULKAN_BATCHED_PREFILL=0
        // forces the byte-identical sequential per-token Prefill loop so regressions isolate.
        _batchedPrefillEnabled =
            Environment.GetEnvironmentVariable("SHARPI_VULKAN_BATCHED_PREFILL") != "0";

        // PR5c (#356): opt-in chunked recurrence scan. Default OFF; only when the device fits the
        // ~34 KB shared tile (SupportsGdnChunkedPrefill). Argmax-stable, not byte-exact.
        bool chunkedRequested =
            Environment.GetEnvironmentVariable("SHARPI_VULKAN_GDN_CHUNKED_PREFILL") == "1";
        _chunkedPrefillEnabled = chunkedRequested && gpu.SupportsGdnChunkedPrefill;
        if (chunkedRequested && !gpu.SupportsGdnChunkedPrefill)
            Console.Error.WriteLine(
                "[VulkanHybridGdnForwardPass] SHARPI_VULKAN_GDN_CHUNKED_PREFILL=1 requested but the device's " +
                $"maxComputeSharedMemorySize ({gpu.MaxComputeSharedMemoryBytes} B) is below the ~34 KB the chunked " +
                "scan needs — falling back to the byte-exact fused GdnRecurrenceScan.");
    }

    // ================================================================
    //  IForwardPass surface
    // ================================================================

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _maxSeqLen;
    public bool SupportsPartialRewind => false;
    public bool HasMtpHead => _hasMtp;

    // #357 PR3: the MTP head is now loaded, so the draft source exists and MtpDecoder may select
    // this pass. Mirror CUDA SupportsBatchVerify :3334, minus the terms that don't apply on Vulkan:
    // there is no SnapKV on this pass (no KvCacheCompacted term) and GDN always runs on GPU here
    // (no _cpuGdn term; the ring is the only rollback mechanism, so _gdnRingSlots >= 1 is required).
    public bool SupportsBatchVerify => _hasMtp
        && (!_hp.IsMoE || _cpuMoe)
        && _gdnRingSlots >= 1
        && Environment.GetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY") != "1";

    /// <summary>Ceiling for a single <see cref="BatchVerify"/> batch = ring slots + 1
    /// (the slots reserved at construction, SHARPI_MTP_BATCH_MAX). 1 when no ring.</summary>
    public int MaxBatchVerifyTokens => _gdnRingSlots >= 1 ? _gdnRingSlots + 1 : 1;

    // VulkanBackend has no thread-affine context (only CUDA does). No-op per IThreadAffineBackend.
    public void BindToCurrentThread() { }

    /// <summary>Host-side bookkeeping cache (slot/length only; KV payload lives on the GPU).</summary>
    public PagedKvCache Cache => _kvCache;

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ThrowIfFaulted();
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        // Batched prefill (issue #356 PR5b): one dispatch per trunk stage over a chunk of tokens.
        // Admissible when: more than one token; the whole prefill stays within the AttentionBatched
        // shared-scores range (startPos + N ≤ 4096); and both host caches are a clean append at
        // startPos (device state advances chunk-by-chunk from there). Otherwise fall back to the
        // byte-identical sequential per-token Forward loop. The gate kill-switch forces sequential.
        if (_batchedPrefillEnabled
            && tokens.Count > 1
            && startPos + tokens.Count <= 4096
            && _kvCache.Length == startPos
            && _gdnStateCache.Length == startPos)
        {
            return PrefillBatched(tokens, startPos);
        }

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    /// <summary>Forward one token through the hybrid Vulkan + CPU stack (mirror :3164-3301).</summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        ThrowIfFaulted();
        // 1. Embedding → _gpuHidden (own record/submit bracket so the per-layer trunk
        //    starts from a fresh session; EmbedToken writes then a barrier before reads).
        _gpu.BeginRecord();
        EmbedToken(_gpuHidden, token);
        _gpu.RecordBarrier();

        // 2. Reserve KV cache page (layer-0 invariant; even if layer 0 is GDN) — :3174.
        _kvCache.ReserveBlock();

        // 3. Trunk layers. The session opened above stays open across all-GPU work; the
        //    dense CPU-FFN boundary closes and reopens it (see FfnDispatch). On entry to
        //    each layer the current session is recording.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // ── Pre-block residual + norm on GPU (:3180-3181) ──────
            CopyGpuBuffer(_gpuResidual, _gpuHidden);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            if (_hp.LayerTypes![layer] == LayerType.Attention)
                GpuAttnBlock(layer, position);
            else
                GpuGdnBlock(layer, position);

            // Residual add on GPU (:3198). The block's final op is a GpuMatMul writing
            // _gpuHidden with no trailing barrier; the read-after-write into AddInPlace needs
            // an explicit compute→compute barrier (CUDA gets this from implicit stream order).
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);
            _gpu.RecordBarrier();

            // ── Pre-FFN residual + norm on GPU (:3203-3204) ────────
            CopyGpuBuffer(_gpuResidual, _gpuHidden);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuPostAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // Dense FFN (:3206-3224). GPU-resident layers run in-session; CPU layers close
            // the session, run on the host, and reopen it (FfnDispatch keeps the session
            // recording on return).
            FfnDispatch(layer);

            // Residual add (:3244). The dense GPU FFN's final op (down-projection GpuMatMul)
            // writes _gpuHidden with no trailing barrier (the CPU-FFN path already barriers
            // after its pinned copy-back); guard the read-after-write into AddInPlace.
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);
            _gpu.RecordBarrier();
        }

        // 4. Advance position counters (:3250-3251).
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();

        // 5. Capture the pre-output-norm hidden for MTP (issue #29; mirror CUDA :3253-3265) into a
        //    dedicated device buffer BEFORE the in-place output-norm overwrites _gpuHidden. The host
        //    download happens after the submit (the in-session pinned-map idiom of this backend).
        if (_hasMtp)
        {
            CopyGpuBuffer(_gpuMtpHistDev!, _gpuHidden);
            _gpu.RecordBarrier();
        }

        // 6. Final norm + output projection on GPU (:3268-3269), then queue the logits D2H
        //    in-session and submit (mirror HybridForwardPass.cs:597-605).
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden);
        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        // 7. Download the captured pre-output-norm hidden into _lastHidden + the absolute-position
        //    history slot (mirror CUDA :3279-3290). Own session via the dedicated pinned buffer.
        if (_hasMtp)
            DownloadMtpHidden(_gpuMtpHistDev!, position);

        return _logitsBuf;
    }

    /// <summary>Download a captured pre-output-norm device hidden into the host MTP <see cref="_lastHidden"/>
    /// buffer and the absolute-position history slot for <paramref name="position"/>. Uses the dedicated
    /// pinned buffer in its own session so it never clashes with the shared logits staging.</summary>
    private void DownloadMtpHidden(Tensor src, int position)
    {
        _gpu.BeginRecord();
        CopyGpuBuffer(_pinnedMtpHidden!, src);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();
        float* p = _gpu.MapPinned(_pinnedMtpHidden!);
        var hidden = new ReadOnlySpan<float>(p, _embDim);
        hidden.CopyTo(new Span<float>(_lastHidden, _embDim));
        EnsureMtpHiddenHistoryCap(position + 1);
        hidden.CopyTo(new Span<float>(_mtpPrefillHiddens + (long)position * _embDim, _embDim));
        _gpu.UnmapPinned(_pinnedMtpHidden!);
        if (_mtpHiddenHistoryLength < position + 1)
            _mtpHiddenHistoryLength = position + 1;
    }

    // ================================================================
    //  Batched prefill (issue #356 PR5b) — one dispatch per trunk stage over a chunk of
    //  tokens. Byte-identical to N sequential Forward calls: each batched op reproduces the
    //  per-row computation of its single-token sibling (GdnRecurrenceScan ≡ N
    //  GdnRecurrenceDecode, MatMulBatched ≡ N MatMul, AttentionBatched ≡ N Attention, …).
    //  The GDN+attention TRUNK is batched (the win); the FFN runs per-row through the existing
    //  scalar helpers (MoE/CPU-FFN batching is out of scope for PR5b). MatMulBatched caps nTok
    //  at 8, so the admissible N is processed in sub-chunks of ≤ MaxBatchChunk tokens.
    // ================================================================

    private ReadOnlySpan<float> PrefillBatched(IReadOnlyList<int> tokens, int startPos)
    {
        EnsureBatchedScratch(MaxBatchChunk);
        int total = tokens.Count;
        int processed = 0;
        // Pessimistic fault latch (mirror CudaHybridGdnForwardPass): the per-chunk GDN-state
        // mutation + host length-counter advance is non-transactional, so poison the pass for the
        // whole region and clear only after every chunk completed consistently. A throw mid-chunk
        // (device-lost / OOM) leaves _faulted set, blocking any retry on corrupt recurrent state.
        _faulted = true;
        while (processed < total)
        {
            int n = Math.Min(MaxBatchChunk, total - processed);
            int chunkStartPos = startPos + processed;
            bool lastChunk = processed + n >= total;
            PrefillChunk(tokens, processed, n, chunkStartPos, lastChunk);
            processed += n;
        }
        _faulted = false;
        return _logitsBuf;
    }

    /// <summary>Run one ≤8-token chunk through the batched trunk. On the final chunk, the last
    /// token's final-norm + lm_head produces <see cref="_logitsBuf"/> (prefill returns last-token
    /// logits). Advances both host caches by <paramref name="n"/> on return.</summary>
    private void PrefillChunk(IReadOnlyList<int> tokens, int baseIdx, int n, int chunkStartPos, bool lastChunk)
    {
        int embDim = _embDim;
        // n-sized aliases: AddInPlace dispatches over dst.ElementCount, so the residual buffers
        // must report exactly n×embDim (not the MaxBatchChunk cap) to avoid touching stale rows.
        // RmsNormBatched takes an explicit n; CopyGpuBuffer copies the whole buffer (size-based),
        // both harmless on the cap-sized buffers — but aliasing keeps the extent honest throughout.
        Tensor hidden   = Alias(_gpuBtHidden!,   n, embDim);
        Tensor residual = Alias(_gpuBtResidual!, n, embDim);
        Tensor norm     = Alias(_gpuBtNorm!,     n, embDim);

        // Reserve KV pages covering this chunk's positions (one per token, like the scalar
        // Forward's per-token ReserveBlock), then open the trunk session and embed all N tokens.
        for (int i = 0; i < n; i++)
            _kvCache.ReserveBlockAt(chunkStartPos + i);

        long rowBytes = (long)embDim * sizeof(float);
        _gpu.BeginRecord();
        for (int i = 0; i < n; i++)
        {
            // VulkanBackend has no offset sub-view, so embed into the scalar _gpuHidden (offset 0)
            // then copy that row into row i of the batched hidden buffer. The embed shaders are
            // deterministic regardless of destination buffer, so this is byte-identical to a direct
            // per-row embed. A barrier separates the embed write from the copy read each iteration.
            EmbedToken(_gpuHidden, tokens[baseIdx + i]);
            _gpu.RecordBarrier();
            CopyGpuBufferRegion(hidden, (long)i * rowBytes, _gpuHidden, 0, rowBytes);
            _gpu.RecordBarrier();
        }

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Pre-block residual + norm over all N rows.
            CopyGpuBuffer(residual, hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNormBatched(norm, hidden, _gpuAttnNorm[layer], embDim, n, _hp.RmsNormEps);
            _gpu.RecordBarrier();

            if (_hp.LayerTypes![layer] == LayerType.Attention)
                GpuAttnBlockBatched(layer, n, chunkStartPos);
            else
                GpuGdnBlockBatched(layer, n);

            // Block residual add (block wrote `hidden`).
            _gpu.RecordBarrier();
            _gpu.AddInPlace(hidden, residual);
            _gpu.RecordBarrier();

            // Pre-FFN residual + norm over all N rows.
            CopyGpuBuffer(residual, hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNormBatched(norm, hidden, _gpuPostAttnNorm[layer], embDim, n, _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // FFN per-row through the existing scalar helpers (byte-identical to sequential).
            FfnDispatchBatched(layer, n);

            // FFN residual add.
            _gpu.RecordBarrier();
            _gpu.AddInPlace(hidden, residual);
            _gpu.RecordBarrier();
        }

        // Advance the host caches by N (device state advanced over the chunk inside the scan/conv
        // + KV-append). The session is still recording here.
        for (int i = 0; i < n; i++) { _kvCache.IncrementPosition(); _gdnStateCache.IncrementPosition(); }

        if (!lastChunk)
        {
            // Intermediate chunk: just submit the recorded trunk; no logits needed.
            _gpu.EndRecordAndSubmit();
            // MTP hidden history: _gpuBtHidden now holds this chunk's n pre-output-norm hiddens.
            if (_hasMtp) CaptureMtpChunkHiddens(hidden, chunkStartPos, n, setLastFromLastRow: false);
            return;
        }

        // Final chunk: final norm + output projection on the LAST token's row only, then download.
        // Copy the last row into the scalar _gpuHidden (no offset sub-view on Vulkan), then RmsNorm
        // in place — exactly the scalar Forward's final-norm step. _gpuBtHidden (= `hidden`) is
        // preserved (the final-norm writes _gpuHidden, a copy of the last row), so the MTP capture
        // below reads the unmodified pre-output-norm rows.
        CopyGpuBufferRegion(_gpuHidden, 0, hidden, (long)(n - 1) * embDim * sizeof(float), (long)embDim * sizeof(float));
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden);
        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        // MTP hidden history for the final chunk + set _lastHidden from its last token (mirror the
        // scalar Forward's _lastHidden population so batched prefill drives MTP too; CUDA :2573-2577).
        if (_hasMtp) CaptureMtpChunkHiddens(hidden, chunkStartPos, n, setLastFromLastRow: true);
    }

    /// <summary>Download a batched chunk's n pre-output-norm hiddens (rows of <paramref name="hidden"/>,
    /// = <see cref="_gpuBtHidden"/>) into <c>_mtpPrefillHiddens[chunkStartPos..]</c> + bump the history
    /// length. When <paramref name="setLastFromLastRow"/>, also copy the last row into
    /// <see cref="_lastHidden"/>. Own submit (no session open) via the grow-on-demand staging path.</summary>
    private void CaptureMtpChunkHiddens(Tensor hidden, int chunkStartPos, int n, bool setLastFromLastRow)
    {
        EnsureMtpHiddenHistoryCap(chunkStartPos + n);
        var dst = new Span<float>(_mtpPrefillHiddens + (long)chunkStartPos * _embDim, n * _embDim);
        _gpu.Download(Alias(hidden, n, _embDim), dst);
        if (_mtpHiddenHistoryLength < chunkStartPos + n)
            _mtpHiddenHistoryLength = chunkStartPos + n;
        if (setLastFromLastRow)
            new ReadOnlySpan<float>(_mtpPrefillHiddens + (long)(chunkStartPos + n - 1) * _embDim, _embDim)
                .CopyTo(new Span<float>(_lastHidden, _embDim));
    }

    /// <summary>Batched GDN block over N tokens — the op-for-op batched mirror of
    /// <see cref="GpuGdnBlock"/> (which is the byte-exact reference). Writes the block output into
    /// rows [0,N) of <see cref="_gpuBtHidden"/>.
    /// <para>When <paramref name="snapRing"/> is set (MTP batched verify, #357 PR2), the conv-state
    /// ring capture runs BEFORE the conv-state update and the recurrence scan mirrors the post-update
    /// per-head matrix state into the device GDN snapshot ring at every non-final token boundary
    /// (slots [0,N-1)), forcing the byte-exact fused <see cref="VulkanBackend.GdnRecurrenceScan"/>
    /// (the chunked prefill scan is bypassed). When clear (#356 batched prefill), behavior is
    /// unchanged.</para></summary>
    private void GpuGdnBlockBatched(int layer, int n, bool snapRing = false)
    {
        int convCh = _gdnConvChannels, valDim = _gdnValueDim, nVH = _gdnNumVHeads;
        int kDim = _gdnKeyDim, hd = _gdnHeadDim, embDim = _embDim;
        // n-sized aliases so MatMulBatched (rows/cols = ElementCount/n) and SiLU (ElementCount
        // dispatch bound) see the active chunk's extent, not the MaxBatchChunk cap.
        Tensor hidden = Alias(_gpuBtHidden!, n, embDim), norm = Alias(_gpuBtNorm!, n, embDim);
        Tensor qkvAll = Alias(_gpuBtQkv!, n, convCh), qkvConvAll = Alias(_gpuBtQkvConv!, n, convCh), zAll = Alias(_gpuBtZ!, n, valDim);
        Tensor qHeadAll = Alias(_gpuBtQHead!, n, valDim), kHeadAll = Alias(_gpuBtKHead!, n, valDim);
        Tensor alphaAll = Alias(_gpuBtAlpha!, n, nVH), betaAll = Alias(_gpuBtBeta!, n, nVH), gdnOutAll = Alias(_gpuBtGdnOut!, n, valDim);
        var scanState = _gpuGdnScanState[layer]!;
        var convState = _gpuGdnConvState[layer]!;

        // Ring geometry (snapRing only). nCapture = N-1: capture state after each non-final token.
        int gdnIdx = snapRing ? _gdnStateCache.GdnLayerOf(layer) : -1;
        int numGdn = _gdnStateCache.NumGdnLayers;
        int scanF = _gdnStateCache.ScanStateFloatsPerLayer;
        int convF = _gdnStateCache.ConvStateFloatsPerLayer;
        int nCapture = snapRing ? n - 1 : 0;

        // 1. Joint QKV + z (gate) + alpha/beta projections, batched.
        GpuMatMulBatched(qkvAll,   _gpuWAttnQkv[layer],  norm, n);
        GpuMatMulBatched(zAll,     _gpuWAttnGate[layer], norm, n);
        GpuMatMulBatched(alphaAll, _gpuWSsmAlpha[layer], norm, n);
        GpuMatMulBatched(betaAll,  _gpuWSsmBeta[layer],  norm, n);
        _gpu.RecordBarrier();

        // 2. Depthwise causal conv1d over all tokens (reads convState), then advance convState.
        //    WAR barrier between the read (decode) and the write (state update).
        _gpu.GdnConv1dDecodeBatched(qkvAll, convState, _gpuSsmConv1d[layer], qkvConvAll,
            convCh, _gdnConvKernel, n);
        _gpu.RecordBarrier();
        // snapRing: capture each non-final token's post-decode conv state into the ring BEFORE the
        // state update overwrites it (WAR hazard — capture READS convState, update WRITES it).
        if (snapRing && nCapture > 0 && convF > 0 && _gpuGdnRingConv is { } ringConv)
        {
            _gpu.GdnConv1dStateCaptureRing(qkvAll, convState, ringConv, (long)gdnIdx * convF,
                convCh, _gdnConvKernel, numGdn * convF, nCapture);
            _gpu.RecordBarrier();
        }
        _gpu.GdnConv1dStateUpdateBatched(qkvAll, convState, convCh, _gdnConvKernel, n);
        _gpu.RecordBarrier();

        // 3. SiLU over the whole [N × convCh].
        _gpu.SiLU(qkvConvAll);
        _gpu.RecordBarrier();

        // 4. L2-norm the Q (offset 0) and K (offset kDim) regions per head per token.
        _gpu.GdnL2NormPerHeadBatched(qkvConvAll, 0,    _gdnNumKHeads, hd, convCh, n, eps: 1e-6f);
        _gpu.GdnL2NormPerHeadBatched(qkvConvAll, kDim, _gdnNumKHeads, hd, convCh, n, eps: 1e-6f);
        _gpu.RecordBarrier();

        // 5. Tile Q and K heads (GQA broadcast) into the [N × valueDim] head buffers.
        _gpu.GdnTileHeadsBatched(qkvConvAll, 0,    qHeadAll, 0, _gdnNumKHeads, _gdnKvRepeat, hd, convCh, valDim, n);
        _gpu.GdnTileHeadsBatched(qkvConvAll, kDim, kHeadAll, 0, _gdnNumKHeads, _gdnKvRepeat, hd, convCh, valDim, n);
        _gpu.RecordBarrier();

        // 6. Recurrence scan over the chunk. Default: fused sequential scan (byte-exact vs N
        //    GdnRecurrenceDecode). Opt-in (PR5c): FlashQLA chunk-parallel scan (argmax-stable, not
        //    byte-exact). Both take identical args; v reads from the silu'd conv output's V region
        //    (vHeadOff = 2*kDim, stride convCh), q/k from the tiled head buffers, alpha/beta from
        //    [N × nVH], z from [N × valDim]. (The chunked win is small until N can exceed
        //    MaxBatchChunk=8 — see _chunkedPrefillEnabled; the validated kernel ships regardless.)
        if (snapRing)
            // MTP verify: byte-exact fused scan WITH ring capture (mirrors the post-update per-head
            // matrix state into slots [0,N-1)). The chunked prefill scan is never used here.
            _gpu.GdnRecurrenceScan(
                scanState, qHeadAll, kHeadAll, qkvConvAll,
                alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                zAll, gdnOutAll,
                nVH, hd, normEps: 1e-6f,
                qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                zStride: valDim, oStride: valDim, nTok: n,
                ringScan: _gpuGdnRingScan, ringScanFloatOffset: (long)gdnIdx * scanF,
                ringSlotStride: numGdn * scanF, nCapture: nCapture);
        else if (_chunkedPrefillEnabled)
            _gpu.GdnChunkedPrefill(
                scanState, qHeadAll, kHeadAll, qkvConvAll,
                alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                zAll, gdnOutAll,
                nVH, hd, normEps: 1e-6f,
                qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                zStride: valDim, oStride: valDim, nTok: n);
        else
            _gpu.GdnRecurrenceScan(
                scanState, qHeadAll, kHeadAll, qkvConvAll,
                alphaAll, betaAll, _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
                zAll, gdnOutAll,
                nVH, hd, normEps: 1e-6f,
                qStride: valDim, kStride: valDim, vStride: convCh, vHeadOff: 2 * kDim,
                zStride: valDim, oStride: valDim, nTok: n);
        _gpu.RecordBarrier();

        // 7. Output projection: blockOut = WSsmOut @ gdnOutAll → _gpuBtHidden.
        GpuMatMulBatched(hidden, _gpuWSsmOut[layer], gdnOutAll, n);
    }

    /// <summary>Batched attention block over N tokens — the op-for-op batched mirror of
    /// <see cref="GpuAttnBlock"/>. Writes the block output into rows [0,N) of
    /// <see cref="_gpuBtHidden"/>. Caller guarantees <c>chunkStartPos + N ≤ 4096</c>.</summary>
    private void GpuAttnBlockBatched(int layer, int n, int chunkStartPos)
    {
        int kvDim = _numKvHeads * _headDim, qDim = _numHeads * _headDim, embDim = _embDim;
        // n-sized aliases (see GpuGdnBlockBatched): MatMulBatched, SplitQGBatched's size check, and
        // SigmoidMulInPlace's ElementCount dispatch all need the active chunk extent, not the cap.
        Tensor hidden = Alias(_gpuBtHidden!, n, embDim), norm = Alias(_gpuBtNorm!, n, embDim);
        Tensor qGateAll = Alias(_gpuBtQGate!, n, qDim * 2), qAll = Alias(_gpuBtQ!, n, qDim), gateAll = Alias(_gpuBtGate!, n, qDim);
        Tensor kAll = Alias(_gpuBtK!, n, kvDim), vAll = Alias(_gpuBtV!, n, kvDim), attnOutAll = Alias(_gpuBtAttnOut!, n, qDim);

        // 1. Batched Q‖G / K / V projections.
        GpuMatMulBatched(qGateAll, _gpuWQGate[layer], norm, n);
        GpuMatMulBatched(kAll,     _gpuWK[layer],     norm, n);
        GpuMatMulBatched(vAll,     _gpuWV[layer],     norm, n);
        _gpu.RecordBarrier();

        // 2. De-interleave [Q‖G] → Q, G (arg order q, g, qg).
        _gpu.SplitQGBatched(qAll, gateAll, qGateAll, _numHeads, _headDim, n);
        _gpu.RecordBarrier();

        // 3. Per-head Q/K RMSNorm BEFORE RoPE.
        _gpu.HeadNormBatched(qAll, _gpuQNorm[layer], (uint)_numHeads,   (uint)_headDim, n, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNormBatched(kAll, _gpuKNorm[layer], (uint)_numKvHeads, (uint)_headDim, n, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.RecordBarrier();

        // 4. Partial NEOX RoPE on the first ropeDim of each head; token i at position chunkStartPos+i.
        _gpu.RoPEPartialBatched(qAll, chunkStartPos, _headDim, _ropeDim, _hp.RopeTheta, _numHeads,   n, neox: true);
        _gpu.RoPEPartialBatched(kAll, chunkStartPos, _headDim, _ropeDim, _hp.RopeTheta, _numKvHeads, n, neox: true);
        _gpu.RecordBarrier();

        // 5. Batched KV-append (token i → slot chunkStartPos+i; fp32 KV).
        _gpu.KvAppendBatched(kAll, vAll, _gpuKCache[layer]!, _gpuVCache[layer]!,
            (uint)kvDim, (uint)chunkStartPos, n, (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        // 6. Batched GQA SDPA: query i (pos chunkStartPos+i) attends causally over [0, chunkStartPos+i].
        _gpu.AttentionBatched(qAll, _gpuKCache[layer]!, _gpuVCache[layer]!, attnOutAll,
            (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, (uint)chunkStartPos, n, (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        // 7. Fused sigmoid GLU gate over the whole [N × qDim], then batched O projection.
        _gpu.SigmoidMulInPlace(attnOutAll, gateAll);
        _gpu.RecordBarrier();
        GpuMatMulBatched(hidden, _gpuWO[layer], attnOutAll, n);
    }

    /// <summary>Per-row FFN dispatch for the batched trunk: copies each row of the batched
    /// post-attn norm into the scalar <see cref="_gpuNormBuf"/>, runs the existing scalar FFN
    /// helper (dense-GPU / dense-CPU / CPU-MoE / GPU-SLRU — which writes <see cref="_gpuHidden"/>
    /// and manages its own session breaks), then copies <see cref="_gpuHidden"/> back into the
    /// row of the batched hidden buffer. Byte-identical to sequential because each scalar FFN call
    /// reproduces exactly the per-token FFN. On entry/return the session is recording.</summary>
    private void FfnDispatchBatched(int layer, int n)
    {
        int embDim = _embDim;
        Tensor hidden = _gpuBtHidden!, norm = _gpuBtNorm!;
        long rowBytes = (long)embDim * sizeof(float);
        for (int i = 0; i < n; i++)
        {
            // Row i of the batched post-attn norm → scalar norm buffer (compute-stage copy,
            // covered by the surrounding RecordBarrier()s; same as CopyGpuBuffer).
            CopyGpuBufferRegion(_gpuNormBuf, 0, norm, (long)i * rowBytes, rowBytes);
            _gpu.RecordBarrier();

            // Existing scalar FFN (reads _gpuNormBuf, writes _gpuHidden; may close/reopen session).
            FfnDispatch(layer);

            // _gpuHidden (row result) → row i of the batched hidden buffer.
            _gpu.RecordBarrier();
            CopyGpuBufferRegion(hidden, (long)i * rowBytes, _gpuHidden, 0, rowBytes);
            _gpu.RecordBarrier();
        }
    }

    /// <summary>Batched weight-stationary matvec over N rows (Q4_K/Q6_K amortize the weight read;
    /// other dtypes fall back to N single-row matvecs inside <see cref="VulkanBackend.MatMulBatched"/>).
    /// Byte-identical to N <see cref="GpuMatMul"/> calls. The dtype is the one recorded at upload.</summary>
    private void GpuMatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll, int n)
    {
        _gpu.MatMulBatched(outputAll, matrix, inputAll, n,
            _gpuWeightDTypes.TryGetValue(matrix.Handle, out var dt) ? dt : DType.Float32);
    }

    /// <summary>
    /// Lightweight aliasing view: a new <see cref="Tensor"/> sharing <paramref name="full"/>'s GPU
    /// buffer handle but reporting only the first <paramref name="rows"/> tokens × <paramref name="dim"/>
    /// elements. The batched-scratch buffers are sized for the fixed <see cref="MaxBatchChunk"/>; ops
    /// that derive their extent from <c>ElementCount</c> (MatMulBatched rows/cols; SplitQGBatched's
    /// size check; AddInPlace / SiLU dispatch bounds) must see the active chunk's <c>n</c>, not the
    /// cap. A <see cref="Tensor"/> is just (shape, dtype, handle), so this is free and reads/writes
    /// the same buffer from offset 0 — bit-identical to operating on a natively n-sized buffer.
    /// </summary>
    private static Tensor Alias(Tensor full, int rows, int dim) =>
        new(TensorShape.D1((long)rows * dim), full.DType, full.Handle);

    /// <summary>Allocate the batched-trunk scratch sized for <paramref name="cap"/> tokens (once;
    /// cap is the fixed MaxBatchChunk so no resize churn). All buffers are [cap × dim] row-major.</summary>
    private void EnsureBatchedScratch(int cap)
    {
        if (_btCap >= cap) return;
        FreeBatchedScratch();
        int embDim = _embDim, convCh = _gdnConvChannels, valDim = _gdnValueDim, nVH = _gdnNumVHeads;
        int qDim = _numHeads * _headDim, kvDim = _numKvHeads * _headDim;

        _gpuBtHidden   = _gpu.Allocate(TensorShape.D1((long)cap * embDim));
        _gpuBtResidual = _gpu.Allocate(TensorShape.D1((long)cap * embDim));
        _gpuBtNorm     = _gpu.Allocate(TensorShape.D1((long)cap * embDim));

        _gpuBtQkv     = _gpu.Allocate(TensorShape.D1((long)cap * convCh));
        _gpuBtQkvConv = _gpu.Allocate(TensorShape.D1((long)cap * convCh));
        _gpuBtZ       = _gpu.Allocate(TensorShape.D1((long)cap * valDim));
        _gpuBtQHead   = _gpu.Allocate(TensorShape.D1((long)cap * valDim));
        _gpuBtKHead   = _gpu.Allocate(TensorShape.D1((long)cap * valDim));
        _gpuBtAlpha   = _gpu.Allocate(TensorShape.D1((long)cap * nVH));
        _gpuBtBeta    = _gpu.Allocate(TensorShape.D1((long)cap * nVH));
        _gpuBtGdnOut  = _gpu.Allocate(TensorShape.D1((long)cap * valDim));

        _gpuBtQGate   = _gpu.Allocate(TensorShape.D1((long)cap * qDim * 2));
        _gpuBtQ       = _gpu.Allocate(TensorShape.D1((long)cap * qDim));
        _gpuBtGate    = _gpu.Allocate(TensorShape.D1((long)cap * qDim));
        _gpuBtK       = _gpu.Allocate(TensorShape.D1((long)cap * kvDim));
        _gpuBtV       = _gpu.Allocate(TensorShape.D1((long)cap * kvDim));
        _gpuBtAttnOut = _gpu.Allocate(TensorShape.D1((long)cap * qDim));

        _btCap = cap;
    }

    private void FreeBatchedScratch()
    {
        if (_btCap == 0) return;
        void F(ref Tensor? t) { if (t is { } v) { _gpu.Free(v); t = null; } }
        F(ref _gpuBtHidden); F(ref _gpuBtResidual); F(ref _gpuBtNorm);
        F(ref _gpuBtQkv); F(ref _gpuBtQkvConv); F(ref _gpuBtZ); F(ref _gpuBtQHead); F(ref _gpuBtKHead);
        F(ref _gpuBtAlpha); F(ref _gpuBtBeta); F(ref _gpuBtGdnOut);
        F(ref _gpuBtQGate); F(ref _gpuBtQ); F(ref _gpuBtGate); F(ref _gpuBtK); F(ref _gpuBtV); F(ref _gpuBtAttnOut);
        _btCap = 0;
    }

    // ================================================================
    //  MTP batched verify (#357 PR2) — the Vulkan analogue of CUDA's BatchVerify.
    // ================================================================

    /// <summary>
    /// k-token MTP batched verify (issues #30/#207/#357). Runs the #356 batched trunk (batched
    /// projections, batched attention at [startPos, startPos+k), fused delta-net scan with the
    /// device GDN snapshot ring captured at every non-final token boundary), per-row FFN, and an
    /// all-position [k×vocab] lm_head. Returns result[i] = logits after tokens[i]; rollback is
    /// <see cref="RestoreBatchSnapshot"/>. Byte-identical to k sequential <see cref="Forward"/>
    /// calls (every batched op == N single-token ops; ring capture is a separate-buffer write).
    /// </summary>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        ThrowIfFaulted();
        ArgumentNullException.ThrowIfNull(tokens);
        int k = tokens.Length;
        if (k == 0) return Array.Empty<float[]>();
        if (startPos < 0 || startPos + k > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos),
                $"BatchVerify range [{startPos}, {startPos + k}) exceeds the context window (maxSeqLen={_maxSeqLen}).");
        // PR2: SupportsBatchVerify stays false until PR3 wires the MTP head, so guard on the
        // concrete preconditions (ring present + clean caches) — the test drives BatchVerify directly.
        if (_gpuGdnRingScan is null || _gdnRingSlots < 1)
            throw new InvalidOperationException(
                "BatchVerify requires an allocated GDN snapshot ring (the GGUF must declare a NEXTN/MTP " +
                "head and SHARPI_DISABLE_MTP must be unset).");
        if (k > MaxBatchVerifyTokens)
            throw new ArgumentOutOfRangeException(nameof(tokens), k,
                $"BatchVerify token count exceeds MaxBatchVerifyTokens ({MaxBatchVerifyTokens}); " +
                "raise SHARPI_MTP_BATCH_MAX (ring slots are reserved at construction).");
        if (_kvCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _kvCache.Length={_kvCache.Length} != startPos={startPos}.");
        if (_gdnStateCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _gdnStateCache.Length={_gdnStateCache.Length} != startPos={startPos}.");
        if (k == 1)
        {
            // A single token amortizes nothing — plain Forward is strictly better (and still
            // advances the caches by one). It captures no ring snapshot, so afterward there is
            // no restorable batched-verify state: clear the flag so RestoreBatchSnapshot reports
            // "no snapshot held" rather than acting on a stale prior k>1 verify's bounds.
            var l = Forward(tokens[0], startPos);
            _batchSnapshotValid = false;
            return [l.ToArray()];
        }

        int embDim = _embDim;
        long rowBytes = (long)embDim * sizeof(float);
        // k ≤ MaxBatchVerifyTokens ≤ MaxBatchChunk today (ResolveMtpBatchMax clamps to [2,8]), but
        // Math.Max keeps the trunk scratch correctly sized if that coupling ever loosens — the
        // n-sized Alias views below would otherwise read past an undersized buffer on the GPU.
        EnsureBatchedScratch(Math.Max(MaxBatchChunk, k));
        EnsureBatchVerifyScratch(k);           // [k×vocab] logits

        // Pessimistic fault latch (mirror the batched prefill): the GDN-state mutation + length
        // advance is non-transactional, so poison the pass until the whole verify completes.
        _faulted = true;
        _batchSnapshotValid = false;

        Tensor hidden   = Alias(_gpuBtHidden!,   k, embDim);
        Tensor residual = Alias(_gpuBtResidual!, k, embDim);
        Tensor norm     = Alias(_gpuBtNorm!,     k, embDim);

        for (int i = 0; i < k; i++) _kvCache.ReserveBlockAt(startPos + i);

        _gpu.BeginRecord();
        for (int i = 0; i < k; i++)
        {
            EmbedToken(_gpuHidden, tokens[i]);
            _gpu.RecordBarrier();
            CopyGpuBufferRegion(hidden, (long)i * rowBytes, _gpuHidden, 0, rowBytes);
            _gpu.RecordBarrier();
        }

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            CopyGpuBuffer(residual, hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNormBatched(norm, hidden, _gpuAttnNorm[layer], embDim, k, _hp.RmsNormEps);
            _gpu.RecordBarrier();

            if (_hp.LayerTypes![layer] == LayerType.Attention)
                GpuAttnBlockBatched(layer, k, startPos);
            else
                GpuGdnBlockBatched(layer, k, snapRing: true);

            _gpu.RecordBarrier();
            _gpu.AddInPlace(hidden, residual);
            _gpu.RecordBarrier();

            CopyGpuBuffer(residual, hidden);
            _gpu.RecordBarrier();
            _gpu.RmsNormBatched(norm, hidden, _gpuPostAttnNorm[layer], embDim, k, _hp.RmsNormEps);
            _gpu.RecordBarrier();

            FfnDispatchBatched(layer, k);

            _gpu.RecordBarrier();
            _gpu.AddInPlace(hidden, residual);
            _gpu.RecordBarrier();
        }

        for (int i = 0; i < k; i++) { _kvCache.IncrementPosition(); _gdnStateCache.IncrementPosition(); }
        _batchStartPos = startPos;
        _batchK = k;
        _faulted = false;

        // All-position logits: batched output-norm over the k post-trunk hiddens + batched lm_head.
        // The output-norm writes _gpuBtNorm (normAll), so `hidden` (= _gpuBtHidden) is preserved
        // and still holds the k pre-output-norm hiddens for the MTP capture below.
        Tensor normAll   = Alias(_gpuBtNorm!, k, embDim);
        Tensor logitsAll = Alias(_gpuBvLogitsAll!, k, _hp.VocabSize);
        _gpu.RmsNormBatched(normAll, hidden, _gpuOutputNorm, embDim, k, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMulBatched(logitsAll, _gpuOutputWeight, normAll, k);
        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuBvLogitsAll!, k * _hp.VocabSize);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_bvLogitsHost.AsSpan(0, k * _hp.VocabSize));

        // MTP hidden history (issues #33/#106; mirror CUDA :3846-3853): row i of `hidden` holds the
        // pre-output-norm hidden for token startPos+i; _lastHidden = row k-1. Own submit (no session).
        if (_hasMtp) CaptureMtpChunkHiddens(hidden, startPos, k, setLastFromLastRow: true);

        _batchSnapshotValid = true;
        int vocab = _hp.VocabSize;
        var result = new float[k][];
        for (int i = 0; i < k; i++)
        {
            var row = new float[vocab];
            Array.Copy(_bvLogitsHost!, (long)i * vocab, row, 0, vocab);
            result[i] = row;
        }
        return result;
    }

    /// <summary>(Re)allocate the exact-size [k×vocab] batched-verify logits tensor + host buffer.
    /// Exact (not grow-only): MatMulBatched derives rows/cols from ElementCount/k.</summary>
    private void EnsureBatchVerifyScratch(int k)
    {
        if (_bvCap == k) return;
        if (_gpuBvLogitsAll is { } l) { _gpu.Free(l); _gpuBvLogitsAll = null; }
        _bvCap = -1;
        long logitsTotal = (long)k * _hp.VocabSize;
        if (logitsTotal > int.MaxValue)
            throw new NotSupportedException(
                $"Batched verify logits buffer ({k}×{_hp.VocabSize}) exceeds int.MaxValue.");
        _gpuBvLogitsAll = _gpu.Allocate(TensorShape.D1(logitsTotal));
        _bvLogitsHost = new float[(int)logitsTotal];
        _bvCap = k;
    }

    /// <summary>Ring slot → live device GDN state (scan + conv) for one layer. No-op for attention
    /// layers. Records device-to-device copies into the CURRENT session (caller brackets the submit).</summary>
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
            CopyGpuBufferRegion(scanT, 0, _gpuGdnRingScan,
                ((long)slot * numGdn * scanF + gdnIdx * scanF) * sizeof(float), scanBytes);
        if (_gpuGdnConvState[layer] is { } convT && convBytes > 0 && _gpuGdnRingConv is { } convRing)
            CopyGpuBufferRegion(convT, 0, convRing,
                ((long)slot * numGdn * convF + gdnIdx * convF) * sizeof(float), convBytes);
    }

    /// <summary>Roll the caches + device GDN state back to position <paramref name="lengthAfter"/> of
    /// the most recent <see cref="BatchVerify"/>: ring slot (lengthAfter-startPos-1) holds the state
    /// after the token at lengthAfter-1. Mirror CUDA :3580-3616 (GPU-GDN branch).</summary>
    public void RestoreBatchSnapshot(int lengthAfter)
    {
        if (!_batchSnapshotValid)
            throw new InvalidOperationException(
                "RestoreBatchSnapshot: no batched-verify snapshot is held. Call BatchVerify first.");
        int slot = lengthAfter - _batchStartPos - 1;
        if (slot < 0 || slot >= _batchK - 1)
            throw new ArgumentOutOfRangeException(nameof(lengthAfter), lengthAfter,
                $"RestoreBatchSnapshot: lengthAfter must be in [{_batchStartPos + 1}, " +
                $"{_batchStartPos + _batchK - 1}] — the most recent verify covered " +
                $"[{_batchStartPos}, {_batchStartPos + _batchK}).");

        _gpu.BeginRecord();
        for (int layer = 0; layer < _hp.NumLayers; layer++)
            RestoreGdnRingSlot(slot, layer);
        _gpu.EndRecordAndSubmit();

        _gdnStateCache.SetLength(lengthAfter);
        _kvCache.TruncateTo(lengthAfter);
        // Atomic with the trunk rewind (mirror CUDA :3612-3614): rewind MTP attention KV (the device
        // _gpuMtpKCache is a flat ring — future KvAppends overwrite stale slots) and clamp the
        // hidden-history length so PrefillMtp(suffix, startPos=lengthAfter) sees a consistent view.
        _mtpKvCache?.TruncateTo(lengthAfter);
        if (_hasMtp && _mtpHiddenHistoryLength > lengthAfter)
            _mtpHiddenHistoryLength = lengthAfter;
        _batchSnapshotValid = false;
    }

    public void ResetCache()
    {
        _kvCache.Reset();
        _gdnStateCache.Reset();
        // Zero GPU-resident scan + conv state for every GDN layer (mirror :3057-3062),
        // bracketed in its own record/submit session. Also zero the MTP KV cache when present.
        _gpu.BeginRecord();
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            if (_gpuGdnScanState[i] is { } scan) _gpu.Clear(scan);
            if (_gpuGdnConvState[i] is { } conv) _gpu.Clear(conv);
        }
        if (_hasMtp)
        {
            if (_gpuMtpKCache is { } kT) _gpu.Clear(kT);
            if (_gpuMtpVCache is { } vT) _gpu.Clear(vT);
        }
        _gpu.EndRecordAndSubmit();
        // MTP cache + hidden-history reset (mirror CUDA :3064-3074). Unconditional history reset
        // (no-op on non-MTP passes; guards against a future late-bind leaving stale state).
        if (_hasMtp) _mtpKvCache?.Reset();
        _mtpHiddenHistoryLength = 0;
    }

    public void TruncateTo(int length)
    {
        // GDN state is destructively updated; only a no-op (==Length) or full reset (0)
        // is supported (mirror :2985-3048, dense subset). SupportsPartialRewind == false.
        if (length == _gdnStateCache.Length)
        {
            _kvCache.TruncateTo(length);
            // Keep MTP attention KV in lockstep with the trunk (mirror CUDA :2990-2993) so a future
            // RestoreBatchSnapshot-without-MtpTruncateTo caller can't leave stale entries past length.
            _mtpKvCache?.TruncateTo(length);
            return;
        }
        if (length == 0)
        {
            ResetCache();
            return;
        }
        throw new NotSupportedException(
            $"VulkanHybridGdnForwardPass.TruncateTo({length}): GDN state is destructively updated and " +
            $"cannot be partially rewound; only length == 0 or current ({_gdnStateCache.Length}) is supported. " +
            "SupportsPartialRewind == false — check it before invoking TruncateTo with an intermediate length.");
    }

    // ================================================================
    //  GPU GDN block — full-GPU mirror of GpuGdnBlockAt :4493-4546.
    //  Consumes _gpuNormBuf, writes the block output into _gpuHidden.
    //  Barrier after every write whose result the next op reads.
    // ================================================================

    private void GpuGdnBlock(int layer, int position)
    {
        _ = position; // GDN recurrence is positional via the device state, not a push constant.
        var scanState = _gpuGdnScanState[layer]!;
        var convState = _gpuGdnConvState[layer]!;

        // 1. Joint QKV projection and z (gate) projection (:4499-4500).
        GpuMatMul(_gpuGdnQkv, _gpuWAttnQkv[layer], _gpuNormBuf);
        GpuMatMul(_gpuGdnZVec, _gpuWAttnGate[layer], _gpuNormBuf);
        _gpu.RecordBarrier();

        // 2. Depthwise causal conv1d (updates convState in place) (:4503).
        _gpu.GdnConv1dDecode(_gpuGdnQkv, convState, _gpuSsmConv1d[layer], _gpuGdnQkvConv,
            _gdnConvChannels, _gdnConvKernel);
        _gpu.RecordBarrier();

        // 3. SiLU on the conv output (whole 8192) (:4507 — SiLU, not SiLUInPlace on Vulkan).
        _gpu.SiLU(_gpuGdnQkvConv);
        _gpu.RecordBarrier();

        // 4. L2-norm per K-head on the Q and K slices (:4514-4517). Layout of _gpuGdnQkvConv:
        //      [0 .. key_dim)            → Q (k_heads × head_dim)
        //      [key_dim .. 2*key_dim)    → K
        //      [2*key_dim .. conv_chan)  → V (v_heads × head_dim)
        _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, 0, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        _gpu.GdnL2NormPerHead(_gpuGdnQkvConv, _gdnKeyDim, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        _gpu.RecordBarrier();

        // 5. Tile K-heads → V-head count (Hk=16, Hv=32, repeat=2) (:4520-4523).
        _gpu.GdnTileHeads(_gpuGdnQkvConv, 0, _gpuGdnQHead, 0,
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);
        _gpu.GdnTileHeads(_gpuGdnQkvConv, _gdnKeyDim, _gpuGdnKHead, 0,
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);
        _gpu.RecordBarrier();

        // 6. Alpha / Beta per-v-head projections (:4526-4527).
        GpuMatMul(_gpuGdnAlpha, _gpuWSsmAlpha[layer], _gpuNormBuf);
        GpuMatMul(_gpuGdnBeta,  _gpuWSsmBeta[layer],  _gpuNormBuf);

        // 7. Copy the V slice (final value_dim floats of _gpuGdnQkvConv) into _gpuGdnVHead
        //    (:4532-4534; byte offsets, the SiLU barrier above already published the V slice).
        CopyGpuBufferRegion(_gpuGdnVHead, 0,
            _gpuGdnQkvConv, 2L * _gdnKeyDim * sizeof(float),
            (long)_gdnValueDim * sizeof(float));
        _gpu.RecordBarrier();

        // 8. Recurrence: rank-1 state update + per-head RMSNorm + SiLU(z) gate (GPU) (:4537).
        _gpu.GdnRecurrenceDecode(
            scanState, _gpuGdnQHead, _gpuGdnKHead, _gpuGdnVHead,
            _gpuGdnAlpha, _gpuGdnBeta,
            _gpuSsmA[layer], _gpuSsmDtBias[layer], _gpuSsmNormW[layer],
            _gpuGdnZVec, _gpuGdnOut,
            _gdnNumVHeads, _gdnHeadDim, normEps: 1e-6f);
        _gpu.RecordBarrier();

        // 9. Output projection: ssm_out (input value_dim=4096, output emb_dim) (:4545).
        GpuMatMul(_gpuHidden, _gpuWSsmOut[layer], _gpuGdnOut);
    }

    // ================================================================
    //  GPU attention block — GLU-gated Q, partial NEOX RoPE on first 64 dims.
    //  Mirror of GpuAttnBlockAt :4060-4129.
    // ================================================================

    private void GpuAttnBlock(int layer, int position)
    {
        int kvDim = _numKvHeads * _headDim;
        int kvPosition = _kvCache.Length;

        GpuMatMul(_gpuQGate, _gpuWQGate[layer], _gpuNormBuf);
        GpuMatMul(_gpuK, _gpuWK[layer], _gpuNormBuf);
        GpuMatMul(_gpuV, _gpuWV[layer], _gpuNormBuf);
        _gpu.RecordBarrier();

        // Arg order is q, g, qg (:4069).
        _gpu.SplitQG(_gpuQ, _gpuGate, _gpuQGate, _numHeads, _headDim);
        _gpu.RecordBarrier();

        // Per-head Q/K RMSNorm BEFORE RoPE (:4071-4072).
        _gpu.HeadNorm(_gpuQ, _gpuQNorm[layer], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNorm(_gpuK, _gpuKNorm[layer], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.RecordBarrier();

        // Partial NEOX RoPE on the first ropeDim of each head (:4074-4075).
        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RecordBarrier();

        // Append K/V to the per-layer VRAM cache (fp32) (:4099-4101).
        _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[layer]!, _gpuVCache[layer]!,
            (uint)kvDim, (uint)kvPosition, (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        // GQA scaled-dot-product attention (:4122). window:0 = full causal.
        int seqLen = kvPosition + 1;
        _gpu.Attention(_gpuQ, _gpuKCache[layer]!, _gpuVCache[layer]!, _gpuAttnOut, _gpuAttnScratch,
            (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, (uint)seqLen, (uint)_maxSeqLen, window: 0u);
        _gpu.RecordBarrier();

        // Fused sigmoid GLU gate: attnOut *= sigmoid(gate) (:4126).
        _gpu.SigmoidMulInPlace(_gpuAttnOut, _gpuGate);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWO[layer], _gpuAttnOut);
    }

    // ================================================================
    //  MTP / NEXTN head (#357 PR3) — the Vulkan analogue of
    //  CudaHybridGdnForwardPass.MtpForward / GpuMtpAttnBlock / PrefillMtp,
    //  translated to the record/submit + pinned-map session model.
    //  Mirror CUDA :4151-4383 op-for-op; the only deviations are the Vulkan
    //  explicit-barrier requirement and the host download/upload going through
    //  the dedicated _pinnedMtpHidden buffer (no UploadInto / async D2H here).
    // ================================================================

    /// <inheritdoc />
    public ReadOnlySpan<float> LastHidden =>
        _hasMtp ? new ReadOnlySpan<float>(_lastHidden, _embDim) : default;

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpLastHidden =>
        _mtpSelfHidden != null ? new ReadOnlySpan<float>(_mtpSelfHidden, _embDim) : default;

    /// <inheritdoc />
    public ReadOnlySpan<float> HiddenAt(int position)
    {
        if (!_hasMtp || position < 0 || position >= _mtpHiddenHistoryLength)
            return default;
        return new ReadOnlySpan<float>(_mtpPrefillHiddens + (long)position * _embDim, _embDim);
    }

    /// <summary>Load the NEXTN/MTP head weights + caches + per-step scratch (mirror CUDA :1301-1448).
    /// No-op when the GGUF declares no MTP head. Called from the ctor AFTER the dense-FFN routing so
    /// the dense FFN scratch buffers exist (allocated here when no trunk FFN layer landed on GPU).</summary>
    private void LoadMtpHead(VulkanBackend gpu)
    {
        if (!_hasMtp) return;
        int mtpLayerIdx = _hp.NumLayers;

        _gpuMtpAttnNorm     = UploadWeight($"blk.{mtpLayerIdx}.attn_norm.weight");
        _gpuMtpWQGate       = UploadWeight($"blk.{mtpLayerIdx}.attn_q.weight");
        _gpuMtpWK           = UploadWeight($"blk.{mtpLayerIdx}.attn_k.weight");
        _gpuMtpWV           = UploadWeight($"blk.{mtpLayerIdx}.attn_v.weight");
        _gpuMtpWO           = UploadWeight($"blk.{mtpLayerIdx}.attn_output.weight");
        _gpuMtpQNorm        = UploadWeight($"blk.{mtpLayerIdx}.attn_q_norm.weight");
        _gpuMtpKNorm        = UploadWeight($"blk.{mtpLayerIdx}.attn_k_norm.weight");
        _gpuMtpPostAttnNorm = UploadWeight($"blk.{mtpLayerIdx}.post_attention_norm.weight");

        // MoE-MTP vs dense-MTP probe (mirror CUDA :1325-1333). MoE MTP requires trunk MoE + CPU-MoE
        // (the routed-expert stack at the MTP block won't co-reside with the trunk experts on a
        // 12 GB GPU; the GPU-SLRU cache reserves no slots for the extra layer).
        _mtpIsMoE = _model.FindTensor($"blk.{mtpLayerIdx}.ffn_gate_exps.weight") is not null;
        if (_mtpIsMoE && !_hp.IsMoE)
            throw new NotSupportedException(
                "MoE MTP head requires trunk MoE. Dense-trunk + MoE-MTP-head is not a supported configuration.");
        if (_mtpIsMoE && !_cpuMoe)
            throw new NotSupportedException(
                "MoE MTP head requires CPU MoE mode (SHARPI_CPU_MOE=1). The GPU-SLRU expert cache " +
                "doesn't reserve slots for the MTP block.");

        if (_mtpIsMoE)
        {
            _gpuMtpWGateShexp = UploadWeight($"blk.{mtpLayerIdx}.ffn_gate_shexp.weight");
            _gpuMtpWUpShexp   = UploadWeight($"blk.{mtpLayerIdx}.ffn_up_shexp.weight");
            _gpuMtpWDownShexp = UploadWeight($"blk.{mtpLayerIdx}.ffn_down_shexp.weight");
            _cpuMtpFfnGateInp = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_gate_inp.weight");
            _cpuMtpFfnGateExps = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_gate_exps.weight");
            _cpuMtpFfnUpExps   = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_up_exps.weight");
            _cpuMtpFfnDownExps = ResolveCpuWeight($"blk.{mtpLayerIdx}.ffn_down_exps.weight");
            _cpuMtpFfnGateInpShexp = LoadF32Tensor($"blk.{mtpLayerIdx}.ffn_gate_inp_shexp.weight", _embDim);
        }
        else
        {
            _gpuMtpFfnGate = UploadWeight($"blk.{mtpLayerIdx}.ffn_gate.weight");
            _gpuMtpFfnUp   = UploadWeight($"blk.{mtpLayerIdx}.ffn_up.weight");
            _gpuMtpFfnDown = UploadWeight($"blk.{mtpLayerIdx}.ffn_down.weight");
        }

        _gpuMtpEnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.enorm.weight");
        _gpuMtpHnorm          = UploadWeight($"blk.{mtpLayerIdx}.nextn.hnorm.weight");
        _gpuMtpSharedHeadNorm = UploadWeight($"blk.{mtpLayerIdx}.nextn.shared_head_norm.weight");
        // eh_proj is Q8_0 in GGUF; UploadWeight dequants non-{F32,Q4_K,Q5_K,Q6_K,Q8_0,Q4_0} to F32,
        // but Q8_0 is kept raw on Vulkan and the matvec dispatches on it directly — either way the
        // [embDim*2 → embDim] projection serves _gpuMtpConcatBuf.
        _gpuMtpEhProj         = UploadWeight($"blk.{mtpLayerIdx}.nextn.eh_proj.weight");

        // MTP attention KV cache on GPU (one slot; same fp32 layout as a trunk attention layer).
        int mtpKvDim = _numKvHeads * _headDim;
        _gpuMtpKCache = AllocateTracked(TensorShape.D1((long)_maxSeqLen * mtpKvDim));
        _gpuMtpVCache = AllocateTracked(TensorShape.D1((long)_maxSeqLen * mtpKvDim));
        ClearBracketed(_gpuMtpKCache);
        ClearBracketed(_gpuMtpVCache);

        // Per-step scratch (device).
        _gpuMtpEmbedBuf     = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuMtpEnormBuf     = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuMtpHnormBuf     = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuMtpConcatBuf    = gpu.Allocate(TensorShape.D1(_embDim * 2));
        _gpuLastHidden      = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuMtpSelfHiddenDev = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuMtpHistDev      = gpu.Allocate(TensorShape.D1(_embDim));
        _pinnedMtpHidden    = gpu.AllocatePinned(TensorShape.D1(_embDim));

        // Bookkeeping cache (PagedKvCache for the layer-0 invariant + length tracking).
        _mtpKvCache = new PagedKvCache(numLayers: 1, _numKvHeads, _headDim);

        // Host MTP buffers.
        _lastHidden    = Alloc(_embDim);
        _mtpSelfHidden = Alloc(_embDim);

        // The MTP dense FFN runs on GPU regardless of trunk FFN placement; allocate the dense FFN
        // scratch when TryUploadDenseFfnLayers didn't (no trunk FFN layer landed on GPU). Mirror
        // CUDA :1398-1402; cost is 2 × intermDim × 4 B.
        if (!_hp.IsMoE && _gpuFfnGateBufDense is null)
        {
            _gpuFfnGateBufDense = AllocateTracked(TensorShape.D1(_intermDim));
            _gpuFfnUpBufDense   = AllocateTracked(TensorShape.D1(_intermDim));
        }

        Console.Error.WriteLine(
            $"[VulkanHybridGdnForwardPass] MTP/NEXTN head loaded (blk.{mtpLayerIdx}, {(_mtpIsMoE ? "MoE" : "dense")} FFN). " +
            "HasMtpHead + SupportsBatchVerify enabled.");
    }

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
    {
        ThrowIfFaulted();
        if (!_hasMtp)
            throw new InvalidOperationException(
                "MtpForward called on a VulkanHybridGdnForwardPass that did not load an MTP head. " +
                "Check HasMtpHead before calling.");
        if (prevHidden.Length != _embDim)
            throw new ArgumentException(
                $"prevHidden length {prevHidden.Length} != EmbeddingDim {_embDim}.", nameof(prevHidden));

        long embBytes = (long)_embDim * sizeof(float);

        // 1. Upload prevHidden into _gpuLastHidden via the dedicated pinned buffer (no UploadInto on
        //    Vulkan): map → copy host span in → unmap → device copy in-session.
        {
            float* p = _gpu.MapPinned(_pinnedMtpHidden!);
            prevHidden.CopyTo(new Span<float>(p, _embDim));
            _gpu.UnmapPinned(_pinnedMtpHidden!);
        }

        _gpu.BeginRecord();
        CopyGpuBuffer(_gpuLastHidden!, _pinnedMtpHidden!);
        _gpu.RecordBarrier();

        // 2. Embed token → _gpuMtpEmbedBuf.
        EmbedToken(_gpuMtpEmbedBuf!, token);
        _gpu.RecordBarrier();

        // 3. enorm(embedding) → _gpuMtpEnormBuf; hnorm(prevHidden) → _gpuMtpHnormBuf.
        _gpu.RmsNorm(_gpuMtpEnormBuf!, _gpuMtpEmbedBuf!, _gpuMtpEnorm!, _hp.RmsNormEps);
        _gpu.RmsNorm(_gpuMtpHnormBuf!, _gpuLastHidden!,  _gpuMtpHnorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();

        // 4. Concat [enorm(e) ‖ hnorm(h)] into _gpuMtpConcatBuf [embDim*2]. The enorm half comes
        //    FIRST (the transformers Qwen3NextNextNDecoderLayer order); the inverted order produces
        //    0% draft acceptance (see the CPU/CUDA MtpForward notes).
        CopyGpuBufferRegion(_gpuMtpConcatBuf!, 0,        _gpuMtpEnormBuf!, 0, embBytes);
        CopyGpuBufferRegion(_gpuMtpConcatBuf!, embBytes, _gpuMtpHnormBuf!, 0, embBytes);
        _gpu.RecordBarrier();

        // 5. eh_proj @ concat → _gpuHidden.
        GpuMatMul(_gpuHidden, _gpuMtpEhProj!, _gpuMtpConcatBuf!);
        _gpu.RecordBarrier();

        // 6. Residual + attn_norm.
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuMtpAttnNorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();

        // 7. MTP attention block (writes _gpuHidden).
        GpuMtpAttnBlock(position);
        _gpu.RecordBarrier();

        // 8. Residual add.
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        // 9. Residual + post_attention_norm.
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuMtpPostAttnNorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();

        // 10. FFN — dense (27B-MTP on GPU) or MoE (35B-A3B-MTP via CPU MoE). Both write _gpuHidden;
        //     the MoE path closes/reopens the session like the trunk CpuMoeFfn.
        if (_mtpIsMoE)
        {
            CpuMtpMoeFfn();
        }
        else
        {
            GpuMatMul(_gpuFfnGateBufDense!, _gpuMtpFfnGate!, _gpuNormBuf);
            GpuMatMul(_gpuFfnUpBufDense!,   _gpuMtpFfnUp!,   _gpuNormBuf);
            _gpu.RecordBarrier();
            _gpu.SiLuMul(_gpuFfnGateBufDense!, _gpuFfnUpBufDense!);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuHidden, _gpuMtpFfnDown!, _gpuFfnGateBufDense!);
        }
        _gpu.RecordBarrier();

        // 11. Residual add.
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        // 11b. Capture the MTP block's residual output BEFORE the in-place shared-head norm (issue
        //      #30 chained drafting). Device copy now; host download after the logits submit.
        CopyGpuBuffer(_gpuMtpSelfHiddenDev!, _gpuHidden);
        _gpu.RecordBarrier();

        // 12. shared_head_norm (NOT the main output_norm) → output.weight (shared lm_head).
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuMtpSharedHeadNorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden);
        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        // 13. Download the captured self-hidden into the host _mtpSelfHidden (issue #30) via the
        //     dedicated pinned buffer in its own session.
        _gpu.BeginRecord();
        CopyGpuBuffer(_pinnedMtpHidden!, _gpuMtpSelfHiddenDev!);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();
        {
            float* p = _gpu.MapPinned(_pinnedMtpHidden!);
            new ReadOnlySpan<float>(p, _embDim).CopyTo(new Span<float>(_mtpSelfHidden, _embDim));
            _gpu.UnmapPinned(_pinnedMtpHidden!);
        }

        return _logitsBuf;
    }

    /// <summary>MTP attention block on GPU. Mirrors <see cref="GpuAttnBlock"/> but uses the MTP
    /// head's per-head norm + projection weights + its own KV cache. Reuses the trunk attention
    /// scratch (_gpuQGate/_gpuQ/_gpuGate/_gpuK/_gpuV/_gpuAttnOut). Writes _gpuHidden.
    /// Mirror CUDA :4265-4314.</summary>
    private void GpuMtpAttnBlock(int position)
    {
        int kvDim = _numKvHeads * _headDim;
        var mtpCache = _mtpKvCache!;
        var kCache = _gpuMtpKCache!;
        var vCache = _gpuMtpVCache!;

        GpuMatMul(_gpuQGate, _gpuMtpWQGate!, _gpuNormBuf);
        GpuMatMul(_gpuK,     _gpuMtpWK!,     _gpuNormBuf);
        GpuMatMul(_gpuV,     _gpuMtpWV!,     _gpuNormBuf);
        _gpu.RecordBarrier();

        // De-interleave Q‖gate per head (arg order q, g, qg).
        _gpu.SplitQG(_gpuQ, _gpuGate, _gpuQGate, _numHeads, _headDim);
        _gpu.RecordBarrier();

        // Per-head Q/K RMSNorm BEFORE RoPE.
        _gpu.HeadNorm(_gpuQ, _gpuMtpQNorm!, (uint)_numHeads,   (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.HeadNorm(_gpuK, _gpuMtpKNorm!, (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        _gpu.RecordBarrier();

        // Partial NEOX RoPE on the first ropeDim of each head.
        _gpu.RoPEPartial(_gpuQ, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RoPEPartial(_gpuK, position, _headDim, _ropeDim, _hp.RopeTheta, neox: true);
        _gpu.RecordBarrier();

        // Layer-0 invariant: reserve a block before appending at a new page boundary.
        mtpCache.ReserveBlock();
        int kvPosition = mtpCache.Length;
        _gpu.KvAppend(_gpuK, _gpuV, kCache, vCache, (uint)kvDim, (uint)kvPosition, (uint)_maxSeqLen);
        _gpu.RecordBarrier();

        int seqLen = kvPosition + 1;
        _gpu.Attention(_gpuQ, kCache, vCache, _gpuAttnOut, _gpuAttnScratch,
            (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim, (uint)seqLen, (uint)_maxSeqLen, window: 0u);
        _gpu.RecordBarrier();

        // Fused sigmoid GLU gate.
        _gpu.SigmoidMulInPlace(_gpuAttnOut, _gpuGate);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuMtpWO!, _gpuAttnOut);

        mtpCache.IncrementPosition();
    }

    /// <summary>CPU-MoE FFN for the MTP block (mirror of <see cref="CpuMoeFfn"/> with the single MTP
    /// weight set). On entry the session is recording; on return it is recording again. Writes _gpuHidden.</summary>
    private void CpuMtpMoeFfn()
    {
        int numExperts = _numExperts;
        int numActive = _numActiveExperts;
        int expertDim = _expertDim;

        // 1. Shared expert on GPU, in-session (UNSCALED; the sigmoid scalar gate is applied later).
        GpuMatMul(_gpuFfnGate!, _gpuMtpWGateShexp!, _gpuNormBuf);
        GpuMatMul(_gpuFfnUp!,   _gpuMtpWUpShexp!,   _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.SiLuMul(_gpuFfnGate!, _gpuFfnUp!);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuSharedOut!, _gpuMtpWDownShexp!, _gpuFfnGate!);
        _gpu.RecordBarrier();

        // 2. Copy the post-RmsNorm hidden → pinned, host-barrier, submit so the CPU can read it.
        CopyGpuBuffer(_pinnedNorm!, _gpuNormBuf);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();

        float* normPtr = _gpu.MapPinned(_pinnedNorm!);
        new ReadOnlySpan<float>(normPtr, _embDim).CopyTo(new Span<float>(_cpuNormBuf, _embDim));
        _gpu.UnmapPinned(_pinnedNorm!);

        // 3. Shared-expert scalar gate = sigmoid(ffn_gate_inp_shexp · norm).
        float shexpDot = SimdKernels.DotF32(_cpuMtpFfnGateInpShexp, _cpuNormBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));

        // 4. Router: F32 [embDim, numExperts] MatVec → softmax → top-K.
        var routerW = _cpuMtpFfnGateInp;
        SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, _cpuNormBuf,
            numExperts, _embDim, routerW.DType);
        SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopKPtr(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        // 5. Routed experts (sparse top-K): two batched Parallel.For sweeps (mirror CpuMoeFfn).
        var gateExps = _cpuMtpFfnGateExps;
        var upExps   = _cpuMtpFfnUpExps;
        var downExps = _cpuMtpFfnDownExps;

        int bprG = (_embDim   / DTypeInfo.BlockSize(gateExps.DType)) * DTypeInfo.BytesPerBlock(gateExps.DType);
        int bprU = (_embDim   / DTypeInfo.BlockSize(upExps.DType))   * DTypeInfo.BytesPerBlock(upExps.DType);
        int bprD = (expertDim / DTypeInfo.BlockSize(downExps.DType)) * DTypeInfo.BytesPerBlock(downExps.DType);

        int* sePtr = stackalloc int[numActive];
        float* ewPtr = stackalloc float[numActive];
        for (int i = 0; i < numActive; i++) { sePtr[i] = selectedExperts[i]; ewPtr[i] = expertWeights[i]; }

        byte* gateP = gateExps.DataPtr; byte* upP = upExps.DataPtr; byte* downP = downExps.DataPtr;
        DType gateDt = gateExps.DType, upDt = upExps.DType, downDt = downExps.DType;
        float* gateAll = _cpuExpertGateAll, upAll = _cpuExpertUpAll;
        float* normBuf = _cpuNormBuf, moeOut = _cpuMoeHidden;
        int embDimL = _embDim, expertDimL = expertDim, numActiveL = numActive;
        int bprGL = bprG, bprUL = bprU, bprDL = bprD;

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

        SimdKernels.SiLuMul(_cpuExpertGateAll, _cpuExpertUpAll, numActive * expertDim);

        Parallel.For(0, embDimL, s_moeParallelOpts, r =>
        {
            float sum = 0f;
            for (int k = 0; k < numActiveL; k++)
            {
                int expertIdx = sePtr[k];
                float w = ewPtr[k];
                long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                sum += w * DispatchDot(downP + offD, gateAll + (long)k * expertDimL, expertDimL, downDt);
            }
            moeOut[r] = sum;
        });

        // 6. Reopen the session: upload the routed accumulator → _gpuHidden, then combine the scaled
        //    GPU shared expert.
        float* outPtr = _gpu.MapPinned(_pinnedHidden);
        new ReadOnlySpan<float>(_cpuMoeHidden, _embDim).CopyTo(new Span<float>(outPtr, _embDim));
        _gpu.UnmapPinned(_pinnedHidden);

        _gpu.BeginRecord();
        CopyGpuBuffer(_gpuHidden, _pinnedHidden);
        _gpu.RecordBarrier();
        _gpu.ScaleInPlace(_gpuSharedOut!, shexpScale);
        _gpu.RecordBarrier();
        _gpu.AddInPlace(_gpuHidden, _gpuSharedOut!);
        _gpu.RecordBarrier();
    }

    /// <inheritdoc />
    public void MtpResetCache()
    {
        if (!_hasMtp) return;
        _mtpKvCache?.Reset();
        _gpu.BeginRecord();
        if (_gpuMtpKCache is { } kT) _gpu.Clear(kT);
        if (_gpuMtpVCache is { } vT) _gpu.Clear(vT);
        _gpu.EndRecordAndSubmit();
    }

    /// <inheritdoc />
    public void MtpTruncateTo(int length)
    {
        if (!_hasMtp) return;
        if (length == 0) { MtpResetCache(); return; }
        // Soft truncate — the device _gpuMtpKCache is a flat ring, so future KvAppends overwrite
        // stale slots; only the bookkeeping length rewinds (mirror CUDA :4326-4335).
        _mtpKvCache?.TruncateTo(length);
    }

    /// <inheritdoc />
    /// <remarks>Walks the prompt and calls <see cref="MtpForward"/> at each position to populate the
    /// GPU MTP KV cache. prevHidden h_{startPos+i-1} is read from the absolute-position hidden history
    /// populated by the preceding Prefill/Forward/BatchVerify sweeps (mirror CUDA :4348-4383).</remarks>
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

        float* zeroHidden = startPos == 0
            ? (float*)NativeMemory.AllocZeroed((nuint)((long)_embDim * sizeof(float)))
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

    /// <summary>Grow the MTP hidden-history buffer to hold at least <paramref name="requiredTokens"/>
    /// rows (grow-by-doubling, realloc + copy). Mirror CUDA :2967-2983.</summary>
    private void EnsureMtpHiddenHistoryCap(int requiredTokens)
    {
        if (_mtpPrefillHiddensCap >= requiredTokens) return;
        int newCap = Math.Max(requiredTokens, _mtpPrefillHiddensCap * 2);
        long oldBytes = (long)_mtpHiddenHistoryLength * _embDim * sizeof(float);
        float* fresh = (float*)NativeMemory.Alloc((nuint)((long)newCap * _embDim * sizeof(float)));
        if (_mtpPrefillHiddens != null)
        {
            if (oldBytes > 0)
                NativeMemory.Copy(_mtpPrefillHiddens, fresh, (nuint)oldBytes);
            NativeMemory.Free(_mtpPrefillHiddens);
        }
        _mtpPrefillHiddens = fresh;
        _mtpPrefillHiddensCap = newCap;
    }

    // ================================================================
    //  FFN dispatch (dense only; mirror :3206-3224 + HybridForwardPass.cs:507-566).
    //  On entry the session is recording; on return it is recording again.
    // ================================================================

    private void FfnDispatch(int layer)
    {
        // Mirror CudaHybridGdnForwardPass.Forward :3206-3241: dense → per-layer GPU/CPU
        // placement; MoE → CPU-MoE (headline 12 GB path) or GPU-SLRU MoE (≥24 GB).
        if (_hp.IsMoE)
        {
            if (_cpuMoe) CpuMoeFfn(layer);
            else GpuMoeFfn(layer);
            return;
        }

        if (_gpuWFfnGate is not null && _gpuWFfnGate[layer] is not null)
        {
            // Dense GPU layer (in-session; HybridForwardPass.cs:1350-1360).
            GpuDenseFfn(layer);
            return;
        }

        // Dense CPU layer: copy norm → pinned, host-barrier, submit; run CPU FFN; reopen
        // the session and copy back. Mirrors CpuDenseFfnAt :4618-4632 +
        // HybridForwardPass.cs:507-511 / 530-566.
        CopyGpuBuffer(_pinnedHidden, _gpuNormBuf);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();

        float* normPtr = _gpu.MapPinned(_pinnedHidden);
        new ReadOnlySpan<float>(normPtr, _embDim).CopyTo(new Span<float>(_cpuNormBuf, _embDim));
        _gpu.UnmapPinned(_pinnedHidden);

        CpuDenseFfn(layer); // _cpuNormBuf → _cpuMoeHidden

        float* outPtr = _gpu.MapPinned(_pinnedHidden);
        new ReadOnlySpan<float>(_cpuMoeHidden, _embDim).CopyTo(new Span<float>(outPtr, _embDim));
        _gpu.UnmapPinned(_pinnedHidden);

        _gpu.BeginRecord();
        CopyGpuBuffer(_gpuHidden, _pinnedHidden);
        _gpu.RecordBarrier();
    }

    /// <summary>GPU dense FFN for a layer uploaded by TryUploadDenseFfnLayers (HybridForwardPass.cs:1350-1360).</summary>
    private void GpuDenseFfn(int layer)
    {
        var wGate = _gpuWFfnGate![layer]!;
        var wUp   = _gpuWFfnUp![layer]!;
        var wDown = _gpuWFfnDown![layer]!;

        GpuMatMul(_gpuFfnGateBufDense!, wGate, _gpuNormBuf);
        GpuMatMul(_gpuFfnUpBufDense!,   wUp,   _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.SiLuMul(_gpuFfnGateBufDense!, _gpuFfnUpBufDense!);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuHidden, wDown, _gpuFfnGateBufDense!);
    }

    /// <summary>Single-token CPU dense FFN: _cpuNormBuf → _cpuMoeHidden (mirror CpuDenseFfnAt :4618-4632).</summary>
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

    // ================================================================
    //  CPU-MoE FFN (SHARPI_CPU_MOE=1 / auto on small VRAM) — the headline
    //  12 GB path. Mirror of CudaHybridGdnForwardPass.CpuMoeFfnCore :5049-5214,
    //  adapted to the Vulkan record/submit session model: the shared expert runs
    //  on the GPU in-session and the routed experts + router run on the CPU after a
    //  session break, exactly like the dense-CPU-FFN boundary in FfnDispatch.
    //  On entry the session is recording; on return it is recording again.
    // ================================================================

    private void CpuMoeFfn(int layer)
    {
        int numExperts = _numExperts;
        int numActive = _numActiveExperts;
        int expertDim = _expertDim;

        // 1. Shared expert on GPU, in-session (gate/up → SiLuMul → down → _gpuSharedOut).
        //    UNSCALED here; the sigmoid scalar gate is applied after the CPU computes it.
        GpuMatMul(_gpuFfnGate!, _gpuWGateShexp[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp!,   _gpuWUpShexp[layer],   _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.SiLuMul(_gpuFfnGate!, _gpuFfnUp!);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuSharedOut!, _gpuWDownShexp[layer], _gpuFfnGate!);
        _gpu.RecordBarrier();

        // 2. Copy the post-RmsNorm hidden → pinned, host-barrier, submit so the CPU can
        //    read it (router + routed experts + shexp scalar gate dot).
        CopyGpuBuffer(_pinnedNorm!, _gpuNormBuf);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();

        float* normPtr = _gpu.MapPinned(_pinnedNorm!);
        new ReadOnlySpan<float>(normPtr, _embDim).CopyTo(new Span<float>(_cpuNormBuf, _embDim));
        _gpu.UnmapPinned(_pinnedNorm!);

        // 3. Shared-expert scalar gate = sigmoid(ffn_gate_inp_shexp · norm) (mirror :5075-5076).
        float shexpDot = SimdKernels.DotF32(_cpuFfnGateInpShexp![layer], _cpuNormBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));

        // 4. Router: F32 [embDim, numExperts] MatVec → softmax → top-K (mirror :5081-5088).
        var routerW = _cpuFfnGateInp![layer];
        SimdKernels.MatVec(_cpuRouterLogits, routerW.DataPtr, _cpuNormBuf,
            numExperts, _embDim, routerW.DType);
        SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopKPtr(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        // 5. Routed experts (sparse top-K): two batched Parallel.For sweeps (mirror :5090-5208).
        var gateExps = _cpuFfnGateExps![layer];
        var upExps   = _cpuFfnUpExps![layer];
        var downExps = _cpuFfnDownExps![layer];

        int bprG = (_embDim   / DTypeInfo.BlockSize(gateExps.DType)) * DTypeInfo.BytesPerBlock(gateExps.DType);
        int bprU = (_embDim   / DTypeInfo.BlockSize(upExps.DType))   * DTypeInfo.BytesPerBlock(upExps.DType);
        int bprD = (expertDim / DTypeInfo.BlockSize(downExps.DType)) * DTypeInfo.BytesPerBlock(downExps.DType);

        int* sePtr = stackalloc int[numActive];
        float* ewPtr = stackalloc float[numActive];
        for (int i = 0; i < numActive; i++) { sePtr[i] = selectedExperts[i]; ewPtr[i] = expertWeights[i]; }

        byte* gateP = gateExps.DataPtr; byte* upP = upExps.DataPtr; byte* downP = downExps.DataPtr;
        DType gateDt = gateExps.DType, upDt = upExps.DType, downDt = downExps.DType;
        float* gateAll = _cpuExpertGateAll, upAll = _cpuExpertUpAll;
        float* normBuf = _cpuNormBuf, moeOut = _cpuMoeHidden;
        int embDimL = _embDim, expertDimL = expertDim, numActiveL = numActive;
        int bprGL = bprG, bprUL = bprU, bprDL = bprD;

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

        // Phase B: fused SiLuMul over (numActive × expertDim) contiguous floats.
        SimdKernels.SiLuMul(_cpuExpertGateAll, _cpuExpertUpAll, numActive * expertDim);

        // Phase C: down × weight, fused across all experts into one sweep over embDim rows.
        Parallel.For(0, embDimL, s_moeParallelOpts, r =>
        {
            float sum = 0f;
            for (int k = 0; k < numActiveL; k++)
            {
                int expertIdx = sePtr[k];
                float w = ewPtr[k];
                long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                sum += w * DispatchDot(downP + offD, gateAll + (long)k * expertDimL, expertDimL, downDt);
            }
            moeOut[r] = sum;
        });

        // 6. Reopen the session: upload the routed accumulator → _gpuHidden, then combine the
        //    GPU shared expert (scaled by the CPU-computed sigmoid gate). Barriers per the
        //    Round-1 lesson (every compute read-after-write needs an explicit barrier).
        float* outPtr = _gpu.MapPinned(_pinnedHidden);
        new ReadOnlySpan<float>(_cpuMoeHidden, _embDim).CopyTo(new Span<float>(outPtr, _embDim));
        _gpu.UnmapPinned(_pinnedHidden);

        _gpu.BeginRecord();
        CopyGpuBuffer(_gpuHidden, _pinnedHidden);     // _gpuHidden = routed accumulator
        _gpu.RecordBarrier();
        _gpu.ScaleInPlace(_gpuSharedOut!, shexpScale); // shared expert × sigmoid gate
        _gpu.RecordBarrier();
        _gpu.AddInPlace(_gpuHidden, _gpuSharedOut!);   // + shared expert
        _gpu.RecordBarrier();
    }

    // ================================================================
    //  GPU-SLRU MoE FFN (≥24 GB cards; auto-disabled on the 12 GB test card).
    //  Mirror of HybridForwardPass.GpuMoeFfn :1362-1503 (router readback + SLRU +
    //  CPU fallback + combine), extended with the qwen35moe shared-expert SCALAR
    //  gate (sigmoid(ffn_gate_inp_shexp · norm)) that the OLMoE-style HybridForwardPass
    //  lacks — see CudaHybridGdnForwardPass.GpuMoeFfn :4552-4604.
    //  On entry the session is recording; on return it is recording again.
    // ================================================================

    private void GpuMoeFfn(int layer)
    {
        int numActive = _numActiveExperts;

        // 1. Router on GPU → softmax (qwen35moe uses softmax, not sigmoid gating).
        GpuMatMul(_gpuRouterLogits!, _gpuWGateInp[layer], _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.Softmax(_gpuRouterLogits!);
        _gpu.RecordBarrier();

        // Copy norm → pinned (host-coherent) so the CPU can read it for the shexp scalar
        // gate dot and the CPU fallback. Compute→host barrier flushes shader writes.
        CopyGpuBuffer(_pinnedNorm!, _gpuNormBuf);
        _gpu.RecordComputeToHostBarrier();
        _gpu.EndRecordAndSubmit();
        _gpu.Download(_gpuRouterLogits!, _gpuRouterBuf!);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_gpuRouterBuf!, numActive, selectedExperts, expertWeights, _hp.NormalizeMoeTopKWeights);

        // Promote current-layer experts FIRST (TryGetCached promotes hits), then prefetch.
        Span<bool> isGpu = stackalloc bool[numActive];
        ExpertGpuSlot[] cachedSlots = new ExpertGpuSlot[numActive];
        bool hasCpuFallback = false;
        for (int i = 0; i < numActive; i++)
        {
            isGpu[i] = _expertSlotManager!.TryGetCached(layer, selectedExperts[i], out cachedSlots[i]);
            if (!isGpu[i]) hasCpuFallback = true;
        }
        _prefetcher?.EnqueuePrefetch(layer, selectedExperts);

        // 2. Shared-expert scalar gate = sigmoid(ffn_gate_inp_shexp · norm) on the CPU.
        //    Download the small shexp-gate weight into _cpuNormBuf scratch and dot it against
        //    the mapped norm (mirror CudaHybridGdnForwardPass.GpuMoeFfn :4580-4584).
        float* normPtr = _gpu.MapPinned(_pinnedNorm!);
        _gpu.Download(_gpuWGateInpShexp[layer], new Span<float>(_cpuNormBuf, _embDim));
        float shexpDot = SimdKernels.DotF32(_cpuNormBuf, normPtr, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));

        // 3. CPU fallback for SLRU misses (runs while the GPU is idle).
        if (hasCpuFallback)
            GpuMoeFfnCpuFallback(layer, selectedExperts, expertWeights, isGpu, numActive, normPtr);
        _gpu.UnmapPinned(_pinnedNorm!);

        _gpu.BeginRecord();

        // 4. Shared expert (UNSCALED), then scale by the sigmoid gate.
        GpuMatMul(_gpuFfnGate!, _gpuWGateShexp[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp!,   _gpuWUpShexp[layer],   _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.SiLuMul(_gpuFfnGate!, _gpuFfnUp!);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuSharedOut!, _gpuWDownShexp[layer], _gpuFfnGate!);
        _gpu.RecordBarrier();
        _gpu.ScaleInPlace(_gpuSharedOut!, shexpScale);
        _gpu.RecordBarrier();

        // 5. Routed experts via SLRU (weighted accumulate into _gpuHidden).
        _gpu.Clear(_gpuHidden);
        _gpu.RecordBarrier();
        for (int i = 0; i < numActive; i++)
        {
            if (!isGpu[i]) continue; // handled by CPU fallback
            float expertWeight = expertWeights[i];
            GpuMatMul(_gpuFfnGate!, cachedSlots[i].Gate, _gpuNormBuf);
            GpuMatMul(_gpuFfnUp!,   cachedSlots[i].Up,   _gpuNormBuf);
            _gpu.RecordBarrier();
            _gpu.SiLuMul(_gpuFfnGate!, _gpuFfnUp!);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuExpertOut!, cachedSlots[i].Down, _gpuFfnGate!);
            _gpu.RecordBarrier();
            _gpu.AddScaledInPlace(_gpuHidden, _gpuExpertOut!, expertWeight);
            _gpu.RecordBarrier();
        }

        // 6. Add CPU-computed routed contributions (if any) via the pinned buffer.
        if (hasCpuFallback)
        {
            float* mapped = _gpu.MapPinned(_pinnedFallback!);
            fixed (float* srcPtr = _cpuFallbackBuf)
                new ReadOnlySpan<float>(srcPtr, _embDim).CopyTo(new Span<float>(mapped, _embDim));
            _gpu.UnmapPinned(_pinnedFallback!);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _pinnedFallback!);
            _gpu.RecordBarrier();
        }

        // 7. Add the scaled shared expert.
        _gpu.AddInPlace(_gpuHidden, _gpuSharedOut!);
        _gpu.RecordBarrier();
    }

    /// <summary>CPU compute for SLRU-missed routed experts (mirror HybridForwardPass.GpuMoeFfnCpuFallback :1505-1544).</summary>
    private void GpuMoeFfnCpuFallback(int layer, ReadOnlySpan<int> selectedExperts,
        ReadOnlySpan<float> expertWeights, ReadOnlySpan<bool> isGpu, int numActive, float* normPtr)
    {
        _cpuFallbackBuf  ??= new float[_embDim];
        _cpuFallbackGate ??= new float[_expertDim];
        _cpuFallbackUp   ??= new float[_expertDim];
        Array.Clear(_cpuFallbackBuf);

        var wGateExps = ResolveCpuWeight($"blk.{layer}.ffn_gate_exps.weight");
        var wUpExps   = ResolveCpuWeight($"blk.{layer}.ffn_up_exps.weight");
        var wDownExps = ResolveCpuWeight($"blk.{layer}.ffn_down_exps.weight");

        int bprG = (_embDim   / DTypeInfo.BlockSize(wGateExps.DType)) * DTypeInfo.BytesPerBlock(wGateExps.DType);
        int bprU = (_embDim   / DTypeInfo.BlockSize(wUpExps.DType))   * DTypeInfo.BytesPerBlock(wUpExps.DType);
        int bprD = (_expertDim / DTypeInfo.BlockSize(wDownExps.DType)) * DTypeInfo.BytesPerBlock(wDownExps.DType);

        fixed (float* fallbackPtr = _cpuFallbackBuf)
        fixed (float* gatePtr = _cpuFallbackGate)
        fixed (float* upPtr = _cpuFallbackUp)
        {
            for (int i = 0; i < numActive; i++)
            {
                if (isGpu[i]) continue;
                int e = selectedExperts[i];
                float weight = expertWeights[i];

                // gate/up = expert row · norm; SiLuMul; down weighted-accumulate.
                for (int r = 0; r < _expertDim; r++)
                {
                    gatePtr[r] = DispatchDot(wGateExps.DataPtr + (long)e * _expertDim * bprG + (long)r * bprG,
                        normPtr, _embDim, wGateExps.DType);
                    upPtr[r] = DispatchDot(wUpExps.DataPtr + (long)e * _expertDim * bprU + (long)r * bprU,
                        normPtr, _embDim, wUpExps.DType);
                }
                SimdKernels.SiLuMul(gatePtr, upPtr, _expertDim);
                for (int r = 0; r < _embDim; r++)
                    fallbackPtr[r] += weight * DispatchDot(
                        wDownExps.DataPtr + (long)e * _embDim * bprD + (long)r * bprD,
                        gatePtr, _expertDim, wDownExps.DType);
            }
        }
    }

    private static readonly ParallelOptions s_moeParallelOpts = new()
    {
        MaxDegreeOfParallelism = Environment.ProcessorCount
    };

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
            _ => throw new NotSupportedException($"Routed expert dtype {dtype} not supported in MoE path"),
        };

    // Top-K from a native logit buffer (mirror CudaHybridGdnForwardPass.SelectTopKPtr :5361).
    private static void SelectTopKPtr(float* logits, int n, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                bool already = false;
                for (int j = 0; j < ki; j++) if (indices[j] == i) { already = true; break; }
                if (!already && logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }
        if (normalize && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0) for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // Top-K from a managed logit span (mirror HybridForwardPass.SelectTopK :1605).
    private static void SelectTopK(ReadOnlySpan<float> logits, int k,
        Span<int> indices, Span<float> weights, bool normalize)
    {
        for (int ki = 0; ki < k; ki++)
        {
            int bestIdx = 0;
            float bestVal = float.MinValue;
            for (int i = 0; i < logits.Length; i++)
            {
                bool already = false;
                for (int j = 0; j < ki; j++) if (indices[j] == i) { already = true; break; }
                if (!already && logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
            }
            indices[ki] = bestIdx;
            weights[ki] = bestVal;
        }
        if (normalize && k > 1)
        {
            float sum = 0;
            for (int i = 0; i < k; i++) sum += weights[i];
            if (sum > 0) for (int i = 0; i < k; i++) weights[i] /= sum;
        }
    }

    // ================================================================
    //  SLRU capacity prediction (mirror :5768-5832; backend-neutral metadata).
    // ================================================================

    private int PredictSlruSlots(int numLayers)
    {
        long perLayerNonMoeBytes = 0;
        perLayerNonMoeBytes += 2L * _embDim * sizeof(float);                     // norms
        perLayerNonMoeBytes += (long)_numExperts * _embDim * sizeof(float);      // router
        perLayerNonMoeBytes += (long)_embDim * sizeof(float);                    // shexp gate inp
        perLayerNonMoeBytes += 3L * _embDim * _expertDim * sizeof(float);        // shared gate/up/down (worst case F32)

        long attnPerLayer =
              (long)_embDim * _numHeads * _headDim * 2 * sizeof(float)           // q (output qDim*2)
            + (long)_embDim * _numKvHeads * _headDim * sizeof(float) * 2         // k + v
            + (long)_embDim * _numHeads * _headDim * sizeof(float)               // o
            + (long)_maxSeqLen * _numKvHeads * _headDim * sizeof(float) * 2;     // kv cache (fp32)

        long gdnPerLayer = 0;
        gdnPerLayer += (long)_gdnConvChannels * _embDim / 256 * 144;  // attn_qkv Q4_K
        gdnPerLayer += (long)_gdnValueDim * _embDim / 256 * 144;      // attn_gate Q4_K
        gdnPerLayer += (long)_embDim * _gdnValueDim / 256 * 144;      // ssm_out Q4_K
        gdnPerLayer += (long)_gdnNumVHeads * _embDim * sizeof(float); // ssm_alpha F32
        gdnPerLayer += (long)_gdnNumVHeads * _embDim * sizeof(float); // ssm_beta F32
        gdnPerLayer += (long)_gdnConvKernel * _gdnConvChannels * sizeof(float);  // conv1d
        gdnPerLayer += (long)_gdnNumVHeads * _gdnHeadDim * _gdnHeadDim * sizeof(float); // scan state
        gdnPerLayer += (long)(_gdnConvKernel - 1) * _gdnConvChannels * sizeof(float);   // conv state

        int attnLayers = 0;
        for (int i = 0; i < numLayers; i++)
            if (_hp.LayerTypes![i] == LayerType.Attention) attnLayers++;
        int gdnLayers = numLayers - attnLayers;

        long total = numLayers * perLayerNonMoeBytes
                   + (long)attnLayers * attnPerLayer
                   + (long)gdnLayers * gdnPerLayer
                   + (long)_hp.VocabSize * _embDim * sizeof(float);   // embedding/output

        long vramTotal = (long)_gpu.VramBytes;
        long remaining = vramTotal - total - (2L << 30);
        long perExpert = EstimatePerExpertBytes();
        if (perExpert <= 0) return 1024;
        return (int)Math.Max(64, remaining / perExpert);
    }

    private long EstimatePerExpertBytes()
    {
        // Sum each role's MAX per-expert footprint over all layers (issue #216) via the same
        // metadata helper the slab uses, so the predicted capacity equals what the slab fits.
        long Max(string role, int rows, int cols) =>
            CudaExpertSlotManager.MaxRoleExpertBytes(_model, _hp.NumLayers, role, rows, cols);
        long bytes =
              Max("ffn_gate_exps", _hp.ExpertIntermediateDim, _hp.EmbeddingDim)
            + Max("ffn_up_exps",   _hp.ExpertIntermediateDim, _hp.EmbeddingDim)
            + Max("ffn_down_exps", _hp.EmbeddingDim,          _hp.ExpertIntermediateDim);
        return bytes > 0 ? bytes : (long)(1.81 * 1024 * 1024);
    }

    /// <summary>Load + dequantize a small F32 tensor into native memory (mirror :5618-5633).</summary>
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

    // ================================================================
    //  Dense FFN-on-GPU upload (mirror :4961-5027; VulkanBackend has no
    //  FreeVramBytes, so budget against VramBytes − _uploadedVramBytes − margin).
    // ================================================================

    private void TryUploadDenseFfnLayers(VulkanBackend gpu, int L)
    {
        var gateInfo = _model.FindTensor("blk.0.ffn_gate.weight");
        var upInfo   = _model.FindTensor("blk.0.ffn_up.weight");
        var downInfo = _model.FindTensor("blk.0.ffn_down.weight");
        if (gateInfo is null || upInfo is null || downInfo is null)
            return;

        long perLayerBytes = gateInfo.Value.ByteSize + upInfo.Value.ByteSize + downInfo.Value.ByteSize;

        // Reserve headroom for the KV cache, GDN state, scratch, and allocator overhead.
        // Default 1 GiB; override with SHARPI_DENSE_FFN_GPU_MARGIN_MB (set 0 to push to the wall).
        long safetyMarginBytes = 1024L * 1024 * 1024;
        var marginOverride = Environment.GetEnvironmentVariable("SHARPI_DENSE_FFN_GPU_MARGIN_MB");
        if (marginOverride is not null && int.TryParse(marginOverride, out int marginMb) && marginMb >= 0)
            safetyMarginBytes = (long)marginMb * 1024 * 1024;

        long budget = (long)gpu.VramBytes - _uploadedVramBytes - safetyMarginBytes;
        if (budget < perLayerBytes)
        {
            Console.Error.WriteLine(
                $"[VulkanHybridGdnForwardPass] Dense FFN-on-GPU: budget {budget / (1024 * 1024)} MiB < per-layer " +
                $"{perLayerBytes / (1024 * 1024)} MiB (VRAM {gpu.VramBytes / (1024 * 1024)} MiB − uploaded " +
                $"{_uploadedVramBytes / (1024 * 1024)} MiB − margin {safetyMarginBytes / (1024 * 1024)} MiB). All FFN stays on CPU.");
            return;
        }
        int canUpload = (int)Math.Min(Math.Min(L, _denseFfnGpuCap), budget / perLayerBytes);

        _gpuWFfnGate = new Tensor?[L];
        _gpuWFfnUp   = new Tensor?[L];
        _gpuWFfnDown = new Tensor?[L];

        _gpuFfnGateBufDense = AllocateTracked(TensorShape.D1(_intermDim));
        _gpuFfnUpBufDense   = AllocateTracked(TensorShape.D1(_intermDim));

        int uploaded = 0;
        for (int i = 0; i < L; i++)
        {
            if (uploaded >= canUpload) break;
            try
            {
                _gpuWFfnGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _gpuWFfnUp[i]   = UploadWeight($"blk.{i}.ffn_up.weight");
                _gpuWFfnDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
                uploaded++;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[VulkanHybridGdnForwardPass] FFN-on-GPU upload aborted at layer {i}: {ex.Message}");
                if (_gpuWFfnGate[i] is { } gT) { gpu.Free(gT); _gpuWFfnGate[i] = null; }
                if (_gpuWFfnUp[i]   is { } uT) { gpu.Free(uT); _gpuWFfnUp[i]   = null; }
                break;
            }
        }
        _denseFfnGpuLayers = uploaded;
        Console.Error.WriteLine(
            $"[VulkanHybridGdnForwardPass] Dense FFN-on-GPU: uploaded {uploaded}/{L} layers ({uploaded * perLayerBytes / (1024 * 1024)} MiB); "
            + $"{L - uploaded} stay on CPU. Cap={_denseFfnGpuCap}/{L} (-g).");
    }

    // ================================================================
    //  Embedding / weight upload + helpers
    // ================================================================

    /// <summary>
    /// Dispatch the per-token embedding lookup based on the on-GPU embedding dtype. Q4_K/Q6_K
    /// have direct-read shaders; everything else was F32-expanded at upload (no EmbedLookupQ5K).
    /// </summary>
    private void EmbedToken(Tensor dst, int token)
    {
        switch (_embDType)
        {
            case DType.Q4_K:
                _gpu.EmbedLookupQ4K(_gpuEmbedding, dst, (uint)token, (uint)_embDim);
                break;
            case DType.Q6_K:
                _gpu.EmbedLookupQ6K(_gpuEmbedding, dst, (uint)token, (uint)_embDim);
                break;
            default:
                _gpu.EmbedLookup(_gpuEmbedding, dst, (uint)token, (uint)_embDim);
                break;
        }
    }

    private void GpuMatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        _gpu.MatMul(output, matrix, vector,
            _gpuWeightDTypes.TryGetValue(matrix.Handle, out var dt) ? dt : DType.Float32);
    }

    // CopyGpuBuffer / CopyGpuBufferRegion: compute-stage copies so the RecordBarrier()
    // compute→compute barriers cover them (HybridForwardPass.cs:1653-1663).
    private void CopyGpuBuffer(Tensor dst, Tensor src) => _gpu.RecordComputeCopy(dst, src);

    private void CopyGpuBufferRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
        => _gpu.RecordComputeCopyRegion(dst, dstOffsetBytes, src, srcOffsetBytes, sizeBytes);

    /// <summary>Allocate a device tensor and add its byte size to the VRAM estimate.</summary>
    private Tensor AllocateTracked(TensorShape shape)
    {
        var t = _gpu.Allocate(shape);
        _uploadedVramBytes += shape.ElementCount * sizeof(float);
        return t;
    }

    /// <summary>Clear a tensor inside its own record/submit bracket (Clear records a dispatch).</summary>
    private void ClearBracketed(Tensor t)
    {
        _gpu.BeginRecord();
        _gpu.Clear(t);
        _gpu.EndRecordAndSubmit();
    }

    private readonly unsafe struct CpuWeightRef
    {
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public CpuWeightRef(GgufTensorInfo info, DType dtype, byte* dataPtr)
        { Info = info; DType = dtype; DataPtr = dataPtr; }
    }

    private CpuWeightRef ResolveCpuWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new CpuWeightRef(info, info.DType, _model.GetTensorDataPtr(info));
    }

    private List<(nint Ptr, long Bytes)> BuildCpuPrefaultRegions()
    {
        var regions = new List<(nint, long)>();
        void Add(CpuWeightRef w)
        {
            if (w.DataPtr != null) regions.Add(((nint)w.DataPtr, w.Info.ByteSize));
        }
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            if (_hp.IsMoE)
            {
                // CPU-MoE: routed experts (the bulk of the per-token read traffic) are
                // CPU-resident. GPU-SLRU experts are uploaded on demand → nothing to fault.
                if (_cpuMoe)
                {
                    Add(_cpuFfnGateExps![i]);
                    Add(_cpuFfnUpExps![i]);
                    Add(_cpuFfnDownExps![i]);
                }
                continue;
            }
            // Dense: only FFN layers that did NOT make it onto the GPU are CPU-resident.
            bool onGpu = _gpuWFfnGate is not null && _gpuWFfnGate[i] is not null;
            if (onGpu) continue;
            Add(_cpuWFfnGate![i]);
            Add(_cpuWFfnUp![i]);
            Add(_cpuWFfnDown![i]);
        }
        return regions;
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
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
            _uploadedVramBytes += (long)floats.Length * sizeof(float);
        }
        else if (info.DType is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Q4_0)
        {
            // Vulkan MatMul dispatches on these quants directly — keep them raw.
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
            _uploadedVramBytes += data.Length;
        }
        else
        {
            // Other dtypes (e.g. Q3_K) — Vulkan matvec has no kernel; dequantize to F32.
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
            _uploadedVramBytes += (long)count * sizeof(float);
        }
        return result;
    }

    private Tensor UploadEmbeddingWeight(string name, out DType embDType)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // Q4_K / Q6_K have direct-read embedding shaders — keep raw. Q5_K (no Vulkan
        // EmbedLookupQ5K) and everything else fall through to F32 expansion.
        if (info.DType is DType.Q4_K or DType.Q6_K)
        {
            var result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
            _uploadedVramBytes += data.Length;
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
        _uploadedVramBytes += (long)count * sizeof(float);
        embDType = DType.Float32;
        return t;
    }

    /// <summary>
    /// GGUF stores conv1d as [channels, kernel] row-major; the GDN conv shader wants
    /// [kernel, channels] (mirror :5663-5686). Transpose on the CPU, upload as F32.
    /// </summary>
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
        var src = new float[count];
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(src);
        else
            Dequantize.ToFloat32(data, src, info.DType, count);

        var transposed = new float[expected];
        for (int k = 0; k < kernel; k++)
            for (int c = 0; c < channels; c++)
                transposed[k * channels + c] = src[c * kernel + k];
        var tensor = _gpu.Upload(transposed, TensorShape.D1(expected), exact: true);
        _gpuWeightDTypes[tensor.Handle] = DType.Float32;
        _uploadedVramBytes += (long)expected * sizeof(float);
        return tensor;
    }

    /// <summary>
    /// Whether the embedding + output weights fit in a single 2 GB GPU storage buffer
    /// (mirror ShouldKeepFixedWeightsOnCpu :5700-5708, inverted: true == GPU-resident OK).
    /// </summary>
    private static bool ShouldKeepFixedWeightsOnGpu(GgufTensorInfo embedding, GgufTensorInfo? output)
    {
        const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;
        // The embedding and output have DIFFERENT on-GPU sizes because they take different
        // upload paths: UploadEmbeddingWeight keeps only Q4_K/Q6_K raw (no EmbedLookupQ5K, so
        // Q5_K and others F32-expand), whereas UploadWeight (output) keeps the full raw-quant
        // set. Estimate each with the size it will ACTUALLY occupy.
        if (EstimateEmbeddingGpuBytes(embedding) > maxStorageBufferBytes) return false;
        if (output is not null && EstimateWeightGpuBytes(output.Value) > maxStorageBufferBytes) return false;
        return true;
    }

    /// <summary>On-GPU size of the embedding table (mirror UploadEmbeddingWeight: Q4_K/Q6_K raw, else F32).</summary>
    private static long EstimateEmbeddingGpuBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType is DType.Q4_K or DType.Q6_K)
            return tensor.ByteSize;
        return (long)tensor.ElementCount * sizeof(float);
    }

    /// <summary>On-GPU size of a matvec weight (mirror UploadWeight: raw-quant set kept raw, else F32).</summary>
    private static long EstimateWeightGpuBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType is DType.Q4_K or DType.Q6_K or DType.Q5_K or DType.Q8_0 or DType.Q4_0)
            return tensor.ByteSize;
        return (long)tensor.ElementCount * sizeof(float);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count * (nuint)sizeof(float));

    // ================================================================
    //  Dispose
    // ================================================================

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // SLRU expert cache + prefetcher (GPU-SLRU MoE only).
        _prefetcher?.Dispose();
        _expertSlotManager?.Dispose();

        // CPU scratch.
        NativeMemory.Free(_cpuNormBuf);
        NativeMemory.Free(_cpuMoeHidden);
        if (_cpuFfnGateBuf != null) NativeMemory.Free(_cpuFfnGateBuf);   // dense only
        if (_cpuFfnUpBuf   != null) NativeMemory.Free(_cpuFfnUpBuf);     // dense only
        if (_cpuRouterLogits   != null) NativeMemory.Free(_cpuRouterLogits);   // CPU-MoE only
        if (_cpuExpertGateAll  != null) NativeMemory.Free(_cpuExpertGateAll);
        if (_cpuExpertUpAll    != null) NativeMemory.Free(_cpuExpertUpAll);
        if (_cpuFfnGateInpShexp is not null)
            for (int i = 0; i < _cpuFfnGateInpShexp.Length; i++)
                if (_cpuFfnGateInpShexp[i] != null) NativeMemory.Free(_cpuFfnGateInpShexp[i]);

        // GPU scratch.
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

        if (_gpuFfnGateBufDense is { } gB) _gpu.Free(gB);
        if (_gpuFfnUpBufDense   is { } uB) _gpu.Free(uB);

        // Batched-prefill scratch (issue #356 PR5b).
        FreeBatchedScratch();

        // MTP batched-verify GDN snapshot ring + [k×vocab] logits scratch (#357 PR2).
        if (_gpuGdnRingScan is { } ringScan) _gpu.Free(ringScan);
        if (_gpuGdnRingConv is { } ringConv) _gpu.Free(ringConv);
        if (_gpuBvLogitsAll is { } bvLogits) _gpu.Free(bvLogits);

        // MTP / NEXTN head (#357 PR3): weights, caches, per-step scratch, host buffers.
        if (_gpuMtpAttnNorm     is { } t01) _gpu.Free(t01);
        if (_gpuMtpWQGate       is { } t02) _gpu.Free(t02);
        if (_gpuMtpWK           is { } t03) _gpu.Free(t03);
        if (_gpuMtpWV           is { } t04) _gpu.Free(t04);
        if (_gpuMtpWO           is { } t05) _gpu.Free(t05);
        if (_gpuMtpQNorm        is { } t06) _gpu.Free(t06);
        if (_gpuMtpKNorm        is { } t07) _gpu.Free(t07);
        if (_gpuMtpPostAttnNorm is { } t08) _gpu.Free(t08);
        if (_gpuMtpFfnGate      is { } t09) _gpu.Free(t09);
        if (_gpuMtpFfnUp        is { } t10) _gpu.Free(t10);
        if (_gpuMtpFfnDown      is { } t11) _gpu.Free(t11);
        if (_gpuMtpWGateShexp   is { } t12) _gpu.Free(t12);
        if (_gpuMtpWUpShexp     is { } t13) _gpu.Free(t13);
        if (_gpuMtpWDownShexp   is { } t14) _gpu.Free(t14);
        if (_gpuMtpEnorm        is { } t15) _gpu.Free(t15);
        if (_gpuMtpHnorm        is { } t16) _gpu.Free(t16);
        if (_gpuMtpSharedHeadNorm is { } t17) _gpu.Free(t17);
        if (_gpuMtpEhProj       is { } t18) _gpu.Free(t18);
        if (_gpuMtpKCache       is { } t19) _gpu.Free(t19);
        if (_gpuMtpVCache       is { } t20) _gpu.Free(t20);
        if (_gpuMtpEmbedBuf     is { } t21) _gpu.Free(t21);
        if (_gpuMtpEnormBuf     is { } t22) _gpu.Free(t22);
        if (_gpuMtpHnormBuf     is { } t23) _gpu.Free(t23);
        if (_gpuMtpConcatBuf    is { } t24) _gpu.Free(t24);
        if (_gpuLastHidden      is { } t25) _gpu.Free(t25);
        if (_gpuMtpSelfHiddenDev is { } t26) _gpu.Free(t26);
        if (_gpuMtpHistDev      is { } t27) _gpu.Free(t27);
        if (_pinnedMtpHidden    is { } t28) _gpu.Free(t28);
        _mtpKvCache?.Dispose();
        if (_lastHidden        != null) NativeMemory.Free(_lastHidden);
        if (_mtpSelfHidden     != null) NativeMemory.Free(_mtpSelfHidden);
        if (_mtpPrefillHiddens != null) NativeMemory.Free(_mtpPrefillHiddens);
        if (_cpuMtpFfnGateInpShexp != null) NativeMemory.Free(_cpuMtpFfnGateInpShexp);

        // MoE GPU scratch.
        if (_gpuRouterLogits is { } rl) _gpu.Free(rl);
        if (_gpuFfnGate      is { } fg) _gpu.Free(fg);
        if (_gpuFfnUp        is { } fu) _gpu.Free(fu);
        if (_gpuExpertOut    is { } eo) _gpu.Free(eo);
        if (_gpuSharedOut    is { } so) _gpu.Free(so);
        if (_pinnedNorm      is { } pn) _gpu.Free(pn);
        if (_pinnedFallback  is { } pf) _gpu.Free(pf);

        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            _gpu.Free(_gpuAttnNorm[i]);
            _gpu.Free(_gpuPostAttnNorm[i]);

            if (_hp.IsMoE)
            {
                // Shared-expert weights are GPU-resident in both MoE modes.
                _gpu.Free(_gpuWGateShexp[i]);
                _gpu.Free(_gpuWUpShexp[i]);
                _gpu.Free(_gpuWDownShexp[i]);
                if (!_cpuMoe)
                {
                    // Router + shexp-gate-inp stay on GPU only for the GPU-SLRU path.
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
                if (_gpuKCache[i] is { } kT) _gpu.Free(kT);
                if (_gpuVCache[i] is { } vT) _gpu.Free(vT);
            }
            else
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
                if (_gpuGdnScanState[i] is { } sT) _gpu.Free(sT);
                if (_gpuGdnConvState[i] is { } cT) _gpu.Free(cT);
            }

            if (_gpuWFfnGate is not null)
            {
                if (_gpuWFfnGate[i]  is { } gT) _gpu.Free(gT);
                if (_gpuWFfnUp![i]   is { } uT) _gpu.Free(uT);
                if (_gpuWFfnDown![i] is { } dT) _gpu.Free(dT);
            }
        }

        _gpu.Free(_gpuEmbedding);
        _gpu.Free(_gpuOutputNorm);
        // _gpuOutputWeight may alias _gpuEmbedding (tied embeddings) — only free distinct.
        if (_gpuOutputWeight.Handle != _gpuEmbedding.Handle)
            _gpu.Free(_gpuOutputWeight);

        _gdnStateCache.Dispose();
    }
}
