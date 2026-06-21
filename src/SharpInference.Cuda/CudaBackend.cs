using System.Collections.Concurrent;
using System.Threading;
using SharpInference.Core;

namespace SharpInference.Cuda;

/// <summary>
/// CUDA/cuBLAS compute backend for DiT SGEMM acceleration.
/// Manages CUDA device memory and dispatches cuBLAS GemmEx kernels.
/// Precision is auto-detected at creation time:
///   sm_90+ (Hopper/H100) + CUDA 12 → fp8 E4M3 (cublasGemmEx fp8 requires sm_90)
///   sm_80+ (Ampere/RTX 30xx) → bf16 inputs, fp32 accumulation (no overflow, 2× smaller than fp32)
///   sm_53+ (Pascal/any CUDA GPU) → fp16 inputs, fp32 accumulation (avoids fp16 accum overflow)
///   fallback → fp32
/// All LLM transformer operations throw NotSupportedException; this backend is DiT-only.
/// </summary>
public sealed unsafe class CudaBackend : IComputeBackend, IImageOpsBackend, IDisposable
{
    /// <summary>
    /// Round <paramref name="byteSize"/> up to the bucket size the buffer pool will
    /// actually allocate (next power-of-two, min 64 bytes). Use this when sizing
    /// budgets that share VRAM with pooled allocations — the pool's round-up can
    /// inflate per-allocation footprint up to ~2× and a budget computed from raw
    /// byte sizes will overshoot real capacity. Bypasses the pool entirely when
    /// callers pass <c>exact: true</c> to <see cref="Upload"/> / <see cref="UploadRaw"/>.
    /// </summary>
    public static nuint RoundUpAllocBytes(nuint byteSize) => GpuBufferPool.RoundUp(byteSize);

    private readonly nint _handle;
    private readonly SgemmPrecision _precision;
    private readonly int _smVersion;
    private readonly int _smCount;   // SM (multiprocessor) count — decode-MMQ tile-size routing (#205)
    private readonly nint _stream;
    private readonly ConcurrentDictionary<nint, (nint devPtr, nuint byteSize)> _devPtrs = new();
    // Issue #111: handles registered by View() — non-owning slices into another
    // tensor's allocation. Free() drops the registration without freeing memory.
    private readonly ConcurrentDictionary<nint, byte> _viewHandles = new();
    private long _nextHandle = 1;

    // Pinned host staging buffer for DMA-capable async H2D/D2H transfers.
    // Grows on demand; never shrinks (amortised over the pipeline lifetime).
    private nint   _pinnedBuf;
    private nuint  _pinnedBufSize;
    private const nuint InitialPinnedSize = 32 * 1024 * 1024; // 32 MB

    // Dedicated upload stream for background H2D transfers (UploadBackground*).
    // Separate from _stream so that prefetch DMA can overlap with the compute
    // stream without flushing it. Lazily created on first background upload —
    // costs nothing for backends that never prefetch experts.
    private nint   _uploadStream;
    // Ring of pinned staging buffers for UploadBackground* (issue #217). Each async upload
    // copies into the next slot's cudaMallocHost'd buffer, DMAs from it on the upload stream,
    // and records a BACKEND-OWNED fence event. A slot is only drained (host wait) when it
    // wraps around AsyncRingSlots uploads later — by which time its DMA has long completed —
    // so the host can queue a whole layer's expert uploads without blocking, and they
    // pipeline on the upload stream while the compute stream runs.
    //
    // The previous single-buffer design reused the CALLER's handle event as the staging-reuse
    // fence, but the slot manager's ReleaseUploadHandle destroys that event; the backend then
    // synchronized a freed event and got cudaErrorDeviceUninitialized (201) the moment the
    // path was driven from the inference / prefetcher threads. A backend-owned per-slot fence
    // is never freed by the caller, so it both fixes that and enables the ring.
    private const int AsyncRingSlots = 32;
    private readonly nint[]  _asyncRingBuf   = new nint[AsyncRingSlots];
    private readonly nuint[] _asyncRingSize  = new nuint[AsyncRingSlots];
    private readonly nint[]  _asyncRingFence = new nint[AsyncRingSlots];
    private long _asyncRingIdx;
    private readonly object _asyncUploadLock = new();

    // Maximum im2col tile buffer size. All row-aligned tile sizes fit within this bound.
    private const long MaxTileBytes = 2560L * 1024 * 1024; // 2.5 GiB — fits all layers in a single tile

    // GPU buffer pool: reuse device allocations by rounded size to avoid cudaMalloc overhead.
    // Each MatQ call (GEMM) does 2 alloc+free cycles; pooling eliminates driver round-trips.
    private readonly GpuBufferPool _pool = new();

    // Handles allocated with exact=true (no pool, no power-of-2 rounding). Tracked so
    // Free() can release them with cudaFree directly instead of returning to the pool
    // (where their unique sizes would create per-tensor buckets that never get reused).
    private readonly ConcurrentDictionary<nint, byte> _exactHandles = new();

    private bool _disposed;

    // ── NVRTC / image-ops state ────────────────────────────────────────────
    private readonly object _kernelInitLock = new();
    private bool   _imageKernelsInitialized;
    private bool   _imageKernelsAvailable;
    private nint   _nvModule;           // CUmodule loaded from compiled PTX
    private nint   _im2colKernel;

    // Persistent GPU buffer for im2col tiles — allocated once to MaxTileBytes (2.5 GiB).
    private nint   _im2colBuf;
    private nuint  _im2colBufSize;

    private nint   _biasAddKernel;
    private nint   _leakyReluKernel;
    private nint   _scaleKernel;
    private nint   _addKernel;
    private nint   _addScaledKernel;
    private nint   _clampKernel;
    private nint   _pshuffleKernel;
    private nint   _punshuffleKernel;
    private nint   _upsample2xKernel;

    // LLM transformer kernels (loaded by the same NVRTC compilation as the image kernels).
    private nint   _rmsNormKernel;
    private nint   _headNormKernel;
    private nint   _headNormPureKernel;
    private nint   _headNormPureBatchedKernel;   // #124: batched k_eq_v V-norm
    private nint   _siluMulKernel;
    private nint   _sigmoidKernel;
    private nint   _softmaxKernel;
    private nint   _ropeInterleavedKernel;
    private nint   _ropeNeoxKernel;
    private nint   _ropeNeoxPartialKernel;
    private nint   _ropeNeoxWithFactorsKernel;
    private nint   _ropeNeoxWithFactorsBatchedKernel;
    private nint   _mulKernel;
    private nint   _sigmoidMulInPlaceKernel;
    private nint   _splitQgKernel;
    private nint   _kvAppendKernel;
    // Issue #27: bf16-store KV cache for hybrid GDN models. Halves cache VRAM
    // vs the fp32 ring; arithmetic still happens in fp32 (the bf16 → fp32
    // promotion is done at the kernel read sites).
    private nint   _kvAppendBf16Kernel;
    private nint   _attentionBf16Kernel;
    // SnapKV (issue #58): per-(query, head) attention scoring + position-gather
    // compaction. Used by CudaHybridGdnForwardPass.Prefill when SHARPI_SNAPKV_BUDGET
    // is set. Bf16 variants compose with the bf16-store KV path.
    private nint   _snapKvScoreKernel;
    private nint   _snapKvScoreBf16Kernel;
    private nint   _kvCompactKernel;
    private nint   _kvCompactBf16Kernel;
    private nint   _embedLookupF32Kernel;
    private nint   _embedLookupQ4KKernel;
    private nint   _embedLookupQ5KKernel;
    private nint   _embedLookupQ6KKernel;   // #124: Q6_K tied embedding (Gemma 4 12B)
    private nint   _embedLookupQ80Kernel;
    private nint   _embedLookupQ80BatchedKernel;
    private nint   _dequantRowsQ80Kernel;   // #247: GPU-side PLE pre-pass (Q8_0 rows → f32)
    private nint   _dequantRowsQ6KKernel;   // #247: GPU-side PLE pre-pass (Q6_K rows → f32)
    private nint   _matvecF32Kernel;
    private nint   _matvecQ4KKernel;
    private nint   _matvecQ4KSoaKernel;     // #156: scale-pre-unpacked SoA decode matvec
    private nint   _q4kRepackSoaKernel;     // #156: one-time Q4_K → SoA repack
    private nint   _q6kRepackSoaKernel;     // #204: one-time Q6_K → SoA repack for decode MMQ
    private nint   _matvecQ5KKernel;
    private nint   _matvecQ6KKernel;
    private nint   _matvecQ6KSoaKernel;     // #204: bit-identical Q6_K decode matvec over the SoA layout
    // Q4_0 matvec (issue #124, Gemma 4 12B QAT): keeps the q4_0 weights packed on
    // the GPU. Without it q4_0 falls to the F32-dequant upload (~4× VRAM — a 7 GB
    // model would need ~28 GB, defeating full offload). 8 rows/block × 32 thr/row.
    private nint   _matvecQ40Kernel;
    // Issue #124: dp4a/Q8_1 decode matvec for Q4_0 (Gemma 4 12B QAT primary weights).
    private nint   _matvecQ40Dp4aKernel;
    // Q8_0 matvec (Phase 0 of the Gemma-4 plan): keeps Q8_0 weights packed on
    // the GPU. Without this, Q8_0 weights would dequant to F32 on upload and
    // blow out VRAM ~2.1×. Geometry mirrors Q5_K/Q6_K (8 rows/block × 32 thr/row).
    private nint   _matvecQ80Kernel;
    // Issue #142: dp4a/Q8_1 decode matvec (quantize activation to int8, __dp4a dot).
    private nint   _matvecQ80Dp4aKernel;
    // Issue #149: SoA-layout dp4a decode matvec + the interleaved→SoA repack kernel.
    private nint   _matvecQ80Dp4aSoaKernel;
    private nint   _q80RepackSoaKernel;
    // Issue #149: SoA variants of the remaining Q8_0 readers (fp32 matvec / GEMM-N / dequant).
    private nint   _matvecQ80SoaKernel;
    private nint   _matvecQ80GemmNSoaKernel;
    private nint   _dequantQ80F16SoaKernel;
    // Issue #43: N=2 (two-input, two-output) variants — read each weight row
    // once and accumulate into two outputs. Used by MTP BatchForward2's
    // on-GPU dense FFN to halve weight-bandwidth cost per output.
    private nint   _matvecF32N2Kernel;
    private nint   _matvecQ4KN2Kernel;
    private nint   _matvecQ4KN2SoaKernel;   // #156: N=2 over scale-pre-unpacked SoA weight
    private nint   _matvecQ5KN2Kernel;
    private nint   _matvecQ6KN2Kernel;
    private nint   _matvecQ6KN2SoaKernel;   // #204: N=2 over the Q6_K SoA layout
    // Issue #111: batched GEMM-N variants — one weight matrix, N input vectors,
    // N output rows in a single launch. Each (row, token) runs the identical
    // per-row reduction as the GEMV so results are bit-identical to N sequential
    // matvecs. Collapses the per-token trunk launches that dominate GDN-hybrid
    // prefill into one launch per projection.
    private nint   _matvecF32GemmNKernel;
    private nint   _matvecQ4KGemmNKernel;
    private nint   _matvecQ4KGemmNSoaKernel;  // #156: GEMM-N over scale-pre-unpacked SoA weight
    private nint   _matvecQ5KGemmNKernel;
    private nint   _matvecQ6KGemmNKernel;
    private nint   _matvecQ6KGemmNSoaKernel;  // #204: GEMM-N over the Q6_K SoA layout
    private nint   _matvecQ80GemmNKernel;
    // Issue #194: weight-stationary small-N batched-decode matvecs — token loop inside
    // the block so each weight read is amortized across the batch. One handle per
    // compile-time batch capacity (CudaWsKernels.Variants: 2/4/8/16), indexed by position.
    private readonly nint[] _matvecF32WsKernels    = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ4KWsKernels    = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ4KWsSoaKernels = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ5KWsKernels    = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ6KWsKernels    = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ80WsKernels    = new nint[CudaWsKernels.Variants.Length];
    private readonly nint[] _matvecQ80WsSoaKernels = new nint[CudaWsKernels.Variants.Length];
    // #201: scale-word Q6_K WS variant + Q4_K decode MMQ (BN=16 int8 mma).
    private readonly nint[] _matvecQ6KWsSwKernels    = new nint[CudaWsKernels.Variants.Length];
    // #204: SoA Q6_K WS variant (bit-identical to both AoS WS variants; used when the
    // Q6_K weight has been repacked to SoA, which is now always for 2-D trunk weights).
    private readonly nint[] _matvecQ6KWsSoaKernels   = new nint[CudaWsKernels.Variants.Length];
    private nint _mmqQ4kSoaActsN16Kernel;
    private nint _mmqQ4kSoaActsN16Bm32Kernel;   // #205: BM=32 tile for grid-starved low-row shapes
    // #204: Q6_K decode MMQ tiles (BN=16 int8 mma, BM=64 default + BM=32 for low-row shapes).
    // RepackQ6KSoa frees the interleaved AoS weight (the SoA buffer is the only copy), so every
    // Q6_K reader — including this decode-MMQ tile — reads the SoA layout in place; there is no
    // AoS-direct decode-MMQ variant any more.
    private nint _mmqQ6kSoaActsN16Kernel;
    private nint _mmqQ6kSoaActsN16Bm32Kernel;
    // #205 kill-switch: SHARPI_DECODE_MMQ_BM32=0 forces the BM=64 decode-MMQ tile for all
    // shapes (BM=32 is default-on for grid-starved low-row shapes; output is bit-identical).
    private readonly bool _decodeMmqBm32Enabled =
        Environment.GetEnvironmentVariable("SHARPI_DECODE_MMQ_BM32") != "0";
    // #204 kill-switch: SHARPI_Q6K_DECODE_MMQ=0 disables the Q6_K decode-MMQ tile (the
    // Q6_K trunk shapes fall back to the bit-exact weight-stationary matvec). Default on.
    private readonly bool _q6kDecodeMmqEnabled =
        Environment.GetEnvironmentVariable("SHARPI_Q6K_DECODE_MMQ") != "0";
    // Issue #141: compute-bound prefill GEMM — dequant Q8_0 weight + convert
    // activations to fp16, then one cublasGemmEx (weight read once per batch).
    private nint   _dequantQ80F16Kernel;
    // Issue #156 Item C / C1: Q4_K weight → fp16 dequant for the same prefill GEMM path.
    private nint   _dequantQ4KF16Kernel;
    private nint   _dequantQ4KF16SoaKernel;  // #156: dequant over scale-pre-unpacked SoA weight
    // Issue #162: Q6_K weight → fp16 dequant. Qwen3-8B-Q4_K_M keeps ~half of ffn_down +
    // attn_v in Q6_K; without this the Q6_K trunk matmuls fell to the per-token GEMM-N
    // matvec (weight re-streamed once/token), the dominant large-N prefill cost.
    private nint   _dequantQ6KF16Kernel;
    private nint   _dequantQ6KF16SoaKernel;  // #204: dequant over the Q6_K SoA layout
    private nint   _dequantQ5KF16Kernel;   // #162: same path for Q5_K_M mixes
    private nint   _dequantQ40F16Kernel;   // #124: Q4_0 weight → fp16 (Gemma 4 12B QAT)
    private nint   _f32ToF16Kernel;
    // Issue #141 (MMQ): int8 tensor-core Q8_0×Q8_1 matmul — weight read once as
    // int8, no fp16 HBM round-trip, m16n8k32 s8 mma. Replaces the dequant→GEMM path.
    private nint   _mmqQ80Kernel;
    // Issue #149: SoA-layout MMQ (quants 16B-aligned, scales separate) — kills the
    // Q8_0 qs 2-byte-misalignment funnelshift tax in the weight load.
    private nint   _mmqQ80SoaKernel;
    // Issue #156 C2: int8 tensor-core Q4_K×Q8_1 matmul — weight read once as int8
    // (nibble-expanded, no fp16 dequant temp), m16n8k32 s8 mma + asymmetric min-bias.
    private nint   _mmqQ4kKernel;
    private nint   _mmqQ4kSoaKernel;        // #156: Q4_K MMQ over the SoA weight
    // Issue #124/#173: int8 tensor-core Q4_0×Q8_1 matmul — weight read once as int8
    // (nibble-expanded raw 0..15, no fp16 dequant temp), m16n8k32 s8 mma + the −8·d_w·
    // (d_a·Σq_a) symmetric centering term. Gemma 4 12B QAT prefill.
    private nint   _mmqQ40Kernel;
    // Issue #124/#173 (mirrors #149): Q4_0 SoA repack + the SoA twins of every reader
    // (MMQ, decode dp4a, fp32 matvec, GEMM-fallback dequant) — aligned loads instead of
    // the qs-misalignment funnelshift. Bit-identical to the AoS readers.
    private nint   _q40RepackSoaKernel;
    private nint   _mmqQ40SoaKernel;
    private nint   _matvecQ40SoaKernel;
    private nint   _matvecQ40Dp4aSoaKernel;
    private nint   _dequantQ40F16SoaKernel;
    // Issue #111: batched trunk elementwise/norm kernels (one launch over N tokens).
    private nint   _rmsNormBatchedKernel;
    private nint   _headNormBatchedKernel;
    private nint   _headNormQkKernel;
    private nint   _headNormQkBatchedKernel;
    private nint   _splitQgBatchedKernel;
    private nint   _ropeNeoxPartialBatchedKernel;
    private nint   _attentionKernel;
    // Gemma 4 (Phase 7): sliding-window attention, tanh-GELU FFN, final-logit
    // softcap. Kernel work only — forward-pass wiring lands in Phase 8.
    private nint   _attentionSwaKernel;
    private nint   _attentionSwaBatchedKernel;
    private nint   _attentionSwaBf16Kernel;        // issue #179
    private nint   _attentionSwaBatchedBf16Kernel; // issue #179
    private nint   _geluTanhMulKernel;
    private nint   _geluTanhMulStridedKernel;
    private nint   _softcapKernel;
    private nint   _argmaxPartialKernel;   // #219 greedy argmax — pass 1 (per-block reduction)
    private nint   _argmaxFinalKernel;     // #219 greedy argmax — pass 2 (reduce partials)
    private nint   _argmaxRowsKernel;      // #219 batched argmax — one block per row (MTP/spec verify)
    private nint   _clearF32Kernel;
    private nint   _quantizeQ81Kernel;
    // Track A (#124/#173): SoA Q8_1 activation producer + the SoA-weight+SoA-activation
    // MMQ twins. Splits the 36-B AoS Q8_1 block into a contiguous int8-quants array and
    // a separate {d,s} array so a token's qs are aligned/contiguous — the substrate the
    // coalesced per-token load (Phase B) reads. Bit-identical to the AoS-activation MMQ.
    private nint   _quantizeQ81SoaKernel;
    private nint   _mmqQ80SoaActsKernel;
    private nint   _mmqQ4kSoaActsKernel;
    private nint   _mmqQ40SoaActsKernel;
    // Track B port: cp.async double-buffered SoA-acts MMQ — streams global→shared off
    // the L1TEX LSU pipe (the 78.6% ceiling). Bit-identical to the scalar SoA-acts kernels.
    private nint   _mmqQ80SoaActsCpaKernel;
    private nint   _mmqQ40SoaActsCpaKernel;
    private nint   _bwBaselineKernel;
    // Issue #129: batched GPU-SLRU MoE host-loop replacements. scale_rows applies a
    // per-row scalar (shared-expert sigmoid gate) over [rows × cols] in one launch;
    // moe_weighted_reduce does the whole Phase-3 top-k weighted scatter-reduce +
    // shared add over all N tokens in one launch (replacing ~N·(na+2) tiny ops).
    private nint   _scaleRowsKernel;
    private nint   _moeWeightedReduceKernel;

    // TurboQuant KV-cache compression kernels.
    private nint   _tqRotateQueryKernel;
    private nint   _tqKvAppendKernel;
    private nint   _tqAttentionKernel;

    // qwen35moe Gated-DeltaNet (GDN) kernels.
    private nint   _siluInplaceKernel;
    private nint   _gdnConv1dDecodeKernel;
    private nint   _gdnL2NormPerHeadKernel;
    private nint   _gdnTileHeadsKernel;
    private nint   _gdnRecurrenceDecodeKernel;

    // Issue #114-B: batched GDN trunk + batched-query SDPA kernels.
    private nint   _gdnConv1dDecodeBatchedKernel;
    private nint   _gdnConv1dStateUpdateBatchedKernel;
    private nint   _gdnConv1dStateCaptureRingKernel;   // #290
    private nint   _gdnL2NormPerHeadBatchedKernel;
    private nint   _gdnTileHeadsBatchedKernel;
    private nint   _gdnRecurrenceScanKernel;
    private nint   _gdnChunkedPrefillKernel;
    private nint   _kvAppendBatchedKernel;
    private nint   _kvAppendBatchedBf16Kernel;
    private nint   _fullSeqAttentionKernel;
    private nint   _fullSeqAttentionBf16Kernel;
    private nint   _fullSeqAttentionGlobalKernel;
    private nint   _fullSeqAttentionGlobalBf16Kernel;

    // Issue #197: ragged-batched decode kernels — one launch covers all N sequences'
    // RoPE / KV-append / single-query attention at per-sequence positions against N
    // distinct per-sequence caches (positions + cache pointer tables ride in by-value
    // struct kernel parameters; see CudaRaggedKernels).
    private nint   _ropeNeoxRaggedKernel;
    private nint   _ropeInterleavedRaggedKernel;
    private nint   _kvAppendRaggedKernel;
    private nint   _kvAppendRaggedBf16Kernel;
    private nint   _kvAppendRaggedQ8Kernel;
    private nint   _attentionRaggedKernel;
    private nint   _attentionRaggedBf16Kernel;
    private nint   _attentionRaggedQ8Kernel;
    private nint   _addBiasRowsKernel;
    // Issue #141 (attention): memory-efficient flash-attention prefill (shared K/V
    // tiles reused across a query tile + online softmax) replacing the scalar
    // full_seq / swa_batched kernels' O(n²) global K/V re-reads.
    private nint   _flashAttnPrefillKernel;

    // Issue #146: single-warp mma.sync m16n8k16 fragment-layout validation harness
    // (de-risks the TC flash-attention fragment packing). Test-only.
    private nint   _mmaTestM16N8K16Kernel;

    // Issue #146: tensor-core flash-attention prefill (QK^T + P·V on the mma cores,
    // shared-memory O). The compute-bound successor to the half2 flash kernel.
    private nint   _flashAttnPrefillTcKernel;

    // Issue #147: multi-warp / d-split TC flash (register-resident O, ~10× the
    // single-warp occupancy). Default TC path when head_dim % 64 == 0.
    private nint   _flashAttnPrefillTc2Kernel;
    private nint   _flashAttnPrefillTc2Bf16Kernel;   // issue #179
    private nint   _flashAttnPrefillTc2Q8Kernel;     // issue #179 (q8_0 KV)

    // Issue #179 (q8_0 KV): block-quantized cache variants of the bf16 KV kernels.
    private nint   _kvAppendQ8Kernel;
    private nint   _kvAppendBatchedQ8Kernel;
    private nint   _attentionQ8Kernel;
    private nint   _attentionSwaQ8Kernel;
    private nint   _attentionSwaBatchedQ8Kernel;
    private nint   _fullSeqAttentionQ8Kernel;
    private nint   _fullSeqAttentionGlobalQ8Kernel;
    // Issue #235 (flash-decoding): split-KV decode attention + LSE combine. Shared
    // across fp32/bf16/q8 via the templated splitkv kernel; combine is dtype-agnostic.
    private nint   _attentionSplitKvKernel;
    private nint   _attentionSplitKvBf16Kernel;
    private nint   _attentionSplitKvQ8Kernel;
    private nint   _attentionCombineKernel;
    // Issue #237 (GQA head-sharing): split-KV variant that processes a KV head's whole
    // query group per block (grid num_kv_heads × n_splits), loading each K/V slice once.
    private nint   _attentionSplitKvGroupedKernel;
    private nint   _attentionSplitKvGroupedBf16Kernel;
    private nint   _attentionSplitKvGroupedQ8Kernel;

    // Grow-only global score scratch for the wave-based >4096 batched-query SDPA
    // (issue #118). Sized W × num_heads × score_stride floats; W is chosen so this
    // stays under a bounded budget. Freed in Dispose.
    private nint   _waveScratchBuf;
    private nuint  _waveScratchBufSize;

    // Persistent Q8_1 scratch for the Q4_K matvec input. Grows on demand and
    // never shrinks. Sized in 36-byte sub-blocks (one block_q8_1 per 32 elements).
    private nint   _q81Buf;
    private nuint  _q81BufSize;
    // Issue #43: second Q8_1 scratch for MatMulN2's second input vector.
    // Same grow-only sizing policy as _q81Buf. Only allocated when MatMulN2
    // dispatches the Q4_K path (sequential MatMul callers never touch it).
    private nint   _q81BufB;
    private nuint  _q81BufBSize;
    // Issue #111: Q8_1 scratch for MatMulBatched's N input vectors (laid out as
    // N contiguous q81 rows). Grow-only, same policy as _q81Buf.
    private nint   _q81BatchBuf;
    private nuint  _q81BatchBufSize;
    // Track A (#124/#173): SoA Q8_1 activation scratch for the prefill MMQ. One
    // allocation split [qs: totalSub*32 B][ds: totalSub*4 B] — contiguous int8 quants
    // then the {d,s} uint32 array. Grow-only; only allocated when ActSoaEnabled routes
    // the prefill through the SoA-activation MMQ kernels.
    private nint   _q81BatchSoaBuf;
    private nuint  _q81BatchSoaBufSize;
    // Issue #141: fp16 scratch for the prefill GEMM — dequantized weight and
    // converted activations. Grow-only; sized to the largest trunk matmul.
    private nint   _gemmWf16Buf;
    private nuint  _gemmWf16Size;
    private nint   _gemmAf16Buf;
    private nuint  _gemmAf16Size;

    // Tracks dtype per tensor handle so MatMul can dispatch to the right matvec variant
    // (Q4_K / Q5_K / Q6_K / F32). Norm/bias weights upload as F32; quantized weight bytes
    // get tagged via UploadRaw.
    private readonly ConcurrentDictionary<nint, DType> _tensorDTypes = new();

    public string Name => $"CUDA GPU (cuBLAS, {_precision})";

    public SgemmPrecision BestSgemmPrecision => _precision;

    public bool SupportsGpuDequant => false;

    /// <summary>
    /// Issue #142: route Q8_0 decode matvecs through the dp4a/Q8_1 kernel
    /// (<see cref="DispatchMatVecQ80Dp4a"/>) instead of the fp32-decode kernel.
    /// Default on (<c>SHARPI_Q80_DP4A</c>); the dp4a path quantizes the activation
    /// to int8 so it is argmax-stable, not bit-exact to the fp32 matvec. Settable so
    /// bit-parity oracles can pin both sides to the fp32 path.
    /// </summary>
    public bool Q80Dp4aEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_Q80_DP4A") != "0";

    /// <summary>
    /// Issue #219: compute the greedy argmax on-device (<see cref="Argmax"/>) so a decode token
    /// downloads 8 bytes (index + value) instead of the full-vocab logits with a blocking stream
    /// sync. Bit-exact with the host scan (<c>Sampler.Greedy</c>) for finite logits. Default on;
    /// <c>SHARPI_GPU_ARGMAX=0</c> forces the legacy full-download path. Settable so parity oracles
    /// can pin both sides.
    /// </summary>
    public bool GpuArgmaxEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_GPU_ARGMAX") != "0";

    /// <summary>
    /// Issue #124: route Q4_0 decode matvecs through the dp4a/Q8_1 kernel
    /// (<see cref="DispatchMatVecQ40Dp4a"/>) instead of the per-element fp32 matvec
    /// (<c>llm_matvec_q4_0</c>). Default on (<c>SHARPI_Q40_DP4A</c>); the dp4a path
    /// quantizes the activation to int8 so it is argmax-stable, not bit-exact to the
    /// fp32 matvec. Settable so bit-parity oracles can pin both sides to the fp32 path.
    /// </summary>
    public bool Q40Dp4aEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_Q40_DP4A") != "0";

    /// <summary>
    /// Track A (#124/#173): when on, the prefill MMQ over a SoA-repacked weight reads
    /// activations from the SoA Q8_1 layout (<c>llm_quantize_q8_1_soa</c> → the
    /// <c>llm_mmq_*_soa_acts</c> kernels) instead of the interleaved 36-B AoS block.
    /// Phase A keeps the same load mapping → bit-identical; it is the substrate the
    /// coalesced per-token load (Phase B) is built on. Default OFF
    /// (<c>SHARPI_ACT_SOA</c>=1 enables) until the e2e A/B win is proven; settable so
    /// parity oracles can A/B both sides. Only takes effect for SoA-weight handles
    /// (AoS-weight prefill is unaffected).
    /// </summary>
    public bool ActSoaEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_ACT_SOA") == "1";

    /// <summary>
    /// Track B (#124/#173): route the Q8_0/Q4_0 prefill MMQ through the cp.async
    /// double-buffered kernels (<c>llm_mmq_{q8_0,q4_0}_soa_acts_cpa</c>) — they stream the
    /// weight+activation quants global→shared off the L1TEX LSU pipe and overlap the next
    /// K-tile's copy with the current mma in hardware. Implies the SoA-activation substrate
    /// for those formats. Bit-identical to the scalar-load SoA-acts MMQ (and to AoS).
    /// <b>Default OFF</b> (<c>SHARPI_ACT_SOA_CPA=1</c> to enable). It is a real
    /// <i>kernel-level</i> win (+10–15% at L2-resident FFN matmul probe shapes) but
    /// <b>e2e-NEUTRAL</b> on the real 48-layer prefill: profiled matmul 74→73 ms with it on,
    /// because the isolated probe is L1TEX-bound (cp.async's regime) while the full prefill
    /// streams 7 GB of weights and is bound elsewhere. Kept opt-in as bit-identical,
    /// arch-guarded (cp.async on sm_80+, scalar fallback below) groundwork. See the project
    /// memory's probe≠e2e analysis before re-enabling by default.
    /// </summary>
    public bool ActSoaCpaEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_ACT_SOA_CPA") == "1";

    /// <summary>
    /// Issue #201: decode the Q6_K WS per-super-block scale/d tail (bytes 192..209)
    /// via five aligned word loads + funnel-shift extracts (<c>llm_matvec_q6k_ws_sw_n*</c>)
    /// instead of 10 dependent byte-gather loads. Same bytes, same chain —
    /// bit-identical (CudaMatMulBatchedWsTests verify against N sequential MatMul
    /// calls); cuts the LSU instruction count and the gather latency the serial walk
    /// stalls on. <c>SHARPI_BATCH_DECODE_WS_V2=0</c> restores the #194 byte-gather
    /// kernel. Settable so tests can A/B both generations on one backend.
    /// </summary>
    public bool WsV2Enabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_WS_V2") != "0";

    /// <summary>Total VRAM on the active CUDA device, in bytes. Queried once at backend creation.</summary>
    public ulong VramBytes
    {
        get
        {
            if (CuBlasInterop.MemGetInfo(out _, out nuint total) == 0)
                return total;
            return 0;
        }
    }

    /// <summary>Currently-free VRAM on the active CUDA device, in bytes. Queries the
    /// driver each call so callers can size allocations against live headroom.</summary>
    public ulong FreeVramBytes
    {
        get
        {
            if (CuBlasInterop.MemGetInfo(out nuint free, out _) == 0)
                return free;
            return 0;
        }
    }

    /// <summary>SM compute capability ×10 (e.g. 86 for sm_86 / RTX 30xx Ampere).</summary>
    public int SmVersion => _smVersion;

    /// <summary>
    /// Measures effective host↔device transfer bandwidth with a pinned 64 MiB
    /// <c>cudaMemcpy</c> probe (issue #183: replaces <c>HardwareProfile</c>'s
    /// VRAM-size-bucket PCIe guess with a real number). Returns the slower of the
    /// H2D and D2H directions in GB/s — streaming-cost estimates shouldn't assume
    /// the faster one. Costs ~100 ms; intended to run once at model load. Returns
    /// 0 on any failure so callers can fall back to a heuristic.
    /// </summary>
    public double MeasurePcieBandwidthGBps()
    {
        nuint Size = 64 * 1024 * 1024;
        nint host = 0, dev = 0;
        try
        {
            if (CuBlasInterop.MallocHost(out host, Size) != 0) return 0;
            if (CuBlasInterop.CudaMalloc(out dev, Size) != 0) return 0;

            // Warm both directions once (the driver lazily maps the pinned range).
            if (CuBlasInterop.CudaMemcpy(dev, host, Size, CuBlasInterop.HostToDevice) != 0) return 0;
            if (CuBlasInterop.CudaMemcpy(host, dev, Size, CuBlasInterop.DeviceToHost) != 0) return 0;

            double h2d = TimeDirection(dev, host, CuBlasInterop.HostToDevice, Size);
            double d2h = TimeDirection(host, dev, CuBlasInterop.DeviceToHost, Size);
            if (h2d <= 0 || d2h <= 0) return 0;
            return Math.Min(h2d, d2h);
        }
        catch
        {
            return 0;
        }
        finally
        {
            if (dev != 0) _ = CuBlasInterop.CudaFree(dev);
            if (host != 0) _ = CuBlasInterop.FreeHost(host);
        }

        static double TimeDirection(nint dst, nint src, int kind, nuint size)
        {
            const int Reps = 3;
            double best = 0;
            for (int i = 0; i < Reps; i++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                if (CuBlasInterop.CudaMemcpy(dst, src, size, kind) != 0) return 0;
                sw.Stop();
                // Synchronous cudaMemcpy returns only after the copy completes, so
                // wall time is the transfer time. Best-of-3 rejects scheduler noise.
                best = Math.Max(best, size / sw.Elapsed.TotalSeconds / 1e9);
            }
            return best;
        }
    }

    /// <summary>
    /// CUDA stream that all kernels and memcpys are enqueued on. Callers that need to
    /// schedule their own async work against the backend pipeline can use this handle —
    /// it is owned by the backend and synchronized by <see cref="Synchronize"/>.
    /// </summary>
    public nint Stream => _stream;

    /// <summary>
    /// Returns the device pointer backing the given tensor. Intended for forward-pass
    /// implementations that need to perform raw cudaMemcpy operations between tensors.
    /// </summary>
    public nint GetTensorDevicePtr(Tensor tensor) => GetDevPtr(tensor);

    /// <summary>
    /// Async device-to-device copy of an entire tensor. Element-count of <paramref name="dst"/>
    /// and <paramref name="src"/> must match; both must be FP32 (the only element type the
    /// LLM forward path uses for scratch buffers).
    /// </summary>
    public void CopyDevice(Tensor dst, Tensor src)
    {
        if (dst.ElementCount != src.ElementCount)
            throw new ArgumentException($"CopyDevice: element-count mismatch ({src.ElementCount} → {dst.ElementCount}).");
        nuint bytes = (nuint)(src.ElementCount * sizeof(float));
        nint srcPtr = GetDevPtr(src);
        nint dstPtr = GetDevPtr(dst);
        int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, srcPtr, bytes, CuBlasInterop.DeviceToDevice, _stream);
        if (r != 0)
            throw new InvalidOperationException($"cudaMemcpyAsync (D2D) failed: {r}");
    }

    /// <summary>
    /// Async device-to-device copy of a sub-region. Offsets and size are measured in bytes.
    /// Used by the TurboQuant path to extract a single FP32 KV row out of the layer ring
    /// buffer for compression.
    /// </summary>
    public void CopyDeviceRegion(Tensor dst, long dstByteOffset,
                                 Tensor src, long srcByteOffset, long sizeBytes)
    {
        nint dstPtr = GetDevPtr(dst) + (nint)dstByteOffset;
        nint srcPtr = GetDevPtr(src) + (nint)srcByteOffset;
        int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, srcPtr, (nuint)sizeBytes,
            CuBlasInterop.DeviceToDevice, _stream);
        if (r != 0)
            throw new InvalidOperationException($"cudaMemcpyAsync (D2D region) failed: {r}");
    }

    private CudaBackend(nint handle, SgemmPrecision precision, int smVersion, int smCount, nint stream,
                        nint pinnedBuf, nuint pinnedBufSize)
    {
        _handle        = handle;
        _precision     = precision;
        _smVersion     = smVersion;
        _smCount       = smCount;
        _stream        = stream;
        _pinnedBuf     = pinnedBuf;
        _pinnedBufSize = pinnedBufSize;
    }

    public static bool IsAvailable()
    {
        try
        {
            int status = CuBlasInterop.Create(out nint h);
            if (status == 0) { CuBlasInterop.Destroy(h); return true; }
            return false;
        }
        catch { return false; }
    }

    /// <summary>Create a CudaBackend, auto-detecting the best supported precision.</summary>
    public static CudaBackend Create() => Create(precision: null);

    /// <summary>
    /// Create a CudaBackend with an explicit precision override.
    /// Useful for benchmarking or testing a specific SGEMM path regardless of device capability.
    /// </summary>
    public static CudaBackend Create(SgemmPrecision forcedPrecision) => Create(precision: forcedPrecision);

    private static CudaBackend Create(SgemmPrecision? precision)
    {
        int status = CuBlasInterop.Create(out nint handle);
        if (status != 0)
            throw new InvalidOperationException($"cublasCreate failed: {status}");

        int smVersion = 0;
        if (CuBlasInterop.DeviceGetAttribute(out int major, CuBlasInterop.CudaDevAttrComputeCapabilityMajor, 0) == 0 &&
            CuBlasInterop.DeviceGetAttribute(out int minor, CuBlasInterop.CudaDevAttrComputeCapabilityMinor, 0) == 0)
            smVersion = major * 10 + minor;

        // SM count drives decode-MMQ tile-size routing (#205): a row tile that yields fewer
        // than ~2 full waves of blocks starves the grid, so those shapes take the BM=32 tile.
        int smCount = 0;
        CuBlasInterop.DeviceGetAttribute(out smCount, CuBlasInterop.CudaDevAttrMultiProcessorCount, 0);

        // Dedicated CUDA stream — all memcpy and GEMM are enqueued on this stream,
        // so cudaStreamSynchronize(stream) waits only for our work (not the whole device).
        if (CuBlasInterop.StreamCreate(out nint stream) != 0)
            stream = nint.Zero; // fall back to default stream

        if (stream != nint.Zero)
            CuBlasInterop.SetStream(handle, stream);

        // Enable TF32 tensor cores for Sgemm on Ampere/Ada (sm_80+).
        // TF32: on sm_80+ (Ampere+), enable TF32 tensor cores for cublasSgemm.
        // TF32 rounds mantissa to 10 bits but uses tensor cores — ~2× faster while
        // numerically close to FP32. No algorithm benchmarking overhead with SetMathMode.
        if (smVersion >= 80)
        {
            int mmr = CuBlasInterop.SetMathMode(handle, CuBlasInterop.CUBLAS_TF32_TENSOR_OP_MATH);
            if (mmr != 0)
                Console.Error.WriteLine($"[CudaBackend] cublasSetMathMode(TF32) returned {mmr} — using default math");
        }

        // Pinned (page-locked) staging buffer for DMA-capable async H2D/D2H transfers.
        CuBlasInterop.MallocHost(out nint pinnedBuf, InitialPinnedSize);

        var resolvedPrecision = precision ?? DetectBestPrecision(smVersion);
        var backend = new CudaBackend(handle, resolvedPrecision, smVersion, smCount, stream, pinnedBuf, InitialPinnedSize);

        // The 2.5 GiB im2col tile buffer is allocated lazily on the first Conv2d call.
        // Pre-allocating it here used to push LLM contexts that estimated max-context based
        // on free VRAM into driver-managed system-memory spillover (kernels then ran at
        // ~30 GB/s PCIe instead of ~500 GB/s HBM, cratering Q4_K matvec to ~20 GB/s).
        return backend;
    }

    private static SgemmPrecision DetectBestPrecision(int sm)
    {
        // SHARPI_CUDA_PRECISION = fp32 | fp16 | bf16 | fp8 — debug override for the
        // cuBLAS GEMM compute type. Bypasses the SM-based auto-detection below.
        // Unrecognised values fall through to auto-detect (no error).
        //
        // When to use: isolating whether an output regression is driven by
        // mantissa precision (use fp32 as the high-precision floor) vs algorithm
        // / kernel divergence. The default bf16 path on Ampere+ matches fp32 for
        // most workloads, but greedy decode is sensitive to single-ulp argmax
        // flips at low-margin steps. If output stays identical at fp32, the
        // regression is not precision-related.
        //
        // Override only affects the cuBLAS path; custom NVRTC kernels (Q4_K /
        // Q5_K matvec, RmsNorm, attention) keep their fp32 accumulators.
        var env = Environment.GetEnvironmentVariable("SHARPI_CUDA_PRECISION");
        if (env is not null)
        {
            switch (env.Trim().ToLowerInvariant())
            {
                case "fp32": return SgemmPrecision.Fp32;
                case "fp16": return SgemmPrecision.Fp16;
                case "bf16": return SgemmPrecision.Bf16;
                case "fp8":
                case "fp8e4m3": return SgemmPrecision.Fp8E4M3;
            }
        }
        // fp8 via cublasGemmEx requires sm_90+ (Hopper). Ada Lovelace (sm_89) only supports
        // fp8 through cublasLt (light), not the standard cublasGemmEx API.
        if (sm >= 90 && IsCuda12OrNewer())
            return SgemmPrecision.Fp8E4M3;
        if (sm >= 80) return SgemmPrecision.Bf16;    // Ampere+ has native bf16
        if (sm >= 53) return SgemmPrecision.Fp16;    // Pascal+ supports fp16 GemmEx
        return SgemmPrecision.Fp32;
    }

    /// <summary>
    /// Returns true when the loaded CUDA runtime is version 12 or newer.
    /// fp8 via cublasGemmEx requires CUDA 12+ (CUDA 11 returns NOT_SUPPORTED or hangs).
    /// Runtime version is encoded as major*1000 + minor*10 (e.g. 12010 = CUDA 12.1).
    /// </summary>
    private static bool IsCuda12OrNewer()
    {
        if (CuBlasInterop.RuntimeGetVersion(out int ver) != 0) return false;
        return ver >= 12000;
    }

    // ── Memory management ─────────────────────────────────────────────────

    public Tensor Allocate(TensorShape shape, DType dtype = DType.Float32, bool exact = false)
    {
        // ByteSize handles both scalar (blockSize 1 → elementCount·elemBytes, identical to
        // the old BytesPerElement product) and block-quantized dtypes (q8_0 KV cache, #179:
        // (count/32)·34). Lets the KV cache allocate as DType.Q8_0 without a special path.
        nuint byteSize  = (nuint)DTypeInfo.ByteSize(shape.ElementCount, dtype);
        // exact=true bypasses the power-of-2 pool rounding. Use for one-shot weight
        // uploads that won't be freed/realloc'd during decode — pooling is pure waste
        // for those, and the power-of-2 rounding can inflate per-allocation footprint
        // by up to 2× (a 17 MiB attn_gate rounds to 32 MiB; ~50 % average waste).
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        return new Tensor(shape, dtype, handle);
    }

    /// <summary>
    /// Registers a non-owning view into <paramref name="parent"/> starting at
    /// <paramref name="elemOffset"/> elements, spanning <paramref name="elemCount"/>
    /// elements (issue #111). The returned <see cref="Tensor"/> shares the parent's
    /// device memory; <see cref="Free"/> on it only drops the handle registration and
    /// never frees the underlying allocation. Used by batched prefill to pass a
    /// per-token slice of a <c>[N×dim]</c> batched buffer to the per-position
    /// recurrence / KV-append / attention kernels without an extra device copy.
    /// Element size is taken from the parent dtype (Float32 for activation buffers).
    /// </summary>
    public Tensor View(Tensor parent, long elemOffset, long elemCount, DType dtype = DType.Float32)
    {
        if (!_devPtrs.TryGetValue(parent.Handle, out var pe))
            throw new InvalidOperationException($"View: parent handle {parent.Handle} not registered.");
        if (elemOffset < 0 || elemCount < 0)
            throw new ArgumentOutOfRangeException(nameof(elemOffset),
                $"View: elemOffset ({elemOffset}) and elemCount ({elemCount}) must be non-negative.");
        int elemBytes = DTypeInfo.BytesPerElement(dtype);
        long byteOffset = elemOffset * elemBytes;
        long byteCount = elemCount * elemBytes;
        if (byteOffset < 0 || byteOffset + byteCount > (long)pe.byteSize)
            throw new ArgumentOutOfRangeException(nameof(elemOffset),
                $"View [{elemOffset}, {elemOffset + elemCount}) (×{elemBytes}B) out of parent bounds ({pe.byteSize}B).");
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (pe.devPtr + (nint)byteOffset, (nuint)byteCount);
        _viewHandles[handle] = 0;
        return new Tensor(TensorShape.D1(elemCount), dtype, handle);
    }

    /// <summary>
    /// Byte-precise sibling of <see cref="View"/> for carving fixed-stride expert slots out
    /// of a preallocated SLRU slab (issue #216). Registers a non-owning view into
    /// <paramref name="parent"/> at raw byte offset <paramref name="byteOffset"/> spanning
    /// <paramref name="byteLength"/> bytes, tagged with <paramref name="dtype"/> and the
    /// caller-supplied <paramref name="shape"/>. Unlike <see cref="View"/> the offset is in
    /// raw bytes — <c>BytesPerElement</c> throws for Q4_K/Q5_K/Q6_K, so element-offset
    /// addressing can't express a quantized slab slot. <see cref="Free"/> on the result drops
    /// the handle registration only; the parent slab owns and frees the device memory.
    /// </summary>
    public Tensor ViewRawBytes(Tensor parent, long byteOffset, long byteLength, TensorShape shape, DType dtype)
    {
        if (!_devPtrs.TryGetValue(parent.Handle, out var pe))
            throw new InvalidOperationException($"ViewRawBytes: parent handle {parent.Handle} not registered.");
        if (byteOffset < 0 || byteLength < 0 || byteOffset + byteLength > (long)pe.byteSize)
            throw new ArgumentOutOfRangeException(nameof(byteOffset),
                $"ViewRawBytes [{byteOffset}, {byteOffset + byteLength}) out of parent bounds ({pe.byteSize}B).");
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (pe.devPtr + (nint)byteOffset, (nuint)byteLength);
        _viewHandles[handle] = 0;
        _tensorDTypes[handle] = dtype;
        return new Tensor(shape, dtype, handle);
    }

    public void Free(Tensor tensor)
    {
        // #149: drop any SoA-layout mark (harmless no-op for non-repacked handles) so
        // the set doesn't grow across model load/free cycles.
        _soaHandles.TryRemove(tensor.Handle, out _);
        _soaQ4kHandles.TryRemove(tensor.Handle, out _);   // #156
        _soaQ6kHandles.TryRemove(tensor.Handle, out _);   // #204 (the repacked Q6_K SoA weight)
        _soaQ40Handles.TryRemove(tensor.Handle, out _);   // #124/#173

        if (_viewHandles.TryRemove(tensor.Handle, out _))
        {
            // Non-owning view: drop the handle registration only; the parent owns
            // the device memory and frees it on its own Free().
            _tensorDTypes.TryRemove(tensor.Handle, out _);
            _devPtrs.TryRemove(tensor.Handle, out _);
            return;
        }
        _tensorDTypes.TryRemove(tensor.Handle, out _);
        if (_pinnedAllocs.Remove(tensor.Handle, out var pinned))
        {
            // Pinned allocations bypass the device pool — cudaMallocHost'd memory
            // must be released via cudaFreeHost rather than returned to the pool.
            _devPtrs.TryRemove(tensor.Handle, out _);
            CuBlasInterop.FreeHost(pinned.Ptr);
            return;
        }
        if (_exactHandles.TryRemove(tensor.Handle, out _))
        {
            // exact=true allocations bypass the pool; free directly so the memory
            // returns to the system / driver pool rather than getting stranded in
            // a per-tensor bucket the pool can't reuse.
            if (_devPtrs.TryRemove(tensor.Handle, out var ex))
                CuBlasInterop.CudaFree(ex.devPtr);
            return;
        }
        if (_devPtrs.TryRemove(tensor.Handle, out var entry))
            _pool.Return(entry.byteSize, entry.devPtr);
    }

    public Tensor Upload(ReadOnlySpan<float> data, TensorShape shape, bool exact = false)
    {
        nuint byteSize  = (nuint)(data.Length * sizeof(float));
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (float* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        return new Tensor(shape, DType.Float32, handle);
    }

    public void Download(Tensor src, Span<float> dst)
    {
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)(dst.Length * sizeof(float));
        fixed (float* d = dst)
            DownloadViaStaging(d, devPtr, byteSize);
    }

    /// <summary>
    /// Copy host floats into an existing device tensor (no allocation).
    /// Mirrors <see cref="Upload"/> but reuses the destination's storage — used by
    /// the hybrid forward pass to push CPU-produced hidden state back into the
    /// GPU pipeline without churning tensor handles per token.
    /// </summary>
    public void UploadInto(Tensor dst, ReadOnlySpan<float> src)
    {
        if ((long)src.Length != dst.ElementCount)
            throw new ArgumentException($"UploadInto: element-count mismatch ({src.Length} → {dst.ElementCount}).");
        nint dstPtr = GetDevPtr(dst);
        nuint byteSize = (nuint)(src.Length * sizeof(float));
        fixed (float* s = src)
            UploadViaStaging(dstPtr, s, byteSize);
    }

    /// <summary>
    /// Copy <paramref name="src"/> raw bytes into an existing device buffer (no allocation).
    /// Byte-typed sibling of <see cref="UploadInto"/>: the destination is sized for a max
    /// capacity but a given call may push fewer bytes (e.g. a short final prefill chunk), so
    /// the only requirement is that <paramref name="src"/> fits within <paramref name="dst"/>'s
    /// allocation. Used by the GPU-side PLE pre-pass (issue #247) to stage gathered packed
    /// quant rows before the on-device dequant.
    /// </summary>
    public void UploadRawInto(Tensor dst, ReadOnlySpan<byte> src)
    {
        if (!_devPtrs.TryGetValue(dst.Handle, out var entry))
            throw new InvalidOperationException($"UploadRawInto: handle {dst.Handle} not registered.");
        if ((nuint)src.Length > entry.byteSize)
            throw new ArgumentException($"UploadRawInto: source ({src.Length} bytes) exceeds destination capacity ({entry.byteSize} bytes).");
        if (src.IsEmpty) return;   // nothing to copy; avoids fixed → null-ptr on an empty span
        fixed (byte* s = src)
            UploadViaStaging(entry.devPtr, s, (nuint)src.Length);
    }

    // ── Direct-pinned Download/Upload overloads (issues #48/#49) ──────────
    //
    // The Span<float> Download/UploadInto overloads above route every transfer
    // through the internal pinned staging buffer (_pinnedBuf) with an extra
    // host-side Buffer.MemoryCopy hop. When the caller already owns a pinned
    // host allocation (cudaMallocHost), that staging hop is wasted work.
    // These overloads cudaMemcpyAsync directly between the caller's pinned
    // buffer and the device, avoiding the round-trip via _pinnedBuf.
    //
    // The `pinnedDst`/`pinnedSrc` argument MUST point at memory allocated via
    // cudaMallocHost (see AllocatePinnedHost) — pageable memory will fall back
    // to a slow sync copy under cudaMemcpyAsync, defeating the optimisation.

    /// <summary>
    /// Download a float-typed device tensor into a caller-owned pinned host buffer
    /// via a single <c>cudaMemcpyAsync</c>, skipping the internal <c>_pinnedBuf</c>
    /// staging hop. Synchronous: blocks the host until the transfer completes.
    /// Caller is responsible for ensuring <paramref name="pinnedDst"/> points at
    /// memory allocated via <see cref="AllocatePinnedHost"/> (cudaMallocHost).
    /// (Issue #48.)
    /// </summary>
    public unsafe void Download(Tensor src, nint pinnedDst, int floatCount)
    {
        if ((long)floatCount > src.ElementCount)
            throw new ArgumentException($"Download: floatCount={floatCount} exceeds src.ElementCount={src.ElementCount}.");
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            int r = CuBlasInterop.CudaMemcpyAsync(pinnedDst, devPtr, byteSize,
                                                  CuBlasInterop.DeviceToHost, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (D2H, pinned) failed: {r}");
            CuBlasInterop.StreamSynchronize(_stream);
            // The sync above drains any prior in-flight async H2D on _pinnedBuf.
            _uploadInFlight = false;
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(pinnedDst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (D2H, pinned) failed: {r}");
        }
    }

    /// <summary>
    /// Like <see cref="Download(Tensor, nint, int)"/> but does NOT sync the stream.
    /// The transfer is queued on <c>_stream</c>; the buffer's contents are valid
    /// only after a subsequent stream sync (e.g. the next Download/Synchronize
    /// call). Used to overlap a hidden-state snapshot D2H with subsequent GPU work
    /// whose D2H sync will drain both. (Issue #49.)
    /// </summary>
    public unsafe void DownloadAsync(Tensor src, nint pinnedDst, int floatCount)
    {
        if ((long)floatCount > src.ElementCount)
            throw new ArgumentException($"DownloadAsync: floatCount={floatCount} exceeds src.ElementCount={src.ElementCount}.");
        nint devPtr = GetDevPtr(src);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            int r = CuBlasInterop.CudaMemcpyAsync(pinnedDst, devPtr, byteSize,
                                                  CuBlasInterop.DeviceToHost, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (D2H, pinned async) failed: {r}");
            // No sync — caller is responsible for draining via a subsequent
            // Download/Synchronize. _uploadInFlight is NOT cleared here.
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(pinnedDst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (D2H, pinned async fallback) failed: {r}");
        }
    }

    /// <summary>
    /// Upload from a caller-owned pinned host buffer to the device via a single
    /// <c>cudaMemcpyAsync</c> on <c>_stream</c>, skipping the internal
    /// <c>_pinnedBuf</c> staging hop. Subsequent enqueued GPU operations on
    /// <c>_stream</c> automatically sequence behind this transfer; no host-side
    /// sync is needed unless the caller mutates the pinned buffer before the next
    /// stream sync. Caller is responsible for ensuring <paramref name="pinnedSrc"/>
    /// points at memory allocated via <see cref="AllocatePinnedHost"/>
    /// (cudaMallocHost). (Issue #48.)
    /// </summary>
    public unsafe void UploadInto(Tensor dst, nint pinnedSrc, int floatCount)
    {
        if ((long)floatCount != dst.ElementCount)
            throw new ArgumentException($"UploadInto: element-count mismatch ({floatCount} → {dst.ElementCount}).");
        nint dstPtr = GetDevPtr(dst);
        nuint byteSize = (nuint)floatCount * sizeof(float);
        if (_stream != nint.Zero)
        {
            // No need to drain _uploadInFlight: this transfer reads from the
            // caller's pinned buffer, not _pinnedBuf, so it can't race with a
            // prior async H2D that was using _pinnedBuf as its source.
            int r = CuBlasInterop.CudaMemcpyAsync(dstPtr, pinnedSrc, byteSize,
                                                  CuBlasInterop.HostToDevice, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (H2D, pinned) failed: {r}");
        }
        else
        {
            int r = CuBlasInterop.CudaMemcpy(dstPtr, pinnedSrc, byteSize, CuBlasInterop.HostToDevice);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpy (H2D, pinned) failed: {r}");
        }
    }

    /// <summary>
    /// Allocate a pinned host buffer via <c>cudaMallocHost</c>. Returns
    /// <see cref="IntPtr.Zero"/> on failure. Used by Engine code to allocate
    /// per-token scratch suitable for the direct-pinned
    /// <see cref="Download(Tensor, nint, int)"/> /
    /// <see cref="UploadInto(Tensor, nint, int)"/> overloads.
    /// </summary>
    public static nint AllocatePinnedHost(nuint byteSize)
    {
        if (CuBlasInterop.MallocHost(out nint ptr, byteSize) != 0)
            return nint.Zero;
        return ptr;
    }

    /// <summary>Free a pinned host buffer allocated via <see cref="AllocatePinnedHost"/>.</summary>
    public static void FreePinnedHost(nint ptr)
    {
        if (ptr != nint.Zero)
            CuBlasInterop.FreeHost(ptr);
    }

    // ── Vulkan-API shims (used by CudaHybridForwardPass) ──────────────────
    //
    // CUDA executes ops immediately on the stream and orders dependent work
    // implicitly, so Vulkan's command-buffer recording vocabulary degenerates
    // to either no-ops or a Synchronize. These shims let HybridForwardPass-shape
    // code call the same names on either backend.
    public void BeginRecord() { }
    public void EndRecordAndSubmit() => Synchronize();
    public void RecordBarrier() { }
    public void RecordComputeToHostBarrier() { }
    public void RecordComputeToTransferBarrier() { }
    public void RecordComputeCopy(Tensor dst, Tensor src) => CopyDevice(dst, src);
    public void RecordComputeCopyRegion(Tensor dst, long dstByteOffset,
                                        Tensor src, long srcByteOffset, long sizeBytes)
        => CopyDeviceRegion(dst, dstByteOffset, src, srcByteOffset, sizeBytes);

    // The Vulkan staging-buffer-on-fence pattern collapses to a stream-ordered
    // Download in CUDA. We defer the actual copy until ReadFromStaging so the
    // call sequence still reads Record→Submit→Read at the call site, even
    // though no separate staging buffer is needed under CUDA.
    private Tensor? _stagingPendingSrc;
    private int _stagingPendingCount;
    public void RecordDownloadToStaging(Tensor src, int floatCount)
    {
        _stagingPendingSrc = src;
        _stagingPendingCount = floatCount;
    }
    public void ReadFromStaging(Span<float> dst)
    {
        if (_stagingPendingSrc is not { } src)
            throw new InvalidOperationException(
                "ReadFromStaging called without a prior RecordDownloadToStaging.");
        if (dst.Length < _stagingPendingCount)
            throw new ArgumentException(
                $"ReadFromStaging dst too small: {dst.Length} < {_stagingPendingCount}.");
        Download(src, dst[.._stagingPendingCount]);
        _stagingPendingSrc = null;
    }

    // ── CUDA Graphs (issue #136) ──────────────────────────────────────────
    // Capture the launch-bound Gemma 4 decode region once, then replay it per token
    // with only the position-derived kernel-node params rewritten. The forward passes
    // bracket a PURE on-device-compute region (no H2D/D2H — those need a stream sync,
    // illegal during capture; also no cudaMalloc/cudaFree, so any pooled scratch the
    // captured kernels touch — e.g. the Q8_1 matvec buffer — must already be at its max
    // size before capture. The supported Q8_0 Gemma 4 decode has no Q8_1 quantize and the
    // Q4_K buffer is pre-grown during prefill, so this holds; a capture that does hit an
    // illegal op errors the stream and degrades to direct launches) with
    // TryBeginGraphCapture / TryEndGraphCaptureAndInstantiate
    // and replay via LaunchGraphForPosition. The five position-varying ops
    // (RoPE / RoPEWithFactors / KvAppend / Attention / AttentionSwa) self-register their
    // graph node during capture so replay can update just those scalars. Any failure
    // flips _graphCaptureSupported off and the caller falls back to direct launches.
    private bool _graphCaptureSupported = true;
    private bool _graphCapturing;
    private bool _graphCaptureFailed;
    private nint _capturedGraph;   // CUgraph — kept alive: node handles below belong to it
    private nint _graphExec;       // CUgraphExec
    private readonly List<GraphPosNode> _graphPosNodes = new();
    // Upper bound on a tracked kernel's arg count — sizes the reusable stackalloc cell/ptr
    // buffers in LaunchGraphForPosition and bounds the per-node snapshot in TrackPositionNode.
    // The widest position-varying op (AttentionSwa) has 12 args; 16 leaves headroom.
    private const int GraphMaxKernelArgs = 16;

    /// <summary>True while a graph capture is in progress on <see cref="Stream"/>.</summary>
    public bool GraphCapturing => _graphCapturing;
    /// <summary>True until graph capture is ruled out (e.g. a driver capture error).</summary>
    public bool GraphCaptureSupported => _graphCaptureSupported;
    /// <summary>True once a graph is captured + instantiated and ready to replay.</summary>
    public bool GraphReady => _graphExec != nint.Zero;

    private enum GraphPosKind { Position, PositionPlus1, SwaWindowStart, SwaWindowEnd }

    // One captured kernel node whose params carry per-token-varying position scalars.
    private sealed class GraphPosNode
    {
        public nint Node;
        public nint Func;
        public uint Gx, Gy, Gz, Bx, By, Bz, Sh;
        public nint[] ArgValues = [];                         // snapshot of every kernel-arg cell
        public (int Slot, GraphPosKind Kind, int Window)[] Updates = [];
    }

    // Pack a float's bit pattern into an arg cell (the kernel reads its low 4 bytes).
    private static nint GraphFloatBits(float f) => (nint)(uint)BitConverter.SingleToInt32Bits(f);

    /// <summary>
    /// Begin capturing the decode region into a fresh graph. Drains pending async
    /// transfers first so capture starts from a clean stream. Returns false (and
    /// disables graphs) if the driver rejects capture.
    /// </summary>
    public bool TryBeginGraphCapture()
    {
        if (!_graphCaptureSupported || _graphCapturing) return false;
        Synchronize();                 // drain in-flight H2D/D2H before capture starts
        DiscardGraph();
        _graphPosNodes.Clear();
        _graphCaptureFailed = false;
        int rc = NvrtcInterop.StreamBeginCapture(_stream, NvrtcInterop.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
        if (rc != 0) { _graphCaptureSupported = false; return false; }
        _graphCapturing = true;
        return true;
    }

    /// <summary>
    /// End the in-progress capture and instantiate it into a replayable exec graph.
    /// Returns false (disabling graphs) on any capture/instantiate error or if a tracked
    /// node failed to harvest mid-capture.
    /// </summary>
    public bool TryEndGraphCaptureAndInstantiate()
    {
        if (!_graphCapturing) return false;
        int rc = NvrtcInterop.StreamEndCapture(_stream, out nint graph);
        _graphCapturing = false;
        if (rc != 0 || graph == nint.Zero || _graphCaptureFailed)
        {
            if (graph != nint.Zero) NvrtcInterop.GraphDestroy(graph);
            _graphPosNodes.Clear();
            _graphCaptureSupported = false;
            return false;
        }
        rc = NvrtcInterop.GraphInstantiate(out nint exec, graph, 0);
        if (rc != 0 || exec == nint.Zero)
        {
            NvrtcInterop.GraphDestroy(graph);
            _graphPosNodes.Clear();
            _graphCaptureSupported = false;
            return false;
        }
        _capturedGraph = graph;
        _graphExec = exec;
        return true;
    }

    /// <summary>
    /// Abort an in-progress capture (drains the stream out of capture mode) and give up
    /// on graphs for this backend. Safe to call when not capturing.
    /// </summary>
    public void AbortGraphCapture()
    {
        if (_graphCapturing)
        {
            NvrtcInterop.StreamEndCapture(_stream, out nint g);
            _graphCapturing = false;
            if (g != nint.Zero) NvrtcInterop.GraphDestroy(g);
        }
        // A failure can also reach here *after* TryEndGraphCaptureAndInstantiate already
        // built _graphExec/_capturedGraph (e.g. the first LaunchGraphForPosition threw):
        // at that point _graphCapturing is already false, so free those handles too —
        // otherwise "abort" leaks an exec graph and leaves GraphReady stuck true.
        DiscardGraph();
        _graphPosNodes.Clear();
        _graphCaptureSupported = false;
    }

    /// <summary>
    /// Replay the captured decode graph for <paramref name="position"/>: rewrite each
    /// tracked node's position-derived scalar params, then launch the exec graph on the
    /// compute stream. Bit-identical to a direct-launch decode at the same position.
    /// </summary>
    public void LaunchGraphForPosition(int position)
    {
        if (_graphExec == nint.Zero)
            throw new InvalidOperationException("LaunchGraphForPosition called before a graph was captured.");

        nint* cells = stackalloc nint[GraphMaxKernelArgs];
        nint* ptrs  = stackalloc nint[GraphMaxKernelArgs];
        foreach (var n in _graphPosNodes)
        {
            int cnt = n.ArgValues.Length;
            for (int i = 0; i < cnt; i++) cells[i] = n.ArgValues[i];
            foreach (var u in n.Updates)
                cells[u.Slot] = u.Kind switch
                {
                    GraphPosKind.Position       => position,
                    GraphPosKind.PositionPlus1  => position + 1,
                    GraphPosKind.SwaWindowStart => Math.Max(0, position + 1 - u.Window),
                    GraphPosKind.SwaWindowEnd   => position + 1,
                    _                           => cells[u.Slot],
                };
            for (int i = 0; i < cnt; i++) ptrs[i] = (nint)(cells + i);

            var p = new NvrtcInterop.CudaKernelNodeParams
            {
                Func = n.Func,
                GridDimX = n.Gx, GridDimY = n.Gy, GridDimZ = n.Gz,
                BlockDimX = n.Bx, BlockDimY = n.By, BlockDimZ = n.Bz,
                SharedMemBytes = n.Sh,
                KernelParams = (nint)ptrs,
                Extra = nint.Zero,
            };
            int rc = NvrtcInterop.GraphExecKernelNodeSetParams(_graphExec, n.Node, &p);
            if (rc != 0)
                throw new InvalidOperationException($"cuGraphExecKernelNodeSetParams failed: {rc}");
        }

        int lr = NvrtcInterop.GraphLaunch(_graphExec, _stream);
        if (lr != 0)
            throw new InvalidOperationException($"cuGraphLaunch failed: {lr}");
    }

    private void DiscardGraph()
    {
        if (_graphExec != nint.Zero) { NvrtcInterop.GraphExecDestroy(_graphExec); _graphExec = nint.Zero; }
        if (_capturedGraph != nint.Zero) { NvrtcInterop.GraphDestroy(_capturedGraph); _capturedGraph = nint.Zero; }
    }

    // Called by the position-varying op methods immediately after their cuLaunchKernel
    // while a capture is active. Harvests the just-added graph node from the capture-info
    // leaf set (a linearly-ordered stream has exactly one leaf) and snapshots its kernel-arg
    // values so the position slots can be rewritten per replay. Marks the capture failed
    // (→ fallback) if the leaf set isn't the expected single node.
    private void TrackPositionNode(
        nint func, uint gx, uint gy, uint gz, uint bx, uint by, uint bz, uint sh,
        ReadOnlySpan<nint> argValues,
        (int Slot, GraphPosKind Kind, int Window)[] updates)
    {
        if (_graphCaptureFailed) return;
        int rc = NvrtcInterop.StreamGetCaptureInfo(
            _stream, out int status, out _, out _, out nint deps, out nuint numDeps);
        if (rc != 0 || status != NvrtcInterop.CU_STREAM_CAPTURE_STATUS_ACTIVE
                    || numDeps != 1 || deps == nint.Zero || argValues.Length > GraphMaxKernelArgs)
        {
            _graphCaptureFailed = true;
            return;
        }
        _graphPosNodes.Add(new GraphPosNode
        {
            Node = ((nint*)deps)[0], Func = func,
            Gx = gx, Gy = gy, Gz = gz, Bx = bx, By = by, Bz = bz, Sh = sh,
            ArgValues = argValues.ToArray(),
            Updates = updates,
        });
    }

    // ── Pinned host memory exposed as device-accessible tensors ──
    // Vulkan exposes host-visible-device-local buffers as a single Tensor that
    // both the GPU shaders and CPU mapped pointer can use. Under UVA, CUDA's
    // cudaMallocHost gives the equivalent: one pointer addressable from both
    // host and device. We track the (host = device) pointer per handle so
    // MapPinned/Free can find it without a separate dictionary.
    private readonly Dictionary<nint, (nint Ptr, nuint Bytes)> _pinnedAllocs = new();

    public Tensor AllocatePinned(TensorShape shape, DType dtype = DType.Float32)
    {
        long elemBytes = DTypeInfo.BytesPerElement(dtype);
        nuint byteSize = (nuint)(shape.ElementCount * elemBytes);
        if (CuBlasInterop.MallocHost(out nint hostPtr, byteSize) != 0)
            throw new InvalidOperationException($"cudaMallocHost({byteSize}) failed.");
        // Zero the buffer so initial reads see deterministic values.
        new Span<byte>((void*)hostPtr, (int)byteSize).Clear();

        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        // UVA: the host pointer is also a valid device pointer for kernel launches.
        _devPtrs[handle] = (hostPtr, byteSize);
        _pinnedAllocs[handle] = (hostPtr, byteSize);
        _tensorDTypes[handle] = dtype;
        return new Tensor(shape, dtype, handle);
    }

    public float* MapPinned(Tensor tensor)
    {
        if (!_pinnedAllocs.TryGetValue(tensor.Handle, out var alloc))
            throw new InvalidOperationException($"Tensor {tensor.Handle} was not allocated via AllocatePinned.");
        return (float*)alloc.Ptr;
    }

    public void UnmapPinned(Tensor tensor)
    {
        // No-op under CUDA: cudaMallocHost memory stays mapped for its lifetime.
        _ = tensor;
    }

    public Tensor UploadHalf(ReadOnlySpan<Half> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)(data.Length * 2);
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (Half* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.Float16, handle);
    }

    public void DownloadHalf(Tensor src, Span<Half> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (Half* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)(dst.Length * 2));
    }

    public Tensor UploadBf16(ReadOnlySpan<ushort> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)(data.Length * 2);
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (ushort* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.BFloat16, handle);
    }

    public void DownloadBf16(Tensor src, Span<ushort> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (ushort* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)(dst.Length * 2));
    }

    public Tensor UploadFp8(ReadOnlySpan<byte> data, TensorShape shape)
    {
        nuint byteSize  = (nuint)data.Length;
        nuint allocSize = GpuBufferPool.RoundUp(byteSize);
        nint devPtr = _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (byte* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        return new Tensor(shape, DType.Float8E4M3, handle);
    }

    public void DownloadFp8(Tensor src, Span<byte> dst)
    {
        nint devPtr = GetDevPtr(src);
        fixed (byte* d = dst)
            DownloadViaStaging(d, devPtr, (nuint)dst.Length);
    }

    public Tensor UploadRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype, bool exact = false)
    {
        nuint byteSize  = (nuint)data.Length;
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        fixed (byte* src = data)
            UploadViaStaging(devPtr, src, byteSize);
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        var tensor = new Tensor(shape, dtype, handle);
        _tensorDTypes[handle] = dtype;
        return tensor;
    }

    /// <summary>Allocate an uninitialized device buffer of <paramref name="bytes"/> bytes
    /// (issue #149 SoA repack destination). Mirrors <see cref="UploadRaw"/>'s allocation
    /// without the host→device copy. The shape is a 1-D byte count for bookkeeping.</summary>
    public Tensor AllocateRawBytes(long bytes, DType dtype, bool exact = false)
    {
        nuint byteSize  = (nuint)bytes;
        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }
        var handle = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handle] = (devPtr, allocSize);
        if (exact) _exactHandles[handle] = 0;
        _tensorDTypes[handle] = dtype;
        return new Tensor(TensorShape.D1(bytes), dtype, handle);
    }

    // ── Async background upload path (issue #78) ──────────────────────────
    //
    // Mirrors the Vulkan UploadBackground entry point: dispatches the host→device
    // copy on a dedicated upload stream so the predictive MoE prefetcher can
    // start moving the next layer's experts over PCIe while the compute stream
    // is still finishing the current layer's matmuls.
    //
    // The returned CudaUploadHandle owns a CUDA event recorded at the end of
    // the DMA. Consumers MUST call WaitForUpload on the compute stream (or any
    // other CUDA stream that reads the tensor) before launching dependent work,
    // otherwise the read can race ahead of the DMA. After WaitForUpload, the
    // tensor behaves identically to one returned from the synchronous Upload* —
    // it lives in the same _devPtrs table and is freed via Free() like any
    // other tensor.
    //
    // Concurrent UploadBackground* calls are serialized by _asyncUploadLock while
    // they pick a staging-ring slot and record events. They do NOT serialize on the
    // prior DMA: each call uses its own ring slot and only drains (host wait) when a
    // slot wraps around AsyncRingSlots uploads later, so up to AsyncRingSlots transfers
    // can be in flight at once (see the _asyncRing* fields).

    /// <summary>
    /// Async H2D upload on the dedicated upload stream. The returned tensor's
    /// readiness is tracked by <see cref="CudaUploadHandle.UploadEvent"/>;
    /// callers MUST invoke <see cref="WaitForUpload"/> on the compute stream
    /// (or whichever stream consumes the tensor) before launching dependent
    /// kernels. The event is owned by the handle and released by
    /// <see cref="ReleaseUploadHandle"/>.
    /// </summary>
    public CudaUploadHandle UploadBackground(ReadOnlySpan<float> data, TensorShape shape, bool exact = false)
    {
        nuint byteSize = (nuint)(data.Length * sizeof(float));
        fixed (float* src = data)
        {
            var (tensor, ev) = UploadBackgroundCore(src, byteSize, shape, DType.Float32, exact);
            return new CudaUploadHandle(tensor, ev);
        }
    }

    /// <summary>
    /// Async H2D upload of raw bytes with explicit dtype tagging on the
    /// dedicated upload stream. Same readiness model as
    /// <see cref="UploadBackground(ReadOnlySpan{float}, TensorShape, bool)"/>.
    /// </summary>
    public CudaUploadHandle UploadBackgroundRaw(ReadOnlySpan<byte> data, TensorShape shape, DType dtype, bool exact = false)
    {
        nuint byteSize = (nuint)data.Length;
        fixed (byte* src = data)
        {
            var (tensor, ev) = UploadBackgroundCore(src, byteSize, shape, dtype, exact);
            _tensorDTypes[tensor.Handle] = dtype;
            return new CudaUploadHandle(tensor, ev);
        }
    }

    /// <summary>
    /// Async H2D upload of raw bytes into an EXISTING device buffer / view (issue #216
    /// SLRU slab slots), on the dedicated upload stream. Allocates nothing: the returned
    /// handle's tensor IS <paramref name="dst"/>. Same readiness model as
    /// <see cref="UploadBackgroundRaw"/> — the caller must <see cref="WaitForUpload"/> (or
    /// poll <see cref="IsUploadComplete"/>) before launching kernels that read
    /// <paramref name="dst"/>, then release the handle via <see cref="ReleaseUploadHandle"/>.
    /// </summary>
    public CudaUploadHandle UploadBackgroundRawInto(Tensor dst, ReadOnlySpan<byte> data)
    {
        if (!_devPtrs.TryGetValue(dst.Handle, out var entry))
            throw new InvalidOperationException($"UploadBackgroundRawInto: handle {dst.Handle} not registered.");
        if ((nuint)data.Length > entry.byteSize)
            throw new ArgumentException($"UploadBackgroundRawInto: source ({data.Length} B) exceeds destination ({entry.byteSize} B).");
        fixed (byte* src = data)
        {
            nint ev = StageAndRecordAsync(entry.devPtr, src, (nuint)data.Length);
            return new CudaUploadHandle(dst, ev);
        }
    }

    /// <summary>
    /// Async H2D upload of floats into an EXISTING device buffer / view. Float-typed
    /// sibling of <see cref="UploadBackgroundRawInto"/> (issue #216 slab slots, F32-dequant
    /// fallback dtypes). Allocates nothing; the returned handle's tensor IS <paramref name="dst"/>.
    /// </summary>
    public CudaUploadHandle UploadBackgroundInto(Tensor dst, ReadOnlySpan<float> data)
    {
        if (!_devPtrs.TryGetValue(dst.Handle, out var entry))
            throw new InvalidOperationException($"UploadBackgroundInto: handle {dst.Handle} not registered.");
        nuint byteSize = (nuint)(data.Length * sizeof(float));
        if (byteSize > entry.byteSize)
            throw new ArgumentException($"UploadBackgroundInto: source ({byteSize} B) exceeds destination ({entry.byteSize} B).");
        fixed (float* src = data)
        {
            nint ev = StageAndRecordAsync(entry.devPtr, src, byteSize);
            return new CudaUploadHandle(dst, ev);
        }
    }

    /// <summary>
    /// Make the compute stream wait for the background upload referenced by
    /// <paramref name="handle"/> to complete before launching any further work.
    /// Cheap if the DMA has already finished. Safe to call from any thread
    /// (cudaStreamWaitEvent is async — it inserts a fence into the stream and
    /// returns immediately, the actual wait happens on the device).
    /// </summary>
    public void WaitForUpload(CudaUploadHandle handle)
    {
        if (handle.UploadEvent == nint.Zero) return;
        EnsureUploadStream();
        if (_stream == nint.Zero)
        {
            // No compute stream → use device-wide sync (extremely rare path).
            int sr = CuBlasInterop.EventSynchronize(handle.UploadEvent);
            if (sr != 0)
                throw new InvalidOperationException($"cudaEventSynchronize failed: {sr}");
            return;
        }
        int r = CuBlasInterop.StreamWaitEvent(_stream, handle.UploadEvent, 0);
        if (r != 0)
            throw new InvalidOperationException($"cudaStreamWaitEvent failed: {r}");
    }

    /// <summary>
    /// Non-blocking poll: returns true if the background upload has completed
    /// (cudaEventQuery == cudaSuccess). Allows callers (e.g. SLRU GetOrLoad)
    /// to skip the WaitForUpload fence when the DMA has already drained.
    /// </summary>
    public bool IsUploadComplete(CudaUploadHandle handle)
    {
        if (handle.UploadEvent == nint.Zero) return true;
        int r = CuBlasInterop.EventQuery(handle.UploadEvent);
        return r == CuBlasInterop.CudaSuccess;
    }

    /// <summary>
    /// Destroy the CUDA event owned by <paramref name="handle"/>. Call once the
    /// tensor's readiness no longer needs to be tracked (typically after
    /// <see cref="WaitForUpload"/> for short-lived prefetches, or never if the
    /// tensor is long-lived — destroying the event is a courtesy that returns a
    /// few tens of bytes of driver state, not a correctness requirement).
    /// Idempotent.
    /// </summary>
    public void ReleaseUploadHandle(CudaUploadHandle handle)
    {
        if (handle.UploadEvent != nint.Zero)
            CuBlasInterop.EventDestroy(handle.UploadEvent);
    }

    private (Tensor tensor, nint ev) UploadBackgroundCore(void* src, nuint byteSize, TensorShape shape, DType dtype, bool exact)
    {
        EnsureUploadStream();

        nuint allocSize = exact ? byteSize : GpuBufferPool.RoundUp(byteSize);
        nint devPtr = exact ? nint.Zero : _pool.Rent(allocSize);
        if (devPtr == nint.Zero)
        {
            int status = CuBlasInterop.CudaMalloc(out devPtr, allocSize);
            if (status != 0)
                throw new InvalidOperationException($"cudaMalloc failed: {status}");
        }

        nint ev = StageAndRecordAsync(devPtr, src, byteSize);

        var handleId = (nint)Interlocked.Increment(ref _nextHandle);
        _devPtrs[handleId] = (devPtr, allocSize);
        if (exact) _exactHandles[handleId] = 0;
        var tensor = new Tensor(shape, dtype, handleId);
        return (tensor, ev);
    }

    /// <summary>
    /// Stage <paramref name="byteSize"/> bytes from <paramref name="src"/> through the
    /// pinned async staging ring and issue a <c>cudaMemcpyAsync</c> into
    /// <paramref name="dstPtr"/> on the upload stream, returning a fresh readiness event
    /// (caller owns it, destroys via <see cref="ReleaseUploadHandle"/>). Shared by the
    /// allocating background-upload path (<see cref="UploadBackgroundCore"/>) and the
    /// upload-into-existing-buffer path (<see cref="UploadBackgroundRawInto"/> /
    /// <see cref="UploadBackgroundInto"/>, issue #216 slab slots).
    /// </summary>
    private nint StageAndRecordAsync(nint dstPtr, void* src, nuint byteSize)
    {
        EnsureUploadStream();
        nint ev;
        lock (_asyncUploadLock)
        {
            // Bitwise-AND (AsyncRingSlots is a power of two) instead of % so the index stays
            // in [0, AsyncRingSlots) even in the impossible event the long counter wraps past
            // 2^63 — a negative `% 32` would yield a negative slot and an IndexOutOfRange.
            int slot = (int)(_asyncRingIdx++ & (AsyncRingSlots - 1));

            // Drain THIS slot's previous DMA (AsyncRingSlots uploads ago) before reusing its
            // staging buffer. The fence is backend-owned — never destroyed by the caller — so
            // unlike the old caller-event reuse this can't synchronize a freed event. In steady
            // state the DMA finished long ago, so this does not block the host.
            if (_asyncRingFence[slot] != nint.Zero)
            {
                int dr = CuBlasInterop.EventSynchronize(_asyncRingFence[slot]);
                if (dr != 0)
                    throw new InvalidOperationException($"cudaEventSynchronize (drain async staging slot) failed: {dr}");
            }

            // Grow this slot's staging buffer if the tensor doesn't fit. Free the old buffer and
            // zero the slot FIRST, so if the MallocHost below fails the slot holds no stale or
            // garbage pointer for Dispose to (double-)free.
            if (_asyncRingBuf[slot] == nint.Zero || byteSize > _asyncRingSize[slot])
            {
                if (_asyncRingBuf[slot] != nint.Zero)
                {
                    CuBlasInterop.FreeHost(_asyncRingBuf[slot]);
                    _asyncRingBuf[slot] = nint.Zero;
                }
                nuint oldSize = _asyncRingSize[slot];
                _asyncRingSize[slot] = 0;
                nuint newSize = Math.Max(byteSize, oldSize * 2);
                if (newSize < 1024 * 1024) newSize = 1024 * 1024;
                int mr = CuBlasInterop.MallocHost(out _asyncRingBuf[slot], newSize);
                if (mr != 0)
                {
                    _asyncRingBuf[slot] = nint.Zero;
                    throw new InvalidOperationException($"cudaMallocHost (async upload staging, {newSize} B) failed: {mr}");
                }
                _asyncRingSize[slot] = newSize;
            }

            Buffer.MemoryCopy(src, (void*)_asyncRingBuf[slot], _asyncRingSize[slot], byteSize);

            int rc = CuBlasInterop.CudaMemcpyAsync(dstPtr, _asyncRingBuf[slot], byteSize,
                CuBlasInterop.HostToDevice, _uploadStream);
            if (rc != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (UploadBackground) failed: {rc}");

            // Caller's readiness event (consumed by WaitForUpload's cross-stream wait / an
            // EventQuery poll, then destroyed by ReleaseUploadHandle). DisableTiming.
            int er = CuBlasInterop.EventCreateWithFlags(out ev, CuBlasInterop.EventDisableTiming);
            if (er != 0)
                throw new InvalidOperationException($"cudaEventCreateWithFlags failed: {er}");
            int rr = CuBlasInterop.EventRecord(ev, _uploadStream);
            if (rr != 0)
            {
                CuBlasInterop.EventDestroy(ev);
                throw new InvalidOperationException($"cudaEventRecord failed: {rr}");
            }

            // Backend-owned staging fence for this slot: signals when this slot's DMA (and
            // thus the host's freedom to overwrite its buffer) completes. Created once per
            // slot, re-recorded on each reuse. On a failure here the caller never receives a
            // handle, so destroy the just-created caller event ev ourselves to avoid leaking it.
            if (_asyncRingFence[slot] == nint.Zero)
            {
                int fe = CuBlasInterop.EventCreateWithFlags(out _asyncRingFence[slot], CuBlasInterop.EventDisableTiming);
                if (fe != 0)
                {
                    CuBlasInterop.EventDestroy(ev);
                    throw new InvalidOperationException($"cudaEventCreateWithFlags (staging fence) failed: {fe}");
                }
            }
            int fr = CuBlasInterop.EventRecord(_asyncRingFence[slot], _uploadStream);
            if (fr != 0)
            {
                CuBlasInterop.EventDestroy(ev);
                throw new InvalidOperationException($"cudaEventRecord (staging fence) failed: {fr}");
            }
        }

        return ev;
    }

    private void EnsureUploadStream()
    {
        if (_uploadStream != nint.Zero) return;
        lock (_asyncUploadLock)
        {
            if (_uploadStream != nint.Zero) return;
            EnsurePrimaryContextCurrent();
            int r = CuBlasInterop.StreamCreate(out nint s);
            if (r != 0)
                throw new InvalidOperationException($"cudaStreamCreate (upload stream) failed: {r}");
            _uploadStream = s;
        }
    }

    /// <summary>
    /// CUDA upload stream — exposed for tests and for callers that need to
    /// schedule additional copies on the same async transfer queue.
    /// <see cref="IntPtr.Zero"/> until the first <see cref="UploadBackground"/>
    /// (or <see cref="UploadBackgroundRaw"/>) call creates it.
    /// </summary>
    public nint UploadStream => _uploadStream;

    public void DequantQ5KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    public void DequantQ4KM(Tensor src, Tensor dst, int numBlocks) =>
        throw new NotSupportedException("CudaBackend does not support GPU dequantization");

    // ── SGEMM ─────────────────────────────────────────────────────────────

    /// <summary>
    /// GEMM: C[M,N] = A[M,K] × B[N,K]^T using cublasGemmEx with fp32 accumulation.
    /// fp16 and bf16 inputs both accumulate in fp32 — prevents the overflow that plagued
    /// pure cublasHgemm (fp16 accum overflows for deep DiT layers with large residuals).
    /// Row-major layout is handled via the column-major transpose identity:
    ///   row-major C=A*B^T  ≡  col-major C^T = B*A^T
    /// </summary>
    public void Sgemm(Tensor C, Tensor A, Tensor B, int M, int K, int N)
    {
        nint aPtr = GetDevPtr(A);
        nint bPtr = GetDevPtr(B);
        nint cPtr = GetDevPtr(C);
        float alpha = 1.0f, beta = 0.0f;

        int cudaTypeA = ToCudaDataType(A.DType);
        int cudaTypeB = ToCudaDataType(B.DType);
        int cudaTypeC = ToCudaDataType(C.DType);

        if (cudaTypeA != CuBlasInterop.CUDA_R_32F || cudaTypeB != CuBlasInterop.CUDA_R_32F)
        {
            // fp16, bf16, or fp8: use GemmEx with fp32 accumulation.
            // fp8 E4M3 requires: both A and B fp8, and C must be bf16 or fp16 (not fp32).
            // fp16/bf16 use fp32 accumulation to avoid overflow on large DiT residuals.
            if (cudaTypeA == CuBlasInterop.CUDA_R_8F_E4M3 && cudaTypeC == CuBlasInterop.CUDA_R_32F)
                throw new InvalidOperationException(
                    "fp8 GemmEx: cuBLAS requires bf16/fp16 output (not fp32) when inputs are fp8. " +
                    "Allocate C as DType.BFloat16 and use DownloadBf16.");

            int status = CuBlasInterop.GemmEx(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                bPtr, cudaTypeB, K,
                aPtr, cudaTypeA, K,
                ref beta,
                cPtr, cudaTypeC, N,
                CuBlasInterop.CUBLAS_COMPUTE_32F,
                CuBlasInterop.CUBLAS_GEMM_DEFAULT);
            if (status != 0)
                throw new InvalidOperationException($"cublasGemmEx failed: {status}");
        }
        else if (_smVersion >= 80)
        {
            // Ampere+ with fp32 inputs: use TF32 tensor cores (~8× vs cublasSgemm).
            // TF32 has 10-bit mantissa (same as bf16) with fp32 range — no accuracy loss for DiT inference.
            int status = CuBlasInterop.GemmEx(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                bPtr, CuBlasInterop.CUDA_R_32F, K,
                aPtr, CuBlasInterop.CUDA_R_32F, K,
                ref beta,
                cPtr, CuBlasInterop.CUDA_R_32F, N,
                CuBlasInterop.CUBLAS_COMPUTE_32F_FAST_TF32,
                CuBlasInterop.CUBLAS_GEMM_DEFAULT);
            if (status != 0)
                throw new InvalidOperationException($"cublasGemmEx (TF32) failed: {status}");
        }
        else
        {
            int status = CuBlasInterop.Sgemm(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha, bPtr, K, aPtr, K,
                ref beta, cPtr, N);
            if (status != 0)
                throw new InvalidOperationException($"cublasSgemm failed: {status}");
        }
    }

    private static int ToCudaDataType(DType dtype) => dtype switch
    {
        DType.Float32    => CuBlasInterop.CUDA_R_32F,
        DType.Float16    => CuBlasInterop.CUDA_R_16F,
        DType.BFloat16   => CuBlasInterop.CUDA_R_16BF,
        DType.Float8E4M3 => CuBlasInterop.CUDA_R_8F_E4M3,
        _ => CuBlasInterop.CUDA_R_32F,
    };

    public void Synchronize()
    {
        int status = _stream != nint.Zero
            ? CuBlasInterop.StreamSynchronize(_stream)
            : CuBlasInterop.DeviceSync();
        if (status != 0)
            throw new InvalidOperationException($"CUDA synchronize failed: {status}");
        _uploadInFlight = false;
    }

    // ── Pinned staging buffer ─────────────────────────────────────────────

    /// <summary>
    /// Ensure the pinned staging buffer is at least <paramref name="required"/> bytes.
    /// The buffer grows but never shrinks (amortised cost over pipeline lifetime).
    /// </summary>
    private unsafe void EnsurePinnedBuf(nuint required)
    {
        if (required <= _pinnedBufSize) return;
        if (_pinnedBuf != nint.Zero) CuBlasInterop.FreeHost(_pinnedBuf);
        nuint newSize = Math.Max(required, _pinnedBufSize * 2);
        if (CuBlasInterop.MallocHost(out _pinnedBuf, newSize) != 0)
        {
            _pinnedBuf = nint.Zero; // allocation failed — fall back to sync copies
            _pinnedBufSize = 0;
            return;
        }
        _pinnedBufSize = newSize;
    }

    // Tracks whether the most recent UploadViaStaging call left an async H2D
    // memcpy in flight reading _pinnedBuf. Cleared on the next stream sync —
    // either explicit (Synchronize / DownloadViaStaging) or implicit (next
    // upload's drain). Any operation that overwrites _pinnedBuf must drain
    // first to avoid corrupting an in-flight DMA. See issue #47.
    private bool _uploadInFlight;

    /// <summary>
    /// Copy <paramref name="src"/> to the device via the pinned staging buffer.
    /// Issues a non-blocking <c>cudaMemcpyAsync</c> on <c>_stream</c> after the
    /// host-side memcpy into the pinned buffer (issue #47). The host returns
    /// immediately; the actual PCIe transfer overlaps with subsequent enqueued
    /// GPU work on the same stream. The next operation that reuses
    /// <c>_pinnedBuf</c> drains the prior async copy first, so the buffer is
    /// never read by the DMA while a new write is in progress.
    /// </summary>
    private unsafe void UploadViaStaging(nint devPtr, void* src, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            // Drain any prior in-flight upload before reusing _pinnedBuf.
            // Downloads call StreamSynchronize themselves; consecutive uploads
            // without an interleaved download are the only path that needs this.
            if (_uploadInFlight)
            {
                CuBlasInterop.StreamSynchronize(_stream);
                _uploadInFlight = false;
            }
            Buffer.MemoryCopy(src, (void*)_pinnedBuf, _pinnedBufSize, byteSize);
            int r = CuBlasInterop.CudaMemcpyAsync(devPtr, _pinnedBuf, byteSize,
                                                  CuBlasInterop.HostToDevice, _stream);
            if (r != 0)
                throw new InvalidOperationException($"cudaMemcpyAsync (H2D) failed: {r}");
            _uploadInFlight = true;
        }
        else
        {
            CuBlasInterop.CudaMemcpy(devPtr, (nint)src, byteSize, CuBlasInterop.HostToDevice);
        }
    }

    /// <summary>
    /// Copy from device to <paramref name="dst"/> via the pinned staging buffer (async DMA).
    /// The implicit <c>StreamSynchronize</c> here also drains any in-flight async upload,
    /// since both ops are queued on <c>_stream</c>.
    /// </summary>
    private unsafe void DownloadViaStaging(void* dst, nint devPtr, nuint byteSize)
    {
        EnsurePinnedBuf(byteSize);
        if (_pinnedBuf != nint.Zero && _stream != nint.Zero)
        {
            CuBlasInterop.CudaMemcpyAsync(_pinnedBuf, devPtr, byteSize,
                                          CuBlasInterop.DeviceToHost, _stream);
            CuBlasInterop.StreamSynchronize(_stream);
            _uploadInFlight = false;
            Buffer.MemoryCopy((void*)_pinnedBuf, dst, byteSize, byteSize);
        }
        else
        {
            CuBlasInterop.CudaMemcpy((nint)dst, devPtr, byteSize, CuBlasInterop.DeviceToHost);
        }
    }

    // ── LLM transformer ops ───────────────────────────────────────────────

    public void MatMul(Tensor output, Tensor matrix, Tensor vector)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMul(output, matrix, vector, dtype);
    }

    /// <summary>
    /// Matrix-vector multiply with explicit weight dtype dispatch.
    ///   • Q4_K → llama.cpp-style int8 / __dp4a path: quantize the input vector
    ///     to Q8_1 once, then dispatch a 1-row-per-block × 4-warp cooperative
    ///     matvec that uses __dp4a for the inner dot products.
    ///   • Q5_K / Q6_K / Q8_0 / F32 → custom fp32 matvec kernels (8 rows / block).
    /// Output is FP32 in every case.
    /// </summary>
    public void MatMul(Tensor output, Tensor matrix, Tensor vector, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");

        int rows = (int)output.ElementCount;
        int cols = (int)vector.ElementCount;
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(vector);
        nint yPtr = GetDevPtr(output);

        if (weightDType == DType.Q4_K)
        {
            if (_soaQ4kHandles.ContainsKey(matrix.Handle))
                DispatchMatVecQ4KSoa(wPtr, xPtr, yPtr, rows, cols);
            else
                DispatchMatVecQ4K(wPtr, xPtr, yPtr, rows, cols);
            return;
        }
        // Issue #142: Q8_0 decode matvec via dp4a/Q8_1 (cols % 32 == 0 — every LLM
        // hidden dim qualifies). Falls back to the fp32-decode kernel otherwise.
        if (weightDType == DType.Q8_0 && Q80Dp4aEnabled && (cols & 31) == 0)
        {
            DispatchMatVecQ80Dp4a(wPtr, xPtr, yPtr, rows, cols, soa: _soaHandles.ContainsKey(matrix.Handle));
            return;
        }
        // Issue #124: Q4_0 decode matvec via dp4a/Q8_1 (cols % 32 == 0). Falls back to
        // the per-element fp32 kernel otherwise. Gemma 4 12B QAT keeps all bulk weights
        // in Q4_0; this is the decode counterpart of the prefill GEMM path.
        if (weightDType == DType.Q4_0 && Q40Dp4aEnabled && (cols & 31) == 0)
        {
            DispatchMatVecQ40Dp4a(wPtr, xPtr, yPtr, rows, cols,
                                  soa: _soaQ40Handles.ContainsKey(matrix.Handle));
            return;
        }

        int  pRows = rows, pCols = cols;
        nint* args = stackalloc nint[5]
        {
            (nint)(&wPtr), (nint)(&xPtr), (nint)(&yPtr),
            (nint)(&pRows), (nint)(&pCols)
        };

        bool soa = weightDType == DType.Q8_0 && _soaHandles.ContainsKey(matrix.Handle);
        bool soaQ40 = weightDType == DType.Q4_0 && _soaQ40Handles.ContainsKey(matrix.Handle);
        bool soaQ6k = weightDType == DType.Q6_K && _soaQ6kHandles.ContainsKey(matrix.Handle);
        nint kernel = weightDType switch
        {
            DType.Q4_0    => soaQ40 ? _matvecQ40SoaKernel : _matvecQ40Kernel,
            DType.Q5_K    => _matvecQ5KKernel,
            DType.Q6_K    => soaQ6k ? _matvecQ6KSoaKernel : _matvecQ6KKernel,
            DType.Q8_0    => soa ? _matvecQ80SoaKernel : _matvecQ80Kernel,
            DType.Float32 => _matvecF32Kernel,
            _ => throw new NotSupportedException($"CUDA MatMul: weight dtype {weightDType} not supported (expected Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, or Float32)."),
        };

        uint grid = (uint)((rows + 7) / 8);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec) failed: {r}");
    }

    /// <summary>
    /// Two-input matrix-vector multiply: produces <paramref name="outputA"/> and
    /// <paramref name="outputB"/> from a single <paramref name="matrix"/> read,
    /// dispatched to a per-dtype custom kernel (<c>llm_matvec_*_n2</c>). The
    /// weight matrix is read once per row, then folded into two independent
    /// dot products — halves the per-output weight bandwidth versus two
    /// sequential <see cref="MatMul"/> calls, mirroring the CPU
    /// <c>SimdKernels.MatVec2In</c> design. Used by MTP batched-verify
    /// (issue #43, follow-up to #30) to amortise on-GPU dense FFN weight
    /// reads across the two draft tokens.
    /// </summary>
    public void MatMulN2(Tensor outputA, Tensor outputB,
                        Tensor matrix,
                        Tensor inputA, Tensor inputB,
                        DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");

        if (outputA.ElementCount != outputB.ElementCount)
            throw new ArgumentException(
                $"MatMulN2: outputA.ElementCount ({outputA.ElementCount}) != outputB.ElementCount ({outputB.ElementCount}).");
        if (inputA.ElementCount != inputB.ElementCount)
            throw new ArgumentException(
                $"MatMulN2: inputA.ElementCount ({inputA.ElementCount}) != inputB.ElementCount ({inputB.ElementCount}).");

        int rows = (int)outputA.ElementCount;
        int cols = (int)inputA.ElementCount;
        nint wPtr  = GetDevPtr(matrix);
        nint xAPtr = GetDevPtr(inputA);
        nint xBPtr = GetDevPtr(inputB);
        nint yAPtr = GetDevPtr(outputA);
        nint yBPtr = GetDevPtr(outputB);

        if (weightDType == DType.Q4_K)
        {
            // #156: the dense MTP batched-verify trunk weights may be repacked SoA;
            // route to the bit-identical SoA N=2 reader (no AoS fallback throw).
            DispatchMatVecQ4KN2(wPtr, xAPtr, xBPtr, yAPtr, yBPtr, rows, cols,
                                soa: _soaQ4kHandles.ContainsKey(matrix.Handle));
            return;
        }

        int  pRows = rows, pCols = cols;
        nint* args = stackalloc nint[7]
        {
            (nint)(&wPtr),
            (nint)(&xAPtr), (nint)(&xBPtr),
            (nint)(&yAPtr), (nint)(&yBPtr),
            (nint)(&pRows), (nint)(&pCols)
        };

        bool soaQ6k = weightDType == DType.Q6_K && _soaQ6kHandles.ContainsKey(matrix.Handle);
        nint kernel = weightDType switch
        {
            DType.Q5_K    => _matvecQ5KN2Kernel,
            DType.Q6_K    => soaQ6k ? _matvecQ6KN2SoaKernel : _matvecQ6KN2Kernel,
            DType.Float32 => _matvecF32N2Kernel,
            _ => throw new NotSupportedException(
                $"CUDA MatMulN2: weight dtype {weightDType} not supported (expected Q4_K, Q5_K, Q6_K, or Float32)."),
        };

        uint grid = (uint)((rows + 7) / 8);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_n2) failed: {r}");
    }

    /// <summary>
    /// Convenience overload: looks up the weight dtype from the
    /// <see cref="UploadRaw"/> tag dictionary, matching the single-input
    /// <see cref="MatMul(Tensor, Tensor, Tensor)"/> behaviour.
    /// </summary>
    public void MatMulN2(Tensor outputA, Tensor outputB,
                        Tensor matrix,
                        Tensor inputA, Tensor inputB)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMulN2(outputA, outputB, matrix, inputA, inputB, dtype);
    }

    /// <summary>
    /// Batched matrix-vector multiply (GEMM-N, issue #111): one <paramref name="matrix"/>
    /// applied to <paramref name="nTok"/> input vectors, producing <paramref name="nTok"/>
    /// output rows — in a single kernel launch. Layout is token-major:
    /// <paramref name="inputAll"/> is <c>[nTok × cols]</c> and <paramref name="outputAll"/>
    /// is <c>[nTok × rows]</c> (token <c>t</c>'s slice starts at <c>t × rows</c>).
    ///
    /// <para><b>Bit-exact:</b> each (row, token) pair runs the identical per-row reduction
    /// as the single-token <see cref="MatMul(Tensor,Tensor,Tensor,DType)"/> — same weight
    /// decode, same dp4a/FMA chain, same warp + shared reduce — so the result is
    /// bit-identical to <paramref name="nTok"/> sequential <see cref="MatMul"/> calls. This
    /// is the property the GDN/MTP byte-parity oracles depend on; do not reorder the
    /// reduction. Only the launch count collapses (N → 1), killing the host launch
    /// overhead that dominates GDN-hybrid prefill.</para>
    ///
    /// Supports Q4_K (via per-token Q8_1 quantize + the GEMM-N dp4a kernel) and Float32.
    /// Other dtypes throw — the GDN-hybrid trunk projections are all Q4_K.
    /// </summary>
    public void MatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll,
                              int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatched: outputAll ({outputAll.ElementCount}) and inputAll ({inputAll.ElementCount}) " +
                $"element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        if (weightDType == DType.Q4_K)
        {
            // #156: the GEMM-N fallback prefill (SHARPI_PREFILL_MMQ=0) over a
            // repacked SoA weight uses the bit-identical SoA GEMM-N reader.
            DispatchMatVecQ4KBatched(wPtr, xPtr, yPtr, rows, cols, nTok,
                                     soa: _soaQ4kHandles.ContainsKey(matrix.Handle));
            return;
        }
        if (weightDType is DType.Float32 or DType.Q6_K or DType.Q5_K or DType.Q8_0)
        {
            // All take F32 input; the Q5_K/Q6_K/Q8_0 kernels decode the weight per
            // element. Same (rows+7)/8 × nTok geometry across all four.
            bool soa = weightDType == DType.Q8_0 && _soaHandles.ContainsKey(matrix.Handle);
            bool soaQ6k = weightDType == DType.Q6_K && _soaQ6kHandles.ContainsKey(matrix.Handle);
            nint kernel = weightDType switch
            {
                DType.Q6_K => soaQ6k ? _matvecQ6KGemmNSoaKernel : _matvecQ6KGemmNKernel,
                DType.Q5_K => _matvecQ5KGemmNKernel,
                DType.Q8_0 => soa ? _matvecQ80GemmNSoaKernel : _matvecQ80GemmNKernel,
                _          => _matvecF32GemmNKernel,
            };
            int pRows = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&xPtr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            uint gridX = (uint)((rows + 7) / 8);
            int r = NvrtcInterop.LaunchKernel(kernel, gridX, (uint)nTok, 1,
                                              256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_gemm_n) failed: {r}");
            return;
        }
        throw new NotSupportedException(
            $"CUDA MatMulBatched: weight dtype {weightDType} not supported (expected Q4_K, Q5_K, Q6_K, Q8_0, or Float32).");
    }

    /// <summary>
    /// Convenience overload: looks up the weight dtype from the
    /// <see cref="UploadRaw"/> tag dictionary.
    /// </summary>
    public void MatMulBatched(Tensor outputAll, Tensor matrix, Tensor inputAll, int nTok)
    {
        var dtype = _tensorDTypes.GetValueOrDefault(matrix.Handle, matrix.DType);
        MatMulBatched(outputAll, matrix, inputAll, nTok, dtype);
    }

    /// <summary>
    /// Weight-stationary batched matvec for small-N decode (issue #194). Same contract
    /// and token-major layout as <see cref="MatMulBatched(Tensor,Tensor,Tensor,int,DType)"/>,
    /// but the token loop runs <b>inside</b> the thread block: each weight element is read
    /// from HBM once and applied to all <paramref name="nTok"/> activation rows, so the
    /// weight read — the dominant cost of batched decode — is amortized N× instead of
    /// re-streamed per token (the GEMM-N grid puts the token on <c>blockIdx.y</c> and only
    /// gets L2 reuse; #190 measured ~1.4× aggregate at N=8 from that).
    ///
    /// <para><b>Bit-exact:</b> each (row, token) pair runs the identical per-element
    /// reduction chain as the GEMM-N kernel (same loads, same product association, same
    /// warp/shared reduce order — see <see cref="CudaWsKernels"/>), so output is
    /// bit-identical to <see cref="MatMulBatched(Tensor,Tensor,Tensor,int,DType)"/> and to
    /// <paramref name="nTok"/> sequential <see cref="MatMul(Tensor,Tensor,Tensor,DType)"/>
    /// calls. Callers that need the byte-parity guarantee should still call
    /// <see cref="MatMulBatched(Tensor,Tensor,Tensor,int,DType)"/> — its contract is the
    /// documented one; this entry point only promises argmax-stability.</para>
    ///
    /// <para>Kernels are stamped per compile-time batch capacity (2/4/8/16; smallest ≥
    /// <paramref name="nTok"/> wins) so the per-token accumulators stay register-resident.
    /// <c>nTok == 1</c> and <c>nTok &gt; 16</c> delegate to the GEMM-N path (no weight
    /// reuse to gain at N=1; capacities above 16 spill registers — large N belongs to the
    /// compute-bound MMQ/GEMM prefill path). Supports the same dtypes as the GEMM-N matvec:
    /// Q4_K (AoS + #156 SoA, via per-token Q8_1 quantize + dp4a), Q5_K, Q6_K, Q8_0
    /// (AoS + #149 SoA), and Float32; anything else delegates (and throws) there too.</para>
    /// </summary>
    public void MatMulBatchedWeightStationary(Tensor outputAll, Tensor matrix, Tensor inputAll,
                                              int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatchedWeightStationary: outputAll ({outputAll.ElementCount}) and inputAll " +
                $"({inputAll.ElementCount}) element counts must be divisible by nTok ({nTok}).");

        int maxCapacity = CudaWsKernels.Variants[^1];
        if (nTok == 1 || nTok > maxCapacity ||
            weightDType is not (DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Float32))
        {
            MatMulBatched(outputAll, matrix, inputAll, nTok, weightDType);
            return;
        }

        int variant = 0;
        while (CudaWsKernels.Variants[variant] < nTok) variant++;

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        if (weightDType == DType.Q4_K)
        {
            DispatchMatVecQ4KWs(wPtr, xPtr, yPtr, rows, cols, nTok, variant,
                                soa: _soaQ4kHandles.ContainsKey(matrix.Handle));
            return;
        }

        // F32-input kernels: same (rows+7)/8 × 256-thread geometry as the GEMM-N
        // dispatch, minus the token grid dimension. #204: a SoA-repacked Q6_K weight
        // routes to the bit-identical SoA WS reader; otherwise #201's scale-word variant
        // (default) or the plain #194 AoS WS variant.
        bool q80Soa = weightDType == DType.Q8_0 && _soaHandles.ContainsKey(matrix.Handle);
        bool q6kSoa = weightDType == DType.Q6_K && _soaQ6kHandles.ContainsKey(matrix.Handle);
        nint kernel = weightDType switch
        {
            DType.Q6_K => q6kSoa ? _matvecQ6KWsSoaKernels[variant]
                                 : WsV2Enabled ? _matvecQ6KWsSwKernels[variant] : _matvecQ6KWsKernels[variant],
            DType.Q5_K => _matvecQ5KWsKernels[variant],
            DType.Q8_0 => q80Soa ? _matvecQ80WsSoaKernels[variant] : _matvecQ80WsKernels[variant],
            _          => _matvecF32WsKernels[variant],
        };
        int pRows = rows, pCols = cols, pN = nTok;
        nint* args = stackalloc nint[6]
        {
            (nint)(&wPtr), (nint)(&xPtr), (nint)(&yPtr),
            (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
        };
        uint gridX = (uint)((rows + 7) / 8);
        int r = NvrtcInterop.LaunchKernel(kernel, gridX, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_ws) failed: {r}");
    }

    /// <summary>
    /// Q4_K weight-stationary launch (#194): quantize all <paramref name="nTok"/> inputs to
    /// contiguous Q8_1 (identical to <see cref="DispatchMatVecQ4KBatched"/> — per-token
    /// quantization is unchanged, only the matvec geometry differs), then one
    /// grid = rows, block = 32 × MATVEC_Q4K_NWARPS launch with the token loop inside.
    /// </summary>
    private void DispatchMatVecQ4KWs(nint wPtr, nint xPtr, nint yPtr,
                                     int rows, int cols, int nTok, int variant, bool soa)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_ws requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;                          // per token
        long totalSub = (long)subBlocks * nTok;
        if ((long)cols * nTok > int.MaxValue)
            throw new InvalidOperationException(
                $"MatMulBatchedWeightStationary: cols*nTok ({(long)cols * nTok}) exceeds int range.");
        int qN = (int)((long)cols * nTok);

        EnsureQ81BatchBuf((nuint)(totalSub * 36L));

        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81BatchBuf;
            nint* args = stackalloc nint[3]
            {
                (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)totalSub, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 ws) failed: {rq}");
        }

        {
            nint q81Ptr = _q81BatchBuf;
            int  pRows  = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            // grid = rows; block = 32 × MATVEC_Q4K_NWARPS(8) = 256 threads (token loop inside).
            nint kernel = soa ? _matvecQ4KWsSoaKernels[variant] : _matvecQ4KWsKernels[variant];
            int rm = NvrtcInterop.LaunchKernel(kernel, (uint)rows, 1, 1,
                                               32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_ws{(soa ? "_soa" : "")}) failed: {rm}");
        }
    }

    /// <summary>
    /// Issue #201: int8 tensor-core batched-decode matmul (<c>SHARPI_BATCH_DECODE_MMQ=1</c>).
    /// The bit-exact WS matvecs are frozen into a lane geometry that saturates the
    /// L1TEX/LSU pipe ~3× above the weight-streaming floor at N=8 (see
    /// <see cref="CudaWsKernels"/>); this path relaxes the contract to <b>argmax-stable</b>
    /// — the contract the prefill MMQ holds — and runs a BN=16 decode tile of the
    /// prefill MMQ instead: SoA Q8_1 activations + one m16n8k32 s8 mma per warp per
    /// K-block, each weight byte read from HBM exactly once per step.
    ///
    /// <para>Eligibility: Q4_K with a #156 SoA-repacked weight (the decode default),
    /// cols % 256 == 0, N ≥ 5, and rows ≥ 2048. Below 2048 rows the (rows/64)-block
    /// grid starves the GPU (K/V-projection shapes), and below N=5 the BN=16 mma tile
    /// runs mostly predicated-off — measured on Qwen3-8B @ 4070 Ti: N=2 123 vs the
    /// bit-exact WS path's 139 t/s, N=4 a wash, N=8 317 vs 248. Everything else falls
    /// back to <see cref="MatMulBatchedWeightStationary"/>, so a mixed-dtype trunk
    /// (Qwen3-8B carries Q6_K attn_v/ffn_down halves) routes per tensor.</para>
    /// </summary>
    public void MatMulBatchedDecodeMmq(Tensor outputAll, Tensor matrix, Tensor inputAll,
                                       int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatchedDecodeMmq: outputAll ({outputAll.ElementCount}) and inputAll " +
                $"({inputAll.ElementCount}) element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        // Q4_K (#201/#205) and Q6_K (#204) both have a SoA-repacked-weight decode-MMQ tile;
        // a mixed-dtype trunk (Qwen3-8B Q4_K_M carries Q6_K ffn_down-half/attn_v/lm-head)
        // routes per tensor. Same eligibility floor as Q4_K: cols % 256, N ≥ 5, rows ≥ 2048
        // (below 2048 the (rows/64)-block grid starves — attn_v rows=1024 stays WS).
        bool q4kEligible = weightDType == DType.Q4_K && _soaQ4kHandles.ContainsKey(matrix.Handle);
        // #204: the Q6_K weight is now repacked to SoA in place (RepackQ6KSoa frees the AoS), so
        // the decode-MMQ tile reads `matrix` directly — same as Q4_K. Eligibility = the weight is
        // SoA-registered + the tile compiled + the kill-switch is on (SHARPI_Q6K_DECODE_MMQ).
        // Same floor as Q4_K: cols % 256, N ≥ 5, rows ≥ 2048 (below 2048 the (rows/64)-block grid
        // starves — attn_v rows=1024 stays WS).
        bool q6kEligible = weightDType == DType.Q6_K && _q6kDecodeMmqEnabled
                           && _mmqQ6kSoaActsN16Kernel != nint.Zero
                           && _soaQ6kHandles.ContainsKey(matrix.Handle);
        if ((!q4kEligible && !q6kEligible) || (cols & 0xff) != 0 || nTok < 5 || rows < 2048)
        {
            MatMulBatchedWeightStationary(outputAll, matrix, inputAll, nTok, weightDType);
            return;
        }

        // Both Q4_K and Q6_K read the in-place SoA weight directly (no companion).
        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        int subBlocks = cols / 32;
        long totalSub = (long)subBlocks * nTok;
        if ((long)cols * nTok > int.MaxValue)
            throw new InvalidOperationException(
                $"MatMulBatchedDecodeMmq: cols*nTok ({(long)cols * nTok}) exceeds int range.");
        int qN = (int)((long)cols * nTok);

        // Quantize activations f32 → SoA Q8_1 ([qs: totalSub*32 B][ds: totalSub*4 B],
        // the same producer the prefill SoA-acts MMQ uses).
        EnsureQ81BatchSoaBuf((nuint)(totalSub * 36L));
        nint qsPtr = _q81BatchSoaBuf;
        nint dsPtr = _q81BatchSoaBuf + (nint)(totalSub * 32L);
        {
            nint qInPtr = xPtr;
            nint* args = stackalloc nint[4]
            {
                (nint)(&qInPtr), (nint)(&qsPtr), (nint)(&dsPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81SoaKernel, (uint)totalSub, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1_soa decode-mmq) failed: {rq}");
        }

        {
            int pRows = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[7]
            {
                (nint)(&wPtr), (nint)(&qsPtr), (nint)(&dsPtr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            uint gy = (uint)((nTok + 15) / 16);
            // #205: the BM=64 tile yields ceil(rows/64) blocks. When that's below ~2 full SM
            // waves the grid starves (e.g. rows≈4096 → 64 blocks on a 60-SM 4070 Ti — Q/O
            // proj + the Q4_K ffn_down half), so route those low-row shapes to the BM=32 tile
            // (ceil(rows/32) blocks, 128 threads). High-row shapes (gate/up 12288, lm-head
            // 151936) keep BM=64 — they already fill the grid and BM=32 would double the
            // per-block activation re-staging. Output is bit-identical between the two tiles.
            // _smCount == 0 (attribute query failed) disables the BM=32 route (keeps BM=64).
            // #204: Q6_K uses its SoA decode-MMQ tile (the SoA weight is the only copy); Q4_K
            // uses its SoA tile. Both are bit-identical between BM=64 / BM=32.
            nint bm64 = q6kEligible ? _mmqQ6kSoaActsN16Kernel : _mmqQ4kSoaActsN16Kernel;
            nint bm32 = q6kEligible ? _mmqQ6kSoaActsN16Bm32Kernel : _mmqQ4kSoaActsN16Bm32Kernel;
            bool useBm32 = _decodeMmqBm32Enabled && bm32 != nint.Zero
                           && _smCount > 0 && (rows + 63) / 64 < 2 * _smCount;
            nint kernel = useBm32 ? bm32 : bm64;
            uint gx = useBm32 ? (uint)((rows + 31) / 32) : (uint)((rows + 63) / 64);
            uint block = useBm32 ? 128u : 256u;
            int rm = NvrtcInterop.LaunchKernel(kernel, gx, gy, 1,
                                               block, 1, 1, 0, _stream, args, null);
            string kname = q6kEligible ? "q6k_soa" : "q4k_soa";
            if (rm != 0) throw new InvalidOperationException(
                $"cuLaunchKernel(mmq_{kname}_acts_n16{(useBm32 ? "_bm32" : "")}) failed: {rm}");
        }
    }

    /// <summary>
    /// Compute-bound batched matmul for prefill (issue #141). Dequantizes the
    /// Q8_0 <paramref name="matrix"/> [rows×cols] to an fp16 scratch <b>once</b>,
    /// converts the <paramref name="nTok"/> activation vectors to fp16, then runs a
    /// single <c>cublasGemmEx</c> (fp16×fp16 → fp32, fp32 accumulate). The
    /// <see cref="MatMulBatched"/> matvec GEMM-N re-streams the weight matrix once
    /// per token (memory-bound); this reads each weight once per ~nTok batch, so on
    /// a compute-rich GPU prefill becomes compute-bound — the ~70× pp512 gap vs
    /// llama.cpp the matvec path could never close.
    ///
    /// <para><b>NOT bit-exact</b> to the matvec path: the weight value <c>d*q</c> and
    /// each activation are rounded to fp16 before the tensor-core multiply (fp32
    /// accumulation). Result tracks the fp32 path to fp tolerance (argmax-stable),
    /// not byte-for-byte. Callers that need byte-parity (GDN/MTP draft verify) must
    /// use <see cref="MatMulBatched"/>. Q8_0 and Q4_K weights (issue #156 Item C
    /// added Q4_K via <c>llm_dequant_q4k_to_f16</c>) — other dtypes throw.</para>
    ///
    /// Layout matches <see cref="MatMulBatched"/>: <paramref name="inputAll"/> is
    /// token-major <c>[nTok × cols]</c>, <paramref name="outputAll"/> is
    /// <c>[nTok × rows]</c>.
    /// </summary>
    public void MatMulBatchedGemm(Tensor outputAll, Tensor matrix, Tensor inputAll,
                                  int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (weightDType is not (DType.Q8_0 or DType.Q4_K or DType.Q6_K or DType.Q5_K or DType.Q4_0))
            throw new NotSupportedException(
                $"CUDA MatMulBatchedGemm: weight dtype {weightDType} not supported (Q8_0, Q4_0, Q4_K, Q5_K, or Q6_K).");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatchedGemm: outputAll ({outputAll.ElementCount}) and inputAll " +
                $"({inputAll.ElementCount}) element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        // Q8_0 sub-block is 32 elements; Q4_K/Q5_K/Q6_K super-block is 256. Each dequant
        // kernel loops over the row in its native block size, so cols must align to it.
        int colAlign = weightDType is DType.Q4_K or DType.Q5_K or DType.Q6_K ? 256 : 32;
        if ((cols % colAlign) != 0)
            throw new InvalidOperationException(
                $"CUDA MatMulBatchedGemm ({weightDType}) requires cols % {colAlign} == 0 (got {cols}).");

        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        EnsureGemmWf16((nuint)((long)rows * cols * 2L));
        EnsureGemmAf16((nuint)((long)nTok * cols * 2L));

        // 1) Dequant weight → fp16 (one block of 256 threads per row). Q8_0 is #149
        //    SoA-aware; Q4_K (#156 Item C) decodes the 256-element super-block. Both
        //    write a row-major [rows×cols] fp16 buffer for the GemmEx below.
        {
            nint wp = wPtr, op = _gemmWf16Buf;
            int pRows = rows, pCols = cols;
            // #156: a repacked SoA Q4_K weight on this fallback (SHARPI_PREFILL_MMQ=0)
            // path uses the bit-identical SoA dequant kernel.
            nint dqKern = weightDType switch
            {
                DType.Q4_K => _soaQ4kHandles.ContainsKey(matrix.Handle) ? _dequantQ4KF16SoaKernel : _dequantQ4KF16Kernel,
                DType.Q6_K => _soaQ6kHandles.ContainsKey(matrix.Handle) ? _dequantQ6KF16SoaKernel : _dequantQ6KF16Kernel,   // #162/#204
                DType.Q5_K => _dequantQ5KF16Kernel,   // #162
                DType.Q4_0 => _soaQ40Handles.ContainsKey(matrix.Handle) ? _dequantQ40F16SoaKernel : _dequantQ40F16Kernel,   // #124/#173
                _          => _soaHandles.ContainsKey(matrix.Handle) ? _dequantQ80F16SoaKernel : _dequantQ80F16Kernel,
            };
            nint* args = stackalloc nint[4] { (nint)(&wp), (nint)(&op), (nint)(&pRows), (nint)(&pCols) };
            int r = NvrtcInterop.LaunchKernel(dqKern, (uint)rows, 1, 1,
                                              256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(dequant_{weightDType}_to_f16) failed: {r}");
        }

        // 2) Convert activations fp32 → fp16.
        {
            nint ip = xPtr, op = _gemmAf16Buf;
            int n = nTok * cols;
            nint* args = stackalloc nint[3] { (nint)(&ip), (nint)(&op), (nint)(&n) };
            uint grid = (uint)((n + 255) / 256);
            int r = NvrtcInterop.LaunchKernel(_f32ToF16Kernel, grid, 1, 1,
                                              256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(f32_to_f16) failed: {r}");
        }

        // 3) GemmEx: C[nTok×rows] f32 = A[nTok×cols] f16 × B[rows×cols] f16ᵀ, fp32 accum.
        //    Row-major via the col-major transpose identity (mirrors Sgemm):
        //    row-major C=A·Bᵀ ≡ col-major Cᵀ = B·Aᵀ.
        {
            float alpha = 1f, beta = 0f;
            int M = nTok, K = cols, N = rows;
            int status = CuBlasInterop.GemmEx(
                _handle,
                CuBlasInterop.OpT, CuBlasInterop.OpN,
                N, M, K,
                ref alpha,
                _gemmWf16Buf, CuBlasInterop.CUDA_R_16F, K,
                _gemmAf16Buf, CuBlasInterop.CUDA_R_16F, K,
                ref beta,
                yPtr, CuBlasInterop.CUDA_R_32F, N,
                CuBlasInterop.CUBLAS_COMPUTE_32F,
                CuBlasInterop.CUBLAS_GEMM_DEFAULT);
            if (status != 0)
                throw new InvalidOperationException($"cublasGemmEx (prefill GEMM) failed: {status}");
        }
    }

    /// <summary>
    /// Issue #141 (MMQ): int8 tensor-core batched matmul for Q8_0 weights.
    /// C[nTok×rows] f32 = X[nTok×cols] · Wᵀ where W is Q8_0 [rows×cols]. The input is
    /// quantized to Q8_1 (per-32-block int8 + fp16 scale) and multiplied by the Q8_0
    /// weight via the m16n8k32 s8 mma — the weight is read once as int8, with no fp16
    /// dequant temp written to HBM (the cost that capped <see cref="MatMulBatchedGemm"/>).
    /// Argmax-stable, not bit-exact (both operands are int8-quantized).
    /// </summary>
    public void MatMulBatchedMmq(Tensor outputAll, Tensor matrix, Tensor inputAll,
                                 int nTok, DType weightDType)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (weightDType is not (DType.Q8_0 or DType.Q4_K or DType.Q4_0))
            throw new NotSupportedException(
                $"CUDA MatMulBatchedMmq: weight dtype {weightDType} not supported (Q8_0, Q4_K, or Q4_0).");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatchedMmq: outputAll ({outputAll.ElementCount}) and inputAll " +
                $"({inputAll.ElementCount}) element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        // Q4_K super-blocks are 256-wide (get_scale_min/nibble layout); Q8_0/Q4_0 are 32-wide.
        int colAlign = weightDType == DType.Q4_K ? 256 : 32;
        if ((cols % colAlign) != 0)
            throw new InvalidOperationException(
                $"CUDA MatMulBatchedMmq ({weightDType}) requires cols % {colAlign} == 0 (got {cols}).");

        nint wPtr = GetDevPtr(matrix);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        if ((long)cols * nTok > int.MaxValue)
            throw new InvalidOperationException(
                $"MatMulBatchedMmq: cols*nTok ({(long)cols * nTok}) exceeds int range.");

        int subBlocks = cols / 32;
        long totalSub = (long)subBlocks * nTok;

        // Track A (#124/#173): the SoA-activation MMQ requires a SoA-repacked weight (the
        // _soa_acts kernels read the SoA weight layout). Only then is ActSoaEnabled honored;
        // AoS-weight prefill always takes the interleaved-activation path.
        bool wIsSoa = weightDType switch
        {
            DType.Q4_K => _soaQ4kHandles.ContainsKey(matrix.Handle),
            DType.Q4_0 => _soaQ40Handles.ContainsKey(matrix.Handle),
            _          => _soaHandles.ContainsKey(matrix.Handle),
        };
        // cp.async (opt-in/default-off, #124/#173) implies the SoA-activation substrate —
        // engage it for the formats that HAVE a cp.async kernel (Q8_0/Q4_0). Q4_K only takes
        // the SoA path when ActSoaEnabled is explicitly set (no Q4_K cp.async kernel → scalar).
        bool cpaFmt = weightDType is DType.Q8_0 or DType.Q4_0;
        bool useCpa = ActSoaCpaEnabled && cpaFmt && wIsSoa;
        bool useActSoa = (ActSoaEnabled || useCpa) && wIsSoa;

        uint gx = (uint)((rows + 63) / 64), gy = (uint)((nTok + 127) / 128);

        if (useActSoa)
        {
            // 1) Quantize activations f32 → SoA Q8_1: [qs totalSub*32 B][ds totalSub*4 B].
            EnsureQ81BatchSoaBuf((nuint)(totalSub * 36L));
            nint qsPtr = _q81BatchSoaBuf;
            nint dsPtr = _q81BatchSoaBuf + (nint)(totalSub * 32L);
            {
                nint qIn = xPtr, qQs = qsPtr, qDs = dsPtr;
                int qN = (int)((long)cols * nTok);
                nint* args = stackalloc nint[4] { (nint)(&qIn), (nint)(&qQs), (nint)(&qDs), (nint)(&qN) };
                int rq = NvrtcInterop.LaunchKernel(_quantizeQ81SoaKernel, (uint)totalSub, 1, 1,
                                                   32, 1, 1, 0, _stream, args, null);
                if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1_soa mmq) failed: {rq}");
            }
            // 2) SoA-weight + SoA-activation MMQ.
            {
                int pRows = rows, pCols = cols, pN = nTok;
                nint* args = stackalloc nint[7]
                {
                    (nint)(&wPtr), (nint)(&qsPtr), (nint)(&dsPtr), (nint)(&yPtr),
                    (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
                };
                nint kern = weightDType switch
                {
                    DType.Q4_K => _mmqQ4kSoaActsKernel,
                    DType.Q4_0 => useCpa ? _mmqQ40SoaActsCpaKernel : _mmqQ40SoaActsKernel,
                    _          => useCpa ? _mmqQ80SoaActsCpaKernel : _mmqQ80SoaActsKernel,
                };
                int rm = NvrtcInterop.LaunchKernel(kern, gx, gy, 1, 256, 1, 1, 0, _stream, args, null);
                if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(mmq {weightDType} soa_acts) failed: {rm}");
            }
            return;
        }

        // 1) Quantize activations [nTok×cols] f32 → contiguous Q8_1 (36 B/block). The
        //    per-block quantize is independent, so a single launch over nTok×subBlocks
        //    is bit-identical to per-token quantization (mirrors DispatchMatVecQ4KBatched).
        EnsureQ81BatchBuf((nuint)(totalSub * 36L));
        {
            nint qIn = xPtr, qOut = _q81BatchBuf;
            int qN = (int)((long)cols * nTok);
            nint* args = stackalloc nint[3] { (nint)(&qIn), (nint)(&qOut), (nint)(&qN) };
            int rq = NvrtcInterop.LaunchKernel(_quantizeQ81Kernel, (uint)totalSub, 1, 1,
                                               32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 mmq) failed: {rq}");
        }

        // 2) int8 mma MMQ: grid = ((rows+63)/64, (nTok+127)/128), 256 threads/block,
        //    each block a 64×128 output tile (8 warps × 8 m16n8k32 mma per K-block).
        {
            nint q81 = _q81BatchBuf;
            int pRows = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&q81), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            // Q4_K → the nibble-expanding MMQ kernel; Q4_0 → the symmetric nibble MMQ
            // (#124/#173); Q8_0 → the int8 kernel (#149: if the weight was repacked SoA,
            // the aligned-load variant — same args either way).
            nint kern = weightDType switch
            {
                DType.Q4_K => _soaQ4kHandles.ContainsKey(matrix.Handle) ? _mmqQ4kSoaKernel : _mmqQ4kKernel,   // #156
                DType.Q4_0 => _soaQ40Handles.ContainsKey(matrix.Handle) ? _mmqQ40SoaKernel : _mmqQ40Kernel,   // #124/#173
                _          => _soaHandles.ContainsKey(matrix.Handle) ? _mmqQ80SoaKernel : _mmqQ80Kernel,
            };
            int rm = NvrtcInterop.LaunchKernel(kern, gx, gy, 1,
                                               256, 1, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(mmq {weightDType}) failed: {rm}");
        }
    }

    /// <summary>
    /// Issue #149: SoA-layout variant of <see cref="MatMulBatchedMmq"/>. Identical
    /// math/output, but <paramref name="soaWeight"/> is a single buffer holding the
    /// Q8_0 weight repacked struct-of-arrays — <c>[quants rows*cols B][scales rows*nb
    /// fp16]</c> — so the 32 quants/block are contiguous &amp; 16-byte aligned and the
    /// weight load uses plain aligned uint reads instead of the funnelshift the
    /// interleaved 34-byte block forces. The host splits the buffer at byte rows*cols
    /// into the quant and scale pointers. Bit-identical to <see cref="MatMulBatchedMmq"/>.
    /// </summary>
    public void MatMulBatchedMmqSoa(Tensor outputAll, Tensor soaWeight, Tensor inputAll, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (outputAll.ElementCount % nTok != 0 || inputAll.ElementCount % nTok != 0)
            throw new ArgumentException(
                $"MatMulBatchedMmqSoa: outputAll ({outputAll.ElementCount}) and inputAll " +
                $"({inputAll.ElementCount}) element counts must be divisible by nTok ({nTok}).");

        int rows = (int)(outputAll.ElementCount / nTok);
        int cols = (int)(inputAll.ElementCount / nTok);
        if ((cols & 31) != 0)
            throw new InvalidOperationException(
                $"CUDA MatMulBatchedMmqSoa requires cols % 32 == 0 (got {cols}).");

        nint wPtr = GetDevPtr(soaWeight);
        nint xPtr = GetDevPtr(inputAll);
        nint yPtr = GetDevPtr(outputAll);

        int subBlocks = cols / 32;
        long totalSub = (long)subBlocks * nTok;
        EnsureQ81BatchBuf((nuint)(totalSub * 36L));
        {
            nint qIn = xPtr, qOut = _q81BatchBuf;
            long ce = (long)cols * nTok;
            if (ce > int.MaxValue)
                throw new InvalidOperationException($"MatMulBatchedMmqSoa: cols*nTok ({ce}) exceeds int range.");
            int qN = (int)ce;
            nint* args = stackalloc nint[3] { (nint)(&qIn), (nint)(&qOut), (nint)(&qN) };
            int rq = NvrtcInterop.LaunchKernel(_quantizeQ81Kernel, (uint)totalSub, 1, 1,
                                               32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 mmq-soa) failed: {rq}");
        }
        {
            nint q81 = _q81BatchBuf;
            int pRows = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&q81), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            uint gx = (uint)((rows + 63) / 64), gy = (uint)((nTok + 127) / 128);
            int rm = NvrtcInterop.LaunchKernel(_mmqQ80SoaKernel, gx, gy, 1,
                                               256, 1, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(mmq_q8_0_soa) failed: {rm}");
        }
    }

    /// <summary>Issue #149: device handles of Q8_0 weights repacked into the SoA layout
    /// (see <see cref="RepackQ8_0Soa"/>). <see cref="MatMulBatchedMmq"/> and the decode
    /// matvec auto-route to the aligned-load SoA kernels for these.</summary>
    private readonly ConcurrentDictionary<nint, byte> _soaHandles = new();

    /// <summary>Issue #156: device handles of Q4_K weights repacked into the
    /// scale-pre-unpacked SoA layout (see <see cref="RepackQ4KSoa"/>). The decode
    /// matvec auto-routes these to <c>llm_matvec_q4k_soa</c>.</summary>
    private readonly ConcurrentDictionary<nint, byte> _soaQ4kHandles = new();

    /// <summary>Issue #204: device handles of Q6_K weights repacked into the
    /// scale-pre-unpacked SoA layout ([Q (q6−32) int8][S int8 scales][D fp16 d], see
    /// <see cref="RepackQ6KSoa"/>). Like Q4_K (#156), <see cref="RepackQ6KSoa"/> FREES the
    /// interleaved AoS weight — the SoA buffer is the only copy — so EVERY Q6_K reader
    /// (single-token / N=2 / GEMM-N matvec, WS matvec, prefill GEMM dequant, and the
    /// <see cref="MatMulBatchedDecodeMmq"/> decode-MMQ tile) auto-routes to its <c>*_soa</c>
    /// kernel for these handles.</summary>
    private readonly ConcurrentDictionary<nint, byte> _soaQ6kHandles = new();

    /// <summary>Issue #124/#173: device handles of Q4_0 weights repacked into the SoA
    /// layout (see <see cref="RepackQ4_0Soa"/>). The MMQ / decode dp4a / fp32 matvec /
    /// GEMM-fallback dequant all auto-route to the aligned-load SoA kernels.</summary>
    private readonly ConcurrentDictionary<nint, byte> _soaQ40Handles = new();

    /// <summary>
    /// Issue #149: allocate a new buffer the same size as the interleaved Q8_0
    /// <paramref name="src"/> [rows×nb×34 B], repack it into the SoA layout
    /// [quants rows*cols B][scales rows*nb fp16] on the GPU, free <paramref name="src"/>,
    /// and mark the new handle so the matmul/matvec dispatch uses the SoA kernels.
    /// </summary>
    public Tensor RepackQ8_0Soa(Tensor src, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if ((cols & 31) != 0)
            throw new InvalidOperationException($"RepackQ8_0Soa requires cols % 32 == 0 (got {cols}).");

        long bytes = (long)rows * (cols / 32) * 34L;   // == rows*cols + rows*nb*2
        var dst = AllocateRawBytes(bytes, DType.Q8_0, exact: true);
        nint sPtr = GetDevPtr(src), dPtr = GetDevPtr(dst);
        int pRows = rows, pCols = cols;
        nint* args = stackalloc nint[4] { (nint)(&sPtr), (nint)(&dPtr), (nint)(&pRows), (nint)(&pCols) };
        long totalBlocks = (long)rows * (cols / 32);
        uint grid = (uint)((totalBlocks + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_q80RepackSoaKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(q8_0_repack_soa) failed: {r}");
        Synchronize();
        Free(src);
        _soaHandles[dst.Handle] = 0;
        return dst;
    }

    /// <summary>
    /// Issue #156: repack an interleaved Q4_K weight [rows × nb × 144 B] into the
    /// scale-pre-unpacked SoA layout [Q rows*nb*128][S rows*nb*16][D rows*nb*4]
    /// (see <c>llm_q4k_repack_soa</c>), free <paramref name="src"/>, and mark the new
    /// handle so the decode matvec routes to <c>llm_matvec_q4k_soa</c>. The repacked
    /// scale/min bytes are bit-identical to the AoS switch unpack, so the matvec is
    /// bit-identical to <see cref="DispatchMatVecQ4K"/>. Costs +4 B per 144-B super-block.
    /// </summary>
    public Tensor RepackQ4KSoa(Tensor src, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException($"RepackQ4KSoa requires cols % 256 == 0 (got {cols}).");

        long nb = cols / 256;
        long totalSub = (long)rows * nb;
        long bytes = totalSub * (128L + 16L + 4L);     // Q + S + D regions
        var dst = AllocateRawBytes(bytes, DType.Q4_K, exact: true);
        nint sPtr = GetDevPtr(src), dPtr = GetDevPtr(dst);
        int pRows = rows, pCols = cols;
        nint* args = stackalloc nint[4] { (nint)(&sPtr), (nint)(&dPtr), (nint)(&pRows), (nint)(&pCols) };
        uint grid = (uint)((totalSub + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_q4kRepackSoaKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(q4k_repack_soa) failed: {r}");
        Synchronize();
        Free(src);
        _soaQ4kHandles[dst.Handle] = 0;
        return dst;
    }

    /// <summary>
    /// Issue #204: allocate a new buffer and repack the interleaved Q6_K weight
    /// <paramref name="src"/> [rows × nb × 210 B] into the scale-pre-unpacked SoA layout
    /// [Q rows*nb*256][S rows*nb*16][D rows*nb*4] (see <c>llm_q6k_repack_soa</c>), FREE
    /// <paramref name="src"/>, and mark the new handle so every Q6_K reader routes to its
    /// <c>*_soa</c> kernel. The Q region stores the signed int8 <c>(q6 − 32)</c> per natural
    /// element (the matvec's pre-multiply weight); S the 16 int8 scales verbatim; D the fp16 d.
    ///
    /// <para>Mirrors <see cref="RepackQ4KSoa"/>: the interleaved weight is freed so the SoA
    /// buffer is the ONLY copy (net only ~+0.4 GB over the 210 B/super-block AoS — the Q region
    /// grows from 192 to 256 B and S/D round up). Every reader is SoA-aware
    /// (<c>llm_matvec_q6k_soa</c>, <c>..._n2_soa</c>, <c>..._gemm_n_soa</c>,
    /// <c>..._ws_soa</c>, <c>llm_dequant_q6k_to_f16_soa</c>, and the decode-MMQ tile
    /// <c>llm_mmq_q6k_soa_acts_n16</c>), all bit-identical to their AoS counterparts.</para>
    /// </summary>
    public Tensor RepackQ6KSoa(Tensor src, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException($"RepackQ6KSoa requires cols % 256 == 0 (got {cols}).");

        long nb = cols / 256;
        long totalSub = (long)rows * nb;
        long bytes = totalSub * (256L + 16L + 4L);     // Q + S + D regions
        var dst = AllocateRawBytes(bytes, DType.Q6_K, exact: true);
        nint sPtr = GetDevPtr(src), dPtr = GetDevPtr(dst);
        int pRows = rows, pCols = cols;
        nint* args = stackalloc nint[4] { (nint)(&sPtr), (nint)(&dPtr), (nint)(&pRows), (nint)(&pCols) };
        uint grid = (uint)((totalSub + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_q6kRepackSoaKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(q6k_repack_soa) failed: {r}");
        Synchronize();
        Free(src);
        _soaQ6kHandles[dst.Handle] = 0;
        return dst;
    }

    /// <summary>
    /// Issue #124/#173: allocate a new buffer the same size as the interleaved Q4_0
    /// <paramref name="src"/> [rows×nb×18 B], repack it into the SoA layout
    /// [quants rows*cols/2 B][scales rows*nb fp16] on the GPU, free <paramref name="src"/>,
    /// and mark the new handle so the MMQ / decode dp4a / fp32 matvec / GEMM-fallback
    /// dequant route to the aligned-load SoA kernels. Quant bytes + scales are bit-
    /// identical to the AoS block, so every SoA reader is bit-identical to its AoS twin.
    /// </summary>
    public Tensor RepackQ4_0Soa(Tensor src, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available on this system.");
        if ((cols & 31) != 0)
            throw new InvalidOperationException($"RepackQ4_0Soa requires cols % 32 == 0 (got {cols}).");

        long bytes = (long)rows * (cols / 32) * 18L;   // == rows*cols/2 + rows*nb*2
        var dst = AllocateRawBytes(bytes, DType.Q4_0, exact: true);
        nint sPtr = GetDevPtr(src), dPtr = GetDevPtr(dst);
        int pRows = rows, pCols = cols;
        nint* args = stackalloc nint[4] { (nint)(&sPtr), (nint)(&dPtr), (nint)(&pRows), (nint)(&pCols) };
        long totalBlocks = (long)rows * (cols / 32);
        uint grid = (uint)((totalBlocks + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_q40RepackSoaKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(q4_0_repack_soa) failed: {r}");
        Synchronize();
        Free(src);
        _soaQ40Handles[dst.Handle] = 0;
        return dst;
    }

    /// <summary>
    /// Issue #156: Q4_K decode matvec over the scale-pre-unpacked SoA weight. Same
    /// Q8_1 input quantization as <see cref="DispatchMatVecQ4K"/>, then dispatches
    /// <c>llm_matvec_q4k_soa</c> (1 row/block, MATVEC_Q4K_SOA_NWARPS warps).
    /// </summary>
    private void DispatchMatVecQ4KSoa(nint wPtr, nint xPtr, nint yPtr, int rows, int cols)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_soa requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);

        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81Buf;
            int  qN      = cols;
            nint* args = stackalloc nint[3] { (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN) };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)subBlocks, 1, 1, 32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1) failed: {rq}");
        }

        {
            nint q81Ptr = _q81Buf;
            int  pRows  = rows, pCols = cols;
            nint* args = stackalloc nint[5]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                _matvecQ4KSoaKernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);   // MATVEC_Q4K_SOA_NWARPS = 8
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_soa) failed: {rm}");
        }
    }

    private void EnsureGemmWf16(nuint required)
    {
        if (_gemmWf16Buf != nint.Zero && _gemmWf16Size >= required) return;
        if (_gemmWf16Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_gemmWf16Buf);
            _gemmWf16Buf = nint.Zero;
            _gemmWf16Size = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _gemmWf16Buf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(gemm wf16, {newSize} B) failed: {r}");
        _gemmWf16Size = newSize;
    }

    private void EnsureGemmAf16(nuint required)
    {
        if (_gemmAf16Buf != nint.Zero && _gemmAf16Size >= required) return;
        if (_gemmAf16Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_gemmAf16Buf);
            _gemmAf16Buf = nint.Zero;
            _gemmAf16Size = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _gemmAf16Buf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(gemm af16, {newSize} B) failed: {r}");
        _gemmAf16Size = newSize;
    }

    /// <summary>
    /// Q4_K batched GEMM-N: quantizes all <paramref name="nTok"/> input vectors into a
    /// single contiguous Q8_1 scratch (one launch over <c>nTok × subBlocks</c> blocks —
    /// the per-token quantize is independent so this is bit-identical to per-token
    /// quantization), then dispatches <c>llm_matvec_q4k_gemm_n</c> over a (rows, nTok)
    /// grid.
    /// </summary>
    private void DispatchMatVecQ4KBatched(nint wPtr, nint xPtr, nint yPtr,
                                          int rows, int cols, int nTok, bool soa)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_gemm_n requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;                          // per token
        long totalSub = (long)subBlocks * nTok;
        nuint q81Bytes = (nuint)(totalSub * 36L);
        EnsureQ81BatchBuf(q81Bytes);

        // Quantize all nTok inputs → contiguous Q8_1. The single-token kernel's
        // index math (elem_idx = block_id*32 + lane, out += block_id*36) already
        // covers a contiguous [nTok × cols] batch: block_id runs [0, nTok*subBlocks).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81BatchBuf;
            int  qN      = (int)((long)cols * nTok);
            if ((long)cols * nTok > int.MaxValue)
                throw new InvalidOperationException(
                    $"MatMulBatched: cols*nTok ({(long)cols * nTok}) exceeds int range.");
            nint* args = stackalloc nint[3]
            {
                (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)totalSub, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 batched) failed: {rq}");
        }

        {
            nint q81Ptr = _q81BatchBuf;
            int  pRows  = rows, pCols = cols, pN = nTok;
            nint* args = stackalloc nint[6]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols), (nint)(&pN)
            };
            // grid = (rows, nTok); block = 32 × MATVEC_Q4K_NWARPS(8) = 256 threads.
            int rm = NvrtcInterop.LaunchKernel(
                soa ? _matvecQ4KGemmNSoaKernel : _matvecQ4KGemmNKernel, (uint)rows, (uint)nTok, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_gemm_n{(soa ? "_soa" : "")}) failed: {rm}");
        }
    }

    private void EnsureQ81BatchBuf(nuint required)
    {
        if (_q81BatchBuf != nint.Zero && _q81BatchBufSize >= required) return;
        if (_q81BatchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchBuf);
            _q81BatchBuf = nint.Zero;
            _q81BatchBufSize = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81BatchBuf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 batch scratch, {newSize} B) failed: {r}");
        _q81BatchBufSize = newSize;
    }

    /// <summary>Track A (#124/#173): grow-only SoA Q8_1 activation scratch
    /// (<c>_q81BatchSoaBuf</c>), one allocation holding [qs: totalSub*32 B][ds:
    /// totalSub*4 B]. Same policy as <see cref="EnsureQ81BatchBuf"/>.</summary>
    private void EnsureQ81BatchSoaBuf(nuint required)
    {
        if (_q81BatchSoaBuf != nint.Zero && _q81BatchSoaBufSize >= required) return;
        if (_q81BatchSoaBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchSoaBuf);
            _q81BatchSoaBuf = nint.Zero;
            _q81BatchSoaBufSize = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81BatchSoaBuf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 SoA batch scratch, {newSize} B) failed: {r}");
        _q81BatchSoaBufSize = newSize;
    }

    /// <summary>
    /// Q4_K N=2 matvec: quantizes both input vectors into independent Q8_1
    /// scratches (<c>_q81Buf</c> for A, <c>_q81BufB</c> for B), then dispatches
    /// the cooperative <c>llm_matvec_q4k_n2</c> kernel that reads each weight
    /// super-block once and accumulates into two outputs per row. When the weight
    /// was repacked to the scale-pre-unpacked SoA layout (#156), dispatches the
    /// bit-identical <c>llm_matvec_q4k_n2_soa</c> reader instead.
    /// </summary>
    private void DispatchMatVecQ4KN2(nint wPtr, nint xAPtr, nint xBPtr,
                                     nint yAPtr, nint yBPtr,
                                     int rows, int cols, bool soa)
    {
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k_n2 requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);
        EnsureQ81BufB(q81Bytes);

        // Quantize both inputs into their own Q8_1 scratch (32 threads per sub-block).
        // Two unrolled launches — keeps stackalloc out of any loop (CA2014).
        nint qInA  = xAPtr;
        nint qOutA = _q81Buf;
        nint qInB  = xBPtr;
        nint qOutB = _q81BufB;
        int  qN    = cols;
        nint* qArgsA = stackalloc nint[3] { (nint)(&qInA), (nint)(&qOutA), (nint)(&qN) };
        int rqA = NvrtcInterop.LaunchKernel(
            _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
            32, 1, 1, 0, _stream, qArgsA, null);
        if (rqA != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 A) failed: {rqA}");
        nint* qArgsB = stackalloc nint[3] { (nint)(&qInB), (nint)(&qOutB), (nint)(&qN) };
        int rqB = NvrtcInterop.LaunchKernel(
            _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
            32, 1, 1, 0, _stream, qArgsB, null);
        if (rqB != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 B) failed: {rqB}");

        {
            nint q81PtrA = _q81Buf;
            nint q81PtrB = _q81BufB;
            int  pRows   = rows, pCols = cols;
            nint* args = stackalloc nint[7]
            {
                (nint)(&wPtr),
                (nint)(&q81PtrA), (nint)(&q81PtrB),
                (nint)(&yAPtr),   (nint)(&yBPtr),
                (nint)(&pRows),   (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                soa ? _matvecQ4KN2SoaKernel : _matvecQ4KN2Kernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k_n2{(soa ? "_soa" : "")}) failed: {rm}");
        }
    }

    /// <summary>
    /// Q4_K matvec via the llama.cpp Q8_1 + __dp4a path. Quantizes <paramref name="xPtr"/>
    /// (cols floats) into the persistent Q8_1 scratch in 32-element sub-blocks, then
    /// dispatches the cooperative matvec (1 row per block, 4 warps cooperating).
    /// </summary>
    private void DispatchMatVecQ4K(nint wPtr, nint xPtr, nint yPtr, int rows, int cols)
    {
        // cols must be a multiple of 256 (one Q4_K super-block = 8 Q8_1 sub-blocks).
        // Every transformer hidden dim in practice satisfies this; assert defensively.
        if ((cols & 0xff) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4k requires cols % 256 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);

        // ── Quantize input → Q8_1 (one CUDA block of 32 threads per sub-block).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81Buf;
            int  qN      = cols;
            nint* args = stackalloc nint[3]
            {
                (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN)
            };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1) failed: {rq}");
        }

        // ── Cooperative matvec: 1 row/block, MATVEC_Q4K_NWARPS warps × 32 threads.
        // 8 warps (256 threads) per block delivers enough in-flight instructions
        // to hide global-memory latency on Ada at the modest weight footprint of
        // a single LLM row (a few KB of Q4_K).
        {
            nint q81Ptr = _q81Buf;
            int  pRows  = rows, pCols = cols;
            nint* args = stackalloc nint[5]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                _matvecQ4KKernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4k) failed: {rm}");
        }
    }

    /// <summary>
    /// Q8_0 decode matvec via dp4a (issue #142): quantize the input vector to Q8_1
    /// (32-element sub-blocks), then dispatch <c>llm_matvec_q8_0_dp4a</c> (1 row /
    /// block, MATVEC_Q80_NWARPS warps). The int8·int8 dp4a inner product replaces
    /// the per-element int8→float decode of the fp32 matvec, cutting instruction
    /// count on the bandwidth-bound decode path. Requires <c>cols % 32 == 0</c>.
    /// </summary>
    /// <summary>
    /// Pre-grow the Q8_1 input-quantization scratch to hold a <paramref name="cols"/>-wide
    /// vector (issue #142). Call before a CUDA-graph capture region that contains a dp4a
    /// matvec: capture forbids <c>cudaMalloc</c>, so the buffer must already be at its max
    /// size. No-op if already large enough.
    /// </summary>
    public void EnsureQ81Scratch(int cols)
    {
        if (cols <= 0) return;
        int subBlocks = (cols + 31) / 32;
        EnsureQ81Buf((nuint)((long)subBlocks * 36L));
    }

    private void DispatchMatVecQ80Dp4a(nint wPtr, nint xPtr, nint yPtr, int rows, int cols, bool soa = false)
    {
        if ((cols & 31) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q8_0_dp4a requires cols % 32 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        nuint q81Bytes = (nuint)((long)subBlocks * 36L);
        EnsureQ81Buf(q81Bytes);

        // Quantize input → Q8_1 (32 threads per sub-block).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81Buf;
            int  qN      = cols;
            nint* args = stackalloc nint[3] { (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN) };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 for q8_0) failed: {rq}");
        }

        // Cooperative dp4a matvec: 1 row/block, MATVEC_Q80_NWARPS warps × 32 threads.
        {
            nint q81Ptr = _q81Buf;
            int  pRows  = rows, pCols = cols;
            nint* args = stackalloc nint[5]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                soa ? _matvecQ80Dp4aSoaKernel : _matvecQ80Dp4aKernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q8_0_dp4a) failed: {rm}");
        }
    }

    /// <summary>
    /// Q4_0 decode matvec via dp4a (issue #124): quantize the input vector to Q8_1
    /// (32-element sub-blocks), then dispatch <c>llm_matvec_q4_0_dp4a</c> (1 row /
    /// block, MATVEC_Q40_NWARPS warps). Q4_0 is symmetric, so the asymmetric dp4a
    /// trick uses the stored Q8_1 sum to subtract the 8·Σq centering term once per
    /// block. Requires <c>cols % 32 == 0</c>.
    /// </summary>
    private void DispatchMatVecQ40Dp4a(nint wPtr, nint xPtr, nint yPtr, int rows, int cols, bool soa = false)
    {
        if ((cols & 31) != 0)
            throw new InvalidOperationException(
                $"CUDA matvec_q4_0_dp4a requires cols % 32 == 0 (got {cols}).");

        int subBlocks = cols / 32;
        EnsureQ81Buf((nuint)((long)subBlocks * 36L));

        // Quantize input → Q8_1 (32 threads per sub-block).
        {
            nint qInPtr  = xPtr;
            nint qOutPtr = _q81Buf;
            int  qN      = cols;
            nint* args = stackalloc nint[3] { (nint)(&qInPtr), (nint)(&qOutPtr), (nint)(&qN) };
            int rq = NvrtcInterop.LaunchKernel(
                _quantizeQ81Kernel, (uint)subBlocks, 1, 1,
                32, 1, 1, 0, _stream, args, null);
            if (rq != 0) throw new InvalidOperationException($"cuLaunchKernel(quantize_q8_1 for q4_0) failed: {rq}");
        }

        // Cooperative dp4a matvec: 1 row/block, MATVEC_Q40_NWARPS warps × 32 threads.
        {
            nint q81Ptr = _q81Buf;
            int  pRows  = rows, pCols = cols;
            nint* args = stackalloc nint[5]
            {
                (nint)(&wPtr), (nint)(&q81Ptr), (nint)(&yPtr),
                (nint)(&pRows), (nint)(&pCols)
            };
            int rm = NvrtcInterop.LaunchKernel(
                soa ? _matvecQ40Dp4aSoaKernel : _matvecQ40Dp4aKernel, (uint)rows, 1, 1,
                32, 8, 1, 0, _stream, args, null);
            if (rm != 0) throw new InvalidOperationException($"cuLaunchKernel(matvec_q4_0_dp4a) failed: {rm}");
        }
    }

    private void EnsureQ81Buf(nuint required)
    {
        if (_q81Buf != nint.Zero && _q81BufSize >= required) return;
        if (_q81Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81Buf);
            _q81Buf = nint.Zero;
            _q81BufSize = 0;
        }
        // Grow generously — the largest activation in a typical LLM is intermediate_dim
        // (Qwen3 = 12288 → ~14 KB Q8_1). Round up to the next 64 KB to amortise growth.
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81Buf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 scratch, {newSize} B) failed: {r}");
        _q81BufSize = newSize;
    }

    private void EnsureQ81BufB(nuint required)
    {
        if (_q81BufB != nint.Zero && _q81BufBSize >= required) return;
        if (_q81BufB != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BufB);
            _q81BufB = nint.Zero;
            _q81BufBSize = 0;
        }
        nuint newSize = (required + 0xffffu) & ~(nuint)0xffffu;
        int r = CuBlasInterop.CudaMalloc(out _q81BufB, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc(q8_1 scratch B, {newSize} B) failed: {r}");
        _q81BufBSize = newSize;
    }

    public void AddInPlace(Tensor dst, Tensor src)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n  = (int)dst.ElementCount;
        nint p0 = GetDevPtr(dst);
        nint p1 = GetDevPtr(src);
        int  p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_addKernel, n, args);
    }

    /// <summary>Element-wise multiply: output[i] = a[i] * b[i]. Tensors must be 1-D with matching element counts.</summary>
    public void ElementwiseMul(Tensor output, Tensor a, Tensor b)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long n = output.ElementCount;
        if (a.ElementCount != n || b.ElementCount != n)
            throw new ArgumentException(
                $"ElementwiseMul element-count mismatch: output={n} a={a.ElementCount} b={b.ElementCount}.");

        nint oPtr = GetDevPtr(output);
        nint aPtr = GetDevPtr(a);
        nint bPtr = GetDevPtr(b);
        int  pN = (int)n;
        nint* args = stackalloc nint[4]
        {
            (nint)(&oPtr), (nint)(&aPtr), (nint)(&bPtr), (nint)(&pN)
        };
        uint grid = (uint)(((int)n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_mulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(mul) failed: {r}");
    }

    public void RmsNorm(Tensor output, Tensor x, Tensor weight, float eps = 1e-5f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        nint wPtr = GetDevPtr(weight);
        nint yPtr = GetDevPtr(output);
        int   pN = n;
        float pE = eps;
        nint* args = stackalloc nint[5]
        {
            (nint)(&xPtr), (nint)(&wPtr), (nint)(&yPtr),
            (nint)(&pN),   (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_rmsNormKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rmsnorm) failed: {r}");
    }

    /// <summary>
    /// Batched RmsNorm over <paramref name="nTok"/> rows (issue #111). <paramref name="x"/>
    /// and <paramref name="output"/> are <c>[nTok × dim]</c> contiguous; <paramref name="weight"/>
    /// is shared across rows. One block per token runs the identical reduction as
    /// <see cref="RmsNorm"/>, so output is bit-identical to nTok sequential calls.
    /// </summary>
    public void RmsNormBatched(Tensor output, Tensor x, Tensor weight, int nTok, int dim, float eps = 1e-5f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xPtr = GetDevPtr(x);
        nint wPtr = GetDevPtr(weight);
        nint yPtr = GetDevPtr(output);
        int   pN = dim, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr), (nint)(&wPtr), (nint)(&yPtr),
            (nint)(&pN), (nint)(&pE), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_rmsNormBatchedKernel, (uint)nTok, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rmsnorm_batched) failed: {r}");
    }

    /// <summary>Per-head RMS norm with learned weights (Qwen3 / OLMoE QK norm).
    /// <paramref name="perChannelWeight"/> false → weight is shared <c>[headDim]</c> vector
    /// applied identically to every head (Qwen3); true → weight is
    /// <c>[numHeads * headDim]</c> with one slice per head (OLMoE).</summary>
    public void HeadNorm(Tensor data, Tensor weight, int numHeads, int headDim,
        float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        nint wPtr = GetDevPtr(weight);
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        int  pWS = perChannelWeight ? headDim : 0;
        nint* args = stackalloc nint[6]
        {
            (nint)(&dPtr), (nint)(&wPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pWS)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm) failed: {r}");
    }

    /// <summary>
    /// Batched per-head RmsNorm over <paramref name="nTok"/> rows (issue #111).
    /// <paramref name="data"/> is <c>[nTok × numHeads × headDim]</c>; grid = (numHeads, nTok).
    /// Bit-identical to nTok sequential <see cref="HeadNorm"/> calls.
    /// </summary>
    public void HeadNormBatched(Tensor data, Tensor weight, int numHeads, int headDim,
        int nTok, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        nint wPtr = GetDevPtr(weight);
        int  pHD = headDim, pNH = numHeads, pWS = perChannelWeight ? headDim : 0, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[7]
        {
            (nint)(&dPtr), (nint)(&wPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pWS), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_batched) failed: {r}");
    }

    /// <summary>
    /// Dual per-head RmsNorm of Q and K in one launch (Gemma 4 QK-norm). Grid covers
    /// <c>numHeads + numKvHeads</c> blocks; the first <c>numHeads</c> normalize Q with
    /// <paramref name="qWeight"/>, the rest K with <paramref name="kWeight"/>. Per block
    /// bit-identical to <see cref="HeadNorm"/>.
    /// </summary>
    public void HeadNormQk(Tensor qData, Tensor qWeight, Tensor kData, Tensor kWeight,
        int numHeads, int numKvHeads, int headDim, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(qData), qwP = GetDevPtr(qWeight), kP = GetDevPtr(kData), kwP = GetDevPtr(kWeight);
        int pHD = headDim, pNH = numHeads, pNKV = numKvHeads, pWS = perChannelWeight ? headDim : 0;
        float pE = eps;
        nint* args = stackalloc nint[9]
        {
            (nint)(&qP), (nint)(&qwP), (nint)(&kP), (nint)(&kwP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pNKV), (nint)(&pE), (nint)(&pWS)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormQkKernel, (uint)(numHeads + numKvHeads), 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_qk) failed: {r}");
    }

    /// <summary>Batched <see cref="HeadNormQk"/> over <paramref name="nTok"/> tokens.</summary>
    public void HeadNormQkBatched(Tensor qData, Tensor qWeight, Tensor kData, Tensor kWeight,
        int numHeads, int numKvHeads, int headDim, int nTok, float eps = 1e-6f, bool perChannelWeight = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(qData), qwP = GetDevPtr(qWeight), kP = GetDevPtr(kData), kwP = GetDevPtr(kWeight);
        int pHD = headDim, pNH = numHeads, pNKV = numKvHeads, pWS = perChannelWeight ? headDim : 0, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&qwP), (nint)(&kP), (nint)(&kwP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pNKV), (nint)(&pE), (nint)(&pWS), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormQkBatchedKernel, (uint)(numHeads + numKvHeads), (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_qk_batched) failed: {r}");
    }

    /// <summary>Per-head L2 normalize (no learned weights). Llama-4 style.</summary>
    public void HeadNormPure(Tensor data, int numHeads, int headDim, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        nint* args = stackalloc nint[4]
        {
            (nint)(&dPtr),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormPureKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_pure) failed: {r}");
    }

    /// <summary>Batched per-head L2 normalize (no learned weights) over N tokens.
    /// <paramref name="data"/> is token-major [nTok × numHeads × headDim]; per
    /// (head, token) bit-identical to <see cref="HeadNormPure"/>. Used for the
    /// Gemma 4 12B k_eq_v V-norm in the batched-trunk prefill (issue #124).</summary>
    public void HeadNormPureBatched(Tensor data, int numHeads, int headDim, int nTok, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dPtr = GetDevPtr(data);
        int  pHD = headDim, pNH = numHeads, pNT = nTok;
        float pE = eps;
        nint* args = stackalloc nint[5]
        {
            (nint)(&dPtr), (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pNT)
        };
        int r = NvrtcInterop.LaunchKernel(_headNormPureBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(head_norm_pure_batched) failed: {r}");
    }

    public void Softmax(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xPtr), (nint)(&pN) };
        int r = NvrtcInterop.LaunchKernel(_softmaxKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(softmax) failed: {r}");
    }

    public void Sigmoid(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_sigmoidKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(sigmoid) failed: {r}");
    }

    /// <summary>Fused SiLU(gate) * up in-place into gate.</summary>
    public void SiLuMul(Tensor gate, Tensor up)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)gate.ElementCount;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pN = n;
        nint* args = stackalloc nint[3] { (nint)(&gPtr), (nint)(&uPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_siluMulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(silu_mul) failed: {r}");
    }

    /// <summary>Fused tanh-approximate GELU(gate) * up in-place into gate.
    /// Gemma-style FFN activation. Element-count of <paramref name="gate"/>
    /// must equal element-count of <paramref name="up"/>.</summary>
    public void GeluTanhMul(Tensor gate, Tensor up)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)gate.ElementCount;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pN = n;
        nint* args = stackalloc nint[3] { (nint)(&gPtr), (nint)(&uPtr), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_geluTanhMulKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gelu_tanh_mul) failed: {r}");
    }

    /// <summary>
    /// Strided-up <see cref="GeluTanhMul"/> over <paramref name="nTok"/> tokens:
    /// <paramref name="gate"/> is <c>[nTok × width]</c> contiguous; the up operand for
    /// token t is at <c>up + t*upStride + upOffset</c>. Lets batched PLE inject the
    /// per-layer slice of a <c>[nTok × (L*pleWidth)]</c> projection buffer without a
    /// gather. Per element bit-identical to <see cref="GeluTanhMul"/>.
    /// </summary>
    public void GeluTanhMulStrided(Tensor gate, Tensor up, int width, long upStride, long upOffset, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long total = (long)nTok * width;
        nint gPtr = GetDevPtr(gate);
        nint uPtr = GetDevPtr(up);
        int  pW = width, pNT = nTok;
        long pStride = upStride, pOff = upOffset;
        nint* args = stackalloc nint[6]
        {
            (nint)(&gPtr), (nint)(&uPtr), (nint)(&pW), (nint)(&pStride), (nint)(&pOff), (nint)(&pNT)
        };
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_geluTanhMulStridedKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gelu_tanh_mul_strided) failed: {r}");
    }

    /// <summary>
    /// In-place final-logit softcap: <c>x[i] = tanh(x[i] / cap) * cap</c>.
    /// Used by Gemma 4 to clip extreme logits before sampling.
    /// </summary>
    public void SoftcapInPlace(Tensor x, float cap)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        int  pN = n;
        float pCap = cap;
        nint* args = stackalloc nint[3] { (nint)(&xPtr), (nint)(&pN), (nint)(&pCap) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_softcapKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(softcap_inplace) failed: {r}");
    }

    // #219 GPU greedy argmax scratch (lazily allocated on first Argmax call): the per-block
    // (value, index) partials and the 8-byte (index-bits, value) output. The index buffer is a
    // Float32 tensor used as raw int storage — the kernel writes/reads it as int*.
    private const int ArgmaxBlocks = 256;
    private const int ArgmaxThreads = 256;
    private Tensor? _argmaxPartialVal;
    private Tensor? _argmaxPartialIdx;
    private Tensor? _argmaxOut;

    /// <summary>
    /// Issue #219: greedy argmax of <paramref name="logits"/> computed on-device, returning the
    /// winning <c>(index, value)</c> after a single 8-byte download. Replaces a full-vocab D2H +
    /// host scan on the greedy decode path. Bit-exact with <c>Sampler.Greedy</c> for finite logits,
    /// including the lowest-index tie-break. Launched on the same stream as the producing forward
    /// pass, so it sees the just-computed logits; the final download synchronizes the stream.
    /// </summary>
    public (int Index, float Value) Argmax(Tensor logits)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)logits.ElementCount;
        if (n <= 0) throw new ArgumentException("Argmax requires a non-empty logits tensor.", nameof(logits));

        _argmaxPartialVal ??= Allocate(TensorShape.D1(ArgmaxBlocks));
        _argmaxPartialIdx ??= Allocate(TensorShape.D1(ArgmaxBlocks));
        _argmaxOut        ??= Allocate(TensorShape.D1(2));

        int blocks = Math.Min(ArgmaxBlocks, (n + ArgmaxThreads - 1) / ArgmaxThreads);

        nint lPtr = GetDevPtr(logits);
        nint pvPtr = GetDevPtr(_argmaxPartialVal);
        nint piPtr = GetDevPtr(_argmaxPartialIdx);
        int pN = n;
        nint* a1 = stackalloc nint[4] { (nint)(&lPtr), (nint)(&pN), (nint)(&pvPtr), (nint)(&piPtr) };
        int r1 = NvrtcInterop.LaunchKernel(_argmaxPartialKernel, (uint)blocks, 1, 1, ArgmaxThreads, 1, 1, 0, _stream, a1, null);
        if (r1 != 0) throw new InvalidOperationException($"cuLaunchKernel(llm_argmax_partial) failed: {r1}");

        nint outPtr = GetDevPtr(_argmaxOut);
        int pNumParts = blocks;
        nint* a2 = stackalloc nint[4] { (nint)(&pvPtr), (nint)(&piPtr), (nint)(&pNumParts), (nint)(&outPtr) };
        int r2 = NvrtcInterop.LaunchKernel(_argmaxFinalKernel, 1, 1, 1, ArgmaxThreads, 1, 1, 0, _stream, a2, null);
        if (r2 != 0) throw new InvalidOperationException($"cuLaunchKernel(llm_argmax_final) failed: {r2}");

        // 8-byte D2H + StreamSynchronize (drains the two kernels above). out[0] = index (int bits),
        // out[1] = value (float).
        Span<float> result = stackalloc float[2];
        Download(_argmaxOut, result);
        return (BitConverter.SingleToInt32Bits(result[0]), result[1]);
    }

    // #219 batched-argmax scratch: a [maxRows*2] device output + matching host buffer, grown on demand.
    private Tensor? _argmaxRowsOut;
    private float[] _argmaxRowsHost = [];

    /// <summary>
    /// Issue #219: per-row greedy argmax over a packed <c>[rows × rowStride]</c> logits buffer
    /// (the MTP / speculative verify positions), returning one <c>(index, value)</c> per row after
    /// a single <c>rows*8</c>-byte download instead of the full <c>rows × vocab</c> D2H. Each row's
    /// valid length is <paramref name="validLen"/> (<= <paramref name="rowStride"/>); the argmax is
    /// over <c>[0, validLen)</c>. Same lowest-index tie-break and bit-exactness as <see cref="Argmax"/>.
    /// </summary>
    public (int Index, float Value)[] ArgmaxRows(Tensor logits, int rows, int validLen, int rowStride)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (rows <= 0) return [];

        if (_argmaxRowsOut is null || _argmaxRowsOut.ElementCount < rows * 2)
        {
            if (_argmaxRowsOut is { } prev) Free(prev);   // return the smaller rental before growing
            _argmaxRowsOut = Allocate(TensorShape.D1(rows * 2));
            _argmaxRowsHost = new float[rows * 2];
        }

        nint lPtr = GetDevPtr(logits);
        nint outPtr = GetDevPtr(_argmaxRowsOut);
        int pN = validLen;
        int pStride = rowStride;
        nint* args = stackalloc nint[4] { (nint)(&lPtr), (nint)(&pN), (nint)(&pStride), (nint)(&outPtr) };
        int r = NvrtcInterop.LaunchKernel(_argmaxRowsKernel, (uint)rows, 1, 1, ArgmaxThreads, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(llm_argmax_rows) failed: {r}");

        Download(_argmaxRowsOut, _argmaxRowsHost.AsSpan(0, rows * 2));   // rows*8 bytes + sync
        var result = new (int, float)[rows];
        for (int i = 0; i < rows; i++)
            result[i] = (BitConverter.SingleToInt32Bits(_argmaxRowsHost[i * 2]), _argmaxRowsHost[i * 2 + 1]);
        return result;
    }

    // IComputeBackend contract (#314): standalone in-place SiLU. The fused SwiGLU
    // path still prefers SiLuMul(gate, up); this delegates to the existing NVRTC
    // in-place kernel so the interface is honored uniformly across backends.
    public void SiLU(Tensor x) => SiLUInPlace(x);

    public void RoPE(Tensor x, int position, int headDim, float ropeTheta = 10000f, bool neox = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (headDim / 2);

        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[5]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        nint kernel = neox ? _ropeNeoxKernel : _ropeInterleavedKernel;
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[5];
            av[0] = xPtr; av[1] = pNH; av[2] = pHD; av[3] = pPos; av[4] = GraphFloatBits(pT);
            TrackPositionNode(kernel, grid, 1, 1, 256, 1, 1, 0, av, [(3, GraphPosKind.Position, 0)]);
        }
    }

    /// <summary>
    /// Partial NEOX RoPE: rotate the first <paramref name="ropeDim"/> dimensions of every
    /// head; leave dims <c>[ropeDim, headDim)</c> untouched. Used by qwen35moe attention
    /// (ropeDim=64, headDim=256). Mirrors the CPU reference
    /// <see cref="Cpu.SimdKernels.ApplyRoPECachedNeoxPartial"/>.
    /// </summary>
    public void RoPEPartial(Tensor x, int position, int headDim, int ropeDim, float ropeTheta, bool neox)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (!neox)
            throw new ArgumentException("RoPEPartial currently supports only neox=true.", nameof(neox));
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim.", nameof(ropeDim));

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (ropeDim / 2);

        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pRD = ropeDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pRD), (nint)(&pPos), (nint)(&pT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxPartialKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_partial) failed: {r}");
    }

    /// <summary>
    /// Batched partial NEOX RoPE over <paramref name="nTok"/> rows (issue #111). Token t
    /// rotates at position <paramref name="basePosition"/> + t (prefill assigns contiguous
    /// positions). <paramref name="x"/> is <c>[nTok × numHeads × headDim]</c>. Bit-identical
    /// to nTok sequential <see cref="RoPEPartial"/> calls at the matching positions.
    /// </summary>
    public void RoPEPartialBatched(Tensor x, int basePosition, int headDim, int ropeDim,
        float ropeTheta, int numHeads, int nTok, bool neox)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (!neox)
            throw new ArgumentException("RoPEPartialBatched currently supports only neox=true.", nameof(neox));
        if (ropeDim <= 0 || (ropeDim & 1) != 0)
            throw new ArgumentException("ropeDim must be a positive even number.", nameof(ropeDim));
        if (ropeDim > headDim)
            throw new ArgumentException("ropeDim must be <= headDim.", nameof(ropeDim));

        int totalPairs = numHeads * (ropeDim / 2);
        nint xPtr = GetDevPtr(x);
        int  pNH = numHeads, pHD = headDim, pRD = ropeDim, pPos = basePosition, pNT = nTok;
        float pT = ropeTheta;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pRD), (nint)(&pPos), (nint)(&pT), (nint)(&pNT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxPartialBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_partial_batched) failed: {r}");
    }

    /// <summary>
    /// NEOX RoPE with per-half-dim frequency factors (Gemma 4 global layers /
    /// Gemma-3n). <paramref name="freqFactors"/> is a <c>head_dim/2</c> F32 device
    /// vector that divides each pair's frequency; mirrors llama.cpp
    /// <c>gemma4.cpp:191</c> passing <c>rope_freqs.weight</c> only for non-SWA
    /// layers and the CPU <see cref="Cpu.SimdKernels.BuildRopeTable"/>
    /// globalFreqFactors path.
    /// </summary>
    public void RoPEWithFactors(Tensor x, int position, int headDim, float ropeTheta, Tensor freqFactors)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentException("headDim must be a positive even number.", nameof(headDim));
        if (freqFactors.ElementCount != headDim / 2)
            throw new ArgumentException(
                $"RoPEWithFactors: freqFactors length {freqFactors.ElementCount} != headDim/2 ({headDim / 2}).",
                nameof(freqFactors));

        int numHeads = (int)(x.ElementCount / headDim);
        int totalPairs = numHeads * (headDim / 2);

        nint xPtr = GetDevPtr(x);
        nint fPtr = GetDevPtr(freqFactors);
        int  pNH = numHeads, pHD = headDim, pPos = position;
        float pT = ropeTheta;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT),
            (nint)(&fPtr)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxWithFactorsKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_with_factors) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[6];
            av[0] = xPtr; av[1] = pNH; av[2] = pHD; av[3] = pPos;
            av[4] = GraphFloatBits(pT); av[5] = fPtr;
            TrackPositionNode(_ropeNeoxWithFactorsKernel, grid, 1, 1, 256, 1, 1, 0, av,
                [(3, GraphPosKind.Position, 0)]);
        }
    }

    /// <summary>
    /// Batched <see cref="RoPEWithFactors"/> over <paramref name="nTok"/> tokens (Gemma 4
    /// global layers in batched-trunk prefill). Token <c>t</c> uses position
    /// <c>basePosition + t</c>; <paramref name="x"/> is <c>[nTok × numHeads*headDim]</c>.
    /// Bit-identical per row to the per-token kernel.
    /// </summary>
    public void RoPEWithFactorsBatched(Tensor x, int basePosition, int headDim,
        float ropeTheta, Tensor freqFactors, int numHeads, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentException("headDim must be a positive even number.", nameof(headDim));
        if (freqFactors.ElementCount != headDim / 2)
            throw new ArgumentException(
                $"RoPEWithFactorsBatched: freqFactors length {freqFactors.ElementCount} != headDim/2 ({headDim / 2}).",
                nameof(freqFactors));

        int totalPairs = numHeads * (headDim / 2);
        nint xPtr = GetDevPtr(x);
        nint fPtr = GetDevPtr(freqFactors);
        int  pNH = numHeads, pHD = headDim, pPos = basePosition, pNT = nTok;
        float pT = ropeTheta;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pPos), (nint)(&pT),
            (nint)(&fPtr), (nint)(&pNT)
        };
        uint grid = (uint)((totalPairs + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_ropeNeoxWithFactorsBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_neox_with_factors_batched) failed: {r}");
    }

    /// <summary>
    /// Fused <c>x[i] *= sigmoid(gate[i])</c> in-place. Replaces a Sigmoid + ElementwiseMul
    /// pair for the qwen35moe GLU attention gate.
    /// </summary>
    public void SigmoidMulInPlace(Tensor x, Tensor gate)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (x.ElementCount != gate.ElementCount)
            throw new ArgumentException(
                $"SigmoidMulInPlace element-count mismatch: x={x.ElementCount} gate={gate.ElementCount}.");

        int n = (int)x.ElementCount;
        nint xPtr = GetDevPtr(x);
        nint gPtr = GetDevPtr(gate);
        int  pN = n;
        nint* args = stackalloc nint[3]
        {
            (nint)(&xPtr), (nint)(&gPtr), (nint)(&pN)
        };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_sigmoidMulInPlaceKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(sigmoid_mul_inplace) failed: {r}");
    }

    /// <summary>
    /// Strided de-interleave of qwen35moe's GLU-gated Q output. Input
    /// <paramref name="qg"/> is laid out per head as <c>[Q[headDim] ‖ G[headDim]]</c>
    /// (output stride = <c>2*headDim</c>); this splits into contiguous
    /// <c>q[numHeads*headDim]</c> and <c>g[numHeads*headDim]</c>.
    /// </summary>
    public void SplitQG(Tensor q, Tensor g, Tensor qg, int numHeads, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long expected = (long)numHeads * headDim * 2;
        if (qg.ElementCount != expected)
            throw new ArgumentException(
                $"SplitQG: qg.ElementCount {qg.ElementCount} != numHeads*headDim*2 ({expected}).");
        long perOut = (long)numHeads * headDim;
        if (q.ElementCount != perOut || g.ElementCount != perOut)
            throw new ArgumentException(
                $"SplitQG: q/g element counts must each equal numHeads*headDim ({perOut}); got q={q.ElementCount}, g={g.ElementCount}.");

        nint qgPtr = GetDevPtr(qg);
        nint qPtr  = GetDevPtr(q);
        nint gPtr  = GetDevPtr(g);
        int  pNH = numHeads, pHD = headDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&qgPtr), (nint)(&qPtr), (nint)(&gPtr),
            (nint)(&pNH), (nint)(&pHD)
        };
        int total = numHeads * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_splitQgKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(split_qg) failed: {r}");
    }

    /// <summary>
    /// Batched <see cref="SplitQG"/> over <paramref name="nTok"/> rows (issue #111).
    /// <paramref name="qg"/> is <c>[nTok × numHeads × headDim × 2]</c>; <paramref name="q"/>
    /// and <paramref name="g"/> are <c>[nTok × numHeads × headDim]</c>. Bit-identical to
    /// nTok sequential <see cref="SplitQG"/> calls.
    /// </summary>
    public void SplitQGBatched(Tensor q, Tensor g, Tensor qg, int numHeads, int headDim, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qgPtr = GetDevPtr(qg);
        nint qPtr  = GetDevPtr(q);
        nint gPtr  = GetDevPtr(g);
        int  pNH = numHeads, pHD = headDim, pNT = nTok;
        nint* args = stackalloc nint[6]
        {
            (nint)(&qgPtr), (nint)(&qPtr), (nint)(&gPtr),
            (nint)(&pNH), (nint)(&pHD), (nint)(&pNT)
        };
        int total = numHeads * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_splitQgBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(split_qg_batched) failed: {r}");
    }

    /// <summary>Write fresh K and V vectors for one token into the layer KV cache at <paramref name="position"/>.</summary>
    public void KvAppend(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
                         int kvDim, int position, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kInput);
        nint vPtr = GetDevPtr(vInput);
        nint kcP  = GetDevPtr(kCache);
        nint vcP  = GetDevPtr(vCache);
        int  pKD = kvDim, pPos = position, pMSL = maxSeqLen;
        nint* args = stackalloc nint[7]
        {
            (nint)(&kPtr), (nint)(&vPtr),
            (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pPos), (nint)(&pMSL)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[7];
            av[0] = kPtr; av[1] = vPtr; av[2] = kcP; av[3] = vcP;
            av[4] = pKD; av[5] = pPos; av[6] = pMSL;
            TrackPositionNode(_kvAppendKernel, grid, 1, 1, 256, 1, 1, 0, av, [(5, GraphPosKind.Position, 0)]);
        }
    }

    /// <summary>
    /// Scaled dot-product attention with GQA support. Output: [numHeads * headDim].
    ///
    /// When <c>seqLen ≤ 4096</c> the kernel keeps per-position scores in shared memory and
    /// <paramref name="scoresScratch"/> is ignored. Above that threshold the kernel spills
    /// scores to <paramref name="scoresScratch"/>, which must have room for
    /// <c>numHeads × maxSeqLen</c> floats. Passing a non-null scratch always works.
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void Attention(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                          Tensor? scoresScratch,
                          int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                          float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSL), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[11];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD; av[8] = pSL; av[9] = pMSL;
            av[10] = GraphFloatBits(pScale);
            // pSL = seqLen = position + 1 in the decode path.
            TrackPositionNode(_attentionKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.PositionPlus1, 0)]);
        }
    }

    /// <summary>Flash-decoding split-KV chunk (KV tokens per split block). Must match
    /// the kernel's <c>SPLITKV_CHUNK</c>.</summary>
    public const int SplitKvChunk = 512;

    /// <summary>Max KV splits per head — the combine kernel's <c>SPLITKV_MAX_SPLITS</c>
    /// shared array bound. Callers must keep ceil(maxSeqLen/<see cref="SplitKvChunk"/>) ≤ this
    /// (i.e. maxSeqLen ≤ 131072); <see cref="AttentionSplitKv"/> enforces it.</summary>
    public const int SplitKvMaxSplits = 256;

    /// <summary>
    /// Flash-decoding decode attention (issue #235). Splits each head's KV sequence
    /// into <see cref="SplitKvChunk"/>-sized chunks across <c>numHeads × nSplits</c>
    /// blocks (nSplits = ceil(maxSeqLen/chunk)) so the O(ctx)/token KV read parallelizes
    /// across the SMs instead of the single-block-per-head <see cref="Attention"/>. Each
    /// block emits an un-normalized online-softmax partial (m_i, l_i, Õ_i) into
    /// <paramref name="partialO"/>/<paramref name="partialMeta"/>; the combine kernel
    /// LSE-merges them into <paramref name="output"/>. Argmax-stable, not bit-identical to
    /// <see cref="Attention"/> (the combine reorders the softmax reduction). The grid is
    /// fixed at capture and only seqLen updates per graph replay (out-of-range splits
    /// early-exit), so it is CUDA-graph-capturable. fp32/bf16/q8_0 via <paramref name="kvDType"/>.
    ///
    /// <paramref name="grouped"/> (issue #237): one block handles a KV head's whole query
    /// group (grid <c>numKvHeads × nSplits</c>), loading each K/V slice once and reusing it
    /// across the <c>G = numHeads/numKvHeads</c> query heads — ~G× less KV HBM read on the
    /// bandwidth-bound long-ctx decode, at the cost of G× fewer blocks. Emits the SAME
    /// per-query-head partials (combine + layout unchanged). Requires <c>numHeads % numKvHeads
    /// == 0</c> and <c>G ≤ 8</c>.
    /// </summary>
    public void AttentionSplitKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                                 Tensor partialO, Tensor partialMeta, DType kvDType,
                                 int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                                 float attnScale = -1f, bool grouped = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int nSplits = (maxSeqLen + SplitKvChunk - 1) / SplitKvChunk;
        // The combine kernel sizes its per-head rescale array at SPLITKV_MAX_SPLITS; exceeding
        // it would overrun shared memory. The caller gates maxSeqLen ≤ 131072 so this can't
        // trigger — fail loud rather than silently corrupt if a future caller forgets.
        if (nSplits > SplitKvMaxSplits)
            throw new ArgumentOutOfRangeException(nameof(maxSeqLen), maxSeqLen,
                $"AttentionSplitKv: nSplits {nSplits} exceeds SplitKvMaxSplits {SplitKvMaxSplits} " +
                $"(maxSeqLen must be ≤ {SplitKvMaxSplits * SplitKvChunk}).");

        // GQA head-sharing (#237): G query heads per block; grid X = numKvHeads, one extern
        // shared float per (group head × chunk slot). G ≤ 8 (kernel's dots/acc/po_base arrays).
        // numKvHeads ≥ 2 guards the division below (a caller passing grouped with numKvHeads ≤ 0
        // would otherwise DivideByZero before the guard); the guard's `< 2` short-circuits the `%`.
        int group = (grouped && numKvHeads >= 2) ? numHeads / numKvHeads : 1;
        if (grouped && (numKvHeads < 2 || numHeads % numKvHeads != 0 || group < 2 || group > 8))
            throw new ArgumentOutOfRangeException(nameof(numKvHeads), numKvHeads,
                $"Grouped split-KV requires numKvHeads ≥ 2, numHeads % numKvHeads == 0, and G ∈ [2,8] " +
                $"(got numHeads={numHeads}, numKvHeads={numKvHeads}).");
        uint gridX = grouped ? (uint)numKvHeads : (uint)numHeads;
        uint sharedBytes = grouped ? (uint)(group * SplitKvChunk * sizeof(float)) : 0u;
        nint splitKern = (grouped, kvDType) switch
        {
            (false, DType.BFloat16) => _attentionSplitKvBf16Kernel,
            (false, DType.Q8_0)     => _attentionSplitKvQ8Kernel,
            (false, DType.Float32)  => _attentionSplitKvKernel,
            (true,  DType.BFloat16) => _attentionSplitKvGroupedBf16Kernel,
            (true,  DType.Q8_0)     => _attentionSplitKvGroupedQ8Kernel,
            (true,  DType.Float32)  => _attentionSplitKvGroupedKernel,
            _ => throw new ArgumentOutOfRangeException(nameof(kvDType), kvDType,
                "AttentionSplitKv supports fp32 / bf16 / q8_0 K/V caches only."),
        };

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint poP = GetDevPtr(partialO);
        nint pmP = GetDevPtr(partialMeta);
        nint oP = GetDevPtr(output);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pNS = nSplits;
        float pScale = attnScale;

        // Split kernel: q, k, v, partial_o, partial_meta, num_heads, num_kv_heads, head_dim, seq_len, n_splits, attn_scale
        nint* sargs = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&poP), (nint)(&pmP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSL), (nint)(&pNS), (nint)(&pScale)
        };
        int rs = NvrtcInterop.LaunchKernel(splitKern, gridX, (uint)nSplits, 1, 256, 1, 1, sharedBytes, _stream, sargs, null);
        if (rs != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_splitkv {kvDType} grouped={grouped}) failed: {rs}");

        // Track the split kernel BEFORE the combine launch (single-leaf harvest); only
        // seqLen (arg 8) varies per replay — the fixed grid + early-exit handle the rest.
        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[11];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = poP; av[4] = pmP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD; av[8] = pSL; av[9] = pNS;
            av[10] = GraphFloatBits(pScale);
            TrackPositionNode(splitKern, gridX, (uint)nSplits, 1, 256, 1, 1, sharedBytes, av,
                [(8, GraphPosKind.PositionPlus1, 0)]);
        }

        // Combine kernel: partial_o, partial_meta, out, num_heads, head_dim, n_splits.
        // No per-replay-varying args → captured by stream capture, not tracked.
        nint* cargs = stackalloc nint[6]
        {
            (nint)(&poP), (nint)(&pmP), (nint)(&oP), (nint)(&pNH), (nint)(&pHD), (nint)(&pNS)
        };
        int rc = NvrtcInterop.LaunchKernel(_attentionCombineKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, cargs, null);
        if (rc != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_combine) failed: {rc}");
    }

    // ── Ragged-batched decode ops (issue #197) ─────────────────────────────
    //
    // One launch per op covers all N decode sequences: row t of the [N × dim]
    // activation buffer is processed at positions[t] against kCaches[t]/vCaches[t].
    // Per-sequence positions and cache base pointers ride in by-value struct kernel
    // parameters (capacity 16; larger batches chunk into ceil(N/16) launches), so
    // there is no device-side table, no host→device upload, and no sync on the hot
    // path. These methods issue direct launches only — batched decode never captures
    // a CUDA graph (CudaForwardPass.CreateCache disables graphs), so no Track*Node.

    /// <summary>Sequences per ragged launch — the by-value struct parameter capacity.</summary>
    private const int RaggedChunk = CudaRaggedKernels.ChunkCapacity;

    /// <summary>
    /// Ragged-batched RoPE (issue #197): row <c>t</c> of <paramref name="xAll"/>
    /// (<c>[N × numHeads*headDim]</c>) rotates at <c>positions[t]</c>. Per row
    /// bit-identical to the per-token <see cref="RoPE"/> at that position. Unlike the
    /// <c>*_batched</c> prefill RoPE kernels (consecutive <c>basePos+t</c>), every row
    /// carries its own arbitrary position — the batched-decode contract.
    /// </summary>
    public void RoPEBatchedRagged(Tensor xAll, ReadOnlySpan<int> positions,
        int numHeads, int headDim, float ropeTheta = 10000f, bool neox = false)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        int nTok = positions.Length;
        if (nTok == 0) return;
        if ((long)nTok * numHeads * headDim > xAll.ElementCount)
            throw new ArgumentException(
                $"RoPEBatchedRagged: x has {xAll.ElementCount} elements; need {(long)nTok * numHeads * headDim}.");

        int totalPairs = numHeads * (headDim / 2);
        uint gridX = (uint)((totalPairs + 255) / 256);
        nint kernel = neox ? _ropeNeoxRaggedKernel : _ropeInterleavedRaggedKernel;
        long rowBytes = (long)numHeads * headDim * sizeof(float);
        nint xBase = GetDevPtr(xAll);

        int* pos = stackalloc int[RaggedChunk];
        nint xPtr = 0;
        int  pNH = numHeads, pHD = headDim, pNT = 0;
        float pT = ropeTheta;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xPtr), (nint)(&pNH), (nint)(&pHD), (nint)pos, (nint)(&pT), (nint)(&pNT)
        };
        for (int s = 0; s < nTok; s += RaggedChunk)
        {
            int n = Math.Min(RaggedChunk, nTok - s);
            for (int t = 0; t < n; t++) pos[t] = positions[s + t];
            xPtr = xBase + (nint)((long)s * rowBytes);
            pNT = n;
            int r = NvrtcInterop.LaunchKernel(kernel, gridX, (uint)n, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(rope_ragged) failed: {r}");
        }
    }

    /// <summary>
    /// Ragged-batched KV append (issue #197): row <c>t</c> of the fp32
    /// <paramref name="kInputAll"/>/<paramref name="vInputAll"/> (<c>[N × kvDim]</c>)
    /// is stored into <c>kCaches[t]</c>/<c>vCaches[t]</c> at physical slot
    /// <c>slots[t] % maxSeqLen</c>. Per sequence bit-identical to the matching
    /// per-token append (<see cref="KvAppend"/> / <see cref="KvAppendBf16"/> /
    /// <see cref="KvAppendQ8_0"/> per <paramref name="kvDType"/>).
    /// <para><paramref name="slots"/> is the PHYSICAL cache slot, not the logical token position:
    /// for a SnapKV-compacted cache (#277) the caller passes <c>position - EvictedCount</c> so the
    /// new token lands in the compacted cache (RoPE still rotates at the logical position). When no
    /// sequence is evicted slot == position and this is the plain #197 append.</para>
    /// </summary>
    public void KvAppendBatchedRagged(
        Tensor kInputAll, Tensor vInputAll,
        ReadOnlySpan<Tensor> kCaches, ReadOnlySpan<Tensor> vCaches,
        ReadOnlySpan<int> slots, int kvDim, int maxSeqLen, DType kvDType = DType.Float32)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        int nTok = slots.Length;
        if (nTok == 0) return;
        if (kCaches.Length != nTok || vCaches.Length != nTok)
            throw new ArgumentException("KvAppendBatchedRagged: kCaches/vCaches/slots lengths must match.");
        nint kernel = kvDType switch
        {
            DType.Float32  => _kvAppendRaggedKernel,
            DType.BFloat16 => _kvAppendRaggedBf16Kernel,
            DType.Q8_0     => _kvAppendRaggedQ8Kernel,
            _ => throw new NotSupportedException($"KvAppendBatchedRagged: unsupported KV dtype {kvDType}."),
        };

        uint gridX = (uint)((kvDim + 255) / 256);
        long rowBytes = (long)kvDim * sizeof(float);
        nint kBase = GetDevPtr(kInputAll);
        nint vBase = GetDevPtr(vInputAll);

        nint* kPtrs = stackalloc nint[RaggedChunk];
        nint* vPtrs = stackalloc nint[RaggedChunk];
        int*  pos   = stackalloc int[RaggedChunk];
        nint kIn = 0, vIn = 0;
        int  pKD = kvDim, pMSL = maxSeqLen, pNT = 0;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kIn), (nint)(&vIn), (nint)kPtrs, (nint)vPtrs, (nint)pos,
            (nint)(&pKD), (nint)(&pMSL), (nint)(&pNT)
        };
        for (int s = 0; s < nTok; s += RaggedChunk)
        {
            int n = Math.Min(RaggedChunk, nTok - s);
            for (int t = 0; t < n; t++)
            {
                kPtrs[t] = GetDevPtr(kCaches[s + t]);
                vPtrs[t] = GetDevPtr(vCaches[s + t]);
                pos[t]   = slots[s + t];
            }
            kIn = kBase + (nint)((long)s * rowBytes);
            vIn = vBase + (nint)((long)s * rowBytes);
            pNT = n;
            int r = NvrtcInterop.LaunchKernel(kernel, gridX, (uint)n, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_ragged) failed: {r}");
        }
    }

    /// <summary>
    /// Ragged-batched single-query attention (issue #197): query row <c>t</c> of
    /// <paramref name="qAll"/> (<c>[N × numHeads*headDim]</c>) attends over
    /// <c>kCaches[t]</c>/<c>vCaches[t]</c> slots <c>[0, slots[t] + 1)</c> into
    /// row <c>t</c> of <paramref name="outputAll"/>. Grid is (numHeads, N): all N
    /// sequences' attention blocks run concurrently in one launch. Each (head, sequence)
    /// block keeps the per-token <see cref="Attention"/> kernel's exact reduction chain,
    /// so per sequence the output is bit-identical to the sequential call.
    ///
    /// <para><paramref name="slots"/> is the PHYSICAL last-slot index: the attended range is
    /// <c>[0, slots[t] + 1)</c>. For a SnapKV-compacted cache (#277) the caller passes
    /// <c>position - EvictedCount</c>, so each sequence attends over exactly its compacted length;
    /// when no sequence is evicted slot == position and this is the plain #197 attention.</para>
    ///
    /// When every <c>slots[t] + 1 ≤ 4096</c> scores stay in shared memory and
    /// <paramref name="scoresScratch"/> may be null; above that it must hold
    /// <c>N × numHeads × maxSeqLen</c> floats (per-sequence rows of the per-token
    /// kernel's <c>numHeads × maxSeqLen</c> layout).
    /// </summary>
    public void AttentionBatchedRagged(
        Tensor qAll, ReadOnlySpan<Tensor> kCaches, ReadOnlySpan<Tensor> vCaches,
        Tensor outputAll, Tensor? scoresScratch,
        int numHeads, int numKvHeads, int headDim,
        ReadOnlySpan<int> slots, int maxSeqLen, float attnScale = -1f, DType kvDType = DType.Float32)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        int nTok = slots.Length;
        if (nTok == 0) return;
        if (kCaches.Length != nTok || vCaches.Length != nTok)
            throw new ArgumentException("AttentionBatchedRagged: kCaches/vCaches/slots lengths must match.");
        nint kernel = kvDType switch
        {
            DType.Float32  => _attentionRaggedKernel,
            DType.BFloat16 => _attentionRaggedBf16Kernel,
            DType.Q8_0     => _attentionRaggedQ8Kernel,
            _ => throw new NotSupportedException($"AttentionBatchedRagged: unsupported KV dtype {kvDType}."),
        };

        nint ssBase = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int maxLen = 0;
        for (int i = 0; i < nTok; i++) maxLen = Math.Max(maxLen, slots[i] + 1);
        if (maxLen > 4096)
        {
            // Fail loud, not corrupt: a too-small/absent scratch would make the kernel
            // spill out of bounds (per-sequence rows, unlike the per-token kernel's
            // single numHeads × maxSeqLen block).
            if (scoresScratch is not { } scratch ||
                scratch.ElementCount < (long)nTok * numHeads * maxSeqLen)
                throw new ArgumentException(
                    $"AttentionBatchedRagged: seqLen {maxLen} > 4096 requires a scores scratch of " +
                    $"{nTok}×{numHeads}×{maxSeqLen} floats.");
        }

        long qRowBytes  = (long)numHeads * headDim * sizeof(float);
        long ssRowBytes = (long)numHeads * maxSeqLen * sizeof(float);
        nint qBase = GetDevPtr(qAll);
        nint oBase = GetDevPtr(outputAll);

        nint* kPtrs = stackalloc nint[RaggedChunk];
        nint* vPtrs = stackalloc nint[RaggedChunk];
        int*  pos   = stackalloc int[RaggedChunk];
        nint qPtr = 0, oPtr = 0, ssPtr = 0;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pNT = 0;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qPtr), (nint)kPtrs, (nint)vPtrs, (nint)(&oPtr), (nint)(&ssPtr),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)pos, (nint)(&pMSL), (nint)(&pScale), (nint)(&pNT)
        };
        for (int s = 0; s < nTok; s += RaggedChunk)
        {
            int n = Math.Min(RaggedChunk, nTok - s);
            for (int t = 0; t < n; t++)
            {
                kPtrs[t] = GetDevPtr(kCaches[s + t]);
                vPtrs[t] = GetDevPtr(vCaches[s + t]);
                pos[t]   = slots[s + t];
            }
            qPtr  = qBase + (nint)((long)s * qRowBytes);
            oPtr  = oBase + (nint)((long)s * qRowBytes);
            ssPtr = ssBase == nint.Zero ? nint.Zero : ssBase + (nint)((long)s * ssRowBytes);
            pNT = n;
            int r = NvrtcInterop.LaunchKernel(kernel, (uint)numHeads, (uint)n, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_ragged) failed: {r}");
        }
    }

    /// <summary>
    /// Broadcast bias add over <paramref name="nTok"/> rows (issue #197):
    /// <c>xAll[t][i] += bias[i]</c>. Replaces N per-row <see cref="AddInPlace"/>
    /// launches in the batched-decode attn-bias branch; one fp32 add per element,
    /// bit-identical to the per-row calls.
    /// </summary>
    public void AddBiasBatched(Tensor xAll, Tensor bias, int dim, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (nTok == 0) return;
        if ((long)dim * nTok > xAll.ElementCount)
            throw new ArgumentException(
                $"AddBiasBatched: x has {xAll.ElementCount} elements; need {(long)dim * nTok}.");
        if (bias.ElementCount < dim)
            throw new ArgumentException(
                $"AddBiasBatched: bias has {bias.ElementCount} elements; need {dim}.");

        nint xPtr = GetDevPtr(xAll);
        nint bPtr = GetDevPtr(bias);
        int  pD = dim, pNT = nTok;
        nint* args = stackalloc nint[4] { (nint)(&xPtr), (nint)(&bPtr), (nint)(&pD), (nint)(&pNT) };
        uint gridX = (uint)((dim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_addBiasRowsKernel, gridX, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(add_bias_rows) failed: {r}");
    }

    /// <summary>
    /// Sliding-window attention (Gemma 4 SWA layers). Iterates positions over
    /// <c>[max(0, position+1-windowSize), position+1)</c> instead of the full
    /// prefix. Per-layer <paramref name="headDim"/> is passed explicitly so a
    /// model with varying head_dim across layers (Gemma 4: 256 SWA / 512 global)
    /// dispatches correctly.
    ///
    /// When the effective windowed range ≤ 4096 the kernel keeps per-position
    /// scores in shared memory and <paramref name="scoresScratch"/> is ignored.
    /// Above that threshold the kernel spills to <paramref name="scoresScratch"/>
    /// which must have room for <c>numHeads × maxSeqLen</c> floats.
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwa(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                             Tensor? scoresScratch,
                             int position, int windowSize, int headDim,
                             int numHeads, int numKvHeads, int maxSeqLen,
                             float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int windowEnd   = position + 1;
        int windowStart = windowEnd - windowSize;
        if (windowStart < 0) windowStart = 0;

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int  pWS = windowStart, pWE = windowEnd, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pWS), (nint)(&pWE), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[12];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD;
            av[8] = pWS; av[9] = pWE; av[10] = pMSL; av[11] = GraphFloatBits(pScale);
            // windowStart = max(0, position+1-window); windowEnd = position+1.
            TrackPositionNode(_attentionSwaKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.SwaWindowStart, windowSize), (9, GraphPosKind.SwaWindowEnd, 0)]);
        }
    }

    /// <summary>
    /// Batched <see cref="AttentionSwa"/> over <paramref name="nTok"/> query tokens
    /// (Gemma 4 SWA layers in batched-trunk prefill). Query token <c>i</c> sits at
    /// absolute position <c>startPos+i</c> and attends its sliding window. The window
    /// bounds eff_seq ≤ <paramref name="windowSize"/>, so the shared-scores path always
    /// suffices (windowSize ≤ 4096 required). Bit-identical per (head, token) to the
    /// per-token kernel — no global scores scratch needed.
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwaBatched(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (windowSize <= 0 || windowSize > 4096)
            throw new ArgumentException(
                $"AttentionSwaBatched requires 0 < windowSize ≤ 4096 (shared-scores path); got {windowSize}.",
                nameof(windowSize));

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_batched) failed: {r}");
    }

    /// <summary>
    /// Issue #141 (attention): memory-efficient flash-attention prefill. Replaces
    /// <see cref="AttentionBatched"/> (global, <paramref name="windowSize"/>=0) and
    /// <see cref="AttentionSwaBatched"/> (sliding window, windowSize&gt;0) for the
    /// fp32-KV batched-trunk prefill. A block handles a tile of 8 queries of one head
    /// and streams K/V through shared-memory tiles with an online softmax, so each
    /// key is read from global once per 8 queries instead of once per query — cutting
    /// the scalar kernels' O(n²) (SWA: up to ~512×) redundant K/V traffic. GQA, causal,
    /// optional sliding window, per-layer <paramref name="headDim"/>. Matches the scalar
    /// kernels to fp tolerance (online softmax), not bit-exact.
    /// </summary>
    public void FlashAttentionPrefill(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (headDim > 512)
            throw new NotSupportedException(
                $"FlashAttentionPrefill supports head_dim ≤ 512 (16 dims/lane); got {headDim}.");

        // Pick the streaming K-tile so the per-key shared tile (K fp16 = headDim*2 B +
        // V fp32 = headDim*4 B = 6*headDim B) fits a 48 KB budget (no >48 KB opt-in).
        const int sharedBudget = 48 * 1024;
        int ktTile = sharedBudget / (6 * headDim);
        if (ktTile < 1) ktTile = 1;
        if (ktTile > 32) ktTile = 32;
        uint sharedBytes = (uint)(6 * ktTile * headDim);

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok, pKT = ktTile;
        float pScale = attnScale;
        nint* args = stackalloc nint[13]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pKT), (nint)(&pScale)
        };
        const int faQt = 16;   // FA_QT in the kernel (warps/block = K/V reuse factor)
        uint gy = (uint)((nTok + faQt - 1) / faQt);
        int r = NvrtcInterop.LaunchKernel(_flashAttnPrefillKernel, (uint)numHeads, gy, 1,
                                          (uint)(faQt * 32), 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(flash_attn_prefill) failed: {r}");
    }

    /// <summary>
    /// Issue #146 (test-only): one tensor-core <c>mma.sync.m16n8k16.f32.f16.f16.f32</c>
    /// computing <paramref name="cOut"/>[16×8] = <paramref name="aIn"/>[16×16] ·
    /// <paramref name="bIn"/>[16×8] on a single warp. Validates the A/B/C fragment
    /// layouts used by the TC flash kernel; inputs are fp32 (rounded to fp16 inside),
    /// so the result tracks an fp32 reference to fp16 tolerance, not bit-exactly.
    /// <paramref name="aIn"/> is row-major [16*16], <paramref name="bIn"/> is K-major
    /// [16*8] (bIn[k*8+n] = B[k][n]), <paramref name="cOut"/> is row-major [16*8].
    /// </summary>
    public void MmaTestM16N8K16(Tensor aIn, Tensor bIn, Tensor cOut)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint aP = GetDevPtr(aIn), bP = GetDevPtr(bIn), cP = GetDevPtr(cOut);
        nint* args = stackalloc nint[3] { (nint)(&aP), (nint)(&bP), (nint)(&cP) };
        int r = NvrtcInterop.LaunchKernel(_mmaTestM16N8K16Kernel, 1, 1, 1,
                                          32, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(mma_test_m16n8k16) failed: {r}");
    }

    /// <summary>
    /// Issue #146: tensor-core flash-attention prefill. Drop-in for
    /// <see cref="FlashAttentionPrefill"/> — same args/semantics — but runs both
    /// QK^T and P·V on the mma cores (one warp per 16-query tile, online softmax,
    /// O accumulated in shared fp32). Requires <paramref name="headDim"/> % 16 == 0
    /// and ≤ 512 (shared = 16·headDim·6 B = 48 KB at d=512). Matches the scalar
    /// kernels to fp tolerance (fp16 Q/K/V/P + online softmax), not bit-exact.
    /// </summary>
    public void FlashAttentionPrefillTc(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (headDim > 512 || (headDim & 15) != 0)
            throw new NotSupportedException(
                $"FlashAttentionPrefillTc requires head_dim ≤ 512 and a multiple of 16; got {headDim}.");

        uint sharedBytes = (uint)(16 * headDim * 6);   // O fp32 (×4) + K/V fp16 (×2)

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        uint gy = (uint)((nTok + 15) / 16);
        int r = NvrtcInterop.LaunchKernel(_flashAttnPrefillTcKernel, (uint)numHeads, gy, 1,
                                          32, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(flash_attn_prefill_tc) failed: {r}");
    }

    /// <summary>
    /// Issue #147: multi-warp / d-split tensor-core flash-attention prefill — same
    /// args/semantics as <see cref="FlashAttentionPrefillTc"/> but W=4 warps cooperate
    /// on each 16-query tile with the head dim split across them, so O is register-
    /// resident (no shared-O rescale) and occupancy rises ~10× (RTX 4070 Ti / Ada).
    /// Requires <paramref name="headDim"/> % 64 == 0 (W·16) and ≤ 512. Argmax-stable, not bit-exact.
    /// </summary>
    /// <param name="kvCacheType">K/V cache element dtype (issue #179): Float32 (default),
    /// BFloat16, or Q8_0. The matching templated thunk decodes each element to fp32 on
    /// load; args/shared/grid are identical across dtypes.</param>
    public void FlashAttentionPrefillTc2(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f,
        DType kvCacheType = DType.Float32)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");
        if (headDim > 512 || (headDim & 63) != 0)
            throw new NotSupportedException(
                $"FlashAttentionPrefillTc2 requires head_dim ≤ 512 and a multiple of 64 (W·16); got {headDim}.");

        const int w = 4;
        uint sharedBytes = (uint)(16 * headDim * 2 + w * 256 * 4);   // K/V fp16 + S scratch fp32

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        uint gy = (uint)((nTok + 15) / 16);
        // Explicit per-dtype routing — fail loud on an unexpected dtype rather than
        // silently reinterpreting a narrowed cache through the fp32 kernel (which would
        // stride 34-B q8_0 blocks as 4-B floats → garbage). fp32 is the only fall-through.
        nint kern = kvCacheType switch
        {
            DType.Float32  => _flashAttnPrefillTc2Kernel,
            DType.BFloat16 => _flashAttnPrefillTc2Bf16Kernel,
            DType.Q8_0     => _flashAttnPrefillTc2Q8Kernel,
            _ => throw new ArgumentOutOfRangeException(nameof(kvCacheType), kvCacheType,
                "FlashAttentionPrefillTc2 supports fp32 / bf16 / q8_0 K/V caches only."),
        };
        int r = NvrtcInterop.LaunchKernel(kern, (uint)numHeads, gy, 1,
                                          (uint)(w * 32), 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(flash_attn_prefill_tc2 {kvCacheType}) failed: {r}");
    }

    /// <summary>
    /// Bf16-store variant of <see cref="KvAppend"/>. Inputs stay fp32; the K/V
    /// cache tensors must be <see cref="DType.BFloat16"/>-allocated (half the
    /// element count of an fp32 cache). See issue #27.
    /// </summary>
    public void KvAppendBf16(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
                             int kvDim, int position, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kInput);
        nint vPtr = GetDevPtr(vInput);
        nint kcP  = GetDevPtr(kCache);
        nint vcP  = GetDevPtr(vCache);
        int  pKD = kvDim, pPos = position, pMSL = maxSeqLen;
        nint* args = stackalloc nint[7]
        {
            (nint)(&kPtr), (nint)(&vPtr),
            (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pPos), (nint)(&pMSL)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBf16Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_bf16) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[7];
            av[0] = kPtr; av[1] = vPtr; av[2] = kcP; av[3] = vcP;
            av[4] = pKD; av[5] = pPos; av[6] = pMSL;
            TrackPositionNode(_kvAppendBf16Kernel, grid, 1, 1, 256, 1, 1, 0, av, [(5, GraphPosKind.Position, 0)]);
        }
    }

    /// <summary>
    /// Bf16-read variant of <see cref="Attention"/>. K/V cache tensors must be
    /// <see cref="DType.BFloat16"/>; query, output, and the score scratch stay
    /// fp32. Arithmetic precision matches the fp32 kernel — only the cache
    /// footprint changes.
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBf16(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                              Tensor? scoresScratch,
                              int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                              float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSL), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionBf16Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_bf16) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[11];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD; av[8] = pSL; av[9] = pMSL;
            av[10] = GraphFloatBits(pScale);
            TrackPositionNode(_attentionBf16Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.PositionPlus1, 0)]);
        }
    }

    /// <summary>
    /// Bf16-read variant of <see cref="AttentionSwa"/> (issue #179). K/V cache
    /// tensors must be <see cref="DType.BFloat16"/>; query, output, and score
    /// scratch stay fp32. Same SWA ring, GQA, per-layer head_dim, and graph-capture
    /// position tracking as the fp32 kernel — only the cache footprint is halved.
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwaBf16(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                                 Tensor? scoresScratch,
                                 int position, int windowSize, int headDim,
                                 int numHeads, int numKvHeads, int maxSeqLen,
                                 float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int windowEnd   = position + 1;
        int windowStart = windowEnd - windowSize;
        if (windowStart < 0) windowStart = 0;

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int  pWS = windowStart, pWE = windowEnd, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pWS), (nint)(&pWE), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaBf16Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_bf16) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[12];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD;
            av[8] = pWS; av[9] = pWE; av[10] = pMSL; av[11] = GraphFloatBits(pScale);
            TrackPositionNode(_attentionSwaBf16Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.SwaWindowStart, windowSize), (9, GraphPosKind.SwaWindowEnd, 0)]);
        }
    }

    /// <summary>
    /// Bf16-read variant of <see cref="AttentionSwaBatched"/> (issue #179). K/V cache
    /// tensors must be <see cref="DType.BFloat16"/>. Same windowSize ≤ 4096
    /// shared-scores constraint; bit-identical per (head, token) to the per-token
    /// <see cref="AttentionSwaBf16"/> (modulo bf16 store rounding).
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwaBatchedBf16(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (windowSize <= 0 || windowSize > 4096)
            throw new ArgumentException(
                $"AttentionSwaBatchedBf16 requires 0 < windowSize ≤ 4096 (shared-scores path); got {windowSize}.",
                nameof(windowSize));

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaBatchedBf16Kernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_batched_bf16) failed: {r}");
    }

    // ── Issue #114-B: batched prompt-prefill SDPA ──────────────────────────

    /// <summary>
    /// Batched <see cref="KvAppend"/>: writes the K/V vectors for <paramref name="nTok"/>
    /// tokens into the cache at consecutive positions <c>startPos .. startPos+nTok-1</c>
    /// in a single launch. <paramref name="kAll"/>/<paramref name="vAll"/> are
    /// <c>[nTok × kvDim]</c> token-major. Bit-identical to nTok sequential
    /// <see cref="KvAppend"/> calls.
    /// </summary>
    public void KvAppendBatched(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache,
                                int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kAll), vPtr = GetDevPtr(vAll);
        nint kcP = GetDevPtr(kCache), vcP = GetDevPtr(vCache);
        int pKD = kvDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kPtr), (nint)(&vPtr), (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_batched) failed: {r}");
    }

    /// <summary>Bf16-store variant of <see cref="KvAppendBatched"/> (default KV dtype).</summary>
    public void KvAppendBatchedBf16(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache,
                                    int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kAll), vPtr = GetDevPtr(vAll);
        nint kcP = GetDevPtr(kCache), vcP = GetDevPtr(vCache);
        int pKD = kvDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kPtr), (nint)(&vPtr), (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBatchedBf16Kernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_batched_bf16) failed: {r}");
    }

    /// <summary>
    /// Batched-query scaled dot-product attention (issue #114-B). All
    /// <paramref name="nTok"/> prompt queries attend over their causal prefix in a
    /// single launch (grid = numHeads × nTok), instead of nTok sequential
    /// <see cref="Attention"/> launches. Query i (row of <paramref name="qAll"/>,
    /// <c>[nTok × numHeads·headDim]</c>) attends over cache positions
    /// <c>[0, startPos+i+1)</c>; output written to <paramref name="outAll"/> in the
    /// same layout. Bit-identical to the per-token <see cref="Attention"/> path.
    ///
    /// <para><b>Constraint:</b> uses the shared-scores fast path only, so the caller
    /// MUST guarantee <c>startPos + nTok ≤ 4096</c> (every block's seqLen stays
    /// ≤ MAX_STORED_SCORES). Beyond that, fall back to the per-token loop.</para>
    /// </summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBatched(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                 int numHeads, int numKvHeads, int headDim,
                                 int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (startPos + nTok > 4096)
            throw new ArgumentException(
                $"AttentionBatched requires startPos+nTok ≤ 4096 (shared-scores path); got {startPos}+{nTok}.");

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention) failed: {r}");
    }

    /// <summary>Bf16-read variant of <see cref="AttentionBatched"/> (default KV dtype).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBatchedBf16(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                     int numHeads, int numKvHeads, int headDim,
                                     int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (startPos + nTok > 4096)
            throw new ArgumentException(
                $"AttentionBatchedBf16 requires startPos+nTok ≤ 4096 (shared-scores path); got {startPos}+{nTok}.");

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionBf16Kernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_bf16) failed: {r}");
    }

    // ── Issue #118: wave-based >4096 batched-query SDPA ─────────────────────

    /// <summary>
    /// Bounded scratch budget (in floats) for the wave-based SDPA. The wave width is
    /// chosen so <c>W × numHeads × scoreStride</c> floats fit this budget. Override
    /// via <c>SHARPI_ATTN_WAVE_BUDGET_MB</c> (default 256 MiB).
    /// </summary>
    private static long WaveScratchBudgetFloats()
    {
        long mb = 256;
        var ov = Environment.GetEnvironmentVariable("SHARPI_ATTN_WAVE_BUDGET_MB");
        if (ov is not null && long.TryParse(ov, out long m) && m > 0) mb = m;
        return mb * 1024 * 1024 / sizeof(float);
    }

    private void EnsureWaveScratch(long floats)
    {
        nuint required = (nuint)floats * sizeof(float);
        if (_waveScratchBuf != nint.Zero && _waveScratchBufSize >= required) return;
        if (_waveScratchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_waveScratchBuf);
            _waveScratchBuf = nint.Zero;
            _waveScratchBufSize = 0;
        }
        int r = CuBlasInterop.CudaMalloc(out _waveScratchBuf, required);
        if (r != 0)
            throw new InvalidOperationException($"cudaMalloc(wave attn scratch, {required} bytes) failed: {r}");
        _waveScratchBufSize = required;
    }

    /// <summary>
    /// Wave-based batched-query SDPA for prompt prefill past the 4096-position
    /// shared-scores window (issue #118). Splits the <paramref name="nTok"/> queries
    /// into waves of <c>W</c> (chosen so <c>W × numHeads × (startPos+nTok)</c> floats
    /// of global score scratch fit <see cref="WaveScratchBudgetFloats"/>); each wave is
    /// one launch over <c>(numHeads, W)</c> blocks, each block cloning
    /// <c>llm_attention</c>'s global-scratch path. Bit-identical to the per-token
    /// <see cref="Attention"/> path (same per-(head,query) dot / tree softmax /
    /// V-weighted sum). Query/output layout matches <see cref="AttentionBatched"/>.
    /// </summary>
    public void AttentionBatchedWave(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                     int numHeads, int numKvHeads, int headDim,
                                     int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int scoreStride = startPos + nTok;            // max seq_len any query reaches
        long perQuery = (long)numHeads * scoreStride; // score floats per query row
        long budget = WaveScratchBudgetFloats();
        int W = (int)Math.Max(1, Math.Min(nTok, budget / Math.Max(1, perQuery)));
        EnsureWaveScratch((long)W * perQuery);

        int qDim = numHeads * headDim;
        nint qBase = GetDevPtr(qAll), oBase = GetDevPtr(outAll);
        nint kP = GetDevPtr(kCache), vP = GetDevPtr(vCache);
        nint scP = _waveScratchBuf;
        // Per-launch mutables (qP/oP/spEff/pN change each wave); the rest are constant.
        nint qP = qBase, oP = oBase;
        int spEff = startPos, pN = 0;
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pStride = scoreStride;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP), (nint)(&scP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&spEff), (nint)(&pMSL), (nint)(&pN), (nint)(&pStride)
        };
        for (int waveStart = 0; waveStart < nTok; waveStart += W)
        {
            int wThis = Math.Min(W, nTok - waveStart);
            qP = qBase + (nint)((long)waveStart * qDim * sizeof(float));
            oP = oBase + (nint)((long)waveStart * qDim * sizeof(float));
            spEff = startPos + waveStart;
            pN = wThis;
            int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionGlobalKernel,
                (uint)numHeads, (uint)wThis, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_global) failed: {r}");
        }
    }

    /// <summary>Bf16-read variant of <see cref="AttentionBatchedWave"/> (default KV dtype).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBatchedWaveBf16(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                         int numHeads, int numKvHeads, int headDim,
                                         int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int scoreStride = startPos + nTok;
        long perQuery = (long)numHeads * scoreStride;
        long budget = WaveScratchBudgetFloats();
        int W = (int)Math.Max(1, Math.Min(nTok, budget / Math.Max(1, perQuery)));
        EnsureWaveScratch((long)W * perQuery);

        int qDim = numHeads * headDim;
        nint qBase = GetDevPtr(qAll), oBase = GetDevPtr(outAll);
        nint kP = GetDevPtr(kCache), vP = GetDevPtr(vCache);
        nint scP = _waveScratchBuf;
        nint qP = qBase, oP = oBase;
        int spEff = startPos, pN = 0;
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pStride = scoreStride;
        float pScale = attnScale;
        nint* args = stackalloc nint[13]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP), (nint)(&scP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&spEff), (nint)(&pMSL), (nint)(&pN), (nint)(&pStride), (nint)(&pScale)
        };
        for (int waveStart = 0; waveStart < nTok; waveStart += W)
        {
            int wThis = Math.Min(W, nTok - waveStart);
            qP = qBase + (nint)((long)waveStart * qDim * sizeof(float));
            oP = oBase + (nint)((long)waveStart * qDim * sizeof(float));
            spEff = startPos + waveStart;
            pN = wThis;
            int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionGlobalBf16Kernel,
                (uint)numHeads, (uint)wThis, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_global_bf16) failed: {r}");
        }
    }

    // ── Issue #179: q8_0 KV-cache kernel wrappers ──────────────────────────
    // Block-quantized (~quarter-fp32) variants of the bf16 KV kernels above. Same
    // arg marshalling and grids; only the launched kernel handle differs (the
    // templated <block_q8_0> thunks). The store kernels quantize per 32-lane warp.

    /// <summary>q8_0-store variant of <see cref="KvAppendBf16"/> (issue #179).</summary>
    public void KvAppendQ8_0(Tensor kInput, Tensor vInput, Tensor kCache, Tensor vCache,
                             int kvDim, int position, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kInput);
        nint vPtr = GetDevPtr(vInput);
        nint kcP  = GetDevPtr(kCache);
        nint vcP  = GetDevPtr(vCache);
        int  pKD = kvDim, pPos = position, pMSL = maxSeqLen;
        nint* args = stackalloc nint[7]
        {
            (nint)(&kPtr), (nint)(&vPtr),
            (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pPos), (nint)(&pMSL)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendQ8Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_q8_0) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[7];
            av[0] = kPtr; av[1] = vPtr; av[2] = kcP; av[3] = vcP;
            av[4] = pKD; av[5] = pPos; av[6] = pMSL;
            TrackPositionNode(_kvAppendQ8Kernel, grid, 1, 1, 256, 1, 1, 0, av, [(5, GraphPosKind.Position, 0)]);
        }
    }

    /// <summary>q8_0-read variant of <see cref="AttentionBf16"/> (issue #179).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionQ8_0(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                              Tensor? scoresScratch,
                              int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                              float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSL = seqLen, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSL), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionQ8Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_q8_0) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[11];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD; av[8] = pSL; av[9] = pMSL;
            av[10] = GraphFloatBits(pScale);
            TrackPositionNode(_attentionQ8Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.PositionPlus1, 0)]);
        }
    }

    /// <summary>q8_0-read variant of <see cref="AttentionSwaBf16"/> (issue #179).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwaQ8_0(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                                 Tensor? scoresScratch,
                                 int position, int windowSize, int headDim,
                                 int numHeads, int numKvHeads, int maxSeqLen,
                                 float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int windowEnd   = position + 1;
        int windowStart = windowEnd - windowSize;
        if (windowStart < 0) windowStart = 0;

        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(kCache);
        nint vP = GetDevPtr(vCache);
        nint oP = GetDevPtr(output);
        nint ssP = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int  pWS = windowStart, pWE = windowEnd, pMSL = maxSeqLen;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pWS), (nint)(&pWE), (nint)(&pMSL), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaQ8Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_q8_0) failed: {r}");

        if (_graphCapturing)
        {
            Span<nint> av = stackalloc nint[12];
            av[0] = qP; av[1] = kP; av[2] = vP; av[3] = oP; av[4] = ssP;
            av[5] = pNH; av[6] = pNKV; av[7] = pHD;
            av[8] = pWS; av[9] = pWE; av[10] = pMSL; av[11] = GraphFloatBits(pScale);
            TrackPositionNode(_attentionSwaQ8Kernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, av,
                [(8, GraphPosKind.SwaWindowStart, windowSize), (9, GraphPosKind.SwaWindowEnd, 0)]);
        }
    }

    /// <summary>q8_0-read variant of <see cref="AttentionSwaBatchedBf16"/> (issue #179).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionSwaBatchedQ8_0(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
        int numHeads, int numKvHeads, int headDim,
        int startPos, int windowSize, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (windowSize <= 0 || windowSize > 4096)
            throw new ArgumentException(
                $"AttentionSwaBatchedQ8_0 requires 0 < windowSize ≤ 4096 (shared-scores path); got {windowSize}.",
                nameof(windowSize));

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        int pSP = startPos, pWS = windowSize, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[12]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pSP), (nint)(&pWS), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_attentionSwaBatchedQ8Kernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(attention_swa_batched_q8_0) failed: {r}");
    }

    /// <summary>q8_0-store variant of <see cref="KvAppendBatchedBf16"/> (issue #179).</summary>
    public void KvAppendBatchedQ8_0(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache,
                                    int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kPtr = GetDevPtr(kAll), vPtr = GetDevPtr(vAll);
        nint kcP = GetDevPtr(kCache), vcP = GetDevPtr(vCache);
        int pKD = kvDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&kPtr), (nint)(&vPtr), (nint)(&kcP), (nint)(&vcP),
            (nint)(&pKD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN)
        };
        uint grid = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvAppendBatchedQ8Kernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_append_batched_q8_0) failed: {r}");
    }

    /// <summary>q8_0-read variant of <see cref="AttentionBatchedBf16"/> (issue #179).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBatchedQ8_0(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                     int numHeads, int numKvHeads, int headDim,
                                     int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (startPos + nTok > 4096)
            throw new ArgumentException(
                $"AttentionBatchedQ8_0 requires startPos+nTok ≤ 4096 (shared-scores path); got {startPos}+{nTok}.");

        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kCache), vP = GetDevPtr(vCache), oP = GetDevPtr(outAll);
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pSP = startPos, pMSL = maxSeqLen, pN = nTok;
        float pScale = attnScale;
        nint* args = stackalloc nint[11]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD), (nint)(&pSP), (nint)(&pMSL), (nint)(&pN), (nint)(&pScale)
        };
        int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionQ8Kernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_q8_0) failed: {r}");
    }

    /// <summary>q8_0-read variant of <see cref="AttentionBatchedWaveBf16"/> (issue #179).</summary>
    /// <param name="attnScale">Softmax score scale: a positive value overrides (Gemma 4 passes 1.0); ≤0 (default -1) uses 1/sqrt(headDim).</param>
    public void AttentionBatchedWaveQ8_0(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                         int numHeads, int numKvHeads, int headDim,
                                         int startPos, int maxSeqLen, int nTok, float attnScale = -1f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int scoreStride = startPos + nTok;
        long perQuery = (long)numHeads * scoreStride;
        long budget = WaveScratchBudgetFloats();
        int W = (int)Math.Max(1, Math.Min(nTok, budget / Math.Max(1, perQuery)));
        EnsureWaveScratch((long)W * perQuery);

        int qDim = numHeads * headDim;
        nint qBase = GetDevPtr(qAll), oBase = GetDevPtr(outAll);
        nint kP = GetDevPtr(kCache), vP = GetDevPtr(vCache);
        nint scP = _waveScratchBuf;
        nint qP = qBase, oP = oBase;
        int spEff = startPos, pN = 0;
        int pNH = numHeads, pNKV = numKvHeads, pHD = headDim, pMSL = maxSeqLen, pStride = scoreStride;
        float pScale = attnScale;
        nint* args = stackalloc nint[13]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&vP), (nint)(&oP), (nint)(&scP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&spEff), (nint)(&pMSL), (nint)(&pN), (nint)(&pStride), (nint)(&pScale)
        };
        for (int waveStart = 0; waveStart < nTok; waveStart += W)
        {
            int wThis = Math.Min(W, nTok - waveStart);
            qP = qBase + (nint)((long)waveStart * qDim * sizeof(float));
            oP = oBase + (nint)((long)waveStart * qDim * sizeof(float));
            spEff = startPos + waveStart;
            pN = wThis;
            int r = NvrtcInterop.LaunchKernel(_fullSeqAttentionGlobalQ8Kernel,
                (uint)numHeads, (uint)wThis, 1, 256, 1, 1, 0, _stream, args, null);
            if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(full_seq_attention_global_q8_0) failed: {r}");
        }
    }

    // ================================================================
    //  SnapKV (issue #58): prefill KV eviction support
    // ================================================================

    /// <summary>
    /// Score one captured query <paramref name="q"/> (size <c>numHeads * headDim</c>)
    /// against the layer's K cache (positions <c>[0, promptLen)</c>), softmax across
    /// the causally-valid prefix, and atomicAdd the per-position weights into
    /// <paramref name="scoreAccum"/>. Caller is responsible for zeroing the accumulator
    /// before the first call and looping over the W captured queries × attention
    /// layers; the kernel pools naturally because every call's softmaxed weights are
    /// summed into the same buffer.
    /// </summary>
    /// <param name="qAbsPos">Absolute prompt position of the query (causal mask cutoff).</param>
    public void SnapKvScore(Tensor q, Tensor kCache, Tensor scoreAccum, Tensor scoresScratch,
                            int numHeads, int numKvHeads, int headDim,
                            int promptLen, int qAbsPos, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(q);
        nint kP  = GetDevPtr(kCache);
        nint sP  = GetDevPtr(scoreAccum);
        nint ssP = GetDevPtr(scoresScratch);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pPL = promptLen, pQAP = qAbsPos, pMSL = maxSeqLen;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&sP), (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pPL), (nint)(&pQAP), (nint)(&pMSL)
        };
        int r = NvrtcInterop.LaunchKernel(_snapKvScoreKernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(snapkv_score) failed: {r}");
    }

    /// <summary>
    /// Bf16-K-cache variant of <see cref="SnapKvScore"/>. <paramref name="kCache"/>
    /// must be a <see cref="DType.BFloat16"/> tensor (raw unsigned short storage);
    /// the scoring math stays in fp32.
    /// </summary>
    public void SnapKvScoreBf16(Tensor q, Tensor kCache, Tensor scoreAccum, Tensor scoresScratch,
                                int numHeads, int numKvHeads, int headDim,
                                int promptLen, int qAbsPos, int maxSeqLen)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(q);
        nint kP  = GetDevPtr(kCache);
        nint sP  = GetDevPtr(scoreAccum);
        nint ssP = GetDevPtr(scoresScratch);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pPL = promptLen, pQAP = qAbsPos, pMSL = maxSeqLen;
        nint* args = stackalloc nint[10]
        {
            (nint)(&qP), (nint)(&kP), (nint)(&sP), (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pPL), (nint)(&pQAP), (nint)(&pMSL)
        };
        int r = NvrtcInterop.LaunchKernel(_snapKvScoreBf16Kernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(snapkv_score_bf16) failed: {r}");
    }

    /// <summary>
    /// Gather kept positions of one KV ring (K or V) into a dense
    /// <c>[K * kvDim]</c> prefix of <paramref name="dst"/>. <paramref name="src"/>
    /// and <paramref name="dst"/> MUST be different tensors (the destination is
    /// later copied back over the original ring's <c>[0, K * kvDim)</c> region by
    /// the caller). <paramref name="keepPositions"/> must be sorted ascending and
    /// hold indices in <c>[0, originalLength)</c>.
    /// </summary>
    public void KvCompact(Tensor src, Tensor dst, Tensor keepPositions,
                          int K, int kvDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP  = GetDevPtr(src);
        nint dP  = GetDevPtr(dst);
        nint kpP = GetDevPtr(keepPositions);
        int  pK = K, pKD = kvDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&kpP),
            (nint)(&pK), (nint)(&pKD)
        };
        uint gridX = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvCompactKernel,
            gridX, (uint)K, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_compact) failed: {r}");
    }

    /// <summary>
    /// Bf16-store variant of <see cref="KvCompact"/>. Source and destination must
    /// both be <see cref="DType.BFloat16"/> tensors; the gather copies raw unsigned
    /// short elements with no fp32 round-trip.
    /// </summary>
    public void KvCompactBf16(Tensor src, Tensor dst, Tensor keepPositions,
                              int K, int kvDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP  = GetDevPtr(src);
        nint dP  = GetDevPtr(dst);
        nint kpP = GetDevPtr(keepPositions);
        int  pK = K, pKD = kvDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&kpP),
            (nint)(&pK), (nint)(&pKD)
        };
        uint gridX = (uint)((kvDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_kvCompactBf16Kernel,
            gridX, (uint)K, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(kv_compact_bf16) failed: {r}");
    }

    // ================================================================
    //  TurboQuant KV-cache compression
    // ================================================================

    /// <summary>
    /// Rotate query vectors for TurboQuant attention: applies the Walsh-Hadamard
    /// transform and the per-(layer × kv_head) sign flip. Call once per layer
    /// (before <see cref="TqAttention"/>), reuse the result across all cached positions.
    /// </summary>
    public void TqRotateQuery(Tensor qInput, Tensor rotatedQ, Tensor signPatterns,
                              int numHeads, int numKvHeads, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP  = GetDevPtr(qInput);
        nint rqP = GetDevPtr(rotatedQ);
        nint sP  = GetDevPtr(signPatterns);
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim;
        nint* args = stackalloc nint[6]
        {
            (nint)(&qP), (nint)(&rqP), (nint)(&sP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD)
        };
        int r = NvrtcInterop.LaunchKernel(_tqRotateQueryKernel,
            (uint)numHeads, 1, 1, (uint)headDim, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_rotate_query) failed: {r}");
    }

    /// <summary>
    /// Compress one (K, V) token-position into the layer's TurboQuant cache.
    /// Writes a packed 3-bit block per KV head at byte offset
    /// <c>position * numKvHeads * blockBytes + kvHead * blockBytes</c>.
    /// </summary>
    public void TqKvAppend(Tensor kInput, Tensor vInput, Tensor kCacheTq, Tensor vCacheTq,
                           Tensor signPatterns, Tensor codebook, Tensor boundaries,
                           int kvDim, int headDim, int position, int maxSeqLen,
                           int numKvHeads, int blockBytes)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint kP = GetDevPtr(kInput);
        nint vP = GetDevPtr(vInput);
        nint kc = GetDevPtr(kCacheTq);
        nint vc = GetDevPtr(vCacheTq);
        nint sP = GetDevPtr(signPatterns);
        nint cP = GetDevPtr(codebook);
        nint bP = GetDevPtr(boundaries);
        int  pKD = kvDim, pHD = headDim, pPos = position, pMSL = maxSeqLen,
             pNKV = numKvHeads, pBB = blockBytes;
        nint* args = stackalloc nint[13]
        {
            (nint)(&kP), (nint)(&vP), (nint)(&kc), (nint)(&vc),
            (nint)(&sP), (nint)(&cP), (nint)(&bP),
            (nint)(&pKD), (nint)(&pHD), (nint)(&pPos),
            (nint)(&pMSL), (nint)(&pNKV), (nint)(&pBB)
        };
        int r = NvrtcInterop.LaunchKernel(_tqKvAppendKernel,
            (uint)numKvHeads, 1, 1, (uint)headDim, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_kv_append) failed: {r}");
    }

    /// <summary>
    /// Hybrid TurboQuant attention: scaled dot-product attention over the
    /// TQ-compressed history plus the FP32 recent-window cache.
    ///
    /// When <c>tqSeqLen + fp32SeqLen ≤ 4096</c> the kernel keeps per-position scores
    /// in shared memory and <paramref name="scoresScratch"/> is ignored. Above that
    /// threshold the kernel spills the scores to <paramref name="scoresScratch"/>,
    /// which must have room for <c>numHeads × maxSeqLen</c> floats (one slot per
    /// (head, position) pair). Passing a non-null scratch always works regardless
    /// of context size — callers can allocate once for the worst case.
    /// </summary>
    public void TqAttention(Tensor q, Tensor rotatedQ, Tensor kCacheTq, Tensor vCacheTq,
                            Tensor kCacheFp32, Tensor vCacheFp32, Tensor output, Tensor codebook,
                            Tensor? scoresScratch,
                            int numHeads, int numKvHeads, int headDim,
                            int tqSeqLen, int fp32SeqLen, int maxSeqLen, int blockBytes)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint qP   = GetDevPtr(q);
        nint rqP  = GetDevPtr(rotatedQ);
        nint kctP = GetDevPtr(kCacheTq);
        nint vctP = GetDevPtr(vCacheTq);
        nint kfP  = GetDevPtr(kCacheFp32);
        nint vfP  = GetDevPtr(vCacheFp32);
        nint oP   = GetDevPtr(output);
        nint cbP  = GetDevPtr(codebook);
        nint ssP  = scoresScratch is { } sv ? GetDevPtr(sv) : nint.Zero;
        int  pNH = numHeads, pNKV = numKvHeads, pHD = headDim,
             pTQ = tqSeqLen, pFP = fp32SeqLen, pMSL = maxSeqLen, pBB = blockBytes;
        nint* args = stackalloc nint[16]
        {
            (nint)(&qP), (nint)(&rqP),
            (nint)(&kctP), (nint)(&vctP),
            (nint)(&kfP),  (nint)(&vfP),
            (nint)(&oP),   (nint)(&cbP),
            (nint)(&ssP),
            (nint)(&pNH), (nint)(&pNKV), (nint)(&pHD),
            (nint)(&pTQ), (nint)(&pFP),  (nint)(&pMSL), (nint)(&pBB)
        };
        int r = NvrtcInterop.LaunchKernel(_tqAttentionKernel,
            (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(tq_attention) failed: {r}");
    }

    /// <summary>Look up one row from an F32 embedding table into <paramref name="output"/>.</summary>
    public void EmbedLookup(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        uint grid = (uint)((embDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_embedLookupF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_f32) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q4_K-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256.
    /// </summary>
    public void EmbedLookupQ4K(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ4K requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ4KKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q4k) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q5_K-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256 (Q5_K block size).
    /// </summary>
    public void EmbedLookupQ5K(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ5K requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ5KKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q5k) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q6_K-packed embedding table directly into
    /// <paramref name="output"/> (issue #124, Gemma 4 12B tied token_embd). Keeps the
    /// (3840, 262144) Q6_K table packed (~787 MiB) off the F32 dequant path that would
    /// burn ~4 GB of VRAM. <paramref name="embDim"/> must be a multiple of 256.
    /// </summary>
    public void EmbedLookupQ6K(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ6K requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ6KKernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q6k) failed: {r}");
    }

    /// <summary>
    /// Dequantize one row from a Q8_0-packed embedding table directly into <paramref name="output"/>.
    /// <paramref name="embDim"/> must be a multiple of 256 (8 Q8_0 blocks per outer iteration).
    /// Phase 0 of the Gemma-4 plan: keeps the (10752, 262144) Q8_0 token-embd table off
    /// the dequant-to-F32 upload path that would otherwise blow out VRAM.
    /// </summary>
    public void EmbedLookupQ8_0(Tensor embTable, Tensor output, int tokenId, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ8_0 requires embDim to be a multiple of 256 (got {embDim}).");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(output);
        int  pT = tokenId, pE = embDim;
        nint* args = stackalloc nint[4]
        {
            (nint)(&tP), (nint)(&oP),
            (nint)(&pT), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ80Kernel, 1, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q8_0) failed: {r}");
    }

    /// <summary>
    /// Batched Q8_0 embedding lookup: writes all <paramref name="nTok"/> token rows into
    /// <paramref name="outputAll"/> ([nTok × embDim]) in one launch, reading the per-token
    /// ids from the device buffer <paramref name="tokenIds"/> (nTok int32). Collapses the
    /// prefill's per-token <see cref="EmbedLookupQ8_0"/> + device copy (2·N host launches)
    /// into a single grid.x = nTok launch. Bit-identical to the per-token path.
    /// </summary>
    public void EmbedLookupQ8_0Batched(Tensor embTable, Tensor outputAll, Tensor tokenIds, int nTok, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((embDim & 0xff) != 0)
            throw new ArgumentException($"EmbedLookupQ8_0Batched requires embDim to be a multiple of 256 (got {embDim}).");
        if (nTok <= 0)
            throw new ArgumentOutOfRangeException(nameof(nTok), nTok, "nTok must be > 0.");

        nint tP = GetDevPtr(embTable);
        nint oP = GetDevPtr(outputAll);
        nint idP = GetDevPtr(tokenIds);
        int pN = nTok, pE = embDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&tP), (nint)(&oP), (nint)(&idP), (nint)(&pN), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_embedLookupQ80BatchedKernel, (uint)nTok, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(embed_lookup_q8_0_batched) failed: {r}");
    }

    /// <summary>
    /// Dequantize <paramref name="nRows"/> CONTIGUOUS Q8_0-packed rows in
    /// <paramref name="src"/> into the f32 buffer <paramref name="dst"/> ([nRows × rowDim]):
    /// row i → row i. Issue #247: moves the Gemma-4 PLE pre-pass dequant off the CPU
    /// <c>Parallel.For</c> (and shrinks the host→device upload 4× — packed quant bytes
    /// instead of f32). Bit-identical to <c>Dequantize.ToFloat32(..., Q8_0)</c>.
    /// <paramref name="rowDim"/> must be a multiple of 256.
    /// </summary>
    public void DequantRowsQ8_0(Tensor src, Tensor dst, int nRows, int rowDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((rowDim & 0xff) != 0)
            throw new ArgumentException($"DequantRowsQ8_0 requires rowDim to be a multiple of 256 (got {rowDim}).");
        if (nRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(nRows), nRows, "nRows must be > 0.");

        nint sP = GetDevPtr(src);
        nint dP = GetDevPtr(dst);
        int pN = nRows, pE = rowDim;
        nint* args = stackalloc nint[4] { (nint)(&sP), (nint)(&dP), (nint)(&pN), (nint)(&pE) };
        int r = NvrtcInterop.LaunchKernel(_dequantRowsQ80Kernel, (uint)nRows, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(dequant_rows_q8_0) failed: {r}");
    }

    /// <summary>
    /// Q6_K analogue of <see cref="DequantRowsQ8_0"/> (issue #247). Bit-identical to
    /// <c>Dequantize.ToFloat32(..., Q6_K)</c>. <paramref name="rowDim"/> must be a
    /// multiple of 256.
    /// </summary>
    public void DequantRowsQ6K(Tensor src, Tensor dst, int nRows, int rowDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if ((rowDim & 0xff) != 0)
            throw new ArgumentException($"DequantRowsQ6K requires rowDim to be a multiple of 256 (got {rowDim}).");
        if (nRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(nRows), nRows, "nRows must be > 0.");

        nint sP = GetDevPtr(src);
        nint dP = GetDevPtr(dst);
        int pN = nRows, pE = rowDim;
        nint* args = stackalloc nint[4] { (nint)(&sP), (nint)(&dP), (nint)(&pN), (nint)(&pE) };
        int r = NvrtcInterop.LaunchKernel(_dequantRowsQ6KKernel, (uint)nRows, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(dequant_rows_q6k) failed: {r}");
    }

    /// <summary>Set every element of <paramref name="dst"/> to zero.</summary>
    /// <remarks>
    /// The kernel writes fp32-sized lanes; for sub-fp32 dtypes (e.g. BFloat16)
    /// the byte count is converted to a 4-byte lane count so we don't overrun
    /// the underlying buffer. For non-fp32 element sizes that aren't a multiple
    /// of 4 bytes, callers should instead use a memset path (none exposed yet).
    /// </remarks>
    public void Clear(Tensor dst)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        long byteCount = dst.ElementCount * DTypeInfo.BytesPerElement(dst.DType);
        if ((byteCount & 3) != 0)
            throw new InvalidOperationException(
                $"Clear: dst byte count ({byteCount}) is not a multiple of 4; " +
                "dtype-aware memset path is not implemented yet.");
        int n = (int)(byteCount >> 2);
        nint dP = GetDevPtr(dst);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&dP), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_clearF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(clear) failed: {r}");
    }

    /// <summary>Zero a sub-region of a tensor starting at <paramref name="elementOffset"/>
    /// for <paramref name="elementCount"/> FP32 elements.</summary>
    public void ClearRegion(Tensor dst, long elementOffset, int elementCount)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        nint dP = GetDevPtr(dst) + (nint)(elementOffset * sizeof(float));
        int  pN = elementCount;
        nint* args = stackalloc nint[2] { (nint)(&dP), (nint)(&pN) };
        uint grid = (uint)((elementCount + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_clearF32Kernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(clear region) failed: {r}");
    }

    /// <summary>In-place SiLU activation: x[i] = x[i] / (1 + exp(-x[i])). One thread per element.</summary>
    public void SiLUInPlace(Tensor x)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        int n = (int)x.ElementCount;
        nint xP = GetDevPtr(x);
        int  pN = n;
        nint* args = stackalloc nint[2] { (nint)(&xP), (nint)(&pN) };
        uint grid = (uint)((n + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_siluInplaceKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(silu_inplace) failed: {r}");
    }

    /// <summary>
    /// GDN depthwise causal conv1d for a single decode token. State layout
    /// <c>[(kernel-1), channels]</c> row-major, oldest first; updated in place.
    /// Weight layout <c>[kernel, channels]</c>.
    /// </summary>
    public void GdnConv1dDecode(Tensor x, Tensor state, Tensor weight, Tensor output,
                                int channels, int kernelSize)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x);
        nint sP = GetDevPtr(state);
        nint wP = GetDevPtr(weight);
        nint oP = GetDevPtr(output);
        int  pC = channels, pK = kernelSize;
        nint* args = stackalloc nint[6]
        {
            (nint)(&xP), (nint)(&sP), (nint)(&wP), (nint)(&oP),
            (nint)(&pC), (nint)(&pK)
        };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dDecodeKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_decode) failed: {r}");
    }

    /// <summary>
    /// L2 normalize each <paramref name="headDim"/>-sized slice independently
    /// (no learned weights). Matches <c>GdnKernels.L2NormPerHead</c>:
    /// <c>scale = 1 / max(sqrt(Σ x²), eps)</c>. Operates on the sub-region of
    /// <paramref name="data"/> starting at <paramref name="elementOffset"/>
    /// for <paramref name="numHeads"/> × <paramref name="headDim"/> floats.
    /// </summary>
    public void GdnL2NormPerHead(Tensor data, long elementOffset, int numHeads, int headDim, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dP = GetDevPtr(data) + (nint)(elementOffset * sizeof(float));
        int  pHD = headDim, pNH = numHeads;
        float pE = eps;
        nint* args = stackalloc nint[4]
        {
            (nint)(&dP),
            (nint)(&pHD), (nint)(&pNH), (nint)(&pE)
        };
        int r = NvrtcInterop.LaunchKernel(_gdnL2NormPerHeadKernel, (uint)numHeads, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_l2_norm) failed: {r}");
    }

    /// <summary>
    /// Tile pattern: dst[h_dst, j] = src[h_dst % srcHeads, j] for h_dst ∈ [0, srcHeads·repeat).
    /// Matches <c>GdnKernels.TileHeads</c> (GQA-style broadcast, NOT torch repeat_interleave).
    /// <paramref name="srcOffset"/> and <paramref name="dstOffset"/> are FP32 element offsets.
    /// </summary>
    public void GdnTileHeads(Tensor src, long srcOffset, Tensor dst, long dstOffset,
                             int srcHeads, int repeat, int headDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(src) + (nint)(srcOffset * sizeof(float));
        nint dP = GetDevPtr(dst) + (nint)(dstOffset * sizeof(float));
        int  pSH = srcHeads, pR = repeat, pHD = headDim;
        nint* args = stackalloc nint[5]
        {
            (nint)(&sP), (nint)(&dP),
            (nint)(&pSH), (nint)(&pR), (nint)(&pHD)
        };
        int total = srcHeads * repeat * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnTileHeadsKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_tile_heads) failed: {r}");
    }

    /// <summary>
    /// Single-token Gated DeltaNet recurrence. Updates per-head state matrices in
    /// place and writes per-head post-norm, post-gate output. Mirrors
    /// <c>GdnKernels.GdnRecurrenceDecode</c>.
    /// State layout: <c>[numVHeads, headDim, headDim]</c> row-major (i = key axis,
    /// j = value/output axis).
    /// </summary>
    public void GdnRecurrenceDecode(
        Tensor state, Tensor q, Tensor k, Tensor v,
        Tensor alphaIn, Tensor beta, Tensor ssmA, Tensor dtBias,
        Tensor normWeight, Tensor z, Tensor output,
        int numVHeads, int headDim, float normEps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(state);
        nint qP = GetDevPtr(q);
        nint kP = GetDevPtr(k);
        nint vP = GetDevPtr(v);
        nint aP = GetDevPtr(alphaIn);
        nint bP = GetDevPtr(beta);
        nint aaP = GetDevPtr(ssmA);
        nint dbP = GetDevPtr(dtBias);
        nint nwP = GetDevPtr(normWeight);
        nint zP = GetDevPtr(z);
        nint oP = GetDevPtr(output);
        int  pHV = numVHeads, pD = headDim;
        float pE = normEps;
        nint* args = stackalloc nint[14]
        {
            (nint)(&sP), (nint)(&qP), (nint)(&kP), (nint)(&vP),
            (nint)(&aP), (nint)(&bP), (nint)(&aaP), (nint)(&dbP),
            (nint)(&nwP), (nint)(&zP), (nint)(&oP),
            (nint)(&pHV), (nint)(&pD), (nint)(&pE)
        };
        // Shared memory: 8 × headDim × 4 bytes (sK, sQ, sV, sZ, sNormW, sP, sD, sRed).
        uint sharedBytes = (uint)(8 * headDim * sizeof(float));
        int r = NvrtcInterop.LaunchKernel(_gdnRecurrenceDecodeKernel,
            (uint)numVHeads, 1, 1, (uint)headDim, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_recurrence_decode) failed: {r}");
    }

    // ── Issue #114-B: batched GDN trunk over a chunk of N prompt tokens ────────

    /// <summary>
    /// Batched GDN depthwise conv1d over <paramref name="nTok"/> tokens (read-only
    /// state). <paramref name="x"/>/<paramref name="output"/> are <c>[nTok × channels]</c>;
    /// <paramref name="state"/> is the carried <c>[(K-1) × channels]</c>. Bit-identical to
    /// nTok sequential <see cref="GdnConv1dDecode"/> calls. State is advanced separately by
    /// <see cref="GdnConv1dStateUpdateBatched"/> (so concurrent token blocks read one snapshot).
    /// </summary>
    public void GdnConv1dDecodeBatched(Tensor x, Tensor state, Tensor weight, Tensor output,
                                       int channels, int kernelSize, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x), sP = GetDevPtr(state), wP = GetDevPtr(weight), oP = GetDevPtr(output);
        int pC = channels, pK = kernelSize, pN = nTok;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xP), (nint)(&sP), (nint)(&wP), (nint)(&oP),
            (nint)(&pC), (nint)(&pK), (nint)(&pN)
        };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dDecodeBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_decode_batched) failed: {r}");
    }

    /// <summary>Advance the conv1d state past a chunk of <paramref name="nTok"/> tokens
    /// (matches the sequential state evolution). See <see cref="GdnConv1dDecodeBatched"/>.</summary>
    public void GdnConv1dStateUpdateBatched(Tensor x, Tensor state, int channels, int kernelSize, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint xP = GetDevPtr(x), sP = GetDevPtr(state);
        int pC = channels, pK = kernelSize, pN = nTok;
        nint* args = stackalloc nint[5] { (nint)(&xP), (nint)(&sP), (nint)(&pC), (nint)(&pK), (nint)(&pN) };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dStateUpdateBatchedKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_state_update_batched) failed: {r}");
    }

    /// <summary>
    /// Batched per-head L2-norm over <paramref name="nTok"/> tokens. <paramref name="data"/>
    /// is offset to the region base; <paramref name="rowStride"/> is the per-token element
    /// stride. grid = (numHeads, nTok). Bit-identical to nTok sequential
    /// <see cref="GdnL2NormPerHead"/> calls.
    /// </summary>
    public void GdnL2NormPerHeadBatched(Tensor data, long elementOffset, int numHeads, int headDim,
                                        int rowStride, int nTok, float eps = 1e-6f)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint dP = GetDevPtr(data) + (nint)(elementOffset * sizeof(float));
        int pHD = headDim, pNH = numHeads, pRS = rowStride, pN = nTok;
        float pE = eps;
        nint* args = stackalloc nint[6]
        {
            (nint)(&dP), (nint)(&pHD), (nint)(&pNH), (nint)(&pE), (nint)(&pRS), (nint)(&pN)
        };
        int r = NvrtcInterop.LaunchKernel(_gdnL2NormPerHeadBatchedKernel, (uint)numHeads, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_l2_norm_per_head_batched) failed: {r}");
    }

    /// <summary>
    /// Batched GQA-broadcast tile over <paramref name="nTok"/> tokens. <paramref name="src"/>
    /// is offset to the region base; <paramref name="srcStride"/>/<paramref name="dstStride"/>
    /// are per-token strides. Bit-identical to nTok sequential <see cref="GdnTileHeads"/> calls.
    /// </summary>
    public void GdnTileHeadsBatched(Tensor src, long srcOffset, Tensor dst, long dstOffset,
                                    int srcHeads, int repeat, int headDim,
                                    int srcStride, int dstStride, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(src) + (nint)(srcOffset * sizeof(float));
        nint dP = GetDevPtr(dst) + (nint)(dstOffset * sizeof(float));
        int pSH = srcHeads, pR = repeat, pHD = headDim, pSS = srcStride, pDS = dstStride, pN = nTok;
        nint* args = stackalloc nint[8]
        {
            (nint)(&sP), (nint)(&dP), (nint)(&pSH), (nint)(&pR), (nint)(&pHD),
            (nint)(&pSS), (nint)(&pDS), (nint)(&pN)
        };
        int total = srcHeads * repeat * headDim;
        uint grid = (uint)((total + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnTileHeadsBatchedKernel, grid, (uint)nTok, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_tile_heads_batched) failed: {r}");
    }

    /// <summary>
    /// Fused sequential GDN recurrence scan over <paramref name="nTok"/> tokens: one launch
    /// (one block per v-head) loops the positions internally, carrying the per-head state in
    /// place. Bit-identical to nTok sequential <see cref="GdnRecurrenceDecode"/> calls — the
    /// fused form of the per-token decode, NOT the parallel chunked-scan. Per-head input
    /// strides let q/k come from the tiled <c>[nTok × valueDim]</c> buffers, v from the
    /// silu'd conv output (<paramref name="vHeadOff"/> into a <c>[nTok × convChannels]</c>
    /// buffer), z from a <c>[nTok × valueDim]</c> gate, alpha/beta from <c>[nTok × numVHeads]</c>.
    /// </summary>
    public void GdnRecurrenceScan(
        Tensor state, Tensor qAll, Tensor kAll, Tensor vAll,
        Tensor alphaAll, Tensor betaAll, Tensor ssmA, Tensor dtBias,
        Tensor normWeight, Tensor zAll, Tensor outputAll,
        int numVHeads, int headDim, float normEps,
        int qStride, int kStride, int vStride, int vHeadOff, int zStride, int oStride, int nTok,
        Tensor? ringScan = null, long ringScanFloatOffset = 0, int ringSlotStride = 0, int nCapture = 0)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(state);
        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kAll), vP = GetDevPtr(vAll);
        nint aP = GetDevPtr(alphaAll), bP = GetDevPtr(betaAll);
        nint aaP = GetDevPtr(ssmA), dbP = GetDevPtr(dtBias), nwP = GetDevPtr(normWeight);
        nint zP = GetDevPtr(zAll), oP = GetDevPtr(outputAll);
        int pHV = numVHeads, pD = headDim;
        float pE = normEps;
        int pQS = qStride, pKS = kStride, pVS = vStride, pVO = vHeadOff, pZS = zStride, pOS = oStride, pN = nTok;
        // #290 ring capture: null tensor → null pointer + 0 captures (no-op).
        nint rsP = ringScan is null ? 0 : GetDevPtr(ringScan) + (nint)(ringScanFloatOffset * sizeof(float));
        int pRSS = ringSlotStride, pNC = ringScan is null ? 0 : nCapture;
        nint* args = stackalloc nint[24]
        {
            (nint)(&sP), (nint)(&qP), (nint)(&kP), (nint)(&vP),
            (nint)(&aP), (nint)(&bP), (nint)(&aaP), (nint)(&dbP),
            (nint)(&nwP), (nint)(&zP), (nint)(&oP),
            (nint)(&pHV), (nint)(&pD), (nint)(&pE),
            (nint)(&pQS), (nint)(&pKS), (nint)(&pVS), (nint)(&pVO), (nint)(&pZS), (nint)(&pOS), (nint)(&pN),
            (nint)(&rsP), (nint)(&pRSS), (nint)(&pNC)
        };
        uint sharedBytes = (uint)(8 * headDim * sizeof(float));
        int r = NvrtcInterop.LaunchKernel(_gdnRecurrenceScanKernel,
            (uint)numVHeads, 1, 1, (uint)headDim, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_recurrence_scan) failed: {r}");
    }

    /// <summary>
    /// #290: capture the per-token conv1d states of a batched-verify chunk into the
    /// device snapshot ring in a single launch. Slot <c>i</c> (i ∈ [0,
    /// <paramref name="nCapture"/>)) receives the conv state the sequential decode
    /// loop would hold after token <c>i</c> — byte-identical to
    /// <see cref="GdnConv1dStateUpdateBatched"/> with <c>nTok = i+1</c>. Reads the
    /// PRE-update <paramref name="state"/>, so call it BEFORE advancing the live
    /// conv state. <paramref name="ring"/> points to this layer's region in slot 0;
    /// <paramref name="ringFloatOffset"/> offsets to it; <paramref name="ringSlotStride"/>
    /// is the float stride between consecutive slots.
    /// </summary>
    public void GdnConv1dStateCaptureRing(Tensor x, Tensor state, Tensor ring, long ringFloatOffset,
                                          int channels, int kernelSize, int ringSlotStride, int nCapture)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");
        if (nCapture <= 0) return;

        nint xP = GetDevPtr(x), sP = GetDevPtr(state);
        nint rP = GetDevPtr(ring) + (nint)(ringFloatOffset * sizeof(float));
        int pC = channels, pK = kernelSize, pRSS = ringSlotStride, pNC = nCapture;
        nint* args = stackalloc nint[7]
        {
            (nint)(&xP), (nint)(&sP), (nint)(&rP), (nint)(&pC), (nint)(&pK), (nint)(&pRSS), (nint)(&pNC)
        };
        uint grid = (uint)((channels + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_gdnConv1dStateCaptureRingKernel, grid, (uint)nCapture, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_conv1d_state_capture_ring) failed: {r}");
    }

    /// <summary>GDN_CHUNK in <c>llm_gdn_chunked_prefill</c> — must match the kernel #define.</summary>
    public const int GdnChunkSize = 64;

    /// <summary>
    /// Chunk-parallel GDN prefill over <paramref name="nTok"/> tokens (FlashQLA-style
    /// chunk_gated_delta_rule). Same inputs/strides and in-place state update as
    /// <see cref="GdnRecurrenceScan"/>, but resolves each <see cref="GdnChunkSize"/>-token
    /// block with the parallel delta-rule form instead of the sequential scan — the GPU
    /// mirror of <c>GdnKernels.GdnRecurrenceChunkedPrefill</c>. Numerically equal to the
    /// scan up to FP reduction order. One block per v-head, blockDim = headDim.
    /// </summary>
    public void GdnChunkedPrefill(
        Tensor state, Tensor qAll, Tensor kAll, Tensor vAll,
        Tensor alphaAll, Tensor betaAll, Tensor ssmA, Tensor dtBias,
        Tensor normWeight, Tensor zAll, Tensor outputAll,
        int numVHeads, int headDim, float normEps,
        int qStride, int kStride, int vStride, int vHeadOff, int zStride, int oStride, int nTok)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC kernels are not available.");

        nint sP = GetDevPtr(state);
        nint qP = GetDevPtr(qAll), kP = GetDevPtr(kAll), vP = GetDevPtr(vAll);
        nint aP = GetDevPtr(alphaAll), bP = GetDevPtr(betaAll);
        nint aaP = GetDevPtr(ssmA), dbP = GetDevPtr(dtBias), nwP = GetDevPtr(normWeight);
        nint zP = GetDevPtr(zAll), oP = GetDevPtr(outputAll);
        int pHV = numVHeads, pD = headDim;
        float pE = normEps;
        int pQS = qStride, pKS = kStride, pVS = vStride, pVO = vHeadOff, pZS = zStride, pOS = oStride, pN = nTok;
        nint* args = stackalloc nint[21]
        {
            (nint)(&sP), (nint)(&qP), (nint)(&kP), (nint)(&vP),
            (nint)(&aP), (nint)(&bP), (nint)(&aaP), (nint)(&dbP),
            (nint)(&nwP), (nint)(&zP), (nint)(&oP),
            (nint)(&pHV), (nint)(&pD), (nint)(&pE),
            (nint)(&pQS), (nint)(&pKS), (nint)(&pVS), (nint)(&pVO), (nint)(&pZS), (nint)(&pOS), (nint)(&pN)
        };
        // Shared layout (floats): sNormW[d] + sCum/sG/sB[GDN_CHUNK each] +
        // sKK/sKQ[GDN_CHUNK*GDN_CHUNK each] + sRed[d].
        uint sharedBytes = (uint)((2 * headDim + 3 * GdnChunkSize + 2 * GdnChunkSize * GdnChunkSize) * sizeof(float));
        int r = NvrtcInterop.LaunchKernel(_gdnChunkedPrefillKernel,
            (uint)numVHeads, 1, 1, (uint)headDim, 1, 1, sharedBytes, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(gdn_chunked_prefill) failed: {r}");
    }

    public void FullSeqAttention(Tensor output, Tensor q, Tensor k, Tensor v,
                                 int nTok, int nHeads, int headDim, float scale) =>
        throw new NotSupportedException("CudaBackend.FullSeqAttention is not implemented (LLM path uses single-token Attention).");

    // ── Helpers ───────────────────────────────────────────────────────────

    private nint GetDevPtr(Tensor tensor) =>
        _devPtrs.TryGetValue(tensor.Handle, out var entry)
            ? entry.devPtr
            : throw new InvalidOperationException($"Tensor handle {tensor.Handle} not registered in CudaBackend");

    // ── IImageOpsBackend ──────────────────────────────────────────────────

    /// <summary>
    /// Returns true when NVRTC image kernels are available and compiled.
    /// Triggers lazy compilation on first call.
    /// </summary>
    public bool ImageKernelsAvailable
    {
        get
        {
            EnsureImageKernels();
            return _imageKernelsAvailable;
        }
    }

    /// <summary>
    /// Lazily compile all image-ops CUDA kernels via NVRTC and load the resulting PTX.
    /// Idempotent: subsequent calls are a no-op once initialised (success or failure).
    /// On failure sets <c>_imageKernelsAvailable = false</c> so callers can fall back gracefully.
    /// </summary>
    private void EnsureImageKernels()
    {
        // Always bind the primary CUDA context on the calling thread — even after the
        // module is loaded. cuModuleLoadData (init) and cuLaunchKernel (every launch
        // below) both require a current context, but cuBLAS handles bind to it
        // internally so the rest of the backend works cross-thread without this.
        //
        // Hosted scenarios (e.g. ASP.NET request threads) construct CudaBackend on
        // one thread and decode on another — without this guard cuModuleLoadData
        // returns CUDA_ERROR_INVALID_CONTEXT (201). See issue #94.
        EnsurePrimaryContextCurrent();

        if (_imageKernelsInitialized) return;
        lock (_kernelInitLock)
        {
            if (_imageKernelsInitialized) return;
            try
            {
                CompileAndLoadKernels();
                _imageKernelsAvailable = true;
            }
            catch (Exception ex)
            {
                _imageKernelsAvailable = false;
                // Log to stderr so the user can see the NVRTC failure reason when debugging.
                // Use ToString() not just Message: a GetKernelFunc failure names the specific
                // kernel that didn't bind (e.g. a typo'd q8_0 thunk), and that name only
                // survives in the full exception text + stack, not the terse Message.
                Console.Error.WriteLine($"[CudaBackend] NVRTC kernel init failed: {ex}");
            }
            finally
            {
                _imageKernelsInitialized = true;
            }
        }
    }

    // ── CUDA primary-context binding ──────────────────────────────────────
    //
    // Process-wide retained primary context (device 0), made current on each thread
    // that calls into the NVRTC kernel path. ThreadStatic flag skips redundant
    // cuCtxSetCurrent on a thread that's already bound.

    private static readonly object   s_primaryCtxLock = new();
    private static          nint     s_primaryCtx;
    private static          int      s_primaryCtxDevice = -1;
    [ThreadStatic]
    private static          bool     t_primaryCtxBoundOnThisThread;

    /// <summary>
    /// Make the process-wide device-0 primary CUDA context current on the calling thread (issue
    /// #302). CUDA contexts are thread-affine; a consumer that drives the forward pass from a
    /// thread other than the one that loaded the model (e.g. an engine worker thread, or a
    /// thread-pool continuation) must bind the context first, or in a non-interactive session the
    /// first CUDA call on that thread can hang forever. Idempotent and cheap after the first call
    /// per thread (the underlying <see cref="EnsurePrimaryContextCurrent"/> short-circuits on a
    /// ThreadStatic flag). The forward passes expose this through
    /// <c>IThreadAffineBackend.BindToCurrentThread</c>.
    /// </summary>
    public void BindContextToCurrentThread() => EnsurePrimaryContextCurrent();

    /// <summary>
    /// Ensure the device's primary context is current on the calling thread. Cheap after
    /// the first call per thread (single ThreadStatic check). The retained primary context
    /// is the same one cuBLAS attaches to, so once it's current on a thread the entire
    /// CUDA Driver API surface works there.
    /// </summary>
    internal static void EnsurePrimaryContextCurrent()
    {
        if (t_primaryCtxBoundOnThisThread) return;

        if (s_primaryCtx == nint.Zero)
        {
            lock (s_primaryCtxLock)
            {
                if (s_primaryCtx == nint.Zero)
                {
                    int ir = NvrtcInterop.CuInit(0);
                    if (ir != 0) throw new InvalidOperationException($"cuInit failed: {ir}");

                    int dr = NvrtcInterop.DeviceGet(out int dev, 0);
                    if (dr != 0) throw new InvalidOperationException($"cuDeviceGet failed: {dr}");

                    int rr = NvrtcInterop.DevicePrimaryCtxRetain(out nint ctx, dev);
                    if (rr != 0) throw new InvalidOperationException($"cuDevicePrimaryCtxRetain failed: {rr}");

                    s_primaryCtxDevice = dev;
                    s_primaryCtx       = ctx;
                }
            }
        }

        int sr = NvrtcInterop.CtxSetCurrent(s_primaryCtx);
        if (sr != 0) throw new InvalidOperationException($"cuCtxSetCurrent failed: {sr}");
        t_primaryCtxBoundOnThisThread = true;
    }

    /// <summary>Combined CUDA source for image + text + weight-stationary kernels — one
    /// NVRTC compilation (the cubin cache key hashes this, so adding a source invalidates it).</summary>
    private static string CombinedKernelSource =>
        CudaKernels.Source + "\n" + CudaTextKernels.Source + "\n" + CudaWsKernels.Source +
        "\n" + CudaRaggedKernels.Source;

    private void CompileAndLoadKernels()
    {
        // Ensure the CUDA Driver API context exists (shares the primary context with the runtime).
        NvrtcInterop.CuInit(0);

        // Try to load from cubin cache first (avoids both NVRTC compilation and PTX JIT overhead).
        string cacheFile = GetCubinCachePath();
        if (TryLoadCubinFromCache(cacheFile)) return;

        byte[] srcBytes  = NvrtcInterop.ToUtf8(CombinedKernelSource);
        byte[] nameBytes = NvrtcInterop.ToUtf8("sharpi_kernels.cu");

        nint prog = nint.Zero;
        fixed (byte* pSrc = srcBytes)
        fixed (byte* pName = nameBytes)
        {
            int r = NvrtcInterop.CreateProgram(out prog, pSrc, pName, 0, nint.Zero, nint.Zero);
            if (r != 0) throw new InvalidOperationException($"nvrtcCreateProgram failed: {r}");
        }

        byte[]? binary = null;
        try
        {
            // Compile targeting the actual GPU's SM version to get a cubin (no JIT at launch).
            // Falls back to PTX (with JIT overhead) if the SM version is unknown or cubin fails.
            string archFlag = _smVersion > 0 ? $"--gpu-architecture=sm_{_smVersion}" : "--gpu-architecture=compute_52";
            byte[] archBytes = NvrtcInterop.ToUtf8(archFlag);
            int rc;
            fixed (byte* pArch = archBytes)
            {
                nint opts = (nint)(&pArch);
                rc = NvrtcInterop.CompileProgramWithOptions(prog, 1, opts);
            }
            if (rc != 0)
            {
                NvrtcInterop.GetProgramLogSize(prog, out nuint logSize);
                byte[] logBuf = new byte[(int)logSize];
                string log;
                fixed (byte* pLog = logBuf)
                {
                    NvrtcInterop.GetProgramLog(prog, pLog);
                    log = System.Text.Encoding.UTF8.GetString(logBuf);
                }
                throw new InvalidOperationException($"nvrtcCompileProgram failed ({rc}):\n{log}");
            }

            // Prefer cubin (no JIT) over PTX (lazy JIT on first kernel launch = slow).
            // nvrtcGetCubin requires NVRTC 11.1+; fall through to PTX on older versions.
            bool isCubin = false;
            int cubinRc = -1;
            nuint cubinSize = 0;
            try
            {
                cubinRc = NvrtcInterop.GetCubinSize(prog, out cubinSize);
                if (cubinRc == 0 && cubinSize > 0)
                {
                    binary = new byte[(int)cubinSize];
                    fixed (byte* pBin = binary)
                    {
                        int r2 = NvrtcInterop.GetCubin(prog, pBin);
                        if (r2 != 0) { binary = null; cubinRc = r2; }
                        else isCubin = true;
                    }
                }
            }
            catch (EntryPointNotFoundException)
            {
                // nvrtcGetCubin / nvrtcGetCubinSize unavailable (NVRTC < 11.1).
                binary = null;
            }

            if (binary is null)
            {
                // Fall back to PTX (triggers JIT at first kernel launch, slower).
                Console.Error.WriteLine(
                    $"[CudaBackend] NVRTC produced no cubin for sm_{_smVersion} " +
                    $"(nvrtcGetCubinSize rc={cubinRc}, size={cubinSize}); falling back to PTX. " +
                    "Expect ~6× slower first-token latency from per-kernel PTX→SASS JIT.");
                NvrtcInterop.GetPTXSize(prog, out nuint ptxSize);
                binary = new byte[(int)ptxSize];
                fixed (byte* pPtx = binary)
                {
                    NvrtcInterop.GetPTX(prog, pPtx);
                }
            }

            fixed (byte* pBin = binary)
            {
                int mr = NvrtcInterop.ModuleLoadData(out _nvModule, pBin);
                if (mr != 0) throw new InvalidOperationException($"cuModuleLoadData failed: {mr}");
            }

            // Persist *only real cubin* to disk; caching PTX would make every future run
            // re-pay the JIT cost (and obscure the diagnostic above).
            if (isCubin)
            {
                try { File.WriteAllBytes(cacheFile, binary); }
                catch { /* ignore cache write failures */ }
            }
            else
            {
                // Stale PTX or partial cubin from a previous run would mask the problem.
                try { if (File.Exists(cacheFile)) File.Delete(cacheFile); }
                catch { /* ignore */ }
            }
        }
        finally
        {
            NvrtcInterop.DestroyProgram(ref prog);
        }

        LoadKernelFunctions();
        ForceEagerJit();
    }

    private string GetCubinCachePath()
    {
        // Cache key: SHA-256 of combined kernel source + SM version.
        // Any source change or GPU change invalidates the cache.
        byte[] hash = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(CombinedKernelSource + _smVersion));
        string hex = Convert.ToHexString(hash)[..16];
        return Path.Combine(Path.GetTempPath(), $"sharpi_cubin_sm{_smVersion}_{hex}.bin");
    }

    private bool TryLoadCubinFromCache(string cacheFile)
    {
        if (!File.Exists(cacheFile)) return false;
        try
        {
            byte[] cubinBuf = File.ReadAllBytes(cacheFile);

            // Defend against an older sharpi build that wrote PTX into the cubin path:
            // PTX would still load via cuModuleLoadData, but every kernel would then pay
            // PTX→SASS JIT on first launch. Real cubin is ELF — magic "\x7FELF".
            if (cubinBuf.Length < 4 ||
                cubinBuf[0] != 0x7F || cubinBuf[1] != (byte)'E' ||
                cubinBuf[2] != (byte)'L' || cubinBuf[3] != (byte)'F')
            {
                Console.Error.WriteLine(
                    $"[CudaBackend] Cubin cache at {cacheFile} is not ELF (likely stale PTX " +
                    "from an earlier build); deleting and recompiling.");
                try { File.Delete(cacheFile); } catch { }
                return false;
            }

            fixed (byte* pBin = cubinBuf)
            {
                int mr = NvrtcInterop.ModuleLoadData(out _nvModule, pBin);
                if (mr != 0) return false;
            }
            LoadKernelFunctions();
            ForceEagerJit();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Walk every loaded kernel handle and query <c>CU_FUNC_ATTRIBUTE_NUM_REGS</c>.
    /// Querying a function attribute forces the driver to finalize SASS for that kernel
    /// right now, even under lazy module loading or when the module was loaded from PTX.
    /// Without this, the first decode pays per-kernel JIT during the user's first prefill.
    /// </summary>
    private void ForceEagerJit()
    {
        ReadOnlySpan<nint> kernels = [
            _im2colKernel, _biasAddKernel, _leakyReluKernel, _scaleKernel, _addKernel,
            _addScaledKernel, _clampKernel, _pshuffleKernel, _punshuffleKernel, _upsample2xKernel,
            _rmsNormKernel, _headNormKernel, _headNormPureKernel, _siluMulKernel, _sigmoidKernel,
            _softmaxKernel, _ropeInterleavedKernel, _ropeNeoxKernel, _ropeNeoxPartialKernel,
            _ropeNeoxWithFactorsKernel, _ropeNeoxWithFactorsBatchedKernel,
            _mulKernel, _sigmoidMulInPlaceKernel, _splitQgKernel, _kvAppendKernel,
            _kvAppendBf16Kernel,
            _snapKvScoreKernel, _snapKvScoreBf16Kernel,
            _kvCompactKernel, _kvCompactBf16Kernel,
            _embedLookupF32Kernel, _embedLookupQ4KKernel, _embedLookupQ5KKernel,
            _embedLookupQ6KKernel,
            _embedLookupQ80Kernel, _embedLookupQ80BatchedKernel,
            _dequantRowsQ80Kernel, _dequantRowsQ6KKernel,
            _matvecF32Kernel, _matvecQ40Kernel, _matvecQ4KKernel, _matvecQ5KKernel, _matvecQ6KKernel,
            _matvecQ6KSoaKernel,                                       // #204
            _matvecQ80Kernel,
            _matvecF32N2Kernel, _matvecQ4KN2Kernel, _matvecQ5KN2Kernel, _matvecQ6KN2Kernel,
            _matvecQ6KN2SoaKernel,                                     // #204
            _matvecF32GemmNKernel, _matvecQ4KGemmNKernel, _matvecQ5KGemmNKernel, _matvecQ6KGemmNKernel,
            _matvecQ6KGemmNSoaKernel,                                  // #204
            _matvecQ80GemmNKernel, _mmqQ80Kernel, _mmqQ80SoaKernel, _mmqQ4kKernel, _mmqQ4kSoaKernel,
            _matvecQ80Dp4aSoaKernel, _q80RepackSoaKernel,
            _matvecQ80SoaKernel, _matvecQ80GemmNSoaKernel, _dequantQ80F16SoaKernel,
            // #156/#160: Q4_K SoA repack + decode/N2/GEMM-N/dequant readers, and the
            // AoS dequant (all were missing — eager-JIT them so first decode pays no stutter).
            _q4kRepackSoaKernel, _matvecQ4KSoaKernel, _matvecQ4KN2SoaKernel,
            _matvecQ4KGemmNSoaKernel, _dequantQ4KF16Kernel, _dequantQ4KF16SoaKernel,
            _dequantQ6KF16Kernel, _dequantQ6KF16SoaKernel, _dequantQ5KF16Kernel,   // #162/#204
            _dequantQ40F16Kernel, _headNormPureBatchedKernel, _matvecQ40Dp4aKernel,   // #124
            _mmqQ40Kernel, _q40RepackSoaKernel, _mmqQ40SoaKernel, _matvecQ40SoaKernel,   // #124/#173
            _matvecQ40Dp4aSoaKernel, _dequantQ40F16SoaKernel,   // #124/#173
            _rmsNormBatchedKernel, _headNormBatchedKernel, _headNormQkKernel, _headNormQkBatchedKernel,
            _splitQgBatchedKernel, _ropeNeoxPartialBatchedKernel,
            _attentionKernel, _attentionBf16Kernel, _attentionSwaKernel, _attentionSwaBatchedKernel,
            _attentionSwaBf16Kernel, _attentionSwaBatchedBf16Kernel,
            _geluTanhMulKernel, _geluTanhMulStridedKernel, _softcapKernel,
            _argmaxPartialKernel, _argmaxFinalKernel, _argmaxRowsKernel,   // #219
            _clearF32Kernel, _quantizeQ81Kernel,
            _scaleRowsKernel, _moeWeightedReduceKernel,
            _tqRotateQueryKernel, _tqKvAppendKernel, _tqAttentionKernel,
            _siluInplaceKernel, _gdnConv1dDecodeKernel, _gdnL2NormPerHeadKernel,
            _gdnTileHeadsKernel, _gdnRecurrenceDecodeKernel,
            _gdnConv1dDecodeBatchedKernel, _gdnConv1dStateUpdateBatchedKernel,
            _gdnConv1dStateCaptureRingKernel,   // #290
            _gdnL2NormPerHeadBatchedKernel, _gdnTileHeadsBatchedKernel, _gdnRecurrenceScanKernel,
            _gdnChunkedPrefillKernel,
            _kvAppendBatchedKernel, _kvAppendBatchedBf16Kernel,
            _fullSeqAttentionKernel, _fullSeqAttentionBf16Kernel,
            _fullSeqAttentionGlobalKernel, _fullSeqAttentionGlobalBf16Kernel,
            _flashAttnPrefillKernel, _mmaTestM16N8K16Kernel, _flashAttnPrefillTcKernel,
            _flashAttnPrefillTc2Kernel, _flashAttnPrefillTc2Bf16Kernel, _flashAttnPrefillTc2Q8Kernel,
            _kvAppendQ8Kernel, _kvAppendBatchedQ8Kernel,
            _attentionQ8Kernel, _attentionSwaQ8Kernel, _attentionSwaBatchedQ8Kernel,
            _fullSeqAttentionQ8Kernel, _fullSeqAttentionGlobalQ8Kernel,
            // #235: flash-decoding split-KV + combine.
            _attentionSplitKvKernel, _attentionSplitKvBf16Kernel, _attentionSplitKvQ8Kernel,
            _attentionCombineKernel,
            _attentionSplitKvGroupedKernel, _attentionSplitKvGroupedBf16Kernel, _attentionSplitKvGroupedQ8Kernel,
            // #197: ragged-batched decode kernels.
            _ropeNeoxRaggedKernel, _ropeInterleavedRaggedKernel,
            _kvAppendRaggedKernel, _kvAppendRaggedBf16Kernel, _kvAppendRaggedQ8Kernel,
            _attentionRaggedKernel, _attentionRaggedBf16Kernel, _attentionRaggedQ8Kernel,
            _addBiasRowsKernel,
            _mmqQ4kSoaActsN16Kernel,       // #201
            _mmqQ4kSoaActsN16Bm32Kernel,   // #205
            _q6kRepackSoaKernel,                                       // #204
            _mmqQ6kSoaActsN16Kernel, _mmqQ6kSoaActsN16Bm32Kernel,     // #204
        ];
        foreach (nint k in kernels)
        {
            if (k != nint.Zero)
                NvrtcInterop.FuncGetAttribute(out _, NvrtcInterop.CU_FUNC_ATTRIBUTE_NUM_REGS, k);
        }
        // #194/#201: weight-stationary batched-decode matvec variants.
        ReadOnlySpan<nint[]> wsKernelSets = [
            _matvecF32WsKernels, _matvecQ4KWsKernels, _matvecQ4KWsSoaKernels,
            _matvecQ5KWsKernels, _matvecQ6KWsKernels, _matvecQ80WsKernels, _matvecQ80WsSoaKernels,
            _matvecQ6KWsSwKernels, _matvecQ6KWsSoaKernels,
        ];
        foreach (nint[] set in wsKernelSets)
            foreach (nint k in set)
            {
                if (k != nint.Zero)
                    NvrtcInterop.FuncGetAttribute(out _, NvrtcInterop.CU_FUNC_ATTRIBUTE_NUM_REGS, k);
            }
    }

    private void LoadKernelFunctions()
    {
        _im2colKernel      = GetKernelFunc("im2col");
        _biasAddKernel     = GetKernelFunc("bias_add");
        _leakyReluKernel   = GetKernelFunc("leaky_relu_inplace");
        _scaleKernel       = GetKernelFunc("scale_inplace");
        _addKernel         = GetKernelFunc("add_inplace");
        _addScaledKernel   = GetKernelFunc("add_scaled_inplace");
        _clampKernel       = GetKernelFunc("clamp_inplace");
        _pshuffleKernel    = GetKernelFunc("pixel_shuffle");
        _punshuffleKernel  = GetKernelFunc("pixel_unshuffle");
        _upsample2xKernel  = GetKernelFunc("upsample2x");

        // LLM kernels (same NVRTC module).
        _rmsNormKernel         = GetKernelFunc("llm_rmsnorm");
        _headNormKernel        = GetKernelFunc("llm_head_norm");
        _headNormPureKernel    = GetKernelFunc("llm_head_norm_pure");
        _siluMulKernel         = GetKernelFunc("llm_silu_mul");
        _sigmoidKernel         = GetKernelFunc("llm_sigmoid_inplace");
        _softmaxKernel         = GetKernelFunc("llm_softmax");
        _ropeInterleavedKernel = GetKernelFunc("llm_rope_interleaved");
        _ropeNeoxKernel        = GetKernelFunc("llm_rope_neox");
        _ropeNeoxPartialKernel = GetKernelFunc("llm_rope_neox_partial");
        _ropeNeoxWithFactorsKernel = GetKernelFunc("llm_rope_neox_with_factors");
        _mulKernel             = GetKernelFunc("llm_mul");
        _sigmoidMulInPlaceKernel = GetKernelFunc("llm_sigmoid_mul_inplace");
        _splitQgKernel         = GetKernelFunc("llm_split_qg");
        _kvAppendKernel        = GetKernelFunc("llm_kv_append");
        _kvAppendBf16Kernel    = GetKernelFunc("llm_kv_append_bf16");
        _snapKvScoreKernel     = GetKernelFunc("llm_snapkv_score");
        _snapKvScoreBf16Kernel = GetKernelFunc("llm_snapkv_score_bf16");
        _kvCompactKernel       = GetKernelFunc("llm_kv_compact");
        _kvCompactBf16Kernel   = GetKernelFunc("llm_kv_compact_bf16");
        _embedLookupF32Kernel  = GetKernelFunc("llm_embed_lookup_f32");
        _embedLookupQ4KKernel  = GetKernelFunc("llm_embed_lookup_q4k");
        _embedLookupQ5KKernel  = GetKernelFunc("llm_embed_lookup_q5k");
        _embedLookupQ6KKernel  = GetKernelFunc("llm_embed_lookup_q6k");
        _embedLookupQ80Kernel  = GetKernelFunc("llm_embed_lookup_q8_0");
        _embedLookupQ80BatchedKernel = GetKernelFunc("llm_embed_lookup_q8_0_batched");
        _dequantRowsQ80Kernel  = GetKernelFunc("llm_dequant_rows_q8_0");
        _dequantRowsQ6KKernel  = GetKernelFunc("llm_dequant_rows_q6k");
        _matvecF32Kernel       = GetKernelFunc("llm_matvec_f32");
        _matvecQ4KKernel       = GetKernelFunc("llm_matvec_q4k");
        _matvecQ4KSoaKernel    = GetKernelFunc("llm_matvec_q4k_soa");
        _q4kRepackSoaKernel    = GetKernelFunc("llm_q4k_repack_soa");
        _q6kRepackSoaKernel    = GetKernelFunc("llm_q6k_repack_soa");   // #204
        _matvecQ5KKernel       = GetKernelFunc("llm_matvec_q5k");
        _matvecQ6KKernel       = GetKernelFunc("llm_matvec_q6k");
        _matvecQ6KSoaKernel    = GetKernelFunc("llm_matvec_q6k_soa");   // #204
        _matvecQ40Kernel       = GetKernelFunc("llm_matvec_q4_0");
        _matvecQ40Dp4aKernel   = GetKernelFunc("llm_matvec_q4_0_dp4a");   // #124
        _matvecQ80Kernel       = GetKernelFunc("llm_matvec_q8_0");
        _matvecQ80Dp4aKernel   = GetKernelFunc("llm_matvec_q8_0_dp4a");
        _matvecF32N2Kernel     = GetKernelFunc("llm_matvec_f32_n2");
        _matvecQ4KN2Kernel     = GetKernelFunc("llm_matvec_q4k_n2");
        _matvecQ4KN2SoaKernel  = GetKernelFunc("llm_matvec_q4k_n2_soa");
        _matvecQ5KN2Kernel     = GetKernelFunc("llm_matvec_q5k_n2");
        _matvecQ6KN2Kernel     = GetKernelFunc("llm_matvec_q6k_n2");
        _matvecQ6KN2SoaKernel  = GetKernelFunc("llm_matvec_q6k_n2_soa");   // #204
        _matvecF32GemmNKernel  = GetKernelFunc("llm_matvec_f32_gemm_n");
        _matvecQ4KGemmNKernel  = GetKernelFunc("llm_matvec_q4k_gemm_n");
        _matvecQ4KGemmNSoaKernel = GetKernelFunc("llm_matvec_q4k_gemm_n_soa");
        _matvecQ5KGemmNKernel  = GetKernelFunc("llm_matvec_q5k_gemm_n");
        _matvecQ6KGemmNKernel  = GetKernelFunc("llm_matvec_q6k_gemm_n");
        _matvecQ6KGemmNSoaKernel = GetKernelFunc("llm_matvec_q6k_gemm_n_soa");   // #204
        _matvecQ80GemmNKernel  = GetKernelFunc("llm_matvec_q8_0_gemm_n");
        for (int v = 0; v < CudaWsKernels.Variants.Length; v++)   // #194
        {
            int nt = CudaWsKernels.Variants[v];
            _matvecF32WsKernels[v]    = GetKernelFunc($"llm_matvec_f32_ws_n{nt}");
            _matvecQ4KWsKernels[v]    = GetKernelFunc($"llm_matvec_q4k_ws_n{nt}");
            _matvecQ4KWsSoaKernels[v] = GetKernelFunc($"llm_matvec_q4k_ws_soa_n{nt}");
            _matvecQ5KWsKernels[v]    = GetKernelFunc($"llm_matvec_q5k_ws_n{nt}");
            _matvecQ6KWsKernels[v]    = GetKernelFunc($"llm_matvec_q6k_ws_n{nt}");
            _matvecQ80WsKernels[v]    = GetKernelFunc($"llm_matvec_q8_0_ws_n{nt}");
            _matvecQ80WsSoaKernels[v] = GetKernelFunc($"llm_matvec_q8_0_ws_soa_n{nt}");
            _matvecQ6KWsSwKernels[v]    = GetKernelFunc($"llm_matvec_q6k_ws_sw_n{nt}");      // #201
            _matvecQ6KWsSoaKernels[v]   = GetKernelFunc($"llm_matvec_q6k_ws_soa_n{nt}");     // #204
        }
        _mmqQ4kSoaActsN16Kernel = GetKernelFunc("llm_mmq_q4k_soa_acts_n16");   // #201
        _mmqQ4kSoaActsN16Bm32Kernel = GetKernelFunc("llm_mmq_q4k_soa_acts_n16_bm32");   // #205
        _mmqQ6kSoaActsN16Kernel = GetKernelFunc("llm_mmq_q6k_soa_acts_n16");   // #204
        _mmqQ6kSoaActsN16Bm32Kernel = GetKernelFunc("llm_mmq_q6k_soa_acts_n16_bm32");   // #204
        _dequantQ80F16Kernel   = GetKernelFunc("llm_dequant_q8_0_to_f16");
        _dequantQ4KF16Kernel   = GetKernelFunc("llm_dequant_q4k_to_f16");
        _dequantQ4KF16SoaKernel = GetKernelFunc("llm_dequant_q4k_to_f16_soa");
        _dequantQ6KF16Kernel   = GetKernelFunc("llm_dequant_q6k_to_f16");
        _dequantQ6KF16SoaKernel = GetKernelFunc("llm_dequant_q6k_to_f16_soa");   // #204
        _dequantQ5KF16Kernel   = GetKernelFunc("llm_dequant_q5k_to_f16");
        _dequantQ40F16Kernel   = GetKernelFunc("llm_dequant_q4_0_to_f16");   // #124
        _headNormPureBatchedKernel = GetKernelFunc("llm_head_norm_pure_batched");   // #124
        _f32ToF16Kernel        = GetKernelFunc("llm_f32_to_f16");
        _mmqQ80Kernel          = GetKernelFunc("llm_mmq_q8_0");
        _mmqQ80SoaKernel       = GetKernelFunc("llm_mmq_q8_0_soa");
        _mmqQ4kKernel          = GetKernelFunc("llm_mmq_q4k");
        _mmqQ4kSoaKernel       = GetKernelFunc("llm_mmq_q4k_soa");
        _mmqQ40Kernel          = GetKernelFunc("llm_mmq_q4_0");   // #124/#173
        _q40RepackSoaKernel    = GetKernelFunc("llm_q4_0_repack_soa");   // #124/#173
        _mmqQ40SoaKernel       = GetKernelFunc("llm_mmq_q4_0_soa");
        _matvecQ40SoaKernel    = GetKernelFunc("llm_matvec_q4_0_soa");
        _matvecQ40Dp4aSoaKernel = GetKernelFunc("llm_matvec_q4_0_dp4a_soa");
        _dequantQ40F16SoaKernel = GetKernelFunc("llm_dequant_q4_0_to_f16_soa");
        _matvecQ80Dp4aSoaKernel = GetKernelFunc("llm_matvec_q8_0_dp4a_soa");
        _q80RepackSoaKernel    = GetKernelFunc("llm_q8_0_repack_soa");
        _matvecQ80SoaKernel    = GetKernelFunc("llm_matvec_q8_0_soa");
        _matvecQ80GemmNSoaKernel = GetKernelFunc("llm_matvec_q8_0_gemm_n_soa");
        _dequantQ80F16SoaKernel = GetKernelFunc("llm_dequant_q8_0_to_f16_soa");
        _flashAttnPrefillKernel = GetKernelFunc("llm_flash_attn_prefill_f32");
        _mmaTestM16N8K16Kernel  = GetKernelFunc("llm_mma_test_m16n8k16_f32");
        _flashAttnPrefillTcKernel = GetKernelFunc("llm_flash_attn_prefill_tc");
        _flashAttnPrefillTc2Kernel = GetKernelFunc("llm_flash_attn_prefill_tc2");
        _flashAttnPrefillTc2Bf16Kernel = GetKernelFunc("llm_flash_attn_prefill_tc2_bf16");
        _flashAttnPrefillTc2Q8Kernel = GetKernelFunc("llm_flash_attn_prefill_tc2_q8_0");
        _rmsNormBatchedKernel  = GetKernelFunc("llm_rmsnorm_batched");
        _headNormBatchedKernel = GetKernelFunc("llm_head_norm_batched");
        _headNormQkKernel        = GetKernelFunc("llm_head_norm_qk");
        _headNormQkBatchedKernel = GetKernelFunc("llm_head_norm_qk_batched");
        _splitQgBatchedKernel  = GetKernelFunc("llm_split_qg_batched");
        _ropeNeoxPartialBatchedKernel = GetKernelFunc("llm_rope_neox_partial_batched");
        _ropeNeoxWithFactorsBatchedKernel = GetKernelFunc("llm_rope_neox_with_factors_batched");
        _attentionKernel       = GetKernelFunc("llm_attention");
        _attentionBf16Kernel   = GetKernelFunc("llm_attention_bf16");
        _attentionQ8Kernel     = GetKernelFunc("llm_attention_q8_0");
        _attentionSwaKernel    = GetKernelFunc("llm_attention_swa");
        _attentionSwaBatchedKernel = GetKernelFunc("llm_attention_swa_batched");
        _attentionSwaBf16Kernel = GetKernelFunc("llm_attention_swa_bf16");
        _attentionSwaBatchedBf16Kernel = GetKernelFunc("llm_attention_swa_batched_bf16");
        _attentionSwaQ8Kernel = GetKernelFunc("llm_attention_swa_q8_0");
        _attentionSwaBatchedQ8Kernel = GetKernelFunc("llm_attention_swa_batched_q8_0");
        _attentionSplitKvKernel     = GetKernelFunc("llm_attention_splitkv");        // flash-decoding (#235)
        _attentionSplitKvBf16Kernel = GetKernelFunc("llm_attention_splitkv_bf16");
        _attentionSplitKvQ8Kernel   = GetKernelFunc("llm_attention_splitkv_q8_0");
        _attentionSplitKvGroupedKernel     = GetKernelFunc("llm_attention_splitkv_grouped");      // GQA head-sharing (#237)
        _attentionSplitKvGroupedBf16Kernel = GetKernelFunc("llm_attention_splitkv_grouped_bf16");
        _attentionSplitKvGroupedQ8Kernel   = GetKernelFunc("llm_attention_splitkv_grouped_q8_0");
        _attentionCombineKernel     = GetKernelFunc("llm_attention_combine");
        _kvAppendQ8Kernel      = GetKernelFunc("llm_kv_append_q8_0");
        _geluTanhMulKernel     = GetKernelFunc("llm_gelu_tanh_mul");
        _geluTanhMulStridedKernel = GetKernelFunc("llm_gelu_tanh_mul_strided");
        _softcapKernel         = GetKernelFunc("llm_softcap_inplace");
        _argmaxPartialKernel   = GetKernelFunc("llm_argmax_partial");   // #219
        _argmaxFinalKernel     = GetKernelFunc("llm_argmax_final");     // #219
        _argmaxRowsKernel      = GetKernelFunc("llm_argmax_rows");      // #219
        _clearF32Kernel        = GetKernelFunc("llm_clear_f32");
        _quantizeQ81Kernel     = GetKernelFunc("llm_quantize_q8_1");
        _quantizeQ81SoaKernel  = GetKernelFunc("llm_quantize_q8_1_soa");      // Track A (#124/#173)
        _mmqQ80SoaActsKernel   = GetKernelFunc("llm_mmq_q8_0_soa_acts");      // Track A (#124/#173)
        _mmqQ4kSoaActsKernel   = GetKernelFunc("llm_mmq_q4k_soa_acts");       // Track A (#124/#173)
        _mmqQ40SoaActsKernel   = GetKernelFunc("llm_mmq_q4_0_soa_acts");      // Track A (#124/#173)
        _mmqQ80SoaActsCpaKernel = GetKernelFunc("llm_mmq_q8_0_soa_acts_cpa"); // Track B (#124/#173)
        _mmqQ40SoaActsCpaKernel = GetKernelFunc("llm_mmq_q4_0_soa_acts_cpa"); // Track B (#124/#173)
        _bwBaselineKernel      = GetKernelFunc("llm_bw_baseline");
        _scaleRowsKernel       = GetKernelFunc("llm_scale_rows_inplace");
        _moeWeightedReduceKernel = GetKernelFunc("llm_moe_weighted_reduce");

        // TurboQuant kernels (loaded from the same NVRTC module).
        _tqRotateQueryKernel = GetKernelFunc("llm_tq_rotate_query");
        _tqKvAppendKernel    = GetKernelFunc("llm_tq_kv_append");
        _tqAttentionKernel   = GetKernelFunc("llm_tq_attention");

        // qwen35moe GDN kernels.
        _siluInplaceKernel        = GetKernelFunc("llm_silu_inplace");
        _gdnConv1dDecodeKernel    = GetKernelFunc("llm_gdn_conv1d_decode");
        _gdnL2NormPerHeadKernel   = GetKernelFunc("llm_gdn_l2_norm_per_head");
        _gdnTileHeadsKernel       = GetKernelFunc("llm_gdn_tile_heads");
        _gdnRecurrenceDecodeKernel = GetKernelFunc("llm_gdn_recurrence_decode");

        // qwen35moe GDN batched trunk + batched-query SDPA kernels (issue #114-B).
        _gdnConv1dDecodeBatchedKernel      = GetKernelFunc("llm_gdn_conv1d_decode_batched");
        _gdnConv1dStateUpdateBatchedKernel = GetKernelFunc("llm_gdn_conv1d_state_update_batched");
        _gdnConv1dStateCaptureRingKernel   = GetKernelFunc("llm_gdn_conv1d_state_capture_ring");   // #290
        _gdnL2NormPerHeadBatchedKernel     = GetKernelFunc("llm_gdn_l2_norm_per_head_batched");
        _gdnTileHeadsBatchedKernel         = GetKernelFunc("llm_gdn_tile_heads_batched");
        _gdnRecurrenceScanKernel           = GetKernelFunc("llm_gdn_recurrence_scan");
        _gdnChunkedPrefillKernel           = GetKernelFunc("llm_gdn_chunked_prefill");
        _kvAppendBatchedKernel             = GetKernelFunc("llm_kv_append_batched");
        _kvAppendBatchedBf16Kernel         = GetKernelFunc("llm_kv_append_batched_bf16");
        _kvAppendBatchedQ8Kernel           = GetKernelFunc("llm_kv_append_batched_q8_0");
        _fullSeqAttentionKernel            = GetKernelFunc("llm_full_seq_attention");
        _fullSeqAttentionBf16Kernel        = GetKernelFunc("llm_full_seq_attention_bf16");
        _fullSeqAttentionQ8Kernel          = GetKernelFunc("llm_full_seq_attention_q8_0");
        _fullSeqAttentionGlobalKernel      = GetKernelFunc("llm_full_seq_attention_global");
        _fullSeqAttentionGlobalBf16Kernel  = GetKernelFunc("llm_full_seq_attention_global_bf16");
        _fullSeqAttentionGlobalQ8Kernel    = GetKernelFunc("llm_full_seq_attention_global_q8_0");

        // Issue #197: ragged-batched decode kernels.
        _ropeNeoxRaggedKernel        = GetKernelFunc("llm_rope_neox_ragged");
        _ropeInterleavedRaggedKernel = GetKernelFunc("llm_rope_interleaved_ragged");
        _kvAppendRaggedKernel        = GetKernelFunc("llm_kv_append_ragged");
        _kvAppendRaggedBf16Kernel    = GetKernelFunc("llm_kv_append_ragged_bf16");
        _kvAppendRaggedQ8Kernel      = GetKernelFunc("llm_kv_append_ragged_q8_0");
        _attentionRaggedKernel       = GetKernelFunc("llm_attention_ragged");
        _attentionRaggedBf16Kernel   = GetKernelFunc("llm_attention_ragged_bf16");
        _attentionRaggedQ8Kernel     = GetKernelFunc("llm_attention_ragged_q8_0");
        _addBiasRowsKernel           = GetKernelFunc("llm_add_bias_rows");
    }

    /// <summary>
    /// Diagnostic: launches a pure-streaming copy kernel over <paramref name="bytes"/>
    /// bytes (must be multiple of 4) so the caller can measure achievable HBM bandwidth.
    /// </summary>
    public void RunBandwidthBaseline(nint srcDev, nint dstDev, int bytes)
    {
        EnsureImageKernels();
        int n_uint4 = bytes / 16;
        nint p0 = srcDev, p1 = dstDev;
        int  p2 = n_uint4;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        uint grid = (uint)((n_uint4 + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_bwBaselineKernel, grid, 1, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(bw_baseline) failed: {r}");
    }

    private nint GetKernelFunc(string name)
    {
        byte[] nameBytes = NvrtcInterop.ToUtf8(name);
        fixed (byte* pName = nameBytes)
        {
            int r = NvrtcInterop.ModuleGetFunction(out nint func, _nvModule, pName);
            if (r != 0) throw new InvalidOperationException($"cuModuleGetFunction({name}) failed: {r}");
            return func;
        }
    }

    /// <summary>Launch a 1-D kernel with 1024 threads per block over <paramref name="total"/> elements.</summary>
    private void Launch1D(nint kernel, int total, nint* args)
    {
        uint grid = (uint)((total + 1023) / 1024);
        int r = NvrtcInterop.LaunchKernel(kernel, grid, 1, 1, 1024, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel failed: {r}");
    }

    /// <summary>
    /// Ensure the GPU im2col tile buffer is at least <paramref name="minBytes"/> bytes.
    /// On first call, allocates exactly <see cref="MaxTileBytes"/> (2.5 GiB) so that all
    /// possible tile sizes for any RRDB or upsample layer fit in a single tile without
    /// reallocation. Single-tile mode keeps lda=ldc=N so all cuBLAS reads/writes are
    /// contiguous — multi-tile with strided ldc is never needed.
    /// </summary>
    private void EnsureIm2ColBuf(long minBytes)
    {
        if (_im2colBuf != nint.Zero && _im2colBufSize >= (nuint)minBytes) return;
        if (_im2colBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_im2colBuf);
            _im2colBuf     = nint.Zero;
            _im2colBufSize = 0;
        }
        // Allocate MaxTileBytes so subsequent calls never need to grow.
        // All valid tilePixels (aligned to full rows) produce minBytes ≤ MaxTileBytes.
        nuint newSize = (nuint)MaxTileBytes;
        int r = CuBlasInterop.CudaMalloc(out _im2colBuf, newSize);
        if (r != 0) throw new InvalidOperationException($"cudaMalloc({newSize / 1024 / 1024} MiB im2col buf) failed: {r}");
        _im2colBufSize = newSize;
    }

    /// <inheritdoc/>
    public Tensor Conv2d(Tensor input, Tensor weight, Tensor bias,
                         int inCh, int outCh, int h, int w, int ksize, int padding = -1)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if (ksize != 3)
            throw new NotSupportedException($"Conv2d CUDA only supports ksize=3 (got ksize={ksize}).");

        int N = h * w;
        int K = inCh * 9;   // im2col columns

        nint inputPtr  = GetDevPtr(input);
        nint weightPtr = GetDevPtr(weight);
        nint biasPtr   = GetDevPtr(bias);

        // ── Output allocation (CHW: [outCh, N] row-major) ──────────────────
        var  output    = Allocate(TensorShape.D1((long)outCh * N));
        nint outputPtr = GetDevPtr(output);

        // ── Tile size ───────────────────────────────────────────────────────
        // With MaxTileBytes=2.5 GiB, every real layer fits in a single tile:
        //   RRDB max (K=1728, N=262144): 1.81 GiB  < 2.5 GiB ✓
        //   Upsample  (K=576,  N=4M):   2.41 GiB  < 2.5 GiB ✓
        // Single-tile: lda=tileN=N, ldc=N — all cuBLAS accesses are contiguous.
        int tilePixels = (int)Math.Min((long)N, MaxTileBytes / ((long)K * sizeof(float)));
        tilePixels = Math.Max(tilePixels, w); // at least one full row per tile

        // Align tile to complete rows so ph_start = pixel_start / w is integer
        tilePixels = (tilePixels / w) * w;

        EnsureIm2ColBuf((long)tilePixels * K * sizeof(float));

        float alpha = 1.0f, beta = 0.0f;

        // Hoist kernel-arg pointers outside the loop (CA2014: no stackalloc in loops).
        // Only cp5 (ph_start) and cp6 (tileN) vary per tile; we update them before each launch.
        nint cp0 = inputPtr, cp1 = _im2colBuf;
        int  cp2 = h, cp3 = w, cp4 = N, cp5 = 0, cp6 = 0, cp7 = inCh, cp8 = K;
        nint* args = stackalloc nint[9]
        {
            (nint)(&cp0), (nint)(&cp1),
            (nint)(&cp2), (nint)(&cp3), (nint)(&cp4),
            (nint)(&cp5), (nint)(&cp6),
            (nint)(&cp7), (nint)(&cp8)
        };

        for (int pixelStart = 0; pixelStart < N; pixelStart += tilePixels)
        {
            int tileN    = Math.Min(tilePixels, N - pixelStart);
            cp5 = pixelStart / w;  // ph_start
            cp6 = tileN;

            // ── im2col kernel: fills _im2colBuf[K, tileN] ──────────────────
            // Block (32=pixel, 8=k) — consecutive tx (pixel) → coalesced writes.
            // Grid (ceil(tileN/32), ceil(K/8)).
            {
                uint grX = ((uint)tileN + 31) / 32;
                uint grY = ((uint)K     +  7) / 8;
                int er = NvrtcInterop.LaunchKernel(_im2colKernel, grX, grY, 1, 32, 8, 1, 0, _stream, args, null);
                if (er != 0) throw new InvalidOperationException($"im2col launch failed: {er}");
            }

            // ── GEMM: C = A*B where A=col[K,tileN], B=weight[K,outCh], C=out[tileN,outCh] ─
            // col[K, tileN]: column k starts at k*tileN → lda=tileN (contiguous columns).
            // weight[outCh, K] row-major = [K, outCh] col-major → ldb=K.
            // Output at outputPtr + pixelStart, ldc=N → C[pixel, oc] = out[pixelStart+pixel+oc*N].
            nint gemmDst = outputPtr + (nint)(pixelStart * sizeof(float));
            int gr = CuBlasInterop.Sgemm(
                _handle,
                CuBlasInterop.OpN, CuBlasInterop.OpN,
                tileN, outCh, K,
                ref alpha,
                _im2colBuf, tileN,   // A=[K,tileN] col-major, lda=tileN
                weightPtr,  K,       // B=[K,outCh] col-major, ldb=K
                ref beta,
                gemmDst, N);         // C=[tileN,outCh] col-major, ldc=N
            if (gr != 0) throw new InvalidOperationException($"cublasSgemm (tile {pixelStart}/{N}) failed: {gr}");
        }

        // ── Bias: output[oc, pixel] += bias[oc]  (full output, one kernel) ─
        nint bp0 = outputPtr, bp1 = biasPtr;
        int  bp2 = N, bp3 = outCh;
        nint* bargs = stackalloc nint[4] { (nint)(&bp0), (nint)(&bp1), (nint)(&bp2), (nint)(&bp3) };
        uint grBias = ((uint)N + 255) / 256;
        int br = NvrtcInterop.LaunchKernel(_biasAddKernel, grBias, 1, 1, 256, 1, 1, 0, _stream, bargs, null);
        if (br != 0) throw new InvalidOperationException($"bias_add launch failed: {br}");

        return output;
    }

    /// <inheritdoc/>
    public void LeakyReluInPlace(Tensor x, float negSlope)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint p0 = GetDevPtr(x);
        float p1 = negSlope;
        int   p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_leakyReluKernel, n, args);
    }

    /// <inheritdoc/>
    public void ScaleInPlace(Tensor x, float scale)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint  p0 = GetDevPtr(x);
        float p1 = scale;
        int   p2 = n;
        nint* args = stackalloc nint[3] { (nint)(&p0), (nint)(&p1), (nint)(&p2) };
        Launch1D(_scaleKernel, n, args);
    }

    /// <inheritdoc/>
    public void AddScaledInPlace(Tensor dst, Tensor src, float scale)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)dst.ElementCount;
        nint  p0 = GetDevPtr(dst);
        nint  p1 = GetDevPtr(src);
        float p2 = scale;
        int   p3 = n;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        Launch1D(_addScaledKernel, n, args);
    }

    /// <summary>
    /// Issue #129: per-row scalar multiply over a [rows × cols] buffer:
    /// <c>buf[i*cols + e] *= scales[i]</c>. The device <paramref name="scales"/> buffer
    /// holds one scalar per row. Bit-identical to calling
    /// <see cref="ScaleInPlace"/>(row_i, scales[i]) once per row — a single float
    /// multiply per element, rounded to float. Used to apply the per-token shared-expert
    /// sigmoid gate to the batched shared-expert down output in one launch.
    /// </summary>
    public void ScaleRowsInPlace(Tensor buf, Tensor scales, int rows, int cols)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if ((uint)rows > 65535)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, "ScaleRowsInPlace: rows must fit the CUDA gridDim.y limit (65535).");

        // 2D grid: x walks columns (256-wide blocks), y is the row — the kernel
        // recovers (i, e) from block/thread indices with no integer divide.
        nint p0 = GetDevPtr(buf);
        nint p1 = GetDevPtr(scales);
        int  p2 = rows, p3 = cols;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        uint gridX = (uint)((cols + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_scaleRowsKernel, gridX, (uint)rows, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(scale_rows) failed: {r}");
    }

    /// <summary>
    /// Issue #129: batched MoE top-k weighted scatter-reduce + shared-expert add, in one
    /// launch over all N tokens. For each (token i, element e):
    /// <c>acc = Σ_k downPartial[(i*na+k)*embDim+e] * weights[i*na+k]; acc += shared[i*embDim+e]; shared[i*embDim+e] = acc;</c>
    /// The per-k <c>acc += partial*weight</c> contracts to <c>fmaf</c> (NVRTC fmad=true),
    /// one rounding per term, exactly matching the sequential <c>AddScaledInPlace</c> loop;
    /// the final <c>acc += shared</c> is a plain add (one rounding), matching the sequential
    /// <c>AddInPlace</c>. Routed slots are summed in k=0..na-1 order, shared added last —
    /// byte-identical to the per-token <c>Clear + AddScaledInPlace×na + AddInPlace</c>.
    /// <paramref name="shared"/> is in/out (must already hold the scaled+rounded shared
    /// output per token); each thread owns its (i,e) element so the read-modify-write is
    /// race-free.
    /// </summary>
    public void MoeWeightedReduce(Tensor downPartial, Tensor weights, Tensor shared,
                                  int N, int na, int embDim)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");
        if ((uint)N > 65535)
            throw new ArgumentOutOfRangeException(nameof(N), N, "MoeWeightedReduce: N must fit the CUDA gridDim.y limit (65535).");

        // 2D grid: x walks embDim (256-wide blocks), y is the token — the kernel
        // recovers (i, e) from block/thread indices with no integer divide/modulo.
        nint p0 = GetDevPtr(downPartial);
        nint p1 = GetDevPtr(weights);
        nint p2 = GetDevPtr(shared);
        int  p3 = N, p4 = na, p5 = embDim;
        nint* args = stackalloc nint[6]
            { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5) };
        uint gridX = (uint)((embDim + 255) / 256);
        int r = NvrtcInterop.LaunchKernel(_moeWeightedReduceKernel, gridX, (uint)N, 1, 256, 1, 1, 0, _stream, args, null);
        if (r != 0) throw new InvalidOperationException($"cuLaunchKernel(moe_weighted_reduce) failed: {r}");
    }

    /// <inheritdoc/>
    public void ClampInPlace(Tensor x, float min, float max)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int n = (int)x.ElementCount;
        nint  p0 = GetDevPtr(x);
        float p1 = min, p2 = max;
        int   p3 = n;
        nint* args = stackalloc nint[4] { (nint)(&p0), (nint)(&p1), (nint)(&p2), (nint)(&p3) };
        Launch1D(_clampKernel, n, args);
    }

    /// <inheritdoc/>
    public Tensor CatChannels(Tensor a, int aCh, Tensor b, int bCh, int hw)
    {
        var output = Allocate(TensorShape.D1((long)(aCh + bCh) * hw));
        nint outPtr = GetDevPtr(output);
        nint aPtr   = GetDevPtr(a);
        nint bPtr   = GetDevPtr(b);
        nuint aBytes = (nuint)(aCh * hw * sizeof(float));
        nuint bBytes = (nuint)(bCh * hw * sizeof(float));
        // Two async DMA copies on the same stream — no kernel dispatch overhead.
        CuBlasInterop.CudaMemcpyAsync(outPtr,           aPtr, aBytes, CuBlasInterop.DeviceToDevice, _stream);
        CuBlasInterop.CudaMemcpyAsync(outPtr + (nint)aBytes, bPtr, bBytes, CuBlasInterop.DeviceToDevice, _stream);
        return output;
    }

    /// <inheritdoc/>
    public Tensor PixelShuffleGpu(Tensor input, int inCh, int h, int w, int upscaleFactor)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int outCh = inCh / (upscaleFactor * upscaleFactor);
        var output = Allocate(TensorShape.D1((long)outCh * h * upscaleFactor * w * upscaleFactor));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        int  p2 = outCh, p3 = h, p4 = w, p5 = upscaleFactor;
        nint* args = stackalloc nint[6]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5)
        };
        Launch1D(_pshuffleKernel, outCh * h * upscaleFactor * w * upscaleFactor, args);
        return output;
    }

    /// <inheritdoc/>
    public Tensor PixelUnshuffleGpu(Tensor input, int inCh, int h, int w, int downscaleFactor)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        int outCh = inCh * downscaleFactor * downscaleFactor;
        int outH  = h / downscaleFactor;
        int outW  = w / downscaleFactor;
        var output = Allocate(TensorShape.D1((long)outCh * outH * outW));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        // kernel signature: (input, output, inCh, outH, outW, r)
        int  p2 = inCh, p3 = outH, p4 = outW, p5 = downscaleFactor;
        nint* args = stackalloc nint[6]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4), (nint)(&p5)
        };
        Launch1D(_punshuffleKernel, outCh * outH * outW, args);
        return output;
    }

    /// <inheritdoc/>
    public Tensor Upsample2xGpu(Tensor input, int ch, int h, int w)
    {
        EnsureImageKernels();
        if (!_imageKernelsAvailable)
            throw new NotSupportedException("NVRTC is not available; cannot run CUDA image kernels.");

        var output = Allocate(TensorShape.D1((long)ch * h * 2 * w * 2));
        nint p0 = GetDevPtr(input), p1 = GetDevPtr(output);
        int  p2 = ch, p3 = h, p4 = w;
        nint* args = stackalloc nint[5]
        {
            (nint)(&p0), (nint)(&p1),
            (nint)(&p2), (nint)(&p3), (nint)(&p4)
        };
        Launch1D(_upsample2xKernel, ch * h * 2 * w * 2, args);
        return output;
    }

    /// <inheritdoc/>
    /// <remarks>No-op: CUDA streams are already asynchronous.</remarks>
    public void BeginBatch() { }

    /// <inheritdoc/>
    /// <remarks>No-op: CUDA kernels on the same stream execute in order.</remarks>
    public void BatchBarrier() { }

    /// <inheritdoc/>
    /// <remarks>
    /// No-op for CUDA: all kernels are queued on <c>_stream</c> and execute in order,
    /// so no explicit submission or synchronisation is needed between RDB blocks.
    /// The stream is synchronised exactly once at <see cref="Download"/> time.
    /// </remarks>
    public void EndBatch() { }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var entry in _devPtrs.Values)
            CuBlasInterop.CudaFree(entry.devPtr);
        _devPtrs.Clear();
        _soaHandles.Clear();   // #149
        _soaQ4kHandles.Clear();   // #156
        _soaQ6kHandles.Clear();   // #204
        _soaQ40Handles.Clear();   // #124/#173

        _pool.Dispose();

        if (_nvModule != nint.Zero)
        {
            NvrtcInterop.ModuleUnload(_nvModule);
            _nvModule = nint.Zero;
        }

        if (_im2colBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_im2colBuf);
            _im2colBuf = nint.Zero;
        }
        if (_q81Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81Buf);
            _q81Buf = nint.Zero;
            _q81BufSize = 0;
        }
        if (_q81BufB != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BufB);
            _q81BufB = nint.Zero;
            _q81BufBSize = 0;
        }
        if (_q81BatchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchBuf);
            _q81BatchBuf = nint.Zero;
            _q81BatchBufSize = 0;
        }
        if (_q81BatchSoaBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_q81BatchSoaBuf);
            _q81BatchSoaBuf = nint.Zero;
            _q81BatchSoaBufSize = 0;
        }
        if (_gemmWf16Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_gemmWf16Buf);
            _gemmWf16Buf = nint.Zero;
            _gemmWf16Size = 0;
        }
        if (_gemmAf16Buf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_gemmAf16Buf);
            _gemmAf16Buf = nint.Zero;
            _gemmAf16Size = 0;
        }
        if (_waveScratchBuf != nint.Zero)
        {
            CuBlasInterop.CudaFree(_waveScratchBuf);
            _waveScratchBuf = nint.Zero;
            _waveScratchBufSize = 0;
        }

        // Tear down any captured CUDA graph (issue #136) before stream/context go away.
        if (_graphCapturing)
            NvrtcInterop.StreamEndCapture(_stream, out _);
        DiscardGraph();

        CuBlasInterop.Destroy(_handle);

        if (_stream != nint.Zero)
            CuBlasInterop.StreamDestroy(_stream);

        if (_uploadStream != nint.Zero)
        {
            // Drain any straggling background DMA so we don't tear down the stream
            // out from under a still-in-flight transfer.
            CuBlasInterop.StreamSynchronize(_uploadStream);
            CuBlasInterop.StreamDestroy(_uploadStream);
            _uploadStream = nint.Zero;
        }

        // Free the async staging ring: each slot's pinned buffer + its backend-owned fence
        // event (the upload stream was synchronized + destroyed above, so all DMAs are done).
        for (int i = 0; i < AsyncRingSlots; i++)
        {
            if (_asyncRingFence[i] != nint.Zero)
            {
                CuBlasInterop.EventDestroy(_asyncRingFence[i]);
                _asyncRingFence[i] = nint.Zero;
            }
            if (_asyncRingBuf[i] != nint.Zero)
            {
                CuBlasInterop.FreeHost(_asyncRingBuf[i]);
                _asyncRingBuf[i] = nint.Zero;
                _asyncRingSize[i] = 0;
            }
        }

        if (_pinnedBuf != nint.Zero)
            CuBlasInterop.FreeHost(_pinnedBuf);
    }
}

/// <summary>
/// Pool of reusable CUDA device buffers keyed by rounded allocation size.
/// Eliminates the cudaMalloc/cudaFree overhead on the hot path (one pair per GEMM call).
/// Sizes are rounded up to the next power-of-two so all Allocate/Upload callers must use
/// RoundUp() when deciding how many bytes to cudaMalloc — this guarantees a pooled pointer
/// is always large enough for any request that maps to the same bucket.
/// Thread-safe via per-bucket ConcurrentStack.
/// </summary>
internal sealed class GpuBufferPool : IDisposable
{
    // One stack of available device pointers per power-of-two bucket.
    private readonly ConcurrentDictionary<nuint, ConcurrentStack<nint>> _buckets = new();
    private bool _disposed;

    /// <summary>Round <paramref name="v"/> up to the next power-of-two (minimum 64 bytes).</summary>
    public static nuint RoundUp(nuint v)
    {
        if (v <= 64) return 64;
        v--;
        v |= v >> 1; v |= v >> 2; v |= v >> 4;
        v |= v >> 8; v |= v >> 16; v |= v >> 32;
        return v + 1;
    }

    /// <summary>
    /// Return a cached device pointer for a bucket of exactly <paramref name="bucketSize"/> bytes
    /// (must be a power-of-two, i.e. the result of <see cref="RoundUp"/>), or Zero if none available.
    /// </summary>
    public nint Rent(nuint bucketSize)
    {
        if (_buckets.TryGetValue(bucketSize, out var stack) && stack.TryPop(out nint ptr))
            return ptr;
        return nint.Zero;
    }

    /// <summary>
    /// Return a device pointer to the pool. <paramref name="bucketSize"/> must be the
    /// power-of-two size originally passed to <see cref="Rent"/> (or stored in _devPtrs).
    /// </summary>
    public void Return(nuint bucketSize, nint devPtr)
    {
        if (devPtr == nint.Zero || _disposed) { CuBlasInterop.CudaFree(devPtr); return; }
        _buckets.GetOrAdd(bucketSize, _ => new ConcurrentStack<nint>()).Push(devPtr);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var stack in _buckets.Values)
            while (stack.TryPop(out nint ptr))
                CuBlasInterop.CudaFree(ptr);
        _buckets.Clear();
    }
}
