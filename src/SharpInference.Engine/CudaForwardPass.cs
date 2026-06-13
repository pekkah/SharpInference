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
public sealed unsafe class CudaForwardPass : IForwardPass, IBatchedForwardPass
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
    //
    // Mutable (issue #190): the continuous-batching path swaps in a per-sequence
    // CudaSequenceKvCache's K/V arrays via BindCache around a Prefill call, then restores
    // the owned arrays. _ownedKCache/_ownedVCache are the real allocation home — the only
    // arrays Dispose frees — so a torn bind (mid-exception) never leaks or double-frees.
    private Tensor[] _gpuKCache;
    private Tensor[] _gpuVCache;
    private Tensor[] _ownedKCache = null!;
    private Tensor[] _ownedVCache = null!;

    // KV-cache element dtype (issue #179). F32 (default) or BFloat16 — the latter
    // halves the cache footprint to unlock long context on a 12 GB card. Arithmetic
    // stays fp32 in the kernels; only the *store* is narrowed, so decode is
    // argmax-stable vs fp32 KV. bf16 is rejected up front when composed with
    // TurboQuant or explicit SnapKV (see constructor); auto-SnapKV is disabled under
    // bf16. Set via SHARPI_KV_DTYPE=bf16. Mirrors CudaHybridGdnForwardPass (#27).
    private readonly DType _kvDType;

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
    /// Issue #162: window size (tokens) for chunked batched prefill of prompts longer
    /// than the non-flash 4096 cap. Each window is batched at its own startPos with flash
    /// attention streaming the prior KV, so the N-sized trunk scratch stays bounded to
    /// this many tokens regardless of prompt length. 4096 matches the well-tested
    /// single-shot batch size.
    /// </summary>
    private const int PrefillBatchChunk = 4096;

    /// <summary>
    /// Headroom (in positions) added to a Gemma-4 SWA layer's window when sizing its KV
    /// ring (issue #162). A batched prefill appends a whole batch of K/V before any of
    /// those queries attend, so the ring must hold the window PLUS one batched-append span
    /// or the earliest queries' window would be overwritten by the latest appends. The
    /// widest single append span is the larger of the chunked-prefill window
    /// (<see cref="PrefillBatchChunk"/>) and the 4096 non-flash batched-attention cap, so a
    /// ring of <c>window + SwaRingHeadroom</c> is always large enough. Capped at the model
    /// context (<see cref="SwaRingSize"/>) — a full-context cache needs no ring at all.
    /// </summary>
    private const int SwaRingHeadroom = PrefillBatchChunk > 4096 ? PrefillBatchChunk : 4096;

    /// <summary>
    /// Allocated KV-cache size, in positions, for a Gemma-4 sliding-window layer: the
    /// window plus <see cref="SwaRingHeadroom"/>, capped at the full context. When this
    /// equals the context the cache is full (the ring modulo in the kernels degenerates to
    /// the identity); when the context exceeds it the kernels wrap writes/reads modulo this
    /// size. The value passed as each SWA append/attention call's <c>maxSeqLen</c> argument
    /// MUST equal this so the kernel's <c>pos % maxSeqLen</c> lands in the right ring slot.
    /// </summary>
    private static int SwaRingSize(int maxSeqLen, int window) =>
        (int)Math.Min(maxSeqLen, (long)window + SwaRingHeadroom);

    // ── Per-token KV-dtype dispatch (issue #179) ───────────────────────────
    // Route the per-token decode/prefill KV kernels to their bf16-cache variants
    // when SHARPI_KV_DTYPE=bf16. Arithmetic is identical to the fp32 kernels; only
    // the cache load/store is narrowed. Used by every per-token Forward path
    // (gemma4 + dense, plain + profiled) and the per-token prefill fallback.
    private void KvAppendKv(Tensor k, Tensor v, Tensor kCache, Tensor vCache,
                            int kvDim, int position, int maxSeqLen)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.KvAppendBf16(k, v, kCache, vCache, kvDim, position, maxSeqLen);
        else if (_kvDType == DType.Q8_0)
            _gpu.KvAppendQ8_0(k, v, kCache, vCache, kvDim, position, maxSeqLen);
        else
            _gpu.KvAppend(k, v, kCache, vCache, kvDim, position, maxSeqLen);
    }

    private void AttentionKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                             Tensor? scoresScratch,
                             int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen,
                             float attnScale = -1f)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.AttentionBf16(q, kCache, vCache, output, scoresScratch,
                numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionQ8_0(q, kCache, vCache, output, scoresScratch,
                numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
        else
            _gpu.Attention(q, kCache, vCache, output, scoresScratch,
                numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
    }

    private void AttentionSwaKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output,
                                Tensor? scoresScratch,
                                int position, int windowSize, int headDim,
                                int numHeads, int numKvHeads, int maxSeqLen,
                                float attnScale = -1f)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.AttentionSwaBf16(q, kCache, vCache, output, scoresScratch,
                position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionSwaQ8_0(q, kCache, vCache, output, scoresScratch,
                position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
        else
            _gpu.AttentionSwa(q, kCache, vCache, output, scoresScratch,
                position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
    }

    // SHARPI_KV_DTYPE — issue #179. F32 (default on the dense path; bf16 is opt-in
    // until long-context validated) or BFloat16 (half-footprint KV). Anything else
    // is rejected so a typo doesn't silently fall back. Mirrors the GDN path's parser
    // (#27) but defaults to fp32 here rather than bf16.
    private static DType ParseKvDType(string? envValue) => envValue?.Trim().ToLowerInvariant() switch
    {
        null or ""    => DType.Float32,
        "fp32"        => DType.Float32,
        "bf16"        => DType.BFloat16,
        "q8_0" or "q8" => DType.Q8_0,
        var other     => throw new ArgumentException(
            $"SHARPI_KV_DTYPE must be 'fp32', 'bf16', or 'q8_0' (got '{other}').", nameof(envValue)),
    };

    /// <summary>
    /// Resolves the configured KV-cache dtype from the SHARPI_KV_DTYPE environment variable
    /// (fp32 default, bf16, q8_0), reusing the same parser the constructor uses. Exposed so
    /// the layer planner / loader can price the KV budget at the dtype the forward pass will
    /// actually allocate, instead of assuming fp32. Throws on an invalid value (same contract
    /// as the constructor) so a typo can't silently mis-budget.
    /// </summary>
    public static DType ResolveConfiguredKvDType() =>
        ParseKvDType(Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE"));

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
    private readonly bool _q4kSoa;           // #156: repack 2-D Q4_K weights to scale-unpacked SoA
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

    // Batched-decode logits scratch (issue #190): the output projection is the single
    // largest matmul, so BatchForwardMulti runs it once over all N decode rows (weight
    // read amortized N×) into this [N × vocab] device buffer, downloads it in one copy
    // into _decodeLogitsHost, then splits into the per-sequence float[] results. Lazily
    // (re)sized to the current decode batch N (small + changes only as the batch grows).
    private Tensor? _decodeLogitsAll;
    private float[]? _decodeLogitsHost;
    private int _decodeLogitsCapacity;

    // Batched-decode matmul path (issue #190/#194). Default: the weight-stationary matvec
    // (_gpu.MatMulBatchedWeightStationary) — token loop inside the block, so each weight HBM
    // read is amortized across the batch AND the per-(row,token) reduction chain is identical
    // to the GEMM-N matvec (bit-identical to the per-token decode oracle's kernels).
    // SHARPI_BATCH_DECODE_WS=0 falls back to the #190 GEMM-N matvec (_gpu.MatMulBatched, grid
    // Y = nTok: launch-overhead + L2 reuse only — the A/B baseline WS replaced).
    // SHARPI_BATCH_DECODE_GEMM=1 routes the compute-bound GEMM/MMQ path (GpuMatMulBatchedCore)
    // — argmax-stable, not bit-exact (same contract the prefill batched trunk holds).
    //
    // Measured on Qwen3-8B Q4_K_M @ 4070 Ti (aggregate t/s, single-user Forward = 75): GEMM-N
    // beats compute-bound at every realistic decode batch — N=1 70 vs 15, N=4 99 vs 57, N=8 106
    // vs 98 — because the compute-bound kernels carry large per-step fixed costs (int8 activation
    // conversion + the Q6_K output-weight fp16 dequant every step) that only amortize at the
    // N=4096 scale prefill runs at. Weight-stationary keeps GEMM-N's near-zero fixed costs and
    // adds the weight-read amortization GEMM-N lacks (#194).
    private readonly bool _batchDecodeComputeBound =
        Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_GEMM") == "1";
    private readonly bool _batchDecodeWeightStationary =
        Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_WS") != "0";
    // SHARPI_BATCH_DECODE_MMQ=1 (#201, opt-in): int8 tensor-core decode matmuls for the
    // big Q4_K-SoA shapes (BN=16 mma tile, weight read once per step) — argmax-stable,
    // not bit-exact; small/non-Q4_K shapes fall back to weight-stationary per tensor
    // inside the backend. The bit-exact WS matvecs are L1TEX-bound ~3× above the weight
    // floor at N=8 and their lane geometry is frozen by the bit-identity contract, so
    // this toggle is where the remaining batched-decode headroom lives.
    private readonly bool _batchDecodeMmq =
        Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_MMQ") == "1";

    // Ragged-batched per-sequence attention ops (issue #197). Default on: the per-layer
    // QK-norm/RoPE/KV-append/attention launches collapse from O(N) per-sequence calls
    // (~6·N low-occupancy single-token kernels per layer, the N attention blocks running
    // back-to-back) to O(1) ragged kernels whose grid covers all N sequences at their own
    // positions against their own caches (CudaBackend.*BatchedRagged). Bit-identical per
    // sequence to the per-sequence loop (same kernels' reduction chains, batched grid).
    // SHARPI_BATCH_DECODE_RAGGED=0 restores the #190 per-sequence loop.
    private readonly bool _batchDecodeRagged =
        Environment.GetEnvironmentVariable("SHARPI_BATCH_DECODE_RAGGED") != "0";

    // Per-batch-composition cache pointer table for the ragged kernels: [layer][seq]
    // K/V cache tensors, rebuilt only when the caches array composition changes
    // (sequence admitted/retired — compared by element identity). Avoids re-walking
    // N caches × L layers on every decode step.
    private CudaSequenceKvCache[]? _raggedSnapshot;
    private Tensor[][]? _raggedKLayers;
    private Tensor[][]? _raggedVLayers;

    // Issue #207: non-owning per-sequence-cache view over the OWNED K/V tensors, so the
    // single-user speculative-decode BatchVerify can drive BatchForwardMulti's packed trunk
    // against the owned cache. Never disposed (the owned tensors are freed by Dispose);
    // every layer is marked aliased so even an accidental Dispose can't free them.
    private CudaSequenceKvCache? _ownedCacheView;

    // Ragged attention spill scratch [N × numHeads × maxSeqLen] — the ragged kernel
    // spills per-(sequence, head) score rows when a sequence's length exceeds the
    // 4096-slot shared-memory fast path. Lazily allocated only when such a length
    // actually occurs (43 MB at N=8 / 40K ctx — don't pay it for short decodes).
    private Tensor? _raggedAttnScores;
    private int _raggedAttnScoresCapacity;

    // Empty (no-aliasing) layer set shared by every dense per-sequence cache CreateCache
    // hands out — dense models never share KV across layers (that's the Gemma 4 tail,
    // excluded from batching), so nothing is ever skipped on CudaSequenceKvCache.Dispose.
    private static readonly IReadOnlySet<int> s_noAliasedLayers = new HashSet<int>();

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
        // Issue #156/#160: repack 2-D Q4_K trunk weights into the scale-pre-unpacked
        // SoA layout so the decode matvec, prefill int8 MMQ, the N=2 MTP batched-verify
        // reader, and the GEMM-N/dequant fallback prefill readers all skip the
        // per-super-block 6-bit (scale,min) unpack switch — +7% decode / +5% prefill,
        // bit-identical (Qwen3-8B 70.0 → 74.7 t/s). Default on (SHARPI_Q4K_SOA=0
        // reverts). Dense-only: the MoE Q4_K readers are not SoA-converted, so the
        // repack is gated on !_isMoE at upload.
        _q4kSoa = Environment.GetEnvironmentVariable("SHARPI_Q4K_SOA") != "0";
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
            // Size the auto-context for the KV dtype the operator requested (#220): a bf16/q8_0
            // KV store fits 2×/4× the positions of fp32, but EstimateMaxContext previously priced
            // fp32 unconditionally, so --kv-type was silently ignored for auto-context. Pass the
            // *requested* dtype (pre-auto-narrow): an explicit narrowed choice should expand the
            // window, and the fp32 default still yields the fp32-fit context (the auto-narrow
            // below never fires for an fp32-sized auto-context, since fp32 fits there by
            // construction).
            _maxSeqLen = EstimateMaxContext(model, gpu, hp, ResolveConfiguredKvDType());
        // Invariant (#228): the resolved context never exceeds the model's own maximum, whether
        // it came from an explicit -c or a VRAM-fit estimator. Each branch above already clamps
        // to hp.ContextLength; this is the single chokepoint that guarantees it so no future
        // path (or estimator change) can over-shoot the model max.
        _maxSeqLen = Math.Min(_maxSeqLen, hp.ContextLength);
        // The KV-append/attention kernels index the cache at `pos % _maxSeqLen` (the ring
        // modulo, identity for full caches), so a zero context — e.g. a malformed GGUF with
        // context_length=0 reached via an explicit ctx-size — would be an in-kernel
        // divide-by-zero (GPU trap). Fail loud at construction instead.
        if (_maxSeqLen < 1)
            throw new ArgumentException(
                $"Resolved max context length is {_maxSeqLen}; the model's context_length " +
                "metadata is missing or zero.", nameof(maxContextLength));

        if (_tqEnabled)
        {
            _tqFp32Window = Math.Min(tqFp32Window, _maxSeqLen);
            _tqBlockBytes = TurboQuantOps.BlockSize(tqBits, _headDim);
            // The TQ attention kernel uses a stored-scores fast path up to 4096 positions
            // and a triple-pass recompute path above that. No per-context allocation cap.
        }

        // Bookkeeping-only: the actual KV lives in VRAM (_gpuKCache/_gpuVCache); this host
        // object tracks only the position counter (Length/TruncateTo/Reset). Allocating the
        // full host K/V buffers here is pure waste — numLayers × maxSeqLen × kvDim × 2 floats
        // is tens of GB at long context and OOMs the host before VRAM is the limit (#179).
        _kvCache = KvCache.CreateBookkeepingOnly(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.HeadDim);

        // SnapKV (issue #59) — gated by SHARPI_SNAPKV_BUDGET. Buffers are lazily
        // allocated on the first active prefill in Prefill(). Composition with
        // TurboQuant requires per-block ring bookkeeping that doesn't yet exist
        // (issue #60); explicit opt-in + TQ is rejected up front, and the auto
        // path stays disabled when TQ is on.
        // KV-cache dtype (issue #179). bf16 narrows the store to half the footprint,
        // q8_0 to ~a quarter; both compose with neither TurboQuant (its own quantized
        // ring) nor SnapKV physical compaction (no narrowed compact kernel wired on this
        // path yet), so both are rejected up front and auto-SnapKV is disabled under a
        // narrowed dtype below.
        string? kvDTypeEnv = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        bool kvDTypeExplicit = !string.IsNullOrWhiteSpace(kvDTypeEnv);
        _kvDType = ParseKvDType(kvDTypeEnv);
        _snapKvCfg = SnapKvConfig.FromEnvironment();

        // Auto-narrow the KV dtype (issue #185 item 1). When the resolved context's fp32
        // KV cache won't fit the VRAM budget, the construction-time per-layer K/V Allocate
        // below would fail with "cudaMalloc failed: 2" instead of degrading gracefully —
        // capping context at fp32 even though a narrowed store would fit. Mirror the
        // auto-SnapKV VRAM heuristic: pick a narrowed dtype (bf16 preferred; q8_0 if bf16
        // still won't fit and the geometry supports it) and log the choice, so an oversized
        // -c reaches long context instead of erroring.
        //
        // Precedence vs auto-SnapKV: auto-narrow runs FIRST and wins. SnapKV does NOT
        // shrink this construction-time allocation (the cache is allocated full-_maxSeqLen
        // up front; eviction only bounds the *logical* length at runtime), so it cannot
        // prevent the cudaMalloc failure — only narrowing the element width can. Narrowing
        // is also exact-context, argmax-stable, and full-speed-prefill, whereas SnapKV
        // evicts tokens (lossy). We therefore auto-narrow only when the operator set
        // NEITHER an explicit --kv-type NOR an explicit *positive* SnapKV budget; either
        // explicit choice is respected and never overridden (an explicit fp32 still errors
        // loudly at allocation rather than silently narrowing; an explicit positive SnapKV
        // budget is honoured even though it can't avert the alloc failure — the operator's
        // call). Note SHARPI_SNAPKV_BUDGET=0 means "disable SnapKV" (IsBudgetExplicit=true,
        // Budget=0), the same disable knob the banners advertise — it must NOT suppress
        // auto-narrow, so we gate on Budget > 0 to match the disable semantics used by the
        // SnapKV throws/auto-enable below. When auto-narrow fires it sets a narrowed
        // _kvDType, which flips kvNarrowed below and disables auto-SnapKV.
        if (!_tqEnabled && !kvDTypeExplicit && _kvDType == DType.Float32
            && !(_snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0))
        {
            long availKvBytes = EstimateAvailableKvVram(model, gpu, hp);
            long fp32KvBytes  = EstimateKvCacheBytes(hp, _maxSeqLen, DType.Float32);
            long bf16KvBytes  = EstimateKvCacheBytes(hp, _maxSeqLen, DType.BFloat16);
            _kvDType = ResolveKvDType(
                _kvDType, kvDTypeExplicit, _tqEnabled,
                availKvBytes, fp32KvBytes, bf16KvBytes, Q8KvGeometrySupported(hp),
                out bool autoNarrowed);
            if (autoNarrowed)
            {
                // bf16 footprint is already in hand; only q8_0 (or a best-effort bf16 that
                // still won't fit) needs a recompute for the banner.
                long chosenKvBytes = _kvDType == DType.BFloat16
                    ? bf16KvBytes : EstimateKvCacheBytes(hp, _maxSeqLen, _kvDType);
                // When even the narrowest store we could pick still won't fit, the per-layer
                // Allocate below will cudaMalloc-fail — say so rather than imply success.
                string fitNote = chosenKvBytes > availKvBytes
                    ? " — still over budget, allocation may fail (context too large for this GPU)"
                    : "";
                Console.Error.WriteLine(
                    $"[CudaForwardPass] KV auto-narrowed to {_kvDType} for context {_maxSeqLen}: fp32 KV " +
                    $"~{fp32KvBytes / (1024.0 * 1024.0):F0} MiB exceeds the ~{availKvBytes / (1024.0 * 1024.0):F0} MiB " +
                    $"VRAM KV budget ({_kvDType} ~{chosenKvBytes / (1024.0 * 1024.0):F0} MiB{fitNote}). " +
                    "Set --kv-type fp32 to force fp32 (errors if it won't fit).");
            }
        }

        // bf16 and q8_0 both narrow the KV store; the gating below (TQ, SnapKV,
        // auto-SnapKV) applies identically to either narrowed dtype (issue #179).
        bool kvNarrowed = _kvDType != DType.Float32;
        if (kvNarrowed && _tqEnabled)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE={_kvDType} + TurboQuant is not supported (TQ owns the KV " +
                "quantization). Use one or the other (issue #179).");

        if (_tqEnabled && _snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0)
            throw new NotSupportedException(
                "SnapKV + TurboQuant composition is not yet implemented (issue #60). " +
                "Set SHARPI_SNAPKV_BUDGET=0 to disable or disable --tq.");
        if (kvNarrowed && _snapKvCfg.IsBudgetExplicit && _snapKvCfg.Budget > 0)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE={_kvDType} + SnapKV is not yet implemented (narrowed-KV " +
                "physical compaction kernel not wired on the dense path; issue #179). " +
                "Set SHARPI_SNAPKV_BUDGET=0 to disable SnapKV.");
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
        else if (kvNarrowed)
        {
            // bf16/q8_0 already shrink the KV footprint — the memory win SnapKV
            // auto-enable chases — and the narrowed-KV physical-compaction kernel
            // isn't wired here yet (#179). Don't auto-enable eviction on top.
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

        string kvTag = _kvDType switch { DType.BFloat16 => " [KV bf16]", DType.Q8_0 => " [KV q8_0]", _ => "" };
        Console.Error.WriteLine($"[CudaForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength}){(_tqEnabled ? " [TQ3]" : "")}{kvTag}");

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
        // The owned arrays are filled in-place by the branches below; capture the array
        // references now so BindCache/RestoreOwned (issue #190) and Dispose always target
        // the real allocation home regardless of any transient rebind.
        _ownedKCache = _gpuKCache;
        _ownedVCache = _gpuVCache;
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
                // Gemma 4 12B mixes per-layer KV head counts (8 GQA on SWA, 1 MQA on
                // global). Size each layer's K/V by its own count so global layers
                // don't over-allocate (and match the attention dispatch below).
                int layerKvHeads = hp.LayerKvHeads is { } lkv ? lkv[i] : _numKvHeads;
                int layerKvDim = layerKvHeads * layerHd;
                // SWA layers use a window-sized ring (window + headroom for one batched
                // append span, issue #162); everything else is full-context. The same
                // SwaRingSize value is passed to the kernels as maxSeqLen so their
                // pos % maxSeqLen wraps into this exact ring.
                int layerCtx = (perLayerKv && hp.IsSwaLayer is { } swa && swa[i])
                    ? SwaRingSize(_maxSeqLen, swaWindow)
                    : _maxSeqLen;
                // q8_0 KV packs 32 elements per block; the store kernels' per-warp amax
                // reduction (and DTypeInfo.ByteSize's count/32 sizing) assume each layer's
                // kvDim is a multiple of 32 so blocks never straddle a KV row. Every dense
                // GGUF head_dim (64/128/256) satisfies this, but fail loud rather than
                // silently under-allocate + corrupt if a future geometry doesn't (#179).
                if (_kvDType == DType.Q8_0 && (layerKvDim & 31) != 0)
                    throw new NotSupportedException(
                        $"SHARPI_KV_DTYPE=q8_0 requires every layer's kvDim to be a multiple of 32 " +
                        $"(block_q8_0 = 32 elements/block); layer {i} has kvDim={layerKvDim}. " +
                        "Use --kv-type bf16 or fp32 for this model.");
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim), _kvDType);
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim), _kvDType);
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
                // Gemma 4 12B global layers omit attn_v (attention_k_eq_v): V reuses
                // the raw K projection. Skip the missing tensor; the forward path
                // copies K→V for these layers.
                if (_model.FindTensor($"blk.{i}.attn_v.weight") is not null)
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
        if (embInfo.DType == DType.Q4_K || embInfo.DType == DType.Q8_0 || embInfo.DType == DType.Q6_K)
        {
            // Q6_K is the Gemma 4 12B (QAT) tied token_embd: 3840×262144 dequanted to
            // F32 would burn ~4 GB of VRAM (and OOM full offload on a 12 GB card). Keep
            // it packed (~787 MiB) and gather/dequant one row per token via the dedicated
            // EmbedLookup* kernel; the tied output projection reuses the packed table
            // through the Q6_K matvec (issue #124).
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
        // Must cover every quant the graph can store or the wrong dequant silently
        // produces garbage (#124) — mirror EmbedTokenGemma4/EmbedTokenGpu exactly.
        if (_embIsQuantized)
        {
            switch (_weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K))
            {
                case DType.Q8_0: _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q6_K: _gpu.EmbedLookupQ6K(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q5_K: _gpu.EmbedLookupQ5K(_gpuEmbedding, _hidden, token, _embDim); break;
                default:         _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim); break;
            }
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

            // Order matters: RoPE does NOT commute with per-channel-weighted QK-norm
            // (NEOX RoPE mixes channels i and i+d/2, which carry different learned
            // weights), so we mirror the CPU ForwardPass / HF Qwen3 / llama.cpp
            // build_qwen3 ordering exactly (issue #157):
            //   • weighted QK-norm (Qwen3, OLMoE, …): norm BEFORE RoPE
            //   • L2 QK-norm (Llama-4):               norm AFTER  RoPE (RoPE layers only)
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }

            if (useRoPE)
            {
                _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
            }

            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                _gpu.HeadNormPure(_q, _numHeads, _headDim, _hp.RmsNormEps);
                _gpu.HeadNormPure(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
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
                KvAppendKv(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, kvSlot, _maxSeqLen);
                AttentionKv(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
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
    // Gemma 4 token embedding gather into _hidden, dispatched on the packed
    // embedding dtype. Q6_K is the 12B QAT tied table (issue #124); Q8_0 the E4B
    // case; Q4_K/Q5_K the K-quant conversions; F32 the dequant-on-upload fallback.
    private void EmbedTokenGemma4(int token)
    {
        if (!_embIsQuantized)
        {
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, token, _embDim);
            return;
        }
        switch (_weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K))
        {
            case DType.Q8_0: _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim); break;
            case DType.Q6_K: _gpu.EmbedLookupQ6K(_gpuEmbedding, _hidden, token, _embDim); break;
            case DType.Q5_K: _gpu.EmbedLookupQ5K(_gpuEmbedding, _hidden, token, _embDim); break;
            default:         _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim); break;
        }
    }

    private ReadOnlySpan<float> ForwardGemma4(int token, int position)
    {
        if (s_regionProfile) return ForwardGemma4RegionProfiled(token, position);

        // 1. Embedding lookup
        EmbedTokenGemma4(token);

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
        EmbedTokenGemma4(token);
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
            // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on global).
            int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
            int qDimL = _numHeads * layerHd;
            int kvDimL = layerKv * layerHd;
            int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];
            // Gemma 4 12B global layers carry no attn_v: V reuses the raw K projection
            // (attention_k_eq_v). These layers always own their KV (shared_kv_layers=0).
            bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer] is null;

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
                if (kEqV)
                    CopyDevice(vView, kView);   // V = raw K projection (pre-norm, pre-RoPE)
                else
                    GpuMatMul(vView, _wv[layer]!, _normBuf);
            }

            // Per-head Q/K norm (Gemma 4: shared headDim-sized weight per head).
            // CPU applies norm BEFORE RoPE (UseL2QkNorm == false).
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                if (!kvShared)
                    _gpu.HeadNormQk(qView, _wqNorm![layer], kView, _wkNorm![layer],
                        _numHeads, layerKv, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                else
                    _gpu.HeadNorm(qView, _wqNorm![layer], _numHeads, layerHd,
                        _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }

            // Gemma 4: V gets a plain per-head RMSNorm (no learned weight) before the
            // KV cache, on EVERY KV-owning layer (E4B and 12B alike). Mirrors the CPU
            // ForwardPass + llama.cpp gemma4.cpp:227 (Vcur = ggml_rms_norm(Vcur),
            // unconditional for has_kv layers). V is never RoPE'd. This loop is gemma4-only
            // (layerHd from LayerHeadDim![layer]) so !kvShared == has_kv. For 12B k_eq_v
            // globals V is the raw K projection (copied above); for E4B / 12B SWA it is
            // wv·norm — both get the same V-norm here.
            if (!kvShared)
                _gpu.HeadNormPure(vView, layerKv, layerHd, _hp.RmsNormEps);

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
                    ? SwaRingSize(_maxSeqLen, _hp.SlidingWindowSize)
                    : _maxSeqLen;
                KvAppendKv(kView, vView, _gpuKCache[layer], _gpuVCache[layer],
                    kvDimL, position, layerCtx);
            }

            int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer]
                              && _hp.SlidingWindowSize > 0)
                ? SwaRingSize(_maxSeqLen, _hp.SlidingWindowSize)
                : _maxSeqLen;

            // Gemma 4 uses attention_scale = 1.0 (no 1/sqrt(head_dim) prefactor). Pass
            // it explicitly so the kernel skips its rsqrtf(head_dim) — matching the CPU
            // path's `_layerHeadDim is not null ? 1f : 1/sqrt(hd)` exactly with no
            // prescale round-trip, and dropping a ScaleInPlace launch per layer.
            if (isSwa)
            {
                AttentionSwaKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch,
                    position, _hp.SlidingWindowSize, layerHd,
                    _numHeads, layerKv, effLayerCtx, attnScale: 1f);
            }
            else
            {
                AttentionKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch,
                    _numHeads, layerKv, layerHd, position + 1, effLayerCtx, attnScale: 1f);
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

        // Dispatch on stored embedding dtype — must cover every quant the graph can
        // store (#124); mirror EmbedTokenGemma4/EmbedTokenGpu exactly.
        if (_embIsQuantized)
        {
            switch (_weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K))
            {
                case DType.Q8_0: _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q6_K: _gpu.EmbedLookupQ6K(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q5_K: _gpu.EmbedLookupQ5K(_gpuEmbedding, _hidden, token, _embDim); break;
                default:         _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim); break;
            }
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
            // Same ordering contract as RunDeviceRegion (issue #157): weighted QK-norm
            // before RoPE, L2 QK-norm after RoPE — RoPE does not commute with weighted norm.
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.HeadNorm(_q, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.HeadNorm(_k, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }
            if (useRoPE)
            {
                _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
            }
            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                _gpu.HeadNormPure(_q, _numHeads, _headDim, _hp.RmsNormEps);
                _gpu.HeadNormPure(_k, _numKvHeads, _headDim, _hp.RmsNormEps);
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
            KvAppendKv(_k, _v, _gpuKCache[layer], _gpuVCache[layer], kvDim, kvSlot, _maxSeqLen);
            AttentionKv(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
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
        EmbedTokenGemma4(token);
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
            int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
            int qDimL = _numHeads * layerHd;
            int kvDimL = layerKv * layerHd;
            int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
            bool kvShared = kvSrc >= 0;
            int effLayer = kvShared ? kvSrc : layer;
            bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];
            bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer] is null;

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
                if (kEqV) CopyDevice(vView, kView);
                else      GpuMatMul(vView, _wv[layer]!, _normBuf);
            }
            _gpu.Synchronize();
            AccPhase(PH_QKV, sw, ref t0);

            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                if (!kvShared)
                    _gpu.HeadNormQk(qView, _wqNorm![layer], kView, _wkNorm![layer],
                        _numHeads, layerKv, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                else
                    _gpu.HeadNorm(qView, _wqNorm![layer], _numHeads, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            }
            if (!kvShared)   // Gemma 4 V-norm on every KV-owning layer (see ForwardGemma4)
                _gpu.HeadNormPure(vView, layerKv, layerHd, _hp.RmsNormEps);
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
                    ? SwaRingSize(_maxSeqLen, _hp.SlidingWindowSize)
                    : _maxSeqLen;
                KvAppendKv(kView, vView, _gpuKCache[layer], _gpuVCache[layer], kvDimL, position, layerCtx);
            }
            int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer]
                              && _hp.SlidingWindowSize > 0)
                ? SwaRingSize(_maxSeqLen, _hp.SlidingWindowSize)
                : _maxSeqLen;
            if (isSwa)
                AttentionSwaKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch, position, _hp.SlidingWindowSize, layerHd,
                    _numHeads, layerKv, effLayerCtx, attnScale: 1f);
            else
                AttentionKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                    _attnScoresScratch, _numHeads, layerKv, layerHd, position + 1, effLayerCtx, attnScale: 1f);
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
        // (MoE, SnapKV-active, TQ, non-NEOX RoPE, L2 QK-norm, attn bias, unbatchable
        // weight dtype) falls back to the per-token loop below.
        //
        // Issue #162: the 4096 fast-path cap is a limit of the *non-flash* shared-scores
        // AttentionBatched kernel (it throws above startPos+nTok=4096). The flash prefill
        // kernels stream KV, so when flash is enabled we run the batched path for prompts
        // of any length, chunking into PrefillBatchChunk-token windows so the N-sized trunk
        // scratch stays bounded. Each chunk is batched at its own startPos; flash attends
        // to all prior KV. Without this, a >4096-token prompt drops to the per-token loop
        // (memory-bound, ~8× slower: 432 → 50 t/s on Qwen3-8B Q4_K @ 4070 Ti).
        if (BatchedPrefillEnabled && !snapKvActive && N >= 2
            && startPos + N <= _maxSeqLen && IsBatchedPrefillSupported())
        {
            // Chunking past 4096 requires a streaming attention path (flash) for the
            // global (full-causal) layers — the non-flash AttentionBatched caps at 4096.
            // SWA layers are now correct across chunk boundaries (issue #162): their KV
            // ring is sized window + SwaRingHeadroom (≥ one chunk span), so appending a
            // whole chunk before attending never overwrites a still-needed window, and the
            // flash/append kernels wrap reads/writes modulo the ring (SwaRingSize).
            //
            // Fail-closed guard (kept from the pre-#162 gate): the per-layer SWA dispatch
            // keys off `IsSwaLayer`, which today only Gemma 4 populates. A future arch that
            // sets a model-wide `SlidingWindowSize` WITHOUT a per-layer `IsSwaLayer` pattern
            // would run every layer as full-causal (window silently ignored) — harmless
            // within the proven 4096 cap, but extending that past 4096 would silently drop
            // the window. So only chunk past 4096 when either there's a real per-layer SWA
            // pattern or the model has no window at all.
            // Chunking past the 4096 scalar cap needs a streaming (flash) attention path on
            // every layer. fp32 uses the half2/TC flash kernels; bf16/q8_0 only have the Tc2
            // thunk so far, so they can chunk only when Tc2 covers all layers (head_dim%64).
            bool flashCoversAll = _kvDType is DType.BFloat16 or DType.Q8_0
                ? NarrowedFlashTc2CoversAllLayers()
                : PrefillFlashAttnEnabled;
            bool canChunkPast4096 = flashCoversAll
                && (_hp.IsSwaLayer is not null || _hp.SlidingWindowSize <= 0);
            int cap = canChunkPast4096 ? _maxSeqLen : 4096;
            if (startPos + N <= cap)
            {
                if (N <= PrefillBatchChunk || !canChunkPast4096)
                    return PrefillBatchedTrunk(tokens, startPos);

                // Chunked: flash streams KV across windows, so process PrefillBatchChunk
                // tokens at a time. Only the last chunk's logits are returned (decode
                // starts from the final token); the per-chunk final norm + output proj on
                // earlier chunks is discarded, a negligible cost vs the batched-trunk win.
                int[] all = tokens as int[] ?? System.Linq.Enumerable.ToArray(tokens);
                ReadOnlySpan<float> chunkLogits = default;
                for (int off = 0; off < N; off += PrefillBatchChunk)
                {
                    int len = Math.Min(PrefillBatchChunk, N - off);
                    chunkLogits = PrefillBatchedTrunk(new ArraySegment<int>(all, off, len), startPos + off);
                }
                return chunkLogits;
            }
        }

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
        d is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Q4_0 or DType.Float32;

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
    // Narrowed KV (#179): true when every layer would take the Tc2 flash kernel for the
    // active narrowed dtype (bf16/q8_0), so the batched trunk streams K/V and can chunk a
    // prompt past the 4096 scalar cap. Requires TC flash on (not forced to single-warp)
    // and head_dim % 64 == 0 on every layer.
    private bool NarrowedFlashTc2CoversAllLayers()
    {
        if (_kvDType is not (DType.BFloat16 or DType.Q8_0) || !PrefillFlashTcEnabled || _forceFlashTc1) return false;
        if (_hp.LayerHeadDim is { } lhd)
        {
            foreach (int hd in lhd)
                if ((hd & 63) != 0) return false;
            return true;
        }
        return (_headDim & 63) == 0;
    }

    private bool IsBatchedPrefillSupported()
    {
        // bf16 KV (issue #179): the scalar batched kernels (KvAppendBatchedBf16,
        // AttentionBatchedBf16, AttentionSwaBatchedBf16) read the bf16 cache, so batched
        // prefill is supported up to the shared-scores cap. The flash/TC kernels are still
        // fp32-only, so the prefill gate forces canChunkPast4096=false under bf16 —
        // prompts past 4096 fall back to the per-token loop (also bf16-aware) until the
        // bf16 flash port (1.5b) lands.
        if (_isMoE || _tqEnabled || _hasAttnBias || !_hp.IsNeoxRope) return false;
        if (_hasQkNorm && _hp.UseL2QkNorm) return false;

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            bool kvShared = _hp.KvSourceLayer is { } ksl && ksl[i] >= 0;
            if (!BatchableWeight(_wq[i]) || !BatchableWeight(_wo[i]) ||
                !BatchableWeight(_wGate[i]) || !BatchableWeight(_wUp[i]) ||
                !BatchableWeight(_wDown[i]))
                return false;
            if (!kvShared)
            {
                if (!BatchableWeight(_wk[i])) return false;
                // Gemma 4 12B global layers (attention_k_eq_v) carry no attn_v: V reuses
                // the raw K projection, handled in GpuLayerBatchedTrunk. Only require a
                // batchable _wv when the layer actually has a separate V projection; a
                // null _wv on a non-k_eq_v model is unexpected → fall back to per-token.
                if (_wv[i] is not null) { if (!BatchableWeight(_wv[i])) return false; }
                else if (!_hp.AttentionKEqV) return false;
            }
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
        else if (dt == DType.Q4_0)
        {
            // Issue #124/#173: Gemma 4 12B QAT keeps all bulk matmul weights in Q4_0.
            // Route the trunk matmuls through a compute-bound path (weight read once per
            // batch) instead of the memory-bound per-token GEMM-N matvec (re-streams the
            // whole weight once per token — the dominant large-N prefill cost).
            // PrefillMmqEnabled: the int8 MMQ reads the weight once as int8 (nibble-
            // expanded, no fp16 dequant temp to HBM, int8 tensor cores), with the Q4_0
            // symmetric −8·d_w·Σq centering term. Fallback: dequant→fp16→cuBLAS GEMM.
            // Argmax-stable, not byte-exact. Q4_0 blocks are 32-wide; every Gemma 4 12B
            // hidden dim is a multiple of 32. There is no Q4_0 GEMM-N matvec fallback
            // (CudaBackend.MatMulBatched has no Q4_0 case), so an unaligned width is a
            // hard error rather than a silent wrong-kernel path.
            int cols = (int)(inAll.ElementCount / n);
            if ((cols & 0x1f) == 0)
            {
                if (PrefillMmqEnabled)
                    _gpu.MatMulBatchedMmq(outAll, weights, inAll, n, dt);
                else
                    _gpu.MatMulBatchedGemm(outAll, weights, inAll, n, dt);
            }
            else
                throw new InvalidOperationException(
                    $"Q4_0 batched prefill matmul requires cols % 32 == 0 (got {cols}).");
        }
        else if (dt is DType.Q6_K or DType.Q5_K)
        {
            // Issue #162: mixed _M quants keep trunk tensors in Q6_K (ffn_down + attn_v
            // in Q4_K_M) or Q5_K (q/k/o/gate/up in Q5_K_M). Neither has an int8 MMQ
            // kernel, so route those trunk matmuls through the dequant→fp16→cuBLAS GEMM
            // (weight read once per batch) instead of the memory-bound per-token GEMM-N
            // matvec — the latter re-streams the whole weight once per token and was the
            // dominant large-N prefill cost. Both are 256-wide super-blocks; fall back
            // defensively if cols isn't aligned.
            int cols = (int)(inAll.ElementCount / n);
            if ((cols & 0xff) == 0)
                _gpu.MatMulBatchedGemm(outAll, weights, inAll, n, dt);
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
            // Dispatch on the packed embedding dtype — MUST cover every quant the model
            // graph can store, or the wrong dequant kernel silently produces garbage. The
            // Gemma 4 12B QAT tied table is Q6_K (#124); a bare Q8_0/else split dequanted
            // it as Q4_K → NaN embeddings → all-NaN batched-prefill logits. Mirrors the
            // per-token EmbedTokenGemma4 dispatch exactly.
            switch (_weightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K))
            {
                case DType.Q8_0: _gpu.EmbedLookupQ8_0(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q6_K: _gpu.EmbedLookupQ6K(_gpuEmbedding, _hidden, token, _embDim); break;
                case DType.Q5_K: _gpu.EmbedLookupQ5K(_gpuEmbedding, _hidden, token, _embDim); break;
                default:         _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, token, _embDim); break;
            }
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
        // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on global).
        int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[layer] : _numKvHeads;
        int qDimL = _numHeads * layerHd, kvDimL = layerKv * layerHd;
        int kvSrc = _hp.KvSourceLayer is { } ksl ? ksl[layer] : -1;
        bool kvShared = kvSrc >= 0;
        int effLayer = kvShared ? kvSrc : layer;
        bool isSwa = _hp.IsSwaLayer is { } swa && swa[layer];
        int window = _hp.SlidingWindowSize;
        // Gemma 4 12B global layers carry no attn_v: V reuses the raw K projection
        // (attention_k_eq_v). These layers always own their KV (shared_kv_layers=0).
        bool kEqV = _hp.AttentionKEqV && !isSwa && _wv[layer] is null;

        // Per-layer dense views (the buffers are sized for max head_dim × max KV heads).
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
            if (kEqV)
                _gpu.CopyDevice(vAll, kAll);   // V = raw K projection (pre-norm, pre-RoPE)
            else
                GpuMatMulBatched(vAll, _wv[layer]!, _bpNorm!, N);
        }

        // RoPE and per-head QK-norm must run in the SAME order as the matching per-token
        // oracle, because RoPE does not commute with per-channel-weighted RMSNorm. All
        // weighted-QK-norm dense models (Gemma, Qwen3, …) apply QK-norm BEFORE RoPE — the
        // HF / llama.cpp ordering also followed by the per-token RunDeviceRegion and the CPU
        // ForwardPass (issue #157). NoRopeLayerStep skips RoPE on the same layers as the
        // per-token path; QK-norm (weighted) always runs. L2 QK-norm is gated out upstream
        // (IsBatchedPrefillSupported returns false), so only the weighted norm→rope case lands here.
        bool useRoPE = _hp.NoRopeLayerStep == 0 || (layer + 1) % _hp.NoRopeLayerStep != 0;
        float ropeTheta = isSwa ? _ropeThetaSwa : _hp.RopeTheta;

        void ApplyQkNormBatched()
        {
            if (!_hasQkNorm || _hp.UseL2QkNorm) return;
            if (!kvShared)
                _gpu.HeadNormQkBatched(qAll, _wqNorm![layer], kAll, _wkNorm![layer],
                    _numHeads, layerKv, layerHd, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
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
                    _gpu.RoPEWithFactorsBatched(kAll, startPos, layerHd, ropeTheta, rfTbl, layerKv, N);
            }
            else
            {
                _gpu.RoPEPartialBatched(qAll, startPos, layerHd, layerHd, ropeTheta, _numHeads, N, neox: true);
                if (!kvShared)
                    _gpu.RoPEPartialBatched(kAll, startPos, layerHd, layerHd, ropeTheta, layerKv, N, neox: true);
            }
        }

        // Weighted QK-norm before RoPE for every dense family (issue #157).
        ApplyQkNormBatched();
        // Gemma 4: V gets a plain per-head RmsNorm (no learned weight) before the KV cache
        // on every KV-owning layer (E4B + 12B) — mirrors the per-token ForwardGemma4 +
        // llama.cpp gemma4.cpp:227. For 12B k_eq_v globals V was copied from the RAW K
        // projection above (pre QK-norm); for E4B / 12B SWA it is wv·norm. V is never RoPE'd.
        // This trunk also serves non-gemma4 models (layerHd falls back to _headDim), so the
        // V-norm is gated on the gemma4 master switch — NOT AttentionKEqV (E4B lacks it but
        // still V-norms, matching the CPU reference and avoiding a mixed-norm V cache vs the
        // per-token decode path that previously broke the E4B coherence oracle).
        if (_isGemma4Like && !kvShared)
            _gpu.HeadNormPureBatched(vAll, layerKv, layerHd, N, _hp.RmsNormEps);
        ApplyRopeBatched();

        if (!kvShared)
        {
            int layerCtx = isSwa && window > 0 ? SwaRingSize(_maxSeqLen, window) : _maxSeqLen;
            if (_kvDType == DType.BFloat16)
                _gpu.KvAppendBatchedBf16(kAll, vAll, _gpuKCache[layer], _gpuVCache[layer], kvDimL, startPos, layerCtx, N);
            else if (_kvDType == DType.Q8_0)
                _gpu.KvAppendBatchedQ8_0(kAll, vAll, _gpuKCache[layer], _gpuVCache[layer], kvDimL, startPos, layerCtx, N);
            else
                _gpu.KvAppendBatched(kAll, vAll, _gpuKCache[layer], _gpuVCache[layer], kvDimL, startPos, layerCtx, N);
        }

        int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer] && window > 0)
            ? SwaRingSize(_maxSeqLen, window) : _maxSeqLen;

        if (s_prefillProfile) { _gpu.Synchronize(); _profSw.Restart(); }
        // Gemma 4: attention_scale = 1.0, passed explicitly (kernel skips its rsqrtf).
        // Other models pass _attnScale = -1 so the kernel derives 1/sqrt(head_dim).
        if (_kvDType is DType.BFloat16 or DType.Q8_0)
        {
            // Narrowed KV (bf16/q8_0, #179). The tensor-core flash kernel (Tc2) has a
            // templated thunk per dtype and streams K/V, so head_dim%64 layers use it for
            // any length (incl. chunked prefill past 4096). Other head_dims fall to the
            // scalar batched narrowed kernels, which the gate keeps ≤4096 (canChunkPast4096
            // requires Tc2 covers all layers). The single-warp Tc and half2 flash kernels
            // have no narrowed thunk yet — a trivial follow-up only a non-%64 head_dim model
            // past 4096 would need.
            if (PrefillFlashTcEnabled && !_forceFlashTc1 && (layerHd & 63) == 0)
                _gpu.FlashAttentionPrefillTc2(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N,
                    attnScale: _attnScale, kvCacheType: _kvDType);
            else if (isSwa && _kvDType == DType.BFloat16)
                _gpu.AttentionSwaBatchedBf16(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, window, effLayerCtx, N, attnScale: _attnScale);
            else if (isSwa)
                _gpu.AttentionSwaBatchedQ8_0(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, window, effLayerCtx, N, attnScale: _attnScale);
            else if (_kvDType == DType.BFloat16)
                _gpu.AttentionBatchedBf16(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, effLayerCtx, N, attnScale: _attnScale);
            else
                _gpu.AttentionBatchedQ8_0(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, effLayerCtx, N, attnScale: _attnScale);
        }
        else if (PrefillFlashTcEnabled && (layerHd & 15) == 0)
        {
            // #147 multi-warp/d-split when head_dim is a multiple of 64 (W·16); else the
            // #146 single-warp kernel. SHARPI_PREFILL_FLASH_TC1=1 forces single-warp (A/B).
            if (!_forceFlashTc1 && (layerHd & 63) == 0)
                _gpu.FlashAttentionPrefillTc2(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
            else
                _gpu.FlashAttentionPrefillTc(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                    _numHeads, layerKv, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
        }
        else if (PrefillFlashAttnEnabled)
            _gpu.FlashAttentionPrefill(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, layerKv, layerHd, startPos, isSwa ? window : 0, effLayerCtx, N, attnScale: _attnScale);
        else if (isSwa)
            _gpu.AttentionSwaBatched(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, layerKv, layerHd, startPos, window, effLayerCtx, N, attnScale: _attnScale);
        else
            _gpu.AttentionBatched(qAll, _gpuKCache[effLayer], _gpuVCache[effLayer], attnAll,
                _numHeads, layerKv, layerHd, startPos, effLayerCtx, N, attnScale: _attnScale);
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

    // ── IBatchedForwardPass (issue #190): CUDA continuous batching, dense path ──────────
    //
    // The continuous-batching engine drives this forward pass through the backend-agnostic
    // IBatchedForwardPass surface: it allocates one per-sequence GPU KV cache per admitted
    // request (CreateCache), prefills each (PrefillWithCache / PrefillPackedMulti), then
    // decodes all active sequences together in one weight-amortized batched step
    // (BatchForwardMulti). All calls land on the engine's single batcher thread, so the
    // shared scratch + the BindCache rebind below never race.
    //
    // Scope is DENSE only. MoE (router Download mid-layer), Gemma-4-like (per-layer
    // head_dim / SWA rings / shared-KV aliasing), TurboQuant (compressed ring), and an
    // active SnapKV budget (eviction only runs on the owned cache during a whole-prompt
    // prefill) all throw NotSupportedException here, and the loader keeps
    // batchingSupported=false for them so those models fall back to the single-user engine.

    /// <summary>Whether SnapKV prefill eviction is configured/active (issue #59). The engine
    /// disables chunked/packed prefill when true; the loader keeps batching off when true.</summary>
    public bool SnapKvEnabled => _snapKvEffectiveBudget > 0;

    /// <summary>Bytes of KV one token occupies across all layers (issue #183) — the engine
    /// turns a memory budget into a token admission budget. Honors the KV dtype (#179) via
    /// block-aware <see cref="DTypeInfo.ByteSize"/> — <see cref="DTypeInfo.BytesPerElement"/>
    /// throws for quantized stores (q8_0 packs 32 elements into a 34-byte block).</summary>
    public long KvBytesPerToken =>
        (long)_hp.NumLayers * 2 * DTypeInfo.ByteSize((long)_numKvHeads * _headDim, _kvDType);

    /// <summary>The dequant-once CPU weight cache (issue #189) is a CPU-path optimization;
    /// the CUDA path keeps all weights resident in VRAM, so it never applies here.</summary>
    public bool PrefillDequantCacheActive => false;

    /// <summary>Dtypes the batched-decode GEMM-N matvec (<see cref="CudaBackend.MatMulBatched"/>)
    /// supports. Excludes Q4_0 — it has compute-bound GEMM/MMQ prefill kernels but NO GEMM-N
    /// matvec, so a Q4_0 trunk/output weight would throw at the first batched decode step. Dense
    /// Q4_0 is otherwise rare (the common Q4_0 model, Gemma 4 12B QAT, is already excluded as
    /// per-layer head_dim).</summary>
    private static bool GemmNBatchable(DType d) =>
        d is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Q8_0 or DType.Float32;

    /// <summary>Fail-closed dtype check for a batched-decode weight: an unregistered handle is
    /// treated as NOT batchable, so a weight whose dtype isn't tracked can't slip into the
    /// GEMM-N path and be misdispatched (mirrors <see cref="BatchableWeight"/>).</summary>
    private bool DecodeBatchable(Tensor t) =>
        _weightDTypes.TryGetValue(t.Handle, out var d) && GemmNBatchable(d);

    /// <summary>
    /// Whether this instance can be driven by the continuous-batching engine (issue #190).
    /// Single source of truth for both the loader gate and the runtime guard, so they can't
    /// diverge: dense (non-MoE, non-Gemma-4), no TurboQuant, no active SnapKV, no final-logit
    /// softcap (a dense softcap arch would cap prefill logits but not the un-softcapped batched
    /// decode — keep them out until a batched softcap is wired), and every trunk + output weight
    /// in a GEMM-N-batchable dtype (excludes Q4_0).
    /// </summary>
    public bool SupportsContinuousBatching => _snapKvEffectiveBudget == 0 && DenseBatchedDecodeSupported();

    /// <summary>
    /// The arch/dtype gate shared by <see cref="SupportsContinuousBatching"/> and
    /// <see cref="SupportsBatchVerify"/>: dense (non-MoE, non-Gemma-4), no TurboQuant, no
    /// final-logit softcap, every trunk + output weight GEMM-N-batchable. SnapKV terms are
    /// applied by the callers — continuous batching excludes any configured budget (prefill
    /// into a per-sequence cache could evict), while spec-decode verify only excludes an
    /// actually-compacted owned cache (decode never evicts; an unevicted budget keeps
    /// slot == position).
    /// </summary>
    private bool DenseBatchedDecodeSupported()
    {
        if (_isMoE || _isGemma4Like || _tqEnabled) return false;
        if (_hp.FinalLogitSoftcap > 0f) return false;
        for (int i = 0; i < _hp.NumLayers; i++)
        {
            if (!DecodeBatchable(_wq[i]) || !DecodeBatchable(_wk[i]) ||
                !DecodeBatchable(_wo[i]) || !DecodeBatchable(_wGate[i]) ||
                !DecodeBatchable(_wUp[i]) || !DecodeBatchable(_wDown[i]))
                return false;
            // Dense layers must own a separate V projection (k_eq_v is Gemma-4-only);
            // a null _wv would NRE in BatchForwardMulti, so disable batching defensively.
            if (_wv[i] is not { } wv || !DecodeBatchable(wv)) return false;
        }
        return DecodeBatchable(_wOutput);
    }

    /// <summary>
    /// Guards the batched-serving entry points: continuous batching on the CUDA path is
    /// implemented for dense transformers only. Anything that needs the owned-cache state
    /// machine (TQ ring, SnapKV eviction), per-layer geometry (Gemma 4), a logit softcap, or a
    /// weight dtype the GEMM-N matvec can't drive (Q4_0) is out of scope.
    /// </summary>
    private void ThrowIfBatchingUnsupported(bool decodeOnly = false)
    {
        if (_isMoE)
            throw new NotSupportedException(
                "CUDA continuous batching is not supported for MoE models (router Download/Synchronize per layer).");
        if (_isGemma4Like)
            throw new NotSupportedException(
                "CUDA continuous batching is not supported for Gemma-4-style models (per-layer head_dim, SWA rings, shared-KV aliasing).");
        if (_tqEnabled)
            throw new NotSupportedException(
                "CUDA continuous batching is not supported with the TurboQuant KV cache.");
        // Prefill-capable entry points (CreateCache / PrefillWithCache / PrefillPackedMulti)
        // reject any configured SnapKV budget — a whole-prompt prefill into a bound cache
        // could evict. Decode-only batching (BatchForwardMulti, incl. the issue-#207
        // BatchVerify wrapper) never evicts, so it only rejects an ALREADY-compacted owned
        // cache, where logical position != physical slot.
        if (decodeOnly ? _kvEvictedCount > 0 : _snapKvEffectiveBudget > 0)
            throw new NotSupportedException(decodeOnly
                ? "CUDA batched decode is not supported on a SnapKV-compacted cache (physical slot != logical position)."
                : "CUDA continuous batching is not supported with an active SnapKV budget (eviction runs only on a whole-prompt prefill of the owned cache).");
        if (_hp.FinalLogitSoftcap > 0f)
            throw new NotSupportedException(
                "CUDA continuous batching is not supported with a final-logit softcap (the batched decode finisher does not apply it).");
        // Reaching here, only the weight-dtype loop can make it unsupported.
        if (!DenseBatchedDecodeSupported())
            throw new NotSupportedException(
                "CUDA continuous batching requires every trunk + output weight in a GEMM-N-batchable " +
                "dtype (Q4_K/Q5_K/Q6_K/Q8_0/F32); a Q4_0 weight has no batched-decode matvec kernel.");
    }

    /// <summary>
    /// Allocate a fresh, empty per-sequence GPU KV cache: NumLayers full-context K/V pairs
    /// at the model-wide head_dim / KV-head count and the active KV dtype (#179), mirroring
    /// the dense branch of the constructor's owned-cache allocation. Dense models never
    /// alias KV across layers, so the cache frees every layer on dispose.
    /// </summary>
    internal CudaSequenceKvCache CreateCache()
    {
        ThrowIfBatchingUnsupported();

        // The per-token CUDA-graph decode path bakes the OWNED cache's device pointers into
        // the captured graph; replaying it after BindCache swapped in a per-sequence cache
        // would touch a stale (foreign) pointer. Batched decode issues direct launches and
        // never captures a graph; the only graph-eligible path the engine can still reach is
        // Prefill's single-token Forward fallback. So once this instance is driven by the
        // batching engine (first CreateCache), disable graphs for its lifetime.
        _useCudaGraph = false;

        int kvDim = _numKvHeads * _headDim;
        // q8_0 KV packs 32 elements/block; the store kernels assume each layer's kvDim is a
        // multiple of 32 (mirrors the owned-cache guard in the constructor's dense branch).
        if (_kvDType == DType.Q8_0 && (kvDim & 31) != 0)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE=q8_0 requires kvDim % 32 == 0 (block_q8_0 = 32 elements/block); kvDim={kvDim}.");

        int L = _hp.NumLayers;
        var k = new Tensor[L];
        var v = new Tensor[L];
        // OOM mid-loop is the realistic failure under batch pressure (each admitted sequence
        // reserves 2·NumLayers full-context tensors). Free the partial allocations on throw so
        // the engine's AdmitPending catch (which fails just that request and keeps running)
        // doesn't strand VRAM that no CudaSequenceKvCache owns.
        try
        {
            for (int i = 0; i < L; i++)
            {
                k[i] = _gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), _kvDType);
                v[i] = _gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim), _kvDType);
            }
        }
        catch
        {
            for (int i = 0; i < L; i++)
            {
                if (k[i] is { } ki) _gpu.Free(ki);
                if (v[i] is { } vi) _gpu.Free(vi);
            }
            throw;
        }
        return new CudaSequenceKvCache(_gpu, k, v, s_noAliasedLayers);
    }

    /// <summary>Swap the owned KV-cache pointers for a per-sequence cache's, plumbing its
    /// logical length into the position counter so Prefill appends at the right slots.</summary>
    private void BindCache(CudaSequenceKvCache cache)
    {
        _gpuKCache = cache.K;
        _gpuVCache = cache.V;
        _kvLength = cache.Length;
    }

    /// <summary>Restore the owned KV-cache pointers after a BindCache. Always called in a
    /// finally so an exception mid-Prefill can't leave a foreign cache bound.</summary>
    private void RestoreOwned()
    {
        _gpuKCache = _ownedKCache;
        _gpuVCache = _ownedVCache;
    }

    /// <summary>
    /// Prefill <paramref name="tokens"/> into the per-sequence <paramref name="cache"/> at
    /// <paramref name="startPos"/>. Binds the cache, delegates to the existing
    /// <see cref="Prefill"/> (its batched trunk is both correct and weight-amortized; prefill
    /// captures no per-token graph, so the rebind is safe), writes the advanced length back,
    /// and always restores the owned cache. Returns the logits at the chunk's final token.
    /// </summary>
    internal ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, CudaSequenceKvCache cache, int startPos = 0)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ThrowIfBatchingUnsupported();
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        int savedKvLength = _kvLength;
        BindCache(cache);
        try
        {
            var logits = Prefill(tokens, startPos);
            cache.Length = _kvLength;
            return logits;
        }
        finally
        {
            RestoreOwned();
            _kvLength = savedKvLength;
        }
    }

    /// <summary>
    /// Prefill several pending sequences' chunks. Cross-prompt packing into one forward pass
    /// is a follow-up (issue #190); for now each chunk prefills sequentially into its own
    /// per-sequence cache via <see cref="PrefillWithCache"/> — still correct and still
    /// amortizing the batched-trunk GEMMs within each chunk, just not across prompts.
    /// </summary>
    internal float[]?[] PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, CudaSequenceKvCache[] caches, bool[] wantLogits)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(startPos);
        ArgumentNullException.ThrowIfNull(caches);
        ArgumentNullException.ThrowIfNull(wantLogits);
        ThrowIfBatchingUnsupported();
        int S = chunks.Length;
        if (S == 0) return Array.Empty<float[]?>();
        if (startPos.Length != S || caches.Length != S || wantLogits.Length != S)
            throw new ArgumentException("chunks/startPos/caches/wantLogits lengths must match.");

        var result = new float[]?[S];
        for (int s = 0; s < S; s++)
        {
            var logits = PrefillWithCache(AsList(chunks[s]), caches[s], startPos[s]);
            result[s] = wantLogits[s] ? logits.ToArray() : null;
        }
        return result;
    }

    /// <summary>Expose a token chunk as IReadOnlyList without copying when it's array-backed
    /// (the engine always builds chunks from <c>int[].AsMemory</c>).</summary>
    private static IReadOnlyList<int> AsList(ReadOnlyMemory<int> mem) =>
        MemoryMarshal.TryGetArray(mem, out ArraySegment<int> seg) ? seg : mem.ToArray();

    /// <summary>One batched-decode matmul. Default: the weight-stationary small-N matvec
    /// (#194 — weight HBM read amortized across the batch, bit-identical reduction to the
    /// GEMM-N matvec; N=1 / N&gt;16 delegate to GEMM-N inside the backend).
    /// SHARPI_BATCH_DECODE_WS=0: the #190 GEMM-N matvec. SHARPI_BATCH_DECODE_MMQ=1: the
    /// #201 int8 tensor-core decode tile for big Q4_K-SoA shapes (argmax-stable, WS
    /// fallback per tensor). SHARPI_BATCH_DECODE_GEMM=1: the compute-bound GEMM/MMQ path
    /// (the same routing <see cref="GpuMatMulBatchedCore"/> uses for prefill) —
    /// argmax-stable only, kept as the A/B toggle for very high concurrency.</summary>
    private void BatchDecodeMatMul(Tensor outAll, Tensor weights, Tensor inAll, int n)
    {
        if (_batchDecodeComputeBound)
            GpuMatMulBatchedCore(outAll, weights, inAll, n);
        else if (_batchDecodeMmq)
            _gpu.MatMulBatchedDecodeMmq(outAll, weights, inAll, n, WDType(weights));
        else if (_batchDecodeWeightStationary)
            _gpu.MatMulBatchedWeightStationary(outAll, weights, inAll, n, WDType(weights));
        else
            _gpu.MatMulBatched(outAll, weights, inAll, n, WDType(weights));
    }

    /// <summary>
    /// Ragged batched decode (issue #190): one token per sequence, each at its own position
    /// against its own per-sequence cache, with the dense weight reads amortized N× across
    /// the batch. This is a TRUE batched pass — it issues direct launches and never replays
    /// the per-token CUDA graph (which would bake in owned-cache pointers). It adapts the
    /// batched-trunk prefill (<see cref="PrefillBatchedTrunk"/>) to N sequences: batched
    /// embed → per layer {batched RmsNorm + batched QKV GEMM-N, then per-sequence RoPE /
    /// QK-norm / KV-append / single-query attention against that sequence's cache, then
    /// batched O-proj + batched FFN} → per-row final norm + one batched output GEMM. The
    /// per-sequence attention block mirrors the dense <see cref="RunDeviceRegion"/> ordering
    /// exactly (QK-norm before RoPE, #157), so it is argmax-stable vs the single-user loop.
    /// </summary>
    internal float[][] BatchForwardMulti(int[] tokens, int[] positions, CudaSequenceKvCache[] caches)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(positions);
        ArgumentNullException.ThrowIfNull(caches);
        ThrowIfBatchingUnsupported(decodeOnly: true);
        int N = tokens.Length;
        if (N == 0) return Array.Empty<float[]>();
        if (positions.Length != N || caches.Length != N)
            throw new ArgumentException("tokens/positions/caches lengths must match.");

        int embDim = _embDim;
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int vocab = _hp.VocabSize;

        EnsureBatchedTrunkScratch(N);
        EnsureDecodeLogits(N);

        // Ragged path (#197, default): the per-layer QK-norm/RoPE/KV-append/attention run as
        // O(1) ragged-batched launches over all N sequences. Build the [layer][seq] cache
        // tensor table once per batch composition (identity-compared; steps within a stable
        // batch reuse it) and grab the spill scratch only if some sequence is past the
        // 4096-slot shared-memory fast path.
        bool ragged = _batchDecodeRagged;
        Tensor? raggedScores = null;
        if (ragged)
        {
            EnsureRaggedCacheTable(caches);
            raggedScores = EnsureRaggedAttnScores(N, positions);
        }

        // Per-sequence views into the batched Q/K/V/attnOut scratch — only the legacy
        // per-sequence loop (SHARPI_BATCH_DECODE_RAGGED=0) needs them; the ragged kernels
        // index rows internally. The offsets are fixed across layers (the batched buffers
        // are reused each layer), so build them once and free them at the end rather than
        // per-layer. _maxHeadDim == _headDim on the dense path, so the q/k/v/attn buffers
        // are exactly N×qDim / N×kvDim with no padding stride. (The O-projection bias add
        // needs per-row hidden views too, but only when _hasAttnBias — created inline in
        // that rare branch so the common no-bias path allocates nothing extra.)
        // Allocated INSIDE the try so a mid-loop View throw still frees the views taken so far.
        var qViews = new Tensor[N];
        var kViews = new Tensor[N];
        var vViews = new Tensor[N];
        var aViews = new Tensor[N];
        try
        {
            if (!ragged)
                for (int n = 0; n < N; n++)
                {
                    qViews[n] = _gpu.View(_bpQ!, (long)n * qDim, qDim);
                    kViews[n] = _gpu.View(_bpK!, (long)n * kvDim, kvDim);
                    vViews[n] = _gpu.View(_bpV!, (long)n * kvDim, kvDim);
                    aViews[n] = _gpu.View(_bpAttnOut!, (long)n * qDim, qDim);
                }

            // 1. Embed each sequence's current token into the batched hidden buffer. (No
            //    embedding scale: the dense per-token Forward oracle applies none — that
            //    sqrt(embDim) factor is Gemma-only, which is out of scope here.)
            for (int n = 0; n < N; n++)
            {
                EmbedTokenGpu(tokens[n]); // writes _hidden
                _gpu.CopyDeviceRegion(_bpHidden!, (long)n * embDim * sizeof(float),
                                      _hidden, 0, (long)embDim * sizeof(float));
            }

            // 2. Transformer layers: batched dense GEMMs + per-sequence attention.
            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                _gpu.CopyDevice(_bpResidual!, _bpHidden!);
                _gpu.RmsNormBatched(_bpNorm!, _bpHidden!, _wAttnNorm[layer], N, embDim, _hp.RmsNormEps);

                // Batched QKV. Default weight-stationary matvec (#194: weight read amortized
                // across the batch, bit-identical reduction to the per-token oracle's kernels);
                // SHARPI_BATCH_DECODE_WS=0 / SHARPI_BATCH_DECODE_GEMM=1 select the #190
                // GEMM-N / compute-bound GEMM-MMQ alternatives (see BatchDecodeMatMul).
                BatchDecodeMatMul(_bpQ!, _wq[layer], _bpNorm!, N);
                BatchDecodeMatMul(_bpK!, _wk[layer], _bpNorm!, N);
                BatchDecodeMatMul(_bpV!, _wv[layer]!, _bpNorm!, N);

                bool useRoPE = _hp.NoRopeLayerStep == 0 || (layer + 1) % _hp.NoRopeLayerStep != 0;

                if (ragged)
                {
                    // Ragged-batched (#197): same op sequence as the per-sequence loop below
                    // (bias → QK-norm/RoPE, #157 order → KV append → attention), each op one
                    // launch whose grid covers all N rows at positions[n] against caches[n].
                    // Every kernel keeps its per-token counterpart's reduction chain, so per
                    // sequence this is bit-identical to the loop it replaces. _kvEvictedCount
                    // is 0 (SnapKV is rejected for batching), so the physical slot is pos.
                    if (_hasAttnBias)
                    {
                        _gpu.AddBiasBatched(_bpQ!, _bq![layer], qDim, N);
                        _gpu.AddBiasBatched(_bpK!, _bk![layer], kvDim, N);
                        _gpu.AddBiasBatched(_bpV!, _bv![layer], kvDim, N);
                    }

                    if (_hasQkNorm && !_hp.UseL2QkNorm)
                        _gpu.HeadNormQkBatched(_bpQ!, _wqNorm![layer], _bpK!, _wkNorm![layer],
                            _numHeads, _numKvHeads, _headDim, N, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    if (useRoPE)
                    {
                        _gpu.RoPEBatchedRagged(_bpQ!, positions, _numHeads, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                        _gpu.RoPEBatchedRagged(_bpK!, positions, _numKvHeads, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    }
                    if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                    {
                        _gpu.HeadNormPureBatched(_bpQ!, _numHeads, _headDim, N, _hp.RmsNormEps);
                        _gpu.HeadNormPureBatched(_bpK!, _numKvHeads, _headDim, N, _hp.RmsNormEps);
                    }

                    _gpu.KvAppendBatchedRagged(_bpK!, _bpV!, _raggedKLayers![layer], _raggedVLayers![layer],
                        positions, kvDim, _maxSeqLen, _kvDType);
                    _gpu.AttentionBatchedRagged(_bpQ!, _raggedKLayers[layer], _raggedVLayers[layer],
                        _bpAttnOut!, raggedScores,
                        _numHeads, _numKvHeads, _headDim, positions, _maxSeqLen, _attnScale, _kvDType);
                    // caches[n].Length is advanced once after the pass completes (below).
                }
                else
                // Per-sequence (#190, SHARPI_BATCH_DECODE_RAGGED=0): bias → QK-norm/RoPE (same
                // order as RunDeviceRegion, #157) → KV append into that sequence's own cache →
                // single-query attention over its [0, pos+1). _kvEvictedCount is 0 (SnapKV is
                // rejected for batching), so the physical slot is simply pos.
                for (int n = 0; n < N; n++)
                {
                    int pos = positions[n];
                    Tensor qv = qViews[n], kv = kViews[n], vv = vViews[n], av = aViews[n];

                    if (_hasAttnBias)
                    {
                        _gpu.AddInPlace(qv, _bq![layer]);
                        _gpu.AddInPlace(kv, _bk![layer]);
                        _gpu.AddInPlace(vv, _bv![layer]);
                    }

                    if (_hasQkNorm && !_hp.UseL2QkNorm)
                    {
                        _gpu.HeadNorm(qv, _wqNorm![layer], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                        _gpu.HeadNorm(kv, _wkNorm![layer], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                    }
                    if (useRoPE)
                    {
                        _gpu.RoPE(qv, pos, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                        _gpu.RoPE(kv, pos, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                    }
                    if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
                    {
                        _gpu.HeadNormPure(qv, _numHeads, _headDim, _hp.RmsNormEps);
                        _gpu.HeadNormPure(kv, _numKvHeads, _headDim, _hp.RmsNormEps);
                    }

                    Tensor kc = caches[n].K[layer], vc = caches[n].V[layer];
                    KvAppendKv(kv, vv, kc, vc, kvDim, pos, _maxSeqLen);
                    AttentionKv(qv, kc, vc, av, _attnScoresScratch,
                        _numHeads, _numKvHeads, _headDim, pos + 1, _maxSeqLen, _attnScale);
                    // caches[n].Length is advanced once after the pass completes (below), not
                    // here — a mid-pass throw then leaves the logical length unadvanced.
                }

                // Batched O-projection + residual. The attn-output bias is per-row; the ragged
                // path adds it in one broadcast launch, the legacy loop via per-sequence hidden
                // views created inline here (rare branch) to keep the common no-bias decode
                // path free of unused per-row views.
                BatchDecodeMatMul(_bpHidden!, _wo[layer], _bpAttnOut!, N);
                if (_hasAttnBias)
                {
                    if (ragged)
                        _gpu.AddBiasBatched(_bpHidden!, _bo![layer], embDim, N);
                    else
                        for (int n = 0; n < N; n++)
                        {
                            var hv = _gpu.View(_bpHidden!, (long)n * embDim, embDim);
                            _gpu.AddInPlace(hv, _bo![layer]);
                            _gpu.Free(hv);
                        }
                }
                _gpu.AddInPlace(_bpHidden!, _bpResidual!);

                // FFN (dense SwiGLU), batched across N.
                _gpu.CopyDevice(_bpResidual!, _bpHidden!);
                _gpu.RmsNormBatched(_bpNorm!, _bpHidden!, _wFfnNorm[layer], N, embDim, _hp.RmsNormEps);
                BatchDecodeMatMul(_bpFfnGate!, _wGate[layer], _bpNorm!, N);
                BatchDecodeMatMul(_bpFfnUp!,   _wUp[layer],   _bpNorm!, N);
                _gpu.SiLuMul(_bpFfnGate!, _bpFfnUp!);
                BatchDecodeMatMul(_bpHidden!, _wDown[layer], _bpFfnGate!, N);
                _gpu.AddInPlace(_bpHidden!, _bpResidual!);
            }

            // 3. Final norm + output projection, batched (the output weight is the largest
            //    single matmul, so amortizing its read across N is the main throughput win).
            _gpu.RmsNormBatched(_bpHidden!, _bpHidden!, _wOutputNorm, N, embDim, _hp.RmsNormEps);
            BatchDecodeMatMul(_decodeLogitsAll!, _wOutput, _bpHidden!, N);
            _gpu.Download(_decodeLogitsAll!, _decodeLogitsHost.AsSpan(0, N * vocab));
            _gpu.Synchronize();

            var result = new float[N][];
            for (int n = 0; n < N; n++)
            {
                result[n] = new float[vocab];
                Array.Copy(_decodeLogitsHost!, (long)n * vocab, result[n], 0, vocab);
                // Advance each sequence's logical length now that the append + attention for
                // this token have completed and synchronized (transactional: a mid-pass throw
                // leaves Length untouched).
                caches[n].Length = positions[n] + 1;
            }
            return result;
        }
        finally
        {
            // View arrays may be partially populated if a View call above threw — null-check
            // each (Tensor is a ref type; Free(null) would NRE) so cleanup is exception-safe.
            for (int n = 0; n < N; n++)
            {
                if (qViews[n] is { } qv) _gpu.Free(qv);
                if (kViews[n] is { } kv) _gpu.Free(kv);
                if (vViews[n] is { } vv) _gpu.Free(vv);
                if (aViews[n] is { } av) _gpu.Free(av);
            }
        }
    }

    // ── Speculative-decode batched verify (issue #207) ──────────────────────────────────

    /// <summary>
    /// Whether <see cref="BatchVerify"/> can run: the dense batched-decode configuration
    /// (<see cref="DenseBatchedDecodeSupported"/> — non-MoE, non-Gemma-4, no TurboQuant, no
    /// final-logit softcap, GEMM-N-batchable weights) with an uncompacted cache. Unlike
    /// <see cref="SupportsContinuousBatching"/>, a CONFIGURED SnapKV budget does not disable
    /// verify — only an actual prefill-time eviction does (then physical slot != logical
    /// position and the batched kernels would mis-index). Dynamic: flips false after such a
    /// prefill, so the speculative decoder (which re-checks per step) degrades to sequential
    /// verify — the same once-evicted gating the GDN passes use (#130).
    /// </summary>
    public bool SupportsBatchVerify => _kvEvictedCount == 0 && DenseBatchedDecodeSupported();

    /// <summary>
    /// Batched k-token verify for single-user speculative decoding (issue #207): one packed
    /// pass over the OWNED cache at contiguous positions [<paramref name="startPos"/>,
    /// <paramref name="startPos"/> + k), returning <c>result[i]</c> = logits after
    /// <c>tokens[i]</c>. Reuses <see cref="BatchForwardMulti"/>'s trunk with every row bound
    /// to the same cache: the ragged kernels append all k K/V rows before any row attends,
    /// and row i attends over [0, startPos+i] — i.e. packed causal attention (the legacy
    /// per-sequence fallback loop appends-then-attends in ascending row order, equally
    /// causal). Every matmul routes through <see cref="BatchDecodeMatMul"/>, so the #194
    /// weight-stationary kernels (or the opt-in #201 decode MMQ) amortize the weight HBM
    /// reads k×. Each row keeps the per-token kernels' reduction chains (#194/#197), so the
    /// default WS path is expected bit-identical to k sequential <see cref="Forward"/> calls;
    /// the opt-in compute-bound/MMQ toggles are argmax-stable only. All k K/V entries land in
    /// the cache; the caller rewinds rejected tokens via <see cref="TruncateTo"/>. Issues
    /// direct launches only — the per-token decode CUDA graph (owned-cache pointers) stays
    /// valid for the surrounding Forward steps.
    /// </summary>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (!SupportsBatchVerify)
            throw new NotSupportedException(
                "BatchVerify requires the dense batching-capable configuration (no MoE / " +
                "Gemma-4 / TurboQuant / SnapKV / softcap, GEMM-N-batchable weights) and an " +
                "uncompacted cache. Check SupportsBatchVerify before calling.");
        int k = tokens.Length;
        if (k == 0) return Array.Empty<float[]>();
        if (startPos < 0 || startPos + k > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos),
                $"BatchVerify range [{startPos}, {startPos + k}) exceeds the context window (maxSeqLen={_maxSeqLen}).");

        if (k == 1)
        {
            // A single token amortizes nothing — the per-token Forward (CUDA-graph
            // replayable) is strictly better. Mirrors the CPU BatchVerify fallback.
            var logits = Forward(tokens[0], startPos);
            var seq = new float[1][];
            seq[0] = new float[_hp.VocabSize];
            logits.CopyTo(seq[0]);
            return seq;
        }

        if (_ownedCacheView is null)
        {
            var all = new HashSet<int>();
            for (int l = 0; l < _hp.NumLayers; l++) all.Add(l);
            _ownedCacheView = new CudaSequenceKvCache(_gpu, _ownedKCache, _ownedVCache, all);
        }
        _ownedCacheView.Length = startPos;

        var positions = new int[k];
        for (int i = 0; i < k; i++) positions[i] = startPos + i;
        var caches = new CudaSequenceKvCache[k];
        Array.Fill(caches, _ownedCacheView);

        var result = BatchForwardMulti(tokens, positions, caches);
        // Mirror what k sequential Forward calls would leave behind; the speculative
        // decoder's TruncateTo(startPos + accepted) then rewinds the rejected tail.
        _kvLength = Math.Max(_kvLength, startPos + k);
        return result;
    }

    /// <summary>(Re)allocate the batched-decode logits buffer [<paramref name="n"/> × vocab]
    /// and its host download buffer when the decode batch size changes.</summary>
    private void EnsureDecodeLogits(int n)
    {
        if (_decodeLogitsCapacity == n) return;
        // The device buffer + Array.Copy offsets are computed in long, but the host array
        // length and Download span are int (array lengths are int). Guard the int*int product
        // so an extreme batch×vocab can't silently wrap to a too-small host buffer (unreachable
        // at sane batch sizes — N ≤ maxBatch, ~14K at vocab 152K — but fail loud, not corrupt).
        long total = (long)n * _hp.VocabSize;
        if (total > int.MaxValue)
            throw new NotSupportedException(
                $"Batched decode logits buffer ({n}×{_hp.VocabSize}) exceeds int.MaxValue; reduce SHARPI_MAX_BATCH.");
        if (_decodeLogitsAll is { } old) _gpu.Free(old);
        _decodeLogitsAll = _gpu.Allocate(TensorShape.D1(total));
        _decodeLogitsHost = new float[(int)total];
        _decodeLogitsCapacity = n;
    }

    /// <summary>
    /// (Re)build the ragged kernels' [layer][seq] K/V cache tensor table when the batch
    /// composition changed (issue #197). Compared by element identity: a stable batch
    /// reuses the table across decode steps; any admit/retire (different object at any
    /// slot, or different length) rebuilds it. The snapshot is a defensive clone so a
    /// caller mutating its array between steps can't alias the comparison.
    /// </summary>
    private void EnsureRaggedCacheTable(CudaSequenceKvCache[] caches)
    {
        if (_raggedSnapshot is { } snap && snap.Length == caches.Length)
        {
            bool same = true;
            for (int i = 0; i < caches.Length; i++)
                if (!ReferenceEquals(snap[i], caches[i])) { same = false; break; }
            if (same) return;
        }

        int layers = _hp.NumLayers, count = caches.Length;
        var k = new Tensor[layers][];
        var v = new Tensor[layers][];
        for (int l = 0; l < layers; l++)
        {
            k[l] = new Tensor[count];
            v[l] = new Tensor[count];
            for (int n = 0; n < count; n++)
            {
                k[l][n] = caches[n].K[l];
                v[l][n] = caches[n].V[l];
            }
        }
        _raggedKLayers = k;
        _raggedVLayers = v;
        _raggedSnapshot = (CudaSequenceKvCache[])caches.Clone();
    }

    /// <summary>
    /// Spill scratch for the ragged attention kernel (issue #197): null while every
    /// sequence fits the kernel's 4096-slot shared-memory score path (the common case —
    /// the kernel never dereferences the scratch then); otherwise an
    /// [N × numHeads × maxSeqLen] buffer of per-(sequence, head) score rows, lazily
    /// allocated and re-sized only when the decode batch capacity changes.
    /// </summary>
    private Tensor? EnsureRaggedAttnScores(int n, int[] positions)
    {
        int maxLen = 0;
        for (int i = 0; i < positions.Length; i++) maxLen = Math.Max(maxLen, positions[i] + 1);
        if (maxLen <= 4096) return null;

        if (_raggedAttnScoresCapacity != n)
        {
            if (_raggedAttnScores is { } old) _gpu.Free(old);
            _raggedAttnScores = _gpu.Allocate(TensorShape.D1((long)n * _numHeads * _maxSeqLen));
            _raggedAttnScoresCapacity = n;
        }
        return _raggedAttnScores;
    }

    // Explicit IBatchedForwardPass surface: the engine holds caches as opaque
    // ISequenceKvCache handles; unwrap them to the concrete CUDA cache the methods above take.
    ISequenceKvCache IBatchedForwardPass.CreateCache() => CreateCache();

    ReadOnlySpan<float> IBatchedForwardPass.PrefillWithCache(
        IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos)
        => PrefillWithCache(tokens, (CudaSequenceKvCache)cache, startPos);

    float[]?[] IBatchedForwardPass.PrefillPackedMulti(
        ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits)
        => PrefillPackedMulti(chunks, startPos, Cast(caches), wantLogits);

    float[][] IBatchedForwardPass.BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        => BatchForwardMulti(tokens, positions, Cast(caches));

    private static CudaSequenceKvCache[] Cast(ISequenceKvCache[] caches)
    {
        var r = new CudaSequenceKvCache[caches.Length];
        for (int i = 0; i < caches.Length; i++)
            r[i] = (CudaSequenceKvCache)caches[i];
        return r;
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
        else if (info.DType == DType.Q4_0 || info.DType == DType.Q4_K || info.DType == DType.Q6_K || info.DType == DType.Q8_0)
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
            // #156: same for 2-D Q4_K trunk weights (dense-only; cols % 256 required —
            // every Q4_K hidden dim satisfies it, so the 2-D check is sufficient).
            else if (_q4kSoa && !_isMoE && info.DType == DType.Q4_K && info.NDimensions == 2)
            {
                int cols = (int)info.Dimensions[0];
                int rows = (int)info.Dimensions[1];
                result = _gpu.RepackQ4KSoa(result, rows, cols);
            }
            // #124/#173: same funnelshift-killing SoA repack for 2-D Q4_0 trunk weights
            // (Gemma 4 12B QAT). cols % 32 required — every Q4_0 hidden dim satisfies it.
            // Gated on the same SHARPI_MMQ_SOA flag as Q8_0 (#149); dense-only.
            else if (_mmqSoa && !_isMoE && info.DType == DType.Q4_0 && info.NDimensions == 2)
            {
                int cols = (int)info.Dimensions[0];
                int rows = (int)info.Dimensions[1];
                result = _gpu.RepackQ4_0Soa(result, rows, cols);
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
    /// VRAM left for the KV cache after weights, attention/FFN scratch, and the driver
    /// reserve. This is the budget the context estimator divides by per-token KV bytes,
    /// and the budget the auto-narrow heuristic (issue #185) compares the fp32 KV
    /// footprint against. Single-sourced so both stay in agreement.
    /// </summary>
    internal static long EstimateAvailableKvVram(GgufModel model, CudaBackend gpu, ModelHyperparams hp)
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

        long reserved = KvVramReserveBytes(hp, vramBytes);
        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024;
        return available;
    }

    /// <summary>
    /// VRAM held back from the KV budget for everything that isn't weights/scratch/KV: the CUDA
    /// context/framebuffer, the cuBLAS workspace, the Q8_1 quantization scratch, the pinned host
    /// buffer, the GPU buffer-pool reuse lists, and the transient prefill working set. Pure
    /// (no GPU/GGUF) so it's unit-testable (cf. <see cref="ResolveKvDType"/>).
    /// <para><b>Uniform / dense models</b> keep the proven <c>max(VRAM/3, 2 GiB)</c> reserve.
    /// Their KV grows linearly to fill whatever budget is left, so the reserve is the only thing
    /// bounding the cache size — and an earlier <c>max(VRAM/5, 1 GiB)</c> once left ~24 MiB free
    /// on a 12 GiB card running Qwen3-8B, spilling the ~600 MiB lm-head to system RAM over PCIe
    /// and collapsing prefill ~65→4 t/s (#185). This path is unchanged.</para>
    /// <para><b>SWA / per-layer (Gemma 4) models</b> use a smaller, bounded reserve — a fixed
    /// system allowance (NOT a fraction of total VRAM) plus one <see cref="PrefillBatchChunk"/>'s
    /// activation working set. This is safe precisely because SWA KV <i>saturates</i>: past the
    /// sliding-window ring only the few global layers grow, so the cache asymptotes to
    /// <c>KV(modelMax)</c> and a larger budget cannot grow it to consume the headroom (the dense
    /// failure mode). The old <c>VRAM/3</c> reserve over-reserved ~2.5 GB on a 12 GiB card and
    /// pinned Gemma 4 12B q8_0 auto-context to ~30 K when the full 256 K fits (#228 / #220).</para>
    /// </summary>
    internal static long KvVramReserveBytes(ModelHyperparams hp, long vramBytes)
    {
        if (hp.IsSwaLayer is null)
            return Math.Max(vramBytes / 3, 2L * 1024 * 1024 * 1024);

        // Transient activations for one batched-prefill chunk (norm/qkv/attn/FFN buffers),
        // sized to the model width — the only reserve term that scales with the model rather
        // than the GPU. Generous (overlapping buffers) so the budget can't starve a real prefill.
        long prefillWorkingSet = (long)PrefillBatchChunk *
            (hp.EmbeddingDim * 4L + hp.IntermediateDim * 2L
             + (long)hp.NumHeads * hp.HeadDim + 2L * hp.NumKvHeads * hp.HeadDim) * sizeof(float);
        // Fixed system allowance (CUDA context/framebuffer + cuBLAS workspace + pinned + pool),
        // floored at 2 GiB and rising to VRAM/6 on larger cards where those costs grow.
        long systemReserve = Math.Max(2L * 1024 * 1024 * 1024, vramBytes / 6);
        return systemReserve + prefillWorkingSet;
    }

    /// <summary>
    /// Total KV-cache bytes (K + V, summed over layers) for the given context at the given
    /// element dtype. Mirrors the ctor's per-layer allocation: per-layer head_dim / kv-head
    /// counts and SWA window-ring sizing for gemma4-style models, the flat
    /// <c>NumLayers × kvDim × maxCtx</c> formula otherwise; KV-share layers (Gemma 4 tail)
    /// alias the source and allocate nothing. Used by the auto-narrow heuristic (#185) to
    /// compare the fp32 / bf16 / q8_0 footprints against <see cref="EstimateAvailableKvVram"/>.
    /// <paramref name="gpuLayers"/> (default -1 = all) scopes the sum to the first N
    /// GPU-resident layers, used by TierPlanner to price the GPU KV budget for a candidate split.
    /// </summary>
    internal static long EstimateKvCacheBytes(ModelHyperparams hp, int maxCtx, DType kvDType, int gpuLayers = -1)
    {
        bool perLayerKv = hp.LayerHeadDim is not null;
        int swaWindow = hp.SlidingWindowSize > 0 ? hp.SlidingWindowSize : maxCtx;
        long total = 0;
        int layerCount = gpuLayers < 0 ? hp.NumLayers : Math.Min(gpuLayers, hp.NumLayers);
        for (int i = 0; i < layerCount; i++)
        {
            if (hp.KvSourceLayer is { } ksl && ksl[i] >= 0) continue; // aliased — no own pages
            int layerHd = perLayerKv ? hp.LayerHeadDim![i] : hp.HeadDim;
            int layerKvHeads = hp.LayerKvHeads is { } lkv ? lkv[i] : hp.NumKvHeads;
            long layerKvDim = (long)layerKvHeads * layerHd;
            long layerCtx = (perLayerKv && hp.IsSwaLayer is { } swa && swa[i])
                ? SwaRingSize(maxCtx, swaWindow)
                : maxCtx;
            // The K and V buffers are allocated through the GPU buffer pool
            // (gpu.Allocate, exact:false), which rounds every buffer up to the next power
            // of two (GpuBufferPool.RoundUp). Round each buffer the same way so the
            // estimate matches the VRAM the ctor will actually reserve: a raw byte sum
            // undercounts by up to ~2× per buffer (q8_0 especially — its 34-byte blocks
            // rarely land on a power of two), which could wrongly conclude fp32 fits and
            // defeat the auto-narrow, leaving the original cudaMalloc failure.
            long bufBytes = (long)CudaBackend.RoundUpAllocBytes(
                (nuint)DTypeInfo.ByteSize(layerCtx * layerKvDim, kvDType));
            total += 2 * bufBytes; // K + V
        }
        return total;
    }

    /// <summary>
    /// True when every (non-aliased) layer's kvDim is a multiple of 32 — the q8_0 block size.
    /// The q8_0 KV store quantizes per 32-lane warp and assumes blocks never straddle a KV
    /// row, so a layer with kvDim % 32 != 0 cannot use q8_0 (the ctor would throw). The
    /// auto-narrow heuristic checks this before falling to q8_0 (#185).
    /// </summary>
    internal static bool Q8KvGeometrySupported(ModelHyperparams hp)
    {
        bool perLayerKv = hp.LayerHeadDim is not null;
        for (int i = 0; i < hp.NumLayers; i++)
        {
            if (hp.KvSourceLayer is { } ksl && ksl[i] >= 0) continue;
            int layerHd = perLayerKv ? hp.LayerHeadDim![i] : hp.HeadDim;
            int layerKvHeads = hp.LayerKvHeads is { } lkv ? lkv[i] : hp.NumKvHeads;
            if ((((long)layerKvHeads * layerHd) & 31) != 0) return false;
        }
        return true;
    }

    /// <summary>
    /// The auto-narrow decision (issue #185 item 1), factored out as a pure function so it
    /// can be unit-tested without a GPU or model. Returns the KV dtype to use and sets
    /// <paramref name="autoNarrowed"/> when it picked a narrowed dtype the operator did not
    /// request. Rules, in order:
    /// <list type="bullet">
    ///   <item>An explicit operator choice, a TQ run (own KV path), or a non-fp32 request is
    ///         returned unchanged — explicit choices are never overridden.</item>
    ///   <item>If fp32 KV fits the budget, fp32 is kept.</item>
    ///   <item>Else bf16 if it fits; else q8_0 if the geometry supports it (narrowest store);
    ///         else bf16 best-effort (the only narrowed store valid for any geometry — may
    ///         still cudaMalloc-fail, but halves the footprint vs fp32).</item>
    /// </list>
    /// </summary>
    internal static DType ResolveKvDType(
        DType requested, bool explicitChoice, bool tqEnabled,
        long availableKvBytes, long fp32KvBytes, long bf16KvBytes, bool q8Supported,
        out bool autoNarrowed)
    {
        autoNarrowed = false;
        if (explicitChoice || tqEnabled || requested != DType.Float32) return requested;
        if (fp32KvBytes <= availableKvBytes) return requested;
        autoNarrowed = true;
        if (bf16KvBytes <= availableKvBytes) return DType.BFloat16;
        return q8Supported ? DType.Q8_0 : DType.BFloat16;
    }

    /// <summary>
    /// VRAM-based context-length estimator: take the KV-cache budget from
    /// <see cref="EstimateAvailableKvVram"/> and find the largest context whose KV cache fits,
    /// at the element width <paramref name="kvDType"/> the forward pass will actually allocate.
    /// For gemma4-style models (per-layer head_dim / SWA rings / KV-share aliasing) this binary-
    /// searches against <see cref="EstimateKvCacheBytes"/> — the same allocator-exact arithmetic
    /// the constructor reserves — so bf16/q8_0 correctly buy ~2×/4× the positions of fp32 (#220).
    /// Uniform-attention models keep the flat <c>NumLayers × kvDim × maxCtx</c> fp32 formula.
    /// </summary>
    public static int EstimateMaxContext(
        GgufModel model, CudaBackend gpu, ModelHyperparams hp, DType kvDType = DType.Float32)
        => SolveMaxCtxForKv(hp, EstimateAvailableKvVram(model, gpu, hp), kvDType);

    /// <summary>
    /// Pure (GPU-free, GGUF-free) core of <see cref="EstimateMaxContext"/>: the largest context
    /// whose KV cache fits <paramref name="availableKvBytes"/> at element width
    /// <paramref name="kvDType"/>. Factored out for unit testing (cf. <see cref="ResolveKvDType"/>).
    /// <para>Gemma 4-style models (per-layer head_dim + SWA pattern) binary-search against
    /// <see cref="EstimateKvCacheBytes"/> — the allocator-exact arithmetic (dtype + SWA ring +
    /// KV-share skip + per-layer KV heads + pow2 round-up) the ctor reserves and
    /// <c>TierPlanner.SolveGpuCtxForPerLayerKv</c> uses — so the estimate can't drift from what is
    /// actually allocated and bf16/q8_0 correctly buy more context (#220). Because SWA layers
    /// stop growing past their ring cap, the gain over fp32 exceeds the bare width ratio once the
    /// context clears that cap. Uniform-attention models keep the flat fp32 formula unchanged.</para>
    /// </summary>
    internal static int SolveMaxCtxForKv(ModelHyperparams hp, long availableKvBytes, DType kvDType)
    {
        const int floorCtx = 512;
        int cap = hp.ContextLength;

        // A model whose context is at/below the floor clamps to the cap (and avoids the
        // Math.Clamp(_, 512, cap) below throwing when cap < 512). Mirrors the floor convention:
        // return the small ctx and let the ctor's allocation fail loudly if even that won't fit.
        if (cap <= floorCtx)
            return cap;

        if (hp.LayerHeadDim is not null && hp.IsSwaLayer is not null)
        {
            // EstimateKvCacheBytes is monotonic non-decreasing in ctx, so an upper-bound binary
            // search converges. Floor 512 mirrors the uniform clamp below.
            if (EstimateKvCacheBytes(hp, floorCtx, kvDType) > availableKvBytes)
                return floorCtx;
            int lo = floorCtx, hi = cap;
            while (lo < hi)
            {
                int mid = lo + (hi - lo + 1) / 2;
                if (EstimateKvCacheBytes(hp, mid, kvDType) <= availableKvBytes)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            return lo;
        }

        // Uniform-attention models: unchanged flat fp32 formula (#220 is scoped to the
        // SWA/per-layer Gemma path; dtype-aware sizing for uniform models is out of scope).
        long bytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * hp.HeadDim * sizeof(float);
        int maxCtx = (int)(availableKvBytes / bytesPerToken);
        return Math.Clamp(maxCtx, floorCtx, cap);
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
        // layout. This set must match UploadWeight's raw-upload branch exactly
        // ({Float32, Q4_0, Q4_K, Q6_K, Q8_0}) — every other dtype (e.g. Q5_K) is
        // CPU-dequantized to F32 there and so genuinely occupies fp32 bytes on the GPU.
        // Q8_0 is ~1.0625 bytes/elem and Q4_0 ~0.56 vs the 4 bytes/elem the F32 fallback
        // would burn; omitting Q4_0 (the Gemma 4 12B QAT weight dtype) over-counts weights
        // ~7×, floors the KV budget, and would wrongly auto-narrow models that fit (#185).
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_0
            || tensor.DType == DType.Q4_K || tensor.DType == DType.Q6_K
            || tensor.DType == DType.Q8_0)
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
            // Gemma 4 12B global layers (k_eq_v) carry K but no V (_wv[i] is null).
            if (!kvShared)
            {
                _gpu.Free(_wk[i]);
                if (_wv[i] is not null) _gpu.Free(_wv[i]);
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
            // Free the OWNED arrays (issue #190): _gpuKCache/_gpuVCache may transiently
            // point at a per-sequence cache mid-bind, but every bind restores in a finally,
            // so at teardown they equal the owned arrays — free those directly to be robust.
            _gpu.Free(_ownedKCache[i]);
            _gpu.Free(_ownedVCache[i]);
        }

        // Batched-decode logits scratch (issue #190; allocated only if batching ran).
        if (_decodeLogitsAll is { } dl) _gpu.Free(dl);
        // Ragged attention spill scratch (issue #197; allocated only on a >4096-length
        // batched decode). The [layer][seq] cache table holds borrowed tensor refs the
        // per-sequence caches own — nothing to free there.
        if (_raggedAttnScores is { } ras) _gpu.Free(ras);

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
