using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Pipeline;
using SharpInference.TurboQuant;
using SharpInference.Cuda;

namespace SharpInference.Engine;

/// <summary>
/// Hybrid GPU/CPU forward pass for models larger than VRAM.
/// First N layers run on GPU (CUDA / cuBLAS + NVRTC kernels), remaining layers on CPU (AVX2 SIMD).
/// Hidden state transfers via pinned host memory at GPU↔CPU boundaries.
/// </summary>
public sealed unsafe class CudaHybridForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly CudaBackend _gpu;
    private readonly ModelHyperparams _hp;
    private readonly LayerPlacement _placement;

    // Dimensions
    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim, _expertDim;
    private readonly int _nGpuLayers, _nCpuLayers;

    // ── GPU resources (layers 0..nGpuLayers-1) ──
    private readonly Tensor _gpuHidden, _gpuResidual, _gpuNormBuf;
    private readonly Tensor _gpuQ, _gpuK, _gpuV, _gpuAttnOut;
    private readonly Tensor _gpuFfnGate, _gpuFfnUp;
    private readonly Tensor _gpuLogits;
    private readonly Tensor? _gpuRouterLogits, _gpuMoeSharedOut, _gpuMoeExpertOut;
    private readonly Tensor? _gpuEmbedding, _gpuOutputWeight, _gpuOutputNorm;
    private readonly bool _embIsQuantized;
    private readonly Tensor[] _gpuAttnNorm, _gpuWq, _gpuWk, _gpuWv, _gpuWo;
    private readonly Tensor[] _gpuFfnNorm, _gpuWGate, _gpuWUp, _gpuWDown;
    private readonly Tensor[]? _gpuWGateInp, _gpuWGateShexp, _gpuWUpShexp, _gpuWDownShexp;
    // Attention bias tensors for GPU layers (null when the model has no attention bias).
    private readonly Tensor[]? _gpuBq, _gpuBk, _gpuBv, _gpuBo;
    private readonly Tensor[]? _gpuQNorm, _gpuKNorm;
    private readonly Tensor[] _gpuKCache, _gpuVCache;
    private readonly Tensor[]? _gpuTqKCache, _gpuTqVCache, _gpuSignPatterns;
    private readonly Tensor? _gpuCodebook, _gpuBoundaries, _gpuRotatedQ, _gpuEvictK, _gpuEvictV;
    // Per-query-head softmax-scores scratch in VRAM, sized [numHeads × maxSeqLen]
    // (or a 1-float placeholder when _maxSeqLen ≤ 4096). Shared by both the TQ and
    // FP32 attention shaders when seq_len exceeds the shared-memory fast-path cap.
    private readonly Tensor _gpuAttnScoresScratch;
    // Flash-decoding split-KV (#235/#237/#238): partials for the split decode-attention path on
    // the GPU-resident attention layers. Same layout/gate as CudaForwardPass; null → per-head.
    private Tensor? _splitKvPartialO;
    private Tensor? _splitKvPartialMeta;
    private readonly bool _splitDecodeEnabled =
        Environment.GetEnvironmentVariable("SHARPI_SPLIT_DECODE") != "0";
    private readonly string? _splitGroupedMode =
        Environment.GetEnvironmentVariable("SHARPI_SPLIT_DECODE_GROUPED");
    // Hybrid split threshold (#238) — HIGHER than the dense 2048. Decode A/B (Coder-30B q8, 4070
    // Ti): split wins 1.49× @6K, 1.63× @8K, 2.08× @16K, but is only break-even at 4096 (1.03×) and
    // a small fast all-GPU MoE (OLMoE, 16 heads, ctx-capped at 4096) REGRESSES 2× there — attention
    // is a tiny share of its fast decode, so the split's fixed overhead dominates. seqLen > 4096
    // excludes OLMoE's whole range while keeping every confirmed Coder win.
    private const int HybridSplitMinSeq = 4096;
    private readonly Dictionary<nint, DType> _gpuWeightDTypes = new();

    // ── CPU resources (layers nGpuLayers..numLayers-1) ──
    private readonly float* _cpuHidden, _cpuResidual, _cpuNormBuf;
    private readonly float* _cpuQ, _cpuK, _cpuV, _cpuAttnOut;
    private readonly float* _cpuFfnGate, _cpuFfnUp;
    private readonly float* _cpuAttnScores;
    private readonly CpuWeightRef[] _cpuAttnNorm, _cpuWq, _cpuWk, _cpuWv, _cpuWo;
    private readonly CpuWeightRef[] _cpuFfnNorm, _cpuWGate, _cpuWUp, _cpuWDown;
    private readonly float*[] _cpuBq, _cpuBk, _cpuBv, _cpuBo;
    private readonly float*[] _cpuQNorm, _cpuKNorm;
    private readonly CpuWeightRef[]? _cpuWGateInp, _cpuWGateShexp, _cpuWUpShexp, _cpuWDownShexp;
    private readonly CpuWeightRef[]? _cpuWGateExps, _cpuWUpExps, _cpuWDownExps;
    // CPU-MoE mode: routed experts for the GPU-TRUNK layers (0.._nGpuLayers-1) run on the
    // CPU from mmap instead of streaming through the SLRU cache. Index i maps directly to
    // model layer i. Sized _nGpuLayers and populated only when _cpuMoe is true; null otherwise.
    private readonly CpuWeightRef[]? _cpuMoeGateInp, _cpuMoeGateExps, _cpuMoeUpExps, _cpuMoeDownExps;
    private readonly CpuWeightRef _cpuEmbedding, _cpuOutputWeight, _cpuOutputNorm;
    private readonly float* _cpuRouterLogits;
    private readonly float* _cpuSharedOut;
    private readonly float* _cpuExpertGate;
    private readonly float* _cpuExpertUp;
    // Dedicated MoE down-projection scratch sized embDim. Mirrors
    // ForwardPass._moeDownTemp; replaces the _cpuAttnOut reuse that breaks
    // whenever numHeads*headDim < embDim.
    private readonly float* _cpuMoeDownTemp;
    private readonly KvCache _cpuKvCache;
    private readonly TurboQuantKvCache? _cpuTqKvCache;
    private readonly float* _cpuRotatedQuery; // scratch for TQ query rotation [headDim]
    private readonly float* _cpuDecompBuf;    // scratch for TQ value decompress [headDim]
    private readonly Dictionary<string, nint> _cpuNormCache = new();

    // ── Shared ──
    private readonly Tensor _pinnedHidden; // host-visible buffer for GPU↔CPU transfer
    private readonly float[] _logitsBuf;
    private readonly float[]? _gpuRouterBuf;
    private readonly bool _hasAttnBias, _hasQkNorm, _isMoE, _hasSharedExpert;
    private readonly bool _tqEnabled;
    // GPU-resident KV-cache element dtype (#179/#230): fp32 (default), bf16 (½), or q8_0 (¼),
    // resolved from SHARPI_KV_DTYPE. Applies to the GPU-trunk layers only — CPU-offloaded layers
    // keep their own fp32 SimdKernels KV. Forced fp32 under TurboQuant (TQ owns its quantized ring).
    private readonly DType _kvDType;
    private readonly int _tqFp32Window;
    private readonly int _tqBlockBytes;
    private int _gpuTqCompressedLen;
    private int _gpuFp32WriteIdx;
    private int _gpuFp32Count;
    private int _kvLength;
    private readonly int _maxSeqLen;

    // Precomputed RoPE cos/sin tables for CPU layers [maxSeqLen * halfDim]
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;
    private readonly int _ropeHalfDim;

    // ── Gemma 4 plumbing ──────────────────────────────────────────────────
    // Mirrors CudaForwardPass and ForwardPass: per-layer head_dim variance, dual
    // RoPE (10K SWA / 1M global), KV-share alias dispatch, SWA-vs-full attention
    // split, post-attn/post-ffw norms, layer_output_scale, PLE injection, final
    // softcap, GeluTanh activation. _isGemma4Like is the master switch.
    private readonly bool _isGemma4Like;
    private readonly int _maxHeadDim;
    private readonly float _ropeThetaSwa;
    // SWA cos/sin tables — built at the SWA layers' RoPE dim (typically 256 → halfDim=128).
    private readonly float* _ropeCosTableSwa;
    private readonly float* _ropeSinTableSwa;
    private readonly int _ropeHalfDimSwa;
    // Per-layer global-RoPE frequency factors (rope_freqs.weight, size maxHeadDim/2);
    // CPU bakes them into _ropeCosTable/_ropeSinTable, GPU applies via RoPEWithFactors.
    private readonly float[]? _globalFreqFactors;
    private readonly Tensor? _gpuRopeFreqs;
    // KV-share layer set per tier — Append is skipped and reads route to the source
    // layer's pages. Mirrors CudaForwardPass._kvAliasedLayers. The constructor enforces
    // that all shared-KV source layers live on the CPU side so cross-tier reads never
    // happen (see Gemma4ValidateHybridSplit).
    private readonly HashSet<int> _gpuKvAliasedLayers = new();

    // Per-layer post-norms (Gemma 4) — same length as the tier's layer count.
    private readonly Tensor[]? _gpuPostAttnNorm, _gpuPostFfwNorm;
    private readonly CpuWeightRef[]? _cpuPostAttnNorm, _cpuPostFfwNorm;

    // Per-layer scalar gain applied AFTER the PLE injection (matches llama.cpp order).
    private readonly float[]? _layerOutputScale;

    // Gemma 4 PLE: table is CPU-resident (~4.2 GB at Q8_0 for E4B), per-layer
    // projections are uploaded to GPU once. BuildPerLayerProjectionsCpu computes the
    // 42×256 per-layer-slice array on CPU each token; CPU layers consume it directly,
    // GPU layers consume an uploaded copy.
    private readonly CpuWeightRef? _pleTokenEmbed;
    private readonly float[]? _perLayerModelProjF32;
    private readonly CpuWeightRef? _perLayerProjNorm;
    private readonly CpuWeightRef[]? _cpuInpGate, _cpuPleProj, _cpuPlePostNorm;
    private readonly Tensor[]? _gpuInpGate, _gpuPleProj, _gpuPlePostNorm;
    // Per-token PLE scratch (CPU).
    private readonly float* _pleRowBuf;       // [stackedDim = numLayers * pleWidth] f32
    private readonly float* _projPerLayer;    // [stackedDim] f32 — per-layer projection cache
    private readonly float* _pleX;            // [pleWidth] f32
    private readonly float* _pleY;            // [embDim]  f32
    // Per-token PLE scratch (GPU) — full projection uploaded once, viewed per layer.
    private readonly Tensor? _gpuProjPerLayerGpu;  // [nGpu*pleWidth] f32, one H2D/token
    private readonly Tensor[]? _gpuProjSliceViews; // [nGpu] static views into the above
    private readonly Tensor? _gpuPleX;        // [pleWidth] f32
    private readonly Tensor? _gpuPleY;        // [embDim] f32
    private readonly int _pleWidth;

    // ── Expert slot cache (MoE GPU layers, lazy/evictable expert loading) ──
    // Routed experts for GPU-tier MoE layers are streamed through this SLRU cache
    // (mirror of CudaHybridGdnForwardPass), rather than every expert being uploaded
    // resident. This lets non-GDN MoE models (Mixtral, Qwen3-30B-A3B, Qwen3-Coder)
    // run with more layers on the GPU than the full expert footprint would allow.
    // Null when not MoE or there are no GPU layers. Loads are synchronous on miss
    // (no prefetcher): the GDN path established this is fast enough for k=8 decode.
    private readonly CudaExpertSlotManager? _expertSlotManager;

    // CPU-MoE auto-select (mirror of CudaHybridGdnForwardPass). When the expert-cache
    // budget can hold fewer than ~half the GPU-trunk experts, routed-expert MoE runs on
    // the CPU (mmap) instead of streaming through the SLRU cache. Decided in the ctor
    // before the GPU upload loop. SHARPI_CPU_MOE=0|1 overrides. Always false for non-MoE.
    private readonly bool _cpuMoe;

    // ── Issue #123: batched-trunk prompt prefill ──────────────────────────────
    // Gate (default ON). When set, Prefill dispatches the supported configs to the
    // batched-trunk path (PrefillBatchedTrunk); otherwise the per-token Forward loop.
    internal static bool BatchedPrefillEnabled =
        Environment.GetEnvironmentVariable("SHARPI_BATCHED_PREFILL") != "0";

    // Issue #162: route the batched attention projections (q/k/v/o) through the
    // compute-bound path (Q8_0/Q4_K int8 MMQ, Q6_K/Q5_K dequant→fp16 GEMM, weight read
    // once per batch) instead of the per-token GEMM-N matvec re-stream. NOTE: only the
    // attention projections are batched in this path (#123 scope) — the FFN/MoE stage
    // stays per-token — so this helps the q/k/v/o matmuls only. Argmax-stable, NOT
    // byte-exact → the bit-parity oracle (CudaHybridBatchedPrefillTests) pins it OFF.
    // SHARPI_HYBRID_PREFILL_COMPUTE=0 reverts to the byte-exact per-token matvec.
    internal static bool HybridPrefillComputeEnabled =
        Environment.GetEnvironmentVariable("SHARPI_HYBRID_PREFILL_COMPUTE") != "0";

    // Test observable (issue #123): set by Prefill to whether the last call dispatched the
    // batched-trunk path (true) or fell back to the per-token loop (false). Lets the parity
    // oracles assert the batched path actually ran instead of passing vacuously when the
    // config is gated out (e.g. non-batchable dtype on a given box).
    internal bool LastPrefillWasBatched;

    // Test observable (issue #230): the resolved GPU KV-cache dtype. Lets the parity oracle
    // confirm --kv-type actually applied (the cache is genuinely narrowed) instead of passing
    // vacuously if the env plumbing regressed and KV silently stayed fp32 (fp32-vs-fp32 is
    // trivially argmax-stable).
    internal DType KvCacheDType => _kvDType;

    // Issue #218 test/bench hook: force the low-VRAM "fixed weights on CPU" config
    // (embedding + output table CPU-resident, _gpuEmbedding/_gpuOutputWeight null) even on a
    // box where ShouldKeepFixedWeightsOnCpu would keep them on the GPU. Lets the batched-CPU-
    // embed prefill path (#218) be exercised and A/B-benched on a model whose embedding fits in
    // VRAM. Read once in the constructor; OR'd into cpuEmbeddingOutputOnly. Default off.
    internal static bool ForceCpuResidentEmbedding =
        Environment.GetEnvironmentVariable("SHARPI_FORCE_CPU_EMBED") == "1";

    // Device scratch for the batched attention trunk, allocated lazily by
    // EnsureBatchedTrunkScratch(N) and sized to the largest N seen so far. The
    // residual-stream buffer [N × embDim] carries the hidden state across layers;
    // the q/k/v/attnOut/norm buffers are the batched analogues of the per-token
    // _gpuQ/_gpuK/_gpuV/_gpuAttnOut/_gpuNormBuf.
    private Tensor? _gpuBpStream;     // [N × embDim]
    private Tensor? _gpuBpNormAll;    // [N × embDim]  (attn-norm / ffn-norm output)
    private Tensor? _gpuBpQ;          // [N × numHeads*headDim]
    private Tensor? _gpuBpK;          // [N × numKvHeads*headDim]
    private Tensor? _gpuBpV;          // [N × numKvHeads*headDim]
    private Tensor? _gpuBpAttnOut;    // [N × numHeads*headDim]
    private Tensor? _gpuBpBlockOut;   // [N × embDim]  (o-proj out + post-attn residual)
    private int _bpCapN;              // current capacity (tokens) of the buffers above
    // Pinned host buffer for the batched GPU→CPU hidden transfer [N × embDim].
    private Tensor? _pinnedHiddenAll;
    private int _pinnedHiddenAllN;
    // Issue #218: dedicated pinned host buffer for the batched CPU→GPU embedding upload
    // [N × embDim], used only when the embedding is CPU-resident (_gpuEmbedding is null).
    // Distinct from _pinnedHidden/_pinnedHiddenAll so the embed staging never races the
    // per-token hidden transfer; the whole block is CPU-written then uploaded in one async
    // H2D, so there is no per-token write-after-read on a shared buffer.
    private Tensor? _pinnedEmbedAll;
    private int _pinnedEmbedAllN;

    // Non-transactional fault latch (mirror of CudaHybridGdnForwardPass._faulted):
    // set at batched-prefill entry, cleared only after the KV/position counters have
    // been advanced consistently. A mid-prefill throw leaves the KV cache partially
    // appended while _kvLength still reads the chunk start, so any retry on this pass
    // would run on inconsistent state. ThrowIfFaulted() blocks that.
    private bool _faulted;

    public int MaxSeqLen => _maxSeqLen;
    public LayerPlacement Placement => _placement;

    /// <summary>Vocabulary size of this model.</summary>
    public int VocabSize => _hp.VocabSize;

    /// <summary>
    /// Truncate the KV cache to the given length, discarding positions >= length.
    /// Used by speculative decoding to rewind rejected draft tokens.
    /// </summary>
    public void TruncateTo(int length)
    {
        _kvLength = length;
        _gpuTqCompressedLen = Math.Min(_gpuTqCompressedLen, length);
        if (_cpuTqKvCache != null)
            _cpuTqKvCache.TruncateTo(length);
        else
            _cpuKvCache.TruncateTo(length);
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => true;

    public CudaHybridForwardPass(GgufModel model, CudaBackend gpu, ModelHyperparams hp,
        LayerPlacement placement, bool enableTq = false, int tqFp32Window = 256, int tqBits = 3,
        int expertSlotCapacity = -1)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _placement = placement;
        _nGpuLayers = placement.GpuLayers;
        _nCpuLayers = placement.CpuLayers;
        _maxSeqLen = placement.RecommendedCtxSize;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;
        _expertDim = hp.IsMoE ? hp.ExpertIntermediateDim : hp.IntermediateDim;
        _hasAttnBias = hp.HasAttnBias;
        _hasQkNorm = hp.HasQkNorm;
        _isMoE = hp.IsMoE;
        _hasSharedExpert = hp.HasSharedExpert;

        // Gemma 4 / per-layer head_dim path. _maxHeadDim sizes the Q/K/V/attnOut
        // scratch buffers and the CPU KV cache row width; per-layer views carve out
        // the active head_dim each layer. _ropeThetaSwa drives the SWA RoPE table.
        _isGemma4Like = hp.LayerHeadDim is not null;
        _maxHeadDim = _headDim;
        if (hp.LayerHeadDim is { } lhdMax)
            for (int i = 0; i < hp.NumLayers; i++)
                if (lhdMax[i] > _maxHeadDim) _maxHeadDim = lhdMax[i];
        _ropeThetaSwa = hp.RopeThetaSwa;
        if (_isGemma4Like && _isMoE)
            throw new NotSupportedException(
                "CudaHybridForwardPass does not yet support MoE Gemma-style models. " +
                "(The dense Gemma 4 path is supported; an MoE variant would need separate plumbing.)");
        if (_isGemma4Like && enableTq)
            throw new NotSupportedException(
                "CudaHybridForwardPass + TurboQuant is not supported for per-layer head_dim " +
                "architectures (e.g. Gemma 4). Disable --tq.");
        if (_isGemma4Like)
            Gemma4ValidateHybridSplit(hp, _nGpuLayers);

        bool cpuEmbeddingOutputOnly = ForceCpuResidentEmbedding || ShouldKeepFixedWeightsOnCpu(
            model.FindTensor("token_embd.weight")!.Value,
            model.FindTensor("output.weight"));
        _tqEnabled = enableTq;
        if (_tqEnabled && _headDim is not 128 and not 256)
            throw new NotSupportedException($"TurboQuant currently supports head dimensions 128 and 256; model head dim is {_headDim}.");

        // KV-cache dtype (#230): honor SHARPI_KV_DTYPE on this layer-split hybrid path — it was
        // silently ignored (KV always fp32), so --kv-type q8_0 was a no-op even though TierPlanner
        // already prices the KV budget at the narrowed dtype (RunCommand passes ResolveConfiguredKvDType
        // to TierPlanner.Plan). That left a plan/runtime mismatch: the planner reserved a smaller KV
        // and a larger expert-cache budget while the runtime allocated the full fp32 cache. Narrowing
        // applies to the GPU-trunk KV only; CPU layers keep their fp32 store. Reuses the dense path's
        // narrowed append/attention kernels (#179) via the *Kv dispatch helpers below.
        DType requestedKv = CudaForwardPass.ResolveConfiguredKvDType();
        if (_tqEnabled && requestedKv != DType.Float32)
            throw new NotSupportedException(
                $"SHARPI_KV_DTYPE={requestedKv} + TurboQuant is not supported (TQ owns the KV quantization). " +
                "Use one or the other.");
        _kvDType = _tqEnabled ? DType.Float32 : requestedKv;
        // Only the GPU-resident layers carry the narrowed cache (CPU-offloaded layers keep fp32),
        // so scope the q8 geometry check to them — checking all layers would falsely reject q8
        // when only a CPU-tail layer is incompatible (and trivially passes when _nGpuLayers == 0).
        if (_kvDType == DType.Q8_0 && !CudaForwardPass.Q8KvGeometrySupported(hp, _nGpuLayers))
            throw new NotSupportedException(
                "SHARPI_KV_DTYPE=q8_0 requires every GPU-resident layer's kvDim (kvHeads × headDim) to be a " +
                "multiple of 32 (the q8_0 block size); this model's geometry is incompatible. Use bf16 or fp32.");

        _tqFp32Window = enableTq ? Math.Min(tqFp32Window, _maxSeqLen) : 0;
        _tqBlockBytes = enableTq ? TurboQuantOps.BlockSize(tqBits, _headDim) : 0;
        _gpuRouterBuf = _isMoE && _nGpuLayers > 0 ? new float[hp.NumExperts] : null;

        // ── Auto-detect CPU-MoE vs GPU SLRU MoE ─────────────────────────────────
        // Mirrors CudaHybridGdnForwardPass: SLRU only pays off when most experts fit in
        // VRAM. Predict the slot count from the planner's expert-cache budget (VRAM left
        // after trunk + KV) before uploading any MoE weights. Route MoE through CPU when
        // fewer than ~half of all GPU-trunk experts can be cached on the GPU.
        if (_isMoE)
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
                long perExpert = PerExpertBytes();
                long budget = placement.ExpertCacheBudgetBytes;
                int predictedSlots = perExpert > 0 ? (int)(budget / perExpert) : 0;
                int totalExperts = _nGpuLayers * hp.NumExperts;
                double ratio = totalExperts > 0 ? (double)predictedSlots / totalExperts : 1.0;
                _cpuMoe = ratio < 0.5;
                Console.Error.WriteLine(
                    $"[CudaHybridForwardPass] MoE auto-select: expert-cache capacity ≈ {predictedSlots}/{totalExperts} ({ratio:P0}) → {(_cpuMoe ? "CPU" : "GPU SLRU")} MoE.  Override with SHARPI_CPU_MOE=0|1.");
            }
        }
        else
        {
            _cpuMoe = false;
        }

        string kvTag = _kvDType switch { DType.BFloat16 => " [KV bf16]", DType.Q8_0 => " [KV q8_0]", _ => "" };
        Console.Error.WriteLine($"[HybridForwardPass] {placement.Summary()}{(enableTq ? $" [TQ{tqBits}]" : "")}{kvTag}");

        bool vramTrace = Environment.GetEnvironmentVariable("SHARPI_TRACE_VRAM") == "1";
        void TraceVram(string label)
        {
            if (vramTrace)
                Console.Error.WriteLine($"[VRAM] {label}: free={gpu.FreeVramBytes / (1024 * 1024)} MiB");
        }
        TraceVram("constructor entry");

        // ── Allocate GPU scratch buffers ──
        // Q/K/V/attnOut are sized to _maxHeadDim so per-layer view tensors can carve
        // out the active head_dim on Gemma 4 (256 SWA / 512 global).
        _gpuHidden = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuResidual = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuNormBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _gpuQ = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
        _gpuK = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _gpuV = gpu.Allocate(TensorShape.D1((long)_numKvHeads * _maxHeadDim));
        _gpuAttnOut = gpu.Allocate(TensorShape.D1((long)_numHeads * _maxHeadDim));
        // For MoE the dense FFN path is unreachable, so size scratch tightly to the
        // expert intermediate dim. Vulkan tolerates the over-allocation via
        // robustBufferAccess (OOB reads return 0); CUDA's Q4_K matvec doesn't —
        // it derives `cols` from input.ElementCount and would walk past the end of
        // expert weight tensors with `cols=max(intermDim, expertDim)`.
        int gpuFfnScratch = _isMoE ? _expertDim : _intermDim;
        _gpuFfnGate = gpu.Allocate(TensorShape.D1(gpuFfnScratch));
        _gpuFfnUp = gpu.Allocate(TensorShape.D1(gpuFfnScratch));
        _gpuLogits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _gpuRouterLogits = _isMoE && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(hp.NumExperts)) : null;
        _gpuMoeSharedOut = _isMoE && _hasSharedExpert && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _gpuMoeExpertOut = _isMoE && _nGpuLayers > 0 ? gpu.Allocate(TensorShape.D1(_embDim)) : null;
        _logitsBuf = new float[hp.VocabSize];

        // Pinned buffer for hidden state transfer (embDim floats)
        _pinnedHidden = gpu.AllocatePinned(TensorShape.D1(_embDim));

        _cpuEmbedding = ResolveCpuWeight("token_embd.weight");
        _cpuOutputNorm = ResolveCpuWeight("output_norm.weight");
        _cpuOutputWeight = model.FindTensor("output.weight") is not null
            ? ResolveCpuWeight("output.weight")
            : _cpuEmbedding;

        // ── Upload GPU weights (embedding + output + first N layers) ──
        if (!cpuEmbeddingOutputOnly)
        {
            _gpuEmbedding = UploadEmbeddingWeight("token_embd.weight", out _embIsQuantized);
        }
        else
        {
            _gpuEmbedding = null;
            _embIsQuantized = false;
        }

        if (!cpuEmbeddingOutputOnly)
        {
            _gpuOutputNorm = UploadWeight("output_norm.weight");
            _gpuOutputWeight = model.FindTensor("output.weight") is not null
                ? UploadWeight("output.weight")
                : _gpuEmbedding;
        }
        else
        {
            _gpuOutputNorm = null;
            _gpuOutputWeight = null;
        }

        _gpuAttnNorm = new Tensor[_nGpuLayers];
        _gpuWq = new Tensor[_nGpuLayers]; _gpuWk = new Tensor[_nGpuLayers];
        _gpuWv = new Tensor[_nGpuLayers]; _gpuWo = new Tensor[_nGpuLayers];
        _gpuFfnNorm = new Tensor[_nGpuLayers];
        _gpuWGate = new Tensor[_nGpuLayers]; _gpuWUp = new Tensor[_nGpuLayers]; _gpuWDown = new Tensor[_nGpuLayers];
        _gpuWGateInp = _isMoE ? new Tensor[_nGpuLayers] : null;
        _gpuWGateShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;
        _gpuWUpShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;
        _gpuWDownShexp = _isMoE && _hasSharedExpert ? new Tensor[_nGpuLayers] : null;

        if (_hasAttnBias) { _gpuBq = new Tensor[_nGpuLayers]; _gpuBk = new Tensor[_nGpuLayers]; _gpuBv = new Tensor[_nGpuLayers]; _gpuBo = new Tensor[_nGpuLayers]; }
        if (_hasQkNorm) { _gpuQNorm = new Tensor[_nGpuLayers]; _gpuKNorm = new Tensor[_nGpuLayers]; }
        // Gemma 4 per-layer post-norms (sized to GPU layer count; null for non-gemma4).
        if (_isGemma4Like && hp.HasPostAttnNorm) _gpuPostAttnNorm = new Tensor[_nGpuLayers];
        if (_isGemma4Like && hp.HasPostFfwNorm)  _gpuPostFfwNorm  = new Tensor[_nGpuLayers];

        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[_nGpuLayers];
        _gpuVCache = new Tensor[_nGpuLayers];
        long tqUintsPerLayer = 0;
        if (_tqEnabled)
        {
            int maxTqPositions = Math.Max(0, _maxSeqLen - _tqFp32Window);
            long tqBytesPerPos = (long)_numKvHeads * _tqBlockBytes;
            tqUintsPerLayer = (maxTqPositions * tqBytesPerPos + 3) / 4;
            _gpuTqKCache = new Tensor[_nGpuLayers];
            _gpuTqVCache = new Tensor[_nGpuLayers];
            _gpuSignPatterns = new Tensor[_nGpuLayers];
            var centroids = TurboQuantCodebooks.GetCentroids(tqBits, _headDim).ToArray();
            _gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));
            var boundaries = TurboQuantCodebooks.GetBoundaries(tqBits, _headDim).ToArray();
            _gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));
            _gpuRotatedQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
            _gpuEvictK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
            _gpuEvictV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        }

        // Both Vulkan attention shaders spill softmax scores to VRAM when seq_len
        // exceeds the 4096-slot shared-memory fast path; a 1-float placeholder is
        // enough for shorter contexts but the descriptor must always be bound.
        {
            long scratchElems = _maxSeqLen > 4096 ? (long)_numHeads * _maxSeqLen : 1L;
            _gpuAttnScoresScratch = gpu.Allocate(TensorShape.D1(scratchElems));
        }

        // Flash-decoding split-KV partials (#238) — allocate when the context is long enough to
        // benefit and within the combine kernel's split bound. Mirrors CudaForwardPass; skipped
        // under TurboQuant (TQ decode uses its own TqAttention path, never the split kernels).
        if (_splitDecodeEnabled && !_tqEnabled
            && _maxSeqLen > HybridSplitMinSeq && _maxSeqLen <= CudaForwardPass.SplitKvMaxCtx)
        {
            long nSplitsMax = (_maxSeqLen + CudaBackend.SplitKvChunk - 1) / CudaBackend.SplitKvChunk;
            _splitKvPartialO = gpu.Allocate(TensorShape.D1((long)_numHeads * nSplitsMax * _maxHeadDim));
            _splitKvPartialMeta = gpu.Allocate(TensorShape.D1((long)_numHeads * nSplitsMax * 2));
        }

        TraceVram("before per-layer weight upload");
        Console.Error.Write($"[HybridForwardPass] Uploading {_nGpuLayers} GPU layers...");
        for (int i = 0; i < _nGpuLayers; i++)
        {
            bool kvShared = _isGemma4Like && hp.KvSourceLayer is { } ksl && ksl[i] >= 0;
            // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the
            // raw K projection, so there is no attn_v.weight (or bias) to upload. The
            // global layers are exactly the non-SWA ones; KvSourceLayer is null on the 12B
            // so kEqV and kvShared never overlap. Mirrors CudaForwardPass.RunGemma4DeviceRegion.
            bool kEqV = hp.AttentionKEqV && hp.IsSwaLayer is { } swaUp && !swaUp[i];
            _gpuAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _gpuWq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            // KV-share layers (Gemma 4 tail) carry no attn_k/attn_v of their own.
            // Validated above that all such layers live on the CPU side, so we never
            // hit this branch in the GPU loop — assert with a helpful message.
            if (kvShared)
                throw new InvalidOperationException(
                    $"GPU layer {i} is KV-shared with source layer {hp.KvSourceLayer![i]}; " +
                    "Gemma4ValidateHybridSplit should have rejected this configuration.");
            _gpuWk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            // k_eq_v global layers leave _gpuWv[i] as default(Tensor); GpuLayerGemma4
            // copies the raw K projection into V instead (and Dispose's Free is a no-op
            // on the default handle).
            if (!kEqV)
                _gpuWv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _gpuWo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _gpuFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            if (_gpuPostAttnNorm is not null)
                _gpuPostAttnNorm[i] = UploadWeight($"blk.{i}.post_attention_norm.weight");
            if (_gpuPostFfwNorm is not null)
                _gpuPostFfwNorm[i]  = UploadWeight($"blk.{i}.post_ffw_norm.weight");
            if (_isMoE)
            {
                // CPU-MoE: the router (ffn_gate_inp) runs on the CPU in the compute-half,
                // so leave _gpuWGateInp[i] default/null here. GPU-SLRU mode uploads it.
                if (!_cpuMoe)
                    _gpuWGateInp![i]  = UploadWeight($"blk.{i}.ffn_gate_inp.weight");
                // Routed experts are NOT uploaded here — they stream through the
                // CudaExpertSlotManager SLRU cache (created after this loop). The router
                // and shared expert stay resident since they run on every token.
                if (_hasSharedExpert)
                {
                    _gpuWGateShexp![i] = UploadWeight($"blk.{i}.ffn_gate_shexp.weight");
                    _gpuWUpShexp![i] = UploadWeight($"blk.{i}.ffn_up_shexp.weight");
                    _gpuWDownShexp![i] = UploadWeight($"blk.{i}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _gpuWGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
                _gpuWUp[i] = UploadWeight($"blk.{i}.ffn_up.weight");
                _gpuWDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");
            }

            if (_hasAttnBias)
            {
                _gpuBq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _gpuBk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                if (!kEqV)
                    _gpuBv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                _gpuBo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpuQNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _gpuKNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            if (_tqEnabled)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuTqKCache![i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuTqVCache![i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuSignPatterns![i] = UploadTqSignPatterns(i);
            }
            else
            {
                // Gemma 4: each GPU layer sizes its KV cache by per-layer head_dim and
                // (for SWA layers) caps at SlidingWindowSize. Non-gemma4 stays at the
                // model-wide head_dim × full context.
                //
                // This path is DECODE-ONLY for Gemma 4 (IsBatchedPrefillSupported requires
                // !_isGemma4Like), so the SWA ring only needs to hold `window` positions:
                // each token appends one K/V then attends its window before the next append,
                // so the bare-window ring is never overwritten early. The shared kernels'
                // `pos % max_seq_len` (max_seq_len == this layerCtx) wraps writes/reads into
                // it — which also fixes the pre-#162 latent OOB here (positions ≥ window used
                // to index a window-sized cache absolutely). It deliberately does NOT use
                // CudaForwardPass.SwaRingSize (the window+chunk headroom) because that's only
                // needed by the batched-append (chunked-prefill) path, which never runs here.
                int layerHd = _isGemma4Like ? hp.LayerHeadDim![i] : _headDim;
                // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on the global
                // k_eq_v layers). Sizing the cache tight reclaims VRAM — the whole point of
                // the hybrid (<12 GB) path — and matches CudaForwardPass's per-layer cache.
                int layerKv = hp.LayerKvHeads is { } lkv ? lkv[i] : _numKvHeads;
                int layerKvDim = layerKv * layerHd;
                int layerCtx = (_isGemma4Like && hp.IsSwaLayer is { } swa && swa[i] && hp.SlidingWindowSize > 0)
                    ? Math.Min(_maxSeqLen, hp.SlidingWindowSize)
                    : _maxSeqLen;
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim), _kvDType);
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)layerCtx * layerKvDim), _kvDType);
            }
            Console.Error.Write(".");
        }
        Console.Error.WriteLine(" done.");
        TraceVram("after all weight uploads");

        // MoE GPU layers stream routed experts through an SLRU cache. Size it from
        // the *actual* free VRAM remaining now that attention weights, KV cache and
        // scratch are uploaded (cudaMemGetInfo via FreeVramBytes), capped at the full
        // GPU-layer expert count. Capping at totalGpuExperts means the cache can never
        // hold more than the old eager path did — which TierPlanner already verified
        // fits — so there is no new OOM risk; the budget term only *shrinks* capacity
        // when VRAM is tight (e.g. the user forced extra GPU layers via -g), enabling
        // streaming instead of an OOM.
        if (_isMoE && _nGpuLayers > 0 && _cpuMoe)
        {
            Console.Error.WriteLine(
                "[CudaHybridForwardPass] CPU-MoE mode: routed experts for the GPU-trunk layers " +
                "run on the CPU from mmap; no SLRU expert cache is allocated.");
        }
        if (_isMoE && _nGpuLayers > 0 && !_cpuMoe)
        {
            long perExpert = PerExpertBytes();
            long reserve = 512L << 20; // 512 MiB headroom for transient per-GEMM scratch
            long free = (long)gpu.FreeVramBytes;
            var plan = MoeCacheSizing.Plan(_nGpuLayers, hp.NumExperts, hp.NumActiveExperts,
                free, perExpert, reserve);
            int totalGpuExperts = _nGpuLayers * hp.NumExperts;
            Console.Error.WriteLine(
                $"[CudaHybridForwardPass] SLRU expert cache: {plan.Slots} slots / {totalGpuExperts} total " +
                $"({hp.NumExperts} experts × {_nGpuLayers} GPU layers, per-expert ≈ {perExpert / 1024} KiB, " +
                $"free VRAM ≈ {free / (1024 * 1024)} MiB).");
            switch (plan.Status)
            {
                case MoeCacheSizingStatus.BudgetExhausted:
                    // Budget couldn't fit even one expert; capacity was clamped to 1.
                    // Decode will thrash (~every routed expert misses); louder than the
                    // BelowRecommended warning because the perf hit is catastrophic.
                    Console.Error.WriteLine(
                        "[CudaHybridForwardPass] WARNING: free VRAM cannot fit a single expert; " +
                        "cache clamped to 1 slot. Every routed expert will miss and stream from CPU. " +
                        "Reduce -g or use --backend vulkan.");
                    break;
                case MoeCacheSizingStatus.UnknownExpertSize:
                    Console.Error.WriteLine(
                        "[CudaHybridForwardPass] WARNING: could not measure per-expert size " +
                        "(missing blk.0.ffn_*_exps tensor); cache fell back to total. " +
                        "Will fail at runtime if total VRAM is exceeded.");
                    break;
                case MoeCacheSizingStatus.BelowRecommended:
                    int pct = plan.RecommendedSlots > 0 ? plan.Slots * 100 / plan.RecommendedSlots : 0;
                    Console.Error.WriteLine(
                        $"[CudaHybridForwardPass] WARNING: cache ({plan.Slots}) is {pct}% of the " +
                        $"routing-locality recommendation (~{plan.RecommendedSlots} = 2× active per layer); " +
                        $"expert hit rate may suffer. Fewer GPU layers (-g) or more VRAM would help.");
                    break;
            }
            _expertSlotManager = new CudaExpertSlotManager(gpu, model, hp, plan.Slots, _gpuWeightDTypes);
        }

        // CPU-MoE: resolve the GPU-trunk layers' routed-expert weights as CPU mmap views.
        // These layers (0.._nGpuLayers-1) keep attention + shared expert on the GPU but run
        // the router + routed FFN on the CPU. Index i maps directly to model layer i.
        if (_isMoE && _cpuMoe)
        {
            _cpuMoeGateInp  = new CpuWeightRef[_nGpuLayers];
            _cpuMoeGateExps = new CpuWeightRef[_nGpuLayers];
            _cpuMoeUpExps   = new CpuWeightRef[_nGpuLayers];
            _cpuMoeDownExps = new CpuWeightRef[_nGpuLayers];
            for (int i = 0; i < _nGpuLayers; i++)
            {
                _cpuMoeGateInp[i]  = ResolveCpuWeight($"blk.{i}.ffn_gate_inp.weight");
                _cpuMoeGateExps[i] = ResolveCpuWeight($"blk.{i}.ffn_gate_exps.weight");
                _cpuMoeUpExps[i]   = ResolveCpuWeight($"blk.{i}.ffn_up_exps.weight");
                _cpuMoeDownExps[i] = ResolveCpuWeight($"blk.{i}.ffn_down_exps.weight");
            }
        }

        // ── Resolve CPU weights (layers nGpuLayers..numLayers-1) ──
        // Per-layer head_dim path widens Q/K/V/attnOut scratch to _maxHeadDim so the
        // per-layer views can carve out the active head_dim each layer.
        _cpuHidden = Alloc(_embDim);
        _cpuResidual = Alloc(_embDim);
        _cpuNormBuf = Alloc(_embDim);
        _cpuQ = Alloc(_numHeads * _maxHeadDim);
        _cpuK = Alloc(_numKvHeads * _maxHeadDim);
        _cpuV = Alloc(_numKvHeads * _maxHeadDim);
        _cpuAttnOut = Alloc(_numHeads * _maxHeadDim);
        _cpuFfnGate = Alloc(_intermDim);
        _cpuFfnUp = Alloc(_intermDim);
        _cpuAttnScores = Alloc(_numHeads * _maxSeqLen);
        _cpuRouterLogits = _isMoE ? Alloc(hp.NumExperts) : null;
        _cpuSharedOut = _isMoE && _hasSharedExpert ? Alloc(_embDim) : null;
        _cpuExpertGate = _isMoE ? Alloc(_expertDim) : null;
        _cpuExpertUp = _isMoE ? Alloc(_expertDim) : null;
        _cpuMoeDownTemp = _isMoE ? Alloc(_embDim) : null;

        // Precompute RoPE cos/sin tables for CPU layers. Gemma 4 builds two: the
        // primary table at the (max) head_dim / RopeTheta (1M), the SWA table at the
        // SWA layers' head_dim / RopeThetaSwa (10K). When `rope_freqs.weight` exists
        // it bakes into the primary table so ApplyRoPECachedNeox matches llama.cpp's
        // RoPEWithFactors path for the global layers.
        _ropeHalfDim = _maxHeadDim / 2;
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDim * sizeof(float)));
        if (_isGemma4Like
            && model.FindTensor("rope_freqs.weight") is GgufTensorInfo rfInfo
            && rfInfo.DType == DType.Float32
            && rfInfo.ElementCount == _maxHeadDim / 2)
        {
            var rfData = model.GetTensorData(rfInfo);
            _globalFreqFactors = new float[_maxHeadDim / 2];
            MemoryMarshal.Cast<byte, float>(rfData).Slice(0, _globalFreqFactors.Length)
                .CopyTo(_globalFreqFactors);
            fixed (float* ff = _globalFreqFactors)
                SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable,
                    _maxSeqLen, _maxHeadDim, hp.RopeTheta, ff);
        }
        else
        {
            SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, _maxSeqLen, _maxHeadDim, hp.RopeTheta);
        }
        if (_isGemma4Like)
        {
            // SWA table — sized to the smallest SWA-layer head_dim (256 on E4B).
            int swaHd = _maxHeadDim;
            if (hp.LayerHeadDim is { } lhdSwa && hp.IsSwaLayer is { } swaMask)
                for (int li = 0; li < hp.NumLayers; li++)
                    if (swaMask[li]) swaHd = Math.Min(swaHd, lhdSwa[li]);
            _ropeHalfDimSwa = swaHd / 2;
            _ropeCosTableSwa = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDimSwa * sizeof(float)));
            _ropeSinTableSwa = (float*)NativeMemory.Alloc((nuint)((long)_maxSeqLen * _ropeHalfDimSwa * sizeof(float)));
            SimdKernels.BuildRopeTable(_ropeCosTableSwa, _ropeSinTableSwa, _maxSeqLen, swaHd,
                _ropeThetaSwa > 0f ? _ropeThetaSwa : hp.RopeTheta);
        }

        _cpuAttnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWq = new CpuWeightRef[_nCpuLayers]; _cpuWk = new CpuWeightRef[_nCpuLayers];
        _cpuWv = new CpuWeightRef[_nCpuLayers]; _cpuWo = new CpuWeightRef[_nCpuLayers];
        _cpuFfnNorm = new CpuWeightRef[_nCpuLayers];
        _cpuWGate = new CpuWeightRef[_nCpuLayers]; _cpuWUp = new CpuWeightRef[_nCpuLayers]; _cpuWDown = new CpuWeightRef[_nCpuLayers];
        _cpuBq = new float*[_nCpuLayers]; _cpuBk = new float*[_nCpuLayers];
        _cpuBv = new float*[_nCpuLayers]; _cpuBo = new float*[_nCpuLayers];
        _cpuQNorm = new float*[_nCpuLayers]; _cpuKNorm = new float*[_nCpuLayers];
        if (_isMoE)
        {
            _cpuWGateInp = new CpuWeightRef[_nCpuLayers];
            _cpuWGateExps = new CpuWeightRef[_nCpuLayers];
            _cpuWUpExps = new CpuWeightRef[_nCpuLayers];
            _cpuWDownExps = new CpuWeightRef[_nCpuLayers];
            if (_hasSharedExpert)
            {
                _cpuWGateShexp = new CpuWeightRef[_nCpuLayers];
                _cpuWUpShexp = new CpuWeightRef[_nCpuLayers];
                _cpuWDownShexp = new CpuWeightRef[_nCpuLayers];
            }
        }
        // Gemma 4 per-layer Q/K-norm allocations need to handle KV-share layers where
        // attn_k_norm is absent (the source layer's K is already normed). The arrays
        // are still indexed by ci so non-shared layers can populate them; shared
        // layers leave the K-norm slot null.
        if (_isGemma4Like && hp.HasPostAttnNorm) _cpuPostAttnNorm = new CpuWeightRef[_nCpuLayers];
        if (_isGemma4Like && hp.HasPostFfwNorm)  _cpuPostFfwNorm  = new CpuWeightRef[_nCpuLayers];

        for (int ci = 0; ci < _nCpuLayers; ci++)
        {
            int li = ci + _nGpuLayers; // actual layer index
            bool kvShared = _isGemma4Like && hp.KvSourceLayer is { } ksl && ksl[li] >= 0;
            // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the raw
            // K projection. Leave _cpuWv[ci] as default(CpuWeightRef) (DataPtr == null) — the
            // detector CpuLayerGemma4 uses to copy K→V. Per-layer KV head count too.
            bool kEqV = hp.AttentionKEqV && hp.IsSwaLayer is { } swaCpu && !swaCpu[li];
            int layerKvCpu = hp.LayerKvHeads is { } lkvCpu ? lkvCpu[li] : _numKvHeads;
            _cpuAttnNorm[ci] = ResolveCpuWeight($"blk.{li}.attn_norm.weight");
            _cpuWq[ci] = ResolveCpuWeight($"blk.{li}.attn_q.weight");
            if (!kvShared)
            {
                _cpuWk[ci] = ResolveCpuWeight($"blk.{li}.attn_k.weight");
                if (!kEqV)
                    _cpuWv[ci] = ResolveCpuWeight($"blk.{li}.attn_v.weight");
            }
            _cpuWo[ci] = ResolveCpuWeight($"blk.{li}.attn_output.weight");
            _cpuFfnNorm[ci] = ResolveCpuWeight($"blk.{li}.ffn_norm.weight");
            if (_cpuPostAttnNorm is not null)
                _cpuPostAttnNorm[ci] = ResolveCpuWeight($"blk.{li}.post_attention_norm.weight");
            if (_cpuPostFfwNorm is not null)
                _cpuPostFfwNorm[ci] = ResolveCpuWeight($"blk.{li}.post_ffw_norm.weight");
            if (_isMoE)
            {
                _cpuWGateInp![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_inp.weight");
                _cpuWGateExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_exps.weight");
                _cpuWUpExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_up_exps.weight");
                _cpuWDownExps![ci] = ResolveCpuWeight($"blk.{li}.ffn_down_exps.weight");
                if (_hasSharedExpert)
                {
                    _cpuWGateShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_gate_shexp.weight");
                    _cpuWUpShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_up_shexp.weight");
                    _cpuWDownShexp![ci] = ResolveCpuWeight($"blk.{li}.ffn_down_shexp.weight");
                }
            }
            else
            {
                _cpuWGate[ci] = ResolveCpuWeight($"blk.{li}.ffn_gate.weight");
                _cpuWUp[ci] = ResolveCpuWeight($"blk.{li}.ffn_up.weight");
                _cpuWDown[ci] = ResolveCpuWeight($"blk.{li}.ffn_down.weight");
            }

            // Gemma 4 has per-layer head_dim (256 SWA / 512 global); other arches
            // pin to scalar _headDim. Mirrors CudaForwardPass per-layer dispatch.
            int layerHd = _hp.LayerHeadDim is { } lhd ? lhd[li] : _headDim;
            if (_hasAttnBias && !kvShared)
            {
                _cpuBq[ci] = LoadCpuBias($"blk.{li}.attn_q.bias", _numHeads * layerHd);
                _cpuBk[ci] = LoadCpuBias($"blk.{li}.attn_k.bias", layerKvCpu * layerHd);
                if (!kEqV)
                    _cpuBv[ci] = LoadCpuBias($"blk.{li}.attn_v.bias", layerKvCpu * layerHd);
                _cpuBo[ci] = LoadCpuBias($"blk.{li}.attn_output.bias", _embDim);
            }
            else if (_hasAttnBias)
            {
                _cpuBq[ci] = LoadCpuBias($"blk.{li}.attn_q.bias", _numHeads * layerHd);
                _cpuBo[ci] = LoadCpuBias($"blk.{li}.attn_output.bias", _embDim);
            }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                int qNormSize = _hp.IsPerChannelQkNorm ? (int)_numHeads * layerHd : layerHd;
                _cpuQNorm[ci] = LoadCpuBias($"blk.{li}.attn_q_norm.weight", qNormSize);
                if (!kvShared)
                {
                    int kNormSize = _hp.IsPerChannelQkNorm ? (int)_numKvHeads * layerHd : layerHd;
                    _cpuKNorm[ci] = LoadCpuBias($"blk.{li}.attn_k_norm.weight", kNormSize);
                }
            }
        }

        // CPU KV cache row width must accommodate the widest per-layer head_dim on
        // Gemma 4 (zero-padded tail per layer; mirror of ForwardPass). For non-gemma4
        // _maxHeadDim == _headDim so the row width is unchanged.
        _cpuKvCache = new KvCache(_nCpuLayers, _maxSeqLen, _numKvHeads, _maxHeadDim);

        // Per-layer scalar output_scale (Gemma 4). Loaded for all layers (GPU+CPU)
        // because the GPU half also reads it per layer in GpuLayerGemma4.
        if (_isGemma4Like && hp.HasLayerOutputScale)
        {
            _layerOutputScale = new float[hp.NumLayers];
            for (int li = 0; li < hp.NumLayers; li++)
                _layerOutputScale[li] = LoadScalarF32($"blk.{li}.layer_output_scale.weight");
        }

        // PLE — CPU-resident table, projections uploaded to GPU and resolved on CPU
        // for the CPU half. BuildPerLayerProjectionsCpu runs once per token across
        // both tiers.
        _pleWidth = (_isGemma4Like && hp.HasPerLayerTokenEmbd) ? hp.PerLayerEmbeddingWidth : 0;
        if (_isGemma4Like && hp.HasPerLayerTokenEmbd)
        {
            _pleTokenEmbed = ResolveCpuWeight("per_layer_token_embd.weight");
            _perLayerProjNorm = ResolveCpuWeight("per_layer_proj_norm.weight");

            var projInfo = model.FindTensor("per_layer_model_proj.weight")
                ?? throw new InvalidOperationException("Missing per_layer_model_proj.weight");
            var projData = model.GetTensorData(projInfo);
            int projCount = (int)projInfo.ElementCount;
            _perLayerModelProjF32 = new float[projCount];
            if (projInfo.DType == DType.Float32)
                MemoryMarshal.Cast<byte, float>(projData).Slice(0, projCount).CopyTo(_perLayerModelProjF32);
            else
                Dequantize.ToFloat32(projData, _perLayerModelProjF32.AsSpan(), projInfo.DType, projCount);

            _cpuInpGate     = new CpuWeightRef[hp.NumLayers];
            _cpuPleProj     = new CpuWeightRef[hp.NumLayers];
            _cpuPlePostNorm = new CpuWeightRef[hp.NumLayers];
            for (int li = 0; li < hp.NumLayers; li++)
            {
                _cpuInpGate[li]     = ResolveCpuWeight($"blk.{li}.inp_gate.weight");
                _cpuPleProj[li]     = ResolveCpuWeight($"blk.{li}.proj.weight");
                _cpuPlePostNorm[li] = ResolveCpuWeight($"blk.{li}.post_norm.weight");
            }

            int stackedDim = hp.NumLayers * _pleWidth;
            _pleRowBuf    = Alloc(stackedDim);
            _projPerLayer = Alloc(stackedDim);
            _pleX         = Alloc(_pleWidth);
            _pleY         = Alloc(_embDim);

            // GPU-side PLE weights for the GPU half (per-layer for layers 0..nGpu-1).
            if (_nGpuLayers > 0)
            {
                _gpuInpGate     = new Tensor[_nGpuLayers];
                _gpuPleProj     = new Tensor[_nGpuLayers];
                _gpuPlePostNorm = new Tensor[_nGpuLayers];
                for (int i = 0; i < _nGpuLayers; i++)
                {
                    _gpuInpGate[i]     = UploadWeight($"blk.{i}.inp_gate.weight");
                    _gpuPleProj[i]     = UploadWeight($"blk.{i}.proj.weight");
                    _gpuPlePostNorm[i] = UploadWeight($"blk.{i}.post_norm.weight");
                }
                _gpuProjPerLayerGpu = gpu.Allocate(TensorShape.D1(_nGpuLayers * _pleWidth));
                _gpuProjSliceViews  = new Tensor[_nGpuLayers];
                for (int i = 0; i < _nGpuLayers; i++)
                    _gpuProjSliceViews[i] = gpu.View(_gpuProjPerLayerGpu, (long)i * _pleWidth, _pleWidth);
                _gpuPleX       = gpu.Allocate(TensorShape.D1(_pleWidth));
                _gpuPleY       = gpu.Allocate(TensorShape.D1(_embDim));
            }
        }

        // Populate the GPU KV-aliased layer set (Gemma 4 shared-KV tail). The
        // constructor validation already guarantees all aliased GPU layers point to
        // GPU sources (no cross-tier reads).
        if (_isGemma4Like && hp.KvSourceLayer is { } kslGpu)
        {
            for (int i = 0; i < _nGpuLayers; i++)
                if (kslGpu[i] >= 0) _gpuKvAliasedLayers.Add(i);
        }

        // Upload rope_freqs.weight to GPU when present (Gemma 4 global-layer factors).
        if (_isGemma4Like
            && model.FindTensor("rope_freqs.weight") is GgufTensorInfo rfGpu
            && rfGpu.DType == DType.Float32
            && rfGpu.ElementCount == _maxHeadDim / 2
            && _nGpuLayers > 0)
        {
            _gpuRopeFreqs = UploadWeight("rope_freqs.weight");
        }

        // Pre-fault every CPU-resident mmap weight page so the first request doesn't
        // stall on demand paging (issue #221). NOT gated on _nCpuLayers > 0: the
        // CPU-MoE routed experts and the Gemma 4 PLE table are CPU-resident even when
        // every transformer layer is GPU-offloaded (-g -1), and those are the dominant
        // cold-start cost. MmapPrefault filters empty configs and honours SHARPI_PREFAULT.
        MmapPrefault.Run("CudaHybridForwardPass", BuildCpuPrefaultRegions());

        if (_tqEnabled && _nCpuLayers > 0)
        {
            _cpuTqKvCache = new TurboQuantKvCache(
                _nCpuLayers, _maxSeqLen, _numKvHeads, _headDim,
                _tqFp32Window, tqBits,
                layerIndexBase: _nGpuLayers, totalLayerCountForSeeds: _hp.NumLayers);
            _cpuRotatedQuery = Alloc(_numHeads * _headDim);
            _cpuDecompBuf = Alloc(_numHeads * _headDim);
        }
    }

    // ================================================================
    //  Forward Pass
    // ================================================================

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        ThrowIfFaulted();
        if (_isGemma4Like) return ForwardGemma4(token, position);

        // ── Phase 1: GPU layers ──
        _gpu.BeginRecord();

        // Embed token on GPU when the embedding table fits there, otherwise
        // dequantize on CPU and upload just the hidden state row.
        if (_gpuEmbedding is not null)
        {
            if (_embIsQuantized)
                _gpu.EmbedLookupQ4K(_gpuEmbedding, _gpuHidden, token, _embDim);
            else
                _gpu.EmbedLookup(_gpuEmbedding, _gpuHidden, token, _embDim);
            _gpu.RecordBarrier();
        }
        else
        {
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            CpuEmbedToken(token, pinned);
            _gpu.UnmapPinned(_pinnedHidden);
            CopyGpuBuffer(_gpuHidden, _pinnedHidden);
            _gpu.RecordBarrier();
        }

        for (int i = 0; i < _nGpuLayers; i++)
        {
            GpuLayer(i, position);
        }

        if (_tqEnabled && _nGpuLayers > 0)
        {
            if (_gpuFp32Count >= _tqFp32Window)
                _gpuTqCompressedLen++;

            _gpuFp32WriteIdx = (_gpuFp32WriteIdx + 1) % _tqFp32Window;
            if (_gpuFp32Count < _tqFp32Window)
                _gpuFp32Count++;
        }

        if (_nCpuLayers > 0)
        {
            // Download hidden state to pinned buffer.
            CopyGpuBuffer(_pinnedHidden, _gpuHidden);
            _gpu.RecordBarrier();
        }

        _gpu.EndRecordAndSubmit();

        if (_nCpuLayers > 0)
        {
            // ── Phase 2: Transfer GPU hidden → CPU ──
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new Span<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);

            // ── Phase 3: CPU layers ──
            for (int ci = 0; ci < _nCpuLayers; ci++)
            {
                CpuLayer(ci, position);
            }

            // Increment CPU KV cache
            if (_cpuTqKvCache != null)
                _cpuTqKvCache.IncrementPosition();
            else
                _cpuKvCache.IncrementPosition();

            if (_gpuOutputWeight is not null)
            {
                // ── Phase 4: Transfer CPU hidden → GPU ──
                pinned = _gpu.MapPinned(_pinnedHidden);
                new ReadOnlySpan<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(pinned, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                // Upload pinned → GPU hidden, then final norm + output
                _gpu.BeginRecord();
                CopyGpuBuffer(_gpuHidden, _pinnedHidden);
                _gpu.RecordBarrier();
            }
            else
            {
                ComputeCpuOutput();
                _kvLength = position + 1;
                return _logitsBuf;
            }
        }
        else
        {
            if (_gpuOutputWeight is null)
            {
                CopyGpuBuffer(_pinnedHidden, _gpuHidden);
                _gpu.RecordBarrier();
                _gpu.EndRecordAndSubmit();

                float* pinned = _gpu.MapPinned(_pinnedHidden);
                new Span<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                ComputeCpuOutput();
                _kvLength = position + 1;
                return _logitsBuf;
            }

            _gpu.BeginRecord();
        }

        // ── Phase 5: Final norm + output projection on GPU ──
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden);

        _gpu.RecordComputeToTransferBarrier();
        _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
        _gpu.EndRecordAndSubmit();
        _gpu.ReadFromStaging(_logitsBuf);

        _kvLength = position + 1;
        return _logitsBuf;
    }

    public void ResetCache()
    {
        _kvLength = 0;
        _gpuTqCompressedLen = 0;
        _gpuFp32WriteIdx = 0;
        _gpuFp32Count = 0;
        if (_cpuTqKvCache != null)
            _cpuTqKvCache.Reset();
        else
            _cpuKvCache.Reset();
    }

    /// <inheritdoc/>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ThrowIfFaulted();
        int n = tokens.Count;
        // Issue #123: batched-trunk prefill for the supported pure-attention configs
        // (non-Gemma4, non-TurboQuant, NEOX RoPE, learned/no QK-norm — never L2). The
        // batched trunk collapses the per-position attention launches whose cost grows
        // with context. Everything else falls back to the bit-exact per-token loop.
        if (BatchedPrefillEnabled && n >= 2 && IsBatchedPrefillSupported)
        {
            LastPrefillWasBatched = true;
            return PrefillBatchedTrunk(tokens, startPos);
        }

        LastPrefillWasBatched = false;
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < n; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    /// <summary>
    /// Configs the issue-#123 batched-trunk prefill handles bit-identically. Gemma 4
    /// (per-layer head_dim / dual-RoPE / PLE), TurboQuant-KV, non-NEOX RoPE (the only
    /// batched RoPE kernel is NEOX), and L2 QK-norm (no batched HeadNormPure) all fall
    /// back to the per-token <see cref="Forward"/> loop. Qwen3-Coder-30B (NEOX, learned
    /// QK-norm, no attn-bias, MoE) is supported with either embedding placement.
    /// <para>
    /// CPU-resident embedding (<c>_gpuEmbedding is null</c>, the low-VRAM / 12 GB-default
    /// config) is supported as of issue #218: the embed step dequantizes all N rows into a
    /// dedicated pinned staging block and uploads them in one async H2D, so there is no
    /// per-token write-after-read on a shared buffer (which is why <c>_gpuEmbedding is not
    /// null</c> is no longer required here). See <see cref="PrefillBatchedTrunk"/> step 1.
    /// </para>
    /// </summary>
    private bool IsBatchedPrefillSupported =>
        !_isGemma4Like
        && !_tqEnabled
        && _hp.IsNeoxRope
        && !(_hasQkNorm && _hp.UseL2QkNorm)
        && !_hasAttnBias
        && BatchedTrunkWeightsSupported();

    // The trunk's batched q/k/v/o GEMM-N kernels only handle Q4_K/Q5_K/Q6_K/F32. A
    // model with e.g. a Q8_0 attention weight must fall back to the per-token loop
    // rather than fault mid-prefill. Checked once per Prefill (cheap: 4 × nGpuLayers).
    private bool BatchedTrunkWeightsSupported()
    {
        for (int i = 0; i < _nGpuLayers; i++)
            if (!Batchable(_gpuWq[i]) || !Batchable(_gpuWk[i]) ||
                !Batchable(_gpuWv[i]) || !Batchable(_gpuWo[i]))
                return false;
        return true;

        bool Batchable(Tensor t)
        {
            var dt = _gpuWeightDTypes.TryGetValue(t.Handle, out var d) ? d : DType.Float32;
            return dt is DType.Q4_K or DType.Q5_K or DType.Q6_K or DType.Float32;
        }
    }

    private void ThrowIfFaulted()
    {
        if (_faulted)
            throw new InvalidOperationException(
                "CudaHybridForwardPass is in a faulted state: a previous batched prefill threw " +
                "after mutating the KV cache but before advancing its length counters, leaving the " +
                "cache inconsistent. This pass must be discarded (reload the model).");
    }

    // ================================================================
    //  Issue #123: batched-trunk prompt prefill
    // ================================================================

    /// <summary>
    /// (Re)allocate the batched-trunk device scratch to hold exactly <paramref name="n"/>
    /// tokens. Exact-size — not grow-only — so the stream / norm buffers' element counts
    /// always equal <c>n × embDim</c>, which the whole-buffer CopyDevice / AddInPlace
    /// launches rely on (mirrors CudaHybridGdnForwardPass.EnsureStreamAll).
    /// </summary>
    private void EnsureBatchedTrunkScratch(int n)
    {
        if (_bpCapN == n && _gpuBpStream is not null) return;
        FreeBatchedTrunkScratch();
        long qDim = (long)_numHeads * _headDim;
        long kvDim = (long)_numKvHeads * _headDim;
        _gpuBpStream   = _gpu.Allocate(TensorShape.D1((long)n * _embDim));
        _gpuBpNormAll  = _gpu.Allocate(TensorShape.D1((long)n * _embDim));
        _gpuBpBlockOut = _gpu.Allocate(TensorShape.D1((long)n * _embDim));
        _gpuBpQ        = _gpu.Allocate(TensorShape.D1((long)n * qDim));
        _gpuBpK        = _gpu.Allocate(TensorShape.D1((long)n * kvDim));
        _gpuBpV        = _gpu.Allocate(TensorShape.D1((long)n * kvDim));
        _gpuBpAttnOut  = _gpu.Allocate(TensorShape.D1((long)n * qDim));
        _bpCapN = n;
    }

    private void FreeBatchedTrunkScratch()
    {
        if (_gpuBpStream is { } s)   { _gpu.Free(s);   _gpuBpStream = null; }
        if (_gpuBpNormAll is { } na) { _gpu.Free(na);  _gpuBpNormAll = null; }
        if (_gpuBpBlockOut is { } b) { _gpu.Free(b);   _gpuBpBlockOut = null; }
        if (_gpuBpQ is { } q)        { _gpu.Free(q);   _gpuBpQ = null; }
        if (_gpuBpK is { } k)        { _gpu.Free(k);   _gpuBpK = null; }
        if (_gpuBpV is { } v)        { _gpu.Free(v);   _gpuBpV = null; }
        if (_gpuBpAttnOut is { } ao) { _gpu.Free(ao);  _gpuBpAttnOut = null; }
        _bpCapN = 0;
    }

    // Exact-size (matches EnsureBatchedTrunkScratch) so CopyDevice(pinnedAll, stream)
    // has equal element counts on both sides.
    private Tensor EnsurePinnedHiddenAll(int n)
    {
        if (_pinnedHiddenAll is { } p && _pinnedHiddenAllN == n) return p;
        if (_pinnedHiddenAll is { } old) { _gpu.Free(old); _pinnedHiddenAll = null; }
        _pinnedHiddenAll = _gpu.AllocatePinned(TensorShape.D1((long)n * _embDim));
        _pinnedHiddenAllN = n;
        return _pinnedHiddenAll;
    }

    // Issue #218: exact-size pinned staging for the batched CPU→GPU embedding upload, so
    // CopyDevice(stream, pinnedEmbed) has equal element counts on both sides (n × embDim).
    private Tensor EnsurePinnedEmbedAll(int n)
    {
        if (_pinnedEmbedAll is { } p && _pinnedEmbedAllN == n) return p;
        if (_pinnedEmbedAll is { } old) { _gpu.Free(old); _pinnedEmbedAll = null; }
        _pinnedEmbedAll = _gpu.AllocatePinned(TensorShape.D1((long)n * _embDim));
        _pinnedEmbedAllN = n;
        return _pinnedEmbedAll;
    }

    /// <summary>
    /// Batched-trunk prompt prefill for the pure-attention hybrid path (issue #123).
    /// The attention trunk of the <c>_nGpuLayers</c> GPU layers runs as batched launches
    /// over the N prompt tokens (RmsNormBatched → MatMulBatched q/k/v → RoPE-batched →
    /// HeadNorm-batched → KvAppendBatched → AttentionBatched/Wave → MatMulBatched o-proj →
    /// batched residual), collapsing the per-position attention launches whose cost grows
    /// with context. The FFN/MoE stage stays per token (the issue's "batched-TRUNK" scope):
    /// the SLRU expert loads / dense GEMV are amortized over the prompt cheaply enough, and
    /// per-token reuse of <see cref="GpuMoeFfn"/>/<see cref="GpuDenseFfn"/> keeps the FFN
    /// math bit-identical to <see cref="Forward"/>. The CPU layers (when <c>_nCpuLayers &gt;
    /// 0</c>, as on Qwen3-Coder-30B with -g auto) then run per token over the N hidden rows.
    ///
    /// <para>Produces bit-identical KV cache and final-token logits to the sequential
    /// per-token <see cref="Forward"/> loop: every batched trunk kernel runs the same
    /// per-row math as its single-token counterpart (proven by the backend per-kernel
    /// oracles), and the FFN/CPU stages run the identical single-token sequences.</para>
    ///
    /// <para><b>Not transactional</b> (mirror of CudaHybridGdnForwardPass): the KV pages are
    /// written as the trunk runs but <c>_kvLength</c> / the CPU KV position counters are
    /// advanced only at the end, so a mid-prefill throw leaves the cache inconsistent
    /// (<c>_faulted</c> latched). Such a failure is fatal for this pass.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillBatchedTrunk(IReadOnlyList<int> tokens, int startPos)
    {
        int N = tokens.Count;
        int embDim = _embDim;
        EnsureBatchedTrunkScratch(N);
        var stream = _gpuBpStream!;

        // Pessimistic fault latch: the KV append below is committed to the cache tensors
        // immediately, but the length counters (_kvLength / CPU KV positions) are advanced
        // only at the end. Clear only once they are consistent (step 5).
        _faulted = true;

        // 1. Embed every token into the residual-stream buffer [N × embDim].
        if (_gpuEmbedding is not null)
        {
            // GPU-resident embedding: look each row up on the GPU and scatter into the stream.
            for (int i = 0; i < N; i++)
            {
                if (_embIsQuantized) _gpu.EmbedLookupQ4K(_gpuEmbedding, _gpuHidden, tokens[i], embDim);
                else                 _gpu.EmbedLookup(_gpuEmbedding, _gpuHidden, tokens[i], embDim);
                _gpu.CopyDeviceRegion(stream, (long)i * embDim * sizeof(float),
                                      _gpuHidden, 0, (long)embDim * sizeof(float));
            }
        }
        else
        {
            // Issue #218: CPU-resident embedding (low-VRAM config). Dequantize all N rows on
            // the CPU into a dedicated pinned block [N × embDim] (the SAME per-row dequant as
            // CpuEmbedToken → byte-identical to the per-token Forward path), then upload the
            // whole block in ONE async H2D into the stream. The block is fully CPU-written
            // before the single copy is enqueued, so there is no per-token write-after-read
            // on a shared staging buffer — the hazard that gated this config out pre-#218.
            var pinnedEmbed = EnsurePinnedEmbedAll(N);
            float* host = _gpu.MapPinned(pinnedEmbed);
            for (int i = 0; i < N; i++)
                CpuEmbedToken(tokens[i], host + (long)i * embDim);
            _gpu.UnmapPinned(pinnedEmbed);
            _gpu.CopyDevice(stream, pinnedEmbed);   // pinned host → device stream (UVA H2D, N × embDim)
        }

        // 2. GPU layers: batched attention trunk + per-token FFN/MoE.
        for (int i = 0; i < _nGpuLayers; i++)
            GpuLayerBatchedTrunk(i, N, startPos);

        // 3. CPU layers (if any): transfer the N hidden rows GPU→CPU in one batched copy,
        //    then run each CPU layer per token over the N rows (reusing CpuLayer). The
        //    CPU KV cache is positional, so token i must run all CPU layers at position
        //    startPos+i before token i+1 — i.e. loop tokens outer, layers inner, exactly
        //    as the sequential Forward loop would.
        if (_nCpuLayers > 0)
        {
            var pinnedAll = EnsurePinnedHiddenAll(N);
            _gpu.CopyDevice(pinnedAll, stream);     // device stream → pinned host (N × embDim)
            _gpu.Synchronize();
            float* hostAll = _gpu.MapPinned(pinnedAll);

            for (int i = 0; i < N; i++)
            {
                int position = startPos + i;
                new ReadOnlySpan<float>(hostAll + (long)i * embDim, embDim)
                    .CopyTo(new Span<float>(_cpuHidden, embDim));

                for (int ci = 0; ci < _nCpuLayers; ci++)
                    CpuLayer(ci, position);

                if (_cpuTqKvCache != null) _cpuTqKvCache.IncrementPosition();
                else                       _cpuKvCache.IncrementPosition();

                // Only the LAST token's hidden feeds the output projection; copy it back.
                if (i == N - 1)
                    new ReadOnlySpan<float>(_cpuHidden, embDim)
                        .CopyTo(new Span<float>(hostAll + (long)i * embDim, embDim));
            }
            _gpu.UnmapPinned(pinnedAll);

            // Bring the last token's (CPU) hidden back to _gpuHidden for the output stage,
            // or compute the output entirely on CPU when there is no GPU output weight.
            if (_gpuOutputWeight is null)
            {
                _kvLength = startPos + N;
                _faulted = false;
                ComputeCpuOutput();
                return _logitsBuf;
            }

            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new ReadOnlySpan<float>(_cpuHidden, embDim).CopyTo(new Span<float>(pinned, embDim));
            _gpu.UnmapPinned(_pinnedHidden);
            _gpu.CopyDevice(_gpuHidden, _pinnedHidden);
        }
        else
        {
            // All-GPU: the last token's hidden is the final stream row.
            _gpu.CopyDeviceRegion(_gpuHidden, 0, stream, (long)(N - 1) * embDim * sizeof(float),
                                  (long)embDim * sizeof(float));

            if (_gpuOutputWeight is null)
            {
                // No GPU output weight: compute the output on CPU from the last hidden row.
                _gpu.CopyDevice(_pinnedHidden, _gpuHidden);
                _gpu.Synchronize();
                float* pinned = _gpu.MapPinned(_pinnedHidden);
                new ReadOnlySpan<float>(pinned, embDim).CopyTo(new Span<float>(_cpuHidden, embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                _kvLength = startPos + N;
                _faulted = false;
                ComputeCpuOutput();
                return _logitsBuf;
            }
        }

        // 4. Advance the GPU KV length counter (CPU counters already advanced above).
        _kvLength = startPos + N;
        _faulted = false;

        // 5. Final norm + output projection on GPU for the last token.
        _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
        GpuMatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden);
        _gpu.Download(_gpuLogits, _logitsBuf);
        return _logitsBuf;
    }

    /// <summary>
    /// One GPU layer of the batched-trunk prefill: batched attention block over all
    /// <paramref name="n"/> tokens, then the per-token FFN/MoE stage. Mirrors
    /// <see cref="GpuLayer"/> kernel-for-kernel, only batched across N.
    /// </summary>
    private void GpuLayerBatchedTrunk(int i, int n, int startPos)
    {
        var stream   = _gpuBpStream!;
        var normAll  = _gpuBpNormAll!;
        var blockOut = _gpuBpBlockOut!;
        var qAll     = _gpuBpQ!;
        var kAll     = _gpuBpK!;
        var vAll     = _gpuBpV!;
        var attnOut  = _gpuBpAttnOut!;
        int qDim  = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;

        // ── Attention norm (batched).
        _gpu.RmsNormBatched(normAll, stream, _gpuAttnNorm[i], n, _embDim, _hp.RmsNormEps);

        // ── Q/K/V projections (batched GEMM-N).
        GpuMatMulBatched(qAll, _gpuWq[i], normAll, n);
        GpuMatMulBatched(kAll, _gpuWk[i], normAll, n);
        GpuMatMulBatched(vAll, _gpuWv[i], normAll, n);

        // ── Attention bias is gated OUT of the batched path (Coder-30B has none); the
        //    supported configs never set _hasAttnBias here (see IsBatchedPrefillSupported
        //    + the Prefill dispatch). Guard defensively so a future bias model can't run
        //    silently wrong — it would have fallen back, so this is unreachable.
        if (_hasAttnBias)
            throw new InvalidOperationException("Batched-trunk prefill does not support attention bias.");

        bool useRoPE = _hp.NoRopeLayerStep == 0 || (i + 1) % _hp.NoRopeLayerStep != 0;

        // ── QK-norm BEFORE RoPE (batched, learned). RoPE does not commute with weighted
        //    QK-norm, so this must run in the same order as the per-token GpuLayer path and
        //    the CPU/HF/llama.cpp reference (issue #157). L2/pure QK-norm is gated out by
        //    IsBatchedPrefillSupported (no batched kernel); the explicit !UseL2QkNorm guard
        //    keeps _gpuQNorm/_gpuKNorm (null for L2) from dereferencing if that gate ever changes.
        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            _gpu.HeadNormBatched(qAll, _gpuQNorm![i], _numHeads,   _headDim, n, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            _gpu.HeadNormBatched(kAll, _gpuKNorm![i], _numKvHeads, _headDim, n, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
        }

        // ── RoPE (batched, full headDim NEOX). NoPE layers skip RoPE, matching GpuLayer.
        if (useRoPE)
        {
            // RoPEPartialBatched with ropeDim==headDim matches the per-token RoPE(neox)
            // path GpuLayer uses: same freq formula (2·j/headDim) and pair layout, every
            // dim rotated. IsBatchedPrefillSupported already required NEOX. This RoPE step
            // (and the RoPE≡RoPEPartial(ropeDim=headDim) equivalence) is covered end-to-end
            // by BatchedPrefill_BitwiseMatchesSequential_Coder, not the per-kernel oracle
            // (which compares RoPEPartialBatched against RoPEPartial).
            _gpu.RoPEPartialBatched(qAll, startPos, _headDim, _headDim, _hp.RopeTheta, _numHeads,   n, neox: true);
            _gpu.RoPEPartialBatched(kAll, startPos, _headDim, _headDim, _hp.RopeTheta, _numKvHeads, n, neox: true);
        }

        // ── KV append + SDPA (batched). Shared-scores fast path when startPos+n ≤ 4096,
        //    wave-based global-scratch SDPA above (issue #118). Both bit-identical to the
        //    per-position KvAppend + Attention loop in GpuLayer.
        KvAppendBatchedKv(kAll, vAll, _gpuKCache[i], _gpuVCache[i], kvDim, startPos, _maxSeqLen, n);
        if (startPos + n <= 4096)
            AttentionBatchedKv(qAll, _gpuKCache[i], _gpuVCache[i], attnOut,
                _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, n);
        else
            AttentionBatchedWaveKv(qAll, _gpuKCache[i], _gpuVCache[i], attnOut,
                _numHeads, _numKvHeads, _headDim, startPos, _maxSeqLen, n);

        // ── Output projection (batched) → blockOut, then post-attn residual add.
        GpuMatMulBatched(blockOut, _gpuWo[i], attnOut, n);
        _gpu.AddInPlace(blockOut, stream);   // blockOut = o-proj(attn) + pre-attn residual

        // ── FFN norm (batched) over the post-attn residual.
        _gpu.RmsNormBatched(normAll, blockOut, _gpuFfnNorm[i], n, _embDim, _hp.RmsNormEps);

        // ── FFN / MoE stage: per token (issue #123 scope is the trunk). Reuse the exact
        //    single-token GpuMoeFfn/GpuDenseFfn (reads _gpuNormBuf, writes _gpuHidden), then
        //    add the post-attn residual row and scatter the new hidden back into the stream.
        for (int t = 0; t < n; t++)
        {
            _gpu.CopyDeviceRegion(_gpuNormBuf, 0, normAll, (long)t * _embDim * sizeof(float),
                                  (long)_embDim * sizeof(float));
            if (_isMoE)
            {
                if (_cpuMoe) GpuTrunkCpuMoeFfn(i);
                else         GpuMoeFfn(i);
            }
            else GpuDenseFfn(i);

            _gpu.CopyDeviceRegion(_gpuResidual, 0, blockOut, (long)t * _embDim * sizeof(float),
                                  (long)_embDim * sizeof(float));
            _gpu.AddInPlace(_gpuHidden, _gpuResidual);
            _gpu.CopyDeviceRegion(stream, (long)t * _embDim * sizeof(float),
                                  _gpuHidden, 0, (long)_embDim * sizeof(float));
        }
    }

    private void GpuMatMulBatched(Tensor outputAll, Tensor weights, Tensor inputAll, int nTok)
    {
        var dt = _gpuWeightDTypes.TryGetValue(weights.Handle, out var d) ? d : DType.Float32;
        // #162: read each weight once per batch (Q8_0/Q4_K → int8 MMQ; Q6_K/Q5_K → dequant
        // →fp16 GEMM) instead of the per-token GEMM-N matvec re-stream. Argmax-stable,
        // gated so the bit-parity oracle keeps the matvec path. Guarded on block alignment.
        if (HybridPrefillComputeEnabled && nTok > 1)
        {
            int cols = (int)(inputAll.ElementCount / nTok);
            switch (dt)
            {
                case DType.Q8_0 when (cols & 31) == 0:
                    _gpu.MatMulBatchedMmq(outputAll, weights, inputAll, nTok, dt); return;
                case DType.Q4_K when (cols & 0xff) == 0:
                    _gpu.MatMulBatchedMmq(outputAll, weights, inputAll, nTok, dt); return;
                case DType.Q6_K when (cols & 0xff) == 0:
                case DType.Q5_K when (cols & 0xff) == 0:
                    _gpu.MatMulBatchedGemm(outputAll, weights, inputAll, nTok, dt); return;
            }
        }
        _gpu.MatMulBatched(outputAll, weights, inputAll, nTok, dt);
    }

    // ================================================================
    //  GPU Layer (same pattern as GpuForwardPass)
    // ================================================================

    // ── Narrowed-KV dispatch (#179/#230) ──────────────────────────────────────
    // Route every GPU-trunk KV append/attention through the dtype the cache was allocated at
    // (_kvDType). Mirrors CudaForwardPass's KvAppendKv/AttentionKv/AttentionSwaKv; fp32 is the
    // unchanged default. The TQ path is excluded (forces _kvDType == fp32).
    private void KvAppendKv(Tensor k, Tensor v, Tensor kCache, Tensor vCache, int kvDim, int position, int maxSeqLen)
    {
        if (_kvDType == DType.BFloat16) _gpu.KvAppendBf16(k, v, kCache, vCache, kvDim, position, maxSeqLen);
        else if (_kvDType == DType.Q8_0) _gpu.KvAppendQ8_0(k, v, kCache, vCache, kvDim, position, maxSeqLen);
        else _gpu.KvAppend(k, v, kCache, vCache, kvDim, position, maxSeqLen);
    }

    private void AttentionKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output, Tensor? scoresScratch,
                             int numHeads, int numKvHeads, int headDim, int seqLen, int maxSeqLen, float attnScale = -1f)
    {
        // Flash-decoding split-KV (#238): at long ctx the single-block-per-head kernels below
        // underutilize the GPU. Mirrors the dense CudaForwardPass gate (incl. #237 grouped).
        if (_splitKvPartialO is { } splitO && _splitKvPartialMeta is { } splitMeta
            && seqLen > HybridSplitMinSeq && maxSeqLen <= _maxSeqLen)
        {
            bool grouped = CudaForwardPass.ShouldUseGroupedSplit(_splitGroupedMode, _kvDType, numHeads, numKvHeads, seqLen);
            _gpu.AttentionSplitKv(q, kCache, vCache, output, splitO, splitMeta, _kvDType,
                numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale, grouped: grouped);
            return;
        }

        if (_kvDType == DType.BFloat16)
            _gpu.AttentionBf16(q, kCache, vCache, output, scoresScratch, numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionQ8_0(q, kCache, vCache, output, scoresScratch, numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
        else
            _gpu.Attention(q, kCache, vCache, output, scoresScratch, numHeads, numKvHeads, headDim, seqLen, maxSeqLen, attnScale);
    }

    private void AttentionSwaKv(Tensor q, Tensor kCache, Tensor vCache, Tensor output, Tensor? scoresScratch,
                                int position, int windowSize, int headDim, int numHeads, int numKvHeads, int maxSeqLen, float attnScale = -1f)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.AttentionSwaBf16(q, kCache, vCache, output, scoresScratch, position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionSwaQ8_0(q, kCache, vCache, output, scoresScratch, position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
        else
            _gpu.AttentionSwa(q, kCache, vCache, output, scoresScratch, position, windowSize, headDim, numHeads, numKvHeads, maxSeqLen, attnScale);
    }

    private void KvAppendBatchedKv(Tensor kAll, Tensor vAll, Tensor kCache, Tensor vCache, int kvDim, int startPos, int maxSeqLen, int nTok)
    {
        if (_kvDType == DType.BFloat16) _gpu.KvAppendBatchedBf16(kAll, vAll, kCache, vCache, kvDim, startPos, maxSeqLen, nTok);
        else if (_kvDType == DType.Q8_0) _gpu.KvAppendBatchedQ8_0(kAll, vAll, kCache, vCache, kvDim, startPos, maxSeqLen, nTok);
        else _gpu.KvAppendBatched(kAll, vAll, kCache, vCache, kvDim, startPos, maxSeqLen, nTok);
    }

    private void AttentionBatchedKv(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                    int numHeads, int numKvHeads, int headDim, int startPos, int maxSeqLen, int nTok)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.AttentionBatchedBf16(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionBatchedQ8_0(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
        else
            _gpu.AttentionBatched(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
    }

    private void AttentionBatchedWaveKv(Tensor qAll, Tensor kCache, Tensor vCache, Tensor outAll,
                                        int numHeads, int numKvHeads, int headDim, int startPos, int maxSeqLen, int nTok)
    {
        if (_kvDType == DType.BFloat16)
            _gpu.AttentionBatchedWaveBf16(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
        else if (_kvDType == DType.Q8_0)
            _gpu.AttentionBatchedWaveQ8_0(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
        else
            _gpu.AttentionBatchedWave(qAll, kCache, vCache, outAll, numHeads, numKvHeads, headDim, startPos, maxSeqLen, nTok);
    }

    private void GpuLayer(int i, int position)
    {
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuQ, _gpuWq[i], _gpuNormBuf);
        GpuMatMul(_gpuK, _gpuWk[i], _gpuNormBuf);
        GpuMatMul(_gpuV, _gpuWv[i], _gpuNormBuf);
        _gpu.RecordBarrier();

        if (_hasAttnBias)
        {
            _gpu.AddInPlace(_gpuQ, _gpuBq![i]);
            _gpu.AddInPlace(_gpuK, _gpuBk![i]);
            _gpu.AddInPlace(_gpuV, _gpuBv![i]);
            _gpu.RecordBarrier();
        }

        {
            // NoPE: skip RoPE for NoPE layers
            bool useRoPE = _hp.NoRopeLayerStep == 0
                || (i + 1) % _hp.NoRopeLayerStep != 0;

            // Ordering (issue #157): RoPE does NOT commute with per-channel-weighted
            // QK-norm, so mirror the CPU ForwardPass / HF Qwen3 / llama.cpp order:
            //   • weighted QK-norm (Qwen3, …): norm BEFORE RoPE
            //   • L2 QK-norm (Llama-4):        norm AFTER  RoPE (RoPE layers only)
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            {
                _gpu.HeadNorm(_gpuQ, _gpuQNorm![i], _numHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.HeadNorm(_gpuK, _gpuKNorm![i], _numKvHeads, _headDim, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
                _gpu.RecordBarrier();
            }

            if (useRoPE)
            {
                _gpu.RoPE(_gpuQ, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RoPE(_gpuK, position, _headDim, _hp.RopeTheta, _hp.IsNeoxRope);
                _gpu.RecordBarrier();
            }

            if (_hasQkNorm && _hp.UseL2QkNorm && useRoPE)
            {
                _gpu.HeadNormPure(_gpuQ, _numHeads, _headDim, _hp.RmsNormEps);
                _gpu.HeadNormPure(_gpuK, _numKvHeads, _headDim, _hp.RmsNormEps);
                _gpu.RecordBarrier();
            }
        }

        if (_tqEnabled)
        {
            int kvDim = _numKvHeads * _headDim;
            long rowBytes = (long)kvDim * sizeof(float);

            if (_gpuFp32Count >= _tqFp32Window)
            {
                CopyGpuBufferRegion(_gpuEvictK!, 0, _gpuKCache[i], (long)_gpuFp32WriteIdx * rowBytes, rowBytes);
                CopyGpuBufferRegion(_gpuEvictV!, 0, _gpuVCache[i], (long)_gpuFp32WriteIdx * rowBytes, rowBytes);
                _gpu.RecordBarrier();

                _gpu.TqKvAppend(_gpuEvictK!, _gpuEvictV!, _gpuTqKCache![i], _gpuTqVCache![i],
                    _gpuSignPatterns![i], _gpuCodebook!, _gpuBoundaries!,
                    kvDim, _headDim, _gpuTqCompressedLen,
                    _maxSeqLen, _numKvHeads, _tqBlockBytes);
                _gpu.RecordBarrier();
            }

            _gpu.KvAppend(_gpuK, _gpuV, _gpuKCache[i], _gpuVCache[i],
                kvDim, _gpuFp32WriteIdx, _tqFp32Window);
            _gpu.RecordBarrier();

            _gpu.TqRotateQuery(_gpuQ, _gpuRotatedQ!, _gpuSignPatterns![i],
                _numHeads, _numKvHeads, _headDim);
            _gpu.RecordBarrier();

            int fp32SeqLen = Math.Min(_gpuFp32Count + 1, _tqFp32Window);
            _gpu.TqAttention(_gpuQ, _gpuRotatedQ!, _gpuTqKCache![i], _gpuTqVCache![i],
                _gpuKCache[i], _gpuVCache[i], _gpuAttnOut, _gpuCodebook!,
                _gpuAttnScoresScratch!,
                _numHeads, _numKvHeads, _headDim,
                _gpuTqCompressedLen, fp32SeqLen, _maxSeqLen, _tqBlockBytes);
        }
        else
        {
            KvAppendKv(_gpuK, _gpuV, _gpuKCache[i], _gpuVCache[i],
                (_numKvHeads * _headDim), position, _maxSeqLen);
            _gpu.RecordBarrier();

            AttentionKv(_gpuQ, _gpuKCache[i], _gpuVCache[i], _gpuAttnOut,
                _gpuAttnScoresScratch,
                _numHeads, _numKvHeads, _headDim,
                (position + 1), _maxSeqLen);
        }
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWo[i], _gpuAttnOut);
        if (_hasAttnBias)
        {
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuBo![i]);
        }
        _gpu.RecordBarrier();
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();

        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuFfnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        if (_isMoE)
        {
            if (_cpuMoe) GpuTrunkCpuMoeFfn(i);
            else         GpuMoeFfn(i);
        }
        else
            GpuDenseFfn(i);
        _gpu.RecordBarrier();

        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();
    }

    // ================================================================
    //  CPU Layer (same pattern as ForwardPass)
    // ================================================================

    private void CpuLayer(int ci, int position)
    {
        // Save residual
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-attention RMS norm
        var normW = GetCpuNormWeight(_cpuAttnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, normW, _embDim, _hp.RmsNormEps);

        // Q/K/V projections
        SimdKernels.MatVec(_cpuQ, _cpuWq[ci].DataPtr, _cpuNormBuf, _numHeads * _headDim, _embDim, _cpuWq[ci].DType);
        SimdKernels.MatVec(_cpuK, _cpuWk[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWk[ci].DType);
        SimdKernels.MatVec(_cpuV, _cpuWv[ci].DataPtr, _cpuNormBuf, _numKvHeads * _headDim, _embDim, _cpuWv[ci].DType);

        if (_hasAttnBias)
        {
            SimdKernels.AddInPlace(_cpuQ, _cpuBq[ci], _numHeads * _headDim);
            SimdKernels.AddInPlace(_cpuK, _cpuBk[ci], _numKvHeads * _headDim);
            SimdKernels.AddInPlace(_cpuV, _cpuBv[ci], _numKvHeads * _headDim);
        }

        // NoPE: actual layer index = ci + _nGpuLayers
        int actualLayer = ci + _nGpuLayers;
        bool useRoPE = _hp.NoRopeLayerStep == 0
            || (actualLayer + 1) % _hp.NoRopeLayerStep != 0;

        if (useRoPE)
        {
            var cos = _ropeCosTable + (long)position * _ropeHalfDim;
            var sin = _ropeSinTable + (long)position * _ropeHalfDim;
            if (_hp.IsNeoxRope)
            {
                SimdKernels.ApplyRoPECachedNeox(_cpuQ, cos, sin, _numHeads, _headDim);
                SimdKernels.ApplyRoPECachedNeox(_cpuK, cos, sin, _numKvHeads, _headDim);
            }
            else
            {
                SimdKernels.ApplyRoPECached(_cpuQ, cos, sin, _numHeads, _headDim);
                SimdKernels.ApplyRoPECached(_cpuK, cos, sin, _numKvHeads, _headDim);
            }
        }

        // QK-norm: for L2 (Llama-4), only on RoPE layers per llama.cpp
        if (_hasQkNorm && (_hp.UseL2QkNorm ? useRoPE : true))
        {
            if (_hp.UseL2QkNorm)
            {
                PerHeadPureRmsNorm(_cpuQ, _numHeads, _headDim, _hp.RmsNormEps);
                PerHeadPureRmsNorm(_cpuK, _numKvHeads, _headDim, _hp.RmsNormEps);
            }
            else
            {
                if (_hp.IsPerChannelQkNorm)
                {
                    PerChannelRmsNorm(_cpuQ, _cpuQNorm[ci], (int)_numHeads,   _headDim, _hp.RmsNormEps);
                    PerChannelRmsNorm(_cpuK, _cpuKNorm[ci], (int)_numKvHeads, _headDim, _hp.RmsNormEps);
                }
                else
                {
                    PerHeadRmsNorm(_cpuQ, _cpuQNorm[ci], (int)_numHeads,   _headDim, _hp.RmsNormEps);
                    PerHeadRmsNorm(_cpuK, _cpuKNorm[ci], (int)_numKvHeads, _headDim, _hp.RmsNormEps);
                }
            }
        }

        // KV cache append (ci = CPU layer index)
        if (_cpuTqKvCache != null)
        {
            _cpuTqKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }
        else
        {
            _cpuKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_cpuV, _numKvHeads * _headDim));
        }

        // Attention
        if (_cpuTqKvCache != null)
            CpuTqAttention(ci, position);
        else
            CpuAttention(ci, position);

        // Output projection
        SimdKernels.MatVec(_cpuHidden, _cpuWo[ci].DataPtr, _cpuAttnOut, _embDim, _numHeads * _headDim, _cpuWo[ci].DType);
        if (_hasAttnBias)
            SimdKernels.AddInPlace(_cpuHidden, _cpuBo[ci], _embDim);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);

        // Save residual for FFN
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-FFN RMS norm
        var ffnNormW = GetCpuNormWeight(_cpuFfnNorm[ci]);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, ffnNormW, _embDim, _hp.RmsNormEps);

        if (_isMoE)
            CpuMoeFfn(ci);
        else
            CpuDenseFfn(ci);

        // Residual
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);
    }

    private void CpuDenseFfn(int ci)
    {
        SimdKernels.MatVecDual(_cpuFfnGate, _cpuWGate[ci].DataPtr, _cpuFfnUp, _cpuWUp[ci].DataPtr,
            _cpuNormBuf, _intermDim, _embDim, _cpuWGate[ci].DType, _cpuWUp[ci].DType);
        if (_hp.FfnActivation == FfnActivation.GeluApprox)
            SimdKernels.GeluTanhMul(_cpuFfnGate, _cpuFfnUp, _cpuFfnGate, _intermDim);
        else
            SimdKernels.SiLuMul(_cpuFfnGate, _cpuFfnUp, _intermDim);
        SimdKernels.MatVec(_cpuHidden, _cpuWDown[ci].DataPtr, _cpuFfnGate, _embDim, _intermDim, _cpuWDown[ci].DType);
    }

    // ================================================================
    //  Gemma 4 Forward — orchestrates embed scale + PLE pre-pass + per-tier layer
    //  dispatch + final norm/output/softcap. Mirrors CudaForwardPass.ForwardGemma4
    //  for the GPU half and ForwardPass.Forward (Gemma 4 path) for the CPU half.
    // ================================================================

    // CUDA Graph decode (issue #136): the GPU layer loop has static topology across
    // decode tokens (only `position` varies), so capture it once on the first decode
    // token and replay per token — collapsing the per-layer launches into one
    // cuGraphLaunch + a handful of node-param updates. Off by default; opt in with
    // SHARPI_CUDA_GRAPH=1 or the UseCudaGraph setter. Falls back to direct launches on
    // any capture/replay failure. The hybrid win is smaller than the all-GPU path's
    // since host transfers + CPU layers bracket only a fraction of the layers.
    private bool _useCudaGraph =
        Environment.GetEnvironmentVariable("SHARPI_CUDA_GRAPH") == "1";
    private bool _graphCaptured;

    /// <summary>
    /// Enable/disable CUDA-graph capture+replay for the Gemma 4 GPU layer loop. Defaults
    /// from the <c>SHARPI_CUDA_GRAPH</c> env var; set before the first decode (tests/bench).
    /// </summary>
    public bool UseCudaGraph { get => _useCudaGraph; set => _useCudaGraph = value; }

    // CUDA-graph capture/replay for the Gemma 4 GPU layer loop. Returns true if the loop
    // ran via the graph (captured on the first decode token, replayed after); false means
    // the caller must run the layers with direct launches. Any failure disables graphs for
    // the rest of the session and degrades to direct launches.
    private bool TryRunGpuLayersGemma4ViaGraph(int position)
    {
        if (!_useCudaGraph || !_gpu.GraphCaptureSupported || _tqEnabled)
            return false;

        // Steady state: graph already captured — just replay at the new position.
        if (_graphCaptured && _gpu.GraphReady)
        {
            try { _gpu.LaunchGraphForPosition(position); return true; }
            catch { _useCudaGraph = false; _graphCaptured = false; return false; }
        }

        if (_graphCaptured)
            return false;

        // First decode token: capture the loop (records onto the stream without executing)
        // then launch it for real. On any failure the loop was NOT executed, so the caller
        // re-runs it directly.
        try
        {
            if (!_gpu.TryBeginGraphCapture())
                return false;
            for (int i = 0; i < _nGpuLayers; i++)
                GpuLayerGemma4(i, position);
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

    private ReadOnlySpan<float> ForwardGemma4(int token, int position)
    {
        // PLE pre-pass needs the hidden state, so embed first.
        // Embedding lives wherever ShouldKeepFixedWeightsOnCpu decided.
        if (_gpuEmbedding is not null)
        {
            _gpu.BeginRecord();
            if (_embIsQuantized)
            {
                var embDType = _gpuWeightDTypes.GetValueOrDefault(_gpuEmbedding.Handle, DType.Q4_K);
                if (embDType == DType.Q8_0) _gpu.EmbedLookupQ8_0(_gpuEmbedding, _gpuHidden, token, _embDim);
                else _gpu.EmbedLookupQ4K(_gpuEmbedding, _gpuHidden, token, _embDim);
            }
            else
            {
                _gpu.EmbedLookup(_gpuEmbedding, _gpuHidden, token, _embDim);
            }
            _gpu.RecordBarrier();

            // Download embedded hidden to CPU for the PLE pre-pass (and any CPU layers).
            CopyGpuBuffer(_pinnedHidden, _gpuHidden);
            _gpu.RecordBarrier();
            _gpu.EndRecordAndSubmit();
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new ReadOnlySpan<float>(pinned, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);
        }
        else
        {
            CpuEmbedToken(token, _cpuHidden);
        }

        // Embedding scale (× sqrt(embDim) on Gemma 4).
        if (_hp.EmbeddingScale != 1f)
            SimdKernels.ScaleInPlace(_cpuHidden, _hp.EmbeddingScale, _embDim);

        // PLE pre-pass on CPU — produces _projPerLayer[NumLayers * pleWidth].
        if (_hp.HasPerLayerTokenEmbd)
            BuildPerLayerProjectionsCpu(token);

        // Re-upload the (possibly scaled) hidden state to GPU for the GPU half.
        if (_nGpuLayers > 0)
        {
            float* pinned = _gpu.MapPinned(_pinnedHidden);
            new ReadOnlySpan<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(pinned, _embDim));
            _gpu.UnmapPinned(_pinnedHidden);
            _gpu.BeginRecord();
            CopyGpuBuffer(_gpuHidden, _pinnedHidden);
            _gpu.RecordBarrier();

            // Upload the full per-layer PLE projection once (was L tiny per-layer
            // uploads): GPU layer i reads slice i via a static non-owning view.
            if (_hp.HasPerLayerTokenEmbd && _gpuInpGate is not null)
            {
                _gpu.UploadInto(_gpuProjPerLayerGpu!,
                    new ReadOnlySpan<float>(_projPerLayer, _nGpuLayers * _pleWidth));
                _gpu.RecordBarrier();
            }

            // The GPU layer loop has static topology across decode tokens (only
            // `position` varies), so optionally capture it once into a CUDA graph and
            // replay per token (issue #136). Uploads above and the download below stay
            // outside the captured region (they are host transfers).
            if (!TryRunGpuLayersGemma4ViaGraph(position))
                for (int i = 0; i < _nGpuLayers; i++)
                    GpuLayerGemma4(i, position);

            if (_nCpuLayers > 0)
            {
                CopyGpuBuffer(_pinnedHidden, _gpuHidden);
                _gpu.RecordBarrier();
            }
            _gpu.EndRecordAndSubmit();

            if (_nCpuLayers > 0)
            {
                float* pinnedDown = _gpu.MapPinned(_pinnedHidden);
                new ReadOnlySpan<float>(pinnedDown, _embDim).CopyTo(new Span<float>(_cpuHidden, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);
            }
        }

        if (_nCpuLayers > 0)
        {
            for (int ci = 0; ci < _nCpuLayers; ci++)
                CpuLayerGemma4(ci, position);

            _cpuKvCache.IncrementPosition();

            if (_gpuOutputWeight is not null)
            {
                float* pinned = _gpu.MapPinned(_pinnedHidden);
                new ReadOnlySpan<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(pinned, _embDim));
                _gpu.UnmapPinned(_pinnedHidden);

                _gpu.BeginRecord();
                CopyGpuBuffer(_gpuHidden, _pinnedHidden);
                _gpu.RecordBarrier();
                _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
                _gpu.RecordBarrier();
                GpuMatMul(_gpuLogits, _gpuOutputWeight, _gpuHidden);
                if (_hp.FinalLogitSoftcap > 0f)
                {
                    _gpu.RecordBarrier();
                    _gpu.SoftcapInPlace(_gpuLogits, _hp.FinalLogitSoftcap);
                }
                _gpu.RecordComputeToTransferBarrier();
                _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
                _gpu.EndRecordAndSubmit();
                _gpu.ReadFromStaging(_logitsBuf);
            }
            else
            {
                // Final norm + output on CPU.
                var outNormW = GetCpuNormWeight(_cpuOutputNorm);
                SimdKernels.RmsNormWide(_cpuNormBuf, _cpuHidden, outNormW, _embDim, _hp.RmsNormEps);
                fixed (float* logits = _logitsBuf)
                {
                    SimdKernels.MatVec(logits, _cpuOutputWeight.DataPtr, _cpuNormBuf,
                        _hp.VocabSize, _embDim, _cpuOutputWeight.DType);
                    if (_hp.FinalLogitSoftcap > 0f)
                        SimdKernels.SoftcapInPlace(logits, _hp.VocabSize, _hp.FinalLogitSoftcap);
                }
            }
        }
        else
        {
            // All-GPU layer set already produced _gpuHidden; finalise on GPU.
            _gpu.BeginRecord();
            _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuOutputNorm!, _hp.RmsNormEps);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuLogits, _gpuOutputWeight!, _gpuHidden);
            if (_hp.FinalLogitSoftcap > 0f)
            {
                _gpu.RecordBarrier();
                _gpu.SoftcapInPlace(_gpuLogits, _hp.FinalLogitSoftcap);
            }
            _gpu.RecordComputeToTransferBarrier();
            _gpu.RecordDownloadToStaging(_gpuLogits, _logitsBuf.Length);
            _gpu.EndRecordAndSubmit();
            _gpu.ReadFromStaging(_logitsBuf);
        }

        _kvLength = position + 1;
        return _logitsBuf;
    }

    // ────────────────────────────────────────────────────────────────────
    //  Gemma 4 GPU layer — mirrors CudaForwardPass.ForwardGemma4 inner loop.
    // ────────────────────────────────────────────────────────────────────

    private void GpuLayerGemma4(int i, int position)
    {
        int layerHd = _hp.LayerHeadDim![i];
        // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on the global layers).
        int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[i] : _numKvHeads;
        int qDimL = _numHeads * layerHd;
        int kvDimL = layerKv * layerHd;
        bool isSwa = _hp.IsSwaLayer is { } swa && swa[i];
        bool kvShared = _gpuKvAliasedLayers.Contains(i);
        // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the raw K
        // projection. _gpuWv[i] is default(Tensor) on those layers (not uploaded).
        bool kEqV = _hp.AttentionKEqV && !isSwa;
        int effLayer = i; // validated above: no kv-shared GPU layers

        var qView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _gpuQ.Handle);
        var kView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _gpuK.Handle);
        var vView = new Tensor(TensorShape.D1(kvDimL), DType.Float32, _gpuV.Handle);
        var attnOutView = new Tensor(TensorShape.D1(qDimL), DType.Float32, _gpuAttnOut.Handle);

        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuAttnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();

        GpuMatMul(qView, _gpuWq[i], _gpuNormBuf);
        if (!kvShared)
        {
            GpuMatMul(kView, _gpuWk[i], _gpuNormBuf);
            if (kEqV)
                CopyGpuBuffer(vView, kView);   // V = raw K projection (pre-norm, pre-RoPE)
            else
                GpuMatMul(vView, _gpuWv[i], _gpuNormBuf);
        }
        _gpu.RecordBarrier();

        // Per-head Q/K norm (Gemma 4: shared headDim-sized weight per head, applied
        // BEFORE RoPE since UseL2QkNorm == false).
        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            if (!kvShared)
                _gpu.HeadNormQk(qView, _gpuQNorm![i], kView, _gpuKNorm![i],
                    _numHeads, layerKv, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            else
                _gpu.HeadNorm(qView, _gpuQNorm![i], _numHeads, layerHd, _hp.RmsNormEps, _hp.IsPerChannelQkNorm);
            _gpu.RecordBarrier();
        }

        // Gemma 4: V gets a plain per-head RMSNorm (no learned weight) before the KV cache
        // on every KV-owning layer (llama.cpp gemma4.cpp:227, unconditional for has_kv).
        // This method is gemma4-only (layerHd from LayerHeadDim![i]), so !kvShared == has_kv.
        // Applied for E4B too — matching the full-GPU CudaForwardPass AND the hybrid CPU half
        // (CpuLayerGemma4 already V-norms unconditionally); the old AttentionKEqV gate left
        // the E4B hybrid GPU tier un-normed while the CPU tier normed → split-internal
        // inconsistency. Mirrors the CPU ForwardPass reference.
        if (!kvShared)
        {
            _gpu.HeadNormPure(vView, layerKv, layerHd, _hp.RmsNormEps);
            _gpu.RecordBarrier();
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
        _gpu.RecordBarrier();

        if (!kvShared)
        {
            int layerCtx = isSwa && _hp.SlidingWindowSize > 0
                ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
                : _maxSeqLen;
            KvAppendKv(kView, vView, _gpuKCache[i], _gpuVCache[i], kvDimL, position, layerCtx);
            _gpu.RecordBarrier();
        }

        int effLayerCtx = (_hp.IsSwaLayer is { } swaEff && swaEff[effLayer]
                          && _hp.SlidingWindowSize > 0)
            ? Math.Min(_maxSeqLen, _hp.SlidingWindowSize)
            : _maxSeqLen;

        // Gemma 4: attention_scale = 1.0, passed explicitly (kernel skips its rsqrtf).
        if (isSwa)
        {
            AttentionSwaKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                _gpuAttnScoresScratch,
                position, _hp.SlidingWindowSize, layerHd,
                _numHeads, layerKv, effLayerCtx, attnScale: 1f);
        }
        else
        {
            AttentionKv(qView, _gpuKCache[effLayer], _gpuVCache[effLayer], attnOutView,
                _gpuAttnScoresScratch,
                _numHeads, layerKv, layerHd, position + 1, effLayerCtx, attnScale: 1f);
        }
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWo[i], attnOutView);
        _gpu.RecordBarrier();

        // Post-attn RmsNorm before residual.
        if (_gpuPostAttnNorm is not null)
        {
            _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuPostAttnNorm[i], _hp.RmsNormEps);
            _gpu.RecordBarrier();
        }
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        // FFN.
        CopyGpuBuffer(_gpuResidual, _gpuHidden);
        _gpu.RecordBarrier();
        _gpu.RmsNorm(_gpuNormBuf, _gpuHidden, _gpuFfnNorm[i], _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuFfnGate, _gpuWGate[i], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp,   _gpuWUp[i],   _gpuNormBuf);
        _gpu.RecordBarrier();
        _gpu.GeluTanhMul(_gpuFfnGate, _gpuFfnUp);
        _gpu.RecordBarrier();
        GpuMatMul(_gpuHidden, _gpuWDown[i], _gpuFfnGate);
        _gpu.RecordBarrier();

        if (_gpuPostFfwNorm is not null)
        {
            _gpu.RmsNorm(_gpuHidden, _gpuHidden, _gpuPostFfwNorm[i], _hp.RmsNormEps);
            _gpu.RecordBarrier();
        }
        _gpu.AddInPlace(_gpuHidden, _gpuResidual);
        _gpu.RecordBarrier();

        // PLE injection — the per-layer projection slice is already resident
        // (uploaded once per token); read it via the static view.
        if (_hp.HasPerLayerTokenEmbd && _gpuInpGate is not null)
        {
            GpuMatMul(_gpuPleX!, _gpuInpGate[i], _gpuHidden);
            _gpu.RecordBarrier();
            _gpu.GeluTanhMul(_gpuPleX!, _gpuProjSliceViews![i]);
            _gpu.RecordBarrier();
            GpuMatMul(_gpuPleY!, _gpuPleProj![i], _gpuPleX!);
            _gpu.RecordBarrier();
            _gpu.RmsNorm(_gpuPleY!, _gpuPleY!, _gpuPlePostNorm![i], _hp.RmsNormEps);
            _gpu.RecordBarrier();
            _gpu.AddInPlace(_gpuHidden, _gpuPleY!);
            _gpu.RecordBarrier();
        }

        if (_layerOutputScale is not null)
        {
            _gpu.ScaleInPlace(_gpuHidden, _layerOutputScale[i]);
            _gpu.RecordBarrier();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Gemma 4 CPU layer — mirrors ForwardPass.Forward Gemma 4 inner loop.
    // ────────────────────────────────────────────────────────────────────

    private void CpuLayerGemma4(int ci, int position)
    {
        int li = ci + _nGpuLayers;
        int layerHd = _hp.LayerHeadDim![li];
        // Per-layer KV head count (Gemma 4 12B: 8 GQA on SWA, 1 MQA on the global layers).
        int layerKv = _hp.LayerKvHeads is { } lkv ? lkv[li] : _numKvHeads;
        int qDimL = _numHeads * layerHd;
        int kvDimL = layerKv * layerHd;
        bool isSwa = _hp.IsSwaLayer is { } swa && swa[li];
        var ksl = _hp.KvSourceLayer;
        bool kvShared = ksl is not null && ksl[li] >= 0;
        // Gemma 4 12B global layers carry no attn_v (attention_k_eq_v): V reuses the raw K
        // projection. _cpuWv[ci] is default(CpuWeightRef) (DataPtr == null) on those layers.
        bool kEqV = _hp.AttentionKEqV && !isSwa;
        int kvSrcLi = kvShared ? ksl![li] : li;
        int effCi = kvSrcLi - _nGpuLayers; // validated to be on CPU side
        int windowSize = isSwa && _hp.SlidingWindowSize > 0 ? _hp.SlidingWindowSize : -1;

        // Save residual.
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));

        // Pre-attention RmsNorm.
        var normW = GetCpuNormWeight(_cpuAttnNorm[ci]);
        SimdKernels.RmsNormWide(_cpuNormBuf, _cpuHidden, normW, _embDim, _hp.RmsNormEps);

        // Per-layer head_dim: clear Q/K/V scratch tails so MatVec only fills active rows.
        new Span<float>(_cpuQ, _numHeads * _maxHeadDim).Clear();
        new Span<float>(_cpuK, _numKvHeads * _maxHeadDim).Clear();
        new Span<float>(_cpuV, _numKvHeads * _maxHeadDim).Clear();

        SimdKernels.MatVec(_cpuQ, _cpuWq[ci].DataPtr, _cpuNormBuf, qDimL, _embDim, _cpuWq[ci].DType);
        if (!kvShared)
        {
            if (kEqV)
            {
                // Global k_eq_v layers: no attn_v weight — V is the raw K projection
                // (copied BEFORE QK-norm and RoPE, then pure-RMS-normed below). Mirrors
                // CudaForwardPass CopyDevice(vView, kView) + the CPU ForwardPass core.
                SimdKernels.MatVec(_cpuK, _cpuWk[ci].DataPtr, _cpuNormBuf, kvDimL, _embDim, _cpuWk[ci].DType);
                new ReadOnlySpan<float>(_cpuK, kvDimL).CopyTo(new Span<float>(_cpuV, kvDimL));
            }
            else
            {
                // Fuse K and V via MatVecDual — same row count, same input, FP-order
                // drift is acceptable on Gemma 4 (internal-only argmax parity test).
                SimdKernels.MatVecDual(_cpuK, _cpuWk[ci].DataPtr, _cpuV, _cpuWv[ci].DataPtr,
                    _cpuNormBuf, kvDimL, _embDim, _cpuWk[ci].DType, _cpuWv[ci].DType);
            }
        }

        if (_hasAttnBias)
        {
            SimdKernels.AddInPlace(_cpuQ, _cpuBq[ci], qDimL);
            if (!kvShared)
            {
                SimdKernels.AddInPlace(_cpuK, _cpuBk[ci], kvDimL);
                if (!kEqV)
                    SimdKernels.AddInPlace(_cpuV, _cpuBv[ci], kvDimL);
            }
        }

        // Per-head Q/K norm (Gemma 4: applied BEFORE RoPE, UseL2QkNorm == false).
        if (_hasQkNorm && !_hp.UseL2QkNorm)
        {
            if (_hp.IsPerChannelQkNorm)
            {
                PerChannelRmsNorm(_cpuQ, _cpuQNorm[ci], _numHeads, layerHd, _hp.RmsNormEps);
                if (!kvShared)
                    PerChannelRmsNorm(_cpuK, _cpuKNorm[ci], layerKv, layerHd, _hp.RmsNormEps);
            }
            else
            {
                PerHeadRmsNorm(_cpuQ, _cpuQNorm[ci], _numHeads, layerHd, _hp.RmsNormEps);
                if (!kvShared)
                    PerHeadRmsNorm(_cpuK, _cpuKNorm[ci], layerKv, layerHd, _hp.RmsNormEps);
            }
        }

        // Gemma 4: V per-head pure RmsNorm (no learned weight) before cache.
        if (!kvShared)
            PerHeadPureRmsNorm(_cpuV, layerKv, layerHd, _hp.RmsNormEps);

        // RoPE — dual-table (global vs SWA).
        bool useSwaTable = isSwa && _ropeCosTableSwa != null;
        int halfDim = useSwaTable ? _ropeHalfDimSwa : _ropeHalfDim;
        float* cosTab = useSwaTable ? _ropeCosTableSwa : _ropeCosTable;
        float* sinTab = useSwaTable ? _ropeSinTableSwa : _ropeSinTable;
        float* cos = cosTab + (long)position * halfDim;
        float* sin = sinTab + (long)position * halfDim;
        if (_hp.IsNeoxRope)
        {
            SimdKernels.ApplyRoPECachedNeox(_cpuQ, cos, sin, _numHeads, layerHd);
            if (!kvShared)
                SimdKernels.ApplyRoPECachedNeox(_cpuK, cos, sin, layerKv, layerHd);
        }
        else
        {
            SimdKernels.ApplyRoPECached(_cpuQ, cos, sin, _numHeads, layerHd);
            if (!kvShared)
                SimdKernels.ApplyRoPECached(_cpuK, cos, sin, layerKv, layerHd);
        }

        // KV append. The cache row is _maxKvDim wide on Gemma 4; the active head_dim
        // populates the leading slots, trailing slots are zero (cleared above).
        if (!kvShared)
        {
            int appendLen = _cpuKvCache.KvDim; // _numKvHeads * _maxHeadDim
            _cpuKvCache.Append(ci,
                new ReadOnlySpan<float>(_cpuK, appendLen),
                new ReadOnlySpan<float>(_cpuV, appendLen));
        }

        // Attention reads `effCi` on Gemma 4. The cache stride between kv heads is
        // _maxHeadDim (the cache row width). The active KV head count (layerKv) drives the
        // head→KV-group ratio: MQA global layers (layerKv=1) map all heads to KV head 0.
        CpuAttentionGemma4(ci, effCi, position, layerHd, qDimL, windowSize, layerKv);

        // Output projection. _cpuWo is [embDim, qDimL].
        SimdKernels.MatVec(_cpuHidden, _cpuWo[ci].DataPtr, _cpuAttnOut, _embDim, qDimL, _cpuWo[ci].DType);

        if (_hasAttnBias)
            SimdKernels.AddInPlace(_cpuHidden, _cpuBo[ci], _embDim);

        // Post-attention RmsNorm BEFORE residual add.
        if (_cpuPostAttnNorm is not null)
        {
            var paNormW = GetCpuNormWeight(_cpuPostAttnNorm[ci]);
            SimdKernels.RmsNormWide(_cpuHidden, _cpuHidden, paNormW, _embDim, _hp.RmsNormEps);
        }
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);

        // FFN.
        new Span<float>(_cpuHidden, _embDim).CopyTo(new Span<float>(_cpuResidual, _embDim));
        var ffnNormW = GetCpuNormWeight(_cpuFfnNorm[ci]);
        SimdKernels.RmsNormWide(_cpuNormBuf, _cpuHidden, ffnNormW, _embDim, _hp.RmsNormEps);
        SimdKernels.MatVecDual(_cpuFfnGate, _cpuWGate[ci].DataPtr, _cpuFfnUp, _cpuWUp[ci].DataPtr,
            _cpuNormBuf, _intermDim, _embDim, _cpuWGate[ci].DType, _cpuWUp[ci].DType);
        SimdKernels.GeluTanhMul(_cpuFfnGate, _cpuFfnUp, _cpuFfnGate, _intermDim);
        SimdKernels.MatVec(_cpuHidden, _cpuWDown[ci].DataPtr, _cpuFfnGate, _embDim, _intermDim, _cpuWDown[ci].DType);

        // Post-FFN RmsNorm BEFORE residual.
        if (_cpuPostFfwNorm is not null)
        {
            var pfNormW = GetCpuNormWeight(_cpuPostFfwNorm[ci]);
            SimdKernels.RmsNormWide(_cpuHidden, _cpuHidden, pfNormW, _embDim, _hp.RmsNormEps);
        }
        SimdKernels.AddInPlace(_cpuHidden, _cpuResidual, _embDim);

        // PLE injection (after post-FFN residual, before layer_output_scale).
        if (_hp.HasPerLayerTokenEmbd && _cpuInpGate is not null)
            ApplyPerLayerEmbeddingCpu(li);

        // Per-layer scalar output_scale.
        if (_layerOutputScale is not null)
            SimdKernels.ScaleInPlace(_cpuHidden, _layerOutputScale[li], _embDim);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Gemma 4 attention — per-layer head_dim + SWA window + alias support.
    //  Within a position's cache row the KV heads are packed contiguously at the
    //  per-layer head_dim stride (kvHead*hd), matching the contiguous K/V MatVec
    //  write — the cache ROW is _maxHeadDim-wide per head only to size the page;
    //  the trailing slots stay zero when layerHd < _maxHeadDim (12B SWA = 256/512).
    //  Mirrors ForwardPass.Attention (its slotStride is likewise computed-but-unused).
    // ────────────────────────────────────────────────────────────────────

    private void CpuAttentionGemma4(int ci, int effCi, int position, int hd, int qDim, int windowSize, int kvHeads)
    {
        int endSeq = position + 1;
        int startSeq = windowSize > 0 ? Math.Max(0, endSeq - windowSize) : 0;
        int scoreLen = endSeq - startSeq;
        // Gemma 4 uses attn_scale = 1.0 (no 1/sqrt(hd) prefactor).
        float scale = 1.0f;
        // Head→KV-group ratio for the ACTIVE layer (kvHeads, not the model-level
        // _numKvHeads): Gemma 4 12B global layers are MQA (kvHeads=1 → all _numHeads map to
        // KV head 0), SWA layers GQA (kvHeads=8). For non-per-layer models kvHeads ==
        // _numKvHeads so hpkg == _headsPerKvGroup.
        int maxSeqLen = _maxSeqLen; int hpkg = _numHeads / kvHeads;
        int slotStride = _numKvHeads * _maxHeadDim; // cache row width
        var q = _cpuQ; var attnOut = _cpuAttnOut; var scores = _cpuAttnScores;
        var kvCache = _cpuKvCache;
        int hdLocal = hd; int startLocal = startSeq;
        int eff = effCi;

        // Zero the active attnOut window (qDim floats) so the slot reuse from a
        // wider previous layer doesn't bleed into this layer's output projection.
        new Span<float>(attnOut, _numHeads * _maxHeadDim).Clear();

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hdLocal;
            float* outHead = attnOut + h * hdLocal;
            float* headScores = scores + (long)h * maxSeqLen;

            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* kVec = (kvCache.KeyAt(eff, t)) + kvHead * hdLocal;
                headScores[i] = SimdKernels.DotF32(qHead, kVec, hdLocal) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, scoreLen);

            for (int d = 0; d < hdLocal; d++) outHead[d] = 0;

            for (int i = 0; i < scoreLen; i++)
            {
                int t = startLocal + i;
                float* vVec = (kvCache.ValueAt(eff, t)) + kvHead * hdLocal;
                float w = headScores[i];
                if (Fma.IsSupported && hdLocal >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hdLocal; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hdLocal; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
        _ = qDim; _ = slotStride;
    }

    // ────────────────────────────────────────────────────────────────────
    //  PLE pre-pass + injection — CPU side. The PLE table is too large for
    //  GPU residency at Q8 (~4.2 GB on E4B); the projection is small enough
    //  to run on CPU each token and a 256-float slice uploads cheaply for
    //  the GPU half.
    // ────────────────────────────────────────────────────────────────────

    private void BuildPerLayerProjectionsCpu(int token)
    {
        int L = _hp.NumLayers;
        int stackedDim = L * _pleWidth;
        var pleRef = _pleTokenEmbed!.Value;

        int bytesPerRow = (stackedDim / DTypeInfo.BlockSize(pleRef.DType))
                        * DTypeInfo.BytesPerBlock(pleRef.DType);
        byte* rowPtr = pleRef.DataPtr + (long)token * bytesPerRow;
        if (pleRef.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, stackedDim)
                .CopyTo(new Span<float>(_pleRowBuf, stackedDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, _pleRowBuf, stackedDim, pleRef.DType);
        }
        SimdKernels.ScaleInPlace(_pleRowBuf, MathF.Sqrt(_pleWidth), stackedDim);

        // proj_per_layer = per_layer_model_proj @ hidden  → [stackedDim]
        fixed (float* proj = _perLayerModelProjF32!)
        {
            SimdKernels.MatVec(_projPerLayer, (byte*)proj, _cpuHidden,
                stackedDim, _embDim, DType.Float32);
        }
        SimdKernels.ScaleInPlace(_projPerLayer, 1.0f / MathF.Sqrt(_embDim), stackedDim);

        float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);
        var projNormW = GetCpuNormWeight(_perLayerProjNorm!.Value);
        for (int liIdx = 0; liIdx < L; liIdx++)
        {
            float* slice = _projPerLayer + (long)liIdx * _pleWidth;
            SimdKernels.RmsNormWide(slice, slice, projNormW, _pleWidth, _hp.RmsNormEps);
            SimdKernels.AddInPlace(slice, _pleRowBuf + (long)liIdx * _pleWidth, _pleWidth);
            SimdKernels.ScaleInPlace(slice, invSqrt2, _pleWidth);
        }
    }

    private void ApplyPerLayerEmbeddingCpu(int li)
    {
        float* slice = _projPerLayer + (long)li * _pleWidth;
        SimdKernels.MatVec(_pleX, _cpuInpGate![li].DataPtr, _cpuHidden,
            _pleWidth, _embDim, _cpuInpGate[li].DType);
        SimdKernels.GeluTanhMul(_pleX, slice, _pleX, _pleWidth);
        SimdKernels.MatVec(_pleY, _cpuPleProj![li].DataPtr, _pleX,
            _embDim, _pleWidth, _cpuPleProj[li].DType);
        var postW = GetCpuNormWeight(_cpuPlePostNorm![li]);
        SimdKernels.RmsNormWide(_pleY, _pleY, postW, _embDim, _hp.RmsNormEps);
        SimdKernels.AddInPlace(_cpuHidden, _pleY, _embDim);
    }

    // Router + routed-expert FFN on the CPU. Reads `normIn` ([_embDim]), writes the routed
    // MoE output into `hiddenOut` ([_embDim], cleared first). Does NOT include the shared
    // expert (callers add it). Bit-identical to the inlined loop it replaces.
    private void CpuMoeRouted(CpuWeightRef gateInp, CpuWeightRef gateExps,
        CpuWeightRef upExps, CpuWeightRef downExps, float* normIn, float* hiddenOut)
    {
        int numExperts = _hp.NumExperts;
        int numActive  = _hp.NumActiveExperts;

        SimdKernels.MatVec(_cpuRouterLogits, gateInp.DataPtr, normIn, numExperts, _embDim, gateInp.DType);
        if (_hp.UseSigmoidGating) SimdKernels.SigmoidInPlace(_cpuRouterLogits, numExperts);
        else                      SimdKernels.SoftmaxInPlace(_cpuRouterLogits, numExperts);

        Span<int>   selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights   = stackalloc float[numActive];
        SelectTopK(_cpuRouterLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        new Span<float>(hiddenOut, _embDim).Clear();

        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight  = expertWeights[k];

            ExpertMatVec(_cpuExpertGate, gateExps, expertIdx, _expertDim, _embDim, normIn);
            ExpertMatVec(_cpuExpertUp,   upExps,   expertIdx, _expertDim, _embDim, normIn);

            if (_hp.UseSigmoidGating)
            {
                SimdKernels.ScaleInPlace(_cpuExpertGate, weight, _expertDim);
                SimdKernels.ScaleInPlace(_cpuExpertUp,   weight, _expertDim);
                weight = 1.0f;
            }

            SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, _expertDim);
            ExpertMatVecDown(hiddenOut, downExps, expertIdx, _embDim, _expertDim, _cpuExpertGate, weight);
        }
    }

    private void CpuMoeFfn(int ci)
    {
        if (_hasSharedExpert)
        {
            SimdKernels.MatVec(_cpuExpertGate, _cpuWGateShexp![ci].DataPtr, _cpuNormBuf, _expertDim, _embDim, _cpuWGateShexp[ci].DType);
            SimdKernels.MatVec(_cpuExpertUp,   _cpuWUpShexp![ci].DataPtr,   _cpuNormBuf, _expertDim, _embDim, _cpuWUpShexp[ci].DType);
            SimdKernels.SiLuMul(_cpuExpertGate, _cpuExpertUp, _expertDim);
            SimdKernels.MatVec(_cpuSharedOut, _cpuWDownShexp![ci].DataPtr, _cpuExpertGate, _embDim, _expertDim, _cpuWDownShexp[ci].DType);
        }
        CpuMoeRouted(_cpuWGateInp![ci], _cpuWGateExps![ci], _cpuWUpExps![ci], _cpuWDownExps![ci], _cpuNormBuf, _cpuHidden);
        if (_hasSharedExpert)
            SimdKernels.AddInPlace(_cpuHidden, _cpuSharedOut, _embDim);
    }

    private void CpuAttention(int ci, int position)
    {
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int maxSeqLen = _maxSeqLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        var q = _cpuQ; var attnOut = _cpuAttnOut; var scores = _cpuAttnScores;
        var kvCache = _cpuKvCache;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * maxSeqLen;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = kvCache.KeyAt(ci, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = kvCache.ValueAt(ci, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
    }

    private void CpuTqAttention(int ci, int position)
    {
        var tq = _cpuTqKvCache!;
        int seqLen = position + 1;
        int tqLen = tq.GetTqLength(ci);
        int fp32Start = tqLen;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int maxSeqLen = _maxSeqLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        int tqBlkSz = tq.TqBlockSize;
        var q = _cpuQ; var attnOut = _cpuAttnOut; var scores = _cpuAttnScores;
        var rotated = _cpuRotatedQuery; var decomp = _cpuDecompBuf;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * maxSeqLen;
            float* headRotated = rotated + h * hd;
            float* headDecomp = decomp + h * hd;

            var keyCompressor = tq.GetKeyCompressor(ci, kvHead);
            keyCompressor.RotateQuery(
                new ReadOnlySpan<float>(qHead, hd),
                new Span<float>(headRotated, hd));

            // FastScan K-scoring (issue #34) — see ForwardPass.cs for details.
            tq.ComputeKScores(ci, kvHead, headRotated, scale, headScores);

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* kVec = tq.Fp32KeyAt(ci, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            // FastScan V-aggregation (issue #34 Phase 3) — see ForwardPass.cs.
            tq.ComputeVAggregation(ci, kvHead, headScores, outHead);

            for (int t = fp32Start; t < seqLen; t++)
            {
                float* vVec = tq.Fp32ValueAt(ci, t) + kvHead * hd;
                float w = headScores[t];
                if (Fma.IsSupported && hd >= 8)
                {
                    var wv = Vector256.Create(w);
                    int d = 0;
                    for (; d + 8 <= hd; d += 8)
                    {
                        var o = Avx.LoadVector256(outHead + d);
                        var v = Avx.LoadVector256(vVec + d);
                        Avx.Store(outHead + d, Fma.MultiplyAdd(wv, v, o));
                    }
                    for (; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
                else
                {
                    for (int d = 0; d < hd; d++)
                        outHead[d] += w * vVec[d];
                }
            }
        });
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

        SimdKernels.MatVec(_cpuMoeDownTemp, expertData, input, rows, cols, packedTensor.DType);
        for (int i = 0; i < rows; i++)
            output[i] += weight * _cpuMoeDownTemp[i];
    }

    private static void SelectTopK(float* logits, int n, int k,
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

    // ================================================================
    //  Helpers
    // ================================================================

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
    /// pre-fault sweep. Covers per-CPU-layer weights, the CPU-MoE routed experts (the
    /// big cold-start cost, present even with all layers GPU-offloaded), and the
    /// Gemma 4 PLE table. Unresolved slots (null <c>DataPtr</c>) — e.g. Gemma 4
    /// KV-share / k==v layers — are skipped. Biases and QK-norms are excluded: they're
    /// dequantized into separate buffers, not read from the mmap at inference time.</summary>
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
        // Resolve a tensor by name straight from the mmap, tolerating absence — a
        // pre-fault must never make an otherwise-loadable model fail to load.
        void AddByName(string name)
        {
            if (_model.FindTensor(name) is { } info)
                regions.Add(((nint)_model.GetTensorDataPtr(info), info.ByteSize));
        }

        // Embedding/output mmap refs are read at inference only when they're NOT
        // uploaded to VRAM (_gpu* null == cpuEmbeddingOutputOnly). Skipping the
        // GiB-scale token_embd when it lives on the GPU avoids a large pointless read.
        if (_gpuEmbedding is null) Add1(_cpuEmbedding);
        if (_gpuOutputWeight is null)
        {
            Add1(_cpuOutputNorm);
            if (_cpuOutputWeight.DataPtr != _cpuEmbedding.DataPtr) Add1(_cpuOutputWeight); // tied weights alias
        }

        Add(_cpuAttnNorm); Add(_cpuWq); Add(_cpuWk); Add(_cpuWv); Add(_cpuWo);
        Add(_cpuFfnNorm); Add(_cpuPostAttnNorm); Add(_cpuPostFfwNorm);

        if (_isMoE)
        {
            Add(_cpuWGateInp); Add(_cpuWGateExps); Add(_cpuWUpExps); Add(_cpuWDownExps);
            if (_hasSharedExpert) { Add(_cpuWGateShexp); Add(_cpuWUpShexp); Add(_cpuWDownShexp); }
        }
        else
        {
            Add(_cpuWGate); Add(_cpuWUp); Add(_cpuWDown);
        }

        if (_isMoE && _nGpuLayers > 0)
        {
            if (_cpuMoe)
            {
                // CPU-MoE: GPU-trunk routed experts run on the CPU from these cached
                // mmap refs every token (the -g -1 cold-start cost).
                Add(_cpuMoeGateInp); Add(_cpuMoeGateExps); Add(_cpuMoeUpExps); Add(_cpuMoeDownExps);
            }
            else
            {
                // GPU-SLRU MoE: the routed experts aren't cached on the host, but the
                // SLRU streams each one from the mmap on first use. Fault them so the
                // first request's cache fills don't stall (mirrors the Vulkan path).
                for (int li = 0; li < _nGpuLayers; li++)
                {
                    AddByName($"blk.{li}.ffn_gate_exps.weight");
                    AddByName($"blk.{li}.ffn_up_exps.weight");
                    AddByName($"blk.{li}.ffn_down_exps.weight");
                }
            }
        }

        // Gemma 4 PLE: the GiB-scale per-layer token-embedding table + per-layer projections.
        if (_pleTokenEmbed is { } ple) Add1(ple);
        if (_perLayerProjNorm is { } pln) Add1(pln);
        Add(_cpuInpGate); Add(_cpuPleProj); Add(_cpuPlePostNorm);

        return regions;
    }

    private float* LoadCpuBias(string name, int count)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing bias tensor: {name}");
        var data = _model.GetTensorData(info);
        var buf = Alloc(count);
        if (info.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        return buf;
    }

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
    /// Reject hybrid splits that would require cross-tier KV reads for Gemma 4's
    /// shared_kv_layers tail. The shared-KV source layers (typically the last two
    /// own-KV layers, indices 22 and 23 on E4B with shared_kv_layers=18) MUST live
    /// on the CPU side so the shared layers (24..41, all on CPU) can read them.
    /// </summary>
    private static void Gemma4ValidateHybridSplit(ModelHyperparams hp, int nGpuLayers)
    {
        if (hp.KvSourceLayer is not { } ksl) return;
        int minSrc = int.MaxValue;
        for (int i = 0; i < hp.NumLayers; i++)
            if (ksl[i] >= 0 && ksl[i] < minSrc) minSrc = ksl[i];
        if (minSrc == int.MaxValue) return; // no aliased layers
        if (nGpuLayers > minSrc)
            throw new NotSupportedException(
                $"Gemma 4 hybrid split with -g {nGpuLayers} would place shared-KV source layer " +
                $"{minSrc} on the GPU while its dependent shared-KV layers run on the CPU; " +
                "cross-tier KV reads are not implemented. " +
                $"Use -g <= {minSrc} (own-KV sources stay on CPU) or -g {hp.NumLayers} " +
                "(full-GPU via CudaForwardPass).");
    }

    private float* GetCpuNormWeight(in CpuWeightRef tensor)
    {
        if (_cpuNormCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;
        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);
        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);
        _cpuNormCache[tensor.Name] = (nint)buf;
        return buf;
    }

    private void CpuEmbedToken(int token, float* dest)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_cpuEmbedding.DType))
                        * DTypeInfo.BytesPerBlock(_cpuEmbedding.DType);
        byte* rowPtr = _cpuEmbedding.DataPtr + (long)token * bytesPerRow;
        if (_cpuEmbedding.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dest, _embDim));
        }
        else
        {
            Dequantize.ToFloat32(
                new ReadOnlySpan<byte>(rowPtr, bytesPerRow),
                new Span<float>(dest, _embDim),
                _cpuEmbedding.DType,
                _embDim);
        }
    }

    private void ComputeCpuOutput()
    {
        var outputNorm = GetCpuNormWeight(_cpuOutputNorm);
        SimdKernels.RmsNorm(_cpuNormBuf, _cpuHidden, outputNorm, _embDim, _hp.RmsNormEps);
        fixed (float* logits = _logitsBuf)
            SimdKernels.MatVec(logits, _cpuOutputWeight.DataPtr, _cpuNormBuf, _hp.VocabSize, _embDim, _cpuOutputWeight.DType);
    }

    private static bool ShouldKeepFixedWeightsOnCpu(GgufTensorInfo embedding, GgufTensorInfo? output)
    {
        const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;
        // Embedding goes through F32 dequant for any non-Q4_K format (no Q6_K embed
        // shader exists), so its post-upload size can be much larger than the raw
        // GGUF byte size — use the embedding-aware estimator here.
        if (EstimateGpuEmbeddingBytes(embedding) > maxStorageBufferBytes)
            return true;
        if (output is not null && EstimateGpuTensorBytes(output.Value) > maxStorageBufferBytes)
            return true;
        return false;
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K || tensor.DType == DType.Q6_K)
            return (tensor.ByteSize + 3) & ~3L;

        return tensor.ElementCount * sizeof(float);
    }

    // Embedding-table upload mirrors GpuForwardPass: only Q4_K has a quantized
    // EmbedLookup shader, so any other dtype must be dequantized to F32 first
    // (otherwise the F32 EmbedLookup shader reinterprets raw quantized bytes
    // as floats — producing NaN/huge values, the cause of issues #3/#19).
    private static long EstimateGpuEmbeddingBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Q4_K)
            return (tensor.ByteSize + 3) & ~3L;
        return tensor.ElementCount * sizeof(float);
    }

    private Tensor UploadEmbeddingWeight(string name, out bool isQuantized)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // exact=true: the embedding table is permanent for the session; skip the
        // pool round-up (a Q4_K embed near 715 MiB would otherwise inflate to 1 GiB).
        Tensor result;
        if (info.DType == DType.Q4_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Q4_K;
            isQuantized = true;
        }
        else
        {
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            if (info.DType == DType.Float32)
                MemoryMarshal.Cast<byte, float>(data).CopyTo(f32);
            else
                Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
            isQuantized = false;
        }
        return result;
    }

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
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_0)
        {
            // Q4_0 blocks are 18 bytes — not a multiple of 4 — so the byte→float
            // pack used for Q4_K/Q6_K below would truncate. Upload the raw bytes
            // verbatim (issue #124, Gemma 4 12B QAT).
            result = _gpu.UploadRaw(data, TensorShape.D1(data.Length), info.DType, exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount), exact: true);
            _gpuWeightDTypes[result.Handle] = info.DType;
        }
        else
        {
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count), exact: true);
            _gpuWeightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    private void GpuDenseFfn(int layer)
    {
        GpuMatMul(_gpuFfnGate, _gpuWGate[layer], _gpuNormBuf);
        GpuMatMul(_gpuFfnUp, _gpuWUp[layer], _gpuNormBuf);
        _gpu.RecordBarrier();

        _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
        _gpu.RecordBarrier();

        GpuMatMul(_gpuHidden, _gpuWDown[layer], _gpuFfnGate);
    }

    private void GpuMoeFfn(int layer)
    {
        // SLRU-streamed variant (mirror of CudaHybridGdnForwardPass.GpuMoeFfn): each
        // selected routed expert is fetched via _expertSlotManager.GetOrLoad, which
        // returns a cached slot or synchronously uploads-then-caches on miss. CUDA's
        // implicit stream ordering removes the explicit barrier vocabulary the Vulkan
        // version needs.
        int numActive = _hp.NumActiveExperts;

        GpuMatMul(_gpuRouterLogits!, _gpuWGateInp![layer], _gpuNormBuf);
        if (_hp.UseSigmoidGating) _gpu.Sigmoid(_gpuRouterLogits!);
        else                       _gpu.Softmax(_gpuRouterLogits!);

        _gpu.Download(_gpuRouterLogits!, _gpuRouterBuf!);
        _gpu.Synchronize();

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_gpuRouterBuf!, numActive, selectedExperts, expertWeights, _hp.NormalizeMoeTopKWeights);

        if (_hasSharedExpert)
        {
            GpuMatMul(_gpuFfnGate, _gpuWGateShexp![layer], _gpuNormBuf);
            GpuMatMul(_gpuFfnUp,   _gpuWUpShexp![layer],   _gpuNormBuf);
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuMoeSharedOut!, _gpuWDownShexp![layer], _gpuFfnGate);
        }

        _gpu.Clear(_gpuHidden);

        for (int i = 0; i < numActive; i++)
        {
            int expertIdx = selectedExperts[i];
            float expertWeight = expertWeights[i];
            var slot = _expertSlotManager!.GetOrLoad(layer, expertIdx);

            GpuMatMul(_gpuFfnGate, slot.Gate, _gpuNormBuf);
            GpuMatMul(_gpuFfnUp,   slot.Up,   _gpuNormBuf);

            if (_hp.UseSigmoidGating)
            {
                _gpu.ScaleInPlace(_gpuFfnGate, expertWeight);
                _gpu.ScaleInPlace(_gpuFfnUp,   expertWeight);
            }

            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuMoeExpertOut!, slot.Down, _gpuFfnGate);

            if (_hp.UseSigmoidGating)
                _gpu.AddInPlace(_gpuHidden, _gpuMoeExpertOut!);
            else
                _gpu.AddScaledInPlace(_gpuHidden, _gpuMoeExpertOut!, expertWeight);
        }

        if (_hasSharedExpert)
            _gpu.AddInPlace(_gpuHidden, _gpuMoeSharedOut!);
    }

    // CPU-MoE trunk path (issue #215): the GPU computes attention/norms (already done by the
    // caller, leaving the post-FFN-norm hidden in _gpuNormBuf) plus the shared expert; the
    // routed experts run on the CPU. Mirrors CudaHybridGdnForwardPass's CPU-MoE per-layer
    // path. Writes the MoE output into _gpuHidden (no residual add — the caller does that),
    // exactly like GpuMoeFfn.
    private void GpuTrunkCpuMoeFfn(int layer)
    {
        // Shared expert on the GPU first, overlapping the CPU routed work below. (Targets
        // Qwen3-Coder / OLMoE have no shared expert, so this branch is dead there; kept for
        // qwen2moe-class models.)
        if (_hasSharedExpert)
        {
            GpuMatMul(_gpuFfnGate, _gpuWGateShexp![layer], _gpuNormBuf);
            GpuMatMul(_gpuFfnUp,   _gpuWUpShexp![layer],   _gpuNormBuf);
            _gpu.SiLuMul(_gpuFfnGate, _gpuFfnUp);
            GpuMatMul(_gpuMoeSharedOut!, _gpuWDownShexp![layer], _gpuFfnGate);
        }

        // Download the normed hidden, run router + routed experts on the CPU, upload back.
        _gpu.Download(_gpuNormBuf, (nint)_cpuNormBuf, _embDim);
        CpuMoeRouted(_cpuMoeGateInp![layer], _cpuMoeGateExps![layer],
                     _cpuMoeUpExps![layer], _cpuMoeDownExps![layer], _cpuNormBuf, _cpuHidden);
        _gpu.UploadInto(_gpuHidden, (nint)_cpuHidden, _embDim);

        if (_hasSharedExpert)
            _gpu.AddInPlace(_gpuHidden, _gpuMoeSharedOut!);
    }

    // Routed-expert weights are uploaded lazily by CudaExpertSlotManager on cache
    // miss, not eagerly here. (The shared expert and router stay resident.)

    /// <summary>
    /// On-VRAM bytes for one expert's three weight tensors (gate + up + down), used to size the
    /// SLRU slot capacity. Sums each role's MAX per-expert footprint over all layers via
    /// <see cref="CudaExpertSlotManager.MaxRoleExpertBytes"/>, so this prices EXACTLY what the
    /// exact-size slab allocates (issue #216). Sizing from blk.0 alone under-counts mixed-quant
    /// (K_M/UD) roles — the planner would then derive more slots than the slab fits and
    /// over-commit VRAM. Q4_K/Q5_K/Q6_K stay raw; other dtypes expand to F32.
    /// </summary>
    private long PerExpertBytes()
    {
        long Max(string role, int rows, int cols) =>
            CudaExpertSlotManager.MaxRoleExpertBytes(_model, _hp.NumLayers, role, rows, cols);
        return Max("ffn_gate_exps", _expertDim, _embDim)
             + Max("ffn_up_exps",   _expertDim, _embDim)
             + Max("ffn_down_exps", _embDim,    _expertDim);
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

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        _gpu.MatMul(output, weights, input,
            _gpuWeightDTypes.TryGetValue(weights.Handle, out var dt) ? dt : DType.Float32);
    }

    // Compute-shader-based buffer copies. Critical: `vkCmdCopyBuffer` runs in the Transfer
    // pipeline stage, which is NOT synchronized with the rest of this forward pass — every
    // RecordBarrier() in this file is a Compute→Compute memory barrier and does nothing for
    // transfer-stage writes. Using a compute copy keeps everything in the compute stage so
    // those barriers are correct.
    private void CopyGpuBuffer(Tensor dst, Tensor src) => _gpu.RecordComputeCopy(dst, src);

    private void CopyGpuBufferRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
    {
        _gpu.RecordComputeCopyRegion(dst, dstOffsetBytes, src, srcOffsetBytes, sizeBytes);
    }

    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    private static void PerChannelRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight + h * headDim, headDim, eps);
    }

    private static void PerHeadPureRmsNorm(float* data, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.PureRmsNorm(data + h * headDim, data + h * headDim, headDim, eps);
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)count, (nuint)sizeof(float));

    // ================================================================
    //  Disposal
    // ================================================================

    public void Dispose()
    {
        _gpu.Free(_gpuHidden); _gpu.Free(_gpuResidual); _gpu.Free(_gpuNormBuf);
        _gpu.Free(_gpuQ); _gpu.Free(_gpuK); _gpu.Free(_gpuV); _gpu.Free(_gpuAttnOut);
        _gpu.Free(_gpuFfnGate); _gpu.Free(_gpuFfnUp); _gpu.Free(_gpuLogits);
        if (_splitKvPartialO is { } spo) _gpu.Free(spo);
        if (_splitKvPartialMeta is { } spm) _gpu.Free(spm);
        if (_gpuRouterLogits is not null) _gpu.Free(_gpuRouterLogits);
        if (_gpuMoeSharedOut is not null) _gpu.Free(_gpuMoeSharedOut);
        if (_gpuMoeExpertOut is not null) _gpu.Free(_gpuMoeExpertOut);
        _gpu.Free(_pinnedHidden);

        // Issue #123: batched-trunk prefill scratch.
        FreeBatchedTrunkScratch();
        if (_pinnedHiddenAll is { } pha) { _gpu.Free(pha); _pinnedHiddenAll = null; }
        // Issue #218: batched CPU-embed staging buffer.
        if (_pinnedEmbedAll is { } pea) { _gpu.Free(pea); _pinnedEmbedAll = null; }

        for (int i = 0; i < _nGpuLayers; i++)
        {
            _gpu.Free(_gpuAttnNorm[i]); _gpu.Free(_gpuFfnNorm[i]);
            // _gpuWv[i] is null on Gemma 4 12B global k_eq_v layers (no attn_v uploaded).
            _gpu.Free(_gpuWq[i]); _gpu.Free(_gpuWk[i]);
            if (_gpuWv[i] is not null) _gpu.Free(_gpuWv[i]);
            _gpu.Free(_gpuWo[i]);
            if (_isMoE)
            {
                // _gpuWGateInp[i] is null on CPU-MoE layers (router runs on CPU; not uploaded).
                if (_gpuWGateInp![i] is not null) _gpu.Free(_gpuWGateInp[i]);
                // Routed-expert tensors are owned by _expertSlotManager (freed in its Dispose).
                if (_hasSharedExpert)
                {
                    _gpu.Free(_gpuWGateShexp![i]);
                    _gpu.Free(_gpuWUpShexp![i]);
                    _gpu.Free(_gpuWDownShexp![i]);
                }
            }
            else
            {
                _gpu.Free(_gpuWGate[i]); _gpu.Free(_gpuWUp[i]); _gpu.Free(_gpuWDown[i]);
            }
            if (_gpuPostAttnNorm is not null) _gpu.Free(_gpuPostAttnNorm[i]);
            if (_gpuPostFfwNorm is not null)  _gpu.Free(_gpuPostFfwNorm[i]);
            if (_gpuInpGate is not null)
            {
                _gpu.Free(_gpuInpGate[i]);
                _gpu.Free(_gpuPleProj![i]);
                _gpu.Free(_gpuPlePostNorm![i]);
            }
            // KV cache: shared-KV layers alias another layer's handle (Gemma 4); skip
            // the Free here, the owning layer releases it.
            if (!_gpuKvAliasedLayers.Contains(i))
            {
                _gpu.Free(_gpuKCache[i]); _gpu.Free(_gpuVCache[i]);
            }
            if (_tqEnabled)
            {
                _gpu.Free(_gpuTqKCache![i]);
                _gpu.Free(_gpuTqVCache![i]);
                _gpu.Free(_gpuSignPatterns![i]);
            }

            if (_hasAttnBias)
            { _gpu.Free(_gpuBq![i]); _gpu.Free(_gpuBk![i]); _gpu.Free(_gpuBv![i]); _gpu.Free(_gpuBo![i]); }
            if (_hasQkNorm && !_hp.UseL2QkNorm)
            { _gpu.Free(_gpuQNorm![i]); _gpu.Free(_gpuKNorm![i]); }
        }
        if (_tqEnabled)
        {
            _gpu.Free(_gpuCodebook!);
            _gpu.Free(_gpuBoundaries!);
            _gpu.Free(_gpuRotatedQ!);
            _gpu.Free(_gpuEvictK!);
            _gpu.Free(_gpuEvictV!);
        }
        _gpu.Free(_gpuAttnScoresScratch);
        if (_gpuOutputNorm is not null)
            _gpu.Free(_gpuOutputNorm);
        if (_gpuOutputWeight is not null && _gpuOutputWeight.Handle != _gpuEmbedding?.Handle)
            _gpu.Free(_gpuOutputWeight);
        if (_gpuEmbedding is not null)
            _gpu.Free(_gpuEmbedding);

        NativeMemory.Free(_cpuHidden); NativeMemory.Free(_cpuResidual); NativeMemory.Free(_cpuNormBuf);
        NativeMemory.Free(_cpuQ); NativeMemory.Free(_cpuK); NativeMemory.Free(_cpuV); NativeMemory.Free(_cpuAttnOut);
        NativeMemory.Free(_cpuFfnGate); NativeMemory.Free(_cpuFfnUp); NativeMemory.Free(_cpuAttnScores);
        NativeMemory.Free(_ropeCosTable); NativeMemory.Free(_ropeSinTable);
        if (_ropeCosTableSwa != null) NativeMemory.Free(_ropeCosTableSwa);
        if (_ropeSinTableSwa != null) NativeMemory.Free(_ropeSinTableSwa);
        if (_pleRowBuf != null)    NativeMemory.Free(_pleRowBuf);
        if (_projPerLayer != null) NativeMemory.Free(_projPerLayer);
        if (_pleX != null)         NativeMemory.Free(_pleX);
        if (_pleY != null)         NativeMemory.Free(_pleY);
        if (_gpuProjSliceViews is { } psv) foreach (var v in psv) _gpu.Free(v);
        if (_gpuProjPerLayerGpu is { } projUp) _gpu.Free(projUp);
        if (_gpuPleX is { } pleX)  _gpu.Free(pleX);
        if (_gpuPleY is { } pleY)  _gpu.Free(pleY);
        if (_gpuRopeFreqs is { } rfFree) _gpu.Free(rfFree);
        if (_cpuRouterLogits != null) NativeMemory.Free(_cpuRouterLogits);
        if (_cpuSharedOut != null) NativeMemory.Free(_cpuSharedOut);
        if (_cpuExpertGate != null) NativeMemory.Free(_cpuExpertGate);
        if (_cpuExpertUp != null) NativeMemory.Free(_cpuExpertUp);
        if (_cpuMoeDownTemp != null) NativeMemory.Free(_cpuMoeDownTemp);
        if (_cpuRotatedQuery != null) NativeMemory.Free(_cpuRotatedQuery);
        if (_cpuDecompBuf != null) NativeMemory.Free(_cpuDecompBuf);
        _cpuKvCache.Dispose();
        _cpuTqKvCache?.Dispose();
        if (_expertSlotManager is not null)
        {
            // SHARPI_EXPERT_STATS=<path>: dump SLRU hit rate + top experts per layer
            // (parity with CudaHybridGdnForwardPass).
            var statsPath = Environment.GetEnvironmentVariable("SHARPI_EXPERT_STATS");
            if (!string.IsNullOrEmpty(statsPath))
            {
                // Diagnostic-only: a write failure must never skip the slot manager's
                // Dispose below (which frees GPU tensors), so swallow + log.
                try
                {
                    using var w = new StreamWriter(statsPath);
                    _expertSlotManager.Profiler.PrintStats(w);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[CudaHybridForwardPass] Failed to write expert stats to {statsPath}: {ex.Message}");
                }
            }
            _expertSlotManager.Dispose();
        }
    }
}
