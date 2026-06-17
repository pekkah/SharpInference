using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Engine;

/// <summary>
/// CPU forward pass for hybrid Gated-DeltaNet + sparse-attention + MoE models
/// (currently <c>qwen35moe</c>, e.g. Qwen3.6-35B-A3B).
///
/// Layer dispatch follows <see cref="ModelHyperparams.LayerTypes"/>:
/// - <see cref="LayerType.GatedDeltaNet"/> blocks run the GDN recurrent kernel from
///   <see cref="GdnKernels"/> with per-sequence state in <see cref="GdnStateCache"/>.
/// - <see cref="LayerType.Attention"/> blocks run full softmax attention with a
///   GLU-gated Q split, partial NEOX RoPE (first <see cref="ModelHyperparams.RopeDim"/>
///   dims), and the standard <see cref="PagedKvCache"/>.
///
/// All 40 trunk layers (10 attn + 30 GDN) share the same MoE FFN structure: a
/// 256-expert top-8 router PLUS a shared expert gated by a per-token sigmoid
/// scalar (the <c>ffn_gate_inp_shexp</c> weight).
///
/// This is the v1 reference path: no batched prefill (sequential T-step decode),
/// no speculative decoding (GDN state is destructive), no TurboQuant (KV cache
/// only covers attention layers, which is a minority of layers anyway).
/// </summary>
/// <remarks>
/// Layout notes (verified against llama.cpp <c>src/models/qwen35moe.cpp</c>):
/// - <c>attn_q.weight</c> per-head GLU layout: <c>[Q[256], G[256]]</c> per head,
///   16 heads → total 8192 (per-head GLU gate is multiplied AFTER attention as
///   <c>attn_out *= sigmoid(gate)</c>, NOT SiLU).
/// - <c>attn_qkv.weight</c> channel order in the GDN block: <c>Q‖K‖V</c> at
///   offsets <c>0 / KeyDim / 2*KeyDim</c> = <c>0 / 2048 / 4096</c>.
/// - <c>ssm_conv1d.weight</c> GGUF layout is <c>[ne0=4, ne1=8192]</c> = channels
///   row-major (<c>w[c*4 + k]</c>); we transpose to <c>[kernel, channels]</c> at
///   load time to feed <see cref="GdnKernels.CausalDepthwiseConv1dDecode"/>.
/// - Partial RoPE: <c>ropeDim=64</c> NEOX rotation, headDim=256 for attention.
/// - <c>ffn_gate_inp_shexp.weight</c> is a <c>[n_embd]</c> vector → scalar per
///   token via dot product → sigmoid → per-channel multiply on shared expert out.
/// </remarks>
public sealed unsafe class HybridGdnForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;
    private readonly GdnConfig _gdn;
    private readonly PagedKvCache _kvCache;
    private readonly GdnStateCache _gdnStateCache;
    private readonly int _ctxLen;

    // Norm-weight / small-F32 cache.
    private readonly Dictionary<string, nint> _normCache = new();

    // ── Common scratch ─────────────────────────────────────────────────
    private readonly float* _hidden;        // [embDim]
    private readonly float* _residual;      // [embDim]
    private readonly float* _normBuf;       // [embDim]
    private readonly float* _logits;        // [vocabSize]


    // ── Attention scratch (full-attn layers) ───────────────────────────
    private readonly float* _qGate;         // [numHeads * headDim * 2] = 8192 — interleaved Q‖gate per head
    private readonly float* _q;              // [numHeads * headDim]    = 4096
    private readonly float* _gate;           // [numHeads * headDim]    = 4096 (pre-sigmoid)
    private readonly float* _k;              // [numKvHeads * headDim]  = 512
    private readonly float* _v;              // [numKvHeads * headDim]  = 512
    private readonly float* _attnOut;        // [numHeads * headDim]    = 4096
    private readonly float* _attnScores;     // [numHeads * ctxLen]

    // ── GDN scratch ────────────────────────────────────────────────────
    private readonly float* _qkv;            // [ConvChannels]   = 8192 — pre-conv
    private readonly float* _qkvConv;        // [ConvChannels]   = 8192 — post-conv
    private readonly float* _z;              // [ValueDim]       = 4096
    private readonly float* _qVHeads;        // [NumVHeads*HeadDim] = 4096 — Q broadcast K→V heads
    private readonly float* _kVHeads;        // [NumVHeads*HeadDim] = 4096
    private readonly float* _alpha;          // [NumVHeads]      = 32
    private readonly float* _beta;           // [NumVHeads]      = 32
    private readonly float* _gdnOut;         // [ValueDim]       = 4096

    // ── MoE scratch (only allocated when _hp.IsMoE) ────────────────────
    private readonly float* _routerLogits;   // [NumExperts]      = 256
    private readonly float* _sharedOut;      // [embDim]          = 2048
    private readonly float* _expertGate;     // [ExpertIntermDim] = 512 — used for the shared expert
    private readonly float* _expertUp;       // [ExpertIntermDim] = 512 — used for the shared expert
    // Batched-routed-expert buffers: gate/up rows for all 8 active experts laid out as
    // [numActive × expertDim]. Together with the down sweep, this folds the per-expert
    // 24 Parallel.For sweeps per layer into 2 (one combined gate+up sweep, one combined
    // down + weighted-accumulate sweep) — amortising TPL barrier overhead over much
    // larger work units. Mirrors CudaHybridGdnForwardPass.CpuMoeFfn.
    private readonly float* _expertGateAll;  // [numActive × ExpertIntermDim]
    private readonly float* _expertUpAll;    // [numActive × ExpertIntermDim]
    // Q8_KS prepacked inputs (per-32-element sub-block scales, issue #107) for
    // the SHARPI_Q3K_Q8K / SHARPI_Q8_0_Q8K routed-expert dot kernels (allocated
    // when either latch is on). Phase A consumes _normInQ8K (one row, rewritten
    // per MoeFfnCore call when gate or up is Q3_K); Phase C consumes numActive
    // contiguous expertDim-sized slices in _expertGateAllQ8K (rewritten per
    // call when down is Q3_K). BatchForward2 sequences token-1 and token-2
    // MoeFfnCore calls back-to-back, so the single shared scratch is rewritten
    // before it's read on each side — no t1/t2 collision.
    private readonly byte* _normInQ8K;
    private readonly byte* _expertGateAllQ8K;
    private readonly int _expertGateAllQ8KStride;

    // ── Dense FFN scratch (only allocated when !_hp.IsMoE, e.g. qwen35 27B) ──
    private readonly float* _ffnGate;        // [IntermediateDim]
    private readonly float* _ffnUp;          // [IntermediateDim]
    private readonly int _intermDim;         // hp.IntermediateDim (dense FFN); 0 when MoE

    // ── Token-2 scratch for MTP batched verify (issue #30) — allocated only when
    //     _hasMtp && !_hp.IsMoE. Mirrors the single-token scratch above; sized to
    //     give BatchForward2 a parallel residual stream for the second token. ──
    private readonly float* _hidden2;        // [embDim]
    private readonly float* _residual2;      // [embDim]
    private readonly float* _normBuf2;       // [embDim]
    private readonly float* _ffnGate2;       // [intermDim]
    private readonly float* _ffnUp2;         // [intermDim]
    private readonly float* _logits2;        // [vocabSize]
    // Lane-3/4 batched-verify scratch (issue #209): DenseFfn4 / the quad lm_head dot
    // amortize one CPU mmap weight read across four draft tokens via MatVec4In, so they
    // need four distinct gate/up FFN slabs and four distinct vocab-sized logits sinks.
    private readonly float* _ffnGate3;       // [intermDim]
    private readonly float* _ffnUp3;         // [intermDim]
    private readonly float* _ffnGate4;       // [intermDim]
    private readonly float* _ffnUp4;         // [intermDim]
    private readonly float* _logits3;        // [vocabSize]
    private readonly float* _logits4;        // [vocabSize]

    // Per-token-boundary GDN snapshot ring used by the batched verify paths
    // (issues #30 / #207-goal-4). Slot j holds every GDN layer's state AFTER the
    // batch's token j (j ∈ [0, k-2]), so a rejection at draft position j+1 can
    // restore via RestoreBatchSnapshot(startPos + j + 1). Slot layout:
    //   offset(slot, gdnIdx) = slot × NumGdnLayers × LayerSnapshotBytes
    //                        + gdnIdx × LayerSnapshotBytes
    // BatchForward2 (the legacy 2-token path) writes slot 0 only. The buffer
    // starts at 1 slot (constructor) and grows lazily in BatchVerify.
    private byte* _batchSnapshotBuf;
    private long _batchSnapshotCap;
    private bool _batchSnapshotValid;
    private int _batchSnapshotSlots;   // ring slots currently allocated
    private int _batchStartPos;        // startPos of the most recent batched verify
    private int _batchK;               // token count of the most recent batched verify

    // Batched-verify residual streams [k × embDim] (lazily grown; issue #30 k-token
    // generalization). BatchForward2 keeps its dedicated _hidden2/... pair.
    private float* _bvHiddenAll;
    private float* _bvResidAll;
    private float* _bvNormAll;
    private int _bvCap;

    // MTP block-out hidden of the most recent MtpForward (pre-shared-head-norm),
    // used as the next chained draft's prevHidden (issue #30 multi-token drafting).
    private float* _mtpSelfHidden;

    // Max tokens per BatchVerify call (= 1 + max MTP draft chain length); the host
    // snapshot ring grows lazily to k-1 slots. Instance-resolved at construction so
    // tests can override per instance; the knob semantics live in one place
    // (GdnStateCache.ResolveMtpBatchMax) shared with the CUDA pass.
    private readonly int _mtpBatchMax = GdnStateCache.ResolveMtpBatchMax();

    // ── Dimensions (cached) ────────────────────────────────────────────
    private readonly int _embDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headsPerKvGroup;
    private readonly int _ropeDim;
    private readonly int _ropeHalfDim;
    private readonly int _gdnHeadDim;
    private readonly int _gdnNumVHeads;
    private readonly int _gdnNumKHeads;
    private readonly int _gdnKvRepeat;       // = NumVHeads / NumKHeads
    private readonly int _gdnValueDim;
    private readonly int _gdnKeyDim;
    private readonly int _gdnConvChannels;
    private readonly int _gdnConvKernel;

    // ── Tensor refs (Q/K/V/O for attn, qkv/gate/out for GDN, MoE for both) ──
    private readonly TensorRef _embTensor;
    private readonly TensorRef[] _attnNorm;        // [L]
    private readonly TensorRef[] _postAttnNorm;    // [L] — pre-MoE
    private readonly TensorRef[] _wGateInp;        // [L] — router (F32, [embDim, numExperts])
    private readonly TensorRef[] _wGateShexp;      // [L]
    private readonly TensorRef[] _wUpShexp;        // [L]
    private readonly TensorRef[] _wDownShexp;      // [L]
    private readonly TensorRef[] _wGateExps;       // [L] — packed [numExperts, expertDim, embDim]
    private readonly TensorRef[] _wUpExps;         // [L]
    private readonly TensorRef[] _wDownExps;       // [L]

    // Full-attention layer tensors (TensorRef[L]; only entries at attention layers are valid)
    private readonly TensorRef[] _wQGate;          // [L] — attn_q (GLU-gated, output 8192)
    private readonly TensorRef[] _wK;              // [L]
    private readonly TensorRef[] _wV;              // [L]
    private readonly TensorRef[] _wO;              // [L] — attn_output

    // Per-head Q/K norm weights for attention (preloaded F32, shared headDim-wide gain)
    private readonly float*[] _qNorm;              // [L][headDim]
    private readonly float*[] _kNorm;              // [L][headDim]

    // GDN layer tensors (TensorRef[L]; only entries at GDN layers are valid)
    private readonly TensorRef[] _wQkv;            // [L] — attn_qkv (output ConvChannels=8192)
    private readonly TensorRef[] _wZGate;          // [L] — attn_gate (output ValueDim=4096)
    private readonly TensorRef[] _ssmOut;          // [L] — ssm_out (output embDim, input ValueDim)
    private readonly TensorRef[] _ssmAlpha;        // [L] — F32 [embDim, NumVHeads]
    private readonly TensorRef[] _ssmBeta;         // [L] — F32 [embDim, NumVHeads]

    // Small F32 GDN weights (preloaded into private buffers, decode-once).
    private readonly float*[] _ssmConv1d;          // [L][kernel * channels] — TRANSPOSED from GGUF [c,k] to [k,c]
    private readonly float*[] _ssmA;               // [L][NumVHeads]
    private readonly float*[] _ssmDtBias;          // [L][NumVHeads]
    private readonly float*[] _ssmNormW;           // [L][HeadDim]

    // Shared-expert per-token gate vector (F32 [embDim]).
    private readonly float*[] _wGateInpShexp;      // [L]

    // Dense FFN tensors (TensorRef[L]; populated only when !_hp.IsMoE). The MoE arrays
    // above remain null in that case. Mirrors ForwardPass._wGate / _wUp / _wDown.
    private readonly TensorRef[] _wFfnGate;        // [L] — ffn_gate.weight (dense)
    private readonly TensorRef[] _wFfnUp;          // [L] — ffn_up.weight (dense)
    private readonly TensorRef[] _wFfnDown;        // [L] — ffn_down.weight (dense)

    // RoPE tables (sized by ropeDim/2, not headDim/2).
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;

    // Diagnostic: per-layer activation trace (env: SHARPI_TRACE_LAYERS=1). Emits one line
    // per block plus embedding/pre-logits + top-5 logits to stderr. Modelled on
    // SHARPI_TRACE_NORMS in ForwardPass.cs.
    private static readonly bool _traceLayers =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_LAYERS") == "1";

    // Per-layer logit probe (env: SHARPI_PROBE_LOGITS=1). For each trunk-layer
    // residual output, projects through output_norm + lm_head and emits the
    // rank/logit of a few diagnostic token ids (set via SHARPI_PROBE_IDS as
    // comma-separated). Allocates an extra embDim scratch + the existing logit
    // buffer is reused.
    private static readonly bool _probeLogits =
        Environment.GetEnvironmentVariable("SHARPI_PROBE_LOGITS") == "1";
    private static readonly int[] _probeIds = ParseProbeIds();
    private static int[] ParseProbeIds()
    {
        var s = Environment.GetEnvironmentVariable("SHARPI_PROBE_IDS");
        if (string.IsNullOrEmpty(s)) return new[] { 198, 248046, 248045, 271 };
        var parts = s.Split(',');
        var ids = new List<int>(parts.Length);
        foreach (var p in parts) if (int.TryParse(p.Trim(), out var id)) ids.Add(id);
        return ids.ToArray();
    }

    // Bisection-only env vars: zero out one block type's contribution to localize a bug.
    // Default off; leaving in for future parity work.
    private static readonly bool _bypassGdn =
        Environment.GetEnvironmentVariable("SHARPI_BYPASS_GDN") == "1";
    private static readonly bool _bypassAttn =
        Environment.GetEnvironmentVariable("SHARPI_BYPASS_ATTN") == "1";
    private static readonly bool _bypassMoe =
        Environment.GetEnvironmentVariable("SHARPI_BYPASS_MOE") == "1";

    // Chunk-parallel GDN prompt prefill (FlashQLA-style chunk_gated_delta_rule):
    // Prefill resolves the GDN recurrence over the whole prompt via
    // GdnKernels.GdnRecurrenceChunkedPrefill instead of the per-token scan, ~1.3×
    // faster end-to-end on the CPU backend (measured: qwen35moe 8.3→11.0 t/s prefill).
    // DEFAULT ON for CPU (SHARPI_GDN_CHUNKED_PREFILL=0 to disable) — BUT the Prefill
    // gate below additionally excludes models with a native MTP head: the chunked
    // form is numerically equal to the scan only up to FP reduction order, and on the
    // knife-edge "thinking-or-not" boundary token of the Qwen3.6-MTP models that ULP
    // difference flips the trajectory off llama.cpp's (MtpDecoder_GreedyParity_LlamaCpp).
    // MTP models therefore stay on the byte-exact per-token scan. Exposed as a settable
    // property so a parity test can toggle it without env vars.
    public static bool GdnChunkedPrefillEnabled { get; set; } =
        Environment.GetEnvironmentVariable("SHARPI_GDN_CHUNKED_PREFILL") != "0";

    // Q3_K_Q8K / Q8_0_Q8K kernel gates. Auto-on when the model has routed-expert
    // weights in that dtype (APEX mixed-precision tier — e.g. Carnice).
    // SHARPI_Q3K_Q8K / SHARPI_Q8_0_Q8K = "1" or "0" override. Mirrors the
    // CudaHybridGdnForwardPass latches; see the comment there for the
    // BatchForward2 / per-call prepack lifetime notes.
    private readonly bool _q3kQ8KEnabled;
    private readonly bool _q8_0Q8KEnabled;

    // Output projection.
    private readonly TensorRef _outputNorm;
    private readonly TensorRef _outputWeight;

    // ── MTP / NEXTN head (Multi-Token Prediction) — issue #25 ──────────
    // Loaded when hp.NumMtpLayers > 0. Lives at GGUF block index NumLayers
    // (= blk.{NumLayers}). One standard attention+FFN block plus four
    // nextn.* tensors. Output uses the shared lm_head (_outputWeight) but
    // its OWN pre-output norm (nextn.shared_head_norm), not _outputNorm.
    private readonly bool _hasMtp;
    private readonly PagedKvCache? _mtpKvCache;       // standalone KV cache (1 layer)
    private float* _lastHidden;                       // [embDim] pre-output-norm hidden of latest main Forward

    // MTP attention block tensors (same layout as a main full-attention layer)
    private readonly TensorRef _mtpAttnNorm;
    private readonly TensorRef _mtpWQGate;            // Q‖gate interleaved per head (output qDim*2)
    private readonly TensorRef _mtpWK;
    private readonly TensorRef _mtpWV;
    private readonly TensorRef _mtpWO;
    private readonly float* _mtpQNorm;                // [headDim]
    private readonly float* _mtpKNorm;                // [headDim]
    private readonly TensorRef _mtpPostAttnNorm;

    // MTP dense FFN tensors (Q4_K gate/up, Q6_K down typically). Populated only when
    // !_mtpIsMoE — qwen35 27B-MTP path.
    private readonly TensorRef _mtpFfnGate;
    private readonly TensorRef _mtpFfnUp;
    private readonly TensorRef _mtpFfnDown;

    // MTP MoE FFN tensors (qwen35moe 35B-A3B-MTP). Populated only when _mtpIsMoE.
    // Mirrors the per-layer trunk MoE arrays but holds a single block's worth.
    private readonly bool _mtpIsMoE;
    private readonly TensorRef _mtpWGateInp;       // router [embDim, numExperts]
    private readonly TensorRef _mtpWGateShexp;
    private readonly TensorRef _mtpWUpShexp;
    private readonly TensorRef _mtpWDownShexp;
    private readonly TensorRef _mtpWGateExps;      // [numExperts, expertDim, embDim]
    private readonly TensorRef _mtpWUpExps;
    private readonly TensorRef _mtpWDownExps;
    private readonly float* _mtpWGateInpShexpVec;  // [embDim] F32 — shared-expert sigmoid gate

    // nextn.* tensors: pre-fc enorm + hnorm, eh_proj, post-block shared_head_norm
    private readonly float* _mtpEnorm;                // [embDim] F32 gain
    private readonly float* _mtpHnorm;                // [embDim] F32 gain
    private readonly float* _mtpSharedHeadNorm;       // [embDim] F32 gain
    private readonly TensorRef _mtpEhProj;            // [embDim*2 → embDim]; loaded Q8_0 in GGUF, dequant'd to F32

    // MTP scratch (allocated only when _hasMtp)
    private readonly float* _mtpEmbedBuf;             // [embDim]
    private readonly float* _mtpEnormBuf;             // [embDim]
    private readonly float* _mtpHnormBuf;             // [embDim]
    private readonly float* _mtpConcatBuf;            // [embDim * 2]
    private readonly float* _mtpEhProjF32;            // dequant'd eh_proj weight [embDim*2 × embDim] row-major F32

    // Pre-output-norm hidden history indexed by absolute position (slot p = h_p).
    // Sticky across turns so PrefillMtp(startPos>0) can read h_{startPos-1} from
    // slot startPos-1 after a snapshot restore (issue #106).
    private float* _mtpPrefillHiddens;                // [_mtpPrefillHiddensCap × embDim], slot p = h_p
    private int _mtpPrefillHiddensCap;                // allocated capacity in tokens
    private int _mtpHiddenHistoryLength;              // slots [0.._mtpHiddenHistoryLength) populated

    public HybridGdnForwardPass(GgufModel model, IComputeBackend backend, ModelHyperparams hp,
        int maxContextLength = 0)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hp);
        if (!hp.IsHybridSsm)
            throw new ArgumentException("HybridGdnForwardPass requires hp.IsHybridSsm=true (hybrid GDN+attention model).", nameof(hp));
        if (hp.Gdn is null)
            throw new ArgumentException("HybridGdnForwardPass requires hp.Gdn != null (GDN config).", nameof(hp));
        if (hp.LayerTypes is null)
            throw new ArgumentException("HybridGdnForwardPass requires hp.LayerTypes != null (per-layer block type).", nameof(hp));
        // MoE variants (qwen35moe) need both routed and shared experts; dense variants
        // (qwen35 27B-MTP) use a plain ffn_gate/up/down triplet. Reject the partial
        // MoE case (router but no shared expert) — that's not a config we've seen.
        if (hp.IsMoE && !hp.HasSharedExpert)
            throw new ArgumentException("HybridGdnForwardPass with MoE requires HasSharedExpert (qwen35moe).", nameof(hp));
        if (!hp.IsMoE && hp.IntermediateDim <= 0)
            throw new ArgumentException("HybridGdnForwardPass dense FFN requires hp.IntermediateDim > 0.", nameof(hp));
        // QK-norm tensor sizing (lines 397-398, 435-436) and the per-head RmsNorm
        // dispatch in attention layers (lines 871-872, 1055-1056) both assume the
        // Qwen3-style shared-weight convention (one [headDim] weight reused across
        // heads). A model with per-channel QK norm would need both the loaders and
        // the norm calls updated; reject loudly so the wrong-norm-silent-output
        // failure mode can't sneak in.
        if (hp.IsPerChannelQkNorm)
            throw new NotSupportedException(
                "HybridGdnForwardPass does not support per-channel QK norm. " +
                "qwen35moe is the only supported GDN architecture and uses shared QK-norm weights. " +
                "To enable per-channel QK norm, fix both the LoadF32Tensor sizes for _qNorm/_kNorm/_mtpQNorm/_mtpKNorm " +
                "and the PerHeadRmsNorm call sites in the main + MTP attention paths.");

        _model = model;
        _hp = hp;
        _gdn = hp.Gdn;

        int ctxLen = maxContextLength > 0
            ? Math.Min(maxContextLength, hp.ContextLength)
            : Math.Min(hp.ContextLength, 32768);
        _ctxLen = ctxLen;

        _embDim = hp.EmbeddingDim;
        _headDim = hp.HeadDim;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
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

        _kvCache = new PagedKvCache(hp.NumLayers, hp.NumKvHeads, _headDim);
        _gdnStateCache = new GdnStateCache(hp.LayerTypes, _gdn);

        // ── Scratch allocations ─────────────────────────────────────────
        _hidden = Alloc(_embDim);
        _residual = Alloc(_embDim);
        _normBuf = Alloc(_embDim);
        _logits = Alloc(hp.VocabSize);

        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        _qGate = Alloc(qDim * 2);
        _q = Alloc(qDim);
        _gate = Alloc(qDim);
        _k = Alloc(kvDim);
        _v = Alloc(kvDim);
        _attnOut = Alloc(qDim);
        _attnScores = Alloc(_numHeads * ctxLen);

        _qkv = Alloc(_gdnConvChannels);
        _qkvConv = Alloc(_gdnConvChannels);
        _z = Alloc(_gdnValueDim);
        _qVHeads = Alloc(_gdnNumVHeads * _gdnHeadDim);
        _kVHeads = Alloc(_gdnNumVHeads * _gdnHeadDim);
        _alpha = Alloc(_gdnNumVHeads);
        _beta = Alloc(_gdnNumVHeads);
        _gdnOut = Alloc(_gdnValueDim);

        // Resolve Q3_K_Q8K / Q8_0_Q8K kernel gates. Auto-on when the model has
        // routed-expert weights in that dtype (APEX mixed-precision tier — e.g.
        // Carnice). SHARPI_Q3K_Q8K / SHARPI_Q8_0_Q8K = "1" or "0" override.
        bool hasQ3KRouted  = HasRoutedExpertsOfDType(model, hp, DType.Q3_K);
        bool hasQ8_0Routed = HasRoutedExpertsOfDType(model, hp, DType.Q8_0);
        _q3kQ8KEnabled  = ResolveGate("SHARPI_Q3K_Q8K",  hasQ3KRouted);
        _q8_0Q8KEnabled = ResolveGate("SHARPI_Q8_0_Q8K", hasQ8_0Routed);
        if (hp.IsMoE && (_q3kQ8KEnabled || _q8_0Q8KEnabled))
        {
            var enabled = new List<string>(2);
            if (_q3kQ8KEnabled)  enabled.Add($"Q3_K_Q8K (Q3_K routed: {hasQ3KRouted})");
            if (_q8_0Q8KEnabled) enabled.Add($"Q8_0_Q8K (Q8_0 routed: {hasQ8_0Routed})");
            Console.Error.WriteLine(
                $"[HybridGdnForwardPass] Routed-MoE Q8_K-input kernels enabled: {string.Join(", ", enabled)}. Override with SHARPI_Q3K_Q8K=0 / SHARPI_Q8_0_Q8K=0.");
        }

        if (hp.IsMoE)
        {
            _routerLogits = Alloc(hp.NumExperts);
            _sharedOut = Alloc(_embDim);
            _expertGate = Alloc(hp.ExpertIntermediateDim);
            _expertUp = Alloc(hp.ExpertIntermediateDim);
            _expertGateAll = Alloc(hp.NumActiveExperts * hp.ExpertIntermediateDim);
            _expertUpAll = Alloc(hp.NumActiveExperts * hp.ExpertIntermediateDim);
            if (_q3kQ8KEnabled || _q8_0Q8KEnabled)
            {
                // Q8_KS layout (per-32-element sub-block scales) closes the
                // parity gap that #103 surfaced — see DotQ3K_Q8KS / #107.
                _expertGateAllQ8KStride = SimdKernels.Q8KSScratchBytes(hp.ExpertIntermediateDim);
                _normInQ8K = (byte*)NativeMemory.Alloc((nuint)SimdKernels.Q8KSScratchBytes(_embDim));
                _expertGateAllQ8K = (byte*)NativeMemory.Alloc(
                    (nuint)(hp.NumActiveExperts * _expertGateAllQ8KStride));
            }
            else
            {
                _normInQ8K = null;
                _expertGateAllQ8K = null;
                _expertGateAllQ8KStride = 0;
            }
            _intermDim = 0;
        }
        else
        {
            _intermDim = hp.IntermediateDim;
            _ffnGate = Alloc(_intermDim);
            _ffnUp = Alloc(_intermDim);
            _normInQ8K = null;
            _expertGateAllQ8K = null;
            _expertGateAllQ8KStride = 0;
        }

        // RoPE tables sized for partial rotation (ropeDim/2 entries per position).
        _ropeCosTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        _ropeSinTable = (float*)NativeMemory.Alloc((nuint)((long)ctxLen * _ropeHalfDim * sizeof(float)));
        SimdKernels.BuildRopeTable(_ropeCosTable, _ropeSinTable, ctxLen, _ropeDim, hp.RopeTheta);

        // ── Resolve tensors ──────────────────────────────────────────────
        _embTensor = ResolveTensor("token_embd.weight");

        int L = hp.NumLayers;
        _attnNorm = new TensorRef[L];
        _postAttnNorm = new TensorRef[L];
        _wGateInp = new TensorRef[L];
        _wGateShexp = new TensorRef[L];
        _wUpShexp = new TensorRef[L];
        _wDownShexp = new TensorRef[L];
        _wGateExps = new TensorRef[L];
        _wUpExps = new TensorRef[L];
        _wDownExps = new TensorRef[L];

        _wQGate = new TensorRef[L]; _wK = new TensorRef[L]; _wV = new TensorRef[L]; _wO = new TensorRef[L];
        _qNorm = new float*[L]; _kNorm = new float*[L];

        _wQkv = new TensorRef[L]; _wZGate = new TensorRef[L]; _ssmOut = new TensorRef[L];
        _ssmAlpha = new TensorRef[L]; _ssmBeta = new TensorRef[L];
        _ssmConv1d = new float*[L]; _ssmA = new float*[L]; _ssmDtBias = new float*[L]; _ssmNormW = new float*[L];

        _wGateInpShexp = new float*[L];

        _wFfnGate = new TensorRef[L]; _wFfnUp = new TensorRef[L]; _wFfnDown = new TensorRef[L];

        for (int i = 0; i < L; i++)
        {
            // Common to both block types.
            _attnNorm[i] = ResolveTensor($"blk.{i}.attn_norm.weight");
            _postAttnNorm[i] = ResolveTensor($"blk.{i}.post_attention_norm.weight");

            if (hp.IsMoE)
            {
                _wGateInp[i] = ResolveTensor($"blk.{i}.ffn_gate_inp.weight");
                _wGateShexp[i] = ResolveTensor($"blk.{i}.ffn_gate_shexp.weight");
                _wUpShexp[i] = ResolveTensor($"blk.{i}.ffn_up_shexp.weight");
                _wDownShexp[i] = ResolveTensor($"blk.{i}.ffn_down_shexp.weight");
                _wGateExps[i] = ResolveTensor($"blk.{i}.ffn_gate_exps.weight");
                _wUpExps[i] = ResolveTensor($"blk.{i}.ffn_up_exps.weight");
                _wDownExps[i] = ResolveTensor($"blk.{i}.ffn_down_exps.weight");
                _wGateInpShexp[i] = LoadF32Tensor($"blk.{i}.ffn_gate_inp_shexp.weight", _embDim);
            }
            else
            {
                _wFfnGate[i] = ResolveTensor($"blk.{i}.ffn_gate.weight");
                _wFfnUp[i]   = ResolveTensor($"blk.{i}.ffn_up.weight");
                _wFfnDown[i] = ResolveTensor($"blk.{i}.ffn_down.weight");
            }

            if (hp.LayerTypes[i] == LayerType.Attention)
            {
                _wQGate[i] = ResolveTensor($"blk.{i}.attn_q.weight");
                _wK[i] = ResolveTensor($"blk.{i}.attn_k.weight");
                _wV[i] = ResolveTensor($"blk.{i}.attn_v.weight");
                _wO[i] = ResolveTensor($"blk.{i}.attn_output.weight");
                _qNorm[i] = LoadF32Tensor($"blk.{i}.attn_q_norm.weight", _headDim);
                _kNorm[i] = LoadF32Tensor($"blk.{i}.attn_k_norm.weight", _headDim);
            }
            else
            {
                _wQkv[i] = ResolveTensor($"blk.{i}.attn_qkv.weight");
                _wZGate[i] = ResolveTensor($"blk.{i}.attn_gate.weight");
                _ssmOut[i] = ResolveTensor($"blk.{i}.ssm_out.weight");
                _ssmAlpha[i] = ResolveTensor($"blk.{i}.ssm_alpha.weight");
                _ssmBeta[i] = ResolveTensor($"blk.{i}.ssm_beta.weight");
                _ssmA[i] = LoadF32Tensor($"blk.{i}.ssm_a", _gdnNumVHeads);
                _ssmDtBias[i] = LoadF32Tensor($"blk.{i}.ssm_dt.bias", _gdnNumVHeads);
                _ssmNormW[i] = LoadF32Tensor($"blk.{i}.ssm_norm.weight", _gdnHeadDim);
                _ssmConv1d[i] = LoadConv1dTransposed($"blk.{i}.ssm_conv1d.weight",
                    _gdnConvKernel, _gdnConvChannels);
            }
        }

        _outputNorm = ResolveTensor("output_norm.weight");
        _outputWeight = model.FindTensor("output.weight") is not null
            ? ResolveTensor("output.weight")
            : _embTensor; // tied embeddings

        // ── MTP / NEXTN head (issue #25) — block at index NumLayers ────────
        // Loaded only when the GGUF reports nextn_predict_layers > 0 AND all
        // expected tensors are present. Multi-head MTP (NumMtpLayers > 1) is
        // out of scope for v1; we load only the first head.
        _hasMtp = hp.NumMtpLayers > 0
                  && model.FindTensor($"blk.{hp.NumLayers}.nextn.eh_proj.weight") is not null;
        if (_hasMtp)
        {
            int mtpLayerIdx = hp.NumLayers;

            _mtpAttnNorm       = ResolveTensor($"blk.{mtpLayerIdx}.attn_norm.weight");
            _mtpWQGate         = ResolveTensor($"blk.{mtpLayerIdx}.attn_q.weight");
            _mtpWK             = ResolveTensor($"blk.{mtpLayerIdx}.attn_k.weight");
            _mtpWV             = ResolveTensor($"blk.{mtpLayerIdx}.attn_v.weight");
            _mtpWO             = ResolveTensor($"blk.{mtpLayerIdx}.attn_output.weight");
            _mtpQNorm          = LoadF32Tensor($"blk.{mtpLayerIdx}.attn_q_norm.weight", _headDim);
            _mtpKNorm          = LoadF32Tensor($"blk.{mtpLayerIdx}.attn_k_norm.weight", _headDim);
            _mtpPostAttnNorm   = ResolveTensor($"blk.{mtpLayerIdx}.post_attention_norm.weight");

            // MoE-MTP vs dense-MTP probe (issue #44). qwen35moe-A3B-MTP places MoE
            // FFN at the MTP block (ffn_gate_exps/_shexp), qwen35 27B-MTP places dense
            // ffn_gate/up/down. We assume MTP-MoE only co-exists with trunk MoE —
            // the GGUF MoE hyperparams (numExperts, expertDim) drive both paths.
            _mtpIsMoE = model.FindTensor($"blk.{mtpLayerIdx}.ffn_gate_exps.weight") is not null;
            if (_mtpIsMoE && !hp.IsMoE)
                throw new NotSupportedException(
                    "MoE MTP head requires trunk MoE (NumExperts/ExpertIntermediateDim from hyperparams). " +
                    "Dense-trunk + MoE-MTP-head is not a configuration we've seen.");

            if (_mtpIsMoE)
            {
                _mtpWGateInp        = ResolveTensor($"blk.{mtpLayerIdx}.ffn_gate_inp.weight");
                _mtpWGateShexp      = ResolveTensor($"blk.{mtpLayerIdx}.ffn_gate_shexp.weight");
                _mtpWUpShexp        = ResolveTensor($"blk.{mtpLayerIdx}.ffn_up_shexp.weight");
                _mtpWDownShexp      = ResolveTensor($"blk.{mtpLayerIdx}.ffn_down_shexp.weight");
                _mtpWGateExps       = ResolveTensor($"blk.{mtpLayerIdx}.ffn_gate_exps.weight");
                _mtpWUpExps         = ResolveTensor($"blk.{mtpLayerIdx}.ffn_up_exps.weight");
                _mtpWDownExps       = ResolveTensor($"blk.{mtpLayerIdx}.ffn_down_exps.weight");
                _mtpWGateInpShexpVec = LoadF32Tensor($"blk.{mtpLayerIdx}.ffn_gate_inp_shexp.weight", _embDim);
            }
            else
            {
                _mtpFfnGate    = ResolveTensor($"blk.{mtpLayerIdx}.ffn_gate.weight");
                _mtpFfnUp      = ResolveTensor($"blk.{mtpLayerIdx}.ffn_up.weight");
                _mtpFfnDown    = ResolveTensor($"blk.{mtpLayerIdx}.ffn_down.weight");
            }

            _mtpEnorm          = LoadF32Tensor($"blk.{mtpLayerIdx}.nextn.enorm.weight", _embDim);
            _mtpHnorm          = LoadF32Tensor($"blk.{mtpLayerIdx}.nextn.hnorm.weight", _embDim);
            _mtpSharedHeadNorm = LoadF32Tensor($"blk.{mtpLayerIdx}.nextn.shared_head_norm.weight", _embDim);

            // eh_proj is Q8_0 in GGUF; dequant to F32 once at load. The matvec is
            // 10240→5120 once per draft step (~52 M MACs), and a native Q8_0 MatVec
            // path doesn't exist in SimdKernels.MatVec's specialised switch. F32 is
            // 200 MiB extra residence vs Q8_0's ~50 MiB but eliminates dequant cost
            // on the hot path. Switching to a Q8_0 MatVec is a follow-up.
            _mtpEhProj = ResolveTensor($"blk.{mtpLayerIdx}.nextn.eh_proj.weight");
            long ehProjElems = _mtpEhProj.Info.ElementCount;
            if (ehProjElems != (long)_embDim * 2 * _embDim)
                throw new InvalidOperationException(
                    $"blk.{mtpLayerIdx}.nextn.eh_proj.weight: expected {(long)_embDim * 2 * _embDim} elements " +
                    $"({_embDim}*2 × {_embDim}), got {ehProjElems}.");
            _mtpEhProjF32 = Alloc((int)ehProjElems);
            if (_mtpEhProj.DType == DType.Float32)
            {
                var srcSpan = MemoryMarshal.Cast<byte, float>(
                    _model.GetTensorData(_mtpEhProj.Info)).Slice(0, (int)ehProjElems);
                srcSpan.CopyTo(new Span<float>(_mtpEhProjF32, (int)ehProjElems));
            }
            else
            {
                Dequantize.ToFloat32(_model.GetTensorData(_mtpEhProj.Info),
                    new Span<float>(_mtpEhProjF32, (int)ehProjElems), _mtpEhProj.DType, ehProjElems);
            }

            // MTP attention has its OWN paged KV cache (one layer). Position numbering
            // matches the main trunk's — pos P in MTP = pos P in main = "the token that
            // sits at position P" — so accept/reject rewinds can use the same length.
            _mtpKvCache = new PagedKvCache(numLayers: 1, _numKvHeads, _headDim);

            // MTP scratch — small (≤ 10240 floats per buffer, ≪ 1 MiB total).
            _mtpEmbedBuf  = Alloc(_embDim);
            _mtpEnormBuf  = Alloc(_embDim);
            _mtpHnormBuf  = Alloc(_embDim);
            _mtpConcatBuf = Alloc(_embDim * 2);

            // Pre-norm hidden capture buffer; the post-trunk pre-output-norm hidden
            // is needed as MTP input. Sized at embDim, refreshed each Forward.
            _lastHidden = Alloc(_embDim);

            // MTP self-chaining hidden (issue #30): the MTP block's own residual
            // output, captured before the shared-head norm in MtpForward.
            _mtpSelfHidden = Alloc(_embDim);

            // Issue #30 / #45: batched verify scratch. _hidden2/_residual2/_normBuf2/
            // _logits2 are needed for any MTP-bearing model. The dense FFN intermediate
            // buffers (_ffnGate2/_ffnUp2) are only used on the dense path; the MoE path
            // runs MoeFfnCore sequentially per token and shares _expertGate / _expertUp
            // / _expertGateAll / _expertUpAll between t1 and t2.
            _hidden2 = Alloc(_embDim);
            _residual2 = Alloc(_embDim);
            _normBuf2 = Alloc(_embDim);
            _logits2 = Alloc(hp.VocabSize);
            // Lane-3/4 lm_head sinks (issue #209) — the quad lm_head dot runs for both
            // dense and MoE MTP models, so the logits sinks live outside the !IsMoE gate.
            _logits3 = Alloc(hp.VocabSize);
            _logits4 = Alloc(hp.VocabSize);
            if (!hp.IsMoE)
            {
                _ffnGate2 = Alloc(_intermDim);
                _ffnUp2 = Alloc(_intermDim);
                _ffnGate3 = Alloc(_intermDim);
                _ffnUp3 = Alloc(_intermDim);
                _ffnGate4 = Alloc(_intermDim);
                _ffnUp4 = Alloc(_intermDim);
            }

            long perLayerBytes = _gdnStateCache.LayerSnapshotBytes;
            long totalSnapBytes = perLayerBytes * _gdnStateCache.NumGdnLayers;
            if (totalSnapBytes > 0)
            {
                // One ring slot up front (covers BatchForward2); BatchVerify grows
                // the ring lazily via EnsureBatchSnapshotSlots(k - 1).
                _batchSnapshotBuf = (byte*)NativeMemory.Alloc((nuint)totalSnapBytes);
                _batchSnapshotCap = totalSnapBytes;
                _batchSnapshotSlots = 1;
            }
        }

        PrefaultWeights();
    }

    // ============================================================
    //  IForwardPass surface
    // ============================================================

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => _kvCache.MaxSeqLen;

    /// <summary>
    /// Sequential prefill: walks the prompt one token at a time, advancing both
    /// the KV cache (attention layers) and the GDN recurrent state.
    /// Batched prefill is a Phase-7 optimization; the parallel-chunking scan from
    /// llama.cpp's <c>build_delta_net_chunking</c> would land there.
    /// </summary>
    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (tokens is null || tokens.Count == 0)
            throw new ArgumentException("Token list is empty", nameof(tokens));

        // Size the hidden buffer up front so Forward's per-step writes don't
        // each trigger a grow.
        if (_hasMtp)
            EnsureMtpHiddenHistoryCap(startPos + tokens.Count);

        // Chunk-parallel GDN prefill (default on; ~1.3× CPU prefill). Falls back to the
        // per-token loop for single tokens (no chunking benefit), the bypass-debug flags,
        // a cache not positioned at startPos (the chunked path assumes a clean append),
        // or when the model has a native MTP head (_hasMtp): the chunked recurrence is
        // not bit-exact, and on the Qwen3.6-MTP "thinking-or-not" knife-edge token that
        // FP-reorder flips the generation trajectory off the per-token/llama.cpp
        // reference (MtpDecoder_GreedyParity_LlamaCpp). MTP models keep the exact scan.
        if (GdnChunkedPrefillEnabled && tokens.Count > 1 && !_hasMtp
            && !_bypassGdn && !_bypassAttn && !_bypassMoe
            && _kvCache.Length == startPos && _gdnStateCache.Length == startPos)
        {
            return PrefillChunked(tokens, startPos);
        }

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    /// <summary>
    /// Chunk-parallel prompt prefill: processes the prompt layer-major (all tokens
    /// through layer L before L+1 — provably equivalent to the token-major
    /// <see cref="Forward"/> loop because each token's per-layer work is independent
    /// given the prior layer's outputs and the KV/GDN caches are written in position
    /// order). Attention and MoE/FFN run per-token via the exact same kernels and
    /// order as <see cref="Forward"/> (byte-identical); only the GDN recurrence is
    /// replaced by the batched <see cref="GdnKernels.GdnRecurrenceChunkedPrefill"/>
    /// (FlashQLA-style chunk_gated_delta_rule), which differs from the per-token scan
    /// only by floating-point reduction order.
    ///
    /// <para>Gated behind <see cref="GdnChunkedPrefillEnabled"/>; the per-token loop
    /// remains the default. Requires the caches positioned at <paramref name="startPos"/>.</para>
    /// </summary>
    private ReadOnlySpan<float> PrefillChunked(IReadOnlyList<int> tokens, int startPos)
    {
        int n = tokens.Count;
        int e = _embDim;
        int valueDim = _gdnValueDim;
        int hv = _gdnNumVHeads;

        // N-wide host scratch. Allocated per call (prefill is not a per-token hot
        // path) and released in finally. Hidden/residual/norm are [n × embDim];
        // the GDN batched inputs are [n × valueDim] (q/k/v/z), [n × numVHeads]
        // (alpha/beta) and [n × valueDim] (gdn output). Pointers are null-init'd
        // and allocated INSIDE the try so a mid-sequence Alloc failure (OOM) frees
        // the blocks already taken — NativeMemory.Free(null) is a safe no-op.
        float* hid = null, res = null, nrm = null;
        float* gQ = null, gK = null, gV = null, gZ = null, gA = null, gB = null, gO = null;
        try
        {
            hid = Alloc(n * e);
            res = Alloc(n * e);
            nrm = Alloc(n * e);
            gQ = Alloc(n * valueDim);
            gK = Alloc(n * valueDim);
            gV = Alloc(n * valueDim);
            gZ = Alloc(n * valueDim);
            gA = Alloc(n * hv);
            gB = Alloc(n * hv);
            gO = Alloc(n * valueDim);

            for (int t = 0; t < n; t++) EmbedTokenInto(tokens[t], hid + (long)t * e);
            for (int t = 0; t < n; t++) _kvCache.ReserveBlockAt(startPos + t);

            for (int layer = 0; layer < _hp.NumLayers; layer++)
            {
                bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;

                // ── Pre-block residual + attn-norm (per token). ──────────
                Copy(res, hid, n * e);
                float* attnNormW = GetNormWeight(_attnNorm[layer]);
                for (int t = 0; t < n; t++)
                    SimdKernels.RmsNorm(nrm + (long)t * e, hid + (long)t * e, attnNormW, e, _hp.RmsNormEps);

                if (isAttn)
                {
                    // Per-token attention in position order — t reads K/V of 0..t.
                    for (int t = 0; t < n; t++)
                        AttnBlockAt(layer, position: startPos + t, kvPosition: startPos + t,
                            normIn: nrm + (long)t * e, hiddenOut: hid + (long)t * e);
                }
                else
                {
                    GdnBlockChunked(layer, n, nrm, hid, gQ, gK, gV, gZ, gA, gB, gO);
                }

                // Residual add (per token).
                for (int t = 0; t < n; t++)
                    SimdKernels.AddInPlace(hid + (long)t * e, res + (long)t * e, e);

                // ── Pre-FFN residual + post-attn-norm (per token). ───────
                Copy(res, hid, n * e);
                float* postNormW = GetNormWeight(_postAttnNorm[layer]);
                for (int t = 0; t < n; t++)
                    SimdKernels.RmsNorm(nrm + (long)t * e, hid + (long)t * e, postNormW, e, _hp.RmsNormEps);

                // ── FFN per token (same kernels/order as Forward). ───────
                for (int t = 0; t < n; t++)
                {
                    float* normIn = nrm + (long)t * e;
                    float* hiddenOut = hid + (long)t * e;
                    if (_hp.IsMoE)
                        MoeFfnCore(
                            _wGateInp[layer],
                            _wGateShexp[layer], _wUpShexp[layer], _wDownShexp[layer],
                            _wGateExps[layer], _wUpExps[layer], _wDownExps[layer],
                            _wGateInpShexp[layer],
                            normInExt: normIn, hiddenOutExt: hiddenOut);
                    else
                        DenseFfnAt(layer, normIn, hiddenOut);
                }

                // Post-FFN residual add (per token).
                for (int t = 0; t < n; t++)
                    SimdKernels.AddInPlace(hid + (long)t * e, res + (long)t * e, e);
            }

            // Advance both caches by N (one bump per token, in order).
            for (int t = 0; t < n; t++)
            {
                _kvCache.IncrementPosition();
                _gdnStateCache.IncrementPosition();
            }

            // MTP hidden-history capture (pre-output-norm hidden per token).
            if (_hasMtp)
            {
                EnsureMtpHiddenHistoryCap(startPos + n);
                for (int t = 0; t < n; t++)
                    new ReadOnlySpan<float>(hid + (long)t * e, e)
                        .CopyTo(new Span<float>(_mtpPrefillHiddens + (long)(startPos + t) * e, e));
                if (_mtpHiddenHistoryLength < startPos + n)
                    _mtpHiddenHistoryLength = startPos + n;
                new ReadOnlySpan<float>(hid + (long)(n - 1) * e, e)
                    .CopyTo(new Span<float>(_lastHidden, e));
            }

            // Final norm + lm_head for the LAST token (prefill returns its logits).
            float* outNormW = GetNormWeight(_outputNorm);
            SimdKernels.RmsNorm(_hidden, hid + (long)(n - 1) * e, outNormW, e, _hp.RmsNormEps);
            FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, e);
            return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
        }
        finally
        {
            NativeMemory.Free(hid); NativeMemory.Free(res); NativeMemory.Free(nrm);
            NativeMemory.Free(gQ); NativeMemory.Free(gK); NativeMemory.Free(gV);
            NativeMemory.Free(gZ); NativeMemory.Free(gA); NativeMemory.Free(gB);
            NativeMemory.Free(gO);
        }
    }

    /// <summary>
    /// Batched GDN block for the chunked-prefill path. Runs the per-token pre-recurrence
    /// stages (joint QKV + z projection, depthwise conv1d, SiLU, split, per-K-head L2 norm,
    /// K→V tile, alpha/beta projection) into the [n × …] scratch buffers — using the exact
    /// same kernels and order as <see cref="GdnBlockAt"/> — then resolves the recurrence for
    /// all n tokens in one <see cref="GdnKernels.GdnRecurrenceChunkedPrefill"/> call, and
    /// finally applies the per-token ssm-out projection. Conv state threads token-by-token;
    /// scan state advances once for the whole chunk (bit-equivalent to n per-token updates
    /// up to FP reduction order).
    /// </summary>
    private void GdnBlockChunked(
        int layer, int n, float* nrm, float* hid,
        float* gQ, float* gK, float* gV, float* gZ, float* gA, float* gB, float* gO)
    {
        int e = _embDim;
        int convCh = _gdnConvChannels;
        int keyDim = _gdnKeyDim;
        int valueDim = _gdnValueDim;
        int hv = _gdnNumVHeads;
        int hd = _gdnHeadDim;

        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        float* scanState = _gdnStateCache.ScanStateAt(gdnIdx);
        float* convState = _gdnStateCache.ConvStateAt(gdnIdx);
        int convStateLen = _gdnStateCache.ConvStateFloatsPerLayer;
        int scanStateLen = _gdnStateCache.ScanStateFloatsPerLayer;

        var aRef = _ssmAlpha[layer];
        var bRef = _ssmBeta[layer];

        // Pre-recurrence stages, per token (conv state threads in token order).
        for (int t = 0; t < n; t++)
        {
            float* normIn = nrm + (long)t * e;

            FusedMatVec(_qkv, _wQkv[layer], normIn, convCh, e);
            FusedMatVec(_z, _wZGate[layer], normIn, valueDim, e);

            GdnKernels.CausalDepthwiseConv1dDecode(
                new ReadOnlySpan<float>(_qkv, convCh),
                new Span<float>(convState, convStateLen),
                new ReadOnlySpan<float>(_ssmConv1d[layer], _gdnConvKernel * convCh),
                new Span<float>(_qkvConv, convCh),
                convCh, _gdnConvKernel);

            GdnKernels.SiLu(new Span<float>(_qkvConv, convCh), new ReadOnlySpan<float>(_qkvConv, convCh));

            var qPre = new Span<float>(_qkvConv, keyDim);
            var kPre = new Span<float>(_qkvConv + keyDim, keyDim);
            new ReadOnlySpan<float>(_qkvConv + 2 * keyDim, valueDim)
                .CopyTo(new Span<float>(gV + (long)t * valueDim, valueDim));

            GdnKernels.L2NormPerHead(qPre, _gdnNumKHeads, hd, eps: 1e-6f);
            GdnKernels.L2NormPerHead(kPre, _gdnNumKHeads, hd, eps: 1e-6f);
            GdnKernels.TileHeads(qPre, new Span<float>(gQ + (long)t * valueDim, valueDim),
                _gdnNumKHeads, _gdnKvRepeat, hd);
            GdnKernels.TileHeads(kPre, new Span<float>(gK + (long)t * valueDim, valueDim),
                _gdnNumKHeads, _gdnKvRepeat, hd);

            new ReadOnlySpan<float>(_z, valueDim).CopyTo(new Span<float>(gZ + (long)t * valueDim, valueDim));

            SimdKernels.MatVecDual(
                gA + (long)t * hv, aRef.DataPtr,
                gB + (long)t * hv, bRef.DataPtr,
                normIn, hv, e, aRef.DType, bRef.DType);
        }

        // One batched recurrence over all n tokens.
        GdnKernels.GdnRecurrenceChunkedPrefill(
            n,
            new ReadOnlySpan<float>(gQ, n * valueDim),
            new ReadOnlySpan<float>(gK, n * valueDim),
            new ReadOnlySpan<float>(gV, n * valueDim),
            new ReadOnlySpan<float>(gA, n * hv),
            new ReadOnlySpan<float>(gB, n * hv),
            new ReadOnlySpan<float>(_ssmA[layer], hv),
            new ReadOnlySpan<float>(_ssmDtBias[layer], hv),
            new ReadOnlySpan<float>(_ssmNormW[layer], hd),
            new ReadOnlySpan<float>(gZ, n * valueDim),
            new Span<float>(scanState, scanStateLen),
            new Span<float>(gO, n * valueDim),
            hv, hd, normEps: 1e-6f);

        // Per-token ssm-out projection → block output.
        for (int t = 0; t < n; t++)
            FusedMatVec(hid + (long)t * e, _ssmOut[layer], gO + (long)t * valueDim, e, valueDim);
    }

    /// <summary>Dense FFN on external in/out pointers (the <see cref="DenseFfn"/> body,
    /// parameterised for the chunked-prefill per-token loop).</summary>
    private void DenseFfnAt(int layer, float* normIn, float* hiddenOut)
    {
        SimdKernels.MatVecDual(
            _ffnGate, _wFfnGate[layer].DataPtr,
            _ffnUp, _wFfnUp[layer].DataPtr,
            normIn, _intermDim, _embDim,
            _wFfnGate[layer].DType, _wFfnUp[layer].DType);
        SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(hiddenOut, _wFfnDown[layer], _ffnGate, _embDim, _intermDim);
    }

    private void EnsureMtpHiddenHistoryCap(int requiredTokens)
    {
        if (_mtpPrefillHiddensCap >= requiredTokens) return;
        // Grow by doubling so the per-decode-token Forward calls don't trigger
        // an Alloc+Copy+Free at every position past the prompt — that's O(N^2)
        // host memcpy on a long decode. Doubling makes it amortized O(1) per call
        // at the cost of one over-allocation.
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

    /// <summary>
    /// Truncate caches to <paramref name="length"/>. For hybrid GDN models the GDN
    /// recurrent state is destructively updated, so this method only accepts:
    /// <list type="bullet">
    ///   <item><c>length == 0</c> (full reset).</item>
    ///   <item><c>length == Length</c> (no-op).</item>
    ///   <item><c>length == <see cref="SnapshotLength"/></c> when a snapshot was
    ///         captured by <see cref="CaptureSnapshot"/> at end of the previous
    ///         decode (issue #21 — restores GDN state from the held snapshot).</item>
    /// </list>
    /// Any other length still throws <see cref="NotSupportedException"/>.
    /// </summary>
    public void TruncateTo(int length)
    {
        if (length == _gdnStateCache.Length)
        {
            _kvCache.TruncateTo(length);
            // Keep the MTP attention KV in lockstep with the trunk KV even on the
            // no-op-for-GDN path. BatchForward2 + a rejected draft can leave
            // _mtpKvCache past `length` without an accompanying MtpTruncateTo; an
            // unconditional soft truncate here makes the invariant explicit
            // instead of caller-tracked.
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
            // Issue #21: restore GDN state from the snapshot, soft-truncate the
            // trunk KV. Issue #106: also rewind the MTP attention KV and hidden
            // history so PrefillMtp(suffix, startPos=length) sees a consistent
            // view; slots [0..length) survive the rewind.
            _gdnStateCache.RestoreFrom(_snapshotBuf, _snapshotCap);
            _kvCache.TruncateTo(length);
            _mtpKvCache?.TruncateTo(length);
            if (_hasMtp && _mtpHiddenHistoryLength > length)
                _mtpHiddenHistoryLength = length;
            return;
        }
        throw new NotSupportedException(
            $"HybridGdnForwardPass.TruncateTo({length}): Gated DeltaNet state is destructively " +
            $"updated and cannot be partially rewound; only length == 0 (Reset), length == {_gdnStateCache.Length} " +
            $"(current), or length == SnapshotLength ({_snapshotLength}) is supported. " +
            "SupportsPartialRewind == false on this pass — callers should check it before invoking " +
            "TruncateTo with an intermediate length.");
    }

    public void ResetCache()
    {
        _kvCache.Reset();
        _gdnStateCache.Reset();
        _mtpKvCache?.Reset();
        _mtpHiddenHistoryLength = 0;
        ClearSnapshot();
    }

    /// <inheritdoc />
    public bool SupportsPartialRewind => false;

    /// <inheritdoc />
    public bool HasMtpHead => _hasMtp;

    /// <inheritdoc />
    public ReadOnlySpan<float> LastHidden =>
        _hasMtp ? new ReadOnlySpan<float>(_lastHidden, _embDim) : default;

    // ── Snapshot / restore (issue #21) ─────────────────────────────────
    // One snapshot is held per forward-pass instance. The buffer is lazily
    // allocated on first capture and reused thereafter — sizes are constant
    // for the lifetime of the pass (model dims don't change).
    private byte* _snapshotBuf;       // pinned native, lazy-allocated to _gdnStateCache.SnapshotBytes
    private long _snapshotCap;        // allocated capacity in bytes
    private int _snapshotLength = -1; // -1 ⇒ no snapshot held

    /// <inheritdoc />
    public int SnapshotLength => _snapshotLength;

    /// <inheritdoc />
    public bool SupportsSnapshot => true;

    /// <inheritdoc />
    public void CaptureSnapshot()
    {
        EnsureSnapshotBuf();
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

    /// <summary>Run one token through the hybrid transformer.</summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Embedding lookup
        EmbedToken(token);

        if (_traceLayers) EmitLayerTrace(position, -1, "emb");

        // 2. Reserve a KV block slot for this token. Layer 0 is GDN (no KV), so the first
        //    attention layer (index 3) would otherwise hit PagedKvCache's "layer-0-must-be-first"
        //    invariant. ReserveBlock populates the block table for all layers without
        //    allocating any per-layer page — pages stay null until each layer's first Append.
        _kvCache.ReserveBlock();

        // 3. Trunk layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // ── Pre-block residual + norm ────────────────────────────
            Copy(_residual, _hidden, _embDim);
            var attnNormW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, attnNormW, _embDim, _hp.RmsNormEps);

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            bool bypass = isAttn ? _bypassAttn : _bypassGdn;
            if (bypass)
            {
                // Identity-skip the block: zero out _hidden so the residual add yields _residual.
                new Span<float>(_hidden, _embDim).Clear();
            }
            else if (isAttn)
            {
                if (_traceLayers) EmitBufTrace(position, layer, "attn-pre-norm", _normBuf, _embDim);
                AttnBlock(layer, position);
            }
            else
            {
                if (_traceLayers) EmitBufTrace(position, layer, "gdn-pre-norm", _normBuf, _embDim);
                GdnBlock(layer, position);
            }

            // Residual add
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceLayers) EmitLayerTrace(position, layer, isAttn ? "attn-resid" : "gdn-resid");

            // ── Pre-MoE residual + norm ──────────────────────────────
            Copy(_residual, _hidden, _embDim);
            var postNormW = GetNormWeight(_postAttnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf, _hidden, postNormW, _embDim, _hp.RmsNormEps);

            if (_bypassMoe)
                new Span<float>(_hidden, _embDim).Clear();
            else if (_hp.IsMoE)
                MoeFfn(layer);
            else
                DenseFfn(layer);

            // Residual add
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceLayers) EmitLayerTrace(position, layer, "moe-resid");
            if (_probeLogits) ProbeResidual(position, layer);
        }

        // 4. Advance position counters
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();

        // 5. Capture pre-output-norm hidden for MTP (issue #25). RmsNorm below
        //    overwrites _hidden in place, so snapshot now into both _lastHidden
        //    (current-step pointer) and the absolute-position history slot
        //    (sticky across turns for PrefillMtp(startPos>0), issue #106).
        if (_hasMtp)
        {
            var hSpan = new ReadOnlySpan<float>(_hidden, _embDim);
            hSpan.CopyTo(new Span<float>(_lastHidden, _embDim));
            EnsureMtpHiddenHistoryCap(position + 1);
            hSpan.CopyTo(new Span<float>(_mtpPrefillHiddens + (long)position * _embDim, _embDim));
            if (_mtpHiddenHistoryLength < position + 1)
                _mtpHiddenHistoryLength = position + 1;
        }

        // 6. Final norm + output projection
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);

        if (_traceLayers) EmitLayerTrace(position, _hp.NumLayers, "pre-logits");

        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);

        if (_traceLayers) EmitTopLogits(position);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    // ============================================================
    //  BatchForward2 — MTP batched verify (issue #30)
    // ============================================================
    //
    // Runs two adjacent tokens through the trunk in a single pass so the FFN
    // gate/up/down MatMuls fold to a single weight read per row via MatVec2In.
    // Attn/GDN sublayers stay sequential (t1 then t2) so t2's attention can read
    // t1's just-written K/V and GDN state. A per-layer snapshot of the GDN state
    // is captured between t1 and t2 so a rejected draft can be rolled back to
    // <c>startPos + 1</c> without recomputing t1.
    //
    // <para><b>State on entry:</b> <c>_kvCache.Length == startPos</c>,
    // <c>_gdnStateCache.Length == startPos</c>.</para>
    // <para><b>State on success:</b>
    //   <c>_kvCache.Length == startPos + 2</c>,
    //   <c>_gdnStateCache.Length == startPos + 2</c>,
    //   <c>_lastHidden</c> = h@startPos+1 (t2's pre-output-norm hidden),
    //   both tokens' pre-output-norm hiddens written to the MTP hidden history,
    //   per-layer GDN snapshot captured at "after t1, before t2" — available for
    //   <see cref="RestoreBatchSnapshot"/>.</para>
    // <para><b>Return:</b> two logit slices (predict-startPos+1, predict-startPos+2).</para>
    //
    // <para><b>Restrictions:</b> MoE FFN not supported (MatVec2In is dense-only).
    // The pass is otherwise the same as two sequential <see cref="Forward"/> calls.</para>

    /// <summary>True when the attention KV cache has been SnapKV-compacted, i.e.
    /// the physical slot count (<see cref="PagedKvCache.Length"/>) has dropped
    /// below the logical RoPE position (<see cref="PagedKvCache.LogicalLength"/>).
    /// <c>IncrementPosition</c> advances both together and <c>TruncateTo</c>/<c>Reset</c>
    /// keep them equal, so this is an exact, stable "eviction occurred" signal that
    /// is false in all normal (non-evicted) operation (issue #130). Inert in this pass —
    /// this GDN CPU pass doesn't implement SnapKV eviction (the dense CPU <c>ForwardPass</c>
    /// and the CUDA passes do), so it never calls <c>Compact</c> — but kept symmetric with
    /// the CUDA pass; null-guarded for safety.</summary>
    private bool KvCacheCompacted =>
        _kvCache is not null && _kvCache.Length != _kvCache.LogicalLength;

    /// <summary>True when this pass implements <see cref="BatchForward2"/>. The
    /// <c>SHARPI_DISABLE_BATCH_VERIFY=1</c> env var forces the legacy sequential
    /// MTP path for parity bisection. Issue #45: MoE MTP models are supported via
    /// a sequential per-token MoE FFN inside the otherwise-batched trunk.
    /// Issue #130: batched-verify (<see cref="BatchForward2"/>) cannot run on a
    /// SnapKV-evicted cache — its precondition requires <c>_kvCache.Length == startPos</c>
    /// (logical position), but eviction leaves <c>Length</c> at the budget K while
    /// the logical RoPE position stays at the prompt length N. We gate off when the
    /// cache is compacted so <see cref="MtpDecoder"/> falls back to the eviction-safe
    /// sequential <c>Forward</c> path; making batched-verify coexist with eviction is
    /// the #130 follow-up.</summary>
    public bool SupportsBatchVerify =>
        _hasMtp
        && !KvCacheCompacted
        && Environment.GetEnvironmentVariable("SHARPI_DISABLE_BATCH_VERIFY") != "1";

    /// <summary>
    /// Run two adjacent tokens (t1 at <paramref name="startPos"/>, t2 at
    /// <paramref name="startPos"/>+1) through the trunk with batched FFN. Returns
    /// the two predict-next-position logit slices via out parameters. Both slices
    /// are backed by per-pass scratch and remain valid until the next forward call.
    /// </summary>
    public void BatchForward2(int t1, int t2, int startPos,
        out ReadOnlySpan<float> logits1, out ReadOnlySpan<float> logits2)
    {
        if (!SupportsBatchVerify)
            throw new InvalidOperationException(
                "BatchForward2 is only supported on dense-FFN MTP passes. " +
                "Check SupportsBatchVerify before calling.");
        if (startPos < 0)
            throw new ArgumentOutOfRangeException(nameof(startPos), startPos, "startPos must be >= 0.");
        if (_kvCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchForward2: _kvCache.Length={_kvCache.Length} != startPos={startPos}. " +
                "Caches must be at startPos before the batched verify call. A SnapKV-evicted " +
                "(compacted) cache is unsupported here (issue #130) — callers must check " +
                "SupportsBatchVerify, which returns false once the cache is compacted, and fall " +
                "back to the sequential Forward path.");
        if (_gdnStateCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchForward2: _gdnStateCache.Length={_gdnStateCache.Length} != startPos={startPos}.");

        // 1. Embed both tokens.
        EmbedTokenInto(t1, _hidden);
        EmbedTokenInto(t2, _hidden2);

        // 2. Reserve KV blocks covering both positions. Both tokens almost always
        //    share a page (PageSize=16); the call handles the straddle case too.
        _kvCache.ReserveBlockAt(startPos);
        _kvCache.ReserveBlockAt(startPos + 1);

        _batchSnapshotValid = false;
        long layerSnapBytes = _gdnStateCache.LayerSnapshotBytes;

        // 3. Trunk layers — interleave per layer so the FFN can batch via MatVec2In.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // ── Pre-block residual + attn-norm for BOTH tokens ───────
            Copy(_residual,  _hidden,  _embDim);
            Copy(_residual2, _hidden2, _embDim);
            var attnNormW = GetNormWeight(_attnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf,  _hidden,  attnNormW, _embDim, _hp.RmsNormEps);
            SimdKernels.RmsNorm(_normBuf2, _hidden2, attnNormW, _embDim, _hp.RmsNormEps);

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;

            // ── Attn/GDN: sequential t1 then t2 ──────────────────────
            // t1 first so t2's attention reads t1's K/V (correct causal order).
            if (isAttn)
            {
                AttnBlockAt(layer, position: startPos,     kvPosition: startPos,     normIn: _normBuf,  hiddenOut: _hidden);
                AttnBlockAt(layer, position: startPos + 1, kvPosition: startPos + 1, normIn: _normBuf2, hiddenOut: _hidden2);
            }
            else
            {
                GdnBlockAt(layer, position: startPos,     normIn: _normBuf,  hiddenOut: _hidden);
                // Snapshot this layer's GDN state right after t1 has updated it.
                int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
                _gdnStateCache.SnapshotLayerInto(gdnIdx,
                    _batchSnapshotBuf + (long)gdnIdx * layerSnapBytes,
                    layerSnapBytes);
                GdnBlockAt(layer, position: startPos + 1, normIn: _normBuf2, hiddenOut: _hidden2);
            }

            // ── Residual add for both ────────────────────────────────
            SimdKernels.AddInPlace(_hidden,  _residual,  _embDim);
            SimdKernels.AddInPlace(_hidden2, _residual2, _embDim);

            // ── Pre-FFN residual + post_attention_norm for both ──────
            Copy(_residual,  _hidden,  _embDim);
            Copy(_residual2, _hidden2, _embDim);
            var postNormW = GetNormWeight(_postAttnNorm[layer]);
            SimdKernels.RmsNorm(_normBuf,  _hidden,  postNormW, _embDim, _hp.RmsNormEps);
            SimdKernels.RmsNorm(_normBuf2, _hidden2, postNormW, _embDim, _hp.RmsNormEps);

            // ── FFN ────────────────────────────────────────────────
            // Dense: batched 2-input MatVec2In (the bandwidth win lives here).
            // MoE (issue #45): per-token sequential MoE — the routed-expert
            // top-K usually differs across t1 and t2, so no shared weight reads.
            // Attn/GDN above and lm_head below are what amortise for MoE MTP.
            if (_hp.IsMoE)
            {
                MoeFfnCore(
                    _wGateInp[layer],
                    _wGateShexp[layer], _wUpShexp[layer], _wDownShexp[layer],
                    _wGateExps[layer], _wUpExps[layer], _wDownExps[layer],
                    _wGateInpShexp[layer],
                    normInExt: _normBuf,  hiddenOutExt: _hidden);
                MoeFfnCore(
                    _wGateInp[layer],
                    _wGateShexp[layer], _wUpShexp[layer], _wDownShexp[layer],
                    _wGateExps[layer], _wUpExps[layer], _wDownExps[layer],
                    _wGateInpShexp[layer],
                    normInExt: _normBuf2, hiddenOutExt: _hidden2);
            }
            else
            {
                DenseFfn2(layer, _normBuf, _normBuf2, _hidden, _hidden2);
            }

            // ── Post-FFN residual add for both ───────────────────────
            SimdKernels.AddInPlace(_hidden,  _residual,  _embDim);
            SimdKernels.AddInPlace(_hidden2, _residual2, _embDim);
        }

        // 4. Advance both caches by 2 (one bump per token, in order).
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();
        _batchStartPos = startPos;
        _batchK = 2;

        // 5. Snapshot the pre-output-norm hiddens before final norm overwrites them.
        var h1Span = new ReadOnlySpan<float>(_hidden,  _embDim);
        var h2Span = new ReadOnlySpan<float>(_hidden2, _embDim);
        h2Span.CopyTo(new Span<float>(_lastHidden, _embDim));

        // Issue #106: also write the absolute-position slots in the hidden history
        // buffer so future snapshot-restore + PrefillMtp(startPos = past decode
        // position) calls can read the right h_{p-1}. RestoreBatchSnapshot will
        // shrink _mtpHiddenHistoryLength back if t2 is rejected and only t1 commits.
        if (_hasMtp)
        {
            EnsureMtpHiddenHistoryCap(startPos + 2);
            h1Span.CopyTo(new Span<float>(_mtpPrefillHiddens + (long)startPos       * _embDim, _embDim));
            h2Span.CopyTo(new Span<float>(_mtpPrefillHiddens + (long)(startPos + 1) * _embDim, _embDim));
            if (_mtpHiddenHistoryLength < startPos + 2)
                _mtpHiddenHistoryLength = startPos + 2;
        }

        // 6. Final norm + output projection for both tokens. The lm_head can use
        //    MatVec2In so the vocab-sized weight matrix is read once.
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden,  _hidden,  outNormW, _embDim, _hp.RmsNormEps);
        SimdKernels.RmsNorm(_hidden2, _hidden2, outNormW, _embDim, _hp.RmsNormEps);

        SimdKernels.MatVec2In(_logits, _logits2,
            _outputWeight.DataPtr, _hidden, _hidden2,
            _hp.VocabSize, _embDim, _outputWeight.DType);

        _batchSnapshotValid = true;
        logits1 = new ReadOnlySpan<float>(_logits,  _hp.VocabSize);
        logits2 = new ReadOnlySpan<float>(_logits2, _hp.VocabSize);
    }

    /// <summary>
    /// Roll the caches back to an intermediate point of the most recent batched
    /// verify (<see cref="BatchForward2"/> or <see cref="BatchVerify"/>) using the
    /// per-token-boundary GDN snapshot ring. <paramref name="lengthAfter"/> selects
    /// ring slot <c>lengthAfter - startPos - 1</c>: the state captured after the
    /// batch's token at position <c>lengthAfter - 1</c>. Used by <see cref="MtpDecoder"/>
    /// on a rejected draft; the correction token then either replays via
    /// <see cref="Forward"/> (legacy N=2 path) or rides in the next verify batch
    /// (folded k-token path).
    /// </summary>
    /// <param name="lengthAfter">Cache length to restore to; must lie in
    /// <c>[startPos + 1, startPos + k - 1]</c> of the most recent batched verify.</param>
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

        long layerSnapBytes = _gdnStateCache.LayerSnapshotBytes;
        long slotBytes = layerSnapBytes * _gdnStateCache.NumGdnLayers;
        for (int gdnIdx = 0; gdnIdx < _gdnStateCache.NumGdnLayers; gdnIdx++)
        {
            _gdnStateCache.RestoreLayerFrom(gdnIdx,
                _batchSnapshotBuf + slot * slotBytes + (long)gdnIdx * layerSnapBytes,
                layerSnapBytes);
        }
        _gdnStateCache.SetLength(lengthAfter);
        _kvCache.TruncateTo(lengthAfter);
        // Shrink the hidden history and the MTP attention KV alongside the
        // trunk so the rollback is atomic — MtpDecoder.DecodeBatched also
        // calls MtpTruncateTo right after, but folding it in here removes
        // the implicit "caller must also rewind MTP" contract.
        _mtpKvCache?.TruncateTo(lengthAfter);
        if (_hasMtp && _mtpHiddenHistoryLength > lengthAfter)
            _mtpHiddenHistoryLength = lengthAfter;
        _batchSnapshotValid = false;
    }

    /// <inheritdoc />
    public int MaxBatchVerifyTokens => _mtpBatchMax;

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

    /// <summary>
    /// k-token batched verify for the MTP folded decode loop (issue #30 /
    /// #207 goal 4). Generalizes <see cref="BatchForward2"/>: processes
    /// <paramref name="tokens"/> at positions <c>[startPos, startPos + k)</c> with
    /// per-token sequential attn/GDN sublayers (causal order; GDN snapshot ring
    /// captured after every non-final token) and pair-batched dense FFN / lm_head
    /// via <see cref="SimdKernels.MatVec2In"/> — each weight row read once per pair.
    /// Returns <c>result[i]</c> = logits after <c>tokens[i]</c>.
    /// <para>Per-position outputs are bit-identical regardless of k: every token's
    /// math runs through the same kernels with the same inputs (the odd-k tail is
    /// processed as a duplicated-input MatVec2In pair so no kernel switch occurs).
    /// This matches the BatchForward2 precision class — argmax-stable vs the
    /// sequential <see cref="Forward"/> path, whose dense FFN uses MatVecDual.</para>
    /// </summary>
    public float[][] BatchVerify(int[] tokens, int startPos)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (!SupportsBatchVerify)
            throw new InvalidOperationException(
                "BatchVerify is only supported on MTP passes with an uncompacted cache. " +
                "Check SupportsBatchVerify before calling.");
        int k = tokens.Length;
        if (k == 0) return Array.Empty<float[]>();
        if (startPos < 0 || startPos + k > MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(startPos),
                $"BatchVerify range [{startPos}, {startPos + k}) exceeds the context window (maxSeqLen={MaxSeqLen}).");
        if (k > MaxBatchVerifyTokens)
            throw new ArgumentOutOfRangeException(nameof(tokens), k,
                $"BatchVerify token count exceeds MaxBatchVerifyTokens ({MaxBatchVerifyTokens}); " +
                "raise SHARPI_MTP_BATCH_MAX or shorten the draft chain.");
        if (k == 1)
        {
            // A single token amortizes nothing — same fallback as the dense passes.
            var l = Forward(tokens[0], startPos);
            return [l.ToArray()];
        }
        if (_kvCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _kvCache.Length={_kvCache.Length} != startPos={startPos}. " +
                "Caches must sit exactly at startPos (a SnapKV-compacted cache is gated off " +
                "via SupportsBatchVerify).");
        if (_gdnStateCache.Length != startPos)
            throw new InvalidOperationException(
                $"BatchVerify: _gdnStateCache.Length={_gdnStateCache.Length} != startPos={startPos}.");

        int embDim = _embDim;
        EnsureBatchVerifyScratch(k);
        EnsureBatchSnapshotSlots(k - 1);

        // 1. Embed all tokens into independent residual streams + reserve KV blocks.
        for (int i = 0; i < k; i++)
        {
            EmbedTokenInto(tokens[i], _bvHiddenAll + (long)i * embDim);
            _kvCache.ReserveBlockAt(startPos + i);
        }

        _batchSnapshotValid = false;
        long layerSnapBytes = _gdnStateCache.LayerSnapshotBytes;
        long slotBytes = layerSnapBytes * _gdnStateCache.NumGdnLayers;

        // 2. Trunk layers — per-token attn/GDN (t_i before t_{i+1} so each token's
        //    attention reads its predecessors' K/V and GDN state), batched FFN.
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            var attnNormW = GetNormWeight(_attnNorm[layer]);
            for (int i = 0; i < k; i++)
            {
                float* h = _bvHiddenAll + (long)i * embDim;
                Copy(_bvResidAll + (long)i * embDim, h, embDim);
                SimdKernels.RmsNorm(_bvNormAll + (long)i * embDim, h, attnNormW, embDim, _hp.RmsNormEps);
            }

            bool isAttn = _hp.LayerTypes![layer] == LayerType.Attention;
            if (isAttn)
            {
                for (int i = 0; i < k; i++)
                    AttnBlockAt(layer, position: startPos + i, kvPosition: startPos + i,
                                normIn: _bvNormAll + (long)i * embDim,
                                hiddenOut: _bvHiddenAll + (long)i * embDim);
            }
            else
            {
                int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
                for (int i = 0; i < k; i++)
                {
                    GdnBlockAt(layer, position: startPos + i,
                               normIn: _bvNormAll + (long)i * embDim,
                               hiddenOut: _bvHiddenAll + (long)i * embDim);
                    // Ring slot i = state after token i (rollback-to-startPos+i+1).
                    if (i < k - 1)
                        _gdnStateCache.SnapshotLayerInto(gdnIdx,
                            _batchSnapshotBuf + i * slotBytes + (long)gdnIdx * layerSnapBytes,
                            layerSnapBytes);
                }
            }

            for (int i = 0; i < k; i++)
                SimdKernels.AddInPlace(_bvHiddenAll + (long)i * embDim, _bvResidAll + (long)i * embDim, embDim);

            var postNormW = GetNormWeight(_postAttnNorm[layer]);
            for (int i = 0; i < k; i++)
            {
                float* h = _bvHiddenAll + (long)i * embDim;
                Copy(_bvResidAll + (long)i * embDim, h, embDim);
                SimdKernels.RmsNorm(_bvNormAll + (long)i * embDim, h, postNormW, embDim, _hp.RmsNormEps);
            }

            if (_hp.IsMoE)
            {
                // Per-token MoE (issue #45): routed top-K differs per token, so no
                // shared expert weight reads — same as BatchForward2.
                for (int i = 0; i < k; i++)
                    MoeFfnCore(
                        _wGateInp[layer],
                        _wGateShexp[layer], _wUpShexp[layer], _wDownShexp[layer],
                        _wGateExps[layer], _wUpExps[layer], _wDownExps[layer],
                        _wGateInpShexp[layer],
                        normInExt: _bvNormAll + (long)i * embDim,
                        hiddenOutExt: _bvHiddenAll + (long)i * embDim);
            }
            else
            {
                // MatVec4In quads (issue #209); the final partial group duplicates its
                // last real token into the empty lanes (output → _hidden2 sink) so EVERY
                // token goes through the identical kernel — per-position bits don't
                // depend on k parity.
                for (int i = 0; i < k; i += 4)
                {
                    MtpBatchTail.Group4(i, k, out int j0, out int j1, out int j2, out int j3, out int nReal);
                    DenseFfn4(layer,
                        _bvNormAll + (long)j0 * embDim, _bvNormAll + (long)j1 * embDim,
                        _bvNormAll + (long)j2 * embDim, _bvNormAll + (long)j3 * embDim,
                        _bvHiddenAll + (long)j0 * embDim,
                        nReal > 1 ? _bvHiddenAll + (long)j1 * embDim : _hidden2,
                        nReal > 2 ? _bvHiddenAll + (long)j2 * embDim : _hidden2,
                        nReal > 3 ? _bvHiddenAll + (long)j3 * embDim : _hidden2);
                }
            }

            for (int i = 0; i < k; i++)
                SimdKernels.AddInPlace(_bvHiddenAll + (long)i * embDim, _bvResidAll + (long)i * embDim, embDim);
        }

        // 3. Advance both caches by k.
        for (int i = 0; i < k; i++)
        {
            _kvCache.IncrementPosition();
            _gdnStateCache.IncrementPosition();
        }
        _batchStartPos = startPos;
        _batchK = k;

        // 4. Hidden history (issue #33/#106) + LastHidden before the final norm
        //    overwrites the streams in place.
        if (_hasMtp)
        {
            EnsureMtpHiddenHistoryCap(startPos + k);
            for (int i = 0; i < k; i++)
                new ReadOnlySpan<float>(_bvHiddenAll + (long)i * embDim, embDim).CopyTo(
                    new Span<float>(_mtpPrefillHiddens + (long)(startPos + i) * embDim, embDim));
            if (_mtpHiddenHistoryLength < startPos + k)
                _mtpHiddenHistoryLength = startPos + k;
            Copy(_lastHidden, _bvHiddenAll + (long)(k - 1) * embDim, embDim);
        }

        // 5. Final norm + lm_head, MatVec4In quads (vocab-sized weight read once per
        //    four tokens — issue #209). The final partial group's duplicated-tail lanes
        //    re-run the last real token into the _logits{2..4} sinks and are discarded.
        var outNormW = GetNormWeight(_outputNorm);
        for (int i = 0; i < k; i++)
        {
            float* h = _bvHiddenAll + (long)i * embDim;
            SimdKernels.RmsNorm(h, h, outNormW, embDim, _hp.RmsNormEps);
        }

        var result = new float[k][];
        for (int i = 0; i < k; i += 4)
        {
            MtpBatchTail.Group4(i, k, out int j0, out int j1, out int j2, out int j3, out int nReal);
            SimdKernels.MatVec4In(_logits, _logits2, _logits3, _logits4, _outputWeight.DataPtr,
                _bvHiddenAll + (long)j0 * embDim, _bvHiddenAll + (long)j1 * embDim,
                _bvHiddenAll + (long)j2 * embDim, _bvHiddenAll + (long)j3 * embDim,
                _hp.VocabSize, embDim, _outputWeight.DType);
            result[j0] = new ReadOnlySpan<float>(_logits, _hp.VocabSize).ToArray();
            if (nReal > 1) result[j1] = new ReadOnlySpan<float>(_logits2, _hp.VocabSize).ToArray();
            if (nReal > 2) result[j2] = new ReadOnlySpan<float>(_logits3, _hp.VocabSize).ToArray();
            if (nReal > 3) result[j3] = new ReadOnlySpan<float>(_logits4, _hp.VocabSize).ToArray();
        }

        _batchSnapshotValid = true;
        return result;
    }

    /// <summary>Grow the [k × embDim] batched-verify residual streams (grow-only).
    /// Fields are nulled before each re-allocation so a mid-sequence OOM leaves
    /// null pointers (clean re-entry / Dispose) instead of dangling ones.</summary>
    private void EnsureBatchVerifyScratch(int k)
    {
        if (_bvCap >= k) return;
        nuint bytes = (nuint)((long)k * _embDim * sizeof(float));
        if (_bvHiddenAll != null) { NativeMemory.Free(_bvHiddenAll); _bvHiddenAll = null; }
        if (_bvResidAll != null) { NativeMemory.Free(_bvResidAll); _bvResidAll = null; }
        if (_bvNormAll != null) { NativeMemory.Free(_bvNormAll); _bvNormAll = null; }
        _bvCap = 0;
        _bvHiddenAll = (float*)NativeMemory.AllocZeroed(bytes);
        _bvResidAll = (float*)NativeMemory.AllocZeroed(bytes);
        _bvNormAll = (float*)NativeMemory.AllocZeroed(bytes);
        _bvCap = k;
    }

    /// <summary>Grow the GDN snapshot ring to at least <paramref name="slots"/> slots
    /// (grow-only; contents need not survive — the ring is rewritten every batch).
    /// Same null-before-realloc discipline as <see cref="EnsureBatchVerifyScratch"/>.</summary>
    private void EnsureBatchSnapshotSlots(int slots)
    {
        if (_batchSnapshotSlots >= slots) return;
        long slotBytes = _gdnStateCache.LayerSnapshotBytes * _gdnStateCache.NumGdnLayers;
        if (slotBytes <= 0) { _batchSnapshotSlots = slots; return; }
        if (_batchSnapshotBuf != null) { NativeMemory.Free(_batchSnapshotBuf); _batchSnapshotBuf = null; }
        _batchSnapshotSlots = 0;
        _batchSnapshotBuf = (byte*)NativeMemory.Alloc((nuint)(slotBytes * slots));
        _batchSnapshotCap = slotBytes * slots;
        _batchSnapshotSlots = slots;
    }

    /// <summary>
    /// Batched gate × up → down dense FFN for two tokens sharing the same weight
    /// matrices. Each weight row is touched once per row iteration and dotted
    /// against both inputs via <see cref="SimdKernels.MatVec2In"/>; the gate × up
    /// SiLU and the down projection are both folded into single passes.
    /// </summary>
    private void DenseFfn2(int layer, float* normIn1, float* normIn2,
                           float* hiddenOut1, float* hiddenOut2)
    {
        SimdKernels.MatVec2In(
            _ffnGate, _ffnGate2,
            _wFfnGate[layer].DataPtr, normIn1, normIn2,
            _intermDim, _embDim, _wFfnGate[layer].DType);
        SimdKernels.MatVec2In(
            _ffnUp, _ffnUp2,
            _wFfnUp[layer].DataPtr, normIn1, normIn2,
            _intermDim, _embDim, _wFfnUp[layer].DType);

        SimdKernels.SiLuMul(_ffnGate,  _ffnUp,  _intermDim);
        SimdKernels.SiLuMul(_ffnGate2, _ffnUp2, _intermDim);

        SimdKernels.MatVec2In(
            hiddenOut1, hiddenOut2,
            _wFfnDown[layer].DataPtr, _ffnGate, _ffnGate2,
            _embDim, _intermDim, _wFfnDown[layer].DType);
    }

    /// <summary>
    /// Batched gate × up → down dense FFN for four tokens sharing the same weight
    /// matrices (issue #209). Each weight row is touched once and dotted against all
    /// four inputs via <see cref="SimdKernels.MatVec4In"/> — one weight read per four
    /// tokens versus <see cref="DenseFfn2"/>'s one-per-two. The four gate/up scratch
    /// slabs stay distinct because SiLU consumes each lane before the down projection;
    /// duplicated-tail filler lanes point their <c>hiddenOut</c> at a shared sink.
    /// Per-token bits are identical to <see cref="DenseFfn2"/> / single-token decode.
    /// </summary>
    private void DenseFfn4(int layer,
        float* n0, float* n1, float* n2, float* n3,
        float* out0, float* out1, float* out2, float* out3)
    {
        SimdKernels.MatVec4In(
            _ffnGate, _ffnGate2, _ffnGate3, _ffnGate4,
            _wFfnGate[layer].DataPtr, n0, n1, n2, n3,
            _intermDim, _embDim, _wFfnGate[layer].DType);
        SimdKernels.MatVec4In(
            _ffnUp, _ffnUp2, _ffnUp3, _ffnUp4,
            _wFfnUp[layer].DataPtr, n0, n1, n2, n3,
            _intermDim, _embDim, _wFfnUp[layer].DType);

        SimdKernels.SiLuMul(_ffnGate,  _ffnUp,  _intermDim);
        SimdKernels.SiLuMul(_ffnGate2, _ffnUp2, _intermDim);
        SimdKernels.SiLuMul(_ffnGate3, _ffnUp3, _intermDim);
        SimdKernels.SiLuMul(_ffnGate4, _ffnUp4, _intermDim);

        SimdKernels.MatVec4In(
            out0, out1, out2, out3,
            _wFfnDown[layer].DataPtr, _ffnGate, _ffnGate2, _ffnGate3, _ffnGate4,
            _embDim, _intermDim, _wFfnDown[layer].DType);
    }

    // EmbedToken-into-explicit-pointer variant used by BatchForward2 so t1 and
    // t2 can be embedded into independent residual streams (_hidden, _hidden2).
    private void EmbedTokenInto(int token, float* dst)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                        * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* rowPtr = _embTensor.DataPtr + (long)token * bytesPerRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(dst, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, dst, _embDim, _embTensor.DType);
        }
    }

    // ============================================================
    //  Trace helpers (SHARPI_TRACE_LAYERS=1)
    // ============================================================

    private static float L2NormF(float* x, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++) { double v = x[i]; s += v * v; }
        return (float)Math.Sqrt(s);
    }

    private void EmitLayerTrace(int position, int layer, string blockType)
    {
        EmitBufTrace(position, layer, blockType, _hidden, _embDim);
    }

    /// <summary>
    /// Emits a single trace line: <c>[pos=P Llayer tag] l2=X first8=[...]</c>.
    /// Used for both the residual stream and arbitrary intra-block scratch buffers
    /// (q_conv_predelta, gdn_out, etc.) to allow per-tensor parity with llama.cpp's
    /// eval-callback dump.
    /// </summary>
    private static void EmitBufTrace(int position, int layer, string tag, float* buf, int n)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        float l2 = L2NormF(buf, n);
        double sum = 0;
        for (int i = 0; i < n; i++) sum += buf[i];
        var sb = new System.Text.StringBuilder(220);
        sb.Append("[pos=").Append(position).Append(" L");
        if (layer < 0) sb.Append("--"); else sb.Append(layer);
        sb.Append(' ').Append(tag).Append("] l2=")
          .Append(l2.ToString("G6", inv))
          .Append(" sum=").Append(((float)sum).ToString("G6", inv))
          .Append("  first8=[");
        int k = Math.Min(8, n);
        for (int i = 0; i < k; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(buf[i].ToString("G6", inv));
        }
        sb.Append("]  last3=[");
        int lastStart = Math.Max(k, n - 3);
        for (int i = lastStart; i < n; i++)
        {
            if (i > lastStart) sb.Append(", ");
            sb.Append(buf[i].ToString("G6", inv));
        }
        sb.Append(']');
        Console.Error.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Diagnostic only (SHARPI_PROBE_LOGITS=1). Projects the residual at
    /// <paramref name="layer"/> through output_norm + lm_head and dumps each
    /// probe-id's logit and rank to stderr. Cost: one extra MatVec per probed
    /// layer; safe because the post-layer _hidden is replicated into a private
    /// scratch before the norm so the trunk state isn't disturbed.
    /// </summary>
    private void ProbeResidual(int position, int layer)
    {
        // Probe is intended for the final prefill position only (otherwise the
        // output is unmanageable). Skip prior positions.
        var pos0Env = Environment.GetEnvironmentVariable("SHARPI_PROBE_POS");
        if (pos0Env != null && int.TryParse(pos0Env, out var probePos) && position != probePos) return;

        // Copy _hidden into _residual scratch (will be overwritten on the next
        // pre-block residual save at the top of the layer loop, so this is safe).
        new ReadOnlySpan<float>(_hidden, _embDim).CopyTo(new Span<float>(_residual, _embDim));

        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_residual, _residual, outNormW, _embDim, _hp.RmsNormEps);

        FusedMatVec(_logits, _outputWeight, _residual, _hp.VocabSize, _embDim);

        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var sb = new System.Text.StringBuilder(220);
        sb.Append("[probe pos=").Append(position).Append(" L").Append(layer).Append("]");
        int V = _hp.VocabSize;
        // Find top-1 token + its logit.
        int top = 0; float topV = _logits[0];
        for (int i = 1; i < V; i++) if (_logits[i] > topV) { topV = _logits[i]; top = i; }
        sb.Append(" top=").Append(top).Append('@').Append(topV.ToString("G6", inv));
        foreach (var id in _probeIds)
        {
            if ((uint)id >= (uint)V) continue;
            float v = _logits[id];
            int rank = 0;
            for (int i = 0; i < V; i++) if (_logits[i] > v) rank++;
            sb.Append("  ").Append(id).Append('=').Append(v.ToString("G6", inv)).Append("(r").Append(rank).Append(')');
        }
        Console.Error.WriteLine(sb.ToString());
    }

    private void EmitTopLogits(int position)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        const int K = 5;
        Span<int> idx = stackalloc int[K];
        Span<float> val = stackalloc float[K];
        for (int i = 0; i < K; i++) { idx[i] = -1; val[i] = float.MinValue; }
        int V = _hp.VocabSize;
        for (int i = 0; i < V; i++)
        {
            float lv = _logits[i];
            // Insertion into a small sorted-descending list.
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

    // ============================================================
    //  AttnBlock — full softmax attention with GLU-gated Q
    // ============================================================

    private void AttnBlock(int layer, int position) =>
        AttnBlockAt(layer, position, kvPosition: position, normIn: _normBuf, hiddenOut: _hidden);

    /// <summary>
    /// Attention block parameterised on the input-norm pointer, output-hidden pointer,
    /// RoPE position, and KV write position. Used by both the per-token <see cref="Forward"/>
    /// path (where <paramref name="kvPosition"/> == <paramref name="position"/> ==
    /// current <see cref="PagedKvCache.Length"/>) and by <c>BatchForward2</c> (where t2
    /// passes <paramref name="kvPosition"/> = <c>startPos + 1</c> while the cache's
    /// <c>_length</c> has not yet been bumped). The K/V write goes through
    /// <see cref="PagedKvCache.AppendAt"/> so callers control length-advancement explicitly.
    /// </summary>
    private void AttnBlockAt(int layer, int position, int kvPosition,
                             float* normIn, float* hiddenOut)
    {
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int twoHd = _headDim * 2;   // 512: per-head [Q256, G256]

        // 1. Project: attn_q → [Q‖G] interleaved per head (output 8192).
        FusedMatVec(_qGate, _wQGate[layer], normIn, qDim * 2, _embDim);

        // 2. De-interleave: per head h, _q[h*hd : (h+1)*hd] ← _qGate[h*2hd : h*2hd+hd]
        //                              _gate[h*hd : (h+1)*hd] ← _qGate[h*2hd+hd : (h+1)*2hd]
        for (int h = 0; h < _numHeads; h++)
        {
            float* src = _qGate + h * twoHd;
            float* dstQ = _q + h * _headDim;
            float* dstG = _gate + h * _headDim;
            new ReadOnlySpan<float>(src, _headDim).CopyTo(new Span<float>(dstQ, _headDim));
            new ReadOnlySpan<float>(src + _headDim, _headDim).CopyTo(new Span<float>(dstG, _headDim));
        }

        FusedMatVec(_k, _wK[layer], normIn, kvDim, _embDim);
        FusedMatVec(_v, _wV[layer], normIn, kvDim, _embDim);

        // 3. Per-head Q/K RMSNorm (Qwen3-style: norm BEFORE RoPE; weight is shared across heads).
        PerHeadRmsNorm(_q, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
        PerHeadRmsNorm(_k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);

        // 4. Partial NEOX RoPE — rotates first ropeDim dims, passes through dims [ropeDim, headDim).
        float* cos = _ropeCosTable + (long)position * _ropeHalfDim;
        float* sin = _ropeSinTable + (long)position * _ropeHalfDim;
        SimdKernels.ApplyRoPECachedNeoxPartial(_q, cos, sin, _numHeads, _headDim, _ropeDim);
        SimdKernels.ApplyRoPECachedNeoxPartial(_k, cos, sin, _numKvHeads, _headDim, _ropeDim);

        // 5. Append K/V to the explicit slot. Callers manage cache length advancement.
        _kvCache.AppendAt(layer, kvPosition,
            new ReadOnlySpan<float>(_k, kvDim),
            new ReadOnlySpan<float>(_v, kvDim));

        // 6. Scaled dot-product attention (GQA). Reads K/V at positions 0..kvPosition.
        Attention(layer, kvPosition);

        // 7. Apply GLU gate: attn_out *= sigmoid(gate). (per llama.cpp qwen35moe.cpp build_layer_attn)
        ApplySigmoidGate(_attnOut, _gate, qDim);

        // 8. Output projection (input dim = numHeads * headDim = 4096; output dim = embDim).
        FusedMatVec(hiddenOut, _wO[layer], _attnOut, _embDim, qDim);
    }

    private void Attention(int layer, int position)
    {
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;
        var cache = _kvCache;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = cache.KeyAt(layer, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }

            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = cache.ValueAt(layer, t) + kvHead * hd;
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

    // ============================================================
    //  MTP / NEXTN head — Multi-Token Prediction (issue #25)
    // ============================================================

    /// <inheritdoc />
    public ReadOnlySpan<float> MtpForward(int token, int position, ReadOnlySpan<float> prevHidden)
    {
        if (!_hasMtp)
            throw new InvalidOperationException(
                "MtpForward called on a HybridGdnForwardPass that did not load an MTP head. " +
                "Check HasMtpHead before calling.");
        if (prevHidden.Length != _embDim)
            throw new ArgumentException(
                $"prevHidden length {prevHidden.Length} != EmbeddingDim {_embDim}.", nameof(prevHidden));

        // 1. Embed token into _mtpEmbedBuf [embDim].
        int bytesPerEmbRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                           * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* embRowPtr = _embTensor.DataPtr + (long)token * bytesPerEmbRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)embRowPtr, _embDim)
                .CopyTo(new Span<float>(_mtpEmbedBuf, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(embRowPtr, _mtpEmbedBuf, _embDim, _embTensor.DType);
        }

        // 2. enorm(embedding) → _mtpEnormBuf, hnorm(prevHidden) → _mtpHnormBuf.
        //    Both apply RMSNorm with their respective per-channel gain vector.
        SimdKernels.RmsNorm(_mtpEnormBuf, _mtpEmbedBuf, _mtpEnorm, _embDim, _hp.RmsNormEps);
        fixed (float* prevHiddenPtr = prevHidden)
        {
            SimdKernels.RmsNorm(_mtpHnormBuf, prevHiddenPtr, _mtpHnorm, _embDim, _hp.RmsNormEps);
        }

        // 3. Concat [enorm(e) ‖ hnorm(h)] into _mtpConcatBuf [embDim*2]. The
        //    Qwen3-Next reference (transformers `Qwen3NextNextNDecoderLayer`)
        //    does `torch.cat([enormed, hnormed], dim=-1)` — embedding half first,
        //    hidden half second. Issue #40: the doc string in qwen35moe-plan.md
        //    had the order inverted; the 0% MTP draft acceptance on the bench
        //    was the symptom. Diagnostic confirmed: with this order, MTP top-5
        //    aligns with main top-5; with the inverted order the head produced
        //    semantically unrelated drafts (e.g. "CAD" instead of "python").
        new ReadOnlySpan<float>(_mtpEnormBuf, _embDim)
            .CopyTo(new Span<float>(_mtpConcatBuf, _embDim));
        new ReadOnlySpan<float>(_mtpHnormBuf, _embDim)
            .CopyTo(new Span<float>(_mtpConcatBuf + _embDim, _embDim));

        // 4. eh_proj @ concat → _hidden (reuse shared scratch; main forward isn't
        //    interleaved with MTP forward — the engine serializes them).
        SimdKernels.MatVecF32(_hidden, _mtpEhProjF32, _mtpConcatBuf, _embDim, _embDim * 2);

        // 5. Residual + attn_norm.
        Copy(_residual, _hidden, _embDim);
        var mtpAttnNormW = GetNormWeight(_mtpAttnNorm);
        SimdKernels.RmsNorm(_normBuf, _hidden, mtpAttnNormW, _embDim, _hp.RmsNormEps);

        // 6. Attention block (standard gated attention; same shape as a main
        //    full-attention layer). Inlined to keep the MTP KV cache + per-head
        //    norm tensors decoupled from the trunk arrays.
        MtpAttnBlock(position);

        // 7. Residual add.
        SimdKernels.AddInPlace(_hidden, _residual, _embDim);

        // 8. Residual + post_attention_norm.
        Copy(_residual, _hidden, _embDim);
        var mtpPostNormW = GetNormWeight(_mtpPostAttnNorm);
        SimdKernels.RmsNorm(_normBuf, _hidden, mtpPostNormW, _embDim, _hp.RmsNormEps);

        // 9. FFN — MoE (qwen35moe 35B-A3B-MTP) or dense (qwen35 27B-MTP).
        if (_mtpIsMoE)
        {
            MoeFfnCore(
                _mtpWGateInp,
                _mtpWGateShexp, _mtpWUpShexp, _mtpWDownShexp,
                _mtpWGateExps, _mtpWUpExps, _mtpWDownExps,
                _mtpWGateInpShexpVec);
        }
        else
        {
            SimdKernels.MatVecDual(
                _ffnGate, _mtpFfnGate.DataPtr,
                _ffnUp,   _mtpFfnUp.DataPtr,
                _normBuf, _intermDim, _embDim,
                _mtpFfnGate.DType, _mtpFfnUp.DType);
            SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
            FusedMatVec(_hidden, _mtpFfnDown, _ffnGate, _embDim, _intermDim);
        }

        // 10. Residual add.
        SimdKernels.AddInPlace(_hidden, _residual, _embDim);

        // 10b. Capture the MTP block's residual output BEFORE the in-place
        //      shared-head norm overwrites it (issue #30): multi-token drafting
        //      chains the head on itself, feeding this as the next draft's
        //      prevHidden. See IForwardPass.MtpLastHidden.
        Copy(_mtpSelfHidden, _hidden, _embDim);

        // 11. shared_head_norm (NOT the main output_norm) → output.weight (shared lm_head).
        SimdKernels.RmsNorm(_hidden, _hidden, _mtpSharedHeadNorm, _embDim, _hp.RmsNormEps);
        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    /// <summary>
    /// MTP attention block. Mirrors <see cref="AttnBlock"/> but uses the MTP head's
    /// per-head norm and projection weights, plus its own paged KV cache.
    /// </summary>
    private void MtpAttnBlock(int position)
    {
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int twoHd = _headDim * 2;
        var mtpCache = _mtpKvCache!;

        // Project Q‖gate, K, V. Layer index in the MTP cache is always 0.
        FusedMatVec(_qGate, _mtpWQGate, _normBuf, qDim * 2, _embDim);
        for (int h = 0; h < _numHeads; h++)
        {
            float* src = _qGate + h * twoHd;
            new ReadOnlySpan<float>(src, _headDim).CopyTo(new Span<float>(_q + h * _headDim, _headDim));
            new ReadOnlySpan<float>(src + _headDim, _headDim)
                .CopyTo(new Span<float>(_gate + h * _headDim, _headDim));
        }
        FusedMatVec(_k, _mtpWK, _normBuf, kvDim, _embDim);
        FusedMatVec(_v, _mtpWV, _normBuf, kvDim, _embDim);

        PerHeadRmsNorm(_q, _mtpQNorm, _numHeads, _headDim, _hp.RmsNormEps);
        PerHeadRmsNorm(_k, _mtpKNorm, _numKvHeads, _headDim, _hp.RmsNormEps);

        float* cos = _ropeCosTable + (long)position * _ropeHalfDim;
        float* sin = _ropeSinTable + (long)position * _ropeHalfDim;
        SimdKernels.ApplyRoPECachedNeoxPartial(_q, cos, sin, _numHeads, _headDim, _ropeDim);
        SimdKernels.ApplyRoPECachedNeoxPartial(_k, cos, sin, _numKvHeads, _headDim, _ropeDim);

        // Layer-0 invariant: reserve a block on every Append-first-of-token call.
        mtpCache.ReserveBlock();
        mtpCache.Append(layer: 0,
            new ReadOnlySpan<float>(_k, kvDim),
            new ReadOnlySpan<float>(_v, kvDim));

        // Attention scores + softmax + weighted V (mirror of Attention but
        // walks the MTP cache instead of the trunk cache).
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(_headDim);
        int ctxLen = _ctxLen; int hd = _headDim; int hpkg = _headsPerKvGroup;
        var q = _q; var attnOut = _attnOut; var scores = _attnScores;

        Parallel.For(0, _numHeads, h =>
        {
            int kvHead = h / hpkg;
            float* qHead = q + h * hd;
            float* outHead = attnOut + h * hd;
            float* headScores = scores + (long)h * ctxLen;

            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = mtpCache.KeyAt(layer: 0, t) + kvHead * hd;
                headScores[t] = SimdKernels.DotF32(qHead, kVec, hd) * scale;
            }
            SimdKernels.SoftmaxInPlace(headScores, seqLen);

            for (int d = 0; d < hd; d++) outHead[d] = 0;
            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = mtpCache.ValueAt(layer: 0, t) + kvHead * hd;
                float w = headScores[t];
                for (int d = 0; d < hd; d++) outHead[d] += w * vVec[d];
            }
        });

        // GLU gate.
        ApplySigmoidGate(_attnOut, _gate, qDim);

        // Output projection.
        FusedMatVec(_hidden, _mtpWO, _attnOut, _embDim, qDim);

        mtpCache.IncrementPosition();
    }

    /// <inheritdoc />
    public void MtpResetCache()
    {
        _mtpKvCache?.Reset();
    }

    /// <inheritdoc />
    public void MtpTruncateTo(int length)
    {
        _mtpKvCache?.TruncateTo(length);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Issue #33 / #106: Walks <paramref name="tokens"/> and calls
    /// <see cref="MtpForward(int, int, ReadOnlySpan{float})"/> at each prompt position
    /// so the MTP attention KV cache is populated for positions
    /// [<paramref name="startPos"/>..<paramref name="startPos"/>+N-1]. The previous
    /// hidden <c>h_{startPos+i-1}</c> is read from the absolute-position hidden history
    /// buffer (populated by every preceding <see cref="Prefill"/> /
    /// <see cref="Forward"/> / <see cref="BatchForward2"/> when MTP is loaded). When
    /// <paramref name="startPos"/> is 0 a zero vector is used for the i=0 slot
    /// (llama.cpp's "no previous hidden" convention at the start of a sequence).
    /// When <paramref name="startPos"/> &gt; 0 (prefix reuse / canonical snapshot
    /// restore), h_{startPos-1} is read from the buffer's slot startPos-1; the snapshot
    /// branch in <see cref="TruncateTo"/> guarantees this slot survives the restore.
    /// </remarks>
    public void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0)
    {
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

    // ============================================================
    //  GdnBlock — Gated DeltaNet recurrent step
    // ============================================================

    private void GdnBlock(int layer, int position) =>
        GdnBlockAt(layer, position, normIn: _normBuf, hiddenOut: _hidden);

    /// <summary>
    /// GDN recurrent block parameterised on input-norm + output-hidden pointers. The
    /// recurrence state itself remains the layer-local <see cref="GdnStateCache"/> slot
    /// — the caller is responsible for snapshotting it between calls when running the
    /// batched verify path (issue #30).
    /// </summary>
    private void GdnBlockAt(int layer, int position, float* normIn, float* hiddenOut)
    {
        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        float* scanState = _gdnStateCache.ScanStateAt(gdnIdx);
        float* convState = _gdnStateCache.ConvStateAt(gdnIdx);
        int convStateLen = _gdnStateCache.ConvStateFloatsPerLayer;
        int scanStateLen = _gdnStateCache.ScanStateFloatsPerLayer;

        // 1. Joint QKV projection and z (gate) projection.
        FusedMatVec(_qkv, _wQkv[layer], normIn, _gdnConvChannels, _embDim);
        FusedMatVec(_z, _wZGate[layer], normIn, _gdnValueDim, _embDim);
        if (_traceLayers) {
            EmitBufTrace(position, layer, "gdn-qkv-mixed",  _qkv, _gdnConvChannels);
            EmitBufTrace(position, layer, "gdn-z",          _z,   _gdnValueDim);
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(384);
            sb.Append("[pos=").Append(position).Append(" L").Append(layer).Append(" gdn-z-perhead] l2=[");
            for (int h = 0; h < _gdnNumVHeads; h++) {
                if (h > 0) sb.Append(", ");
                sb.Append(L2NormF(_z + h * _gdnHeadDim, _gdnHeadDim).ToString("G6", inv));
            }
            sb.Append(']');
            Console.Error.WriteLine(sb.ToString());
        }

        // 2. Depthwise causal conv1d over the joint QKV stream.
        //    Weight is preloaded in [kernel, channels] order (transposed from GGUF's [c, k]).
        GdnKernels.CausalDepthwiseConv1dDecode(
            new ReadOnlySpan<float>(_qkv, _gdnConvChannels),
            new Span<float>(convState, convStateLen),
            new ReadOnlySpan<float>(_ssmConv1d[layer], _gdnConvKernel * _gdnConvChannels),
            new Span<float>(_qkvConv, _gdnConvChannels),
            _gdnConvChannels, _gdnConvKernel);
        if (_traceLayers) EmitBufTrace(position, layer, "gdn-conv-raw",  _qkvConv, _gdnConvChannels);

        // 3. Element-wise SiLU on the conv output (q/k/v share the same activation).
        GdnKernels.SiLu(
            new Span<float>(_qkvConv, _gdnConvChannels),
            new ReadOnlySpan<float>(_qkvConv, _gdnConvChannels));
        if (_traceLayers) EmitBufTrace(position, layer, "gdn-conv-silu", _qkvConv, _gdnConvChannels);

        // 4. Split Q‖K‖V at offsets 0, KeyDim, 2*KeyDim. Q and K are at K-head shape
        //    (16 heads × 128); V is at V-head shape (32 heads × 128).
        var qPre = new Span<float>(_qkvConv, _gdnKeyDim);                       // [0 .. 2048)
        var kPre = new Span<float>(_qkvConv + _gdnKeyDim, _gdnKeyDim);          // [2048 .. 4096)
        var vV   = new ReadOnlySpan<float>(_qkvConv + 2 * _gdnKeyDim, _gdnValueDim); // [4096 .. 8192)

        // 5. Per-K-head L2 norm.
        GdnKernels.L2NormPerHead(qPre, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        GdnKernels.L2NormPerHead(kPre, _gdnNumKHeads, _gdnHeadDim, eps: 1e-6f);
        if (_traceLayers) {
            EmitBufTrace(position, layer, "gdn-q-predelta", _qkvConv, _gdnKeyDim);
            EmitBufTrace(position, layer, "gdn-k-predelta", _qkvConv + _gdnKeyDim, _gdnKeyDim);
            EmitBufTrace(position, layer, "gdn-v-predelta", _qkvConv + 2 * _gdnKeyDim, _gdnValueDim);
        }

        // 6. Broadcast K→V head count via tile pattern (Hk=16 → Hv=32 here,
        //    pairing (0,16), (1,17), ...). NOT torch's repeat_interleave — see
        //    GdnKernels.TileHeads doc + ops.cpp:10553 (iq1 = iv1 % neq1).
        GdnKernels.TileHeads(qPre, new Span<float>(_qVHeads, _gdnNumVHeads * _gdnHeadDim),
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);
        GdnKernels.TileHeads(kPre, new Span<float>(_kVHeads, _gdnNumVHeads * _gdnHeadDim),
            _gdnNumKHeads, _gdnKvRepeat, _gdnHeadDim);

        // 7. Alpha / Beta per-v-head pre-activations (F32 weights of shape [embDim, NumVHeads]).
        //    Both projections share the same input vector and (typically) the same dtype —
        //    fuse them so the row-parallel sweep only fires once.
        var aRef = _ssmAlpha[layer];
        var bRef = _ssmBeta[layer];
        SimdKernels.MatVecDual(
            _alpha, aRef.DataPtr,
            _beta,  bRef.DataPtr,
            normIn, _gdnNumVHeads, _embDim, aRef.DType, bRef.DType);
        if (_traceLayers) {
            EmitBufTrace(position, layer, "gdn-alpha",      _alpha, _gdnNumVHeads);
            EmitBufTrace(position, layer, "gdn-beta",       _beta,  _gdnNumVHeads);
        }

        // 8. Recurrence: rank-1 state update + per-head RMSNorm + SiLU(z) gate, all fused.
        GdnKernels.GdnRecurrenceDecode(
            q:          new ReadOnlySpan<float>(_qVHeads, _gdnNumVHeads * _gdnHeadDim),
            k:          new ReadOnlySpan<float>(_kVHeads, _gdnNumVHeads * _gdnHeadDim),
            v:          vV,
            alphaIn:    new ReadOnlySpan<float>(_alpha, _gdnNumVHeads),
            beta:       new ReadOnlySpan<float>(_beta,  _gdnNumVHeads),
            ssmA:       new ReadOnlySpan<float>(_ssmA[layer],      _gdnNumVHeads),
            dtBias:     new ReadOnlySpan<float>(_ssmDtBias[layer], _gdnNumVHeads),
            normWeight: new ReadOnlySpan<float>(_ssmNormW[layer],  _gdnHeadDim),
            z:          new ReadOnlySpan<float>(_z, _gdnValueDim),
            state:      new Span<float>(scanState, scanStateLen),
            output:     new Span<float>(_gdnOut, _gdnValueDim),
            numVHeads:  _gdnNumVHeads,
            headDim:    _gdnHeadDim,
            normEps:    1e-6f,
            layer:      layer,
            position:   position);
        if (_traceLayers) {
            EmitBufTrace(position, layer, "gdn-out",       _gdnOut, _gdnValueDim);
            // Per-head L2 of gdn-out (32 heads x 128 dims). Helps spot a single
            // misbehaving head when the global L2 is off but head 0 looks fine.
            var inv = System.Globalization.CultureInfo.InvariantCulture;
            var sb = new System.Text.StringBuilder(384);
            sb.Append("[pos=").Append(position).Append(" L").Append(layer).Append(" gdn-out-perhead] l2=[");
            for (int h = 0; h < _gdnNumVHeads; h++) {
                if (h > 0) sb.Append(", ");
                sb.Append(L2NormF(_gdnOut + h * _gdnHeadDim, _gdnHeadDim).ToString("G6", inv));
            }
            sb.Append(']');
            Console.Error.WriteLine(sb.ToString());
        }

        // 9. Output projection: ssm_out (input ValueDim=4096, output embDim=2048).
        FusedMatVec(hiddenOut, _ssmOut[layer], _gdnOut, _embDim, _gdnValueDim);
        if (_traceLayers) EmitBufTrace(position, layer, "gdn-proj",     hiddenOut, _embDim);
    }

    // ============================================================
    //  DenseFfn — standard gate × up → down (qwen35 27B-MTP and other
    //  dense hybrid GDN variants without MoE).
    // ============================================================

    private void DenseFfn(int layer)
    {
        SimdKernels.MatVecDual(
            _ffnGate, _wFfnGate[layer].DataPtr,
            _ffnUp,   _wFfnUp[layer].DataPtr,
            _normBuf, _intermDim, _embDim,
            _wFfnGate[layer].DType, _wFfnUp[layer].DType);
        SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(_hidden, _wFfnDown[layer], _ffnGate, _embDim, _intermDim);
    }

    // ============================================================
    //  MoeFfn — 256-expert top-8 router + shared expert (gated)
    // ============================================================

    private void MoeFfn(int layer) =>
        MoeFfnCore(
            wGateInp:      _wGateInp[layer],
            wGateShexp:    _wGateShexp[layer],
            wUpShexp:      _wUpShexp[layer],
            wDownShexp:    _wDownShexp[layer],
            wGateExps:     _wGateExps[layer],
            wUpExps:       _wUpExps[layer],
            wDownExps:     _wDownExps[layer],
            wGateInpShexp: _wGateInpShexp[layer],
            normInExt:     null,
            hiddenOutExt:  null);

    private void MoeFfnCore(
        TensorRef wGateInp,
        TensorRef wGateShexp, TensorRef wUpShexp, TensorRef wDownShexp,
        TensorRef wGateExps, TensorRef wUpExps, TensorRef wDownExps,
        float* wGateInpShexp,
        float* normInExt = null,
        float* hiddenOutExt = null)
    {
        // Issue #45: normInExt/hiddenOutExt let BatchForward2 reuse this routine
        // for the second token (normIn=_normBuf2, hiddenOut=_hidden2). Default
        // null preserves the per-call signature used by single-token Forward
        // and MtpForward (normIn=_normBuf, hiddenOut=_hidden). Routed-expert
        // scratch (_expertGate, _expertGateAll, etc.) is reused safely because
        // each MoeFfnCore call's lifetime is fully contained within the call.
        float* normInLocal = normInExt != null ? normInExt : _normBuf;
        float* hiddenOutLocal = hiddenOutExt != null ? hiddenOutExt : _hidden;
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        // 1. Router (softmax top-K). ffn_gate_inp.weight is F32 [embDim, numExperts].
        FusedMatVec(_routerLogits, wGateInp, normInLocal, numExperts, _embDim);
        SimdKernels.SoftmaxInPlace(_routerLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        // 2. Shared expert: ffn_down @ (SiLU(ffn_gate @ x) * (ffn_up @ x)), then per-token
        //    scalar gate via sigmoid(ffn_gate_inp_shexp · x). Use MatVecDual to fuse
        //    gate+up into a single Parallel.For sweep when dtypes match (the common case).
        SimdKernels.MatVecDual(
            _expertGate, wGateShexp.DataPtr,
            _expertUp,   wUpShexp.DataPtr,
            normInLocal, expertDim, _embDim, wGateShexp.DType, wUpShexp.DType);
        SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
        FusedMatVec(_sharedOut,  wDownShexp, _expertGate, _embDim, expertDim);

        // per llama.cpp build_layer_ffn @ src/models/qwen35moe.cpp:
        //   shared_gate = ffn_gate_inp_shexp @ x         // {n_embd} · {n_embd} → scalar per token
        //   shared_gate = sigmoid(shared_gate)
        //   ffn_shexp = ffn_shexp * shared_gate          // broadcast scalar over channels
        float shexpDot = SimdKernels.DotF32(wGateInpShexp, normInLocal, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        SimdKernels.ScaleInPlace(_sharedOut, shexpScale, _embDim);

        // 3. Routed experts (sparse top-K), batched into 2 Parallel.For sweeps
        //    instead of 24 per-expert ones — gate+up across all 8 experts in
        //    one sweep, then down+weighted-accumulate across all 8 experts in
        //    another. Mirrors CudaHybridGdnForwardPass.CpuMoeFfn.
        var gateExps = wGateExps;
        var upExps = wUpExps;
        var downExps = wDownExps;

        int bprG = (_embDim   / DTypeInfo.BlockSize(gateExps.DType))
                 * DTypeInfo.BytesPerBlock(gateExps.DType);
        int bprU = (_embDim   / DTypeInfo.BlockSize(upExps.DType))
                 * DTypeInfo.BytesPerBlock(upExps.DType);
        int bprD = (expertDim / DTypeInfo.BlockSize(downExps.DType))
                 * DTypeInfo.BytesPerBlock(downExps.DType);

        // Stash the small per-token arrays in native pointers so worker threads
        // can read them without lambda-capturing the stackalloc spans. Parallel.For
        // is synchronous, so the stack frame stays alive until all workers complete.
        int* sePtr = stackalloc int[numActive];
        float* ewPtr = stackalloc float[numActive];
        for (int i = 0; i < numActive; i++)
        {
            sePtr[i] = selectedExperts[i];
            ewPtr[i] = expertWeights[i];
        }

        byte* gateP = gateExps.DataPtr;
        byte* upP   = upExps.DataPtr;
        byte* downP = downExps.DataPtr;
        DType gateDt = gateExps.DType;
        DType upDt   = upExps.DType;
        DType downDt = downExps.DType;
        float* gateAll = _expertGateAll;
        float* upAll   = _expertUpAll;
        float* normBuf = normInLocal;
        float* hiddenOut = hiddenOutLocal;
        int embDimL = _embDim;
        int expertDimL = expertDim;
        int numActiveL = numActive;
        int bprGL = bprG, bprUL = bprU, bprDL = bprD;

        // SHARPI_Q3K_Q8K=1 / SHARPI_Q8_0_Q8K=1: prepack the Phase-A input as
        // Q8_K once so all numActive*expertDim Q3_K / Q8_0 rows can hit the
        // int-domain dot kernels.
        bool useQ8KGate = (_q3kQ8KEnabled  && gateDt == DType.Q3_K)
                       || (_q8_0Q8KEnabled && gateDt == DType.Q8_0);
        bool useQ8KUp   = (_q3kQ8KEnabled  && upDt   == DType.Q3_K)
                       || (_q8_0Q8KEnabled && upDt   == DType.Q8_0);
        byte* normInQ8K = _normInQ8K;
        if (useQ8KGate || useQ8KUp)
            SimdKernels.QuantizeRowToQ8KS(normInLocal, _embDim, normInQ8K);

        // Phase A: gate + up rows for all (k, r) tuples. Specialised on the
        // gate/up dtype pair so the inner dot is inlined directly — saves the
        // per-iter switch on enum across ~327 K dispatches per token.
        if (gateDt == DType.Q4_K && upDt == DType.Q4_K)
        {
            Parallel.For(0, numActiveL * expertDimL, s_moeParallelOpts, idx =>
            {
                int k = idx / expertDimL;
                int r = idx % expertDimL;
                int expertIdx = sePtr[k];
                long offG = (long)expertIdx * expertDimL * bprGL + (long)r * bprGL;
                long offU = (long)expertIdx * expertDimL * bprUL + (long)r * bprUL;
                gateAll[idx] = SimdKernels.DotQ4K(gateP + offG, normBuf, embDimL);
                upAll[idx]   = SimdKernels.DotQ4K(upP   + offU, normBuf, embDimL);
            });
        }
        else if (useQ8KGate && useQ8KUp && gateDt == upDt)
        {
            // Both rows take the same Q8_K-prepacked dot — hoist the dispatch
            // outside the inner loop so the JIT can inline the call. Carnice's
            // Q3_K-dense routed layers are the dominant hot case here; the
            // gateDt == upDt guard ensures we still pick the right kernel.
            DType dt = gateDt;
            Parallel.For(0, numActiveL * expertDimL, s_moeParallelOpts, idx =>
            {
                int k = idx / expertDimL;
                int r = idx % expertDimL;
                int expertIdx = sePtr[k];
                long offG = (long)expertIdx * expertDimL * bprGL + (long)r * bprGL;
                long offU = (long)expertIdx * expertDimL * bprUL + (long)r * bprUL;
                gateAll[idx] = DispatchDotQ8K(gateP + offG, normInQ8K, embDimL, dt);
                upAll[idx]   = DispatchDotQ8K(upP   + offU, normInQ8K, embDimL, dt);
            });
        }
        else
        {
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
        }

        // Phase B: one fused SiLuMul over (numActive × expertDim) contiguous
        // floats. SiLuMul is element-wise, so expert boundaries don't matter.
        SimdKernels.SiLuMul(_expertGateAll, _expertUpAll, numActive * expertDim);

        // SHARPI_Q3K_Q8K=1 / SHARPI_Q8_0_Q8K=1 Phase-C prepack: each routed
        // expert k has its own post-SiLuMul gate slice (gateAll + k*expertDim),
        // so we quantise numActive distinct slices into a stacked Q8_K buffer
        // ahead of the embDim-row Parallel.For, and the dot reads
        // gateAllQ8K + k*stride.
        bool useQ8KDown = (_q3kQ8KEnabled  && downDt == DType.Q3_K)
                       || (_q8_0Q8KEnabled && downDt == DType.Q8_0);
        byte* gateAllQ8K = _expertGateAllQ8K;
        int   gateAllQ8KStride = _expertGateAllQ8KStride;
        if (useQ8KDown)
        {
            for (int k = 0; k < numActiveL; k++)
                SimdKernels.QuantizeRowToQ8KS(
                    gateAll + (long)k * expertDimL,
                    expertDimL,
                    gateAllQ8K + (long)k * gateAllQ8KStride);
        }

        // Phase C: down × weight, fused across all 8 experts into one sweep
        // over embDim output rows. Hot dtypes (Q4_K / Q5_K / Q6_K) get
        // specialised loops so the inner 8-iter accumulator can inline the dot.
        switch (downDt)
        {
            case DType.Q4_K:
                Parallel.For(0, embDimL, s_moeParallelOpts, r =>
                {
                    float sum = 0f;
                    for (int k = 0; k < numActiveL; k++)
                    {
                        int expertIdx = sePtr[k];
                        float w = ewPtr[k];
                        long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                        sum += w * SimdKernels.DotQ4K(downP + offD,
                                                      gateAll + (long)k * expertDimL,
                                                      expertDimL);
                    }
                    hiddenOut[r] = sum;
                });
                break;
            case DType.Q5_K:
                Parallel.For(0, embDimL, s_moeParallelOpts, r =>
                {
                    float sum = 0f;
                    for (int k = 0; k < numActiveL; k++)
                    {
                        int expertIdx = sePtr[k];
                        float w = ewPtr[k];
                        long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                        sum += w * SimdKernels.DotQ5K(downP + offD,
                                                      gateAll + (long)k * expertDimL,
                                                      expertDimL);
                    }
                    hiddenOut[r] = sum;
                });
                break;
            case DType.Q6_K:
                Parallel.For(0, embDimL, s_moeParallelOpts, r =>
                {
                    float sum = 0f;
                    for (int k = 0; k < numActiveL; k++)
                    {
                        int expertIdx = sePtr[k];
                        float w = ewPtr[k];
                        long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                        sum += w * SimdKernels.DotQ6K(downP + offD,
                                                      gateAll + (long)k * expertDimL,
                                                      expertDimL);
                    }
                    hiddenOut[r] = sum;
                });
                break;
            default:
                if (useQ8KDown)
                {
                    DType downDtL = downDt;
                    Parallel.For(0, embDimL, s_moeParallelOpts, r =>
                    {
                        float sum = 0f;
                        for (int k = 0; k < numActiveL; k++)
                        {
                            int expertIdx = sePtr[k];
                            float w = ewPtr[k];
                            long offD = (long)expertIdx * embDimL * bprDL + (long)r * bprDL;
                            sum += w * DispatchDotQ8K(
                                downP + offD,
                                gateAllQ8K + (long)k * gateAllQ8KStride,
                                expertDimL, downDtL);
                        }
                        hiddenOut[r] = sum;
                    });
                }
                else
                {
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
                        hiddenOut[r] = sum;
                    });
                }
                break;
        }

        // 4. Add shared expert output.
        SimdKernels.AddInPlace(hiddenOutLocal, _sharedOut, _embDim);
    }

    // ParallelOptions for the routed-MoE sweeps. Pinning to ProcessorCount avoids
    // the ThreadPool oversubscription that would otherwise add 8+ workers when
    // these short-but-heavy parallel loops fire back-to-back per layer.
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
            _ => throw new NotSupportedException($"Routed expert dtype {dtype} not supported in batched path"),
        };

    // Same idea as DispatchDot but the input is already prepacked to Q8_KS
    // (per-32-element scales — issue #107) once per CpuMoeFfnCore call (Phase A:
    // normInLocal; Phase C: each gateAll slice), so individual rows hit the
    // int-domain dot kernels. Only Q3_K and Q8_0 are wired today — the caller
    // guards entry via the corresponding useQ8K* flag.
    private static float DispatchDotQ8K(byte* row, byte* q8kScratch, int cols, DType dtype) =>
        dtype switch
        {
            DType.Q3_K => SimdKernels.DotQ3K_Q8KS(row, q8kScratch, cols),
            DType.Q8_0 => SimdKernels.DotQ8_0_Q8KS(row, q8kScratch, cols),
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
        return autoDetect;
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

    // ============================================================
    //  Embedding lookup
    // ============================================================

    private void EmbedToken(int token)
    {
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(_embTensor.DType))
                        * DTypeInfo.BytesPerBlock(_embTensor.DType);
        byte* rowPtr = _embTensor.DataPtr + (long)token * bytesPerRow;
        if (_embTensor.DType == DType.Float32)
        {
            new ReadOnlySpan<float>((float*)rowPtr, _embDim)
                .CopyTo(new Span<float>(_hidden, _embDim));
        }
        else
        {
            SimdKernels.DequantRow(rowPtr, _hidden, _embDim, _embTensor.DType);
        }
    }

    // ============================================================
    //  Fused MatVec / norm cache / tensor resolution
    // ============================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FusedMatVec(float* output, in TensorRef tensor, float* input, int rows, int cols)
    {
        SimdKernels.MatVec(output, tensor.DataPtr, input, rows, cols, tensor.DType);
    }

    private float* GetNormWeight(in TensorRef tensor)
    {
        if (_normCache.TryGetValue(tensor.Name, out var cached))
            return (float*)cached;

        var data = _model.GetTensorData(tensor.Info);
        int count = (int)tensor.Info.ElementCount;
        var buf = Alloc(count);

        if (tensor.DType == DType.Float32)
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count).CopyTo(new Span<float>(buf, count));
        else
            Dequantize.ToFloat32(data, new Span<float>(buf, count), tensor.DType, count);

        _normCache[tensor.Name] = (nint)buf;
        return buf;
    }

    /// <summary>
    /// Preload a small F32 (or quantized) tensor into a freshly allocated native buffer.
    /// Asserts the element count matches <paramref name="expectedCount"/>.
    /// </summary>
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

    /// <summary>
    /// Load the depthwise conv1d weight and TRANSPOSE it from the GGUF storage layout
    /// <c>[channels, kernel]</c> (ne0=kernel fast-axis per llama.cpp's
    /// <c>ggml_compute_forward_ssm_conv_f32</c> with <c>c[i0 + i1*nc]</c>) into the
    /// <c>[kernel, channels]</c> row-major layout expected by
    /// <see cref="GdnKernels.CausalDepthwiseConv1dDecode"/>'s <c>weight[k*channels + c]</c>
    /// access pattern.
    /// </summary>
    private float* LoadConv1dTransposed(string name, int kernel, int channels)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        int expected = kernel * channels;
        int count = (int)info.ElementCount;
        if (count != expected)
            throw new InvalidOperationException(
                $"Tensor {name}: expected {expected} elements ({kernel}*{channels}), got {count}.");

        // Source in [channels, kernel] order: src[c*kernel + k]
        // Destination in [kernel, channels] order: dst[k*channels + c]
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

    private TensorRef ResolveTensor(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        return new TensorRef(name, info, info.DType, _model.GetTensorDataPtr(info));
    }

    /// <summary>
    /// Pre-fault every weight page so the first request doesn't stall on demand paging
    /// (issue #221). The whole model is mmap-resident on this CPU/Vulkan GDN pass, so
    /// <see cref="MmapPrefault.RamGate.Always"/> skips the RAM-fit heuristic (subject only
    /// to the <c>SHARPI_PREFAULT=0</c> kill switch).
    /// </summary>
    private void PrefaultWeights()
    {
        var regions = new List<(nint, long)>();
        void Add(TensorRef t)
        {
            if (t.DataPtr != null) regions.Add(((nint)t.DataPtr, t.Info.ByteSize));
        }

        Add(_embTensor); Add(_outputNorm); Add(_outputWeight);
        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            Add(_attnNorm[i]);
            Add(_postAttnNorm[i]);
            if (_hp.IsMoE)
            {
                Add(_wGateInp[i]);
                Add(_wGateShexp[i]); Add(_wUpShexp[i]); Add(_wDownShexp[i]);
                Add(_wGateExps[i]); Add(_wUpExps[i]); Add(_wDownExps[i]);
            }
            else
            {
                Add(_wFfnGate[i]); Add(_wFfnUp[i]); Add(_wFfnDown[i]);
            }
            if (_hp.LayerTypes![i] == LayerType.Attention)
            {
                Add(_wQGate[i]); Add(_wK[i]); Add(_wV[i]); Add(_wO[i]);
            }
            else
            {
                Add(_wQkv[i]); Add(_wZGate[i]); Add(_ssmOut[i]);
                Add(_ssmAlpha[i]); Add(_ssmBeta[i]);
            }
        }

        if (_hasMtp)
        {
            Add(_mtpAttnNorm);
            Add(_mtpWQGate); Add(_mtpWK); Add(_mtpWV); Add(_mtpWO);
            Add(_mtpPostAttnNorm);
            if (_mtpIsMoE)
            {
                Add(_mtpWGateInp);
                Add(_mtpWGateShexp); Add(_mtpWUpShexp); Add(_mtpWDownShexp);
                Add(_mtpWGateExps); Add(_mtpWUpExps); Add(_mtpWDownExps);
            }
            else
            {
                Add(_mtpFfnGate); Add(_mtpFfnUp); Add(_mtpFfnDown);
            }
        }

        MmapPrefault.Run("HybridGdnForwardPass", regions, MmapPrefault.RamGate.Always);
    }

    // ============================================================
    //  Helpers
    // ============================================================

    private static void PerHeadRmsNorm(float* data, float* weight, int numHeads, int headDim, float eps)
    {
        for (int h = 0; h < numHeads; h++)
            SimdKernels.RmsNorm(data + h * headDim, data + h * headDim, weight, headDim, eps);
    }

    /// <summary>In-place: <c>x[i] *= sigmoid(gate[i])</c>.</summary>
    private static void ApplySigmoidGate(float* x, float* gate, int size)
    {
        for (int i = 0; i < size; i++)
        {
            float g = gate[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-g));
            x[i] *= sig;
        }
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));

    private static void Copy(float* dst, float* src, int size) =>
        new ReadOnlySpan<float>(src, size).CopyTo(new Span<float>(dst, size));

    // ============================================================
    //  Dispose — free all native buffers
    // ============================================================

    private bool _disposed;
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Scratch buffers
        NativeMemory.Free(_hidden);
        NativeMemory.Free(_residual);
        NativeMemory.Free(_normBuf);
        NativeMemory.Free(_logits);
        NativeMemory.Free(_qGate);
        NativeMemory.Free(_q);
        NativeMemory.Free(_gate);
        NativeMemory.Free(_k);
        NativeMemory.Free(_v);
        NativeMemory.Free(_attnOut);
        NativeMemory.Free(_attnScores);
        NativeMemory.Free(_qkv);
        NativeMemory.Free(_qkvConv);
        NativeMemory.Free(_z);
        NativeMemory.Free(_qVHeads);
        NativeMemory.Free(_kVHeads);
        NativeMemory.Free(_alpha);
        NativeMemory.Free(_beta);
        NativeMemory.Free(_gdnOut);
        if (_hp.IsMoE)
        {
            NativeMemory.Free(_routerLogits);
            NativeMemory.Free(_sharedOut);
            NativeMemory.Free(_expertGate);
            NativeMemory.Free(_expertUp);
            NativeMemory.Free(_expertGateAll);
            NativeMemory.Free(_expertUpAll);
            if (_normInQ8K != null) NativeMemory.Free(_normInQ8K);
            if (_expertGateAllQ8K != null) NativeMemory.Free(_expertGateAllQ8K);
        }
        else
        {
            NativeMemory.Free(_ffnGate);
            NativeMemory.Free(_ffnUp);
        }
        NativeMemory.Free(_ropeCosTable);
        NativeMemory.Free(_ropeSinTable);

        // Preloaded F32 weights
        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            if (_qNorm[i] != null) NativeMemory.Free(_qNorm[i]);
            if (_kNorm[i] != null) NativeMemory.Free(_kNorm[i]);
            if (_ssmConv1d[i] != null) NativeMemory.Free(_ssmConv1d[i]);
            if (_ssmA[i] != null) NativeMemory.Free(_ssmA[i]);
            if (_ssmDtBias[i] != null) NativeMemory.Free(_ssmDtBias[i]);
            if (_ssmNormW[i] != null) NativeMemory.Free(_ssmNormW[i]);
            if (_wGateInpShexp[i] != null) NativeMemory.Free(_wGateInpShexp[i]);
        }

        // Norm cache
        foreach (var p in _normCache.Values)
            NativeMemory.Free((float*)p);
        _normCache.Clear();

        if (_snapshotBuf != null)
        {
            NativeMemory.Free(_snapshotBuf);
            _snapshotBuf = null;
        }

        if (_hasMtp)
        {
            if (_mtpQNorm != null) NativeMemory.Free(_mtpQNorm);
            if (_mtpKNorm != null) NativeMemory.Free(_mtpKNorm);
            if (_mtpEnorm != null) NativeMemory.Free(_mtpEnorm);
            if (_mtpHnorm != null) NativeMemory.Free(_mtpHnorm);
            if (_mtpSharedHeadNorm != null) NativeMemory.Free(_mtpSharedHeadNorm);
            if (_mtpEhProjF32 != null) NativeMemory.Free(_mtpEhProjF32);
            if (_mtpEmbedBuf != null) NativeMemory.Free(_mtpEmbedBuf);
            if (_mtpEnormBuf != null) NativeMemory.Free(_mtpEnormBuf);
            if (_mtpHnormBuf != null) NativeMemory.Free(_mtpHnormBuf);
            if (_mtpConcatBuf != null) NativeMemory.Free(_mtpConcatBuf);
            if (_mtpWGateInpShexpVec != null) NativeMemory.Free(_mtpWGateInpShexpVec);
            if (_lastHidden != null) NativeMemory.Free(_lastHidden);
            if (_mtpPrefillHiddens != null)
            {
                NativeMemory.Free(_mtpPrefillHiddens);
                _mtpPrefillHiddens = null;
                _mtpPrefillHiddensCap = 0;
            }
            // Issues #30/#45 batched-verify scratch. _ffnGate2/_ffnUp2 are only
            // populated on the dense path; null-check covers the MoE case.
            if (_hidden2 != null) NativeMemory.Free(_hidden2);
            if (_residual2 != null) NativeMemory.Free(_residual2);
            if (_normBuf2 != null) NativeMemory.Free(_normBuf2);
            if (_ffnGate2 != null) NativeMemory.Free(_ffnGate2);
            if (_ffnUp2 != null) NativeMemory.Free(_ffnUp2);
            if (_ffnGate3 != null) NativeMemory.Free(_ffnGate3);
            if (_ffnUp3 != null) NativeMemory.Free(_ffnUp3);
            if (_ffnGate4 != null) NativeMemory.Free(_ffnGate4);
            if (_ffnUp4 != null) NativeMemory.Free(_ffnUp4);
            if (_logits2 != null) NativeMemory.Free(_logits2);
            if (_logits3 != null) NativeMemory.Free(_logits3);
            if (_logits4 != null) NativeMemory.Free(_logits4);
            if (_mtpSelfHidden != null) NativeMemory.Free(_mtpSelfHidden);
            if (_bvHiddenAll != null) NativeMemory.Free(_bvHiddenAll);
            if (_bvResidAll != null) NativeMemory.Free(_bvResidAll);
            if (_bvNormAll != null) NativeMemory.Free(_bvNormAll);
            if (_batchSnapshotBuf != null)
            {
                NativeMemory.Free(_batchSnapshotBuf);
                _batchSnapshotBuf = null;
            }
            _mtpKvCache?.Dispose();
        }

        _kvCache.Dispose();
        _gdnStateCache.Dispose();
    }

    // ============================================================
    //  TensorRef — local copy of the ForwardPass.cs helper struct
    // ============================================================

    private readonly unsafe struct TensorRef
    {
        public readonly string Name;
        public readonly GgufTensorInfo Info;
        public readonly DType DType;
        public readonly byte* DataPtr;

        public TensorRef(string name, GgufTensorInfo info, DType dtype, byte* dataPtr)
        {
            Name = name; Info = info; DType = dtype; DataPtr = dataPtr;
        }
    }
}
