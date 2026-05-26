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

    // ── Dense FFN scratch (only allocated when !_hp.IsMoE, e.g. qwen35 27B) ──
    private readonly float* _ffnGate;        // [IntermediateDim]
    private readonly float* _ffnUp;          // [IntermediateDim]
    private readonly int _intermDim;         // hp.IntermediateDim (dense FFN); 0 when MoE

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

    // MTP dense FFN tensors (Q4_K gate/up, Q6_K down typically)
    private readonly TensorRef _mtpFfnGate;
    private readonly TensorRef _mtpFfnUp;
    private readonly TensorRef _mtpFfnDown;

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

    // Issue #33: buffer of per-position pre-output-norm hiddens from the most
    // recent Prefill, so a follow-up PrefillMtp can drive MtpForward at every
    // prompt position with h_{i-1}. Lazy-allocated; capacity grows as needed.
    private float* _mtpPrefillHiddens;                // [_mtpPrefillHiddensCap × embDim]
    private int _mtpPrefillHiddensCap;                // allocated capacity in tokens
    private int _mtpPrefillHiddensCount;              // hiddens stored by last Prefill
    private int _mtpPrefillHiddensStartPos;           // startPos of last Prefill

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

        if (hp.IsMoE)
        {
            _routerLogits = Alloc(hp.NumExperts);
            _sharedOut = Alloc(_embDim);
            _expertGate = Alloc(hp.ExpertIntermediateDim);
            _expertUp = Alloc(hp.ExpertIntermediateDim);
            _expertGateAll = Alloc(hp.NumActiveExperts * hp.ExpertIntermediateDim);
            _expertUpAll = Alloc(hp.NumActiveExperts * hp.ExpertIntermediateDim);
            _intermDim = 0;
        }
        else
        {
            _intermDim = hp.IntermediateDim;
            _ffnGate = Alloc(_intermDim);
            _ffnUp = Alloc(_intermDim);
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
            _mtpFfnGate        = ResolveTensor($"blk.{mtpLayerIdx}.ffn_gate.weight");
            _mtpFfnUp          = ResolveTensor($"blk.{mtpLayerIdx}.ffn_up.weight");
            _mtpFfnDown        = ResolveTensor($"blk.{mtpLayerIdx}.ffn_down.weight");

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

        // Issue #33: when MTP is loaded, buffer per-position pre-output-norm hiddens
        // so a follow-up PrefillMtp can populate the MTP KV cache without redoing
        // the main trunk. Cost: N × embDim memcpy per prefill (negligible).
        if (_hasMtp)
        {
            EnsureMtpPrefillHiddensCap(tokens.Count);
            _mtpPrefillHiddensCount = tokens.Count;
            _mtpPrefillHiddensStartPos = startPos;
        }

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
        {
            logits = Forward(tokens[i], startPos + i);
            if (_hasMtp)
            {
                // _lastHidden = h_{startPos+i} (set inside Forward when _hasMtp).
                new ReadOnlySpan<float>(_lastHidden, _embDim).CopyTo(
                    new Span<float>(_mtpPrefillHiddens + (long)i * _embDim, _embDim));
            }
        }
        return logits;
    }

    private void EnsureMtpPrefillHiddensCap(int requiredTokens)
    {
        if (_mtpPrefillHiddensCap >= requiredTokens) return;
        if (_mtpPrefillHiddens != null) NativeMemory.Free(_mtpPrefillHiddens);
        _mtpPrefillHiddens = (float*)NativeMemory.Alloc(
            (nuint)((long)requiredTokens * _embDim * sizeof(float)));
        _mtpPrefillHiddensCap = requiredTokens;
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
            return;
        }
        if (length == 0)
        {
            ResetCache();
            return;
        }
        if (length == _snapshotLength && _snapshotLength >= 0)
        {
            // Issue #21: restore GDN state from the end-of-decode snapshot, then
            // soft-truncate the KV cache to the matching position.
            _gdnStateCache.RestoreFrom(_snapshotBuf, _snapshotCap);
            _kvCache.TruncateTo(length);
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
        //    overwrites _hidden in place, so snapshot now. Cheap (embDim copy).
        if (_hasMtp)
            new ReadOnlySpan<float>(_hidden, _embDim).CopyTo(new Span<float>(_lastHidden, _embDim));

        // 6. Final norm + output projection
        var outNormW = GetNormWeight(_outputNorm);
        SimdKernels.RmsNorm(_hidden, _hidden, outNormW, _embDim, _hp.RmsNormEps);

        if (_traceLayers) EmitLayerTrace(position, _hp.NumLayers, "pre-logits");

        FusedMatVec(_logits, _outputWeight, _hidden, _hp.VocabSize, _embDim);

        if (_traceLayers) EmitTopLogits(position);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
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

    private void AttnBlock(int layer, int position)
    {
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int twoHd = _headDim * 2;   // 512: per-head [Q256, G256]

        // 1. Project: attn_q → [Q‖G] interleaved per head (output 8192).
        FusedMatVec(_qGate, _wQGate[layer], _normBuf, qDim * 2, _embDim);

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

        FusedMatVec(_k, _wK[layer], _normBuf, kvDim, _embDim);
        FusedMatVec(_v, _wV[layer], _normBuf, kvDim, _embDim);

        // 3. Per-head Q/K RMSNorm (Qwen3-style: norm BEFORE RoPE; weight is shared across heads).
        PerHeadRmsNorm(_q, _qNorm[layer], _numHeads, _headDim, _hp.RmsNormEps);
        PerHeadRmsNorm(_k, _kNorm[layer], _numKvHeads, _headDim, _hp.RmsNormEps);

        // 4. Partial NEOX RoPE — rotates first ropeDim dims, passes through dims [ropeDim, headDim).
        float* cos = _ropeCosTable + (long)position * _ropeHalfDim;
        float* sin = _ropeSinTable + (long)position * _ropeHalfDim;
        SimdKernels.ApplyRoPECachedNeoxPartial(_q, cos, sin, _numHeads, _headDim, _ropeDim);
        SimdKernels.ApplyRoPECachedNeoxPartial(_k, cos, sin, _numKvHeads, _headDim, _ropeDim);

        // 5. Append K/V to cache.
        _kvCache.Append(layer,
            new ReadOnlySpan<float>(_k, kvDim),
            new ReadOnlySpan<float>(_v, kvDim));

        // 6. Scaled dot-product attention (GQA).
        Attention(layer, position);

        // 7. Apply GLU gate: attn_out *= sigmoid(gate). (per llama.cpp qwen35moe.cpp build_layer_attn)
        ApplySigmoidGate(_attnOut, _gate, qDim);

        // 8. Output projection (input dim = numHeads * headDim = 4096; output dim = embDim).
        FusedMatVec(_hidden, _wO[layer], _attnOut, _embDim, qDim);
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

        // 3. Concat [hnorm(h) ‖ enorm(e)] into _mtpConcatBuf [embDim*2]. Order
        //    matches the GGUF eh_proj weight layout (PR #20533 / Qwen3.6 MTP);
        //    parity verification (issue #25 acceptance #10) will confirm.
        new ReadOnlySpan<float>(_mtpHnormBuf, _embDim)
            .CopyTo(new Span<float>(_mtpConcatBuf, _embDim));
        new ReadOnlySpan<float>(_mtpEnormBuf, _embDim)
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

        // 9. Dense FFN (gate × up → down).
        SimdKernels.MatVecDual(
            _ffnGate, _mtpFfnGate.DataPtr,
            _ffnUp,   _mtpFfnUp.DataPtr,
            _normBuf, _intermDim, _embDim,
            _mtpFfnGate.DType, _mtpFfnUp.DType);
        SimdKernels.SiLuMul(_ffnGate, _ffnUp, _intermDim);
        FusedMatVec(_hidden, _mtpFfnDown, _ffnGate, _embDim, _intermDim);

        // 10. Residual add.
        SimdKernels.AddInPlace(_hidden, _residual, _embDim);

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
    /// Issue #33: Walks <paramref name="tokens"/> and calls
    /// <see cref="MtpForward(int, int, ReadOnlySpan{float})"/> at each prompt position
    /// so the MTP attention KV cache is populated for positions
    /// [<paramref name="startPos"/>..<paramref name="startPos"/>+N-1]. The previous
    /// hidden <c>h_{i-1}</c> is read from the buffer captured during the matching
    /// <see cref="Prefill"/> call. For the first position when
    /// <paramref name="startPos"/> is 0, a zero vector is used (matches llama.cpp's
    /// "no previous hidden" convention at the start of a sequence). When
    /// <paramref name="startPos"/> &gt; 0 (prefix reuse), <c>h_{startPos-1}</c> would
    /// have to come from a prior turn — not currently retained, so callers must
    /// disable prefix reuse when driving MTP.
    /// </remarks>
    public void PrefillMtp(IReadOnlyList<int> tokens, int startPos = 0)
    {
        if (!_hasMtp) return;
        if (tokens is null || tokens.Count == 0) return;

        int N = tokens.Count;
        if (_mtpPrefillHiddensCount < N || _mtpPrefillHiddensStartPos != startPos)
            throw new InvalidOperationException(
                $"PrefillMtp({N} tokens, startPos={startPos}) requires a preceding " +
                $"Prefill with the same startPos and at least {N} tokens; the buffer " +
                $"holds {_mtpPrefillHiddensCount} hiddens at startPos={_mtpPrefillHiddensStartPos}.");

        // For position startPos+i, prevHidden = h_{startPos+i-1}:
        //   i == 0 && startPos == 0  →  zero vector
        //   i == 0 && startPos > 0   →  unsupported (would need h_{startPos-1} from previous turn)
        //   i > 0                    →  _mtpPrefillHiddens[(i-1) * embDim]
        if (startPos > 0)
            throw new InvalidOperationException(
                "PrefillMtp with startPos > 0 is not supported: h_{startPos-1} from a prior " +
                "turn is not retained. The caller (InferenceEngine) should disable prefix " +
                "reuse when MTP is active so PrefillMtp is always called with startPos == 0.");

        // Zero buffer for the i=0 prevHidden slot.
        float* zeroHidden = (float*)NativeMemory.AllocZeroed((nuint)(_embDim * sizeof(float)));
        try
        {
            for (int i = 0; i < N; i++)
            {
                float* prevH = (i == 0) ? zeroHidden : _mtpPrefillHiddens + (long)(i - 1) * _embDim;
                _ = MtpForward(tokens[i], startPos + i, new ReadOnlySpan<float>(prevH, _embDim));
            }
        }
        finally
        {
            NativeMemory.Free(zeroHidden);
        }
    }

    // ============================================================
    //  GdnBlock — Gated DeltaNet recurrent step
    // ============================================================

    private void GdnBlock(int layer, int position)
    {
        int gdnIdx = _gdnStateCache.GdnLayerOf(layer);
        float* scanState = _gdnStateCache.ScanStateAt(gdnIdx);
        float* convState = _gdnStateCache.ConvStateAt(gdnIdx);
        int convStateLen = _gdnStateCache.ConvStateFloatsPerLayer;
        int scanStateLen = _gdnStateCache.ScanStateFloatsPerLayer;

        // 1. Joint QKV projection and z (gate) projection.
        FusedMatVec(_qkv, _wQkv[layer], _normBuf, _gdnConvChannels, _embDim);
        FusedMatVec(_z, _wZGate[layer], _normBuf, _gdnValueDim, _embDim);
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
            _normBuf, _gdnNumVHeads, _embDim, aRef.DType, bRef.DType);
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
        FusedMatVec(_hidden, _ssmOut[layer], _gdnOut, _embDim, _gdnValueDim);
        if (_traceLayers) EmitBufTrace(position, layer, "gdn-proj",     _hidden, _embDim);
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

    private void MoeFfn(int layer)
    {
        int numExperts = _hp.NumExperts;
        int numActive = _hp.NumActiveExperts;
        int expertDim = _hp.ExpertIntermediateDim;

        // 1. Router (softmax top-K). ffn_gate_inp.weight is F32 [embDim, numExperts].
        FusedMatVec(_routerLogits, _wGateInp[layer], _normBuf, numExperts, _embDim);
        SimdKernels.SoftmaxInPlace(_routerLogits, numExperts);

        Span<int> selectedExperts = stackalloc int[numActive];
        Span<float> expertWeights = stackalloc float[numActive];
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights,
            normalize: _hp.NormalizeMoeTopKWeights);

        // 2. Shared expert: ffn_down @ (SiLU(ffn_gate @ x) * (ffn_up @ x)), then per-token
        //    scalar gate via sigmoid(ffn_gate_inp_shexp · x). Use MatVecDual to fuse
        //    gate+up into a single Parallel.For sweep when dtypes match (the common case).
        var gateShexp = _wGateShexp[layer];
        var upShexp   = _wUpShexp[layer];
        SimdKernels.MatVecDual(
            _expertGate, gateShexp.DataPtr,
            _expertUp,   upShexp.DataPtr,
            _normBuf, expertDim, _embDim, gateShexp.DType, upShexp.DType);
        SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
        FusedMatVec(_sharedOut,  _wDownShexp[layer], _expertGate, _embDim, expertDim);

        // per llama.cpp build_layer_ffn @ src/models/qwen35moe.cpp:
        //   shared_gate = ffn_gate_inp_shexp @ x         // {n_embd} · {n_embd} → scalar per token
        //   shared_gate = sigmoid(shared_gate)
        //   ffn_shexp = ffn_shexp * shared_gate          // broadcast scalar over channels
        float shexpDot = SimdKernels.DotF32(_wGateInpShexp[layer], _normBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        SimdKernels.ScaleInPlace(_sharedOut, shexpScale, _embDim);

        // 3. Routed experts (sparse top-K), batched into 2 Parallel.For sweeps
        //    instead of 24 per-expert ones — gate+up across all 8 experts in
        //    one sweep, then down+weighted-accumulate across all 8 experts in
        //    another. Mirrors CudaHybridGdnForwardPass.CpuMoeFfn.
        var gateExps = _wGateExps[layer];
        var upExps = _wUpExps[layer];
        var downExps = _wDownExps[layer];

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
        float* normBuf = _normBuf;
        float* hiddenOut = _hidden;
        int embDimL = _embDim;
        int expertDimL = expertDim;
        int numActiveL = numActive;
        int bprGL = bprG, bprUL = bprU, bprDL = bprD;

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
        else
        {
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
        }

        // Phase B: one fused SiLuMul over (numActive × expertDim) contiguous
        // floats. SiLuMul is element-wise, so expert boundaries don't matter.
        SimdKernels.SiLuMul(_expertGateAll, _expertUpAll, numActive * expertDim);

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
                break;
        }

        // 4. Add shared expert output.
        SimdKernels.AddInPlace(_hidden, _sharedOut, _embDim);
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
            DType.Q4_K    => SimdKernels.DotQ4K(row, input, cols),
            DType.Q5_K    => SimdKernels.DotQ5K(row, input, cols),
            DType.Q6_K    => SimdKernels.DotQ6K(row, input, cols),
            DType.Float32 => SimdKernels.DotF32((float*)row, input, cols),
            _ => throw new NotSupportedException($"Routed expert dtype {dtype} not supported in batched path"),
        };

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

    private void PrefaultWeights()
    {
        var tensors = new List<TensorRef> { _embTensor, _outputNorm, _outputWeight };
        int L = _hp.NumLayers;
        for (int i = 0; i < L; i++)
        {
            tensors.Add(_attnNorm[i]);
            tensors.Add(_postAttnNorm[i]);
            if (_hp.IsMoE)
            {
                tensors.Add(_wGateInp[i]);
                tensors.Add(_wGateShexp[i]); tensors.Add(_wUpShexp[i]); tensors.Add(_wDownShexp[i]);
                tensors.Add(_wGateExps[i]); tensors.Add(_wUpExps[i]); tensors.Add(_wDownExps[i]);
            }
            else
            {
                tensors.Add(_wFfnGate[i]); tensors.Add(_wFfnUp[i]); tensors.Add(_wFfnDown[i]);
            }
            if (_hp.LayerTypes![i] == LayerType.Attention)
            {
                tensors.Add(_wQGate[i]); tensors.Add(_wK[i]); tensors.Add(_wV[i]); tensors.Add(_wO[i]);
            }
            else
            {
                tensors.Add(_wQkv[i]); tensors.Add(_wZGate[i]); tensors.Add(_ssmOut[i]);
                tensors.Add(_ssmAlpha[i]); tensors.Add(_ssmBeta[i]);
            }
        }

        long touchSum = 0;
        Parallel.ForEach(tensors, tensor =>
        {
            long size = tensor.Info.ByteSize;
            byte* ptr = tensor.DataPtr;
            long localSum = 0;
            for (long off = 0; off < size; off += 4096)
                localSum += ptr[off];
            if (size > 0)
                localSum += ptr[size - 1];
            Interlocked.Add(ref touchSum, localSum);
        });
        if (touchSum == long.MinValue) Console.Write(touchSum);
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
            if (_lastHidden != null) NativeMemory.Free(_lastHidden);
            if (_mtpPrefillHiddens != null)
            {
                NativeMemory.Free(_mtpPrefillHiddens);
                _mtpPrefillHiddens = null;
                _mtpPrefillHiddensCap = 0;
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
