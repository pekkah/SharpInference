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

    // ── MoE scratch ────────────────────────────────────────────────────
    private readonly float* _routerLogits;   // [NumExperts]      = 256
    private readonly float* _sharedOut;      // [embDim]          = 2048
    private readonly float* _expertGate;     // [ExpertIntermDim] = 512
    private readonly float* _expertUp;       // [ExpertIntermDim] = 512
    private readonly float* _expertTmp;      // [embDim]          = 2048 — accumulator for down-proj

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

    // RoPE tables (sized by ropeDim/2, not headDim/2).
    private readonly float* _ropeCosTable;
    private readonly float* _ropeSinTable;

    // Diagnostic: per-layer activation trace (env: SHARPI_TRACE_LAYERS=1). Emits one line
    // per block plus embedding/pre-logits + top-5 logits to stderr. Modelled on
    // SHARPI_TRACE_NORMS in ForwardPass.cs.
    private static readonly bool _traceLayers =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_LAYERS") == "1";

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
        if (!hp.IsMoE)
            throw new ArgumentException("HybridGdnForwardPass currently requires MoE FFN (qwen35moe).", nameof(hp));
        if (!hp.HasSharedExpert)
            throw new ArgumentException("HybridGdnForwardPass currently requires HasSharedExpert (qwen35moe).", nameof(hp));

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

        _routerLogits = Alloc(hp.NumExperts);
        _sharedOut = Alloc(_embDim);
        _expertGate = Alloc(hp.ExpertIntermediateDim);
        _expertUp = Alloc(hp.ExpertIntermediateDim);
        _expertTmp = Alloc(_embDim);

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

        for (int i = 0; i < L; i++)
        {
            // Common to both block types.
            _attnNorm[i] = ResolveTensor($"blk.{i}.attn_norm.weight");
            _postAttnNorm[i] = ResolveTensor($"blk.{i}.post_attention_norm.weight");
            _wGateInp[i] = ResolveTensor($"blk.{i}.ffn_gate_inp.weight");
            _wGateShexp[i] = ResolveTensor($"blk.{i}.ffn_gate_shexp.weight");
            _wUpShexp[i] = ResolveTensor($"blk.{i}.ffn_up_shexp.weight");
            _wDownShexp[i] = ResolveTensor($"blk.{i}.ffn_down_shexp.weight");
            _wGateExps[i] = ResolveTensor($"blk.{i}.ffn_gate_exps.weight");
            _wUpExps[i] = ResolveTensor($"blk.{i}.ffn_up_exps.weight");
            _wDownExps[i] = ResolveTensor($"blk.{i}.ffn_down_exps.weight");
            _wGateInpShexp[i] = LoadF32Tensor($"blk.{i}.ffn_gate_inp_shexp.weight", _embDim);

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
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < tokens.Count; i++)
            logits = Forward(tokens[i], startPos + i);
        return logits;
    }

    /// <summary>
    /// Truncate caches to <paramref name="length"/>. For hybrid GDN models this is
    /// only valid at the no-op boundary (length == current) or full reset (length == 0);
    /// the GDN recurrent state is destructively updated and cannot be partially rewound.
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
        throw new NotSupportedException(
            $"HybridGdnForwardPass.TruncateTo({length}): Gated DeltaNet state is destructively " +
            $"updated and cannot be partially rewound; only length == 0 (Reset) or length == {_gdnStateCache.Length} " +
            "(current) is supported. Speculative decoding is disabled for hybrid GDN models in v1.");
    }

    public void ResetCache()
    {
        _kvCache.Reset();
        _gdnStateCache.Reset();
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
            else
                MoeFfn(layer);

            // Residual add
            SimdKernels.AddInPlace(_hidden, _residual, _embDim);

            if (_traceLayers) EmitLayerTrace(position, layer, "moe-resid");
        }

        // 4. Advance position counters
        _kvCache.IncrementPosition();
        _gdnStateCache.IncrementPosition();

        // 5. Final norm + output projection
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
        sb.Append(']');
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
        FusedMatVec(_alpha, _ssmAlpha[layer], _normBuf, _gdnNumVHeads, _embDim);
        FusedMatVec(_beta,  _ssmBeta[layer],  _normBuf, _gdnNumVHeads, _embDim);
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
            normEps:    1e-6f);
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
        SelectTopK(_routerLogits, numExperts, numActive, selectedExperts, expertWeights);

        // 2. Shared expert: ffn_down @ (SiLU(ffn_gate @ x) * (ffn_up @ x)), then per-token
        //    scalar gate via sigmoid(ffn_gate_inp_shexp · x).
        FusedMatVec(_expertGate, _wGateShexp[layer], _normBuf, expertDim, _embDim);
        FusedMatVec(_expertUp,   _wUpShexp[layer],   _normBuf, expertDim, _embDim);
        SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
        FusedMatVec(_sharedOut,  _wDownShexp[layer], _expertGate, _embDim, expertDim);

        // per llama.cpp build_layer_ffn @ src/models/qwen35moe.cpp:
        //   shared_gate = ffn_gate_inp_shexp @ x         // {n_embd} · {n_embd} → scalar per token
        //   shared_gate = sigmoid(shared_gate)
        //   ffn_shexp = ffn_shexp * shared_gate          // broadcast scalar over channels
        float shexpDot = SimdKernels.DotF32(_wGateInpShexp[layer], _normBuf, _embDim);
        float shexpScale = 1.0f / (1.0f + MathF.Exp(-shexpDot));
        SimdKernels.ScaleInPlace(_sharedOut, shexpScale, _embDim);

        // 3. Routed experts (sparse top-K).
        new Span<float>(_hidden, _embDim).Clear();
        for (int k = 0; k < numActive; k++)
        {
            int expertIdx = selectedExperts[k];
            float weight = expertWeights[k];

            ExpertMatVec(_expertGate, _wGateExps[layer], expertIdx, expertDim, _embDim, _normBuf);
            ExpertMatVec(_expertUp,   _wUpExps[layer],   expertIdx, expertDim, _embDim, _normBuf);
            SimdKernels.SiLuMul(_expertGate, _expertUp, expertDim);
            ExpertMatVecDown(_hidden, _wDownExps[layer], expertIdx, _embDim, expertDim,
                _expertGate, weight);
        }

        // 4. Add shared expert output.
        SimdKernels.AddInPlace(_hidden, _sharedOut, _embDim);
    }

    private void ExpertMatVec(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(output, expertData, input, rows, cols, packedTensor.DType);
    }

    private void ExpertMatVecDown(float* output, in TensorRef packedTensor,
        int expertIdx, int rows, int cols, float* input, float weight)
    {
        int bytesPerRow = (cols / DTypeInfo.BlockSize(packedTensor.DType))
                        * DTypeInfo.BytesPerBlock(packedTensor.DType);
        long expertOffset = (long)expertIdx * rows * bytesPerRow;
        byte* expertData = packedTensor.DataPtr + expertOffset;
        SimdKernels.MatVec(_expertTmp, expertData, input, rows, cols, packedTensor.DType);
        SimdKernels.WeightedAddInPlace(output, _expertTmp, weight, rows);
    }

    private static void SelectTopK(float* logits, int n, int k,
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
            tensors.Add(_wGateInp[i]);
            tensors.Add(_wGateShexp[i]); tensors.Add(_wUpShexp[i]); tensors.Add(_wDownShexp[i]);
            tensors.Add(_wGateExps[i]); tensors.Add(_wUpExps[i]); tensors.Add(_wDownExps[i]);
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
        NativeMemory.Free(_routerLogits);
        NativeMemory.Free(_sharedOut);
        NativeMemory.Free(_expertGate);
        NativeMemory.Free(_expertUp);
        NativeMemory.Free(_expertTmp);
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
