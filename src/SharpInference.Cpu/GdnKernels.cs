namespace SharpInference.Cpu;

// ─────────────────────────────────────────────────────────────────────────────
//  Gated DeltaNet (GDN) CPU kernels — Phase 2 of the qwen35moe port.
//
//  The recurrence formula implemented here is taken verbatim from llama.cpp
//  master @ src/models/delta-net-base.cpp, function
//  llm_build_delta_net_base::build_delta_net_autoregressive
//  (https://github.com/ggml-org/llama.cpp/blob/master/src/models/delta-net-base.cpp).
//
//  In ggml notation the state tensor `s` has shape [S_v, S_v, H_v, n_seqs]
//  (first dim is the contiguous/fastest axis). The autoregressive step is:
//
//      s = s * exp(g)                          # element-wise decay, g broadcasts over dim 0
//      sk = sum_rows(s * k)                    # k broadcasts over dim 1 → sk[j] = Σ_i S[i,j]·k[i]
//      d = (v - transpose(sk)) * b             # d[j] = b · (v[j] - Σ_i S[i,j]·k[i])
//      s = s + (k_broadcast * transpose(d))    # S[i,j] += k[i] · d[j]   (outer product k ⊗ d)
//      o = sum_rows(s * q)                     # o[j] = Σ_i S[i,j]·q[i]
//
//  Adopting indices i ≡ key axis and j ≡ value/output axis, this code stores
//  the per-head state as a flat row-major D×D block where
//
//      state[i * D + j]   ≡   S_h[i, j]
//
//  and the recurrence is, for every v-head h:
//
//      decay = exp(softplus(alphaIn[h] + dtBias[h]) · ssmA[h])      // ssmA is already negative
//      b     = sigmoid(beta[h])
//      S    *= decay                                                // scalar multiply
//      p[j]  = Σ_i S[i,j] · k_h[i]                                  // "S^T·k" readout
//      d[j]  = b · (v_h[j] - p[j])
//      S[i,j] += k_h[i] · d[j]                                      // rank-1 outer update
//      o_h[j] = Σ_i S[i,j] · q_h[i]                                 // "S^T·q" readout
//
//  After the recurrence, each head's output is RMS-normalized over the head_dim
//  axis with the shared `ssm_norm.weight` gain, then point-wise multiplied by
//  SiLU(z) (the gate stream computed outside this kernel).
//
//  L2 norm convention: matches llama.cpp's `ggml_compute_forward_l2_norm_f32`
//  (ggml/src/ggml-cpu/ops.cpp lines 4129–4140):
//      scale = 1 / max(sqrt(Σ x²), eps)
//  i.e. epsilon is a floor on the divisor, NOT inside the sqrt. Diverges from
//  the kernel-spec doc text but is required for parity with llama.cpp.
//
//  QKV split order in the joint `attn_qkv` stream is Q‖K‖V (queries first,
//  keys second, values last) per llama.cpp src/models/qwen35moe.cpp lines
//  412–436 (`build_layer_attn_linear`, ggml_view_4d offsets 0,
//  head_k_dim·num_k_heads, 2·head_k_dim·num_k_heads). The kernels in this file
//  receive q/k/v as separate spans; the caller is responsible for slicing.
// ─────────────────────────────────────────────────────────────────────────────

using System.Runtime.CompilerServices;

/// <summary>
/// Stateless CPU kernels for the Gated DeltaNet (GDN) recurrent block used by
/// qwen35moe. Everything is single-precision; no allocations on the hot path.
/// </summary>
/// <remarks>
/// These are reference implementations — correct first, fast later. SIMD/AVX
/// optimization of the per-head matrix update is a future PR.
/// </remarks>
public static class GdnKernels
{
    // ────────────────────────────────────────────────────────────────────
    //  Diagnostic trace for the qwen3.6-27B-MTP pos-12 parity bug
    //  (env: SHARPI_TRACE_GDN_INTERNAL=1). Dumps per-head-0 GDN internals
    //  — decay scalar, state-l2-after-decay, state-l2-after-rank1, p-readout
    //  (q · S, pre-RMSNorm/SiLU) — at layers selected by SHARPI_TRACE_GDN_LAYERS
    //  (default "0,20,40,60") and position selected by SHARPI_TRACE_GDN_POS
    //  (default 12). Only fires when the caller threads layer/position through.
    // ────────────────────────────────────────────────────────────────────
    private static readonly bool _traceGdnInternal =
        Environment.GetEnvironmentVariable("SHARPI_TRACE_GDN_INTERNAL") == "1";
    private static readonly int[] _traceGdnLayers = ParseTraceGdnLayers();
    private static readonly int _traceGdnPos = ParseTraceGdnPos();
    private static int[] ParseTraceGdnLayers()
    {
        var s = Environment.GetEnvironmentVariable("SHARPI_TRACE_GDN_LAYERS");
        if (string.IsNullOrEmpty(s)) return new[] { 0, 20, 40, 60 };
        var parts = s.Split(',');
        var ids = new List<int>(parts.Length);
        foreach (var p in parts) if (int.TryParse(p.Trim(), out var v)) ids.Add(v);
        return ids.ToArray();
    }
    private static int ParseTraceGdnPos()
    {
        var s = Environment.GetEnvironmentVariable("SHARPI_TRACE_GDN_POS");
        if (string.IsNullOrEmpty(s) || !int.TryParse(s, out var v)) return 12;
        return v;
    }
    private static bool ShouldTraceGdn(int layer, int position)
    {
        if (!_traceGdnInternal) return false;
        if (layer < 0 || position < 0) return false;
        if (position != _traceGdnPos) return false;
        return Array.IndexOf(_traceGdnLayers, layer) >= 0;
    }

    // ────────────────────────────────────────────────────────────────────
    //  Element-wise scalar activations
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Element-wise softplus: <c>dst[i] = log1p(exp(src[i]))</c> with a linear
    /// short-circuit at <c>src[i] ≥ 20</c> to avoid <c>exp</c> overflow.
    /// </summary>
    public static void Softplus(Span<float> dst, ReadOnlySpan<float> src)
    {
        if (dst.Length != src.Length)
            throw new ArgumentException("dst and src must have equal length.");
        for (int i = 0; i < src.Length; i++)
        {
            float x = src[i];
            dst[i] = x >= 20.0f ? x : MathF.Log(1.0f + MathF.Exp(x));
        }
    }

    /// <summary>
    /// Element-wise sigmoid: <c>dst[i] = 1 / (1 + exp(-src[i]))</c>.
    /// </summary>
    public static void Sigmoid(Span<float> dst, ReadOnlySpan<float> src)
    {
        if (dst.Length != src.Length)
            throw new ArgumentException("dst and src must have equal length.");
        for (int i = 0; i < src.Length; i++)
            dst[i] = 1.0f / (1.0f + MathF.Exp(-src[i]));
    }

    /// <summary>
    /// Element-wise SiLU (a.k.a. Swish): <c>dst[i] = src[i] · sigmoid(src[i])</c>.
    /// </summary>
    /// <remarks>
    /// <see cref="SimdKernels"/> already has a fused <c>SiLuMul(gate, up, size)</c>
    /// path for the FFN gate-up product. This variant is the unfused scalar form
    /// used by the GDN block when the SiLU output is needed independently of a
    /// multiplicand. Prefer the SimdKernels version when both are available.
    /// </remarks>
    public static void SiLu(Span<float> dst, ReadOnlySpan<float> src)
    {
        if (dst.Length != src.Length)
            throw new ArgumentException("dst and src must have equal length.");
        for (int i = 0; i < src.Length; i++)
        {
            float x = src[i];
            dst[i] = x / (1.0f + MathF.Exp(-x));
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Causal depthwise conv1d (no bias)
    //
    //  State layout: [kernel-1, channels] row-major. state[k * C + c] is the
    //  (k+1)-token-ago sample for channel c — i.e. state[0..C] is the OLDEST
    //  retained token, state[(K-2)*C..(K-1)*C] is the most recent prior token.
    //  After producing output for the current x, shift one row up and write
    //  x into the last row.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Causal depthwise conv1d for a single token. Updates <paramref name="state"/>
    /// in place to roll in the new token.
    /// </summary>
    /// <param name="x">Current-token activations, length <paramref name="channels"/>.</param>
    /// <param name="state">Ring of the previous <c>kernel-1</c> tokens,
    /// length <c>channels · (kernel-1)</c>. Layout <c>[kernel-1, channels]</c>
    /// row-major, oldest first.</param>
    /// <param name="weight">Conv weights of shape <c>[kernel, channels]</c> row-major;
    /// <c>weight[k · channels + c]</c> is the tap for channel <c>c</c> at offset
    /// <c>k</c> (k=0 = oldest, k=kernel-1 = current).</param>
    /// <param name="output">Output activations, length <paramref name="channels"/>.</param>
    /// <param name="channels">Per-token channel count.</param>
    /// <param name="kernel">Conv kernel size (e.g. 4 for qwen35moe).</param>
    public static void CausalDepthwiseConv1dDecode(
        ReadOnlySpan<float> x,
        Span<float> state,
        ReadOnlySpan<float> weight,
        Span<float> output,
        int channels,
        int kernel)
    {
        if (kernel < 1) throw new ArgumentOutOfRangeException(nameof(kernel));
        if (x.Length != channels) throw new ArgumentException("x length != channels");
        if (output.Length != channels) throw new ArgumentException("output length != channels");
        if (weight.Length != kernel * channels) throw new ArgumentException("weight length != kernel*channels");
        int sLen = (kernel - 1) * channels;
        if (state.Length != sLen) throw new ArgumentException("state length != (kernel-1)*channels");

        Conv1dStep(x, state, weight, output, channels, kernel);
        ShiftConvState(x, state, channels, kernel);
    }

    /// <summary>
    /// Causal depthwise conv1d for a contiguous run of <paramref name="tokens"/>.
    /// Inputs <paramref name="x"/> and <paramref name="output"/> are
    /// <c>[tokens, channels]</c> row-major. After the call, <paramref name="state"/>
    /// holds the most recent <c>kernel-1</c> tokens of <paramref name="x"/> (older
    /// tokens have rolled out).
    /// </summary>
    public static void CausalDepthwiseConv1dPrefill(
        ReadOnlySpan<float> x,
        Span<float> state,
        ReadOnlySpan<float> weight,
        Span<float> output,
        int tokens,
        int channels,
        int kernel)
    {
        if (kernel < 1) throw new ArgumentOutOfRangeException(nameof(kernel));
        if (x.Length != tokens * channels) throw new ArgumentException("x length != tokens*channels");
        if (output.Length != tokens * channels) throw new ArgumentException("output length != tokens*channels");
        if (weight.Length != kernel * channels) throw new ArgumentException("weight length != kernel*channels");
        int sLen = (kernel - 1) * channels;
        if (state.Length != sLen) throw new ArgumentException("state length != (kernel-1)*channels");

        for (int t = 0; t < tokens; t++)
        {
            var xt = x.Slice(t * channels, channels);
            var ot = output.Slice(t * channels, channels);
            Conv1dStep(xt, state, weight, ot, channels, kernel);
            ShiftConvState(xt, state, channels, kernel);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Conv1dStep(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> state,   // [kernel-1, channels] oldest-first
        ReadOnlySpan<float> weight,  // [kernel, channels]
        Span<float> output,
        int channels,
        int kernel)
    {
        // output[c] = Σ_{k=0..K-2} weight[k,c] · state[k,c]  +  weight[K-1,c] · x[c]
        // Initialize from the current-token tap so we can accumulate state taps without a branch.
        ReadOnlySpan<float> wCur = weight.Slice((kernel - 1) * channels, channels);
        for (int c = 0; c < channels; c++)
            output[c] = wCur[c] * x[c];

        for (int k = 0; k < kernel - 1; k++)
        {
            ReadOnlySpan<float> wk = weight.Slice(k * channels, channels);
            ReadOnlySpan<float> sk = state.Slice(k * channels, channels);
            for (int c = 0; c < channels; c++)
                output[c] += wk[c] * sk[c];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ShiftConvState(
        ReadOnlySpan<float> x,
        Span<float> state,
        int channels,
        int kernel)
    {
        if (kernel <= 1) return;  // nothing to retain
        int retained = kernel - 1;
        // Drop oldest row (k=0); shift rows [1..retained-1] down to [0..retained-2].
        if (retained > 1)
        {
            var src = state.Slice(channels, (retained - 1) * channels);
            var dst = state[..((retained - 1) * channels)];
            src.CopyTo(dst);
        }
        // Write current x into the newest slot (k = retained-1).
        x.CopyTo(state.Slice((retained - 1) * channels, channels));
    }

    // ────────────────────────────────────────────────────────────────────
    //  L2 normalization at head granularity
    //
    //  Matches llama.cpp's ggml_compute_forward_l2_norm_f32:
    //      scale = 1 / max(sqrt(Σ x²), eps)
    //  This is NOT the more common "sqrt(sum_sq + eps)" — the difference only
    //  matters for very small vectors but Phase 5 parity testing requires the
    //  exact ggml convention.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// In-place L2 normalize each <paramref name="headDim"/>-sized slice of
    /// <paramref name="x"/> independently. There are <paramref name="numHeads"/>
    /// such slices laid out contiguously.
    /// </summary>
    /// <param name="x">Buffer of length <c>numHeads · headDim</c>.</param>
    /// <param name="numHeads">Number of heads.</param>
    /// <param name="headDim">Per-head feature count.</param>
    /// <param name="eps">Lower bound on the divisor (matches ggml_l2_norm).</param>
    public static void L2NormPerHead(Span<float> x, int numHeads, int headDim, float eps = 1e-6f)
    {
        if (x.Length != numHeads * headDim) throw new ArgumentException("x length != numHeads*headDim");
        if (eps < 0f) throw new ArgumentOutOfRangeException(nameof(eps));

        for (int h = 0; h < numHeads; h++)
        {
            var slice = x.Slice(h * headDim, headDim);
            double sum = 0.0;
            for (int i = 0; i < headDim; i++)
            {
                float v = slice[i];
                sum += (double)v * v;
            }
            float norm = MathF.Sqrt((float)sum);
            float divisor = norm > eps ? norm : eps;
            float invDiv = 1.0f / divisor;
            for (int i = 0; i < headDim; i++)
                slice[i] *= invDiv;
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Repeat-interleave heads (GQA-style K-broadcast)
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Broadcast each src head to <paramref name="repeat"/> dst heads using <b>tile</b>
    /// semantics — i.e. dst head <c>h</c> takes its data from <c>src[h % srcHeads]</c>.
    /// For qwen35moe (Hk=16 → Hv=32, repeat=2) this produces the pairing
    /// <c>(0,16), (1,17), ..., (15,31)</c>, matching llama.cpp's
    /// <c>ggml_compute_forward_gated_delta_net_one_chunk</c> (<c>iq1 = iv1 % neq1</c>)
    /// and <c>ggml_compute_forward_repeat_f32</c> (outer rep loop × inner src loop).
    /// </summary>
    /// <remarks>
    /// Note this is <i>NOT</i> torch's <c>repeat_interleave</c> pattern
    /// (<c>(0,1), (2,3), ...</c>) — that pairing would mix heads incorrectly for
    /// GDN's GQA-style K→V broadcast.
    /// </remarks>
    /// <param name="src">Input of shape <c>[srcHeads, headDim]</c> row-major.</param>
    /// <param name="dst">Output of shape <c>[srcHeads · repeat, headDim]</c> row-major.</param>
    /// <param name="srcHeads">Source head count.</param>
    /// <param name="repeat">Repetition factor (e.g. 2 for qwen35moe).</param>
    /// <param name="headDim">Per-head feature count.</param>
    public static void TileHeads(
        ReadOnlySpan<float> src,
        Span<float> dst,
        int srcHeads,
        int repeat,
        int headDim)
    {
        if (repeat < 1) throw new ArgumentOutOfRangeException(nameof(repeat));
        if (src.Length != srcHeads * headDim) throw new ArgumentException("src length mismatch");
        if (dst.Length != srcHeads * repeat * headDim) throw new ArgumentException("dst length mismatch");

        // Tile pattern: dst[h * headDim] copies from src[(h % srcHeads) * headDim]
        // ⇒ outer loop over repetitions, inner loop over src heads.
        for (int r = 0; r < repeat; r++)
        {
            int dstHeadOffset = r * srcHeads;
            for (int h = 0; h < srcHeads; h++)
                src.Slice(h * headDim, headDim)
                    .CopyTo(dst.Slice((dstHeadOffset + h) * headDim, headDim));
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Gated DeltaNet recurrent step
    //
    //  State layout: per-head row-major D×D, flattened over Hv heads:
    //      state[h * D*D + i * D + j] ≡ S_h[i, j]
    //  with i ≡ "key axis" and j ≡ "value/output axis", matching the index
    //  convention worked out from the ggml ops at the top of this file.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Single-token Gated DeltaNet update. Mutates <paramref name="state"/> in
    /// place and writes the post-norm, post-gate output to <paramref name="output"/>.
    /// </summary>
    /// <param name="q">Query tensor of shape <c>[numVHeads, headDim]</c>
    /// (K already broadcast from Hk to Hv by the caller — for true Q this is the
    /// already-projected Q from the conv stream).</param>
    /// <param name="k">Key tensor of shape <c>[numVHeads, headDim]</c>, broadcast
    /// from <c>numKHeads</c> by the caller.</param>
    /// <param name="v">Value tensor of shape <c>[numVHeads, headDim]</c>.</param>
    /// <param name="alphaIn">Per-v-head alpha pre-activation (from <c>ssm_alpha</c>
    /// projection), length <c>numVHeads</c>.</param>
    /// <param name="beta">Per-v-head beta pre-activation (from <c>ssm_beta</c>
    /// projection), length <c>numVHeads</c>.</param>
    /// <param name="ssmA">Per-v-head decay coefficient (negative, from <c>ssm_a</c>),
    /// length <c>numVHeads</c>.</param>
    /// <param name="dtBias">Per-v-head bias added to alpha before softplus,
    /// length <c>numVHeads</c>.</param>
    /// <param name="normWeight">Per-head-dim RMSNorm gain shared across all heads
    /// (from <c>ssm_norm.weight</c>), length <c>headDim</c>.</param>
    /// <param name="z">Pre-activation of the SiLU gate (from <c>attn_gate</c>),
    /// length <c>numVHeads · headDim</c>.</param>
    /// <param name="state">Per-head D×D state matrices, length
    /// <c>numVHeads · headDim · headDim</c>. Updated in place.</param>
    /// <param name="output">Output buffer, length <c>numVHeads · headDim</c>.</param>
    /// <param name="numVHeads">Number of value heads (Hv = 32 for qwen35moe).</param>
    /// <param name="headDim">Head dimension (D = 128 for qwen35moe).</param>
    /// <param name="normEps">RMSNorm epsilon for the post-recurrence per-head norm.</param>
    public static void GdnRecurrenceDecode(
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> alphaIn,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> ssmA,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> z,
        Span<float> state,
        Span<float> output,
        int numVHeads,
        int headDim,
        float normEps = 1e-6f,
        int layer = -1,
        int position = -1)
    {
        ValidateRecurrenceArgs(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z, state, output,
            numVHeads, headDim);

        GdnStepInternal(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            state, output, numVHeads, headDim, normEps, layer, position);
    }

    /// <summary>
    /// Sequential T-step Gated DeltaNet recurrence. Tensors except
    /// <paramref name="state"/> and <paramref name="normWeight"/> are
    /// <c>[tokens, ...]</c> row-major (token-major outer dim).
    /// </summary>
    /// <param name="tokens">Number of tokens to process.</param>
    /// <param name="q">[tokens, numVHeads, headDim] row-major.</param>
    /// <param name="k">[tokens, numVHeads, headDim] row-major.</param>
    /// <param name="v">[tokens, numVHeads, headDim] row-major.</param>
    /// <param name="alphaIn">[tokens, numVHeads] row-major.</param>
    /// <param name="beta">[tokens, numVHeads] row-major.</param>
    /// <param name="ssmA">[numVHeads] — shared across tokens.</param>
    /// <param name="dtBias">[numVHeads] — shared across tokens.</param>
    /// <param name="normWeight">[headDim] — shared across tokens and heads.</param>
    /// <param name="z">[tokens, numVHeads * headDim] row-major.</param>
    /// <param name="state">[numVHeads * headDim * headDim] — updated in place.</param>
    /// <param name="output">[tokens, numVHeads, headDim] row-major.</param>
    public static void GdnRecurrencePrefill(
        int tokens,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> alphaIn,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> ssmA,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> z,
        Span<float> state,
        Span<float> output,
        int numVHeads,
        int headDim,
        float normEps = 1e-6f)
    {
        int hv = numVHeads;
        int d = headDim;
        int perTokQkv = hv * d;
        int perTokScalar = hv;

        if (tokens < 0) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (q.Length != tokens * perTokQkv) throw new ArgumentException("q length mismatch");
        if (k.Length != tokens * perTokQkv) throw new ArgumentException("k length mismatch");
        if (v.Length != tokens * perTokQkv) throw new ArgumentException("v length mismatch");
        if (alphaIn.Length != tokens * perTokScalar) throw new ArgumentException("alphaIn length mismatch");
        if (beta.Length != tokens * perTokScalar) throw new ArgumentException("beta length mismatch");
        if (ssmA.Length != hv) throw new ArgumentException("ssmA length mismatch");
        if (dtBias.Length != hv) throw new ArgumentException("dtBias length mismatch");
        if (normWeight.Length != d) throw new ArgumentException("normWeight length mismatch");
        if (z.Length != tokens * perTokQkv) throw new ArgumentException("z length mismatch");
        if (state.Length != hv * d * d) throw new ArgumentException("state length mismatch");
        if (output.Length != tokens * perTokQkv) throw new ArgumentException("output length mismatch");

        for (int t = 0; t < tokens; t++)
        {
            GdnStepInternal(
                q.Slice(t * perTokQkv, perTokQkv),
                k.Slice(t * perTokQkv, perTokQkv),
                v.Slice(t * perTokQkv, perTokQkv),
                alphaIn.Slice(t * perTokScalar, perTokScalar),
                beta.Slice(t * perTokScalar, perTokScalar),
                ssmA, dtBias, normWeight,
                z.Slice(t * perTokQkv, perTokQkv),
                state,
                output.Slice(t * perTokQkv, perTokQkv),
                hv, d, normEps);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Chunked (parallel) Gated DeltaNet prefill — the "chunk_gated_delta_rule"
    //  form popularised by FLA / FlashQLA (https://github.com/QwenLM/FlashQLA).
    //
    //  Instead of walking tokens one at a time (GdnRecurrencePrefill), a chunk of
    //  C tokens is resolved with a handful of dense linear-algebra passes whose
    //  inner kernels are matmuls — the shape that maps to Tensor Cores on GPU and
    //  BLAS GEMM on CPU. This is the algorithmic prerequisite for the FlashQLA-style
    //  GPU kernel: it is the numerical oracle the GPU path is parity-checked against.
    //
    //  Derivation (per v-head; incoming chunk state S0, local token index t = 0..C-1,
    //  i ≡ key axis, j ≡ value axis, layout S[i,j] = state[i*d + j] as elsewhere).
    //
    //  Sequential step (token t, S_prev = state after token t-1, S_{-1} = S0):
    //      a_t      = exp(softplus(alpha_t + dtBias)·ssmA)          // scalar decay, ssmA<0 ⇒ a_t∈(0,1]
    //      b_t      = sigmoid(beta_t)                               // scalar
    //      p_t      = a_t · (S_prevᵀ k_t)                           // readout BEFORE the t-th update
    //      u_t      = b_t · (v_t − p_t)                             // "pseudo-value" (the rank-1 payload)
    //      S_t      = a_t·S_prev + k_t ⊗ u_t
    //      o_t      = (S_tᵀ q_t)/√d                                 // readout AFTER the t-th update
    //
    //  Unrolling S_t in terms of S0 with cumulative log-decay cum_t = Σ_{s≤t} log a_s
    //  (so g_t = exp(cum_t) and the inter-token decay product Π_{r=s+1..t} a_r =
    //  exp(cum_t − cum_s) ≤ 1) yields a strictly-lower-triangular system for U:
    //
    //      A[t,s] = b_t · exp(cum_t−cum_s) · (k_sᵀ k_t)             (s < t)
    //      rhs_t  = b_t·v_t − b_t·g_t·(S0ᵀ k_t)
    //      (I + A) U = RHS         ⇒  u_t = rhs_t − Σ_{s<t} A[t,s] u_s   (forward substitution)
    //
    //  Outputs and the carried-out state are then pure matmul-shaped reductions:
    //      o_t   = ( g_t·(S0ᵀ q_t) + Σ_{s≤t} exp(cum_t−cum_s)·(k_sᵀ q_t)·u_s ) / √d
    //      S_new = g_{C-1}·S0 + Σ_s exp(cum_{C-1}−cum_s)·(k_s ⊗ u_s)
    //
    //  log a_t is computed directly as softplus(alpha)·ssmA (no log/exp round-trip),
    //  and all decay products go through cum_t differences so g_t may underflow to 0
    //  (fully-decayed state) without ever forming an over/underflowing ratio. The
    //  post-recurrence per-head RMSNorm + SiLU(z) gate is byte-identical to the
    //  sequential path. Output/state differ from GdnRecurrencePrefill only by FP
    //  reduction order (this path accumulates in double), i.e. to within tolerance.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>Default chunk length for <see cref="GdnRecurrenceChunkedPrefill"/>.
    /// 64 matches the FLA / FlashQLA tile and keeps the intra-chunk decay products
    /// well-conditioned.</summary>
    public const int DefaultGdnChunkSize = 64;

    /// <summary>
    /// Chunk-parallel Gated DeltaNet prefill. Same inputs/outputs and in-place state
    /// update as <see cref="GdnRecurrencePrefill"/>, but resolves each
    /// <paramref name="chunkSize"/>-token block with the parallel "chunk_gated_delta_rule"
    /// form (matmul-shaped) instead of a per-token scan. Numerically equal to the
    /// sequential path up to floating-point reduction order.
    /// </summary>
    /// <param name="chunkSize">Tokens per parallel block (default
    /// <see cref="DefaultGdnChunkSize"/>). Larger blocks amortise more but widen the
    /// intra-chunk decay span; 64 is the validated default.</param>
    public static unsafe void GdnRecurrenceChunkedPrefill(
        int tokens,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> alphaIn,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> ssmA,
        ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> z,
        Span<float> state,
        Span<float> output,
        int numVHeads,
        int headDim,
        float normEps = 1e-6f,
        int chunkSize = DefaultGdnChunkSize)
    {
        int hv = numVHeads;
        int d = headDim;
        int perTokQkv = hv * d;
        int perTokScalar = hv;

        if (tokens < 0) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (chunkSize < 1) throw new ArgumentOutOfRangeException(nameof(chunkSize));
        if (q.Length != tokens * perTokQkv) throw new ArgumentException("q length mismatch");
        if (k.Length != tokens * perTokQkv) throw new ArgumentException("k length mismatch");
        if (v.Length != tokens * perTokQkv) throw new ArgumentException("v length mismatch");
        if (alphaIn.Length != tokens * perTokScalar) throw new ArgumentException("alphaIn length mismatch");
        if (beta.Length != tokens * perTokScalar) throw new ArgumentException("beta length mismatch");
        if (ssmA.Length != hv) throw new ArgumentException("ssmA length mismatch");
        if (dtBias.Length != hv) throw new ArgumentException("dtBias length mismatch");
        if (normWeight.Length != d) throw new ArgumentException("normWeight length mismatch");
        if (z.Length != tokens * perTokQkv) throw new ArgumentException("z length mismatch");
        if (state.Length != hv * d * d) throw new ArgumentException("state length mismatch");
        if (output.Length != tokens * perTokQkv) throw new ArgumentException("output length mismatch");

        if (tokens == 0) return;

        int dd = d * d;
        float readoutScale = 1.0f / MathF.Sqrt((float)d);
        int C = chunkSize;

        // Per-chunk scratch, allocated once per call and reused across heads/chunks.
        //   cum/g/bScal: per-token scalars (≤ C). stackalloc'd for the common
        //     small-chunk case (no heap traffic / bounds checks on the hot scan);
        //     a caller passing an atypically large chunkSize falls back to the heap
        //     so the stack can't overflow (the GPU sibling pins GDN_CHUNK at 64).
        //   u   : pseudo-values U, row-major [C, d]
        //   proj: batched state projection S0ᵀK then (reused) S0ᵀQ, row-major [C, d]
        //         (both too large for the stack — always heap).
        const int stackChunkCap = 256;
        Span<double> cumSpan = C <= stackChunkCap ? stackalloc double[C] : new double[C];
        Span<float> gSpan = C <= stackChunkCap ? stackalloc float[C] : new float[C];
        Span<float> bSpan = C <= stackChunkCap ? stackalloc float[C] : new float[C];
        float[] u = new float[C * d];
        float[] proj = new float[C * d];

        fixed (double* cum = cumSpan)
        fixed (float* gP = gSpan, bP = bSpan)
        fixed (float* qP = q, kP = k, vP = v, zP = z, normP = normWeight,
                      statePtr = state, outPtr = output,
                      uP = u, projP = proj)
        {
            for (int h = 0; h < hv; h++)
            {
                float* S = statePtr + (long)h * dd;
                float ssmAh = ssmA[h];
                float dtBiasH = dtBias[h];

                for (int c0 = 0; c0 < tokens; c0 += C)
                {
                    int cN = Math.Min(C, tokens - c0);

                    // ── Per-token scalars: cumulative log-decay, g_t, b_t. ────
                    double run = 0.0;
                    for (int t = 0; t < cN; t++)
                    {
                        int gt = c0 + t;
                        float alphaX = alphaIn[gt * perTokScalar + h] + dtBiasH;
                        float dt = alphaX >= 20.0f ? alphaX : MathF.Log(1.0f + MathF.Exp(alphaX));
                        run += (double)dt * ssmAh;     // log a_t (≤ 0)
                        cum[t] = run;
                        gP[t] = (float)Math.Exp(run);
                        bP[t] = 1.0f / (1.0f + MathF.Exp(-beta[gt * perTokScalar + h]));
                    }

                    // ── Batched S0ᵀK → proj[t,:] (read S0 once, stream over rows). ──
                    //   proj[t,j] = Σ_i K_t[i]·S0[i,j].  Outer i keeps the S0 row hot.
                    new Span<float>(projP, cN * d).Clear();
                    for (int i = 0; i < d; i++)
                    {
                        float* s0row = S + (long)i * d;
                        for (int t = 0; t < cN; t++)
                        {
                            float ki = qkAt(kP, c0 + t, h, i, hv, d);
                            if (ki != 0f) AxpyF32(projP + (long)t * d, s0row, ki, d);
                        }
                    }

                    // ── Forward substitution: u_t = rhs_t − Σ_{s<t} A[t,s] u_s. ──
                    for (int t = 0; t < cN; t++)
                    {
                        float* kt = kP + (long)(((c0 + t) * hv + h) * d);
                        float* vt = vP + (long)(((c0 + t) * hv + h) * d);
                        float* ut = uP + (long)t * d;
                        float bt = bP[t], gT = gP[t];
                        float* pkt = projP + (long)t * d;

                        for (int j = 0; j < d; j++) ut[j] = bt * (vt[j] - gT * pkt[j]);

                        for (int s = 0; s < t; s++)
                        {
                            float* ks = kP + (long)(((c0 + s) * hv + h) * d);
                            float kdot = SimdKernels.DotF32(ks, kt, d);
                            float a = bt * (float)Math.Exp(cum[t] - cum[s]) * kdot;
                            if (a != 0f) AxpyF32(ut, uP + (long)s * d, -a, d);
                        }
                    }

                    // ── Batched S0ᵀQ → proj[t,:] (reuse the buffer; K no longer needed). ──
                    new Span<float>(projP, cN * d).Clear();
                    for (int i = 0; i < d; i++)
                    {
                        float* s0row = S + (long)i * d;
                        for (int t = 0; t < cN; t++)
                        {
                            float qi = qkAt(qP, c0 + t, h, i, hv, d);
                            if (qi != 0f) AxpyF32(projP + (long)t * d, s0row, qi, d);
                        }
                    }

                    // ── Outputs + per-head RMSNorm + SiLU(z) gate. ────────────
                    for (int t = 0; t < cN; t++)
                    {
                        float* qt = qP + (long)(((c0 + t) * hv + h) * d);
                        float* oh = outPtr + (long)(((c0 + t) * hv + h) * d);
                        float gT = gP[t];
                        float* pqt = projP + (long)t * d;

                        for (int j = 0; j < d; j++) oh[j] = gT * pqt[j];

                        for (int s = 0; s <= t; s++)
                        {
                            float* ks = kP + (long)(((c0 + s) * hv + h) * d);
                            float kqdot = SimdKernels.DotF32(ks, qt, d);
                            float r = (float)Math.Exp(cum[t] - cum[s]) * kqdot;
                            if (r != 0f) AxpyF32(oh, uP + (long)s * d, r, d);
                        }

                        for (int j = 0; j < d; j++) oh[j] *= readoutScale;

                        double sumSq = 0.0;
                        for (int j = 0; j < d; j++) { float ov = oh[j]; sumSq += (double)ov * ov; }
                        float scale = 1.0f / MathF.Sqrt((float)(sumSq / d) + normEps);
                        float* zt = zP + (long)(((c0 + t) * hv + h) * d);
                        for (int j = 0; j < d; j++)
                        {
                            float normed = oh[j] * scale * normP[j];
                            float zv = zt[j];
                            float silu = zv / (1.0f + MathF.Exp(-zv));
                            oh[j] = normed * silu;
                        }
                    }

                    // ── Carry state out: S_new = g_{C-1}·S0 + Σ_s r_s·(k_s ⊗ u_s). ──
                    float gLast = gP[cN - 1];
                    for (int idx = 0; idx < dd; idx++) S[idx] *= gLast;
                    for (int s = 0; s < cN; s++)
                    {
                        float* ks = kP + (long)(((c0 + s) * hv + h) * d);
                        float* us = uP + (long)s * d;
                        float rs = (float)Math.Exp(cum[cN - 1] - cum[s]);
                        for (int i = 0; i < d; i++)
                        {
                            float coeff = rs * ks[i];
                            if (coeff != 0f) AxpyF32(S + (long)i * d, us, coeff, d);
                        }
                    }
                }
            }
        }
    }

    /// <summary>Per-token/-head element access into a <c>[tokens, hv, d]</c> buffer.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float qkAt(float* buf, int tok, int head, int i, int hv, int d) =>
        buf[(long)((tok * hv + head) * d) + i];

    /// <summary><c>y[0..n] += c · x[0..n]</c> (FMA-accelerated AXPY).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AxpyF32(float* y, float* x, float c, int n)
    {
        int i = 0;
        if (System.Runtime.Intrinsics.X86.Fma.IsSupported)
        {
            var vc = System.Runtime.Intrinsics.Vector256.Create(c);
            for (; i + 8 <= n; i += 8)
            {
                var vy = System.Runtime.Intrinsics.X86.Avx.LoadVector256(y + i);
                var vx = System.Runtime.Intrinsics.X86.Avx.LoadVector256(x + i);
                System.Runtime.Intrinsics.X86.Avx.Store(y + i,
                    System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vx, vc, vy));
            }
        }
        for (; i < n; i++) y[i] += c * x[i];
    }

    private static void ValidateRecurrenceArgs(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
        ReadOnlySpan<float> alphaIn, ReadOnlySpan<float> beta,
        ReadOnlySpan<float> ssmA, ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> normWeight, ReadOnlySpan<float> z,
        Span<float> state, Span<float> output,
        int numVHeads, int headDim)
    {
        int qkvLen = numVHeads * headDim;
        if (q.Length != qkvLen) throw new ArgumentException("q length mismatch");
        if (k.Length != qkvLen) throw new ArgumentException("k length mismatch");
        if (v.Length != qkvLen) throw new ArgumentException("v length mismatch");
        if (alphaIn.Length != numVHeads) throw new ArgumentException("alphaIn length mismatch");
        if (beta.Length != numVHeads) throw new ArgumentException("beta length mismatch");
        if (ssmA.Length != numVHeads) throw new ArgumentException("ssmA length mismatch");
        if (dtBias.Length != numVHeads) throw new ArgumentException("dtBias length mismatch");
        if (normWeight.Length != headDim) throw new ArgumentException("normWeight length mismatch");
        if (z.Length != qkvLen) throw new ArgumentException("z length mismatch");
        if (state.Length != numVHeads * headDim * headDim) throw new ArgumentException("state length mismatch");
        if (output.Length != qkvLen) throw new ArgumentException("output length mismatch");
    }

    private static void GdnStepInternal(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
        ReadOnlySpan<float> alphaIn, ReadOnlySpan<float> beta,
        ReadOnlySpan<float> ssmA, ReadOnlySpan<float> dtBias,
        ReadOnlySpan<float> normWeight, ReadOnlySpan<float> z,
        Span<float> state, Span<float> output,
        int hv, int d, float normEps,
        int layer = -1, int position = -1)
    {
        int dd = d * d;

        // Scratch on the stack to avoid allocations. headDim ≤ 128 in the
        // target model so 128 floats = 512 bytes is safe on the stack.
        Span<float> p = stackalloc float[d];
        Span<float> dvec = stackalloc float[d];

        // Readout scale: 1 / sqrt(headDim). Matches llama.cpp
        // ggml_compute_forward_gated_delta_net_one_chunk (ops.cpp:10547,
        // attn_data[j] = sum * scale at :10613). RMSNorm is scale-invariant
        // EXCEPT at its eps floor, so omitting this caused per-head magnitude
        // drift on small inner products → gibberish decode at qwen35moe scale.
        float readoutScale = 1.0f / MathF.Sqrt((float)d);

        bool traceThis = ShouldTraceGdn(layer, position);

        for (int h = 0; h < hv; h++)
        {
            // Per-head scalar gates.
            float alphaX = alphaIn[h] + dtBias[h];
            float dt = alphaX >= 20.0f ? alphaX : MathF.Log(1.0f + MathF.Exp(alphaX));   // softplus
            float decay = MathF.Exp(dt * ssmA[h]);                                       // ssmA already negative
            float bScalar = 1.0f / (1.0f + MathF.Exp(-beta[h]));                         // sigmoid

            // Per-head slices.
            Span<float> S = state.Slice(h * dd, dd);
            ReadOnlySpan<float> kh = k.Slice(h * d, d);
            ReadOnlySpan<float> vh = v.Slice(h * d, d);
            ReadOnlySpan<float> qh = q.Slice(h * d, d);
            Span<float> oh = output.Slice(h * d, d);

            // (1) Decay: S *= decay.
            for (int idx = 0; idx < dd; idx++)
                S[idx] *= decay;

            if (traceThis && h == 0)
                EmitGdnInternal(layer, position, "post-decay",
                    decay: decay, bScalar: bScalar, S: S, payload: default, d: d);

            // (2) Predict: p[j] = Σ_i S[i,j] · k[i].  (S^T @ k)
            // Iterate rows i, accumulating k[i] * S[i, :] into p.
            //
            // NOTE: ggml stores the state TRANSPOSED (s_out[j*S_v + i] = S[i,j])
            // and computes p[j] as a contiguous-row ggml_vec_dot_f32 with a
            // 4-accumulator tree-reduce. That reordering was hypothesized to be
            // the source of the 0.26% sum / 0.06% l2 drift at L0 gdn-out on
            // Qwen3.6-27B-MTP pos-12. A direct port — per-head transpose-in/out
            // scratch with ggml-order dot for predict+readout, FMA AXPY for the
            // rank-1 update — was tested 2026-05-26 and produced zero change at
            // L0 gdn-out (l2=3.86054 sum=-7.79791 either way vs llama r12
            // 3.8582/-7.7774). The hypothesis was re-prioritised at L40+ where
            // state magnitudes are larger; see SHARPI_TRACE_GDN_INTERNAL.
            p.Clear();
            for (int i = 0; i < d; i++)
            {
                float ki = kh[i];
                int rowBase = i * d;
                for (int j = 0; j < d; j++)
                    p[j] += ki * S[rowBase + j];
            }

            // (3) Delta: d[j] = b · (v[j] - p[j]).
            for (int j = 0; j < d; j++)
                dvec[j] = bScalar * (vh[j] - p[j]);

            if (traceThis && h == 0)
                EmitGdnInternal(layer, position, "dvec",
                    decay: float.NaN, bScalar: float.NaN, S: default, payload: dvec, d: d);

            // (4) Rank-1 update: S[i,j] += k[i] · d[j].
            for (int i = 0; i < d; i++)
            {
                float ki = kh[i];
                int rowBase = i * d;
                for (int j = 0; j < d; j++)
                    S[rowBase + j] += ki * dvec[j];
            }

            if (traceThis && h == 0)
                EmitGdnInternal(layer, position, "post-rank1",
                    decay: float.NaN, bScalar: float.NaN, S: S, payload: default, d: d);

            // (5) Readout: o[j] = (Σ_i S[i,j] · q[i]) / sqrt(d).  (S^T @ q, scaled)
            oh.Clear();
            for (int i = 0; i < d; i++)
            {
                float qi = qh[i];
                int rowBase = i * d;
                for (int j = 0; j < d; j++)
                    oh[j] += qi * S[rowBase + j];
            }
            for (int j = 0; j < d; j++)
                oh[j] *= readoutScale;

            if (traceThis && h == 0)
                EmitGdnInternal(layer, position, "p-readout",
                    decay: float.NaN, bScalar: float.NaN, S: default, payload: oh, d: d);

            // (6) Per-head RMSNorm with shared headDim-wide gain.
            double sumSq = 0.0;
            for (int j = 0; j < d; j++)
            {
                float ov = oh[j];
                sumSq += (double)ov * ov;
            }
            float scale = 1.0f / MathF.Sqrt((float)(sumSq / d) + normEps);
            for (int j = 0; j < d; j++)
                oh[j] = oh[j] * scale * normWeight[j];

            // (7) SiLU(z) gate: o *= z * sigmoid(z).
            ReadOnlySpan<float> zh = z.Slice(h * d, d);
            for (int j = 0; j < d; j++)
            {
                float zv = zh[j];
                float silu = zv / (1.0f + MathF.Exp(-zv));
                oh[j] *= silu;
            }
        }
    }

    // SHARPI_TRACE_GDN_INTERNAL emission. Format mirrors EmitBufTrace in
    // HybridGdnForwardPass so the grep recipe is the same on both sides.
    // For "post-decay" we also report the scalar decay and bScalar so the
    // gate magnitudes are visible alongside the state norm.
    private static void EmitGdnInternal(
        int layer, int position, string tag,
        float decay, float bScalar, ReadOnlySpan<float> S, ReadOnlySpan<float> payload, int d)
    {
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var sb = new System.Text.StringBuilder(384);
        sb.Append("[pos=").Append(position).Append(" L").Append(layer)
          .Append(" gdn-internal head=0 ").Append(tag).Append(']');

        if (!float.IsNaN(decay))
            sb.Append(" decay=").Append(decay.ToString("G7", inv));
        if (!float.IsNaN(bScalar))
            sb.Append(" b=").Append(bScalar.ToString("G7", inv));

        if (S.Length > 0)
        {
            double s2 = 0, su = 0;
            for (int i = 0; i < S.Length; i++) { double vv = S[i]; s2 += vv * vv; su += vv; }
            sb.Append(" S_l2=").Append(Math.Sqrt(s2).ToString("G6", inv));
            sb.Append(" S_sum=").Append(((float)su).ToString("G6", inv));
        }

        if (payload.Length > 0)
        {
            double s2 = 0, su = 0;
            for (int i = 0; i < payload.Length; i++) { double vv = payload[i]; s2 += vv * vv; su += vv; }
            sb.Append(" l2=").Append(Math.Sqrt(s2).ToString("G6", inv));
            sb.Append(" sum=").Append(((float)su).ToString("G6", inv));
            sb.Append(" first8=[");
            int k = Math.Min(8, payload.Length);
            for (int i = 0; i < k; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(payload[i].ToString("G6", inv));
            }
            sb.Append(']');
        }

        Console.Error.WriteLine(sb.ToString());
    }
}
