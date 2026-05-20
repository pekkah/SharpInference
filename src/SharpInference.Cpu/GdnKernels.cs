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
        float normEps = 1e-6f)
    {
        ValidateRecurrenceArgs(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z, state, output,
            numVHeads, headDim);

        GdnStepInternal(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            state, output, numVHeads, headDim, normEps);
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
        int hv, int d, float normEps)
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

            // (2) Predict: p[j] = Σ_i S[i,j] · k[i].  (S^T @ k)
            // Iterate rows i, accumulating k[i] * S[i, :] into p.
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

            // (4) Rank-1 update: S[i,j] += k[i] · d[j].
            for (int i = 0; i < d; i++)
            {
                float ki = kh[i];
                int rowBase = i * d;
                for (int j = 0; j < d; j++)
                    S[rowBase + j] += ki * dvec[j];
            }

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
}
