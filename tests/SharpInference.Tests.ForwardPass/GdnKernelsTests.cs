using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Phase 2 unit tests for the Gated DeltaNet (GDN) CPU kernels that back
/// the qwen35moe port. These exercise the kernels in isolation (no model
/// load, no ForwardPass integration).
/// </summary>
public sealed class GdnKernelsTests
{
    // ────────────────────────────────────────────────────────────────────
    //  Causal depthwise conv1d
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Conv1dDecode_HandComputed_K2()
    {
        // channels=4, kernel=2. State holds 1 prior token.
        // weight layout [kernel, channels]: row 0 = older tap, row 1 = current tap.
        //   weight[0] = [0.1, 0.2, 0.3, 0.4]   (applied to state)
        //   weight[1] = [1.0, 1.0, 1.0, 1.0]   (applied to current x)
        // state[0]    = [10, 20, 30, 40]       (prior token)
        //  x          = [1,  2,  3,  4 ]
        // expected[c] = state[c] * weight[0,c] + x[c] * weight[1,c]
        //             = [10*0.1 + 1, 20*0.2 + 2, 30*0.3 + 3, 40*0.4 + 4]
        //             = [2.0, 6.0, 12.0, 20.0]
        const int channels = 4;
        const int kernel = 2;

        float[] weight = [0.1f, 0.2f, 0.3f, 0.4f,  1f, 1f, 1f, 1f];
        float[] state  = [10f, 20f, 30f, 40f];
        float[] x      = [1f, 2f, 3f, 4f];
        float[] output = new float[channels];

        GdnKernels.CausalDepthwiseConv1dDecode(x, state, weight, output, channels, kernel);

        Assert.Equal(2.0f,  output[0], 6);
        Assert.Equal(6.0f,  output[1], 6);
        Assert.Equal(12.0f, output[2], 6);
        Assert.Equal(20.0f, output[3], 6);

        // After the step, state should equal x (kernel=2 → 1-row state, replaced wholesale).
        Assert.Equal(x, state);
    }

    [Fact]
    public void Conv1dDecodePrefillEquivalence_K4()
    {
        const int tokens = 5;
        const int channels = 4;
        const int kernel = 4;

        var rng = new Random(0x6D6F65);
        float[] x = RandomArray(rng, tokens * channels, -1f, 1f);
        float[] weight = RandomArray(rng, kernel * channels, -1f, 1f);

        // Path A: 5× decode starting from zero state.
        float[] stateA = new float[(kernel - 1) * channels];
        float[] outputA = new float[tokens * channels];
        for (int t = 0; t < tokens; t++)
        {
            GdnKernels.CausalDepthwiseConv1dDecode(
                x.AsSpan(t * channels, channels),
                stateA,
                weight,
                outputA.AsSpan(t * channels, channels),
                channels, kernel);
        }

        // Path B: single prefill call.
        float[] stateB = new float[(kernel - 1) * channels];
        float[] outputB = new float[tokens * channels];
        GdnKernels.CausalDepthwiseConv1dPrefill(x, stateB, weight, outputB, tokens, channels, kernel);

        for (int i = 0; i < outputA.Length; i++)
            Assert.Equal(outputA[i], outputB[i], 6);
        for (int i = 0; i < stateA.Length; i++)
            Assert.Equal(stateA[i], stateB[i], 6);
    }

    [Fact]
    public void Conv1dPrefill_StateContainsLastKernelMinusOneTokens()
    {
        // After processing tokens t=0..T-1 with K-1 retained, state should equal x[T-K+1..T-1].
        const int tokens = 5;
        const int channels = 3;
        const int kernel = 3;

        var rng = new Random(42);
        float[] x = RandomArray(rng, tokens * channels, -1f, 1f);
        float[] weight = RandomArray(rng, kernel * channels, -1f, 1f);
        float[] state = new float[(kernel - 1) * channels];
        float[] output = new float[tokens * channels];

        GdnKernels.CausalDepthwiseConv1dPrefill(x, state, weight, output, tokens, channels, kernel);

        // Oldest retained = token T-K+1 = T-2; newest retained = token T-1.
        // state[0..C] should be x[T-2], state[C..2C] should be x[T-1].
        for (int c = 0; c < channels; c++)
        {
            Assert.Equal(x[(tokens - 2) * channels + c], state[c], 6);
            Assert.Equal(x[(tokens - 1) * channels + c], state[channels + c], 6);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  L2NormPerHead
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void L2NormPerHead_KnownValues()
    {
        // 2 heads × 4 dims. Head 0 has norm 5, head 1 has norm 1.
        float[] x = [3f, 4f, 0f, 0f,   1f, 0f, 0f, 0f];

        GdnKernels.L2NormPerHead(x, numHeads: 2, headDim: 4);

        Assert.Equal(0.6f, x[0], 6);
        Assert.Equal(0.8f, x[1], 6);
        Assert.Equal(0.0f, x[2], 6);
        Assert.Equal(0.0f, x[3], 6);

        Assert.Equal(1.0f, x[4], 6);
        Assert.Equal(0.0f, x[5], 6);
        Assert.Equal(0.0f, x[6], 6);
        Assert.Equal(0.0f, x[7], 6);
    }

    [Fact]
    public void L2NormPerHead_HandlesNearZeroVectorWithEpsFloor()
    {
        // ggml convention: divisor = max(sqrt(sum_sq), eps). For a near-zero
        // input the divisor falls back to eps, so the output is x/eps and
        // must be finite.
        float[] x = [1e-8f, 0f, 0f, 0f];
        GdnKernels.L2NormPerHead(x, numHeads: 1, headDim: 4, eps: 1e-6f);

        Assert.True(float.IsFinite(x[0]));
        // x[0] / eps = 1e-8 / 1e-6 = 0.01
        Assert.Equal(0.01f, x[0], 5);
        Assert.Equal(0f, x[1], 6);
    }

    // ────────────────────────────────────────────────────────────────────
    //  RepeatInterleaveHeads
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void TileHeads_Factor2_PairsByModularIndex()
    {
        // 2 source heads of dim 3, repeat 2× → 4 dst heads, tile pattern.
        // Pairing: dst[0] = src[0 % 2] = src[0]; dst[1] = src[1 % 2] = src[1];
        //          dst[2] = src[2 % 2] = src[0]; dst[3] = src[3 % 2] = src[1].
        // (This is NOT torch's repeat_interleave — that would yield 1,2,3, 1,2,3, 4,5,6, 4,5,6.)
        float[] src = [1f, 2f, 3f,   4f, 5f, 6f];
        float[] dst = new float[4 * 3];

        GdnKernels.TileHeads(src, dst, srcHeads: 2, repeat: 2, headDim: 3);

        Assert.Equal(new float[] { 1, 2, 3,  4, 5, 6,  1, 2, 3,  4, 5, 6 }, dst);
    }

    [Fact]
    public void TileHeads_Factor1_IsAStraightCopy()
    {
        float[] src = [1f, 2f, 3f,  4f, 5f, 6f,  7f, 8f, 9f];
        float[] dst = new float[3 * 3];
        GdnKernels.TileHeads(src, dst, srcHeads: 3, repeat: 1, headDim: 3);
        Assert.Equal(src, dst);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Gated DeltaNet recurrence
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void GdnRecurrence_DecayOnly_ZeroInputs_ProducesZero()
    {
        const int hv = 2, d = 2;
        float[] q = new float[hv * d];
        float[] k = new float[hv * d];
        float[] v = new float[hv * d];
        float[] alphaIn = new float[hv];
        float[] beta = new float[hv];                       // sigmoid(0) = 0.5
        float[] ssmA = [-1f, -0.5f];                        // non-zero but irrelevant when dt=0... actually dt=softplus(0)=log2≠0
        float[] dtBias = new float[hv];
        float[] normWeight = [1f, 1f];
        float[] z = [0.1f, 0.1f, 0.1f, 0.1f];               // gate; only the post-step sign matters
        float[] state = new float[hv * d * d];
        float[] output = new float[hv * d];

        GdnKernels.GdnRecurrenceDecode(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            state, output, numVHeads: hv, headDim: d);

        // With zero q/k/v, state stays zero and output is zero everywhere.
        foreach (var s in state) Assert.Equal(0f, s, 6);
        foreach (var o in output) Assert.Equal(0f, o, 6);
    }

    [Fact]
    public void GdnRecurrence_IdentityState_ReadoutMatchesQ()
    {
        // With S = I, k=v=zero (so the predict/delta/update steps are no-ops),
        // decay must also be 1 to preserve the identity. ssmA[h]=0 gives decay=exp(0)=1.
        // Then o[j] = Σ_i I[i,j] · q[i] = q[j].
        // Post-norm with unit gain and post-gate with SiLU(z), the output is q[j] * scale * silu(z[j]).
        const int hv = 1, d = 2;

        float[] q = [1f, 0f];
        float[] k = [0f, 0f];
        float[] v = [0f, 0f];
        float[] alphaIn = [0f];
        float[] beta = [0f];
        float[] ssmA = [0f];           // decay = 1
        float[] dtBias = [0f];
        float[] normWeight = [1f, 1f];

        // Use z huge so silu(z) ≈ z, isolating the readout numerics.
        float[] z = [50f, 50f];
        float[] state = [1f, 0f,   0f, 1f];  // identity 2×2
        float[] output = new float[hv * d];

        GdnKernels.GdnRecurrenceDecode(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            state, output, hv, d);

        // After step: o_raw = q = [1, 0].
        // Post-RMSNorm: sumSq=1, mean_sq=0.5, scale=1/sqrt(0.5 + 1e-6) ≈ √2.
        // Post-norm o = [√2, 0]. Post-silu (z=50): silu(50) ≈ 50.
        // → output ≈ [√2 * 50, 0]
        Assert.Equal(MathF.Sqrt(2f) * 50f, output[0], 3);
        Assert.Equal(0f, output[1], 4);

        // State must be untouched (k=0 means no update; decay=1).
        Assert.Equal(new[] { 1f, 0f, 0f, 1f }, state);
    }

    [Fact]
    public void GdnRecurrence_RankOneUpdate_KnownStateChange()
    {
        // Verify the rank-1 update with hand math. hv=1, d=2.
        // Start S = 0, k=[1,0], v=[2,3], q=[1,0], beta=0 (b=0.5),
        // alphaIn=0, dtBias=0 → dt=softplus(0)=ln2 ≈ 0.6931, decay=exp(dt*ssmA).
        // With ssmA=0, decay=1.
        //   p = S^T·k = 0
        //   d = 0.5 · (v - 0) = [1, 1.5]
        //   S[i,j] += k[i] · d[j]
        //     i=0: S[0,0]=1, S[0,1]=1.5
        //     i=1: zero (k[1]=0)
        //   o[j] = Σ_i S[i,j]·q[i] = S[0,j]·1 + S[1,j]·0 = S[0,:] = [1, 1.5]
        // Then norm + silu(z).
        const int hv = 1, d = 2;
        float[] q = [1f, 0f];
        float[] k = [1f, 0f];
        float[] v = [2f, 3f];
        float[] alphaIn = [0f];
        float[] beta = [0f];           // b = 0.5
        float[] ssmA = [0f];           // decay = 1
        float[] dtBias = [0f];
        float[] normWeight = [1f, 1f];
        float[] z = [100f, 100f];      // silu(z) ≈ z
        float[] state = new float[d * d];
        float[] output = new float[d];

        GdnKernels.GdnRecurrenceDecode(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            state, output, hv, d);

        // Verify state row 0 is [1, 1.5], row 1 is [0, 0].
        Assert.Equal(1.0f, state[0], 5);
        Assert.Equal(1.5f, state[1], 5);
        Assert.Equal(0.0f, state[2], 5);
        Assert.Equal(0.0f, state[3], 5);

        // o_raw = [1, 1.5]. sumSq = 1+2.25 = 3.25; mean_sq = 1.625; scale = 1/sqrt(1.625+1e-6).
        float scale = 1.0f / MathF.Sqrt(1.625f + 1e-6f);
        float expected0 = 1.0f * scale * (100f / (1f + MathF.Exp(-100f)));
        float expected1 = 1.5f * scale * (100f / (1f + MathF.Exp(-100f)));
        Assert.Equal(expected0, output[0], 3);
        Assert.Equal(expected1, output[1], 3);
    }

    [Fact]
    public void GdnRecurrence_DecodePrefillEquivalence()
    {
        const int tokens = 4;
        const int hv = 2;
        const int d = 2;
        var rng = new Random(unchecked((int)0xDEADBEEF));

        float[] q = RandomArray(rng, tokens * hv * d, -0.5f, 0.5f);
        float[] k = RandomArray(rng, tokens * hv * d, -0.5f, 0.5f);
        float[] v = RandomArray(rng, tokens * hv * d, -0.5f, 0.5f);
        float[] alphaIn = RandomArray(rng, tokens * hv, -0.5f, 0.5f);
        float[] beta = RandomArray(rng, tokens * hv, -0.5f, 0.5f);
        float[] ssmA = [-0.3f, -0.2f];
        float[] dtBias = [0.1f, -0.1f];
        float[] normWeight = RandomArray(rng, d, 0.5f, 1.5f);
        float[] z = RandomArray(rng, tokens * hv * d, -1f, 1f);

        // Path A: tokens × decode.
        float[] stateA = new float[hv * d * d];
        float[] outputA = new float[tokens * hv * d];
        int perTokQkv = hv * d;
        int perTokScalar = hv;
        for (int t = 0; t < tokens; t++)
        {
            GdnKernels.GdnRecurrenceDecode(
                q.AsSpan(t * perTokQkv, perTokQkv),
                k.AsSpan(t * perTokQkv, perTokQkv),
                v.AsSpan(t * perTokQkv, perTokQkv),
                alphaIn.AsSpan(t * perTokScalar, perTokScalar),
                beta.AsSpan(t * perTokScalar, perTokScalar),
                ssmA, dtBias, normWeight,
                z.AsSpan(t * perTokQkv, perTokQkv),
                stateA,
                outputA.AsSpan(t * perTokQkv, perTokQkv),
                hv, d);
        }

        // Path B: single prefill.
        float[] stateB = new float[hv * d * d];
        float[] outputB = new float[tokens * hv * d];
        GdnKernels.GdnRecurrencePrefill(tokens, q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
            stateB, outputB, hv, d);

        for (int i = 0; i < outputA.Length; i++)
            Assert.Equal(outputA[i], outputB[i], 5);
        for (int i = 0; i < stateA.Length; i++)
            Assert.Equal(stateA[i], stateB[i], 5);
    }

    [Fact]
    public void GdnRecurrence_ExtremeParams_StaysFinite()
    {
        // Push alphaIn far positive (so softplus saturates linearly to ~big),
        // and ssmA very negative (so decay → 0). With the linear-shortcut
        // softplus and bounded exp, nothing should overflow.
        const int hv = 2, d = 4;

        float[] q = [0.5f, -0.5f, 0.25f, -0.25f, 0.1f, 0.2f, 0.3f, 0.4f];
        float[] k = [0.3f, -0.3f, 0.15f, -0.15f, 0.2f, -0.2f, 0.1f, -0.1f];
        float[] v = [1f, 2f, 3f, 4f, -1f, -2f, -3f, -4f];

        // Extreme parameters.
        float[] alphaIn = [100f, -100f];   // softplus(100+0) ≈ 100; softplus(-100) ≈ 0
        float[] beta = [50f, -50f];        // sigmoid(±50) ≈ 1, 0
        float[] ssmA = [-100f, -1e-3f];    // decay = exp(100 * -100) → 0 for head 0
        float[] dtBias = [0f, 0f];
        float[] normWeight = [1f, 1f, 1f, 1f];
        float[] z = [50f, -50f, 50f, -50f,  50f, -50f, 50f, -50f];

        float[] state = new float[hv * d * d];
        float[] output = new float[hv * d];

        // Run several steps to amplify any numerical instability.
        for (int step = 0; step < 8; step++)
        {
            GdnKernels.GdnRecurrenceDecode(q, k, v, alphaIn, beta, ssmA, dtBias, normWeight, z,
                state, output, hv, d);

            foreach (var s in state)
                Assert.True(float.IsFinite(s), $"state contains non-finite value at step {step}");
            foreach (var o in output)
                Assert.True(float.IsFinite(o), $"output contains non-finite value at step {step}");
        }
    }

    // ────────────────────────────────────────────────────────────────────
    //  Element-wise helpers
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Softplus_LinearShortcutAndStandardBranch()
    {
        float[] src = [0f, 1f, -1f, 25f, 100f];
        float[] dst = new float[src.Length];

        GdnKernels.Softplus(dst, src);

        Assert.Equal(MathF.Log(2f),               dst[0], 6);   // softplus(0) = ln2
        Assert.Equal(MathF.Log(1f + MathF.E),     dst[1], 6);   // softplus(1)
        Assert.Equal(MathF.Log(1f + MathF.Exp(-1f)), dst[2], 6);
        Assert.Equal(25f,                          dst[3], 4);   // shortcut: src ≥ 20
        Assert.Equal(100f,                         dst[4], 3);   // shortcut
        Assert.True(float.IsFinite(dst[4]));
    }

    [Fact]
    public void Sigmoid_KnownValues()
    {
        float[] src = [0f, 10f, -10f];
        float[] dst = new float[src.Length];

        GdnKernels.Sigmoid(dst, src);

        Assert.Equal(0.5f, dst[0], 6);
        Assert.True(dst[1] > 0.999f);
        Assert.True(dst[2] < 0.001f);
    }

    [Fact]
    public void SiLu_KnownValues()
    {
        float[] src = [0f, 1f, -1f];
        float[] dst = new float[src.Length];

        GdnKernels.SiLu(dst, src);

        // silu(0) = 0; silu(1) = 1/(1+e^-1); silu(-1) = -1/(1+e)
        Assert.Equal(0f, dst[0], 6);
        Assert.Equal(1f / (1f + MathF.Exp(-1f)), dst[1], 6);
        Assert.Equal(-1f / (1f + MathF.E),       dst[2], 6);
    }

    // ────────────────────────────────────────────────────────────────────
    //  Helpers
    // ────────────────────────────────────────────────────────────────────

    private static float[] RandomArray(Random rng, int n, float lo, float hi)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return arr;
    }
}
