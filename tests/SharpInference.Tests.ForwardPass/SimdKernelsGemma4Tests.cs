using System.Runtime.Intrinsics.X86;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Parity and numerical-property tests for the Gemma-4 CPU helper kernels added
/// in Phase 2 of the Gemma 4 E4B plan: <see cref="SimdKernels.GeluTanhMul"/>,
/// <see cref="SimdKernels.ScaleInPlace"/> and <see cref="SimdKernels.SoftcapInPlace"/>.
///
/// <para><b>GeluTanhMul</b> implements the tanh approximation of GELU fused with
/// the up-projection multiply, i.e.
/// <c>out[i] = 0.5 * g * (1 + tanh(sqrt(2/π) * (g + 0.044715 * g^3))) * up[i]</c>.
/// We cross-check the AVX2 dispatcher against an internal scalar reference using
/// <see cref="MathF.Tanh"/> (no exp approximation) at a tight max-abs-diff bound,
/// plus two numerical edge cases (zero input ⇒ zero, large positive ⇒ gate·up).</para>
///
/// <para><b>SoftcapInPlace</b> clips logits via <c>x = tanh(x/cap) * cap</c>; for
/// |x| ≫ cap the output must have magnitude ≤ cap, and for |x| ≪ cap the output
/// must pass through with negligible error.</para>
///
/// AVX2-gated cases follow the existing <c>SimdKernelsQ8KSTests</c> guard:
/// <c>if (!Avx2.IsSupported || !Fma.IsSupported) return;</c>. The scalar
/// fallbacks are exercised indirectly on hosts without AVX2.
/// </summary>
public sealed unsafe class SimdKernelsGemma4Tests
{
    [Fact]
    public void GeluTanhMul_MatchesScalar()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(unchecked((int)0xBEEFCAFE));
        foreach (int n in new[] { 64, 257, 1024, 4096 })
        {
            var gate = new float[n];
            var up = new float[n];
            var avxOut = new float[n];
            var scalarOut = new float[n];

            for (int i = 0; i < n; i++)
            {
                // GELU is exercised in the range where it matters (~|x| < 5);
                // pick inputs in [-3, 3] for activations and [-2, 2] for up.
                gate[i] = (float)(rng.NextDouble() * 6.0 - 3.0);
                up[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            }

            fixed (float* g = gate)
            fixed (float* u = up)
            fixed (float* oa = avxOut)
            fixed (float* os = scalarOut)
            {
                SimdKernels.GeluTanhMul(g, u, oa, n);
                SimdKernels.GeluTanhMul_Scalar(g, u, os, n);
            }

            float maxAbs = 0f;
            int worstIdx = -1;
            for (int i = 0; i < n; i++)
            {
                float d = MathF.Abs(avxOut[i] - scalarOut[i]);
                if (d > maxAbs) { maxAbs = d; worstIdx = i; }
            }
            Console.WriteLine(
                $"GeluTanhMul avx-vs-scalar n={n}: maxAbs={maxAbs:E3} (idx={worstIdx})");
            Assert.True(maxAbs < 1e-5f,
                $"GeluTanhMul AVX2 vs scalar diff too large at n={n}: maxAbs={maxAbs:E3}");
        }
    }

    [Fact]
    public void GeluTanhMul_ZeroInput_ReturnsZero()
    {
        const int n = 64;
        var gate = new float[n];   // all zeros
        var up = new float[n];
        for (int i = 0; i < n; i++) up[i] = (i + 1) * 0.5f;
        var outp = new float[n];

        fixed (float* g = gate)
        fixed (float* u = up)
        fixed (float* o = outp)
            SimdKernels.GeluTanhMul(g, u, o, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(0f, outp[i]);
    }

    [Fact]
    public void GeluTanhMul_LargePositive_ApproachesGateTimesUp()
    {
        // For large positive g, tanh(...) → 1, so gelu_tanh(g) → g, and
        // out[i] → gate[i] * up[i].
        const int n = 64;
        var gate = new float[n];
        var up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = 20f + (i % 8);   // very large positive
            up[i] = 0.25f * (i + 1);
        }
        var outp = new float[n];

        fixed (float* g = gate)
        fixed (float* u = up)
        fixed (float* o = outp)
            SimdKernels.GeluTanhMul(g, u, o, n);

        float maxRel = 0f;
        for (int i = 0; i < n; i++)
        {
            float expected = gate[i] * up[i];
            float rel = MathF.Abs(outp[i] - expected) / MathF.Abs(expected);
            if (rel > maxRel) maxRel = rel;
        }
        Console.WriteLine($"GeluTanhMul large-positive max rel diff = {maxRel:E3}");
        Assert.True(maxRel < 1e-4f,
            $"GeluTanhMul large-positive should approach gate*up; maxRel={maxRel:E3}");
    }

    [Fact]
    public void ScaleInPlace_MultipliesEveryElement()
    {
        const int n = 257;   // odd size to exercise the AVX2 tail
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = 1.0f;

        fixed (float* p = x)
            SimdKernels.ScaleInPlace(p, 3.5f, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(3.5f, x[i]);
    }

    [Fact]
    public void SoftcapInPlace_LargeMagnitudeClamps()
    {
        const int n = 128;
        const float cap = 30f;
        var x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = (i % 2 == 0) ? 100f + i : -100f - i;

        fixed (float* p = x)
            SimdKernels.SoftcapInPlace(p, n, cap);

        for (int i = 0; i < n; i++)
        {
            Assert.True(MathF.Abs(x[i]) <= cap + 1e-3f,
                $"SoftcapInPlace did not clamp idx={i}: x={x[i]} (cap={cap})");
        }
    }

    [Fact]
    public void SoftcapInPlace_SmallMagnitudePassesThrough()
    {
        // For |x| ≪ cap, tanh(x/cap)*cap ≈ x with relative error < (x/cap)^2 / 3.
        // With cap=30 and |x|=0.1, x/cap ≈ 3.3e-3, so error ≈ 3.7e-6 — well under
        // the 1e-3 absolute tolerance asked for in the task spec.
        const int n = 128;
        const float cap = 30f;
        var x = new float[n];
        var orig = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (i % 2 == 0) ? 0.1f : -0.1f;
            orig[i] = x[i];
        }

        fixed (float* p = x)
            SimdKernels.SoftcapInPlace(p, n, cap);

        float maxAbs = 0f;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(x[i] - orig[i]);
            if (d > maxAbs) maxAbs = d;
        }
        Console.WriteLine($"SoftcapInPlace small-magnitude max abs diff = {maxAbs:E3}");
        Assert.True(maxAbs < 1e-3f,
            $"SoftcapInPlace small-magnitude must pass through; maxAbs={maxAbs:E3}");
    }
}
