using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #141: the all-GPU batched-trunk prefill for Gemma 4 routes its Q8_0 trunk
/// matmuls through the compute-bound cuBLAS GEMM (<see cref="CudaBackend.MatMulBatchedGemm"/>),
/// which dequantizes the weight and rounds activations to fp16 before a tensor-core
/// multiply. That is <b>not</b> bit-exact to the per-token fp32 matvec loop, so this
/// oracle asserts argmax-stability plus a tolerance envelope rather than the
/// &gt;95%-bit-exact match the pre-#141 matvec GEMM-N path satisfied.
///
/// A second case (<see cref="Gemma4_E4B_BatchedPrefill_GemmOff_MatchesSequentialBitExact"/>)
/// keeps the strict bit-exact oracle for the matvec GEMM-N path (PrefillGemmEnabled=false),
/// so the #136 batched primitives stay individually verified.
///
/// Single model instance, toggled via <see cref="CudaForwardPass.BatchedPrefillEnabled"/>
/// / <see cref="CudaForwardPass.PrefillGemmEnabled"/> with a <c>ResetCache</c> between
/// runs (two 8 GB instances would not co-reside). Silent-skips when CUDA or the GGUF
/// is absent.
/// </summary>
public sealed class Gemma4CudaBatchedPrefillTests
{
    private const string ModelFile = "gemma-4-E4B-it-Q8_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absolute = { $@"E:\models\{ModelFile}", $@"C:\p\sharpi\models\{ModelFile}" };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_GemmMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        // A prompt long enough to exercise SWA windowing is not required for parity,
        // but >1 token is (batched path needs N≥2). Mixes ids across the vocab.
        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Batched cuBLAS-GEMM prefill (#141 default on).
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillGemmEnabled = true;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk prefill did not engage — check IsGemma4BatchedPrefillSupported gating.");

        // Sequential per-token loop on the same instance (the fp32 reference).
        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);

        // The decisive parity signal: argmax must match the fp32 reference. The
        // GEMM rounds weights + activations to fp16, so logits track to fp tolerance
        // (post-softcap, |logit| ≲ FinalLogitSoftcap), not bit-for-bit.
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        // fp16 trunk over 42 layers + softcap: a few tenths of a logit is expected,
        // a whole-number divergence would signal a real wiring bug.
        Assert.True(maxAbs < 1.0f,
            $"GEMM-batched vs sequential logits diverged beyond fp16 tolerance: maxAbs={maxAbs}.");

        // Top-5 set should be stable under fp16 rounding (order may shuffle below #1).
        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"GEMM-batched top-5 overlaps the fp32 reference in only {overlap}/5 slots; " +
            "fp16 GEMM is diverging more than rounding explains.");
    }

    [Fact]
    public void Gemma4_E4B_BatchedPrefill_GemmOff_MatchesSequentialBitExact()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        var tokens = new int[] { 2, 651, 6037, 576, 6081, 603, 1234, 4567, 8901, 222, 333, 444 };

        // Pin Q8_0 matvec to the fp32-decode kernel on both sides so the matvec
        // GEMM-N batched path can be compared bit-exactly to the per-token loop
        // (the default dp4a path quantizes activations and is only argmax-stable).
        gpu.Q80Dp4aEnabled = false;

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Batched matvec GEMM-N path (#136), bit-exact to per-token by construction.
        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillGemmEnabled = false;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        int exact = 0;
        for (int i = 0; i < sequential.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(sequential[i]) == BitConverter.SingleToInt32Bits(batched[i]))
                exact++;
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        }
        Assert.True(maxAbs < 1e-2f,
            $"matvec-batched vs sequential logits diverged: maxAbs={maxAbs}, bit-exact {exact}/{sequential.Length}.");

        // Every batched primitive is individually bit-identical to its per-token form.
        float exactFrac = (float)exact / sequential.Length;
        Assert.True(exactFrac > 0.95f,
            $"matvec-batched vs sequential only {exact}/{sequential.Length} ({exactFrac:P1}) bit-exact " +
            $"(expected >95%); maxAbs={maxAbs}.");
    }

    private static HashSet<int> TopKSet(ReadOnlySpan<float> logits, int k)
    {
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        var arr = logits.ToArray();
        Array.Sort(idx, (a, b) => arr[b].CompareTo(arr[a]));
        var set = new HashSet<int>();
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }
}
