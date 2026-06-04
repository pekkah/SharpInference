using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #136: the all-GPU batched-trunk prefill for Gemma 4 must produce the
/// same post-prompt logits as the bit-exact per-token <see cref="CudaForwardPass.Forward"/>
/// loop. Every batched primitive it uses (Q8_0/Q5_K/… GEMM-N, RoPEWithFactorsBatched,
/// AttentionSwaBatched, AttentionBatched, HeadNormBatched, RmsNormBatched,
/// KvAppendBatched, strided GeluTanhMul) is individually proven bit-identical to its
/// per-token form, so the assembled trunk should match bit-for-bit; the test allows a
/// hair of tolerance only to stay robust to driver-level reassociation.
///
/// Single model instance, toggled via <see cref="CudaForwardPass.BatchedPrefillEnabled"/>
/// with a <c>ResetCache</c> between runs (two 8 GB instances would not co-reside).
/// Silent-skips when CUDA or the GGUF is absent.
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
    public void Gemma4_E4B_BatchedPrefill_MatchesSequential()
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

        // Batched (default on).
        fwd.BatchedPrefillEnabled = true;
        var batched = fwd.Prefill(tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk prefill did not engage — check IsGemma4BatchedPrefillSupported gating.");

        // Sequential per-token loop on the same instance.
        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);

        // Argmax must match exactly; logits must match to a hair.
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f, maxRel = 0f;
        int exact = 0;
        for (int i = 0; i < sequential.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(sequential[i]) == BitConverter.SingleToInt32Bits(batched[i]))
                exact++;
            float d = MathF.Abs(sequential[i] - batched[i]);
            maxAbs = MathF.Max(maxAbs, d);
            maxRel = MathF.Max(maxRel, d / (MathF.Abs(sequential[i]) + 1e-6f));
        }
        Assert.True(maxAbs < 1e-2f,
            $"Batched vs sequential logits diverged: maxAbs={maxAbs}, maxRel={maxRel}, " +
            $"bit-exact {exact}/{sequential.Length}.");
    }
}
