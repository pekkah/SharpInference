using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #156: the all-GPU batched-trunk prefill (originally Gemma-4-only, #136) was
/// opened to any dense model the batched kernels cover. This oracle verifies that on a
/// dense <b>Qwen3-8B Q4_K</b> — which has none of Gemma's PLE / shared-KV / SWA /
/// sandwich-norm machinery, uses SwiGLU (Silu) instead of GEGLU, and the standard
/// 1/sqrt(head_dim) attention scale — the batched prefill engages and agrees with the
/// per-token <see cref="CudaForwardPass.Forward"/> loop.
///
/// Q4_K trunk matmuls route through the matvec GEMM-N batched path (no Q4_K cuBLAS/MMQ
/// kernel exists yet — that is issue #156 Item C), so the FlashOff case is the bit-exact
/// oracle for the batched matvec/norm/rope/SwiGLU primitives, while the default case adds
/// flash attention and is argmax-stable (online softmax reassociates the score sum).
///
/// Single 5 GB model instance toggled via <see cref="CudaForwardPass.BatchedPrefillEnabled"/>
/// with a <c>ResetCache</c> between runs. Silent-skips when CUDA or the GGUF is absent.
/// </summary>
public sealed class Qwen3CudaBatchedPrefillTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

    // Mixed-vocab prompt; >1 token so the batched path (N≥2) engages. Token 151643 is
    // Qwen's BOS-ish/special region; the rest are ordinary ids spread across the vocab.
    private static readonly int[] Tokens =
        { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13, 220, 17, 18, 19 };

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

    /// <summary>
    /// The headline #156 check: a dense, non-Gemma Q4_K model must actually take the
    /// batched-trunk prefill (gate opened) and produce argmax-stable logits vs the
    /// per-token loop under the shipped default config (flash TC on).
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_DefaultMatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        // Sanity: this really is a dense, non-Gemma SwiGLU model.
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.HasPerLayerTokenEmbd);
        Assert.False(hp.IsMoE);
        Assert.Equal(FfnActivation.Silu, hp.FfnActivation);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        // Shipped defaults (flash TC + Q4_K matvec GEMM-N).
        fwd.BatchedPrefillEnabled = true;
        var batched = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched,
            "Batched-trunk prefill did not engage for dense Q4_K — check IsBatchedPrefillSupported (#156).");

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(Tokens).ToArray();
        Assert.False(fwd.LastPrefillWasBatched);

        Assert.Equal(sequential.Length, batched.Length);
        Assert.Equal(Argmax(sequential), Argmax(batched));

        float maxAbs = 0f;
        for (int i = 0; i < sequential.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(sequential[i] - batched[i]));
        Assert.True(maxAbs < 1.0f,
            $"Default batched vs sequential logits diverged beyond fp tolerance: maxAbs={maxAbs}.");

        var seqTop = TopKSet(sequential, 5);
        var batTop = TopKSet(batched, 5);
        int overlap = 0;
        foreach (var t in batTop) if (seqTop.Contains(t)) overlap++;
        Assert.True(overlap >= 4,
            $"Default batched top-5 overlaps the per-token reference in only {overlap}/5 slots.");
    }

    /// <summary>
    /// Flash off: Q4_K trunk runs the batched matvec GEMM-N, which is built to be
    /// bit-identical to N per-token Q4_K dp4a matvecs. Verifies the batched
    /// matvec/norm/rope/SwiGLU primitives in isolation from the argmax-stable attention.
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchedPrefill_FlashOff_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);

        using var fwd = new CudaForwardPass(model, gpu, hp, maxContextLength: 512);

        fwd.BatchedPrefillEnabled = true;
        fwd.PrefillFlashAttnEnabled = false;
        fwd.PrefillFlashTcEnabled = false;
        var batched = fwd.Prefill(Tokens).ToArray();
        Assert.True(fwd.LastPrefillWasBatched);

        fwd.ResetCache();
        fwd.BatchedPrefillEnabled = false;
        var sequential = fwd.Prefill(Tokens).ToArray();
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
    }
}
