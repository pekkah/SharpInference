using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #308: foundation for single-user speculative decoding on the dense Vulkan
/// (full-offload) path. This first PR ships <see cref="GpuForwardPass.BatchVerify"/> as a
/// correct K-loop reference — it loops the existing single-query <see cref="GpuForwardPass.Forward"/>
/// k times — so it establishes the interface, the contiguous-append semantics, and the
/// <see cref="GpuForwardPass.TruncateTo"/> rollback contract that the later weight-amortizing
/// batched-matvec PR will reuse. It does NOT yet amortize the weight reads, and no CLI spec gate
/// is flipped.
///
/// Because the K-loop IS k sequential Forward calls, BatchVerify is bit-identical to the
/// sequential reference by construction; the parity oracle below asserts this with a tight
/// tolerance (exact for greedy argmax, &lt;1e-4 on the raw logits) to lock in the KV/position
/// plumbing. The rollback oracle mirrors <see cref="CudaSpecBatchVerifyTests"/>'s
/// TruncateAndCommit shape.
///
/// Small dense model (Qwen3-0.6B-Q8_0) — on GPU, fast. Silent-skips when Vulkan is unavailable
/// or the GGUF isn't on disk.
/// </summary>
public sealed class VulkanSpecBatchVerifyTests
{
    private const string ModelFile = "Qwen3-0.6B-Q8_0.gguf";

    private static readonly int[] Prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13 };

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{ModelFile}",
            $@"E:\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
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

    // SnapKV pinned off: BatchVerify is unsupported once SnapKV evicts, and VRAM-scaled
    // auto-SnapKV could otherwise engage and flip SupportsBatchVerify to false. Pinning
    // mirrors CudaSpecBatchVerifyTests.NewFwd.
    private static GpuForwardPass NewFwd(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int ctx = 512)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new GpuForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev); }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static float MaxAbsDiff(float[] reference, float[] candidate)
    {
        Assert.Equal(reference.Length, candidate.Length);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - candidate[i]));
        return maxAbs;
    }

    /// <summary>
    /// Gate: a small dense (non-Gemma-4, non-TurboQuant) model with an uncompacted cache must
    /// report SupportsBatchVerify on the Vulkan path.
    /// </summary>
    [Fact]
    public void Qwen3_0_6B_DenseModel_ReportsSupportsBatchVerify()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim); // dense, not Gemma-4
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify,
            "Dense Qwen3-0.6B Q8_0 must report SupportsBatchVerify on the Vulkan path.");
    }

    /// <summary>
    /// Parity oracle: BatchVerify's per-position logits for k packed tokens must reproduce k
    /// sequential Forward calls at every position. Since the K-loop reference IS those calls, the
    /// match is bit-exact by construction — asserted with exact argmax equality and a &lt;1e-4
    /// tolerance on the raw logits. This verifies the BatchVerify KV/position plumbing. Run at
    /// k=4 and k=6.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    public void Qwen3_0_6B_BatchVerify_MatchesSequentialForward(int k)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify);

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;

        // Greedy-chain k tokens so the verified positions carry realistic activations.
        var tokens = new int[k];
        tokens[0] = Argmax(prefillLogits);

        // Sequential reference: k Forward calls from the prefilled cache, capturing logits at
        // every position.
        var reference = new float[k][];
        for (int i = 0; i < k; i++)
        {
            var logits = fwd.Forward(tokens[i], P + i);
            reference[i] = logits.ToArray();
            if (i + 1 < k) tokens[i + 1] = Argmax(logits);
        }

        // Rewind (soft — stale K/V stays and is overwritten by BatchVerify's appends) and verify.
        fwd.TruncateTo(P);
        float[][] batch = fwd.BatchVerify(tokens, P);

        Assert.Equal(k, batch.Length);
        for (int i = 0; i < k; i++)
        {
            Assert.Equal(Argmax(reference[i]), Argmax(batch[i]));
            float maxAbs = MaxAbsDiff(reference[i], batch[i]);
            Assert.True(maxAbs < 1e-4f,
                $"Position {i}: batched vs sequential logits diverged beyond the bit-exact " +
                $"K-loop tolerance: maxAbs={maxAbs}.");
        }

        // After BatchVerify the cache must hold exactly P + k positions (all k K/V appended).
        Assert.Equal(P + k, fwd.KvLength);
    }

    /// <summary>
    /// Rollback oracle — the full speculative-step shape: BatchVerify k tokens (some deliberately
    /// wrong), TruncateTo(P+accepted), then Forward the correction at P+accepted. The post-rollback
    /// logits must match the sequential trajectory that never saw the rejected tokens — catching
    /// stale-KV leaks past the truncation point (the rejected rows' K/V stays in VRAM and must be
    /// masked by the seqLen/_kvLength rewind and overwritten by the commit). Mirrors
    /// <see cref="CudaSpecBatchVerifyTests"/>'s TruncateAndCommit oracle.
    /// </summary>
    [Fact]
    public void Qwen3_0_6B_BatchVerify_TruncateAndCommit_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify);

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;
        int t0 = Argmax(prefillLogits);

        // Sequential reference trajectory: accept t0, then the correction t1. This path only ever
        // appends the accepted tokens.
        int t1 = Argmax(fwd.Forward(t0, P));
        float[] reference = fwd.Forward(t1, P + 1).ToArray();

        // Spec-step shape: rewind to P, verify [t0, junk, junk, junk] (junk = off-chain tokens
        // that will be rejected), accept only t0, commit t1.
        fwd.TruncateTo(P);
        int junk = (t0 + 7919) % hp.VocabSize;
        float[][] batch = fwd.BatchVerify([t0, junk, junk, junk], P);
        Assert.Equal(t1, Argmax(batch[0])); // verify logits after t0 must still pick t1

        // Roll back the rejected tail; the rejected K/V rows at [P+1, P+4) stay in VRAM but must be
        // ignored (masked by the rewound _kvLength) and overwritten by the commit.
        fwd.TruncateTo(P + 1);
        float[] committed = fwd.Forward(t1, P + 1).ToArray();

        Assert.Equal(Argmax(reference), Argmax(committed));
        float maxAbs = MaxAbsDiff(reference, committed);
        Assert.True(maxAbs < 1e-4f,
            $"Post-rollback commit diverged from the sequential trajectory: maxAbs={maxAbs}.");
    }
}
