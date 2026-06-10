using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Cuda;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #207: single-user speculative decoding on the dense CUDA path.
/// <see cref="CudaForwardPass.BatchVerify"/> runs one packed k-token pass over the OWNED
/// cache at contiguous positions [P, P+k) — <see cref="CudaForwardPass.BatchForwardMulti"/>'s
/// trunk with every row bound to the same cache — so the weight HBM reads are amortized k×
/// vs the k sequential <see cref="CudaForwardPass.Forward"/> calls it replaces.
///
/// Correctness contract (chunked-prefill class): argmax-stable vs sequential Forward at every
/// verified position, asserted with the maxAbs/top-5 tolerances of
/// <see cref="CudaBatchForwardMultiTests"/>. The default WS path keeps the per-token kernels'
/// reduction chains (#194/#197), so the e2e greedy spec output is expected to EXACTLY match
/// the non-spec greedy baseline.
///
/// One ~5 GB Qwen3-8B instance per test; the sequential reference runs first and BatchVerify
/// follows after a soft <see cref="CudaForwardPass.TruncateTo"/> rewind — deliberately the
/// production flow (verify overwrites the stale rewound K/V slots). Silent-skips when CUDA
/// or the GGUF is absent — mirrors <see cref="CudaBatchForwardMultiTests"/>.
/// </summary>
public sealed class CudaSpecBatchVerifyTests
{
    private const string TargetModelFile = "Qwen3-8B-Q4_K_M.gguf";
    private const string DraftModelFile = "Qwen3-0.6B-Q8_0.gguf";

    private static readonly int[] Prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    // SnapKV pinned off: BatchVerify is unsupported under an active SnapKV budget, and
    // VRAM-scaled auto-SnapKV could otherwise engage on a smaller GPU and flip
    // SupportsBatchVerify to false (same pinning as CudaBatchForwardMultiTests.NewFwd).
    private static CudaForwardPass NewFwd(GgufModel model, CudaBackend gpu, ModelHyperparams hp, int ctx = 512)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new CudaForwardPass(model, gpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev); }
    }

    private static string? FindModelPath(string file)
    {
        string[] absolute = { $@"E:\models\{file}", $@"C:\p\sharpi\models\{file}" };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", file);
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

    private static (float maxAbs, int overlap) Compare(float[] reference, float[] candidate)
    {
        Assert.Equal(reference.Length, candidate.Length);
        float maxAbs = 0f;
        for (int i = 0; i < reference.Length; i++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i] - candidate[i]));
        var refTop = TopKSet(reference, 5);
        var candTop = TopKSet(candidate, 5);
        int overlap = 0;
        foreach (var t in candTop) if (refTop.Contains(t)) overlap++;
        return (maxAbs, overlap);
    }

    /// <summary>
    /// Headline pass-level oracle: BatchVerify's per-position logits for k packed tokens
    /// must reproduce k sequential Forward calls at every position (argmax equal +
    /// maxAbs/top-5 within the cross-path tolerance). Run at k=4 and k=6 — 6 is not a
    /// capacity-stamped WS kernel size (2/4/8/16), so it also exercises the pad-to-capacity
    /// dispatch.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    public void Qwen3_8B_BatchVerify_MatchesSequentialForward(int k)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(TargetModelFile);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify,
            "Dense Qwen3-8B Q4_K_M must report SupportsBatchVerify on the CUDA path.");

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;

        // Greedy-chain k tokens so the verified positions carry realistic activations.
        var tokens = new int[k];
        tokens[0] = Argmax(prefillLogits);

        // Sequential reference: k Forward calls, capturing logits at every position.
        var reference = new float[k][];
        for (int i = 0; i < k; i++)
        {
            var logits = fwd.Forward(tokens[i], P + i);
            reference[i] = logits.ToArray();
            if (i + 1 < k) tokens[i + 1] = Argmax(logits);
        }

        // Rewind (soft — stale K/V stays and must be overwritten) and batch-verify.
        fwd.TruncateTo(P);
        float[][] batch = fwd.BatchVerify(tokens, P);

        Assert.Equal(k, batch.Length);
        for (int i = 0; i < k; i++)
        {
            var (maxAbs, overlap) = Compare(reference[i], batch[i]);
            Assert.Equal(Argmax(reference[i]), Argmax(batch[i]));
            Assert.True(overlap >= 4,
                $"Position {i}: batched top-5 overlaps the sequential reference in only {overlap}/5 slots (maxAbs={maxAbs}).");
            Assert.True(maxAbs < 1.0f,
                $"Position {i}: batched vs sequential logits diverged beyond tolerance: maxAbs={maxAbs}.");
        }
    }

    /// <summary>
    /// Rollback oracle — the full speculative step shape: BatchVerify k tokens (some
    /// deliberately wrong), TruncateTo(P+accepted), then Forward the correction at
    /// P+accepted. The post-rollback logits must match the sequential trajectory that
    /// never saw the rejected tokens — catches stale-KV leaks past the truncation point
    /// (the rejected rows' K/V stays in the cache and must be masked by seqLen and
    /// overwritten by the commit).
    /// </summary>
    [Fact]
    public void Qwen3_8B_BatchVerify_TruncateAndCommit_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(TargetModelFile);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify);

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;
        int t0 = Argmax(prefillLogits);

        // Sequential reference trajectory: accept t0, then the correction t1.
        int t1 = Argmax(fwd.Forward(t0, P));
        float[] reference = fwd.Forward(t1, P + 1).ToArray();

        // Spec-step shape: rewind to P, verify [t0, junk, junk, junk] (junk = off-chain
        // tokens that will be rejected), accept only t0, commit t1.
        fwd.TruncateTo(P);
        int junk = (t0 + 7919) % hp.VocabSize;
        float[][] batch = fwd.BatchVerify([t0, junk, junk, junk], P);
        Assert.Equal(t1, Argmax(batch[0])); // verify logits after t0 must still pick t1

        fwd.TruncateTo(P + 1);
        float[] committed = fwd.Forward(t1, P + 1).ToArray();

        var (maxAbs, overlap) = Compare(reference, committed);
        Assert.Equal(Argmax(reference), Argmax(committed));
        Assert.True(overlap >= 4,
            $"Post-rollback commit top-5 overlap {overlap}/5 (maxAbs={maxAbs}).");
        Assert.True(maxAbs < 1.0f,
            $"Post-rollback commit diverged from the sequential trajectory: maxAbs={maxAbs}.");
    }

    /// <summary>
    /// E2E greedy parity: SpeculativeDecoder with a CUDA Qwen3-8B target and a CPU
    /// Qwen3-0.6B draft must emit EXACTLY the target's own non-spec greedy continuation —
    /// the spec-decode invariant (the draft only proposes; every emitted token is argmax of
    /// target logits). The default WS verify keeps per-token reduction chains, so unlike the
    /// pass-level tolerance oracles this asserts exact token equality over 48 tokens; a
    /// mismatch means a real verify/rollback bug (or an FP-borderline token — investigate
    /// before weakening, per the SnapKV-parity precedent).
    /// </summary>
    [Fact]
    public void Qwen3_8B_SpecDecode_GreedyParity_E2E()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var targetPath = FindModelPath(TargetModelFile);
        var draftPath = FindModelPath(DraftModelFile);
        if (targetPath is null || draftPath is null) return;

        const int DecodeTokens = 48;

        using var targetModel = GgufModel.Open(targetPath);
        var targetHp = ModelHyperparams.FromGgufMetadata(targetModel.Metadata, targetModel);
        using var target = NewFwd(targetModel, gpu, targetHp);
        Assert.True(target.SupportsBatchVerify);

        // Non-spec greedy baseline on the target alone.
        target.ResetCache();
        var logits = target.Prefill(Prompt);
        int P = Prompt.Length;
        var baseline = new List<int>();
        int tok = Argmax(logits);
        for (int i = 0; i < DecodeTokens; i++)
        {
            baseline.Add(tok);
            logits = target.Forward(tok, P + i);
            tok = Argmax(logits);
        }

        // Spec decode with the 0.6B CPU draft (same Qwen3 tokenizer/vocab).
        using var draftModel = GgufModel.Open(draftPath);
        var draftHp = ModelHyperparams.FromGgufMetadata(draftModel.Metadata, draftModel);
        Assert.Equal(targetHp.VocabSize, draftHp.VocabSize);
        using var cpu = new CpuBackend();
        using var draft = new SharpInference.Engine.ForwardPass(draftModel, cpu, draftHp);

        target.ResetCache();
        var targetLogits = target.Prefill(Prompt).ToArray();
        var draftLogits = draft.Prefill(Prompt).ToArray();

        var spec = new SpeculativeDecoder(target, draft, lookahead: 4);
        spec.Initialize(P, targetLogits, draftLogits);

        var emitted = new List<int>();
        spec.Decode(DecodeTokens, [], emitted.Add);

        Assert.Equal(baseline, emitted);
    }
}
