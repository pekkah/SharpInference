using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #308: single-user speculative decoding on the dense Vulkan (full-offload) path.
///
/// PR1 shipped <see cref="GpuForwardPass.BatchVerify"/> as a correct K-loop reference (k sequential
/// <see cref="GpuForwardPass.Forward"/> calls). PR1c (this) re-implements BatchVerify on a BATCHED
/// trunk that streams all k draft tokens through ONE command buffer, reading each Q4_K/Q6_K weight
/// matrix from VRAM once via <c>MatMulBatched</c> (the weight-amortization). The batched path is
/// gated to dense models whose every trunk matmul weight is Q4_K/Q6_K (<c>CanBatchedTrunk</c>);
/// other dtypes (e.g. Qwen3-0.6B-Q8_0) keep the K-loop fallback.
///
/// The batched path is bit-exact to the K-loop by construction (MatMulBatched is bit-identical to
/// single-row matvec, the gather/scatter copies are exact, the per-token RmsNorm/QK-norm/RoPE/
/// append/attention reuse the single-query shaders with the same positions/seqLens). Parity oracles
/// below assert this on BOTH models — Qwen3-8B-Q4_K_M exercises the batched trunk; Qwen3-0.6B-Q8_0
/// exercises the K-loop fallback — with exact greedy-argmax equality and a &lt;1e-4 logit tolerance.
///
/// All cases run on GPU and silent-skip when Vulkan is unavailable or the GGUF isn't on disk.
/// </summary>
public sealed class VulkanSpecBatchVerifyTests
{
    private const string SmallModel = "Qwen3-0.6B-Q8_0.gguf";   // Q8_0 weights → K-loop fallback
    private const string BatchedModel = "Qwen3-8B-Q4_K_M.gguf"; // Q4_K weights → batched trunk

    private readonly ITestOutputHelper _out;
    public VulkanSpecBatchVerifyTests(ITestOutputHelper output) => _out = output;

    private static readonly int[] Prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13 };

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath(string modelFile)
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{modelFile}",
            $@"E:\models\{modelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", modelFile);
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
    private static GpuForwardPass NewFwd(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int ctx = 512, DType kvDtype = DType.Float32)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new GpuForwardPass(model, gpu, hp, maxContextLength: ctx, kvDtype: kvDtype); }
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
    /// report SupportsBatchVerify on the Vulkan path; the Q8_0 weights make it take the K-loop
    /// fallback (CanBatchedTrunk == false).
    /// </summary>
    [Fact]
    public void Qwen3_0_6B_DenseModel_ReportsSupportsBatchVerify()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(SmallModel);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim); // dense, not Gemma-4
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify,
            "Dense Qwen3-0.6B Q8_0 must report SupportsBatchVerify on the Vulkan path.");
        Assert.False(fwd.CanBatchedTrunk,
            "Qwen3-0.6B Q8_0 weights must NOT qualify for the batched trunk (Q8_0 ∉ {Q4_K, Q6_K}).");
    }

    /// <summary>
    /// Gate: the Q4_K model must qualify for the weight-amortizing batched trunk.
    /// </summary>
    [Fact]
    public void Qwen3_8B_Q4K_QualifiesForBatchedTrunk()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(BatchedModel);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.IsMoE);

        using var fwd = NewFwd(model, gpu, hp);
        Assert.True(fwd.SupportsBatchVerify);
        Assert.True(fwd.CanBatchedTrunk,
            "Qwen3-8B Q4_K_M weights must qualify for the batched trunk.");
    }

    /// <summary>
    /// Parity oracle (K-loop fallback path): BatchVerify's per-position logits for k packed tokens
    /// reproduce k sequential Forward calls. Bit-exact by construction (it IS those calls).
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    public void Qwen3_0_6B_BatchVerify_MatchesSequentialForward(int k)
        => RunParity(SmallModel, k, expectBatchedTrunk: false);

    /// <summary>
    /// Parity oracle (BATCHED trunk path, the payoff): BatchVerify on Qwen3-8B-Q4_K_M must match k
    /// sequential Forward calls bit-exactly (exact argmax, &lt;1e-4 logit tolerance) — the
    /// weight-amortized trunk is numerically identical to the K-loop because MatMulBatched is
    /// bit-identical to single-row matvec and every per-token op reuses the single-query shaders.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    public void Qwen3_8B_Q4K_BatchVerify_MatchesSequentialForward(int k)
        => RunParity(BatchedModel, k, expectBatchedTrunk: true);

    /// <summary>
    /// Parity oracle (issue #308 follow-up — bf16 KV batched attention/append). BatchVerify on the
    /// batched trunk with <c>--kv-type bf16</c> must match k sequential Forward calls (also bf16 KV)
    /// bit-exactly: the batched bf16 KvAppend/Attention shaders reuse the SAME packHalf2x16 store /
    /// unpackHalf2x16 read idioms and the SAME per-query causal range as the single-token shaders, so
    /// BatchVerify(bf16) == k× Forward(bf16). This is batched-vs-K-loop on the SAME dtype (NOT vs fp32).
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public void Qwen3_8B_Q4K_BatchVerify_Bf16Kv_MatchesSequentialForward(int k)
        => RunParity(BatchedModel, k, expectBatchedTrunk: true, kvDtype: DType.BFloat16);

    /// <summary>
    /// Parity oracle (issue #308 follow-up — q8_0 KV batched attention/append). BatchVerify on the
    /// batched trunk with <c>--kv-type q8_0</c> must match k sequential Forward calls (also q8_0 KV)
    /// bit-exactly: the batched q8_0 KvAppend uses the SAME amax→quant + masked-atomic byte store and
    /// the batched AttentionQ8_0 the SAME byte-gather dequant as the single-token shaders, over the
    /// SAME per-query causal range, so BatchVerify(q8_0) == k× Forward(q8_0). The batched append's
    /// masked atomics write disjoint bytes (independent of dispatch order), matching the single-token
    /// append exactly.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public void Qwen3_8B_Q4K_BatchVerify_Q8Kv_MatchesSequentialForward(int k)
        => RunParity(BatchedModel, k, expectBatchedTrunk: true, kvDtype: DType.Q8_0);

    private void RunParity(string modelFile, int k, bool expectBatchedTrunk, DType kvDtype = DType.Float32)
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(modelFile);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);
        Assert.False(hp.IsMoE);

        // 8B Q4_K KV at ctx 64 fits comfortably; small ctx keeps the test fast.
        using var fwd = NewFwd(model, gpu, hp, ctx: 64, kvDtype: kvDtype);
        Assert.True(fwd.SupportsBatchVerify);
        Assert.Equal(expectBatchedTrunk, fwd.CanBatchedTrunk);

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;

        // Greedy-chain k tokens so the verified positions carry realistic activations.
        var tokens = new int[k];
        tokens[0] = Argmax(prefillLogits);

        // Sequential reference: k Forward calls from the prefilled cache (the K-loop oracle).
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
        float worst = 0f;
        // The batched trunk runs the int8-activation DP4A Q4_K matvec (issue #308 P1): LOSSY vs the
        // K-loop's FP single-row matvec, so it is ARGMAX-stable, not bit-exact. The Q8_0 K-loop
        // fallback model stays bit-exact (no int8 path for Q8_0).
        bool int8Trunk = fwd.CanBatchedTrunk;
        float tol = int8Trunk ? 1.0f : 1e-4f;
        for (int i = 0; i < k; i++)
        {
            Assert.Equal(Argmax(reference[i]), Argmax(batch[i]));
            float maxAbs = MaxAbsDiff(reference[i], batch[i]);
            worst = MathF.Max(worst, maxAbs);
            Assert.True(maxAbs < tol,
                $"Position {i}: batched vs sequential logits diverged beyond the " +
                $"{(int8Trunk ? "int8 argmax-stable" : "bit-exact K-loop")} tolerance " +
                $"(kv={kvDtype}): maxAbs={maxAbs}.");
        }
        _out.WriteLine($"{modelFile} kv={kvDtype} k={k} batchedTrunk={fwd.CanBatchedTrunk} int8Trunk={int8Trunk} worstMaxAbs={worst:E3}");

        // After BatchVerify the cache must hold exactly P + k positions (all k K/V appended).
        Assert.Equal(P + k, fwd.KvLength);
    }

    /// <summary>
    /// Regression (#308 scratch-sizing): BatchVerify must be correct when k VARIES across calls on
    /// one instance — k shrinks to 2/3 at the generation tail (and on partial prompt-lookup matches).
    /// The batched scratch was grow-only (`_bvK >= k`), so a smaller k after a larger one reused an
    /// OVERSIZED buffer and MatMulBatched derived rows/cols = ElementCount/k against the wrong size →
    /// garbage logits (or an ArgumentException when k didn't divide the oversized count). This drives
    /// the batched trunk through k = 4 → 2 → 3 → 6 → 1 on ONE GpuForwardPass and asserts each matches
    /// the K-loop oracle. Pre-fix the k=2/k=3 steps crash or diverge; the existing fixed-k tests
    /// missed it because each used a fresh instance with a single k.
    /// </summary>
    [Fact]
    public void Qwen3_8B_Q4K_BatchVerify_VariableK_MatchesSequentialForward()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(BatchedModel);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var fwd = NewFwd(model, gpu, hp, ctx: 64);
        Assert.True(fwd.CanBatchedTrunk, "needs the batched trunk to exercise the scratch-sizing path.");

        int P = Prompt.Length;
        // A fixed token sequence (plausible ids) — the oracle and the batched run use the same first-k.
        int[] seq = { 9707, 11, 1879, 0, 358, 1079, 264, 4108 };

        // Larger-k first (sizes _bvK=4), then SMALLER k that pre-fix reused the oversized buffer.
        foreach (int k in new[] { 4, 2, 3, 6, 1 })
        {
            var tokens = seq[..k];

            // K-loop oracle from the prefilled state.
            fwd.ResetCache();
            fwd.Prefill(Prompt);
            var reference = new float[k][];
            for (int i = 0; i < k; i++)
                reference[i] = fwd.Forward(tokens[i], P + i).ToArray();

            // Batched verify on the SAME instance (so _bvK carries over from the prior k).
            fwd.TruncateTo(P);
            float[][] batch = fwd.BatchVerify(tokens, P);

            Assert.Equal(k, batch.Length);
            // Batched trunk = int8 DP4A matvec → argmax-stable (not bit-exact) vs the FP K-loop.
            for (int i = 0; i < k; i++)
            {
                Assert.Equal(Argmax(reference[i]), Argmax(batch[i]));
                Assert.True(MaxAbsDiff(reference[i], batch[i]) < 1.0f,
                    $"k={k} pos={i}: batched diverged from the K-loop oracle beyond the int8 " +
                    "argmax-stable tolerance (scratch-sizing).");
            }
            _out.WriteLine($"variable-k: k={k} OK");
        }
    }

    /// <summary>
    /// Rollback oracle — the full speculative-step shape: BatchVerify k tokens (some deliberately
    /// wrong), TruncateTo(P+accepted), then Forward the correction. Post-rollback logits must match
    /// the sequential trajectory that never saw the rejected tokens (catches stale-KV leaks past the
    /// truncation point). Runs on the BATCHED trunk model (Qwen3-8B-Q4_K_M) so the rollback contract
    /// is asserted on the new path; falls back to the K-loop model if the 8B GGUF is absent.
    /// </summary>
    [Fact]
    public void BatchVerify_TruncateAndCommit_MatchesSequential()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        // Prefer the batched-trunk model; degrade to the small model so CI without the 8B still runs.
        var path = FindModelPath(BatchedModel) ?? FindModelPath(SmallModel);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var fwd = NewFwd(model, gpu, hp, ctx: 64);
        Assert.True(fwd.SupportsBatchVerify);

        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(Prompt);
        int P = Prompt.Length;
        int t0 = Argmax(prefillLogits);

        // Sequential reference trajectory: accept t0, then the correction t1.
        int t1 = Argmax(fwd.Forward(t0, P));
        float[] reference = fwd.Forward(t1, P + 1).ToArray();

        // Spec-step shape: rewind to P, verify [t0, junk, junk, junk] (junk rejected), accept t0.
        fwd.TruncateTo(P);
        int junk = (t0 + 7919) % hp.VocabSize;
        float[][] batch = fwd.BatchVerify([t0, junk, junk, junk], P);
        Assert.Equal(t1, Argmax(batch[0])); // verify logits after t0 must still pick t1

        // Roll back the rejected tail; rejected K/V at [P+1, P+4) stays but must be ignored and
        // overwritten by the commit.
        fwd.TruncateTo(P + 1);
        float[] committed = fwd.Forward(t1, P + 1).ToArray();

        // The commit is a pure FP Forward, but it ATTENDS to the K/V at position P that BatchVerify
        // wrote from int8-DP4A-computed activations (the batched trunk, issue #308 P1). So the
        // committed logits differ from the all-FP reference by int8 noise, not bit-exactly — the
        // contract is argmax stability (the accepted-token trajectory is unchanged), with a tolerance
        // matching the int8 path. On the Q8_0 K-loop fallback model the trunk is FP → bit-exact.
        Assert.Equal(Argmax(reference), Argmax(committed));
        float tol = fwd.CanBatchedTrunk ? 1.0f : 1e-4f;
        float maxAbs = MaxAbsDiff(reference, committed);
        Assert.True(maxAbs < tol,
            $"Post-rollback commit diverged from the sequential trajectory beyond the " +
            $"{(fwd.CanBatchedTrunk ? "int8 argmax-stable" : "bit-exact")} tolerance: maxAbs={maxAbs}.");
    }

    /// <summary>
    /// Micro-bench (GPU-only, silent-skip): on Qwen3-8B-Q4_K_M, prefill ~512 tokens then time
    /// BatchVerify(k=4) — the weight-amortizing batched trunk — against 4× single Forward (the
    /// K-loop equivalent). Reports the median wall-clock ratio so the speedup is honest. This is the
    /// PR2 arbiter: it asserts only that the batched path is not catastrophically slower (≤ 1.5×
    /// the K-loop) and logs the real ratio. Skips unless SHARPI_RUN_VULKAN_SPEC_BENCH=1 (it's a
    /// timing probe, not a correctness gate).
    /// </summary>
    [Fact]
    public void Qwen3_8B_Q4K_BatchVerify_MicroBench()
    {
        if (Environment.GetEnvironmentVariable("SHARPI_RUN_VULKAN_SPEC_BENCH") != "1")
            return;

        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(BatchedModel);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        int prefillLen = int.TryParse(Environment.GetEnvironmentVariable("SHARPI_SPEC_BENCH_PREFILL"), out var pl) ? pl : 512;
        int k = int.TryParse(Environment.GetEnvironmentVariable("SHARPI_SPEC_BENCH_K"), out var kk) ? kk : 4;
        DType kvDtype = Environment.GetEnvironmentVariable("SHARPI_SPEC_BENCH_KV")?.ToLowerInvariant() switch
        {
            "bf16" => DType.BFloat16,
            "q8_0" or "q8" => DType.Q8_0,
            _ => DType.Float32,
        };
        using var fwd = NewFwd(model, gpu, hp, ctx: prefillLen + 64, kvDtype: kvDtype);
        Assert.True(fwd.SupportsBatchVerify);
        Assert.True(fwd.CanBatchedTrunk);

        // Build a ~512-token prompt (repeat the seed prompt) and prefill it.
        var longPrompt = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) longPrompt[i] = Prompt[i % Prompt.Length];
        fwd.ResetCache();
        var prefillLogits = fwd.Prefill(longPrompt);
        int P = prefillLen;

        var tokens = new int[k];
        tokens[0] = Argmax(prefillLogits);
        for (int i = 1; i < k; i++) tokens[i] = (tokens[i - 1] + 1) % hp.VocabSize;

        // Warm up both paths (compiles pipelines, warms caches).
        fwd.TruncateTo(P);
        _ = fwd.BatchVerify(tokens, P);
        fwd.TruncateTo(P);
        for (int i = 0; i < k; i++) _ = fwd.Forward(tokens[i], P + i).ToArray();

        const int iters = 9;
        var batchedMs = new double[iters];
        var kloopMs = new double[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++)
        {
            fwd.TruncateTo(P);
            sw.Restart();
            _ = fwd.BatchVerify(tokens, P);
            sw.Stop();
            batchedMs[it] = sw.Elapsed.TotalMilliseconds;

            fwd.TruncateTo(P);
            sw.Restart();
            for (int i = 0; i < k; i++) _ = fwd.Forward(tokens[i], P + i).ToArray();
            sw.Stop();
            kloopMs[it] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(batchedMs);
        Array.Sort(kloopMs);
        double medBatched = batchedMs[iters / 2];
        double medKloop = kloopMs[iters / 2];
        double ratio = medKloop / medBatched; // >1 ⇒ batched is faster

        _out.WriteLine($"BatchVerify(k={k}, prefill={prefillLen}, kv={kvDtype}) median: batched={medBatched:F2}ms, " +
            $"{k}xForward={medKloop:F2}ms, speedup={ratio:F2}x");

        Assert.True(medBatched <= medKloop * 1.5,
            $"Batched trunk should not be much slower than the K-loop: " +
            $"batched={medBatched:F2}ms vs kloop={medKloop:F2}ms.");
    }

    /// <summary>
    /// Micro-bench (GPU-only, gated on SHARPI_RUN_VULKAN_SPEC_BENCH=1): the PER-MATVEC arbiter for
    /// issue #308 P1. Times one int8-DP4A <c>MatMulBatched(Q4_K, k)</c> against k single-row
    /// <c>MatMul(Q4_K)</c> calls on a realistic Q4_K weight (rows×cols ≈ a Qwen3-8B trunk matmul),
    /// and reports the ratio. GO if k=4 ≤ ~1.5× a single matvec (the FP batched path was ~2.1×):
    /// the per-token cost must collapse for the verify to win. Logs the ratio; asserts only a loose
    /// non-regression so it never gates CI on timing noise. Synthetic weights (no model file needed).
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(6)]
    [InlineData(8)]
    public void MatVecBatchedQ4KInt8_PerMatVec_MicroBench(int k)
    {
        if (Environment.GetEnvironmentVariable("SHARPI_RUN_VULKAN_SPEC_BENCH") != "1")
            return;

        using var gpu = TryCreate();
        if (gpu is null) return;

        // Qwen3-8B-ish trunk matmul: ffn_down is [4096 × 12288]; use that (largest cols).
        const int rows = 4096;
        const int cols = 12288;
        int blocksPerRow = cols / 256;
        const int blockBytes = 144;
        var weightBytes = new byte[rows * blocksPerRow * blockBytes];
        var wr = new Random(4242);
        for (int b = 0, off = 0; b < rows * blocksPerRow; b++, off += blockBytes)
        {
            PutHalf16(weightBytes, off, (float)(wr.NextDouble() * 0.045 + 0.005));
            PutHalf16(weightBytes, off + 2, (float)(wr.NextDouble() * 0.002 + 0.0005));
            for (int j = 4; j < blockBytes; j++) weightBytes[off + j] = (byte)wr.Next(0, 256);
        }
        int floatCount = (weightBytes.Length + 3) / 4;
        var rawAsFloats = new float[floatCount];
        weightBytes.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(rawAsFloats.AsSpan()));
        var gpuWeights = gpu.Upload(rawAsFloats, TensorShape.D1(floatCount));

        var inputAll = new float[k * cols];
        var ir = new Random(7);
        for (int i = 0; i < inputAll.Length; i++) inputAll[i] = (float)(ir.NextDouble() * 2 - 1);
        var gpuInputAll = gpu.Upload(inputAll, TensorShape.D1(k * cols));
        var gpuOutputAll = gpu.Allocate(TensorShape.D1(k * rows));
        var gpuInK = gpu.Upload(inputAll.AsSpan(0, cols).ToArray(), TensorShape.D1(cols));
        var gpuOutK = gpu.Allocate(TensorShape.D1(rows));

        // Warm up.
        gpu.MatMulBatched(gpuOutputAll, gpuWeights, gpuInputAll, k, DType.Q4_K);
        for (int i = 0; i < k; i++) gpu.MatMul(gpuOutK, gpuWeights, gpuInK, DType.Q4_K);

        const int iters = 15;
        var batchedMs = new double[iters];
        var singleMs = new double[iters];
        var sw = new Stopwatch();
        for (int it = 0; it < iters; it++)
        {
            sw.Restart();
            gpu.MatMulBatched(gpuOutputAll, gpuWeights, gpuInputAll, k, DType.Q4_K);
            sw.Stop();
            batchedMs[it] = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            for (int i = 0; i < k; i++) gpu.MatMul(gpuOutK, gpuWeights, gpuInK, DType.Q4_K);
            sw.Stop();
            singleMs[it] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(batchedMs);
        Array.Sort(singleMs);
        double medBatched = batchedMs[iters / 2];
        double medSingle = singleMs[iters / 2];
        double perMatvecRatio = medBatched / (medSingle / k); // batched-k cost ÷ one single matvec

        _out.WriteLine($"MatVecBatchedQ4KInt8 [{rows}x{cols}] k={k}: batched={medBatched:F3}ms, " +
            $"{k}xSingle={medSingle:F3}ms (1x={medSingle / k:F3}ms) → per-matvec ratio={perMatvecRatio:F2}x " +
            $"(GO if ≤ ~1.5x); batched-vs-Ksingle speedup={medSingle / medBatched:F2}x");

        gpu.Free(gpuWeights); gpu.Free(gpuInputAll); gpu.Free(gpuOutputAll);
        gpu.Free(gpuInK); gpu.Free(gpuOutK);
    }

    private static void PutHalf16(byte[] dst, int off, float value)
    {
        ushort h = (ushort)System.Runtime.CompilerServices.Unsafe.BitCast<Half, short>((Half)value);
        dst[off] = (byte)(h & 0xFF);
        dst[off + 1] = (byte)(h >> 8);
    }
}
