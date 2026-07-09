using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Draft-level tests for <see cref="CudaDSparkDraftModel"/> on the same tiny
/// synthetic checkpoint <see cref="DSparkDraftModelTests"/> validates the CPU
/// model against (issue #428 — the launch-count optimizations need a guard
/// tighter than the E2E suites, whose verify pass masks draft numerics).
/// Silent-skips when CUDA is unavailable, mirroring CudaDSparkE2ETests.
///
/// Two kinds of assertions:
/// <list type="bullet">
/// <item>CUDA vs CPU proposals: exact token match + confidence tolerance. The
/// CUDA backbone rounds activations to fp16 per GEMM, so this holds only while
/// the synthetic head's argmax margins exceed that noise — it does today, and a
/// draft-path change that flips it deserves a look either way.</item>
/// <item>CUDA vs CUDA (incremental append / truncate-reappend vs fresh): exact
/// equality — same launches over same values must reproduce bit-identical
/// proposals, whatever the absolute numerics.</item>
/// </list>
/// </summary>
public sealed class CudaDSparkDraftModelTests
{
    private const int TapDim = 16;   // mirrors DSparkDraftModelTests.TapDim

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

    [Fact]
    public void ProposeBlock_MatchesCpuModel()
    {
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var head = new DSparkDraftModelTests.SyntheticHead(withConfidence: true);
        using var cpu = head.CreateModel();
        using var gpu = head.CreateCudaModel(cuda);

        const int ctx = 5;
        var taps = DSparkDraftModelTests.MakeTaps(ctx, seed: 12345);
        cpu.AppendContext(taps, 0, ctx);
        gpu.AppendContext(taps, 0, ctx);
        Assert.Equal(cpu.ContextLength, gpu.ContextLength);

        var pCpu = cpu.ProposeBlock(anchorToken: 7, anchorPos: ctx);
        var pGpu = gpu.ProposeBlock(anchorToken: 7, anchorPos: ctx);

        Assert.Equal(pCpu.Tokens, pGpu.Tokens);
        Assert.Equal(pCpu.Confidences.Length, pGpu.Confidences.Length);
        for (int j = 0; j < pCpu.Confidences.Length; j++)
            Assert.True(Math.Abs(pCpu.Confidences[j] - pGpu.Confidences[j]) <= 2e-2,
                $"confidence[{j}]: cpu={pCpu.Confidences[j]}, gpu={pGpu.Confidences[j]}");
    }

    [Fact]
    public void AppendContext_Incremental_EqualsBatch()
    {
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var head = new DSparkDraftModelTests.SyntheticHead(withConfidence: true);
        var taps = DSparkDraftModelTests.MakeTaps(5, seed: 777);

        using var batch = head.CreateCudaModel(cuda);
        batch.AppendContext(taps, 0, 5);

        using var incremental = head.CreateCudaModel(cuda);
        incremental.AppendContext(taps.AsSpan(0, 3 * TapDim), 0, 3);
        incremental.AppendContext(taps.AsSpan(3 * TapDim), 3, 2);
        Assert.Equal(5, incremental.ContextLength);

        var pBatch = batch.ProposeBlock(anchorToken: 4, anchorPos: 5);
        var pIncremental = incremental.ProposeBlock(anchorToken: 4, anchorPos: 5);

        Assert.Equal(pBatch.Tokens, pIncremental.Tokens);
        Assert.Equal(pBatch.Confidences, pIncremental.Confidences);
    }

    [Fact]
    public void TruncateContext_Reappend_EqualsFresh()
    {
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var head = new DSparkDraftModelTests.SyntheticHead(withConfidence: true);
        var taps = DSparkDraftModelTests.MakeTaps(5, seed: 999);

        using var fresh = head.CreateCudaModel(cuda);
        fresh.AppendContext(taps, 0, 5);
        var expected = fresh.ProposeBlock(anchorToken: 9, anchorPos: 5);

        using var truncated = head.CreateCudaModel(cuda);
        truncated.AppendContext(taps, 0, 5);
        truncated.TruncateContext(3);
        Assert.Equal(3, truncated.ContextLength);
        truncated.AppendContext(taps.AsSpan(3 * TapDim), 3, 2);
        var actual = truncated.ProposeBlock(anchorToken: 9, anchorPos: 5);

        Assert.Equal(expected.Tokens, actual.Tokens);
        Assert.Equal(expected.Confidences, actual.Confidences);
    }

    [Fact]
    public void ConsecutiveProposals_AdvanceContext()
    {
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var head = new DSparkDraftModelTests.SyntheticHead(withConfidence: true);
        using var gpu = head.CreateCudaModel(cuda);

        // Decoder-shaped round trip: propose → commit taps over the block rows
        // the proposal projected its scratch K/V into → propose again. Verifies
        // the in-place block K/V rows are correctly overwritten by AppendContext.
        var taps = DSparkDraftModelTests.MakeTaps(6, seed: 4242);
        gpu.AppendContext(taps.AsSpan(0, 4 * TapDim), 0, 4);
        var p1 = gpu.ProposeBlock(anchorToken: 2, anchorPos: 4);
        Assert.Equal(gpu.BlockSize, p1.Tokens.Length);

        gpu.AppendContext(taps.AsSpan(4 * TapDim), 4, 2);
        var p2 = gpu.ProposeBlock(anchorToken: 5, anchorPos: 6);
        Assert.Equal(gpu.BlockSize, p2.Tokens.Length);

        // Same state rebuilt fresh must reproduce the second proposal exactly.
        using var fresh = head.CreateCudaModel(cuda);
        fresh.AppendContext(taps, 0, 6);
        var pFresh = fresh.ProposeBlock(anchorToken: 5, anchorPos: 6);
        Assert.Equal(pFresh.Tokens, p2.Tokens);
        Assert.Equal(pFresh.Confidences, p2.Confidences);
    }
}
