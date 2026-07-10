using SharpInference.Cli;
using SharpInference.Engine;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="PerplexityCommand.TryValidateFlags"/> (the perplexity
/// harness's --tq/--tq-mode/--tq-window/-g validation, issue #180 P0) and
/// <see cref="PerplexityCommand.NegativeLogLikelihood"/> (the log-softmax NLL math).
/// Same model-free pattern as <see cref="CpuMoeFlagsTests"/>: internal static helpers
/// exercised directly.
/// </summary>
public sealed class PerplexityFlagsTests
{
    [Theory]
    [InlineData(false, "lloydmax", 256, 0, TqQuantizer.LloydMax)]  // fp32 baseline (tq off, mode ignored)
    [InlineData(true, "lloydmax", 256, 0, TqQuantizer.LloydMax)]
    [InlineData(true, "lloyd-max", 256, 0, TqQuantizer.LloydMax)]
    [InlineData(true, "", 256, 0, TqQuantizer.LloydMax)]
    [InlineData(true, "kvarn", 256, 0, TqQuantizer.KVarN)]
    [InlineData(true, "KVarN", 128, 0, TqQuantizer.KVarN)]         // case-insensitive; min window
    [InlineData(true, "lloydmax", 32, 0, TqQuantizer.LloydMax)]    // Lloyd-Max min window (one FastScan tile)
    [InlineData(false, "lloydmax", 256, -1, TqQuantizer.LloydMax)] // full CUDA offload (Task 5a): fp32 baseline
    [InlineData(true, "kvarn", 256, -1, TqQuantizer.KVarN)]        // full CUDA offload (Task 5a): kvarn gate
    public void ValidCombos_Resolve(bool tq, string mode, int window, int ngl, TqQuantizer expected)
    {
        bool ok = PerplexityCommand.TryValidateFlags(tq, mode, window, ngl, out var quantizer, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal(expected, quantizer);
    }

    [Fact]
    public void Kvarn_WithoutTq_IsRejected()
    {
        bool ok = PerplexityCommand.TryValidateFlags(tq: false, "kvarn", 256, 0, out _, out string? error);

        Assert.False(ok);
        Assert.Contains("requires --tq", error);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(36)]
    [InlineData(-2)]
    public void GpuLayers_Partial_IsRejected(int ngl)
    {
        // 0 (CPU) and -1 (full CUDA offload, issue #180 Task 5a) are the only
        // supported placements; anything partial is rejected.
        bool ok = PerplexityCommand.TryValidateFlags(tq: true, "kvarn", 256, ngl, out _, out string? error);

        Assert.False(ok);
        Assert.Contains("-g 0", error);
    }

    [Fact]
    public void UnknownMode_IsRejected()
    {
        bool ok = PerplexityCommand.TryValidateFlags(tq: true, "fastscan", 256, 0, out _, out string? error);

        Assert.False(ok);
        Assert.Contains("fastscan", error);
    }

    [Theory]
    [InlineData("kvarn", 127)]   // below one KVarN tile
    [InlineData("lloydmax", 31)] // below one FastScan tile
    public void Window_BelowTileFloor_IsRejected(string mode, int window)
    {
        bool ok = PerplexityCommand.TryValidateFlags(tq: true, mode, window, 0, out _, out string? error);

        Assert.False(ok);
        Assert.Contains("--tq-window", error);
    }

    [Fact]
    public void Nll_UniformLogits_IsLogVocab()
    {
        float[] logits = new float[64];   // all zeros → uniform softmax → NLL = ln(64)
        double nll = PerplexityCommand.NegativeLogLikelihood(logits, target: 17);

        Assert.Equal(Math.Log(64), nll, precision: 10);
    }

    [Fact]
    public void Nll_IsShiftInvariant_AndOrdersCorrectly()
    {
        float[] logits = [1.0f, 3.0f, 0.5f, -2.0f];
        float[] shifted = [101.0f, 103.0f, 100.5f, 98.0f];

        double a = PerplexityCommand.NegativeLogLikelihood(logits, 1);
        double b = PerplexityCommand.NegativeLogLikelihood(shifted, 1);
        Assert.Equal(a, b, precision: 6);

        // The argmax token must be the cheapest (smallest NLL).
        double argmaxNll = PerplexityCommand.NegativeLogLikelihood(logits, 1);
        double otherNll = PerplexityCommand.NegativeLogLikelihood(logits, 3);
        Assert.True(argmaxNll < otherNll);
    }

    [Fact]
    public void Nll_NaNLogits_ReturnsNonFinite()
    {
        float[] logits = [0f, float.NaN, 1f];
        double nll = PerplexityCommand.NegativeLogLikelihood(logits, 0);

        Assert.False(double.IsFinite(nll));
    }

    [Fact]
    public void Nll_TargetOutOfRange_ReturnsNaN()
    {
        double nll = PerplexityCommand.NegativeLogLikelihood([0f, 0f], target: 2);

        Assert.True(double.IsNaN(nll));
    }
}
