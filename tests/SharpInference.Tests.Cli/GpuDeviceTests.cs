using SharpInference.Cli;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="GpuDevice.Resolve"/> — the llama.cpp-style <c>--device</c> parser.
/// Each test starts from a cleared <c>CUDA_VISIBLE_DEVICES</c> (restored on dispose) so the
/// env-var side effect is deterministic. xunit.runner.json disables collection parallelism, so
/// these env-mutating tests never run concurrently.
/// </summary>
public sealed class GpuDeviceTests : IDisposable
{
    private const string CvdVar = "CUDA_VISIBLE_DEVICES";
    private readonly string? _savedCvd;

    public GpuDeviceTests()
    {
        _savedCvd = Environment.GetEnvironmentVariable(CvdVar);
        Environment.SetEnvironmentVariable(CvdVar, null);
    }

    public void Dispose() => Environment.SetEnvironmentVariable(CvdVar, _savedCvd);

    [Theory]
    [InlineData("0", 0)]
    [InlineData("1", 1)]
    [InlineData("CUDA0", 0)]
    [InlineData("Vulkan1", 1)]
    [InlineData("GPU2", 2)]
    [InlineData("cuda3", 3)]   // case-insensitive backend prefix
    public void Resolve_ConcreteIndex_ReturnsIndex(string input, int expected)
    {
        int index = GpuDevice.Resolve(input, out bool none);
        Assert.Equal(expected, index);
        Assert.False(none);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("auto")]
    [InlineData("AUTO")]
    public void Resolve_Auto_ReturnsMinusOne(string? input)
    {
        int index = GpuDevice.Resolve(input, out bool none);
        Assert.Equal(-1, index);
        Assert.False(none);
    }

    [Theory]
    [InlineData("none")]
    [InlineData("None")]
    [InlineData("cpu")]
    [InlineData("CPU")]
    public void Resolve_None_SetsNoneFlag(string input)
    {
        int index = GpuDevice.Resolve(input, out bool none);
        Assert.Equal(-1, index);
        Assert.True(none);
    }

    [Theory]
    [InlineData("-1")]              // must NOT mis-parse as device 1
    [InlineData("+1")]
    [InlineData("0,1")]            // multi-device split unsupported
    [InlineData("foo")]
    [InlineData("CUDA")]          // named device with no index
    [InlineData("0x1F")]
    [InlineData("99999999999999")] // int overflow
    public void Resolve_Invalid_Throws(string input)
    {
        Assert.Throws<InvalidOperationException>(() => GpuDevice.Resolve(input, out _));
    }

    [Fact]
    public void Resolve_ConcreteIndex_PinsCudaVisibleDevicesWhenUnset()
    {
        int index = GpuDevice.Resolve("CUDA2", out _);
        Assert.Equal(2, index);
        Assert.Equal("2", Environment.GetEnvironmentVariable(CvdVar));
    }

    [Fact]
    public void Resolve_DoesNotOverrideUserSetCudaVisibleDevices()
    {
        Environment.SetEnvironmentVariable(CvdVar, "7");
        int index = GpuDevice.Resolve("0", out _);
        Assert.Equal(0, index);
        Assert.Equal("7", Environment.GetEnvironmentVariable(CvdVar)); // preserved, not clobbered
    }

    [Fact]
    public void Resolve_Auto_DoesNotTouchCudaVisibleDevices()
    {
        GpuDevice.Resolve("auto", out _);
        Assert.Null(Environment.GetEnvironmentVariable(CvdVar));
    }
}
