using SharpInference.Cli;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="RunCommand.TryApplyCpuMoeFlags"/> — the llama.cpp-style MoE
/// placement flags (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>, issue #80). Each test starts from a
/// cleared <c>SHARPI_CPU_MOE</c> (restored on dispose) so the env-var side effect is
/// deterministic. xunit.runner.json disables collection parallelism, so these env-mutating
/// tests never run concurrently with each other.
/// </summary>
public sealed class CpuMoeFlagsTests : IDisposable
{
    private const string Var = "SHARPI_CPU_MOE";
    private readonly string? _saved;

    public CpuMoeFlagsTests()
    {
        _saved = Environment.GetEnvironmentVariable(Var);
        Environment.SetEnvironmentVariable(Var, null);
    }

    public void Dispose() => Environment.SetEnvironmentVariable(Var, _saved);

    [Fact]
    public void CpuMoe_True_SetsEnvToOne()
    {
        bool ok = RunCommand.TryApplyCpuMoeFlags(cpuMoe: true, nCpuMoe: null, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal("1", Environment.GetEnvironmentVariable(Var));
    }

    [Fact]
    public void CpuMoe_True_OverridesInheritedEnv()
    {
        Environment.SetEnvironmentVariable(Var, "0"); // operator had forced GPU experts
        bool ok = RunCommand.TryApplyCpuMoeFlags(cpuMoe: true, nCpuMoe: null, out _);

        Assert.True(ok);
        Assert.Equal("1", Environment.GetEnvironmentVariable(Var)); // explicit flag wins
    }

    [Fact]
    public void Defaults_LeaveEnvUntouched()
    {
        Environment.SetEnvironmentVariable(Var, "preexisting"); // e.g. SHARPI_CPU_MOE in the shell
        bool ok = RunCommand.TryApplyCpuMoeFlags(cpuMoe: false, nCpuMoe: null, out string? error);

        Assert.True(ok);
        Assert.Null(error);
        Assert.Equal("preexisting", Environment.GetEnvironmentVariable(Var)); // no write → auto-select preserved
    }

    [Theory]
    [InlineData(20)]   // partial split
    [InlineData(0)]    // even the no-op count is rejected — the feature isn't implemented
    [InlineData(-1)]
    public void NCpuMoe_Provided_IsDeferredWithError(int n)
    {
        Environment.SetEnvironmentVariable(Var, "preexisting");
        bool ok = RunCommand.TryApplyCpuMoeFlags(cpuMoe: false, nCpuMoe: n, out string? error);

        Assert.False(ok);
        Assert.NotNull(error);
        Assert.Contains("--cpu-moe", error);   // points at the supported all-or-nothing flag
        Assert.Contains("#80", error);          // rationale references the tracking issue
        Assert.Equal("preexisting", Environment.GetEnvironmentVariable(Var)); // rejected → no env write
    }

    [Fact]
    public void NCpuMoe_TakesPrecedenceOverCpuMoe()
    {
        // Both flags passed: the unsupported one must be reported rather than silently honoring --cpu-moe.
        bool ok = RunCommand.TryApplyCpuMoeFlags(cpuMoe: true, nCpuMoe: 4, out string? error);

        Assert.False(ok);
        Assert.NotNull(error);
        Assert.Null(Environment.GetEnvironmentVariable(Var)); // nothing written
    }
}
