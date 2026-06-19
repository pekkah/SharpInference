using SharpInference.Cli;

namespace SharpInference.Tests.Cli;

/// <summary>
/// Unit tests for <see cref="RunCommand.ResolveThinkingOff"/> — the --thinking / --no-thinking
/// precedence. Gemma 4 defaults reasoning off (its stock instruct models aren't reasoning-trained);
/// every other arch defaults on. --no-thinking always wins; --thinking forces it on.
/// </summary>
public sealed class ThinkingResolutionTests
{
    [Theory]
    // Gemma 4: defaults OFF, --thinking opts in, --no-thinking wins over a conflicting --thinking.
    [InlineData("gemma4", false, false, true)]
    [InlineData("gemma4", true,  false, false)]
    [InlineData("gemma4", false, true,  true)]
    [InlineData("gemma4", true,  true,  true)]
    // Non-gemma4 (reasoning on by default): --thinking is redundant-but-consistent, --no-thinking off.
    [InlineData("qwen3",  false, false, false)]
    [InlineData("qwen3",  true,  false, false)]
    [InlineData("qwen3",  false, true,  true)]
    [InlineData("qwen3",  true,  true,  true)]
    [InlineData("llama",  false, false, false)]
    public void ResolveThinkingOff_FollowsPrecedence(string arch, bool thinking, bool noThinking, bool expectedOff) =>
        Assert.Equal(expectedOff, RunCommand.ResolveThinkingOff(arch, thinking, noThinking));
}
