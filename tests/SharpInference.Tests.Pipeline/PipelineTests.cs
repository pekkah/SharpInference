
namespace SharpInference.Tests.Pipeline;

public sealed class PipelineTests
{
    [Fact]
    public void ExpertCache_Miss_ReturnsFalse()
    {
        var cache = new SharpInference.Pipeline.ExpertCache(capacity: 4);
        Assert.False(cache.TryGet(0, out _));
        cache.Dispose();
    }}
