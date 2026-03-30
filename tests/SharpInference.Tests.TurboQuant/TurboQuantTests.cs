
namespace SharpInference.Tests.TurboQuant;

public sealed class TurboQuantTests
{
    [Fact]
    public void LloydMaxCodebook_Quantise_ReturnsValidIndex()
    {
        var cb = new SharpInference.TurboQuant.LloydMaxCodebook
        {
            Boundaries = [-0.5f, 0f, 0.5f],
            Centroids = [-0.75f, -0.25f, 0.25f, 0.75f],
        };
        var idx = cb.Quantise(0.1f);
        Assert.True(idx < cb.Centroids.Length);
    }}
