using SharpInference.Core;
namespace SharpInference.Tests.Core;

public sealed class CoreTests
{
    [Fact]
    public void TensorShape_ElementCount_IsProduct()
    {
        var shape = TensorShape.D3(2, 3, 4);
        Assert.Equal(24L, shape.ElementCount);
    }

    [Fact]
    public void TensorShape_Rank_MatchesDims()
    {
        var shape = TensorShape.D2(8, 16);
        Assert.Equal(2, shape.Rank);
    }}
