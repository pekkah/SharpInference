using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

public sealed class KvCacheTests : IDisposable
{
    private readonly KvCache _cache;

    public KvCacheTests()
    {
        // 2 layers, max 16 positions, 2 KV heads, head dim 4 → kvDim = 8
        _cache = new KvCache(numLayers: 2, maxSeqLen: 16, numKvHeads: 2, headDim: 4);
    }

    public void Dispose() => _cache.Dispose();

    [Fact]
    public void NewCache_LengthIsZero()
    {
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void Append_IncreasesLengthAfterIncrement()
    {
        float[] k = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] v = [9, 10, 11, 12, 13, 14, 15, 16];

        _cache.Append(0, k, v);
        _cache.Append(1, k, v);
        _cache.IncrementPosition();

        Assert.Equal(1, _cache.Length);
    }

    [Fact]
    public void GetKeys_ReturnsAppendedData()
    {
        float[] k = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] v = [9, 10, 11, 12, 13, 14, 15, 16];

        _cache.Append(0, k, v);
        _cache.IncrementPosition();

        var keys = _cache.GetKeys(0);
        Assert.Equal(8, keys.Length); // 1 position * kvDim(8)
        Assert.Equal(1f, keys[0]);
        Assert.Equal(8f, keys[7]);
    }

    [Fact]
    public void GetValues_ReturnsAppendedData()
    {
        float[] k = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] v = [9, 10, 11, 12, 13, 14, 15, 16];

        _cache.Append(0, k, v);
        _cache.IncrementPosition();

        var values = _cache.GetValues(0);
        Assert.Equal(8, values.Length);
        Assert.Equal(9f, values[0]);
        Assert.Equal(16f, values[7]);
    }

    [Fact]
    public void MultipleAppends_CorrectSequentialStorage()
    {
        float[] k1 = [1, 1, 1, 1, 1, 1, 1, 1];
        float[] v1 = [2, 2, 2, 2, 2, 2, 2, 2];
        float[] k2 = [3, 3, 3, 3, 3, 3, 3, 3];
        float[] v2 = [4, 4, 4, 4, 4, 4, 4, 4];

        _cache.Append(0, k1, v1);
        _cache.IncrementPosition();
        _cache.Append(0, k2, v2);
        _cache.IncrementPosition();

        Assert.Equal(2, _cache.Length);

        var keys = _cache.GetKeys(0);
        Assert.Equal(16, keys.Length); // 2 positions * 8
        Assert.Equal(1f, keys[0]);   // first position
        Assert.Equal(3f, keys[8]);   // second position
    }

    [Fact]
    public void LayersAreIndependent()
    {
        float[] k0 = [1, 0, 0, 0, 0, 0, 0, 0];
        float[] k1 = [0, 0, 0, 0, 0, 0, 0, 2];
        float[] v = [0, 0, 0, 0, 0, 0, 0, 0];

        _cache.Append(0, k0, v);
        _cache.Append(1, k1, v);
        _cache.IncrementPosition();

        var keys0 = _cache.GetKeys(0);
        var keys1 = _cache.GetKeys(1);

        Assert.Equal(1f, keys0[0]);
        Assert.Equal(0f, keys0[7]);
        Assert.Equal(0f, keys1[0]);
        Assert.Equal(2f, keys1[7]);
    }

    [Fact]
    public void Reset_SetsLengthToZero()
    {
        float[] k = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] v = [1, 2, 3, 4, 5, 6, 7, 8];

        _cache.Append(0, k, v);
        _cache.IncrementPosition();
        Assert.Equal(1, _cache.Length);

        _cache.Reset();
        Assert.Equal(0, _cache.Length);
    }

    [Fact]
    public void KvDim_IsCorrect()
    {
        Assert.Equal(8, _cache.KvDim); // 2 heads * 4 head_dim
    }
}
