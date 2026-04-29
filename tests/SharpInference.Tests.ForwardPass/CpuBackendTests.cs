using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Tests.ForwardPass;

public sealed class CpuBackendTests : IDisposable
{
    private readonly CpuBackend _backend = new();
    private readonly List<Tensor> _tensors = [];

    public void Dispose()
    {
        foreach (var t in _tensors)
            _backend.Free(t);
        _backend.Dispose();
    }

    private Tensor Upload(float[] data, TensorShape shape)
    {
        var t = _backend.Upload(data, shape);
        _tensors.Add(t);
        return t;
    }

    private Tensor Alloc(TensorShape shape)
    {
        var t = _backend.Allocate(shape);
        _tensors.Add(t);
        return t;
    }

    private float[] Download(Tensor t)
    {
        var result = new float[t.ElementCount];
        _backend.Download(t, result);
        return result;
    }

    // --- Upload / Download ---

    [Fact]
    public void Upload_Download_RoundTrips()
    {
        float[] data = [1f, 2f, 3f, 4f];
        var t = Upload(data, TensorShape.D1(4));
        var result = Download(t);
        Assert.Equal(data, result);
    }

    [Fact]
    public void Allocate_ReturnsZeroed()
    {
        var t = Alloc(TensorShape.D1(8));
        var result = Download(t);
        Assert.All(result, v => Assert.Equal(0f, v));
    }

    // --- MatMul ---

    [Fact]
    public void MatMul_IdentityMatrix()
    {
        // 3x3 identity matrix * [1,2,3] = [1,2,3]
        float[] matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        float[] vector = [1f, 2f, 3f];
        var m = Upload(matrix, TensorShape.D2(3, 3));
        var v = Upload(vector, TensorShape.D1(3));
        var o = Alloc(TensorShape.D1(3));

        _backend.MatMul(o, m, v);
        var result = Download(o);

        Assert.Equal([1f, 2f, 3f], result);
    }

    [Fact]
    public void MatMul_2x3_Times_3()
    {
        // [[1,2,3],[4,5,6]] * [1,1,1] = [6, 15]
        float[] matrix = [1, 2, 3, 4, 5, 6];
        float[] vector = [1f, 1f, 1f];
        var m = Upload(matrix, TensorShape.D2(2, 3));
        var v = Upload(vector, TensorShape.D1(3));
        var o = Alloc(TensorShape.D1(2));

        _backend.MatMul(o, m, v);
        var result = Download(o);

        Assert.Equal([6f, 15f], result);
    }

    [Fact]
    public void MatMul_KnownValues()
    {
        // [[2,0],[0,3]] * [4,5] = [8, 15]
        float[] matrix = [2, 0, 0, 3];
        float[] vector = [4f, 5f];
        var m = Upload(matrix, TensorShape.D2(2, 2));
        var v = Upload(vector, TensorShape.D1(2));
        var o = Alloc(TensorShape.D1(2));

        _backend.MatMul(o, m, v);
        var result = Download(o);

        Assert.Equal([8f, 15f], result);
    }

    // --- AddInPlace ---

    [Fact]
    public void AddInPlace_AddsElementwise()
    {
        float[] a = [1f, 2f, 3f, 4f];
        float[] b = [10f, 20f, 30f, 40f];
        var ta = Upload(a, TensorShape.D1(4));
        var tb = Upload(b, TensorShape.D1(4));

        _backend.AddInPlace(ta, tb);
        var result = Download(ta);

        Assert.Equal([11f, 22f, 33f, 44f], result);
    }

    // --- ElementwiseMul ---

    [Fact]
    public void ElementwiseMul_MultipliesElementwise()
    {
        float[] a = [1f, 2f, 3f, 4f];
        float[] b = [5f, 6f, 7f, 8f];
        var ta = Upload(a, TensorShape.D1(4));
        var tb = Upload(b, TensorShape.D1(4));
        var to = Alloc(TensorShape.D1(4));

        _backend.ElementwiseMul(to, ta, tb);
        var result = Download(to);

        Assert.Equal([5f, 12f, 21f, 32f], result);
    }

    // --- RmsNorm ---

    [Fact]
    public void RmsNorm_UnitWeight_NormalizesCorrectly()
    {
        // x = [1, 1, 1, 1], weight = [1, 1, 1, 1]
        // rms = sqrt(mean([1,1,1,1]) + eps) ≈ 1.0
        // output ≈ [1, 1, 1, 1] (each normalized to ~1)
        float[] x = [1f, 1f, 1f, 1f];
        float[] w = [1f, 1f, 1f, 1f];
        var tx = Upload(x, TensorShape.D1(4));
        var tw = Upload(w, TensorShape.D1(4));
        var to = Alloc(TensorShape.D1(4));

        _backend.RmsNorm(to, tx, tw);
        var result = Download(to);

        for (int i = 0; i < 4; i++)
            Assert.InRange(result[i], 0.999f, 1.001f);
    }

    [Fact]
    public void RmsNorm_WithWeight_ScalesCorrectly()
    {
        // x = [3, 4], weight = [2, 0.5]
        // rms = sqrt((9+16)/2 + eps) = sqrt(12.5 + eps) ≈ 3.5355
        // normalized = [3/3.5355, 4/3.5355] = [0.8485, 1.1314]
        // output = [0.8485*2, 1.1314*0.5] = [1.6971, 0.5657]
        float[] x = [3f, 4f];
        float[] w = [2f, 0.5f];
        var tx = Upload(x, TensorShape.D1(2));
        var tw = Upload(w, TensorShape.D1(2));
        var to = Alloc(TensorShape.D1(2));

        _backend.RmsNorm(to, tx, tw);
        var result = Download(to);

        Assert.InRange(result[0], 1.696f, 1.698f);
        Assert.InRange(result[1], 0.565f, 0.567f);
    }

    // --- Softmax ---

    [Fact]
    public void Softmax_SumsToOne()
    {
        float[] data = [1f, 2f, 3f, 4f];
        var t = Upload(data, TensorShape.D1(4));

        _backend.Softmax(t);
        var result = Download(t);

        float sum = result.Sum();
        Assert.InRange(sum, 0.999f, 1.001f);
    }

    [Fact]
    public void Softmax_IsMonotonic()
    {
        float[] data = [1f, 2f, 3f, 4f];
        var t = Upload(data, TensorShape.D1(4));

        _backend.Softmax(t);
        var result = Download(t);

        // Larger inputs → larger softmax outputs
        for (int i = 1; i < result.Length; i++)
            Assert.True(result[i] > result[i - 1]);
    }

    [Fact]
    public void Softmax_KnownValues()
    {
        // softmax([0, 0]) = [0.5, 0.5]
        float[] data = [0f, 0f];
        var t = Upload(data, TensorShape.D1(2));

        _backend.Softmax(t);
        var result = Download(t);

        Assert.InRange(result[0], 0.499f, 0.501f);
        Assert.InRange(result[1], 0.499f, 0.501f);
    }

    [Fact]
    public void Softmax_NumericallyStable_LargeValues()
    {
        // Should not overflow with large values
        float[] data = [1000f, 1001f, 1002f];
        var t = Upload(data, TensorShape.D1(3));

        _backend.Softmax(t);
        var result = Download(t);

        Assert.All(result, v => Assert.False(float.IsNaN(v)));
        Assert.All(result, v => Assert.False(float.IsInfinity(v)));
        Assert.InRange(result.Sum(), 0.999f, 1.001f);
    }

    // --- SiLU ---

    [Fact]
    public void SiLU_ZeroInput_ReturnsZero()
    {
        float[] data = [0f];
        var t = Upload(data, TensorShape.D1(1));

        _backend.SiLU(t);
        var result = Download(t);

        Assert.InRange(result[0], -0.001f, 0.001f);
    }

    [Fact]
    public void SiLU_KnownValues()
    {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        // SiLU(-1) = -1 / (1 + exp(1)) ≈ -0.2689
        float[] data = [1f, -1f];
        var t = Upload(data, TensorShape.D1(2));

        _backend.SiLU(t);
        var result = Download(t);

        Assert.InRange(result[0], 0.730f, 0.732f);
        Assert.InRange(result[1], -0.270f, -0.268f);
    }

    [Fact]
    public void SiLU_LargePositive_ApproachesIdentity()
    {
        // For large positive x, sigmoid(x) → 1, so SiLU(x) → x
        float[] data = [10f];
        var t = Upload(data, TensorShape.D1(1));

        _backend.SiLU(t);
        var result = Download(t);

        Assert.InRange(result[0], 9.999f, 10.001f);
    }

    // --- RoPE ---

    [Fact]
    public void RoPE_Position0_NoRotation()
    {
        // At position 0, all angles are 0, so cos=1, sin=0 → no change
        float[] data = [1f, 2f, 3f, 4f];
        var t = Upload(data, TensorShape.D1(4));

        _backend.RoPE(t, position: 0, headDim: 4);
        var result = Download(t);

        Assert.Equal(1f, result[0], 0.001f);
        Assert.Equal(2f, result[1], 0.001f);
        Assert.Equal(3f, result[2], 0.001f);
        Assert.Equal(4f, result[3], 0.001f);
    }

    [Fact]
    public void RoPE_PreservesNorm()
    {
        // Default (interleaved/LLaMA): pairs are (data[0], data[1]) and (data[2], data[3])
        float[] data = [3f, 4f, 1f, 0f];
        var t = Upload(data, TensorShape.D1(4));

        float norm0Before = MathF.Sqrt(3f * 3f + 4f * 4f);  // pair (3,4)
        float norm1Before = MathF.Sqrt(1f * 1f + 0f * 0f);  // pair (1,0)

        _backend.RoPE(t, position: 5, headDim: 4);
        var result = Download(t);

        float norm0After = MathF.Sqrt(result[0] * result[0] + result[1] * result[1]);
        float norm1After = MathF.Sqrt(result[2] * result[2] + result[3] * result[3]);

        Assert.InRange(norm0After, norm0Before - 0.01f, norm0Before + 0.01f);
        Assert.InRange(norm1After, norm1Before - 0.01f, norm1Before + 0.01f);
    }

    [Fact]
    public void RoPE_Neox_PreservesNorm()
    {
        // NEOX convention: pairs are (data[0], data[halfDim]) and (data[1], data[halfDim+1])
        float[] data = [3f, 4f, 1f, 0f];
        var t = Upload(data, TensorShape.D1(4));

        float norm0Before = MathF.Sqrt(3f * 3f + 1f * 1f);  // pair (data[0], data[2])
        float norm1Before = MathF.Sqrt(4f * 4f + 0f * 0f);  // pair (data[1], data[3])

        _backend.RoPE(t, position: 5, headDim: 4, ropeTheta: 10000f, neox: true);
        var result = Download(t);

        float norm0After = MathF.Sqrt(result[0] * result[0] + result[2] * result[2]);
        float norm1After = MathF.Sqrt(result[1] * result[1] + result[3] * result[3]);

        Assert.InRange(norm0After, norm0Before - 0.01f, norm0Before + 0.01f);
        Assert.InRange(norm1After, norm1Before - 0.01f, norm1Before + 0.01f);
    }

    [Fact]
    public void RoPE_DifferentPositions_DifferentResults()
    {
        float[] data1 = [1f, 0f, 0f, 1f];
        float[] data2 = [1f, 0f, 0f, 1f];
        var t1 = Upload(data1, TensorShape.D1(4));
        var t2 = Upload(data2, TensorShape.D1(4));

        _backend.RoPE(t1, position: 1, headDim: 4);
        _backend.RoPE(t2, position: 2, headDim: 4);

        var r1 = Download(t1);
        var r2 = Download(t2);

        // At least one element should differ
        bool differ = false;
        for (int i = 0; i < 4; i++)
            if (MathF.Abs(r1[i] - r2[i]) > 0.001f) differ = true;
        Assert.True(differ, "RoPE at different positions should produce different results");
    }
}
