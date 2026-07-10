using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using SharpInference.TurboQuant;

namespace SharpInference.Bench;

/// <summary>
/// KVarN fused K-score micro-bench (issue #180 P1): scalar reference vs the
/// AVX2 kernel inside <see cref="KVarNCompressor.KeyScores"/>, D=128, walking
/// 64 compressed tiles per invocation (8192 positions — the decode depth where
/// the compressed region dominates the token loop). OperationsPerInvoke is the
/// tile count, so the reported time is ns/tile. The scalar path is forced via
/// the internal <see cref="KVarNCompressor.ForceScalar"/> hook.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public unsafe class KVarNKeyScoreBench
{
    private const int Dim = 128;
    private const int NumTiles = 64;

    private KVarNCompressor _comp = null!;
    private int _kTileBytes;
    private byte* _kTiles;
    private float* _rotatedQuery;
    private float* _scores;

    [GlobalSetup]
    public void Setup()
    {
        _comp = new KVarNCompressor(Dim);
        _kTileBytes = _comp.KeyTileBytes;
        _kTiles = (byte*)NativeMemory.AllocZeroed((nuint)((long)NumTiles * _kTileBytes));
        _rotatedQuery = (float*)NativeMemory.AllocZeroed(Dim * sizeof(float));
        _scores = (float*)NativeMemory.AllocZeroed((nuint)((long)NumTiles * KVarNCompressor.TileTokens * sizeof(float)));

        var rng = new Random(42);
        float[] tileData = new float[KVarNCompressor.TileTokens * Dim];
        for (int tile = 0; tile < NumTiles; tile++)
        {
            for (int i = 0; i < tileData.Length; i++)
                tileData[i] = (float)(rng.NextDouble() * 2 - 1);
            _comp.CompressKeyTile(tileData, new Span<byte>(_kTiles + (long)tile * _kTileBytes, _kTileBytes));
        }

        float[] q = new float[Dim];
        for (int i = 0; i < Dim; i++)
            q[i] = (float)(rng.NextDouble() * 2 - 1);
        _comp.RotateQuery(q, new Span<float>(_rotatedQuery, Dim));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.Free(_kTiles);
        NativeMemory.Free(_rotatedQuery);
        NativeMemory.Free(_scores);
    }

    [Benchmark(Baseline = true, OperationsPerInvoke = NumTiles)]
    public float Scalar()
    {
        KVarNCompressor.ForceScalar = true;
        try
        {
            return RunAllTiles();
        }
        finally
        {
            KVarNCompressor.ForceScalar = false;
        }
    }

    [Benchmark(OperationsPerInvoke = NumTiles)]
    public float Avx2() => RunAllTiles();

    private float RunAllTiles()
    {
        var query = new ReadOnlySpan<float>(_rotatedQuery, Dim);
        for (int tile = 0; tile < NumTiles; tile++)
        {
            _comp.KeyScores(
                new ReadOnlySpan<byte>(_kTiles + (long)tile * _kTileBytes, _kTileBytes),
                query,
                new Span<float>(_scores + (long)tile * KVarNCompressor.TileTokens, KVarNCompressor.TileTokens));
        }
        return _scores[0]; // defeat DCE
    }
}

/// <summary>
/// KVarN fused V-aggregate micro-bench: scalar vs AVX2 inside
/// <see cref="KVarNCompressor.AggregateValues"/>, D=128 over 64 tiles.
/// Reported time is ns/tile (OperationsPerInvoke = tile count).
/// </summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public unsafe class KVarNVAggregateBench
{
    private const int Dim = 128;
    private const int NumTiles = 64;

    private KVarNCompressor _comp = null!;
    private int _vTileBytes;
    private byte* _vTiles;
    private float* _weights;
    private float* _acc;

    [GlobalSetup]
    public void Setup()
    {
        _comp = new KVarNCompressor(Dim);
        _vTileBytes = _comp.ValueTileBytes;
        _vTiles = (byte*)NativeMemory.AllocZeroed((nuint)((long)NumTiles * _vTileBytes));
        _weights = (float*)NativeMemory.AllocZeroed((nuint)((long)NumTiles * KVarNCompressor.TileTokens * sizeof(float)));
        _acc = (float*)NativeMemory.AllocZeroed(Dim * sizeof(float));

        var rng = new Random(43);
        float[] tileData = new float[KVarNCompressor.TileTokens * Dim];
        for (int tile = 0; tile < NumTiles; tile++)
        {
            for (int i = 0; i < tileData.Length; i++)
                tileData[i] = (float)(rng.NextDouble() * 2 - 1);
            _comp.CompressValueTile(tileData, new Span<byte>(_vTiles + (long)tile * _vTileBytes, _vTileBytes));
        }

        // Softmax-shaped positive weights (all nonzero: worst case for the kernels).
        int total = NumTiles * KVarNCompressor.TileTokens;
        double sum = 0;
        for (int t = 0; t < total; t++)
        {
            _weights[t] = MathF.Exp((float)(rng.NextDouble() * 4 - 2));
            sum += _weights[t];
        }
        for (int t = 0; t < total; t++)
            _weights[t] = (float)(_weights[t] / sum);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.Free(_vTiles);
        NativeMemory.Free(_weights);
        NativeMemory.Free(_acc);
    }

    [Benchmark(Baseline = true, OperationsPerInvoke = NumTiles)]
    public float Scalar()
    {
        KVarNCompressor.ForceScalar = true;
        try
        {
            return RunAllTiles();
        }
        finally
        {
            KVarNCompressor.ForceScalar = false;
        }
    }

    [Benchmark(OperationsPerInvoke = NumTiles)]
    public float Avx2() => RunAllTiles();

    private float RunAllTiles()
    {
        var acc = new Span<float>(_acc, Dim);
        acc.Clear();
        for (int tile = 0; tile < NumTiles; tile++)
        {
            _comp.AggregateValues(
                new ReadOnlySpan<byte>(_vTiles + (long)tile * _vTileBytes, _vTileBytes),
                new ReadOnlySpan<float>(_weights + (long)tile * KVarNCompressor.TileTokens, KVarNCompressor.TileTokens),
                acc);
        }
        return _acc[0]; // defeat DCE
    }
}
