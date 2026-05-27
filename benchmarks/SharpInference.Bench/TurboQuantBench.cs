using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using SharpInference.TurboQuant;

namespace SharpInference.Bench;

/// <summary>
/// Per-query K-scoring micro-bench. Compares the existing per-block
/// <see cref="TurboQuantOps.DequantDot4Avx2"/> path against the tiled
/// FastScan kernel at the context lengths the engine sees on Qwen3-8B /
/// Qwen3.6 27B-MTP with TurboQuant enabled. The gate decision for issue #34
/// is "≥ 1.5× standalone speedup vs the per-block path"; this bench is what
/// produces that number.
/// </summary>
[MemoryDiagnoser]
[WarmupCount(3)]
[IterationCount(5)]
public unsafe class FastScanKScoreBench
{
    private const int Dim = 128;
    private const int Bits = 4;

    [Params(1024, 4096, 8192)]
    public int NumPositions;

    private int _blockSize;
    private int _tileBytes;
    private int _vTileBytes;
    private byte* _blocks;        // contiguous per-block compressed buffers
    private byte* _tiles;         // contiguous FastScan K-tiles
    private byte* _vTiles;        // contiguous FastScan V-tiles
    private float* _rotatedQuery;
    private float* _weights;      // attention weights, one per position
    private float* _decompScratch; // per-position dequant scratch for baseline V
    private float* _centroids;
    private float* _signPattern;
    private sbyte* _lut;
    private sbyte* _vLut;
    private float* _scoresOut;
    private float* _vAcc;
    private int _numTiles;

    [GlobalSetup]
    public void Setup()
    {
        _blockSize  = TurboQuantOps.BlockSize(Bits, Dim);
        _tileBytes  = FastScan.TileBytes(Dim);
        _vTileBytes = FastScan.VTileBytes(Dim);
        _numTiles = NumPositions / FastScan.TileSize;

        // Quantize NumPositions random vectors using the engine-native packer
        // so the bench measures what production data looks like.
        _blocks = (byte*)NativeMemory.AllocZeroed((nuint)(NumPositions * _blockSize));
        _tiles  = (byte*)NativeMemory.AllocZeroed((nuint)(_numTiles * _tileBytes));
        _vTiles = (byte*)NativeMemory.AllocZeroed((nuint)(_numTiles * _vTileBytes));
        _rotatedQuery = (float*)NativeMemory.AllocZeroed((nuint)(Dim * sizeof(float)));
        _weights      = (float*)NativeMemory.AllocZeroed((nuint)(NumPositions * sizeof(float)));
        _decompScratch = (float*)NativeMemory.AllocZeroed((nuint)(Dim * sizeof(float)));
        _centroids    = (float*)NativeMemory.AllocZeroed(16 * sizeof(float));
        _signPattern  = (float*)NativeMemory.AllocZeroed((nuint)(Dim * sizeof(float)));
        _lut          = (sbyte*)NativeMemory.AllocZeroed((nuint)(Dim * 16));
        _vLut         = (sbyte*)NativeMemory.AllocZeroed((nuint)(FastScan.TileSize * 16));
        _scoresOut    = (float*)NativeMemory.AllocZeroed((nuint)(NumPositions * sizeof(float)));
        _vAcc         = (float*)NativeMemory.AllocZeroed((nuint)(Dim * sizeof(float)));

        var rng = new Random(42);
        var input = new float[Dim];
        var signPattern = WalshHadamard.GenerateSignPattern(Dim, layerIndex: 0);
        var centroidsArr  = TurboQuantCodebooks.Centroids4Bit_D128.ToArray();
        var boundariesArr = TurboQuantCodebooks.Boundaries4Bit_D128.ToArray();
        for (int v = 0; v < 16; v++) _centroids[v] = centroidsArr[v];
        for (int i = 0; i < Dim; i++) _signPattern[i] = signPattern[i];

        for (int t = 0; t < NumPositions; t++)
        {
            for (int i = 0; i < Dim; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            TurboQuantOps.Quantize(
                input,
                new Span<byte>(_blocks + (long)t * _blockSize, _blockSize),
                signPattern, centroidsArr, boundariesArr,
                bits: Bits, Dim);

            _weights[t] = (float)(rng.NextDouble() * 0.1);
        }

        // Pre-pack tiles once — the engine integration will do this at write
        // time when issue #34 ships, so it doesn't belong in the hot loop here.
        for (int tile = 0; tile < _numTiles; tile++)
        {
            FastScan.PackTile4Bit(
                new ReadOnlySpan<byte>(_blocks + (long)tile * FastScan.TileSize * _blockSize, FastScan.TileSize * _blockSize),
                new Span<byte>(_tiles + (long)tile * _tileBytes, _tileBytes),
                Dim);
            FastScan.PackVTile4Bit(
                new ReadOnlySpan<byte>(_blocks + (long)tile * FastScan.TileSize * _blockSize, FastScan.TileSize * _blockSize),
                new Span<byte>(_vTiles + (long)tile * _vTileBytes, _vTileBytes),
                Dim);
        }

        // Rotated query is freshly built per token by the engine, so we
        // include LUT construction in the FastScan benchmark below.
        for (int i = 0; i < Dim; i++)
            _rotatedQuery[i] = (float)(rng.NextDouble() * 2 - 1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.Free(_blocks);
        NativeMemory.Free(_tiles);
        NativeMemory.Free(_vTiles);
        NativeMemory.Free(_rotatedQuery);
        NativeMemory.Free(_weights);
        NativeMemory.Free(_decompScratch);
        NativeMemory.Free(_centroids);
        NativeMemory.Free(_signPattern);
        NativeMemory.Free(_lut);
        NativeMemory.Free(_vLut);
        NativeMemory.Free(_scoresOut);
        NativeMemory.Free(_vAcc);
    }

    /// <summary>
    /// Baseline: one <see cref="TurboQuantOps.DequantDot4Avx2"/> per position.
    /// Matches the current ForwardPass.cs K-scoring hot loop.
    /// </summary>
    [Benchmark(Baseline = true)]
    public float PerBlockAvx2()
    {
        float sum = 0f;
        for (int t = 0; t < NumPositions; t++)
        {
            sum += TurboQuantOps.DequantDot4Avx2(
                _blocks + (long)t * _blockSize,
                _rotatedQuery,
                _centroids,
                Dim);
        }
        return sum;
    }

    /// <summary>
    /// FastScan: build the per-query i8 LUT once, then process 32 positions
    /// per tile via pshufb. Includes LUT construction so the comparison is
    /// apples-to-apples (the engine also builds the rotated query once per head).
    /// </summary>
    [Benchmark]
    public float FastScanAvx2()
    {
        float scale = FastScan.BuildLut4Bit(
            new ReadOnlySpan<float>(_rotatedQuery, Dim),
            new ReadOnlySpan<float>(_centroids, 16),
            new Span<sbyte>(_lut, Dim * 16),
            Dim);

        float sum = 0f;
        for (int tile = 0; tile < _numTiles; tile++)
        {
            FastScan.KScoreTile4BitAvx2(
                _tiles + (long)tile * _tileBytes,
                _lut,
                scale,
                attnScale: 1.0f,
                _scoresOut + (long)tile * FastScan.TileSize,
                Dim);
        }
        // Touch the result so DCE can't elide the loop.
        for (int t = 0; t < NumPositions; t++) sum += _scoresOut[t];
        return sum;
    }

    /// <summary>
    /// Baseline V-aggregation: full per-position <see cref="TurboQuantOps.Dequantize"/>
    /// (centroids → sign-flip → IWHT in original domain) followed by a scalar
    /// weighted sum. Mirrors the current ForwardPass.cs V hot loop.
    /// </summary>
    [Benchmark]
    public float PerBlockVAggregate()
    {
        Span<float> acc = stackalloc float[Dim];
        acc.Clear();

        for (int t = 0; t < NumPositions; t++)
        {
            TurboQuantOps.Dequantize(
                new ReadOnlySpan<byte>(_blocks + (long)t * _blockSize, _blockSize),
                new Span<float>(_decompScratch, Dim),
                new ReadOnlySpan<float>(_signPattern, Dim),
                new ReadOnlySpan<float>(_centroids, 16),
                bits: Bits, Dim);

            float w = _weights[t];
            for (int d = 0; d < Dim; d++)
                acc[d] += w * _decompScratch[d];
        }

        float sum = 0f;
        for (int d = 0; d < Dim; d++) sum += acc[d];
        return sum;
    }

    /// <summary>
    /// FastScan V-aggregation: per tile we build the 32 × 16 LUT
    /// (effective-weight × centroid) and stream codes through a pshufb pair
    /// over 32 dims at a time. Result is in the rotated domain; the engine
    /// applies sign-flip + IWHT once per kv-head at the end (deferred IWHT
    /// is the structural win Phase 2 will pick up). For an apples-to-apples
    /// comparison we include the deferred IWHT cost here too.
    /// </summary>
    [Benchmark]
    public float FastScanVAggregate()
    {
        // Zero the rotated accumulator.
        for (int d = 0; d < Dim; d++) _vAcc[d] = 0f;

        var effectiveW = new float[FastScan.TileSize];
        var weightsBuf = new float[FastScan.TileSize];

        for (int tile = 0; tile < _numTiles; tile++)
        {
            byte* tilePtr = _vTiles + (long)tile * _vTileBytes;
            // Read the 32 fp16 norms and combine with attention weights into effectiveW.
            for (int t = 0; t < FastScan.TileSize; t++)
            {
                float norm = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(
                    new ReadOnlySpan<byte>(tilePtr + t * 2, 2));
                effectiveW[t] = _weights[tile * FastScan.TileSize + t] * norm;
            }

            float vScale = FastScan.BuildVLut4Bit(
                effectiveW,
                new ReadOnlySpan<float>(_centroids, 16),
                new Span<sbyte>(_vLut, FastScan.TileSize * 16));

            FastScan.VAggregateTile4BitAvx2(tilePtr, _vLut, vScale, _vAcc, Dim);
        }

        // Deferred sign-flip + inverse WHT, mirroring the engine integration cost.
        for (int d = 0; d < Dim; d++) _vAcc[d] *= _signPattern[d];
        WalshHadamard.Transform(
            new ReadOnlySpan<float>(_vAcc, Dim),
            new Span<float>(_vAcc, Dim),
            Dim);

        float sum = 0f;
        for (int d = 0; d < Dim; d++) sum += _vAcc[d];
        return sum;
    }
}
