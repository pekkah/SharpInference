using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// Selects the codec used for the compressed region of <see cref="TurboQuantKvCache"/>
/// (issue #180: KVarN is built as a selectable quantizer inside the existing TurboQuant
/// cache machinery, not a second cache type).
/// </summary>
public enum TqQuantizer
{
    /// <summary>
    /// Lloyd-Max codebooks in FastScan tiles (3-4 bit, 32-position tiles with
    /// per-block staging) — the original TurboQuant path. Known to severely degrade
    /// quality on QK-norm models such as Qwen3, whose attention logits are large
    /// enough that the codec's ~7% relative K-score error (3-bit) destroys the
    /// softmax (issue #432: Qwen3-0.6B wikitext-2 PPL 945.6 vs 15.47 fp32; 4-bit
    /// improves to 51.8 but still degrades). Prefer <see cref="KVarN"/> — the CLI
    /// and server default to it wherever it is supported.
    /// </summary>
    LloydMax = 0,

    /// <summary>
    /// KVarN (issue #180): Hadamard rotation + dual-axis Sinkhorn variance
    /// normalization + asymmetric RTN — 4-bit keys / 2-bit values in whole
    /// 128-token tiles assembled directly from the FP32 window.
    /// </summary>
    KVarN = 1,
}

/// <summary>
/// Hybrid FP32/TurboQuant KV cache: recent tokens in FP32, older tokens compressed to 2-4 bits.
/// The FP32 window provides full-precision attention for recently generated tokens.
/// When tokens age out of the window, they are compressed via the configured
/// <see cref="TqQuantizer"/>: Lloyd-Max FastScan tiles (default) or KVarN 128-token tiles.
/// </summary>
public sealed unsafe class TurboQuantKvCache : IDisposable
{
    private readonly int _numLayers;
    private readonly int _maxSeqLen;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _kvDim;         // numKvHeads * headDim
    private readonly int _fp32WindowSize; // recent tokens in FP32

    // Per-layer FP32 window (linear shift-down buffer, slot 0 = oldest — not a ring)
    private readonly float*[] _fp32Keys;
    private readonly float*[] _fp32Values;

    // Per-layer TQ compressed storage. In Lloyd-Max mode both keys and values
    // live in FastScan tile layout (one tile = 32 positions, contiguous per
    // kv-head) with the in-flight 0..31 positions staged in per-block format.
    // K and V tiles use different internal layouts (K is dim-major, V is
    // position-major — see FastScan.TileBytes / VTileBytes) but happen to be
    // the same total bytes. In KVarN mode the same pointers hold KVarN 128-token
    // tiles (strides KVarNCompressor.KeyTileBytes / ValueTileBytes); there is no
    // staging — whole tiles are compressed straight out of the FP32 window, so
    // the window owns everything mod 128 and the staging buffers stay null.
    private readonly byte*[] _tqKeyTiles;      // [layer][numTiles * numKvHeads * _keyTileStride]
    private readonly byte*[] _tqKeyStaging;    // [layer][numKvHeads * TileSize * _tqBlockSize] (Lloyd-Max only)
    private readonly byte*[] _tqValueTiles;    // [layer][numTiles * numKvHeads * _valueTileStride]
    private readonly byte*[] _tqValueStaging;  // [layer][numKvHeads * TileSize * _tqBlockSize] (Lloyd-Max only)
    private readonly KvCacheCompressor[][]? _keyCompressors;   // [layer][kvHead] (Lloyd-Max only)
    private readonly KvCacheCompressor[][]? _valueCompressors;

    // KVarN mode (issue #180). One shared compressor: the write path (tile
    // compression inside Append) is single-threaded, and the read methods
    // (RotateQuery / KeyScores / AggregateValues) are stateless and safe to
    // call from the per-head Parallel.For in TqAttention.
    private readonly KVarNCompressor? _kvarn;
    private readonly float[]? _kvarnGather;   // [TileTokens * headDim] per-head gather scratch (write path)

    private readonly TqQuantizer _quantizer;
    private readonly int _tileTokens;         // positions per compressed tile (32 FastScan / 128 KVarN)
    private readonly int _keyTileStride;      // bytes per (tile, kv-head) K tile
    private readonly int _valueTileStride;    // bytes per (tile, kv-head) V tile
    private readonly int _tqBlockSize;        // bytes per compressed block per KV head (Lloyd-Max only)
    private readonly int _tileBytes;          // FastScan.TileBytes(_headDim)  (K-layout, Lloyd-Max only)
    private readonly int _vTileBytes;         // FastScan.VTileBytes(_headDim) (V-layout, Lloyd-Max only)
    private readonly int _stagingBytesPerHead; // TileSize * _tqBlockSize (Lloyd-Max only)
    private readonly int _bits;

    private int _totalLength;    // total positions stored (TQ + FP32)
    private readonly int[] _layerTqLengths; // positions in TQ storage per layer
    private bool _disposed;

    /// <summary>Total positions stored in the cache (compressed + FP32).</summary>
    public int Length => _totalLength;

    /// <summary>Number of compressed positions.</summary>
    public int TqLength => _numLayers > 0 ? _layerTqLengths[0] : 0;

    /// <summary>Number of FP32 positions.</summary>
    public int Fp32Length => _totalLength - TqLength;

    /// <summary>
    /// Largest compressed-region length across all layers — the floor below which a rewind
    /// would discard compressed data. Batched prefill advances each layer's counter
    /// independently (see <see cref="ResetTotalLengthForBatchedPrefill"/>), so this can
    /// exceed <see cref="TqLength"/> (layer 0's view) mid-prefill. Callers gating a rewind
    /// must use this, not <see cref="TqLength"/>.
    /// </summary>
    public int MaxTqLength
    {
        get
        {
            int max = 0;
            foreach (int len in _layerTqLengths)
                if (len > max) max = len;
            return max;
        }
    }

    /// <summary>FP32 window size.</summary>
    public int Fp32WindowSize => _fp32WindowSize;

    public int KvDim => _kvDim;
    public int MaxSeqLen => _maxSeqLen;
    public int HeadDim => _headDim;
    public int NumKvHeads => _numKvHeads;
    public int TqBlockSize => _tqBlockSize;
    public int Bits => _bits;
    public int TileBytes => _tileBytes;
    public int FastScanTileSize => SharpInference.TurboQuant.FastScan.TileSize;

    /// <summary>The quantizer used for the compressed region.</summary>
    public TqQuantizer Quantizer => _quantizer;

    /// <summary>Positions per compressed tile (32 for Lloyd-Max FastScan, 128 for KVarN).</summary>
    public int TileSizeTokens => _tileTokens;

    public TurboQuantKvCache(int numLayers, int maxSeqLen, int numKvHeads, int headDim,
        int fp32WindowSize = 256, int bits = 3, int layerIndexBase = 0, int totalLayerCountForSeeds = 0,
        TqQuantizer quantizer = TqQuantizer.LloydMax)
    {
        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _kvDim = numKvHeads * headDim;
        _fp32WindowSize = fp32WindowSize;
        _bits = bits;
        _quantizer = quantizer;
        if (quantizer == TqQuantizer.KVarN)
        {
            // KVarN (issue #180): whole 128-token tiles are compressed straight
            // out of the FP32 window (the dual-axis Sinkhorn normalization needs
            // the full tile assembled; partial tiles are rejected by design), so
            // the window must be able to hold at least one tile whenever a
            // compressed region can exist.
            if (maxSeqLen > fp32WindowSize && fp32WindowSize < KVarNCompressor.TileTokens)
                throw new ArgumentException(
                    $"KVarN quantizer requires fp32WindowSize >= {KVarNCompressor.TileTokens} (one full tile) " +
                    $"when maxSeqLen ({maxSeqLen}) exceeds the FP32 window; got fp32WindowSize={fp32WindowSize}.",
                    nameof(fp32WindowSize));
            _kvarn = new KVarNCompressor(headDim); // validates power-of-2 head dim in [8, 1024]
            _kvarnGather = new float[KVarNCompressor.TileTokens * headDim];
            _tileTokens = KVarNCompressor.TileTokens;
            _keyTileStride = _kvarn.KeyTileBytes;
            _valueTileStride = _kvarn.ValueTileBytes;
            // Lloyd-Max per-block/FastScan sizing is unused in KVarN mode.
            _tqBlockSize = 0;
            _tileBytes = 0;
            _vTileBytes = 0;
            _stagingBytesPerHead = 0;
        }
        else
        {
            _tqBlockSize = TurboQuantOps.BlockSize(bits, headDim);
            _tileBytes = SharpInference.TurboQuant.FastScan.TileBytes(headDim);
            _vTileBytes = SharpInference.TurboQuant.FastScan.VTileBytes(headDim);
            _stagingBytesPerHead = SharpInference.TurboQuant.FastScan.TileSize * _tqBlockSize;
            _tileTokens = SharpInference.TurboQuant.FastScan.TileSize;
            _keyTileStride = _tileBytes;
            _valueTileStride = _vTileBytes;
        }
        if (totalLayerCountForSeeds == 0)
            totalLayerCountForSeeds = numLayers;
        _layerTqLengths = new int[numLayers];

        // FP32 window storage per layer
        _fp32Keys = new float*[numLayers];
        _fp32Values = new float*[numLayers];
        var fp32Bytes = (nuint)((long)fp32WindowSize * _kvDim * sizeof(float));
        for (int i = 0; i < numLayers; i++)
        {
            _fp32Keys[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
            _fp32Values[i] = (float*)NativeMemory.AllocZeroed(fp32Bytes);
        }

        // TQ compressed storage per layer. Lloyd-Max: K and V each get tile +
        // 32-slot per-block staging, with K and V tile layouts being transposes
        // of each other (same total bytes per (tile, head)). KVarN: K4/V2 tiles
        // only (no staging — whole tiles come out of the FP32 window). The
        // ceil-divide tile count covers both promotion cadences: Lloyd-Max
        // promotes one position at a time (max TQ length maxSeqLen − window),
        // KVarN promotes 128 at a time (max TQ length maxSeqLen − window + 127,
        // reached only in whole-tile steps — the same ceiling).
        int maxTqPositions = maxSeqLen - fp32WindowSize;
        if (maxTqPositions < 0) maxTqPositions = 0;
        int maxTiles = (maxTqPositions + _tileTokens - 1) / _tileTokens;

        _tqKeyTiles     = new byte*[numLayers];
        _tqKeyStaging   = new byte*[numLayers];
        _tqValueTiles   = new byte*[numLayers];
        _tqValueStaging = new byte*[numLayers];
        if (maxTqPositions > 0)
        {
            var keyTileBytes   = (nuint)((long)maxTiles * numKvHeads * _keyTileStride);
            var valueTileBytes = (nuint)((long)maxTiles * numKvHeads * _valueTileStride);
            var stageBytes     = (nuint)((long)numKvHeads * _stagingBytesPerHead);
            for (int i = 0; i < numLayers; i++)
            {
                _tqKeyTiles[i]   = (byte*)NativeMemory.AllocZeroed(keyTileBytes);
                _tqValueTiles[i] = (byte*)NativeMemory.AllocZeroed(valueTileBytes);
                if (quantizer == TqQuantizer.LloydMax)
                {
                    _tqKeyStaging[i]   = (byte*)NativeMemory.AllocZeroed(stageBytes);
                    _tqValueStaging[i] = (byte*)NativeMemory.AllocZeroed(stageBytes);
                }
            }
        }

        // Create Lloyd-Max compressors per layer per KV head (KVarN needs no
        // per-(layer, head) state — no codebooks, no sign-pattern seeds).
        if (quantizer == TqQuantizer.LloydMax)
        {
            _keyCompressors = new KvCacheCompressor[numLayers][];
            _valueCompressors = new KvCacheCompressor[numLayers][];
            for (int layer = 0; layer < numLayers; layer++)
            {
                _keyCompressors[layer] = new KvCacheCompressor[numKvHeads];
                _valueCompressors[layer] = new KvCacheCompressor[numKvHeads];
                for (int head = 0; head < numKvHeads; head++)
                {
                    int globalLayer = layer + layerIndexBase;
                    _keyCompressors[layer][head] = new KvCacheCompressor(bits, headDim, globalLayer * numKvHeads + head);
                    _valueCompressors[layer][head] = new KvCacheCompressor(bits, headDim, (globalLayer + totalLayerCountForSeeds) * numKvHeads + head);
                }
            }
        }
    }

    public TurboQuantKvCache(ModelHyperparams hp, int fp32WindowSize = 256, int bits = 3,
        int layerIndexBase = 0, int totalLayerCountForSeeds = 0, TqQuantizer quantizer = TqQuantizer.LloydMax)
        : this(hp.NumLayers, hp.ContextLength, hp.NumKvHeads, hp.HeadDim,
               fp32WindowSize, bits, layerIndexBase, totalLayerCountForSeeds, quantizer)
    {
    }

    /// <summary>
    /// Append K/V vectors for a given layer. If the FP32 window is full,
    /// the oldest FP32 position is compressed to TQ first.
    /// </summary>
    public void Append(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_totalLength >= _maxSeqLen)
            throw new InvalidOperationException(
                $"TQ KV cache full: {_totalLength} >= {_maxSeqLen}");

        int fp32Count = _totalLength - _layerTqLengths[layer];

        // If FP32 window is full, promote the oldest entries into the compressed
        // region: one position (Lloyd-Max staging) or one whole tile (KVarN).
        if (fp32Count >= _fp32WindowSize)
        {
            if (_quantizer == TqQuantizer.KVarN)
                CompressOldestFp32TileKVarN(layer);
            else
                CompressOldestFp32(layer);
            fp32Count = _totalLength - _layerTqLengths[layer];
        }

        // Write to FP32 window at the current FP32 slot
        int fp32Slot = fp32Count;
        long offset = (long)fp32Slot * _kvDim;
        key[.._kvDim].CopyTo(new Span<float>(_fp32Keys[layer] + offset, _kvDim));
        value[.._kvDim].CopyTo(new Span<float>(_fp32Values[layer] + offset, _kvDim));
    }

    /// <summary>
    /// Advances the position counter. Call once per token after appending to all layers.
    /// </summary>
    public void IncrementPosition() => _totalLength++;

    /// <summary>
    /// Returns true if the given position is in the TQ-compressed region.
    /// </summary>
    public bool IsCompressed(int position) => position < TqLength;

    /// <summary>Returns the number of compressed positions for the given layer.</summary>
    public int GetTqLength(int layer) => _layerTqLengths[layer];

    /// <summary>
    /// Returns a pointer to a FP32 key at the given position for a given layer.
    /// Only valid for positions in the FP32 window (position >= TqLength).
    /// </summary>
    public float* Fp32KeyAt(int layer, int position)
    {
        int fp32Idx = position - _layerTqLengths[layer];
        return _fp32Keys[layer] + (long)fp32Idx * _kvDim;
    }

    /// <summary>
    /// Returns a pointer to a FP32 value at the given position for a given layer.
    /// Only valid for positions in the FP32 window.
    /// </summary>
    public float* Fp32ValueAt(int layer, int position)
    {
        int fp32Idx = position - _layerTqLengths[layer];
        return _fp32Values[layer] + (long)fp32Idx * _kvDim;
    }

    /// <summary>
    /// Pointer to a compressed K-tile (<see cref="TileSizeTokens"/> positions ×
    /// one kv-head): FastScan layout in Lloyd-Max mode, KVarN packed layout in
    /// KVarN mode. Tile index is 0..NumKeyTiles(layer)-1; in Lloyd-Max mode
    /// positions beyond <c>NumKeyTiles × 32</c> are in the staging buffer and
    /// accessed via <see cref="KeyStagingBlockAt"/> (KVarN has no staging).
    /// </summary>
    public byte* KeyTileAt(int layer, int tileIdx, int kvHead)
    {
        long byteOffset = ((long)tileIdx * _numKvHeads + kvHead) * _keyTileStride;
        return _tqKeyTiles[layer] + byteOffset;
    }

    /// <summary>
    /// Pointer to one per-block compressed key in the in-flight staging buffer.
    /// Valid for <c>stagingIdx in 0..KeyStagingCount(layer)-1</c>.
    /// </summary>
    public byte* KeyStagingBlockAt(int layer, int stagingIdx, int kvHead)
    {
        long byteOffset = (long)kvHead * _stagingBytesPerHead + stagingIdx * _tqBlockSize;
        return _tqKeyStaging[layer] + byteOffset;
    }

    /// <summary>Number of complete compressed tiles for this layer's K cache.</summary>
    public int NumKeyTiles(int layer) =>
        _layerTqLengths[layer] / _tileTokens;

    /// <summary>
    /// Number of positions currently in the staging buffer (0..31 in Lloyd-Max
    /// mode; always 0 in KVarN mode — the TQ length only grows in whole tiles).
    /// </summary>
    public int KeyStagingCount(int layer) =>
        _layerTqLengths[layer] % _tileTokens;

    /// <summary>Pointer to a compressed V-tile (<see cref="TileSizeTokens"/> positions × one kv-head).</summary>
    public byte* ValueTileAt(int layer, int tileIdx, int kvHead)
    {
        long byteOffset = ((long)tileIdx * _numKvHeads + kvHead) * _valueTileStride;
        return _tqValueTiles[layer] + byteOffset;
    }

    /// <summary>Pointer to one per-block compressed value in the V staging buffer.</summary>
    public byte* ValueStagingBlockAt(int layer, int stagingIdx, int kvHead)
    {
        long byteOffset = (long)kvHead * _stagingBytesPerHead + stagingIdx * _tqBlockSize;
        return _tqValueStaging[layer] + byteOffset;
    }

    /// <summary>
    /// Returns the Lloyd-Max key compressor for the given layer and KV head.
    /// Used to rotate queries and compute fused dequant-dot. Unavailable in
    /// KVarN mode — use <see cref="RotateQuery"/> / <see cref="ComputeKScores"/>.
    /// </summary>
    public KvCacheCompressor GetKeyCompressor(int layer, int kvHead) =>
        _keyCompressors is { } kc
            ? kc[layer][kvHead]
            : throw new InvalidOperationException(
                "Lloyd-Max compressors are unavailable in KVarN quantizer mode; use RotateQuery/ComputeKScores/ComputeVAggregation.");

    /// <summary>
    /// Returns the Lloyd-Max value compressor for the given layer and KV head.
    /// Unavailable in KVarN mode.
    /// </summary>
    public KvCacheCompressor GetValueCompressor(int layer, int kvHead) =>
        _valueCompressors is { } vc
            ? vc[layer][kvHead]
            : throw new InvalidOperationException(
                "Lloyd-Max compressors are unavailable in KVarN quantizer mode; use RotateQuery/ComputeKScores/ComputeVAggregation.");

    /// <summary>
    /// Rotate a query vector into this cache's compressed-domain basis: plain
    /// Walsh-Hadamard in KVarN mode, per-(layer, kv-head) sign-flip +
    /// Walsh-Hadamard in Lloyd-Max mode. Call once per head per decode step;
    /// the result feeds <see cref="ComputeKScores"/>. Thread-safe (stateless).
    /// </summary>
    public void RotateQuery(int layer, int kvHead, ReadOnlySpan<float> query, Span<float> rotatedQuery)
    {
        if (_quantizer == TqQuantizer.KVarN)
            _kvarn!.RotateQuery(query, rotatedQuery);
        else
            _keyCompressors![layer][kvHead].RotateQuery(query, rotatedQuery);
    }

    /// <summary>
    /// Sets the global position counter to <paramref name="length"/> without
    /// touching per-layer TQ state. Intended for batched prefill: the per-layer
    /// loop processes N tokens through one layer (advancing _totalLength from
    /// startPos to startPos+N), then snaps the counter back to startPos before
    /// the next layer. Per-layer K/V (FastScan tiles + staging + FP32 window)
    /// stays intact, so each layer's TQ state evolves independently across the
    /// shared global counter.
    /// </summary>
    public void ResetTotalLengthForBatchedPrefill(int length) => _totalLength = length;

    /// <summary>Resets the cache for a new generation.</summary>
    public void Reset()
    {
        _totalLength = 0;
        Array.Clear(_layerTqLengths);
    }

    /// <summary>
    /// Truncates the cache to the given length, which must fall within the FP32 window
    /// (i.e., length >= TqLength). Positions in the compressed TQ region cannot be undone.
    /// Throws <see cref="NotSupportedException"/> if length would truncate into TQ-compressed data.
    /// Note the guaranteed rewind depth differs by quantizer: Lloyd-Max promotes 32 positions
    /// at a time (depth ≥ window − 31), KVarN promotes whole 128-token tiles (depth ≥
    /// window − 127 — just 1 position at the minimum window of 128). Callers that rewind
    /// (speculative decoding) must keep the window comfortably above their draft length.
    /// </summary>
    public void TruncateTo(int length)
    {
        int tqLen = TqLength;
        if (length < tqLen)
            throw new NotSupportedException(
                $"TruncateTo({length}) cannot truncate into TQ-compressed region (tqLength={tqLen}). " +
                "Speculative decoding is not supported when truncation would enter the compressed range.");
        _totalLength = length;
    }

    /// <summary>
    /// Reports estimated memory usage in bytes.
    /// </summary>
    public long EstimatedMemoryBytes()
    {
        long fp32Bytes = (long)_numLayers * 2 * _fp32WindowSize * _kvDim * sizeof(float);
        int maxTqPositions = _maxSeqLen - _fp32WindowSize;
        if (maxTqPositions <= 0) return fp32Bytes;

        int maxTiles = (maxTqPositions + _tileTokens - 1) / _tileTokens;
        long keyTileBytes   = (long)_numLayers * maxTiles * _numKvHeads * _keyTileStride;
        long valueTileBytes = (long)_numLayers * maxTiles * _numKvHeads * _valueTileStride;
        long stageBytes     = (long)_numLayers * 2 * _numKvHeads * _stagingBytesPerHead;
        return fp32Bytes + keyTileBytes + valueTileBytes + stageBytes;
    }

    /// <summary>
    /// Compute K-scores (q·k for all TQ-compressed positions) for one
    /// (layer, kv-head). Lloyd-Max: walks complete FastScan tiles via the i8
    /// LUT kernel and the staging tail via per-block <c>DequantDot</c>. KVarN:
    /// walks whole 128-token tiles via <see cref="KVarNCompressor.KeyScores"/>
    /// (no staging tail exists). Final score per position is multiplied by
    /// <paramref name="attnScale"/>.
    /// </summary>
    /// <param name="rotatedQuery">Pre-rotated query (caller invokes
    /// <see cref="RotateQuery"/> once per head).</param>
    /// <param name="attnScale">Attention scale, typically 1/√headDim.</param>
    /// <param name="scoresOut">Output buffer, length ≥ <c>TqLength</c>.</param>
    public void ComputeKScores(
        int layer,
        int kvHead,
        float* rotatedQuery,
        float attnScale,
        float* scoresOut)
    {
        if (_quantizer == TqQuantizer.KVarN)
        {
            ComputeKScoresKVarN(layer, kvHead, rotatedQuery, attnScale, scoresOut);
            return;
        }

        int hd = _headDim;
        int totalTq = _layerTqLengths[layer];
        int numFullTiles = totalTq / SharpInference.TurboQuant.FastScan.TileSize;
        int stagingCount = totalTq % SharpInference.TurboQuant.FastScan.TileSize;

        var centroids = SharpInference.TurboQuant.TurboQuantCodebooks.GetCentroids(_bits, hd);

        // Per-query LUT, stack-allocated for the head dims we ship today
        // (128 and 256 — capped at 4096 bytes here).
        Span<sbyte> lut = stackalloc sbyte[256 * 16];
        lut = lut.Slice(0, hd * 16);

        float lutScale = _bits == 4
            ? SharpInference.TurboQuant.FastScan.BuildLut4Bit(
                new ReadOnlySpan<float>(rotatedQuery, hd), centroids, lut, hd)
            : SharpInference.TurboQuant.FastScan.BuildLut3Bit(
                new ReadOnlySpan<float>(rotatedQuery, hd), centroids, lut, hd);

        fixed (sbyte* lutPtr = lut)
        {
            for (int tileIdx = 0; tileIdx < numFullTiles; tileIdx++)
            {
                byte* tile = KeyTileAt(layer, tileIdx, kvHead);
                SharpInference.TurboQuant.FastScan.KScoreTile4BitAvx2(
                    tile, lutPtr, lutScale, attnScale,
                    scoresOut + (long)tileIdx * SharpInference.TurboQuant.FastScan.TileSize,
                    hd);
            }
        }

        if (stagingCount > 0)
        {
            var keyCompressor = _keyCompressors![layer][kvHead];
            var rotatedQuerySpan = new ReadOnlySpan<float>(rotatedQuery, hd);
            int tailStart = numFullTiles * SharpInference.TurboQuant.FastScan.TileSize;
            for (int s = 0; s < stagingCount; s++)
            {
                byte* block = KeyStagingBlockAt(layer, s, kvHead);
                float dot = keyCompressor.DequantDot(
                    new ReadOnlySpan<byte>(block, _tqBlockSize),
                    rotatedQuerySpan);
                scoresOut[tailStart + s] = dot * attnScale;
            }
        }
    }

    /// <summary>
    /// KVarN K-scoring (issue #180): the compressed region always holds whole
    /// 128-token tiles (the FP32 window owns everything mod 128), each scored
    /// via the fused <see cref="KVarNCompressor.KeyScores"/> which folds the
    /// per-channel Sinkhorn/RTN factors into the query once per tile.
    /// </summary>
    private void ComputeKScoresKVarN(
        int layer,
        int kvHead,
        float* rotatedQuery,
        float attnScale,
        float* scoresOut)
    {
        int tileTokens = KVarNCompressor.TileTokens;
        int numTiles = _layerTqLengths[layer] / tileTokens;
        if (numTiles == 0) return;

        var comp = _kvarn!;
        var rotatedQuerySpan = new ReadOnlySpan<float>(rotatedQuery, _headDim);
        for (int tileIdx = 0; tileIdx < numTiles; tileIdx++)
        {
            var tile = new ReadOnlySpan<byte>(KeyTileAt(layer, tileIdx, kvHead), _keyTileStride);
            var scores = new Span<float>(scoresOut + (long)tileIdx * tileTokens, tileTokens);
            comp.KeyScores(tile, rotatedQuerySpan, scores);
            for (int t = 0; t < tileTokens; t++)
                scores[t] *= attnScale;
        }
    }

    /// <summary>
    /// V-aggregation for one (layer, kv-head): accumulates the weighted sum
    /// <c>Σ_t weights[t] · decompressed_value[t]</c> into <paramref name="outAcc"/>
    /// (length ≥ headDim). The FastScan kernels accumulate in the rotated domain;
    /// the per-head sign-flip and inverse Walsh-Hadamard transform are applied
    /// once at the end (instead of <c>tqLength</c> times in the per-block path),
    /// which is the main structural win of Phase 3.
    /// </summary>
    public void ComputeVAggregation(
        int layer,
        int kvHead,
        float* weights,
        float* outAcc)
    {
        if (_quantizer == TqQuantizer.KVarN)
        {
            ComputeVAggregationKVarN(layer, kvHead, weights, outAcc);
            return;
        }

        int hd = _headDim;
        int totalTq = _layerTqLengths[layer];
        int numFullTiles = totalTq / SharpInference.TurboQuant.FastScan.TileSize;
        int stagingCount = totalTq % SharpInference.TurboQuant.FastScan.TileSize;

        if (totalTq == 0) return;

        var centroids = SharpInference.TurboQuant.TurboQuantCodebooks.GetCentroids(_bits, hd);

        // Rotated-domain accumulator (stack-allocated, capped at hd ≤ 256).
        Span<float> rotatedAcc = stackalloc float[256];
        rotatedAcc = rotatedAcc.Slice(0, hd);
        rotatedAcc.Clear();

        Span<float> effectiveW = stackalloc float[SharpInference.TurboQuant.FastScan.TileSize];
        Span<sbyte> vLut = stackalloc sbyte[SharpInference.TurboQuant.FastScan.TileSize * 16];

        // Phase 1: full V-tiles via FastScan.
        fixed (float* rotAccPtr = rotatedAcc)
        fixed (sbyte* vLutPtr = vLut)
        {
            for (int tileIdx = 0; tileIdx < numFullTiles; tileIdx++)
            {
                byte* tile = ValueTileAt(layer, tileIdx, kvHead);

                // effectiveW[t] = weights[t] · norm[t] from the tile's norm header.
                int baseT = tileIdx * SharpInference.TurboQuant.FastScan.TileSize;
                for (int t = 0; t < SharpInference.TurboQuant.FastScan.TileSize; t++)
                {
                    float norm = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(
                        new ReadOnlySpan<byte>(tile + t * 2, 2));
                    effectiveW[t] = weights[baseT + t] * norm;
                }

                float vScale = _bits == 4
                    ? SharpInference.TurboQuant.FastScan.BuildVLut4Bit(effectiveW, centroids, vLut)
                    : SharpInference.TurboQuant.FastScan.BuildVLut3Bit(effectiveW, centroids, vLut);

                SharpInference.TurboQuant.FastScan.VAggregateTile4BitAvx2(
                    tile, vLutPtr, vScale, rotAccPtr, hd);
            }
        }

        // Phase 2: staging tail — per-block dequant in rotated domain.
        if (stagingCount > 0)
        {
            int tailStart = numFullTiles * SharpInference.TurboQuant.FastScan.TileSize;
            for (int s = 0; s < stagingCount; s++)
            {
                byte* block = ValueStagingBlockAt(layer, s, kvHead);
                float norm = (float)System.Buffers.Binary.BinaryPrimitives.ReadHalfLittleEndian(
                    new ReadOnlySpan<byte>(block, 2));
                float effW = weights[tailStart + s] * norm;

                ReadOnlySpan<byte> packed = new ReadOnlySpan<byte>(block + TurboQuantOps.IndicesOffset, _tqBlockSize - TurboQuantOps.IndicesOffset);
                for (int d = 0; d < hd; d++)
                {
                    int code = _bits == 4
                        ? BitPacking.UnpackBits4(packed, 0, d)
                        : BitPacking.UnpackBits3(packed, 0, d);
                    rotatedAcc[d] += effW * centroids[code];
                }
            }
        }

        // Phase 3: deferred sign-flip + inverse Walsh-Hadamard (once per head),
        // then fold into the caller's accumulator. Both operations are linear,
        // so they commute with the Σ_t accumulation above.
        var signPattern = _valueCompressors![layer][kvHead].SignPattern;
        for (int d = 0; d < hd; d++)
            rotatedAcc[d] *= signPattern[d];

        SharpInference.TurboQuant.WalshHadamard.Transform(rotatedAcc, rotatedAcc, hd);

        for (int d = 0; d < hd; d++)
            outAcc[d] += rotatedAcc[d];
    }

    /// <summary>
    /// KVarN V-aggregation (issue #180): all tiles share the same channel
    /// rotation, so they accumulate into one rotated-domain accumulator and the
    /// inverse Walsh-Hadamard runs ONCE per head
    /// (<see cref="KVarNCompressor.UnrotateOutput"/>) before folding into
    /// <paramref name="outAcc"/>. The caller then adds the FP32-window V
    /// contribution on top of <paramref name="outAcc"/> in the un-rotated
    /// (original) domain — the same deferred-inverse contract the FastScan
    /// path uses.
    /// </summary>
    private void ComputeVAggregationKVarN(
        int layer,
        int kvHead,
        float* weights,
        float* outAcc)
    {
        int hd = _headDim;
        int tileTokens = KVarNCompressor.TileTokens;
        int numTiles = _layerTqLengths[layer] / tileTokens;
        if (numTiles == 0) return;

        var comp = _kvarn!;
        Span<float> rotatedAcc = stackalloc float[hd]; // hd ≤ 1024, validated by KVarNCompressor
        rotatedAcc.Clear();

        for (int tileIdx = 0; tileIdx < numTiles; tileIdx++)
        {
            var tile = new ReadOnlySpan<byte>(ValueTileAt(layer, tileIdx, kvHead), _valueTileStride);
            var w = new ReadOnlySpan<float>(weights + (long)tileIdx * tileTokens, tileTokens);
            comp.AggregateValues(tile, w, rotatedAcc);
        }

        comp.UnrotateOutput(rotatedAcc);
        for (int d = 0; d < hd; d++)
            outAcc[d] += rotatedAcc[d];
    }

    private void CompressOldestFp32(int layer)
    {
        // The oldest FP32 position is at index 0 in the FP32 window
        // (we use simple linear addressing, not a ring buffer, for now)
        int fp32Slot = 0;
        long fp32Offset = (long)fp32Slot * _kvDim;

        int stagingSlot = _layerTqLengths[layer] % SharpInference.TurboQuant.FastScan.TileSize;

        // Compress each KV head independently into the K and V staging slots.
        for (int head = 0; head < _numKvHeads; head++)
        {
            int headOffset = head * _headDim;
            var keySpan = new ReadOnlySpan<float>(_fp32Keys[layer] + fp32Offset + headOffset, _headDim);
            var valSpan = new ReadOnlySpan<float>(_fp32Values[layer] + fp32Offset + headOffset, _headDim);

            long stagingOffset = (long)head * _stagingBytesPerHead + (long)stagingSlot * _tqBlockSize;
            var keyDest = new Span<byte>(_tqKeyStaging[layer]   + stagingOffset, _tqBlockSize);
            var valDest = new Span<byte>(_tqValueStaging[layer] + stagingOffset, _tqBlockSize);

            _keyCompressors![layer][head].Compress(keySpan, keyDest);
            _valueCompressors![layer][head].Compress(valSpan, valDest);
        }

        _layerTqLengths[layer]++;

        // If the staging buffer just filled, pack both K-tile and V-tile.
        // Per-head staging is contiguous, so the packers read it directly.
        if (_layerTqLengths[layer] % SharpInference.TurboQuant.FastScan.TileSize == 0)
        {
            int tileIdx = _layerTqLengths[layer] / SharpInference.TurboQuant.FastScan.TileSize - 1;
            for (int head = 0; head < _numKvHeads; head++)
            {
                var keyStageSpan = new ReadOnlySpan<byte>(
                    _tqKeyStaging[layer] + (long)head * _stagingBytesPerHead,
                    _stagingBytesPerHead);
                var valStageSpan = new ReadOnlySpan<byte>(
                    _tqValueStaging[layer] + (long)head * _stagingBytesPerHead,
                    _stagingBytesPerHead);
                var keyTileSpan = new Span<byte>(KeyTileAt(layer, tileIdx, head),   _tileBytes);
                var valTileSpan = new Span<byte>(ValueTileAt(layer, tileIdx, head), _vTileBytes);

                if (_bits == 4)
                {
                    SharpInference.TurboQuant.FastScan.PackTile4Bit(keyStageSpan, keyTileSpan, _headDim);
                    SharpInference.TurboQuant.FastScan.PackVTile4Bit(valStageSpan, valTileSpan, _headDim);
                }
                else
                {
                    SharpInference.TurboQuant.FastScan.PackTile3Bit(keyStageSpan, keyTileSpan, _headDim);
                    SharpInference.TurboQuant.FastScan.PackVTile3Bit(valStageSpan, valTileSpan, _headDim);
                }
            }
        }

        // Shift FP32 window: move remaining entries down by 1
        int fp32Count = _totalLength - (_layerTqLengths[layer] - 1);
        if (fp32Count > 1)
        {
            long copyBytes = (long)(fp32Count - 1) * _kvDim * sizeof(float);
            Buffer.MemoryCopy(_fp32Keys[layer] + _kvDim, _fp32Keys[layer], copyBytes, copyBytes);
            Buffer.MemoryCopy(_fp32Values[layer] + _kvDim, _fp32Values[layer], copyBytes, copyBytes);
        }
    }

    /// <summary>
    /// KVarN promotion (issue #180): compress the oldest
    /// <see cref="KVarNCompressor.TileTokens"/> FP32-window positions into one
    /// K tile + one V tile per kv-head, then shift the window down by a whole
    /// tile. Unlike the Lloyd-Max path there is no per-position staging — the
    /// dual-axis Sinkhorn normalization needs the entire tile assembled before
    /// it can quantize, so promotion is all-or-nothing per tile and the FP32
    /// window always owns the trailing <c>totalLength mod 128</c> positions.
    /// The constructor guarantees <c>fp32WindowSize >= TileTokens</c> whenever
    /// a compressed region can exist, so a full window always contains a tile.
    /// </summary>
    private void CompressOldestFp32TileKVarN(int layer)
    {
        int tileTokens = KVarNCompressor.TileTokens;
        var comp = _kvarn!;
        int tileIdx = _layerTqLengths[layer] / tileTokens;
        int fp32Count = _totalLength - _layerTqLengths[layer];
        Span<float> gather = _kvarnGather.AsSpan();

        // Per head: gather the oldest 128 window rows into a contiguous
        // [token, channel] tile and compress. Write path is single-threaded
        // (see the _kvarn field note), so the shared gather scratch is safe.
        for (int head = 0; head < _numKvHeads; head++)
        {
            int headOffset = head * _headDim;

            for (int t = 0; t < tileTokens; t++)
                new ReadOnlySpan<float>(_fp32Keys[layer] + (long)t * _kvDim + headOffset, _headDim)
                    .CopyTo(gather.Slice(t * _headDim, _headDim));
            comp.CompressKeyTile(gather, new Span<byte>(KeyTileAt(layer, tileIdx, head), _keyTileStride));

            for (int t = 0; t < tileTokens; t++)
                new ReadOnlySpan<float>(_fp32Values[layer] + (long)t * _kvDim + headOffset, _headDim)
                    .CopyTo(gather.Slice(t * _headDim, _headDim));
            comp.CompressValueTile(gather, new Span<byte>(ValueTileAt(layer, tileIdx, head), _valueTileStride));
        }

        _layerTqLengths[layer] += tileTokens;

        // Shift the remaining FP32 entries down by a whole tile.
        int remaining = fp32Count - tileTokens;
        if (remaining > 0)
        {
            long copyBytes = (long)remaining * _kvDim * sizeof(float);
            Buffer.MemoryCopy(_fp32Keys[layer] + (long)tileTokens * _kvDim, _fp32Keys[layer], copyBytes, copyBytes);
            Buffer.MemoryCopy(_fp32Values[layer] + (long)tileTokens * _kvDim, _fp32Values[layer], copyBytes, copyBytes);
        }
    }

    /// <summary>
    /// SnapKV (issue #60) compaction for the TurboQuant cache: keep only the K/V
    /// positions listed in <paramref name="keep"/> (sorted ascending, all in
    /// <c>[0, N)</c>); discard the rest. After compaction the cache holds
    /// <c>keep.Length</c> entries in slots <c>[0, keep.Length)</c>, with the
    /// TQ region holding the older survivors and the FP32 ring holding the
    /// most-recent ones (matching the steady-state hybrid layout).
    /// </summary>
    /// <remarks>
    /// Phase-1 invariant: kept positions originally in the TQ region stay TQ;
    /// kept positions originally in the FP32 region stay FP32 unless the kept
    /// FP32 count would exceed <see cref="Fp32WindowSize"/>, in which case the
    /// oldest excess survivors are re-quantized into the TQ region (matching
    /// what would have happened if they'd aged out via the normal Append path).
    ///
    /// Implementation: per-layer, allocate fresh tile + staging + FP32 buffers,
    /// write the kept survivors into them in order, then atomically swap the
    /// underlying native pointers. Doing this in-place is doable but error-prone
    /// because tiles are dim-major over 32 positions — any partial overwrite
    /// would corrupt neighbours.
    /// </remarks>
    public void Compact(int[] keep, int N)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(TurboQuantKvCache));
        if (_quantizer == TqQuantizer.KVarN)
            throw new NotSupportedException(
                "SnapKV compaction is not yet supported with the KVarN quantizer (issue #180 follow-up): " +
                "re-quantizing survivors requires re-assembling whole 128-token tiles. " +
                "Disable SnapKV (unset SHARPI_SNAPKV_BUDGET) or use the Lloyd-Max quantizer.");
        if (keep is null) throw new ArgumentNullException(nameof(keep));
        if (N != _totalLength)
            throw new ArgumentException(
                $"Compact: N={N} does not match _totalLength={_totalLength}.", nameof(N));
        int K = keep.Length;
        if (K > N)
            throw new ArgumentException(
                $"Compact: keep count {K} exceeds N={N}.", nameof(keep));
        int prev = -1;
        for (int i = 0; i < K; i++)
        {
            int p = keep[i];
            if (p < 0 || p >= N)
                throw new ArgumentOutOfRangeException(nameof(keep),
                    $"keep[{i}]={p} is outside [0,{N}).");
            if (p <= prev)
                throw new ArgumentException(
                    $"keep must be strictly increasing; got {prev} then {p} at index {i}.", nameof(keep));
            prev = p;
        }

        if (K == N)
        {
            // No-op compaction.
            return;
        }

        int tileSize = SharpInference.TurboQuant.FastScan.TileSize;
        int maxTqPositions = _maxSeqLen - _fp32WindowSize;
        if (maxTqPositions < 0) maxTqPositions = 0;
        int maxTiles = (maxTqPositions + tileSize - 1) / tileSize;
        var fp32Bytes = (nuint)((long)_fp32WindowSize * _kvDim * sizeof(float));
        var keyTileBytes = maxTqPositions > 0
            ? (nuint)((long)maxTiles * _numKvHeads * _tileBytes) : (nuint)0;
        var valueTileBytes = maxTqPositions > 0
            ? (nuint)((long)maxTiles * _numKvHeads * _vTileBytes) : (nuint)0;
        var stageBytes = maxTqPositions > 0
            ? (nuint)((long)_numKvHeads * _stagingBytesPerHead) : (nuint)0;

        // Decompress/repack scratch is sized by constants (_headDim, _tqBlockSize),
        // so we lift the allocations out of the per-layer loop.
        float[] decompKeyArr = new float[_headDim];
        float[] decompValArr = new float[_headDim];
        byte[]  unpackBlockArr = new byte[_tqBlockSize];

        // Per-layer compaction. Each layer's TQ length tracks how far its
        // staging+tile chain has filled, so each layer's split is determined
        // independently from the same shared `keep` set.
        for (int layer = 0; layer < _numLayers; layer++)
        {
            int oldTqLen = _layerTqLengths[layer];

            // Partition the keep set into TQ-source and FP32-source survivors.
            // Phase-1 invariant: preserve original storage class. The FP32
            // overflow → TQ-promotion branch below is defense-in-depth: today's
            // Append (line 157) enforces fp32Count ≤ _fp32WindowSize at all
            // times, so kept-FP32 count after a sorted Compact cannot exceed
            // the window. Kept as a guard against future invariant relaxation
            // (e.g. bulk-load constructors); intentionally unreachable in the
            // current code path. The same overflow logic is exercised in the
            // CUDA/Vulkan phases (#69/#70) where async append paths may stage
            // multiple positions before draining.
            int tqSurvivors = 0;
            for (int i = 0; i < K; i++) if (keep[i] < oldTqLen) tqSurvivors++;
            int fp32Survivors = K - tqSurvivors;

            int newTqLen = tqSurvivors;
            int newFp32Len = fp32Survivors;
            if (newFp32Len > _fp32WindowSize)
            {
                int excess = newFp32Len - _fp32WindowSize;
                newTqLen += excess;     // oldest FP32 survivors re-quantized
                newFp32Len -= excess;
            }

            int newTotal = newTqLen + newFp32Len;
            if (newTotal != K)
                throw new InvalidOperationException(
                    $"Compact: internal accounting bug, newTqLen+newFp32Len={newTqLen + newFp32Len} != K={K}.");
            if (newTqLen > maxTqPositions)
                throw new NotSupportedException(
                    $"Compact: layer {layer} would have newTqLen={newTqLen} but the cache " +
                    $"only has room for {maxTqPositions} TQ positions. The keep set is too " +
                    "long for this cache configuration — increase maxSeqLen or shrink the keep set.");

            // Fresh destination buffers; populate then swap.
            float* newFp32Keys   = (float*)NativeMemory.AllocZeroed(fp32Bytes);
            float* newFp32Values = (float*)NativeMemory.AllocZeroed(fp32Bytes);
            byte*  newKeyTiles     = maxTqPositions > 0 ? (byte*)NativeMemory.AllocZeroed(keyTileBytes)   : null;
            byte*  newValueTiles   = maxTqPositions > 0 ? (byte*)NativeMemory.AllocZeroed(valueTileBytes) : null;
            byte*  newKeyStaging   = maxTqPositions > 0 ? (byte*)NativeMemory.AllocZeroed(stageBytes)     : null;
            byte*  newValueStaging = maxTqPositions > 0 ? (byte*)NativeMemory.AllocZeroed(stageBytes)     : null;

            try
            {
                // Per-head decompress + unpack scratch (decompKeyArr / decompValArr
                // / unpackBlockArr) are sized by constants and lifted to the
                // pre-loop scope above; reused across every layer.

                // 32 per-head staging blocks at a time — when we accumulate
                // tileSize blocks worth for a head, pack into a K- and V-tile.
                // Each layer's `_tqBlockSize` is the source block stride.
                // We funnel writes through these per-head head-major buffers so
                // we can pack them once per completed tile rather than re-packing
                // tiles mid-stream.
                byte* tmpKeyStage   = maxTqPositions > 0
                    ? (byte*)NativeMemory.AllocZeroed((nuint)((long)_numKvHeads * _stagingBytesPerHead)) : null;
                byte* tmpValueStage = maxTqPositions > 0
                    ? (byte*)NativeMemory.AllocZeroed((nuint)((long)_numKvHeads * _stagingBytesPerHead)) : null;
                try
                {
                    Span<float> decompKey = decompKeyArr;
                    Span<float> decompVal = decompValArr;
                    Span<byte>  unpackBlock = unpackBlockArr;
                    // Walk the keep set in order. The newTqLen oldest survivors
                    // land in TQ, the newFp32Len most-recent land in FP32. With
                    // `keep` strictly increasing this maps to a clean prefix /
                    // suffix split: keep[0..newTqLen) → TQ slot i, keep[newTqLen..K)
                    // → FP32 slot (i - newTqLen).
                    for (int i = 0; i < K; i++)
                    {
                        int srcPos = keep[i];
                        bool wasTq = srcPos < oldTqLen;

                        if (i < newTqLen)
                        {
                            // Destination is TQ. Source is either TQ (decompress
                            // then re-quantize — generally lossless modulo a
                            // single dequant→requant round-trip on the same
                            // codebook) or FP32 (quantize directly).
                            for (int head = 0; head < _numKvHeads; head++)
                            {
                                int headOffset = head * _headDim;
                                ReadOnlySpan<float> keySrc;
                                ReadOnlySpan<float> valSrc;

                                if (wasTq)
                                {
                                    // Decompress original TQ block (tile or staging)
                                    // into scratch, then quantize into the new staging slot.
                                    DecompressKeyTqAt(layer, head, srcPos, decompKey);
                                    DecompressValueTqAt(layer, head, srcPos, decompVal);
                                    keySrc = decompKey;
                                    valSrc = decompVal;
                                }
                                else
                                {
                                    int fp32Idx = srcPos - oldTqLen;
                                    long off = (long)fp32Idx * _kvDim + headOffset;
                                    keySrc = new ReadOnlySpan<float>(_fp32Keys[layer]   + off, _headDim);
                                    valSrc = new ReadOnlySpan<float>(_fp32Values[layer] + off, _headDim);
                                }

                                int stageSlot = i % tileSize;
                                long stageOff = (long)head * _stagingBytesPerHead + (long)stageSlot * _tqBlockSize;
                                var keyDst = new Span<byte>(tmpKeyStage   + stageOff, _tqBlockSize);
                                var valDst = new Span<byte>(tmpValueStage + stageOff, _tqBlockSize);

                                _keyCompressors![layer][head].Compress(keySrc, keyDst);
                                _valueCompressors![layer][head].Compress(valSrc, valDst);
                            }

                            // If we just completed a tile (slot 31 of tileSize),
                            // pack the per-head staging into the destination
                            // K- and V-tiles. After packing we can reuse the
                            // temp staging slots — the AllocZeroed in the next
                            // iteration's Compress overwrites the bytes.
                            if ((i + 1) % tileSize == 0)
                            {
                                int tileIdx = i / tileSize;
                                for (int head = 0; head < _numKvHeads; head++)
                                {
                                    var keyStageSpan = new ReadOnlySpan<byte>(
                                        tmpKeyStage + (long)head * _stagingBytesPerHead,
                                        _stagingBytesPerHead);
                                    var valStageSpan = new ReadOnlySpan<byte>(
                                        tmpValueStage + (long)head * _stagingBytesPerHead,
                                        _stagingBytesPerHead);
                                    var keyTileSpan = new Span<byte>(
                                        newKeyTiles + ((long)tileIdx * _numKvHeads + head) * _tileBytes,
                                        _tileBytes);
                                    var valTileSpan = new Span<byte>(
                                        newValueTiles + ((long)tileIdx * _numKvHeads + head) * _vTileBytes,
                                        _vTileBytes);
                                    if (_bits == 4)
                                    {
                                        SharpInference.TurboQuant.FastScan.PackTile4Bit(keyStageSpan, keyTileSpan, _headDim);
                                        SharpInference.TurboQuant.FastScan.PackVTile4Bit(valStageSpan, valTileSpan, _headDim);
                                    }
                                    else
                                    {
                                        SharpInference.TurboQuant.FastScan.PackTile3Bit(keyStageSpan, keyTileSpan, _headDim);
                                        SharpInference.TurboQuant.FastScan.PackVTile3Bit(valStageSpan, valTileSpan, _headDim);
                                    }
                                }
                            }
                        }
                        else
                        {
                            // Destination is FP32. Source is FP32 or TQ.
                            int fp32SlotDst = i - newTqLen;
                            long dstOff = (long)fp32SlotDst * _kvDim;

                            if (wasTq)
                            {
                                for (int head = 0; head < _numKvHeads; head++)
                                {
                                    int headOffset = head * _headDim;
                                    DecompressKeyTqAt(layer, head, srcPos, decompKey);
                                    DecompressValueTqAt(layer, head, srcPos, decompVal);
                                    decompKey.CopyTo(new Span<float>(newFp32Keys   + dstOff + headOffset, _headDim));
                                    decompVal.CopyTo(new Span<float>(newFp32Values + dstOff + headOffset, _headDim));
                                }
                            }
                            else
                            {
                                int fp32IdxSrc = srcPos - oldTqLen;
                                long srcOff = (long)fp32IdxSrc * _kvDim;
                                new ReadOnlySpan<float>(_fp32Keys[layer]   + srcOff, _kvDim)
                                    .CopyTo(new Span<float>(newFp32Keys   + dstOff, _kvDim));
                                new ReadOnlySpan<float>(_fp32Values[layer] + srcOff, _kvDim)
                                    .CopyTo(new Span<float>(newFp32Values + dstOff, _kvDim));
                            }
                        }
                    }

                    // If the TQ tail didn't fill a full tile, leave its bytes
                    // in the destination staging buffer for the decode-time
                    // FastScan staging path to consume. (Mirrors the in-flight
                    // staging contract used by CompressOldestFp32.)
                    int tailCount = newTqLen % tileSize;
                    if (tailCount > 0 && maxTqPositions > 0)
                    {
                        int tailStart = newTqLen - tailCount;
                        for (int head = 0; head < _numKvHeads; head++)
                        {
                            for (int s = 0; s < tailCount; s++)
                            {
                                int srcSlot = (tailStart + s) % tileSize;
                                long srcStageOff = (long)head * _stagingBytesPerHead + (long)srcSlot * _tqBlockSize;
                                long dstStageOff = (long)head * _stagingBytesPerHead + (long)s        * _tqBlockSize;
                                Buffer.MemoryCopy(
                                    tmpKeyStage   + srcStageOff, newKeyStaging   + dstStageOff,
                                    _tqBlockSize, _tqBlockSize);
                                Buffer.MemoryCopy(
                                    tmpValueStage + srcStageOff, newValueStaging + dstStageOff,
                                    _tqBlockSize, _tqBlockSize);
                            }
                        }
                    }
                }
                finally
                {
                    if (tmpKeyStage   != null) NativeMemory.Free(tmpKeyStage);
                    if (tmpValueStage != null) NativeMemory.Free(tmpValueStage);
                }

                // Atomic swap: free the old layer buffers, install the new ones.
                NativeMemory.Free(_fp32Keys[layer]);
                NativeMemory.Free(_fp32Values[layer]);
                _fp32Keys[layer]   = newFp32Keys;
                _fp32Values[layer] = newFp32Values;
                if (_tqKeyTiles[layer]     != null) NativeMemory.Free(_tqKeyTiles[layer]);
                if (_tqValueTiles[layer]   != null) NativeMemory.Free(_tqValueTiles[layer]);
                if (_tqKeyStaging[layer]   != null) NativeMemory.Free(_tqKeyStaging[layer]);
                if (_tqValueStaging[layer] != null) NativeMemory.Free(_tqValueStaging[layer]);
                _tqKeyTiles[layer]     = newKeyTiles;
                _tqValueTiles[layer]   = newValueTiles;
                _tqKeyStaging[layer]   = newKeyStaging;
                _tqValueStaging[layer] = newValueStaging;
                _layerTqLengths[layer] = newTqLen;
            }
            catch
            {
                // Roll back any successfully-allocated destinations for this
                // layer before propagating so we don't leak on a mid-loop throw.
                NativeMemory.Free(newFp32Keys);
                NativeMemory.Free(newFp32Values);
                if (newKeyTiles     != null) NativeMemory.Free(newKeyTiles);
                if (newValueTiles   != null) NativeMemory.Free(newValueTiles);
                if (newKeyStaging   != null) NativeMemory.Free(newKeyStaging);
                if (newValueStaging != null) NativeMemory.Free(newValueStaging);
                throw;
            }
        }

        _totalLength = K;
    }

    /// <summary>
    /// Decompress one (layer, kvHead, position) TQ K block into <paramref name="dst"/>.
    /// Handles both tile-resident and staging-resident positions transparently.
    /// </summary>
    private void DecompressKeyTqAt(int layer, int kvHead, int position, Span<float> dst)
    {
        int tileSize = SharpInference.TurboQuant.FastScan.TileSize;
        int numFullTiles = _layerTqLengths[layer] / tileSize;
        var compressor = _keyCompressors![layer][kvHead];

        if (position < numFullTiles * tileSize)
        {
            int tileIdx = position / tileSize;
            int slot = position % tileSize;
            // Unpack one (tile, slot, kvHead) → per-block format, then dequant.
            Span<byte> block = stackalloc byte[_tqBlockSize];
            UnpackTileSlotKey(layer, tileIdx, kvHead, slot, block);
            compressor.Decompress(block, dst);
        }
        else
        {
            // Staging-resident.
            int stagingIdx = position - numFullTiles * tileSize;
            byte* blockPtr = KeyStagingBlockAt(layer, stagingIdx, kvHead);
            compressor.Decompress(new ReadOnlySpan<byte>(blockPtr, _tqBlockSize), dst);
        }
    }

    /// <summary>V-cache twin of <see cref="DecompressKeyTqAt"/>.</summary>
    private void DecompressValueTqAt(int layer, int kvHead, int position, Span<float> dst)
    {
        int tileSize = SharpInference.TurboQuant.FastScan.TileSize;
        int numFullTiles = _layerTqLengths[layer] / tileSize;
        var compressor = _valueCompressors![layer][kvHead];

        if (position < numFullTiles * tileSize)
        {
            int tileIdx = position / tileSize;
            int slot = position % tileSize;
            Span<byte> block = stackalloc byte[_tqBlockSize];
            UnpackTileSlotValue(layer, tileIdx, kvHead, slot, block);
            compressor.Decompress(block, dst);
        }
        else
        {
            int stagingIdx = position - numFullTiles * tileSize;
            byte* blockPtr = ValueStagingBlockAt(layer, stagingIdx, kvHead);
            compressor.Decompress(new ReadOnlySpan<byte>(blockPtr, _tqBlockSize), dst);
        }
    }

    /// <summary>
    /// Reverse the K-tile pack: extract one position's per-block compressed
    /// representation from a FastScan K-tile. K-tile codes are dim-major with
    /// each per-dim row holding 32 nibbles (16 bytes); per the packer, byte b
    /// holds the code for position b (low nibble) and position 16+b (high nibble).
    /// </summary>
    private void UnpackTileSlotKey(int layer, int tileIdx, int kvHead, int slot, Span<byte> blockOut)
    {
        if (blockOut.Length < _tqBlockSize)
            throw new ArgumentException($"blockOut must be at least {_tqBlockSize} bytes.", nameof(blockOut));

        blockOut.Clear();
        byte* tile = KeyTileAt(layer, tileIdx, kvHead);

        // Copy the position's FP16 norm header.
        blockOut[0] = tile[slot * 2];
        blockOut[1] = tile[slot * 2 + 1];

        // Walk every dim; pull this position's nibble from the tile-byte
        // (slot & 15), high-or-low determined by slot >= 16. Write into the
        // per-block packed layout via the bit-packing helper.
        byte* codes = tile + SharpInference.TurboQuant.FastScan.NormBytesPerTile;
        int byteInRow = slot & 15;
        bool highNibble = slot >= 16;

        for (int d = 0; d < _headDim; d++)
        {
            byte pair = codes[d * SharpInference.TurboQuant.FastScan.CodeBytesPerDim + byteInRow];
            int code = highNibble ? (pair >> 4) & 0x0F : pair & 0x0F;
            if (_bits == 4)
                BitPacking.PackBits4(blockOut, TurboQuantOps.IndicesOffset, d, code);
            else
                BitPacking.PackBits3(blockOut, TurboQuantOps.IndicesOffset, d, code & 0x07);
        }
    }

    /// <summary>
    /// Reverse the V-tile pack: extract one position's per-block compressed
    /// representation from a FastScan V-tile. V-tile codes are position-major
    /// with each per-position row holding dim/2 bytes (one byte per dim pair,
    /// low nibble = even d, high nibble = odd d).
    /// </summary>
    private void UnpackTileSlotValue(int layer, int tileIdx, int kvHead, int slot, Span<byte> blockOut)
    {
        if (blockOut.Length < _tqBlockSize)
            throw new ArgumentException($"blockOut must be at least {_tqBlockSize} bytes.", nameof(blockOut));

        blockOut.Clear();
        byte* tile = ValueTileAt(layer, tileIdx, kvHead);

        blockOut[0] = tile[slot * 2];
        blockOut[1] = tile[slot * 2 + 1];

        byte* codes = tile + SharpInference.TurboQuant.FastScan.NormBytesPerTile;
        int packedPerBlock = _headDim / 2;
        byte* rowPtr = codes + (long)slot * packedPerBlock;

        for (int d = 0; d < _headDim; d += 2)
        {
            byte pair = rowPtr[d / 2];
            int low  = pair & 0x0F;
            int high = (pair >> 4) & 0x0F;
            if (_bits == 4)
            {
                BitPacking.PackBits4(blockOut, TurboQuantOps.IndicesOffset, d,     low);
                BitPacking.PackBits4(blockOut, TurboQuantOps.IndicesOffset, d + 1, high);
            }
            else
            {
                BitPacking.PackBits3(blockOut, TurboQuantOps.IndicesOffset, d,     low  & 0x07);
                BitPacking.PackBits3(blockOut, TurboQuantOps.IndicesOffset, d + 1, high & 0x07);
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            NativeMemory.Free(_fp32Keys[i]);
            NativeMemory.Free(_fp32Values[i]);
            _fp32Keys[i] = null;
            _fp32Values[i] = null;
            if (_tqKeyTiles[i]     != null) { NativeMemory.Free(_tqKeyTiles[i]);     _tqKeyTiles[i]     = null; }
            if (_tqKeyStaging[i]   != null) { NativeMemory.Free(_tqKeyStaging[i]);   _tqKeyStaging[i]   = null; }
            if (_tqValueTiles[i]   != null) { NativeMemory.Free(_tqValueTiles[i]);   _tqValueTiles[i]   = null; }
            if (_tqValueStaging[i] != null) { NativeMemory.Free(_tqValueStaging[i]); _tqValueStaging[i] = null; }
        }
    }
}
