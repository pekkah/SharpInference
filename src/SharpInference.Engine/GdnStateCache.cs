using System.Runtime.InteropServices;

using SharpInference.Core;

namespace SharpInference.Engine;

/// <summary>
/// Per-sequence state cache for the Gated DeltaNet (GDN) recurrent blocks of a
/// hybrid model (e.g. qwen35moe).
///
/// Memory model:
/// - Eagerly allocated: one contiguous native block per GDN layer for the conv1d
///   state and one for the recurrent matrix state. <see cref="Reset"/> zeroes the
///   buffers in place; it does not free them.
/// - State here is destructive (rank-1 in-place updates of the per-head matrix),
///   so partial rewind via <see cref="TruncateTo"/> is not supported — only the
///   degenerate cases <c>length == 0</c> (alias for <see cref="Reset"/>) and
///   <c>length == Length</c> (no-op, used by ContinuousBatchingEngine).
///
/// Per-layer layouts (row-major):
///   ConvState[gdnLayer] : [ConvKernel - 1, ConvChannels]
///                         (most recent ConvKernel-1 tokens of the joint QKV stream)
///   ScanState[gdnLayer] : [NumVHeads, HeadDim, HeadDim]
///                         (per-head [out_dim, in_dim] matrix S_h, contiguous per head)
///
/// Realistic qwen35moe footprint per sequence (NumKHeads=16, NumVHeads=32, HeadDim=128,
/// ConvKernel=4, InnerSize=4096, ConvChannels=8192, 30 GDN layers):
///   ConvState : 30 × 3 × 8192 × 4 B  = 2,949,120 B ≈ 2.8 MiB
///   ScanState : 30 × 32 × 128 × 128 × 4 B = 62,914,560 B = 60 MiB
///   Total                              ≈ 62.8 MiB
/// (Unlike the KV cache, this does not grow with position.)
/// </summary>
public sealed unsafe class GdnStateCache : IDisposable
{
    // Per-layer dims captured at construction.
    private readonly int _convStateFloatsPerLayer;  // (ConvKernel - 1) * ConvChannels
    private readonly int _scanStateFloatsPerLayer;  // NumVHeads * HeadDim * HeadDim
    private readonly nuint _convStateBytesPerLayer;
    private readonly nuint _scanStateBytesPerLayer;

    // One native block per GDN layer (eager allocation).
    private float*[] _convState;
    private float*[] _scanState;

    // Trunk-layer-index → dense GDN-layer-index, or -1 for full-attention layers.
    // Length equals the total trunk layer count passed at construction.
    private readonly int[] _gdnLayerOf;

    private int _length;
    private bool _disposed;

    /// <summary>
    /// Construct an eager-allocated cache for a single sequence.
    /// </summary>
    /// <param name="layerTypes">Per-trunk-layer block type. Length equals the total trunk
    /// layer count of the model. Only entries equal to <see cref="LayerType.GatedDeltaNet"/>
    /// receive cache slots; <see cref="LayerType.Attention"/> entries get
    /// <see cref="GdnLayerOf"/> = -1.</param>
    /// <param name="gdn">GDN configuration from the model hyperparams.</param>
    public GdnStateCache(IReadOnlyList<LayerType> layerTypes, GdnConfig gdn)
    {
        ArgumentNullException.ThrowIfNull(layerTypes);
        ArgumentNullException.ThrowIfNull(gdn);

        if (gdn.ConvKernel < 1)
            throw new ArgumentOutOfRangeException(
                nameof(gdn), gdn.ConvKernel, "GdnConfig.ConvKernel must be >= 1.");
        if (gdn.NumVHeads <= 0 || gdn.HeadDim <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(gdn), "GdnConfig.NumVHeads and HeadDim must be positive.");
        if (gdn.ConvChannels <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(gdn), "GdnConfig.ConvChannels must be positive.");

        // Build the trunk-layer → GDN-layer-index mapping.
        _gdnLayerOf = new int[layerTypes.Count];
        int next = 0;
        for (int i = 0; i < layerTypes.Count; i++)
        {
            if (layerTypes[i] == LayerType.GatedDeltaNet)
                _gdnLayerOf[i] = next++;
            else
                _gdnLayerOf[i] = -1;
        }
        NumGdnLayers = next;

        // Per-layer state shapes.
        // Conv state holds the most recent (ConvKernel - 1) tokens of the joint QKV
        // stream, so it is empty when ConvKernel == 1. Guard the multiplication to
        // avoid pathologic zero-size allocations.
        int convRows = Math.Max(0, gdn.ConvKernel - 1);
        _convStateFloatsPerLayer = convRows * gdn.ConvChannels;
        _scanStateFloatsPerLayer = gdn.NumVHeads * gdn.HeadDim * gdn.HeadDim;

        _convStateBytesPerLayer = (nuint)((long)_convStateFloatsPerLayer * sizeof(float));
        _scanStateBytesPerLayer = (nuint)((long)_scanStateFloatsPerLayer * sizeof(float));

        _convState = new float*[NumGdnLayers];
        _scanState = new float*[NumGdnLayers];
        for (int g = 0; g < NumGdnLayers; g++)
        {
            // AllocZeroed → state starts at zero (the recurrence's natural initial value).
            // A zero conv state means the "prior tokens" are all-zero pads, matching the
            // causal-conv convention for the start of a sequence.
            _convState[g] = _convStateBytesPerLayer > 0
                ? (float*)NativeMemory.AllocZeroed(_convStateBytesPerLayer)
                : null;
            _scanState[g] = _scanStateBytesPerLayer > 0
                ? (float*)NativeMemory.AllocZeroed(_scanStateBytesPerLayer)
                : null;
        }
    }

    /// <summary>Number of GDN layers (dense, excluding attention layers).</summary>
    public int NumGdnLayers { get; }

    /// <summary>Number of tokens consumed by the recurrence so far for this sequence.</summary>
    public int Length => _length;

    /// <summary>
    /// Total native byte footprint of this cache: <c>NumGdnLayers × (convBytes + scanBytes)</c>.
    /// Independent of <see cref="Length"/>.
    /// </summary>
    public long TotalBytes =>
        (long)NumGdnLayers * ((long)_convStateBytesPerLayer + (long)_scanStateBytesPerLayer);

    /// <summary>Floats per layer in the conv1d state buffer: <c>(ConvKernel - 1) * ConvChannels</c>.</summary>
    public int ConvStateFloatsPerLayer => _convStateFloatsPerLayer;

    /// <summary>Floats per layer in the recurrent state buffer: <c>NumVHeads * HeadDim * HeadDim</c>.</summary>
    public int ScanStateFloatsPerLayer => _scanStateFloatsPerLayer;

    /// <summary>
    /// Pointer to the conv1d state for the given GDN-layer index.
    /// Layout: <c>[ConvKernel - 1, ConvChannels]</c> row-major — the most recent
    /// <c>ConvKernel - 1</c> tokens of the joint QKV stream that fed the depthwise conv.
    /// Returns <c>null</c> only in the degenerate case <c>ConvKernel == 1</c>.
    /// </summary>
    public float* ConvStateAt(int gdnLayerIndex)
    {
        if ((uint)gdnLayerIndex >= (uint)NumGdnLayers)
            throw new ArgumentOutOfRangeException(
                nameof(gdnLayerIndex), gdnLayerIndex,
                $"GDN layer index out of range [0, {NumGdnLayers}).");
        return _convState[gdnLayerIndex];
    }

    /// <summary>
    /// Pointer to the recurrent matrix state for the given GDN-layer index.
    /// Layout: <c>[NumVHeads, HeadDim, HeadDim]</c> row-major — per-head
    /// <c>[out_dim, in_dim]</c> matrix S_h, contiguous per head.
    /// </summary>
    public float* ScanStateAt(int gdnLayerIndex)
    {
        if ((uint)gdnLayerIndex >= (uint)NumGdnLayers)
            throw new ArgumentOutOfRangeException(
                nameof(gdnLayerIndex), gdnLayerIndex,
                $"GDN layer index out of range [0, {NumGdnLayers}).");
        return _scanState[gdnLayerIndex];
    }

    /// <summary>
    /// Map an absolute trunk-layer index to a GDN-layer index, or <c>-1</c> when the
    /// trunk layer is a full-attention block (no GDN state).
    /// </summary>
    public int GdnLayerOf(int trunkLayerIndex)
    {
        if ((uint)trunkLayerIndex >= (uint)_gdnLayerOf.Length)
            throw new ArgumentOutOfRangeException(
                nameof(trunkLayerIndex), trunkLayerIndex,
                $"Trunk layer index out of range [0, {_gdnLayerOf.Length}).");
        return _gdnLayerOf[trunkLayerIndex];
    }

    /// <summary>
    /// Zero all state buffers and reset <see cref="Length"/> to 0. Returned memory
    /// stays allocated for reuse on the next request.
    /// </summary>
    public void Reset()
    {
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_convState[g] != null && _convStateBytesPerLayer > 0)
                NativeMemory.Clear(_convState[g], _convStateBytesPerLayer);
            if (_scanState[g] != null && _scanStateBytesPerLayer > 0)
                NativeMemory.Clear(_scanState[g], _scanStateBytesPerLayer);
        }
        _length = 0;
    }

    /// <summary>
    /// Soft truncate to <paramref name="length"/>. For GDN, the recurrence is destructive
    /// (in-place rank-1 updates of the matrix state), so partial rewind is unsupported.
    /// Only the degenerate cases <c>length == 0</c> (alias for <see cref="Reset"/>) and
    /// <c>length == <see cref="Length"/></c> (no-op, used by ContinuousBatchingEngine to
    /// retire a request whose sequence is already at its terminal position) are accepted.
    /// </summary>
    public void TruncateTo(int length)
    {
        if (length == _length) return;          // no-op, matches PagedKvCache convention
        if (length == 0) { Reset(); return; }   // alias for Reset
        throw new InvalidOperationException(
            $"GdnStateCache.TruncateTo({length}): Gated DeltaNet recurrent state is " +
            "destructively updated and cannot be rewound; only length == 0 (Reset) or " +
            $"length == Length ({_length}) is valid. Speculative decoding rewind is not " +
            "supported for hybrid GDN models.");
    }

    /// <summary>
    /// Increment <see cref="Length"/> by 1. Call once per token after all GDN layers
    /// have updated their per-layer state for that token.
    /// </summary>
    public void IncrementPosition() => _length++;

    /// <summary>
    /// Total native byte footprint of a snapshot of this cache: a 4-byte length, a
    /// 4-byte pad (so the float payload is 8-byte aligned), then per-layer conv state
    /// followed by per-layer scan state, all in dense (GDN-layer-index) order.
    /// </summary>
    /// <remarks>
    /// Layout, byte offsets:
    /// <list type="bullet">
    ///   <item><c>0..4</c>:           <c>_length</c> as int32.</item>
    ///   <item><c>4..8</c>:           padding (zero).</item>
    ///   <item><c>8..C0</c>:          <c>NumGdnLayers</c> contiguous conv blocks,
    ///                                each <see cref="ConvStateFloatsPerLayer"/> floats.</item>
    ///   <item><c>C0..end</c>:        <c>NumGdnLayers</c> contiguous scan blocks,
    ///                                each <see cref="ScanStateFloatsPerLayer"/> floats.</item>
    /// </list>
    /// Per-layer pointers that are <c>null</c> (degenerate <c>ConvKernel == 1</c>)
    /// contribute zero bytes — they're skipped in both directions.
    /// </remarks>
    public long SnapshotBytes =>
        sizeof(int) + sizeof(int) /* pad */ +
        (long)NumGdnLayers * ((long)_convStateBytesPerLayer + (long)_scanStateBytesPerLayer);

    /// <summary>
    /// Write the contents of this cache (including <see cref="Length"/>) into
    /// <paramref name="dst"/>. The destination must be at least
    /// <see cref="SnapshotBytes"/> bytes long.
    /// </summary>
    /// <exception cref="ArgumentException">When <paramref name="dstBytes"/> is too small.</exception>
    public void SnapshotInto(byte* dst, long dstBytes)
    {
        long needed = SnapshotBytes;
        if (dst == null)
            throw new ArgumentNullException(nameof(dst));
        if (dstBytes < needed)
            throw new ArgumentException(
                $"GdnStateCache.SnapshotInto: dstBytes={dstBytes} < SnapshotBytes={needed}.",
                nameof(dstBytes));

        // Header: length + 4-byte zero pad for alignment.
        *(int*)dst = _length;
        *(int*)(dst + sizeof(int)) = 0;

        byte* cursor = dst + 2 * sizeof(int);
        // Conv blocks first, scan blocks second — matches SnapshotBytes layout.
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_convState[g] != null && _convStateBytesPerLayer > 0)
            {
                NativeMemory.Copy(_convState[g], cursor, _convStateBytesPerLayer);
                cursor += (long)_convStateBytesPerLayer;
            }
        }
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_scanState[g] != null && _scanStateBytesPerLayer > 0)
            {
                NativeMemory.Copy(_scanState[g], cursor, _scanStateBytesPerLayer);
                cursor += (long)_scanStateBytesPerLayer;
            }
        }
    }

    /// <summary>
    /// Restore the cache contents (and <see cref="Length"/>) from a snapshot previously
    /// produced by <see cref="SnapshotInto"/>. The source must be at least
    /// <see cref="SnapshotBytes"/> bytes long.
    /// </summary>
    /// <exception cref="ArgumentException">When <paramref name="srcBytes"/> is too small.</exception>
    public void RestoreFrom(byte* src, long srcBytes)
    {
        long needed = SnapshotBytes;
        if (src == null)
            throw new ArgumentNullException(nameof(src));
        if (srcBytes < needed)
            throw new ArgumentException(
                $"GdnStateCache.RestoreFrom: srcBytes={srcBytes} < SnapshotBytes={needed}.",
                nameof(srcBytes));

        _length = *(int*)src;
        // Padding byte ignored.

        byte* cursor = src + 2 * sizeof(int);
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_convState[g] != null && _convStateBytesPerLayer > 0)
            {
                NativeMemory.Copy(cursor, _convState[g], _convStateBytesPerLayer);
                cursor += (long)_convStateBytesPerLayer;
            }
        }
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_scanState[g] != null && _scanStateBytesPerLayer > 0)
            {
                NativeMemory.Copy(cursor, _scanState[g], _scanStateBytesPerLayer);
                cursor += (long)_scanStateBytesPerLayer;
            }
        }
    }

    /// <summary>
    /// Per-layer snapshot footprint: <c>convStateBytesPerLayer + scanStateBytesPerLayer</c>.
    /// Used by the MTP batched verify path (issue #30) which writes a snapshot at
    /// the "between t1 and t2" point of <c>BatchForward2</c> by saving each layer's
    /// post-t1 state as that layer's slot in a single buffer.
    /// </summary>
    public long LayerSnapshotBytes =>
        (long)_convStateBytesPerLayer + (long)_scanStateBytesPerLayer;

    /// <summary>
    /// Copy one layer's conv + scan state (no length header) into <paramref name="dst"/>.
    /// Layout: <c>convBytesPerLayer</c> bytes then <c>scanBytesPerLayer</c> bytes.
    /// Used by <c>BatchForward2</c> to write the "after t1" snapshot incrementally as
    /// each layer finishes its t1 attn/GDN block.
    /// </summary>
    public void SnapshotLayerInto(int gdnLayerIndex, byte* dst, long dstBytes)
    {
        if ((uint)gdnLayerIndex >= (uint)NumGdnLayers)
            throw new ArgumentOutOfRangeException(
                nameof(gdnLayerIndex), gdnLayerIndex,
                $"GDN layer index out of range [0, {NumGdnLayers}).");
        long needed = LayerSnapshotBytes;
        if (dst == null) throw new ArgumentNullException(nameof(dst));
        if (dstBytes < needed)
            throw new ArgumentException(
                $"GdnStateCache.SnapshotLayerInto: dstBytes={dstBytes} < LayerSnapshotBytes={needed}.",
                nameof(dstBytes));

        byte* cursor = dst;
        if (_convState[gdnLayerIndex] != null && _convStateBytesPerLayer > 0)
        {
            NativeMemory.Copy(_convState[gdnLayerIndex], cursor, _convStateBytesPerLayer);
            cursor += (long)_convStateBytesPerLayer;
        }
        if (_scanState[gdnLayerIndex] != null && _scanStateBytesPerLayer > 0)
        {
            NativeMemory.Copy(_scanState[gdnLayerIndex], cursor, _scanStateBytesPerLayer);
        }
    }

    /// <summary>
    /// Restore one layer's conv + scan state from a buffer previously written by
    /// <see cref="SnapshotLayerInto"/>. Does NOT touch <see cref="Length"/>; callers
    /// using this for batched-verify rollback must update length explicitly via
    /// <see cref="SetLength"/>.
    /// </summary>
    public void RestoreLayerFrom(int gdnLayerIndex, byte* src, long srcBytes)
    {
        if ((uint)gdnLayerIndex >= (uint)NumGdnLayers)
            throw new ArgumentOutOfRangeException(
                nameof(gdnLayerIndex), gdnLayerIndex,
                $"GDN layer index out of range [0, {NumGdnLayers}).");
        long needed = LayerSnapshotBytes;
        if (src == null) throw new ArgumentNullException(nameof(src));
        if (srcBytes < needed)
            throw new ArgumentException(
                $"GdnStateCache.RestoreLayerFrom: srcBytes={srcBytes} < LayerSnapshotBytes={needed}.",
                nameof(srcBytes));

        byte* cursor = src;
        if (_convState[gdnLayerIndex] != null && _convStateBytesPerLayer > 0)
        {
            NativeMemory.Copy(cursor, _convState[gdnLayerIndex], _convStateBytesPerLayer);
            cursor += (long)_convStateBytesPerLayer;
        }
        if (_scanState[gdnLayerIndex] != null && _scanStateBytesPerLayer > 0)
        {
            NativeMemory.Copy(cursor, _scanState[gdnLayerIndex], _scanStateBytesPerLayer);
        }
    }

    /// <summary>
    /// Set <see cref="Length"/> explicitly. Used by the batched verify path (issue #30)
    /// to rewind length after a per-layer state restore — the per-layer copy does not
    /// itself carry a length header. Throws when negative.
    /// </summary>
    public void SetLength(int length)
    {
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), length, "Length must be non-negative.");
        _length = length;
    }

    /// <summary>
    /// Free all native state buffers. Safe to call twice — subsequent calls are no-ops.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int g = 0; g < NumGdnLayers; g++)
        {
            if (_convState[g] != null)
            {
                NativeMemory.Free(_convState[g]);
                _convState[g] = null;
            }
            if (_scanState[g] != null)
            {
                NativeMemory.Free(_scanState[g]);
                _scanState[g] = null;
            }
        }
    }
}
