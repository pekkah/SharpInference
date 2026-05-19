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
