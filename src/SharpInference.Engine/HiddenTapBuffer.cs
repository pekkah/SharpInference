using System.Runtime.InteropServices;

namespace SharpInference.Engine;

/// <summary>
/// Position-indexed native storage for hidden-state taps
/// (<see cref="Core.IForwardPass.EnableHiddenTaps"/>, PR #413): row = absolute
/// position, slot = tapped-layer index in enable order, each slot holding one
/// embDim-float layer output. Grows geometrically as positions advance (an
/// eager full-context allocation would be GBs for long-context models). Shared
/// by the CPU and CUDA forward passes so the two can't drift on layout or
/// growth semantics. Rows survive <c>TruncateTo</c> — a re-processed position
/// overwrites its row. Single-sequence use only.
/// </summary>
internal sealed unsafe class HiddenTapBuffer : IDisposable
{
    private readonly int _embDim;
    private readonly int _contextCap;
    private readonly int[] _slotByLayer;
    private float* _buf;
    private long _capacityPositions;
    private int _highWater;

    /// <summary>Number of tapped layers (slots per row).</summary>
    public int TapCount { get; }

    /// <summary>Floats per row: TapCount × embDim.</summary>
    public int TapDim => TapCount * _embDim;

    public HiddenTapBuffer(ReadOnlySpan<int> layerIds, int numLayers, int embDim, int contextCap)
    {
        if (layerIds.Length == 0)
            throw new ArgumentException("At least one tap layer is required.", nameof(layerIds));

        _slotByLayer = new int[numLayers];
        Array.Fill(_slotByLayer, -1);
        int prev = -1;
        for (int s = 0; s < layerIds.Length; s++)
        {
            int layer = layerIds[s];
            if (layer <= prev || layer >= numLayers)
                throw new ArgumentOutOfRangeException(nameof(layerIds),
                    $"Tap layer ids must be strictly increasing within [0, {numLayers - 1}]; got {layer}.");
            _slotByLayer[layer] = s;
            prev = layer;
        }

        TapCount = layerIds.Length;
        _embDim = embDim;
        _contextCap = contextCap;
    }

    /// <summary>Slot index of a layer, or -1 when the layer is untapped.</summary>
    public int SlotOf(int layer) => _slotByLayer[layer];

    /// <summary>Concatenated tap row of a captured position; empty when uncaptured.</summary>
    public ReadOnlySpan<float> At(int position)
    {
        if (position < 0 || position >= _highWater) return default;
        return new ReadOnlySpan<float>(_buf + (long)position * TapDim, TapDim);
    }

    /// <summary>
    /// Writable span of one (position, slot) cell (embDim floats), growing the
    /// buffer and advancing the high-water mark as needed.
    /// </summary>
    public Span<float> RowSlot(int position, int slot)
    {
        EnsureCapacity(position);
        if (position >= _highWater) _highWater = position + 1;
        return new Span<float>(_buf + ((long)position * TapCount + slot) * _embDim, _embDim);
    }

    private void EnsureCapacity(int position)
    {
        if (position < _capacityPositions) return;
        long needed = position + 1L;
        // Geometric growth from 256, capped at the model context so the last
        // doubling can't overshoot it; `needed` always wins when it's larger.
        long newCap = Math.Max(Math.Min(
            _capacityPositions == 0 ? 256 : _capacityPositions * 2,
            _contextCap), needed);
        long rowFloats = (long)TapCount * _embDim;
        var newBuf = (float*)NativeMemory.AllocZeroed((nuint)(newCap * rowFloats * sizeof(float)));
        if (_buf != null)
        {
            NativeMemory.Copy(_buf, newBuf, (nuint)(_capacityPositions * rowFloats * sizeof(float)));
            NativeMemory.Free(_buf);
        }
        _buf = newBuf;
        _capacityPositions = newCap;
    }

    public void Dispose()
    {
        if (_buf != null)
        {
            NativeMemory.Free(_buf);
            _buf = null;
        }
    }
}
