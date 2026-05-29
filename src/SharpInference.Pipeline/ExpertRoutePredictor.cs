namespace SharpInference.Pipeline;

/// <summary>
/// Predicts which experts an MoE layer will activate for the next token, using the
/// previous token's selection at the same layer. MoE routing has strong cross-token
/// temporal locality at a fixed layer (the same premise that makes the SLRU expert
/// cache effective), so last-token selections are a cheap, training-free predictor
/// — the PreScope / MoE-Infinity style of activation-aware prefetching.
///
/// <para>
/// The win over reacting to the current layer's own router is <i>lead time</i>: while
/// layer L is still computing, the predictor lets the prefetcher start loading layer
/// L+1's likely experts, hiding the PCIe transfer behind compute. Predictions are only
/// ever prefetch hints — a wrong guess wastes a transfer but never affects output, so
/// no accuracy bound is required.
/// </para>
///
/// Pure CPU state; not thread-safe (call from the single decode thread).
/// </summary>
public sealed class ExpertRoutePredictor
{
    private readonly int _numLayers;
    private readonly int _maxActive;
    private readonly int[] _experts;  // [layer * maxActive + k]
    private readonly int[] _counts;   // valid entries per layer (0 until first seen)

    public ExpertRoutePredictor(int numLayers, int maxActiveExperts)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxActiveExperts <= 0) throw new ArgumentOutOfRangeException(nameof(maxActiveExperts));
        _numLayers = numLayers;
        _maxActive = maxActiveExperts;
        _experts = new int[numLayers * maxActiveExperts];
        _counts = new int[numLayers];
    }

    /// <summary>Record the experts a layer actually selected for the current token.</summary>
    public void Record(int layer, ReadOnlySpan<int> selected)
    {
        if ((uint)layer >= (uint)_numLayers) return;
        int n = Math.Min(selected.Length, _maxActive);
        selected[..n].CopyTo(_experts.AsSpan(layer * _maxActive, n));
        _counts[layer] = n;
    }

    /// <summary>
    /// Predict <paramref name="layer"/>'s experts for the next token (its previous-token
    /// selection). Returns false until the layer has been observed at least once.
    /// </summary>
    public bool TryPredict(int layer, out ReadOnlySpan<int> experts)
    {
        if ((uint)layer >= (uint)_numLayers || _counts[layer] == 0)
        {
            experts = default;
            return false;
        }
        experts = _experts.AsSpan(layer * _maxActive, _counts[layer]);
        return true;
    }

    /// <summary>Forget all history (call when starting a new sequence / on cache reset).</summary>
    public void Reset() => Array.Clear(_counts);
}
