namespace SharpInference.TurboQuant;

/// <summary>
/// Profiles K/V magnitude ratios per layer during warmup to select optimal bit budgets.
/// Run during the first N tokens of prompt processing, then freeze budgets.
/// </summary>
public sealed class MagnitudeProfiler
{
    private readonly int _numLayers;
    private readonly int _warmupTokens;
    private readonly double[] _keySumSq;    // accumulated squared L2 norms per layer
    private readonly double[] _valueSumSq;
    private readonly int[] _sampleCount;
    private bool _frozen;

    /// <summary>Per-layer bit budget, available after <see cref="Freeze"/>.</summary>
    public BitBudget[] Budgets { get; }

    public bool IsFrozen => _frozen;

    public MagnitudeProfiler(int numLayers, int warmupTokens = 256)
    {
        _numLayers = numLayers;
        _warmupTokens = warmupTokens;
        _keySumSq = new double[numLayers];
        _valueSumSq = new double[numLayers];
        _sampleCount = new int[numLayers];
        Budgets = new BitBudget[numLayers];
        // Default budget until profiling completes
        for (int i = 0; i < numLayers; i++)
            Budgets[i] = new BitBudget(3, 3);
    }

    /// <summary>
    /// Record a key and value vector for magnitude profiling.
    /// Call once per layer per token during warmup.
    /// </summary>
    public void Record(int layer, ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (_frozen) return;

        double kSq = 0, vSq = 0;
        for (int i = 0; i < key.Length; i++) kSq += key[i] * (double)key[i];
        for (int i = 0; i < value.Length; i++) vSq += value[i] * (double)value[i];

        _keySumSq[layer] += kSq;
        _valueSumSq[layer] += vSq;
        _sampleCount[layer]++;

        // Auto-freeze when all layers have enough samples
        if (_sampleCount[layer] >= _warmupTokens)
        {
            bool allReady = true;
            for (int i = 0; i < _numLayers; i++)
            {
                if (_sampleCount[i] < _warmupTokens) { allReady = false; break; }
            }
            if (allReady) Freeze();
        }
    }

    /// <summary>
    /// Compute bit budgets from accumulated statistics.
    /// Called automatically after warmup, or manually.
    /// </summary>
    public void Freeze()
    {
        if (_frozen) return;
        _frozen = true;

        for (int i = 0; i < _numLayers; i++)
        {
            if (_sampleCount[i] == 0)
            {
                Budgets[i] = new BitBudget(3, 3);
                continue;
            }

            double avgKeyMag = Math.Sqrt(_keySumSq[i] / _sampleCount[i]);
            double avgValMag = Math.Sqrt(_valueSumSq[i] / _sampleCount[i]);

            // Avoid division by zero
            double ratio = avgValMag > 1e-10 ? avgKeyMag / avgValMag : 1.0;

            // Decision thresholds from llama.cpp community findings:
            // K/V ratio < 10x  → 3-bit uniform for both
            // K/V ratio 10-60x → 4-bit keys, 3-bit values
            // K/V ratio > 100x → 4-bit both (conservative fallback)
            Budgets[i] = ratio switch
            {
                < 10.0 => new BitBudget(3, 3),
                < 60.0 => new BitBudget(4, 3),
                _ => new BitBudget(4, 4)
            };
        }
    }
}

/// <summary>Per-layer bit allocation for keys and values.</summary>
public readonly record struct BitBudget(int KeyBits, int ValueBits);
