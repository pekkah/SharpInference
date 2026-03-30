using System.Text.Json;

namespace SharpInference.TurboQuant;

/// <summary>
/// A trained Lloyd-Max scalar quantisation codebook.
/// Boundaries and centroids are precomputed offline and stored as JSON.
/// </summary>
public sealed class LloydMaxCodebook
{
    public required float[] Boundaries { get; init; }
    public required float[] Centroids { get; init; }
    public int Bits => (int)Math.Log2(Centroids.Length);

    /// <summary>Quantise a single value to the nearest centroid index.</summary>
    public byte Quantise(float value)
    {
        // Binary search over boundaries
        int lo = 0, hi = Boundaries.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (value < Boundaries[mid]) hi = mid;
            else lo = mid + 1;
        }
        return (byte)lo;
    }

    public float Dequantise(byte index) => Centroids[index];

    public static LloydMaxCodebook LoadFromJson(string path)
    {
        using var stream = File.OpenRead(path);
        var doc = JsonSerializer.Deserialize<CodebookJson>(stream)
            ?? throw new InvalidDataException($"Failed to deserialise codebook from {path}");
        return new LloydMaxCodebook
        {
            Boundaries = doc.Boundaries,
            Centroids = doc.Centroids,
        };
    }

    private sealed class CodebookJson
    {
        public float[] Boundaries { get; set; } = [];
        public float[] Centroids { get; set; } = [];
    }
}
