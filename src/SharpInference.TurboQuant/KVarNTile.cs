namespace SharpInference.TurboQuant;

/// <summary>
/// One compressed KVarN tile for a single (layer, kv-head): up to
/// <see cref="KVarN.TileSize"/> tokens of one head, quantized either as keys
/// (per-channel 4-bit) or values (per-token 2-bit).
///
/// This is the P0 reference container (issue #180) — codes live in managed
/// byte arrays, token-major, with the scale vectors kept as <c>float[]</c>.
/// The performance phases (P1 AVX2 / P2 CUDA) replace this with a packed native
/// tile layout and fused kernels; the math is identical.
/// </summary>
public sealed class KVarNTile
{
    /// <summary>Number of tokens stored in this tile (1..TileSize).</summary>
    public int T { get; }

    /// <summary>Channel count (head dimension).</summary>
    public int HeadDim { get; }

    /// <summary>True for key tiles (per-channel 4-bit), false for value tiles (per-token 2-bit).</summary>
    public bool PerChannel { get; }

    /// <summary>Sinkhorn per-channel (column) scales, length headDim.</summary>
    public float[] CScale { get; }

    /// <summary>Sinkhorn per-token (row) scales, length T.</summary>
    public float[] RScale { get; }

    /// <summary>Key folded per-channel quant scale (qscale·cscale), length headDim. Key tiles only.</summary>
    public float[] KQScale { get; }

    /// <summary>Key folded per-channel zero point (zero·cscale), length headDim. Key tiles only.</summary>
    public float[] KZero { get; }

    /// <summary>Value folded per-token quant scale (qscale·rscale), length T. Value tiles only.</summary>
    public float[] VQScale { get; }

    /// <summary>Value folded per-token zero point (zero·rscale), length T. Value tiles only.</summary>
    public float[] VZero { get; }

    private readonly byte[] _codes;   // token-major packed codes
    private readonly int _rowStride;  // packed bytes per token row

    public KVarNTile(int t, int headDim, bool perChannel)
    {
        T = t;
        HeadDim = headDim;
        PerChannel = perChannel;
        CScale = new float[headDim];
        RScale = new float[t];

        if (perChannel)
        {
            // 4-bit codes: two channels per byte.
            _rowStride = (headDim + 1) / 2;
            KQScale = new float[headDim];
            KZero = new float[headDim];
            VQScale = [];
            VZero = [];
        }
        else
        {
            // 2-bit codes: four channels per byte.
            _rowStride = (headDim + 3) / 4;
            VQScale = new float[t];
            VZero = new float[t];
            KQScale = [];
            KZero = [];
        }
        _codes = new byte[(long)t * _rowStride <= int.MaxValue ? t * _rowStride : throw new ArgumentException("Tile too large")];
    }

    /// <summary>Estimated bytes held by this tile (codes + scale vectors).</summary>
    public long EstimatedBytes =>
        _codes.Length
        + (long)(CScale.Length + RScale.Length + KQScale.Length + KZero.Length
                 + VQScale.Length + VZero.Length) * sizeof(float);

    /// <summary>Set a 4-bit key code (token <paramref name="i"/>, channel <paramref name="d"/>).</summary>
    public void SetKeyCode(int i, int d, int code)
    {
        int idx = i * _rowStride + (d >> 1);
        if ((d & 1) == 0)
            _codes[idx] = (byte)((_codes[idx] & 0xF0) | (code & 0x0F));
        else
            _codes[idx] = (byte)((_codes[idx] & 0x0F) | ((code & 0x0F) << 4));
    }

    /// <summary>Get a 4-bit key code.</summary>
    public int GetKeyCode(int i, int d)
    {
        int idx = i * _rowStride + (d >> 1);
        byte b = _codes[idx];
        return (d & 1) == 0 ? b & 0x0F : (b >> 4) & 0x0F;
    }

    /// <summary>Set a 2-bit value code.</summary>
    public void SetValueCode(int i, int d, int code)
    {
        int idx = i * _rowStride + (d >> 2);
        int shift = (d & 3) << 1;
        _codes[idx] = (byte)((_codes[idx] & ~(0x03 << shift)) | ((code & 0x03) << shift));
    }

    /// <summary>Get a 2-bit value code.</summary>
    public int GetValueCode(int i, int d)
    {
        int idx = i * _rowStride + (d >> 2);
        int shift = (d & 3) << 1;
        return (_codes[idx] >> shift) & 0x03;
    }
}
