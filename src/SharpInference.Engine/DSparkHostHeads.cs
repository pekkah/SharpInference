using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Engine;

/// <summary>
/// The sequential (host-side) part of a DSpark draft head, shared by the CPU
/// and CUDA backbones: the rank-r vanilla Markov re-bias applied greedily over
/// the block's parallel base logits, and the confidence head scoring
/// <c>[hidden_j ‖ markov_w1[prev_j]]</c>. Owns the markov/confidence weights
/// (markov_w2 F32, markov_w1 row-gathered in storage dtype, confidence proj)
/// and the per-block scratch. The backbone owns embeddings, layers, fc, norms
/// and lm_head — it hands this class the base logits and final hidden states.
/// </summary>
internal sealed unsafe class DSparkHostHeads : IDisposable
{
    private readonly DSparkConfig _cfg;
    private readonly int _vocab, _rank, _embDim, _block;

    private float* _markovW2;                                   // [vocab, rank]
    private ushort* _markovW1Bf16; private float* _markovW1F32; // [vocab, rank]
    private float* _confW;                                      // [embDim (+ rank)]
    private float _confB;

    private float* _bias;      // [vocab]
    private float* _w1Rows;    // [block, rank]
    private float* _confFeat;  // [embDim + rank]
    private bool _disposed;

    public DSparkHostHeads(DSparkConfig cfg, SafetensorsLoader weights)
    {
        _cfg = cfg;
        _vocab = cfg.VocabSize;
        _rank = cfg.MarkovRank;
        _embDim = cfg.HiddenSize;
        _block = cfg.BlockSize;

        try
        {
            if (_rank > 0)
            {
                _markovW2 = DSparkWeightLoading.LoadF32(weights,
                    "markov_head.markov_w2.weight", [_vocab, _rank]);
                DSparkWeightLoading.LoadRowTable(weights,
                    "markov_head.markov_w1.weight", [_vocab, _rank],
                    out _markovW1Bf16, out _markovW1F32);
            }

            if (cfg.EnableConfidenceHead)
            {
                int confIn = _embDim + (cfg.ConfidenceHeadWithMarkov ? _rank : 0);
                _confW = DSparkWeightLoading.LoadF32(weights,
                    "confidence_head.proj.weight", [1, confIn]);
                var b = weights.ReadF32("confidence_head.proj.bias");
                if (b.Length != 1)
                    throw new InvalidDataException("confidence_head.proj.bias must be a scalar.");
                _confB = b[0];
            }

            _bias = DSparkWeightLoading.Alloc(_vocab);
            _w1Rows = DSparkWeightLoading.Alloc((long)_block * Math.Max(_rank, 1));
            _confFeat = DSparkWeightLoading.Alloc(_embDim + Math.Max(_rank, 1));
        }
        catch
        {
            FreeAll();
            throw;
        }
    }

    /// <summary>
    /// Sequential Markov correction + greedy sampling over one block's parallel
    /// base logits, then confidence scoring. <paramref name="baseLogits"/> is
    /// [block, vocab] row-major and is biased IN PLACE row by row;
    /// <paramref name="blockHidden"/> is the backbone's final-normed [block, embDim].
    /// </summary>
    public DSparkProposal GreedyBlock(float* baseLogits, float* blockHidden, int anchorToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var tokens = new int[_block];
        int prev = anchorToken;
        for (int j = 0; j < _block; j++)
        {
            float* logits = baseLogits + (long)j * _vocab;
            if (_rank > 0)
            {
                float* w1Row = _w1Rows + (long)j * _rank;
                MarkovW1Row(prev, w1Row);
                SimdKernels.MatVecF32(_bias, _markovW2, w1Row, _vocab, _rank);
                SimdKernels.AddInPlace(logits, _bias, _vocab);
            }
            tokens[j] = Sampler.Greedy(new ReadOnlySpan<float>(logits, _vocab));
            prev = tokens[j];
        }

        var conf = new float[_block];
        if (_confW != null)
        {
            int confIn = _embDim + (_cfg.ConfidenceHeadWithMarkov ? _rank : 0);
            for (int j = 0; j < _block; j++)
            {
                new ReadOnlySpan<float>(blockHidden + (long)j * _embDim, _embDim)
                    .CopyTo(new Span<float>(_confFeat, _embDim));
                if (_cfg.ConfidenceHeadWithMarkov)
                    new ReadOnlySpan<float>(_w1Rows + (long)j * _rank, _rank)
                        .CopyTo(new Span<float>(_confFeat + _embDim, _rank));
                float logit = SimdKernels.DotF32(_confW, _confFeat, confIn) + _confB;
                conf[j] = 1f / (1f + MathF.Exp(-logit));
            }
        }
        else
        {
            Array.Fill(conf, 1f);
        }

        return new DSparkProposal(tokens, conf);
    }

    private void MarkovW1Row(int token, float* dst)
    {
        if (_markovW1F32 != null)
        {
            NativeMemory.Copy(_markovW1F32 + (long)token * _rank, dst, (nuint)(_rank * sizeof(float)));
        }
        else
        {
            var src = new ReadOnlySpan<byte>((byte*)(_markovW1Bf16 + (long)token * _rank),
                _rank * sizeof(ushort));
            Dequantize.ToFloat32(src, new Span<float>(dst, _rank), DType.BFloat16, _rank);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        FreeAll();
    }

    private void FreeAll()
    {
        Free(ref _markovW2); Free(ref _confW);
        Free(ref _markovW1F32);
        if (_markovW1Bf16 != null) { NativeMemory.Free(_markovW1Bf16); _markovW1Bf16 = null; }
        Free(ref _bias); Free(ref _w1Rows); Free(ref _confFeat);

        static void Free(ref float* p) { if (p != null) { NativeMemory.Free(p); p = null; } }
    }
}

/// <summary>
/// Safetensors→native loading helpers shared by the DSpark backbones and heads:
/// shape-validated F32 loads into NativeMemory, and storage-dtype row tables
/// (BF16 kept raw for on-access row dequant; anything else widened to F32).
/// </summary>
internal static unsafe class DSparkWeightLoading
{
    public static float* Alloc(long floats) =>
        (float*)NativeMemory.AllocZeroed((nuint)(floats * sizeof(float)));

    public static void ValidateShape(SafetensorsLoader st, string name, int[] expectedShape)
    {
        var shape = st.GetShape(name);
        if (!shape.AsSpan().SequenceEqual(expectedShape))
            throw new InvalidDataException(
                $"DSpark tensor '{name}' has shape [{string.Join(",", shape)}], " +
                $"expected [{string.Join(",", expectedShape)}].");
    }

    public static float* LoadF32(SafetensorsLoader st, string name, int[] expectedShape)
    {
        ValidateShape(st, name, expectedShape);
        var managed = st.ReadF32(name);
        var buf = Alloc(managed.Length);
        managed.AsSpan().CopyTo(new Span<float>(buf, managed.Length));
        return buf;
    }

    /// <summary>
    /// Load a large row-gathered table in its storage dtype: BF16 stays BF16
    /// (half the resident bytes; rows are dequantized on access), anything else
    /// goes through the F32 conversion path.
    /// </summary>
    public static void LoadRowTable(SafetensorsLoader st, string name, int[] expectedShape,
        out ushort* bf16, out float* f32)
    {
        ValidateShape(st, name, expectedShape);

        long elems = (long)expectedShape[0] * expectedShape[1];
        var raw = st.ReadRaw(name, out string dtype);
        if (dtype == "BF16")
        {
            f32 = null;
            bf16 = (ushort*)NativeMemory.Alloc((nuint)raw.LongLength);
            raw.AsSpan().CopyTo(new Span<byte>(bf16, raw.Length));
        }
        else
        {
            bf16 = null;
            var managed = st.ReadF32(name);
            f32 = Alloc(elems);
            managed.AsSpan().CopyTo(new Span<float>(f32, managed.Length));
        }
    }
}
