using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;

namespace SharpInference.Engine;

/// <summary>
/// Scalar CPU forward pass for a dense LLaMA-family transformer.
/// All operations are explicit loops — correct first, fast later.
/// </summary>
public sealed unsafe class ForwardPass : IDisposable
{
    private readonly GgufModel _model;
    private readonly IComputeBackend _backend;
    private readonly ModelHyperparams _hp;
    private readonly KvCache _kvCache;

    // Weight cache: dequantized weights keyed by tensor name.
    // F32 weights are zero-copy pointers into the mmap; quantized are dequantized on first access.
    private readonly Dictionary<string, nint> _weightCache = new();
    private readonly HashSet<string> _ownedWeights = new(); // track allocations we own

    // Preallocated scratch buffers (owned, freed on Dispose)
    private readonly float* _hidden;     // [embDim]
    private readonly float* _residual;   // [embDim]
    private readonly float* _normBuf;    // [embDim]
    private readonly float* _q;          // [numHeads * headDim]
    private readonly float* _k;          // [numKvHeads * headDim]
    private readonly float* _v;          // [numKvHeads * headDim]
    private readonly float* _attnOut;    // [numHeads * headDim]
    private readonly float* _ffnGate;    // [intermDim]
    private readonly float* _ffnUp;      // [intermDim]
    private readonly float* _ffnDown;    // [embDim]
    private readonly float* _logits;     // [vocabSize]
    private readonly float* _attnScores; // [maxSeqLen] for one head

    private readonly int _embDim;
    private readonly int _headDim;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headsPerKvGroup;
    private readonly int _intermDim;

    public ForwardPass(GgufModel model, IComputeBackend backend, ModelHyperparams hp)
    {
        _model = model;
        _backend = backend;
        _hp = hp;
        _kvCache = new KvCache(hp);

        _embDim = hp.EmbeddingDim;
        _headDim = hp.EmbeddingDim / hp.NumHeads;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        _hidden = Alloc(_embDim);
        _residual = Alloc(_embDim);
        _normBuf = Alloc(_embDim);
        _q = Alloc(_numHeads * _headDim);
        _k = Alloc(_numKvHeads * _headDim);
        _v = Alloc(_numKvHeads * _headDim);
        _attnOut = Alloc(_numHeads * _headDim);
        _ffnGate = Alloc(_intermDim);
        _ffnUp = Alloc(_intermDim);
        _ffnDown = Alloc(_embDim);
        _logits = Alloc(hp.VocabSize);
        _attnScores = Alloc(hp.ContextLength);
    }

    public KvCache Cache => _kvCache;

    /// <summary>
    /// Run a single token through the full transformer and return a span over the logits buffer.
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Token embedding lookup
        Embed(token);

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Evict previous layer's dequantized weights to keep memory bounded
            if (layer > 0)
                EvictLayerWeights(layer - 1);
            // Save residual
            Copy(_residual, _hidden, _embDim);

            // Pre-attention RMS norm
            var attnNormWeight = GetWeight($"blk.{layer}.attn_norm.weight", _embDim);
            RmsNorm(_normBuf, _hidden, attnNormWeight, _embDim, _hp.RmsNormEps);

            // Q/K/V projections
            var wq = GetWeight2D($"blk.{layer}.attn_q.weight", _numHeads * _headDim, _embDim);
            var wk = GetWeight2D($"blk.{layer}.attn_k.weight", _numKvHeads * _headDim, _embDim);
            var wv = GetWeight2D($"blk.{layer}.attn_v.weight", _numKvHeads * _headDim, _embDim);

            MatVec(_q, wq, _normBuf, _numHeads * _headDim, _embDim);
            MatVec(_k, wk, _normBuf, _numKvHeads * _headDim, _embDim);
            MatVec(_v, wv, _normBuf, _numKvHeads * _headDim, _embDim);

            // RoPE on Q and K
            ApplyRoPE(_q, position, _numHeads, _headDim, _hp.RopeTheta);
            ApplyRoPE(_k, position, _numKvHeads, _headDim, _hp.RopeTheta);

            // Store K, V in cache
            _kvCache.Append(layer,
                new ReadOnlySpan<float>(_k, _numKvHeads * _headDim),
                new ReadOnlySpan<float>(_v, _numKvHeads * _headDim));

            // Scaled dot-product attention with GQA
            Attention(layer, position);

            // Output projection
            var wo = GetWeight2D($"blk.{layer}.attn_output.weight", _embDim, _numHeads * _headDim);
            MatVec(_hidden, wo, _attnOut, _embDim, _numHeads * _headDim);

            // Residual add
            Add(_hidden, _residual, _embDim);

            // Save residual for FFN
            Copy(_residual, _hidden, _embDim);

            // Pre-FFN RMS norm
            var ffnNormWeight = GetWeight($"blk.{layer}.ffn_norm.weight", _embDim);
            RmsNorm(_normBuf, _hidden, ffnNormWeight, _embDim, _hp.RmsNormEps);

            // SwiGLU FFN: output = Wdown @ (SiLU(Wgate @ x) * (Wup @ x))
            var wGate = GetWeight2D($"blk.{layer}.ffn_gate.weight", _intermDim, _embDim);
            var wUp = GetWeight2D($"blk.{layer}.ffn_up.weight", _intermDim, _embDim);
            var wDown = GetWeight2D($"blk.{layer}.ffn_down.weight", _embDim, _intermDim);

            MatVec(_ffnGate, wGate, _normBuf, _intermDim, _embDim);
            MatVec(_ffnUp, wUp, _normBuf, _intermDim, _embDim);

            // SiLU on gate, then elementwise multiply with up
            for (int i = 0; i < _intermDim; i++)
            {
                float g = _ffnGate[i];
                _ffnGate[i] = g / (1.0f + MathF.Exp(-g)) * _ffnUp[i];
            }

            MatVec(_hidden, wDown, _ffnGate, _embDim, _intermDim);

            // Residual add
            Add(_hidden, _residual, _embDim);
        }

        // Evict last layer's weights
        EvictLayerWeights(_hp.NumLayers - 1);

        // Increment KV cache position after processing all layers
        _kvCache.IncrementPosition();

        // 3. Final RMS norm
        var finalNormWeight = GetWeight("output_norm.weight", _embDim);
        RmsNorm(_hidden, _hidden, finalNormWeight, _embDim, _hp.RmsNormEps);

        // 4. Output projection (LM head) → logits
        // SmolLM2 uses tied embeddings: output weight = embedding weight.
        // GGML stores [ne0=embDim, ne1=vocabSize] which in row-major is [vocabSize, embDim].
        var outputName = _model.FindTensor("output.weight") is not null
            ? "output.weight"
            : "token_embd.weight";
        var outputWeight = GetWeight(outputName, _hp.VocabSize * _embDim);
        MatVec(_logits, outputWeight, _hidden, _hp.VocabSize, _embDim);

        return new ReadOnlySpan<float>(_logits, _hp.VocabSize);
    }

    /// <summary>
    /// Scaled dot-product attention with grouped-query attention (GQA) support.
    /// Writes result into _attnOut.
    /// </summary>
    private void Attention(int layer, int position)
    {
        int seqLen = position + 1; // include current position (already in cache)
        float scale = 1.0f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _numHeads; h++)
        {
            // Which KV head does this query head attend to?
            int kvHead = h / _headsPerKvGroup;

            float* qHead = _q + h * _headDim;
            float* outHead = _attnOut + h * _headDim;

            // Compute attention scores: Q_h · K_t for all cached positions
            for (int t = 0; t < seqLen; t++)
            {
                float* kVec = _kvCache.KeyAt(layer, t) + kvHead * _headDim;
                float dot = 0;
                for (int d = 0; d < _headDim; d++)
                    dot += qHead[d] * kVec[d];
                _attnScores[t] = dot * scale;
            }

            // Softmax over scores [0..seqLen)
            SoftmaxInPlace(_attnScores, seqLen);

            // Weighted sum of values
            for (int d = 0; d < _headDim; d++)
                outHead[d] = 0;

            for (int t = 0; t < seqLen; t++)
            {
                float* vVec = _kvCache.ValueAt(layer, t) + kvHead * _headDim;
                float w = _attnScores[t];
                for (int d = 0; d < _headDim; d++)
                    outHead[d] += w * vVec[d];
            }
        }
    }

    // --- Primitive operations (scalar, no SIMD) ---

    private void Embed(int token)
    {
        // Get the full embedding table (dequantized if needed, cached)
        int vocabSize = _hp.VocabSize;
        var embedPtr = GetWeight("token_embd.weight", vocabSize * _embDim);

        // Copy the row for this token
        var row = embedPtr + (long)token * _embDim;
        Copy(_hidden, row, _embDim);
    }

    private static void MatVec(float* output, float* matrix, float* vector, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
        {
            float sum = 0;
            float* row = matrix + (long)i * cols;
            for (int j = 0; j < cols; j++)
                sum += row[j] * vector[j];
            output[i] = sum;
        }
    }

    /// <summary>
    /// Transposed matrix-vector multiply: output[i] = sum_j(matrix[j,i] * vector[j]).
    /// matrix is [cols, rows] in memory (row-major), so column i of the logical matrix
    /// is strided at matrix[j * rows + i].
    /// Used for tied embeddings where the weight shape is [embDim, vocabSize].
    /// </summary>
    private static void MatVecTransposed(float* output, float* matrix, float* vector, int rows, int cols)
    {
        for (int i = 0; i < rows; i++)
            output[i] = 0;

        for (int j = 0; j < cols; j++)
        {
            float vj = vector[j];
            float* row = matrix + (long)j * rows;
            for (int i = 0; i < rows; i++)
                output[i] += row[i] * vj;
        }
    }

    private static void RmsNorm(float* output, float* input, float* weight, int size, float eps)
    {
        float sumSq = 0;
        for (int i = 0; i < size; i++)
            sumSq += input[i] * input[i];

        float scale = 1.0f / MathF.Sqrt(sumSq / size + eps);
        for (int i = 0; i < size; i++)
            output[i] = input[i] * scale * weight[i];
    }

    private static void ApplyRoPE(float* x, int position, int numHeads, int headDim, float theta)
    {
        int halfDim = headDim / 2;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = 1.0f / MathF.Pow(theta, 2.0f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle);
                float sin = MathF.Sin(angle);

                // Interleaved pairs (mode 0 / LLaMA-style):
                // rotate (x[2i], x[2i+1]) not (x[i], x[i+halfDim])
                int j = 2 * i;
                float x0 = head[j];
                float x1 = head[j + 1];
                head[j] = x0 * cos - x1 * sin;
                head[j + 1] = x0 * sin + x1 * cos;
            }
        }
    }

    private static void SoftmaxInPlace(float* x, int size)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < size; i++)
            if (x[i] > max) max = x[i];

        float sum = 0;
        for (int i = 0; i < size; i++)
        {
            x[i] = MathF.Exp(x[i] - max);
            sum += x[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < size; i++)
            x[i] *= invSum;
    }

    private static void Add(float* dst, float* src, int size)
    {
        for (int i = 0; i < size; i++)
            dst[i] += src[i];
    }

    private static void Copy(float* dst, float* src, int size)
    {
        new ReadOnlySpan<float>(src, size).CopyTo(new Span<float>(dst, size));
    }

    // --- Weight access helpers ---

    /// <summary>
    /// Get a weight tensor as a float pointer. F32 tensors are zero-copy from mmap;
    /// quantized tensors are dequantized on first access and cached.
    /// </summary>
    private float* GetWeight(string name, int expectedSize)
    {
        if (_weightCache.TryGetValue(name, out var cached))
            return (float*)cached;

        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing weight tensor: {name}");

        var data = _model.GetTensorData(info);

        if (info.DType == DType.Float32)
        {
            // The mmap data is already pinned for the lifetime of GgufModel.
            // Copy the pointer from the span — safe as long as GgufModel stays alive.
            // We can't use 'fixed' because it would create a temporary pin that expires.
            // Instead, allocate a buffer and copy (correctness over zero-copy for Phase 1).
            int count2 = (int)info.ElementCount;
            var buf2 = Alloc(count2);
            MemoryMarshal.Cast<byte, float>(data).Slice(0, count2).CopyTo(new Span<float>(buf2, count2));
            _weightCache[name] = (nint)buf2;
            _ownedWeights.Add(name);
            return buf2;
        }

        // Dequantize and cache
        int count = (int)info.ElementCount;
        var buf = Alloc(count);
        Dequantize.ToFloat32(data, new Span<float>(buf, count), info.DType, count);
        _weightCache[name] = (nint)buf;
        _ownedWeights.Add(name);
        return buf;
    }

    /// <summary>Get a 2D weight matrix as a float pointer.</summary>
    private float* GetWeight2D(string name, int rows, int cols)
    {
        return GetWeight(name, rows * cols);
    }

    /// <summary>
    /// Free dequantized weights for a specific layer to bound peak memory.
    /// Global weights (embeddings, norms) are kept.
    /// </summary>
    private void EvictLayerWeights(int layer)
    {
        var prefix = $"blk.{layer}.";
        var toRemove = new List<string>();
        foreach (var name in _weightCache.Keys)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
                toRemove.Add(name);
        }
        foreach (var name in toRemove)
        {
            if (_ownedWeights.Remove(name) && _weightCache.TryGetValue(name, out var ptr))
                NativeMemory.Free((void*)ptr);
            _weightCache.Remove(name);
        }
    }

    private static float* Alloc(int count) =>
        (float*)NativeMemory.AllocZeroed((nuint)(count * sizeof(float)));

    public void Dispose()
    {
        NativeMemory.Free(_hidden);
        NativeMemory.Free(_residual);
        NativeMemory.Free(_normBuf);
        NativeMemory.Free(_q);
        NativeMemory.Free(_k);
        NativeMemory.Free(_v);
        NativeMemory.Free(_attnOut);
        NativeMemory.Free(_ffnGate);
        NativeMemory.Free(_ffnUp);
        NativeMemory.Free(_ffnDown);
        NativeMemory.Free(_logits);
        NativeMemory.Free(_attnScores);

        foreach (var name in _ownedWeights)
        {
            if (_weightCache.TryGetValue(name, out var ptr))
                NativeMemory.Free((void*)ptr);
        }
        _weightCache.Clear();
        _ownedWeights.Clear();

        _kvCache.Dispose();
    }
}
