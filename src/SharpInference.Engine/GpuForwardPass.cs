using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.TurboQuant;
using Vortice.Vulkan;
using SharpInference.Vulkan;
using static Vortice.Vulkan.Vulkan;

namespace SharpInference.Engine;

/// <summary>
/// GPU-accelerated forward pass for LLaMA-family transformers.
/// All weight data resides in VRAM. Compute shaders handle dequantization,
/// MatVec, normalization, attention, and FFN on the GPU.
///
/// For operations not yet GPU-accelerated (attention scoring/aggregation),
/// falls back to CPU with download/upload round-trips.
/// </summary>
public sealed unsafe class GpuForwardPass : IDisposable
{
    private readonly VulkanBackend _gpu;
    private readonly GgufModel _model;
    private readonly ModelHyperparams _hp;

    // Pre-allocated logits download buffer (avoids GC allocation per token)
    private readonly float[] _logitsBuf;

    // GPU scratch buffers
    private readonly Tensor _hidden;     // [embDim]
    private readonly Tensor _residual;   // [embDim]
    private readonly Tensor _normBuf;    // [embDim]
    private readonly Tensor _q;          // [numHeads * headDim]
    private readonly Tensor _k;          // [numKvHeads * headDim]
    private readonly Tensor _v;          // [numKvHeads * headDim]
    private readonly Tensor _attnOut;    // [numHeads * headDim]
    private readonly Tensor _ffnGate;    // [intermDim]
    private readonly Tensor _ffnUp;      // [intermDim]
    private readonly Tensor _logits;     // [vocabSize]

    // Embedding table in VRAM (quantized for large vocabs, F32 for small)
    private readonly Tensor _gpuEmbedding;
    private readonly bool _embIsQuantized;

    // GPU weight tensors (Q4_K/Q6_K bytes uploaded to VRAM)
    private readonly Tensor[] _wAttnNorm;
    private readonly Tensor[] _wq, _wk, _wv, _wo;
    private readonly Tensor[] _wFfnNorm;
    private readonly Tensor[] _wGate, _wUp, _wDown;
    private readonly Tensor _wOutputNorm;
    private readonly Tensor _wOutput;

    // Optional attention biases in VRAM (Qwen models)
    private readonly bool _hasAttnBias;
    private readonly Tensor[]? _bq, _bk, _bv, _bo;

    // Optional per-head Q/K RMSNorm weights in VRAM (Qwen3)
    private readonly bool _hasQkNorm;
    private readonly Tensor[]? _wqNorm, _wkNorm;

    // KV cache in VRAM: per-layer K and V buffers [maxSeqLen, kvDim]
    private readonly Tensor[] _gpuKCache;  // per layer (FP32, or FP32 window when TQ)
    private readonly Tensor[] _gpuVCache;  // per layer
    private readonly int _maxSeqLen;
    private int _kvLength; // current sequence length in cache

    /// <summary>Maximum sequence length (context size) configured for this forward pass.</summary>
    public int MaxSeqLen => _maxSeqLen;

    // CPU KV cache kept for fallback (not used when GPU attention works)
    private readonly Engine.KvCache _kvCache;

    // TurboQuant GPU state (null when TQ disabled)
    private readonly bool _tqEnabled;
    private readonly int _tqFp32Window;
    private readonly int _tqBlockBytes;
    private Tensor[]? _gpuTqKCache;    // per layer, compressed VRAM
    private Tensor[]? _gpuTqVCache;    // per layer, compressed VRAM
    private Tensor? _gpuSignPatterns;  // [numKvHeads * headDim] sign flips
    private Tensor? _gpuCodebook;      // [8] centroids (3-bit)
    private Tensor? _gpuBoundaries;    // [7] decision boundaries
    private Tensor? _rotatedQ;         // [numHeads * headDim] WHT-rotated query
    private Tensor? _evictK;           // [numKvHeads * headDim] scratch for evicted FP32 entry
    private Tensor? _evictV;
    private int _tqCompressedLen;      // positions in TQ storage
    private int _fp32WriteIdx;         // ring buffer write position in FP32 window
    private int _fp32Count;            // number of FP32 positions currently stored

    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim;

    public GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int maxContextLength = 0, bool enableTurboQuant = false, int tqFp32Window = 256, int tqBits = 3)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _tqEnabled = enableTurboQuant;
        _tqFp32Window = tqFp32Window;
        _tqBlockBytes = enableTurboQuant ? TurboQuantOps.BlockSize(tqBits, hp.EmbeddingDim / hp.NumHeads) : 0;

        if (maxContextLength > 0)
        {
            _maxSeqLen = Math.Min(maxContextLength, hp.ContextLength);
        }
        else if (enableTurboQuant)
        {
            _maxSeqLen = EstimateMaxContextTq(model, gpu, hp, tqFp32Window, tqBits);
        }
        else
        {
            _maxSeqLen = EstimateMaxContext(model, gpu, hp);
        }

        _kvCache = new Engine.KvCache(hp.NumLayers, _maxSeqLen, hp.NumKvHeads, hp.EmbeddingDim / hp.NumHeads);
        Console.Error.WriteLine($"[GpuForwardPass] Context size: {_maxSeqLen} (model max: {hp.ContextLength}){(enableTurboQuant ? " [TQ3]" : "")}");

        _embDim = hp.EmbeddingDim;
        _headDim = hp.EmbeddingDim / hp.NumHeads;
        _numHeads = hp.NumHeads;
        _numKvHeads = hp.NumKvHeads;
        _headsPerKvGroup = hp.NumHeads / hp.NumKvHeads;
        _intermDim = hp.IntermediateDim;

        // Allocate GPU scratch buffers
        _hidden = gpu.Allocate(TensorShape.D1(_embDim));
        _residual = gpu.Allocate(TensorShape.D1(_embDim));
        _normBuf = gpu.Allocate(TensorShape.D1(_embDim));
        _q = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _k = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _v = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        _attnOut = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
        _ffnGate = gpu.Allocate(TensorShape.D1(_intermDim));
        _ffnUp = gpu.Allocate(TensorShape.D1(_intermDim));
        _logits = gpu.Allocate(TensorShape.D1(hp.VocabSize));
        _logitsBuf = new float[hp.VocabSize];

        // Allocate VRAM KV cache
        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[hp.NumLayers];
        _gpuVCache = new Tensor[hp.NumLayers];

        if (_tqEnabled)
        {
            // FP32 window: only tqFp32Window positions
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_tqFp32Window * kvDim));
            }

            // TQ compressed cache: (maxSeqLen - fp32Window) positions
            int maxTqPositions = _maxSeqLen - _tqFp32Window;
            long tqBytesPerPos = (long)_numKvHeads * _tqBlockBytes;
            // Allocate as uint buffer (shader accesses via uint[])
            long tqUintsPerLayer = (maxTqPositions * tqBytesPerPos + 3) / 4;
            _gpuTqKCache = new Tensor[hp.NumLayers];
            _gpuTqVCache = new Tensor[hp.NumLayers];
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuTqKCache[i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
                _gpuTqVCache[i] = gpu.Allocate(TensorShape.D1(tqUintsPerLayer));
            }

            // Upload TQ constants to VRAM
            // Sign patterns: use layer 0, head 0 pattern for all (simplification for GPU)
            // A full implementation would use per-layer-per-head patterns
            var signData = WalshHadamard.GenerateSignPattern(_headDim, 0);
            // Tile to numKvHeads
            var fullSigns = new float[_numKvHeads * _headDim];
            for (int h = 0; h < _numKvHeads; h++)
            {
                var headSigns = WalshHadamard.GenerateSignPattern(_headDim, h);
                headSigns.CopyTo(fullSigns.AsSpan(h * _headDim));
            }
            _gpuSignPatterns = gpu.Upload(fullSigns, TensorShape.D1(fullSigns.Length));

            var centroids = TurboQuantCodebooks.GetCentroids(tqBits, _headDim).ToArray();
            _gpuCodebook = gpu.Upload(centroids, TensorShape.D1(centroids.Length));

            var boundaries = TurboQuantCodebooks.GetBoundaries(tqBits, _headDim).ToArray();
            _gpuBoundaries = gpu.Upload(boundaries, TensorShape.D1(boundaries.Length));

            _rotatedQ = gpu.Allocate(TensorShape.D1(_numHeads * _headDim));
            _evictK = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
            _evictV = gpu.Allocate(TensorShape.D1(_numKvHeads * _headDim));
        }
        else
        {
            // Full FP32 cache: [maxSeqLen, kvDim] per layer
            for (int i = 0; i < hp.NumLayers; i++)
            {
                _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
                _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            }
        }

        // Upload all weights to VRAM
        int L = hp.NumLayers;
        _wAttnNorm = new Tensor[L]; _wFfnNorm = new Tensor[L];
        _wq = new Tensor[L]; _wk = new Tensor[L]; _wv = new Tensor[L]; _wo = new Tensor[L];
        _wGate = new Tensor[L]; _wUp = new Tensor[L]; _wDown = new Tensor[L];

        _hasAttnBias = hp.HasAttnBias;
        if (_hasAttnBias)
        {
            _bq = new Tensor[L]; _bk = new Tensor[L];
            _bv = new Tensor[L]; _bo = new Tensor[L];
        }

        _hasQkNorm = hp.HasQkNorm;
        if (_hasQkNorm)
        {
            _wqNorm = new Tensor[L]; _wkNorm = new Tensor[L];
        }

        Console.Error.Write($"[GpuForwardPass] Uploading {L} layers to VRAM...");
        for (int i = 0; i < L; i++)
        {
            _wAttnNorm[i] = UploadWeight($"blk.{i}.attn_norm.weight");
            _wq[i] = UploadWeight($"blk.{i}.attn_q.weight");
            _wk[i] = UploadWeight($"blk.{i}.attn_k.weight");
            _wv[i] = UploadWeight($"blk.{i}.attn_v.weight");
            _wo[i] = UploadWeight($"blk.{i}.attn_output.weight");
            _wFfnNorm[i] = UploadWeight($"blk.{i}.ffn_norm.weight");
            _wGate[i] = UploadWeight($"blk.{i}.ffn_gate.weight");
            _wUp[i] = UploadWeight($"blk.{i}.ffn_up.weight");
            _wDown[i] = UploadWeight($"blk.{i}.ffn_down.weight");

            if (_hasAttnBias)
            {
                _bq![i] = UploadWeight($"blk.{i}.attn_q.bias");
                _bk![i] = UploadWeight($"blk.{i}.attn_k.bias");
                _bv![i] = UploadWeight($"blk.{i}.attn_v.bias");
                _bo![i] = UploadWeight($"blk.{i}.attn_output.bias");
            }

            if (_hasQkNorm)
            {
                _wqNorm![i] = UploadWeight($"blk.{i}.attn_q_norm.weight");
                _wkNorm![i] = UploadWeight($"blk.{i}.attn_k_norm.weight");
            }

            Console.Error.Write(".");
        }
        _wOutputNorm = UploadWeight("output_norm.weight");

        var outputName = model.FindTensor("output.weight") is not null ? "output.weight" : "token_embd.weight";
        _wOutput = UploadWeight(outputName);

        // Upload embedding table to VRAM — keep quantized for Q4_K to save VRAM
        Console.Error.Write(" emb...");
        var embInfo = model.FindTensor("token_embd.weight")!.Value;
        if (embInfo.DType == DType.Q4_K)
        {
            // Upload raw quantized bytes (reinterpret as uint32 for storage buffer)
            var embData = model.GetTensorData(embInfo);
            int floatCount = embData.Length / 4;
            var raw = new float[floatCount];
            embData.CopyTo(MemoryMarshal.AsBytes(raw.AsSpan()));
            _gpuEmbedding = gpu.Upload(raw, TensorShape.D1(floatCount));
            _embIsQuantized = true;
        }
        else
        {
            // Small vocab or F32: dequantize to F32
            var embData = model.GetTensorData(embInfo);
            var embF32 = new float[(int)embInfo.ElementCount];
            Dequantize.ToFloat32(embData, embF32, embInfo.DType, embInfo.ElementCount);
            _gpuEmbedding = gpu.Upload(embF32, TensorShape.D1(embF32.Length));
            _embIsQuantized = false;
        }

        Console.Error.WriteLine(" done.");
    }

    public Engine.KvCache Cache => _kvCache;
    public int KvLength => _kvLength;

    public void ResetCache()
    {
        _kvLength = 0;
        _tqCompressedLen = 0;
        _fp32WriteIdx = 0;
        _fp32Count = 0;
    }

    /// <summary>
    /// Run one token through the transformer on GPU. Returns logits span (downloaded from VRAM).
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // Record ALL dispatches into ONE command buffer
        _gpu.BeginRecord();

        // Embed token (GPU lookup from cached table — no PCIe transfer)
        if (_embIsQuantized)
            _gpu.EmbedLookupQ4K(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
        else
            _gpu.EmbedLookup(_gpuEmbedding, _hidden, (uint)token, (uint)_embDim);
        _gpu.RecordBarrier();

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Copy hidden → residual + RmsNorm (barrier after both)
            CopyBuffer(_residual, _hidden);
            _gpu.RecordTransferBarrier();

            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // Q/K/V all read normBuf (no conflict between them)
            GpuMatMul(_q, _wq[layer], _normBuf);
            GpuMatMul(_k, _wk[layer], _normBuf);
            GpuMatMul(_v, _wv[layer], _normBuf);
            _gpu.RecordBarrier(); // Q/K/V done → bias + RoPE

            if (_hasAttnBias)
            {
                _gpu.AddInPlace(_q, _bq![layer]);
                _gpu.AddInPlace(_k, _bk![layer]);
                _gpu.AddInPlace(_v, _bv![layer]);
                _gpu.RecordBarrier();
            }

            // Per-head Q/K RMSNorm (Qwen3)
            if (_hasQkNorm)
            {
                _gpu.HeadNorm(_q, _wqNorm![layer], (uint)_numHeads, (uint)_headDim, _hp.RmsNormEps);
                _gpu.HeadNorm(_k, _wkNorm![layer], (uint)_numKvHeads, (uint)_headDim, _hp.RmsNormEps);
                _gpu.RecordBarrier();
            }

            // RoPE on Q and K (write different buffers, no conflict)
            _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta);
            _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta);
            // KV append reads K/V written by RoPE
            _gpu.RecordBarrier();

            if (_tqEnabled)
            {
                int kvDimLocal = _numKvHeads * _headDim;
                long rowBytes = (long)kvDimLocal * sizeof(float);

                // If FP32 window is full, compress the oldest entry before overwriting
                if (_fp32Count >= _tqFp32Window)
                {
                    // Copy oldest FP32 row (at _fp32WriteIdx) to evict buffers
                    CopyBufferRegion(_evictK!, 0, _gpuKCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    CopyBufferRegion(_evictV!, 0, _gpuVCache[layer], (long)_fp32WriteIdx * rowBytes, rowBytes);
                    _gpu.RecordTransferBarrier();

                    // Compress evicted entry into TQ cache
                    _gpu.TqKvAppend(_evictK!, _evictV!, _gpuTqKCache![layer], _gpuTqVCache![layer],
                        _gpuSignPatterns!, _gpuCodebook!, _gpuBoundaries!,
                        (uint)kvDimLocal, (uint)_headDim, (uint)_tqCompressedLen,
                        (uint)_maxSeqLen, (uint)_numKvHeads, (uint)_tqBlockBytes);
                    _gpu.RecordBarrier();
                }

                // Write new K/V into FP32 ring buffer at _fp32WriteIdx
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)kvDimLocal, (uint)_fp32WriteIdx, (uint)_tqFp32Window);
                _gpu.RecordBarrier();

                // Rotate query for TQ attention
                _gpu.TqRotateQuery(_q, _rotatedQ!, _gpuSignPatterns!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim);
                _gpu.RecordBarrier();

                // TQ Attention: handles both compressed and FP32 regions
                uint fp32SeqLen = (uint)Math.Min(_fp32Count + 1, _tqFp32Window);
                _gpu.TqAttention(_q, _rotatedQ!, _gpuTqKCache![layer], _gpuTqVCache![layer],
                    _gpuKCache[layer], _gpuVCache[layer], _attnOut, _gpuCodebook!,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                    (uint)_tqCompressedLen, fp32SeqLen, (uint)_maxSeqLen, (uint)_tqBlockBytes);
            }
            else
            {
                _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                    (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
                _gpu.RecordBarrier();

                _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                    (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                    (uint)(position + 1), (uint)_maxSeqLen);
            }
            _gpu.RecordBarrier(); // attnOut done → output projection

            GpuMatMul(_hidden, _wo[layer], _attnOut);
            if (_hasAttnBias)
            {
                _gpu.RecordBarrier();
                _gpu.AddInPlace(_hidden, _bo![layer]);
            }
            _gpu.RecordBarrier(); // hidden written → add

            _gpu.AddInPlace(_hidden, _residual);
            // hidden done → FFN starts (copy + norm)

            CopyBuffer(_residual, _hidden);
            _gpu.RecordTransferBarrier();

            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // gate and up both read normBuf (no conflict)
            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp, _wUp[layer], _normBuf);
            _gpu.RecordBarrier(); // gate/up done → SiLU

            _gpu.SiLuMul(_ffnGate, _ffnUp);
            _gpu.RecordBarrier(); // SiLU done → down projection

            GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            _gpu.RecordBarrier(); // hidden written → add

            _gpu.AddInPlace(_hidden, _residual);
            // No barrier needed here — next layer starts with CopyBuffer which is a transfer
        }

        // Update TQ ring buffer state (after all layers used the same indices)
        if (_tqEnabled)
        {
            if (_fp32Count >= _tqFp32Window)
                _tqCompressedLen++;

            _fp32WriteIdx = (_fp32WriteIdx + 1) % _tqFp32Window;
            if (_fp32Count < _tqFp32Window)
                _fp32Count++;
        }

        // Final norm + output projection
        _gpu.RecordBarrier(); // last layer's AddInPlace → final norm
        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_logits, _wOutput, _hidden);

        // Submit ALL dispatches at once
        _gpu.EndRecordAndSubmit();

        // Download logits to CPU (reuse pre-allocated buffer)
        _gpu.Download(_logits, _logitsBuf);
        return _logitsBuf;
    }

    // ================================================================
    //  Helpers
    // ================================================================

    private void GpuMatMul(Tensor output, Tensor weights, Tensor input)
    {
        var dtype = _weightDTypes.GetValueOrDefault(weights.Handle, DType.Q4_K);
        _gpu.MatMul(output, weights, input, dtype);
    }

    private void EmbedToken(int token)
    {
        // Dequantize embedding row on CPU, upload to GPU
        var info = _model.FindTensor("token_embd.weight")!.Value;
        var data = _model.GetTensorData(info);
        int bytesPerRow = (_embDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        int rowOffset = token * bytesPerRow;

        var embedding = new float[_embDim];
        Dequantize.ToFloat32(
            data.Slice(rowOffset, bytesPerRow),
            embedding, info.DType, _embDim);

        UploadToExisting(_hidden, embedding);
    }

    private void UploadToExisting(Tensor gpuTensor, ReadOnlySpan<float> data)
    {
        ulong byteSize = (ulong)(data.Length * sizeof(float));
        using var staging = GpuBuffer.CreateStaging(_gpu, byteSize,
            Vortice.Vulkan.VkBufferUsageFlags.TransferSrc);
        float* mapped = (float*)staging.Map();
        data.CopyTo(new Span<float>(mapped, data.Length));
        staging.Unmap();

        // Copy staging → device buffer via command buffer
        var vkd = _gpu.Vkd;
        var cmd = _gpu.TransferCmd;
        Vortice.Vulkan.VkCommandBufferBeginInfo beginInfo = new()
        {
            flags = Vortice.Vulkan.VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        vkd.vkBeginCommandBuffer(cmd, &beginInfo).CheckResult();

        Vortice.Vulkan.VkBufferCopy region = new() { size = byteSize };
        vkd.vkCmdCopyBuffer(cmd, staging.Buffer, _gpu.GetBuffer(gpuTensor).Buffer, 1, &region);

        vkd.vkEndCommandBuffer(cmd).CheckResult();
        Vortice.Vulkan.VkSubmitInfo submit = new()
        {
            commandBufferCount = 1,
            pCommandBuffers = &cmd,
        };
        var fence = _gpu.Fence;
        vkd.vkResetFences(1, &fence).CheckResult();
        vkd.vkQueueSubmit(_gpu.ComputeQueue, 1, &submit, fence).CheckResult();
        vkd.vkWaitForFences(1, &fence, true, ulong.MaxValue).CheckResult();
    }

    /// <summary>Record a buffer copy into the current command buffer (must be in recording mode).</summary>
    private void CopyBuffer(Tensor dst, Tensor src)
    {
        var srcBuf = _gpu.GetBuffer(src);
        var dstBuf = _gpu.GetBuffer(dst);
        VkBufferCopy region = new() { size = srcBuf.Size };
        _gpu.Vkd.vkCmdCopyBuffer(_gpu.TransferCmd, srcBuf.Buffer, dstBuf.Buffer, 1, &region);
    }

    /// <summary>Copy a sub-region from src to dst (both in VRAM).</summary>
    private void CopyBufferRegion(Tensor dst, long dstOffsetBytes, Tensor src, long srcOffsetBytes, long sizeBytes)
    {
        var srcBuf = _gpu.GetBuffer(src);
        var dstBuf = _gpu.GetBuffer(dst);
        VkBufferCopy region = new() { srcOffset = (ulong)srcOffsetBytes, dstOffset = (ulong)dstOffsetBytes, size = (ulong)sizeBytes };
        _gpu.Vkd.vkCmdCopyBuffer(_gpu.TransferCmd, srcBuf.Buffer, dstBuf.Buffer, 1, &region);
    }

    // Track quantization type per weight tensor for MatMul dispatch
    private readonly Dictionary<nint, DType> _weightDTypes = new();

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        Tensor result;
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            result = _gpu.Upload(floats, TensorShape.D1(floats.Length));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        else if (info.DType == DType.Q4_K || info.DType == DType.Q6_K)
        {
            // Upload raw quantized bytes (reinterpret as floats for storage buffer)
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _weightDTypes[result.Handle] = info.DType;
        }
        else
        {
            // Other types: dequantize to F32 on CPU
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        return result;
    }

    /// <summary>
    /// Estimates max context when using TurboQuant compressed KV cache.
    /// TQ3 uses ~52 bytes per head per position vs 512 bytes (128 floats) for FP32.
    /// </summary>
    public static int EstimateMaxContextTq(GgufModel model, VulkanBackend gpu, ModelHyperparams hp,
        int fp32WindowSize = 256, int bits = 3)
    {
        long vramBytes = (long)gpu.VramBytes;
        int headDim = hp.EmbeddingDim / hp.NumHeads;
        int blockSize = TurboQuantOps.BlockSize(bits, headDim);

        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += (t.ByteSize + 3) & ~3L;

        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        long reserved = Math.Max(vramBytes / 5, 1024L * 1024 * 1024);
        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024;

        // FP32 window: 2 * layers * kvDim * sizeof(float) per position
        long fp32Bytes = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float) * fp32WindowSize;

        // TQ: 2 * layers * numKvHeads * blockSize per position
        long tqBytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * blockSize;

        long availableForTq = available - fp32Bytes;
        if (availableForTq <= 0) availableForTq = 64L * 1024 * 1024;

        int maxTqPositions = (int)(availableForTq / tqBytesPerToken);
        int maxCtx = Math.Clamp(maxTqPositions + fp32WindowSize, 512, hp.ContextLength);

        return maxCtx;
    }

    private static int EstimateMaxContext(GgufModel model, VulkanBackend gpu, ModelHyperparams hp)
    {
        long vramBytes = (long)gpu.VramBytes;

        // Estimate weight VRAM: raw quantized bytes padded to 4-byte alignment per tensor
        long weightBytes = 0;
        foreach (var t in model.Tensors)
            weightBytes += (t.ByteSize + 3) & ~3L;

        // Scratch buffers (F32): hidden, residual, norm, Q, K, V, attnOut, ffnGate, ffnUp, logits
        int headDim = hp.EmbeddingDim / hp.NumHeads;
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);

        // Reserve for Vulkan overhead, staging buffers, OS/desktop compositor
        // Use 20% of VRAM or 1 GB, whichever is larger
        long reserved = Math.Max(vramBytes / 5, 1024L * 1024 * 1024);

        long available = vramBytes - weightBytes - scratchBytes - reserved;
        if (available <= 0) available = 64L * 1024 * 1024; // minimum fallback

        // KV cache: 2 (K+V) * numLayers * numKvHeads * headDim * sizeof(float) per token
        long bytesPerToken = 2L * hp.NumLayers * hp.NumKvHeads * headDim * sizeof(float);

        int maxCtx = (int)(available / bytesPerToken);
        maxCtx = Math.Clamp(maxCtx, 512, hp.ContextLength);

        return maxCtx;
    }

    public void Dispose()
    {
        _gpu.Free(_hidden); _gpu.Free(_residual); _gpu.Free(_normBuf);
        _gpu.Free(_q); _gpu.Free(_k); _gpu.Free(_v); _gpu.Free(_attnOut);
        _gpu.Free(_ffnGate); _gpu.Free(_ffnUp); _gpu.Free(_logits);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_wAttnNorm[i]); _gpu.Free(_wFfnNorm[i]);
            _gpu.Free(_wq[i]); _gpu.Free(_wk[i]); _gpu.Free(_wv[i]); _gpu.Free(_wo[i]);
            _gpu.Free(_wGate[i]); _gpu.Free(_wUp[i]); _gpu.Free(_wDown[i]);

            if (_hasAttnBias)
            {
                _gpu.Free(_bq![i]); _gpu.Free(_bk![i]);
                _gpu.Free(_bv![i]); _gpu.Free(_bo![i]);
            }

            if (_hasQkNorm)
            {
                _gpu.Free(_wqNorm![i]); _gpu.Free(_wkNorm![i]);
            }
        }
        _gpu.Free(_wOutputNorm); _gpu.Free(_wOutput); _gpu.Free(_gpuEmbedding);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_gpuKCache[i]);
            _gpu.Free(_gpuVCache[i]);
        }

        if (_tqEnabled)
        {
            for (int i = 0; i < _hp.NumLayers; i++)
            {
                _gpu.Free(_gpuTqKCache![i]);
                _gpu.Free(_gpuTqVCache![i]);
            }
            _gpu.Free(_gpuSignPatterns!);
            _gpu.Free(_gpuCodebook!);
            _gpu.Free(_gpuBoundaries!);
            _gpu.Free(_rotatedQ!);
        }

        _kvCache.Dispose();
    }
}
