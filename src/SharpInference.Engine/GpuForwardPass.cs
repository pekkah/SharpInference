using System.Runtime.InteropServices;
using SharpInference.Core;
using SharpInference.Cpu;
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

    // GPU weight tensors (Q4_K/Q6_K bytes uploaded to VRAM)
    private readonly Tensor[] _wAttnNorm;
    private readonly Tensor[] _wq, _wk, _wv, _wo;
    private readonly Tensor[] _wFfnNorm;
    private readonly Tensor[] _wGate, _wUp, _wDown;
    private readonly Tensor _wOutputNorm;
    private readonly Tensor _wOutput;

    // KV cache in VRAM: per-layer K and V buffers [maxSeqLen, kvDim]
    private readonly Tensor[] _gpuKCache;  // per layer
    private readonly Tensor[] _gpuVCache;  // per layer
    private readonly int _maxSeqLen;
    private int _kvLength; // current sequence length in cache

    // CPU KV cache kept for fallback (not used when GPU attention works)
    private readonly Engine.KvCache _kvCache;

    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim;

    public GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _kvCache = new Engine.KvCache(hp);
        _maxSeqLen = hp.ContextLength;

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

        // Allocate VRAM KV cache: per-layer [maxSeqLen, kvDim]
        int kvDim = _numKvHeads * _headDim;
        _gpuKCache = new Tensor[hp.NumLayers];
        _gpuVCache = new Tensor[hp.NumLayers];
        for (int i = 0; i < hp.NumLayers; i++)
        {
            _gpuKCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
            _gpuVCache[i] = gpu.Allocate(TensorShape.D1((long)_maxSeqLen * kvDim));
        }

        // Upload all weights to VRAM
        int L = hp.NumLayers;
        _wAttnNorm = new Tensor[L]; _wFfnNorm = new Tensor[L];
        _wq = new Tensor[L]; _wk = new Tensor[L]; _wv = new Tensor[L]; _wo = new Tensor[L];
        _wGate = new Tensor[L]; _wUp = new Tensor[L]; _wDown = new Tensor[L];

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
            Console.Error.Write(".");
        }
        _wOutputNorm = UploadWeight("output_norm.weight");

        var outputName = model.FindTensor("output.weight") is not null ? "output.weight" : "token_embd.weight";
        _wOutput = UploadWeight(outputName);
        Console.Error.WriteLine(" done.");
    }

    public Engine.KvCache Cache => _kvCache;
    public int KvLength => _kvLength;

    public void ResetCache() => _kvLength = 0;

    /// <summary>
    /// Run one token through the transformer on GPU. Returns logits span (downloaded from VRAM).
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Embed token (CPU dequant → upload to GPU)
        EmbedToken(token);

        // 2. Record ALL transformer layers into ONE command buffer
        _gpu.BeginRecord();

        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Copy hidden → residual
            CopyBuffer(_residual, _hidden);
            _gpu.RecordTransferBarrier();

            // Pre-attention RmsNorm
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            // Q/K/V projections
            GpuMatMul(_q, _wq[layer], _normBuf);
            GpuMatMul(_k, _wk[layer], _normBuf);
            GpuMatMul(_v, _wv[layer], _normBuf);
            _gpu.RecordBarrier();

            // RoPE
            _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta);
            _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta);
            _gpu.RecordBarrier();

            // KV cache append
            _gpu.KvAppend(_k, _v, _gpuKCache[layer], _gpuVCache[layer],
                (uint)(_numKvHeads * _headDim), (uint)position, (uint)_maxSeqLen);
            _gpu.RecordBarrier();

            // Attention
            _gpu.Attention(_q, _gpuKCache[layer], _gpuVCache[layer], _attnOut,
                (uint)_numHeads, (uint)_numKvHeads, (uint)_headDim,
                (uint)(position + 1), (uint)_maxSeqLen);
            _gpu.RecordBarrier();

            // Output projection
            GpuMatMul(_hidden, _wo[layer], _attnOut);
            _gpu.RecordBarrier();

            // Residual add
            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();

            // FFN: save residual, norm, gate/up, SiLU, down, residual add
            CopyBuffer(_residual, _hidden);
            _gpu.RecordTransferBarrier();

            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);
            _gpu.RecordBarrier();

            GpuMatMul(_ffnGate, _wGate[layer], _normBuf);
            GpuMatMul(_ffnUp, _wUp[layer], _normBuf);
            _gpu.RecordBarrier();

            _gpu.SiLuMul(_ffnGate, _ffnUp);
            _gpu.RecordBarrier();

            GpuMatMul(_hidden, _wDown[layer], _ffnGate);
            _gpu.RecordBarrier();

            _gpu.AddInPlace(_hidden, _residual);
            _gpu.RecordBarrier();
        }

        // Final norm + output projection
        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.RecordBarrier();
        GpuMatMul(_logits, _wOutput, _hidden);

        // Submit ALL dispatches at once — single GPU submission for the entire token
        _gpu.EndRecordAndSubmit();

        // Download logits to CPU
        var logits = new float[_hp.VocabSize];
        _gpu.Download(_logits, logits);
        return logits;
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
        vkd.vkQueueSubmit(_gpu.ComputeQueue, 1, &submit, Vortice.Vulkan.VkFence.Null).CheckResult();
        vkd.vkQueueWaitIdle(_gpu.ComputeQueue);
    }

    /// <summary>Record a buffer copy into the current command buffer (must be in recording mode).</summary>
    private void CopyBuffer(Tensor dst, Tensor src)
    {
        var srcBuf = _gpu.GetBuffer(src);
        var dstBuf = _gpu.GetBuffer(dst);
        VkBufferCopy region = new() { size = srcBuf.Size };
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
        else if (info.DType == DType.Q4_K)
        {
            // Upload raw Q4_K bytes (reinterpret as floats for storage buffer)
            int floatCount = data.Length / 4;
            var rawFloats = new float[floatCount];
            data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
            result = _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
            _weightDTypes[result.Handle] = DType.Q4_K;
        }
        else
        {
            // Q6_K and other types: dequantize to F32 on CPU, upload as F32
            // (GPU Q6_K shader TBD — this is correct but slower)
            int count = (int)info.ElementCount;
            var f32 = new float[count];
            Dequantize.ToFloat32(data, f32, info.DType, count);
            result = _gpu.Upload(f32, TensorShape.D1(count));
            _weightDTypes[result.Handle] = DType.Float32;
        }
        return result;
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
        }
        _gpu.Free(_wOutputNorm); _gpu.Free(_wOutput);

        for (int i = 0; i < _hp.NumLayers; i++)
        {
            _gpu.Free(_gpuKCache[i]);
            _gpu.Free(_gpuVCache[i]);
        }

        _kvCache.Dispose();
    }
}
