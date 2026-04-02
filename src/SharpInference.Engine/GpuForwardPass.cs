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

    // KV cache: CPU-side for now (attention runs on CPU)
    private readonly Engine.KvCache _kvCache;

    private readonly int _embDim, _headDim, _numHeads, _numKvHeads, _headsPerKvGroup, _intermDim;

    public GpuForwardPass(GgufModel model, VulkanBackend gpu, ModelHyperparams hp)
    {
        _model = model;
        _gpu = gpu;
        _hp = hp;
        _kvCache = new Engine.KvCache(hp);

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

    /// <summary>
    /// Run one token through the transformer on GPU. Returns logits span (downloaded from VRAM).
    /// </summary>
    public ReadOnlySpan<float> Forward(int token, int position)
    {
        // 1. Embed token (CPU dequant → upload to GPU)
        EmbedToken(token);

        // 2. Transformer layers
        for (int layer = 0; layer < _hp.NumLayers; layer++)
        {
            // Copy hidden → residual (GPU)
            CopyBuffer(_residual, _hidden);

            // Pre-attention RmsNorm (GPU)
            _gpu.RmsNorm(_normBuf, _hidden, _wAttnNorm[layer], _hp.RmsNormEps);

            // Q/K/V projections (GPU MatVec with Q4_K dequant)
            _gpu.MatMul(_q, _wq[layer], _normBuf);
            _gpu.MatMul(_k, _wk[layer], _normBuf);
            _gpu.MatMul(_v, _wv[layer], _normBuf);

            // RoPE (GPU)
            _gpu.RoPE(_q, position, _headDim, _hp.RopeTheta);
            _gpu.RoPE(_k, position, _headDim, _hp.RopeTheta);

            // Attention: download Q/K/V, run on CPU, upload result
            // (GPU attention with causal masking + growing KV cache is complex — CPU fallback for now)
            AttentionCpuFallback(layer, position);

            // Output projection (GPU)
            _gpu.MatMul(_hidden, _wo[layer], _attnOut);

            // Residual add (GPU)
            _gpu.AddInPlace(_hidden, _residual);

            // Save residual for FFN
            CopyBuffer(_residual, _hidden);

            // Pre-FFN RmsNorm (GPU)
            _gpu.RmsNorm(_normBuf, _hidden, _wFfnNorm[layer], _hp.RmsNormEps);

            // SwiGLU FFN (GPU)
            _gpu.MatMul(_ffnGate, _wGate[layer], _normBuf);
            _gpu.MatMul(_ffnUp, _wUp[layer], _normBuf);
            _gpu.SiLuMul(_ffnGate, _ffnUp);
            _gpu.MatMul(_hidden, _wDown[layer], _ffnGate);

            // Residual add (GPU)
            _gpu.AddInPlace(_hidden, _residual);
        }

        _kvCache.IncrementPosition();

        // Final norm + output projection (GPU)
        _gpu.RmsNorm(_hidden, _hidden, _wOutputNorm, _hp.RmsNormEps);
        _gpu.MatMul(_logits, _wOutput, _hidden);

        // Download logits to CPU
        var logits = new float[_hp.VocabSize];
        _gpu.Download(_logits, logits);
        return logits;
    }

    // ================================================================
    //  Attention (CPU fallback — download K/V/Q, compute, upload result)
    // ================================================================

    private void AttentionCpuFallback(int layer, int position)
    {
        int qDim = _numHeads * _headDim;
        int kvDim = _numKvHeads * _headDim;
        int seqLen = position + 1;

        // Download Q, K, V from GPU
        var qBuf = new float[qDim];
        var kBuf = new float[kvDim];
        var vBuf = new float[kvDim];
        _gpu.Download(_q, qBuf);
        _gpu.Download(_k, kBuf);
        _gpu.Download(_v, vBuf);

        // Append K, V to CPU KV cache
        _kvCache.Append(layer, kBuf, vBuf);

        // Compute attention on CPU
        var attnOut = new float[qDim];
        float scale = 1.0f / MathF.Sqrt(_headDim);

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / _headsPerKvGroup;
            var scores = new float[seqLen];

            // Q·K scores
            for (int t = 0; t < seqLen; t++)
            {
                float dot = 0;
                float* kVec = _kvCache.KeyAt(layer, t) + kvHead * _headDim;
                for (int d = 0; d < _headDim; d++)
                    dot += qBuf[h * _headDim + d] * kVec[d];
                scores[t] = dot * scale;
            }

            // Softmax
            float max = scores.Max();
            float sum = 0;
            for (int t = 0; t < seqLen; t++)
            {
                scores[t] = MathF.Exp(scores[t] - max);
                sum += scores[t];
            }
            for (int t = 0; t < seqLen; t++) scores[t] /= sum;

            // Weighted sum of values
            for (int t = 0; t < seqLen; t++)
            {
                float w = scores[t];
                float* vVec = _kvCache.ValueAt(layer, t) + kvHead * _headDim;
                for (int d = 0; d < _headDim; d++)
                    attnOut[h * _headDim + d] += w * vVec[d];
            }
        }

        // Upload attention output back to GPU
        UploadToExisting(_attnOut, attnOut);
    }

    // ================================================================
    //  Helpers
    // ================================================================

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

    private void CopyBuffer(Tensor dst, Tensor src)
    {
        var srcBuf = _gpu.GetBuffer(src);
        var dstBuf = _gpu.GetBuffer(dst);
        ulong size = srcBuf.Size;

        var vkd = _gpu.Vkd;
        var cmd = _gpu.TransferCmd;
        Vortice.Vulkan.VkCommandBufferBeginInfo beginInfo = new()
        {
            flags = Vortice.Vulkan.VkCommandBufferUsageFlags.OneTimeSubmit,
        };
        vkd.vkBeginCommandBuffer(cmd, &beginInfo).CheckResult();

        Vortice.Vulkan.VkBufferCopy region = new() { size = size };
        vkd.vkCmdCopyBuffer(cmd, srcBuf.Buffer, dstBuf.Buffer, 1, &region);

        vkd.vkEndCommandBuffer(cmd).CheckResult();
        Vortice.Vulkan.VkSubmitInfo submit = new()
        {
            commandBufferCount = 1,
            pCommandBuffers = &cmd,
        };
        vkd.vkQueueSubmit(_gpu.ComputeQueue, 1, &submit, Vortice.Vulkan.VkFence.Null).CheckResult();
        vkd.vkQueueWaitIdle(_gpu.ComputeQueue);
    }

    private Tensor UploadWeight(string name)
    {
        var info = _model.FindTensor(name)
            ?? throw new InvalidOperationException($"Missing tensor: {name}");
        var data = _model.GetTensorData(info);

        // Upload raw bytes as floats (reinterpret for storage buffer)
        if (info.DType == DType.Float32)
        {
            var floats = MemoryMarshal.Cast<byte, float>(data);
            return _gpu.Upload(floats, TensorShape.D1(floats.Length));
        }

        // Quantized: upload raw bytes (reinterpret as float array for the storage buffer)
        int floatCount = data.Length / 4;
        if (data.Length % 4 != 0)
            floatCount++; // pad
        var rawFloats = new float[floatCount];
        data.CopyTo(MemoryMarshal.AsBytes(rawFloats.AsSpan()));
        return _gpu.Upload(rawFloats, TensorShape.D1(floatCount));
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

        _kvCache.Dispose();
    }
}
