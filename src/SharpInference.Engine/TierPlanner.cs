using SharpInference.Core;
using SharpInference.TurboQuant;

namespace SharpInference.Engine;

/// <summary>
/// Plans how to split model layers between GPU and CPU based on available hardware.
/// Greedily assigns layers to GPU starting from layer 0, packing as many as fit in VRAM.
/// </summary>
public static class TierPlanner
{
    /// <summary>
    /// Compute optimal layer placement for a given model and hardware profile.
    /// </summary>
    public static LayerPlacement Plan(GgufModel model, ModelHyperparams hp,
        HardwareProfile hardware, bool turboQuant = false, int tqBits = 3,
        int requestedCtxSize = 0)
    {
        if (hardware.VramBytes <= 0)
            return new LayerPlacement(0, hp.NumLayers, 0, 0, requestedCtxSize > 0 ? requestedCtxSize : hp.ContextLength);

        long vramTotal = hardware.VramBytes;
        int headDim = hp.EmbeddingDim / hp.NumHeads;

        // Reserve for Vulkan overhead + scratch buffers
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + hp.IntermediateDim * 2 + hp.VocabSize) * sizeof(float);
        long reserved = Math.Max(vramTotal / 10, 512L * 1024 * 1024); // 10% or 512 MB min
        long vramBudget = vramTotal - scratchBytes - reserved;

        // Priority 1: Embedding + output projection (always on GPU)
        long embBytes = MeasureTensorBytes(model, "token_embd.weight");
        long outputBytes = model.FindTensor("output.weight") != null
            ? MeasureTensorBytes(model, "output.weight")
            : 0; // tied embeddings — shared with token_embd
        long outputNormBytes = MeasureTensorBytes(model, "output_norm.weight");

        long fixedGpuBytes = embBytes + outputBytes + outputNormBytes;
        vramBudget -= fixedGpuBytes;
        if (vramBudget < 0) vramBudget = 0;

        // Measure per-layer weight bytes
        long perLayerBytes = MeasureLayerBytes(model, hp, 0);

        // Priority 2: Assign layers to GPU, greedily
        int gpuLayers = 0;
        long gpuWeightBytes = fixedGpuBytes;
        for (int i = 0; i < hp.NumLayers; i++)
        {
            long layerBytes = MeasureLayerBytes(model, hp, i);
            if (vramBudget >= layerBytes)
            {
                vramBudget -= layerBytes;
                gpuWeightBytes += layerBytes;
                gpuLayers++;
            }
            else
            {
                break; // layers are contiguous from 0
            }
        }

        // KV cache: remaining VRAM budget
        long kvBytesPerToken;
        if (turboQuant)
        {
            int blockSize = TurboQuantOps.BlockSize(tqBits, headDim);
            // FP32 window (256 tokens) + TQ compressed rest
            long fp32WindowBytes = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float) * 256;
            long tqPerToken = 2L * gpuLayers * hp.NumKvHeads * blockSize;
            // Approximate: assume most tokens are TQ-compressed
            kvBytesPerToken = tqPerToken;
            vramBudget -= fp32WindowBytes;
        }
        else
        {
            kvBytesPerToken = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float);
        }

        int gpuCtxSize;
        if (requestedCtxSize > 0)
        {
            gpuCtxSize = requestedCtxSize;
        }
        else if (kvBytesPerToken > 0 && vramBudget > 0)
        {
            gpuCtxSize = (int)(vramBudget / kvBytesPerToken);
            gpuCtxSize = Math.Clamp(gpuCtxSize, 512, hp.ContextLength);
        }
        else
        {
            gpuCtxSize = Math.Min(2048, hp.ContextLength);
        }

        long gpuKvBytes = kvBytesPerToken * gpuCtxSize;

        return new LayerPlacement(
            gpuLayers,
            hp.NumLayers - gpuLayers,
            gpuWeightBytes,
            gpuKvBytes,
            gpuCtxSize);
    }

    private static long MeasureTensorBytes(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        return info is not null ? ((info.Value.ByteSize + 3) & ~3L) : 0;
    }

    private static long MeasureLayerBytes(GgufModel model, ModelHyperparams hp, int layer)
    {
        long total = 0;
        string[] suffixes =
        [
            "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
            "ffn_down.weight"
        ];

        foreach (var suffix in suffixes)
            total += MeasureTensorBytes(model, $"blk.{layer}.{suffix}");

        // Optional biases
        if (hp.HasAttnBias)
        {
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_q.bias");
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_k.bias");
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_v.bias");
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_output.bias");
        }

        if (hp.HasQkNorm)
        {
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_q_norm.weight");
            total += MeasureTensorBytes(model, $"blk.{layer}.attn_k_norm.weight");
        }

        return total;
    }
}

/// <summary>Result of tier placement analysis.</summary>
public sealed record LayerPlacement(
    int GpuLayers,
    int CpuLayers,
    long GpuWeightBytes,
    long GpuKvBytes,
    int RecommendedCtxSize)
{
    public string Summary()
    {
        double gpuWeightMB = GpuWeightBytes / (1024.0 * 1024);
        double gpuKvMB = GpuKvBytes / (1024.0 * 1024);
        return $"GPU: {GpuLayers} layers ({gpuWeightMB:F0} MB weights, {gpuKvMB:F0} MB KV), CPU: {CpuLayers} layers, ctx: {RecommendedCtxSize}";
    }
}
