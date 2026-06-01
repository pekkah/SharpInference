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
        int requestedCtxSize = 0, int tqFp32Window = 256)
    {
        if (hardware.VramBytes <= 0)
            return new LayerPlacement(0, hp.NumLayers, 0, 0, requestedCtxSize > 0 ? requestedCtxSize : hp.ContextLength);

        long vramTotal = hardware.VramBytes;
        int headDim = hp.HeadDim;

        // Reserve for Vulkan overhead + scratch buffers
        long scratchBytes = (long)(hp.EmbeddingDim * 3 + hp.NumHeads * headDim
            + hp.NumKvHeads * headDim * 2 + hp.NumHeads * headDim
            + Math.Max(hp.IntermediateDim, hp.IsMoE ? hp.ExpertIntermediateDim : hp.IntermediateDim) * 2
            + hp.VocabSize + (hp.IsMoE ? hp.NumExperts + hp.EmbeddingDim * 2 : 0)) * sizeof(float);
        long reserved = Math.Max(vramTotal / 10, 512L * 1024 * 1024); // 10% or 512 MB min
        long vramBudget = vramTotal - scratchBytes - reserved;

        // Priority 1: Embedding + output projection (always on GPU)
        bool cpuFixedWeights = ShouldKeepFixedWeightsOnCpu(
            model.FindTensor("token_embd.weight")!.Value,
            model.FindTensor("output.weight"));
        long embBytes = cpuFixedWeights ? 0 : MeasureGpuEmbeddingBytes(model, "token_embd.weight");
        long outputBytes = cpuFixedWeights
            ? 0
            : model.FindTensor("output.weight") != null
                ? MeasureGpuTensorBytes(model, "output.weight")
                : 0; // tied embeddings — shared with token_embd
        long outputNormBytes = cpuFixedWeights ? 0 : MeasureGpuTensorBytes(model, "output_norm.weight");

        // Gemma 4: per_layer_token_embd.weight (PLE table, ~4.2 GB at Q8_0) stays
        // on CPU unconditionally — it must NOT be counted against the GPU weight
        // budget. Without this branch the planner reserves ~4 GB of phantom VRAM
        // and slashes the GPU layer count or context window on cards that could
        // otherwise hold the full Gemma 4 E4B model.
        // (Note: MeasureLayerBytes only iterates the standard per-layer suffixes,
        // so the smaller per-layer PLE tensors — inp_gate / proj / post_norm —
        // are already implicitly excluded.)
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

        int gpuCtxSize;
        int autoCtxCap = Math.Min(hp.ContextLength, 32768);
        if (requestedCtxSize > 0)
        {
            gpuCtxSize = requestedCtxSize;
        }
        else if (gpuLayers > 0 && turboQuant)
        {
            int fp32Window = Math.Min(tqFp32Window, autoCtxCap);
            int tqBlockSize = TurboQuantOps.BlockSize(tqBits, headDim);
            long fp32WindowBytes = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float) * fp32Window;
            long tqBytesPerToken = 2L * gpuLayers * hp.NumKvHeads * tqBlockSize;

            long availableForTq = vramBudget - fp32WindowBytes;
            if (availableForTq <= 0) availableForTq = 64L * 1024 * 1024;

            int maxTqPositions = tqBytesPerToken > 0 ? (int)(availableForTq / tqBytesPerToken) : 0;
            gpuCtxSize = Math.Clamp(maxTqPositions + fp32Window, 512, autoCtxCap);
        }
        else if (gpuLayers > 0 && vramBudget > 0)
        {
            long kvBytesPerToken = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float);
            gpuCtxSize = (int)(vramBudget / kvBytesPerToken);
            gpuCtxSize = Math.Clamp(gpuCtxSize, 512, autoCtxCap);
        }
        else
        {
            gpuCtxSize = Math.Min(2048, autoCtxCap);
        }

        long gpuKvBytes;
        if (gpuLayers == 0)
        {
            gpuKvBytes = 0;
        }
        else if (turboQuant)
        {
            int fp32Window = Math.Min(tqFp32Window, gpuCtxSize);
            int tqBlockSize = TurboQuantOps.BlockSize(tqBits, headDim);
            long fp32WindowBytes = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float) * fp32Window;
            long tqBytesPerToken = 2L * gpuLayers * hp.NumKvHeads * tqBlockSize;
            long tqPositions = Math.Max(0, gpuCtxSize - fp32Window);
            gpuKvBytes = fp32WindowBytes + tqBytesPerToken * tqPositions;
        }
        else
        {
            long kvBytesPerToken = 2L * gpuLayers * hp.NumKvHeads * headDim * sizeof(float);
            gpuKvBytes = kvBytesPerToken * gpuCtxSize;
        }

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

    // Embedding-aware: only Q4_K embed stays quantized on GPU (the only quantized
    // EmbedLookup shader). Anything else gets dequantized to F32 at upload time,
    // so the post-upload footprint is 4 bytes per element regardless of source dtype.
    private static long MeasureGpuEmbeddingBytes(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        if (info is null) return 0;
        if (info.Value.DType == DType.Q4_K)
            return (info.Value.ByteSize + 3) & ~3L;
        return info.Value.ElementCount * sizeof(float);
    }

    private static long MeasureGpuTensorBytes(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        return info is not null ? EstimateGpuTensorBytes(info.Value) : 0;
    }

    private static long MeasureLayerBytes(GgufModel model, ModelHyperparams hp, int layer)
    {
        long total = 0;
        string[] suffixes = hp.IsMoE
            ?
            [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate_inp.weight",
                "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
            ]
            :
            [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                "ffn_down.weight"
            ];

        foreach (var suffix in suffixes)
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.{suffix}");

        if (hp.IsMoE && hp.HasSharedExpert)
        {
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.ffn_gate_shexp.weight");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.ffn_up_shexp.weight");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.ffn_down_shexp.weight");
        }

        // Optional biases
        if (hp.HasAttnBias)
        {
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_q.bias");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_k.bias");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_v.bias");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_output.bias");
        }

        if (hp.HasQkNorm && !hp.UseL2QkNorm)
        {
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_q_norm.weight");
            total += MeasureGpuTensorBytes(model, $"blk.{layer}.attn_k_norm.weight");
        }

        return total;
    }

    private static long EstimateGpuTensorBytes(GgufTensorInfo tensor)
    {
        if (tensor.DType == DType.Float32 || tensor.DType == DType.Q4_K || tensor.DType == DType.Q6_K)
            return (tensor.ByteSize + 3) & ~3L;

        return tensor.ElementCount * sizeof(float);
    }

    private static bool ShouldKeepFixedWeightsOnCpu(GgufTensorInfo embedding, GgufTensorInfo? output)
    {
        const long maxStorageBufferBytes = 2L * 1024 * 1024 * 1024 - 1;
        long embBytes = embedding.DType == DType.Q4_K
            ? (embedding.ByteSize + 3) & ~3L
            : embedding.ElementCount * sizeof(float);
        if (embBytes > maxStorageBufferBytes)
            return true;
        if (output is not null && EstimateGpuTensorBytes(output.Value) > maxStorageBufferBytes)
            return true;
        return false;
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
