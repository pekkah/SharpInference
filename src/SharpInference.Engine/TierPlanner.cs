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
    /// <param name="pinGpuLayers">
    /// When non-null, pins the GPU trunk-layer count (e.g. the user's explicit <c>-g N</c>) instead
    /// of greedily auto-packing, and prices the trunk weights, KV, and <see cref="LayerPlacement.
    /// ExpertCacheBudgetBytes"/> for THAT split. Callers must use this rather than a
    /// <c>placement with { GpuLayers = N }</c> override, which would leave the expert-cache budget
    /// (and KV/weight bytes) computed for the auto split — a stale value the MoE CPU-vs-SLRU
    /// auto-decision then reads (issue #224). Clamped to <c>[0, NumLayers]</c>.
    /// </param>
    public static LayerPlacement Plan(GgufModel model, ModelHyperparams hp,
        HardwareProfile hardware, bool turboQuant = false, int tqBits = 3,
        int requestedCtxSize = 0, int tqFp32Window = 256, DType kvDtype = DType.Float32,
        int? pinGpuLayers = null)
    {
        if (hardware.VramBytes <= 0)
            return new LayerPlacement(0, hp.NumLayers, 0, 0, requestedCtxSize > 0 ? requestedCtxSize : hp.ContextLength);

        long vramTotal = hardware.VramBytes;
        int headDim = hp.HeadDim;

        // Gemma-4-class models have per-layer KV shape (SWA window-rings, KV-share
        // aliasing, per-layer head_dim / kv-heads). Their KV footprint is a piecewise
        // function of context, so price it via the runtime allocator's own helper
        // (CudaForwardPass.EstimateKvCacheBytes) rather than the flat per-token formula —
        // the two can't drift. Uniform-attention models keep the flat formula.
        bool perLayerKvShape = hp.LayerHeadDim is not null
            || hp.IsSwaLayer is not null || hp.KvSourceLayer is not null;

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

        // Priority 2: Assign layers to GPU.
        int gpuLayers = 0;
        long gpuWeightBytes = fixedGpuBytes;
        if (pinGpuLayers is int pin)
        {
            // Caller pinned an explicit GPU trunk count (e.g. -g N). Honor it exactly and price the
            // trunk weights / KV / expert-cache budget for THIS split rather than the greedy auto
            // split. A pin that exceeds VRAM is the user's explicit choice (the forward pass
            // enforces real fit), so we only clamp the leftover budget to >= 0 here.
            gpuLayers = Math.Clamp(pin, 0, hp.NumLayers);
            for (int i = 0; i < gpuLayers; i++)
                gpuWeightBytes += MeasureLayerBytes(model, hp, i);
            vramBudget = Math.Max(0, vramBudget - (gpuWeightBytes - fixedGpuBytes));
        }
        else
        {
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
            if (perLayerKvShape)
            {
                gpuCtxSize = SolveGpuCtxForPerLayerKv(hp, autoCtxCap, vramBudget, kvDtype, gpuLayers);
            }
            else
            {
                long kvBytesPerToken = 2L * gpuLayers * DTypeInfo.ByteSize((long)hp.NumKvHeads * headDim, kvDtype);
                gpuCtxSize = (int)(vramBudget / kvBytesPerToken);
                gpuCtxSize = Math.Clamp(gpuCtxSize, 512, autoCtxCap);
            }
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
        else if (perLayerKvShape)
        {
            gpuKvBytes = CudaForwardPass.EstimateKvCacheBytes(hp, gpuCtxSize, kvDtype, gpuLayers);
        }
        else
        {
            long kvBytesPerToken = 2L * gpuLayers * DTypeInfo.ByteSize((long)hp.NumKvHeads * headDim, kvDtype);
            gpuKvBytes = kvBytesPerToken * gpuCtxSize;
        }

        // Issue #215: for MoE models the routed experts are NOT charged against the trunk
        // weight budget (excluded in MeasureLayerBytes); the VRAM left after trunk weights
        // and KV is the budget the SLRU routed-expert cache can use (or, when too small,
        // the signal to route MoE to CPU). Diagnostic only — the runtime SLRU/CPU-MoE
        // decision is made later from actual free VRAM. Zero for dense models.
        long expertCacheBudget = hp.IsMoE ? Math.Max(0, vramBudget - gpuKvBytes) : 0;

        // Issue #215: the full-residency GPU cost of every routed expert (gate/up/down _exps
        // across all layers). Compared against ExpertCacheBudgetBytes by the CLI to decide
        // whether full-GPU MoE is viable or the model must use the hybrid path (SLRU streaming
        // or CPU-MoE). Zero for dense models.
        long moeRoutedExpertBytes = 0;
        if (hp.IsMoE)
        {
            for (int i = 0; i < hp.NumLayers; i++)
            {
                moeRoutedExpertBytes += MeasureGpuTensorBytes(model, $"blk.{i}.ffn_gate_exps.weight");
                moeRoutedExpertBytes += MeasureGpuTensorBytes(model, $"blk.{i}.ffn_up_exps.weight");
                moeRoutedExpertBytes += MeasureGpuTensorBytes(model, $"blk.{i}.ffn_down_exps.weight");
            }
        }

        return new LayerPlacement(
            gpuLayers,
            hp.NumLayers - gpuLayers,
            gpuWeightBytes,
            gpuKvBytes,
            gpuCtxSize,
            expertCacheBudget,
            moeRoutedExpertBytes);
    }

    // For per-layer-KV models (Gemma 4: SWA window-rings, KV-share aliasing, per-layer
    // head_dim / kv-heads) KV bytes are piecewise in context — SWA layers saturate at
    // their window ring — so the flat divide the uniform path uses would mis-solve the
    // context. Binary-search the largest context whose true allocation (the same
    // CudaForwardPass.EstimateKvCacheBytes the forward pass allocates from) fits the
    // budget. The function is monotonic non-decreasing in context, so an upper-bound
    // search converges; the floor (512) mirrors the uniform path's Math.Clamp lower bound.
    internal static int SolveGpuCtxForPerLayerKv(
        ModelHyperparams hp, int autoCtxCap, long vramBudget, DType kvDtype, int gpuLayers)
    {
        const int floorCtx = 512;
        if (autoCtxCap <= floorCtx ||
            CudaForwardPass.EstimateKvCacheBytes(hp, floorCtx, kvDtype, gpuLayers) > vramBudget)
            return Math.Min(floorCtx, autoCtxCap);
        int lo = floorCtx, hi = autoCtxCap;
        while (lo < hi)
        {
            int mid = lo + (hi - lo + 1) / 2;
            if (CudaForwardPass.EstimateKvCacheBytes(hp, mid, kvDtype, gpuLayers) <= vramBudget)
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    }

    private static long MeasureTensorBytes(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        return info is not null ? ((info.Value.ByteSize + 3) & ~3L) : 0;
    }

    // Embedding-aware: Q4_K / Q8_0 / Q6_K embeddings stay quantized on GPU (each has a
    // dedicated EmbedLookup* kernel — Q6_K added for the Gemma 4 12B QAT tied table,
    // issue #124). Anything else is dequantized to F32 at upload, so its post-upload
    // footprint is 4 bytes per element.
    private static long MeasureGpuEmbeddingBytes(GgufModel model, string name)
    {
        var info = model.FindTensor(name);
        if (info is null) return 0;
        var dt = info.Value.DType;
        if (dt == DType.Q4_K || dt == DType.Q8_0 || dt == DType.Q6_K)
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
        // Issue #215: routed experts (ffn_gate_exps / ffn_up_exps / ffn_down_exps) are
        // NOT eagerly resident per layer — the runtime streams them via the SLRU slot
        // manager or runs them on CPU. Charging them against the trunk weight budget
        // makes the greedy packer over-charge MoE layers and stop placing trunk layers
        // far too early. Measure MoE layers as trunk-only (attn + norms + router + shared
        // expert) and exclude the routed experts here. ffn_gate_inp (the router) IS
        // eagerly resident, so it stays.
        string[] suffixes = hp.IsMoE
            ?
            [
                "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_output.weight", "ffn_norm.weight", "ffn_gate_inp.weight"
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
        // Dtypes the CUDA forward pass keeps packed in VRAM (CudaForwardPass.UploadWeight):
        // Q4_0 (Gemma 4 12B QAT, #124), Q4_K, Q6_K, Q8_0. Others (incl. Q5_K) dequant to F32.
        switch (tensor.DType)
        {
            case DType.Float32:
            case DType.Q4_0:
            case DType.Q4_K:
            case DType.Q6_K:
            case DType.Q8_0:
                return (tensor.ByteSize + 3) & ~3L;
            default:
                return tensor.ElementCount * sizeof(float);
        }
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

/// <summary>
/// Result of tier placement analysis.
/// <paramref name="ExpertCacheBudgetBytes"/> is the VRAM left over after trunk weights
/// and KV for the SLRU routed-expert cache (MoE only; 0 for dense — issue #215).
/// <paramref name="MoeRoutedExpertBytes"/> is the full-residency GPU cost of all routed
/// experts; when it exceeds <paramref name="ExpertCacheBudgetBytes"/> the model can't keep
/// every expert resident and must use the hybrid path (MoE only; 0 for dense — issue #215).
/// </summary>
public sealed record LayerPlacement(
    int GpuLayers,
    int CpuLayers,
    long GpuWeightBytes,
    long GpuKvBytes,
    int RecommendedCtxSize,
    long ExpertCacheBudgetBytes = 0,
    long MoeRoutedExpertBytes = 0)
{
    public string Summary()
    {
        double gpuWeightMB = GpuWeightBytes / (1024.0 * 1024);
        double gpuKvMB = GpuKvBytes / (1024.0 * 1024);
        string moe = ExpertCacheBudgetBytes > 0
            ? $", expert-cache budget: {ExpertCacheBudgetBytes / (1024.0 * 1024):F0} MB" +
              $" (routed experts: {MoeRoutedExpertBytes / (1024.0 * 1024):F0} MB)"
            : "";
        return $"GPU: {GpuLayers} layers ({gpuWeightMB:F0} MB weights, {gpuKvMB:F0} MB KV), CPU: {CpuLayers} layers, ctx: {RecommendedCtxSize}{moe}";
    }
}
