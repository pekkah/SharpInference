using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end Gemma 4 E4B on the Vulkan <see cref="GpuForwardPass"/> with a NARROWED KV cache
/// (bf16 / q8_0), the wiring added in issue #351 Phase 2. The per-layer KV allocation and the
/// RunGemma4Layers append/attention dispatch now dtype-branch (matching the dense path), so the
/// E4B family can use --kv-type bf16|q8_0 for long context — as it already does on CUDA.
///
/// Narrowed KV is argmax-stable but not bit-exact vs fp32 (the established #311 / #325 contract:
/// bf16 packs two fp16 per word; q8_0 block-quantizes per 32 elems). The reference here is the
/// SAME Vulkan fp32 GpuForwardPass (so the only variable is the KV store), and the gate is greedy
/// argmax agreement at the first decoded token plus a non-degenerate, finite short decode. This
/// mirrors <see cref="Gemma4VulkanPleE2ETests"/> (CPU↔Vulkan argmax parity for the fp32 path).
///
/// Uses <c>gemma-4-E4B_q4_0-it.gguf</c> — the QAT q4_0 GGUF with PLE + a KV-share tail. The
/// per-layer kvDim is 512 (SWA, head_dim 256) or 1024 (global, head_dim 512): both even (bf16) and
/// multiples of 32 (q8_0). Silent-skips when Vulkan is unavailable, the device is out of memory for
/// full offload, or the GGUF isn't on disk. NOT run by the implementation pass (the orchestrator
/// verifies on a real GPU).
/// </summary>
public sealed class Gemma4VulkanNarrowedKvE2ETests
{
    private const string ModelFile = "gemma-4-E4B_q4_0-it.gguf";

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"E:\models\{fileName}",
            $@"C:\p\sharpi\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static GpuForwardPass? TryBuild(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, DType kvDtype)
    {
        try
        {
            return new GpuForwardPass(model, gpu, hp, maxContextLength: 2048, kvDtype: kvDtype);
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            return null;                                          // OOM-gated
        }
    }

    // Greedy-decodes NSteps tokens from a fixed prompt, returning the argmax sequence
    // (logits[0] = first decoded token). Disposes the pass.
    private static int[] GreedyDecode(GpuForwardPass fwd, int[] tokens, int nSteps)
    {
        var decoded = new int[nSteps];
        using (fwd)
        {
            var logits = fwd.Prefill(tokens);
            decoded[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < nSteps; i++)
            {
                var step = fwd.Forward(decoded[i - 1], pos++);
                decoded[i] = Argmax(step);
            }
        }
        return decoded;
    }

    /// <summary>
    /// bf16 and q8_0 KV must each agree with the fp32 Vulkan reference at the first decoded argmax
    /// (the tightest pre-drift signal for an argmax-stable store), and the 14-step decode must be
    /// finite and non-degenerate (≥6 distinct tokens). Pins the issue #351 Phase 2 narrowed-KV
    /// wiring for the per-layer gemma4 geometry.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;                                  // Vulkan-gated
        var path = FindModelPath(ModelFile);
        if (path is null) return;                                 // model-gated

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: only meaningful against a real gemma4 GGUF with PLE + a KV-share tail.
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);
        Assert.NotNull(hp.KvSourceLayer);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 }; // "The capital of France is"
        const int NSteps = 14;

        // fp32 Vulkan reference (the only variable across the three runs is the KV store dtype).
        var fp32Pass = TryBuild(model, gpu, hp, DType.Float32);
        if (fp32Pass is null) return;                             // OOM-gated
        var fp32Decoded = GreedyDecode(fp32Pass, tokens, NSteps);

        foreach (var kvDtype in new[] { DType.BFloat16, DType.Q8_0 })
        {
            var pass = TryBuild(model, gpu, hp, kvDtype);
            if (pass is null) return;                             // OOM-gated
            var decoded = GreedyDecode(pass, tokens, NSteps);

            Assert.True(fp32Decoded[0] == decoded[0],
                $"E4B q4_0 Vulkan KV {kvDtype} first-argmax disagrees with fp32: " +
                $"fp32={fp32Decoded[0]} {kvDtype}={decoded[0]}. " +
                $"fp32 decode=[{string.Join(",", fp32Decoded)}] " +
                $"{kvDtype} decode=[{string.Join(",", decoded)}]. " +
                "Narrowed KV is argmax-stable vs fp32 (issue #311 / #325) — a divergence at the " +
                "first token is a per-layer append/attention dispatch or allocation bug, not store noise.");

            int distinct = new HashSet<int>(decoded).Count;
            Assert.True(distinct >= 6,
                $"E4B q4_0 Vulkan KV {kvDtype} {NSteps}-step decode collapsed to {distinct} distinct " +
                $"token(s): [{string.Join(",", decoded)}]. Output is degenerate.");
        }
    }
}
