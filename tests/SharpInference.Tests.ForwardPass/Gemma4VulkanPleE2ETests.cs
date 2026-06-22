using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end Gemma 4 E4B on the Vulkan <see cref="GpuForwardPass"/> exercising the PLE
/// (per-layer token embeddings) + shared-KV path added in issue #351. The full-offload Vulkan
/// trunk now runs the E4B family (PLE injection per layer + a KV-share tail aliased to its source
/// layer); these tests pin that wiring against the CPU <see cref="ForwardPass"/> reference.
///
/// The CPU pass is the independent oracle (it already supports PLE + shared-KV and shares no GPU
/// kernels with Vulkan). Q4_0 / Gemma-4 is argmax-stable but not bit-exact across backends, so the
/// gate is greedy argmax agreement at the first decoded token — the tightest signal we can demand
/// without bit-exactness — plus a coherence check that the decode does not degenerate. This mirrors
/// <see cref="Gemma4CudaForwardPassTests"/> (CPU↔CUDA argmax parity) for the Vulkan backend.
///
/// Uses <c>gemma-4-E4B_q4_0-it.gguf</c> — the QAT q4_0 GGUF that genuinely OMITS attn_k / attn_v /
/// attn_k_norm for its 18 KV-share tail layers (the case the shared-KV weight-skip + cache-alias
/// must handle) AND carries the PLE table. Silent-skips when Vulkan is unavailable, the device is
/// out of memory for full offload, or the GGUF isn't on disk. NOT run by the implementation pass
/// (the orchestrator verifies on a real GPU).
/// </summary>
public sealed class Gemma4VulkanPleE2ETests
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

    private static int[] TopK(ReadOnlySpan<float> logits, int k)
    {
        var result = new int[k];
        var taken = new HashSet<int>();
        for (int ki = 0; ki < k; ki++)
        {
            int best = -1; float bestVal = float.NegativeInfinity;
            for (int i = 0; i < logits.Length; i++)
            {
                if (taken.Contains(i)) continue;
                if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
            }
            result[ki] = best;
            taken.Add(best);
        }
        return result;
    }

    /// <summary>
    /// First-decode argmax parity (Vulkan vs CPU) on a 9-token prefill. The Vulkan PLE + shared-KV
    /// trunk must agree with the CPU reference at the tightest pre-drift signal. Also asserts finite,
    /// non-EOS logits and a non-degenerate short decode.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Q4_0_VulkanForward_MatchesCpuArgmax()
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
        int sharedLayer = -1;
        for (int i = 0; i < hp.KvSourceLayer!.Count; i++)
            if (hp.KvSourceLayer[i] >= 0) { sharedLayer = i; break; }
        Assert.True(sharedLayer >= 0, "expected a KV-share layer in the E4B q4_0 GGUF — wrong file?");
        Assert.Null(model.FindTensor($"blk.{sharedLayer}.attn_k_norm.weight"));

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        int eosId = ReadIntMetadata(model, "tokenizer.ggml.eos_token_id", fallback: 1);
        var tokens = new int[] { bosId, 651, 6037, 576, 6081, 603, 1234, 4567, 8901 };

        // CPU reference (the independent oracle — already supports PLE + shared-KV).
        int cpuArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
            cpuArgmax = Argmax(cpuFwd.Prefill(tokens));

        // Vulkan full-offload path. Skip silently if the device can't fit the full offload.
        GpuForwardPass fwd;
        try
        {
            fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 2048);
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            return;
        }
        using (fwd)
        {
            var logits = fwd.Prefill(tokens);
            Assert.Equal(hp.VocabSize, logits.Length);

            int nonFinite = 0;
            for (int i = 0; i < logits.Length; i++)
                if (!float.IsFinite(logits[i])) nonFinite++;
            Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in E4B q4_0 Vulkan output.");

            int vkArgmax = Argmax(logits);
            if (eosId >= 0)
                Assert.NotEqual(eosId, vkArgmax);

            if (cpuArgmax != vkArgmax)
            {
                var cpuTop5 = string.Join(",", TopK(GetCpuLogits(model, hp, tokens), 5));
                var vkTop5 = string.Join(",", TopK(logits, 5));
                Assert.Fail(
                    $"CPU↔Vulkan E4B q4_0 first-argmax disagree: CPU={cpuArgmax} Vulkan={vkArgmax}. " +
                    $"CPU top5=[{cpuTop5}] Vulkan top5=[{vkTop5}]. A PLE-injection / shared-KV / " +
                    "V-norm ordering bug, not Q4_0 cross-backend noise.");
            }
        }
    }

    /// <summary>
    /// Long-decode coherence guard vs the "first-argmax passes then collapses to a 2-cycle"
    /// failure mode (an attention-scale / RoPE / KV-share mismatch). The first 4 decoded tokens
    /// must match the CPU reference exactly (pre-drift), and a 16-step greedy decode must produce
    /// ≥8 distinct tokens. Mirrors <see cref="Gemma4CudaForwardPassTests"/>'s long-decode test.
    /// </summary>
    [Fact]
    public void Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(ModelFile);
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);
        Assert.True(hp.HasPerLayerTokenEmbd);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        var tokens = new int[] { bosId, 818, 5279, 529, 7001, 563 }; // "The capital of France is"

        // CPU reference for the first 4 decode tokens — pre-drift the argmax should match.
        var cpuFirst4 = new int[4];
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp))
        {
            var logits = cpuFwd.Prefill(tokens);
            cpuFirst4[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < cpuFirst4.Length; i++)
            {
                var step = cpuFwd.Forward(cpuFirst4[i - 1], pos++);
                cpuFirst4[i] = Argmax(step);
            }
        }

        GpuForwardPass fwd;
        try
        {
            fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: 2048);
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            return;
        }

        const int NSteps = 16;
        var vkDecoded = new int[NSteps];
        using (fwd)
        {
            var logits = fwd.Prefill(tokens);
            vkDecoded[0] = Argmax(logits);
            int pos = tokens.Length;
            for (int i = 1; i < vkDecoded.Length; i++)
            {
                var step = fwd.Forward(vkDecoded[i - 1], pos++);
                vkDecoded[i] = Argmax(step);
            }
        }

        for (int i = 0; i < cpuFirst4.Length; i++)
            Assert.True(cpuFirst4[i] == vkDecoded[i],
                $"CPU/Vulkan diverge at decode step {i}: CPU={cpuFirst4[i]} Vulkan={vkDecoded[i]}. " +
                $"Full Vulkan decode: [{string.Join(",", vkDecoded)}]. " +
                "Likely a PLE-injection / shared-KV / attention-scale mismatch — the kind that lets " +
                "first-argmax parity pass while the decode loop degenerates.");

        int distinct = 0;
        for (int i = 0; i < vkDecoded.Length; i++)
        {
            bool seen = false;
            for (int j = 0; j < i; j++) if (vkDecoded[j] == vkDecoded[i]) { seen = true; break; }
            if (!seen) distinct++;
        }
        Assert.True(distinct >= 8,
            $"Vulkan 16-step decode collapsed to {distinct} distinct token(s): " +
            $"[{string.Join(",", vkDecoded)}]. Output is degenerate.");
    }

    // Re-runs a CPU prefill to recover its logits for the divergence diagnostic only (the happy
    // path never calls this). Kept out of the main flow so the common case runs the CPU pass once.
    private static float[] GetCpuLogits(GgufModel model, ModelHyperparams hp, int[] tokens)
    {
        using var cpuBackend = new CpuBackend();
        using var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp);
        return cpuFwd.Prefill(tokens).ToArray();
    }
}
