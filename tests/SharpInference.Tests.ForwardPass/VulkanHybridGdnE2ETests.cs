using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end parity for <see cref="VulkanHybridGdnForwardPass"/> — the Vulkan + CPU hybrid
/// for the DENSE qwen36 Gated-DeltaNet model (Qwen3.6-27B-MTP). The oracle is the CUDA
/// <see cref="CudaHybridGdnForwardPass"/> on the same model: both prefill "Hello" and greedy-
/// decode a short window, and the per-step argmax token streams must match. Q4_K cross-backend
/// is argmax-stable (not bit-exact), so the gate is argmax agreement, finite logits, and a
/// non-degenerate decode.
///
/// <para>27B can't co-reside on both backends on a 12 GB-class card, so the CUDA oracle runs
/// FULLY first (tokens collected, pass + backend disposed), THEN Vulkan runs and the token
/// streams are compared. fp32 KV is pinned on the CUDA oracle (SHARPI_KV_DTYPE=fp32) so the KV
/// dtype is not a confound — the Vulkan pass is fp32-KV by construction.</para>
///
/// Silent-skips when Vulkan or CUDA is unavailable, the device is out of memory, or the GGUF
/// isn't on disk. NOT run by the implementation pass (the orchestrator verifies on a real GPU).
/// </summary>
public sealed class VulkanHybridGdnE2ETests
{
    private const int DecodeSteps = 8;

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static CudaBackend? TryCreateCuda()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? FindDenseModelPath() =>
        FindModelPath("Qwen3.6-27B-MTP-Q4_K_M.gguf");

    private static string? FindMoeModelPath() =>
        FindModelPath("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");

    private static string? FindModelPath(params string[] fileNames)
    {
        string[] absoluteRoots = { @"E:\models", @"C:\p\sharpi\models" };
        foreach (var root in absoluteRoots)
            foreach (var f in fileNames)
            {
                var p = Path.Combine(root, f);
                if (File.Exists(p)) return p;
            }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            foreach (var f in fileNames)
            {
                var p = Path.Combine(dir, "models", f);
                if (File.Exists(p)) return p;
            }
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static void AssertFinite(ReadOnlySpan<float> logits, string where)
    {
        int nonFinite = 0;
        for (int i = 0; i < logits.Length; i++)
            if (!float.IsFinite(logits[i])) nonFinite++;
        Assert.True(nonFinite == 0, $"{nonFinite} non-finite logits in {where}.");
    }

    private static LayerPlacement GdnPlacement(ModelHyperparams hp) => new(
        GpuLayers: hp.NumLayers,
        CpuLayers: 0,
        GpuWeightBytes: 0,
        GpuKvBytes: 0,
        RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));

    /// <summary>
    /// Greedy-decode argmax parity Vulkan vs CUDA on the dense 27B-MTP GDN model: prefill "Hello",
    /// then ~8 greedy decode steps on each backend, asserting token-for-token argmax equality plus
    /// finite + non-degenerate output. CUDA runs first and is disposed before Vulkan (12 GB co-residency).
    /// </summary>
    [Fact]
    public void VulkanHybridGdn_Dense27B_MatchesCudaArgmax()
    {
        // Quick gates first (avoid loading the 16 GB GGUF when we'd skip anyway).
        if (!CudaBackend.IsAvailable()) return;                    // CUDA-gated (oracle)
        var path = FindDenseModelPath();
        if (path is null) return;                                  // model-gated

        // Probe Vulkan availability up front so we don't run the (expensive) CUDA oracle for nothing.
        using (var probe = TryCreateVulkan())
            if (probe is null) return;                             // Vulkan-gated

        // fp32 KV on the CUDA oracle removes the KV-dtype confound (Vulkan pass is fp32-KV).
        string? prevKvDtype = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "fp32");

        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

            // Defensive: this test only fires on a dense hybrid GDN model.
            Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");
            Assert.NotNull(hp.Gdn);
            Assert.NotNull(hp.LayerTypes);
            Assert.False(hp.IsMoE, "Expected a DENSE GDN model (Qwen3.6-27B-MTP) for Round 1; MoE is Round 2.");

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var prompt = tokenizer.Encode("Hello");
            Assert.NotEmpty(prompt);

            // ── CUDA oracle: collect prefill + DecodeSteps greedy argmaxes, then dispose. ──
            var cudaTokens = new int[DecodeSteps];
            {
                using var cuda = TryCreateCuda();
                if (cuda is null) return;
                using var cfwd = new CudaHybridGdnForwardPass(model, cuda, hp, GdnPlacement(hp));

                var logits = cfwd.Prefill(prompt);
                AssertFinite(logits, "CUDA oracle prefill");
                cudaTokens[0] = Argmax(logits);
                int pos = prompt.Count;
                for (int i = 1; i < DecodeSteps; i++)
                {
                    var step = cfwd.Forward(cudaTokens[i - 1], pos++);
                    AssertFinite(step, $"CUDA oracle decode step {i}");
                    cudaTokens[i] = Argmax(step);
                }
            }
            // CUDA backend fully disposed here — VRAM freed for the Vulkan run.

            // ── Vulkan: same prefill + decode. Skip gracefully on OOM. ──
            var vkTokens = new int[DecodeSteps];
            {
                VulkanBackend gpu;
                try { gpu = new VulkanBackend(); }
                catch { return; }
                using (gpu)
                {
                    VulkanHybridGdnForwardPass vfwd;
                    try
                    {
                        vfwd = new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp));
                    }
                    catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
                    {
                        return; // device can't fit the dense GDN trunk — graceful skip
                    }
                    using (vfwd)
                    {
                        var logits = vfwd.Prefill(prompt);
                        AssertFinite(logits, "Vulkan prefill");
                        vkTokens[0] = Argmax(logits);
                        int pos = prompt.Count;
                        for (int i = 1; i < DecodeSteps; i++)
                        {
                            var step = vfwd.Forward(vkTokens[i - 1], pos++);
                            AssertFinite(step, $"Vulkan decode step {i}");
                            vkTokens[i] = Argmax(step);
                        }
                    }
                }
            }

            // ── Token-for-token argmax parity. ──
            for (int i = 0; i < DecodeSteps; i++)
                Assert.True(cudaTokens[i] == vkTokens[i],
                    $"CUDA/Vulkan argmax diverge at step {i}: CUDA={cudaTokens[i]} Vulkan={vkTokens[i]}. " +
                    $"CUDA stream=[{string.Join(",", cudaTokens)}] Vulkan stream=[{string.Join(",", vkTokens)}]. " +
                    "Likely a GDN op-order / gated-attention / CPU-FFN-boundary mismatch, not Q4_K cross-backend noise.");

            // ── Non-degenerate decode (≥3 distinct tokens over the window). ──
            int distinct = 0;
            for (int i = 0; i < vkTokens.Length; i++)
            {
                bool seen = false;
                for (int j = 0; j < i; j++) if (vkTokens[j] == vkTokens[i]) { seen = true; break; }
                if (!seen) distinct++;
            }
            Assert.True(distinct >= 3,
                $"Vulkan GDN decode collapsed to {distinct} distinct token(s): [{string.Join(",", vkTokens)}].");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevKvDtype);
        }
    }

    /// <summary>
    /// Byte-exact self-parity for the Vulkan GDN batched prefill (issue #356 PR5b): prefill a
    /// fixed ~16-token prompt on the dense 27B-MTP GDN model TWICE on two independent
    /// <see cref="VulkanHybridGdnForwardPass"/> instances — once with
    /// <c>SHARPI_VULKAN_BATCHED_PREFILL=1</c> (the batched trunk) and once with <c>=0</c> (the
    /// sequential per-token Forward loop) — then greedy-decode <see cref="DecodeSteps"/> tokens on
    /// each. The two token streams must be BYTE-IDENTICAL: the batched trunk reproduces N
    /// sequential Forwards op-for-op, so any divergence is a wiring/barrier bug, not Q4_K noise.
    /// No CUDA needed (pure Vulkan self-parity). Silent-skips when Vulkan or the GGUF is absent;
    /// graceful skip on device OOM.
    /// </summary>
    [Fact]
    public void VulkanHybridGdn_BatchedPrefill_MatchesSequential()
    {
        var path = FindDenseModelPath();
        if (path is null) return;                                   // model-gated
        using (var probe = TryCreateVulkan())
            if (probe is null) return;                             // Vulkan-gated

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");
        Assert.NotNull(hp.Gdn);
        Assert.NotNull(hp.LayerTypes);

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        // A ~16-token prompt, all positions < 4096 (the AttentionBatched shared-scores range).
        var prompt = tokenizer.Encode("The quick brown fox jumps over the lazy dog near the river bank at dawn.");
        Assert.True(prompt.Count > 1, "Need a multi-token prompt to exercise the batched path.");
        Assert.True(prompt.Count + DecodeSteps <= 4096);

        // Run a full prefill + greedy decode on a fresh pass with the batched gate set to `gateVal`.
        int[] RunWithGate(string gateVal)
        {
            string? prev = Environment.GetEnvironmentVariable("SHARPI_VULKAN_BATCHED_PREFILL");
            Environment.SetEnvironmentVariable("SHARPI_VULKAN_BATCHED_PREFILL", gateVal);
            try
            {
                VulkanBackend gpu;
                try { gpu = new VulkanBackend(); }
                catch { return Array.Empty<int>(); }   // Vulkan vanished → caller skips
                using (gpu)
                {
                    VulkanHybridGdnForwardPass fwd;
                    try { fwd = new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp)); }
                    catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
                    {
                        return Array.Empty<int>();      // device too small → caller skips
                    }
                    using (fwd)
                    {
                        var outTokens = new int[DecodeSteps];
                        var logits = fwd.Prefill(prompt);
                        AssertFinite(logits, $"Vulkan prefill (gate={gateVal})");
                        outTokens[0] = Argmax(logits);
                        int pos = prompt.Count;
                        for (int i = 1; i < DecodeSteps; i++)
                        {
                            var step = fwd.Forward(outTokens[i - 1], pos++);
                            AssertFinite(step, $"Vulkan decode step {i} (gate={gateVal})");
                            outTokens[i] = Argmax(step);
                        }
                        return outTokens;
                    }
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("SHARPI_VULKAN_BATCHED_PREFILL", prev);
            }
        }

        int[] batched = RunWithGate("1");
        if (batched.Length == 0) return;               // OOM / Vulkan-gone → skip
        int[] sequential = RunWithGate("0");
        if (sequential.Length == 0) return;            // OOM / Vulkan-gone → skip

        for (int i = 0; i < DecodeSteps; i++)
            Assert.True(batched[i] == sequential[i],
                $"Batched vs sequential prefill diverge at step {i}: batched={batched[i]} sequential={sequential[i]}. " +
                $"batched=[{string.Join(",", batched)}] sequential=[{string.Join(",", sequential)}]. " +
                "The batched trunk must be byte-identical to N sequential Forwards (#356 PR5b contract) — " +
                "likely a missing RecordBarrier (e.g. the GDN conv WAR) or a chunk-boundary state-advance bug.");
    }

    /// <summary>
    /// Self-parity for the opt-in FlashQLA chunked prefill scan (issue #356 PR5c): prefill the same
    /// prompt on the dense 27B-MTP GDN model with the batched trunk in both runs, toggling only
    /// <c>SHARPI_VULKAN_GDN_CHUNKED_PREFILL</c> (1 = the chunk-parallel GdnChunkedPrefill, unset =
    /// the byte-exact fused GdnRecurrenceScan), then greedy-decode <see cref="DecodeSteps"/> tokens
    /// on each. The chunked scan is argmax-stable (FP reduction order differs, not byte-exact), so
    /// the gate is argmax agreement (the per-step token streams must match). Silent-skips when
    /// Vulkan/GGUF absent or the device can't fit the ~34 KB chunked-scan shared tile.
    /// </summary>
    [Fact]
    public void VulkanHybridGdn_ChunkedPrefill_MatchesFusedScanArgmax()
    {
        var path = FindDenseModelPath();
        if (path is null) return;                                   // model-gated
        using (var probe = TryCreateVulkan())
        {
            if (probe is null) return;                             // Vulkan-gated
            if (!probe.SupportsGdnChunkedPrefill) return;          // device shared-mem too small
        }

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");

        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var prompt = tokenizer.Encode("The quick brown fox jumps over the lazy dog near the river bank at dawn.");
        Assert.True(prompt.Count > 1 && prompt.Count + DecodeSteps <= 4096);

        // Run prefill + greedy decode with the chunked-scan gate set to `gateVal` (batched ON).
        int[] RunWithChunked(string gateVal)
        {
            string? prev = Environment.GetEnvironmentVariable("SHARPI_VULKAN_GDN_CHUNKED_PREFILL");
            Environment.SetEnvironmentVariable("SHARPI_VULKAN_GDN_CHUNKED_PREFILL", gateVal);
            try
            {
                VulkanBackend gpu;
                try { gpu = new VulkanBackend(); }
                catch { return Array.Empty<int>(); }
                using (gpu)
                {
                    VulkanHybridGdnForwardPass fwd;
                    try { fwd = new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp)); }
                    catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
                    {
                        return Array.Empty<int>();
                    }
                    using (fwd)
                    {
                        var outTokens = new int[DecodeSteps];
                        var logits = fwd.Prefill(prompt);
                        AssertFinite(logits, $"Vulkan prefill (chunked={gateVal})");
                        outTokens[0] = Argmax(logits);
                        int pos = prompt.Count;
                        for (int i = 1; i < DecodeSteps; i++)
                        {
                            var step = fwd.Forward(outTokens[i - 1], pos++);
                            AssertFinite(step, $"Vulkan decode step {i} (chunked={gateVal})");
                            outTokens[i] = Argmax(step);
                        }
                        return outTokens;
                    }
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("SHARPI_VULKAN_GDN_CHUNKED_PREFILL", prev);
            }
        }

        int[] chunked = RunWithChunked("1");
        if (chunked.Length == 0) return;
        int[] fused = RunWithChunked("0");
        if (fused.Length == 0) return;

        for (int i = 0; i < DecodeSteps; i++)
            Assert.True(chunked[i] == fused[i],
                $"Chunked vs fused-scan prefill argmax diverge at step {i}: chunked={chunked[i]} fused={fused[i]}. " +
                $"chunked=[{string.Join(",", chunked)}] fused=[{string.Join(",", fused)}]. " +
                "GdnChunkedPrefill is argmax-stable vs GdnRecurrenceScan (FP-noise close for L2-normed inputs).");
    }

    /// <summary>
    /// Greedy-decode argmax parity Vulkan vs CUDA on the GDN+MoE model (Qwen3.6-35B-A3B,
    /// PR4 Round 2). On a 12 GB-class card neither backend can cache the 256 experts × 40
    /// layers in VRAM, so both auto-select (and we pin via SHARPI_CPU_MOE=1) the CPU-MoE
    /// path: the shared expert runs on GPU, routed experts on CPU. fp32 KV is pinned on the
    /// CUDA oracle so the KV dtype is not a confound. CUDA runs fully first and is disposed
    /// before Vulkan (21 GB GGUF — single-backend residency on 12 GB).
    /// </summary>
    [Fact]
    public void VulkanHybridGdn_Moe35B_MatchesCudaArgmax()
    {
        if (!CudaBackend.IsAvailable()) return;                    // CUDA-gated (oracle)
        var path = FindMoeModelPath();
        if (path is null) return;                                  // model-gated

        using (var probe = TryCreateVulkan())
            if (probe is null) return;                             // Vulkan-gated

        // Remove confounds: fp32 KV on both backends + force CPU-MoE on both so the routed-
        // expert placement matches (the auto-heuristic would pick CPU-MoE on a 12 GB card
        // anyway, but pinning makes the test deterministic across cards).
        string? prevKvDtype = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");
        string? prevCpuMoe = Environment.GetEnvironmentVariable("SHARPI_CPU_MOE");
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "fp32");
        Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", "1");

        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

            Assert.True(hp.IsHybridSsm, "Expected hp.IsHybridSsm for the qwen36 GDN model");
            Assert.NotNull(hp.Gdn);
            Assert.NotNull(hp.LayerTypes);
            Assert.True(hp.IsMoE, "Expected a GDN+MoE model (Qwen3.6-35B-A3B) for Round 2.");

            var tokenizer = GgufTokenizer.FromGgufModel(model);
            var prompt = tokenizer.Encode("Hello");
            Assert.NotEmpty(prompt);

            // ── CUDA oracle: collect prefill + DecodeSteps greedy argmaxes, then dispose. ──
            var cudaTokens = new int[DecodeSteps];
            {
                using var cuda = TryCreateCuda();
                if (cuda is null) return;
                using var cfwd = new CudaHybridGdnForwardPass(model, cuda, hp, GdnPlacement(hp));

                var logits = cfwd.Prefill(prompt);
                AssertFinite(logits, "CUDA oracle prefill");
                cudaTokens[0] = Argmax(logits);
                int pos = prompt.Count;
                for (int i = 1; i < DecodeSteps; i++)
                {
                    var step = cfwd.Forward(cudaTokens[i - 1], pos++);
                    AssertFinite(step, $"CUDA oracle decode step {i}");
                    cudaTokens[i] = Argmax(step);
                }
            }
            // CUDA backend fully disposed here — VRAM freed for the Vulkan run.

            // ── Vulkan: same prefill + decode. Skip gracefully on OOM. ──
            var vkTokens = new int[DecodeSteps];
            {
                VulkanBackend gpu;
                try { gpu = new VulkanBackend(); }
                catch { return; }
                using (gpu)
                {
                    VulkanHybridGdnForwardPass vfwd;
                    try
                    {
                        vfwd = new VulkanHybridGdnForwardPass(model, gpu, hp, GdnPlacement(hp));
                    }
                    catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
                    {
                        return; // device can't fit the GDN trunk + shared experts — graceful skip
                    }
                    using (vfwd)
                    {
                        var logits = vfwd.Prefill(prompt);
                        AssertFinite(logits, "Vulkan prefill");
                        vkTokens[0] = Argmax(logits);
                        int pos = prompt.Count;
                        for (int i = 1; i < DecodeSteps; i++)
                        {
                            var step = vfwd.Forward(vkTokens[i - 1], pos++);
                            AssertFinite(step, $"Vulkan decode step {i}");
                            vkTokens[i] = Argmax(step);
                        }
                    }
                }
            }

            // ── Token-for-token argmax parity. ──
            for (int i = 0; i < DecodeSteps; i++)
                Assert.True(cudaTokens[i] == vkTokens[i],
                    $"CUDA/Vulkan argmax diverge at step {i}: CUDA={cudaTokens[i]} Vulkan={vkTokens[i]}. " +
                    $"CUDA stream=[{string.Join(",", cudaTokens)}] Vulkan stream=[{string.Join(",", vkTokens)}]. " +
                    "Likely a MoE router / shared-expert scalar-gate / routed-expert combine / CPU-MoE-boundary mismatch.");

            // ── Non-degenerate decode (≥3 distinct tokens over the window). ──
            int distinct = 0;
            for (int i = 0; i < vkTokens.Length; i++)
            {
                bool seen = false;
                for (int j = 0; j < i; j++) if (vkTokens[j] == vkTokens[i]) { seen = true; break; }
                if (!seen) distinct++;
            }
            Assert.True(distinct >= 3,
                $"Vulkan GDN+MoE decode collapsed to {distinct} distinct token(s): [{string.Join(",", vkTokens)}].");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", prevKvDtype);
            Environment.SetEnvironmentVariable("SHARPI_CPU_MOE", prevCpuMoe);
        }
    }
}
