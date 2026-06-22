using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Vortice.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end Gemma 4 inference parity for the Vulkan <see cref="GpuForwardPass"/> (issue #309).
/// The whole gemma4 trunk — per-layer head_dim (256 SWA / 512 global), per-head Q/K norm + plain
/// V-norm, dual RoPE (global theta + rope_freqs vs SWA theta), attention_scale = 1.0, SWA
/// windowing, k_eq_v global layers (V = raw K projection), sandwich norm, GELU-tanh FFN, per-layer
/// output scale, and the final-logit softcap — was previously CUDA/CPU only; the Vulkan backend
/// rejected gemma4 up front. This pins the new Vulkan path against the independent CPU oracle.
///
/// Per feedback_cross_backend_parity_test, the CPU <see cref="ForwardPass"/> is the independent
/// oracle (it mirrors HF/llama.cpp math and shares no GPU kernels). A short synthetic prompt is
/// teacher-forced through BOTH backends so the trajectory is identical at every position; the
/// Vulkan run is then asserted, per teacher-forced position, to:
/// <list type="bullet">
///   <item>produce only finite logits,</item>
///   <item>keep the CPU oracle's top-1 within Vulkan's top-5 (reorder-tolerant argmax stability —
///         a genuine near-tie can flip top-1 with no kernel bug), and</item>
///   <item>stay within a loose logit max-abs budget (Q4_K is argmax-stable but not bit-exact
///         across backends — a real trunk bug diverges by many logits, e.g. a missing V-norm or a
///         wrong attention scale).</item>
/// </list>
///
/// Silent-skip: no-ops when Vulkan is unavailable OR the GGUF isn't on disk. NOT run by the
/// implementation pass (the orchestrator verifies end-to-end on a real GPU).
/// </summary>
public sealed class Gemma4GpuForwardPassTests
{
    private const string ModelFile = "gemma4-v2-Q4_K_M.gguf";

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath()
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{ModelFile}",
            $@"E:\models\{ModelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", ModelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>Indices of the <paramref name="k"/> largest entries of <paramref name="v"/>.</summary>
    private static HashSet<int> TopK(float[] v, int k)
    {
        var idx = new int[v.Length];
        for (int i = 0; i < v.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => v[b].CompareTo(v[a]));
        var set = new HashSet<int>(k);
        for (int i = 0; i < k && i < idx.Length; i++) set.Add(idx[i]);
        return set;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        float bestVal = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > bestVal) { bestVal = v[i]; best = i; }
        return best;
    }

    private static int ReadIntMetadata(GgufModel model, string key, int fallback)
    {
        if (!model.Metadata.TryGetValue(key, out var v) || v is null) return fallback;
        try { return Convert.ToInt32(v); } catch { return fallback; }
    }

    /// <summary>
    /// Prefill then decode <paramref name="steps"/> tokens on a fresh forward pass, teacher-forcing
    /// the decode on <paramref name="forced"/> (when non-null) so two runs follow an identical
    /// trajectory. Returns per-position logits (index 0 = prefill, 1.. = each decode step) and the
    /// greedy argmax at each. The caller supplies the constructed forward pass (CPU or Vulkan).
    /// </summary>
    private static (float[][] logits, int[] argmax) RunPrefillDecode(
        IForwardPass fwd, int[] tokens, int steps, int[]? forced)
    {
        var perPos = new float[steps + 1][];
        var argmax = new int[steps + 1];

        perPos[0] = fwd.Prefill(tokens).ToArray();
        argmax[0] = Argmax(perPos[0]);

        for (int i = 0; i < steps; i++)
        {
            int fed = forced is not null ? forced[i] : argmax[i];
            perPos[i + 1] = fwd.Forward(fed, tokens.Length + i).ToArray();
            argmax[i + 1] = Argmax(perPos[i + 1]);
        }
        return (perPos, argmax);
    }

    /// <summary>
    /// CPU↔Vulkan logit-level parity across a short teacher-forced Gemma 4 trajectory. The CPU run
    /// is the reference (greedy); Vulkan is forced onto its tokens so the per-position state is the
    /// only thing the backend changes. Q4_K is argmax-stable but not bit-exact across backends, so
    /// the per-position assertions are finite + top-5 overlap + a loose max-abs bound.
    /// </summary>
    [Fact]
    public void Gemma4_VulkanMatchesCpuLogits_TeacherForced()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int steps = 5;
        const int ctx = 2048;
        const float maxAbsTol = 6.0f; // Q4_K cross-backend noise stays well under this; a real
                                      // trunk bug (missing V-norm, wrong attn scale) diverges far more.

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Defensive: only meaningful against a real gemma4 (per-layer head_dim) GGUF.
        Assert.NotNull(hp.LayerHeadDim);

        int bosId = ReadIntMetadata(model, "tokenizer.ggml.bos_token_id", fallback: 2);
        // Synthetic BOS-led mid-vocab prompt (gemma4 IT emits a 1-token EOS on real prompts, so a
        // coherence check would be brittle; parity vs the CPU oracle does not need a real prompt).
        int[] tokens = { bosId, 818, 5279, 529, 7001, 563, 1234 };

        int useCtx = Math.Min(hp.ContextLength, ctx);

        float[][] cpu, vk;
        int[] cpuArgmax;
        using (var cpuBackend = new CpuBackend())
        using (var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp, maxContextLength: useCtx))
            (cpu, cpuArgmax) = RunPrefillDecode(cpuFwd, tokens, steps, forced: null);

        GpuForwardPass vkFwd;
        try
        {
            vkFwd = new GpuForwardPass(model, gpu, hp, maxContextLength: useCtx);
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            // Full gemma4 offload doesn't fit this device's VRAM — skip silently (the
            // orchestrator verifies end-to-end on a GPU with enough memory). Not a code
            // failure: the trunk wiring is exercised by the build + the smaller paths.
            return;
        }
        using (vkFwd)
            (vk, _) = RunPrefillDecode(vkFwd, tokens, steps, forced: cpuArgmax);

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(cpu[p].Length, vk[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < cpu[p].Length; i++)
            {
                Assert.True(float.IsFinite(vk[p][i]),
                    $"non-finite Vulkan gemma4 logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(cpu[p][i] - vk[p][i]));
            }

            Assert.True(TopK(vk[p], 5).Contains(cpuArgmax[p]),
                $"pos {p}: CPU top-1 ({cpuArgmax[p]}) fell out of Vulkan's top-5 (max-abs {maxAbs:F3}) — " +
                "the gemma4 Vulkan trunk reordered the head of the distribution (V-norm / attn-scale / " +
                "RoPE / sandwich-norm bug).");
            Assert.True(maxAbs < maxAbsTol,
                $"pos {p}: Vulkan vs CPU gemma4 logit max-abs {maxAbs:F3} exceeds the structural bound " +
                $"({maxAbsTol:F1}); Q4_K cross-backend noise stays well under it — this magnitude is a " +
                "gemma4 trunk math divergence, not quant noise.");
        }
    }
}
