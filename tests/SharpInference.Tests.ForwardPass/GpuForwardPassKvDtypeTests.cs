using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// bf16 KV-cache parity for the Vulkan <see cref="GpuForwardPass"/> (issue #311). With
/// <c>kvDtype: DType.BFloat16</c> the K/V cache is stored half-width (IEEE fp16 packed
/// two-per-uint via core-GLSL <c>packHalf2x16</c>); kernel arithmetic stays fp32, so decode
/// must be argmax-stable vs the fp32 cache at short context — only the stored value's mantissa
/// is narrowed. The bf16 decode is teacher-forced onto the fp32 trajectory so the KV dtype is
/// the only variable at each position. Asserts, per teacher-forced position:
/// <list type="bullet">
///   <item>all bf16 logits are finite,</item>
///   <item>fp32's top-1 stays within bf16's top-5 — the reorder-tolerant "argmax-stable"
///         criterion (a genuine near-tie can flip top-1 with no kernel bug),</item>
///   <item>the logit max-abs gap is within a small rounding budget.</item>
/// </list>
///
/// The parity test is the correctness gate for the <c>AttentionBf16</c> read shader: a failure
/// most likely means the <c>unpackHalf2x16(buf[idx&gt;&gt;1])[idx&amp;1]</c> lane-select or an
/// indexing divergence from the fp32 shader. SnapKV is forced off so the KV dtype is the only
/// variable. Each case is skipped silently when Vulkan is unavailable or the GGUF isn't on disk.
/// </summary>
public sealed class GpuForwardPassKvDtypeTests
{
    private const string ModelFile = "Qwen3-8B-Q4_K_M.gguf";

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

    /// <summary>
    /// Prefill <paramref name="prompt"/>, then decode <paramref name="steps"/> tokens on a fresh
    /// GpuForwardPass with the given KV dtype. When <paramref name="forced"/> is non-null the
    /// decode is TEACHER-FORCED on those tokens so two runs follow an identical trajectory — the
    /// KV dtype is then the only variable at each decode position. Returns the per-position logits
    /// (index 0 = prefill, 1.. = each decode step) and the greedy argmax at each. SnapKV is forced
    /// off so the KV dtype is isolated.
    /// </summary>
    private static (float[][] logits, int[] argmax) RunPrefillDecode(
        VulkanBackend gpu, string path, DType kvDtype, string prompt, int steps, int ctx, int[]? forced)
    {
        var prevSnap = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0"); // isolate the KV dtype
        try
        {
            using var model = GgufModel.Open(path);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
            var tokenizer = GgufTokenizer.FromGgufModel(model);
            int useCtx = Math.Min(hp.ContextLength, ctx);
            using var fwd = new GpuForwardPass(model, gpu, hp, maxContextLength: useCtx, kvDtype: kvDtype);

            var tokens = tokenizer.Encode(prompt).ToArray();
            var perPos = new float[steps + 1][];
            var argmax = new int[steps + 1];

            var logits = fwd.Prefill(tokens).ToArray();
            perPos[0] = logits;
            argmax[0] = Sampler.Greedy(logits);

            for (int i = 0; i < steps; i++)
            {
                int fed = forced is not null ? forced[i] : argmax[i];
                logits = fwd.Forward(fed, tokens.Length + i).ToArray();
                perPos[i + 1] = logits;
                argmax[i + 1] = Sampler.Greedy(logits);
            }
            return (perPos, argmax);
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevSnap);
        }
    }

    private const string LowEntropyPrompt = "The quick brown fox jumps over the lazy";

    // Qwen3 ChatML. Thinking is left on (the template auto-opens <think>), which only makes the
    // continuation longer/more varied — fine for a coherence check.
    private const string Qwen3TemplatePrompt =
        "<|im_start|>user\nWrite a short sentence about the ocean.<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>
    /// bf16 KV must stay argmax-stable vs fp32 KV on the SAME teacher-forced trajectory. fp32 is
    /// the reference; bf16 is forced onto its tokens so the only variable is the KV store dtype.
    /// This is the correctness gate for the AttentionBf16 fp16-unpack read path.
    /// </summary>
    [Fact]
    public void Bf16Kv_ArgmaxStable_VsFp32()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        const int steps = 6;
        const int ctx = 2048;
        const float maxAbsTol = 2.0f;

        var (f32, f32Argmax) = RunPrefillDecode(gpu, path, DType.Float32, LowEntropyPrompt, steps, ctx, forced: null);
        var (kv, _) = RunPrefillDecode(gpu, path, DType.BFloat16, LowEntropyPrompt, steps, ctx, forced: f32Argmax);

        for (int p = 0; p <= steps; p++)
        {
            Assert.Equal(f32[p].Length, kv[p].Length);
            float maxAbs = 0f;
            for (int i = 0; i < f32[p].Length; i++)
            {
                Assert.True(float.IsFinite(kv[p][i]), $"non-finite bf16 logit at pos {p}, idx {i}.");
                maxAbs = Math.Max(maxAbs, Math.Abs(f32[p][i] - kv[p][i]));
            }
            Assert.True(maxAbs < maxAbsTol,
                $"pos {p} bf16 vs fp32 logit max-abs {maxAbs:F3} exceeds the rounding budget " +
                $"({maxAbsTol:F1}) — likely an AttentionBf16 indexing/lane-select bug, not fp16-store rounding.");
            Assert.True(TopK(kv[p], 5).Contains(f32Argmax[p]),
                $"pos {p} fp32 top-1 ({f32Argmax[p]}) fell out of bf16's top-5 (max-abs {maxAbs:F3}) — " +
                "the bf16 attention read reordered the head of the distribution.");
        }
    }

    /// <summary>
    /// bf16 KV greedy decode (the path picks its OWN tokens — not teacher-forced) on a
    /// template-correct prompt must stay coherent: all logits finite, the first generated token
    /// is not EOS, and ≥2 distinct argmaxes over the run (catches NaN / single-token collapse
    /// from a wrong attention read).
    /// </summary>
    [Fact]
    public void Bf16Kv_GreedyDecode_Coherent()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath();
        if (path is null) return;

        int eosId;
        using (var model = GgufModel.Open(path))
            eosId = GgufTokenizer.FromGgufModel(model).EosTokenId;

        const int steps = 5;
        const int ctx = 2048;

        var (logits, argmax) = RunPrefillDecode(gpu, path, DType.BFloat16, Qwen3TemplatePrompt, steps, ctx, forced: null);

        for (int p = 0; p <= steps; p++)
            for (int i = 0; i < logits[p].Length; i++)
                Assert.True(float.IsFinite(logits[p][i]),
                    $"bf16 greedy: non-finite logit at pos {p}, idx {i} — a bf16-KV attention read bug.");

        Assert.True(argmax[0] != eosId,
            "bf16 greedy: first token was EOS — the template-correct prompt should have a real " +
            "continuation, so this means the bf16 greedy path collapsed.");

        var seen = new HashSet<int>(argmax);
        Assert.True(seen.Count >= 2,
            $"bf16 greedy decode produced only {seen.Count} distinct token(s) " +
            $"([{string.Join(",", argmax)}]) — bf16-KV greedy decode is degenerate.");
    }
}
