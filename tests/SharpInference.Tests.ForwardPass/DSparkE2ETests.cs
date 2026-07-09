using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// DSpark end-to-end greedy parity (docs/dspark-plan.md, PR #413): the
/// <see cref="DSparkDecoder"/> driving a real deepseek-ai/dspark_qwen3_4b_block7
/// draft head against a CPU Qwen3-4B target must emit EXACTLY the target's own
/// non-spec greedy continuation. Acceptance rate is irrelevant to correctness —
/// the head only proposes; every emitted token is argmax of target logits
/// (verified via BatchVerify + TruncateTo rollback), so a token mismatch means a
/// real tap/verify/rollback bug, not a tolerance issue.
///
/// Silent-skips when the Qwen3-4B GGUF or the DSpark head directory
/// (model.safetensors + config.json) is absent, or when SHARPI_SKIP_E2E=1 —
/// mirrors <see cref="CudaSpecBatchVerifyTests"/>. Memory note: the head loads
/// ~4.7 GB resident (F32 backbone + BF16 row-gather tables); everything is
/// disposed deterministically and the baseline pass is freed before the DSpark
/// pass + head are built.
/// </summary>
public sealed class DSparkE2ETests
{
    private const string HeadDirName = "dspark_qwen3_4b_block7";
    private const int DecodeTokens = 24;

    private static readonly string[] TargetGgufCandidates =
    {
        "Qwen3-4B-Q4_K_M.gguf",
        "Qwen_Qwen3-4B-Q4_K_M.gguf",
        "qwen3-4b-q4_k_m.gguf",
    };

    // "Hello, world! I am a virtual model." — same Qwen3 tokenizer ids as
    // CudaSpecBatchVerifyTests.Prompt (Qwen3-4B shares the Qwen3-8B vocab).
    private static readonly int[] Prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13 };

    private static string? FindModelPath(string file)
    {
        string[] absolute =
        {
            $@"C:\models\{file}",
            $@"E:\models\{file}",
            $@"C:\p\sharpi\models\{file}",
        };
        foreach (var p in absolute)
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", file);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindTargetGguf()
    {
        foreach (var file in TargetGgufCandidates)
        {
            var p = FindModelPath(file);
            if (p is not null) return p;
        }
        return null;
    }

    /// <summary>
    /// Locate the DSpark head directory (same models roots / parent-walk as
    /// <see cref="FindModelPath"/>); it must contain both model.safetensors
    /// and config.json to count as present.
    /// </summary>
    private static string? FindDSparkHeadDir()
    {
        static bool IsValid(string d) =>
            File.Exists(Path.Combine(d, "model.safetensors"))
            && File.Exists(Path.Combine(d, "config.json"));

        string[] absolute =
        {
            $@"C:\models\{HeadDirName}",
            $@"E:\models\{HeadDirName}",
            $@"C:\p\sharpi\models\{HeadDirName}",
        };
        foreach (var d in absolute)
            if (IsValid(d)) return d;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var d = Path.Combine(dir, "models", HeadDirName);
            if (IsValid(d)) return d;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    // SnapKV pinned off: an inherited SHARPI_SNAPKV_BUDGET would flip
    // SupportsHiddenTaps (and BatchVerify usability) to false on the CPU pass —
    // same pinning as CudaSpecBatchVerifyTests.NewFwd.
    private static SharpInference.Engine.ForwardPass NewFwd(
        GgufModel model, CpuBackend cpu, ModelHyperparams hp, int ctx = 512)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new SharpInference.Engine.ForwardPass(model, cpu, hp, maxContextLength: ctx); }
        finally { Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prev); }
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>
    /// The load-bearing guarantee: DSpark-decoded output is byte-identical to
    /// plain greedy decode on the same target. Baseline = 24 tokens of
    /// argmax + Forward on a fresh pass; DSpark = fresh pass with hidden taps
    /// enabled BEFORE prefill (decoder contract), the real block-7 head
    /// drafting, and an empty stop set so exactly 24 tokens are emitted.
    /// Also asserts the head actually drafted (TotalDraftsEmitted &gt; 0) —
    /// parity via zero-length proposals would be vacuous.
    /// </summary>
    [Fact]
    public void DSpark_Qwen3_4B_GreedyParity_E2E()
    {
        if (Environment.GetEnvironmentVariable("SHARPI_SKIP_E2E") == "1") return;
        var ggufPath = FindTargetGguf();
        var headDir = FindDSparkHeadDir();
        if (ggufPath is null || headDir is null) return;

        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        using var cpu = new CpuBackend();

        // ── Baseline: plain greedy on pass A (disposed before pass B). ──
        var baseline = new List<int>();
        using (var fwd = NewFwd(model, cpu, hp))
        {
            var logits = fwd.Prefill(Prompt);
            int P = Prompt.Length;
            int tok = Argmax(logits);
            for (int i = 0; i < DecodeTokens; i++)
            {
                baseline.Add(tok);
                logits = fwd.Forward(tok, P + i);
                tok = Argmax(logits);
            }
        }

        // ── DSpark head: config + compatibility (same checks as the CLI). ──
        var cfg = DSparkConfig.FromJsonFile(Path.Combine(headDir, "config.json"));
        Assert.Equal(hp.VocabSize, cfg.VocabSize);
        Assert.Equal(hp.NumLayers, cfg.NumTargetLayers);
        Assert.Equal(hp.EmbeddingDim, cfg.HiddenSize);

        // ── DSpark decode: fresh pass B, taps enabled BEFORE prefill. ──
        using var target = NewFwd(model, cpu, hp);
        using var st = SafetensorsLoader.Open(Path.Combine(headDir, "model.safetensors"));
        using var draft = new DSparkDraftModel(cfg, st, target.MaxSeqLen);
        ((IForwardPass)target).EnableHiddenTaps(cfg.TargetLayerIds);

        var prefillLogits = target.Prefill(Prompt);
        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(Prompt.Length, prefillLogits);

        var emitted = new List<int>();
        decoder.Decode(DecodeTokens, [], emitted.Add);

        Assert.Equal(baseline, emitted);
        Assert.True(decoder.TotalDraftsEmitted > 0,
            "The DSpark head never emitted a draft — greedy parity was trivially " +
            "satisfied by plain decode steps, which defeats the test.");
    }
}
