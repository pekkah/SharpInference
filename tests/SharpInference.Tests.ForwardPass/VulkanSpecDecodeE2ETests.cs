using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vulkan;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Issue #308 PR2 (the finale): end-to-end speculative-decoding correctness on the dense Vulkan
/// (full-offload) path. The CLI now admits Vulkan full-offload of a dense Q4_K/Q6_K model as a
/// spec target for <c>--draft-lookup</c> (prompt-lookup, no draft model); this test exercises the
/// whole wired path — <see cref="SpeculativeDecoder"/> → <see cref="GpuForwardPass.BatchVerify"/>
/// → accept/reject → <see cref="GpuForwardPass.TruncateTo"/> rollback — on real generation.
///
/// Speculative decoding is LOSSLESS at greedy (temp 0): the draft only proposes, and every emitted
/// token is the argmax of the TARGET's logits. So the spec-on stream must EXACTLY equal the target's
/// own non-spec greedy continuation. Asserting exact token equality (not a per-logit tolerance)
/// makes any verify/accept/rollback bug fail the test — divergence means spec is not lossless on
/// Vulkan, which is the load-bearing correctness gate for the feature.
///
/// Runs on GPU and silent-skips when Vulkan or the Q4_K GGUF is unavailable.
/// </summary>
public sealed class VulkanSpecDecodeE2ETests
{
    // Q4_K_M dense model = the batched-trunk-eligible target (CanBatchedTrunk == true). This is
    // the model the CLI --draft-lookup Vulkan path is meant for.
    private const string BatchedModel = "Qwen3-8B-Q4_K_M.gguf";

    private readonly ITestOutputHelper _out;
    public VulkanSpecDecodeE2ETests(ITestOutputHelper output) => _out = output;

    // A deliberately repetitive prompt so prompt-lookup's tail n-gram matching (NgramMin=2)
    // actually proposes tokens during decode — this drives the accept/reject/rollback machinery
    // rather than degrading to plain single-token steps.
    private static readonly int[] Prompt =
    {
        9707, 11, 1879, 0,        // "Hello, world!"
        9707, 11, 1879, 0,        // repeated
        358, 1079, 264, 4108, 1614, 13,
        358, 1079, 264, 4108, 1614, 13,
    };

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? FindModelPath(string modelFile)
    {
        string[] absoluteCandidates =
        {
            $@"C:\p\sharpi\models\{modelFile}",
            $@"E:\models\{modelFile}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", modelFile);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    // SnapKV pinned off: BatchVerify is unsupported once SnapKV evicts, and VRAM-scaled
    // auto-SnapKV could otherwise engage and flip SupportsBatchVerify to false. Mirrors
    // VulkanSpecBatchVerifyTests.NewFwd.
    private static GpuForwardPass NewFwd(GgufModel model, VulkanBackend gpu, ModelHyperparams hp, int ctx)
    {
        var prev = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", "0");
        try { return new GpuForwardPass(model, gpu, hp, maxContextLength: ctx); }
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
    /// E2E greedy parity (prompt-lookup mode): with the n-gram lookup draft (zero draft forwards),
    /// the emitted stream must EXACTLY equal the Vulkan target's non-spec greedy continuation.
    /// Accepted proposals only ever shortcut tokens the target would have picked anyway, so any
    /// difference is a BatchVerify / accept / rollback bug. This is the PR2 correctness gate.
    /// </summary>
    [Fact]
    public void Qwen3_8B_SpecDecode_PromptLookup_GreedyParity_E2E()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;
        var path = FindModelPath(BatchedModel);
        if (path is null) return;

        const int DecodeTokens = 40;

        using var model = GgufModel.Open(path);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.Null(hp.LayerHeadDim);   // dense, not Gemma-4
        Assert.False(hp.IsMoE);

        // ctx must hold the prompt + the decoded tail with headroom.
        int ctx = Prompt.Length + DecodeTokens + 16;
        using var target = NewFwd(model, gpu, hp, ctx);
        Assert.True(target.SupportsBatchVerify,
            "Dense Qwen3-8B Q4_K_M must report SupportsBatchVerify on the Vulkan path.");
        Assert.True(target.CanBatchedTrunk,
            "Qwen3-8B Q4_K_M weights must qualify for the weight-amortized batched trunk.");

        // Non-spec greedy baseline on the target alone.
        target.ResetCache();
        var logits = target.Prefill(Prompt);
        int P = Prompt.Length;
        var baseline = new List<int>();
        int tok = Argmax(logits);
        for (int i = 0; i < DecodeTokens; i++)
        {
            baseline.Add(tok);
            logits = target.Forward(tok, P + i);
            tok = Argmax(logits);
        }

        // Prompt-lookup spec decode over the same target — drives BatchVerify + rollback.
        target.ResetCache();
        var targetLogits = target.Prefill(Prompt).ToArray();
        var spec = new SpeculativeDecoder(target, new PromptLookupDraft(), lookahead: 4);
        spec.Initialize(Prompt, targetLogits);

        var emitted = new List<int>();
        spec.Decode(DecodeTokens, [], emitted.Add);

        _out.WriteLine($"decoded={emitted.Count} acceptanceRate={spec.AcceptanceRate:P1}");
        Assert.Equal(baseline, emitted);
    }
}
