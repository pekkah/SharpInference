using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// DSpark Phase 4 (GPU draft path) validation on real models — silent-skips
/// when CUDA, the Qwen3-4B GGUF, or the dspark_qwen3_4b_block7 head directory
/// is absent (mirrors <see cref="DSparkE2ETests"/> / CudaSpecBatchVerifyTests).
///
/// The load-bearing guarantee is unchanged from the CPU path: DSpark-decoded
/// output must be byte-identical to plain greedy decode ON THE SAME TARGET
/// PASS — the draft (CPU f32 or CUDA fp16) only proposes; every emitted token
/// is argmax of target logits. Draft numerics affect acceptance rate only,
/// which is asserted to be positive so parity isn't vacuously satisfied by
/// all-rejected drafts.
/// </summary>
public sealed class CudaDSparkE2ETests
{
    private const int DecodeTokens = 24;

    // "Hello, world! I am a virtual model." — same ids as DSparkE2ETests.
    private static readonly int[] Prompt = { 9707, 11, 1879, 0, 358, 1079, 264, 4108, 1614, 13 };

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); } catch { return null; }
    }

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

    private static string? FindHeadDir()
    {
        static bool IsValid(string d) =>
            File.Exists(Path.Combine(d, "model.safetensors"))
            && File.Exists(Path.Combine(d, "config.json"));
        foreach (var root in new[] { @"C:\models", @"E:\models", @"C:\p\sharpi\models" })
        {
            var d = Path.Combine(root, "dspark_qwen3_4b_block7");
            if (IsValid(d)) return d;
        }
        return null;
    }

    private static (string Gguf, string HeadDir)? FindModels()
    {
        if (Environment.GetEnvironmentVariable("SHARPI_SKIP_E2E") == "1") return null;
        var gguf = FindModelPath("Qwen3-4B-Q4_K_M.gguf");
        var head = FindHeadDir();
        if (gguf is null || head is null) return null;
        return (gguf, head);
    }

    /// <summary>
    /// CUDA tap capture parity vs the CPU pass: the same prompt prefilled on
    /// both passes with the head's tap layers must produce closely matching
    /// per-position tap rows (different GEMM orders → tolerance, not bit-parity;
    /// the draft consumes these through its own fc projection, so relative
    /// closeness is the meaningful property).
    /// </summary>
    [Fact]
    public void CudaTaps_MatchCpuTaps_OnPrefill()
    {
        if (FindModels() is not var (ggufPath, headDir) || ggufPath is null) return;
        using var cuda = TryCreate();
        if (cuda is null) return;

        var cfg = DSparkConfig.FromJsonFile(Path.Combine(headDir, "config.json"));
        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        using var cpuBackend = new CpuBackend();
        using var cpuFwd = new SharpInference.Engine.ForwardPass(model, cpuBackend, hp, maxContextLength: 256);
        ((IForwardPass)cpuFwd).EnableHiddenTaps(cfg.TargetLayerIds);
        cpuFwd.Prefill(Prompt);

        using var cudaFwd = new CudaForwardPass(model, cuda, hp, maxContextLength: 256);
        ((IForwardPass)cudaFwd).EnableHiddenTaps(cfg.TargetLayerIds);
        cudaFwd.Prefill(Prompt);

        for (int pos = 0; pos < Prompt.Length; pos++)
        {
            var cpuTap = ((IForwardPass)cpuFwd).HiddenTapsAt(pos);
            var cudaTap = ((IForwardPass)cudaFwd).HiddenTapsAt(pos);
            Assert.Equal(cfg.TapDim, cpuTap.Length);
            Assert.Equal(cfg.TapDim, cudaTap.Length);

            // Relative L2 distance per row: CPU f32 matvec vs CUDA fp16-dequant
            // matvec accumulate differently; the rows must still be the same
            // vectors to a small relative error.
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < cpuTap.Length; i++)
            {
                dot += (double)cpuTap[i] * cudaTap[i];
                na += (double)cpuTap[i] * cpuTap[i];
                nb += (double)cudaTap[i] * cudaTap[i];
            }
            double cosine = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30);
            Assert.True(cosine > 0.995,
                $"Tap row at position {pos} diverges between CPU and CUDA (cosine {cosine:F6}).");
        }
    }

    /// <summary>Full-GPU DSpark: CUDA target + CUDA fp16 draft, byte-exact greedy parity.</summary>
    [Fact]
    public void DSpark_CudaTarget_CudaDraft_GreedyParity_E2E()
    {
        if (FindModels() is not var (ggufPath, headDir) || ggufPath is null) return;
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        // Baseline: plain greedy on a fresh CUDA pass, disposed before the DSpark run.
        var baseline = new List<int>();
        using (var fwd = new CudaForwardPass(model, cuda, hp, maxContextLength: 512))
        {
            var logits = fwd.Prefill(Prompt);
            int tok = Argmax(logits);
            for (int i = 0; i < DecodeTokens; i++)
            {
                baseline.Add(tok);
                logits = fwd.Forward(tok, Prompt.Length + i);
                tok = Argmax(logits);
            }
        }

        var cfg = DSparkConfig.FromJsonFile(Path.Combine(headDir, "config.json"));
        using var target = new CudaForwardPass(model, cuda, hp, maxContextLength: 512);
        using var st = SafetensorsLoader.Open(Path.Combine(headDir, "model.safetensors"));
        using var draft = new CudaDSparkDraftModel(cfg, st, cuda, target.MaxSeqLen);
        ((IForwardPass)target).EnableHiddenTaps(cfg.TargetLayerIds);

        var prefillLogits = target.Prefill(Prompt);
        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(Prompt.Length, prefillLogits);

        var emitted = new List<int>();
        decoder.Decode(DecodeTokens, [], emitted.Add);

        Assert.Equal(baseline, emitted);
        Assert.True(decoder.TotalDraftsEmitted > 0, "The CUDA draft never proposed.");
        Assert.True(decoder.TotalDraftsAccepted > 0,
            "No CUDA-drafted token was ever accepted — parity is vacuous and the fp16 " +
            "backbone numerics are likely broken.");
    }

    /// <summary>Heterogeneous: CUDA target + CPU f32 draft (spec §3's Cpu placement).</summary>
    [Fact]
    public void DSpark_CudaTarget_CpuDraft_GreedyParity_E2E()
    {
        if (FindModels() is not var (ggufPath, headDir) || ggufPath is null) return;
        using var cuda = TryCreate();
        if (cuda is null) return;

        using var model = GgufModel.Open(ggufPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);

        var baseline = new List<int>();
        using (var fwd = new CudaForwardPass(model, cuda, hp, maxContextLength: 512))
        {
            var logits = fwd.Prefill(Prompt);
            int tok = Argmax(logits);
            for (int i = 0; i < DecodeTokens; i++)
            {
                baseline.Add(tok);
                logits = fwd.Forward(tok, Prompt.Length + i);
                tok = Argmax(logits);
            }
        }

        var cfg = DSparkConfig.FromJsonFile(Path.Combine(headDir, "config.json"));
        using var target = new CudaForwardPass(model, cuda, hp, maxContextLength: 512);
        using var st = SafetensorsLoader.Open(Path.Combine(headDir, "model.safetensors"));
        using var draft = new DSparkDraftModel(cfg, st, target.MaxSeqLen);
        ((IForwardPass)target).EnableHiddenTaps(cfg.TargetLayerIds);

        var prefillLogits = target.Prefill(Prompt);
        var decoder = new DSparkDecoder(target, draft);
        decoder.Initialize(Prompt.Length, prefillLogits);

        var emitted = new List<int>();
        decoder.Decode(DecodeTokens, [], emitted.Add);

        Assert.Equal(baseline, emitted);
        Assert.True(decoder.TotalDraftsAccepted > 0,
            "No CPU-drafted token accepted against the CUDA target — the tap handoff is likely wrong.");
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
