using SharpInference.Core;
using SharpInference.Cuda;
using SharpInference.Engine;
using SharpInference.Vision;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Image→text for Gemma 4 on the CUDA partial-offload <see cref="CudaHybridForwardPass"/>
/// (issue #252): preprocess → encoder-free projector (CPU) → splice soft tokens into the
/// hybrid decoder via <see cref="CudaHybridForwardPass.ForwardEmbedding"/> → greedy decode.
/// The partial-offload analogue of <see cref="Gemma4VisionE2ETests"/> (CPU) and the
/// full-offload <see cref="CudaForwardPass.ForwardEmbedding"/>; it pins the new hybrid
/// <c>ForwardEmbedding</c> seam, which previously did not exist (vision required CPU
/// <c>-g 0</c> or full CUDA <c>-g -1</c>).
///
/// The load-bearing check is <see cref="Gemma4_CudaHybrid_Image_MatchesCudaFull_Argmax"/>:
/// the SAME image + prompt run through the hybrid (a mostly-GPU split with a few CPU layers)
/// and through full offload must produce the SAME greedy argmax stream. The hybrid GPU half
/// shares kernels with the full path, so the only thing that can diverge is the new
/// ForwardEmbedding seam (no sqrt(d) scale, PLE-from-token-0, the CPU-tier trunk) — exactly
/// what this verifies. q4_0 is argmax-stable across the (identical) GPU kernels and the small
/// CPU tail, so the streams match token-for-token over a short budget.
///
/// Uses <c>gemma-4-12b-it-qat-q4_0.gguf</c> (the model the rest of the CUDA 12B suite uses) +
/// the gemma4uv projector <c>mmproj-gemma-4-12b-it-qat-q4_0.gguf</c>. KV is pinned to fp32 so
/// the hybrid and full passes share the cache dtype regardless of the auto-narrowing budget.
/// Silent-skips when CUDA is unavailable or the GGUFs aren't on disk. The orchestrator runs
/// the heavy GPU verification; the implementation pass only builds.
/// </summary>
public sealed class Gemma4VisionCudaHybridE2ETests : IDisposable
{
    private readonly ITestOutputHelper _out;

    // Pin fp32 KV so the hybrid and full passes use the same cache dtype (CudaForwardPass
    // auto-narrows to bf16/q8_0 when fp32 won't fit the budget — see Gemma4Cuda12BForwardPassTests).
    private readonly string? _prevKvDType = Environment.GetEnvironmentVariable("SHARPI_KV_DTYPE");

    public Gemma4VisionCudaHybridE2ETests(ITestOutputHelper output)
    {
        _out = output;
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", "fp32");
    }

    public void Dispose() =>
        Environment.SetEnvironmentVariable("SHARPI_KV_DTYPE", _prevKvDType);

    private const string TextModel = "gemma-4-12b-it-qat-q4_0.gguf";
    private const string Mmproj = "mmproj-gemma-4-12b-it-qat-q4_0.gguf";

    private static CudaBackend? TryCreate()
    {
        if (!CudaBackend.IsAvailable()) return null;
        try { return CudaBackend.Create(); }
        catch { return null; }
    }

    private static string? Find(string file)
    {
        foreach (var p in new[] { $@"E:\models\{file}", $@"C:\p\sharpi\models\{file}" })
            if (File.Exists(p)) return p;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "models", file);
            if (File.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    /// <summary>
    /// A mostly-GPU split: GPU gets all but the last <paramref name="cpuLayers"/> layers, so the
    /// 12B runs almost entirely on the GPU (the task's "few tokens, mostly-GPU" guidance — avoid
    /// heavy CPU). Still exercises both tiers' Gemma 4 ForwardEmbedding trunk (re-upload seam +
    /// CPU-tier layers + finalise). The byte fields only feed LayerPlacement.Summary;
    /// RecommendedCtxSize sets the hybrid's max sequence length.
    /// </summary>
    private static LayerPlacement MostlyGpuSplit(ModelHyperparams hp, int cpuLayers)
    {
        int gpu = hp.NumLayers - cpuLayers;
        return new LayerPlacement(
            GpuLayers: gpu,
            CpuLayers: cpuLayers,
            GpuWeightBytes: 0,
            GpuKvBytes: 0,
            RecommendedCtxSize: 4096);
    }

    [Fact]
    public void Gemma4_CudaHybrid_Image_MatchesCudaFull_Argmax()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;                                 // CUDA-gated
        var textPath = Find(TextModel);
        var mmprojPath = Find(Mmproj);
        if (textPath is null || mmprojPath is null) return;      // model-gated

        using var model = GgufModel.Open(textPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);                         // real gemma4 GGUF
        var tok = GgufTokenizer.FromGgufModel(model);

        // Project the image once on the CPU embedder; both passes consume the same soft tokens.
        using var vision = VisionModel.Open(mmprojPath);
        var embedder = new GemmaUvVisionEmbedder(vision);
        int embd = hp.EmbeddingDim;
        var img = ImagePreprocessor.Preprocess(SolidColor(96, 96, 220, 30, 30), 96, 96, vision);
        var soft = embedder.Forward(img.Chw, img.Height, img.Width, out int nTok);

        // Image marker ids (gemma4uv runtime wrapping: <|image> ... soft ... <image|>).
        int imgOpen = tok.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
        int imgClose = tok.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;

        // Chat-formatted prompt halves around the image (mirrors Gemma4VisionE2ETests).
        var pre = tok.Encode("<|turn>user\n");
        var post = tok.Encode("What color is this image? Answer in one word.<turn|>\n<|turn>model\n");
        var postIds = post[0] == tok.BosTokenId ? post.Skip(1).ToList() : post.ToList();

        // Full offload (-g -1) reference.
        List<int> fullOut;
        using (var full = new CudaForwardPass(model, gpu, hp, maxContextLength: 4096))
        {
            Assert.True(full.SupportsEmbeddingInput);
            fullOut = RunImagePrompt(full, soft, nTok, embd, pre, imgOpen, imgClose, postIds, tok, "FULL");
        }

        // Partial offload (-g N): all but the last 2 layers on GPU — mostly-GPU, still both tiers.
        List<int> hybridOut;
        using (var hybrid = new CudaHybridForwardPass(model, gpu, hp, MostlyGpuSplit(hp, cpuLayers: 2)))
        {
            Assert.True(hybrid.SupportsEmbeddingInput,
                "CudaHybridForwardPass must report SupportsEmbeddingInput for the gemma4 (#252) vision seam.");
            hybridOut = RunImagePrompt(hybrid, soft, nTok, embd, pre, imgOpen, imgClose, postIds, tok, "HYBRID");
        }

        // Each path is independently coherent (catches an all-EOS / degenerate seam).
        AssertCoherent(fullOut, tok.EosTokenId, "full");
        AssertCoherent(hybridOut, tok.EosTokenId, "hybrid");

        // The load-bearing parity check: hybrid (-g N) == full (-g -1) argmax, token-for-token.
        Assert.True(fullOut.SequenceEqual(hybridOut),
            $"CUDA-hybrid (-g N) and CUDA-full (-g -1) ForwardEmbedding decodes diverge — the hybrid " +
            $"vision seam (no sqrt(d), PLE-from-token-0, CPU-tier trunk) is not faithful to the full path. " +
            $"full=[{string.Join(",", fullOut)}] (\"{tok.Decode(fullOut)}\") " +
            $"hybrid=[{string.Join(",", hybridOut)}] (\"{tok.Decode(hybridOut)}\").");

        // Correctness sanity: the solid-red image names "red".
        Assert.Contains("red", tok.Decode(hybridOut), StringComparison.OrdinalIgnoreCase);
    }

    private List<int> RunImagePrompt(
        IForwardPass fwd, float[] soft, int nTok, int embd,
        IReadOnlyList<int> pre, int imgOpen, int imgClose, IReadOnlyList<int> post,
        GgufTokenizer tok, string label)
    {
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in pre) logits = fwd.Forward(id, pos++);
        logits = fwd.Forward(imgOpen, pos++);
        for (int t = 0; t < nTok; t++)
            logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
        logits = fwd.Forward(imgClose, pos++);
        foreach (int id in post) logits = fwd.Forward(id, pos++);

        var outIds = new List<int>();
        int next = Argmax(logits);
        for (int i = 0; i < 10 && next != tok.EosTokenId; i++)
        {
            outIds.Add(next);
            logits = fwd.Forward(next, pos++);
            next = Argmax(logits);
        }
        _out.WriteLine($"[{label}] {nTok} soft tokens -> {outIds.Count} decoded: \"{tok.Decode(outIds)}\" {string.Join(",", outIds)}");
        return outIds;
    }

    private static void AssertCoherent(List<int> outIds, int eos, string which)
    {
        Assert.True(outIds.Count > 0, $"[{which}] decoded nothing (immediate EOS).");
        Assert.DoesNotContain(eos, outIds);                // we stop before EOS, so none present
        Assert.True(outIds.Distinct().Count() >= 2,
            $"[{which}] decode is degenerate (only {outIds.Distinct().Count()} distinct of {outIds.Count}): {string.Join(",", outIds)}");
    }

    private static byte[] SolidColor(int w, int h, byte r, byte g, byte b)
    {
        var buf = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++) { buf[i * 3] = r; buf[i * 3 + 1] = g; buf[i * 3 + 2] = b; }
        return buf;
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
}
