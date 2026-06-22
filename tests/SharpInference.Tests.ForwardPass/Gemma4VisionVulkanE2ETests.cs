using SharpInference.Core;
using SharpInference.Engine;
using SharpInference.Vision;
using SharpInference.Vulkan;
using Vortice.Vulkan;
using Xunit.Abstractions;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end image→text for Gemma 4 on the Vulkan <see cref="GpuForwardPass"/> (issue #252):
/// preprocess → encoder-free projector (CPU) → splice soft tokens into the Vulkan decoder via
/// <see cref="GpuForwardPass.ForwardEmbedding"/> → greedy decode. The Vulkan analogue of
/// <see cref="Gemma4VisionE2ETests"/>; it pins the new <c>ForwardEmbedding</c> seam end-to-end.
///
/// Validates (a) the capability flag (<see cref="IForwardPass.SupportsEmbeddingInput"/>),
/// (b) correctness (a solid-RED image decodes the word "red") and (c) image-dependence (a
/// solid-BLUE image does NOT decode "red", proving the vision embeddings actually steer the LM).
///
/// The prompt is rendered through the model's OWN chat template with thinking disabled (mirrors
/// the CLI's <c>RunImagePrompt</c>): <c>gemma4-v2</c> otherwise opens a <c>&lt;|channel&gt;thought</c>
/// block and the one-word answer falls outside a short decode budget. The single <c>&lt;|image|&gt;</c>
/// placeholder token is expanded into <c>&lt;|image&gt;</c> … soft tokens … <c>&lt;image|&gt;</c>.
///
/// Uses <c>gemma4-v2-Q4_K_M.gguf</c> (PLE-off, Vulkan-loadable) + the gemma4uv projector
/// <c>mmproj-gemma-4-12b-it-qat-q4_0.gguf</c> (output dim 3840 — a valid pair). Silent-skips when
/// Vulkan is unavailable, the device is out of memory for full offload, or the GGUFs aren't on
/// disk. NOT run by the implementation pass (the orchestrator verifies on a real GPU).
/// </summary>
public sealed class Gemma4VisionVulkanE2ETests
{
    private readonly ITestOutputHelper _out;
    public Gemma4VisionVulkanE2ETests(ITestOutputHelper output) => _out = output;

    private const string TextModel = "gemma4-v2-Q4_K_M.gguf";
    private const string Mmproj = "mmproj-gemma-4-12b-it-qat-q4_0.gguf";

    private static VulkanBackend? TryCreate()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }

    private static string? Find(string file)
    {
        foreach (var p in new[] { $@"C:\p\sharpi\models\{file}", $@"E:\models\{file}" })
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

    [Fact]
    public void Gemma4_Vulkan_Image_SplicesAndSteersDecode()
    {
        using var gpu = TryCreate();
        if (gpu is null) return;                                // Vulkan-gated
        var textPath = Find(TextModel);
        var mmprojPath = Find(Mmproj);
        if (textPath is null || mmprojPath is null) return;     // model-gated

        using var model = GgufModel.Open(textPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        Assert.NotNull(hp.LayerHeadDim);                        // real gemma4 GGUF
        var tok = GgufTokenizer.FromGgufModel(model);
        Assert.NotNull(tok.ChatTemplate);                       // need the template to suppress thinking

        GpuForwardPass fwd;
        try
        {
            fwd = new GpuForwardPass(model, gpu, hp);
        }
        catch (VkException ex) when (ex.Result == VkResult.ErrorOutOfDeviceMemory)
        {
            // Full gemma4 offload doesn't fit this device's VRAM — skip silently (the
            // orchestrator verifies end-to-end on a GPU with enough memory).
            return;
        }
        using (fwd)
        {
            Assert.True(fwd.SupportsEmbeddingInput);

            using var vision = VisionModel.Open(mmprojPath);
            var embedder = new GemmaUvVisionEmbedder(vision);
            int embd = hp.EmbeddingDim;

            // Image marker / placeholder ids (gemma4uv runtime wrapping is
            // <|image> ... soft ... <image|>; the chat template emits the <|image|> placeholder).
            int imgOpen = tok.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
            int imgClose = tok.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;
            int placeholder = tok.SpecialTokens.TryGetValue("<|image|>", out var ph) ? ph : 258880;

            var redOut = RunImagePrompt(fwd, embedder, vision, embd, tok,
                                        imgOpen, imgClose, placeholder,
                                        SolidColor(96, 96, 220, 30, 30), "RED");
            fwd.ResetCache();
            var blueOut = RunImagePrompt(fwd, embedder, vision, embd, tok,
                                         imgOpen, imgClose, placeholder,
                                         SolidColor(96, 96, 30, 30, 220), "BLUE");

            // Coherence: each decode is non-degenerate.
            AssertCoherent(redOut, tok.EosTokenId, "red");
            AssertCoherent(blueOut, tok.EosTokenId, "blue");

            // Image-dependence: different images must produce different token streams.
            Assert.False(redOut.SequenceEqual(blueOut),
                "red and blue images produced identical Vulkan decodes — vision embeddings are not steering the model.");

            // Correctness: greedy temp-0 decode names the dominant color.
            Assert.Contains("red", tok.Decode(redOut), StringComparison.OrdinalIgnoreCase);
            // Image-dependence (negative): the blue image must NOT name "red".
            Assert.DoesNotContain("red", tok.Decode(blueOut), StringComparison.OrdinalIgnoreCase);
        }
    }

    private List<int> RunImagePrompt(
        GpuForwardPass fwd, GemmaUvVisionEmbedder embedder, VisionModel vision, int embd,
        GgufTokenizer tok, int imgOpen, int imgClose, int placeholder,
        byte[] rgb, string label)
    {
        var img = ImagePreprocessor.Preprocess(rgb, 96, 96, vision);
        var soft = embedder.Forward(img.Chw, img.Height, img.Width, out int nTok);

        // Render through the model's own chat template with thinking OFF (mirrors the CLI image
        // path). The <|image|> placeholder marks where the image's soft tokens are spliced.
        var messages = JinjaChatTemplate.BuildMessages(
            "<|image|>What color is this image? Answer in one word.");
        var prompt = tok.ChatTemplate!.Render(new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["add_generation_prompt"] = true,
            ["tools"] = null,
            ["enable_thinking"] = false,
        });
        var allTokens = tok.Encode(prompt).ToList();
        Assert.Equal(1, allTokens.Count(t => t == placeholder));

        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int id in allTokens)
        {
            if (id == placeholder)
            {
                logits = fwd.Forward(imgOpen, pos++);
                for (int t = 0; t < nTok; t++)
                    logits = fwd.ForwardEmbedding(soft.AsSpan(t * embd, embd), pos++);
                logits = fwd.Forward(imgClose, pos++);
            }
            else
            {
                logits = fwd.Forward(id, pos++);
            }
        }

        var outIds = new List<int>();
        int next = Argmax(logits);
        for (int i = 0; i < 12 && next != tok.EosTokenId; i++)
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
