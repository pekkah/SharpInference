using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;
using SharpInference.Vision;
using Xunit.Abstractions;
using EForwardPass = SharpInference.Engine.ForwardPass;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// End-to-end CPU image→text for Gemma 4 12B (issue #250): preprocess → encoder-free
/// projector → splice soft tokens into the decoder via <see cref="ForwardPass.ForwardEmbedding"/>
/// → greedy decode. Validates (a) coherence (finite, varied, not immediately EOS) and
/// (b) image-dependence — two different images must yield different decode streams, proving
/// the vision embeddings actually flow into and steer the language model.
/// </summary>
public sealed class Gemma4VisionE2ETests
{
    private readonly ITestOutputHelper _out;
    public Gemma4VisionE2ETests(ITestOutputHelper output) => _out = output;

    private const string TextModel = "gemma-4-12b-it-qat-q4_0.gguf";
    private const string Mmproj = "mmproj-gemma-4-12b-it-qat-q4_0.gguf";

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

    [Fact]
    public void Gemma4_12B_Image_SplicesAndSteersDecode()
    {
        var textPath = Find(TextModel);
        var mmprojPath = Find(Mmproj);
        if (textPath is null || mmprojPath is null) return;   // model-gated

        using var model = GgufModel.Open(textPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tok = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        using var fwd = new EForwardPass(model, backend, hp);
        Assert.True(fwd.SupportsEmbeddingInput);

        using var vision = VisionModel.Open(mmprojPath);
        var embedder = new GemmaUvVisionEmbedder(vision);
        int embd = hp.EmbeddingDim;

        // Image marker ids (gemma4uv runtime wrapping: <|image> ... soft ... <image|>).
        int imgOpen = tok.SpecialTokens.TryGetValue("<|image>", out var o) ? o : 255999;
        int imgClose = tok.SpecialTokens.TryGetValue("<image|>", out var c) ? c : 258882;

        // Chat-formatted prompt halves around the image. Gemma 4's turn format is
        // <|turn>role\n…<turn|> (NOT Gemma 3's <start_of_turn>); Encode recognizes the
        // special-token strings and prepends BOS to each call, so strip it on the tail.
        var pre = tok.Encode("<|turn>user\n");
        var post = tok.Encode("What color is this image? Answer in one word.<turn|>\n<|turn>model\n");
        var postIds = post[0] == tok.BosTokenId ? post.Skip(1).ToList() : post.ToList();

        var redOut = RunImagePrompt(fwd, embedder, vision, embd, pre, imgOpen, imgClose, postIds,
                                    SolidColor(96, 96, 220, 30, 30), tok, "RED");
        fwd.ResetCache();
        var blueOut = RunImagePrompt(fwd, embedder, vision, embd, pre, imgOpen, imgClose, postIds,
                                     SolidColor(96, 96, 30, 30, 220), tok, "BLUE");

        // Coherence: each decode is non-degenerate.
        AssertCoherent(redOut, tok.EosTokenId, "red");
        AssertCoherent(blueOut, tok.EosTokenId, "blue");

        // Image-dependence: different images must produce different token streams.
        Assert.False(redOut.SequenceEqual(blueOut),
            "red and blue images produced identical decodes — vision embeddings are not steering the model.");

        // Correctness: greedy temp-0 decode names the dominant color (verified via the CLI —
        // the solid-red image decodes "Red" with a wide logit margin, then stops on <turn|>).
        Assert.Contains("red", tok.Decode(redOut), StringComparison.OrdinalIgnoreCase);
    }

    private List<int> RunImagePrompt(
        EForwardPass fwd, GemmaUvVisionEmbedder embedder, VisionModel vision, int embd,
        IReadOnlyList<int> pre, int imgOpen, int imgClose, IReadOnlyList<int> post,
        byte[] rgb, GgufTokenizer tok, string label)
    {
        var img = ImagePreprocessor.Preprocess(rgb, 96, 96, vision);
        var soft = embedder.Forward(img.Chw, img.Height, img.Width, out int nTok);

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
