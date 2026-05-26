using SharpInference.Core;

namespace SharpInference.Tests.ForwardPass;

/// <summary>
/// Investigation harness: dump sharpi's tokenization of the rendered chat
/// prompt to compare with llama.cpp's. Runs only when SHARPI_REPRO_POS13=1.
/// </summary>
public sealed class Repro_Tokenize
{
    private static string? FindMtpModelPath()
    {
        string[] absoluteCandidates =
        {
            @"C:\p\sharpi\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
            @"E:\models\Qwen3.6-27B-MTP-Q4_K_M.gguf",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;
        return null;
    }

    [Fact]
    public void DumpRenderedPromptTokens()
    {
        if (Environment.GetEnvironmentVariable("SHARPI_REPRO_POS13") != "1") return;

        var path = FindMtpModelPath();
        Assert.NotNull(path);

        using var model = GgufModel.Open(path!);
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        string rendered =
            "<|im_start|>user\n" +
            "The capital of France is<|im_end|>\n" +
            "<|im_start|>assistant\n\n";

        var tokens = tokenizer.Encode(rendered).ToList();

        Directory.CreateDirectory(@"C:\p\sharpi\tmp");
        var path2 = @"C:\p\sharpi\tmp\sharpi_tokenize_rendered.txt";
        using var w = new StreamWriter(path2);
        w.WriteLine($"# sharpi tokenization of rendered chat prompt");
        w.WriteLine($"# rendered bytes (UTF-8): {string.Join(",", System.Text.Encoding.UTF8.GetBytes(rendered))}");
        w.WriteLine($"# token_count = {tokens.Count}");
        w.WriteLine($"# tokens = [{string.Join(",", tokens)}]");
        w.WriteLine($"# per-token decode:");
        for (int i = 0; i < tokens.Count; i++)
        {
            var s = tokenizer.Decode(new[] { tokens[i] });
            var escaped = s.Replace("\n", "\\n").Replace("\t", "\\t");
            w.WriteLine($"  {i,3}  {tokens[i],7}  '{escaped}'");
        }

        // Also probe a couple of tokens we care about: 271, 1358, 198.
        w.WriteLine("# special-of-interest:");
        foreach (var id in new[] { 198, 271, 1358 })
        {
            var s = tokenizer.Decode(new[] { id });
            var escaped = s.Replace("\n", "\\n").Replace("\t", "\\t");
            w.WriteLine($"  id={id} → '{escaped}' (bytes: {string.Join(",", System.Text.Encoding.UTF8.GetBytes(s))})");
        }

        Console.Error.WriteLine($"[repro] wrote {path2}");
    }
}
