namespace SharpInference.Tests.Vision;

/// <summary>Locates the Gemma 4 12B mmproj + text model + golden fixtures across dev machines.</summary>
internal static class VisionTestPaths
{
    public const string MmprojFile = "mmproj-gemma-4-12b-it-qat-q4_0.gguf";
    public const string TextModelFile = "gemma-4-12b-it-qat-q4_0.gguf";

    private static string? FindModel(string file)
    {
        string[] candidates = { $@"E:\models\{file}", $@"C:\p\sharpi\models\{file}" };
        foreach (var p in candidates)
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

    public static string? FindMmproj() => FindModel(MmprojFile);
    public static string? FindTextModel() => FindModel(TextModelFile);

    /// <summary>Repo-root-relative golden fixtures produced by scripts/gemma4uv_ref.py.</summary>
    public static string? FindFixtureDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "tests", "fixtures", "gemma4uv");
            if (Directory.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }
}
