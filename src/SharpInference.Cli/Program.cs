using System.Text;
using Spectre.Console.Cli;
using SharpInference.Cli;

// Force UTF-8 for stdin/stdout. On Windows the console defaults to the OEM
// code page, which mangles multi-byte UTF-8 output (CJK, emoji, smart quotes)
// into '?' or replacement glyphs.
Console.OutputEncoding = Encoding.UTF8;
Console.InputEncoding = Encoding.UTF8;

var app = new CommandApp<RunCommand>();
app.Configure(config =>
{
    config.SetApplicationName("sharpi-cli");
    config.SetApplicationVersion("0.1.0");
    config.AddCommand<ListMetadataCommand>("list-metadata")
        .WithDescription("Print all GGUF metadata key/value pairs from a model file");
    config.AddCommand<ListTensorsCommand>("list-tensors")
        .WithDescription("Print the tensor index (name, dtype, shape, bytes) from a model file");
    config.AddCommand<ImageCommand>("image")
        .WithDescription("Generate an image from a text prompt using a native FLUX or Z-Image-Turbo diffusion pipeline (VAE + CLIP-L + T5-XXL + DiT GGUF). See 'sharpi-cli image --help' for required model paths.");
});

return app.Run(RewriteLlamaStyleFlags(args));

// llama.cpp uses single-dash multi-character flags (-ngl, -md, -st, -sys, -dev), but
// Spectre.Console.Cli only allows single-character options after one dash (multi-character
// options must use two dashes). Translate llama's spellings to our equivalent long options
// so muscle-memory copy/paste from llama-cli works. The long forms (--n-gpu-layers, --ngl,
// --device, …) are registered directly and pass through untouched.
//
// Caveat: a token that is *meant* as an option value but happens to equal one of these
// spellings (e.g. `-p -st`) is also rewritten — same ambiguity llama-cli has, and a literal
// "-st" prompt is not a realistic invocation. Translation stops after a bare `--`.
static string[] RewriteLlamaStyleFlags(string[] args)
{
    var map = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["-ngl"] = "--n-gpu-layers",
        ["-md"]  = "--model-draft",
        ["-st"]  = "--single-turn",
        ["-sys"] = "--system-prompt",
        ["-dev"] = "--device",
    };

    var rewritten = new string[args.Length];
    bool endOfOptions = false;
    for (int i = 0; i < args.Length; i++)
    {
        string arg = args[i];
        if (endOfOptions)
        {
            rewritten[i] = arg;
            continue;
        }
        if (arg == "--")
        {
            endOfOptions = true;
            rewritten[i] = arg;
            continue;
        }

        // Support both "-ngl 32" and "-ngl=32".
        int eq = arg.IndexOf('=');
        string key = eq >= 0 ? arg[..eq] : arg;
        rewritten[i] = map.TryGetValue(key, out string? longName)
            ? (eq >= 0 ? longName + arg[eq..] : longName)
            : arg;
    }
    return rewritten;
}
