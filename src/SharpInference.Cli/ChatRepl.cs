using System.Diagnostics;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SharpInference.Cli;

/// <summary>Interactive chat REPL with token streaming.</summary>
public static class ChatRepl
{
    public static async Task RunAsync(string[] args)
    {
        // Parse args
        string? modelPath = null;
        float temperature = 0.7f;
        int topK = 40;
        float topP = 0.9f;
        int maxTokens = 256;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model" or "-m" when i + 1 < args.Length:
                    modelPath = args[++i]; break;
                case "--temp" when i + 1 < args.Length:
                    temperature = float.Parse(args[++i]); break;
                case "--top-k" when i + 1 < args.Length:
                    topK = int.Parse(args[++i]); break;
                case "--top-p" when i + 1 < args.Length:
                    topP = float.Parse(args[++i]); break;
                case "--max-tokens" when i + 1 < args.Length:
                    maxTokens = int.Parse(args[++i]); break;
                case "--help" or "-h":
                    PrintHelp();
                    return;
            }
        }

        if (modelPath is null)
        {
            modelPath = "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
            if (!File.Exists(modelPath))
            {
                Console.Error.WriteLine("Error: No model specified. Use --model <path>");
                PrintHelp();
                return;
            }
        }

        Console.WriteLine($"Loading model: {modelPath}");
        using var model = GgufModel.Open(modelPath);

        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
        Console.WriteLine($"  Architecture: {model.Metadata["general.architecture"]}");
        Console.WriteLine($"  Layers: {hp.NumLayers}, Dim: {hp.EmbeddingDim}, Heads: {hp.NumHeads}/{hp.NumKvHeads}");
        Console.WriteLine($"  Context: {hp.ContextLength}, Vocab: {hp.VocabSize}");

        Console.WriteLine("Loading tokenizer...");
        var tokenizer = GgufTokenizer.FromGgufModel(model);

        Console.WriteLine("Initializing engine...");
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(model, backend, hp);

        var samplingParams = new SamplingParams
        {
            Temperature = temperature,
            TopK = topK,
            TopP = topP,
            MaxNewTokens = maxTokens,
            StopTokenIds = [tokenizer.EosTokenId],
        };

        Console.WriteLine("Ready. Type your message (Ctrl+C to exit).\n");

        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine();
            if (input is null) break;
            if (string.IsNullOrWhiteSpace(input)) continue;

            var prompt = FormatPrompt(input);
            var tokens = tokenizer.Encode(prompt);

            fwd.Cache.Reset();
            Generate(fwd, tokenizer, tokens, samplingParams);
            Console.WriteLine("\n");
        }

        await Task.CompletedTask;
    }

    private static void Generate(
        ForwardPass fwd,
        GgufTokenizer tokenizer,
        IReadOnlyList<int> promptTokens,
        SamplingParams sampling)
    {
        var sw = Stopwatch.StartNew();

        // Prefill: run all prompt tokens through the model
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < promptTokens.Count; i++)
            logits = fwd.Forward(promptTokens[i], i);

        var prefillTime = sw.Elapsed;
        Console.Error.WriteLine(
            $"[prefill: {promptTokens.Count} tokens in {prefillTime.TotalSeconds:F1}s " +
            $"({promptTokens.Count / prefillTime.TotalSeconds:F1} tok/s)]");

        // Decode: generate new tokens one at a time
        int generated = 0;
        var decodeStart = sw.Elapsed;

        for (int i = 0; i < sampling.MaxNewTokens; i++)
        {
            int nextToken = Sampler.Sample(logits, sampling);

            // Check stop conditions
            if (sampling.StopTokenIds is not null && Array.IndexOf(sampling.StopTokenIds, nextToken) >= 0)
                break;

            // Decode and print the token
            var text = tokenizer.Decode([nextToken]);
            Console.Write(text);

            generated++;

            // Forward pass for this new token
            int position = promptTokens.Count + i;
            logits = fwd.Forward(nextToken, position);
        }

        var decodeTime = sw.Elapsed - decodeStart;
        if (generated > 0)
        {
            Console.Error.WriteLine(
                $"\n[decode: {generated} tokens in {decodeTime.TotalSeconds:F1}s " +
                $"({generated / decodeTime.TotalSeconds:F1} tok/s)]");
        }
    }

    private static string FormatPrompt(string userMessage)
    {
        // SmolLM2-Instruct uses ChatML format
        return $"<|im_start|>user\n{userMessage}<|im_end|>\n<|im_start|>assistant\n";
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            Usage: SharpInference [options]

            Options:
              --model, -m <path>   Path to GGUF model file
              --temp <float>       Temperature (default: 0.7)
              --top-k <int>        Top-K (default: 40)
              --top-p <float>      Top-P nucleus (default: 0.9)
              --max-tokens <int>   Max tokens to generate (default: 256)
              --help, -h           Show this help
            """);
    }
}
