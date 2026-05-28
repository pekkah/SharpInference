using System.Globalization;
using System.Text;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

namespace SnapKvEval;

/// <summary>
/// Needle-in-haystack accuracy harness for SnapKV prefill-time KV eviction
/// (issue #51, eval task #61).
///
/// Builds a synthetic long prompt of <c>--length</c> tokens that embeds a
/// distinctive "secret" sentence (colour + 4-digit code) at a known position,
/// asks "What is the secret from earlier?", greedy-decodes a small window and
/// checks whether the secret is recovered. Sweeps over a budget × position
/// grid for one or more models on the deterministic CPU backend, then prints
/// a markdown table to stdout.
///
/// Intended to validate the acceptance criterion from issue #51: the
/// SnapKV-evicted decode should recover the needle at the largest budget
/// across every position.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        var opts = Options.Parse(args);
        if (opts is null) return 1;

        var modelPaths = ResolveModels(opts.ModelPath);
        if (modelPaths.Count == 0)
        {
            Console.WriteLine("No model available — pass --model <path> or place a");
            Console.WriteLine("Q4_K_M GGUF for SmolLM2-1.7B-Instruct or Qwen3-8B under");
            Console.WriteLine("C:\\p\\sharpi\\models\\ or E:\\models\\.");
            return 0;
        }

        var rows = new List<ResultRow>();
        foreach (var modelPath in modelPaths)
        {
            var modelLabel = Path.GetFileNameWithoutExtension(modelPath);
            Console.Error.WriteLine($"[eval] model={modelLabel}");
            foreach (var budget in opts.Budgets)
            {
                foreach (var position in opts.Positions)
                {
                    var row = RunOne(modelPath, modelLabel, budget, position, opts);
                    rows.Add(row);
                    Console.Error.WriteLine(
                        $"[eval]   budget={budget,5} pos={position,-9} score={row.Score:F2} recovered={row.Recovered} decoded={Truncate(row.Decoded, 60)}");
                }
            }
        }

        PrintMarkdownTable(rows);
        PrintPassCriterion(rows, opts);
        return 0;
    }

    private static ResultRow RunOne(string modelPath, string modelLabel, int budget,
        string position, Options opts)
    {
        // Set the budget BEFORE constructing ForwardPass — SnapKvConfig is
        // captured at constructor time.
        var prevBudget = Environment.GetEnvironmentVariable("SHARPI_SNAPKV_BUDGET");
        Environment.SetEnvironmentVariable(
            "SHARPI_SNAPKV_BUDGET", budget.ToString(CultureInfo.InvariantCulture));
        try
        {
            using var model = GgufModel.Open(modelPath);
            var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
            using var backend = new CpuBackend();
            using var fwd = new ForwardPass(model, backend, hp);
            var tokenizer = GgufTokenizer.FromGgufModel(model);

            var (prompt, secret) = BuildNeedlePrompt(
                tokenizer, opts.Length, position, opts.Seed);

            var tokens = tokenizer.Encode(prompt).ToArray();

            ReadOnlySpan<float> logits;
            try
            {
                logits = fwd.Prefill(tokens);
            }
            catch (Exception ex)
            {
                return new ResultRow(modelLabel, budget, position, 0.0,
                    Recovered: false, Decoded: "<prefill error>", Note: ex.GetType().Name);
            }

            var decoded = GreedyDecode(fwd, tokenizer, logits, tokens.Length, opts.MaxDecode);

            // Distinctive needle pattern: "<COLOUR>-<NNNN>". Match the 4-digit
            // code + at least a 4-char prefix of the colour separately, since
            // some tokenizers (Qwen3, SmolLM2) BPE-split the colour mid-word
            // and the model emits e.g. "CRIMS-2268" instead of "CRIMSON-2268".
            // 4-digit code is the high-signal part (~0.01 % false-match rate
            // against a random 1000-9999 hallucination); requiring the colour
            // prefix in addition disambiguates against a model that learned a
            // training-data prior.
            int dash = secret.IndexOf('-');
            string colour = dash > 0 ? secret[..dash] : secret;
            string code = dash > 0 ? secret[(dash + 1)..] : secret;
            string colourPrefix = colour[..Math.Min(4, colour.Length)];
            bool recovered = decoded.Contains(code, StringComparison.OrdinalIgnoreCase)
                          && decoded.Contains(colourPrefix, StringComparison.OrdinalIgnoreCase);
            double score = recovered ? 1.0 : 0.0;
            return new ResultRow(modelLabel, budget, position, score,
                Recovered: recovered, Decoded: decoded, Note: $"recovered: {recovered.ToString().ToLowerInvariant()}");
        }
        finally
        {
            Environment.SetEnvironmentVariable("SHARPI_SNAPKV_BUDGET", prevBudget);
        }
    }

    /// <summary>
    /// Greedy-decode up to <paramref name="maxDecode"/> tokens from the prefill
    /// logits, returning the concatenated UTF-8 string. Stops early on the
    /// tokenizer's EOS.
    /// </summary>
    private static string GreedyDecode(ForwardPass fwd, GgufTokenizer tokenizer,
        ReadOnlySpan<float> logits, int promptLen, int maxDecode)
    {
        var produced = new List<int>(maxDecode);
        var current = logits.ToArray();
        for (int i = 0; i < maxDecode; i++)
        {
            int next = Sampler.Greedy(current);
            if (next == tokenizer.EosTokenId) break;
            produced.Add(next);
            ReadOnlySpan<float> step = fwd.Forward(next, promptLen + i);
            current = step.ToArray();
        }
        return tokenizer.Decode(produced);
    }

    /// <summary>
    /// Build a needle-in-haystack prompt of approximately <paramref name="targetLen"/>
    /// tokens. The needle is "The secret code is &lt;COLOUR&gt;-&lt;NNNN&gt;." with the
    /// colour/code derived from <paramref name="seed"/>; it is inserted at
    /// 5% of the haystack for "beginning" and 50% for "middle".
    /// </summary>
    private static (string Prompt, string Secret) BuildNeedlePrompt(
        GgufTokenizer tokenizer, int targetLen, string position, int seed)
    {
        // Public-domain filler text — a mix of pangrams and a Lewis Carroll
        // snippet. Repeated to reach the target token count.
        const string filler =
            "The quick brown fox jumps over the lazy dog. " +
            "Sphinx of black quartz, judge my vow. " +
            "Pack my box with five dozen liquor jugs. " +
            "How vexingly quick daft zebras jump. " +
            "Alice was beginning to get very tired of sitting by her sister on the bank, " +
            "and of having nothing to do: once or twice she had peeped into the book " +
            "her sister was reading, but it had no pictures or conversations in it, " +
            "and what is the use of a book, thought Alice, without pictures or conversation. " +
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor " +
            "incididunt ut labore et dolore magna aliqua. ";

        // Generate a fresh distinctive code per seed so the model can't lean
        // on training-data priors.
        var rng = new Random(seed);
        string[] colours =
        [
            "FUCHSIA", "TURQUOISE", "VERMILION", "CHARTREUSE", "INDIGO",
            "AMBER", "CRIMSON", "VIOLET", "TEAL", "MAGENTA",
        ];
        string colour = colours[rng.Next(colours.Length)];
        int code = 1000 + rng.Next(9000); // 4-digit
        string secret = $"{colour}-{code.ToString(CultureInfo.InvariantCulture)}";
        string needle = $" The secret code is {secret}. Remember it. ";

        // Build a haystack body of roughly targetLen tokens, then splice the
        // needle in at the desired offset. We pad to slightly above the target
        // before splicing so a final token count check stays in the ballpark.
        var sb = new StringBuilder();
        while (true)
        {
            sb.Append(filler);
            var count = tokenizer.Encode(sb.ToString()).Count;
            if (count >= targetLen) break;
            if (sb.Length > 1_000_000)
                throw new InvalidOperationException(
                    "Filler not producing enough tokens — tokenizer mismatch?");
        }

        // Splice the needle. Use a character offset proportional to position;
        // it doesn't have to be token-exact — only the rough region matters.
        double frac = position switch
        {
            "beginning" => 0.05,
            "middle"    => 0.50,
            "end"       => 0.90,
            _           => 0.50,
        };

        int charOffset = Math.Clamp((int)(sb.Length * frac), 0, sb.Length);
        // Snap to the nearest space so we don't split a word.
        while (charOffset > 0 && charOffset < sb.Length && sb[charOffset] != ' ')
            charOffset++;

        sb.Insert(charOffset, needle);

        // Append the question. The blank lines hint to the model that the
        // continuation should be the answer rather than more filler.
        sb.Append("\n\nQuestion: What is the secret code from earlier?\n\nAnswer:");
        return (sb.ToString(), secret);
    }

    /// <summary>
    /// Resolve the model paths to evaluate. If <paramref name="explicitPath"/>
    /// is provided, return it (unless it doesn't exist). Otherwise probe the
    /// standard SharpInference model directories for both SmolLM2 and Qwen3-8B.
    /// </summary>
    private static List<string> ResolveModels(string? explicitPath)
    {
        if (!string.IsNullOrEmpty(explicitPath))
        {
            if (!File.Exists(explicitPath))
            {
                Console.Error.WriteLine($"[eval] error: model not found: {explicitPath}");
                return [];
            }
            return [explicitPath];
        }

        var candidates = new[]
        {
            @"C:\p\sharpi\models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            @"C:\p\sharpi\models\Qwen3-8B-Q4_K_M.gguf",
            @"E:\models\SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            @"E:\models\Qwen3-8B-Q4_K_M.gguf",
        };

        // Dedup by base filename so we don't run the same model twice if both
        // mirrors are populated; prefer the first hit (i.e. C:\ over E:\).
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var result = new List<string>();
        foreach (var c in candidates)
        {
            var key = Path.GetFileName(c);
            if (File.Exists(c) && seen.Add(key)) result.Add(c);
        }
        return result;
    }

    private static void PrintMarkdownTable(List<ResultRow> rows)
    {
        Console.WriteLine();
        Console.WriteLine("| Model | Budget | Position | Score | Notes |");
        Console.WriteLine("|---|---:|---|---:|---|");
        foreach (var r in rows)
        {
            Console.WriteLine(
                $"| {r.Model} | {r.Budget} | {r.Position,-9} | {r.Score:F2} | {r.Note} |");
        }
    }

    private static void PrintPassCriterion(List<ResultRow> rows, Options opts)
    {
        // The acceptance criterion the harness exists to check: at the largest
        // budget, the needle should be recovered at every requested position.
        int passBudget = opts.Budgets.Max();
        var atTopBudget = rows.Where(r => r.Budget == passBudget).ToList();
        bool pass = atTopBudget.Count > 0 && atTopBudget.All(r => r.Recovered);

        Console.WriteLine();
        Console.WriteLine(
            $"Pass criterion (budget={passBudget} recovers needle at every position): {(pass ? "PASS" : "FAIL")}");
    }

    private static string Truncate(string s, int max)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ');
        return s.Length <= max ? s : s[..max] + "...";
    }

    private sealed record ResultRow(
        string Model,
        int Budget,
        string Position,
        double Score,
        bool Recovered,
        string Decoded,
        string Note);

    private sealed class Options
    {
        public string? ModelPath { get; init; }
        public required int[] Budgets { get; init; }
        public required string[] Positions { get; init; }
        public int Length { get; init; } = 8192;
        public int MaxDecode { get; init; } = 32;
        public int Seed { get; init; } = 42;

        public static Options? Parse(string[] args)
        {
            string? model = null;
            int[] budgets = [128, 256, 512, 1024, 2048];
            string[] positions = ["beginning", "middle"];
            int length = 8192;
            int maxDecode = 32;
            int seed = 42;

            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "--model":
                        model = RequireValue(args, ref i, "--model");
                        break;
                    case "--budgets":
                        budgets = ParseIntList(RequireValue(args, ref i, "--budgets"));
                        break;
                    case "--positions":
                        positions = ParsePositionList(RequireValue(args, ref i, "--positions"));
                        break;
                    case "--length":
                        length = ParseInt(RequireValue(args, ref i, "--length"));
                        break;
                    case "--max-decode":
                        maxDecode = ParseInt(RequireValue(args, ref i, "--max-decode"));
                        break;
                    case "--seed":
                        seed = ParseInt(RequireValue(args, ref i, "--seed"));
                        break;
                    case "-h":
                    case "--help":
                        PrintHelp();
                        return null;
                    default:
                        Console.Error.WriteLine($"[eval] error: unknown argument {a}");
                        PrintHelp();
                        return null;
                }
            }

            if (budgets.Length == 0)
            {
                Console.Error.WriteLine("[eval] error: --budgets must list at least one budget");
                return null;
            }
            if (positions.Length == 0)
            {
                Console.Error.WriteLine("[eval] error: --positions must list at least one position");
                return null;
            }

            return new Options
            {
                ModelPath = model,
                Budgets = budgets,
                Positions = positions,
                Length = length,
                MaxDecode = maxDecode,
                Seed = seed,
            };
        }

        private static string RequireValue(string[] args, ref int i, string name)
        {
            if (i + 1 >= args.Length)
                throw new ArgumentException($"{name} requires a value");
            return args[++i];
        }

        private static int ParseInt(string s) =>
            int.Parse(s, CultureInfo.InvariantCulture);

        private static int[] ParseIntList(string s) =>
            s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
             .Select(ParseInt).ToArray();

        private static string[] ParsePositionList(string s)
        {
            var parts = s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                         .Select(p => p.ToLowerInvariant()).ToArray();
            foreach (var p in parts)
            {
                if (p is not ("beginning" or "middle" or "end"))
                    throw new ArgumentException($"Unknown position '{p}' — expected beginning/middle/end");
            }
            return parts;
        }

        private static void PrintHelp()
        {
            Console.Error.WriteLine("""
SnapKV needle-in-haystack accuracy harness.

Usage:
  dotnet run --project benchmarks/SnapKvEval -- [options]

Options:
  --model <path>         Path to a Q4_K_M GGUF. If omitted, probes the standard
                         SharpInference model directories for SmolLM2-1.7B
                         and Qwen3-8B and runs both if present.
  --budgets a,b,c        Comma-separated SnapKV budgets to sweep. Default:
                         128,256,512,1024,2048.
  --positions p1,p2      Comma-separated needle positions: beginning, middle,
                         end. Default: beginning,middle.
  --length N             Approximate target prompt length in tokens.
                         Default: 8192.
  --max-decode N         Max tokens to decode looking for needle recovery.
                         Default: 32.
  --seed N               RNG seed for needle code generation. Default: 42.
""");
        }
    }
}
