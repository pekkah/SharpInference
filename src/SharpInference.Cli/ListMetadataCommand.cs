using System.ComponentModel;
using Spectre.Console;
using Spectre.Console.Cli;
using SharpInference.Core;

namespace SharpInference.Cli;

/// <summary>
/// Dumps all raw GGUF metadata key/value pairs from a model file.
/// Usage: sharpi-cli list-metadata -m model.gguf
/// </summary>
public sealed class ListMetadataCommand : Command<ListMetadataCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? ModelPath { get; init; }
    }

    protected override int Execute(CommandContext context, Settings settings, CancellationToken cancellation)
    {
        var modelPath = settings.ModelPath;
        if (modelPath is null)
        {
            foreach (var candidate in new[] { "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "model.gguf" })
                if (File.Exists(candidate)) { modelPath = candidate; break; }
        }
        if (modelPath is null || !File.Exists(modelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No model file found. Use [yellow]-m <path>[/]");
            return 1;
        }

        using var model = GgufModel.Open(modelPath);

        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(Path.GetFileName(modelPath))}[/]  " +
            $"GGUF v{model.Header.Version}  |  " +
            $"[cyan]{model.Header.TensorCount}[/] tensors  |  " +
            $"[cyan]{model.Metadata.Count}[/] metadata keys");
        AnsiConsole.WriteLine();

        // Keys whose values are large bulk arrays — show only the count
        var bulkArrayKeys = new HashSet<string>(StringComparer.Ordinal)
        {
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
        };

        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn(new TableColumn("[bold]Key[/]").NoWrap())
            .AddColumn(new TableColumn("[bold]Value[/]"));

        foreach (var kv in model.Metadata.OrderBy(x => x.Key))
        {
            string value;
            if (kv.Value is object[] arr)
            {
                if (bulkArrayKeys.Contains(kv.Key))
                    value = $"[dim](array: {arr.Length} items)[/]";
                else if (arr.Length <= 8)
                    value = Markup.Escape("[" + string.Join(", ", arr.Select(FormatScalar)) + "]");
                else
                {
                    var preview = string.Join(", ", arr.Take(4).Select(FormatScalar));
                    value = Markup.Escape($"[{preview}, … ({arr.Length} items)]");
                }
            }
            else
            {
                value = Markup.Escape(FormatScalar(kv.Value));
            }

            table.AddRow(Markup.Escape(kv.Key), value);
        }

        AnsiConsole.Write(table);
        return 0;
    }

    private static string FormatScalar(object v) => v switch
    {
        bool b    => b ? "true" : "false",
        float f   => f.ToString("G6"),
        double d  => d.ToString("G10"),
        string s  => s,
        byte n    => n.ToString(),
        sbyte n   => n.ToString(),
        ushort n  => n.ToString(),
        short n   => n.ToString(),
        uint n    => n.ToString(),
        int n     => n.ToString(),
        ulong n   => n.ToString(),
        long n    => n.ToString(),
        _         => "(unknown)",
    };
}
