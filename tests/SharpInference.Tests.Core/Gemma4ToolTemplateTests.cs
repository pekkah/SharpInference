using SharpInference.Core;
using Xunit.Abstractions;

namespace SharpInference.Tests.Core;

/// <summary>
/// End-to-end check that the real Gemma 4 GGUF chat template renders tool definitions,
/// assistant tool calls, and tool results through <see cref="JinjaChatTemplate"/>. Exercises
/// the engine features the template depends on (dictsort/default/map filters, is boolean/
/// sequence tests, dict <c>.get()</c>). Model-gated — skips when the GGUF is absent.
/// </summary>
public sealed class Gemma4ToolTemplateTests(ITestOutputHelper output)
{
    private const string ModelPath = @"E:\models\gemma-4-E4B-it-Q8_0.gguf";

    private static JinjaChatTemplate? LoadTemplate()
    {
        using var m = GgufModel.Open(ModelPath);
        return GgufTokenizer.FromGgufModel(m).ChatTemplate;
    }

    private static List<object?> WeatherTool() =>
    [
        new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["type"] = "function",
            ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["name"] = "get_weather",
                ["description"] = "Get current weather for a city",
                ["parameters"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["city"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["type"] = "string", ["description"] = "City name",
                        },
                    },
                    ["required"] = new List<object?> { "city" },
                },
            },
        },
    ];

    [Fact]
    public void Template_RendersToolDefinition()
    {
        if (!File.Exists(ModelPath)) { output.WriteLine($"missing {ModelPath} — skip"); return; }
        var tmpl = LoadTemplate();
        Assert.NotNull(tmpl);

        var rendered = tmpl!.Render(new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = "user", ["content"] = "Weather in Paris?" },
            },
            ["tools"] = WeatherTool(),
            ["add_generation_prompt"] = true,
            ["enable_thinking"] = false,
            ["bos_token"] = "<bos>",
        });

        output.WriteLine(rendered);
        // Tool schema must reach the prompt with named, typed properties (regresses if
        // dictsort/default/map filters silently drop the schema).
        Assert.Contains("<|tool>declaration:get_weather", rendered);
        Assert.Contains("properties:{city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>}", rendered);
        Assert.Contains("required:[<|\"|>city<|\"|>]", rendered);
    }

    [Fact]
    public void Template_RendersToolCallAndResultRoundTrip()
    {
        if (!File.Exists(ModelPath)) { output.WriteLine($"missing {ModelPath} — skip"); return; }
        var tmpl = LoadTemplate();
        Assert.NotNull(tmpl);

        var messages = new List<object?>
        {
            new Dictionary<string, object?>(StringComparer.Ordinal) { ["role"] = "user", ["content"] = "Weather in Paris?" },
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["role"] = "assistant",
                ["content"] = "",
                ["tool_calls"] = new List<object?>
                {
                    new Dictionary<string, object?>(StringComparer.Ordinal)
                    {
                        ["id"] = "call_1", ["type"] = "function",
                        ["function"] = new Dictionary<string, object?>(StringComparer.Ordinal)
                        {
                            ["name"] = "get_weather",
                            ["arguments"] = new Dictionary<string, object?>(StringComparer.Ordinal) { ["city"] = "Paris" },
                        },
                    },
                },
            },
            new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["role"] = "tool", ["tool_call_id"] = "call_1", ["content"] = "sunny, 21C",
            },
        };

        var rendered = tmpl!.Render(new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["tools"] = WeatherTool(),
            ["add_generation_prompt"] = true,
            ["enable_thinking"] = false,
            ["bos_token"] = "<bos>",
        });

        output.WriteLine(rendered);
        // Assistant call uses Gemma's bespoke <|tool_call>call:NAME{...}<tool_call|> wire format.
        Assert.Contains("<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>", rendered);
        // Tool result resolves the function name via tool_call_id and wraps in <|tool_response>.
        Assert.Contains("<|tool_response>response:get_weather{value:<|\"|>sunny, 21C<|\"|>}<tool_response|>", rendered);
    }

    [Fact]
    public void Tokenizer_ExposesAlternateEogToken()
    {
        if (!File.Exists(ModelPath)) { output.WriteLine($"missing {ModelPath} — skip"); return; }
        using var m = GgufModel.Open(ModelPath);
        var tok = GgufTokenizer.FromGgufModel(m);

        // Gemma's configured EOS is <turn|> (id 106); <eos> (id 1) is a DISTINCT token — and
        // ships as token_type NORMAL, not control, so a special-token scan misses it. It must
        // still halt generation, else it decodes as literal "<eos>" text and the model runs on.
        Assert.Contains(tok.EosTokenId, tok.EogTokenIds);   // 106
        Assert.Contains(1, tok.EogTokenIds);                // <eos>, resolved via full-vocab lookup
        Assert.NotEqual(1, tok.EosTokenId);
    }

    [Fact]
    public void Tokenizer_DefinesReasoningChannelTokens()
    {
        if (!File.Exists(ModelPath)) { output.WriteLine($"missing {ModelPath} — skip"); return; }
        using var m = GgufModel.Open(ModelPath);
        var tok = GgufTokenizer.FromGgufModel(m);

        // The server loader routes <|channel> … <channel|> into the reasoning stream so the
        // markers don't leak into assistant content. Both must be positive special tokens.
        Assert.True(tok.SpecialTokens.TryGetValue("<|channel>", out int open) && open > 0);
        Assert.True(tok.SpecialTokens.TryGetValue("<channel|>", out int close) && close > 0);
    }
}
