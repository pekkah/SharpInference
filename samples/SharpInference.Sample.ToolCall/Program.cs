// SharpInference library sample: tool calling with a Qwen3 (or any chat model that
// supports tool_calls in its Jinja chat template).
//
// This sample demonstrates:
//   1. Defining tools as JSON-schema dicts and passing them to the Jinja template
//   2. Streaming the model's response and detecting <tool_call>...</tool_call> output
//   3. Executing the called tools locally and feeding results back as role="tool" messages
//   4. A second model turn that uses the tool results to produce a final answer
//
// Run (CPU):
//   dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
//       -m models/Qwen3-8B-Q4_K_M.gguf
//
// Run (CUDA, all layers on GPU):
//   dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
//       -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -g -1
//
// Run (CUDA hybrid — first 20 layers on GPU, rest on CPU):
//   dotnet run --project samples/SharpInference.Sample.ToolCall -c Release -- \
//       -m models/Qwen3-35B-A3B-Q4_K_M.gguf --backend cuda -g 20
//
// Flags:
//   -m / --model     <path>    Path to GGUF model (or set SHARPI_MODEL env var)
//   -p / --prompt    <text>    User question (default: built-in demo question)
//   --temp           <float>   Sampling temperature (default: 0.6)
//   --backend        cpu|cuda  Compute backend (default: cpu)
//   -g               <int>     GPU layer count: 0=CPU-only, -1=auto/all, N=N layers on GPU

using System.Text;
using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Engine;

// ─── Argument parsing ─────────────────────────────────────────────────────────

string? modelPath  = null;
string? question   = null;
float temperature  = 0.6f;
string backendStr  = "cpu";
int    nGpuLayers  = 0;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-m" or "--model"   when i + 1 < args.Length: modelPath   = args[++i]; break;
        case "-p" or "--prompt"  when i + 1 < args.Length: question    = args[++i]; break;
        case "--temp"            when i + 1 < args.Length: temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--backend"         when i + 1 < args.Length: backendStr  = args[++i].ToLowerInvariant(); break;
        case "-g"                when i + 1 < args.Length: nGpuLayers  = int.Parse(args[++i]); break;
        case "-h" or "--help":
            Console.Error.WriteLine(
                "usage: sharpi-sample-toolcall -m <model.gguf> [-p <question>] [--temp 0.6] [--backend cpu|cuda] [-g <layers>]");
            return 0;
    }
}

// If GPU layers were requested but --backend was not specified, default to cuda.
if (nGpuLayers != 0 && backendStr == "cpu")
    backendStr = "cuda";

modelPath ??= Environment.GetEnvironmentVariable("SHARPI_MODEL");
if (modelPath is null || !File.Exists(modelPath))
{
    Console.Error.WriteLine("error: pass -m <model.gguf> or set SHARPI_MODEL.");
    return 1;
}

question ??= "What is the weather like in Paris and Tokyo, and what is 42 times 17?";

using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    if (cts.IsCancellationRequested) return;
    e.Cancel = true;
    cts.Cancel();
};

// ─── Model loading ────────────────────────────────────────────────────────────

Console.Error.Write($"loading {modelPath} ... ");
var model     = GgufModel.Open(modelPath);
var hp        = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tokenizer = GgufTokenizer.FromGgufModel(model);
Console.Error.WriteLine($"{hp.NumLayers}L · {hp.EmbeddingDim}d · vocab {hp.VocabSize}" +
                        (hp.IsMoE ? $" · MoE {hp.NumExperts}x{hp.ExpertIntermediateDim}d" : ""));

// ─── Backend + forward pass selection ────────────────────────────────────────

IComputeBackend backend;
IForwardPass    forward;
List<IDisposable> owned = [model];

if (backendStr == "cuda")
{
    var cuda = CudaBackend.Create();
    owned.Add(cuda);

    if (hp.IsHybridSsm)
    {
        // Qwen3.6-35B-A3B (hybrid GDN+MoE): SSM/GDN layers run on CPU,
        // attention layers and MoE experts run on GPU via CudaExpertSlotManager.
        // Placement sets GpuLayers = NumLayers; actual routing is driven by
        // hp.LayerTypes internally.
        var hw        = HardwareProfile.Detect(cuda);
        var placement = new LayerPlacement(
            GpuLayers:          hp.NumLayers,
            CpuLayers:          0,
            GpuWeightBytes:     0,
            GpuKvBytes:         0,
            RecommendedCtxSize: Math.Min(hp.ContextLength, 4096));
        Console.Error.WriteLine($"backend: CUDA hybrid GDN ({cuda.Name}, {hw.Summary()})");
        var chgdn = new CudaHybridGdnForwardPass(model, cuda, hp, placement);
        var cpuBack = new CpuBackend();
        owned.Add(cpuBack);
        backend = cpuBack;
        forward = chgdn;
    }
    else
    {
        int gpuLayers;
        if (nGpuLayers == -1)
        {
            var hw        = HardwareProfile.Detect(cuda);
            var placement = TierPlanner.Plan(model, hp, hw);
            gpuLayers     = placement.GpuLayers;
            Console.Error.WriteLine($"backend: CUDA auto → {gpuLayers}/{hp.NumLayers} GPU layers ({hw.Summary()})");
        }
        else
        {
            gpuLayers = nGpuLayers;
            Console.Error.WriteLine($"backend: CUDA ({cuda.Name}, {gpuLayers} GPU layers requested)");
        }

        if (gpuLayers >= hp.NumLayers)
        {
            var cfwd = new CudaForwardPass(model, cuda, hp);
            backend  = cuda;
            forward  = cfwd;
        }
        else if (gpuLayers > 0)
        {
            var cpuBack   = new CpuBackend();
            var hw        = HardwareProfile.Detect(cuda);
            var placement = TierPlanner.Plan(model, hp, hw);
            placement     = placement with { GpuLayers = gpuLayers, CpuLayers = hp.NumLayers - gpuLayers };
            var chfwd     = new CudaHybridForwardPass(model, cuda, hp, placement);
            backend       = cpuBack;
            forward       = chfwd;
            owned.Add(cpuBack);
            Console.Error.WriteLine($"  → hybrid: {gpuLayers} GPU + {hp.NumLayers - gpuLayers} CPU layers");
        }
        else
        {
            Console.Error.WriteLine("  → 0 GPU layers, falling back to CPU");
            cuda.Dispose();
            owned.Remove(cuda);
            var cpuBack = new CpuBackend();
            backend     = cpuBack;
            forward     = new ForwardPass(model, cpuBack, hp);
            owned.Add(cpuBack);
        }
    }
}
else
{
    if (hp.IsHybridSsm)
    {
        Console.Error.WriteLine("backend: CPU (hybrid GDN+MoE)");
        var cpuBack = new CpuBackend();
        backend     = cpuBack;
        forward     = new HybridGdnForwardPass(model, cpuBack, hp);
        owned.Add(cpuBack);
    }
    else
    {
        Console.Error.WriteLine("backend: CPU");
        var cpuBack = new CpuBackend();
        backend     = cpuBack;
        forward     = new ForwardPass(model, cpuBack, hp);
        owned.Add(cpuBack);
    }
}

// Wire up think-token IDs so the engine can split/suppress reasoning blocks.
tokenizer.SpecialTokens.TryGetValue("<think>",  out int thinkTokenId);
tokenizer.SpecialTokens.TryGetValue("</think>", out int endThinkTokenId);

using var engine = new InferenceEngine(
    forward, tokenizer, modelId: Path.GetFileNameWithoutExtension(modelPath),
    thinkTokenId: thinkTokenId > 0 ? thinkTokenId : -1,
    endThinkTokenId: endThinkTokenId > 0 ? endThinkTokenId : -1,
    owned: [.. owned]);

var sampling = new SamplingParams
{
    Temperature   = temperature,
    TopP          = 0.95f,
    MinP          = 0.05f,
    MaxNewTokens  = 512,
    StopTokenIds  = BuildStopTokens(tokenizer),
};

// ─── Tool definitions ─────────────────────────────────────────────────────────
// Qwen3's chat template expects tools as a list of {type:"function", function:{name,description,parameters}}.
// The same format works for any model that follows the OpenAI tool-calling convention in its template.

var toolDefinitions = new List<Dictionary<string, object?>>
{
    MakeFunctionTool(
        name: "get_weather",
        description: "Get the current weather conditions for a given city.",
        parameters: new Dictionary<string, object?>
        {
            ["type"] = "object",
            ["properties"] = new Dictionary<string, object?>
            {
                ["city"] = new Dictionary<string, object?>
                {
                    ["type"]        = "string",
                    ["description"] = "The name of the city, e.g. 'Paris' or 'Tokyo'",
                },
            },
            ["required"] = new List<object?> { "city" },
        }),

    MakeFunctionTool(
        name: "calculate",
        description: "Evaluate a simple arithmetic expression and return the numeric result.",
        parameters: new Dictionary<string, object?>
        {
            ["type"] = "object",
            ["properties"] = new Dictionary<string, object?>
            {
                ["expression"] = new Dictionary<string, object?>
                {
                    ["type"]        = "string",
                    ["description"] = "Arithmetic expression to evaluate, e.g. '42 * 17'",
                },
            },
            ["required"] = new List<object?> { "expression" },
        }),
};

// ─── Turn 1: ask the question with tools available ────────────────────────────

Console.WriteLine($"\nUser: {question}\n");

var messages = new List<Dictionary<string, object?>>
{
    new() { ["role"] = "user", ["content"] = question },
};

string prompt = RenderPrompt(tokenizer, messages, toolDefinitions, enableThinking: false);

Console.Error.Write("Assistant (raw): ");
string rawReply = await StreamGenerate(engine, prompt, sampling, cts.Token);
Console.Error.WriteLine();

// ─── Parse <tool_call> blocks from the model's response ───────────────────────

var (plainText, toolCalls) = JinjaChatTemplate.ParseToolCalls(rawReply);

if (toolCalls.Count == 0)
{
    // Model answered directly without calling any tools.
    Console.WriteLine($"\nAssistant: {rawReply.Trim()}");
    return 0;
}

Console.WriteLine($"\n[model called {toolCalls.Count} tool(s)]");

// ─── Build assistant message that records the tool calls ──────────────────────
// Qwen3 template reads message.tool_calls as a list of {name, arguments} dicts.

var toolCallDicts = toolCalls
    .Select(tc => (Dictionary<string, object?>)new Dictionary<string, object?>
    {
        ["name"]      = tc.Name,
        ["arguments"] = tc.Arguments,
    })
    .ToList<object?>();

messages.Add(new Dictionary<string, object?>
{
    ["role"]       = "assistant",
    ["content"]    = plainText.Trim(),
    ["tool_calls"] = toolCallDicts,
});

// ─── Execute each tool and append role="tool" messages ───────────────────────

foreach (var tc in toolCalls)
{
    string result = ExecuteTool(tc.Name, tc.Arguments);
    Console.WriteLine($"  tool '{tc.Name}' → {result}");

    messages.Add(new Dictionary<string, object?>
    {
        ["role"]    = "tool",
        ["name"]    = tc.Name,
        ["content"] = result,
    });
}

// ─── Turn 2: final answer using tool results ──────────────────────────────────

Console.WriteLine();
prompt = RenderPrompt(tokenizer, messages, tools: null, enableThinking: false);

Console.Write("Assistant: ");
string finalReply = await StreamGenerate(engine, prompt, sampling, cts.Token, writeToStdout: true);
Console.WriteLine();

return 0;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Render a prompt string via the model's Jinja chat template.</summary>
static string RenderPrompt(
    GgufTokenizer tokenizer,
    List<Dictionary<string, object?>> messages,
    List<Dictionary<string, object?>>? tools,
    bool enableThinking)
{
    var msgList = messages.Cast<object?>().ToList();

    if (tokenizer.ChatTemplate is { } template)
    {
        return template.Render(new Dictionary<string, object?>
        {
            ["messages"]              = msgList,
            ["add_generation_prompt"] = true,
            ["tools"]                 = (object?)tools,
            ["enable_thinking"]       = (object?)enableThinking,
        });
    }

    // Fallback: bare ChatML (works for Qwen / SmolLM when no template is embedded).
    var sb = new StringBuilder();
    foreach (var msg in messages)
    {
        var role    = msg.TryGetValue("role",    out var r) ? r as string ?? "" : "";
        var content = msg.TryGetValue("content", out var c) ? c as string ?? "" : "";
        sb.Append("<|im_start|>").Append(role).Append('\n').Append(content).Append("<|im_end|>\n");
    }
    sb.Append("<|im_start|>assistant\n");
    return sb.ToString();
}

/// <summary>
/// Streams tokens from the engine, printing to stderr (raw model output) or stdout.
/// Returns the complete generated text.
/// </summary>
static async Task<string> StreamGenerate(
    InferenceEngine engine,
    string prompt,
    SamplingParams sp,
    CancellationToken ct,
    bool writeToStdout = false)
{
    var buf = new StringBuilder();
    var output = writeToStdout ? Console.Out : Console.Error;

    await foreach (var chunk in engine.GenerateAsync(prompt, sp, ct))
    {
        output.Write(chunk);
        output.Flush();
        buf.Append(chunk);
    }

    return buf.ToString();
}

/// <summary>Dispatches a parsed tool call to its local implementation.</summary>
static string ExecuteTool(string name, IReadOnlyDictionary<string, object?> arguments)
{
    return name switch
    {
        "get_weather" => GetWeather(arguments.TryGetValue("city", out var c) ? c as string ?? "" : ""),
        "calculate"   => Calculate(arguments.TryGetValue("expression", out var e) ? e as string ?? "" : ""),
        _             => $"unknown tool: {name}",
    };
}

/// <summary>Mock weather tool — returns canned data for demo purposes.</summary>
static string GetWeather(string city)
{
    return city.ToLowerInvariant() switch
    {
        "paris"   => "Partly cloudy, 18 °C. Light westerly breeze.",
        "tokyo"   => "Clear skies, 24 °C. Humid.",
        "london"  => "Overcast, 14 °C. Light rain expected.",
        "new york" or "newyork" => "Sunny, 22 °C.",
        "berlin"  => "Mostly sunny, 20 °C.",
        "sydney"  => "Warm and sunny, 28 °C.",
        _         => $"Partly cloudy, 21 °C (conditions for '{city}' are approximate).",
    };
}

/// <summary>
/// Evaluates a simple arithmetic expression using a minimal recursive-descent parser.
/// Supports +, -, *, / and parentheses on integer and decimal literals.
/// </summary>
static string Calculate(string expression)
{
    try
    {
        double result = new ArithmeticParser(expression.Trim()).Parse();
        // Return integer string when the result is whole, e.g. "714" not "714.0"
        return result == Math.Truncate(result) ? ((long)result).ToString() : result.ToString("G");
    }
    catch (Exception ex)
    {
        return $"error: {ex.Message}";
    }
}

/// <summary>
/// Builds a tool-definition dict in the format Qwen3's template expects:
/// <c>{ type: "function", function: { name, description, parameters } }</c>.
/// </summary>
static Dictionary<string, object?> MakeFunctionTool(
    string name,
    string description,
    Dictionary<string, object?> parameters) =>
    new()
    {
        ["type"]     = "function",
        ["function"] = new Dictionary<string, object?>
        {
            ["name"]        = name,
            ["description"] = description,
            ["parameters"]  = parameters,
        },
    };

static int[] BuildStopTokens(GgufTokenizer tok)
{
    var stops = new HashSet<int> { tok.EosTokenId };
    foreach (var name in new[] { "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "<|endoftext|>" })
        if (tok.SpecialTokens.TryGetValue(name, out int id))
            stops.Add(id);
    return [.. stops];
}

// ─── Types ────────────────────────────────────────────────────────────────────

/// <summary>
/// Minimal recursive-descent parser for arithmetic expressions.
/// Grammar:
///   expr   = term   (('+' | '-') term)*
///   term   = factor (('*' | '/') factor)*
///   factor = '(' expr ')' | number
/// </summary>
sealed class ArithmeticParser(string input)
{
    private int _pos;

    public double Parse()
    {
        double result = ParseExpr();
        SkipWhitespace();
        if (_pos < input.Length)
            throw new FormatException($"unexpected character '{input[_pos]}' at position {_pos}");
        return result;
    }

    private double ParseExpr()
    {
        double left = ParseTerm();
        while (true)
        {
            SkipWhitespace();
            if (_pos >= input.Length) break;
            char op = input[_pos];
            if (op is not '+' and not '-') break;
            _pos++;
            double right = ParseTerm();
            left = op == '+' ? left + right : left - right;
        }
        return left;
    }

    private double ParseTerm()
    {
        double left = ParseFactor();
        while (true)
        {
            SkipWhitespace();
            if (_pos >= input.Length) break;
            char op = input[_pos];
            if (op is not '*' and not '/') break;
            _pos++;
            double right = ParseFactor();
            left = op == '*' ? left * right : left / right;
        }
        return left;
    }

    private double ParseFactor()
    {
        SkipWhitespace();
        if (_pos >= input.Length)
            throw new FormatException("unexpected end of expression");

        if (input[_pos] == '(')
        {
            _pos++;
            double val = ParseExpr();
            SkipWhitespace();
            if (_pos >= input.Length || input[_pos] != ')')
                throw new FormatException("missing closing parenthesis");
            _pos++;
            return val;
        }

        // Unary minus
        if (input[_pos] == '-')
        {
            _pos++;
            return -ParseFactor();
        }

        return ParseNumber();
    }

    private double ParseNumber()
    {
        int start = _pos;
        while (_pos < input.Length && (char.IsDigit(input[_pos]) || input[_pos] == '.'))
            _pos++;
        if (_pos == start)
            throw new FormatException($"expected number at position {_pos}");
        return double.Parse(input.AsSpan(start, _pos - start), System.Globalization.CultureInfo.InvariantCulture);
    }

    private void SkipWhitespace()
    {
        while (_pos < input.Length && char.IsWhiteSpace(input[_pos]))
            _pos++;
    }
}
