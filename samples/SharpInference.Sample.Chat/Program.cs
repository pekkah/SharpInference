// SharpInference library sample: minimal streaming chat against a GGUF model.
//
//   dotnet run --project samples/SharpInference.Sample.Chat -c Release -- \
//       -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf
//
// Stdin is read one line at a time (or piped — EOF ends the session); generated
// tokens are streamed to stdout as soon as the engine produces them.

using SharpInference.Core;
using SharpInference.Cpu;
using SharpInference.Engine;

string? modelPath = null;
string? systemPrompt = null;
float temperature = 0.7f;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-m" or "--model" when i + 1 < args.Length:
            modelPath = args[++i]; break;
        case "-s" or "--system" when i + 1 < args.Length:
            systemPrompt = args[++i]; break;
        case "--temp" when i + 1 < args.Length:
            temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "-h" or "--help":
            Console.Error.WriteLine("usage: sharpi-sample-chat -m <model.gguf> [--system <prompt>] [--temp 0.7]");
            return 0;
    }
}

modelPath ??= Environment.GetEnvironmentVariable("SHARPI_MODEL");
if (modelPath is null || !File.Exists(modelPath))
{
    Console.Error.WriteLine("error: pass -m <model.gguf> or set SHARPI_MODEL.");
    return 1;
}

// Ctrl+C ⇒ cancel generation and unblock the read loop. We handle the first
// press cooperatively; a second press lets the runtime terminate the process.
using var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    if (cts.IsCancellationRequested) return;
    e.Cancel = true;
    cts.Cancel();
};

// Load the model and build the engine. The InferenceEngine takes ownership of
// the ForwardPass plus anything passed via `owned`, so a single Dispose chains
// down to release the mmap'd GGUF, the backend, and all native scratch.
Console.Error.Write($"loading {modelPath} ... ");
var model = GgufModel.Open(modelPath);
var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
var tokenizer = GgufTokenizer.FromGgufModel(model);
var backend = new CpuBackend();
var forward = new ForwardPass(model, backend, hp);
using var engine = new InferenceEngine(
    forward, tokenizer, modelId: Path.GetFileNameWithoutExtension(modelPath),
    owned: [backend, model]);
Console.Error.WriteLine($"{hp.NumLayers}L · {hp.EmbeddingDim}d · vocab {hp.VocabSize}");

var sampling = new SamplingParams
{
    Temperature = temperature,
    TopP = 0.95f,
    MinP = 0.05f,
    MaxNewTokens = 512,
    StopTokenIds = BuildStopTokens(tokenizer),
};

// Multi-turn chat. Keeping the full message list and re-rendering it each turn
// lets the engine's prefix-cache reuse KV for the shared prompt prefix —
// successive turns only prefill the new user message instead of starting over.
var history = new List<(string Role, string Content)>();
if (systemPrompt is not null)
    history.Add(("system", systemPrompt));

Console.Error.WriteLine("ready. type a message, Ctrl+D / Ctrl+Z to exit.\n");

while (!cts.IsCancellationRequested)
{
    Console.Write("> ");
    string? line;
    try { line = await Console.In.ReadLineAsync(cts.Token); }
    catch (OperationCanceledException) { break; }
    if (line is null) break;             // stdin closed (Ctrl+D / Ctrl+Z / piped EOF)
    if (line.Length == 0) continue;

    history.Add(("user", line));
    var prompt = RenderPrompt(tokenizer, history);

    // Stream output as it arrives. The engine yields decoded UTF-8 chunks
    // (already joined across multi-byte boundaries) — flush stdout per chunk
    // so output appears live rather than buffered into 4 KB blocks.
    var reply = new System.Text.StringBuilder();
    try
    {
        await foreach (var chunk in engine.GenerateAsync(prompt, sampling, cts.Token))
        {
            Console.Out.Write(chunk);
            Console.Out.Flush();
            reply.Append(chunk);
        }
    }
    catch (OperationCanceledException)
    {
        Console.Error.WriteLine("\n[cancelled]");
        break;
    }
    Console.WriteLine();

    history.Add(("assistant", reply.ToString()));
}

return 0;

// ─────────────────────────────────────────────────────────────────────────────

static string RenderPrompt(GgufTokenizer tok, List<(string Role, string Content)> history)
{
    var messages = new List<object?>(history.Count);
    foreach (var (role, content) in history)
        messages.Add(new Dictionary<string, object?> { ["role"] = role, ["content"] = content });

    // Prefer the model's own Jinja template (stored in GGUF metadata) — that's
    // what the model was trained against, so it gets special-token placement
    // exactly right across architectures.
    if (tok.ChatTemplate is { } template)
    {
        return template.Render(new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["add_generation_prompt"] = true,
            ["tools"] = null,
        });
    }

    // Fallback: bare ChatML. Works for Qwen / SmolLM and most modern chat models.
    var sb = new System.Text.StringBuilder();
    foreach (var (role, content) in history)
        sb.Append("<|im_start|>").Append(role).Append('\n').Append(content).Append("<|im_end|>\n");
    sb.Append("<|im_start|>assistant\n");
    return sb.ToString();
}

static int[] BuildStopTokens(GgufTokenizer tok)
{
    var stops = new HashSet<int> { tok.EosTokenId };
    foreach (var name in new[] { "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "<|endoftext|>" })
        if (tok.SpecialTokens.TryGetValue(name, out int id))
            stops.Add(id);
    return [.. stops];
}
