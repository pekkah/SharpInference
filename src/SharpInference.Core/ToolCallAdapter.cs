using System.Text;
using System.Text.Json;
using SharpInference.Core.Grammar;

namespace SharpInference.Core;

/// <summary>
/// Translation layer between an LLM's free-text tool-call output and the
/// structured <see cref="ParsedToolCall"/> shape the API endpoints emit. One
/// implementation per model-family wire format (Qwen3, Qwen3-Coder, Llama-3,
/// DeepSeek-R1, ...). Resolved at engine-load time via
/// <see cref="ToolCallAdapterRegistry.Get(string)"/> keyed on the GGUF
/// <c>general.architecture</c> field.
///
/// <para>
/// The interface covers both batched (non-streaming) parsing and an
/// open/close-marker model used by the streaming endpoint state machine.
/// </para>
/// </summary>
/// <summary>
/// Result of <see cref="IToolCallAdapter.Parse"/>: the plain text (complete tool-call blocks
/// removed), the structured calls in order, and whether the output ended inside an
/// UNTERMINATED tool-call block — an open marker with no matching close, i.e. the model hit
/// max_tokens / EOS mid-call. A truncated trailing block is deliberately NOT returned as a
/// <see cref="Calls"/> entry (it would be a half-parsed, possibly dangerous call); its raw
/// partial is folded into <see cref="PlainText"/> so the non-streaming endpoints surface it as
/// text and report a length/max_tokens finish — exactly what the streaming state machine does.
/// The two-value <see cref="Deconstruct(out string, out IReadOnlyList{ParsedToolCall})"/> keeps
/// existing <c>(plain, calls) = Parse(...)</c> call sites working unchanged.
/// </summary>
public readonly record struct ToolCallParseResult(
    string PlainText, IReadOnlyList<ParsedToolCall> Calls, bool Truncated)
{
    /// <summary>Back-compat deconstruction that ignores <see cref="Truncated"/>.</summary>
    public void Deconstruct(out string plainText, out IReadOnlyList<ParsedToolCall> calls)
    {
        plainText = PlainText;
        calls = Calls;
    }
}

public interface IToolCallAdapter
{
    /// <summary>Architecture key this adapter handles (e.g. <c>"qwen3moe"</c>).</summary>
    string Architecture { get; }

    /// <summary>
    /// Extracts structured tool calls from a complete model output. Returns the plain text
    /// (with complete tool-call blocks stripped), the parsed calls in order, and whether the
    /// output ended mid-call (see <see cref="ToolCallParseResult"/>). Used by the non-streaming
    /// endpoint code paths.
    /// </summary>
    ToolCallParseResult Parse(string rawOutput);

    /// <summary>
    /// Length of the longest open marker this adapter recognises. The streaming
    /// endpoint keeps up to <c>MaxOpenTagLength - 1</c> trailing bytes buffered so
    /// a marker that arrives split across chunks is still matched.
    /// </summary>
    int MaxOpenTagLength { get; }

    /// <summary>
    /// Scans <paramref name="buffer"/> at or after <paramref name="startSearch"/>
    /// for a tool-call open marker. Returns the index of the marker's first byte
    /// (the streamer flushes everything before it as plain text) and writes the
    /// start of the buffered block region to <paramref name="contentStart"/>.
    /// For most adapters this is past the marker; adapters whose call name lives
    /// inside the marker (e.g. Qwen3-Coder's <c>&lt;function=NAME&gt;</c>) return
    /// the same index for both, so the marker stays in the block. Returns -1 when
    /// no marker is present in the buffer.
    /// </summary>
    int FindOpenMarker(string buffer, int startSearch, out int contentStart);

    /// <summary>
    /// Scans <paramref name="buffer"/> at or after <paramref name="startSearch"/>
    /// for the matching close marker. The return value is the EXCLUSIVE end index
    /// of the buffered block region (i.e. <c>buf[contentStart..returned]</c> is
    /// what gets handed to <see cref="ParseBlock"/>); for most adapters that's
    /// the marker's first byte, but adapters whose parser needs the close marker
    /// itself (e.g. Qwen3-Coder) return one-past-end. <paramref name="afterClose"/>
    /// receives the index past the close marker (where streaming resumes scanning).
    /// Returns -1 when the close marker is not yet present.
    /// </summary>
    int FindCloseMarker(string buffer, int startSearch, out int afterClose);

    /// <summary>
    /// Parses one or more tool calls out of a buffered region. The slice handed in
    /// matches the contract documented on <see cref="FindOpenMarker"/> and
    /// <see cref="FindCloseMarker"/>. Multi-call wire formats (e.g. DeepSeek's
    /// <c>tool_calls_begin..end</c> wrapper) may return more than one call.
    /// </summary>
    IReadOnlyList<ParsedToolCall> ParseBlock(string block);

    /// <summary>
    /// Builds the role/content dictionary for replaying a prior tool result on
    /// this model's chat template. Default Qwen/OpenAI shape is
    /// <c>{role:"tool", content:string}</c>; some families wrap differently.
    /// </summary>
    Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent);

    /// <summary>
    /// Special-token strings that mark the END of this model's tool-call section — i.e. the model
    /// has finished emitting tool calls and is opening the next (tool-result) turn. A host driving
    /// an agentic loop can resolve these against the tokenizer's special tokens and add them to its
    /// stop set so generation halts the instant the tool calls are complete, instead of running on
    /// and hallucinating a trailing turn. Returns the model-family token so the host doesn't have to
    /// hardcode it. Empty for families whose end-of-turn / EOG token already covers this boundary;
    /// Gemma 4 returns <c>&lt;|tool_response&gt;</c> (this QAT line opens a response turn rather than
    /// emitting <c>&lt;end_of_turn&gt;</c>). See issue #304.
    /// </summary>
    IReadOnlyList<string> ToolBoundaryStopMarkers => [];

    /// <summary>
    /// Builds an optional grammar/FSM constraint (issue #374) that forces this model's tool-call
    /// <em>arguments</em> to satisfy the supplied schemas, expressed in this adapter's native call
    /// syntax. Returns <c>null</c> when the family has no constrained-decoding support or no
    /// constrainable tool was supplied — the engine then samples unconstrained (byte-identical to
    /// the default path). Each adapter owns its wire format (Gemma's bespoke <c>&lt;|"|&gt;</c> form
    /// vs Qwen/Llama JSON), so the constraint targets the actual emitted tokens, not generic JSON.
    /// </summary>
    /// <param name="tools">The active tool definitions (name + parsed argument schema).</param>
    /// <param name="vocab">Vocabulary view used to resolve structural tokens and match candidates.</param>
    ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab) => null;
}

/// <summary>
/// Process-wide registry of <see cref="IToolCallAdapter"/> instances keyed by GGUF
/// architecture. Ships defaults for the families enumerated in issue #96; callers
/// can <see cref="Register"/> additional adapters at startup. Lookups for unknown
/// architectures fall through to the Qwen3-style wrapper adapter (no regression
/// for any model whose template already emits <c>&lt;tool_call&gt;</c> blocks).
/// </summary>
public static class ToolCallAdapterRegistry
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, IToolCallAdapter> _adapters =
        new(StringComparer.Ordinal)
        {
            // Wrapper-style: Qwen2, Qwen3, Qwen3.x, Qwen3-MoE variants — all use
            // the <tool_call>...</tool_call> envelope; supports both Qwen3.6 XML
            // (<function=...></function>) and standard JSON ({"name":"...", ...})
            // payloads inside.
            ["qwen2"]      = new QwenToolCallAdapter("qwen2"),
            ["qwen3"]      = new QwenToolCallAdapter("qwen3"),
            ["qwen3moe"]   = new QwenToolCallAdapter("qwen3moe"),
            ["qwen35moe"]  = new QwenToolCallAdapter("qwen35moe"),

            // Qwen3-Coder: bare <function=name>...</function> with no <tool_call>
            // wrapper. Closes #95.
            ["qwen3coder"] = new QwenCoderToolCallAdapter(),

            // Llama-3.x tool-calling format: <|python_tag|>{json}<|eom_id|>
            ["llama"]      = new LlamaToolCallAdapter(),
            ["llama4"]     = new LlamaToolCallAdapter(),

            // DeepSeek-R1 wrapped multi-call envelope.
            ["deepseek2"]  = new DeepSeekToolCallAdapter(),

            // Gemma 4: <|tool_call>call:NAME{k:v,...}<tool_call|> with a bespoke
            // non-JSON argument syntax (<|"|>-quoted strings, bare keys).
            ["gemma4"]     = new Gemma4ToolCallAdapter(),
        };

    /// <summary>
    /// Fallback used when no adapter is registered for the requested architecture.
    /// The Qwen3 wrapper adapter is the broadest compatible default — every model
    /// whose chat template renders OpenAI-style tools tends to emit
    /// <c>&lt;tool_call&gt;</c> blocks under it.
    /// </summary>
    public static IToolCallAdapter DefaultAdapter { get; } = new QwenToolCallAdapter("default");

    /// <summary>Returns the adapter registered for <paramref name="architecture"/>, or the default.</summary>
    public static IToolCallAdapter Get(string? architecture)
    {
        if (architecture is { Length: > 0 } && _adapters.TryGetValue(architecture, out var a))
            return a;
        return DefaultAdapter;
    }

    /// <summary>Registers (or replaces) the adapter for <paramref name="architecture"/>.</summary>
    public static void Register(string architecture, IToolCallAdapter adapter)
    {
        ArgumentException.ThrowIfNullOrEmpty(architecture);
        ArgumentNullException.ThrowIfNull(adapter);
        _adapters[architecture] = adapter;
    }
}

// ── Adapter base helpers ──────────────────────────────────────────────────────

internal static class ToolCallParseHelpers
{
    public static Dictionary<string, object?> DefaultRenderToolResult(string _, string resultContent) =>
        new(StringComparer.Ordinal) { ["role"] = "tool", ["content"] = resultContent };

    /// <summary>
    /// Parses a single <c>&lt;function=name&gt;&lt;parameter=k&gt;v&lt;/parameter&gt;...&lt;/function&gt;</c>
    /// block. Tolerant of a missing trailing <c>&lt;/function&gt;</c> (Qwen3-Coder occasionally
    /// stops generating before emitting it).
    /// </summary>
    public static ParsedToolCall? ParseXmlFunctionBlock(string block)
    {
        const string funcTag    = "<function=";
        const string funcClose  = "</function>";
        const string paramTag   = "<parameter=";
        const string paramClose = "</parameter>";

        int funcStart = block.IndexOf(funcTag, StringComparison.Ordinal);
        if (funcStart < 0) return null;

        int nameEnd = block.IndexOf('>', funcStart);
        if (nameEnd < 0) return null;
        string funcName = block[(funcStart + funcTag.Length)..nameEnd].Trim();
        if (funcName.Length == 0) return null;

        int funcBodyEnd = block.IndexOf(funcClose, nameEnd, StringComparison.Ordinal);
        string funcBody = funcBodyEnd >= 0
            ? block[(nameEnd + 1)..funcBodyEnd]
            : block[(nameEnd + 1)..];

        var args = new Dictionary<string, object?>(StringComparer.Ordinal);
        int p = 0;
        while (p < funcBody.Length)
        {
            int paramStart = funcBody.IndexOf(paramTag, p, StringComparison.Ordinal);
            if (paramStart < 0) break;

            int paramNameEnd = funcBody.IndexOf('>', paramStart);
            if (paramNameEnd < 0) break;
            string paramName = funcBody[(paramStart + paramTag.Length)..paramNameEnd].Trim();

            int valueStart = paramNameEnd + 1;
            int paramEnd   = funcBody.IndexOf(paramClose, valueStart, StringComparison.Ordinal);
            if (paramEnd < 0) break;

            // Skip nameless <parameter=> tags rather than inserting an empty-string key.
            if (paramName.Length > 0)
                args[paramName] = funcBody[valueStart..paramEnd].Trim();
            p = paramEnd + paramClose.Length;
        }

        return new ParsedToolCall(funcName, args);
    }

    /// <summary>
    /// Parses a <c>{"name":"...","arguments":{...}}</c> JSON object (Qwen3 default
    /// payload and Llama-3 emit shape). <c>arguments</c> may itself be a JSON
    /// string (some templates double-encode); both shapes are accepted. When
    /// <paramref name="argumentsKey"/> is absent the alternate key (<c>arguments</c>
    /// vs <c>parameters</c>) is tried as a fallback — Llama-3 fine-tunes drift
    /// between both spellings.
    /// </summary>
    public static ParsedToolCall? ParseJsonCallBlock(string block, string argumentsKey = "arguments")
    {
        try
        {
            using var doc = JsonDocument.Parse(block);
            var root  = doc.RootElement;
            string name = root.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";
            if (name.Length == 0) return null;

            var args = new Dictionary<string, object?>(StringComparer.Ordinal);
            string altKey = argumentsKey == "arguments" ? "parameters" : "arguments";
            if (root.TryGetProperty(argumentsKey, out var argsElem)
                || root.TryGetProperty(altKey, out argsElem))
            {
                object? parsed;
                if (argsElem.ValueKind == JsonValueKind.String)
                {
                    using var innerDoc = JsonDocument.Parse(argsElem.GetString() ?? "{}");
                    parsed = JsonElementToObject(innerDoc.RootElement);
                }
                else
                {
                    parsed = JsonElementToObject(argsElem);
                }
                if (parsed is Dictionary<string, object?> d) args = d;
            }

            return new ParsedToolCall(name, args);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static object? JsonElementToObject(JsonElement el) => el.ValueKind switch
    {
        JsonValueKind.Object => el.EnumerateObject()
                                  .ToDictionary(p => p.Name,
                                                p => JsonElementToObject(p.Value),
                                                StringComparer.Ordinal),
        JsonValueKind.Array  => el.EnumerateArray()
                                  .Select(JsonElementToObject)
                                  .ToList<object?>(),
        JsonValueKind.String => el.GetString(),
        JsonValueKind.Number => el.TryGetInt64(out long l) ? l
                              : el.TryGetDouble(out double d) ? d
                              : (object?)el.GetRawText(),
        JsonValueKind.True   => true,
        JsonValueKind.False  => false,
        _                    => null,
    };
}

// ── Concrete adapters ─────────────────────────────────────────────────────────

/// <summary>
/// Qwen2/3 family: model output wraps each call in
/// <c>&lt;tool_call&gt;...&lt;/tool_call&gt;</c>. The payload inside is either
/// standard JSON (<c>{"name":"...","arguments":{...}}</c>) or — on the
/// Qwen3.6 line — the XML shape <c>&lt;function=name&gt;&lt;parameter=...&gt;...&lt;/parameter&gt;&lt;/function&gt;</c>.
/// </summary>
public sealed class QwenToolCallAdapter(string architecture) : IToolCallAdapter
{
    public const string OpenMarker  = "<tool_call>";
    public const string CloseMarker = "</tool_call>";

    public string Architecture { get; } = architecture;
    public int MaxOpenTagLength => OpenMarker.Length;

    /// <inheritdoc/>
    /// <remarks>Constrains the JSON argument object of Qwen's
    /// <c>&lt;tool_call&gt;{"name":"NAME","arguments":{...}}&lt;/tool_call&gt;</c> wire format
    /// (issue #376). Inert (returns null) when no supplied tool is constrainable, when
    /// <c>&lt;tool_call&gt;</c> isn't a vocabulary token, or when the model emits the Qwen3.6 XML
    /// shape instead of JSON (the constraint simply never engages).</remarks>
    public ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(vocab);
        var c = new JsonToolArgumentConstraint(
            vocab, tools, OpenMarker, JsonToolEnvelope.NameValueObject, argsKeys: ["arguments", "parameters"]);
        return c.HasConstrainableTools ? c : null;
    }

    public ToolCallParseResult Parse(string rawOutput)
    {
        var calls = new List<ParsedToolCall>();
        var plain = new StringBuilder();
        int pos = 0;

        while (pos < rawOutput.Length)
        {
            int start = rawOutput.IndexOf(OpenMarker, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                plain.Append(rawOutput, pos, rawOutput.Length - pos);
                break;
            }

            plain.Append(rawOutput, pos, start - pos);
            int contentStart = start + OpenMarker.Length;
            int end = rawOutput.IndexOf(CloseMarker, contentStart, StringComparison.Ordinal);
            if (end < 0)
            {
                // Unterminated call: surface the partial as text and flag truncation rather
                // than emitting a half-parsed call.
                plain.Append(rawOutput, contentStart, rawOutput.Length - contentStart);
                return new ToolCallParseResult(plain.ToString(), calls, Truncated: true);
            }

            string block = rawOutput[contentStart..end].Trim();
            pos = end + CloseMarker.Length;
            foreach (var c in ParseBlock(block)) calls.Add(c);
        }

        return new ToolCallParseResult(plain.ToString(), calls, Truncated: false);
    }

    public int FindOpenMarker(string buffer, int startSearch, out int contentStart)
    {
        int idx = buffer.IndexOf(OpenMarker, startSearch, StringComparison.Ordinal);
        contentStart = idx < 0 ? -1 : idx + OpenMarker.Length;
        return idx;
    }

    public int FindCloseMarker(string buffer, int startSearch, out int afterClose)
    {
        int idx = buffer.IndexOf(CloseMarker, startSearch, StringComparison.Ordinal);
        afterClose = idx < 0 ? -1 : idx + CloseMarker.Length;
        return idx;
    }

    public IReadOnlyList<ParsedToolCall> ParseBlock(string block)
    {
        block = block.Trim();
        ParsedToolCall? call = block.StartsWith("<function=", StringComparison.Ordinal)
            ? ToolCallParseHelpers.ParseXmlFunctionBlock(block)
            : ToolCallParseHelpers.ParseJsonCallBlock(block);
        return call.HasValue ? [call.Value] : [];
    }

    public Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent) =>
        ToolCallParseHelpers.DefaultRenderToolResult(toolUseId, resultContent);
}

/// <summary>
/// Qwen3-Coder: emits bare <c>&lt;function=name&gt;...&lt;/function&gt;</c> with
/// no surrounding <c>&lt;tool_call&gt;</c> envelope. Closes #95.
/// </summary>
public sealed class QwenCoderToolCallAdapter : IToolCallAdapter
{
    public const string OpenMarker  = "<function=";
    public const string CloseMarker = "</function>";
    /// <summary>Qwen's tool-call envelope tokens. Qwen3-Coder wraps each XML call in
    /// <c>&lt;tool_call&gt;…&lt;/tool_call&gt;</c>; these are single special tokens used by the
    /// argument-grammar constraint as a leak-proof arming gate (the inner <c>&lt;function=&gt;</c> tags
    /// are ordinary text). Mirrors <see cref="QwenToolCallAdapter.OpenMarker"/>.</summary>
    public const string ArmMarker      = "<tool_call>";
    public const string ArmCloseMarker = "</tool_call>";

    public string Architecture => "qwen3coder";
    public int MaxOpenTagLength => OpenMarker.Length;

    /// <inheritdoc/>
    /// <remarks>Constrains the XML argument body of Qwen3-Coder's
    /// <c>&lt;tool_call&gt;&lt;function=NAME&gt;&lt;parameter=KEY&gt;VALUE&lt;/parameter&gt;…&lt;/function&gt;&lt;/tool_call&gt;</c>
    /// wire format (issue #383) — the XML sibling of the JSON/Gemma constraints. Inert (returns null)
    /// when no supplied tool is constrainable or <c>&lt;tool_call&gt;</c> isn't a vocabulary token.</remarks>
    public ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(vocab);
        var c = new QwenCoderToolArgumentConstraint(vocab, tools);
        return c.HasConstrainableTools ? c : null;
    }

    public ToolCallParseResult Parse(string rawOutput)
    {
        var calls = new List<ParsedToolCall>();
        var plain = new StringBuilder();
        int pos = 0;

        while (pos < rawOutput.Length)
        {
            int start = rawOutput.IndexOf(OpenMarker, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                plain.Append(rawOutput, pos, rawOutput.Length - pos);
                break;
            }

            plain.Append(rawOutput, pos, start - pos);
            int end = rawOutput.IndexOf(CloseMarker, start, StringComparison.Ordinal);
            if (end < 0)
            {
                // Unterminated <function=...> (no </function>): surface the partial (marker
                // included, matching the streaming buffer) and flag truncation.
                plain.Append(rawOutput, start, rawOutput.Length - start);
                return new ToolCallParseResult(plain.ToString(), calls, Truncated: true);
            }

            // Block must include the <function=> prefix so the parser can read the name.
            string block = rawOutput[start..(end + CloseMarker.Length)];
            pos = end + CloseMarker.Length;

            var call = ToolCallParseHelpers.ParseXmlFunctionBlock(block);
            if (call.HasValue) calls.Add(call.Value);
        }

        return new ToolCallParseResult(plain.ToString(), calls, Truncated: false);
    }

    public int FindOpenMarker(string buffer, int startSearch, out int contentStart)
    {
        int idx = buffer.IndexOf(OpenMarker, startSearch, StringComparison.Ordinal);
        // contentStart points AT the open marker (not past it) — Qwen3-Coder's call name
        // lives inside the open marker (<function=NAME>), so the block we hand to
        // ParseBlock must include it.
        contentStart = idx;
        return idx;
    }

    public int FindCloseMarker(string buffer, int startSearch, out int afterClose)
    {
        int idx = buffer.IndexOf(CloseMarker, startSearch, StringComparison.Ordinal);
        if (idx < 0) { afterClose = -1; return -1; }
        // Include the close marker in the buffered region: the block we hand to
        // ParseBlock must include </function> so ParseXmlFunctionBlock's name-end
        // scan terminates correctly.
        afterClose = idx + CloseMarker.Length;
        return afterClose;
    }

    public IReadOnlyList<ParsedToolCall> ParseBlock(string block)
    {
        var call = ToolCallParseHelpers.ParseXmlFunctionBlock(block.Trim());
        return call.HasValue ? [call.Value] : [];
    }

    public Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent) =>
        ToolCallParseHelpers.DefaultRenderToolResult(toolUseId, resultContent);
}

/// <summary>
/// Llama-3.x tool-calling: <c>&lt;|python_tag|&gt;{"name":"...","parameters":{...}}&lt;|eom_id|&gt;</c>.
/// The close marker may also be <c>&lt;|eot_id|&gt;</c> on shorter outputs — both are recognised.
/// </summary>
public sealed class LlamaToolCallAdapter : IToolCallAdapter
{
    public const string OpenMarker  = "<|python_tag|>";
    public const string EomMarker   = "<|eom_id|>";
    public const string EotMarker   = "<|eot_id|>";

    public string Architecture => "llama";
    public int MaxOpenTagLength => OpenMarker.Length;

    /// <inheritdoc/>
    /// <remarks>Constrains the JSON argument object of Llama-3's
    /// <c>&lt;|python_tag|&gt;{"name":"NAME","parameters":{...}}</c> wire format (issue #376). Inert
    /// (null) when no tool is constrainable or <c>&lt;|python_tag|&gt;</c> isn't a vocabulary token.</remarks>
    public ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(vocab);
        var c = new JsonToolArgumentConstraint(
            vocab, tools, OpenMarker, JsonToolEnvelope.NameValueObject, argsKeys: ["parameters", "arguments"]);
        return c.HasConstrainableTools ? c : null;
    }

    public ToolCallParseResult Parse(string rawOutput)
    {
        var calls = new List<ParsedToolCall>();
        var plain = new StringBuilder();
        int pos = 0;

        while (pos < rawOutput.Length)
        {
            int start = rawOutput.IndexOf(OpenMarker, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                plain.Append(rawOutput, pos, rawOutput.Length - pos);
                break;
            }

            plain.Append(rawOutput, pos, start - pos);
            int contentStart = start + OpenMarker.Length;
            int end = FindEarliest(rawOutput, contentStart, EomMarker, EotMarker, out int afterClose);
            if (end < 0)
            {
                // No <|eom_id|>/<|eot_id|> close: surface the partial and flag truncation.
                plain.Append(rawOutput, contentStart, rawOutput.Length - contentStart);
                return new ToolCallParseResult(plain.ToString(), calls, Truncated: true);
            }

            string block = rawOutput[contentStart..end];
            pos = afterClose;
            foreach (var c in ParseBlock(block)) calls.Add(c);
        }

        return new ToolCallParseResult(plain.ToString(), calls, Truncated: false);
    }

    public int FindOpenMarker(string buffer, int startSearch, out int contentStart)
    {
        int idx = buffer.IndexOf(OpenMarker, startSearch, StringComparison.Ordinal);
        contentStart = idx < 0 ? -1 : idx + OpenMarker.Length;
        return idx;
    }

    public int FindCloseMarker(string buffer, int startSearch, out int afterClose)
    {
        int idx = FindEarliest(buffer, startSearch, EomMarker, EotMarker, out afterClose);
        return idx;
    }

    public IReadOnlyList<ParsedToolCall> ParseBlock(string block)
    {
        // Llama-3 emits {"name":"...", "parameters":{...}}; some fine-tunes still use
        // "arguments". ParseJsonCallBlock handles both keys.
        var call = ToolCallParseHelpers.ParseJsonCallBlock(block.Trim(), argumentsKey: "parameters");
        return call.HasValue ? [call.Value] : [];
    }

    public Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent) =>
        new(StringComparer.Ordinal)
        {
            ["role"]    = "ipython",   // Llama-3 echoes tool results under role=ipython.
            ["content"] = resultContent,
        };

    private static int FindEarliest(string buffer, int startSearch, string a, string b, out int afterClose)
    {
        int ia = buffer.IndexOf(a, startSearch, StringComparison.Ordinal);
        int ib = buffer.IndexOf(b, startSearch, StringComparison.Ordinal);
        int idx = (ia, ib) switch
        {
            (< 0, < 0) => -1,
            (< 0, _)   => ib,
            (_, < 0)   => ia,
            _          => Math.Min(ia, ib),
        };
        if (idx < 0) { afterClose = -1; return -1; }
        afterClose = idx + (idx == ia ? a.Length : b.Length);
        return idx;
    }
}

/// <summary>
/// DeepSeek-R1 family: <c>&lt;|tool_calls_begin|&gt;...&lt;|tool_calls_end|&gt;</c>
/// wraps one or more <c>&lt;|tool_call_begin|&gt;name&lt;|tool_sep|&gt;{json}&lt;|tool_call_end|&gt;</c>
/// inner blocks.
/// </summary>
public sealed class DeepSeekToolCallAdapter : IToolCallAdapter
{
    public const string OuterOpen  = "<|tool_calls_begin|>";
    public const string OuterClose = "<|tool_calls_end|>";
    public const string InnerOpen  = "<|tool_call_begin|>";
    public const string InnerClose = "<|tool_call_end|>";
    public const string Separator  = "<|tool_sep|>";

    public string Architecture => "deepseek2";
    public int MaxOpenTagLength => OuterOpen.Length;

    /// <inheritdoc/>
    /// <remarks>Constrains the JSON argument object of DeepSeek's
    /// <c>&lt;|tool_call_begin|&gt;NAME&lt;|tool_sep|&gt;{...}&lt;|tool_call_end|&gt;</c> wire format
    /// (issue #376): the name is bare text up to <c>&lt;|tool_sep|&gt;</c>, then the argument object
    /// is emitted directly. Inert (null) when no tool is constrainable or those structural tokens are
    /// absent from the vocabulary.</remarks>
    public ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(vocab);
        var c = new JsonToolArgumentConstraint(
            vocab, tools, InnerOpen, JsonToolEnvelope.NameThenSeparator, separatorMarker: Separator);
        return c.HasConstrainableTools ? c : null;
    }

    public ToolCallParseResult Parse(string rawOutput)
    {
        var calls = new List<ParsedToolCall>();
        var plain = new StringBuilder();
        int pos = 0;

        while (pos < rawOutput.Length)
        {
            int start = rawOutput.IndexOf(OuterOpen, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                plain.Append(rawOutput, pos, rawOutput.Length - pos);
                break;
            }

            plain.Append(rawOutput, pos, start - pos);
            int contentStart = start + OuterOpen.Length;
            int end = rawOutput.IndexOf(OuterClose, contentStart, StringComparison.Ordinal);
            if (end < 0)
            {
                // No <|tool_calls_end|>: surface the partial and flag truncation.
                plain.Append(rawOutput, contentStart, rawOutput.Length - contentStart);
                return new ToolCallParseResult(plain.ToString(), calls, Truncated: true);
            }

            string block = rawOutput[contentStart..end];
            pos = end + OuterClose.Length;
            foreach (var c in ParseBlock(block)) calls.Add(c);
        }

        return new ToolCallParseResult(plain.ToString(), calls, Truncated: false);
    }

    public int FindOpenMarker(string buffer, int startSearch, out int contentStart)
    {
        int idx = buffer.IndexOf(OuterOpen, startSearch, StringComparison.Ordinal);
        contentStart = idx < 0 ? -1 : idx + OuterOpen.Length;
        return idx;
    }

    public int FindCloseMarker(string buffer, int startSearch, out int afterClose)
    {
        int idx = buffer.IndexOf(OuterClose, startSearch, StringComparison.Ordinal);
        afterClose = idx < 0 ? -1 : idx + OuterClose.Length;
        return idx;
    }

    public IReadOnlyList<ParsedToolCall> ParseBlock(string block)
    {
        var results = new List<ParsedToolCall>();
        int p = 0;
        while (p < block.Length)
        {
            int innerStart = block.IndexOf(InnerOpen, p, StringComparison.Ordinal);
            if (innerStart < 0) break;
            int contentStart = innerStart + InnerOpen.Length;
            int innerEnd = block.IndexOf(InnerClose, contentStart, StringComparison.Ordinal);
            if (innerEnd < 0) break;

            string inner = block[contentStart..innerEnd];
            p = innerEnd + InnerClose.Length;

            int sep = inner.IndexOf(Separator, StringComparison.Ordinal);
            if (sep < 0) continue;

            string name = inner[..sep].Trim();
            string argsJson = inner[(sep + Separator.Length)..].Trim();
            if (name.Length == 0) continue;

            var argDict = new Dictionary<string, object?>(StringComparer.Ordinal);
            try
            {
                using var doc = JsonDocument.Parse(argsJson);
                if (ToolCallParseHelpers.JsonElementToObject(doc.RootElement) is Dictionary<string, object?> d)
                    argDict = d;
            }
            catch (JsonException) { /* keep empty arg dict for malformed JSON */ }

            results.Add(new ParsedToolCall(name, argDict));
        }
        return results;
    }

    public Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent) =>
        ToolCallParseHelpers.DefaultRenderToolResult(toolUseId, resultContent);
}

/// <summary>
/// Gemma 4 family. The model wraps each call in
/// <c>&lt;|tool_call&gt;call:NAME{...}&lt;tool_call|&gt;</c> (note the asymmetric
/// pipe placement — open is <c>&lt;|tool_call&gt;</c>, close is <c>&lt;tool_call|&gt;</c>).
///
/// <para>
/// Arguments are NOT JSON. They use Gemma's bespoke key:value syntax produced by the
/// GGUF chat template's <c>format_argument</c> macro: strings are wrapped in the
/// <c>&lt;|"|&gt;</c> quote token, keys are bare (unquoted), booleans render as
/// <c>true</c>/<c>false</c>, <c>null</c> is bare, numbers are bare, nested objects use
/// <c>{k:v,...}</c> and arrays use <c>[v,...]</c>. Example:
/// <c>&lt;|tool_call&gt;call:get_weather{city:&lt;|"|&gt;Paris&lt;|"|&gt;,days:3}&lt;tool_call|&gt;</c>.
/// </para>
/// </summary>
public sealed class Gemma4ToolCallAdapter : IToolCallAdapter
{
    public const string OpenMarker  = "<|tool_call>";
    public const string CloseMarker = "<tool_call|>";
    public const string CallPrefix  = "call:";
    /// <summary>Gemma's string-delimiter token; brackets a string literal on both sides.</summary>
    public const string Quote       = "<|\"|>";
    /// <summary>
    /// Turn-boundary control tokens that open/close a tool-result turn. The model frequently
    /// emits the opening <c>&lt;|tool_response&gt;</c> immediately after a tool call (it is
    /// starting the next, tool-role, turn) — these are special tokens, never user-facing
    /// content, so they are scrubbed from the parsed plain text (issue #150).
    /// </summary>
    private static readonly string[] ResponseMarkers = ["<|tool_response>", "<tool_response|>"];

    /// <summary>Open/close of Gemma 4's reasoning ("thought") channel — same asymmetric-pipe
    /// convention as the tool markers: open <c>&lt;|channel&gt;</c>, close <c>&lt;channel|&gt;</c>,
    /// with a channel label (e.g. <c>thought</c>) and any reasoning content between them.</summary>
    public const string ChannelOpen  = "<|channel>";
    public const string ChannelClose = "<channel|>";

    public string Architecture => "gemma4";
    public int MaxOpenTagLength => OpenMarker.Length;

    private static readonly string[] s_toolBoundaryStopMarkers = ["<|tool_response>"];

    /// <inheritdoc/>
    /// <remarks>Gemma 4's QAT instruct line opens a <c>&lt;|tool_response&gt;</c> turn after its
    /// tool calls instead of emitting <c>&lt;end_of_turn&gt;</c>, so without this stop the engine
    /// runs on past the (complete, parseable) tool-call block and hallucinates a trailing turn —
    /// issue #304. The block is left intact because the stop token is consumed, not emitted.</remarks>
    public IReadOnlyList<string> ToolBoundaryStopMarkers => s_toolBoundaryStopMarkers;

    /// <inheritdoc/>
    /// <remarks>Constrains the argument object of Gemma's native
    /// <c>&lt;|tool_call&gt;call:NAME{...}&lt;tool_call|&gt;</c> wire format (issue #374). Returns
    /// null when no supplied tool is fully constrainable (then generation is unconstrained).</remarks>
    public ITokenConstraint? BuildArgumentConstraint(IReadOnlyList<ToolSchema> tools, GrammarVocabulary vocab)
    {
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentNullException.ThrowIfNull(vocab);
        var constraint = new GemmaToolArgumentConstraint(vocab, tools);
        return constraint.HasConstrainableTools ? constraint : null;
    }

    public ToolCallParseResult Parse(string rawOutput)
    {
        var calls = new List<ParsedToolCall>();
        var plain = new StringBuilder();
        int pos = 0;

        while (pos < rawOutput.Length)
        {
            int start = rawOutput.IndexOf(OpenMarker, pos, StringComparison.Ordinal);
            if (start < 0)
            {
                plain.Append(rawOutput, pos, rawOutput.Length - pos);
                break;
            }

            plain.Append(rawOutput, pos, start - pos);
            int contentStart = start + OpenMarker.Length;
            int end = rawOutput.IndexOf(CloseMarker, contentStart, StringComparison.Ordinal);
            if (end < 0)
            {
                // Model stopped mid-call (no <tool_call|>): surface the partial as text and flag
                // truncation rather than emitting a half-parsed call. Matches the streaming path.
                plain.Append(rawOutput, contentStart, rawOutput.Length - contentStart);
                return new ToolCallParseResult(ScrubControlMarkup(plain), calls, Truncated: true);
            }

            string block = rawOutput[contentStart..end];
            pos = end + CloseMarker.Length;
            foreach (var c in ParseBlock(block)) calls.Add(c);
        }

        return new ToolCallParseResult(ScrubControlMarkup(plain), calls, Truncated: false);
    }

    /// <summary>
    /// Scrubs Gemma 4 turn/channel control markup from the accumulated plain text so it doesn't
    /// surface as assistant content. Operates on the non-block text only — tool-call blocks were
    /// already extracted — so a marker that legitimately appears inside a tool argument is untouched.
    /// Two kinds of markup are removed:
    /// <list type="bullet">
    /// <item>orphan tool-response turn markers (<c>&lt;|tool_response&gt;</c>/<c>&lt;tool_response|&gt;</c>),
    /// emitted when the model opens the next turn after a tool call (issue #150); and</item>
    /// <item>reasoning-channel blocks (<c>&lt;|channel&gt;label … &lt;channel|&gt;</c>): the header,
    /// label, and any reasoning content up to the close are dropped, keeping the text before
    /// <c>&lt;|channel&gt;</c> and after <c>&lt;channel|&gt;</c> — mirroring the GGUF chat template's
    /// own history scrubber (issue #304). An unterminated <c>&lt;|channel&gt;</c> (no close) drops to
    /// the end; a bare <c>&lt;channel|&gt;</c> close with no open is removed in place (the
    /// post-tool generation prompt can prime <c>&lt;|channel&gt;</c>, so the answer pass emits a
    /// lone close).</item>
    /// </list>
    /// </summary>
    private static string ScrubControlMarkup(StringBuilder plain)
    {
        foreach (var marker in ResponseMarkers)
            plain.Replace(marker, "");

        string s = plain.ToString();
        if (s.IndexOf(ChannelOpen, StringComparison.Ordinal) < 0
            && s.IndexOf(ChannelClose, StringComparison.Ordinal) < 0)
            return s;

        var sb = new StringBuilder(s.Length);
        int pos = 0;
        while (pos < s.Length)
        {
            int open = s.IndexOf(ChannelOpen, pos, StringComparison.Ordinal);
            int closeOnly = s.IndexOf(ChannelClose, pos, StringComparison.Ordinal);

            // A close that precedes the next open (or stands alone) is a bare/orphan close —
            // drop the marker, keep the surrounding text.
            if (closeOnly >= 0 && (open < 0 || closeOnly < open))
            {
                sb.Append(s, pos, closeOnly - pos);
                pos = closeOnly + ChannelClose.Length;
                continue;
            }

            if (open < 0) { sb.Append(s, pos, s.Length - pos); break; }

            sb.Append(s, pos, open - pos);                                  // text before the block
            int close = s.IndexOf(ChannelClose, open + ChannelOpen.Length, StringComparison.Ordinal);
            if (close < 0) break;                                           // unterminated: drop to end
            pos = close + ChannelClose.Length;                             // resume after the close
        }
        return sb.ToString();
    }

    public int FindOpenMarker(string buffer, int startSearch, out int contentStart)
    {
        int idx = buffer.IndexOf(OpenMarker, startSearch, StringComparison.Ordinal);
        contentStart = idx < 0 ? -1 : idx + OpenMarker.Length;
        return idx;
    }

    public int FindCloseMarker(string buffer, int startSearch, out int afterClose)
    {
        int idx = buffer.IndexOf(CloseMarker, startSearch, StringComparison.Ordinal);
        afterClose = idx < 0 ? -1 : idx + CloseMarker.Length;
        return idx;
    }

    public IReadOnlyList<ParsedToolCall> ParseBlock(string block)
    {
        block = block.Trim();
        // Strip the leading "call:" if present (tolerant of drift where the model omits it).
        int nameStart = block.StartsWith(CallPrefix, StringComparison.Ordinal) ? CallPrefix.Length : 0;

        int brace = block.IndexOf('{', nameStart);
        if (brace < 0)
        {
            string nm = block[nameStart..].Trim();
            return nm.Length > 0
                ? [new ParsedToolCall(nm, new Dictionary<string, object?>(StringComparer.Ordinal))]
                : [];
        }

        string name = block[nameStart..brace].Trim();
        if (name.Length == 0) return [];

        int p = brace;
        var args = ParseObject(block, ref p);
        return [new ParsedToolCall(name, args)];
    }

    public Dictionary<string, object?> RenderToolResult(string toolUseId, string resultContent) =>
        // Carry the tool_call_id so the Gemma chat template's forward-scan can resolve the
        // originating function name (it matches assistant tool_calls[].id == tool_call_id).
        new(StringComparer.Ordinal)
        {
            ["role"]         = "tool",
            ["content"]      = resultContent,
            ["tool_call_id"] = toolUseId,
        };

    // ── Gemma argument-syntax parser ──────────────────────────────────────────

    private static Dictionary<string, object?> ParseObject(string s, ref int p)
    {
        var dict = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (p < s.Length && s[p] == '{') p++;   // skip '{'
        SkipWs(s, ref p);

        while (p < s.Length && s[p] != '}')
        {
            string key = ReadKey(s, ref p);
            SkipWs(s, ref p);
            if (p < s.Length && s[p] == ':') p++;
            object? val = ParseValue(s, ref p);
            if (key.Length > 0) dict[key] = val;

            SkipWs(s, ref p);
            if (p < s.Length && s[p] == ',') p++;
            SkipWs(s, ref p);
        }
        if (p < s.Length && s[p] == '}') p++;   // skip '}'
        return dict;
    }

    private static List<object?> ParseArray(string s, ref int p)
    {
        var list = new List<object?>();
        if (p < s.Length && s[p] == '[') p++;    // skip '['
        SkipWs(s, ref p);

        while (p < s.Length && s[p] != ']')
        {
            list.Add(ParseValue(s, ref p));
            SkipWs(s, ref p);
            if (p < s.Length && s[p] == ',') p++;
            SkipWs(s, ref p);
        }
        if (p < s.Length && s[p] == ']') p++;    // skip ']'
        return list;
    }

    private static object? ParseValue(string s, ref int p)
    {
        SkipWs(s, ref p);
        if (p >= s.Length) return null;

        if (IsAt(s, p, Quote)) return ReadQuoted(s, ref p);
        char c = s[p];
        if (c == '{') return ParseObject(s, ref p);
        if (c == '[') return ParseArray(s, ref p);

        // Bareword scalar: read up to the next structural delimiter.
        int start = p;
        while (p < s.Length && s[p] is not (',' or '}' or ']')) p++;
        return ParseScalar(s[start..p].Trim());
    }

    private static string ReadKey(string s, ref int p)
    {
        SkipWs(s, ref p);
        if (IsAt(s, p, Quote)) return ReadQuoted(s, ref p);
        int start = p;
        while (p < s.Length && s[p] is not (':' or '}' or ',')) p++;
        return s[start..p].Trim();
    }

    private static string ReadQuoted(string s, ref int p)
    {
        p += Quote.Length;   // skip opening <|"|>
        int end = s.IndexOf(Quote, p, StringComparison.Ordinal);
        string val = end < 0 ? s[p..] : s[p..end];
        p = end < 0 ? s.Length : end + Quote.Length;
        return val;
    }

    private static object? ParseScalar(string token) => token switch
    {
        "true"  => true,
        "false" => false,
        "null" or "" => null,
        _ => long.TryParse(token, System.Globalization.NumberStyles.Integer,
                           System.Globalization.CultureInfo.InvariantCulture, out long l) ? l
           : double.TryParse(token, System.Globalization.NumberStyles.Float,
                           System.Globalization.CultureInfo.InvariantCulture, out double d) ? (object?)d
           : token,
    };

    private static bool IsAt(string s, int p, string marker) =>
        p + marker.Length <= s.Length && s.AsSpan(p, marker.Length).SequenceEqual(marker);

    private static void SkipWs(string s, ref int p)
    {
        while (p < s.Length && char.IsWhiteSpace(s[p])) p++;
    }
}
