using System.Text;
using System.Text.Json;

namespace SharpInference.Core.Grammar;

/// <summary>
/// The JSON-Schema value kinds a tool argument can take. Mirrors the <c>type</c> keyword of a
/// JSON Schema; <see cref="Any"/> is the fallback for a missing/unknown/union type (the grammar
/// leaves such values structurally permissive rather than forbidding generation).
/// </summary>
public enum JsonSchemaKind
{
    Any = 0,
    String,
    Number,
    Integer,
    Boolean,
    Null,
    Array,
    Object,
}

/// <summary>
/// A value-type descriptor parsed from one JSON-Schema node (recursive). Carries just the
/// facets the argument grammar enforces: the <see cref="Kind"/>, an optional restricted
/// <see cref="EnumValues"/> set, the element type for arrays (<see cref="Items"/>), and the
/// nested shape for objects (<see cref="Object"/>). Free-form facets (descriptions, min/max,
/// pattern, …) are intentionally dropped — the grammar constrains structure, not content.
/// </summary>
public sealed class ToolSchemaNode
{
    public JsonSchemaKind Kind { get; }

    /// <summary>
    /// When non-null, the value is restricted to exactly these rendered token strings (a JSON
    /// Schema <c>enum</c>). For a <see cref="JsonSchemaKind.String"/> node each entry is the raw
    /// string (emitted inside the <c>&lt;|"|&gt;</c> quotes); for non-string kinds each entry is the
    /// bare literal (e.g. a number). Capped at 64 entries (the constraint's bitmask width); a larger
    /// enum degrades to an unrestricted value of the same kind.
    /// </summary>
    public IReadOnlyList<string>? EnumValues { get; }

    /// <summary>Element type for <see cref="JsonSchemaKind.Array"/>; null = unconstrained elements.</summary>
    public ToolSchemaNode? Items { get; }

    /// <summary>Nested shape for <see cref="JsonSchemaKind.Object"/>; null = unconstrained object body.</summary>
    public ToolSchemaObject? Object { get; }

    public ToolSchemaNode(
        JsonSchemaKind kind,
        IReadOnlyList<string>? enumValues = null,
        ToolSchemaNode? items = null,
        ToolSchemaObject? @object = null)
    {
        Kind = kind;
        EnumValues = enumValues is { Count: > 0 and <= 64 } ? enumValues : null;
        Items = items;
        Object = @object;
    }

    /// <summary>A fully-unconstrained value (any well-formed shape). Shared singleton.</summary>
    public static ToolSchemaNode Any { get; } = new(JsonSchemaKind.Any);
}

/// <summary>One declared property of an object: its name, value type, and whether it is required.</summary>
public sealed record ToolSchemaProperty(string Name, ToolSchemaNode Value, bool Required);

/// <summary>
/// The parsed shape of an object: its ordered declared <see cref="Properties"/>. When
/// <see cref="Open"/> is true the object body is left unconstrained (no declared properties, or an
/// explicit <c>additionalProperties: true</c>) — the grammar then only balances braces rather than
/// restricting keys. Property count is capped at 64 (the constraint's key bitmask width); a wider
/// object falls back to <see cref="Open"/>.
/// </summary>
public sealed class ToolSchemaObject
{
    public IReadOnlyList<ToolSchemaProperty> Properties { get; }

    /// <summary>Object body is unconstrained (balance braces only, accept any keys/values).</summary>
    public bool Open { get; }

    public ToolSchemaObject(IReadOnlyList<ToolSchemaProperty> properties, bool open)
    {
        Properties = properties;
        Open = open || properties.Count is 0 or > 64;
    }
}

/// <summary>
/// A tool's name plus the parsed shape of its argument object. Produced by
/// <see cref="FromOpenAiFunction"/> from an OpenAI-style function definition and consumed by an
/// architecture's argument-grammar builder (see
/// <see cref="IToolCallAdapter.BuildArgumentConstraint"/>).
/// </summary>
public sealed record ToolSchema(string Name, ToolSchemaObject Arguments)
{
    /// <summary>
    /// Parses a tool's <c>parameters</c> JSON-Schema object into a <see cref="ToolSchema"/>. A null /
    /// non-object / malformed schema yields an <see cref="ToolSchemaObject.Open"/> argument object
    /// (the grammar then constrains nothing for that tool — never blocks generation).
    /// </summary>
    // Cap on schema nesting parsed. Tool schemas come straight from the request body, so an
    // adversarially/accidentally deep schema must not blow the stack (a .NET StackOverflow is
    // uncatchable and crashes the process). Past the cap a node degrades to Any / an open object,
    // which the constraint leaves unconstrained anyway — no correctness loss for realistic schemas.
    private const int MaxParseDepth = 32;

    public static ToolSchema FromOpenAiFunction(string name, JsonElement? parameters)
    {
        if (parameters is not { ValueKind: JsonValueKind.Object } p)
            return new ToolSchema(name, new ToolSchemaObject([], open: true));
        return new ToolSchema(name, ParseObject(p, depth: 0));
    }

    private static ToolSchemaObject ParseObject(JsonElement schema, int depth)
    {
        if (depth >= MaxParseDepth)
            return new ToolSchemaObject([], open: true);   // too deep — treat as unconstrained

        var required = new HashSet<string>(StringComparer.Ordinal);
        if (schema.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array)
            foreach (var r in req.EnumerateArray())
                if (r.ValueKind == JsonValueKind.String && r.GetString() is { } s)
                    required.Add(s);

        bool open = false;
        if (schema.TryGetProperty("additionalProperties", out var ap) && ap.ValueKind == JsonValueKind.True)
            open = true;

        var props = new List<ToolSchemaProperty>();
        if (schema.TryGetProperty("properties", out var properties)
            && properties.ValueKind == JsonValueKind.Object)
        {
            foreach (var prop in properties.EnumerateObject())
            {
                var node = ParseNode(prop.Value, depth + 1);
                props.Add(new ToolSchemaProperty(prop.Name, node, required.Contains(prop.Name)));
            }
        }

        return new ToolSchemaObject(props, open);
    }

    private static ToolSchemaNode ParseNode(JsonElement node, int depth)
    {
        if (node.ValueKind != JsonValueKind.Object || depth >= MaxParseDepth)
            return ToolSchemaNode.Any;

        var kind = ParseKind(node);

        // enum: collect the rendered literal strings. Applies regardless of declared kind.
        IReadOnlyList<string>? enumValues = null;
        if (node.TryGetProperty("enum", out var en) && en.ValueKind == JsonValueKind.Array)
        {
            var vals = new List<string>();
            foreach (var e in en.EnumerateArray())
            {
                string? rendered = e.ValueKind switch
                {
                    JsonValueKind.String => e.GetString(),
                    JsonValueKind.Number => e.GetRawText(),
                    JsonValueKind.True => "true",
                    JsonValueKind.False => "false",
                    JsonValueKind.Null => "null",
                    _ => null,
                };
                if (rendered is { Length: > 0 }) vals.Add(rendered);
            }
            if (vals.Count > 0) enumValues = vals;
            // A string-typed enum stays String (values are quoted); an untyped enum of strings is
            // also treated as String so the values render inside the quote token.
            if (kind == JsonSchemaKind.Any && enumValues is not null
                && en.EnumerateArray().Any(e => e.ValueKind == JsonValueKind.String))
                kind = JsonSchemaKind.String;
        }

        ToolSchemaNode? items = null;
        if (kind == JsonSchemaKind.Array && node.TryGetProperty("items", out var it))
            items = ParseNode(it, depth + 1);

        ToolSchemaObject? obj = null;
        if (kind == JsonSchemaKind.Object)
            obj = ParseObject(node, depth + 1);

        return new ToolSchemaNode(kind, enumValues, items, obj);
    }

    private static JsonSchemaKind ParseKind(JsonElement node)
    {
        if (!node.TryGetProperty("type", out var t))
            return JsonSchemaKind.Any;

        // "type" may be a single string or an array of strings (a union); take the first concrete
        // entry for a union and leave Any when nothing maps.
        string? typeStr = t.ValueKind switch
        {
            JsonValueKind.String => t.GetString(),
            JsonValueKind.Array => t.EnumerateArray()
                                    .Select(e => e.ValueKind == JsonValueKind.String ? e.GetString() : null)
                                    .FirstOrDefault(s => s is not null and not "null"),
            _ => null,
        };

        return typeStr?.ToLowerInvariant() switch
        {
            "string" => JsonSchemaKind.String,
            "number" => JsonSchemaKind.Number,
            "integer" => JsonSchemaKind.Integer,
            "boolean" => JsonSchemaKind.Boolean,
            "null" => JsonSchemaKind.Null,
            "array" => JsonSchemaKind.Array,
            "object" => JsonSchemaKind.Object,
            _ => JsonSchemaKind.Any,
        };
    }

    /// <summary>UTF-8 bytes of <paramref name="s"/> — small helper used when precomputing literal
    /// match tables for the constraint.</summary>
    internal static byte[] Utf8(string s) => Encoding.UTF8.GetBytes(s);
}
