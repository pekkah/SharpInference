namespace SharpInference.Core.Grammar;

// ── Compiled schema ───────────────────────────────────────────────────────────
//
// A parsed ToolSchema is "compiled" once per request into match tables (UTF-8 key/enum bytes,
// required-key bitmask) so the per-token hot loop never re-encodes strings. The compiled form is
// wire-format-agnostic — both the Gemma constraint (GemmaToolArgumentConstraint, bespoke
// <|"|>-quoted syntax) and the JSON constraint (JsonToolArgumentConstraint, standard JSON for
// Qwen/Llama/DeepSeek) consume the same CompiledObject/CompiledNode and only differ in how they
// walk the structural bytes. Only tools whose schema is FULLY constrainable (every value is a
// concrete type / typed array / nested typed object — no Any-typed value, no open object) are
// compiled; a tool that isn't is left out of the constraint and generates its arguments
// unconstrained, so the constraint never blocks generation.

/// <summary>Compiled value-type descriptor (see <see cref="ToolSchemaNode"/>).</summary>
internal sealed class CompiledNode
{
    public required JsonSchemaKind Kind { get; init; }
    /// <summary>For an enum / boolean / null: the candidate literal byte sequences (≤64).</summary>
    public byte[][]? Literals { get; init; }
    /// <summary>True when <see cref="Literals"/> are matched as a quoted string (string enum).</summary>
    public bool QuotedLiterals { get; init; }
    /// <summary>Element type for an array.</summary>
    public CompiledNode? Items { get; init; }
    /// <summary>Nested object shape.</summary>
    public CompiledObject? Object { get; init; }
    /// <summary>Integer (no fractional part) vs general number.</summary>
    public bool IntegerOnly { get; init; }
}

/// <summary>Compiled object shape: per-property name bytes + value node, plus the required-key mask.</summary>
internal sealed class CompiledObject
{
    public required byte[][] KeyBytes { get; init; }
    public required CompiledNode[] Values { get; init; }
    public required ulong RequiredMask { get; init; }
    public int Count => KeyBytes.Length;
}

/// <summary>
/// Compiles a parsed <see cref="ToolSchemaObject"/> into the match tables the argument-grammar
/// state machines drive. Shared by every architecture's constraint so the "which schemas are
/// constrainable" rule lives in exactly one place. Returns <c>null</c> for any schema that isn't
/// fully constrainable (open body, an <c>Any</c>-typed value, an untyped array, …) — the caller
/// then leaves that tool unconstrained.
/// </summary>
internal static class ToolSchemaCompiler
{
    // Depth-capped so compilation can't recurse unbounded — the parser already caps nesting, but the
    // ToolSchema records are public, so a caller that builds a deeply-nested schema in-memory (not via
    // the parser) would otherwise risk an uncatchable StackOverflow. Past the cap the tool is simply
    // treated as non-constrainable (null), matching the "loosely-typed → unconstrained" contract.
    private const int MaxDepth = 32;

    private static readonly byte[][] s_boolLiterals = [ToolSchema.Utf8("true"), ToolSchema.Utf8("false")];
    private static readonly byte[][] s_nullLiterals = [ToolSchema.Utf8("null")];

    public static CompiledObject? TryCompileObject(ToolSchemaObject obj) => TryCompileObject(obj, 0);

    private static CompiledObject? TryCompileObject(ToolSchemaObject obj, int depth)
    {
        if (depth >= MaxDepth || obj.Open || obj.Properties.Count is 0 or > 64) return null;
        var keys = new byte[obj.Properties.Count][];
        var values = new CompiledNode[obj.Properties.Count];
        ulong reqMask = 0;
        for (int i = 0; i < obj.Properties.Count; i++)
        {
            var p = obj.Properties[i];
            if (p.Name.Length == 0) return null;
            keys[i] = ToolSchema.Utf8(p.Name);
            var node = TryCompileNode(p.Value, depth + 1);
            if (node is null) return null;                  // a non-constrainable value → skip the tool
            values[i] = node;
            if (p.Required) reqMask |= 1UL << i;
        }
        return new CompiledObject { KeyBytes = keys, Values = values, RequiredMask = reqMask };
    }

    private static CompiledNode? TryCompileNode(ToolSchemaNode node, int depth)
    {
        if (depth >= MaxDepth) return null;
        switch (node.Kind)
        {
            case JsonSchemaKind.String:
                byte[][]? strEnum = node.EnumValues is { } se ? Encode(se) : null;
                return new CompiledNode { Kind = JsonSchemaKind.String, Literals = strEnum, QuotedLiterals = strEnum is not null };

            case JsonSchemaKind.Number:
            case JsonSchemaKind.Integer:
                byte[][]? numEnum = node.EnumValues is { } ne ? Encode(ne) : null;
                return new CompiledNode
                {
                    Kind = node.Kind,
                    Literals = numEnum,
                    QuotedLiterals = false,
                    IntegerOnly = node.Kind == JsonSchemaKind.Integer,
                };

            case JsonSchemaKind.Boolean:
                return new CompiledNode { Kind = JsonSchemaKind.Boolean, Literals = s_boolLiterals };

            case JsonSchemaKind.Null:
                return new CompiledNode { Kind = JsonSchemaKind.Null, Literals = s_nullLiterals };

            case JsonSchemaKind.Array:
                // An untyped array (no items) isn't fully constrainable — skip the tool.
                if (node.Items is null) return null;
                var item = TryCompileNode(node.Items, depth + 1);
                return item is null ? null : new CompiledNode { Kind = JsonSchemaKind.Array, Items = item };

            case JsonSchemaKind.Object:
                if (node.Object is null) return null;
                var obj = TryCompileObject(node.Object, depth + 1);
                return obj is null ? null : new CompiledNode { Kind = JsonSchemaKind.Object, Object = obj };

            default:
                return null;   // Any / unknown — not constrainable
        }
    }

    private static byte[][] Encode(IReadOnlyList<string> values)
    {
        var r = new byte[values.Count][];
        for (int i = 0; i < values.Count; i++) r[i] = ToolSchema.Utf8(values[i]);
        return r;
    }
}
