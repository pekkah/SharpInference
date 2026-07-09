namespace SharpInference.Core.Grammar;

// ── Compiled schema ───────────────────────────────────────────────────────────
//
// A parsed ToolSchema is "compiled" once per request into match tables (UTF-8 key/enum bytes,
// required-key bitmask) so the per-token hot loop never re-encodes strings. The compiled form is
// wire-format-agnostic — both the Gemma constraint (GemmaToolArgumentConstraint, bespoke
// <|"|>-quoted syntax) and the JSON constraint (JsonToolArgumentConstraint, standard JSON for
// Qwen/Llama/DeepSeek) consume the same CompiledObject/CompiledNode and only differ in how they
// walk the structural bytes. A loosely-typed VALUE no longer disqualifies its tool (issue #378): it
// compiles to a FreeValue node the constraints accept as any well-formed value, so the surrounding
// structure stays enforced. A tool is left unconstrained only when its argument OBJECT itself is open
// (no declared properties) — the constraint then never blocks generation.

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
    /// <summary>Ordered mode (issue #425): keys must appear in declaration order — optional keys may
    /// be skipped, but a later key can never precede an earlier one.</summary>
    public bool Ordered { get; init; }
    public int Count => KeyBytes.Length;
}

/// <summary>
/// Compiles a parsed <see cref="ToolSchemaObject"/> into the match tables the argument-grammar
/// state machines drive. Shared by every architecture's constraint so the "which schemas are
/// constrainable" rule lives in exactly one place.
///
/// <para>
/// A loosely-typed VALUE (an <c>Any</c>-typed value with no <c>type</c>, an open object, or an
/// untyped array) no longer disqualifies the whole tool (issue #378): it compiles to the
/// <see cref="FreeValue"/> node (<see cref="JsonSchemaKind.Any"/>), which the constraints accept as
/// any single well-formed value while still enforcing the object's <em>structure</em> — declared key
/// names, required-once, and the typed siblings. <see cref="TryCompileObject"/> returns <c>null</c>
/// only when the object body ITSELF is open (no declared properties to enforce), so a partially-typed
/// tool stays constrained on its typed/required parts instead of being dropped wholesale.
/// </para>
/// </summary>
internal static class ToolSchemaCompiler
{
    /// <summary>A fully-free value: any single well-formed value (string, scalar, array, object) the
    /// constraints balance to completion without restricting its contents. Shared singleton.</summary>
    public static CompiledNode FreeValue { get; } = new() { Kind = JsonSchemaKind.Any };

    // Depth-capped so compilation can't recurse unbounded — the parser already caps nesting, but the
    // ToolSchema records are public, so a caller that builds a deeply-nested schema in-memory (not via
    // the parser) would otherwise risk an uncatchable StackOverflow. Past the cap the tool is simply
    // treated as non-constrainable (null), matching the "loosely-typed → unconstrained" contract.
    private const int MaxDepth = 32;

    private static readonly byte[][] s_boolLiterals = [ToolSchema.Utf8("true"), ToolSchema.Utf8("false")];
    private static readonly byte[][] s_nullLiterals = [ToolSchema.Utf8("null")];

    public static CompiledObject? TryCompileObject(ToolSchemaObject obj) => TryCompileObject(obj, ordered: false, 0);

    /// <summary>Compiles with ordered-properties mode (issue #425): every object in the schema tree
    /// requires its keys in declaration order (optional keys skippable, never reordered).</summary>
    public static CompiledObject? TryCompileObject(ToolSchemaObject obj, bool ordered) => TryCompileObject(obj, ordered, 0);

    private static CompiledObject? TryCompileObject(ToolSchemaObject obj, bool ordered, int depth)
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
            // A loosely-typed value compiles to FreeValue (issue #378) rather than disqualifying the
            // tool — the key/required structure stays enforced, only the value is left free.
            values[i] = CompileNode(p.Value, ordered, depth + 1);
            if (p.Required) reqMask |= 1UL << i;
        }
        return new CompiledObject { KeyBytes = keys, Values = values, RequiredMask = reqMask, Ordered = ordered };
    }

    /// <summary>Compiles one value node, degrading any loosely-typed value (Any / untyped array /
    /// open or too-deep object) to <see cref="FreeValue"/> rather than null — so a partially-typed
    /// object still constrains its typed siblings (issue #378).</summary>
    private static CompiledNode CompileNode(ToolSchemaNode node, bool ordered, int depth)
    {
        if (depth >= MaxDepth) return FreeValue;            // too deep to constrain → free
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
                // An untyped array (no item shape) is left free; a typed array constrains its items.
                if (node.Items is null) return FreeValue;
                return new CompiledNode { Kind = JsonSchemaKind.Array, Items = CompileNode(node.Items, ordered, depth + 1) };

            case JsonSchemaKind.Object:
                // An open / too-deep nested object is left free; a typed nested object recurses.
                var obj = node.Object is null ? null : TryCompileObject(node.Object, ordered, depth + 1);
                return obj is null ? FreeValue : new CompiledNode { Kind = JsonSchemaKind.Object, Object = obj };

            default:
                return FreeValue;   // Any / unknown — free value
        }
    }

    private static byte[][] Encode(IReadOnlyList<string> values)
    {
        var r = new byte[values.Count][];
        for (int i = 0; i < values.Count; i++) r[i] = ToolSchema.Utf8(values[i]);
        return r;
    }
}
