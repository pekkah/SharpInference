using System.Text;
using System.Text.Json;
using SharpInference.Core.Grammar;

namespace SharpInference.Tests.Core;

/// <summary>
/// Pure (model-independent) tests for JSON-Schema → <see cref="ToolSchema"/> derivation (issue
/// #374): types, required keys, enums, arrays, nested objects, and the open/unconstrained fallbacks.
/// </summary>
public sealed class ToolSchemaParseTests
{
    private static ToolSchema Parse(string name, string json)
    {
        using var doc = JsonDocument.Parse(json);
        return ToolSchema.FromOpenAiFunction(name, doc.RootElement.Clone());
    }

    [Fact]
    public void Types_And_Required_AreParsed()
    {
        var s = Parse("t", """
            {"type":"object",
             "properties":{
               "location":{"type":"string"},
               "days":{"type":"integer"},
               "ratio":{"type":"number"},
               "active":{"type":"boolean"}},
             "required":["location"]}
            """);

        Assert.Equal("t", s.Name);
        Assert.False(s.Arguments.Open);
        var props = s.Arguments.Properties;
        Assert.Equal(4, props.Count);

        Assert.Equal(JsonSchemaKind.String, props[0].Value.Kind);
        Assert.True(props[0].Required);
        Assert.Equal(JsonSchemaKind.Integer, props[1].Value.Kind);
        Assert.False(props[1].Required);
        Assert.Equal(JsonSchemaKind.Number, props[2].Value.Kind);
        Assert.Equal(JsonSchemaKind.Boolean, props[3].Value.Kind);
    }

    [Fact]
    public void Enum_IsParsed_AsRestrictedSet()
    {
        var s = Parse("t", """
            {"type":"object","properties":{"unit":{"type":"string","enum":["celsius","fahrenheit"]}}}
            """);
        var node = s.Arguments.Properties[0].Value;
        Assert.Equal(JsonSchemaKind.String, node.Kind);
        Assert.NotNull(node.EnumValues);
        Assert.Equal(["celsius", "fahrenheit"], node.EnumValues!);
    }

    [Fact]
    public void Array_ItemTypeIsParsed()
    {
        var s = Parse("t", """
            {"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}}}
            """);
        var node = s.Arguments.Properties[0].Value;
        Assert.Equal(JsonSchemaKind.Array, node.Kind);
        Assert.NotNull(node.Items);
        Assert.Equal(JsonSchemaKind.String, node.Items!.Kind);
    }

    [Fact]
    public void NestedObject_IsParsedRecursively()
    {
        var s = Parse("t", """
            {"type":"object","properties":{
               "filter":{"type":"object",
                         "properties":{"city":{"type":"string"}},
                         "required":["city"]}}}
            """);
        var node = s.Arguments.Properties[0].Value;
        Assert.Equal(JsonSchemaKind.Object, node.Kind);
        Assert.NotNull(node.Object);
        Assert.Single(node.Object!.Properties);
        Assert.True(node.Object.Properties[0].Required);
    }

    [Fact]
    public void NoProperties_OrNullSchema_IsOpen()
    {
        Assert.True(Parse("t", """{"type":"object","properties":{}}""").Arguments.Open);
        Assert.True(ToolSchema.FromOpenAiFunction("t", null).Arguments.Open);
        // additionalProperties:true → open even with declared properties.
        Assert.True(Parse("t", """
            {"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":true}
            """).Arguments.Open);
    }

    [Fact]
    public void UnionType_TakesFirstConcrete_IgnoringNull()
    {
        var s = Parse("t", """{"type":"object","properties":{"x":{"type":["string","null"]}}}""");
        Assert.Equal(JsonSchemaKind.String, s.Arguments.Properties[0].Value.Kind);
    }

    [Fact]
    public void DeeplyNestedSchema_DoesNotStackOverflow_DegradesPastCap()
    {
        // A property nested past the parser's depth cap (via arrays — 1 JSON level each, so we stay
        // under System.Text.Json's own depth limit) must parse without recursing unbounded and
        // blowing the stack (a .NET StackOverflowException is uncatchable). Deep levels degrade to
        // Any rather than recursing forever.
        const int n = 55;   // > ToolSchema parse cap (32), < System.Text.Json depth limit (64)
        var sb = new StringBuilder();
        sb.Append("""{"type":"object","properties":{"x":""");
        for (int i = 0; i < n; i++) sb.Append("""{"type":"array","items":""");
        sb.Append("""{"type":"string"}""");
        for (int i = 0; i < n; i++) sb.Append('}');
        sb.Append("}}");

        var s = Parse("t", sb.ToString());
        Assert.Single(s.Arguments.Properties);
        Assert.Equal(JsonSchemaKind.Array, s.Arguments.Properties[0].Value.Kind);  // top levels parse
    }
}
