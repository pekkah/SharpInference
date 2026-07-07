namespace SharpInference.Core.Grammar;

/// <summary>
/// Whole-turn JSON-Schema output constraint (issue #423 follow-up): constrains the ENTIRE assistant
/// response to a caller-supplied JSON Schema, with no tool-call envelope -- the SharpInference
/// analogue of OpenAI/llama.cpp "structured output" (<c>response_format: json_schema</c>). Built on
/// the same frame-stack matcher <see cref="JsonToolArgumentConstraint"/> uses for a tool call's
/// argument object (issue #376), just engaged from the first token instead of after a preamble.
///
/// <para>
/// Shares <see cref="JsonToolArgumentConstraint"/>'s known limitations (no <c>$ref</c>,
/// <c>oneOf</c>/<c>anyOf</c>/<c>allOf</c>, <c>pattern</c>, <c>minLength</c>/<c>maxLength</c>,
/// <c>minimum</c>/<c>maximum</c>, <c>minItems</c>/<c>maxItems</c>). Unlike that constraint's
/// "silently degrade to unconstrained" philosophy for tool-calling (a best-effort nicety), an
/// explicit schema-constrained request is a hard requirement -- the constructor throws when the
/// schema can't be compiled, rather than silently generating unconstrained output.
/// </para>
/// </summary>
public sealed class JsonSchemaOutputConstraint : ITokenConstraint
{
    private readonly JsonToolArgumentConstraint _inner;

    /// <param name="vocab">Vocabulary view for the loaded model (issue #374).</param>
    /// <param name="schema">
    /// The schema's root object (e.g. <c>ToolSchema.FromOpenAiFunction("_", schemaElement).Arguments</c>
    /// for a raw JSON Schema <see cref="System.Text.Json.JsonElement"/>). Must be an object schema
    /// declaring at least one property.
    /// </param>
    /// <exception cref="ArgumentException">
    /// The schema could not be compiled to a constraint -- it isn't an object schema, declares no
    /// properties, or uses only unsupported keywords.
    /// </exception>
    public JsonSchemaOutputConstraint(GrammarVocabulary vocab, ToolSchemaObject schema)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        ArgumentNullException.ThrowIfNull(schema);
        var compiled = ToolSchemaCompiler.TryCompileObject(schema)
            ?? throw new ArgumentException(
                "schema could not be compiled to a constraint (must be an object schema with at " +
                "least one declared property; unsupported keywords like $ref/oneOf/pattern are " +
                "not constrained)", nameof(schema));
        _inner = JsonToolArgumentConstraint.ForWholeBody(vocab, compiled);
    }

    public bool IsConstraining => _inner.IsConstraining;
    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits) => _inner.Filter(logits);
    public void Accept(int token) => _inner.Accept(token);
    public void Reset() => _inner.Reset();
}
