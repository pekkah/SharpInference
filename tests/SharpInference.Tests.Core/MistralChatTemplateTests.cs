using SharpInference.Core;

namespace SharpInference.Tests.Core;

/// <summary>
/// The Jinja constructs Mistral's v3 (Tekken) chat template depends on. The focused facts pin one
/// previously mis-parsed or silently no-op'd construct each; the final four render the real
/// template verbatim.
/// </summary>
public sealed class MistralChatTemplateTests
{
    private static Dictionary<string, object?> Msg(string role, string content) =>
        new() { ["role"] = role, ["content"] = content };

    private static string Render(string template, params (string Key, object? Val)[] ctx)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var (k, v) in ctx) dict[k] = v;
        return new JinjaChatTemplate(template).Render(dict);
    }

    [Theory]
    [InlineData("{{ 7 % 3 }}", "1")]
    [InlineData("{{ 1 + 2 * 3 }}", "7")]     // * binds tighter than +
    [InlineData("{{ 7 // 2 }}", "3")]
    [InlineData("{{ -7 // 2 }}", "-4")]      // Python floors toward -inf, not toward zero
    [InlineData("{{ -7 % 2 }}", "1")]        // modulo takes the divisor's sign
    // Mistral's alternation guard. Before `%` existed the parser stopped at it and silently
    // discarded `% 2 == 0)`, leaving `... != ns.index` — true for the first user message, which
    // tripped the template's own raise_exception on every render.
    [InlineData("{% set ns = namespace() %}{% set ns.index = 0 %}" +
                "{{ ('user' == 'user') != (ns.index % 2 == 0) }}", "False")]
    public void Arithmetic_Operators(string template, string expected)
    {
        Assert.Equal(expected, Render(template));
    }

    [Fact]
    public void SelectAttr_WithEqualTo_NarrowsToMatchingRole()
    {
        const string t = """
            {%- set users = messages | selectattr("role", "equalto", "user") | list -%}
            {{ users | length }}:{{ users[-1]["content"] }}
            """;
        // The point of the filter: users[-1] must be "c" (the last USER message), not "b".
        var messages = new List<object?> { Msg("user", "a"), Msg("assistant", "b"), Msg("user", "c") };
        Assert.Equal("2:c", Render(t, ("messages", messages)));

        // A caller-built List<Dictionary<…>> is not a List<object?>; `| list` used to return null
        // for it, silently emptying the pipeline.
        var typed = new List<Dictionary<string, object?>> { Msg("user", "a"), Msg("assistant", "b") };
        Assert.Equal("1:a", Render(t, ("messages", typed)));
    }

    [Fact]
    public void IsTest_WithArgument_IsParsedAndApplied()
    {
        Assert.Equal("yes", Render("""{% if "user" is equalto("user") %}yes{% else %}no{% endif %}"""));
        Assert.Equal("no", Render("""{% if "tool" is equalto("user") %}yes{% else %}no{% endif %}"""));
    }

    [Fact]
    public void StringIndexAndSlice_SupportNegativeBounds()
    {
        // `out[:-1]` reopens a serialized JSON object so the template can splice in a call id.
        Assert.Equal("{\"name\": \"f\"", Render("""{% set out = '{"name": "f"}' %}{{ out[:-1] }}"""));
        Assert.Equal("o", Render("{{ 'hello'[-1] }}"));
    }

    [Fact]
    public void ForLoop_InlineIfFilter_ExcludesItems_AndRenumbersLoopVars()
    {
        // The condition must be evaluated per item with the loop variables bound, not once up
        // front with them undefined (which is how it parsed as a ternary and filtered nothing).
        const string t = """{% for k, v in d.items() if k != "return" %}{{ k }}={{ v }}{% if not loop.last %};{% endif %}{% endfor %}""";
        var d = new Dictionary<string, object?> { ["name"] = "f", ["return"] = "int", ["desc"] = "x" };
        Assert.Equal("name=f;desc=x", Render(t, ("d", d)));
    }

    [Fact]
    public void EosToken_IsSeededIntoTheContext_UnlessCallerSuppliesOne()
    {
        var tmpl = new JinjaChatTemplate("{{ bos_token }}A{{ eos_token }}")
        {
            BosToken = "<s>",
            EosToken = "</s>",
        };
        Assert.Equal("<s>A</s>", tmpl.Render(new Dictionary<string, object?>()));
        Assert.Equal("<s>AX", tmpl.Render(new Dictionary<string, object?> { ["eos_token"] = "X" }));
    }

    /// <summary>Mistral v3 (Tekken) chat template, verbatim from the model repo.</summary>
    private const string MistralV3Template = """
        {%- if messages[0]["role"] == "system" %}
            {%- set system_message = messages[0]["content"] %}
            {%- set loop_messages = messages[1:] %}
        {%- else %}
            {%- set loop_messages = messages %}
        {%- endif %}
        {%- if not tools is defined %}
            {%- set tools = none %}
        {%- endif %}
        {%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}

        {%- set ns = namespace() %}
        {%- set ns.index = 0 %}
        {%- for message in loop_messages %}
            {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}
                {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}
                    {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}
                {%- endif %}
                {%- set ns.index = ns.index + 1 %}
            {%- endif %}
        {%- endfor %}

        {{- bos_token }}
        {%- for message in loop_messages %}
            {%- if message["role"] == "user" %}
                {%- if loop.last and system_message is defined %}
                    {{- "[INST]" + system_message + "\n\n" + message["content"] + "[/INST]" }}
                {%- else %}
                    {{- "[INST]" + message["content"] + "[/INST]" }}
                {%- endif %}
            {%- elif message["role"] == "assistant" %}
                {{- message["content"] + eos_token}}
            {%- endif %}
        {%- endfor %}
        """;

    private static string RenderMistral(params object?[] messages) =>
        new JinjaChatTemplate(MistralV3Template) { BosToken = "<s>", EosToken = "</s>" }
            .Render(new Dictionary<string, object?> { ["messages"] = messages.ToList() });

    [Fact]
    public void MistralV3_RendersInstructTurns()
    {
        Assert.Equal("<s>[INST]Hello[/INST]", RenderMistral(Msg("user", "Hello")));

        // Mistral has no system role: the system text prepends the FINAL user message.
        Assert.Equal("<s>[INST]Be brief.\n\nHello[/INST]",
            RenderMistral(Msg("system", "Be brief."), Msg("user", "Hello")));

        // Assistant turns close with eos_token, which previously rendered empty — leaving
        // multi-turn history with no turn boundaries at all.
        Assert.Equal("<s>[INST]Hi[/INST]Hello!</s>[INST]Bye[/INST]",
            RenderMistral(Msg("user", "Hi"), Msg("assistant", "Hello!"), Msg("user", "Bye")));
    }

    [Fact]
    public void MistralV3_NonAlternatingRoles_RaiseTheTemplateException()
    {
        // The guard must still fire when it genuinely should — proof the fix corrected the
        // condition rather than merely disabling it. ChatTemplateException specifically: it marks
        // this as rejected INPUT, which is what lets the API endpoints answer 400 rather than 500.
        var ex = Assert.Throws<ChatTemplateException>(
            () => RenderMistral(Msg("user", "a"), Msg("user", "b")));
        Assert.Contains("must alternate", ex.Message);
    }
}
