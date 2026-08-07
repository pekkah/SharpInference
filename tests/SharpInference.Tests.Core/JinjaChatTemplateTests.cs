using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class JinjaChatTemplateTests
{
    private static string Render(string source, IReadOnlyDictionary<string, object?>? ctx = null) =>
        new JinjaChatTemplate(source).Render(ctx ?? new Dictionary<string, object?>());

    // ── List displays ────────────────────────────────────────────────────────
    // Gemma's chat template gates its system-message handling on
    // `messages[0]['role'] in ['system', 'developer']`. The parser had no '[' case in
    // primary position, so it stopped at the bracket, warned "Unsupported expression",
    // and let the test fall through unevaluated.

    [Fact]
    public void ListLiteral_AsInOperand_MatchesMember() =>
        Assert.Equal("yes", Render("{% if 'system' in ['system', 'developer'] %}yes{% else %}no{% endif %}"));

    [Fact]
    public void ListLiteral_AsInOperand_RejectsNonMember() =>
        Assert.Equal("no", Render("{% if 'user' in ['system', 'developer'] %}yes{% else %}no{% endif %}"));

    [Fact]
    public void ListLiteral_NotIn_Negates() =>
        Assert.Equal("yes", Render("{% if 'user' not in ['system', 'developer'] %}yes{% else %}no{% endif %}"));

    /// <summary>The exact Gemma shape: a subscripted message role tested against a list display.</summary>
    [Fact]
    public void ListLiteral_GemmaRoleGate_Evaluates()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<object?>
            {
                new Dictionary<string, object?> { ["role"] = "developer", ["content"] = "x" },
            },
        };
        Assert.Equal("sys", Render(
            "{% if messages[0]['role'] in ['system', 'developer'] %}sys{% else %}other{% endif %}", ctx));
    }

    [Fact]
    public void ListLiteral_Empty_IsFalsyAndContainsNothing()
    {
        Assert.Equal("no", Render("{% if 'a' in [] %}yes{% else %}no{% endif %}"));
        Assert.Equal("empty", Render("{% if [] %}full{% else %}empty{% endif %}"));
    }

    [Fact]
    public void ListLiteral_Iterates_InOrder() =>
        Assert.Equal("a,b,c,", Render("{% for x in ['a', 'b', 'c'] %}{{ x }},{% endfor %}"));

    [Fact]
    public void ListLiteral_SupportsExpressionElementsAndTrailingComma()
    {
        var ctx = new Dictionary<string, object?> { ["r"] = "system" };
        Assert.Equal("yes", Render("{% if 'system' in [r, 'developer',] %}yes{% else %}no{% endif %}", ctx));
    }

    /// <summary>Postfix subscripting must still win on a name — `messages[0]` is an index, not a display.</summary>
    [Fact]
    public void ListLiteral_DoesNotShadowSubscript()
    {
        var ctx = new Dictionary<string, object?> { ["xs"] = new List<object?> { "a", "b" } };
        Assert.Equal("b", Render("{{ xs[1] }}", ctx));
        Assert.Equal("2", Render("{{ ['a', 'b'] | length }}"));
    }

    [Fact]
    public void Range_Stop_ProducesZeroToStopExclusive() =>
        Assert.Equal("0,1,2,3,", Render("{% for i in range(4) %}{{ i }},{% endfor %}"));

    [Fact]
    public void Range_Zero_ProducesEmpty() =>
        Assert.Equal("", Render("{% for i in range(0) %}{{ i }},{% endfor %}"));

    // Regression: range(-1) used to crash with ArgumentOutOfRangeException because the
    // implementation forwarded a negative count to Enumerable.Range. Python/Jinja silently
    // yield an empty range instead.
    [Fact]
    public void Range_NegativeStop_ProducesEmpty() =>
        Assert.Equal("", Render("{% for i in range(-1) %}{{ i }},{% endfor %}"));

    // The exact pattern from the upstream consumer bug report:
    // range(messages | length - 1) when messages is empty → length 0 → range(-1).
    [Fact]
    public void Range_LengthMinusOne_OnEmptyList_DoesNotThrow()
    {
        var ctx = new Dictionary<string, object?> { ["messages"] = new List<object?>() };
        Assert.Equal("", Render("{% for i in range(messages | length - 1) %}x{% endfor %}", ctx));
    }

    [Fact]
    public void Range_StartStop_ProducesStartInclusiveToStopExclusive() =>
        Assert.Equal("2,3,4,", Render("{% for i in range(2, 5) %}{{ i }},{% endfor %}"));

    [Fact]
    public void Range_StartGreaterThanStop_ProducesEmpty() =>
        Assert.Equal("", Render("{% for i in range(5, 2) %}{{ i }},{% endfor %}"));

    [Fact]
    public void Range_PositiveStep_SkipsValues() =>
        Assert.Equal("0,2,4,6,8,", Render("{% for i in range(0, 10, 2) %}{{ i }},{% endfor %}"));

    [Fact]
    public void Range_NegativeStep_CountsDown() =>
        Assert.Equal("5,4,3,2,1,", Render("{% for i in range(5, 0, -1) %}{{ i }},{% endfor %}"));

    [Fact]
    public void Range_StepZero_Throws() =>
        Assert.Throws<InvalidOperationException>(() =>
            Render("{% for i in range(0, 5, 0) %}{{ i }},{% endfor %}"));

    // ── Macro tests ──────────────────────────────────────────────────────────────

    [Fact]
    public void Macro_Basic_RendersBody()
    {
        const string src = "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}{{ greet('World') }}";
        Assert.Equal("Hello World!", Render(src));
    }

    [Fact]
    public void Macro_WithDefault_UsesDefaultWhenNotProvided()
    {
        const string src = "{% macro greet(name, punct='.') %}Hi {{ name }}{{ punct }}{% endmacro %}{{ greet('Alice') }}|{{ greet('Bob', '!') }}";
        Assert.Equal("Hi Alice.|Hi Bob!", Render(src));
    }

    [Fact]
    public void Macro_CalledMultipleTimes_RendersCorrectly()
    {
        const string src = "{% macro item(x) %}[{{ x }}]{% endmacro %}{% for i in range(3) %}{{ item(i) }}{% endfor %}";
        Assert.Equal("[0][1][2]", Render(src));
    }

    [Fact]
    public void Macro_CanAccessOuterContextVariable()
    {
        const string src = "{% macro show() %}{{ val }}{% endmacro %}{{ show() }}";
        var ctx = new Dictionary<string, object?> { ["val"] = "42" };
        Assert.Equal("42", Render(src, ctx));
    }

    // ── bos_token seeding ─────────────────────────────────────────────────────────
    // Chat templates (Gemma, Llama, …) open with `{{- bos_token -}}`. The model's BOS string is
    // injected by GgufTokenizer via JinjaChatTemplate.BosToken; without it the prompt ships with
    // no BOS token and Gemma degenerates. These lock the seeding semantics.

    [Fact]
    public void BosToken_WhenSet_RendersIntoTemplate()
    {
        var tmpl = new JinjaChatTemplate("{{- bos_token -}}<|turn>user\n") { BosToken = "<bos>" };
        Assert.Equal("<bos><|turn>user\n", tmpl.Render(new Dictionary<string, object?>()));
    }

    [Fact]
    public void BosToken_WhenNull_RendersEmpty()
    {
        // Default (no BosToken, e.g. add_bos_token=false models like Qwen) — byte-identical to before.
        var tmpl = new JinjaChatTemplate("{{- bos_token -}}<|turn>user\n");
        Assert.Equal("<|turn>user\n", tmpl.Render(new Dictionary<string, object?>()));
    }

    [Fact]
    public void BosToken_ExplicitContextValue_Wins()
    {
        var tmpl = new JinjaChatTemplate("{{- bos_token -}}x") { BosToken = "<bos>" };
        var ctx = new Dictionary<string, object?> { ["bos_token"] = "<s>" };
        Assert.Equal("<s>x", tmpl.Render(ctx));
    }

    [Fact]
    public void BosToken_WhenTemplateIgnoresIt_OutputUnchanged()
    {
        // Seeding a BOS string must not alter templates that never reference bos_token.
        var tmpl = new JinjaChatTemplate("<|turn>user\nhi") { BosToken = "<bos>" };
        Assert.Equal("<|turn>user\nhi", tmpl.Render(new Dictionary<string, object?>()));
    }

    // ── loop.previtem / loop.nextitem tests ──────────────────────────────────────

    [Fact]
    public void Loop_Previtem_IsNullOnFirstIteration()
    {
        const string src = "{% for x in items %}{% if loop.previtem is defined %}{{ loop.previtem }},{% endif %}{% endfor %}";
        var ctx = new Dictionary<string, object?> { ["items"] = new List<object?> { "a", "b", "c" } };
        Assert.Equal("a,b,", Render(src, ctx));
    }

    [Fact]
    public void Loop_Nextitem_IsNullOnLastIteration()
    {
        const string src = "{% for x in items %}{% if loop.nextitem is defined %}{{ loop.nextitem }},{% endif %}{% endfor %}";
        var ctx = new Dictionary<string, object?> { ["items"] = new List<object?> { "a", "b", "c" } };
        Assert.Equal("b,c,", Render(src, ctx));
    }

    // ── |items filter + tuple unpacking tests ────────────────────────────────────

    [Fact]
    public void Filter_Items_IteratesKeyValuePairs()
    {
        const string src = "{% for k, v in d|items %}{{ k }}={{ v }};{% endfor %}";
        var ctx = new Dictionary<string, object?> { ["d"] = new Dictionary<string, object?> { ["x"] = "1", ["y"] = "2" } };
        var result = Render(src, ctx);
        // Dict order is insertion order in .NET; both pairs must appear
        Assert.Contains("x=1;", result);
        Assert.Contains("y=2;", result);
    }

    [Fact]
    public void TupleUnpacking_ForLoop_AssignsBothVars()
    {
        const string src = "{% for a, b in pairs %}{{ a }}:{{ b }};{% endfor %}";
        var ctx = new Dictionary<string, object?>
        {
            ["pairs"] = new List<object?>
            {
                new List<object?> { "foo", "bar" },
                new List<object?> { "baz", "qux" },
            }
        };
        Assert.Equal("foo:bar;baz:qux;", Render(src, ctx));
    }

    // ── Filters/tests added for the Gemma 4 tool template ─────────────────────

    [Fact]
    public void Dictsort_YieldsKeyValuePairsSortedByKey()
    {
        // Insertion order is b, a — dictsort must emit a then b.
        const string src = "{% for k, v in d | dictsort %}{{ k }}={{ v }};{% endfor %}";
        var ctx = new Dictionary<string, object?>
        {
            ["d"] = new Dictionary<string, object?> { ["b"] = 2L, ["a"] = 1L },
        };
        Assert.Equal("a=1;b=2;", Render(src, ctx));
    }

    [Fact]
    public void DefaultFilter_ReplacesUndefined()
    {
        var ctx = new Dictionary<string, object?> { ["x"] = null };
        Assert.Equal("fallback", Render("{{ x | default('fallback') }}", ctx));
    }

    [Fact]
    public void DefaultFilter_KeepsDefinedValue()
    {
        var ctx = new Dictionary<string, object?> { ["x"] = "real" };
        Assert.Equal("real", Render("{{ x | default('fallback') }}", ctx));
    }

    [Fact]
    public void DefaultFilter_EmptyListFallbackInForLoop()
    {
        // `for item in missing | default([])` must iterate zero times, not crash.
        Assert.Equal("", Render("{% for i in missing | default([]) %}{{ i }}{% endfor %}"));
    }

    [Fact]
    public void MapFilter_AppliesNamedFilterToEachElement()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["xs"] = new List<object?> { "a", "b" },
        };
        Assert.Equal("A,B,", Render("{% for x in xs | map('upper') | list %}{{ x }},{% endfor %}", ctx));
    }

    [Fact]
    public void IsBoolean_Test()
    {
        var ctx = new Dictionary<string, object?> { ["t"] = true, ["s"] = "x" };
        Assert.Equal("yes", Render("{% if t is boolean %}yes{% else %}no{% endif %}", ctx));
        Assert.Equal("no", Render("{% if s is boolean %}yes{% else %}no{% endif %}", ctx));
    }

    [Fact]
    public void IsSequence_Test()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["lst"] = new List<object?> { 1L },
            ["str"] = "x",
        };
        Assert.Equal("yes", Render("{% if lst is sequence %}yes{% else %}no{% endif %}", ctx));
        // A bare string is not classified as a sequence here (template guards `is string` first).
        Assert.Equal("no", Render("{% if str is sequence %}yes{% else %}no{% endif %}", ctx));
    }

    [Fact]
    public void JoinFilter_ConcatenatesWithSeparator()
    {
        var ctx = new Dictionary<string, object?> { ["xs"] = new List<object?> { "a", "b", "c" } };
        Assert.Equal("a-b-c", Render("{{ xs | join('-') }}", ctx));
    }

    [Fact]
    public void DictGet_ReturnsValueOrFallback()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["d"] = new Dictionary<string, object?> { ["present"] = "P" },
        };
        Assert.Equal("P", Render("{{ d.get('present', 'fallback') }}", ctx));
        Assert.Equal("fallback", Render("{{ d.get('missing', 'fallback') }}", ctx));
        // No fallback arg → empty string for a missing key.
        Assert.Equal("", Render("{{ d.get('missing') }}", ctx));
    }

    [Fact]
    public void DictKeysAndValues_Iterate()
    {
        var ctx = new Dictionary<string, object?>
        {
            ["d"] = new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 2L },
        };
        Assert.Equal("a,b,", Render("{% for k in d.keys() %}{{ k }},{% endfor %}", ctx));
        Assert.Equal("1,2,", Render("{% for v in d.values() %}{{ v }},{% endfor %}", ctx));
    }

    // ── Index/slice on naturally-typed C# lists (issue #131) ─────────────────────
    // GetIndex / GetSlice used to match only List<object?>. A C# caller naturally builds
    // a messages list as List<Dictionary<string,object?>>, which (generic invariance) is
    // NOT a List<object?>, so messages[0] returned null and templates that branch on
    // messages[0].role == 'system' silently dropped the system block.

    [Fact]
    public void Index_OnListOfTypedDicts_ResolvesElementAndAttr()
    {
        var ctx = new Dictionary<string, object?>
        {
            // The "obvious" C# shape — typed list of typed dicts, not List<object?>.
            ["messages"] = new List<Dictionary<string, object?>>
            {
                new() { ["role"] = "system", ["content"] = "You are Ayu." },
                new() { ["role"] = "user",   ["content"] = "hi" },
            },
        };
        Assert.Equal("system", Render("{{ messages[0].role }}", ctx));
        Assert.Equal("hi", Render("{{ messages[1].content }}", ctx));
        // Negative index wraps from the end (the GetIndex path under test).
        Assert.Equal("user", Render("{{ messages[-1].role }}", ctx));
    }

    [Fact]
    public void SystemBlockBranch_OnListOfTypedDicts_IsNotDropped()
    {
        // The exact failure shape from issue #131: a Qwen3-style template guarding the
        // system block on messages[0].role == 'system'.
        const string src =
            "{% if messages[0].role == 'system' %}<sys>{{ messages[0].content }}</sys>{% endif %}" +
            "{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}";
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<Dictionary<string, object?>>
            {
                new() { ["role"] = "system", ["content"] = "S" },
                new() { ["role"] = "user",   ["content"] = "U" },
            },
        };
        Assert.Equal("<sys>S</sys><system>S</system><user>U</user>", Render(src, ctx));
    }

    [Fact]
    public void Slice_OnListOfTypedDicts_DropsFirstElement()
    {
        // messages[1:] is the common "skip the system message" idiom — exercises GetSlice.
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<Dictionary<string, object?>>
            {
                new() { ["role"] = "system", ["content"] = "S" },
                new() { ["role"] = "user",   ["content"] = "U" },
                new() { ["role"] = "assistant", ["content"] = "A" },
            },
        };
        Assert.Equal("U,A,", Render("{% for m in messages[1:] %}{{ m.content }},{% endfor %}", ctx));
    }

    [Fact]
    public void Slice_NegativeStop_OnListOfTypedDicts_DropsLastElement()
    {
        // messages[:-1] — negative-stop arithmetic is the error-prone part of GetSlice.
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<Dictionary<string, object?>>
            {
                new() { ["content"] = "A" },
                new() { ["content"] = "B" },
                new() { ["content"] = "C" },
            },
        };
        Assert.Equal("A,B,", Render("{% for m in messages[:-1] %}{{ m.content }},{% endfor %}", ctx));
    }

    [Fact]
    public void Slice_StepZero_Throws() =>
        // A zero step must throw rather than spin forever in GetSlice (matches range(…, 0)).
        Assert.Throws<InvalidOperationException>(() =>
        {
            var ctx = new Dictionary<string, object?> { ["xs"] = new List<object?> { 1L, 2L, 3L } };
            Render("{% for x in xs[::0] %}{{ x }}{% endfor %}", ctx);
        });

    [Fact]
    public void Index_OnArrayOfTypedDicts_Resolves()
    {
        // The fix matches System.Collections.IList, which covers T[] too — pin the array claim.
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new[]
            {
                new Dictionary<string, object?> { ["role"] = "system", ["content"] = "S" },
                new Dictionary<string, object?> { ["role"] = "user",   ["content"] = "U" },
            },
        };
        Assert.Equal("system", Render("{{ messages[0].role }}", ctx));
        Assert.Equal("U", Render("{% for m in messages[1:] %}{{ m.content }}{% endfor %}", ctx));
    }

    [Fact]
    public void Index_OutOfRange_OnListOfTypedDicts_ReturnsNullNotThrow()
    {
        // The i >= 0 && i < Count guard must hold for the broadened IList match too.
        var ctx = new Dictionary<string, object?>
        {
            ["messages"] = new List<Dictionary<string, object?>>
            {
                new() { ["content"] = "only" },
            },
        };
        Assert.Equal("", Render("{{ messages[5].content }}", ctx));
    }

    // ── Delimiters inside string literals ────────────────────────────────────
    // The lexer located the closing delimiter with a plain IndexOf, so a `}}` or `%}`
    // sitting inside a quoted literal ended the tag: the expression was truncated and
    // the tail of the literal leaked into the output as raw text. Mistral's template
    // closes every [AVAILABLE_TOOLS] entry with `{{- "}}" }}`, so every tool definition
    // shipped a stray `"` where its braces belonged — invalid JSON, no exception, and
    // the only visible symptom was the model quietly no longer calling tools.

    [Theory]
    [InlineData("""{{ "}}" }}""", "}}")]
    [InlineData("""{{ "a}}b" }}""", "a}}b")]
    [InlineData("""{{- "}}" }}""", "}}")]
    [InlineData("""{{ '}}' }}""", "}}")]
    [InlineData("""{{ "%}" }}""", "%}")]
    [InlineData("""{{ "}" }}""", "}")]        // single brace was never affected
    [InlineData("""{{ "{{" }}""", "{{")]      // opening delimiter was never affected
    public void ClosingDelimiter_InsideStringLiteral_DoesNotEndExpression(string template, string expected) =>
        Assert.Equal(expected, Render(template));

    [Fact]
    public void BlockDelimiter_InsideStringLiteral_DoesNotEndBlock() =>
        Assert.Equal("A", Render("""{% if x == "%}" %}A{% endif %}""",
                                 new Dictionary<string, object?> { ["x"] = "%}" }));

    [Fact]
    public void EscapedQuote_InsideLiteralContainingDelimiter_IsSkipped() =>
        Assert.Equal("""a"}}b""", Render(""" {{ "a\"}}b" }}""".Trim()));

    [Fact]
    public void UnterminatedLiteral_StillReportsUnclosedTag() =>
        Assert.Throws<FormatException>(() => Render("""{{ "abc }}"""));

    [Fact]
    public void Comment_IsNotQuoteAware()
    {
        // Jinja2's comment state has no string rules: {# … #} ends at the first #},
        // and an apostrophe in prose must stay inert rather than swallowing the rest.
        Assert.Equal("", Render("{# don't quote-scan this #}"));
        Assert.Equal("x", Render("{# a \" b #}x"));
    }

    [Fact]
    public void MistralToolBlock_RendersParseableJson()
    {
        // End-to-end over the construct from Mistral's own chat template. Descriptions are
        // deliberately quote-free: the template interpolates string values with no escaping
        // (`'"' + key + '": "' + val + '"'`), which breaks the block independently of the
        // lexer — that is the template's flaw, not one this fix can address.
        const string template = """
            {{- "[AVAILABLE_TOOLS] [" }}
            {%- for tool in tools %}
                {%- set tool = tool.function %}
                {{- '{"type": "function", "function": {' }}
                {%- for key, val in tool.items() %}
                    {%- if val is string %}
                        {{- '"' + key + '": "' + val + '"' }}
                    {%- else %}
                        {{- '"' + key + '": ' + val|tojson }}
                    {%- endif %}
                    {%- if not loop.last %}
                        {{- ", " }}
                    {%- endif %}
                {%- endfor %}
                {{- "}}" }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- else %}
                    {{- "]" }}
                {%- endif %}
            {%- endfor %}
            {{- "[/AVAILABLE_TOOLS]" }}
            """;

        var ctx = new Dictionary<string, object?>
        {
            ["tools"] = new List<Dictionary<string, object?>>
            {
                new()
                {
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"] = "get_weather",
                        ["description"] = "Get the current weather in a location",
                        ["parameters"] = new Dictionary<string, object?> { ["type"] = "object" },
                    },
                },
                new()
                {
                    ["function"] = new Dictionary<string, object?>
                    {
                        ["name"] = "list_files",
                        ["description"] = "List files under a path",
                        ["parameters"] = new Dictionary<string, object?> { ["type"] = "object" },
                    },
                },
            },
        };

        string rendered = Render(template, ctx);

        const string open = "[AVAILABLE_TOOLS] ";
        const string close = "[/AVAILABLE_TOOLS]";
        int start = rendered.IndexOf(open, StringComparison.Ordinal) + open.Length;
        int end = rendered.IndexOf(close, StringComparison.Ordinal);
        Assert.True(end > start, $"tool block markers missing in: {rendered}");

        string json = rendered[start..end];
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.Equal(2, doc.RootElement.GetArrayLength());
        Assert.Equal("get_weather",
            doc.RootElement[0].GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("list_files",
            doc.RootElement[1].GetProperty("function").GetProperty("name").GetString());
    }
}
