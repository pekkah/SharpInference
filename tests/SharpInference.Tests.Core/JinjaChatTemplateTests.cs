using SharpInference.Core;

namespace SharpInference.Tests.Core;

public sealed class JinjaChatTemplateTests
{
    private static string Render(string source, IReadOnlyDictionary<string, object?>? ctx = null) =>
        new JinjaChatTemplate(source).Render(ctx ?? new Dictionary<string, object?>());

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
}
