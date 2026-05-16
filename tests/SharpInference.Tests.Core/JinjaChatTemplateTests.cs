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
}
