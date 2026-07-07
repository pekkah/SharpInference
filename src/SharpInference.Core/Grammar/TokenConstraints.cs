namespace SharpInference.Core.Grammar;

/// <summary>
/// Null-safe combinator for stacking independent <see cref="ITokenConstraint"/>s (issue #423) --
/// e.g. a caller-supplied whole-turn output constraint alongside a tool-argument constraint that
/// only becomes active once a tool call opens mid-turn. Prefer this over manually null-checking and
/// constructing <see cref="AndTokenConstraint"/> at call sites.
/// </summary>
public static class TokenConstraints
{
    /// <summary>
    /// Folds out nulls from <paramref name="constraints"/> and returns <c>null</c> (none left), the
    /// sole survivor, or an <see cref="AndTokenConstraint"/> AND-composing every survivor. Also the
    /// 2-arg case (e.g. <c>Combine(a, b)</c>) -- a params method accepts any fixed argument count,
    /// so there's no need for a separate 2-arg overload.
    /// </summary>
    public static ITokenConstraint? Combine(params ITokenConstraint?[] constraints)
    {
        List<ITokenConstraint>? survivors = null;
        foreach (var c in constraints)
        {
            if (c is null) continue;
            (survivors ??= []).Add(c);
        }
        return survivors switch
        {
            null => null,
            { Count: 1 } => survivors[0],
            _ => new AndTokenConstraint(survivors),
        };
    }
}
