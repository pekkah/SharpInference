namespace SharpInference.Core.Grammar;

/// <summary>
/// AND-composes a list of independent <see cref="ITokenConstraint"/>s, masking to the intersection
/// of what each currently-constraining inner allows (issue #423) -- e.g. a caller-supplied whole-turn
/// output grammar stacked with a tool-argument constraint once a tool call opens mid-turn.
///
/// <para>
/// This is the opposite of <see cref="CompositeToolArgumentConstraint"/>, which OR-composes
/// mutually-exclusive <em>alternative</em> sub-constraints (at most one is ever active at a time,
/// e.g. a JSON-vs-XML tool-call wire format choice). <see cref="AndTokenConstraint"/> is for
/// genuinely independent concerns that can be simultaneously active and must both be satisfied.
/// </para>
///
/// <para>
/// <see cref="Accept"/> and <see cref="Reset"/> always forward to every inner unconditionally --
/// each needs to see every token to track its own state regardless of which inner (if any) is
/// currently constraining. <see cref="Filter"/> only builds an intersection while 2+ inners are
/// simultaneously constraining; with 0 or 1 constraining it behaves exactly like the
/// zero-allocation single-constraint path every <see cref="ITokenConstraint"/> promises.
/// </para>
/// </summary>
public sealed class AndTokenConstraint : ITokenConstraint
{
    private readonly ITokenConstraint[] _inner;
    private float[]? _masked;

    public AndTokenConstraint(IReadOnlyList<ITokenConstraint> inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        if (inner.Count < 2) throw new ArgumentException("at least two inner constraints are required", nameof(inner));
        _inner = [.. inner];
    }

    public bool IsConstraining
    {
        get
        {
            foreach (var c in _inner) if (c.IsConstraining) return true;
            return false;
        }
    }

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        // Count how many inners are currently constraining without allocating -- 0/1 take the
        // zero-copy fast path every other constraint promises.
        ITokenConstraint? sole = null;
        int constrainingCount = 0;
        foreach (var c in _inner)
        {
            if (!c.IsConstraining) continue;
            sole = c;
            if (++constrainingCount > 1) break;
        }
        if (constrainingCount == 0) return logits;
        if (constrainingCount == 1) return sole!.Filter(logits);

        // 2+ constraining simultaneously: intersect into our own scratch buffer.
        var masked = _masked ??= new float[logits.Length];
        if (masked.Length != logits.Length) return logits;   // vocab mismatch -- never wedge
        logits.CopyTo(masked);

        foreach (var c in _inner)
        {
            if (!c.IsConstraining) continue;
            var view = c.Filter(logits);
            // An inner that hit its own dead state returns the SAME span as `logits` (every
            // ITokenConstraint's documented convention) -- skip it rather than contribute a no-op
            // pass that would otherwise be indistinguishable from "everything is forbidden". A
            // length mismatch (a misbehaving caller-supplied constraint, e.g. via
            // SharpInferenceServerOptions.OutputConstraintFactory) is skipped the same way rather
            // than indexed out of bounds -- never crash on a third-party constraint bug.
            if (view == logits || view.Length != masked.Length) continue;
            for (int i = 0; i < masked.Length; i++)
                if (float.IsNegativeInfinity(view[i])) masked[i] = float.NegativeInfinity;
        }

        bool anyLegal = false;
        for (int i = 0; i < masked.Length; i++)
            if (!float.IsNegativeInfinity(masked[i])) { anyLegal = true; break; }

        // Dead intersection (no legal token survives the AND): never wedge -- fall back unmasked,
        // matching every other ITokenConstraint's dead-state contract.
        return anyLegal ? masked : logits;
    }

    public void Accept(int token)
    {
        foreach (var c in _inner) c.Accept(token);
    }

    public void Reset()
    {
        foreach (var c in _inner) c.Reset();
    }
}
